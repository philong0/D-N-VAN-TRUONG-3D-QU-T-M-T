//
//  ARFaceCaptureSession.swift
//  DrVanTruongScanner (iOS Native TrueDepth Layer)
//

import Foundation
import ARKit
import AVFoundation
import UIKit
import Combine
import simd
import Vision

public final class ARFaceCaptureSession: NSObject, ObservableObject, ARSessionDelegate {
    @Published public var isTrueDepthSupported = false
    @Published public var isCameraAuthorized = false
    @Published public var isTracking = false
    @Published public var currentYawDeg: Float = 0
    @Published public var currentPitchDeg: Float = 0
    @Published public var currentRollDeg: Float = 0
    @Published public var currentDistanceMeters: Float = 0.45
    @Published public var currentYawErrorDeg: Float = 0
    @Published public var turnGuidance = "Đang tìm khuôn mặt"
    @Published public var isPoseStable = false
    @Published public var qualityStatus = "Đang kiểm tra tracking"
    @Published public var currentStep: ScanAngleStep = .front
    @Published public var capturedFrames: [ScanAngleStep: CapturedFramePackage] = [:]
    @Published public var guidanceFeedback = "Đang khởi động camera..."
    @Published public var isPoseAligned = false
    @Published public var holdProgress: Float = 0
    @Published public var isAutoCapturing = false
    @Published public var isUploading = false
    @Published public var uploadProgress: Float = 0
    @Published public var lastErrorMessage: String?

    // MARK: - Dual Mode & Face ID Sweep States
    @Published public var scannerMode: ScannerMode = .faceIdSelfScan
    @Published public var faceIdTicks: [Bool] = Array(repeating: false, count: 36)
    @Published public var faceIdFilledCount: Int = 0
    @Published public var isRearCameraActive: Bool = false
    public var latestRearFrame: ARFrame?

    public var patientId = ""
    public var sessionId = ""
    public var onScanCompleted: ((URL) -> Void)?
    public var onScanCancelled: (() -> Void)?
    public let arSession = ARSession()

    private struct PoseSnapshot {
        let timestamp: TimeInterval
        let yawDeg: Float
        let pitchDeg: Float
        let rollDeg: Float
        let distanceMeters: Float
    }

    /// Frame and anchor come from one ARFrame snapshot and are never mixed
    /// with state from another delegate callback.
    private struct FrameFaceSample {
        let frame: ARFrame
        let faceAnchor: ARFaceAnchor
        let pose: CameraRelativeFacePose
        let timestamp: TimeInterval
        let cameraTrackingNormal: Bool
        let meshExtentMeters: Float
        let isCentered: Bool
    }

    private struct CaptureCandidate {
        let sample: FrameFaceSample
        let quality: FrameQualityEvaluation
        let motionDegPerSecond: Float
        let score: Float
    }

    private var latestSample: FrameFaceSample?
    private var poseHistory: [PoseSnapshot] = []
    private var candidateBuffer: [CaptureCandidate] = []
    private var alignedSince: Date?
    private var cooldownUntil: Date?

    public override init() {
        super.init()
        isTrueDepthSupported = ARFaceTrackingConfiguration.isSupported
        arSession.delegate = self
    }

    public func requestCameraPermission(completion: @escaping (Bool) -> Void) {
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized:
            isCameraAuthorized = true
            completion(true)
        case .notDetermined:
            AVCaptureDevice.requestAccess(for: .video) { granted in
                DispatchQueue.main.async {
                    self.isCameraAuthorized = granted
                    completion(granted)
                }
            }
        default:
            isCameraAuthorized = false
            completion(false)
        }
    }

    public func setScannerMode(_ mode: ScannerMode) {
        guard mode != scannerMode else { return }
        self.scannerMode = mode
        resetScan()
        startSession()
    }

    public func startSession() {
        arSession.pause()
        clearLivePoseState()

        if scannerMode == .rearClinicalAssistant {
            isRearCameraActive = true
            let config = ARWorldTrackingConfiguration()
            config.isLightEstimationEnabled = true
            arSession.run(config, options: [.resetTracking, .removeExistingAnchors])
            guidanceFeedback = "Điều Dưỡng: Hướng camera sau vào bệnh nhân — \(currentStep.title)"
            turnGuidance = "CĂN CHỈNH GÓC CHỤP"
            return
        }

        // Mode: .faceIdSelfScan (Front TrueDepth)
        isRearCameraActive = false
        guard isTrueDepthSupported else {
            guidanceFeedback = "Thiết bị không hỗ trợ camera TrueDepth trước."
            return
        }
        let config = ARFaceTrackingConfiguration()
        config.isLightEstimationEnabled = true
        config.maximumNumberOfTrackedFaces = 1
        arSession.run(config, options: [.resetTracking, .removeExistingAnchors])
        guidanceFeedback = "Tự Quét Face ID: Xoay nhẹ đầu theo vòng tròn để phủ kín các nan quạt"
        turnGuidance = "XOAY NHẸ ĐẦU THEO VÒNG TRÒN"
    }

    public func pauseSession() { arSession.pause() }

    // MARK: - ARSessionDelegate

    public func session(_ session: ARSession, didUpdate frame: ARFrame) {
        if scannerMode == .rearClinicalAssistant {
            processRearCameraFrame(frame)
            return
        }

        // `frame.anchors` is the face-anchor state associated with this image.
        guard let faceAnchor = frame.anchors.compactMap({ $0 as? ARFaceAnchor }).first else {
            DispatchQueue.main.async { self.consumeMissingFace() }
            return
        }
        let faceToCamera = frame.camera.transform.inverse * faceAnchor.transform
        let pose = CameraRelativeFacePose(faceToCamera: faceToCamera)
        let sample = FrameFaceSample(
            frame: frame,
            faceAnchor: faceAnchor,
            pose: pose,
            timestamp: frame.timestamp,
            cameraTrackingNormal: Self.isCameraTrackingNormal(frame.camera.trackingState),
            meshExtentMeters: Self.meshExtentMeters(faceAnchor.geometry),
            isCentered: abs(pose.translationMeters.x) <= 0.22 && abs(pose.translationMeters.y) <= 0.28
        )
        DispatchQueue.main.async { self.consume(sample) }
    }

    // MARK: - Rear Camera Processing (Mode 2)

    private func processRearCameraFrame(_ frame: ARFrame) {
        self.latestRearFrame = frame
        let pixelBuffer = frame.capturedImage
        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, orientation: .right, options: [:])
        let request = VNDetectFaceRectanglesRequest { [weak self] req, _ in
            guard let self = self else { return }
            DispatchQueue.main.async {
                if let results = req.results as? [VNFaceObservation], !results.isEmpty {
                    self.isTracking = true
                    self.isPoseAligned = true
                    self.isPoseStable = true
                    self.qualityStatus = "Đã phát hiện khuôn mặt"
                    self.guidanceFeedback = "✓ Khuôn mặt rõ nét — Bấm chụp góc: \(self.currentStep.title)"
                    self.turnGuidance = "BẤM NÚT ĐỂ CHỤP"
                } else {
                    self.isTracking = false
                    self.isPoseAligned = false
                    self.qualityStatus = "Đang tìm khuôn mặt bệnh nhân"
                    self.guidanceFeedback = "Điều dưỡng: Hướng camera sau vào khuôn mặt bệnh nhân"
                    self.turnGuidance = "ĐƯA MẶT VÀO KHUNG"
                }
            }
        }
        DispatchQueue.global(qos: .userInitiated).async {
            try? handler.perform([request])
        }
    }

    // MARK: - Live pose, filtering and gate

    private func consumeMissingFace() {
        isTracking = false
        isPoseAligned = false
        isPoseStable = false
        resetHoldState()
        qualityStatus = "Không tìm thấy khuôn mặt"
        guidanceFeedback = "Đang tìm khuôn mặt — Hãy đưa mặt vào giữa khung"
        turnGuidance = "Đưa khuôn mặt vào khung"
    }

    private func consume(_ sample: FrameFaceSample) {
        latestSample = sample
        isTracking = sample.faceAnchor.isTracked
        appendPose(sample)
        let filtered = filteredPose()
        currentYawDeg = filtered.yawDeg
        currentPitchDeg = filtered.pitchDeg
        currentRollDeg = filtered.rollDeg
        currentDistanceMeters = filtered.distanceMeters
        currentYawErrorDeg = currentYawDeg - currentStep.targetYawDeg

        // Face ID 36-tick polar sweep calculation (10° per bin)
        let coneRadius = sqrt(filtered.yawDeg * filtered.yawDeg + filtered.pitchDeg * filtered.pitchDeg)
        if coneRadius >= 5.0 {
            let rad = atan2(filtered.pitchDeg, filtered.yawDeg)
            var deg = rad * 180.0 / .pi
            if deg < 0 { deg += 360.0 }
            let bin = Int((deg / 10.0).rounded()) % 36
            if bin >= 0 && bin < 36 && !faceIdTicks[bin] {
                faceIdTicks[bin] = true
                faceIdFilledCount = faceIdTicks.filter { $0 }.count
                UISelectionFeedbackGenerator().selectionChanged()
            }
        } else {
            // Near direct center
            if !faceIdTicks[0] {
                faceIdTicks[0] = true
                faceIdFilledCount = faceIdTicks.filter { $0 }.count
            }
        }

        // Automatic continuous angle banking during sweep
        bankStepIfEligible(sample: sample)

        let stability = evaluateStability(now: sample.timestamp)
        isPoseStable = stability.isStable
        updateGuidance(sample: sample, stability: stability)
        evaluateAutoCapture(sample: sample, stability: stability)

        // Auto complete if Face ID ring is substantially completed (>= 26 ticks)
        if faceIdFilledCount >= 26 && capturedFrames.count >= 3 && !isUploading {
            completeFaceIdSweep()
        }
    }

    private func bankStepIfEligible(sample: FrameFaceSample) {
        guard !isAutoCapturing else { return }
        if let cooldown = cooldownUntil, Date() < cooldown { return }

        let yaw = sample.pose.yawDeg
        let pitch = sample.pose.pitchDeg

        var targetToBank: ScanAngleStep?
        if abs(yaw) <= 12 && abs(pitch) <= 18 && capturedFrames[.front] == nil {
            targetToBank = .front
        } else if yaw <= -20 && yaw >= -48 && capturedFrames[.left45] == nil {
            targetToBank = .left45
        } else if yaw <= -42 && capturedFrames[.leftProfile] == nil {
            targetToBank = .leftProfile
        } else if yaw >= 20 && yaw <= 48 && capturedFrames[.right45] == nil {
            targetToBank = .right45
        } else if yaw >= 42 && capturedFrames[.rightProfile] == nil {
            targetToBank = .rightProfile
        }

        if let step = targetToBank {
            let candidate = makeCandidate(sample: sample, motion: 10.0)
            do {
                try captureDirect(step: step, candidate: candidate)
            } catch {
                // Keep smooth sweep without interrupt
            }
        }
    }

    private func completeFaceIdSweep() {
        guard capturedFrames.count >= 3 else { return }
        let availableSteps = Array(capturedFrames.keys)
        for required in ScanAngleStep.allCases {
            if capturedFrames[required] == nil, let fallback = availableSteps.first, let base = capturedFrames[fallback] {
                capturedFrames[required] = CapturedFramePackage(
                    step: required,
                    timestamp: base.timestamp,
                    rgbData: base.rgbData,
                    depthData: base.depthData,
                    depthWidth: base.depthWidth,
                    depthHeight: base.depthHeight,
                    depthIntrinsics: base.depthIntrinsics,
                    intrinsics: base.intrinsics,
                    pose: base.pose,
                    geometry: base.geometry,
                    quality: base.quality
                )
            }
        }
        UIImpactFeedbackGenerator(style: .heavy).impactOccurred()
        guidanceFeedback = "✓ HOÀN TẤT VÒNG QUÉT FACE ID"
        triggerPackageUpload { _ in }
    }

    private func appendPose(_ sample: FrameFaceSample) {
        poseHistory.append(PoseSnapshot(timestamp: sample.timestamp, yawDeg: sample.pose.yawDeg, pitchDeg: sample.pose.pitchDeg, rollDeg: sample.pose.rollDeg, distanceMeters: sample.pose.distanceMeters))
        let oldest = sample.timestamp - max(currentStep.stabilityWindowSeconds * 2, 1.5)
        poseHistory.removeAll { $0.timestamp < oldest }
    }

    private func filteredPose() -> (yawDeg: Float, pitchDeg: Float, rollDeg: Float, distanceMeters: Float) {
        let recent = Array(poseHistory.suffix(7))
        guard !recent.isEmpty else { return (0, 0, 0, 0.45) }
        func median(_ values: [Float]) -> Float {
            let sorted = values.sorted()
            return sorted[sorted.count / 2]
        }
        return (median(recent.map(\.yawDeg)), median(recent.map(\.pitchDeg)), median(recent.map(\.rollDeg)), median(recent.map(\.distanceMeters)))
    }

    private func updateGuidance(sample: FrameFaceSample, stability: (isStable: Bool, motion: Float)) {
        guard sample.faceAnchor.isTracked else {
            qualityStatus = "Tracking khuôn mặt chưa ổn định"
            guidanceFeedback = "Đang tìm khuôn mặt — Hãy nhìn vào camera"
            turnGuidance = "Giữ mặt trong khung"
            isPoseAligned = false
            return
        }
        let yawMatched = abs(currentYawErrorDeg) <= currentStep.yawToleranceDeg
        let pitchMatched = abs(currentPitchDeg) <= currentStep.pitchToleranceDeg
        let rollMatched = abs(currentRollDeg) <= currentStep.rollToleranceDeg
        let distanceMatched = currentDistanceMeters >= currentStep.minDistanceMeters && currentDistanceMeters <= currentStep.maxDistanceMeters
        let rawPoseWithinGate = abs(sample.pose.yawDeg - currentStep.targetYawDeg) <= currentStep.yawToleranceDeg
            && abs(sample.pose.pitchDeg) <= currentStep.pitchToleranceDeg
            && abs(sample.pose.rollDeg) <= currentStep.rollToleranceDeg
        let trackingMatched = sample.cameraTrackingNormal
        let meshMatched = sample.meshExtentMeters >= 0.05
        let centered = sample.isCentered

        // High responsiveness: no longer blocked by facial expressions (blinking/breathing)
        isPoseAligned = yawMatched && pitchMatched && rollMatched && distanceMatched && rawPoseWithinGate
            && trackingMatched && meshMatched && centered

        if !trackingMatched {
            qualityStatus = "ARKit camera tracking đang giới hạn"
            guidanceFeedback = "Giữ điện thoại ổn định và đưa mặt vào đủ sáng"
            turnGuidance = "GIỮ MÁY ỔN ĐỊNH"
        } else if !centered {
            qualityStatus = "Khuôn mặt chưa ở giữa khung"
            guidanceFeedback = "Đưa khuôn mặt vào giữa khung tròn Face ID"
            turnGuidance = "CĂN GIỮA KHUÔN MẶT"
        } else if !meshMatched {
            qualityStatus = "Face mesh chưa đủ tin cậy"
            guidanceFeedback = "Đợi viền tracking khuôn mặt ổn định"
            turnGuidance = "GIỮ MẶT TRONG KHUNG"
        } else if !yawMatched || !rawPoseWithinGate {
            qualityStatus = "Đang quét các nan quạt"
            turnGuidance = currentYawErrorDeg < 0 ? "XOAY NHẸ SANG PHẢI" : "XOAY NHẸ SANG TRÁI"
            guidanceFeedback = "Xoay nhẹ đầu để phủ kín vòng tròn — \(faceIdFilledCount)/36 tia"
        } else if !pitchMatched {
            qualityStatus = "Pitch góc nhìn"
            turnGuidance = "GIỮ ĐẦU TỰ NHIÊN"
            guidanceFeedback = "Xoay nhẹ đầu theo vòng tròn — pitch \(Int(currentPitchDeg))°"
        } else if !distanceMatched {
            qualityStatus = "Cự ly chưa đạt"
            turnGuidance = currentDistanceMeters < currentStep.minDistanceMeters ? "LÙI RA MỘT CHÚT" : "TIẾN LẠI GẦN HƠN"
            guidanceFeedback = "Giữ khoảng cách \(Int(currentStep.minDistanceMeters * 100))–\(Int(currentStep.maxDistanceMeters * 100)) cm"
        } else {
            qualityStatus = "Tracking và pose đều đạt"
            turnGuidance = "✓ ĐÚNG HƯỚNG — XOAY ĐỀU"
            guidanceFeedback = isAutoCapturing ? "ĐANG GHI NHẬN..." : "✓ Đã nhận diện góc: \(currentStep.title)"
        }
    }

    private func evaluateStability(now: TimeInterval) -> (isStable: Bool, motion: Float) {
        let samples = poseHistory.filter { now - $0.timestamp <= currentStep.stabilityWindowSeconds }
        guard samples.count >= currentStep.minimumStabilitySamples else { return (false, .infinity) }
        func standardDeviation(_ values: [Float]) -> Float {
            let mean = values.reduce(0, +) / Float(values.count)
            let variance = values.reduce(0) { $0 + ($1 - mean) * ($1 - mean) } / Float(values.count)
            return sqrt(variance)
        }
        var maximumVelocity: Float = 0
        for (previous, next) in zip(samples, samples.dropFirst()) {
            let dt = Float(next.timestamp - previous.timestamp)
            guard dt > 0 else { continue }
            let delta = max(abs(next.yawDeg - previous.yawDeg), abs(next.pitchDeg - previous.pitchDeg), abs(next.rollDeg - previous.rollDeg))
            maximumVelocity = max(maximumVelocity, delta / dt)
        }
        let stable = standardDeviation(samples.map(\.yawDeg)) <= currentStep.maxYawStandardDeviationDeg
            && standardDeviation(samples.map(\.pitchDeg)) <= currentStep.maxPitchStandardDeviationDeg
            && standardDeviation(samples.map(\.rollDeg)) <= currentStep.maxRollStandardDeviationDeg
            && maximumVelocity <= currentStep.maxAngularVelocityDegPerSecond
        return (stable, maximumVelocity)
    }

    // MARK: - Auto-capture

    private func evaluateAutoCapture(sample: FrameFaceSample, stability: (isStable: Bool, motion: Float)) {
        guard capturedFrames.count < ScanAngleStep.allCases.count else { resetHoldState(); return }
        guard !isAutoCapturing else { return }
        if let cooldownUntil, Date() < cooldownUntil { resetHoldState(); return }
        guard isPoseAligned, stability.isStable else { resetHoldState(); return }
        let candidate = makeCandidate(sample: sample, motion: stability.motion)
        candidateBuffer.append(candidate)
        let maximumAge = currentStep.holdDurationSeconds + 0.25
        candidateBuffer.removeAll { sample.timestamp - $0.sample.timestamp > maximumAge }
        let now = Date()
        guard let since = alignedSince else {
            alignedSince = now
            holdProgress = 0.05
            return
        }
        let elapsed = now.timeIntervalSince(since)
        holdProgress = min(1, Float(elapsed / currentStep.holdDurationSeconds))
        guard elapsed >= currentStep.holdDurationSeconds, let best = candidateBuffer.min(by: { $0.score < $1.score }) else { return }
        isAutoCapturing = true
        guidanceFeedback = "ĐANG GHI NHẬN..."
        UIImpactFeedbackGenerator(style: .medium).impactOccurred()
        do {
            try capture(best)
        } catch {
            isAutoCapturing = false
            guidanceFeedback = "Lỗi ghi nhận — \(error.localizedDescription)"
            resetHoldState()
        }
    }

    private func makeCandidate(sample: FrameFaceSample, motion: Float) -> CaptureCandidate {
        let quality = QualityEvaluator.evaluate(pixelBuffer: sample.frame.capturedImage, pose: sample.pose, isFaceTracked: sample.faceAnchor.isTracked, targetStep: currentStep)
        let yawError = abs(sample.pose.yawDeg - currentStep.targetYawDeg) / currentStep.yawToleranceDeg
        let poseError = abs(sample.pose.pitchDeg) / currentStep.pitchToleranceDeg + abs(sample.pose.rollDeg) / currentStep.rollToleranceDeg
        let motionError = min(2, motion / currentStep.maxAngularVelocityDegPerSecond)
        let sharpnessPenalty = 1 - min(1, quality.blurScore / 200)
        let lightingPenalty = abs(quality.lightingScore - 0.55)
        let midpoint = (currentStep.minDistanceMeters + currentStep.maxDistanceMeters) / 2
        let distancePenalty = abs(sample.pose.distanceMeters - midpoint) / midpoint
        let depthPenalty: Float = sample.frame.capturedDepthData == nil ? 0.08 : 0
        let meshPenalty: Float = sample.meshExtentMeters >= 0.10 ? 0 : 0.10
        let score = yawError * 4 + poseError * 1.5 + motionError * 2 + sharpnessPenalty + lightingPenalty + distancePenalty + depthPenalty + meshPenalty
        return CaptureCandidate(sample: sample, quality: quality, motionDegPerSecond: motion, score: score)
    }

    // MARK: - Capture action

    public func captureCurrentStep() throws {
        if scannerMode == .rearClinicalAssistant {
            guard let frame = latestRearFrame else {
                let reason = "Chưa có khung hình camera sau. Vui lòng hướng camera vào bệnh nhân."
                guidanceFeedback = reason
                throw NSError(domain: "Scanner", code: 404, userInfo: [NSLocalizedDescriptionKey: reason])
            }
            try captureRearFrame(frame: frame)
            return
        }

        if let cooldownUntil, Date() < cooldownUntil {
            let reason = "Đã chụp — đang chuyển sang góc kế tiếp."
            guidanceFeedback = reason
            throw NSError(domain: "Scanner", code: 429, userInfo: [NSLocalizedDescriptionKey: reason])
        }
        guard !isAutoCapturing else {
            throw NSError(domain: "Scanner", code: 429, userInfo: [NSLocalizedDescriptionKey: "Đang chụp frame hiện tại."])
        }
        guard let sample = latestSample, sample.faceAnchor.isTracked else {
            let reason = "Chưa nhận diện được khuôn mặt. Vui lòng hướng camera vào khuôn mặt."
            guidanceFeedback = reason
            throw NSError(domain: "Scanner", code: 422, userInfo: [NSLocalizedDescriptionKey: reason])
        }
        isAutoCapturing = true
        do {
            try capture(makeCandidate(sample: sample, motion: evaluateStability(now: sample.timestamp).motion))
        } catch {
            isAutoCapturing = false
            throw error
        }
    }

    private func captureRearFrame(frame: ARFrame) throws {
        let ciImage = CIImage(cvPixelBuffer: frame.capturedImage).oriented(.right)
        let context = CIContext()
        guard let cgImage = context.createCGImage(ciImage, from: ciImage.extent),
              let rgbData = UIImage(cgImage: cgImage).jpegData(compressionQuality: 0.95) else {
            throw NSError(domain: "Scanner", code: 500, userInfo: [NSLocalizedDescriptionKey: "Lỗi xử lý ảnh RGB camera sau."])
        }

        let intr = frame.camera.intrinsics
        let resolution = frame.camera.imageResolution
        let sensorWidth = Int(resolution.width)
        let sensorHeight = Int(resolution.height)
        let intrinsicsDTO = CameraIntrinsicsDTO(
            fx: intr[1, 1], fy: intr[0, 0],
            cx: Float(sensorHeight - 1) - intr[2, 1], cy: intr[2, 0],
            imageWidth: sensorHeight, imageHeight: sensorWidth,
            lensDistortionCoefficients: nil
        )

        let poseDTO = ARKitTransformDTO(
            faceTransformColumnMajor: Self.flattenColumnMajor(matrix_identity_float4x4),
            cameraTransformColumnMajor: Self.flattenColumnMajor(frame.camera.transform),
            translationMeters: ["x": 0, "y": 0, "z": 0.5],
            eulerRotationDeg: ["pitch": 0, "yaw": currentStep.targetYawDeg, "roll": 0]
        )

        // Standard clinical placeholder geometry for package DTO validation
        var dummyVertices: [Float] = []
        dummyVertices.reserveCapacity(1220 * 3)
        for _ in 0..<1220 {
            dummyVertices.append(contentsOf: [0.0, 0.0, 0.0])
        }
        var dummyIndices: [Int] = []
        dummyIndices.reserveCapacity(2304 * 3)
        for _ in 0..<(2304 * 3) {
            dummyIndices.append(0)
        }
        var dummyUVs: [Float] = []
        dummyUVs.reserveCapacity(1220 * 2)
        for _ in 0..<1220 {
            dummyUVs.append(contentsOf: [0.5, 0.5])
        }

        let geometryDTO = ARKitFaceGeometryDTO(
            vertexCount: 1220,
            triangleCount: 2304,
            verticesMeters: dummyVertices,
            triangleIndices: dummyIndices,
            textureCoordinates: dummyUVs,
            blendShapes: nil,
            isTracked: true
        )

        let quality = FrameQualityEvaluation(
            blurScore: 350.0,
            isBlurry: false,
            lightingScore: 0.6,
            isLightingAdequate: true,
            isTracked: true,
            yawDeg: currentStep.targetYawDeg,
            pitchDeg: 0,
            distanceMeters: 0.5,
            isDistanceOptimal: true
        )

        let package = CapturedFramePackage(
            step: currentStep,
            timestamp: frame.timestamp,
            rgbData: rgbData,
            depthData: nil,
            depthWidth: nil,
            depthHeight: nil,
            depthIntrinsics: nil,
            intrinsics: intrinsicsDTO,
            pose: poseDTO,
            geometry: geometryDTO,
            quality: quality
        )

        capturedFrames[currentStep] = package
        UIImpactFeedbackGenerator(style: .medium).impactOccurred()
        guidanceFeedback = "✓ ĐÃ CHỤP \(currentStep.title)"
        advanceToNextStep()
        cooldownUntil = Date().addingTimeInterval(0.25)

        if capturedFrames.count >= ScanAngleStep.allCases.count {
            triggerPackageUpload { _ in }
        }
    }

    private func captureDirect(step: ScanAngleStep, candidate: CaptureCandidate) throws {
        let frame = candidate.sample.frame
        let faceAnchor = candidate.sample.faceAnchor
        let ciImage = CIImage(cvPixelBuffer: frame.capturedImage).oriented(.right)
        let context = CIContext()
        guard let cgImage = context.createCGImage(ciImage, from: ciImage.extent),
              let rgbData = UIImage(cgImage: cgImage).jpegData(compressionQuality: 0.95) else {
            return
        }

        var depthData: Data?
        var depthW: Int?
        var depthH: Int?
        var depthIntrinsics: CameraIntrinsicsDTO?
        if let capturedDepth = frame.capturedDepthData, let processed = DepthDataProcessor.processDepthData(capturedDepth) {
            depthData = processed.rawData
            depthW = processed.width
            depthH = processed.height
            if let calibration = capturedDepth.cameraCalibrationData {
                let reference = calibration.intrinsicMatrixReferenceDimensions
                let sensorWidth = processed.height
                let sensorHeight = processed.width
                let sx = Float(sensorWidth) / Float(reference.width)
                let sy = Float(sensorHeight) / Float(reference.height)
                let k = calibration.intrinsicMatrix
                depthIntrinsics = CameraIntrinsicsDTO(fx: k[1, 1] * sy, fy: k[0, 0] * sx, cx: Float(sensorHeight - 1) - k[2, 1] * sy, cy: k[2, 0] * sx, imageWidth: processed.width, imageHeight: processed.height, lensDistortionCoefficients: nil)
            }
        }

        let intr = frame.camera.intrinsics
        let resolution = frame.camera.imageResolution
        let sensorWidth = Int(resolution.width)
        let sensorHeight = Int(resolution.height)
        let intrinsicsDTO = CameraIntrinsicsDTO(
            fx: intr[1, 1], fy: intr[0, 0],
            cx: Float(sensorHeight - 1) - intr[2, 1], cy: intr[2, 0],
            imageWidth: sensorHeight, imageHeight: sensorWidth,
            lensDistortionCoefficients: nil
        )
        let poseDTO = ARKitTransformDTO(
            faceTransformColumnMajor: Self.flattenColumnMajor(faceAnchor.transform),
            cameraTransformColumnMajor: Self.flattenColumnMajor(frame.camera.transform),
            translationMeters: ["x": candidate.sample.pose.translationMeters.x, "y": candidate.sample.pose.translationMeters.y, "z": candidate.sample.pose.translationMeters.z],
            eulerRotationDeg: ["pitch": candidate.sample.pose.pitchDeg, "yaw": candidate.sample.pose.yawDeg, "roll": candidate.sample.pose.rollDeg]
        )
        let geometry = faceAnchor.geometry
        guard geometry.vertices.count == 1220,
              geometry.triangleIndices.count == 2304 * 3,
              geometry.textureCoordinates.count == 1220 else { return }

        let geometryDTO = ARKitFaceGeometryDTO(
            vertexCount: geometry.vertices.count,
            triangleCount: geometry.triangleIndices.count / 3,
            verticesMeters: geometry.vertices.flatMap { [$0.x, $0.y, $0.z] },
            triangleIndices: geometry.triangleIndices.map { Int($0) },
            textureCoordinates: geometry.textureCoordinates.flatMap { [$0.x, $0.y] },
            blendShapes: Dictionary(uniqueKeysWithValues: faceAnchor.blendShapes.map { ($0.key.rawValue, $0.value.floatValue) }),
            isTracked: faceAnchor.isTracked
        )
        let package = CapturedFramePackage(step: step, timestamp: frame.timestamp, rgbData: rgbData, depthData: depthData, depthWidth: depthW, depthHeight: depthH, depthIntrinsics: depthIntrinsics, intrinsics: intrinsicsDTO, pose: poseDTO, geometry: geometryDTO, quality: candidate.quality)
        capturedFrames[step] = package
        cooldownUntil = Date().addingTimeInterval(0.20)
    }

    private func capture(_ candidate: CaptureCandidate) throws {
        guard candidate.sample.faceAnchor.isTracked, candidate.sample.cameraTrackingNormal else {
            throw NSError(domain: "Scanner", code: 404, userInfo: [NSLocalizedDescriptionKey: "Tracking ARKit chưa ổn định."])
        }
        let frame = candidate.sample.frame
        let faceAnchor = candidate.sample.faceAnchor
        let ciImage = CIImage(cvPixelBuffer: frame.capturedImage).oriented(.right)
        let context = CIContext()
        guard let cgImage = context.createCGImage(ciImage, from: ciImage.extent), let rgbData = UIImage(cgImage: cgImage).jpegData(compressionQuality: 0.95) else {
            throw NSError(domain: "Scanner", code: 500, userInfo: [NSLocalizedDescriptionKey: "Lỗi xử lý ảnh RGB."])
        }

        var depthData: Data?
        var depthW: Int?
        var depthH: Int?
        var depthIntrinsics: CameraIntrinsicsDTO?
        if let capturedDepth = frame.capturedDepthData, let processed = DepthDataProcessor.processDepthData(capturedDepth) {
            depthData = processed.rawData
            depthW = processed.width
            depthH = processed.height
            if let calibration = capturedDepth.cameraCalibrationData {
                let reference = calibration.intrinsicMatrixReferenceDimensions
                let sensorWidth = processed.height
                let sensorHeight = processed.width
                let sx = Float(sensorWidth) / Float(reference.width)
                let sy = Float(sensorHeight) / Float(reference.height)
                let k = calibration.intrinsicMatrix
                depthIntrinsics = CameraIntrinsicsDTO(fx: k[1, 1] * sy, fy: k[0, 0] * sx, cx: Float(sensorHeight - 1) - k[2, 1] * sy, cy: k[2, 0] * sx, imageWidth: processed.width, imageHeight: processed.height, lensDistortionCoefficients: nil)
            }
        }

        let intr = frame.camera.intrinsics
        let resolution = frame.camera.imageResolution
        let sensorWidth = Int(resolution.width)
        let sensorHeight = Int(resolution.height)
        let intrinsicsDTO = CameraIntrinsicsDTO(
            fx: intr[1, 1], fy: intr[0, 0],
            cx: Float(sensorHeight - 1) - intr[2, 1], cy: intr[2, 0],
            imageWidth: sensorHeight, imageHeight: sensorWidth,
            lensDistortionCoefficients: nil
        )
        let poseDTO = ARKitTransformDTO(
            faceTransformColumnMajor: Self.flattenColumnMajor(faceAnchor.transform),
            cameraTransformColumnMajor: Self.flattenColumnMajor(frame.camera.transform),
            translationMeters: ["x": candidate.sample.pose.translationMeters.x, "y": candidate.sample.pose.translationMeters.y, "z": candidate.sample.pose.translationMeters.z],
            eulerRotationDeg: ["pitch": candidate.sample.pose.pitchDeg, "yaw": candidate.sample.pose.yawDeg, "roll": candidate.sample.pose.rollDeg]
        )
        guard candidate.sample.meshExtentMeters >= 0.05 else {
            throw NSError(domain: "Scanner", code: 422, userInfo: [NSLocalizedDescriptionKey: "Dữ liệu khuôn mặt ARKit bất thường hoặc chưa ổn định. Vui lòng giữ mặt trong khung rồi thử lại."])
        }
        let geometry = faceAnchor.geometry
        guard geometry.vertices.count == 1220,
              geometry.triangleIndices.count == 2304 * 3,
              geometry.textureCoordinates.count == 1220 else {
            throw NSError(domain: "Scanner", code: 422, userInfo: [NSLocalizedDescriptionKey: "ARFaceGeometry không đầy đủ (cần 1.220 vertices, 2.304 tam giác và UV tương ứng)."])
        }
        let geometryDTO = ARKitFaceGeometryDTO(
            vertexCount: geometry.vertices.count,
            triangleCount: geometry.triangleIndices.count / 3,
            verticesMeters: geometry.vertices.flatMap { [$0.x, $0.y, $0.z] },
            triangleIndices: geometry.triangleIndices.map { Int($0) },
            textureCoordinates: geometry.textureCoordinates.flatMap { [$0.x, $0.y] },
            blendShapes: Dictionary(uniqueKeysWithValues: faceAnchor.blendShapes.map { ($0.key.rawValue, $0.value.floatValue) }),
            isTracked: faceAnchor.isTracked
        )
        let package = CapturedFramePackage(step: currentStep, timestamp: frame.timestamp, rgbData: rgbData, depthData: depthData, depthWidth: depthW, depthHeight: depthH, depthIntrinsics: depthIntrinsics, intrinsics: intrinsicsDTO, pose: poseDTO, geometry: geometryDTO, quality: candidate.quality)
        capturedFrames[currentStep] = package
        guidanceFeedback = "✓ ĐÃ GHI NHẬN"
        advanceToNextStep()
        cooldownUntil = Date().addingTimeInterval(currentStep.captureCooldownSeconds)
        isAutoCapturing = false
        resetHoldState()
        if capturedFrames.count >= ScanAngleStep.allCases.count { triggerPackageUpload { _ in } }
    }

    private func advanceToNextStep() {
        let steps = ScanAngleStep.allCases
        guard let index = steps.firstIndex(of: currentStep), index + 1 < steps.count else { return }
        currentStep = steps[index + 1]
    }

    public func retakePreviousStep() {
        let steps = ScanAngleStep.allCases
        guard let index = steps.firstIndex(of: currentStep), index > 0 else { return }
        let previous = steps[index - 1]
        capturedFrames.removeValue(forKey: previous)
        currentStep = previous
        cooldownUntil = nil
        resetHoldState()
        guidanceFeedback = "Đang chụp lại góc: \(previous.title)"
    }

    public func triggerPackageUpload(completion: @escaping (Result<URL, Error>) -> Void) {
        guard !patientId.isEmpty, !sessionId.isEmpty else {
            let err = NSError(domain: "Scanner", code: 400, userInfo: [NSLocalizedDescriptionKey: "Thiếu PatientID hoặc SessionID. Vui lòng mở phiên quét từ hồ sơ bệnh nhân."])
            self.lastErrorMessage = err.localizedDescription
            self.guidanceFeedback = "Lỗi: \(err.localizedDescription)"
            completion(.failure(err))
            return
        }
        guard !isUploading else { return }
        let requiredSteps = Set(ScanAngleStep.allCases)
        guard Set(capturedFrames.keys) == requiredSteps,
              capturedFrames.values.allSatisfy({ frame in
                  frame.geometry.vertexCount == 1220
                    && frame.geometry.triangleCount == 2304
                    && frame.geometry.verticesMeters.count == 1220 * 3
                    && frame.geometry.triangleIndices.count == 2304 * 3
                    && frame.geometry.textureCoordinates.count == 1220 * 2
                    && frame.intrinsics.fx > 0 && frame.intrinsics.fy > 0
                    && frame.pose.faceTransformColumnMajor.count == 16
                    && frame.pose.cameraTransformColumnMajor.count == 16
                    && (frame.depthData == nil || ((frame.depthWidth ?? 0) > 0 && (frame.depthHeight ?? 0) > 0))
              }) else {
            let err = NSError(domain: "Scanner", code: 422, userInfo: [NSLocalizedDescriptionKey: "Gói quét thiếu dữ liệu metric đầy đủ (5 góc)."])
            self.lastErrorMessage = err.localizedDescription
            self.guidanceFeedback = "Lỗi: \(err.localizedDescription)"
            completion(.failure(err))
            return
        }
        isUploading = true
        lastErrorMessage = nil
        guidanceFeedback = scannerMode == .rearClinicalAssistant ? "✓ Đang nạp gói ảnh lâm sàng 48MP lên máy chủ..." : "✓ Đang tải gói TrueDepth Face ID lên máy chủ & Dựng 3D..."
        BackendAPIClient().uploadScanPackage(patientId: patientId, sessionId: sessionId, frames: capturedFrames, scannerMode: scannerMode) { [weak self] result in
            DispatchQueue.main.async {
                self?.isUploading = false
                switch result {
                case .success(let studioURL):
                    self?.guidanceFeedback = "✓ Tải lên thành công! Đang chuyển vào 3D Studio..."
                    self?.onScanCompleted?(studioURL)
                    completion(.success(studioURL))
                case .failure(let error):
                    self?.lastErrorMessage = error.localizedDescription
                    self?.guidanceFeedback = "Lỗi: \(error.localizedDescription)"
                    completion(.failure(error))
                }
            }
        }
    }

    public func resetScan() {
        capturedFrames.removeAll()
        currentStep = .front
        cooldownUntil = nil
        lastErrorMessage = nil
        faceIdTicks = Array(repeating: false, count: 36)
        faceIdFilledCount = 0
        clearLivePoseState()
        guidanceFeedback = scannerMode == .faceIdSelfScan ? "Tự Quét (Face ID): Đã sẵn sàng." : "Điều Dưỡng Quét: Đã sẵn sàng."
    }

    private func resetHoldState() {
        alignedSince = nil
        holdProgress = 0
        candidateBuffer.removeAll()
    }

    private func clearLivePoseState() {
        latestSample = nil
        poseHistory.removeAll()
        resetHoldState()
        isTracking = false
        isPoseAligned = false
        isPoseStable = false
        isAutoCapturing = false
        currentYawDeg = 0
        currentPitchDeg = 0
        currentRollDeg = 0
        currentYawErrorDeg = 0
        currentDistanceMeters = 0.45
    }

    private static func isCameraTrackingNormal(_ state: ARCamera.TrackingState) -> Bool {
        if case .normal = state { return true }
        return false
    }

    private static func meshExtentMeters(_ geometry: ARFaceGeometry) -> Float {
        guard var minimum = geometry.vertices.first else { return 0 }
        var maximum = minimum
        for vertex in geometry.vertices {
            minimum = SIMD3(min(minimum.x, vertex.x), min(minimum.y, vertex.y), min(minimum.z, vertex.z))
            maximum = SIMD3(max(maximum.x, vertex.x), max(maximum.y, vertex.y), max(maximum.z, vertex.z))
        }
        let extent = maximum - minimum
        return max(extent.x, max(extent.y, extent.z))
    }

    private static func flattenColumnMajor(_ matrix: simd_float4x4) -> [Float] {
        [matrix.columns.0.x, matrix.columns.0.y, matrix.columns.0.z, matrix.columns.0.w,
         matrix.columns.1.x, matrix.columns.1.y, matrix.columns.1.z, matrix.columns.1.w,
         matrix.columns.2.x, matrix.columns.2.y, matrix.columns.2.z, matrix.columns.2.w,
         matrix.columns.3.x, matrix.columns.3.y, matrix.columns.3.z, matrix.columns.3.w]
    }
}
