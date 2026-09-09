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

    public func startSession() {
        arSession.pause()
        clearLivePoseState()
        guard isTrueDepthSupported else {
            guidanceFeedback = "Thiết bị không hỗ trợ camera TrueDepth trước."
            return
        }
        // This scanner is intentionally face-tracking only. ARWorldTracking
        // and the rear camera cannot be substituted for a TrueDepth scan.
        let config = ARFaceTrackingConfiguration()
        config.isLightEstimationEnabled = true
        config.maximumNumberOfTrackedFaces = 1
        arSession.run(config, options: [.resetTracking, .removeExistingAnchors])
        guidanceFeedback = "Camera trước TrueDepth: Nhìn thẳng vào màn hình"
    }

    public func pauseSession() { arSession.pause() }

    // MARK: - ARSessionDelegate

    public func session(_ session: ARSession, didUpdate frame: ARFrame) {
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
            isCentered: abs(pose.translationMeters.x) <= 0.16 && abs(pose.translationMeters.y) <= 0.20
        )
        DispatchQueue.main.async { self.consume(sample) }
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
        let stability = evaluateStability(now: sample.timestamp)
        isPoseStable = stability.isStable
        updateGuidance(sample: sample, stability: stability)
        evaluateAutoCapture(sample: sample, stability: stability)
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
        // A smoothed HUD may not make a raw out-of-band sample pass.
        let rawPoseWithinGate = abs(sample.pose.yawDeg - currentStep.targetYawDeg) <= currentStep.yawToleranceDeg
            && abs(sample.pose.pitchDeg) <= currentStep.pitchToleranceDeg
            && abs(sample.pose.rollDeg) <= currentStep.rollToleranceDeg
        let expressionMatched = isNeutralExpression(sample.faceAnchor)
        let trackingMatched = sample.cameraTrackingNormal
        let meshMatched = sample.meshExtentMeters >= 0.05
        let centered = sample.isCentered
        isPoseAligned = yawMatched && pitchMatched && rollMatched && distanceMatched && rawPoseWithinGate
            && expressionMatched && trackingMatched && meshMatched && centered

        if !trackingMatched {
            qualityStatus = "ARKit camera tracking đang giới hạn"
            guidanceFeedback = "Giữ điện thoại ổn định và đưa mặt vào đủ sáng"
            turnGuidance = "GIỮ MÁY ỔN ĐỊNH"
        } else if !centered {
            qualityStatus = "Khuôn mặt chưa ở giữa khung"
            guidanceFeedback = "Đưa khuôn mặt vào giữa khung hướng dẫn"
            turnGuidance = "CĂN GIỮA KHUÔN MẶT"
        } else if !meshMatched {
            qualityStatus = "Face mesh chưa đủ tin cậy"
            guidanceFeedback = "Đợi viền tracking khuôn mặt ổn định"
            turnGuidance = "GIỮ MẶT TRONG KHUNG"
        } else if !yawMatched || !rawPoseWithinGate {
            qualityStatus = "Đang căn góc"
            turnGuidance = currentYawErrorDeg < 0 ? "QUAY THÊM SANG PHẢI" : "QUAY THÊM SANG TRÁI"
            guidanceFeedback = "\(turnGuidance) — còn \(String(format: "%.1f", abs(currentYawErrorDeg)))°"
        } else if !pitchMatched {
            qualityStatus = "Pitch chưa đạt"
            turnGuidance = "GIỮ ĐẦU THẲNG"
            guidanceFeedback = "Không ngước/cúi — pitch \(Int(currentPitchDeg))°"
        } else if !rollMatched {
            qualityStatus = "Roll chưa đạt"
            turnGuidance = "GIỮ ĐẦU THẲNG"
            guidanceFeedback = "Không nghiêng đầu — roll \(Int(currentRollDeg))°"
        } else if !distanceMatched {
            qualityStatus = "Cự ly chưa đạt"
            turnGuidance = currentDistanceMeters < currentStep.minDistanceMeters ? "LÙI RA MỘT CHÚT" : "TIẾN LẠI GẦN HƠN"
            guidanceFeedback = "Giữ khoảng cách \(Int(currentStep.minDistanceMeters * 100))–\(Int(currentStep.maxDistanceMeters * 100)) cm"
        } else if !expressionMatched {
            qualityStatus = "Biểu cảm chưa trung tính"
            turnGuidance = "THẢ LỎNG KHUÔN MẶT"
            guidanceFeedback = "Thả lỏng môi, mắt và hàm rồi giữ yên"
        } else if !stability.isStable {
            qualityStatus = "Đang chờ đầu ổn định"
            turnGuidance = "GIỮ NGUYÊN"
            guidanceFeedback = "Đúng góc — giữ đầu ổn định"
        } else {
            qualityStatus = "Tracking, pose và ổn định đều đạt"
            turnGuidance = "ĐÚNG GÓC — GIỮ NGUYÊN"
            guidanceFeedback = isAutoCapturing ? "ĐANG CHỤP..." : "✓ ĐÚNG GÓC: Giữ yên..."
        }
    }

    private func isNeutralExpression(_ anchor: ARFaceAnchor) -> Bool {
        let keys: [ARFaceAnchor.BlendShapeLocation] = [.jawOpen, .mouthSmileLeft, .mouthSmileRight, .mouthFrownLeft, .mouthFrownRight, .eyeBlinkLeft, .eyeBlinkRight]
        return keys.allSatisfy { (anchor.blendShapes[$0]?.floatValue ?? 0) <= 0.18 }
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
        guidanceFeedback = "ĐANG CHỤP..."
        UIImpactFeedbackGenerator(style: .heavy).impactOccurred()
        do {
            try capture(best)
        } catch {
            isAutoCapturing = false
            guidanceFeedback = "Lỗi tự động chụp — \(error.localizedDescription)"
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

    private func capture(_ candidate: CaptureCandidate) throws {
        guard candidate.sample.faceAnchor.isTracked, candidate.sample.cameraTrackingNormal else {
            throw NSError(domain: "Scanner", code: 404, userInfo: [NSLocalizedDescriptionKey: "Tracking ARKit chưa ổn định."])
        }
        guard candidate.quality.isTracked, candidate.quality.isDistanceOptimal, candidate.quality.isLightingAdequate, !candidate.quality.isBlurry else {
            throw NSError(domain: "Scanner", code: 422, userInfo: [NSLocalizedDescriptionKey: "Frame chưa đạt chất lượng TrueDepth (tracking/ánh sáng/độ nét/khoảng cách)."])
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
        // JPEG pixels are rotated 90° CW from the sensor buffer. Persist K
        // in those portrait JPEG coordinates so texture projection agrees
        // with the saved RGB frame.
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
        guidanceFeedback = "✓ ĐÃ CHỤP"
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
            completion(.failure(NSError(domain: "Scanner", code: 400, userInfo: [NSLocalizedDescriptionKey: "Thiếu PatientID hoặc SessionID."])))
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
                    // A native TrueDepth baseline is a metric depth
                    // reconstruction.  Do not accept an ARFace-only package
                    // and silently turn it into a lower-fidelity path.
                    && frame.depthData != nil
                    && (frame.depthWidth ?? 0) > 0 && (frame.depthHeight ?? 0) > 0
                    && frame.depthIntrinsics != nil
                    && (frame.depthData?.count == (frame.depthWidth ?? 0) * (frame.depthHeight ?? 0) * MemoryLayout<Float32>.size)
                    && frame.pose.faceTransformColumnMajor.count == 16
                    && frame.pose.cameraTransformColumnMajor.count == 16
                    && !frame.rgbData.isEmpty
                    && frame.depthData != nil
                    && frame.depthWidth != nil && frame.depthHeight != nil
                    && frame.depthIntrinsics != nil
                    && (frame.depthData?.count == (frame.depthWidth ?? 0) * (frame.depthHeight ?? 0) * MemoryLayout<Float32>.size)
              }) else {
            completion(.failure(NSError(domain: "Scanner", code: 422, userInfo: [NSLocalizedDescriptionKey: "Gói quét thiếu dữ liệu ARKit metric đầy đủ; không tải lên hoặc hạ cấp sang ảnh 2D."])) )
            return
        }
        isUploading = true
        lastErrorMessage = nil
        guidanceFeedback = "✓ Đang tải gói TrueDepth / ARKit lên máy chủ & Dựng 3D..."
        BackendAPIClient().uploadScanPackage(patientId: patientId, sessionId: sessionId, frames: capturedFrames) { [weak self] result in
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
        clearLivePoseState()
        guidanceFeedback = "Đã sẵn sàng quét lại."
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

    private func captureBlockReason() -> String {
        if !isTracking { return "Chưa nhận diện được khuôn mặt ổn định." }
        if !isPoseAligned { return guidanceFeedback }
        if !isPoseStable { return "Đúng góc nhưng đầu còn chuyển động — hãy giữ yên." }
        return "Chưa có frame TrueDepth hợp lệ."
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
