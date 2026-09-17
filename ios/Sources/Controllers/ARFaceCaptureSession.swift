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
import CryptoKit

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
    @Published public var isPoseStable = true
    @Published public var qualityStatus = "Sẵn sàng"
    @Published public var currentStep: ScanAngleStep = .front
    @Published public var capturedFrames: [ScanAngleStep: CapturedFramePackage] = [:]
    @Published public var guidanceFeedback = "Đưa khuôn mặt vào vòng tròn"
    @Published public var isPoseAligned = false
    @Published public var holdProgress: Float = 0
    @Published public var isAutoCapturing = false
    @Published public var isUploading = false
    @Published public var uploadProgress: Float = 0
    @Published public var uploadStatusMessage: String = ""
    @Published public var lastErrorMessage: String?

    // MARK: - Dual Mode & Face ID Sweep States
    @Published public var scannerMode: ScannerMode = .faceIdSelfScan
    @Published public var faceIdTicks: [Bool] = Array(repeating: false, count: 36)
    @Published public var faceIdFilledCount: Int = 0
    @Published public var sweepFrames: [Int: CapturedFramePackage] = [:]
    @Published public var clinicalPhotos: [String: CapturedFramePackage] = [:]
    @Published public var isRearCameraActive: Bool = false
    @Published public var isFaceInFramingRect: Bool = false
    @Published public var isScanningActive: Bool = false
    @Published public var isSweepCompleted: Bool = false
    public var latestRearFrame: ARFrame?
    public var sweepStartTime: TimeInterval?

    public var patientId = ""
    public var sessionId = ""
    public var onScanCompleted: ((URL) -> Void)?
    public var onScanCancelled: (() -> Void)?
    public let arSession = ARSession()

    // Shared reusable CIContext for ultra-fast GPU rendering without main thread freeze
    private static let sharedCIContext = CIContext(options: [.useSoftwareRenderer: false])
    // Dedicated background queue for image processing & JPEG compression
    private let processingQueue = DispatchQueue(label: "com.drvantruong.scanner.processingQueue", qos: .userInitiated)
    private var isBankingInProgress = false
    private var lastBankedTimestamp: TimeInterval = 0
    private var candidateBank = ContinuousKeyframeBank()
    private var candidatePackages: [String: CapturedFramePackage] = [:]
    private var captureGeneration = UUID()
    private var lastPoseSample: (time: Double, yaw: Float, pitch: Float)?
    private var frozenFrames: [String: CapturedFramePackage]?
    private var frozenClinical: [String: CapturedFramePackage] = [:]
    private var activeApiClient: BackendAPIClient?
    private var uploadAttempt = ScanUploadAttempt()
    private var runningMode: ScannerMode?
    private let speech = AVSpeechSynthesizer()
    private var spokenPhase: ContinuousKeyframeBank.Phase?
    @Published public var voiceGuidanceEnabled = true
    @Published public var requiresNewScan = false

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
        guard runningMode != scannerMode, frozenFrames == nil else { return }
        runningMode = scannerMode
        arSession.pause()
        clearLivePoseState()

        if scannerMode == .rearClinicalAssistant {
            isRearCameraActive = true
            let config = ARWorldTrackingConfiguration()
            config.isLightEstimationEnabled = true
            arSession.run(config, options: [.resetTracking, .removeExistingAnchors])
            guidanceFeedback = "Điều Dưỡng: Hướng camera sau vào khuôn mặt bệnh nhân"
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
        guidanceFeedback = "Đưa khuôn mặt vào trong vòng tròn"
        turnGuidance = "XOAY NHẸ ĐẦU THEO VÒNG TRÒN"
    }

    public func pauseSession() {
        speech.stopSpeaking(at: .immediate)
        try? AVAudioSession.sharedInstance().setActive(false, options: .notifyOthersOnDeactivation)
        runningMode = nil
        uploadAttempt.invalidate()
        arSession.pause()
        isScanningActive = false
        captureGeneration = UUID()
        isBankingInProgress = false
    }

    public func sessionWasInterrupted(_ session: ARSession) {
        DispatchQueue.main.async {
            guard self.isScanningActive else { return }
            self.isTracking = false
            self.speech.stopSpeaking(at: .immediate)
            self.guidanceFeedback = "Camera đang tạm dừng. Dữ liệu đã quét được giữ lại."
        }
    }

    public func sessionInterruptionEnded(_ session: ARSession) {
        DispatchQueue.main.async {
            guard self.isScanningActive, self.frozenFrames == nil else { return }
            self.runningMode = nil
            self.startSession()
            self.spokenPhase = nil
            self.announceScanPhase()
        }
    }

    // MARK: - ARSessionDelegate

    public func session(_ session: ARSession, didUpdate frame: ARFrame) {
        if scannerMode == .rearClinicalAssistant {
            processRearCameraFrame(frame)
            return
        }

        guard let faceAnchor = frame.anchors.compactMap({ $0 as? ARFaceAnchor }).first else {
            DispatchQueue.main.async {
                self.isTracking = false
                self.isPoseAligned = false
                self.guidanceFeedback = "Đưa khuôn mặt vào trong vòng tròn"
            }
            return
        }

        let faceToCamera = frame.camera.transform.inverse * faceAnchor.transform
        let pose = CameraRelativeFacePose(faceToCamera: faceToCamera)

        DispatchQueue.main.async {
            self.consumeFaceIdFrame(frame: frame, faceAnchor: faceAnchor, pose: pose)
        }
    }

    // MARK: - Continuous native capture: all state transitions follow successful banking.
    private func consumeFaceIdFrame(frame: ARFrame, faceAnchor: ARFaceAnchor, pose: CameraRelativeFacePose) {
        isTracking = faceAnchor.isTracked
        currentYawDeg = pose.yawDeg; currentPitchDeg = pose.pitchDeg
        currentRollDeg = pose.rollDeg; currentDistanceMeters = pose.distanceMeters
        isFaceInFramingRect = faceAnchor.isTracked && abs(pose.yawDeg) <= 15 && abs(pose.pitchDeg) <= 25
            && pose.distanceMeters >= 0.25 && pose.distanceMeters <= 0.70
        let previous = lastPoseSample
        lastPoseSample = (frame.timestamp, pose.yawDeg, pose.pitchDeg)
        guard isScanningActive, frozenFrames == nil else { return }
        guidanceFeedback = candidateBank.guidance
        guard faceAnchor.isTracked else {
            guidanceFeedback = "Đang tìm lại khuôn mặt; giữ máy ổn định"
            return
        }
        guard !isBankingInProgress, frame.timestamp-lastBankedTimestamp >= 0.12 else { return }
        let dt = previous.map { frame.timestamp-$0.time } ?? 0
        let motion = dt > 0 ? Double(max(abs(pose.yawDeg-(previous?.yaw ?? pose.yawDeg)),
                                        abs(pose.pitchDeg-(previous?.pitch ?? pose.pitchDeg))))/dt : 0
        guard motion <= 90, abs(pose.pitchDeg) <= 25, abs(pose.rollDeg) <= 25,
              abs(pose.yawDeg) <= 65 else { return }
        let blink = max(faceAnchor.blendShapes[.eyeBlinkLeft]?.doubleValue ?? 0,
                        faceAnchor.blendShapes[.eyeBlinkRight]?.doubleValue ?? 0)
        guard blink < 0.6 else { return }
        isBankingInProgress = true
        lastBankedTimestamp = frame.timestamp
        let generation = captureGeneration
        processingQueue.async { [weak self] in
            guard let self = self else { return }
            // Everything is derived from this exact retained ARFrame/anchor,
            // never from arSession.currentFrame during asynchronous processing.
            let package = autoreleasepool {
                self.createPackage(from: frame, faceAnchor: faceAnchor, pose: pose, step: .front, viewTag: "candidate")
            }
            let hash = package.map { SHA256.hash(data: $0.rgbData).map { String(format: "%02x", $0) }.joined() }
            DispatchQueue.main.async {
                guard generation == self.captureGeneration else { return }
                self.isBankingInProgress = false
                guard self.isScanningActive, let package = package, let hash = hash else { return }
                let q = package.quality
                guard q.isTracked, !q.isBlurry, q.isLightingAdequate, q.isDistanceOptimal else {
                    self.guidanceFeedback = "Giữ máy ổn định, xoay chậm trong ánh sáng đều"
                    return
                }
                let score = log1p(Double(q.blurScore))/8 - abs(Double(q.lightingScore)-0.5)
                    - motion/180 - blink/2 - abs(Double(q.pitchDeg))/90 + (package.depthData == nil ? 0 : 0.1)
                let candidate = ContinuousSample(hash: hash, timestamp: package.timestamp,
                    yaw: Double(q.yawDeg), pitch: Double(q.pitchDeg), score: score)
                guard self.candidateBank.bank(candidate) else { return }
                self.candidatePackages[hash] = package
                let retained = Set(self.candidateBank.samples.map(\.hash))
                self.candidatePackages = self.candidatePackages.filter { retained.contains($0.key) }
                let progress = min(35, self.candidateBank.phase.rawValue*8 + min(3, self.candidateBank.samples.count/7))
                self.faceIdFilledCount = progress
                self.faceIdTicks = (0..<36).map { $0 < progress }
                self.guidanceFeedback = self.candidateBank.guidance
                self.announceScanPhase()
                if self.candidateBank.phase == .ready { self.completeFaceIdSweep() }
            }
        }
    }

    public func completeFaceIdSweep() {
        guard !isBankingInProgress, !isUploading, frozenFrames == nil,
              candidateBank.phase == .ready else { return }
        let selected = candidateBank.selected()
        let photos = candidateBank.clinical()
        guard (7...10).contains(selected.count), photos.count == 4 else { return }
        var frames: [String: CapturedFramePackage] = [:]
        for (i, sample) in selected.enumerated() {
            guard let package = candidatePackages[sample.hash] else { return }
            frames[String(format: "sweep_%02d", i)] = package
        }
        for (role, sample) in photos {
            guard let package = candidatePackages[sample.hash] else { return }
            frozenClinical[role] = package
        }
        frozenFrames = frames
        clinicalPhotos = frozenClinical
        // Release unselected image/depth buffers and stop the camera once sealed.
        candidatePackages.removeAll()
        pauseSession()
        isScanningActive = false
        guidanceFeedback = "Đang đóng gói dữ liệu đã chọn"
        print("Continuous scan: banked=\(candidateBank.acceptedCount), selected=\(frames.count), clinical=\(photos.count)")
        for sample in selected { print("Keyframe timestamp=\(sample.timestamp) yaw=\(sample.yaw) pitch=\(sample.pitch) score=\(sample.score) hash=\(sample.hash)") }
        triggerPackageUpload { _ in }
    }

    // MARK: - Mode 2: Rear Camera Processing

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
                    self.guidanceFeedback = "✓ Khuôn mặt rõ nét — Bấm chụp góc: \(self.currentStep.title)"
                    self.turnGuidance = "BẤM NÚT ĐỂ CHỤP"
                } else {
                    self.isTracking = false
                    self.isPoseAligned = false
                    self.guidanceFeedback = "Điều dưỡng: Hướng camera sau vào khuôn mặt bệnh nhân"
                    self.turnGuidance = "ĐƯA MẶT VÀO KHUNG"
                }
            }
        }
        processingQueue.async {
            try? handler.perform([request])
        }
    }

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

        // Native continuous mode has no manual per-frame capture path.
    }

    private func captureRearFrame(frame: ARFrame) throws {
        let ciImage = CIImage(cvPixelBuffer: frame.capturedImage).oriented(.right)
        guard let cgImage = Self.sharedCIContext.createCGImage(ciImage, from: ciImage.extent),
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

        var dummyVertices: [Float] = []
        dummyVertices.reserveCapacity(1220 * 3)
        for _ in 0..<1220 { dummyVertices.append(contentsOf: [0.0, 0.0, 0.0]) }
        var dummyIndices: [Int] = []
        dummyIndices.reserveCapacity(2304 * 3)
        for _ in 0..<(2304 * 3) { dummyIndices.append(0) }
        var dummyUVs: [Float] = []
        dummyUVs.reserveCapacity(1220 * 2)
        for _ in 0..<1220 { dummyUVs.append(contentsOf: [0.5, 0.5]) }

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

        if capturedFrames.count >= ScanAngleStep.allCases.count {
            triggerPackageUpload { _ in }
        }
    }

    private func createPackage(from frame: ARFrame, faceAnchor: ARFaceAnchor, pose: CameraRelativeFacePose, step: ScanAngleStep, viewTag: String = "") -> CapturedFramePackage? {
        let ciImage = CIImage(cvPixelBuffer: frame.capturedImage).oriented(.right)
        guard let cgImage = Self.sharedCIContext.createCGImage(ciImage, from: ciImage.extent),
              let rgbData = UIImage(cgImage: cgImage).jpegData(compressionQuality: 0.95) else {
            return nil
        }

        var depthData: Data?
        var depthW: Int?
        var depthH: Int?
        var depthIntrinsics: CameraIntrinsicsDTO?
        if abs(frame.capturedDepthDataTimestamp - frame.timestamp) <= 0.035,
           let capturedDepth = frame.capturedDepthData, let processed = DepthDataProcessor.processDepthData(capturedDepth),
           processed.validPointCount > processed.width * processed.height / 10 {
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
            translationMeters: ["x": pose.translationMeters.x, "y": pose.translationMeters.y, "z": pose.translationMeters.z],
            eulerRotationDeg: ["pitch": pose.pitchDeg, "yaw": pose.yawDeg, "roll": pose.rollDeg]
        )

        let geometry = faceAnchor.geometry
        guard geometry.vertices.count == 1220,
              geometry.triangleIndices.count == 2304 * 3,
              geometry.textureCoordinates.count == 1220 else { return nil }

        let faceToCamera = frame.camera.transform.inverse * faceAnchor.transform
        let visibleCount = geometry.vertices.filter { vertex in
            let point = faceToCamera * SIMD4<Float>(vertex.x, vertex.y, vertex.z, 1)
            let z = -point.z
            guard z > 0.05 else { return false }
            let u = intrinsicsDTO.fx * point.y / z + intrinsicsDTO.cx
            let v = intrinsicsDTO.fy * point.x / z + intrinsicsDTO.cy
            return u >= 0 && v >= 0 && u < Float(intrinsicsDTO.imageWidth) && v < Float(intrinsicsDTO.imageHeight)
        }.count
        guard Float(visibleCount) / Float(geometry.vertices.count) >= 0.9 else { return nil }

        let geometryDTO = ARKitFaceGeometryDTO(
            vertexCount: geometry.vertices.count,
            triangleCount: geometry.triangleIndices.count / 3,
            verticesMeters: geometry.vertices.flatMap { [$0.x, $0.y, $0.z] },
            triangleIndices: geometry.triangleIndices.map { Int($0) },
            textureCoordinates: geometry.textureCoordinates.flatMap { [$0.x, $0.y] },
            blendShapes: Dictionary(uniqueKeysWithValues: faceAnchor.blendShapes.map { ($0.key.rawValue, $0.value.floatValue) }),
            isTracked: faceAnchor.isTracked
        )

        let quality = QualityEvaluator.evaluate(pixelBuffer: frame.capturedImage, pose: pose,
            isFaceTracked: faceAnchor.isTracked, targetStep: .front)

        return CapturedFramePackage(
            step: step,
            viewTag: viewTag.isEmpty ? step.rawValue : viewTag,
            timestamp: frame.timestamp,
            rgbData: rgbData,
            depthData: depthData,
            depthWidth: depthW,
            depthHeight: depthH,
            depthIntrinsics: depthIntrinsics,
            intrinsics: intrinsicsDTO,
            pose: poseDTO,
            geometry: geometryDTO,
            quality: quality
        )
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
        guidanceFeedback = "Đang chụp lại góc: \(previous.title)"
    }

    public func triggerPackageUpload(completion: @escaping (Result<URL, Error>) -> Void) {
        guard !patientId.isEmpty else {
            lastErrorMessage = "Thiếu hồ sơ bệnh nhân. Hãy mở quét từ hồ sơ cần quét."
            return
        }
        guard let attempt = uploadAttempt.begin() else { return }
        if sessionId.isEmpty {
            sessionId = UUID().uuidString
        }

        isUploading = true
        lastErrorMessage = nil
        requiresNewScan = false
        guidanceFeedback = scannerMode == .rearClinicalAssistant ? "✓ Đang tải gói ảnh lâm sàng 48MP lên máy chủ..." : "✓ Đang tải gói TrueDepth Face ID lên AI Engine..."

        let allFramesToUpload: [String: CapturedFramePackage]
        if scannerMode == .faceIdSelfScan {
            guard let sealed = frozenFrames, !isBankingInProgress else {
                isUploading = false
                _ = uploadAttempt.finish(attempt)
                completion(.failure(NSError(domain: "Scanner", code: 409, userInfo: [NSLocalizedDescriptionKey: "Dữ liệu chưa được finalization."])))
                return
            }
            allFramesToUpload = sealed
        } else {
            allFramesToUpload = Dictionary(uniqueKeysWithValues: capturedFrames.map { ($0.key.rawValue, $0.value) })
        }
        let apiClient = activeApiClient ?? BackendAPIClient()
        self.activeApiClient = apiClient
        apiClient.onPackageFinalized = { [weak self] in
            guard let self = self, self.uploadAttempt.isCurrent(attempt) else { return }
            self.isSweepCompleted = true
            self.faceIdFilledCount = 36
            self.faceIdTicks = Array(repeating: true, count: 36)
        }
        apiClient.onProgressUpdate = { [weak self] progress, message in
            DispatchQueue.main.async {
                guard let self = self, self.uploadAttempt.isCurrent(attempt) else { return }
                self.uploadProgress = progress
                self.uploadStatusMessage = message
                self.guidanceFeedback = message
            }
        }

        apiClient.uploadScanPackage(patientId: patientId, sessionId: sessionId, frames: allFramesToUpload, clinicalPhotos: frozenClinical, scannerMode: scannerMode) { [weak self] result in
            DispatchQueue.main.async {
                guard let self = self, self.uploadAttempt.finish(attempt) else { return }
                self.isUploading = false
                switch result {
                case .success(let studioURL):
                    self.guidanceFeedback = "✓ Tải lên thành công! Đang mở 3D Studio..."
                    self.onScanCompleted?(studioURL)
                    completion(.success(studioURL))
                case .failure(let error):
                    self.lastErrorMessage = error.localizedDescription
                    self.requiresNewScan = (error as NSError).domain == "Reconstruction" && (error as NSError).code == 422
                    self.guidanceFeedback = "Lỗi: \(error.localizedDescription)"
                    completion(.failure(error))
                }
            }
        }
    }

    public func resetScan() {
        uploadAttempt.invalidate()
        speech.stopSpeaking(at: .immediate)
        spokenPhase = nil
        requiresNewScan = false
        if frozenFrames != nil { sessionId = UUID().uuidString }
        captureGeneration = UUID()
        candidateBank = ContinuousKeyframeBank()
        candidatePackages.removeAll()
        frozenFrames = nil; frozenClinical.removeAll()
        activeApiClient = nil
        lastBankedTimestamp = 0; lastPoseSample = nil
        isScanningActive = false
        isSweepCompleted = false
        isUploading = false
        uploadProgress = 0
        uploadStatusMessage = ""
        capturedFrames.removeAll()
        sweepFrames.removeAll()
        clinicalPhotos.removeAll()
        currentStep = .front
        lastErrorMessage = nil
        faceIdTicks = Array(repeating: false, count: 36)
        faceIdFilledCount = 0
        sweepStartTime = nil
        isFaceInFramingRect = false
        isBankingInProgress = false
        clearLivePoseState()
        guidanceFeedback = scannerMode == .faceIdSelfScan ? "Đưa khuôn mặt vào trong vòng tròn" : "Điều Dưỡng Quét: Đã sẵn sàng."
    }

    public func startActiveSweep() {
        resetScan()
        startSession()
        isScanningActive = true
        guidanceFeedback = "Định vị khuôn mặt trong vòng tròn"
        announceScanPhase()
    }

    private func announceScanPhase() {
        guard candidateBank.phase != spokenPhase else { return }
        spokenPhase = candidateBank.phase
        UIImpactFeedbackGenerator(style: .light).impactOccurred()
        guard voiceGuidanceEnabled else { return }
        try? AVAudioSession.sharedInstance().setCategory(.playback, mode: .spokenAudio, options: .duckOthers)
        try? AVAudioSession.sharedInstance().setActive(true)
        // Spoken direction changes only after a valid frame was banked.
        let utterance = AVSpeechUtterance(string: candidateBank.guidance)
        utterance.voice = AVSpeechSynthesisVoice(language: "vi-VN")
        utterance.rate = AVSpeechUtteranceDefaultSpeechRate * 0.9
        speech.stopSpeaking(at: .immediate)
        speech.speak(utterance)
    }

    private func clearLivePoseState() {
        isTracking = false
        isPoseAligned = false
        isAutoCapturing = false
        currentYawDeg = 0
        currentPitchDeg = 0
        currentRollDeg = 0
        currentYawErrorDeg = 0
        currentDistanceMeters = 0.45
    }

    private static func flattenColumnMajor(_ matrix: simd_float4x4) -> [Float] {
        [matrix.columns.0.x, matrix.columns.0.y, matrix.columns.0.z, matrix.columns.0.w,
         matrix.columns.1.x, matrix.columns.1.y, matrix.columns.1.z, matrix.columns.1.w,
         matrix.columns.2.x, matrix.columns.2.y, matrix.columns.2.z, matrix.columns.2.w,
         matrix.columns.3.x, matrix.columns.3.y, matrix.columns.3.z, matrix.columns.3.w]
    }
}
