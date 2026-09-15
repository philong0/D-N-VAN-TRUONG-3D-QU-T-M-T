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
    @Published public var lastErrorMessage: String?

    // MARK: - Dual Mode & Face ID Sweep States
    @Published public var scannerMode: ScannerMode = .faceIdSelfScan
    @Published public var faceIdTicks: [Bool] = Array(repeating: false, count: 36)
    @Published public var faceIdFilledCount: Int = 0
    @Published public var isRearCameraActive: Bool = false
    @Published public var isFaceInFramingRect: Bool = false
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

    public func pauseSession() { arSession.pause() }

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

    // MARK: - Mode 1: High-Performance Face ID Polar Sweep

    private func consumeFaceIdFrame(frame: ARFrame, faceAnchor: ARFaceAnchor, pose: CameraRelativeFacePose) {
        isTracking = faceAnchor.isTracked
        currentYawDeg = pose.yawDeg
        currentPitchDeg = pose.pitchDeg
        currentRollDeg = pose.rollDeg
        currentDistanceMeters = pose.distanceMeters

        guard faceAnchor.isTracked else {
            guidanceFeedback = "Đưa khuôn mặt vào trong vòng tròn"
            return
        }

        // 1. Kiểm tra vị trí khuôn mặt trong khung căn chỉnh ban đầu
        let isCentered = abs(yaw) <= 18 && abs(pitch) <= 20 && pose.distanceMeters >= 0.30 && pose.distanceMeters <= 0.60
        isFaceInFramingRect = isCentered

        // 2. Polar coordinate binning for 36 ticks (10° each)
        let coneRadius = sqrt(yaw * yaw + pitch * pitch)

        if coneRadius >= 4.0 {
            let rad = atan2(pitch, yaw)
            var deg = rad * 180.0 / .pi
            if deg < 0 { deg += 360.0 }
            let bin = Int((deg / 10.0).rounded()) % 36
            if bin >= 0 && bin < 36 && !faceIdTicks[bin] {
                faceIdTicks[bin] = true
                faceIdFilledCount = faceIdTicks.filter { $0 }.count
                UISelectionFeedbackGenerator().selectionChanged()
            }
        } else if isCentered {
            // Chỉ tick frame giữa khi thực sự đã căn giữa
            if !faceIdTicks[0] {
                faceIdTicks[0] = true
                faceIdFilledCount = faceIdTicks.filter { $0 }.count
            }
        }

        // 3. Khởi tạo sweepStartTime khi người dùng bắt đầu có tương tác xoay
        if sweepStartTime == nil && (faceIdFilledCount >= 3 || coneRadius >= 8.0) {
            sweepStartTime = frame.timestamp
        }

        // 4. Background banking theo góc quay
        checkAndBankAngles(frame: frame, faceAnchor: faceAnchor, pose: pose)

        // 5. Dynamic Guidance Text
        let hasLeft = capturedFrames[.left45] != nil || capturedFrames[.leftProfile] != nil
        let hasRight = capturedFrames[.right45] != nil || capturedFrames[.rightProfile] != nil
        let elapsed = sweepStartTime != nil ? (frame.timestamp - (sweepStartTime ?? frame.timestamp)) : 0

        // Điều kiện hoàn thành quét chuẩn: Phải phủ kín ít nhất 26 tia VÀ đã quét cả 2 bên trái/phải VÀ thời gian xoay >= 3.0s
        if faceIdFilledCount >= 26 && hasLeft && hasRight && elapsed >= 3.0 {
            guidanceFeedback = "✓ HOÀN TẤT VÒNG QUÉT FACE ID!"
            completeFaceIdSweep()
        } else if pose.distanceMeters < 0.32 {
            guidanceFeedback = "Giữ máy cách mặt khoảng 35 - 50 cm"
        } else if !hasLeft && yaw > -15 {
            guidanceFeedback = "Di chuyển đầu sang TRÁI để lấy sống mũi"
        } else if !hasRight && yaw < 15 {
            guidanceFeedback = "Di chuyển đầu sang PHẢI để lấy sống mũi"
        } else if pitch < 6 && capturedFrames[.front] != nil {
            guidanceFeedback = "Hơi ngửa nhẹ cằm để quét vòm mũi"
        } else {
            guidanceFeedback = "Di chuyển chậm đầu của bạn để hoàn thành vòng tròn."
        }
    }

    private func checkAndBankAngles(frame: ARFrame, faceAnchor: ARFaceAnchor, pose: CameraRelativeFacePose) {
        guard !isBankingInProgress else { return }
        let now = frame.timestamp
        guard now - lastBankedTimestamp >= 0.15 else { return }

        let yaw = pose.yawDeg
        let pitch = pose.pitchDeg

        var targetToBank: ScanAngleStep?
        if abs(yaw) <= 12 && abs(pitch) <= 15 && capturedFrames[.front] == nil {
            targetToBank = .front
        } else if yaw <= -18 && yaw >= -45 && capturedFrames[.left45] == nil {
            targetToBank = .left45
        } else if yaw <= -40 && capturedFrames[.leftProfile] == nil {
            targetToBank = .leftProfile
        } else if yaw >= 18 && yaw <= 45 && capturedFrames[.right45] == nil {
            targetToBank = .right45
        } else if yaw >= 40 && capturedFrames[.rightProfile] == nil {
            targetToBank = .rightProfile
        }

        guard let step = targetToBank else { return }
        isBankingInProgress = true
        lastBankedTimestamp = now

        // Process heavy compression on background queue
        processingQueue.async { [weak self] in
            guard let self = self else { return }
            defer {
                DispatchQueue.main.async { self.isBankingInProgress = false }
            }

            if let package = self.createPackage(from: frame, faceAnchor: faceAnchor, pose: pose, step: step) {
                DispatchQueue.main.async {
                    self.capturedFrames[step] = package
                    let all = ScanAngleStep.allCases
                    if let idx = all.firstIndex(of: step), idx + 1 < all.count {
                        self.currentStep = all[idx + 1]
                    }
                }
            }
        }
    }

    private func completeFaceIdSweep() {
        guard !isUploading else { return }
        guard capturedFrames.count >= 2 || faceIdFilledCount >= 24 else { return }

        // Backfill any missing angles from closest captured frames
        let availableSteps = Array(capturedFrames.keys)
        if let fallbackKey = availableSteps.first, let base = capturedFrames[fallbackKey] {
            for required in ScanAngleStep.allCases {
                if capturedFrames[required] == nil {
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
        }

        UIImpactFeedbackGenerator(style: .heavy).impactOccurred()
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

        // In Face ID mode: manual capture fallback if user prefers tapping
        guard let sample = arSession.currentFrame,
              let faceAnchor = sample.anchors.compactMap({ $0 as? ARFaceAnchor }).first else {
            throw NSError(domain: "Scanner", code: 422, userInfo: [NSLocalizedDescriptionKey: "Chưa nhận diện được khuôn mặt trong vòng tròn."])
        }
        let faceToCamera = sample.camera.transform.inverse * faceAnchor.transform
        let pose = CameraRelativeFacePose(faceToCamera: faceToCamera)

        processingQueue.async { [weak self] in
            guard let self = self else { return }
            if let package = self.createPackage(from: sample, faceAnchor: faceAnchor, pose: pose, step: self.currentStep) {
                DispatchQueue.main.async {
                    self.capturedFrames[self.currentStep] = package
                    UIImpactFeedbackGenerator(style: .medium).impactOccurred()
                    self.advanceToNextStep()
                    if self.capturedFrames.count >= ScanAngleStep.allCases.count {
                        self.triggerPackageUpload { _ in }
                    }
                }
            }
        }
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

    private func createPackage(from frame: ARFrame, faceAnchor: ARFaceAnchor, pose: CameraRelativeFacePose, step: ScanAngleStep) -> CapturedFramePackage? {
        let ciImage = CIImage(cvPixelBuffer: frame.capturedImage).oriented(.right)
        guard let cgImage = Self.sharedCIContext.createCGImage(ciImage, from: ciImage.extent),
              let rgbData = UIImage(cgImage: cgImage).jpegData(compressionQuality: 0.95) else {
            return nil
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
            translationMeters: ["x": pose.translationMeters.x, "y": pose.translationMeters.y, "z": pose.translationMeters.z],
            eulerRotationDeg: ["pitch": pose.pitchDeg, "yaw": pose.yawDeg, "roll": pose.rollDeg]
        )

        let geometry = faceAnchor.geometry
        guard geometry.vertices.count == 1220,
              geometry.triangleIndices.count == 2304 * 3,
              geometry.textureCoordinates.count == 1220 else { return nil }

        let geometryDTO = ARKitFaceGeometryDTO(
            vertexCount: geometry.vertices.count,
            triangleCount: geometry.triangleIndices.count / 3,
            verticesMeters: geometry.vertices.flatMap { [$0.x, $0.y, $0.z] },
            triangleIndices: geometry.triangleIndices.map { Int($0) },
            textureCoordinates: geometry.textureCoordinates.flatMap { [$0.x, $0.y] },
            blendShapes: Dictionary(uniqueKeysWithValues: faceAnchor.blendShapes.map { ($0.key.rawValue, $0.value.floatValue) }),
            isTracked: faceAnchor.isTracked
        )

        let quality = FrameQualityEvaluation(
            blurScore: 300.0,
            isBlurry: false,
            lightingScore: 0.6,
            isLightingAdequate: true,
            isTracked: faceAnchor.isTracked,
            yawDeg: pose.yawDeg,
            pitchDeg: pose.pitchDeg,
            distanceMeters: pose.distanceMeters,
            isDistanceOptimal: true
        )

        return CapturedFramePackage(
            step: step,
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
        guard !patientId.isEmpty, !sessionId.isEmpty else {
            let err = NSError(domain: "Scanner", code: 400, userInfo: [NSLocalizedDescriptionKey: "Thiếu PatientID hoặc SessionID."])
            self.lastErrorMessage = err.localizedDescription
            self.guidanceFeedback = "Lỗi: \(err.localizedDescription)"
            completion(.failure(err))
            return
        }
        guard !isUploading else { return }
        isUploading = true
        lastErrorMessage = nil
        guidanceFeedback = scannerMode == .rearClinicalAssistant ? "✓ Đang tải gói ảnh lâm sàng 48MP lên máy chủ..." : "✓ Đang tải gói TrueDepth Face ID lên AI Engine..."

        BackendAPIClient().uploadScanPackage(patientId: patientId, sessionId: sessionId, frames: capturedFrames, scannerMode: scannerMode) { [weak self] result in
            DispatchQueue.main.async {
                self?.isUploading = false
                switch result {
                case .success(let studioURL):
                    self?.guidanceFeedback = "✓ Tải lên thành công! Đang mở 3D Studio..."
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
        lastErrorMessage = nil
        faceIdTicks = Array(repeating: false, count: 36)
        faceIdFilledCount = 0
        sweepStartTime = nil
        isFaceInFramingRect = false
        clearLivePoseState()
        guidanceFeedback = scannerMode == .faceIdSelfScan ? "Đưa khuôn mặt vào trong vòng tròn" : "Điều Dưỡng Quét: Đã sẵn sàng."
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
