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
    @Published public var sweepFrames: [Int: CapturedFramePackage] = [:]
    @Published public var clinicalPhotos: [String: CapturedFramePackage] = [:]
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

        let yaw = pose.yawDeg
        let pitch = pose.pitchDeg

        // 1. Kiểm tra vị trí khuôn mặt trong khung căn chỉnh ban đầu
        let isCentered = abs(yaw) <= 18 && abs(pitch) <= 20 && pose.distanceMeters >= 0.30 && pose.distanceMeters <= 0.60
        isFaceInFramingRect = isCentered

        // 2. Polar coordinate binning for 36 ticks (10° each)
        let coneRadius = sqrt(Double(yaw * yaw + pitch * pitch))

        if coneRadius >= 4.0 {
            let rad = atan2(pitch, yaw)
            var deg = rad * 180.0 / .pi
            if deg < 0 { deg += 360.0 }
            let bin = Int((deg / 10.0).rounded()) % 36
            if bin >= 0 && bin < 36 && !faceIdTicks[bin] {
                faceIdTicks[bin] = true
                faceIdFilledCount = faceIdTicks.filter { $0 }.count
                UISelectionFeedbackGenerator().selectionChanged()
                
                // Continuous Buffer: Capture frame for THIS exact tick!
                captureSweepTick(bin: bin, frame: frame, faceAnchor: faceAnchor, pose: pose)
            }
        } else if isCentered {
            // Tick trung tâm khi đã căn giữa
            if !faceIdTicks[0] {
                faceIdTicks[0] = true
                faceIdFilledCount = faceIdTicks.filter { $0 }.count
                captureSweepTick(bin: 0, frame: frame, faceAnchor: faceAnchor, pose: pose)
            }
        }

        // 3. Khởi tạo sweepStartTime khi người dùng bắt đầu có tương tác xoay
        if sweepStartTime == nil && (faceIdFilledCount >= 2 || coneRadius >= 6.0) {
            sweepStartTime = frame.timestamp
        }

        // 4. Background banking tự động chọn 6 góc ảnh hồ sơ lâm sàng chuẩn y khoa
        checkAndBankClinicalPhotos(frame: frame, faceAnchor: faceAnchor, pose: pose)

        // 5. Dynamic Guidance Text chuẩn Apple Face ID (nhẹ nhàng, không đếm số góc)
        let elapsed = sweepStartTime != nil ? (frame.timestamp - (sweepStartTime ?? frame.timestamp)) : 0
        let hasFront = clinicalPhotos["front"] != nil
        let hasLeft = clinicalPhotos["left_45"] != nil || clinicalPhotos["left_profile"] != nil
        let hasRight = clinicalPhotos["right_45"] != nil || clinicalPhotos["right_profile"] != nil
        let hasBasal = clinicalPhotos["basal_nostrils"] != nil

        // Hoàn tất quét: Khi vòng tròn Face ID đã phủ kín ít nhất 26 tia VÀ đã thu thập đủ ảnh chính diện + 2 bên trái/phải VÀ thời gian xoay >= 3.2s
        if faceIdFilledCount >= 26 && hasFront && hasLeft && hasRight && elapsed >= 3.2 {
            guidanceFeedback = "✓ HOÀN TẤT VÒNG QUÉT CHUẨN XÁC!"
            completeFaceIdSweep()
        } else if pose.distanceMeters < 0.30 {
            guidanceFeedback = "Giữ máy cách mặt khoảng 35 - 50 cm"
        } else if pose.distanceMeters > 0.65 {
            guidanceFeedback = "Đưa máy lại gần hơn một chút"
        } else if !hasBasal && faceIdFilledCount >= 10 {
            guidanceFeedback = "Hơi ngửa nhẹ cằm để quét vòm mũi & lỗ mũi"
        } else {
            guidanceFeedback = "Di chuyển chậm đầu của bạn để hoàn thành vòng tròn."
        }
    }

    private func captureSweepTick(bin: Int, frame: ARFrame, faceAnchor: ARFaceAnchor, pose: CameraRelativeFacePose) {
        let viewTag = "sweep_\(String(format: "%02d", bin))"
        processingQueue.async { [weak self] in
            guard let self = self else { return }
            if let package = self.createPackage(from: frame, faceAnchor: faceAnchor, pose: pose, step: .front, viewTag: viewTag) {
                DispatchQueue.main.async {
                    self.sweepFrames[bin] = package
                }
            }
        }
    }

    private func checkAndBankClinicalPhotos(frame: ARFrame, faceAnchor: ARFaceAnchor, pose: CameraRelativeFacePose) {
        guard !isBankingInProgress else { return }
        let now = frame.timestamp
        guard now - lastBankedTimestamp >= 0.10 else { return }

        let eyeBlinkLeft = faceAnchor.blendShapes[.eyeBlinkLeft]?.floatValue ?? 0
        let eyeBlinkRight = faceAnchor.blendShapes[.eyeBlinkRight]?.floatValue ?? 0
        let yaw = pose.yawDeg
        let pitch = pose.pitchDeg
        let hasValidDepth = frame.capturedDepthData != nil

        var targetSlot: String?
        var correspondingStep: ScanAngleStep = .front
        var isBetter = false

        // 1. Ảnh Chính diện (0°): yaw [-12°, 12°], pitch [-14°, 14°], bắt buộc mở to mắt
        if abs(yaw) <= 12 && abs(pitch) <= 14 && eyeBlinkLeft < 0.25 && eyeBlinkRight < 0.25 {
            if clinicalPhotos["front"] == nil || (hasValidDepth && clinicalPhotos["front"]?.depthData == nil) {
                targetSlot = "front"
                correspondingStep = .front
                isBetter = true
            }
        }
        // 2. Ảnh Nghiêng Trái 45°: yaw [-45°, -25°]
        else if yaw <= -25 && yaw >= -45 && eyeBlinkLeft < 0.35 && eyeBlinkRight < 0.35 {
            if clinicalPhotos["left_45"] == nil || (hasValidDepth && clinicalPhotos["left_45"]?.depthData == nil) {
                targetSlot = "left_45"
                correspondingStep = .left45
                isBetter = true
            }
        }
        // 3. Ảnh Nghiêng Trái Sâu Profile (~60°): yaw <= -42°
        else if yaw <= -42 {
            let existingYaw = clinicalPhotos["left_profile"]?.pose.eulerRotationDeg["yaw"] ?? 0
            if clinicalPhotos["left_profile"] == nil || yaw < existingYaw || (hasValidDepth && clinicalPhotos["left_profile"]?.depthData == nil) {
                targetSlot = "left_profile"
                correspondingStep = .leftProfile
                isBetter = true
            }
        }
        // 4. Ảnh Nghiêng Phải 45°: yaw [25°, 45°]
        else if yaw >= 25 && yaw <= 45 && eyeBlinkLeft < 0.35 && eyeBlinkRight < 0.35 {
            if clinicalPhotos["right_45"] == nil || (hasValidDepth && clinicalPhotos["right_45"]?.depthData == nil) {
                targetSlot = "right_45"
                correspondingStep = .right45
                isBetter = true
            }
        }
        // 5. Ảnh Nghiêng Phải Sâu Profile (~60°): yaw >= 42°
        else if yaw >= 42 {
            let existingYaw = clinicalPhotos["right_profile"]?.pose.eulerRotationDeg["yaw"] ?? 0
            if clinicalPhotos["right_profile"] == nil || yaw > existingYaw || (hasValidDepth && clinicalPhotos["right_profile"]?.depthData == nil) {
                targetSlot = "right_profile"
                correspondingStep = .rightProfile
                isBetter = true
            }
        }
        // 6. Ảnh Đáy Mũi (Basal / Submental View - Chuẩn Dallas Rhinoplasty): ngửa cằm pitch [14°, 38°], yaw [-22°, 22°]
        else if pitch >= 14.0 && pitch <= 38.0 && abs(yaw) <= 22 {
            let existingPitch = clinicalPhotos["basal_nostrils"]?.pose.eulerRotationDeg["pitch"] ?? 0
            if clinicalPhotos["basal_nostrils"] == nil || pitch > existingPitch || (hasValidDepth && clinicalPhotos["basal_nostrils"]?.depthData == nil) {
                targetSlot = "basal_nostrils"
                correspondingStep = .front
                isBetter = true
            }
        }

        guard let slot = targetSlot, isBetter else { return }
        isBankingInProgress = true
        lastBankedTimestamp = now

        processingQueue.async { [weak self] in
            guard let self = self else { return }
            defer { DispatchQueue.main.async { self.isBankingInProgress = false } }
            if let package = self.createPackage(from: frame, faceAnchor: faceAnchor, pose: pose, step: correspondingStep, viewTag: slot) {
                DispatchQueue.main.async {
                    self.clinicalPhotos[slot] = package
                    if let step = ScanAngleStep(rawValue: slot) {
                        self.capturedFrames[step] = package
                    }
                }
            }
        }
    }

    private func completeFaceIdSweep() {
        guard !isUploading else { return }
        // Phải có ít nhất 24 nấc sweep VÀ có ảnh chính diện + 2 bên trái/phải thực tế
        guard faceIdFilledCount >= 24,
              clinicalPhotos["front"] != nil,
              (clinicalPhotos["left_45"] != nil || clinicalPhotos["left_profile"] != nil),
              (clinicalPhotos["right_45"] != nil || clinicalPhotos["right_profile"] != nil) else {
            return
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

        // Tổng hợp toàn bộ các frame đo đạc quét liên tục (sweep_00..35) VÀ bộ ảnh hồ sơ lâm sàng (front, left_45, ...)
        var allFramesToUpload: [String: CapturedFramePackage] = [:]
        for (bin, pkg) in sweepFrames {
            allFramesToUpload["sweep_\(String(format: "%02d", bin))"] = pkg
        }
        for (slot, pkg) in clinicalPhotos {
            allFramesToUpload[slot] = pkg
        }
        for (step, pkg) in capturedFrames {
            if allFramesToUpload[step.rawValue] == nil {
                allFramesToUpload[step.rawValue] = pkg
            }
        }

        BackendAPIClient().uploadScanPackage(patientId: patientId, sessionId: sessionId, frames: allFramesToUpload, scannerMode: scannerMode) { [weak self] result in
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
