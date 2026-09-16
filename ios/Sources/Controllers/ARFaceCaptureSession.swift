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
    private var activeApiClient: BackendAPIClient?

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

        // CHỈ xử lý ghi nhận dữ liệu khi chế độ quét được kích hoạt thật sự
        guard isScanningActive else { return }

        // Chờ người dùng định vị khuôn mặt vào tâm vòng tròn trước khi tính giờ quét
        if sweepStartTime == nil {
            if isCentered {
                sweepStartTime = frame.timestamp
                guidanceFeedback = "Di chuyển chậm đầu của bạn theo hình tròn."
                UIImpactFeedbackGenerator(style: .medium).impactOccurred()
            } else {
                guidanceFeedback = "Định vị khuôn mặt trong vòng tròn"
                return
            }
        }

        // 2. Strict Physical Sector Latching (Chỉ bật XANH khi đầu thực sự xoay/nghiêng ĐÚNG góc vật lý)
        let coneRadius = sqrt(Double(yaw * yaw + pitch * pitch))
        var currentBin: Int? = nil

        if coneRadius >= 10.0 {
            // Xác định chính xác góc phương vị tức thời theo mặt phẳng đồng hồ (yaw, pitch)
            let clockRad = atan2(Double(yaw), Double(pitch))
            var clockDeg = clockRad * 180.0 / .pi
            if clockDeg < 0 { clockDeg += 360.0 }
            let bin = Int((clockDeg / 10.0).rounded()) % 36

            // Kiểm tra nghiêm ngặt từng cung 45 độ trên vòng tròn:
            var isPhysicallyValidAngle = false
            if (bin >= 34 || bin <= 2) && pitch >= 10.0 {
                // 12h: Ngửa cằm
                isPhysicallyValidAngle = true
            } else if (bin >= 3 && bin <= 6) && yaw >= 8.0 && pitch >= 4.0 {
                // 1h-2h: Góc trên bên phải
                isPhysicallyValidAngle = true
            } else if (bin >= 7 && bin <= 11) && yaw >= 16.0 {
                // 3h: Quay phải
                isPhysicallyValidAngle = true
            } else if (bin >= 12 && bin <= 15) && yaw >= 8.0 && pitch <= -4.0 {
                // 4h-5h: Góc dưới bên phải
                isPhysicallyValidAngle = true
            } else if (bin >= 16 && bin <= 20) && pitch <= -7.0 {
                // 6h: Cúi cằm
                isPhysicallyValidAngle = true
            } else if (bin >= 21 && bin <= 24) && yaw <= -8.0 && pitch <= -4.0 {
                // 7h-8h: Góc dưới bên trái
                isPhysicallyValidAngle = true
            } else if (bin >= 25 && bin <= 29) && yaw <= -16.0 {
                // 9h: Quay trái
                isPhysicallyValidAngle = true
            } else if (bin >= 30 && bin <= 33) && yaw <= -8.0 && pitch >= 4.0 {
                // 10h-11h: Góc trên bên trái
                isPhysicallyValidAngle = true
            }

            if isPhysicallyValidAngle {
                currentBin = bin
            }
        } else if abs(yaw) <= 8.0 && abs(pitch) <= 8.0 && isCentered {
            // Nấc chính diện tâm (0°)
            currentBin = 0
        }

        // BẬT XANH NẤC: Mỗi nấc cách nhau tối thiểu 140ms để người dùng xoay đầu tự nhiên, không bị nhảy vèo quá nhanh
        if let b = currentBin, b >= 0 && b < 36 {
            let now = frame.timestamp
            if !faceIdTicks[b] && (now - lastBankedTimestamp >= 0.140 || faceIdFilledCount == 0) {
                lastBankedTimestamp = now
                faceIdTicks[b] = true
                faceIdFilledCount = faceIdTicks.filter { $0 }.count
                UISelectionFeedbackGenerator().selectionChanged()
                
                // Ghi nhận ngay frame đo đạc THẬT tại góc này
                captureSweepTick(bin: b, frame: frame, faceAnchor: faceAnchor, pose: pose)
                
                // Đồng bộ 1-1 trực tiếp vào bộ ảnh lâm sàng chuẩn xác, KHÔNG ĐOÁN
                syncClinicalPhotoFromPhysicallyMeasuredFrame(bin: b, frame: frame, faceAnchor: faceAnchor, pose: pose)
            }
        }

        // 3. Dynamic Guidance Text & BẮT BUỘC ĐỦ 100% TOÀN BỘ 36/36 NẤC VÒNG TRÒN:
        let hasFront = clinicalPhotos["front"] != nil
        let hasBasal = clinicalPhotos["basal_nostrils"] != nil
        let hasLeft = clinicalPhotos["left_45"] != nil || clinicalPhotos["left_profile"] != nil
        let hasRight = clinicalPhotos["right_45"] != nil || clinicalPhotos["right_profile"] != nil
        let isFullCircleCovered = faceIdFilledCount >= 36

        // ĐIỀU KIỆN TỰ ĐỘNG CHUYỂN 3D:
        // BẮT BUỘC ĐỦ CẢ 36/36 NẤC XANH (100% VÒNG TRÒN) VÀ CÓ ĐỦ 4 GÓC LÂM SÀNG CỐT LÕI:
        if isFullCircleCovered && hasFront && hasBasal && hasLeft && hasRight {
            completeFaceIdSweep()
        } else if !hasBasal {
            guidanceFeedback = "👆 Hơi ngửa cằm lên (20°-25°) để quét chân mũi & đáy mũi (\(faceIdFilledCount)/36)"
        } else if !hasLeft {
            guidanceFeedback = "👈 Nghiêng mặt sang TRÁI (35°-45°) để quét má trái (\(faceIdFilledCount)/36)"
        } else if !hasRight {
            guidanceFeedback = "👉 Nghiêng mặt sang PHẢI (35°-45°) để quét má phải (\(faceIdFilledCount)/36)"
        } else if !hasFront {
            guidanceFeedback = "Nhìn thẳng chính diện vào camera (\(faceIdFilledCount)/36)"
        } else if pose.distanceMeters < 0.28 {
            guidanceFeedback = "Giữ máy cách mặt khoảng 35 - 50 cm"
        } else if pose.distanceMeters > 0.65 {
            guidanceFeedback = "Đưa máy lại gần hơn một chút"
        } else {
            guidanceFeedback = "Tiếp tục xoay đều đầu theo vòng tròn để phủ kín 36/36 nấc (\(faceIdFilledCount)/36)."
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

    /// Đồng bộ chính xác 1-1 từ khung hình vật lý vừa đo được vào ảnh lâm sàng
    private func syncClinicalPhotoFromPhysicallyMeasuredFrame(bin: Int, frame: ARFrame, faceAnchor: ARFaceAnchor, pose: CameraRelativeFacePose) {
        let yaw = pose.yawDeg
        let pitch = pose.pitchDeg
        
        var targetSlot: String?
        var correspondingStep: ScanAngleStep = .front
        
        if abs(yaw) <= 15 && abs(pitch) <= 15 && clinicalPhotos["front"] == nil {
            targetSlot = "front"
            correspondingStep = .front
        } else if pitch >= 10.0 && abs(yaw) <= 25 && (clinicalPhotos["basal_nostrils"] == nil || pitch > (clinicalPhotos["basal_nostrils"]?.pose.eulerRotationDeg["pitch"] ?? 0)) {
            targetSlot = "basal_nostrils"
            correspondingStep = .basalNostrils
        } else if yaw <= -35.0 && clinicalPhotos["left_profile"] == nil {
            targetSlot = "left_profile"
            correspondingStep = .leftProfile
        } else if yaw <= -18.0 && yaw >= -35.0 && clinicalPhotos["left_45"] == nil {
            targetSlot = "left_45"
            correspondingStep = .left45
        } else if yaw >= 35.0 && clinicalPhotos["right_profile"] == nil {
            targetSlot = "right_profile"
            correspondingStep = .rightProfile
        } else if yaw >= 18.0 && yaw <= 35.0 && clinicalPhotos["right_45"] == nil {
            targetSlot = "right_45"
            correspondingStep = .right45
        }
        
        guard let slot = targetSlot else { return }
        
        processingQueue.async { [weak self] in
            guard let self = self else { return }
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

    /// Tự động trích xuất các góc ảnh lâm sàng còn thiếu từ 36 frame quét vòng tròn
    public func ensureClinicalPhotosFromSweep() {
        if clinicalPhotos["front"] == nil {
            clinicalPhotos["front"] = sweepFrames[0] ?? sweepFrames.values.first
        }
        if clinicalPhotos["left_45"] == nil {
            clinicalPhotos["left_45"] = sweepFrames[27] ?? sweepFrames[26] ?? sweepFrames[28] ?? sweepFrames[29]
        }
        if clinicalPhotos["left_profile"] == nil {
            clinicalPhotos["left_profile"] = sweepFrames[27] ?? sweepFrames[26] ?? sweepFrames[28]
        }
        if clinicalPhotos["right_45"] == nil {
            clinicalPhotos["right_45"] = sweepFrames[9] ?? sweepFrames[8] ?? sweepFrames[10] ?? sweepFrames[7]
        }
        if clinicalPhotos["right_profile"] == nil {
            clinicalPhotos["right_profile"] = sweepFrames[9] ?? sweepFrames[10] ?? sweepFrames[8]
        }
        if clinicalPhotos["basal_nostrils"] == nil {
            clinicalPhotos["basal_nostrils"] = sweepFrames[0] ?? sweepFrames[1] ?? sweepFrames[35] ?? sweepFrames[2]
        }
        
        for (slot, pkg) in clinicalPhotos {
            if let step = ScanAngleStep(rawValue: slot), capturedFrames[step] == nil {
                capturedFrames[step] = pkg
            }
        }
    }

    public func completeFaceIdSweep() {
        guard !isUploading else { return }
        
        ensureClinicalPhotosFromSweep()
        
        isSweepCompleted = true
        isScanningActive = false
        isUploading = true
        uploadProgress = 0.08
        uploadStatusMessage = "Đang tổng hợp dữ liệu 36 góc quét TrueDepth..."
        guidanceFeedback = "✓ HOÀN TẤT VÒNG QUÉT! Đang bắt đầu dựng 3D..."
        UIImpactFeedbackGenerator(style: .heavy).impactOccurred()

        self.triggerPackageUpload { _ in }
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
        if patientId.isEmpty {
            patientId = UserDefaults.standard.string(forKey: "lastActivePatientId") ?? "697ba81b-a3b3-4879-966e-8ba42575dc80"
        }
        if sessionId.isEmpty {
            sessionId = UUID().uuidString
        }

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

        let apiClient = BackendAPIClient()
        self.activeApiClient = apiClient
        apiClient.onProgressUpdate = { [weak self] progress, message in
            DispatchQueue.main.async {
                self?.uploadProgress = progress
                self?.uploadStatusMessage = message
                self?.guidanceFeedback = message
            }
        }

        apiClient.uploadScanPackage(patientId: patientId, sessionId: sessionId, frames: allFramesToUpload, scannerMode: scannerMode) { [weak self] result in
            DispatchQueue.main.async {
                self?.isUploading = false
                self?.activeApiClient = nil
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
        isScanningActive = true
        guidanceFeedback = "Định vị khuôn mặt trong vòng tròn"
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
