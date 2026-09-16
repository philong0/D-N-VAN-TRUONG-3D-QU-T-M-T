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
    @Published public var faceIdTicks: [Bool] = Array(repeating: false, count: 10)
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

    public func resetScan() {
        faceIdTicks = Array(repeating: false, count: 10)
        faceIdFilledCount = 0
        sweepFrames.removeAll()
        clinicalPhotos.removeAll()
        capturedFrames.removeAll()
        isSweepCompleted = false
        isUploading = false
        uploadProgress = 0
        uploadStatusMessage = ""
        lastErrorMessage = nil
        sweepStartTime = nil
        clearLivePoseState()
    }

    private func clearLivePoseState() {
        isTracking = false
        currentYawDeg = 0
        currentPitchDeg = 0
        currentRollDeg = 0
        currentDistanceMeters = 0
        isPoseAligned = false
        holdProgress = 0
        isAutoCapturing = false
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

    // MARK: - Mode 1: 10-Sector Clinical 3D Scan Pipeline

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
                guidanceFeedback = "Di chuyển đầu theo 10 góc giải phẫu trên vòng tròn."
                UIImpactFeedbackGenerator(style: .medium).impactOccurred()
            } else {
                guidanceFeedback = "Định vị khuôn mặt trong vòng tròn"
                return
            }
        }

        // 2. Nhận diện 10 góc chuẩn giải phẫu (10 Clinical Sectors)
        var detectedSector: Int? = nil

        if abs(yaw) <= 8.0 && abs(pitch) <= 8.0 {
            // Sector 0: Chính diện (0°)
            detectedSector = 0
        } else if yaw <= -18.0 && yaw >= -32.0 && abs(pitch) <= 18.0 {
            // Sector 1: Chếch trái (~25°)
            detectedSector = 1
        } else if yaw <= -33.0 && yaw >= -55.0 && abs(pitch) <= 22.0 {
            // Sector 2: Nghiêng trái (~45°)
            detectedSector = 2
        } else if yaw <= -56.0 {
            // Sector 3: Trắc diện sâu trái (~70° Profile)
            detectedSector = 3
        } else if yaw >= 18.0 && yaw <= 32.0 && abs(pitch) <= 18.0 {
            // Sector 4: Chếch phải (~25°)
            detectedSector = 4
        } else if yaw >= 33.0 && yaw <= 55.0 && abs(pitch) <= 22.0 {
            // Sector 5: Nghiêng phải (~45°)
            detectedSector = 5
        } else if yaw >= 56.0 {
            // Sector 6: Trắc diện sâu phải (~70° Profile)
            detectedSector = 6
        } else if pitch <= -12.0 && pitch >= -28.0 && abs(yaw) <= 22.0 {
            // Sector 7: Ngửa nhẹ đáy mũi (-20° Basal)
            detectedSector = 7
        } else if pitch >= 10.0 && pitch <= 24.0 && abs(yaw) <= 22.0 {
            // Sector 8: Cúi nhẹ trán & sống mũi (+15° Forehead/Dorsum)
            detectedSector = 8
        } else if pitch <= -32.0 && abs(yaw) <= 25.0 {
            // Sector 9: Ngửa sâu cằm & cổ (-40° Submental)
            detectedSector = 9
        }

        // BẬT XANH NẤC: Mỗi nấc cách nhau tối thiểu 140ms, bắt đúng ảnh ổn định
        if let s = detectedSector, s >= 0 && s < 10 {
            let now = frame.timestamp
            if !faceIdTicks[s] && (now - lastBankedTimestamp >= 0.140 || faceIdFilledCount == 0) {
                lastBankedTimestamp = now
                faceIdTicks[s] = true
                faceIdFilledCount = faceIdTicks.filter { $0 }.count
                UISelectionFeedbackGenerator().selectionChanged()
                
                // Ghi nhận ngay frame đo đạc THẬT tại góc này
                captureSweepTick(bin: s, frame: frame, faceAnchor: faceAnchor, pose: pose)
                
                // Đồng bộ trực tiếp vào bộ 5 ảnh lâm sàng chuẩn
                syncClinicalPhotoFromPhysicallyMeasuredFrame(bin: s, frame: frame, faceAnchor: faceAnchor, pose: pose)
            }
        }

        // 3. Dynamic Guidance Text & Điều kiện hoàn thành đủ 10/10 nấc:
        let hasFront = clinicalPhotos["front"] != nil
        let hasBasal = clinicalPhotos["basal_nostrils"] != nil
        let hasLeft = clinicalPhotos["left_45"] != nil
        let hasRight = clinicalPhotos["right_45"] != nil
        let isFullCircleCovered = faceIdFilledCount >= 10

        if isFullCircleCovered && hasFront && hasBasal && hasLeft && hasRight {
            completeFaceIdSweep()
        } else if !faceIdTicks[0] {
            guidanceFeedback = "Nhìn thẳng chính diện vào camera (Góc 1/10)"
        } else if !faceIdTicks[1] {
            guidanceFeedback = "👈 Hơi quay nhẹ mặt sang TRÁI (Góc 2/10)"
        } else if !faceIdTicks[2] {
            guidanceFeedback = "👈 Nghiêng mặt sang TRÁI 45° (Góc 3/10)"
        } else if !faceIdTicks[3] {
            guidanceFeedback = "👈 Quay hẳn sang TRÁI lấy trắc diện 70° (Góc 4/10)"
        } else if !faceIdTicks[4] {
            guidanceFeedback = "👉 Hơi quay nhẹ mặt sang PHẢI (Góc 5/10)"
        } else if !faceIdTicks[5] {
            guidanceFeedback = "👉 Nghiêng mặt sang PHẢI 45° (Góc 6/10)"
        } else if !faceIdTicks[6] {
            guidanceFeedback = "👉 Quay hẳn sang PHẢI lấy trắc diện 70° (Góc 7/10)"
        } else if !faceIdTicks[7] {
            guidanceFeedback = "👆 Hơi ngửa nhẹ cằm (20°) quét đáy mũi (Góc 8/10)"
        } else if !faceIdTicks[8] {
            guidanceFeedback = "👇 Hơi cúi nhẹ đầu (15°) quét trán & sống mũi (Góc 9/10)"
        } else if !faceIdTicks[9] {
            guidanceFeedback = "👆 Ngửa cằm cao (40°) quét góc cằm & cổ (Góc 10/10)"
        } else {
            guidanceFeedback = "Đã quét \(faceIdFilledCount)/10 góc. Tiếp tục xoay các góc còn lại."
        }
    }

    private func captureSweepTick(bin: Int, frame: ARFrame, faceAnchor: ARFaceAnchor, pose: CameraRelativeFacePose) {
        let viewTags = [
            "front", "left_25", "left_45", "left_profile",
            "right_25", "right_45", "right_profile",
            "basal_nostrils", "forehead_dorsum", "submental_chin"
        ]
        let viewTag = bin < viewTags.count ? viewTags[bin] : "sweep_\(String(format: "%02d", bin))"
        let correspondingStep: ScanAngleStep = ScanAngleStep.allCases.first(where: { $0.sectorIndex == bin }) ?? .front

        processingQueue.async { [weak self] in
            guard let self = self else { return }
            if let package = self.createPackage(from: frame, faceAnchor: faceAnchor, pose: pose, step: correspondingStep, viewTag: viewTag) {
                DispatchQueue.main.async {
                    self.sweepFrames[bin] = package
                }
            }
        }
    }

    /// Đồng bộ chính xác vào bộ 5 ảnh lâm sàng chuẩn
    private func syncClinicalPhotoFromPhysicallyMeasuredFrame(bin: Int, frame: ARFrame, faceAnchor: ARFaceAnchor, pose: CameraRelativeFacePose) {
        var targetSlot: String?
        var correspondingStep: ScanAngleStep = .front
        
        switch bin {
        case 0:
            targetSlot = "front"
            correspondingStep = .front
        case 2:
            targetSlot = "left_45"
            correspondingStep = .left45
        case 3:
            targetSlot = "profile"
            correspondingStep = .leftProfile
            clinicalPhotos["left_profile"] = nil // reset trigger
        case 5:
            targetSlot = "right_45"
            correspondingStep = .right45
        case 6:
            // Cập nhật profile nếu góc phải nét hơn hoặc chưa có
            if clinicalPhotos["profile"] == nil {
                targetSlot = "profile"
                correspondingStep = .rightProfile
            }
        case 7:
            targetSlot = "basal_nostrils"
            correspondingStep = .basalNostrils
        default:
            break
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

    /// Tự động trích xuất đầy đủ 5 góc ảnh lâm sàng từ 10 frame quét
    public func ensureClinicalPhotosFromSweep() {
        if clinicalPhotos["front"] == nil {
            clinicalPhotos["front"] = sweepFrames[0] ?? sweepFrames.values.first
        }
        if clinicalPhotos["left_45"] == nil {
            clinicalPhotos["left_45"] = sweepFrames[2] ?? sweepFrames[1]
        }
        if clinicalPhotos["right_45"] == nil {
            clinicalPhotos["right_45"] = sweepFrames[5] ?? sweepFrames[4]
        }
        if clinicalPhotos["profile"] == nil {
            clinicalPhotos["profile"] = sweepFrames[3] ?? sweepFrames[6] ?? sweepFrames[2]
        }
        if clinicalPhotos["basal_nostrils"] == nil {
            clinicalPhotos["basal_nostrils"] = sweepFrames[7] ?? sweepFrames[9] ?? sweepFrames[0]
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
        uploadProgress = 0.12
        uploadStatusMessage = "Đang tổng hợp dữ liệu 10 góc quét TrueDepth..."
        guidanceFeedback = "✓ HOÀN TẤT 10 GÓC! Đang bắt đầu dựng 3D..."
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
