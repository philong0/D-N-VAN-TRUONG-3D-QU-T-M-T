//
//  ARFaceCaptureSession.swift
//  DrVanTruongScanner (iOS Native TrueDepth Layer)
//

import Foundation
import ARKit
import AVFoundation
import UIKit
import Combine

public enum CameraPosition: String {
    case front = "front"
    case back = "back"
}

public final class ARFaceCaptureSession: NSObject, ObservableObject, ARSessionDelegate {
    
    // Observable UI states
    @Published public var isTrueDepthSupported: Bool = false
    @Published public var isCameraAuthorized: Bool = false
    @Published public var isTracking: Bool = false
    @Published public var cameraPosition: CameraPosition = .front
    
    @Published public var currentYawDeg: Float = 0.0
    @Published public var currentPitchDeg: Float = 0.0
    @Published public var currentDistanceMeters: Float = 0.45
    @Published public var currentStep: ScanAngleStep = .front
    @Published public var capturedFrames: [ScanAngleStep: CapturedFramePackage] = [:]
    @Published public var guidanceFeedback: String = "Đang khởi động camera..."
    @Published public var isPoseAligned: Bool = false
    @Published public var holdProgress: Float = 0.0
    @Published public var isUploading: Bool = false
    @Published public var uploadProgress: Float = 0.0
    
    public var patientId: String = ""
    public var sessionId: String = ""
    public var onScanCompleted: ((URL) -> Void)?
    public var onScanCancelled: (() -> Void)?
    
    public let arSession = ARSession()
    private var currentFrame: ARFrame?
    private var currentFaceAnchor: ARFaceAnchor?

    private var alignedSince: Date?
    private var isAutoCapturing = false
    // 2026-09-07 fix — raised from 0.55s: combined with the previously
    // unbounded angle bands (see updateGuidance's own fix note above),
    // 0.55s let a fast, continuous head turn auto-capture a target it was
    // only passing through, not stopping at. Real-device complaint: "quét
    // rất nhanh, tôi không làm được gì" (scans very fast, I couldn't do
    // anything). 0.9s is still fast enough not to feel laggy once the bands
    // are correctly bounded, but long enough that a genuine sweep-through
    // (not a deliberate stop) won't hold long enough to trigger.
    private let holdDurationSeconds: Double = 0.9
    
    public override init() {
        super.init()
        self.isTrueDepthSupported = ARFaceTrackingConfiguration.isSupported
        self.arSession.delegate = self
    }
    
    public func requestCameraPermission(completion: @escaping (Bool) -> Void) {
        let status = AVCaptureDevice.authorizationStatus(for: .video)
        switch status {
        case .authorized:
            self.isCameraAuthorized = true
            completion(true)
        case .notDetermined:
            AVCaptureDevice.requestAccess(for: .video) { granted in
                DispatchQueue.main.async {
                    self.isCameraAuthorized = granted
                    completion(granted)
                }
            }
        default:
            self.isCameraAuthorized = false
            completion(false)
        }
    }
    
    public func startSession() {
        arSession.pause()
        
        if cameraPosition == .front {
            guard isTrueDepthSupported else {
                self.guidanceFeedback = "Thiết bị không hỗ trợ camera TrueDepth trước."
                return
            }
            let config = ARFaceTrackingConfiguration()
            config.isLightEstimationEnabled = true
            config.maximumNumberOfTrackedFaces = 1
            arSession.run(config, options: [.resetTracking, .removeExistingAnchors])
            self.guidanceFeedback = "Camera trước TrueDepth: Nhìn thẳng vào màn hình"
        } else {
            let config = ARWorldTrackingConfiguration()
            if ARWorldTrackingConfiguration.supportsUserFaceTracking {
                config.userFaceTrackingEnabled = true
            }
            if ARWorldTrackingConfiguration.supportsFrameSemantics(.sceneDepth) {
                config.frameSemantics.insert(.sceneDepth)
            }
            arSession.run(config, options: [.resetTracking, .removeExistingAnchors])
            self.guidanceFeedback = "Camera sau: Hướng camera vào khuôn mặt bệnh nhân"
        }
    }
    
    public func switchCamera() {
        cameraPosition = (cameraPosition == .front) ? .back : .front
        alignedSince = nil
        holdProgress = 0.0
        startSession()
    }
    
    public func pauseSession() {
        arSession.pause()
    }
    
    // MARK: - ARSessionDelegate
    
    public func session(_ session: ARSession, didUpdate anchors: [ARAnchor]) {
        for anchor in anchors {
            if let faceAnchor = anchor as? ARFaceAnchor {
                self.currentFaceAnchor = faceAnchor
                DispatchQueue.main.async {
                    self.isTracking = faceAnchor.isTracked
                    let euler = faceAnchor.eulerAngles
                    self.currentPitchDeg = euler.x * 180.0 / .pi
                    self.currentYawDeg = euler.y * 180.0 / .pi
                    self.currentDistanceMeters = abs(faceAnchor.transform.columns.3.z)
                    self.updateGuidance()
                    self.evaluateAutoCapture()
                }
            }
        }
    }
    
    public func session(_ session: ARSession, didUpdate frame: ARFrame) {
        self.currentFrame = frame
    }
    
    private func updateGuidance() {
        guard isTracking else {
            guidanceFeedback = (cameraPosition == .front) ? "Đang tìm khuôn mặt — Hãy nhìn vào màn hình" : "Đang tìm khuôn mặt — Hướng camera vào bệnh nhân"
            isPoseAligned = false
            alignedSince = nil
            holdProgress = 0.0
            return
        }
        
        // 2026-09-07 fix — real-device test found this scanning "very fast,
        // couldn't do anything" (5/5 marked done while the live angle
        // reading was nowhere near the last target). Root cause: this
        // switch hardcoded its OWN separate yaw bands, disagreeing with the
        // real per-step data model (`ScanAngleStep.targetYawDeg`/
        // `yawToleranceDeg` in ScanModels.swift) that already exists and is
        // even shown in the UI's own title/instruction text (e.g. "80°" for
        // leftProfile/rightProfile) — but was never actually used here.
        // Two concrete bugs from that: (1) `leftProfile`/`rightProfile` had
        // NO upper bound at all (`yaw <= -50.0`, `yaw >= 50.0`) — ANY angle
        // past 50° counted as "80°", including 51° or 179°; (2) `left45`'s
        // band (-22..-68) overlapped `leftProfile`'s effectively-unbounded
        // one, so a single continuous head turn could satisfy BOTH targets
        // within the same short sweep. Now driven by the one real model
        // (symmetric tolerance around the actual target), so a fast sweep
        // genuinely cannot satisfy a target it hasn't reached.
        let yaw = currentYawDeg
        let target = currentStep.targetYawDeg
        let tolerance = currentStep.yawToleranceDeg
        // Rear camera mirrors left/right in this project's own convention
        // (same real photography-geometry reasoning already documented in
        // the web app's camera-normalization.ts) — flip the effective
        // target's sign, not the tolerance.
        let effectiveTarget = (cameraPosition == .back) ? -target : target
        let matched = abs(yaw - effectiveTarget) <= tolerance
        let message = matched ? "✓ ĐÚNG GÓC: Giữ yên..." : currentStep.title

        isPoseAligned = matched
        guidanceFeedback = message
    }
    
    // MARK: - Auto-Capture
    private func evaluateAutoCapture() {
        guard capturedFrames.count < ScanAngleStep.allCases.count else {
            alignedSince = nil
            holdProgress = 0.0
            return
        }
        guard isPoseAligned, !isAutoCapturing else {
            alignedSince = nil
            holdProgress = 0.0
            return
        }
        let now = Date()
        guard let since = alignedSince else {
            alignedSince = now
            holdProgress = 0.1
            return
        }
        
        let elapsed = now.timeIntervalSince(since)
        holdProgress = min(1.0, Float(elapsed / holdDurationSeconds))
        
        guard elapsed >= holdDurationSeconds else {
            return
        }

        alignedSince = nil
        holdProgress = 1.0
        isAutoCapturing = true
        
        let generator = UIImpactFeedbackGenerator(style: .heavy)
        generator.prepare()
        generator.impactOccurred()
        
        do {
            try captureCurrentStep()
        } catch {
            isAutoCapturing = false
            guidanceFeedback = "Lỗi tự động chụp — Hãy bấm nút chụp thủ công bên dưới."
        }
    }

    // MARK: - Capture Action

    public func captureCurrentStep() throws {
        guard let frame = currentFrame else {
            throw NSError(domain: "Scanner", code: 404, userInfo: [NSLocalizedDescriptionKey: "Chưa có dữ liệu camera frame."])
        }
        
        let pixelBuffer = frame.capturedImage
        let quality = QualityEvaluator.evaluate(
            pixelBuffer: pixelBuffer,
            faceAnchor: currentFaceAnchor,
            targetStep: currentStep
        )
        
        // 1. Convert PixelBuffer to JPEG Data
        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
        let context = CIContext()
        guard let cgImage = context.createCGImage(ciImage, from: ciImage.extent) else {
            throw NSError(domain: "Scanner", code: 500, userInfo: [NSLocalizedDescriptionKey: "Lỗi xử lý ảnh RGB."])
        }
        let uiImage = UIImage(cgImage: cgImage)
        guard let rgbData = uiImage.jpegData(compressionQuality: 0.95) else {
            throw NSError(domain: "Scanner", code: 500, userInfo: [NSLocalizedDescriptionKey: "Lỗi nén JPEG."])
        }
        
        // 2. Extract Depth Buffer (from front TrueDepth AVDepthData if provided by ARKit/AVFoundation)
        var depthData: Data? = nil
        var depthW: Int? = nil
        var depthH: Int? = nil
        
        if let capturedDepth = frame.capturedDepthData,
           let processed = DepthDataProcessor.processDepthData(capturedDepth) {
            depthData = processed.rawData
            depthW = processed.width
            depthH = processed.height
        }
        
        // 3. Camera Intrinsics (Exact Apple Hardware Intrinsics)
        let intr = frame.camera.intrinsics
        let res = frame.camera.imageResolution
        let intrinsicsDTO = CameraIntrinsicsDTO(
            fx: intr[0, 0],
            fy: intr[1, 1],
            cx: intr[2, 0],
            cy: intr[2, 1],
            imageWidth: Int(res.width),
            imageHeight: Int(res.height),
            lensDistortionCoefficients: nil
        )
        
        // 4. Pose — BOTH real ARKit transforms, kept separate (D-poseboth):
        // `faceAnchor.transform` (face-local -> world) and
        // `frame.camera.transform` (camera-local -> world) are DIFFERENT
        // real quantities. A prior version of this capture only ever sent
        // ONE of them (`currentFaceAnchor?.transform ?? frame.camera.transform`)
        // under a single generic "pose" field — the reconstruction server
        // needs BOTH to compute the face's real pose relative to the camera
        // for THIS exact frame (faceToCamera = inverse(cameraToWorld) *
        // faceToWorld, same convention documented in this project's own
        // ios-app/ARFaceCaptureController.swift scaffold) — collapsing them
        // into one silently discards real data needed for correct
        // multi-frame registration.
        func flattenColumnMajor(_ m: simd_float4x4) -> [Float] {
            [
                m.columns.0.x, m.columns.0.y, m.columns.0.z, m.columns.0.w,
                m.columns.1.x, m.columns.1.y, m.columns.1.z, m.columns.1.w,
                m.columns.2.x, m.columns.2.y, m.columns.2.z, m.columns.2.w,
                m.columns.3.x, m.columns.3.y, m.columns.3.z, m.columns.3.w,
            ]
        }
        let faceTransform = currentFaceAnchor?.transform ?? matrix_identity_float4x4
        let cameraTransform = frame.camera.transform
        let poseDTO = ARKitTransformDTO(
            faceTransformColumnMajor: flattenColumnMajor(faceTransform),
            cameraTransformColumnMajor: flattenColumnMajor(cameraTransform),
            translationMeters: ["x": faceTransform.columns.3.x, "y": faceTransform.columns.3.y, "z": faceTransform.columns.3.z],
            eulerRotationDeg: ["pitch": currentPitchDeg, "yaw": currentYawDeg, "roll": 0.0]
        )
        
        // 5. ARFaceGeometry
        var verticesFlat: [Float] = []
        var trianglesFlat: [Int] = []
        var uvsFlat: [Float] = []
        var blendShapesMap: [String: Float] = [:]
        
        if let geometry = currentFaceAnchor?.geometry {
            for v in geometry.vertices {
                verticesFlat.append(v.x)
                verticesFlat.append(v.y)
                verticesFlat.append(v.z)
            }
            for t in geometry.triangleIndices {
                trianglesFlat.append(Int(t))
            }
            for uv in geometry.textureCoordinates {
                uvsFlat.append(uv.x)
                uvsFlat.append(uv.y)
            }
        }
        if let blendShapes = currentFaceAnchor?.blendShapes {
            for (k, v) in blendShapes {
                blendShapesMap[k.rawValue] = v.floatValue
            }
        }
        
        let geometryDTO = ARKitFaceGeometryDTO(
            vertexCount: verticesFlat.count / 3,
            triangleCount: trianglesFlat.count / 3,
            verticesMeters: verticesFlat,
            triangleIndices: trianglesFlat,
            textureCoordinates: uvsFlat,
            blendShapes: blendShapesMap,
            isTracked: isTracking
        )
        
        let package = CapturedFramePackage(
            step: currentStep,
            timestamp: frame.timestamp,
            rgbData: rgbData,
            depthData: depthData,
            depthWidth: depthW,
            depthHeight: depthH,
            intrinsics: intrinsicsDTO,
            pose: poseDTO,
            geometry: geometryDTO,
            quality: quality
        )
        
        DispatchQueue.main.async {
            self.capturedFrames[self.currentStep] = package
            self.advanceToNextStep()
            self.isAutoCapturing = false
            self.holdProgress = 0.0
            
            if self.capturedFrames.count >= ScanAngleStep.allCases.count {
                self.triggerPackageUpload { _ in }
            }
        }
    }
    
    private func advanceToNextStep() {
        let allSteps = ScanAngleStep.allCases
        guard let currentIndex = allSteps.firstIndex(of: currentStep) else { return }
        
        let nextIndex = currentIndex + 1
        if nextIndex < allSteps.count {
            self.currentStep = allSteps[nextIndex]
        }
    }
    
    public func retakePreviousStep() {
        let allSteps = ScanAngleStep.allCases
        guard let currentIndex = allSteps.firstIndex(of: currentStep), currentIndex > 0 else {
            return
        }
        let prevStep = allSteps[currentIndex - 1]
        capturedFrames.removeValue(forKey: prevStep)
        currentStep = prevStep
        alignedSince = nil
        holdProgress = 0.0
        isAutoCapturing = false
        guidanceFeedback = "Đang chụp lại góc: \(prevStep.title)"
    }
    
    public func triggerPackageUpload(completion: @escaping (Result<URL, Error>) -> Void) {
        guard !patientId.isEmpty, !sessionId.isEmpty else {
            completion(.failure(NSError(domain: "Scanner", code: 400, userInfo: [NSLocalizedDescriptionKey: "Thiếu PatientID hoặc SessionID."])))
            return
        }
        
        self.isUploading = true
        self.guidanceFeedback = "✓ Đang tải gói TrueDepth lên máy chủ & Dựng 3D..."
        
        let client = BackendAPIClient()
        client.uploadScanPackage(patientId: patientId, sessionId: sessionId, frames: capturedFrames) { [weak self] result in
            DispatchQueue.main.async {
                self?.isUploading = false
                switch result {
                case .success(let studioURL):
                    self?.guidanceFeedback = "✓ Tải lên thành công! Đang chuyển vào 3D Studio..."
                    self?.onScanCompleted?(studioURL)
                    completion(.success(studioURL))
                case .failure(let error):
                    self?.guidanceFeedback = "Lỗi tải lên máy chủ: \(error.localizedDescription)"
                    completion(.failure(error))
                }
            }
        }
    }
    
    public func resetScan() {
        self.capturedFrames.removeAll()
        self.currentStep = .front
        self.guidanceFeedback = "Đã sẵn sàng quét lại."
        self.alignedSince = nil
        self.holdProgress = 0.0
        self.isAutoCapturing = false
    }
}

extension ARFaceAnchor {
    public var eulerAngles: simd_float3 {
        let m = self.transform
        let pitch = asin(-m.columns.2.y)
        let yaw = atan2(m.columns.2.x, m.columns.2.z)
        let roll = atan2(m.columns.0.y, m.columns.1.y)
        return simd_float3(pitch, yaw, roll)
    }
}

