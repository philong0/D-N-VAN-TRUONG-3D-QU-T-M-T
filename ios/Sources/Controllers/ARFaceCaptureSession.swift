//
//  ARFaceCaptureSession.swift
//  DrVanTruongScanner (iOS Native TrueDepth Layer)
//

import Foundation
import ARKit
import AVFoundation
import UIKit
import Combine

public final class ARFaceCaptureSession: NSObject, ObservableObject, ARSessionDelegate {
    
    // Observable UI states
    @Published public var isTrueDepthSupported: Bool = false
    @Published public var isCameraAuthorized: Bool = false
    @Published public var isTracking: Bool = false
    @Published public var currentYawDeg: Float = 0.0
    @Published public var currentPitchDeg: Float = 0.0
    @Published public var currentDistanceMeters: Float = 0.45
    @Published public var currentStep: ScanAngleStep = .front
    @Published public var capturedFrames: [ScanAngleStep: CapturedFramePackage] = [:]
    @Published public var guidanceFeedback: String = "Đang khởi động camera..."
    @Published public var isPoseAligned: Bool = false
    
    public let arSession = ARSession()
    private var currentFrame: ARFrame?
    private var currentFaceAnchor: ARFaceAnchor?

    // D-autocapture — dwell timer + re-entrancy guard for automatic
    // capture (see evaluateAutoCapture below).
    private var alignedSince: Date?
    private var isAutoCapturing = false
    
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
        guard isTrueDepthSupported else {
            self.guidanceFeedback = "Thiết bị không có camera TrueDepth (Cần iPhone X trở lên hoặc iPad Pro)."
            return
        }
        
        let config = ARFaceTrackingConfiguration()
        config.isLightEstimationEnabled = true
        config.maximumNumberOfTrackedFaces = 1
        
        arSession.run(config, options: [.resetTracking, .removeExistingAnchors])
        self.guidanceFeedback = "Đang tìm khuôn mặt..."
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
            guidanceFeedback = "Đang tìm khuôn mặt — Hãy nhìn vào màn hình"
            isPoseAligned = false
            alignedSince = nil
            return
        }
        
        let yaw = currentYawDeg
        var matched = false
        var message = ""
        
        switch currentStep {
        case .front:
            matched = abs(yaw) <= 15.0
            message = matched ? "✓ ĐÚNG GÓC: Giữ yên nhìn thẳng" : "Nhìn thẳng vào camera (0°)"
        case .left45:
            matched = yaw <= -25.0 && yaw >= -65.0
            message = matched ? "✓ ĐÚNG GÓC: Giữ yên nghiêng trái" : "Từ từ quay mặt sang Trái 45°"
        case .leftProfile:
            matched = yaw <= -55.0
            message = matched ? "✓ ĐÚNG GÓC: Giữ yên trắc diện trái" : "Quay ngang hẳn sang Trái (70°-90°)"
        case .right45:
            matched = yaw >= 25.0 && yaw <= 65.0
            message = matched ? "✓ ĐÚNG GÓC: Giữ yên nghiêng phải" : "Từ từ quay mặt sang Phải 45°"
        case .rightProfile:
            matched = yaw >= 55.0
            message = matched ? "✓ ĐÚNG GÓC: Giữ yên trắc diện phải" : "Quay ngang hẳn sang Phải (70°-90°)"
        }
        
        isPoseAligned = matched
        guidanceFeedback = message
    }
    
    // MARK: - Auto-Capture
    private func evaluateAutoCapture() {
        guard capturedFrames.count < ScanAngleStep.allCases.count else {
            alignedSince = nil
            return
        }
        guard isPoseAligned, !isAutoCapturing else {
            alignedSince = nil
            return
        }
        let now = Date()
        guard let since = alignedSince else {
            alignedSince = now
            return
        }
        
        // Cần giữ yên đúng góc 0.6 giây để chống rung nhòe và tránh chụp liên tiếp
        guard now.timeIntervalSince(since) >= 0.6 else {
            return
        }

        alignedSince = nil
        isAutoCapturing = true
        UIImpactFeedbackGenerator(style: .heavy).impactOccurred()
        do {
            try captureCurrentStep()
        } catch {
            isAutoCapturing = false
            guidanceFeedback = "Lỗi tự động chụp — hãy bấm nút chụp thủ công bên dưới."
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
    
    public func resetScan() {
        self.capturedFrames.removeAll()
        self.currentStep = .front
        self.guidanceFeedback = "Đã sẵn sàng quét lại."
        self.alignedSince = nil
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

