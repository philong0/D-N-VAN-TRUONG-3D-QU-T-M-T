//
//  ScanModels.swift
//  DrVanTruongScanner (iOS Native TrueDepth Layer)
//

import Foundation
import ARKit
import simd

public enum ScanAngleStep: String, CaseIterable, Identifiable {
    case front = "front"
    case left45 = "left_45"
    case leftProfile = "left_profile"
    case right45 = "right_45"
    case rightProfile = "right_profile"
    
    public var id: String { rawValue }
    
    public var title: String {
        switch self {
        case .front: return "1/5: Chính Diện (0°)"
        case .left45: return "2/5: Nghiêng Trái (~40°)"
        case .leftProfile: return "3/5: Trắc Diện Trái (~55°)"
        case .right45: return "4/5: Nghiêng Phải (~40°)"
        case .rightProfile: return "5/5: Trắc Diện Phải (~55°)"
        }
    }
    
    public var targetYawDeg: Float {
        switch self {
        case .front: return 0.0
        case .left45: return -40.0
        case .leftProfile: return -55.0
        case .right45: return 40.0
        case .rightProfile: return 55.0
        }
    }
    
    public var yawToleranceDeg: Float {
        switch self {
        case .front: return 18.0
        case .left45, .right45: return 18.0
        case .leftProfile, .rightProfile: return 22.0
        }
    }

    /// Comfortable clinical tolerances calibrated for real hand-held TrueDepth scanning.
    public var pitchToleranceDeg: Float { 25.0 }
    public var rollToleranceDeg: Float { 25.0 }
    public var minDistanceMeters: Float { 0.20 }
    public var maxDistanceMeters: Float { 0.70 }
    public var stabilityWindowSeconds: Double { 0.30 }
    public var holdDurationSeconds: Double { 0.35 }
    public var maxYawStandardDeviationDeg: Float { 8.0 }
    public var maxPitchStandardDeviationDeg: Float { 8.0 }
    public var maxRollStandardDeviationDeg: Float { 8.0 }
    public var maxAngularVelocityDegPerSecond: Float { 50.0 }
    public var minimumStabilitySamples: Int { 3 }
    public var captureCooldownSeconds: Double { 0.35 }
    
    public var instruction: String {
        switch self {
        case .front: return "Nhìn thẳng vào camera (0°)"
        case .left45: return "Nghiêng nhẹ mặt sang TRÁI (35°-45°)"
        case .leftProfile: return "Nghiêng sang TRÁI để lộ sống mũi (~55°)"
        case .right45: return "Nghiêng nhẹ mặt sang PHẢI (35°-45°)"
        case .rightProfile: return "Nghiêng sang PHẢI để lộ sống mũi (~55°)"
        }
    }
}

/// Pose expressed in the camera coordinate system for one exact ARFrame.
/// Computed using robust forward / up vector projection without gimbal lock.
public struct CameraRelativeFacePose {
    public let transform: simd_float4x4
    public let yawDeg: Float
    public let pitchDeg: Float
    public let rollDeg: Float
    public let distanceMeters: Float
    public let translationMeters: SIMD3<Float>

    public init(faceToCamera transform: simd_float4x4) {
        self.transform = transform
        
        // Raw head vectors in ARKit camera sensor coordinates
        let fwdX = transform.columns.2.x
        let fwdY = transform.columns.2.y
        let fwdZ = transform.columns.2.z
        
        let upX = transform.columns.1.x
        let upY = transform.columns.1.y
        
        // In iPhone Portrait orientation (device held upright):
        // Screen Right (+X) = Sensor +Y
        // Screen Up (+Y) = Sensor -X
        // Screen Towards User (+Z) = Sensor +Z
        let screenFwdX = fwdY
        let screenFwdY = -fwdX
        let screenFwdZ = fwdZ
        
        let screenUpX = upY
        let screenUpY = -upX
        
        // When facing camera directly: screenFwd ~ [0, 0, 1], screenUp ~ [0, 1] -> Yaw=0°, Pitch=0°, Roll=0°
        let yaw = atan2(screenFwdX, screenFwdZ) * 180.0 / .pi
        let pitch = atan2(screenFwdY, sqrt(screenFwdX * screenFwdX + screenFwdZ * screenFwdZ)) * 180.0 / .pi
        let roll = atan2(screenUpX, screenUpY) * 180.0 / .pi
        
        self.yawDeg = yaw
        self.pitchDeg = pitch
        self.rollDeg = roll
        
        let translation = SIMD3<Float>(transform.columns.3.x, transform.columns.3.y, transform.columns.3.z)
        self.translationMeters = translation
        self.distanceMeters = simd_length(translation)
    }
}

public struct CameraIntrinsicsDTO: Codable {
    public let fx: Float
    public let fy: Float
    public let cx: Float
    public let cy: Float
    public let imageWidth: Int
    public let imageHeight: Int
    public let lensDistortionCoefficients: [Float]?
}

public struct ARKitTransformDTO: Codable {
    /// Column-major flat 16-float 4x4 matrices (`simd_float4x4`'s own
    /// memory layout) — face-local -> world, and camera-local -> world,
    /// kept as two SEPARATE real transforms (see D-poseboth in
    /// ARFaceCaptureSession.swift for why collapsing them into one was a bug).
    public let faceTransformColumnMajor: [Float]
    public let cameraTransformColumnMajor: [Float]
    public let translationMeters: [String: Float]
    public let eulerRotationDeg: [String: Float]
}

public struct ARKitFaceGeometryDTO: Codable {
    public let vertexCount: Int
    public let triangleCount: Int
    public let verticesMeters: [Float]
    public let triangleIndices: [Int]
    public let textureCoordinates: [Float]
    public let blendShapes: [String: Float]?
    public let isTracked: Bool
}

public struct FrameQualityEvaluation: Codable {
    public let blurScore: Float
    public let isBlurry: Bool
    public let lightingScore: Float
    public let isLightingAdequate: Bool
    public let isTracked: Bool
    public let yawDeg: Float
    public let pitchDeg: Float
    public let distanceMeters: Float
    public let isDistanceOptimal: Bool // ScanAngleStep camera-relative distance policy
}

public struct CapturedFramePackage: Identifiable {
    public let id = UUID()
    public let step: ScanAngleStep
    public let timestamp: Double
    public let rgbData: Data
    public let depthData: Data? // Float32 millimeters
    public let depthWidth: Int?
    public let depthHeight: Int?
    /// Intrinsics in the depth-map pixel coordinate system. RGB intrinsics
    /// cannot be reused unscaled for a lower-resolution depth map.
    public let depthIntrinsics: CameraIntrinsicsDTO?
    public let intrinsics: CameraIntrinsicsDTO
    public let pose: ARKitTransformDTO
    public let geometry: ARKitFaceGeometryDTO
    public let quality: FrameQualityEvaluation
}

public struct ScanPackageManifestDTO: Codable {
    public let schemaVersion: String // "2.0.0"
    public let captureSource: String // "native_ios" | "fixture"
    public let deviceModel: String
    public let systemVersion: String
    public let hasTrueDepth: Bool
    public let patientId: String
    public let sessionId: String
    public let capturedAt: String
    public let frames: [FrameEntryDTO]

    /// D-contractfix — field names AND shape here must exactly match what
    /// `src/app/api/patients/[id]/scan-sessions/[sessionId]/package/route.ts`
    /// reads off each `manifest.frames[i]` entry (`frameDTO.view`,
    /// `.rgbFileName`, `.depthFileName`, `.geometry`, `.intrinsics`,
    /// `.pose`, `.timestamp`) — a PRIOR version of this struct only sent
    /// path STRINGS (`geometryRelativePath`/`cameraRelativePath`) with no
    /// actual data anywhere in the upload, and never sent `pose` at all;
    /// the backend silently received `frameDTO.geometry === undefined` for
    /// every real capture, so `patient_native_fusion.py` never saw any real
    /// ARFaceGeometry/depth/intrinsics even from a real TrueDepth device —
    /// confirmed by direct code reading before this fix, not assumed.
    public struct FrameEntryDTO: Codable {
        public let view: String
        public let timestamp: Double
        public let rgbFileName: String
        public let depthFileName: String?
        public let depthWidth: Int?
        public let depthHeight: Int?
        public let depthIntrinsics: CameraIntrinsicsDTO?
        public let isTracked: Bool
        public let yawDeg: Float
        public let pitchDeg: Float
        public let geometry: ARKitFaceGeometryDTO?
        public let intrinsics: CameraIntrinsicsDTO
        public let pose: ARKitTransformDTO
        public let quality: FrameQualityEvaluation
    }
}

public struct QualityMetricDTO: Codable {
    public let status: String
    public let detail: String?
}

public struct ScanQualityReportDTO: Codable {
    public let overall: String
    public let coverage: QualityMetricDTO?
    public let frameQuality: QualityMetricDTO?
    public let trackingQuality: QualityMetricDTO?
    public let poseCoverage: QualityMetricDTO?
    public let regions: [String: QualityMetricDTO]?
    public let evaluatedAt: String?
}

public struct ReconstructionDTO: Codable {
    public let provider: String?
    public let requestedAt: String?
    public let baselineModelFileName: String?
    public let baselineObjFileName: String?
    public let error: String?
}

public struct ScanSessionDTO: Codable {
    public let id: String?
    public let status: String?
    public let quality: ScanQualityReportDTO?
    public let reconstruction: ReconstructionDTO?
}

public struct PackageUploadResponseDTO: Codable {
    public let success: Bool?
    public let message: String?
    public let error: String?
    public let details: String?
    public let studioUrl: String?
    public let session: ScanSessionDTO?
    public let quality: ScanQualityReportDTO?
}
