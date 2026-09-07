//
//  ScanModels.swift
//  DrVanTruongScanner (iOS Native TrueDepth Layer)
//

import Foundation
import ARKit

public enum ScanAngleStep: String, CaseIterable, Identifiable {
    case front = "front"
    case left45 = "left_45"
    case leftProfile = "left_profile"
    case right45 = "right_45"
    case rightProfile = "right_profile"
    
    public var id: String { rawValue }
    
    public var title: String {
        switch self {
        case .front: return "Chính Diện 0°"
        case .left45: return "Nghiêng Trái 45°"
        case .leftProfile: return "Góc Nghiêng Trái 60°-75°"
        case .right45: return "Nghiêng Phải 45°"
        case .rightProfile: return "Góc Nghiêng Phải 60°-75°"
        }
    }
    
    public var targetYawDeg: Float {
        switch self {
        case .front: return 0.0
        case .left45: return -30.0
        case .leftProfile: return -50.0
        case .right45: return 30.0
        case .rightProfile: return 50.0
        }
    }
    
    public var yawToleranceDeg: Float {
        switch self {
        case .front: return 20.0
        case .left45: return 25.0
        case .leftProfile: return 28.0
        case .right45: return 25.0
        case .rightProfile: return 28.0
        }
    }
    
    public var instruction: String {
        switch self {
        case .front: return "Nhìn thẳng vào camera, giữ nét mặt tự nhiên"
        case .left45: return "Từ từ quay đầu sang bên phải của bạn (máy nghiêng trái)"
        case .leftProfile: return "Nghiêng sâu hơn sang phải để lấy đường viền sống mũi"
        case .right45: return "Từ từ quay đầu sang bên trái của bạn (máy nghiêng phải)"
        case .rightProfile: return "Nghiêng sâu hơn sang trái để lấy đường viền đối diện"
        }
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
    public let isDistanceOptimal: Bool // 30cm - 60cm
}

public struct CapturedFramePackage: Identifiable {
    public let id = UUID()
    public let step: ScanAngleStep
    public let timestamp: Double
    public let rgbData: Data
    public let depthData: Data? // Float32 millimeters
    public let depthWidth: Int?
    public let depthHeight: Int?
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
        public let isTracked: Bool
        public let yawDeg: Float
        public let pitchDeg: Float
        public let geometry: ARKitFaceGeometryDTO?
        public let intrinsics: CameraIntrinsicsDTO
        public let pose: ARKitTransformDTO
    }
}
