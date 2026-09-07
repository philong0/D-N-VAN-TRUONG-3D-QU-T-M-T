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
        case .front: return "1/5: Chính Diện (0°)"
        case .left45: return "2/5: Nghiêng Trái (45°)"
        case .leftProfile: return "3/5: Trắc Diện Trái (80°)"
        case .right45: return "4/5: Nghiêng Phải (45°)"
        case .rightProfile: return "5/5: Trắc Diện Phải (80°)"
        }
    }
    
    public var targetYawDeg: Float {
        switch self {
        case .front: return 0.0
        case .left45: return -45.0
        case .leftProfile: return -80.0
        case .right45: return 45.0
        case .rightProfile: return 80.0
        }
    }
    
    public var yawToleranceDeg: Float {
        switch self {
        case .front: return 10.0
        case .left45: return 10.0
        case .leftProfile: return 10.0
        case .right45: return 10.0
        case .rightProfile: return 10.0
        }
    }

    /// A reproducible acceptance specification used by both the live HUD and
    /// the capture gate.  A labelled view is never accepted merely because a
    /// frame happened to arrive while the patient was turning their head.
    public var pitchToleranceDeg: Float { 10.0 }
    public var minDistanceMeters: Float { 0.28 }
    public var maxDistanceMeters: Float { 0.55 }
    
    public var instruction: String {
        switch self {
        case .front: return "Nhìn thẳng trực tiếp vào camera (0°)"
        case .left45: return "Từ từ xoay mặt sang TRÁI một góc 45°"
        case .leftProfile: return "Quay hẳn mặt sang TRÁI (góc ngang 80°)"
        case .right45: return "Từ từ xoay mặt sang PHẢI một góc 45°"
        case .rightProfile: return "Quay hẳn mặt sang PHẢI (góc ngang 80°)"
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
