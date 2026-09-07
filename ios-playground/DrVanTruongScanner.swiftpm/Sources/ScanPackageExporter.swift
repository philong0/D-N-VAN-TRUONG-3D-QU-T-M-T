//
//  ScanPackageExporter.swift
//  DrVanTruong3DStudio (iOS Native TrueDepth Layer)
//
//  Exports versioned scan packages compliant with DrVanTruong3DStudio ScanSession API.
//

import Foundation

public struct VersionedScanManifest: Codable {
    public let packageVersion: String // "2.0.0"
    public let device: DeviceInfo
    public let patientId: String
    public let sessionId: String
    public let captureTimestamp: String
    public let frames: [FrameManifestEntry]
    public let qualitySummary: QualitySummary
    
    public struct DeviceInfo: Codable {
        public let model: String
        public let systemName: String
        public let systemVersion: String
        public let hasTrueDepth: Bool
        public let arkitSupported: Bool
    }
    
    public struct FrameManifestEntry: Codable {
        public let viewTag: String
        public let timestamp: Double
        public let rgbRelativePath: String // "frames/front.jpg"
        public let depthRelativePath: String? // "depth/front_depth.raw"
        public let geometryRelativePath: String? // "geometry/front_geometry.json"
        public let cameraRelativePath: String // "camera/intrinsics.json"
        public let isTracked: Bool
        public let yawDeg: Float
        public let pitchDeg: Float
    }
    
    public struct QualitySummary: Codable {
        public let totalViewsCaptured: Int
        public let trackedViewsCount: Int
        public let depthAvailable: Bool
        public let overallAssessment: String
    }
}

public final class ScanPackageExporter {
    
    public static func createPackageStructure(
        patientId: String,
        sessionId: String,
        baseDirectory: URL
    ) throws -> (
        packageRoot: URL,
        framesDir: URL,
        geometryDir: URL,
        depthDir: URL,
        cameraDir: URL,
        qualityDir: URL
    ) {
        let packageRoot = baseDirectory.appendingPathComponent("ScanPackage_\(sessionId)")
        let framesDir = packageRoot.appendingPathComponent("frames")
        let geometryDir = packageRoot.appendingPathComponent("geometry")
        let depthDir = packageRoot.appendingPathComponent("depth")
        let cameraDir = packageRoot.appendingPathComponent("camera")
        let qualityDir = packageRoot.appendingPathComponent("quality")
        
        let fm = FileManager.default
        try fm.createDirectory(at: framesDir, withIntermediateDirectories: true)
        try fm.createDirectory(at: geometryDir, withIntermediateDirectories: true)
        try fm.createDirectory(at: depthDir, withIntermediateDirectories: true)
        try fm.createDirectory(at: cameraDir, withIntermediateDirectories: true)
        try fm.createDirectory(at: qualityDir, withIntermediateDirectories: true)
        
        return (packageRoot, framesDir, geometryDir, depthDir, cameraDir, qualityDir)
    }
}
