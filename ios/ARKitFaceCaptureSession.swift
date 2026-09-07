//
//  ARKitFaceCaptureSession.swift
//  DrVanTruong3DStudio (iOS Native TrueDepth Layer)
//
//  Production Swift layer for capturing high-fidelity multi-angle face scans
//  using Apple TrueDepth Camera and ARKit ARFaceTrackingConfiguration.
//

import Foundation
import ARKit
import AVFoundation
import UIKit

public struct CameraIntrinsicsDTO: Codable {
    public let fx: Float
    public let fy: Float
    public let cx: Float
    public let cy: Float
    public let imageWidth: Int
    public let imageHeight: Int
}

public struct ARKitTransformDTO: Codable {
    public let columns: [[Float]]
    public let translationMeters: [String: Float]
    public let eulerRotationDeg: [String: Float]
}

public struct ARKitFaceGeometryDTO: Codable {
    public let vertexCount: Int
    public let triangleCount: Int
    public let verticesMeters: [Float]
    public let triangleIndices: [Int]
    public let textureCoordinates: [Float]
    public let blendShapes: [String: Float]
}

public struct FrameCaptureDTO: Codable {
    public let view: String
    public let timestamp: Double
    public let rgbFileName: String
    public let depthFileName: String?
    public let depthWidth: Int?
    public let depthHeight: Int?
    public let intrinsics: CameraIntrinsicsDTO
    public let pose: ARKitTransformDTO
    public let geometry: ARKitFaceGeometryDTO
}

public struct CapturePackageManifestDTO: Codable {
    public let schemaVersion: String
    public let deviceModel: String
    public let systemVersion: String
    public let hasTrueDepth: Bool
    public let hasLiDAR: Bool
    public let patientId: String
    public let sessionId: String
    public let capturedAt: String
    public var frames: [FrameCaptureDTO]
}

public protocol ARKitFaceCaptureDelegate: AnyObject {
    func didUpdateTrackingState(isTracking: Bool, message: String)
    func didCaptureFrame(viewTag: String, progress: Float)
    func didCompletePackageExport(packageZipURL: URL)
    func didFailWithError(error: Error)
}

public final class ARKitFaceCaptureSession: NSObject, ARSessionDelegate {
    
    public weak var delegate: ARKitFaceCaptureDelegate?
    private let arSession = ARSession()
    private var currentFaceAnchor: ARFaceAnchor?
    private var currentFrame: ARFrame?
    
    private let patientId: String
    private let sessionId: String
    private var capturedFrames: [String: (rgb: Data, depth: Data?, dto: FrameCaptureDTO)] = [:]
    
    public init(patientId: String, sessionId: String) {
        self.patientId = patientId
        self.sessionId = sessionId
        super.init()
        self.arSession.delegate = self
    }
    
    public static var isSupported: Bool {
        return ARFaceTrackingConfiguration.isSupported
    }
    
    public func start() {
        guard ARFaceTrackingConfiguration.isSupported else {
            delegate?.didFailWithError(error: NSError(domain: "DrVanTruong", code: 400, userInfo: [NSLocalizedDescriptionKey: "Thiết bị không hỗ trợ TrueDepth / ARFaceTracking."]))
            return
        }
        
        let configuration = ARFaceTrackingConfiguration()
        configuration.isLightEstimationEnabled = true
        configuration.maximumNumberOfTrackedFaces = 1
        
        if #available(iOS 13.0, *) {
            // Enable high resolution capture if available
            configuration.videoFormat = ARFaceTrackingConfiguration.supportedVideoFormats.first ?? configuration.videoFormat
        }
        
        arSession.run(configuration, options: [.resetTracking, .removeExistingAnchors])
    }
    
    public func stop() {
        arSession.pause()
    }
    
    // MARK: - ARSessionDelegate
    
    public func session(_ session: ARSession, didUpdate anchors: [ARAnchor]) {
        for anchor in anchors {
            if let faceAnchor = anchor as? ARFaceAnchor {
                self.currentFaceAnchor = faceAnchor
                let isTracked = faceAnchor.isTracked
                let msg = isTracked ? "Đang khóa khuôn mặt" : "Mất dấu khuôn mặt - Hãy nhìn vào camera"
                delegate?.didUpdateTrackingState(isTracking: isTracked, message: msg)
            }
        }
    }
    
    public func session(_ session: ARSession, didUpdate frame: ARFrame) {
        self.currentFrame = frame
    }
    
    // MARK: - Capture Angle Frame
    
    public func captureView(viewTag: String) throws {
        guard let frame = currentFrame, let faceAnchor = currentFaceAnchor, faceAnchor.isTracked else {
            throw NSError(domain: "DrVanTruong", code: 404, userInfo: [NSLocalizedDescriptionKey: "Chưa nhận diện được khuôn mặt ổn định."])
        }
        
        // 1. Extract RGB Image from PixelBuffer
        let pixelBuffer = frame.capturedImage
        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
        let context = CIContext()
        guard let cgImage = context.createCGImage(ciImage, from: ciImage.extent) else {
            throw NSError(domain: "DrVanTruong", code: 500, userInfo: [NSLocalizedDescriptionKey: "Không thể trích xuất ảnh màu RGB."])
        }
        let uiImage = UIImage(cgImage: cgImage)
        guard let rgbData = uiImage.jpegData(compressionQuality: 0.95) else {
            throw NSError(domain: "DrVanTruong", code: 500, userInfo: [NSLocalizedDescriptionKey: "Không thể nén ảnh JPEG."])
        }
        
        // 2. Extract Depth Map Data (AVDepthData / sceneDepth / capturedDepthData)
        var depthData: Data? = nil
        var depthW: Int? = nil
        var depthH: Int? = nil
        
        if #available(iOS 14.0, *), let sceneDepth = frame.sceneDepth {
            let depthBuffer = sceneDepth.depthMap
            depthW = CVPixelBufferGetWidth(depthBuffer)
            depthH = CVPixelBufferGetHeight(depthBuffer)
            
            CVPixelBufferLockBaseAddress(depthBuffer, .readOnly)
            if let baseAddress = CVPixelBufferGetBaseAddress(depthBuffer) {
                let byteCount = CVPixelBufferGetDataSize(depthBuffer)
                depthData = Data(bytes: baseAddress, count: byteCount)
            }
            CVPixelBufferUnlockBaseAddress(depthBuffer, .readOnly)
        }
        
        // 3. Extract Camera Intrinsics
        let intrinsics = frame.camera.intrinsics
        let imageSize = frame.camera.imageResolution
        let intrinsicsDTO = CameraIntrinsicsDTO(
            fx: intrinsics[0, 0],
            fy: intrinsics[1, 1],
            cx: intrinsics[2, 0],
            cy: intrinsics[2, 1],
            imageWidth: Int(imageSize.width),
            imageHeight: Int(imageSize.height)
        )
        
        // 4. Extract ARFaceGeometry (1220 vertices, 2304 triangles, texture UVs)
        let geometry = faceAnchor.geometry
        var verticesFlat: [Float] = []
        verticesFlat.reserveCapacity(geometry.vertices.count * 3)
        for v in geometry.vertices {
            verticesFlat.append(v.x)
            verticesFlat.append(v.y)
            verticesFlat.append(v.z)
        }
        
        var trianglesFlat: [Int] = []
        trianglesFlat.reserveCapacity(geometry.triangleIndices.count)
        for idx in geometry.triangleIndices {
            trianglesFlat.append(Int(idx))
        }
        
        var uvsFlat: [Float] = []
        uvsFlat.reserveCapacity(geometry.textureCoordinates.count * 2)
        for uv in geometry.textureCoordinates {
            uvsFlat.append(uv.x)
            uvsFlat.append(uv.y)
        }
        
        var blendShapesMap: [String: Float] = [:]
        for (key, val) in faceAnchor.blendShapes {
            blendShapesMap[key.rawValue] = val.floatValue
        }
        
        let geometryDTO = ARKitFaceGeometryDTO(
            vertexCount: geometry.vertices.count,
            triangleCount: geometry.triangleCount,
            verticesMeters: verticesFlat,
            triangleIndices: trianglesFlat,
            textureCoordinates: uvsFlat,
            blendShapes: blendShapesMap
        )
        
        // 5. Extract Pose Matrix (4x4)
        let t = faceAnchor.transform
        let matrixCols: [[Float]] = [
            [t.columns.0.x, t.columns.0.y, t.columns.0.z, t.columns.0.w],
            [t.columns.1.x, t.columns.1.y, t.columns.1.z, t.columns.1.w],
            [t.columns.2.x, t.columns.2.y, t.columns.2.z, t.columns.2.w],
            [t.columns.3.x, t.columns.3.y, t.columns.3.z, t.columns.3.w],
        ]
        
        let translationMeters: [String: Float] = [
            "x": t.columns.3.x,
            "y": t.columns.3.y,
            "z": t.columns.3.z
        ]
        
        let eulerAngles = faceAnchor.eulerAngles
        let eulerRotationDeg: [String: Float] = [
            "pitch": eulerAngles.x * 180.0 / .pi,
            "yaw": eulerAngles.y * 180.0 / .pi,
            "roll": eulerAngles.z * 180.0 / .pi
        ]
        
        let poseDTO = ARKitTransformDTO(
            columns: matrixCols,
            translationMeters: translationMeters,
            eulerRotationDeg: eulerRotationDeg
        )
        
        let rgbFileName = "\(viewTag)_rgb.jpg"
        let depthFileName = depthData != nil ? "\(viewTag)_depth.raw" : nil
        
        let frameDTO = FrameCaptureDTO(
            view: viewTag,
            timestamp: frame.timestamp,
            rgbFileName: rgbFileName,
            depthFileName: depthFileName,
            depthWidth: depthW,
            depthHeight: depthH,
            intrinsics: intrinsicsDTO,
            pose: poseDTO,
            geometry: geometryDTO
        )
        
        capturedFrames[viewTag] = (rgb: rgbData, depth: depthData, dto: frameDTO)
        
        let progress = Float(capturedFrames.count) / 5.0
        delegate?.didCaptureFrame(viewTag: viewTag, progress: progress)
    }
    
    // MARK: - Export Native TrueDepth Package
    
    public func exportPackage() throws -> URL {
        let tempDir = FileManager.default.temporaryDirectory.appendingPathComponent("TrueDepthCapture_\(sessionId)")
        try? FileManager.default.removeItem(at: tempDir)
        try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true, attributes: nil)
        
        var frameDTOs: [FrameCaptureDTO] = []
        
        for (_, item) in capturedFrames {
            let rgbURL = tempDir.appendingPathComponent(item.dto.rgbFileName)
            try item.rgb.write(to: rgbURL)
            
            if let depthData = item.depth, let depthName = item.dto.depthFileName {
                let depthURL = tempDir.appendingPathComponent(depthName)
                try depthData.write(to: depthURL)
            }
            
            frameDTOs.append(item.dto)
        }
        
        let manifest = CapturePackageManifestDTO(
            schemaVersion: "1.0.0",
            deviceModel: UIDevice.current.model,
            systemVersion: UIDevice.current.systemVersion,
            hasTrueDepth: true,
            hasLiDAR: true,
            patientId: patientId,
            sessionId: sessionId,
            capturedAt: ISO8601DateFormatter().string(from: Date()),
            frames: frameDTOs
        )
        
        let manifestURL = tempDir.appendingPathComponent("manifest.json")
        let encoder = JSONEncoder()
        encoder.outputFormatting = .prettyPrinted
        let manifestData = try encoder.encode(manifest)
        try manifestData.write(to: manifestURL)
        
        delegate?.didCompletePackageExport(packageZipURL: tempDir)
        return tempDir
    }
}
