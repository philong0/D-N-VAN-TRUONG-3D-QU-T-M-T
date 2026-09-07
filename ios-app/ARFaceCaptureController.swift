import ARKit
import UIKit

/// Wraps a single `ARSession` running `ARFaceTrackingConfiguration` (the
/// TrueDepth face-tracking mode) and exposes exactly two operations to the
/// rest of the app: `start()` and `captureCurrentFrame()`. No SceneKit/
/// RealityKit rendering happens here — this is a pure data-capture session,
/// the actual camera preview the user sees is the web page's own `<video>`
/// element rendering the SAME device camera via `getUserMedia` underneath
/// the WKWebView (see `FaceScannerView.swift` for why running both a WebView
/// camera preview AND an AR session concurrently is fine on iOS: they are
/// independent capture sessions against the same hardware, not a conflict —
/// verified against Apple's documented behavior for `AVCaptureSession` +
/// `ARSession` co-existing, not verified on a physical device here).
final class ARFaceCaptureController: NSObject, ARSessionDelegate {
    private let session = ARSession()
    private let ciContext = CIContext()

    /// Guarded by `queue` — ARSessionDelegate callbacks land on an
    /// ARKit-internal thread, `captureCurrentFrame` can be called from the
    /// WKScriptMessageHandler's main-thread callback, so this needs a lock
    /// rather than assuming main-thread-only access.
    private let queue = DispatchQueue(label: "arface.capture.state")
    private var latestFrame: ARFrame?
    private var latestFaceAnchor: ARFaceAnchor?

    private(set) var isRunning = false

    /// TrueDepth face tracking needs the front camera's depth sensor —
    /// absent on any device without Face ID hardware (all iPads/iPhones
    /// without a notch/Dynamic Island TrueDepth camera). Checked before ever
    /// starting a session so the web page gets a clean "unavailable" instead
    /// of a crash or a silent black frame.
    static var isSupported: Bool {
        ARFaceTrackingConfiguration.isSupported
    }

    func start() throws {
        guard ARFaceTrackingConfiguration.isSupported else {
            throw CaptureError.unsupportedDevice
        }
        let configuration = ARFaceTrackingConfiguration()
        // Real per-vertex face geometry (not just the 52 ARKit blend-shape
        // coefficients) is exactly what this whole feature exists to get —
        // `.geometry` on `ARFaceAnchor` is populated automatically once face
        // tracking starts, this flag isn't needed for that; kept absent
        // intentionally (world tracking / plane detection are irrelevant
        // here and would only add battery/CPU cost).
        session.delegate = self
        session.run(configuration, options: [.resetTracking, .removeExistingAnchors])
        isRunning = true
    }

    func stop() {
        session.pause()
        isRunning = false
        queue.sync {
            latestFrame = nil
            latestFaceAnchor = nil
        }
    }

    // MARK: - ARSessionDelegate

    func session(_ session: ARSession, didUpdate frame: ARFrame) {
        queue.sync {
            latestFrame = frame
            latestFaceAnchor = frame.anchors.compactMap { $0 as? ARFaceAnchor }.first
        }
    }

    func session(_ session: ARSession, didFailWithError error: Error) {
        NSLog("[ARFaceCaptureController] session failed: \(error.localizedDescription)")
        isRunning = false
    }

    // MARK: - Capture

    enum CaptureError: LocalizedError {
        case unsupportedDevice
        case noFaceDetected
        case imageEncodeFailed

        var errorDescription: String? {
            switch self {
            case .unsupportedDevice:
                return "Thiết bị này không có cảm biến TrueDepth (cần iPhone/iPad có Face ID)."
            case .noFaceDetected:
                return "Chưa nhận diện được khuôn mặt trong khung hình. Hãy giữ mặt trong khung hướng dẫn."
            case .imageEncodeFailed:
                return "Không mã hoá được khung hình camera."
            }
        }
    }

    /// One real capture: the face mesh ARKit is tracking RIGHT NOW (already
    /// fitted to the real TrueDepth depth data, not a photo-inferred guess),
    /// the real camera intrinsics for the frame the mesh was seen in, the
    /// face's real 3D pose relative to that camera, and a JPEG of the same
    /// frame to bake real photo color onto that real mesh server-side.
    func captureCurrentFrame() throws -> ARKitCapturePayload {
        let (frame, faceAnchor): (ARFrame?, ARFaceAnchor?) = queue.sync { (latestFrame, latestFaceAnchor) }
        guard let frame, let faceAnchor else {
            throw CaptureError.noFaceDetected
        }
        let geometry = faceAnchor.geometry

        // faceAnchor.transform: face-local -> world (ARKit session origin).
        // frame.camera.transform: camera-local -> world.
        // faceToCamera = inverse(cameraToWorld) * faceToWorld — expresses
        // every face-local vertex directly in this frame's own camera
        // space, decoupled from ARKit's world-tracking drift/origin (each
        // capture only ever needs ITS OWN camera's relative pose, never a
        // cross-capture world alignment). Column-major, matching simd's own
        // in-memory layout — ai-engine/arkit_reconstruction.py must parse
        // it the same way (documented there).
        let cameraToWorld = frame.camera.transform
        let faceToWorld = faceAnchor.transform
        let faceToCamera = cameraToWorld.inverse * faceToWorld

        let vertices: [Float] = geometry.vertices.flatMap { [$0.x, $0.y, $0.z] }
        let textureCoordinates: [Float] = geometry.textureCoordinates.flatMap { [$0.x, $0.y] }
        // ARFaceGeometry.triangleIndices is [Int16]; JSON doesn't care about
        // integer width, widen to Int for a plain JSON number array.
        let triangleIndices: [Int] = geometry.triangleIndices.map { Int($0) }

        let intrinsics = frame.camera.intrinsics // simd_float3x3, pixel units, for frame.camera.imageResolution
        let resolution = frame.camera.imageResolution

        guard let jpegData = Self.jpegData(from: frame.capturedImage, using: ciContext) else {
            throw CaptureError.imageEncodeFailed
        }

        return ARKitCapturePayload(
            vertexCount: geometry.vertices.count,
            triangleCount: geometry.triangleIndices.count / 3,
            vertices: vertices,
            triangleIndices: triangleIndices,
            textureCoordinates: textureCoordinates,
            faceToCameraColumnMajor: faceToCamera.columnMajorArray,
            intrinsics: CameraIntrinsicsPayload(
                fx: intrinsics.columns.0.x,
                fy: intrinsics.columns.1.y,
                cx: intrinsics.columns.2.x,
                cy: intrinsics.columns.2.y,
                imageWidth: Int(resolution.width),
                imageHeight: Int(resolution.height)
            ),
            jpegData: jpegData,
            capturedAt: Date()
        )
    }

    /// `ARFrame.capturedImage` is the raw landscape sensor buffer. Keep that
    /// orientation in the uploaded JPEG: its pixel coordinates then remain
    /// exactly the same coordinate system as `frame.camera.intrinsics`.
    /// Rotating only the JPEG for a portrait preview without rotating K and
    /// the projection pose made oblique/profile texture projections warp.
    /// A UI preview may rotate this image separately, but reconstruction
    /// must receive the unmirrored sensor image.
    private static func jpegData(from pixelBuffer: CVPixelBuffer, using context: CIContext) -> Data? {
        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
        guard let cgImage = context.createCGImage(ciImage, from: ciImage.extent) else { return nil }
        let uiImage = UIImage(cgImage: cgImage)
        return uiImage.jpegData(compressionQuality: 0.9)
    }
}

struct CameraIntrinsicsPayload {
    let fx: Float
    let fy: Float
    let cx: Float
    let cy: Float
    let imageWidth: Int
    let imageHeight: Int
}

struct ARKitCapturePayload {
    let vertexCount: Int
    let triangleCount: Int
    let vertices: [Float]
    let triangleIndices: [Int]
    let textureCoordinates: [Float]
    /// 16 floats, column-major (simd_float4x4's own memory layout):
    /// element [4*col + row]. Maps a face-local point to this capture's own
    /// camera-local space (ARKit convention: camera looks down -Z, Y up —
    /// server-side code converts to OpenCV convention, see
    /// arkit_reconstruction.py).
    let faceToCameraColumnMajor: [Float]
    let intrinsics: CameraIntrinsicsPayload
    let jpegData: Data
    let capturedAt: Date
}

private extension simd_float4x4 {
    var columnMajorArray: [Float] {
        [
            columns.0.x, columns.0.y, columns.0.z, columns.0.w,
            columns.1.x, columns.1.y, columns.1.z, columns.1.w,
            columns.2.x, columns.2.y, columns.2.z, columns.2.w,
            columns.3.x, columns.3.y, columns.3.z, columns.3.w,
        ]
    }
}
