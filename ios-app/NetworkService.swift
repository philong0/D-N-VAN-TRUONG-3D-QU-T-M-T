import Foundation

/// Uploads one `ARKitCapturePayload` (see `ARFaceCaptureController.swift`)
/// straight to the SAME endpoint the web capture flow already posts JPEGs
/// to: `POST /api/patients/{patientId}/scan-sessions/{sessionId}/frames`.
/// That route (see `src/app/api/patients/[id]/scan-sessions/[sessionId]/
/// frames/route.ts`) now accepts two additional optional multipart fields,
/// `geometry` and `intrinsics` (both JSON strings) — everything else about
/// the request (field names `frame`/`view`/`depthAvailable`, response
/// shape) is unchanged from the web camera path, so the existing
/// `ScanFrame` storage/quality-check logic needs no native-specific branch.
final class NetworkService {
    /// Set to wherever the Next.js app is actually reachable from the
    /// device — see README.md's setup step 5. `http://localhost:3000` only
    /// resolves inside the iOS Simulator; a physical device needs a LAN IP
    /// or the deployed HTTPS origin.
    var baseURL: URL

    init(baseURL: URL) {
        self.baseURL = baseURL
    }

    enum UploadError: LocalizedError {
        case invalidResponse
        case serverError(status: Int, message: String)

        var errorDescription: String? {
            switch self {
            case .invalidResponse:
                return "Phản hồi không hợp lệ từ máy chủ."
            case .serverError(let status, let message):
                return "Lỗi máy chủ (\(status)): \(message)"
            }
        }
    }

    struct UploadResult: Decodable {
        struct Frame: Decodable {
            let id: String
            let view: String
            let fileName: String
        }
        let frame: Frame
        let fileUrl: String
    }

    /// `view` must be one of the 5 `ScanCaptureView` values the web app
    /// already defines (`front`/`left_45`/`left_profile`/`right_45`/
    /// `right_profile`) — this native uploader does not introduce a new
    /// vocabulary, it fills in the SAME steps `GuidedFaceScan.tsx` already
    /// guides the user through, just with real depth data attached.
    func uploadCapture(
        patientId: String,
        sessionId: String,
        view: String,
        payload: ARKitCapturePayload
    ) async throws -> UploadResult {
        let url = baseURL
            .appendingPathComponent("api/patients/\(patientId)/scan-sessions/\(sessionId)/frames")

        let geometryJSON = try Self.encodeGeometry(payload)
        let intrinsicsJSON = try Self.encodeIntrinsics(payload.intrinsics)

        let boundary = "Boundary-\(UUID().uuidString)"
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")

        var body = Data()
        func appendField(name: String, value: String) {
            body.append("--\(boundary)\r\n".data(using: .utf8)!)
            body.append("Content-Disposition: form-data; name=\"\(name)\"\r\n\r\n".data(using: .utf8)!)
            body.append(value.data(using: .utf8)!)
            body.append("\r\n".data(using: .utf8)!)
        }
        func appendFile(name: String, filename: String, mimeType: String, data: Data) {
            body.append("--\(boundary)\r\n".data(using: .utf8)!)
            body.append(
                "Content-Disposition: form-data; name=\"\(name)\"; filename=\"\(filename)\"\r\n"
                    .data(using: .utf8)!
            )
            body.append("Content-Type: \(mimeType)\r\n\r\n".data(using: .utf8)!)
            body.append(data)
            body.append("\r\n".data(using: .utf8)!)
        }

        appendField(name: "view", value: view)
        appendField(name: "depthAvailable", value: "true")
        appendField(name: "geometry", value: geometryJSON)
        appendField(name: "intrinsics", value: intrinsicsJSON)
        appendFile(name: "frame", filename: "\(view).jpg", mimeType: "image/jpeg", data: payload.jpegData)
        body.append("--\(boundary)--\r\n".data(using: .utf8)!)

        request.httpBody = body

        let (data, response) = try await URLSession.shared.data(for: request)
        guard let http = response as? HTTPURLResponse else { throw UploadError.invalidResponse }
        guard (200...299).contains(http.statusCode) else {
            let message = String(data: data, encoding: .utf8) ?? "unknown"
            throw UploadError.serverError(status: http.statusCode, message: message)
        }
        return try JSONDecoder().decode(UploadResult.self, from: data)
    }

    /// Shape matches `ARKitFaceGeometryPayload` in `src/lib/scan/face-scanner.ts`
    /// (kept in sync manually — both sides are small and rarely change) plus
    /// the face-local->camera-local transform this capture computed, which
    /// `ai-engine/arkit_reconstruction.py` parses (see that file for the
    /// exact column-major convention and ARKit->OpenCV axis conversion).
    private static func encodeGeometry(_ payload: ARKitCapturePayload) throws -> String {
        let dict: [String: Any] = [
            "vertexCount": payload.vertexCount,
            "triangleCount": payload.triangleCount,
            "vertices": payload.vertices,
            "triangleIndices": payload.triangleIndices,
            "textureCoordinates": payload.textureCoordinates,
            "faceToCameraColumnMajor": payload.faceToCameraColumnMajor,
            "capturedAt": ISO8601DateFormatter().string(from: payload.capturedAt),
        ]
        let data = try JSONSerialization.data(withJSONObject: dict)
        return String(data: data, encoding: .utf8) ?? "{}"
    }

    private static func encodeIntrinsics(_ intrinsics: CameraIntrinsicsPayload) throws -> String {
        let dict: [String: Any] = [
            "fx": intrinsics.fx,
            "fy": intrinsics.fy,
            "cx": intrinsics.cx,
            "cy": intrinsics.cy,
            "imageWidth": intrinsics.imageWidth,
            "imageHeight": intrinsics.imageHeight,
        ]
        let data = try JSONSerialization.data(withJSONObject: dict)
        return String(data: data, encoding: .utf8) ?? "{}"
    }

    /// Simple reachability check the SwiftUI view uses to show a clear
    /// error instead of a confusing hang if `baseURL` is wrong/unreachable
    /// (the single most likely real-world setup mistake — see README step 5).
    /// No dedicated `/api/health` route exists in this project, so this just
    /// hits the app's own root page, which every Next.js deployment serves.
    func checkReachable() async -> Bool {
        var request = URLRequest(url: baseURL)
        request.timeoutInterval = 4
        do {
            let (_, response) = try await URLSession.shared.data(for: request)
            return (response as? HTTPURLResponse).map { (200...499).contains($0.statusCode) } ?? false
        } catch {
            return false
        }
    }
}
