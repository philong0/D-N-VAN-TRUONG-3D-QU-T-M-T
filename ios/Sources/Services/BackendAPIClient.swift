//
//  BackendAPIClient.swift
//  DrVanTruongScanner (iOS Native TrueDepth Layer)
//

import Foundation
import UIKit

public struct PatientSummaryDTO: Codable, Identifiable {
    public let id: String
    public let fullName: String
    public let phone: String
    public let createdAt: String
}

public final class BackendAPIClient: ObservableObject {

    // 2026-09-05 fix — this used to be a hardcoded dead tunnel URL
    // (expo-correct-quote-seal.trycloudflare.com) with no relation at all
    // to whatever the user configured in RootView's own Settings sheet
    // (`@AppStorage("clinicServerURL")`) — the WebView would load the
    // correct user-entered address while every actual upload here
    // (uploadScanPackage/triggerReconstruction) silently kept hitting the
    // dead default. Reading from the SAME UserDefaults key at init time
    // keeps both in sync without a second settings UI to maintain.
    @Published public var serverBaseURL: String = UserDefaults.standard.string(forKey: "clinicServerURL") ?? "https://YOUR-SERVER-URL-HERE"
    @Published public var isUploading: Bool = false
    @Published public var uploadProgress: Float = 0.0
    @Published public var uploadStatusMessage: String = ""
    @Published public var studioURL: URL? = nil

    public init() {}

    // D-patientpicker — the setup screen used to make the operator hand-type
    // a raw Patient UUID + Session UUID with no way to look them up from
    // inside the app. These two calls replace that: fetch the REAL patient
    // list from the same server the web app reads (`GET /api/patients`),
    // then create a real scan session for whichever one is picked (the
    // exact same `POST .../scan-sessions` + `PATCH action:"start"` calls
    // `GuidedFaceScan.tsx` already makes on the web side).
    public func fetchPatients(completion: @escaping (Result<[PatientSummaryDTO], Error>) -> Void) {
        guard let url = URL(string: "\(serverBaseURL)/api/patients") else {
            completion(.failure(NSError(domain: "API", code: 400, userInfo: [NSLocalizedDescriptionKey: "URL máy chủ không hợp lệ."])))
            return
        }
        URLSession.shared.dataTask(with: url) { data, _, error in
            if let error = error {
                DispatchQueue.main.async { completion(.failure(error)) }
                return
            }
            guard let data = data else {
                DispatchQueue.main.async {
                    completion(.failure(NSError(domain: "API", code: 500, userInfo: [NSLocalizedDescriptionKey: "Không có dữ liệu trả về từ máy chủ."])))
                }
                return
            }
            struct Wrapper: Codable { let patients: [PatientSummaryDTO] }
            do {
                let wrapper = try JSONDecoder().decode(Wrapper.self, from: data)
                DispatchQueue.main.async { completion(.success(wrapper.patients)) }
            } catch {
                DispatchQueue.main.async { completion(.failure(error)) }
            }
        }.resume()
    }

    public func createScanSession(patientId: String, completion: @escaping (Result<String, Error>) -> Void) {
        guard let url = URL(string: "\(serverBaseURL)/api/patients/\(patientId)/scan-sessions") else {
            completion(.failure(NSError(domain: "API", code: 400, userInfo: [NSLocalizedDescriptionKey: "URL máy chủ không hợp lệ."])))
            return
        }
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try? JSONSerialization.data(withJSONObject: ["scannerKind": "ios_native"])

        URLSession.shared.dataTask(with: request) { data, _, error in
            if let error = error {
                DispatchQueue.main.async { completion(.failure(error)) }
                return
            }
            guard let data = data,
                  let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                  let session = json["session"] as? [String: Any],
                  let sessionId = session["id"] as? String else {
                DispatchQueue.main.async {
                    completion(.failure(NSError(domain: "API", code: 500, userInfo: [NSLocalizedDescriptionKey: "Không tạo được phiên quét mới."])))
                }
                return
            }
            self.markScanSessionStarted(patientId: patientId, sessionId: sessionId) { startResult in
                switch startResult {
                case .success:
                    DispatchQueue.main.async { completion(.success(sessionId)) }
                case .failure(let err):
                    DispatchQueue.main.async { completion(.failure(err)) }
                }
            }
        }.resume()
    }

    private func markScanSessionStarted(patientId: String, sessionId: String, completion: @escaping (Result<Void, Error>) -> Void) {
        guard let url = URL(string: "\(serverBaseURL)/api/patients/\(patientId)/scan-sessions/\(sessionId)") else {
            completion(.failure(NSError(domain: "API", code: 400, userInfo: [NSLocalizedDescriptionKey: "URL máy chủ không hợp lệ."])))
            return
        }
        var request = URLRequest(url: url)
        request.httpMethod = "PATCH"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try? JSONSerialization.data(withJSONObject: ["action": "start"])
        URLSession.shared.dataTask(with: request) { _, _, error in
            if let error = error {
                completion(.failure(error))
            } else {
                completion(.success(()))
            }
        }.resume()
    }
    
    public func uploadScanPackage(
        patientId: String,
        sessionId: String,
        frames: [ScanAngleStep: CapturedFramePackage],
        completion: @escaping (Result<URL, Error>) -> Void
    ) {
        guard let url = URL(string: "\(serverBaseURL)/api/patients/\(patientId)/scan-sessions/\(sessionId)/package") else {
            completion(.failure(NSError(domain: "API", code: 400, userInfo: [NSLocalizedDescriptionKey: "URL máy chủ không hợp lệ."])))
            return
        }
        
        DispatchQueue.main.async {
            self.isUploading = true
            self.uploadProgress = 0.1
            self.uploadStatusMessage = "Đang đóng gói dữ liệu TrueDepth v2..."
        }
        
        // Build Manifest — every real per-frame value embedded INLINE
        // (geometry/intrinsics/pose), matching package/route.ts's actual
        // read contract exactly (see D-contractfix in ScanModels.swift).
        var frameDTOs: [ScanPackageManifestDTO.FrameEntryDTO] = []
        for (step, frame) in frames {
            let entry = ScanPackageManifestDTO.FrameEntryDTO(
                view: step.rawValue,
                timestamp: frame.timestamp,
                rgbFileName: "\(step.rawValue).jpg",
                depthFileName: frame.depthData != nil ? "\(step.rawValue)_depth.raw" : nil,
                isTracked: frame.geometry.isTracked,
                yawDeg: frame.quality.yawDeg,
                pitchDeg: frame.quality.pitchDeg,
                geometry: frame.geometry.vertexCount > 0 ? frame.geometry : nil,
                intrinsics: frame.intrinsics,
                pose: frame.pose
            )
            frameDTOs.append(entry)
        }
        
        let manifest = ScanPackageManifestDTO(
            schemaVersion: "2.0.0",
            captureSource: "native_ios",
            deviceModel: UIDevice.current.model,
            systemVersion: UIDevice.current.systemVersion,
            hasTrueDepth: true,
            patientId: patientId,
            sessionId: sessionId,
            capturedAt: ISO8601DateFormatter().string(from: Date()),
            frames: frameDTOs
        )
        
        guard let manifestData = try? JSONEncoder().encode(manifest),
              let manifestStr = String(data: manifestData, encoding: .utf8) else {
            completion(.failure(NSError(domain: "API", code: 500, userInfo: [NSLocalizedDescriptionKey: "Không thể mã hóa manifest.json."])))
            return
        }
        
        // Build Multipart Form Data
        let boundary = "Boundary-\(UUID().uuidString)"
        var body = Data()
        
        func appendFormField(name: String, value: String) {
            body.append("--\(boundary)\r\n".data(using: .utf8)!)
            body.append("Content-Disposition: form-data; name=\"\(name)\"\r\n\r\n".data(using: .utf8)!)
            body.append("\(value)\r\n".data(using: .utf8)!)
        }
        
        func appendFileData(name: String, fileName: String, mimeType: String, data: Data) {
            body.append("--\(boundary)\r\n".data(using: .utf8)!)
            body.append("Content-Disposition: form-data; name=\"\(name)\"; filename=\"\(fileName)\"\r\n".data(using: .utf8)!)
            body.append("Content-Type: \(mimeType)\r\n\r\n".data(using: .utf8)!)
            body.append(data)
            body.append("\r\n".data(using: .utf8)!)
        }
        
        appendFormField(name: "manifest", value: manifestStr)
        
        for (step, frame) in frames {
            appendFileData(name: "\(step.rawValue).jpg", fileName: "\(step.rawValue).jpg", mimeType: "image/jpeg", data: frame.rgbData)
            if let depthData = frame.depthData {
                appendFileData(name: "\(step.rawValue)_depth.raw", fileName: "\(step.rawValue)_depth.raw", mimeType: "application/octet-stream", data: depthData)
            }
        }
        body.append("--\(boundary)--\r\n".data(using: .utf8)!)
        
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")
        request.httpBody = body
        // Same reasoning as triggerReconstruction's own timeoutInterval fix
        // below — real depth+RGB payloads for 5 views can be slow to
        // upload on a weak connection; 60s default is too tight a margin.
        request.timeoutInterval = 180
        
        DispatchQueue.main.async {
            self.uploadProgress = 0.4
            self.uploadStatusMessage = "Đang tải dữ liệu lên máy chủ..."
        }
        
        URLSession.shared.dataTask(with: request) { data, response, error in
            if let error = error {
                DispatchQueue.main.async {
                    self.isUploading = false
                    completion(.failure(error))
                }
                return
            }
            
            DispatchQueue.main.async {
                self.uploadProgress = 0.7
                self.uploadStatusMessage = "Đang kích hoạt Tái tạo 3D (Reconstruction)..."
            }
            
            self.triggerReconstruction(patientId: patientId, sessionId: sessionId) { recResult in
                DispatchQueue.main.async {
                    self.isUploading = false
                    switch recResult {
                    case .success(let studioURL):
                        self.studioURL = studioURL
                        self.uploadProgress = 1.0
                        self.uploadStatusMessage = "✓ Tái tạo baseline.glb thành công!"
                        completion(.success(studioURL))
                    case .failure(let recErr):
                        completion(.failure(recErr))
                    }
                }
            }
        }.resume()
    }
    
    private func triggerReconstruction(
        patientId: String,
        sessionId: String,
        completion: @escaping (Result<URL, Error>) -> Void
    ) {
        guard let url = URL(string: "\(serverBaseURL)/api/patients/\(patientId)/scan-sessions/\(sessionId)") else {
            completion(.failure(NSError(domain: "API", code: 400, userInfo: [NSLocalizedDescriptionKey: "URL reconstruction không hợp lệ."])))
            return
        }
        
        var req = URLRequest(url: url)
        req.httpMethod = "PATCH"
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        req.httpBody = try? JSONSerialization.data(withJSONObject: ["action": "request_reconstruction"])
        // 2026-09-07 fix — `URLSession.shared`'s default
        // `timeoutIntervalForRequest` is 60s (Apple's own default). This
        // one request stays open, with zero bytes sent back, for the
        // ENTIRE real 3D reconstruction (backend's own `execFile` budget
        // just raised to 480s — see reconstruction-service.ts's own fix
        // note for the real measured evidence this came from: a real scan
        // whose reconstruction kept running and wrote a real baseline.glb
        // minutes after the app had already given up and fallen back to a
        // fresh scan). Set per-request here to stay in sync with that
        // budget without swapping `URLSession.shared` for a custom session
        // everywhere else in this file.
        req.timeoutInterval = 500

        URLSession.shared.dataTask(with: req) { data, resp, err in
            if let err = err {
                completion(.failure(err))
                return
            }
            
            guard let studio = URL(string: "\(self.serverBaseURL)/patients/\(patientId)/studio") else {
                completion(.failure(NSError(domain: "API", code: 500, userInfo: [NSLocalizedDescriptionKey: "Lỗi tạo Studio URL."])))
                return
            }
            completion(.success(studio))
        }.resume()
    }
}
