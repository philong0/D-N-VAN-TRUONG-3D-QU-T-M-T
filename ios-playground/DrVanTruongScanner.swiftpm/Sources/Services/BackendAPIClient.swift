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
                depthWidth: frame.depthWidth,
                depthHeight: frame.depthHeight,
                depthIntrinsics: frame.depthIntrinsics,
                isTracked: frame.geometry.isTracked,
                yawDeg: frame.quality.yawDeg,
                pitchDeg: frame.quality.pitchDeg,
                geometry: frame.geometry.vertexCount > 0 ? frame.geometry : nil,
                intrinsics: frame.intrinsics,
                pose: frame.pose,
                quality: frame.quality
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
        request.timeoutInterval = 480
        
        DispatchQueue.main.async {
            self.uploadProgress = 0.3
            self.uploadStatusMessage = "Đang tải dữ liệu TrueDepth / ARKit & Dựng 3D..."
        }
        
        URLSession.shared.dataTask(with: request) { data, response, error in
            DispatchQueue.main.async {
                self.isUploading = false
            }
            
            if let error = error {
                DispatchQueue.main.async {
                    self.uploadStatusMessage = "Lỗi kết nối: \(error.localizedDescription)"
                    completion(.failure(error))
                }
                return
            }
            
            guard let httpResponse = response as? HTTPURLResponse else {
                let err = NSError(domain: "API", code: 500, userInfo: [NSLocalizedDescriptionKey: "Phản hồi không hợp lệ từ máy chủ."])
                DispatchQueue.main.async {
                    self.uploadStatusMessage = err.localizedDescription
                    completion(.failure(err))
                }
                return
            }
            
            guard let data = data else {
                let err = NSError(domain: "API", code: httpResponse.statusCode, userInfo: [NSLocalizedDescriptionKey: "Máy chủ không trả về dữ liệu (HTTP \(httpResponse.statusCode))."])
                DispatchQueue.main.async {
                    self.uploadStatusMessage = err.localizedDescription
                    completion(.failure(err))
                }
                return
            }
            
            let decoder = JSONDecoder()
            let parsedResponse = try? decoder.decode(PackageUploadResponseDTO.self, from: data)
            
            // 1. Kiểm tra HTTP Status (200..299)
            if httpResponse.statusCode < 200 || httpResponse.statusCode >= 300 {
                let errReason = parsedResponse?.error ?? parsedResponse?.details ?? "Lỗi máy chủ (\(httpResponse.statusCode))."
                let err = NSError(domain: "API", code: httpResponse.statusCode, userInfo: [NSLocalizedDescriptionKey: errReason])
                DispatchQueue.main.async {
                    self.uploadStatusMessage = errReason
                    completion(.failure(err))
                }
                return
            }
            
            // 2. Kiểm tra JSON response
            guard let respObj = parsedResponse, respObj.success == true else {
                let errReason = parsedResponse?.error ?? "Gói dữ liệu quét không được chấp nhận."
                let err = NSError(domain: "API", code: 422, userInfo: [NSLocalizedDescriptionKey: errReason])
                DispatchQueue.main.async {
                    self.uploadStatusMessage = errReason
                    completion(.failure(err))
                }
                return
            }
            
            // 3. Kiểm tra Quality Gate
            if let quality = respObj.quality, quality.overall == "fail" {
                let qualityReason = quality.coverage?.detail ?? quality.frameQuality?.detail ?? "Chất lượng quét chưa đạt yêu cầu của Quality Gate."
                let err = NSError(domain: "QualityGate", code: 422, userInfo: [NSLocalizedDescriptionKey: qualityReason])
                DispatchQueue.main.async {
                    self.uploadStatusMessage = qualityReason
                    completion(.failure(err))
                }
                return
            }
            
            // 4. Kiểm tra session status "ready" và baseline GLB hợp lệ
            let sessionStatus = respObj.session?.status
            let glbFileName = respObj.session?.reconstruction?.baselineModelFileName
            
            guard sessionStatus == "ready" && glbFileName != nil && !glbFileName!.isEmpty else {
                let reconErr = respObj.session?.reconstruction?.error ?? "Chưa tạo được mô hình baseline 3D hợp lệ từ dữ liệu quét."
                let err = NSError(domain: "Reconstruction", code: 500, userInfo: [NSLocalizedDescriptionKey: reconErr])
                DispatchQueue.main.async {
                    self.uploadStatusMessage = reconErr
                    completion(.failure(err))
                }
                return
            }
            
            guard let studio = URL(string: "\(self.serverBaseURL)/patients/\(patientId)/studio") else {
                let err = NSError(domain: "API", code: 500, userInfo: [NSLocalizedDescriptionKey: "Lỗi tạo đường dẫn 3D Studio."])
                DispatchQueue.main.async {
                    self.uploadStatusMessage = err.localizedDescription
                    completion(.failure(err))
                }
                return
            }
            
            DispatchQueue.main.async {
                self.studioURL = studio
                self.uploadProgress = 1.0
                self.uploadStatusMessage = "✓ Tái tạo baseline.glb thành công!"
                completion(.success(studio))
            }
        }.resume()
    }
}
