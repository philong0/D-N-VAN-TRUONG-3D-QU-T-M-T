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

    public var activeServerURL: String {
        var url = serverBaseURL.trimmingCharacters(in: .whitespacesAndNewlines)
        if url.isEmpty {
            url = "http://149.118.63.240:3000"
        }
        if !url.hasPrefix("http://") && !url.hasPrefix("https://") {
            url = "http://" + url
        }
        if url.hasSuffix("/") {
            url = String(url.dropLast())
        }
        return url
    }

    @Published public var serverBaseURL: String = UserDefaults.standard.string(forKey: "clinicServerURL") ?? "http://149.118.63.240:3000" {
        didSet {
            UserDefaults.standard.set(serverBaseURL, forKey: "clinicServerURL")
        }
    }
    @Published public var isUploading: Bool = false
    @Published public var uploadProgress: Float = 0.0
    @Published public var uploadStatusMessage: String = ""
    @Published public var studioURL: URL? = nil
    public var onProgressUpdate: ((Float, String) -> Void)?

    public init() {}

    // D-patientpicker — the setup screen used to make the operator hand-type
    // a raw Patient UUID + Session UUID with no way to look them up from
    // inside the app. These two calls replace that: fetch the REAL patient
    // list from the same server the web app reads (`GET /api/patients`),
    // then create a real scan session for whichever one is picked (the
    // exact same `POST .../scan-sessions` + `PATCH action:"start"` calls
    // `GuidedFaceScan.tsx` already makes on the web side).
    public func fetchPatients(completion: @escaping (Result<[PatientSummaryDTO], Error>) -> Void) {
        guard let url = URL(string: "\(activeServerURL)/api/patients") else {
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
        guard let url = URL(string: "\(activeServerURL)/api/patients/\(patientId)/scan-sessions") else {
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
        guard let url = URL(string: "\(activeServerURL)/api/patients/\(patientId)/scan-sessions/\(sessionId)") else {
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
        frames: [String: CapturedFramePackage],
        scannerMode: ScannerMode = .faceIdSelfScan,
        completion: @escaping (Result<URL, Error>) -> Void
    ) {
        guard let url = URL(string: "\(activeServerURL)/api/patients/\(patientId)/scan-sessions/\(sessionId)/package") else {
            completion(.failure(NSError(domain: "API", code: 400, userInfo: [NSLocalizedDescriptionKey: "URL máy chủ không hợp lệ."])))
            return
        }
        
        let isRear = scannerMode == .rearClinicalAssistant
        DispatchQueue.main.async {
            self.isUploading = true
            self.uploadProgress = 0.1
            self.uploadStatusMessage = isRear ? "Đang đóng gói ảnh lâm sàng camera sau..." : "Đang đóng gói dữ liệu TrueDepth Face ID..."
        }
        
        // Build Manifest — every real per-frame value embedded INLINE
        // (geometry/intrinsics/pose), matching package/route.ts's actual
        // read contract exactly (see D-contractfix in ScanModels.swift).
        var frameDTOs: [ScanPackageManifestDTO.FrameEntryDTO] = []
        for (viewTag, frame) in frames {
            let entry = ScanPackageManifestDTO.FrameEntryDTO(
                view: viewTag,
                timestamp: frame.timestamp,
                rgbFileName: "\(viewTag).jpg",
                depthFileName: frame.depthData != nil ? "\(viewTag)_depth.raw" : nil,
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
            captureSource: isRear ? "native_ios_rear" : "native_ios",
            deviceModel: UIDevice.current.model,
            systemVersion: UIDevice.current.systemVersion,
            hasTrueDepth: !isRear,
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
        
        for (viewTag, frame) in frames {
            appendFileData(name: "\(viewTag).jpg", fileName: "\(viewTag).jpg", mimeType: "image/jpeg", data: frame.rgbData)
            if let depthData = frame.depthData {
                appendFileData(name: "\(viewTag)_depth.raw", fileName: "\(viewTag)_depth.raw", mimeType: "application/octet-stream", data: depthData)
            }
        }
        body.append("--\(boundary)--\r\n".data(using: .utf8)!)
        
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")
        request.httpBody = body
        request.timeoutInterval = 480
        
        DispatchQueue.main.async {
            self.uploadProgress = 0.20
            self.uploadStatusMessage = "Đang tải dữ liệu TrueDepth lên máy chủ..."
            self.onProgressUpdate?(0.20, "Đang tải dữ liệu TrueDepth lên máy chủ...")
        }
        
        URLSession.shared.dataTask(with: request) { data, response, error in
            if let error = error {
                DispatchQueue.main.async {
                    self.isUploading = false
                    self.uploadStatusMessage = "Lỗi kết nối: \(error.localizedDescription)"
                    self.onProgressUpdate?(self.uploadProgress, "Lỗi kết nối: \(error.localizedDescription)")
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
                var errReason = parsedResponse?.error ?? parsedResponse?.details
                if errReason == nil, let rawText = String(data: data, encoding: .utf8)?.trimmingCharacters(in: .whitespacesAndNewlines), !rawText.isEmpty {
                    if !rawText.hasPrefix("<") {
                        errReason = rawText
                    }
                }
                let finalReason = errReason ?? "Lỗi máy chủ (\(httpResponse.statusCode))."
                let err = NSError(domain: "API", code: httpResponse.statusCode, userInfo: [NSLocalizedDescriptionKey: finalReason])
                DispatchQueue.main.async {
                    self.uploadStatusMessage = finalReason
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
            
            // 4. Kiểm tra session status
            let sessionStatus = respObj.session?.status
            let glbFileName = respObj.session?.reconstruction?.baselineModelFileName
            
            // Nếu đã sẵn sàng baseline.glb (trường hợp hiếm khi đã dựng xong tức thì)
            if sessionStatus == "ready" && glbFileName != nil && !glbFileName!.isEmpty {
                guard let studio = URL(string: "\(self.activeServerURL)/patients/\(patientId)/studio") else {
                    let err = NSError(domain: "API", code: 500, userInfo: [NSLocalizedDescriptionKey: "Lỗi tạo đường dẫn 3D Studio."])
                    DispatchQueue.main.async {
                        self.uploadStatusMessage = err.localizedDescription
                        self.onProgressUpdate?(self.uploadProgress, err.localizedDescription)
                        completion(.failure(err))
                    }
                    return
                }
                
                DispatchQueue.main.async {
                    self.studioURL = studio
                    self.uploadProgress = 1.0
                    self.uploadStatusMessage = "✓ Tái tạo baseline.glb thành công!"
                    self.onProgressUpdate?(1.0, "✓ Tái tạo baseline.glb thành công!")
                    completion(.success(studio))
                }
                return
            }
            
            // Nếu server đang xử lý bất đồng bộ ("processing") -> Tiến hành Polling trạng thái kèm cập nhật UI
            self.pollReconstructionReady(patientId: patientId, sessionId: sessionId, startTime: Date(), completion: completion)
        }.resume()
    }

    private func pollReconstructionReady(patientId: String, sessionId: String, startTime: Date, completion: @escaping (Result<URL, Error>) -> Void) {
        guard let url = URL(string: "\(activeServerURL)/api/patients/\(patientId)/scan-sessions/\(sessionId)") else {
            completion(.failure(NSError(domain: "API", code: 400, userInfo: [NSLocalizedDescriptionKey: "URL polling không hợp lệ"])))
            return
        }
        
        var request = URLRequest(url: url)
        request.httpMethod = "GET"
        request.timeoutInterval = 15
        
        URLSession.shared.dataTask(with: request) { [weak self] data, response, error in
            guard let self = self else { return }
            
            let elapsed = Date().timeIntervalSince(startTime)
            if elapsed > 180 { // 3 phút timeout tối đa
                DispatchQueue.main.async {
                    self.isUploading = false
                    let timeoutErr = NSError(domain: "Reconstruction", code: 408, userInfo: [NSLocalizedDescriptionKey: "Quá thời gian chờ AI Engine dựng 3D. Vui lòng thử lại."])
                    self.uploadStatusMessage = timeoutErr.localizedDescription
                    self.onProgressUpdate?(self.uploadProgress, timeoutErr.localizedDescription)
                    completion(.failure(timeoutErr))
                }
                return
            }
            
            // Xử lý gián đoạn mạng tạm thời: không fail ngay mà thử lại sau 2s
            if let error = error {
                print("[BackendAPIClient] Poll warning: \(error.localizedDescription), retrying...")
                DispatchQueue.global().asyncAfter(deadline: .now() + 2.0) { [weak self] in
                    self?.pollReconstructionReady(patientId: patientId, sessionId: sessionId, startTime: startTime, completion: completion)
                }
                return
            }
            
            // Tính toán % tiến trình tăng liên tục từ 35% -> 95% trong 55s
            let smoothProgress = min(0.95, Float(0.35 + (elapsed / 55.0) * 0.58))
            
            // Cập nhật thông điệp và tiến trình theo thời gian thực để người dùng thấy rõ AI đang làm việc
            DispatchQueue.main.async {
                self.isUploading = true
                let msg: String
                if elapsed < 8 {
                    msg = "AI Engine đang tiếp nhận & đối chiếu các góc quét TrueDepth..."
                } else if elapsed < 20 {
                    msg = "Đang dựng cấu trúc nhân trắc GNM Full-Head (tai, cằm, mũi)..."
                } else if elapsed < 35 {
                    msg = "Đang tinh chỉnh độ dày mô mềm & tính đối xứng lâm sàng..."
                } else if elapsed < 55 {
                    msg = "Đang hoàn thiện vân da bề mặt Ultra-HD (PBR Shader)..."
                } else {
                    msg = "Đang hoàn tất đóng gói mô hình 3D..."
                }
                self.uploadProgress = smoothProgress
                self.uploadStatusMessage = msg
                self.onProgressUpdate?(smoothProgress, msg)
            }
            
            if let data = data,
               let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
               let session = json["session"] as? [String: Any] {
                
                let status = session["status"] as? String
                let recon = session["reconstruction"] as? [String: Any]
                let baselineGLB = recon?["baselineModelFileName"] as? String
                
                if status == "ready" || (baselineGLB != nil && !baselineGLB!.isEmpty) {
                    DispatchQueue.main.async {
                        self.uploadProgress = 1.0
                        let successMsg = "✓ Dựng 3D hoàn tất! Đang chuyển vào 3D Studio..."
                        self.uploadStatusMessage = successMsg
                        self.onProgressUpdate?(1.0, successMsg)
                        DispatchQueue.main.asyncAfter(deadline: .now() + 0.6) {
                            self.isUploading = false
                            if let studio = URL(string: "\(self.activeServerURL)/patients/\(patientId)/studio") {
                                self.studioURL = studio
                                completion(.success(studio))
                            }
                        }
                    }
                    return
                }
                
                if status == "failed" {
                    let errReason = recon?["error"] as? String ?? "AI Engine không thể tái tạo mô hình 3D từ dữ liệu quét này."
                    DispatchQueue.main.async {
                        self.isUploading = false
                        let err = NSError(domain: "Reconstruction", code: 422, userInfo: [NSLocalizedDescriptionKey: errReason])
                        self.uploadStatusMessage = errReason
                        self.onProgressUpdate?(self.uploadProgress, errReason)
                        completion(.failure(err))
                    }
                    return
                }
            }
            
            // Tiếp tục poll sau 2.0 giây
            DispatchQueue.global().asyncAfter(deadline: .now() + 2.0) { [weak self] in
                self?.pollReconstructionReady(patientId: patientId, sessionId: sessionId, startTime: startTime, completion: completion)
            }
        }.resume()
    }

    public func uploadScanPackage(
        patientId: String,
        sessionId: String,
        frames: [ScanAngleStep: CapturedFramePackage],
        scannerMode: ScannerMode = .faceIdSelfScan,
        completion: @escaping (Result<URL, Error>) -> Void
    ) {
        let stringKeyed = Dictionary(uniqueKeysWithValues: frames.map { ($0.key.rawValue, $0.value) })
        uploadScanPackage(patientId: patientId, sessionId: sessionId, frames: stringKeyed, scannerMode: scannerMode, completion: completion)
    }
}
