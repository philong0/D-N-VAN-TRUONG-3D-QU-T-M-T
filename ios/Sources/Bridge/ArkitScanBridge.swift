//
//  ArkitScanBridge.swift
//  DrVanTruongScanner (WKWebView <-> ARKit bridge)
//
//  Implements the native side of the `window.webkit.messageHandlers
//  .arkitScanBridge` contract that `src/lib/scan/face-scanner.ts`'s
//  `IOSNativeScanner` already defines and that
//  `src/components/scan/GuidedFaceScan.tsx` already calls when it detects
//  this bridge is present. The web app's own guided 5-step scan UI stays
//  exactly as-is — only the capture step underneath is now real ARKit
//  TrueDepth instead of `getUserMedia`.
//
//  Message protocol (matches face-scanner.ts's `callArkitBridge` exactly):
//    JS -> native: { requestId, action: "start"|"capture"|"stop", patientId?, sessionId?, view? }
//    native -> JS: window.__arkitBridgeResolve(requestId, payload)
//               or window.__arkitBridgeReject(requestId, message)
//

import Foundation
import WebKit
import Combine

public final class ArkitScanBridge: NSObject, ObservableObject, WKScriptMessageHandler {

    @Published public var isPresentingScanner: Bool = false

    public let captureSession = ARFaceCaptureSession()
    private let apiClient = BackendAPIClient()

    public weak var webView: WKWebView?

    private var patientId: String?
    private var sessionId: String?
    private var pendingCaptures: [String: AnyCancellable] = [:]

    // MARK: - WKScriptMessageHandler

    public func userContentController(_ userContentController: WKUserContentController, didReceive message: WKScriptMessage) {
        guard let body = message.body as? [String: Any],
              let action = body["action"] as? String,
              let requestId = body["requestId"] as? String else {
            return
        }

        switch action {
        case "start":
            guard let pid = body["patientId"] as? String, let sid = body["sessionId"] as? String else {
                reject(requestId, "Thiếu patientId/sessionId cho action start.")
                return
            }
            self.patientId = pid
            self.sessionId = sid
            captureSession.resetScan()
            isPresentingScanner = true
            resolve(requestId, payload: ["started": true])

        case "capture":
            guard let view = body["view"] as? String else {
                reject(requestId, "Thiếu view cho action capture.")
                return
            }
            handleCapture(requestId: requestId, viewRaw: view)

        case "stop":
            pendingCaptures.values.forEach { $0.cancel() }
            pendingCaptures.removeAll()
            captureSession.pauseSession()
            isPresentingScanner = false
            resolve(requestId, payload: ["stopped": true])

        default:
            reject(requestId, "Hành động không hỗ trợ: \(action)")
        }
    }

    // MARK: - Capture (waits for the real ARKit auto-capture / shutter tap
    // already implemented in ARFaceCaptureSession + ARFaceScannerView; this
    // bridge does not trigger capture itself, it observes it)

    private func handleCapture(requestId: String, viewRaw: String) {
        guard let step = ScanAngleStep(rawValue: viewRaw) else {
            reject(requestId, "View không hợp lệ: \(viewRaw)")
            return
        }

        if let frame = captureSession.capturedFrames[step] {
            finishCapture(requestId: requestId, frame: frame)
            return
        }

        let cancellable = captureSession.$capturedFrames
            .receive(on: DispatchQueue.main)
            .sink { [weak self] frames in
                guard let self = self, let frame = frames[step] else { return }
                self.pendingCaptures.removeValue(forKey: requestId)?.cancel()
                self.finishCapture(requestId: requestId, frame: frame)
            }
        pendingCaptures[requestId] = cancellable
    }

    /// If this was the last of the 5 angles, upload the real package
    /// (RGB + depth + ARFaceGeometry + intrinsics) and trigger
    /// reconstruction using the existing, already-verified
    /// `BackendAPIClient.uploadScanPackage` — only resolving the bridge
    /// promise once that real upload completes, so the awaiting JS side
    /// (`await nativeScanner.captureForView(...)`) never moves on before
    /// the data actually reached the server.
    private func finishCapture(requestId: String, frame: CapturedFramePackage) {
        let fileUrl = "data:image/jpeg;base64,\(frame.rgbData.base64EncodedString())"
        let vertexCount = frame.geometry.vertexCount
        let isLastView = captureSession.capturedFrames.count == ScanAngleStep.allCases.count

        guard isLastView else {
            resolve(requestId, payload: ["fileUrl": fileUrl, "vertexCount": vertexCount])
            return
        }

        guard let patientId = self.patientId, let sessionId = self.sessionId else {
            reject(requestId, "Thiếu patientId/sessionId khi tải gói quét lên.")
            return
        }

        apiClient.uploadScanPackage(
            patientId: patientId,
            sessionId: sessionId,
            frames: captureSession.capturedFrames
        ) { [weak self] result in
            switch result {
            case .success:
                self?.resolve(requestId, payload: ["fileUrl": fileUrl, "vertexCount": vertexCount])
            case .failure(let error):
                self?.reject(requestId, error.localizedDescription)
            }
        }
    }

    // MARK: - JS callbacks

    private func resolve(_ requestId: String, payload: [String: Any]) {
        guard let data = try? JSONSerialization.data(withJSONObject: payload),
              let json = String(data: data, encoding: .utf8) else {
            reject(requestId, "Không mã hóa được kết quả.")
            return
        }
        runJS("window.__arkitBridgeResolve && window.__arkitBridgeResolve('\(requestId)', \(json));")
    }

    private func reject(_ requestId: String, _ message: String) {
        let escaped = message.replacingOccurrences(of: "'", with: "\\'")
        runJS("window.__arkitBridgeReject && window.__arkitBridgeReject('\(requestId)', '\(escaped)');")
    }

    private func runJS(_ script: String) {
        DispatchQueue.main.async { [weak self] in
            self?.webView?.evaluateJavaScript(script, completionHandler: nil)
        }
    }
}
