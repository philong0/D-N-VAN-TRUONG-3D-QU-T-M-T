//
//  ArkitScanBridge.swift
//  DrVanTruongScanner (WKWebView <-> ARKit bridge)
//
//  Implements the native side of the `window.webkit.messageHandlers
//  .arkitScanBridge` contract. `src/lib/scan/face-scanner.ts`'s
//  `IOSNativeScanner` defines a richer "start"/"capture"/"stop" version of
//  this protocol, but `src/components/scan/GuidedFaceScan.tsx` (the
//  component that actually runs today) never imports or calls it —
//  confirmed by grep, zero references. The real, live flow only ever sends
//  ONE "start" message; the entire 5-angle capture + upload sequence
//  happens inside the native `ARFaceScannerView`/`ARFaceCaptureSession` UI
//  after that, with no further JS round-trips.
//
//  2026-09-07 fix — this file used to ALSO implement a "capture" action
//  (`handleCapture`/`finishCapture` below) that called
//  `BackendAPIClient.uploadScanPackage` a SECOND, independent way,
//  triggered only if the (dead, never-sent) JS "capture" message ever
//  arrived. Real-device review correctly flagged this as a structural
//  double-upload/double-reconstruction risk even though it never actually
//  fired in practice. Removed rather than left dormant — the ONLY real
//  upload path now is `ARFaceCaptureSession.triggerPackageUpload`,
//  called automatically once at 5/5 real captures.
//
//  Message protocol:
//    JS -> native: { requestId, action: "start"|"stop", patientId?, sessionId? }
//    native -> JS: window.__arkitBridgeResolve(requestId, payload)
//               or window.__arkitBridgeReject(requestId, message)
//

import Foundation
import WebKit

public final class ArkitScanBridge: NSObject, ObservableObject, WKScriptMessageHandler {

    @Published public var isPresentingScanner: Bool = false

    public let captureSession = ARFaceCaptureSession()

    public weak var webView: WKWebView?

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
            captureSession.patientId = pid
            captureSession.sessionId = sid
            captureSession.resetScan()
            
            captureSession.onScanCompleted = { [weak self] studioURL in
                self?.isPresentingScanner = false
                self?.resolve(requestId, payload: ["completed": true, "studioURL": studioURL.absoluteString])
                self?.runJS("window.location.href = '/patients/\(pid)/studio';")
            }
            captureSession.onScanCancelled = { [weak self] in
                self?.isPresentingScanner = false
                self?.reject(requestId, "Người dùng đã hủy phiên quét.")
            }
            
            isPresentingScanner = true

        case "stop":
            captureSession.pauseSession()
            isPresentingScanner = false
            resolve(requestId, payload: ["stopped": true])

        default:
            reject(requestId, "Hành động không hỗ trợ: \(action)")
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
