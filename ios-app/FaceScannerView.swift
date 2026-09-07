import SwiftUI
import WebKit

/// Hosts the existing web scan flow (`/patients/[id]/scan`, i.e.
/// `GuidedFaceScan.tsx`) inside a `WKWebView` and injects the
/// `arkitScanBridge` message handler `src/lib/scan/face-scanner.ts`'s
/// `IOSNativeScanner` already knows how to talk to. This view intentionally
/// contains almost no scan UI of its own — the web page still owns the
/// step-by-step guidance, quality report, and patient-record navigation;
/// this file's only job is to answer two messages from that page:
///
///   `{"action": "start"}`   -> start the real ARSession
///   `{"action": "capture"}` -> capture one real ARKit frame, upload it,
///                              resolve the page's pending JS promise
///
/// via `window.__arkitBridgeResolve(requestId, payload)` /
/// `window.__arkitBridgeReject(requestId, message)`, which `face-scanner.ts`
/// wires up (see that file for the exact promise-bridging protocol).
struct FaceScannerView: View {
    let initialPath: String
    let webAppBaseURL: URL

    @StateObject private var model: FaceScannerViewModel

    init(initialPath: String = "/patients", webAppBaseURL: URL) {
        self.initialPath = initialPath
        self.webAppBaseURL = webAppBaseURL
        _model = StateObject(wrappedValue: FaceScannerViewModel(webAppBaseURL: webAppBaseURL))
    }

    init(patientId: String, webAppBaseURL: URL) {
        self.init(initialPath: "/patients/\(patientId)/scan", webAppBaseURL: webAppBaseURL)
    }

    var body: some View {
        ZStack(alignment: .top) {
            WebViewContainer(model: model)
                .ignoresSafeArea()

            if let message = model.bannerMessage {
                Text(message)
                    .font(.footnote.weight(.semibold))
                    .foregroundColor(.white)
                    .padding(.horizontal, 14)
                    .padding(.vertical, 8)
                    .background(Capsule().fill(Color.black.opacity(0.75)))
                    .padding(.top, 12)
                    .transition(.opacity)
            }
        }
        .onAppear {
            model.load(path: initialPath)
        }
    }
}

/// Owns the `WKWebView`, the `ARFaceCaptureController`, and the current scan
/// session/patient context needed to know where to upload a capture to
/// (`sessionId` arrives from the page via the bridge's `start` message,
/// since the web page — not this native shell — creates the scan session
/// through its existing `/api/patients/{id}/scan-sessions` call).
@MainActor
final class FaceScannerViewModel: NSObject, ObservableObject {
    @Published var bannerMessage: String?

    let webAppBaseURL: URL
    fileprivate weak var webView: WKWebView?

    private let captureController = ARFaceCaptureController()
    private lazy var networkService = NetworkService(baseURL: webAppBaseURL)
    private var patientId: String?
    private var sessionId: String?

    init(webAppBaseURL: URL) {
        self.webAppBaseURL = webAppBaseURL
    }

    func load(path: String) {
        guard let webView else { return }
        let url = webAppBaseURL.appendingPathComponent(path)
        webView.load(URLRequest(url: url))
    }

    fileprivate func attach(_ webView: WKWebView) {
        self.webView = webView
    }

    // MARK: - Bridge message handling

    /// `body` matches the JS-side call in `face-scanner.ts`:
    /// `{ requestId, action, patientId, sessionId, view }`.
    func handleBridgeMessage(_ body: [String: Any]) {
        guard let requestId = body["requestId"] as? String,
              let action = body["action"] as? String else { return }

        switch action {
        case "start":
            patientId = body["patientId"] as? String
            sessionId = body["sessionId"] as? String
            startCapture(requestId: requestId)
        case "capture":
            let view = body["view"] as? String
            captureAndUpload(requestId: requestId, view: view)
        case "stop":
            captureController.stop()
            resolve(requestId: requestId, payload: ["stopped": true])
        default:
            reject(requestId: requestId, message: "Unknown bridge action: \(action)")
        }
    }

    private func startCapture(requestId: String) {
        guard ARFaceCaptureController.isSupported else {
            reject(requestId: requestId, message: "Thiết bị không hỗ trợ TrueDepth/ARKit face tracking.")
            return
        }
        do {
            try captureController.start()
            bannerMessage = "Đã kết nối cảm biến TrueDepth"
            resolve(requestId: requestId, payload: ["started": true])
        } catch {
            reject(requestId: requestId, message: error.localizedDescription)
        }
    }

    private func captureAndUpload(requestId: String, view: String?) {
        guard let patientId, let sessionId else {
            reject(requestId: requestId, message: "Chưa có patientId/sessionId — gọi 'start' trước.")
            return
        }
        guard let view else {
            reject(requestId: requestId, message: "Thiếu tên góc scan (view).")
            return
        }
        do {
            let payload = try captureController.captureCurrentFrame()
            Task {
                do {
                    let result = try await networkService.uploadCapture(
                        patientId: patientId,
                        sessionId: sessionId,
                        view: view,
                        payload: payload
                    )
                    self.bannerMessage = "✓ Đã lưu \(view) (TrueDepth thật)"
                    self.resolve(requestId: requestId, payload: [
                        "view": result.frame.view,
                        "fileUrl": result.fileUrl,
                        "vertexCount": payload.vertexCount,
                        "hasDepth": true,
                    ])
                } catch {
                    self.reject(requestId: requestId, message: error.localizedDescription)
                }
            }
        } catch {
            reject(requestId: requestId, message: error.localizedDescription)
        }
    }

    // MARK: - Resolving the page's pending JS promise

    private func resolve(requestId: String, payload: [String: Any]) {
        guard let webView, let json = Self.jsonString(payload) else { return }
        let script = "window.__arkitBridgeResolve && window.__arkitBridgeResolve('\(requestId)', \(json));"
        webView.evaluateJavaScript(script, completionHandler: nil)
    }

    private func reject(requestId: String, message: String) {
        guard let webView else { return }
        let escaped = message.replacingOccurrences(of: "'", with: "\\'").replacingOccurrences(of: "\n", with: " ")
        let script = "window.__arkitBridgeReject && window.__arkitBridgeReject('\(requestId)', '\(escaped)');"
        webView.evaluateJavaScript(script, completionHandler: nil)
    }

    private static func jsonString(_ dict: [String: Any]) -> String? {
        guard let data = try? JSONSerialization.data(withJSONObject: dict) else { return nil }
        return String(data: data, encoding: .utf8)
    }
}

/// UIKit `WKWebView` bridged into SwiftUI, with `arkitScanBridge` registered
/// as a script message handler before any page load starts.
private struct WebViewContainer: UIViewRepresentable {
    @ObservedObject var model: FaceScannerViewModel

    func makeCoordinator() -> Coordinator {
        Coordinator(model: model)
    }

    func makeUIView(context: Context) -> WKWebView {
        let contentController = WKUserContentController()
        contentController.add(context.coordinator, name: "arkitScanBridge")

        let configuration = WKWebViewConfiguration()
        configuration.userContentController = contentController
        // Web camera fallback (getUserMedia, used by GuidedFaceScan.tsx's
        // own preview <video>) still needs to work inside this WebView even
        // though the REAL capture now goes through ARKit — the live preview
        // the user sees stays the browser's own camera feed either way.
        configuration.allowsInlineMediaPlayback = true
        configuration.mediaTypesRequiringUserActionForPlayback = []

        let webView = WKWebView(frame: .zero, configuration: configuration)
        webView.navigationDelegate = context.coordinator
        webView.allowsBackForwardNavigationGestures = true
        model.attach(webView)
        return webView
    }

    func updateUIView(_ uiView: WKWebView, context: Context) {}

    final class Coordinator: NSObject, WKScriptMessageHandler, WKNavigationDelegate {
        let model: FaceScannerViewModel

        init(model: FaceScannerViewModel) {
            self.model = model
        }

        func userContentController(_ userContentController: WKUserContentController, didReceive message: WKScriptMessage) {
            guard message.name == "arkitScanBridge", let body = message.body as? [String: Any] else { return }
            Task { @MainActor in
                model.handleBridgeMessage(body)
            }
        }

        func webView(_ webView: WKWebView, didFailProvisionalNavigation navigation: WKNavigation!, withError error: Error) {
            NSLog("[FaceScannerView] navigation failed: \(error.localizedDescription)")
        }
    }
}
