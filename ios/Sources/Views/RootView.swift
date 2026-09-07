//
//  RootView.swift
//  DrVanTruongScanner (All-in-One iOS Native TrueDepth Studio)
//

import SwiftUI
import WebKit
import ARKit
import Foundation

/// D-bridgehost — hosts the `arkitScanBridge` WKScriptMessageHandler so the
/// exact same web app (GuidedFaceScan.tsx) that already knows how to detect
/// and drive a native TrueDepth scanner (see `face-scanner.ts`'s
/// `IOSNativeScanner`) gets real ARKit capture instead of `getUserMedia`,
/// with zero change to which URL/page loads — this is the SAME CRM/studio
/// site, not a separate app screen.
struct WebView: UIViewRepresentable {
    let url: URL
    let bridge: ArkitScanBridge

    func makeUIView(context: Context) -> WKWebView {
        let contentController = WKUserContentController()
        contentController.add(bridge, name: "arkitScanBridge")
        let config = WKWebViewConfiguration()
        config.userContentController = contentController
        config.allowsInlineMediaPlayback = true

        let webView = WKWebView(frame: .zero, configuration: config)
        bridge.webView = webView
        webView.load(URLRequest(url: url))
        return webView
    }

    func updateUIView(_ uiView: WKWebView, context: Context) {}
}

public struct RootView: View {
    @StateObject private var scanBridge = ArkitScanBridge()

    // D-nofloatingbutton — the previous floating "Quét TrueDepth" button
    // opened a SEPARATE native scan flow hardcoded to one fixed test
    // patient ID, regardless of which patient page was actually open in
    // the WebView underneath — it could never scan the right patient, and
    // every real attempt at using it left no scan session behind at all
    // (confirmed: zero `ios_native` sessions ever recorded server-side).
    // Removed rather than fixed: the user explicitly does not want a
    // separate native scan screen — the bridge below already puts real
    // TrueDepth capture behind the web app's own "Quét mặt mới" button,
    // for whichever patient is actually open, with no second UI to keep
    // in sync.
    //
    // 2026-09-05 fix — this URL used to be a hardcoded `let` pointing at a
    // dead, one-time tunnel address from a past session
    // (expo-correct-quote-seal.trycloudflare.com), with NO way to change it
    // from inside the app at all: every real test silently tried to reach a
    // server that no longer exists. Tunnel addresses (trycloudflare.com,
    // loca.lt, etc.) are only ever temporary — they change every time the
    // tunnel is restarted — so this MUST be editable at runtime, not baked
    // into the binary. Switched to `@AppStorage` (persists across app
    // launches, same mechanism `ios-app/DrVanTruongScannerApp.swift`'s own
    // working settings screen already uses) plus a real settings sheet.
    @AppStorage("clinicServerURL") private var serverURLString: String = "https://YOUR-SERVER-URL-HERE"
    @State private var showingSettings = false

    public init() {}

    public var body: some View {
        ZStack(alignment: .topTrailing) {
            if let url = URL(string: serverURLString), serverURLString != "https://YOUR-SERVER-URL-HERE" {
                // GIAO DIỆN VIP CRM: QUẢN LÝ HỒ SƠ + 3D STUDIO — cùng 1 trang
                // web này giờ tự nhận ra `arkitScanBridge` và bước "Quét mặt
                // mới" bên trong nó tự chuyển sang quét TrueDepth thật, cho
                // đúng bệnh nhân đang mở trên màn hình.
                WebView(url: url, bridge: scanBridge)
                    .edgesIgnoringSafeArea(.all)
                    .fullScreenCover(isPresented: $scanBridge.isPresentingScanner) {
                        ARFaceScannerView(
                            captureSession: scanBridge.captureSession,
                            isCompleted: .constant(false)
                        )
                    }
            } else {
                VStack(spacing: 16) {
                    Text("Chưa cấu hình địa chỉ máy chủ").font(.headline)
                    Text("Bấm nút cài đặt (góc trên bên phải) để nhập địa chỉ web hiện tại của bạn.")
                        .font(.footnote)
                        .foregroundColor(.secondary)
                        .multilineTextAlignment(.center)
                        .padding(.horizontal, 32)
                }
            }

            Button {
                showingSettings = true
            } label: {
                Image(systemName: "gearshape.fill")
                    .padding(10)
                    .background(.ultraThinMaterial, in: Circle())
            }
            .padding()
        }
        .sheet(isPresented: $showingSettings) {
            NavigationView {
                Form {
                    Section("Địa chỉ máy chủ (web app hiện tại)") {
                        TextField("https://...", text: $serverURLString)
                            .keyboardType(.URL)
                            .autocorrectionDisabled()
                            .textInputAutocapitalization(.never)
                        Text("Lưu ý: nếu server dùng tunnel tạm (trycloudflare.com, loca.lt...), địa chỉ này đổi mỗi lần server khởi động lại — cần cập nhật lại đây mỗi lần đổi.")
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                }
                .navigationTitle("Cài đặt máy chủ")
                .toolbar {
                    Button("Xong") { showingSettings = false }
                }
            }
        }
    }
}
