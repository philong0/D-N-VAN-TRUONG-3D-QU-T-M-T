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
        webView.isOpaque = false
        webView.backgroundColor = .systemBackground
        webView.scrollView.contentInsetAdjustmentBehavior = .never
        webView.load(URLRequest(url: url))
        return webView
    }

    func updateUIView(_ uiView: WKWebView, context: Context) {}
}

public struct RootView: View {
    @StateObject private var scanBridge = ArkitScanBridge()
    @AppStorage("clinicServerURL") private var serverURLString: String = "https://mpeg-atmospheric-consultant-despite.trycloudflare.com"
    @State private var showingSettings = false

    public init() {}

    public var body: some View {
        ZStack(alignment: .bottomTrailing) {
            if let url = URL(string: serverURLString), !serverURLString.isEmpty {
                WebView(url: url, bridge: scanBridge)
                    .ignoresSafeArea(.all)
                    .fullScreenCover(isPresented: $scanBridge.isPresentingScanner) {
                        ARFaceScannerView(
                            captureSession: scanBridge.captureSession,
                            isCompleted: Binding(
                                get: { !scanBridge.isPresentingScanner },
                                set: { if $0 { scanBridge.isPresentingScanner = false } }
                            )
                        )
                    }
            } else {
                VStack(spacing: 16) {
                    Text("Chưa cấu hình địa chỉ máy chủ").font(.headline)
                    Text("Bấm nút cài đặt để nhập địa chỉ web hiện tại của bạn.")
                        .font(.footnote)
                        .foregroundColor(.secondary)
                        .multilineTextAlignment(.center)
                        .padding(.horizontal, 32)
                }
            }

            // Nút cài đặt nhỏ gọn, tinh tế ở góc dưới phải tránh che header
            Button {
                showingSettings = true
            } label: {
                Image(systemName: "gearshape.fill")
                    .font(.system(size: 14, weight: .semibold))
                    .foregroundColor(.white.opacity(0.85))
                    .padding(8)
                    .background(Color.black.opacity(0.4), in: Circle())
                    .shadow(radius: 4)
            }
            .padding(.trailing, 16)
            .padding(.bottom, 70)
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
