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
    let reloadTrigger: UUID

    func makeCoordinator() -> Coordinator {
        Coordinator(bridge: bridge)
    }

    class Coordinator: NSObject, WKNavigationDelegate {
        let bridge: ArkitScanBridge

        init(bridge: ArkitScanBridge) {
            self.bridge = bridge
        }

        func webView(_ webView: WKWebView, didFailProvisionalNavigation navigation: WKNavigation!, withError error: Error) {
            print("WebView didFailProvisionalNavigation:", error.localizedDescription)
            let fallback = "https://lens-inside-silence-bearing.trycloudflare.com"
            if let target = URL(string: fallback), webView.url?.absoluteString != fallback {
                DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
                    webView.load(URLRequest(url: target))
                }
            }
        }

        func webView(_ webView: WKWebView, didFail navigation: WKNavigation!, withError error: Error) {
            print("WebView didFail:", error.localizedDescription)
            let fallback = "https://lens-inside-silence-bearing.trycloudflare.com"
            if let target = URL(string: fallback), webView.url?.absoluteString != fallback {
                DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
                    webView.load(URLRequest(url: target))
                }
            }
        }

        @objc func handleRefreshControl(sender: UIRefreshControl) {
            bridge.webView?.reload()
            sender.endRefreshing()
        }
    }

    func makeUIView(context: Context) -> WKWebView {
        let contentController = WKUserContentController()
        contentController.add(bridge, name: "arkitScanBridge")
        let config = WKWebViewConfiguration()
        config.userContentController = contentController
        config.allowsInlineMediaPlayback = true
        config.defaultWebpagePreferences.preferredContentMode = .mobile

        let webView = WKWebView(frame: .zero, configuration: config)
        bridge.webView = webView
        webView.navigationDelegate = context.coordinator
        webView.isOpaque = false
        webView.backgroundColor = .systemBackground
        webView.scrollView.contentInsetAdjustmentBehavior = .never
        
        let refreshControl = UIRefreshControl()
        refreshControl.addTarget(context.coordinator, action: #selector(Coordinator.handleRefreshControl), for: .valueChanged)
        webView.scrollView.refreshControl = refreshControl
        
        webView.load(URLRequest(url: url))
        return webView
    }

    func updateUIView(_ uiView: WKWebView, context: Context) {
        if let currentURL = uiView.url?.absoluteString, currentURL != url.absoluteString {
            uiView.load(URLRequest(url: url))
        }
    }
}

public struct RootView: View {
    @StateObject private var scanBridge = ArkitScanBridge()
    @AppStorage("clinicServerURL") private var serverURLString: String = "http://149.118.63.240"
    @State private var showingSettings = false
    @State private var reloadTrigger = UUID()

    public init() {
        let defaultURL = "http://149.118.63.240"
        let current = UserDefaults.standard.string(forKey: "clinicServerURL") ?? ""
        if current.isEmpty || current.contains("trycloudflare.com") {
            UserDefaults.standard.set(defaultURL, forKey: "clinicServerURL")
        }
    }

    public var body: some View {
        ZStack(alignment: .bottomTrailing) {
            let activeURLString = serverURLString.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty 
                ? "https://lens-inside-silence-bearing.trycloudflare.com" 
                : serverURLString.trimmingCharacters(in: .whitespacesAndNewlines)

            if let url = URL(string: activeURLString) {
                WebView(url: url, bridge: scanBridge, reloadTrigger: reloadTrigger)
                    .id(reloadTrigger)
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
            }

            // Nút cài đặt nhỏ gọn, tinh tế ở góc dưới phải
            Button {
                showingSettings = true
            } label: {
                HStack(spacing: 6) {
                    Image(systemName: "gearshape.fill")
                    Text("Đổi Kênh Máy Chủ")
                        .font(.caption2.bold())
                }
                .foregroundColor(.white)
                .padding(.horizontal, 12)
                .padding(.vertical, 8)
                .background(Color.black.opacity(0.75), in: Capsule())
                .shadow(radius: 4)
            }
            .padding(.trailing, 16)
            .padding(.bottom, 70)
        }
        .sheet(isPresented: $showingSettings) {
            NavigationView {
                Form {
                    Section(header: Text("Chọn Kênh Kết Nối")) {
                        Button(action: {
                            serverURLString = "http://149.118.63.240"
                            UserDefaults.standard.set(serverURLString, forKey: "clinicServerURL")
                            reloadTrigger = UUID()
                            scanBridge.webView?.load(URLRequest(url: URL(string: serverURLString)!))
                            showingSettings = false
                        }) {
                            HStack {
                                Image(systemName: "bolt.fill").foregroundColor(.orange)
                                VStack(alignment: .leading) {
                                    Text("Kênh 1: Máy chủ IP Trực tiếp (Khuyên dùng)").bold()
                                    Text("http://149.118.63.240").font(.caption).foregroundColor(.secondary)
                                }
                                Spacer()
                            }
                        }

                        Button(action: {
                            serverURLString = "https://lens-inside-silence-bearing.trycloudflare.com"
                            UserDefaults.standard.set(serverURLString, forKey: "clinicServerURL")
                            reloadTrigger = UUID()
                            scanBridge.webView?.load(URLRequest(url: URL(string: serverURLString)!))
                            showingSettings = false
                        }) {
                            HStack {
                                Image(systemName: "lock.shield.fill").foregroundColor(.blue)
                                VStack(alignment: .leading) {
                                    Text("Kênh 2: Cloudflare HTTPS Tunnel").bold()
                                    Text("https://lens-inside-silence-bearing.trycloudflare.com").font(.caption).foregroundColor(.secondary)
                                }
                                Spacer()
                            }
                        }
                    }

                    Section(header: Text("Tùy Chỉnh Địa Chỉ Máy Chủ")) {
                        TextField("http://...", text: $serverURLString)
                            .keyboardType(.URL)
                            .autocorrectionDisabled()
                            .textInputAutocapitalization(.never)
                        
                        Button(action: {
                            let clean = serverURLString.trimmingCharacters(in: .whitespacesAndNewlines)
                            UserDefaults.standard.set(clean, forKey: "clinicServerURL")
                            reloadTrigger = UUID()
                            scanBridge.webView?.load(URLRequest(url: URL(string: clean) ?? URL(string: "http://149.118.63.240")!))
                            showingSettings = false
                        }) {
                            HStack {
                                Spacer()
                                Image(systemName: "arrow.clockwise")
                                Text("Lưu & Tải Lại Trang")
                                    .bold()
                                Spacer()
                            }
                        }
                    }
                }
                .navigationTitle("Cài đặt máy chủ")
                .toolbar {
                    Button("Đóng") { showingSettings = false }
                }
            }
        }
    }
}
