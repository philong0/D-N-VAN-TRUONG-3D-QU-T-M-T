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
        if uiView.url?.host != url.host || uiView.url == nil {
            uiView.load(URLRequest(url: url))
        }
    }
}

public struct RootView: View {
    @StateObject private var scanBridge = ArkitScanBridge()
    @AppStorage("clinicServerURL") private var serverURLString: String = "https://packets-leading-fonts-upc.trycloudflare.com"
    @State private var showingSettings = false
    @State private var reloadTrigger = UUID()

    public init() {}

    public var body: some View {
        ZStack(alignment: .bottomTrailing) {
            if let url = URL(string: serverURLString.trimmingCharacters(in: .whitespacesAndNewlines)), !serverURLString.isEmpty {
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
                    .font(.system(size: 16, weight: .bold))
                    .foregroundColor(.white)
                    .padding(10)
                    .background(Color.black.opacity(0.65), in: Circle())
                    .shadow(radius: 4)
            }
            .padding(.trailing, 16)
            .padding(.bottom, 70)
        }
        .sheet(isPresented: $showingSettings) {
            NavigationView {
                Form {
                    Section(header: Text("Địa chỉ máy chủ (Web Studio)")) {
                        TextField("https://...", text: $serverURLString)
                            .keyboardType(.URL)
                            .autocorrectionDisabled()
                            .textInputAutocapitalization(.never)
                        Text("Lưu ý: Nếu server dùng tunnel tạm trycloudflare.com, địa chỉ sẽ thay đổi khi khởi động lại. Cập nhật URL mới tại đây khi cần.")
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                    
                    Section {
                        Button(action: {
                            let clean = serverURLString.trimmingCharacters(in: .whitespacesAndNewlines)
                            UserDefaults.standard.set(clean, forKey: "clinicServerURL")
                            reloadTrigger = UUID()
                            scanBridge.webView?.load(URLRequest(url: URL(string: clean) ?? URL(string: "http://localhost:3000")!))
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
                    Button("Xong") { showingSettings = false }
                }
            }
        }
    }
}
