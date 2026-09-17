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
    @AppStorage("clinicServerURL") private var serverURLString: String = "https://lens-inside-silence-bearing.trycloudflare.com"
    @State private var showingSettings = false
    @State private var reloadTrigger = UUID()

    // D-urlwipe — bản trước ở đây tự động GHI ĐÈ link Cloudflare Tunnel đang
    // hoạt động về lại địa chỉ IP thô "http://149.118.63.240" mỗi lần app
    // khởi động (nếu giá trị đã lưu trống hoặc chứa "trycloudflare.com") --
    // đây là nguyên nhân THẬT của lỗi "This page couldn't load" xảy ra CHẮC
    // CHẮN sau mỗi lần cài mới (UserDefaults trống): IP thô đó chỉ tự kiểm
    // tra được thành công khi chạy curl NGAY TRÊN máy chủ, không đại diện
    // cho khả năng điện thoại thật (mạng di động/WiFi khác) truy cập được
    // qua Internet -- đúng lý do Cloudflare Tunnel được dùng ngay từ đầu.
    // D-cleanupstale — máy đã từng cài bản lỗi trước đó có thể vẫn còn lưu
    // sẵn "http://149.118.63.240" trong UserDefaults từ lần chạy trước; chỉ
    // xoá đoạn GHI ĐÈ (ở trên) không tự dọn giá trị xấu ĐÃ LƯU SẴN đó, vì
    // @AppStorage chỉ dùng giá trị mặc định khi CHƯA có gì lưu. Dọn đúng 1
    // lần duy nhất, chỉ khi giá trị đã lưu CHÍNH XÁC là IP thô nói trên
    // (không đụng vào bất kỳ URL nào khác người dùng có thể đã tự đặt).
    public init() {
        if UserDefaults.standard.string(forKey: "clinicServerURL") == "http://149.118.63.240" {
            UserDefaults.standard.removeObject(forKey: "clinicServerURL")
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
                        // D-urlwipe — "IP Trực tiếp" chỉ thật sự truy cập được khi
                        // đứng cùng mạng nội bộ với máy chủ; điện thoại thật (mạng
                        // di động/WiFi khác) sẽ luôn ra "This page couldn't load"
                        // với kênh này -- không còn gắn nhãn "Khuyên dùng" nữa,
                        // đổi thứ tự để Cloudflare Tunnel (đã xác nhận hoạt động
                        // thật qua Internet) là lựa chọn chính.
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
                                    Text("Kênh 1: Cloudflare HTTPS Tunnel (Khuyên dùng)").bold()
                                    Text("https://lens-inside-silence-bearing.trycloudflare.com").font(.caption).foregroundColor(.secondary)
                                }
                                Spacer()
                            }
                        }

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
                                    Text("Kênh 2: Máy chủ IP Trực tiếp (chỉ dùng khi cùng mạng nội bộ)").bold()
                                    Text("http://149.118.63.240").font(.caption).foregroundColor(.secondary)
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
