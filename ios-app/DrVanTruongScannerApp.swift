import SwiftUI

@main
struct DrVanTruongScannerApp: App {
    var body: some Scene {
        WindowGroup {
            MainClinicAppView()
        }
    }
}

/// Ứng dụng tích hợp toàn diện: Nạp trực tiếp toàn bộ giao diện quản lý bệnh nhân,
/// 3D Studio và kích hoạt cảm biến TrueDepth Face ID quét 3D ngay trong 1 App duy nhất.
struct MainClinicAppView: View {
    @AppStorage("clinicServerURL") private var serverURL: String = "https://eddie-screw-influence-goto.trycloudflare.com"
    @State private var showingSettings = false

    var body: some View {
        ZStack(alignment: .topTrailing) {
            if let url = URL(string: serverURL) {
                FaceScannerView(initialPath: "/patients", webAppBaseURL: url)
                    .ignoresSafeArea()
            } else {
                VStack(spacing: 16) {
                    Text("Vui lòng cấu hình URL máy chủ").font(.headline)
                    Button("Cài đặt URL") { showingSettings = true }
                }
            }
        }
        .sheet(isPresented: $showingSettings) {
            NavigationView {
                Form {
                    Section("Cấu hình máy chủ phòng khám") {
                        TextField("https://...", text: $serverURL)
                            .keyboardType(.URL)
                            .autocorrectionDisabled()
                            .textInputAutocapitalization(.never)
                    }
                }
                .navigationTitle("Cài đặt Máy chủ")
                .toolbar {
                    Button("Lưu") { showingSettings = false }
                }
            }
        }
    }
}
