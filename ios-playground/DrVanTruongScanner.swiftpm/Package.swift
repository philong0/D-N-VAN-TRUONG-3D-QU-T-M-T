// swift-tools-version: 5.9

// UNTESTED — chuẩn bị để mở thử bằng Swift Playgrounds trên iPad, thay cho
// Xcode trên Mac. Không thể build-thử ở môi trường này (server Linux, không
// có Xcode/Swift Playgrounds), nên cú pháp `Product.iOSApplication` bên dưới
// là dựa trên tài liệu công khai của Apple, chưa được biên dịch thật để xác
// nhận 100% đúng. Nếu Playgrounds báo lỗi ngay khi mở, khả năng cao chỉ cần
// chỉnh nhỏ trong chính giao diện cài đặt của Playgrounds (tên hiển thị, mô
// tả quyền Camera, icon) chứ không cần sửa lại toàn bộ file.

import PackageDescription

let package = Package(
    name: "DrVanTruongScanner",
    platforms: [
        .iOS(.v16)
    ],
    products: [
        .iOSApplication(
            name: "DrVanTruongScanner",
            targets: ["AppModule"],
            bundleIdentifier: "com.drvantruong.scanner",
            teamIdentifier: nil,
            displayVersion: "1.0",
            bundleVersion: "1",
            appIcon: .placeholder(icon: .camera),
            accentColor: .presetColor(.blue),
            supportedDeviceFamilies: [.phone, .pad],
            supportedInterfaceOrientations: [.portrait],
            capabilities: [
                .camera(purposeString: "Ứng dụng cần quyền truy cập Camera TrueDepth để quét cấu trúc 3D khuôn mặt và điểm mốc giải phẫu phục vụ tư vấn thẩm mỹ.")
            ]
        )
    ],
    targets: [
        .executableTarget(
            name: "AppModule",
            path: "Sources"
        )
    ]
)
