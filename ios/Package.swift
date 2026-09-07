// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "DrVanTruongScanner",
    platforms: [
        .iOS(.v16)
    ],
    products: [
        .library(
            name: "DrVanTruongScanner",
            targets: ["DrVanTruongScanner"]
        ),
    ],
    targets: [
        .target(
            name: "DrVanTruongScanner",
            path: "Sources"
        ),
        .testTarget(
            name: "DrVanTruongScannerTests",
            dependencies: ["DrVanTruongScanner"],
            path: "Tests"
        ),
    ]
)
