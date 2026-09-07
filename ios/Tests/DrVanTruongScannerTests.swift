import XCTest
@testable import DrVanTruongScanner

final class DrVanTruongScannerTests: XCTestCase {
    
    func testScanAngleStepCases() {
        let steps = ScanAngleStep.allCases
        XCTAssertEqual(steps.count, 5, "Scan session requires exactly 5 clinical angles")
        XCTAssertEqual(steps[0], .front)
        XCTAssertEqual(steps[1], .left45)
        XCTAssertEqual(steps[2], .leftProfile)
        XCTAssertEqual(steps[3], .right45)
        XCTAssertEqual(steps[4], .rightProfile)
    }
    
    func testManifestEncoding() throws {
        let manifest = ScanPackageManifestDTO(
            schemaVersion: "2.0.0",
            captureSource: "native_ios",
            deviceModel: "iPhone 14 Pro",
            systemVersion: "iOS 17.5",
            hasTrueDepth: true,
            patientId: "patient-123",
            sessionId: "session-456",
            capturedAt: "2026-08-30T05:00:00Z",
            frames: []
        )
        
        let encoder = JSONEncoder()
        let data = try encoder.encode(manifest)
        let decoder = JSONDecoder()
        let decoded = try decoder.decode(ScanPackageManifestDTO.self, from: data)
        
        XCTAssertEqual(decoded.schemaVersion, "2.0.0")
        XCTAssertEqual(decoded.captureSource, "native_ios")
        XCTAssertEqual(decoded.hasTrueDepth, true)
        XCTAssertEqual(decoded.patientId, "patient-123")
    }
}
