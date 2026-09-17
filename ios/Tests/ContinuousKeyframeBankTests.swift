import XCTest
@testable import DrVanTruongScanner

final class ContinuousKeyframeBankTests: XCTestCase {
    func testNaturalSweepSelectsIndependentFramesAndReturnsToCenter() {
        var bank = ContinuousKeyframeBank()
        let yaws = [0, -6, -12, -18, -24, -30, -24, -18, -12, -6, 0, 6, 12, 18, 24, 30, 24, 18, 12, 6, 0]
        for (i, yaw) in yaws.enumerated() {
            _ = bank.bank(ContinuousSample(hash: "physical-\(i)", timestamp: Double(i), yaw: Double(yaw), pitch: 0, score: 1))
        }
        XCTAssertEqual(bank.phase, .ready)
        let selected = bank.selected()
        XCTAssertTrue((7...10).contains(selected.count))
        XCTAssertEqual(Set(selected.map(\.hash)).count, selected.count)
        XCTAssertEqual(Set(selected.map(\.timestamp)).count, selected.count)
        XCTAssertEqual(bank.clinical().count, 4)
    }

    func testDuplicateDoesNotAdvanceProgressAndBetterCandidateReplacesFirst() {
        var bank = ContinuousKeyframeBank()
        let first = ContinuousSample(hash: "first", timestamp: 1, yaw: 0, pitch: 0, score: 0)
        XCTAssertTrue(bank.bank(first))
        XCTAssertFalse(bank.bank(first))
        XCTAssertFalse(bank.bank(ContinuousSample(hash: "clone", timestamp: 1, yaw: -30, pitch: 0, score: 3)))
        XCTAssertEqual(bank.phase, .left)
        for i in 2...5 {
            _ = bank.bank(ContinuousSample(hash: "better-\(i)", timestamp: Double(i), yaw: 0, pitch: 0, score: Double(i)))
        }
        XCTAssertFalse(bank.samples.contains(first))
        XCTAssertEqual(bank.samples.count, 3)
        XCTAssertEqual(bank.phase, .left)
    }

    func testOnlyOneUploadAndOneTerminalCallbackPerAttempt() {
        var upload = ScanUploadAttempt()
        let first = upload.begin()!
        XCTAssertNil(upload.begin())
        XCTAssertTrue(upload.finish(first))
        XCTAssertFalse(upload.finish(first))
        let retry = upload.begin()!
        XCTAssertNotEqual(first, retry)
        XCTAssertFalse(upload.finish(first))
        XCTAssertTrue(upload.isCurrent(retry))
    }

    func testResetOrCloseIgnoresLateCompletionFromPreviousScan() {
        var upload = ScanUploadAttempt()
        let old = upload.begin()!
        upload.invalidate()
        let new = upload.begin()!
        XCTAssertFalse(upload.isCurrent(old))
        XCTAssertFalse(upload.finish(old))
        XCTAssertTrue(upload.isCurrent(new))
    }

    func testReconstructionRejectionIsTerminalInsteadOfPollingAgain() {
        for status in ["needs_rescan", "failed", "rejected"] {
            XCTAssertEqual(ScanReconstructionState(status: status), .rejected)
        }
        XCTAssertEqual(ScanReconstructionState(status: "ready"), .ready)
        XCTAssertEqual(ScanReconstructionState(status: "processing"), .processing)
        XCTAssertEqual(ScanReconstructionState(status: nil), .processing)
    }

    func testClinicalFrontPrefersNeutralPitchOverTiltedYawMatch() {
        var bank = ContinuousKeyframeBank()
        _ = bank.bank(ContinuousSample(hash: "tilted", timestamp: 1, yaw: 0, pitch: 24, score: 1))
        _ = bank.bank(ContinuousSample(hash: "neutral", timestamp: 2, yaw: 2, pitch: 0, score: 1))
        XCTAssertEqual(bank.clinical()["front"]?.hash, "neutral")
    }
}
