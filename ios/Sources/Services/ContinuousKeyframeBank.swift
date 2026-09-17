import Foundation

/// Pure selection/state logic. A sample enters only AFTER synchronized bytes
/// have been serialized and validated; UI ticks never cause capture.
struct ContinuousSample: Equatable {
    let hash: String
    let timestamp: Double
    let yaw: Double
    let pitch: Double
    let score: Double
}

struct ContinuousKeyframeBank {
    enum Phase: Int { case center, left, right, returnCenter, ready }
    private(set) var phase = Phase.center
    private(set) var samples: [ContinuousSample] = []
    private(set) var acceptedCount = 0
    private var identities = Set<String>()
    private var timestamps = Set<Double>()

    var guidance: String {
        switch phase {
        case .center: return "Nhìn thẳng vào camera"
        case .left: return "Xoay đầu chậm sang trái"
        case .right: return "Xoay đầu chậm sang phải"
        case .returnCenter: return "Trở về giữa, nhìn thẳng vào camera"
        case .ready: return "Đang hoàn tất dữ liệu quét"
        }
    }

    mutating func bank(_ sample: ContinuousSample) -> Bool {
        guard sample.timestamp.isFinite, sample.yaw.isFinite, sample.pitch.isFinite,
              sample.score.isFinite, !identities.contains(sample.hash),
              !timestamps.contains(sample.timestamp), phase != .ready else { return false }
        identities.insert(sample.hash); timestamps.insert(sample.timestamp)
        acceptedCount += 1
        samples.append(sample)
        // Keep the three best observations per broad yaw neighborhood, not
        // the first one. Bounded memory while retaining temporal alternatives.
        let bucket = Int(floor(sample.yaw / 6))
        let same = samples.filter { Int(floor($0.yaw / 6)) == bucket }.sorted { $0.score > $1.score }
        if same.count > 3 {
            let retained = Set(same.prefix(3).map(\.hash))
            samples.removeAll { Int(floor($0.yaw / 6)) == bucket && !retained.contains($0.hash) }
        }
        switch phase {
        case .center: if abs(sample.yaw) <= 10 { phase = .left }
        case .left: if sample.yaw <= -25 { phase = .right }
        case .right: if sample.yaw >= 25 { phase = .returnCenter }
        case .returnCenter:
            if abs(sample.yaw) <= 10 && selected().count >= 7 && clinical().count == 4 { phase = .ready }
        case .ready: break
        }
        return true
    }

    func selected() -> [ContinuousSample] {
        guard samples.count >= 7, let left = samples.min(by: { $0.yaw < $1.yaw }),
              let right = samples.max(by: { $0.yaw < $1.yaw }), left.yaw <= -25, right.yaw >= 25 else { return [] }
        let ranked = samples.sorted { $0.score == $1.score ? $0.timestamp < $1.timestamp : $0.score > $1.score }
        var chosen: [ContinuousSample] = []
        // Broad adaptive targets span the actually observed sweep. Every
        // physical observation can win at most once, based on measured quality.
        for i in 0..<9 {
            let target = left.yaw + (right.yaw-left.yaw) * Double(i) / 8
            let pool = ranked.filter { s in
                !chosen.contains(where: { $0.hash == s.hash || abs($0.yaw-s.yaw) < 3 })
                    && abs(s.yaw-target) <= 8
            }
            if let best = pool.max(by: {
                $0.score - abs($0.yaw-target)/30 < $1.score - abs($1.yaw-target)/30
            }) { chosen.append(best) }
        }
        return chosen.count >= 7 ? chosen.sorted { $0.timestamp < $1.timestamp } : []
    }

    func clinical() -> [String: ContinuousSample] {
        guard let minYaw = samples.map(\.yaw).min(), let maxYaw = samples.map(\.yaw).max() else { return [:] }
        let targets: [(String, Double)] = [("front", 0), ("left_oblique", minYaw * 0.55),
                                           ("left_lateral", minYaw), ("right_oblique", maxYaw * 0.7)]
        var result: [String: ContinuousSample] = [:]
        for (role, target) in targets {
            let candidates = samples.filter { sample in
                !result.values.contains(where: { $0.hash == sample.hash }) && abs(sample.yaw-target) <= 10
            }
            if let best = candidates.max(by: {
                $0.score-abs($0.yaw-target)/20-abs($0.pitch)/30 < $1.score-abs($1.yaw-target)/20-abs($1.pitch)/30
            }) { result[role] = best }
        }
        return result
    }
}

/// Guards duplicate uploads and callbacks from a scan that has been reset/closed.
struct ScanUploadAttempt {
    private var current: UUID?
    mutating func begin() -> UUID? {
        guard current == nil else { return nil }
        let id = UUID()
        current = id
        return id
    }
    func isCurrent(_ id: UUID) -> Bool { current == id }
    mutating func finish(_ id: UUID) -> Bool {
        guard isCurrent(id) else { return false }
        current = nil
        return true
    }
    mutating func invalidate() { current = nil }
}

enum ScanReconstructionState {
    case processing, ready, rejected
    init(status: String?) {
        switch status {
        case "ready": self = .ready
        case "failed", "needs_rescan", "rejected": self = .rejected
        default: self = .processing
        }
    }
}

enum ScannerRelease {
    static let identifier = "SCAN-20260917-R3"
}
