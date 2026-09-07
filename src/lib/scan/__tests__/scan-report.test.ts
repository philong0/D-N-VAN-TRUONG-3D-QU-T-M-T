import { describe, it, expect } from "vitest";
import { buildScanReport, emptyRejectTally, tallyReject, type CapturedFrameRecord } from "../scan-report";
import { TARGET_ANGLES } from "../scan-state-machine";
import { FINAL_REVIEW } from "../scan-constants";

function record(overrides: Partial<CapturedFrameRecord>): CapturedFrameRecord {
  return {
    kind: "checkpoint",
    targetId: "front",
    timestampMs: 0,
    yaw: 0,
    pitch: 0,
    roll: 0,
    faceBounds: { left: 0.3, top: 0.3, right: 0.7, bottom: 0.7 },
    faceCoverage: 0.5,
    brightness: 140,
    sharpness: 40,
    motion: 0.3,
    landmarkStability: 0.95,
    angleError: 0,
    qualityScore: 95,
    ...overrides,
  };
}

/** Builds a full, real set of checkpoint records satisfying every target's requiredFrames, spanning the full yaw sweep. */
function fullCoverageRecords(): CapturedFrameRecord[] {
  const records: CapturedFrameRecord[] = [];
  let ts = 0;
  for (const target of TARGET_ANGLES) {
    for (let i = 0; i < target.requiredFrames; i++) {
      ts += 1000; // well beyond the min spacing, so this is never flagged as a duplicate
      records.push(record({ targetId: target.id, yaw: target.targetYaw, timestampMs: ts }));
    }
  }
  return records;
}

describe("buildScanReport", () => {
  it("15. reports fail when a target is missing frames, and never claims completion", () => {
    const records = fullCoverageRecords().filter((r) => r.targetId !== "left_45"); // drop an entire target
    const report = buildScanReport(records, emptyRejectTally(), records.length, 0, 10000);
    expect(report.finalQuality).toBe("fail");
    expect(report.failReasons.some((r) => r.includes("TRÁI 45"))).toBe(true);
  });

  it("16. reports pass when every target has enough frames and coverage reaches both sides", () => {
    const records = fullCoverageRecords();
    const report = buildScanReport(records, emptyRejectTally(), records.length, 0, 20000);
    expect(report.finalQuality).toBe("pass");
    expect(report.coverage.left).toBe(true);
    expect(report.coverage.right).toBe(true);
    expect(report.yawRange.min).toBeLessThanOrEqual(FINAL_REVIEW.MIN_LEFT_YAW_DEG);
    expect(report.yawRange.max).toBeGreaterThanOrEqual(FINAL_REVIEW.MIN_RIGHT_YAW_DEG);
  });

  it("17. flags near-duplicate frames for the same target instead of silently counting them toward the minimum", () => {
    const target = TARGET_ANGLES[0];
    const records = [
      record({ targetId: target.id, timestampMs: 1000 }),
      record({ targetId: target.id, timestampMs: 1050 }), // far under MIN_MS_BETWEEN_CAPTURES
      record({ targetId: target.id, timestampMs: 1100 }),
    ];
    const report = buildScanReport(records, emptyRejectTally(), records.length, 0, 5000);
    expect(report.duplicateCount).toBeGreaterThan(0);
  });

  it("tallies reject reasons accurately across many ticks", () => {
    let tally = emptyRejectTally();
    tally = tallyReject(tally, ["too_dark"]);
    tally = tallyReject(tally, ["too_dark", "blurry"]);
    tally = tallyReject(tally, []);
    expect(tally.too_dark).toBe(2);
    expect(tally.blurry).toBe(1);

    const report = buildScanReport([], tally, 3, 0, 1000);
    expect(report.brightnessFailures).toBe(2);
    expect(report.sharpnessFailures).toBe(1);
    expect(report.rejectedFrames).toBe(3);
  });
});
