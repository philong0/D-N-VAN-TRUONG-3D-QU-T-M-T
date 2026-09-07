import { describe, it, expect } from "vitest";
import { selectReconstructionFrames, type BurstFrameCandidate } from "../reconstruction-frame-selection";

function c(fileName: string, yaw: number | null | undefined, overrides: Partial<BurstFrameCandidate> = {}): BurstFrameCandidate {
  return { fileName, yaw, qualityScore: 80, sharpness: 40, motion: 0.5, timestampMs: 0, ...overrides };
}

describe("selectReconstructionFrames", () => {
  it("selects correct coverage from a SHUFFLED array -- order in the array must not matter", () => {
    // Deliberately out of order, and the array's own first/last elements
    // are neither -45 nor +45 -- a positional (0%/25%/50%/75%) selector
    // would pick the wrong frames entirely.
    const shuffled: BurstFrameCandidate[] = [
      c("f_plus30.jpg", 30),
      c("f_minus45.jpg", -45),
      c("f_zero.jpg", 0),
      c("f_plus15.jpg", 15),
      c("f_minus15.jpg", -15),
      c("f_plus45.jpg", 45),
      c("f_minus30.jpg", -30),
    ];
    const { selected, report } = selectReconstructionFrames(shuffled);
    expect(report.selectionMethod).toBe("yaw_quality_overlap");
    expect(selected.angle1).toBe("f_zero.jpg");
    expect(selected.angle2).toBe("f_minus45.jpg");
    expect(selected.angle4).toBe("f_plus45.jpg");
    expect(report.coverage.left).toBe(true);
    expect(report.coverage.right).toBe(true);
    expect(report.coverage.front).toBe(true);
  });

  it("the array's first element is not necessarily assigned to angle2/left, nor the last to angle4/right", () => {
    const candidates: BurstFrameCandidate[] = [
      c("first.jpg", 45), // array-first is actually the RIGHT-most real angle
      c("mid.jpg", 0),
      c("last.jpg", -45), // array-last is actually the LEFT-most real angle
    ];
    const { selected } = selectReconstructionFrames(candidates);
    expect(selected.angle2).toBe("last.jpg"); // left slot correctly finds the -45 frame regardless of position
    expect(selected.angle4).toBe("first.jpg"); // right slot correctly finds the +45 frame regardless of position
  });

  it("when several frames share nearly the same yaw, the higher qualityScore one wins", () => {
    const candidates: BurstFrameCandidate[] = [
      c("front_blurry.jpg", 1, { qualityScore: 30, sharpness: 5, motion: 4 }),
      c("front_sharp.jpg", -1, { qualityScore: 95, sharpness: 60, motion: 0.2 }),
    ];
    const { selected } = selectReconstructionFrames(candidates);
    expect(selected.angle1).toBe("front_sharp.jpg");
  });

  it("reports duplicateCount for near-identical yaw+timestamp frames instead of silently keeping them all", () => {
    const candidates: BurstFrameCandidate[] = [
      c("a.jpg", -45, { timestampMs: 1000 }),
      c("b.jpg", -44, { timestampMs: 1100 }), // same real moment as a.jpg
      c("c.jpg", 0, { timestampMs: 5000 }),
      c("d.jpg", 45, { timestampMs: 9000 }),
    ];
    const { report } = selectReconstructionFrames(candidates);
    expect(report.duplicateCount).toBeGreaterThan(0);
  });

  it("reports missing LEFT coverage honestly instead of claiming completeness", () => {
    const candidates: BurstFrameCandidate[] = [c("front.jpg", 0), c("right.jpg", 45)];
    const { report } = selectReconstructionFrames(candidates);
    expect(report.coverage.left).toBe(false);
    expect(report.coverage.right).toBe(true);
  });

  it("reports missing RIGHT coverage honestly instead of claiming completeness", () => {
    const candidates: BurstFrameCandidate[] = [c("front.jpg", 0), c("left.jpg", -45)];
    const { report } = selectReconstructionFrames(candidates);
    expect(report.coverage.right).toBe(false);
    expect(report.coverage.left).toBe(true);
  });

  it("never fabricates yaw from array position when NO frame has real pose metadata", () => {
    const candidates: BurstFrameCandidate[] = [
      c("burst_0000.jpg", null),
      c("burst_0001.jpg", undefined),
      c("burst_0002.jpg", null),
    ];
    const { selected, report } = selectReconstructionFrames(candidates);
    expect(selected).toEqual({});
    expect(report.selectionMethod).toBe("positional_fallback_no_metadata");
    expect(report.selectedFrameCount).toBe(0);
  });

  it("only fills the profile (angle3) slot with a genuine ~90deg frame, never a 45deg one", () => {
    const noProfile: BurstFrameCandidate[] = [c("front.jpg", 0), c("left45.jpg", -45), c("right45.jpg", 45)];
    const { selected: withoutProfile, report: reportNoProfile } = selectReconstructionFrames(noProfile);
    expect(withoutProfile.angle3).toBeUndefined();
    expect(reportNoProfile.coverage.profile).toBe(false);

    const withProfile: BurstFrameCandidate[] = [...noProfile, c("left_profile.jpg", -88)];
    const { selected, report } = selectReconstructionFrames(withProfile);
    expect(selected.angle3).toBe("left_profile.jpg");
    expect(report.coverage.profile).toBe(true);
  });

  it("mixes real checkpoint AND transition frames from the whole candidate pool, not just 7 fixed frames", () => {
    const many: BurstFrameCandidate[] = [
      c("front.jpg", 0, { qualityScore: 90 }),
      c("t1.jpg", -22, { qualityScore: 70, targetId: null }),
      c("left45_ok.jpg", -45, { qualityScore: 60 }),
      c("left45_better.jpg", -46, { qualityScore: 92 }),
      c("t2.jpg", 25, { qualityScore: 65, targetId: null }),
      c("right45.jpg", 45, { qualityScore: 85 }),
    ];
    const { selected } = selectReconstructionFrames(many);
    // Among the two real -45-ish candidates, the higher-quality one wins.
    expect(selected.angle2).toBe("left45_better.jpg");
  });
});
