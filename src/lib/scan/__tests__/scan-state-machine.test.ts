import { describe, it, expect } from "vitest";
import { createInitialScanState, scanTick, TARGET_ANGLES } from "../scan-state-machine";
import type { FrameEvaluation } from "../frame-evaluator";
import { STABILITY } from "../scan-constants";

function ev(overrides: Partial<FrameEvaluation> = {}): FrameEvaluation {
  return {
    accepted: true,
    score: 100,
    yaw: 0,
    pitch: 0,
    roll: 0,
    brightness: 140,
    sharpness: 40,
    motion: 0.5,
    faceCoverage: 0.5,
    landmarkStability: 0.95,
    angleError: 0,
    rejectReasons: [],
    ...overrides,
  };
}

/** Drives the state machine through PRECHECK into the FRONT stage. */
function passPrecheck(state = createInitialScanState()) {
  let s = state;
  let t = 1000;
  for (let i = 0; i < STABILITY.PRECHECK_STABLE_TICKS; i++) {
    const r = scanTick(s, ev({ yaw: 0, angleError: 0 }), t);
    s = r.state;
    t += 200;
  }
  return { state: s, t };
}

describe("scanTick", () => {
  it("stays in precheck until stable, and resets progress when quality drops", () => {
    let s = createInitialScanState();
    let t = 0;
    // Accumulate one tick short of the current threshold (0 ticks if the
    // threshold itself is 1 -- still a valid, trivially-true check of the
    // starting state either way).
    for (let i = 0; i < STABILITY.PRECHECK_STABLE_TICKS - 1; i++) {
      s = scanTick(s, ev({ angleError: 0 }), (t += 200)).state;
    }
    expect(s.stage).toBe("precheck");
    const stableCountBeforeDrop = s.precheckStableCount;
    // ...then a rejected tick resets progress downward instead of advancing.
    s = scanTick(s, ev({ accepted: false, rejectReasons: ["no_face"] }), (t += 200)).state;
    expect(s.stage).toBe("precheck");
    expect(s.precheckStableCount).toBeLessThanOrEqual(stableCountBeforeDrop);
    expect(s.precheckStableCount).toBe(Math.max(0, stableCountBeforeDrop - 1));
  });

  it("advances precheck -> front only after enough consecutive stable good ticks", () => {
    const { state } = passPrecheck();
    expect(state.stage).toBe("front");
    expect(state.targetIndex).toBe(0);
  });

  it("14. a sudden jump straight to +45deg while still targeting FRONT does not bank frames or skip ahead to right_45", () => {
    let { state, t } = passPrecheck();
    expect(state.stage).toBe("front");

    // Simulate the user (or a bogus reading) jumping straight to +45 for many ticks.
    for (let i = 0; i < 20; i++) {
      const r = scanTick(state, ev({ yaw: 45, angleError: 45 }), (t += 200));
      state = r.state;
      expect(r.capture).toBeNull();
      expect(r.targetCompleted).toBeNull();
    }

    // Still stuck on "front" -- FRONT's own requiredFrames was never satisfied,
    // and no intermediate/right-side target was ever silently marked done.
    expect(state.stage).toBe("front");
    expect(state.acceptedByTarget.front).toBe(0);
    expect(state.acceptedByTarget.right_45).toBe(0);
  });

  it("15/16 groundwork: completing FRONT transitions into left_transition, never directly into left_15", () => {
    let { state, t } = passPrecheck();
    const front = TARGET_ANGLES[0];
    for (let i = 0; i < front.requiredFrames; i++) {
      // Each capture requires its own freshly-accumulated stability run.
      for (let k = 0; k < STABILITY.CHECKPOINT_STABLE_TICKS; k++) {
        const r = scanTick(state, ev({ yaw: 0, angleError: 0 }), (t += 300));
        state = r.state;
      }
    }
    expect(state.acceptedByTarget.front).toBe(front.requiredFrames);
    expect(state.stage).toBe("left_transition");
  });

  it("does not capture two checkpoint frames closer together than the minimum spacing", () => {
    let { state, t } = passPrecheck();
    // Reach stability once.
    for (let k = 0; k < STABILITY.CHECKPOINT_STABLE_TICKS; k++) {
      const r = scanTick(state, ev({ yaw: 0, angleError: 0 }), (t += 300));
      state = r.state;
    }
    expect(state.acceptedByTarget.front).toBe(1);
    // Immediately (same instant) fire more "stable" ticks -- capture must not
    // fire again until STABILITY.MIN_MS_BETWEEN_CAPTURES has elapsed AND a
    // fresh stable run has re-accumulated.
    for (let k = 0; k < STABILITY.CHECKPOINT_STABLE_TICKS; k++) {
      const r = scanTick(state, ev({ yaw: 0, angleError: 0 }), t);
      state = r.state;
      expect(r.capture).toBeNull();
    }
    expect(state.acceptedByTarget.front).toBe(1);
  });

  it("full happy path reaches quality_review with every target satisfied", () => {
    let { state, t } = passPrecheck();
    let safety = 0;
    while (state.stage !== "quality_review" && safety < 5000) {
      safety++;
      const target = TARGET_ANGLES[Math.min(state.targetIndex, TARGET_ANGLES.length - 1)];
      const yaw = target.targetYaw; // pretend the user is exactly on the current/next checkpoint
      const r = scanTick(state, ev({ yaw, angleError: 0 }), (t += 200));
      state = r.state;
    }
    expect(state.stage).toBe("quality_review");
    for (const target of TARGET_ANGLES) {
      expect(state.acceptedByTarget[target.id]).toBeGreaterThanOrEqual(target.requiredFrames);
    }
  });
});
