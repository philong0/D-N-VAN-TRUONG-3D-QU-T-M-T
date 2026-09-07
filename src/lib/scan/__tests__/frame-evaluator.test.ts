import { describe, it, expect } from "vitest";
import { evaluateFrame, type FrameMeasurement } from "../frame-evaluator";
import { POSE, IMAGE_QUALITY, STABILITY } from "../scan-constants";

/** A frame that should pass every check, centered, well-lit, sharp, stable, looking straight at the camera. */
function goodMeasurement(overrides: Partial<FrameMeasurement> = {}): FrameMeasurement {
  return {
    faceDetected: true,
    landmarksReal: true,
    yaw: 0,
    pitch: 0,
    roll: 0,
    faceBounds: { left: 0.35, top: 0.3, right: 0.65, bottom: 0.75 },
    brightness: 140,
    sharpness: 40,
    motion: 0.5,
    landmarkStability: 0.95,
    ...overrides,
  };
}

describe("evaluateFrame", () => {
  it("1. rejects when no face is detected", () => {
    const r = evaluateFrame(goodMeasurement({ faceDetected: false }), 0);
    expect(r.accepted).toBe(false);
    expect(r.rejectReasons).toContain("no_face");
  });

  it("2. rejects when the face is too close", () => {
    const r = evaluateFrame(goodMeasurement({ faceBounds: { left: 0.1, top: 0.02, right: 0.9, bottom: 0.98 } }), 0);
    expect(r.accepted).toBe(false);
    expect(r.rejectReasons).toContain("too_close");
  });

  it("3. rejects when the face is too far", () => {
    const r = evaluateFrame(goodMeasurement({ faceBounds: { left: 0.45, top: 0.45, right: 0.55, bottom: 0.55 } }), 0);
    expect(r.accepted).toBe(false);
    expect(r.rejectReasons).toContain("too_far");
  });

  it("4. rejects when the face drifts outside the oval (off-center)", () => {
    const r = evaluateFrame(goodMeasurement({ faceBounds: { left: 0.02, top: 0.3, right: 0.22, bottom: 0.75 } }), 0);
    expect(r.accepted).toBe(false);
    expect(r.rejectReasons).toContain("off_center_x");
  });

  it("5. a checkpoint capture at the WRONG yaw is not 'on target' even though the frame itself is technically clean", () => {
    // The frame passes the technical quality gate (evaluateFrame doesn't
    // reject purely for being far from the CURRENT target's angle), but
    // angleError against a 45deg target must be large so the state machine
    // (tested separately) never counts it as a checkpoint capture.
    const r = evaluateFrame(goodMeasurement({ yaw: 2 }), 45);
    expect(r.accepted).toBe(true);
    expect(r.angleError).toBeGreaterThan(40);
  });

  it("6. rejects when pitch exceeds the limit", () => {
    const r = evaluateFrame(goodMeasurement({ pitch: POSE.MAX_PITCH_DEG + 10 }), 0);
    expect(r.accepted).toBe(false);
    expect(r.rejectReasons).toContain("pitch_exceeded");
  });

  it("7. rejects when roll exceeds the limit", () => {
    const r = evaluateFrame(goodMeasurement({ roll: POSE.MAX_ROLL_DEG + 10 }), 0);
    expect(r.accepted).toBe(false);
    expect(r.rejectReasons).toContain("roll_exceeded");
  });

  it("8. rejects when brightness is too low", () => {
    const r = evaluateFrame(goodMeasurement({ brightness: Math.max(0, IMAGE_QUALITY.MIN_BRIGHTNESS - 10) }), 0);
    expect(r.accepted).toBe(false);
    expect(r.rejectReasons).toContain("too_dark");
  });

  it("9. rejects when brightness is too high (overexposed)", () => {
    const r = evaluateFrame(goodMeasurement({ brightness: IMAGE_QUALITY.MAX_BRIGHTNESS + 10 }), 0);
    expect(r.accepted).toBe(false);
    expect(r.rejectReasons).toContain("overexposed");
  });

  it("10. rejects a blurry frame", () => {
    const r = evaluateFrame(goodMeasurement({ sharpness: Math.max(0, IMAGE_QUALITY.MIN_SHARPNESS - 5) }), 0);
    expect(r.accepted).toBe(false);
    expect(r.rejectReasons).toContain("blurry");
  });

  it("11. rejects a frame with too much motion", () => {
    const r = evaluateFrame(goodMeasurement({ motion: IMAGE_QUALITY.MAX_MOTION_PCT + 10 }), 0);
    expect(r.accepted).toBe(false);
    expect(r.rejectReasons).toContain("motion_blur");
  });

  it("12. rejects when landmark tracking is unstable", () => {
    const r = evaluateFrame(goodMeasurement({ landmarkStability: Math.max(0, STABILITY.MIN_LANDMARK_STABILITY - 0.1) }), 0);
    expect(r.accepted).toBe(false);
    expect(r.rejectReasons).toContain("unstable_tracking");
  });

  it("13. accepts a genuinely good, on-pose, stable, sharp, well-lit frame", () => {
    const r = evaluateFrame(goodMeasurement(), 0);
    expect(r.accepted).toBe(true);
    expect(r.rejectReasons).toHaveLength(0);
    expect(r.angleError).toBe(0);
  });

  it("a real face with only the skin-blob fallback (no real landmarks) is not trusted for pose", () => {
    const r = evaluateFrame(goodMeasurement({ landmarksReal: false }), 0);
    expect(r.accepted).toBe(false);
    expect(r.rejectReasons).toContain("not_real_landmarks");
  });
});
