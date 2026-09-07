import { describe, it, expect } from "vitest";
import { computeCameraNormalization, applyYawSign, type CameraFrameContext } from "../camera-normalization";

describe("computeCameraNormalization", () => {
  it("flips yaw sign for FRONT camera (regression: this used to be the one NOT flipped, which was backwards)", () => {
    const { coordinateTransform } = computeCameraNormalization({ cameraFacing: "user", mirroredPreview: true });
    expect(coordinateTransform.flipYawSign).toBe(true);
  });

  it("flips yaw sign for REAR camera identically to front -- same optics, same correction", () => {
    const { coordinateTransform } = computeCameraNormalization({ cameraFacing: "environment", mirroredPreview: false });
    expect(coordinateTransform.flipYawSign).toBe(true);
  });

  it("mirroredPreview never changes the coordinate transform -- it is a display-only flag", () => {
    const a = computeCameraNormalization({ cameraFacing: "user", mirroredPreview: true });
    const b = computeCameraNormalization({ cameraFacing: "user", mirroredPreview: false });
    expect(a.coordinateTransform).toEqual(b.coordinateTransform);
  });

  it("front and rear produce the exact same transform given the same orientation inputs", () => {
    const front = computeCameraNormalization({ cameraFacing: "user", mirroredPreview: true, sensorOrientation: 90, displayOrientation: 0 });
    const rear = computeCameraNormalization({ cameraFacing: "environment", mirroredPreview: false, sensorOrientation: 90, displayOrientation: 0 });
    expect(front.coordinateTransform).toEqual(rear.coordinateTransform);
  });

  it("computes rotationDeg from the net difference between sensor and display orientation", () => {
    expect(computeCameraNormalization({ cameraFacing: "user", mirroredPreview: true, sensorOrientation: 90, displayOrientation: 0 }).coordinateTransform.rotationDeg).toBe(90);
    expect(computeCameraNormalization({ cameraFacing: "user", mirroredPreview: true, sensorOrientation: 0, displayOrientation: 0 }).coordinateTransform.rotationDeg).toBe(0);
    expect(computeCameraNormalization({ cameraFacing: "user", mirroredPreview: true, sensorOrientation: 270, displayOrientation: 90 }).coordinateTransform.rotationDeg).toBe(180);
  });

  it("defaults orientation to 0 when not provided, never throwing", () => {
    const ctx: CameraFrameContext = { cameraFacing: "user", mirroredPreview: true };
    expect(() => computeCameraNormalization(ctx)).not.toThrow();
    expect(computeCameraNormalization(ctx).coordinateTransform.rotationDeg).toBe(0);
  });
});

describe("applyYawSign", () => {
  it("negates a real-right-turn raw signal into the canonical positive (subject-right) reading", () => {
    // A subject turning to their real right produces a NEGATIVE raw signal
    // (see module docstring for the photography-geometry proof) -- after
    // the canonical transform, this must read POSITIVE.
    const { coordinateTransform } = computeCameraNormalization({ cameraFacing: "user", mirroredPreview: true });
    const rawSignalForRealRightTurn = -15;
    expect(applyYawSign(rawSignalForRealRightTurn, coordinateTransform)).toBe(15);
  });

  it("negates a real-left-turn raw signal into the canonical negative (subject-left) reading", () => {
    const { coordinateTransform } = computeCameraNormalization({ cameraFacing: "user", mirroredPreview: true });
    const rawSignalForRealLeftTurn = 15;
    expect(applyYawSign(rawSignalForRealLeftTurn, coordinateTransform)).toBe(-15);
  });

  it("produces identical canonical output for front and rear given the same raw signal", () => {
    const front = computeCameraNormalization({ cameraFacing: "user", mirroredPreview: true }).coordinateTransform;
    const rear = computeCameraNormalization({ cameraFacing: "environment", mirroredPreview: false }).coordinateTransform;
    const raw = -8.5;
    expect(applyYawSign(raw, front)).toBe(applyYawSign(raw, rear));
  });
});
