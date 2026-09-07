import { describe, expect, it } from "vitest";
import { turnProxyToYawDeg, MAX_REACHABLE_YAW_DEG } from "../pose-mapping";

describe("turnProxyToYawDeg", () => {
  it("clears the hardest real checkpoint requirement (60deg target, ±10 tolerance -> 70deg edge) with real margin", () => {
    // This is the exact real-device bug: the old formula's ceiling (~48.6deg)
    // sat BELOW the 50deg lower edge of the LEFT60/RIGHT60 acceptance window,
    // making those checkpoints unreachable no matter how far a user turned.
    expect(MAX_REACHABLE_YAW_DEG).toBeGreaterThan(70);
    expect(turnProxyToYawDeg(1)).toBeGreaterThanOrEqual(50);
    expect(turnProxyToYawDeg(-1)).toBeLessThanOrEqual(-50);
  });

  it("is monotonically increasing across the full input range", () => {
    let prev = turnProxyToYawDeg(-1);
    for (let x = -0.9; x <= 1.0001; x += 0.1) {
      const y = turnProxyToYawDeg(x);
      expect(y).toBeGreaterThanOrEqual(prev);
      prev = y;
    }
  });

  it("is odd-symmetric (turning left vs right by the same raw ratio yields mirrored degrees)", () => {
    for (const x of [0.1, 0.25, 0.5, 0.75, 0.95, 1]) {
      expect(turnProxyToYawDeg(-x)).toBe(-turnProxyToYawDeg(x));
    }
  });

  it("returns 0 for a neutral (frontal) reading", () => {
    expect(turnProxyToYawDeg(0)).toBe(0);
  });

  it("clamps out-of-range input the same as in-range boundary input", () => {
    expect(turnProxyToYawDeg(5)).toBe(turnProxyToYawDeg(1));
    expect(turnProxyToYawDeg(-5)).toBe(turnProxyToYawDeg(-1));
  });
});
