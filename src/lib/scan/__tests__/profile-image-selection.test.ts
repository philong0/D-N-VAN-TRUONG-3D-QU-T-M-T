import { describe, it, expect } from "vitest";
import { selectBestProfileImages } from "../profile-image-selection";
import type { BurstFrameCandidate } from "../reconstruction-frame-selection";

function c(fileName: string, yaw: number | null, overrides: Partial<BurstFrameCandidate> = {}): BurstFrameCandidate {
  return { fileName, yaw, qualityScore: 80, sharpness: 40, motion: 0.5, pitch: 0, roll: 0, timestampMs: 0, ...overrides };
}

/** A realistic 50-frame session sweeping roughly -45..+45 with some noise, unordered on purpose. */
function fiftyFrameSession(): BurstFrameCandidate[] {
  const frames: BurstFrameCandidate[] = [];
  for (let i = 0; i < 50; i++) {
    const yaw = -45 + (i * 90) / 49 + (i % 3 === 0 ? 1 : -1); // small jitter, still real coverage
    frames.push(c(`burst_${String(i).padStart(4, "0")}.jpg`, Math.round(yaw * 10) / 10, { qualityScore: 60 + (i % 7) * 5, timestampMs: i * 200 }));
  }
  // Shuffle deterministically so array order carries no information.
  return frames
    .map((f, i) => ({ f, k: (i * 2654435761) % 2147483647 }))
    .sort((a, b) => a.k - b.k)
    .map((x) => x.f);
}

describe("selectBestProfileImages", () => {
  it("1. from 50 real frames, selects exactly 4 profile images", () => {
    const { images, missingRoles } = selectBestProfileImages(fiftyFrameSession());
    expect(images).toHaveLength(4);
    expect(missingRoles).toHaveLength(0);
    expect(new Set(images.map((i) => i.role)).size).toBe(4);
  });

  it("2. a shuffled array still assigns roles by real yaw, not position", () => {
    const shuffled: BurstFrameCandidate[] = [
      c("a.jpg", 30),
      c("b.jpg", -45),
      c("c.jpg", 0),
      c("d.jpg", 15),
      c("e.jpg", -15),
      c("f.jpg", 45),
      c("g.jpg", -30),
    ];
    const { images } = selectBestProfileImages(shuffled);
    const byRole = Object.fromEntries(images.map((i) => [i.role, i]));
    expect(byRole.front?.fileName).toBe("c.jpg");
    expect(byRole.left?.fileName).toBe("b.jpg");
    expect(byRole.right?.fileName).toBe("f.jpg");
  });

  it("3. no frame near 0deg -- front is reported missing, never faked", () => {
    const candidates: BurstFrameCandidate[] = [c("left.jpg", -45), c("right.jpg", 45)];
    const { images, missingRoles } = selectBestProfileImages(candidates);
    expect(images.find((i) => i.role === "front")).toBeUndefined();
    expect(missingRoles).toContain("front");
  });

  it("4. no left-side frame -- left is reported missing", () => {
    const candidates: BurstFrameCandidate[] = [c("front.jpg", 0), c("right.jpg", 45)];
    const { images, missingRoles } = selectBestProfileImages(candidates);
    expect(images.find((i) => i.role === "left")).toBeUndefined();
    expect(missingRoles).toContain("left");
  });

  it("5. no right-side frame -- right is reported missing", () => {
    const candidates: BurstFrameCandidate[] = [c("front.jpg", 0), c("left.jpg", -45)];
    const { images, missingRoles } = selectBestProfileImages(candidates);
    expect(images.find((i) => i.role === "right")).toBeUndefined();
    expect(missingRoles).toContain("right");
  });

  it("6. multiple frames at nearly the same angle -- the higher qualityScore one wins", () => {
    const candidates: BurstFrameCandidate[] = [
      c("front_low.jpg", 1, { qualityScore: 40 }),
      c("front_high.jpg", -1, { qualityScore: 97 }),
      c("left.jpg", -45, { qualityScore: 90 }),
      c("right.jpg", 45, { qualityScore: 90 }),
    ];
    const { images } = selectBestProfileImages(candidates);
    expect(images.find((i) => i.role === "front")?.fileName).toBe("front_high.jpg");
  });

  it("7. the 4 selected images never share near-identical yaw", () => {
    const candidates: BurstFrameCandidate[] = [
      c("front.jpg", 0, { qualityScore: 90 }),
      c("left.jpg", -45, { qualityScore: 90 }),
      c("right.jpg", 45, { qualityScore: 90 }),
      c("near_front.jpg", 3, { qualityScore: 99 }), // highest quality overall, but too close to front.jpg
      c("mid_left.jpg", -25, { qualityScore: 70 }),
    ];
    const { images } = selectBestProfileImages(candidates);
    const yaws = images.map((i) => i.yaw);
    for (let i = 0; i < yaws.length; i++) {
      for (let j = i + 1; j < yaws.length; j++) {
        expect(Math.abs(yaws[i] - yaws[j])).toBeGreaterThanOrEqual(12);
      }
    }
    // Confirms the near-duplicate high-quality frame was correctly passed over for three_quarter.
    expect(images.find((i) => i.role === "three_quarter")?.fileName).toBe("mid_left.jpg");
  });

  it("8. returned metadata matches the real source frame exactly", () => {
    const candidates: BurstFrameCandidate[] = [
      c("front.jpg", 2, { pitch: 3, roll: -1, qualityScore: 88, timestampMs: 12345 }),
      c("left.jpg", -45, { pitch: 1, roll: 2, qualityScore: 77, timestampMs: 22222 }),
      c("right.jpg", 45, { pitch: -2, roll: 0, qualityScore: 81, timestampMs: 33333 }),
    ];
    const { images } = selectBestProfileImages(candidates);
    const front = images.find((i) => i.role === "front")!;
    expect(front.fileName).toBe("front.jpg");
    expect(front.sourceFrameId).toBe("front.jpg");
    expect(front.yaw).toBe(2);
    expect(front.pitch).toBe(3);
    expect(front.roll).toBe(-1);
    expect(front.qualityScore).toBe(88);
    expect(front.timestamp).toBe(12345);
  });

  it("never selects the same file for two roles", () => {
    const candidates: BurstFrameCandidate[] = [c("only.jpg", 0, { qualityScore: 100 })];
    const { images } = selectBestProfileImages(candidates);
    const fileNames = images.map((i) => i.fileName);
    expect(new Set(fileNames).size).toBe(fileNames.length);
  });
});
