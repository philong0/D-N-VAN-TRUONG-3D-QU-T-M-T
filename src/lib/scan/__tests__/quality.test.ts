import { describe, expect, it } from "vitest";
import { evaluateScanQuality } from "../quality";
import type { ScanFrame } from "@/lib/types";

const REQUIRED_VIEWS = ["front", "left_45", "left_profile", "right_45", "right_profile"] as const;

function realNativeFrame(view: (typeof REQUIRED_VIEWS)[number]): ScanFrame {
  return {
    id: `${view}-id`,
    view,
    fileName: `${view}.jpg`,
    capturedAt: new Date().toISOString(),
    byteSize: 500_000,
    mimeType: "image/jpeg",
    depthAvailable: true,
    // Real shape actually written by the /package upload route (see
    // route.ts) -- geometry/intrinsics saved inline here, never in the
    // unused geometryFileName/intrinsicsFileName fields.
    poseMetadata: { geometry: { vertexCount: 1220, triangleCount: 2304 } },
    cameraMetadata: { fx: 970.29, fy: 970.29, cx: 723.4, cy: 542.0 },
  };
}

describe("evaluateScanQuality — ios_native real-depth recognition", () => {
  it("2026-09-07 regression: recognizes real TrueDepth data actually saved by the package route (geometry/intrinsics inline, not the unused *FileName fields)", () => {
    const frames = REQUIRED_VIEWS.map(realNativeFrame);
    const report = evaluateScanQuality(frames, "ios_native");
    expect(report.overall).toBe("pass");
    expect(report.trackingQuality.status).toBe("pass");
    expect(report.trackingQuality.detail).not.toMatch(/Web camera/);
    expect(report.geometryConsistency?.status).toBe("pass");
  });

  it("still correctly reports no-depth for a real web_camera session (no regression the other way)", () => {
    const frames = REQUIRED_VIEWS.map((view) => ({ ...realNativeFrame(view), depthAvailable: false, poseMetadata: undefined, cameraMetadata: undefined }));
    const report = evaluateScanQuality(frames, "web_camera");
    expect(report.overall).toBe("warning");
    expect(report.trackingQuality.detail).toMatch(/Web camera/);
  });

  it("does not fake nativeDepth=true when depth/geometry/intrinsics are genuinely missing on an ios_native session", () => {
    const frames = REQUIRED_VIEWS.map((view) => ({
      id: `${view}-id`,
      view,
      fileName: `${view}.jpg`,
      capturedAt: new Date().toISOString(),
      byteSize: 500_000,
      mimeType: "image/jpeg",
      depthAvailable: false,
    }));
    const report = evaluateScanQuality(frames, "ios_native");
    expect(report.overall).toBe("warning");
  });
});
