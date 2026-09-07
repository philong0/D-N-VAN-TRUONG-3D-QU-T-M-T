/**
 * burst-candidates.ts
 *
 * Shared `ScanFrame[] -> BurstFrameCandidate[]` extraction, used by both
 * reconstruction-frame-selection's caller (reconstruction-service.ts) and
 * profile-image-selection's caller (profile-preview-service.ts) — one place
 * reading `qualityMetadata`'s real field names, so the two selection
 * consumers can never silently drift apart on what counts as a valid
 * measurement.
 */
import type { ScanFrame } from "@/lib/types";
import type { BurstFrameCandidate } from "./reconstruction-frame-selection";

function num(v: unknown): number | null {
  return typeof v === "number" && Number.isFinite(v) ? v : null;
}

export function buildBurstCandidates(frames: ScanFrame[]): BurstFrameCandidate[] {
  return frames
    .filter((f) => f.view === "burst")
    .map((frame): BurstFrameCandidate => {
      const meta = (frame.qualityMetadata ?? {}) as Record<string, unknown>;
      return {
        fileName: frame.fileName,
        yaw: num(meta.yaw),
        pitch: num(meta.pitch),
        roll: num(meta.roll),
        qualityScore: num(meta.qualityScore),
        sharpness: num(meta.sharpness),
        motion: num(meta.motion),
        targetId: typeof meta.targetId === "string" ? meta.targetId : null,
        timestampMs: num(meta.timestampMs),
      };
    });
}
