/**
 * profile-image-selection.ts
 *
 * `selectBestProfileImages()` — after a guided scan finishes, picks exactly
 * 4 real frames (front / left / right / three_quarter) to save as the
 * patient's PROFILE PREVIEW — a small, doctor-facing set for quick visual
 * reference. This is deliberately separate from reconstruction input:
 * reconstruction keeps every accepted frame (see reconstruction-frame-
 * selection.ts, which feeds the GNM pipeline's own angle1-4 slots); this
 * module only decides what gets shown in the patient dossier.
 *
 * Same discipline as reconstruction-frame-selection.ts: real measured yaw
 * decides role assignment, never array position; a role with no real
 * evidence is reported missing, never faked; near-duplicate yaw is
 * actively avoided for the fourth ("three_quarter") slot so the 4 saved
 * images are genuinely distinct views, not 4 near-identical frames.
 */
import { candidateScore, pickBestNear, type BurstFrameCandidate } from "./reconstruction-frame-selection";

export type ProfileRole = "front" | "left" | "right" | "three_quarter";

export interface SelectedProfileImage {
  role: ProfileRole;
  fileName: string;
  yaw: number;
  pitch: number;
  roll: number;
  qualityScore: number;
  timestamp: number;
  sourceFrameId: string;
}

export interface ProfileImageSelectionResult {
  images: SelectedProfileImage[];
  missingRoles: ProfileRole[];
}

const FRONT_MAX_ABS_YAW = 10;
const SIDE_MIN_ABS_YAW = 10;
const LEFT_TARGET_YAW = -45;
const RIGHT_TARGET_YAW = 45;
/** The 4th ("three_quarter") image must differ from ALL 3 already-picked by at least this many degrees of yaw, so the 4 saved images are genuinely distinct views rather than 4 near-identical frames. */
const MIN_DISTINCT_YAW_DEG = 12;

function toSelected(role: ProfileRole, c: BurstFrameCandidate): SelectedProfileImage {
  return {
    role,
    fileName: c.fileName,
    yaw: c.yaw as number,
    pitch: c.pitch ?? 0,
    roll: c.roll ?? 0,
    qualityScore: c.qualityScore ?? 0,
    timestamp: c.timestampMs ?? 0,
    sourceFrameId: c.fileName,
  };
}

/**
 * Selects up to 4 real, distinct frames from real measured yaw + quality —
 * never from array position, never fabricated. A role is only included in
 * `images` when a real candidate satisfies it; otherwise it appears in
 * `missingRoles` (the caller decides what to show/warn — this function
 * never substitutes a wrong-angle frame just to fill the slot).
 */
export function selectBestProfileImages(candidates: BurstFrameCandidate[]): ProfileImageSelectionResult {
  const withYaw = candidates.filter((c): c is BurstFrameCandidate & { yaw: number } => typeof c.yaw === "number" && Number.isFinite(c.yaw));

  const images: SelectedProfileImage[] = [];
  const missingRoles: ProfileRole[] = [];
  const usedFileNames = new Set<string>();

  const frontPool = withYaw.filter((c) => Math.abs(c.yaw) < FRONT_MAX_ABS_YAW);
  const front = pickBestNear(0, frontPool);
  if (front) {
    images.push(toSelected("front", front));
    usedFileNames.add(front.fileName);
  } else {
    missingRoles.push("front");
  }

  const leftPool = withYaw.filter((c) => c.yaw <= -SIDE_MIN_ABS_YAW);
  const left = pickBestNear(LEFT_TARGET_YAW, leftPool);
  if (left) {
    images.push(toSelected("left", left));
    usedFileNames.add(left.fileName);
  } else {
    missingRoles.push("left");
  }

  const rightPool = withYaw.filter((c) => c.yaw >= SIDE_MIN_ABS_YAW);
  const right = pickBestNear(RIGHT_TARGET_YAW, rightPool);
  if (right) {
    images.push(toSelected("right", right));
    usedFileNames.add(right.fileName);
  } else {
    missingRoles.push("right");
  }

  // "three_quarter": the best-quality REMAINING frame that adds genuinely
  // new coverage -- at least MIN_DISTINCT_YAW_DEG of yaw away from every
  // role already picked. Falls back to the best remaining frame overall
  // (still real, still not a duplicate file) only if no candidate meets
  // the distinctness bar -- reported honestly via `missingRoles` instead
  // of silently pretending a near-duplicate is a distinct extra view.
  const chosenYaws = images.map((img) => img.yaw);
  const remaining = withYaw.filter((c) => !usedFileNames.has(c.fileName));
  const distinctEnough = remaining.filter((c) => chosenYaws.every((y) => Math.abs(c.yaw - y) >= MIN_DISTINCT_YAW_DEG));

  const pickBestByQuality = (pool: BurstFrameCandidate[]): BurstFrameCandidate | null =>
    pool.length === 0 ? null : [...pool].sort((a, b) => candidateScore(b) - candidateScore(a))[0];

  const threeQuarter = pickBestByQuality(distinctEnough);
  if (threeQuarter) {
    images.push(toSelected("three_quarter", threeQuarter));
    usedFileNames.add(threeQuarter.fileName);
  } else {
    missingRoles.push("three_quarter");
  }

  return { images, missingRoles };
}
