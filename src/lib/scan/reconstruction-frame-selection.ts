/**
 * reconstruction-frame-selection.ts
 *
 * Picks which real burst frames feed the (protected, untouched) GNM
 * reconstruction pipeline's fixed 4-slot contract (angle1=front,
 * angle2=left-oblique, angle3=left-profile [only when a genuine ~90deg
 * frame exists], angle4=right-oblique — see reconstruct_gnm_fullhead.py,
 * which is NOT modified by this module).
 *
 * Root cause this replaces: `reconstruct_cli.py`'s old burst-dir path chose
 * frames by ARRAY POSITION (0%, 25%, 50%, 75% through the sorted file
 * list), silently assuming the capture was one continuous sweep from
 * left-profile to right-45 in that exact order. The guided scanner (see
 * scan-state-machine.ts) instead produces named CHECKPOINTS in a fixed
 * yaw order (0, -15, -30, -45, +15, +30, +45) plus quality-gated
 * transition frames — its own real per-frame metadata (yaw/pitch/roll/
 * quality) is the only trustworthy signal for which file is which angle;
 * position in the burst directory is not.
 *
 * This module is pure TypeScript (no filesystem access) so it is directly
 * unit-testable: it takes the candidate list already assembled from
 * ScanFrame.qualityMetadata (by the caller, reconstruction-service.ts) and
 * returns which file (if any) should fill each reconstruction slot, plus a
 * real, non-fabricated selection report.
 */

export interface BurstFrameCandidate {
  fileName: string;
  /** Degrees, negative = left. `null`/`undefined` means this frame has no usable pose metadata. */
  yaw?: number | null;
  pitch?: number | null;
  roll?: number | null;
  qualityScore?: number | null;
  sharpness?: number | null;
  motion?: number | null;
  targetId?: string | null;
  timestampMs?: number | null;
}

export type ReconstructionSlot = "angle1" | "angle2" | "angle3" | "angle4";

export interface SelectedFrameInfo {
  filename: string;
  yaw: number;
  pitch: number;
  roll: number;
  qualityScore: number;
  targetId: string | null;
  timestamp: number;
}

export type SelectionMethod = "yaw_quality_overlap" | "positional_fallback_no_metadata";

export interface FrameSelectionReport {
  inputFrameCount: number;
  selectedFrameCount: number;
  selectedFrames: SelectedFrameInfo[];
  yawMin: number | null;
  yawMax: number | null;
  leftCount: number;
  frontCount: number;
  rightCount: number;
  duplicateCount: number;
  coverage: { left: boolean; front: boolean; right: boolean; profile: boolean };
  selectionMethod: SelectionMethod;
}

export interface FrameSelectionResult {
  selected: Partial<Record<ReconstructionSlot, string>>;
  report: FrameSelectionReport;
}

// Engineering thresholds for THIS selection step only (not clinical
// standards, not the scanner's own capture-time gate in scan-constants.ts —
// this runs after capture, over whatever real frames already exist).
const FRONT_MAX_ABS_YAW = 8;
const SIDE_MIN_ABS_YAW = 8;
const PROFILE_MIN_ABS_YAW = 60;
/** Two candidates count as "the same real moment" for duplicate auditing. */
const DUPLICATE_YAW_DELTA_DEG = 3;
const DUPLICATE_TIME_DELTA_MS = 300;
/** Two candidates' angular distance to a target must differ by more than this to prefer the closer one outright; within this margin, quality breaks the tie. */
const ANGLE_TIE_MARGIN_DEG = 2;

/** Exported so other selection modules (e.g. profile-image-selection.ts) share the exact same real-quality scoring instead of re-deriving their own. */
export function candidateScore(c: BurstFrameCandidate): number {
  const quality = c.qualityScore ?? 50;
  const motionPenalty = (c.motion ?? 0) * 2;
  const sharpnessBonus = (c.sharpness ?? 0) * 0.1;
  return quality - motionPenalty + sharpnessBonus;
}

/** Exported so other selection modules share the exact same "closest real yaw, quality breaks ties" logic. */
export function pickBestNear(targetYaw: number, pool: BurstFrameCandidate[]): BurstFrameCandidate | null {
  if (pool.length === 0) return null;
  const sorted = [...pool].sort((a, b) => {
    const da = Math.abs((a.yaw as number) - targetYaw);
    const db = Math.abs((b.yaw as number) - targetYaw);
    if (Math.abs(da - db) > ANGLE_TIE_MARGIN_DEG) return da - db;
    return candidateScore(b) - candidateScore(a);
  });
  return sorted[0];
}

function toInfo(c: BurstFrameCandidate): SelectedFrameInfo {
  return {
    filename: c.fileName,
    yaw: c.yaw as number,
    pitch: c.pitch ?? 0,
    roll: c.roll ?? 0,
    qualityScore: c.qualityScore ?? 0,
    targetId: c.targetId ?? null,
    timestamp: c.timestampMs ?? 0,
  };
}

function countDuplicates(withYaw: BurstFrameCandidate[]): number {
  const sorted = [...withYaw].sort((a, b) => (a.yaw as number) - (b.yaw as number));
  let duplicates = 0;
  for (let i = 1; i < sorted.length; i++) {
    const yawClose = Math.abs((sorted[i].yaw as number) - (sorted[i - 1].yaw as number)) <= DUPLICATE_YAW_DELTA_DEG;
    const a = sorted[i].timestampMs;
    const b = sorted[i - 1].timestampMs;
    const timeClose = typeof a === "number" && typeof b === "number" && Math.abs(a - b) <= DUPLICATE_TIME_DELTA_MS;
    if (yawClose && timeClose) duplicates++;
  }
  return duplicates;
}

/**
 * Selects up to 4 real frames (one per reconstruction slot) from real,
 * measured yaw/pose metadata — NEVER from array position. If not a single
 * candidate has usable yaw metadata, returns an empty selection with
 * `selectionMethod: "positional_fallback_no_metadata"` so the caller can
 * fall back to the pre-existing positional method WHILE logging that the
 * fallback has no real pose evidence (never silently pretend it does).
 */
export function selectReconstructionFrames(candidates: BurstFrameCandidate[]): FrameSelectionResult {
  const inputFrameCount = candidates.length;
  const withYaw = candidates.filter((c): c is BurstFrameCandidate & { yaw: number } => typeof c.yaw === "number" && Number.isFinite(c.yaw));

  if (withYaw.length === 0) {
    return {
      selected: {},
      report: {
        inputFrameCount,
        selectedFrameCount: 0,
        selectedFrames: [],
        yawMin: null,
        yawMax: null,
        leftCount: 0,
        frontCount: 0,
        rightCount: 0,
        duplicateCount: 0,
        coverage: { left: false, front: false, right: false, profile: false },
        selectionMethod: "positional_fallback_no_metadata",
      },
    };
  }

  const frontPool = withYaw.filter((c) => Math.abs(c.yaw) < FRONT_MAX_ABS_YAW);
  const leftPool = withYaw.filter((c) => c.yaw <= -SIDE_MIN_ABS_YAW);
  const rightPool = withYaw.filter((c) => c.yaw >= SIDE_MIN_ABS_YAW);
  const leftProfilePool = withYaw.filter((c) => c.yaw <= -PROFILE_MIN_ABS_YAW);

  const front = pickBestNear(0, frontPool.length ? frontPool : withYaw);
  const left = pickBestNear(-45, leftPool);
  const right = pickBestNear(45, rightPool);
  // Only ever fill the profile slot with a REAL profile-range frame — never
  // relabel a 45deg oblique shot as a 90deg profile (the GNM texture bake's
  // own angle3-specific weighting assumes a real profile, see module
  // docstring; this scanner currently tops out at 45deg, so this pool is
  // legitimately empty most of the time, and the slot is left unfilled
  // rather than fabricated).
  const profile = leftProfilePool.length ? pickBestNear(-90, leftProfilePool) : null;

  const selected: Partial<Record<ReconstructionSlot, string>> = {};
  const selectedFrames: SelectedFrameInfo[] = [];
  if (front) { selected.angle1 = front.fileName; selectedFrames.push(toInfo(front)); }
  if (left) { selected.angle2 = left.fileName; selectedFrames.push(toInfo(left)); }
  if (profile) { selected.angle3 = profile.fileName; selectedFrames.push(toInfo(profile)); }
  if (right) { selected.angle4 = right.fileName; selectedFrames.push(toInfo(right)); }

  const yaws = withYaw.map((c) => c.yaw);
  const yawMin = Math.min(...yaws);
  const yawMax = Math.max(...yaws);

  return {
    selected,
    report: {
      inputFrameCount,
      selectedFrameCount: selectedFrames.length,
      selectedFrames,
      yawMin,
      yawMax,
      leftCount: leftPool.length,
      frontCount: frontPool.length,
      rightCount: rightPool.length,
      duplicateCount: countDuplicates(withYaw),
      coverage: {
        left: yawMin <= -40,
        front: frontPool.length > 0,
        right: yawMax >= 40,
        profile: Boolean(profile),
      },
      selectionMethod: "yaw_quality_overlap",
    },
  };
}
