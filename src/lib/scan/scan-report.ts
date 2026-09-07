/**
 * scan-report.ts
 *
 * Builds the honest, real end-of-scan report (spec Part 11 final review +
 * Part 12 debug log) from the frames actually banked during the session and
 * the reject-reason tally accumulated tick by tick. Pure function — no
 * fabricated numbers, no hardcoded pass/fail outside the named constants in
 * scan-constants.ts.
 */
import type { RejectReason } from "./frame-evaluator";
import { STABILITY, FINAL_REVIEW } from "./scan-constants";
import { TARGET_ANGLES } from "./scan-state-machine";

export interface CapturedFrameRecord {
  kind: "checkpoint" | "transition";
  targetId: string;
  timestampMs: number;
  yaw: number;
  pitch: number;
  roll: number;
  faceBounds: { left: number; top: number; right: number; bottom: number };
  faceCoverage: number;
  brightness: number;
  sharpness: number;
  motion: number;
  landmarkStability: number;
  angleError: number;
  qualityScore: number;
}

export type RejectTally = Record<RejectReason, number>;

export function emptyRejectTally(): RejectTally {
  return {
    no_face: 0,
    not_real_landmarks: 0,
    too_close: 0,
    too_far: 0,
    off_center_x: 0,
    off_center_y: 0,
    roll_exceeded: 0,
    pitch_exceeded: 0,
    too_dark: 0,
    overexposed: 0,
    blurry: 0,
    motion_blur: 0,
    unstable_tracking: 0,
  };
}

export function tallyReject(tally: RejectTally, reasons: RejectReason[]): RejectTally {
  if (reasons.length === 0) return tally;
  const next = { ...tally };
  for (const r of reasons) next[r] = (next[r] || 0) + 1;
  return next;
}

export interface ScanReport {
  totalFrames: number;
  acceptedFrames: number;
  rejectedFrames: number;
  acceptedByTarget: Record<string, number>;
  yawRange: { min: number; max: number };
  pitchRange: { min: number; max: number };
  rollRange: { min: number; max: number };
  brightnessFailures: number;
  sharpnessFailures: number;
  motionFailures: number;
  poseFailures: number;
  coverage: { front: boolean; left: boolean; right: boolean };
  /** Largest gap (degrees) between two consecutive banked frames when sorted by yaw — a real overlap/gap proxy. */
  largestYawGapDeg: number;
  duplicateCount: number;
  scanDurationMs: number;
  finalQuality: "pass" | "fail";
  failReasons: string[];
}

export function buildScanReport(
  records: CapturedFrameRecord[],
  rejectTally: RejectTally,
  totalTicks: number,
  scanStartMs: number,
  scanEndMs: number
): ScanReport {
  const acceptedByTarget: Record<string, number> = {};
  for (const t of TARGET_ANGLES) acceptedByTarget[t.id] = 0;
  for (const r of records) {
    if (r.kind === "checkpoint") acceptedByTarget[r.targetId] = (acceptedByTarget[r.targetId] || 0) + 1;
  }

  const yaws = records.map((r) => r.yaw);
  const pitches = records.map((r) => r.pitch);
  const rolls = records.map((r) => r.roll);
  const yawRange = yaws.length ? { min: Math.min(...yaws), max: Math.max(...yaws) } : { min: 0, max: 0 };
  const pitchRange = pitches.length ? { min: Math.min(...pitches), max: Math.max(...pitches) } : { min: 0, max: 0 };
  const rollRange = rolls.length ? { min: Math.min(...rolls), max: Math.max(...rolls) } : { min: 0, max: 0 };

  const sortedYaws = [...yaws].sort((a, b) => a - b);
  let largestYawGapDeg = 0;
  for (let i = 1; i < sortedYaws.length; i++) {
    largestYawGapDeg = Math.max(largestYawGapDeg, sortedYaws[i] - sortedYaws[i - 1]);
  }

  // Real duplicate audit: two CHECKPOINT frames for the SAME target closer
  // together than the minimum spacing the state machine is supposed to
  // enforce. Should be 0 by construction — reported for real verification,
  // never assumed.
  let duplicateCount = 0;
  const byTarget = new Map<string, number[]>();
  for (const r of records) {
    if (r.kind !== "checkpoint") continue;
    const arr = byTarget.get(r.targetId) || [];
    arr.push(r.timestampMs);
    byTarget.set(r.targetId, arr);
  }
  for (const arr of byTarget.values()) {
    const sorted = [...arr].sort((a, b) => a - b);
    for (let i = 1; i < sorted.length; i++) {
      if (sorted[i] - sorted[i - 1] < STABILITY.MIN_MS_BETWEEN_CAPTURES) duplicateCount++;
    }
  }

  const rejectedFrames = Object.values(rejectTally).reduce((a, b) => a + b, 0);
  const brightnessFailures = rejectTally.too_dark + rejectTally.overexposed;
  const sharpnessFailures = rejectTally.blurry;
  const motionFailures = rejectTally.motion_blur;
  const poseFailures = rejectTally.roll_exceeded + rejectTally.pitch_exceeded + rejectTally.off_center_x + rejectTally.off_center_y + rejectTally.too_close + rejectTally.too_far;

  const failReasons: string[] = [];
  for (const t of TARGET_ANGLES) {
    const got = acceptedByTarget[t.id] || 0;
    if (got < t.requiredFrames) {
      failReasons.push(`Thiếu góc ${t.label}: mới có ${got}/${t.requiredFrames} khung hình đạt chuẩn.`);
    }
  }
  const reachedLeft = yawRange.min <= FINAL_REVIEW.MIN_LEFT_YAW_DEG;
  const reachedRight = yawRange.max >= FINAL_REVIEW.MIN_RIGHT_YAW_DEG;
  if (!reachedLeft) failReasons.push(`Chưa quay đủ sang trái (đo được ${yawRange.min.toFixed(0)}°, cần tới ${FINAL_REVIEW.MIN_LEFT_YAW_DEG}°).`);
  if (!reachedRight) failReasons.push(`Chưa quay đủ sang phải (đo được ${yawRange.max.toFixed(0)}°, cần tới ${FINAL_REVIEW.MIN_RIGHT_YAW_DEG}°).`);

  return {
    totalFrames: totalTicks,
    acceptedFrames: records.length,
    rejectedFrames,
    acceptedByTarget,
    yawRange,
    pitchRange,
    rollRange,
    brightnessFailures,
    sharpnessFailures,
    motionFailures,
    poseFailures,
    coverage: {
      front: (acceptedByTarget.front || 0) >= (TARGET_ANGLES.find((t) => t.id === "front")?.requiredFrames || 3),
      left: reachedLeft,
      right: reachedRight,
    },
    largestYawGapDeg,
    duplicateCount,
    scanDurationMs: Math.max(0, scanEndMs - scanStartMs),
    finalQuality: failReasons.length === 0 ? "pass" : "fail",
    failReasons,
  };
}
