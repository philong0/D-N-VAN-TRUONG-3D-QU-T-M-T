/**
 * frame-evaluator.ts
 *
 * The ONE frame-quality evaluator for the guided face scanner (spec Part 9).
 * Every accept/reject decision — precheck, per-target capture, transition
 * capture — goes through `evaluateFrame()`. No other module should invent
 * its own ad hoc quality condition; extend the thresholds in
 * scan-constants.ts and this function instead.
 *
 * Pure function: no DOM, no camera, no React — takes plain numbers already
 * measured elsewhere (face-geometry.ts does the actual pixel/landmark
 * measurement), so it is directly unit-testable.
 */
import { FRAMING, POSE, IMAGE_QUALITY, STABILITY } from "./scan-constants";

export interface FrameMeasurement {
  faceDetected: boolean;
  /** True only when a real MediaPipe FaceMesh landmark set was used for this measurement (not the skin-blob fallback). */
  landmarksReal: boolean;
  yaw: number;
  pitch: number;
  roll: number;
  /** Normalized face bounding box, 0..1 against frame width/height. */
  faceBounds: { left: number; top: number; right: number; bottom: number };
  /** Mean luma, 0-255. */
  brightness: number;
  /** Laplacian-energy sharpness proxy (see scan-constants.ts). */
  sharpness: number;
  /** Frame-to-frame face-center displacement, % of frame width. */
  motion: number;
  /** 0..1, how stable the landmark positions have been over the last few ticks. */
  landmarkStability: number;
}

export type RejectReason =
  | "no_face"
  | "not_real_landmarks"
  | "too_close"
  | "too_far"
  | "off_center_x"
  | "off_center_y"
  | "roll_exceeded"
  | "pitch_exceeded"
  | "too_dark"
  | "overexposed"
  | "blurry"
  | "motion_blur"
  | "unstable_tracking";

export interface FrameEvaluation {
  accepted: boolean;
  score: number; // 0..100
  yaw: number;
  pitch: number;
  roll: number;
  brightness: number;
  sharpness: number;
  motion: number;
  faceCoverage: number; // face bbox height ratio, proxy for distance
  landmarkStability: number;
  /** |yaw - targetYaw|; NaN when no target angle is relevant (e.g. plain precheck-only call without a target). */
  angleError: number;
  rejectReasons: RejectReason[];
}

/**
 * Evaluates ONE measured frame against the scanner's real technical quality
 * gate. `targetYaw` is the angle this evaluation should measure error
 * against (0 for precheck/front, ±15/±30/±45 for the other checkpoints); it
 * does NOT affect accept/reject by itself — angleError is reported so the
 * caller (scan-state-machine.ts) can separately decide "on target" vs
 * "good but still transitioning".
 */
export function evaluateFrame(m: FrameMeasurement, targetYaw: number): FrameEvaluation {
  const reasons: RejectReason[] = [];

  if (!m.faceDetected) {
    reasons.push("no_face");
  } else if (!m.landmarksReal) {
    reasons.push("not_real_landmarks");
  }

  const faceH = m.faceBounds.bottom - m.faceBounds.top;
  const centerX = (m.faceBounds.left + m.faceBounds.right) / 2;
  const centerY = (m.faceBounds.top + m.faceBounds.bottom) / 2;

  if (m.faceDetected) {
    if (faceH > FRAMING.MAX_FACE_HEIGHT_RATIO) reasons.push("too_close");
    else if (faceH < FRAMING.MIN_FACE_HEIGHT_RATIO) reasons.push("too_far");

    if (centerX < FRAMING.CENTER_X_MIN || centerX > FRAMING.CENTER_X_MAX) reasons.push("off_center_x");
    if (centerY < FRAMING.CENTER_Y_MIN || centerY > FRAMING.CENTER_Y_MAX) reasons.push("off_center_y");

    if (Math.abs(m.roll) > POSE.MAX_ROLL_DEG) reasons.push("roll_exceeded");
    if (Math.abs(m.pitch) > POSE.MAX_PITCH_DEG) reasons.push("pitch_exceeded");
  }

  if (m.brightness < IMAGE_QUALITY.MIN_BRIGHTNESS) reasons.push("too_dark");
  if (m.brightness > IMAGE_QUALITY.MAX_BRIGHTNESS) reasons.push("overexposed");
  if (m.sharpness < IMAGE_QUALITY.MIN_SHARPNESS) reasons.push("blurry");
  if (m.motion > IMAGE_QUALITY.MAX_MOTION_PCT) reasons.push("motion_blur");
  if (m.landmarkStability < STABILITY.MIN_LANDMARK_STABILITY) reasons.push("unstable_tracking");

  const accepted = m.faceDetected && reasons.length === 0;

  // Simple, transparent scoring: start at 100, subtract a fixed penalty per
  // failed check plus a continuous term for how far brightness/sharpness
  // are from their own thresholds — good enough to rank frames of a given
  // target against each other, not a calibrated clinical metric.
  let score = 100;
  score -= reasons.length * 15;
  score -= Math.max(0, IMAGE_QUALITY.MIN_SHARPNESS - m.sharpness) * 0.5;
  score -= Math.max(0, IMAGE_QUALITY.MIN_BRIGHTNESS - m.brightness) * 0.3;
  score -= Math.max(0, m.brightness - IMAGE_QUALITY.MAX_BRIGHTNESS) * 0.3;
  score = Math.max(0, Math.min(100, Math.round(score)));

  return {
    accepted,
    score,
    yaw: m.yaw,
    pitch: m.pitch,
    roll: m.roll,
    brightness: m.brightness,
    sharpness: m.sharpness,
    motion: m.motion,
    faceCoverage: faceH,
    landmarkStability: m.landmarkStability,
    angleError: Number.isFinite(targetYaw) ? Math.abs(m.yaw - targetYaw) : NaN,
    rejectReasons: reasons,
  };
}
