/**
 * scan-constants.ts
 *
 * Single source of truth for every threshold the guided face scanner uses to
 * accept/reject a frame. Centralized here (instead of scattered literals
 * inside GuidedFaceScan.tsx) so there is exactly one place to recalibrate.
 *
 * IMPORTANT (2026-09-03 scanner-quality audit): none of the numeric values
 * below are a validated clinical/medical standard. They are ENGINEERING
 * thresholds — chosen from ordinary photography/computer-vision reasoning
 * (typical webcam/phone-camera dynamic range, typical face-mesh detector
 * jitter, typical head-turn speed) so the pipeline has *some* real gate
 * instead of none. They have not been calibrated against a labeled dataset
 * of real patient scans. Treat every constant here as a starting point that
 * should be tuned once real accept/reject-rate data exists, not as a
 * "y khoa chuẩn" (clinically validated) figure.
 */

export const FRAME_TICK_MS = 180;

// ---------------------------------------------------------------------------
// Face framing (oval) gate
// ---------------------------------------------------------------------------
export const FRAMING = {
  /** face bounding-box height / frame height. Below this the face is too far away. */
  MIN_FACE_HEIGHT_RATIO: 0.15,
  /** Above this the face is too close (risk of lens distortion at close range). */
  MAX_FACE_HEIGHT_RATIO: 0.90,
  CENTER_X_MIN: 0.15,
  CENTER_X_MAX: 0.85,
  CENTER_Y_MIN: 0.15,
  CENTER_Y_MAX: 0.85,
} as const;

// ---------------------------------------------------------------------------
// Head pose gate
// ---------------------------------------------------------------------------
export const POSE = {
  /** Max allowed |roll| (in-plane head tilt, degrees) for ANY accepted frame. */
  MAX_ROLL_DEG: 30,
  /** Max allowed |pitch| (up/down head tilt, degrees) for ANY accepted frame. */
  MAX_PITCH_DEG: 30,
  /** Precheck ("giữ đầu thẳng, nhìn thẳng camera") requires |yaw| under this. */
  PRECHECK_MAX_YAW_DEG: 15,
} as const;

// ---------------------------------------------------------------------------
// Image quality gate
// ---------------------------------------------------------------------------
export const IMAGE_QUALITY = {
  /** Mean luma (0-255). Below this = too dark to trust real skin-tone/detail. */
  MIN_BRIGHTNESS: 25,
  /** Mean luma (0-255). Above this = blown-out / overexposed. */
  MAX_BRIGHTNESS: 252,
  /** Sharpness proxy: mean absolute Laplacian (edge energy). */
  MIN_SHARPNESS: 5,
  /** Motion proxy: relaxed for natural handheld head turns. */
  MAX_MOTION_PCT: 35.0,
} as const;

// ---------------------------------------------------------------------------
// Landmark stability gate
// ---------------------------------------------------------------------------
export const STABILITY = {
  /** 0..1 score derived from landmark stability. Relaxed for handheld mobile scanning. */
  MIN_LANDMARK_STABILITY: 0.15,
  /** Consecutive good precheck ticks required before leaving PRECHECK. */
  PRECHECK_STABLE_TICKS: 1,
  /** Consecutive good on-target ticks required before banking ONE checkpoint frame. */
  CHECKPOINT_STABLE_TICKS: 1,
  /** Minimum ms between two banked frames. */
  MIN_MS_BETWEEN_CAPTURES: 150,
  /** Minimum |yaw| delta (degrees) between two banked TRANSITION frames. */
  MIN_TRANSITION_YAW_DELTA_DEG: 2,
} as const;

// ---------------------------------------------------------------------------
// Final input-quality review (Part 11) — gate before submit
// ---------------------------------------------------------------------------
export const FINAL_REVIEW = {
  /** Every checkpoint target must have reached at least this many banked frames. */
  MIN_FRAMES_PER_TARGET: 3,
  /** Real measured min yaw (most-left) must be at or beyond this to call "left coverage" complete. */
  MIN_LEFT_YAW_DEG: -40,
  /** Real measured max yaw (most-right) must be at or beyond this to call "right coverage" complete. */
  MIN_RIGHT_YAW_DEG: 40,
  /** Reject-rate (rejected / (accepted+rejected)) above this triggers a "chất lượng chung thấp" warning. */
  MAX_REJECT_RATE_WARNING: 0.6,
} as const;
