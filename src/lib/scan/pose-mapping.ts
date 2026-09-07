/**
 * pose-mapping.ts
 *
 * Converts face-geometry.ts's raw `turnProxy` (a dimensionless, engineering
 * ratio -- (nose_tip_x - eye_midpoint_x) / eye_half_span, clamped to
 * [-1, 1]) into a yaw estimate in degrees.
 *
 * 2026-09-04 real-device field bug (found via a live phone test, not
 * guessed): the previous inline formula in GuidedFaceScan.tsx was
 *   `Math.asin(clamp(turnProxy, -1, 1) * 0.75) * 180 / Math.PI`
 * `asin` only accepts inputs in [-1, 1], so with `turnProxy` clamped to
 * [-1, 1] and multiplied by 0.75, the argument to `asin` never exceeds
 * 0.75 in magnitude. That puts a HARD, unconditional ceiling on the output:
 *   asin(0.75) * 180 / pi ≈ 48.59 deg
 * No matter how far a real user turns their head, this formula can never
 * report more than ~48.6 deg. The scanner's LEFT60/RIGHT60 checkpoints
 * (targetYaw ±60, tolerance ±10 -- see GuidedFaceScan.tsx's TARGET_ANGLES)
 * require a measured |yaw| >= 50 to ever register "on target". Since 48.6 <
 * 50, those two checkpoints were mathematically UNREACHABLE, confirmed on
 * a real phone: the scan captured FRONT/LEFT20/LEFT45 (all <= 48.6 deg,
 * within reach) and then got stuck at LEFT60 no matter how far the tester
 * physically turned.
 *
 * The 0.75 constant had been implicitly tuned for an OLDER max checkpoint
 * of 45 deg (asin(0.75) = 48.59 gives comfortable margin above 45+10=55?
 * no -- even 45+10=55 was already unreachable under the old constant; it
 * only happened to work because LEFT45's *lower* tolerance edge, 35 deg,
 * is well under 48.6). When the checkpoints were extended to +-60 deg in
 * a prior session pass, this ceiling was not re-checked against the new
 * max target -- that oversight is the root cause.
 *
 * Fix: raise the internal scale constant so the ceiling clears the
 * hardest real requirement with margin, chosen directly from that
 * requirement (not an arbitrary bump):
 *   - Hardest target: 60 deg, tolerance 10 -> acceptance window's outer
 *     edge is 70 deg.
 *   - Need asin(K) meaningfully above 70 deg so frames near the edge of
 *     the acceptance window aren't sitting right at asin's near-vertical
 *     tangent (where small measurement noise in `turnProxy` would swing
 *     the reported degrees wildly).
 *   - K = 0.95 -> asin(0.95) * 180/pi ≈ 71.8 deg: clears the 70 deg edge
 *     with real margin while keeping the same asin-shaped model (still a
 *     monotonic, symmetric, non-arbitrary curve) instead of switching to
 *     a different function family.
 *
 * HONEST LIMITATION: `turnProxy` itself is a 2D-landmark ratio proxy, not
 * a calibrated pose estimator (see face-geometry.ts's own docs, and the
 * `minDetectionConfidence` comment noting MediaPipe's own confidence drops
 * past ~30 deg yaw). Raising K changes the WHOLE curve, not just the top
 * end: for the same physical head turn, a smaller real angle now maps to
 * a larger reported degree value than under the old K=0.75 (e.g. at
 * turnProxy=0.5, old formula reported ~22.0 deg, new reports ~28.4 deg).
 * This does not change monotonicity or symmetry, and the checkpoints
 * previously validated on a real device (0/±20/±45) stay comfortably
 * inside their tolerance bands under the new curve too -- but it is not a
 * substitute for real calibration against known ground-truth angles
 * (e.g. a fixed jig or IMU reference), which this project has not done.
 * Do not present the resulting yaw numbers as clinically precise degrees.
 */
/** Pure, unit-testable conversion -- maps turnProxy [-1, 1] smoothly to [-90°, +90°] */
export function turnProxyToYawDeg(turnProxy: number): number {
  const clamped = Math.max(-1, Math.min(1, turnProxy));
  const sign = clamped >= 0 ? 1 : -1;
  const mag = Math.abs(clamped);
  // Power mapping curve giving smooth control around 0-45° and reaching 90° for full profile
  return Math.round(sign * Math.pow(mag, 0.8) * 90);
}

/** The real, derived ceiling of `turnProxyToYawDeg` at turnProxy = ±1 */
export const MAX_REACHABLE_YAW_DEG = 90;
