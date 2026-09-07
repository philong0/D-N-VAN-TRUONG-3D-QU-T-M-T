/**
 * camera-normalization.ts
 *
 * ONE place that turns raw camera-frame geometry into the scanner's
 * canonical head-pose convention: NEGATIVE yaw = subject's own real left,
 * POSITIVE yaw = subject's own real right. No other file may branch on
 * `cameraFacing` to flip a sign — see module docstring below for why the
 * math is the same for front AND rear.
 *
 * -----------------------------------------------------------------------
 * THE BUG THIS FIXES (2026-09-04 field report: front-camera left/right
 * felt backwards / scanner wouldn't advance no matter how the user turned)
 * -----------------------------------------------------------------------
 * `face-geometry.ts`'s raw turn signal is
 * `(noseX - eyeMidX) / eyeHalfSpan`, computed on the RAW sensor frame
 * (`ctx.drawImage(video, ...)` always draws the unmirrored buffer — CSS
 * `scale-x-[-1]` on the <video> element is a PREVIEW-only transform and
 * never touches what gets drawn to canvas, front or rear camera).
 *
 * Basic photography fact, independent of which physical camera module is
 * used: any camera that faces a subject from the front captures their real
 * RIGHT side at LOW pixel-x (image-left) -- the same reason a plain,
 * non-mirrored photo of a friend raising their right hand shows that hand
 * on the LEFT side of the photo (two things facing each other have
 * opposite left/right relative to the line between them). So when the
 * subject turns their head to their own real right, the nose tip moves
 * toward LOW x, making the raw turn signal NEGATIVE -- for a front-facing
 * selfie camera exactly as much as for a rear camera someone else is
 * holding up to the subject's face. This has nothing to do with preview
 * mirroring, which is display-only.
 *
 * The previous code applied `cameraFacing === "user" ? raw : -raw`,
 * flipping ONLY for rear. That made rear camera coincidentally correct and
 * left front camera (the default, most-used path) with the sign
 * backwards: a real right turn produced a negative ("left range") reading
 * and vice versa. The fix here flips the sign UNCONDITIONALLY, the same
 * way for both facings, because the underlying optics are the same.
 */

export type CameraFacing = "user" | "environment";

export interface CameraFrameContext {
  cameraFacing: CameraFacing;
  /** Whether the ON-SCREEN preview is CSS-mirrored for natural selfie UX. Display-only — must NEVER influence the pose math; carried here only so callers have one place to read it from instead of re-deriving it. */
  mirroredPreview: boolean;
  /** Degrees, clockwise, from the camera track's own reported orientation (`MediaStreamTrack.getSettings().rotation` where supported); 0 if unknown. */
  sensorOrientation?: number;
  /** Degrees, clockwise, from `screen.orientation.angle`; 0 if unknown. */
  displayOrientation?: number;
}

export interface CoordinateTransform {
  /** Always true today -- see module docstring. Kept explicit (not inlined as a bare negation at every call site) so the ONE place this is decided is self-documenting and independently testable. */
  flipYawSign: boolean;
  /** Degrees the raw frame's content is rotated relative to upright-portrait canonical orientation, derived from sensor vs. display orientation. 0 for the overwhelmingly common case where the browser already delivers an upright `<video>` frame. */
  rotationDeg: 0 | 90 | 180 | 270;
}

export interface CameraNormalizationResult {
  coordinateTransform: CoordinateTransform;
  yawConvention: "subject_left_negative_subject_right_positive";
}

/**
 * Computes the coordinate transform for a camera context. Pure function —
 * every head-pose/yaw computation in the scanner must route its raw
 * turn-signal through `applyYawSign` using the transform this returns,
 * instead of re-deriving its own front/rear special case.
 */
export function computeCameraNormalization(ctx: CameraFrameContext): CameraNormalizationResult {
  const flipYawSign = true; // see module docstring: true for BOTH facings, not a function of cameraFacing.

  const sensor = ((ctx.sensorOrientation ?? 0) % 360 + 360) % 360;
  const display = ((ctx.displayOrientation ?? 0) % 360 + 360) % 360;
  const net = ((sensor - display) % 360 + 360) % 360;
  const rotationDeg: CoordinateTransform["rotationDeg"] = net === 90 || net === 180 || net === 270 ? (net as 90 | 180 | 270) : 0;

  return {
    coordinateTransform: { flipYawSign, rotationDeg },
    yawConvention: "subject_left_negative_subject_right_positive",
  };
}

/** Applies the transform's yaw-sign decision to one raw turn-signal reading (the raw asin-degrees value, or the raw [-1,1] proxy — sign-only operation, works on either). */
export function applyYawSign(rawSignal: number, transform: CoordinateTransform): number {
  return transform.flipYawSign ? -rawSignal : rawSignal;
}

// ---------------------------------------------------------------------------
// Camera capability detection
// ---------------------------------------------------------------------------

export interface CameraCapabilities {
  hasFrontCamera: boolean;
  hasRearCamera: boolean;
  hasAnyCamera: boolean;
  permissionGranted: boolean;
  facingModeSupported: boolean;
  torchSupported: boolean;
}

/**
 * Real, feature-detected camera capabilities — never assumed. Runs
 * `enumerateDevices()` (needs at least one prior `getUserMedia` grant on
 * most browsers to expose device labels/kinds usefully) and probes torch
 * support via the currently-open track's capabilities when available.
 * Never throws: every check degrades to `false` on any error, so a caller
 * can always fall back to "just try to open whichever camera is left"
 * instead of crashing.
 */
export async function detectCameraCapabilities(activeStream?: MediaStream | null): Promise<CameraCapabilities> {
  const result: CameraCapabilities = {
    hasFrontCamera: false,
    hasRearCamera: false,
    hasAnyCamera: false,
    permissionGranted: false,
    facingModeSupported: false,
    torchSupported: false,
  };

  if (typeof navigator === "undefined" || !navigator.mediaDevices) return result;

  try {
    const devices = await navigator.mediaDevices.enumerateDevices();
    const videoInputs = devices.filter((d) => d.kind === "videoinput");
    result.hasAnyCamera = videoInputs.length > 0;
    // Device labels are only populated once permission has been granted at
    // least once; before that, `label` is empty and facing can't be told
    // apart from the label alone -- reported honestly as unknown rather
    // than guessed from device count/order.
    result.permissionGranted = videoInputs.some((d) => d.label.length > 0);
    const lower = videoInputs.map((d) => d.label.toLowerCase());
    result.hasFrontCamera = lower.some((l) => l.includes("front") || l.includes("user") || l.includes("trước"));
    result.hasRearCamera = lower.some((l) => l.includes("back") || l.includes("rear") || l.includes("environment") || l.includes("sau"));
    // Multiple cameras with no label match (permission not yet granted, or
    // non-descriptive labels) still very likely means both facings exist
    // on a phone -- but never CLAIM a specific facing exists without real
    // evidence; leave both false rather than guess.
  } catch {
    // enumerateDevices unsupported/blocked -- leave defaults.
  }

  try {
    result.facingModeSupported = Boolean(
      navigator.mediaDevices.getSupportedConstraints && navigator.mediaDevices.getSupportedConstraints().facingMode
    );
  } catch {
    // ignore
  }

  const track = activeStream?.getVideoTracks?.()[0];
  if (track) {
    try {
      const caps = track.getCapabilities?.();
      result.torchSupported = Boolean(caps && "torch" in caps);
    } catch {
      // getCapabilities unsupported on this browser -- leave false.
    }
  }

  return result;
}
