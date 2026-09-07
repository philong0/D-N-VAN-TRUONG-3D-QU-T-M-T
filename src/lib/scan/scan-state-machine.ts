/**
 * scan-state-machine.ts
 *
 * The guided face scanner's explicit state machine (spec Part 4):
 *
 *   PRECHECK
 *     -> FRONT
 *     -> LEFT_TRANSITION -> LEFT_15 -> LEFT_30 -> LEFT_45
 *     -> RIGHT_TRANSITION -> RIGHT_15 -> RIGHT_30 -> RIGHT_45
 *     -> QUALITY_REVIEW -> COMPLETE
 *
 * Pure, DOM-free, React-free: `scanTick()` takes the current runtime state
 * plus one already-evaluated frame (see frame-evaluator.ts) and returns the
 * next state and what (if anything) to do with this tick — capture a frame,
 * advance a target, finish the scan. No state can be skipped: the only way
 * out of a stage is the explicit transition table below, driven by real
 * measured, temporally-stable evaluations — never a timer, never a manual
 * "mark complete".
 */
import type { FrameEvaluation } from "./frame-evaluator";
import { POSE, STABILITY } from "./scan-constants";

export interface AngleTarget {
  id: string;
  targetYaw: number;
  label: string;
  shortLabel: string;
  tolerance: number;
  requiredFrames: number;
  voicePrompt: string;
  direction: "left" | "right" | "center";
}

export const TARGET_ANGLES: AngleTarget[] = [
  { id: "front", targetYaw: 0, label: "0° CHÍNH DIỆN", shortLabel: "FRONT", tolerance: 10, requiredFrames: 2, voicePrompt: "Giữ đầu thẳng và nhìn vào camera.", direction: "center" },
  { id: "left_15", targetYaw: -15, label: "NGHIÊNG TRÁI 15°", shortLabel: "TRÁI 15°", tolerance: 10, requiredFrames: 2, voicePrompt: "Chậm rãi quay sang trái.", direction: "left" },
  { id: "left_30", targetYaw: -30, label: "NGHIÊNG TRÁI 30°", shortLabel: "TRÁI 30°", tolerance: 12, requiredFrames: 2, voicePrompt: "Tiếp tục quay sang trái.", direction: "left" },
  { id: "left_45", targetYaw: -45, label: "NGHIÊNG TRÁI 45°", shortLabel: "TRÁI 45°", tolerance: 14, requiredFrames: 2, voicePrompt: "Giữ ở góc nghiêng trái.", direction: "left" },
  { id: "right_15", targetYaw: 15, label: "NGHIÊNG PHẢI 15°", shortLabel: "PHẢI 15°", tolerance: 10, requiredFrames: 2, voicePrompt: "Chậm rãi quay sang phải.", direction: "right" },
  { id: "right_30", targetYaw: 30, label: "NGHIÊNG PHẢI 30°", shortLabel: "PHẢI 30°", tolerance: 12, requiredFrames: 2, voicePrompt: "Tiếp tục quay sang phải.", direction: "right" },
  { id: "right_45", targetYaw: 45, label: "NGHIÊNG PHẢI 45°", shortLabel: "PHẢI 45°", tolerance: 14, requiredFrames: 2, voicePrompt: "Giữ ở góc nghiêng phải.", direction: "right" },
];

export type StageId =
  | "precheck"
  | "front"
  | "left_transition"
  | "left_15"
  | "left_30"
  | "left_45"
  | "right_transition"
  | "right_15"
  | "right_30"
  | "right_45"
  | "quality_review"
  | "complete";

export interface CapturedFrameInfo {
  kind: "checkpoint" | "transition";
  targetId: string;
  targetYaw: number;
  angleError: number;
}

export interface ScanRuntimeState {
  stage: StageId;
  /** Index into TARGET_ANGLES. During a *_transition stage, this is the index of the checkpoint being approached. */
  targetIndex: number;
  precheckStableCount: number;
  candidateStableCount: number;
  acceptedByTarget: Record<string, number>;
  lastCaptureTsMs: number;
  lastTransitionYawByDirection: { left: number | null; right: number | null };
}

export function createInitialScanState(): ScanRuntimeState {
  const acceptedByTarget: Record<string, number> = {};
  for (const t of TARGET_ANGLES) acceptedByTarget[t.id] = 0;
  return {
    stage: "precheck",
    targetIndex: 0,
    precheckStableCount: 0,
    candidateStableCount: 0,
    acceptedByTarget,
    lastCaptureTsMs: 0,
    lastTransitionYawByDirection: { left: null, right: null },
  };
}

export interface TickResult {
  state: ScanRuntimeState;
  capture: CapturedFrameInfo | null;
  onTarget: boolean;
  targetCompleted: string | null;
  /** The NEW stage, only set on the tick where the stage actually changed. */
  enteredStage: StageId | null;
  scanCompleted: boolean;
}

function directionForStage(stage: StageId): "left" | "right" | null {
  if (stage === "left_transition") return "left";
  if (stage === "right_transition") return "right";
  return null;
}

function noChange(state: ScanRuntimeState, onTarget: boolean, capture: CapturedFrameInfo | null = null): TickResult {
  return { state, capture, onTarget, targetCompleted: null, enteredStage: null, scanCompleted: false };
}

/**
 * Advances the state machine by exactly one tick. `ev.angleError` must
 * already have been computed by the CALLER against whichever target angle
 * is currently relevant (TARGET_ANGLES[state.targetIndex].targetYaw for
 * every stage except "precheck", which uses 0) — this function never
 * recomputes it, so caller and state machine can never disagree about which
 * target a given tick was measured against.
 */
export function scanTick(state: ScanRuntimeState, ev: FrameEvaluation, nowMs: number): TickResult {
  if (state.stage === "precheck") {
    const onTarget = ev.accepted && ev.angleError <= POSE.PRECHECK_MAX_YAW_DEG;
    if (onTarget) {
      const precheckStableCount = state.precheckStableCount + 1;
      if (precheckStableCount >= STABILITY.PRECHECK_STABLE_TICKS) {
        const next: ScanRuntimeState = { ...state, stage: "front", targetIndex: 0, precheckStableCount: 0, candidateStableCount: 0 };
        return { state: next, capture: null, onTarget, targetCompleted: null, enteredStage: "front", scanCompleted: false };
      }
      return noChange({ ...state, precheckStableCount }, onTarget);
    }
    return noChange({ ...state, precheckStableCount: Math.max(0, state.precheckStableCount - 1) }, onTarget);
  }

  const transitionDirection = directionForStage(state.stage);
  if (transitionDirection) {
    const nextTarget = TARGET_ANGLES[state.targetIndex];
    const onNext = ev.accepted && ev.angleError <= nextTarget.tolerance;

    if (onNext) {
      const candidateStableCount = state.candidateStableCount + 1;
      if (candidateStableCount >= STABILITY.CHECKPOINT_STABLE_TICKS) {
        const next: ScanRuntimeState = { ...state, stage: nextTarget.id as StageId, candidateStableCount: 0 };
        return { state: next, capture: null, onTarget: true, targetCompleted: null, enteredStage: nextTarget.id as StageId, scanCompleted: false };
      }
      return noChange({ ...state, candidateStableCount }, true);
    }

    if (ev.accepted) {
      const lastYaw = state.lastTransitionYawByDirection[transitionDirection];
      const farEnough = lastYaw === null || Math.abs(ev.yaw - lastYaw) >= STABILITY.MIN_TRANSITION_YAW_DELTA_DEG;
      const timeOk = nowMs - state.lastCaptureTsMs >= STABILITY.MIN_MS_BETWEEN_CAPTURES;
      const resetState: ScanRuntimeState = { ...state, candidateStableCount: 0 };
      if (farEnough && timeOk) {
        const capture: CapturedFrameInfo = { kind: "transition", targetId: nextTarget.id, targetYaw: nextTarget.targetYaw, angleError: ev.angleError };
        const next: ScanRuntimeState = {
          ...resetState,
          lastCaptureTsMs: nowMs,
          lastTransitionYawByDirection: { ...state.lastTransitionYawByDirection, [transitionDirection]: ev.yaw },
        };
        return noChange(next, false, capture);
      }
      return noChange(resetState, false);
    }

    return noChange({ ...state, candidateStableCount: 0 }, false);
  }

  // A checkpoint target stage (front, left_15, left_30, left_45, right_15, right_30, right_45).
  const target = TARGET_ANGLES[state.targetIndex];
  const onTarget = ev.accepted && ev.angleError <= target.tolerance;

  if (!onTarget) {
    return noChange({ ...state, candidateStableCount: 0 }, false);
  }

  const candidateStableCount = state.candidateStableCount + 1;
  const readyToCapture = candidateStableCount >= STABILITY.CHECKPOINT_STABLE_TICKS
    && nowMs - state.lastCaptureTsMs >= STABILITY.MIN_MS_BETWEEN_CAPTURES;

  if (!readyToCapture) {
    return noChange({ ...state, candidateStableCount }, true);
  }

  const acceptedByTarget = { ...state.acceptedByTarget, [target.id]: (state.acceptedByTarget[target.id] || 0) + 1 };
  const capture: CapturedFrameInfo = { kind: "checkpoint", targetId: target.id, targetYaw: target.targetYaw, angleError: ev.angleError };

  if (acceptedByTarget[target.id] < target.requiredFrames) {
    const next: ScanRuntimeState = { ...state, acceptedByTarget, candidateStableCount: 0, lastCaptureTsMs: nowMs };
    return { state: next, capture, onTarget: true, targetCompleted: null, enteredStage: null, scanCompleted: false };
  }

  // This target just reached its required frame count -- advance.
  const nextIndex = state.targetIndex + 1;
  if (nextIndex >= TARGET_ANGLES.length) {
    const next: ScanRuntimeState = { ...state, acceptedByTarget, candidateStableCount: 0, lastCaptureTsMs: nowMs, stage: "quality_review" };
    return { state: next, capture, onTarget: true, targetCompleted: target.id, enteredStage: "quality_review", scanCompleted: true };
  }

  const nextTarget = TARGET_ANGLES[nextIndex];
  const isDirectionChange = target.direction !== nextTarget.direction && nextTarget.direction !== "center";
  const nextStage: StageId = isDirectionChange ? (nextTarget.direction === "left" ? "left_transition" : "right_transition") : (nextTarget.id as StageId);

  const next: ScanRuntimeState = {
    ...state,
    acceptedByTarget,
    candidateStableCount: 0,
    lastCaptureTsMs: nowMs,
    targetIndex: nextIndex,
    stage: nextStage,
  };
  return { state: next, capture, onTarget: true, targetCompleted: target.id, enteredStage: nextStage, scanCompleted: false };
}
