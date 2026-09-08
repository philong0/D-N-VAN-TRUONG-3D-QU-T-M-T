"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import Link from "next/link";
import { analyzeFaceFrame, type FrameFaceAnalysis } from "@/lib/face-geometry";
import { computeCameraNormalization, applyYawSign, detectCameraCapabilities, type CameraCapabilities } from "@/lib/scan/camera-normalization";
import { evaluateFrame, type FrameMeasurement, type FrameEvaluation } from "@/lib/scan/frame-evaluator";
import { turnProxyToYawDeg } from "@/lib/scan/pose-mapping";
import type { ScanSession } from "@/lib/types";

export interface TargetAngleConfig {
  id: string;
  name: string;
  filename: string;
  targetYaw: number;
  tolerance: number;
  prompt: string;
  retryPrompt: string;
}

export const TARGET_ANGLES: Record<string, TargetAngleConfig> = {
  FRONT: { id: "FRONT", name: "CHÍNH DIỆN 0°", filename: "front_origin.jpg", targetYaw: 0, tolerance: 18, prompt: "Nhìn thẳng vào camera", retryPrompt: "Giữ đầu thẳng tự nhiên" },
  LEFT45: { id: "LEFT45", name: "NGHIÊNG TRÁI 45°", filename: "left45_origin.jpg", targetYaw: -45, tolerance: 22, prompt: "Nghiêng mặt sang Trái 45 độ", retryPrompt: "Xoay nhẹ mặt sang trái" },
  // D-realistictarget — the capture-accept check further down this file
  // only ever required |yaw| >= 50° for these two checkpoints (the actual
  // gating logic, unchanged) — but the label/prompt here said "80°" and
  // "quay ngang hẳn" (turn all the way sideways), which is both physically
  // awkward to judge one-handed and simply not what the code was checking
  // for. Users were fighting to reach an angle 30°+ past what was ever
  // required. Target/copy now matches the real ~50-55° the gate accepts.
  LEFT80: { id: "LEFT80", name: "TRẮC DIỆN TRÁI ~55°", filename: "left80_origin.jpg", targetYaw: -55, tolerance: 25, prompt: "Nghiêng sâu sang Trái, để lộ rõ gò má và sống mũi", retryPrompt: "Nghiêng thêm chút nữa sang trái" },
  RIGHT45: { id: "RIGHT45", name: "NGHIÊNG PHẢI 45°", filename: "right45_origin.jpg", targetYaw: 45, tolerance: 22, prompt: "Nghiêng mặt sang Phải 45 độ", retryPrompt: "Xoay nhẹ mặt sang phải" },
  RIGHT80: { id: "RIGHT80", name: "TRẮC DIỆN PHẢI ~55°", filename: "right80_origin.jpg", targetYaw: 55, tolerance: 25, prompt: "Nghiêng sâu sang Phải, để lộ rõ gò má và sống mũi", retryPrompt: "Nghiêng thêm chút nữa sang phải" },
};

const ORDERED_TARGET_KEYS = ["FRONT", "LEFT45", "LEFT80", "RIGHT45", "RIGHT80"] as const;
type TargetKey = (typeof ORDERED_TARGET_KEYS)[number] | "DONE";
const MAX_BUFFER_FRAMES = 50;
const RECONSTRUCTION_TARGET_COUNT = 10;
const TIMEOUT_DURATION = 4.0; // giây trước khi nhắc lại / cho phép bỏ qua thủ công

type ScanStage = "scanning" | "review" | "uploading" | "result" | "error";

interface CapturedPhotoItem {
  key: string;
  name: string;
  filename: string;
  url: string;
  qualityScore: number;
  yaw: number;
  pitch: number;
}

export default function GuidedFaceScan({ patientId }: { patientId: string }) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);

  const [stage, setStage] = useState<ScanStage>("scanning");
  const [session, setSession] = useState<ScanSession | null>(null);
  const [cameraFacing, setCameraFacing] = useState<"user" | "environment">("user");
  const [mediaStream, setMediaStream] = useState<MediaStream | null>(null);
  const [isStartingCamera, setIsStartingCamera] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [voiceEnabled, setVoiceEnabled] = useState(true);
  const [isScanStarted, setIsScanStarted] = useState(true);
  const [holdProgress, setHoldProgress] = useState(0);

  // Header slide in/out state
  const [isHeaderOpen, setIsHeaderOpen] = useState(true);

  // Review & Zoom Modal state
  const [reviewPhotos, setReviewPhotos] = useState<CapturedPhotoItem[]>([]);
  const [zoomedPhoto, setZoomedPhoto] = useState<CapturedPhotoItem | null>(null);

  // Targets state: 7 yaw checkpoints
  const buildInitialTargetsState = (): Record<string, { captured: boolean; filename: string }> => {
    const init: Record<string, { captured: boolean; filename: string }> = {};
    for (const key of ORDERED_TARGET_KEYS) init[key] = { captured: false, filename: TARGET_ANGLES[key].filename };
    return init;
  };
  const [targetsState, setTargetsState] = useState<Record<string, { captured: boolean; filename: string }>>(buildInitialTargetsState);
  const targetsStateRef = useRef(targetsState);

  const [currentTargetKey, setCurrentTargetKey] = useState<TargetKey>("FRONT");
  const currentTargetKeyRef = useRef<TargetKey>("FRONT");

  // Real-time Telemetry (Pitch & Yaw)
  const [telemetry, setTelemetry] = useState({ pitch: 0, yaw: 0 });
  const [voiceInstruction, setVoiceInstruction] = useState<string>(`GỢI Ý GIỌNG NÓI: ${TARGET_ANGLES.FRONT.prompt}`);
  const [hapticLog, setHapticLog] = useState<string | null>(null);
  const [bufferCount, setBufferCount] = useState<number>(0);

  // Frame Pools
  const frameBufferBlobsRef = useRef<Array<{ blob: Blob; metadata: Record<string, unknown> }>>([]);
  const originFramesRef = useRef<Record<string, { blob: Blob; metadata: Record<string, unknown> }>>({});

  const lastStateChangeTimeRef = useRef<number>(0);
  const retryCountRef = useRef<number>(0);
  const stableTargetTicksRef = useRef<number>(0);
  const isFinalizingRef = useRef(false);

  const [, setCameraCapabilities] = useState<CameraCapabilities | null>(null);
  const sensorOrientationRef = useRef<number>(0);
  const displayOrientationRef = useRef<number>(0);
  const [noFaceRearWarning, setNoFaceRearWarning] = useState(false);
  const noFaceSinceRef = useRef<number | null>(null);
  const lastFaceCenterRef = useRef<{ x: number; y: number } | null>(null);
  const [lastEvaluation, setLastEvaluation] = useState<FrameEvaluation | null>(null);
  const [canManuallySkip, setCanManuallySkip] = useState(false);

  // Native ARKit TrueDepth Bridge State
  // D-hydrationfix — the previous version read `window.webkit.messageHandlers
  // .arkitScanBridge` via a lazy `useState` initializer, reasoning that
  // since the bridge is injected before this page's script runs, the read
  // is "stable." That's true for a plain client-only render, but Next.js
  // SSR-renders this component on the SERVER first (`window` undefined ->
  // `false`), then hydrates on the CLIENT — and inside the native app's
  // WKWebView the bridge genuinely IS present at that first client render,
  // so the initializer returns `true` there. Server said `false`, client's
  // first render said `true`: a real hydration mismatch (exactly the error
  // seen in production — "Hydration failed because the server rendered
  // HTML didn't match the client"). React recovers by discarding and
  // regenerating the whole tree, which can orphan in-flight scan state
  // (camera stream, capture step, bridge callbacks) — a plausible cause of
  // scans that finish capturing but never reach reconstruction.
  // Fix: always render the SSR-safe default (`false`) on the first client
  // render too, matching the server exactly, then flip it in a `useEffect`
  // (which only runs post-hydration — this is the standard, hydration-safe
  // pattern for browser/environment feature detection, not the same class
  // of "unnecessary setState in effect" the prior comment was avoiding).
  // Detection itself is folded into the mount effect below (D-arkitfirst)
  // so it can gate whether the web camera ever starts at all.
  const [isArkitAvailable, setIsArkitAvailable] = useState(false);
  const [isArkitScanning, setIsArkitScanning] = useState(false);

  const lastSpokenTextRef = useRef<string>("");
  const lastSpokenTimeRef = useRef<number>(0);
  const audioCtxRef = useRef<AudioContext | null>(null);

  useEffect(() => {
    targetsStateRef.current = targetsState;
  }, [targetsState]);

  useEffect(() => {
    currentTargetKeyRef.current = currentTargetKey;
  }, [currentTargetKey]);

  // Haptic feedback & Web Vibration API
  const triggerHaptic = useCallback((type: "light" | "strong", logMsg: string) => {
    setHapticLog(logMsg);
    if (typeof window !== "undefined" && navigator?.vibrate) {
      try {
        if (type === "light") {
          navigator.vibrate(80);
        } else {
          navigator.vibrate([150, 100, 250]);
        }
      } catch {
        // ignore
      }
    }
  }, []);

  const playChime = useCallback((freq = 880, duration = 0.15) => {
    try {
      if (!audioCtxRef.current) {
        const AudioCtx = window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
        if (AudioCtx) audioCtxRef.current = new AudioCtx();
      }
      const ctx = audioCtxRef.current;
      if (ctx) {
        if (ctx.state === "suspended") ctx.resume();
        const osc = ctx.createOscillator();
        const gain = ctx.createGain();
        osc.type = "sine";
        osc.frequency.setValueAtTime(freq, ctx.currentTime);
        gain.gain.setValueAtTime(0.2, ctx.currentTime);
        gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + duration);
        osc.connect(gain);
        gain.connect(ctx.destination);
        osc.start();
        osc.stop(ctx.currentTime + duration);
      }
    } catch {
      // audio policy
    }
  }, []);

  const speakGuidance = useCallback((text: string, force = false) => {
    if (!voiceEnabled || typeof window === "undefined" || !("speechSynthesis" in window)) return;
    const now = performance.now();
    if (!force && lastSpokenTextRef.current === text && now - lastSpokenTimeRef.current < 2500) return;
    if (!force && now - lastSpokenTimeRef.current < 1500) return;

    try {
      window.speechSynthesis.cancel();
      const utterance = new SpeechSynthesisUtterance(text);
      utterance.lang = "vi-VN";
      utterance.rate = 1.05;
      utterance.pitch = 1.0;
      lastSpokenTextRef.current = text;
      lastSpokenTimeRef.current = now;
      window.speechSynthesis.speak(utterance);
    } catch {
      // speech
    }
  }, [voiceEnabled]);

  const stopCamera = useCallback(() => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }
    setMediaStream(null);
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
  }, []);

  useEffect(() => {
    return () => {
      stopCamera();
      if (typeof window !== "undefined" && "speechSynthesis" in window) {
        window.speechSynthesis.cancel();
      }
    };
  }, [stopCamera]);

  useEffect(() => {
    if (typeof window === "undefined") return;
    const readAngle = () => {
      try {
        displayOrientationRef.current = window.screen?.orientation?.angle ?? 0;
      } catch {
        displayOrientationRef.current = 0;
      }
    };
    readAngle();
    window.screen?.orientation?.addEventListener?.("change", readAngle);
    return () => window.screen?.orientation?.removeEventListener?.("change", readAngle);
  }, []);

  const startCamera = useCallback(async (facing: "user" | "environment") => {
    stopCamera();
    setErrorMessage(null);
    setIsStartingCamera(true);
    try {
      if (typeof navigator === "undefined" || !navigator.mediaDevices?.getUserMedia) {
        throw new Error("Trình duyệt không hỗ trợ mở camera.");
      }

      let stream: MediaStream | null = null;
      let actualFacing: "user" | "environment" = facing;
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: { ideal: facing }, width: { ideal: 1280 }, height: { ideal: 720 } },
          audio: false,
        });
      } catch {
        const fallbackFacing = facing === "user" ? "environment" : "user";
        try {
          stream = await navigator.mediaDevices.getUserMedia({
            video: { facingMode: { ideal: fallbackFacing } },
            audio: false,
          });
          actualFacing = fallbackFacing;
        } catch {
          stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
        }
      }

      if (stream) {
        streamRef.current = stream;
        setMediaStream(stream);
        if (actualFacing !== facing) {
          setCameraFacing(actualFacing);
          setErrorMessage(
            actualFacing === "user"
              ? "Không tìm thấy camera sau — đã chuyển sang camera trước."
              : "Không tìm thấy camera trước — đã chuyển sang camera sau."
          );
        }
        if (videoRef.current) {
          const v = videoRef.current;
          v.srcObject = stream;
          v.setAttribute("playsinline", "true");
          v.setAttribute("webkit-playsinline", "true");
          v.setAttribute("autoplay", "true");
          v.setAttribute("muted", "true");
          v.muted = true;
          v.onloadedmetadata = () => {
            v.play().catch(() => {});
          };
          v.play().catch(() => {});
        }
        if (actualFacing === facing) setErrorMessage(null);

        try {
          const track = stream.getVideoTracks()[0];
          const settings = track?.getSettings?.() as (MediaTrackSettings & { rotation?: number }) | undefined;
          sensorOrientationRef.current = typeof settings?.rotation === "number" ? settings.rotation : 0;
        } catch {
          sensorOrientationRef.current = 0;
        }

        detectCameraCapabilities(stream)
          .then(setCameraCapabilities)
          .catch(() => setCameraCapabilities(null));
      }
    } catch (err) {
      console.warn("Camera init error:", err);
      setErrorMessage("Không thể mở Camera. Vui lòng cho phép quyền truy cập Camera trong Cài đặt trình duyệt.");
    } finally {
      setIsStartingCamera(false);
    }
  }, [stopCamera]);

  // Khởi tạo phiên quét
  const initSession = useCallback(async () => {
    try {
      const res = await fetch(`/api/patients/${patientId}/scan-sessions`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ scannerKind: "web_camera" }),
      });
      const created = await res.json();
      if (created?.session?.id) {
        const startRes = await fetch(`/api/patients/${patientId}/scan-sessions/${created.session.id}`, {
          method: "PATCH",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ action: "start" }),
        });
        const started = await startRes.json();
        if (started?.session) {
          setSession(started.session);
        }
      }
    } catch (err) {
      console.warn("Session init error:", err);
    }
  }, [patientId]);

  // Kích hoạt quét ARKit TrueDepth Native 3D của iOS
  const startNativeArkitScan = useCallback(async () => {
    try {
      setIsArkitScanning(true);
      stopCamera();

      // 1. Tạo session ios_native
      const res = await fetch(`/api/patients/${patientId}/scan-sessions`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ scannerKind: "ios_native" }),
      });
      const data = await res.json();
      const currentSession = data.session;
      if (!currentSession?.id) throw new Error("Không khởi tạo được phiên quét TrueDepth.");

      setSession(currentSession);

      // 2. Cài đặt bridge callback
      const w = window as unknown as {
        webkit?: { messageHandlers?: { arkitScanBridge?: { postMessage: (b: unknown) => void } } };
        __arkitBridgeResolve?: (reqId: string, payload: unknown) => void;
        __arkitBridgeReject?: (reqId: string, err: string) => void;
        __arkitBridgePending?: Map<string, { resolve: (v: unknown) => void; reject: (e: Error) => void }>;
      };

      if (!w.__arkitBridgePending) {
        w.__arkitBridgePending = new Map();
        w.__arkitBridgeResolve = (reqId, payload) => {
          const p = w.__arkitBridgePending?.get(reqId);
          if (p) {
            w.__arkitBridgePending?.delete(reqId);
            p.resolve(payload);
          }
        };
        w.__arkitBridgeReject = (reqId, msg) => {
          const p = w.__arkitBridgePending?.get(reqId);
          if (p) {
            w.__arkitBridgePending?.delete(reqId);
            p.reject(new Error(msg));
          }
        };
      }

      const reqId = `arkit-${Date.now()}`;
      const bridgePromise = new Promise((resolve, reject) => {
        w.__arkitBridgePending?.set(reqId, { resolve, reject });
      });

      // 3. Gửi lệnh cho native app mở ARFaceScannerView toàn màn hình
      w.webkit?.messageHandlers?.arkitScanBridge?.postMessage({
        action: "start",
        requestId: reqId,
        patientId,
        sessionId: currentSession.id,
      });

      // 4. Chờ quét 5 góc và upload package xong
      await bridgePromise;

      // 5. Chuyển thẳng vào 3D Studio để xem mô hình 3D lập tức
      window.location.href = `/patients/${patientId}/studio`;
    } catch (err) {
      console.warn("Native ARKit scan error:", err);
      setIsArkitScanning(false);
      startCamera(cameraFacing);
    }
  }, [cameraFacing, patientId, stopCamera]);

  // D-arkitfirst — a prior version of this effect ALWAYS started the plain
  // web camera + created a `web_camera` session here unconditionally, with
  // native TrueDepth capture only reachable via a separate button the user
  // had to notice and tap in time. Since the web-camera capture loop
  // auto-progresses and can auto-finish on its own (see the target-angle
  // effect below), it would frequently win the race and get uploaded/
  // reconstructed even when the user genuinely scanned with TrueDepth —
  // producing a `web_camera` session and a template-limited (or QC-
  // rejected) result instead of the real ARKit one, with no visible sign
  // anything went wrong. Fix: detect the native bridge FIRST; if present,
  // launch the real TrueDepth scanner immediately and never start the web
  // camera at all — the two paths must never run concurrently.
  useEffect(() => {
    let cancelled = false;
    const timer = setTimeout(() => {
      if (cancelled) return;
      const hasArkitBridge = Boolean(
        (window as unknown as { webkit?: { messageHandlers?: { arkitScanBridge?: unknown } } })
          ?.webkit?.messageHandlers?.arkitScanBridge
      );
      setIsArkitAvailable(hasArkitBridge);
      if (hasArkitBridge) {
        startNativeArkitScan();
      } else {
        startCamera(cameraFacing);
        initSession();
      }
    }, 0);
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const handleStartScan = useCallback(() => {
    setIsScanStarted(true);
    playChime(660, 0.2);
    triggerHaptic("strong", "Bắt đầu quét khuôn mặt");
    speakGuidance("Bắt đầu quét! Giữ đầu thẳng, mở to mắt tự nhiên, nhìn vào camera.", true);
    lastStateChangeTimeRef.current = Date.now();
  }, [playChime, triggerHaptic, speakGuidance]);

  // Chụp ảnh góc hiện tại ngay lập tức thủ công
  const manualSnapCurrentTarget = useCallback(() => {
    const key = currentTargetKeyRef.current;
    if (key === "DONE") return;
    const video = videoRef.current;
    if (!video || video.videoWidth === 0) return;

    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    canvas.toBlob((blob) => {
      if (!blob) return;
      originFramesRef.current[key] = {
        blob,
        metadata: {
          yaw: telemetry.yaw,
          pitch: telemetry.pitch,
          faceDetected: true,
          qualityScore: 95,
          userSnapped: true,
          timestampMs: Date.now(),
          cameraFacing,
          sensorOrientation: sensorOrientationRef.current,
          displayOrientation: displayOrientationRef.current,
          mirrorApplied: cameraFacing === "user",
        },
      };
    }, "image/jpeg", 0.95);

    const updated = { ...targetsStateRef.current, [key]: { ...targetsStateRef.current[key], captured: true } };
    targetsStateRef.current = updated;
    setTargetsState(updated);
    playChime(880, 0.15);
    triggerHaptic("light", `[ĐÃ CHỤP] -> ${TARGET_ANGLES[key]?.name}`);

    stableTargetTicksRef.current = 0;
    setHoldProgress(0);
    lastStateChangeTimeRef.current = Date.now();
    retryCountRef.current = 0;

    const remaining = ORDERED_TARGET_KEYS.filter((k) => !updated[k]?.captured);
    if (remaining.length > 0) {
      speakGuidance(TARGET_ANGLES[remaining[0]].prompt, true);
    }
  }, [cameraFacing, playChime, speakGuidance, telemetry, triggerHaptic]);

  // Bỏ qua góc hiện tại thủ công
  const manualSkipCurrentTarget = useCallback(() => {
    const key = currentTargetKeyRef.current;
    if (key === "DONE") return;
    const video = videoRef.current;
    if (!video || video.videoWidth === 0) return;

    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    canvas.toBlob((blob) => {
      if (!blob) return;
      originFramesRef.current[key] = {
        blob,
        metadata: {
          yaw: telemetry.yaw,
          pitch: telemetry.pitch,
          faceDetected: Boolean(lastEvaluation?.rejectReasons.every((r) => r !== "no_face")),
          qualityScore: lastEvaluation?.score ?? 0,
          userSkipped: true,
          timestampMs: Date.now(),
          cameraFacing,
          sensorOrientation: sensorOrientationRef.current,
          displayOrientation: displayOrientationRef.current,
          mirrorApplied: cameraFacing === "user",
        },
      };
    }, "image/jpeg", 0.95);

    const updated = { ...targetsStateRef.current, [key]: { ...targetsStateRef.current[key], captured: true } };
    targetsStateRef.current = updated;
    setTargetsState(updated);
    setCanManuallySkip(false);
    retryCountRef.current = 0;
    stableTargetTicksRef.current = 0;
    lastStateChangeTimeRef.current = Date.now();

    const remaining = ORDERED_TARGET_KEYS.filter((k) => !updated[k]?.captured);
    if (remaining.length > 0) speakGuidance(TARGET_ANGLES[remaining[0]].prompt, true);
  }, [cameraFacing, lastEvaluation, speakGuidance, telemetry]);

  // Chuyển sang màn hình Thẩm định ảnh gốc (Photo Review)
  const openReviewGallery = useCallback(() => {
    stopCamera();
    triggerHaptic("strong", "[HAPTIC RUNG MẠNH] >>> ĐÃ QUÉT ĐỦ CÁC GÓC! <<<");
    playChime(1046, 0.4);
    speakGuidance("Quét hoàn tất! Vui lòng kiểm tra lại ảnh chụp các góc trước khi khởi tạo 3D.", true);

    const photos: CapturedPhotoItem[] = [];
    for (const key of ORDERED_TARGET_KEYS) {
      const entry = originFramesRef.current[key];
      if (entry) {
        const url = URL.createObjectURL(entry.blob);
        const meta = entry.metadata;
        photos.push({
          key,
          name: TARGET_ANGLES[key]?.name || key,
          filename: TARGET_ANGLES[key]?.filename || `${key.toLowerCase()}_origin.jpg`,
          url,
          qualityScore: typeof meta.qualityScore === "number" ? meta.qualityScore : 90,
          yaw: typeof meta.yaw === "number" ? meta.yaw : TARGET_ANGLES[key]?.targetYaw || 0,
          pitch: typeof meta.pitch === "number" ? meta.pitch : 0,
        });
      }
    }
    setReviewPhotos(photos);
    setStage("review");
  }, [playChime, speakGuidance, stopCamera, triggerHaptic]);

  // Upload và hoàn tất dựng 3D Mesh
  const finalizeScan = useCallback(async () => {
    if (isFinalizingRef.current || !session) return;
    isFinalizingRef.current = true;
    setStage("uploading");
    stopCamera();

    try {
      // 1. Tải toàn bộ frame buffer + ảnh origin lên server
      const form = new FormData();
      for (const item of frameBufferBlobsRef.current) {
        form.append("frames", item.blob, "burst.jpg");
        form.append("metadata", JSON.stringify(item.metadata));
      }

      for (const [key, entry] of Object.entries(originFramesRef.current)) {
        form.append("frames", entry.blob, `${key.toLowerCase()}_origin.jpg`);
        form.append("metadata", JSON.stringify({ originKey: key, targetId: key, ...entry.metadata }));
      }

      await fetch(`/api/patients/${patientId}/scan-sessions/${session.id}/burst-frames`, {
        method: "POST",
        body: form,
      });

      // 2. Finalize session
      const finalizeRes = await fetch(`/api/patients/${patientId}/scan-sessions/${session.id}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ action: "finalize" }),
      });
      const data = await finalizeRes.json();
      let finalSession = data.session || session;

      // 3. Request Reconstruction
      try {
        const reconRes = await fetch(`/api/patients/${patientId}/scan-sessions/${finalSession.id}`, {
          method: "PATCH",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ action: "request_reconstruction" }),
        });
        const reconData = await reconRes.json();
        if (reconRes.ok && reconData.session) {
          finalSession = reconData.session;
        }
      } catch (reconErr) {
        console.warn("Reconstruction trigger error:", reconErr);
      }

      // 4. Chọn 4 ảnh profile preview đồng bộ vào hồ sơ
      try {
        await fetch(`/api/patients/${patientId}/profile-preview`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ sessionId: finalSession.id }),
        });
      } catch (previewErr) {
        console.warn("Profile preview selection error:", previewErr);
      }

      setSession(finalSession);
      setStage("result");
    } catch (error) {
      setErrorMessage(error instanceof Error ? error.message : "Lỗi lưu dữ liệu phiên scan.");
      setStage("error");
    }
  }, [patientId, session, stopCamera]);

  // Vòng lặp nhận diện góc (Head Pose & Quality Gate mỗi 120ms)
  useEffect(() => {
    if (stage !== "scanning" || !mediaStream) return;

    const sampleCanvas = document.createElement("canvas");
    sampleCanvas.width = 320;
    sampleCanvas.height = 240;
    const sctx = sampleCanvas.getContext("2d", { willReadFrequently: true });
    if (!sctx) return;

    const interval = setInterval(async () => {
      if (isFinalizingRef.current) return;
      const video = videoRef.current;
      if (!video || video.readyState < 2 || video.videoWidth === 0) return;

      sctx.drawImage(video, 0, 0, sampleCanvas.width, sampleCanvas.height);

      let analysis: FrameFaceAnalysis | null = null;
      try {
        analysis = await analyzeFaceFrame(sampleCanvas);
      } catch {
        analysis = null;
      }

      const currentTargets = targetsStateRef.current;
      const activeTargets = ORDERED_TARGET_KEYS.filter((k) => !currentTargets[k]?.captured);

      let activeTargetKey: TargetKey = "DONE";
      if (activeTargets.length > 0) {
        activeTargetKey = activeTargets[0];
        setCurrentTargetKey(activeTargetKey);
      } else {
        setCurrentTargetKey("DONE");
      }

      if (!analysis || !analysis.faceDetected) {
        if (noFaceSinceRef.current === null) noFaceSinceRef.current = Date.now();
        else if (Date.now() - noFaceSinceRef.current > 1500) {
          setNoFaceRearWarning(true);
          speakGuidance("Không nhận diện được khuôn mặt. Vui lòng hướng camera vào mặt.");
        }
        lastFaceCenterRef.current = null;
        return;
      }
      noFaceSinceRef.current = null;
      setNoFaceRearWarning(false);

      const calculatedYaw = turnProxyToYawDeg(analysis.turnProxy);
      const calculatedPitch = Math.round(analysis.pitchDeg || 0);

      const { coordinateTransform } = computeCameraNormalization({
        cameraFacing,
        mirroredPreview: cameraFacing === "user",
        sensorOrientation: sensorOrientationRef.current,
        displayOrientation: displayOrientationRef.current,
      });
      const normalizedYaw = applyYawSign(calculatedYaw, coordinateTransform);
      const normalizedPitch = calculatedPitch;

      setTelemetry({ pitch: normalizedPitch, yaw: normalizedYaw });

      const faceCenter = { x: (analysis.faceBounds.left + analysis.faceBounds.right) / 2, y: (analysis.faceBounds.top + analysis.faceBounds.bottom) / 2 };
      const motion = lastFaceCenterRef.current
        ? Math.hypot(faceCenter.x - lastFaceCenterRef.current.x, faceCenter.y - lastFaceCenterRef.current.y) * 100
        : 0;
      lastFaceCenterRef.current = faceCenter;

      const measurement: FrameMeasurement = {
        faceDetected: analysis.faceDetected,
        landmarksReal: analysis.landmarksReal,
        yaw: normalizedYaw,
        pitch: normalizedPitch,
        roll: analysis.landmarksReal ? analysis.rollDeg : 0,
        faceBounds: analysis.faceBounds,
        brightness: analysis.brightness,
        sharpness: analysis.sharpness,
        motion,
        landmarkStability: analysis.landmarkStability,
      };
      const targetYawForTick = activeTargetKey === "DONE" ? 0 : TARGET_ANGLES[activeTargetKey].targetYaw;
      const evaluation: FrameEvaluation = evaluateFrame(measurement, targetYawForTick);
      setLastEvaluation(evaluation);

      const cameraMeta = {
        cameraFacing,
        sensorOrientation: sensorOrientationRef.current,
        displayOrientation: displayOrientationRef.current,
        mirrorApplied: cameraFacing === "user",
        coordinateTransform,
      };
      const frameMetadata = {
        faceDetected: true,
        targetId: activeTargetKey === "DONE" ? null : activeTargetKey,
        yaw: normalizedYaw,
        pitch: normalizedPitch,
        roll: measurement.roll,
        qualityScore: evaluation.score,
        sharpness: evaluation.sharpness,
        brightness: evaluation.brightness,
        motion: evaluation.motion,
        landmarkStability: evaluation.landmarkStability,
        landmarksReal: analysis.landmarksReal,
        timestampMs: Date.now(),
        ...cameraMeta,
      };

      // Thêm frame vào bộ đệm reconstruction
      const fullCanvas = document.createElement("canvas");
      fullCanvas.width = video.videoWidth || 1280;
      fullCanvas.height = video.videoHeight || 720;
      const fctx = fullCanvas.getContext("2d");

      if (fctx && (evaluation.accepted || analysis.faceDetected) && frameBufferBlobsRef.current.length < MAX_BUFFER_FRAMES) {
        fctx.drawImage(video, 0, 0, fullCanvas.width, fullCanvas.height);
        fullCanvas.toBlob(
          (blob) => {
            if (blob && frameBufferBlobsRef.current.length < MAX_BUFFER_FRAMES) {
              frameBufferBlobsRef.current.push({ blob, metadata: frameMetadata });
              setBufferCount(frameBufferBlobsRef.current.length);
            }
          },
          "image/jpeg",
          0.85
        );
      }

      // Kiểm tra góc active với khoảng sai số thông minh & siêu nhạy (9 góc: 0, ±20, ±45, ±60, ±80)
      if (activeTargetKey !== "DONE") {
        const targetCfg = TARGET_ANGLES[activeTargetKey];
        
        // Xác định góc lọt vào vùng quét mục tiêu
        let isAngleMatch = false;
        if (activeTargetKey === "FRONT") {
          isAngleMatch = Math.abs(normalizedYaw) <= 18;
        } else if (activeTargetKey === "LEFT45") {
          isAngleMatch = normalizedYaw <= -22 && normalizedYaw >= -68 && (analysis.landmarksReal || Math.abs(normalizedYaw) >= 30);
        } else if (activeTargetKey === "LEFT80") {
          isAngleMatch = normalizedYaw <= -50;
        } else if (activeTargetKey === "RIGHT45") {
          isAngleMatch = normalizedYaw >= 22 && normalizedYaw <= 68 && (analysis.landmarksReal || Math.abs(normalizedYaw) >= 30);
        } else if (activeTargetKey === "RIGHT80") {
          isAngleMatch = normalizedYaw >= 50;
        }

        // Ưu tiên chất lượng: nhận diện mặt, đủ sáng
        const isQualityAcceptable = analysis.faceDetected && analysis.brightness >= 8;

        const onTarget = isAngleMatch && isQualityAcceptable;
        const REQUIRED_STABLE_TICKS = 5; // ~600ms giữ yên chính xác, chống nhảy bước

        setVoiceInstruction(`GỢI Ý GIỌNG NÓI: ${targetCfg.prompt}`);
        setCanManuallySkip(false);

        if (onTarget) {
          stableTargetTicksRef.current += 1;
          const progress = Math.min(100, Math.round((stableTargetTicksRef.current / REQUIRED_STABLE_TICKS) * 100));
          setHoldProgress(progress);

          // Khi giữ yên đủ -> Chụp frame gốc
          if (stableTargetTicksRef.current >= REQUIRED_STABLE_TICKS) {
            stableTargetTicksRef.current = 0;
            setHoldProgress(0);

            if (fctx) {
              fctx.drawImage(video, 0, 0, fullCanvas.width, fullCanvas.height);
              fullCanvas.toBlob((blob) => {
                if (blob) originFramesRef.current[activeTargetKey] = { blob, metadata: frameMetadata };
              }, "image/jpeg", 0.95);
            }

            const updated = { ...targetsStateRef.current };
            updated[activeTargetKey] = { ...updated[activeTargetKey], captured: true };
            targetsStateRef.current = updated;
            setTargetsState(updated);

            playChime(880, 0.15);
            triggerHaptic("light", `[ĐÃ CHỤP NÉT] -> ${targetCfg.name}`);

            lastStateChangeTimeRef.current = Date.now();
            retryCountRef.current = 0;

            const remaining = ORDERED_TARGET_KEYS.filter((k) => !updated[k]?.captured);
            if (remaining.length > 0) {
              const nextTarget = TARGET_ANGLES[remaining[0]];
              speakGuidance(nextTarget.prompt, true);
            } else {
              // Đã chụp đủ cả 5 góc -> Chuyển ngay sang Review & Dựng 3D
              clearInterval(interval);
              setTimeout(() => {
                openReviewGallery();
              }, 400);
              return;
            }
          }
        } else {
          // Khi trượt tick nhẹ, giảm trừ dần chứ không xóa về 0 ngay lập tức
          stableTargetTicksRef.current = Math.max(0, stableTargetTicksRef.current - 1);
          setHoldProgress(Math.min(100, Math.round((stableTargetTicksRef.current / REQUIRED_STABLE_TICKS) * 100)));
          const currentTime = Date.now();
          if ((currentTime - lastStateChangeTimeRef.current) / 1000 > TIMEOUT_DURATION) {
            retryCountRef.current += 1;
            lastStateChangeTimeRef.current = currentTime;

            if (retryCountRef.current === 1) {
              speakGuidance(targetCfg.retryPrompt, true);
            } else if (retryCountRef.current >= 2) {
              setCanManuallySkip(true);
              speakGuidance("Nếu khó giữ đúng góc này, bạn có thể bấm nút chụp hoặc bỏ qua bên dưới.", true);
            }
          }
        }
      }

      // Chuyển sang màn hình thẩm định ảnh khi đã thu đủ các góc
      const allCaptured = ORDERED_TARGET_KEYS.every((k) => targetsStateRef.current[k]?.captured);
      if (allCaptured && !isFinalizingRef.current) {
        clearInterval(interval);
        openReviewGallery();
      }
    }, 120);

    return () => clearInterval(interval);
  }, [stage, mediaStream, session, cameraFacing, openReviewGallery, playChime, speakGuidance, triggerHaptic]);

  const capturedCount = ORDERED_TARGET_KEYS.filter((k) => targetsState[k]?.captured).length;
  const isDone = capturedCount === ORDERED_TARGET_KEYS.length;

  return (
    <div className="fixed inset-0 z-[99999] h-[100dvh] w-screen bg-black text-white select-none overflow-hidden flex flex-col justify-between">
      {/* ------------------------------------------------------------------ */}
      {/* 1. TOÀN MÀN HÌNH CAMERA (FULL-BLEED IMMERSIVE) */}
      {/* ------------------------------------------------------------------ */}
      <div className="absolute inset-0 w-full h-full overflow-hidden z-0 pointer-events-none bg-black">
        <video
          ref={videoRef}
          playsInline
          webkit-playsinline="true"
          x5-playsinline="true"
          autoPlay
          muted
          controls={false}
          disablePictureInPicture
          className={`w-full h-full object-cover pointer-events-none ${
            cameraFacing === "user" ? "transform scale-x-[-1]" : ""
          }`}
        />
      </div>

      {/* Dark Vignette Overlay */}
      <div className="absolute inset-0 pointer-events-none z-10 bg-radial-gradient from-transparent via-black/10 to-black/70" />

      {/* ------------------------------------------------------------------ */}
      {/* 2. SLIDE-OUT HEADER (TRƯỢT RA / VÔ THÔNG MINH - CHỈ HIỆN KHI SCAN) */}
      {/* ------------------------------------------------------------------ */}
      {stage === "scanning" && (
        <div className="fixed top-0 inset-x-0 z-50 pointer-events-none transition-all duration-300">
          <div
            className={`pointer-events-auto bg-black/90 backdrop-blur-xl border-b border-white/20 px-4 pt-[calc(env(safe-area-inset-top,44px)+10px)] pb-3 shadow-2xl transition-transform duration-300 ${
              isHeaderOpen ? "translate-y-0" : "-translate-y-full"
            }`}
          >
            <div className="flex items-center justify-between gap-2 max-w-4xl mx-auto">
              {/* Close Button */}
              <Link
                href={`/patients/${patientId}`}
                className="rounded-full bg-white/10 hover:bg-white/20 h-10 w-10 border border-white/20 text-white flex items-center justify-center font-bold text-sm shadow transition shrink-0"
                title="Thoát quét"
              >
                ✕
              </Link>

              {/* Camera Toggle Button (Cam Sau / Cam Trước) */}
              <button
                type="button"
                onClick={async () => {
                  const nextFacing = cameraFacing === "user" ? "environment" : "user";
                  setCameraFacing(nextFacing);
                  await startCamera(nextFacing);
                }}
                className="rounded-full bg-emerald-500 hover:bg-emerald-400 active:scale-95 px-4 py-2 border border-emerald-300 text-xs font-black text-black flex items-center gap-1.5 shadow-lg transition"
              >
                <span className="text-sm">🔄</span>
                <span>{cameraFacing === "user" ? "Chuyển Camera Sau" : "Chuyển Camera Trước"}</span>
              </button>

              {/* Voice Guidance Toggle */}
              <button
                type="button"
                onClick={() => setVoiceEnabled((v) => !v)}
                className="rounded-full bg-white/10 hover:bg-white/20 px-3 py-2 border border-white/20 text-xs font-bold text-white flex items-center gap-1 shadow transition"
                title="Bật/Tắt giọng nói"
              >
                <span>{voiceEnabled ? "🔊" : "🔇"}</span>
              </button>

              {/* Head Pose Telemetry */}
              <div className="rounded-xl bg-white/10 border border-white/20 px-3 py-1.5 text-right shadow">
                <p className="text-[9px] font-black uppercase text-emerald-400">HEAD POSE</p>
                <p className="text-[11px] font-bold text-zinc-200 font-mono">
                  P: <span className="text-emerald-400">{telemetry.pitch}°</span> | Y: <span className="text-emerald-400">{telemetry.yaw}°</span>
                </p>
              </div>
            </div>
          </div>

          {/* Tab gập/mở Header */}
          <div className="flex justify-center pointer-events-auto">
            <button
              type="button"
              onClick={() => setIsHeaderOpen((v) => !v)}
              className="bg-black/85 hover:bg-black backdrop-blur-md border border-white/20 border-t-0 px-4 py-1 rounded-b-xl text-[10px] font-bold text-zinc-300 hover:text-white flex items-center gap-1 shadow-lg transition active:scale-95"
            >
              <span>{isHeaderOpen ? "▲ Thu gọn menu" : "▼ Mở điều khiển & Camera sau"}</span>
            </button>
          </div>
        </div>
      )}

      {/* ------------------------------------------------------------------ */}
      {/* 3. DẢI TIẾN TRÌNH CÁC GÓC ĐỘ (ĐẶT Ở TRÊN CAO, KHÔNG NẰM GIỮA MẶT) */}
      {/* ------------------------------------------------------------------ */}
      {stage === "scanning" && isArkitAvailable && (
        <div className="relative z-40 mx-4 mt-[calc(env(safe-area-inset-top,44px)+4.5rem)] rounded-2xl bg-gradient-to-r from-emerald-950/95 to-teal-950/95 border-2 border-emerald-400 p-3.5 text-center shadow-2xl backdrop-blur-xl pointer-events-auto">
          <div className="flex items-center justify-center gap-2 mb-1">
            <span className="text-xl animate-pulse">⚡📱</span>
            <h4 className="text-xs font-black uppercase text-emerald-300 tracking-wide">
              PHÁT HIỆN IPHONE TRUEDEPTH / ARKIT
            </h4>
          </div>
          <p className="text-[10px] text-zinc-300 mb-2">
            Quét 3D thực tế với cảm biến Apple ARKit & TrueDepth (1.220 điểm không gian + độ sâu TrueDepth).
          </p>
          <button
            type="button"
            onClick={startNativeArkitScan}
            disabled={isArkitScanning}
            className="w-full rounded-xl bg-emerald-400 hover:bg-emerald-300 active:scale-95 py-2.5 text-xs font-black text-black shadow-xl flex items-center justify-center gap-2 transition"
          >
            <span>🚀</span>
            <span>{isArkitScanning ? "ĐANG QUÉT TRUEDEPTH..." : "MỞ MÁY QUÉT TRUEDEPTH 3D APPLE"}</span>
          </button>
        </div>
      )}

      {stage === "scanning" && (
        <div className="relative z-30 flex justify-center px-4 mt-[calc(env(safe-area-inset-top,44px)+1rem)] pointer-events-none">
          <div className="pointer-events-auto w-full max-w-sm rounded-2xl bg-black/85 backdrop-blur-md border border-white/20 p-2.5 flex items-center justify-between shadow-2xl">
            {ORDERED_TARGET_KEYS.map((key) => {
              const cfg = TARGET_ANGLES[key];
              const done = targetsState[key]?.captured;
              const isCurrent = key === currentTargetKey;
              return (
                <div key={key} className="flex flex-col items-center flex-1">
                  <div
                    className={`h-4 w-4 rounded-full flex items-center justify-center text-[9px] font-black transition-all ${
                      done
                        ? "bg-emerald-400 text-black shadow-md shadow-emerald-400/50"
                        : isCurrent
                        ? "bg-white text-black ring-2 ring-emerald-400 scale-125 animate-pulse"
                        : "bg-zinc-700 text-transparent"
                    }`}
                  >
                    {done ? "✓" : ""}
                  </div>
                  <span className={`text-[8px] font-bold mt-1 ${isCurrent ? "text-white" : done ? "text-emerald-400" : "text-zinc-400"}`}>
                    {cfg.targetYaw}°
                  </span>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Thông báo trạng thái camera */}
      {isStartingCamera && (
        <div className="relative z-30 mx-4 mt-2 rounded-2xl bg-black/80 border border-white/20 p-2.5 text-center backdrop-blur-md shadow-xl">
          <p className="text-xs font-bold text-zinc-200">🎥 Đang mở camera...</p>
        </div>
      )}

      {noFaceRearWarning && (
        <div className="relative z-30 mx-4 mt-2 rounded-2xl bg-amber-950/90 border border-amber-500/50 p-3 text-center backdrop-blur-md shadow-2xl">
          <p className="text-xs font-bold text-amber-200">⚠️ Không nhận diện được khuôn mặt.</p>
          <p className="text-[10px] text-amber-300/80 mt-0.5">
            {cameraFacing === "environment" ? "Hướng camera sau vào khuôn mặt cần quét." : "Đưa khuôn mặt vào giữa khung hình."}
          </p>
        </div>
      )}

      {errorMessage && (
        <div className="relative z-30 mx-4 mt-2 rounded-2xl bg-rose-950/90 border border-rose-500/50 p-3 text-center backdrop-blur-md shadow-2xl">
          <p className="text-xs font-bold text-rose-200">{errorMessage}</p>
          <button
            type="button"
            onClick={() => startCamera(cameraFacing)}
            className="mt-2 rounded-xl bg-rose-500 px-4 py-1.5 text-xs font-black text-black shadow hover:bg-rose-400"
          >
            🔄 Thử Mở Lại Camera
          </button>
        </div>
      )}

      {/* ------------------------------------------------------------------ */}
      {/* 4. KHUNG OVAL ĐỊNH VỊ TRONG SUỐT */}
      {/* ------------------------------------------------------------------ */}
      {stage === "scanning" && (
        <main className="relative z-10 flex-1 flex flex-col items-center justify-center px-4">
          <div className="relative w-[78vw] max-w-[320px] aspect-[3/4] flex items-center justify-center pointer-events-none">
            {/* Viền Oval mỏng thanh lịch */}
            <div
              className={`absolute inset-0 rounded-full border-2 transition-all duration-200 flex items-center justify-center ${
                holdProgress > 0
                  ? "border-emerald-400 shadow-[0_0_25px_rgba(52,211,153,0.8)] scale-[1.02]"
                  : "border-white/50 shadow-2xl shadow-emerald-500/20"
              }`}
            >
              <div className={`h-1.5 w-1.5 rounded-full ${holdProgress > 0 ? "bg-emerald-300" : "bg-emerald-400/80"} animate-ping`} />
            </div>

            {/* Kim chỉ thị góc trên viền Oval */}
            <div
              className="absolute inset-0 pointer-events-none transition-transform duration-150"
              style={{ transform: `rotate(${Math.max(-66, Math.min(66, telemetry.yaw * 1.1))}deg)` }}
            >
              <div className="absolute left-1/2 top-1 h-3 w-1 -translate-x-1/2 rounded-full bg-emerald-400 shadow-[0_0_8px_rgba(52,211,153,0.9)]" />
            </div>

            {/* Thông báo Giữ Yên Lấy Nét */}
            {holdProgress > 0 && (
              <div className="absolute -bottom-10 px-4 py-1.5 rounded-full bg-emerald-500 text-black text-xs font-black shadow-xl flex items-center gap-1.5 animate-pulse">
                <span className="inline-block h-2 w-2 rounded-full bg-black animate-ping" />
                <span>ĐANG GIỮ YÊN LẤY NÉT ({holdProgress}%)</span>
              </div>
            )}

            {isDone && <div className="absolute inset-0 rounded-full border-4 border-emerald-400 animate-ping pointer-events-none" />}
          </div>
        </main>
      )}

      {/* ------------------------------------------------------------------ */}
      {/* 5. MÀN HÌNH KIỂM TRA ẢNH GỐC ĐÃ QUÉT (MANDATORY PHOTO REVIEW) */}
      {/* ------------------------------------------------------------------ */}
      {stage === "review" && (
        <main className="relative z-30 flex-1 flex flex-col items-center justify-start overflow-y-auto px-4 pt-16 pb-28 pointer-events-auto">
          <div className="w-full max-w-xl bg-black/90 backdrop-blur-2xl border border-white/20 rounded-3xl p-5 shadow-2xl">
            <div className="text-center mb-4">
              <span className="inline-block px-3 py-1 rounded-full bg-emerald-500/20 text-emerald-400 font-bold text-xs mb-1">
                ✓ ĐÃ THU THẬP ĐỦ CÁC GÓC
              </span>
              <h2 className="text-lg font-black text-white">Kiểm Tra Ảnh Gốc Đã Quét</h2>
              <p className="text-xs text-zinc-400 mt-0.5">
                Chạm vào ảnh để phóng to kiểm tra nét mặt &amp; góc quay trước khi tạo 3D.
              </p>
            </div>

            {/* Grid ảnh gốc */}
            <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
              {reviewPhotos.map((photo) => (
                <div
                  key={photo.key}
                  onClick={() => setZoomedPhoto(photo)}
                  className="group relative aspect-[3/4] rounded-2xl overflow-hidden border border-white/20 bg-zinc-900 cursor-pointer shadow hover:border-emerald-400 transition"
                >
                  <img src={photo.url} alt={photo.name} className="w-full h-full object-cover group-hover:scale-105 transition duration-300" />
                  <div className="absolute inset-x-0 bottom-0 bg-gradient-to-t from-black/90 via-black/50 to-transparent p-2">
                    <p className="text-[10px] font-black text-white">{photo.name}</p>
                    <p className="text-[9px] text-emerald-400 font-mono">Yaw: {Math.round(photo.yaw)}°</p>
                  </div>
                  <div className="absolute top-1.5 right-1.5 h-5 w-5 rounded-full bg-black/70 backdrop-blur-md flex items-center justify-center text-[10px] text-white">
                    🔍
                  </div>
                </div>
              ))}
            </div>

            {/* Nút hành động */}
            <div className="mt-5 flex flex-col gap-2.5">
              <button
                type="button"
                onClick={finalizeScan}
                className="w-full rounded-2xl bg-emerald-500 hover:bg-emerald-400 py-3.5 text-sm font-black text-black shadow-xl transition flex items-center justify-center gap-2 active:scale-98"
              >
                <span>🚀</span>
                <span>XÁC NHẬN ẢNH CHUẨN &amp; KHỞI TẠO 3D</span>
              </button>
              <button
                type="button"
                onClick={() => {
                  setStage("scanning");
                  setIsScanStarted(false);
                  const reset = buildInitialTargetsState();
                  targetsStateRef.current = reset;
                  setTargetsState(reset);
                  setCurrentTargetKey("FRONT");
                  frameBufferBlobsRef.current = [];
                  originFramesRef.current = {};
                  isFinalizingRef.current = false;
                  retryCountRef.current = 0;
                  stableTargetTicksRef.current = 0;
                  setCanManuallySkip(false);
                  startCamera(cameraFacing);
                }}
                className="w-full rounded-2xl bg-white/10 hover:bg-white/20 py-2.5 text-xs font-bold text-zinc-300 transition"
              >
                🔄 Quét lại từ đầu
              </button>
            </div>
          </div>
        </main>
      )}

      {/* Zoom Modal xem ảnh to */}
      {zoomedPhoto && (
        <div className="fixed inset-0 z-[100000] bg-black/95 backdrop-blur-2xl flex flex-col items-center justify-between p-4 pointer-events-auto">
          <div className="w-full flex items-center justify-between pt-[max(env(safe-area-inset-top),12px)]">
            <span className="font-black text-sm text-emerald-400">{zoomedPhoto.name} ({Math.round(zoomedPhoto.yaw)}°)</span>
            <button
              type="button"
              onClick={() => setZoomedPhoto(null)}
              className="rounded-full bg-white/20 hover:bg-white/30 h-10 w-10 text-white font-bold flex items-center justify-center"
            >
              ✕
            </button>
          </div>
          <div className="relative max-h-[75vh] max-w-full aspect-[3/4] rounded-3xl overflow-hidden border border-white/20 shadow-2xl my-auto">
            <img src={zoomedPhoto.url} alt={zoomedPhoto.name} className="w-full h-full object-contain" />
          </div>
          <p className="text-xs text-zinc-400 pb-4">Bấm ✕ để quay lại danh sách ảnh</p>
        </div>
      )}

      {/* ------------------------------------------------------------------ */}
      {/* 6. FOOTER (GỢI Ý GIỌNG NÓI & NÚT HOÀN TẤT - CHỈ HIỆN KHI SCAN) */}
      {/* ------------------------------------------------------------------ */}
      {stage === "scanning" && (
        <footer className="relative z-20 w-full pb-[max(env(safe-area-inset-bottom),16px)] px-4 flex flex-col items-center pointer-events-auto">
          {/* Banner Giọng nói & Hướng dẫn */}
          {isScanStarted && (
            <div className="w-full max-w-sm text-center mb-2">
              <div className="py-2.5 px-4 rounded-2xl bg-black/85 backdrop-blur-xl border border-white/20 shadow-2xl">
                <p className="text-xs font-black uppercase text-amber-300">
                  {voiceInstruction}
                </p>
                <div className="mt-1 flex items-center justify-between text-[10px] text-zinc-400">
                  <span>Tiến độ: <strong className="text-emerald-400">{capturedCount}/{ORDERED_TARGET_KEYS.length} góc</strong></span>
                  <span>Buffer frame: <strong className="text-emerald-400">{bufferCount}/{RECONSTRUCTION_TARGET_COUNT}+</strong></span>
                </div>
                {hapticLog && (
                  <p className="text-[9px] font-mono text-emerald-400 mt-1 truncate">
                    {hapticLog}
                  </p>
                )}
              </div>
            </div>
          )}

          {/* Nút hành động nhanh trong lúc quét */}
          {isScanStarted && (
            <div className="w-full max-w-sm flex items-center gap-2 mb-2">
              <button
                type="button"
                onClick={manualSnapCurrentTarget}
                className="flex-1 rounded-2xl bg-white/20 hover:bg-white/30 border border-white/30 py-2.5 text-xs font-black text-white shadow-lg active:scale-95 transition flex items-center justify-center gap-1.5"
              >
                <span>📸</span>
                <span>CHỤP GÓC NÀY NGAY</span>
              </button>
              <button
                type="button"
                onClick={manualSkipCurrentTarget}
                className="rounded-2xl bg-amber-500/80 hover:bg-amber-400 border border-amber-400/40 px-3.5 py-2.5 text-xs font-black text-black shadow-lg active:scale-95 transition"
                title="Bỏ qua góc này"
              >
                ⏭ Bỏ qua
              </button>
            </div>
          )}

          {/* Nút Hoàn tất quét tức thì */}
          {isScanStarted && (
            <div className="w-full max-w-sm">
              <button
                type="button"
                onClick={openReviewGallery}
                className="w-full rounded-2xl bg-emerald-500 hover:bg-emerald-400 py-3 text-xs font-black text-black shadow-xl active:scale-98 transition flex items-center justify-center gap-2"
              >
                <span>📸</span>
                <span>XEM LẠI ẢNH GỐC &amp; KHỞI TẠO 3D ({capturedCount}/{ORDERED_TARGET_KEYS.length})</span>
              </button>
            </div>
          )}
        </footer>
      )}

      {/* ------------------------------------------------------------------ */}
      {/* 7. MÀN HÌNH ĐANG LƯU & DỰNG 3D (UPLOADING OVERLAY) */}
      {/* ------------------------------------------------------------------ */}
      {stage === "uploading" && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/90 backdrop-blur-2xl pointer-events-auto">
          <div className="w-full max-w-sm rounded-3xl bg-zinc-900 border border-white/20 p-6 text-center shadow-2xl">
            <div className="text-4xl animate-spin mb-3">⚡</div>
            <h3 className="text-lg font-black text-white">Đang Lưu Ảnh &amp; Dựng Mô Hình 3D</h3>
            <p className="text-xs text-zinc-400 mt-1">Đang lưu toàn bộ ảnh gốc vào hồ sơ bệnh nhân...</p>
          </div>
        </div>
      )}

      {/* ------------------------------------------------------------------ */}
      {/* 8. MÀN HÌNH HOÀN TẤT & VÀO 3D STUDIO (RESULT OVERLAY) */}
      {/* ------------------------------------------------------------------ */}
      {stage === "result" && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/90 backdrop-blur-2xl pointer-events-auto">
          <div className="w-full max-w-sm rounded-3xl bg-zinc-900 border border-emerald-500/50 p-6 text-center shadow-2xl">
            <div className="text-4xl mb-2">🎉</div>
            <h3 className="text-lg font-black text-white">Đã Lưu Hồ Sơ &amp; Tạo 3D Thành Công</h3>
            <p className="text-xs text-zinc-400 mt-1">Ảnh gốc đã được đồng bộ vào hồ sơ bệnh án.</p>
            <div className="mt-6 flex flex-col gap-3">
              <button
                type="button"
                onClick={() => {
                  stopCamera();
                  window.location.href = `/patients/${patientId}/studio`;
                }}
                className="w-full rounded-2xl bg-emerald-500 hover:bg-emerald-400 active:scale-95 py-4 text-sm font-black text-black shadow-xl transition flex items-center justify-center gap-2"
              >
                <span>🧊</span>
                <span>VÀO 3D STUDIO TƯ VẤN VIP →</span>
              </button>
              <button
                type="button"
                onClick={() => {
                  stopCamera();
                  window.location.href = `/patients/${patientId}`;
                }}
                className="w-full rounded-2xl bg-white/10 hover:bg-white/20 active:scale-95 py-3 text-xs font-bold text-white transition text-center"
              >
                📁 Về chi tiết hồ sơ bệnh nhân
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
