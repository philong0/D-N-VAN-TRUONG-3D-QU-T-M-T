import type { ScannerKind } from "@/lib/types";

/**
 * Official Apple ARKit Face Geometry Contract Specification.
 *
 * In native iOS (Swift/ARKit), `ARSCNFaceGeometry` / `ARFaceGeometry` provides:
 * - `vertices`: `[simd_float3]` (1220 vertex coordinates in meters relative to face anchor).
 * - `triangleIndices`: `[Int16]` (2304 triangles defining facial mesh topology).
 * - `textureCoordinates`: `[simd_float2]` (UV map normalized 0.0 - 1.0).
 * - `transform`: `simd_float4x4` (4x4 matrix camera pose & face orientation).
 * - `blendShapes`: Dictionary of 52 facial blend shape coefficients.
 */
export interface ARKitFaceAnchorTransform {
  matrix: number[]; // 16 elements (4x4 matrix)
  translation: { x: number; y: number; z: number };
  rotation: { roll: number; pitch: number; yaw: number };
}

export interface ARKitFaceGeometryPayload {
  vertexCount: number; // 1220 vertices in ARKit
  triangleCount: number; // 2304 triangles in ARKit
  vertices?: Float32Array | number[]; // [x, y, z, x, y, z...]
  triangleIndices?: Int16Array | number[];
  textureCoordinates?: Float32Array | number[]; // [u, v, u, v...]
  faceAnchorTransform: ARKitFaceAnchorTransform;
  timestamp: number;
}

export interface NativeFaceCapturePayload {
  view: string;
  rgbImageDataUriOrBlob: Blob | string;
  depthMapDataUriOrBlob?: Blob | string;
  geometry?: ARKitFaceGeometryPayload;
  deviceModel: string;
  systemVersion: string;
  hasTrueDepth: boolean;
}

export interface FaceScannerCapabilities {
  rgbFrames: boolean;
  poseMetadata: boolean;
  depthData: boolean;
  arkitMeshGeometry: boolean;
}

export interface FaceScanner {
  readonly kind: ScannerKind;
  readonly capabilities: FaceScannerCapabilities;
  start(video: HTMLVideoElement): Promise<void>;
  capture(video: HTMLVideoElement): Promise<Blob>;
  stop(): void;
}

/**
 * Browser fallback for development and guided photographic capture only.
 * It does not report depth, ARKit mesh or hardware-verified face pose.
 */
export class WebCameraScanner implements FaceScanner {
  readonly kind = "web_camera" as const;
  readonly capabilities = {
    rgbFrames: true,
    poseMetadata: false,
    depthData: false,
    arkitMeshGeometry: false,
  };
  private stream: MediaStream | null = null;

  async start(video: HTMLVideoElement): Promise<void> {
    this.stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: { ideal: "user" }, width: { ideal: 1280 }, height: { ideal: 720 } },
      audio: false,
    });
    video.srcObject = this.stream;
    await video.play();
  }

  capture(video: HTMLVideoElement): Promise<Blob> {
    return new Promise((resolve, reject) => {
      const canvas = document.createElement("canvas");
      canvas.width = video.videoWidth || 1280;
      canvas.height = video.videoHeight || 720;
      const context = canvas.getContext("2d");
      if (!context) return reject(new Error("Không khởi tạo được canvas camera."));
      context.drawImage(video, 0, 0, canvas.width, canvas.height);
      canvas.toBlob((blob) => (blob ? resolve(blob) : reject(new Error("Không lấy được frame camera."))), "image/jpeg", 0.92);
    });
  }

  stop() {
    this.stream?.getTracks().forEach((track) => track.stop());
    this.stream = null;
  }
}

/**
 * Integration boundary for native iOS Swift ARKit app wrapper (see
 * `ios-app/FaceScannerView.swift`). That app hosts THIS web app's own
 * `/patients/[id]/scan` page inside a `WKWebView` and registers itself as
 * `window.webkit.messageHandlers.arkitScanBridge` — a native message
 * handler this class posts `{requestId, action, ...}` messages to, getting
 * the result back via two global callbacks the native side invokes with
 * `evaluateJavaScript` (`window.__arkitBridgeResolve`/`__arkitBridgeReject`,
 * registered once by `installArkitBridgeCallbacks()` below). Every real
 * capture — vertices, triangle indices, texture coordinates, the face's
 * pose relative to the camera, real camera intrinsics — comes from the
 * TrueDepth sensor via ARKit; nothing here fabricates or infers geometry.
 */
interface ArkitBridgeWindow extends Window {
  webkit?: { messageHandlers?: { arkitScanBridge?: { postMessage: (body: unknown) => void } } };
  __arkitBridgeResolve?: (requestId: string, payload: unknown) => void;
  __arkitBridgeReject?: (requestId: string, message: string) => void;
  __arkitBridgePending?: Map<string, { resolve: (v: unknown) => void; reject: (e: Error) => void }>;
}

function arkitWindow(): ArkitBridgeWindow {
  return window as ArkitBridgeWindow;
}

/** Idempotent — safe to call from every `IOSNativeScanner` instance/method,
 * only actually registers the two global callbacks once. Must run before
 * the FIRST `postMessage` to the native side, since the native code calls
 * these by name without waiting for any "ready" handshake. */
function installArkitBridgeCallbacks(): void {
  const w = arkitWindow();
  if (w.__arkitBridgePending) return;
  w.__arkitBridgePending = new Map();
  w.__arkitBridgeResolve = (requestId, payload) => {
    const pending = w.__arkitBridgePending?.get(requestId);
    if (!pending) return;
    w.__arkitBridgePending?.delete(requestId);
    pending.resolve(payload);
  };
  w.__arkitBridgeReject = (requestId, message) => {
    const pending = w.__arkitBridgePending?.get(requestId);
    if (!pending) return;
    w.__arkitBridgePending?.delete(requestId);
    pending.reject(new Error(message));
  };
}

function callArkitBridge<T>(body: Record<string, unknown>, timeoutMs = 20000): Promise<T> {
  installArkitBridgeCallbacks();
  const w = arkitWindow();
  const bridge = w.webkit?.messageHandlers?.arkitScanBridge;
  if (!bridge) return Promise.reject(new Error("arkitScanBridge không sẵn sàng — không chạy trong app iOS Native."));

  const requestId = `${Date.now()}-${Math.random().toString(36).slice(2)}`;
  return new Promise<T>((resolve, reject) => {
    const timer = window.setTimeout(() => {
      w.__arkitBridgePending?.delete(requestId);
      reject(new Error("Hết thời gian chờ phản hồi từ ARKit bridge."));
    }, timeoutMs);
    w.__arkitBridgePending?.set(requestId, {
      resolve: (v) => {
        window.clearTimeout(timer);
        resolve(v as T);
      },
      reject: (e) => {
        window.clearTimeout(timer);
        reject(e);
      },
    });
    bridge.postMessage({ requestId, ...body });
  });
}

export class IOSNativeScanner implements FaceScanner {
  readonly kind = "ios_native" as const;
  readonly capabilities = {
    rgbFrames: true,
    poseMetadata: true,
    depthData: true,
    arkitMeshGeometry: true,
  };

  /** Set by `startForPatient` before `start()`/`capture()` are usable — the
   * native side needs to know WHERE to upload (`POST .../scan-sessions/
   * {sessionId}/frames`), and only the web page (which created the session
   * via the existing `/api/patients/{id}/scan-sessions` call) knows that id. */
  private patientId: string | null = null;
  private sessionId: string | null = null;

  static isAvailable() {
    return typeof window !== "undefined" && Boolean(arkitWindow().webkit?.messageHandlers?.arkitScanBridge);
  }

  /** `GuidedFaceScan.tsx` calls this (instead of the bare `start()` the
   * `FaceScanner` interface declares) once it has a real `sessionId` — the
   * plain `start(video)` below exists only to satisfy the shared interface
   * and will throw if this hasn't been called first. */
  configureSession(patientId: string, sessionId: string): void {
    this.patientId = patientId;
    this.sessionId = sessionId;
  }

  async start(_video: HTMLVideoElement): Promise<void> {
    if (!this.patientId || !this.sessionId) {
      throw new Error("IOSNativeScanner.configureSession(patientId, sessionId) phải được gọi trước start().");
    }
    await callArkitBridge<{ started: boolean }>({
      action: "start",
      patientId: this.patientId,
      sessionId: this.sessionId,
    });
  }

  /** Unlike `WebCameraScanner.capture`, this does not read from `video` —
   * the real frame comes from ARKit's own session, already running against
   * the same physical camera. `view` (which of the 5 guided angles this
   * capture is for) is threaded through via `captureForView` since the
   * shared `FaceScanner` interface's `capture(video)` has no view parameter;
   * `GuidedFaceScan.tsx` calls `captureForView` directly for this scanner. */
  async capture(_video: HTMLVideoElement): Promise<Blob> {
    throw new Error("IOSNativeScanner: dùng captureForView(view), không phải capture(video).");
  }

  async captureForView(view: string): Promise<{ fileUrl: string; vertexCount: number }> {
    return callArkitBridge<{ fileUrl: string; vertexCount: number }>({ action: "capture", view });
  }

  stop() {
    if (!this.patientId || !this.sessionId) return;
    void callArkitBridge({ action: "stop" }).catch(() => {
      // Best-effort — the native ARSession is torn down when the WebView
      // navigates away regardless, this just frees it a little sooner.
    });
  }
}

/** Placeholder protocol adapter for a future dedicated 3D scanner hardware SDK. */
export class FutureScanner implements FaceScanner {
  readonly kind = "future" as const;
  readonly capabilities = {
    rgbFrames: false,
    poseMetadata: false,
    depthData: false,
    arkitMeshGeometry: false,
  };

  async start(_video: HTMLVideoElement): Promise<void> {
    throw new Error("Chưa có scanner hardware SDK được cấu hình.");
  }

  async capture(_video: HTMLVideoElement): Promise<Blob> {
    throw new Error("Chưa có scanner hardware SDK được cấu hình.");
  }

  stop() {}
}

