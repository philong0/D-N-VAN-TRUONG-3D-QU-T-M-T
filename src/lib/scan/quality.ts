import type { ScanCaptureView, ScanFrame, ScanQualityMetric, ScanQualityReport, ScannerKind } from "@/lib/types";
import { FINAL_REVIEW } from "./scan-constants";

export const REQUIRED_SCAN_VIEWS: ScanCaptureView[] = ["front", "left_45", "left_profile", "right_45", "right_profile"];

export const VIEW_LABELS: Record<ScanCaptureView, string> = {
  front: "Chính diện (0°)",
  left_45: "Nghiêng trái (45°)",
  left_profile: "Góc nghiêng trái (90°)",
  right_45: "Nghiêng phải (45°)",
  right_profile: "Góc nghiêng phải (90°)",
  burst: "Quét liên tục nhiều góc",
};

// D-mindense — must match dense_correspondence.py's own MIN_FRAMES_FOR_DENSE
// so the web UI's own quality gate agrees with the backend gate on what
// "enough frames for a real dense reconstruction" means, instead of two
// independently-guessed thresholds silently drifting apart.
export const MIN_BURST_FRAMES = 15;

const pass = (detail: string): ScanQualityMetric => ({ status: "pass", detail });
const warning = (detail: string): ScanQualityMetric => ({ status: "warning", detail });
const fail = (detail: string): ScanQualityMetric => ({ status: "fail", detail });
const unavailable = (detail: string): ScanQualityMetric => ({ status: "not_available", detail });

/**
 * Evaluates scan quality honestly based on uploaded evidence.
 * Does not fake depth or pose verification for web camera capture.
 */
export function evaluateBurstScanQuality(frames: ScanFrame[]): ScanQualityReport {
  const burstFrames = frames.filter((f) => f.view === "burst");
  const corruptFrames = burstFrames.filter((f) => f.byteSize < 8 * 1024);
  // 2026-09-03 scanner-quality audit fix: this used to read
  // `metadata.faceDetected`/`metadata.turnProxy`, fields the real scanner
  // (GuidedFaceScan.tsx) never sent (it sent `onTarget`/`measuredYaw`
  // instead) — `tracked`/`turnSamples` were therefore ALWAYS empty and
  // `hasMeasuredSweep` ALWAYS false, so every real burst session scored
  // "fail" regardless of actual quality, silently blocking auto
  // reconstruction. The scanner now sends real, richer per-frame metadata
  // (`faceDetected`, `yaw` in degrees, `brightness`, `sharpness`, etc. — see
  // frame-evaluator.ts's FrameEvaluation / GuidedFaceScan.tsx's upload
  // metadata) matching these field names.
  const tracked = burstFrames
    .map((frame) => frame.qualityMetadata)
    .filter((metadata): metadata is Record<string, unknown> => Boolean(metadata && metadata.faceDetected === true));
  const yawSamples = tracked
    .map((metadata) => metadata.yaw)
    .filter((value): value is number => typeof value === "number" && Number.isFinite(value));
  const hasMeasuredSweep = yawSamples.length >= Math.min(10, burstFrames.length)
    && Math.min(...yawSamples) <= FINAL_REVIEW.MIN_LEFT_YAW_DEG
    && Math.max(...yawSamples) >= FINAL_REVIEW.MIN_RIGHT_YAW_DEG;
  const frameQualityMetric = corruptFrames.length > 0
    ? warning(`${corruptFrames.length}/${burstFrames.length} frame có dung lượng nhỏ bất thường, có thể bị mờ/tối.`)
    : pass(`Tất cả ${burstFrames.length} frame đều hợp lệ về định dạng và dung lượng.`);

  const enough = burstFrames.length >= MIN_BURST_FRAMES;
  const coverage = !enough
    ? fail(`Mới thu ${burstFrames.length}/${MIN_BURST_FRAMES} frame tối thiểu — cần quét lại, xoay đầu chậm hơn hoặc lâu hơn.`)
    : !hasMeasuredSweep
    ? fail("Chưa đo được sweep trái-phải đủ rộng từ FaceMesh tracking; cần quay chậm hơn qua cả hai bên.")
    : pass(`Đã thu ${burstFrames.length} frame liên tục với sweep trái-phải được đo từ ${yawSamples.length} frame tracking.`);

  // D-noregionclaim — a burst session's regional coverage (mũi/cằm/profile
  // trái-phải) genuinely depends on how much the head actually turned
  // during capture, which this server-side check has no way to verify
  // from frame count alone (unlike the 5 named-view flow, where "captured
  // left_profile" IS the region check) — reporting a specific per-region
  // verdict here would be a fabricated claim, not a real measurement.
  const regions: ScanQualityReport["regions"] = {
    burst: coverage,
  };

  return {
    overall: enough && hasMeasuredSweep ? "warning" : "fail",
    coverage,
    trackingQuality: unavailable("Web camera burst: không có cảm biến TrueDepth/LiDAR."),
    frameQuality: frameQualityMetric,
    lightingQuality: warning("Nhân viên xác nhận ánh sáng đều trong suốt quá trình quay."),
    faceVisibility: warning("Nhân viên xác nhận khách hàng không đeo kính, tóc không che trán/mũi/cằm."),
    poseCoverage: hasMeasuredSweep
      ? warning(`Đã đo yaw thực tế từ ${yawSamples.length} frame; pose camera metric sẽ được xác minh trong reconstruction.`)
      : fail("Chưa có bằng chứng tracking cho coverage trái/phải thực tế."),
    geometryConsistency: unavailable("Độ nhất quán hình học được đo thật trong lúc dựng mesh (dense_correspondence.py), chưa có kết quả tại bước tải lên."),
    regions,
    evaluatedAt: new Date().toISOString(),
  };
}

export function evaluateScanQuality(frames: ScanFrame[], scannerKind: ScannerKind): ScanQualityReport {
  const captured = new Set(frames.map((frame) => frame.view));
  const missing = REQUIRED_SCAN_VIEWS.filter((view) => !captured.has(view));
  const complete = missing.length === 0;
  const nativeDepth = scannerKind === "ios_native" && REQUIRED_SCAN_VIEWS.every((view) => {
    const frame = frames.find((item) => item.view === view);
    return Boolean(frame?.depthAvailable && frame.geometryFileName && frame.intrinsicsFileName);
  });

  const regions: ScanQualityReport["regions"] = {};

  // View-specific checks
  for (const view of REQUIRED_SCAN_VIEWS) {
    const frame = frames.find((f) => f.view === view);
    if (!frame) {
      regions[view] = fail(`Thiếu ảnh góc ${VIEW_LABELS[view]}.`);
    } else if (frame.byteSize < 15 * 1024) {
      regions[view] = warning(`${VIEW_LABELS[view]}: Kích thước file nhỏ (${Math.round(frame.byteSize / 1024)}KB), kiểm tra độ mờ.`);
    } else {
      regions[view] = pass(`${VIEW_LABELS[view]}: Đã có ảnh đầy đủ.`);
    }
  }

  // Anatomical region checks
  regions.nose = captured.has("front") && (captured.has("left_45") || captured.has("right_45"))
    ? (nativeDepth ? pass("Vùng sống & đầu mũi có đủ dữ liệu đa góc + depth.") : warning("Vùng mũi có ảnh 2D đa góc; chưa có depth phần cứng."))
    : fail("Thiếu góc nghiêng để tái tạo giải phẫu sống & đầu mũi.");

  regions.chin = captured.has("front") && (captured.has("left_profile") || captured.has("right_profile"))
    ? (nativeDepth ? pass("Vùng cằm & viền hàm có đủ góc profile + depth.") : warning("Vùng cằm có ảnh profile; chưa có depth phần cứng."))
    : fail("Thiếu góc profile 90° để đo đạc độ nhô cằm Pog.");

  regions.leftProfile = captured.has("left_profile")
    ? pass("Góc nghiêng trái 90° hoàn tất.")
    : fail("Thiếu góc nghiêng trái 90°.");

  regions.rightProfile = captured.has("right_profile")
    ? pass("Góc nghiêng phải 90° hoàn tất.")
    : fail("Thiếu góc nghiêng phải 90°.");

  // Check if any individual frame is suspiciously small (likely corrupt/blank)
  const corruptFrames = frames.filter((f) => f.byteSize < 8 * 1024);
  const frameQualityMetric = corruptFrames.length > 0
    ? fail(`${corruptFrames.length} frame có dung lượng quá nhỏ, có thể bị lỗi khi chụp.`)
    : pass(`Tất cả ${frames.length} frame đều hợp lệ về định dạng và dung lượng.`);

  if (!complete) {
    return {
      overall: "fail",
      coverage: fail(`Thiếu ${missing.length}/5 góc bắt buộc (${missing.map((v) => VIEW_LABELS[v]).join(", ")}).`),
      trackingQuality: unavailable("Chưa có tracker native để xác thực pose."),
      frameQuality: frameQualityMetric,
      lightingQuality: unavailable("Web camera: Nhân viên kiểm tra độ sáng bằng mắt."),
      faceVisibility: unavailable("Web camera: Chưa có mô hình nhận diện che khuất."),
      poseCoverage: fail(`Mới thu thập được ${captured.size}/5 góc. Cần quét lại đủ 5 góc.`),
      geometryConsistency: unavailable("Chưa đủ 5 góc để kiểm tra tính nhất quán hình học."),
      regions,
      evaluatedAt: new Date().toISOString(),
    };
  }

  return {
    overall: nativeDepth ? "pass" : "warning",
    coverage: pass("Đã thu thập đủ 5/5 góc hướng dẫn chuẩn y khoa."),
    trackingQuality: nativeDepth
      ? pass("Cảm biến TrueDepth / ARKit đã ghi nhận pose tracking.")
      : unavailable("Web camera: Không có cảm biến TrueDepth/LiDAR. Pose theo hướng dẫn thủ công."),
    frameQuality: frameQualityMetric,
    lightingQuality: warning("Nhân viên xác nhận ánh sáng đều, không bị lóa hoặc bóng tối đậm."),
    faceVisibility: warning("Nhân viên xác nhận khách hàng không đeo kính, tóc không che trán/mũi/cằm."),
    poseCoverage: nativeDepth
      ? pass("Đã thu đủ 5 góc với metadata tracking phần cứng.")
      : warning("Đã thu đủ 5 góc theo hướng dẫn; cần kiểm tra trực quan trước khi reconstruction."),
    geometryConsistency: nativeDepth
      ? pass("Độ nhất quán hình học sẵn sàng gửi Reconstruction Service.")
      : warning("Web camera: Chưa có depth map phần cứng để xác thực độ sâu đa góc."),
    regions,
    evaluatedAt: new Date().toISOString(),
  };
}
