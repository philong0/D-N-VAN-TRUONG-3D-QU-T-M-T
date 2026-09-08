export type ServiceKey =
  | "nang-nguc"
  | "sua-mui-cau-truc"
  | "cat-mi"
  | "don-cam-vline"
  | "cang-da-mat"
  | "hut-mo";

/**
 * 4 upload slots. What each slot actually depicts is dynamic — it depends on
 * the patient's registered service (Mũi / Mắt / Ngực each define their own 4
 * clinical angles; everything else falls back to generic "Góc 1..4"). See
 * lib/photo-angles.ts for the label/hint resolution.
 */
export type PhotoAngle = "angle1" | "angle2" | "angle3" | "angle4";

export interface PhotoAsset {
  fileName: string;
  angle: PhotoAngle;
  uploadedAt: string;
  width: number;
  height: number;
}

export interface Model3DAsset {
  generatedAt: string;
  sourcePhotos: PhotoAngle[];
  /** ai-engine's RECONSTRUCTION_METHOD string for whichever pipeline actually
   * produced this asset (Phase 3) — optional/absent for a Model3DAsset
   * written before Phase 3 existed. Not used for staleness checks itself
   * (that's `EXPECTED_RECONSTRUCTION_METHOD` vs. the real on-disk
   * `reconstruction/metadata.json`, see lib/gnm/reconstruction-service.ts) —
   * kept here only so the patient dossier/report can show which pipeline
   * version a given "before" model actually came from. */
  method?: string;
  warnings?: string[];
  coverageFraction?: number;
}

/** A capture-only record. It is deliberately separate from `model3d`: a scan
 * session is not a baseline model until an actual reconstruction provider has
 * produced and validated one. */
export type ScanSessionStatus =
  | "created"
  | "capturing"
  | "uploading"
  | "processing"
  | "reconstructing"
  | "quality_check"
  | "ready"
  | "failed"
  | "needs_rescan";

export type ScanQualityStatus = "pass" | "warning" | "fail" | "not_available";
export type ScanCaptureView = "front" | "left_45" | "left_profile" | "right_45" | "right_profile" | "burst";
export type ScannerKind = "web_camera" | "ios_native" | "future";

export interface ScanFrame {
  id: string;
  view: ScanCaptureView;
  /** Only present (and only meaningful) when view === "burst" — a burst
   * session holds many frames under the same view label, so this is what
   * distinguishes and orders them (real capture order), unlike the 5 named
   * views which are each unique by construction. */
  sequenceIndex?: number;
  fileName: string;
  capturedAt: string;
  byteSize: number;
  mimeType: string;
  /** Native capture may populate this later. Web capture never fabricates it. */
  poseMetadata?: Record<string, unknown>;
  cameraMetadata?: Record<string, unknown>;
  /** Measured browser-side frame quality and FaceMesh tracking summary. */
  qualityMetadata?: Record<string, unknown>;
  /** File references are present only for native ARKit captures. Keeping the
   * binary geometry out of patients.json prevents duplicate sensitive data. */
  geometryFileName?: string;
  intrinsicsFileName?: string;
  depthAvailable: boolean;
}

export interface ScanQualityMetric {
  status: ScanQualityStatus;
  detail: string;
}

export interface ScanQualityReport {
  overall: "pass" | "warning" | "fail";
  coverage: ScanQualityMetric;
  trackingQuality: ScanQualityMetric;
  frameQuality: ScanQualityMetric;
  lightingQuality: ScanQualityMetric;
  faceVisibility: ScanQualityMetric;
  poseCoverage: ScanQualityMetric;
  geometryConsistency?: ScanQualityMetric;
  regions: Partial<Record<ScanCaptureView | "nose" | "chin" | "leftProfile" | "rightProfile", ScanQualityMetric>>;
  evaluatedAt: string;
}

/**
 * Exactly 4 doctor-facing preview images picked from a completed guided
 * scan's real accepted frames (see src/lib/scan/profile-image-selection.ts)
 * — for quick visual reference only. Deliberately separate from
 * `ScanSession.frames`/reconstruction input: reconstruction still keeps
 * every accepted frame; this is just the 4 best representative shots.
 */
export type ProfileRole = "front" | "left" | "right" | "three_quarter";

export interface ProfilePreviewImage {
  fileName: string;
  yaw: number;
  pitch: number;
  roll: number;
  qualityScore: number;
  timestamp: number;
  sourceFrameId: string;
  savedAt: string;
}

export interface ScanSession {
  id: string;
  patientId: string;
  scannerKind: ScannerKind;
  status: ScanSessionStatus;
  createdAt: string;
  updatedAt: string;
  frames: ScanFrame[];
  quality?: ScanQualityReport;
  /**
   * The scanner's own real, measured end-of-scan report (frame counts,
   * yaw/pitch/roll ranges, reject-reason tallies, coverage, duplicate
   * audit — see src/lib/scan/scan-report.ts's `ScanReport`), sent verbatim
   * from the browser at "finalize". Stored for debugging/audit (spec Part
   * 12); never used to fabricate or override the server's own
   * `quality` evaluation above.
   */
  clientScanReport?: Record<string, unknown>;
  /** Reserved for an actual provider output, not a browser-generated mesh. */
  reconstruction?: {
    provider: string;
    requestedAt: string;
    baselineModelFileName?: string;
    baselineObjFileName?: string;
    error?: string;
  };
  error?: string;
}

/** Doctor-entered measurements used by the AI Clinical Advisor's safety checks. Not auto-derived from photos. */
export interface ClinicalBaseline {
  chestBaseWidthMm?: number;
  softTissueThicknessCm?: number;
  updatedAt?: string;
}

export interface NoseMorph {
  /** Δ chiều cao sống mũi so với hiện trạng (mm) */
  heightMm: number;
  /** Δ độ nhô đầu mũi / Tip Projection (mm) */
  tipProjectionMm: number;
  /** Góc Mũi–Môi tuyệt đối, chuẩn thẩm mỹ 95°–100° */
  nasolabialAngleDeg: number;
  /** Góc Trán–Mũi tuyệt đối, chuẩn thẩm mỹ 115°–130° */
  nasofrontalAngleDeg: number;
  /** Δ trụ mũi (mm) */
  columellaMm: number;
}

export interface EyeMorph {
  /** Δ chiều cao nếp mí (mm) */
  creaseHeightMm: number;
  /** Khoảng cách 2 góc mắt trong, tuyệt đối (mm) */
  intercanthalDistanceMm: number;
  /** MRD-1 (Marginal Reflex Distance 1), tuyệt đối (mm) */
  mrd1Mm: number;
}

export interface ChinMorph {
  /** Δ vị trí điểm Pog (mm) */
  pogPositionMm: number;
  /** Δ góc hàm V-line (độ, âm = thu gọn) */
  vlineAngleDeg: number;
  /** Δ chiều dài cằm (mm) */
  chinLengthMm: number;
}

export interface BreastMorph {
  /** Size túi ngực, tuyệt đối (cc) */
  sizeCc: number;
  /** Base Width túi ngực, tuyệt đối (mm) */
  baseWidthMm: number;
  /** Độ nhô / Projection, tuyệt đối (mm) */
  projectionMm: number;
  /** Dáng túi: 0 = Tròn, 1 = Giọt nước */
  shape: number;
}

export interface MorphParams {
  nose: NoseMorph;
  eye: EyeMorph;
  chin: ChinMorph;
  breast: BreastMorph;
}

export type WarningSeverity = "info" | "warning" | "danger";

export interface ClinicalWarning {
  code: string;
  severity: WarningSeverity;
  message: string;
}

export interface FacialThirds {
  upper: number;
  middle: number;
  lower: number;
  balanced: boolean;
}

export interface AIAssessment {
  facialThirds: FacialThirds;
  warnings: ClinicalWarning[];
  summary: string;
}

export interface SimulationRecord {
  savedAt: string;
  params: MorphParams;
  afterImageFileName: string;
  aiAssessment: AIAssessment;
}

export type PatientStatus =
  | "moi-tao"
  | "da-tai-anh"
  | "da-tao-mo-hinh"
  | "da-mo-phong";

export interface Patient {
  id: string;
  createdAt: string;
  updatedAt: string;
  fullName: string;
  phone: string;
  address: string;
  /** Tiền sử dị ứng thuốc */
  allergies: string;
  /** Bệnh lý nền */
  underlyingConditions: string;
  /** Tiền sử phẫu thuật */
  surgicalHistory: string;
  services: ServiceKey[];
  photos: Partial<Record<PhotoAngle, PhotoAsset>>;
  /**
   * The 4 doctor-facing profile preview images auto-selected from the
   * latest completed guided scan (see ProfilePreviewImage above). Separate
   * from `photos` (the 4 clinical upload slots feeding the reconstruction
   * texture pipeline) and from `scanSessions[].frames` (the full
   * reconstruction input, never trimmed to 4).
   */
  profilePreview?: Partial<Record<ProfileRole, ProfilePreviewImage>>;
  /** Optional for backward compatibility with existing patient JSON. */
  scanSessions?: ScanSession[];
  clinicalBaseline?: ClinicalBaseline;
  model3d?: {
    before?: Model3DAsset;
  };
  simulation?: SimulationRecord;
  status: PatientStatus;
}

export type PatientInput = Pick<
  Patient,
  | "fullName"
  | "phone"
  | "address"
  | "allergies"
  | "underlyingConditions"
  | "surgicalHistory"
  | "services"
>;
