"use client";

import { useEffect, useMemo, useRef, useState, useTransition } from "react";
import type { Canvas3DHandle } from "@/components/Canvas3D";
import ClinicalAdvisorPanel from "@/components/ClinicalAdvisorPanel";
import ControlPanel3D from "@/components/ControlPanel3D";
import SplitCompare from "@/components/SplitCompare";
import { PHOTO_ANGLES, resolvePrimaryCategory, type SlotLabel } from "@/lib/photo-angles";
import { DEFAULT_MORPH_PARAMS, SERVICES_CATALOG, type MorphGroup } from "@/lib/services-catalog";
import { BUTTON_PRIMARY_CLASS, BUTTON_SECONDARY_CLASS, INPUT_CLASS } from "@/lib/ui";
import type { ClinicalBaseline, MorphParams, PhotoAngle, PhotoAsset, ServiceKey } from "@/lib/types";
import { saveSimulationAction, updateClinicalBaselineAction } from "../../actions";

interface StudioClientProps {
  patientId: string;
  photos: Partial<Record<PhotoAngle, PhotoAsset>>;
  slotLabels: SlotLabel[];
  services: ServiceKey[];
  clinicalBaseline?: ClinicalBaseline;
  initialParams?: MorphParams;
  /** Previously-saved 2D "after" simulation photo, if any — see SplitCompare.tsx. */
  afterImageUrl?: string | null;
}

interface QuickView {
  label: string;
  slot: number;
  azimuthDeg: number;
  /** Degrees from straight overhead — 90 = eye-level (Canvas3D's own default). Omitted = eye-level. */
  polarDeg?: number;
}

// Quick views: -90° (Profile Trái), -45° (Nghiêng Trái), 0° (Chính diện), +45° (Nghiêng Phải), +90° (Profile Phải)
const BASE_QUICK_VIEWS: QuickView[] = [
  { label: "-90°", slot: 1, azimuthDeg: -90 },
  { label: "-45°", slot: 1, azimuthDeg: -45 },
  { label: "0°", slot: 0, azimuthDeg: 0 },
  { label: "+45°", slot: 2, azimuthDeg: 45 },
  { label: "+90°", slot: 2, azimuthDeg: 90 },
];

// Slot 3's real meaning depends on the patient's primary service (see
// lib/photo-angles.ts's NOSE_SLOTS/EYE_SLOTS/BREAST_SLOTS/OTHER_SLOTS) — only
// "nose" unambiguously means a from-below base view (nostrils/columella/jaw),
// which is the one the reference explicitly requires being reachable with one
// click. For eye/breast/other, slot 3 means something the camera can't
// represent as a simple azimuth/polar preset (an expression, a body angle),
// so it's deliberately left without a quick-view preset here — unchanged
// from before (that slot was already unused for camera presets pre-BƯỚC 4).
const BELOW_VIEW: QuickView = { label: "Dưới lên", slot: 3, azimuthDeg: 0, polarDeg: 130 };

export default function StudioClient({
  patientId,
  photos,
  slotLabels,
  services,
  clinicalBaseline,
  initialParams,
  afterImageUrl,
}: StudioClientProps) {
  const [params, setParams] = useState<MorphParams>(initialParams ?? DEFAULT_MORPH_PARAMS);
  const [selectedRegion, setSelectedRegion] = useState<MorphGroup | null>(null);
  const [activeSlot, setActiveSlot] = useState(0);
  const [saved, setSaved] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isSaving, startSaving] = useTransition();
  // Workspace chrome — purely presentational, never touches the mounted
  // SplitCompare/Canvas3D instances below (no remount on toggle), so camera
  // position and the fitted GNM/3DDFA identity are untouched by any of this.
  const [isThumbDrawerOpen, setIsThumbDrawerOpen] = useState(false);
  const [isPanelOpen, setIsPanelOpen] = useState(true);
  const [isAutoRotating, setIsAutoRotating] = useState(false);
  // Fix 1 (2026-08-24 visual-defect audit) — "identity-only" review toggle:
  // see Canvas3D.tsx's own `renderMode` doc. Purely a client-side color
  // override on the existing mesh, so toggling never re-fits/re-fetches.
  const [renderMode, setRenderMode] = useState<"full" | "identity-only">("full");
  const [isFullscreen, setIsFullscreen] = useState(false);
  const canvasRef = useRef<Canvas3DHandle>(null);
  const rootRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    function onFullscreenChange() {
      setIsFullscreen(document.fullscreenElement === rootRef.current);
    }
    document.addEventListener("fullscreenchange", onFullscreenChange);
    return () => document.removeEventListener("fullscreenchange", onFullscreenChange);
  }, []);

  const unlockedGroups = useMemo(() => {
    const set = new Set<MorphGroup>();
    for (const key of services) {
      const service = SERVICES_CATALOG.find((s) => s.key === key);
      if (service?.morphGroup) set.add(service.morphGroup);
    }
    return set;
  }, [services]);

  const photoUrl = (angle: PhotoAngle) => {
    const asset = photos[angle];
    return asset ? `/api/files/${patientId}/photos/${asset.fileName}` : null;
  };
  const frontPhotoUrl = photoUrl(PHOTO_ANGLES[0]); // angle1, 0°
  // Multi-view GNM fit: combine the frontal (0°) photo with the 45° and 90°
  // photos when available (see Canvas3D's `obliquePhotoUrl`/`profilePhotoUrl`
  // props / lib/gnm/fit.ts's `fitGnmHeadToMultiViewLandmarks`, which takes
  // any number of views). If MediaPipe can't detect a face in the 90° shot
  // (true profiles are unreliable for MediaPipe — verified empirically),
  // Canvas3D silently drops that view and fits from whichever of 0°/45° are
  // still available.
  const obliquePhotoUrl = photoUrl(PHOTO_ANGLES[1]); // angle2, 45°
  const profilePhotoUrl = photoUrl(PHOTO_ANGLES[2]); // angle3, 90°
  // "From below" base-view photo (angle4) — only meaningful for the "nose"
  // service category (see lib/photo-angles.ts's NOSE_SLOTS); combined into
  // the same multi-view ridge solve as the other views (see Canvas3D's
  // `belowPhotoUrl` prop / lib/gnm/fit.ts's multi-view fit).
  const belowPhotoUrl = resolvePrimaryCategory(services) === "nose" ? photoUrl(PHOTO_ANGLES[3]) : null;

  const showBelowView = resolvePrimaryCategory(services) === "nose" && Boolean(photos[PHOTO_ANGLES[3]]);
  const quickViews = showBelowView ? [...BASE_QUICK_VIEWS, BELOW_VIEW] : BASE_QUICK_VIEWS;
  const thumbnailSlots = showBelowView ? [0, 1, 2, 3] : [0, 1, 2];

  function goToView(view: QuickView) {
    setActiveSlot(view.slot);
    canvasRef.current?.goToAngle(view.azimuthDeg, view.polarDeg);
  }

  function resetView() {
    setActiveSlot(0);
    canvasRef.current?.resetView();
  }

  async function toggleFullscreen() {
    if (document.fullscreenElement) {
      await document.exitFullscreen();
      return;
    }
    if (rootRef.current) {
      await rootRef.current.requestFullscreen();
      // Reference behavior: entering fullscreen collapses the side drawers so
      // the 3D canvas gets the space — the toolbar stays usable to reopen
      // them, camera/simulation state is untouched either way.
      setIsThumbDrawerOpen(false);
      setIsPanelOpen(false);
    }
  }

  function updateSlider(group: MorphGroup, sliderKey: string, value: number) {
    setSaved(false);
    setParams((prev) => ({
      ...prev,
      [group]: { ...prev[group], [sliderKey]: value },
    }));
  }

  function resetSliders() {
    setSaved(false);
    setParams(DEFAULT_MORPH_PARAMS);
  }

  function handleSave() {
    setError(null);
    const dataUrl = canvasRef.current?.captureSnapshot ? canvasRef.current.captureSnapshot() : null;
    if (!dataUrl) {
      setError("Không thể chụp ảnh mô phỏng. Vui lòng thử lại.");
      return;
    }

    startSaving(async () => {
      try {
        const blob = await (await fetch(dataUrl)).blob();
        const formData = new FormData();
        formData.set("params", JSON.stringify(params));
        formData.set("snapshot", new File([blob], "after.png", { type: "image/png" }));
        await saveSimulationAction(patientId, formData);
        setSaved(true);
      } catch {
        setError("Lưu kết quả mô phỏng thất bại. Vui lòng thử lại.");
      }
    });
  }

  return (
    <div
      ref={rootRef}
      className={
        isFullscreen
          ? "fixed inset-0 z-50 flex h-screen w-screen flex-col gap-0 bg-zinc-950 p-0"
          : "mt-8 flex flex-col gap-0 overflow-hidden rounded-2xl border border-zinc-800 bg-zinc-950 shadow-xl"
      }
    >
      {/* Toolbar — dark studio chrome (luôn hiện, không che canvas). */}
      <div className="flex flex-wrap items-center gap-2 border-b border-zinc-800 bg-zinc-900/90 px-3 py-2.5 backdrop-blur">
        <button
          type="button"
          onClick={() => setIsThumbDrawerOpen((v) => !v)}
          aria-pressed={isThumbDrawerOpen}
          className={`rounded-full border px-3 py-1.5 text-xs font-medium transition-colors ${
            isThumbDrawerOpen
              ? "border-accent bg-accent/15 text-accent"
              : "border-zinc-700 text-zinc-300 hover:border-accent/50 hover:text-zinc-100"
          }`}
        >
          {isThumbDrawerOpen ? "Ẩn ảnh gốc ▲" : "Ảnh gốc ▾"}
        </button>

        <div className="mx-1 h-5 w-px bg-zinc-700" />

        {quickViews.map((view) => (
          <button
            key={view.label}
            type="button"
            onClick={() => goToView(view)}
            aria-pressed={activeSlot === view.slot}
            className={`rounded-full border px-3 py-1.5 text-xs font-medium transition-colors ${
              activeSlot === view.slot
                ? "border-accent bg-accent/15 text-accent shadow-[var(--accent-glow)]"
                : "border-zinc-700 text-zinc-300 hover:border-accent/50 hover:text-zinc-100"
            }`}
          >
            {view.label}
          </button>
        ))}

        <div className="mx-1 h-5 w-px bg-zinc-700" />

        <button
          type="button"
          onClick={resetView}
          className="rounded-full border border-zinc-700 px-3 py-1.5 text-xs font-medium text-zinc-300 hover:border-accent/50 hover:text-zinc-100"
        >
          ⤾ Reset
        </button>
        <button
          type="button"
          onClick={() => setIsAutoRotating((v) => !v)}
          aria-pressed={isAutoRotating}
          className={`rounded-full border px-3 py-1.5 text-xs font-medium transition-colors ${
            isAutoRotating
              ? "border-accent bg-accent/15 text-accent"
              : "border-zinc-700 text-zinc-300 hover:border-accent/50 hover:text-zinc-100"
          }`}
        >
          ⟳ Tự xoay
        </button>
        <button
          type="button"
          onClick={() => setRenderMode((v) => (v === "identity-only" ? "full" : "identity-only"))}
          aria-pressed={renderMode === "identity-only"}
          title="Chỉ tô màu thật vùng mặt identity-critical (mắt/mũi/miệng/má/trán); phần đầu còn lại chuyển sang tông màu generic — không cắt hình học, ranh giới mesh không đổi."
          className={`rounded-full border px-3 py-1.5 text-xs font-medium transition-colors ${
            renderMode === "identity-only"
              ? "border-accent bg-accent/15 text-accent"
              : "border-zinc-700 text-zinc-300 hover:border-accent/50 hover:text-zinc-100"
          }`}
        >
          ◐ Chỉ vùng mặt
        </button>

        <div className="ml-auto flex items-center gap-2">
          <button
            type="button"
            onClick={() => setIsPanelOpen((v) => !v)}
            aria-pressed={isPanelOpen}
            className={`rounded-full border px-3 py-1.5 text-xs font-medium transition-colors ${
              isPanelOpen
                ? "border-accent bg-accent/15 text-accent"
                : "border-zinc-700 text-zinc-300 hover:border-accent/50 hover:text-zinc-100"
            }`}
          >
            {isPanelOpen ? "Ẩn công cụ ›" : "‹ Công cụ"}
          </button>
          <button
            type="button"
            onClick={toggleFullscreen}
            className="rounded-full border border-zinc-700 px-3 py-1.5 text-xs font-medium text-zinc-300 hover:border-accent/50 hover:text-zinc-100"
          >
            {isFullscreen ? "⤢ Thoát toàn màn hình" : "⛶ Toàn màn hình"}
          </button>
        </div>
      </div>

      <p className="border-b border-zinc-800 bg-zinc-900/50 px-3 py-1.5 text-xs text-zinc-500">
        Kéo chuột trái: xoay (ngang ±45°, dọc từ trên đỉnh đầu đến dưới cằm) · Chuột phải: pan · Lăn chuột: zoom ·
        Nhấp đúp vào canvas: reset view
      </p>

      {/* Workspace: On mobile, stack vertically so 3D Canvas gets full width; on desktop (lg:), use flex-row */}
      <div className="flex flex-col lg:flex-row flex-1 gap-0 lg:min-h-0 w-full">
        {isThumbDrawerOpen && (
          <div className="hidden lg:block w-[112px] shrink-0 overflow-hidden border-r border-zinc-800">
            <div className="flex h-full flex-col gap-2 overflow-y-auto bg-zinc-900/60 p-2">
              {thumbnailSlots.map((i) => {
                const angle = PHOTO_ANGLES[i];
                const url = photoUrl(angle);
                const view = quickViews.find((v) => v.slot === i);
                const active = i === activeSlot;
                return (
                  <button
                    key={angle}
                    type="button"
                    onClick={() => view && goToView(view)}
                    disabled={!view}
                    className={`flex w-full shrink-0 flex-col gap-1 rounded-lg border p-1 text-left transition-colors disabled:cursor-default disabled:opacity-70 ${
                      active
                        ? "border-accent bg-accent/10 shadow-[var(--accent-glow)]"
                        : "border-zinc-700 hover:border-accent/50"
                    }`}
                  >
                    <div className="aspect-[3/4] w-full overflow-hidden rounded-md bg-zinc-800">
                      {url ? (
                        // eslint-disable-next-line @next/next/no-img-element
                        <img src={url} alt={slotLabels[i]?.label ?? angle} className="h-full w-full object-cover" />
                      ) : (
                        <div className="flex h-full items-center justify-center text-[9px] text-zinc-500">Trống</div>
                      )}
                    </div>
                    <p className={`text-[10px] leading-tight ${active ? "font-medium text-accent" : "text-zinc-400"}`}>
                      {view?.label ?? slotLabels[i]?.label ?? ""}
                    </p>
                  </button>
                );
              })}
            </div>
          </div>
        )}

        {/* Main 3D Canvas Viewport */}
        <div className={`flex-1 ${isFullscreen ? "h-full min-h-0 w-full" : "h-[50vh] sm:h-[60vh] lg:h-[75vh] min-h-[360px] lg:min-h-[500px] w-full relative"}`}>
          <SplitCompare
            ref={canvasRef}
            photoUrl={frontPhotoUrl}
            obliquePhotoUrl={obliquePhotoUrl}
            profilePhotoUrl={profilePhotoUrl}
            belowPhotoUrl={belowPhotoUrl}
            patientId={patientId}
            morph={params}
            unlockedGroups={unlockedGroups}
            afterImageUrl={afterImageUrl}
            autoRotate={isAutoRotating}
            renderMode={renderMode}
          />
        </div>

        {/* Side / Bottom Control Panel */}
        {isPanelOpen && (
          <div className="w-full lg:w-[340px] shrink-0 overflow-y-auto border-t lg:border-t-0 lg:border-l border-zinc-800 bg-zinc-900 p-4 text-zinc-100 flex flex-col gap-4">
            <ControlPanel3D
              params={params}
              unlockedGroups={unlockedGroups}
              selectedRegion={selectedRegion}
              onSelectRegion={setSelectedRegion}
              onChange={updateSlider}
            />

            <div className="flex flex-col gap-2 border-t border-zinc-700 pt-4">
              <button
                type="button"
                onClick={resetSliders}
                className="rounded-lg border border-zinc-700 px-4 py-2 text-sm font-medium text-zinc-200 transition-colors hover:bg-zinc-800"
              >
                Đặt lại mặc định
              </button>
              <button type="button" onClick={handleSave} disabled={isSaving} className={BUTTON_PRIMARY_CLASS}>
                {isSaving ? "Đang lưu..." : "LƯU KẾT QUẢ MÔ PHỎNG 3D VÀO BỆNH ÁN"}
              </button>
              {error && <p className="text-xs text-danger">{error}</p>}
              {saved && <p className="text-xs text-success">Đã lưu kết quả mô phỏng.</p>}
            </div>

            <div className="rounded-lg bg-zinc-800 p-3">
              <ClinicalAdvisorPanel morph={params} clinicalBaseline={clinicalBaseline} />
            </div>

            <details className="rounded-lg border border-zinc-700 p-4">
              <summary className="cursor-pointer text-sm font-medium text-zinc-100">
                Thông số lâm sàng nền (nhập bởi bác sĩ)
              </summary>
              <form action={updateClinicalBaselineAction.bind(null, patientId)} className="mt-3 flex flex-col gap-3">
                <div>
                  <label htmlFor="chestBaseWidthMm" className="mb-1 block text-xs text-zinc-400">
                    Bề rộng khung ngực đo được (mm)
                  </label>
                  <input
                    id="chestBaseWidthMm"
                    name="chestBaseWidthMm"
                    type="number"
                    step="1"
                    defaultValue={clinicalBaseline?.chestBaseWidthMm}
                    className={INPUT_CLASS}
                    placeholder="vd: 130"
                  />
                </div>
                <div>
                  <label htmlFor="softTissueThicknessCm" className="mb-1 block text-xs text-zinc-400">
                    Độ dày mô mềm che phủ (cm)
                  </label>
                  <input
                    id="softTissueThicknessCm"
                    name="softTissueThicknessCm"
                    type="number"
                    step="0.1"
                    defaultValue={clinicalBaseline?.softTissueThicknessCm}
                    className={INPUT_CLASS}
                    placeholder="vd: 1.5"
                  />
                </div>
                <button type="submit" className={BUTTON_SECONDARY_CLASS + " !px-4 !py-2 text-xs"}>
                  Lưu thông số nền
                </button>
              </form>
            </details>
          </div>
        )}
      </div>

      <p className="border-t border-zinc-800 bg-zinc-900/50 px-3 py-2 text-center text-[11px] leading-relaxed text-zinc-500">
        Ứng dụng này chỉ phục vụ mục đích minh họa và tham khảo. Nó không xác nhận hoặc đảm bảo bất kỳ kết quả nào.
      </p>
    </div>
  );
}
