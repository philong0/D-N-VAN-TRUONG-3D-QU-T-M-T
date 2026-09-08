import type { MorphParams, ServiceKey } from "./types";

export interface ServiceDefinition {
  key: ServiceKey;
  label: string;
  /** Which morph slider group (if any) this service unlocks in the 3D Studio. */
  morphGroup: "nose" | "breast" | "chin" | "eye" | null;
}

export const SERVICES_CATALOG: ServiceDefinition[] = [
  { key: "nang-nguc", label: "Nâng ngực", morphGroup: "breast" },
  { key: "sua-mui-cau-truc", label: "Sửa mũi cấu trúc", morphGroup: "nose" },
  { key: "cat-mi", label: "Cắt / Nhấn mí", morphGroup: "eye" },
  { key: "don-cam-vline", label: "Độn cằm V-line", morphGroup: "chin" },
  { key: "cang-da-mat", label: "Căng da mặt", morphGroup: null },
  { key: "hut-mo", label: "Hút mỡ", morphGroup: null },
];

export function serviceLabel(key: ServiceKey): string {
  return SERVICES_CATALOG.find((s) => s.key === key)?.label ?? key;
}

export function isValidServiceKey(value: string): value is ServiceKey {
  return SERVICES_CATALOG.some((s) => s.key === value);
}

export const DEFAULT_MORPH_PARAMS: MorphParams = {
  nose: {
    heightMm: 0,
    tipProjectionMm: 0,
    nasolabialAngleDeg: 97,
    nasofrontalAngleDeg: 122,
    columellaMm: 0,
  },
  eye: {
    creaseHeightMm: 0,
    intercanthalDistanceMm: 32,
    mrd1Mm: 4.5,
  },
  chin: {
    pogPositionMm: 0,
    vlineAngleDeg: 0,
    chinLengthMm: 0,
  },
  breast: {
    sizeCc: 250,
    baseWidthMm: 120,
    projectionMm: 45,
    shape: 0.5,
  },
};

export interface SliderDefinition {
  key: string;
  label: string;
  min: number;
  max: number;
  step: number;
  unit?: string;
  /** Reference "safe/ideal" sub-range highlighted in the slider UI, if any. */
  idealRange?: [number, number];
}

export type MorphGroup = "nose" | "breast" | "chin" | "eye";

export const MORPH_SLIDERS: Record<MorphGroup, { title: string; sliders: SliderDefinition[] }> = {
  nose: {
    title: "Mũi",
    sliders: [
      { key: "heightMm", label: "Nâng cao sống mũi (Δ)", min: -2, max: 5, step: 0.1, unit: "mm" },
      { key: "tipProjectionMm", label: "Bọc đầu mũi / Tip Projection (Δ)", min: -2, max: 4, step: 0.1, unit: "mm" },
      { key: "nasolabialAngleDeg", label: "Góc Mũi–Môi (Nasolabial)", min: 85, max: 110, step: 1, unit: "°", idealRange: [95, 100] },
      { key: "nasofrontalAngleDeg", label: "Góc Trán–Mũi (Nasofrontal)", min: 105, max: 140, step: 1, unit: "°", idealRange: [115, 130] },
      { key: "columellaMm", label: "Trụ mũi (Δ)", min: -2, max: 3, step: 0.1, unit: "mm" },
    ],
  },
  breast: {
    title: "Ngực",
    sliders: [
      { key: "sizeCc", label: "Size túi ngực", min: 150, max: 500, step: 10, unit: "cc" },
      { key: "baseWidthMm", label: "Base Width", min: 90, max: 140, step: 1, unit: "mm" },
      { key: "projectionMm", label: "Projection", min: 30, max: 60, step: 1, unit: "mm" },
      { key: "shape", label: "Dáng (Tròn ↔ Giọt nước)", min: 0, max: 1, step: 0.01 },
    ],
  },
  chin: {
    title: "Cằm / Hàm",
    sliders: [
      { key: "pogPositionMm", label: "Đường Pog (Δ)", min: -2, max: 5, step: 0.1, unit: "mm" },
      { key: "vlineAngleDeg", label: "Góc cằm V-line (Δ thu gọn)", min: -10, max: 0, step: 0.5, unit: "°" },
      { key: "chinLengthMm", label: "Chiều dài cằm (Δ)", min: -2, max: 5, step: 0.1, unit: "mm" },
    ],
  },
  eye: {
    title: "Mắt",
    sliders: [
      { key: "creaseHeightMm", label: "Nhấn mí — chiều cao nếp mí (Δ)", min: 0, max: 4, step: 0.1, unit: "mm" },
      { key: "intercanthalDistanceMm", label: "Khoảng cách 2 góc mắt trong", min: 28, max: 38, step: 0.5, unit: "mm" },
      { key: "mrd1Mm", label: "MRD-1", min: 2, max: 6, step: 0.1, unit: "mm" },
    ],
  },
};
