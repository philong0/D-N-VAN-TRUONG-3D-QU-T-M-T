import type { PhotoAngle, ServiceKey } from "./types";

export const PHOTO_ANGLES: PhotoAngle[] = ["angle1", "angle2", "angle3", "angle4"];

export type PrimaryPhotoCategory = "nose" | "eye" | "breast" | "other";

export interface SlotLabel {
  label: string;
  hint: string;
}

const NOSE_SLOTS: SlotLabel[] = [
  { label: "Chính diện (0°)", hint: "Soi độ thẳng sống mũi & cánh mũi." },
  { label: "Nghiêng 45°", hint: "Soi độ cao sống mũi & độ nhô đầu mũi." },
  { label: "Nghiêng 90° (Ngang)", hint: "Đo góc Mũi-Môi & đường E-Line." },
  { label: "Dưới lên (Base View)", hint: "Soi trụ mũi, vách ngăn & lỗ mũi hạt chanh." },
];

const EYE_SLOTS: SlotLabel[] = [
  { label: "Mắt nhìn thẳng (0°)", hint: "Đánh giá nếp mí & sụp mi." },
  { label: "Mắt nhắm nhẹ", hint: "Soi nếp gấp cơ nâng mi, da thừa & bọng mỡ." },
  { label: "Mắt ngước nhìn lên", hint: "Đánh giá túi mỡ dưới & nếp nhăn." },
  { label: "Nghiêng 45° Mắt", hint: "Soi hốc mắt & góc mắt trong/ngoài." },
];

const BREAST_SLOTS: SlotLabel[] = [
  { label: "Chính diện Thân trên (0°)", hint: "Đo khe ngực & độ lệch 2 bên." },
  { label: "Nghiêng 45° Thân trên", hint: "Soi độ nhô bầu ngực & cực trên." },
  { label: "Nghiêng 90° (Góc Ngang Body)", hint: "Đánh giá độ sa trễ & chân ngực." },
  { label: "Cúi 45° từ trên xuống", hint: "Đánh giá độ đổ tự nhiên mô tuyến vú." },
];

const OTHER_SLOTS: SlotLabel[] = [
  { label: "Góc 1", hint: "" },
  { label: "Góc 2", hint: "" },
  { label: "Góc 3", hint: "" },
  { label: "Góc 4", hint: "" },
];

/**
 * Only one photo composition can drive the 4 upload slots at a time, so when
 * a patient has several services we pick a single primary category by this
 * priority. Services without a dedicated angle set (cằm, căng da, hút mỡ)
 * fall back to the generic slots.
 */
export function resolvePrimaryCategory(services: ServiceKey[]): PrimaryPhotoCategory {
  if (services.includes("sua-mui-cau-truc")) return "nose";
  if (services.includes("cat-mi")) return "eye";
  if (services.includes("nang-nguc")) return "breast";
  return "other";
}

export function getSlotLabels(services: ServiceKey[]): SlotLabel[] {
  switch (resolvePrimaryCategory(services)) {
    case "nose":
      return NOSE_SLOTS;
    case "eye":
      return EYE_SLOTS;
    case "breast":
      return BREAST_SLOTS;
    default:
      return OTHER_SLOTS;
  }
}
