import type { PatientStatus } from "./types";

export const STATUS_LABELS: Record<PatientStatus, string> = {
  "moi-tao": "Mới tạo hồ sơ",
  "da-tai-anh": "Đã tải ảnh trước phẫu thuật",
  "da-tao-mo-hinh": "Đã tạo mô hình 3D",
  "da-mo-phong": "Đã mô phỏng & lưu kết quả",
};

export const STATUS_STEP: Record<PatientStatus, number> = {
  "moi-tao": 1,
  "da-tai-anh": 2,
  "da-tao-mo-hinh": 3,
  "da-mo-phong": 4,
};
