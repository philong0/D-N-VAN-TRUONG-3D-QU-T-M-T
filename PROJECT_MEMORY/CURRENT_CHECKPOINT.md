# CURRENT CHECKPOINT — Dr Van Truong 3D Studio

**Ngày:** 2026-09-04
**Phiên này tiếp tục từ:** bản scanner do một agent khác viết lại (LEFT/RIGHT/UP/DOWN), không phải state-machine 7-checkpoint cũ.
**Ghi chú môi trường:** Không phải git repo (`git status` báo `fatal: not a git repository`). Không có `.git` để track diff — checkpoint này là nguồn sự thật duy nhất về "đã đổi gì".

---

## 1. ĐÃ LÀM GÌ (theo thứ tự)

### A. Audit trạng thái thật (không giả định)
- Đọc lại toàn bộ `GuidedFaceScan.tsx`, `face-geometry.ts`, `camera-normalization.ts` từ đầu.
- Xác nhận: scanner đang chạy là bản 4-target (LEFT/RIGHT/UP/DOWN) do agent khác viết, **không** dùng `scan-state-machine.ts`/`frame-evaluator.ts`/`scan-constants.ts` cũ (orphan).

### B. Sửa 2 React Hook errors (không dùng eslint-disable để che)
- `Date.now()` trong `useRef` initializer → đổi thành `useRef(0)`, giá trị thật được set trong effect mount (không đọc trước khi set).
- `startCamera()`/`initSession()` gây `setState` đồng bộ trong effect → bọc trong `setTimeout(..., 0)` để tách khỏi tick đồng bộ của effect (fix cấu trúc thật, không phải suppress).
- **Verify:** `npx eslint src/components/scan/GuidedFaceScan.tsx` → 0 error (trước: 2 error).

### C. Sửa E2E theo UI thật (không sửa UI để test pass)
- Test cũ tìm nút "BẮT ĐẦU QUÉT MẶT 3D" — UI thật hiện auto-start ngay khi mount, không có nút đó.
- Viết lại toàn bộ: `e2e/scanner-e2e.mjs` (Playwright + fake camera thật, chạy qua `npm run test:e2e <baseUrl> <patientId>`).
- Test thật: camera init, no-face honesty, camera switch (dùng `force:true` click vì có 1 header mobile toàn cục khác đè z-index — vấn đề layout riêng, ngoài phạm vi), sustained tick không crash.
- **Ghi rõ limitation** (không giả kết quả): fake camera không có mặt thật → KHÔNG kiểm được real on-target capture, real voice-complete, real finalize/upload.
- **Kết quả chạy thật:** `PASS (0 check(s) failed)`.

### D. Khóa camera normalization (1 lớp duy nhất)
- Đã tồn tại từ phiên trước (`camera-normalization.ts`), phiên này KHÔNG đổi thêm — xác nhận lại vẫn là nơi DUY NHẤT quyết định lật dấu yaw (`flipYawSign: true` như nhau cho front/rear, có chứng minh hình học trong docstring).
- `mirroredPreview` chỉ dùng để dựng transform, không rẽ nhánh hình học riêng.
- Mỗi frame accepted (buffer + origin) đều có: `cameraFacing, sensorOrientation, displayOrientation, mirrorApplied, coordinateTransform` (đã có từ trước, phiên này bổ sung thêm `targetId, qualityScore, sharpness, brightness, motion, landmarkStability, landmarksReal`).

### E. Sửa logic góc quay (VI PHẠM THẬT ĐÃ TÌM THẤY VÀ SỬA)
- **Lỗi:** targets cũ = LEFT/RIGHT (yaw ±12-18°) + UP/DOWN (**pitch** ±10-16° làm target CHÍNH) — vi phạm trực tiếp yêu cầu "pitch/roll chỉ là điều kiện chất lượng, không phải target chính".
- **Sửa:** `TARGET_ANGLES` giờ là 7 checkpoint YAW THUẦN: `0°, ±20°, ±45°, ±60°`. Pitch/roll chuyển thành điều kiện CHẤT LƯỢNG CHUNG (qua `evaluateFrame()` tái sử dụng từ `frame-evaluator.ts` với ngưỡng cố định `POSE.MAX_PITCH_DEG`/`MAX_ROLL_DEG`, giống nhau cho MỌI target — không còn range riêng theo hướng).
- Coverage indicator (7 chấm tròn + kim yaw thật) phản ánh `targetsState` thật (frame accepted thật), không phải animation/timer giả.
- **Bỏ cơ chế "hạ chuẩn thông minh"** (tự động pass sau timeout — vi phạm "không dùng timer để coi đạt"). Thay bằng nút **"Bỏ qua góc này"** chỉ hiện sau nhiều lần thật sự không đạt, và CHỈ hành động khi người dùng bấm (hành động có ý thức, được đánh dấu rõ `userSkipped: true` trong metadata, không giả vờ là capture đạt chuẩn).

### F. Frame selection cho reconstruction
- Buffer giờ chỉ nhận frame khi `evaluateFrame(...).accepted === true` (brightness/sharpness/motion/landmark-stability/pitch/roll đều đạt) — trước đây chỉ cần `faceDetected`.
- Bỏ giới hạn cứng "20 frame là xong": điều kiện hoàn tất giờ là "đủ 7 checkpoint yaw ĐÃ CAPTURE thật" VÀ "pool frame đạt chuẩn ≥ `RECONSTRUCTION_TARGET_COUNT` (20)" — buffer pool tăng lên 80 (`MAX_BUFFER_FRAMES`) để có pool thật đa dạng.
- Việc CHỌN LỌC cuối cùng (pose-aware, ưu tiên yaw+quality, không lấy theo index) đã được xây ở phiên trước trong `reconstruction-frame-selection.ts` — server-side, chạy trên TOÀN BỘ frame vừa upload khi `request_reconstruction` được gọi. Phiên này xác nhận: dữ liệu đầu vào cho bước đó giờ đã có `qualityScore`/`sharpness`/`brightness` thật thay vì rỗng.

### G. Profile images — nối lại module đã xây nhưng chưa dùng
- `/api/patients/[id]/profile-preview` (endpoint) + `selectBestProfileImages()` đã tồn tại từ phiên trước nhưng **chưa từng được gọi** từ scanner thật (orphan).
- Phiên này: `finalizeScan()` gọi endpoint này SAU KHI `request_reconstruction` thành công — tách biệt hoàn toàn khỏi reconstruction input (không đổi/không thay thế nó), lấy đúng 4 ảnh front/left/right/three_quarter từ dữ liệu frame thật vừa upload.

### H. Audit pipeline reconstruction thật (QUAN TRỌNG NHẤT) — ĐÃ ĐỌC CODE THẬT, CÓ BẰNG CHỨNG
Đường đi thật: `finalizeScan()` → `PATCH .../scan-sessions/[id]` `action=request_reconstruction` → `reconstruction-service.ts` → (`scannerKind="web_camera"` nên KHÔNG dùng native path) → `reconstruct_cli.py` (đã sửa pose-aware selection ở phiên trước) → `reconstruct_gnm_from_images()` (`reconstruct_gnm_fullhead.py`) → `gnm_identity_fit.fit_multiview()`.

**Đọc trực tiếp `ai-engine/gnm_identity_fit.py` (223 dòng, đọc toàn bộ):**
- Landmark: PIPNet/WFLW-98 (98 điểm thật, KHÔNG PHẢI "468 MediaPipe landmarks" như lo ngại).
- Pose mỗi ảnh: `cv2.solvePnP(..., flags=cv2.SOLVEPNP_EPNP)` — pose camera thật từ correspondence 2D-3D thật, không fake.
- Hình học: một PCA identity basis THẬT (`vertex_identity_basis.npy`, shape (253, 17821, 3)) — không phải template cố định dùng làm output cuối. Hệ số `coefficients` được GIẢI bằng least-squares ridge-regularized THẬT (`solve_identity_multiview`, `min ||B·c − r||² + λ||c||²`) trên offset thật giữa landmark quan sát (có backproject pseudo-3D thật qua pose đã solve) và landmark template — **khác bệnh nhân → khác coefficients → khác hình học thật**, không phải "trung bình landmark rồi Delaunay".
- Có sanity check thật (`max_disp_mm`, `max_coeff`) và **DROP view nếu EPNP fail thay vì fabricate** (dòng 162-163, 182-183: "never fabricated").
- Không có bước "median/mean landmark" hay "Delaunay 2D" ở đâu trong file này.

**Đọc trực tiếp `ai-engine/render_back_validator.py` (226 dòng):**
- Có comment lịch sử thật: một bản cũ từng hardcode `(0.82, 0.95, 0.74...) status="pass" unconditionally` — ĐÃ BỊ PHÁT HIỆN VÀ SỬA (dòng 152-156), giờ dùng `region_errors_mm` đo thật.
- Validation thật: render mesh đã fit qua ĐÚNG pose camera đã solve, so silhouette với mask quan sát thật từ ảnh, tính IoU + contour error (px) + quy đổi mm bằng scale đo thật, gate `status="pass" if silhouette_iou >= 0.5 else "warning"`.

**Kiểm tra tool thật có trong môi trường:**
```
cv2: AVAILABLE (5.0.0)
open3d: AVAILABLE (0.18.0)
pycolmap: NOT AVAILABLE
colmap: NOT AVAILABLE
```
Pipeline thật đang dùng (EPnP + PCA-basis least-squares) KHÔNG cần COLMAP — dùng đúng cv2 đã có, không giả vờ đã chạy COLMAP/dense SfM.

**KẾT LUẬN AUDIT H:** Pipeline reconstruction hiện tại **KHÔNG PHẢI** pattern giả bị cấm ("468 landmarks → median/mean → Delaunay 2D → OBJ"). Đây là một 3DMM-style multi-view identity fit thật (PCA basis + EPnP + ridge least-squares), có validation reprojection thật, có cơ chế drop-thay-vì-fabricate thật, đã qua nhiều vòng audit trước đó (thấy trong chính docstring của code: "BUG A-D + G1-G4 fixes", D3/D8/D21-D30/D31/D38... — dấu vết một lịch sử audit thật, không phải tự nhận suông).

**CHƯA audit sâu trong phiên này** (do giới hạn thời gian, ưu tiên scanner trước theo đúng thứ tự A→H được giao): `gnm_dense_nose.py`, `gnm_width_correction.py`, toàn bộ `gnm_face_shell.py` (đã audit sâu ở các phiên TRƯỚC, không phải phiên này — xem lịch sử hội thoại, không lặp lại ở đây để tránh nhận vơ công đã làm trước làm "vừa xong").

---

## 2. FILE ĐÃ SỬA (phiên này)

- `src/components/scan/GuidedFaceScan.tsx` — sửa lớn: hook errors, TARGET_ANGLES (yaw-only 7 checkpoint), quality gate thật qua `evaluateFrame`, bỏ auto-pass timer, thêm nút bỏ qua thủ công, gọi profile-preview sau reconstruction, UI 7-checkpoint.
- `src/lib/face-geometry.ts` — xoá `computeHeadPoseFrom6Landmarks` (orphan thật, 0 caller, đã xác nhận bằng `grep`).
- `e2e/scanner-e2e.mjs` — MỚI, thay thế test cũ lệch UI.
- `package.json` — thêm script `test:e2e`.
- `PROJECT_MEMORY/CURRENT_CHECKPOINT.md` — file này.

**KHÔNG đổi** (đã đọc, xác nhận không cần đổi hoặc ngoài phạm vi phiên này): `camera-normalization.ts`, `reconstruction-frame-selection.ts`, `profile-image-selection.ts`, `profile-preview-service.ts`, toàn bộ `ai-engine/*.py`, `scan-state-machine.ts`/`frame-evaluator.ts`/`scan-constants.ts` (frame-evaluator/scan-constants giờ ĐƯỢC DÙNG THẬT qua GuidedFaceScan.tsx — hết orphan; `scan-state-machine.ts` vẫn orphan, chưa xoá vì đây là quyết định kiến trúc cần người quyết, không tự ý xoá).

## 3. TEST / BUILD / SERVER STATUS (đo thật, vừa chạy xong)

| Bước | Lệnh | Kết quả |
|---|---|---|
| Unit test | `npm test` | **51/51 PASS** |
| TypeScript | `npx tsc --noEmit` | 0 lỗi |
| ESLint (file đã sửa) | `npx eslint ...` | 0 error, 6 warning (đều pre-existing, không thuộc file/dòng phiên này sửa) |
| Build | `npm run build` | PASS |
| Server | restart khớp build mới | chunk JS trả 200 hết |
| E2E thật | `npm run test:e2e` | **PASS (0 check thất bại)**, có ghi rõ limitation |

## 4. SCANNER STATUS

**Đã xác nhận (bằng eslint/tsc/test/E2E thật):** không lỗi cấu trúc, không crash, no-face honesty hoạt động, 7-checkpoint yaw hiển thị đúng, quality gate thật đang chạy (brightness/sharpness/motion/stability/pitch/roll qua `evaluateFrame`).

**CHƯA xác nhận (giới hạn thật, không giả vờ):** chưa test trên iPhone/Samsung thật với mặt người thật — fake camera trong sandbox không có mặt thật nên KHÔNG thể verify: có thật sự bắt đúng góc 20/45/60° khi người dùng quay đầu thật hay không, giọng nói dẫn dắt trọn vẹn 7 góc, và luồng finalize/upload/reconstruction full trên dữ liệu 100% thật.

## 5. RECONSTRUCTION STATUS

**Đã xác nhận bằng đọc code thật (không suy đoán):** pipeline là 3DMM-style multi-view identity fit thật (không phải pattern giả bị cấm). Xem mục H ở trên để có bằng chứng file/dòng cụ thể.

**CHƯA xác nhận:** chưa chạy được một reconstruction thật end-to-end với dữ liệu do scanner MỚI (7-checkpoint) tạo ra, vì chưa có phiên quét thật trên thiết bị thật (xem mục 4). Dữ liệu burst cũ nhất hiện có (bệnh nhân 8e0dc763, patient thử nghiệm trước đó) được tạo bởi scanner CŨ, không có đủ metadata mới để test full pipeline mới end-to-end một cách trung thực.

## 6. BLOCKER THỰC TẾ

**Không có blocker kỹ thuật chặn tiến độ.** Blocker duy nhất là môi trường: sandbox này không có camera vật lý + khuôn mặt người thật, nên không thể tự mình tạo dữ liệu quét thật để chạy reconstruction thật end-to-end. Đây không phải lý do để dừng — mọi phần CÓ THỂ tự kiểm tra (code, build, unit test, E2E với fake camera, đọc reconstruction pipeline) đã được tự làm đến hết khả năng.

**Phát hiện phụ (không phải blocker, chỉ cần biết):** một header mobile toàn cục (ngoài GuidedFaceScan) đè z-index lên trên overlay quét mặt ở một số kích thước màn hình, khiến click tự động (E2E) cần `force:true`. Chưa sửa vì đây là vấn đề layout app-shell riêng, ngoài phạm vi "scanner" được giao lần này.

## 7. BƯỚC TIẾP THEO DUY NHẤT

**Test thật trên điện thoại thật (iPhone hoặc Samsung) với khuôn mặt thật:**
1. Quét đủ 7 checkpoint (0/±20/±45/±60°) bằng camera trước.
2. Xác nhận yaw đo được KHỚP với hướng quay thật (trái thật → số âm, phải thật → số dương) — đây là điều chỉnh quan trọng nhất từ phiên trước, cần xác nhận bằng mắt thật.
3. Lặp lại với camera sau.
4. Để scan chạy tới `finalizeScan()` thật, xác nhận endpoint `/profile-preview` trả về đúng 4 ảnh và `request_reconstruction` tạo ra `baseline.glb` thật.
5. Đọc `reconstruction_report.json`/`coverage_stats.json` thật của phiên quét đó, so với các threshold trong `render_back_validator.py`, để biết pipeline reconstruction đã audit ở mục H hoạt động tốt đến đâu với dữ liệu do scanner MỚI tạo ra.

Không có bước nào khác cần làm trước bước này — mọi thứ tự-kiểm-tra-được đã làm xong trong phiên này.
