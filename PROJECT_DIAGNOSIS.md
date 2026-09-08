# PROJECT_DIAGNOSIS.md — Phân tích kiến trúc pipeline dựng mặt 3D (GNM)

> Quét toàn bộ `/home/ubuntu/dr-vantruong-3d-studio` ngày 2026-08-27. Tài liệu này đọc trực tiếp mã nguồn hiện tại (không dựa vào giả định) — mọi kết luận đều trích dẫn `file:line` cụ thể. Một số root-cause đã được TEAM TRƯỚC đo đạc và ghi lại ngay trong docstring của code (đánh dấu **[ĐÃ ĐO ĐẠC]**); phần còn lại là phân tích kỹ thuật của lần quét này (đánh dấu **[PHÂN TÍCH MỚI]**).

## 0. Bức tranh tổng thể

```
Ảnh điện thoại (4 góc: 0°/45°/90°/dưới cằm)
        │
        ▼
detect_pose.py  — YuNet (bbox) + PIPNet-98 (landmark) + SolvePnP/EPNP (pose)
        │
        ▼
gnm_identity_fit.py :: fit_multiview()  — RIDGE LEAST-SQUARES (không phải Adam/L-BFGS/Gauss-Newton)
        │
        ▼
gnm_dense_nose.py (Phase 8) + gnm_width_correction.py  — vá lỗi "méo hình" của bước trên
        │
        ▼
gnm_vertex_trust.py :: blend_positions_by_trust()  — trộn với template ở vùng ít bằng chứng ảnh
        │
        ├──► gnm_texture_bake.py / gnm_texture_atlas.py  — vertex-color + atlas 19 vùng (ICM, KHÔNG blend)
        │
        └──► gnm_face_shell.py :: bake_unified_face_texture()  — texture HD 2048² hợp nhất (multiband CÓ blend)
                     │
                     ▼
             gnm_pyrender_occlusion.py — z-buffer thật (pyrender/OpenGL) chống occlusion sai
```

Hai pipeline texture chạy **song song và độc lập**:
- **Pipeline chính (backend, Python)**: `gnm_face_shell.py` bake ra `face_HD.png` — đây là pipeline **đang chạy trong production** (`/reconstruct` endpoint, [ai-engine/main.py:608-646](ai-engine/main.py#L608-L646)).
- **Pipeline fallback (frontend, TypeScript)**: `src/lib/gnm/texture-blend.ts` — chỉ chạy khi `faceShellTexture` load lỗi, xem mục 3.4.

---

## 1. `gnm_identity_fit.py` + `main.py` — Thuật toán tối ưu & tính nhất quán (Random Seed)

### 1.1 Thuật toán KHÔNG phải Adam / L-BFGS / Gauss-Newton

Cả 3 bước tính toán chính trong `fit_multiview()` ([ai-engine/gnm_identity_fit.py:159-207](ai-engine/gnm_identity_fit.py#L159-L207)) đều là **nghiệm dạng đóng (closed-form)**, không có vòng lặp gradient descent nào:

| Bước | Hàm | Thuật toán thực tế |
|---|---|---|
| Ước lượng pose camera từ landmark 2D | `solve_epnp()` [L107-113](ai-engine/gnm_identity_fit.py#L107-L113) | `cv2.solvePnP(..., flags=cv2.SOLVEPNP_EPNP)` — **EPnP** (Efficient PnP), một thuật toán tuyến tính hoá kín (giải hệ tuyến tính nhỏ qua SVD), **không phải Gauss-Newton/Levenberg-Marquardt** (đó là cờ `SOLVEPNP_ITERATIVE`, không dùng ở đây) |
| Căn chỉnh rigid landmark quan sát vào template | `solve_procrustes()` [L78-90](ai-engine/gnm_identity_fit.py#L78-L90) | **Kabsch algorithm** qua SVD (`np.linalg.svd`) — nghiệm đóng, không lặp |
| Giải hệ số nhận dạng (identity coefficients) | `solve_identity_multiview()` [L116-141](ai-engine/gnm_identity_fit.py#L116-L141) | **Ridge-regularized linear least squares** (hệ phương trình chuẩn tắc): <br>`M = ΣBᵀB + λI`, `rhs = ΣBᵀr`, giải `np.linalg.solve(M, rhs)` — tương đương hồi quy Tikhonov/ridge, λ = `DEFAULT_REGULARIZATION = 1e-5` [L39](ai-engine/gnm_identity_fit.py#L39) |

**Kết luận**: Đây thực chất là bài toán **fit tuyến tính có regularization** (203 hệ số PCA identity — basis shape `(253, 17821, 3)` [L57](ai-engine/gnm_identity_fit.py#L57), không phải một mạng neural hay bài toán tối ưu phi tuyến cần Adam/L-BFGS. Việc giải bằng `np.linalg.solve` trên ma trận đối xứng dương (M luôn khả nghịch nhờ số hạng `λI`) là chính xác tuyệt đối trong một bước, không cần nhiều epoch/iteration.

### 1.2 Random Seed: KHÔNG tìm thấy random chưa khóa trong đường suy luận (inference)

Đã grep toàn bộ `ai-engine/*.py` (loại trừ `venv/`, `TripoSR/`, `models/pipnet/lib/*` — code training PIPNet gốc) tìm `random|seed|RANSAC`:

- **Không có lệnh gọi `np.random`/`random.*`/`torch.manual_seed` nào trong đường chạy thực tế** (`fit_multiview`, `main.py`'s `/reconstruct`, `/fit-multiview-atlas`, `detect_pose`).
- Từ khóa `seed` duy nhất tìm thấy là biến cục bộ trong `gnm_face_shell.py:111` (`get_face_shell_topology`) — đây là **điểm bắt đầu BFS để unwrap UV** (`seed = int(remaining_arr[np.argmax(radius[...])])`), chọn **tất định** bằng `argmax`, không phải giá trị ngẫu nhiên.
- `cv2.solvePnP` với cờ `EPNP` **không dùng RANSAC** (không có outlier rejection ngẫu nhiên) — mọi điểm landmark được dùng trực tiếp, deterministic.
- `PIPNet` (`detect_pose.py`) chạy đúng `.eval()` [L93](ai-engine/detect_pose.py#L93) + `torch.no_grad()` [L140](ai-engine/detect_pose.py#L140) → không có Dropout/augmentation ngẫu nhiên lúc inference. Các hàm `random_translate/random_blur/random_occlusion/random_flip/random_rotate` trong `models/pipnet/lib/data_utils.py` **chỉ dùng khi TRAIN model PIPNet** (không được `import` bởi `detect_pose.py`), không ảnh hưởng runtime.

**→ Trả lời câu hỏi**: Pipeline nhận dạng hiện tại **không có cơ chế random chưa khóa seed** gây non-determinism. Nếu 2 lần chạy cho kết quả khác nhau với **cùng 1 bộ ảnh đầu vào**, nguyên nhân nhiều khả năng nằm ở:
- Sai khác nhỏ do đa luồng BLAS/OpenMP trong SVD/`np.linalg.solve` (thường chỉ lệch ở bậc 1e-10, không đủ gây khác biệt nhìn thấy được bằng mắt);
- Hoặc **input ảnh thực sự khác nhau** giữa các lần chụp (góc chụp, ánh sáng) — nguyên nhân phổ biến hơn nhiều so với thuật toán.

Một điểm cần lưu ý: dòng `d = np.sign(np.linalg.det(Vt.T @ U.T)); D = np.diag([1,1, d if d != 0 else 1])` ở `solve_procrustes()` [L84-85](ai-engine/gnm_identity_fit.py#L84-L85) xử lý trường hợp suy biến (determinant = 0, ma trận phản chiếu) bằng cách mặc định `d=1` — đây KHÔNG phải nguồn random, nhưng là một **rẽ nhánh im lặng** (silent branch) có thể che giấu một pose bất thường nếu nó xảy ra thường xuyên hơn dự kiến; đáng để log cảnh báo nếu `d == 0` thực sự xảy ra trên dữ liệu thật.

---

## 2. `gnm_face_shell.py` — Camera, Focal Length & Nguyên nhân khả dĩ gây "mặt bẹt"

### 2.1 Camera model: PERSPECTIVE (phối cảnh thật), không phải Orthographic

Toàn bộ pipeline — từ `solve_epnp` (nhận dạng), `gnm_pyrender_occlusion.py` (z-buffer), đến `bake_unified_face_texture` (chiếu texture) — đều dùng **camera pinhole phối cảnh chuẩn (perspective projection)**:

- Ma trận nội tại (intrinsics) `K = [[fx,0,cx],[0,fy,cy],[0,0,1]]` xuất hiện nhất quán ở **7 nơi** trong `ai-engine/` ([main.py:133,172,249,300](ai-engine/main.py#L133), [detect_pose.py:203](ai-engine/detect_pose.py#L203), `regression/capture_*.py`).
- Phép chiếu điểm 3D → 2D trong `bake_unified_face_texture` [L331-336](ai-engine/gnm_face_shell.py#L331-L336) dùng công thức phối cảnh chuẩn `px = fx*X/Z + cx` (chia cho Z — đúng là perspective, không phải orthographic `px = X + cx` không chia Z).
- `gnm_pyrender_occlusion.py` dùng `pyrender.IntrinsicsCamera(fx, fy, cx, cy, znear, zfar)` [L93](ai-engine/gnm_pyrender_occlusion.py#L93) — đây **chắc chắn là camera phối cảnh thật của OpenGL**, không phải `pyrender.OrthographicCamera`.

→ **Không có việc nhầm lẫn Orthographic ở đâu cả.** Camera đúng là Perspective xuyên suốt.

### 2.2 Focal length: XẤP XỈ, không tính từ EXIF/hiệu chỉnh thật — đây là chỗ đáng ngờ nhất

Tiêu cự (`focal length` tính bằng pixel) **không được đo từ metadata ảnh thật** (EXIF) hay hiệu chỉnh camera (calibration), mà được **xấp xỉ bằng chiều rộng ảnh**:

```python
# ai-engine/detect_pose.py:203
focal = float(w)  # same approximation validated in scratchpad — no real EXIF/calibration data exists
camera_matrix = np.array([[focal, 0, w/2], [0, focal, h/2], [0, 0, 1]])
```
Comment tại chỗ **tự thừa nhận** đây là xấp xỉ ("no real EXIF/calibration data exists"), không phải giá trị đo thật. Công thức tương đương lặp lại y hệt ở `main.py` (4 chỗ) và `gnm_identity_fit.py`'s `fit_multiview` nhận `camera_matrix` này làm input trực tiếp.

**[PHÂN TÍCH MỚI] — Vì sao xấp xỉ này CÓ THỂ gây cảm giác "mặt bẹt":**

`focal = w` tương ứng góc nhìn ngang (FOV) = `2·atan(w / (2·f)) = 2·atan(0.5) ≈ 53°`. Camera sau của điện thoại thật (main lens) thường có FOV ngang thực tế **65°-80°** (ống kính góc rộng hơn giả định 53° khá nhiều). Hệ quả:

1. Trong `backproject_pseudo3d()` [ai-engine/gnm_identity_fit.py:97-104](ai-engine/gnm_identity_fit.py#L97-L104), độ lệch ngang/dọc của điểm landmark quan sát được tính là `(u - cx) / fx * Zc`. Nếu `fx` bị ước lượng **quá lớn** so với thực tế (do giả định FOV hẹp hơn thật), thì độ lệch X/Y suy ra từ ảnh 2D bị **thu nhỏ lại một cách hệ thống** so với độ lệch thật — trong khi Z (độ sâu) vẫn giữ nguyên theo template. Kết quả: offset dùng để fit identity coefficients bị **lệch tỷ lệ giữa các trục X/Y so với Z**, có xu hướng đẩy nghiệm ridge-regression về gần template (phẳng hơn) thay vì phản ánh đúng độ nhô/lõm thật của khuôn mặt.
2. Vì `solve_procrustes()` chạy NGAY SAU bước back-project và tự chuẩn hoá lại scale toàn cục, sai số về **scale tổng thể** được triệt tiêu phần lớn — nhưng sai số về **hình dạng phi đều (anisotropic)** giữa các view (0°/45°/90° có Z rất khác nhau) thì KHÔNG được Procrustes sửa, vì Procrustes chỉ là phép biến đổi rigid (không co giãn khác nhau theo từng trục).

**[ĐÃ ĐO ĐẠC — bằng chứng độc lập, mạnh hơn giả thuyết focal length]**: Nguyên nhân gốc rễ về "mặt phẳng/thiếu chiều sâu" thực ra **đã được chính team trước xác nhận bằng đo đạc thật**, ghi rõ trong docstring của `gnm_dense_nose.py` [L9-11](ai-engine/gnm_dense_nose.py#L9-L11):

> *"8A: root cause -- gnm_correspondence.py represents the ENTIRE nose with exactly 1 real point (WFLW 57, weighted 10x). A single point cannot constrain bridge width/height/tip projection, only tip position."*

Tức là: bộ 47-48 điểm landmark WFLW dùng để fit identity chỉ có **1 điểm duy nhất** đại diện cho cả sống mũi — hoàn toàn không đủ ràng buộc để tái tạo độ nhô 3D thật (chỉ khoá được vị trí đầu mũi, không khoá được sống mũi cao hay thấp). Đây chính là lý do khuôn mặt fit ra có xu hướng "phẳng" ở vùng mũi/gò má — không phải do camera hay rasterizer, mà do **thiếu ràng buộc hình học (landmark sparsity)** trong chính bước identity-fit tuyến tính. Module `gnm_dense_nose.py` (Phase 8) được viết ra CHÍNH ĐỂ vá lỗi này bằng cách bơm thêm ~828 điểm bề mặt mũi dày đặc từ 3DDFA_V2/BFM vào ràng buộc.

Tương tự, `gnm_width_correction.py` [L1-11](ai-engine/gnm_width_correction.py#L1-L11) ghi nhận một lỗi anh em: khuôn mặt fit ra **"quá bè" (too wide)** ~5-7% so với ảnh thật, cũng do "phần diện tích giữa các landmark" (má, thái dương) không được ràng buộc trực tiếp bởi bất kỳ điểm landmark nào — cùng một nguyên nhân gốc: **PCA identity basis 253 chiều + landmark thưa không đủ để khoá hình dạng bề mặt ngoài landmark**.

**Tóm lại về "mặt bẹt"**:
- **Nguyên nhân chính, đã đo đạc**: landmark mũi chỉ có 1 điểm → thiếu ràng buộc chiều sâu → đã được vá một phần bởi `gnm_dense_nose.py`, nhưng vá theo kiểu "cộng thêm", không sửa gốc rễ ở `gnm_identity_fit.py`.
- **Nguyên nhân phụ, chưa được đo đạc/xác nhận trong code**: xấp xỉ `focal = w` không phản ánh đúng ống kính điện thoại thật, có thể khuếch đại thêm sai số hình dạng phi đều giữa các góc chụp. **Khuyến nghị**: nếu ứng dụng có thể đọc EXIF (`FocalLengthIn35mmFilm` hoặc thông số cảm biến) từ ảnh gốc trước khi nén/resize, thay `focal = float(w)` bằng giá trị tính từ EXIF sẽ loại trừ hẳn nghi vấn này mà không cần sửa thuật toán fit.

### 2.3 `gnm_face_shell.py` tự nó không phải nơi tính hình dạng 3D

Cần làm rõ: **`gnm_face_shell.py` không tham gia vào việc dựng hình dạng 3D** (đó là việc của `gnm_identity_fit.py`/`gnm_dense_nose.py`/`gnm_width_correction.py`). File này chỉ nhận `fitted_positions` đã dựng xong làm input, rồi (a) cắt ra tập con ~10,390 vertex thuộc "shell" (mặt+tai+cổ) và (b) unwrap UV + bake texture. Vì vậy nếu khuôn mặt "bẹt" khi xem trên `Canvas3D`, gốc rễ hình học nằm ở mục 2.2 (identity fit), còn `gnm_face_shell.py` chỉ có thể làm cho vấn đề **trông rõ hơn hoặc mờ hơn** qua texture/shading, không tạo ra nó.

---

## 3. `gnm_pyrender_occlusion.py` + hệ thống blend texture — Nguyên nhân "loang lổ, nhòe"

### 3.1 Occlusion: đã có z-buffer thật, nhưng đây là bản vá lần 3

Docstring của `gnm_pyrender_occlusion.py` [L1-38](ai-engine/gnm_pyrender_occlusion.py#L1-L38) ghi lại lịch sử đo đạc rất rõ:

1. **Bản đầu tiên**: `bake_unified_face_texture` hoàn toàn KHÔNG có occlusion test, chỉ dùng `facing_weight` (góc pháp tuyến so với camera) → không phân biệt được "mặt hướng về camera" với "mặt hướng về camera nhưng bị cằm/tai/má xa che khuất" → gây ra **mảng texture cổ/cằm bị đặt sai vị trí** và **viền profile bị nhân đôi**.
2. **Bản vá lần 2**: tái sử dụng z-buffer barycentric thủ công có sẵn (`gnm_texture_bake.py`) nhưng chạy trên TOÀN BỘ 17,821 vertex (bao gồm cả răng, lưỡi, mắt trong — hình học nội thất không bao giờ xuất hiện trong ảnh thật) → **độ hiển thị của môi sụp xuống ~0%** vì các vertex nội thất chiếu gần camera hơn bề mặt môi thật trong buffer thủ công đó.
3. **Bản vá lần 3 (hiện tại)**: giới hạn z-buffer chỉ trên tập "shell" (không có hình học nội thất) → môi hiển thị 87.6-97.6%, NHƯNG lộ ra lỗi mới: **sọc z-fighting tự va chạm ở vùng miệng/cằm**, vì phép nội suy barycentric thủ công nội suy Z trực tiếp trong không gian màn hình — **không "perspective-correct"** (đúng ra phải nội suy 1/Z).
4. **Bản hiện tại (`gnm_pyrender_occlusion.py`)**: thay hẳn bằng `pyrender` (OpenGL software rasterizer qua Mesa llvmpipe/EGL) — rasterizer GPU-chuẩn, tự động perspective-correct, đã kiểm chứng với ground-truth thủ công (tường phẳng + vật che biết trước, `scratchpad/spike_pyrender_occlusion_unit.py`).

→ Về mặt thuật toán, occlusion **hiện đã đúng** (perspective-correct, đã kiểm chứng). Rủi ro còn lại nằm ở 2 chỗ kỹ thuật:
- `epsilon = 0.003m` (3mm dung sai, [L56](ai-engine/gnm_pyrender_occlusion.py#L56)) là **hằng số cố định, tự chọn tay** ("re-measured on its own merits" — không có phép sweep/A-B test nào được ghi lại như các module khác). Nếu dung sai này quá nhỏ ở vùng cong gấp (rãnh mũi-má, viền tai), một số texel có thể bị loại nhầm (false-reject) → lộ ra vùng "trống" phải nội suy từ ảnh khác → góp phần vào cảm giác loang lổ.
- Comment L99-106 xác nhận `RenderFlags.DEPTH_ONLY` **bị lỗi trên chính backend llvmpipe/EGL đang dùng** (trả về buffer toàn số 0) — code phải né bằng cách render cả màu lẫn depth rồi bỏ màu đi. Đây là một **dependency lỗi đã biết, chưa được vá ở nguồn** (không phải bug do code project viết sai, mà do tổ hợp thư viện) — nếu nâng cấp `pyrender`/Mesa sau này, cần re-test lại giả định này.

### 3.2 Cơ chế "phân tách vùng mặt" ĐÃ ĐƯỢC ĐO ĐẠC là nguồn gốc loang lổ — và đã trải qua 2 lần sửa

Đây là phần quan trọng nhất trả lời câu hỏi "loang lổ, nhòe" — lịch sử ghi trong `gnm_texture_bake.py` [L240-390](ai-engine/gnm_texture_bake.py#L240-L390):

- **Thiết kế gốc**: mỗi vùng giải phẫu (`REGION_SPEC`: trán, má, mũi, cằm, gò má, thái dương, quanh mắt...) **tự chọn độc lập** ảnh nguồn tốt nhất cho riêng nó, dựa trên facing-weight.
- **Lỗi đo được (comment "D-continuity", [L251-287](ai-engine/gnm_texture_bake.py#L251-L287))**: ranh giới các vùng này là ranh giới **giải phẫu**, không phải ranh giới **topology lưới**. Hai vertex kề nhau trên cùng 1 tam giác có thể rơi vào 2 group khác nhau, mỗi group chọn nguồn ảnh khác nhau một cách độc lập → tạo **đường biên cứng** ngay giữa trán/mắt/mũi/má, đo được trên bệnh nhân thật (`0e9e1d90-...`): **5 mảng rời rạc cùng nguồn**, một bên mắt rơi vào group chọn nhầm ảnh thiếu sáng → **mảng trắng loang trên mắt**.
- **Bản vá (ICM — Iterated Conditional Modes)**: bỏ hẳn cách nhóm theo giải phẫu, giải bài toán gán nhãn (mỗi vertex chọn 1 ảnh nguồn DUY NHẤT, không blend) trên đúng đồ thị kề của lưới (`_select_all_icm`, [gnm_texture_bake.py:292-369](ai-engine/gnm_texture_bake.py#L292-L369)) — mô hình Potts: chi phí = `(1 - facing_weight)` của ảnh được chọn + `SWITCH_PENALTY=0.6` cho mỗi hàng xóm lưới bất đồng nhãn, lặp tối đa 25 vòng đến khi hội tụ.
- **Đo đạc sau vá**: số cạnh lưới là "đường biên nguồn" giảm 5.4% (3364→3182), số mảng liền-cùng-nguồn giảm 35% (184→119, tức các mảng lớn hơn, ít vụn hơn), lỗi "mắt trắng" biến mất trên render offline thật.
- **Lỗi vòng 2 ("D-continuity2", [L372-390](ai-engine/gnm_texture_bake.py#L372-L390))**: bản vá ICM ở trên chỉ áp dụng cho **vertex-color layer** (lớp màu cơ bản), còn **lớp atlas 19 vùng** (`build_multi_region_atlas`, phủ ~95% khuôn mặt) vẫn dùng cách chọn độc lập theo từng vùng riêng → **vẫn còn đường biên cứng y hệt**, đo lại được trên CẢ 2 bệnh nhân thật (`0e9e1d90` và `0913e4c9`) → xác nhận đây là lỗi **kiến trúc chung**, không phải lỗi 1 lần. Vá bằng cách chia sẻ CÙNG một kết quả ICM cho cả 2 lớp (`compute_icm_labels`, dùng chung ở [main.py:310-316](ai-engine/main.py#L310-L316)).

**→ Kết luận cho lớp vertex-color/atlas (KHÔNG phải `face_HD.png`)**: cơ chế chọn ảnh nguồn theo ICM **không blend/trộn màu** — mỗi vertex lấy đúng 1 pixel gốc từ 1 ảnh. Điều này về lý thuyết loại trừ hiện tượng "nhòe" (không có phép trộn nào cả), nhưng KHÔNG loại trừ hiện tượng "loang lổ dạng khối" nếu `SWITCH_PENALTY=0.6` (hằng số cố định, không phải kết quả tối ưu qua sweep) đặt sai — quá thấp thì trở lại vụn mảnh như bản gốc, quá cao thì cưỡng ép cả vùng lớn dùng chung 1 ảnh dù ảnh đó không phải lựa chọn tốt nhất cục bộ.

### 3.3 Lớp texture HD hợp nhất (`gnm_face_shell.py`) dùng blend LIÊN TỤC — khác hẳn ICM

Ngược lại với mục 3.2, `bake_unified_face_texture()` [ai-engine/gnm_face_shell.py:213-485](ai-engine/gnm_face_shell.py#L213-L485) — đây là texture **đang chạy chính thức trong `/reconstruct`** — dùng chiến lược khác hẳn: **trộn liên tục (không rời rạc)** qua 2 tầng:

1. Trọng số mỗi view = `facing_weight × valid × smoothstep-ramp theo vùng` ([L399-421](ai-engine/gnm_face_shell.py#L399-L421)) — thay ranh giới cứng bằng dải chuyển tiếp mượt ~6-8mm (chính comment ghi nhận: ranh giới cứng cũ "produced the visible seam LINE" — cùng họ lỗi với mục 3.2 nhưng được vá bằng cách khác: làm mượt thay vì làm liền-khối).
2. **Multi-band Laplacian pyramid blending** (`multiband_blend_views`, [L151-210](ai-engine/gnm_face_shell.py#L151-L210)) — trộn theo từng dải tần số ảnh: **chi tiết tần cao (lỗ chân lông, chân mày) dùng `argmax` (winner-take-all)** để tránh nhòe/ghost ([L179-194](ai-engine/gnm_face_shell.py#L179-L194) — comment tự giải thích: trộn trung bình 2 ảnh phơi sáng độc lập ở tần cao tạo "double-exposure/ghost"), còn **tần thấp (màu da tổng thể) dùng trung bình có trọng số** ([L197-203](ai-engine/gnm_face_shell.py#L197-L203)).

**[PHÂN TÍCH MỚI] Đây chính là nơi "nhòe" thực sự có thể xảy ra** (khác với lớp ICM ở 3.2 vốn không blend): mặc dù thiết kế đã cố tránh nhòe ở tần cao bằng `argmax`, **tầng gốc (base band, sau khi hạ độ phân giải `levels=4` lần bằng `pyrDown`) vẫn dùng trung bình trọng số** — đây là dải màu/độ sáng tổng thể, ít gây nhòe chi tiết nhưng CÓ THỂ gây **loang màu/tông da không đều** nếu 2 ảnh nguồn có tông màu/ánh sáng lệch nhau nhiều (không có bước color-correction/white-balance giữa các ảnh trước khi blend — code không có bước nào chuẩn hoá màu giữa 4 ảnh chụp).

Ngoài ra, `BOUNDARY_BLUR_KSIZE = 31` ([L460](ai-engine/gnm_face_shell.py#L460)) áp Gaussian blur bán kính 31px lên vùng biên tai/cổ (loại trừ vùng "sharp core" mặt chính, [L467](ai-engine/gnm_face_shell.py#L467)) — đây là nguồn **nhòe có chủ đích, đã biết trước**, không phải bug, chỉ ảnh hưởng vùng rìa thấp-trust (tai, cổ), không phải vùng mặt chính.

### 3.4 `texture-blend.ts` — KHÔNG phải nguồn gốc "loang lổ" chính, vì đây là fallback phụ

Câu hỏi đề cập `texture-blend.ts` như cơ chế đang gây lỗi, nhưng quét thực tế cho thấy: **đây không phải pipeline chính**. Bằng chứng:

- `computeFeatherBlendColors()` chỉ được gọi ở đúng 1 nơi: [src/components/Canvas3D.tsx:1211](src/components/Canvas3D.tsx#L1211), bên trong một IIFE async có comment tự mô tả là **"Giai đoạn C — best-effort ENHANCEMENT"** chạy sau khi mesh chính đã hiển thị xong, không block luồng chính.
- Việc lớp này có được RENDER hay không phụ thuộc điều kiện tại [Canvas3D.tsx:1959](src/components/Canvas3D.tsx#L1959): `multiViewFitState.faceShellGeometry && multiViewFitState.faceShellTexture ? (dùng face_HD.png) : (dùng GnmHeadMesh với feather-blend)`. Tức là **`texture-blend.ts` chỉ chạy khi bake `face_HD.png` ở backend THẤT BẠI hoặc không tải được** — nó là con đường dự phòng (fallback), không phải con đường sản xuất chính.
- Thuật toán của nó ([texture-blend.ts:39-84](src/lib/gnm/texture-blend.ts#L39-L84)) chọn **1 màu tốt nhất mỗi vertex** (không phải blend đa nguồn thật sự — biến `bestWeight/bestR/bestG/bestB` là kết quả của vòng lặp lấy max, không phải trung bình), rồi dùng alpha đó để MIX (không nhân) với texture nền qua shader patch `featherBlendOnBeforeCompile` [L125-134](src/lib/gnm/texture-blend.ts#L125-L134). Về bản chất thuật toán, cơ chế này **ít có khả năng gây nhòe hơn** phương pháp Laplacian pyramid ở mục 3.3, vì mỗi vertex chỉ lấy đúng 1 pixel nguồn.

**→ Nếu người dùng đang thấy ảnh loang lổ/nhòe trên môi trường thực tế**, khả năng cao nhất là đang xem **layer `face_HD.png`** (mục 3.3, pipeline chính) hoặc **layer atlas/vertex-color** (mục 3.2, trước khi ICM hợp nhất được áp dụng đầy đủ), KHÔNG PHẢI `texture-blend.ts` — trừ khi log console thực tế cho thấy `[GNM] Face Shell texture decode failed` (Canvas3D.tsx:768), nghĩa là đang rơi vào nhánh fallback này.

---

## 4. TripoSR — Trạng thái tích hợp: HOÀN TOÀN LƠ LỬNG, chưa nối vào pipeline chính

Kết luận rõ ràng sau khi quét toàn bộ repo:

- **Không có file nào ngoài `ai-engine/TripoSR/` tham chiếu đến TripoSR** — đã grep `TripoSR|triposr` trên toàn bộ `.py/.ts/.tsx` của project, không có kết quả nào bên ngoài chính thư mục đó.
- `main.py` (server FastAPI chính, cổng 8001) **không `import` bất kỳ thứ gì từ `TripoSR/`**.
- `dev-all.sh` ([dev-all.sh:1-40](dev-all.sh)) — script khởi động dev chính thức của project — chỉ khởi động `ai-engine/main.py` (FastAPI) và `npm run dev` (Next.js). **Không có dòng nào khởi động hay gọi TripoSR.**
- Bản thân `TripoSR/run_face.py` ([ai-engine/TripoSR/run_face.py:1-25](ai-engine/TripoSR/run_face.py)) là một **script CLI độc lập, chạy tay**: đọc 1 ảnh tĩnh cố định `face.jpg` trong thư mục hiện hành, xuất ra `output/face.glb` — không nhận input động từ pipeline bệnh nhân (không có `patient_id`, không đọc từ `.data/patients/`, không ghi vào `public/models/patients/`).
- Thời điểm file: toàn bộ `ai-engine/TripoSR/` được thêm vào lúc **26/8/2026, 07:21-07:43** — tức là bản thử nghiệm **rất mới, chỉ vài giờ trước thời điểm quét này**, gần như chắc chắn đang trong giai đoạn *"chạy thử xem TripoSR dựng mặt từ 1 ảnh ra sao"* độc lập với kiến trúc GNM multi-view hiện tại (vốn dùng 4 ảnh + landmark + PCA identity basis, khác hẳn cách tiếp cận image-to-3D của TripoSR).
- Output test đã có: `ai-engine/TripoSR/output/test_cpu.glb` tồn tại (sinh lúc 07:39), xác nhận script **chạy được** trên máy này (CPU-only, không GPU), nhưng chỉ là file output rời rạc, chưa có cầu nối nào đưa `.glb` đó vào `public/models/patients/<id>/...` hay vào `Canvas3D.tsx`.

**→ Trả lời trực tiếp câu hỏi**: TripoSR **chưa được liên kết vào pipeline chính dưới bất kỳ hình thức nào** — không có API endpoint, không có import, không được khởi động cùng hệ thống, không ghi output vào đúng chỗ frontend đọc. Đây đúng nghĩa là **một nhánh thử nghiệm treo lơ lửng (dangling spike)**, hợp lý nếu đang trong giai đoạn đánh giá có nên thay thế/bổ sung cho pipeline GNM multi-view hiện tại hay không.

---

## 5. Tổng hợp khuyến nghị (theo mức ưu tiên)

1. **Nếu nghi ngờ "mặt bẹt" là vấn đề ưu tiên cao nhất**: kiểm tra trước tiên xem `gnm_dense_nose.py`/`gnm_width_correction.py` có đang chạy thành công hay đang fail-silent (cả 2 đều "best-effort, never blocks" — nếu 3DDFA_V2 lỗi, dense-nose sẽ tự bỏ qua mà không báo lỗi rõ ràng cho người dùng cuối, chỉ thêm vào mảng `warnings` mà UI có thể không hiển thị). Xem `result["warnings"]` thực tế trả về từ `/reconstruct` cho ca bệnh nhân đang bị phàn nàn trước khi sửa thuật toán.
2. **Cân nhắc thay `focal = float(w)`** ([detect_pose.py:203](ai-engine/detect_pose.py#L203)) bằng giá trị đọc từ EXIF ảnh gốc nếu có — chi phí thấp, loại trừ hẳn một biến số nghi vấn.
3. **`SWITCH_PENALTY=0.6`** và **`OCCLUSION_EPSILON_M=0.003`** đều là hằng số chọn tay, không có sweep/A-B ghi lại như các hằng số khác trong project (ví dụ W=75 ở dense-nose có hẳn 1 bảng sweep 15→200) — nếu muốn giảm loang lổ, đây là 2 tham số đáng thử nghiệm có hệ thống trước.
4. **`texture-blend.ts` không phải nguyên nhân chính** — trước khi sửa nó, xác nhận qua console log xem hệ thống có đang thực sự rơi vào nhánh fallback này không.
5. **TripoSR** cần một quyết định rõ ràng: hoặc dọn dẹp (nếu chỉ là thử nghiệm đã xong việc), hoặc lên kế hoạch tích hợp chính thức (endpoint riêng + ghi đúng chỗ `public/models/patients/<id>/`) nếu muốn giữ lại.
