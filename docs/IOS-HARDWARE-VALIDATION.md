# HƯỚNG DẪN KIỂM CHỨNG PHẦN CỨNG NATIVE iOS TRUEDEPTH (HARDWARE VALIDATION RUNBOOK)

Tài liệu này hướng dẫn chi tiết quy trình chuẩn bị, build và chạy ứng dụng quét 3D native `DrVanTruongScanner` trên thiết bị iPhone / iPad Pro thật để thực hiện xác thực lâm sàng ngoài đời thực.

---

### 1. MAC REQUIREMENTS (YÊU CẦU MÁY MAC)
* **Hệ điều hành:** macOS Sonoma (14.0 trở lên) hoặc macOS Sequoia (15.0 trở lên).
* **Cổng kết nối:** Cáp USB-C / Lightning chuẩn để kết nối iPhone/iPad với Mac.
* **Bộ nhớ:** Tối thiểu 16GB RAM để xử lý mượt mà tác vụ build ARKit/SceneKit.

---

### 2. XCODE VERSION & TOOLS
* **Xcode:** Xcode 15.0 trở lên (khuyên dùng Xcode 15.4 hoặc Xcode 16.0).
* **Command Line Tools:** Cài đặt qua lệnh `xcode-select --install`.
* **Swift Version:** Swift 5.9+.

---

### 3. APPLE DEVELOPER SIGNING (CHỨNG CHỈ NHÀ PHÁT TRIỂN)
1. Mở Xcode $\rightarrow$ `Settings` (hoặc `Preferences`) $\rightarrow$ `Accounts`.
2. Đăng nhập Apple ID (tài khoản cá nhân Free Developer Account hoặc tài khoản Thẩm mỹ viện Dr. Văn Trương).
3. Mở project:
   ```bash
   open ios/DrVanTruongScanner.xcodeproj
   ```
4. Trong mục **Target `DrVanTruongScanner`** $\rightarrow$ tab **Signing & Capabilities**:
   * Tích chọn **Automatically manage signing**.
   * Chọn **Team** tương ứng với tài khoản Apple của bạn.
   * Bundle Identifier: `com.drvantruong.scanner` (có thể đổi tiền tố tổ chức nếu cần).

---

### 4. SUPPORTED DEVICE (THIẾT BỊ PHẦN CỨNG HỖ TRỢ)
* **Bắt buộc:** Thiết bị phải có camera trước **TrueDepth** (Đèn chiếu hồng ngoại cấu trúc điểm).
* **Danh sách tương thích:**
  * iPhone X, XR, XS, XS Max.
  * iPhone 11, 11 Pro, 11 Pro Max.
  * iPhone 12, 12 mini, 12 Pro, 12 Pro Max.
  * iPhone 13, 13 mini, 13 Pro, 13 Pro Max.
  * iPhone 14, 14 Plus, 14 Pro, 14 Pro Max.
  * iPhone 15, 15 Plus, 15 Pro, 15 Pro Max.
  * iPhone 16, 16 Plus, 16 Pro, 16 Pro Max.
  * iPad Pro 11-inch (Gen 1, 2, 3, 4, M4) & iPad Pro 12.9-inch (Gen 3, 4, 5, 6).
* *Lưu ý:* iPhone SE (các đời) không có camera TrueDepth nên không thể kích hoạt `ARFaceTrackingConfiguration`.

---

### 5. CAMERA PERMISSIONS & PRIVACY
* Ứng dụng đã khai báo trong `ios/Resources/Info.plist`:
  * `NSCameraUsageDescription`: *"Ứng dụng cần quyền truy cập Camera TrueDepth để quét cấu trúc 3D khuôn mặt và điểm mốc giải phẫu phục vụ tư vấn thẩm mỹ."*
  * `UIRequiredDeviceCapabilities`: `arm64`, `arkit`.
* Khi mở app lần đầu trên iPhone, nhấn **"Cho phép" (Allow)** khi hộp thoại hỏi quyền camera xuất hiện.

---

### 6. BUILD & RUN (CÁC BƯỚC NẠP APP LÊN IPHONE)
1. Cắm iPhone vào máy Mac bằng cáp kết nối.
2. Mở khóa màn hình iPhone và chọn **"Tin cậy máy tính này" (Trust This Computer)**.
3. Trên iPhone: Vào `Cài đặt` $\rightarrow$ `Quyền riêng tư & Bảo mật` $\rightarrow$ bật **Chế độ nhà phát triển (Developer Mode)** $\rightarrow$ Khởi động lại máy.
4. Trong Xcode, ở thanh công cụ phía trên, chọn thiết bị đích là chiếc iPhone vừa cắm.
5. Nhấn tổ hợp phím **`Cmd + R`** (hoặc nút Play ▶) để biên dịch và nạp app trực tiếp vào iPhone.

---

### 7. PATIENT / SESSION SETUP (CẤU HÌNH PHIÊN KHÁM)
1. Mở ứng dụng **DrVanTruong Scanner** trên iPhone.
2. Nhập:
   * **Mã Bệnh Nhân (Patient UUID):** Copy từ đường dẫn hồ sơ trên web (ví dụ: `5a6ff79f-e9ff-4a06-ad0c-3b2dde2c907b`).
   * **Mã Phiên Quét (Session UUID):** Mã phiên scan được tạo tại hồ sơ bệnh nhân.
   * **Địa chỉ máy chủ (Host):** Nhập IP máy chủ Next.js (ví dụ: `http://192.168.1.150:3000` hoặc domain staging của phòng khám).
3. Nhấn **"BẮT ĐẦU QUÉT KHUÔN MẶT 3D"**.

---

### 8. SCAN PROCEDURE (QUY TRÌNH THAO TÁC QUÉT 5 GÓC)
1. **Khoảng cách:** Đặt máy cách khuôn mặt bệnh nhân khoảng **35 – 45 cm**.
2. **Căn chỉnh:** Đưa khuôn mặt vào trong vòng elip HUD màu xanh.
3. **Thực hiện theo chỉ dẫn trên màn hình:**
   * **Góc 1 (Chính diện 0°):** Nhìn thẳng, nét mặt tự nhiên $\rightarrow$ Nhấn chụp (hoặc auto-trigger).
   * **Góc 2 (Nghiêng trái 30°-45°):** Quay nhẹ đầu sang phải người khám $\rightarrow$ Chụp.
   * **Góc 3 (Nghiêng sâu 60°-75°):** Nghiêng sâu hơn để lấy sống mũi sắc nét $\rightarrow$ Chụp.
   * **Góc 4 (Nghiêng phải 30°-45°):** Quay nhẹ đầu sang trái người khám $\rightarrow$ Chụp.
   * **Góc 5 (Nghiêng sâu 60°-75°):** Nghiêng sâu để lấy viền mặt đối diện $\rightarrow$ Chụp.

---

### 9. PACKAGE UPLOAD & TRANSMISSION
1. Sau khi chụp đủ 5 góc, màn hình tổng kết **ScanSummaryUploadView** hiển thị 5 ảnh thu nhỏ.
2. Nhấn nút **"TẢI LÊN & TÁI TẠO BASELINE 3D"**.
3. Ứng dụng đóng gói `manifest.json`, 5 file ảnh RGB, đệm độ sâu 16-bit và tọa độ 1220 đỉnh gửi về endpoint:
   `POST /api/patients/[id]/scan-sessions/[sessionId]/package`.

---

### 10. RECONSTRUCTION EXECUTION
1. Server Next.js nhận gói scan và kích hoạt `ai-engine/reconstruct_native_truedepth.py`.
2. Engine Python:
   * Khớp hệ trục tọa độ thực tế theo camera intrinsics.
   * Đồng bộ lưới giải phẫu 10.995 đỉnh kín nước theo mốc TrueDepth.
   * Chiếu và hòa trộn texture Laplacian Pyramid 2048x2048 (`face_HD.png`).
   * Xuất file nhị phân tiêu chuẩn `baseline.glb`.

---

### 11. BASELINE.GLB OUTPUT VERIFICATION
Kiểm tra các tệp sinh ra tại thư mục lưu trữ của bệnh nhân trên server:
* `public/models/patients/[id]/reconstruction/baseline.glb` (~1.76 MB).
* `public/models/patients/[id]/reconstruction/baseline.json`.
* `public/models/patients/[id]/reconstruction/landmarks.json`.
* `public/models/patients/[id]/reconstruction/quality.json`.

---

### 12. 3D STUDIO VERIFICATION (KIỂM TRA TRÊN WEB STUDIO)
1. Trên ứng dụng iOS, nhấn nút **"MỞ 3D STUDIO TƯ VẤN VIP →"** (hoặc mở trình duyệt trên máy tính tại `/patients/[id]/studio`).
2. Quan sát mô hình `baseline.glb` tải lên Three.js `Canvas3D`:
   * Xoay góc 360°, kiểm tra độ tương đồng sống mũi, chóp mũi, cằm với bệnh nhân thật.
   * Thử kéo thanh trượt nắn chỉnh sống mũi (Rhinoplasty) và độn cằm để kiểm tra độ đàn hồi bề mặt.

---

### 13. WHAT MEASUREMENTS TO RECORD (CÁC CHỈ SỐ CẦN GHI NHẬN)
* Thời gian quét (giây).
* Số lượng điểm mốc 3D khớp hợp lệ (Landmark count).
* Sai số tái chiếu trung bình (Mean Reprojection Error in px / %).
* Khoảng cách gian đồng tử đo được ($mm$).
* Mức độ ăn khớp đường viền sống mũi khi nhìn góc nghiêng 90°.

---

### 14. FAILURE CONDITIONS & TROUBLESHOOTING
| Lỗi thường gặp | Nguyên nhân | Cách khắc phục |
| :--- | :--- | :--- |
| **"Thiết bị không có TrueDepth"** | Chạy trên iPhone SE hoặc Simulator | Đổi sang iPhone 11/12/13/14/15/16 Pro hoặc iPad Pro thật. |
| **"Mất dấu khuôn mặt" ở góc nghiêng** | Quay đầu quá 80° làm khuất mắt xa | Hướng dẫn bệnh nhân chỉ quay góc 60°-70° vừa đủ thấy sống mũi. |
| **Không tải được lên Server** | Sai IP máy chủ hoặc khác mạng Wi-Fi | Đảm bảo iPhone và máy tính chạy server cùng kết nối vào một mạng Wi-Fi nội bộ. |
| **Ảnh bị nhòe / tối** | Phòng khám thiếu sáng hoặc rung tay | Tăng đèn rọi khám, giữ chắc tay khi chụp. |

---

### 15. LOGS & DEBUG PACKAGE EXPORT
* Gói scan thô lưu tạm trên iPhone tại thư mục `Documents/ScanPackage_[sessionId]/`.
* Có thể kéo thư mục scan này ra máy Mac thông qua **Xcode** $\rightarrow$ `Devices and Simulators` $\rightarrow$ `Installed Apps` $\rightarrow$ `Download Container`.
