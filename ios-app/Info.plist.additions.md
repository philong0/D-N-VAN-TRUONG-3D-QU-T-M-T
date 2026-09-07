# Info tab keys to add in Xcode

Add these in the target's **Info** tab (or edit `Info.plist` source directly
if you prefer — both write the same keys):

| Key | Type | Value |
|---|---|---|
| `NSCameraUsageDescription` | String | `Ứng dụng cần quyền Camera để quét khuôn mặt 3D bằng cảm biến TrueDepth cho hồ sơ bệnh nhân.` |
| `NSPhotoLibraryUsageDescription` | String | `Cho phép chọn ảnh khuôn mặt từ thư viện khi không dùng được camera trực tiếp.` (only needed if you keep the native file-picker fallback) |

ARKit itself needs no `Info.plist` capability entry — `ARFaceTrackingConfiguration.isSupported`
is just a runtime check (see `ARFaceCaptureController.swift`), and the app
will simply refuse to start face tracking on a non-TrueDepth device instead
of crashing (already handled in code).

If you want the App Store listing to correctly gate installs to
TrueDepth-capable devices only, also add under **Required device
capabilities** (`UIRequiredDeviceCapabilities` array):

```
arkit
```
