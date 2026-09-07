# Dr. VanTruong Scanner — iOS TrueDepth wrapper app

## What this is

A thin native iOS app whose only job is to give the existing web app
(`GuidedFaceScan.tsx`, served from this same Next.js project) access to a
REAL TrueDepth face mesh — something no browser API can do. It does **not**
duplicate the scan UI: it hosts the web app's own `/patients/[id]/scan` page
inside a `WKWebView` and injects one native capability into it, matching the
contract `src/lib/scan/face-scanner.ts` already defines for `IOSNativeScanner`:

- `window.webkit.messageHandlers.arkitScanBridge` — a `WKScriptMessageHandler`
  the web page's JS posts `{action: "start"}` / `{action: "capture"}` to.
- The native side runs `ARSession` with `ARFaceTrackingConfiguration`,
  captures the real `ARFaceGeometry` (1220 vertices / 2304 triangles, fixed
  topology across frames), the face anchor's world transform, the camera's
  REAL intrinsics (`ARCamera.intrinsics` — not the `focal = image_width`
  approximation the web/photo-upload pipeline has to use), and the RGB
  camera frame at that instant.
- The captured payload is uploaded directly (native `URLSession`, not routed
  back through JS — geometry+image is real binary/JSON data, no reason to
  round-trip it through `postMessage`) to the SAME endpoint the web capture
  flow already posts to:
  `POST /api/patients/{patientId}/scan-sessions/{sessionId}/frames`
  now extended (see `src/app/api/.../frames/route.ts`) to accept an optional
  `geometry` (JSON) and `intrinsics` (JSON) field alongside `frame`/`view`.
- After a successful upload, the native side calls back into the page's JS
  (`window.__arkitCaptureResolve(...)`) so the *existing* `GuidedFaceScan.tsx`
  step-by-step flow (already built, already does quality scoring) continues
  completely unchanged — this app adds real depth data underneath it, it
  does not replace the UI.

Why a WebView wrapper instead of a fully separate native UI: the guided
5-angle capture flow, quality report, and patient-record integration already
exist and are tested in the web app. Rebuilding all of that in SwiftUI would
duplicate a lot of logic for no accuracy benefit — the only thing native
code can do that the browser can't is talk to ARKit/TrueDepth, so that's the
only thing this app does.

## Files

- `DrVanTruongScannerApp.swift` — `@main` app entry point, launches `FaceScannerView`.
- `FaceScannerView.swift` — SwiftUI view hosting the `WKWebView`, registers the
  `arkitScanBridge` message handler, owns an `ARFaceCaptureController`.
- `ARFaceCaptureController.swift` — `ARSessionDelegate` wrapper: starts/stops
  face tracking, captures one `ARFrame` + `ARFaceAnchor` on demand, encodes
  it into the JSON shape `face-scanner.ts`'s `ARKitFaceGeometryPayload`
  expects, extracts a JPEG of the camera frame.
- `NetworkService.swift` — multipart upload of one captured view to the
  Next.js backend, plus a small health-check call.
- `Info.plist.additions.md` — the exact keys to add in Xcode (camera usage
  string, ARKit requirement) since this repo has no `.xcodeproj` checked in.

## Setting this up in Xcode (no `.xcodeproj` is checked in)

TrueDepth/ARKit requires a real device (iPhone X or later, or an iPad Pro
with a TrueDepth front camera) and a paid or free Apple Developer account to
install a development build — none of that exists in this Linux sandbox, so
this code has been written carefully against the public ARKit/WebKit APIs
but has **not** been compiled or run on a device. Steps to get it running:

1. On a Mac with Xcode 15+: **File → New → Project → iOS → App**. Product
   name `DrVanTruongScanner`, interface **SwiftUI**, language **Swift**,
   minimum deployment target **iOS 14.0** (`ARFaceTrackingConfiguration` has
   existed since iOS 11, `ARFaceGeometry.textureCoordinates` since iOS 12 —
   14.0 is a safe modern floor).
2. Delete the template's generated `ContentView.swift`/`*App.swift`, drag
   the 4 `.swift` files from this folder into the project (check "Copy items
   if needed").
3. Project target → **Signing & Capabilities** → add **"ARKit"** isn't a
   capability toggle (no entry needed) — ARKit just needs the usage-string
   key below. Do add camera + (if you want native photo-library fallback)
   photo-library usage strings.
4. Project target → **Info** tab → add the keys listed in
   `Info.plist.additions.md`.
5. In `FaceScannerView.swift`, set `webAppBaseURL` to wherever this Next.js
   app is actually reachable from the device (a LAN IP + port during dev,
   e.g. `http://192.168.1.20:3000`, or the deployed HTTPS URL in
   production — `http://localhost:3000` only works in the iOS Simulator,
   never on a physical device).
6. Build & run on a TrueDepth-capable device. The web page's own
   `IOSNativeScanner.isAvailable()` check (`face-scanner.ts`) will now
   return `true` inside this app's WebView, and `GuidedFaceScan.tsx`'s
   capture step will call into real ARKit instead of `getUserMedia`.

## Known gaps / what's next

- Depth is captured as ARKit's own fitted face **mesh** (1220 real 3D
  vertices from the TrueDepth sensor + Apple's on-device face-tracking
  model), not a raw depth map — this is the right level for face shape
  (it's literally what ARKit exists to give you) and is far more accurate
  than the 2D-photo/landmark pipeline, but it means the reconstruction
  server (`ai-engine/arkit_reconstruction.py`) works with a *fixed* 1220-vertex
  topology, different from the GNM 17821-vertex template the photo-upload
  pipeline uses. These are two independent reconstruction outputs; see that
  file's own docstring.
- `NetworkService.swift`'s base URL and any auth are placeholders — this repo
  has no auth layer today (matches the rest of the API), so none was added.
- Not compiled/run — review the Swift carefully against the current
  ARKit/WebKit SDK before shipping; API shapes here match public Apple
  documentation as of this writing but a real Xcode build is the only real
  verification.
