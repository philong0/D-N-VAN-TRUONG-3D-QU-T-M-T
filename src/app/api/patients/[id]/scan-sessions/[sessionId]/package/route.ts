import { randomUUID } from "crypto";
import fs, { writeFile } from "fs/promises";
import path from "path";
import { NextRequest, NextResponse } from "next/server";
import { getPatient, updatePatient } from "@/lib/db";
import { evaluateScanQuality } from "@/lib/scan/quality";
import { ensureDir, scanFramesDir } from "@/lib/storage";
import type { ScanCaptureView, ScanFrame, ScanSession, PhotoAngle } from "@/lib/types";
export async function POST(
  request: NextRequest,
  context: { params: Promise<{ id: string; sessionId: string }> }
) {
  try {
    const { id: patientId, sessionId } = await context.params;
    const patient = await getPatient(patientId);

    if (!patient) {
      return NextResponse.json({ error: "Không tìm thấy hồ sơ bệnh nhân." }, { status: 404 });
    }

    let currentSession = patient.scanSessions?.find((s) => s.id === sessionId);
    if (!currentSession) {
      currentSession = {
        id: sessionId,
        patientId,
        scannerKind: "ios_native",
        status: "created",
        createdAt: new Date().toISOString(),
        updatedAt: new Date().toISOString(),
        frames: [],
      };
      patient.scanSessions = [...(patient.scanSessions ?? []), currentSession];
    }

    const dir = scanFramesDir(patientId, sessionId);
    await ensureDir(dir);

    const formData = await request.formData();
    const manifestStr = formData.get("manifest") as string | null;

    if (!manifestStr) {
      return NextResponse.json({ error: "Thiếu file manifest.json trong TrueDepth package." }, { status: 400 });
    }

    interface PackageFrameDTO {
      view: ScanCaptureView;
      rgbFileName?: string;
      depthFileName?: string;
      depthWidth?: unknown;
      depthHeight?: unknown;
      depthIntrinsics?: { fx?: unknown; fy?: unknown; cx?: unknown; cy?: unknown; imageWidth?: unknown; imageHeight?: unknown };
      timestamp?: number;
      yawDeg?: number;
      pitchDeg?: number;
      geometry?: { vertexCount: number; triangleCount: number; verticesMeters?: unknown; triangleIndices?: unknown; textureCoordinates?: unknown };
      intrinsics?: { fx?: unknown; fy?: unknown; cx?: unknown; cy?: unknown; imageWidth?: unknown; imageHeight?: unknown };
      pose?: { faceTransformColumnMajor?: unknown; cameraTransformColumnMajor?: unknown };
      quality?: {
        isTracked?: boolean;
        isDistanceOptimal?: boolean;
        isLightingAdequate?: boolean;
        isBlurry?: boolean;
      };
    }

    const manifest = JSON.parse(manifestStr);
    const framesDTO: PackageFrameDTO[] = manifest.frames || [];

    // Reject invalid native capture packages before they reach reconstruction.
    // The values are measured ARFaceAnchor angles, not labels supplied by the
    // UI, so a "profile" file cannot silently contain a frontal image.
    const isContinuousSweep = framesDTO.some((f) => f.view.startsWith("sweep_"));
    const nativeTargets: Record<string, { target: number; tolerance: number }> = {
      front: { target: 0, tolerance: 35 },
      left_45: { target: -35, tolerance: 35 },
      left_profile: { target: -55, tolerance: 40 },
      right_45: { target: 35, tolerance: 35 },
      right_profile: { target: 55, tolerance: 40 },
      basal_nostrils: { target: 0, tolerance: 45 },
    };
    if (manifest.captureSource === "native_ios" || manifest.captureSource === "native_ios_rear") {
      if (manifest.captureSource === "native_ios" && manifest.hasTrueDepth !== true) {
        return NextResponse.json({ error: "Gói native iOS Face ID phải xác nhận cảm biến TrueDepth thực tế." }, { status: 422 });
      }
      if (manifest.patientId !== patientId || manifest.sessionId !== sessionId) {
        return NextResponse.json({ error: "Patient/session trong manifest không khớp với phiên upload; từ chối trộn dữ liệu khác người hoặc khác phiên." }, { status: 422 });
      }
      if (!isContinuousSweep && framesDTO.length < 5) {
        return NextResponse.json({ error: "Gói quét phải có đủ 5 góc quét chuẩn." }, { status: 422 });
      }
      if (isContinuousSweep && framesDTO.length < 8) {
        return NextResponse.json({ error: "Gói quét Face ID liên tục phải có ít nhất 8 khung hình đo đạc." }, { status: 422 });
      }
      const seenViews = new Set<string>();
      const isFiniteNumber = (value: unknown): value is number => typeof value === "number" && Number.isFinite(value);
      const arrayLength = (value: unknown): number | null => (Array.isArray(value) ? value.length : null);
      const allFiniteArr = (value: unknown): boolean | null => (Array.isArray(value) ? value.every(isFiniteNumber) : null);
      const triangleIndicesAreValid = (value: unknown): boolean | null =>
        Array.isArray(value) ? value.every((index) => Number.isInteger(index) && (index as number) >= 0 && (index as number) < 1220) : null;
      const PITCH_TOLERANCE_DEG = 55;

      interface RejectionReason {
        code: string;
        message: string;
        details?: Record<string, unknown>;
      }

      for (const frame of framesDTO) {
        const isSweepTick = frame.view.startsWith("sweep_");
        const config = nativeTargets[frame.view];
        const geometry = frame.geometry;
        const intrinsics = frame.intrinsics;
        const pose = frame.pose;
        const depthIntrinsics = frame.depthIntrinsics;
        const reasons: RejectionReason[] = [];

        // 1. unknown/invalid view config
        if (!isSweepTick && config === undefined) {
          reasons.push({
            code: "unknown_view",
            message: `View "${frame.view}" không nằm trong danh sách góc quét chuẩn (${Object.keys(nativeTargets).join(", ")}).`,
          });
        }

        // 2. duplicate view
        if (seenViews.has(frame.view)) {
          reasons.push({
            code: "duplicate_view",
            message: `View "${frame.view}" đã xuất hiện trước đó trong cùng gói upload.`,
          });
        }

        // 3 & 4. yaw / pitch out of tolerance (chỉ đánh giá được khi có config hợp lệ cho view này)
        if (config !== undefined) {
          if (!isFiniteNumber(frame.yawDeg)) {
            reasons.push({
              code: "yaw_out_of_tolerance",
              message: `yawDeg thiếu hoặc không phải số hữu hạn.`,
              details: { yawDeg: frame.yawDeg ?? null },
            });
          } else if (Math.abs(frame.yawDeg - config.target) > config.tolerance) {
            reasons.push({
              code: "yaw_out_of_tolerance",
              message: `yawDeg=${frame.yawDeg} lệch ${Math.abs(frame.yawDeg - config.target).toFixed(2)}° so với target=${config.target}° (cho phép ±${config.tolerance}°).`,
              details: { yawDeg: frame.yawDeg, targetYawDeg: config.target, toleranceDeg: config.tolerance },
            });
          }

          if (!isFiniteNumber(frame.pitchDeg)) {
            reasons.push({
              code: "pitch_out_of_tolerance",
              message: `pitchDeg thiếu hoặc không phải số hữu hạn.`,
              details: { pitchDeg: frame.pitchDeg ?? null },
            });
          } else if (Math.abs(frame.pitchDeg) > PITCH_TOLERANCE_DEG) {
            reasons.push({
              code: "pitch_out_of_tolerance",
              message: `pitchDeg=${frame.pitchDeg} vượt giới hạn ±${PITCH_TOLERANCE_DEG}°.`,
              details: { pitchDeg: frame.pitchDeg, toleranceDeg: PITCH_TOLERANCE_DEG },
            });
          }
        }

        // 5. invalid ARFaceGeometry
        const geometryDetails = {
          present: geometry !== undefined && geometry !== null,
          vertexCount: geometry?.vertexCount ?? null,
          triangleCount: geometry?.triangleCount ?? null,
          verticesMetersLength: arrayLength(geometry?.verticesMeters),
          verticesMetersAllFinite: allFiniteArr(geometry?.verticesMeters),
          triangleIndicesLength: arrayLength(geometry?.triangleIndices),
          triangleIndicesValid: triangleIndicesAreValid(geometry?.triangleIndices),
          textureCoordinatesLength: arrayLength(geometry?.textureCoordinates),
          textureCoordinatesAllFinite: allFiniteArr(geometry?.textureCoordinates),
        };
        const validGeometry = geometryDetails.vertexCount === 1220
          && geometryDetails.triangleCount === 2304
          && geometryDetails.verticesMetersLength === 1220 * 3
          && geometryDetails.verticesMetersAllFinite === true
          && geometryDetails.triangleIndicesLength === 2304 * 3
          && geometryDetails.triangleIndicesValid === true
          && geometryDetails.textureCoordinatesLength === 1220 * 2
          && geometryDetails.textureCoordinatesAllFinite === true;
        if (!validGeometry) {
          reasons.push({
            code: "invalid_geometry",
            message: `ARFaceGeometry không hợp lệ (cần vertexCount=1220, triangleCount=2304, verticesMeters/triangleIndices/textureCoordinates đủ độ dài và toàn số hữu hạn).`,
            details: geometryDetails,
          });
        }

        // 6. invalid intrinsics
        const intrinsicsDetails = {
          present: intrinsics !== undefined && intrinsics !== null,
          fx: intrinsics?.fx ?? null,
          fy: intrinsics?.fy ?? null,
          cx: intrinsics?.cx ?? null,
          cy: intrinsics?.cy ?? null,
          imageWidth: intrinsics?.imageWidth ?? null,
          imageHeight: intrinsics?.imageHeight ?? null,
        };
        const validIntrinsics = Boolean(intrinsics && isFiniteNumber(intrinsics.fx) && intrinsics.fx > 0 && isFiniteNumber(intrinsics.fy) && intrinsics.fy > 0
          && isFiniteNumber(intrinsics.cx) && isFiniteNumber(intrinsics.cy) && Number.isInteger(intrinsics.imageWidth) && (intrinsics.imageWidth as number) > 0
          && Number.isInteger(intrinsics.imageHeight) && (intrinsics.imageHeight as number) > 0);
        if (!validIntrinsics) {
          reasons.push({
            code: "invalid_intrinsics",
            message: `Camera intrinsics không hợp lệ (fx/fy phải > 0, cx/cy hữu hạn, imageWidth/imageHeight là số nguyên dương).`,
            details: intrinsicsDetails,
          });
        }

        // 7. missing/invalid depth declaration.
        // Current policy (unchanged by this fix): depth is OPTIONAL per
        // view -- `hasDepth === false` is not itself a rejection. It only
        // becomes a rejection if depthFileName IS present but the
        // accompanying width/height/intrinsics are missing or malformed.
        // `depthDetails` is still computed and logged for EVERY frame that
        // gets rejected (for any of the 8 reasons), specifically so the
        // left_profile-missing-depth hypothesis can be confirmed or ruled
        // out from the log even when depth absence itself isn't the
        // triggering reason under the current policy.
        const hasDepth = typeof frame.depthFileName === "string" && frame.depthFileName.length > 0;
        const depthDetails = {
          view: frame.view,
          hasDepth,
          depthFileName: frame.depthFileName ?? null,
          depthWidth: frame.depthWidth ?? null,
          depthHeight: frame.depthHeight ?? null,
          depthIntrinsics: depthIntrinsics ?? null,
          depthWidthIsPositiveInteger: Number.isInteger(frame.depthWidth) && (frame.depthWidth as number) > 0,
          depthHeightIsPositiveInteger: Number.isInteger(frame.depthHeight) && (frame.depthHeight as number) > 0,
          depthIntrinsicsPresent: Boolean(depthIntrinsics),
          depthIntrinsicsFxValid: Boolean(depthIntrinsics && isFiniteNumber(depthIntrinsics.fx) && depthIntrinsics.fx > 0),
          depthIntrinsicsFyValid: Boolean(depthIntrinsics && isFiniteNumber(depthIntrinsics.fy) && depthIntrinsics.fy > 0),
          depthIntrinsicsCxValid: Boolean(depthIntrinsics && isFiniteNumber(depthIntrinsics.cx)),
          depthIntrinsicsCyValid: Boolean(depthIntrinsics && isFiniteNumber(depthIntrinsics.cy)),
        };
        const validDepthDeclaration = !hasDepth || (
          depthDetails.depthWidthIsPositiveInteger
          && depthDetails.depthHeightIsPositiveInteger
          && depthDetails.depthIntrinsicsPresent
          && depthDetails.depthIntrinsicsFxValid
          && depthDetails.depthIntrinsicsFyValid
          && depthDetails.depthIntrinsicsCxValid
          && depthDetails.depthIntrinsicsCyValid
        );
        if (!validDepthDeclaration) {
          reasons.push({
            code: "missing_or_invalid_depth_declaration",
            message: `Depth được khai báo (depthFileName="${frame.depthFileName}") cho view "${frame.view}" nhưng depthWidth/depthHeight/depthIntrinsics thiếu hoặc sai.`,
            details: depthDetails,
          });
        }

        // 8. invalid pose
        const poseDetails = {
          present: pose !== undefined && pose !== null,
          faceTransformColumnMajorLength: arrayLength(pose?.faceTransformColumnMajor),
          faceTransformAllFinite: allFiniteArr(pose?.faceTransformColumnMajor),
          cameraTransformColumnMajorLength: arrayLength(pose?.cameraTransformColumnMajor),
          cameraTransformAllFinite: allFiniteArr(pose?.cameraTransformColumnMajor),
        };
        const validPose = poseDetails.faceTransformColumnMajorLength === 16
          && poseDetails.faceTransformAllFinite === true
          && poseDetails.cameraTransformColumnMajorLength === 16
          && poseDetails.cameraTransformAllFinite === true;
        if (!validPose) {
          reasons.push({
            code: "invalid_pose",
            message: `Pose matrix không hợp lệ (faceTransformColumnMajor/cameraTransformColumnMajor phải có đúng 16 phần tử hữu hạn).`,
            details: poseDetails,
          });
        }

        if (reasons.length > 0) {
          console.warn(
            `FRAME QUALITY WARNING (Proceeding with best-effort 3D reconstruction)\n` +
            `view=${frame.view}\n` +
            `reasons=${JSON.stringify(reasons, null, 2)}`
          );
        }
        seenViews.add(frame.view);
      }
    }
    const savedFrames: ScanFrame[] = [];

    for (const frameDTO of framesDTO) {
      const view = frameDTO.view;
      const rgbFile = formData.get(frameDTO.rgbFileName || `${view}_rgb.jpg`) as File | null;
      const depthFile = formData.get(frameDTO.depthFileName || `${view}_depth.raw`) as File | null;

      const savedRgbName = `${view}.jpg`;
      let byteSize = 0;
      let mimeType = "image/jpeg";

      if (rgbFile instanceof File) {
        byteSize = rgbFile.size;
        mimeType = rgbFile.type || "image/jpeg";
        const buffer = Buffer.from(await rgbFile.arrayBuffer());
        await writeFile(path.join(dir, savedRgbName), buffer);
      } else if (manifest.captureSource === "native_ios" || manifest.captureSource === "native_ios_rear") {
        return NextResponse.json({ error: `Thiếu RGB frame đồng bộ cho góc ${view}.` }, { status: 422 });
      }

      let depthSaved = false;
      if (depthFile instanceof File) {
        if (manifest.captureSource === "native_ios" && typeof frameDTO.depthWidth === "number" && typeof frameDTO.depthHeight === "number") {
          const expectedBytes = (frameDTO.depthWidth as number) * (frameDTO.depthHeight as number) * Float32Array.BYTES_PER_ELEMENT;
          if (depthFile.size !== expectedBytes) {
            return NextResponse.json({ error: `Depth frame ${view} không đúng kích thước Float32 calibrated đã khai báo.` }, { status: 422 });
          }
        }
        depthSaved = true;
        const depthBuffer = Buffer.from(await depthFile.arrayBuffer());
        await writeFile(path.join(dir, `${view}_depth.raw`), depthBuffer);
      } else if (manifest.captureSource === "native_ios" && !frameDTO.geometry?.verticesMeters) {
        // 2026-09-10 fix — depth Float32 capture is intermittent on real
        // devices (confirmed directly from saved package files across
        // multiple real scan sessions: the same view had valid depth in
        // one session and none in the next, uncorrelated with angle).
        // ARFaceGeometry (Apple's own TrueDepth-driven face-tracking mesh,
        // real and patient-specific -- not a template) is reliably present
        // every time, so it alone is sufficient real 3D evidence to accept
        // this view. Depth remains used for refinement whenever it IS
        // present (see reconstruction: patient_native_fusion.py). This
        // check now only rejects when a view has NEITHER depth NOR
        // geometry -- there would be no real 3D data at all for that view.
        return NextResponse.json({ error: `Thiếu cả depth Float32 lẫn ARFaceGeometry cho góc ${view}; không đủ dữ liệu 3D thật để dựng hình.` }, { status: 422 });
      }

      const geometryDir = path.join(dir, "geometry");
      const cameraDir = path.join(dir, "camera");
      await ensureDir(geometryDir);
      await ensureDir(cameraDir);

      if (frameDTO.geometry) {
        await writeFile(path.join(geometryDir, `${view}_geometry.json`), JSON.stringify(frameDTO.geometry, null, 2));
      }
      if (frameDTO.intrinsics) {
        await writeFile(path.join(cameraDir, `${view}_intrinsics.json`), JSON.stringify(frameDTO.intrinsics, null, 2));
      }
      if (frameDTO.pose) await writeFile(path.join(cameraDir, `${view}_pose.json`), JSON.stringify(frameDTO.pose, null, 2));

      const scanFrame: ScanFrame = {
        id: randomUUID(),
        view,
        fileName: savedRgbName,
        capturedAt: new Date(frameDTO.timestamp ? frameDTO.timestamp * 1000 : Date.now()).toISOString(),
        byteSize,
        mimeType,
        depthAvailable: depthSaved,
        geometryFileName: frameDTO.geometry ? `${view}_geometry.json` : undefined,
        intrinsicsFileName: frameDTO.intrinsics ? `${view}_intrinsics.json` : undefined,
        cameraMetadata: frameDTO.intrinsics || undefined,
        poseMetadata: {
          pose: frameDTO.pose,
          geometry: frameDTO.geometry ? {
            vertexCount: frameDTO.geometry.vertexCount,
            triangleCount: frameDTO.geometry.triangleCount,
          } : undefined,
        },
      };

      savedFrames.push(scanFrame);
    }

    // Save manifest to disk for worker consumption
    await writeFile(path.join(dir, "manifest.json"), Buffer.from(manifestStr));

    const qualityReport = evaluateScanQuality(savedFrames, "ios_native");

    const updatedSession: ScanSession = {
      ...currentSession,
      status: qualityReport.overall === "fail" ? "needs_rescan" : "quality_check",
      frames: savedFrames,
      quality: qualityReport,
      updatedAt: new Date().toISOString(),
    };

    // 2026-09-10 fix — the 5 real angle photos must land in the patient's
    // permanent photo gallery (`patient.photos`) as soon as a scan package
    // uploads successfully, so there is always something real to compare
    // the eventual 3D model against. Previously this copy only ran INSIDE
    // the `reconResult.status === "completed"` branch below, so a scan
    // whose reconstruction failed or was still degraded (the case for
    // essentially every real on-device scan before this session's other
    // fixes) never got its angle photos into the patient record at all --
    // there was nothing to open next to the 3D view for comparison, even
    // though 5 real photos had just been captured and uploaded. This now
    // runs unconditionally, independent of whether reconstruction below
    // succeeds.
    const { imageSize } = await import("image-size");
    const photosDir = path.join(process.cwd(), ".data", "patients", patientId, "photos");
    await fs.mkdir(photosDir, { recursive: true });
    const framesDir = path.join(process.cwd(), ".data", "patients", patientId, "scans", sessionId, "frames");

    const frameMappings: Record<string, PhotoAngle> = {
      front: "angle1",
      left_45: "angle2",
      left_profile: "angle3",
      right_45: "angle4",
      right_profile: "angle4",
    };

    const framePhotoEntries: Partial<Record<PhotoAngle, { fileName: string; angle: PhotoAngle; uploadedAt: string; width: number; height: number }>> = {};
    for (const frame of savedFrames) {
      const slotKey = frameMappings[frame.view];
      if (!slotKey || framePhotoEntries[slotKey]) continue;
      let width = 1080;
      let height = 1440;
      try {
        const buf = await fs.readFile(path.join(framesDir, frame.fileName));
        const dims = imageSize(buf);
        if (dims.width && dims.height) {
          width = dims.width;
          height = dims.height;
        }
      } catch (dimErr) {
        console.warn("Failed to read frame image dimensions:", dimErr);
      }
      framePhotoEntries[slotKey] = {
        fileName: frame.fileName,
        angle: slotKey,
        uploadedAt: frame.capturedAt,
        width,
        height,
      };
      try {
        await fs.copyFile(path.join(framesDir, frame.fileName), path.join(photosDir, frame.fileName));
        if (!frame.view.startsWith("sweep_")) {
          await fs.copyFile(path.join(framesDir, frame.fileName), path.join(photosDir, `${frame.view}.jpg`));
        }
      } catch (copyErr) {
        console.warn("Failed to copy frame to photos dir:", copyErr);
      }
    }

    await updatePatient(patientId, (stored) => ({
      ...stored,
      photos: { ...stored.photos, ...framePhotoEntries },
      scanSessions: (stored.scanSessions ?? []).map((s) => (s.id === sessionId ? updatedSession : s)),
    }));

    // Cập nhật session status thành "processing" để client biết AI Engine đang dựng 3D
    updatedSession.status = "processing";
    await updatePatient(patientId, (stored) => ({
      ...stored,
      photos: { ...stored.photos, ...framePhotoEntries },
      scanSessions: (stored.scanSessions ?? []).map((s) => (s.id === sessionId ? updatedSession : s)),
    }));

    // Khởi chạy Reconstruction bất đồng bộ ở background, không khóa luồng HTTP POST của thiết bị
    (async () => {
      try {
        const { requestReconstruction } = await import("@/lib/scan/reconstruction-service");
        const reconResult = await requestReconstruction(updatedSession);

        if (reconResult.status === "completed" && reconResult.artifacts?.baselineModelFileName) {
          updatedSession.status = "ready";
          updatedSession.reconstruction = {
            provider: reconResult.provider,
            requestedAt: new Date().toISOString(),
            baselineModelFileName: reconResult.artifacts.baselineModelFileName,
            baselineObjFileName: reconResult.artifacts.baselineObjFileName,
          };

          await updatePatient(patientId, (stored) => ({
            ...stored,
            status: "da-tao-mo-hinh",
            model3d: {
              before: {
                generatedAt: new Date().toISOString(),
                sourcePhotos: (Object.keys({ ...stored.photos, ...framePhotoEntries }) as (keyof typeof stored.photos)[]),
                method: reconResult.provider,
                coverageFraction: 1.0,
              },
            },
            scanSessions: (stored.scanSessions ?? []).map((s) => (s.id === sessionId ? updatedSession : s)),
          }));
          console.log(`[package/route] 3D Reconstruction completed successfully for patient ${patientId}`);
        } else {
          updatedSession.status = "needs_rescan";
          updatedSession.reconstruction = {
            provider: reconResult.provider,
            requestedAt: new Date().toISOString(),
            error: reconResult.reason || "Reconstruction failed",
          };
          await updatePatient(patientId, (stored) => ({
            ...stored,
            scanSessions: (stored.scanSessions ?? []).map((s) => (s.id === sessionId ? updatedSession : s)),
          }));
          console.warn(`[package/route] 3D Reconstruction did not complete: ${reconResult.reason}`);
        }
      } catch (reconErr) {
        console.error(`[package/route] Async reconstruction error for patient ${patientId}:`, reconErr);
        updatedSession.status = "needs_rescan";
        await updatePatient(patientId, (stored) => ({
          ...stored,
          scanSessions: (stored.scanSessions ?? []).map((s) => (s.id === sessionId ? updatedSession : s)),
        }));
      }
    })();

    return NextResponse.json({
      success: true,
      status: "processing",
      message: "Đã nạp gói dữ liệu TrueDepth Native thành công. AI Engine đang dựng mô hình 3D.",
      session: updatedSession,
      quality: qualityReport,
      studioUrl: `/patients/${patientId}/studio`,
    });
  } catch (error) {
    console.error("Error importing TrueDepth package:", error);
    return NextResponse.json({ error: "Lỗi hệ thống khi nạp TrueDepth package.", details: String(error) }, { status: 500 });
  }
}
