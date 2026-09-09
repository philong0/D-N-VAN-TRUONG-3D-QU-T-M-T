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

    const currentSession = patient.scanSessions?.find((s) => s.id === sessionId);
    if (!currentSession || currentSession.patientId !== patientId) {
      return NextResponse.json({ error: "Scan session không hợp lệ cho hồ sơ này." }, { status: 404 });
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
    const nativeTargets: Record<string, { target: number; tolerance: number }> = {
      front: { target: 0, tolerance: 20 },
      left_45: { target: -40, tolerance: 22 },
      left_profile: { target: -55, tolerance: 25 },
      right_45: { target: 40, tolerance: 22 },
      right_profile: { target: 55, tolerance: 25 },
    };
    if (manifest.captureSource === "native_ios") {
      if (manifest.hasTrueDepth !== true) {
        return NextResponse.json({ error: "Gói native iOS phải xác nhận cảm biến TrueDepth thực tế." }, { status: 422 });
      }
      if (manifest.patientId !== patientId || manifest.sessionId !== sessionId) {
        return NextResponse.json({ error: "Patient/session trong manifest không khớp với phiên upload; từ chối trộn dữ liệu khác người hoặc khác phiên." }, { status: 422 });
      }
      if (framesDTO.length !== Object.keys(nativeTargets).length) {
        return NextResponse.json({ error: "Gói TrueDepth phải có đủ 5 góc quét chuẩn." }, { status: 422 });
      }
      const seenViews = new Set<string>();
      for (const frame of framesDTO) {
        const config = nativeTargets[frame.view];
        const geometry = frame.geometry;
        const intrinsics = frame.intrinsics;
        const pose = frame.pose;
        const isFiniteNumber = (value: unknown): value is number => typeof value === "number" && Number.isFinite(value);
        const validGeometry = geometry?.vertexCount === 1220
          && geometry.triangleCount === 2304
          && Array.isArray(geometry.verticesMeters) && geometry.verticesMeters.length === 1220 * 3
          && geometry.verticesMeters.every(isFiniteNumber)
          && Array.isArray(geometry.triangleIndices) && geometry.triangleIndices.length === 2304 * 3
          && geometry.triangleIndices.every((index) => Number.isInteger(index) && index >= 0 && index < 1220)
          && Array.isArray(geometry.textureCoordinates) && geometry.textureCoordinates.length === 1220 * 2
          && geometry.textureCoordinates.every(isFiniteNumber);
        const validIntrinsics = Boolean(intrinsics && isFiniteNumber(intrinsics.fx) && intrinsics.fx > 0 && isFiniteNumber(intrinsics.fy) && intrinsics.fy > 0
          && isFiniteNumber(intrinsics.cx) && isFiniteNumber(intrinsics.cy) && Number.isInteger(intrinsics.imageWidth) && (intrinsics.imageWidth as number) > 0
          && Number.isInteger(intrinsics.imageHeight) && (intrinsics.imageHeight as number) > 0);
        const depthIntrinsics = frame.depthIntrinsics;
        const validDepthDeclaration = typeof frame.depthFileName === "string" && frame.depthFileName.length > 0
          && Number.isInteger(frame.depthWidth) && (frame.depthWidth as number) > 0
          && Number.isInteger(frame.depthHeight) && (frame.depthHeight as number) > 0
          && Boolean(depthIntrinsics && isFiniteNumber(depthIntrinsics.fx) && depthIntrinsics.fx > 0
            && isFiniteNumber(depthIntrinsics.fy) && depthIntrinsics.fy > 0
            && isFiniteNumber(depthIntrinsics.cx) && isFiniteNumber(depthIntrinsics.cy));
        const validPose = Boolean(pose && Array.isArray(pose.faceTransformColumnMajor) && pose.faceTransformColumnMajor.length === 16
          && pose.faceTransformColumnMajor.every(isFiniteNumber) && Array.isArray(pose.cameraTransformColumnMajor)
          && pose.cameraTransformColumnMajor.length === 16 && pose.cameraTransformColumnMajor.every(isFiniteNumber));
        if (
          config === undefined ||
          seenViews.has(frame.view) ||
          typeof frame.yawDeg !== "number" || Math.abs(frame.yawDeg - config.target) > config.tolerance ||
          typeof frame.pitchDeg !== "number" || Math.abs(frame.pitchDeg) > 25 ||
          !validGeometry || !validIntrinsics || !validDepthDeclaration || !validPose
        ) {
          return NextResponse.json({ error: `Frame ${frame.view} không đạt pose/quality TrueDepth; cần quét lại.` }, { status: 422 });
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
      } else if (manifest.captureSource === "native_ios") {
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
      } else if (manifest.captureSource === "native_ios") {
        return NextResponse.json({ error: `Thiếu metric depth Float32 đồng bộ cho góc ${view}; native reconstruction bị từ chối.` }, { status: 422 });
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

    // Auto-trigger reconstruction directly upon package upload
    try {
      const { requestReconstruction } = await import("@/lib/scan/reconstruction-service");
      const { imageSize } = await import("image-size");
      const reconResult = await requestReconstruction(updatedSession);

      if (reconResult.status === "completed" && reconResult.artifacts?.baselineModelFileName) {
        updatedSession.status = "ready";
        updatedSession.reconstruction = {
          provider: reconResult.provider,
          requestedAt: new Date().toISOString(),
          baselineModelFileName: reconResult.artifacts.baselineModelFileName,
          baselineObjFileName: reconResult.artifacts.baselineObjFileName,
        };

        // Synchronize scan frames to patient.photos gallery
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
          } catch (copyErr) {
            console.warn("Failed to copy frame to photos dir:", copyErr);
          }
        }

        await updatePatient(patientId, (stored) => {
          const syncedPhotos = { ...stored.photos, ...framePhotoEntries };
          return {
            ...stored,
            photos: syncedPhotos,
            status: "da-tao-mo-hinh",
            model3d: {
              before: {
                generatedAt: new Date().toISOString(),
                sourcePhotos: (Object.keys(syncedPhotos) as (keyof typeof stored.photos)[]),
                method: reconResult.provider,
                coverageFraction: 1.0,
              },
            },
            scanSessions: (stored.scanSessions ?? []).map((s) => (s.id === sessionId ? updatedSession : s)),
          };
        });
      } else {
        await updatePatient(patientId, (stored) => ({
          ...stored,
          scanSessions: (stored.scanSessions ?? []).map((s) => (s.id === sessionId ? updatedSession : s)),
        }));
      }
    } catch (reconErr) {
      console.warn("Auto-reconstruction note in package route:", reconErr);
      await updatePatient(patientId, (stored) => ({
        ...stored,
        scanSessions: (stored.scanSessions ?? []).map((s) => (s.id === sessionId ? updatedSession : s)),
      }));
    }

    return NextResponse.json({
      success: true,
      message: "Đã nạp gói dữ liệu TrueDepth Native từ thiết bị iOS thành công.",
      session: updatedSession,
      quality: qualityReport,
      studioUrl: `/patients/${patientId}/studio`,
    });
  } catch (error) {
    console.error("Error importing TrueDepth package:", error);
    return NextResponse.json({ error: "Lỗi hệ thống khi nạp TrueDepth package.", details: String(error) }, { status: 500 });
  }
}
