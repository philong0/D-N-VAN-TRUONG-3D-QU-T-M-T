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
      timestamp?: number;
      yawDeg?: number;
      pitchDeg?: number;
      geometry?: { vertexCount: number; triangleCount: number };
      intrinsics?: Record<string, unknown>;
      pose?: Record<string, unknown>;
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
    const nativeTargets: Record<string, number> = {
      front: 0, left_45: -45, left_profile: -80, right_45: 45, right_profile: 80,
    };
    if (manifest.captureSource === "native_ios") {
      if (framesDTO.length !== Object.keys(nativeTargets).length) {
        return NextResponse.json({ error: "Gói TrueDepth phải có đủ 5 góc quét chuẩn." }, { status: 422 });
      }
      for (const frame of framesDTO) {
        const target = nativeTargets[frame.view];
        const quality = frame.quality;
        if (
          target === undefined ||
          typeof frame.yawDeg !== "number" || Math.abs(frame.yawDeg - target) > 10 ||
          typeof frame.pitchDeg !== "number" || Math.abs(frame.pitchDeg) > 10 ||
          !quality?.isTracked || !quality.isDistanceOptimal || !quality.isLightingAdequate || quality.isBlurry ||
          !frame.geometry || frame.geometry.vertexCount < 1200 || frame.geometry.triangleCount < 2300
        ) {
          return NextResponse.json({ error: `Frame ${frame.view} không đạt pose/quality TrueDepth; cần quét lại.` }, { status: 422 });
        }
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
      }

      let depthSaved = false;
      if (depthFile instanceof File) {
        depthSaved = true;
        const depthBuffer = Buffer.from(await depthFile.arrayBuffer());
        await writeFile(path.join(dir, `${view}_depth.raw`), depthBuffer);
      }

      const geometryDir = path.join(dir, "geometry");
      const cameraDir = path.join(dir, "camera");
      await ensureDir(geometryDir);
      await ensureDir(cameraDir);

      if (frameDTO.geometry) {
        await writeFile(path.join(geometryDir, `${view}_geometry.json`), JSON.stringify(frameDTO.geometry, null, 2));
      }
      if (frameDTO.intrinsics) {
        await writeFile(path.join(cameraDir, "intrinsics.json"), JSON.stringify(frameDTO.intrinsics, null, 2));
      }

      const scanFrame: ScanFrame = {
        id: randomUUID(),
        view,
        fileName: savedRgbName,
        capturedAt: new Date(frameDTO.timestamp ? frameDTO.timestamp * 1000 : Date.now()).toISOString(),
        byteSize,
        mimeType,
        depthAvailable: depthSaved || Boolean(manifest.hasTrueDepth),
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

    let updatedSession: ScanSession = {
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
