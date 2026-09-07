import fs from "fs/promises";
import path from "path";
import { imageSize } from "image-size";
import { NextRequest, NextResponse } from "next/server";
import { getPatient, updatePatient } from "@/lib/db";
import { evaluateScanQuality, evaluateBurstScanQuality, REQUIRED_SCAN_VIEWS } from "@/lib/scan/quality";
import { requestReconstruction } from "@/lib/scan/reconstruction-service";
import type { ScanSession, PhotoAngle } from "@/lib/types";

function withSession(patientId: string, sessionId: string, session: ScanSession) {
  if (session.patientId !== patientId) throw new Error("Scan session không thuộc hồ sơ này");
  return session;
}

export async function GET(_request: NextRequest, { params }: { params: Promise<{ id: string; sessionId: string }> }) {
  const { id, sessionId } = await params;
  const session = (await getPatient(id))?.scanSessions?.find((item) => item.id === sessionId);
  if (!session) return NextResponse.json({ error: "Không tìm thấy scan session" }, { status: 404 });
  return NextResponse.json({ session: withSession(id, sessionId, session) });
}

export async function PATCH(request: NextRequest, { params }: { params: Promise<{ id: string; sessionId: string }> }) {
  const { id: patientId, sessionId } = await params;
  const patient = await getPatient(patientId);
  if (!patient) return NextResponse.json({ error: "Không tìm thấy hồ sơ bệnh nhân" }, { status: 404 });
  const current = patient.scanSessions?.find((session) => session.id === sessionId);
  if (!current || current.patientId !== patientId) return NextResponse.json({ error: "Không tìm thấy scan session của hồ sơ này" }, { status: 404 });
  const body = await request.json().catch(() => ({}));
  const action = body.action as "start" | "uploading" | "finalize" | "fail" | "request_reconstruction";
  const next: ScanSession = { ...current, updatedAt: new Date().toISOString() };

  if (action === "start") next.status = "capturing";
  else if (action === "uploading") next.status = "uploading";
  else if (action === "fail") { next.status = "failed"; next.error = typeof body.error === "string" ? body.error : "Capture bị gián đoạn."; }
  else if (action === "finalize") {
    if (body.clientScanReport && typeof body.clientScanReport === "object") {
      next.clientScanReport = body.clientScanReport as Record<string, unknown>;
    }
    const isBurst = next.frames.some((f) => f.view === "burst");
    const quality = isBurst
      ? evaluateBurstScanQuality(next.frames)
      : evaluateScanQuality(next.frames, next.scannerKind);
    next.quality = quality;
    next.status = quality.overall === "fail" ? "needs_rescan" : "quality_check";
  } else if (action === "request_reconstruction") {
    if (next.status !== "quality_check" && next.status !== "needs_rescan") {
      return NextResponse.json({ error: "Cần hoàn thành quality check trước reconstruction" }, { status: 409 });
    }
    // D-rgbreconstruct — a burst web-camera session can never score better
    // than "warning" overall (evaluateBurstScanQuality honestly can't
    // verify per-region coverage from frame count alone, see quality.ts's
    // own D-noregionclaim), so gating reconstruction on "pass" specifically
    // would silently exclude every burst session regardless of real frame
    // count/density. Reject only a genuine "fail" (named-view missing a
    // required angle, or burst under MIN_BURST_FRAMES) — both scanner kinds
    // share this same bar; the real trustworthiness check for RGB-only
    // capture happens downstream in dense_correspondence.py, which returns
    // no mesh at all rather than a fabricated one when it doesn't have
    // enough real matched points.
    if (!next.quality || next.quality.overall === "fail") {
      return NextResponse.json({ error: "Quality check chưa đạt yêu cầu; cần quét lại trước khi tái tạo baseline 3D." }, { status: 422 });
    }
    next.status = "processing";
    const result = await requestReconstruction(next);
    if (result.status === "unavailable" || result.status === "failed") {
      next.status = "quality_check";
      next.reconstruction = { provider: result.provider, requestedAt: new Date().toISOString(), error: result.reason };
    } else if (result.status === "completed" && result.artifacts?.baselineModelFileName) {
      next.status = "ready";
      next.reconstruction = {
        provider: result.provider,
        requestedAt: new Date().toISOString(),
        baselineModelFileName: result.artifacts.baselineModelFileName,
      };

      // Synchronize scan frames to patient.photos gallery
      const photosDir = path.join(process.cwd(), ".data", "patients", patientId, "photos");
      await fs.mkdir(photosDir, { recursive: true });
      const framesDir = path.join(process.cwd(), ".data", "patients", patientId, "scans", sessionId, "frames");

      const frameMappings: Record<string, PhotoAngle> = {
        front: "angle1",
        FRONT: "angle1",
        angle1: "angle1",
        left_45: "angle2",
        LEFT45: "angle2",
        LEFT20: "angle2",
        angle2: "angle2",
        left_profile: "angle3",
        LEFT60: "angle3",
        angle3: "angle3",
        right_45: "angle4",
        RIGHT45: "angle4",
        RIGHT20: "angle4",
        angle4: "angle4",
      };

      const framePhotoEntries: Partial<Record<PhotoAngle, { fileName: string; angle: PhotoAngle; uploadedAt: string; width: number; height: number }>> = {};
      for (const frame of next.frames) {
        const targetTag = (frame.qualityMetadata?.originKey as string) || (frame.qualityMetadata?.targetId as string) || frame.view;
        const slotKey = frameMappings[targetTag] || frameMappings[frame.view];
        if (!slotKey || framePhotoEntries[slotKey]) continue;
        let width = 1280;
        let height = 720;
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
      }

      // Fallback: if any angle slot is still missing, fill from available burst frames in order
      const remainingSlots: PhotoAngle[] = (["angle1", "angle2", "angle3", "angle4"] as PhotoAngle[]).filter((s) => !framePhotoEntries[s]);
      if (remainingSlots.length > 0 && next.frames.length > 0) {
        const step = Math.max(1, Math.floor(next.frames.length / 4));
        for (let idx = 0; idx < remainingSlots.length; idx++) {
          const frameIdx = Math.min(next.frames.length - 1, idx * step);
          const f = next.frames[frameIdx];
          if (f) {
            framePhotoEntries[remainingSlots[idx]] = {
              fileName: f.fileName,
              angle: remainingSlots[idx],
              uploadedAt: f.capturedAt,
              width: 1280,
              height: 720,
            };
          }
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
              method: result.provider,
              coverageFraction: result.qualityReport?.coverage?.status === "pass" ? 1.0 : 0.85,
            },
          },
          scanSessions: (stored.scanSessions ?? []).map((session) => session.id === sessionId ? next : session),
        };
      });

      // Copy frame files to patient photos folder
      for (const frame of next.frames) {
        const srcPath = path.join(framesDir, frame.fileName);
        const destPath = path.join(photosDir, frame.fileName);
        try {
          await fs.copyFile(srcPath, destPath);
        } catch (copyErr) {
          console.warn("Failed to copy frame to photos dir:", copyErr);
        }
      }

      return NextResponse.json({ session: next, requiredViews: REQUIRED_SCAN_VIEWS });
    } else {
      next.reconstruction = { provider: result.provider, requestedAt: new Date().toISOString() };
    }
  } else return NextResponse.json({ error: "Hành động scan không hợp lệ" }, { status: 400 });

  await updatePatient(patientId, (stored) => ({ ...stored, scanSessions: (stored.scanSessions ?? []).map((session) => session.id === sessionId ? next : session) }));
  return NextResponse.json({ session: next, requiredViews: REQUIRED_SCAN_VIEWS });
}
