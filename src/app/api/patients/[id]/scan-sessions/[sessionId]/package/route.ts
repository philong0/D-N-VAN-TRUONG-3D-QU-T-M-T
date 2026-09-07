import { randomUUID } from "crypto";
import { writeFile } from "fs/promises";
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

    const manifest = JSON.parse(manifestStr);
    const framesDTO: any[] = manifest.frames || [];
    const savedFrames: ScanFrame[] = [];

    for (const frameDTO of framesDTO) {
      const view: ScanCaptureView = frameDTO.view;
      const rgbFile = formData.get(frameDTO.rgbFileName || `${view}_rgb.jpg`) as File | null;
      const depthFile = formData.get(frameDTO.depthFileName || `${view}_depth.raw`) as File | null;

      let savedRgbName = `${view}.jpg`;
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

    // Auto-trigger 3D Reconstruction for seamless 1-step scanning
    if (qualityReport.overall !== "fail") {
      scanSessions: (stored.scanSessions ?? []).map((s) => (s.id === sessionId ? updatedSession : s)),
    }));

    return NextResponse.json({
      success: true,
      message: "Đã nạp gói dữ liệu TrueDepth Native từ thiết bị iOS thành công.",
      session: updatedSession,
      quality: qualityReport,
      studioUrl: `/patients/${patientId}/studio`,
    });
    console.error("Error importing TrueDepth package:", error);
    return NextResponse.json({ error: "Lỗi hệ thống khi nạp TrueDepth package.", details: String(error) }, { status: 500 });
  }
}
