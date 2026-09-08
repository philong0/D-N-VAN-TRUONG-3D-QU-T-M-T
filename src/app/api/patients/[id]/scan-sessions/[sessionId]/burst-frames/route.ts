import { randomUUID } from "crypto";
import { writeFile } from "fs/promises";
import path from "path";
import { NextRequest, NextResponse } from "next/server";
import { getPatient, updatePatient } from "@/lib/db";
import { evaluateBurstScanQuality } from "@/lib/scan/quality";
import { scanFramesDir, ensureDir } from "@/lib/storage";
import type { ScanFrame } from "@/lib/types";

const extensionByMime: Record<string, string> = { "image/jpeg": "jpg", "image/png": "png", "image/webp": "webp" };

/**
 * D-bulkburst — accepts MANY frames in one multipart request (a continuous
 * head-turn capture produces 60-100+ frames; the existing /frames route
 * only ever accepts one file per POST, keyed by a fixed view name, and
 * OVERWRITES any existing frame under that name — neither property works
 * for a burst sequence). Frames are named `burst_{sequenceIndex}.{ext}`
 * and APPENDED to the session's frame list, never overwritten, so repeated
 * uploads from a retried/resumed capture accumulate rather than clobber.
 */
export async function POST(request: NextRequest, { params }: { params: Promise<{ id: string; sessionId: string }> }) {
  const { id: patientId, sessionId } = await params;
  const patient = await getPatient(patientId);
  const current = patient?.scanSessions?.find((session) => session.id === sessionId);
  if (!patient || !current || current.patientId !== patientId) {
    return NextResponse.json({ error: "Scan session không hợp lệ cho hồ sơ này" }, { status: 404 });
  }
  if (current.status !== "uploading" && current.status !== "capturing") {
    return NextResponse.json({ error: "Session chưa ở trạng thái nhận frame" }, { status: 409 });
  }

  const formData = await request.formData();
  const files = formData.getAll("frames");
  const metadataItems = formData.getAll("metadata");
  if (files.length === 0) {
    return NextResponse.json({ error: "Không có frame nào được gửi lên" }, { status: 400 });
  }

  const dir = scanFramesDir(patientId, sessionId);
  await ensureDir(dir);

  const existingBurstCount = current.frames.filter((f) => f.view === "burst").length;
  const newFrames: ScanFrame[] = [];

  for (let i = 0; i < files.length; i++) {
    const file = files[i];
    if (!(file instanceof File)) continue;
    const ext = extensionByMime[file.type];
    if (!ext || file.size === 0 || file.size > 12 * 1024 * 1024) continue;

    const sequenceIndex = existingBurstCount + newFrames.length;
    const fileName = `burst_${String(sequenceIndex).padStart(4, "0")}.${ext}`;
    const buffer = Buffer.from(await file.arrayBuffer());
    await writeFile(path.join(dir, fileName), buffer);

    let qualityMetadata: Record<string, unknown> | undefined;
    const rawMetadata = metadataItems[i];
    if (typeof rawMetadata === "string" && rawMetadata.length <= 8_192) {
      try {
        const candidate = JSON.parse(rawMetadata);
        if (candidate && typeof candidate === "object" && !Array.isArray(candidate)) qualityMetadata = candidate as Record<string, unknown>;
      } catch {
        // Metadata is optional; never invent it when a browser cannot supply it.
      }
    }

    newFrames.push({
      id: randomUUID(),
      view: "burst",
      sequenceIndex,
      fileName,
      capturedAt: new Date().toISOString(),
      byteSize: file.size,
      mimeType: file.type,
      depthAvailable: false,
      qualityMetadata,
    });
  }

  if (newFrames.length === 0) {
    return NextResponse.json({ error: "Không có frame hợp lệ nào trong yêu cầu (định dạng/dung lượng không hợp lệ)" }, { status: 400 });
  }

  const allFrames = [...current.frames, ...newFrames];
  const quality = evaluateBurstScanQuality(allFrames);

  await updatePatient(patientId, (stored) => ({
    ...stored,
    scanSessions: (stored.scanSessions ?? []).map((s) =>
      s.id !== sessionId
        ? s
        : { ...s, scannerKind: "web_camera", frames: allFrames, quality, updatedAt: new Date().toISOString() }
    ),
  }));

  return NextResponse.json({ savedCount: newFrames.length, totalBurstFrames: existingBurstCount + newFrames.length, quality }, { status: 201 });
}
