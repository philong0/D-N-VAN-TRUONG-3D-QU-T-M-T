import { writeFile } from "fs/promises";
import path from "path";
import { NextRequest, NextResponse } from "next/server";
import { getPatient } from "@/lib/db";
import { scanFramesDir, ensureDir } from "@/lib/storage";

/**
 * D-belowchin — dedicated upload for the new "BELOW" (chin-underside)
 * checkpoint, added ADDITIVELY alongside the existing generic
 * `/burst-frames` route (never modifies it). That route renames every
 * upload to `burst_NNNN.{ext}` and files it under the generic angle1-4
 * `frameMappings` contract (see the scan-sessions PATCH route) — this photo
 * needs its own real device-orientation sidecar, not just an image, and it
 * must land as the EXACT filenames `below.jpg` / `below_orientation.json`
 * that `ai-engine/reconstruct_gnm_fullhead.py` looks for directly, so it is
 * kept on its own simple, independent path instead of overloading the
 * existing burst contract with a new special case.
 */
export async function POST(request: NextRequest, { params }: { params: Promise<{ id: string; sessionId: string }> }) {
  try {
    const { id: patientId, sessionId } = await params;
    const patient = await getPatient(patientId);
    if (!patient) {
      return NextResponse.json({ error: "Không tìm thấy hồ sơ bệnh nhân" }, { status: 404 });
    }

    let formData: FormData;
    try {
      formData = await request.formData();
    } catch {
      return NextResponse.json({ error: "Body yêu cầu không phải multipart/form-data hợp lệ" }, { status: 400 });
    }

    const image = formData.get("image");
    const orientationRaw = formData.get("orientation");

    if (!(image instanceof File) || image.size === 0 || image.size > 12 * 1024 * 1024) {
      return NextResponse.json({ error: "Ảnh dưới cằm không hợp lệ" }, { status: 400 });
    }

    let orientation: Record<string, unknown> = { orientationAvailable: false };
    if (typeof orientationRaw === "string" && orientationRaw.length <= 8_192) {
      try {
        const parsed = JSON.parse(orientationRaw);
        if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) orientation = parsed as Record<string, unknown>;
      } catch {
        // Missing/invalid orientation is not an upload error -- the photo is
        // still saved; the reconstruction step will simply drop it from the
        // 3D texture bake rather than guess a pose (see chin_underside_anchor.py).
      }
    }

    const dir = scanFramesDir(patientId, sessionId);
    await ensureDir(dir);

    const buffer = Buffer.from(await image.arrayBuffer());
    await writeFile(path.join(dir, "below.jpg"), buffer);
    await writeFile(path.join(dir, "below_orientation.json"), JSON.stringify(orientation, null, 2));

    return NextResponse.json({ ok: true, orientationAvailable: Boolean(orientation.orientationAvailable) }, { status: 201 });
  } catch (err) {
    console.error("[below-frame] Error:", err);
    return NextResponse.json({ error: err instanceof Error ? err.message : "Lỗi lưu ảnh dưới cằm" }, { status: 400 });
  }
}
