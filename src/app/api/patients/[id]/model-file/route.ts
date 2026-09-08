import { readFile, stat } from "fs/promises";
import path from "path";
import { NextRequest, NextResponse } from "next/server";
import { DATA_DIR } from "@/lib/storage";

export async function GET(
  _request: NextRequest,
  context: { params: Promise<{ id: string }> }
) {
  const { id: patientId } = await context.params;

  const candidatePaths = [
    path.join(process.cwd(), "public", "models", "patients", patientId, "reconstruction", "baseline.glb"),
    path.join(process.cwd(), ".data", "patients", patientId, "reconstruction", "baseline.glb"),
    path.join(process.cwd(), ".data", "patients", patientId, "models", "baseline.glb"),
    path.join(process.cwd(), ".data", "patients", patientId, "model.glb"),
    path.join(DATA_DIR, "patients", patientId, "models", "baseline.glb"),
    path.join(DATA_DIR, "patients", patientId, "reconstruction", "baseline.glb"),
    path.join(DATA_DIR, "patients", patientId, "model.glb"),
    path.join(process.cwd(), "public", "models", "patients", patientId, "baseline.glb"),
    path.join(process.cwd(), "public", "models", "patients", patientId, "model.glb"),
  ];

  for (const candidate of candidatePaths) {
    try {
      const s = await stat(candidate);
      if (s.size > 0) {
        const data = await readFile(candidate);
        return new NextResponse(new Uint8Array(data), {
          headers: {
            "Content-Type": "model/gltf-binary",
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
            "Content-Length": String(data.length),
            "Content-Disposition": `inline; filename="baseline-${patientId}.glb"`,
          },
        });
      }
    } catch {
      continue;
    }
  }

  return NextResponse.json(
    { error: "Chưa có mô hình 3D cho bệnh nhân này." },
    { status: 404, headers: { "Cache-Control": "no-cache, no-store, must-revalidate" } }
  );
}
