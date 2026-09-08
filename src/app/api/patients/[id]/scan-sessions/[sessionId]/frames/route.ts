import { randomUUID } from "crypto";
import { writeFile } from "fs/promises";
import path from "path";
import { NextRequest, NextResponse } from "next/server";
import { getPatient, updatePatient } from "@/lib/db";
import { scanFramesDir, ensureDir } from "@/lib/storage";
import type { ScanCaptureView, ScanFrame } from "@/lib/types";

const views: ScanCaptureView[] = ["front", "left_45", "left_profile", "right_45", "right_profile"];
const extensionByMime: Record<string, string> = { "image/jpeg": "jpg", "image/png": "png", "image/webp": "webp" };

function readJsonField(value: FormDataEntryValue | null, name: string): Record<string, unknown> | null {
  if (typeof value !== "string" || value.length === 0) return null;
  try {
    const parsed: unknown = JSON.parse(value);
    if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) throw new Error();
    return parsed as Record<string, unknown>;
  } catch {
    throw new Error(`${name} không phải JSON hợp lệ.`);
  }
}

function isFiniteNumber(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

/** Reject a browser-made or incomplete object before it can be labelled as
 * native evidence. ARFaceGeometry has 1,220 vertices and 2,304 triangles;
 * allow a narrow range for supported OS revisions but require the actual
 * arrays and face-to-camera transform needed for metric reconstruction. */
function validateNativeCapture(
  geometry: Record<string, unknown> | null,
  intrinsics: Record<string, unknown> | null,
): string | null {
  const vertices = geometry?.vertices;
  const triangles = geometry?.triangleIndices;
  const textureCoordinates = geometry?.textureCoordinates;
  const faceToCamera = geometry?.faceToCameraColumnMajor;
  if (!Array.isArray(vertices) || vertices.length < 3_300 || vertices.length > 4_200 || !vertices.every(isFiniteNumber)) {
    return "Geometry ARKit không có mảng vertices hợp lệ.";
  }
  if (!Array.isArray(triangles) || triangles.length < 6_000 || triangles.length % 3 !== 0 || !triangles.every((item) => Number.isInteger(item))) {
    return "Geometry ARKit không có topology tam giác hợp lệ.";
  }
  if (!Array.isArray(textureCoordinates) || textureCoordinates.length < 2_000 || !textureCoordinates.every(isFiniteNumber)) {
    return "Geometry ARKit không có UV hợp lệ.";
  }
  if (!Array.isArray(faceToCamera) || faceToCamera.length !== 16 || !faceToCamera.every(isFiniteNumber)) {
    return "Geometry ARKit thiếu pose face-to-camera của frame.";
  }
  if (!intrinsics || !isFiniteNumber(intrinsics.fx) || !isFiniteNumber(intrinsics.fy) || intrinsics.fx <= 0 || intrinsics.fy <= 0 ||
    !isFiniteNumber(intrinsics.cx) || !isFiniteNumber(intrinsics.cy) || !Number.isInteger(intrinsics.imageWidth) || !Number.isInteger(intrinsics.imageHeight)) {
    return "Camera intrinsics của frame không hợp lệ.";
  }
  return null;
}

export async function POST(request: NextRequest, { params }: { params: Promise<{ id: string; sessionId: string }> }) {
  const { id: patientId, sessionId } = await params;
  const patient = await getPatient(patientId);
  const current = patient?.scanSessions?.find((session) => session.id === sessionId);
  if (!patient || !current || current.patientId !== patientId) return NextResponse.json({ error: "Scan session không hợp lệ cho hồ sơ này" }, { status: 404 });
  if (current.status !== "uploading" && current.status !== "capturing") return NextResponse.json({ error: "Session chưa ở trạng thái nhận frame" }, { status: 409 });
  const formData = await request.formData();
  const file = formData.get("frame");
  const viewValue = formData.get("view");
  if (!(file instanceof File) || !views.includes(viewValue as ScanCaptureView)) return NextResponse.json({ error: "Thiếu frame hoặc góc scan không hợp lệ" }, { status: 400 });
  const ext = extensionByMime[file.type];
  if (!ext || file.size === 0 || file.size > 12 * 1024 * 1024) return NextResponse.json({ error: "Frame phải là ảnh JPG, PNG hoặc WebP không quá 12MB" }, { status: 400 });
  const view = viewValue as ScanCaptureView;
  let geometry: Record<string, unknown> | null;
  let intrinsics: Record<string, unknown> | null;
  try {
    geometry = readJsonField(formData.get("geometry"), "geometry");
    intrinsics = readJsonField(formData.get("intrinsics"), "intrinsics");
  } catch (error) {
    return NextResponse.json({ error: error instanceof Error ? error.message : "Native metadata không hợp lệ." }, { status: 400 });
  }
  if (current.scannerKind === "ios_native") {
    const nativeError = validateNativeCapture(geometry, intrinsics);
    if (nativeError) {
      return NextResponse.json({ error: `${nativeError} Không được hạ cấp thành ảnh 2D.` }, { status: 400 });
    }
  }
  const fileName = `${view}.${ext}`;
  const dir = scanFramesDir(patientId, sessionId);
  await ensureDir(dir);
  await writeFile(path.join(dir, fileName), Buffer.from(await file.arrayBuffer()));
  let geometryFileName: string | undefined;
  let intrinsicsFileName: string | undefined;
  if (geometry) {
    const geometryDir = path.join(dir, "geometry");
    await ensureDir(geometryDir);
    geometryFileName = `${view}_geometry.json`;
    await writeFile(path.join(geometryDir, geometryFileName), JSON.stringify(geometry));
  }
  if (intrinsics) {
    const cameraDir = path.join(dir, "camera");
    await ensureDir(cameraDir);
    intrinsicsFileName = `${view}_intrinsics.json`;
    await writeFile(path.join(cameraDir, intrinsicsFileName), JSON.stringify(intrinsics));
  }
  const depthAvailable = formData.get("depthAvailable") === "true" && Boolean(geometry);
  const frame: ScanFrame = {
    id: randomUUID(), view, fileName, capturedAt: new Date().toISOString(), byteSize: file.size, mimeType: file.type, depthAvailable,
    geometryFileName, intrinsicsFileName,
    poseMetadata: geometry ? { faceToCameraColumnMajor: geometry.faceToCameraColumnMajor, vertexCount: geometry.vertexCount, triangleCount: geometry.triangleCount } : undefined,
    cameraMetadata: intrinsics ?? undefined,
  };
  await updatePatient(patientId, (stored) => ({ ...stored, scanSessions: (stored.scanSessions ?? []).map((session) => session.id !== sessionId ? session : { ...session, updatedAt: new Date().toISOString(), frames: [...session.frames.filter((item) => item.view !== view), frame] }) }));
  return NextResponse.json({ frame, fileUrl: `/api/files/${patientId}/scans/${sessionId}/frames/${fileName}` }, { status: 201 });
}
