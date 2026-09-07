import { NextRequest, NextResponse } from "next/server";
import { getPatient, updatePatient } from "@/lib/db";
import { saveProfilePreviewImages } from "@/lib/scan/profile-preview-service";

/**
 * Small, dedicated endpoint (spec: "kiểm tra API hiện tại ... nếu chưa có
 * endpoint phù hợp thì tạo endpoint nhỏ, rõ ràng") for turning a completed
 * scan session's real accepted frames into the patient's 4-image profile
 * preview. Deliberately separate from the reconstruction PATCH action on
 * /scan-sessions/[sessionId] — this never touches reconstruction state, so
 * it can be re-run (e.g. to regenerate the preview) without re-running or
 * risking the reconstruction pipeline.
 */
export async function POST(request: NextRequest, { params }: { params: Promise<{ id: string }> }) {
  const { id: patientId } = await params;
  const patient = await getPatient(patientId);
  if (!patient) return NextResponse.json({ error: "Không tìm thấy hồ sơ bệnh nhân" }, { status: 404 });

  const body = await request.json().catch(() => ({}));
  const sessionId = typeof body.sessionId === "string" ? body.sessionId : undefined;
  const session = sessionId
    ? patient.scanSessions?.find((s) => s.id === sessionId)
    : patient.scanSessions?.slice().sort((a, b) => b.updatedAt.localeCompare(a.updatedAt))[0];

  if (!session) return NextResponse.json({ error: "Không tìm thấy scan session" }, { status: 404 });

  const result = await saveProfilePreviewImages(patientId, session.id, session.frames);

  // Đồng bộ ảnh scan vào danh mục ảnh hồ sơ của bệnh nhân
  const roleToAngle: Record<string, "angle1" | "angle2" | "angle3" | "angle4"> = {
    front: "angle1",
    three_quarter: "angle2",
    left: "angle3",
    right: "angle4",
  };

  const newPhotoEntries: Partial<Record<"angle1" | "angle2" | "angle3" | "angle4", { fileName: string; angle: "angle1" | "angle2" | "angle3" | "angle4"; uploadedAt: string; width: number; height: number }>> = {};
  for (const [role, img] of Object.entries(result.images)) {
    if (!img) continue;
    const angleKey = roleToAngle[role] || "angle1";
    newPhotoEntries[angleKey] = {
      fileName: `profile-preview/${img.fileName}`,
      angle: angleKey,
      uploadedAt: img.savedAt,
      width: 1080,
      height: 1440,
    };
  }

  await updatePatient(patientId, (stored) => {
    return {
      ...stored,
      photos: {
        ...(stored.photos || {}),
        ...newPhotoEntries,
      },
      profilePreview: { ...stored.profilePreview, ...result.images },
      model3d: stored.model3d || {
        before: {
          generatedAt: new Date().toISOString(),
          sourcePhotos: ["angle1", "angle2", "angle3"],
        },
      },
    };
  });

  return NextResponse.json({
    profilePreview: result.images,
    missingRoles: result.missingRoles,
    totalAcceptedFrames: result.totalAcceptedFrames,
  });
}
