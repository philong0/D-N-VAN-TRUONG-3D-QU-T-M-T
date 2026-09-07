/**
 * profile-preview-service.ts
 *
 * Turns a completed scan session's real accepted frames into the patient's
 * 4-image profile preview: selects roles (profile-image-selection.ts),
 * copies the real source files into the patient's own profile-preview
 * folder (never mutates/crops them — verbatim copies, so no face
 * information is lost), and returns exactly what the caller should persist
 * onto `Patient.profilePreview`.
 *
 * Does not read or write patients.json itself — callers (the dedicated
 * profile-preview API route, and the existing reconstruction PATCH handler)
 * own that, so this stays a plain, file-system-only service.
 */
import { copyFile } from "fs/promises";
import path from "path";
import { ensureDir, patientDir, scanFramesDir } from "@/lib/storage";
import type { ScanFrame, ProfilePreviewImage, ProfileRole } from "@/lib/types";
import { buildBurstCandidates } from "./burst-candidates";
import { selectBestProfileImages } from "./profile-image-selection";

export function profilePreviewDir(patientId: string): string {
  return path.join(patientDir(patientId), "profile-preview");
}

export interface ProfilePreviewSaveResult {
  images: Partial<Record<ProfileRole, ProfilePreviewImage>>;
  missingRoles: ProfileRole[];
  totalAcceptedFrames: number;
}

/**
 * `frames` must be the session's full accepted-frame list (reconstruction
 * input — see reconstruction-frame-selection.ts's own module docstring for
 * why the two selections deliberately share this same real source data but
 * produce different, independent outputs).
 */
export async function saveProfilePreviewImages(
  patientId: string,
  sessionId: string,
  frames: ScanFrame[]
): Promise<ProfilePreviewSaveResult> {
  const candidates = buildBurstCandidates(frames);
  const { images, missingRoles } = selectBestProfileImages(candidates);

  const outDir = profilePreviewDir(patientId);
  await ensureDir(outDir);
  const srcDir = scanFramesDir(patientId, sessionId);

  const saved: Partial<Record<ProfileRole, ProfilePreviewImage>> = {};
  const savedAt = new Date().toISOString();

  for (const img of images) {
    const ext = path.extname(img.fileName) || ".jpg";
    const destFileName = `${img.role}${ext}`;
    await copyFile(path.join(srcDir, img.fileName), path.join(outDir, destFileName));
    saved[img.role] = {
      fileName: destFileName,
      yaw: img.yaw,
      pitch: img.pitch,
      roll: img.roll,
      qualityScore: img.qualityScore,
      timestamp: img.timestamp,
      sourceFrameId: img.sourceFrameId,
      savedAt,
    };
  }

  return { images: saved, missingRoles, totalAcceptedFrames: candidates.length };
}
