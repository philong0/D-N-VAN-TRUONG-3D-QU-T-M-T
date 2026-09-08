"use server";

import { imageSize } from "image-size";
import { redirect } from "next/navigation";
import { writeFile } from "fs/promises";
import { buildAssessment } from "@/lib/clinical-advisor";
import { isValidServiceKey } from "@/lib/services-catalog";
import { createPatient, getPatient, updatePatient } from "@/lib/db";
import { generateModelFromPhotos } from "@/lib/mesh-generator";
import { PHOTO_ANGLES } from "@/lib/photo-angles";
import { ensureDir, patientDir, patientPhotosDir } from "@/lib/storage";
import type { ClinicalBaseline, MorphParams, PhotoAngle } from "@/lib/types";

export async function createPatientAction(formData: FormData): Promise<void> {
  const fullName = String(formData.get("fullName") ?? "").trim();
  const phone = String(formData.get("phone") ?? "").trim();
  const address = String(formData.get("address") ?? "").trim();
  const allergies = String(formData.get("allergies") ?? "").trim();
  const underlyingConditions = String(formData.get("underlyingConditions") ?? "").trim();
  const surgicalHistory = String(formData.get("surgicalHistory") ?? "").trim();
  const services = formData.getAll("services").map(String).filter(isValidServiceKey);

  if (!fullName || !phone) {
    throw new Error("Họ tên và Số điện thoại là bắt buộc");
  }

  const patient = await createPatient({
    fullName,
    phone,
    address,
    allergies,
    underlyingConditions,
    surgicalHistory,
    services,
  });

  redirect(`/patients/${patient.id}/photos`);
}

const EXT_BY_MIME: Record<string, string> = {
  "image/jpeg": "jpg",
  "image/png": "png",
  "image/webp": "webp",
};

export async function uploadPhotosAction(
  patientId: string,
  formData: FormData
): Promise<void> {
  const patient = await getPatient(patientId);
  if (!patient) throw new Error("Không tìm thấy hồ sơ bệnh nhân");

  const dir = patientPhotosDir(patientId);
  await ensureDir(dir);

  const updates: Partial<Record<PhotoAngle, { fileName: string; width: number; height: number }>> = {};

  for (const angle of PHOTO_ANGLES) {
    const file = formData.get(angle);
    if (!(file instanceof File) || file.size === 0) continue;

    const ext = EXT_BY_MIME[file.type];
    if (!ext) throw new Error(`Định dạng ảnh không được hỗ trợ: ${file.type}`);

    const buffer = Buffer.from(await file.arrayBuffer());
    const fileName = `${angle}.${ext}`;
    await writeFile(`${dir}/${fileName}`, buffer);

    const { width, height } = imageSize(buffer);
    updates[angle] = { fileName, width, height };
  }

  if (Object.keys(updates).length === 0) {
    redirect(`/patients/${patientId}/photos`);
  }

  await updatePatient(patientId, (p) => {
    const photos = { ...p.photos };
    for (const angle of PHOTO_ANGLES) {
      const update = updates[angle];
      if (update) {
        photos[angle] = { angle, uploadedAt: new Date().toISOString(), ...update };
      }
    }
    const allUploaded = PHOTO_ANGLES.every((a) => photos[a]);
    return {
      ...p,
      photos,
      status: allUploaded && p.status === "moi-tao" ? "da-tai-anh" : p.status,
    };
  });

  redirect(`/patients/${patientId}/photos`);
}

export async function generateModelAction(patientId: string): Promise<void> {
  const patient = await getPatient(patientId);
  if (!patient) throw new Error("Không tìm thấy hồ sơ bệnh nhân");

  const model = await generateModelFromPhotos(patient);

  await updatePatient(patientId, (p) => ({
    ...p,
    model3d: { ...p.model3d, before: model },
    status: p.status === "da-mo-phong" ? p.status : "da-tao-mo-hinh",
  }));

  redirect(`/patients/${patientId}/model`);
}

export async function updateClinicalBaselineAction(
  patientId: string,
  formData: FormData
): Promise<void> {
  const patient = await getPatient(patientId);
  if (!patient) throw new Error("Không tìm thấy hồ sơ bệnh nhân");

  const chestRaw = formData.get("chestBaseWidthMm");
  const tissueRaw = formData.get("softTissueThicknessCm");
  const chestBaseWidthMm = chestRaw ? Number(chestRaw) : undefined;
  const softTissueThicknessCm = tissueRaw ? Number(tissueRaw) : undefined;

  const clinicalBaseline: ClinicalBaseline = {
    chestBaseWidthMm: Number.isFinite(chestBaseWidthMm) ? chestBaseWidthMm : undefined,
    softTissueThicknessCm: Number.isFinite(softTissueThicknessCm) ? softTissueThicknessCm : undefined,
    updatedAt: new Date().toISOString(),
  };

  await updatePatient(patientId, (p) => ({ ...p, clinicalBaseline }));

  redirect(`/patients/${patientId}/studio`);
}

export async function saveSimulationAction(
  patientId: string,
  formData: FormData
): Promise<void> {
  const patient = await getPatient(patientId);
  if (!patient) throw new Error("Không tìm thấy hồ sơ bệnh nhân");
  if (!patient.model3d?.before) throw new Error("Chưa có mô hình 3D cho hồ sơ này");

  const paramsRaw = formData.get("params");
  const snapshot = formData.get("snapshot");
  if (typeof paramsRaw !== "string" || !(snapshot instanceof File)) {
    throw new Error("Thiếu dữ liệu mô phỏng");
  }

  const params = JSON.parse(paramsRaw) as MorphParams;
  const aiAssessment = buildAssessment(params, patient.clinicalBaseline);

  const dir = patientDir(patientId);
  await ensureDir(dir);

  const afterImageFileName = "after.png";
  const buffer = Buffer.from(await snapshot.arrayBuffer());
  await writeFile(`${dir}/${afterImageFileName}`, buffer);

  await updatePatient(patientId, (p) => ({
    ...p,
    simulation: { savedAt: new Date().toISOString(), params, afterImageFileName, aiAssessment },
    status: "da-mo-phong",
  }));

  redirect(`/patients/${patientId}/report`);
}
