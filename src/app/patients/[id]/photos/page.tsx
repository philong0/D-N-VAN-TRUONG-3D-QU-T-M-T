import Link from "next/link";
import { notFound } from "next/navigation";
import WorkflowSteps from "@/components/WorkflowSteps";
import { getPatient } from "@/lib/db";
import { BUTTON_PRIMARY_CLASS } from "@/lib/ui";
import Dropzone3D from "./Dropzone3D";

export default async function PatientPhotosPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = await params;
  const patient = await getPatient(id);
  if (!patient) notFound();

  const has3DModel = Boolean(patient.model3d?.before);

  return (
    <div className="mx-auto w-full max-w-5xl px-6 py-10">
      <WorkflowSteps current={2} />

      <div className="mt-6 flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 mb-8">
        <div>
          <h1 className="text-2xl font-bold tracking-tight text-card-foreground">Bước 2 · Ảnh bệnh án &amp; Scan Session</h1>
          <p className="mt-1 text-sm text-muted">
            Bệnh nhân: <span className="font-semibold text-foreground">{patient.fullName}</span> · Thu ảnh y tế, tạo phiên capture và kiểm tra chất lượng trước reconstruction.
          </p>
        </div>

        {has3DModel && (
          <Link
            href={`/patients/${patient.id}/studio`}
            className={BUTTON_PRIMARY_CLASS + " py-3 px-6 text-sm font-bold shadow-lg flex items-center justify-center gap-2 shrink-0"}
          >
            <span>🧊</span>
            <span>VÀO 3D STUDIO TƯ VẤN VIP →</span>
          </Link>
        )}
      </div>

      {/* DROPZONE NHẬN FILE 3D THẬT */}
      <Dropzone3D patientId={patient.id} has3DModel={has3DModel} />
    </div>
  );
}
