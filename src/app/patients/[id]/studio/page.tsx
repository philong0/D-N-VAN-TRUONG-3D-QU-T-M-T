import Link from "next/link";
import { notFound } from "next/navigation";
import WorkflowSteps from "@/components/WorkflowSteps";
import { getPatient } from "@/lib/db";
import { getSlotLabels } from "@/lib/photo-angles";
import StudioClient from "./StudioClient";

export default async function PatientStudioPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = await params;
  const patient = await getPatient(id);
  if (!patient) notFound();

  const hasModel = Boolean(
    patient.model3d?.before ||
    (patient.photos && Object.keys(patient.photos).length > 0) ||
    (patient.scanSessions && patient.scanSessions.length > 0) ||
    patient.profilePreview
  );

  return (
    <div className="mx-auto w-full max-w-6xl px-6 py-12">
      <WorkflowSteps current={4} />

      <h1 className="mt-6 text-2xl font-semibold tracking-tight text-card-foreground">
        Bước 4 · 3D Studio Simulation
      </h1>
      <p className="mt-1 text-sm text-muted">
        Bệnh nhân: <span className="font-medium text-foreground">{patient.fullName}</span> · Nắn chỉnh trực
        tiếp trên ảnh thật, theo dõi AI Clinical Advisor và lưu kết quả mô phỏng &ldquo;sau phẫu
        thuật&rdquo; vào bệnh án.
      </p>

      {!hasModel ? (
        <div className="mt-8 rounded-lg border border-warning/40 bg-warning/10 px-4 py-3 text-sm text-warning">
          Chưa khởi tạo mô hình 3D cho bệnh nhân này.{" "}
          <Link href={`/patients/${patient.id}/scan`} className="font-medium underline">
            Quay lại Quét 5 góc 3D
          </Link>{" "}
          hoặc{" "}
          <Link href={`/patients/${patient.id}/model`} className="font-medium underline">
            Khởi tạo từ ảnh
          </Link>
          .
        </div>
      ) : (
        <StudioClient
          patientId={patient.id}
          photos={patient.photos}
          slotLabels={getSlotLabels(patient.services)}
          services={patient.services}
          clinicalBaseline={patient.clinicalBaseline}
          initialParams={patient.simulation?.params}
          afterImageUrl={
            patient.simulation ? `/api/files/${patient.id}/${patient.simulation.afterImageFileName}` : null
          }
        />
      )}
    </div>
  );
}
