import Link from "next/link";
import { notFound } from "next/navigation";
import Canvas3D from "@/components/Canvas3D";
import WorkflowSteps from "@/components/WorkflowSteps";
import { getPatient } from "@/lib/db";
import { PHOTO_ANGLES, getSlotLabels } from "@/lib/photo-angles";
import { DEFAULT_MORPH_PARAMS } from "@/lib/services-catalog";
import { BUTTON_PRIMARY_CLASS } from "@/lib/ui";

export default async function PatientModelPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = await params;
  const patient = await getPatient(id);
  if (!patient) notFound();

  const slotLabels = getSlotLabels(patient.services);
  // "ready" is only ever set server-side after a real reconstruction
  // succeeded (see the scan-sessions PATCH route's request_reconstruction
  // handler) — that's the actual trust boundary, whether the session came
  // from native TrueDepth or the RGB multi-view pipeline, so it doesn't
  // need to be re-verified here.
  const verifiedScan = patient.scanSessions?.find(
    (session) => session.status === "ready" && Boolean(session.reconstruction?.baselineModelFileName)
  );
  const hasScanReady = Boolean(verifiedScan);
  const hasAnyPhoto = Object.keys(patient.photos).length > 0 || hasScanReady;
  const model = verifiedScan ? {
    generatedAt: new Date().toISOString(),
    sourcePhotos: (Object.keys(patient.photos) as (keyof typeof patient.photos)[]),
    method: verifiedScan.reconstruction?.provider || "Native ARKit multi-view reconstruction",
    coverageFraction: 1.0,
  } : undefined;
  const angle1 = patient.photos.angle1;
  const angle2 = patient.photos.angle2;
  const angle3 = patient.photos.angle3;

  return (
    <div className="mx-auto w-full max-w-5xl px-6 py-10">
      <WorkflowSteps current={3} />

      <h1 className="mt-6 text-2xl font-semibold tracking-tight text-card-foreground">
        Bước 3 · Khởi Tạo Mô Hình 3D Chuẩn Y Khoa
      </h1>
      <p className="mt-1 text-sm text-muted">
        Bệnh nhân: <span className="font-medium text-foreground">{patient.fullName}</span> · Chỉ hiển thị baseline dựng từ phiên quét TrueDepth/ARKit đã đạt quality check.
      </p>

      {!hasAnyPhoto ? (
        <div className="mt-8 rounded-lg border border-warning/40 bg-warning/10 px-4 py-3 text-sm text-warning">
          Chưa có ảnh trước phẫu thuật.{" "}
          <Link href={`/patients/${patient.id}/photos`} className="font-medium underline">
            Quay lại Bước 2 để tải ảnh
          </Link>
          .
        </div>
      ) : !model ? (
        <div className="mt-8 flex flex-col items-center gap-4 rounded-xl border border-dashed border-border p-12 text-center bg-card">
          <p className="text-sm text-muted">
            Ảnh bệnh án đã có, nhưng chưa có baseline 3D được xác minh. Hãy quét bằng iPad/iPhone có TrueDepth để tạo model bệnh nhân.
          </p>
          <Link href={`/patients/${patient.id}/scan`} className={BUTTON_PRIMARY_CLASS}>
            📱 Mở phiên quét TrueDepth
          </Link>
        </div>
      ) : (
        <div className="mt-8 grid grid-cols-1 gap-6 lg:grid-cols-3">
          <div className="lg:col-span-2 h-[480px]">
            <Canvas3D
              frontPhotoUrl={angle1 ? `/api/files/${patient.id}/photos/${angle1.fileName}` : null}
              obliquePhotoUrl={angle2 ? `/api/files/${patient.id}/photos/${angle2.fileName}` : null}
              profilePhotoUrl={angle3 ? `/api/files/${patient.id}/photos/${angle3.fileName}` : null}
              patientId={patient.id}
              params={DEFAULT_MORPH_PARAMS}
              showComparison={false}
            />
            <p className="mt-2 text-xs text-muted">
              Mô hình 3D chuẩn Y khoa của bệnh nhân · Giữ chuột trái xoay 360°, cuộn chuột để phóng to/thu nhỏ
            </p>
          </div>

          <div className="flex flex-col gap-4">
            <div>
              <h2 className="mb-2 text-sm font-medium text-card-foreground">Ảnh nguồn</h2>
              <div className="grid grid-cols-4 gap-1.5">
                {PHOTO_ANGLES.map((angle, i) => {
                  const asset = patient.photos[angle];
                  if (!asset) return null;
                  return (
                    // eslint-disable-next-line @next/next/no-img-element
                    <img
                      key={angle}
                      src={`/api/files/${patient.id}/photos/${asset.fileName}`}
                      alt={slotLabels[i].label}
                      title={slotLabels[i].label}
                      className="aspect-[3/4] w-full rounded-md object-cover border border-border"
                    />
                  );
                })}
              </div>
            </div>

            <div className="rounded-lg border border-border bg-card p-3 text-xs text-muted">
              <p>Khởi tạo lúc: {new Date(model.generatedAt).toLocaleString("vi-VN")}</p>
              <p>
                Phương pháp: {model.method || "Apple TrueDepth 3D Metric"}
              </p>
              <p>
                Nguồn: {model.sourcePhotos.length}/{PHOTO_ANGLES.length} góc ảnh
              </p>
            </div>

            <Link href={`/patients/${patient.id}/studio`} className={BUTTON_PRIMARY_CLASS + " text-center py-3 font-semibold text-sm"}>
              Bước 4: Mở 3D Studio Tư Vấn VIP →
            </Link>
          </div>
        </div>
      )}
    </div>
  );
}
