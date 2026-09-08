import Link from "next/link";
import { notFound } from "next/navigation";
import WorkflowSteps from "@/components/WorkflowSteps";
import { getPatient } from "@/lib/db";
import { MORPH_SLIDERS, serviceLabel } from "@/lib/services-catalog";
import PrintButton from "./PrintButton";

const SEVERITY_LABEL: Record<string, string> = {
  info: "Thông tin",
  warning: "Cảnh báo",
  danger: "Nguy cơ",
};

export default async function PatientReportPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = await params;
  const patient = await getPatient(id);
  if (!patient) notFound();

  const simulation = patient.simulation;
  const frontPhoto = patient.photos.angle1;

  return (
    <div className="mx-auto w-full max-w-4xl px-6 py-12 print:max-w-none print:bg-white print:px-0 print:py-4 print:text-black">
      <div className="print:hidden">
        <WorkflowSteps current={4} />
      </div>

      <div className="mt-6 flex items-start justify-between gap-4 print:mt-0">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight text-card-foreground print:text-black">
            Báo cáo Trước / Sau mô phỏng
          </h1>
          <p className="mt-1 text-sm text-muted print:text-black">
            Bệnh nhân: <span className="font-medium text-foreground print:text-black">{patient.fullName}</span> · ĐT: {patient.phone}
          </p>
        </div>
        <PrintButton />
      </div>

      {!simulation ? (
        <div className="mt-8 rounded-lg border border-warning/40 bg-warning/10 px-4 py-3 text-sm text-warning print:hidden">
          Chưa có kết quả mô phỏng nào được lưu.{" "}
          <Link href={`/patients/${patient.id}/studio`} className="font-medium underline">
            Vào 3D Studio để mô phỏng
          </Link>
          .
        </div>
      ) : (
        <>
          <div className="mt-8 grid grid-cols-1 gap-6 sm:grid-cols-2">
            <div>
              <h2 className="mb-2 text-sm font-semibold text-accent/60 print:text-black">TRƯỚC PHẪU THUẬT</h2>
              <div className="aspect-square w-full overflow-hidden rounded-xl border border-accent/15 bg-background print:border-black">
                {frontPhoto ? (
                  // eslint-disable-next-line @next/next/no-img-element
                  <img
                    src={`/api/files/${patient.id}/photos/${frontPhoto.fileName}`}
                    alt="Trước phẫu thuật"
                    className="h-full w-full object-cover"
                  />
                ) : null}
              </div>
            </div>
            <div>
              <h2 className="mb-2 text-sm font-semibold text-accent print:text-black">SAU MÔ PHỎNG 3D</h2>
              <div className="aspect-square w-full overflow-hidden rounded-xl border border-accent/50 bg-background shadow-[var(--accent-glow)] print:border-black print:shadow-none">
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                  src={`/api/files/${patient.id}/${simulation.afterImageFileName}`}
                  alt="Sau mô phỏng 3D"
                  className="h-full w-full object-cover"
                />
              </div>
            </div>
          </div>

          <div className="mt-8">
            <h2 className="mb-2 text-sm font-semibold text-card-foreground print:text-black">
              AI Clinical Advisor — Đánh giá
            </h2>
            <div className="rounded-lg border border-border bg-card p-4 print:border-black print:bg-white">
              <div className="flex h-5 w-full overflow-hidden rounded-full border border-border print:border-black">
                <div
                  className="bg-accent print:bg-zinc-300"
                  style={{ width: `${simulation.aiAssessment.facialThirds.upper}%` }}
                />
                <div
                  className="bg-muted print:bg-zinc-400"
                  style={{ width: `${simulation.aiAssessment.facialThirds.middle}%` }}
                />
                <div
                  className="bg-success print:bg-zinc-500"
                  style={{ width: `${simulation.aiAssessment.facialThirds.lower}%` }}
                />
              </div>
              <p className="mt-1 text-xs text-muted print:text-black">
                Tỷ lệ 3 tầng mặt: {simulation.aiAssessment.facialThirds.upper}% / {simulation.aiAssessment.facialThirds.middle}% / {simulation.aiAssessment.facialThirds.lower}% —{" "}
                {simulation.aiAssessment.facialThirds.balanced ? "cân đối" : "lệch khỏi chuẩn ~1/3"}
              </p>

              {simulation.aiAssessment.warnings.length > 0 && (
                <ul className="mt-3 flex flex-col gap-1.5 text-xs">
                  {simulation.aiAssessment.warnings.map((w) => (
                    <li key={w.code} className="text-foreground print:text-black">
                      [{SEVERITY_LABEL[w.severity]}] {w.message}
                    </li>
                  ))}
                </ul>
              )}

              <p className="mt-3 border-t border-border pt-3 text-xs font-medium text-foreground print:border-black print:text-black">
                {simulation.aiAssessment.summary}
              </p>
            </div>
          </div>

          <div className="mt-8 grid grid-cols-1 gap-6 sm:grid-cols-2">
            <div>
              <h2 className="mb-2 text-sm font-semibold text-card-foreground print:text-black">
                Dịch vụ đăng ký
              </h2>
              <ul className="list-inside list-disc text-sm text-foreground print:text-black">
                {patient.services.length === 0 && <li>Không có</li>}
                {patient.services.map((s) => (
                  <li key={s}>{serviceLabel(s)}</li>
                ))}
              </ul>
            </div>
            <div>
              <h2 className="mb-2 text-sm font-semibold text-card-foreground print:text-black">
                Thông số mô phỏng
              </h2>
              <dl className="space-y-1 text-sm text-foreground print:text-black">
                {(["nose", "eye", "chin", "breast"] as const).map((group) =>
                  MORPH_SLIDERS[group].sliders.map((slider) => {
                    const value = (simulation.params[group] as unknown as Record<string, number>)[slider.key];
                    return (
                      <div key={`${group}-${slider.key}`} className="flex justify-between gap-4">
                        <dt>{slider.label}</dt>
                        <dd>
                          {value}
                          {slider.unit ?? ""}
                        </dd>
                      </div>
                    );
                  })
                )}
              </dl>
            </div>
          </div>

          <div className="mt-6 grid grid-cols-1 gap-4 sm:grid-cols-3">
            {patient.allergies && (
              <div>
                <h2 className="mb-1 text-sm font-semibold text-card-foreground print:text-black">
                  Tiền sử dị ứng
                </h2>
                <p className="text-sm text-foreground print:text-black">{patient.allergies}</p>
              </div>
            )}
            {patient.underlyingConditions && (
              <div>
                <h2 className="mb-1 text-sm font-semibold text-card-foreground print:text-black">
                  Bệnh lý nền
                </h2>
                <p className="text-sm text-foreground print:text-black">{patient.underlyingConditions}</p>
              </div>
            )}
            {patient.surgicalHistory && (
              <div>
                <h2 className="mb-1 text-sm font-semibold text-card-foreground print:text-black">
                  Tiền sử phẫu thuật
                </h2>
                <p className="text-sm text-foreground print:text-black">{patient.surgicalHistory}</p>
              </div>
            )}
          </div>

          <p className="mt-8 text-xs text-muted print:text-black">
            Lưu lúc: {new Date(simulation.savedAt).toLocaleString("vi-VN")} · Báo cáo do công cụ mô phỏng
            &amp; đánh giá hỗ trợ AI tạo ra, chỉ mang tính tham khảo — không thay thế chẩn đoán/thăm khám
            trực tiếp bởi bác sĩ chuyên khoa.
          </p>
        </>
      )}
    </div>
  );
}
