import Link from "next/link";
import { listPatients } from "@/lib/db";
import { BADGE_CLASS, CARD_CLASS } from "@/lib/ui";

const SEVERITY_BADGE: Record<string, string> = {
  danger: "bg-danger/15 text-danger",
  warning: "bg-warning/15 text-warning",
  ok: "bg-success/15 text-success",
};

export default async function ReportsPage() {
  const patients = await listPatients();
  const withReports = patients.filter((p) => p.simulation);

  return (
    <div className="mx-auto w-full max-w-4xl px-6 py-12">
      <h1 className="text-2xl font-semibold tracking-tight text-card-foreground">📊 Báo cáo & AI Advisor</h1>
      <p className="mt-1 text-sm text-muted">
        Danh sách hồ sơ đã lưu kết quả mô phỏng 3D kèm đánh giá AI Clinical Advisor.
      </p>

      {withReports.length === 0 ? (
        <div className={CARD_CLASS + " mt-8 border-dashed p-12 text-center text-sm text-muted"}>
          Chưa có báo cáo mô phỏng nào được lưu.
        </div>
      ) : (
        <ul className={CARD_CLASS + " mt-8 divide-y divide-border"}>
          {withReports.map((p) => {
            const warnings = p.simulation!.aiAssessment.warnings;
            const hasDanger = warnings.some((w) => w.severity === "danger");
            const hasWarning = warnings.some((w) => w.severity === "warning");
            const level = hasDanger ? "danger" : hasWarning ? "warning" : "ok";
            const levelLabel = hasDanger ? `${warnings.length} cảnh báo nguy cơ` : hasWarning ? `${warnings.length} cảnh báo` : "An toàn";

            return (
              <li key={p.id}>
                <Link
                  href={`/patients/${p.id}/report`}
                  className="flex flex-col gap-2 p-4 hover:bg-accent/5 sm:flex-row sm:items-center sm:justify-between"
                >
                  <div>
                    <p className="font-medium text-card-foreground">{p.fullName}</p>
                    <p className="text-sm text-muted">{p.simulation!.aiAssessment.summary}</p>
                  </div>
                  <div className="flex shrink-0 items-center gap-2">
                    <span className={BADGE_CLASS + " " + SEVERITY_BADGE[level]}>{levelLabel}</span>
                    <span className="text-xs text-muted">
                      {new Date(p.simulation!.savedAt).toLocaleDateString("vi-VN")}
                    </span>
                  </div>
                </Link>
              </li>
            );
          })}
        </ul>
      )}
    </div>
  );
}
