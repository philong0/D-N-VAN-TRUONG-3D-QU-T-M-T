import { buildAssessment } from "@/lib/clinical-advisor";
import type { ClinicalBaseline, MorphParams, WarningSeverity } from "@/lib/types";

interface ClinicalAdvisorPanelProps {
  morph: MorphParams;
  clinicalBaseline?: ClinicalBaseline;
}

const SEVERITY_STYLE: Record<WarningSeverity, string> = {
  info: "border-border bg-muted-bg text-foreground",
  warning: "border-warning/40 bg-warning/10 text-warning",
  danger: "border-danger/40 bg-danger/10 text-danger",
};

const SEVERITY_ICON: Record<WarningSeverity, string> = {
  info: "ℹ️",
  warning: "⚠️",
  danger: "⛔",
};

export default function ClinicalAdvisorPanel({ morph, clinicalBaseline }: ClinicalAdvisorPanelProps) {
  const assessment = buildAssessment(morph, clinicalBaseline);
  const { facialThirds, warnings, summary } = assessment;

  return (
    <div className="flex flex-col gap-4 rounded-lg border border-border p-4">
      <div>
        <h2 className="mb-1 text-sm font-semibold text-card-foreground">
          🤖 AI Clinical Advisor — Tỷ lệ Vàng 1/3 khuôn mặt
        </h2>
        <p className="text-xs text-muted">
          Ước tính real-time dựa trên các thông số bác sĩ điều chỉnh, không đo trực tiếp từ ảnh gốc.
        </p>
      </div>

      <div className="flex h-6 w-full overflow-hidden rounded-full border border-border">
        <div
          className="flex items-center justify-center bg-accent text-[10px] font-medium text-accent-foreground"
          style={{ width: `${facialThirds.upper}%` }}
        >
          {facialThirds.upper}%
        </div>
        <div
          className="flex items-center justify-center bg-muted text-[10px] font-medium text-white"
          style={{ width: `${facialThirds.middle}%` }}
        >
          {facialThirds.middle}%
        </div>
        <div
          className="flex items-center justify-center bg-success text-[10px] font-medium text-white"
          style={{ width: `${facialThirds.lower}%` }}
        >
          {facialThirds.lower}%
        </div>
      </div>
      <div className="flex justify-between text-[10px] text-muted">
        <span>Tầng trên</span>
        <span>Tầng giữa</span>
        <span>Tầng dưới</span>
      </div>
      <p className={`text-xs font-medium ${facialThirds.balanced ? "text-success" : "text-warning"}`}>
        {facialThirds.balanced ? "✓ Tỷ lệ 3 tầng mặt cân đối" : "△ Tỷ lệ 3 tầng mặt lệch khỏi chuẩn ~1/3"}
      </p>

      <div className="border-t border-border pt-3">
        <p className="mb-2 text-xs font-semibold text-card-foreground">Cảnh báo an toàn mô mềm</p>
        {warnings.length === 0 ? (
          <p className="rounded-md border border-success/40 bg-success/10 px-3 py-2 text-xs text-success">
            ✓ Không phát hiện cảnh báo với thông số hiện tại.
          </p>
        ) : (
          <ul className="flex flex-col gap-2">
            {warnings.map((w) => (
              <li
                key={w.code}
                className={`rounded-md border px-3 py-2 text-xs ${SEVERITY_STYLE[w.severity]}`}
              >
                {SEVERITY_ICON[w.severity]} {w.message}
              </li>
            ))}
          </ul>
        )}
      </div>

      <p className="border-t border-border pt-3 text-xs text-muted">{summary}</p>
    </div>
  );
}
