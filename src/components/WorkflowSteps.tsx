const STEPS = [
  { n: 1, label: "Chụp 4 Góc Chuẩn" },
  { n: 2, label: "Quét Mặt 3D (TrueDepth / ARKit)" },
  { n: 3, label: "3D Studio & Mô Phỏng Phẫu Thuật" },
];

export default function WorkflowSteps({ current }: { current: number }) {
  return (
    <ol className="flex flex-wrap items-center gap-1.5 sm:gap-2 text-xs sm:text-sm">
      {STEPS.map((step, i) => (
        <li key={step.n} className="flex items-center gap-1.5 sm:gap-2">
          <span
            className={`flex h-6 w-6 sm:h-7 sm:w-7 items-center justify-center rounded-full text-[11px] sm:text-xs font-semibold shrink-0 ${
              step.n === current
                ? "bg-accent text-accent-foreground"
                : step.n < current
                  ? "bg-accent/15 text-accent"
                  : "bg-muted-bg text-muted"
            }`}
          >
            {step.n}
          </span>
          <span className={`text-xs sm:text-sm ${step.n === current ? "font-medium text-card-foreground" : "text-muted"}`}>
            {step.label}
          </span>
          {i < STEPS.length - 1 && <span className="mx-0.5 sm:mx-1 text-border">→</span>}
        </li>
      ))}
    </ol>
  );
}
