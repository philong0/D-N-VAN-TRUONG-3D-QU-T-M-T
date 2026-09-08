export const CARD_CLASS = "rounded-2xl border border-border bg-card shadow-[0_2px_12px_rgba(28,35,39,0.035)]";

export const INPUT_CLASS =
  "w-full rounded-xl border border-border bg-card px-3.5 py-3 text-base text-foreground outline-none placeholder:text-muted focus:border-accent focus:ring-2 focus:ring-accent/15";

export const BUTTON_PRIMARY_CLASS =
  "inline-flex items-center justify-center rounded-xl bg-accent px-5 py-3 text-sm font-semibold text-accent-foreground shadow-[var(--accent-glow)] transition hover:bg-accent-hover active:scale-[0.98] active:bg-accent-active disabled:opacity-60 disabled:shadow-none";

export const BUTTON_SECONDARY_CLASS =
  "inline-flex items-center justify-center rounded-xl border border-border bg-card px-5 py-3 text-sm font-semibold text-foreground transition hover:border-accent/40 hover:bg-accent/5 active:scale-[0.98] disabled:opacity-60";

export const BADGE_CLASS = "rounded-full px-3 py-1 text-xs font-medium";

/** Neon Đỏ Thương Hiệu value badge for clinical sliders (mm / °); unitless values stay neutral. */
export const NEON_BADGE_BY_UNIT: Record<string, string> = {
  mm: "border border-accent/50 bg-accent/10 text-accent shadow-[var(--accent-glow)]",
  "°": "border border-accent/50 bg-accent/10 text-accent shadow-[var(--accent-glow)]",
  default: "border border-border bg-muted-bg text-foreground",
};
