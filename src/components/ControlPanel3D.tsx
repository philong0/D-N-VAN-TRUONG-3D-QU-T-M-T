import { MORPH_SLIDERS, type MorphGroup } from "@/lib/services-catalog";
import { NEON_BADGE_BY_UNIT } from "@/lib/ui";
import type { MorphParams } from "@/lib/types";

interface ControlPanel3DProps {
  params: MorphParams;
  unlockedGroups: Set<MorphGroup>;
  selectedRegion: MorphGroup | null;
  onSelectRegion: (region: MorphGroup) => void;
  onChange: (group: MorphGroup, sliderKey: string, value: number) => void;
}

const GROUP_ORDER: MorphGroup[] = ["nose", "eye", "chin", "breast"];

const OUT_OF_IDEAL_BADGE =
  "border border-warning/60 bg-warning/10 text-warning shadow-[0_0_10px_-2px_var(--warning)]";

export default function ControlPanel3D({
  params,
  unlockedGroups,
  selectedRegion,
  onSelectRegion,
  onChange,
}: ControlPanel3DProps) {
  const groups = GROUP_ORDER.filter((g) => unlockedGroups.has(g));

  if (groups.length === 0) {
    return (
      <p className="text-sm text-zinc-400">
        Hồ sơ chưa đăng ký dịch vụ nào hỗ trợ mô phỏng 3D (Mũi / Mắt / Cằm / Ngực).
      </p>
    );
  }

  return (
    <div className="flex flex-col gap-3">
      {groups.map((group) => {
        const isSelected = group === selectedRegion;
        return (
          <details
            key={group}
            open={isSelected}
            className={`rounded-lg border transition-colors ${
              isSelected ? "border-accent/60 bg-zinc-800 shadow-[var(--accent-glow)]" : "border-zinc-700 bg-zinc-800/50"
            }`}
          >
            <summary
              onClick={(e) => {
                // Always land on this group when opened via the native <summary> toggle too, not just the slider focus below.
                e.preventDefault();
                onSelectRegion(group);
              }}
              className={`flex cursor-pointer list-none items-center justify-between px-3 py-2.5 text-sm font-semibold ${
                isSelected ? "text-accent" : "text-zinc-100"
              }`}
            >
              {MORPH_SLIDERS[group].title}
              <span className={`text-xs transition-transform ${isSelected ? "rotate-180" : ""}`}>▾</span>
            </summary>
            <div className="flex flex-col gap-3 px-3 pb-3">
              {MORPH_SLIDERS[group].sliders.map((slider) => {
                const value = (params[group] as unknown as Record<string, number>)[slider.key];
                const outOfIdeal =
                  slider.idealRange && (value < slider.idealRange[0] || value > slider.idealRange[1]);
                const badgeClass = outOfIdeal
                  ? OUT_OF_IDEAL_BADGE
                  : (NEON_BADGE_BY_UNIT[slider.unit ?? "default"] ?? NEON_BADGE_BY_UNIT.default);
                return (
                  <div key={slider.key}>
                    <div className="mb-1.5 flex items-center justify-between gap-2 text-xs text-zinc-400">
                      <label htmlFor={`${group}-${slider.key}`}>
                        {slider.label}
                        {slider.idealRange && (
                          <span className="ml-1 text-zinc-500">
                            (chuẩn {slider.idealRange[0]}–{slider.idealRange[1]}
                            {slider.unit ?? ""})
                          </span>
                        )}
                      </label>
                      <span
                        className={`rounded-full px-2 py-0.5 font-mono text-[11px] font-semibold tabular-nums ${badgeClass}`}
                      >
                        {value > 0 && !slider.idealRange ? "+" : ""}
                        {value}
                        {slider.unit ?? ""}
                      </span>
                    </div>
                    <input
                      id={`${group}-${slider.key}`}
                      type="range"
                      min={slider.min}
                      max={slider.max}
                      step={slider.step}
                      value={value}
                      onChange={(e) => onChange(group, slider.key, Number(e.target.value))}
                      onFocus={() => onSelectRegion(group)}
                      className="w-full accent-accent"
                    />
                  </div>
                );
              })}
            </div>
          </details>
        );
      })}
    </div>
  );
}
