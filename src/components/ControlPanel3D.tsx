import { MORPH_SLIDERS, type MorphGroup } from "@/lib/services-catalog";
import { NEON_BADGE_BY_UNIT } from "@/lib/ui";
import type { MorphParams } from "@/lib/types";
import type { SkinToneConfig } from "@/components/Canvas3D";

interface ControlPanel3DProps {
  params: MorphParams;
  unlockedGroups: Set<MorphGroup>;
  selectedRegion: MorphGroup | null;
  onSelectRegion: (region: MorphGroup) => void;
  onChange: (group: MorphGroup, sliderKey: string, value: number) => void;
  skinTone?: SkinToneConfig;
  onSkinToneChange?: (newTone: SkinToneConfig) => void;
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
  skinTone,
  onSkinToneChange,
}: ControlPanel3DProps) {
  const groups = GROUP_ORDER.filter((g) => unlockedGroups.has(g));

  return (
    <div className="flex flex-col gap-3">
      {/* Tùy chỉnh tông màu da & ánh sáng */}
      {skinTone && onSkinToneChange && (
        <details
          open={true}
          className="rounded-lg border border-amber-500/50 bg-zinc-800/80 shadow-[0_0_15px_rgba(251,191,36,0.15)]"
        >
          <summary className="flex cursor-pointer list-none items-center justify-between px-3 py-2.5 text-sm font-semibold text-amber-300">
            <span className="flex items-center gap-1.5">
              <span>🎨</span> Tông Màu Da &amp; Ánh Sáng 3D
            </span>
            <span className="text-xs">▾</span>
          </summary>
          <div className="flex flex-col gap-3 px-3 pb-3 pt-1">
            {/* Quick presets */}
            <div>
              <span className="text-[10px] uppercase font-bold tracking-wider text-zinc-400 block mb-1.5">
                Tông màu mẫu 1-click
              </span>
              <div className="grid grid-cols-2 gap-1.5">
                {[
                  { label: "🌟 Chuẩn Gốc", config: { brightness: 1.10, warmth: 0, smoothness: 0.82 } },
                  { label: "🌸 Trắng Hồng", config: { brightness: 1.20, warmth: 6, smoothness: 0.86 } },
                  { label: "☀️ Vàng Ấm", config: { brightness: 1.15, warmth: 10, smoothness: 0.82 } },
                  { label: "💎 Sáng Y Khoa", config: { brightness: 1.28, warmth: -4, smoothness: 0.88 } },
                ].map((p) => (
                  <button
                    key={p.label}
                    type="button"
                    onClick={() => onSkinToneChange(p.config)}
                    className="rounded-md border border-zinc-700 bg-zinc-800 px-2 py-1 text-left text-[11px] font-medium text-zinc-200 hover:border-amber-400/60 hover:bg-zinc-700 transition"
                  >
                    {p.label}
                  </button>
                ))}
              </div>
            </div>

            {/* Sliders */}
            <div className="space-y-2.5 pt-1">
              <div>
                <div className="flex justify-between text-zinc-400 mb-1 text-xs">
                  <span>Độ sáng da</span>
                  <span className="font-mono text-amber-300 font-bold">{Math.round(skinTone.brightness * 100)}%</span>
                </div>
                <input
                  type="range"
                  min="0.80"
                  max="1.35"
                  step="0.02"
                  value={skinTone.brightness}
                  onChange={(e) => onSkinToneChange({ ...skinTone, brightness: Number(e.target.value) })}
                  className="w-full accent-amber-400"
                />
              </div>

              <div>
                <div className="flex justify-between text-zinc-400 mb-1 text-xs">
                  <span>Sắc độ ấm / hồng</span>
                  <span className="font-mono text-amber-300 font-bold">
                    {skinTone.warmth > 0 ? `+${skinTone.warmth}` : skinTone.warmth}
                  </span>
                </div>
                <input
                  type="range"
                  min="-20"
                  max="20"
                  step="1"
                  value={skinTone.warmth}
                  onChange={(e) => onSkinToneChange({ ...skinTone, warmth: Number(e.target.value) })}
                  className="w-full accent-amber-400"
                />
                <div className="flex justify-between text-[9px] text-zinc-500 mt-0.5">
                  <span>Mát sáng (-20)</span>
                  <span>Chuẩn (0)</span>
                  <span>Hồng ấm (+20)</span>
                </div>
              </div>

              <div>
                <div className="flex justify-between text-zinc-400 mb-1 text-xs">
                  <span>Độ mịn lì (Matte)</span>
                  <span className="font-mono text-amber-300 font-bold">{Math.round(skinTone.smoothness * 100)}%</span>
                </div>
                <input
                  type="range"
                  min="0.50"
                  max="0.98"
                  step="0.02"
                  value={skinTone.smoothness}
                  onChange={(e) => onSkinToneChange({ ...skinTone, smoothness: Number(e.target.value) })}
                  className="w-full accent-amber-400"
                />
              </div>
            </div>
          </div>
        </details>
      )}

      {groups.length === 0 && (
        <p className="text-sm text-zinc-400">
          Hồ sơ chưa đăng ký dịch vụ nào hỗ trợ mô phỏng 3D (Mũi / Mắt / Cằm / Ngực).
        </p>
      )}

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
