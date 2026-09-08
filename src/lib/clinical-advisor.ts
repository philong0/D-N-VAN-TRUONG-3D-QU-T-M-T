import type { AIAssessment, ClinicalBaseline, ClinicalWarning, FacialThirds, MorphParams } from "./types";

/**
 * AI Clinical Advisor — a rule-based decision-support calculator, not a
 * validated diagnostic model. Facial-thirds are derived purely from the
 * slider deltas relative to baseline (not measured from the patient's real
 * photos via computer vision), and the safety checks compare slider deltas
 * against literature-reference thresholds plus doctor-entered measurements
 * (chestBaseWidthMm, softTissueThicknessCm). Every output here is an
 * estimate for clinician review, never an automated diagnosis.
 */

const IDEAL_THIRD = 33.33;

export function computeFacialThirds(morph: MorphParams): FacialThirds {
  const lowerShift = morph.chin.chinLengthMm * 0.9 + morph.chin.pogPositionMm * 0.4;
  const middleShift = morph.nose.heightMm * 0.5 + morph.nose.tipProjectionMm * 0.3;

  let lower = IDEAL_THIRD + lowerShift;
  let middle = IDEAL_THIRD + middleShift;
  let upper = 100 - lower - middle;

  // Keep values in a sane display range without letting one third collapse.
  lower = Math.max(15, Math.min(55, lower));
  middle = Math.max(15, Math.min(55, middle));
  upper = Math.max(15, Math.min(55, 100 - lower - middle));
  const total = upper + middle + lower;
  upper = (upper / total) * 100;
  middle = (middle / total) * 100;
  lower = (lower / total) * 100;

  const maxDeviation = Math.max(
    Math.abs(upper - IDEAL_THIRD),
    Math.abs(middle - IDEAL_THIRD),
    Math.abs(lower - IDEAL_THIRD)
  );

  return {
    upper: Number(upper.toFixed(1)),
    middle: Number(middle.toFixed(1)),
    lower: Number(lower.toFixed(1)),
    balanced: maxDeviation <= 3,
  };
}

const NASOLABIAL_RANGE: [number, number] = [95, 100];
const NASOFRONTAL_RANGE: [number, number] = [115, 130];
const NOSE_SKIN_TENSION_LIMIT_MM = 2.5;
const MIN_SAFE_SOFT_TISSUE_CM = 1;

export function evaluateSafety(
  morph: MorphParams,
  clinicalBaseline: ClinicalBaseline | undefined
): ClinicalWarning[] {
  const warnings: ClinicalWarning[] = [];

  if (morph.nose.heightMm > NOSE_SKIN_TENSION_LIMIT_MM) {
    warnings.push({
      code: "nose-skin-tension",
      severity: "danger",
      message: `Nâng sống mũi +${morph.nose.heightMm.toFixed(1)}mm vượt ngưỡng an toàn ${NOSE_SKIN_TENSION_LIMIT_MM}mm — nguy cơ căng da, lộ sụn/bóng đỏ đầu mũi.`,
    });
  }

  const [nlMin, nlMax] = NASOLABIAL_RANGE;
  if (morph.nose.nasolabialAngleDeg < nlMin || morph.nose.nasolabialAngleDeg > nlMax) {
    warnings.push({
      code: "nasolabial-out-of-range",
      severity: "warning",
      message: `Góc Mũi–Môi ${morph.nose.nasolabialAngleDeg}° nằm ngoài khoảng thẩm mỹ chuẩn ${nlMin}°–${nlMax}°.`,
    });
  }

  const [nfMin, nfMax] = NASOFRONTAL_RANGE;
  if (morph.nose.nasofrontalAngleDeg < nfMin || morph.nose.nasofrontalAngleDeg > nfMax) {
    warnings.push({
      code: "nasofrontal-out-of-range",
      severity: "warning",
      message: `Góc Trán–Mũi ${morph.nose.nasofrontalAngleDeg}° nằm ngoài khoảng thẩm mỹ chuẩn ${nfMin}°–${nfMax}°.`,
    });
  }

  const chestWidth = clinicalBaseline?.chestBaseWidthMm;
  if (chestWidth && morph.breast.baseWidthMm > chestWidth) {
    warnings.push({
      code: "implant-exceeds-chest",
      severity: "danger",
      message: `Base Width túi ngực ${morph.breast.baseWidthMm}mm vượt quá bề rộng khung ngực đo được ${chestWidth}mm — nguy cơ tràn lề / symmastia.`,
    });
  }

  const tissue = clinicalBaseline?.softTissueThicknessCm;
  if (tissue !== undefined && tissue < MIN_SAFE_SOFT_TISSUE_CM) {
    warnings.push({
      code: "thin-soft-tissue",
      severity: "danger",
      message: `Mô mềm che phủ ước tính ${tissue}cm (<${MIN_SAFE_SOFT_TISSUE_CM}cm) — nguy cơ co thắt bao xơ / sờ thấy viền túi.`,
    });
  }

  return warnings;
}

export function buildAssessment(
  morph: MorphParams,
  clinicalBaseline: ClinicalBaseline | undefined
): AIAssessment {
  const facialThirds = computeFacialThirds(morph);
  const warnings = evaluateSafety(morph, clinicalBaseline);

  const hasDanger = warnings.some((w) => w.severity === "danger");
  const summary = hasDanger
    ? "Có cảnh báo nguy cơ — cần bác sĩ chuyên khoa xem xét trước khi thực hiện."
    : warnings.length > 0
      ? "Một số chỉ số nằm ngoài khoảng thẩm mỹ chuẩn tham khảo, cân nhắc điều chỉnh."
      : "Các chỉ số nằm trong khoảng an toàn & thẩm mỹ chuẩn tham khảo.";

  return { facialThirds, warnings, summary };
}
