import type { FieldWindowAggregation } from "@/lib/dataset/nativeFieldWindow";

export interface SignedFieldStats {
  min: number;
  max: number;
  p2: number;
  p98: number;
  /** Whole-field upper tail used to bound sparse volume occupancy. */
  p995?: number;
  /** 98th percentile conditioned on positive samples, when available. */
  positiveP98?: number;
  /** 2nd percentile conditioned on negative samples, when available. */
  negativeP02?: number;
}

/**
 * Mixing ratio marking the presence of condensate, in kg/kg (0.1 g/kg). The conventional
 * cloud-outline value, and the floor the rendered-supercell cloud body already uses.
 */
const HYDROMETEOR_PRESENCE_KG_KG = 1.0e-4;

/**
 * Physically meaningful isosurface values, in each field's own unit.
 *
 * Condensate is listed because it is sparse: most of a domain holds none, so a
 * threshold derived from the value range lands near the peak and draws the densest
 * core rather than the cloud body. Water vapour is deliberately absent — it is
 * present everywhere, so a range-derived default describes it well.
 */
const PHYSICAL_ISOSURFACE_THRESHOLDS: Readonly<Record<string, number>> = {
  w: 10.0,
  vorticity_magnitude: 0.1,
  vorticity_z: 0.1,
  qc: HYDROMETEOR_PRESENCE_KG_KG,
  qr: HYDROMETEOR_PRESENCE_KG_KG,
  qi: HYDROMETEOR_PRESENCE_KG_KG,
  qs: HYDROMETEOR_PRESENCE_KG_KG,
  qg: HYDROMETEOR_PRESENCE_KG_KG,
  qh: HYDROMETEOR_PRESENCE_KG_KG,
  cloud_condensate: HYDROMETEOR_PRESENCE_KG_KG,
  total_condensate: HYDROMETEOR_PRESENCE_KG_KG,
  cloud_liquid_condensate: HYDROMETEOR_PRESENCE_KG_KG,
  cloud_frozen_condensate: HYDROMETEOR_PRESENCE_KG_KG,
};

/**
 * The documented default isosurface value for a field, or undefined to derive one from the data.
 *
 * Own-property tested: field names include user-authored derived fields, and a bare index
 * would answer names like "constructor" with an inherited function.
 */
export function physicalIsosurfaceThreshold(fieldName: string | null): number | undefined {
  if (!fieldName) return undefined;
  return Object.hasOwn(PHYSICAL_ISOSURFACE_THRESHOLDS, fieldName)
    ? PHYSICAL_ISOSURFACE_THRESHOLDS[fieldName]
    : undefined;
}

/** Fields whose sign carries physically meaningful rotation sense (vorticity + budget tendency terms). */
export function defaultsToSignedPair(fieldName: string | null): boolean {
  if (!fieldName) return false;
  return fieldName === "vorticity_z"
    || fieldName === "vorticity_r"
    || fieldName === "vorticity_theta"
    || fieldName === "horizontal_vorticity_streamwise"
    || fieldName === "horizontal_vorticity_crosswise"
    || fieldName === "stretching_term"
    || fieldName === "tilting_term"
    || fieldName === "baroclinic_term";
}

/** A paired contour is meaningful only when both signs exist in the frame. */
export function supportsSignedPair(stats: SignedFieldStats | null): boolean {
  return Boolean(stats && stats.min < 0 && stats.max > 0);
}

/** Robust absolute scale for a shared +T/-T contour control. */
export function signedMagnitudeScale(stats: SignedFieldStats): number {
  const robust = Math.max(Math.abs(stats.p2), Math.abs(stats.p98));
  const full = Math.max(Math.abs(stats.min), Math.abs(stats.max));
  return robust > 0 && Number.isFinite(robust) ? robust : full;
}

export function signedThresholdRange(stats: SignedFieldStats): {
  min: number;
  max: number;
  step: number;
} {
  const scale = signedMagnitudeScale(stats);
  if (!(scale > 0)) return { min: 0, max: 1, step: 0.01 };
  return {
    min: scale * 0.02,
    max: scale,
    step: scale / 200,
  };
}

/** Okubo-Weiss: W<0 is rotation-dominated (coherent core); W>0 is strain-dominated. */
export function defaultsToRotationDominatedSurface(fieldName: string | null): boolean {
  return fieldName === "okubo_weiss" || fieldName === "lambda2_criterion";
}

export function rotationDominatedThresholdRange(stats: SignedFieldStats): {
  min: number;
  max: number;
  step: number;
} {
  const robustNegative = stats.p2 < 0 && Number.isFinite(stats.p2)
    ? Math.abs(stats.p2)
    : Math.abs(Math.min(0, stats.min));
  const scale = robustNegative > 0 ? robustNegative : Math.max(1e-6, Math.abs(stats.min));
  return {
    min: -scale,
    max: -scale * 0.02,
    step: scale / 200,
  };
}

export function initialRotationDominatedThreshold(stats: SignedFieldStats): number {
  const range = rotationDominatedThresholdRange(stats);
  return range.min * 0.35;
}

export function defaultsToPositiveQSurface(fieldName: string | null): boolean {
  return fieldName === "q_criterion" || fieldName === "swirling_strength";
}

/** Threshold-preserving LOD for marching cubes. A cell mean can erase a
 * compact feature before contouring ever sees it. Signed-pair fields retain
 * the strongest resolved sense of rotation in each block; negative-core
 * diagnostics retain minima; all other surfaces retain maxima. */
export function isosurfaceWindowAggregation(
  fieldName: string | null,
): FieldWindowAggregation {
  if (defaultsToSignedPair(fieldName)) return "magnitude_extreme";
  if (defaultsToRotationDominatedSurface(fieldName)) return "block_min";
  return "block_max";
}

/** A reduced signed field needs both extrema lattices. A magnitude-extreme
 * lattice is sufficient for locating one dominant sign, but cannot feed a
 * paired contour because the opposite sign may be discarded in every block. */
export function isosurfaceWindowAggregations(
  fieldName: string | null,
  reduced: boolean,
): readonly FieldWindowAggregation[] {
  if (reduced && defaultsToSignedPair(fieldName)) return ["block_max", "block_min"];
  return [isosurfaceWindowAggregation(fieldName)];
}

export function positiveQThresholdRange(stats: SignedFieldStats): {
  min: number;
  max: number;
  step: number;
} {
  // Prefer positive-conditioned p98: a compact vortex can be well below the global p98.
  const robustPositive = stats.positiveP98 != null &&
      stats.positiveP98 > 0 && Number.isFinite(stats.positiveP98)
    ? stats.positiveP98
    : stats.p98 > 0 && Number.isFinite(stats.p98)
    ? stats.p98
    : Math.max(0, stats.max);
  const scale = robustPositive > 0 ? robustPositive : Math.max(1e-6, Math.abs(stats.max));
  return { min: scale * 0.02, max: scale, step: scale / 200 };
}

export function initialPositiveQThreshold(stats: SignedFieldStats): number {
  return positiveQThresholdRange(stats).max * 0.35;
}
