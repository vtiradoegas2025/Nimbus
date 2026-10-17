import { signedMagnitudeScale, supportsSignedPair, type SignedFieldStats } from "./coherentStructures";

export type ScalarPlaneReduction = "column-max" | "surface";
export type ScalarPlaneKind = "dbz" | "signed" | "sequential";

export interface ScalarPlanePolicy {
  kind: ScalarPlaneKind;
  reduction: ScalarPlaneReduction;
  colormap: string;
  min: number;
  max: number;
}

const PREFERRED_SCALAR_PLANES = [
  "reflectivity_dbz",
  "buoyancy",
  "theta_prime",
] as const;

export function availableScalarPlanes(fields: readonly string[]): string[] {
  return PREFERRED_SCALAR_PLANES.filter((field) => fields.includes(field));
}

export function scalarPlaneReduction(fieldName: string): ScalarPlaneReduction {
  return fieldName.includes("dbz") || fieldName.includes("reflectiv")
    ? "column-max"
    : "surface";
}

export function resolveScalarPlanePolicy(
  fieldName: string,
  stats: SignedFieldStats,
): ScalarPlanePolicy {
  const reduction = scalarPlaneReduction(fieldName);
  if (reduction === "column-max") {
    return { kind: "dbz", reduction, colormap: "dBZ floor", min: -30, max: 75 };
  }
  if (supportsSignedPair(stats)) {
    const magnitude = signedMagnitudeScale(stats);
    return {
      kind: "signed",
      reduction,
      colormap: "Cool to Warm",
      min: -magnitude,
      max: magnitude,
    };
  }
  const robust = Number.isFinite(stats.p2) && Number.isFinite(stats.p98) && stats.p98 > stats.p2;
  let min = robust ? stats.p2 : stats.min;
  let max = robust ? stats.p98 : stats.max;
  if (!(max > min)) {
    const center = Number.isFinite(min) ? min : 0;
    const padding = Math.max(1e-6, Math.abs(center) * 0.01);
    min = center - padding;
    max = center + padding;
  }
  return {
    kind: "sequential",
    reduction,
    colormap: "Viridis",
    min,
    max,
  };
}

export function describeScalarPlane(policy: ScalarPlanePolicy): string {
  const reduction = policy.reduction === "column-max" ? "column maximum" : "lowest valid level";
  const mapping = policy.kind === "signed" ? "symmetric diverging" : policy.kind;
  return `registered ${reduction} plane · ${mapping}`;
}
