import type { SignedFieldStats } from "./coherentStructures";
import { signedMagnitudeScale, supportsSignedPair } from "./coherentStructures";
import type { FieldWindowAggregation } from "@/lib/dataset/nativeFieldWindow";
import type { ScalarFieldTransform } from "@/lib/vtk/vtkPreparation";

/** Fixed so optical density does not change with the exported lattice. */
export const VOLUME_OPACITY_REFERENCE_M = 1_000;
export const VOLUME_CONTEXT_OPACITY_FACTOR = 0.08;

/** Badge text, derived so it cannot drift from the constant. */
export const OPACITY_REFERENCE_LABEL = VOLUME_OPACITY_REFERENCE_M >= 1_000
  ? `${VOLUME_OPACITY_REFERENCE_M / 1_000} km`
  : `${VOLUME_OPACITY_REFERENCE_M} m`;

export type VolumeTransferKind = "sequential" | "signed" | "dbz";
export type VolumeOpacityClass =
  | "sparse-positive"
  | "signed-perturbation"
  | "background-stratified"
  | "reflectivity"
  | "generic";

export interface VolumeTransferPolicy {
  kind: VolumeTransferKind;
  opacityClass: VolumeOpacityClass;
  min: number;
  max: number;
  /** Optional structural upper bound, independent of the displayed colour range. */
  opacityMax?: number;
}

export interface VolumeOpacityPoint {
  scalar: number;
  opacity: number;
}

export function defaultsToSignedVolume(fieldName: string | null): boolean {
  if (!fieldName) return false;
  return fieldName === "w"
    || fieldName === "buoyancy"
    || fieldName === "theta_prime"
    || fieldName === "p_prime"
    || fieldName === "okubo_weiss"
    || fieldName === "vorticity_z"
    || fieldName === "vorticity_r"
    || fieldName === "vorticity_theta"
    || fieldName === "horizontal_vorticity_streamwise"
    || fieldName === "horizontal_vorticity_crosswise"
    || fieldName === "analytic_signed_couplet"
    /* Vorticity budget terms. These are tendencies, so the sign says whether
     * the term generates or destroys rotation — rendering them on a sequential
     * ramp shows only magnitude and silently drops the direction, which is the
     * quantity a budget analysis is actually asking about. */
    || fieldName === "stretching_term"
    || fieldName === "tilting_term"
    || fieldName === "baroclinic_term";
}

/** True if the field is in dBZ; NWS ramp anchors are physical dBZ values and cannot describe other fields. */
export function isReflectivityField(fieldName: string | null): boolean {
  if (!fieldName) return false;
  return fieldName.includes("dbz") || fieldName.includes("reflectiv");
}

export function volumeOpacityClass(fieldName: string | null): VolumeOpacityClass {
  const field = fieldName?.trim().toLowerCase() ?? "";
  if (!field) return "generic";
  if (isReflectivityField(field)) return "reflectivity";
  if (defaultsToSignedVolume(field)) return "signed-perturbation";
  if (field.includes("vorticity_magnitude")
      || field.includes("speed")
      || field.includes("condensate")
      || field === "qc" || field === "qr" || field === "qg" || field === "qh"
      || field === "qi" || field === "qs"
      || field.includes("cloud") || field.includes("precip")
      || field === "analytic_thin_sheet" || field === "analytic_helical_tube") {
    return "sparse-positive";
  }
  if (field === "temperature" || field === "theta" || field === "theta_e"
      || field === "relative_humidity" || field === "dewpoint"
      || field === "qv" || field === "p" || field === "rho"
      || field === "analytic_smooth_gradient" || field === "analytic_stratified_anomaly") {
    return "background-stratified";
  }
  return "generic";
}

export function supportsVolumeStructure(fieldName: string | null): boolean {
  return volumeOpacityClass(fieldName) === "background-stratified"
    || fieldName === "p_prime" || fieldName === "theta_prime";
}

export function volumeFieldWindowAggregation(fieldName: string | null, scalarTransform: ScalarFieldTransform = "none"): FieldWindowAggregation {
  if (scalarTransform === "horizontal-level-anomaly") return "cell_mean";
  const opacityClass = volumeOpacityClass(fieldName);
  if (opacityClass === "reflectivity") return "linear_reflectivity_mean";
  if (opacityClass === "signed-perturbation") return "magnitude_extreme";
  if (opacityClass === "sparse-positive") return "block_max";
  return "cell_mean";
}

export function volumeOpacityReferenceM(opacityClass: VolumeOpacityClass): number {
  if (opacityClass === "sparse-positive") return 100;
  if (opacityClass === "signed-perturbation") return 500;
  return VOLUME_OPACITY_REFERENCE_M;
}

export function volumeUsesStructuralLighting(opacityClass: VolumeOpacityClass): boolean {
  return opacityClass !== "signed-perturbation"
    && opacityClass !== "background-stratified";
}

export function volumeUsesOpacityNormals(_opacityClass: VolumeOpacityClass): boolean {
  return false;
}

/** Rank-2 reflectivity products (column-max planes); excluded from volume rendering. */
const NON_VOLUMETRIC_REFLECTIVITY = new Set(["composite_reflectivity"]);

/** Preferred volumetric reflectivity field for a 3-D scene; explicit priority, not first-match. */
export function preferredVolumetricReflectivityField(
  fields: readonly string[],
): string | null {
  if (fields.includes("reflectivity_dbz")) return "reflectivity_dbz";
  return fields.find(
    (name) => isReflectivityField(name) && !NON_VOLUMETRIC_REFLECTIVITY.has(name),
  ) ?? null;
}

export function defaultVolumeColormap(fieldName: string | null): string {
  if (defaultsToSignedVolume(fieldName)) return "Cool to Warm";
  if (isReflectivityField(fieldName)) return "dBZ";
  if (volumeOpacityClass(fieldName) === "background-stratified") return "Viridis";
  return "Inferno";
}

export function volumeOpacitySelectionKey(
  fieldName: string | null,
  transfer: { colormap: string; opacityScale: number; scalarTransform: ScalarFieldTransform; stormStructure?: boolean },
): string {
  // Palette semantics can move the zero-opacity support. Positive density
  // changes optical depth without moving that support or changing the bricks.
  const policy = resolveVolumeTransferPolicy(
    fieldName, transfer.colormap, { min: -1, max: 1, p2: -1, p98: 1 }, transfer.scalarTransform,
  );
  return JSON.stringify([policy.kind, transfer.opacityScale > 0, transfer.scalarTransform, transfer.stormStructure === true]);
}

export function resolveVolumeTransferPolicy(
  fieldName: string | null,
  colormap: string,
  stats: SignedFieldStats,
  scalarTransform: ScalarFieldTransform = "none",
  stormStructure = false,
): VolumeTransferPolicy {
  if (scalarTransform === "horizontal-level-anomaly") {
    const magnitude = structureMagnitudeScale(stats);
    return {
      kind: "signed",
      opacityClass: "signed-perturbation",
      min: -magnitude,
      max: magnitude,
    };
  }
  const opacityClass = volumeOpacityClass(fieldName);
  // The fixed -30..75 window is a property of reflectivity, not of the palette.
  // Keyed on the palette alone it was applied to whatever field was loaded, so
  // NWS dBZ on vorticity (0..0.004) put every value below the 5 dBZ opacity
  // anchor and on one colour stop — a flat, transparent smear.
  if (colormap === "dBZ" && isReflectivityField(fieldName)) {
    return { kind: "dbz", opacityClass: "reflectivity", min: -30, max: 75 };
  }
  // Reflectivity straddles zero (clear air is negative dBZ), but it is a log
  // intensity, not a diverging field. Letting a palette promote it to "signed"
  // gave clear air the same opacity as a downpour and filled the domain solid.
  const signed = supportsSignedPair(stats)
    && !isReflectivityField(fieldName)
    && (defaultsToSignedVolume(fieldName) || colormap === "Cool to Warm");
  if (signed) {
    const magnitude = stormStructure ? structureMagnitudeScale(stats) : signedMagnitudeScale(stats);
    return { kind: "signed", opacityClass, min: -magnitude, max: magnitude };
  }
  if (Number.isFinite(stats.p2) && Number.isFinite(stats.p98) && stats.p98 > stats.p2) {
    const opacityMax = opacityClass === "sparse-positive"
        && stats.p995 != null
        && Number.isFinite(stats.p995)
        && stats.p995 > stats.p98
      ? stats.p995
      : undefined;
    return {
      kind: "sequential",
      opacityClass,
      min: stats.p2,
      max: opacityMax ?? stats.p98,
      ...(opacityMax === undefined ? {} : { opacityMax }),
    };
  }
  return { kind: "sequential", opacityClass, min: stats.min, max: stats.max };
}

function structureMagnitudeScale(stats: SignedFieldStats): number {
  const tails = Number.isFinite(stats.positiveP98) && Number.isFinite(stats.negativeP02)
    ? Math.max(Math.abs(stats.positiveP98!), Math.abs(stats.negativeP02!))
    : Math.max(Math.abs(stats.min), Math.abs(stats.max));
  // Separate signs preserve a sparse core against its compensating background
  // without letting one extreme sample hide the rest of a developed volume.
  return Math.max(signedMagnitudeScale(stats), Number.isFinite(tails) ? tails * 0.25 : 0);
}

/** Opacity of `scale` stacked copies of a medium of opacity `alpha`; bounded by 1 without clamping. */
export function opticalDepthOpacity(alpha: number, scale: number): number {
  if (!(scale > 0) || !(alpha > 0)) return 0;
  if (alpha >= 1) return 1;
  // Exact at unit scale: 1 - (1 - a)^1 is a float epsilon off `a`, and the
  // default curve should be bit-identical to what it always was.
  if (scale === 1) return alpha;
  return 1 - Math.pow(1 - alpha, scale);
}

export function volumeOpacityPoints(
  policy: Pick<VolumeTransferPolicy, "kind" | "min" | "max" | "opacityMax"> & {
    opacityClass?: VolumeOpacityClass;
  },
  scale: number,
): VolumeOpacityPoint[] {
  const { min, max, kind } = policy;
  const opacityClass = policy.opacityClass ?? (kind === "signed"
    ? "signed-perturbation"
    : kind === "dbz" ? "reflectivity" : "generic");
  const finiteScale = Number.isFinite(scale) ? Math.max(0, scale) : 0;
  if (!Number.isFinite(min) || !Number.isFinite(max) || !(max > min)) {
    const center = Number.isFinite(min) ? min : 0;
    return [
      { scalar: center - 1, opacity: 0 },
      { scalar: center + 1, opacity: 0 },
    ];
  }
  if (kind === "signed") {
    const magnitude = Math.max(Math.abs(min), Math.abs(max));
    const faint = opticalDepthOpacity(0.045, finiteScale);
    const strong = opticalDepthOpacity(0.82, finiteScale);
    return [
      { scalar: -magnitude, opacity: strong },
      { scalar: -magnitude * 0.62, opacity: opticalDepthOpacity(0.3, finiteScale) },
      { scalar: -magnitude * 0.42, opacity: faint },
      { scalar: -magnitude * 0.28, opacity: 0 },
      { scalar: magnitude * 0.28, opacity: 0 },
      { scalar: magnitude * 0.42, opacity: faint },
      { scalar: magnitude * 0.62, opacity: opticalDepthOpacity(0.3, finiteScale) },
      { scalar: magnitude, opacity: strong },
    ];
  }
  if (kind === "dbz") {
    // Physical dBZ anchors (not range fractions): clear air <5, precip@20, heavy@40, hail@55.
    // Scaled via opticalDepthOpacity to avoid top-down clamping that flattens the storm.
    return [
      { scalar: min, opacity: 0 },
      { scalar: 5, opacity: 0 },
      { scalar: 20, opacity: opticalDepthOpacity(0.1, finiteScale) },
      { scalar: 40, opacity: opticalDepthOpacity(0.42, finiteScale) },
      { scalar: 55, opacity: opticalDepthOpacity(0.72, finiteScale) },
      { scalar: max, opacity: opticalDepthOpacity(0.9, finiteScale) },
    ];
  }
  const opacityMaximum = policy.opacityMax != null
      && Number.isFinite(policy.opacityMax)
      && policy.opacityMax > min
    ? policy.opacityMax
    : max;
  const range = opacityMaximum - min;
  if (opacityClass === "sparse-positive") {
    return [
      { scalar: min, opacity: 0 },
      { scalar: min + 0.45 * range, opacity: 0 },
      { scalar: min + 0.58 * range, opacity: opticalDepthOpacity(0.08, finiteScale) },
      { scalar: min + 0.74 * range, opacity: opticalDepthOpacity(0.32, finiteScale) },
      { scalar: min + 0.9 * range, opacity: opticalDepthOpacity(0.72, finiteScale) },
      { scalar: opacityMaximum, opacity: opticalDepthOpacity(0.94, finiteScale) },
    ];
  }
  if (opacityClass === "background-stratified") {
    return [
      { scalar: min, opacity: 0 },
      { scalar: min + 0.15 * range, opacity: opticalDepthOpacity(0.004, finiteScale) },
      { scalar: min + 0.4 * range, opacity: opticalDepthOpacity(0.008, finiteScale) },
      { scalar: min + 0.65 * range, opacity: opticalDepthOpacity(0.015, finiteScale) },
      { scalar: min + 0.85 * range, opacity: opticalDepthOpacity(0.035, finiteScale) },
      { scalar: max, opacity: opticalDepthOpacity(0.1, finiteScale) },
    ];
  }
  // Generic scalar volumes used to assign a small opacity to the entire
  // p2..p98 interval. Across a storm-scale ray, even that faint tail
  // accumulated into an opaque, voxel-shaped block. Keep the lower half of
  // the robust range exactly empty, then bring the feature-bearing tail in
  // gradually. Optical-depth scaling preserves the curve's separation when
  // the user raises density instead of clamping several anchors to 1.
  return [
    { scalar: min, opacity: 0 },
    { scalar: min + 0.45 * range, opacity: 0 },
    { scalar: min + 0.65 * range, opacity: opticalDepthOpacity(0.015, finiteScale) },
    { scalar: min + 0.82 * range, opacity: opticalDepthOpacity(0.12, finiteScale) },
    { scalar: min + 0.94 * range, opacity: opticalDepthOpacity(0.42, finiteScale) },
    { scalar: max, opacity: opticalDepthOpacity(0.72, finiteScale) },
  ];
}
