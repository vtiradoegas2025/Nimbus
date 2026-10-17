import { opticalDepthOpacity } from "./volumeTransferPolicy";

/** One physical opacity reference keeps the same condensate display-optically stable
 * when the output lattice, renderer LOD, or display reconstruction changes. */
export const CLOUD_OPACITY_REFERENCE_M = 1_000;

/** Peak scalar opacity across CLOUD_OPACITY_REFERENCE_M. This preserves the
 * former fine-grid (~100 m) accumulated opacity without inheriting its
 * resolution dependence. */
export const CLOUD_PEAK_OPACITY = 0.22;

/** Precipitation is denser than suspended condensate at the same control value,
 * but still uses optical-depth scaling so its high end does not collapse into
 * an opaque slab. */
export const PRECIP_PEAK_OPACITY = 0.5;

export function scaledCloudPeakOpacity(scale: number): number {
  return opticalDepthOpacity(CLOUD_PEAK_OPACITY, scale);
}

export function scaledPrecipPeakOpacity(scale: number): number {
  return opticalDepthOpacity(PRECIP_PEAK_OPACITY, scale);
}

/** Keep the cloud as spatial context while making embedded solved-field
 * diagnostics readable. This changes only display opacity; scalar values,
 * thresholds, and extracted geometry remain untouched. */
export const CLOUD_DIAGNOSTIC_REVEAL_SCALE = 0.35;

/** Compatibility fallbacks for callers without a valid pane admission. */
export const SUPERCELL_AUXILIARY_VOXEL_CAP = 4_000_000;
export const SUPERCELL_EXTREME_AUXILIARY_VOXEL_CAP = 8_000_000;
export const SUPERCELL_EXTREME_GRID_SPACING_M = 30;

export function supercellAuxiliaryVoxelBudget(
  paneVoxelCap: number,
  minimumGridSpacingM: number | null | undefined,
): number {
  const safePaneCap = Number.isFinite(paneVoxelCap)
    ? Math.max(1, Math.floor(paneVoxelCap))
    : SUPERCELL_AUXILIARY_VOXEL_CAP;
  void minimumGridSpacingM;
  // Contour output is now counted and reserved before marching cubes. Reusing
  // the pane's admitted scalar working set avoids clipping a known updraft or
  // vortex merely to satisfy the obsolete 4M/8M pre-extraction proxy.
  return safePaneCap;
}

export function cloudDensityForDiagnostics(
  density: number,
  revealDiagnostics: boolean,
): number {
  const safeDensity = Number.isFinite(density) ? Math.max(0, density) : 0;
  return revealDiagnostics
    ? safeDensity * CLOUD_DIAGNOSTIC_REVEAL_SCALE
    : safeDensity;
}

export type CloudEdgeMode = "off" | "subtle" | "full";

export interface CloudOpacityPoint {
  scalar: number;
  opacity: number;
}

/**
 * A volume-native cloud boundary curve. The former polygon shell wrote depth
 * before the volume pass and clipped the ray-marched cloud into a grey skin.
 * These modes instead alter only the condensate transfer curve: stronger edge
 * modes narrow the scalar interval over which the cloud becomes visible, while
 * keeping tenuous condensate optically thin enough to avoid a path-length haze.
 */
export function cloudOpacityPoints(
  floor: number,
  displayMax: number,
  peakOpacity: number,
  edge: CloudEdgeMode,
): CloudOpacityPoint[] {
  const span = Math.max(displayMax - floor, Number.EPSILON);
  const peak = Math.min(1, Math.max(0, peakOpacity));
  const saturationFraction = edge === "full" ? 0.64 : edge === "subtle" ? 0.82 : 1;
  const points: CloudOpacityPoint[] = [
    { scalar: 0, opacity: 0 },
    { scalar: floor, opacity: 0 },
  ];

  // A sparse piecewise-linear curve had abrupt slope changes at its onset and
  // shoulder nodes. Computing normals from that opacity field turned those
  // mathematical knees into nested illuminated shells. Sample a quintic
  // smootherstep instead: it retains the exact floor and peak, is monotone,
  // and has zero slope at both ends. Edge modes now control how early the same
  // smooth curve reaches optical thickness, rather than inserting an isovalue-
  // like shoulder into the cloud.
  const samples = 16;
  for (let sample = 1; sample <= samples; sample += 1) {
    const fraction = sample / samples;
    const u = Math.min(1, fraction / saturationFraction);
    const smooth = u * u * u * (u * (u * 6 - 15) + 10);
    points.push({
      scalar: floor + span * fraction,
      opacity: peak * smooth,
    });
  }
  return points;
}
