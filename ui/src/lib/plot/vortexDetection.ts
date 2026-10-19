/* Locate the near-surface rotation maximum in a 3D ω_z field on the Cartesian C-grid.
 * `vorticity_z` manifest key = ω_z (cartesian_cgrid_3d.cpp:1060); pass it as `vorticityZ`. */

/** Grid dimensions (number of cells in each axis, integers). */
export interface GridShape {
  nx: number;
  ny: number;
  nz: number;
}

/** Uniform cell spacing in each axis (metres). */
export interface GridSpacing {
  dx: number;
  dy: number;
  dz: number;
}

export interface VortexDetectionOptions {
  /** AGL search ceiling (m); restricts to near-surface tornadic rotation vs. 3-8 km mesocyclone. */
  nearSurfaceDepthM: number;
  /** Min |ζ| (s⁻¹) to qualify; default 0.01 s⁻¹ (moderate supercell). Returns null if not exceeded. */
  significanceThresholdPerS: number;
}

const DEFAULT_NEAR_SURFACE_DEPTH_M = 1000;
const DEFAULT_SIGNIFICANCE_THRESHOLD_PER_S = 0.01;

export interface VortexCore {
  /** Grid-relative x position of the peak |ζ| cell (metres). */
  xM: number;
  /** Grid-relative y position of the peak |ζ| cell (metres). */
  yM: number;
  /** AGL height of the peak |ζ| cell (metres). */
  zM: number;
  /** Peak |ζ| (s⁻¹). Always positive. */
  peakVorticityPerS: number;
  /** +1 = cyclonic (CCW NH), -1 = anticyclonic. */
  rotationSign: 1 | -1;
  /** Half-maximum half-width (m) averaged over ±x/±y; scale hint, not a precise boundary. */
  estimatedCoreRadiusM: number;
}

/** Locate vortex core in ω_z flat row-major [ix+nx*(iy+ny*iz)]. Returns null if below significance threshold. */
export function findRotationMaximum(
  vorticityZ: Float32Array | Float64Array | readonly number[],
  shape: GridShape,
  spacing: GridSpacing,
  options: Partial<VortexDetectionOptions> = {},
): VortexCore | null {
  const { nx, ny } = shape;
  return findRotationMaximumWithAccessor(
    (ix, iy, iz) => vorticityZ[ix + nx * (iy + ny * iz)],
    shape,
    spacing,
    options,
  );
}

/**
 * Locate a vortex core directly in an x-fastest, interleaved [x,y,z] vector
 * buffer. This is the layout already produced for vtkImageData, so callers do
 * not need to allocate and copy a separate vertical-component volume.
 */
export function findInterleavedVerticalRotationMaximum(
  vectors: Float32Array | Float64Array | readonly number[],
  shape: GridShape,
  spacing: GridSpacing,
  options: Partial<VortexDetectionOptions> = {},
): VortexCore | null {
  const { nx, ny } = shape;
  return findRotationMaximumWithAccessor(
    (ix, iy, iz) => vectors[(ix + nx * (iy + ny * iz)) * 3 + 2],
    shape,
    spacing,
    options,
  );
}

type RotationValueAt = (ix: number, iy: number, iz: number) => number;

function findRotationMaximumWithAccessor(
  valueAt: RotationValueAt,
  shape: GridShape,
  spacing: GridSpacing,
  options: Partial<VortexDetectionOptions>,
): VortexCore | null {
  const { nx, ny, nz } = shape;
  const { dx, dy, dz } = spacing;

  if (
    nx <= 0 ||
    ny <= 0 ||
    nz <= 0 ||
    !Number.isFinite(dx) ||
    dx <= 0 ||
    !Number.isFinite(dy) ||
    dy <= 0 ||
    !Number.isFinite(dz) ||
    dz <= 0
  ) {
    return null;
  }

  const nearSurfaceDepthM =
    options.nearSurfaceDepthM ?? DEFAULT_NEAR_SURFACE_DEPTH_M;
  const significanceThresholdPerS =
    options.significanceThresholdPerS ?? DEFAULT_SIGNIFICANCE_THRESHOLD_PER_S;

  // Maximum k-index whose cell centre is within the near-surface search layer.
  // Cell centres sit at (k + 0.5) * dz; we keep k where that is <= depth.
  const kMax = Math.min(
    nz - 1,
    Math.floor((nearSurfaceDepthM - 0.5 * dz) / dz),
  );

  if (kMax < 0) {
    // Layer is shallower than half a cell — nothing to search.
    return null;
  }

  let peakAbsVorticity = 0;
  let peakIx = -1;
  let peakIy = -1;
  let peakIz = -1;
  let peakValue = 0; // signed, for rotation-sign extraction

  for (let iz = 0; iz <= kMax; iz++) {
    for (let iy = 0; iy < ny; iy++) {
      for (let ix = 0; ix < nx; ix++) {
        const value = valueAt(ix, iy, iz);
        if (!Number.isFinite(value)) continue;
        const absValue = Math.abs(value);
        if (absValue > peakAbsVorticity) {
          peakAbsVorticity = absValue;
          peakIx = ix;
          peakIy = iy;
          peakIz = iz;
          peakValue = value;
        }
      }
    }
  }

  if (peakAbsVorticity < significanceThresholdPerS) {
    // No significant rotation found — do not invent a vortex.
    return null;
  }

  const xM = (peakIx + 0.5) * dx;
  const yM = (peakIy + 0.5) * dy;
  const zM = (peakIz + 0.5) * dz;

  const estimatedCoreRadiusM = estimateCoreRadius(
    valueAt,
    shape,
    spacing,
    peakIx,
    peakIy,
    peakIz,
    peakAbsVorticity,
  );

  return {
    xM,
    yM,
    zM,
    peakVorticityPerS: peakAbsVorticity,
    rotationSign: peakValue >= 0 ? 1 : -1,
    estimatedCoreRadiusM,
  };
}

/** Half-maximum |ζ| half-width along ±x/±y through peak cell, averaged; discretized to grid-cell half-width. */
function estimateCoreRadius(
  valueAt: RotationValueAt,
  shape: GridShape,
  spacing: GridSpacing,
  peakIx: number,
  peakIy: number,
  peakIz: number,
  peakAbsVorticity: number,
): number {
  const { nx, ny } = shape;
  const { dx, dy } = spacing;
  const halfMax = 0.5 * peakAbsVorticity;

  // Walk outward in +x until |ζ| drops below half-maximum; same in -x.
  let radiusXPlus = 0;
  for (let ix = peakIx + 1; ix < nx; ix++) {
    const v = valueAt(ix, peakIy, peakIz);
    if (!Number.isFinite(v) || Math.abs(v) < halfMax) {
      radiusXPlus = (ix - peakIx) * dx;
      break;
    }
    if (ix === nx - 1) radiusXPlus = (ix - peakIx + 1) * dx;
  }

  let radiusXMinus = 0;
  for (let ix = peakIx - 1; ix >= 0; ix--) {
    const v = valueAt(ix, peakIy, peakIz);
    if (!Number.isFinite(v) || Math.abs(v) < halfMax) {
      radiusXMinus = (peakIx - ix) * dx;
      break;
    }
    if (ix === 0) radiusXMinus = (peakIx - ix + 1) * dx;
  }

  let radiusYPlus = 0;
  for (let iy = peakIy + 1; iy < ny; iy++) {
    const v = valueAt(peakIx, iy, peakIz);
    if (!Number.isFinite(v) || Math.abs(v) < halfMax) {
      radiusYPlus = (iy - peakIy) * dy;
      break;
    }
    if (iy === ny - 1) radiusYPlus = (iy - peakIy + 1) * dy;
  }

  let radiusYMinus = 0;
  for (let iy = peakIy - 1; iy >= 0; iy--) {
    const v = valueAt(peakIx, iy, peakIz);
    if (!Number.isFinite(v) || Math.abs(v) < halfMax) {
      radiusYMinus = (peakIy - iy) * dy;
      break;
    }
    if (iy === 0) radiusYMinus = (peakIy - iy + 1) * dy;
  }

  // Average the four directional estimates.  Fall back to one grid-cell
  // spacing when all four arms remain at maximum out to the domain boundary
  // (the feature fills the domain — the estimate is a lower bound).
  const contributions = [radiusXPlus, radiusXMinus, radiusYPlus, radiusYMinus].filter(
    (r) => r > 0,
  );
  if (contributions.length === 0) {
    return Math.min(dx, dy);
  }
  return contributions.reduce((a, b) => a + b, 0) / contributions.length;
}
