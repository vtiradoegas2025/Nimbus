/*
 * Earth-coordinate accumulation for lifetime swaths. Accumulates into a grid fixed to the Earth
 * (union of all frame footprints); positions via nearest-cell binning (max half-cell error, disclosed).
 * Pure — no React, no vtk.
 */

export interface SwathFrameGeometry {
  /** Physical cells along x and y for this frame. */
  nx: number;
  ny: number;
  /** Uniform cell spacing in metres. */
  dxM: number;
  dyM: number;
  /** Earth coordinate of this frame's first scalar cell centre, in metres.
   *  Null for a stationary run that declares no Earth origin — every frame then
   *  shares one implicit origin. */
  earthOriginM?: readonly [number, number] | null;
}

/** Minimal manifest grid metadata needed to locate a horizontal swath footprint. */
export interface SwathGridMetadata {
  nr?: number;
  nth?: number;
  nz?: number;
  drM?: number;
  dyM?: number;
  horizontalScalarOriginM?: readonly [number, number];
  scalarCoordinatesM?: readonly [readonly number[], readonly number[], readonly number[]];
  scalarOriginM?: readonly [number, number, number];
  physicalBoundsM?: { lower: readonly [number, number, number] };
  validIndexRange?: {
    lower: readonly [number, number, number];
    upper: readonly [number, number, number];
  };
}

/**
 * Convert manifest geometry to the halo-free, first-cell-centre convention used
 * by the accumulator. `earthGridFaceOriginM` is the moving-domain grid FACE
 * origin; adding (validLower + 1/2) cells is therefore required.
 */
export function swathFrameGeometry(
  grid: SwathGridMetadata | null | undefined,
  earthGridFaceOriginM?: readonly [number, number] | null,
  shape?: readonly [number, number],
): SwathFrameGeometry | null {
  if (!grid) return null;
  const dxM = grid.drM;
  const dyM = grid.dyM ?? dxM;
  if (!(dxM != null && dxM > 0) || !(dyM != null && dyM > 0)) return null;

  const range = grid.validIndexRange;
  const lowerX = range ? Math.trunc(range.lower[0]) : 0;
  const lowerY = range ? Math.trunc(range.lower[1]) : 0;
  const validNx = range ? Math.trunc(range.upper[0]) - lowerX + 1 : grid.nr;
  const validNy = range ? Math.trunc(range.upper[1]) - lowerY + 1 : grid.nth;
  const nx = shape?.[0] ?? validNx;
  const ny = shape?.[1] ?? validNy;
  if (!(nx != null && nx > 0) || !(ny != null && ny > 0)) return null;

  let origin: readonly [number, number] | null = null;
  if (earthGridFaceOriginM
      && Number.isFinite(earthGridFaceOriginM[0])
      && Number.isFinite(earthGridFaceOriginM[1])) {
    origin = [
      earthGridFaceOriginM[0] + (lowerX + 0.5) * dxM,
      earthGridFaceOriginM[1] + (lowerY + 0.5) * dyM,
    ];
  } else {
    const exactX = grid.scalarCoordinatesM?.[0];
    const exactY = grid.scalarCoordinatesM?.[1];
    const scalarX = exactX?.[lowerX]
      ?? (grid.scalarOriginM?.[0] != null ? grid.scalarOriginM[0] + lowerX * dxM : undefined)
      ?? (grid.horizontalScalarOriginM?.[0] != null
        ? grid.horizontalScalarOriginM[0] + lowerX * dxM
        : undefined)
      ?? (grid.physicalBoundsM?.lower[0] != null
        ? grid.physicalBoundsM.lower[0] + 0.5 * dxM
        : undefined);
    const scalarY = exactY?.[lowerY]
      ?? (grid.scalarOriginM?.[1] != null ? grid.scalarOriginM[1] + lowerY * dyM : undefined)
      ?? (grid.horizontalScalarOriginM?.[1] != null
        ? grid.horizontalScalarOriginM[1] + lowerY * dyM
        : undefined)
      ?? (grid.physicalBoundsM?.lower[1] != null
        ? grid.physicalBoundsM.lower[1] + 0.5 * dyM
        : undefined);
    if (Number.isFinite(scalarX) && Number.isFinite(scalarY)) {
      origin = [scalarX as number, scalarY as number];
    }
  }

  return { nx, ny, dxM, dyM, earthOriginM: origin };
}

export interface EarthSwathGrid {
  /** Earth coordinate of the accumulator's first cell centre, in metres. */
  originM: readonly [number, number];
  nx: number;
  ny: number;
  dxM: number;
  dyM: number;
  /** Largest absolute placement error introduced by nearest-cell binning, in
   *  metres. At most half a cell per axis; reported so the plot can disclose it. */
  maxPlacementErrorM: number;
}

/** Plotly ultimately materializes the swath as a JS number matrix, so keep the
 * accumulator at roughly a 2048² plane. This is checked before allocation. */
export const SWATH_ACCUMULATOR_CELL_CAP = 4_194_304;

/** A 3D swath must inspect every source cell to preserve its column extrema.
 * Above 256 MiB per frame the UI requires a precomputed rank-2 column product
 * rather than silently replacing maxima/minima with block means. */
export const SWATH_3D_SOURCE_CELL_CAP = (256 * 1024 * 1024) / Float32Array.BYTES_PER_ELEMENT;

/** Null when a swath accumulator is safe to allocate; otherwise a user-facing refusal. */
export function swathAccumulatorBudgetError(nx: number, ny: number): string | null {
  const cells = nx * ny;
  if (!Number.isSafeInteger(nx) || !Number.isSafeInteger(ny) || nx <= 0 || ny <= 0) {
    return "The swath grid has invalid horizontal dimensions.";
  }
  if (!Number.isSafeInteger(cells) || cells > SWATH_ACCUMULATOR_CELL_CAP) {
    return `The Earth-fixed swath would contain ${Number.isFinite(cells) ? cells.toLocaleString() : "too many"} cells; `
      + `the interactive limit is ${SWATH_ACCUMULATOR_CELL_CAP.toLocaleString()}. `
      + "Use a coarser rank-2 column diagnostic or export the full-resolution swath.";
  }
  return null;
}

/** Halo-free source-cell count used to admit exact rank-3 column reductions. */
export function validGridCellCount(grid: SwathGridMetadata | null | undefined): number | null {
  if (!grid) return null;
  const full = [grid.nr, grid.nth, grid.nz] as Array<number | undefined>;
  const range = grid.validIndexRange;
  const counts = range
    ? range.upper.map((upper, axis) => Math.trunc(upper) - Math.trunc(range.lower[axis]) + 1)
    : full;
  if (counts.some((count) => !Number.isSafeInteger(count) || (count as number) <= 0)) return null;
  const cells = (counts as number[]).reduce((product, count) => product * count, 1);
  return Number.isSafeInteger(cells) ? cells : null;
}

/** Earth-fixed accumulator covering the union of all frame footprints. Null if spacing/count disagree. */
export function earthSwathGrid(
  frames: readonly SwathFrameGeometry[],
): EarthSwathGrid | null {
  if (frames.length === 0) return null;
  const first = frames[0];
  if (!(first.nx > 0) || !(first.ny > 0)) return null;
  if (!(first.dxM > 0) || !(first.dyM > 0)) return null;

  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;

  for (const frame of frames) {
    if (frame.nx !== first.nx || frame.ny !== first.ny) return null;
    if (frame.dxM !== first.dxM || frame.dyM !== first.dyM) return null;
    const origin = frame.earthOriginM ?? [0, 0];
    if (!Number.isFinite(origin[0]) || !Number.isFinite(origin[1])) return null;
    minX = Math.min(minX, origin[0]);
    minY = Math.min(minY, origin[1]);
    maxX = Math.max(maxX, origin[0] + (frame.nx - 1) * frame.dxM);
    maxY = Math.max(maxY, origin[1] + (frame.ny - 1) * frame.dyM);
  }
  if (!Number.isFinite(minX) || !Number.isFinite(minY)) return null;

  // Anchor on minimum origin: stationary run lands at offset (0,0), no copy needed.
  const nx = Math.round((maxX - minX) / first.dxM) + 1;
  const ny = Math.round((maxY - minY) / first.dyM) + 1;
  return {
    originM: [minX, minY],
    nx,
    ny,
    dxM: first.dxM,
    dyM: first.dyM,
    maxPlacementErrorM: 0.5 * Math.hypot(first.dxM, first.dyM),
  };
}

/** Whole-cell offset of a frame's (0,0) in the accumulator (rounded, so extremum is always model-produced). */
export function earthCellOffset(
  frame: SwathFrameGeometry,
  grid: EarthSwathGrid,
): readonly [number, number] {
  const origin = frame.earthOriginM ?? [0, 0];
  return [
    Math.round((origin[0] - grid.originM[0]) / grid.dxM),
    Math.round((origin[1] - grid.originM[1]) / grid.dyM),
  ];
}

/** Whether the accumulator is larger than one frame — i.e. the domain moved. */
export function swathSpansMultipleFootprints(
  frames: readonly SwathFrameGeometry[],
  grid: EarthSwathGrid,
): boolean {
  const first = frames[0];
  return first != null && (grid.nx !== first.nx || grid.ny !== first.ny);
}

/** One-line provenance string for the canvas (discloses binning error for moving domains). */
export function describeEarthSwath(
  grid: EarthSwathGrid,
  moved: boolean,
): string {
  if (!moved) return "accumulated on the run's fixed grid";
  const errorM = Math.round(grid.maxPlacementErrorM);
  return `accumulated in Earth coordinates across a moving domain · ${grid.nx}x${grid.ny} cells`
    + ` · values exact, positions binned to the nearest cell (max ${errorM} m)`;
}
