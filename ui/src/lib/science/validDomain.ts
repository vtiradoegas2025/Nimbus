import {
  cartesianDyMeters,
  cartesianScalarOrigin,
  type GridSpacing,
} from "./cartesianInterp";
import type { NpyN } from "@/lib/platform/npy";
import { gridForWindowedField } from "./windowedGrid";

export interface PlotGrid extends GridSpacing {
  nr?: number;
  nth?: number;
  nz?: number;
  coordinateSystem?: string;
  staggering?: string;
  drM?: number;
  dyM?: number;
  dthetaRad?: number;
  dzM?: number;
  scalarOriginM?: [number, number, number];
  physicalBoundsM?: {
    lower: [number, number, number];
    upper: [number, number, number];
  };
  validIndexRange?: {
    lower: [number, number, number];
    upper: [number, number, number];
  };
  haloWidth?: {
    lower: [number, number, number];
    upper: [number, number, number];
  };
}

export interface ValidPlotDomain {
  field: NpyN;
  grid: PlotGrid | null;
  cropped: boolean;
}

/** Valid (halo-free) cell count along an axis from grid metadata alone. Null if no validIndexRange. */
export function validAxisLength(
  axis: 0 | 1 | 2,
  grid: PlotGrid | null | undefined,
): number | null {
  const range = grid?.validIndexRange;
  if (!range) return null;
  const lower = Math.trunc(range.lower[axis]);
  const upper = Math.trunc(range.upper[axis]);
  if (!Number.isFinite(lower) || !Number.isFinite(upper) || lower < 0 || upper < lower) {
    return null;
  }
  return upper - lower + 1;
}

/**
 * Crop a per-index axis reduction to the valid domain; updates grid metadata for `axis` only.
 * Falls back to unchanged input when metadata is absent or mismatched.
 */
export function cropAxisReduction<T>(
  axis: 0 | 1 | 2,
  values: T[],
  grid: PlotGrid | null | undefined,
): { values: T[]; grid: PlotGrid | null } {
  const unchanged = { values, grid: grid ?? null };
  const range = grid?.validIndexRange;
  if (!range) return unchanged;
  const lower = Math.trunc(range.lower[axis]);
  const upper = Math.trunc(range.upper[axis]);
  if (!Number.isFinite(lower) || !Number.isFinite(upper)
      || lower < 0 || upper < lower || upper >= values.length) {
    return unchanged;
  }
  if (lower === 0 && upper === values.length - 1) return unchanged;

  const count = upper - lower + 1;
  const fullCoordinates = grid!.scalarCoordinatesM?.[axis];
  const exact = fullCoordinates && fullCoordinates.length === values.length
    ? fullCoordinates.slice(lower, upper + 1)
    : null;
  const spacing = ([grid!.drM, cartesianDyMeters(grid!, grid!.drM ?? 1), grid!.dzM][axis]) ?? 1;
  const origin = grid!.scalarOriginM
    ? ([...grid!.scalarOriginM] as [number, number, number])
    : null;
  if (origin) origin[axis] = exact?.[0] ?? origin[axis] + lower * spacing;

  const counts: [number, number, number] = [
    grid!.nr ?? values.length, grid!.nth ?? values.length, grid!.nz ?? values.length,
  ];
  counts[axis] = count;
  const coordinates = grid!.scalarCoordinatesM && exact
    ? grid!.scalarCoordinatesM.map((current, index) => index === axis ? exact : current) as
      [number[], number[], number[]]
    : undefined;
  const validLower = [...range.lower] as [number, number, number];
  const validUpper = [...range.upper] as [number, number, number];
  validLower[axis] = 0;
  validUpper[axis] = count - 1;
  const halo = grid!.haloWidth;
  const haloLower = halo ? [...halo.lower] as [number, number, number] : undefined;
  const haloUpper = halo ? [...halo.upper] as [number, number, number] : undefined;
  if (haloLower && haloUpper) { haloLower[axis] = 0; haloUpper[axis] = 0; }

  return {
    values: values.slice(lower, upper + 1),
    grid: {
      ...grid!,
      nr: counts[0],
      nth: counts[1],
      nz: counts[2],
      scalarCoordinatesM: coordinates,
      ...(origin ? { scalarOriginM: origin } : {}),
      validIndexRange: { lower: validLower, upper: validUpper },
      ...(haloLower && haloUpper ? { haloWidth: { lower: haloLower, upper: haloUpper } } : {}),
    },
  };
}

/** Remove halo cells from a rank-1/2/3 field and shift coordinate metadata to first valid centre. */
export function cropToValidPlotDomain(
  field: NpyN,
  grid: PlotGrid | null | undefined,
): ValidPlotDomain {
  const effectiveGrid = gridForWindowedField(field, grid);
  if (!effectiveGrid?.validIndexRange || field.shape.length < 1 || field.shape.length > 3) {
    return { field, grid: effectiveGrid, cropped: false };
  }
  grid = effectiveGrid;
  const validIndexRange = effectiveGrid.validIndexRange;
  const rank = field.shape.length;
  const lower = validIndexRange.lower.map(Math.trunc) as [number, number, number];
  const upper = validIndexRange.upper.map(Math.trunc) as [number, number, number];
  for (let axis = 0; axis < rank; axis++) {
    if (lower[axis] < 0 || upper[axis] < lower[axis] || upper[axis] >= field.shape[axis]) {
      return { field, grid, cropped: false };
    }
  }
  const outputShape = field.shape.map(
    (dim, axis) => axis < rank ? upper[axis] - lower[axis] + 1 : dim,
  );
  const cropped = outputShape.some((dim, axis) => dim !== field.shape[axis]);
  if (!cropped) return { field, grid, cropped: false };

  const outputLength = outputShape.reduce((product, dim) => product * dim, 1);
  const output = new Float32Array(outputLength);
  if (rank === 1) {
    output.set(field.data.subarray(lower[0], upper[0] + 1));
  } else if (rank === 2) {
    const sourceCols = field.shape[1];
    const outputCols = outputShape[1];
    let dst = 0;
    for (let i = lower[0]; i <= upper[0]; i++) {
      const sourceBase = i * sourceCols + lower[1];
      output.set(field.data.subarray(sourceBase, sourceBase + outputCols), dst);
      dst += outputCols;
    }
  } else {
    const sourceNy = field.shape[1];
    const sourceNz = field.shape[2];
    const outputNz = outputShape[2];
    let dst = 0;
    for (let i = lower[0]; i <= upper[0]; i++) {
      for (let j = lower[1]; j <= upper[1]; j++) {
        const sourceBase = (i * sourceNy + j) * sourceNz + lower[2];
        output.set(field.data.subarray(sourceBase, sourceBase + outputNz), dst);
        dst += outputNz;
      }
    }
  }

  const dx = grid.drM ?? 1;
  const dy = cartesianDyMeters(grid, dx);
  const dz = grid.dzM ?? 1;
  const fullShape: [number, number, number] = [
    field.shape[0] ?? grid.nr ?? 1,
    field.shape[1] ?? grid.nth ?? 1,
    field.shape[2] ?? grid.nz ?? 1,
  ];
  const fullOrigin = cartesianScalarOrigin(grid, fullShape);
  const counts: [number, number, number] = [
    outputShape[0] ?? grid.nr ?? 1,
    outputShape[1] ?? grid.nth ?? 1,
    outputShape[2] ?? grid.nz ?? 1,
  ];
  const fallbackSpacing: [number, number, number] = [dx, dy, dz];
  const croppedCoordinates = grid.scalarCoordinatesM?.map((values, axis) => {
    if (values.length !== fullShape[axis]) return null;
    const start = axis < rank ? lower[axis] : 0;
    return values.slice(start, start + counts[axis]);
  });
  const hasExactCoordinates = croppedCoordinates?.every((values) =>
    values != null && values.length > 0 && values.every(Number.isFinite)
  ) ?? false;
  const exactCoordinates = hasExactCoordinates
    ? croppedCoordinates as [number[], number[], number[]]
    : null;
  const spacing = fallbackSpacing.map((value, axis) => {
    const coordinates = exactCoordinates?.[axis];
    return coordinates && coordinates.length > 1
      ? (coordinates.at(-1)! - coordinates[0]) / (coordinates.length - 1)
      : value;
  }) as [number, number, number];
  const origin = fallbackSpacing.map((value, axis) =>
    exactCoordinates?.[axis][0] ?? fullOrigin[axis] + lower[axis] * value
  ) as [number, number, number];
  const boundsLower = origin.map((value, axis) => {
    const coordinates = exactCoordinates?.[axis];
    const firstSpacing = coordinates && coordinates.length > 1
      ? coordinates[1] - coordinates[0]
      : spacing[axis];
    return value - 0.5 * firstSpacing;
  }) as [number, number, number];
  const boundsUpper = origin.map((_value, axis) => {
    const coordinates = exactCoordinates?.[axis];
    if (coordinates && coordinates.length > 1) {
      return coordinates.at(-1)! + 0.5 * (coordinates.at(-1)! - coordinates.at(-2)!);
    }
    return boundsLower[axis] + counts[axis] * spacing[axis];
  }) as [number, number, number];
  const zero: [number, number, number] = [0, 0, 0];
  const last: [number, number, number] = counts.map((count) => count - 1) as [number, number, number];

  return {
    field: { shape: outputShape, data: output },
    grid: {
      ...grid,
      nr: counts[0],
      nth: counts[1],
      nz: counts[2],
      // Always overwrite; spread carries uncropped arrays that no longer match.
      scalarCoordinatesM: exactCoordinates ?? undefined,
      scalarOriginM: origin,
      physicalBoundsM: { lower: boundsLower, upper: boundsUpper },
      validIndexRange: { lower: zero, upper: last },
      haloWidth: { lower: zero, upper: zero },
    },
    cropped: true,
  };
}
