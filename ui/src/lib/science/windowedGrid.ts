import { cartesianAxisSpacing as axisSpacing, cartesianScalarAxisOrigin as sourceAxisOrigin } from "./cartesianCoordinates";
import type { NpyN } from "@/lib/platform/npy";

/** Structural grid subset shared by Plotly and VTK window consumers. */
export interface WindowAdjustableGrid {
  nr?: number;
  nth?: number;
  nz?: number;
  drM?: number | null;
  dyM?: number | null;
  dthetaRad?: number | null;
  horizontalScalarOriginM?: [number, number] | null;
  dzM?: number | null;
  scalarOriginM?: [number, number, number] | null;
  scalarCoordinatesM?: [number[], number[], number[]];
  physicalBoundsM?: {
    lower: [number, number, number];
    upper: [number, number, number];
  } | null;
  validIndexRange?: {
    lower: [number, number, number];
    upper: [number, number, number];
  } | null;
  haloWidth?: {
    lower: [number, number, number];
    upper: [number, number, number];
  } | null;
}

/** Coordinates of sampled source cells or aggregate block centres. */
export function windowAxisCoordinates(
  grid: WindowAdjustableGrid,
  axis: number,
  sourceCount: number,
  origin: number,
  extent: number,
  stride: number,
  outputCount: number,
  pointSample: boolean,
): number[] {
  const exact = grid.scalarCoordinatesM?.[axis];
  const spacing = axisSpacing(grid, axis);
  const base = sourceAxisOrigin(grid, axis, sourceCount, spacing);
  const end = origin + extent;
  return Array.from({ length: outputCount }, (_, output) => {
    const start = origin + output * stride;
    if (pointSample) return exact?.length === sourceCount ? exact[start] : base + start * spacing;
    const stop = Math.min(end, start + stride);
    if (exact?.length === sourceCount) {
      let total = 0;
      let count = 0;
      for (let index = start; index < stop; index += 1) {
        const coordinate = exact[index];
        if (Number.isFinite(coordinate)) {
          total += coordinate;
          count += 1;
        }
      }
      if (count > 0) return total / count;
    }
    return base + (start + 0.5 * (stop - start - 1)) * spacing;
  });
}

function separableCentroidAxis(field: NpyN, axis: number): number[] | null {
  const centroids = field.windowGeometry?.centroidsM;
  const source = axis === 0 ? centroids?.x : axis === 1 ? centroids?.y : centroids?.z;
  if (!source || field.shape.length !== 3 || source.length !== field.data.length) return null;
  const values = new Array<number>(field.shape[axis]);
  const seen = new Uint8Array(field.shape[axis]);
  for (let linear = 0; linear < source.length; linear += 1) {
    const i = Math.floor(linear / (field.shape[1] * field.shape[2]));
    const j = Math.floor(linear / field.shape[2]) % field.shape[1];
    const k = linear % field.shape[2];
    const index = axis === 0 ? i : axis === 1 ? j : k;
    const value = source[linear];
    if (!Number.isFinite(value)) return null;
    if (!seen[index]) {
      values[index] = value;
      seen[index] = 1;
    } else {
      const tolerance = 1e-9 * Math.max(1, Math.abs(values[index]), Math.abs(value));
      if (Math.abs(values[index] - value) > tolerance) return null;
    }
  }
  return seen.every((entry) => entry === 1) ? values : null;
}

/**
 * Replace full-source grid metadata with the physical sampling lattice of
 * a bounded field-window response. The resulting valid range is local and
 * halo-free, so downstream crop logic cannot apply the source halo twice.
 */
export function gridForWindowedField<T extends WindowAdjustableGrid>(
  field: NpyN,
  grid: T | null | undefined,
): T | null {
  if (!grid) return null;
  const geometry = field.windowGeometry;
  if (!geometry) return grid;
  const rank = field.shape.length;
  const sourceShape = geometry.sourceShape;
  const origin = geometry.origin;
  const extent = geometry.extent;
  const stride = geometry.stride;
  if (!sourceShape || !origin || !extent || !stride
      || [sourceShape, origin, extent, stride].some((values) => values.length !== rank)
      || rank < 1 || rank > 3) {
    return grid;
  }

  const coordinates = Array.from({ length: 3 }, (_, axis) => {
    if (axis >= rank) return grid.scalarCoordinatesM?.[axis] ?? [];
    const physicalCentroids = separableCentroidAxis(field, axis);
    if (physicalCentroids) return physicalCentroids;
    return windowAxisCoordinates(
      grid,
      axis,
      sourceShape[axis],
      origin[axis],
      extent[axis],
      stride[axis],
      field.shape[axis],
      geometry.sampleLocation === "source_cell_center",
    );
  }) as [number[], number[], number[]];
  const counts: [number, number, number] = [
    field.shape[0] ?? grid.nr ?? 1,
    field.shape[1] ?? grid.nth ?? 1,
    field.shape[2] ?? grid.nz ?? 1,
  ];
  const spacing = coordinates.map((axis, index) => axis.length > 1
    ? (axis.at(-1)! - axis[0]) / (axis.length - 1)
    : axisSpacing(grid, index)) as [number, number, number];
  const scalarOrigin: [number, number, number] = coordinates.map((axis, index) =>
    axis[0] ?? grid.scalarOriginM?.[index] ?? 0
  ) as [number, number, number];
  const lower = scalarOrigin.map((value, axis) => value - 0.5 * spacing[axis]) as
    [number, number, number];
  const upper = scalarOrigin.map((value, axis) =>
    value + Math.max(0, counts[axis] - 1) * spacing[axis] + 0.5 * spacing[axis]
  ) as [number, number, number];
  const zero: [number, number, number] = [0, 0, 0];
  const last = counts.map((count) => count - 1) as [number, number, number];

  return {
    ...grid,
    nr: counts[0],
    nth: counts[1],
    nz: counts[2],
    // Exact block centres are authoritative. Omitting declared spacings makes
    // VTK regularize a shortened final block instead of pretending it is full.
    drM: undefined,
    dyM: undefined,
    dzM: undefined,
    scalarCoordinatesM: coordinates,
    scalarOriginM: scalarOrigin,
    physicalBoundsM: { lower, upper },
    validIndexRange: { lower: zero, upper: last },
    haloWidth: { lower: zero, upper: zero },
  };
}
