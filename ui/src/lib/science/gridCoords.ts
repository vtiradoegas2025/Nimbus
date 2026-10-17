import type { SliceAxis } from "@/lib/platform/npy";
import {
  cartesianDyMeters,
  cartesianScalarAxisOrigin,
  isCartesianGrid,
  type GridSpacing,
} from "./cartesianInterp";

/* Physical-coordinate helpers for the Cartesian (x, y, z) grid; shared across 2D plots. */

/** Axis letters for the Cartesian grid. */
export const CARTESIAN_AXIS_LABELS: Record<SliceAxis, string> = { 0: "x", 1: "y", 2: "z" };

/** Bare axis label. Terrain-following vertical uses "k" (non-separable). */
export function axisLabel(axis: SliceAxis, grid: GridSpacing | null | undefined): string {
  if (axis === 2 && grid?.terrainFollowing) return "k";
  return CARTESIAN_AXIS_LABELS[axis];
}

/** Slice-axis dropdown options; terrain vertical is "k". */
export function axisOptions(
  grid: GridSpacing | null | undefined,
): ReadonlyArray<{ value: SliceAxis; label: string }> {
  return [
    { value: 0, label: CARTESIAN_AXIS_LABELS[0] },
    { value: 1, label: CARTESIAN_AXIS_LABELS[1] },
    { value: 2, label: grid?.terrainFollowing ? "k" : CARTESIAN_AXIS_LABELS[2] },
  ];
}

/** Cell-center coordinates (km) from an explicit first-center origin. */
function centers(cellCount: number, spacingKm: number, originKm: number): number[] {
  return Array.from({ length: cellCount }, (_, i) => originKm + i * spacingKm);
}

function scalarOriginM(
  grid: GridSpacing,
  axis: SliceAxis,
  cellCount: number,
  spacingM: number,
): number {
  return cartesianScalarAxisOrigin(grid, axis, cellCount, spacingM);
}

function axisSpacingM(axis: SliceAxis, grid: GridSpacing): number | undefined {
  if (axis === 0) return grid.drM ?? undefined;
  if (axis === 1) {
    if (grid.dyM != null && grid.dyM > 0) return grid.dyM;
    if (grid.drM != null && grid.drM > 0) return cartesianDyMeters(grid, grid.drM);
    return undefined;
  }
  return grid.dzM ?? undefined;
}

function exactAxisMeters(
  axis: SliceAxis,
  cellCount: number,
  grid: GridSpacing,
): readonly number[] | null {
  const values = grid.scalarCoordinatesM?.[axis];
  return values?.length === cellCount
    && values.every(Number.isFinite)
    && values.every((value, index) => index === 0 || value > values[index - 1])
    ? values
    : null;
}

function representativeSpacing(values: readonly number[]): number {
  return values.length > 1 ? (values.at(-1)! - values[0]) / (values.length - 1) : 1;
}

export function hasPhysicalAxisCoordinates(
  axis: SliceAxis,
  cellCount: number,
  grid: GridSpacing | null | undefined,
): boolean {
  if (!grid || cellCount <= 0 || !isCartesianGrid(grid.coordinateSystem)) return false;
  if (axis === 2 && grid.terrainFollowing) return false;
  return exactAxisMeters(axis, cellCount, grid) != null
    || (axisSpacingM(axis, grid) ?? 0) > 0;
}

/** Interpretation of axis values; "level" is populated but non-physical (terrain). */
export type AxisUnit =
  /** Physical kilometres. */
  | "km"
  /** Model level k on a terrain-following grid: an integer, not a height. */
  | "level"
  /** Bare cell index, because the grid declared no usable spacing. */
  | "index";

export interface AxisCoords {
  /** Coordinate per cell along this axis, in `unit`, or undefined to fall back
   *  to Plotly's default cell-index axis. */
  values: number[] | undefined;
  /** Axis title with units. */
  title: string;
  /** Plotly hover format suffix appended to %{x} / %{y}. */
  hoverFmt: string;
  /** Spacing between adjacent cell centers in `unit`, or 1 when the axis falls
   *  back to cell indices. Used to scale vector arrows per axis. */
  spacing: number;
  /** Interpretation of `values` and `spacing`. */
  unit: AxisUnit;
}

/**
 * True when two axes sample the same physical positions (matching unit + coordinates).
 * Cell count alone is insufficient; uses unit-aware tolerance for float64 reconstruction noise.
 */
export function axesAreCoLocated(a: AxisCoords, b: AxisCoords): boolean {
  if (a.unit !== b.unit) return false;
  if (a.unit !== "km") return (a.values?.length ?? 0) === (b.values?.length ?? 0);
  if (a.values == null || b.values == null) return false;
  if (a.values.length !== b.values.length || a.values.length === 0) return false;
  const first = a.values[0];
  const last = a.values[a.values.length - 1];
  // Tolerance: fraction of a cell (span-relative) plus float64 magnitude floor.
  const cellSpacing = a.values.length > 1
    ? Math.abs(last - first) / (a.values.length - 1)
    : 0;
  const magnitude = Math.max(Math.abs(first), Math.abs(last));
  const tolerance = Math.max(cellSpacing * 1e-6, magnitude * 1e-12, 1e-12);
  for (let index = 0; index < a.values.length; index += 1) {
    if (Math.abs(a.values[index] - b.values[index]) > tolerance) return false;
  }
  return true;
}

/**
 * True when both display axes are horizontal km-physical; enables 1:1 aspect lock.
 * Always false for vertical planes (conventional exaggeration) and non-km axes.
 */
export function planeSupportsLockedAspect(
  rowAxis: SliceAxis,
  colAxis: SliceAxis,
  rowCoords: AxisCoords,
  colCoords: AxisCoords,
): boolean {
  if (rowAxis === 2 || colAxis === 2) return false;
  return rowCoords.unit === "km" && colCoords.unit === "km";
}

/** Plotly axis fragments for 1:1 aspect. `constrain:"domain"` shrinks the plot area (not the range). */
export const LOCKED_ASPECT_AXIS = { constrain: "domain" } as const;

/** Assignable to Plotly `AxisName` without importing plotly types. */
export type LockedAspectAnchor =
  | "x" | "x2" | "x3" | "x4"
  | "y" | "y2" | "y3" | "y4";

/** Partner axis of a locked pair; `anchor` varies per subplot. */
export function lockedAspectPartnerAxis(anchor: LockedAspectAnchor = "x") {
  return { scaleanchor: anchor, scaleratio: 1, constrain: "domain" } as const;
}

/** Physical scalar-center coordinate (km) for one cell index. Uses plotted-field cellCount, not manifest. */
export function scalarCoordinateKm(
  axis: SliceAxis,
  index: number,
  cellCount: number,
  grid: GridSpacing | null | undefined,
): number | undefined {
  if (!grid || cellCount <= 0 || !isCartesianGrid(grid.coordinateSystem)) return undefined;
  if (axis === 2 && grid.terrainFollowing) return undefined;
  if (index < 0 || index >= cellCount) return undefined;
  const exact = exactAxisMeters(axis, cellCount, grid);
  if (exact) return exact[index] / 1000;
  const spacingM = axisSpacingM(axis, grid);
  if (spacingM == null || !(spacingM > 0)) return undefined;
  return (scalarOriginM(grid, axis, cellCount, spacingM) + index * spacingM) / 1000;
}

/** Nearest z scalar-center index for a height in km. Returns 0 when vertical metadata is unavailable. */
export function heightKmToZIndex(
  km: number,
  grid: GridSpacing | null | undefined,
  dim: number,
): number {
  if (!grid || dim <= 0) return 0;
  if (grid.terrainFollowing) return 0;
  const exact = exactAxisMeters(2, dim, grid);
  if (exact) {
    const targetM = km * 1000;
    let bestIndex = 0;
    let bestDistance = Math.abs(exact[0] - targetM);
    for (let index = 1; index < exact.length; index += 1) {
      const distance = Math.abs(exact[index] - targetM);
      if (distance >= bestDistance) break;
      bestIndex = index;
      bestDistance = distance;
    }
    return bestIndex;
  }
  if (!grid.dzM || !(grid.dzM > 0)) return 0;
  const originM = scalarOriginM(grid, 2, dim, grid.dzM);
  const fractionalIndex = ((km * 1000) - originM) / grid.dzM;
  return Math.max(0, Math.min(dim - 1, Math.round(fractionalIndex)));
}

export function axisCoords(
  axis: SliceAxis,
  cellCount: number,
  grid: GridSpacing | null,
): AxisCoords {
  if (!grid || cellCount <= 0) {
    return { values: undefined, title: CARTESIAN_AXIS_LABELS[axis], hoverFmt: "", spacing: 1, unit: "index" };
  }

  if (isCartesianGrid(grid.coordinateSystem)) {
    if (axis === 2 && grid.terrainFollowing) {
      return {
        values: Array.from({ length: cellCount }, (_, index) => index),
        title: "model level k",
        hoverFmt: ":.0f",
        spacing: 1,
        unit: "level",
      };
    }
    const exact = exactAxisMeters(axis, cellCount, grid);
    if (exact) {
      return {
        values: exact.map((value) => value / 1000),
        title: `${CARTESIAN_AXIS_LABELS[axis]} (km)`,
        hoverFmt: ":.2f km",
        spacing: representativeSpacing(exact) / 1000,
        unit: "km",
      };
    }
    if (axis === 0 && grid.drM) {
      const dxKm = grid.drM / 1000;
      const originKm = scalarOriginM(grid, 0, cellCount, grid.drM) / 1000;
      return { values: centers(cellCount, dxKm, originKm), title: "x (km)", hoverFmt: ":.2f km", spacing: dxKm, unit: "km" };
    }
    if (axis === 1) {
      const dyM = grid.drM ? cartesianDyMeters(grid, grid.drM) : grid.dyM;
      if (dyM) {
        const dyKm = dyM / 1000;
        const originKm = scalarOriginM(grid, 1, cellCount, dyM) / 1000;
        return { values: centers(cellCount, dyKm, originKm), title: "y (km)", hoverFmt: ":.2f km", spacing: dyKm, unit: "km" };
      }
    }
    if (axis === 2 && grid.dzM) {
      const dzKm = grid.dzM / 1000;
      const originKm = scalarOriginM(grid, 2, cellCount, grid.dzM) / 1000;
      return { values: centers(cellCount, dzKm, originKm), title: "z (km)", hoverFmt: ":.2f km", spacing: dzKm, unit: "km" };
    }
  }

  // Missing/degenerate spacing (or a non-sentinel dtheta) -> bare cell-index axis.
  return { values: undefined, title: CARTESIAN_AXIS_LABELS[axis], hoverFmt: "", spacing: 1, unit: "index" };
}
