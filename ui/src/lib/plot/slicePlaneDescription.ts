import { sliceDisplayAxes, type SliceAxis } from "../platform/npy";
import type { GridInfo } from "../run/runOutputs";
import { axisLabel, scalarCoordinateKm } from "../science/gridCoords";

export function slicePlaneDescription(grid: GridInfo | null | undefined, axis: SliceAxis, validIndex: number, physicalHeight = false) {
  const { rowAxis, colAxis } = sliceDisplayAxes(axis);
  const counts = [grid?.nr, grid?.nth, grid?.nz];
  const sourceIndex = validIndex + (grid?.validIndexRange?.lower[axis] ?? 0);
  const coordinate = grid?.geometryKind === "index" ? null
    : scalarCoordinateKm(axis, sourceIndex, counts[axis] ?? 0, grid);
  const displayGrid = physicalHeight ? { ...grid, terrainFollowing: false } : grid;
  const plane = `${axisLabel(colAxis, displayGrid)}–${axisLabel(rowAxis, displayGrid)} ${axis === 2 ? "plan" : "section"}`;
  const location = coordinate == null ? `${axisLabel(axis, grid)} index ${validIndex}`
    : `${axisLabel(axis, grid)} = ${Number(coordinate.toPrecision(6))} km · index ${validIndex}`;
  return { plane, location };
}
