export interface CartesianCoordinateGrid {
  drM?: number | null;
  dyM?: number | null;
  dthetaRad?: number | null;
  dzM?: number | null;
  horizontalScalarOriginM?: readonly [number, number] | null;
  scalarOriginM?: readonly [number, number, number] | null;
  scalarCoordinatesM?: readonly (readonly number[])[];
  physicalBoundsM?: { lower: readonly number[] } | null;
  validIndexRange?: { lower: readonly number[] } | null;
}

export function cartesianDyMeters(grid: CartesianCoordinateGrid, drM: number): number {
  if (grid.dyM != null && Number.isFinite(grid.dyM) && grid.dyM > 0) return grid.dyM;
  // Older Cartesian manifests stored metres in the cylindrical spacing slot.
  const legacy = grid.dthetaRad;
  return legacy != null && Number.isFinite(legacy) && legacy > 2 * Math.PI ? legacy : drM;
}

export function cartesianAxisSpacing(grid: CartesianCoordinateGrid, axis: number): number {
  if (axis === 0) return grid.drM ?? 1;
  if (axis === 1) return cartesianDyMeters(grid, grid.drM ?? 1);
  return grid.dzM ?? 1;
}

export function cartesianScalarAxisOrigin(
  grid: CartesianCoordinateGrid,
  axis: number,
  cellCount: number,
  spacingM: number,
): number {
  const exact = grid.scalarCoordinatesM?.[axis];
  if (exact?.length === cellCount && Number.isFinite(exact[0])) return exact[0];
  const declared = grid.scalarOriginM?.[axis]
    ?? (axis < 2 ? grid.horizontalScalarOriginM?.[axis] : undefined);
  if (declared != null && Number.isFinite(declared)) return declared;
  const lower = grid.physicalBoundsM?.lower[axis];
  if (lower != null && Number.isFinite(lower)) {
    // Physical bounds enclose valid cells, whereas this origin includes halos.
    return lower + (0.5 - (grid.validIndexRange?.lower[axis] ?? 0)) * spacingM;
  }
  return axis === 2 ? 0.5 * spacingM : -(cellCount - 1) * spacingM * 0.5;
}
