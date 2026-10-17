import type { Npy3D, NpyN, SliceAxis } from "@/lib/platform/npy";
import type { GridInfo } from "@/lib/run/runOutputs";
import { terrainWindowToCartesianField } from "./cartesianInterp";

export interface TerrainSliceRemap {
  field: NpyN;
  grid: GridInfo | null;
  remapped: boolean;
}

/**
 * Convert a one-cell-thick vertical terrain window to separable physical
 * x/y/z coordinates before Plotly sees it. Horizontal model-level slices are
 * already honest plan views and deliberately retain their native level.
 */
export function remapTerrainVerticalSlice(
  field: NpyN,
  grid: GridInfo | null | undefined,
  sliceAxis: SliceAxis,
): TerrainSliceRemap {
  if (field.shape.length !== 3 || !grid?.terrainFollowing || sliceAxis === 2) {
    return { field, grid: grid ?? null, remapped: false };
  }
  if (field.shape[sliceAxis] !== 1) {
    throw new Error("Terrain vertical-slice remapping requires a one-cell-thick bounded window");
  }
  const physical = terrainWindowToCartesianField(field as Npy3D, grid);
  const [nr, nth, nz] = physical.field.shape;
  const lower = physical.origin.map((origin, axis) => origin - 0.5 * physical.spacing[axis]) as
    [number, number, number];
  const upper = physical.origin.map(
    (origin, axis) => origin + (physical.field.shape[axis] - 0.5) * physical.spacing[axis],
  ) as [number, number, number];
  return {
    field: physical.field,
    grid: {
      ...grid,
      nr,
      nth,
      nz,
      terrainFollowing: false,
      geometryKind: "physical",
      drM: physical.spacing[0],
      dyM: physical.spacing[1],
      dzM: physical.spacing[2],
      scalarOriginM: physical.origin,
      scalarCoordinatesM: physical.scalarCoordinatesM,
      physicalBoundsM: { lower, upper },
      validIndexRange: { lower: [0, 0, 0], upper: [nr - 1, nth - 1, nz - 1] },
      haloWidth: { lower: [0, 0, 0], upper: [0, 0, 0] },
    },
    remapped: true,
  };
}
