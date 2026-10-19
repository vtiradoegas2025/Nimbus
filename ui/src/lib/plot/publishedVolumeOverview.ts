import { levelReadWindow, type FieldBrickHierarchy } from "./fieldBrickHierarchy";
import { planBoundedVolumeWindows } from "./fieldLoadPlanning";
import type { FieldLoadWindow, GridInfo } from "../run/runOutputs";

export interface PublishedVolumeOverviewPlan {
  level: number;
  windows: FieldLoadWindow[];
  sampledCells: number;
  disclosure: string;
}

export class PublishedVolumeOverviewUnavailableError extends Error {}

export function planPublishedVolumeOverview(
  hierarchy: FieldBrickHierarchy,
  grid: GridInfo,
  maxCells: number,
): PublishedVolumeOverviewPlan {
  if (!Number.isSafeInteger(maxCells) || maxCells < 8) throw new Error("Invalid overview cell budget");
  const level = hierarchy.levels.find(level => level.source.kind === "npy_lod"
    && (!grid.terrainFollowing || level.source.geometry != null)
    && level.shape.reduce((cells, count) => cells * count, 1) <= maxCells);
  if (!level) throw new PublishedVolumeOverviewUnavailableError("No published overview fits this scene's working set");
  const tiled = planBoundedVolumeWindows({
    nr: level.shape[0], nth: level.shape[1], nz: level.shape[2],
    drM: 1, dyM: 1, dzM: 1, terrainFollowing: grid.terrainFollowing,
  }, maxCells);
  if (!tiled.ok) throw new Error(tiled.reason);
  if (tiled.plan.stride.some(stride => stride !== 1)) throw new Error("Published overview was unexpectedly reduced again");
  const windows = tiled.plan.windows.map(window => levelReadWindow(
    hierarchy, level, window.origin as [number, number, number], window.extent as [number, number, number], grid.terrainFollowing,
  ));
  return {
    level: level.level, windows, sampledCells: tiled.plan.sampledCells,
    disclosure: `published cell-mean overview ${level.stride.join("×")} · display ${level.shape.join("×")}`,
  };
}
