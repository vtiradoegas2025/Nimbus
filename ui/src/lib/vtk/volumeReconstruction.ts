import { levelReadWindow, type FieldBrickHierarchy, type VolumeBrickPlan, type VolumeBrickPlanItem } from "../plot/fieldBrickHierarchy";
import { windowAxisCoordinates } from "../science/windowedGrid";
import type { GridSpacing } from "../science/cartesianInterp";
import type { Npy3D } from "../platform/npy";

type Triple = [number, number, number];
export interface VolumeReconstruction { shape: Triple; origin: Triple; spacing: Triple }

function lowerBound(values: readonly number[], target: number): number {
  let low = 0, high = values.length;
  while (low < high) {
    const mid = (low + high) >>> 1;
    if (values[mid] < target) low = mid + 1; else high = mid;
  }
  return low;
}

export function needsVolumeReconstruction(grid: GridSpacing): boolean {
  return Boolean(grid.terrainFollowing || grid.scalarCoordinatesM?.some((axis) => axis.length > 2
    && axis.some((value, index) => Math.abs(value - axis[0] - index * (axis[1] - axis[0])) > Math.abs(axis[1] - axis[0]) * 1e-7)));
}

export function reconstructVolumePlan(hierarchy: FieldBrickHierarchy, grid: GridSpacing, plan: VolumeBrickPlan): VolumeBrickPlan {
  if (!needsVolumeReconstruction(grid) || !plan.items.length) return plan;
  const level = hierarchy.levels[plan.level];
  const sourceCounts = [(grid as { nr?: number }).nr, (grid as { nth?: number }).nth, (grid as { nz?: number }).nz]
    .map((value, axis) => value ?? hierarchy.sourceShape[axis] + hierarchy.sourceIndexOffset[axis]) as Triple;
  const sourceGrid = grid.horizontalScalarOriginM ? { ...grid, scalarOriginM: [grid.horizontalScalarOriginM[0], grid.horizontalScalarOriginM[1], 0] as Triple } : grid;
  const axes = [0, 1, 2].map((axis) => windowAxisCoordinates(sourceGrid, axis, sourceCounts[axis],
    hierarchy.sourceIndexOffset[axis], hierarchy.sourceShape[axis], level.stride[axis], level.shape[axis], false));
  const origin = axes.map((axis) => axis[0]) as Triple;
  const spacing = axes.map((axis, i) => axis.length > 1 ? (axis.at(-1)! - axis[0]) / (axis.length - 1)
    : ([grid.drM, grid.dyM ?? grid.drM, grid.dzM][i] ?? 1) * level.stride[i]) as Triple;
  if (grid.terrainFollowing) {
    if (!grid.physicalBoundsM) throw new Error("Terrain volume reconstruction requires global physical bounds");
    const lower = grid.physicalBoundsM.lower[2], upper = grid.physicalBoundsM.upper[2];
    if (!Number.isFinite(lower) || !(upper > lower)) throw new Error("Invalid terrain volume bounds");
    spacing[2] = (upper - lower) / level.shape[2];
    origin[2] = lower + 0.5 * spacing[2];
  }
  if (spacing.some((value) => !(value > 0) || !Number.isFinite(value))) throw new Error("Invalid volume reconstruction spacing");
  let owners = plan.items.map((item) => ({ item,
    start: item.window.origin.map((value, axis) => (value - hierarchy.sourceIndexOffset[axis]) / level.stride[axis] + (item.ownedOffset?.[axis] ?? 0)) as Triple,
    extent: item.ownedShape ?? item.window.extent.map((value, axis) => Math.ceil(value / level.stride[axis])) as Triple,
  }));
  // Model-index slabs overlap in physical z over terrain. Own horizontal columns instead.
  if (grid.terrainFollowing) {
    const selected = new Set(plan.items.map((item) => item.brickIndex.slice(0, 2).join(":")));
    const columns = new Map<string, typeof owners[number]>();
    for (const brick of level.bricks) {
      const key = brick.index.slice(0, 2).join(":");
      if (!selected.has(key) || columns.has(key)) continue;
      const item = plan.items.find((item) => item.brickIndex[0] === brick.index[0] && item.brickIndex[1] === brick.index[1])!;
      columns.set(key, { item: { ...item, brickId: `${brick.id}:column` },
        start: [brick.origin[0], brick.origin[1], 0], extent: [brick.extent[0], brick.extent[1], level.shape[2]] });
    }
    owners = [...columns.values()];
  }
  const items: VolumeBrickPlanItem[] = [];
  for (const owner of owners) {
    const cut = (axis: number, index: number) => {
      if (index <= 0) return 0;
      if (index >= level.shape[axis]) return level.shape[axis];
      const face = (axes[axis][index - 1] + axes[axis][index]) / 2;
      return Math.max(0, Math.min(level.shape[axis], Math.ceil((face - origin[axis]) / spacing[axis] - 1e-9)));
    };
    const first = owner.start.map((value, axis) => grid.terrainFollowing && axis === 2 ? 0 : cut(axis, value)) as Triple;
    const end = owner.start.map((value, axis) => grid.terrainFollowing && axis === 2 ? level.shape[2] : cut(axis, value + owner.extent[axis])) as Triple;
    if (end.some((value, axis) => value <= first[axis])) continue;
    const supportFirst = first.map((value) => Math.max(0, value - 2)) as Triple;
    const supportEnd = end.map((value, axis) => Math.min(level.shape[axis], value + 2)) as Triple;
    const readFirst = supportFirst.map((value, axis) => grid.terrainFollowing && axis === 2 ? 0
      : Math.max(0, lowerBound(axes[axis], origin[axis] + value * spacing[axis]) - 1)) as Triple;
    const readEnd = supportEnd.map((value, axis) => grid.terrainFollowing && axis === 2 ? level.shape[2]
      : Math.min(level.shape[axis], lowerBound(axes[axis], origin[axis] + (value - 1) * spacing[axis]) + 2)) as Triple;
    const shape = supportEnd.map((value, axis) => value - supportFirst[axis]) as Triple;
    const reconstruction = { shape, origin: supportFirst.map((value, axis) => origin[axis] + value * spacing[axis]) as Triple, spacing };
    items.push({ ...owner.item, window: levelReadWindow(hierarchy, level, readFirst,
      readEnd.map((value, axis) => value - readFirst[axis]) as Triple, grid.terrainFollowing),
      reconstruction, preparedScalarBytes: shape.reduce((n, size) => n * size, 4),
      ownedOffset: first.map((value, axis) => value - supportFirst[axis]) as Triple,
      ownedShape: end.map((value, axis) => value - first[axis]) as Triple });
  }
  return { ...plan, items, residentBytes: items.reduce((sum, item) => sum + item.preparedScalarBytes, 0), haloSamples: 2 };
}

export function sampleReconstruction(field: Npy3D, axes: readonly (readonly number[])[], target: VolumeReconstruction): Float32Array {
  const maps = axes.map((axis, a) => {
    if (axis.length !== field.shape[a] || axis.some((value, i) => !Number.isFinite(value) || i > 0 && value <= axis[i - 1])) throw new Error("Invalid reconstruction source coordinates");
    return Array.from({ length: target.shape[a] }, (_, i) => {
      const value = target.origin[a] + i * target.spacing[a];
      const upper = Math.min(axis.length - 1, lowerBound(axis, value));
      const lower = Math.max(0, upper - 1);
      const fraction = upper === lower ? 0 : Math.max(0, Math.min(1, (value - axis[lower]) / (axis[upper] - axis[lower])));
      const weight = fraction < 1e-12 ? 0 : fraction > 1 - 1e-12 ? 1 : fraction;
      return { lower, upper, weight };
    });
  });
  const [nx, ny, nz] = target.shape;
  const values = new Float32Array(nx * ny * nz);
  for (let x = 0; x < nx; x++) for (let y = 0; y < ny; y++) for (let z = 0; z < nz; z++) {
    const mx = maps[0][x], my = maps[1][y], mz = maps[2][z];
    let sum = 0, invalid = false;
    for (let mask = 0; mask < 8; mask++) {
      const sx = mask & 1 ? mx.upper : mx.lower;
      const sy = mask & 2 ? my.upper : my.lower;
      const sz = mask & 4 ? mz.upper : mz.lower;
      const weight = (mask & 1 ? mx.weight : 1 - mx.weight)
        * (mask & 2 ? my.weight : 1 - my.weight) * (mask & 4 ? mz.weight : 1 - mz.weight);
      if (weight === 0) continue;
      const value = field.data[(sx * field.shape[1] + sy) * field.shape[2] + sz];
      if (!Number.isFinite(value)) invalid = true;
      else sum += weight * value;
    }
    values[(x * ny + y) * nz + z] = invalid ? NaN : sum;
  }
  return values;
}
