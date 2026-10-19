import type { GridSpacing } from "@/lib/science/cartesianInterp";
import type { VolumeBrickPlanItem } from "@/lib/plot/fieldBrickHierarchy";
import type { PreparedVtkImage } from "./vtkPreparation";

export type VolumeBounds = [number, number, number, number, number, number];

export function volumeHaloSamples(grid: GridSpacing): number {
  if (grid.terrainFollowing) return 0;
  for (const coordinates of grid.scalarCoordinatesM ?? []) {
    if (!coordinates || coordinates.length < 3) continue;
    const spacing = coordinates[1] - coordinates[0];
    if (!(spacing > 0) || coordinates.some((value, index) =>
      !Number.isFinite(value) || Math.abs(value - coordinates[0] - index * spacing) > Math.abs(spacing) * 1e-7)) return 0;
  }
  return 2;
}

export function withVolumeBrickOwnership(prepared: PreparedVtkImage, item: VolumeBrickPlanItem): PreparedVtkImage {
  if (!item.ownedOffset || !item.ownedShape) return prepared;
  if (item.ownedOffset.some((offset, axis) => !Number.isSafeInteger(offset) || offset < 0
      || !Number.isSafeInteger(item.ownedShape![axis]) || item.ownedShape![axis] < 1
      || offset + item.ownedShape![axis] > prepared.shape[axis])) {
    throw new Error("Volume brick ownership exceeds prepared support");
  }
  return { ...prepared, volumeOwnership: { offset: item.ownedOffset, shape: item.ownedShape } };
}

export function volumeOwnedBounds(prepared: PreparedVtkImage): VolumeBounds {
  if (prepared.volumeClipBounds) return prepared.volumeClipBounds;
  const offset = prepared.volumeOwnership?.offset ?? [0, 0, 0];
  const shape = prepared.volumeOwnership?.shape ?? prepared.shape;
  return [0, 1, 2].flatMap((axis) => [
    prepared.origin[axis] + (offset[axis] - 0.5) * prepared.spacing[axis],
    prepared.origin[axis] + (offset[axis] + shape[axis] - 0.5) * prepared.spacing[axis],
  ]) as VolumeBounds;
}

export function volumeOpenUpperFaces(prepared: PreparedVtkImage): [number, number, number] {
  if (prepared.volumeUpperOpen) return prepared.volumeUpperOpen;
  const ownership = prepared.volumeOwnership;
  return [0, 1, 2].map((axis) => Number(ownership != null
    && ownership.offset[axis] + ownership.shape[axis] < prepared.shape[axis])) as [number, number, number];
}

export function volumeOwnedScalarRows(prepared: PreparedVtkImage): Float32Array[] {
  if (prepared.volumeClipBounds) {
    const bounds = prepared.volumeClipBounds;
    const open = volumeOpenUpperFaces(prepared);
    const offset = [0, 1, 2].map((axis) => Math.max(0,
      Math.ceil((bounds[2 * axis] - prepared.origin[axis]) / prepared.spacing[axis] - 1e-9))) as [number, number, number];
    const shape = offset.map((start, axis) => {
      const upper = (bounds[2 * axis + 1] - prepared.origin[axis]) / prepared.spacing[axis];
      const end = open[axis] ? Math.ceil(upper - 1e-9) : Math.floor(upper + 1e-9) + 1;
      return Math.max(0, Math.min(prepared.shape[axis], end) - start);
    }) as [number, number, number];
    return volumeOwnedScalarRows({ ...prepared, volumeClipBounds: undefined, volumeOwnership: { offset, shape } });
  }
  if (!prepared.volumeOwnership) return [prepared.values];
  const { offset, shape } = prepared.volumeOwnership;
  const [nx, ny] = prepared.shape;
  const rows: Float32Array[] = [];
  for (let z = 0; z < shape[2]; z++) for (let y = 0; y < shape[1]; y++) {
    const start = offset[0] + nx * (offset[1] + y + ny * (offset[2] + z));
    rows.push(prepared.values.subarray(start, start + shape[0]));
  }
  return rows;
}

export function uniqueVolumeImages<T extends { prepared: PreparedVtkImage }>(bricks: readonly T[]): T[] {
  return [...new Map(bricks.map((brick) => [brick.prepared.values, brick])).values()];
}

export type VolumeOrder = { items: number[]; unresolved: boolean }
  | { axis: number; split: number; lower: VolumeOrder; upper: VolumeOrder; unresolved: boolean };

export function buildVolumeOrder(bounds: readonly VolumeBounds[]): VolumeOrder {
  if (bounds.some((box) => box.some((value) => !Number.isFinite(value))
      || [0, 1, 2].some((axis) => box[axis * 2] >= box[axis * 2 + 1]))) {
    throw new Error("Volume ordering requires finite positive bounds");
  }
  const visit = (items: number[]): VolumeOrder => {
    if (items.length < 2) return { items, unresolved: false };
    let best: { axis: number; split: number; lower: number[]; upper: number[]; score: number } | null = null;
    for (let axis = 0; axis < 3; axis++) {
      const candidates = new Set(items.flatMap((index) => [bounds[index][axis * 2], bounds[index][axis * 2 + 1]]));
      for (const split of candidates) {
        const tolerance = Math.max(1, Math.abs(split)) * 1e-10;
        const lower: number[] = [];
        const upper: number[] = [];
        for (const index of items) {
          if (bounds[index][axis * 2 + 1] <= split + tolerance) lower.push(index);
          else if (bounds[index][axis * 2] >= split - tolerance) upper.push(index);
        }
        if (!lower.length || !upper.length || lower.length + upper.length !== items.length) continue;
        const score = Math.abs(lower.length - upper.length);
        if (!best || score < best.score) best = { axis, split, lower, upper, score };
      }
    }
    if (!best) return { items, unresolved: true };
    const lower = visit(best.lower);
    const upper = visit(best.upper);
    return { axis: best.axis, split: best.split, lower, upper, unresolved: lower.unresolved || upper.unresolved };
  };
  return visit(bounds.map((_, index) => index));
}

export function orderedVolumeIndices(tree: VolumeOrder, eye: readonly number[], direction?: readonly number[]): number[] {
  if ("items" in tree) return tree.items;
  const upperFirst = direction ? direction[tree.axis] > 0 : eye[tree.axis] < tree.split;
  const first = upperFirst ? tree.upper : tree.lower;
  const second = upperFirst ? tree.lower : tree.upper;
  return [...orderedVolumeIndices(first, eye, direction), ...orderedVolumeIndices(second, eye, direction)];
}
