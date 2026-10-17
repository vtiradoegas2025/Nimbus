import type { PreparedVtkImage } from "./vtkPreparation";
import { volumeOwnedBounds, type VolumeBounds } from "./volumeBrickGeometry";

type PhysicalBounds = { lower: readonly [number, number, number]; upper: readonly [number, number, number] };

function box(bounds: PhysicalBounds): VolumeBounds {
  if (![...bounds.lower, ...bounds.upper].every(Number.isFinite)
      || bounds.lower.some((value, axis) => value >= bounds.upper[axis])) {
    throw new Error("Nested volume ownership requires finite positive bounds");
  }
  return [bounds.lower[0], bounds.upper[0], bounds.lower[1], bounds.upper[1], bounds.lower[2], bounds.upper[2]];
}

function intersection(a: VolumeBounds, b: VolumeBounds): VolumeBounds | null {
  const result = a.map((value, index) => index % 2 ? Math.min(value, b[index]) : Math.max(value, b[index])) as VolumeBounds;
  return [0, 1, 2].some((axis) => result[2 * axis] >= result[2 * axis + 1]) ? null : result;
}

export function partitionNestedVolume(prepared: PreparedVtkImage, layerBounds: PhysicalBounds,
  childBounds: PhysicalBounds | null, rootBounds: PhysicalBounds): PreparedVtkImage[] {
  const domain = box(layerBounds);
  const root = box(rootBounds);
  const child = childBounds ? intersection(domain, box(childBounds)) : null;
  const regions: VolumeBounds[] = [];
  if (!child) regions.push(domain);
  else {
    // Partition the layer globally before intersecting individual bricks. Local
    // subtraction alone can leave a nonseparable parent/child actor ordering.
    const remainder = [...domain] as VolumeBounds;
    for (let axis = 0; axis < 3; axis++) {
      const lower = 2 * axis, upper = lower + 1;
      if (remainder[lower] < child[lower]) {
        const slab = [...remainder] as VolumeBounds;
        slab[upper] = child[lower]; regions.push(slab);
      }
      if (remainder[upper] > child[upper]) {
        const slab = [...remainder] as VolumeBounds;
        slab[lower] = child[upper]; regions.push(slab);
      }
      remainder[lower] = child[lower]; remainder[upper] = child[upper];
    }
  }
  return regions.flatMap((region) => {
    const owned = intersection(volumeOwnedBounds(prepared), region);
    return owned ? [{ ...prepared, volumeClipBounds: owned,
      volumeUpperOpen: [0, 1, 2].map((axis) => Number(owned[2 * axis + 1] < root[2 * axis + 1])) as [number, number, number],
    }] : [];
  });
}
