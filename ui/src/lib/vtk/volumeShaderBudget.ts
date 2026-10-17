const settledDistances = new WeakMap<object, number>();

export function setSettledVolumeDistance(mapper: object, distance: number): void {
  if (Number.isFinite(distance) && distance > 0) settledDistances.set(mapper, distance);
}

export function settledVolumeDistance(mapper: object): number | undefined {
  return settledDistances.get(mapper);
}

export function volumePlanningSampleDistance(mapper: object, liveDistance: number): number {
  return Math.min(settledDistances.get(mapper) ?? liveDistance, liveDistance);
}

const flatPreviews = new WeakSet<object>();
export function setFlatVolumePreview(mapper: object, active: boolean): void {
  if (active) flatPreviews.add(mapper); else flatPreviews.delete(mapper);
}
export function flatVolumePreview(mapper: object): boolean { return flatPreviews.has(mapper); }
