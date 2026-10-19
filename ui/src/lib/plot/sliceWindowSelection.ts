import type { NpyN, SliceAxis } from "@/lib/platform/npy";

export function loadedSliceWindowIndex(
  field: NpyN | null,
  axis: SliceAxis,
  validLower: readonly number[] | undefined,
  requestedIndex: number,
): number {
  const origin = field?.windowGeometry?.origin?.[axis];
  return origin == null ? requestedIndex : origin - (validLower?.[axis] ?? 0);
}

export function sourceSliceControlsAvailable(rank: number | null, count: number, error: string | null): boolean {
  return rank === 3 && Number.isSafeInteger(count) && count > 0 && error == null;
}
