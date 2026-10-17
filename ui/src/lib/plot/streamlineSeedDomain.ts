import type { GridInfo } from "@/lib/run/runOutputs";

type Triple = [number, number, number];

function axisSpan(
  grid: GridInfo | null,
  axis: number,
  fallbackCount: number,
  fallbackSpacing: number,
): number {
  const coordinates = grid?.scalarCoordinatesM?.[axis];
  if (coordinates && coordinates.length >= 2) {
    const span = coordinates.at(-1)! - coordinates[0];
    if (Number.isFinite(span) && span > 0) return span;
  }
  const count = [grid?.nr, grid?.nth, grid?.nz][axis];
  const spacing = [grid?.drM, grid?.dyM, grid?.dzM][axis];
  if (Number.isSafeInteger(count) && count! > 1 && Number.isFinite(spacing) && spacing! > 0) {
    return (count! - 1) * spacing!;
  }
  const bounds = grid?.physicalBoundsM;
  if (bounds) {
    const span = bounds.upper[axis] - bounds.lower[axis];
    if (Number.isFinite(span) && span > 0) return span;
  }
  return Math.max(0, fallbackCount - 1) * fallbackSpacing;
}

export function streamlineBoxSeedDimensions(
  grid: GridInfo | null,
  preparedShape: Triple,
  preparedSpacing: Triple,
  preparedOrigin: Triple,
): { halfSize: Triple; centerZ: number; domainSpan: Triple } {
  const domainSpan = preparedShape.map((count, axis) =>
    axisSpan(grid, axis, count, preparedSpacing[axis])) as Triple;
  const coordinateZ = grid?.scalarCoordinatesM?.[2]?.[0];
  const originZ = Number.isFinite(coordinateZ)
    ? coordinateZ!
    : Number.isFinite(grid?.scalarOriginM?.[2])
      ? grid!.scalarOriginM![2]
      : preparedOrigin[2];
  return {
    halfSize: [domainSpan[0] * 0.09, domainSpan[1] * 0.09, domainSpan[2] * 0.4],
    centerZ: originZ + domainSpan[2] * 0.45,
    domainSpan,
  };
}
