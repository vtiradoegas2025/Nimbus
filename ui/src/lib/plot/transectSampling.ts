import { axisCoords } from "@/lib/science/gridCoords";
import type { Npy3D } from "@/lib/platform/npy";
import type { GridInfo } from "@/lib/run/runOutputs";

export interface PhysicalPoint2D { xKm: number; yKm: number }

export interface VerticalTransect {
  distanceKm: number[];
  zKm: number[];
  values: number[][];
  points: PhysicalPoint2D[];
  sampleCount: number;
}

interface Bracket { lower: number; upper: number; weight: number }

function bracket(values: readonly number[], target: number): Bracket {
  if (target <= values[0]) return { lower: 0, upper: 0, weight: 0 };
  if (target >= values.at(-1)!) return { lower: values.length - 1, upper: values.length - 1, weight: 0 };
  let low = 0;
  let high = values.length - 1;
  while (high - low > 1) {
    const mid = Math.floor((low + high) / 2);
    if (values[mid] <= target) low = mid; else high = mid;
  }
  return { lower: low, upper: high, weight: (target - values[low]) / (values[high] - values[low]) };
}

function axes(field: Npy3D, grid: GridInfo | null): [number[], number[], number[]] {
  const result = ([0, 1, 2] as const).map((axis) => axisCoords(axis, field.shape[axis], grid).values);
  if (result.some((values) => !values)) throw new Error("Arbitrary physical transects require complete x/y/z scalar-center coordinates.");
  return result as [number[], number[], number[]];
}

function bilinearAt(field: Npy3D, x: Bracket, y: Bracket, zIndex: number): number {
  const [, ny, nz] = field.shape;
  const samples = [
    [x.lower, y.lower, (1 - x.weight) * (1 - y.weight)],
    [x.lower, y.upper, (1 - x.weight) * y.weight],
    [x.upper, y.lower, x.weight * (1 - y.weight)],
    [x.upper, y.upper, x.weight * y.weight],
  ] as const;
  let sum = 0;
  let weightSum = 0;
  for (const [i, j, weight] of samples) {
    if (weight <= 1e-12) continue;
    const value = field.data[(i * ny + j) * nz + zIndex];
    if (!Number.isFinite(value)) return Number.NaN;
    sum += value * weight;
    weightSum += weight;
  }
  if (weightSum === 0) {
    const value = field.data[(x.lower * ny + y.lower) * nz + zIndex];
    return Number.isFinite(value) ? value : Number.NaN;
  }
  return sum / weightSum;
}

export function sampleHorizontalPoint(
  field: Npy3D,
  grid: GridInfo | null,
  point: PhysicalPoint2D,
  zIndex: number,
): number {
  const [x, y] = axes(field, grid);
  const safeZ = Math.max(0, Math.min(field.shape[2] - 1, Math.floor(zIndex)));
  return bilinearAt(field, bracket(x, point.xKm), bracket(y, point.yKm), safeZ);
}

export function buildVerticalTransect(
  field: Npy3D,
  grid: GridInfo | null,
  start: PhysicalPoint2D,
  end: PhysicalPoint2D,
  requestedSamples?: number,
): VerticalTransect {
  const [x, y, z] = axes(field, grid);
  const lengthKm = Math.hypot(end.xKm - start.xKm, end.yKm - start.yKm);
  if (!(lengthKm > 0)) throw new Error("Transect endpoints must be distinct.");
  const nativeSpacing = Math.max(1e-9, Math.min(
    x.length > 1 ? (x.at(-1)! - x[0]) / (x.length - 1) : lengthKm,
    y.length > 1 ? (y.at(-1)! - y[0]) / (y.length - 1) : lengthKm,
  ));
  const sampleCount = Math.max(2, Math.min(512, Math.floor(requestedSamples ?? Math.ceil(lengthKm / nativeSpacing) + 1)));
  const distanceKm = Array.from({ length: sampleCount }, (_, index) => lengthKm * index / (sampleCount - 1));
  const points = distanceKm.map((distance) => {
    const fraction = distance / lengthKm;
    return { xKm: start.xKm + (end.xKm - start.xKm) * fraction, yKm: start.yKm + (end.yKm - start.yKm) * fraction };
  });
  const brackets = points.map((point) => [bracket(x, point.xKm), bracket(y, point.yKm)] as const);
  const values = z.map((_, zIndex) => brackets.map(([xb, yb]) => bilinearAt(field, xb, yb, zIndex)));
  return { distanceKm, zKm: z, values, points, sampleCount };
}
