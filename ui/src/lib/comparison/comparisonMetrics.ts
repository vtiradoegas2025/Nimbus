export interface PairMetrics {
  count: number;
  mean: number;
  mae: number;
  rmse: number;
  maxAbs: number;
}

export interface VectorPairMetrics {
  count: number;
  meanX: number;
  meanY: number;
  meanMagnitude: number;
  rmse: number;
  maxMagnitude: number;
}

export interface PairMetricMoments {
  count: number;
  weightSum: number;
  sum: number;
  absoluteSum: number;
  squareSum: number;
  maxAbs: number;
}

export interface VectorMetricMoments {
  count: number;
  weightSum: number;
  sumX: number;
  sumY: number;
  magnitudeSum: number;
  squareSum: number;
  maxMagnitude: number;
}

export interface IndexRegion3D {
  lower: readonly [number, number, number];
  upper: readonly [number, number, number];
}

export interface ChunkedMetricScanOptions {
  cancelled?: () => boolean;
  onProgress?: (progress: number) => void;
  chunkSize?: number;
  /** Exact physical cell/block volumes. Omitted for uniform-cell metrics. */
  weights?: ArrayLike<number>;
}

export interface VolumeDisplaySample {
  x: number[];
  y: number[];
  z: number[];
  left: number[];
  right: number[];
  difference: number[];
  stride: number;
  selectedCells: number;
}

export function emptyPairMetricMoments(): PairMetricMoments {
  return { count: 0, weightSum: 0, sum: 0, absoluteSum: 0, squareSum: 0, maxAbs: 0 };
}

export function mergePairMetricMoments(
  target: PairMetricMoments,
  source: PairMetricMoments,
): PairMetricMoments {
  target.count += source.count;
  target.weightSum += source.weightSum;
  target.sum += source.sum;
  target.absoluteSum += source.absoluteSum;
  target.squareSum += source.squareSum;
  target.maxAbs = Math.max(target.maxAbs, source.maxAbs);
  return target;
}

export function finishPairMetricMoments(moments: PairMetricMoments): PairMetrics {
  if (moments.count === 0 || !(moments.weightSum > 0)) throw new Error("Matched fields contain no finite cell pairs.");
  return {
    count: moments.count,
    mean: moments.sum / moments.weightSum,
    mae: moments.absoluteSum / moments.weightSum,
    rmse: Math.sqrt(moments.squareSum / moments.weightSum),
    maxAbs: moments.maxAbs,
  };
}

export function emptyVectorMetricMoments(): VectorMetricMoments {
  return { count: 0, weightSum: 0, sumX: 0, sumY: 0, magnitudeSum: 0, squareSum: 0, maxMagnitude: 0 };
}

export function mergeVectorMetricMoments(
  target: VectorMetricMoments,
  source: VectorMetricMoments,
): VectorMetricMoments {
  target.count += source.count;
  target.weightSum += source.weightSum;
  target.sumX += source.sumX;
  target.sumY += source.sumY;
  target.magnitudeSum += source.magnitudeSum;
  target.squareSum += source.squareSum;
  target.maxMagnitude = Math.max(target.maxMagnitude, source.maxMagnitude);
  return target;
}

export function finishVectorMetricMoments(moments: VectorMetricMoments): VectorPairMetrics {
  if (moments.count === 0 || !(moments.weightSum > 0)) throw new Error("Selected vector volume contains no finite component pairs.");
  return {
    count: moments.count,
    meanX: moments.sumX / moments.weightSum,
    meanY: moments.sumY / moments.weightSum,
    meanMagnitude: moments.magnitudeSum / moments.weightSum,
    rmse: Math.sqrt(moments.squareSum / moments.weightSum),
    maxMagnitude: moments.maxMagnitude,
  };
}

export async function scanPairMetricMoments(
  left: ArrayLike<number>,
  right: ArrayLike<number>,
  options: ChunkedMetricScanOptions = {},
): Promise<PairMetricMoments> {
  if (left.length !== right.length) throw new Error("Matched field tiles changed length.");
  if (options.weights && options.weights.length !== left.length) {
    throw new Error("Physical metric weights changed length.");
  }
  const result = emptyPairMetricMoments();
  const chunkSize = Math.max(10_000, Math.floor(options.chunkSize ?? 250_000));
  for (let start = 0; start < left.length; start += chunkSize) {
    if (options.cancelled?.()) throw new DOMException("Comparison metric scan cancelled", "AbortError");
    const end = Math.min(left.length, start + chunkSize);
    for (let index = start; index < end; index += 1) {
      const a = left[index]; const b = right[index];
      if (!Number.isFinite(a) || !Number.isFinite(b)) continue;
      const difference = a - b;
      const absolute = Math.abs(difference);
      const weight = options.weights?.[index] ?? 1;
      if (!Number.isFinite(weight) || weight <= 0) continue;
      result.count += 1;
      result.weightSum += weight;
      result.sum += weight * difference;
      result.absoluteSum += weight * absolute;
      result.squareSum += weight * difference * difference;
      result.maxAbs = Math.max(result.maxAbs, absolute);
    }
    if (end < left.length) await new Promise<void>((resolve) => window.setTimeout(resolve, 0));
  }
  return result;
}

export async function scanVectorMetricMoments(
  leftX: ArrayLike<number>,
  leftY: ArrayLike<number>,
  rightX: ArrayLike<number>,
  rightY: ArrayLike<number>,
  options: ChunkedMetricScanOptions = {},
): Promise<VectorMetricMoments> {
  if ([leftY, rightX, rightY].some((field) => field.length !== leftX.length)) {
    throw new Error("Matched vector tiles changed shape.");
  }
  if (options.weights && options.weights.length !== leftX.length) {
    throw new Error("Physical vector metric weights changed length.");
  }
  const result = emptyVectorMetricMoments();
  const chunkSize = Math.max(10_000, Math.floor(options.chunkSize ?? 250_000));
  for (let start = 0; start < leftX.length; start += chunkSize) {
    if (options.cancelled?.()) throw new DOMException("Vector metric scan cancelled", "AbortError");
    const end = Math.min(leftX.length, start + chunkSize);
    for (let index = start; index < end; index += 1) {
      const values = [leftX[index], leftY[index], rightX[index], rightY[index]];
      if (!values.every(Number.isFinite)) continue;
      const dx = values[0] - values[2];
      const dy = values[1] - values[3];
      const magnitude = Math.hypot(dx, dy);
      const weight = options.weights?.[index] ?? 1;
      if (!Number.isFinite(weight) || weight <= 0) continue;
      result.count += 1;
      result.weightSum += weight;
      result.sumX += weight * dx;
      result.sumY += weight * dy;
      result.magnitudeSum += weight * magnitude;
      result.squareSum += weight * magnitude * magnitude;
      result.maxMagnitude = Math.max(result.maxMagnitude, magnitude);
    }
    if (end < leftX.length) await new Promise<void>((resolve) => window.setTimeout(resolve, 0));
  }
  return result;
}

/** Validate that two terrain tiles describe the same physical cells and
 * return their exact volumes for metric weighting. Uniform tiles return null. */
export function matchedPhysicalVolumes(left: NpyN, right: NpyN): Float64Array | null {
  const leftVolumes = left.windowGeometry?.volumesM3;
  const rightVolumes = right.windowGeometry?.volumesM3;
  const leftCentroids = left.windowGeometry?.centroidsM;
  const rightCentroids = right.windowGeometry?.centroidsM;
  if (!leftVolumes && !rightVolumes && !leftCentroids && !rightCentroids) return null;
  if (!leftVolumes || !rightVolumes || !leftCentroids || !rightCentroids
      || leftVolumes.length !== rightVolumes.length) {
    throw new Error("Terrain comparison geometry is incomplete on one matched source.");
  }
  for (let index = 0; index < leftVolumes.length; index++) {
    const values = [
      leftVolumes[index], rightVolumes[index],
      leftCentroids.x[index], rightCentroids.x[index],
      leftCentroids.y[index], rightCentroids.y[index],
      leftCentroids.z[index], rightCentroids.z[index],
    ];
    if (!values.every(Number.isFinite)) throw new Error("Terrain comparison geometry contains a non-finite value.");
    for (const [a, b] of [[values[0], values[1]], [values[2], values[3]], [values[4], values[5]], [values[6], values[7]]]) {
      const tolerance = 1e-9 * Math.max(1, Math.abs(a), Math.abs(b));
      if (Math.abs(a - b) > tolerance) {
        throw new Error("Terrain comparison grids are not physically co-located cell by cell.");
      }
    }
  }
  return leftVolumes;
}

function positiveCellWidths(centers: readonly number[]): number[] {
  if (centers.length === 0 || centers.some((value) => !Number.isFinite(value))) {
    throw new Error("Volume display requires finite axis coordinates.");
  }
  if (centers.length === 1) return [1];
  for (let index = 1; index < centers.length; index += 1) {
    if (!(centers[index] > centers[index - 1])) {
      throw new Error("Volume display axis coordinates must increase strictly.");
    }
  }
  const edges = new Array<number>(centers.length + 1);
  edges[0] = centers[0] - 0.5 * (centers[1] - centers[0]);
  for (let index = 1; index < centers.length; index += 1) {
    edges[index] = 0.5 * (centers[index - 1] + centers[index]);
  }
  edges[centers.length] = centers.at(-1)! + 0.5 * (centers.at(-1)! - centers.at(-2)!);
  return centers.map((_, index) => edges[index + 1] - edges[index]);
}

/** Physical-volume block means for A/B comparison. Non-finite contributor
 *  invalidates its block. Separable Cartesian/stretched axes only. */
export function volumeMeanPairSample(
  left: ArrayLike<number>,
  right: ArrayLike<number>,
  shape: readonly [number, number, number],
  region: IndexRegion3D,
  strideInput: number,
  coordinates: readonly [readonly number[], readonly number[], readonly number[]],
): VolumeDisplaySample {
  const totalLength = shape[0] * shape[1] * shape[2];
  if (left.length !== totalLength || right.length !== totalLength) {
    throw new Error("Matched display volumes changed shape after alignment.");
  }
  for (let axis = 0; axis < 3; axis += 1) {
    if (coordinates[axis].length !== shape[axis]
      || !Number.isInteger(region.lower[axis])
      || !Number.isInteger(region.upper[axis])
      || region.lower[axis] < 0
      || region.upper[axis] < region.lower[axis]
      || region.upper[axis] >= shape[axis]) {
      throw new Error(`Volume display geometry is invalid on axis ${axis}.`);
    }
  }
  if (!Number.isFinite(strideInput) || strideInput < 1) {
    throw new Error("Volume display stride must be positive and finite.");
  }
  const stride = Math.max(1, Math.floor(strideInput));
  const widths = coordinates.map(positiveCellWidths) as [number[], number[], number[]];
  const counts = region.upper.map((upper, axis) => upper - region.lower[axis] + 1) as [number, number, number];
  const sample: VolumeDisplaySample = {
    x: [], y: [], z: [], left: [], right: [], difference: [], stride,
    selectedCells: counts[0] * counts[1] * counts[2],
  };

  for (let i0 = region.lower[0]; i0 <= region.upper[0]; i0 += stride) {
    const iEnd = Math.min(region.upper[0] + 1, i0 + stride);
    for (let j0 = region.lower[1]; j0 <= region.upper[1]; j0 += stride) {
      const jEnd = Math.min(region.upper[1] + 1, j0 + stride);
      for (let k0 = region.lower[2]; k0 <= region.upper[2]; k0 += stride) {
        const kEnd = Math.min(region.upper[2] + 1, k0 + stride);
        let weightSum = 0;
        let leftSum = 0;
        let rightSum = 0;
        let xSum = 0;
        let ySum = 0;
        let zSum = 0;
        let valid = true;
        for (let i = i0; i < iEnd && valid; i += 1) {
          for (let j = j0; j < jEnd && valid; j += 1) {
            for (let k = k0; k < kEnd; k += 1) {
              const index = (i * shape[1] + j) * shape[2] + k;
              const leftValue = left[index];
              const rightValue = right[index];
              if (!Number.isFinite(leftValue) || !Number.isFinite(rightValue)) {
                valid = false;
                break;
              }
              const weight = widths[0][i] * widths[1][j] * widths[2][k];
              weightSum += weight;
              leftSum += weight * leftValue;
              rightSum += weight * rightValue;
              xSum += weight * coordinates[0][i];
              ySum += weight * coordinates[1][j];
              zSum += weight * coordinates[2][k];
            }
          }
        }
        if (!valid || !(weightSum > 0) || !Number.isFinite(weightSum)) continue;
        const leftMean = leftSum / weightSum;
        const rightMean = rightSum / weightSum;
        sample.x.push(xSum / weightSum);
        sample.y.push(ySum / weightSum);
        sample.z.push(zSum / weightSum);
        sample.left.push(leftMean);
        sample.right.push(rightMean);
        sample.difference.push(leftMean - rightMean);
      }
    }
  }
  return sample;
}

/** Exact finite-pair A−B metrics with bounded main-thread work per turn. */
export async function scanPairMetrics(
  left: ArrayLike<number>,
  right: ArrayLike<number>,
  options: ChunkedMetricScanOptions = {},
): Promise<PairMetrics> {
  if (left.length !== right.length) {
    throw new Error(`Matched fields changed length: ${left.length} versus ${right.length}.`);
  }
  const chunkSize = Math.max(10_000, Math.floor(options.chunkSize ?? 250_000));
  let count = 0;
  let sum = 0;
  let absoluteSum = 0;
  let squareSum = 0;
  let maxAbs = 0;
  for (let start = 0; start < left.length; start += chunkSize) {
    if (options.cancelled?.()) throw new DOMException("Comparison metric scan cancelled", "AbortError");
    const end = Math.min(left.length, start + chunkSize);
    for (let index = start; index < end; index += 1) {
      const leftValue = left[index];
      const rightValue = right[index];
      if (!Number.isFinite(leftValue) || !Number.isFinite(rightValue)) continue;
      const difference = leftValue - rightValue;
      const absolute = Math.abs(difference);
      count += 1;
      sum += difference;
      absoluteSum += absolute;
      squareSum += difference * difference;
      maxAbs = Math.max(maxAbs, absolute);
    }
    options.onProgress?.(left.length === 0 ? 1 : end / left.length);
    if (end < left.length) await new Promise<void>((resolve) => window.setTimeout(resolve, 0));
  }
  if (count === 0) throw new Error("Matched fields contain no finite cell pairs.");
  return {
    count,
    mean: sum / count,
    mae: absoluteSum / count,
    rmse: Math.sqrt(squareSum / count),
    maxAbs,
  };
}

/** Exact two-component vector A−B metrics over an inclusive 3-D index box. */
export async function scanVectorPairMetrics(
  leftX: ArrayLike<number>,
  leftY: ArrayLike<number>,
  rightX: ArrayLike<number>,
  rightY: ArrayLike<number>,
  shape: readonly [number, number, number],
  region: IndexRegion3D,
  options: ChunkedMetricScanOptions = {},
): Promise<VectorPairMetrics> {
  const totalLength = shape[0] * shape[1] * shape[2];
  if ([leftX, leftY, rightX, rightY].some((field) => field.length !== totalLength)) {
    throw new Error("Matched vector components changed shape after alignment.");
  }
  for (let axis = 0; axis < 3; axis += 1) {
    if (
      !Number.isInteger(region.lower[axis])
      || !Number.isInteger(region.upper[axis])
      || region.lower[axis] < 0
      || region.upper[axis] < region.lower[axis]
      || region.upper[axis] >= shape[axis]
    ) throw new Error(`Vector metric region is invalid on axis ${axis}.`);
  }
  const selectedTotal = (region.upper[0] - region.lower[0] + 1)
    * (region.upper[1] - region.lower[1] + 1)
    * (region.upper[2] - region.lower[2] + 1);
  const chunkSize = Math.max(10_000, Math.floor(options.chunkSize ?? 250_000));
  let processed = 0;
  let count = 0;
  let sumX = 0;
  let sumY = 0;
  let magnitudeSum = 0;
  let squareSum = 0;
  let maxMagnitude = 0;
  for (let i = region.lower[0]; i <= region.upper[0]; i += 1) {
    for (let j = region.lower[1]; j <= region.upper[1]; j += 1) {
      for (let k = region.lower[2]; k <= region.upper[2]; k += 1) {
        if (options.cancelled?.()) throw new DOMException("Vector metric scan cancelled", "AbortError");
        const index = (i * shape[1] + j) * shape[2] + k;
        const values = [leftX[index], leftY[index], rightX[index], rightY[index]];
        if (values.every(Number.isFinite)) {
          const dx = values[0] - values[2];
          const dy = values[1] - values[3];
          const magnitude = Math.hypot(dx, dy);
          count += 1;
          sumX += dx;
          sumY += dy;
          magnitudeSum += magnitude;
          squareSum += magnitude * magnitude;
          maxMagnitude = Math.max(maxMagnitude, magnitude);
        }
        processed += 1;
        if (processed % chunkSize === 0) {
          options.onProgress?.(processed / selectedTotal);
          await new Promise<void>((resolve) => window.setTimeout(resolve, 0));
        }
      }
    }
  }
  options.onProgress?.(1);
  if (count === 0) throw new Error("Selected vector volume contains no finite component pairs.");
  return {
    count,
    meanX: sumX / count,
    meanY: sumY / count,
    meanMagnitude: magnitudeSum / count,
    rmse: Math.sqrt(squareSum / count),
    maxMagnitude,
  };
}

/** Exact scalar A−B metrics over an inclusive 3-D index box without copying it. */
export async function scanScalarRegionMetrics(
  left: ArrayLike<number>,
  right: ArrayLike<number>,
  shape: readonly [number, number, number],
  region: IndexRegion3D,
  options: ChunkedMetricScanOptions = {},
): Promise<PairMetrics> {
  const totalLength = shape[0] * shape[1] * shape[2];
  if (left.length !== totalLength || right.length !== totalLength) throw new Error("Matched scalar volumes changed shape after alignment.");
  for (let axis = 0; axis < 3; axis += 1) {
    if (!Number.isInteger(region.lower[axis]) || !Number.isInteger(region.upper[axis]) || region.lower[axis] < 0 || region.upper[axis] < region.lower[axis] || region.upper[axis] >= shape[axis]) {
      throw new Error(`Scalar metric region is invalid on axis ${axis}.`);
    }
  }
  const selectedTotal = (region.upper[0] - region.lower[0] + 1) * (region.upper[1] - region.lower[1] + 1) * (region.upper[2] - region.lower[2] + 1);
  const chunkSize = Math.max(10_000, Math.floor(options.chunkSize ?? 250_000));
  let processed = 0;
  let count = 0;
  let sum = 0;
  let absoluteSum = 0;
  let squareSum = 0;
  let maxAbs = 0;
  for (let i = region.lower[0]; i <= region.upper[0]; i += 1) {
    for (let j = region.lower[1]; j <= region.upper[1]; j += 1) {
      for (let k = region.lower[2]; k <= region.upper[2]; k += 1) {
        if (options.cancelled?.()) throw new DOMException("Scalar region scan cancelled", "AbortError");
        const index = (i * shape[1] + j) * shape[2] + k;
        const leftValue = left[index];
        const rightValue = right[index];
        if (Number.isFinite(leftValue) && Number.isFinite(rightValue)) {
          const difference = leftValue - rightValue;
          const absolute = Math.abs(difference);
          count += 1;
          sum += difference;
          absoluteSum += absolute;
          squareSum += difference * difference;
          maxAbs = Math.max(maxAbs, absolute);
        }
        processed += 1;
        if (processed % chunkSize === 0) {
          options.onProgress?.(processed / selectedTotal);
          await new Promise<void>((resolve) => window.setTimeout(resolve, 0));
        }
      }
    }
  }
  options.onProgress?.(1);
  if (count === 0) throw new Error("Selected scalar volume contains no finite pairs.");
  return { count, mean: sum / count, mae: absoluteSum / count, rmse: Math.sqrt(squareSum / count), maxAbs };
}

export function formatComparisonMetric(value: number): string {
  if (!Number.isFinite(value)) return "—";
  if (value === 0) return "0";
  const magnitude = Math.abs(value);
  return magnitude >= 10_000 || magnitude < 0.001
    ? value.toExponential(3)
    : value.toPrecision(4);
}
import type { NpyN } from "@/lib/platform/npy";
