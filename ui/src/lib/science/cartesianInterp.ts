import type { Npy3D } from "@/lib/platform/npy";
import { gridForWindowedField } from "./windowedGrid";
import { cartesianDyMeters, cartesianScalarAxisOrigin } from "./cartesianCoordinates";
export { cartesianDyMeters, cartesianScalarAxisOrigin } from "./cartesianCoordinates";

/*
 * Cartesian (x, y, z) field → VTK voxel grid wrapper.
 * Aliases regular fields; resamples stretched axes; sanitizes non-finite cells; area-mean LOD on x/y.
 */

export interface CartesianVoxelGrid {
  /** (nx, ny, nz) shape of the Cartesian output. */
  shape: [number, number, number];
  /** Voxel spacing along each axis, in meters. */
  spacing: [number, number, number];
  /** World-space coordinate of scalar cell (0,0,0); VTK point-origin convention. */
  origin: [number, number, number];
  /** C-order: `data[ix*ny*nz + iy*nz + iz]`; preserveMissing retains NaN support until VTK packing. */
  data: Float32Array;
  /** Count observed before VTK-safe finite substitution. Isosurfaces use this
   * to fail closed because any fill value can create a false mask boundary. */
  sourceNonFiniteCount?: number;
  /** Present only when the renderer received a reduced visualization lattice. */
  lod?: {
    nativeShape: [number, number, number];
    stride: number;
    aggregation: "area_mean_xy";
  };
}

export interface TerrainCartesianField {
  /** Physical-z conservative field; non-finite source coverage remains NaN. */
  field: Npy3D;
  /** Separable physical cell centres in metres after the terrain remap. */
  scalarCoordinatesM: [number[], number[], number[]];
  spacing: [number, number, number];
  origin: [number, number, number];
}

function terrainBlockWidth(
  field: Npy3D,
  grid: GridSpacing,
  axis: 0 | 1,
  outputIndex: number,
): number {
  const geometry = field.windowGeometry!;
  const sourceExtent = geometry.extent?.[axis] ?? field.shape[axis];
  const stride = geometry.stride?.[axis] ?? 1;
  const sourceCells = Math.min(stride, sourceExtent - outputIndex * stride);
  const nominal = axis === 0
    ? grid.drM ?? 1
    : cartesianDyMeters(grid, grid.drM ?? 1);
  return Math.max(nominal, sourceCells * nominal);
}

/**
 * Conservatively remap a bounded terrain-following field onto a regular
 * physical Cartesian lattice for vtkImageData. The native reader supplies an
 * exact physical-volume mean, centroid and volume for every source block. We
 * retain those horizontal blocks and redistribute each column by vertical
 * overlap, so integrated piecewise-constant content is conserved instead of
 * flattening model level k onto a nominal z.
 */
export function terrainWindowToCartesianField(
  field: Npy3D,
  grid: GridSpacing,
  vertical?: { origin: number; spacing: number; count: number },
): TerrainCartesianField {
  const geometry = field.windowGeometry;
  const centroids = geometry?.centroidsM;
  const volumes = geometry?.volumesM3;
  if (!grid.terrainFollowing || geometry?.weighting !== "physical_cell_volume"
      || !centroids || !volumes || volumes.length !== field.data.length
      || centroids.x.length !== field.data.length
      || centroids.y.length !== field.data.length
      || centroids.z.length !== field.data.length) {
    throw new Error(
      "Terrain-following 3D rendering requires bounded physical centroids and exact cell volumes",
    );
  }
  const [nx, ny, nz] = field.shape;
  const xCoordinates = new Float64Array(nx);
  const yCoordinates = new Float64Array(ny);
  const xWeights = new Float64Array(nx);
  const yWeights = new Float64Array(ny);
  let lowerZ = Infinity;
  let upperZ = -Infinity;
  for (let i = 0; i < nx; i++) {
    const widthX = terrainBlockWidth(field, grid, 0, i);
    for (let j = 0; j < ny; j++) {
      const area = widthX * terrainBlockWidth(field, grid, 1, j);
      for (let k = 0; k < nz; k++) {
        const index = (i * ny + j) * nz + k;
        const volume = volumes[index];
        const x = centroids.x[index];
        const y = centroids.y[index];
        const z = centroids.z[index];
        if (![volume, x, y, z].every(Number.isFinite) || volume <= 0 || area <= 0) {
          throw new Error(`Terrain geometry is invalid at bounded cell (${i},${j},${k})`);
        }
        xCoordinates[i] += volume * x;
        xWeights[i] += volume;
        yCoordinates[j] += volume * y;
        yWeights[j] += volume;
        const thickness = volume / area;
        lowerZ = Math.min(lowerZ, z - 0.5 * thickness);
        upperZ = Math.max(upperZ, z + 0.5 * thickness);
      }
    }
  }
  for (let i = 0; i < nx; i++) xCoordinates[i] /= xWeights[i];
  for (let j = 0; j < ny; j++) yCoordinates[j] /= yWeights[j];
  if (!(upperZ > lowerZ) || !Number.isFinite(lowerZ) || !Number.isFinite(upperZ)) {
    throw new Error("Terrain geometry has no finite physical vertical extent");
  }
  const fullColumn = geometry.origin?.[2] === (grid.validIndexRange?.lower[2] ?? 0)
    && geometry.extent?.[2] === (grid.validIndexRange
      ? grid.validIndexRange.upper[2] - grid.validIndexRange.lower[2] + 1
      : geometry.sourceShape?.[2]);
  if (!vertical && fullColumn && grid.physicalBoundsM) {
    const lower = grid.physicalBoundsM.lower[2], upper = grid.physicalBoundsM.upper[2];
    if (Number.isFinite(lower) && upper > lower) vertical = { origin: lower + (upper - lower) / (2 * nz), spacing: (upper - lower) / nz, count: nz };
  }
  const targetNz = vertical?.count ?? nz;
  const dz = vertical?.spacing ?? (upperZ - lowerZ) / Math.max(1, nz);
  if (vertical) lowerZ = vertical.origin - dz / 2;
  if (!(dz > 0) || !Number.isSafeInteger(targetNz) || targetNz < 1) throw new Error("Invalid terrain target lattice");
  const horizontalOrigin = grid.horizontalScalarOriginM ?? grid.scalarOriginM;
  if (horizontalOrigin && geometry.origin && geometry.stride && geometry.extent) {
    for (const axis of [0, 1]) {
      const coordinates = axis === 0 ? xCoordinates : yCoordinates;
      const spacing = axis === 0 ? grid.drM : cartesianDyMeters(grid, grid.drM ?? 1);
      if (!(spacing && spacing > 0)) continue;
      for (let i = 0; i < coordinates.length; i++) {
        const first = geometry.origin[axis] + i * geometry.stride[axis];
        const width = Math.min(geometry.stride[axis], geometry.extent[axis] - i * geometry.stride[axis]);
        coordinates[i] = horizontalOrigin[axis] + (first + (width - 1) / 2) * spacing;
      }
    }
  }
  const remapped = new Float32Array(nx * ny * targetNz);
  for (let i = 0; i < nx; i++) {
    const widthX = terrainBlockWidth(field, grid, 0, i);
    for (let j = 0; j < ny; j++) {
      const area = widthX * terrainBlockWidth(field, grid, 1, j);
      const columnBase = (i * ny + j) * nz;
      let firstSource = 0;
      const sourceLowerBounds = new Float64Array(nz);
      const sourceUpperBounds = new Float64Array(nz);
      let ordered = true;
      for (let k = 0; k < nz; k++) {
        const half = volumes[columnBase + k] / (2 * area);
        sourceLowerBounds[k] = centroids.z[columnBase + k] - half;
        sourceUpperBounds[k] = centroids.z[columnBase + k] + half;
        if (k > 0 && (sourceLowerBounds[k] < sourceLowerBounds[k - 1] || sourceUpperBounds[k] < sourceUpperBounds[k - 1])) ordered = false;
      }
      for (let targetK = 0; targetK < targetNz; targetK++) {
        const targetLower = lowerZ + targetK * dz;
        const targetUpper = targetLower + dz;
        let sum = 0;
        let overlapSum = 0;
        let invalid = false;
        while (ordered && firstSource < nz && sourceUpperBounds[firstSource] <= targetLower) firstSource++;
        for (let sourceK = firstSource; sourceK < nz; sourceK++) {
          const sourceIndex = columnBase + sourceK;
          if (ordered && sourceLowerBounds[sourceK] >= targetUpper) break;
          const sourceLower = sourceLowerBounds[sourceK];
          const sourceUpper = sourceUpperBounds[sourceK];
          const overlap = Math.max(0, Math.min(targetUpper, sourceUpper) - Math.max(targetLower, sourceLower));
          if (overlap <= 0) continue;
          const value = field.data[sourceIndex];
          if (!Number.isFinite(value)) invalid = true;
          else sum += overlap * value;
          overlapSum += overlap;
        }
        remapped[(i * ny + j) * targetNz + targetK] = invalid || overlapSum <= 0
          ? Number.NaN
          : sum / overlapSum;
      }
    }
  }
  const dx = nx > 1 ? (xCoordinates[nx - 1] - xCoordinates[0]) / (nx - 1) : grid.drM ?? 1;
  const dy = ny > 1 ? (yCoordinates[ny - 1] - yCoordinates[0]) / (ny - 1) : cartesianDyMeters(grid, grid.drM ?? 1);
  const origin: [number, number, number] = [
    xCoordinates[0],
    yCoordinates[0],
    lowerZ + 0.5 * dz,
  ];
  return {
    field: { shape: [nx, ny, targetNz], data: remapped },
    scalarCoordinatesM: [
      Array.from(xCoordinates),
      Array.from(yCoordinates),
      Array.from({ length: targetNz }, (_, k) => origin[2] + k * dz),
    ],
    spacing: [dx, dy, dz],
    origin,
  };
}

export function terrainWindowToCartesianGrid(
  field: Npy3D,
  grid: GridSpacing,
  preserveMissing = false,
): CartesianVoxelGrid {
  const remapped = terrainWindowToCartesianField(field, grid);
  return cartesianBoxToImageData(
    remapped.field,
    remapped.spacing[0],
    remapped.spacing[1],
    remapped.spacing[2],
    {
      originM: remapped.origin,
      preserveMissing,
      sourceXCoordinatesM: remapped.scalarCoordinatesM[0],
      sourceYCoordinatesM: remapped.scalarCoordinatesM[1],
    },
  );
}

export interface CartesianGridRefinement {
  grid: CartesianVoxelGrid;
  /** Integer subdivisions applied to each original voxel interval. */
  factors: [number, number, number];
}

/**
 * Piecewise-trilinear refinement onto a denser display lattice (visualization only; no new extrema).
 * Per-axis factors are reduced to satisfy `maxVoxels`; already-fine grids alias without copy.
 */
export function refineCartesianGridLinear(
  source: CartesianVoxelGrid,
  options: {
    targetSpacingM?: number;
    maxFactor?: number;
    maxVoxels?: number;
  } = {},
): CartesianGridRefinement {
  const targetSpacing = Math.max(1, options.targetSpacingM ?? 250);
  const maxFactor = Math.max(1, Math.floor(options.maxFactor ?? 4));
  const maxVoxels = Math.max(8, Math.floor(options.maxVoxels ?? Number.MAX_SAFE_INTEGER));
  const factors = source.spacing.map((spacing) =>
    Math.max(1, Math.min(maxFactor, Math.ceil(spacing / targetSpacing)))
  ) as [number, number, number];
  const refinedShape = () => source.shape.map((count, axis) =>
    (count - 1) * factors[axis] + 1
  ) as [number, number, number];
  const voxelCount = (shape: [number, number, number]) => shape[0] * shape[1] * shape[2];

  // If the source itself is already over the caller's ceiling, refinement
  // cannot help. (Normal preparation downsamples before this point.)
  if (voxelCount(source.shape) > maxVoxels) {
    return { grid: source, factors: [1, 1, 1] };
  }

  let shape = refinedShape();
  while (voxelCount(shape) > maxVoxels) {
    let axisToReduce = -1;
    let finestEffectiveSpacing = Infinity;
    for (let axis = 0; axis < 3; axis++) {
      if (factors[axis] <= 1) continue;
      const effectiveSpacing = source.spacing[axis] / factors[axis];
      // Remove subdivisions from the finest axis: isotropic degradation.
      if (effectiveSpacing < finestEffectiveSpacing) {
        finestEffectiveSpacing = effectiveSpacing;
        axisToReduce = axis;
      }
    }
    if (axisToReduce < 0) break;
    factors[axisToReduce] -= 1;
    shape = refinedShape();
  }

  if (factors.every((factor) => factor === 1)) {
    return { grid: source, factors };
  }

  const [sourceNx, sourceNy, sourceNz] = source.shape;
  const [nx, ny, nz] = shape;
  const [fx, fy, fz] = factors;
  const sourceSx = sourceNy * sourceNz;
  const sourceSy = sourceNz;
  const out = new Float32Array(nx * ny * nz);

  const xMap = interpolationAxis(nx, fx, sourceNx);
  const yMap = interpolationAxis(ny, fy, sourceNy);
  const zMap = interpolationAxis(nz, fz, sourceNz);
  let destination = 0;
  for (let x = 0; x < nx; x++) {
    const x0 = xMap.lower[x];
    const x1 = xMap.upper[x];
    const tx = xMap.weight[x];
    for (let y = 0; y < ny; y++) {
      const y0 = yMap.lower[y];
      const y1 = yMap.upper[y];
      const ty = yMap.weight[y];
      const base00 = x0 * sourceSx + y0 * sourceSy;
      const base01 = x0 * sourceSx + y1 * sourceSy;
      const base10 = x1 * sourceSx + y0 * sourceSy;
      const base11 = x1 * sourceSx + y1 * sourceSy;
      for (let z = 0; z < nz; z++) {
        const z0 = zMap.lower[z];
        const z1 = zMap.upper[z];
        const tz = zMap.weight[z];
        const c00 = lerp(source.data[base00 + z0], source.data[base00 + z1], tz);
        const c01 = lerp(source.data[base01 + z0], source.data[base01 + z1], tz);
        const c10 = lerp(source.data[base10 + z0], source.data[base10 + z1], tz);
        const c11 = lerp(source.data[base11 + z0], source.data[base11 + z1], tz);
        const c0 = lerp(c00, c01, ty);
        const c1 = lerp(c10, c11, ty);
        out[destination++] = lerp(c0, c1, tx);
      }
    }
  }

  return {
    grid: {
      shape,
      spacing: [source.spacing[0] / fx, source.spacing[1] / fy, source.spacing[2] / fz],
      origin: source.origin,
      data: out,
      sourceNonFiniteCount: source.sourceNonFiniteCount,
    },
    factors,
  };
}

function interpolationAxis(
  outputCount: number,
  factor: number,
  sourceCount: number,
): { lower: Int32Array; upper: Int32Array; weight: Float32Array } {
  const lower = new Int32Array(outputCount);
  const upper = new Int32Array(outputCount);
  const weight = new Float32Array(outputCount);
  for (let output = 0; output < outputCount; output++) {
    const coordinate = output / factor;
    const lo = Math.min(sourceCount - 1, Math.floor(coordinate));
    lower[output] = lo;
    upper[output] = Math.min(sourceCount - 1, lo + 1);
    weight[output] = coordinate - lo;
  }
  return { lower, upper, weight };
}

function lerp(a: number, b: number, weight: number): number {
  if (weight === 0) return a;
  if (weight === 1) return b;
  return a + (b - a) * weight;
}

interface AxisAggregationWeight {
  index: number;
  weight: number;
}

/** Conservative 1D area-overlap weights; endpoint-preserving; handles stretched source coordinates. */
function areaAggregationAxis(
  sourceCount: number,
  outputCount: number,
  sourceCoordinates?: readonly number[],
): AxisAggregationWeight[][] {
  const source = sourceCoordinates && sourceCoordinates.length === sourceCount
    ? sourceCoordinates
    : Array.from({ length: sourceCount }, (_, index) => index);
  const sourceEdges = new Float64Array(sourceCount + 1);
  sourceEdges[0] = source[0] - 0.5 * (source[1] - source[0]);
  for (let index = 1; index < sourceCount; index++) {
    sourceEdges[index] = 0.5 * (source[index - 1] + source[index]);
  }
  sourceEdges[sourceCount] = source[sourceCount - 1]
    + 0.5 * (source[sourceCount - 1] - source[sourceCount - 2]);

  const targetCenters = new Float64Array(outputCount);
  for (let output = 0; output < outputCount; output++) {
    targetCenters[output] = outputCount > 1
      ? source[0] + output * (source[sourceCount - 1] - source[0]) / (outputCount - 1)
      : 0.5 * (sourceEdges[0] + sourceEdges[sourceCount]);
  }
  const targetEdges = new Float64Array(outputCount + 1);
  targetEdges[0] = sourceEdges[0];
  for (let output = 1; output < outputCount; output++) {
    targetEdges[output] = 0.5 * (targetCenters[output - 1] + targetCenters[output]);
  }
  targetEdges[outputCount] = sourceEdges[sourceCount];

  const result: AxisAggregationWeight[][] = [];
  let firstSource = 0;
  for (let output = 0; output < outputCount; output++) {
    const lower = targetEdges[output];
    const upper = targetEdges[output + 1];
    while (firstSource + 1 < sourceCount && sourceEdges[firstSource + 1] <= lower) {
      firstSource += 1;
    }
    const weights: AxisAggregationWeight[] = [];
    let total = 0;
    for (let sourceIndex = firstSource;
      sourceIndex < sourceCount && sourceEdges[sourceIndex] < upper;
      sourceIndex++) {
      const overlap = Math.max(
        0,
        Math.min(upper, sourceEdges[sourceIndex + 1]) - Math.max(lower, sourceEdges[sourceIndex]),
      );
      if (overlap > 0) {
        weights.push({ index: sourceIndex, weight: overlap });
        total += overlap;
      }
    }
    for (const weight of weights) weight.weight /= total;
    result.push(weights);
  }
  return result;
}

function linearFiniteSample(a: number, b: number, weight: number): number {
  const aFinite = Number.isFinite(a);
  const bFinite = Number.isFinite(b);
  if (aFinite && bFinite) return lerp(a, b, weight);
  if (aFinite) return a;
  if (bFinite) return b;
  return Number.NaN;
}

/**
 * Wrap a Cartesian field as a VTK voxel grid. Aliases regular layouts without copy; resamples
 * stretched axes; applies area-mean x/y LOD for stride > 1; replaces non-finite with finite min.
 */
export function cartesianBoxToImageData(
  field: Npy3D,
  dxM: number,
  dyM: number,
  dzM: number,
  opts: {
    stride?: number;
    originM?: [number, number, number];
    sourceXCoordinatesM?: readonly number[];
    sourceYCoordinatesM?: readonly number[];
    sourceZCoordinatesM?: readonly number[];
    preserveMissing?: boolean;
  } = {},
): CartesianVoxelGrid {
  const [nxFull, nyFull, nzFull] = field.shape;
  const stride = Math.max(1, Math.floor(opts.stride ?? 1));
  const nz = nzFull;

  const { min, nonFiniteCount } = scanFinite(field.data);
  const hasNonFinite = nonFiniteCount > 0;
  const fill = opts.preserveMissing ? Number.NaN : Number.isFinite(min) ? min : 0;
  let nx = nxFull;
  let ny = nyFull;
  let dx = dxM;
  let dy = dyM;
  let data: Float32Array;

  nx = stride === 1 ? nxFull : Math.min(nxFull, Math.max(2, Math.ceil(nxFull / stride)));
  ny = stride === 1 ? nyFull : Math.min(nyFull, Math.max(2, Math.ceil(nyFull / stride)));
  const xMap = opts.sourceXCoordinatesM
    ? coordinateInterpolationAxis(opts.sourceXCoordinatesM, nx)
    : null;
  const yMap = opts.sourceYCoordinatesM
    ? coordinateInterpolationAxis(opts.sourceYCoordinatesM, ny)
    : null;
  const zMap = opts.sourceZCoordinatesM
    ? coordinateInterpolationAxis(opts.sourceZCoordinatesM, nz)
    : null;
  const aggregateXY = stride > 1 && (nx < nxFull || ny < nyFull);
  const xAggregation = aggregateXY
    ? areaAggregationAxis(nxFull, nx, opts.sourceXCoordinatesM)
    : null;
  const yAggregation = aggregateXY
    ? areaAggregationAxis(nyFull, ny, opts.sourceYCoordinatesM)
    : null;

  if (stride === 1 && !xMap && !yMap && !zMap) {
    if (!hasNonFinite) {
      // Clean + already in the (ix*ny*nz + iy*nz + iz) layout -- alias, no copy.
      data = field.data;
    } else {
      // Same layout, but scrub non-finite cells so VTK never ingests a NaN.
      data = new Float32Array(field.data.length);
      for (let i = 0; i < data.length; i++) {
        const v = field.data[i];
        data[i] = Number.isFinite(v) ? v : fill;
      }
    }
  } else {
    // Preserve full physical extent across strides so layers at different voxel caps register.
    // dx = fullX/(nx-1) spans the same bounds regardless of stride.
    const fullX = (nxFull - 1) * dxM;
    const fullY = (nyFull - 1) * dyM;
    dx = nx > 1 ? fullX / (nx - 1) : dxM;
    dy = ny > 1 ? fullY / (ny - 1) : dyM;
    data = new Float32Array(nx * ny * nz);
    const srcSx = nyFull * nzFull;
    const srcSy = nzFull;
    for (let ix = 0; ix < nx; ix++) {
      const xWeights = xAggregation?.[ix];
      const six = nx > 1 ? Math.min(nxFull - 1, Math.round((ix * (nxFull - 1)) / (nx - 1))) : 0;
      const x0 = xMap?.lower[ix] ?? six;
      const x1 = xMap?.upper[ix] ?? six;
      const tx = xMap?.weight[ix] ?? 0;
      for (let iy = 0; iy < ny; iy++) {
        const yWeights = yAggregation?.[iy];
        const siy = ny > 1 ? Math.min(nyFull - 1, Math.round((iy * (nyFull - 1)) / (ny - 1))) : 0;
        const y0 = yMap?.lower[iy] ?? siy;
        const y1 = yMap?.upper[iy] ?? siy;
        const ty = yMap?.weight[iy] ?? 0;
        const dstBase = (ix * ny + iy) * nz;
        for (let iz = 0; iz < nz; iz++) {
          const z0 = zMap?.lower[iz] ?? iz;
          const z1 = zMap?.upper[iz] ?? iz;
          const tz = zMap?.weight[iz] ?? 0;
          if (xWeights && yWeights) {
            let sum = 0;
            let finiteWeight = 0;
            let missing = false;
            for (const xWeight of xWeights) {
              for (const yWeight of yWeights) {
                const sourceBase = xWeight.index * srcSx + yWeight.index * srcSy;
                const value = linearFiniteSample(
                  field.data[sourceBase + z0],
                  field.data[sourceBase + z1],
                  tz,
                );
                if (Number.isFinite(value)) {
                  const weight = xWeight.weight * yWeight.weight;
                  sum += weight * value;
                  finiteWeight += weight;
                } else if (xWeight.weight * yWeight.weight > 0) missing = true;
              }
            }
            data[dstBase + iz] = finiteWeight > 0 && !(opts.preserveMissing && missing) ? sum / finiteWeight : fill;
          } else {
            data[dstBase + iz] = trilinearFinite(
              field.data,
              srcSx,
              srcSy,
              x0,
              x1,
              y0,
              y1,
              z0,
              z1,
              tx,
              ty,
              tz,
              fill,
            );
          }
        }
      }
    }
  }

  return {
    shape: [nx, ny, nz],
    spacing: [dx, dy, dzM],
    // New manifests provide the physical scalar origin; centered fallback for legacy runs.
    origin: opts.originM ?? [-(nx - 1) * dx * 0.5, -(ny - 1) * dy * 0.5, 0.5 * dzM],
    data,
    sourceNonFiniteCount: nonFiniteCount,
    ...(aggregateXY ? {
      lod: {
        nativeShape: [nxFull, nyFull, nzFull] as [number, number, number],
        stride,
        aggregation: "area_mean_xy" as const,
      },
    } : {}),
  };
}

/** Map a strictly-increasing stretched coordinate to a uniform display lattice (same endpoints). */
function coordinateInterpolationAxis(
  source: readonly number[],
  outputCount = source.length,
): { lower: Int32Array; upper: Int32Array; weight: Float32Array } | null {
  if (source.length < 2 || source.some((value, index) =>
    !Number.isFinite(value) || index > 0 && !(value > source[index - 1])
  )) return null;
  const count = source.length;
  if (outputCount < 2) return null;
  const spacing = (source[count - 1] - source[0]) / (outputCount - 1);
  if (outputCount === count && source.every((value, index) => Math.abs(value - source[0] - index * spacing) <= Math.max(1, Math.abs(value), Math.abs(source[0])) * Number.EPSILON * 8)) return null;
  const lower = new Int32Array(outputCount);
  const upper = new Int32Array(outputCount);
  const weight = new Float32Array(outputCount);
  let bracket = 0;
  for (let output = 0; output < outputCount; output++) {
    const target = output === outputCount - 1 ? source[count - 1] : source[0] + output * spacing;
    while (bracket + 1 < count - 1 && source[bracket + 1] < target) bracket += 1;
    lower[output] = bracket;
    upper[output] = Math.min(count - 1, bracket + 1);
    const span = source[upper[output]] - source[bracket];
    weight[output] = span > 0 ? (target - source[bracket]) / span : 0;
  }
  return { lower, upper, weight };
}

function trilinearFinite(
  source: Float32Array,
  sx: number,
  sy: number,
  x0: number,
  x1: number,
  y0: number,
  y1: number,
  z0: number,
  z1: number,
  tx: number,
  ty: number,
  tz: number,
  fill: number,
): number {
  const base00 = x0 * sx + y0 * sy;
  const base01 = x0 * sx + y1 * sy;
  const base10 = x1 * sx + y0 * sy;
  const base11 = x1 * sx + y1 * sy;
  const c00 = interpolateFinite(source[base00 + z0], source[base00 + z1], tz, fill);
  const c01 = interpolateFinite(source[base01 + z0], source[base01 + z1], tz, fill);
  const c10 = interpolateFinite(source[base10 + z0], source[base10 + z1], tz, fill);
  const c11 = interpolateFinite(source[base11 + z0], source[base11 + z1], tz, fill);
  const c0 = interpolateFinite(c00, c01, ty, fill);
  const c1 = interpolateFinite(c10, c11, ty, fill);
  return interpolateFinite(c0, c1, tx, fill);
}

function interpolateFinite(
  lower: number,
  upper: number,
  weight: number,
  fill: number,
): number {
  if (weight <= 1e-7) return Number.isFinite(lower) ? lower : fill;
  if (weight >= 1 - 1e-7) return Number.isFinite(upper) ? upper : fill;
  return Number.isFinite(lower) && Number.isFinite(upper)
    ? lerp(lower, upper, weight)
    : fill;
}

/** Finite minimum and non-finite presence of `src` in one pass. */
function scanFinite(src: Float32Array): { min: number; nonFiniteCount: number } {
  let min = Infinity;
  let nonFiniteCount = 0;
  for (let i = 0; i < src.length; i++) {
    const v = src[i];
    if (Number.isFinite(v)) {
      if (v < min) min = v;
    } else {
      nonFiniteCount += 1;
    }
  }
  return { min, nonFiniteCount };
}

/** True for Cartesian or missing coordinate-system metadata (Nimbus is Cartesian-only). */
export function isCartesianGrid(
  coordinateSystem: string | number | null | undefined,
): boolean {
  return typeof coordinateSystem !== "string" || coordinateSystem.toLowerCase() === "cartesian";
}

/** Relevant subset of runOutputs.GridInfo; redeclared to avoid UI-layer dependency. */
export interface GridSpacing {
  coordinateSystem?: string | null;
  geometryKind?: "physical" | "index";
  terrainFollowing?: boolean;
  horizontalScalarOriginM?: [number, number];
  drM?: number | null;
  dyM?: number | null;
  dthetaRad?: number | null;
  dzM?: number | null;
  scalarCoordinatesM?: [number[], number[], number[]];
  scalarOriginM?: [number, number, number] | null;
  physicalBoundsM?: {
    lower: [number, number, number];
    upper: [number, number, number];
  } | null;
  validIndexRange?: {
    lower: [number, number, number];
    upper: [number, number, number];
  } | null;
}

function finiteTriplet(value: unknown): value is [number, number, number] {
  return Array.isArray(value) && value.length === 3 && value.every(Number.isFinite);
}

/** World coordinate of scalar cell (0,0,0). Explicit in v2 manifests; bounds or centered fallback. */
export function cartesianScalarOrigin(
  grid: GridSpacing,
  shape: [number, number, number],
): [number, number, number] {
  const dx = grid.drM ?? 1;
  const dy = cartesianDyMeters(grid, dx);
  const dz = grid.dzM ?? 1;
  return [
    cartesianScalarAxisOrigin(grid, 0, shape[0], dx),
    cartesianScalarAxisOrigin(grid, 1, shape[1], dy),
    cartesianScalarAxisOrigin(grid, 2, shape[2], dz),
  ];
}

/** Index/world transform used by plot-coordinate round-trip checks. */
export function cartesianScalarCoordinate(origin: number, spacing: number, index: number): number {
  return origin + index * spacing;
}

export function cartesianScalarFractionalIndex(
  origin: number,
  spacing: number,
  coordinate: number,
): number {
  return (coordinate - origin) / spacing;
}


/*
 * Default worker ceilings, in cells kept after reduction.
 *
 * 16M holds a 512x512x160 run to stride 2 — one column in four — at roughly 42 MB per scalar
 * field. Below it the stride grows fast: the supercell composite sat at 2M, which strided that
 * run to one column in 25 and to one in 100 inside a 2x2 grid, softening the structure the
 * scene exists to show. The legacy in-memory path block-averages x/y. Native and built-in
 * fields use bounded read-time x/y/z aggregation before they reach this fallback. The
 * production Volume planner supplies its larger tiled-source budget explicitly, so this
 * default is not a whole-volume or per-request transport ceiling.
 */
export const CARTESIAN_VOXEL_CAP_VOLUME = 16_000_000;
export const CARTESIAN_VOXEL_CAP_ISOSURFACE = 16_000_000;
export const CARTESIAN_VOXEL_CAP_STREAMLINES = 16_000_000;
export const CARTESIAN_VOXEL_CAP_SUPERCELL = 16_000_000;

/** Minimum stride to keep nx*ny*nz under `cap`; reduces only x/y (z is never reduced). */
export function cartesianPreviewStride(
  nx: number,
  ny: number,
  nz: number,
  cap: number = CARTESIAN_VOXEL_CAP_VOLUME,
): number {
  if (![nx, ny, nz, cap].every(
    (value) => Number.isSafeInteger(value) && value > 0,
  )) {
    throw new Error("Cartesian preview dimensions and voxel budget must be positive integers");
  }
  if (nx * ny * nz <= cap) return 1;
  const maximumUsefulStride = Math.max(nx, ny);
  for (let stride = 2; stride <= maximumUsefulStride; stride++) {
    const reduced = Math.ceil(nx / stride) * Math.ceil(ny / stride) * nz;
    if (reduced <= cap) return stride;
  }
  throw new Error(
    `The ${nx}×${ny}×${nz} field cannot fit the ${cap.toLocaleString()}-voxel budget `
      + "without reducing the vertical axis; use a bounded field window with physical geometry",
  );
}

/** Wrap any scalar 3D field as a VTK voxel grid. Single entry point for scalar 3D widgets. */
export function fieldToCartesianGrid(
  field: Npy3D,
  grid: GridSpacing,
  opts: { stride?: number; maxVoxels?: number; preserveMissing?: boolean } = {},
): CartesianVoxelGrid {
  if (grid.terrainFollowing) return terrainWindowToCartesianGrid(field, grid, opts.preserveMissing);
  grid = gridForWindowedField(field, grid) ?? grid;
  assertSupportedPhysicalCoordinates(grid, field.shape);
  const { field: validField, lower } = cropFieldToValidRange(field, grid);
  const [nx, ny, nz] = validField.shape;
  const fullXCoordinates = validatedCoordinateAxis(grid.scalarCoordinatesM?.[0], field.shape[0]);
  const fullYCoordinates = validatedCoordinateAxis(grid.scalarCoordinatesM?.[1], field.shape[1]);
  const fullZCoordinates = validatedCoordinateAxis(grid.scalarCoordinatesM?.[2], field.shape[2]);
  const validXCoordinates = fullXCoordinates?.slice(lower[0], lower[0] + nx) ?? null;
  const validYCoordinates = fullYCoordinates?.slice(lower[1], lower[1] + ny) ?? null;
  const validZCoordinates = fullZCoordinates?.slice(lower[2], lower[2] + nz) ?? null;
  const drM = coordinateMeanSpacing(validXCoordinates) ?? grid.drM ?? 1;
  const dyM = coordinateMeanSpacing(validYCoordinates) ?? grid.dyM ?? cartesianDyMeters(grid, drM);
  const dzM = coordinateMeanSpacing(validZCoordinates) ?? grid.dzM ?? 1;
  const fullOrigin = cartesianScalarOrigin(grid, field.shape);
  const validOrigin: [number, number, number] = [
    validXCoordinates?.[0] ?? fullOrigin[0] + lower[0] * drM,
    validYCoordinates?.[0] ?? fullOrigin[1] + lower[1] * dyM,
    validZCoordinates?.[0] ?? fullOrigin[2] + lower[2] * dzM,
  ];
  // maxVoxels lowers the cap for heavy renderers; undefined → volume default.
  const stride = opts.stride ?? cartesianPreviewStride(nx, ny, nz, opts.maxVoxels);
  return cartesianBoxToImageData(validField, drM, dyM, dzM, {
    stride,
    preserveMissing: opts.preserveMissing,
    originM: validOrigin,
    sourceXCoordinatesM: validXCoordinates ?? undefined,
    sourceYCoordinatesM: validYCoordinates ?? undefined,
    sourceZCoordinatesM: validZCoordinates ?? undefined,
  });
}

function validatedCoordinateAxis(
  values: readonly number[] | null | undefined,
  expectedLength: number,
): readonly number[] | null {
  return values?.length === expectedLength
    && values.every((value, index) =>
      Number.isFinite(value) && (index === 0 || value > values[index - 1])
    )
    ? values
    : null;
}

function coordinateMeanSpacing(values: readonly number[] | null): number | null {
  return values && values.length > 1
    ? (values.at(-1)! - values[0]) / (values.length - 1)
    : null;
}

function assertSupportedPhysicalCoordinates(
  grid: GridSpacing,
  shape: readonly number[],
): void {
  if (grid.terrainFollowing) {
    throw new Error(
      "Terrain-following 3D fields require the exact physical height lattice; uniform-grid rendering is disabled to prevent a flattened result",
    );
  }
  if (grid.geometryKind === "index") {
    throw new Error(
      "3D rendering requires complete finite, increasing x/y/z coordinates with declared length units; this dataset is available for index-based 2D inspection",
    );
  }
  if (grid.geometryKind === "physical") {
    const spacings = [grid.drM, grid.dyM, grid.dzM];
    const complete = spacings.every((spacing, axis) =>
      (spacing != null && spacing > 0)
      || validatedCoordinateAxis(grid.scalarCoordinatesM?.[axis], shape[axis]) != null
    );
    if (!complete) {
      throw new Error(
        "3D rendering requires a complete finite, increasing physical coordinate or positive spacing for every x/y/z axis",
      );
    }
  }
}

/** Crop halo cells per manifest valid range (inclusive). Falls back to full field if missing. */
export function cropFieldToValidRange(
  field: Npy3D,
  grid: GridSpacing,
): { field: Npy3D; lower: [number, number, number] } {
  const fullLower: [number, number, number] = [0, 0, 0];
  const range = grid.validIndexRange;
  if (!range || !finiteTriplet(range.lower) || !finiteTriplet(range.upper)) {
    return { field, lower: fullLower };
  }
  const lower = range.lower.map(Math.trunc) as [number, number, number];
  const upper = range.upper.map(Math.trunc) as [number, number, number];
  for (let axis = 0; axis < 3; axis++) {
    if (lower[axis] < 0 || upper[axis] < lower[axis] || upper[axis] >= field.shape[axis]) {
      return { field, lower: fullLower };
    }
  }
  if (
    lower.every((value) => value === 0) &&
    upper.every((value, axis) => value === field.shape[axis] - 1)
  ) {
    return { field, lower };
  }
  const shape: [number, number, number] = [
    upper[0] - lower[0] + 1,
    upper[1] - lower[1] + 1,
    upper[2] - lower[2] + 1,
  ];
  const data = new Float32Array(shape[0] * shape[1] * shape[2]);
  const [, sourceNy, sourceNz] = field.shape;
  let dst = 0;
  for (let i = lower[0]; i <= upper[0]; i++) {
    for (let j = lower[1]; j <= upper[1]; j++) {
      const sourceBase = (i * sourceNy + j) * sourceNz + lower[2];
      const count = shape[2];
      data.set(field.data.subarray(sourceBase, sourceBase + count), dst);
      dst += count;
    }
  }
  return { field: { shape, data }, lower };
}

/**
 * Wrap a (u_x, u_y, u_z) vector field as an interleaved 3-component VTK voxel grid.
 * No rotation; non-finite components → zero vector; geometry matches cartesianBoxToImageData.
 */
export function cartesianVectorBoxToImageData(
  uField: Npy3D,
  vField: Npy3D,
  wField: Npy3D,
  dxM: number,
  dyM: number,
  dzM: number,
  opts: {
    stride?: number;
    originM?: [number, number, number];
    sourceXCoordinatesM?: readonly number[];
    sourceYCoordinatesM?: readonly number[];
    sourceZCoordinatesM?: readonly number[];
    preserveMissing?: boolean;
  } = {},
): CartesianVoxelGrid {
  const [nxFull, nyFull, nzFull] = uField.shape;
  if (
    vField.shape[0] !== nxFull ||
    vField.shape[1] !== nyFull ||
    vField.shape[2] !== nzFull ||
    wField.shape[0] !== nxFull ||
    wField.shape[1] !== nyFull ||
    wField.shape[2] !== nzFull
  ) {
    throw new Error(
      `Vector components must share shape; got u=${uField.shape}, v=${vField.shape}, w=${wField.shape}`,
    );
  }
  const stride = Math.max(1, Math.floor(opts.stride ?? 1));
  const nz = nzFull;
  const nx = stride === 1 ? nxFull : Math.min(nxFull, Math.max(2, Math.ceil(nxFull / stride)));
  const ny = stride === 1 ? nyFull : Math.min(nyFull, Math.max(2, Math.ceil(nyFull / stride)));
  // Full-extent spacing: dx = fullX/(nx-1) registers across strides/caps.
  const fullX = (nxFull - 1) * dxM;
  const fullY = (nyFull - 1) * dyM;
  const dx = nx > 1 ? fullX / (nx - 1) : dxM;
  const dy = ny > 1 ? fullY / (ny - 1) : dyM;

  const srcU = uField.data;
  const srcV = vField.data;
  const srcW = wField.data;
  const sourceNonFiniteCount = scanFinite(srcU).nonFiniteCount
    + scanFinite(srcV).nonFiniteCount
    + scanFinite(srcW).nonFiniteCount;
  const srcSx = nyFull * nzFull;
  const srcSy = nzFull;
  const out = new Float32Array(nx * ny * nz * 3);
  const xMap = opts.sourceXCoordinatesM
    ? coordinateInterpolationAxis(opts.sourceXCoordinatesM, nx)
    : null;
  const yMap = opts.sourceYCoordinatesM
    ? coordinateInterpolationAxis(opts.sourceYCoordinatesM, ny)
    : null;
  const zMap = opts.sourceZCoordinatesM
    ? coordinateInterpolationAxis(opts.sourceZCoordinatesM, nz)
    : null;
  const aggregateXY = stride > 1 && (nx < nxFull || ny < nyFull);
  const xAggregation = aggregateXY
    ? areaAggregationAxis(nxFull, nx, opts.sourceXCoordinatesM)
    : null;
  const yAggregation = aggregateXY
    ? areaAggregationAxis(nyFull, ny, opts.sourceYCoordinatesM)
    : null;
  for (let ix = 0; ix < nx; ix++) {
    const xWeights = xAggregation?.[ix];
    const six = nx > 1 ? Math.min(nxFull - 1, Math.round((ix * (nxFull - 1)) / (nx - 1))) : 0;
    const x0 = xMap?.lower[ix] ?? six;
    const x1 = xMap?.upper[ix] ?? six;
    const tx = xMap?.weight[ix] ?? 0;
    for (let iy = 0; iy < ny; iy++) {
      const yWeights = yAggregation?.[iy];
      const siy = ny > 1 ? Math.min(nyFull - 1, Math.round((iy * (nyFull - 1)) / (ny - 1))) : 0;
      const y0 = yMap?.lower[iy] ?? siy;
      const y1 = yMap?.upper[iy] ?? siy;
      const ty = yMap?.weight[iy] ?? 0;
      const dstBase = (ix * ny + iy) * nz * 3;
      for (let iz = 0; iz < nz; iz++) {
        const d = dstBase + iz * 3;
        const z0 = zMap?.lower[iz] ?? iz;
        const z1 = zMap?.upper[iz] ?? iz;
        const tz = zMap?.weight[iz] ?? 0;
        for (let component = 0; component < 3; component++) {
          const source = component === 0 ? srcU : component === 1 ? srcV : srcW;
          if (xWeights && yWeights) {
            let sum = 0;
            let finiteWeight = 0;
            let missing = false;
            for (const xWeight of xWeights) {
              for (const yWeight of yWeights) {
                const sourceBase = xWeight.index * srcSx + yWeight.index * srcSy;
                const value = linearFiniteSample(
                  source[sourceBase + z0],
                  source[sourceBase + z1],
                  tz,
                );
                if (Number.isFinite(value)) {
                  const weight = xWeight.weight * yWeight.weight;
                  sum += weight * value;
                  finiteWeight += weight;
                } else if (xWeight.weight * yWeight.weight > 0) missing = true;
              }
            }
            out[d + component] = finiteWeight > 0 && !(opts.preserveMissing && missing)
              ? sum / finiteWeight : opts.preserveMissing ? Number.NaN : 0;
          } else {
            out[d + component] = trilinearFinite(
              source,
              srcSx,
              srcSy,
              x0,
              x1,
              y0,
              y1,
              z0,
              z1,
              tx,
              ty,
              tz,
              opts.preserveMissing ? Number.NaN : 0,
            );
          }
        }
      }
    }
  }

  return {
    shape: [nx, ny, nz],
    spacing: [dx, dy, dzM],
    origin: opts.originM ?? [-(nx - 1) * dx * 0.5, -(ny - 1) * dy * 0.5, 0.5 * dzM],
    data: out,
    sourceNonFiniteCount,
    ...(aggregateXY ? {
      lod: {
        nativeShape: [nxFull, nyFull, nzFull] as [number, number, number],
        stride,
        aggregation: "area_mean_xy" as const,
      },
    } : {}),
  };
}

/** Wrap a (u, v, w) vector field as a VTK voxel grid. Single entry point for the streamline widget. */
export function vectorFieldToCartesianGrid(
  uField: Npy3D,
  vField: Npy3D,
  wField: Npy3D,
  grid: GridSpacing,
  opts: { stride?: number; maxVoxels?: number; preserveMissing?: boolean } = {},
): CartesianVoxelGrid {
  if (grid.terrainFollowing) {
    const u = terrainWindowToCartesianGrid(uField, grid, opts.preserveMissing);
    const v = terrainWindowToCartesianGrid(vField, grid, opts.preserveMissing);
    const w = terrainWindowToCartesianGrid(wField, grid, opts.preserveMissing);
    if (u.shape.some((count, axis) => count !== v.shape[axis] || count !== w.shape[axis])) {
      throw new Error("Terrain vector components remapped onto different Cartesian lattices");
    }
    const data = new Float32Array(u.data.length * 3);
    for (let index = 0; index < u.data.length; index++) {
      data[index * 3] = u.data[index];
      data[index * 3 + 1] = v.data[index];
      data[index * 3 + 2] = w.data[index];
    }
    return { ...u, data, sourceNonFiniteCount: (u.sourceNonFiniteCount ?? 0)
      + (v.sourceNonFiniteCount ?? 0) + (w.sourceNonFiniteCount ?? 0) };
  }
  grid = gridForWindowedField(uField, grid) ?? grid;
  assertSupportedPhysicalCoordinates(grid, uField.shape);
  const croppedU = cropFieldToValidRange(uField, grid);
  const croppedV = cropFieldToValidRange(vField, grid);
  const croppedW = cropFieldToValidRange(wField, grid);
  const [nx, ny, nz] = croppedU.field.shape;
  const fullXCoordinates = validatedCoordinateAxis(grid.scalarCoordinatesM?.[0], uField.shape[0]);
  const fullYCoordinates = validatedCoordinateAxis(grid.scalarCoordinatesM?.[1], uField.shape[1]);
  const fullZCoordinates = validatedCoordinateAxis(grid.scalarCoordinatesM?.[2], uField.shape[2]);
  const validXCoordinates = fullXCoordinates?.slice(croppedU.lower[0], croppedU.lower[0] + nx) ?? null;
  const validYCoordinates = fullYCoordinates?.slice(croppedU.lower[1], croppedU.lower[1] + ny) ?? null;
  const validZCoordinates = fullZCoordinates?.slice(croppedU.lower[2], croppedU.lower[2] + nz) ?? null;
  const drM = coordinateMeanSpacing(validXCoordinates) ?? grid.drM ?? 1;
  const dyM = coordinateMeanSpacing(validYCoordinates) ?? grid.dyM ?? cartesianDyMeters(grid, drM);
  const dzM = coordinateMeanSpacing(validZCoordinates) ?? grid.dzM ?? 1;
  const fullOrigin = cartesianScalarOrigin(grid, uField.shape);
  const validOrigin: [number, number, number] = [
    validXCoordinates?.[0] ?? fullOrigin[0] + croppedU.lower[0] * drM,
    validYCoordinates?.[0] ?? fullOrigin[1] + croppedU.lower[1] * dyM,
    validZCoordinates?.[0] ?? fullOrigin[2] + croppedU.lower[2] * dzM,
  ];
  const stride = opts.stride ?? cartesianPreviewStride(nx, ny, nz, opts.maxVoxels);
  return cartesianVectorBoxToImageData(
    croppedU.field,
    croppedV.field,
    croppedW.field,
    drM,
    dyM,
    dzM,
    {
      stride,
      preserveMissing: opts.preserveMissing,
      originM: validOrigin,
      sourceXCoordinatesM: validXCoordinates ?? undefined,
      sourceYCoordinatesM: validYCoordinates ?? undefined,
      sourceZCoordinatesM: validZCoordinates ?? undefined,
    },
  );
}

/**
 * Repack C-order (z-fastest) scalar buffer into VTK x-fastest order for `setScalars`.
 * Always copies; non-cubic grids would otherwise be scrambled.
 */
export function cartesianGridToVtkScalars(cart: CartesianVoxelGrid): Float32Array {
  const [nx, ny, nz] = cart.shape;
  const src = cart.data;
  const nxy = nx * ny;
  const out = new Float32Array(nxy * nz);
  // Walk source (z-fastest), scatter to x-fastest destination.
  let s = 0;
  for (let i = 0; i < nx; i++) {
    for (let j = 0; j < ny; j++) {
      const col = i + j * nx;
      for (let k = 0; k < nz; k++) {
        out[col + k * nxy] = src[s++];
      }
    }
  }
  return out;
}

/** Repack 3-component interleaved buffer to VTK x-fastest tuple order for `setVectors`. */
export function cartesianGridToVtkVectors(cart: CartesianVoxelGrid): Float32Array {
  const [nx, ny, nz] = cart.shape;
  const src = cart.data;
  const nxy = nx * ny;
  const out = new Float32Array(nxy * nz * 3);
  let s = 0;
  for (let i = 0; i < nx; i++) {
    for (let j = 0; j < ny; j++) {
      const col = i + j * nx;
      for (let k = 0; k < nz; k++) {
        const d = (col + k * nxy) * 3;
        out[d] = src[s++];
        out[d + 1] = src[s++];
        out[d + 2] = src[s++];
      }
    }
  }
  return out;
}

/** Reorder an exclusively owned vector grid without allocating a second 3-component lattice. */
export function cartesianGridToVtkVectorsInPlace(cart: CartesianVoxelGrid): Float32Array {
  const [nx, ny, nz] = cart.shape;
  const values = cart.data;
  const tuples = nx * ny * nz;
  if (values.length !== tuples * 3) {
    throw new Error("Cartesian vector grid does not contain exactly three values per cell");
  }
  const visited = new Uint8Array(Math.ceil(tuples / 8));
  const isVisited = (index: number) => (visited[index >> 3] & (1 << (index & 7))) !== 0;
  const markVisited = (index: number) => { visited[index >> 3] |= 1 << (index & 7); };
  const nxy = nx * ny;
  const destination = (source: number) => {
    const k = source % nz;
    const column = (source - k) / nz;
    const j = column % ny;
    const i = (column - j) / ny;
    return i + j * nx + k * nxy;
  };
  for (let start = 0; start < tuples; start += 1) {
    if (isVisited(start)) continue;
    let current = start;
    let x = values[current * 3];
    let y = values[current * 3 + 1];
    let z = values[current * 3 + 2];
    do {
      const next = destination(current);
      const offset = next * 3;
      const nextX = values[offset];
      const nextY = values[offset + 1];
      const nextZ = values[offset + 2];
      values[offset] = x;
      values[offset + 1] = y;
      values[offset + 2] = z;
      markVisited(current);
      current = next;
      x = nextX;
      y = nextY;
      z = nextZ;
    } while (current !== start);
  }
  return values;
}

/** Max vector magnitude over a 3-component grid. Returns 0 for empty/all-zero. Non-finite skipped. */
export function cartesianVectorMaxMagnitude(cart: CartesianVoxelGrid): number {
  const d = cart.data;
  let maxSq = 0;
  for (let i = 0; i + 2 < d.length; i += 3) {
    const m = d[i] * d[i] + d[i + 1] * d[i + 1] + d[i + 2] * d[i + 2];
    if (m > maxSq) maxSq = m; // NaN comparisons are false -> skipped
  }
  return Math.sqrt(maxSq);
}

/** Speed quantile (0..1) via 256-bucket histogram; ignores non-finite and zero samples. */
export function cartesianVectorSpeedQuantile(cart: CartesianVoxelGrid, fraction: number): number {
  const d = cart.data;
  let maxSq = 0;
  for (let i = 0; i + 2 < d.length; i += 3) {
    const m = d[i] * d[i] + d[i + 1] * d[i + 1] + d[i + 2] * d[i + 2];
    if (m > maxSq) maxSq = m; // NaN compares false -> skipped
  }
  const maxMag = Math.sqrt(maxSq);
  if (!(maxMag > 0)) return 0;
  const B = 256;
  const hist = new Int32Array(B);
  let total = 0;
  for (let i = 0; i + 2 < d.length; i += 3) {
    const m = Math.sqrt(d[i] * d[i] + d[i + 1] * d[i + 1] + d[i + 2] * d[i + 2]);
    if (!(m > 0)) continue;
    const b = Math.min(B - 1, Math.floor((m / maxMag) * B));
    hist[b]++;
    total++;
  }
  if (total === 0) return 0;
  const target = Math.max(0, Math.min(1, fraction)) * total;
  let cum = 0;
  for (let b = 0; b < B; b++) {
    cum += hist[b];
    if (cum >= target) return ((b + 0.5) / B) * maxMag;
  }
  return maxMag;
}
