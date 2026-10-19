import { cloudCoordinateError } from "./cloudCoordinates";
import type { FieldLoadWindow, GridInfo, StepInfo } from "@/lib/run/runOutputs";
import type { Npy3D } from "@/lib/platform/npy";
import type { PreparedVtkImage } from "@/lib/vtk/vtkPreparation";
import { cartesianDyMeters, isCartesianGrid } from "@/lib/science/cartesianInterp";
import { windowAxisCoordinates } from "@/lib/science/windowedGrid";
import { motionMode, windsAreEarthRelative } from "@/lib/run/frameMotion";
import { inverse3, type AffineWind } from "./affineWind";
import { createCloudMotionSampler } from "./cloudMotion";

export interface CloudWindSummary extends AffineWind {
  residualMs: number;
  weighting: "condensate-volume" | "condensate-mass";
  spreadMs: number;
  supportedFraction: number;
  cloudSamples: number;
  sourceStep: number;
  sourceTimeS: number;
}
export type CloudWindEstimate = { ok: true; summary: CloudWindSummary } | { ok: false; reason: string };
export const CLOUD_WIND_MAX_SOURCE_BYTES = 4 * 1024 * 1024;
export const CLOUD_WIND_MAX_COMPRESSED_SOURCE_BYTES = 32 * 1024 * 1024;
export const CLOUD_WIND_MAX_SAMPLES = 32_768;

function earthWindOffset(step: StepInfo): [number, number] | null {
  if (motionMode(step) == null) return null;
  if (windsAreEarthRelative(step)) return [0, 0];
  if (motionMode(step) === "stationary" && step.motion?.frameVelocityMs == null) return [0, 0];
  const velocity = step.motion?.frameVelocityMs;
  return step.motion?.windsEarthRelative === false && velocity?.length === 2 && velocity.every(Number.isFinite)
    ? [velocity[0], velocity[1]] : null;
}

export function cloudDensityAvailable(step: StepInfo | null | undefined): boolean {
  return Boolean(step?.fields.includes("rho") && ["kg/m^3", "kg/m3", "kgm-3", "kgm^-3"].includes(
    (step.fieldUnits?.rho ?? "").replaceAll("−", "-").replaceAll(" ", "")));
}

export function cloudWindUnavailable(step: StepInfo | null | undefined): string | null {
  if (!step?.grid || step.grid.geometryKind === "index" || !isCartesianGrid(step.grid.coordinateSystem)) return "Wind preview requires physical Cartesian coordinates";
  const coordinateError = cloudCoordinateError(step);
  if (coordinateError) return coordinateError;
  const grid = step.grid;
  if ([grid.nr, grid.nth, grid.nz].some((size, axis) => {
    const coordinates = grid.scalarCoordinatesM?.[axis];
    if (grid.terrainFollowing && axis === 2) return !Number.isSafeInteger(size) || size! < 1;
    if (!Number.isSafeInteger(size) || size! < 1 || coordinates && coordinates.length !== size) return true;
    if (coordinates && coordinates.length === size) return coordinates.some((value, index) => !Number.isFinite(value) || index > 0 && value <= coordinates[index - 1]);
    const spacing = [grid.drM, cartesianDyMeters(grid, grid.drM ?? NaN), grid.dzM][axis];
    return !Number.isFinite(spacing) || !(spacing! > 0);
  })) return "Wind preview requires physical sample coordinates";
  if (!["u", "v", "w"].every(field => step.fields.includes(field))) return "Three-dimensional wind fields were not saved";
  if (!["u", "v", "w"].every(field => ["m/s", "ms-1", "ms^-1"].includes((step.fieldUnits?.[field] ?? "").replaceAll("−", "-").replaceAll(" ", "")))) return "Wind preview requires declared wind units in m/s";
  if (!earthWindOffset(step)) return "The saved wind coordinate frame is unknown";
  return null;
}

export function planCloudWindWindow(grid: GridInfo | null | undefined, compressed = false): FieldLoadWindow | null {
  if (!grid) return null;
  const shape = [grid.nr, grid.nth, grid.nz];
  if (shape.some(size => !Number.isSafeInteger(size) || size! < 1)) return null;
  const lower = grid.validIndexRange?.lower ?? [0, 0, 0];
  const upper = grid.validIndexRange?.upper ?? shape.map(size => size! - 1);
  if (lower.length !== 3 || upper.length !== 3 || lower.some((value, axis) => !Number.isSafeInteger(value) || value < 0
      || !Number.isSafeInteger(upper[axis]) || upper[axis] < value || upper[axis] >= shape[axis]!)) return null;
  const extent = upper.map((value, axis) => value - lower[axis] + 1);
  const stride = extent.map(size => Math.max(1, Math.ceil(size / 32)));
  const output = extent.map((size, axis) => Math.ceil(size / stride[axis]));
  const cells = output[0] * output[1] * output[2];
  const spanZ = (output[2] - 1) * stride[2] + 1;
  const geometryBytes = grid.terrainFollowing ? 96 * cells
    + 24 * output[0] * output[1] * (spanZ <= output[2] * 16 ? spanZ : output[2]) : 0;
  const sourceBudget = compressed || grid.terrainFollowing ? CLOUD_WIND_MAX_COMPRESSED_SOURCE_BYTES : CLOUD_WIND_MAX_SOURCE_BYTES;
  if (geometryBytes >= sourceBudget || !compressed && output[0] * output[1] * extent[2] * 4 + geometryBytes > sourceBudget) return null;
  return { aggregation: "point_sample", origin: [...lower], extent, stride,
    maxOutputBytes: CLOUD_WIND_MAX_SAMPLES * (grid.terrainFollowing ? 36 : 4) + (grid.terrainFollowing ? 8 : 0), maxSourceBytes: sourceBudget };
}

export function estimateCloudWind(
  step: StepInfo, window: FieldLoadWindow, fields: { u: Npy3D; v: Npy3D; w: Npy3D; rho?: Npy3D | null },
  cloud: PreparedVtkImage, context: PreparedVtkImage | null,
): CloudWindEstimate {
  const { u, v, w, rho } = fields;
  const unavailable = cloudWindUnavailable(step);
  if (unavailable) return { ok: false, reason: unavailable };
  if (rho && !cloudDensityAvailable(step)) return { ok: false, reason: "Air density units are unavailable" };
  const grid = step.grid!;
  if ([window.origin, window.extent, window.stride].some(axis => axis.length !== 3)
      || window.origin.some((origin, axis) => !Number.isSafeInteger(origin) || origin < 0
        || !Number.isSafeInteger(window.extent[axis]) || window.extent[axis] < 1
        || !Number.isSafeInteger(window.stride[axis]) || window.stride[axis] < 1
        || origin + window.extent[axis] > [grid.nr!, grid.nth!, grid.nz!][axis])) return { ok: false, reason: "Invalid wind sampling window" };
  const terrain = Boolean(grid.terrainFollowing);
  const expected = window.extent.map((size, axis) => Math.ceil(size / window.stride[axis]));
  if (window.aggregation !== "point_sample" || expected.reduce((count, size) => count * size, 1) > CLOUD_WIND_MAX_SAMPLES
      || [u, v, w, ...(rho ? [rho] : [])].some(field => field.shape.some((size, axis) => size !== expected[axis])
        || field.data.length !== expected.reduce((count, size) => count * size, 1)
        || !field.windowGeometry || (terrain
          ? !field.windowGeometry.centroidsM || !field.windowGeometry.volumesM3 || field.windowGeometry.weighting !== "physical_cell_volume"
          : field.windowGeometry.centroidsM != null || field.windowGeometry.weighting !== "uniform_cell_volume")
        || field.windowGeometry.aggregation !== "point_sample" || field.windowGeometry.sampleLocation !== "source_cell_center"
        || JSON.stringify(field.windowGeometry.sourceShape) !== JSON.stringify([grid.nr, grid.nth, grid.nz])
        || ["origin", "extent", "stride"].some(key => JSON.stringify(field.windowGeometry![key as "origin"]) !== JSON.stringify(window[key as "origin"])))) {
    return { ok: false, reason: "Wind samples do not match the requested source lattice" };
  }
  const sourceShape = [grid.nr!, grid.nth!, grid.nz!];
  const resolvedGrid = { ...grid, dyM: cartesianDyMeters(grid, grid.drM ?? 1) };
  const axes = expected.map((count, axis) => windowAxisCoordinates(resolvedGrid, axis, sourceShape[axis], window.origin[axis], window.extent[axis], window.stride[axis], count, true));
  if (axes.some((axis, a) => (!terrain || a < 2) && axis.some((value, index) => !Number.isFinite(value) || index > 0 && value <= axis[index - 1]))) return { ok: false, reason: "Invalid wind sample coordinates" };
  const widths = axes.map((axis, a) => {
    if (terrain) {
      const end = window.extent[a] - 0.5;
      return axis.map((_, i) => (i + 1 < axis.length ? (i + 0.5) * window.stride[a] : end)
        - (i > 0 ? (i - 0.5) * window.stride[a] : -0.5));
    }
    const coordinate = (index: number) => windowAxisCoordinates(resolvedGrid, a, sourceShape[a], index, 1, 1, 1, true)[0];
    const first = window.origin[a], last = first + window.extent[a] - 1;
    const spacing = [resolvedGrid.drM, resolvedGrid.dyM, resolvedGrid.dzM][a] ?? 1;
    const lowerFace = coordinate(first) - 0.5 * (sourceShape[a] > 1
      ? first > 0 ? coordinate(first) - coordinate(first - 1) : coordinate(1) - coordinate(0) : spacing);
    const upperFace = coordinate(last) + 0.5 * (sourceShape[a] > 1
      ? last + 1 < sourceShape[a] ? coordinate(last + 1) - coordinate(last) : coordinate(last) - coordinate(last - 1) : spacing);
    return axis.map((value, index) => {
      const before = index > 0 ? (axis[index - 1] + value) * 0.5 : Math.max(lowerFace, grid.physicalBoundsM?.lower[a] ?? -Infinity);
      const after = index + 1 < axis.length ? (value + axis[index + 1]) * 0.5 : Math.min(upperFace, grid.physicalBoundsM?.upper[a] ?? Infinity);
      return after - before;
    });
  });
  const centroids = u.windowGeometry?.centroidsM;
  const volumes = u.windowGeometry?.volumesM3;
  if (terrain && [u, v, w, ...(rho ? [rho] : [])].some(field => {
    const geometry = field.windowGeometry!;
    return [geometry.centroidsM!.x, geometry.centroidsM!.y, geometry.centroidsM!.z, geometry.volumesM3!]
      .some((values, axis) => values.length !== field.data.length || values.some((value, i) =>
        !Number.isFinite(value) || axis === 3 && value <= 0
        || value !== [centroids!.x, centroids!.y, centroids!.z, volumes!][axis][i]));
  })) return { ok: false, reason: "Wind fields disagree on physical terrain geometry" };
  const fine = createCloudMotionSampler(cloud), coarse = context ? createCloudMotionSampler(context) : null;
  const offset = earthWindOffset(step)!;
  const upper = cloud.origin.map((value, axis) => value + (cloud.shape[axis] - 1) * cloud.spacing[axis]);
  let weight = 0, supportedWeight = 0, fittedWeight = 0, variance = 0, cloudSamples = 0;
  const centerM: [number, number, number] = [0, 0, 0], velocityMs: [number, number, number] = [0, 0, 0];
  const covariance = new Array<number>(9).fill(0), cross = new Array<number>(9).fill(0);

  for (let x = 0; x < expected[0]; x++) for (let y = 0; y < expected[1]; y++) for (let z = 0; z < expected[2]; z++) {
    const index = (x * expected[1] + y) * expected[2] + z;
    const px = terrain ? centroids!.x[index] : axes[0][x], py = terrain ? centroids!.y[index] : axes[1][y], pz = terrain ? centroids!.z[index] : axes[2][z];
    const insideFine = px >= cloud.origin[0] && px <= upper[0]
      && py >= cloud.origin[1] && py <= upper[1] && pz >= cloud.origin[2] && pz <= upper[2];
    const density = insideFine ? fine(px, py, pz) : coarse?.(px, py, pz) ?? null;
    if (density == null || density <= 0.00005) continue;
    const cloudWeight = density * widths[0][x] * widths[1][y] * widths[2][z] * (terrain ? volumes![index] : 1);
    if (!(cloudWeight > 0) || !Number.isFinite(cloudWeight)) return { ok: false, reason: "Invalid cloud wind weight" };
    weight += cloudWeight; cloudSamples++;
    const vx = u.data[index] + offset[0], vy = v.data[index] + offset[1], vz = w.data[index];
    const airDensity = rho ? rho.data[index] : 1;
    if (!Number.isFinite(vx) || !Number.isFinite(vy) || !Number.isFinite(vz) || !Number.isFinite(airDensity) || airDensity <= 0) continue;
    const sampleWeight = cloudWeight * airDensity;
    if (!Number.isFinite(sampleWeight)) return { ok: false, reason: "Invalid cloud wind weight" };
    supportedWeight += cloudWeight;
    const nextWeight = fittedWeight + sampleWeight;
    const fraction = sampleWeight / nextWeight, factor = sampleWeight * (1 - fraction);
    const dp = [px - centerM[0], py - centerM[1], pz - centerM[2]];
    const dv = [vx - velocityMs[0], vy - velocityMs[1], vz - velocityMs[2]];
    for (let row = 0; row < 3; row++) {
      centerM[row] += fraction * dp[row]; velocityMs[row] += fraction * dv[row];
      variance += factor * dv[row] * dv[row];
      for (let col = 0; col < 3; col++) {
        covariance[row * 3 + col] += factor * dp[row] * dp[col];
        cross[row * 3 + col] += factor * dv[row] * dp[col];
      }
    }
    fittedWeight = nextWeight;
  }
  if (cloudSamples < 8 || !(weight > 0) || !Number.isFinite(weight) || !Number.isFinite(fittedWeight)
      || supportedWeight / weight < 0.95) return { ok: false, reason: "Insufficient cloud-supported wind samples" };
  const scales = [0, 1, 2].map(axis => Math.sqrt(Math.max(1e-12, covariance[axis * 3 + axis] / fittedWeight)));
  const correlation = covariance.map((value, index) => value / fittedWeight / scales[Math.floor(index / 3)] / scales[index % 3] + (index % 4 === 0 ? 1e-8 : 0));
  const inverse = inverse3(correlation);
  if (!inverse) return { ok: false, reason: "Wind sample positions do not constrain the motion fit" };
  const gradientPerS = Array.from({ length: 9 }, (_, index) => {
    const row = Math.floor(index / 3), col = index % 3;
    return [0, 1, 2].reduce((sum, k) => sum + cross[row * 3 + k] / fittedWeight / scales[k] * inverse[k * 3 + col] / scales[col], 0);
  });
  const explained = gradientPerS.reduce((sum, value, index) => sum + value * cross[index], 0);
  let fittedVariance = 0;
  for (let row = 0; row < 3; row++) for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) fittedVariance += gradientPerS[row * 3 + i] * covariance[i * 3 + j] * gradientPerS[row * 3 + j];
  const spreadMs = Math.sqrt(Math.max(0, variance / fittedWeight));
  const residualMs = Math.sqrt(Math.max(0, (variance - 2 * explained + fittedVariance) / fittedWeight));
  if (!Number.isSafeInteger(step.index) || step.index < 0 || ![...velocityMs, ...centerM, ...gradientPerS, spreadMs, residualMs, step.simulationTimeS].every(Number.isFinite)) return { ok: false, reason: "Invalid cloud wind estimate" };
  return { ok: true, summary: { velocityMs, centerM, gradientPerS, spreadMs, residualMs,
    weighting: rho ? "condensate-mass" : "condensate-volume", supportedFraction: supportedWeight / weight,
    cloudSamples, sourceStep: step.index, sourceTimeS: step.simulationTimeS! } };
}
