import { affineWindTransform, averageAffineWind, transformPoint, type AffineWind, type AffineMatrix } from "./affineWind";
import type { CloudWindEstimate } from "./cloudWind";
import type { PreparedVtkImage } from "@/lib/vtk/vtkPreparation";

export type CloudMotionSource = {
  role: "cloud" | "context";
  runDir: string;
  step: number;
  publicationGeneration: string | null;
  shape: readonly number[];
  spacingM: readonly number[];
  originM: readonly number[];
};

export interface CloudMotionFrame {
  key: string;
  domainKey: string;
  step: number;
  timeS: number;
  center: readonly [number, number, number];
  cloud: PreparedVtkImage;
  context: PreparedVtkImage | null;
  wind?: CloudWindEstimate;
  coordinateError?: string | null;
  sources?: readonly CloudMotionSource[];
}

function validImage(image: PreparedVtkImage): boolean {
  const cells = image.shape.reduce((count, size) => count * size, 1);
  return image.shape.length === 3 && image.shape.every(size => Number.isSafeInteger(size) && size > 0)
    && Number.isSafeInteger(cells) && image.values.length === cells
    && (image.validity == null || image.validity.length === cells)
    && image.origin.length === 3 && image.origin.every(Number.isFinite)
    && image.spacing.length === 3 && image.spacing.every(value => Number.isFinite(value) && value > 0);
}

export function sampleCloudMotion(image: PreparedVtkImage, position: readonly number[]): number | null {
  return position.length === 3 && validImage(image) ? sampleCloud(image, position[0], position[1], position[2]) : null;
}

export function createCloudMotionSampler(image: PreparedVtkImage): (x: number, y: number, z: number) => number | null {
  return validImage(image) ? (x, y, z) => sampleCloud(image, x, y, z) : () => null;
}

function snapCoordinate(value: number): number {
  const nearest = Math.round(value);
  return Math.abs(value - nearest) < 1e-7 ? nearest : value;
}

function sampleCloud(image: PreparedVtkImage, px: number, py: number, pz: number): number | null {
  const x = snapCoordinate((px - image.origin[0]) / image.spacing[0]);
  const y = snapCoordinate((py - image.origin[1]) / image.spacing[1]);
  const z = snapCoordinate((pz - image.origin[2]) / image.spacing[2]);
  const [nx, ny, nz] = image.shape;
  if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)
      || x < 0 || y < 0 || z < 0 || x > nx - 1 || y > ny - 1 || z > nz - 1) return null;
  const ix = Math.floor(x), iy = Math.floor(y), iz = Math.floor(z);
  const tx = x - ix, ty = y - iy, tz = z - iz;
  let result = 0;
  for (let dz = 0; dz <= Number(tz > 0); dz++) for (let dy = 0; dy <= Number(ty > 0); dy++) for (let dx = 0; dx <= Number(tx > 0); dx++) {
    const weight = (dx ? tx : 1 - tx) * (dy ? ty : 1 - ty) * (dz ? tz : 1 - tz);
    const index = ix + dx + nx * (iy + dy + ny * (iz + dz));
    const value = image.values[index];
    if (image.validity?.[index] === 0 || !Number.isFinite(value) || value < 0) return null;
    result += weight * value;
  }
  return result;
}

export function planCloudMotion(previous: CloudMotionFrame, current: CloudMotionFrame):
  { ok: true; displacement: [number, number, number]; difference: number; sampledPoints: number; method: "cloud-centroid" | "wind"; wind?: AffineWind } | { ok: false; reason: string } {
  if (previous.coordinateError || current.coordinateError) return { ok: false, reason: previous.coordinateError ?? current.coordinateError! };
  if (previous.domainKey !== current.domainKey || Boolean(previous.context) !== Boolean(current.context)) {
    return { ok: false, reason: "Cloud grid or display settings changed" };
  }
  if (!Number.isFinite(previous.timeS) || !Number.isFinite(current.timeS)
      || !Number.isFinite(current.timeS - previous.timeS)
      || !Number.isSafeInteger(previous.step) || !Number.isSafeInteger(current.step)
      || previous.step < 0 || current.timeS <= previous.timeS || current.step <= previous.step) {
    return { ok: false, reason: "Saved-frame boundary or backward step" };
  }
  if (previous.center.length !== 3 || current.center.length !== 3 || ![...previous.center, ...current.center].every(Number.isFinite)) {
    return { ok: false, reason: "No supported cloud position" };
  }
  let displacement: [number, number, number] = [current.center[0] - previous.center[0], current.center[1] - previous.center[1], 0];
  const method = previous.wind || current.wind ? "wind" : "cloud-centroid";
  let wind: AffineWind | undefined, forwardTransform: AffineMatrix | undefined, backwardTransform: AffineMatrix | undefined;
  if (method === "wind") {
    if (!previous.wind?.ok) return { ok: false, reason: previous.wind?.reason ?? "Waiting for a wind-supported cloud frame" };
    if (!current.wind?.ok) return { ok: false, reason: current.wind?.reason ?? "Saved wind data is unavailable" };
    const left = previous.wind.summary, right = current.wind.summary;
    if (left.sourceTimeS !== previous.timeS || right.sourceTimeS !== current.timeS
        || [left, right].some(wind => wind.velocityMs.length !== 3 || wind.centerM.length !== 3 || wind.gradientPerS.length !== 9
          || ![...wind.velocityMs, ...wind.centerM, ...wind.gradientPerS].every(Number.isFinite)
          || !Number.isFinite(wind.residualMs) || wind.residualMs < 0 || !(wind.supportedFraction >= 0.95 && wind.supportedFraction <= 1))) {
      return { ok: false, reason: "Wind evidence does not match the saved cloud frames" };
    }
    const dt = current.timeS - previous.timeS;
    try {
      wind = averageAffineWind(left, right);
      forwardTransform = affineWindTransform(wind, dt);
      backwardTransform = affineWindTransform(wind, -dt);
      displacement = transformPoint(forwardTransform, wind.centerM).map((value, axis) => value - wind!.centerM[axis]) as [number, number, number];
    } catch (error) { return { ok: false, reason: error instanceof Error ? error.message : "Invalid wind deformation" }; }
  }
  if (!validImage(previous.cloud) || !validImage(current.cloud)) return { ok: false, reason: "Invalid cloud sampling geometry" };
  const diameter = Math.min(...[previous.cloud, current.cloud].flatMap(image => image.shape.slice(0, 2).map((size, axis) => size * image.spacing[axis])));
  if (previous.wind?.ok && current.wind?.ok
      && Math.max(previous.wind.summary.residualMs, current.wind.summary.residualMs) * (current.timeS - previous.timeS) > diameter * 0.15) {
    return { ok: false, reason: "Unresolved wind variation exceeds the cloud preview range" };
  }
  const height = Math.min(previous.cloud.shape[2] * previous.cloud.spacing[2], current.cloud.shape[2] * current.cloud.spacing[2]);
  if (Math.hypot(displacement[0], displacement[1]) > Math.min(4000, diameter * 0.25) || Math.abs(displacement[2]) > Math.min(2000, height * 0.25)) {
    return { ok: false, reason: "Cloud displacement exceeds the supported preview range" };
  }
  const layers = [[previous.cloud, current.cloud], ...(previous.context && current.context ? [[previous.context, current.context]] : [])];
  if (layers.some(pair => pair.some(image => !validImage(image)))) return { ok: false, reason: "Invalid cloud sampling geometry" };
  let maximumDifference = 0, totalCloudSamples = 0, sampledPoints = 0;
  for (const [before, after] of layers) for (const forward of [true, false]) {
    const source = forward ? before : after, target = forward ? after : before;
    const counts = cloudMotionProbeShape(source.shape);
    const indices = counts.map((count, axis) => Array.from({ length: count }, (_, index) =>
      Math.round(index * (source.shape[axis] - 1) / Math.max(1, count - 1))));
    const sign = forward ? 1 : -1;
    let difference = 0, magnitude = 0, cloudSamples = 0, supported = 0;
    for (const z of indices[2]) for (const y of indices[1]) for (const x of indices[0]) {
      const index = x + source.shape[0] * (y + source.shape[1] * z);
      const value = source.values[index];
      const own = source.validity?.[index] === 0 || !Number.isFinite(value) || value < 0 ? null : value;
      const px = source.origin[0] + x * source.spacing[0], py = source.origin[1] + y * source.spacing[1], pz = source.origin[2] + z * source.spacing[2];
      const matrix = forward ? forwardTransform : backwardTransform;
      const tx = matrix ? matrix[0] * px + matrix[4] * py + matrix[8] * pz + matrix[12] : px + sign * displacement[0];
      const ty = matrix ? matrix[1] * px + matrix[5] * py + matrix[9] * pz + matrix[13] : py + sign * displacement[1];
      const tz = matrix ? matrix[2] * px + matrix[6] * py + matrix[10] * pz + matrix[14] : pz + sign * displacement[2];
      const other = sampleCloud(target, tx, ty, tz);
      sampledPoints++;
      if ((own ?? 0) <= 0.00005 && (other ?? 0) <= 0.00005) continue;
      cloudSamples++;
      if (Math.hypot(tx - px, ty - py) > Math.min(4000, diameter * 0.25) || Math.abs(tz - pz) > Math.min(2000, height * 0.25)) {
        return { ok: false, reason: "Cloud displacement exceeds the supported preview range" };
      }
      if (own == null || other == null) continue;
      supported++;
      difference += Math.abs(own - other);
      magnitude += own + other;
    }
    totalCloudSamples += cloudSamples;
    if (!cloudSamples) continue;
    if (supported / cloudSamples < 0.85 || magnitude <= 0) return { ok: false, reason: "Insufficient shared cloud support" };
    maximumDifference = Math.max(maximumDifference, difference / magnitude);
    if (maximumDifference > 0.35) return { ok: false, reason: "Cloud shape changes too much for this preview" };
  }
  if (!totalCloudSamples) return { ok: false, reason: "Insufficient shared cloud support" };
  return { ok: true, displacement, difference: maximumDifference, sampledPoints, method, ...(wind ? { wind } : {}) };
}

export function cloudMotionProbeShape(shape: readonly [number, number, number]): [number, number, number] {
  if (shape.length !== 3 || shape.some(size => !Number.isSafeInteger(size) || size <= 0)
      || !Number.isSafeInteger(shape[0] * shape[1] * shape[2])) throw new Error("Invalid cloud probe shape");
  const counts = [...shape] as [number, number, number];
  while (counts[0] * counts[1] * counts[2] > 16_384) {
    const axis = counts.indexOf(Math.max(...counts));
    counts[axis] = Math.ceil(counts[axis] / 2);
  }
  return counts;
}
