import type { PreparedVtkImage } from "./vtkPreparation";
import { scalarIntervalHasOpacity } from "@/lib/plot/fieldBrickHierarchy";

export interface VolumeBackground {
  heights: number[];
  values: number[];
  threshold: number;
  positiveOnly: boolean;
  interior?: [number, number, number, number];
  domain?: [number, number, number, number];
  featherM?: [number, number];
}

export function backgroundInterior(profile: VolumeBackground, image: Pick<PreparedVtkImage, "spacing">): VolumeBackground["interior"] {
  if (!profile.interior || !profile.domain) return profile.interior;
  return profile.interior.map((value, face) => {
    const axis = Math.floor(face / 2);
    const margin = Math.min(image.spacing[axis] * 2, (profile.domain![axis * 2 + 1] - profile.domain![axis * 2]) * 0.2);
    return face % 2 ? Math.min(value, profile.domain![face] - margin) : Math.max(value, profile.domain![face] + margin);
  }) as VolumeBackground["interior"];
}

export function usesVolumeBackground(field: string | null): boolean {
  return ["vorticity_magnitude", "horizontal_wind_speed", "wind_speed", "u", "v", "u_r", "u_theta"].includes(field ?? "");
}

export function backgroundAffectsOpacity(profile: VolumeBackground | null, points: readonly (readonly [number, number])[]): boolean {
  if (!profile || !points.length) return false;
  const low = Math.min(...profile.values) - profile.threshold * 2;
  const high = Math.max(...profile.values) + profile.threshold * 2;
  return scalarIntervalHasOpacity(profile.positiveOnly ? points[0][0] : low, high, points);
}

export function backgroundAt(profile: VolumeBackground, height: number): number {
  const { heights, values } = profile;
  let low = 0, high = heights.length - 1;
  if (height <= heights[0]) return values[0];
  if (height >= heights[high]) return values[high];
  while (high - low > 1) {
    const middle = (low + high) >>> 1;
    if (heights[middle] <= height) low = middle; else high = middle;
  }
  const fraction = (height - heights[low]) / (heights[high] - heights[low]);
  return values[low] + fraction * (values[high] - values[low]);
}

export function volumeBackgroundProfile(image: PreparedVtkImage, positiveOnly: boolean,
  interior?: VolumeBackground["interior"], featherM?: VolumeBackground["featherM"]): VolumeBackground {
  const [nx, ny, nz] = image.shape;
  const heights: number[] = [], values: number[] = [];
  let magnitude = 0, deviation = 0;
  for (let z = 0; z < nz; z++) {
    const samples: number[] = [];
    for (let index = z * nx * ny; index < (z + 1) * nx * ny; index++) {
      const x = image.origin[0] + (index % nx) * image.spacing[0];
      const y = image.origin[1] + (Math.floor(index / nx) % ny) * image.spacing[1];
      if (interior && (x <= interior[0] || x >= interior[1] || y <= interior[2] || y >= interior[3])) continue;
      if (image.validity?.[index] === 0 || !Number.isFinite(image.values[index])) continue;
      samples.push(image.values[index]); magnitude = Math.max(magnitude, Math.abs(image.values[index]));
    }
    if (!samples.length) continue;
    samples.sort((a, b) => a - b);
    const middle = Math.floor(samples.length / 2);
    const median = samples.length % 2 ? samples[middle] : (samples[middle - 1] + samples[middle]) / 2;
    heights.push(image.origin[2] + z * image.spacing[2]); values.push(median);
    for (const value of samples) deviation = Math.max(deviation, positiveOnly ? value - median : Math.abs(value - median));
  }
  if (!heights.length) throw new Error("No finite samples are available to estimate the horizontal background");
  return { heights, values, positiveOnly, interior, featherM,
    threshold: Math.max(deviation * 0.08, magnitude * 1e-6, 1e-12) };
}

export function focusedVolume(image: PreparedVtkImage, profile: VolumeBackground, crop: boolean): PreparedVtkImage {
  const [nx, ny, nz] = image.shape;
  const low = [nx, ny, nz], high = [-1, -1, -1];
  const interior = backgroundInterior(profile, image);
  for (let z = 0; z < nz; z++) {
    const background = backgroundAt(profile, image.origin[2] + z * image.spacing[2]);
    for (let y = 0; y < ny; y++) for (let x = 0; x < nx; x++) {
      const px = image.origin[0] + x * image.spacing[0], py = image.origin[1] + y * image.spacing[1];
      if (interior && (px <= interior[0] || px >= interior[1] || py <= interior[2] || py >= interior[3])) continue;
      const index = x + nx * (y + ny * z);
      if (image.validity?.[index] === 0) continue;
      const delta = image.values[index] - background;
      if ((profile.positiveOnly ? delta : Math.abs(delta)) <= profile.threshold) continue;
      if (!Number.isFinite(delta)) continue;
      low[0] = Math.min(low[0], x); low[1] = Math.min(low[1], y); low[2] = Math.min(low[2], z);
      high[0] = Math.max(high[0], x); high[1] = Math.max(high[1], y); high[2] = Math.max(high[2], z);
    }
  }
  if (high[0] < 0) {
    const shape = image.shape.map(count => Math.min(2, count)) as [number, number, number];
    const count = shape[0] * shape[1] * shape[2];
    return crop ? { ...image, focusEmpty: true, focusCropped: true,
      shape, values: new Float32Array(count), validity: new Uint8Array(count),
      structuralSupportBounds: null, absoluteFocusBounds: null } : { ...image, focusEmpty: true };
  }
  if (!crop) return { ...image, focusEmpty: false };
  for (let axis = 0; axis < 3; axis++) {
    low[axis] = Math.max(0, low[axis] - 2); high[axis] = Math.min(image.shape[axis] - 1, high[axis] + 2);
  }
  const shape = low.map((value, axis) => high[axis] - value + 1) as [number, number, number];
  if (shape.every((value, axis) => value === image.shape[axis])) return image;
  const values = new Float32Array(shape[0] * shape[1] * shape[2]);
  const validity = image.validity ? new Uint8Array(values.length) : undefined;
  for (let z = 0; z < shape[2]; z++) for (let y = 0; y < shape[1]; y++) {
    const start = low[0] + nx * (y + low[1] + ny * (z + low[2]));
    const target = shape[0] * (y + shape[1] * z);
    values.set(image.values.subarray(start, start + shape[0]), target);
    validity?.set(image.validity!.subarray(start, start + shape[0]), target);
  }
  const origin = image.origin.map((value, axis) => value + low[axis] * image.spacing[axis]) as [number, number, number];
  return { ...image, values, validity, shape, origin, focusCropped: true,
    structuralSupportBounds: origin.flatMap((value, axis) => [value, value + (shape[axis] - 1) * image.spacing[axis]]) as PreparedVtkImage["structuralSupportBounds"] };
}
