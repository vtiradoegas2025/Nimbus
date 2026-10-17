import type { CartesianVoxelGrid } from "@/lib/science/cartesianInterp";
import { presentationDetailWeight } from "./renderFidelity";

export interface CloudPresentationDetailMetadata {
  applied: boolean;
  seed: number;
  weight: number;
  baseScaleM: number;
  coherence: "storm-relative" | "resolved-flow";
  /** Condensate-weighted resolved-cloud center used only as the noise frame. */
  anchorM: [number, number, number];
  /** Condensate-weighted translation removed before local flow deformation. */
  meanFlowMps?: [number, number, number];
  /** Length of each smoothly cross-faded advection epoch. */
  advectionWindowS?: number;
  supportConstrained: true;
  /** Preserves the finite scalar sum on the display lattice. Air-density
   * weighting is unavailable; per-source-cell normalization creates seams. */
  scalarSumPreserved: true;
}

export interface CloudPresentationDetailResult {
  grid: CartesianVoxelGrid;
  metadata: CloudPresentationDetailMetadata | null;
}

export const MAX_CLOUD_PRESENTATION_DETAIL_VOXELS = 2_000_000;

export function cloudPresentationDetailEligible(
  source: CartesianVoxelGrid,
  reconstructed: CartesianVoxelGrid,
  factors: [number, number, number],
): boolean {
  const voxels = reconstructed.shape[0] * reconstructed.shape[1] * reconstructed.shape[2];
  return presentationDetailWeight(source.spacing) > 0
    && voxels <= MAX_CLOUD_PRESENTATION_DETAIL_VOXELS
    && factors.some((factor) => factor > 1);
}

/**
 * Add deterministic display-only density variation inside resolved cloud support.
 * A single scalar-sum normalization keeps the noise field continuous;
 * normalizing each source cell independently creates block-aligned texture seams.
 * Presentation mode only.
 */
export function addCloudPresentationDetail(
  source: CartesianVoxelGrid,
  reconstructed: CartesianVoxelGrid,
  factors: [number, number, number],
  seed: number,
  flow?: { grid: CartesianVoxelGrid; simulationTimeS: number },
): CloudPresentationDetailResult {
  const weight = presentationDetailWeight(source.spacing);
  if (!cloudPresentationDetailEligible(source, reconstructed, factors)) {
    return { grid: reconstructed, metadata: null };
  }

  const data = new Float32Array(reconstructed.data.length);
  const [nx, ny, nz] = reconstructed.shape;
  const [dx, dy, dz] = reconstructed.spacing;
  const [ox, oy, oz] = reconstructed.origin;
  const anchorM = cloudCondensateCentroid(source);
  const meanFlowMps = flow ? cloudCondensateMeanVelocity(source, flow.grid) : null;
  // Base scale > reconstructed Nyquist: avoids crystalline-threshold noise on 250 m lattices.
  const baseScaleM = Math.max(...source.spacing) * 1.6;
  // Global normalization no longer damps each source cell independently, so a
  // lower modulation reaches the paired fine-grid variability target without
  // giving the cloud an over-cranked, pebbled texture.
  const detailStrength = 0.6 * weight;

  let baseSum = 0;
  let candidateSum = 0;
  for (let x = 0; x < nx; x++) {
    for (let y = 0; y < ny; y++) {
      const base = (x * ny + y) * nz;
      for (let z = 0; z < nz; z++) {
        const index = base + z;
        const value = reconstructed.data[index];
        if (!Number.isFinite(value)) { data[index] = Number.NaN; continue; }
        if (!(value > 0)) { data[index] = value; continue; }
        const noise = cloudDetailNoise(
          [ox + x * dx, oy + y * dy, oz + z * dz],
          anchorM,
          baseScaleM,
          seed,
          flow,
          meanFlowMps,
        );
        const candidate = value * Math.exp(detailStrength * noise);
        data[index] = candidate;
        baseSum += value;
        candidateSum += candidate;
      }
    }
  }
  if (candidateSum > 0 && baseSum > 0) {
    const normalization = baseSum / candidateSum;
    for (let index = 0; index < data.length; index += 1) {
      if (data[index] > 0 && Number.isFinite(data[index])) data[index] *= normalization;
    }
  }

  return {
    grid: { ...reconstructed, data },
    metadata: {
      applied: true,
      seed: seed >>> 0,
      weight,
      baseScaleM,
      coherence: flow && meanFlowMps ? "resolved-flow" : "storm-relative",
      anchorM,
      meanFlowMps: meanFlowMps ?? undefined,
      advectionWindowS: flow && meanFlowMps ? FLOW_ADVECTION_WINDOW_S : undefined,
      supportConstrained: true,
      scalarSumPreserved: true,
    },
  };
}

const FLOW_ADVECTION_WINDOW_S = 60;

/** Two overlapping noise epochs for smooth, frame-order-independent advection. */
function cloudDetailNoise(
  positionM: [number, number, number],
  anchorM: [number, number, number],
  scaleM: number,
  seed: number,
  flow: { grid: CartesianVoxelGrid; simulationTimeS: number } | undefined,
  meanFlowMps: [number, number, number] | null,
): number {
  if (!flow || !meanFlowMps || !Number.isFinite(flow.simulationTimeS)) {
    return cloudFbm(
      (positionM[0] - anchorM[0]) / scaleM,
      (positionM[1] - anchorM[1]) / scaleM,
      (positionM[2] - anchorM[2]) / scaleM,
      seed,
    );
  }
  const timeS = Math.max(0, flow.simulationTimeS);
  const epoch = Math.floor(timeS / FLOW_ADVECTION_WINDOW_S);
  const phaseS = timeS - epoch * FLOW_ADVECTION_WINDOW_S;
  const blend = fade(phaseS / FLOW_ADVECTION_WINDOW_S);
  const velocity = sampleVectorTrilinear(flow.grid, positionM);
  if (!velocity) return cloudFbm(
    (positionM[0] - anchorM[0]) / scaleM,
    (positionM[1] - anchorM[1]) / scaleM,
    (positionM[2] - anchorM[2]) / scaleM, seed,
  );
  const anomaly: [number, number, number] = [
    velocity[0] - meanFlowMps[0],
    velocity[1] - meanFlowMps[1],
    velocity[2] - meanFlowMps[2],
  ];
  if (Math.hypot(...anomaly) * FLOW_ADVECTION_WINDOW_S < scaleM * 1e-9) {
    return cloudFbm(
      (positionM[0] - anchorM[0]) / scaleM,
      (positionM[1] - anchorM[1]) / scaleM,
      (positionM[2] - anchorM[2]) / scaleM,
      seed,
    );
  }
  const sampleEpoch = (noiseEpoch: number, ageS: number) => cloudFbm(
    (positionM[0] - anchorM[0] - anomaly[0] * ageS) / scaleM,
    (positionM[1] - anchorM[1] - anomaly[1] * ageS) / scaleM,
    (positionM[2] - anchorM[2] - anomaly[2] * ageS) / scaleM,
    seed + Math.imul(noiseEpoch, 0x9e3779b9),
  );
  const previous = sampleEpoch(epoch - 1, phaseS + FLOW_ADVECTION_WINDOW_S);
  const current = sampleEpoch(epoch, phaseS);
  return lerp(previous, current, blend);
}

function cloudCondensateMeanVelocity(
  source: CartesianVoxelGrid,
  flow: CartesianVoxelGrid,
): [number, number, number] | null {
  const [nx, ny, nz] = source.shape;
  const [dx, dy, dz] = source.spacing;
  const [ox, oy, oz] = source.origin;
  const sum = [0, 0, 0];
  let mass = 0;
  let index = 0;
  for (let x = 0; x < nx; x++) {
    for (let y = 0; y < ny; y++) {
      for (let z = 0; z < nz; z++, index++) {
        const condensate = source.data[index];
        if (!(condensate > 0) || !Number.isFinite(condensate)) continue;
        const velocity = sampleVectorTrilinear(flow, [ox + x * dx, oy + y * dy, oz + z * dz]);
        if (!velocity) continue;
        mass += condensate;
        for (let component = 0; component < 3; component++) {
          sum[component] += condensate * velocity[component];
        }
      }
    }
  }
  return mass > 0
    ? [sum[0] / mass, sum[1] / mass, sum[2] / mass]
    : null;
}

function sampleVectorTrilinear(
  grid: CartesianVoxelGrid,
  positionM: readonly [number, number, number],
): [number, number, number] | null {
  if (positionM.some((value, axis) => !Number.isFinite(value)
      || value < grid.origin[axis]
      || value > grid.origin[axis] + (grid.shape[axis] - 1) * grid.spacing[axis])) return null;
  const coordinate = positionM.map((value, axis) => {
    const spacing = grid.spacing[axis];
    const upper = Math.max(0, grid.shape[axis] - 1);
    return spacing > 0
      ? Math.min(upper, Math.max(0, (value - grid.origin[axis]) / spacing))
      : 0;
  });
  const lo = coordinate.map(Math.floor);
  const hi = coordinate.map((value, axis) => Math.min(grid.shape[axis] - 1, Math.ceil(value)));
  const t = coordinate.map((value, axis) => value - lo[axis]);
  const result: [number, number, number] = [0, 0, 0];
  for (let ix = 0; ix < 2; ix++) for (let iy = 0; iy < 2; iy++) for (let iz = 0; iz < 2; iz++) {
    const x = ix ? hi[0] : lo[0];
    const y = iy ? hi[1] : lo[1];
    const z = iz ? hi[2] : lo[2];
    const weight = (ix ? t[0] : 1 - t[0]) *
      (iy ? t[1] : 1 - t[1]) *
      (iz ? t[2] : 1 - t[2]);
    if (weight === 0) continue;
    const base = ((x * grid.shape[1] + y) * grid.shape[2] + z) * 3;
    for (let component = 0; component < 3; component++) {
      const value = grid.data[base + component];
      if (!Number.isFinite(value)) return null;
      result[component] += weight * value;
    }
  }
  return result;
}

/** Condensate-mass centroid of the resolved source grid; used as noise origin for translational coherence. */
export function cloudCondensateCentroid(
  source: CartesianVoxelGrid,
): [number, number, number] {
  const [nx, ny, nz] = source.shape;
  const [dx, dy, dz] = source.spacing;
  const [ox, oy, oz] = source.origin;
  let mass = 0;
  let xMoment = 0;
  let yMoment = 0;
  let zMoment = 0;
  let index = 0;
  for (let x = 0; x < nx; x++) {
    for (let y = 0; y < ny; y++) {
      for (let z = 0; z < nz; z++, index++) {
        const value = source.data[index];
        if (!(value > 0) || !Number.isFinite(value)) continue;
        mass += value;
        xMoment += value * (ox + x * dx);
        yMoment += value * (oy + y * dy);
        zMoment += value * (oz + z * dz);
      }
    }
  }
  if (mass > 0) return [xMoment / mass, yMoment / mass, zMoment / mass];
  return [
    ox + 0.5 * Math.max(0, nx - 1) * dx,
    oy + 0.5 * Math.max(0, ny - 1) * dy,
    oz + 0.5 * Math.max(0, nz - 1) * dz,
  ];
}

function cloudFbm(x: number, y: number, z: number, seed: number): number {
  let value = 0;
  let amplitude = 0.58;
  let frequency = 1;
  let normalization = 0;
  for (let octave = 0; octave < 2; octave++) {
    value += amplitude * valueNoise3(x * frequency, y * frequency, z * frequency, seed + octave * 0x9e3779b9);
    normalization += amplitude;
    amplitude *= 0.5;
    frequency *= 2;
  }
  return normalization > 0 ? value / normalization : 0;
}

function valueNoise3(x: number, y: number, z: number, seed: number): number {
  const x0 = Math.floor(x);
  const y0 = Math.floor(y);
  const z0 = Math.floor(z);
  const tx = fade(x - x0);
  const ty = fade(y - y0);
  const tz = fade(z - z0);
  const c000 = latticeNoise(x0, y0, z0, seed);
  const c100 = latticeNoise(x0 + 1, y0, z0, seed);
  const c010 = latticeNoise(x0, y0 + 1, z0, seed);
  const c110 = latticeNoise(x0 + 1, y0 + 1, z0, seed);
  const c001 = latticeNoise(x0, y0, z0 + 1, seed);
  const c101 = latticeNoise(x0 + 1, y0, z0 + 1, seed);
  const c011 = latticeNoise(x0, y0 + 1, z0 + 1, seed);
  const c111 = latticeNoise(x0 + 1, y0 + 1, z0 + 1, seed);
  const x00 = lerp(c000, c100, tx);
  const x10 = lerp(c010, c110, tx);
  const x01 = lerp(c001, c101, tx);
  const x11 = lerp(c011, c111, tx);
  return lerp(lerp(x00, x10, ty), lerp(x01, x11, ty), tz);
}

function latticeNoise(x: number, y: number, z: number, seed: number): number {
  let h = (seed ^ Math.imul(x, 0x1f123bb5) ^ Math.imul(y, 0x5f356495) ^ Math.imul(z, 0x6c8e9cf5)) >>> 0;
  h ^= h >>> 16;
  h = Math.imul(h, 0x7feb352d) >>> 0;
  h ^= h >>> 15;
  h = Math.imul(h, 0x846ca68b) >>> 0;
  h ^= h >>> 16;
  return (h / 0xffffffff) * 2 - 1;
}

function fade(value: number): number {
  return value * value * (3 - 2 * value);
}

function lerp(a: number, b: number, weight: number): number {
  return a + (b - a) * weight;
}
