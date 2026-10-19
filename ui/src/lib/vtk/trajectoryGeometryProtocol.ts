import type { SceneScale } from "@/lib/plot/sceneScale";

export interface TrajectoryDataset {
  shape: [number, number, number];
  values: Float32Array;
}

export interface TrajectoryGeometryParameters {
  colorColumn: number;
  tubeRadius: number;
  scale: SceneScale;
  maximumParcels: number;
}

export const TRAJECTORY_TUBE_SIDES = 6;

export interface TrajectoryGeometryUpperBounds {
  selectedParcels: number;
  sourceSamples: number;
  retainedBytes: number;
  scratchBytes: number;
}

export function trajectoryGeometryUpperBounds(
  shape: readonly number[],
  maximumParcels: number,
): TrajectoryGeometryUpperBounds {
  if (shape.length !== 3 || !shape.every((value) => Number.isSafeInteger(value) && value >= 0)
      || shape[2]! < 6 || !Number.isSafeInteger(maximumParcels) || maximumParcels < 1) {
    throw new Error("Trajectory geometry bounds require a valid trajectory shape");
  }
  const [parcels, samples] = shape as readonly [number, number, number];
  const stride = parcels > maximumParcels ? Math.ceil(parcels / maximumParcels) : 1;
  const selectedParcels = parcels === 0 ? 0 : Math.ceil(parcels / stride);
  const sourceSamples = selectedParcels * samples;
  const maximumSegments = selectedParcels * Math.max(1, Math.floor((samples + 1) / 3));
  const retainedBytes = 216 * sourceSamples + 416 * maximumSegments;
  const scratchBytes = 236 * sourceSamples + 420 * maximumSegments + 4 * selectedParcels;
  if (![sourceSamples, retainedBytes, scratchBytes].every(Number.isSafeInteger)) {
    throw new Error("Trajectory geometry bounds exceed safe integer arithmetic");
  }
  return { selectedParcels, sourceSamples, retainedBytes, scratchBytes };
}

export interface TrajectoryTubeGeometry {
  points: Float32Array;
  lines: Uint32Array;
  polys: Uint32Array;
  strips: Uint32Array;
  normals: Float32Array;
  scalars: Float32Array;
}

export interface TrajectoryGeometryResult {
  geometry: TrajectoryTubeGeometry | null;
  min: number | null;
  max: number | null;
  parcels: number;
  sourceSamples: number;
  retainedBytes: number;
  knownScratchBytes: number;
  elapsedMs: number;
  timeRangeS?: [number, number] | null;
}

export type TrajectoryGeometryWorkerRequest =
  | ({
      kind: "extract";
      requestId: number;
      datasetId: number;
      dataset?: TrajectoryDataset;
    } & TrajectoryGeometryParameters)
  | { kind: "close" };

export type TrajectoryGeometryWorkerReply =
  | { kind: "ready" }
  | {
      kind: "result";
      requestId: number;
      datasetId: number;
      ok: true;
      result: TrajectoryGeometryResult;
    }
  | {
      kind: "result";
      requestId: number;
      datasetId: number;
      ok: false;
      error: string;
    };

export function trajectoryGeometryBuffers(
  geometry: TrajectoryTubeGeometry | null,
): ArrayBufferLike[] {
  if (!geometry) return [];
  return [...new Set<ArrayBufferLike>([
    geometry.points.buffer,
    geometry.lines.buffer,
    geometry.polys.buffer,
    geometry.strips.buffer,
    geometry.normals.buffer,
    geometry.scalars.buffer,
  ])];
}

export function trajectoryGeometryTransferables(
  result: TrajectoryGeometryResult,
): ArrayBuffer[] {
  return trajectoryGeometryBuffers(result.geometry)
    .filter((buffer): buffer is ArrayBuffer => buffer instanceof ArrayBuffer);
}
