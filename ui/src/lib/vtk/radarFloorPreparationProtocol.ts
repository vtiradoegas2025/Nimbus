import type { SignedFieldStats } from "@/lib/plot/coherentStructures";
import type { FloorReduction, TerrainFloorMesh } from "@/lib/plot/radarFloor";

export interface RadarFloorTerrainInput {
  centroids: { x: Float64Array; y: Float64Array; z: Float64Array };
  volumesM3: Float64Array;
  sourceShape: [number, number];
  strideXY: number;
  dxM: number;
  dyM: number;
}

export interface RadarFloorPreparationResult {
  values: Float32Array | null;
  stats: SignedFieldStats | null;
  terrainMesh: TerrainFloorMesh | null;
  knownScratchBytes: number;
  elapsedMs: number;
}

export type RadarFloorPreparationWorkerRequest =
  | {
      kind: "prepare";
      requestId: number;
      values: Float32Array;
      shape: [number, number, number];
      reduction: FloorReduction;
      terrain: RadarFloorTerrainInput | null;
    }
  | { kind: "close" };

export type RadarFloorPreparationWorkerReply =
  | { kind: "ready" }
  | {
      kind: "result";
      requestId: number;
      ok: true;
      result: RadarFloorPreparationResult;
    }
  | {
      kind: "result";
      requestId: number;
      ok: false;
      error: string;
    };

export function radarFloorPreparationBuffers(
  result: RadarFloorPreparationResult,
): ArrayBufferLike[] {
  return [...new Set<ArrayBufferLike>([
    ...(result.values ? [result.values.buffer] : []),
    ...(result.terrainMesh ? [
      result.terrainMesh.points.buffer,
      result.terrainMesh.polys.buffer,
      result.terrainMesh.tcoords.buffer,
    ] : []),
  ])];
}
