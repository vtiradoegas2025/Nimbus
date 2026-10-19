import type { SceneScale } from "@/lib/plot/sceneScale";
import type { BrickBoundaryContacts } from "@/lib/plot/fieldBrickHierarchy";
import type { PreparedVectorImage, VectorPreparationRequest } from "./vtkPreparation";

export type PreparedVectorMetadata = Omit<PreparedVectorImage, "values">;

export interface StreamlineImage {
  shape: [number, number, number];
  spacing: [number, number, number];
  origin: [number, number, number];
  values: Float32Array;
}

export interface StreamlineGeometry {
  points: Float32Array;
  lines: Uint32Array;
  polys: Uint32Array;
  strips: Uint32Array;
  normals: Float32Array;
  scalars: Float32Array | null;
}

export interface StreamlineDirectionGeometry {
  path: StreamlineGeometry | null;
  arrows: StreamlineGeometry | null;
}

export interface StreamlineExtractionResult {
  forward: StreamlineDirectionGeometry;
  backward: StreamlineDirectionGeometry;
  representation: "tubes" | "gpu-lines";
  integratedPoints: number;
  boundaryContacts: BrickBoundaryContacts;
  /** Exact exposed typed-array scratch observed in the worker. */
  knownScratchBytes: number;
  elapsedMs: number;
}

export interface StreamlineExtractionParameters {
  seeds: Float32Array;
  integrationStep: number;
  maxSteps: number;
  tubeRadius: number;
  scale: SceneScale;
}

export type StreamlineWorkerRequest =
  | ({
      kind: "extract";
      requestId: number;
      datasetId: number;
      image?: StreamlineImage;
      preparation?: VectorPreparationRequest;
    } & StreamlineExtractionParameters)
  | {
      kind: "prepare";
      requestId: number;
      datasetId: number;
      request: VectorPreparationRequest;
    }
  | { kind: "close" };

export type StreamlineWorkerReply =
  | { kind: "ready" }
  | {
      kind: "prepared";
      requestId: number;
      datasetId: number;
      ok: true;
      metadata: PreparedVectorMetadata;
      bytes: number;
    }
  | {
      kind: "prepared";
      requestId: number;
      datasetId: number;
      ok: false;
      error: string;
    }
  | {
      kind: "result";
      requestId: number;
      datasetId: number;
      ok: true;
      result: StreamlineExtractionResult;
    }
  | {
      kind: "result";
      requestId: number;
      datasetId: number;
      ok: false;
      error: string;
    };

export function streamlineResultTransferables(result: StreamlineExtractionResult): ArrayBuffer[] {
  const buffers: ArrayBuffer[] = [];
  for (const geometry of [
    result.forward.path,
    result.forward.arrows,
    result.backward.path,
    result.backward.arrows,
  ]) {
    if (!geometry) continue;
    buffers.push(
      geometry.points.buffer as ArrayBuffer,
      geometry.lines.buffer as ArrayBuffer,
      geometry.polys.buffer as ArrayBuffer,
      geometry.strips.buffer as ArrayBuffer,
      geometry.normals.buffer as ArrayBuffer,
    );
    if (geometry.scalars) buffers.push(geometry.scalars.buffer as ArrayBuffer);
  }
  return buffers;
}
