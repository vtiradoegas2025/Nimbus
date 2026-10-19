import type {
  ScalarPreparationRequest,
  VelocityDiagnosticPreparationRequest,
} from "./vtkPreparation";
import type { VortexSurfaceGeometry } from "./vortexContourProtocol";

export interface BrickedPreparedStats {
  min: number;
  max: number;
  p2: number;
  p98: number;
  positiveP98?: number;
  sourceNonFiniteCount: number;
}

export type BrickedScalarPreparationRequest =
  | ScalarPreparationRequest
  | VelocityDiagnosticPreparationRequest;

export type BrickedContourWorkerRequest =
  | {
      kind: "load";
      requestId: number;
      datasetIdentity: string;
      brickId: string;
      request: BrickedScalarPreparationRequest;
      terrainCellBounds?: [number, number, number, number, number, number];
    }
  | { kind: "extract"; requestId: number; datasetIdentity: string; brickIds: string[]; threshold: number }
  | { kind: "reset"; datasetIdentity: string }
  | { kind: "close" };

export type BrickedContourWorkerReply =
  | { kind: "ready" }
  | { kind: "loaded"; requestId: number; datasetIdentity: string; brickId: string; ok: true; bytes: number; stats: BrickedPreparedStats }
  | { kind: "loaded"; requestId: number; datasetIdentity: string; brickId: string; ok: false; error: string }
  | {
      kind: "result";
      requestId: number;
      datasetIdentity: string;
      ok: true;
      chunks: Array<{ brickId: string; geometry: VortexSurfaceGeometry | null }>;
      knownScratchBytes: number;
    }
  | { kind: "result"; requestId: number; datasetIdentity: string; ok: false; error: string };
