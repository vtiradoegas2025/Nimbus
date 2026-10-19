import type {
  MesocycloneEnvelopeAssessment,
  TornadoVortexAssessment,
} from "@/lib/plot/vortexCore";
import type { TornadicCondensationAssessment } from "@/lib/plot/tornadicCondensation";
import type { PreparedVtkImage, ScalarPreparationRequest } from "./vtkPreparation";

export type PreparedScalarMetadata = Omit<PreparedVtkImage, "values">;

export interface VortexContourImage {
  shape: [number, number, number];
  spacing: [number, number, number];
  origin: [number, number, number];
  values: Float32Array;
  validity?: Uint8Array;
}

export interface VortexContourThresholds {
  core: number | null;
  opposite: number | null;
  mesocyclone: number | null;
}

export type VortexExtractionInputMode = "stop" | "hold" | "extract";

/** A missing prepared image while the layer remains enabled is a frame-change
 * handoff, not a request to destroy the warm worker and its VTK pipeline. */
export function vortexExtractionInputMode(
  enabled: boolean,
  hasPreparedImage: boolean,
  hasContour: boolean,
): VortexExtractionInputMode {
  if (!enabled || !hasContour) return "stop";
  return hasPreparedImage ? "extract" : "hold";
}

export interface VortexSurfaceGeometry {
  points: Float32Array;
  polys: Uint32Array;
  normals: Float32Array;
  bounds: [number, number, number, number, number, number] | null;
  triangles: number;
  bytes: number;
}

export interface VortexContourUsage {
  triangles: number;
  bytes: number;
}

export interface VortexContourAdmission {
  maximumTriangles: number;
  maximumOutputBytes: number;
  maximumScratchBytes: number;
  crossingCells: number;
}

export interface VortexContourResult {
  core: VortexSurfaceGeometry | null;
  opposite: VortexSurfaceGeometry | null;
  /** Number of nontrivial connected same-sign components retained in `core`. */
  coreComponentCount: number;
  /** Number of nontrivial opposite-sign companions retained in `opposite`. */
  oppositeComponentCount: number;
  mesocyclone: VortexSurfaceGeometry | null;
  mesocycloneAssessment: MesocycloneEnvelopeAssessment | null;
  vortexAssessment: TornadoVortexAssessment | null;
  /** Solved qc+qi retained only near this connected vortex. */
  condensation: VortexContourImage | null;
  condensationBounds: [number, number, number, number, number, number] | null;
  condensationAssessment: TornadicCondensationAssessment | null;
  /** Includes the raw mesocyclone envelope, matching the retained-output guard. */
  usage: VortexContourUsage;
  /** Exact exposed typed-array scratch observed in the worker. vtk.js-native
   * and driver allocations that expose no backing store are not included. */
  knownScratchBytes: number;
  elapsedMs: number;
}

export type VortexContourWorkerRequest =
  | {
      kind: "extract";
      requestId: number;
      datasetId: number;
      image?: VortexContourImage;
      preparation?: ScalarPreparationRequest;
      condensatePreparation?: ScalarPreparationRequest;
      reportProgress?: boolean;
      thresholds: VortexContourThresholds;
      /** Solved low-level signed extremum used to select one physical hierarchy. */
      anchor?: [number, number, number] | null;
      /** Co-registered solved qc+qi lattice; never a procedural funnel. */
      condensateImage?: VortexContourImage;
      condensateFloor?: number | null;
    }
  | {
      kind: "continue";
      requestId: number;
      datasetId: number;
      admission: VortexContourAdmission;
    }
  | {
      kind: "prepare";
      requestId: number;
      datasetId: number;
      request: ScalarPreparationRequest;
    }
  | { kind: "release"; requestId: number }
  | { kind: "close" };

export type VortexContourWorkerReply =
  | { kind: "ready" }
  | { kind: "released"; requestId: number }
  | {
      kind: "progress";
      requestId: number;
      datasetId: number;
      stage: "received" | "primary-prepared" | "condensate-prepared" | "extracting"
        | "core-extracted" | "opposite-extracted" | "mesocyclone-extracted"
        | "connectivity-complete" | "condensation-complete" | "result-built"
        | "scratch-complete";
    }
  | {
      kind: "admission";
      requestId: number;
      datasetId: number;
      admission: VortexContourAdmission;
    }
  | {
      kind: "prepared";
      requestId: number;
      datasetId: number;
      ok: true;
      metadata: PreparedScalarMetadata;
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
      result: VortexContourResult;
    }
  | {
      kind: "result";
      requestId: number;
      datasetId: number;
      ok: false;
      error: string;
    };

export function vortexContourTransferables(
  result: VortexContourResult,
): ArrayBuffer[] {
  const buffers: ArrayBuffer[] = [];
  for (const surface of [result.core, result.opposite, result.mesocyclone]) {
    if (!surface) continue;
    buffers.push(
      surface.points.buffer as ArrayBuffer,
      surface.polys.buffer as ArrayBuffer,
      surface.normals.buffer as ArrayBuffer,
    );
  }
  if (result.condensation) {
    buffers.push(result.condensation.values.buffer as ArrayBuffer);
    if (result.condensation.validity) buffers.push(result.condensation.validity.buffer as ArrayBuffer);
  }
  return buffers;
}
