import type { NestedSurfaceBounds } from "./nestedSurfaceOwnership";
import type { VortexSurfaceGeometry } from "./vortexContourProtocol";

export interface NestedSurfaceCompositionInput {
  geometries: Array<VortexSurfaceGeometry | null>;
  childBounds: Array<NestedSurfaceBounds | null>;
}

export interface NestedSurfaceCompositionUpperBound {
  outputTriangles: number;
  outputBytes: number;
  scratchBytes: number;
}

/** One triangle clipped against six box planes can emit at most 48 exterior
 * triangles; boundary pairing contributes at most 12, and the surviving leaf
 * contributes one. All composition output uses three unshared vertices. */
export function nestedSurfaceCompositionUpperBound(
  primary: NestedSurfaceCompositionInput,
  opposite: NestedSurfaceCompositionInput,
): NestedSurfaceCompositionUpperBound {
  const geometries = [...primary.geometries, ...opposite.geometries]
    .filter((geometry): geometry is VortexSurfaceGeometry => geometry != null);
  const inputTriangles = geometries.reduce((sum, geometry) => sum + geometry.triangles, 0);
  const inputBytes = geometries.reduce((sum, geometry) => sum + geometry.bytes, 0);
  const outputTriangles = Math.max(1, inputTriangles * 61);
  const outputBytes = outputTriangles * 88;
  const scratchBytes = inputBytes + outputBytes;
  if (![outputTriangles, outputBytes, scratchBytes].every(Number.isSafeInteger)) {
    throw new Error("Nested surface topology bound exceeds safe integer arithmetic");
  }
  return { outputTriangles, outputBytes, scratchBytes };
}

export type NestedSurfaceCompositionWorkerRequest =
  | {
      kind: "begin";
      requestId: number;
      identity: string;
      childBounds: Array<NestedSurfaceBounds | null>;
      layerCount: number;
      maximumOutputBytes: number;
      maximumTriangles: number;
    }
  | {
      kind: "layer";
      requestId: number;
      identity: string;
      layerIndex: number;
      primary: VortexSurfaceGeometry | null;
      opposite: VortexSurfaceGeometry | null;
    }
  | { kind: "finish"; requestId: number; identity: string }
  | { kind: "close" };

export type NestedSurfaceCompositionWorkerReply =
  | { kind: "ready" }
  | { kind: "ack"; requestId: number; identity: string; nextLayerIndex: number }
  | {
      kind: "result";
      requestId: number;
      identity: string;
      ok: true;
      primary: { geometry: VortexSurfaceGeometry | null; transitionTriangles: number };
      opposite: { geometry: VortexSurfaceGeometry | null; transitionTriangles: number };
      knownScratchBytes: number;
    }
  | { kind: "result"; requestId: number; identity: string; ok: false; error: string };
