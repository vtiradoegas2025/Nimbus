import type { OwnedBrickSurface } from "./vtkBrickedContour";
import type { VortexSurfaceGeometry } from "./vortexContourProtocol";

export type BrickedSurfaceCompositionWorkerRequest =
  | {
      kind: "compose";
      requestId: number;
      identity: string;
      surfaces: OwnedBrickSurface[];
      coordinateTolerance: number;
    }
  | { kind: "close" };

export type BrickedSurfaceCompositionWorkerReply =
  | { kind: "ready" }
  | {
      kind: "result";
      requestId: number;
      identity: string;
      ok: true;
      geometry: VortexSurfaceGeometry | null;
    }
  | {
      kind: "result";
      requestId: number;
      identity: string;
      ok: false;
      error: string;
    };
