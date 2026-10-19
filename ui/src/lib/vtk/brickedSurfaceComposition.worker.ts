import { mergeOwnedBrickSurfaces } from "./vtkBrickedContour";
import type {
  BrickedSurfaceCompositionWorkerReply,
  BrickedSurfaceCompositionWorkerRequest,
} from "./brickedSurfaceCompositionProtocol";

const scope = self as unknown as {
  onmessage: ((event: { data: BrickedSurfaceCompositionWorkerRequest }) => void) | null;
  postMessage: (reply: BrickedSurfaceCompositionWorkerReply, transfer?: ArrayBuffer[]) => void;
  close: () => void;
};

scope.onmessage = (event) => {
  const request = event.data;
  if (request.kind === "close") {
    scope.close();
    return;
  }
  try {
    const geometry = mergeOwnedBrickSurfaces(request.surfaces, request.coordinateTolerance);
    const transfer = geometry
      ? [geometry.points.buffer, geometry.polys.buffer, geometry.normals.buffer] as ArrayBuffer[]
      : [];
    scope.postMessage({
      kind: "result",
      requestId: request.requestId,
      identity: request.identity,
      ok: true,
      geometry,
    }, transfer);
  } catch (error) {
    scope.postMessage({
      kind: "result",
      requestId: request.requestId,
      identity: request.identity,
      ok: false,
      error: error instanceof Error ? error.message : String(error),
    });
  }
};

scope.postMessage({ kind: "ready" });

export {};
