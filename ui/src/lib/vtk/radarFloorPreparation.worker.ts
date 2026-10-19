import { floorStats, reduceFloor, terrainFloorMesh, terrainFloorSurface } from "@/lib/plot/radarFloor";
import {
  radarFloorPreparationBuffers,
  type RadarFloorPreparationWorkerReply,
  type RadarFloorPreparationWorkerRequest,
} from "./radarFloorPreparationProtocol";

const scope = self as unknown as {
  onmessage: ((event: { data: RadarFloorPreparationWorkerRequest }) => void) | null;
  postMessage: (message: RadarFloorPreparationWorkerReply, transfer: ArrayBuffer[]) => void;
  close: () => void;
};

scope.onmessage = (event) => {
  if (event.data.kind === "close") {
    scope.close();
    return;
  }
  const { requestId, values, shape, reduction, terrain } = event.data;
  try {
    const expected = shape[0] * shape[1] * shape[2];
    if (!Number.isSafeInteger(expected) || expected < 1 || values.length !== expected) {
      throw new Error("Radar floor source shape does not match its values");
    }
    const startedAt = performance.now();
    const reduced = reduceFloor(values, shape, reduction);
    const stats = floorStats(reduced);
    const terrainSurfaceM = stats && terrain
      ? terrainFloorSurface(
        terrain.centroids,
        terrain.volumesM3,
        [shape[0], shape[1]],
        terrain.sourceShape,
        terrain.strideXY,
        terrain.dxM,
        terrain.dyM,
      )
      : null;
    const terrainMesh = terrainSurfaceM
      ? terrainFloorMesh(terrainSurfaceM, [shape[0], shape[1]])
      : null;
    const result = {
      values: stats ? reduced : null,
      stats,
      terrainMesh,
      knownScratchBytes: reduced.byteLength + (terrainSurfaceM?.z.byteLength ?? 0),
      elapsedMs: performance.now() - startedAt,
    };
    const transfer = radarFloorPreparationBuffers(result)
      .filter((buffer): buffer is ArrayBuffer => buffer instanceof ArrayBuffer);
    scope.postMessage({ kind: "result", requestId, ok: true, result }, transfer);
  } catch (error) {
    scope.postMessage({
      kind: "result",
      requestId,
      ok: false,
      error: error instanceof Error ? error.message : String(error),
    }, []);
  }
};

scope.postMessage({ kind: "ready" }, []);

export {};
