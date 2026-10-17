import {
  applyPreparedGlobalHorizontalLevelAnomaly,
  excludePreparedScalarBounds,
  prepareGlobalAnomalyBrickBatch,
  prepareVtkRequest,
  vtkPreparationTransferables,
} from "./vtkPreparation";
import type { VolumeBrickWorkerReply, VolumeBrickWorkerRequest } from "./volumeBrickProtocol";

const scope = self as unknown as {
  onmessage: ((event: { data: VolumeBrickWorkerRequest }) => void) | null;
  postMessage: (reply: VolumeBrickWorkerReply, transfer?: ArrayBuffer[]) => void;
  close: () => void;
};

scope.onmessage = (event) => {
  const request = event.data;
  if (request.kind === "close") {
    scope.postMessage({ kind: "closed" });
    scope.close();
    return;
  }
  const replyKind = request.kind === "prepare-anomaly-batch" ? "prepared-batch" : "prepared";
  try {
    if (request.kind === "prepare-anomaly-batch") {
      const assembled = prepareGlobalAnomalyBrickBatch(request.requests);
      if (assembled) {
        const transfer = assembled.flatMap((item) => vtkPreparationTransferables(item));
        scope.postMessage({ kind: "prepared-batch", requestId: request.requestId, ok: true,
          prepared: assembled, bytes: transfer.reduce((sum, buffer) => sum + buffer.byteLength, 0) }, transfer);
        return;
      }
      const raw = request.requests.map((item) => {
        const prepared = prepareVtkRequest({
          ...item,
          scalarTransform: "none",
          excludePhysicalBoundsM: undefined,
        });
        if (!("values" in prepared)) throw new Error("Volume brick batch did not produce scalar images");
        return prepared;
      });
      const prepared = applyPreparedGlobalHorizontalLevelAnomaly(raw).map((item, index) =>
        excludePreparedScalarBounds(item, request.requests[index].excludePhysicalBoundsM));
      const transfer = prepared.flatMap((item) => vtkPreparationTransferables(item));
      scope.postMessage({
        kind: "prepared-batch",
        requestId: request.requestId,
        ok: true,
        prepared,
        bytes: transfer.reduce((sum, buffer) => sum + buffer.byteLength, 0),
      }, transfer);
      return;
    }
    const prepared = prepareVtkRequest(request.request);
    if (!("values" in prepared)) throw new Error("Volume brick preparation did not produce a scalar image");
    const transfer = vtkPreparationTransferables(prepared);
    scope.postMessage({
      kind: "prepared",
      requestId: request.requestId,
      ok: true,
      prepared,
      bytes: transfer.reduce((sum, buffer) => sum + buffer.byteLength, 0),
    }, transfer);
  } catch (error) {
    scope.postMessage({
      kind: replyKind,
      requestId: request.requestId,
      ok: false,
      error: error instanceof Error ? error.message : String(error),
    });
  }
};

scope.postMessage({ kind: "ready" });
