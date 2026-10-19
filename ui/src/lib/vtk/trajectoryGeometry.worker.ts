import { buildTrajectoryTubeGeometry } from "./trajectoryGeometry";
import {
  trajectoryGeometryTransferables,
  type TrajectoryDataset,
  type TrajectoryGeometryWorkerReply,
  type TrajectoryGeometryWorkerRequest,
} from "./trajectoryGeometryProtocol";

const scope = self as unknown as {
  onmessage: ((event: { data: TrajectoryGeometryWorkerRequest }) => void) | null;
  postMessage: (message: TrajectoryGeometryWorkerReply, transfer: ArrayBuffer[]) => void;
  close: () => void;
};

let dataset: TrajectoryDataset | null = null;
let activeDatasetId = -1;

scope.onmessage = (event) => {
  if (event.data.kind === "close") {
    dataset = null;
    scope.close();
    return;
  }
  const { requestId, datasetId } = event.data;
  try {
    if (event.data.dataset) {
      dataset = event.data.dataset;
      activeDatasetId = datasetId;
    }
    if (!dataset || activeDatasetId !== datasetId) {
      throw new Error("Trajectory worker does not have the requested dataset");
    }
    const result = buildTrajectoryTubeGeometry(dataset, event.data);
    scope.postMessage(
      { kind: "result", requestId, datasetId, ok: true, result },
      trajectoryGeometryTransferables(result),
    );
  } catch (error) {
    scope.postMessage({
      kind: "result",
      requestId,
      datasetId,
      ok: false,
      error: error instanceof Error ? error.message : String(error),
    }, []);
  }
};

scope.postMessage({ kind: "ready" }, []);

export {};
