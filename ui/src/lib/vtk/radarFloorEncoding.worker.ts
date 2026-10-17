import { encodeFloorRgba } from "@/lib/plot/radarFloor";
import type {
  RadarFloorEncodingWorkerReply,
  RadarFloorEncodingWorkerRequest,
} from "./radarFloorEncodingProtocol";

const scope = self as unknown as {
  onmessage: ((event: { data: RadarFloorEncodingWorkerRequest }) => void) | null;
  postMessage: (message: RadarFloorEncodingWorkerReply, transfer: ArrayBuffer[]) => void;
  close: () => void;
};

scope.onmessage = (event) => {
  if (event.data.kind === "close") {
    scope.close();
    return;
  }
  const { requestId, values, shape, policy } = event.data;
  try {
    const expected = shape[0] * shape[1];
    if (!Number.isSafeInteger(expected) || expected < 1 || values.length !== expected) {
      throw new Error("Radar floor encoding shape does not match its values");
    }
    const startedAt = performance.now();
    const rgba = encodeFloorRgba(values, shape, policy);
    let paintedCells = 0;
    for (let alpha = 3; alpha < rgba.length; alpha += 4) {
      if (rgba[alpha]! > 0) paintedCells += 1;
    }
    const result = { rgba, paintedCells, elapsedMs: performance.now() - startedAt };
    scope.postMessage(
      { kind: "result", requestId, ok: true, result },
      [rgba.buffer],
    );
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
