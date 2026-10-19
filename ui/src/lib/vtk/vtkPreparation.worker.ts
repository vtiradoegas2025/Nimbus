import {
  prepareVtkRequest,
  type VtkPreparationRequest,
} from "./vtkPreparation";
import {
  sharePreparedVtkResult,
  downgradeSharedPreparedResult,
  transferablePreparedBuffers,
} from "./vtkDatasetHandoff";

const workerScope = self as unknown as {
  onmessage: ((event: { data:
    | VtkPreparationRequest
    | { kind: "prepare"; requestId: number; request: VtkPreparationRequest }
    | { kind: "shared-buffer-probe-ack"; buffer: SharedArrayBuffer }
    | { kind: "close" }
  }) => void) | null;
  postMessage: (message: unknown, transfer: ArrayBuffer[]) => void;
  close: () => void;
};

let sharedGeometrySupported = false;
let probeSettled = false;

function ready(): void {
  if (probeSettled) return;
  probeSettled = true;
  workerScope.postMessage({ kind: "ready", sharedGeometrySupported }, []);
}

workerScope.onmessage = (event) => {
  if (event.data.kind === "shared-buffer-probe-ack") {
    sharedGeometrySupported = event.data.buffer.byteLength === 4
      && new Uint32Array(event.data.buffer)[0] === 0x4e494d42;
    ready();
    return;
  }
  if (event.data.kind === "close") {
    workerScope.close();
    return;
  }
  const envelope = event.data.kind === "prepare" ? event.data : null;
  const request = envelope?.request ?? event.data as VtkPreparationRequest;
  try {
    const result = prepareVtkRequest(request);
    const handoff = sharePreparedVtkResult(
      result,
      request.datasetHandoff === "shared-geometry" && sharedGeometrySupported,
    );
    try {
      workerScope.postMessage({
        kind: "result",
        requestId: envelope?.requestId,
        ok: true,
        result,
        handoff,
      }, transferablePreparedBuffers(result));
    } catch (error) {
      // A successful round trip is required before promotion, but a platform
      // can still reject the larger payload. Rebuild one transferable AB copy
      // from the intact shared grid and retry without detaching it.
      const retry = downgradeSharedPreparedResult(result);
      workerScope.postMessage({
        kind: "result",
        requestId: envelope?.requestId,
        ok: true,
        result,
        handoff: {
          ...retry,
        },
        sharedTransferError: error instanceof Error ? error.message : String(error),
      }, transferablePreparedBuffers(result));
    }
  } catch (error) {
    workerScope.postMessage({
      kind: "result",
      requestId: envelope?.requestId,
      ok: false,
      error: error instanceof Error ? error.message : String(error),
    }, []);
  }
};

// A real clone/echo round trip catches worker-backed WebKit runtimes that
// expose SharedArrayBuffer but reject it at the message boundary.
try {
  if (typeof SharedArrayBuffer === "undefined" || self.crossOriginIsolated !== true) {
    ready();
  } else {
    const probe = new SharedArrayBuffer(4);
    new Uint32Array(probe)[0] = 0x4e494d42;
    workerScope.postMessage({ kind: "shared-buffer-probe", buffer: probe }, []);
    setTimeout(ready, 250);
  }
} catch {
  ready();
}

export {};
