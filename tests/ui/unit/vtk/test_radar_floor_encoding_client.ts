import {
  RadarFloorEncodingService,
  type RadarFloorEncodingWorker,
} from "../../../../ui/src/lib/vtk/radarFloorEncodingClient.ts";
import type {
  RadarFloorEncodingWorkerReply,
  RadarFloorEncodingWorkerRequest,
} from "../../../../ui/src/lib/vtk/radarFloorEncodingProtocol.ts";
import { getVisualizationResourceSnapshot } from "../../../../ui/src/lib/core/visualizationResourceGovernor.ts";

let passed = 0;
const failures: string[] = [];
const check = (condition: boolean, label: string) => {
  if (condition) passed += 1;
  else failures.push(label);
};

class FakeWorker implements RadarFloorEncodingWorker {
  onmessage: ((event: MessageEvent<RadarFloorEncodingWorkerReply>) => void) | null = null;
  onerror: ((event: ErrorEvent) => void) | null = null;
  onmessageerror: ((event: MessageEvent) => void) | null = null;
  requests: RadarFloorEncodingWorkerRequest[] = [];
  hold = false;
  terminated = false;

  constructor() {
    queueMicrotask(() => this.reply({ kind: "ready" }));
  }

  postMessage(request: RadarFloorEncodingWorkerRequest): void {
    this.requests.push(request);
    if (request.kind !== "encode" || this.hold) return;
    const rgba = new Uint8ClampedArray(request.shape[0] * request.shape[1] * 4);
    rgba.set([1, 2, 3, 255], 0);
    this.reply({
      kind: "result",
      requestId: request.requestId,
      ok: true,
      result: { rgba, paintedCells: 1, elapsedMs: 1 },
    });
  }

  terminate(): void {
    this.terminated = true;
  }

  private reply(reply: RadarFloorEncodingWorkerReply): void {
    queueMicrotask(() => this.onmessage?.({ data: reply } as MessageEvent<RadarFloorEncodingWorkerReply>));
  }
}

const policy = {
  kind: "dbz" as const,
  reduction: "column-max" as const,
  colormap: "dBZ floor",
  min: -30,
  max: 75,
};
const values = new Float32Array([10, 20, 30, 40]);
const baseline = getVisualizationResourceSnapshot();
const worker = new FakeWorker();
const service = new RadarFloorEncodingService(() => worker);
const result = await service.encode(values, [2, 2], policy);
let snapshot = getVisualizationResourceSnapshot();
check(result.rgba.byteLength === 16 && result.paintedCells === 1,
  "the service returns the worker's encoded pixels and visibility count");
check(snapshot.byStage["worker-resident"].committedBytes
  === baseline.byStage["worker-resident"].committedBytes,
"the input-copy reservation retires when encoding completes");
check(snapshot.byStage["prepared-main"].residentBytes
  === baseline.byStage["prepared-main"].residentBytes + result.rgba.byteLength,
"the output reservation transitions to the exact returned RGBA buffer");
result.release();
service.shutdown();
snapshot = getVisualizationResourceSnapshot();
check(worker.terminated && snapshot.committedBytes === baseline.committedBytes,
  "release and shutdown retire output and worker ownership");

const heldWorker = new FakeWorker();
heldWorker.hold = true;
const heldService = new RadarFloorEncodingService(() => heldWorker);
const controller = new AbortController();
const cancelled = heldService.encode(values, [2, 2], policy, controller.signal);
await new Promise((resolve) => setTimeout(resolve, 0));
const during = getVisualizationResourceSnapshot();
check(during.byStage["worker-resident"].reservedBytes >= values.byteLength
  && during.byStage["prepared-main"].reservedBytes >= 16,
"input copy and exact output bytes are admitted before worker execution");
controller.abort();
let sawAbort = false;
try {
  await cancelled;
} catch (error) {
  sawAbort = error instanceof DOMException && error.name === "AbortError";
}
heldService.shutdown();
check(sawAbort && heldWorker.terminated
  && getVisualizationResourceSnapshot().committedBytes === baseline.committedBytes,
"latest-wins cancellation terminates active encoding and releases both reservations");

console.log(`\nradar floor encoding client tests: ${passed} passed, ${failures.length} failed.`);
if (failures.length) {
  for (const failure of failures) console.error(`FAIL: ${failure}`);
  process.exitCode = 1;
}
