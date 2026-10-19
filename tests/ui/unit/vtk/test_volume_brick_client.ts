import { VolumeBrickPreparationClient } from "../../../../ui/src/lib/vtk/volumeBrickClient.ts";
import type { VolumeBrickWorkerReply, VolumeBrickWorkerRequest } from "../../../../ui/src/lib/vtk/volumeBrickProtocol.ts";
import type { ScalarPreparationRequest } from "../../../../ui/src/lib/vtk/vtkPreparation.ts";
import { handoffVtkPreparationInput } from "../../../../ui/src/lib/vtk/vtkPreparationInputHandoff.ts";

let passed = 0;
const failures: string[] = [];
function check(condition: boolean, label: string): void {
  if (condition) passed += 1;
  else failures.push(label);
}

class FakeWorker {
  onmessage: ((event: MessageEvent<VolumeBrickWorkerReply>) => void) | null = null;
  onerror: ((event: ErrorEvent) => void) | null = null;
  onmessageerror: ((event: MessageEvent) => void) | null = null;
  prepares = 0;
  terminated = 0;
  closed = 0;
  stall = false;

  constructor() { queueMicrotask(() => this.reply({ kind: "ready" })); }

  postMessage(request: VolumeBrickWorkerRequest): void {
    if (request.kind === "close") { this.closed++; this.reply({ kind: "closed" }); return; }
    this.prepares += 1;
    if (this.stall) return;
    if (request.kind === "prepare-anomaly-batch") {
      const prepared = request.requests.map((item, index) => ({
        shape: item.field.shape,
        spacing: [1, 1, 1] as [number, number, number],
        origin: [index * 2, 0, 0] as [number, number, number],
        values: Float32Array.from(item.field.data),
        sourceNonFiniteCount: 0,
        scalarTransform: "horizontal-level-anomaly" as const,
        samplingProvenance: {} as never,
      }));
      this.reply({
        kind: "prepared-batch",
        requestId: request.requestId,
        ok: true,
        bytes: prepared.reduce((sum, item) => sum + item.values.byteLength, 0),
        prepared,
      });
      return;
    }
    const values = Float32Array.from(request.request.field.data);
    this.reply({
      kind: "prepared",
      requestId: request.requestId,
      ok: true,
      bytes: values.byteLength,
      prepared: {
        shape: request.request.field.shape,
        spacing: [1, 1, 1],
        origin: [0, 0, 0],
        values,
        sourceNonFiniteCount: 0,
        stats: { min: 1, max: 1, p2: 1, p98: 1 },
        samplingProvenance: {
          sourceShape: request.request.field.shape,
          sourceSpacing: [1, 1, 1],
          sampledShape: request.request.field.shape,
          sampledSpacing: [1, 1, 1],
          displayShape: request.request.field.shape,
          displaySpacing: [1, 1, 1],
          sampling: "native",
          reconstruction: "none",
          reconstructionFactors: [1, 1, 1],
          coordinateResampling: "none",
          coordinateResamplingAxes: [],
        },
      },
    });
  }

  terminate(): void { this.terminated += 1; }
  private reply(data: VolumeBrickWorkerReply): void {
    queueMicrotask(() => this.onmessage?.({ data } as MessageEvent<VolumeBrickWorkerReply>));
  }
}

const request: ScalarPreparationRequest = {
  kind: "scalar",
  field: { shape: [2, 2, 2], data: new Float32Array(8).fill(1) },
  grid: { coordinateSystem: "cartesian", drM: 1, dyM: 1, dzM: 1 },
  refineForDisplay: false,
};
const worker = new FakeWorker();
const client = new VolumeBrickPreparationClient(worker);
const prepared = await client.prepare(request, new AbortController().signal);
check(prepared.values.length === 8 && worker.prepares === 1,
  "volume brick client transfers one bounded scalar request");
const cancelled = new AbortController();
cancelled.abort();
let cancellationObserved = false;
try {
  await client.prepare(request, cancelled.signal);
} catch (error) {
  cancellationObserved = error instanceof DOMException && error.name === "AbortError";
}
check(cancellationObserved && worker.prepares === 1,
  "cancelled volume brick work never enters the worker");
const batch = await client.prepareAnomalyBatch([request, request], new AbortController().signal);
check(batch.length === 2 && batch.every((item) =>
  item.scalarTransform === "horizontal-level-anomaly"),
"volume brick client commits one complete anomaly batch");
client.shutdown();
await new Promise<void>((resolve) => queueMicrotask(resolve));
check(worker.closed === 1 && worker.terminated === 0, "idle volume brick workers acknowledge self-close without double termination");
let retiredErrorHandled = false;
worker.onerror?.({ preventDefault: () => { retiredErrorHandled = true; } } as ErrorEvent);
check(retiredErrorHandled, "retired workers consume late module-loading errors");
const neverUsedWorker = new FakeWorker();
const neverUsedClient = new VolumeBrickPreparationClient(neverUsedWorker);
neverUsedClient.shutdown();
check(neverUsedWorker.terminated === 0, "startup cancellation queues self-close without interrupting module loading");
let startupAbortObserved = false;
try {
  await neverUsedClient.prepare(request, new AbortController().signal);
} catch (error) {
  startupAbortObserved = error instanceof DOMException && error.name === "AbortError";
}
check(startupAbortObserved, "closing before the first request preserves cancellation for callers");
await new Promise<void>((resolve) => queueMicrotask(resolve));
check(neverUsedWorker.closed === 1 && neverUsedWorker.terminated === 0 && neverUsedWorker.prepares === 0,
  "a cancelled startup releases its worker without posting scalar data");
const busyWorker = new FakeWorker();
busyWorker.stall = true;
const busyClient = new VolumeBrickPreparationClient(busyWorker);
const interrupted = busyClient.prepare(request, new AbortController().signal).catch((error: unknown) => error);
for (let attempt = 0; attempt < 20 && busyWorker.prepares === 0; attempt++) {
  await new Promise<void>((resolve) => setTimeout(resolve, 0));
}
check(busyWorker.prepares === 1, "cancellation fixture has an in-flight request");
busyClient.shutdown();
const interruptedError = await interrupted;
check(busyWorker.terminated === 1 && busyWorker.closed === 0
  && interruptedError instanceof DOMException && interruptedError.name === "AbortError",
"busy workers are terminated directly and callers observe cancellation");

const unhandled: unknown[] = [];
const onUnhandled = (reason: unknown) => { unhandled.push(reason); };
process.on("unhandledRejection", onUnhandled);
try {
  for (const batch of [false, true]) {
    for (const failure of ["shutdown", "worker-error", "consumer-abort"] as const) {
      let release!: () => void;
      let entered!: () => void;
      const arrival = new Promise<void>(resolve => { entered = resolve; });
      const paused = new Promise<void>(resolve => { release = resolve; });
      const handoff: typeof handoffVtkPreparationInput = async (input, cancelled) => {
        entered();
        await paused;
        return handoffVtkPreparationInput(input, cancelled);
      };
      const raceWorker = new FakeWorker();
      const raceClient = new VolumeBrickPreparationClient(raceWorker, handoff);
      const controller = new AbortController();
      const observed = (batch
        ? raceClient.prepareAnomalyBatch([request, request], controller.signal)
        : raceClient.prepare(request, controller.signal)).catch((error: unknown) => error);
      await arrival;
      if (failure === "shutdown") raceClient.shutdown();
      else if (failure === "consumer-abort") controller.abort();
      else raceWorker.onerror?.({ message: "original worker failure", preventDefault: () => {} } as ErrorEvent);
      await new Promise<void>(resolve => setTimeout(resolve, 0));
      check(unhandled.length === 0, `${batch ? "batch" : "scalar"} ${failure} during handoff owns every internal rejection immediately`);
      release();
      const error = await observed;
      check(failure === "worker-error"
        ? error instanceof Error && error.message === "original worker failure"
        : error instanceof DOMException && error.name === "AbortError",
      `${batch ? "batch" : "scalar"} ${failure} remains observable as the original consumer failure`);
      check(raceWorker.prepares === 0, `${batch ? "batch" : "scalar"} ${failure} never posts retired or aborted field data`);
      raceClient.shutdown();
      await new Promise<void>(resolve => setTimeout(resolve, 0));
    }
  }
  check(unhandled.length === 0, "worker-close races emit no unhandled rejection after all handoffs retire");
} finally {
  process.off("unhandledRejection", onUnhandled);
}

console.log(`\nvolume brick client tests: ${passed} passed, ${failures.length} failed.`);
if (failures.length > 0) {
  for (const failure of failures) console.error(`FAIL: ${failure}`);
  process.exitCode = 1;
}
