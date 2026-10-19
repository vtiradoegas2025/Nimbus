import { BrickedContourClient } from "../../../../ui/src/lib/vtk/brickedContourClient.ts";
import type { BrickedContourWorkerReply, BrickedContourWorkerRequest } from "../../../../ui/src/lib/vtk/brickedContourProtocol.ts";
import {
  getVisualizationResourceSnapshot,
  visualizationResourceGovernor,
} from "../../../../ui/src/lib/core/visualizationResourceGovernor.ts";
import type { ScalarPreparationRequest } from "../../../../ui/src/lib/vtk/vtkPreparation.ts";
import type { VortexSurfaceGeometry } from "../../../../ui/src/lib/vtk/vortexContourProtocol.ts";

let passed = 0;
const failures: string[] = [];
function check(condition: boolean, label: string): void {
  if (condition) passed += 1;
  else failures.push(label);
}

class FakeWorker {
  onmessage: ((event: MessageEvent<BrickedContourWorkerReply>) => void) | null = null;
  onerror: ((event: ErrorEvent) => void) | null = null;
  onmessageerror: ((event: MessageEvent) => void) | null = null;
  loads = 0;
  extracts = 0;
  resets = 0;
  terminated = 0;
  failNextLoad = false;
  geometry: VortexSurfaceGeometry | null = null;
  holdNextExtraction = false;
  holdExtractionNumber: number | null = null;
  failExtractionNumber: number | null = null;
  heldExtraction: Extract<BrickedContourWorkerRequest, { kind: "extract" }> | null = null;

  constructor() {
    queueMicrotask(() => this.reply({ kind: "ready" }));
  }

  postMessage(request: BrickedContourWorkerRequest): void {
    if (request.kind === "reset") {
      this.resets += 1;
      return;
    }
    if (request.kind === "close") return;
    if (request.kind === "load") {
      this.loads += 1;
      if (this.failNextLoad) {
        this.failNextLoad = false;
        this.reply({
          kind: "loaded",
          requestId: request.requestId,
          datasetIdentity: request.datasetIdentity,
          brickId: request.brickId,
          ok: false,
          error: "injected load failure",
        });
      } else {
        this.reply({
          kind: "loaded",
          requestId: request.requestId,
          datasetIdentity: request.datasetIdentity,
          brickId: request.brickId,
          ok: true,
          bytes: fieldOf(request).data.byteLength,
          stats: { min: valueOf(request), max: valueOf(request), p2: valueOf(request), p98: valueOf(request), sourceNonFiniteCount: 0 },
        });
      }
      return;
    }
    this.extracts += 1;
    if (this.failExtractionNumber === this.extracts) {
      this.failExtractionNumber = null;
      this.reply({ kind: "result", requestId: request.requestId,
        datasetIdentity: request.datasetIdentity, ok: false, error: "injected extraction failure" });
      return;
    }
    if (this.holdNextExtraction || this.holdExtractionNumber === this.extracts) {
      this.holdNextExtraction = false;
      this.holdExtractionNumber = null;
      this.heldExtraction = request;
      return;
    }
    this.replyExtraction(request);
  }

  completeHeldExtraction(): void {
    const request = this.heldExtraction;
    if (!request) throw new Error("No held extraction");
    this.heldExtraction = null;
    this.replyExtraction(request);
  }

  private replyExtraction(request: Extract<BrickedContourWorkerRequest, { kind: "extract" }>): void {
    this.reply({
      kind: "result",
      requestId: request.requestId,
      datasetIdentity: request.datasetIdentity,
      ok: true,
      chunks: request.brickIds.map((brickId) => ({ brickId, geometry: this.geometry })),
      knownScratchBytes: 0,
    });
  }

  terminate(): void { this.terminated += 1; }

  crash(): void {
    this.onerror?.({
      message: "injected worker crash",
      preventDefault: () => undefined,
    } as unknown as ErrorEvent);
  }

  private reply(data: BrickedContourWorkerReply): void {
    queueMicrotask(() => this.onmessage?.({ data } as MessageEvent<BrickedContourWorkerReply>));
  }
}

function valueOf(request: Extract<BrickedContourWorkerRequest, { kind: "load" }>): number {
  return fieldOf(request).data[0];
}

function fieldOf(request: Extract<BrickedContourWorkerRequest, { kind: "load" }>) {
  return request.request.kind === "scalar" ? request.request.field : request.request.fields[0];
}

function request(value: number): ScalarPreparationRequest {
  return {
    kind: "scalar",
    field: { shape: [2, 2, 2], data: new Float32Array(8).fill(value) },
    grid: {
      coordinateSystem: "cartesian",
      drM: 100,
      dyM: 100,
      dzM: 100,
      scalarOriginM: [0, 0, 0],
    },
    refineForDisplay: false,
    datasetHandoff: "shared-geometry",
  };
}

const baseline = getVisualizationResourceSnapshot();
const worker = new FakeWorker();
const client = new BrickedContourClient(worker);
await client.reset("frame-a");
check(worker.resets === 1, "client establishes one worker dataset generation");
await client.admitBricks([
  { brickId: "a", inputBytes: 32, outputBytes: 32 },
  { brickId: "b", inputBytes: 32, outputBytes: 32 },
]);
check(client.admittedBrickCount() === 2, "whole selected generation is admitted once");
await client.loadBrick("a", request(1), new AbortController().signal);
await client.loadBrick("a", request(1), new AbortController().signal);
await client.loadBrick("b", request(2), new AbortController().signal);
check(worker.loads === 2, "resident bricks transfer once per generation");
check(client.loadedBrickCount() === 2, "client reports two resident bricks");
check(getVisualizationResourceSnapshot().byStage["worker-resident"].residentBytes > baseline.byStage["worker-resident"].residentBytes, "resident brick bytes are owned");
await client.extract(["a", "b"], 0.5, new AbortController().signal);
await client.extract(["a", "b"], 0.75, new AbortController().signal);
check(worker.extracts === 4, "threshold edits execute once per brick without one aggregate envelope");
check(worker.loads === 2, "threshold edits transfer no resident source lattice");
worker.geometry = {
  points: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
  polys: new Uint32Array([3, 0, 1, 2]),
  normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
  bounds: [0, 1, 0, 1, 0, 0], triangles: 1, bytes: 88,
};
worker.holdNextExtraction = true;
const admittedExtraction = client.extract(
  ["a"], 0.8, new AbortController().signal, { retentionKey: "primary" },
);
await Promise.resolve();
check(getVisualizationResourceSnapshot().byStage["retained-geometry"].reservedBytes
  === baseline.byStage["retained-geometry"].reservedBytes + 8 * 5 * 88,
"topology-derived contour output is reserved before worker extraction begins");
check(getVisualizationResourceSnapshot().byStage["worker-scratch"].reservedBytes
  === baseline.byStage["worker-scratch"].reservedBytes + 8 * 5 * 88,
"the same per-brick topology bound admits extraction scratch before execution");
worker.completeHeldExtraction();
await admittedExtraction;
check(getVisualizationResourceSnapshot().byStage["retained-geometry"].residentBytes
  === baseline.byStage["retained-geometry"].residentBytes + 88,
"extraction admission transitions to exact geometry residency");

worker.holdExtractionNumber = worker.extracts + 2;
const progressiveExtraction = client.extract(
  ["a", "b"], 0.9, new AbortController().signal, { retentionKey: "primary" },
);
await new Promise((resolve) => setTimeout(resolve, 0));
check(getVisualizationResourceSnapshot().byStage["retained-geometry"].reservedBytes
  === baseline.byStage["retained-geometry"].reservedBytes + 8 * 5 * 88,
"multi-brick extraction admits only the active brick envelope, not the aggregate worst case");
worker.completeHeldExtraction();
await progressiveExtraction;
const retainedBeforeFailure = getVisualizationResourceSnapshot().byStage["retained-geometry"].residentBytes;
worker.failExtractionNumber = worker.extracts + 2;
let progressiveFailure = false;
try {
  await client.extract(["a", "b"], 1.1, new AbortController().signal, { retentionKey: "primary" });
} catch (error) {
  progressiveFailure = error instanceof Error && error.message.includes("injected extraction failure");
}
check(progressiveFailure
  && getVisualizationResourceSnapshot().byStage["retained-geometry"].residentBytes === retainedBeforeFailure,
"a failed progressive replacement releases partial output and preserves the prior generation");

await client.reset("frame-b");
check(client.loadedBrickCount() === 0, "new frame retires the prior brick set");
check(getVisualizationResourceSnapshot().byStage["worker-resident"].residentBytes === baseline.byStage["worker-resident"].residentBytes, "frame replacement releases prior worker residency");
check(getVisualizationResourceSnapshot().byStage["retained-geometry"].residentBytes
  === baseline.byStage["retained-geometry"].residentBytes,
"frame replacement releases client-owned contour geometry");
worker.failNextLoad = true;
await client.admitBricks([{ brickId: "bad", inputBytes: 32, outputBytes: 32 }]);
let failureObserved = false;
try {
  await client.loadBrick("bad", request(3), new AbortController().signal);
} catch (error) {
  failureObserved = error instanceof Error && error.message.includes("injected load failure");
}
check(failureObserved, "brick load failure rejects its consumer");
check(getVisualizationResourceSnapshot().reservedBytes === baseline.reservedBytes, "failed load releases reservations");

client.shutdown();
check(worker.terminated === 1, "shutdown terminates the resident brick worker");
check(getVisualizationResourceSnapshot().residentBytes === baseline.residentBytes, "shutdown returns resident accounting to baseline");

const crashWorker = new FakeWorker();
const crashClient = new BrickedContourClient(crashWorker);
await crashClient.reset("crash-frame");
await crashClient.admitBricks([{ brickId: "a", inputBytes: 32, outputBytes: 32 }]);
await crashClient.loadBrick("a", request(4), new AbortController().signal);
crashWorker.crash();
await new Promise((resolve) => setTimeout(resolve, 0));
check(crashClient.isClosed(), "worker failure closes the poisoned resident set");
check(getVisualizationResourceSnapshot().residentBytes === baseline.residentBytes, "worker failure releases every resident brick");

const deniedWorker = new FakeWorker();
const deniedClient = new BrickedContourClient(deniedWorker);
await deniedClient.reset("denied-frame");
visualizationResourceGovernor.setCapacity({ totalBytes: baseline.committedBytes + 1 });
let admissionDenied = false;
try {
  await deniedClient.admitBricks([{ brickId: "too-large", inputBytes: 32, outputBytes: 32 }]);
} catch (error) {
  admissionDenied = error instanceof Error && error.message.length > 0;
}
check(admissionDenied, "aggregate generation admission fails before field work");
check(deniedWorker.loads === 0, "denied generation performs zero brick loads");
visualizationResourceGovernor.setCapacity({ totalBytes: baseline.capacityBytes });
deniedClient.shutdown();

const cancelledWorker = new FakeWorker();
const cancelledClient = new BrickedContourClient(cancelledWorker);
const cancelledReset = new AbortController();
cancelledReset.abort();
let resetCancelled = false;
try {
  await cancelledClient.reset("stale-frame", cancelledReset.signal);
} catch (error) {
  resetCancelled = error instanceof DOMException && error.name === "AbortError";
}
check(resetCancelled && cancelledWorker.resets === 0,
  "cancelled stale effect cannot reset the resident generation");
await cancelledClient.reset("current-frame");
const cancelledAdmission = new AbortController();
cancelledAdmission.abort();
let admissionCancelled = false;
try {
  await cancelledClient.admitBricks(
    [{ brickId: "stale", inputBytes: 32, outputBytes: 32 }],
    cancelledAdmission.signal,
  );
} catch (error) {
  admissionCancelled = error instanceof DOMException && error.name === "AbortError";
}
check(admissionCancelled && cancelledClient.admittedBrickCount() === 0,
  "cancelled stale effect cannot replace aggregate admission");
cancelledClient.shutdown();

const externalWorker = new FakeWorker();
const externalClient = new BrickedContourClient(externalWorker, { externallyReserved: true });
await externalClient.reset("externally-admitted-frame");
await externalClient.admitBricks([{ brickId: "a", inputBytes: 32, outputBytes: 32 }]);
check(getVisualizationResourceSnapshot().committedBytes === baseline.committedBytes,
  "externally admitted client does not duplicate hierarchy accounting");
await externalClient.loadBrick("a", request(5), new AbortController().signal);
check(externalClient.residentBytes() === 32,
  "externally admitted client reports exact worker residency to its hierarchy owner");
check(getVisualizationResourceSnapshot().committedBytes === baseline.committedBytes,
  "external worker load remains owned only by the hierarchy reservation");
externalClient.shutdown();

{
  const originalWorker = Object.getOwnPropertyDescriptor(globalThis, "Worker");
  const urls: string[] = [];
  Object.defineProperty(globalThis, "Worker", { configurable: true, value: class extends FakeWorker {
    constructor(url: string | URL, options?: WorkerOptions) {
      super();
      urls.push(String(url));
      check(options?.type === "module" && options.name === "nimbus-bricked-contour-set",
        "native brick worker generations preserve module execution and worker identity");
    }
  } });
  try {
    const first = new BrickedContourClient();
    await first.reset("first-generation");
    first.shutdown();
    const second = new BrickedContourClient();
    await second.reset("replacement-generation");
    second.shutdown();
    const parsed = urls.map(url => new URL(url, "https://nimbus.invalid/"));
    check(parsed.length === 2 && parsed[0].pathname === parsed[1].pathname,
      "replacement workers load the same maintained script");
    check(parsed.every(url => /^\d+$/.test(url.searchParams.get("generation") ?? ""))
      && parsed[0].searchParams.get("generation") !== parsed[1].searchParams.get("generation"),
    "replacement brick workers cannot reuse a cancelled cached module response");
  } finally {
    if (originalWorker) Object.defineProperty(globalThis, "Worker", originalWorker);
    else Reflect.deleteProperty(globalThis, "Worker");
  }
}

console.log(`\nbricked contour client tests: ${passed} passed, ${failures.length} failed.`);
if (failures.length > 0) {
  for (const failure of failures) console.error(`FAIL: ${failure}`);
  process.exitCode = 1;
}
