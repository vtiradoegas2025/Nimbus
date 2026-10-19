import type { PreparedVectorImage } from "../../../../ui/src/lib/vtk/vtkPreparation.ts";
import type { StreamlineExtractionParameters, StreamlineExtractionResult, StreamlineWorkerReply, StreamlineWorkerRequest } from "../../../../ui/src/lib/vtk/streamlineExtractionProtocol.ts";
import { SharedStreamlineExtractionService, type StreamlineExtractionWorker } from "../../../../ui/src/lib/vtk/streamlineExtractionClient.ts";

let passed = 0;
let failed = 0;
const failures: string[] = [];
function check(condition: boolean, label: string): void {
  if (condition) passed += 1;
  else { failed += 1; failures.push(`FAIL ${label}`); }
}

class FakeWorker implements StreamlineExtractionWorker {
  onmessage: ((event: MessageEvent<StreamlineWorkerReply>) => void) | null = null;
  onerror: ((event: ErrorEvent) => void) | null = null;
  onmessageerror: ((event: MessageEvent) => void) | null = null;
  readonly requests: Array<Extract<StreamlineWorkerRequest, { kind: "extract" }>> = [];
  terminated = 0;
  postMessage(message: StreamlineWorkerRequest): void {
    if (message.kind === "extract") this.requests.push(message);
  }
  terminate(): void { this.terminated += 1; }
  ready(): void { this.onmessage?.({ data: { kind: "ready" } } as MessageEvent<StreamlineWorkerReply>); }
  fail(message: string): void {
    this.onerror?.({ message, preventDefault: () => {} } as ErrorEvent);
  }
  succeed(index: number, result: StreamlineExtractionResult): void {
    const request = this.requests[index]!;
    this.onmessage?.({ data: {
      kind: "result",
      requestId: request.requestId,
      datasetId: request.datasetId,
      ok: true,
      result,
    } } as MessageEvent<StreamlineWorkerReply>);
  }
}

const prepared = {
  shape: [2, 2, 2], spacing: [1, 1, 1], origin: [0, 0, 0],
  values: new Float32Array(2 * 2 * 2 * 3), sourceNonFiniteCount: 0,
  samplingProvenance: {}, stormCenter: [0, 0], speed95: 1,
  featureSeeds: new Float32Array(), vortexCore: null,
} as unknown as PreparedVectorImage;
const parameters: StreamlineExtractionParameters = {
  seeds: new Float32Array([0, 0, 0]), integrationStep: 0.5,
  maxSteps: 10, tubeRadius: 1, scale: { xy: 1, z: 1 },
};
const result: StreamlineExtractionResult = {
  forward: { path: null, arrows: null }, backward: { path: null, arrows: null },
  representation: "tubes", integratedPoints: 0, knownScratchBytes: 16, elapsedMs: 1,
  boundaryContacts: { lower: [false, false, false], upper: [false, false, false] },
};

const worker = new FakeWorker();
const service = new SharedStreamlineExtractionService(() => worker);
const firstController = new AbortController();
const first = service.enqueue(prepared, parameters, firstController.signal);
const second = service.enqueue(prepared, { ...parameters, seeds: parameters.seeds.slice() });
worker.ready();
check(worker.requests.length === 1, "identical pane requests execute once");
firstController.abort();
check(await first.then(() => false, (error) => error instanceof DOMException && error.name === "AbortError"), "one pane cancels independently");
check(worker.terminated === 0, "one cancellation preserves the shared worker");
worker.succeed(0, result);
check(await second === result, "remaining pane receives shared geometry");

const warm = service.enqueue(prepared, { ...parameters, tubeRadius: 2 });
check(worker.requests.length === 2 && worker.requests[1]!.image == null, "parameter edits reuse the warm worker dataset");
worker.succeed(1, result);
await warm;
service.shutdown();
check(worker.terminated === 1, "service shutdown terminates the shared worker");

const failedWorker = new FakeWorker();
const retryWorker = new FakeWorker();
const retryWorkers = [failedWorker, retryWorker];
const retryService = new SharedStreamlineExtractionService(() => retryWorkers.shift()!);
const retried = retryService.enqueue(prepared, parameters);
failedWorker.ready();
failedWorker.fail("synthetic worker loss");
await new Promise((resolve) => setTimeout(resolve, 30));
retryWorker.ready();
check(retryWorker.requests.length === 1, "active extraction restarts once on a fresh worker");
retryWorker.succeed(0, result);
check(await retried === result, "worker restart preserves the active consumer result");
retryService.shutdown();

const cancelledWorker = new FakeWorker();
const cancellationService = new SharedStreamlineExtractionService(() => cancelledWorker);
const cancellation = new AbortController();
const abandoned = cancellationService.enqueue(prepared, parameters, cancellation.signal);
cancelledWorker.ready();
cancellation.abort();
check(await abandoned.then(() => false, (error) => error instanceof DOMException && error.name === "AbortError"), "last-consumer cancellation rejects deterministically");
check(cancelledWorker.terminated === 1, "last-consumer cancellation terminates synchronous worker work");
cancellationService.shutdown();

console.log(`\nstreamlineExtractionClient tests: ${passed} passed, ${failed} failed.`);
if (failures.length) {
  console.log(failures.join("\n"));
  process.exit(1);
}
