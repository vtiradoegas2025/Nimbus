import { getVisualizationResourceSnapshot } from "../../../../ui/src/lib/core/visualizationResourceGovernor.ts";
import { prepareVtkRequest, type VectorPreparationRequest } from "../../../../ui/src/lib/vtk/vtkPreparation.ts";
import type {
  StreamlineExtractionParameters,
  StreamlineExtractionResult,
  StreamlineWorkerReply,
  StreamlineWorkerRequest,
} from "../../../../ui/src/lib/vtk/streamlineExtractionProtocol.ts";
import {
  StreamlineWorkerOwnedPreparationService,
  type WorkerOwnedPreparationWorker,
} from "../../../../ui/src/lib/vtk/streamlineWorkerOwnedPreparation.ts";
import { preparedDatasetRegistrySnapshot, resolvePreparedDataset } from "../../../../ui/src/lib/vtk/vtkPreparedDatasetRegistry.ts";

let passed = 0;
const failures: string[] = [];
function check(condition: boolean, label: string): void {
  if (condition) passed += 1;
  else failures.push(`FAIL ${label}`);
}
const flushHandoff = () => new Promise<void>((resolve) => setTimeout(resolve, 0));

class FakeWorker implements WorkerOwnedPreparationWorker {
  onmessage: ((event: MessageEvent<StreamlineWorkerReply>) => void) | null = null;
  onerror: ((event: ErrorEvent) => void) | null = null;
  onmessageerror: ((event: MessageEvent) => void) | null = null;
  readonly requests: Array<Extract<StreamlineWorkerRequest, { kind: "prepare" }>> = [];
  readonly extractions: Array<Extract<StreamlineWorkerRequest, { kind: "extract" }>> = [];
  readonly transfers: Transferable[][] = [];
  terminated = 0;
  postMessage(message: StreamlineWorkerRequest, transfer: Transferable[] = []): void {
    this.transfers.push(transfer);
    if (message.kind === "prepare") this.requests.push(message);
    if (message.kind === "extract") this.extractions.push(message);
  }
  terminate(): void { this.terminated += 1; }
  ready(): void { this.onmessage?.({ data: { kind: "ready" } } as MessageEvent<StreamlineWorkerReply>); }
  fail(message: string): void { this.onerror?.({ message, preventDefault: () => {} } as ErrorEvent); }
  succeed(index: number): void {
    const message = this.requests[index]!;
    const prepared = prepareVtkRequest(message.request);
    const { values: _values, ...metadata } = prepared;
    this.onmessage?.({ data: {
      kind: "prepared",
      requestId: message.requestId,
      datasetId: message.datasetId,
      ok: true,
      metadata,
      bytes: prepared.values.byteLength,
    } } as MessageEvent<StreamlineWorkerReply>);
  }
  succeedExtraction(index: number, result: StreamlineExtractionResult): void {
    const message = this.extractions[index]!;
    this.onmessage?.({ data: {
      kind: "result",
      requestId: message.requestId,
      datasetId: message.datasetId,
      ok: true,
      result,
    } } as MessageEvent<StreamlineWorkerReply>);
  }
}

const shape = [3, 3, 3] as [number, number, number];
const field = (offset: number) => ({
  shape,
  data: Float32Array.from({ length: 27 }, (_, index) => index + offset),
});
const request: VectorPreparationRequest = {
  kind: "vector",
  fields: [field(0), field(1), field(2)],
  grid: { drM: 1, dyM: 1, dzM: 1 },
  maxVoxels: 27,
  datasetHandoff: "shared-geometry",
};
const baseline = getVisualizationResourceSnapshot();
const extractionParameters: StreamlineExtractionParameters = {
  seeds: new Float32Array([0, 0, 0]),
  integrationStep: 0.5,
  maxSteps: 10,
  tubeRadius: 1,
  scale: { xy: 1, z: 1 },
};
const extractionResult: StreamlineExtractionResult = {
  forward: { path: null, arrows: null },
  backward: { path: null, arrows: null },
  representation: "tubes",
  integratedPoints: 0,
  knownScratchBytes: 16,
  boundaryContacts: { lower: [false, false, false], upper: [false, false, false] },
  elapsedMs: 1,
};
const worker = new FakeWorker();
const service = new StreamlineWorkerOwnedPreparationService(() => worker);
const firstController = new AbortController();
const first = service.prepare(request, firstController.signal);
const second = service.prepare({ ...request });
worker.ready();
await flushHandoff();
check(worker.requests.length === 1, "identical vector preparations execute once in the combined worker");
check(
  request.fields.every((source, index) => source.data.byteLength > 0
    && worker.requests[0]!.request.fields[index].data.buffer !== source.data.buffer)
    && worker.transfers[0].length === 3,
  "vector handoff retains sources and transfers three staged component buffers",
);
firstController.abort();
check(await first.then(() => false, (error) => error instanceof DOMException && error.name === "AbortError"), "one remote-handle consumer cancels independently");
check(worker.terminated === 0, "one cancellation preserves the shared preparation worker");
worker.succeed(0);
const secondLease = await second;
check(secondLease.handle.residency === "worker", "combined preparation returns a worker-resident handle");
check(!("values" in secondLease.handle.metadata), "combined worker returns metadata without its lattice");
check(preparedDatasetRegistrySnapshot().workerHandles === 1, "remote preparation is visible in the handle registry");
check(
  getVisualizationResourceSnapshot().byStage["worker-resident"].residentBytes
    > baseline.byStage["worker-resident"].residentBytes,
  "combined worker owns the retained vector lattice",
);
check(
  (() => { try { resolvePreparedDataset(secondLease.handle); return false; } catch { return true; } })(),
  "worker-owned vector lattice cannot resolve on main",
);
const extractedFirst = service.extract(secondLease.handle, extractionParameters);
const extractedSecond = service.extract(secondLease.handle, {
  ...extractionParameters,
  seeds: extractionParameters.seeds.slice(),
});
check(worker.extractions.length === 1, "identical remote-handle extractions execute once");
check(worker.extractions[0]!.preparation == null, "warm remote extraction transfers no lattice or preparation request");
worker.succeedExtraction(0, extractionResult);
check(await extractedFirst === extractionResult && await extractedSecond === extractionResult, "remote consumers share extracted geometry");
secondLease.release();
service.shutdown();
check(worker.terminated === 1, "service shutdown terminates the combined worker");
check(preparedDatasetRegistrySnapshot().handles === 0, "remote lease teardown clears the registry");
check(
  getVisualizationResourceSnapshot().residentBytes === baseline.residentBytes,
  "combined worker teardown releases its resource reservation",
);

const failedWorker = new FakeWorker();
const retryWorker = new FakeWorker();
const retryWorkers = [failedWorker, retryWorker];
const retryService = new StreamlineWorkerOwnedPreparationService(() => retryWorkers.shift()!);
const retried = retryService.prepare(request);
failedWorker.ready();
failedWorker.fail("synthetic combined-worker loss");
await new Promise((resolve) => setTimeout(resolve, 30));
retryWorker.ready();
await flushHandoff();
check(retryWorker.requests.length === 1, "worker-owned preparation restarts once after worker loss");
retryWorker.succeed(0);
const retryLease = await retried;
check(retryLease.handle.residency === "worker", "restart reconstructs a worker-resident dataset");
retryLease.release();
retryService.shutdown();
check(
  getVisualizationResourceSnapshot().residentBytes === baseline.residentBytes,
  "restart teardown releases both failed and replacement reservations",
);

const preparedWorker = new FakeWorker();
const reconstructionWorker = new FakeWorker();
const reconstructionWorkers = [preparedWorker, reconstructionWorker];
const reconstructionService = new StreamlineWorkerOwnedPreparationService(
  () => reconstructionWorkers.shift()!,
);
const reconstructionLeasePromise = reconstructionService.prepare(request);
preparedWorker.ready();
await flushHandoff();
preparedWorker.succeed(0);
const reconstructionLease = await reconstructionLeasePromise;
preparedWorker.fail("worker lost after preparation");
const reconstructed = reconstructionService.extract(
  reconstructionLease.handle,
  extractionParameters,
);
reconstructionWorker.ready();
await flushHandoff();
check(
  reconstructionWorker.extractions.length === 1
    && reconstructionWorker.extractions[0]!.preparation != null,
  "extraction after worker loss carries the immutable preparation request",
);
reconstructionWorker.succeedExtraction(0, extractionResult);
check(await reconstructed === extractionResult, "extraction-time reconstruction publishes geometry");
reconstructionLease.release();
reconstructionService.shutdown();
check(
  getVisualizationResourceSnapshot().residentBytes === baseline.residentBytes,
  "reconstructed worker teardown returns to baseline",
);

console.log(`\nstreamlineWorkerOwnedPreparation tests: ${passed} passed, ${failures.length} failed.`);
if (failures.length) {
  console.log(failures.join("\n"));
  process.exit(1);
}
