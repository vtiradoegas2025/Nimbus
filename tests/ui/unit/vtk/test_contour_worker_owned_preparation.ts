import { getVisualizationResourceSnapshot } from "../../../../ui/src/lib/core/visualizationResourceGovernor.ts";
import { prepareVtkRequest, type ScalarPreparationRequest } from "../../../../ui/src/lib/vtk/vtkPreparation.ts";
import type { VortexContourResult, VortexContourWorkerReply, VortexContourWorkerRequest } from "../../../../ui/src/lib/vtk/vortexContourProtocol.ts";
import {
  ContourWorkerOwnedPreparationService,
  type ContourOwnedPreparationWorker,
  workerOwnedContourProgressSnapshot,
} from "../../../../ui/src/lib/vtk/contourWorkerOwnedPreparation.ts";
import { preparedDatasetRegistrySnapshot, resolvePreparedDataset } from "../../../../ui/src/lib/vtk/vtkPreparedDatasetRegistry.ts";

let passed = 0;
const failures: string[] = [];
function check(condition: boolean, label: string): void {
  if (condition) passed += 1;
  else failures.push(`FAIL ${label}`);
}
const flushHandoff = () => new Promise<void>((resolve) => setTimeout(resolve, 0));
class FakeWorker implements ContourOwnedPreparationWorker {
  onmessage: ((event: MessageEvent<VortexContourWorkerReply>) => void) | null = null;
  onerror: ((event: ErrorEvent) => void) | null = null;
  onmessageerror: ((event: MessageEvent) => void) | null = null;
  requests: Array<Extract<VortexContourWorkerRequest, { kind: "prepare" }>> = [];
  extractions: Array<Extract<VortexContourWorkerRequest, { kind: "extract" }>> = [];
  transfers: Transferable[][] = [];
  terminated = 0;
  autoRelease = true;
  releases: Array<Extract<VortexContourWorkerRequest, { kind: "release" }>> = [];
  postMessage(message: VortexContourWorkerRequest, transfer: Transferable[] = []): void {
    this.transfers.push(transfer);
    if (message.kind === "release") {
      this.releases.push(message);
      if (this.autoRelease) this.release(this.releases.length - 1);
    }
    if (message.kind === "prepare") this.requests.push(message);
    if (message.kind === "extract") {
      this.extractions.push(message);
      this.onmessage?.({ data: { kind: "admission", requestId: message.requestId,
        datasetId: message.datasetId, admission: { maximumTriangles: 135,
          maximumOutputBytes: 11_880, maximumScratchBytes: 11_880, crossingCells: 27 } }
      } as MessageEvent<VortexContourWorkerReply>);
    }
  }
  terminate(): void { this.terminated += 1; }
  release(index: number): void {
    this.onmessage?.({ data: { kind: "released", requestId: this.releases[index].requestId } } as MessageEvent<VortexContourWorkerReply>);
  }
  ready(): void { this.onmessage?.({ data: { kind: "ready" } } as MessageEvent<VortexContourWorkerReply>); }
  fail(message: string): void { this.onerror?.({ message, preventDefault: () => {} } as ErrorEvent); }
  succeed(index: number): void {
    const message = this.requests[index]!;
    const prepared = prepareVtkRequest(message.request);
    const { values: _values, ...metadata } = prepared;
    this.onmessage?.({ data: {
      kind: "prepared", requestId: message.requestId, datasetId: message.datasetId,
      ok: true, metadata, bytes: prepared.values.byteLength,
    } } as MessageEvent<VortexContourWorkerReply>);
  }
  succeedExtraction(index: number, result: VortexContourResult): void {
    const message = this.extractions[index]!;
    this.onmessage?.({ data: {
      kind: "result", requestId: message.requestId, datasetId: message.datasetId,
      ok: true, result,
    } } as MessageEvent<VortexContourWorkerReply>);
  }
}
const request: ScalarPreparationRequest = {
  kind: "scalar",
  field: {
    shape: [3, 3, 3],
    data: Float32Array.from({ length: 27 }, (_, index) => index - 13),
  },
  grid: { drM: 1, dyM: 1, dzM: 1 },
  maxVoxels: 27,
  featureSign: 1,
  datasetHandoff: "shared-geometry",
};
const baseline = getVisualizationResourceSnapshot();
const contourResult = {
  core: null, opposite: null, coreComponentCount: 0, oppositeComponentCount: 0,
  mesocyclone: null, mesocycloneAssessment: null, vortexAssessment: null,
  condensation: null, condensationBounds: null, condensationAssessment: null,
  usage: { triangles: 0, bytes: 0 }, knownScratchBytes: 16, elapsedMs: 1,
} satisfies VortexContourResult;
const worker = new FakeWorker();
const service = new ContourWorkerOwnedPreparationService(() => worker);
const controller = new AbortController();
const first = service.prepare(request, controller.signal);
const second = service.prepare({ ...request });
worker.ready();
await flushHandoff();
check(worker.requests.length === 1, "identical scalar preparations execute once in contour worker");
check(
  request.field.data.byteLength > 0
    && worker.requests[0]!.request.field.data.buffer !== request.field.data.buffer
    && worker.transfers[0].length === 1,
  "scalar handoff retains its source and transfers one staged field buffer",
);
controller.abort();
check(await first.then(() => false, (error) => error instanceof DOMException && error.name === "AbortError"), "one scalar consumer cancels independently");
check(worker.terminated === 0, "one cancellation preserves the shared contour worker");
worker.succeed(0);
const lease = await second;
check(lease.handle.residency === "worker", "contour preparation returns worker-resident handle");
check(!("values" in lease.handle.metadata), "contour worker returns metadata without lattice");
check(preparedDatasetRegistrySnapshot().workerHandles === 1, "remote scalar is visible in registry");
check(
  getVisualizationResourceSnapshot().byStage["worker-resident"].residentBytes
    > baseline.byStage["worker-resident"].residentBytes,
  "contour worker owns prepared scalar residency",
);
check(
  (() => { try { resolvePreparedDataset(lease.handle); return false; } catch { return true; } })(),
  "remote scalar cannot resolve on main",
);
const extraction = service.extract(lease.handle, null, {
  thresholds: { core: 0, opposite: null, mesocyclone: null },
  anchor: null,
  condensateFloor: null,
});
await flushHandoff();
check(worker.extractions.length === 1, "remote scalar extraction uses its preparation worker");
check(
  worker.extractions[0]!.preparation == null
    && worker.extractions[0]!.datasetId === worker.requests[0]!.datasetId,
  "first contour extraction reuses its resident prepared field",
);
check(
  getVisualizationResourceSnapshot().byStage["worker-resident"].reservedBytes
    === baseline.byStage["worker-resident"].reservedBytes,
  "resident field extraction needs no duplicate input or preparation reservation",
);
check(
  getVisualizationResourceSnapshot().byStage["retained-geometry"].reservedBytes
    >= baseline.byStage["retained-geometry"].reservedBytes + 27 * 5 * 88,
  "worker-owned extraction reserves its topology-derived output before dispatch",
);
check(
  getVisualizationResourceSnapshot().byStage["worker-scratch"].reservedBytes
    >= baseline.byStage["worker-scratch"].reservedBytes + 27 * 5 * 88,
  "worker-owned extraction reserves scratch before dispatch",
);
worker.succeedExtraction(0, contourResult);
check(await extraction === contourResult, "remote scalar extraction publishes geometry");
const cachedExtraction = await service.extract(lease.handle, null, {
  thresholds: { core: 0, opposite: null, mesocyclone: null },
  anchor: null,
  condensateFloor: null,
});
check(cachedExtraction === contourResult, "equivalent generation adopts cached completed geometry");
check(worker.extractions.length === 1, "cached completed geometry avoids duplicate worker extraction");
const thresholdEdit = service.extract(lease.handle, null, {
  thresholds: { core: 1, opposite: null, mesocyclone: null },
  anchor: null,
  condensateFloor: null,
});
check(worker.extractions.length === 2, "changed threshold schedules another extraction");
check(
  worker.extractions[1]!.preparation == null
    && worker.extractions[1]!.datasetId === worker.extractions[0]!.datasetId,
  "threshold edit reuses the immutable loaded pair identity",
);
worker.succeedExtraction(1, contourResult);
await thresholdEdit;
worker.autoRelease = false;
const nextFrame = service.prepare({
  ...request,
  field: { ...request.field, data: request.field.data.map((value) => value + 2) },
});
await flushHandoff();
check(worker.requests.length === 1, "replacement waits for the worker to acknowledge old-field release");
check(getVisualizationResourceSnapshot().byStage["worker-resident"].reservedBytes === 0,
  "replacement staging does not overlap the displaced field while release is pending");
worker.release(0);
worker.autoRelease = true;
await flushHandoff();
worker.succeed(1);
const nextLease = await nextFrame;
const parameters = {
  thresholds: { core: 1, opposite: null, mesocyclone: null },
  anchor: null,
  condensateFloor: null,
};
const nextExtraction = service.extract(nextLease.handle, null, parameters);
check(worker.extractions.length === 3, "same-shape new scalar values invalidate completed contours");
check(
  worker.extractions[2]?.preparation == null
    && worker.extractions[2]?.datasetId === worker.requests[1].datasetId,
  "new frame extraction reuses only the newly prepared dataset",
);
worker.succeedExtraction(2, { ...contourResult, elapsedMs: 2 });
check((await nextExtraction).elapsedMs === 2, "new frame publishes its own contour result");
const previousExtraction = service.extract(lease.handle, null, parameters);
await flushHandoff();
check(
  worker.extractions[3]?.preparation?.field.data[0] === request.field.data[0],
  "returning to an evicted frame rebuilds its original scalar input",
);
worker.succeedExtraction(3, contourResult);
await previousExtraction;
const rebuiltLease = await service.prepare(request);
check(worker.requests.length === 2, "a contour-rebuilt field satisfies later preparation without another copy");
check(rebuiltLease.handle.id === lease.handle.id, "a second pane shares the contour-rebuilt field handle");
const displacedAgain = service.prepare({
  ...request,
  field: { ...request.field, data: request.field.data.map((value) => value + 4) },
});
await flushHandoff();
worker.succeed(2);
const displacedAgainLease = await displacedAgain;
const revisited = service.prepare(request);
await flushHandoff();
worker.succeed(3);
const revisitedLease = await revisited;
check(worker.requests[3].datasetId !== worker.requests[0].datasetId,
  "re-preparing a displaced scalar assigns a new worker dataset identity");
const revisitedExtraction = service.extract(revisitedLease.handle, null, {
  ...parameters, thresholds: { ...parameters.thresholds, core: 2 },
});
await flushHandoff();
const revisitedMessage = worker.extractions[4];
check(revisitedMessage.preparation != null
    || revisitedMessage.datasetId === worker.requests[3].datasetId,
  "revisited scalar either rebuilds its cached pair identity or names the actual resident dataset");
worker.succeedExtraction(4, contourResult);
check(await revisitedExtraction === contourResult, "revisited scalar completes its changed-threshold extraction");
const oldLeaseExtraction = service.extract(lease.handle, null, {
  ...parameters, thresholds: { ...parameters.thresholds, core: 3 },
});
await flushHandoff();
check(worker.extractions[5].preparation == null
    && worker.extractions[5].datasetId === revisitedMessage.datasetId,
  "an older equivalent lease reuses the verified rebuilt worker dataset");
worker.succeedExtraction(5, contourResult);
await oldLeaseExtraction;
revisitedLease.release();
displacedAgainLease.release();
rebuiltLease.release();
nextLease.release();
lease.release();
service.shutdown();
check(worker.terminated === 1, "contour preparation service shuts down worker");
check(preparedDatasetRegistrySnapshot().handles === 0, "scalar remote lease leaves no registry entry");
check(getVisualizationResourceSnapshot().residentBytes === baseline.residentBytes, "scalar worker teardown returns to baseline");

const collisionWorkerA = new FakeWorker();
const collisionWorkerB = new FakeWorker();
const collisionServiceA = new ContourWorkerOwnedPreparationService(
  () => collisionWorkerA,
  "collision-service-a",
);
const collisionServiceB = new ContourWorkerOwnedPreparationService(
  () => collisionWorkerB,
  "collision-service-b",
);
const largerRequest: ScalarPreparationRequest = {
  ...request,
  field: {
    shape: [4, 4, 4],
    data: Float32Array.from({ length: 64 }, (_, index) => index - 31),
  },
  maxVoxels: 64,
};
const collisionLeasePromiseA = collisionServiceA.prepare(request);
const collisionLeasePromiseB = collisionServiceB.prepare(largerRequest);
collisionWorkerA.ready();
collisionWorkerB.ready();
await flushHandoff();
collisionWorkerA.succeed(0);
collisionWorkerB.succeed(0);
const [collisionLeaseA, collisionLeaseB] = await Promise.all([
  collisionLeasePromiseA,
  collisionLeasePromiseB,
]);
check(
  getVisualizationResourceSnapshot().byStage["worker-resident"].residentBytes
    - baseline.byStage["worker-resident"].residentBytes === (27 + 64) * Float32Array.BYTES_PER_ELEMENT,
  "independent contour services account local dataset ids without global collisions",
);
const collisionExtractionA = collisionServiceA.extract(collisionLeaseA.handle, null, {
  thresholds: { core: 0, opposite: null, mesocyclone: null },
  anchor: null,
  condensateFloor: null,
});
const collisionExtractionB = collisionServiceB.extract(collisionLeaseB.handle, null, {
  thresholds: { core: 0, opposite: null, mesocyclone: null },
  anchor: null,
  condensateFloor: null,
});
await flushHandoff();
collisionWorkerA.succeedExtraction(0, contourResult);
collisionWorkerB.succeedExtraction(0, contourResult);
check(
  (await Promise.all([collisionExtractionA, collisionExtractionB])).every((result) => result === contourResult),
  "independent contour services resolve colliding local extraction ids",
);
check(
  getVisualizationResourceSnapshot().byStage["worker-resident"].reservedBytes === 0,
  "independent contour extraction transitions release every capacity reservation",
);

const failedPublication = collisionServiceA.extract(collisionLeaseA.handle, null, {
  thresholds: { core: 2, opposite: null, mesocyclone: null },
  anchor: null,
  condensateFloor: null,
});
const invalidResult = Object.defineProperty({ ...contourResult }, "core", {
  get(): never { throw new Error("invalid contour publication"); },
}) as VortexContourResult;
collisionWorkerA.succeedExtraction(1, invalidResult);
check(
  await failedPublication.then(
    () => false,
    (error) => error instanceof Error && error.message === "invalid contour publication",
  ),
  "contour result publication failures reject consumers instead of stranding loading state",
);
check(
  workerOwnedContourProgressSnapshot()[`collision-service-a:${collisionWorkerA.extractions[1]!.datasetId}`] === "service-failed",
  "contour publication failures retain namespace-specific lifecycle evidence",
);
const pairedExtraction = collisionServiceA.extract(collisionLeaseA.handle, collisionLeaseB.handle, parameters);
await flushHandoff();
check(
  collisionWorkerA.extractions[2]?.preparation != null
    && collisionWorkerA.extractions[2]?.condensatePreparation != null,
  "adding a condensate dataset rebuilds the complete requested pair",
);
check(
  getVisualizationResourceSnapshot().byStage["worker-resident"].reservedBytes === 0
    && getVisualizationResourceSnapshot().byStage["worker-resident"].residentBytes
      - baseline.byStage["worker-resident"].residentBytes === (27 + 64 + 64) * 4,
  "rebuild acknowledgement retires staging and old residency before contour completion",
);
collisionWorkerA.succeedExtraction(2, contourResult);
await pairedExtraction;
collisionLeaseA.release();
collisionLeaseB.release();
collisionServiceA.shutdown();
collisionServiceB.shutdown();
check(
  getVisualizationResourceSnapshot().residentBytes === baseline.residentBytes
    && getVisualizationResourceSnapshot().reservedBytes === baseline.reservedBytes,
  "namespaced contour service teardown returns resource accounting to baseline",
);
check(
  !Object.keys(workerOwnedContourProgressSnapshot()).some((key) => key.startsWith("collision-service-")),
  "contour service teardown evicts its lifecycle evidence",
);

const retryWorkers = [new FakeWorker(), new FakeWorker()];
let retryWorkerIndex = 0;
const retryService = new ContourWorkerOwnedPreparationService(() => retryWorkers[retryWorkerIndex++]);
const retryPreparation = retryService.prepare(request);
retryWorkers[0].ready();
await flushHandoff();
retryWorkers[0].succeed(0);
const retryLease = await retryPreparation;
const retryExtraction = retryService.extract(retryLease.handle, null, parameters);
retryWorkers[0].fail("lost worker");
await new Promise((resolve) => setTimeout(resolve, 35));
retryWorkers[1].ready();
await flushHandoff();
check(retryWorkers[1].extractions[0]?.preparation != null, "worker restart rebuilds the lost resident field");
retryWorkers[1].succeedExtraction(0, contourResult);
check(await retryExtraction === contourResult, "restarted extraction resolves its original consumers");
retryLease.release();
retryService.shutdown();
check(
  getVisualizationResourceSnapshot().residentBytes === baseline.residentBytes
    && getVisualizationResourceSnapshot().reservedBytes === baseline.reservedBytes,
  "restarted contour worker releases resident and staged allocations",
);

console.log(`\ncontourWorkerOwnedPreparation tests: ${passed} passed, ${failures.length} failed.`);
if (failures.length) {
  console.log(failures.join("\n"));
  process.exit(1);
}
