import type { PreparedVtkImage } from "../../../../ui/src/lib/vtk/vtkPreparation.ts";
import type { VortexContourResult, VortexContourWorkerReply, VortexContourWorkerRequest } from "../../../../ui/src/lib/vtk/vortexContourProtocol.ts";
import {
  SharedVortexContourDatasetService,
  type VortexContourExtractionWorker,
} from "../../../../ui/src/lib/vtk/vortexContourClient.ts";

let passed = 0;
let failed = 0;
const failures: string[] = [];
function check(condition: boolean, label: string): void {
  if (condition) passed += 1;
  else { failed += 1; failures.push(`FAIL ${label}`); }
}

class FakeWorker implements VortexContourExtractionWorker {
  onmessage: ((event: MessageEvent<VortexContourWorkerReply>) => void) | null = null;
  onerror: ((event: ErrorEvent) => void) | null = null;
  onmessageerror: ((event: MessageEvent) => void) | null = null;
  readonly requests: Array<Extract<VortexContourWorkerRequest, { kind: "extract" }>> = [];
  terminated = 0;
  postMessage(message: VortexContourWorkerRequest): void {
    if (message.kind === "extract") {
      this.requests.push(message);
      this.onmessage?.({ data: { kind: "admission", requestId: message.requestId,
        datasetId: message.datasetId, admission: { maximumTriangles: 40,
          maximumOutputBytes: 3520, maximumScratchBytes: 3520, crossingCells: 8 } }
      } as MessageEvent<VortexContourWorkerReply>);
    }
  }
  terminate(): void { this.terminated += 1; }
  ready(): void { this.onmessage?.({ data: { kind: "ready" } } as MessageEvent<VortexContourWorkerReply>); }
  fail(message: string): void { this.onerror?.({ message, preventDefault: () => {} } as ErrorEvent); }
  succeed(index: number, result: VortexContourResult): void {
    const request = this.requests[index]!;
    this.onmessage?.({ data: {
      kind: "result", requestId: request.requestId, datasetId: request.datasetId,
      ok: true, result,
    } } as MessageEvent<VortexContourWorkerReply>);
  }
}

function prepared(value = 1): PreparedVtkImage {
  return {
    shape: [2, 2, 2], spacing: [1, 1, 1], origin: [0, 0, 0],
    values: new Float32Array(8).fill(value), sourceNonFiniteCount: 0,
    samplingProvenance: {},
  } as unknown as PreparedVtkImage;
}
const result = {
  core: null, opposite: null, coreComponentCount: 0, oppositeComponentCount: 0,
  mesocyclone: null, mesocycloneAssessment: null, vortexAssessment: null,
  condensation: null, condensationBounds: null, condensationAssessment: null,
  usage: { triangles: 0, bytes: 0 }, knownScratchBytes: 16, elapsedMs: 1,
} satisfies VortexContourResult;
const parameters = {
  thresholds: { core: 1, opposite: -1, mesocyclone: null },
  anchor: [0, 0, 0] as [number, number, number],
  condensateFloor: null,
};

const diagnostic = prepared();
const condensate = prepared(2);
const worker = new FakeWorker();
const service = new SharedVortexContourDatasetService(diagnostic, condensate, () => worker);
const firstController = new AbortController();
const first = service.enqueue(parameters, firstController.signal);
const second = service.enqueue({ ...parameters, thresholds: { ...parameters.thresholds } });
worker.ready();
check(worker.requests.length === 1, "identical pane contours execute once");
check(worker.requests[0]!.image != null && worker.requests[0]!.condensateImage != null, "diagnostic and condensate pair load atomically");
firstController.abort();
check(await first.then(() => false, (error) => error instanceof DOMException && error.name === "AbortError"), "one pane cancels independently");
check(worker.terminated === 0, "one cancellation preserves another contour consumer");
worker.succeed(0, result);
check(await second === result, "remaining pane receives shared contour geometry");

const warm = service.enqueue({ ...parameters, thresholds: { ...parameters.thresholds, core: 2 } });
check(worker.requests.length === 2 && worker.requests[1]!.image == null && worker.requests[1]!.condensateImage == null, "threshold edit reuses both warm lattices");
worker.succeed(1, result);
await warm;
service.shutdown();
check(worker.terminated === 1, "dataset service shutdown terminates its worker");

const failedWorker = new FakeWorker();
const retryWorker = new FakeWorker();
const retryWorkers = [failedWorker, retryWorker];
const retryService = new SharedVortexContourDatasetService(diagnostic, null, () => retryWorkers.shift()!);
const retried = retryService.enqueue(parameters);
failedWorker.ready();
failedWorker.fail("synthetic contour worker loss");
await new Promise((resolve) => setTimeout(resolve, 30));
retryWorker.ready();
check(retryWorker.requests.length === 1 && retryWorker.requests[0]!.image != null, "worker restart reloads the exact dataset");
retryWorker.succeed(0, result);
check(await retried === result, "worker restart preserves the active contour consumer");
retryService.shutdown();

const cancelledWorker = new FakeWorker();
const cancellationService = new SharedVortexContourDatasetService(diagnostic, null, () => cancelledWorker);
const cancellation = new AbortController();
const abandoned = cancellationService.enqueue(parameters, cancellation.signal);
cancelledWorker.ready();
cancellation.abort();
check(await abandoned.then(() => false, (error) => error instanceof DOMException && error.name === "AbortError"), "last contour consumer cancels deterministically");
check(cancelledWorker.terminated === 1, "last contour consumer terminates synchronous worker work");
cancellationService.shutdown();

console.log(`\nvortexContourClient tests: ${passed} passed, ${failed} failed.`);
if (failures.length) {
  console.log(failures.join("\n"));
  process.exit(1);
}
