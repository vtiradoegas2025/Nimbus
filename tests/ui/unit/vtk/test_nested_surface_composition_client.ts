import { NestedSurfaceCompositionClient } from "../../../../ui/src/lib/vtk/nestedSurfaceCompositionClient.ts";
import type { NestedSurfaceCompositionWorkerReply, NestedSurfaceCompositionWorkerRequest } from "../../../../ui/src/lib/vtk/nestedSurfaceCompositionProtocol.ts";
import { getVisualizationResourceSnapshot } from "../../../../ui/src/lib/core/visualizationResourceGovernor.ts";
import type { VortexSurfaceGeometry } from "../../../../ui/src/lib/vtk/vortexContourProtocol.ts";

let passed = 0;
const failures: string[] = [];
const check = (condition: boolean, label: string) => condition ? passed += 1 : failures.push(label);

function triangle(offset: number): VortexSurfaceGeometry {
  const points = new Float32Array([offset, 0, 0, offset + 1, 0, 0, offset, 1, 0]);
  const polys = new Uint32Array([3, 0, 1, 2]);
  const normals = new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]);
  return { points, polys, normals, bounds: [offset, offset + 1, 0, 1, 0, 0], triangles: 1,
    bytes: points.byteLength + polys.byteLength + normals.byteLength };
}

class FakeWorker {
  onmessage: ((event: MessageEvent<NestedSurfaceCompositionWorkerReply>) => void) | null = null;
  onerror: ((event: ErrorEvent) => void) | null = null;
  requests: string[] = [];
  hold = false;
  terminated = false;
  constructor() { queueMicrotask(() => this.reply({ kind: "ready" })); }
  postMessage(request: NestedSurfaceCompositionWorkerRequest): void {
    this.requests.push(request.kind === "layer" ? `layer-${request.layerIndex}` : request.kind);
    if (this.hold || request.kind === "close") return;
    if (request.kind === "begin") {
      this.reply({ kind: "ack", requestId: request.requestId, identity: request.identity, nextLayerIndex: 0 });
    } else if (request.kind === "layer") {
      this.reply({ kind: "ack", requestId: request.requestId, identity: request.identity,
        nextLayerIndex: request.layerIndex + 1 });
    } else {
      this.reply({ kind: "result", requestId: request.requestId, identity: request.identity, ok: true,
        primary: { geometry: triangle(0), transitionTriangles: 2 },
        opposite: { geometry: null, transitionTriangles: 0 }, knownScratchBytes: 84 });
    }
  }
  terminate(): void { this.terminated = true; }
  private reply(reply: NestedSurfaceCompositionWorkerReply): void {
    queueMicrotask(() => this.onmessage?.({ data: reply } as MessageEvent<NestedSurfaceCompositionWorkerReply>));
  }
}

const bounds = { lower: [0, 0, 0] as [number, number, number], upper: [1, 1, 1] as [number, number, number] };
const baseline = getVisualizationResourceSnapshot();
const worker = new FakeWorker();
const client = new NestedSurfaceCompositionClient(worker);
const result = await client.compose("generation-a", {
  geometries: [triangle(-1), triangle(0)], childBounds: [bounds, null],
}, { geometries: [null, null], childBounds: [bounds, null] }, new AbortController().signal);
check(worker.requests.slice(0, 4).join(",") === "begin,layer-0,layer-1,finish",
  "composition streams one acknowledged layer at a time");
check(result.primary.geometry?.triangles === 1 && result.primary.transitionTriangles === 2,
  "composition publishes the final worker geometry and transition count");
check(getVisualizationResourceSnapshot().byStage["retained-geometry"].residentBytes
  === baseline.byStage["retained-geometry"].residentBytes + result.primary.geometry!.bytes,
"output reservation transitions to exact retained geometry");
client.shutdown();
check(worker.terminated && getVisualizationResourceSnapshot().committedBytes === baseline.committedBytes,
  "shutdown terminates the worker and releases exact output ownership");

const heldWorker = new FakeWorker(); heldWorker.hold = true;
const heldClient = new NestedSurfaceCompositionClient(heldWorker);
const controller = new AbortController();
const cancelled = heldClient.compose("generation-b", {
  geometries: [triangle(-1), triangle(0)], childBounds: [bounds, null],
}, { geometries: [null, null], childBounds: [bounds, null] }, controller.signal);
await new Promise((resolve) => setTimeout(resolve, 0));
check(getVisualizationResourceSnapshot().byStage["retained-geometry"].reservedBytes
  === baseline.byStage["retained-geometry"].reservedBytes + 2 * 61 * 88,
"nested output is reserved from input topology rather than a fixed global ceiling");
check(getVisualizationResourceSnapshot().byStage["worker-scratch"].reservedBytes
  === baseline.byStage["worker-scratch"].reservedBytes + 2 * 88 + 2 * 61 * 88,
"nested clipping and output scratch are admitted before the first streamed layer");
controller.abort();
let cancellationObserved = false;
try { await cancelled; } catch (error) {
  cancellationObserved = error instanceof DOMException && error.name === "AbortError";
}
check(cancellationObserved && getVisualizationResourceSnapshot().committedBytes === baseline.committedBytes,
  "cancellation releases pre-dispatch hierarchy reservations");
heldClient.shutdown();

console.log(`\nnested surface composition client tests: ${passed} passed, ${failures.length} failed.`);
if (failures.length) {
  failures.forEach((failure) => console.error(`FAIL: ${failure}`));
  process.exitCode = 1;
}
