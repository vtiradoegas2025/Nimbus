import {
  BrickedSurfaceCompositionClient,
} from "../../../../ui/src/lib/vtk/brickedSurfaceCompositionClient.ts";
import type {
  BrickedSurfaceCompositionWorkerReply,
  BrickedSurfaceCompositionWorkerRequest,
} from "../../../../ui/src/lib/vtk/brickedSurfaceCompositionProtocol.ts";
import { getVisualizationResourceSnapshot } from "../../../../ui/src/lib/core/visualizationResourceGovernor.ts";
import type { OwnedBrickSurface } from "../../../../ui/src/lib/vtk/vtkBrickedContour.ts";
import type { VortexSurfaceGeometry } from "../../../../ui/src/lib/vtk/vortexContourProtocol.ts";

let passed = 0;
const failures: string[] = [];
const check = (condition: boolean, label: string) => condition ? passed += 1 : failures.push(label);

function triangle(offset: number): VortexSurfaceGeometry {
  const points = new Float32Array([offset, 0, 0, offset + 1, 0, 0, offset, 1, 0]);
  const polys = new Uint32Array([3, 0, 1, 2]);
  const normals = new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]);
  return { points, polys, normals, bounds: [offset, offset + 1, 0, 1, 0, 0],
    triangles: 1, bytes: points.byteLength + polys.byteLength + normals.byteLength };
}

class FakeWorker {
  onmessage: ((event: MessageEvent<BrickedSurfaceCompositionWorkerReply>) => void) | null = null;
  onerror: ((event: ErrorEvent) => void) | null = null;
  hold = false;
  terminated = false;
  constructor() { queueMicrotask(() => this.reply({ kind: "ready" })); }
  postMessage(request: BrickedSurfaceCompositionWorkerRequest): void {
    if (request.kind === "close" || this.hold) return;
    this.reply({ kind: "result", requestId: request.requestId, identity: request.identity,
      ok: true, geometry: triangle(0) });
  }
  terminate(): void { this.terminated = true; }
  private reply(reply: BrickedSurfaceCompositionWorkerReply): void {
    queueMicrotask(() => this.onmessage?.({ data: reply } as MessageEvent<BrickedSurfaceCompositionWorkerReply>));
  }
}

const ownership = {
  ownedCellBounds: [-Infinity, Infinity, -Infinity, Infinity, -Infinity, Infinity],
  includeUpper: [true, true, true],
} as Pick<OwnedBrickSurface, "ownedCellBounds" | "includeUpper">;
const surfaces = [
  { brickId: "a", geometry: triangle(0), ...ownership },
  { brickId: "b", geometry: triangle(1), ...ownership },
];
const baseline = getVisualizationResourceSnapshot();
const heldWorker = new FakeWorker(); heldWorker.hold = true;
const heldClient = new BrickedSurfaceCompositionClient(heldWorker);
const controller = new AbortController();
const cancelled = heldClient.compose("held", "primary", surfaces, 1e-5, controller.signal);
await new Promise((resolve) => setTimeout(resolve, 0));
check(getVisualizationResourceSnapshot().byStage["retained-geometry"].reservedBytes
  === baseline.byStage["retained-geometry"].reservedBytes + 176,
"composition output is bounded by exact input geometry rather than a global ceiling");
check(getVisualizationResourceSnapshot().byStage["worker-scratch"].reservedBytes
  === baseline.byStage["worker-scratch"].reservedBytes + 352,
"deduplication scratch is admitted before transferring brick geometry");
controller.abort();
let sawAbort = false;
try { await cancelled; } catch (error) {
  sawAbort = error instanceof DOMException && error.name === "AbortError";
}
heldClient.shutdown();
check(sawAbort && heldWorker.terminated
  && getVisualizationResourceSnapshot().committedBytes === baseline.committedBytes,
"composition cancellation releases all output and scratch ownership");

const worker = new FakeWorker();
const client = new BrickedSurfaceCompositionClient(worker);
const geometry = await client.compose("complete", "primary", surfaces, 1e-5, new AbortController().signal);
check(geometry?.bytes === 88
  && getVisualizationResourceSnapshot().byStage["retained-geometry"].residentBytes
    === baseline.byStage["retained-geometry"].residentBytes + 88,
"composition transitions its envelope to exact merged geometry");
client.release("primary");
client.shutdown();
check(worker.terminated && getVisualizationResourceSnapshot().committedBytes === baseline.committedBytes,
"composition release and shutdown return the ledger to baseline");

console.log(`\nbricked surface composition client tests: ${passed} passed, ${failures.length} failed.`);
if (failures.length) {
  for (const failure of failures) console.error(`FAIL: ${failure}`);
  process.exitCode = 1;
}
