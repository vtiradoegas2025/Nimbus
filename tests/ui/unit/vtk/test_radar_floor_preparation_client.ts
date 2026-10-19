import {
  RadarFloorPreparationService,
  type RadarFloorPreparationWorker,
} from "../../../../ui/src/lib/vtk/radarFloorPreparationClient.ts";
import type {
  RadarFloorPreparationWorkerReply,
  RadarFloorPreparationWorkerRequest,
  RadarFloorTerrainInput,
} from "../../../../ui/src/lib/vtk/radarFloorPreparationProtocol.ts";
import { getVisualizationResourceSnapshot } from "../../../../ui/src/lib/core/visualizationResourceGovernor.ts";

let passed = 0;
const failures: string[] = [];
const check = (condition: boolean, label: string) => {
  if (condition) passed += 1;
  else failures.push(label);
};

class FakeWorker implements RadarFloorPreparationWorker {
  onmessage: ((event: MessageEvent<RadarFloorPreparationWorkerReply>) => void) | null = null;
  onerror: ((event: ErrorEvent) => void) | null = null;
  onmessageerror: ((event: MessageEvent) => void) | null = null;
  hold = false;
  terminated = false;

  constructor() {
    queueMicrotask(() => this.reply({ kind: "ready" }));
  }

  postMessage(request: RadarFloorPreparationWorkerRequest): void {
    if (request.kind !== "prepare" || this.hold) return;
    const cells = request.shape[0] * request.shape[1];
    const values = new Float32Array(cells).fill(4);
    const terrainMesh = request.terrain ? {
      points: new Float32Array(cells * 3).fill(1),
      polys: new Uint32Array((request.shape[0] - 1) * (request.shape[1] - 1) * 8),
      tcoords: new Float32Array(cells * 2),
    } : null;
    this.reply({
      kind: "result",
      requestId: request.requestId,
      ok: true,
      result: {
        values,
        stats: { min: 4, max: 4, p2: 4, p98: 4, positiveP98: 4 },
        terrainMesh,
        knownScratchBytes: values.byteLength + cells * 8,
        elapsedMs: 1,
      },
    });
  }

  terminate(): void {
    this.terminated = true;
  }

  private reply(reply: RadarFloorPreparationWorkerReply): void {
    queueMicrotask(() => this.onmessage?.({ data: reply } as MessageEvent<RadarFloorPreparationWorkerReply>));
  }
}

const cells = 4;
const source = new Float32Array(cells * 2).fill(4);
const terrain: RadarFloorTerrainInput = {
  centroids: {
    x: new Float64Array(cells).fill(1),
    y: new Float64Array(cells).fill(2),
    z: new Float64Array(cells).fill(12),
  },
  volumesM3: new Float64Array(cells).fill(4),
  sourceShape: [2, 2],
  strideXY: 1,
  dxM: 1,
  dyM: 1,
};
const baseline = getVisualizationResourceSnapshot();
const worker = new FakeWorker();
const service = new RadarFloorPreparationService(() => worker);
const prepared = await service.prepare(source, [2, 2, 2], "column-max", terrain);
let snapshot = getVisualizationResourceSnapshot();
const expectedOutputBytes = cells * (4 + 3 * 4 + 2 * 4) + 8 * 4;
check(prepared.values?.length === cells && prepared.terrainMesh?.points.length === cells * 3,
  "the service returns reduced values and terrain geometry");
check(snapshot.byStage["worker-resident"].committedBytes
  === baseline.byStage["worker-resident"].committedBytes,
"source and physical-geometry worker copies retire after preparation");
check(snapshot.byStage["prepared-main"].residentBytes
  === baseline.byStage["prepared-main"].residentBytes + expectedOutputBytes,
"the result transitions to exact reduced and terrain buffer ownership");
prepared.release();
service.shutdown();
check(worker.terminated
  && getVisualizationResourceSnapshot().committedBytes === baseline.committedBytes,
"release and shutdown retire preparation output and worker state");

const heldWorker = new FakeWorker();
heldWorker.hold = true;
const heldService = new RadarFloorPreparationService(() => heldWorker);
const controller = new AbortController();
const cancelled = heldService.prepare(source, [2, 2, 2], "column-max", terrain, controller.signal);
await new Promise((resolve) => setTimeout(resolve, 0));
snapshot = getVisualizationResourceSnapshot();
check(snapshot.byStage["worker-resident"].reservedBytes > 0
  && snapshot.byStage["worker-scratch"].reservedBytes >= cells * 4
  && snapshot.byStage["prepared-main"].reservedBytes >= expectedOutputBytes,
"input, percentile scratch, and complete output are admitted before execution");
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
"active cancellation terminates reduction and releases all reservations");

console.log(`\nradar floor preparation client tests: ${passed} passed, ${failures.length} failed.`);
if (failures.length) {
  for (const failure of failures) console.error(`FAIL: ${failure}`);
  process.exitCode = 1;
}
