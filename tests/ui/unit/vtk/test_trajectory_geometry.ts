import { buildTrajectoryTubeGeometry } from "../../../../ui/src/lib/vtk/trajectoryGeometry.ts";
import { mkdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import {
  TrajectoryGeometryService,
  type TrajectoryGeometryWorker,
} from "../../../../ui/src/lib/vtk/trajectoryGeometryClient.ts";
import {
  trajectoryGeometryUpperBounds,
  type TrajectoryGeometryWorkerReply,
  type TrajectoryGeometryWorkerRequest,
} from "../../../../ui/src/lib/vtk/trajectoryGeometryProtocol.ts";
import { getVisualizationResourceSnapshot } from "../../../../ui/src/lib/core/visualizationResourceGovernor.ts";

let passed = 0;
const failures: string[] = [];
function check(condition: boolean, label: string): void {
  if (condition) passed += 1;
  else failures.push(label);
}

const values = new Float32Array(2 * 3 * 9).fill(Number.NaN);
function sample(parcel: number, index: number, t: number, x: number, y: number, z: number): void {
  const offset = (parcel * 3 + index) * 9;
  values.set([t, x, y, z, z / 10, z / 100], offset);
}
sample(0, 0, 0, 0, 0, 10);
sample(0, 1, 1, 100, 0, 20);
sample(0, 2, 2, 200, 0, 30);
sample(1, 0, 0, 0, 100, 999);
const dataset = { shape: [2, 3, 9] as [number, number, number], values };
const result = buildTrajectoryTubeGeometry(dataset, {
  colorColumn: 3,
  tubeRadius: 1,
  scale: { xy: 1, z: 2 },
  maximumParcels: 400,
});
check(result.parcels === 1 && result.sourceSamples === 3,
  "only parcels with at least two finite samples are emitted");
check(JSON.stringify(result.timeRangeS) === "[0,2]", "trajectory metadata describes the whole finite recorded time interval");
const unorderedTimes = values.slice();
unorderedTimes[0] = 50; unorderedTimes[9] = -10; unorderedTimes[18] = 15;
unorderedTimes[27] = Number.NaN;
const unordered = buildTrajectoryTubeGeometry({ ...dataset, values: unorderedTimes }, {
  colorColumn: 3, tubeRadius: 1, scale: { xy: 1, z: 2 }, maximumParcels: 400,
});
check(JSON.stringify(unordered.timeRangeS) === "[-10,50]", "history interval handles unsorted times and missing time samples without inventing a selected frame");
check(unorderedTimes[0] === 50 && Number.isNaN(unorderedTimes[27]), "history interval reduction preserves exact source timestamps");
check(result.min === 10 && result.max === 30,
  "discarded parcels do not contaminate the colour range");
check(Boolean(result.geometry) && result.geometry!.points.length > 0,
  "the worker builder returns nonempty tube geometry");
check(result.geometry!.scalars.length === result.geometry!.points.length / 3,
  "tube output retains one named colour scalar per point");
check(result.geometry!.normals.length === result.geometry!.points.length,
  "tube output retains one normal tuple per point");
check(result.retainedBytes > 0 && result.knownScratchBytes > 0,
  "tube output and worker scratch are measured independently");
const upperBounds = trajectoryGeometryUpperBounds(dataset.shape, 400);
check(result.retainedBytes <= upperBounds.retainedBytes
  && result.knownScratchBytes <= upperBounds.scratchBytes,
"topology-derived output and scratch bounds cover the exact worker allocations");
check(values[3] === 10 && values[12] === 20,
  "worker geometry construction leaves source samples unchanged");
check(result.knownScratchBytes < result.retainedBytes,
  "dedicated VTK tube output buffers become retained geometry without a second full output copy");
check(Object.values(result.geometry!).every(view => view.buffer !== values.buffer),
  "retained tube buffers never alias the borrowed trajectory source");

const fragmentedValues = new Float32Array(7 * 9).fill(Number.NaN);
for (const [sampleIndex, x, z, color] of [[0, 0, 10, 10], [1, 100, 20, 20], [3, 500, 500, 999], [5, 1000, 30, 30], [6, 1100, 40, 40]]) {
  fragmentedValues.set([sampleIndex, x, 0, z, color, 0], sampleIndex * 9);
}
const fragmented = buildTrajectoryTubeGeometry({ shape: [1, 7, 9], values: fragmentedValues }, {
  colorColumn: 4, tubeRadius: 1, scale: { xy: 1, z: 1 }, maximumParcels: 1,
});
check(fragmented.parcels === 1 && fragmented.sourceSamples === 4 && fragmented.min === 10 && fragmented.max === 40,
  "missing trajectory samples split runs and isolated samples do not alter visible counts or ranges");
let crossings = 0;
for (const cells of [fragmented.geometry!.lines, fragmented.geometry!.polys, fragmented.geometry!.strips]) {
  for (let cursor = 0; cursor < cells.length;) {
    const count = cells[cursor++];
    let low = Infinity, high = -Infinity;
    for (let vertex = 0; vertex < count; vertex++) {
      const x = fragmented.geometry!.points[cells[cursor++] * 3];
      low = Math.min(low, x); high = Math.max(high, x);
    }
    if (low < 200 && high > 900) crossings++;
  }
}
check(crossings === 0, "trajectory tube cells never bridge missing coordinate/colour support");
const fragmentedBounds = trajectoryGeometryUpperBounds([1, 7, 9], 1);
check(fragmented.retainedBytes <= fragmentedBounds.retainedBytes && fragmented.knownScratchBytes <= fragmentedBounds.scratchBytes,
  "segmented tube caps remain covered by conservative output and scratch admission");
const missingColor = fragmentedValues.slice();
missingColor[1 * 9 + 4] = Number.NaN;
missingColor[6 * 9 + 4] = Number.NaN;
const isolated = buildTrajectoryTubeGeometry({ shape: [1, 7, 9], values: missingColor }, {
  colorColumn: 4, tubeRadius: 1, scale: { xy: 1, z: 1 }, maximumParcels: 1,
});
check(isolated.geometry === null && isolated.parcels === 0 && isolated.min === null && isolated.max === null,
  "isolated finite records across invalid colour samples never become a fabricated path");

class FakeWorker implements TrajectoryGeometryWorker {
  onmessage: ((event: MessageEvent<TrajectoryGeometryWorkerReply>) => void) | null = null;
  onerror: ((event: ErrorEvent) => void) | null = null;
  onmessageerror: ((event: MessageEvent) => void) | null = null;
  requests: TrajectoryGeometryWorkerRequest[] = [];
  hold = false;
  terminated = false;

  constructor() {
    queueMicrotask(() => this.reply({ kind: "ready" }));
  }

  postMessage(request: TrajectoryGeometryWorkerRequest): void {
    this.requests.push(request);
    if (request.kind !== "extract" || this.hold) return;
    const geometry = {
      points: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
      lines: new Uint32Array(),
      polys: new Uint32Array([3, 0, 1, 2]),
      strips: new Uint32Array(),
      normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
      scalars: new Float32Array([0, 1, 2]),
    };
    const retainedBytes = Object.values(geometry)
      .reduce((sum, array) => sum + array.byteLength, 0);
    this.reply({
      kind: "result",
      requestId: request.requestId,
      datasetId: request.datasetId,
      ok: true,
      result: {
        geometry,
        min: 0,
        max: 2,
        parcels: 1,
        sourceSamples: 3,
        retainedBytes,
        knownScratchBytes: 64,
        elapsedMs: 1,
      },
    });
  }

  terminate(): void {
    this.terminated = true;
  }

  private reply(reply: TrajectoryGeometryWorkerReply): void {
    queueMicrotask(() => this.onmessage?.({ data: reply } as MessageEvent<TrajectoryGeometryWorkerReply>));
  }
}

const baseline = getVisualizationResourceSnapshot();
const worker = new FakeWorker();
const service = new TrajectoryGeometryService(() => worker);
const trajectory = { shape: [2, 3, 9], data: values };
const first = await service.extract(trajectory, {
  colorColumn: 3,
  tubeRadius: 1,
  scale: { xy: 1, z: 1 },
  maximumParcels: 400,
}, 4096, 4096);
let snapshot = getVisualizationResourceSnapshot();
check(snapshot.byStage["worker-resident"].residentBytes
  === baseline.byStage["worker-resident"].residentBytes + values.byteLength,
"the service retains and accounts for one worker-owned source dataset");
check(snapshot.byStage["retained-geometry"].residentBytes
  === baseline.byStage["retained-geometry"].residentBytes + first.retainedBytes,
"the output envelope transitions to exact retained geometry");
first.release();
const second = await service.extract(trajectory, {
  colorColumn: 4,
  tubeRadius: 2,
  scale: { xy: 1, z: 2 },
  maximumParcels: 400,
}, 4096, 4096);
const extracts = worker.requests.filter((request) => request.kind === "extract");
check(extracts.length === 2 && extracts[0]!.dataset != null && extracts[1]!.dataset == null,
  "parameter edits reuse the warm worker dataset without another source copy");
second.release();
service.shutdown();
snapshot = getVisualizationResourceSnapshot();
check(worker.terminated && snapshot.committedBytes === baseline.committedBytes,
  "release and shutdown retire output, worker source, and worker process ownership");

const spanValues = new Float32Array(new SharedArrayBuffer(108 * Float32Array.BYTES_PER_ELEMENT));
spanValues.set(values, 0);
spanValues.set(Float32Array.from(values, value => Number.isFinite(value) ? value + 100 : value), 54);
const spanWorker = new FakeWorker();
const spanService = new TrajectoryGeometryService(() => spanWorker);
const spanInputs = [
  { shape: [2, 3, 9], data: spanValues.subarray(0, 54) },
  { shape: [2, 3, 9], data: spanValues.subarray(54) },
  { shape: [3, 2, 9], data: spanValues.subarray(54) },
  { shape: [3, 2, 9], data: new Float32Array(spanValues.buffer, 54 * 4, 54) },
  { shape: [1, 3, 9], data: spanValues.subarray(54, 81) },
];
for (const input of spanInputs) {
  const geometry = await spanService.extract(input, { colorColumn: 3, tubeRadius: 1, scale: { xy: 1, z: 1 }, maximumParcels: 400 }, 4096, 4096);
  geometry.release();
}
const spanRequests = spanWorker.requests.filter(request => request.kind === "extract");
check(spanRequests.length === 5 && spanRequests.slice(0, 3).every(request => request.dataset != null)
  && spanRequests[4].dataset != null, "trajectory aliases with changed offsets, shapes, or lengths hand off the exact new dataset");
check(new Set([spanRequests[0].datasetId, spanRequests[1].datasetId, spanRequests[2].datasetId, spanRequests[4].datasetId]).size === 4,
  "different trajectory spans and parcel layouts never share a worker dataset identifier");
check(spanRequests[1].dataset?.values[3] === 110 && spanRequests[2].dataset?.shape.join(":") === "3:2:9",
  "trajectory handoff preserves the selected span values and shape");
check(spanRequests[3].dataset == null && spanRequests[3].datasetId === spanRequests[2].datasetId,
  "an equivalent immutable trajectory view retains zero-copy warm reuse");
spanService.shutdown();
check(spanValues.byteLength === 432 && getVisualizationResourceSnapshot().committedBytes === baseline.committedBytes,
  "shared alias source stays intact and all worker geometry ownership retires");

const heldWorker = new FakeWorker();
heldWorker.hold = true;
const heldService = new TrajectoryGeometryService(() => heldWorker);
const controller = new AbortController();
const cancelled = heldService.extract(trajectory, {
  colorColumn: 3,
  tubeRadius: 1,
  scale: { xy: 1, z: 1 },
  maximumParcels: 400,
}, 4096, 4096, controller.signal);
await new Promise((resolve) => setTimeout(resolve, 0));
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
"active cancellation terminates uninterruptible work and releases every reservation");

console.log(`\ntrajectory geometry tests: ${passed} passed, ${failures.length} failed.`);
if (process.env.NIMBUS_3D_AUDIT_OUTPUT && failures.length === 0) {
  mkdirSync(process.env.NIMBUS_3D_AUDIT_OUTPUT, { recursive: true });
  writeFileSync(join(process.env.NIMBUS_3D_AUDIT_OUTPUT, "trajectory-geometry.json"), JSON.stringify({
    assertions: passed, sourceRetained: values[3] === 10 && values[12] === 20,
    outputOwnedWithoutDuplicateCopy: result.knownScratchBytes < result.retainedBytes,
    retainedBytes: result.retainedBytes, knownScratchBytes: result.knownScratchBytes,
    gapCase: { parcels: fragmented.parcels, sourceSamples: fragmented.sourceSamples, min: fragmented.min, max: fragmented.max, crossings,
      retainedBytes: fragmented.retainedBytes, knownScratchBytes: fragmented.knownScratchBytes, upperBounds: fragmentedBounds },
    aliasViews: spanRequests.map(request => ({ datasetId: request.datasetId, handedOff: request.dataset != null,
      shape: request.dataset?.shape ?? null, byteOffset: request.dataset?.values.byteOffset ?? null, byteLength: request.dataset?.values.byteLength ?? null })),
  }, null, 2));
}
if (failures.length) {
  for (const failure of failures) console.error(`FAIL: ${failure}`);
  process.exitCode = 1;
}
