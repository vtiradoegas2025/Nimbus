import { getVisualizationResourceSnapshot } from "../../../../ui/src/lib/core/visualizationResourceGovernor.ts";
import { VtkWorkerDatasetOwnership } from "../../../../ui/src/lib/vtk/vtkWorkerDatasetOwnership.ts";

let passed = 0;
const failures: string[] = [];
function check(condition: boolean, label: string): void {
  if (condition) passed += 1;
  else failures.push(`FAIL ${label}`);
}

const baseline = getVisualizationResourceSnapshot();
const owner = new VtkWorkerDatasetOwnership("streamline-worker-test");
const first = new ArrayBuffer(120);
owner.begin(1, [{ buffer: first, sharedWithMain: false }]);
check(
  getVisualizationResourceSnapshot().byStage["worker-resident"].residentBytes
    === baseline.byStage["worker-resident"].residentBytes + 120,
  "pending transferable dataset reports worker-resident bytes",
);
owner.commit(1);
check(
  getVisualizationResourceSnapshot().residentBytes === baseline.residentBytes + 120,
  "committed worker dataset remains resident",
);

const replacement = new ArrayBuffer(80);
owner.begin(2, [{ buffer: replacement, sharedWithMain: false }]);
check(
  getVisualizationResourceSnapshot().residentBytes === baseline.residentBytes + 200,
  "old and replacement datasets remain accounted until worker acknowledgement",
);
owner.commit(2);
check(
  getVisualizationResourceSnapshot().residentBytes === baseline.residentBytes + 80,
  "worker acknowledgement releases the replaced lattice",
);

const abandoned = new ArrayBuffer(40);
owner.begin(3, [{ buffer: abandoned, sharedWithMain: false }]);
owner.discard(3);
check(
  getVisualizationResourceSnapshot().residentBytes === baseline.residentBytes + 80,
  "failed post discards only its pending dataset",
);

const shared = new SharedArrayBuffer(64);
const mainOwner = new VtkWorkerDatasetOwnership("main-buffer-test");
mainOwner.begin(1, [{ buffer: shared, sharedWithMain: true }]);
mainOwner.commit(1);
const workerOwner = new VtkWorkerDatasetOwnership("shared-worker-test");
workerOwner.begin(1, [{ buffer: shared, sharedWithMain: true }]);
workerOwner.commit(1);
check(
  getVisualizationResourceSnapshot().byStage["prepared-main"].residentBytes
    === baseline.byStage["prepared-main"].residentBytes + 64,
  "SAB-backed main and worker ownership counts one physical allocation",
);
mainOwner.reset();
check(
  getVisualizationResourceSnapshot().byStage["prepared-main"].residentBytes
    === baseline.byStage["prepared-main"].residentBytes + 64,
  "shared worker reference retains the SAB after main ownership releases",
);
workerOwner.reset();

owner.reset();
check(
  getVisualizationResourceSnapshot().residentBytes === baseline.residentBytes,
  "worker retirement releases current and pending dataset ownership",
);

const admitted = new VtkWorkerDatasetOwnership("preallocated-worker-test");
admitted.reserveCopy(1, 96);
check(
  getVisualizationResourceSnapshot().byStage["worker-resident"].reservedBytes
    === baseline.byStage["worker-resident"].reservedBytes + 96,
  "worker copy capacity is reserved before allocation",
);
const admittedBuffer = new ArrayBuffer(96);
admitted.materialize(1, [{ buffer: admittedBuffer, sharedWithMain: false }]);
check(
  getVisualizationResourceSnapshot().byStage["worker-resident"].reservedBytes
    === baseline.byStage["worker-resident"].reservedBytes
    && getVisualizationResourceSnapshot().byStage["worker-resident"].residentBytes
      === baseline.byStage["worker-resident"].residentBytes + 96,
  "materialization atomically converts reserved copy bytes into exact residency",
);
admitted.commit(1);
admitted.reset();
check(
  getVisualizationResourceSnapshot().residentBytes === baseline.residentBytes,
  "retiring a pre-admitted worker copy releases its exact allocation",
);

console.log(`\nVTK worker dataset ownership tests: ${passed} passed, ${failures.length} failed.`);
if (failures.length) {
  console.log("\nFailures:");
  for (const failure of failures) console.log(`  ${failure}`);
  process.exit(1);
}
