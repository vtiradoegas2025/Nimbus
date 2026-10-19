import {
  VISUALIZATION_RESOURCE_SCHEMA,
  VisualizationResourceGovernor,
  visualizationBufferClaim,
  type VisualizationResourcePlan,
} from "../../../../ui/src/lib/core/visualizationResourceGovernor.ts";

let passed = 0;
const failures: string[] = [];
function check(condition: boolean, label: string): void {
  if (condition) passed += 1;
  else failures.push(`FAIL ${label}`);
}

const governor = new VisualizationResourceGovernor({
  totalBytes: 400,
  stageBytes: {
    "prepared-main": 220,
    "retained-geometry": 160,
  },
});

const initial = governor.stats();
check(
  initial.schema === VISUALIZATION_RESOURCE_SCHEMA
    && initial.version === 3
    && initial.residentBytes === 0,
  "resource ledger starts with a versioned empty snapshot",
);
check(
  Object.isFrozen(initial) && Object.isFrozen(initial.byStage)
    && Object.isFrozen(initial.byStage["prepared-main"]),
  "resource snapshots and stage records are immutable",
);
check(governor.stats() === initial, "snapshot identity is stable between transitions");
governor.setHardwareFacts({ source: "tauri", systemRamBytes: 64 * 1024, logicalCores: 12 });
check(
  governor.stats().hardware.source === "tauri"
    && governor.stats().hardware.systemRamBytes === 64 * 1024
    && governor.stats().hardware.logicalCores === 12,
  "hardware facts are explicit inputs rather than inferred capacity",
);

const reservedGovernor = new VisualizationResourceGovernor({ totalBytes: 100 });
const reserved = reservedGovernor.reserve({
  owner: "scratch-plan",
  generation: "frame-1",
  priority: "interactive",
  claims: [{
    allocationId: "scratch-a",
    stage: "worker-scratch",
    bytes: 60,
    accounting: "reserved",
  }],
});
check(reserved.ok, "planned working capacity can be reserved without claiming residency");
if (!reserved.ok) throw new Error(reserved.reason);
check(
  reservedGovernor.stats().residentBytes === 0
    && reservedGovernor.stats().reservedBytes === 60
    && reservedGovernor.stats().committedBytes === 60
    && reservedGovernor.stats().byStage["worker-scratch"].reservedBytes === 60,
  "resident, reserved, and committed bytes remain distinct",
);
const reservedOverflow = reservedGovernor.reserve({
  owner: "scratch-plan-2",
  generation: "frame-1",
  priority: "interactive",
  claims: [{
    allocationId: "scratch-b",
    stage: "worker-scratch",
    bytes: 50,
    accounting: "reserved",
  }],
});
check(
  !reservedOverflow.ok && reservedOverflow.requiredBytes === 110,
  "aggregate admission includes resident and reserved capacity",
);
reservedGovernor.release(reserved.reservation.id);

let notifications = 0;
const unsubscribe = governor.subscribe(() => { notifications += 1; });
const first = governor.reserve({
  owner: "iso-pane",
  generation: "frame-1",
  priority: "interactive",
  claims: [
    { allocationId: "prepared-a", stage: "prepared-main", bytes: 120 },
    { allocationId: "geometry-a", stage: "retained-geometry", bytes: 80 },
  ],
});
check(first.ok, "an admitted visualization reservation succeeds");
if (!first.ok) throw new Error(first.reason);
check(
  governor.stats().residentBytes === 200
    && governor.stats().byStage["prepared-main"].residentBytes === 120
    && governor.stats().byStage["retained-geometry"].residentBytes === 80,
  "reservation residency is exact by stage and aggregate",
);

const shared = governor.reserve({
  owner: "comparison-pane",
  generation: "frame-1",
  priority: "visible",
  claims: [{ allocationId: "prepared-a", stage: "prepared-main", bytes: 120 }],
});
check(shared.ok, "a second consumer can share an identical allocation");
if (!shared.ok) throw new Error(shared.reason);
check(
  governor.stats().reservations === 2
    && governor.stats().allocations === 2
    && governor.stats().residentBytes === 200,
  "shared physical allocations are counted once across reservations",
);

const rejected = governor.reserve({
  owner: "streamline-pane",
  generation: "frame-1",
  priority: "interactive",
  claims: [{ allocationId: "vector-a", stage: "prepared-main", bytes: 120 }],
});
check(
  !rejected.ok && rejected.stage === "prepared-main"
    && rejected.requiredBytes === 240 && rejected.availableBytes === 220,
  "stage admission reports the exact required and available bytes",
);
check(
  governor.stats().residentBytes === 200 && governor.stats().rejected === 1,
  "rejected admission is failure-atomic",
);

const transitionRejected = governor.transition(first.reservation.id, [
  { allocationId: "prepared-a", stage: "prepared-main", bytes: 120 },
  { allocationId: "geometry-b", stage: "retained-geometry", bytes: 170 },
]);
check(
  !transitionRejected.ok && transitionRejected.stage === "retained-geometry",
  "transition admission enforces the destination stage before mutation",
);
check(
  governor.stats().residentBytes === 200
    && governor.stats().byStage["retained-geometry"].residentBytes === 80,
  "rejected transition preserves the prior reservation",
);

const transitioned = governor.transition(first.reservation.id, [
  { allocationId: "prepared-a", stage: "prepared-main", bytes: 120 },
  { allocationId: "geometry-b", stage: "retained-geometry", bytes: 60 },
]);
check(transitioned.ok, "an admitted ownership transition succeeds");
check(
  governor.stats().residentBytes === 180
    && governor.stats().byStage["retained-geometry"].residentBytes === 60
    && governor.stats().transitions === 1,
  "ownership transition atomically replaces unshared allocations",
);

check(governor.release(first.reservation.id), "first reservation releases");
check(
  governor.stats().residentBytes === 120
    && governor.stats().byStage["prepared-main"].residentBytes === 120,
  "shared allocation remains resident for its remaining consumer",
);
check(!governor.release(first.reservation.id), "reservation release is idempotent");
check(governor.release(shared.reservation.id), "last shared consumer releases");
check(
  governor.stats().residentBytes === 0
    && governor.stats().allocations === 0
    && governor.stats().highWaterBytes === 200,
  "last release clears residency while preserving high water",
);

const backing = new ArrayBuffer(64);
const claimA = visualizationBufferClaim(backing, "prepared-main");
const claimB = visualizationBufferClaim(backing, "prepared-main");
check(
  claimA.allocationId === claimB.allocationId && claimA.bytes === 64,
  "buffer claims preserve physical backing-store identity",
);

let conflictCaught = false;
try {
  governor.reserve({
    owner: "invalid",
    generation: "frame-2",
    priority: "background",
    claims: [
      { allocationId: "conflict", stage: "prepared-main", bytes: 1 },
      { allocationId: "conflict", stage: "gpu", bytes: 1 },
    ],
  });
} catch {
  conflictCaught = true;
}
check(conflictCaught, "conflicting aliases fail before ledger mutation");

const plan: VisualizationResourcePlan = {
  identity: "run:frame:field:roi",
  execution: "worker-cpu",
  representation: "native-roi",
  sampledShape: [140, 140, 100],
  sampledCellUpperBound: 1_960_000,
  expectedPeakBytes: {
    nativeRead: 7_840_000,
    decoded: 7_840_000,
    preparedMain: 7_840_000,
    workerResident: 7_840_000,
    workerScratch: 0,
    handoff: 7_840_000,
    retainedGeometry: 0,
    gpu: 0,
  },
  outputBudget: { bytes: 128 * 1024 * 1024, primitives: 300_000 },
  provenance: "native 14x14x10 km diagnostic ROI",
};
check(
  plan.sampledShape?.reduce((product, value) => product * value, 1) === 1_960_000
    && plan.sampledCellUpperBound === 1_960_000,
  "resource-plan contract represents the production supercell diagnostic lattice",
);

unsubscribe();
const notificationsBefore = notifications;
governor.setCapacity({ totalBytes: 500 });
check(notifications === notificationsBefore, "unsubscribe stops resource-ledger notifications");

console.log(`\nvisualization resource governor tests: ${passed} passed, ${failures.length} failed.`);
if (failures.length) {
  console.log("\nFailures:");
  for (const failure of failures) console.log(`  ${failure}`);
  process.exit(1);
}
