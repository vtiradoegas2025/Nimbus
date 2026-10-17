/* Regression tests for identity-preserving queue polling and progress events. */

import {
  applyRunProgressSnapshot,
  runSnapshotsEqual,
} from "../../../../ui/src/lib/core/hooks.ts";
import type { ProgressEvent, QueuedRun } from "../../../../ui/src/lib/core/types.ts";

let passed = 0;
let failed = 0;
const failures: string[] = [];
const check = (label: string, condition: boolean) => {
  if (condition) passed++;
  else {
    failed++;
    failures.push(label);
  }
};

const run: QueuedRun = {
  id: "run-1",
  caseName: "supercell",
  scheme: "klemp_wilhelmson_split_cartesian_cgrid",
  status: "running",
  startedAtMs: 1,
  endedAtMs: null,
  simTimeS: 10,
  step: 5,
  velMax: 20,
  durationS: 1200,
  outputDir: "/runs/run-1",
};
const runs = [run];

check("newly allocated identical poll snapshots compare equal", runSnapshotsEqual(runs, [{ ...run }]));
check("a changed field invalidates a poll snapshot", !runSnapshotsEqual(runs, [{ ...run, step: 6 }]));

const duplicate: ProgressEvent = {
  runId: run.id,
  progress: { simTimeS: 10, step: 5, velMax: 20 },
};
check(
  "duplicate progress preserves array identity",
  applyRunProgressSnapshot(runs, duplicate) === runs,
);

const partialDuplicate: ProgressEvent = {
  runId: run.id,
  progress: { simTimeS: 10, step: null, velMax: null },
};
check(
  "missing optional progress fields preserve their last values and identity",
  applyRunProgressSnapshot(runs, partialDuplicate) === runs,
);

const changed = applyRunProgressSnapshot(runs, {
  runId: run.id,
  progress: { simTimeS: 11, step: 6, velMax: 21 },
});
check("new progress returns a new array", changed !== runs);
check("new progress does not mutate the prior record", runs[0].step === 5);
check("new progress updates the selected record", changed[0].step === 6 && changed[0].simTimeS === 11);
check(
  "an event for an unknown run preserves array identity",
  applyRunProgressSnapshot(runs, { ...duplicate, runId: "missing" }) === runs,
);

console.log(`\nrun-state tests: ${passed} passed, ${failed} failed.`);
if (failures.length > 0) {
  console.log("\nFailures:");
  for (const failure of failures) console.log(`  ${failure}`);
  process.exit(1);
}

