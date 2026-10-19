/* Deterministic incremental live-run discovery tests. */

import { reconcileLiveSteps } from "../../../../ui/src/lib/dataset/liveStepDiscovery.ts";

let passed = 0;
let failed = 0;
const failures: string[] = [];

function equal<T>(actual: T, expected: T, label: string): void {
  if (actual === expected) passed++;
  else {
    failed++;
    failures.push(`FAIL ${label}: expected ${String(expected)}, got ${String(actual)}`);
  }
}

type State = "complete" | "incomplete" | "invalid_manifest" | "legacy";
interface Step {
  index: number;
  publicationState: State;
  publicationGeneration: string | null;
}
const step = (index: number, state: State = "complete", generation = "run-a"): Step => ({
  index,
  publicationState: state,
  publicationGeneration: state === "legacy" ? null : generation,
});

// Unchanged polls preserve the exact array/object working set.
{
  const current = [step(0), step(10)];
  const result = reconcileLiveSteps(current, [step(0), step(10)], 0, "run-a", true);
  equal(result.steps, current, "unchanged scan preserves array identity");
  equal(result.selectedIndex, 0, "non-tail selection remains pinned");
}

// A user at the tail follows newly completed frames; incomplete output stays hidden.
{
  const current = [step(0), step(10)];
  const result = reconcileLiveSteps(
    current,
    [step(20), step(10), step(30, "incomplete"), step(0)],
    1,
    "run-a",
    true,
  );
  equal(result.steps.length, 3, "one completed frame appended");
  equal(result.steps[0], current[0], "existing frame object retained");
  equal(result.steps[2].index, 20, "appended inventory sorted by absolute step");
  equal(result.selectedIndex, 2, "tail selection follows newest complete frame");
  equal(result.incompleteCount, 1, "in-progress frame counted but hidden");
}

// Playback or a user away from the tail preserves the absolute selected step.
{
  const current = [step(0), step(10), step(20)];
  const result = reconcileLiveSteps(
    current,
    [...current, step(30)],
    1,
    "run-a",
    false,
  );
  equal(result.steps[result.selectedIndex].index, 10, "absolute selection preserved");
}

// A publication replacement cannot retain old frames or cache identity.
{
  const result = reconcileLiveSteps(
    [step(0), step(10)],
    [step(0, "complete", "run-b"), step(10, "incomplete", "run-b")],
    1,
    "run-a",
    true,
  );
  equal(result.generationChanged, true, "generation replacement detected");
  equal(result.publicationGeneration, "run-b", "replacement generation selected");
  equal(result.steps.length, 1, "old generation inventory discarded");
  equal(result.steps[0].publicationGeneration, "run-b", "only replacement frame remains");
  equal(result.selectedIndex, 0, "replacement selects its available tail");
}

// An empty directory scan has no marker identity and must not erase known data.
{
  const current = [step(0)];
  const result = reconcileLiveSteps(current, [], 0, "run-a", true);
  equal(result.publicationGeneration, "run-a", "empty scan retains known generation");
  equal(result.steps, current, "empty scan retains completed inventory");
}

// A run written without a publication marker is selectable like any other; Studio
// no longer singles it out, so the only thing to assert is that it is not dropped.
{
  const result = reconcileLiveSteps([], [step(0, "legacy")], 0, null, true);
  equal(result.steps.length, 1, "unmarked frame remains selectable");
  equal(result.incompleteCount, 0, "unmarked frame is not counted as incomplete");
}

console.log(`\nliveStepDiscovery tests: ${passed} passed, ${failed} failed.`);
if (failures.length > 0) {
  console.log("\nFailures:");
  for (const failure of failures) console.log(`  ${failure}`);
  process.exit(1);
}
