/*
 * Tests for src/lib/stepCompleteness.ts.
 *
 * Run via:
 *   cd ui && node --experimental-strip-types scripts/test_step_completeness.ts
 *
 * Guards the data-integrity contract: current partial/corrupt frames are hidden,
 * markerless runs remain available only through the bridge's explicit legacy
 * classification, and validated current frames are selectable.
 */

import { stepIsComplete, filterCompleteSteps } from "../../../../ui/src/lib/run/stepCompleteness.ts";

let passed = 0;
let failed = 0;
const failures: string[] = [];

function assertTrue(cond: boolean, label: string): void {
  if (cond) passed++;
  else {
    failed++;
    failures.push(`FAIL ${label}`);
  }
}

type PublicationState = "complete" | "incomplete" | "invalid_manifest" | "legacy";
type Step = { index: number; publicationState: PublicationState };
const step = (i: number, publicationState: PublicationState): Step => ({
  index: i,
  publicationState,
});
const complete = (i: number): Step => step(i, "complete");
const torn = (i: number): Step => step(i, "incomplete");

// ---------- stepIsComplete ----------
assertTrue(stepIsComplete(complete(0)) === true, "validated current -> complete");
assertTrue(stepIsComplete(torn(0)) === false, "first-frame partial -> incomplete");
assertTrue(
  stepIsComplete(step(0, "invalid_manifest")) === false,
  "corrupt current manifest -> invalid",
);
assertTrue(stepIsComplete(step(0, "legacy")) === true, "explicit legacy -> visible");

// ---------- trailing torn step (the crashed-run case) is dropped ----------
{
  const out = filterCompleteSteps([complete(0), complete(1), torn(2)]);
  assertTrue(out.length === 2, "trailing torn step dropped");
  assertTrue(out.every((s) => s.publicationState === "complete"), "only complete steps remain");
  assertTrue(out[0].index === 0 && out[1].index === 1, "order + identity preserved");
}

// ---------- any torn steps (mid + trailing) are dropped ----------
{
  const out = filterCompleteSteps([complete(0), torn(1), complete(2), torn(3)]);
  assertTrue(out.length === 2 && out[0].index === 0 && out[1].index === 2, "all torn steps dropped");
}

// ---------- all complete -> unchanged ----------
{
  const out = filterCompleteSteps([complete(0), complete(1)]);
  assertTrue(out.length === 2, "all-complete run kept intact");
}

// ---------- zero complete current frames -> honest waiting state ----------
{
  const out = filterCompleteSteps([torn(0), torn(1), torn(2)]);
  assertTrue(out.length === 0, "partial first current frames are all hidden");
}

// ---------- explicit legacy path remains available ----------
{
  const out = filterCompleteSteps([step(0, "legacy"), step(1, "legacy")]);
  assertTrue(out.length === 2, "explicit legacy frames remain visible");
}

// ---------- corrupt current manifest is hidden alongside partial frames ----------
{
  const out = filterCompleteSteps([
    step(0, "invalid_manifest"),
    complete(1),
  ]);
  assertTrue(out.length === 1 && out[0].index === 1, "corrupt current frame hidden");
}

// ---------- empty input ----------
assertTrue(filterCompleteSteps([]).length === 0, "empty -> empty");

console.log(`\nstepCompleteness tests: ${passed} passed, ${failed} failed.`);
if (failures.length > 0) {
  console.log("\nFailures:");
  for (const f of failures) console.log("  " + f);
  process.exit(1);
}
process.exit(0);
