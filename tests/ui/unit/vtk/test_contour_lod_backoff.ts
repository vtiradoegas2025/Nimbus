/*
 * Contour LOD backoff for `src/lib/vtk/contourLodBackoff.ts`.
 *
 * The rule this encodes: a triangle ceiling is the renderer's constraint, not
 * the user's. An over-large surface coarsens its own lattice and discloses it,
 * and only becomes an error once coarsening has genuinely run out.
 *
 * Run via:
 *   cd ui && npm run test:contour-lod-backoff
 */

import {
  contourBackoff,
  contourVoxelBudget,
  MAX_CONTOUR_LOD_DIVISOR,
  MIN_CONTOUR_VOXEL_BUDGET,
  nextContourLodDivisor,
} from "../../../../ui/src/lib/vtk/contourLodBackoff.ts";

let failed = 0;
let passed = 0;
const failures: string[] = [];

function assertTrue(cond: boolean, label: string): void {
  if (cond) {
    passed++;
  } else {
    failed++;
    failures.push(`FAIL ${label}`);
  }
}

function check(label: string, fn: () => void): void {
  try {
    fn();
  } catch (e) {
    failed++;
    failures.push(`THREW ${label}: ${(e as Error).message}`);
  }
}

// ---------- Budget ----------

check("an unbacked-off layer keeps its full budget", () => {
  assertTrue(contourVoxelBudget(16_000_000, 1) === 16_000_000, "divisor 1 is the base budget");
});

check("each step halves the lattice budget", () => {
  assertTrue(contourVoxelBudget(16_000_000, 2) === 8_000_000, "one step");
  assertTrue(contourVoxelBudget(16_000_000, 4) === 4_000_000, "two steps");
  assertTrue(contourVoxelBudget(16_000_000, 8) === 2_000_000, "three steps");
});

check("the lattice never falls below a storm-resolving floor", () => {
  // Backing off forever would eventually render a smooth blob and call it data.
  assertTrue(
    contourVoxelBudget(16_000_000, MAX_CONTOUR_LOD_DIVISOR) >= MIN_CONTOUR_VOXEL_BUDGET,
    "coarsest step still resolves",
  );
  assertTrue(
    contourVoxelBudget(1000, 64) === MIN_CONTOUR_VOXEL_BUDGET,
    "a tiny base budget clamps to the floor rather than going to zero",
  );
});

check("a nonsense budget or divisor cannot produce a nonsense lattice", () => {
  for (const bad of [0, -1, Number.NaN, Number.POSITIVE_INFINITY]) {
    assertTrue(contourVoxelBudget(bad, 1) >= MIN_CONTOUR_VOXEL_BUDGET, `budget ${String(bad)}`);
    assertTrue(contourVoxelBudget(16_000_000, bad) > 0, `divisor ${String(bad)}`);
  }
  assertTrue(contourVoxelBudget(16_000_000, 0.5) === 16_000_000, "a divisor below 1 cannot refine");
});

// ---------- Steps ----------

check("the ladder terminates at the coarsest allowed step", () => {
  let divisor = 1;
  let steps = 0;
  for (;;) {
    const next = nextContourLodDivisor(divisor);
    if (next == null) break;
    divisor = next;
    steps++;
    assertTrue(steps < 100, "the ladder cannot loop forever");
    if (steps >= 100) break;
  }
  assertTrue(divisor === MAX_CONTOUR_LOD_DIVISOR, `ends at ${MAX_CONTOUR_LOD_DIVISOR}`);
  assertTrue(nextContourLodDivisor(MAX_CONTOUR_LOD_DIVISOR) == null, "no step past the end");
});

// ---------- Policy ----------

check("a surface within budget is drawn and says nothing", () => {
  const outcome = contourBackoff(null, 1, "|w| threshold");
  assertTrue(outcome.nextDivisor == null, "no backoff");
  assertTrue(outcome.error == null, "no error");
});

check("an over-large surface coarsens instead of refusing", () => {
  // The reported defect: "Combined contour output has 4988736 triangles; limit
  // is 4000000" hid the layer and made the ceiling the user's problem.
  const outcome = contourBackoff("Combined contour output has 4988736 triangles", 1, "|w| threshold");
  assertTrue(outcome.nextDivisor === 2, "backs off one step");
  assertTrue(outcome.error == null, "the user is not asked to do anything yet");
});

check("backing off continues while there is room", () => {
  let divisor = 1;
  for (let step = 0; step < 6; step++) {
    const outcome = contourBackoff("too large", divisor, "|w| threshold");
    assertTrue(outcome.error == null, `step ${step} still has room`);
    assertTrue(outcome.nextDivisor != null, `step ${step} coarsens`);
    divisor = outcome.nextDivisor!;
  }
  assertTrue(divisor === MAX_CONTOUR_LOD_DIVISOR, "six halvings reach the floor");
});

check("only a genuinely exhausted ladder becomes an error", () => {
  const outcome = contourBackoff("too large", MAX_CONTOUR_LOD_DIVISOR, "|w| threshold");
  assertTrue(outcome.nextDivisor == null, "nowhere left to go");
  assertTrue(outcome.error != null, "now it is reported");
});

check("the error names a control that changes the outcome", () => {
  // The prior message quoted a triangle count and a limit, neither of which the
  // user can act on.
  const message = contourBackoff("too large", MAX_CONTOUR_LOD_DIVISOR, "|w| threshold").error!;
  assertTrue(message.includes("|w| threshold"), "names the control");
  assertTrue(!/\d{4,}/.test(message), "quotes no raw internal counts");
  assertTrue(!message.toLowerCase().includes("limit is"), "does not state an internal ceiling");
});

check("each layer names its own control", () => {
  const vortex = contourBackoff("too large", MAX_CONTOUR_LOD_DIVISOR, "core threshold").error!;
  assertTrue(vortex.includes("core threshold"), "vortex layer");
  const cloud = contourBackoff("too large", MAX_CONTOUR_LOD_DIVISOR, "cloud floor").error!;
  assertTrue(cloud.includes("cloud floor"), "cloud layer");
});

console.log(`\ncontour LOD backoff tests: ${passed} passed, ${failed} failed.`);
if (failures.length > 0) {
  console.log("\nFailures:");
  for (const f of failures) console.log("  " + f);
  process.exit(1);
}
process.exit(0);
