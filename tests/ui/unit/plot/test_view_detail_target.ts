import assert from "node:assert/strict";
import { clampGridHorizontalPointM, nativeDetailCoversDomain, planNativeStormDetailWindows, nativeDetailRecenteringThresholdM } from "../../../../ui/src/lib/plot/fieldLoadPlanning.ts";
const grid = { nr: 5, nth: 5, nz: 3, drM: 1000, dyM: 1000, dzM: 100, scalarOriginM: [0, 0, 0] as [number, number, number] };
assert.deepEqual(clampGridHorizontalPointM(grid, [123, 456]), [123, 456], "camera focus must not snap to the coarse parent lattice");
assert.deepEqual(clampGridHorizontalPointM(grid, [-10000, 10000]), [-500, 4500]);
assert.equal(clampGridHorizontalPointM(grid, [NaN, 0]), null);
assert.equal(clampGridHorizontalPointM(grid, [0]), null);
const valid = { ...grid, validIndexRange: { lower: [1, 1, 0] as [number, number, number], upper: [3, 3, 2] as [number, number, number] } };
assert.deepEqual(clampGridHorizontalPointM(valid, [-10000, 10000]), [500, 3500]);
assert.deepEqual(clampGridHorizontalPointM({ ...grid, physicalBoundsM: { lower: [0, 0, 0], upper: [4000, 4000, 300] } }, [-10000, 10000]), [0, 4000]);
const stretched = { ...grid, scalarCoordinatesM: [[0, 500, 1500, 3000, 5000], [0, 1000, 2000, 3000, 4000], [0, 100, 200]] as [number[], number[], number[]] };
assert.deepEqual(clampGridHorizontalPointM(stretched, [-10000, 10000]), [-250, 4500]);
for (const cap of [16_000_000, 4_000_000, 250_000]) {
  const plan = planNativeStormDetailWindows({ nr: 512, nth: 512, nz: 160, drM: 100, dyM: 100, dzM: 100 }, [-20000, 18000], cap * 0.75, [28000, 28000, 10000], undefined, true);
  assert.ok(plan.ok);
  if (plan.ok) {
    assert.ok(plan.plan.sampledCells <= cap * 0.75);
    assert.deepEqual(plan.plan.stride, [1, 1, 1]);
    assert.equal(plan.plan.sampledShape[2], 160);
    assert.ok(nativeDetailRecenteringThresholdM(plan.plan) > 0);
  }
}
assert.equal(nativeDetailCoversDomain(grid, 1000, [28000, 28000, 10000], true), true);
assert.equal(nativeDetailCoversDomain({ ...grid, scalarCoordinatesM: [[0, 10000, 20000, 30000, 40000], [0, 1000, 2000, 3000, 4000], [0, 100, 200]] }, 1000, [28000, 28000, 10000], true), false,
  "context decisions use the actual coordinate-aware native plan, not nominal spacing");
console.log("View detail targeting: continuous focus, physical/valid bounds, and native full-column budgets passed.");
