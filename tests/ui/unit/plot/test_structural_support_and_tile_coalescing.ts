import assert from "node:assert/strict";
import { coalesceNativeFieldWindows, planBoundedStructuralSupportWindows, planNativeStructuralSupportWindows } from "../../../../ui/src/lib/plot/fieldLoadPlanning.ts";
import type { FieldLoadWindow, GridInfo } from "../../../../ui/src/lib/run/runOutputs.ts";

const grid: GridInfo = { nr: 512, nth: 512, nz: 160, drM: 100, dyM: 100, dzM: 100, scalarOriginM: [-25_550, -25_550, 50] };
const bounds = { lower: [-9500, -14500, 250] as [number, number, number], upper: [9500, 14500, 14750] as [number, number, number] };
assert.equal(planNativeStructuralSupportWindows(grid, bounds, 1_000_000).ok, false);
const complete = planBoundedStructuralSupportWindows(grid, bounds, 1_000_000);
assert.ok(complete.ok);
if (complete.ok) {
  assert.ok(complete.plan.sampledCells <= 1_000_000);
  assert.ok(complete.plan.stride.some(stride => stride > 1));
  const lower = complete.plan.windows[0].origin;
  const upper = lower.map((_, a) => Math.max(...complete.plan.windows.map(w => w.origin[a] + w.extent[a] - 1)));
  for (let a = 0; a < 3; a++) {
    assert.ok(grid.scalarOriginM![a] + lower[a] * 100 <= bounds.lower[a]);
    assert.ok(grid.scalarOriginM![a] + upper[a] * 100 >= bounds.upper[a]);
  }
}
const enormous: GridInfo = { nr: 10_000, nth: 10_000, nz: 1000, drM: 1, dyM: 1, dzM: 1, scalarOriginM: [0, 0, 0] };
assert.equal(planBoundedStructuralSupportWindows(enormous, { lower: [0, 0, 0], upper: [9999, 9999, 999] }, 1_000_000).ok, false);
assert.equal(planBoundedStructuralSupportWindows(grid, { ...bounds, lower: [NaN, 0, 0] }, 1_000_000).ok, false);
const tile = (origin: number[], extent: number[], stride = [2, 2, 2], aggregation: FieldLoadWindow["aggregation"] = "cell_mean"): FieldLoadWindow => ({
  origin, extent, stride, aggregation, maxOutputBytes: 64, maxSourceBytes: 256,
});
const native = { sourceKind: "native" as const };
const joined = [tile([0, 0, 0], [2, 4, 4]), tile([2, 0, 0], [2, 4, 4])];
const coalesced = coalesceNativeFieldWindows(joined, native);
assert.equal(coalesced.length, 1);
assert.deepEqual(coalesced[0].origin, [0, 0, 0]);
assert.deepEqual(coalesced[0].extent, [4, 4, 4]);
assert.equal(coalesced[0].aggregation, "cell_mean");
for (const unsafe of [
  [tile([0, 0, 0], [2, 4, 4], [2, 2, 2], "block_max"), tile([2, 0, 0], [2, 4, 4], [2, 2, 2], "block_min")],
  [tile([0, 0, 0], [2, 4, 4]), tile([4, 0, 0], [2, 4, 4])],
  [tile([0, 0, 0], [3, 4, 4]), tile([3, 0, 0], [1, 4, 4])],
  [tile([0, 0, 0], [2, 2, 1], [1, 1, 1]), tile([1, 0, 0], [1, 2, 1], [1, 1, 1]), tile([3, 0, 0], [1, 2, 1], [1, 1, 1])],
]) assert.deepEqual(coalesceNativeFieldWindows(unsafe, native), unsafe);
assert.deepEqual(coalesceNativeFieldWindows(joined, { ...native, terrainFollowing: true }), joined);
assert.deepEqual(coalesceNativeFieldWindows(joined, { sourceKind: "studio_derived" }), joined);
console.log("Structural support retains full bounded coverage; tile fusion preserves reductions, phase, gaps and overlap semantics");
