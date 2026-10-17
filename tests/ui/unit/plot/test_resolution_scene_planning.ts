import assert from "node:assert/strict";
import { planNativeStormDetailWindows, planInteractiveVolumeWindows, MAX_INTERACTIVE_FULL_DOMAIN_SOURCE_CELLS } from "../../../../ui/src/lib/plot/fieldLoadPlanning.ts";
import { planSupercellSceneBudget } from "../../../../ui/src/lib/plot/supercellSceneBudget.ts";
import type { GridInfo } from "../../../../ui/src/lib/run/runOutputs.ts";

let cases = 0;
for (const spacing of [100, 50, 25, 10, 0.5]) {
  for (const anisotropic of [false, true]) {
    const dx = spacing, dy = spacing * (anisotropic ? 1.5 : 1), dz = spacing * (anisotropic ? 2 : 1);
    const grid: GridInfo = {
      coordinateSystem: "cartesian", nr: Math.ceil(51_200 / dx), nth: Math.ceil(51_200 / dy), nz: Math.ceil(16_000 / dz),
      drM: dx, dyM: dy, dzM: dz, scalarOriginM: [dx / 2, dy / 2, dz / 2],
    };
    grid.validIndexRange = { lower: [1, 1, 0], upper: [grid.nr! - 2, grid.nth! - 2, grid.nz! - 1] };
    for (const ram of [null, 8, 48].map(v => v == null ? null : v * 1024 ** 3)) {
      for (const paneScale of [1, 0.25]) {
        for (const fieldCount of [1, 4, 8]) {
          const shared = planSupercellSceneBudget({
            baseCells: Math.floor(16_000_000 * paneScale), paneScale, systemRamBytes: ram,
            fields: Array.from({ length: fieldCount }, (_, i) => ({ identity: `field-${i}`, role: "detail", bytesPerCell: 4 })),
            preparedChannels: { detail: fieldCount === 1 ? 1 : 3 },
            scratchChannels: { detail: fieldCount === 1 ? 0 : 2 },
          });
          const outcome = planNativeStormDetailWindows(grid, [25_600, 25_600], shared.detailCells, [28_000, 28_000, 10_000], undefined, true);
          assert.ok(outcome.ok, `Native planning failed at ${spacing} m, ${fieldCount} fields`);
          if (!outcome.ok) continue;
          assert.ok(outcome.plan.sampledCells <= shared.detailCells);
          assert.deepEqual(outcome.plan.stride, [1, 1, 1]);
          assert.equal(outcome.plan.sampledShape[2], grid.nz);
          const range = outcome.plan.detailIndexRange;
          for (let axis = 0; axis < 3; axis++) {
            assert.ok(range.lower[axis] >= grid.validIndexRange!.lower[axis]);
            assert.ok(range.upper[axis] <= grid.validIndexRange!.upper[axis]);
            assert.ok(outcome.plan.detailSpanM[axis] > 0);
          }
          const interactive = planInteractiveVolumeWindows(grid, shared.detailCells, [25_600, 25_600]);
          assert.ok(interactive.ok);
          if ((grid.nr! - 2) * (grid.nth! - 2) * grid.nz! > MAX_INTERACTIVE_FULL_DOMAIN_SOURCE_CELLS) {
            assert.ok(interactive.ok && interactive.mode === "native-detail");
            assert.ok(interactive.ok && interactive.plan.sampledCells <= shared.detailCells);
          }
          cases++;
        }
      }
    }
  }
}
const fine: GridInfo = { nr: 128, nth: 128, nz: 64, drM: 0.5, dyM: 0.5, dzM: 0.5, scalarOriginM: [0, 0, 0] };
const precise = planNativeStormDetailWindows(fine, [20, 20], 100_000, [10, 10, 10]);
assert.ok(precise.ok);
if (precise.ok) {
  assert.deepEqual(precise.plan.sampledShape, [20, 20, 20]);
  assert.deepEqual(precise.plan.detailSpanM, [10, 10, 10]);
}
for (const drM of [0, -1, NaN, Infinity]) {
  assert.equal(planNativeStormDetailWindows({ ...fine, drM }, [20, 20], 100_000).ok, false);
}
const constrained = planSupercellSceneBudget({ baseCells: 4_000_000, paneScale: 0.25,
  systemRamBytes: 4 * 1024 ** 3,
  fields: (["detail", "context", "auxiliary"] as const).flatMap(role => Array.from({ length: 8 }, (_, index) => ({
    identity: `${index}`, role, bytesPerCell: 36,
  }))), preparedChannels: { detail: 3, context: 3, auxiliary: 3 }, scratchChannels: { detail: 2, context: 2, auxiliary: 2 } });
const longColumn: GridInfo = { nr: 102400, nth: 102400, nz: 32000, drM: 0.5, dyM: 0.5, dzM: 0.5,
  scalarOriginM: [0.25, 0.25, 0.25], validIndexRange: { lower: [1, 1, 0], upper: [102398, 102398, 31999] } };
const impossibleVolume = planNativeStormDetailWindows(longColumn, [25600, 25600], constrained.detailCells,
  [28000, 28000, 10000], undefined, true);
assert.equal(impossibleVolume.ok, false, "a scene budget cannot produce a zero-width 1×1×32000 native image");
const minimumVolume = planNativeStormDetailWindows(longColumn, [25600, 25600], 4 * 32000,
  [28000, 28000, 10000], undefined, true);
assert.ok(minimumVolume.ok);
if (minimumVolume.ok) assert.deepEqual(minimumVolume.plan.sampledShape, [2, 2, 32000]);
console.log(`Resolution/anisotropy/field-count/viewer/pane matrix: ${cases} metadata-only native plans passed`);
