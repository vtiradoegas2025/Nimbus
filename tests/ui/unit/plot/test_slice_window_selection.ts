import assert from "node:assert/strict";
import { loadedSliceWindowIndex, sourceSliceControlsAvailable } from "../../../../ui/src/lib/plot/sliceWindowSelection.ts";
import type { NpyN } from "../../../../ui/src/lib/platform/npy.ts";

let passed = 0;
const check = (label: string, run: () => void) => { run(); passed++; console.log(`PASS ${label}`); };
const plane = (origin: number[]): NpyN => ({ shape: [4, 1, 3], data: new Float32Array(12),
  windowGeometry: { sourceShape: [4, 36, 3], origin, extent: [4, 1, 3], stride: [1, 1, 1],
    aggregation: "cell_mean", sampleLocation: "physical_volume_centroid", weighting: "uniform_cell_volume", sourceFormat: "npy", centroidsM: null } });
check("declared source control range stays available while its one-cell plane is loading", () => {
  assert.equal(sourceSliceControlsAvailable(3, 36, null), true);
  assert.equal(loadedSliceWindowIndex(plane([0, 0, 0]), 1, undefined, 2), 0);
});
check("a completed index2 window is labelled2 without treating its one-cell shape as the source extent", () => {
  assert.equal(loadedSliceWindowIndex(plane([0, 2, 0]), 1, undefined, 2), 2);
});
check("halo-relative selections preserve original window ownership", () => {
  assert.equal(loadedSliceWindowIndex(plane([0, 3, 0]), 1, [0, 1, 0], 2), 2);
});
check("same-source pending selection cannot relabel old values with the requested index", () => {
  assert.equal(loadedSliceWindowIndex(plane([0, 8, 0]), 1, undefined, 20), 8);
});
check("a complete raw volume can be sliced synchronously at the requested index", () => {
  assert.equal(loadedSliceWindowIndex({ shape: [4, 36, 3], data: new Float32Array(432) }, 1, undefined, 20), 20);
});
for (const [rank, count, error] of [[2, 36, null], [null, 36, null], [3, 0, null], [3, 36, "invalid grid"]] as const) {
  check("unavailable rank, extent, or source plan cannot enable a fabricated slice", () => {
    assert.equal(sourceSliceControlsAvailable(rank, count, error), false);
  });
}
console.log(`Slice window selection: ${passed} behavioral cases passed.`);
