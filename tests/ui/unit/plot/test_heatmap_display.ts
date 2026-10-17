import assert from "node:assert/strict";
import type { Data } from "plotly.js";
import { smoothHeatmapDisplay } from "../../../../ui/src/lib/plot/heatmapDisplay.ts";

const z = [[0, 1, 2], [3, 4, 5]];
const trace = { type: "heatmap", x: [0, 2, 4], y: [1, 3], z, zsmooth: false, zmin: 0, zmax: 5 } as Data;
const source = [trace];
assert.equal(smoothHeatmapDisplay(source, false).data, source);
const smooth = smoothHeatmapDisplay(source, true);
assert.equal(smooth.smoothed, 1);
assert.equal((smooth.data[0] as typeof trace & { z: number[][] }).z, z);
assert.deepEqual(smooth.data[0], { ...trace, zsmooth: "fast", connectgaps: false });
assert.equal((trace as { zsmooth: boolean }).zsmooth, false);
for (const invalid of [null, NaN, Infinity]) {
  const gap = [{ ...trace, z: [[0, invalid, 2], [3, 4, 5]] } as Data];
  assert.equal(smoothHeatmapDisplay(gap, true).data, gap);
}
const irregular = [{ ...trace, x: [0, 2, 9] } as Data];
assert.equal(smoothHeatmapDisplay(irregular, true).data, irregular);
const line = [{ type: "scatter", x: [0, 1], y: [0, 1] } as Data];
assert.equal(smoothHeatmapDisplay(line, true).data, line);
assert.equal(smoothHeatmapDisplay(source, true, { xaxis: { type: "log" } }).smoothed, 0);
console.log("Optional heatmap display preserves source values, holes and irregular grids.");
