/* Renderer LOD must bound Plotly work without changing the Python result/export. */

import { preparePythonArrayDisplay, pythonArrayViewportFromRelayout } from "../../../../ui/src/lib/plot/pythonArrayDisplay.ts";
import assert from "node:assert/strict";

let passed = 0;
const failures: string[] = [];
function check(condition: boolean, label: string): void {
  if (condition) passed += 1;
  else failures.push(`FAIL ${label}`);
}

const nativeLine = preparePythonArrayDisplay(Float32Array.from([1, 2, 3]), [3]);
check(nativeLine?.kind === "line" && !nativeLine.reduced, "small lines remain native");
check(
  nativeLine?.kind === "line" && nativeLine.x.join() === "0,1,2" && nativeLine.y.join() === "1,2,3",
  "native line coordinates and values are exact",
);

const signal = new Float32Array(1_002);
signal.fill(5);
signal[123] = -80;
signal[456] = 120;
signal[700] = Number.NaN;
const reducedLine = preparePythonArrayDisplay(signal, [signal.length], { maxLinePoints: 32 });
check(reducedLine?.kind === "line" && reducedLine.reduced, "large lines use display LOD");
check(
  reducedLine?.kind === "line" && reducedLine.displayPoints <= 32,
  "line display stays inside its renderer budget",
);
check(
  reducedLine?.kind === "line" && reducedLine.x[0] === 0 && reducedLine.x.at(-1) === 1_001,
  "line display retains both source endpoints",
);
check(
  reducedLine?.kind === "line" && reducedLine.y.includes(-80) && reducedLine.y.includes(120),
  "ordered buckets retain finite minima and maxima",
);
check(
  reducedLine?.kind === "line" && reducedLine.y.some(Number.isNaN),
  "ordered buckets retain a missing-data break",
);
check(
  reducedLine?.kind === "line" && reducedLine.x.every((value, index, values) => index === 0 || value > values[index - 1]),
  "line display remains in source order",
);

const rows = 23;
const columns = 17;
const field = Float32Array.from({ length: rows * columns }, (_, index) => index);
const heatmap = preparePythonArrayDisplay(field, [rows, columns], { maxHeatmapCells: 80 });
check(heatmap?.kind === "heatmap" && heatmap.reduced, "large heatmaps use display LOD");
check(
  heatmap?.kind === "heatmap" && heatmap.displayShape[0] * heatmap.displayShape[1] <= 80,
  "heatmap display stays inside its renderer budget",
);
check(
  heatmap?.kind === "heatmap"
    && heatmap.x[0] === 0 && heatmap.x.at(-1) === columns - 1
    && heatmap.y[0] === 0 && heatmap.y.at(-1) === rows - 1,
  "heatmap point sampling includes all four boundaries",
);
check(
  heatmap?.kind === "heatmap" && heatmap.z.every((row, displayRow) =>
    row.every((value, displayColumn) =>
      value === heatmap.y[displayRow] * columns + heatmap.x[displayColumn])),
  "heatmap values map to the disclosed source row and column indices",
);

const nativeHeatmap = preparePythonArrayDisplay(field, [rows, columns], { maxHeatmapCells: 400 });
check(
  nativeHeatmap?.kind === "heatmap" && !nativeHeatmap.reduced
    && nativeHeatmap.displayShape.join() === `${rows},${columns}`,
  "heatmaps that fit are not reduced",
);
check(preparePythonArrayDisplay(new Float32Array(8), [2, 2, 2]) === null, "rank-3 output is not mis-plotted");

let shapeRejected = false;
try {
  preparePythonArrayDisplay(new Float32Array(3), [2, 2]);
} catch {
  shapeRejected = true;
}
check(shapeRejected, "shape/buffer disagreement fails closed");

const zoomed = preparePythonArrayDisplay(field, [rows, columns], {
  maxHeatmapCells: 80, xRange: [5.2, 8.8], yRange: [10.2, 13.8],
});
check(zoomed?.kind === "heatmap" && !zoomed.reduced, "zoom recovers native samples inside the visible region");
check(zoomed?.kind === "heatmap" && zoomed.x.join() === "4,5,6,7,8,9,10"
  && zoomed.y.join() === "9,10,11,12,13,14,15", "zoom retains global indices and bracketing cells");
check(zoomed?.kind === "heatmap" && zoomed.z.every((row, y) => row.every((value, x) =>
  value === zoomed.y[y] * columns + zoomed.x[x])), "zoom values use source row stride");
const reversed = preparePythonArrayDisplay(field, [rows, columns], {
  maxHeatmapCells: 80, xRange: [8.8, 5.2], yRange: [13.8, 10.2],
});
check(JSON.stringify(reversed) === JSON.stringify(zoomed), "reversed axes select the same native cells");
const outside = preparePythonArrayDisplay(field, [rows, columns], { xRange: [-20, -10], yRange: [40, 50] });
check(outside?.kind === "heatmap" && outside.x.length === 0 && outside.y.length === 0,
  "panning outside the data does not invent edge values");
const invalidRange = preparePythonArrayDisplay(field, [rows, columns], { maxHeatmapCells: 80, xRange: [NaN, 2] });
check(JSON.stringify(invalidRange) === JSON.stringify(heatmap), "nonfinite viewport falls back to full extent");
const zoomLine = preparePythonArrayDisplay(signal, [signal.length], { maxLinePoints: 32, xRange: [695, 705] });
check(zoomLine?.kind === "line" && !zoomLine.reduced && zoomLine.x[0] === 694
  && Number.isNaN(zoomLine.y[700 - 694]), "line zoom preserves source indices and missing-data breaks");
check(field[0] === 0 && field.at(-1) === rows * columns - 1 && signal[456] === 120,
  "viewport reduction leaves full-resolution export buffers intact");

const pendingX = pythonArrayViewportFromRelayout({ "xaxis.range": [510, 520] }, null, [1200, 1200], [0, 1]);
assert.deepEqual(pendingX, { xMin: 510, xMax: 520, yMin: 0, yMax: 1199 });
assert.equal(pythonArrayViewportFromRelayout({ autosize: true }, pendingX!, [1200, 1200], [0, 1]), undefined,
  "a resize cannot replace a pending zoom during debounce");
const pendingBoth = pythonArrayViewportFromRelayout({ "yaxis.range": [610, 620] }, pendingX!, [1200, 1200], [0, 1]);
assert.deepEqual(pendingBoth, { xMin: 510, xMax: 520, yMin: 610, yMax: 620 }, "split-axis changes retain the pending other axis");
assert.equal(pythonArrayViewportFromRelayout({ "xaxis.autorange": true }, pendingBoth!, [1200, 1200], [0, 1]), null);

console.log(`\npython array display tests: ${passed} passed, ${failures.length} failed.`);
if (failures.length) {
  console.log("\nFailures:");
  for (const failure of failures) console.log(`  ${failure}`);
  process.exit(1);
}

