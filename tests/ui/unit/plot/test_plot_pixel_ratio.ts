import assert from "node:assert/strict";
import { plotViewerPixelRatio } from "../../../../ui/src/lib/plot/plotPixelRatio.ts";
for (const ratio of [1, 1.25, 1.5, 2, 3, 4]) assert.equal(plotViewerPixelRatio(ratio), ratio);
for (const ratio of [NaN, Infinity, -1, 0]) assert.equal(plotViewerPixelRatio(ratio), 1);
assert.equal(plotViewerPixelRatio(0.5), 1);
assert.equal(plotViewerPixelRatio(8), 4, "Plotly accepts backing ratios only from one through four");
console.log("Plotly backing density follows the viewer within its supported range.");
