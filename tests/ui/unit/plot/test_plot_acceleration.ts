import type { Data } from "plotly.js";
import {
  acceleratePlotlyData,
  DENSE_SCATTER_GL_THRESHOLD,
  plotlyTracePointCount,
  scatterTraceCanUseWebGl,
} from "../../../../ui/src/lib/plot/plotAcceleration.ts";
import { swathProgressSnapshotCount } from "../../../../ui/src/lib/plot/swathAccumulatorProtocol.ts";

let passed = 0;
let failed = 0;

function equal(actual: unknown, expected: unknown, label: string): void {
  if (actual === expected) passed += 1;
  else {
    failed += 1;
    console.error(`FAIL ${label}: expected ${String(expected)}, got ${String(actual)}`);
  }
}

const dense = {
  type: "scatter",
  mode: "lines",
  x: Array.from({ length: DENSE_SCATTER_GL_THRESHOLD }, (_, index) => index),
  y: new Float32Array(DENSE_SCATTER_GL_THRESHOLD),
} as unknown as Data;
const denseImplicitScatter = { ...dense, type: undefined } as unknown as Data;
const small = { type: "scatter", mode: "lines", x: [0, 1], y: [1, 2] } as Data;
const text = { ...dense, mode: "lines+text" } as unknown as Data;
const filled = { ...dense, fill: "tozeroy" } as unknown as Data;
const dashed = { ...dense, line: { dash: "dash" } } as unknown as Data;
const spline = { ...dense, line: { shape: "spline" } } as unknown as Data;
const stacked = { ...dense, stackgroup: "budget", groupnorm: "percent" } as unknown as Data;
const stepped = { ...dense, line: { shape: "hv" } } as unknown as Data;
const heatmap = { type: "heatmap", z: [[1, 2], [3, 4]] } as Data;

equal(plotlyTracePointCount(dense), DENSE_SCATTER_GL_THRESHOLD, "typed arrays count as source points");
equal(scatterTraceCanUseWebGl(dense), true, "dense line qualifies for WebGL");
equal(scatterTraceCanUseWebGl(denseImplicitScatter), true, "implicit Python scatter qualifies for WebGL");
equal(scatterTraceCanUseWebGl(small), false, "small line remains publication-sharp SVG");
equal(scatterTraceCanUseWebGl(text), false, "text semantics remain SVG");
equal(scatterTraceCanUseWebGl(filled), false, "area fill semantics remain SVG");
equal(scatterTraceCanUseWebGl(dashed), false, "dashed line semantics remain SVG");
equal(scatterTraceCanUseWebGl(spline), false, "spline interpolation remains SVG");
equal(scatterTraceCanUseWebGl(stacked), false, "stacked and normalized values remain SVG");
equal(scatterTraceCanUseWebGl(stepped), true, "supported stepwise line remains accelerated");
const semanticTraces = [spline, stacked];
equal(acceleratePlotlyData(semanticTraces).data, semanticTraces, "semantic traces retain their renderer and identity");

const denseResult = acceleratePlotlyData([dense]);
equal(denseResult.data[0].type, "scattergl", "dense line selects scattergl");
equal(denseResult.backend, "webgl", "dense line reports WebGL backend");
equal(denseResult.acceleratedTraceCount, 1, "accelerated trace is counted");
equal((dense as { type: string }).type, "scatter", "source trace is not mutated");

const rasterInput = [heatmap];
const rasterResult = acceleratePlotlyData(rasterInput);
equal(rasterResult.backend, "raster", "heatmap reports maintained raster backend");
equal(rasterResult.data, rasterInput, "unchanged raster input preserves array identity");

const mixedResult = acceleratePlotlyData([dense, heatmap]);
equal(mixedResult.backend, "mixed", "hybrid plot reports mixed backend");

// The exact array identity matters: small plots should not trigger a needless
// Plotly.react because the acceleration policy merely inspected them.
const smallInput = [small];
equal(acceleratePlotlyData(smallInput).data, smallInput, "unchanged data preserves array identity");
equal(swathProgressSnapshotCount(262_143), 24, "small swaths retain rich progressive feedback");
equal(swathProgressSnapshotCount(262_144), 8, "medium swaths bound snapshot materialization");
equal(swathProgressSnapshotCount(1_048_576), 4, "million-cell swaths avoid redundant full copies");

console.log(`Plot acceleration tests: ${passed} passed, ${failed} failed`);
if (failed > 0) process.exit(1);
