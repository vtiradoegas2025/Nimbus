/*
 * Tests for `src/lib/quiver.ts` — the pure vector-field (quiver) builder used
 * by the VectorField plot.
 *
 * Run via:
 *   cd ui && node --experimental-strip-types scripts/test_quiver.ts
 */

import {
  buildQuiver,
  quiverStrideFor,
  sampledMaxMagnitude,
  MAX_QUIVER_ARROWS,
  type QuiverInput,
} from "../../../../ui/src/lib/plot/quiver.ts";
import {
  describeVectorSampling,
  vectorReferenceKey,
} from "../../../../ui/src/lib/plot/vectorPresentation.ts";

// ---------- Tiny assertion API (matches test_cartesian_interp.ts) ----------

let failed = 0;
let passed = 0;
const failures: string[] = [];

function assertTrue(cond: boolean, label: string): void {
  if (cond) passed++;
  else {
    failed++;
    failures.push(`FAIL ${label}`);
  }
}

function approx(a: number, b: number, eps = 1e-9): boolean {
  return Math.abs(a - b) <= eps;
}

function check(label: string, fn: () => void): void {
  try {
    fn();
  } catch (e) {
    failed++;
    failures.push(`THREW ${label}: ${(e as Error).message}`);
  }
}

// ---------- Helpers ----------

function grid(
  rows: number,
  cols: number,
  fn: (i: number, j: number) => number,
): number[][] {
  const out: number[][] = new Array(rows);
  for (let i = 0; i < rows; i++) {
    out[i] = new Array<number>(cols);
    for (let j = 0; j < cols; j++) out[i][j] = fn(i, j);
  }
  return out;
}

function base(over: Partial<QuiverInput>): QuiverInput {
  const rows = over.rows ?? 4;
  const cols = over.cols ?? 4;
  return {
    u: over.u ?? grid(rows, cols, () => 0),
    v: over.v ?? grid(rows, cols, () => 0),
    rows,
    cols,
    xCoords: over.xCoords,
    yCoords: over.yCoords,
    xSpacing: over.xSpacing ?? 1,
    ySpacing: over.ySpacing ?? 1,
    stride: over.stride ?? 1,
    scale: over.scale ?? 1,
    referenceMagnitude: over.referenceMagnitude,
  };
}

// 9 array entries per arrow: shaft + 2 head segments, each 2 points + a null.
const ENTRIES_PER_ARROW = 9;

// ---------- 1. Uniform field: one arrow per cell, correct magnitude ----------

check("uniform u=1,v=0 produces one arrow per cell with magnitude 1", () => {
  const r = buildQuiver(base({ u: grid(4, 4, () => 1), v: grid(4, 4, () => 0) }));
  assertTrue(r.arrowCount === 16, "16 arrows for a 4x4 stride-1 grid");
  assertTrue(approx(r.maxMagnitude, 1), "max magnitude is 1");
  assertTrue(r.x.length === 16 * ENTRIES_PER_ARROW, "x has 9 entries per arrow");
  assertTrue(r.x.length === r.y.length, "x and y arrays are the same length");
});

// ---------- 2. Zero field: no arrows ----------

check("all-zero field yields no arrows", () => {
  const r = buildQuiver(base({}));
  assertTrue(r.arrowCount === 0, "no arrows");
  assertTrue(r.maxMagnitude === 0, "zero max magnitude");
  assertTrue(r.x.length === 0 && r.y.length === 0, "empty coordinate arrays");
});

// ---------- 3. NaN/Inf samples are skipped, never poison the output ----------

check("a NaN component is skipped and never appears in the output", () => {
  const u = grid(3, 3, (i, j) => (i === 1 && j === 1 ? NaN : 1));
  const r = buildQuiver(base({ rows: 3, cols: 3, u, v: grid(3, 3, () => 0) }));
  assertTrue(r.arrowCount === 8, "8 arrows (the NaN cell is dropped)");
  assertTrue(approx(r.maxMagnitude, 1), "max magnitude ignores the NaN cell");
  const noNaN = r.x.every((e) => e === null || Number.isFinite(e)) &&
    r.y.every((e) => e === null || Number.isFinite(e));
  assertTrue(noNaN, "no NaN/Inf leaks into the coordinate arrays");
});

// ---------- 4. Decimation by stride ----------

check("stride decimates the sampled grid", () => {
  const r = buildQuiver(
    base({ rows: 4, cols: 4, u: grid(4, 4, () => 1), v: grid(4, 4, () => 1), stride: 2 }),
  );
  // i,j in {0,2} -> 2x2 = 4 sampled points.
  assertTrue(r.arrowCount === 4, "stride 2 on 4x4 samples 4 arrows");
});

// ---------- 5. Auto scale: a max-magnitude vector spans stride*spacing ----------

check("the longest arrow spans stride * spacing in display units", () => {
  const r = buildQuiver(
    base({ rows: 2, cols: 2, u: grid(2, 2, () => 2), v: grid(2, 2, () => 0), stride: 1, scale: 1, xSpacing: 1 }),
  );
  // First arrow: tail at index (0,0), head one cell to the +x. Shaft is the
  // first segment: x[0]=tail, x[1]=head, x[2]=null.
  const shaftDx = (r.x[1] as number) - (r.x[0] as number);
  assertTrue(approx(shaftDx, 1), "shaft length equals stride * xSpacing for the max vector");
  assertTrue(approx(r.y[1] as number, r.y[0] as number), "no y displacement for a purely-x vector");
});

// ---------- 6. Physical coordinate placement ----------

check("arrow tails sit on the provided physical coordinates", () => {
  const r = buildQuiver(
    base({
      rows: 2,
      cols: 2,
      u: grid(2, 2, () => 1),
      v: grid(2, 2, () => 0),
      xCoords: [10, 20],
      yCoords: [5, 15],
    }),
  );
  // First sampled cell (i=0, j=0): tail = (xCoords[0], yCoords[0]).
  assertTrue(approx(r.x[0] as number, 10), "tail x = xCoords[0]");
  assertTrue(approx(r.y[0] as number, 5), "tail y = yCoords[0]");
});

// ---------- 7. Shared comparison normalization ----------

check("a shared reference magnitude preserves cross-panel arrow ratios", () => {
  const r = buildQuiver(base({
    rows: 2,
    cols: 2,
    u: grid(2, 2, () => 1),
    v: grid(2, 2, () => 0),
    referenceMagnitude: 2,
  }));
  const shaftDx = (r.x[1] as number) - (r.x[0] as number);
  assertTrue(approx(r.maxMagnitude, 1), "reported maximum remains the panel's actual magnitude");
  assertTrue(approx(shaftDx, 0.5), "arrow length uses the shared reference magnitude");
});

// ---------- 8. Frame-to-frame comparability ----------
//
// REGRESSION: with no reference, buildQuiver normalizes to the CURRENT frame's
// maximum, so a 5 m/s frame and a 50 m/s frame draw identical arrows and an
// animation shows a storm intensifying without the arrows ever growing. The
// caller establishes the reference with sampledMaxMagnitude and holds it across
// frames; arrow length must then be strictly proportional to magnitude.

check("sampledMaxMagnitude matches what buildQuiver normalizes against", () => {
  const u = grid(4, 4, (i, j) => (i === 2 && j === 1 ? 7 : 1));
  const v = grid(4, 4, () => 0);
  assertTrue(approx(sampledMaxMagnitude(u, v, 4, 4, 1), 7), "stride-1 sample finds the peak");
  assertTrue(approx(buildQuiver(base({ u, v })).maxMagnitude, 7), "buildQuiver agrees");
  // Decimation must be reported honestly: at stride 2 the peak cell (2,1) is
  // not sampled, so both the helper and the trace see 1.
  assertTrue(approx(sampledMaxMagnitude(u, v, 4, 4, 2), 1), "stride-2 sample skips the peak cell");
  assertTrue(
    approx(buildQuiver(base({ u, v, stride: 2 })).maxMagnitude, 1),
    "buildQuiver reports the same decimated maximum",
  );
});

check("a held reference makes arrow length proportional to magnitude across frames", () => {
  const weak = grid(2, 2, () => 5);
  const strong = grid(2, 2, () => 50);
  const zero = grid(2, 2, () => 0);
  // The scale a caller holds across both frames: the running peak.
  const reference = Math.max(
    sampledMaxMagnitude(weak, zero, 2, 2, 1),
    sampledMaxMagnitude(strong, zero, 2, 2, 1),
  );
  assertTrue(approx(reference, 50), "the running peak is the stronger frame's maximum");

  const shaft = (u: number[][]) => {
    const r = buildQuiver(base({ rows: 2, cols: 2, u, v: zero, referenceMagnitude: reference }));
    return (r.x[1] as number) - (r.x[0] as number);
  };
  const weakShaft = shaft(weak);
  const strongShaft = shaft(strong);
  assertTrue(approx(strongShaft / weakShaft, 10), "a 10x stronger frame draws 10x longer arrows");
  assertTrue(approx(strongShaft, 1), "the peak frame still spans one cell at scale 1");

  // Without the reference each frame renormalizes to itself: the bug.
  const unpinned = (u: number[][]) => {
    const r = buildQuiver(base({ rows: 2, cols: 2, u, v: zero }));
    return (r.x[1] as number) - (r.x[0] as number);
  };
  assertTrue(
    approx(unpinned(weak), unpinned(strong)),
    "unpinned frames collapse to identical arrows (documents what the reference prevents)",
  );
});

// ---------- Report ----------

/* ---------- Arrow ceiling ----------
 * These arrows are SVG line segments at nine coordinates each. The requested
 * stride is a fixed divisor, so the emitted count still grows with the square
 * of the grid: a 500x500 plan slice at the default stride of 2 is 62,500 arrows
 * and over half a million coordinates in one path, which locks the main thread
 * hard enough to read as an application freeze. */

check("a plane within the ceiling keeps the requested stride", () => {
  assertTrue(quiverStrideFor(40, 40, 2) === 2, "small plane is untouched at stride 2");
  assertTrue(quiverStrideFor(40, 40, 1) === 1, "small plane is untouched at stride 1");
});

check("a dense plane is thinned past the requested stride", () => {
  const step = quiverStrideFor(500, 500, 2);
  assertTrue(step > 2, `500x500 at stride 2 must be thinned, got ${step}`);
  const count = Math.ceil(500 / step) * Math.ceil(500 / step);
  assertTrue(
    count <= MAX_QUIVER_ARROWS,
    `thinned count ${count} must be within ${MAX_QUIVER_ARROWS}`,
  );
  // Smallest sufficient stride, not an over-correction that throws away detail.
  const looser = step - 1;
  assertTrue(
    Math.ceil(500 / looser) * Math.ceil(500 / looser) > MAX_QUIVER_ARROWS,
    "the chosen stride is the smallest one that fits",
  );
});

check("the ceiling never thins below the user's request", () => {
  assertTrue(quiverStrideFor(500, 500, 64) === 64, "an already-coarse request stands");
  assertTrue(quiverStrideFor(10, 10, 4) === 4, "a coarse request on a small plane stands");
});

check("degenerate shapes do not spin", () => {
  assertTrue(quiverStrideFor(0, 0, 2) === 2, "an empty plane returns the request");
  assertTrue(quiverStrideFor(-5, 10, 3) === 3, "a negative extent returns the request");
  assertTrue(quiverStrideFor(1, 1, 1) === 1, "a single cell returns the request");
});

check("buildQuiver enforces the ceiling and reports the stride it used", () => {
  const rows = 400;
  const cols = 400;
  const result = buildQuiver(base({
    rows,
    cols,
    u: grid(rows, cols, () => 1),
    v: grid(rows, cols, () => 1),
    stride: 1,
    scale: 1,
  }));
  assertTrue(
    result.arrowCount <= MAX_QUIVER_ARROWS,
    `emitted ${result.arrowCount} arrows, ceiling is ${MAX_QUIVER_ARROWS}`,
  );
  assertTrue(result.effectiveStride > 1, "the applied stride is reported back to the caller");
  // Nine coordinates per arrow: shaft, plus two head segments, null-separated.
  assertTrue(
    result.x.length === result.arrowCount * 9,
    `coordinate count ${result.x.length} must match ${result.arrowCount} arrows`,
  );
});

check("an unthinned plane reports the stride it was asked for", () => {
  const result = buildQuiver(base({
    rows: 8,
    cols: 8,
    u: grid(8, 8, () => 1),
    v: grid(8, 8, () => 0),
    stride: 2,
    scale: 1,
  }));
  assertTrue(result.effectiveStride === 2, "no thinning means the requested stride is reported");
});

check("a zero field still reports its stride", () => {
  const result = buildQuiver(base({ rows: 400, cols: 400, stride: 1, scale: 1 }));
  assertTrue(result.arrowCount === 0, "a zero field emits no arrows");
  assertTrue(result.effectiveStride > 1, "the early return still reports the applied stride");
});

check("the export scale key uses the same data-coordinate arrow scaling", () => {
  const key = vectorReferenceKey(20, 1, 2, 1, [0, 10], [0, 10]);
  assertTrue(key != null, "physical axes produce a scale key");
  assertTrue(approx(key!.headX - key!.tailX, key!.magnitude / 10),
    "key length is magnitude/reference times the full-reference arrow length");
  assertTrue(key!.headX > key!.tailX, "key arrow points in the positive display direction");
});

check("a vector key is omitted without physical two-point axes", () => {
  assertTrue(vectorReferenceKey(20, 1, 2, 1, undefined, [0, 1]) == null,
    "index fallback does not pretend to be a physical scale key");
});

check("bounded vector aggregation is disclosed in display-axis order", () => {
  const detail = describeVectorSampling({
    shape: [2, 3, 1],
    data: new Float32Array(6),
    windowGeometry: {
      aggregation: "cell_mean",
      weighting: "uniform_cell_volume",
      sampleLocation: "block_center",
      centroidsM: null,
      stride: [4, 2, 1],
    },
  }, 0, 1);
  assertTrue(detail === "native cell mean 2×4", "horizontal × vertical native block is explicit");
});

check("stride-one windows do not claim reduction", () => {
  const detail = describeVectorSampling({
    shape: [2, 2], data: new Float32Array(4),
    windowGeometry: {
      aggregation: "cell_mean", weighting: "uniform_cell_volume",
      sampleLocation: "source_cell_center", centroidsM: null, stride: [1, 1],
    },
  }, 0, 1);
  assertTrue(detail == null, "native-resolution bounded reads remain undisclosed noise");
});

console.log(`\nquiver tests: ${passed} passed, ${failed} failed.`);
if (failures.length > 0) {
  console.log("\nFailures:");
  for (const f of failures) console.log("  " + f);
  process.exit(1);
}
process.exit(0);
