/*
 * Taking part of a field before it crosses into the interpreter.
 *
 * This is what removes a constraint that should never have been the user's
 * problem. The Python panel copies a field whole into a 32-bit interpreter, so
 * the size of a run decided whether the panel could be used at all — a
 * 512x512x160 field is 160 MiB, a few exhaust the webview, and the only lever
 * was to refuse. Reducing at the reader means the interpreter receives what
 * was asked for rather than what the file happens to be, and a domain of any
 * size is workable at some resolution.
 *
 * So the properties here are about trust in the reduced array: that it holds
 * the cells it claims to, that a crop and a stride compose in the documented
 * order, and that asking for everything costs nothing.
 *
 * Run:
 *   node --experimental-strip-types --import ../tests/ui/register-ts-resolve.mjs \
 *     ../tests/ui/unit/run/test_field_window.ts
 */

import {
  createSubsampledWindowAssembly,
  planPythonDenseFieldTiles,
  planPythonFieldTiles,
  resolveWindow,
  strideForBudget,
  windowedShape,
  windowField,
} from "../../../../ui/src/lib/run/fieldWindow.ts";
import type { NpyN } from "../../../../ui/src/lib/platform/npy.ts";
import { nativeFieldWindowBodyBytes } from "../../../../ui/src/lib/dataset/nativeFieldWindow.ts";

let passed = 0;
const failures: string[] = [];
function check(condition: boolean, label: string): void {
  if (condition) passed += 1;
  else failures.push(`FAIL ${label}`);
}

/** A field whose every cell states its own index, so a gather can be checked. */
function ramp(shape: number[]): NpyN {
  const total = shape.reduce((a, b) => a * b, 1);
  const data = new Float32Array(total);
  for (let i = 0; i < total; i += 1) data[i] = i;
  return { shape, data };
}

const VOLUME = ramp([6, 5, 4]);
const at = (i: number, j: number, k: number) => (i * 5 + j) * 4 + k;

// ---------- 1. Everything means everything, and costs nothing ----------

check(windowField(VOLUME, undefined) === VOLUME, "no window returns the same array");
check(
  windowField(VOLUME, { stride: [1, 1, 1], box: null }) === VOLUME,
  "a window asking for all of it is not a copy either",
);

// ---------- 2. A stride gathers the cells it claims to ----------

const strided = windowField(VOLUME, { stride: [2, 2, 2], box: null });
check(strided.shape.join() === "3,3,2", `stride shape, got ${strided.shape.join()}`);
check(strided.data[0] === at(0, 0, 0), "first strided cell");
check(strided.data[1] === at(0, 0, 2), "stride steps the fastest axis");
check(strided.data[2] === at(0, 2, 0), "and wraps to the next row");
check(
  strided.data[strided.data.length - 1] === at(4, 4, 2),
  "and the last cell is the last one on the stride, not the last of the field",
);

// A stride that does not divide the extent keeps the cells that exist rather
// than running off the end.
const odd = windowField(VOLUME, { stride: [4, 3, 3], box: null });
check(odd.shape.join() === "2,2,2", `ragged stride shape, got ${odd.shape.join()}`);
check(odd.data[odd.data.length - 1] === at(4, 3, 3), "ragged stride ends inside the field");

// ---------- 3. A box crops, and composes with the stride ----------

const cropped = windowField(VOLUME, { stride: [1, 1, 1], box: [[2, 5], [1, 4], [0, 2]] });
check(cropped.shape.join() === "3,3,2", `box shape, got ${cropped.shape.join()}`);
check(cropped.data[0] === at(2, 1, 0), "the box starts where it says");

const both = windowField(VOLUME, { stride: [2, 1, 1], box: [[2, 6], [0, 5], [0, 4]] });
check(both.shape.join() === "2,5,4", `box+stride shape, got ${both.shape.join()}`);
check(
  both.data[0] === at(2, 0, 0) && both.data[both.data.length - 1] === at(4, 4, 3),
  "the box is applied first and the stride walks inside it",
);

// ---------- 4. Nonsense is clamped, not refused ----------

const past = windowField(VOLUME, { stride: [1, 1, 1], box: [[0, 999], [0, 999], [0, 999]] });
check(past.shape.join() === "6,5,4", "bounds past the end mean the end");
const inverted = windowField(VOLUME, { stride: [1, 1, 1], box: [[4, 2], [0, 5], [0, 4]] });
check(inverted.shape[0] === 0, "a backwards box yields nothing rather than reading wildly");
const zeroStride = windowField(VOLUME, { stride: [0, -3, 1], box: null });
check(zeroStride.shape.join() === "6,5,4", "a stride below one is one, not a division by zero");

// ---------- 5. Rank 2 and rank 1, which the manifest also produces ----------

const column = windowField(ramp([4, 3]), { stride: [2, 1], box: null });
check(column.shape.join() === "2,3", "a column diagnostic is rank 2 and windows too");
check(column.data[0] === 0 && column.data[3] === 6, "rank-2 gather walks rows");
const profile = windowField(ramp([5]), { stride: [2], box: null });
check(profile.shape.join() === "3" && profile.data[2] === 4, "rank 1 works");
let refusedRank4 = false;
try {
  windowField(ramp([2, 2, 2, 2]), { stride: [2, 1, 1, 1], box: null });
} catch {
  refusedRank4 = true;
}
check(refusedRank4, "an unsupported rank is refused rather than silently unwindowed");

// ---------- 6. Shape can be predicted without doing the work ----------

for (const window of [
  { stride: [2, 3, 1], box: null },
  { stride: [1, 1, 1], box: [[1, 4], [0, 5], [2, 3]] as [number, number][] },
  { stride: [3, 2, 2], box: [[0, 6], [1, 5], [0, 4]] as [number, number][] },
]) {
  check(
    windowedShape(VOLUME.shape, window).join() === windowField(VOLUME, window).shape.join(),
    "windowedShape agrees with the gather it predicts",
  );
}
const resolved = resolveWindow([6, 5, 4], { stride: [2], box: null });
check(
  resolved.steps.join() === "2,1,1",
  "a stride shorter than the rank leaves the remaining axes whole",
);

// ---------- 7. A stride that brings a real run inside a budget ----------

const BEAST_XL = [512, 512, 160];
const budget = 64 * 1024 * 1024;
const stride = strideForBudget(BEAST_XL, budget);
check(stride > 1, "a 160 MiB field needs reducing to fit 64 MiB");
const reduced = windowedShape(BEAST_XL, { stride: [stride, stride, 1], box: null });
check(
  reduced.reduce((a, b) => a * b, 1) * 4 <= budget,
  `stride ${stride} actually brings it under, got `
  + `${Math.round(reduced.reduce((a, b) => a * b, 1) * 4 / 1024 / 1024)} MiB`,
);
check(
  reduced[2] === BEAST_XL[2],
  "and leaves the vertical alone, which is where the structure is",
);
check(
  strideForBudget([64, 64, 20], budget) === 1,
  "a field that already fits is not reduced for nothing",
);

// ---------- 8. Native transport tiles preserve Python point sampling ----------

const pythonWindow = {
  stride: [2, 2, 3],
  box: [[1, 6], [0, 5], [1, 4]] as [number, number][],
};
const pythonPlan = planPythonFieldTiles(VOLUME.shape, pythonWindow, 32);
check(pythonPlan.windows.length > 1, "large Python windows use bounded sparse transport tiles");
check(
  pythonPlan.windows.every((request) =>
    request.aggregation === "point_sample"
    && request.stride.join() === pythonWindow.stride.join()),
  "transport performs the requested point stride at the native reader",
);
const pythonAssembly = createSubsampledWindowAssembly(VOLUME.shape, pythonWindow);
for (const request of pythonPlan.windows) {
  const box = request.origin.map((start, axis) =>
    [start, start + request.extent[axis]] as [number, number]);
  pythonAssembly.addTile(windowField(VOLUME, { stride: request.stride, box }), request);
}
const pythonExpected = windowField(VOLUME, pythonWindow);
check(
  pythonAssembly.field.shape.join() === pythonExpected.shape.join()
    && [...pythonAssembly.field.data].join() === [...pythonExpected.data].join(),
  "tile assembly is identical to Python's crop-then-point-stride semantics",
);

const densePlan = planPythonDenseFieldTiles(VOLUME.shape, pythonWindow, 32);
check(
  densePlan.windows.length > pythonPlan.windows.length
    && densePlan.windows.every((request) =>
      request.aggregation === "cell_mean" && request.stride.every((step) => step === 1)),
  "compressed/terrain fallback bounds exact dense tiles instead of allocating the source field",
);
const denseAssembly = createSubsampledWindowAssembly(VOLUME.shape, pythonWindow);
for (const request of densePlan.windows) {
  const box = request.origin.map((start, axis) =>
    [start, start + request.extent[axis]] as [number, number]);
  denseAssembly.addTile(windowField(VOLUME, { stride: request.stride, box }), request);
}
check(
  denseAssembly.field.shape.join() === pythonExpected.shape.join()
    && [...denseAssembly.field.data].join() === [...pythonExpected.data].join(),
  "dense fallback preserves the same crop-then-point-stride result",
);

// ---------- 9. Terrain Python reduction budgets and carries geometry ----------

const terrainPlan = planPythonFieldTiles(VOLUME.shape, pythonWindow, 360, 36, "cell_mean");
check(
  terrainPlan.outputBytes === nativeFieldWindowBodyBytes(
    terrainPlan.outputShape.reduce((a, b) => a * b, 1), 32,
  ),
  "terrain Python planning counts value, centroid, volume, and alignment bytes",
);
check(
  terrainPlan.windows.every((request) => request.aggregation === "cell_mean"),
  "terrain remap requests physical-volume blocks rather than sparse point samples",
);
const terrainReflectivityPlan = planPythonFieldTiles(
  VOLUME.shape,
  pythonWindow,
  360,
  36,
  "linear_reflectivity_mean",
);
check(
  terrainReflectivityPlan.windows.every((request) =>
    request.aggregation === "linear_reflectivity_mean"
      && request.extent.every((extent, axis) => extent >= request.stride[axis])),
  "terrain reflectivity reduces complete source blocks in linear-Z space",
);
const terrainAssembly = createSubsampledWindowAssembly(VOLUME.shape, pythonWindow);
for (const request of terrainPlan.windows) {
  const shape = request.extent.map((extent, axis) =>
    Math.floor((extent - 1) / request.stride[axis]) + 1);
  const count = shape.reduce((a, b) => a * b, 1);
  terrainAssembly.addTile({
    shape,
    data: Float32Array.from({ length: count }, (_, index) => index + 1),
    windowGeometry: {
      aggregation: "cell_mean",
      weighting: "physical_cell_volume",
      sampleLocation: "physical_volume_centroid",
      centroidsM: {
        x: Float64Array.from({ length: count }, (_, index) => request.origin[0] + index + 0.5),
        y: Float64Array.from({ length: count }, (_, index) => request.origin[1] + index + 0.5),
        z: Float64Array.from({ length: count }, (_, index) => request.origin[2] + index + 0.5),
      },
      volumesM3: new Float64Array(count).fill(2),
      sourceShape: [...VOLUME.shape],
      origin: [...request.origin],
      extent: [...request.extent],
      stride: [...request.stride],
      sourceFormat: "npy",
    },
  }, request);
}
check(
  terrainAssembly.field.windowGeometry?.volumesM3?.length === terrainAssembly.field.data.length
    && terrainAssembly.field.windowGeometry.volumesM3.every((volume) => volume === 2),
  "terrain tile assembly preserves one exact physical volume per Python cell",
);
check(
  terrainAssembly.field.windowGeometry?.sourceFormat === "npy",
  "Python tile assembly preserves the serving storage format",
);

console.log(`\nfield window tests: ${passed} passed, ${failures.length} failed.`);
if (failures.length) {
  console.log("\nFailures:");
  for (const failure of failures) console.log(`  ${failure}`);
  process.exit(1);
}
