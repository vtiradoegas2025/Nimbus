import assert from "node:assert/strict";
import { encodeNpy, encodeCsv } from "../../../../ui/src/lib/platform/pythonExport.ts";
import { decodePythonNumericArray, pythonArrayPreview } from "../../../../ui/src/lib/platform/pythonNumericArray.ts";
import { timeHeightGeometry } from "../../../../ui/src/lib/plot/timeHeightGeometry.ts";
import type { GridInfo } from "../../../../ui/src/lib/run/runOutputs.ts";

for (const source of [Float32Array.of(1.25, -2), Float64Array.of(1, 1 + 1e-9),
  BigInt64Array.of(16777216n, 16777217n, 9007199254740993n),
  BigUint64Array.of(18446744073709551615n), Int16Array.of(-32768, 32767)]) {
  const bytes = encodeNpy(source, [source.length]);
  const headerLength = new DataView(bytes.buffer).getUint16(8, true);
  const header = new TextDecoder().decode(bytes.subarray(10, 10 + headerLength));
  const dtype = /'descr': '([^']+)'/.exec(header)![1];
  assert.deepEqual(decodePythonNumericArray(bytes.slice(10 + headerLength), dtype), source);
}
assert.equal(encodeCsv(BigInt64Array.of(9007199254740993n), [1]), "9007199254740993\n");
assert.equal(Number(encodeCsv(Float64Array.of(1 + 1e-9), [1]).trim()), 1 + 1e-9);
assert.equal(pythonArrayPreview(BigInt64Array.of(9007199254740993n)), null);
assert.deepEqual(pythonArrayPreview(BigInt64Array.of(16777216n, 16777217n)), Float64Array.of(16777216, 16777217));
assert.throws(() => decodePythonNumericArray(new Uint8Array(3), "<f8"));
assert.throws(() => encodeNpy(Float64Array.of(1), [2]));

const grid = (z: number[], lower = 0, upper = z.length - 1) => ({
  coordinateSystem: "cartesian", nr: 1, nth: 1, nz: z.length, drM: 100, dzM: 100,
  scalarCoordinatesM: [[50], [50], z], validIndexRange: { lower: [0, 0, lower], upper: [0, 0, upper] },
}) as GridInfo;
const result = timeHeightGeometry([
  { step: 0, timeS: 0, values: [-99, 10, 20], grid: grid([-50, 50, 150], 1, 2) },
  { step: 1, timeS: 2, values: [], grid: grid([50, 150]) },
  { step: 2, timeS: 4, values: [30, 40, 50], grid: grid([100, 300, 700]) },
], 2)!;
assert.ok(!("error" in result));
assert.deepEqual(result.groups.map((group) => group.y), [[0.05, 0.15], [0.1, 0.3, 0.7]]);
assert.deepEqual(result.groups.map((group) => group.x), [[-1, 1], [3, 5]]);
assert.deepEqual(result.groups.map((group) => group.z), [[[10], [20]], [[30], [40], [50]]]);
assert.deepEqual(result.range, { min: 10, max: 50 });
const uniform = timeHeightGeometry([0, 1].map((step) => ({ step, timeS: step, grid: grid([50, 150]), values: [step, NaN] })), 2)!;
assert.ok(!("error" in uniform));
assert.equal(uniform.groups.length, 1);
assert.deepEqual(uniform.groups[0].z, [[0, 1], [null, null]]);
console.log("Numerical export, preview precision, changing grids, halo crop, and missing-interval checks passed");

const { trajectoryBudgetSeries } = await import("../../../../ui/src/lib/plot/trajectoryBudgetSeries.ts");
const trajectory = { shape: [1, 3, 9], data: Float32Array.from([
  0, 0, 0, 1000, 0, 0, 1, 2, 3,
  1, 0, 0, 2000, 0, 0, 4, NaN, 6,
  2, 0, 0, 3000, 0, 0, 7, 8, 9,
]) };
assert.deepEqual(trajectoryBudgetSeries(trajectory, 0, "t"), { xs: [0, 1, 2], term: [[1, null, 7], [2, null, 8], [3, null, 9]], sum: [6, null, 24] });
assert.deepEqual(trajectoryBudgetSeries(trajectory, 0, "z")?.xs, [1, 2, 3]);
console.log("Internal trajectory gaps remain gaps in every budget curve");

const irregular = timeHeightGeometry([0, 2, 8].map((timeS, step) => ({ step, timeS, grid: grid([50]), values: [step] })), 2)!;
assert.ok(!("error" in irregular));
assert.deepEqual(irregular.groups[0].times, [0, 2, 8]);
assert.deepEqual(irregular.groups[0].x, [-1, 1, 5, 11]);
const mixed = timeHeightGeometry([
  { step: 0, timeS: 0, grid: grid([50]), values: [1] },
  { step: 1, timeS: 1, grid: null, values: [2] },
], 2)!;
assert.ok("error" in mixed);
console.log("Irregular sample times and incompatible-coordinate rejection passed");
