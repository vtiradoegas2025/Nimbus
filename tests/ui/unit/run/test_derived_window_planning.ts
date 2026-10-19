import assert from "node:assert/strict";
import { partitionDerivedWindow } from "../../../../ui/src/lib/run/derivedWindowPlanning.ts";
import { aggregateDerivedWindow, assembleWindowedField } from "../../../../ui/src/lib/run/windowedField.ts";
import type { FieldLoadWindow } from "../../../../ui/src/lib/run/runOutputs.ts";

const window: FieldLoadWindow = { origin: [3, 5, 1], extent: [13, 11, 7], stride: [3, 2, 3], aggregation: "cell_mean" };
const sourceShape = [20, 20, 10];
const derived = (tile: FieldLoadWindow) => {
  const data: number[] = [];
  for (let x = 0; x < tile.extent[0]; x++) for (let y = 0; y < tile.extent[1]; y++) for (let z = 0; z < tile.extent[2]; z++) {
    const u = x + tile.origin[0] - 8, v = y + tile.origin[1] - 9;
    data.push(Math.hypot(u, v) + (z + tile.origin[2]) * 0.01);
  }
  return { shape: tile.extent, data: Float32Array.from(data) };
};
for (const aggregation of ["cell_mean", "block_max", "block_min", "magnitude_extreme", "point_sample", "linear_reflectivity_mean"] as const) {
  const request = { ...window, aggregation };
  const tiles = partitionDerivedWindow(request, false, 512);
  assert.ok(tiles.length > 1);
  for (const tile of tiles) {
    assert.ok(tile.extent.reduce((a, b) => a * b, 4) <= 512);
    assert.ok(tile.origin.every((start, axis) => (start - request.origin[axis]) % request.stride[axis] === 0));
  }
  const expected = aggregateDerivedWindow(derived(request), request, sourceShape);
  const actual = assembleWindowedField(tiles.map(tile => aggregateDerivedWindow(derived(tile), tile, sourceShape)), tiles);
  assert.deepEqual(actual.shape, expected.shape);
  assert.deepEqual(actual.data, expected.data, `${aggregation} keeps nonlinear derive-before-reduce semantics`);
  assert.deepEqual(actual.windowGeometry?.origin, expected.windowGeometry?.origin);
}
const beast: FieldLoadWindow = { origin: [0, 0, 0], extent: [512, 512, 160], stride: [8, 8, 4], aggregation: "block_max", maxOutputBytes: 655360 };
for (const terrain of [false, true]) {
  const tiles = partitionDerivedWindow({ ...beast, maxOutputBytes: terrain ? 5898240 : 655360 }, terrain);
  assert.equal(tiles.reduce((sum, tile) => sum + tile.extent.reduce((a, b) => a * b, 1), 0), 512 * 512 * 160);
  assert.ok(tiles.every(tile => tile.extent.reduce((a, b) => a * b, terrain ? 36 : 4) + (terrain ? 7 : 0) <= 16 * 1024 * 1024));
}
assert.throws(() => partitionDerivedWindow({ ...window, stride: [13, 11, 7] }, false, 512), /One derived-field aggregation cell/);
assert.throws(() => partitionDerivedWindow({ ...window, extent: [NaN, 1, 1] }), /Invalid/);
assert.throws(() => partitionDerivedWindow({ ...window, maxOutputBytes: 1 }), /output exceeds/);
console.log("Derived dependency tiles preserve complete nonlinear aggregation bins within native limits");
