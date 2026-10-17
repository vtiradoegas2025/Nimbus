import assert from "node:assert/strict";
import vtkPolyData from "@kitware/vtk.js/Common/DataModel/PolyData.js";
import { applyImmutablePolyDataGeometry } from "../../../../ui/src/lib/vtk/vtkImmutablePolyData.ts";

const target = vtkPolyData.newInstance();
const geometry = {
  points: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
  polys: new Uint32Array([3, 0, 1, 2]),
  strips: new Uint32Array(),
  lines: new Uint32Array(),
  normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
  scalars: new Float32Array([0, 1, 2]),
};
assert.equal(applyImmutablePolyDataGeometry(target, geometry), true);
const scalars = target.getPointData().getScalars();
const normals = target.getPointData().getNormals();
const times = () => [target.getMTime(), target.getPoints().getMTime(), target.getPolys().getMTime(),
  target.getPointData().getMTime(), scalars.getMTime(), normals.getMTime()];
const initialTimes = times();
assert.equal(applyImmutablePolyDataGeometry(target, geometry), false);
assert.deepEqual(times(), initialTimes);
assert.equal(target.getPointData().getScalars(), scalars);
assert.equal(target.getPointData().getNormals(), normals);
assert.equal(applyImmutablePolyDataGeometry(target, { ...geometry, points: new Float32Array(geometry.points.buffer), normals: new Float32Array(geometry.normals.buffer) }), false);
assert.deepEqual(times(), initialTimes);

assert.throws(() => applyImmutablePolyDataGeometry(target, { ...geometry, normals: new Float32Array(3) }));
assert.throws(() => applyImmutablePolyDataGeometry(target, { ...geometry, scalars: new Float32Array(2) }));
assert.throws(() => applyImmutablePolyDataGeometry(target, { ...geometry, points: new Float32Array(8) }));
assert.deepEqual(times(), initialTimes);
assert.deepEqual([...target.getPoints().getData()], [...geometry.points]);

const changed = { ...geometry, points: new Float32Array([10, 0, 0, 11, 0, 0, 10, 1, 0]) };
assert.equal(applyImmutablePolyDataGeometry(target, changed), true);
assert.deepEqual([...target.getPoints().getData()], [...changed.points]);
assert.notEqual(target.getPointData().getNormals(), normals);
assert.deepEqual([...target.getPointData().getNormals().getData()], [...geometry.normals]);
assert.deepEqual([...target.getPointData().getScalars().getData()], [0, 1, 2]);

const points = new Float32Array([...geometry.points, ...changed.points]);
assert.equal(applyImmutablePolyDataGeometry(target, { ...geometry, points: points.subarray(0, 9) }), true);
assert.equal(applyImmutablePolyDataGeometry(target, { ...geometry, points: points.subarray(9) }), true);
assert.deepEqual([...target.getPoints().getData()], [...changed.points]);

const unlit = { points: geometry.points, lines: new Uint32Array([3, 0, 1, 2]) };
assert.equal(applyImmutablePolyDataGeometry(target, unlit), true);
assert.equal(target.getPointData().getArrays().length, 0);
assert.equal(target.getPolys().getData().length, 0);
assert.deepEqual([...target.getLines().getData()], [3, 0, 1, 2]);
assert.equal(applyImmutablePolyDataGeometry(target, unlit), false);
assert.equal(applyImmutablePolyDataGeometry(target, null), true);
assert.equal(target.getPoints().getData().length, 0);
assert.equal(target.getLines().getData().length, 0);
const emptyTime = target.getMTime();
assert.equal(applyImmutablePolyDataGeometry(target, null), false);
assert.equal(target.getMTime(), emptyTime);
target.delete();
console.log("Immutable mesh spans preserve attribute identity/mtime; changed geometry, invalid-input atomicity, and empty ownership clear passed");
