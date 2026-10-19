import { streamlineMagnitudeMaximum } from "../../../../ui/src/lib/vtk/streamlineDirection.ts";
import {
  buildStreamlineArrowGeometry,
  vectorTupleMagnitudes,
  sampleStreamlineMagnitudes,
} from "../../../../ui/src/lib/vtk/streamlineDirection.ts";
import assert from "node:assert/strict";
import vtkImageData from "@kitware/vtk.js/Common/DataModel/ImageData.js";
import vtkPolyData from "@kitware/vtk.js/Common/DataModel/PolyData.js";
import vtkDataArray from "@kitware/vtk.js/Common/Core/DataArray.js";
import vtkImageStreamline from "@kitware/vtk.js/Filters/General/ImageStreamline.js";
import { extractStreamlinesLocally } from "../../../../ui/src/lib/vtk/vtkStreamlineFallback.ts";
import { prepareVtkRequest } from "../../../../ui/src/lib/vtk/vtkPreparation.ts";

const sampled = sampleStreamlineMagnitudes([0, 0, 0, 0.5, 0, 0, 1, 0, 0], (position, vector) => {
  vector.set([4 * position[0] - 2, 0, 0]); return true;
});
assert.deepEqual([...sampled], [2, 0, 2]);
assert.throws(() => sampleStreamlineMagnitudes([0, 0, 0], () => false), /outside/);
assert.throws(() => sampleStreamlineMagnitudes([0, 0, 0], (_, vector) => { vector.fill(NaN); return true; }), /finite/);
assert.throws(() => sampleStreamlineMagnitudes([0, 0], () => true), /complete coordinates/);

const image = vtkImageData.newInstance(); image.setDimensions(2, 2, 2);
const vectorData = vtkDataArray.newInstance({ numberOfComponents: 3, values: Float32Array.from({ length: 24 }, (_, i) => [3, 4, 0][i % 3]) });
image.getPointData().setVectors(vectorData);
const seeds = vtkPolyData.newInstance(); seeds.getPoints().setData(new Float32Array([0.1, 0.1, 0.1]), 3);
const integrator = vtkImageStreamline.newInstance({ integrationStep: 0.01, maximumNumberOfSteps: 10 });
integrator.setInputData(image); integrator.setInputData(seeds, 1);
const integrated = integrator.getOutputData();
assert.equal(integrated.getPointData().getVectors(), null);
const actualMagnitudes = sampleStreamlineMagnitudes(integrated.getPoints().getData(), (position, vector) => integrator.vectorAt(position, vectorData, image, vector));
assert.equal(actualMagnitudes.length, 10);
assert.ok(actualMagnitudes.every(value => Math.abs(value - 5) < 1e-6));
integrator.delete(); seeds.delete(); image.delete(); vectorData.delete();

const field = (value: number) => ({ shape: [2, 2, 2] as [number, number, number], data: new Float32Array(8).fill(value) });
const prepared = prepareVtkRequest({ kind: "vector", fields: [field(3), field(4), field(0)],
  grid: { drM: 1, dyM: 1, dzM: 1, scalarOriginM: [0, 0, 0] } });
const geometry = extractStreamlinesLocally(prepared, { seeds: new Float32Array([0.5, 0.5, 0.5]), integrationStep: 0.01,
  maxSteps: 10, tubeRadius: 0.01, scale: { xy: 2, z: 3 } });
for (const direction of [geometry.forward, geometry.backward]) for (const mesh of [direction.path, direction.arrows]) {
  assert.ok(mesh?.scalars);
  assert.equal(mesh.scalars.length, mesh.points.length / 3);
  assert.ok(mesh.scalars.every(value => Math.abs(value - 5) < 1e-6));
}

let passed = 0;
const failures: string[] = [];
const check = (condition: boolean, label: string) => {
  if (condition) passed += 1;
  else failures.push(`FAIL ${label}`);
};
const close = (a: number, b: number, tolerance = 1e-6) => Math.abs(a - b) <= tolerance;

const magnitudes = vectorTupleMagnitudes(
  Float32Array.from([3, 4, 0, 0, 0, 2]),
  2,
);
check(magnitudes?.length === 2, "one magnitude is emitted per point");
check(close(magnitudes?.[0] ?? -1, 5), "3-4-5 tuple magnitude is exact");
check(close(magnitudes?.[1] ?? -1, 2), "second tuple magnitude is exact");
check(vectorTupleMagnitudes(new Float32Array(5), 2) == null, "mismatched vector tuple count is refused");

const points = Float32Array.from([
  0, 0, 0,
  1, 0, 0,
  2, 0, 0,
  3, 0, 0,
  4, 0, 0,
]);
const lines = Uint32Array.from([5, 0, 1, 2, 3, 4]);
const scalars = Float32Array.from([1, 2, 3, 4, 5]);
const forward = buildStreamlineArrowGeometry(points, lines, "forward", 0.5, scalars)!;
check(forward.markerCount === 1, "short polyline receives one direction cone");
check(forward.points.length === 15, "one cone owns a tip and four base points");
check(forward.polys.length === 21, "one cone owns four sides and one base cell");
check(forward.points[0] > forward.points[3], "forward cone tip lies downstream of its base");
check(forward.scalars?.every((value) => value === forward.scalars![0]) === true,
  "the sampled magnitude is copied to every cone vertex");

const backward = buildStreamlineArrowGeometry(points, lines, "backward", 0.5, scalars)!;
check(backward.points[0] < backward.points[3],
  "backward branch tangent is reversed so its cone points with the vector");

const longPoints = Float32Array.from(Array.from({ length: 30 }, (_, index) =>
  [index, 0, 0]).flat());
const longLines = Uint32Array.from([30, ...Array.from({ length: 30 }, (_, index) => index)]);
const capped = buildStreamlineArrowGeometry(longPoints, longLines, "forward", 1, null, 1)!;
check(capped.markerCount === 1, "direction-marker ceiling is enforced");
check(capped.scalars == null, "uncoloured source does not invent magnitude scalars");

console.log(`\nstreamline direction tests: ${passed} passed, ${failures.length} failed.`);
if (failures.length > 0) {
  console.log(failures.join("\n"));
  process.exit(1);
}

if (streamlineMagnitudeMaximum([new Float32Array([1, 2]), new Float32Array([3]), null]) !== 3) throw new Error("Path color maximum must include both directions");
if (streamlineMagnitudeMaximum([null, new Float32Array()]) !== 0) throw new Error("Empty paths have no magnitude range");
