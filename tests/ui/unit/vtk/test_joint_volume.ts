import assert from "node:assert/strict";
import shader from "@kitware/vtk.js/Rendering/OpenGL/glsl/vtkVolumeFS.glsl.js";
import vtkShaderProgram from "@kitware/vtk.js/Rendering/OpenGL/ShaderProgram.js";
import { configureCoherentVolumeSampling } from "../../../../ui/src/lib/vtk/coherentVolumeSampling.ts";
import { configureJointVolumeShader } from "../../../../ui/src/lib/vtk/jointVolumeShader.ts";
import { jointVolumeGeometry, jointVolumeViewGeometry } from "../../../../ui/src/lib/vtk/jointVolumeGeometry.ts";

let properties: any = {};
const mapper = { getViewSpecificProperties: () => properties, setViewSpecificProperties: (value: object) => { properties = value; } };
configureCoherentVolumeSampling(mapper);
configureJointVolumeShader(mapper);
let result = shader;
for (const rule of properties.OpenGL.ShaderReplacements) {
  const next = vtkShaderProgram.substitute(result, rule.originalValue, rule.replacementValue, rule.replaceAll).result;
  assert.ok(next !== result, `joint shader replacement must match: ${rule.originalValue}`);
  result = next;
}
const inputs = [
  { getOrigin: () => [0.5, 0.5, 0.5], getSpacing: () => [1, 1, 1], getDimensions: () => [2, 2, 2], getDirection: () => [1,0,0,0,1,0,0,0,1] },
  { getOrigin: () => [0.25, 0.25, 1.25], getSpacing: () => [0.5, 0.5, 0.5], getDimensions: () => [4, 4, 4], getDirection: () => [1,0,0,0,1,0,0,0,1] },
];
const geometry = jointVolumeGeometry(inputs);
assert.deepEqual(geometry.bounds, [0,2,0,2,0,3]);
assert.deepEqual(geometry.lower, [0,0,0]); assert.deepEqual(geometry.upper, [1,1,1.5]);
for (const point of [[0,0,0], [0.5,1,1.25], [2,2,3]]) for (const [i, input] of inputs.entries()) {
  const expected = point.map((value, axis) => (value - input.getOrigin()[axis]) / input.getSpacing()[axis] / input.getDimensions()[axis] + 0.5 / input.getDimensions()[axis]);
  const actual = point.map((value, axis) => value / 2 * geometry.layers[i].scale[axis] + geometry.layers[i].shift[axis]);
  assert.deepEqual(actual, expected);
}
assert.throws(() => jointVolumeGeometry([{ ...inputs[0], getSpacing: () => [1, 0, 1] }]), /finite/);
assert.throws(() => jointVolumeGeometry([]), /one to four/);
const shifted = jointVolumeGeometry([{ ...inputs[1], getOrigin: () => [1e9 + 0.25, 1e9 + 0.25, 1e9 + 0.25] }]);
const view = [0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 1, 0, 1e9 + 1, -1e9 - 1, -1e9 - 1, 1];
const mapped = jointVolumeViewGeometry(shifted, view)[0];
const pointVC = [-0.25, 0.125, 0.375];
const index = [0, 1, 2].map((row) => pointVC.reduce((sum, value, column) =>
  sum + Math.fround(mapped.toIndex[column * 3 + row]) * (value - Math.fround(mapped.origin[column])), 0));
assert.deepEqual(index, [0.5625, 0.625, 0.6875]);
console.log("Joint shader-template and independent lattice mapping checks passed");
