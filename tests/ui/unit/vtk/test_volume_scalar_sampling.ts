import assert from "node:assert/strict";
import vtkDataArray from "@kitware/vtk.js/Common/Core/DataArray.js";
import { assertVolumeTextureDimensions, createFloat32VolumeTexture, VOLUME_SCALAR_SAMPLING_GLSL } from "../../../../ui/src/lib/vtk/volumeScalarSampling.ts";

assert.doesNotThrow(() => assertVolumeTextureDimensions([2048, 2, 64], 2048));
for (const dimensions of [[2049, 2, 64], [2, 2049, 64], [2, 2, 2049], [1, 1, 32000]]) {
  assert.throws(() => assertVolumeTextureDimensions(dimensions, 2048), /3D texture limit/);
}
for (const dimensions of [[0, 2, 2], [0.5, 2, 2], [2, 2], [NaN, 2, 2]]) {
  assert.throws(() => assertVolumeTextureDimensions(dimensions, 2048), /positive safe integers/);
}
for (const maximum of [0, NaN, Infinity, 2.5]) {
  assert.throws(() => assertVolumeTextureDimensions([2, 2, 2], maximum), /3D texture limit/);
}
const data = vtkDataArray.newInstance({ values: new Float32Array(8), numberOfComponents: 1 });
const view = { getContext: () => ({ MAX_3D_TEXTURE_SIZE: 0x8073, getParameter: () => 2,
  isContextLost: () => false, getExtension: () => null }) };
assert.throws(() => createFloat32VolumeTexture(view as unknown as Parameters<typeof createFloat32VolumeTexture>[0], [4, 2, 2], data), /3D texture limit/);
assert.equal(data.getData().length, 8);
assert.ok(VOLUME_SCALAR_SAMPLING_GLSL.includes("highp sampler3D"));
assert.ok(VOLUME_SCALAR_SAMPLING_GLSL.includes("texelFetch"));
assert.ok(VOLUME_SCALAR_SAMPLING_GLSL.includes("coordinate * vec3(dimensions) - vec3(0.5)"));
data.delete();
console.log("Exact-axis GPU texture limits reject before allocation; shared Float32 sampler includes highp cell-center interpolation");
