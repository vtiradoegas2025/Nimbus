import assert from "node:assert/strict";
import { backgroundAt, backgroundAffectsOpacity, focusedVolume, volumeBackgroundProfile } from "../../../../ui/src/lib/vtk/volumeBackground.ts";
import { prepareVtkRequest } from "../../../../ui/src/lib/vtk/vtkPreparation.ts";
import { vtkPreparationIdentity } from "../../../../ui/src/lib/vtk/vtkPreparationClient.ts";

const shape: [number, number, number] = [20, 20, 12];
const data = new Float32Array(shape.reduce((a, b) => a * b));
for (let x = 0; x < 20; x++) for (let y = 0; y < 20; y++) for (let z = 0; z < 12; z++) {
  data[(x * 20 + y) * 12 + z] = 5 + z * 0.1 + (x >= 8 && x <= 10 && y >= 8 && y <= 10 && z >= 4 && z <= 6 ? 5 : 0);
}
const request = { kind: "scalar" as const, field: { shape, data }, grid: { drM: 100, dyM: 100, dzM: 100 } };
const image = prepareVtkRequest(request);
const original = image.values.slice();
const profile = volumeBackgroundProfile(image, true);
assert.ok(Math.abs(profile.values[4] - 5.4) < 1e-6);
const focus = focusedVolume(image, profile, true);
assert.ok(focus.values.length < image.values.length / 4);
assert.deepEqual(image.values, original, "selection never changes original scalars");
const offsets = focus.origin.map((value, axis) => Math.round((value - image.origin[axis]) / image.spacing[axis]));
for (let z = 0; z < focus.shape[2]; z++) for (let y = 0; y < focus.shape[1]; y++) for (let x = 0; x < focus.shape[0]; x++) {
  assert.equal(focus.values[x + focus.shape[0] * (y + focus.shape[1] * z)],
    image.values[x + offsets[0] + 20 * (y + offsets[1] + 20 * (z + offsets[2]))]);
}
const flat = { ...image, values: Float32Array.from(image.values, (_, index) => 5 + Math.floor(index / 400) * 0.1) };
assert.equal(focusedVolume(flat, profile, false).focusEmpty, true, "a horizontally stratified brick can skip upload");
assert.equal(focusedVolume(flat, profile, true).values.length, 8, "an empty reduced view retains only a tiny transparent placeholder");
assert.notEqual(vtkPreparationIdentity(request), vtkPreparationIdentity({ ...request, background: profile, cropBackground: true }));
assert.equal(backgroundAt({ ...profile, heights: [0, 100], values: [2, 4] }, 25), 2.5);
assert.equal(backgroundAt(profile, -10000), profile.values[0]);
assert.equal(backgroundAffectsOpacity(profile, [[0, 0], [10, 0], [20, 1]]), false,
  "an already-transparent background needs no per-sample profile lookup");
assert.equal(backgroundAffectsOpacity(profile, [[0, 0], [2, 0], [8, 1]]), true);
const boundary = { ...flat, values: flat.values.slice() };
for (let z = 0; z < 12; z++) for (let y = 0; y < 20; y++) boundary.values[20 * (y + 20 * z)] *= 0.5;
const interior: [number, number, number, number] = [image.origin[0] + 100, image.origin[0] + 1800, image.origin[1] + 100, image.origin[1] + 1800];
const boundaryProfile = volumeBackgroundProfile(boundary, false, interior, [100, 100]);
assert.ok(boundaryProfile.threshold < 1e-4, "a boundary stencil cannot set the interior contrast threshold");
assert.equal(focusedVolume(boundary, boundaryProfile, false).focusEmpty, true,
  "a boundary-only sheet consumes no storm-structure brick");
console.log("Volume background: raw colors/scalars preserved, conservative support crop, uniform-brick suppression and distinct preparation identities");
