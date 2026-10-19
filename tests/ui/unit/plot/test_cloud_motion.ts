import assert from "node:assert/strict";
import { planCloudMotion, sampleCloudMotion, cloudMotionProbeShape, type CloudMotionFrame } from "../../../../ui/src/lib/plot/cloudMotion.ts";
import { prepareVtkRequest } from "../../../../ui/src/lib/vtk/vtkPreparation.ts";
const image = prepareVtkRequest({ kind: "scalar", field: { shape: [12, 12, 12], data: new Float32Array(1728).fill(0.001) },
  grid: { drM: 100, dyM: 100, dzM: 100 } });
const first: CloudMotionFrame = { key: "a", domainKey: "grid", step: 0, timeS: 0, center: [550, 550, 550], cloud: image, context: null };
const next: CloudMotionFrame = { ...first, key: "b", step: 1, timeS: 60, center: [750, 550, 550], cloud: { ...image, origin: [image.origin[0] + 200, image.origin[1], image.origin[2]] } };
const plan = planCloudMotion(first, next);
assert.ok(plan.ok);
if (plan.ok) { assert.deepEqual(plan.displacement, [200, 0, 0]); assert.equal(plan.difference, 0); }
assert.equal(planCloudMotion(first, { ...next, domainKey: "different" }).ok, false);
assert.equal(planCloudMotion(first, { ...next, timeS: 0 }).ok, false);
assert.equal(planCloudMotion(first, { ...next, step: 0 }).ok, false);
assert.equal(planCloudMotion(first, { ...next, center: [10000, 550, 550] }).ok, false);
assert.equal(planCloudMotion(first, { ...next, center: [NaN, 550, 550] }).ok, false);
assert.equal(planCloudMotion(first, { ...first, step: 1, timeS: 60, cloud: { ...image, values: new Float32Array(1728).fill(0.0001) } }).ok, false);
const mask = new Uint8Array(1728).fill(255); mask[1] = 0;
const masked = { ...image, validity: mask };
assert.ok(sampleCloudMotion(masked, image.origin));
assert.equal(sampleCloudMotion(masked, [image.origin[0] + 50, image.origin[1], image.origin[2]]), null);
assert.equal(sampleCloudMotion(masked, [image.origin[0] - 1, image.origin[1], image.origin[2]]), null);
assert.equal(planCloudMotion(first, { ...next, cloud: { ...next.cloud, validity: mask.fill(0) } }).ok, false);
console.log("Cloud motion: translated support, shape-change rejection, endpoints, and missing interpolation corners passed.");

const half = { ...image, shape: [6, 12, 12] as [number, number, number], values: new Float32Array(864).fill(0.001) };
assert.equal(planCloudMotion(first, { ...next, center: first.center, cloud: half }).ok, false,
  "cloud leaving the incoming footprint must be checked from the previous footprint too");
assert.equal(planCloudMotion({ ...first, cloud: half }, { ...next, center: first.center, cloud: image }).ok, false);
const faint = { ...image, values: new Float32Array(1728).fill(0.0001) };
assert.equal(planCloudMotion({ ...first, context: image }, { ...next, center: first.center, cloud: faint, context: image }).ok, false,
  "an unchanged coarse context must not conceal an incompatible native cloud change");
assert.equal(planCloudMotion({ ...first, context: image }, { ...next, center: first.center, cloud: image, context: faint }).ok, false);
for (const shape of [[1, 1, 1_000_000], [2, 3, 100_000], [512, 512, 160]] as const) {
  const probes = cloudMotionProbeShape(shape);
  assert.ok(probes.reduce((count, size) => count * size, 1) <= 16_384);
  assert.ok(probes.every((size, axis) => size > 0 && size <= shape[axis]));
}
assert.throws(() => cloudMotionProbeShape([Infinity, 1, 1]));
assert.equal(sampleCloudMotion(image, [0, 0]), null);
assert.equal(sampleCloudMotion({ ...image, spacing: [0, 100, 100] }, [0, 0, 0]), null);
assert.equal(planCloudMotion(first, { ...next, cloud: { ...image, values: new Float32Array(1) } }).ok, false);
assert.equal(planCloudMotion(first, { ...next, cloud: { ...image, validity: new Uint8Array(1) } }).ok, false);
const decimal = { ...image, origin: [103.7, 203.1, 17.3] as [number, number, number], spacing: [0.3, 0.7, 0.9] as [number, number, number] };
assert.ok(sampleCloudMotion(decimal, decimal.origin.map((value, axis) => value + 11 * decimal.spacing[axis])));
if (plan.ok) assert.ok(plan.sampledPoints <= 32_768);
console.log("Cloud motion: both endpoint footprints, native/context compatibility, malformed inputs, and anisotropic probe budgets passed.");
const multilinear = { ...image, shape: [2, 2, 2] as [number, number, number], origin: [0, 0, 0] as [number, number, number],
  spacing: [1, 1, 1] as [number, number, number], values: Float32Array.from([1, 2, 4, 8, 16, 32, 64, 128]) };
for (const [x, y, z] of [[0.23, 0.47, 0.61], [1, 0.2, 0.4], [0, 0, 0], [1, 1, 1]]) {
  assert.ok(Math.abs(sampleCloudMotion(multilinear, [x, y, z])! - (1 + x) * (1 + 3 * y) * (1 + 15 * z)) < 1e-10);
}
const lastCornerMissing = { ...multilinear, validity: Uint8Array.from([255, 255, 255, 255, 255, 255, 255, 0]) };
assert.equal(sampleCloudMotion(lastCornerMissing, [0.23, 0.47, 0.61]), null);
assert.equal(sampleCloudMotion(lastCornerMissing, [0, 0, 0]), 1);
assert.equal(planCloudMotion(first, { ...next, step: NaN }).ok, false);
assert.equal(planCloudMotion(first, { ...next, timeS: Infinity }).ok, false);
console.log("Cloud motion: exact multilinear interior/face/corner sampling and invalid frame identities passed.");
