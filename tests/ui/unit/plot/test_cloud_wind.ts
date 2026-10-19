import assert from "node:assert/strict";
import { cloudWindUnavailable, estimateCloudWind as estimateThreeDWind, planCloudWindWindow } from "../../../../ui/src/lib/plot/cloudWind.ts";
import { planCloudMotion } from "../../../../ui/src/lib/plot/cloudMotion.ts";
import type { CloudMotionFrame } from "../../../../ui/src/lib/plot/cloudMotion.ts";
import type { Npy3D } from "../../../../ui/src/lib/platform/npy.ts";
import type { StepInfo } from "../../../../ui/src/lib/run/runOutputs.ts";
import { prepareVtkRequest } from "../../../../ui/src/lib/vtk/vtkPreparation.ts";
const grid = { nr: 4, nth: 4, nz: 4, drM: 100, dyM: 100, dzM: 100, scalarOriginM: [0, 0, 0] as [number, number, number] };
const step: StepInfo = { index: 0, simulationTimeS: 0, fields: ["u", "v", "w"], fieldUnits: { u: "m/s", v: "m s-1", w: "m/s" }, grid, publicationGeneration: "test", publicationState: "complete" };
const window = planCloudWindWindow(grid)!;
const field = (value: number): Npy3D => ({ shape: [4, 4, 4], data: new Float32Array(64).fill(value), windowGeometry: {
  aggregation: "point_sample", weighting: "uniform_cell_volume", centroidsM: null, sampleLocation: "source_cell_center", sourceShape: [4, 4, 4], origin: [0, 0, 0], extent: [4, 4, 4], stride: [1, 1, 1],
} });
const estimateCloudWind = (step: StepInfo, request: import("../../../../ui/src/lib/run/runOutputs.ts").FieldLoadWindow,
  u: Npy3D, v: Npy3D, cloud: import("../../../../ui/src/lib/vtk/vtkPreparation.ts").PreparedVtkImage,
  context: import("../../../../ui/src/lib/vtk/vtkPreparation.ts").PreparedVtkImage | null) => estimateThreeDWind(step, request, { u, v, w: field(0) }, cloud, context);
const u = field(2), v = field(-1);
const cloud = prepareVtkRequest({ kind: "scalar", field: { shape: [4, 4, 4], data: new Float32Array(64).fill(0.001) }, grid });
const estimate = estimateCloudWind(step, window, u, v, cloud, null);
assert.ok(estimate.ok);
assert.deepEqual(estimate.summary.velocityMs, [2, -1, 0]);
assert.equal(estimate.summary.spreadMs, 0);
assert.equal(estimate.summary.supportedFraction, 1);
const first: CloudMotionFrame = { key: "a", domainKey: "same", step: 0, timeS: 0, cloud, context: null, center: [150, 150, 150], wind: estimate };
const next: CloudMotionFrame = { ...first, key: "b", step: 1, timeS: 20, cloud: { ...cloud, origin: [40, -20, 0] },
  wind: { ok: true, summary: { ...estimate.summary, sourceStep: 1, sourceTimeS: 20 } } };
const plan = planCloudMotion(first, next);
assert.ok(plan.ok);
if (plan.ok) { assert.equal(plan.method, "wind"); assert.deepEqual(plan.displacement, [40, -20, 0]); }
assert.equal(planCloudMotion(first, { ...next, wind: estimate }).ok, false, "stale wind times cannot steer a later cloud frame");
assert.equal(planCloudMotion(first, { ...next, wind: { ok: true, summary: { ...estimate.summary, sourceTimeS: 20, residualMs: 100 } } }).ok, false);
assert.equal(planCloudMotion(first, { ...next, wind: { ok: false, reason: "missing wind" } }).ok, false);
const shifted = estimateCloudWind({ ...step, motion: { mode: "constant_velocity", horizontalFrameContract: "constant_translation_earth_wind_grid_transport_v1", windsEarthRelative: false, frameVelocityMs: [3, 4] } }, window, u, v, cloud, null);
assert.ok(shifted.ok);
if (shifted.ok) assert.deepEqual(shifted.summary.velocityMs, [5, 3, 0]);
assert.ok(cloudWindUnavailable({ ...step, fieldUnits: { u: "knots", v: "m/s" } }));
assert.ok(cloudWindUnavailable({ ...step, motion: { mode: "constant_velocity" } }));
assert.ok(cloudWindUnavailable({ ...step, grid: { ...grid, coordinateSystem: "cylindrical" } }));
assert.ok(cloudWindUnavailable({ ...step, grid: { ...grid, drM: undefined } }));
const invalid = { ...u, data: u.data.slice() }; invalid.data.fill(NaN, 0, 32);
assert.equal(estimateCloudWind(step, window, invalid, v, cloud, null).ok, false);
assert.equal(estimateCloudWind(step, window, { ...u, windowGeometry: undefined }, v, cloud, null).ok, false);
assert.equal(estimateCloudWind(step, window, u, v, { ...cloud, validity: new Uint8Array(64) }, cloud).ok, false,
  "coarse cloud must not fill missing native support when weighting the winds");
const weightedCloud = { ...cloud, values: cloud.values.slice() }, shear = field(0);
for (let x = 0; x < 4; x++) for (let y = 0; y < 4; y++) for (let z = 0; z < 4; z++) {
  weightedCloud.values[x + 4 * (y + 4 * z)] = 0.001 * (x + 1);
  shear.data[(x * 4 + y) * 4 + z] = 2 * x;
}
const weighted = estimateCloudWind(step, window, shear, v, weightedCloud, null);
assert.ok(weighted.ok);
if (weighted.ok) { assert.ok(Math.abs(weighted.summary.velocityMs[0] - 4) < 1e-6); assert.ok(Math.abs(weighted.summary.spreadMs - 2) < 1e-6); }
const beast = planCloudWindWindow({ nr: 512, nth: 512, nz: 160 })!;
assert.deepEqual(beast.stride, [16, 16, 5]);
assert.equal(32 * 32 * beast.extent[2] * 4, 655360);
assert.equal(beast.maxOutputBytes, 131072);
assert.equal(planCloudWindWindow({ nr: 512, nth: 512, nz: 100000 }), null);
assert.equal(planCloudWindWindow({ ...grid, validIndexRange: { lower: [-1, 0, 0], upper: [3, 3, 3] } }), null);
console.log("Cloud winds: bounded sparse reads, weighted transport, Earth-frame correction, stale/missing support, shear rejection, and exact advection passed.");
const massStep = { ...step, fields: [...step.fields, "rho"], fieldUnits: { ...step.fieldUnits, rho: "kg/m^3" } };
const air = field(1), vertical = field(0), east = field(0), north = field(0);
const expectedGradient = [0.001, 0.002, -0.001, -0.002, 0.0005, 0.001, 0.001, -0.001, 0.002];
for (let x = 0; x < 4; x++) for (let y = 0; y < 4; y++) for (let z = 0; z < 4; z++) {
  const index = (x * 4 + y) * 4 + z;
  const position = [100 * x - 150, 100 * y - 150, 100 * z - 150];
  air.data[index] = 0.7 + 0.2 * x;
  for (const [row, target] of [east, north, vertical].entries()) target.data[index] = [3,-2,1][row]
    + position.reduce((sum, value, axis) => sum + expectedGradient[row * 3 + axis] * value, 0);
}
const fitted = estimateThreeDWind(massStep, window, { u: east, v: north, w: vertical, rho: air }, weightedCloud, null);
assert.ok(fitted.ok);
if (fitted.ok) {
  assert.equal(fitted.summary.weighting, "condensate-mass");
  fitted.summary.gradientPerS.forEach((value, index) => assert.ok(Math.abs(value - expectedGradient[index]) < 1e-8));
  assert.ok(fitted.summary.residualMs < 1e-5);
  fitted.summary.velocityMs.forEach((value, row) => assert.ok(Math.abs(value - [3,-2,1][row]
    - fitted.summary.centerM.reduce((sum, position, col) => sum + expectedGradient[row * 3 + col] * (position - 150), 0)) < 1e-6));
}
const badVertical = { ...vertical, data: new Float32Array(64).fill(NaN) };
assert.equal(estimateThreeDWind(massStep, window, { u: east, v: north, w: badVertical, rho: air }, cloud, null).ok, false);
assert.equal(estimateThreeDWind(massStep, window, { u: east, v: north, w: vertical, rho: field(-1) }, cloud, null).ok, false);
console.log("Three-dimensional wind fit: exact affine gradients, density weighting, weighted center velocity, and invalid vertical/density support passed.");

const compressedTall = planCloudWindWindow({ nr: 512, nth: 512, nz: 100000 }, true)!;
assert.equal(compressedTall.maxSourceBytes, 32 * 1024 * 1024);
assert.equal(compressedTall.maxOutputBytes, 131072);
assert.ok(compressedTall.extent.reduce((count, size, axis) => count * Math.ceil(size / compressedTall.stride[axis]), 1) <= 32768);
