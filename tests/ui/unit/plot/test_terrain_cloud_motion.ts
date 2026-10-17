import assert from "node:assert/strict";
import { cloudCoordinateError } from "../../../../ui/src/lib/plot/cloudCoordinates.ts";
import { cloudWindUnavailable, estimateCloudWind, planCloudWindWindow } from "../../../../ui/src/lib/plot/cloudWind.ts";
import { planCloudMotion } from "../../../../ui/src/lib/plot/cloudMotion.ts";
import type { StepInfo } from "../../../../ui/src/lib/run/runOutputs.ts";
import type { Npy3D } from "../../../../ui/src/lib/platform/npy.ts";
import { prepareVtkRequest } from "../../../../ui/src/lib/vtk/vtkPreparation.ts";

const grid = { nr: 4, nth: 4, nz: 4, drM: 100, dyM: 100, dzM: 80,
  terrainFollowing: true, horizontalScalarOriginM: [0, 0] as [number, number] };
const step: StepInfo = { index: 0, simulationTimeS: 0, grid, fields: ["u", "v", "w"],
  fieldUnits: { u: "m/s", v: "m/s", w: "m/s" }, publicationState: "complete", publicationGeneration: "a" };
assert.equal(cloudWindUnavailable(step), null);
const request = planCloudWindWindow(grid)!;
assert.ok(request.maxOutputBytes! >= 64 * 36);
const cloud = prepareVtkRequest({ kind: "scalar", field: { shape: [12, 12, 12], data: new Float32Array(1728).fill(0.001) },
  grid: { drM: 100, dyM: 100, dzM: 100, scalarOriginM: [0, 0, 0] } });
const x = new Float64Array(64), y = new Float64Array(64), z = new Float64Array(64), volume = new Float64Array(64);
for (let i = 0; i < 64; i++) {
  x[i] = Math.floor(i / 16) * 100; y[i] = Math.floor(i / 4) % 4 * 100;
  const dz = 80 + x[i] * 0.04;
  z[i] = x[i] * 0.1 + (i % 4 + 0.5) * dz;
  volume[i] = 10000 * dz;
}
const field = (fn: (i: number) => number): Npy3D => ({ shape: [4,4,4], data: Float32Array.from(x, (_, i) => fn(i)),
  windowGeometry: { aggregation: "point_sample", weighting: "physical_cell_volume", sampleLocation: "source_cell_center",
    sourceShape: [4,4,4], origin: [0,0,0], extent: [4,4,4], stride: [1,1,1],
    centroidsM: { x, y, z }, volumesM3: volume } });
const fields = { u: field(i => 2 + 0.01 * x[i] + 0.002 * z[i]), v: field(i => 3 - 0.005 * y[i]), w: field(i => 1 + 0.004 * z[i]) };
const fit = estimateCloudWind(step, request, fields, cloud, null);
assert.ok(fit.ok);
for (const [i, value] of [0.01,0,0.002,0,-0.005,0,0,0,0.004].entries()) assert.ok(Math.abs(fit.summary.gradientPerS[i] - value) < 1e-7);
assert.ok(fit.summary.residualMs < 1e-6);
const mismatched = { ...fields.v, windowGeometry: { ...fields.v.windowGeometry!, centroidsM: { x, y, z: Float64Array.from(z, v => v + 1) } } };
assert.equal(estimateCloudWind(step, request, { ...fields, v: mismatched }, cloud, null).ok, false);
const moving = { ...step, motion: { mode: "moving_nest", horizontalFrameContract: "constant_translation_earth_wind_grid_transport_v1", windsEarthRelative: true } };
assert.equal(cloudCoordinateError(moving), null);
assert.ok(cloudCoordinateError({ ...moving, motion: { ...moving.motion, horizontalFrameContract: "future" } }));
const before = { key: "root:0", domainKey: "root:physical", step: 0, timeS: 0, center: [550,550,550] as const, cloud, context: cloud };
const fine = { ...cloud, shape: [23,23,23] as [number,number,number], spacing: [50,50,50] as [number,number,number], values: new Float32Array(23 ** 3).fill(0.001) };
const after = { ...before, key: "child:4", step: 1, timeS: 30, cloud: fine };
const transition = planCloudMotion(before, after);
assert.ok(transition.ok);
assert.deepEqual(transition.displacement, [0,0,0]);
assert.ok(transition.sampledPoints <= 65536);
assert.equal(planCloudMotion(before, { ...after, coordinateError: "unknown coordinates" }).ok, false);
console.log("Terrain wind affine reference, physical-geometry mismatch, Earth registration and native-resolution transition checks passed");
