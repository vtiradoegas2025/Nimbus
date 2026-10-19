import assert from "node:assert/strict";
import { deriveTornadicCondensation } from "../../../../ui/src/lib/plot/tornadicCondensation.ts";
import { addCloudPresentationDetail } from "../../../../ui/src/lib/plot/cloudPresentationDetail.ts";
import { prepareVtkRequest } from "../../../../ui/src/lib/vtk/vtkPreparation.ts";

const shape: [number, number, number] = [51, 51, 20];
const grid = { shape, spacing: [100, 100, 100] as [number, number, number], origin: [0, 0, 0] as [number, number, number] };
const index = (x: number, y: number, z: number) => x + 51 * (y + 51 * z);
const q = new Float32Array(51 * 51 * 20), omega = q.slice();
for (let z = 0; z < 20; z++) for (let y = 0; y < 51; y++) for (let x = 0; x < 51; x++) {
  const core = Math.abs(x - 25) <= 1 && Math.abs(y - 25) <= 1;
  omega[index(x, y, z)] = core && z >= 1 && z <= 10 ? 0.2 : 0;
  q[index(x, y, z)] = z >= 12 || (core && z >= 6) ? 0.001 : 0;
}
const derive = (values: Float32Array, rotation = omega, validity?: Uint8Array) => deriveTornadicCondensation(
  { ...grid, values, validity }, { ...grid, values: rotation }, [2500, 2500, 100],
  [2400, 2600, 2400, 2600, 100, 1000], 0.1, 0.00005, true,
);
assert.equal(derive(q).assessment.present, true);
assert.equal(derive(q.slice().fill(NaN)).image, null);
assert.equal(derive(q.slice().fill(Infinity)).image, null);
for (const missingField of ["cloud", "vorticity"]) {
  const cloud = q.slice(), rotation = omega.slice();
  (missingField === "cloud" ? cloud : rotation).fill(NaN, 51 * 51 * 8, 51 * 51 * 9);
  const result = derive(cloud, rotation);
  assert.ok(result.image);
  assert.equal(result.image.validity?.[index(25, 25, 8)], 0);
  assert.equal(result.image.values[index(25, 25, 8)], 0);
  assert.ok(result.image.values.every(Number.isFinite));
  assert.equal(result.image.values[index(25, 25, 12)], 0, "connected selection cannot cross the missing plane");
  assert.ok(result.image.values[index(25, 25, 6)] > 0);
}
const missingAmbient = q.slice();
for (let y = 0; y < 51; y++) for (let x = 0; x < 51; x++)
  if (Math.hypot(x - 25, y - 25) >= 20) missingAmbient[index(x, y, 0)] = NaN;
assert.equal(derive(missingAmbient).assessment.ambientCloudBaseM, null);
assert.equal(derive(missingAmbient).assessment.present, false, "unknown ambient cloud base is not funnel evidence");
const mask = new Uint8Array(q.length).fill(255); mask[index(25, 25, 6)] = 0;
assert.equal(derive(q, omega, mask).image?.validity?.[index(25, 25, 6)], 0);
const diagonal = new Float32Array(27).fill(NaN); diagonal[0] = diagonal[4] = 0.001;
const tiny = { shape: [3, 3, 3] as [number, number, number], spacing: grid.spacing, origin: grid.origin };
const separate = deriveTornadicCondensation({ ...tiny, values: diagonal }, { ...tiny, values: new Float32Array(27).fill(0.2) },
  [100, 100, 0], [0, 200, 0, 200, 0, 200], 0.1, 0.00005, true);
assert.equal(separate.assessment.voxelCount, 1, "diagonal traversal cannot cut across missing interpolation corners");

for (const floor of [0, -1, NaN, Infinity]) {
  const invalid = deriveTornadicCondensation({ ...grid, values: q }, { ...grid, values: omega },
    [2500, 2500, 100], [2400, 2600, 2400, 2600, 100, 1000], 0.1, floor, true);
  assert.equal(invalid.assessment.reason, "invalid-input");
  assert.equal(invalid.image, null);
}
for (const bad of [
  { ...grid, values: q, spacing: [0, 100, 100] as [number, number, number] },
  { ...grid, values: q, origin: [NaN, 0, 0] as [number, number, number] },
  { ...grid, values: q.subarray(1) },
  { ...grid, values: q, validity: new Uint8Array(1) },
]) {
  assert.equal(deriveTornadicCondensation(bad, { ...grid, values: omega }, [2500, 2500, 100],
    [2400, 2600, 2400, 2600, 100, 1000], 0.1, 0.00005, true).assessment.reason, "invalid-input");
}

const source = { shape: [2, 2, 2] as [number, number, number], spacing: [500, 500, 500] as [number, number, number],
  origin: [0, 0, 0] as [number, number, number], data: new Float32Array([0.001, 0.002, NaN, 0, 0.003, Infinity, 0.002, 0.001]) };
const detail = addCloudPresentationDetail(source, source, [2, 2, 2], 42);
assert.ok(Number.isNaN(detail.grid.data[2]) && Number.isNaN(detail.grid.data[5]));
assert.equal(detail.grid.data[3], 0);
const finiteSum = (values: Float32Array) => values.reduce((sum, value) => sum + (Number.isFinite(value) ? value : 0), 0);
assert.ok(Math.abs(finiteSum(source.data) - finiteSum(detail.grid.data)) < 1e-8);
const noFlow = addCloudPresentationDetail(source, source, [2, 2, 2], 42,
  { grid: { ...source, data: new Float32Array(24).fill(NaN) }, simulationTimeS: 20 });
assert.equal(noFlow.metadata?.coherence, "storm-relative");
assert.deepEqual(noFlow.grid.data, detail.grid.data);
const field = { shape: [3, 3, 3] as [number, number, number], data: new Float32Array(27).fill(0.001) };
field.data[13] = NaN;
const prepared = prepareVtkRequest({ kind: "cloud", field, grid: { drM: 500, dyM: 500, dzM: 500 }, volumeMaxVoxels: 10000,
  surfaceMaxVoxels: 10000, fidelityMode: "presentation", detailSeed: 42 });
assert.equal(prepared.presentationDetail, null);
const resolved = prepareVtkRequest({ kind: "cloud", field, grid: { drM: 500, dyM: 500, dzM: 500 }, volumeMaxVoxels: 10000,
  surfaceMaxVoxels: 10000, fidelityMode: "scientific" });
assert.deepEqual(prepared.surface.values, resolved.surface.values);
assert.deepEqual(prepared.surface.samplingProvenance, resolved.surface.samplingProvenance);
assert.ok(prepared.volume.validity && prepared.surface.validity);
assert.deepEqual(prepared.volume.validity, prepared.surface.validity);
assert.ok(prepared.surface.values.every(Number.isFinite));
console.log("Supercell support: gaps, invalid samples, ambient evidence, connectivity, and Presentation masks passed.");
