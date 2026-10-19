import assert from "node:assert/strict";
import { prepareVtkRequest, applyPreparedGlobalHorizontalLevelAnomaly, vtkPreparationTransferables } from "../../../../ui/src/lib/vtk/vtkPreparation.ts";
import { sharePreparedVtkResult, downgradeSharedPreparedResult, transferablePreparedBuffers } from "../../../../ui/src/lib/vtk/vtkDatasetHandoff.ts";
import { fieldToCartesianGrid, refineCartesianGridLinear } from "../../../../ui/src/lib/science/cartesianInterp.ts";
import { vtkPreparationOutputUpperBoundBytes, vtkPreparationResultBytes } from "../../../../ui/src/lib/vtk/vtkPreparationResourcePlanning.ts";
import { volumeImageBytes } from "../../../../ui/src/lib/vtk/volumeValidityBudget.ts";

const shape: [number, number, number] = [3, 2, 2];
const grid = { drM: 500, dyM: 500, dzM: 500, scalarOriginM: [0, 0, 0] as [number, number, number] };
const data = Float32Array.from([NaN, NaN, 280, 290, 282, 292, 284, 294, 286, 296, 288, 298]);
const request = { kind: "scalar" as const, field: { shape, data }, grid, scalarTransform: "horizontal-level-anomaly" as const };
const prepared = prepareVtkRequest(request);
assert.equal(prepared.validity?.filter(v => v === 0).length, 2);
for (let z = 0; z < 2; z++) {
  const values = Array.from({ length: 6 }, (_, i) => i + z * 6).filter(i => prepared.validity![i] !== 0).map(i => prepared.values[i]);
  assert.equal(values.reduce((a,b) => a+b,0), 0);
  assert.deepEqual([...values].sort((a,b) => a-b), [-4,-2,0,2,4]);
}
assert.equal(prepared.stats?.min, -4);
assert.equal(prepared.stats?.max, 4);
assert.ok(vtkPreparationResultBytes(prepared) <= vtkPreparationOutputUpperBoundBytes(request));
assert.equal(volumeImageBytes(prepared), 60);
const second = prepareVtkRequest({ ...request, scalarTransform: "none" });
const batch = applyPreparedGlobalHorizontalLevelAnomaly([second]);
assert.deepEqual(batch[0].validity, prepared.validity);
assert.deepEqual(batch[0].values, prepared.values);

const scalarGrid = fieldToCartesianGrid({ shape, data }, grid, { preserveMissing: true });
const refined = refineCartesianGridLinear(scalarGrid, { targetSpacingM: 250, maxVoxels: 1000 }).grid;
assert.equal(refined.data[(2 * 3 + 2) * 3], 284);
assert.ok(Number.isNaN(refined.data[0]));
assert.ok(Number.isNaN(refined.data[3]));
const positive = prepareVtkRequest({ kind:"scalar", field:{shape,data:Float32Array.from(data, v => Number.isFinite(v) ? 300 : NaN)},grid });
assert.equal(positive.stats?.min, 300);
assert.equal(positive.stats?.max, 300);
const zero = prepareVtkRequest({kind:"scalar",field:{shape,data:new Float32Array(12)},grid});
assert.equal(zero.validity, undefined);
assert.equal(volumeImageBytes(zero), 48);

sharePreparedVtkResult(prepared, true);
assert.ok(prepared.validity!.buffer instanceof SharedArrayBuffer);
assert.equal(transferablePreparedBuffers(prepared).length, 0);
downgradeSharedPreparedResult(prepared);
assert.ok(prepared.validity!.buffer instanceof ArrayBuffer);
assert.equal(transferablePreparedBuffers(prepared).length, 2);
const clone = structuredClone(prepared, { transfer: vtkPreparationTransferables(prepared) });
assert.equal(prepared.validity!.byteLength, 0);
assert.equal(clone.validity!.filter(v => v === 0).length, 2);
assert.ok(Number.isNaN(data[0]));
assert.equal(data[4],282);
console.log("Missing support, anomaly statistics, exact endpoints, finite zero, byte bounds, shared fallback, and worker transfer passed");
