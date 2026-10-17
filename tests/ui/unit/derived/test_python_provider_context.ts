import assert from "node:assert/strict";
import { pythonProviderContext } from "../../../../ui/src/lib/derived/pythonProviderContext.ts";
import type { ExternalProviderExecutionRequest } from "../../../../ui/src/lib/derived/externalDerivedProviders.ts";
import type { NpyN } from "../../../../ui/src/lib/platform/npy.ts";

const field: NpyN = { shape: [2, 2, 2], data: Float32Array.from({ length: 8 }, (_, index) => index), windowGeometry: {
  sourceShape: [4, 4, 4], origin: [1, 1, 2], extent: [2, 2, 2], stride: [1, 1, 1],
  aggregation: "point_sample", weighting: "uniform_cell_volume", sampleLocation: "source_cell_center", sourceFormat: "npy", centroidsM: null,
} };
const request = {
  datasetIdentity: "source-fixture", frame: { index: 42, simulationTimeS: 300, publicationGeneration: "published" },
  grid: { nr: 4, nth: 4, nz: 4, coordinateSystem: "cartesian", drM: 10, dyM: 10, dzM: 100,
    scalarCoordinatesM: [[10, 20, 30, 40], [5, 15, 25, 35], [100, 200, 300, 400]],
    validIndexRange: { lower: [1, 1, 1], upper: [2, 2, 3] } },
  provider: { inputs: [{ alias: "vertical_velocity", fieldId: "w", unit: "m/s" }] },
  inputs: { vertical_velocity: { field, unit: "m/s", identity: "source-frame-w" } },
} as unknown as ExternalProviderExecutionRequest;
const context = pythonProviderContext(request);
assert.deepEqual([context.grid!.nr, context.grid!.nth, context.grid!.nz], [2, 2, 2]);
assert.deepEqual(context.grid!.scalarCoordinatesM, [[20, 30], [15, 25], [300, 400]]);
assert.deepEqual(context.grid!.validIndexRange, { lower: [0, 0, 0], upper: [1, 1, 1] });
assert.deepEqual(context.fieldUnits, { vertical_velocity: "m/s" });
assert.equal(context.stepIndex, 42);
assert.equal(context.frameIndex, null, "an unknown ordinal is not falsely reported as frame zero or simulation step 42");
assert.equal(context.simulationTimeS, 300);
assert.equal(context.fields[0].field.data, field.data, "projection retains the immutable source buffer");
assert.deepEqual(request.grid!.scalarCoordinatesM, [[10, 20, 30, 40], [5, 15, 25, 35], [100, 200, 300, 400]]);
const terrain = pythonProviderContext({ ...request, grid: { ...request.grid!, terrainFollowing: true } });
assert.equal(terrain.grid!.terrainFollowing, true);
assert.deepEqual(terrain.modelLevelIndices, [2, 3], "cropped terrain tiles retain their source model-level indices");
console.log("Python provider window coordinates, units, terrain indices, time and source ownership: passed");
