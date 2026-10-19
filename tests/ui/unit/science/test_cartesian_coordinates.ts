import assert from "node:assert/strict";
import { cartesianScalarOrigin, cartesianDyMeters } from "../../../../ui/src/lib/science/cartesianInterp.ts";
import { windowAxisCoordinates } from "../../../../ui/src/lib/science/windowedGrid.ts";
import { gridHorizontalCenterM, minimumNominalGridSpacingM, planBoundedSliceWindows } from "../../../../ui/src/lib/plot/fieldLoadPlanning.ts";
import type { GridInfo } from "../../../../ui/src/lib/run/runOutputs.ts";

const legacy: GridInfo = { nr: 9, nth: 11, nz: 5, drM: 100, dthetaRad: 40, dzM: 80 };
assert.deepEqual(cartesianScalarOrigin(legacy, [9, 11, 5]), [-400, -200, 40]);
assert.deepEqual(windowAxisCoordinates(legacy, 1, 11, 2, 5, 2, 3, true), [-120, -40, 40]);
assert.deepEqual(windowAxisCoordinates(legacy, 1, 11, 2, 5, 2, 3, false), [-100, -20, 40]);
assert.equal(minimumNominalGridSpacingM(legacy), 40);
assert.equal(cartesianDyMeters({ ...legacy, dyM: 175 }, 100), 175);
assert.equal(cartesianDyMeters({ dthetaRad: Math.PI / 4 }, 100), 100);
assert.equal(cartesianDyMeters({ dyM: Infinity, dthetaRad: Infinity }, 100), 100);

const terrain: GridInfo = {
  nr: 12, nth: 10, nz: 6, drM: 100, dyM: 200, dzM: 50,
  terrainFollowing: true, horizontalScalarOriginM: [10_050, -19_900],
  physicalBoundsM: { lower: [10_200, -19_600, 0], upper: [11_000, -18_400, 300] },
  validIndexRange: { lower: [2, 2, 0], upper: [9, 7, 5] },
};
assert.deepEqual(cartesianScalarOrigin(terrain, [12, 10, 6]), [10_050, -19_900, 25]);
assert.deepEqual(gridHorizontalCenterM(terrain), [10_650, -18_900]);
assert.deepEqual(windowAxisCoordinates(terrain, 0, 12, 2, 4, 2, 2, true), [10_250, 10_450]);
const boundsOnly = { ...terrain, horizontalScalarOriginM: undefined };
assert.deepEqual(cartesianScalarOrigin(boundsOnly, [12, 10, 6]), [10_050, -19_900, 25]);
assert.deepEqual(gridHorizontalCenterM(boundsOnly), gridHorizontalCenterM(terrain));

const plan = planBoundedSliceWindows(legacy, 2, 2, 1000, {
  xMin: -0.1, xMax: 0.1, yMin: -0.08, yMax: 0.08,
});
const explicit = planBoundedSliceWindows({ ...legacy, dyM: 40 }, 2, 2, 1000, {
  xMin: -0.1, xMax: 0.1, yMin: -0.08, yMax: 0.08,
});
assert.deepEqual(plan, explicit);
assert.equal(plan.ok, true);
console.log("Cartesian coordinates: legacy spacing, terrain origins, halo bounds and viewport consistency passed");
