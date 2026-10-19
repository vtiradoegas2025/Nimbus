import assert from "node:assert/strict";
import { slicePlaneDescription } from "../../../../ui/src/lib/plot/slicePlaneDescription.ts";
import type { GridInfo } from "../../../../ui/src/lib/run/runOutputs.ts";

const grid: GridInfo = { coordinateSystem: "cartesian", nr: 3, nth: 4, nz: 2,
  scalarCoordinatesM: [[-1000, 0, 1000], [-4000, -1000, 1000, 4000], [250, 1000]],
  validIndexRange: { lower: [1, 1, 0], upper: [2, 2, 1] } };
assert.deepEqual(slicePlaneDescription(grid, 1, 0), { plane: "x–z section", location: "y = -1 km · index 0" });
assert.deepEqual(slicePlaneDescription(grid, 0, 1), { plane: "y–z section", location: "x = 1 km · index 1" });
assert.equal(slicePlaneDescription(grid, 2, 1).location, "z = 1 km · index 1");
assert.equal(slicePlaneDescription({ ...grid, terrainFollowing: true }, 2, 1).location, "k index 1");
assert.equal(slicePlaneDescription({ ...grid, geometryKind: "index" }, 1, 1).location, "y index 1");
assert.equal(slicePlaneDescription(null, 1, 4).location, "y index 4");
console.log("Section positions preserve stretched physical centers, valid-domain offsets, and model/index labels.");
