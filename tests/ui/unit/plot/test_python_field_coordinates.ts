import assert from "node:assert/strict";
import type { GridInfo } from "../../../../ui/src/lib/run/runOutputs.ts";
import { pythonFieldCoordinates } from "../../../../ui/src/lib/plot/pythonFieldCoordinates.ts";

const shape = [4, 3, 5], counts = [2, 2, 2], starts = [1, 0, 2], strides = [2, 2, 2];
const indices = [[1, 3], [0, 2], [2, 4]];
const grid: GridInfo = {
  nr: 4, nth: 3, nz: 5, coordinateSystem: "cartesian", geometryKind: "physical",
  drM: 100, dyM: 200, dzM: 300, scalarOriginM: [1000, 2000, 3000],
  horizontalScalarOriginM: [-9000, -8000],
};
const coordinates = (g: GridInfo | null | undefined) => pythonFieldCoordinates(g, shape, counts, starts, strides);
for (const unknown of [null, undefined, { nr: 4, nth: 3, nz: 5 }, { ...grid, geometryKind: "index" as const },
  { ...grid, coordinateSystem: "unknown" }]) {
  const result = coordinates(unknown);
  assert.deepEqual(result.map(axis => axis.values), indices);
  assert.deepEqual(result.map(axis => axis.unit), ["index", "index", "index"]);
}
assert.deepEqual(coordinates(grid).map(axis => axis.values), [[1100, 1300], [2000, 2400], [3600, 4200]],
  "the complete scalar origin owns coordinates over a stale legacy horizontal origin");
const exact: GridInfo = { ...grid, scalarCoordinatesM: [[10, 20, 40, 80], [-5, 3, 19], [100, 150, 300, 700, 1500]] };
assert.deepEqual(coordinates(exact).map(axis => axis.values), [[20, 80], [-5, 19], [300, 1500]],
  "cropping and stride preserve exact nonuniform saved coordinates");
const terrain = coordinates({ ...exact, terrainFollowing: true });
assert.deepEqual(terrain[2], { name: "z", values: [2, 4], unit: "model_level" });
const remapped = coordinates({ ...grid, terrainFollowing: false, scalarOriginM: [20000, 30000, 4000] });
assert.deepEqual(remapped.map(axis => axis.values), [[20100, 20300], [30000, 30400], [4600, 5200]],
  "a remapped window's physical origin overrides source-grid legacy horizontal origins");
assert.deepEqual(remapped.map(axis => axis.unit), ["m", "m", "m"]);
const bounds = { ...grid, scalarOriginM: undefined, horizontalScalarOriginM: undefined,
  physicalBoundsM: { lower: [5000, 6000, 7000] as [number, number, number], upper: [5400, 6600, 8500] as [number, number, number] },
  validIndexRange: { lower: [1, 1, 1] as [number, number, number], upper: [2, 1, 3] as [number, number, number] } };
assert.deepEqual(coordinates(bounds).map(axis => axis.values), [[5050, 5250], [5900, 6300], [7450, 8050]],
  "physical bounds locate scalar centers relative to valid cells, not storage halos");
const cylindrical = coordinates({ ...grid, coordinateSystem: "cylindrical", dthetaRad: 0.25 });
assert.deepEqual(cylindrical.map(axis => axis.unit), ["m", "rad", "m"]);
assert.deepEqual(cylindrical.map(axis => axis.name), ["r", "theta", "z"]);
assert.equal(grid.horizontalScalarOriginM![0], -9000);
console.log("Python field coordinates: index honesty, origin precedence, exact axes, crop/stride, terrain levels and remapped windows passed.");
