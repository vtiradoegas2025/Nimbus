import { executeBrickedContourPlan, mergeOwnedBrickSurfaces, type OwnedBrickSurface } from "../../../../ui/src/lib/vtk/vtkBrickedContour.ts";
import type { ContourBrickPlan } from "../../../../ui/src/lib/plot/fieldBrickHierarchy.ts";
import type { VortexSurfaceGeometry } from "../../../../ui/src/lib/vtk/vortexContourProtocol.ts";
import { contourTerrainBrick } from "../../../../ui/src/lib/vtk/vtkTerrainContour.ts";

let passed = 0;
const failures: string[] = [];
function check(condition: boolean, label: string): void {
  if (condition) passed += 1;
  else failures.push(label);
}
function throws(fn: () => unknown, text: string, label: string): void {
  try { fn(); failures.push(label); }
  catch (error) { check(error instanceof Error && error.message.includes(text), label); }
}

function geometry(points: number[], triangles: number[][], normals?: number[]): VortexSurfaceGeometry {
  const typedPoints = Float32Array.from(points);
  const typedNormals = Float32Array.from(normals ?? points.map((_, index) => index % 3 === 2 ? 1 : 0));
  const polys = Uint32Array.from(triangles.flatMap((triangle) => [3, ...triangle]));
  return {
    points: typedPoints,
    normals: typedNormals,
    polys,
    bounds: null,
    triangles: triangles.length,
    bytes: typedPoints.byteLength + typedNormals.byteLength + polys.byteLength,
  };
}

const sharedTriangle = geometry(
  [1, 0, 0, 1, 1, 0, 1, 0, 1],
  [[0, 1, 2]],
);
const chunks: OwnedBrickSurface[] = [
  {
    brickId: "l0:0:0:0",
    geometry: sharedTriangle,
    ownedCellBounds: [0, 1, 0, 2, 0, 2],
    includeUpper: [false, true, true],
  },
  {
    brickId: "l0:1:0:0",
    geometry: sharedTriangle,
    ownedCellBounds: [1, 2, 0, 2, 0, 2],
    includeUpper: [true, true, true],
  },
];
const owned = mergeOwnedBrickSurfaces(chunks);
check(owned?.triangles === 1, "shared-plane triangle has one deterministic cell owner");
check(owned?.points.length === 9, "seam points are retained once");
check(owned?.polys.join(",") === "3,0,1,2", "merged triangle connectivity is canonical");
check(owned?.bounds?.join(",") === "1,1,0,1,0,1", "merged bounds are exact");
check(owned?.normals[2] === 1, "merged normals remain normalized");

const left = geometry(
  [0, 0, 0, 1, 0, 0, 1, 1, 0],
  [[0, 1, 2]],
  [0, 0, 1, 0, 1, 1, 0, 0, 1],
);
const right = geometry(
  [1 + 1e-7, 0, 0, 2, 0, 0, 1 + 1e-7, 1, 0],
  [[0, 1, 2]],
  [0, 1, 1, 0, 0, 1, 0, 1, 1],
);
const joined = mergeOwnedBrickSurfaces([
  { brickId: "a", geometry: left, ownedCellBounds: [0, 1, 0, 1, -1, 1], includeUpper: [false, true, true] },
  { brickId: "b", geometry: right, ownedCellBounds: [1, 2, 0, 1, -1, 1], includeUpper: [true, true, true] },
]);
check(joined?.triangles === 2, "neighboring owned cells retain both surfaces");
check(joined?.points.length === 12, "coordinate tolerance welds two seam vertices");
check(joined?.normals.every(Number.isFinite) === true, "welded normals are finite");
check(Math.abs(Math.hypot(joined!.normals[3], joined!.normals[4], joined!.normals[5]) - 1) < 1e-6, "welded normals are normalized");
check(joined?.bytes === joined!.points.byteLength + joined!.polys.byteLength + joined!.normals.byteLength, "merged bytes are measured");
check(mergeOwnedBrickSurfaces([]) === null, "empty brick result stays empty");
throws(() => mergeOwnedBrickSurfaces(chunks, 0), "positive", "invalid weld tolerance fails");
throws(() => mergeOwnedBrickSurfaces([{
  ...chunks[0],
  geometry: { ...sharedTriangle, normals: new Float32Array(0) },
}]), "inconsistent", "invalid geometry fails before publication");

const plan: ContourBrickPlan = {
  field: "w",
  publicationGeneration: "g",
  level: 0,
  threshold: 1,
  items: ["a", "b"].map((brickId, index) => ({
    brickId,
    brickIndex: [index, 0, 0],
    readWindow: {
      aggregation: "cell_mean",
      origin: [index, 0, 0],
      extent: [2, 2, 2],
      stride: [1, 1, 1],
      maxOutputBytes: 32,
      maxSourceBytes: 32,
    },
    ownedCellOrigin: [index, 0, 0],
    ownedCellExtent: [1, 1, 1],
    scalarRange: { minimum: 0, maximum: 2, finiteCount: 8, nonFiniteCount: 0 },
  })),
  culledBrickCount: 0,
  sourceBytes: 64,
};
const order: string[] = [];
const progress: number[] = [];
const executed = await executeBrickedContourPlan(plan, {
  signal: new AbortController().signal,
  extract: async (item) => {
    order.push(item.brickId);
    return item.brickId === "a" ? left : right;
  },
  ownership: (item) => item.brickId === "a"
    ? { ownedCellBounds: [0, 1, 0, 1, -1, 1], includeUpper: [false, true, true] }
    : { ownedCellBounds: [1, 2, 0, 1, -1, 1], includeUpper: [true, true, true] },
  onProgress: (completed) => progress.push(completed),
});
check(order.join(",") === "a,b", "brick coordinator extracts in deterministic plan order");
check(progress.join(",") === "1,2", "brick coordinator reports monotone completion");
check(executed?.triangles === 2, "brick coordinator publishes one merged surface");

const cancelled = new AbortController();
let cancellationObserved = false;
try {
  await executeBrickedContourPlan(plan, {
    signal: cancelled.signal,
    extract: async (_item) => {
      cancelled.abort();
      return left;
    },
    ownership: () => ({ ownedCellBounds: [0, 1, 0, 1, -1, 1], includeUpper: [true, true, true] }),
  });
} catch (error) {
  cancellationObserved = error instanceof DOMException && error.name === "AbortError";
}
check(cancellationObserved, "brick coordinator cancellation is failure-atomic");
let budgetObserved = false;
try {
  await executeBrickedContourPlan(plan, {
    signal: new AbortController().signal,
    extract: async () => left,
    ownership: () => ({ ownedCellBounds: [0, 2, 0, 1, -1, 1], includeUpper: [true, true, true] }),
    maximumTriangles: 1,
  });
} catch (error) {
  budgetObserved = error instanceof Error && error.message.includes("Progressive contour output");
}
check(budgetObserved, "combined brick output stops before over-budget publication");

const terrainShape: [number, number, number] = [3, 3, 3];
const terrainValues = new Float32Array(27);
const terrainX = new Float64Array(27);
const terrainY = new Float64Array(27);
const terrainZ = new Float64Array(27);
for (let i = 0; i < 3; i += 1) {
  for (let j = 0; j < 3; j += 1) {
    for (let k = 0; k < 3; k += 1) {
      const sample = (i * 3 + j) * 3 + k;
      terrainValues[sample] = i;
      terrainX[sample] = i;
      terrainY[sample] = j;
      terrainZ[sample] = k + 0.2 * i + 0.1 * j;
    }
  }
}
const terrainSurface = contourTerrainBrick(
  terrainShape,
  terrainValues,
  { x: terrainX, y: terrainY, z: terrainZ },
  0.5,
);
check((terrainSurface?.triangles ?? 0) > 0, "terrain marching tetrahedra emits a physical surface");
check(terrainSurface?.points.every((value, offset) => offset % 3 !== 0 || Math.abs(value - 0.5) < 1e-6) === true,
  "terrain contour interpolates the scalar crossing in physical x");
check((terrainSurface?.bounds?.[5] ?? 0) - (terrainSurface?.bounds?.[4] ?? 0) > 1,
  "terrain contour preserves nonplanar physical height support");
check(contourTerrainBrick(
  terrainShape,
  new Float32Array(27),
  { x: terrainX, y: terrainY, z: terrainZ },
  1,
) === null, "uniform terrain scalar has no false contour");
throws(() => contourTerrainBrick(
  terrainShape,
  terrainValues,
  { x: terrainX.subarray(1), y: terrainY, z: terrainZ },
  0.5,
), "disagree", "terrain contour rejects mismatched physical geometry");

const terrainWindow = (originX: number) => {
  const shape: [number, number, number] = [4, 3, 3];
  const values = new Float32Array(36);
  const x = new Float64Array(36);
  const y = new Float64Array(36);
  const z = new Float64Array(36);
  for (let i = 0; i < 4; i += 1) {
    for (let j = 0; j < 3; j += 1) {
      for (let k = 0; k < 3; k += 1) {
        const sample = (i * 3 + j) * 3 + k;
        values[sample] = j;
        x[sample] = originX + i;
        y[sample] = j;
        z[sample] = k + 0.2 * (originX + i);
      }
    }
  }
  return { shape, values, centroidsM: { x, y, z } };
};
const terrainLeft = terrainWindow(0);
const terrainRight = terrainWindow(1);
const terrainLeftSurface = contourTerrainBrick(
  terrainLeft.shape,
  terrainLeft.values,
  terrainLeft.centroidsM,
  0.5,
  { cellBounds: [0, 2, 0, 2, 0, 2] },
)!;
const terrainRightSurface = contourTerrainBrick(
  terrainRight.shape,
  terrainRight.values,
  terrainRight.centroidsM,
  0.5,
  { cellBounds: [1, 3, 0, 2, 0, 2] },
)!;
const terrainJoined = mergeOwnedBrickSurfaces([
  {
    brickId: "terrain-left",
    geometry: terrainLeftSurface,
    ownedCellBounds: [-Infinity, Infinity, -Infinity, Infinity, -Infinity, Infinity],
    includeUpper: [true, true, true],
  },
  {
    brickId: "terrain-right",
    geometry: terrainRightSurface,
    ownedCellBounds: [-Infinity, Infinity, -Infinity, Infinity, -Infinity, Infinity],
    includeUpper: [true, true, true],
  },
]);
check(terrainJoined?.triangles === terrainLeftSurface.triangles + terrainRightSurface.triangles,
  "terrain logical cell ownership emits no duplicate seam triangles");
check((terrainJoined?.points.length ?? Infinity)
    < terrainLeftSurface.points.length + terrainRightSurface.points.length,
  "terrain seam vertices weld across overlapping physical halo coordinates");

console.log(`\nvtk bricked contour tests: ${passed} passed, ${failures.length} failed.`);
if (failures.length > 0) {
  for (const failure of failures) console.error(`FAIL: ${failure}`);
  process.exitCode = 1;
}
