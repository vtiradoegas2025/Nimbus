import { clipSurfaceOutsideBounds, stitchNestedSurfaceTransition } from "../../../../ui/src/lib/vtk/nestedSurfaceOwnership.ts";
import type { VortexSurfaceGeometry } from "../../../../ui/src/lib/vtk/vortexContourProtocol.ts";
import { mergeOwnedBrickSurfaces } from "../../../../ui/src/lib/vtk/vtkBrickedContour.ts";

const geometry: VortexSurfaceGeometry = {
  points: new Float32Array([-1, 0.5, 0.5, 2, 0.5, 0.5, -1, 1.5, 0.5]),
  normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
  polys: new Uint32Array([3, 0, 1, 2]), bounds: [-1, 2, 0.5, 1.5, 0.5, 0.5], triangles: 1, bytes: 88,
};
const bounds = { lower: [0, 0, 0] as [number, number, number], upper: [1, 1, 1] as [number, number, number] };
const clipped = clipSurfaceOutsideBounds(geometry, bounds);
if (!clipped || clipped.triangles < 1) throw new Error("Crossing parent triangle was discarded");
for (let index = 0; index < clipped.points.length; index += 3) {
  const [x, y, z] = clipped.points.subarray(index, index + 3);
  if (x > 0 && x < 1 && y > 0 && y < 1 && z > 0 && z < 1) throw new Error("Parent vertex remains strictly child-owned");
}
if (clipSurfaceOutsideBounds({ ...geometry, points: new Float32Array([0.2, 0.2, 0.2, 0.8, 0.2, 0.2, 0.2, 0.8, 0.2]) }, bounds) !== null) {
  throw new Error("Fully child-owned parent triangle was retained");
}
if (!clipSurfaceOutsideBounds({ ...geometry, points: new Float32Array([-2, 0, 0, -1, 0, 0, -2, 1, 0]) }, bounds)) {
  throw new Error("Outside parent triangle was removed");
}

function openSquareTube(lower: number, upper: number, inset: number): VortexSurfaceGeometry {
  const ring = [[inset, inset], [1 - inset, inset], [1 - inset, 1 - inset], [inset, 1 - inset]];
  const points = Float32Array.from([lower, upper].flatMap((z) => ring.flatMap(([x, y]) => [x, y, z])));
  const polys: number[] = [];
  for (let index = 0; index < 4; index += 1) {
    const next = (index + 1) % 4;
    polys.push(3, index, next, 4 + next, 3, index, 4 + next, 4 + index);
  }
  return { points, polys: Uint32Array.from(polys), normals: new Float32Array(points.length).fill(1),
    bounds: [inset, 1 - inset, inset, 1 - inset, lower, upper], triangles: 8,
    bytes: points.byteLength + polys.length * 4 + points.byteLength };
}
const transition = stitchNestedSurfaceTransition(
  openSquareTube(-0.5, 0, 0.1), openSquareTube(0, 0.5, 0.2), bounds, 1e-6,
);
if (!transition || transition.triangles !== 8) throw new Error("Coarse/fine transition strip was not closed");
if (transition.points.some((value, index) => index % 3 === 2 && Math.abs(value) > 1e-6)) {
  throw new Error("Transition strip left its ownership plane");
}

const collapsed = stitchNestedSurfaceTransition(openSquareTube(-0.5, 0, 0.1), null, bounds, 1e-6);
if (!collapsed || collapsed.triangles !== 4) throw new Error("Unmatched transition topology did not collapse closed");

const parentTube = openSquareTube(-0.5, 0, 0.1);
const childTube = openSquareTube(0, 0.5, 0.2);
const stitched = mergeOwnedBrickSurfaces([
  { brickId: "parent", geometry: parentTube, ownedCellBounds: [-Infinity, Infinity, -Infinity, Infinity, -Infinity, Infinity], includeUpper: [true, true, true] },
  { brickId: "transition", geometry: transition, ownedCellBounds: [-Infinity, Infinity, -Infinity, Infinity, -Infinity, Infinity], includeUpper: [true, true, true] },
  { brickId: "child", geometry: childTube, ownedCellBounds: [-Infinity, Infinity, -Infinity, Infinity, -Infinity, Infinity], includeUpper: [true, true, true] },
], 1e-6);
if (!stitched) throw new Error("Nested transition did not compose");
const edgeUse = new Map<string, number>();
for (let cursor = 0; cursor < stitched.polys.length;) {
  const count = stitched.polys[cursor++];
  const ids = Array.from(stitched.polys.subarray(cursor, cursor + count)); cursor += count;
  for (let index = 0; index < ids.length; index += 1) {
    const a = ids[index]; const b = ids[(index + 1) % ids.length];
    const za = stitched.points[a * 3 + 2]; const zb = stitched.points[b * 3 + 2];
    if (Math.abs(za) > 1e-6 || Math.abs(zb) > 1e-6) continue;
    const key = a < b ? `${a}:${b}` : `${b}:${a}`;
    edgeUse.set(key, (edgeUse.get(key) ?? 0) + 1);
  }
}
if ([...edgeUse.values()].some((count) => count !== 2)) throw new Error("Nested transition left an open ownership-plane edge");

console.log("\nnested surface ownership tests: 7 passed, 0 failed.");
