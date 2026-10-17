import type { VortexSurfaceGeometry } from "./vortexContourProtocol";
import {
  contourTopologyUpperBound,
  contourUsageViolation,
} from "./vtkContourBudget";

type Triple = [number, number, number];

const TETRAHEDRA = [
  [0, 1, 3, 7],
  [0, 3, 2, 7],
  [0, 2, 6, 7],
  [0, 6, 4, 7],
  [0, 4, 5, 7],
  [0, 5, 1, 7],
] as const;

const CORNERS = [
  [0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],
  [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1],
] as const;

function checkedCells(shape: readonly number[]): number {
  if (shape.length !== 3 || shape.some((extent) => !Number.isSafeInteger(extent) || extent < 2)) {
    throw new Error("Terrain contour requires a three-dimensional lattice with at least two samples per axis");
  }
  const cells = shape.reduce((product, extent) => product * extent, 1);
  if (!Number.isSafeInteger(cells)) throw new Error("Terrain contour lattice exceeds safe integer range");
  return cells;
}

function interpolate(
  a: number,
  b: number,
  threshold: number,
  values: Float32Array,
  points: readonly (Float32Array | Float64Array)[],
): Triple {
  const denominator = values[b] - values[a];
  const t = denominator === 0 ? 0.5 : Math.max(0, Math.min(1,
    (threshold - values[a]) / denominator));
  return [0, 1, 2].map((axis) =>
    points[axis][a] + (points[axis][b] - points[axis][a]) * t) as Triple;
}

export function contourTerrainBrick(
  shape: readonly [number, number, number],
  values: Float32Array,
  centroidsM: {
    x: Float32Array | Float64Array;
    y: Float32Array | Float64Array;
    z: Float32Array | Float64Array;
  },
  threshold: number,
  options: {
    maximumTriangles?: number;
    maximumBytes?: number;
    cellBounds?: readonly [number, number, number, number, number, number];
  } = {},
): VortexSurfaceGeometry | null {
  const samples = checkedCells(shape);
  if (values.length !== samples
      || centroidsM.x.length !== samples
      || centroidsM.y.length !== samples
      || centroidsM.z.length !== samples) {
    throw new Error("Terrain contour scalar and physical-coordinate lattices disagree");
  }
  if (!Number.isFinite(threshold)) throw new Error("Terrain contour threshold must be finite");
  const pointsByAxis = [centroidsM.x, centroidsM.y, centroidsM.z] as const;
  const positions: number[] = [];
  const normals: number[] = [];
  const polys: number[] = [];
  const bounds: [number, number, number, number, number, number] = [
    Infinity, -Infinity, Infinity, -Infinity, Infinity, -Infinity,
  ];
  const [nx, ny, nz] = shape;
  const cellBounds = options.cellBounds ?? [0, nx - 1, 0, ny - 1, 0, nz - 1];
  if (cellBounds.some((value) => !Number.isSafeInteger(value))
      || cellBounds[0] < 0 || cellBounds[1] > nx - 1 || cellBounds[1] < cellBounds[0]
      || cellBounds[2] < 0 || cellBounds[3] > ny - 1 || cellBounds[3] < cellBounds[2]
      || cellBounds[4] < 0 || cellBounds[5] > nz - 1 || cellBounds[5] < cellBounds[4]) {
    throw new Error("Terrain contour cell ownership bounds are invalid");
  }
  const ownedSamples = Math.max(1,
    (cellBounds[1] - cellBounds[0] + 1)
      * (cellBounds[3] - cellBounds[2] + 1)
      * (cellBounds[5] - cellBounds[4] + 1));
  const topologyBound = contourTopologyUpperBound(ownedSamples, "terrain-tetrahedra");
  const index = (i: number, j: number, k: number) => (i * ny + j) * nz + k;
  let triangles = 0;

  const appendTriangle = (triangle: readonly [Triple, Triple, Triple]) => {
    const ab = triangle[1].map((value, axis) => value - triangle[0][axis]) as Triple;
    const ac = triangle[2].map((value, axis) => value - triangle[0][axis]) as Triple;
    const normal: Triple = [
      ab[1] * ac[2] - ab[2] * ac[1],
      ab[2] * ac[0] - ab[0] * ac[2],
      ab[0] * ac[1] - ab[1] * ac[0],
    ];
    const length = Math.hypot(...normal);
    if (!(length > 0) || !Number.isFinite(length)) return;
    normal[0] /= length;
    normal[1] /= length;
    normal[2] /= length;
    const base = positions.length / 3;
    for (const point of triangle) {
      positions.push(...point);
      normals.push(...normal);
      for (let axis = 0; axis < 3; axis += 1) {
        bounds[axis * 2] = Math.min(bounds[axis * 2], point[axis]);
        bounds[axis * 2 + 1] = Math.max(bounds[axis * 2 + 1], point[axis]);
      }
    }
    polys.push(3, base, base + 1, base + 2);
    triangles += 1;
    const bytes = positions.length * Float32Array.BYTES_PER_ELEMENT
      + normals.length * Float32Array.BYTES_PER_ELEMENT
      + polys.length * Uint32Array.BYTES_PER_ELEMENT;
    const violation = contourUsageViolation(
      { triangles, bytes },
      "Terrain brick contour output",
      options.maximumBytes ?? topologyBound.bytes,
      options.maximumTriangles ?? topologyBound.triangles,
    );
    if (violation) throw new Error(violation);
  };

  for (let i = cellBounds[0]; i < cellBounds[1]; i += 1) {
    for (let j = cellBounds[2]; j < cellBounds[3]; j += 1) {
      for (let k = cellBounds[4]; k < cellBounds[5]; k += 1) {
        const cube = CORNERS.map(([di, dj, dk]) => index(i + di, j + dj, k + dk));
        if (cube.some((sample) => !Number.isFinite(values[sample])
            || !Number.isFinite(centroidsM.x[sample])
            || !Number.isFinite(centroidsM.y[sample])
            || !Number.isFinite(centroidsM.z[sample]))) continue;
        for (const tetrahedron of TETRAHEDRA) {
          const vertices = tetrahedron.map((corner) => cube[corner]);
          const inside = vertices.filter((sample) => values[sample] >= threshold);
          const outside = vertices.filter((sample) => values[sample] < threshold);
          if (inside.length === 0 || outside.length === 0) continue;
          if (inside.length === 1 || outside.length === 1) {
            const loneInside = inside.length === 1;
            const lone = (loneInside ? inside : outside)[0];
            const others = loneInside ? outside : inside;
            const triangle = others.map((other) =>
              interpolate(lone, other, threshold, values, pointsByAxis)) as [Triple, Triple, Triple];
            appendTriangle(loneInside ? triangle : [triangle[0], triangle[2], triangle[1]]);
            continue;
          }
          const [a, b] = inside;
          const [c, d] = outside;
          const ac = interpolate(a, c, threshold, values, pointsByAxis);
          const ad = interpolate(a, d, threshold, values, pointsByAxis);
          const bc = interpolate(b, c, threshold, values, pointsByAxis);
          const bd = interpolate(b, d, threshold, values, pointsByAxis);
          appendTriangle([ac, bc, bd]);
          appendTriangle([ac, bd, ad]);
        }
      }
    }
  }
  if (triangles === 0) return null;
  const typedPoints = Float32Array.from(positions);
  const typedNormals = Float32Array.from(normals);
  const typedPolys = Uint32Array.from(polys);
  return {
    points: typedPoints,
    normals: typedNormals,
    polys: typedPolys,
    bounds,
    triangles,
    bytes: typedPoints.byteLength + typedNormals.byteLength + typedPolys.byteLength,
  };
}
