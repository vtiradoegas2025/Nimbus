import type { VortexSurfaceGeometry } from "./vortexContourProtocol";
import { mergeOwnedBrickSurfaces, type OwnedBrickSurface } from "./vtkBrickedContour";

type Vertex = { point: [number, number, number]; normal: [number, number, number] };
type Bounds = { lower: readonly [number, number, number]; upper: readonly [number, number, number] };
export type NestedSurfaceBounds = Bounds;
type BoundarySample = Vertex & { key: string };

function splitPolygon(vertices: Vertex[], axis: number, boundary: number, keepGreater: boolean) {
  const inside: Vertex[] = []; const outside: Vertex[] = [];
  const side = (vertex: Vertex) => keepGreater
    ? vertex.point[axis] >= boundary : vertex.point[axis] <= boundary;
  const interpolate = (a: Vertex, b: Vertex) => {
    const span = b.point[axis] - a.point[axis];
    const t = span === 0 ? 0 : (boundary - a.point[axis]) / span;
    const point = a.point.map((value, i) => value + t * (b.point[i] - value)) as [number, number, number];
    const normal = a.normal.map((value, i) => value + t * (b.normal[i] - value)) as [number, number, number];
    const length = Math.hypot(...normal) || 1;
    return { point, normal: normal.map((value) => value / length) as [number, number, number] };
  };
  for (let index = 0; index < vertices.length; index += 1) {
    const a = vertices[index]; const b = vertices[(index + 1) % vertices.length];
    const aInside = side(a); const bInside = side(b);
    (aInside ? inside : outside).push(a);
    if (aInside !== bInside) {
      const crossing = interpolate(a, b);
      inside.push(crossing); outside.push(crossing);
    }
  }
  return { inside, outside };
}

export function clipSurfaceOutsideBounds(
  geometry: VortexSurfaceGeometry | null,
  bounds: Bounds,
): VortexSurfaceGeometry | null {
  if (!geometry) return null;
  if (![...bounds.lower, ...bounds.upper].every(Number.isFinite)
      || bounds.lower.some((value, axis) => value >= bounds.upper[axis])) {
    throw new Error("Nested surface exclusion bounds are invalid");
  }
  const output: Vertex[][] = [];
  for (let cursor = 0; cursor < geometry.polys.length;) {
    const count = geometry.polys[cursor++];
    const polygon: Vertex[] = [];
    for (let index = 0; index < count; index += 1) {
      const pointId = geometry.polys[cursor++];
      polygon.push({
        point: [geometry.points[pointId * 3], geometry.points[pointId * 3 + 1], geometry.points[pointId * 3 + 2]],
        normal: geometry.normals.length === geometry.points.length
          ? [geometry.normals[pointId * 3], geometry.normals[pointId * 3 + 1], geometry.normals[pointId * 3 + 2]]
          : [0, 0, 1],
      });
    }
    let inside = [polygon];
    for (let axis = 0; axis < 3; axis += 1) for (const [boundary, greater] of [[bounds.lower[axis], true], [bounds.upper[axis], false]] as const) {
      const next: Vertex[][] = [];
      for (const fragment of inside) {
        const split = splitPolygon(fragment, axis, boundary, greater);
        if (split.outside.length >= 3) output.push(split.outside);
        if (split.inside.length >= 3) next.push(split.inside);
      }
      inside = next;
    }
  }
  const points: number[] = []; const normals: number[] = []; const polys: number[] = [];
  for (const polygon of output) for (let index = 1; index + 1 < polygon.length; index += 1) {
    polys.push(3);
    for (const vertex of [polygon[0], polygon[index], polygon[index + 1]]) {
      polys.push(points.length / 3); points.push(...vertex.point); normals.push(...vertex.normal);
    }
  }
  if (polys.length === 0) return null;
  const pointArray = Float32Array.from(points);
  const values = [0, 1, 2].map((axis) => points.filter((_, index) => index % 3 === axis));
  const resultBounds = [Math.min(...values[0]), Math.max(...values[0]), Math.min(...values[1]), Math.max(...values[1]), Math.min(...values[2]), Math.max(...values[2])] as [number, number, number, number, number, number];
  const polyArray = Uint32Array.from(polys);
  const normalArray = Float32Array.from(normals);
  return { points: pointArray, polys: polyArray, normals: normalArray, bounds: resultBounds,
    triangles: polyArray.length / 4, bytes: pointArray.byteLength + polyArray.byteLength + normalArray.byteLength };
}

function coordinateKey(point: readonly number[], tolerance: number): string {
  return point.map((value) => Math.round(value / tolerance)).join(":");
}

function boundaryCurves(
  geometry: VortexSurfaceGeometry | null,
  axis: number,
  boundary: number,
  tolerance: number,
): BoundarySample[][] {
  if (!geometry) return [];
  const edges = new Map<string, { a: BoundarySample; b: BoundarySample; count: number }>();
  const sample = (pointId: number): BoundarySample => {
    const point = [geometry.points[pointId * 3], geometry.points[pointId * 3 + 1], geometry.points[pointId * 3 + 2]] as [number, number, number];
    const normal = geometry.normals.length === geometry.points.length
      ? [geometry.normals[pointId * 3], geometry.normals[pointId * 3 + 1], geometry.normals[pointId * 3 + 2]] as [number, number, number]
      : [0, 0, 1] as [number, number, number];
    return { point, normal, key: coordinateKey(point, tolerance) };
  };
  for (let cursor = 0; cursor < geometry.polys.length;) {
    const count = geometry.polys[cursor++];
    const ids = Array.from(geometry.polys.subarray(cursor, cursor + count));
    cursor += count;
    for (let index = 0; index < ids.length; index += 1) {
      const a = sample(ids[index]);
      const b = sample(ids[(index + 1) % ids.length]);
      if (Math.abs(a.point[axis] - boundary) > tolerance
          || Math.abs(b.point[axis] - boundary) > tolerance || a.key === b.key) continue;
      const key = a.key < b.key ? `${a.key}|${b.key}` : `${b.key}|${a.key}`;
      const existing = edges.get(key);
      if (existing) existing.count += 1;
      else edges.set(key, { a, b, count: 1 });
    }
  }
  const boundaryEdges = [...edges.values()].filter((edge) => edge.count === 1);
  const adjacency = new Map<string, BoundarySample[]>();
  const samples = new Map<string, BoundarySample>();
  for (const edge of boundaryEdges) {
    samples.set(edge.a.key, edge.a); samples.set(edge.b.key, edge.b);
    adjacency.set(edge.a.key, [...adjacency.get(edge.a.key) ?? [], edge.b]);
    adjacency.set(edge.b.key, [...adjacency.get(edge.b.key) ?? [], edge.a]);
  }
  if ([...adjacency.values()].some((neighbors) => neighbors.length > 2)) {
    throw new Error("Nested contour boundary is branched and cannot be stitched unambiguously");
  }
  const unused = new Set(boundaryEdges.map((edge) => edge.a.key < edge.b.key
    ? `${edge.a.key}|${edge.b.key}` : `${edge.b.key}|${edge.a.key}`));
  const curves: BoundarySample[][] = [];
  while (unused.size > 0) {
    const firstEdgeKey = unused.values().next().value as string;
    const [firstA, firstB] = firstEdgeKey.split("|");
    const endpoint = [firstA, firstB].find((key) => (adjacency.get(key)?.length ?? 0) === 1);
    const start = endpoint ?? firstA;
    const curve: BoundarySample[] = [samples.get(start)!];
    let previous: string | null = null;
    let current = start;
    while (true) {
      const next = (adjacency.get(current) ?? []).find((candidate) => {
        if (candidate.key === previous) return false;
        const key = current < candidate.key ? `${current}|${candidate.key}` : `${candidate.key}|${current}`;
        return unused.has(key);
      });
      if (!next) break;
      const edgeKey = current < next.key ? `${current}|${next.key}` : `${next.key}|${current}`;
      unused.delete(edgeKey);
      previous = current;
      current = next.key;
      curve.push(samples.get(current)!);
      if (current === start) break;
    }
    if (curve.length >= 2) curves.push(curve);
  }
  return curves;
}

function resampleCurve(curve: readonly BoundarySample[], count: number): Vertex[] {
  const cumulative = [0];
  for (let index = 1; index < curve.length; index += 1) {
    cumulative.push(cumulative[index - 1] + Math.hypot(
      ...curve[index].point.map((value, axis) => value - curve[index - 1].point[axis]),
    ));
  }
  const total = cumulative.at(-1) ?? 0;
  if (!(total > 0)) throw new Error("Nested contour boundary curve has zero length");
  return Array.from({ length: count }, (_, sampleIndex) => {
    const distance = total * sampleIndex / (count - 1);
    let segment = 1;
    while (segment < cumulative.length - 1 && cumulative[segment] < distance) segment += 1;
    const lower = cumulative[segment - 1];
    const span = cumulative[segment] - lower;
    const t = span > 0 ? (distance - lower) / span : 0;
    const left = curve[segment - 1]; const right = curve[segment];
    const point = left.point.map((value, axis) => value + t * (right.point[axis] - value)) as [number, number, number];
    const normal = left.normal.map((value, axis) => value + t * (right.normal[axis] - value)) as [number, number, number];
    const norm = Math.hypot(...normal) || 1;
    return { point, normal: normal.map((value) => value / norm) as [number, number, number] };
  });
}

function curveCentroid(curve: readonly BoundarySample[]): [number, number, number] {
  return [0, 1, 2].map((axis) => curve.reduce((sum, sample) => sum + sample.point[axis], 0) / curve.length) as [number, number, number];
}

export function stitchNestedSurfaceTransition(
  parent: VortexSurfaceGeometry | null,
  child: VortexSurfaceGeometry | null,
  bounds: Bounds,
  tolerance: number,
): VortexSurfaceGeometry | null {
  if (!Number.isFinite(tolerance) || tolerance <= 0) throw new Error("Nested contour stitch tolerance must be positive");
  const strips: Array<{ parent: Vertex[]; child: Vertex[]; closed: boolean }> = [];
  for (let axis = 0; axis < 3; axis += 1) for (const boundary of [bounds.lower[axis], bounds.upper[axis]]) {
    const parentCurves = boundaryCurves(parent, axis, boundary, tolerance);
    const childCurves = boundaryCurves(child, axis, boundary, tolerance);
    const remaining = new Set(childCurves.map((_, index) => index));
    const collapse = (curve: readonly BoundarySample[], count: number): Vertex[] => {
      const centroid = curveCentroid(curve);
      const normal = [0, 1, 2].map((coordinate) => curve.reduce(
        (sum, sample) => sum + sample.normal[coordinate], 0,
      ) / curve.length) as [number, number, number];
      const norm = Math.hypot(...normal) || 1;
      return Array.from({ length: count }, () => ({ point: [...centroid] as [number, number, number],
        normal: normal.map((value) => value / norm) as [number, number, number] }));
    };
    for (const parentCurve of parentCurves) {
      const parentCenter = curveCentroid(parentCurve);
      const childIndex = [...remaining].sort((left, right) => {
        const distance = (index: number) => Math.hypot(...curveCentroid(childCurves[index])
          .map((value, coordinate) => value - parentCenter[coordinate]));
        return distance(left) - distance(right);
      })[0] as number | undefined;
      if (childIndex == null) {
        const count = Math.max(2, parentCurve.length);
        strips.push({ parent: resampleCurve(parentCurve, count), child: collapse(parentCurve, count),
          closed: parentCurve[0].key === parentCurve.at(-1)?.key });
        continue;
      }
      remaining.delete(childIndex);
      let childCurve = childCurves[childIndex];
      const parentClosed = parentCurve[0].key === parentCurve.at(-1)?.key;
      const childClosed = childCurve[0].key === childCurve.at(-1)?.key;
      if (parentClosed !== childClosed) {
        const parentCount = Math.max(2, parentCurve.length);
        const childCount = Math.max(2, childCurve.length);
        strips.push({ parent: resampleCurve(parentCurve, parentCount), child: collapse(parentCurve, parentCount), closed: parentClosed });
        strips.push({ parent: collapse(childCurve, childCount), child: resampleCurve(childCurve, childCount), closed: childClosed });
        continue;
      }
      if (!parentClosed) {
        const direct = Math.hypot(...parentCurve[0].point.map((value, coordinate) => value - childCurve[0].point[coordinate]));
        const reversed = Math.hypot(...parentCurve[0].point.map((value, coordinate) => value - childCurve.at(-1)!.point[coordinate]));
        if (reversed < direct) childCurve = [...childCurve].reverse();
      }
      const count = Math.max(2, parentCurve.length, childCurve.length);
      strips.push({ parent: resampleCurve(parentCurve, count), child: resampleCurve(childCurve, count), closed: parentClosed });
    }
    for (const childIndex of remaining) {
      const childCurve = childCurves[childIndex];
      const count = Math.max(2, childCurve.length);
      strips.push({ parent: collapse(childCurve, count), child: resampleCurve(childCurve, count),
        closed: childCurve[0].key === childCurve.at(-1)?.key });
    }
  }
  const points: number[] = []; const normals: number[] = []; const polys: number[] = [];
  const addTriangle = (a: number, b: number, c: number) => {
    const point = (index: number) => points.slice(index * 3, index * 3 + 3);
    const pa = point(a); const pb = point(b); const pc = point(c);
    const ab = pb.map((value, axis) => value - pa[axis]);
    const ac = pc.map((value, axis) => value - pa[axis]);
    const areaTwice = Math.hypot(
      ab[1] * ac[2] - ab[2] * ac[1],
      ab[2] * ac[0] - ab[0] * ac[2],
      ab[0] * ac[1] - ab[1] * ac[0],
    );
    if (areaTwice > tolerance * tolerance) polys.push(3, a, b, c);
  };
  for (const strip of strips) {
    const segmentCount = strip.closed ? strip.parent.length - 1 : strip.parent.length - 1;
    const parentOffset = points.length / 3;
    for (const vertex of strip.parent) { points.push(...vertex.point); normals.push(...vertex.normal); }
    const childOffset = points.length / 3;
    for (const vertex of strip.child) { points.push(...vertex.point); normals.push(...vertex.normal); }
    for (let index = 0; index < segmentCount; index += 1) {
      addTriangle(parentOffset + index, childOffset + index, childOffset + index + 1);
      addTriangle(parentOffset + index, childOffset + index + 1, parentOffset + index + 1);
    }
  }
  if (polys.length === 0) return null;
  const pointArray = Float32Array.from(points); const polyArray = Uint32Array.from(polys);
  const normalArray = Float32Array.from(normals);
  const resultBounds: [number, number, number, number, number, number] = [Infinity, -Infinity, Infinity, -Infinity, Infinity, -Infinity];
  for (let index = 0; index < pointArray.length; index += 3) for (let axis = 0; axis < 3; axis += 1) {
    resultBounds[axis * 2] = Math.min(resultBounds[axis * 2], pointArray[index + axis]);
    resultBounds[axis * 2 + 1] = Math.max(resultBounds[axis * 2 + 1], pointArray[index + axis]);
  }
  return { points: pointArray, polys: polyArray, normals: normalArray, bounds: resultBounds,
    triangles: polyArray.length / 4, bytes: pointArray.byteLength + polyArray.byteLength + normalArray.byteLength };
}

const UNBOUNDED_OWNERSHIP = {
  ownedCellBounds: [-Infinity, Infinity, -Infinity, Infinity, -Infinity, Infinity] as [number, number, number, number, number, number],
  includeUpper: [true, true, true] as [boolean, boolean, boolean],
};

export function composeNestedSurfaceGeometry(
  geometries: readonly (VortexSurfaceGeometry | null)[],
  childBounds: readonly (NestedSurfaceBounds | null)[],
): { geometry: VortexSurfaceGeometry | null; transitionTriangles: number } {
  if (geometries.length !== childBounds.length) {
    throw new Error("Nested surface geometry and ownership chains differ");
  }
  const owned: OwnedBrickSurface[] = [];
  let transitionTriangles = 0;
  for (let index = 0; index < geometries.length; index += 1) {
    const source = geometries[index];
    const bounds = childBounds[index];
    const clipped = bounds ? clipSurfaceOutsideBounds(source, bounds) : source;
    if (clipped) owned.push({ brickId: `${index}:owned`, geometry: clipped, ...UNBOUNDED_OWNERSHIP });
    if (!bounds || index + 1 >= geometries.length) continue;
    const span = Math.max(...bounds.upper.map((upper, axis) => upper - bounds.lower[axis]));
    const tolerance = Math.max(1e-5, span * 1e-8);
    const transition = stitchNestedSurfaceTransition(clipped, geometries[index + 1], bounds, tolerance);
    if (transition) {
      transitionTriangles += transition.triangles;
      owned.push({ brickId: `${index}:transition`, geometry: transition, ...UNBOUNDED_OWNERSHIP });
    }
  }
  return { geometry: mergeOwnedBrickSurfaces(owned, 1e-5), transitionTriangles };
}
