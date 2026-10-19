import type { VortexSurfaceGeometry } from "./vortexContourProtocol";

function finitePoint(point: readonly number[] | null | undefined): point is readonly [number, number, number] {
  return point?.length === 3 && point.every(Number.isFinite);
}

function find(parent: Int32Array, value: number): number {
  let root = value;
  while (parent[root] !== root) root = parent[root]!;
  while (parent[value] !== value) {
    const next = parent[value]!;
    parent[value] = root;
    value = next;
  }
  return root;
}

function union(parent: Int32Array, rank: Uint8Array, left: number, right: number): void {
  let a = find(parent, left);
  let b = find(parent, right);
  if (a === b) return;
  if (rank[a]! < rank[b]!) [a, b] = [b, a];
  parent[b] = a;
  if (rank[a] === rank[b]) rank[a] = rank[a]! + 1;
}

export interface ConnectedSurfaceHierarchy {
  /** Component nearest the solved-field anchor; owns physical classification
   * and condensation association. */
  primary: VortexSurfaceGeometry | null;
  /** Primary plus nearby resolved companion vortices in one draw buffer. */
  combined: VortexSurfaceGeometry | null;
  componentCount: number;
}

interface ComponentSummary {
  root: number;
  distanceSquared: number;
  triangles: number;
}

function compactComponents(
  surface: VortexSurfaceGeometry | null,
  parent: Int32Array,
  cellOffsets: readonly number[],
  roots: ReadonlySet<number>,
): VortexSurfaceGeometry | null {
  if (!surface || roots.size === 0) return null;
  const pointCount = Math.floor(surface.points.length / 3);
  const selectedOffsets = cellOffsets.filter((cellOffset) =>
    roots.has(find(parent, surface.polys[cellOffset + 1]!))
  );
  if (selectedOffsets.length === 0) return null;
  const pointMap = new Int32Array(pointCount);
  pointMap.fill(-1);
  const compactPoints: number[] = [];
  const compactNormals: number[] = [];
  const compactPolys: number[] = [];
  const hasNormals = surface.normals.length === surface.points.length;
  const bounds: [number, number, number, number, number, number] = [
    Infinity, -Infinity, Infinity, -Infinity, Infinity, -Infinity,
  ];
  for (const cellOffset of selectedOffsets) {
    const count = surface.polys[cellOffset]!;
    compactPolys.push(count);
    for (let local = 1; local <= count; local += 1) {
      const sourcePoint = surface.polys[cellOffset + local]!;
      let targetPoint = pointMap[sourcePoint]!;
      if (targetPoint < 0) {
        targetPoint = compactPoints.length / 3;
        pointMap[sourcePoint] = targetPoint;
        const base = sourcePoint * 3;
        const x = surface.points[base]!;
        const y = surface.points[base + 1]!;
        const z = surface.points[base + 2]!;
        compactPoints.push(x, y, z);
        if (hasNormals) {
          compactNormals.push(
            surface.normals[base]!,
            surface.normals[base + 1]!,
            surface.normals[base + 2]!,
          );
        }
        bounds[0] = Math.min(bounds[0], x);
        bounds[1] = Math.max(bounds[1], x);
        bounds[2] = Math.min(bounds[2], y);
        bounds[3] = Math.max(bounds[3], y);
        bounds[4] = Math.min(bounds[4], z);
        bounds[5] = Math.max(bounds[5], z);
      }
      compactPolys.push(targetPoint);
    }
  }
  const points = Float32Array.from(compactPoints);
  const polys = Uint32Array.from(compactPolys);
  const normals = Float32Array.from(compactNormals);
  return {
    points,
    polys,
    normals,
    bounds,
    triangles: selectedOffsets.length,
    bytes: points.byteLength + polys.byteLength + normals.byteLength,
  };
}

/**
 * Preserve a physically anchored vortex hierarchy without dropping resolved
 * suction-vortex companions. Classification remains tied to the nearest
 * component, while nearby nontrivial components are compacted into one GPU
 * buffer. Tiny marching-cubes flecks are rejected before they consume the
 * retained triangle budget.
 */
export function connectedSurfaceHierarchyNearestAnchor(
  surface: VortexSurfaceGeometry | null,
  anchor: readonly [number, number, number] | null | undefined,
  maxDistanceM = Number.POSITIVE_INFINITY,
  options: { maxComponents?: number; minimumTriangles?: number } = {},
): ConnectedSurfaceHierarchy {
  if (!surface) return { primary: null, combined: null, componentCount: 0 };
  if (!finitePoint(anchor)) return { primary: surface, combined: surface, componentCount: 1 };
  const pointCount = Math.floor(surface.points.length / 3);
  if (pointCount === 0 || surface.polys.length === 0) {
    return { primary: null, combined: null, componentCount: 0 };
  }

  const parent = Int32Array.from({ length: pointCount }, (_, index) => index);
  const rank = new Uint8Array(pointCount);
  const cellOffsets: number[] = [];
  let offset = 0;
  while (offset < surface.polys.length) {
    const count = surface.polys[offset]!;
    if (count < 3 || offset + count >= surface.polys.length) {
      return { primary: null, combined: null, componentCount: 0 };
    }
    const first = surface.polys[offset + 1]!;
    if (first >= pointCount) return { primary: null, combined: null, componentCount: 0 };
    cellOffsets.push(offset);
    for (let local = 2; local <= count; local += 1) {
      const point = surface.polys[offset + local]!;
      if (point >= pointCount) return { primary: null, combined: null, componentCount: 0 };
      union(parent, rank, first, point);
    }
    offset += count + 1;
  }

  const summaries = new Map<number, ComponentSummary>();
  for (let point = 0; point < pointCount; point += 1) {
    const root = find(parent, point);
    const base = point * 3;
    const dx = surface.points[base]! - anchor[0];
    const dy = surface.points[base + 1]! - anchor[1];
    const dz = surface.points[base + 2]! - anchor[2];
    const distanceSquared = dx * dx + dy * dy + dz * dz;
    const summary = summaries.get(root);
    if (summary) summary.distanceSquared = Math.min(summary.distanceSquared, distanceSquared);
    else summaries.set(root, { root, distanceSquared, triangles: 0 });
  }
  for (const cellOffset of cellOffsets) {
    const root = find(parent, surface.polys[cellOffset + 1]!);
    const summary = summaries.get(root);
    if (summary) summary.triangles += 1;
  }

  const maximumDistanceSquared = maxDistanceM * maxDistanceM;
  const ranked = [...summaries.values()]
    .filter((summary) => summary.distanceSquared <= maximumDistanceSquared)
    .sort((left, right) => left.distanceSquared - right.distanceSquared
      || right.triangles - left.triangles);
  const primarySummary = ranked[0];
  if (!primarySummary) return { primary: null, combined: null, componentCount: 0 };
  const minimumTriangles = Math.max(1, Math.floor(options.minimumTriangles ?? 8));
  const maxComponents = Math.max(1, Math.floor(options.maxComponents ?? 16));
  const retained = ranked.filter((summary, index) =>
    index === 0 || summary.triangles >= minimumTriangles
  ).slice(0, maxComponents);
  const primary = compactComponents(surface, parent, cellOffsets, new Set([primarySummary.root]));
  const combined = retained.length === 1
    ? primary
    : compactComponents(surface, parent, cellOffsets, new Set(retained.map(({ root }) => root)));
  return { primary, combined, componentCount: retained.length };
}

/** Compatibility helper for callers that explicitly need one component. */
export function connectedSurfaceNearestAnchor(
  surface: VortexSurfaceGeometry | null,
  anchor: readonly [number, number, number] | null | undefined,
  maxDistanceM = Number.POSITIVE_INFINITY,
): VortexSurfaceGeometry | null {
  return connectedSurfaceHierarchyNearestAnchor(surface, anchor, maxDistanceM, {
    maxComponents: 1,
    minimumTriangles: 1,
  }).primary;
}
