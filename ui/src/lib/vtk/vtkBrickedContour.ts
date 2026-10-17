import type { VortexSurfaceGeometry } from "./vortexContourProtocol";
import type { ContourBrickPlan, ContourBrickPlanItem } from "@/lib/plot/fieldBrickHierarchy";
import { contourUsageViolation } from "./vtkContourBudget";

type Bounds = [number, number, number, number, number, number];

export interface OwnedBrickSurface {
  brickId: string;
  geometry: VortexSurfaceGeometry;
  ownedCellBounds: Bounds;
  includeUpper: [boolean, boolean, boolean];
}

function ownsPoint(
  bounds: Bounds,
  includeUpper: readonly boolean[],
  point: readonly number[],
): boolean {
  for (let axis = 0; axis < 3; axis += 1) {
    const value = point[axis];
    const lower = bounds[axis * 2];
    const upper = bounds[axis * 2 + 1];
    if (value < lower || (includeUpper[axis] ? value > upper : value >= upper)) return false;
  }
  return true;
}

function pointKey(points: Float32Array, index: number, tolerance: number): string {
  const offset = index * 3;
  return [points[offset], points[offset + 1], points[offset + 2]]
    .map((value) => Math.round(value / tolerance))
    .join(":");
}

function triangleIndices(polys: Uint32Array): Array<[number, number, number]> {
  const triangles: Array<[number, number, number]> = [];
  for (let offset = 0; offset < polys.length;) {
    const count = polys[offset];
    if (count !== 3 || offset + 3 >= polys.length) {
      throw new Error("Bricked contour geometry must contain explicit triangles");
    }
    triangles.push([polys[offset + 1], polys[offset + 2], polys[offset + 3]]);
    offset += 4;
  }
  return triangles;
}

export function mergeOwnedBrickSurfaces(
  input: readonly OwnedBrickSurface[],
  coordinateTolerance = 1e-5,
): VortexSurfaceGeometry | null {
  if (!Number.isFinite(coordinateTolerance) || coordinateTolerance <= 0) {
    throw new Error("Bricked contour merge tolerance must be positive");
  }
  const chunks = [...input].sort((left, right) => left.brickId.localeCompare(right.brickId));
  const positions: number[] = [];
  const normalSums: number[] = [];
  const pointIndices = new Map<string, number>();
  const triangleKeys = new Set<string>();
  const triangles: Array<[number, number, number]> = [];

  for (const chunk of chunks) {
    const { points, normals, polys } = chunk.geometry;
    if (points.length % 3 !== 0 || normals.length !== points.length) {
      throw new Error(`Brick ${chunk.brickId} has inconsistent point or normal arrays`);
    }
    for (const triangle of triangleIndices(polys)) {
      if (triangle.some((index) => index * 3 + 2 >= points.length)) {
        throw new Error(`Brick ${chunk.brickId} triangle references an absent point`);
      }
      const centroid = [0, 1, 2].map((axis) => triangle.reduce(
        (sum, point) => sum + points[point * 3 + axis], 0,
      ) / 3);
      if (!ownsPoint(chunk.ownedCellBounds, chunk.includeUpper, centroid)) continue;
      const global = triangle.map((local) => {
        const key = pointKey(points, local, coordinateTolerance);
        let index = pointIndices.get(key);
        if (index == null) {
          index = positions.length / 3;
          pointIndices.set(key, index);
          const offset = local * 3;
          positions.push(points[offset], points[offset + 1], points[offset + 2]);
          normalSums.push(0, 0, 0);
        }
        const localOffset = local * 3;
        const globalOffset = index * 3;
        normalSums[globalOffset] += normals[localOffset];
        normalSums[globalOffset + 1] += normals[localOffset + 1];
        normalSums[globalOffset + 2] += normals[localOffset + 2];
        return index;
      }) as [number, number, number];
      const identity = [...global].sort((left, right) => left - right).join(":");
      if (triangleKeys.has(identity)) continue;
      triangleKeys.add(identity);
      triangles.push(global);
    }
  }
  if (triangles.length === 0) return null;
  const normals = new Float32Array(normalSums.length);
  for (let offset = 0; offset < normalSums.length; offset += 3) {
    const length = Math.hypot(normalSums[offset], normalSums[offset + 1], normalSums[offset + 2]);
    if (length > 0) {
      normals[offset] = normalSums[offset] / length;
      normals[offset + 1] = normalSums[offset + 1] / length;
      normals[offset + 2] = normalSums[offset + 2] / length;
    }
  }
  const points = Float32Array.from(positions);
  const polys = new Uint32Array(triangles.length * 4);
  triangles.forEach((triangle, index) => {
    const offset = index * 4;
    polys[offset] = 3;
    polys[offset + 1] = triangle[0];
    polys[offset + 2] = triangle[1];
    polys[offset + 3] = triangle[2];
  });
  const bounds: Bounds = [Infinity, -Infinity, Infinity, -Infinity, Infinity, -Infinity];
  for (let offset = 0; offset < points.length; offset += 3) {
    bounds[0] = Math.min(bounds[0], points[offset]);
    bounds[1] = Math.max(bounds[1], points[offset]);
    bounds[2] = Math.min(bounds[2], points[offset + 1]);
    bounds[3] = Math.max(bounds[3], points[offset + 1]);
    bounds[4] = Math.min(bounds[4], points[offset + 2]);
    bounds[5] = Math.max(bounds[5], points[offset + 2]);
  }
  return {
    points,
    polys,
    normals,
    bounds,
    triangles: triangles.length,
    bytes: points.byteLength + polys.byteLength + normals.byteLength,
  };
}

export async function executeBrickedContourPlan(
  plan: ContourBrickPlan,
  input: {
    signal: AbortSignal;
    extract: (item: ContourBrickPlanItem, signal: AbortSignal) => Promise<VortexSurfaceGeometry | null>;
    ownership: (item: ContourBrickPlanItem) => Pick<OwnedBrickSurface, "ownedCellBounds" | "includeUpper">;
    coordinateTolerance?: number;
    maximumOutputBytes?: number;
    maximumTriangles?: number;
    onProgress?: (completed: number, total: number) => void;
  },
): Promise<VortexSurfaceGeometry | null> {
  const surfaces: OwnedBrickSurface[] = [];
  let completed = 0;
  let retainedBytes = 0;
  let retainedTriangles = 0;
  for (const item of plan.items) {
    if (input.signal.aborted) throw new DOMException("Bricked contour extraction cancelled", "AbortError");
    const geometry = await input.extract(item, input.signal);
    if (input.signal.aborted) throw new DOMException("Bricked contour extraction cancelled", "AbortError");
    if (geometry && geometry.triangles > 0) {
      retainedBytes += geometry.bytes;
      retainedTriangles += geometry.triangles;
      const violation = input.maximumOutputBytes == null && input.maximumTriangles == null
        ? null
        : contourUsageViolation(
          { bytes: retainedBytes, triangles: retainedTriangles },
          "Progressive contour output",
          input.maximumOutputBytes ?? Number.MAX_SAFE_INTEGER,
          input.maximumTriangles ?? Number.MAX_SAFE_INTEGER,
        );
      if (violation) throw new Error(violation);
      surfaces.push({ brickId: item.brickId, geometry, ...input.ownership(item) });
    }
    completed += 1;
    input.onProgress?.(completed, plan.items.length);
  }
  return mergeOwnedBrickSurfaces(surfaces, input.coordinateTolerance);
}
