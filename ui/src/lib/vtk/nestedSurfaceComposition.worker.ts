import { clipSurfaceOutsideBounds, stitchNestedSurfaceTransition, type NestedSurfaceBounds } from "./nestedSurfaceOwnership";
import type { OwnedBrickSurface } from "./vtkBrickedContour";
import { contourUsageViolation } from "./vtkContourBudget";
import type { NestedSurfaceCompositionWorkerReply, NestedSurfaceCompositionWorkerRequest } from "./nestedSurfaceCompositionProtocol";
import type { VortexSurfaceGeometry } from "./vortexContourProtocol";

const scope = self as unknown as {
  onmessage: ((event: { data: NestedSurfaceCompositionWorkerRequest }) => void) | null;
  postMessage: (reply: NestedSurfaceCompositionWorkerReply, transfer?: ArrayBuffer[]) => void;
  close: () => void;
};
const OWNERSHIP = {
  ownedCellBounds: [-Infinity, Infinity, -Infinity, Infinity, -Infinity, Infinity] as [number, number, number, number, number, number],
  includeUpper: [true, true, true] as [boolean, boolean, boolean],
};
interface Accumulator {
  previous: VortexSurfaceGeometry | null;
  owned: OwnedBrickSurface[];
  transitionTriangles: number;
}
interface Active {
  requestId: number;
  identity: string;
  childBounds: Array<NestedSurfaceBounds | null>;
  layerCount: number;
  nextLayerIndex: number;
  maximumOutputBytes: number;
  maximumTriangles: number;
  knownScratchBytes: number;
  primary: Accumulator;
  opposite: Accumulator;
}
let active: Active | null = null;

function appendLayer(target: Accumulator, geometry: VortexSurfaceGeometry | null, bounds: NestedSurfaceBounds | null, index: number): void {
  if (index === 0) { target.previous = geometry; return; }
  const previous = target.previous;
  const clipped = bounds ? clipSurfaceOutsideBounds(previous, bounds) : previous;
  if (clipped) target.owned.push({ brickId: `${index - 1}:owned`, geometry: clipped, ...OWNERSHIP });
  if (bounds) {
    const span = Math.max(...bounds.upper.map((upper, axis) => upper - bounds.lower[axis]));
    const transition = stitchNestedSurfaceTransition(clipped, geometry, bounds, Math.max(1e-5, span * 1e-8));
    if (transition) {
      target.transitionTriangles += transition.triangles;
      target.owned.push({ brickId: `${index - 1}:transition`, geometry: transition, ...OWNERSHIP });
    }
  }
  target.previous = geometry;
}

function concatenate(owned: readonly OwnedBrickSurface[]): VortexSurfaceGeometry | null {
  if (owned.length === 0) return null;
  const pointValues = owned.reduce((sum, item) => sum + item.geometry.points.length, 0);
  const polyValues = owned.reduce((sum, item) => sum + item.geometry.polys.length, 0);
  const normalValues = owned.reduce((sum, item) => sum + item.geometry.normals.length, 0);
  const points = new Float32Array(pointValues);
  const polys = new Uint32Array(polyValues);
  const normals = new Float32Array(normalValues);
  const bounds: [number, number, number, number, number, number] = [Infinity, -Infinity, Infinity, -Infinity, Infinity, -Infinity];
  let pointCursor = 0; let polyCursor = 0; let normalCursor = 0; let pointOffset = 0; let triangles = 0;
  for (const item of owned) {
    const geometry = item.geometry;
    if (geometry.points.length !== geometry.normals.length || geometry.polys.length % 4 !== 0) {
      throw new Error("Nested surface fragment arrays are inconsistent");
    }
    points.set(geometry.points, pointCursor);
    normals.set(geometry.normals, normalCursor);
    for (let index = 0; index < geometry.polys.length; index += 4) {
      if (geometry.polys[index] !== 3) throw new Error("Nested surface fragment is not triangulated");
      polys[polyCursor++] = 3;
      polys[polyCursor++] = geometry.polys[index + 1] + pointOffset;
      polys[polyCursor++] = geometry.polys[index + 2] + pointOffset;
      polys[polyCursor++] = geometry.polys[index + 3] + pointOffset;
      triangles += 1;
    }
    pointCursor += geometry.points.length;
    normalCursor += geometry.normals.length;
    pointOffset += geometry.points.length / 3;
    if (geometry.bounds) for (let axis = 0; axis < 3; axis += 1) {
      bounds[axis * 2] = Math.min(bounds[axis * 2], geometry.bounds[axis * 2]);
      bounds[axis * 2 + 1] = Math.max(bounds[axis * 2 + 1], geometry.bounds[axis * 2 + 1]);
    }
  }
  return { points, polys, normals, bounds: bounds.every(Number.isFinite) ? bounds : null,
    triangles, bytes: points.byteLength + polys.byteLength + normals.byteLength };
}

function finish(target: Accumulator): { geometry: VortexSurfaceGeometry | null; transitionTriangles: number } {
  if (target.previous) target.owned.push({ brickId: "leaf", geometry: target.previous, ...OWNERSHIP });
  const geometry = concatenate(target.owned);
  target.previous = null;
  target.owned = [];
  return { geometry, transitionTriangles: target.transitionTriangles };
}

function buffers(geometry: VortexSurfaceGeometry | null): ArrayBuffer[] {
  return geometry ? [geometry.points.buffer, geometry.polys.buffer, geometry.normals.buffer] as ArrayBuffer[] : [];
}

scope.onmessage = (event) => {
  const request = event.data;
  if (request.kind === "close") { active = null; scope.close(); return; }
  try {
    if (request.kind === "begin") {
      if (request.layerCount < 2 || request.childBounds.length !== request.layerCount) {
        throw new Error("Nested surface stream has an invalid hierarchy");
      }
      active = {
        ...request, nextLayerIndex: 0, knownScratchBytes: 0,
        primary: { previous: null, owned: [], transitionTriangles: 0 },
        opposite: { previous: null, owned: [], transitionTriangles: 0 },
      };
      scope.postMessage({ kind: "ack", requestId: request.requestId, identity: request.identity, nextLayerIndex: 0 });
      return;
    }
    if (!active || request.requestId !== active.requestId || request.identity !== active.identity) {
      throw new Error("Nested surface stream identity changed");
    }
    if (request.kind === "layer") {
      if (request.layerIndex !== active.nextLayerIndex || request.layerIndex >= active.layerCount) {
        throw new Error("Nested surface layer arrived out of order");
      }
      active.knownScratchBytes = Math.max(active.knownScratchBytes,
        (request.primary?.bytes ?? 0) + (request.opposite?.bytes ?? 0));
      const bounds = request.layerIndex === 0 ? null : active.childBounds[request.layerIndex - 1];
      appendLayer(active.primary, request.primary, bounds, request.layerIndex);
      appendLayer(active.opposite, request.opposite, bounds, request.layerIndex);
      active.nextLayerIndex += 1;
      scope.postMessage({ kind: "ack", requestId: request.requestId, identity: request.identity,
        nextLayerIndex: active.nextLayerIndex });
      return;
    }
    if (active.nextLayerIndex !== active.layerCount) throw new Error("Nested surface stream finished incompletely");
    const completed = active;
    active = null;
    const primary = finish(completed.primary);
    const opposite = finish(completed.opposite);
    const usage = [primary.geometry, opposite.geometry].reduce(
      (total, geometry) => ({ bytes: total.bytes + (geometry?.bytes ?? 0), triangles: total.triangles + (geometry?.triangles ?? 0) }),
      { bytes: 0, triangles: 0 },
    );
    const violation = contourUsageViolation(usage, "Nested contour output",
      completed.maximumOutputBytes, completed.maximumTriangles);
    if (violation) throw new Error(violation);
    scope.postMessage({ kind: "result", requestId: request.requestId, identity: request.identity, ok: true,
      primary, opposite, knownScratchBytes: completed.knownScratchBytes },
    [...buffers(primary.geometry), ...buffers(opposite.geometry)]);
  } catch (error) {
    active = null;
    scope.postMessage({ kind: "result", requestId: request.requestId, identity: request.identity, ok: false,
      error: error instanceof Error ? error.message : String(error) });
  }
};

scope.postMessage({ kind: "ready" });
