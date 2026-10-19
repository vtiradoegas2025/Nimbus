import type { SceneScale } from "@/lib/plot/sceneScale";
import { assertVtkArrayAllocation } from "@/lib/vtk/vtkArrayOwnership";

/* Display-scaled polyline copy for tube rendering. Owns the scaled buffer
 * outright — vtkTransformPolyDataFilter aliases (and clears) input cell arrays
 * on its second execution, silently emptying all geometry downstream. */

/** Minimal structural views of the vtk.js objects this module touches. */
interface ReadableArray {
  /** vtk.js widens this to number[] | TypedArray, so read it as ArrayLike. */
  getData: () => ArrayLike<number> | null;
}

export interface ScaledGeometrySource {
  getPoints: () => ReadableArray | null;
  getLines: () => ReadableArray | null;
  getPointData: () => { getScalars: () => ReadableArray | null } | null;
}

export interface ScaledGeometryTarget {
  getPoints: () => { setData: (values: Float32Array, components: number) => void };
  getLines: () => { setData: (values: Uint32Array) => void };
  modified: () => void;
}

/** Points, topology and scalars for one tube input, in display coordinates. */
export interface ScaledGeometry {
  points: Float32Array;
  lines: Uint32Array;
  /** Null when the source carries no point scalars (an uncoloured tube set). */
  scalars: Float32Array | null;
}

/** Scale xyz triples for display (x/y by scale.xy, z by scale.z). Always allocates; vtk.js retains the buffer. */
export function scalePointTriples(
  source: ArrayLike<number>,
  scale: SceneScale,
): Float32Array {
  if (source.length % 3 !== 0) {
    throw new Error(`Point buffer length ${source.length} is not a multiple of 3`);
  }
  assertVtkArrayAllocation(source.length / 3, 3);
  const scaled = new Float32Array(source.length);
  for (let i = 0; i < source.length; i += 3) {
    scaled[i] = source[i] * scale.xy;
    scaled[i + 1] = source[i + 1] * scale.xy;
    scaled[i + 2] = source[i + 2] * scale.z;
  }
  return scaled;
}

/**
 * Build the display-scaled geometry for a polyline source.
 *
 * Returns null when there is nothing tubeable -- no points, or no line topology
 * (an integration that produced no polylines, or a source a filter has since
 * emptied) -- so the caller hides its actor instead of rendering a stale frame.
 */
export function buildScaledGeometry(
  source: ScaledGeometrySource | null | undefined,
  scale: SceneScale,
): ScaledGeometry | null {
  const points = source?.getPoints()?.getData();
  const lines = source?.getLines()?.getData();
  if (!points || points.length === 0) return null;
  if (!lines || lines.length === 0) return null;

  const scalars = source?.getPointData()?.getScalars()?.getData() ?? null;
  return {
    points: scalePointTriples(points, scale),
    // Fresh copies, never the source's own buffers: a downstream vtk.js filter
    // may alias what we hand it and then clear it in place (see the note above).
    lines: Uint32Array.from(lines),
    scalars: scalars ? Float32Array.from(scalars) : null,
  };
}

/**
 * Install scaled points and topology on a caller-owned tube input.
 *
 * Scalars are attached separately by the caller, since binding them needs a real
 * vtkDataArray and the owned-slot rule in lib/vtk/vtkArrayOwnership.ts.
 */
export function applyScaledGeometry(
  target: ScaledGeometryTarget,
  geometry: ScaledGeometry,
): void {
  target.getPoints().setData(geometry.points, 3);
  target.getLines().setData(geometry.lines);
  target.modified();
}
