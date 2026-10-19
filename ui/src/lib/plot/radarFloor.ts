/* Radar floor: hyperslab planning, column reduction, RGBA encoding of 2-D reflectivity/scalar data.
 * Volume indexed [x][y][z], z=0 ground, z up. Output lattice [nx'][ny'] same x/y convention. */

import type { NativeFieldWindowRequest } from "@/lib/dataset/nativeFieldWindow";
import {
  MAX_NATIVE_FIELD_WINDOW_BYTES,
  MAX_NATIVE_FIELD_WINDOW_SOURCE_BYTES,
} from "@/lib/dataset/nativeFieldWindow";
import type { DomainBoundsM } from "@/lib/plot/domainAxes";
import { DBZ_VTK_STOPS } from "@/lib/plot/plotColormaps";
import type { ScalarPlanePolicy } from "@/lib/plot/scalarPlanePolicy";
import type { SceneScale } from "@/lib/plot/sceneScale";
import type { SignedFieldStats } from "@/lib/plot/coherentStructures";

// ---------- Public types ----------

/** Which vertical reduction produces the floor value at each (x, y) column. */
export type FloorReduction = "surface" | "column-max";

/** How the floor lattice is obtained; cost varies by orders of magnitude — see `resolveFloorSource`. */
export type FloorSourceKind =
  /** A rank-2 product the engine already computed. One value per column, read
   *  directly. */
  | "native-2d"
  /** One cell thick at the ground, taken from the rank-3 field. */
  | "surface-slab"
  /** The full rank-3 column, reduced in the browser. */
  | "column";

/** Which field the floor reads, and in what shape. */
export interface FloorSource {
  /** The field actually requested. May differ from the field the user picked. */
  fieldName: string;
  kind: FloorSourceKind;
  /** True when `fieldName` differs from the requested field, so the UI can
   *  disclose the substitution rather than quietly showing another product. */
  substituted: boolean;
}

/** Hyperslab plan result. `achievedNx/Ny` are strided counts, not nx/ny.
 * Vectors are rank 2 for `native-2d` sources, rank 3 otherwise. */
export interface FloorHyperslabPlan {
  /** NativeFieldWindowRequest fields, ready to embed. */
  origin: number[];
  extent: number[];
  stride: number[];
  /** Number of x samples the server will return: ceil((nx) / strideXY). */
  achievedNx: number;
  /** Number of y samples the server will return: ceil((ny) / strideXY). */
  achievedNy: number;
  /** The stride applied to both x and y axes. */
  strideXY: number;
  /** Bytes the backend must read (product of `extent`, unaffected by stride); used as `maxSourceBytes`. */
  sourceBytes: number;
}

/** Why no plan could be built. Carried so the UI can say which limit was hit. */
export type FloorPlanFailure =
  /** A grid dimension is zero, negative, fractional or non-finite. */
  | "degenerate-grid"
  /** Even at the protocol's ceiling, the aggregation would read too much. */
  | "source-budget";

export type FloorPlanOutcome =
  | { ok: true; plan: FloorHyperslabPlan }
  | { ok: false; failure: FloorPlanFailure; sourceBytes: number };

// ---------- Named constants (no bare magic numbers in production code) ----------

/** WebGL2-guaranteed minimum MAX_TEXTURE_SIZE; caps floor texture per axis. */
const WEBGL2_MIN_MAX_TEXTURE_SIZE = 4096;

/** dBZ below which clear-air alpha ramps to 0; 18 dBZ matches NWS ND precipitation threshold. */
const CLEAR_AIR_THRESHOLD_DBZ = 18;

/** dBZ where alpha ramp starts; fully transparent below this, linear to 255 at CLEAR_AIR_THRESHOLD_DBZ. */
const ALPHA_RAMP_START_DBZ = 10;

/** Bytes per float32 element, used when computing byte budget vs. lattice size. */
const FLOAT32_BYTES = 4;

/** Engine-computed rank-2 column-max products keyed by their rank-3 source field; exact substitutions only. */
const NATIVE_COLUMN_MAX_PRODUCTS: Readonly<Record<string, string>> = {
  reflectivity_dbz: "composite_reflectivity",
};

// ---------- Source selection ----------

/** Resolve the actual field/source to read; substitutes a native-2d product only when it is exact. */
export function resolveFloorSource(
  fieldName: string,
  reduction: FloorReduction,
  availableFields: readonly string[],
): FloorSource {
  if (reduction === "surface") {
    return { fieldName, kind: "surface-slab", substituted: false };
  }
  const native = NATIVE_COLUMN_MAX_PRODUCTS[fieldName];
  if (native && availableFields.includes(native)) {
    return { fieldName: native, kind: "native-2d", substituted: true };
  }
  return { fieldName, kind: "column", substituted: false };
}

// ---------- Hyperslab planning ----------

/** Source cells the aggregation must read per column for each source kind. */
function columnDepthFor(kind: FloorSourceKind, nz: number): number {
  if (kind === "column") return nz;
  return 1;
}

/** Plan the hyperslab for one floor source.
 * Satisfies both OUTPUT (strided, ≤MAX_NATIVE_FIELD_WINDOW_BYTES) and SOURCE (unstrided extent, ≤MAX_SOURCE) budgets. */
export function planFloorHyperslab(
  nx: number,
  ny: number,
  nz: number,
  kind: FloorSourceKind,
  options: {
    outputBytesPerCell?: number;
    outputDepth?: number;
    maxOutputCells?: number;
  } = {},
): FloorPlanOutcome {
  if (
    !Number.isFinite(nx) || !Number.isFinite(ny) || !Number.isFinite(nz)
    || nx <= 0 || ny <= 0 || nz <= 0
    || !Number.isSafeInteger(nx) || !Number.isSafeInteger(ny) || !Number.isSafeInteger(nz)
  ) {
    return { ok: false, failure: "degenerate-grid", sourceBytes: 0 };
  }

  const depth = columnDepthFor(kind, nz);
  const outputBytesPerCell = options.outputBytesPerCell ?? FLOAT32_BYTES;
  const outputDepth = options.outputDepth ?? depth;
  const maxOutputCells = options.maxOutputCells ?? Number.MAX_SAFE_INTEGER;
  if (!Number.isSafeInteger(outputBytesPerCell) || outputBytesPerCell < FLOAT32_BYTES
      || !Number.isSafeInteger(outputDepth) || outputDepth <= 0
      || !Number.isSafeInteger(maxOutputCells) || maxOutputCells <= 0) {
    return { ok: false, failure: "degenerate-grid", sourceBytes: 0 };
  }
  const sourceBytes = nx * ny * depth * FLOAT32_BYTES;
  if (sourceBytes > MAX_NATIVE_FIELD_WINDOW_SOURCE_BYTES) {
    return { ok: false, failure: "source-budget", sourceBytes };
  }

  // Find the smallest stride s ≥ 1 such that:
  //   (a) ceil(nx / s) × ceil(ny / s) × depth × FLOAT32_BYTES ≤ output budget
  //   (b) ceil(nx / s) ≤ WEBGL2_MIN_MAX_TEXTURE_SIZE
  //   (c) ceil(ny / s) ≤ WEBGL2_MIN_MAX_TEXTURE_SIZE
  //
  // We iterate from 1 upward. The loop is bounded: the values shrink as s grows,
  // so it terminates in at most max(nx, ny) iterations. In practice it stops
  // within a handful of steps because the lattice shrinks quadratically.
  for (let stride = 1; stride <= Math.max(nx, ny); stride++) {
    const sampledNx = Math.ceil(nx / stride);
    const sampledNy = Math.ceil(ny / stride);

    if (sampledNx > WEBGL2_MIN_MAX_TEXTURE_SIZE) continue;
    if (sampledNy > WEBGL2_MIN_MAX_TEXTURE_SIZE) continue;

    const bytes = sampledNx * sampledNy * outputDepth * outputBytesPerCell;
    if (bytes <= MAX_NATIVE_FIELD_WINDOW_BYTES
        && sampledNx * sampledNy * outputDepth <= maxOutputCells) {
      // Vectors must match the source field's rank, so a rank-2 product gets
      // rank-2 vectors. Sending a z axis for a 2-D field is rejected by the
      // response validator as a rank disagreement.
      const rank2 = kind === "native-2d";
      return {
        ok: true,
        plan: {
          origin: rank2 ? [0, 0] : [0, 0, 0],
          extent: rank2 ? [nx, ny] : [nx, ny, depth],
          stride: rank2 ? [stride, stride] : [stride, stride, 1],
          achievedNx: sampledNx,
          achievedNy: sampledNy,
          strideXY: stride,
          sourceBytes,
        },
      };
    }
  }

  // Unreachable in practice: at stride = max(nx,ny) the sampled lattice is 1×1.
  // Guard here for correctness completeness.
  return { ok: false, failure: "degenerate-grid", sourceBytes };
}

// ---------- Plane placement ----------

/** Sheet lift above the domain floor, as a fraction of domain height; clears the box edge. */
const FLOOR_Z_LIFT_FRACTION = 0.001;

/** Corner triple in DISPLAY metres, in the vtkPlaneSource convention. */
export interface FloorPlaneGeometry {
  /** Corner at (u=0, v=0). */
  origin: [number, number, number];
  /** Corner at (u=1, v=0) — the +x edge, so u maps to x. */
  point1: [number, number, number];
  /** Corner at (u=0, v=1) — the +y edge, so v maps to y. */
  point2: [number, number, number];
}

/** Recover the physical bottom interface represented by a one-cell terrain
 * slab. Centroid minus half exact thickness is the terrain surface; partial
 * x/y edge blocks use their actual contributing width. */
export function terrainFloorSurface(
  centroids: { x: Float64Array; y: Float64Array; z: Float64Array },
  volumesM3: Float64Array,
  shape: [number, number],
  sourceShape: [number, number],
  strideXY: number,
  dxM: number,
  dyM: number,
): { x: Float64Array; y: Float64Array; z: Float64Array } {
  const [nx, ny] = shape;
  const count = nx * ny;
  if (nx <= 0 || ny <= 0 || !Number.isSafeInteger(strideXY) || strideXY <= 0
      || !(dxM > 0) || !(dyM > 0)
      || centroids.x.length !== count || centroids.y.length !== count
      || centroids.z.length !== count || volumesM3.length !== count) {
    throw new Error("Terrain floor geometry does not match its bounded lattice");
  }
  const z = new Float64Array(count);
  for (let xi = 0; xi < nx; xi += 1) {
    const sourceWidth = Math.min(strideXY, sourceShape[0] - xi * strideXY);
    for (let yi = 0; yi < ny; yi += 1) {
      const index = xi * ny + yi;
      const sourceHeight = Math.min(strideXY, sourceShape[1] - yi * strideXY);
      const area = sourceWidth * dxM * sourceHeight * dyM;
      const volume = volumesM3[index];
      if (!(area > 0) || !(volume > 0)
          || !Number.isFinite(centroids.x[index])
          || !Number.isFinite(centroids.y[index])
          || !Number.isFinite(centroids.z[index])) {
        throw new Error(`Terrain floor cell ${index} has invalid physical geometry`);
      }
      z[index] = centroids.z[index] - 0.5 * volume / area;
    }
  }
  return { x: centroids.x, y: centroids.y, z };
}

export interface TerrainFloorMesh {
  points: Float32Array;
  polys: Uint32Array;
  tcoords: Float32Array;
}

export function terrainFloorMesh(
  surface: { x: Float64Array; y: Float64Array; z: Float64Array },
  shape: [number, number],
): TerrainFloorMesh {
  const [nx, ny] = shape;
  const count = nx * ny;
  if (nx < 2 || ny < 2 || surface.x.length !== count
      || surface.y.length !== count || surface.z.length !== count) {
    throw new Error("Terrain floor mesh does not match its bounded lattice");
  }
  const points = new Float32Array(count * 3);
  const tcoords = new Float32Array(count * 2);
  for (let yi = 0; yi < ny; yi += 1) {
    for (let xi = 0; xi < nx; xi += 1) {
      const source = xi * ny + yi;
      const target = yi * nx + xi;
      points[target * 3] = surface.x[source]!;
      points[target * 3 + 1] = surface.y[source]!;
      points[target * 3 + 2] = surface.z[source]!;
      tcoords[target * 2] = xi / (nx - 1);
      tcoords[target * 2 + 1] = yi / (ny - 1);
    }
  }
  const polys = new Uint32Array((nx - 1) * (ny - 1) * 8);
  let cursor = 0;
  for (let yi = 0; yi + 1 < ny; yi += 1) {
    for (let xi = 0; xi + 1 < nx; xi += 1) {
      const lowerLeft = yi * nx + xi;
      const lowerRight = lowerLeft + 1;
      const upperLeft = lowerLeft + nx;
      const upperRight = upperLeft + 1;
      polys.set([3, lowerLeft, lowerRight, upperLeft,
        3, lowerRight, upperRight, upperLeft], cursor);
      cursor += 8;
    }
  }
  return { points, polys, tcoords };
}

/**
 * Place the floor sheet in display space. True metres in, display metres out:
 * x/y carry `scale.xy` and z carries `scale.z`, the same transform the data
 * actors get from `applySceneScale`, so the sheet stays registered under any
 * display scale. Returns null for non-finite bounds or scales.
 */
export function floorPlaneGeometry(
  bounds: DomainBoundsM | null,
  scale: SceneScale,
): FloorPlaneGeometry | null {
  if (!bounds) return null;
  if (!Number.isFinite(scale.xy) || !Number.isFinite(scale.z)) return null;
  const finiteCorner = (corner: readonly number[]) =>
    corner.length === 3 && corner.every((value) => Number.isFinite(value));
  if (!finiteCorner(bounds.lower) || !finiteCorner(bounds.upper)) return null;

  const zLift = FLOOR_Z_LIFT_FRACTION * (bounds.upper[2] - bounds.lower[2]);
  const z = (bounds.lower[2] + zLift) * scale.z;
  return {
    origin: [bounds.lower[0] * scale.xy, bounds.lower[1] * scale.xy, z],
    point1: [bounds.upper[0] * scale.xy, bounds.lower[1] * scale.xy, z],
    point2: [bounds.lower[0] * scale.xy, bounds.upper[1] * scale.xy, z],
  };
}

// ---------- Reduction ----------

/** Reduce 3-D float32 [nx,ny,nz] (x slowest, z fastest) to 2-D [nx,ny].
 * `surface`: first finite k per column; `column-max`: max finite over full column. All-NaN → NaN. */
export function reduceFloor(
  values: Float32Array,
  shape: [number, number, number],
  reduction: FloorReduction,
): Float32Array {
  const [nx, ny, nz] = shape;
  const out = new Float32Array(nx * ny);

  for (let xi = 0; xi < nx; xi++) {
    for (let yi = 0; yi < ny; yi++) {
      const columnBase = (xi * ny + yi) * nz;

      if (reduction === "surface") {
        // Walk upward from k = 0; take the first finite value.
        let found = NaN;
        for (let zi = 0; zi < nz; zi++) {
          const v = values[columnBase + zi];
          if (Number.isFinite(v)) {
            found = v;
            break;
          }
        }
        out[xi * ny + yi] = found;
      } else {
        // column-max: max over all finite values.
        let max = NaN;
        for (let zi = 0; zi < nz; zi++) {
          const v = values[columnBase + zi];
          if (Number.isFinite(v)) {
            if (!Number.isFinite(max) || v > max) max = v;
          }
        }
        out[xi * ny + yi] = max;
      }
    }
  }

  return out;
}

// ---------- RGBA encoding ----------

/** Lerp between two VTK color stops, t clamped to [0,1]; returns [r,g,b] in [0,255]. */
function lerpDbzColor(
  lo: (typeof DBZ_VTK_STOPS)[number],
  hi: (typeof DBZ_VTK_STOPS)[number],
  t: number,
): [number, number, number] {
  const tc = Math.max(0, Math.min(1, t));
  return [
    Math.round((lo.r + (hi.r - lo.r) * tc) * 255),
    Math.round((lo.g + (hi.g - lo.g) * tc) * 255),
    Math.round((lo.b + (hi.b - lo.b) * tc) * 255),
  ];
}

/** Map scalar in [min, max] to [0,1], clamped; 0 for NaN/non-finite. */
function scalarToUnit(value: number, min: number, max: number): number {
  if (!Number.isFinite(value)) return 0;
  const span = max - min;
  if (!(span > 0)) return 0;
  return Math.max(0, Math.min(1, (value - min) / span));
}

/** Sample [r,g,b] from a sorted [t,r,g,b] ramp at unit-interval t. */
function sampleLinearRamp(
  stops: ReadonlyArray<readonly [number, number, number, number]>,
  t: number,
): [number, number, number] {
  if (t <= stops[0]![0]) return [stops[0]![1], stops[0]![2], stops[0]![3]];
  const last = stops[stops.length - 1]!;
  if (t >= last[0]) return [last[1], last[2], last[3]];
  for (let i = 1; i < stops.length; i++) {
    const hi = stops[i]!;
    const lo = stops[i - 1]!;
    if (t <= hi[0]) {
      const span = hi[0] - lo[0];
      const frac = span > 0 ? (t - lo[0]) / span : 0;
      return [
        Math.round(lo[1] + (hi[1] - lo[1]) * frac),
        Math.round(lo[2] + (hi[2] - lo[2]) * frac),
        Math.round(lo[3] + (hi[3] - lo[3]) * frac),
      ];
    }
  }
  return [last[1], last[2], last[3]];
}

// Viridis and Cool-to-Warm baked as unit-interval [t, r, g, b] ramps; approximate Plotly entries.
const VIRIDIS_RAMP: ReadonlyArray<readonly [number, number, number, number]> = [
  [0.0, 68, 1, 84],
  [0.25, 59, 82, 139],
  [0.5, 33, 145, 140],
  [0.75, 94, 201, 98],
  [1.0, 253, 231, 37],
];

const COOL_TO_WARM_RAMP: ReadonlyArray<readonly [number, number, number, number]> = [
  [0.0, 59, 76, 192],
  [0.5, 221, 221, 221],
  [1.0, 180, 4, 38],
];

/** Encode 2-D floor Float32Array [nx,ny] as RGBA8. Transposes field order (x slowest) to image row-major
 * (y slowest) so u↔x and v↔y for vtkPlaneSource TCoords. dBZ: DBZ_VTK_STOPS + clear-air alpha ramp. */
export function encodeFloorRgba(
  values: Float32Array,
  shape: [number, number],
  policy: ScalarPlanePolicy,
): Uint8ClampedArray<ArrayBuffer> {
  const [nx, ny] = shape;
  const out = new Uint8ClampedArray(nx * ny * 4);

  for (let xi = 0; xi < nx; xi++) {
    for (let yi = 0; yi < ny; yi++) {
      const srcIndex = xi * ny + yi;
      // Field order in, image row-major out — see the layout note above.
      const dstBase = (yi * nx + xi) * 4;
      const v = values[srcIndex];

      if (!Number.isFinite(v)) {
        // Transparent for NaN / ±Infinity — never paint false data.
        out[dstBase] = 0;
        out[dstBase + 1] = 0;
        out[dstBase + 2] = 0;
        out[dstBase + 3] = 0;
        continue;
      }

      if (policy.kind === "dbz") {
        // 1. Compute color via DBZ_VTK_STOPS linear interpolation.
        const stops = DBZ_VTK_STOPS;
        let r: number;
        let g: number;
        let b: number;

        if (v <= stops[0]!.scalar) {
          r = Math.round(stops[0]!.r * 255);
          g = Math.round(stops[0]!.g * 255);
          b = Math.round(stops[0]!.b * 255);
        } else if (v >= stops[stops.length - 1]!.scalar) {
          const last = stops[stops.length - 1]!;
          r = Math.round(last.r * 255);
          g = Math.round(last.g * 255);
          b = Math.round(last.b * 255);
        } else {
          // Binary search for the bracketing interval.
          let lo = 0;
          let hi = stops.length - 1;
          while (hi - lo > 1) {
            const mid = (lo + hi) >> 1;
            if (stops[mid]!.scalar <= v) lo = mid; else hi = mid;
          }
          const loStop = stops[lo]!;
          const hiStop = stops[hi]!;
          const span = hiStop.scalar - loStop.scalar;
          const t = span > 0 ? (v - loStop.scalar) / span : 0;
          [r, g, b] = lerpDbzColor(loStop, hiStop, t);
        }

        // 2. Alpha: ramp from 0 at ALPHA_RAMP_START_DBZ to 255 at CLEAR_AIR_THRESHOLD_DBZ.
        let alpha: number;
        if (v < ALPHA_RAMP_START_DBZ) {
          alpha = 0;
        } else if (v < CLEAR_AIR_THRESHOLD_DBZ) {
          const rampSpan = CLEAR_AIR_THRESHOLD_DBZ - ALPHA_RAMP_START_DBZ;
          alpha = Math.round(((v - ALPHA_RAMP_START_DBZ) / rampSpan) * 255);
        } else {
          alpha = 255;
        }

        out[dstBase] = r!;
        out[dstBase + 1] = g!;
        out[dstBase + 2] = b!;
        out[dstBase + 3] = alpha;
      } else if (policy.kind === "signed") {
        const t = scalarToUnit(v, policy.min, policy.max);
        const [r, g, b] = sampleLinearRamp(COOL_TO_WARM_RAMP, t);
        out[dstBase] = r;
        out[dstBase + 1] = g;
        out[dstBase + 2] = b;
        out[dstBase + 3] = 255;
      } else {
        // sequential
        const t = scalarToUnit(v, policy.min, policy.max);
        const [r, g, b] = sampleLinearRamp(VIRIDIS_RAMP, t);
        out[dstBase] = r;
        out[dstBase + 1] = g;
        out[dstBase + 2] = b;
        out[dstBase + 3] = 255;
      }
    }
  }

  padMissingRgbForLinearFiltering(out, shape);
  return out;
}

/**
 * Linear texture filtering interpolates RGB even when the neighbouring texel
 * has zero alpha. Missing samples were encoded as transparent black, so their
 * black RGB leaked into the last visible radar texel as a grey fringe. Copy a
 * neighbouring valid colour into only those missing texels while keeping alpha
 * exactly zero; missing data remains invisible and cannot write false radar.
 */
export function padMissingRgbForLinearFiltering(
  rgba: Uint8ClampedArray<ArrayBuffer>,
  shape: [number, number],
): void {
  const [nx, ny] = shape;
  if (rgba.length !== nx * ny * 4) return;
  for (let y = 0; y < ny; y += 1) {
    for (let x = 0; x < nx; x += 1) {
      const base = (y * nx + x) * 4;
      if (rgba[base + 3] !== 0 || rgba[base] !== 0
          || rgba[base + 1] !== 0 || rgba[base + 2] !== 0) continue;
      let bestBase = -1;
      let bestAlpha = 0;
      for (let oy = -1; oy <= 1; oy += 1) {
        const sy = y + oy;
        if (sy < 0 || sy >= ny) continue;
        for (let ox = -1; ox <= 1; ox += 1) {
          const sx = x + ox;
          if ((ox === 0 && oy === 0) || sx < 0 || sx >= nx) continue;
          const candidate = (sy * nx + sx) * 4;
          const alpha = rgba[candidate + 3] ?? 0;
          if (alpha > bestAlpha) {
            bestAlpha = alpha;
            bestBase = candidate;
          }
        }
      }
      if (bestBase < 0) continue;
      rgba[base] = rgba[bestBase] ?? 0;
      rgba[base + 1] = rgba[bestBase + 1] ?? 0;
      rgba[base + 2] = rgba[bestBase + 2] ?? 0;
    }
  }
}

/** Max request id bytes; mirrors MAX_REQUEST_ID_BYTES in outputs.rs and datasets.rs. */
export const MAX_FIELD_REQUEST_ID_BYTES = 128;

/** The character set both backend validators accept. */
const REQUEST_ID_ILLEGAL = /[^A-Za-z0-9._-]/g;

/** Build a floor read request id. Uses a per-read `sequence` (not path) to ensure uniqueness and avoid in-flight collisions. */
export function floorRequestId(
  sequence: number,
  stepIndex: number,
  fieldName: string,
  reduction: FloorReduction,
): string {
  const prefix = `floor:${sequence}:${stepIndex}:`;
  // Truncate the one variable-length segment rather than the whole id, so the
  // sequence stays intact and ids stay unique at any field-name length.
  const budget = MAX_FIELD_REQUEST_ID_BYTES - prefix.length - reduction.length - 1;
  const field = fieldName
    .replace(REQUEST_ID_ILLEGAL, "_")
    .slice(0, Math.max(0, budget));
  return `${prefix}${field}:${reduction}`;
}

/** Build the NativeFieldWindowRequest from a plan; `maxSourceBytes` is stated explicitly, not defaulted. */
export function floorWindowRequest(
  requestId: string,
  plan: FloorHyperslabPlan,
  aggregation: NativeFieldWindowRequest["aggregation"] = "cell_mean",
): NativeFieldWindowRequest {
  return {
    schemaVersion: 2,
    requestId,
    aggregation,
    origin: [...plan.origin],
    extent: [...plan.extent],
    stride: [...plan.stride],
    maxOutputBytes: MAX_NATIVE_FIELD_WINDOW_BYTES,
    maxSourceBytes: Math.max(plan.sourceBytes, 1),
  };
}

/** Robust percentile-based stats over a reduced floor (p2/p98); excludes NaN. Returns null if no finite values. */
export function floorStats(values: Float32Array): SignedFieldStats | null {
  const finite = new Float32Array(values.length);
  let finiteCount = 0;
  for (let i = 0; i < values.length; i++) {
    const v = values[i]!;
    if (Number.isFinite(v)) finite[finiteCount++] = v;
  }
  if (finiteCount === 0) return null;
  const sorted = finite.subarray(0, finiteCount);
  sorted.sort();

  const at = (fraction: number) => {
    const index = Math.min(
      finiteCount - 1,
      Math.max(0, Math.round(fraction * (finiteCount - 1))),
    );
    return sorted[index]!;
  };

  let positiveStart = 0;
  while (positiveStart < finiteCount && sorted[positiveStart]! <= 0) positiveStart += 1;
  const positiveCount = finiteCount - positiveStart;
  const positiveP98 = positiveCount === 0 ? undefined : sorted[
    positiveStart + Math.min(
      positiveCount - 1,
      Math.max(0, Math.round(0.98 * (positiveCount - 1))),
    )
  ]!;

  return {
    min: sorted[0]!,
    max: sorted[finiteCount - 1]!,
    p2: at(0.02),
    p98: at(0.98),
    ...(positiveP98 === undefined ? {} : { positiveP98 }),
  };
}

// ---------- Re-export named constants for tests and wiring ----------
export { WEBGL2_MIN_MAX_TEXTURE_SIZE, CLEAR_AIR_THRESHOLD_DBZ, ALPHA_RAMP_START_DBZ };
