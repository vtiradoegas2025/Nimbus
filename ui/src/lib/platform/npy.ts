/*
 * NumPy .npy reader for the formats Nimbus produces (matches src/io/npy_writer.cpp).
 * Supports: <f4, C-order, rank 1-3, versions 1.0/2.0. Anything else throws.
 * Ref: https://numpy.org/doc/stable/reference/generated/numpy.lib.format.html
 */

const MAGIC = [0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59]; // \x93NUMPY
export const MAX_FIELD_RANK = 3;
export const MAX_FIELD_DIMENSION = 16_384;
export const MAX_FIELD_ELEMENTS = 64 * 1024 * 1024;
export const MAX_FIELD_DECODED_BYTES =
  MAX_FIELD_ELEMENTS * Float32Array.BYTES_PER_ELEMENT;
export const MAX_NPY_HEADER_BYTES = 64 * 1024;

export interface FieldDescriptor {
  format: "npy";
  version: readonly [number, number];
  rank: number;
  dimensions: readonly number[];
  elementCount: number;
  decodedBytes: number;
}

export interface NpyParseOptions {
  /** One or more exact shapes accepted by the selected step manifest. */
  expectedShapes?: readonly (readonly number[])[];
  expectedRank?: number;
  expectedLastDimension?: number;
}

export interface Npy3D {
  shape: [number, number, number];
  /** Always Float32Array — Nimbus writes <f4 only. */
  data: Float32Array;
  /** Present on native bounded windows. Terrain windows carry one exact
   * physical-volume centroid per returned block. */
  windowGeometry?: FieldWindowGeometry;
}

export interface FieldWindowGeometry {
  aggregation?: "cell_mean" | "linear_reflectivity_mean"
    | "block_max" | "block_min" | "magnitude_extreme" | "point_sample";
  weighting: "uniform_cell_volume" | "physical_cell_volume";
    sampleLocation: "block_center" | "source_cell_center" | "physical_volume_centroid";
  centroidsM: { x: Float64Array; y: Float64Array; z: Float64Array } | null;
  /** Exact physical volume represented by each terrain output block. */
  volumesM3?: Float64Array | null;
  /**
   * Native lattice this window was aggregated from, and the block size per axis.
   *
   * Carried because the returned array is already reduced: without it a consumer comparing
   * shapes sees no reduction and would disclose a block mean as native resolution.
   */
  sourceShape?: number[];
  /** Source-index origin and extent represented by this returned lattice. */
  origin?: number[];
  extent?: number[];
  stride?: number[];
  /** File/container path that actually served this bounded read. */
  sourceFormat?: "npy" | "npy_lod" | "zfp3d" | "netcdf" | "netcdf_classic"
    | "netcdf_64bit_offset" | "netcdf_64bit_data" | "netcdf4"
    | "netcdf4_classic" | "netcdf_unknown";
}

export interface NpyN {
  shape: number[];
  data: Float32Array;
  /** Present on arrays read from a file; derived plot transforms need not
   * preserve the source-file descriptor. */
  descriptor?: FieldDescriptor;
  windowGeometry?: FieldWindowGeometry;
  /** Metadata used only by the Python bridge to construct an honest xarray
   * object for on-demand reads. It stays out of the numeric cache identity. */
  pythonMetadata?: PythonFieldMetadata;
}

export interface PythonFieldMetadata {
  fieldName: string;
  unit: string | null;
  source: string;
  sourceName: string;
  frameIndex: number;
  stepIndex: number;
  simulationTimeS: number | null;
  coordinateSystem: string;
  terrainFollowing: boolean;
  /** True when the source mesh was terrain-following even if this particular
   * Python array was conservatively remapped to physical Cartesian z. */
  sourceTerrainFollowing?: boolean;
  coordinateResampling?: "none" | "conservative-terrain-to-cartesian";
  validOnly: boolean;
  /** Fidelity/provenance is explicit so Python analysis never has to infer it
   * from the returned shape (which is ambiguous after a crop or reduction). */
  dtype: "float32";
  sourceShape: number[];
  outputShape: number[];
  sampling: "native" | "cell_mean" | "linear_reflectivity_mean"
    | "block_max" | "block_min" | "magnitude_extreme" | "point_sample";
  aggregation: FieldWindowGeometry["aggregation"] | null;
  weighting: FieldWindowGeometry["weighting"] | null;
  sampleLocation: FieldWindowGeometry["sampleLocation"] | null;
  sourceFormat: FieldWindowGeometry["sourceFormat"] | null;
  compression: {
    codec: string;
    mode: string;
    modeParameter: number;
    modeParameterUnits: string;
    errorContract: string;
    sparseThreshold: number;
    float16Prefilter: boolean;
    decodedDtype: string;
  } | null;
  stride: number[];
  box: Array<[number, number]>;
  coordinates: Array<{
    name: "x" | "y" | "z" | "r" | "theta";
    values: number[];
    unit: "m" | "rad" | "model_level" | "index";
  }>;
}

export function parseNpy(
  buffer: ArrayBuffer | Uint8Array,
  options: NpyParseOptions = {},
): NpyN {
  const bytes =
    buffer instanceof Uint8Array ? buffer : new Uint8Array(buffer);
  if (bytes.length < 8) {
    throw new Error(`Truncated NPY: ${bytes.length} bytes`);
  }
  for (let i = 0; i < MAGIC.length; i++) {
    if (bytes[i] !== MAGIC[i]) {
      throw new Error("Not an NPY file (magic mismatch)");
    }
  }

  const major = bytes[6];
  const minor = bytes[7];
  if ((major !== 1 && major !== 2) || minor !== 0) {
    throw new Error(`Unsupported NPY version: ${major}.${minor}`);
  }

  let headerLen: number;
  let headerOffset: number;
  if (major === 1) {
    if (bytes.length < 10) throw new Error("Truncated NPY v1 header");
    headerLen = bytes[8] | (bytes[9] << 8);
    headerOffset = 10;
  } else {
    if (bytes.length < 12) throw new Error("Truncated NPY v2 header");
    headerLen =
      bytes[8] +
      bytes[9] * 0x100 +
      bytes[10] * 0x1_0000 +
      bytes[11] * 0x100_0000;
    headerOffset = 12;
  }

  if (headerLen <= 0 || headerLen > MAX_NPY_HEADER_BYTES) {
    throw new Error(
      `NPY header is ${headerLen} bytes; limit is ${MAX_NPY_HEADER_BYTES}`,
    );
  }

  const headerEnd = headerOffset + headerLen;
  if (!Number.isSafeInteger(headerEnd) || bytes.length < headerEnd) {
    throw new Error(
      `Truncated NPY header: need ${headerEnd}, have ${bytes.length}`,
    );
  }

  const headerText = new TextDecoder("ascii").decode(
    bytes.subarray(headerOffset, headerEnd),
  );

  const header = parseHeader(headerText);

  if (header.descr !== "<f4") {
    throw new Error(
      `Unsupported NPY dtype '${header.descr}' (only <f4 supported)`,
    );
  }
  if (header.fortranOrder) {
    throw new Error("Fortran-order NPY not supported");
  }

  const { elementCount: expectedElements, decodedBytes: expectedBytes } =
    validateFieldShape(header.shape);
  if (
    options.expectedRank !== undefined &&
    header.shape.length !== options.expectedRank
  ) {
    throw new Error(
      `NPY rank ${header.shape.length} does not match expected rank ${options.expectedRank}`,
    );
  }
  if (
    options.expectedLastDimension !== undefined &&
    header.shape.at(-1) !== options.expectedLastDimension
  ) {
    throw new Error(
      `NPY last dimension ${header.shape.at(-1)} does not match expected ${options.expectedLastDimension}`,
    );
  }
  if (
    options.expectedShapes &&
    !options.expectedShapes.some((shape) => shapesEqual(shape, header.shape))
  ) {
    const expected = options.expectedShapes
      .map((shape) => shape.join("×"))
      .join(" or ");
    throw new Error(
      `NPY shape ${header.shape.join("×")} does not match manifest shape ${expected}`,
    );
  }
  const bodyBytes = bytes.length - headerEnd;
  if (bodyBytes !== expectedBytes) {
    throw new Error(
      `NPY body size mismatch: need ${expectedBytes} bytes, have ${bodyBytes}`,
    );
  }

  // NPY header is padded to 64-byte boundary so body is usually 4-byte aligned.
  // Fall back to a copy only when the absolute offset is misaligned (off-spec writer).
  const absOffset = bytes.byteOffset + headerEnd;
  let data: Float32Array;
  if (absOffset % 4 === 0) {
    data = new Float32Array(bytes.buffer, absOffset, expectedElements);
  } else {
    const aligned = new ArrayBuffer(expectedBytes);
    new Uint8Array(aligned).set(
      bytes.subarray(headerEnd, headerEnd + expectedBytes),
    );
    data = new Float32Array(aligned);
  }

  return {
    shape: header.shape,
    data,
    descriptor: {
      format: "npy",
      version: [major, minor],
      rank: header.shape.length,
      dimensions: header.shape,
      elementCount: expectedElements,
      decodedBytes: expectedBytes,
    },
  };
}

function shapesEqual(a: readonly number[], b: readonly number[]): boolean {
  return a.length === b.length && a.every((dim, index) => dim === b[index]);
}

export function validateFieldShape(shape: readonly number[]): {
  elementCount: number;
  decodedBytes: number;
} {
  if (shape.length < 1 || shape.length > MAX_FIELD_RANK) {
    throw new Error(
      `Unsupported field rank ${shape.length} (expected 1..=${MAX_FIELD_RANK})`,
    );
  }
  let elementCount = 1;
  for (const dim of shape) {
    if (
      !Number.isSafeInteger(dim) ||
      dim <= 0 ||
      dim > MAX_FIELD_DIMENSION
    ) {
      throw new Error(
        `Field dimension ${dim} outside 1..=${MAX_FIELD_DIMENSION}`,
      );
    }
    elementCount *= dim;
    if (!Number.isSafeInteger(elementCount) || elementCount > MAX_FIELD_ELEMENTS) {
      throw new Error(
        `Field has ${elementCount} elements; limit is ${MAX_FIELD_ELEMENTS}`,
      );
    }
  }
  const decodedBytes = elementCount * Float32Array.BYTES_PER_ELEMENT;
  if (
    !Number.isSafeInteger(decodedBytes) ||
    decodedBytes > MAX_FIELD_DECODED_BYTES
  ) {
    throw new Error(
      `Decoded field is ${decodedBytes} bytes; limit is ${MAX_FIELD_DECODED_BYTES}`,
    );
  }
  return { elementCount, decodedBytes };
}

export function parseNpy3D(buffer: ArrayBuffer | Uint8Array): Npy3D {
  const arr = parseNpy(buffer);
  if (arr.shape.length !== 3) {
    throw new Error(
      `Expected 3D field, got ${arr.shape.length}D (shape=${arr.shape.join("×")})`,
    );
  }
  return { shape: arr.shape as [number, number, number], data: arr.data };
}

interface ParsedHeader {
  descr: string;
  fortranOrder: boolean;
  shape: number[];
}

/** Parse the Python-literal dict in an NPY header. Matches three known fields; ignores the rest. */
function parseHeader(text: string): ParsedHeader {
  const descrMatches = [
    ...text.matchAll(/'descr'\s*:\s*'([^']+)'/g),
  ];
  const fortranMatches = [
    ...text.matchAll(/'fortran_order'\s*:\s*(True|False)/g),
  ];
  const shapeMatches = [
    ...text.matchAll(/'shape'\s*:\s*\(([^)]*)\)/g),
  ];

  if (
    descrMatches.length !== 1 ||
    fortranMatches.length !== 1 ||
    shapeMatches.length !== 1
  ) {
    throw new Error(`Malformed NPY header: ${text}`);
  }
  const [descrMatch] = descrMatches;
  const [fortranMatch] = fortranMatches;
  const [shapeMatch] = shapeMatches;

  const shapeRaw = shapeMatch[1].trim();
  const parts = shapeRaw.split(",").map((s) => s.trim());
  if (parts.at(-1) === "" && parts.length > 1) parts.pop();
  if (parts.some((part) => !/^(0|[1-9]\d*)$/.test(part))) {
    throw new Error(`Bad dimension in NPY shape: '${shapeRaw}'`);
  }
  const shape = parts.map(Number);

  return {
    descr: descrMatch[1],
    fortranOrder: fortranMatch[1] === "True",
    shape,
  };
}

/* ---------- Slicing helpers ---------- */

export type SliceAxis = 0 | 1 | 2;

export interface Slice2D {
  /** rows × cols. Indexed as data[row][col]. Plotly's heatmap takes this shape. */
  data: number[][];
  rows: number;
  cols: number;
  /** Axis labels for the slice plane. E.g., axis=2 (z-slice) → 'r' (rows) × 'θ' (cols). */
  rowAxis: number;
  colAxis: number;
}

export function sliceDisplayAxes(axis: SliceAxis): { rowAxis: SliceAxis; colAxis: SliceAxis } {
  if (axis === 2) return { rowAxis: 0, colAxis: 1 };
  return { rowAxis: 2, colAxis: axis === 0 ? 1 : 0 };
}

/**
 * Extract a 2D slice at fixed `index` along `axis` from a (nr, nth, nz) field.
 * Vertical sections put height on Plotly's row/screen-y axis and the
 * horizontal coordinate on its column/screen-x axis. Horizontal slices retain
 * the established (nr,nth) row/column ordering.
 */
export function slice2D(field: Npy3D, axis: SliceAxis, index: number): Slice2D {
  const [nr, nth, nz] = field.shape;
  const fixedDim = field.shape[axis];
  if (index < 0 || index >= fixedDim) {
    throw new Error(
      `Slice index ${index} out of range for axis ${axis} (dim ${fixedDim})`,
    );
  }

  const data = field.data;
  // Strides in C order: stride[i] = product of dims after i.
  const sr = nth * nz;
  const sth = nz;
  const displayAxes = sliceDisplayAxes(axis);

  switch (axis) {
    case 0: {
      const rows = nz;
      const cols = nth;
      const out: number[][] = new Array(rows);
      const base = index * sr;
      for (let k = 0; k < rows; k++) {
        const row = new Array<number>(cols);
        for (let j = 0; j < cols; j++) row[j] = data[base + j * sth + k];
        out[k] = row;
      }
      return { data: out, rows, cols, ...displayAxes };
    }
    case 1: {
      const rows = nz;
      const cols = nr;
      const out: number[][] = new Array(rows);
      const offJ = index * sth;
      for (let k = 0; k < rows; k++) {
        const row = new Array<number>(cols);
        for (let i = 0; i < cols; i++) row[i] = data[i * sr + offJ + k];
        out[k] = row;
      }
      return { data: out, rows, cols, ...displayAxes };
    }
    case 2: {
      const rows = nr;
      const cols = nth;
      const out: number[][] = new Array(rows);
      for (let i = 0; i < rows; i++) {
        const row = new Array<number>(cols);
        const offI = i * sr;
        for (let j = 0; j < cols; j++) row[j] = data[offI + j * sth + index];
        out[i] = row;
      }
      return { data: out, rows, cols, ...displayAxes };
    }
  }
}

/** Min/max of the field data, ignoring NaN. Used to pick colormap range. */
export function fieldRange(field: NpyN): { min: number; max: number } {
  const d = field.data;
  let min = Number.POSITIVE_INFINITY;
  let max = Number.NEGATIVE_INFINITY;
  for (let i = 0; i < d.length; i++) {
    const v = d[i];
    if (!Number.isFinite(v)) continue;
    if (v < min) min = v;
    if (v > max) max = v;
  }
  if (!Number.isFinite(min)) {
    return { min: 0, max: 0 };
  }
  return { min, max };
}

/**
 * Approximate percentiles for robust color/opacity ranges. Samples up to ~200k
 * finite values (strided); `ps` are fractions in [0,1]; returns NaN on empty fields.
 */
export function fieldPercentiles(field: NpyN, ps: number[]): number[] {
  const d = field.data;
  const n = d.length;
  const SAMPLE_CAP = 200_000;
  const stride = Math.max(1, Math.floor(n / SAMPLE_CAP));
  const sample: number[] = [];
  for (let i = 0; i < n; i += stride) {
    const v = d[i];
    if (Number.isFinite(v)) sample.push(v);
  }
  if (sample.length === 0) return ps.map(() => NaN);
  sample.sort((a, b) => a - b);
  const last = sample.length - 1;
  return ps.map((p) => {
    const clamped = p < 0 ? 0 : p > 1 ? 1 : p;
    return sample[Math.round(clamped * last)];
  });
}

/* ---------- 1D extracts + scalar reductions ---------- */

export type Reducer = "max" | "min" | "mean" | "rms";

/** Reduce a whole field to a single NaN-skipping scalar. */
export function reduceScalar(field: NpyN, reducer: Reducer): number {
  const d = field.data;
  switch (reducer) {
    case "max": {
      let acc = Number.NEGATIVE_INFINITY;
      for (let i = 0; i < d.length; i++) {
        const v = d[i];
        if (Number.isFinite(v) && v > acc) acc = v;
      }
      return acc === Number.NEGATIVE_INFINITY ? NaN : acc;
    }
    case "min": {
      let acc = Number.POSITIVE_INFINITY;
      for (let i = 0; i < d.length; i++) {
        const v = d[i];
        if (Number.isFinite(v) && v < acc) acc = v;
      }
      return acc === Number.POSITIVE_INFINITY ? NaN : acc;
    }
    case "mean": {
      let acc = 0;
      let n = 0;
      for (let i = 0; i < d.length; i++) {
        const v = d[i];
        if (Number.isFinite(v)) {
          acc += v;
          n += 1;
        }
      }
      return n > 0 ? acc / n : NaN;
    }
    case "rms": {
      let acc = 0;
      let n = 0;
      for (let i = 0; i < d.length; i++) {
        const v = d[i];
        if (Number.isFinite(v)) {
          acc += v * v;
          n += 1;
        }
      }
      return n > 0 ? Math.sqrt(acc / n) : NaN;
    }
  }
}

/**
 * Extract a 1D vector from a 3D (nr, nth, nz) field along `axis`, with the
 * other two axes fixed. fixedA/fixedB follow remaining axes in numeric order.
 */
export function profile1D(
  field: Npy3D,
  axis: SliceAxis,
  fixedA: number,
  fixedB: number,
): Float32Array {
  const [nr, nth, nz] = field.shape;
  const data = field.data;
  const sr = nth * nz;
  const sth = nz;

  switch (axis) {
    case 0: {
      // vary r, θ=fixedA, z=fixedB
      assertInRange(fixedA, nth, "θ");
      assertInRange(fixedB, nz, "z");
      const out = new Float32Array(nr);
      const offTh = fixedA * sth;
      for (let i = 0; i < nr; i++) {
        out[i] = data[i * sr + offTh + fixedB];
      }
      return out;
    }
    case 1: {
      // vary θ, r=fixedA, z=fixedB
      assertInRange(fixedA, nr, "r");
      assertInRange(fixedB, nz, "z");
      const out = new Float32Array(nth);
      const base = fixedA * sr;
      for (let j = 0; j < nth; j++) {
        out[j] = data[base + j * sth + fixedB];
      }
      return out;
    }
    case 2: {
      // vary z, r=fixedA, θ=fixedB
      assertInRange(fixedA, nr, "r");
      assertInRange(fixedB, nth, "θ");
      const out = new Float32Array(nz);
      const base = fixedA * sr + fixedB * sth;
      for (let k = 0; k < nz; k++) {
        out[k] = data[base + k];
      }
      return out;
    }
  }
}

function assertInRange(i: number, dim: number, name: string): void {
  if (i < 0 || i >= dim) {
    throw new Error(`Index ${i} out of range for ${name} (dim ${dim})`);
  }
}
