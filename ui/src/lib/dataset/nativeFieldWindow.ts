import { invokeCommand } from "@/lib/platform/appInvoke";
/**
 * Bounded native field-window/LOD protocol.
 *
 * This API is intentionally separate from `read_field_bytes`: callers opt in
 * to a strict output budget, explicit spatial bounds, and request-scoped
 * cancellation. The native response is a compact binary envelope so field
 * values never inflate into a JSON number array across Tauri IPC.
 */

const MAGIC = new Uint8Array([0x4e, 0x4d, 0x42, 0x4c, 0x4f, 0x44, 0x31, 0x00]); // NMBLOD1\0
const PREFIX_BYTES = MAGIC.byteLength + Uint32Array.BYTES_PER_ELEMENT;
const MAX_HEADER_BYTES = 64 * 1024;
export const MAX_NATIVE_FIELD_WINDOW_BYTES = 64 * 1024 * 1024;
export const MAX_NATIVE_FIELD_WINDOW_ENVELOPE_OVERHEAD_BYTES =
  PREFIX_BYTES + MAX_HEADER_BYTES;

/**
 * Ceiling on the bytes a `cell_mean` window is allowed to READ, as opposed to
 * the bytes it returns. Mirrors `MAX_FIELD_WINDOW_SOURCE_BYTES` in
 * `src-tauri/src/commands/outputs.rs`.
 *
 * This is a separate budget because striding does not reduce it. A cell-mean
 * aggregation visits every source cell inside `extent` in order to average the
 * blocks it returns, so a request can be small on the wire and enormous to
 * serve: a full-column window over a 512x512x160 grid returns 42 MB at stride 2
 * but must read all 168 MB to do it.
 *
 * Callers that plan a window against the output budget alone will therefore
 * build requests the backend rejects — which is exactly how the radar floor's
 * column-maximum reduction came to fail on any grid deeper than about 64 MB.
 * `maxSourceBytes` is left unset by default and the backend then applies a
 * budget a quarter of this size, so a caller reading deep columns must ask for
 * the headroom explicitly.
 */
export const MAX_NATIVE_FIELD_WINDOW_SOURCE_BYTES = 256 * 1024 * 1024;

/** Exact schema-v5 body bytes, including alignment before float64 geometry. */
export function nativeFieldWindowBodyBytes(
  elementCount: number,
  geometryBytesPerElement = 0,
): number {
  if (!Number.isSafeInteger(elementCount) || elementCount < 0
    || !Number.isSafeInteger(geometryBytesPerElement) || geometryBytesPerElement < 0) {
    throw new Error("Field-window body accounting requires nonnegative safe integers");
  }
  const fieldBytes = elementCount * Float32Array.BYTES_PER_ELEMENT;
  const padding = geometryBytesPerElement > 0 ? (8 - (fieldBytes % 8)) % 8 : 0;
  const total = fieldBytes + padding + elementCount * geometryBytesPerElement;
  if (!Number.isSafeInteger(total)) throw new Error("Field-window body byte accounting overflow");
  return total;
}

export type FieldWindowAggregation =
  | "cell_mean"
  | "linear_reflectivity_mean"
  | "block_max"
  | "block_min"
  | "magnitude_extreme"
  | "point_sample";

export interface NativeFieldWindowRequest {
  schemaVersion: 2;
  requestId: string;
  aggregation: FieldWindowAggregation;
  origin: number[];
  extent: number[];
  stride: number[];
  maxOutputBytes?: number;
  maxSourceBytes?: number;
}

export interface NativeFieldWindowHeader {
  schema: "nimbus.studio.field-window";
  schemaVersion: 4 | 5;
  requestId: string;
  aggregation: FieldWindowAggregation;
  weighting: "uniform_cell_volume" | "physical_cell_volume";
  missingDataPolicy: "propagate_non_finite";
  sampleLocation: "block_center" | "source_cell_center" | "physical_volume_centroid";
  geometryKind: "separable_uniform" | "terrain_following";
  geometryDtype: "<f8" | null;
  geometryLayout: "component_major_x_y_z_volume" | null;
  geometryComponents: ["x_m", "y_m", "z_m", "volume_m3"] | null;
  sourceShape: number[];
  origin: number[];
  extent: number[];
  stride: number[];
  outputShape: number[];
  elementCount: number;
  decodedBytes: number;
  geometryDecodedBytes: number;
  /** Schema v5 wire padding between float32 values and float64 geometry. */
  geometryPaddingBytes: number;
  sourceWorkingBytes: number;
  sourceWorkingAccounting: "inspected_payload_plus_explicit_working_set";
  dtype: "<f4";
  order: "C";
  sourceFormat: "npy" | "npy_lod" | "zfp3d" | "netcdf" | "netcdf_classic" | "netcdf4" | "netcdf4_classic";
  cancellationMode: "cooperative" | "boundary_only";
  nonInterruptiblePhase: "zfp_decode" | "nc_get_vars_float" | "nc_get_vars_float_and_double" | null;
}

export interface NativeFieldWindow {
  header: NativeFieldWindowHeader;
  data: Float32Array;
  /** Exact terrain block centroids. Absent for separable-uniform windows. */
  centroidsM: { x: Float64Array; y: Float64Array; z: Float64Array } | null;
  /** Exact physical volume represented by each terrain output block. */
  volumesM3: Float64Array | null;
}

export interface CancelFieldReadResult {
  requestId: string;
  found: boolean;
  cancellationRequested: boolean;
  phase: "registered" | "npy_header" | "npy_rows" | "zfp_decode" | "sampling" | null;
  interruptibility: "cooperative" | "boundary_only" | null;
}

export interface NativeFormatCapability {
  windowRead: boolean;
  cancellation: "cooperative" | "boundary_only" | "stale_result_only";
  limitation: string | null;
}

export interface NativeFieldReadCapabilities {
  schema: "nimbus.studio.native-field-read-capabilities";
  schemaVersion: 5;
  maxWindowBytes: number;
  maxSourceBytes: number;
  aggregations: readonly FieldWindowAggregation[];
  weighting: "uniform_or_physical_cell_volume";
  terrainFollowingWindowRead: true;
  terrainFollowingReductions: true;
  geometryLimitation: string;
  maxActiveRequests: number;
  npy: NativeFormatCapability;
  zfp3d: NativeFormatCapability;
  netcdfClassic: NativeFormatCapability;
  netcdf4: NativeFormatCapability;
}

function isPositiveSafeInteger(value: unknown): value is number {
  return Number.isSafeInteger(value) && (value as number) > 0;
}

function integerVector(value: unknown, label: string, positive: boolean): number[] {
  if (!Array.isArray(value) || value.length < 1 || value.length > 3) {
    throw new Error(`${label} must contain 1–3 integers`);
  }
  return value.map((entry) => {
    if (!Number.isSafeInteger(entry) || (positive ? entry <= 0 : entry < 0)) {
      throw new Error(`${label} contains an invalid integer`);
    }
    return entry;
  });
}

function validateHeader(value: unknown): NativeFieldWindowHeader {
  if (!value || typeof value !== "object") throw new Error("Field-window header must be an object");
  const raw = value as Record<string, unknown>;
  if (raw.schema !== "nimbus.studio.field-window"
    || (raw.schemaVersion !== 4 && raw.schemaVersion !== 5)) {
    throw new Error("Unsupported field-window response schema");
  }
  if (typeof raw.requestId !== "string" || !raw.requestId) throw new Error("Field-window response has no request ID");
  if (!["cell_mean", "linear_reflectivity_mean", "block_max", "block_min", "magnitude_extreme", "point_sample"].includes(raw.aggregation as string)
    || raw.missingDataPolicy !== "propagate_non_finite") {
    throw new Error("Unsupported field-window aggregation contract");
  }
  if ((raw.aggregation === "point_sample") !== (raw.sampleLocation === "source_cell_center")) {
    throw new Error("Field-window sampling location disagrees with its aggregation");
  }
  const uniformGeometry = raw.weighting === "uniform_cell_volume"
    && (raw.sampleLocation === "block_center" || raw.sampleLocation === "source_cell_center")
    && raw.geometryKind === "separable_uniform"
    && raw.geometryDtype === null
    && raw.geometryLayout === null
    && raw.geometryComponents === null;
  const terrainGeometry = raw.weighting === "physical_cell_volume"
    && (raw.sampleLocation === "physical_volume_centroid" || raw.sampleLocation === "source_cell_center")
    && raw.geometryKind === "terrain_following"
    && raw.geometryDtype === "<f8"
    && raw.geometryLayout === "component_major_x_y_z_volume"
    && Array.isArray(raw.geometryComponents)
    && raw.geometryComponents.length === 4
    && raw.geometryComponents[0] === "x_m"
    && raw.geometryComponents[1] === "y_m"
    && raw.geometryComponents[2] === "z_m"
    && raw.geometryComponents[3] === "volume_m3";
  if (!uniformGeometry && !terrainGeometry) throw new Error("Field-window geometry contract is inconsistent");
  const sourceShape = integerVector(raw.sourceShape, "sourceShape", true);
  const origin = integerVector(raw.origin, "origin", false);
  const extent = integerVector(raw.extent, "extent", true);
  const stride = integerVector(raw.stride, "stride", true);
  const outputShape = integerVector(raw.outputShape, "outputShape", true);
  const rank = sourceShape.length;
  if ([origin, extent, stride, outputShape].some((vector) => vector.length !== rank)) {
    throw new Error("Field-window response vectors disagree on rank");
  }
  for (let axis = 0; axis < rank; axis++) {
    if (origin[axis]! + extent[axis]! > sourceShape[axis]!) {
      throw new Error(`Field-window response exceeds source axis ${axis}`);
    }
    const expected = Math.floor((extent[axis]! - 1) / stride[axis]!) + 1;
    if (outputShape[axis] !== expected) {
      throw new Error(`Field-window output shape disagrees with axis ${axis} aggregation`);
    }
  }
  if (!isPositiveSafeInteger(raw.elementCount) || !isPositiveSafeInteger(raw.decodedBytes)) {
    throw new Error("Field-window response has invalid byte/count metadata");
  }
  const calculatedElements = outputShape.reduce((product, dimension) => product * dimension, 1);
  if (!Number.isSafeInteger(calculatedElements) || calculatedElements !== raw.elementCount) {
    throw new Error("Field-window element count does not match its output shape");
  }
  if (raw.decodedBytes !== calculatedElements * Float32Array.BYTES_PER_ELEMENT) {
    throw new Error("Field-window decoded byte count does not match float32 output");
  }
  const expectedGeometryBytes = terrainGeometry
    ? calculatedElements * 4 * Float64Array.BYTES_PER_ELEMENT
    : 0;
  if (!Number.isSafeInteger(raw.geometryDecodedBytes)
    || raw.geometryDecodedBytes !== expectedGeometryBytes) {
    throw new Error("Field-window geometry byte count does not match its contract");
  }
  const expectedGeometryPadding = raw.schemaVersion === 5 && expectedGeometryBytes > 0
    ? (8 - ((raw.decodedBytes as number) % 8)) % 8
    : 0;
  const geometryPaddingBytes = raw.schemaVersion === 4
    ? 0
    : raw.geometryPaddingBytes;
  if (!Number.isSafeInteger(geometryPaddingBytes)
    || geometryPaddingBytes !== expectedGeometryPadding) {
    throw new Error("Field-window geometry padding does not match its schema");
  }
  if (!isPositiveSafeInteger(raw.sourceWorkingBytes)
    || raw.sourceWorkingBytes > MAX_NATIVE_FIELD_WINDOW_SOURCE_BYTES
    || raw.sourceWorkingAccounting !== "inspected_payload_plus_explicit_working_set") {
    throw new Error("Field-window source/working byte accounting is invalid");
  }
  if ((raw.decodedBytes as number) + geometryPaddingBytes + expectedGeometryBytes > MAX_NATIVE_FIELD_WINDOW_BYTES) {
    throw new Error(`Field-window response exceeds ${MAX_NATIVE_FIELD_WINDOW_BYTES} bytes`);
  }
  if (raw.dtype !== "<f4" || raw.order !== "C") throw new Error("Unsupported field-window array encoding");
  const sourceFormats = ["npy", "npy_lod", "zfp3d", "netcdf", "netcdf_classic", "netcdf4", "netcdf4_classic"] as const;
  if (!sourceFormats.includes(raw.sourceFormat as typeof sourceFormats[number])) throw new Error("Unknown field-window source format");
  if (raw.cancellationMode !== "cooperative" && raw.cancellationMode !== "boundary_only") {
    throw new Error("Unknown field-window cancellation mode");
  }
  if (raw.nonInterruptiblePhase !== null
    && raw.nonInterruptiblePhase !== "zfp_decode"
    && raw.nonInterruptiblePhase !== "nc_get_vars_float"
    && raw.nonInterruptiblePhase !== "nc_get_vars_float_and_double") {
    throw new Error("Unknown field-window non-interruptible phase");
  }
  const cooperative = raw.sourceFormat === "npy" || raw.sourceFormat === "npy_lod"
    || raw.sourceFormat === "netcdf_classic";
  if (cooperative !== (raw.cancellationMode === "cooperative")) {
    throw new Error("Field-window source and cancellation metadata disagree");
  }
  if ((raw.sourceFormat === "zfp3d") !== (raw.nonInterruptiblePhase === "zfp_decode")) {
    throw new Error("Field-window source and blocked-phase metadata disagree");
  }
  const netcdfSource = typeof raw.sourceFormat === "string" && raw.sourceFormat.startsWith("netcdf");
  const blockingNetcdfSource = netcdfSource && raw.sourceFormat !== "netcdf_classic";
  const netcdfPhase = raw.nonInterruptiblePhase === "nc_get_vars_float"
    || raw.nonInterruptiblePhase === "nc_get_vars_float_and_double";
  if (blockingNetcdfSource !== netcdfPhase) {
    throw new Error("Field-window netCDF source and blocked-phase metadata disagree");
  }
  return {
    schema: raw.schema,
    schemaVersion: raw.schemaVersion,
    requestId: raw.requestId,
    aggregation: raw.aggregation as FieldWindowAggregation,
    weighting: raw.weighting as NativeFieldWindowHeader["weighting"],
    missingDataPolicy: raw.missingDataPolicy,
    sampleLocation: raw.sampleLocation as NativeFieldWindowHeader["sampleLocation"],
    geometryKind: raw.geometryKind as NativeFieldWindowHeader["geometryKind"],
    geometryDtype: raw.geometryDtype as NativeFieldWindowHeader["geometryDtype"],
    geometryLayout: raw.geometryLayout as NativeFieldWindowHeader["geometryLayout"],
    geometryComponents: raw.geometryComponents as NativeFieldWindowHeader["geometryComponents"],
    sourceShape,
    origin,
    extent,
    stride,
    outputShape,
    elementCount: raw.elementCount,
    decodedBytes: raw.decodedBytes,
    geometryDecodedBytes: raw.geometryDecodedBytes,
    geometryPaddingBytes,
    sourceWorkingBytes: raw.sourceWorkingBytes,
    sourceWorkingAccounting: raw.sourceWorkingAccounting,
    dtype: raw.dtype,
    order: raw.order,
    sourceFormat: raw.sourceFormat as NativeFieldWindowHeader["sourceFormat"],
    cancellationMode: raw.cancellationMode,
    nonInterruptiblePhase: raw.nonInterruptiblePhase,
  };
}

export function parseNativeFieldWindow(buffer: ArrayBuffer | Uint8Array): NativeFieldWindow {
  const bytes = buffer instanceof Uint8Array ? buffer : new Uint8Array(buffer);
  if (bytes.byteLength < PREFIX_BYTES) throw new Error("Truncated field-window response");
  for (let index = 0; index < MAGIC.byteLength; index++) {
    if (bytes[index] !== MAGIC[index]) throw new Error("Field-window response magic mismatch");
  }
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const headerBytes = view.getUint32(MAGIC.byteLength, true);
  if (headerBytes < 2 || headerBytes > MAX_HEADER_BYTES) throw new Error("Invalid field-window header length");
  const bodyOffset = PREFIX_BYTES + headerBytes;
  if (!Number.isSafeInteger(bodyOffset) || bodyOffset > bytes.byteLength) throw new Error("Truncated field-window header");
  let decoded: unknown;
  try {
    decoded = JSON.parse(new TextDecoder().decode(bytes.subarray(PREFIX_BYTES, bodyOffset)));
  } catch {
    throw new Error("Invalid field-window JSON header");
  }
  const header = validateHeader(decoded);
  if (bytes.byteLength - bodyOffset !== header.decodedBytes + header.geometryPaddingBytes + header.geometryDecodedBytes) {
    throw new Error("Field-window body size does not match metadata");
  }
  const absoluteOffset = bytes.byteOffset + bodyOffset;
  let data: Float32Array;
  if (absoluteOffset % Float32Array.BYTES_PER_ELEMENT === 0) {
    data = new Float32Array(bytes.buffer, absoluteOffset, header.elementCount);
  } else {
    const aligned = new Uint8Array(header.decodedBytes);
    aligned.set(bytes.subarray(bodyOffset, bodyOffset + header.decodedBytes));
    data = new Float32Array(aligned.buffer);
  }
  const geometryPaddingOffset = bodyOffset + header.decodedBytes;
  const geometryOffset = geometryPaddingOffset + header.geometryPaddingBytes;
  for (const value of bytes.subarray(geometryPaddingOffset, geometryOffset)) {
    if (value !== 0) throw new Error("Field-window geometry padding is not zero-filled");
  }
  let centroidsM: NativeFieldWindow["centroidsM"] = null;
  let volumesM3: NativeFieldWindow["volumesM3"] = null;
  if (header.geometryDecodedBytes > 0) {
    const geometryElements = header.elementCount * 4;
    const absoluteGeometryOffset = bytes.byteOffset + geometryOffset;
    let geometry: Float64Array;
    if (absoluteGeometryOffset % Float64Array.BYTES_PER_ELEMENT === 0) {
      geometry = new Float64Array(bytes.buffer, absoluteGeometryOffset, geometryElements);
    } else {
      const aligned = new Uint8Array(header.geometryDecodedBytes);
      aligned.set(bytes.subarray(geometryOffset, geometryOffset + header.geometryDecodedBytes));
      geometry = new Float64Array(aligned.buffer);
    }
    centroidsM = {
      x: geometry.subarray(0, header.elementCount),
      y: geometry.subarray(header.elementCount, 2 * header.elementCount),
      z: geometry.subarray(2 * header.elementCount, 3 * header.elementCount),
    };
    volumesM3 = geometry.subarray(3 * header.elementCount, 4 * header.elementCount);
  }
  return { header, data, centroidsM, volumesM3 };
}

export function isNativeFieldReadCancelled(error: unknown): boolean {
  const message = error instanceof Error ? error.message : String(error);
  return message.includes("field read cancelled:");
}

export async function cancelNativeFieldRead(requestId: string): Promise<CancelFieldReadResult> {
  return invokeCommand<CancelFieldReadResult>("cancel_field_read", { requestId });
}

export async function nativeFieldReadCapabilities(): Promise<NativeFieldReadCapabilities> {
  return invokeCommand<NativeFieldReadCapabilities>("native_field_read_capabilities");
}

export async function readNativeFieldWindow(
  input: {
    runDir: string;
    stepIndex: number;
    fieldName: string;
    expectedGeneration?: string;
    request: NativeFieldWindowRequest;
  },
  signal?: AbortSignal,
): Promise<NativeFieldWindow> {
  if (signal?.aborted) throw new DOMException("Field read aborted", "AbortError");
  const onAbort = () => {
    // Cancellation acknowledgement is intentionally not used as completion;
    // the read promise rejects only after the native worker observes the flag.
    void cancelNativeFieldRead(input.request.requestId).catch(() => undefined);
  };
  signal?.addEventListener("abort", onAbort, { once: true });
  try {
    const response = await invokeCommand<ArrayBuffer>("read_field_window", {
      runDir: input.runDir,
      stepIndex: input.stepIndex,
      fieldName: input.fieldName,
      expectedGeneration: input.expectedGeneration,
      request: input.request,
    });
    const parsed = parseNativeFieldWindow(response);
    if (parsed.header.requestId !== input.request.requestId) throw new Error("Field-window response request ID mismatch");
    if (signal?.aborted) throw new DOMException("Field read aborted", "AbortError");
    return parsed;
  } finally {
    signal?.removeEventListener("abort", onAbort);
  }
}

export async function readNativeFieldBrickWindow(
  input: {
    runDir: string;
    stepIndex: number;
    fieldName: string;
    expectedGeneration: string;
    level: number;
    request: NativeFieldWindowRequest;
  },
  signal?: AbortSignal,
): Promise<NativeFieldWindow> {
  if (!Number.isSafeInteger(input.level) || input.level <= 0) {
    throw new Error("Durable field-brick LOD level must be positive");
  }
  if (signal?.aborted) throw new DOMException("Field read aborted", "AbortError");
  const onAbort = () => {
    void cancelNativeFieldRead(input.request.requestId).catch(() => undefined);
  };
  signal?.addEventListener("abort", onAbort, { once: true });
  try {
    const response = await invokeCommand<ArrayBuffer>("read_field_brick_window", {
      runDir: input.runDir,
      stepIndex: input.stepIndex,
      fieldName: input.fieldName,
      expectedGeneration: input.expectedGeneration,
      level: input.level,
      request: input.request,
    });
    const parsed = parseNativeFieldWindow(response);
    if (parsed.header.requestId !== input.request.requestId) {
      throw new Error("Field-brick LOD response request ID mismatch");
    }
    if (parsed.header.sourceFormat !== "npy_lod") {
      throw new Error("Field-brick LOD response used the wrong source payload");
    }
    if (signal?.aborted) throw new DOMException("Field read aborted", "AbortError");
    return parsed;
  } finally {
    signal?.removeEventListener("abort", onAbort);
  }
}
