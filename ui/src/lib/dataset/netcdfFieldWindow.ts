import { invokeCommand } from "@/lib/platform/appInvoke";
import type { FieldWindowAggregation } from "@/lib/dataset/nativeFieldWindow";
/**
 * Bounded native netCDF hyperslab protocol.
 *
 * Spatial vectors are always canonical x/y(/z); `recordIndex` selects time.
 * `extent` is the source span and `stride` determines returned samples.
 */

const MAGIC = new Uint8Array([0x4e, 0x4d, 0x42, 0x4e, 0x43, 0x57, 0x31, 0x00]); // NMBNCW1\0
const PREFIX_BYTES = MAGIC.byteLength + Uint32Array.BYTES_PER_ELEMENT;
const MAX_HEADER_BYTES = 64 * 1024;
export const MAX_NETCDF_FIELD_WINDOW_BYTES = 64 * 1024 * 1024;
export const MAX_NETCDF_FIELD_SOURCE_BYTES = 256 * 1024 * 1024;

export interface NetcdfFieldWindowRequest {
  schemaVersion: 2;
  requestId: string;
  aggregation: FieldWindowAggregation;
  origin: number[];
  extent: number[];
  stride: number[];
  maxOutputBytes?: number;
  maxOutputSamples?: number;
  maxSourceBytes?: number;
}

export type NetcdfSourceFormat =
  | "netcdf_classic"
  | "netcdf_64bit_offset"
  | "netcdf4"
  | "netcdf4_classic"
  | "netcdf_64bit_data"
  | "netcdf_unknown";

export interface NetcdfFieldWindowHeader {
  schema: "nimbus.studio.netcdf-window";
  schemaVersion: 2;
  requestId: string;
  aggregation: FieldWindowAggregation;
  weighting: "uniform_cell_volume";
  missingDataPolicy: "propagate_non_finite";
  sampleLocation: "block_center" | "source_cell_center";
  sourceShape: number[];
  origin: number[];
  extent: number[];
  stride: number[];
  outputShape: number[];
  elementCount: number;
  decodedBytes: number;
  sourceReadBytes: number;
  sourceReadAccounting: "file_payload_exact" | "converted_hyperslab_output";
  nativeInternalAllocationBounded: boolean;
  dtype: "<f4";
  order: "C";
  sourceFormat: NetcdfSourceFormat;
  cancellationMode: "cooperative" | "boundary_only";
  nonInterruptiblePhase: "nc_get_vars_float" | null;
}

export interface NetcdfFieldWindow {
  header: NetcdfFieldWindowHeader;
  data: Float32Array;
}

export interface CancelNetcdfFieldReadResult {
  requestId: string;
  found: boolean;
  cancellationRequested: boolean;
  phase: "registered" | "header" | "classic_rows" | "native_call" | "transpose" | "encode" | null;
  interruptibility: "cooperative" | "boundary_only" | null;
}

function positiveSafeInteger(value: unknown): value is number {
  return Number.isSafeInteger(value) && (value as number) > 0;
}

function integerVector(value: unknown, label: string, positive: boolean): number[] {
  if (!Array.isArray(value) || value.length < 2 || value.length > 3) {
    throw new Error(`${label} must contain 2 or 3 integers`);
  }
  return value.map((entry) => {
    if (!Number.isSafeInteger(entry) || (positive ? entry <= 0 : entry < 0)) {
      throw new Error(`${label} contains an invalid integer`);
    }
    return entry;
  });
}

function validateHeader(value: unknown): NetcdfFieldWindowHeader {
  if (!value || typeof value !== "object") throw new Error("netCDF-window header must be an object");
  const raw = value as Record<string, unknown>;
  if (raw.schema !== "nimbus.studio.netcdf-window" || raw.schemaVersion !== 2) {
    throw new Error("Unsupported netCDF-window response schema");
  }
  if (typeof raw.requestId !== "string" || !raw.requestId) {
    throw new Error("netCDF-window response has no request ID");
  }
  if (!["cell_mean", "linear_reflectivity_mean", "block_max", "block_min", "magnitude_extreme", "point_sample"].includes(raw.aggregation as string)
    || raw.weighting !== "uniform_cell_volume"
    || raw.missingDataPolicy !== "propagate_non_finite"
    || (raw.sampleLocation !== "block_center" && raw.sampleLocation !== "source_cell_center")) {
    throw new Error("Unsupported netCDF-window aggregation contract");
  }
  if ((raw.aggregation === "point_sample") !== (raw.sampleLocation === "source_cell_center")) {
    throw new Error("netCDF-window sampling location disagrees with its aggregation");
  }
  const sourceShape = integerVector(raw.sourceShape, "sourceShape", true);
  const origin = integerVector(raw.origin, "origin", false);
  const extent = integerVector(raw.extent, "extent", true);
  const stride = integerVector(raw.stride, "stride", true);
  const outputShape = integerVector(raw.outputShape, "outputShape", true);
  const rank = sourceShape.length;
  if ([origin, extent, stride, outputShape].some((vector) => vector.length !== rank)) {
    throw new Error("netCDF-window response vectors disagree on rank");
  }
  for (let axis = 0; axis < rank; axis++) {
    if (origin[axis] + extent[axis] > sourceShape[axis]) {
      throw new Error(`netCDF-window response exceeds source axis ${axis}`);
    }
    const expected = Math.floor((extent[axis] - 1) / stride[axis]) + 1;
    if (outputShape[axis] !== expected) {
      throw new Error(`netCDF-window output shape disagrees with axis ${axis} sampling`);
    }
  }
  if (!positiveSafeInteger(raw.elementCount) || !positiveSafeInteger(raw.decodedBytes)) {
    throw new Error("netCDF-window response has invalid byte/count metadata");
  }
  const calculatedElements = outputShape.reduce((product, dimension) => product * dimension, 1);
  if (!Number.isSafeInteger(calculatedElements) || calculatedElements !== raw.elementCount) {
    throw new Error("netCDF-window element count does not match output shape");
  }
  if (raw.decodedBytes !== calculatedElements * Float32Array.BYTES_PER_ELEMENT) {
    throw new Error("netCDF-window decoded byte count does not match float32 output");
  }
  if ((raw.decodedBytes as number) > MAX_NETCDF_FIELD_WINDOW_BYTES) {
    throw new Error(`netCDF-window response exceeds ${MAX_NETCDF_FIELD_WINDOW_BYTES} bytes`);
  }
  if (!positiveSafeInteger(raw.sourceReadBytes) || (raw.sourceReadBytes as number) > MAX_NETCDF_FIELD_SOURCE_BYTES) {
    throw new Error("netCDF-window response has invalid source-read byte metadata");
  }
  if (raw.sourceReadAccounting !== "file_payload_exact" && raw.sourceReadAccounting !== "converted_hyperslab_output") {
    throw new Error("netCDF-window response has unknown source-read accounting");
  }
  if (typeof raw.nativeInternalAllocationBounded !== "boolean") {
    throw new Error("netCDF-window response has invalid native-allocation metadata");
  }
  if (raw.dtype !== "<f4" || raw.order !== "C") {
    throw new Error("Unsupported netCDF-window array encoding");
  }
  const sourceFormats: NetcdfSourceFormat[] = [
    "netcdf_classic",
    "netcdf_64bit_offset",
    "netcdf4",
    "netcdf4_classic",
    "netcdf_64bit_data",
    "netcdf_unknown",
  ];
  if (!sourceFormats.includes(raw.sourceFormat as NetcdfSourceFormat)) {
    throw new Error("Unknown netCDF-window source format");
  }
  if (raw.cancellationMode !== "cooperative" && raw.cancellationMode !== "boundary_only") {
    throw new Error("Unknown netCDF-window cancellation mode");
  }
  if (raw.nonInterruptiblePhase !== null && raw.nonInterruptiblePhase !== "nc_get_vars_float") {
    throw new Error("Unknown netCDF-window non-interruptible phase");
  }
  const classic = raw.sourceFormat === "netcdf_classic" || raw.sourceFormat === "netcdf_64bit_offset";
  if (classic !== (raw.cancellationMode === "cooperative") || classic !== (raw.nonInterruptiblePhase === null)) {
    throw new Error("netCDF-window source and cancellation metadata disagree");
  }
  if (classic !== (raw.sourceReadAccounting === "file_payload_exact") || classic !== raw.nativeInternalAllocationBounded) {
    throw new Error("netCDF-window source and allocation metadata disagree");
  }
  return {
    schema: raw.schema,
    schemaVersion: raw.schemaVersion,
    requestId: raw.requestId,
    aggregation: raw.aggregation as FieldWindowAggregation,
    weighting: raw.weighting,
    missingDataPolicy: raw.missingDataPolicy,
    sampleLocation: raw.sampleLocation,
    sourceShape,
    origin,
    extent,
    stride,
    outputShape,
    elementCount: raw.elementCount,
    decodedBytes: raw.decodedBytes,
    sourceReadBytes: raw.sourceReadBytes,
    sourceReadAccounting: raw.sourceReadAccounting,
    nativeInternalAllocationBounded: raw.nativeInternalAllocationBounded,
    dtype: raw.dtype,
    order: raw.order,
    sourceFormat: raw.sourceFormat as NetcdfSourceFormat,
    cancellationMode: raw.cancellationMode,
    nonInterruptiblePhase: raw.nonInterruptiblePhase,
  };
}

export function parseNetcdfFieldWindow(buffer: ArrayBuffer | Uint8Array): NetcdfFieldWindow {
  const bytes = buffer instanceof Uint8Array ? buffer : new Uint8Array(buffer);
  if (bytes.byteLength < PREFIX_BYTES) throw new Error("Truncated netCDF-window response");
  for (let index = 0; index < MAGIC.byteLength; index++) {
    if (bytes[index] !== MAGIC[index]) throw new Error("netCDF-window response magic mismatch");
  }
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const headerBytes = view.getUint32(MAGIC.byteLength, true);
  if (headerBytes < 2 || headerBytes > MAX_HEADER_BYTES) {
    throw new Error("Invalid netCDF-window header length");
  }
  const bodyOffset = PREFIX_BYTES + headerBytes;
  if (!Number.isSafeInteger(bodyOffset) || bodyOffset > bytes.byteLength) {
    throw new Error("Truncated netCDF-window header");
  }
  let decoded: unknown;
  try {
    decoded = JSON.parse(new TextDecoder().decode(bytes.subarray(PREFIX_BYTES, bodyOffset)));
  } catch {
    throw new Error("Invalid netCDF-window JSON header");
  }
  const header = validateHeader(decoded);
  if (bytes.byteLength - bodyOffset !== header.decodedBytes) {
    throw new Error("netCDF-window body size does not match metadata");
  }
  const absoluteOffset = bytes.byteOffset + bodyOffset;
  let data: Float32Array;
  if (absoluteOffset % Float32Array.BYTES_PER_ELEMENT === 0) {
    data = new Float32Array(bytes.buffer, absoluteOffset, header.elementCount);
  } else {
    const aligned = new Uint8Array(header.decodedBytes);
    aligned.set(bytes.subarray(bodyOffset));
    data = new Float32Array(aligned.buffer);
  }
  return { header, data };
}

export function isNetcdfFieldReadCancelled(error: unknown): boolean {
  const message = error instanceof Error ? error.message : String(error);
  return message.includes("netCDF field read cancelled:");
}

export async function cancelNetcdfFieldRead(requestId: string): Promise<CancelNetcdfFieldReadResult> {
  return invokeCommand<CancelNetcdfFieldReadResult>("cancel_netcdf_field_read", { requestId });
}

export async function readNetcdfFieldWindow(
  input: {
    path: string;
    variable: string;
    recordIndex: number;
    request: NetcdfFieldWindowRequest;
  },
  signal?: AbortSignal,
): Promise<NetcdfFieldWindow> {
  if (signal?.aborted) throw new DOMException("netCDF field read aborted", "AbortError");
  const onAbort = () => {
    // Receipt acknowledges the flag only. The read promise settles when the
    // worker observes it, potentially after a blocking libnetcdf call.
    void cancelNetcdfFieldRead(input.request.requestId).catch(() => undefined);
  };
  signal?.addEventListener("abort", onAbort, { once: true });
  try {
    const response = await invokeCommand<ArrayBuffer>("read_netcdf_field_window", input);
    const parsed = parseNetcdfFieldWindow(response);
    if (parsed.header.requestId !== input.request.requestId) {
      throw new Error("netCDF-window response request ID mismatch");
    }
    if (signal?.aborted) throw new DOMException("netCDF field read aborted", "AbortError");
    return parsed;
  } finally {
    signal?.removeEventListener("abort", onAbort);
  }
}
