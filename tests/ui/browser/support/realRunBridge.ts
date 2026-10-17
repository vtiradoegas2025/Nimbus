import { readFileSync } from "node:fs";
import { join } from "node:path";
import type { IncomingMessage, ServerResponse } from "node:http";
import type { Plugin } from "vite";
import { parseNpy, type NpyN } from "../../../../ui/src/lib/platform/npy";

const MAGIC = new TextEncoder().encode("NMBLOD1\0");
const MAX_REQUEST_BYTES = 64 * 1024;
const MAX_OUTPUT_BYTES = 64 * 1024 * 1024;
const MAX_SOURCE_BYTES = 256 * 1024 * 1024;

type Aggregation = "cell_mean" | "linear_reflectivity_mean"
  | "block_max" | "block_min" | "magnitude_extreme" | "point_sample";

interface WindowRequest {
  requestId: string;
  origin: [number, number, number];
  extent: [number, number, number];
  stride: [number, number, number];
  aggregation: Aggregation;
  maxOutputBytes?: number;
  maxSourceBytes?: number;
}

interface BridgeRequest {
  stepIndex: number;
  fieldName: string;
  request: WindowRequest;
}

const BEAST_REFERENCE_STEPS = new Set([0, 1, 2, 3, 5, 23, 24]);
const BEAST_REFERENCE_FIELDS = new Set(["vorticity_magnitude", "theta", "theta_prime", "p", "p_prime", "qv", "buoyancy", "cloud_condensate", "reflectivity_dbz", "temperature", "u", "v", "w", "qr", "rho", "total_condensate", "vorticity_z"]);

function readBody(request: IncomingMessage): Promise<string> {
  return new Promise((resolve, reject) => {
    let body = "";
    request.setEncoding("utf8");
    request.on("data", (chunk: string) => {
      body += chunk;
      if (body.length > MAX_REQUEST_BYTES) reject(new Error("real-run bridge request is too large"));
    });
    request.on("end", () => resolve(body));
    request.on("error", reject);
  });
}

function validVector(value: unknown): value is [number, number, number] {
  return Array.isArray(value) && value.length === 3
    && value.every((item) => Number.isSafeInteger(item) && item >= 0);
}

function parseRequest(value: unknown): BridgeRequest {
  if (value == null || typeof value !== "object") throw new Error("real-run bridge request is not an object");
  const envelope = value as Record<string, unknown>;
  const request = envelope.request as Record<string, unknown> | null;
  if (!BEAST_REFERENCE_STEPS.has(Number(envelope.stepIndex))
      || !BEAST_REFERENCE_FIELDS.has(String(envelope.fieldName)) || !request) {
    throw new Error("real-run bridge accepts only maintained Beast XL volume fields and frames");
  }
  if (!validVector(request.origin) || !validVector(request.extent) || !validVector(request.stride)
      || request.stride.some((stride) => stride < 1)) {
    throw new Error("real-run bridge window vectors are invalid");
  }
  const aggregation = String(request.aggregation) as Aggregation;
  if (!["cell_mean", "linear_reflectivity_mean", "block_max", "block_min", "magnitude_extreme", "point_sample"].includes(aggregation)) {
    throw new Error(`real-run bridge aggregation '${aggregation}' is unsupported`);
  }
  return {
    stepIndex: Number(envelope.stepIndex),
    fieldName: String(envelope.fieldName),
    request: {
      requestId: String(request.requestId ?? ""),
      origin: request.origin,
      extent: request.extent,
      stride: request.stride,
      aggregation,
      maxOutputBytes: typeof request.maxOutputBytes === "number" ? request.maxOutputBytes : undefined,
      maxSourceBytes: typeof request.maxSourceBytes === "number" ? request.maxSourceBytes : undefined,
    },
  };
}

function encodeWindow(field: NpyN, request: WindowRequest): Uint8Array {
  if (field.shape.length !== 3) throw new Error("Beast reference field is not rank 3");
  const shape = field.shape as [number, number, number];
  for (let axis = 0; axis < 3; axis += 1) {
    if (request.origin[axis] + request.extent[axis] > shape[axis]) {
      throw new Error(`real-run bridge window exceeds source axis ${axis}`);
    }
  }
  const outputShape = request.extent.map((extent, axis) =>
    Math.ceil(extent / request.stride[axis])) as [number, number, number];
  const outputElements = outputShape[0] * outputShape[1] * outputShape[2];
  const decodedBytes = outputElements * Float32Array.BYTES_PER_ELEMENT;
  const sourceElements = request.aggregation === "point_sample"
    ? outputShape[0] * outputShape[1] * request.extent[2]
    : request.extent[0] * request.extent[1] * request.extent[2];
  const sourceWorkingBytes = sourceElements * Float32Array.BYTES_PER_ELEMENT;
  const outputBudget = request.maxOutputBytes ?? 32 * 1024 * 1024;
  const sourceBudget = request.maxSourceBytes ?? 64 * 1024 * 1024;
  if (outputBudget < 1 || outputBudget > MAX_OUTPUT_BYTES || decodedBytes > outputBudget) {
    throw new Error(`real-run window output needs ${decodedBytes} bytes; budget is ${outputBudget}`);
  }
  if (sourceBudget < 1 || sourceBudget > MAX_SOURCE_BYTES || sourceWorkingBytes > sourceBudget) {
    throw new Error(`real-run window source needs ${sourceWorkingBytes} bytes; budget is ${sourceBudget}`);
  }

  const output = new Float32Array(outputElements);
  const [ny, nz] = [shape[1], shape[2]];
  let destination = 0;
  for (let ox = 0; ox < outputShape[0]; ox += 1) {
    const x0 = request.origin[0] + ox * request.stride[0];
    const x1 = Math.min(request.origin[0] + request.extent[0], x0 + request.stride[0]);
    for (let oy = 0; oy < outputShape[1]; oy += 1) {
      const y0 = request.origin[1] + oy * request.stride[1];
      const y1 = Math.min(request.origin[1] + request.extent[1], y0 + request.stride[1]);
      for (let oz = 0; oz < outputShape[2]; oz += 1) {
        const z0 = request.origin[2] + oz * request.stride[2];
        const z1 = Math.min(request.origin[2] + request.extent[2], z0 + request.stride[2]);
        if (request.aggregation === "point_sample") {
          output[destination++] = field.data[(x0 * ny + y0) * nz + z0];
          continue;
        }
        if (request.stride[0] === 1 && request.stride[1] === 1 && request.stride[2] === 1) {
          output[destination++] = field.data[(x0 * ny + y0) * nz + z0];
          continue;
        }
        let count = 0;
        let sum = 0;
        let selected = request.aggregation === "block_min"
          ? Number.POSITIVE_INFINITY
          : Number.NEGATIVE_INFINITY;
        let invalid = false;
        for (let x = x0; x < x1; x += 1) {
          for (let y = y0; y < y1; y += 1) {
            const column = (x * ny + y) * nz;
            for (let z = z0; z < z1; z += 1) {
              const value = field.data[column + z];
              if (!Number.isFinite(value)) {
                invalid = true;
                continue;
              }
              count += 1;
              if (request.aggregation === "block_max") selected = Math.max(selected, value);
              else if (request.aggregation === "block_min") selected = Math.min(selected, value);
              else if (request.aggregation === "magnitude_extreme") {
                if (selected === Number.NEGATIVE_INFINITY || Math.abs(value) > Math.abs(selected)) {
                  selected = value;
                }
              } else if (request.aggregation === "linear_reflectivity_mean") {
                sum += 10 ** (value / 10);
              } else sum += value;
            }
          }
        }
        output[destination++] = invalid || count === 0
          ? Number.NaN
          : request.aggregation === "block_max"
              || request.aggregation === "block_min"
              || request.aggregation === "magnitude_extreme"
            ? selected
            : request.aggregation === "linear_reflectivity_mean"
              ? 10 * Math.log10(sum / count)
              : sum / count;
      }
    }
  }

  const header = {
    schema: "nimbus.studio.field-window",
    schemaVersion: 4,
    requestId: request.requestId,
    aggregation: request.aggregation,
    weighting: "uniform_cell_volume",
    missingDataPolicy: "propagate_non_finite",
    sampleLocation: request.aggregation === "point_sample" ? "source_cell_center" : "block_center",
    geometryKind: "separable_uniform",
    geometryDtype: null,
    geometryLayout: null,
    geometryComponents: null,
    sourceShape: shape,
    origin: request.origin,
    extent: request.extent,
    stride: request.stride,
    outputShape,
    elementCount: output.length,
    decodedBytes: output.byteLength,
    geometryDecodedBytes: 0,
    sourceWorkingBytes,
    sourceWorkingAccounting: "inspected_payload_plus_explicit_working_set",
    dtype: "<f4",
    order: "C",
    sourceFormat: "npy",
    cancellationMode: "cooperative",
    nonInterruptiblePhase: null,
  };
  const json = new TextEncoder().encode(JSON.stringify(header));
  const encoded = new Uint8Array(MAGIC.byteLength + 4 + json.byteLength + output.byteLength);
  encoded.set(MAGIC, 0);
  new DataView(encoded.buffer).setUint32(MAGIC.byteLength, json.byteLength, true);
  encoded.set(json, MAGIC.byteLength + 4);
  encoded.set(new Uint8Array(output.buffer), MAGIC.byteLength + 4 + json.byteLength);
  return encoded;
}

export function beastReferenceBridgePlugin(runDir: string | undefined): Plugin | null {
  if (!runDir) return null;
  let field: { stepIndex: number; fieldName: string; value: NpyN } | null = null;
  const handle = async (request: IncomingMessage, response: ServerResponse) => {
    try {
      const parsed = parseRequest(JSON.parse(await readBody(request)));
      if (field?.stepIndex !== parsed.stepIndex || field?.fieldName !== parsed.fieldName) {
        const read = (name: string) => parseNpy(readFileSync(join(
          runDir, `step_${String(parsed.stepIndex).padStart(6, "0")}`, `${name}.npy`,
        )));
        const value = read(parsed.fieldName);
        field = {
          stepIndex: parsed.stepIndex,
          fieldName: parsed.fieldName,
          value,
        };
      }
      const encoded = encodeWindow(field.value, parsed.request);
      response.writeHead(200, {
        "content-type": "application/octet-stream",
        "content-length": encoded.byteLength,
        "cache-control": "no-store",
      });
      response.end(encoded);
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      response.writeHead(400, { "content-type": "text/plain; charset=utf-8" });
      response.end(message);
    }
  };
  return {
    name: "nimbus-beast-reference-bridge",
    configureServer(server) {
      server.middlewares.use("/__nimbus_real_run/field-window", handle);
    },
    configurePreviewServer(server) {
      server.middlewares.use("/__nimbus_real_run/field-window", handle);
    },
  };
}
