import assert from "node:assert/strict";
import { parseNetcdfFieldWindow } from "../../../../ui/src/lib/dataset/netcdfFieldWindow.ts";

function envelope(header: Record<string, unknown>, values: number[]): Uint8Array {
  const json = new TextEncoder().encode(JSON.stringify(header));
  const bytes = new Uint8Array(12 + json.byteLength + values.length * 4);
  bytes.set([0x4e, 0x4d, 0x42, 0x4e, 0x43, 0x57, 0x31, 0], 0);
  new DataView(bytes.buffer).setUint32(8, json.byteLength, true);
  bytes.set(json, 12);
  const body = new DataView(bytes.buffer, 12 + json.byteLength);
  values.forEach((value, index) => body.setFloat32(index * 4, value, true));
  return bytes;
}

function alignedEnvelope(header: Record<string, unknown>, values: number[]): Uint8Array {
  const raw = JSON.stringify(header);
  const padding = (8 - ((12 + new TextEncoder().encode(raw).byteLength) % 8)) % 8;
  const json = new TextEncoder().encode(raw + " ".repeat(padding));
  const bytes = new Uint8Array(12 + json.byteLength + values.length * 4);
  bytes.set([0x4e, 0x4d, 0x42, 0x4e, 0x43, 0x57, 0x31, 0], 0);
  new DataView(bytes.buffer).setUint32(8, json.byteLength, true);
  bytes.set(json, 12);
  const body = new DataView(bytes.buffer, 12 + json.byteLength);
  values.forEach((value, index) => body.setFloat32(index * 4, value, true));
  return bytes;
}

const header = {
  schema: "nimbus.studio.netcdf-window",
  schemaVersion: 2,
  requestId: "cm1:w:17",
  aggregation: "cell_mean",
  weighting: "uniform_cell_volume",
  missingDataPolicy: "propagate_non_finite",
  sampleLocation: "block_center",
  sourceShape: [8, 6, 4],
  origin: [1, 0, 0],
  extent: [6, 6, 4],
  stride: [2, 3, 2],
  outputShape: [3, 2, 2],
  elementCount: 12,
  decodedBytes: 48,
  sourceReadBytes: 48,
  sourceReadAccounting: "converted_hyperslab_output",
  nativeInternalAllocationBounded: false,
  dtype: "<f4",
  order: "C",
  sourceFormat: "netcdf4",
  cancellationMode: "boundary_only",
  nonInterruptiblePhase: "nc_get_vars_float",
};

const values = Array.from({ length: 12 }, (_, index) => index + 0.25);
const parsed = parseNetcdfFieldWindow(envelope(header, values));
assert.deepEqual(parsed.header.outputShape, [3, 2, 2]);
assert.deepEqual(Array.from(parsed.data), values);
const aligned = alignedEnvelope(header, values);
assert.equal(
  parseNetcdfFieldWindow(aligned).data.buffer,
  aligned.buffer,
  "aligned netCDF field data is a zero-copy IPC view",
);
assert.equal(
  parseNetcdfFieldWindow(envelope(
    { ...header, aggregation: "linear_reflectivity_mean" },
    values,
  )).header.aggregation,
  "linear_reflectivity_mean",
);
for (const aggregation of ["block_max", "block_min", "magnitude_extreme"] as const) {
  assert.equal(
    parseNetcdfFieldWindow(envelope({ ...header, aggregation }, values)).header.aggregation,
    aggregation,
  );
}
assert.equal(
  parseNetcdfFieldWindow(envelope({
    ...header,
    aggregation: "point_sample",
    sampleLocation: "source_cell_center",
  }, values)).header.sampleLocation,
  "source_cell_center",
);
assert.throws(
  () => parseNetcdfFieldWindow(envelope({ ...header, aggregation: "point_sample" }, values)),
  /sampling location/,
);
assert.throws(
  () => parseNetcdfFieldWindow(envelope({ ...header, aggregation: "dbz_mean" }, values)),
  /aggregation contract/,
);

const classic = parseNetcdfFieldWindow(envelope({
  ...header,
  sourceFormat: "netcdf_classic",
  cancellationMode: "cooperative",
  nonInterruptiblePhase: null,
  sourceReadAccounting: "file_payload_exact",
  nativeInternalAllocationBounded: true,
}, values));
assert.equal(classic.header.cancellationMode, "cooperative");

assert.throws(
  () => parseNetcdfFieldWindow(envelope({ ...header, outputShape: [4, 2, 2] }, values)),
  /output shape disagrees/,
);
assert.throws(
  () => parseNetcdfFieldWindow(envelope({ ...header, cancellationMode: "cooperative" }, values)),
  /source and cancellation metadata disagree/,
);
assert.throws(
  () => parseNetcdfFieldWindow(envelope({ ...header, sourceReadBytes: 300 * 1024 * 1024 }, values)),
  /source-read byte metadata/,
);
assert.throws(
  () => parseNetcdfFieldWindow(envelope({ ...header, decodedBytes: 44 }, values)),
  /decoded byte count/,
);
assert.throws(
  () => parseNetcdfFieldWindow(envelope(header, values).subarray(0, -1)),
  /body size/,
);

console.log("netCDF field-window contract tests passed");
