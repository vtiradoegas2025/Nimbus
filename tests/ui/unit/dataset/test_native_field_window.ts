import assert from "node:assert/strict";
import {
  nativeFieldWindowBodyBytes,
  parseNativeFieldWindow,
} from "../../../../ui/src/lib/dataset/nativeFieldWindow.ts";

assert.equal(nativeFieldWindowBodyBytes(1, 32), 40, "one terrain sample admits four alignment bytes");
assert.equal(nativeFieldWindowBodyBytes(2, 32), 72, "even terrain sample counts need no alignment bytes");

function envelope(header: Record<string, unknown>, values: number[], geometry: number[] = []): Uint8Array {
  const json = new TextEncoder().encode(JSON.stringify(header));
  const bytes = new Uint8Array(12 + json.byteLength + values.length * 4 + geometry.length * 8);
  bytes.set([0x4e, 0x4d, 0x42, 0x4c, 0x4f, 0x44, 0x31, 0], 0);
  new DataView(bytes.buffer).setUint32(8, json.byteLength, true);
  bytes.set(json, 12);
  const body = new DataView(bytes.buffer, 12 + json.byteLength);
  values.forEach((value, index) => body.setFloat32(index * 4, value, true));
  geometry.forEach((value, index) => body.setFloat64(values.length * 4 + index * 8, value, true));
  return bytes;
}

function alignedEnvelope(header: Record<string, unknown>, values: number[], geometry: number[] = []): Uint8Array {
  const raw = JSON.stringify(header);
  const padding = (8 - ((12 + new TextEncoder().encode(raw).byteLength) % 8)) % 8;
  const json = new TextEncoder().encode(raw + " ".repeat(padding));
  const geometryPaddingBytes = Number(header.geometryPaddingBytes ?? 0);
  const bytes = new Uint8Array(12 + json.byteLength + values.length * 4 + geometryPaddingBytes + geometry.length * 8);
  bytes.set([0x4e, 0x4d, 0x42, 0x4c, 0x4f, 0x44, 0x31, 0], 0);
  new DataView(bytes.buffer).setUint32(8, json.byteLength, true);
  bytes.set(json, 12);
  const body = new DataView(bytes.buffer, 12 + json.byteLength);
  values.forEach((value, index) => body.setFloat32(index * 4, value, true));
  geometry.forEach((value, index) => body.setFloat64(values.length * 4 + geometryPaddingBytes + index * 8, value, true));
  return bytes;
}

const header = {
  schema: "nimbus.studio.field-window",
  schemaVersion: 4,
  requestId: "plot:theta:7",
  aggregation: "cell_mean",
  weighting: "uniform_cell_volume",
  missingDataPolicy: "propagate_non_finite",
  sampleLocation: "block_center",
  geometryKind: "separable_uniform",
  geometryDtype: null,
  geometryLayout: null,
  geometryComponents: null,
  sourceShape: [8, 6, 4],
  origin: [1, 0, 0],
  extent: [6, 6, 4],
  stride: [2, 3, 2],
  outputShape: [3, 2, 2],
  elementCount: 12,
  decodedBytes: 48,
  geometryDecodedBytes: 0,
  sourceWorkingBytes: 576,
  sourceWorkingAccounting: "inspected_payload_plus_explicit_working_set",
  dtype: "<f4",
  order: "C",
  sourceFormat: "npy",
  cancellationMode: "cooperative",
  nonInterruptiblePhase: null,
};

const parsed = parseNativeFieldWindow(envelope(header, Array.from({ length: 12 }, (_, i) => i + 0.5)));
assert.deepEqual(parsed.header.outputShape, [3, 2, 2]);
assert.deepEqual(Array.from(parsed.data), Array.from({ length: 12 }, (_, i) => i + 0.5));
assert.equal(parsed.centroidsM, null);
assert.equal(parsed.volumesM3, null);
const aligned = alignedEnvelope(header, Array.from({ length: 12 }, (_, i) => i + 0.5));
const alignedParsed = parseNativeFieldWindow(aligned);
assert.equal(alignedParsed.data.buffer, aligned.buffer, "aligned native field data is a zero-copy IPC view");
assert.equal(
  parseNativeFieldWindow(envelope(
    { ...header, aggregation: "linear_reflectivity_mean" },
    Array.from({ length: 12 }, (_, i) => i),
  )).header.aggregation,
  "linear_reflectivity_mean",
);
const classicNetcdf = parseNativeFieldWindow(envelope({
  ...header,
  sourceFormat: "netcdf_classic",
  cancellationMode: "cooperative",
  nonInterruptiblePhase: null,
}, Array.from({ length: 12 }, (_, i) => i)));
assert.equal(
  classicNetcdf.header.cancellationMode,
  "cooperative",
  "classic netCDF is a valid cooperative native-window source",
);
for (const aggregation of ["block_max", "block_min", "magnitude_extreme"] as const) {
  assert.equal(
    parseNativeFieldWindow(envelope({ ...header, aggregation }, Array(12).fill(1))).header.aggregation,
    aggregation,
  );
}
assert.equal(
  parseNativeFieldWindow(envelope({
    ...header,
    aggregation: "point_sample",
    sampleLocation: "source_cell_center",
    sourceWorkingBytes: 48,
  }, Array(12).fill(1))).header.sampleLocation,
  "source_cell_center",
);
assert.throws(
  () => parseNativeFieldWindow(envelope({ ...header, aggregation: "point_sample" }, Array(12).fill(1))),
  /sampling location/,
);
assert.throws(
  () => parseNativeFieldWindow(envelope({ ...header, aggregation: "dbz_mean" }, Array(12).fill(0))),
  /aggregation contract/,
);

const terrainHeader = {
  ...header,
  requestId: "plot:terrain:8",
  extent: [2, 1, 2],
  stride: [2, 1, 2],
  outputShape: [1, 1, 1],
  elementCount: 1,
  decodedBytes: 4,
  weighting: "physical_cell_volume",
  sampleLocation: "physical_volume_centroid",
  geometryKind: "terrain_following",
  geometryDtype: "<f8",
  geometryLayout: "component_major_x_y_z_volume",
  geometryComponents: ["x_m", "y_m", "z_m", "volume_m3"],
  geometryDecodedBytes: 32,
};
const terrain = parseNativeFieldWindow(envelope(terrainHeader, [30], [525, 450, 26, 7_000_000]));
assert.deepEqual(Array.from(terrain.data), [30]);
assert.deepEqual(Array.from(terrain.centroidsM!.x), [525]);
assert.deepEqual(Array.from(terrain.centroidsM!.y), [450]);
assert.deepEqual(Array.from(terrain.centroidsM!.z), [26]);
assert.deepEqual(Array.from(terrain.volumesM3!), [7_000_000]);
const alignedTerrainEnvelope = alignedEnvelope(
  { ...terrainHeader, schemaVersion: 5, geometryPaddingBytes: 4 },
  [30],
  [525, 450, 26, 7_000_000],
);
const alignedTerrain = parseNativeFieldWindow(alignedTerrainEnvelope);
assert.equal(alignedTerrain.data.buffer, alignedTerrainEnvelope.buffer, "aligned terrain values share the IPC buffer");
assert.equal(alignedTerrain.centroidsM!.x.buffer, alignedTerrainEnvelope.buffer, "aligned float64 geometry shares the IPC buffer");
const corruptPadding = alignedTerrainEnvelope.slice();
const corruptHeaderBytes = new DataView(corruptPadding.buffer).getUint32(8, true);
corruptPadding[12 + corruptHeaderBytes + Float32Array.BYTES_PER_ELEMENT] = 1;
assert.throws(
  () => parseNativeFieldWindow(corruptPadding),
  /padding is not zero-filled/,
  "schema-v5 geometry padding is validated rather than ignored",
);
const terrainPoint = parseNativeFieldWindow(alignedEnvelope({
  ...terrainHeader,
  schemaVersion: 5,
  aggregation: "point_sample",
  sampleLocation: "source_cell_center",
  geometryPaddingBytes: 4,
}, [30], [375, 450, 10, 1_500_000]));
assert.equal(terrainPoint.header.sampleLocation, "source_cell_center", "terrain point samples retain exact source-cell geometry");
assert.throws(
  () => parseNativeFieldWindow(alignedEnvelope({
    ...terrainHeader, schemaVersion: 5, geometryPaddingBytes: 0,
  }, [30], [525, 450, 26, 7_000_000])),
  /geometry padding/,
);

assert.throws(() => parseNativeFieldWindow(envelope({ ...header, decodedBytes: 44 }, Array(12).fill(0))), /decoded byte count/);
assert.throws(() => parseNativeFieldWindow(envelope({ ...header, cancellationMode: "boundary_only" }, Array(12).fill(0))), /disagree/);
assert.throws(
  () => parseNativeFieldWindow(envelope({ ...terrainHeader, geometryDecodedBytes: 0 }, [30], [525, 450, 26, 7_000_000])),
  /geometry byte count/,
);

const truncated = envelope(header, Array(12).fill(0)).subarray(0, -1);
assert.throws(() => parseNativeFieldWindow(truncated), /body size/);

console.log("native field-window contract tests passed");
