import assert from "node:assert/strict";
import { performance } from "node:perf_hooks";
import { encodeCsv, encodeNpy, MAX_CSV_CELLS } from "../../../../ui/src/lib/platform/pythonExport.ts";
import { decodePythonNumericArray, pythonArrayDtype } from "../../../../ui/src/lib/platform/pythonNumericArray.ts";

const inputs = [Float32Array.of(-0, NaN, Infinity, -Infinity, 1.25), Float64Array.of(1, 1 + 1e-12),
  Int8Array.of(-128, 127), Uint8Array.of(0, 255), Int16Array.of(-32768, 32767),
  Uint16Array.of(0, 65535), Int32Array.of(-2147483648, 2147483647), Uint32Array.of(0, 4294967295),
  BigInt64Array.of(-9223372036854775808n, 9007199254740993n), BigUint64Array.of(18446744073709551615n)];
for (const input of inputs) {
  const source = new Uint8Array(input.buffer, input.byteOffset, input.byteLength).slice();
  const encoded = encodeNpy(input, [input.length]);
  const offset = 10 + new DataView(encoded.buffer).getUint16(8, true);
  assert.deepEqual(encoded.subarray(offset), source);
  assert.deepEqual(decodePythonNumericArray(encoded.slice(offset), pythonArrayDtype(input)), input);
  assert.deepEqual(new Uint8Array(input.buffer, input.byteOffset, input.byteLength), source);
}
const payload = new Uint32Array([0x7fc01234, 0x80000000, 0x7f800000]);
const nanArray = new Float32Array(payload.buffer);
const encodedNan = encodeNpy(nanArray, [3]);
const nanOffset = 10 + new DataView(encodedNan.buffer).getUint16(8, true);
assert.deepEqual(encodedNan.subarray(nanOffset), new Uint8Array(payload.buffer), "NPY preserves NaN payload, negative zero and infinity bits");
const parent = Float64Array.of(99, 1.125, 1 + 1e-12, 88);
const view = parent.subarray(1, 3);
const encodedView = encodeNpy(view, [2]);
const viewOffset = 10 + new DataView(encodedView.buffer).getUint16(8, true);
assert.deepEqual(decodePythonNumericArray(encodedView.slice(viewOffset), "<f8"), view);
assert.equal(parent[0], 99);
assert.equal(parent[3], 88);

for (const shape of [[-1, -1], [1.5], [NaN], [Number.MAX_SAFE_INTEGER, 2], [2, 2]]) {
  assert.throws(() => encodeNpy(Float32Array.of(1), shape));
  assert.throws(() => encodeCsv(Float32Array.of(1), shape));
}
assert.throws(() => encodeNpy(Float64Array.of(1), [1], "<i8"));
assert.throws(() => decodePythonNumericArray(new Uint8Array(8), "|f8"));
const unaligned = new Uint8Array(9);
new DataView(unaligned.buffer).setFloat64(1, 1 + 1e-12, true);
assert.deepEqual(decodePythonNumericArray(unaligned.subarray(1), "<f8"), Float64Array.of(1 + 1e-12));
assert.throws(() => encodeCsv(Int32Array.of(1), [1], "<f4"));
assert.throws(() => encodeNpy(Uint8Array.of(2), [1], "|b1"));
assert.doesNotThrow(() => encodeNpy(Uint8Array.of(0, 1), [2], "|b1"));
assert.doesNotThrow(() => encodeNpy(Float32Array.of(2), []));
assert.equal(encodeCsv(new Float32Array(0), [0]), "");
assert.equal(encodeCsv(new Float32Array(0), [0, 4]), "");
assert.throws(() => encodeCsv(new Float32Array(MAX_CSV_CELLS + 1), [MAX_CSV_CELLS + 1]));

const large = new Float32Array(16 * 1024 * 1024);
large[large.length - 1] = -1.125;
const started = performance.now();
const bytes = encodeNpy(large, [256, 256, 256]);
const durationMs = performance.now() - started;
const dataOffset = 10 + new DataView(bytes.buffer).getUint16(8, true);
assert.equal(new DataView(bytes.buffer).getFloat32(dataOffset + large.byteLength - 4, true), -1.125);
console.log(`Python numeric exports: passed; 64 MiB NPY bulk copy ${durationMs.toFixed(2)} ms on this local host`);
