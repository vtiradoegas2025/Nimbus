import assert from "node:assert/strict";
import { pythonProviderResult } from "../../../../ui/src/lib/derived/pythonProviderResult.ts";
import type { RunResult } from "../../../../ui/src/lib/platform/pyodideProtocol.ts";

const result = (data: Float32Array | Float64Array | Int32Array, raised = false): RunResult => ({
  raised, stdout: "", stderr: raised ? "ValueError: interrupted provider" : "",
  outputs: [{ kind: "ndarray", label: null, data, shape: [data.length] }], produced: [],
});
const data = Float32Array.of(1.25, NaN, -Infinity, -0);
const accepted = pythonProviderResult(result(data));
assert.equal(accepted.values, data.buffer, "a complete worker-owned Float32 buffer is reused without a second copy");
assert.deepEqual(new Float32Array(accepted.values), data);
assert.throws(() => pythonProviderResult(result(Int32Array.of(1))), /dtype float32/, "integer bits cannot masquerade as floating point values");
assert.throws(() => pythonProviderResult(result(Float64Array.of(1 + 1e-12))), /dtype float32/, "a provider chooses precision conversion explicitly");
assert.throws(() => pythonProviderResult(result(Float32Array.of(1), true)), /raised an exception/, "partial output from a failed provider is not publishable");
const source = Float32Array.of(99, 2.5, 3.5, 88);
const crop = source.subarray(1, 3);
const croppedResult = pythonProviderResult(result(crop));
assert.notEqual(croppedResult.values, source.buffer);
assert.deepEqual(new Float32Array(croppedResult.values), Float32Array.of(2.5, 3.5));
const invalid = result(Float32Array.of(1));
invalid.outputs[0] = { kind: "ndarray", label: null, data: Float32Array.of(1), shape: [2] };
assert.throws(() => pythonProviderResult(invalid), /shape/);
console.log("Python provider precision, failed-run rejection, buffer ownership and shape: passed");
