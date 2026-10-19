import type { RunResult } from "@/lib/platform/pyodideProtocol";

export function pythonProviderResult(result: RunResult): { shape: number[]; values: ArrayBuffer } {
  if (result.raised) throw new Error(`Python provider raised an exception: ${result.stderr.slice(0, 1_000)}`);
  const array = result.outputs.find(output => output.kind === "ndarray");
  if (!array || array.kind !== "ndarray") throw new Error("Python provider must assign a NumPy ndarray to 'result'");
  if (!(array.data instanceof Float32Array) || (array.dtype != null && array.dtype !== "<f4")) {
    throw new Error("Python provider result must have dtype float32; choose that dtype explicitly in the provider source");
  }
  const elements = array.shape.reduce((product, dimension) => product * dimension, 1);
  if (array.shape.length === 0 || array.shape.some(dimension => !Number.isSafeInteger(dimension) || dimension <= 0)
      || !Number.isSafeInteger(elements) || elements !== array.data.length) {
    throw new Error("Python provider result shape does not match its buffer");
  }
  const values = array.data.buffer instanceof ArrayBuffer && array.data.byteOffset === 0
    && array.data.byteLength === array.data.buffer.byteLength
    ? array.data.buffer : array.data.slice().buffer as ArrayBuffer;
  return { shape: [...array.shape], values };
}
