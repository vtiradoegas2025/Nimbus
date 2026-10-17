export function scalarMultisetFingerprint(
  arrays: readonly Float32Array[],
): string {
  const bits = new Uint32Array(1);
  const value = new Float32Array(bits.buffer);
  let count = 0;
  let xor = 0;
  let sum = 0;
  let sumSquares = 0;
  for (const array of arrays) {
    for (const scalar of array) {
      value[0] = scalar;
      const word = bits[0] >>> 0;
      xor = (xor ^ word) >>> 0;
      sum = (sum + word) >>> 0;
      sumSquares = (sumSquares + Math.imul(word, word)) >>> 0;
      count += 1;
    }
  }
  return [count, xor, sum, sumSquares]
    .map((part, index) => index === 0 ? String(part) : part.toString(16).padStart(8, "0"))
    .join(":");
}

export function scalarQuantizedMultisetFingerprint(
  arrays: readonly Float32Array[],
  quantum = 1e-5,
): string {
  if (!(quantum > 0) || !Number.isFinite(quantum)) {
    throw new Error("Scalar fingerprint quantum must be positive and finite");
  }
  let count = 0;
  let xor = 0;
  let sum = 0;
  let sumSquares = 0;
  for (const array of arrays) {
    for (const scalar of array) {
      const quantized = Number.isFinite(scalar) ? Math.round(scalar / quantum) | 0 : 0x7fffffff;
      const word = quantized >>> 0;
      xor = (xor ^ word) >>> 0;
      sum = (sum + word) >>> 0;
      sumSquares = (sumSquares + Math.imul(word, word)) >>> 0;
      count += 1;
    }
  }
  return [count, xor, sum, sumSquares]
    .map((part, index) => index === 0 ? String(part) : part.toString(16).padStart(8, "0"))
    .join(":");
}
