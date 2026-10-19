export type PythonNumericArray = Float32Array | Float64Array
  | Int8Array | Uint8Array | Int16Array | Uint16Array | Int32Array | Uint32Array
  | BigInt64Array | BigUint64Array;

const constructors = {
  f4: Float32Array, f8: Float64Array, i1: Int8Array, u1: Uint8Array,
  b1: Uint8Array, i2: Int16Array, u2: Uint16Array, i4: Int32Array,
  u4: Uint32Array, i8: BigInt64Array, u8: BigUint64Array,
};

export function pythonDtypeBytes(dtype: string): number {
  const ctor = constructors[dtype.slice(1) as keyof typeof constructors];
  if (!ctor || !["<", "|"].includes(dtype[0]) || (dtype[0] === "|" && ctor.BYTES_PER_ELEMENT !== 1)) throw new Error(`Unsupported Python dtype: ${dtype}`);
  return ctor.BYTES_PER_ELEMENT;
}

export function decodePythonNumericArray(bytes: Uint8Array, dtype: string): PythonNumericArray {
  const size = pythonDtypeBytes(dtype);
  if (bytes.byteLength % size !== 0) throw new Error("Python array byte count does not match dtype");
  const ctor = constructors[dtype.slice(1) as keyof typeof constructors];
  const aligned = bytes.byteOffset % size === 0 ? bytes : bytes.slice();
  return new ctor(aligned.buffer as ArrayBuffer, aligned.byteOffset, aligned.byteLength / size);
}

export function pythonArrayDtype(data: PythonNumericArray): string {
  const entry = Object.entries(constructors).find(([, ctor]) => data instanceof ctor);
  if (!entry) throw new Error("Unsupported Python array");
  return `${data.BYTES_PER_ELEMENT === 1 ? "|" : "<"}${entry[0]}`;
}

export function pythonArrayPreview(data: PythonNumericArray): ArrayLike<number> | null {
  if (data instanceof BigInt64Array || data instanceof BigUint64Array) {
    for (const value of data) {
      if (value > BigInt(Number.MAX_SAFE_INTEGER) || value < BigInt(Number.MIN_SAFE_INTEGER)) return null;
    }
    return Float64Array.from(data, Number);
  }
  return data;
}
