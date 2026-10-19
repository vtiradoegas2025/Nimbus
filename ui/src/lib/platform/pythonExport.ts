import { pythonArrayDtype, pythonDtypeBytes, type PythonNumericArray } from "./pythonNumericArray";
/*
 * Encode Python outputs (figures, animations, arrays) into exportable files.
 * Animation export reuses the GIF encoder; frames are pre-rendered so no re-run is needed.
 */

/** Decode base64 to bytes. `atob` yields a binary string; each code unit is
 *  one byte, so it maps directly onto a Uint8Array. */
export function base64ToBytes(base64: string): Uint8Array {
  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let index = 0; index < binary.length; index += 1) {
    bytes[index] = binary.charCodeAt(index);
  }
  return bytes;
}

export function pngDataUri(base64: string): string {
  return `data:image/png;base64,${base64}`;
}

/** Measure size from the first frame (GIF has one canvas). bbox_inches="tight" can vary size per frame. */
export async function measurePngSize(
  base64: string,
): Promise<{ width: number; height: number }> {
  const image = new Image();
  image.src = pngDataUri(base64);
  await new Promise<void>((resolve, reject) => {
    image.onload = () => resolve();
    image.onerror = () => reject(new Error("Animation frame could not be decoded"));
  });
  if (!image.naturalWidth || !image.naturalHeight) {
    throw new Error("Animation frame has no pixel dimensions");
  }
  return { width: image.naturalWidth, height: image.naturalHeight };
}

/** Encode pre-rendered PNG frames into GIF bytes. */
export async function encodeFramesToGif(
  framesBase64: readonly string[],
  fps: number,
): Promise<Uint8Array> {
  if (framesBase64.length === 0) throw new Error("Nothing to encode: the animation has no frames");
  const size = await measurePngSize(framesBase64[0]);
  // Dynamic import keeps pure helpers (above) reachable from node tests without pulling in gifenc.
  const { GifRecorder } = await import("./gifRecorder");
  const recorder = new GifRecorder({ fps, ...size });
  try {
    for (const frame of framesBase64) {
      await recorder.writeDataUriFrame(pngDataUri(frame));
    }
    return await recorder.finish();
  } finally { recorder.dispose(); }
}

/* ---------- Arrays ---------- */

// ~12 B/value → 1M cells ≈ 12 MB string. Above this, .npy is the correct choice.
export const MAX_CSV_CELLS = 1_000_000;

const HOST_LITTLE_ENDIAN = new Uint8Array(Uint16Array.of(1).buffer)[0] === 1;

export function validatePythonNumericArray(data: PythonNumericArray, shape: readonly number[], dtype: string, validateValues = true): void {
  const actualDtype = pythonArrayDtype(data);
  if (dtype !== actualDtype && !(dtype === "|b1" && actualDtype === "|u1")) {
    throw new Error("Array dtype does not match its numeric buffer");
  }
  if (shape.length > 64 || shape.some(dimension => !Number.isSafeInteger(dimension) || dimension < 0)) {
    throw new Error("Array shape must contain nonnegative safe integers");
  }
  const cells = shape.reduce((product, dimension) => product * dimension, 1);
  if (!Number.isSafeInteger(cells) || cells !== data.length || pythonDtypeBytes(dtype) !== data.BYTES_PER_ELEMENT) {
    throw new Error("Array shape or dtype mismatch");
  }
  if (validateValues && dtype === "|b1" && data.some(value => value !== 0 && value !== 1)) throw new Error("Boolean array contains a value other than zero or one");
}

/**
 * Encode a numeric array as NPY v1.0 with its original dtype, in C order.
 */
export function encodeNpy(data: PythonNumericArray, shape: readonly number[], dtype = pythonArrayDtype(data)): Uint8Array {
  validatePythonNumericArray(data, shape, dtype);
  // 1-D shape needs a trailing comma or Python parses `(3)` as an integer, not a tuple.
  const dims = shape.length === 1
    ? `${shape[0]},`
    : shape.map((n) => `${n}`).join(", ");
  const descr = `{'descr': '${dtype}', 'fortran_order': False, 'shape': (${dims}), }`;
  // Header must end on a 64-byte boundary (including the 10-byte prefix); must end with \n.
  // numpy tolerates mis-padding; memory-mapped readers do not.
  const prefix = 10;
  const unpadded = descr.length + 1;
  const padded = Math.ceil((prefix + unpadded) / 64) * 64 - prefix;
  const header = descr + " ".repeat(padded - unpadded) + "\n";
  if (header.length > 0xffff) throw new Error("Array metadata exceeds the NPY v1 header limit");

  const out = new Uint8Array(prefix + header.length + data.byteLength);
  out.set([0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59, 1, 0], 0);
  out[8] = header.length & 0xff;
  out[9] = (header.length >> 8) & 0xff;
  for (let i = 0; i < header.length; i += 1) out[prefix + i] = header.charCodeAt(i);
  const source = new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
  if (HOST_LITTLE_ENDIAN || data.BYTES_PER_ELEMENT === 1) {
    out.set(source, prefix + header.length);
    return out;
  }
  for (let offset = 0; offset < source.length; offset += data.BYTES_PER_ELEMENT) {
    for (let byte = 0; byte < data.BYTES_PER_ELEMENT; byte++) {
      out[prefix + header.length + offset + byte] = source[offset + data.BYTES_PER_ELEMENT - byte - 1];
    }
  }
  return out;
}

// Round-trip digits depend on the source precision; integers never pass through Number.
function csvValue(value: number | bigint, precision: number): string {
  if (typeof value === "bigint") return String(value);
  if (Number.isNaN(value)) return "nan";
  if (value === Infinity) return "inf";
  if (value === -Infinity) return "-inf";
  if (Number.isInteger(value) && Math.abs(value) < 1e15) return `${value}`;
  return `${Number(value.toPrecision(precision))}`;
}

/** Encode a 1-D or 2-D numeric array as CSV. Rank >= 3 is refused; shape cannot be recovered from CSV. */
export function encodeCsv(data: PythonNumericArray, shape: readonly number[], dtype = pythonArrayDtype(data)): string {
  validatePythonNumericArray(data, shape, dtype);
  if (data.length > MAX_CSV_CELLS) throw new Error(`CSV export exceeds its ${MAX_CSV_CELLS}-value limit; save as .npy instead`);
  const format = (value: number | bigint) => csvValue(value, dtype === "<f4" ? 9 : 17);
  if (shape.length === 1) {
    return data.length ? Array.from({ length: data.length }, (_, i) => format(data[i])).join("\n") + "\n" : "";
  }
  if (shape.length !== 2) {
    throw new Error(
      `CSV holds a table, so it cannot represent a ${shape.length}-dimensional `
      + `array without discarding its shape. Save it as .npy instead.`,
    );
  }
  const [rows, columns] = shape;
  const lines: string[] = [];
  for (let r = 0; r < rows; r += 1) {
    const line: string[] = [];
    for (let c = 0; c < columns; c += 1) line.push(format(data[r * columns + c]));
    lines.push(line.join(","));
  }
  return lines.length ? lines.join("\n") + "\n" : "";
}
