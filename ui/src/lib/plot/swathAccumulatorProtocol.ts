export interface SwathAccumulatorGeometry {
  accWidth: number;
  accHeight: number;
  isMax: boolean;
}

export type SwathAccumulatorRequest =
  | ({ kind: "initialize" } & SwathAccumulatorGeometry)
  | {
      kind: "resize";
      accWidth: number;
      accHeight: number;
      shiftR: number;
      shiftT: number;
    }
  | {
      kind: "accumulate";
      requestId: number;
      data: Float32Array;
      shape: number[];
      nr: number;
      nth: number;
      offsetR: number;
      offsetT: number;
      snapshot: boolean;
    }
  | { kind: "close" };

export type SwathAccumulatorReply =
  | { kind: "ready" }
  | { kind: "initialized" }
  | { kind: "resized" }
  | {
      kind: "result";
      requestId: number;
      ok: true;
      snapshot: Float64Array | null;
    }
  | {
      kind: "result";
      requestId: number;
      ok: false;
      error: string;
    };

/** Large planes publish fewer intermediate copies; the final result is exact. */
export function swathProgressSnapshotCount(cells: number): number {
  if (cells >= 1_048_576) return 4;
  if (cells >= 262_144) return 8;
  return 24;
}

export function resizeSwathAccumulator(
  accum: Float64Array,
  seen: Uint8Array,
  oldWidth: number,
  oldHeight: number,
  newWidth: number,
  newHeight: number,
  shiftR: number,
  shiftT: number,
  isMax: boolean,
): { accum: Float64Array; seen: Uint8Array } {
  if (accum.length !== oldWidth * oldHeight || seen.length !== accum.length) {
    throw new Error("Swath accumulator storage disagrees with its geometry");
  }
  if (![oldWidth, oldHeight, newWidth, newHeight].every((value) =>
    Number.isSafeInteger(value) && value > 0
  ) || !Number.isSafeInteger(shiftR) || !Number.isSafeInteger(shiftT)
      || shiftR < 0 || shiftT < 0
      || shiftR + oldHeight > newHeight || shiftT + oldWidth > newWidth) {
    throw new Error("Swath accumulator resize does not contain the previous grid");
  }
  if (oldWidth === newWidth && oldHeight === newHeight && shiftR === 0 && shiftT === 0) {
    return { accum, seen };
  }
  const resized = new Float64Array(newWidth * newHeight).fill(isMax ? -Infinity : Infinity);
  const resizedSeen = new Uint8Array(resized.length);
  for (let row = 0; row < oldHeight; row += 1) {
    const oldStart = row * oldWidth;
    const newStart = (row + shiftR) * newWidth + shiftT;
    resized.set(accum.subarray(oldStart, oldStart + oldWidth), newStart);
    resizedSeen.set(seen.subarray(oldStart, oldStart + oldWidth), newStart);
  }
  return { accum: resized, seen: resizedSeen };
}
