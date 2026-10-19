import type {
  SwathAccumulatorReply,
  SwathAccumulatorRequest,
} from "./swathAccumulatorProtocol";
import { resizeSwathAccumulator } from "./swathAccumulatorProtocol";

const scope = self as unknown as {
  onmessage: ((event: { data: SwathAccumulatorRequest }) => void) | null;
  postMessage: (message: SwathAccumulatorReply, transfer?: ArrayBuffer[]) => void;
  close: () => void;
};

let accum: Float64Array | null = null;
let seen: Uint8Array | null = null;
let accWidth = 0;
let accHeight = 0;
let isMax = true;

function initialize(width: number, height: number, maximum: boolean): void {
  if (!Number.isSafeInteger(width) || !Number.isSafeInteger(height) || width < 1 || height < 1) {
    throw new Error("Swath accumulator dimensions are invalid");
  }
  accWidth = width;
  accHeight = height;
  isMax = maximum;
  accum = new Float64Array(width * height).fill(maximum ? -Infinity : Infinity);
  seen = new Uint8Array(width * height);
}

function resize(width: number, height: number, shiftR: number, shiftT: number): void {
  if (!accum || !seen) throw new Error("Swath accumulator was not initialized");
  const next = resizeSwathAccumulator(
    accum, seen, accWidth, accHeight, width, height, shiftR, shiftT, isMax,
  );
  accum = next.accum;
  seen = next.seen;
  accWidth = width;
  accHeight = height;
}

function accumulate(request: Extract<SwathAccumulatorRequest, { kind: "accumulate" }>): void {
  if (!accum || !seen) throw new Error("Swath accumulator was not initialized");
  const is3D = request.shape.length === 3;
  const nz = is3D ? request.shape[2]! : 1;
  for (let r = 0; r < request.nr; r += 1) {
    for (let t = 0; t < request.nth; t += 1) {
      let cell: number;
      if (is3D) {
        let reduced = isMax ? -Infinity : Infinity;
        let any = false;
        const base = (r * request.nth + t) * nz;
        for (let k = 0; k < nz; k += 1) {
          const value = request.data[base + k]!;
          if (!Number.isFinite(value)) continue;
          reduced = isMax ? Math.max(reduced, value) : Math.min(reduced, value);
          any = true;
        }
        if (!any) continue;
        cell = reduced;
      } else {
        const value = request.data[r * request.nth + t]!;
        if (!Number.isFinite(value)) continue;
        cell = value;
      }
      const ar = r + request.offsetR;
      const at = t + request.offsetT;
      const index = ar * accWidth + at;
      if (index < 0 || index >= accum.length) continue;
      accum[index] = isMax ? Math.max(accum[index]!, cell) : Math.min(accum[index]!, cell);
      seen[index] = 1;
    }
  }
}

function materialize(): Float64Array {
  const output = new Float64Array(accum!.length);
  for (let index = 0; index < output.length; index += 1) {
    output[index] = seen![index] ? accum![index]! : Number.NaN;
  }
  return output;
}

scope.onmessage = (event) => {
  const request = event.data;
  if (request.kind === "close") {
    accum = null;
    seen = null;
    scope.close();
    return;
  }
  if (request.kind === "initialize") {
    try {
      initialize(request.accWidth, request.accHeight, request.isMax);
      scope.postMessage({ kind: "initialized" });
    } catch (error) {
      scope.postMessage({
        kind: "result",
        requestId: -1,
        ok: false,
        error: error instanceof Error ? error.message : String(error),
      });
    }
    return;
  }
  if (request.kind === "resize") {
    try {
      resize(request.accWidth, request.accHeight, request.shiftR, request.shiftT);
      scope.postMessage({ kind: "resized" });
    } catch (error) {
      scope.postMessage({
        kind: "result",
        requestId: -1,
        ok: false,
        error: error instanceof Error ? error.message : String(error),
      });
    }
    return;
  }
  try {
    accumulate(request);
    const snapshot = request.snapshot ? materialize() : null;
    scope.postMessage(
      { kind: "result", requestId: request.requestId, ok: true, snapshot },
      snapshot ? [snapshot.buffer as ArrayBuffer] : [],
    );
  } catch (error) {
    scope.postMessage({
      kind: "result",
      requestId: request.requestId,
      ok: false,
      error: error instanceof Error ? error.message : String(error),
    });
  }
};

scope.postMessage({ kind: "ready" });

export {};
