import type { ScalarPlanePolicy } from "@/lib/plot/scalarPlanePolicy";

export interface RadarFloorEncodingResult {
  rgba: Uint8ClampedArray<ArrayBuffer>;
  paintedCells: number;
  elapsedMs: number;
}

export type RadarFloorEncodingWorkerRequest =
  | {
      kind: "encode";
      requestId: number;
      values: Float32Array;
      shape: [number, number];
      policy: ScalarPlanePolicy;
    }
  | { kind: "close" };

export type RadarFloorEncodingWorkerReply =
  | { kind: "ready" }
  | {
      kind: "result";
      requestId: number;
      ok: true;
      result: RadarFloorEncodingResult;
    }
  | {
      kind: "result";
      requestId: number;
      ok: false;
      error: string;
    };
