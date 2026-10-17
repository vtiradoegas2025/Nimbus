import { ResearchAccumulator, type ScanChunk, type ScanSpec, type ScanResult } from "./statistics";
export type ScanRequest = {
    kind: "start";
    spec: ScanSpec;
} | {
    kind: "append";
    chunk: ScanChunk;
} | {
    kind: "finish";
} | {
    kind: "surface-start";
} | {
    kind: "surface-append";
    values: Float32Array;
} | {
    kind: "surface-finish";
};
export type ScanReply = {
    ok: true;
    result?: ScanResult;
    surface?: {
        min: number | null;
        max: number | null;
        mean: number | null;
        finite: number;
        total: number;
    };
} | {
    ok: false;
    error: string;
};
const scope = self as unknown as {
    onmessage: (event: MessageEvent<ScanRequest>) => void;
    postMessage: (reply: ScanReply) => void;
    close: () => void;
};
let accumulator: ResearchAccumulator | null = null;
let surface = { min: Infinity, max: -Infinity, sum: 0, finite: 0, total: 0 };
scope.onmessage = ({ data }) => {
    try {
        if (data.kind === "surface-start")
            surface = { min: Infinity, max: -Infinity, sum: 0, finite: 0, total: 0 };
        else if (data.kind === "surface-append") {
            for (const value of data.values) {
                surface.total++;
                if (Number.isFinite(value)) {
                    surface.finite++;
                    surface.min = Math.min(surface.min, value);
                    surface.max = Math.max(surface.max, value);
                    surface.sum += value;
                }
            }
        }
        else if (data.kind === "surface-finish") {
            scope.postMessage({ ok: true, surface: { min: surface.finite ? surface.min : null, max: surface.finite ? surface.max : null, mean: surface.finite ? surface.sum / surface.finite : null, finite: surface.finite, total: surface.total } });
            scope.close();
            return;
        }
        else if (data.kind === "start")
            accumulator = new ResearchAccumulator(data.spec);
        else if (data.kind === "append") {
            if (!accumulator)
                throw new Error("Analysis has not started");
            accumulator.append(data.chunk);
        }
        else {
            if (!accumulator)
                throw new Error("Analysis has not started");
            scope.postMessage({ ok: true, result: accumulator.finish() });
            scope.close();
            return;
        }
        scope.postMessage({ ok: true });
    }
    catch (error) {
        scope.postMessage({ ok: false, error: error instanceof Error ? error.message : String(error) });
        scope.close();
    }
};
