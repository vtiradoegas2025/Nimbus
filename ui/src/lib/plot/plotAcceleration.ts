import type { Data } from "plotly.js";

/**
 * SVG is excellent for publication-scale line work, but its DOM cost grows
 * with every point. Above this threshold the same scientific samples are sent
 * to Plotly's WebGL scatter renderer. No decimation is performed here.
 */
export const DENSE_SCATTER_GL_THRESHOLD = 5_000;

export type PlotRenderBackend = "empty" | "svg" | "raster" | "webgl" | "mixed";

export interface PlotAccelerationResult {
  data: Data[];
  backend: PlotRenderBackend;
  acceleratedTraceCount: number;
  pointCount: number;
}

type TraceRecord = Record<string, unknown>;

function arrayLength(value: unknown): number {
  if (Array.isArray(value)) return value.length;
  if (ArrayBuffer.isView(value) && "length" in value) {
    const length = (value as ArrayBufferView & { length: unknown }).length;
    return typeof length === "number" ? length : 0;
  }
  return 0;
}

export function plotlyTracePointCount(trace: Data): number {
  const record = trace as unknown as TraceRecord;
  return Math.max(arrayLength(record.x), arrayLength(record.y));
}

/**
 * Keep the automatic conversion deliberately conservative. Text, area fills,
 * stacked values, splines, and dashed lines have renderer-specific semantics; those stay on SVG. Dense
 * ordinary lines/markers (including vector glyph polylines and Python Plotly
 * output) are scientifically identical in scattergl.
 */
export function scatterTraceCanUseWebGl(
  trace: Data,
  threshold = DENSE_SCATTER_GL_THRESHOLD,
): boolean {
  const record = trace as unknown as TraceRecord;
  // Plotly's default trace type is scatter; Python figures commonly omit it.
  if (record.type != null && record.type !== "scatter") return false;
  if (plotlyTracePointCount(trace) < threshold) return false;

  const mode = typeof record.mode === "string" ? record.mode : "markers";
  if (mode.includes("text")) return false;
  if (!mode.includes("lines") && !mode.includes("markers")) return false;
  if (record.fill != null && record.fill !== "none") return false;
  if (record.stackgroup != null && record.stackgroup !== "") return false;

  const line = record.line;
  if (line && typeof line === "object") {
    if ("dash" in line) return false;
    if ("shape" in line && line.shape === "spline") return false;
  }
  return true;
}

function traceBackend(trace: Data): "svg" | "raster" | "webgl" {
  const type = (trace as unknown as TraceRecord).type;
  if (type === "scattergl") return "webgl";
  // Plotly 3.x removed heatmapgl. Its maintained heatmap/image paths upload a
  // raster layer while axes, labels and overlays remain DOM/SVG.
  if (type === "heatmap" || type === "image") return "raster";
  return "svg";
}

function summarizeBackend(data: Data[]): PlotRenderBackend {
  if (data.length === 0) return "empty";
  const backends = new Set(data.map(traceBackend));
  return backends.size === 1 ? [...backends][0] : "mixed";
}

/** Select the fastest renderer without dropping or resampling source points. */
export function acceleratePlotlyData(data: Data[]): PlotAccelerationResult {
  let acceleratedTraceCount = 0;
  let pointCount = 0;
  let changed = false;

  const accelerated = data.map((trace) => {
    pointCount += plotlyTracePointCount(trace);
    if (!scatterTraceCanUseWebGl(trace)) return trace;
    acceleratedTraceCount += 1;
    changed = true;
    return { ...(trace as unknown as TraceRecord), type: "scattergl" } as Data;
  });

  return {
    data: changed ? accelerated : data,
    backend: summarizeBackend(accelerated),
    acceleratedTraceCount,
    pointCount,
  };
}
