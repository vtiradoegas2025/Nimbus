import type { Data, Layout, LayoutAxis } from "plotly.js";

function regularCoordinates(value: unknown): boolean {
  if (value == null) return true;
  if (!Array.isArray(value) && !ArrayBuffer.isView(value)) return false;
  const values = value as ArrayLike<unknown>;
  if (values.length < 2) return false;
  const first = values[0], second = values[1];
  if (typeof first !== "number" || typeof second !== "number" || !Number.isFinite(first) || !Number.isFinite(second)) return false;
  const step = second - first;
  if (!step) return false;
  for (let i = 2; i < values.length; i++) {
    const current = values[i];
    const expected = first + i * step;
    if (typeof current !== "number" || !Number.isFinite(current)
        || Math.abs(current - expected) > Math.max(1, Math.abs(expected), Math.abs(step)) * 1e-8) return false;
  }
  return true;
}

export function smoothHeatmapDisplay(data: Data[], enabled: boolean, layout?: Partial<Layout>): { data: Data[]; smoothed: number } {
  if (!enabled) return { data, smoothed: 0 };
  let smoothed = 0;
  const result = data.map(trace => {
    const record = trace as unknown as Record<string, unknown>;
    if (record.type !== "heatmap" || !regularCoordinates(record.x) || !regularCoordinates(record.y)) return trace;
    for (const axis of ["x", "y"]) {
      const name = String(record[`${axis}axis`] ?? axis).replace(/^([xy])(\d*)$/, "$1axis$2");
      const type = (layout as Record<string, Partial<LayoutAxis>> | undefined)?.[name]?.type;
      if (type != null && type !== "linear" && type !== "-") return trace;
    }
    const z = record.z;
    if (!Array.isArray(z) || z.length < 2 || !Array.isArray(z[0]) || z[0].length < 2) return trace;
    const width = z[0].length;
    for (const row of z) {
      if (!Array.isArray(row) || row.length !== width) return trace;
      for (const value of row) if (typeof value !== "number" || !Number.isFinite(value)) return trace;
    }
    smoothed++;
    return { ...trace, zsmooth: "fast", connectgaps: false } as Data;
  });
  return { data: smoothed ? result : data, smoothed };
}
