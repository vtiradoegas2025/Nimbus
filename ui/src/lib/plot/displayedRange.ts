/*
 * The value range of the 2-D data actually on screen. Pure — no React.
 * Unit-tested in scripts/plot/test_displayed_range.ts.
 */
import type { ValueRange } from "./viewSync";

/**
 * Range of the drawn cells, skipping non-finite values. Null when nothing is finite,
 * so callers keep whatever range they were already showing rather than collapsing to zero.
 *
 * Colouring a slice by the range of the whole volume washes it out whenever the extremes
 * live at other heights: on a beast_xl frame, w at the ground spans 6% of the column's
 * range and vorticity_z at the ground is identically zero, so both painted flat.
 */
export function displayedRange(
  rows: ReadonlyArray<ReadonlyArray<number> | Float32Array | Float64Array> | null | undefined,
): ValueRange | null {
  if (!rows) return null;
  let min = Number.POSITIVE_INFINITY;
  let max = Number.NEGATIVE_INFINITY;
  for (const row of rows) {
    for (const value of row) {
      if (!Number.isFinite(value)) continue;
      if (value < min) min = value;
      if (value > max) max = value;
    }
  }
  return Number.isFinite(min) && Number.isFinite(max) ? { min, max } : null;
}

/**
 * Identity for a latched colour range.
 *
 * The latch holds a range steady across frames so one physical value keeps one colour
 * through an animation. It must therefore change when the data underneath changes for a
 * reason other than time — a different field, run, or slice plane — or a range fitted to
 * one height stays applied to another.
 */
export function displayedRangeScope(
  runDir: string,
  fieldName: string | null,
  sliceAxis: number | null,
  sliceIndex: number | null,
): string {
  return `${runDir}|${fieldName ?? ""}|${sliceAxis ?? "-"}|${sliceIndex ?? "-"}`;
}
