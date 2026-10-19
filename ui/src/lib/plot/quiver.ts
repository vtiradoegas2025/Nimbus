/* 2D quiver builder. Each component is scaled to its own axis's cell spacing so arrows are
 * readable under axis-unit mismatch (r km vs θ deg). Max-magnitude arrow spans `stride` cells. */

export interface QuiverInput {
  /** x-component (along columns / horizontal display axis), indexed [row][col]. */
  u: number[][];
  /** y-component (along rows / vertical display axis), indexed [row][col]. */
  v: number[][];
  rows: number;
  cols: number;
  /** Physical coordinate per column; falls back to the column index. */
  xCoords?: number[];
  /** Physical coordinate per row; falls back to the row index. */
  yCoords?: number[];
  /** Physical spacing between adjacent columns (display x units). */
  xSpacing: number;
  /** Physical spacing between adjacent rows (display y units). */
  ySpacing: number;
  /** Sample every `stride` cells (>= 1). */
  stride: number;
  /** User multiplier on the auto length scale. */
  scale: number;
  /** Optional shared normalization across panels. The sampled maximum is still
   * reported, but arrow lengths use this magnitude when it is finite/positive. */
  referenceMagnitude?: number;
}

export interface QuiverResult {
  /** Plotly line-trace x with `null` separators between arrows/segments. */
  x: (number | null)[];
  /** Plotly line-trace y with `null` separators between arrows/segments. */
  y: (number | null)[];
  /** Largest vector magnitude over the sampled points (data units). */
  maxMagnitude: number;
  /** Number of arrows actually emitted (finite, non-degenerate samples). */
  arrowCount: number;
  /** Arrow tail positions, one per emitted arrow, for an invisible hover-target trace. */
  originX: number[];
  originY: number[];
  /** Per-arrow [u, v, |V|] in DATA units, aligned with originX/originY. */
  components: Array<[number, number, number]>;
  /**
   * The stride actually sampled. Larger than the requested one when the plane
   * would otherwise exceed MAX_QUIVER_ARROWS; callers must disclose it, because
   * a thinned field is a different picture from the one the control claims.
   */
  effectiveStride: number;
}

/** Arrow ceiling per quiver trace; keeps Plotly SVG path size within comfortable range. */
export const MAX_QUIVER_ARROWS = 8_000;

/** Smallest stride keeping ceil(rows/s)*ceil(cols/s) within MAX_QUIVER_ARROWS. */
export function quiverStrideFor(rows: number, cols: number, requested: number): number {
  const base = Math.max(1, Math.floor(requested));
  if (!(rows > 0) || !(cols > 0)) return base;
  let step = base;
  // Each increment strictly reduces the count and the loop is bounded by the
  // larger axis, at which point exactly one arrow remains.
  while (
    step < Math.max(rows, cols)
    && Math.ceil(rows / step) * Math.ceil(cols / step) > MAX_QUIVER_ARROWS
  ) {
    step += 1;
  }
  return step;
}

const HEAD_ANGLE_RAD = (25 * Math.PI) / 180;
const HEAD_FRACTION = 0.35;

function rotate(vx: number, vy: number, ang: number): [number, number] {
  const c = Math.cos(ang);
  const s = Math.sin(ang);
  return [vx * c - vy * s, vx * s + vy * c];
}

/** Largest finite vector magnitude over the sampled grid; use as a cross-frame normalization reference. */
export function sampledMaxMagnitude(
  u: number[][],
  v: number[][],
  rows: number,
  cols: number,
  stride: number,
): number {
  const step = Math.max(1, Math.floor(stride));
  let maxMagnitude = 0;
  for (let i = 0; i < rows; i += step) {
    for (let j = 0; j < cols; j += step) {
      const a = u[i]?.[j];
      const b = v[i]?.[j];
      if (!Number.isFinite(a) || !Number.isFinite(b)) continue;
      const m = Math.hypot(a, b);
      if (m > maxMagnitude) maxMagnitude = m;
    }
  }
  return maxMagnitude;
}

export function buildQuiver(input: QuiverInput): QuiverResult {
  const { u, v, rows, cols, xCoords, yCoords, stride, scale } = input;
  const step = quiverStrideFor(rows, cols, stride);
  const xSpacing = input.xSpacing || 1;
  const ySpacing = input.ySpacing || 1;

  const maxMagnitude = sampledMaxMagnitude(u, v, rows, cols, step);

  const x: (number | null)[] = [];
  const y: (number | null)[] = [];
  const originX: number[] = [];
  const originY: number[] = [];
  const components: Array<[number, number, number]> = [];
  if (!(maxMagnitude > 0)) {
    return {
      x, y, originX, originY, components,
      maxMagnitude: 0, arrowCount: 0, effectiveStride: step,
    };
  }

  // A max-magnitude vector spans `step` cells in each axis (times user scale).
  const referenceMagnitude = input.referenceMagnitude && input.referenceMagnitude > 0
    ? input.referenceMagnitude
    : maxMagnitude;
  const scaleX = (step * xSpacing * scale) / referenceMagnitude;
  const scaleY = (step * ySpacing * scale) / referenceMagnitude;

  let arrowCount = 0;
  for (let i = 0; i < rows; i += step) {
    for (let j = 0; j < cols; j += step) {
      const a = u[i]?.[j];
      const b = v[i]?.[j];
      if (!Number.isFinite(a) || !Number.isFinite(b)) continue;
      if (a === 0 && b === 0) continue;

      const x0 = xCoords ? xCoords[j] : j;
      const y0 = yCoords ? yCoords[i] : i;
      const dx = a * scaleX;
      const dy = b * scaleY;
      const x1 = x0 + dx;
      const y1 = y0 + dy;

      // Shaft.
      x.push(x0, x1, null);
      y.push(y0, y1, null);

      // Two-segment head: rotate the reversed shaft by ±head angle.
      const [lx, ly] = rotate(-dx, -dy, HEAD_ANGLE_RAD);
      const [rx, ry] = rotate(-dx, -dy, -HEAD_ANGLE_RAD);
      x.push(x1, x1 + lx * HEAD_FRACTION, null);
      y.push(y1, y1 + ly * HEAD_FRACTION, null);
      x.push(x1, x1 + rx * HEAD_FRACTION, null);
      y.push(y1, y1 + ry * HEAD_FRACTION, null);

      originX.push(x0);
      originY.push(y0);
      components.push([a, b, Math.hypot(a, b)]);
      arrowCount++;
    }
  }

  return { x, y, originX, originY, components, maxMagnitude, arrowCount, effectiveStride: step };
}
