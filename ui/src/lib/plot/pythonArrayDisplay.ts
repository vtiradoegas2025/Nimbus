/**
 * Renderer-safe presentation for arrays returned by the Python worker.
 *
 * The transferred numeric array remains untouched for export. Only the Plotly
 * presentation is reduced, because expanding a legal 64 MiB array into boxed
 * JS numbers can consume hundreds of MiB and lock up WebKit.
 */

import { planeViewportFromRelayout, type PlaneViewportBounds } from "./viewSync";

export function pythonArrayViewportFromRelayout(
  event: Readonly<Record<string, unknown>>,
  previous: PlaneViewportBounds | null,
  shape: readonly number[],
  colorRange: readonly [number, number] | null,
): PlaneViewportBounds | null | undefined {
  if (!Object.entries(event).some(([key, value]) => /^(x|y)axis\.range(?:\[[01]\])?$/.test(key)
      || /^(x|y)axis\.autorange$/.test(key) && value === true)) return undefined;
  return planeViewportFromRelayout(event, previous ?? {
    xMin: 0, xMax: (shape.at(-1) ?? 1) - 1,
    yMin: shape.length === 2 ? 0 : colorRange?.[0] ?? 0,
    yMax: shape.length === 2 ? shape[0] - 1 : colorRange?.[1] ?? 1,
  });
}

export const MAX_PYTHON_LINE_DISPLAY_POINTS = 50_000;
export const MAX_PYTHON_HEATMAP_DISPLAY_CELLS = 1_000_000;

export interface PythonLineDisplay {
  kind: "line";
  x: number[];
  y: number[];
  sourcePoints: number;
  displayPoints: number;
  reduced: boolean;
  method: "native" | "ordered_bucket_extrema";
}

export interface PythonHeatmapDisplay {
  kind: "heatmap";
  x: number[];
  y: number[];
  z: number[][];
  sourceShape: readonly [number, number];
  displayShape: readonly [number, number];
  reduced: boolean;
  method: "native" | "point_sample";
}

export type PythonArrayDisplay = PythonLineDisplay | PythonHeatmapDisplay;

function checkedShape(data: ArrayLike<number>, shape: readonly number[]): void {
  const elements = shape.reduce((product, count) => product * count, 1);
  if (shape.some((count) => !Number.isSafeInteger(count) || count < 0)
      || !Number.isSafeInteger(elements) || elements !== data.length) {
    throw new Error("Python array display shape does not match its buffer");
  }
}

/** Ordered min/max buckets preserve finite peaks and at least one missing-data
 * break per bucket. The first and final source points are always retained. */
function lineDisplay(
  data: ArrayLike<number>,
  maxPoints: number,
  bounds: readonly [number, number],
): PythonLineDisplay {
  const sourcePoints = data.length;
  const [first, end] = bounds;
  const visiblePoints = Math.max(0, end - first);
  if (visiblePoints <= maxPoints) {
    return {
      kind: "line",
      x: Array.from({ length: visiblePoints }, (_, index) => first + index),
      y: Array.from({ length: visiblePoints }, (_, index) => data[first + index]),
      sourcePoints,
      displayPoints: visiblePoints,
      reduced: false,
      method: "native",
    };
  }
  if (!Number.isSafeInteger(maxPoints) || maxPoints < 8) {
    throw new Error("Python line display budget must hold at least eight points");
  }

  // A bucket contributes finite min/max plus one non-finite break at most.
  const bucketCount = Math.max(1, Math.floor((maxPoints - 2) / 3));
  const bucketWidth = Math.max(1, Math.ceil((visiblePoints - 2) / bucketCount));
  const selected = [first];
  for (let start = first + 1; start < end - 1; start += bucketWidth) {
    const stop = Math.min(end - 1, start + bucketWidth);
    let minimum = Number.POSITIVE_INFINITY;
    let maximum = Number.NEGATIVE_INFINITY;
    let minimumIndex = -1;
    let maximumIndex = -1;
    let missingIndex = -1;
    for (let index = start; index < stop; index += 1) {
      const value = data[index];
      if (!Number.isFinite(value)) {
        if (missingIndex < 0) missingIndex = index;
      } else {
        if (value < minimum) { minimum = value; minimumIndex = index; }
        if (value > maximum) { maximum = value; maximumIndex = index; }
      }
    }
    const candidates = [minimumIndex, maximumIndex, missingIndex]
      .filter((index, position, values) => index >= 0 && values.indexOf(index) === position)
      .sort((left, right) => left - right);
    selected.push(...candidates);
  }
  selected.push(end - 1);
  return {
    kind: "line",
    x: selected,
    y: selected.map((index) => data[index]),
    sourcePoints,
    displayPoints: selected.length,
    reduced: true,
    method: "ordered_bucket_extrema",
  };
}

function sampledIndices(count: number, stride: number): number[] {
  if (count === 0) return [];
  const indices: number[] = [];
  for (let index = 0; index < count; index += stride) indices.push(index);
  if (indices.at(-1) !== count - 1) indices.push(count - 1);
  return indices;
}

function heatmapDisplay(
  data: ArrayLike<number>,
  shape: readonly [number, number],
  maxCells: number,
  rowBounds: readonly [number, number],
  columnBounds: readonly [number, number],
): PythonHeatmapDisplay {
  if (!Number.isSafeInteger(maxCells) || maxCells < 4) {
    throw new Error("Python heatmap display budget must hold at least four cells");
  }
  const rows = Math.max(0, rowBounds[1] - rowBounds[0]);
  const columns = Math.max(0, columnBounds[1] - columnBounds[0]);
  let rowStride = 1;
  let columnStride = 1;
  let rowIndices = sampledIndices(rows, rowStride);
  let columnIndices = sampledIndices(columns, columnStride);
  while (rowIndices.length * columnIndices.length > maxCells) {
    if (rowIndices.length >= columnIndices.length) rowStride += 1;
    else columnStride += 1;
    rowIndices = sampledIndices(rows, rowStride);
    columnIndices = sampledIndices(columns, columnStride);
  }
  rowIndices = rowIndices.map(index => index + rowBounds[0]);
  columnIndices = columnIndices.map(index => index + columnBounds[0]);
  const z = rowIndices.map((row) =>
    columnIndices.map((column) => data[row * shape[1] + column]));
  const reduced = rowStride > 1 || columnStride > 1;
  return {
    kind: "heatmap",
    x: columnIndices,
    y: rowIndices,
    z,
    sourceShape: shape,
    displayShape: [rowIndices.length, columnIndices.length],
    reduced,
    method: reduced ? "point_sample" : "native",
  };
}

function visibleBounds(count: number, range?: readonly [number, number]): [number, number] {
  if (!range || !range.every(Number.isFinite)) return [0, count];
  return [Math.max(0, Math.min(count, Math.floor(Math.min(...range)) - 1)),
    Math.max(0, Math.min(count, Math.ceil(Math.max(...range)) + 2))];
}

export function preparePythonArrayDisplay(
  data: ArrayLike<number>,
  shape: readonly number[],
  options: { maxLinePoints?: number; maxHeatmapCells?: number;
    xRange?: readonly [number, number]; yRange?: readonly [number, number] } = {},
): PythonArrayDisplay | null {
  checkedShape(data, shape);
  if (shape.length === 1) {
    return lineDisplay(data, options.maxLinePoints ?? MAX_PYTHON_LINE_DISPLAY_POINTS, visibleBounds(shape[0], options.xRange));
  }
  if (shape.length === 2) {
    return heatmapDisplay(
      data,
      shape as readonly [number, number],
      options.maxHeatmapCells ?? MAX_PYTHON_HEATMAP_DISPLAY_CELLS,
      visibleBounds(shape[0], options.yRange),
      visibleBounds(shape[1], options.xRange),
    );
  }
  return null;
}

