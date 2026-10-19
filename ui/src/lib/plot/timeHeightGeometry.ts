import type { GridInfo } from "../run/runOutputs";
import type { SliceAxis } from "../platform/npy";
import { axisCoords } from "../science/gridCoords";
import { cropAxisReduction } from "../science/validDomain";

export interface TimeHeightColumn {
  step: number;
  timeS: number | null;
  values: (number | null)[];
  grid: GridInfo | null;
}

export function timeHeightGeometry(series: readonly TimeHeightColumn[], axis: SliceAxis) {
  const haveAllTimes = series.every((column) => column.timeS != null && Number.isFinite(column.timeS));
  const times = series.map((column) => haveAllTimes ? column.timeS! : column.step);
  const increasing = times.every((time, index) => index === 0 || time > times[index - 1]);
  const xs = increasing ? times : series.map((_, index) => index);
  const edges = xs.map((value, index) => index === 0
    ? value - (xs.length > 1 ? (xs[1] - value) / 2 : 0.5)
    : (value + xs[index - 1]) / 2);
  if (xs.length) edges.push(xs.at(-1)! + (xs.length > 1 ? (xs.at(-1)! - xs.at(-2)!) / 2 : 0.5));
  const columns = series.map((column) => {
    const cropped = cropAxisReduction(axis, column.values, column.grid);
    const coordinates = axisCoords(axis, cropped.values.length, cropped.grid);
    return { values: cropped.values, coordinates,
      ys: coordinates.values ?? Array.from({ length: cropped.values.length }, (_, index) => index) };
  });
  const reference = columns.find((column) => column.values.length > 0);
  if (!reference) return null;
  if (columns.some((column) => column.values.length > 0 && column.coordinates.unit !== reference.coordinates.unit)) {
    return { error: "Frames mix physical coordinates and model indices on this axis. Choose an axis with consistent units." } as const;
  }
  const groups: Array<{ x: number[]; y: number[]; z: (number | null)[][]; times: number[] }> = [];
  let group: typeof groups[number] | null = null;
  for (let index = 0; index < columns.length; index++) {
    const column = columns[index];
    if (!column.values.length) { group = null; continue; }
    if (!group || group.y.length !== column.ys.length || group.y.some((value, i) => value !== column.ys[i])) {
      group = { x: [edges[index]], y: column.ys, z: column.ys.map(() => []), times: [] };
      groups.push(group);
    }
    group.x.push(edges[index + 1]);
    group.times.push(xs[index]);
    column.values.forEach((value, level) => group!.z[level].push(value != null && Number.isFinite(value) ? value : null));
  }
  let min = Infinity, max = -Infinity;
  for (const column of columns) for (const value of column.values) {
    if (value != null && Number.isFinite(value)) { min = Math.min(min, value); max = Math.max(max, value); }
  }
  return { groups, coordinates: reference.coordinates, range: Number.isFinite(min) ? { min, max } : null,
    xRange: [edges[0], edges.at(-1)!] as [number, number],
    xTitle: increasing ? (haveAllTimes ? "t (s)" : "step") : "frame ordinal",
    xHover: increasing ? (haveAllTimes ? "t=%{x:.1f}s" : "step %{x}") : "frame %{x}" };
}
