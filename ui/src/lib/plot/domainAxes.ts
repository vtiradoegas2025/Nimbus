import type { StepInfo } from "@/lib/run/runOutputs";
import { cartesianDyMeters } from "@/lib/science/cartesianInterp";
import type { SceneScale } from "@/lib/plot/sceneScale";

/*
 * Tick geometry for the 3D domain box. Pure — no vtk.js, no React.
 * Physical grids use metres; index grids use source sample indices. Display
 * coordinates apply SceneScale without changing the labelled coordinate.
 */

/** Face-to-face domain bounds in TRUE metres. */
export interface DomainBoundsM {
  lower: [number, number, number];
  upper: [number, number, number];
}

/** Physical bounds in true metres. Falls back to centred x=y=0, z rising from 0. */
export function domainBoundsM(
  grid: StepInfo["grid"] | null | undefined,
): DomainBoundsM | null {
  if (!grid || grid.geometryKind === "index") return null;
  const { nr, nth, nz, drM, dzM } = grid;
  if (!nr || !nz || !drM || !dzM) return null;

  const bounds = grid.physicalBoundsM;
  if (bounds) {
    return {
      lower: [bounds.lower[0], bounds.lower[1], bounds.lower[2]],
      upper: [bounds.upper[0], bounds.upper[1], bounds.upper[2]],
    };
  }

  const xExtM = (nr - 1) * drM;
  const yExtM = nth ? (nth - 1) * cartesianDyMeters(grid, drM) : xExtM;
  const zExtM = (nz - 1) * dzM;
  return {
    lower: [-0.5 * xExtM, -0.5 * yExtM, 0],
    upper: [0.5 * xExtM, 0.5 * yExtM, zExtM],
  };
}

/** Index rulers use source sample positions without creating physical bounds. */
export function domainAxisBounds(
  grid: StepInfo["grid"] | null | undefined,
): DomainBoundsM | null {
  if (grid?.geometryKind !== "index") return domainBoundsM(grid);
  const counts = [grid.nr, grid.nth, grid.nz];
  if (counts.some(count => count == null || !Number.isSafeInteger(count) || count < 1)) return null;
  const lower = grid.validIndexRange?.lower ?? [0, 0, 0];
  const upper = grid.validIndexRange?.upper ?? counts.map(count => count! - 1);
  if (lower.length !== 3 || upper.length !== 3 || lower.some((value, axis) =>
    !Number.isSafeInteger(value) || !Number.isSafeInteger(upper[axis])
      || value < 0 || upper[axis] < value || upper[axis] >= counts[axis]!)) return null;
  return { lower: [...lower], upper: [...upper] } as DomainBoundsM;
}

export function describeDomainGrid(
  grid: StepInfo["grid"] | null | undefined,
): { caption: string; notice: string | null } | null {
  if (!grid) return null;
  if (grid.geometryKind === "index") {
    const counts = [grid.nr, grid.nth, grid.nz];
    const count = (value: number | undefined) => value != null && Number.isSafeInteger(value) && value > 0
      ? String(value) : "unavailable";
    return { caption: counts.every(value => value != null && Number.isSafeInteger(value) && value > 0)
      ? `${counts.join("×")} samples`
      : counts.map((value, axis) => `${["i", "j", "k"][axis]} ${count(value)}`).join(" · "),
    notice: "physical spacing unavailable" };
  }
  const bounds = domainBoundsM(grid);
  if (!bounds) return null;
  const formatKm = (axis: number) => {
    const value = (bounds.upper[axis] - bounds.lower[axis]) / 1000;
    return value >= 10 ? value.toFixed(0) : value.toFixed(1);
  };
  return { caption: `x ${formatKm(0)} · y ${formatKm(1)} · z ${formatKm(2)} km`, notice: null };
}

/** 1/2/5 × 10^n step (m) so labels land on round kilometre values. */
export function niceTickStepM(spanM: number, targetCount: number): number {
  if (!Number.isFinite(spanM) || spanM <= 0) return 0;
  if (!Number.isFinite(targetCount) || targetCount < 1) return 0;
  const rough = spanM / targetCount;
  const magnitude = 10 ** Math.floor(Math.log10(rough));
  const normalized = rough / magnitude;
  const nice = normalized <= 1 ? 1 : normalized <= 2 ? 2 : normalized <= 5 ? 5 : 10;
  return nice * magnitude;
}

/** Round tick values in TRUE metres covering [lowerM, upperM]. */
export function axisTickValuesM(
  lowerM: number,
  upperM: number,
  targetCount = 5,
): number[] {
  if (!Number.isFinite(lowerM) || !Number.isFinite(upperM)) return [];
  const span = upperM - lowerM;
  const step = niceTickStepM(span, targetCount);
  if (step <= 0) return [];

  // Tolerate float drift so a tick sitting exactly on a domain face is kept
  // rather than dropped by a last-bit comparison.
  const tolerance = step * 1e-6;
  const first = Math.ceil((lowerM - tolerance) / step) * step;
  const values: number[] = [];
  for (let v = first; v <= upperM + tolerance; v += step) {
    // Re-round: repeated addition accumulates error that shows up in labels.
    values.push(Math.round(v / step) * step);
  }
  return values;
}

export function axisTickValuesIndex(lower: number, upper: number, targetCount = 5): number[] {
  if (![lower, upper].every(Number.isSafeInteger)) return [];
  const proposed = niceTickStepM(upper - lower, targetCount);
  if (proposed <= 0) return [];
  const step = Math.max(1, proposed);
  const values: number[] = [];
  for (let value = Math.ceil(lower / step) * step; value <= upper; value += step) values.push(value);
  return values;
}

/** Format a true-metre tick as kilometres, without trailing noise. */
export function formatAxisTickKm(valueM: number): string {
  const km = valueM / 1000;
  if (!Number.isFinite(km)) return "";
  if (km === 0) return "0";
  const abs = Math.abs(km);
  const decimals = abs >= 10 ? 0 : abs >= 1 ? 1 : 2;
  return Number(km.toFixed(decimals)).toString();
}

export type DomainAxis = "x" | "y" | "z";

export interface DomainAxisTick {
  axis: DomainAxis;
  /** Source coordinate: metres on physical grids, sample index on index grids. */
  valueM: number;
  /** Tick mark start, in DISPLAY metres. */
  from: [number, number, number];
  /** Tick mark end, in DISPLAY metres. */
  to: [number, number, number];
  /** Anchor for the DOM label, in DISPLAY metres, clear of the mark. */
  labelAt: [number, number, number];
  /** Label text in the declared axis unit. */
  label: string;
}

export interface DomainAxisTicks {
  ticks: DomainAxisTick[];
  /** Axis name anchors, in DISPLAY metres, at each axis midpoint. */
  axisLabels: { axis: DomainAxis; at: [number, number, number]; label: string }[];
}

/** Fraction of the largest display extent used for a tick mark's length. */
const TICK_MARK_FRACTION = 0.02;
/** Label sits this many mark-lengths beyond the mark, clear of the box. */
const LABEL_STANDOFF = 2.2;

/** Which box edges carry ticks, chosen from the camera to avoid overlap with storm geometry. */
export interface AxisCorners {
  /** Run the x ticks along the upper-y face rather than the lower-y face. */
  xAtUpperY: boolean;
  /** Run the y ticks along the upper-x face rather than the lower-x face. */
  yAtUpperX: boolean;
  /** Vertical edge for the z ticks, as [atUpperX, atUpperY]. */
  zCorner: [boolean, boolean];
}

export const DEFAULT_AXIS_CORNERS: AxisCorners = {
  xAtUpperY: false,
  yAtUpperX: false,
  zCorner: [false, false],
};

/** Place horizontal rulers on the face nearest the camera. */
export function chooseHorizontalCorners(
  bounds: DomainBoundsM,
  cameraX: number,
  cameraY: number,
): { xAtUpperY: boolean; yAtUpperX: boolean } {
  const centreX = 0.5 * (bounds.lower[0] + bounds.upper[0]);
  const centreY = 0.5 * (bounds.lower[1] + bounds.upper[1]);
  return {
    xAtUpperY: Number.isFinite(cameraY) ? cameraY > centreY : false,
    yAtUpperX: Number.isFinite(cameraX) ? cameraX > centreX : false,
  };
}

/** The four vertical edges, as [atUpperX, atUpperY] corner flags. */
export const VERTICAL_EDGE_CORNERS: ReadonlyArray<[boolean, boolean]> = [
  [false, false],
  [true, false],
  [true, true],
  [false, true],
];

/** Pick the vertical edge projecting furthest to the side (leftmost screen X). */
export function chooseVerticalEdge(
  screenXs: ReadonlyArray<number | null>,
): [boolean, boolean] {
  let best = -1;
  let bestX = Infinity;
  for (let i = 0; i < screenXs.length && i < VERTICAL_EDGE_CORNERS.length; i++) {
    const x = screenXs[i];
    if (x == null || !Number.isFinite(x)) continue;
    if (x < bestX) {
      bestX = x;
      best = i;
    }
  }
  return best >= 0 ? VERTICAL_EDGE_CORNERS[best] : DEFAULT_AXIS_CORNERS.zCorner;
}

/** Tick marks and labels for three domain axes in display coordinates (corner frame, not full cage). */
export function buildDomainAxisTicks(
  grid: StepInfo["grid"] | null | undefined,
  scale: SceneScale,
  corners: AxisCorners = DEFAULT_AXIS_CORNERS,
  targetCount = 5,
): DomainAxisTicks | null {
  const bounds = domainAxisBounds(grid);
  if (!bounds) return null;
  const indexOnly = grid?.geometryKind === "index";
  if (!Number.isFinite(scale.xy) || !Number.isFinite(scale.z)) return null;

  const toDisplay = (p: [number, number, number]): [number, number, number] =>
    [p[0] * scale.xy, p[1] * scale.xy, p[2] * scale.z];

  const spans = [0, 1, 2].map((i) =>
    Math.abs(toDisplay(bounds.upper)[i] - toDisplay(bounds.lower)[i]));
  const markLen = Math.max(...spans) * TICK_MARK_FRACTION;
  if (!(markLen > 0)) return null;

  const pick = (axisIndex: 0 | 1, upper: boolean) =>
    upper ? bounds.upper[axisIndex] : bounds.lower[axisIndex];
  // Marks point away from the box, along whichever side each ruler sits on.
  const sign = (upper: boolean) => (upper ? 1 : -1);

  // x/y rulers on floor; z ruler climbs a vertical edge.
  const [zAtUpperX, zAtUpperY] = corners.zCorner;
  const layout: Record<DomainAxis, {
    /** Fixed coordinates of the edge, by axis index. */
    base: [number, number, number];
    /** Outward direction of the tick marks, in display space. */
    direction: [number, number, number];
  }> = {
    x: {
      base: [0, pick(1, corners.xAtUpperY), bounds.lower[2]],
      direction: [0, sign(corners.xAtUpperY), 0],
    },
    y: {
      base: [pick(0, corners.yAtUpperX), 0, bounds.lower[2]],
      direction: [sign(corners.yAtUpperX), 0, 0],
    },
    z: {
      base: [pick(0, zAtUpperX), pick(1, zAtUpperY), 0],
      direction: [sign(zAtUpperX), sign(zAtUpperY), 0],
    },
  };

  const ticks: DomainAxisTick[] = [];
  const axes: DomainAxis[] = ["x", "y", "z"];
  const place = (
    axis: DomainAxis,
    axisIndex: number,
    coordinateM: number,
  ) => {
    const { base, direction } = layout[axis];
    const point: [number, number, number] = [base[0], base[1], base[2]];
    point[axisIndex] = coordinateM;
    const from = toDisplay(point);
    // Normalize direction so diagonal z mark has the same length as x/y marks.
    const length = Math.hypot(direction[0], direction[1], direction[2]) || 1;
    const along = (factor: number): [number, number, number] => [
      from[0] + (direction[0] / length) * markLen * factor,
      from[1] + (direction[1] / length) * markLen * factor,
      from[2] + (direction[2] / length) * markLen * factor,
    ];
    return { from, along };
  };

  for (let axisIndex = 0; axisIndex < axes.length; axisIndex++) {
    const axis = axes[axisIndex];
    const values = (indexOnly ? axisTickValuesIndex : axisTickValuesM)(
      bounds.lower[axisIndex],
      bounds.upper[axisIndex],
      targetCount,
    );
    for (const valueM of values) {
      // Drop z tick at floor to avoid triple-number stack at the corner.
      if (axis === "z" && valueM === bounds.lower[2]) continue;
      const { from, along } = place(axis, axisIndex, valueM);
      ticks.push({
        axis,
        valueM,
        from,
        to: along(1),
        labelAt: along(LABEL_STANDOFF),
        label: indexOnly ? String(valueM) : formatAxisTickKm(valueM),
      });
    }
  }

  const axisLabels = axes.map((axis, axisIndex) => {
    const midpoint = 0.5 * (bounds.lower[axisIndex] + bounds.upper[axisIndex]);
    const { along } = place(axis, axisIndex, midpoint);
    return {
      axis,
      at: along(LABEL_STANDOFF * 2.4),
      label: indexOnly ? `${["i", "j", "k"][axisIndex]} (index)` : `${axis} (km)`,
    };
  });

  return { ticks, axisLabels };
}
