/* Plot chrome label helpers. All stat/threshold labels carry units to prevent unit ambiguity. */

import { unitFor, type FieldUnits } from "@/lib/science/fieldUnits";
import { uniformRangeValue } from "@/lib/plot/plotColormaps";
import type { ValueRange } from "@/lib/plot/viewSync";

/** `Mean (dBZ)` when a unit is known, otherwise the bare statistic name. */
export function statisticLabel(statistic: string, field: string, units: FieldUnits): string {
  const unit = unitFor(field, units);
  return unit ? `${statistic} (${unit})` : statistic;
}

/** `Probability > 30 dBZ`; unit trails the threshold value, not the quantity name. */
export function thresholdLabel(threshold: number, field: string, units: FieldUnits): string {
  const unit = unitFor(field, units);
  return `Probability > ${threshold}${unit ? ` ${unit}` : ""}`;
}

/** Label that distinguishes an honest constant field from a failed solid fill. */
export function uniformFieldLabel(range: ValueRange | null, label: string): string | null {
  const value = uniformRangeValue(range);
  if (value == null) return null;
  const magnitude = Math.abs(value);
  const formatted = magnitude !== 0 && (magnitude < 0.001 || magnitude >= 10_000)
    ? value.toExponential(3)
    : Number(value.toPrecision(4)).toString();
  return `Uniform field · ${label} = ${formatted}`;
}

export function scalarColorRangeLabel(range: ValueRange | null, colorRange: ValueRange): string | null {
  if (range == null) return "No finite data in this slice";
  if (range.max <= colorRange.min) return "All visible values are at or below the color scale";
  if (range.min >= colorRange.max) return "All visible values are at or above the color scale";
  const span = colorRange.max - colorRange.min;
  const fraction = (range.max - range.min) / span;
  if (span > 0 && fraction > 0 && fraction < 0.05) {
    const percent = fraction * 100;
    return `Visible range uses ${percent < 0.1 ? "<0.1" : percent.toFixed(1)}% of the color scale`;
  }
  return null;
}

/* ---------- Contour layers that produced nothing ---------- */

/** Why an enabled contour layer drew nothing; reported distinctly because remedies differ. */
export type ContourSkipReason =
  /** Plane mismatch (e.g. column product on a vertical section). Fix: change slice. */
  | "plane"
  /** No contour levels set. */
  | "levels"
  /** Requested levels do not intersect the finite displayed field range. */
  | "range";

export interface ContourSkip {
  field: string;
  reason: ContourSkipReason;
}

/** One sentence per distinct skip reason. Returns [] when nothing was skipped. */
export function describeContourSkips(skips: readonly ContourSkip[], planeLabel: string): string[] {
  const fieldsFor = (reason: ContourSkipReason) =>
    skips.filter((skip) => skip.reason === reason).map((skip) => skip.field);

  const lines: string[] = [];
  const wrongPlane = fieldsFor("plane");
  if (wrongPlane.length > 0) {
    lines.push(`${wrongPlane.join(", ")} hidden — not on the fill's ${planeLabel} plane`);
  }
  const withoutLevels = fieldsFor("levels");
  if (withoutLevels.length > 0) {
    lines.push(`${withoutLevels.join(", ")} has no contour levels — enter one or more values`);
  }
  const outsideRange = fieldsFor("range");
  if (outsideRange.length > 0) {
    lines.push(`${outsideRange.join(", ")} has contour levels outside the displayed data range — adjust levels`);
  }
  return lines;
}
