/* Shared colormap definitions. Domain-specific palettes (NWS dBZ, ColdPool) live here. */

import type { FieldUnits } from "@/lib/science/fieldUnits";

/** Plotly's colorscale-array shape: [t, rgb-string] with t ∈ [0,1]. */
export type PlotlyColorscaleArray = ReadonlyArray<[number, string]>;

/* CSS gradient stops (approx) for the 3D scalar colorbar overlay (VTK draws no legend itself). */
const COLORBAR_GRADIENT_STOPS: Record<string, string> = {
  Inferno: "#000004, #420a68, #932667, #dd513a, #fca50a, #fcffa4",
  "Black-Body Radiation": "#000000, #7a0000, #d73027, #fc8d59, #fee08b, #ffffff",
  Viridis: "#440154, #3b528b, #21918c, #5ec962, #fde725",
  Plasma: "#0d0887, #6a00a8, #b12a90, #e16462, #fca636, #f0f921",
  "Cool to Warm": "#3b4cc0, #b4b4b4, #b40426",
  RdBu: "#053061, #4393c3, #f7f7f7, #d6604d, #67001f",
  ColdPool: "#053061, #4393c3, #f7f7f7, #d6604d, #67001f",
  Greys: "#ffffff, #000000",
};

/** A CSS `linear-gradient(to top, ...)` for the named colormap (bottom = low). */
export function cssColorbarGradient(name: string): string {
  if (name === "dBZ") return `linear-gradient(to top, ${dbzCssStops(DBZ_VTK_STOPS)})`;
  if (name === "dBZ volume") {
    return `linear-gradient(to top, ${dbzCssStops(DBZ_VTK_STOPS_VOLUME)})`;
  }
  if (name === "dBZ floor") return `linear-gradient(to top, ${dbzFloorCssStops()})`;
  const stops = COLORBAR_GRADIENT_STOPS[name] ?? COLORBAR_GRADIENT_STOPS.Inferno;
  return `linear-gradient(to top, ${stops})`;
}

/* NWS reflectivity ramp [-30, 75] dBZ. Anchors: -30=grey, 0=blue, 10=green, 30=yellow, 50=red, 70=magenta, 75=white. */
export const DBZ_RANGE: readonly [number, number] = [-30, 75];

/**
 * True for palettes anchored to physical values (currently only "dBZ").
 * Callers must pin DBZ_RANGE when this returns true; fitting the range moves every anchor.
 */
export function usesFixedAnchorRange(colormap: ExtendedColorscale): boolean {
  return colormap === "dBZ";
}

/**
 * True if the field should use the dBZ palette + [-30, 75] range.
 * Driven by unit from manifest; falls back to name heuristic. Excludes the linear `radar` field (mm^6/m^3).
 */
export function isDbzField(field: string | null, units: FieldUnits): boolean {
  if (!field) return false;
  const unit = units?.[field];
  if (unit != null && unit !== "") return unit === "dBZ";
  const f = field.toLowerCase();
  return f !== "radar" && (f.includes("dbz") || f.includes("reflectiv"));
}

function dBzToT(dbz: number): number {
  return (dbz - DBZ_RANGE[0]) / (DBZ_RANGE[1] - DBZ_RANGE[0]);
}

export const DBZ_PHYSICAL_STOPS = [
  { scalar: -30, rgb: [192, 192, 192] },
  { scalar: -15, rgb: [96, 96, 96] },
  { scalar: -5, rgb: [40, 80, 140] },
  { scalar: 0, rgb: [0, 160, 255] },
  { scalar: 5, rgb: [0, 200, 200] },
  { scalar: 10, rgb: [0, 200, 80] },
  { scalar: 20, rgb: [80, 210, 40] },
  { scalar: 30, rgb: [255, 255, 0] },
  { scalar: 40, rgb: [255, 160, 0] },
  { scalar: 50, rgb: [255, 0, 0] },
  { scalar: 60, rgb: [200, 0, 80] },
  { scalar: 70, rgb: [255, 0, 255] },
  { scalar: 75, rgb: [255, 255, 255] },
] as const;

export const DBZ_COLORSCALE: PlotlyColorscaleArray = DBZ_PHYSICAL_STOPS.map(
  ({ scalar, rgb }) => [dBzToT(scalar), `rgb(${rgb.join(",")})`],
);

/* ColdPool: symmetric blue→white→red, white at zero. RdBu has off-centre grey midpoint. */
export const COLD_POOL_COLORSCALE: PlotlyColorscaleArray = [
  [0.0, "rgb(33,102,172)"],
  [0.25, "rgb(103,169,207)"],
  [0.5, "rgb(247,247,247)"],
  [0.75, "rgb(239,138,98)"],
  [1.0, "rgb(178,24,43)"],
];

/* ---------- Plotly colorscale resolution ---------- */

/** Built-in Plotly colorscale names exposed in the plot pickers. */
export type BuiltinColorscale = "Viridis" | "Plasma" | "Inferno" | "RdBu" | "Greys";
/** Built-ins plus the domain palettes defined above. */
export type ExtendedColorscale = BuiltinColorscale | "dBZ" | "ColdPool";

/** Clone readonly colorscale to mutable [number, string][] for Plotly's TS types. */
export type PlotlyColorscaleArrayMutable = Array<[number, string]>;

export function cloneScale(
  src: ReadonlyArray<readonly [number, string]>,
): PlotlyColorscaleArrayMutable {
  return src.map(([t, c]) => [t, c]);
}

/** Resolve an extended colormap id to a Plotly colorscale (string or cloned array). */
export function resolveColorscale(
  c: ExtendedColorscale,
): string | PlotlyColorscaleArrayMutable {
  if (c === "dBZ") return cloneScale(DBZ_COLORSCALE);
  if (c === "ColdPool") return cloneScale(COLD_POOL_COLORSCALE);
  return c;
}

/** True for diverging palettes whose neutral midpoint must land at zero; callers use symmetric range. */
export function isDivergingColorscale(c: ExtendedColorscale): boolean {
  return c === "RdBu" || c === "ColdPool";
}

/** The constant represented by a finite min=max range, otherwise null. */
export function uniformRangeValue(
  range: { min: number; max: number } | null,
): number | null {
  if (range == null || !Number.isFinite(range.min) || !Number.isFinite(range.max)) return null;
  if (Math.abs(range.max - range.min) > 1e-12) return null;
  return (range.min + range.max) / 2;
}

/** Give a constant field a small honest colour interval instead of asking a
 * renderer to infer a scale from min=max. The annotation layer separately
 * discloses that the data themselves remain uniform. */
export function padDegenerateRange(
  range: { min: number; max: number } | null,
): { min: number; max: number } {
  if (range == null) return { min: 0, max: 1 };
  const center = uniformRangeValue(range);
  if (center == null) return range;
  const padding = Math.max(0.5, Math.abs(center) * 1e-3);
  return { min: center - padding, max: center + padding };
}

/** Plotly range fragment that preserves the physical meaning of a colorscale. */
export function plotlyColorRange(
  colormap: ExtendedColorscale,
  range: { min: number; max: number } | null,
): { zmin?: number; zmax?: number } {
  if (usesFixedAnchorRange(colormap)) {
    return { zmin: DBZ_RANGE[0], zmax: DBZ_RANGE[1] };
  }
  if (range == null) return {};
  if (!isDivergingColorscale(colormap)) {
    const uniform = uniformRangeValue(range);
    if (uniform == null) return {};
    const padded = padDegenerateRange(range);
    return { zmin: padded.min, zmax: padded.max };
  }
  let magnitude = Math.max(Math.abs(range.min), Math.abs(range.max));
  if (!Number.isFinite(magnitude)) return {};
  if (magnitude <= 1e-12) magnitude = 0.5;
  return { zmin: -magnitude, zmax: magnitude };
}

/* ---------- VTK.js ColorTransferFunction helpers ---------- */

/** A single (scalar, r, g, b) anchor for vtkColorTransferFunction.addRGBPoint. */
export interface VtkColorStop {
  scalar: number;
  r: number;
  g: number;
  b: number;
}

/** NWS dBZ stops in [0..1] RGB + dBZ-scalar space for vtkColorTransferFunction.addRGBPoint(). */
export const DBZ_VTK_STOPS: ReadonlyArray<VtkColorStop> = DBZ_PHYSICAL_STOPS.map(
  ({ scalar, rgb }) => ({
    scalar,
    r: rgb[0] / 255,
    g: rgb[1] / 255,
    b: rgb[2] / 255,
  }),
);

/* Volume-rendering dBZ variant: channels compressed to [FLOOR, CEIL] so ray accumulation doesn't saturate. */
const DBZ_VOLUME_CHANNEL_FLOOR = 0.11;
const DBZ_VOLUME_CHANNEL_CEIL = 0.94;

function dbzVolumeChannel(value: number): number {
  return DBZ_VOLUME_CHANNEL_FLOOR
    + value * (DBZ_VOLUME_CHANNEL_CEIL - DBZ_VOLUME_CHANNEL_FLOOR);
}

export const DBZ_VTK_STOPS_VOLUME: ReadonlyArray<VtkColorStop> = DBZ_VTK_STOPS.map(
  (stop) => ({
    scalar: stop.scalar,
    r: dbzVolumeChannel(stop.r),
    g: dbzVolumeChannel(stop.g),
    b: dbzVolumeChannel(stop.b),
  }),
);

function dbzCssStops(stops: ReadonlyArray<VtkColorStop>): string {
  return stops.map((stop) => {
    const percentage = Number((dBzToT(stop.scalar) * 100).toFixed(4));
    const rgb = [stop.r, stop.g, stop.b].map((value) => Math.round(value * 255));
    return `rgb(${rgb.join(",")}) ${percentage}%`;
  }).join(", ");
}

function dbzFloorCssStops(): string {
  const clear = "rgb(14,16,19)";
  const visible = DBZ_VTK_STOPS.filter((stop) => stop.scalar >= 20);
  return [
    `${clear} 0%`,
    `${clear} ${Number((dBzToT(18) * 100).toFixed(4))}%`,
    dbzCssStops(visible),
  ].join(", ");
}
