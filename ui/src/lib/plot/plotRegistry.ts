import type { SliceAxis } from "@/lib/platform/npy";

/*
 * Plot registry — taxonomy + data-aware gating. Pure — no React, no Tauri.
 * Unit-tested in scripts/test_plot_registry.ts. Gating = pure functions of available-field set.
 */

export type PlotType =
  | "slice"
  | "vector"
  | "composite"
  | "timeseries"
  | "timeheight"
  | "swath"
  | "vertical"
  | "radial"
  | "hodograph"
  | "skewt"
  | "volume"
  | "isosurface"
  | "streamlines"
  | "supercell"
  | "trajectories"
  | "trajectory_budget"
  | "python";

export type PlotCategory =
  | "plan-views"
  | "cross-sections"
  | "soundings"
  | "radar"
  | "volumetric"
  | "time-series"
  | "swaths"
  | "trajectories"
  | "scripting";

/** Sentinel marking that a run wrote trajectories.npy; injected into the available-field set for gating. */
export const TRAJECTORY_CAPABILITY = "__trajectories__";

/** Internal data-shape capabilities injected by Plot Studio after every field
 * has been resolved through the dataset adapter's registered rank metadata.
 * They are deliberately not field names: a rank-2 column product must not make
 * a rank-3 renderer appear available merely because the run contains data. */
export const PLANE_SCALAR_CAPABILITY = "__plane_scalar_field__";
export const VOLUME_SCALAR_CAPABILITY = "__volume_scalar_field__";
export const NATIVE_REDUCTION_CAPABILITY = "__native_reduction_field__";
export const NATIVE_VOLUME_REDUCTION_CAPABILITY = "__native_volume_reduction_field__";
export const PLAN_VECTOR_CAPABILITY = "__plan_vector_pair__";
export const CROSS_SECTION_VECTOR_CAPABILITY = "__cross_section_vector_pair__";

/**
 * AND of clauses; each clause is an OR of field names. "anyField" = any field will do.
 * e.g. [["u"],["v"],["w"]] = needs all; [["reflectivity_dbz","radar"]] = needs any.
 */
export type FieldReq = ReadonlyArray<ReadonlyArray<string>> | "anyField";

/** querySelector the movie recorder uses to find this plot's live canvas. */
export type RecorderSelector =
  | ".plot-canvas"
  | ".volume-render__canvas"
  | ".volume-render__canvas, .plot-canvas";

export interface PlotTypeMeta {
  value: PlotType;
  /** Short label shown in the picker. */
  label: string;
  /** Hover tooltip. */
  hint: string;
  /** Fields the run must export for this renderer to produce anything. */
  requires: FieldReq;
  /** Animates across the step slider (drives whether the frame bar shows). */
  perFrame: boolean;
  /** 2D Plotly (.plot-canvas) vs 3D VTK (.volume-render__canvas) canvas. */
  recorderSelector: RecorderSelector;
  /**
   * The diagram is read vertically and must not be stretched to the full width
   * of a workstation window.
   *
   * A sounding plots height or pressure up the y axis over a narrow x range.
   * Given a 1360x320 letterbox it becomes unreadable in two different ways: a
   * Skew-T's isotherms flatten, so the 45-degree skew the diagram is named for
   * no longer holds and slopes cannot be judged by eye; and an equal-aspect
   * hodograph collapses to a small square against one edge with the rest of the
   * panel empty. Both are the same defect — a portrait figure in a landscape
   * frame — so the layout caps their width and centres them instead.
   */
  portrait?: boolean;
}

// Field requirements verified against each widget. Supercell gates on condensate only (other layers self-disable).
export const PLOT_TYPE_META: ReadonlyArray<PlotTypeMeta> = [
  {
    value: "slice",
    label: "Slice",
    hint: "2D heatmap across a fixed axis",
    requires: [[PLANE_SCALAR_CAPABILITY]],
    perFrame: true,
    recorderSelector: ".plot-canvas",
  },
  {
    value: "vector",
    label: "Vector field",
    hint: "Quiver of one physically matched vector family on a slice",
    requires: [[PLAN_VECTOR_CAPABILITY, CROSS_SECTION_VECTOR_CAPABILITY]],
    perFrame: true,
    recorderSelector: ".plot-canvas",
  },
  {
    value: "composite",
    label: "Composite",
    hint: "Filled field + contour overlays — the canonical supercell figure",
    requires: [[PLANE_SCALAR_CAPABILITY]],
    perFrame: true,
    recorderSelector: ".plot-canvas",
  },
  {
    value: "timeseries",
    label: "Time-series",
    hint: "Reducer vs t for one field",
    requires: [[NATIVE_REDUCTION_CAPABILITY]],
    perFrame: false,
    recorderSelector: ".plot-canvas",
  },
  {
    value: "timeheight",
    label: "Time-height",
    hint: "Hovmöller — per-level reducer vs time (z by default)",
    requires: [[NATIVE_VOLUME_REDUCTION_CAPABILITY]],
    perFrame: false,
    recorderSelector: ".plot-canvas",
  },
  {
    value: "swath",
    label: "Swath",
    hint: "Per-cell max/min over the whole run — storm & rotation tracks",
    requires: [[PLANE_SCALAR_CAPABILITY]],
    perFrame: false,
    recorderSelector: ".plot-canvas",
  },
  {
    value: "vertical",
    label: "Vertical",
    hint: "Field vs height (z) at a fixed point",
    requires: [[VOLUME_SCALAR_CAPABILITY]],
    perFrame: true,
    recorderSelector: ".plot-canvas",
    portrait: true,
  },
  {
    value: "radial",
    label: "X profile",
    hint: "Field along Cartesian x at fixed y and z",
    requires: [[VOLUME_SCALAR_CAPABILITY]],
    perFrame: true,
    recorderSelector: ".plot-canvas",
  },
  {
    value: "hodograph",
    label: "Hodograph",
    hint: "u(z) vs v(z)",
    requires: [["u"], ["v"]],
    perFrame: true,
    recorderSelector: ".plot-canvas",
    portrait: true,
  },
  {
    value: "skewt",
    label: "Skew-T",
    hint: "Skew-T log-P sounding",
    requires: [["p"], ["theta", "temperature"]],
    perFrame: true,
    recorderSelector: ".plot-canvas",
    portrait: true,
  },
  {
    value: "volume",
    label: "Volume",
    hint: "3D volume render of any scalar field",
    requires: [[VOLUME_SCALAR_CAPABILITY]],
    perFrame: true,
    recorderSelector: ".volume-render__canvas",
  },
  {
    value: "isosurface",
    label: "Isosurface",
    hint: "Marching-cubes surface at a threshold",
    requires: [[VOLUME_SCALAR_CAPABILITY]],
    perFrame: true,
    recorderSelector: ".volume-render__canvas",
  },
  {
    value: "streamlines",
    label: "Streamlines",
    hint: "Streamlines from a seed box (velocity or vorticity)",
    requires: [["u"], ["v"], ["w"]],
    perFrame: true,
    recorderSelector: ".volume-render__canvas",
  },
  {
    value: "supercell",
    label: "Rendered supercell",
    hint: "Composite storm render: cloud volume + dBZ floor + updraft",
    requires: [["cloud_condensate", "total_condensate"]],
    perFrame: true,
    recorderSelector: ".volume-render__canvas",
  },
  {
    value: "trajectories",
    label: "3D paths",
    hint: "3D Lagrangian parcel paths (run the sim with --trajectories)",
    requires: [[TRAJECTORY_CAPABILITY]],
    perFrame: false,
    recorderSelector: ".volume-render__canvas",
  },
  {
    value: "trajectory_budget",
    label: "Vorticity budget",
    hint: "Tilting / stretching / baroclinic generation along a parcel path — the tornadogenesis-mechanism plot",
    requires: [[TRAJECTORY_CAPABILITY]],
    perFrame: false,
    recorderSelector: ".plot-canvas",
  },
  {
    value: "python",
    label: "Python",
    hint: "Pyodide REPL — numpy + matplotlib, full PyData stack",
    requires: "anyField",
    perFrame: true,
    recorderSelector: ".plot-canvas",
  },
];

export interface CategoryMeta {
  value: PlotCategory;
  label: string;
  /** Hover tooltip for the category pill. */
  hint: string;
  /** Renderers in this category's picker. Empty for product-driven categories (radar = presets only). */
  plotTypes: ReadonlyArray<PlotType>;
  /** Default slice axis for slice/vector in this category (plan=z, cross-section=r). */
  defaultSliceAxis?: SliceAxis;
}

export const CATEGORY_META: ReadonlyArray<CategoryMeta> = [
  {
    value: "plan-views",
    label: "Plan views",
    hint: "Horizontal cross-sections at a height",
    plotTypes: ["slice", "vector", "composite"],
    defaultSliceAxis: 2,
  },
  {
    value: "cross-sections",
    label: "Cross-sections",
    hint: "Vertical slices through the storm",
    plotTypes: ["slice", "vector", "composite"],
    // θ-slice → r-z plane (Cartesian: y-slice → x-z plane): the classic
    // vertical cross-section through the updraft.
    defaultSliceAxis: 1,
  },
  {
    value: "soundings",
    label: "Soundings",
    hint: "Skew-T, hodograph, vertical & Cartesian x profiles",
    plotTypes: ["skewt", "hodograph", "vertical", "radial"],
  },
  {
    value: "radar",
    label: "Radar",
    hint: "Simulated reflectivity products",
    plotTypes: [],
  },
  {
    value: "volumetric",
    label: "3D",
    hint: "Volume, isosurface, streamlines, rendered supercell, parcel paths",
    // Parcel paths are listed here as well as under Trajectories: this is the tab
    // users look on for anything rendered in 3D. A type may appear in several
    // categories — nothing derives a category from a plot type.
    plotTypes: ["volume", "isosurface", "streamlines", "supercell", "trajectories"],
  },
  {
    value: "time-series",
    label: "Time series",
    hint: "Domain reducer vs time, and time-height (Hovmöller)",
    plotTypes: ["timeseries", "timeheight"],
  },
  {
    value: "swaths",
    label: "Swaths",
    hint: "Accumulated max/min over the whole run (storm & rotation tracks)",
    plotTypes: ["swath"],
  },
  {
    value: "trajectories",
    label: "Trajectories",
    hint: "Lagrangian parcel paths and vorticity budgets (run the sim with --trajectories)",
    plotTypes: ["trajectories", "trajectory_budget"],
  },
  {
    value: "scripting",
    label: "Scripting",
    hint: "Python authored plots",
    plotTypes: ["python"],
  },
];

/** Gating-relevant subset of a plot preset. PlotPreset (plotPresets.ts)
 *  structurally satisfies this, so the registry stays free of any import of
 *  the preset list (keeps it pure and avoids a module cycle). */
export interface PresetGatingInfo {
  category: PlotCategory;
  /** Additional categories this preset also appears under (one id, shared gating). */
  secondaryCategories?: readonly PlotCategory[];
  /** Primary field the preset plots. */
  field: string;
  /** Explicit requirement override (e.g. radar's reflectivity_dbz OR radar).
   *  Defaults to "the primary field must be present". */
  requires?: FieldReq;
}

const META_BY_TYPE = new Map<PlotType, PlotTypeMeta>(
  PLOT_TYPE_META.map((m): [PlotType, PlotTypeMeta] => [m.value, m]),
);

/** Metadata for a plot type. Throws on an unknown type — every PlotType
 *  union member must have exactly one PLOT_TYPE_META row (asserted in tests). */
export function metaFor(t: PlotType): PlotTypeMeta {
  const m = META_BY_TYPE.get(t);
  if (!m) throw new Error(`No PLOT_TYPE_META row for plot type "${t}"`);
  return m;
}

const CATEGORY_BY_VALUE = new Map<PlotCategory, CategoryMeta>(
  CATEGORY_META.map((c): [PlotCategory, CategoryMeta] => [c.value, c]),
);

export function categoryMeta(c: PlotCategory): CategoryMeta {
  const m = CATEGORY_BY_VALUE.get(c);
  if (!m) throw new Error(`No CATEGORY_META row for category "${c}"`);
  return m;
}

/** Plot types that animate across the step slider; frame bar shows iff active type is in this set. */
export const PER_FRAME_PLOTS: PlotType[] = PLOT_TYPE_META.filter(
  (m) => m.perFrame,
).map((m) => m.value);

/**
 * Plot types that call `onFrameDisplayChange` with the frame they have actually drawn.
 *
 * Playback and the movie recorder gate on these reports, so a type absent here cannot be waited on:
 * it advances on the timer alone. Adding a type here without wiring the callback stalls playback
 * until the gate's hold bound expires.
 */
const FRAME_IDENTITY_REPORTERS: ReadonlySet<PlotType> = new Set<PlotType>([
  "slice",
  "composite",
  "vector",
  "volume",
  "isosurface",
  "streamlines",
  "supercell",
]);

export function reportsFrameIdentity(t: PlotType): boolean {
  return FRAME_IDENTITY_REPORTERS.has(t);
}

/* ---------- field availability + gating predicate ---------- */

/** Union of all field names across loaded steps (gating on union prevents flicker when scrubbing). */
export function availableFields(
  steps: ReadonlyArray<{ fields: ReadonlyArray<string> }>,
): Set<string> {
  const out = new Set<string>();
  for (const step of steps) {
    for (const f of step.fields) out.add(f);
  }
  return out;
}

/** True when the run (its available field set) satisfies `req`. */
export function reqSatisfied(req: FieldReq, available: ReadonlySet<string>): boolean {
  if (req === "anyField") return available.size > 0;
  return req.every((clause) => clause.some((name) => available.has(name)));
}

/** Unsatisfied AND clauses (each still an OR list); `anyField` yields one empty clause when no fields exist. */
export function missingRequirementClauses(
  req: FieldReq,
  available: ReadonlySet<string>,
): ReadonlyArray<ReadonlyArray<string>> {
  if (req === "anyField") return available.size > 0 ? [] : [[]];
  return req.filter((clause) => !clause.some((name) => available.has(name)));
}

export function isPlotAvailable(t: PlotType, available: ReadonlySet<string>): boolean {
  return reqSatisfied(metaFor(t).requires, available);
}

/** A preset's requirement: its explicit `requires`, else "primary field present". */
export function presetRequirement(p: PresetGatingInfo): FieldReq {
  return p.requires ?? [[p.field]];
}

export function isPresetAvailable(
  p: PresetGatingInfo,
  available: ReadonlySet<string>,
): boolean {
  return reqSatisfied(presetRequirement(p), available);
}

/** Renderers in a category that the run supports, in declared order. */
export function visiblePlotTypes(
  category: PlotCategory,
  available: ReadonlySet<string>,
): PlotTypeMeta[] {
  return categoryMeta(category)
    .plotTypes.map(metaFor)
    .filter((m) => reqSatisfied(categoryRequirement(category, m), available));
}

/** A rank-2 field is a plan map, not a vertical cross-section. Vector pairs are
 * likewise plane-specific: u/v qualifies a horizontal slice, while u/w
 * qualifies the default x/z cross-section. */
function categoryRequirement(category: PlotCategory, meta: PlotTypeMeta): FieldReq {
  if (category === "cross-sections") {
    if (meta.value === "slice" || meta.value === "composite") {
      return [[VOLUME_SCALAR_CAPABILITY]];
    }
    if (meta.value === "vector") return [[CROSS_SECTION_VECTOR_CAPABILITY]];
  }
  if (category === "plan-views" && meta.value === "vector") {
    return [[PLAN_VECTOR_CAPABILITY]];
  }
  return meta.requires;
}

/** Presets assigned to a category (regardless of availability). */
export function presetsForCategory<T extends PresetGatingInfo>(
  category: PlotCategory,
  presets: ReadonlyArray<T>,
): T[] {
  return presets.filter(
    (p) => p.category === category || p.secondaryCategories?.includes(category),
  );
}

/** Presets in a category that the run supports. */
export function visiblePresets<T extends PresetGatingInfo>(
  category: PlotCategory,
  available: ReadonlySet<string>,
  presets: ReadonlyArray<T>,
): T[] {
  return presetsForCategory(category, presets).filter((p) =>
    isPresetAvailable(p, available),
  );
}

/** True if at least one renderer or preset in the category is supported by the run. */
export function categoryHasContent(
  category: PlotCategory,
  available: ReadonlySet<string>,
  presets: ReadonlyArray<PresetGatingInfo>,
): boolean {
  return (
    visiblePlotTypes(category, available).length > 0 ||
    visiblePresets(category, available, presets).length > 0
  );
}

/** Categories with any supported content, in declared order. */
export function visibleCategories(
  available: ReadonlySet<string>,
  presets: ReadonlyArray<PresetGatingInfo>,
): CategoryMeta[] {
  return CATEGORY_META.filter((c) =>
    categoryHasContent(c.value, available, presets),
  );
}

/** First category with content (for the run-change fallback), or null when
 *  the run exports nothing plottable. */
export function firstAvailableCategory(
  available: ReadonlySet<string>,
  presets: ReadonlyArray<PresetGatingInfo>,
): PlotCategory | null {
  return visibleCategories(available, presets)[0]?.value ?? null;
}

/** First supported renderer in a category (for the run-change fallback),
 *  or null (e.g. a product-only category like radar, or none supported). */
export function firstAvailablePlotType(
  category: PlotCategory,
  available: ReadonlySet<string>,
): PlotType | null {
  return visiblePlotTypes(category, available)[0]?.value ?? null;
}
