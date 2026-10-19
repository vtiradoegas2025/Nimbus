import {
  isPlotAvailable,
  isPresetAvailable,
  type PlotType,
  type PlotCategory,
  type FieldReq,
} from "./plotRegistry";

/* Plot presets: one-click bundles of (plotType, field, colormap, rangeMode, slice). Gated on `requires`. */

export interface PlotPreset {
  id: string;
  /** Short label shown in the pill. */
  label: string;
  /** Hover tooltip explaining what this preset does. */
  description: string;
  /** Picker section this preset belongs to. */
  category: PlotCategory;
  /** Further sections it also appears under. See `PresetGatingInfo`. */
  secondaryCategories?: readonly PlotCategory[];
  /** Optional sub-section label within the category (e.g. "Dynamics",
   *  "Hydrometeors"). Presets sharing a subgroup render under one small label. */
  subgroup?: string;
  plotType: PlotType;
  /** Primary field name. Widget shows "field not exported" if missing. */
  field: string;
  /** Colormap id. SlicePlot / VolumeRender3D resolve these. */
  colormap?:
    | "Viridis"
    | "Plasma"
    | "Inferno"
    | "RdBu"
    | "Greys"
    | "dBZ"
    | "ColdPool";
  /** Range mode: "auto"=min/max+pad, "dBZ"=-30..75, "symmetric"=±max(|min|,|max|), "fixed"=fixedRange. */
  rangeMode?: "auto" | "dBZ" | "symmetric" | "fixed";
  symmetricMax?: number;
  fixedRange?: readonly [number, number];
  /** For slice plots: which axis to slice along. 0=r, 1=θ, 2=z. */
  sliceAxis?: 0 | 1 | 2;
  /** Slice index hint: "surface"=0, "mid"=nz/2, "top"=nz-1, or literal index. */
  sliceIndexHint?: "surface" | "mid" | "top" | number;
  /** Preferred height in km AGL (z-axis only); takes precedence over sliceIndexHint when spacing known. */
  sliceHeightKm?: number;
  volumeFocus?: {
    axis: 0 | 1 | 2;
    sourceIndex: number;
    coordinateM: number | null;
    centerM: readonly [number | null, number | null, number | null];
  };
  /** Gating override (e.g. OR-list). Defaults to primary `field` must be present. */
  requires?: FieldReq;
}

export const PLOT_PRESETS: ReadonlyArray<PlotPreset> = [
  /* ============================ Radar ============================ */
  {
    id: "surface-dbz",
    label: "Surface dBZ",
    description: "Reflectivity at the lowest level, NWS palette. Hook-echo / supercell signature.",
    category: "radar",
    plotType: "slice",
    field: "reflectivity_dbz",
    colormap: "dBZ",
    rangeMode: "dBZ",
    sliceAxis: 2,
    sliceIndexHint: "surface",
    requires: [["reflectivity_dbz", "radar"]],
  },
  {
    id: "composite-dbz",
    label: "Composite dBZ",
    description: "Max-over-column reflectivity, NWS palette. Best for spotting hook echo.",
    category: "radar",
    plotType: "slice",
    field: "composite_reflectivity",
    colormap: "dBZ",
    rangeMode: "dBZ",
    requires: [["composite_reflectivity"]],
  },
  {
    id: "volume-dbz",
    label: "3D dBZ volume",
    description: "Volume render of reflectivity. Shows full storm structure.",
    category: "radar",
    // Also a volumetric render: users look for it on the 3D tab, where the
    // bare Volume button otherwise opens on vorticity_magnitude.
    secondaryCategories: ["volumetric"],
    plotType: "volume",
    field: "reflectivity_dbz",
    colormap: "dBZ",
    rangeMode: "dBZ",
    requires: [["reflectivity_dbz", "radar"]],
  },

  /* ========================== Plan views ========================== */
  {
    id: "supercell-composite",
    label: "Supercell composite",
    description: "Shaded w + reflectivity & vorticity contours on one plan view — the canonical figure.",
    category: "plan-views",
    plotType: "composite",
    field: "w",
    colormap: "RdBu",
    rangeMode: "symmetric",
    sliceAxis: 2,
    sliceHeightKm: 1,
    requires: [["w"]],
  },
  {
    id: "midlevel-updraft",
    label: "Mid-level updraft",
    description: "Vertical velocity w at ~5 km AGL — the main updraft.",
    category: "plan-views",
    plotType: "slice",
    field: "w",
    colormap: "RdBu",
    rangeMode: "symmetric",
    sliceAxis: 2,
    sliceHeightKm: 5,
    requires: [["w"]],
  },
  {
    id: "low-vorticity",
    label: "Low-level ζ",
    description: "Vertical vorticity at ~875 m AGL — low-level mesocyclone / couplet.",
    category: "plan-views",
    plotType: "slice",
    field: "vorticity_z",
    colormap: "RdBu",
    rangeMode: "symmetric",
    sliceAxis: 2,
    sliceHeightKm: 0.875,
    requires: [["vorticity_z"]],
  },
  {
    id: "mid-vorticity",
    label: "Mid-level ζ",
    description: "Vertical vorticity at ~4875 m AGL — midlevel mesocyclone.",
    category: "plan-views",
    plotType: "slice",
    field: "vorticity_z",
    colormap: "RdBu",
    rangeMode: "symmetric",
    sliceAxis: 2,
    sliceHeightKm: 4.875,
    requires: [["vorticity_z"]],
  },
  {
    id: "updraft-helicity",
    label: "Updraft helicity",
    description: "Mid-level (2-5 km) updraft helicity — the standard mesocyclone rotation proxy.",
    category: "plan-views",
    plotType: "slice",
    field: "updraft_helicity",
    colormap: "RdBu",
    rangeMode: "symmetric",
    requires: [["updraft_helicity"]],
  },
  {
    id: "cold-pool",
    label: "Cold pool (θ')",
    description: "Surface θ' with diverging palette (±4 K). Outflow boundaries.",
    category: "plan-views",
    plotType: "slice",
    field: "theta_prime",
    colormap: "ColdPool",
    rangeMode: "symmetric",
    symmetricMax: 4,
    sliceAxis: 2,
    sliceIndexHint: "surface",
    requires: [["theta_prime"]],
  },
  {
    id: "p-prime",
    label: "Pressure pert. (p')",
    description: "Surface p' with diverging palette. Mesolow / dynamic pressure structure.",
    category: "plan-views",
    plotType: "slice",
    field: "p_prime",
    colormap: "ColdPool",
    rangeMode: "symmetric",
    sliceAxis: 2,
    sliceIndexHint: "surface",
    requires: [["p_prime"]],
  },
  {
    id: "lowlevel-buoyancy",
    label: "Low-level buoyancy",
    description: "Buoyancy near 500 m AGL. Highlights positively buoyant inflow and negatively buoyant outflow.",
    category: "plan-views",
    subgroup: "Thermodynamics",
    plotType: "slice",
    field: "buoyancy",
    colormap: "RdBu",
    rangeMode: "symmetric",
    sliceAxis: 2,
    sliceHeightKm: 0.5,
    requires: [["buoyancy"]],
  },
  {
    id: "surface-temperature",
    label: "Surface temperature",
    description: "Lowest-model-level air temperature derived from qualified total pressure and potential temperature.",
    category: "plan-views",
    subgroup: "Thermodynamics",
    plotType: "slice",
    field: "temperature",
    colormap: "Viridis",
    rangeMode: "auto",
    sliceAxis: 2,
    sliceIndexHint: "surface",
    requires: [["temperature"]],
  },
  {
    id: "surface-dewpoint",
    label: "Surface dewpoint",
    description: "Lowest-model-level dewpoint derived from qualified pressure and water-vapor mixing ratio.",
    category: "plan-views",
    subgroup: "Thermodynamics",
    plotType: "slice",
    field: "dewpoint",
    colormap: "Viridis",
    rangeMode: "auto",
    sliceAxis: 2,
    sliceIndexHint: "surface",
    requires: [["dewpoint"]],
  },
  {
    id: "surface-rh",
    label: "Surface RH",
    description: "Lowest-model-level relative humidity over liquid water, derived without clipping supersaturation.",
    category: "plan-views",
    subgroup: "Thermodynamics",
    plotType: "slice",
    field: "relative_humidity",
    colormap: "Viridis",
    rangeMode: "auto",
    sliceAxis: 2,
    sliceIndexHint: "surface",
    requires: [["relative_humidity"]],
  },
  {
    id: "surface-theta-e",
    label: "Surface θe",
    description: "Lowest-model-level equivalent potential temperature. Imported CM1 values disclose the Studio derivation in field readiness.",
    category: "plan-views",
    subgroup: "Thermodynamics",
    plotType: "slice",
    field: "theta_e",
    colormap: "Inferno",
    rangeMode: "auto",
    sliceAxis: 2,
    sliceIndexHint: "surface",
    requires: [["theta_e"]],
  },
  {
    id: "midlevel-pv",
    label: "Mid-level PV",
    description: "Ertel potential vorticity near 5 km AGL. Highlights dynamically distinct storm structures.",
    category: "plan-views",
    subgroup: "Dynamics",
    plotType: "slice",
    field: "potential_vorticity",
    colormap: "RdBu",
    rangeMode: "symmetric",
    sliceAxis: 2,
    sliceHeightKm: 5,
    requires: [["potential_vorticity"]],
  },
  {
    id: "surface-cape",
    label: "CAPE",
    description: "Column convective available potential energy on the horizontal grid.",
    category: "plan-views",
    subgroup: "Thermodynamics",
    plotType: "slice",
    field: "cape",
    colormap: "Inferno",
    rangeMode: "auto",
    requires: [["cape"]],
  },
  {
    id: "surface-cin",
    label: "CIN",
    description: "Column convective inhibition on the horizontal grid.",
    category: "plan-views",
    subgroup: "Thermodynamics",
    plotType: "slice",
    field: "cin",
    // Sequential: CIN is single-signed by convention.
    colormap: "Inferno",
    rangeMode: "auto",
    requires: [["cin"]],
  },
  {
    id: "accumulated-rain",
    label: "Accumulated rain",
    description: "Storm-total surface rainfall in millimetres.",
    category: "plan-views",
    subgroup: "Precipitation",
    plotType: "slice",
    field: "accumulated_rainfall",
    colormap: "Viridis",
    rangeMode: "auto",
    requires: [["accumulated_rainfall"]],
  },
  {
    id: "precipitation-rate",
    label: "Precipitation rate",
    description: "Instantaneous water-equivalent surface precipitation rate in millimetres per hour.",
    category: "plan-views",
    subgroup: "Precipitation",
    plotType: "slice",
    field: "precip_rate",
    colormap: "Inferno",
    rangeMode: "auto",
    requires: [["precip_rate"]],
  },
  {
    id: "precipitable-water",
    label: "Precipitable water",
    description: "Column water-vapor depth in millimetres.",
    category: "plan-views",
    subgroup: "Moisture paths",
    plotType: "slice",
    field: "precipitable_water",
    colormap: "Viridis",
    rangeMode: "auto",
    requires: [["precipitable_water"]],
  },
  {
    id: "liquid-water-path",
    label: "Liquid water path",
    description: "Column-integrated cloud plus rain liquid water mass.",
    category: "plan-views",
    subgroup: "Moisture paths",
    plotType: "slice",
    field: "liquid_water_path",
    colormap: "Inferno",
    rangeMode: "auto",
    requires: [["liquid_water_path"]],
  },
  {
    id: "cloud-water-path",
    label: "Cloud water path",
    description: "Column-integrated cloud-water mass, excluding rainwater.",
    category: "plan-views",
    subgroup: "Moisture paths",
    plotType: "slice",
    field: "cloud_water_path",
    colormap: "Viridis",
    rangeMode: "auto",
    requires: [["cloud_water_path"]],
  },
  {
    id: "surface-convergence",
    label: "Divergence",
    description: "Three-dimensional velocity divergence at the lowest saved level. Negative values indicate local convergence.",
    category: "plan-views",
    plotType: "slice",
    field: "divergence",
    colormap: "RdBu",
    rangeMode: "symmetric",
    sliceAxis: 2,
    sliceIndexHint: "surface",
    requires: [["divergence"]],
  },
  /* Plan views · Dynamics — p' decomposition + vorticity budget. Requires studio_full or wider. */
  {
    id: "dynamic-pressure",
    label: "Dynamic pressure",
    description: "Anelastic pressure potential forced by resolved velocity advection, with a per-level zero-volume-mean reference.",
    category: "plan-views",
    subgroup: "Dynamics",
    plotType: "slice",
    field: "dynamic_pressure",
    colormap: "RdBu",
    rangeMode: "symmetric",
    sliceAxis: 2,
    sliceIndexHint: "surface",
    requires: [["dynamic_pressure"]],
  },
  {
    id: "buoyancy-pressure",
    label: "Buoyancy pressure",
    description: "Anelastic pressure potential forced by buoyancy, with a per-level zero-volume-mean reference.",
    category: "plan-views",
    subgroup: "Dynamics",
    plotType: "slice",
    field: "buoyancy_pressure",
    colormap: "RdBu",
    rangeMode: "symmetric",
    sliceAxis: 2,
    sliceIndexHint: "surface",
    requires: [["buoyancy_pressure"]],
  },
  {
    id: "ppgf-z",
    label: "Vertical PGF",
    description: "Vertical acceleration from the full pressure gradient, −(1/ρ)∂p/∂z. Gravity is a separate term.",
    category: "plan-views",
    subgroup: "Dynamics",
    plotType: "slice",
    field: "pressure_gradient_force_z",
    colormap: "RdBu",
    rangeMode: "symmetric",
    sliceAxis: 2,
    sliceIndexHint: "mid",
    requires: [["pressure_gradient_force_z"]],
  },
  {
    id: "stretching-term",
    label: "Stretching",
    description: "Vorticity stretching term. Vortex intensification by convergence.",
    category: "plan-views",
    subgroup: "Dynamics",
    plotType: "slice",
    field: "stretching_term",
    colormap: "RdBu",
    rangeMode: "symmetric",
    sliceAxis: 2,
    sliceIndexHint: "surface",
    requires: [["stretching_term"]],
  },
  {
    id: "tilting-term",
    label: "Tilting",
    description: "Vorticity tilting term. Horizontal-to-vertical vorticity conversion.",
    category: "plan-views",
    subgroup: "Dynamics",
    plotType: "slice",
    field: "tilting_term",
    colormap: "RdBu",
    rangeMode: "symmetric",
    sliceAxis: 2,
    sliceIndexHint: "surface",
    requires: [["tilting_term"]],
  },
  {
    id: "baroclinic-term",
    label: "Baroclinic",
    description: "Baroclinic vorticity generation. Horizontal buoyancy gradients.",
    category: "plan-views",
    subgroup: "Dynamics",
    plotType: "slice",
    field: "baroclinic_term",
    colormap: "RdBu",
    rangeMode: "symmetric",
    sliceAxis: 2,
    sliceIndexHint: "surface",
    requires: [["baroclinic_term"]],
  },

  /* ======================== Cross-sections ======================== */
  {
    id: "refl-xsection",
    label: "Reflectivity",
    description: "Vertical reflectivity cross-section — WER / BWER / overhang.",
    category: "cross-sections",
    plotType: "slice",
    field: "reflectivity_dbz",
    colormap: "dBZ",
    rangeMode: "dBZ",
    sliceAxis: 1,
    requires: [["reflectivity_dbz", "radar"]],
  },
  {
    id: "updraft-xsection",
    label: "Updraft (w)",
    description: "Vertical w cross-section — updraft tilt and depth.",
    category: "cross-sections",
    plotType: "slice",
    field: "w",
    colormap: "RdBu",
    rangeMode: "symmetric",
    sliceAxis: 1,
    requires: [["w"]],
  },
  {
    id: "thetae-xsection",
    label: "θ_e",
    description: "Equivalent potential temperature cross-section — inflow / instability.",
    category: "cross-sections",
    plotType: "slice",
    field: "theta_e",
    colormap: "Inferno",
    rangeMode: "auto",
    sliceAxis: 1,
    requires: [["theta_e"]],
  },
  /* Cross-sections · Hydrometeors — species depend on the microphysics scheme
     (Kessler has no qi/qs); gating hides the absent ones. */
  {
    id: "xsec-qc",
    label: "Cloud (qc)",
    description: "Cloud-water mixing ratio cross-section.",
    category: "cross-sections",
    subgroup: "Hydrometeors",
    plotType: "slice",
    field: "qc",
    colormap: "Viridis",
    rangeMode: "auto",
    sliceAxis: 1,
    requires: [["qc"]],
  },
  {
    id: "xsec-qr",
    label: "Rain (qr)",
    description: "Rain mixing ratio cross-section.",
    category: "cross-sections",
    subgroup: "Hydrometeors",
    plotType: "slice",
    field: "qr",
    colormap: "Viridis",
    rangeMode: "auto",
    sliceAxis: 1,
    requires: [["qr"]],
  },
  {
    id: "xsec-qg",
    label: "Graupel (qg)",
    description: "Graupel mixing ratio cross-section.",
    category: "cross-sections",
    subgroup: "Hydrometeors",
    plotType: "slice",
    field: "qg",
    colormap: "Viridis",
    rangeMode: "auto",
    sliceAxis: 1,
    requires: [["qg"]],
  },
  {
    id: "xsec-qh",
    label: "Hail (qh)",
    description: "Hail mixing ratio cross-section.",
    category: "cross-sections",
    subgroup: "Hydrometeors",
    plotType: "slice",
    field: "qh",
    colormap: "Viridis",
    rangeMode: "auto",
    sliceAxis: 1,
    requires: [["qh"]],
  },
  {
    id: "xsec-qs",
    label: "Snow (qs)",
    description: "Snow mixing ratio cross-section.",
    category: "cross-sections",
    subgroup: "Hydrometeors",
    plotType: "slice",
    field: "qs",
    colormap: "Viridis",
    rangeMode: "auto",
    sliceAxis: 1,
    requires: [["qs"]],
  },
  {
    id: "xsec-qi",
    label: "Ice (qi)",
    description: "Cloud-ice mixing ratio cross-section.",
    category: "cross-sections",
    subgroup: "Hydrometeors",
    plotType: "slice",
    field: "qi",
    colormap: "Viridis",
    rangeMode: "auto",
    sliceAxis: 1,
    requires: [["qi"]],
  },

  /* =========================== 3D / volumetric =========================== */
  /* Vorticity budget in 3D (Davies-Jones; Markowski & Richardson). Side-by-side = budget figure. */
  {
    id: "stretching-volume",
    label: "Stretching (3D)",
    description: "Vorticity stretching through the depth of the storm. Intensification by convergence.",
    category: "volumetric",
    subgroup: "Vorticity budget",
    plotType: "volume",
    field: "stretching_term",
    // No colormap: defaultVolumeColormap resolves signed fields automatically.
    rangeMode: "symmetric",
    requires: [["stretching_term"]],
  },
  {
    id: "tilting-volume",
    label: "Tilting (3D)",
    description: "Horizontal-to-vertical vorticity conversion through the storm depth.",
    category: "volumetric",
    subgroup: "Vorticity budget",
    plotType: "volume",
    field: "tilting_term",
    // No colormap: defaultVolumeColormap resolves signed fields automatically.
    rangeMode: "symmetric",
    requires: [["tilting_term"]],
  },
  {
    id: "baroclinic-volume",
    label: "Baroclinic (3D)",
    description: "Baroclinic vorticity generation along horizontal buoyancy gradients — the cold-pool boundary source.",
    category: "volumetric",
    subgroup: "Vorticity budget",
    plotType: "volume",
    field: "baroclinic_term",
    // No colormap: defaultVolumeColormap resolves signed fields automatically.
    rangeMode: "symmetric",
    requires: [["baroclinic_term"]],
  },
  {
    id: "vorticity-volume",
    label: "Vorticity volume",
    description: "3D |ω| volume — vortex tubes + tornadic structure.",
    category: "volumetric",
    plotType: "volume",
    field: "vorticity_magnitude",
    colormap: "Inferno",
    rangeMode: "auto",
    requires: [["vorticity_magnitude"]],
  },
  {
    id: "updraft-iso",
    label: "Updraft envelope",
    description: "Isosurface of w = 10 m/s. Mesocyclone updraft shape.",
    category: "volumetric",
    plotType: "isosurface",
    field: "w",
    colormap: "Inferno",
    rangeMode: "auto",
    requires: [["w"]],
  },

  /* ============================= Swaths ============================= */
  {
    id: "max-reflectivity-swath",
    label: "Max reflectivity",
    description: "Greatest reflectivity at each point over the whole run — the storm track.",
    category: "swaths",
    plotType: "swath",
    field: "composite_reflectivity",
    colormap: "dBZ",
    requires: [["composite_reflectivity"]],
  },
  {
    id: "rotation-tracks",
    label: "Rotation tracks",
    description: "Greatest column-max vorticity over the run — mesocyclone / tornado tracks.",
    category: "swaths",
    plotType: "swath",
    field: "column_max_vorticity",
    colormap: "Inferno",
    requires: [["column_max_vorticity"]],
  },
  {
    id: "max-uh-swath",
    label: "Max UH (rotation track)",
    description: "Peak updraft helicity over the run — the canonical mesocyclone / tornado track.",
    category: "swaths",
    plotType: "swath",
    field: "updraft_helicity",
    colormap: "Inferno",
    requires: [["updraft_helicity"]],
  },
  {
    id: "max-updraft-swath",
    label: "Max updraft",
    description: "Greatest column-max w over the run — the updraft swath.",
    category: "swaths",
    plotType: "swath",
    field: "column_max_w",
    colormap: "Inferno",
    requires: [["column_max_w"]],
  },
];

/** CM1-familiar aliases over the canonical plot registry (not a second plotting system). */
export type Cm1FamiliarViewDefinition = {
  id: string;
  label: string;
  description: string;
} & (
  | { kind: "preset"; presetIds: readonly string[] }
  | { kind: "renderer"; category: PlotCategory; plotType: PlotType }
);

export const CM1_FAMILIAR_VIEWS: ReadonlyArray<Cm1FamiliarViewDefinition> = [
  {
    id: "reflectivity",
    label: "Reflectivity",
    description: "Lowest-level reflectivity, falling back to composite reflectivity when that is the qualified product.",
    kind: "preset",
    presetIds: ["surface-dbz", "composite-dbz"],
  },
  {
    id: "updraft",
    label: "Vertical velocity",
    description: "Mid-level vertical velocity through the primary updraft.",
    kind: "preset",
    presetIds: ["midlevel-updraft"],
  },
  {
    id: "rotation",
    label: "Low-level rotation",
    description: "Low-level vertical vorticity and mesocyclone structure.",
    kind: "preset",
    presetIds: ["low-vorticity"],
  },
  {
    id: "cold-pool",
    label: "Cold pool",
    description: "Surface potential-temperature perturbation and outflow structure.",
    kind: "preset",
    presetIds: ["cold-pool"],
  },
  {
    id: "pressure-perturbation",
    label: "Pressure perturbation",
    description: "Surface perturbation pressure and mesolow structure.",
    kind: "preset",
    presetIds: ["p-prime"],
  },
  {
    id: "theta-e",
    label: "Surface θe",
    description: "Lowest-level equivalent potential temperature when its qualified dependencies are available.",
    kind: "preset",
    presetIds: ["surface-theta-e"],
  },
  {
    id: "sounding",
    label: "Skew-T",
    description: "Thermodynamic profile at a selected horizontal point.",
    kind: "renderer",
    category: "soundings",
    plotType: "skewt",
  },
  {
    id: "hodograph",
    label: "Hodograph",
    description: "Wind profile through the selected column.",
    kind: "renderer",
    category: "soundings",
    plotType: "hodograph",
  },
];

export type ResolvedCm1FamiliarView =
  | (Cm1FamiliarViewDefinition & { kind: "preset"; preset: PlotPreset | null; available: boolean })
  | (Cm1FamiliarViewDefinition & { kind: "renderer"; available: boolean });

export function resolveCm1FamiliarViews(
  availableFields: ReadonlySet<string>,
  presets: ReadonlyArray<PlotPreset> = PLOT_PRESETS,
): ResolvedCm1FamiliarView[] {
  return CM1_FAMILIAR_VIEWS.map((definition) => {
    if (definition.kind === "renderer") {
      return {
        ...definition,
        available: isPlotAvailable(definition.plotType, availableFields),
      };
    }
    const preset = definition.presetIds
      .map((id) => presets.find((candidate) => candidate.id === id) ?? null)
      .find((candidate): candidate is PlotPreset =>
        candidate != null && isPresetAvailable(candidate, availableFields)
      ) ?? null;
    return { ...definition, preset, available: preset != null };
  });
}
