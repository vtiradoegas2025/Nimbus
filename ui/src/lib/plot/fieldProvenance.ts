/**
 * Per-field provenance (nimbus_output / cm1_history / studio_derived) to drive accurate gating advice.
 * Drift-tested against field_registry.hpp and the config schema by test_field_provenance.ts.
 */
import { cm1HistoryMappingForCanonical } from "@/lib/cm1/cm1HistoryMapping";
import { derivedFieldDefinition } from "@/lib/derived/derivedFieldRegistry";
import type { DatasetSourceKind } from "@/lib/dataset/datasetSources";
import {
  CROSS_SECTION_VECTOR_CAPABILITY,
  NATIVE_REDUCTION_CAPABILITY,
  NATIVE_VOLUME_REDUCTION_CAPABILITY,
  PLANE_SCALAR_CAPABILITY,
  PLAN_VECTOR_CAPABILITY,
  TRAJECTORY_CAPABILITY,
  VOLUME_SCALAR_CAPABILITY,
} from "@/lib/plot/plotRegistry";

/** Shape class assigned by the engine. core_volume/derived_volume = 3D; others = single-level. */
export type NimbusFieldKind =
  | "core_volume"
  | "derived_volume"
  | "surface"
  | "column"
  | "radar_synthetic";

/** All field ids the Nimbus output registry can publish. Mirrored from field_registry.hpp; drift-tested. */
export const NIMBUS_OUTPUT_FIELDS: Readonly<Record<string, NimbusFieldKind>> = {
  accumulated_rainfall: "surface",
  angular_momentum: "core_volume",
  angular_momentum_tendency: "core_volume",
  backward_trajectories: "derived_volume",
  baroclinic_term: "core_volume",
  buoyancy: "derived_volume",
  buoyancy_pressure: "core_volume",
  bwer: "radar_synthetic",
  cape: "column",
  cin: "column",
  circulation_material_surface: "derived_volume",
  cloud_base_height: "column",
  cloud_condensate: "derived_volume",
  cloud_frozen_condensate: "derived_volume",
  cloud_liquid_condensate: "derived_volume",
  cloud_top_height: "column",
  cloud_water_path: "column",
  cold_pool_boundary: "surface",
  column_max_vorticity: "column",
  column_max_w: "column",
  composite_reflectivity: "column",
  cross_section: "derived_volume",
  dewpoint: "derived_volume",
  divergence: "derived_volume",
  dynamic_pressure: "core_volume",
  ehi: "column",
  el: "column",
  forward_trajectories: "derived_volume",
  helicity_density: "derived_volume",
  hodograph_aligned_cross_section: "derived_volume",
  horizontal_vorticity_crosswise: "derived_volume",
  horizontal_vorticity_streamwise: "derived_volume",
  k_index: "column",
  kdp: "derived_volume",
  lcl: "column",
  lfc: "column",
  lifted_index: "column",
  liquid_water_path: "column",
  mesocyclone_diagnostic: "radar_synthetic",
  okubo_weiss: "derived_volume",
  p: "core_volume",
  p_prime: "core_volume",
  parcel_buoyancy_trajectory: "derived_volume",
  pbl_height: "surface",
  potential_vorticity: "derived_volume",
  ppi_sweep: "radar_synthetic",
  precip_rate: "surface",
  precipitable_water: "column",
  pressure_gradient_force_x: "derived_volume",
  pressure_gradient_force_y: "derived_volume",
  pressure_gradient_force_z: "derived_volume",
  q_vectors: "derived_volume",
  qc: "core_volume",
  qg: "core_volume",
  qh: "core_volume",
  qi: "core_volume",
  qr: "core_volume",
  qs: "core_volume",
  qv: "core_volume",
  radar: "core_volume",
  reflectivity_dbz: "derived_volume",
  radial_velocity: "derived_volume",
  relative_humidity: "derived_volume",
  rhi_slice: "derived_volume",
  rhi_sweep: "radar_synthetic",
  rho: "core_volume",
  rhohv: "derived_volume",
  saturation_mixing_ratio: "derived_volume",
  scp: "column",
  showalter_index: "column",
  skin_temperature: "surface",
  srh_0_1km: "column",
  srh_0_3km: "column",
  storm_relative_winds: "derived_volume",
  stp: "column",
  streamlines: "derived_volume",
  stretching_term: "core_volume",
  surface_friction_velocity: "surface",
  surface_latent_heat_flux: "surface",
  surface_moisture_flux: "surface",
  surface_pressure_perturbation: "surface",
  surface_sensible_heat_flux: "surface",
  surface_stress_u: "surface",
  surface_stress_v: "surface",
  t2: "surface",
  td2: "surface",
  temperature: "derived_volume",
  theta: "core_volume",
  theta_e: "derived_volume",
  theta_prime: "derived_volume",
  theta_v: "derived_volume",
  theta_w: "derived_volume",
  tilting_term: "core_volume",
  tke: "core_volume",
  total_condensate: "derived_volume",
  total_totals: "column",
  tracer: "core_volume",
  trajectory_paths: "derived_volume",
  turbulent_diffusion_term: "derived_volume",
  u: "core_volume",
  u10: "surface",
  updraft_helicity: "column",
  v: "core_volume",
  v10: "surface",
  vil: "column",
  vorticity_magnitude: "derived_volume",
  vorticity_r: "core_volume",
  vorticity_theta: "core_volume",
  vorticity_trajectory: "derived_volume",
  vorticity_z: "core_volume",
  vrot: "radar_synthetic",
  w: "core_volume",
  zdr: "derived_volume",
};

/** Preset values Studio can recommend. `output.preset` is an enum; drift-tested against configSchema. */
export type RecommendedOutputPreset = "studio_full" | "dynamics_full";

/** The narrowest preset that publishes Studio's whole Nimbus-native catalog. */
export const STUDIO_CATALOG_PRESET = "studio_full" as const;

/** The only preset that publishes every registry field, at the greatest cost. */
export const WIDEST_OUTPUT_PRESET = "dynamics_full" as const;

/** Fields published by `output.preset: studio_full`. Mirrored from output_config.cpp; drift-tested. */
export const STUDIO_CATALOG_PRESET_FIELDS: ReadonlySet<string> = new Set([
  "u", "v", "w", "theta", "p", "rho", "qv", "qr",
  "vorticity_z", "vorticity_magnitude", "total_condensate",
  "cloud_condensate", "cloud_liquid_condensate", "cloud_frozen_condensate",
  "temperature", "buoyancy", "theta_prime", "p_prime", "composite_reflectivity",
  "stp", "updraft_helicity", "cape", "cin", "lcl", "lfc", "el",
  "srh_0_1km", "srh_0_3km", "vil", "cloud_base_height", "cloud_top_height",
  "surface_sensible_heat_flux", "surface_latent_heat_flux", "pbl_height",
  "stretching_term", "tilting_term", "baroclinic_term",
  "vorticity_r", "vorticity_theta",
  "qc", "qg", "qh", "qs", "qi",
  "dynamic_pressure", "buoyancy_pressure", "pressure_gradient_force_z",
  "divergence", "column_max_vorticity", "column_max_w",
  "accumulated_rainfall", "precip_rate",
  "precipitable_water", "liquid_water_path", "cloud_water_path",
  "potential_vorticity",
]);

/**
 * The narrowest preset that publishes `field`, or null when no preset can
 * because the engine has no such field.
 */
export function presetPublishing(field: string): RecommendedOutputPreset | null {
  if (!nimbusFieldKind(field)) return null;
  return STUDIO_CATALOG_PRESET_FIELDS.has(field)
    ? STUDIO_CATALOG_PRESET
    : WIDEST_OUTPUT_PRESET;
}

/** Narrowest preset that publishes every engine field in `fields`; skips non-publishable names. */
export function presetPublishingAll(fields: Iterable<string>): RecommendedOutputPreset {
  for (const field of fields) {
    if (presetPublishing(field) === WIDEST_OUTPUT_PRESET) return WIDEST_OUTPUT_PRESET;
  }
  return STUDIO_CATALOG_PRESET;
}

/** A place a gating field can legitimately come from. */
export type FieldOrigin =
  | "nimbus_output"
  | "cm1_history"
  | "studio_derived"
  | "run_artifact"
  | "dataset_capability";

const INTERNAL_PLOT_CAPABILITIES = new Set([
  PLANE_SCALAR_CAPABILITY,
  VOLUME_SCALAR_CAPABILITY,
  NATIVE_REDUCTION_CAPABILITY,
  NATIVE_VOLUME_REDUCTION_CAPABILITY,
  PLAN_VECTOR_CAPABILITY,
  CROSS_SECTION_VECTOR_CAPABILITY,
]);

function isInternalPlotCapability(field: string): boolean {
  return INTERNAL_PLOT_CAPABILITIES.has(field);
}

/** Engine shape class for `field`, or null. Uses Object.hasOwn to guard prototype-inherited names. */
export function nimbusFieldKind(field: string): NimbusFieldKind | null {
  return Object.hasOwn(NIMBUS_OUTPUT_FIELDS, field) ? NIMBUS_OUTPUT_FIELDS[field] : null;
}

/**
 * True when a field carries a vertical dimension, so a 3-D renderer can draw it.
 *
 * Unknown names are treated as volumetric on purpose: the catalog covers engine and
 * Studio-derived output, but not user expressions, external providers, or diagnostics
 * such as q_criterion. Hiding a field Studio cannot classify would remove data the user
 * has, which is worse than offering it and reporting an honest shape error.
 */
export function isVolumetricField(field: string): boolean {
  const kind = nimbusFieldKind(field);
  if (kind == null) return true;
  return kind === "core_volume" || kind === "derived_volume";
}

/** The subset of `fields` a 3-D renderer can draw, in the order given. */
export function volumetricFields(fields: readonly string[]): string[] {
  return fields.filter(isVolumetricField);
}

/** Origins that can supply `field`, in preference order. Empty means no known catalog (gating bug). */
export function fieldOrigins(field: string): FieldOrigin[] {
  if (field === TRAJECTORY_CAPABILITY) return ["run_artifact"];
  if (isInternalPlotCapability(field)) return ["dataset_capability"];
  const origins: FieldOrigin[] = [];
  if (nimbusFieldKind(field)) origins.push("nimbus_output");
  if (cm1HistoryMappingForCanonical(field)) origins.push("cm1_history");
  if (derivedFieldDefinition(field)) origins.push("studio_derived");
  return origins;
}

/** True when the dataset's own source can never supply this field, so no
 *  amount of re-exporting or re-opening will unblock the view. */
export function fieldIsForeignToSource(
  field: string,
  sourceKind: DatasetSourceKind,
): boolean {
  if (isInternalPlotCapability(field)) return false;
  const origins = fieldOrigins(field);
  if (origins.length === 0 || origins.includes("studio_derived")) return false;
  const native: FieldOrigin = sourceKind === "cm1_netcdf" ? "cm1_history" : "nimbus_output";
  return !origins.includes(native) && !origins.includes("run_artifact");
}

// Describes shape (rank + location), not physics, because tranches mix diagnostics and prognostics.
const KIND_NOUN: Readonly<Record<NimbusFieldKind, string>> = {
  core_volume: "a 3-D volume field",
  derived_volume: "a 3-D diagnostic field",
  surface: "a single-level surface field",
  column: "a single-level column product",
  radar_synthetic: "a single-level synthetic radar product",
};

/** Concrete action (or impossibility reason) for a missing field, phrased for the active dataset. */
export function describeMissingFieldRemedy(
  field: string,
  sourceKind: DatasetSourceKind,
): string {
  if (field === PLANE_SCALAR_CAPABILITY) {
    return sourceKind === "cm1_netcdf"
      ? "The opened history set has no qualified rank-2 plane or rank-3 volume field. Open history output containing a centered scalar field."
      : "This run has no rank-2 plane or rank-3 volume field in its published output. Export a scalar field and rerun.";
  }
  if (field === VOLUME_SCALAR_CAPABILITY) {
    return sourceKind === "cm1_netcdf"
      ? "The opened history set has no qualified rank-3 scalar field. Open history output containing a centered x/y/z field."
      : "This run has no published rank-3 scalar field. Export a volume field and rerun.";
  }
  if (field === NATIVE_REDUCTION_CAPABILITY) {
    return "This temporal reducer requires at least one dataset-native field; derived providers use a different read path and are not presented as native history.";
  }
  if (field === NATIVE_VOLUME_REDUCTION_CAPABILITY) {
    return "Time-height requires at least one dataset-native rank-3 field so every level is reduced from the authoritative history payload.";
  }
  if (field === PLAN_VECTOR_CAPABILITY) {
    return "Plan-view vectors require one physically matched horizontal component pair with equal registered ranks. Export u/v or the corresponding horizontal vorticity components.";
  }
  if (field === CROSS_SECTION_VECTOR_CAPABILITY) {
    return "Vertical vector cross-sections require a physically matched in-plane rank-3 pair, such as u/w for the default x/z plane.";
  }
  if (field === TRAJECTORY_CAPABILITY) {
    return sourceKind === "cm1_netcdf"
      ? "Studio reads trajectories only from a Nimbus run's trajectory artifact; a CM1 history set cannot supply them."
      : "Set output.trajectories to true in the run configuration and rerun. Trajectories are a run-level artifact rather than a per-step field, so no output.preset value produces them.";
  }
  const kind = nimbusFieldKind(field);
  const cm1 = cm1HistoryMappingForCanonical(field);
  if (sourceKind === "cm1_netcdf") {
    if (cm1) {
      return `The opened history set has no qualified '${cm1.nativeName}' variable. Open a history set that wrote it, or check CM1 field readiness for a unit or rank blocker.`;
    }
    return kind
      ? `No CM1 history variable maps to '${field}'; it is ${KIND_NOUN[kind]} that only a Nimbus run publishes.`
      : `No CM1 history variable maps to '${field}'.`;
  }
  if (kind) {
    const preset = presetPublishing(field) ?? WIDEST_OUTPUT_PRESET;
    return `Nimbus can export '${field}' (${KIND_NOUN[kind]}); this run's output selection omitted it. Set output.preset to ${preset} and rerun.`;
  }
  if (cm1) {
    return `The Nimbus engine has no '${field}' field, so no output selection can produce it. This view reads the CM1 history variable '${cm1.nativeName}' and applies to opened CM1 datasets.`;
  }
  return `Studio has no source for '${field}'.`;
}

/** A two-to-four word reason class, for a compact chip next to a blocked view. */
export function summarizeMissingFieldReason(
  field: string,
  sourceKind: DatasetSourceKind,
): string {
  if (field === PLANE_SCALAR_CAPABILITY) return "No plane field";
  if (field === VOLUME_SCALAR_CAPABILITY) return "No 3-D field";
  if (field === NATIVE_REDUCTION_CAPABILITY) return "No native field";
  if (field === NATIVE_VOLUME_REDUCTION_CAPABILITY) return "No native 3-D field";
  if (field === PLAN_VECTOR_CAPABILITY) return "No horizontal vector pair";
  if (field === CROSS_SECTION_VECTOR_CAPABILITY) return "No vertical vector pair";
  if (field === TRAJECTORY_CAPABILITY) return "Trajectory artifact";
  if (fieldIsForeignToSource(field, sourceKind)) {
    return sourceKind === "cm1_netcdf" ? "Nimbus runs only" : "CM1 datasets only";
  }
  return sourceKind === "cm1_netcdf" ? "Not in this history set" : "Not in this run's output";
}

/** Missing products grouped by corrective action; avoids repeating the same advice per product. */
export type RemedyGroup =
  | "exportable"
  | "inapplicable"
  | "trajectory_artifact"
  | "no_readable_fields";

export interface MissingClauseAdvice {
  /** The alternatives, any one of which would unblock the view. */
  fields: readonly string[];
  /** First alternative the dataset can supply; drives grouping/preset logic. */
  subject: string;
  /** Short reason class, for a chip beside the blocked view. */
  reason: string;
  /** The full standalone sentence: what to do, or why nothing can be done. */
  remedy: string;
  /** The shared corrective action this product falls under. */
  group: RemedyGroup;
  /** Shape class or other detail not already stated by the group heading. */
  detail: string;
  /** True when this dataset's source can supply none of the alternatives, so
   *  the view is inapplicable here rather than merely unexported. */
  foreign: boolean;
}

/** Heading and one-line action for a remedy group. `preset` must be explicit (default = expensive). */
export function describeRemedyGroup(
  group: RemedyGroup,
  sourceKind: DatasetSourceKind,
  preset: RecommendedOutputPreset,
): { title: string; action: string } {
  const cm1 = sourceKind === "cm1_netcdf";
  switch (group) {
    case "exportable":
      return cm1
        ? {
          title: "Present in CM1 history output, absent here",
          action: "Open a history set that wrote the named variable, or check CM1 field readiness for a unit or rank blocker on it.",
        }
        : preset === STUDIO_CATALOG_PRESET
          ? {
            title: "Exportable — this run's output selection omitted them",
            action: `Set output.preset to ${STUDIO_CATALOG_PRESET} in the run configuration and rerun. The preset is an enum rather than a field list, and ${STUDIO_CATALOG_PRESET} is the narrowest value that publishes every product below.`,
          }
          : {
            title: "Exportable, but outside the Studio catalog",
            action: `Set output.preset to ${WIDEST_OUTPUT_PRESET} in the run configuration and rerun. These sit outside ${STUDIO_CATALOG_PRESET}, so only ${WIDEST_OUTPUT_PRESET} — every registry field — publishes them, at several times the volume per step.`,
          };
    case "inapplicable":
      return cm1
        ? {
          title: "Not producible from a CM1 history set",
          action: "No CM1 history variable maps to these; they are published only by a Nimbus run. Opening another history set will not add them.",
        }
        : {
          title: "Not producible from a Nimbus run",
          action: "The engine has no definition for these, so no output selection or rerun can produce them. They come from CM1 history variables and apply to opened CM1 datasets.",
        };
    case "trajectory_artifact":
      return cm1
        ? {
          title: "Trajectory artifact",
          action: "Studio reads trajectories only from a Nimbus run's trajectory artifact; a CM1 history set cannot supply them.",
        }
        : {
          title: "Trajectory artifact",
          action: "Set output.trajectories to true in the run configuration and rerun. Trajectories are a run-level artifact rather than a per-step field, so no output.preset value produces them; output.trajectory_seed_time_s and output.trajectory_count control when and how many parcels are released.",
        };
    case "no_readable_fields":
      return cm1
        ? {
          title: "Nothing readable in the opened files",
          action: "No variable in the opened history set qualified for plotting. Check CM1 field readiness for the per-variable blocker.",
        }
        : {
          title: "Nothing readable in this run",
          action: "This run published no readable field. Check that the run completed and wrote a current publication manifest.",
        };
  }
}

/** Advice for one OR clause; driven by the first alternative the dataset can supply. */
export function describeMissingClause(
  clause: readonly string[],
  sourceKind: DatasetSourceKind,
): MissingClauseAdvice {
  if (clause.length === 0) {
    return {
      fields: [],
      subject: "",
      reason: "No readable fields",
      remedy: describeRemedyGroup("no_readable_fields", sourceKind, STUDIO_CATALOG_PRESET).action,
      group: "no_readable_fields",
      detail: "",
      foreign: false,
    };
  }
  const actionable = clause.find((field) => !fieldIsForeignToSource(field, sourceKind));
  const subject = actionable ?? clause[0];
  const foreign = actionable == null;
  const remedy = describeMissingFieldRemedy(subject, sourceKind);
  return {
    fields: clause,
    subject,
    reason: summarizeMissingFieldReason(subject, sourceKind),
    remedy: actionable && clause.length > 1
      ? `${remedy} Any one of ${clause.join(", ")} satisfies this view.`
      : remedy,
    group: subject === TRAJECTORY_CAPABILITY
      ? "trajectory_artifact"
      : foreign ? "inapplicable" : "exportable",
    detail: describeClauseDetail(clause, subject, sourceKind),
    foreign,
  };
}

/** Render order: actionable first, then artifact, then inapplicable. */
export const REMEDY_GROUP_ORDER: readonly RemedyGroup[] = [
  "no_readable_fields",
  "exportable",
  "trajectory_artifact",
  "inapplicable",
];

/** One rendered block of advice: a heading, one action, and its products. */
export interface RemedySection<T> {
  /** Stable render key (not group name: exportable splits by preset and can appear twice). */
  key: string;
  group: RemedyGroup;
  /** Prescribed preset. Meaningful only for the exportable group on a Nimbus run. */
  preset: RecommendedOutputPreset;
  title: string;
  action: string;
  entries: T[];
}

/**
 * Group per-clause advice into renderable sections. Exportable group splits by preset on Nimbus runs
 * so products coverable by studio_full are not forced onto dynamics_full by a single outlier.
 */
export function groupRemedyAdvice<T>(
  items: readonly T[],
  adviceOf: (item: T) => MissingClauseAdvice,
  sourceKind: DatasetSourceKind,
): Array<RemedySection<T>> {
  const sections: Array<RemedySection<T>> = [];
  const describe = (
    key: string,
    group: RemedyGroup,
    preset: RecommendedOutputPreset,
    entries: T[],
  ): void => {
    if (entries.length === 0) return;
    sections.push({ key, group, preset, ...describeRemedyGroup(group, sourceKind, preset), entries });
  };
  for (const group of REMEDY_GROUP_ORDER) {
    const members = items.filter((item) => adviceOf(item).group === group);
    if (members.length === 0) continue;
    if (group === "exportable" && sourceKind === "nimbus_run") {
      // Cheapest first: studio_full before dynamics_full.
      for (const preset of [STUDIO_CATALOG_PRESET, WIDEST_OUTPUT_PRESET] as const) {
        describe(
          `${group}:${preset}`,
          group,
          preset,
          members.filter((item) => presetPublishingAll([adviceOf(item).subject]) === preset),
        );
      }
      continue;
    }
    describe(group, group, presetPublishingAll(members.map((item) => adviceOf(item).subject)), members);
  }
  return sections;
}

/** Detail not stated by the group heading: cross-source location or shape class. */
function describeClauseDetail(
  clause: readonly string[],
  subject: string,
  sourceKind: DatasetSourceKind,
): string {
  if (isInternalPlotCapability(subject)) return "rank/provider contract";
  if (subject === TRAJECTORY_CAPABILITY) return "";
  const cm1 = cm1HistoryMappingForCanonical(subject);
  const kind = nimbusFieldKind(subject);
  const foreign = fieldIsForeignToSource(subject, sourceKind);
  const parts: string[] = [];
  if (sourceKind === "cm1_netcdf") {
    if (foreign) {
      if (kind) parts.push(`Nimbus publishes it as ${KIND_NOUN[kind]}`);
    } else if (cm1) {
      parts.push(`CM1 variable '${cm1.nativeName}'`);
    }
  } else if (foreign) {
    if (cm1) parts.push(`CM1 history variable '${cm1.nativeName}'`);
  } else if (kind) {
    parts.push(KIND_NOUN[kind]);
  }
  if (clause.length > 1) parts.push(`any one of ${clause.join(", ")} will do`);
  return parts.join(" · ");
}
