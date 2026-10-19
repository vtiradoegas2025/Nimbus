/** CM1-oriented navigation landmarks. same_scalar = directly transferable;
 *  review/different must never be presented as numerical equivalence. */

export type Cm1ConceptRelationship = "same_scalar" | "review" | "different";

export interface Cm1SectionLandmark {
  sectionId: string;
  familiarConcept: string;
  cm1Keys: readonly string[];
  note: string;
}

export interface Cm1FieldLandmark {
  nimbusPath: string;
  cm1Keys: readonly string[];
  familiarConcept: string;
  relationship: Cm1ConceptRelationship;
  note: string;
}

export const CM1_SECTION_LANDMARKS: readonly Cm1SectionLandmark[] = [
  {
    sectionId: "grid",
    familiarConcept: "Domain and mesh",
    cm1Keys: ["nx", "ny", "nz", "dx", "dy", "dz", "ztop", "stretch_z"],
    note: "Uniform Cartesian scalars transfer directly; stretched vertical meshes do not.",
  },
  {
    sectionId: "time",
    familiarConcept: "Integration and history cadence",
    cm1Keys: ["run_time", "timax", "dtl", "nsound", "tapfrq"],
    note: "The familiar timing scalars lead here, subject to exact-binary preflight.",
  },
  {
    sectionId: "dynamics",
    familiarConcept: "Pressure and acoustic dynamics",
    cm1Keys: ["psolver"],
    note: "Nimbus dynamics are a separate solver family, not a psolver translation.",
  },
  {
    sectionId: "boundary_conditions",
    familiarConcept: "Lateral, lower, and upper boundaries",
    cm1Keys: ["wbc", "ebc", "sbc", "nbc", "bbc", "tbc"],
    note: "CM1 numeric modes require an explicit numerical-behavior decision.",
  },
  {
    sectionId: "damping",
    familiarConcept: "Upper-domain damping",
    cm1Keys: ["irdamp", "zd", "rdalpha"],
    note: "The concepts are familiar; the reference state and damping formulation differ.",
  },
  {
    sectionId: "coriolis",
    familiarConcept: "Coriolis setup",
    cm1Keys: ["icor", "lat0", "fcor"],
    note: "Reconcile the CM1 mode before choosing Nimbus latitude-based f-plane controls.",
  },
  {
    sectionId: "microphysics",
    familiarConcept: "Moist physics family",
    cm1Keys: ["ptype"],
    note: "Scheme-family names may overlap, but implementations are not interchangeable.",
  },
  {
    sectionId: "turbulence",
    familiarConcept: "LES and subgrid mixing",
    cm1Keys: ["cm1setup", "sgsmodel", "tconfig"],
    note: "Nimbus separates choices that CM1 may couple; closure identity is unqualified.",
  },
  {
    sectionId: "diffusion",
    familiarConcept: "Numerical mixing",
    cm1Keys: ["difforder", "kdiff2", "kdiff6"],
    note: "Operator and coefficient meanings differ and values must not be copied mechanically.",
  },
  {
    sectionId: "boundary_layer",
    familiarConcept: "PBL closure",
    cm1Keys: ["ipbl", "cm1setup"],
    note: "Scheme families overlap only partially and require option-specific review.",
  },
  {
    sectionId: "sounding",
    familiarConcept: "Thermodynamic sounding",
    cm1Keys: ["isnd", "input_sounding"],
    note: "Nimbus can project a reviewed text sounding; exact CM1 initialization is not claimed.",
  },
  {
    sectionId: "hodograph",
    familiarConcept: "Wind initialization",
    cm1Keys: ["iwnd", "input_sounding"],
    note: "Wind ownership depends on CM1 isnd/iwnd and the selected Nimbus source.",
  },
  {
    sectionId: "output",
    familiarConcept: "History output",
    cm1Keys: ["tapfrq", "output_format", "output_filetype", "output_*"],
    note: "Nimbus netCDF and field bundles do not reproduce CM1 file-layout codes or flags.",
  },
  {
    sectionId: "checkpoint",
    familiarConcept: "Restart cadence and continuation",
    cm1Keys: ["rstfrq", "irst", "rstnum", "restart_*"],
    note: "Cadence can be reviewed; Nimbus never consumes a CM1 restart container.",
  },
] as const;

export const CM1_FIELD_LANDMARKS: readonly Cm1FieldLandmark[] = [
  field("grid.nx", ["nx"], "x grid points", "same_scalar", "Same Cartesian cell-count scalar."),
  field("grid.ny", ["ny"], "y grid points", "same_scalar", "Same Cartesian cell-count scalar."),
  field("grid.nz", ["nz"], "vertical grid points", "same_scalar", "Same vertical cell-count scalar."),
  field("grid.dx", ["dx"], "x spacing", "same_scalar", "Same metre-valued uniform spacing."),
  field("grid.dy", ["dy"], "y spacing", "same_scalar", "Same metre-valued uniform spacing."),
  field("grid.dz", ["dz"], "vertical spacing", "same_scalar", "Direct only when CM1 vertical stretching is off."),
  field("grid.ztop", ["ztop", "stretch_z"], "model top and stretching", "review", "Height alone does not reproduce a stretched CM1 mesh."),
  field("time.run_time", ["run_time", "timax"], "integration limit", "same_scalar", "Same seconds-valued limit after resolving which CM1 key is authoritative."),
  field("time.dt_large", ["dtl"], "large timestep", "same_scalar", "Same large dynamics timestep in seconds."),
  field("time.acoustic_substeps", ["nsound"], "acoustic substeps", "same_scalar", "Same explicit count; zero/automatic behavior still needs preflight."),
  field("time.output_interval", ["tapfrq"], "3-D history cadence", "same_scalar", "Same history cadence in seconds."),
  field("dynamics.scheme", ["psolver"], "pressure/acoustic solver", "different", "Nimbus does not expose CM1 psolver choices as equivalent schemes."),
  field("boundary_conditions.lateral.west", ["wbc"], "west boundary", "review", "Translate the numeric mode and review numerical behavior."),
  field("boundary_conditions.lateral.east", ["ebc"], "east boundary", "review", "Translate the numeric mode and review numerical behavior."),
  field("boundary_conditions.lateral.south", ["sbc"], "south boundary", "review", "Translate the numeric mode and review numerical behavior."),
  field("boundary_conditions.lateral.north", ["nbc"], "north boundary", "review", "Translate the numeric mode and review numerical behavior."),
  field("boundary_conditions.bottom", ["bbc"], "lower boundary", "review", "Free/no-slip candidates exist; CM1 semi-slip has no direct control."),
  field("boundary_conditions.top", ["tbc"], "upper boundary", "review", "Free/no-slip candidates exist; CM1 semi-slip coupling is not reproduced."),
  field("damping.enabled", ["irdamp"], "upper damping mode", "different", "Nimbus damps a different reference-state contract."),
  field("damping.z_d", ["zd"], "damping onset height", "different", "Same onset-height concept within a different formulation."),
  field("damping.alpha", ["rdalpha"], "damping rate", "different", "Same inverse-time dimension within a different formulation."),
  field("coriolis.enabled", ["icor"], "Coriolis mode", "review", "Resolve the active CM1 Coriolis mode first."),
  field("coriolis.mode", ["icor"], "Coriolis mode", "review", "Nimbus exposes an f-plane contract, not CM1 mode numbers."),
  field("coriolis.latitude_deg", ["lat0", "fcor"], "latitude / Coriolis parameter", "review", "Nimbus exposes latitude rather than a direct fcor scalar."),
  field("microphysics.scheme", ["ptype"], "microphysics family", "different", "Option names may overlap; implementation identity is unqualified."),
  field("turbulence.scheme", ["sgsmodel", "cm1setup"], "SGS closure", "different", "Closure families overlap, but implementations and tunables differ."),
  field("turbulence.mode", ["tconfig"], "mixing geometry", "different", "Both distinguish mixing geometry; coefficient formulations differ."),
  field("diffusion.operator_order", ["difforder"], "diffusion operator order", "different", "Nimbus order does not reproduce the complete CM1 diffusion contract."),
  field("diffusion.K_h", ["kdiff2", "kdiff6"], "horizontal diffusion coefficient", "different", "Coefficient meaning depends on the operator and cannot be copied."),
  field("diffusion.K_v", ["kdiff2", "kdiff6"], "vertical diffusion coefficient", "different", "Coefficient meaning depends on the operator and cannot be copied."),
  field("boundary_layer.scheme", ["ipbl", "cm1setup"], "PBL family", "different", "Only partial scheme-family overlap is established."),
  field("initial_conditions.sounding.type", ["isnd"], "sounding source mode", "review", "External and analytic CM1 modes need source-specific translation."),
  field("initial_conditions.hodograph.type", ["iwnd"], "wind source mode", "review", "Reconcile wind ownership with isnd and input_sounding."),
  field("output.preset", ["output_*"], "history field flags", "review", "Nimbus publishes bundles and derived fields, not identical per-variable flags."),
  field("output.format", ["output_format", "output_filetype"], "history file format/layout", "review", "Choose Nimbus netCDF independently; CM1 layout codes do not transfer."),
  field("extensions.checkpoint.every_s", ["rstfrq"], "restart cadence", "review", "Reuse only the reviewed seconds cadence, never a CM1 restart file."),
] as const;

function field(
  nimbusPath: string,
  cm1Keys: readonly string[],
  familiarConcept: string,
  relationship: Cm1ConceptRelationship,
  note: string,
): Cm1FieldLandmark {
  return { nimbusPath, cm1Keys, familiarConcept, relationship, note };
}

const SECTIONS_BY_ID = new Map(CM1_SECTION_LANDMARKS.map((item) => [item.sectionId, item]));
const FIELDS_BY_PATH = new Map(CM1_FIELD_LANDMARKS.map((item) => [item.nimbusPath, item]));

export function cm1SectionLandmark(sectionId: string): Cm1SectionLandmark | null {
  return SECTIONS_BY_ID.get(sectionId) ?? null;
}

export function cm1FieldLandmark(nimbusPath: string): Cm1FieldLandmark | null {
  return FIELDS_BY_PATH.get(nimbusPath) ?? null;
}

export function cm1RelationshipLabel(relationship: Cm1ConceptRelationship): string {
  switch (relationship) {
    case "same_scalar":
      return "same scalar";
    case "review":
      return "review mapping";
    case "different":
      return "different formulation";
  }
}
