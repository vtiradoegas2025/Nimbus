/* Nimbus YAML config schema — drives the form editor.
 * Authoritative source: src/init/config/v1_loader.cpp (1:1 key mapping). */

export type ScalarValue = number | string | boolean | null;
export type ConfigValue =
  | ScalarValue
  | { [key: string]: ConfigValue }
  | ConfigValue[];

export type ConfigObject = { [key: string]: ConfigValue };

/* ---------- Field metadata ---------- */

export type FieldKind =
  | "int"
  | "float"
  | "bool"
  | "string"
  | "enum";

interface FieldBase {
  /** YAML key under the section root (dot-separated for nested keys). */
  key: string;
  label: string;
  /** Short hint shown beneath the label or in a tooltip. */
  hint?: string;
  /**
   * Longer plain-language explainer surfaced by the (i) info button on
   * each field row. Two or three sentences: what the value controls,
   * how it affects the simulation, and the rule-of-thumb for choosing.
   */
  description?: string;
  /** If present, only show this field when the predicate returns true.
   *  Receives the section sub-tree (the object at section.path). */
  visibleWhen?: (section: ConfigObject) => boolean;
  /** If present, gray out + disable the input when the predicate returns
   *  true. Receives the full top-level config so it can read coordinate-
   *  system or other cross-section signals. */
  disabledWhen?: (root: ConfigObject) => boolean;
  /** Why `disabledWhen` fired, in one sentence naming the setting that owns
   *  the value. Rendered as a visible note on the row — a grayed-out control
   *  with no stated reason reads as a bug, and the `description` explainer is
   *  collapsed by default so it cannot carry this. */
  disabledReason?: string;
  /** Render read-only in the form because the value is derived from another
   *  field, or because only one value is supported. The user changes it by
   *  changing whichever field is the master, never directly. */
  locked?: boolean;
  /** Why the field is locked, shown on the padlock tooltip. Name the control
   *  that actually governs the value; if nothing does, say so. Falls back to a
   *  generic string, so a lock without a reason is never mis-explained. */
  lockedReason?: string;
}

export interface IntField extends FieldBase {
  kind: "int";
  default: number;
  min?: number;
  max?: number;
  step?: number;
  unit?: string;
}

export interface FloatField extends FieldBase {
  kind: "float";
  default: number;
  min?: number;
  max?: number;
  step?: number;
  unit?: string;
}

export interface BoolField extends FieldBase {
  kind: "bool";
  default: boolean;
}

export interface StringField extends FieldBase {
  kind: "string";
  default: string;
  placeholder?: string;
}

export interface EnumField extends FieldBase {
  kind: "enum";
  default: string;
  /** Show disabled option reasons beneath the control. Used when an exact
   * binary capability (rather than user configuration) owns availability. */
  showDisabledReasons?: boolean;
  options: Array<{
    value: string;
    label: string;
    /** When true, the option is unselectable. Use for schemes whose
     *  C++ implementation is scaffolding only (uninitialized tendency
     *  buffers, runtime "not implemented" throws, etc.). */
    disabled?: boolean;
    /** Tooltip shown on hover. Surface the disabled reason ("skeleton
     *  only — use slab"), or a status caveat for enabled options
     *  ("toy diagnostic, not a real RT solver"). */
    note?: string;
  }>;
}

export type FieldSchema =
  | IntField
  | FloatField
  | BoolField
  | StringField
  | EnumField;

/**
 * Marks a section as a turn-off-able physics module. The form shows an
 * on/off control; turning it OFF requires the user to type "disable" (a
 * deliberate, friction-ful confirmation), turning it back ON is one click.
 * Both write `offValue` / `onValue` to `path`. The simulation still runs
 * with the module off.
 */
export interface SectionDisable {
  /** Qualified config path of the on/off switch (e.g. "radiation.scheme"). */
  path: string;
  /** Value that means "off" (e.g. "none" or false). */
  offValue: ScalarValue;
  /** Value restored when re-enabling (e.g. "simple_grey" or true). */
  onValue: ScalarValue;
  /** Human noun for what gets turned off (e.g. "radiation"). */
  noun: string;
}

export interface SectionSchema {
  id: string;
  label: string;
  description: string;
  /** YAML path the section maps to (undefined = top-level). */
  path?: string;
  fields: FieldSchema[];
  /**
   * When true, the form renders this section's fields read-only — these are
   * solver internals, set via a preset or the Raw-YAML view, not hand-edited
   * field by field.
   */
  locked?: boolean;
  /** When set, the section is a turn-off-able physics module (see above). */
  disable?: SectionDisable;
}

/* ---------- Per-section schemas (loader order) ---------- */

/* grid — v1_loader take_grid(). CM1 &param0 / &param11. */
const gridSection: SectionSchema = {
  id: "grid",
  label: "Grid",
  description: "Spatial discretization of the Cartesian (x, y, z) box.",
  path: "grid",
  fields: [
    {
      kind: "enum",
      key: "coordinate_system",
      label: "Coordinate system",
      hint: "Cartesian (x, y, z) flat-Earth box. Fixed — Nimbus Studio is Cartesian-only.",
      description:
        "Coordinate basis the dynamics solver uses. Nimbus Studio runs the " +
        "Cartesian (x, y, z) solver: a flat-Earth box spanning x and y " +
        "horizontally and z vertically — the right geometry for supercell and " +
        "mesoscale convection, with no preferred rotation axis. Fixed; the " +
        "dynamics kernel follows the staggering selection.",
      default: "cartesian",
      locked: true,
      lockedReason:
        "Nimbus Studio ships only the Cartesian solver, so there is nothing to choose. " +
        "Staggering selects which Cartesian dynamics kernel runs.",
      options: [{ value: "cartesian", label: "Cartesian (x, y, z)" }],
    },
    {
      kind: "enum",
      key: "staggering",
      label: "Staggering",
      hint: "C-grid and declared flat/mapped A-grid families are qualified; accelerators remain C-grid-first.",
      description:
        "Where the velocity components live relative to the pressure cell. " +
        "C-grid (Arakawa C) stores u on x-faces and v on y-faces, which removes " +
        "the native collocated 2Δx pressure null mode. Nimbus's repaired " +
        "A-grid couples that mode conservatively and passes declared flat CPU " +
        "families plus the exact stationary Bell/Schär mapped CPU family. Other " +
        "geometry, motion, nesting, and accelerator combinations remain incomplete. C-grid stays the " +
        "production default.",
      default: "c_grid",
      options: [
        { value: "c_grid", label: "C-grid (staggered)" },
        { value: "collocated", label: "Collocated (A-grid)" },
      ],
    },
    {
      kind: "int",
      key: "nx",
      label: "x cells (NX)",
      hint: "Number of grid points in the x dimension.",
      description:
        "Cell count in the x direction. More cells = finer resolution but " +
        "cost scales linearly with NX·NY·NZ. Storm-scale supercell domains " +
        "typically run 96–256 cells per horizontal direction.",
      default: 64,
      min: 16,
      max: 256,
      step: 16,
    },
    {
      kind: "int",
      key: "ny",
      label: "y cells (NY)",
      hint: "Number of grid points in the y dimension.",
      description:
        "Cell count in the y direction. Pair with NX for the horizontal " +
        "footprint; supercell domains usually keep NX and NY comparable so " +
        "the storm has room to organize in both directions.",
      default: 64,
      min: 16,
      max: 256,
      step: 16,
    },
    {
      kind: "int",
      key: "nz",
      label: "Vertical cells (NZ)",
      description:
        "Cell count in the vertical direction. The model top sits at " +
        "ztop; standard tornado domains have NZ between 32 and 64 to span " +
        "0–16 km. For convection-resolving supercell work, ≥ 80 is common.",
      default: 32,
      min: 16,
      max: 128,
      step: 4,
    },
    {
      kind: "float",
      key: "dx",
      label: "x spacing",
      hint: "Physical width of one x cell.",
      description:
        "Physical width of one x cell, in meters. CFL is dt·U/dx, so smaller " +
        "dx forces smaller dt. Storm-scale supercell work uses 500–1000 m; " +
        "finer convective detail runs 100–250 m.",
      default: 250,
      min: 1,
      unit: "m",
    },
    {
      kind: "float",
      key: "dy",
      label: "y spacing",
      hint: "Physical width of one y cell.",
      description:
        "Physical width of one y cell, in meters. Pair with dx; equal dx and " +
        "dy gives square horizontal cells, the usual choice for an isotropic " +
        "supercell domain.",
      default: 250,
      min: 1,
      unit: "m",
    },
    {
      kind: "float",
      key: "dz",
      label: "Vertical spacing",
      description:
        "Vertical cell height in meters. Model top = NZ·dz (or ztop). For " +
        "tornado boundary-layer detail, dz ≈ 100–250 m; for tropopause-" +
        "spanning supercell runs, 500 m is common. Extreme dz/dx aspect " +
        "ratios trip the warning in the diagnostics strip.",
      default: 250,
      min: 1,
      unit: "m",
    },
    {
      kind: "float",
      key: "ztop",
      label: "Domain top height",
      hint: "Rigid-lid height (m). CM1 derives from nz·dz; Nimbus pins it.",
      description:
        "Height of the rigid lid above the surface, in meters. The vertical " +
        "grid is stretched between the surface and this height. 16 km spans " +
        "the troposphere for storm-scale work.",
      default: 16000,
      min: 100,
      step: 500,
      unit: "m",
    },
  ],
};

function rkCatalogEnvironmentReady(root: ConfigObject): boolean {
  const products = readPath(root, "radar.products");
  const turbulenceScheme = String(readPath(root, "turbulence.scheme") ?? "");
  const standardSmagorinsky = turbulenceScheme === "smagorinsky" &&
    String(readPath(root, "turbulence.filter_width") ?? "cubic_root") === "cubic_root" &&
    readPath(root, "turbulence.dynamic_coefficient") !== true &&
    String(readPath(root, "turbulence.stability_correction") ?? "none") === "none";
  const standardTke = turbulenceScheme === "tke" &&
    String(readPath(root, "turbulence.filter_width") ?? "cubic_root") === "cubic_root" &&
    readPath(root, "turbulence.dynamic_coefficient") !== true &&
    String(readPath(root, "turbulence.stability_correction") ?? "tke_based") === "tke_based";
  const advection = String(readPath(root, "advection.scheme") ?? "");
  const standardColumnPhysics =
    (readPath(root, "surface.scheme") === "none" &&
      readPath(root, "surface.apply_fluxes") === false &&
      readPath(root, "boundary_layer.scheme") === "none" &&
      ["none", "rrtmg"].includes(String(readPath(root, "radiation.scheme") ?? ""))) ||
    (readPath(root, "surface.scheme") === "slab" &&
      readPath(root, "surface.apply_fluxes") === true &&
      readPath(root, "boundary_layer.scheme") === "slab" &&
      readPath(root, "radiation.scheme") === "simple_grey" &&
      String(readPath(root, "radiation.simple_grey_optical_model") ?? "pressure_clear_sky_v1") ===
        "pressure_clear_sky_v1") ||
    (readPath(root, "surface.scheme") === "slab" &&
      readPath(root, "surface.apply_fluxes") === true &&
      readPath(root, "boundary_layer.scheme") === "ysu" &&
      readPath(root, "radiation.scheme") === "rrtmg") ||
    (readPath(root, "surface.scheme") === "slab" &&
      readPath(root, "surface.apply_fluxes") === true &&
      readPath(root, "boundary_layer.scheme") === "mynn" &&
      readPath(root, "radiation.scheme") === "rrtmg");
  return readPath(root, "grid.coordinate_system") === "cartesian" &&
    readPath(root, "grid.staggering") === "c_grid" &&
    readPath(root, "extensions.compute.backend") === "cpu" &&
    readPath(root, "extensions.compute.allow_fallback") === false &&
    readPath(root, "extensions.compute.gpu_apply_microphysics") === false &&
    readPath(root, "extensions.compute.gpu_apply_slow") === false &&
    readPath(root, "extensions.compute.gpu_apply_sgs") === false &&
    readPath(root, "time.adaptive_dt") === false &&
    readPath(root, "domain_motion.mode") === "stationary" &&
    readPath(root, "domain_motion.lifetime_planning.enabled") === false &&
    readPath(root, "domain_motion.tracking.enabled") === false &&
    readPath(root, "domain_motion.tracking.recycle_domain") === false &&
    readPath(root, "domain_motion.nesting.enabled") === false &&
    readPath(root, "dynamics.scheme") === "klemp_wilhelmson_split_cartesian_cgrid" &&
    ["none", "tvd", "weno5"].includes(advection) &&
    ["none", "kessler", "lin", "thompson", "milbrandt"].includes(
      String(readPath(root, "microphysics.scheme") ?? ""),
    ) &&
    (turbulenceScheme === "none" || standardSmagorinsky || standardTke) &&
    readPath(root, "diffusion.scheme") === "explicit" &&
    String(readPath(root, "diffusion.coefficient_source") ?? "constant") === "constant" &&
    standardColumnPhysics &&
    readPath(root, "terrain.scheme") === "none" &&
    readPath(root, "extensions.chaos.scheme") === "none" &&
    readPath(root, "radar.enabled") !== true &&
    (!Array.isArray(products) || products.length === 0) &&
    readPath(root, "boundary_conditions.lateral.west") === "periodic" &&
    readPath(root, "boundary_conditions.lateral.east") === "periodic" &&
    readPath(root, "boundary_conditions.lateral.south") === "periodic" &&
    readPath(root, "boundary_conditions.lateral.north") === "periodic" &&
    readPath(root, "boundary_conditions.bottom") === "rigid_slip" &&
    readPath(root, "boundary_conditions.top") === "rigid" &&
    readPath(root, "damping.enabled") === false &&
    readPath(root, "damping.lateral_sponge_enabled") === false &&
    readPath(root, "damping.w_only_top_sponge") === false &&
    readPath(root, "coriolis.enabled") === false;
}

/* time — v1_loader take_time(). CM1 &param1 / &param2. */
const timeSection: SectionSchema = {
  id: "time",
  label: "Time control",
  description:
    "Run length, dynamics dt, acoustic-substep count, output cadence. " +
    "CM1 namelist origin: &param1 (run_time, dtl) + &param2 (tapfrq).",
  path: "time",
  fields: [
    {
      kind: "enum",
      key: "integrator",
      label: "Time integrator",
      default: "split_explicit",
      options: [
        { value: "split_explicit", label: "Split explicit (KW78)" },
        { value: "rk3", label: "SSPRK3" },
        { value: "rk4", label: "RK4" },
      ],
      disabledWhen: (root) => !rkCatalogEnvironmentReady(root),
      disabledReason:
        "Time-integrator availability is controlled by grid staggering, compute backend, " +
        "and the coupled science selections. RK selection currently requires the flat periodic C-grid CPU dry/Kessler " +
        "TVD/WENO environment with optional standard Smagorinsky and constant explicit " +
        "diffusion, slab-PBL/simple-grey, or Lin/YSU/RRTMG column physics. Under WENO5, " +
        "bounded moisture, tracer, hydrometeor, and TKE fields retain their declared " +
        "conservative TVD/MC reconstruction while WENO5 owns potential temperature. " +
        "Motion, terrain, radar, damping, and Coriolis must be inactive.",
      description:
        "Split explicit is the broad production integrator. SSPRK3 and RK4 stage all " +
        "15 public prognostic fields in their currently admitted flat CPU environment.",
    },
    {
      kind: "float",
      key: "run_time",
      label: "Run time",
      hint: "Total simulated duration, in seconds.",
      description:
        "How long the simulation integrates for. Mirrors CM1's `run_time` " +
        "in &param1. Output cadence is set by `output_interval` below; the " +
        "simulator emits frames every output_interval seconds until run_time.",
      default: 1200,
      min: 1,
      step: 60,
      unit: "s",
    },
    {
      kind: "float",
      key: "dt_large",
      label: "Dynamics dt",
      hint: "Large (slow-tendency) time step. CFL-checked at startup.",
      description:
        "Slow-tendency time step in seconds. The simulator checks CFL " +
        "against the smallest cell spacing and either aborts or auto-" +
        "subcycles when violated. Rule of thumb: dt ≤ 0.6·dx_min / U_max " +
        "with U_max ≈ 50 m/s. Maps to CM1's `dtl` in &param1.",
      default: 0.5,
      min: 0.01,
      step: 0.05,
      unit: "s",
    },
    {
      kind: "int",
      key: "acoustic_substeps",
      label: "Acoustic substeps",
      hint: "0 = auto from acoustic CFL.",
      description:
        "How many small substeps each `dt_large` is divided into for the " +
        "acoustic (sound-wave) terms. 0 = the simulator computes the count " +
        "from acoustic CFL automatically. Mirrors CM1's `nsound` knob.",
      default: 0,
      min: 0,
      max: 32,
    },
    {
      kind: "bool",
      key: "adaptive_dt",
      label: "Adaptive dt",
      description:
        "Lets the simulator shrink dt when CFL spikes during violent " +
        "transients and grow it back when conditions relax. Useful for " +
        "long runs that pass through unstable phases.",
      default: false,
    },
    {
      kind: "float",
      key: "cfl_safety",
      label: "CFL safety factor",
      description:
        "Fraction of the theoretical CFL bound the simulator targets. " +
        "0.7 is the conventional choice. Lower (0.5) for extra-tight " +
        "stability margins; higher (0.9) once stability has been verified.",
      default: 0.7,
      min: 0.1,
      max: 0.95,
      step: 0.05,
    },
    {
      kind: "int",
      key: "output_interval",
      label: "Output interval",
      hint: "Whole seconds between saved states. 0 = final state only.",
      description:
        "How often the simulator writes a snapshot to disk. Mirrors CM1's " +
        "`tapfrq` in &param2. The runtime currently accepts whole seconds. " +
        "Higher playback fps cannot recover motion between saved states. " +
        "Save the fields used together in a view at a matching cadence. " +
        "0 disables periodic output and saves only the final state.",
      default: 30,
      min: 0,
      step: 1,
      unit: "s",
    },
  ],
};

/* dynamics — v1_loader take_dynamics(). scheme is a free string the
 * factory resolves (canonical compound ids + legacy aliases). */
const dynamicsSection: SectionSchema = {
  id: "dynamics",
  label: "Dynamics",
  description:
    "Dynamical kernel selection. The v1 canonical ID is derived from the " +
    "staggering on the Cartesian grid; the legacy short name `cartesian` " +
    "aliases the collocated ID.",
  path: "dynamics",
  fields: [
    {
      kind: "enum",
      key: "scheme",
      // Hidden: scheme is derived from staggering. Kept as a config key because the loader still consumes it.
      visibleWhen: () => false,
      locked: true,
      label: "Scheme (derived)",
      hint: "Derived from staggering — not chosen.",
      description:
        "Which dynamical kernel computes the momentum + thermodynamic " +
        "tendencies. The canonical v1 ID encodes the Cartesian coordinate " +
        "and the staggering. The C-grid variant is production-qualified; " +
        "the collocated A-grid variant exposes its qualified flat CPU dynamics, " +
        "Kessler, and simple-grey family while unsupported physics and accelerator " +
        "combinations remain rejected. " +
        "The legacy short name `cartesian` still resolves via the factory's " +
        "alias map.",
      default: "klemp_wilhelmson_split_cartesian_cgrid",
      options: [
        {
          value: "klemp_wilhelmson_split_cartesian_cgrid",
          label: "Cartesian · C-grid",
          note:
            "Staggered Cartesian C-grid. The validated production path for " +
            "supercell / mesoscale convection.",
        },
        {
          value: "klemp_wilhelmson_split_cartesian_agrid",
          label: "Cartesian · A-grid",
          note:
            "Collocated Cartesian (A-grid). Qualified flat CPU family with " +
            "domain-aware Kessler and simple-grey radiation; other " +
            "active physics and GPU execution are rejected.",
        },
        {
          value: "cartesian",
          label: "Legacy: cartesian",
          note:
            "Legacy alias for klemp_wilhelmson_split_cartesian_agrid.",
        },
      ],
    },
    {
      kind: "bool",
      key: "axisymmetric",
      // Hidden: Cartesian box has no rotation axis; loader keeps a pinned stub.
      visibleWhen: () => false,
      locked: true,
      label: "Axisymmetric (j=0 replicate)",
      hint: "Cartesian box has no rotation axis — always off.",
      description:
        "Legacy axis-replicate optimization that no longer applies; the " +
        "Cartesian box has no rotation axis, so this is pinned off.",
      default: false,
    },
  ],
};

const advectionSection: SectionSchema = {
  id: "advection",
  label: "Advection",
  description: "Transported-scalar face reconstruction used by the production dynamics path.",
  path: "advection",
  fields: [
    {
      kind: "enum",
      key: "scheme",
      label: "Scheme",
      description:
        "None, TVD, and WENO5 are public transport equations. The exact-binary " +
        "catalog and preflight determine which complete science combinations are qualified.",
      default: "tvd",
      options: [
        { value: "none", label: "None (hold transported scalars)" },
        { value: "tvd", label: "TVD" },
        { value: "weno5", label: "WENO5" },
      ],
    },
  ],
};

/* boundary_conditions — v1_loader take_boundary_conditions(). CM1 &param3. */
const LATERAL_BC_OPTIONS = [
  { value: "periodic", label: "Periodic" },
  { value: "radiative", label: "Radiative (KW78 · C-grid)" },
  { value: "zero_gradient", label: "Zero-gradient" },
  { value: "open", label: "Open (legacy auto)" },
  { value: "rigid", label: "Rigid (no-slip)" },
  { value: "rigid_slip", label: "Rigid (free-slip)" },
];

const boundaryConditionsSection: SectionSchema = {
  id: "boundary_conditions",
  label: "Boundary conditions",
  description:
    "Per-face lateral + bottom + top. CM1 namelist origin: &param3 " +
    "(wbc, ebc, sbc, nbc, bbc, tbc).",
  path: "boundary_conditions",
  fields: [
    {
      kind: "enum",
      key: "lateral.west",
      label: "West (x-min)",
      hint: "CM1 wbc.",
      description:
        "Lateral BC on the −x face. `periodic` couples to the east face " +
        "(both must be periodic). `radiative` is the C-grid KW78 outgoing-wave " +
        "closure; `zero_gradient` is an explicit copy closure. Legacy `open` " +
        "resolves to radiative on C-grid and zero-gradient on A-grid. `rigid` " +
        "zeros all velocity; `rigid_slip` zeros only the normal component.",
      default: "open",
      options: LATERAL_BC_OPTIONS,
    },
    {
      kind: "enum",
      key: "lateral.east",
      label: "East (x-max)",
      hint: "CM1 ebc. If west=periodic, this must also be periodic.",
      description:
        "Lateral BC on the +x face. Pair with `lateral.west` for periodic " +
        "configurations (the pair acts as a single periodic boundary).",
      default: "open",
      options: LATERAL_BC_OPTIONS,
    },
    {
      kind: "enum",
      key: "lateral.south",
      label: "South (y-min)",
      hint: "CM1 sbc.",
      description: "Lateral BC on the −y face. Same options as east/west.",
      default: "open",
      options: LATERAL_BC_OPTIONS,
    },
    {
      kind: "enum",
      key: "lateral.north",
      label: "North (y-max)",
      hint: "CM1 nbc.",
      description: "Lateral BC on the +y face. Pair with south for periodic.",
      default: "open",
      options: LATERAL_BC_OPTIONS,
    },
    {
      kind: "enum",
      key: "bottom",
      label: "Bottom (surface)",
      hint: "CM1 bbc=1 (slip) / bbc=2 (no-slip).",
      description:
        "Bottom BC at z=0. `rigid_slip` zeros w but lets u/v slip past the " +
        "surface (free-slip); `rigid_noslip` clamps all three to zero. " +
        "CM1 default is no-slip.",
      default: "rigid_noslip",
      options: [
        { value: "rigid_slip", label: "Rigid (free-slip)" },
        { value: "rigid_noslip", label: "Rigid (no-slip)" },
      ],
    },
    {
      kind: "enum",
      key: "top",
      label: "Top (model lid)",
      hint: "CM1 tbc=1 (rigid) / tbc=2 (rigid with Rayleigh sponge).",
      description:
        "Top BC at the model lid. `rigid` zeros w + zero-gradient on u/v. " +
        "`rigid_with_sponge` adds a Rayleigh damping layer near the top " +
        "(parameters live in §damping). `open` lifts the w constraint.",
      default: "rigid_with_sponge",
      options: [
        { value: "rigid", label: "Rigid lid" },
        { value: "rigid_with_sponge", label: "Rigid + Rayleigh sponge" },
        { value: "open", label: "Open" },
      ],
    },
  ],
};

/* damping — v1_loader take_damping(). CM1 &param7. */
const dampingSection: SectionSchema = {
  id: "damping",
  label: "Rayleigh damping",
  description:
    "Top sponge that absorbs vertically-propagating gravity waves to keep " +
    "them from reflecting off the rigid lid. CM1 origin: &param7.",
  path: "damping",
  disable: {
    path: "damping.enabled",
    offValue: false,
    onValue: true,
    noun: "Rayleigh damping",
  },
  fields: [
    {
      kind: "bool",
      key: "enabled",
      label: "Sponge enabled",
      hint: "Maps to CM1 irdamp=1.",
      description:
        "When on, applies Rayleigh damping in a layer near the model top " +
        "to absorb upward-propagating gravity waves. Required for any run " +
        "with active convection on a finite-depth domain.",
      default: true,
    },
    {
      kind: "float",
      key: "z_d",
      label: "Sponge base height",
      hint: "Altitude where the sponge starts (CM1 zd).",
      description:
        "Altitude in meters at which the sponge layer begins. Damping is " +
        "zero below z_d and ramps to `alpha` at the lid. CM1 default is " +
        "12000 m; for a custom domain pick ~70 % of model top.",
      default: 12000,
      min: 0,
      step: 100,
      unit: "m",
      visibleWhen: (s) => readPath(s, "enabled") === true,
    },
    {
      kind: "float",
      key: "alpha",
      label: "Sponge strength (α)",
      hint: "Inverse e-folding time at the lid (CM1 alph).",
      description:
        "Maximum Rayleigh coefficient at the model top, in 1/s. Larger α " +
        "absorbs more aggressively. CM1's default is ~0.0033 (5 min " +
        "e-folding); Nimbus cgrid presets use α=0.05.",
      default: 0.05,
      min: 0,
      step: 0.001,
      unit: "1/s",
      visibleWhen: (s) => readPath(s, "enabled") === true,
    },
    {
      kind: "enum",
      key: "mode",
      label: "Damping mode",
      hint: "How the sponge is integrated.",
      description:
        "`explicit` applies the damping inside the dynamics RHS each step. " +
        "`implicit` uses a backward-Euler step that is unconditionally " +
        "stable; useful when α·dt approaches 1.",
      default: "explicit",
      options: [
        { value: "explicit", label: "Explicit" },
        { value: "implicit", label: "Implicit" },
      ],
      visibleWhen: (s) => readPath(s, "enabled") === true,
    },
  ],
};

/* coriolis — v1_loader take_coriolis(). CM1 &param12. */
const coriolisSection: SectionSchema = {
  id: "coriolis",
  label: "Coriolis",
  description:
    "Coriolis force on horizontal momentum. CM1 origin: &param12 " +
    "(icor, betaplane, lat0).",
  path: "coriolis",
  disable: {
    path: "coriolis.enabled",
    offValue: false,
    onValue: true,
    noun: "Coriolis",
  },
  fields: [
    {
      kind: "bool",
      key: "enabled",
      label: "Coriolis enabled",
      hint: "CM1 icor=1.",
      description:
        "When on, applies the Coriolis force on horizontal momentum " +
        "(perturbation form, following Rotunno & Klemp 1982). Required for " +
        "any storm-scale or mesoscale simulation.",
      default: false,
    },
    {
      kind: "enum",
      key: "mode",
      label: "Mode",
      hint: "f-plane = constant f; beta-plane = f varies with y.",
      description:
        "f-plane uses a single f value from `latitude_deg` at domain " +
        "center. beta-plane would add ∂f/∂y; currently parsed but treated " +
        "as f-plane internally with a fallback log line.",
      default: "f_plane",
      options: [
        { value: "f_plane", label: "f-plane" },
        { value: "beta_plane", label: "β-plane (f-plane fallback today)" },
      ],
      visibleWhen: (s) => readPath(s, "enabled") === true,
    },
    {
      kind: "float",
      key: "latitude_deg",
      label: "Latitude",
      hint: "f = 2·Ω·sin(latitude). CM1 lat0.",
      description:
        "Latitude in degrees. f = 2·Ω·sin(lat) with Ω = 7.292e-5 rad/s. " +
        "35° N is the Weisman-Klemp 1982 default; 0° gives zero Coriolis " +
        "even when enabled.",
      default: 35,
      min: -90,
      max: 90,
      step: 1,
      unit: "°",
      visibleWhen: (s) => readPath(s, "enabled") === true,
    },
  ],
};

/* microphysics — v1_loader take_microphysics(). CM1 &param4. */
const microphysicsSection: SectionSchema = {
  id: "microphysics",
  label: "Microphysics",
  description: "Cloud / precipitation scheme.",
  path: "microphysics",
  disable: {
    path: "microphysics.scheme",
    offValue: "none",
    onValue: "kessler",
    noun: "microphysics",
  },
  fields: [
    {
      kind: "enum",
      key: "scheme",
      label: "Scheme",
      description:
        "Which microphysical parameterization handles cloud + rain + ice. " +
        "'None' = dry dynamics (CFD only). 'Kessler' is warm-rain only " +
        "(qv → qc → qr, no ice). 'Lin' adds single-moment ice / snow / " +
        "graupel — the standard severe-storm choice. 'Thompson' here is a " +
        "single-moment Lin-class mixed-phase chain (not the WRF aerosol-aware " +
        "double-moment scheme). 'Milbrandt' carries six transported number " +
        "moments and published-form process equations, but remains unavailable " +
        "until its complete independent and coupled qualification campaign passes.",
      default: "none",
      options: [
        { value: "none", label: "None (dry dynamics)" },
        { value: "kessler", label: "Kessler (warm rain)" },
        { value: "lin", label: "Lin (single-moment ice)" },
        { value: "thompson", label: "Thompson (single-moment ice)" },
        { value: "milbrandt", label: "Milbrandt-Yau (double moment)" },
      ],
    },
  ],
};

/* turbulence — v1_loader take_turbulence(). CM1 &param5. */
const turbulenceSection: SectionSchema = {
  id: "turbulence",
  label: "Turbulence (SGS)",
  description: "Sub-grid-scale turbulent diffusion model.",
  path: "turbulence",
  disable: {
    path: "turbulence.scheme",
    offValue: "none",
    onValue: "smagorinsky",
    noun: "turbulence (SGS)",
  },
  fields: [
    {
      kind: "enum",
      key: "scheme",
      label: "Scheme",
      description:
        "Sub-grid-scale (SGS) turbulence closure. 'None' relies on the " +
        "diffusion operator alone. 'Smagorinsky' is the standard LES " +
        "closure: eddy viscosity scales with Δ²·|S|. 'TKE' carries a " +
        "prognostic turbulent-kinetic-energy field (1.5-order closure).",
      default: "smagorinsky",
      options: [
        { value: "none", label: "None" },
        { value: "smagorinsky", label: "Smagorinsky" },
        { value: "tke", label: "TKE (prognostic, 1.5-order)" },
      ],
    },
    {
      kind: "enum",
      key: "mode",
      label: "Mode",
      hint: "3D = full Smagorinsky tensor. 2D = horizontal-only.",
      description:
        "How many dimensions the strain-rate tensor uses. '3D' is full " +
        "anisotropic LES. '2D' restricts to the horizontal plane — cheaper, " +
        "appropriate when vertical mixing is handled by a separate PBL " +
        "scheme.",
      default: "3d",
      options: [
        { value: "3d", label: "3D (full tensor)" },
        { value: "2d", label: "2D (horizontal only)" },
      ],
      visibleWhen: (s) => readPath(s, "scheme") !== "none",
    },
    {
      kind: "float",
      key: "Cs",
      label: "Smagorinsky Cs",
      description:
        "Smagorinsky constant. Classical isotropic-turbulence value is " +
        "0.17–0.21. Lower (0.10) for shear-dominated flows to reduce " +
        "over-dissipation. Cs=0 disables Smagorinsky while keeping the " +
        "scheme name.",
      default: 0.18,
      min: 0,
      max: 0.5,
      step: 0.01,
      visibleWhen: (s) => readPath(s, "scheme") === "smagorinsky",
    },
    {
      kind: "float",
      key: "Pr_t",
      label: "Turbulent Prandtl Pr_t",
      hint: "Eddy viscosity / eddy thermal diffusivity ratio for heat.",
      description:
        "Turbulent Prandtl number — ratio of eddy viscosity to eddy " +
        "thermal diffusivity. Sets how strongly heat mixes relative to " +
        "momentum. Nimbus uses the cloud-LES value 1/3 by default; larger " +
        "values reduce heat mixing relative to momentum.",
      default: 1 / 3,
      min: 0.1,
      max: 5,
      step: 0.05,
      visibleWhen: (s) => readPath(s, "scheme") !== "none",
    },
    {
      kind: "float",
      key: "Sc_t",
      label: "Turbulent Schmidt Sc_t",
      hint: "Eddy viscosity / eddy diffusivity ratio for moisture / tracers.",
      description:
        "Turbulent Schmidt number — analogue of Pr_t for water vapor and " +
        "other scalars. Controls moisture mixing in the boundary layer. " +
        "Nimbus uses the cloud-LES value 1/3 by default.",
      default: 1 / 3,
      min: 0.1,
      max: 5,
      step: 0.05,
      visibleWhen: (s) => readPath(s, "scheme") !== "none",
    },
  ],
};

/* diffusion — v1_loader take_diffusion(). CM1 &param6. (Top-level in v1.) */
const diffusionSection: SectionSchema = {
  id: "diffusion",
  label: "Numerical diffusion",
  description:
    "Live-density finite-volume numerical diffusion. CM1 origin: &param6.",
  path: "diffusion",
  fields: [
    {
      kind: "enum",
      key: "scheme",
      label: "Scheme",
      description:
        "Explicit supports horizontal and vertical diffusion subject to its " +
        "stability limit. Implicit is a CPU, vertical-only backward-Euler " +
        "compatibility path. Set both coefficients to 0 to disable diffusion.",
      default: "explicit",
      options: [
        { value: "explicit", label: "Explicit" },
        { value: "implicit", label: "Implicit vertical" },
      ],
    },
    {
      kind: "int",
      key: "operator_order",
      label: "Operator order",
      hint: "Implemented runtime operator: second-order Laplacian.",
      description:
        "Both schemes implement the same second-order live-density finite-volume vertical Laplacian; explicit also implements the horizontal operator. Other orders are rejected before launch.",
      default: 2,
      min: 2,
      max: 2,
      step: 1,
      visibleWhen: (s) => readPath(s, "scheme") !== "none",
    },
    {
      kind: "float",
      key: "K_h",
      label: "Horizontal K",
      hint: "Density-weighted explicit horizontal diffusion coefficient.",
      description:
        "Explicit horizontal second-order diffusion coefficient (m²/s). Scalars use cell density; momentum uses staggering-correct velocity control-volume density on both A- and C-grids. Implicit horizontal diffusion requires zero.",
      default: 0,
      min: 0,
      max: 1000,
      step: 1,
      unit: "m²/s",
      visibleWhen: (s) => readPath(s, "scheme") === "explicit",
    },
    {
      kind: "float",
      key: "K_v",
      label: "Vertical K (K_v)",
      hint: "Vertical eddy diffusivity. 0 = inviscid.",
      description:
        "Vertical diffusivity coefficient (m²/s). Both schemes use the same live-density finite-volume fluxes, declared layer thicknesses, closed outer numerical flux, and staggering-specific momentum control volumes. " +
        "Tornado simulations often run K_v ≈ 50–200 m²/s; pure dry-dynamics " +
        "LES can run inviscid (0) and rely on Smagorinsky.",
      default: 0,
      min: 0,
      step: 10,
      unit: "m²/s",
      visibleWhen: (s) => readPath(s, "scheme") !== "none",
    },
    {
      kind: "enum",
      key: "apply_to",
      label: "Apply diffusion to",
      description:
        "Which fields receive the diffusion operator. 'Scalars' = θ, qv, " +
        "qc, qr only (the usual sub-grid mixing target). 'Scalars + " +
        "momentum' also smooths u, v, w (numerical viscosity for stability).",
      default: "scalars",
      options: [
        { value: "scalars", label: "Scalars only" },
        { value: "scalars_and_momentum", label: "Scalars + momentum" },
      ],
      visibleWhen: (s) => readPath(s, "scheme") !== "none",
    },
  ],
};

/* surface — v1_loader take_surface(). CM1 &param9. */
const surfaceSection: SectionSchema = {
  id: "surface",
  label: "Surface",
  description:
    "Surface scheme and whether its fluxes feed back into the dynamics. " +
    "CM1 origin: &param9.",
  path: "surface",
  disable: {
    path: "surface.scheme",
    offValue: "none",
    onValue: "slab",
    noun: "the surface scheme",
  },
  fields: [
    {
      kind: "enum",
      key: "scheme",
      label: "Surface scheme",
      description:
        "Which surface parameterization handles the lower boundary. " +
        "'Slab' is the Nimbus default (single-layer prognostic surface " +
        "temperature, simple drag); 'None' disables surface coupling " +
        "(free-slip + no fluxes).",
      default: "slab",
      options: [
        { value: "slab", label: "Slab" },
        { value: "none", label: "None" },
      ],
    },
    {
      kind: "bool",
      key: "apply_fluxes",
      label: "Apply surface fluxes",
      hint: "Whether sensible-/latent-heat fluxes feed the dynamics.",
      description:
        "When on, surface sensible + latent heat fluxes are added to the " +
        "lowest-layer θ and qv tendencies. Off = surface scheme advances " +
        "its own temperature but the dynamics is decoupled.",
      default: true,
      visibleWhen: (s) => readPath(s, "scheme") !== "none",
    },
    {
      kind: "float", key: "z0m_m", label: "Momentum roughness", default: 0.1,
      min: 0.000001, step: 0.01, unit: "m",
      description: "Aerodynamic roughness length used for momentum exchange.",
    },
    {
      kind: "float", key: "z0h_m", label: "Thermal roughness", default: 0.01,
      min: 0.000001, step: 0.001, unit: "m",
      description: "Aerodynamic roughness length used for scalar heat exchange.",
    },
    {
      kind: "float", key: "albedo", label: "Shortwave albedo", default: 0.2,
      min: 0, max: 1, step: 0.01,
      description: "Fraction of downwelling shortwave radiation reflected by the surface.",
    },
    {
      kind: "float", key: "emissivity", label: "Longwave emissivity", default: 0.95,
      min: 0, max: 1, step: 0.01,
      description: "Surface longwave emissivity; one minus this value is reflected downwelling LW.",
    },
    {
      kind: "float", key: "tsk_k", label: "Surface skin temperature", default: -1,
      min: -1, max: 400, step: 0.5, unit: "K",
      description: "Use -1 to hold the initial surface-air state for surface fluxes. Active longwave replaces this with an explicit 150–400 K skin.",
    },
    {
      kind: "float", key: "qsfc_kgkg", label: "Surface vapor mixing ratio", default: -1,
      min: -1, max: 0.1, step: 0.001, unit: "kg/kg",
      description: "Use -1 to hold the initial surface-air moisture; otherwise prescribe 0–0.1 kg/kg for surface moisture fluxes.",
      visibleWhen: (s) => readPath(s, "scheme") !== "none",
    },
  ],
};

/* boundary_layer — v1_loader take_boundary_layer(). WRF-lineage PBL. */
const boundaryLayerSection: SectionSchema = {
  id: "boundary_layer",
  label: "Boundary layer (PBL)",
  description: "Planetary-boundary-layer parameterization (WRF lineage).",
  path: "boundary_layer",
  disable: {
    path: "boundary_layer.scheme",
    offValue: "none",
    onValue: "slab",
    noun: "the PBL scheme",
  },
  fields: [
    {
      kind: "enum",
      key: "scheme",
      label: "Scheme",
      description:
        "Planetary boundary-layer (PBL) parameterization. 'Slab' is the " +
        "Nimbus default and matches the default slab surface coupling; " +
        "'None' disables PBL mixing. YSU and MYNN are WRF-lineage schemes; " +
        "review their exact-binary maturity evidence before science runs.",
      default: "slab",
      options: [
        { value: "slab", label: "Slab mixed layer" },
        { value: "none", label: "None" },
        {
          value: "ysu",
          label: "YSU (Yonsei University)",
          note:
            "WRF-lineage nonlocal PBL. Confirm it's wired up in this build " +
            "before using for production.",
        },
        {
          value: "mynn",
          label: "MYNN (Mellor–Yamada–Nakanishi–Niino)",
          note:
            "WRF-lineage TKE PBL. Confirm it's wired up in this build " +
            "before using for production.",
        },
      ],
    },
  ],
};

/* radiation — v1_loader take_radiation(). CM1 &param8 + Nimbus forcing. */
const radiationSection: SectionSchema = {
  id: "radiation",
  label: "Radiation",
  description: "Radiative transfer with an explicit idealized solar-forcing contract.",
  path: "radiation",
  disable: {
    path: "radiation.scheme",
    offValue: "none",
    onValue: "simple_grey",
    noun: "radiation",
  },
  fields: [
    {
      kind: "enum",
      key: "scheme",
      label: "Scheme",
      description:
        "Radiative-transfer parameterization. 'None' skips radiation " +
        "entirely (fine for short tornado-scale runs). 'Simple grey' is a " +
        "toy column model (Beer-Lambert exponential decay) — useful for " +
        "smoke-testing diurnal heating, not for quantitative flux results.",
      default: "none",
      options: [
        { value: "none", label: "None" },
        {
          value: "simple_grey",
          label: "Simple grey (toy / diagnostic only)",
          note:
            "Beer–Lambert exponential decay, not a real RT solver. Don't " +
            "trust quantitative LW/SW fluxes for science runs.",
        },
      ],
    },
    {
      kind: "enum",
      key: "solar_forcing",
      label: "Solar forcing",
      description: "Explicit idealized diurnal geometry; no location or clock is inferred.",
      default: "idealized_diurnal",
      options: [{ value: "idealized_diurnal", label: "Idealized diurnal cycle" }],
      visibleWhen: (s) => readPath(s, "scheme") === "simple_grey",
    },
    {
      kind: "float", key: "latitude_deg", label: "Latitude", default: 35,
      min: -90, max: 90, step: 1, unit: "°",
      description: "Latitude used by the idealized solar zenith calculation.",
      visibleWhen: (s) => readPath(s, "scheme") === "simple_grey",
    },
    {
      kind: "float", key: "declination_deg", label: "Solar declination", default: 15,
      min: -23.5, max: 23.5, step: 0.5, unit: "°",
      description: "Fixed solar declination for the idealized day.",
      visibleWhen: (s) => readPath(s, "scheme") === "simple_grey",
    },
    {
      kind: "float", key: "local_solar_time_at_start_h", label: "Initial local solar time",
      default: 0, min: 0, max: 23.999, step: 0.25, unit: "h",
      description: "Local solar hour at simulation time zero.",
      visibleWhen: (s) => readPath(s, "scheme") === "simple_grey",
    },
    {
      kind: "bool", key: "do_lw", label: "Longwave", default: true,
      description: "Enable the grey longwave band.",
      visibleWhen: (s) => readPath(s, "scheme") === "simple_grey",
    },
    {
      kind: "bool", key: "do_sw", label: "Shortwave", default: true,
      description: "Enable the Beer–Lambert shortwave band.",
      visibleWhen: (s) => readPath(s, "scheme") === "simple_grey",
    },
    {
      kind: "float", key: "cadence_s", label: "Radiation cadence", default: 300,
      min: 0.001, step: 10, unit: "s",
      description: "Time between radiation tendency updates.",
      visibleWhen: (s) => readPath(s, "scheme") === "simple_grey",
    },
    {
      kind: "bool", key: "clear_sky_only", label: "Clear-sky optical contract", default: true,
      description: "Must remain enabled until cloud and precipitation optics exist.",
      visibleWhen: (s) => readPath(s, "scheme") === "simple_grey",
    },
    {
      kind: "float", key: "tau_lw_ref", label: "Reference LW optical depth", default: 6,
      min: 0, step: 0.1,
      description: "Reference longwave optical depth.",
      visibleWhen: (s) => readPath(s, "scheme") === "simple_grey",
    },
    {
      kind: "float", key: "tau_sw_ref", label: "Reference SW optical depth", default: 0.22,
      min: 0, step: 0.01,
      description: "Reference shortwave optical depth.",
      visibleWhen: (s) => readPath(s, "scheme") === "simple_grey",
    },
    {
      kind: "float", key: "n_lw", label: "LW pressure exponent", default: 4,
      min: 0.001, step: 0.1,
      description: "Pressure-profile exponent for longwave optical depth.",
      visibleWhen: (s) => readPath(s, "scheme") === "simple_grey",
    },
    {
      kind: "float", key: "n_sw", label: "SW pressure exponent", default: 2,
      min: 0.001, step: 0.1,
      description: "Pressure-profile exponent for shortwave optical depth.",
      visibleWhen: (s) => readPath(s, "scheme") === "simple_grey",
    },
    {
      kind: "int", key: "terrain_sw_cosine_nodes", label: "Terrain SW cosine nodes", default: 4,
      min: 4, max: 32, step: 1,
      description: "Gauss–Legendre cosine nodes for mapped simple-grey or RRTMG transport.",
      visibleWhen: (s) => ["simple_grey", "rrtmg"].includes(String(readPath(s, "scheme"))),
    },
    {
      kind: "int", key: "terrain_sw_azimuth_nodes", label: "Terrain SW azimuth nodes", default: 8,
      min: 8, max: 128, step: 4,
      description: "Azimuth samples for mapped Lambertian transport; must be a multiple of four.",
      visibleWhen: (s) => ["simple_grey", "rrtmg"].includes(String(readPath(s, "scheme"))),
    },
    {
      kind: "int", key: "terrain_sw_maximum_bounces", label: "Terrain SW bounce ceiling", default: 32,
      min: 1, max: 256, step: 1,
      description: "Maximum terrain re-reflection iterations before the run fails closed.",
      visibleWhen: (s) => ["simple_grey", "rrtmg"].includes(String(readPath(s, "scheme"))),
    },
    {
      kind: "float", key: "terrain_sw_relative_residual_tolerance", label: "Terrain SW relative residual", default: 1e-10,
      min: 0, max: 1e-6, step: 1e-11,
      description: "Relative remaining reflected-power tolerance; runtime also enforces its accuracy ceiling.",
      visibleWhen: (s) => ["simple_grey", "rrtmg"].includes(String(readPath(s, "scheme"))),
    },
    {
      kind: "float", key: "terrain_sw_absolute_residual_tolerance_w", label: "Terrain SW absolute residual", default: 1e-9,
      min: 0, step: 1e-9, unit: "W",
      description: "Absolute remaining reflected-power tolerance used with the relative tolerance.",
      visibleWhen: (s) => ["simple_grey", "rrtmg"].includes(String(readPath(s, "scheme"))),
    },
  ],
};

/* radar — v1_loader take_radar(). Product selection is rendered by the
 * descriptor-backed RadarCatalogControl because radar.products is an ordered
 * set rather than a single enum token. */
const radarSection: SectionSchema = {
  id: "radar",
  label: "Radar products",
  description: "Synthetic reflectivity, radial velocity, and polarimetric diagnostics.",
  path: "radar",
  fields: [
    {
      kind: "enum",
      key: "reflectivity_tier",
      label: "Reflectivity operator",
      default: "fast_da",
      options: [
        { value: "fast_da", label: "Fast data-assimilation" },
        { value: "psd_moment", label: "PSD moment" },
      ],
      description: "Reflectivity moment closure used when reflectivity is selected.",
    },
    {
      kind: "enum",
      key: "velocity_sampling",
      label: "Velocity sampling",
      default: "point",
      options: [
        { value: "point", label: "Point sample" },
        { value: "beam_volume", label: "Beam volume" },
      ],
      description: "Spatial sampling contract for synthetic radial velocity.",
    },
    {
      kind: "enum",
      key: "zdr_tier",
      label: "Polarimetric operator",
      default: "polarimetric_fo",
      options: [
        { value: "simple", label: "Simple" },
        { value: "polarimetric_fo", label: "Polarimetric first order" },
        { value: "zhang2021_rain_mp", label: "Zhang 2021 rain moments" },
      ],
      description: "KDP and rhoHV require the Zhang 2021 rain-moment operator.",
    },
    {
      kind: "float", key: "location.x_m", label: "Radar x", default: 0,
      step: 100, unit: "m", description: "Radar x coordinate in the simulation frame.",
    },
    {
      kind: "float", key: "location.y_m", label: "Radar y", default: 0,
      step: 100, unit: "m", description: "Radar y coordinate in the simulation frame.",
    },
    {
      kind: "float", key: "location.z_m", label: "Radar altitude", default: 0,
      step: 10, unit: "m", description: "Radar height in the simulation frame.",
    },
    {
      kind: "bool",
      key: "apply_scatterer_correction",
      label: "Scatterer correction",
      default: true,
      description: "Apply the selected operator's hydrometeor scatterer correction.",
    },
  ],
};

/* terrain — v1_loader take_terrain(). The strict schema retains active
 * requests even while the exact-binary capability ledger disables the
 * unqualified bell/Schar choices in Studio. */
const terrainSection: SectionSchema = {
  id: "terrain",
  label: "Terrain",
  description:
    "Lower-boundary topography and its terrain-following vertical coordinate. " +
    "The exact-binary capability catalog controls whether a non-flat scheme is selectable.",
  path: "terrain",
  fields: [
    {
      kind: "enum",
      key: "scheme",
      label: "Terrain shape",
      default: "none",
      showDisabledReasons: true,
      options: [
        { value: "none", label: "Flat terrain" },
        { value: "bell", label: "Bell mountain" },
        { value: "schar", label: "Schär mountain" },
      ],
      description:
        "Analytic lower-boundary shape. Non-flat options stay disabled until " +
        "the selected dynamics/backend combination has qualified metric-aware operators.",
    },
    {
      kind: "enum",
      key: "coordinate",
      label: "Vertical coordinate",
      default: "btf",
      options: [
        { value: "btf", label: "Basic terrain-following" },
        { value: "smoothed", label: "Exponentially smoothed" },
        { value: "cartesian", label: "Cartesian (flat only)" },
      ],
      description:
        "Maps computational levels to physical height above topography. " +
        "Active terrain requires BTF or smoothed coordinates.",
    },
    {
      kind: "bool",
      key: "compute_derivatives",
      label: "Compute terrain derivatives",
      default: true,
      description: "Required for metric-consistent gradients and fluxes on active terrain.",
    },
    {
      kind: "bool",
      key: "compute_metrics",
      label: "Compute terrain metrics",
      default: true,
      description: "Required for Jacobian-weighted terrain-following dynamics.",
    },
    {
      kind: "float",
      key: "smoothing_decay",
      label: "Smoothing decay",
      default: 0.1,
      min: 0,
      step: 0.05,
      description: "Vertical decay rate for the smoothed terrain coordinate.",
    },
    {
      kind: "float",
      key: "bell.h0",
      label: "Bell height",
      default: 1000,
      min: 0,
      step: 100,
      unit: "m",
      visibleWhen: (s) => readPath(s, "scheme") === "bell",
    },
    {
      kind: "float",
      key: "bell.a",
      label: "Bell half-width",
      default: 5000,
      min: 0.001,
      step: 100,
      unit: "m",
      visibleWhen: (s) => readPath(s, "scheme") === "bell",
    },
    {
      kind: "bool",
      key: "bell.axisymmetric",
      label: "Axisymmetric bell",
      default: false,
      visibleWhen: (s) => readPath(s, "scheme") === "bell",
    },
    {
      kind: "float",
      key: "schar.H",
      label: "Schär height",
      default: 1000,
      min: 0,
      step: 100,
      unit: "m",
      visibleWhen: (s) => readPath(s, "scheme") === "schar",
    },
    {
      kind: "float",
      key: "schar.a",
      label: "Schär envelope width",
      default: 5000,
      min: 0.001,
      step: 100,
      unit: "m",
      visibleWhen: (s) => readPath(s, "scheme") === "schar",
    },
    {
      kind: "float",
      key: "schar.ell",
      label: "Schär wavelength",
      default: 4000,
      min: 0.001,
      step: 100,
      unit: "m",
      visibleWhen: (s) => readPath(s, "scheme") === "schar",
    },
  ],
};

/* initial_conditions.sounding — v1_loader take_initial_conditions(). */
const soundingSection: SectionSchema = {
  id: "sounding",
  label: "Sounding",
  description:
    "Base thermodynamic profile. The procedural CAPE source exposes only " +
    "the controls it actually consumes; Parametric targets exposes the " +
    "CAPE/CIN/LCL/LFC/EL target family; File loads an external sounding.",
  path: "initial_conditions.sounding",
  fields: [
    {
      kind: "enum",
      key: "type",
      label: "Source",
      description:
        "Where the initial atmospheric profile comes from. Procedural CAPE " +
        "tunes a compact profile to CAPE; Parametric targets translates the " +
        "diagnostic target family; File loads an external sounding text file.",
      default: "parametric_cape",
      options: [
        { value: "parametric_cape", label: "Procedural CAPE" },
        { value: "parametric_targets", label: "Parametric targets" },
        { value: "file", label: "File…" },
      ],
    },
    {
      kind: "string",
      key: "file_path",
      label: "Sounding file",
      placeholder: "configs/soundings/wk1982.txt",
      description:
        "Absolute or repo-relative path to a sounding file. Nimbus detects " +
        "native CM1 input_sounding files by content, including canonical " +
        "extensionless filenames; the original file is read without conversion.",
      default: "",
      visibleWhen: (s) => readPath(s, "type") === "file",
    },
    {
      kind: "float",
      key: "cape_jkg",
      label: "Target CAPE",
      hint: "Convective Available Potential Energy.",
      description:
        "CAPE request for a parametric sounding (J/kg). The two parametric " +
        "sources consume this value; file soundings derive CAPE from the " +
        "loaded column.",
      default: 2500,
      min: 0,
      step: 100,
      unit: "J/kg",
      visibleWhen: (s) => readPath(s, "type") !== "file",
    },
    {
      kind: "float",
      key: "cin_jkg",
      label: "Target CIN",
      hint: "Convective inhibition (cap strength).",
      description:
        "Target Convective INhibition (J/kg). Sets the strength of the " +
        "capping inversion. 50 is moderate; 200+ holds storms back until " +
        "afternoon.",
      default: 50,
      min: 0,
      step: 25,
      unit: "J/kg",
      max: 300,
      visibleWhen: (s) => readPath(s, "type") === "parametric_targets",
    },
    {
      kind: "float",
      key: "lcl_m",
      label: "Target LCL height",
      hint: "Lifted condensation level (cloud base).",
      description:
        "Target Lifted Condensation Level (m) — where a surface parcel " +
        "saturates when lifted dry-adiabatically. Sets the cloud base. " +
        "1200 is moderate; lower (500) for tropical-moist runs.",
      default: 1200,
      min: 1,
      step: 100,
      unit: "m",
      visibleWhen: (s) => readPath(s, "type") === "parametric_targets",
    },
    {
      kind: "float",
      key: "lfc_m",
      label: "Target LFC height",
      hint: "Level of free convection.",
      description:
        "Target Level of Free Convection (m) — where a lifted parcel " +
        "becomes warmer than the environment and rises on its own. LFC " +
        "above LCL means a capping inversion is in play.",
      default: 2000,
      min: 50,
      step: 100,
      unit: "m",
      visibleWhen: (s) => readPath(s, "type") === "parametric_targets",
    },
    {
      kind: "float",
      key: "el_m",
      label: "Target EL height",
      hint: "Equilibrium level (anvil top).",
      description:
        "Target Equilibrium Level (m) — where the parcel cools back to " +
        "environmental temperature, marking the storm anvil top. 12 km is " +
        "mid-latitude; 16+ km is tropical-deep convection.",
      default: 12000,
      min: 8000,
      step: 500,
      unit: "m",
      visibleWhen: (s) => readPath(s, "type") === "parametric_targets",
    },
    {
      kind: "float",
      key: "tropopause_m",
      label: "Tropopause height",
      hint: "Procedural profile transition height.",
      description:
        "Height where the procedural CAPE profile transitions into its " +
        "stratospheric layer. This is a profile boundary, not a requested " +
        "or diagnosed equilibrium level.",
      default: 12000,
      min: 8000,
      step: 500,
      unit: "m",
      visibleWhen: (s) => readPath(s, "type") === "parametric_cape",
    },
    {
      kind: "float",
      key: "surface_theta_k",
      label: "Surface θ",
      hint: "Surface potential temperature.",
      description:
        "Potential temperature at the surface (K). Sets the bottom anchor " +
        "of the parametric sounding. 300 K is roughly 27 °C at 1000 hPa, a " +
        "warm-season convective baseline.",
      default: 300,
      min: 250,
      max: 350,
      step: 0.5,
      unit: "K",
      visibleWhen: (s) => readPath(s, "type") !== "file",
    },
    {
      kind: "float",
      key: "moisture_qv_kgkg",
      label: "Surface qv",
      hint: "Surface water-vapor mixing ratio.",
      description:
        "Water vapor mixing ratio at the surface (kg/kg). Higher qv adds " +
        "CAPE and moisture for precipitation. 0.014 (14 g/kg) is humid-" +
        "Plains convective; 0.020 is tropical-moist.",
      default: 0.014,
      min: 0.004,
      max: 0.024,
      step: 0.001,
      unit: "kg/kg",
      visibleWhen: (s) => readPath(s, "type") === "parametric_cape",
    },
  ],
};

/* initial_conditions.hodograph — v1_loader take_initial_conditions(). */
const hodographSection: SectionSchema = {
  id: "hodograph",
  label: "Hodograph",
  description: "Wind profile vs height — surface, 1 km, 6 km layers.",
  path: "initial_conditions.hodograph",
  fields: [
    {
      kind: "enum",
      key: "type",
      label: "Type",
      description:
        "How the wind profile is built. Auto consumes the u/v column embedded " +
        "in a file sounding when present. Custom uses the explicit anchors " +
        "below; WK param generates the analytic Weisman-Klemp profile.",
      default: "custom",
      options: [
        { value: "auto", label: "Auto (embedded file winds)" },
        { value: "custom", label: "Custom (u/v anchors below)" },
        { value: "wk_param", label: "Weisman-Klemp 1982 (analytic)" },
      ],
    },
    {
      kind: "float",
      key: "u_sfc_ms",
      label: "u surface",
      description:
        "Zonal (east-west) wind at the surface. Sets the lowest anchor of " +
        "the hodograph.",
      default: 0,
      step: 0.5,
      unit: "m/s",
    },
    {
      kind: "float",
      key: "v_sfc_ms",
      label: "v surface",
      description: "Meridional (north-south) wind at the surface.",
      default: 0,
      step: 0.5,
      unit: "m/s",
    },
    {
      kind: "float",
      key: "u_1km_ms",
      label: "u @ 1 km",
      description:
        "Zonal wind at 1 km AGL. (Surface → 1 km shear is the prime " +
        "tornado-genesis indicator.)",
      default: 0,
      step: 0.5,
      unit: "m/s",
    },
    {
      kind: "float",
      key: "v_1km_ms",
      label: "v @ 1 km",
      description: "Meridional wind at 1 km AGL.",
      default: 0,
      step: 0.5,
      unit: "m/s",
    },
    {
      kind: "float",
      key: "u_6km_ms",
      label: "u @ 6 km",
      description:
        "Zonal wind at 6 km AGL. (Surface → 6 km bulk shear drives " +
        "supercell organization.)",
      default: 0,
      step: 0.5,
      unit: "m/s",
    },
    {
      kind: "float",
      key: "v_6km_ms",
      label: "v @ 6 km",
      description: "Meridional wind at 6 km AGL.",
      default: 0,
      step: 0.5,
      unit: "m/s",
    },
  ],
};

/* initial_conditions.trigger — v1_loader take_initial_conditions().
 * NOTE: bubble/vortex centers are in METERS (center_*_m). */
const triggerSection: SectionSchema = {
  id: "trigger",
  label: "Trigger",
  description:
    "Initial perturbation — warm bubble or Rankine vortex.",
  path: "initial_conditions.trigger",
  fields: [
    {
      kind: "enum",
      key: "type",
      label: "Type",
      description:
        "What initial perturbation kicks off the simulation. 'Bubble' is " +
        "the classic Klemp-Wilhelmson warm-bubble convection initiator. " +
        "'Vortex seed' drops in a Rankine-combined vortex for tornado-" +
        "mature tests.",
      default: "none",
      options: [
        { value: "none", label: "None" },
        { value: "bubble", label: "Warm bubble" },
        { value: "vortex_seed", label: "Vortex seed (Rankine)" },
      ],
    },

    // Warm bubble subgroup
    {
      kind: "float",
      key: "bubble.dtheta_k",
      label: "Δθ",
      description:
        "Peak potential-temperature anomaly inside the bubble (K). Drives " +
        "the initial buoyancy. 2 K is gentle; 4 K is the WK1982 standard; " +
        "6 K is aggressive.",
      default: 2,
      step: 0.5,
      unit: "K",
      visibleWhen: (s) => readPath(s, "type") === "bubble",
    },
    {
      kind: "float",
      key: "bubble.cloud_water_kgkg",
      label: "Cloud-water seed",
      description:
        "Peak cloud-liquid mixing ratio in the Gaussian bubble. Use zero " +
        "for a thermal-only trigger or a small positive value to initialize " +
        "a resolved condensate anomaly for microphysics experiments.",
      default: 0,
      step: 0.0001,
      unit: "kg/kg",
      visibleWhen: (s) => readPath(s, "type") === "bubble",
    },
    {
      kind: "float",
      key: "bubble.radius_km",
      label: "Bubble radius",
      description:
        "Horizontal half-width of the warm bubble (km). Larger bubbles " +
        "seed larger initial updrafts. 10 km matches the WK1982 baseline.",
      default: 10,
      step: 1,
      unit: "km",
      visibleWhen: (s) => readPath(s, "type") === "bubble",
    },
    {
      kind: "float",
      key: "bubble.center_x_m",
      label: "Bubble center x",
      hint: "Distance of the bubble centroid from the x origin (m).",
      description:
        "Horizontal offset of the bubble's centroid in x (m), measured from " +
        "the domain origin.",
      default: 0,
      step: 100,
      unit: "m",
      visibleWhen: (s) => readPath(s, "type") === "bubble",
    },
    {
      kind: "float",
      key: "bubble.center_y_m",
      label: "Bubble center y",
      hint: "Distance of the bubble centroid from the y origin (m).",
      description:
        "Horizontal offset of the bubble's centroid in y (m), measured from " +
        "the domain origin.",
      default: 0,
      step: 100,
      unit: "m",
      visibleWhen: (s) => readPath(s, "type") === "bubble",
    },
    {
      kind: "float",
      key: "bubble.center_z_m",
      label: "Bubble center z",
      description:
        "Height of the bubble's centroid above the surface (m). 1500 m " +
        "sits the perturbation just above the boundary layer.",
      default: 1500,
      step: 100,
      unit: "m",
      visibleWhen: (s) => readPath(s, "type") === "bubble",
    },

    // Vortex seed subgroup
    {
      kind: "float",
      key: "vortex_seed.r_max_m",
      label: "r_max",
      description:
        "Radius of maximum tangential velocity for the Rankine vortex (m). " +
        "Inside r_max the wind rises linearly (solid-body); outside it " +
        "decays as 1/r.",
      default: 6000,
      step: 250,
      unit: "m",
      visibleWhen: (s) => readPath(s, "type") === "vortex_seed",
    },
    {
      kind: "float",
      key: "vortex_seed.v_max_ms",
      label: "v_max",
      description:
        "Peak tangential wind at r=r_max (m/s). 40 is moderate tornado; " +
        "80+ is EF4/EF5 territory.",
      default: 40,
      step: 1,
      unit: "m/s",
      visibleWhen: (s) => readPath(s, "type") === "vortex_seed",
    },
    {
      kind: "float",
      key: "vortex_seed.z_top_m",
      label: "z_top",
      description:
        "Top of the vortex column (m). Above z_top the tangential wind is " +
        "zeroed.",
      default: 6000,
      step: 250,
      unit: "m",
      visibleWhen: (s) => readPath(s, "type") === "vortex_seed",
    },
    {
      kind: "float",
      key: "vortex_seed.center_x_m",
      label: "Vortex center x",
      hint: "Center offset of the vortex axis along x (m).",
      description:
        "Horizontal offset of the vortex axis in x (m), measured from the " +
        "domain origin.",
      default: 0,
      step: 100,
      unit: "m",
      visibleWhen: (s) => readPath(s, "type") === "vortex_seed",
    },
    {
      kind: "float",
      key: "vortex_seed.center_y_m",
      label: "Vortex center y",
      hint: "Center offset of the vortex axis along y (m).",
      description:
        "Horizontal offset of the vortex axis in y (m), measured from the " +
        "domain origin.",
      default: 0,
      step: 100,
      unit: "m",
      visibleWhen: (s) => readPath(s, "type") === "vortex_seed",
    },
    {
      kind: "float",
      key: "vortex_seed.center_z_m",
      label: "Vortex center z",
      hint: "Vertical center of the vortex column (m).",
      description:
        "Vertical midpoint of the vortex column (m). The wind decays from " +
        "this height to z_top above and to 0 below.",
      default: 3000,
      step: 250,
      unit: "m",
      visibleWhen: (s) => readPath(s, "type") === "vortex_seed",
    },
  ],
};

/* output — v1_loader take_output(). */
const outputSection: SectionSchema = {
  id: "output",
  label: "Output",
  description: "What to write and in which format. Cadence is time.output_interval.",
  path: "output",
  fields: [
    {
      kind: "enum",
      key: "preset",
      label: "Field preset",
      description:
        "Which fields get written per snapshot. 'Minimal' (u, v, w, p, ρ, θ, " +
        "qv, qr) is the cheapest and is the bounded qualified-ZFP set. " +
        "'Presentation' adds dBZ, vorticity, theta-" +
        "prime — the set Plot Studio screenshots use. 'Studio catalog' is a " +
        "superset of presentation carrying every remaining product Plot " +
        "Studio's own views read: the vorticity budget, all hydrometeor " +
        "species, both horizontal vorticity components, the pressure " +
        "decomposition, divergence, and the column/precipitation products. " +
        "'Dynamics full' writes every registered field, which is several times " +
        "the volume for fields no view requests.",
      default: "minimal",
      options: [
        { value: "minimal", label: "Minimal (8 core fields)" },
        { value: "presentation", label: "Presentation set" },
        { value: "studio_full", label: "Studio catalog (unblocks every native view)" },
        { value: "dynamics_full", label: "Dynamics full (verbose)" },
        { value: "radar_ready", label: "Radar reflectivity + velocity" },
        { value: "radar_polarimetric_ready", label: "Radar polarimetric moments" },
      ],
    },
    {
      kind: "enum",
      key: "format",
      label: "Format",
      description:
        "Output file format. 'NumPy 3D' writes one .npy per field per step. " +
        "NetCDF writes one self-describing history.nc per step with lossless " +
        "compression that preserves the saved values. The 'ZFP' " +
        "tiers are lossy compressed. Availability comes from the exact " +
        "simulator binary selected in Studio.",
      default: "npy_3d",
      options: [
        { value: "npy_3d", label: "NumPy 3D (.npy)" },
        { value: "netcdf", label: "NetCDF4, lossless (.nc)" },
        { value: "zfp_low", label: "ZFP — low fidelity (1e-2)" },
        { value: "zfp_medium", label: "ZFP — medium fidelity (1e-3)" },
        { value: "zfp_high", label: "ZFP — high fidelity (1e-4)" },
      ],
    },
    {
      kind: "int",
      key: "netcdf_deflate_level",
      label: "Lossless compression",
      hint: "0 disables compression; 2 is the default.",
      description:
        "Higher levels trade compression time for smaller files. " +
        "Every level preserves the saved field values exactly.",
      default: 2,
      min: 0,
      max: 9,
      visibleWhen: (section) => section.format === "netcdf",
    },
    {
      kind: "bool",
      key: "netcdf_shuffle",
      label: "Improve compression",
      hint: "Reorder bytes before compression without changing values.",
      description:
        "Groups similar bytes to help lossless compression. " +
        "The original byte order is restored when the file is read.",
      default: true,
      visibleWhen: (section) => section.format === "netcdf",
    },
    {
      kind: "int",
      key: "netcdf_target_chunk_kib",
      label: "Read chunk target",
      unit: "KiB",
      hint: "Controls the blocks used for compression and partial reads.",
      description:
        "Sets the target size of a stored block. Smaller blocks can suit " +
        "slice and subvolume reads; this does not change grid resolution.",
      default: 1024,
      min: 1,
      visibleWhen: (section) => section.format === "netcdf",
    },
    {
      kind: "bool",
      key: "async_io",
      label: "Async I/O",
      hint: "Write snapshots on a background thread.",
      description:
        "Uses a bounded background queue to overlap writing with simulation. " +
        "It retains additional snapshot data and waits when the writer falls behind. " +
        "Turn it off to reduce queued snapshot memory.",
      default: true,
    },
    {
      kind: "bool",
      key: "live_shm",
      label: "Live shared-memory stream",
      hint: "Mirror outputs to a shared-memory region for live viz.",
      description:
        "Streams selected fields to a POSIX shared-memory segment as " +
        "they're computed — separate from the on-disk snapshot. Lets an " +
        "external viz tool render live without polling the disk.",
      default: false,
    },
    {
      kind: "string",
      key: "live_shm_fields",
      label: "Live fields",
      hint: "Comma-separated negotiated volume fields.",
      description:
        "The simulator validates the complete, duplicate-free field contract " +
        "before allocating the shared-memory region.",
      default: "w, reflectivity_dbz, vorticity_z",
      visibleWhen: (s) => readPath(s, "live_shm") === true,
    },
    {
      kind: "float",
      key: "live_shm_interval_s",
      label: "Live publish interval",
      unit: "s",
      hint: "Simulation-time cadence for coherent live bundles.",
      description:
        "Publishes the initial state, then the first completed model step at " +
        "or beyond each boundary. This does not change solver or science cadence.",
      default: 1,
      min: 0.001,
      step: 0.1,
      visibleWhen: (s) => readPath(s, "live_shm") === true &&
        readPath(s, "live_shm_full_rate") !== true,
    },
    {
      kind: "bool",
      key: "live_shm_full_rate",
      label: "Publish every model step",
      hint: "Explicitly opt in to full-rate field transposition.",
      description:
        "Overrides the interval and publishes one coherent live bundle after " +
        "every completed model step. Integration and durable output are unchanged.",
      default: false,
      visibleWhen: (s) => readPath(s, "live_shm") === true,
    },
    {
      kind: "bool",
      key: "trajectories",
      label: "Lagrangian trajectories",
      hint: "Follow parcels through the storm and write their paths.",
      description:
        "Integrates a lattice of parcels online and writes the whole run's " +
        "paths once, to trajectories.npy plus a trajectories.json provenance " +
        "record. This is a run-level artifact rather than a per-step field, so " +
        "no field preset produces it — Studio's Trajectories and Trajectory " +
        "budget views stay hidden until a run turns this on. Each sample " +
        "carries the parcel's vertical velocity, vertical vorticity, and the " +
        "tilting, stretching, and baroclinic terms acting on it.",
      default: false,
    },
    {
      kind: "float",
      key: "trajectory_seed_time_s",
      label: "Trajectory seed time",
      unit: "s",
      hint: "Simulation time at which the parcel lattice is released.",
      description:
        "Parcels are released once, at this time. Seeding after the storm has " +
        "organized follows air into an existing updraft; seeding at zero " +
        "follows the environment the storm forms out of.",
      default: 0,
      min: 0,
      step: 1,
      visibleWhen: (s) => readPath(s, "trajectories") === true,
    },
    {
      kind: "int",
      key: "trajectory_count",
      label: "Trajectory seed count",
      hint: "Parcels per horizontal direction, so the lattice is count × count wide.",
      description:
        "Horizontal seed count in each direction. The seed box is clipped to " +
        "the physical domain, so a small grid still produces a populated " +
        "artifact rather than an empty one.",
      default: 5,
      min: 1,
      step: 1,
      visibleWhen: (s) => readPath(s, "trajectories") === true,
    },
    {
      kind: "float",
      key: "trajectory_sample_interval_s",
      label: "Trajectory sample interval",
      unit: "s",
      hint: "Record parcel diagnostics at this cadence while integrating every model step.",
      description:
        "A positive interval bounds diagnostic history growth without changing " +
        "parcel motion. Zero preserves legacy recording on every completed model step.",
      default: 0,
      min: 0,
      step: 1,
      visibleWhen: (s) => readPath(s, "trajectories") === true,
    },
    {
      kind: "int",
      key: "trajectory_max_samples_per_parcel",
      label: "Retained trajectory samples",
      hint: "Maximum latest samples retained for each parcel.",
      description:
        "Includes the seed sample. Once full, the oldest sample is retired before " +
        "publication and checkpointing, keeping memory and checkpoint size bounded.",
      default: 16384,
      min: 1,
      step: 1,
      visibleWhen: (s) => readPath(s, "trajectories") === true,
    },
    {
      kind: "int",
      key: "keep_last_frames",
      label: "Retained history frames",
      hint: "Newest completed frames retained per history series; predictive parents are preserved automatically.",
      description:
        "A positive value bounds per-field history fanout. Zero is an exploratory legacy setting and is rejected for qualified ZFP history.",
      default: 0,
      min: 0,
      step: 1,
    },
    {
      kind: "int",
      key: "minimum_free_space_mib",
      label: "History free-space reserve",
      unit: "MiB",
      hint: "Space retained after admitting the next estimated frame.",
      description:
        "Nimbus checks the next frame estimate plus this reserve before publication. Qualified ZFP requires a positive value.",
      default: 0,
      min: 0,
      step: 256,
    },
    {
      kind: "int",
      key: "maximum_retained_history_files",
      label: "Retained history file ceiling",
      hint: "Fail before integration when the conservative per-series file estimate exceeds this value.",
      description:
        "Counts current per-field payloads, manifests, validation sidecars, batched index files, geometry, and fixed metadata. It does not imply packed storage.",
      default: 0,
      min: 0,
      step: 16,
    },
    {
      kind: "enum",
      key: "zfp_mode",
      label: "ZFP error mode",
      description:
        "Accuracy mode supplies an absolute error bound in each field's own " +
        "units and is the only qualified science mode. Precision and fixed-rate " +
        "modes remain available only through Raw YAML for exploratory visualization.",
      default: "accuracy",
      options: [{ value: "accuracy", label: "Accuracy (bounded)" }],
      visibleWhen: (s) => String(readPath(s, "format")).startsWith("zfp"),
    },
    {
      kind: "int",
      key: "zfp_keyframe_interval",
      label: "ZFP keyframe interval",
      hint: "0 makes every frame independently decodable; 4 bounds the default delta chain.",
      description:
        "Frames between keyframes store residuals against decoded parents. " +
        "The manifest records the exact parent chain; smaller intervals improve " +
        "random access at a modest storage cost.",
      default: 4,
      min: 0,
      step: 1,
      visibleWhen: (s) => String(readPath(s, "format")).startsWith("zfp"),
    },
    {
      kind: "bool",
      key: "zfp_predictive_delta",
      label: "Predictive ZFP deltas",
      description:
        "Use two decoded parent frames to predict the next field. The qualified " +
        "writer and Studio reader verify the exact two-parent chain.",
      default: true,
      visibleWhen: (s) =>
        String(readPath(s, "format")).startsWith("zfp") &&
        Number(readPath(s, "zfp_keyframe_interval")) > 0,
    },
    {
      kind: "enum",
      key: "zfp_field_execution",
      label: "ZFP field execution",
      description:
        "Serial is the qualified default. Field-parallel keeps libzfp serial " +
        "inside isolated worker workspaces and commits staged artifacts in field order.",
      default: "serial",
      options: [
        { value: "serial", label: "Serial (default)" },
        { value: "field_parallel", label: "Field-parallel" },
      ],
      visibleWhen: (s) => String(readPath(s, "format")).startsWith("zfp"),
    },
    {
      kind: "int",
      key: "zfp_field_workers",
      label: "ZFP field workers",
      hint: "0 selects a backend-aware bounded count.",
      description:
        "CPU runs reserve solver capacity; accelerator runs may use otherwise-idle CPU cores.",
      default: 0,
      min: 0,
      max: 16,
      step: 1,
      visibleWhen: (s) =>
        String(readPath(s, "format")).startsWith("zfp") &&
        readPath(s, "zfp_field_execution") === "field_parallel",
    },
    {
      kind: "int",
      key: "zfp_field_scratch_mib",
      label: "ZFP parallel scratch budget",
      unit: "MiB",
      description:
        "Hard admission ceiling for isolated codec buffers and transactional decoded-history staging.",
      default: 64,
      min: 1,
      max: 4096,
      step: 16,
      visibleWhen: (s) =>
        String(readPath(s, "format")).startsWith("zfp") &&
        readPath(s, "zfp_field_execution") === "field_parallel",
    },
    {
      kind: "float",
      key: "zfp_sparse_threshold",
      label: "ZFP sparse threshold",
      description:
        "Values below this threshold are zeroed before compression. Qualified " +
        "science requires exactly zero because thresholding adds loss outside " +
        "the codec error bound.",
      default: 0,
      min: 0,
      step: 1e-9,
      visibleWhen: (s) => String(readPath(s, "format")).startsWith("zfp"),
    },
    {
      kind: "bool",
      key: "zfp_float16_prefilter",
      label: "Float16 ZFP prefilter",
      description:
        "Pre-quantization reduces storage but adds unbounded field-dependent " +
        "loss. It is disabled and forbidden in qualified science output.",
      default: false,
      visibleWhen: (s) => String(readPath(s, "format")).startsWith("zfp"),
    },
    {
      kind: "bool",
      key: "zfp_per_field_tolerances",
      label: "Unit-aware ZFP budgets",
      description:
        "Apply the qualified absolute-error budget separately in m/s, K, Pa, " +
        "kg/m³, and kg/kg. Required for qualified ZFP history.",
      default: true,
      visibleWhen: (s) => String(readPath(s, "format")).startsWith("zfp"),
    },
    ...([
      ["u", "u error budget", 1e-4, "m/s"],
      ["v", "v error budget", 1e-4, "m/s"],
      ["w", "w error budget", 1e-4, "m/s"],
      ["theta", "θ error budget", 1e-3, "K"],
      ["p", "Pressure error budget", 5e-2, "Pa"],
      ["rho", "Density error budget", 1e-6, "kg/m³"],
      ["qv", "Water-vapor error budget", 1e-8, "kg/kg"],
      ["qr", "Rain-water error budget", 1e-8, "kg/kg"],
    ] as const).map(([field, label, value, unit]) => ({
      kind: "float" as const,
      key: `zfp_field_tolerance.${field}`,
      label,
      description:
        `Maximum configured ZFP absolute error for ${field}. Larger values ` +
        "are rejected in qualified execution.",
      default: value,
      min: Number.MIN_VALUE,
      step: value / 10,
      unit,
      visibleWhen: (s: ConfigObject) =>
        String(readPath(s, "format")).startsWith("zfp") &&
        readPath(s, "zfp_per_field_tolerances") === true,
    })),
  ],
};

/* extensions.compute — v1_loader take_extensions(). */
const computeSection: SectionSchema = {
  id: "compute",
  label: "Compute backend",
  description: "Where the per-step kernels run. Nimbus extension (not CM1).",
  path: "extensions.compute",
  fields: [
    {
      kind: "enum",
      key: "backend",
      label: "Backend",
      hint: "Choose CPU, Vulkan, or Metal. Fallback is controlled separately.",
      description:
        "Which compute engine runs supported kernels. Every selected science " +
        "profile is checked against the exact simulator binary before launch.",
      default: "cpu",
      options: [
        { value: "cpu", label: "CPU" },
        { value: "vulkan", label: "Vulkan" },
        { value: "metal", label: "Metal" },
      ],
    },
    {
      kind: "enum",
      key: "memory_admission",
      label: "Memory admission",
      description: "Whether an accelerator memory estimate is advisory or blocks launch.",
      default: "warn",
      options: [
        { value: "warn", label: "Warn" },
        { value: "reject", label: "Reject" },
      ],
    },
    {
      kind: "bool",
      key: "allow_fallback",
      label: "Allow CPU fallback",
      description: "Permit unsupported accelerator work to execute on CPU. Qualified execution disables this.",
      default: true,
      disabledWhen: (root) =>
        readPath(root, "extensions.execution.mode") === "qualified" ||
        readPath(root, "extensions.compute.backend") === "cpu",
      disabledReason: "CPU fallback is inapplicable on CPU and forbidden for qualified execution.",
    },
    {
      kind: "bool",
      key: "validate_parity",
      label: "Validate accelerator parity",
      description: "Replay sampled accelerator dispatches through the CPU oracle and compare their outputs.",
      default: false,
      visibleWhen: (section) => readPath(section, "backend") !== "cpu",
    },
    {
      kind: "bool",
      key: "cuda_assist",
      label: "NVIDIA CUDA assist",
      description: "Use CUDA for supported compute kernels through Vulkan. Requires a compatible NVIDIA GPU and driver.",
      default: false,
      visibleWhen: (section) => readPath(section, "cuda_assist") === true ||
        ["vulkan", "vk", "gpu"].includes(String(readPath(section, "backend")).toLowerCase()),
    },
    {
      kind: "bool",
      key: "gpu_apply_microphysics",
      label: "Apply microphysics on accelerator",
      description: "Request the native microphysics state-commit path when supported by the selected profile and guard mode.",
      default: true,
      visibleWhen: (section) => readPath(section, "backend") !== "cpu",
    },
    {
      kind: "bool",
      key: "gpu_apply_slow",
      label: "Apply slow dynamics on accelerator",
      description: "Request the native slow-dynamics state-commit path when admitted by the exact binary.",
      default: false,
      visibleWhen: (section) => readPath(section, "backend") !== "cpu",
    },
    {
      kind: "bool",
      key: "gpu_apply_sgs",
      label: "Apply SGS on accelerator",
      description: "Request the native SGS state-commit path when supported by the selected profile and guard mode.",
      default: false,
      visibleWhen: (section) => readPath(section, "backend") !== "cpu",
    },
  ],
};

/* extensions.chaos — v1_loader take_extensions(). */
const chaosSection: SectionSchema = {
  id: "chaos",
  label: "Chaos / ensemble",
  description:
    "Stochastic perturbation infrastructure for ensembles. Nimbus " +
    "extension (not CM1).",
  path: "extensions.chaos",
  fields: [
    {
      kind: "enum",
      key: "scheme",
      label: "Scheme",
      description:
        "Stochastic perturbation scheme. 'None' disables chaos. " +
        "'Initial conditions' perturbs only the IC; 'Boundary layer' " +
        "perturbs BL tendencies; 'Full stochastic' runs the complete SKEB-" +
        "style forcing.",
      default: "none",
      options: [
        { value: "none", label: "None" },
        { value: "initial_conditions", label: "Initial conditions only" },
        { value: "boundary_layer", label: "Boundary layer" },
        { value: "full_stochastic", label: "Full stochastic (SKEB)" },
      ],
    },
    {
      kind: "int",
      key: "seed",
      label: "Random seed",
      hint: "RNG seed for reproducibility.",
      description:
        "Seed for the chaos RNG. Pinning this guarantees reproducible " +
        "perturbations across runs. Useful for ensemble verification.",
      default: 0,
      min: 0,
      step: 1,
      visibleWhen: (s) => readPath(s, "scheme") !== "none",
    },
    {
      kind: "int",
      key: "member_id",
      label: "Member id",
      hint: "Ensemble member index.",
      description:
        "Index of this run within an ensemble. Different member_ids with " +
        "the same seed produce statistically independent realizations.",
      default: 0,
      min: 0,
      step: 1,
      visibleWhen: (s) => readPath(s, "scheme") !== "none",
    },
  ],
};

/* extensions.execution — explicit scientific recovery policy. */
const executionSection: SectionSchema = {
  id: "execution",
  label: "Execution policy",
  description:
    "Choose whether Nimbus may recover for exploration or must fail closed for qualified evidence.",
  path: "extensions.execution",
  fields: [
    {
      kind: "enum",
      key: "mode",
      label: "Run mode",
      hint: "Controls recovery, substitution, and repair behavior.",
      description:
        "Exploratory mode discloses recoverable scheme failures and may continue with zero tendencies. " +
        "Qualified mode aborts on scheme exceptions or state repairs and disables compute and sounding fallback.",
      default: "exploratory",
      options: [
        { value: "exploratory", label: "Exploratory — disclose and continue" },
        { value: "qualified", label: "Qualified — fail closed" },
      ],
    },
  ],
};

/* extensions.checkpoint — Nimbus-only portable continuation workflow. */
const checkpointSection: SectionSchema = {
  id: "checkpoint",
  label: "Checkpoints and continuation",
  description:
    "Save portable Nimbus state between completed timesteps or continue it as a new, provenance-linked branch. This does not read or execute CM1 restart files.",
  path: "extensions.checkpoint",
  fields: [
    {
      kind: "int",
      key: "every_s",
      label: "Checkpoint cadence",
      unit: "s",
      hint: "Simulation-time interval; 0 disables checkpoints.",
      description:
        "Nimbus commits a transactional portable checkpoint at this simulation-time cadence and at successful run completion. History-output cadence is independent.",
      default: 0,
      min: 0,
      step: 1,
    },
    {
      kind: "string",
      key: "directory",
      label: "Checkpoint directory",
      hint: "Optional; defaults to the run output directory's checkpoints folder.",
      description:
        "Choose a local destination for checkpoint binaries and integrity sidecars. Leave blank for the branch-safe per-run default.",
      default: "",
      placeholder: "<run output>/checkpoints",
    },
    {
      kind: "string",
      key: "resume_from",
      label: "Continue from checkpoint",
      hint: "Optional path to a .nimbus checkpoint with its matching .json sidecar.",
      description:
        "Studio launches a new output branch from this completed-timestep checkpoint. Nimbus verifies integrity and exact scientific/grid compatibility before replacing any live state.",
      default: "",
      placeholder: "/path/to/checkpoint_step_000000100.nimbus",
    },
  ],
};

/* extensions.validation — v1_loader take_extensions(). */
const validationSection: SectionSchema = {
  id: "validation",
  label: "Validation guards",
  description:
    "Runtime sanity checks (NaN/CFL guards). Nimbus extension (not CM1).",
  path: "extensions.validation",
  fields: [
    {
      kind: "enum",
      key: "guard_mode",
      label: "Guard mode",
      hint: "How runtime sanity checks behave.",
      description:
        "Behavior of runtime validity guards (NaN detection, CFL violation, " +
        "etc.). 'Off' = no checks (fastest). 'Warn' = log violations, keep " +
        "running. 'Error' = halt the run on any violation. Qualified execution " +
        "pins this to Error.",
      default: "warn",
      disabledWhen: (root) =>
        readPath(root, "extensions.execution.mode") === "qualified",
      disabledReason:
        "Pinned to Error because Execution policy is set to Qualified. " +
        "Choose a different run mode to change it.",
      options: [
        { value: "off", label: "Off" },
        { value: "warn", label: "Warn (log only)" },
        { value: "error", label: "Error (abort on violation)" },
      ],
    },
  ],
};

/* SCHEMA_SECTIONS order mirrors v1_loader.cpp's section walk + §20. */
export const SCHEMA_SECTIONS: SectionSchema[] = [
  gridSection,
  timeSection,
  dynamicsSection,
  advectionSection,
  boundaryConditionsSection,
  dampingSection,
  coriolisSection,
  microphysicsSection,
  turbulenceSection,
  diffusionSection,
  surfaceSection,
  boundaryLayerSection,
  radiationSection,
  radarSection,
  terrainSection,
  soundingSection,
  hodographSection,
  triggerSection,
  outputSection,
  computeSection,
  chaosSection,
  executionSection,
  checkpointSection,
  validationSection,
];

/* ---------- Helpers ---------- */

/**
 * Read a dot-separated path out of a config object. Used by `visibleWhen`
 * predicates and by SchemaForm when reading a field value.
 */
export function readPath(obj: ConfigObject, path: string): ConfigValue | undefined {
  const parts = path.split(".");
  let cur: ConfigValue | undefined = obj;
  for (const p of parts) {
    if (cur === null || cur === undefined) return undefined;
    if (typeof cur !== "object" || Array.isArray(cur)) return undefined;
    cur = (cur as ConfigObject)[p];
  }
  return cur;
}

/**
 * Write a dot-separated path, mutating-in-place after a structural clone.
 * Returns the new root object so consumers can replace the state.
 */
export function writePath(
  obj: ConfigObject,
  path: string,
  value: ConfigValue,
): ConfigObject {
  const root = structuredClone(obj);
  const parts = path.split(".");
  let cur: ConfigObject = root;
  for (let i = 0; i < parts.length - 1; i++) {
    const p = parts[i];
    const next = cur[p];
    if (typeof next !== "object" || next === null || Array.isArray(next)) {
      cur[p] = {};
    }
    cur = cur[p] as ConfigObject;
  }
  cur[parts[parts.length - 1]] = value;
  return root;
}

/** Remove one leaf path from a structural clone. Empty parent objects are
 * retained; YAML serialization already handles those consistently. */
function deletePath(obj: ConfigObject, path: string): ConfigObject {
  const root = structuredClone(obj);
  const parts = path.split(".");
  let cur: ConfigObject = root;
  for (let i = 0; i < parts.length - 1; i++) {
    const next = cur[parts[i]];
    if (typeof next !== "object" || next === null || Array.isArray(next)) {
      return root;
    }
    cur = next as ConfigObject;
  }
  delete cur[parts[parts.length - 1]];
  return root;
}

/**
 * Compose the canonical dynamics-scheme id from the staggering the user picks.
 * The scheme is derived, not chosen, so the user never navigates a scheme
 * list. Nimbus Studio is Cartesian-only (no rotation axis, so never
 * axisymmetric); only the staggering varies. Output ids match the canonical
 * names in src/init/scheme_profile.cpp, so the loader accepts every value the
 * form can produce.
 */
export function deriveSchemeFromGeometry(staggering: string): string {
  return staggering === "c_grid"
    ? "klemp_wilhelmson_split_cartesian_cgrid"
    : "klemp_wilhelmson_split_cartesian_agrid";
}

/**
 * Re-derive the coupled fields after a staggering change. The coordinate is
 * pinned to Cartesian and axisymmetric to false (the Cartesian box has no
 * rotation axis); only the staggering is the user's choice (C-grid is the
 * default + validated path, collocated remains available as legacy). The
 * read-only dynamics.scheme is then computed from the staggering.
 */
function reconcileGeometry(config: ConfigObject): ConfigObject {
  const staggering = String(readPath(config, "grid.staggering") ?? "");
  let next = writePath(config, "grid.coordinate_system", "cartesian");
  next = writePath(next, "dynamics.axisymmetric", false);
  next = writePath(next, "dynamics.scheme", deriveSchemeFromGeometry(staggering));
  return next;
}

const GEOMETRY_PATHS = new Set(["grid.staggering"]);

const SOUNDING_TYPE_PATH = "initial_conditions.sounding.type";

const SOUNDING_DEFAULTS: Record<string, ScalarValue> = {
  cape_jkg: 2500,
  cin_jkg: 50,
  lcl_m: 1200,
  lfc_m: 2000,
  el_m: 12000,
  tropopause_m: 12000,
  surface_theta_k: 300,
  moisture_qv_kgkg: 0.014,
  file_path: "",
};

/**
 * Keep the editor's sounding subtree source-specific. The strict C++ loader
 * intentionally rejects controls that the selected source cannot consume;
 * leaving now-hidden values behind when the source selector changes would
 * therefore make Studio emit a config that cannot run.
 */
function reconcileSoundingSource(
  config: ConfigObject,
  requestedType?: string,
): ConfigObject {
  const type = requestedType ?? String(readPath(config, SOUNDING_TYPE_PATH) ?? "");
  const prefix = "initial_conditions.sounding.";
  const allSourceFields = Object.keys(SOUNDING_DEFAULTS);
  const allowed =
    type === "parametric_targets"
      ? new Set(["cape_jkg", "cin_jkg", "lcl_m", "lfc_m", "el_m", "surface_theta_k"])
      : type === "file"
        ? new Set(["file_path"])
        : new Set(["cape_jkg", "tropopause_m", "surface_theta_k", "moisture_qv_kgkg"]);

  let next = config;
  for (const key of allSourceFields) {
    const path = prefix + key;
    if (!allowed.has(key)) {
      next = deletePath(next, path);
    }
  }
  for (const key of allowed) {
    const path = prefix + key;
    if (readPath(next, path) === undefined) {
      next = writePath(next, path, SOUNDING_DEFAULTS[key]);
    }
  }
  return next;
}

/**
 * Write a single field value, applying geometry coupling. Changing the
 * staggering re-derives the dynamics scheme (and re-pins coordinate=cartesian
 * + axisymmetric=false), so the form can never produce an invalid combination
 * and the scheme field stays read-only. Every other field writes straight
 * through.
 */
export function applyFieldChange(
  config: ConfigObject,
  path: string,
  value: ConfigValue,
): ConfigObject {
  const next = writePath(config, path, value);
  if (path === "extensions.compute.backend" && !["vulkan", "vk", "gpu"].includes(String(value).toLowerCase())) {
    return writePath(next, "extensions.compute.cuda_assist", false);
  }
  if (GEOMETRY_PATHS.has(path)) return reconcileGeometry(next);
  if (path === "extensions.execution.mode") {
    const withGuard = writePath(
      next,
      "extensions.validation.guard_mode",
      value === "qualified" ? "error" : "warn",
    );
    const withFallback = writePath(
      withGuard,
      "extensions.compute.allow_fallback",
      value !== "qualified",
    );
    return withFallback;
  }
  if (path === "radiation.scheme" && value === "simple_grey") {
    const currentSkin = readPath(next, "surface.tsk_k");
    const withSkin = typeof currentSkin !== "number" || currentSkin < 150
      ? writePath(next, "surface.tsk_k", 300)
      : next;
    return withSkin;
  }
  if (path === SOUNDING_TYPE_PATH) {
    return reconcileSoundingSource(next, String(value));
  }
  return next;
}

/**
 * Build a config with each section's defaults applied. Used as the starting
 * point for a fresh "New config" and as the merge target for uploads, so
 * missing fields fall back to schema defaults rather than `undefined`.
 *
 * Output is stamped `schema: nimbus.v1` and contains only the keys
 * v1_loader.cpp consumes (no extra top-level sections).
 */
export function defaultsForSchema(): ConfigObject {
  let result: ConfigObject = {
    schema: "nimbus.v1",
  };
  for (const section of SCHEMA_SECTIONS) {
    for (const field of section.fields) {
      const fullKey = qualifyKey(section, field);
      result = writePath(result, fullKey, fieldDefaultAsScalar(field));
    }
  }
  return reconcileSoundingSource(result);
}

export function reconcileFormConfig(config: ConfigObject): ConfigObject {
  let next = reconcileGeometry(config);
  next = reconcileSoundingSource(
    next,
    String(readPath(next, "initial_conditions.sounding.type") ?? "parametric_cape"),
  );
  if (readPath(next, "extensions.execution.mode") === "qualified") {
    next = writePath(next, "extensions.validation.guard_mode", "error");
    next = writePath(next, "extensions.compute.allow_fallback", false);
  }
  return next;
}

/** Section.path + field.key with proper join. Keys starting with `/` are
 * absolute (top-level keys, e.g. `/schema`). */
export function qualifyKey(section: SectionSchema, field: FieldSchema): string {
  if (field.key.startsWith("/")) return field.key.slice(1);
  if (!section.path) return field.key;
  return `${section.path}.${field.key}`;
}

function fieldDefaultAsScalar(field: FieldSchema): ScalarValue {
  return field.default;
}
