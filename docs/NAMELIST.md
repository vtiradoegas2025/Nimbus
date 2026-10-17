# Nimbus namelist: complete v1 configuration reference

This is the parameter reference for a Nimbus `schema: nimbus.v1` input file.
It includes **every strict-v1 key**, including options that are guarded,
unfinished, conditional or retained only to reject an obsolete request.
Nimbus reads nested YAML mappings; it does not read a CM1 Fortran `&param`
namelist. Similar names do not promise the same equations, defaults, units or
numeric option codes as CM1. The [mathematics guide](MATHEMATICS.pdf) explains
those differences and the [architecture guide](ARCHITECTURE.pdf) explains how
the selected components connect.

The inventory contains **302 finite keys**: 288 fixed keys and 14 permitted
chaos-amplitude map entries. It also documents the open
`output.zfp_field_tolerance.<field>` mapping. The command-line and legacy
compatibility sections distinguish other input routes from this v1 namespace.
Studio's form is a view of this configuration, not its complete definition;
some settings require editing YAML.

## How to read this reference

**When omitted** reports the C++ configuration default or the explicit
conditional rule, not a Studio draft or a resolution preset. A required value
has no usable omission default. A dormant parameter can have a typed default
even when writing it explicitly for the wrong scheme is rejected. Consequently,
this whole reference must not be pasted into a single configuration: mutually
exclusive and rejected keys are intentionally listed together.

The type column describes the scalar/list parser. `Real > 0` and `real ≥ 0`
also require finite values. Section and field notes add coupled restrictions;
scientific admission can be narrower still. Enum aliases are listed explicitly
when the parser retains them. Unknown v1 keys are rejected. Parameter names
are case-sensitive: `Cs`, `Pr_t`, `Sc_t`, `K_h`, `K_v`, `H` and `h0` retain
their spelling.

Use two-space-indented mappings, plain scalar values and inline lists such as
`[w, reflectivity_dbz]`. The native reader is a restricted indentation/flattening
reader, not a general YAML implementation: do not use anchors, multiline
scalars or block sequences. It treats the first `#` as a comment delimiter
even inside quotes. Avoid duplicate keys; duplicate flattened names currently
overwrite earlier values. Avoid `#` inside paths or other string values.

For example, this shows the syntax of the required fields; it is not a
recommended physical experiment or a qualified configuration:

```yaml
schema: nimbus.v1
grid:
  nx: 64
  ny: 64
  nz: 64
  dx: 250
  dy: 250
  dz: 250
time:
  run_time: 60
  dt_large: 1
  output_interval: 10
```

Start an actual experiment from a [maintained configuration](../configs/README.md).
For the [build-and-open check](BUILDING.md#build-and-open-only), no input file,
simulation or external coefficient tables are necessary.

## Availability and scientific status

A recognized key, a parsed configuration, an admitted profile and a Fully
Qualified capability are different things. Use the actual executable's
`--scheme-catalog=json` and ordinary validation path for the complete combination.
`--validate-only` prepares a one-dimensional initial-condition context but runs
no timesteps; it is separate from merely opening Studio.

The runtime value `extensions.execution.mode: qualified` names a strict execution
policy. It is **not** a claim of Fully Qualified science. That status additionally
requires all applicable engineering, bounded tests, long campaigns, checked
evidence and independent scientific review. The required reporting states are
Engineering incomplete, Ready, Campaign validated, Externally reviewed and
Fully qualified; see [AGENTS.md](../AGENTS.md).

Tracked broader gaps include remaining native physics combinations, moist
two-way nesting, the complete bounded-history ZFP scope, the retained vertical-only
SGS gate, Milbrandt triple moments and exact catalog/CLI/Studio parity. Some
narrow profiles within these families can be admitted without completing the
whole tracked item. The [readiness registry](../src/core/runtime/readiness.cpp)
and runtime catalog retain that distinction. Acknowledging a limitation does
not authorize a rejected option.

Two particularly easy mistakes are documented in the tables: the loader rejects
an explicit `microphysics.milbrandt.triple_moment` key even when false, and the
vortex seed's `center_z_m` currently does not control its vertical placement.

## Parameter index
- [Schema](#schema)
- [Grid](#grid)
- [Time](#time)
- [Domain motion](#domain-motion)
- [Dynamics](#dynamics)
- [Boundary conditions](#boundary-conditions)
- [Damping](#damping)
- [Coriolis](#coriolis)
- [Microphysics](#microphysics)
- [Turbulence](#turbulence)
- [Advection](#advection)
- [Diffusion](#diffusion)
- [Surface](#surface)
- [Boundary layer](#boundary-layer)
- [Radiation](#radiation)
- [Terrain](#terrain)
- [Initial conditions](#initial-conditions)
- [Radar](#radar)
- [Output](#output)
- [Extensions](#extensions)
- [Per-field ZFP map](#per-field-zfp-map)
- [Command-line companion](#command-line-companion)
- [Legacy compatibility](#legacy-compatibility)

## Schema

Every v1 file begins with the explicit format stamp.

Top-level scalar; do not nest it under another `schema` key.

| Key | Type / choices | When omitted | Meaning and restrictions |
| --- | --- | --- | --- |
| [`schema`](../src/init/config/v1_loader.cpp#L436) | String | **Required: `nimbus.v1`** | Required format stamp: `nimbus.v1`. This is Nimbus YAML, not a Fortran `&param` namelist. |

## Grid

Grid dimensions, spacing and staggering define the representation on which every operator acts. Counts include the allocation conventions described in the mathematics guide; face/cell and physical/halo distinctions still apply.

Prefix: `grid`. Each name below is relative to that prefix.

| Key | Type / choices | When omitted | Meaning and restrictions |
| --- | --- | --- | --- |
| [`coordinate_system`](../src/init/config/v1_loader.cpp#L456) | `cartesian` | `cartesian` | Coordinate basis. The public v1 loader accepts Cartesian geometry. Aliases: `cart` → `cartesian`. |
| [`dx`](../src/init/config/v1_loader.cpp#L461) | Real > 0 | **Required** | Horizontal x spacing in m, strictly positive. |
| [`dy`](../src/init/config/v1_loader.cpp#L462) | Real > 0 | **Required** | Horizontal y spacing in m, strictly positive. |
| [`dz`](../src/init/config/v1_loader.cpp#L463) | Real > 0 | **Required** | Nominal computational vertical spacing in m, strictly positive. Physical thicknesses on mapped grids come from their metrics. |
| [`nx`](../src/init/config/v1_loader.cpp#L458) | Integer > 0 | **Required** | Allocated x dimension, positive integer. Physical cells, closure planes and staggered faces must be interpreted through the grid owner; do not infer physical inventory solely from this count. |
| [`ny`](../src/init/config/v1_loader.cpp#L459) | Integer > 0 | **Required** | Allocated y dimension, positive integer; the physical-domain conventions are the same as for `nx`. |
| [`nz`](../src/init/config/v1_loader.cpp#L460) | Integer > 0 | **Required** | Allocated vertical dimension, positive integer. Active terrain also requires consistent `ztop` and `dz`. |
| [`staggering`](../src/init/config/v1_loader.cpp#L457) | `collocated`, `c_grid` | `c_grid` | Velocity placement: `c_grid` uses cell faces; `collocated` uses the A-grid. This changes the discrete operators and the admitted backend/physics combinations. Aliases: `a_grid` → `collocated`; `cgrid` → `c_grid`. |
| [`ztop`](../src/init/config/v1_loader.cpp#L464) | Real > 0 | `16000` | Domain-top height in m. For active terrain it must equal `nz * dz`, and terrain must remain below it. |

## Time

The large timestep, acoustic subdivisions and output interval are different clocks. Actual runtime CFL and scientific admission can impose stronger restrictions than the scalar loader.

Prefix: `time`. Each name below is relative to that prefix.

| Key | Type / choices | When omitted | Meaning and restrictions |
| --- | --- | --- | --- |
| [`acoustic_substeps`](../src/init/config/v1_loader.cpp#L479) | Integer ≥ 0 | `6` | Requested minimum acoustic subdivisions. `0` asks for automatic selection; positive requests can be increased by the acoustic CFL check. |
| [`adaptive_dt`](../src/init/config/v1_loader.cpp#L480) | Boolean | `false` | Allow timestep adaptation. Fixed-cadence diagnostics and some exact admitted profiles require this to remain false. |
| [`cfl_safety`](../src/init/config/v1_loader.cpp#L481) | Real > 0 | `0.7` | Positive dimensionless CFL safety factor. A loader-accepted value is not by itself a stability guarantee. |
| [`dt_large`](../src/init/config/v1_loader.cpp#L475) | Real > 0 | **Required** | Large timestep in s, strictly positive. Stability, cadence and the selected dynamics impose additional limits. |
| [`integrator`](../src/init/config/v1_loader.cpp#L470) | `split_explicit`, `rk3`, `rk4` | `split_explicit` | Time-integration selection. The scheme name alone does not admit every dynamics/backend combination; use the exact runtime catalog. Aliases: `ssprk3` → `rk3`. |
| [`output_interval`](../src/init/config/v1_loader.cpp#L483) | Whole seconds | **Required** | Periodic history interval in s, whole-second-valued. `0` disables periodic history output; other output streams have separate cadence controls. |
| [`run_time`](../src/init/config/v1_loader.cpp#L474) | Whole seconds | **Required** | Requested run duration in s. Must be positive, whole-second-valued and representable by the runtime scheduler; `60.0` is accepted but `60.5` is not. |

## Domain motion

Domain motion separates the coordinate-frame velocity, a preflight lifetime envelope, diagnostic storm tracking, discrete state recycling and child-grid nesting. These controls interact but are not interchangeable: diagnostic thresholds do not alter the simulated storm, and enabling tracking does not authorize recycling. Parser-valid values still pass through the complete-profile admission rules; especially, arbitrary moving nests and moist two-way combinations are not made Ready by selecting a listed value.

Prefix: `domain_motion`. Each name below is relative to that prefix.

| Key | Type / choices | When omitted | Meaning and restrictions |
| --- | --- | --- | --- |
| [`frame_velocity.u_ms`](../src/init/config/v1_loader.cpp#L493) | Finite real | `0` | Earth-relative x velocity of the translating coordinate frame. It must be finite and zero for stationary or tracked_window mode; prescribed frame motion is distinct from the environmental wind and predicted storm motion. Units: m/s. |
| [`frame_velocity.v_ms`](../src/init/config/v1_loader.cpp#L494) | Finite real | `0` | Earth-relative y velocity of the translating coordinate frame. It must be finite and zero for stationary or tracked_window mode; it changes the coordinate-frame motion rather than prescribing an atmospheric meridional wind. Units: m/s. |
| [`lifetime_planning.action`](../src/init/config/v1_loader.cpp#L498) | `warn`, `error` | `error` | Choose warn or error when the predicted storm envelope cannot remain inside the available domain. Enabled lifetime planning requires error in the runtime policy named qualified; that policy name does not mean the experiment is Fully Qualified. |
| [`lifetime_planning.cross_track_buffer_m`](../src/init/config/v1_loader.cpp#L507) | Real ≥ 0 | `0` | Cross-track allowance on either side of the predicted storm path. It must be finite and nonnegative, and strictly positive when lifetime planning is enabled. Units: m. |
| [`lifetime_planning.downstream_buffer_m`](../src/init/config/v1_loader.cpp#L506) | Real ≥ 0 | `0` | Spatial allowance ahead of the predicted storm along its motion direction. It must be finite and nonnegative, and strictly positive when lifetime planning is enabled. Units: m. |
| [`lifetime_planning.enabled`](../src/init/config/v1_loader.cpp#L497) | Boolean | `false` | Enable the preflight storm-lifetime envelope calculation using the run duration, resolved wind profile and selected motion estimate. All three spatial buffer widths must then be strictly positive; planning diagnoses predicted containment and does not itself move or enlarge the domain. |
| [`lifetime_planning.explicit_u_ms`](../src/init/config/v1_loader.cpp#L503) | Finite real | `0` | Finite Earth-relative x component of the planned storm motion when motion_source=explicit. This is the storm trajectory estimate, distinct from frame_velocity.u_ms. Units: m/s. |
| [`lifetime_planning.explicit_v_ms`](../src/init/config/v1_loader.cpp#L504) | Finite real | `0` | Finite Earth-relative y component of the planned storm motion when motion_source=explicit. This is the storm trajectory estimate, distinct from frame_velocity.v_ms. Units: m/s. |
| [`lifetime_planning.motion_source`](../src/init/config/v1_loader.cpp#L499) | `bunkers_right`, `bunkers_left`, `explicit` | `bunkers_right` | Choose the Earth-relative storm-motion estimate used by lifetime planning: bunkers_right, bunkers_left, or explicit. The Bunkers choices use the resolved environmental wind profile; explicit uses the two supplied velocity components. |
| [`lifetime_planning.motion_uncertainty_ms`](../src/init/config/v1_loader.cpp#L508) | Real ≥ 0 | `0` | Nonnegative speed uncertainty used to widen the predicted storm envelope as forecast lead time grows. It supplements the geometric buffers and does not add random perturbations to the simulated winds. Units: m/s. |
| [`lifetime_planning.upstream_buffer_m`](../src/init/config/v1_loader.cpp#L505) | Real ≥ 0 | `0` | Spatial allowance behind the predicted storm along its motion direction. It must be finite and nonnegative, and strictly positive when lifetime planning is enabled so the envelope has finite extent. Units: m. |
| [`mode`](../src/init/config/v1_loader.cpp#L489) | `stationary`, `constant_velocity`, `tracked_window`, `moving_nest` | `stationary` | Select stationary coordinates, a continuously translating frame (constant_velocity), whole-cell tracked-window recycling, or a moving child nest. Stationary and tracked_window require zero continuous frame velocity; tracked_window additionally requires tracking, explicit recycling authority and a positive shift limit. Public moving_nest and two-way combinations remain restricted to exact admitted profiles; selecting a mode does not establish general moving-nest or moist two-way readiness. |
| [`nesting.boundary_stencil_parent_cells`](../src/init/config/v1_loader.cpp#L551) | Integer > 0 | `1` | Positive parent-cell margin reserved for the child boundary/interpolation stencil. Initial placement and later relocation must preserve that coverage rather than extrapolating through a missing parent stencil. Units: parent cells. |
| [`nesting.child_cells_x`](../src/init/config/v1_loader.cpp#L547) | Integer ≥ 0 | `0` | Nonnegative child x dimension; it must be positive when nesting is enabled. It must span a whole number of parent cells and exceed twice the relaxation width together with the y dimension, leaving a nonempty interior. Units: child cells. |
| [`nesting.child_cells_y`](../src/init/config/v1_loader.cpp#L548) | Integer ≥ 0 | `0` | Nonnegative child y dimension; it must be positive when nesting is enabled. It must span a whole number of parent cells and exceed twice the relaxation width together with the x dimension, leaving a nonempty interior. Units: child cells. |
| [`nesting.child_origin_parent_cell_i`](../src/init/config/v1_loader.cpp#L543) | Integer ≥ 0 | `0` | Nonnegative initial x-index of the child origin on the parent grid. The full child footprint and required interpolation stencil must fit inside the available parent domain; moving-child operation can later update the origin through its admitted relocation owner. Units: parent cells. |
| [`nesting.child_origin_parent_cell_j`](../src/init/config/v1_loader.cpp#L545) | Integer ≥ 0 | `0` | Nonnegative initial y-index of the child origin on the parent grid. The full child footprint and required interpolation stencil must fit inside the available parent domain; this is an index, not an Earth-coordinate distance. Units: parent cells. |
| [`nesting.enabled`](../src/init/config/v1_loader.cpp#L542) | Boolean | `false` | Enable the refined child grid and its parent-to-child boundary coupling. Child dimensions and relaxation width must be positive and geometrically compatible, with valid parent stencil coverage; backend, physics and feedback admission still depend on the complete configuration. |
| [`nesting.feedback`](../src/init/config/v1_loader.cpp#L555) | `one_way`, `two_way` | `one_way` | Choose one_way parent forcing or two_way feedback with child-to-parent restriction and interface-flux correction. two_way is accepted only for the exact maintained stationary or moving/recycling combinations recognized by ordinary admission. General moist two-way coupling remains Engineering incomplete; a recognized spelling or a passing component test does not remove that limitation. |
| [`nesting.parent_ratio_x`](../src/init/config/v1_loader.cpp#L539) | Integer > 0 | `3` | Integer ratio of parent to child x spacing, at least 2. Enabled child_cells_x must be divisible by this ratio so the child spans whole parent cells; parent_time_step_ratio must be at least the largest horizontal ratio. Units: ratio. |
| [`nesting.parent_ratio_y`](../src/init/config/v1_loader.cpp#L540) | Integer > 0 | `3` | Integer ratio of parent to child y spacing, at least 2. Enabled child_cells_y must be divisible by this ratio so the child spans whole parent cells; parent_time_step_ratio must be at least the largest horizontal ratio. Units: ratio. |
| [`nesting.parent_time_step_ratio`](../src/init/config/v1_loader.cpp#L541) | Integer > 0 | `3` | Positive number of child time steps per parent time step. It must be at least max(parent_ratio_x,parent_ratio_y); choosing a larger integer alone does not bypass acoustic, cadence or profile-admission checks. Units: ratio. |
| [`nesting.relaxation_e_folding_time_s`](../src/init/config/v1_loader.cpp#L553) | Real > 0 | `2` | Strictly positive e-folding time for relaxing the child boundary zone toward parent-supplied values. It controls boundary coupling strength; it is distinct from the parent/child time-step ratio. Units: s. |
| [`nesting.relaxation_width_child_cells`](../src/init/config/v1_loader.cpp#L549) | Integer ≥ 0 | `0` | Width of each child boundary relaxation zone. It may be zero only while nesting is disabled; when enabled it must be positive and twice the width must be smaller than both child dimensions. Units: child cells. |
| [`tracking.branch_preference`](../src/init/config/v1_loader.cpp#L533) | `continuity`, `bunkers_right`, `bunkers_left` | `continuity` | Choose continuity, bunkers_right, or bunkers_left when associating physically admissible storm branches. Environmental right/left motion is derived from the resolved wind profile and is not inferred solely from updraft-helicity sign. |
| [`tracking.branch_velocity_uncertainty_ms`](../src/init/config/v1_loader.cpp#L537) | Real > 0 | `5` | Strictly positive uncertainty scale for comparing candidate branch velocities with environmental right/left-motion estimates. It belongs to branch association rather than lifetime-envelope widening. Units: m/s. |
| [`tracking.cooldown_s`](../src/init/config/v1_loader.cpp#L516) | Real ≥ 0 | `300` | Nonnegative cooldown used by the movement hysteresis after a committed shift while the storm remains inside the inner margin. It is not an unconditional minimum separation between all shifts: boundary threats and trusted-interior containment are evaluated separately. Units: s. |
| [`tracking.enabled`](../src/init/config/v1_loader.cpp#L511) | Boolean | `false` | Enable periodic storm-feature detection and continuity evidence. Enabled tracking requires a positive interval, positive inner margin and valid signal/continuity thresholds; tracking alone does not authorize prognostic recycling. The qualified execution policy admits tracking only in its maintained complete-profile combinations. |
| [`tracking.inner_margin_m`](../src/init/config/v1_loader.cpp#L514) | Real ≥ 0 | `0` | Desired clearance between the detected storm envelope and domain boundaries. It is nonnegative when inactive and must be positive with tracking; the movement policy derives a wider release margin from this value and the horizontal mesh spacing. Units: m. |
| [`tracking.interval_s`](../src/init/config/v1_loader.cpp#L513) | Real > 0 | `60` | Positive interval between storm-tracking samples. Tracking and optional moving_reference_v1 qualification diagnostics must use the same interval when that diagnostic preset is selected, with compatible fixed time-step cadence. Units: s. |
| [`tracking.lost_after_samples`](../src/init/config/v1_loader.cpp#L531) | Integer > 0 | `3` | Positive count of consecutive missing detections after which continuity becomes lost. Shorter gaps can retain a coasting track, which is distinct from a fresh confident detection. Units: samples. |
| [`tracking.maximum_motion_acceleration_ms2`](../src/init/config/v1_loader.cpp#L528) | Real ≥ 0 | `0.25` | Nonnegative maximum inferred change in storm translation velocity per unit time accepted by the continuity gate. This controls association between samples, not a physical acceleration limiter in the solver. Units: m/s^2. |
| [`tracking.maximum_motion_speed_ms`](../src/init/config/v1_loader.cpp#L527) | Real > 0 | `75` | Strictly positive maximum inferred translation speed accepted by the storm-continuity association gate. Exceeding it makes a proposed association untrustworthy rather than modifying the atmospheric wind. Units: m/s. |
| [`tracking.maximum_shift_cells`](../src/init/config/v1_loader.cpp#L515) | Integer ≥ 0 | `0` | Nonnegative integer bound on a proposed whole-cell translation in each horizontal direction. Zero leaves no live movement allowance; tracked_window or recycle_domain=true requires a positive value, and parent-stencil/containment checks can further limit a proposal. Units: cells. |
| [`tracking.minimum_absolute_updraft_helicity_m2s2`](../src/init/config/v1_loader.cpp#L523) | Real > 0 | `25` | Strictly positive absolute updraft-helicity threshold for rotating-core evidence. Both signs can pass this magnitude threshold; right/left-moving branch preference is configured separately. Units: m^2/s^2. |
| [`tracking.minimum_component_cells`](../src/init/config/v1_loader.cpp#L525) | Integer > 0 | `4` | Positive integer minimum horizontal connected-component size for storm candidates. It suppresses isolated small detections and is distinct from the physical core-area threshold. Units: cells. |
| [`tracking.minimum_condensate_path_kgm2`](../src/init/config/v1_loader.cpp#L519) | Real > 0 | `0.1` | Strictly positive condensate-column-path threshold used to identify a storm envelope. It is a vertically integrated water-mass diagnostic, not a cell mixing-ratio threshold. Units: kg/m^2. |
| [`tracking.minimum_confidence`](../src/init/config/v1_loader.cpp#L517) | Real ≥ 0 | `0.8` | Minimum confidence required for usable storm evidence. The loader accepts [0,1] while tracking is disabled and requires (0,1] while enabled; this is an internal evidence score, not a calibrated probability of forecast correctness. |
| [`tracking.minimum_continuity_samples`](../src/init/config/v1_loader.cpp#L530) | Integer > 0 | `3` | Number of samples required to establish storm continuity, constrained to integers from 2 through 5 by the fixed history capacity. A single isolated detection cannot satisfy this continuity requirement. Units: samples. |
| [`tracking.minimum_core_area_m2`](../src/init/config/v1_loader.cpp#L526) | Real > 0 | `1000000` | Strictly positive minimum physical area of a candidate storm core. It supplies a geometric size criterion in addition to minimum_component_cells, so a cell-count threshold alone does not define the same storm size at every resolution. Units: m^2. |
| [`tracking.minimum_updraft_ms`](../src/init/config/v1_loader.cpp#L518) | Real > 0 | `5` | Strictly positive updraft threshold used to identify candidate storm cores. This diagnostic threshold changes which features the tracker follows; it does not clip or limit the simulated vertical velocity. Units: m/s. |
| [`tracking.recycle_domain`](../src/init/config/v1_loader.cpp#L512) | Boolean | `false` | Explicitly authorize a supported tracking decision to recycle prognostic domain state. This requires tracking.enabled=true and a positive maximum_shift_cells; a nonzero movement limit alone cannot turn diagnostic tracking into state mutation. |
| [`tracking.topology_confirmation_samples`](../src/init/config/v1_loader.cpp#L532) | Integer > 0 | `2` | Positive number of consecutive topology observations required before confirming a split or merger. This delays a change of branch topology until repeated evidence supports it. Units: samples. |
| [`tracking.updraft_helicity_bottom_m_agl`](../src/init/config/v1_loader.cpp#L520) | Real ≥ 0 | `2000` | Nonnegative lower height of the updraft-helicity diagnostic layer. The upper bound must exceed it; terrain-aware geometric sampling uses height above ground rather than a fixed model index. Units: m above ground. |
| [`tracking.updraft_helicity_top_m_agl`](../src/init/config/v1_loader.cpp#L522) | Real > 0 | `5000` | Positive upper height of the updraft-helicity diagnostic layer, strictly above its lower bound. The diagnostic helps identify rotating cores; the sign of helicity does not replace the separate environmental branch-motion classification. Units: m above ground. |

## Dynamics

Select the resolved-motion implementation. The dynamics name and grid staggering must agree.

Prefix: `dynamics`. Each name below is relative to that prefix.

| Key | Type / choices | When omitted | Meaning and restrictions |
| --- | --- | --- | --- |
| [`axisymmetric`](../src/init/config/v1_loader.cpp#L747) | Boolean | `false` | Retained parsed/serialized compatibility flag, used in admission predicates. The current factory selects Cartesian A-grid/C-grid by scheme and staggering; this flag is not projected into a separate implemented axisymmetric solver. |
| [`scheme`](../src/init/config/v1_loader.cpp#L746) | String | `"klemp_wilhelmson_split"` | Dynamics factory ID. The default `klemp_wilhelmson_split` alias resolves using geometry/staggering; canonical Cartesian implementations are `klemp_wilhelmson_split_cartesian_agrid` and `klemp_wilhelmson_split_cartesian_cgrid`. The factory rejects unknown or incompatible selections. |

## Boundary conditions

Boundary names select closures, not decorative edge labels. Periodic pairing, terrain geometry, acoustic treatment and source budgets must remain compatible.

Prefix: `boundary_conditions`. Each name below is relative to that prefix.

| Key | Type / choices | When omitted | Meaning and restrictions |
| --- | --- | --- | --- |
| [`bottom`](../src/init/config/v1_loader.cpp#L814) | `rigid_noslip`, `rigid_slip` | `rigid_noslip` | Lower velocity closure: rigid no-slip or rigid free-slip. Surface exchange is configured separately. |
| [`lateral.east`](../src/init/config/v1_loader.cpp#L755) | `periodic`, `radiative`, `zero_gradient`, `open`, `rigid`, `rigid_slip` | `open` → grid-dependent closure | East lateral closure. `open` is a compatibility token resolving to `radiative` on C-grid and `zero_gradient` on collocated grids. Opposite faces must agree when periodic; radiative and zero-gradient closures are physically different. |
| [`lateral.north`](../src/init/config/v1_loader.cpp#L757) | `periodic`, `radiative`, `zero_gradient`, `open`, `rigid`, `rigid_slip` | `open` → grid-dependent closure | North lateral closure. `open` is a compatibility token resolving to `radiative` on C-grid and `zero_gradient` on collocated grids. Opposite faces must agree when periodic; radiative and zero-gradient closures are physically different. |
| [`lateral.south`](../src/init/config/v1_loader.cpp#L756) | `periodic`, `radiative`, `zero_gradient`, `open`, `rigid`, `rigid_slip` | `open` → grid-dependent closure | South lateral closure. `open` is a compatibility token resolving to `radiative` on C-grid and `zero_gradient` on collocated grids. Opposite faces must agree when periodic; radiative and zero-gradient closures are physically different. |
| [`lateral.west`](../src/init/config/v1_loader.cpp#L754) | `periodic`, `radiative`, `zero_gradient`, `open`, `rigid`, `rigid_slip` | `open` → grid-dependent closure | West lateral closure. `open` is a compatibility token resolving to `radiative` on C-grid and `zero_gradient` on collocated grids. Opposite faces must agree when periodic; radiative and zero-gradient closures are physically different. |
| [`top`](../src/init/config/v1_loader.cpp#L815) | `rigid`, `rigid_with_sponge`, `open` | `rigid_with_sponge` | Upper closure. `open` retains a non-prognostic lid w face and zero-gradient horizontal winds; it is not a fully transmitting radiative top. Sponge settings are separate. |

## Damping

Nimbus exposes the main upper sponge, a lateral band and an additional w-only lid absorber independently. A disabled main upper sponge does not by itself disable the other two.

Prefix: `damping`. Each name below is relative to that prefix.

| Key | Type / choices | When omitted | Meaning and restrictions |
| --- | --- | --- | --- |
| [`alpha`](../src/init/config/v1_loader.cpp#L823) | Real ≥ 0 | `0.05` | Main Rayleigh damping coefficient in s⁻¹; nonnegative, with zero removing its drag. |
| [`enabled`](../src/init/config/v1_loader.cpp#L821) | Boolean | `true` | Enable the main upper Rayleigh sponge. Lateral and w-only absorbers have their own switches. |
| [`lateral_alpha`](../src/init/config/v1_loader.cpp#L826) | Real ≥ 0 | `0.05` | Maximum lateral drag coefficient in s⁻¹; zero disables its drag. The band relaxes toward the environmental profile. |
| [`lateral_sponge_enabled`](../src/init/config/v1_loader.cpp#L827) | Boolean | `true` | Master switch for the lateral sponge, independent of its configured width/coefficient. |
| [`lateral_width`](../src/init/config/v1_loader.cpp#L825) | Integer ≥ 0 | `20` | Lateral sponge width in grid points per non-periodic face. Zero disables the band. |
| [`mode`](../src/init/config/v1_loader.cpp#L824) | `explicit`, `implicit` | `explicit` | Explicit or implicit damping application. |
| [`w_only_alpha_max`](../src/init/config/v1_loader.cpp#L830) | Real ≥ 0 | `0.2` | Peak w-only drag coefficient in s⁻¹, nonnegative. |
| [`w_only_depth_cells`](../src/init/config/v1_loader.cpp#L829) | Integer ≥ 0 | `8` | Depth of the additional upper absorber in cells; a nonnegative integer. Zero is accepted. |
| [`w_only_top_sponge`](../src/init/config/v1_loader.cpp#L828) | Boolean | `true` | Enable the additional w-only absorber next to the upper lid. |
| [`z_d`](../src/init/config/v1_loader.cpp#L822) | Real ≥ 0 | `12000` | Height in m where the main upper sponge begins. |

## Coriolis

Rotation settings are independent of radiation solar geometry.

Prefix: `coriolis`. Each name below is relative to that prefix.

| Key | Type / choices | When omitted | Meaning and restrictions |
| --- | --- | --- | --- |
| [`enabled`](../src/init/config/v1_loader.cpp#L836) | Boolean | `false` | Enable rotational tendencies. |
| [`latitude_deg`](../src/init/config/v1_loader.cpp#L838) | Finite real | `35` | Reference latitude in degrees for rotation. Radiation has a separate latitude control. |
| [`mode`](../src/init/config/v1_loader.cpp#L837) | `f_plane`, `beta_plane` | `f_plane` | `f_plane` uses the reference Coriolis parameter; `beta_plane` includes its meridional variation where the selected runtime admits it. |

## Microphysics

Each scheme parameter is active only for its declared scheme. The strict loader rejects explicitly supplied inactive microphysics controls. The guarded/obsolete Milbrandt keys remain listed below so their behavior is unambiguous.

Prefix: `microphysics`. Each name below is relative to that prefix.

| Key | Type / choices | When omitted | Meaning and restrictions |
| --- | --- | --- | --- |
| [`kessler.accretion_coefficient`](../src/init/config/v1_loader.cpp#L861) | Real ≥ 0 | `2.2` | Kessler empirical accretion coefficient (2.2 in the retained bulk law). This is a coefficient in that law, not a general first-order rate. Explicit use requires Kessler. |
| [`kessler.evaporation_efficiency`](../src/init/config/v1_loader.cpp#L863) | Real ≥ 0 | `1` | Dimensionless multiplier of Kessler evaporation. Explicit use requires Kessler. |
| [`lin.accretion_coefficient`](../src/init/config/v1_loader.cpp#L865) | Real ≥ 0 | `2.2` | Lin empirical accretion coefficient in the retained rain/cloud collection law; explicit use requires Lin. |
| [`lin.evaporation_rate_s_inv`](../src/init/config/v1_loader.cpp#L867) | Real ≥ 0 | `0.001` | Lin rain-evaporation rate parameter in s⁻¹; explicit use requires Lin. |
| [`milbrandt.evaporation_rate_s_inv`](../src/init/config/v1_loader.cpp#L897) | Real ≥ 0 | `0.001` | Obsolete fixed-rate control. **Explicit presence is rejected for every scheme.** Retained in this inventory so old input files can be understood; remove the key. |
| [`milbrandt.hail_processes`](../src/init/config/v1_loader.cpp#L902) | Boolean | `true` | Enable the admitted Milbrandt hail-process path. Explicit use requires Milbrandt; it does not select triple moments or complete their coupling. |
| [`milbrandt.triple_moment`](../src/init/config/v1_loader.cpp#L900) | Boolean | `false` internally; **omit key** | **Engineering incomplete: explicit presence is rejected, including `false`.** The retained typed default is false, but callers must omit this key. Internal third-moment owners do not yet establish ordinary runtime selection. |
| [`scheme`](../src/init/config/v1_loader.cpp#L852) | `none`, `kessler`, `lin`, `thompson`, `milbrandt` | `none` | Select no microphysics, Kessler, Lin, Thompson or Milbrandt. Thompson needs provisioned tables for a simulation. Milbrandt here means its admitted ordinary moment scope; selecting the name does not enable triple moments. |
| [`shared.autoconversion_rate_s_inv`](../src/init/config/v1_loader.cpp#L859) | Real ≥ 0 | `0.001` | Autoconversion rate in s⁻¹. Explicit use is restricted to Kessler, Lin or Thompson. |
| [`shared.cloud_water_threshold_kgkg`](../src/init/config/v1_loader.cpp#L857) | Real ≥ 0 | `0.001` | Cloud-water threshold for autoconversion, in kg water per kg dry air. This explicit key is active only with Kessler, Lin or Thompson. |
| [`thompson.evaporation_rate_s_inv`](../src/init/config/v1_loader.cpp#L869) | Real ≥ 0 | `0.001` | Thompson evaporation rate parameter in s⁻¹; explicit use requires Thompson. |
| [`thompson.ice_friendly_aerosol_surface_m3`](../src/init/config/v1_loader.cpp#L874) | Real > 0 | `2000000` | Surface ice-friendly aerosol number density in m⁻³. Explicit use requires Thompson and must exceed the fixed 500,000 m⁻³ background. |
| [`thompson.water_friendly_aerosol_surface_m3`](../src/init/config/v1_loader.cpp#L871) | Real > 0 | `350000000` | Surface water-friendly aerosol number density in m⁻³. Explicit use requires Thompson and must exceed the fixed 50,000,000 m⁻³ free-troposphere background. |

## Turbulence

These closures parameterize unresolved motion. Direction, filter width, stability treatment and dynamic/TKE choices are physical assumptions as well as implementation constraints.

Prefix: `turbulence`. Each name below is relative to that prefix.

| Key | Type / choices | When omitted | Meaning and restrictions |
| --- | --- | --- | --- |
| [`Cs`](../src/init/config/v1_loader.cpp#L1004) | Real ≥ 0 | `0.18` | Dimensionless static Smagorinsky coefficient; zero disables that coefficient's viscosity without changing the scheme name. |
| [`Pr_t`](../src/init/config/v1_loader.cpp#L1005) | Real > 0 | `0.333333333333333` | Positive turbulent Prandtl number controlling scalar heat diffusion relative to viscosity. |
| [`Sc_t`](../src/init/config/v1_loader.cpp#L1006) | Real > 0 | `0.333333333333333` | Positive turbulent Schmidt number controlling carried-scalar diffusion relative to viscosity. |
| [`dynamic_coefficient`](../src/init/config/v1_loader.cpp#L978) | Boolean | `false` | Enable dynamic Smagorinsky diagnosis. Requires Smagorinsky and no separate stability correction; retained geometry/backend restrictions also apply. |
| [`filter_width`](../src/init/config/v1_loader.cpp#L972) | `cubic_root`, `dx`, `user` | `cubic_root` | Unresolved length convention: cubic-root cell volume, horizontal dx, or a user-supplied physical width. This changes the closure, not just a numerical tuning constant. Aliases: `cubic-root` → `cubic_root`; `horizontal` → `dx`. |
| [`mode`](../src/init/config/v1_loader.cpp#L971) | `2d`, `vertical_only`, `3d` | `3d` | Directional scope: horizontal `2d`, `vertical_only` or full `3d`. Static vertical-only SGS still has a tracked incomplete qualification gate; exact-profile admission must be checked. |
| [`scheme`](../src/init/config/v1_loader.cpp#L970) | `none`, `smagorinsky`, `tke` | `smagorinsky` | Unresolved-turbulence closure: disabled, Smagorinsky or prognostic TKE. |
| [`stability_correction`](../src/init/config/v1_loader.cpp#L975) | `none`, `ri`, `tke_based` | `none` | No correction, Richardson correction (`ri`, Smagorinsky only), or TKE-based correction (`tke_based`, TKE only). Dynamic Smagorinsky cannot combine with a separate correction. Aliases: `richardson` → `ri`; `tke-based` → `tke_based`. |
| [`user_filter_width_m`](../src/init/config/v1_loader.cpp#L979) | Real ≥ 0 | `0` | Filter width in m. Must be positive with `filter_width: user`; otherwise it must remain zero. |

## Advection

This choice controls transported-scalar reconstruction, separately from the time integrator and pressure/acoustic scheme.

Prefix: `advection`. Each name below is relative to that prefix.

| Key | Type / choices | When omitted | Meaning and restrictions |
| --- | --- | --- | --- |
| [`scheme`](../src/init/config/v1_loader.cpp#L1045) | `auto`, `none`, `tvd`, `weno5` | `auto` | Scalar face reconstruction. `auto` resolves to WENO5 on C-grid and TVD on collocated grids. Disabled transport and each explicit reconstruction still require compatible scientific admission. Aliases: `off` → `none`; `disabled` → `none`; `mc` → `tvd`; `weno` → `weno5`. |

## Diffusion

Diffusion coefficients and the discretization applying them are separate selections. The linear-height branch has conditional required endpoints.

Prefix: `diffusion`. Each name below is relative to that prefix.

| Key | Type / choices | When omitted | Meaning and restrictions |
| --- | --- | --- | --- |
| [`K_h`](../src/init/config/v1_loader.cpp#L1072) | Real ≥ 0 | `0` | Horizontal diffusivity in m² s⁻¹, nonnegative. |
| [`K_v`](../src/init/config/v1_loader.cpp#L1073) | Real ≥ 0 | `0` | Constant vertical diffusivity in m² s⁻¹, nonnegative. Must be zero when `coefficient_source: linear_height` is selected. |
| [`K_v_surface`](../src/init/config/v1_loader.cpp#L1079) | Real ≥ 0 | `0` stored; **required for linear_height** | Lower endpoint diffusivity in m² s⁻¹. Required for `linear_height`; explicit use is rejected for the constant source. |
| [`K_v_top`](../src/init/config/v1_loader.cpp#L1081) | Real ≥ 0 | `0` stored; **required for linear_height** | Upper endpoint diffusivity in m² s⁻¹. Required for `linear_height`; explicit use is rejected for the constant source. |
| [`apply_to`](../src/init/config/v1_loader.cpp#L1074) | `scalars`, `scalars_and_momentum` | `scalars` | Apply diffusion to scalars alone or to scalars and momentum. Exact operator/backend support remains profile-dependent. |
| [`coefficient_source`](../src/init/config/v1_loader.cpp#L1075) | `constant`, `linear_height` | `constant` | Constant coefficients or a linearly height-varying vertical coefficient. `linear_height` requires implicit diffusion and explicit surface/top endpoint coefficients. |
| [`operator_order`](../src/init/config/v1_loader.cpp#L1063) | Integer > 0 | `2` | Spatial diffusion order. Only order 2 is implemented/admitted by this loader. |
| [`scheme`](../src/init/config/v1_loader.cpp#L1052) | `none`, `explicit`, `implicit` | `explicit` | Explicit or implicit diffusion. Although the enum parser recognizes `none`, the strict v1 section rejects it; disable constant diffusion by setting its coefficients to zero. |

## Surface

The surface supplies prescribed exchange and radiative properties. Nimbus does not thereby implement prognostic soil, vegetation or an independent land-surface energy model.

Prefix: `surface`. Each name below is relative to that prefix.

| Key | Type / choices | When omitted | Meaning and restrictions |
| --- | --- | --- | --- |
| [`albedo`](../src/init/config/v1_loader.cpp#L1121) | Finite real | `0.2` | Shortwave reflected fraction, dimensionless in [0, 1]. |
| [`apply_fluxes`](../src/init/config/v1_loader.cpp#L1118) | Boolean | `true` | Apply surface exchange through the boundary-layer owner. Must be false with `surface.scheme: none`. |
| [`emissivity`](../src/init/config/v1_loader.cpp#L1122) | Finite real | `0.95` | Longwave surface emissivity, dimensionless in [0, 1]. |
| [`qsfc_kgkg`](../src/init/config/v1_loader.cpp#L1127) | Finite real | `-1` | Prescribed surface vapor mixing ratio, kg kg⁻¹ dry air, within [0, 0.1], or `-1` to retain the initial surface-air value. |
| [`scheme`](../src/init/config/v1_loader.cpp#L1117) | `none`, `slab` | `slab` | No surface exchange or the prescribed slab surface. Slab requires an active boundary-layer owner; there is no independent prognostic soil/land-surface model. |
| [`tsk_k`](../src/init/config/v1_loader.cpp#L1126) | Finite real | `-1` | Prescribed skin temperature in K, within [150, 400], or `-1` to retain the initial surface-air temperature as the prescribed skin value. |
| [`z0h_m`](../src/init/config/v1_loader.cpp#L1120) | Real > 0 | `0.01` | Positive thermal/moisture roughness length in m. |
| [`z0m_m`](../src/init/config/v1_loader.cpp#L1119) | Real > 0 | `0.1` | Positive momentum roughness length in m. |

## Boundary layer

PBL columns own surface exchange and interval-mean tendencies. Their compatibility with resolved SGS and the thermal path must be checked for the whole selected profile.

Prefix: `boundary_layer`. Each name below is relative to that prefix.

| Key | Type / choices | When omitted | Meaning and restrictions |
| --- | --- | --- | --- |
| [`cadence_s`](../src/init/config/v1_loader.cpp#L1157) | Finite real | `60` | PBL refresh interval in s, finite and greater than the 1e-12 s scheduling tolerance. Sources are carried between refreshes according to the column owner. |
| [`clear_sky_only`](../src/init/config/v1_loader.cpp#L1159) | Boolean | `true` | Restrict the column treatment to its clear-sky formulation. False is accepted only with YSU, with additional coupled-profile constraints. |
| [`pbl_top_method`](../src/init/config/v1_loader.cpp#L1152) | `ri`, `theta`, `tke` | `ri` | Diagnose PBL top from Richardson number (`ri`), potential temperature (`theta`) or TKE. The TKE choice requires MYNN; exact alternatives retain profile-specific gates. |
| [`scheme`](../src/init/config/v1_loader.cpp#L1150) | `none`, `slab`, `ysu`, `mynn` | `slab` | Column mixing closure: disabled, slab, YSU or MYNN. Surface and PBL selections must be consistent; neither silently infers the other. |

## Radiation

Radiation is an explicitly configured source with prescribed solar geometry, a selected optical closure and a per-level update cadence. Solar geometry is required for every active radiation configuration, longwave requires an explicit skin temperature, and RRTMG requires explicit ozone and external coefficient tables; the tables are unnecessary just to build or open Studio. Simple-grey pressure optics, simple-grey absorbing water/cloud optics and RRTMG gas/cloud optics have different assumptions. Native column execution does not imply that every terrain, shortwave, nested or cloud combination is implemented; in particular mapped RRTMG shortwave accelerator ownership remains guarded. The runtime policy word qualified and historical identifiers are not claims of Fully Qualified status.

Prefix: `radiation`. Each name below is relative to that prefix.

| Key | Type / choices | When omitted | Meaning and restrictions |
| --- | --- | --- | --- |
| [`cadence_s`](../src/init/config/v1_loader.cpp#L1224) | Real > 0 | `300` | Nominal interval between radiation updates. It must be positive and exceed the 1e-12 s scheduling tolerance; when simple-grey subcycling is enabled the effective update interval is cadence_s/subcycle_count and must satisfy the same tolerance. Between updates, the runtime retains the last completed radiative power/tendency state. Units: s. |
| [`clear_sky_only`](../src/init/config/v1_loader.cpp#L1227) | Boolean | `true` | Control whether condensate contributes to optical properties. For simple_grey it must be true with pressure_clear_sky_v1 and false with absorbing_water_cloud_path_v1; cloudy operation requires matching vapor, cloud-liquid and cloud-ice fields. RRTMG uses its own gas and bulk-cloud optical mapping, so this flag does not choose a scattering model by itself. |
| [`declination_deg`](../src/init/config/v1_loader.cpp#L1220) | Finite real | `15` | Fixed solar declination used by the idealized diurnal cycle, constrained to [-23.5,23.5]. It must be supplied explicitly for active radiation; the solver does not infer an annual cycle from a calendar date. Units: degrees. |
| [`do_lw`](../src/init/config/v1_loader.cpp#L1222) | Boolean | `true` | Enable longwave absorption and emission in the selected radiation scheme. Active longwave requires surface.tsk_k to be explicitly written in [150,400] K; an inherited surface-air sentinel is not accepted for this forcing. |
| [`do_sw`](../src/init/config/v1_loader.cpp#L1223) | Boolean | `true` | Enable shortwave solar transfer in the selected radiation scheme. Active radiation cannot disable both shortwave and longwave; terrain shortwave uses a separate geometric transport owner, and mapped RRTMG shortwave lacks an admitted native accelerator owner. |
| [`enable_diagnostics`](../src/init/config/v1_loader.cpp#L1235) | Boolean | `false` | Retain radiation interface-flux diagnostics in the per-level runtime for inspection. Explicit use is accepted only for simple_grey; mapped shortwave rejects this option because retained interface diagnostics are not implemented for that geometric transport path. |
| [`latitude_deg`](../src/init/config/v1_loader.cpp#L1219) | Finite real | `35` | Geographic latitude used in the idealized solar-zenith calculation, constrained to [-90,90]. It must be supplied explicitly for active radiation and is independent of coriolis.latitude_deg. Units: degrees. |
| [`local_solar_time_at_start_h`](../src/init/config/v1_loader.cpp#L1221) | Finite real | `0` | Local solar time at model start, constrained to [0,24). It must be explicit for active radiation; elapsed model time advances the diurnal forcing from this starting phase. Units: h. |
| [`n_lw`](../src/init/config/v1_loader.cpp#L1239) | Real > 0 | `4` | Strictly positive pressure exponent shaping the simple-grey longwave optical-depth profile. Increasing it concentrates a given reference optical depth more strongly toward higher pressure; it is not a timestep or vertical-grid parameter. |
| [`n_sw`](../src/init/config/v1_loader.cpp#L1240) | Real > 0 | `2` | Strictly positive pressure exponent shaping the simple-grey shortwave optical-depth profile. It redistributes the grey absorption with pressure and does not select the angular quadrature used by mapped terrain transport. |
| [`rrtmg_ccl4_vmr`](../src/init/config/v1_loader.cpp#L1269) | Real ≥ 0 | `0` | Carbon tetrachloride dry-air mole fraction supplied to the RRTMG column optics, constrained to [0,1e-4]. Explicit use is rejected unless scheme=rrtmg; enter a fraction rather than a numerical ppm/ppb value. Units: mol/mol dry air. |
| [`rrtmg_cfc11_vmr`](../src/init/config/v1_loader.cpp#L1266) | Real ≥ 0 | `0` | CFC-11 dry-air mole fraction supplied to the RRTMG column optics, constrained to [0,1e-4]. Explicit use is rejected unless scheme=rrtmg; enter a fraction rather than a numerical ppm/ppb value. Units: mol/mol dry air. |
| [`rrtmg_cfc12_vmr`](../src/init/config/v1_loader.cpp#L1267) | Real ≥ 0 | `0` | CFC-12 dry-air mole fraction supplied to the RRTMG column optics, constrained to [0,1e-4]. Explicit use is rejected unless scheme=rrtmg; enter a fraction rather than a numerical ppm/ppb value. Units: mol/mol dry air. |
| [`rrtmg_cfc22_vmr`](../src/init/config/v1_loader.cpp#L1268) | Real ≥ 0 | `0` | CFC-22 dry-air mole fraction supplied to the RRTMG column optics, constrained to [0,1e-4]. Explicit use is rejected unless scheme=rrtmg; enter a fraction rather than a numerical ppm/ppb value. Units: mol/mol dry air. |
| [`rrtmg_ch4_vmr`](../src/init/config/v1_loader.cpp#L1261) | Real ≥ 0 | `1.9e-06` | Methane dry-air mole fraction supplied to the RRTMG column optics, constrained to [0,1e-3]. Explicit use is rejected unless scheme=rrtmg; enter a fraction rather than a numerical ppm/ppb value. Units: mol/mol dry air. |
| [`rrtmg_co2_vmr`](../src/init/config/v1_loader.cpp#L1260) | Real ≥ 0 | `0.00042` | Carbon dioxide dry-air mole fraction supplied to the RRTMG column optics, constrained to [0,1e-2]. Explicit use is rejected unless scheme=rrtmg; enter a fraction rather than a numerical ppm/ppb value. Units: mol/mol dry air. |
| [`rrtmg_co_vmr`](../src/init/config/v1_loader.cpp#L1263) | Real ≥ 0 | `0` | Carbon monoxide dry-air mole fraction supplied to the RRTMG column optics, constrained to [0,1e-3]. Explicit use is rejected unless scheme=rrtmg; enter a fraction rather than a numerical ppm/ppb value. Units: mol/mol dry air. |
| [`rrtmg_ice_effective_radius_um`](../src/init/config/v1_loader.cpp#L1273) | Real > 0 | `30` | Bulk cloud-ice effective radius supplied to RRTMG optical lookup calculations, constrained to [13,130]. Explicit use requires scheme=rrtmg; it does not expose a general ice-habit or scattering-model selector. Units: micrometres. |
| [`rrtmg_liquid_effective_radius_um`](../src/init/config/v1_loader.cpp#L1271) | Real > 0 | `10` | Bulk cloud-liquid effective radius supplied to RRTMG optical lookup calculations, constrained to [2.5,60]. Explicit use requires scheme=rrtmg; it is a prescribed optical parameter, not an independently prognosed droplet spectrum. Units: micrometres. |
| [`rrtmg_n2_vmr`](../src/init/config/v1_loader.cpp#L1265) | Real ≥ 0 | `0.7808` | Molecular nitrogen dry-air mole fraction supplied to the RRTMG column optics, constrained to [0,1]. Explicit use is rejected unless scheme=rrtmg; enter a fraction rather than a numerical ppm/ppb value. Units: mol/mol dry air. |
| [`rrtmg_n2o_vmr`](../src/init/config/v1_loader.cpp#L1262) | Real ≥ 0 | `3.35e-07` | Nitrous oxide dry-air mole fraction supplied to the RRTMG column optics, constrained to [0,1e-3]. Explicit use is rejected unless scheme=rrtmg; enter a fraction rather than a numerical ppm/ppb value. Units: mol/mol dry air. |
| [`rrtmg_o2_vmr`](../src/init/config/v1_loader.cpp#L1264) | Real ≥ 0 | `0.2095` | Molecular oxygen dry-air mole fraction supplied to the RRTMG column optics, constrained to [0,1]. Explicit use is rejected unless scheme=rrtmg; enter a fraction rather than a numerical ppm/ppb value. Units: mol/mol dry air. |
| [`rrtmg_ozone_vmr`](../src/init/config/v1_loader.cpp#L1259) | Real ≥ 0 | `0`; **explicit value required for RRTMG** | Ozone dry-air mole fraction supplied to the RRTMG column optics, constrained to [0,1e-2]. Explicit use is rejected unless scheme=rrtmg; enter a fraction rather than a numerical ppm/ppb value. It must be explicitly supplied when RRTMG is active; there is no hidden ozone climatology. Units: mol/mol dry air. |
| [`scheme`](../src/init/config/v1_loader.cpp#L1216) | `none`, `simple_grey`, `rrtmg` | `none` | Choose none, simple_grey, or rrtmg. Active radiation requires explicit solar geometry, at least one of do_lw/do_sw, and an explicit valid skin temperature for longwave; RRTMG additionally requires coefficient files and explicit ozone. Complete backend/terrain/physics combinations remain subject to ordinary admission, including the guarded mapped RRTMG shortwave accelerator path. |
| [`simple_grey_lw_ice_absorption_m2kg`](../src/init/config/v1_loader.cpp#L1247) | Real ≥ 0 | `45` | Nonnegative mass-absorption coefficient multiplying the cloud ice path in the simple-grey longwave absorbing-water/cloud optical model. Explicit use requires scheme=simple_grey; its material-path contribution is active with absorbing_water_cloud_path_v1, not the pressure-only clear-sky model. Units: m^2/kg. |
| [`simple_grey_lw_liquid_absorption_m2kg`](../src/init/config/v1_loader.cpp#L1244) | Real ≥ 0 | `85` | Nonnegative mass-absorption coefficient multiplying the cloud liquid path in the simple-grey longwave absorbing-water/cloud optical model. Explicit use requires scheme=simple_grey; its material-path contribution is active with absorbing_water_cloud_path_v1, not the pressure-only clear-sky model. Units: m^2/kg. |
| [`simple_grey_lw_vapor_absorption_m2kg`](../src/init/config/v1_loader.cpp#L1241) | Real ≥ 0 | `0.02` | Nonnegative mass-absorption coefficient multiplying the water vapor path in the simple-grey longwave absorbing-water/cloud optical model. Explicit use requires scheme=simple_grey; its material-path contribution is active with absorbing_water_cloud_path_v1, not the pressure-only clear-sky model. Units: m^2/kg. |
| [`simple_grey_optical_model`](../src/init/config/v1_loader.cpp#L1228) | String | `"pressure_clear_sky_v1"` | Choose pressure_clear_sky_v1 or absorbing_water_cloud_path_v1 for simple-grey optical depth. The first requires clear_sky_only=true; the second requires false and adds state-dependent vapor/cloud path absorption without a general scattering closure. Explicit use is rejected unless scheme=simple_grey, and strict execution still checks the full supported profile. |
| [`simple_grey_sw_ice_absorption_m2kg`](../src/init/config/v1_loader.cpp#L1256) | Real ≥ 0 | `12` | Nonnegative mass-absorption coefficient multiplying the cloud ice path in the simple-grey shortwave absorbing-water/cloud optical model. Explicit use requires scheme=simple_grey; its material-path contribution is active with absorbing_water_cloud_path_v1, not the pressure-only clear-sky model. Units: m^2/kg. |
| [`simple_grey_sw_liquid_absorption_m2kg`](../src/init/config/v1_loader.cpp#L1253) | Real ≥ 0 | `20` | Nonnegative mass-absorption coefficient multiplying the cloud liquid path in the simple-grey shortwave absorbing-water/cloud optical model. Explicit use requires scheme=simple_grey; its material-path contribution is active with absorbing_water_cloud_path_v1, not the pressure-only clear-sky model. Units: m^2/kg. |
| [`simple_grey_sw_vapor_absorption_m2kg`](../src/init/config/v1_loader.cpp#L1250) | Real ≥ 0 | `0.001` | Nonnegative mass-absorption coefficient multiplying the water vapor path in the simple-grey shortwave absorbing-water/cloud optical model. Explicit use requires scheme=simple_grey; its material-path contribution is active with absorbing_water_cloud_path_v1, not the pressure-only clear-sky model. Units: m^2/kg. |
| [`solar_forcing`](../src/init/config/v1_loader.cpp#L1218) | `idealized_diurnal` | `idealized_diurnal` | Select the implemented idealized_diurnal solar-forcing model. Every active radiation configuration must state this key explicitly, even if only longwave is enabled; the geometry uses prescribed latitude, fixed declination and evolving local solar time rather than a date/location ephemeris. |
| [`subcycle_count`](../src/init/config/v1_loader.cpp#L1233) | Integer > 0 | `1` | Number of simple-grey update intervals per nominal radiation cadence. It must be 1 with use_subcycling=false or an integer in [2,64] with true; explicit use is rejected outside simple_grey, and the resulting interval must remain above the scheduling tolerance. |
| [`tau_lw_ref`](../src/init/config/v1_loader.cpp#L1237) | Real ≥ 0 | `6` | Nonnegative reference optical depth of the simple-grey pressure-dependent longwave profile. It controls broadband attenuation together with n_lw and does not replace RRTMG gas coefficients; the cloudy absorbing model retains this pressure contribution and adds material-path absorption. |
| [`tau_sw_ref`](../src/init/config/v1_loader.cpp#L1238) | Real ≥ 0 | `0.22` | Nonnegative reference optical depth of the simple-grey pressure-dependent shortwave profile. It controls broadband attenuation together with n_sw; this is a grey absorption parameter rather than an aerosol scattering or RRTMG spectral coefficient. |
| [`terrain_sw_absolute_residual_tolerance_w`](../src/init/config/v1_loader.cpp#L1279) | Real ≥ 0 | `1e-09` | Nonnegative absolute unresolved-power tolerance for mapped shortwave reflection iteration. This is total power in watts, not flux in W/m^2 or a temperature tolerance; it is used alongside the relative residual criterion. Units: W. |
| [`terrain_sw_azimuth_nodes`](../src/init/config/v1_loader.cpp#L1275) | Integer > 0 | `8` | Azimuthal quadrature resolution used by mapped shortwave transport. It must be a multiple of four, from 4 through 128 for RRTMG or 8 through 128 for other scheme selections. Units: nodes. |
| [`terrain_sw_cosine_nodes`](../src/init/config/v1_loader.cpp#L1274) | Integer > 0 | `4` | Polar-direction quadrature resolution used by mapped shortwave transport. The integer range is [1,32] for RRTMG and [4,32] for other scheme selections; a larger value increases angular work but does not establish accuracy or native-backend support. Units: nodes. |
| [`terrain_sw_maximum_bounces`](../src/init/config/v1_loader.cpp#L1276) | Integer > 0 | `32` | Maximum number of iterative terrain-reflection passes in mapped shortwave transport, constrained to integers in [1,256]. This bounds work; the unresolved reflected power and residual tolerances determine whether the solve has converged. Units: bounces. |
| [`terrain_sw_relative_residual_tolerance`](../src/init/config/v1_loader.cpp#L1277) | Real ≥ 0 | `1e-10` | Relative unresolved-power tolerance used to stop mapped shortwave reflection iteration. It must be finite, nonnegative and no larger than 1e-6; it is a solver convergence budget, not a statement of physical radiative accuracy. |
| [`use_subcycling`](../src/init/config/v1_loader.cpp#L1231) | Boolean | `false` | Enable more frequent simple-grey updates by dividing cadence_s by subcycle_count. Explicit use is rejected for other schemes; disabled subcycling requires count=1, while enabled subcycling requires an integer count in [2,64]. |

## Terrain

Analytic terrain changes physical heights, volumes, derivatives and metric terms. Active terrain requires compatible coordinates and a consistent computational/physical domain top.

Prefix: `terrain`. Each name below is relative to that prefix.

| Key | Type / choices | When omitted | Meaning and restrictions |
| --- | --- | --- | --- |
| [`bell.a`](../src/init/config/v1_loader.cpp#L1462) | Real > 0 | `5000` | Positive bell horizontal scale in m. |
| [`bell.axisymmetric`](../src/init/config/v1_loader.cpp#L1463) | Boolean | `false` | Use the axisymmetric bell shape rather than the x-only ridge. This controls topography, separately from `dynamics.axisymmetric`. |
| [`bell.h0`](../src/init/config/v1_loader.cpp#L1461) | Real ≥ 0 | `1000` | Bell peak height in m; nonnegative and below `grid.ztop` for active bell terrain. |
| [`compute_derivatives`](../src/init/config/v1_loader.cpp#L1454) | Boolean | `true` | Compute topographic derivatives. Must be true for active terrain. |
| [`compute_metrics`](../src/init/config/v1_loader.cpp#L1455) | Boolean | `true` | Compute coordinate metrics. Must be true for active terrain. |
| [`coordinate`](../src/init/config/v1_loader.cpp#L1449) | `btf`, `smoothed`, `cartesian` | `btf` | Basic terrain following (`btf`), smoothed terrain following, or Cartesian. Active terrain rejects `cartesian` because it would ignore topography. Aliases: `terrain_following` → `btf`; `terrain-following` → `btf`; `sigma` → `btf`; `smooth` → `smoothed`; `flat` → `cartesian`; `none` → `cartesian`. |
| [`feature_center_earth_x_m`](../src/init/config/v1_loader.cpp#L1457) | Finite real | `0` stored; **required for earth_fixed** | Earth-frame x coordinate of the analytic feature center in m. Explicitly required for `earth_fixed`; explicit presence is rejected for `grid_attached`. |
| [`feature_center_earth_y_m`](../src/init/config/v1_loader.cpp#L1459) | Finite real | `0` stored; **required for earth_fixed** | Earth-frame y coordinate of the analytic feature center in m. Explicitly required for `earth_fixed`; explicit presence is rejected for `grid_attached`. |
| [`horizontal_placement`](../src/init/config/v1_loader.cpp#L1450) | `grid_attached`, `earth_fixed` | `grid_attached` | Feature position tied to the grid or fixed in the Earth frame. Earth-fixed terrain and moving coordinates require their matching geometry owners. |
| [`schar.H`](../src/init/config/v1_loader.cpp#L1464) | Real ≥ 0 | `1000` | Schär peak-height scale in m; nonnegative and below `grid.ztop` for active Schär terrain. |
| [`schar.a`](../src/init/config/v1_loader.cpp#L1465) | Real > 0 | `5000` | Positive Schär Gaussian-envelope scale in m. |
| [`schar.ell`](../src/init/config/v1_loader.cpp#L1466) | Real > 0 | `4000` | Positive Schär cosine-wave scale in m. |
| [`scheme`](../src/init/config/v1_loader.cpp#L1448) | `none`, `bell`, `schar` | `none` | Disabled terrain, an analytic bell or a Schär wave. Active terrain changes physical geometry and admission. Aliases: `flat` → `none`; `schaer` → `schar`. |
| [`smoothing_decay`](../src/init/config/v1_loader.cpp#L1456) | Real ≥ 0 | `0.1` | Nonnegative decay control for the smoothed coordinate; zero is accepted. Its meaning follows the vertical mapping, not an independent physical viscosity. |

## Initial conditions

Choose thermodynamics, wind and a trigger separately. A target value is not a guarantee of an independently controllable realized diagnostic; source-specific limits and precedence are part of the experiment.

Prefix: `initial_conditions`. Each name below is relative to that prefix.

| Key | Type / choices | When omitted | Meaning and restrictions |
| --- | --- | --- | --- |
| [`hodograph.type`](../src/init/config/v1_loader.cpp#L1711) | `auto`, `custom`, `wk_param` | `auto` | Wind source. `auto` preserves winds supplied by a file sounding, otherwise uses the configured anchors; `custom` forces the anchors, and `wk_param` selects the retained parameterized layout. Wind and thermodynamic source selection are separate. |
| [`hodograph.u_1km_ms`](../src/init/config/v1_loader.cpp#L1714) | Finite real | `0` | u-wind anchor at 1 km, in m s⁻¹. Used when the chosen hodograph branch consumes the configured anchors; file-wind preservation can take precedence. |
| [`hodograph.u_6km_ms`](../src/init/config/v1_loader.cpp#L1716) | Finite real | `0` | u-wind anchor at 6 km, in m s⁻¹. Used when the chosen hodograph branch consumes the configured anchors; file-wind preservation can take precedence. |
| [`hodograph.u_sfc_ms`](../src/init/config/v1_loader.cpp#L1712) | Finite real | `0` | u-wind anchor at surface, in m s⁻¹. Used when the chosen hodograph branch consumes the configured anchors; file-wind preservation can take precedence. |
| [`hodograph.v_1km_ms`](../src/init/config/v1_loader.cpp#L1715) | Finite real | `0` | v-wind anchor at 1 km, in m s⁻¹. Used when the chosen hodograph branch consumes the configured anchors; file-wind preservation can take precedence. |
| [`hodograph.v_6km_ms`](../src/init/config/v1_loader.cpp#L1717) | Finite real | `0` | v-wind anchor at 6 km, in m s⁻¹. Used when the chosen hodograph branch consumes the configured anchors; file-wind preservation can take precedence. |
| [`hodograph.v_sfc_ms`](../src/init/config/v1_loader.cpp#L1713) | Finite real | `0` | v-wind anchor at surface, in m s⁻¹. Used when the chosen hodograph branch consumes the configured anchors; file-wind preservation can take precedence. |
| [`sounding.cape_jkg`](../src/init/config/v1_loader.cpp#L1559) | Real ≥ 0 | `2500` | Requested CAPE in J kg⁻¹ for either parametric source; not accepted with file input. The procedural inverse search has bounded accuracy and iteration limits. |
| [`sounding.cin_jkg`](../src/init/config/v1_loader.cpp#L1560) | Real ≥ 0 | `50` | Requested nonnegative CIN magnitude in J kg⁻¹; accepted only for `parametric_targets`. |
| [`sounding.el_m`](../src/init/config/v1_loader.cpp#L1563) | Real ≥ 0 | `12000` | Requested equilibrium-level height in m, only for `parametric_targets`; must be at least 8,000 m and above the lower target heights. |
| [`sounding.file_path`](../src/init/config/v1_loader.cpp#L1573) | String | Empty; **required for file source** | Sounding path, required and nonempty for `type: file`; inactive parametric controls must be removed. Keep relative paths consistent with the launch directory. Reader format detection is automatic; `file.scheme_id` is not an accepted v1 YAML key. |
| [`sounding.lcl_m`](../src/init/config/v1_loader.cpp#L1561) | Real ≥ 0 | `1200` | Requested LCL height in m, only for `parametric_targets`. Cannot be explicitly combined with `moisture_qv_kgkg`; the source normally derives moisture from this target. |
| [`sounding.lfc_m`](../src/init/config/v1_loader.cpp#L1562) | Real ≥ 0 | `2000` | Requested LFC height in m, only for `parametric_targets`; must be at least 50 m and consistent with the other target heights. |
| [`sounding.moisture_qv_kgkg`](../src/init/config/v1_loader.cpp#L1568) | Real ≥ 0 | `0.014` (CAPE); derived (targets) | Optional parametric surface vapor mixing ratio in kg kg⁻¹ dry air, restricted to [0.004, 0.024]. CAPE mode uses 0.014 when omitted; target mode derives it from LCL. File input rejects it, and target mode rejects simultaneous explicit LCL and moisture. |
| [`sounding.surface_theta_k`](../src/init/config/v1_loader.cpp#L1565) | Real > 0 | `300` | Parametric surface potential temperature in K, at least 250 K; not accepted with file input. |
| [`sounding.tropopause_m`](../src/init/config/v1_loader.cpp#L1564) | Real ≥ 0 | `12000` | Procedural tropopause height in m, at least 8,000 m; accepted only for `parametric_cape`. It is not an alias for an independently realized EL. |
| [`sounding.type`](../src/init/config/v1_loader.cpp#L1545) | `parametric_cape`, `parametric_targets`, `file` | `parametric_cape` | Environmental thermodynamic source: procedural CAPE profile, diagnostic-target profile, or file. Each branch has its own permitted keys; targets describe an idealized family, not guaranteed exact realized diagnostics. |
| [`trigger.bubble.center_x_m`](../src/init/config/v1_loader.cpp#L1727) | Finite real | `0` | Bubble center x coordinate in m. |
| [`trigger.bubble.center_y_m`](../src/init/config/v1_loader.cpp#L1728) | Finite real | `0` | Bubble center y coordinate in m. |
| [`trigger.bubble.center_z_m`](../src/init/config/v1_loader.cpp#L1729) | Finite real | `1500` | Bubble center height in m; the physical-height convention matters on terrain. |
| [`trigger.bubble.cloud_water_kgkg`](../src/init/config/v1_loader.cpp#L1723) | Real ≥ 0 | `0` | Additional peak cloud-water seed in kg kg⁻¹ dry air, nonnegative. This adds initial condensate independently; it is not deducted from vapor. |
| [`trigger.bubble.dtheta_k`](../src/init/config/v1_loader.cpp#L1722) | Finite real | `2` | Peak potential-temperature perturbation in K. The source can also adjust vapor when its saturation-preservation branch applies; see the mathematics guide. |
| [`trigger.bubble.radius_km`](../src/init/config/v1_loader.cpp#L1726) | Real > 0 | `10` | Positive bubble radius in km. The retained helper has a 100 m lower radius bound and a truncated Gaussian shape. |
| [`trigger.type`](../src/init/config/v1_loader.cpp#L1720) | `none`, `bubble`, `vortex_seed` | `none` | No imposed trigger, Gaussian warm bubble or Rankine vortex seed. A trigger changes initial inventories and need not be a conservative phase transfer. |
| [`trigger.vortex_seed.center_x_m`](../src/init/config/v1_loader.cpp#L1734) | Finite real | `0` | Horizontal vortex-center x coordinate in m. |
| [`trigger.vortex_seed.center_y_m`](../src/init/config/v1_loader.cpp#L1735) | Finite real | `0` | Horizontal vortex-center y coordinate in m. |
| [`trigger.vortex_seed.center_z_m`](../src/init/config/v1_loader.cpp#L1736) | Finite real | `3000` | **Engineering incomplete behavior:** parsed and serialized, but the current seed application does not use this value to shift the vortex vertically. Do not interpret it as implemented vertical placement. |
| [`trigger.vortex_seed.r_max_m`](../src/init/config/v1_loader.cpp#L1731) | Real > 0 | `6000` | Positive radius of maximum tangential wind in m for the Rankine seed. |
| [`trigger.vortex_seed.v_max_ms`](../src/init/config/v1_loader.cpp#L1732) | Finite real | `40` | Finite signed peak tangential wind amplitude in m s⁻¹. A negative value reverses the circulation; the radius and upper cutoff remain positive. |
| [`trigger.vortex_seed.z_top_m`](../src/init/config/v1_loader.cpp#L1733) | Real > 0 | `6000` | Positive hard upper height cutoff in m for the seed. |

## Radar

Synthetic radar products are forward operators derived from a supported hydrometeor state. Their names do not promise full instrument physics or observational validation.

Prefix: `radar`. Each name below is relative to that prefix.

| Key | Type / choices | When omitted | Meaning and restrictions |
| --- | --- | --- | --- |
| [`apply_scatterer_correction`](../src/init/config/v1_loader.cpp#L1947) | Boolean | `true` | Apply the retained hydrometeor/scatterer velocity correction when forming radar velocity. |
| [`enabled`](../src/init/config/v1_loader.cpp#L1936) | Boolean | `false` | Enable synthetic radar products. Requires a nonempty product list; disabled radar rejects a nonempty list. |
| [`location.x_m`](../src/init/config/v1_loader.cpp#L1944) | Finite real | `0` | Radar location x coordinate in m, used by the viewing/beam geometry. |
| [`location.y_m`](../src/init/config/v1_loader.cpp#L1945) | Finite real | `0` | Radar location y coordinate in m. |
| [`location.z_m`](../src/init/config/v1_loader.cpp#L1946) | Finite real | `0` | Radar height in m. |
| [`products`](../src/init/config/v1_loader.cpp#L1938) | Inline list | `[]` | Inline list of distinct products: `reflectivity_dbz`, `radial_velocity`, `zdr`, `kdp`, `rhohv`. KDP and rhohv require `zdr_tier: zhang2021_rain_mp`; microphysics/output gates also apply. |
| [`reflectivity_tier`](../src/init/config/v1_loader.cpp#L1941) | String | `"fast_da"` | `fast_da` or `psd_moment`. These are bulk forward-operator approximations; the latter is not a general full-spectrum scattering solver. |
| [`velocity_sampling`](../src/init/config/v1_loader.cpp#L1942) | String | `"point"` | `point` or `beam_volume`; the retained beam-volume path uses a neighborhood aggregation, not an unrestricted antenna/ray-tracing model. |
| [`zdr_tier`](../src/init/config/v1_loader.cpp#L1943) | String | `"polarimetric_fo"` | `simple`, `polarimetric_fo` or `zhang2021_rain_mp`. Product, hydrometeor and exact-profile restrictions remain in force. |

## Output

Choose durable history independently from shared-memory previews, sparse diagnostics and parcel trajectories. Compression, temporal encoding, retention and field selection all affect the publication contract. Do not remove NPY while its replacement qualification remains unfinished.

Prefix: `output`. Each name below is relative to that prefix.

| Key | Type / choices | When omitted | Meaning and restrictions |
| --- | --- | --- | --- |
| [`async_io`](../src/init/config/v1_loader.cpp#L1756) | Boolean | `true` | Use asynchronous history writing. Publication and shutdown still have to complete outstanding writes; this does not change scientific cadence. |
| [`format`](../src/init/config/v1_loader.cpp#L1746) | `npy_3d`, `netcdf`, `zfp_low`, `zfp_medium`, `zfp_high` | `npy_3d` | History encoding: NPY, netCDF, or a ZFP tier. netCDF and ZFP require their compiled libraries. NPY remains supported until its replacement meets the full agreed qualification scope. |
| [`keep_last_frames`](../src/init/config/v1_loader.cpp#L1855) | Integer ≥ 0 | `0` | Nonnegative history retention count; zero retains all completed frames. A positive setting deliberately removes older completed output according to the retention owner. |
| [`live_shm`](../src/init/config/v1_loader.cpp#L1757) | Boolean | `false` | Publish a separate local shared-memory stream. It does not replace durable history files. |
| [`live_shm_fields`](../src/init/config/v1_loader.cpp#L1759) | Inline list | `["w", "reflectivity_dbz", "vorticity_z"]` | Inline list of supported live-stream fields. The live-field validator checks membership and layout; choose fields available in the selected run. |
| [`live_shm_full_rate`](../src/init/config/v1_loader.cpp#L1763) | Boolean | `false` | Request live publication at every accepted step instead of the ordinary live interval, subject to the runtime stream policy. |
| [`live_shm_interval_s`](../src/init/config/v1_loader.cpp#L1761) | Real > 0 | `1` | Positive live shared-memory publication interval in s. |
| [`maximum_retained_history_files`](../src/init/config/v1_loader.cpp#L1856) | Integer ≥ 0 | `0` | Nonnegative retained-history file budget; zero disables this cap. The runtime checks whether the requested publication/retention plan can fit. |
| [`minimum_free_space_mib`](../src/init/config/v1_loader.cpp#L1858) | Integer ≥ 0 | `0` | Nonnegative free-space reserve in MiB; zero disables this reserve check. It does not reserve physical disk blocks. |
| [`netcdf_deflate_level`](../src/init/config/v1_loader.cpp#L1783) | Integer ≥ 0 | `2` | netCDF compression level from 0 through 9; 0 disables deflate. Used with netCDF history. |
| [`netcdf_shuffle`](../src/init/config/v1_loader.cpp#L1789) | Boolean | `true` | Enable byte shuffling before netCDF compression. |
| [`netcdf_target_chunk_kib`](../src/init/config/v1_loader.cpp#L1790) | Integer > 0 | `1024` | Positive netCDF target chunk size in KiB. Chunk layout affects I/O cost; it does not alter the grid. |
| [`preset`](../src/init/config/v1_loader.cpp#L1742) | `minimal`, `radar_ready`, `radar_zdr_exploratory`, `radar_polarimetric_ready`, `presentation`, `studio_full`, `dynamics_full`, `science_validation` | `minimal` | Named field set. Names containing `ready` are preset identifiers, not a statement that the selected experiment is Fully Qualified. Radar presets must agree with radar/microphysics selections. |
| [`qualification_diagnostics.interval_s`](../src/init/config/v1_loader.cpp#L1775) | Integer > 0 | `30` | Positive integer interval in s, accepted only with `moving_reference_v1`. Must equal enabled tracking cadence, be reachable exactly by fixed root/comparison timesteps, and divide run duration. |
| [`qualification_diagnostics.preset`](../src/init/config/v1_loader.cpp#L1771) | `none`, `moving_reference_v1` | `none` | Optional fixed diagnostic stream. `moving_reference_v1` publishes its prescribed lossless moving-reference fields and requires the matching flat-grid, tracking and cadence contract; it is not a user-defined field list. |
| [`tiered_write_cadence`](../src/init/config/v1_loader.cpp#L1852) | Boolean | `false` | Use separate cadence classes for published fields. This is format-independent, but not every temporal/history combination is admitted. |
| [`trajectories`](../src/init/config/v1_loader.cpp#L1776) | Boolean | `false` | Enable online Lagrangian parcel output, with its own seeding and sample controls. |
| [`trajectory_count`](../src/init/config/v1_loader.cpp#L1778) | Integer > 0 | `5` | Positive number of parcels requested. |
| [`trajectory_max_samples_per_parcel`](../src/init/config/v1_loader.cpp#L1781) | Integer > 0 | `16384` | Positive upper bound on retained samples for each parcel. |
| [`trajectory_sample_interval_s`](../src/init/config/v1_loader.cpp#L1779) | Real ≥ 0 | `0` | Nonnegative parcel sample spacing in s; zero retains the owner’s step-based sampling policy. |
| [`trajectory_seed_time_s`](../src/init/config/v1_loader.cpp#L1777) | Real ≥ 0 | `0` | Nonnegative parcel seeding time in s. |
| [`write_cadence_medium_s`](../src/init/config/v1_loader.cpp#L1853) | Integer > 0 | `30` | Positive integer medium-tier history interval in s. |
| [`write_cadence_slow_s`](../src/init/config/v1_loader.cpp#L1854) | Integer > 0 | `60` | Positive integer slow-tier history interval in s. |
| [`zfp_field_execution`](../src/init/config/v1_loader.cpp#L1803) | String | `"serial"` | `serial` or `field_parallel` compression scheduling. The field-parallel path has separate scratch/worker bounds and retained readiness obligations. |
| [`zfp_field_scratch_mib`](../src/init/config/v1_loader.cpp#L1813) | Integer > 0 | `64` | Scratch budget in MiB, integer in [1, 4096], for field compression scheduling. |
| [`zfp_field_workers`](../src/init/config/v1_loader.cpp#L1812) | Integer ≥ 0 | `0` | Requested compression workers in [0, 16]; zero leaves worker selection to runtime policy. |
| [`zfp_float16_prefilter`](../src/init/config/v1_loader.cpp#L1801) | Boolean | `false` | Request a float16 prefilter before coding. This changes precision before ZFP and is not covered merely by a ZFP accuracy bound. |
| [`zfp_keyframe_interval`](../src/init/config/v1_loader.cpp#L1799) | Integer ≥ 0 | `0` | Nonnegative keyframe interval; zero means every frame is independently decodable. Nonzero delta chains are a separate opt-in history contract. |
| [`zfp_loose_tolerance`](../src/init/config/v1_loader.cpp#L1829) | Real ≥ 0 | `0.01` | Nonnegative loose-class absolute tolerance in each affected field's native units. |
| [`zfp_mode`](../src/init/config/v1_loader.cpp#L1796) | String | `"accuracy"` | Requested ZFP coding mode: `accuracy`, `precision` or `rate`. The selected history/profile contract can impose a narrower allowed scope. |
| [`zfp_moderate_tolerance`](../src/init/config/v1_loader.cpp#L1828) | Real ≥ 0 | `0.001` | Nonnegative moderate-class absolute tolerance in each affected field's native units. |
| [`zfp_per_field_tolerances`](../src/init/config/v1_loader.cpp#L1826) | Boolean | `false` | Enable the field-class tolerance policy rather than a single global field tolerance. |
| [`zfp_precision_bits`](../src/init/config/v1_loader.cpp#L1797) | Integer > 0 | `16` | Integer precision-mode bit count in [1, 32]. History admission can impose a narrower scope. |
| [`zfp_predictive_delta`](../src/init/config/v1_loader.cpp#L1802) | Boolean | `false` | Request temporal predictive/delta encoding; restart, random access and admitted history constraints still apply. |
| [`zfp_rate_bps`](../src/init/config/v1_loader.cpp#L1798) | Integer > 0 | `8` | Positive integer rate-mode bits per scalar. Codec and history admission impose additional limits. |
| [`zfp_sparse_threshold`](../src/init/config/v1_loader.cpp#L1800) | Real ≥ 0 | `0` | Nonnegative sparse-elision threshold in native field units. Elision is a lossy transformation subject to the selected scientific contract. |
| [`zfp_tight_tolerance`](../src/init/config/v1_loader.cpp#L1827) | Real ≥ 0 | `0.0001` | Nonnegative tight-class absolute tolerance in each affected field's native units. |
| [`zfp_tolerance`](../src/init/config/v1_loader.cpp#L1795) | Real ≥ 0 | `0` | Absolute accuracy tolerance in the field's native units; zero derives the tier default (low 1e-2, medium 1e-3, high 1e-4). Consult per-field policy before assigning one tolerance across different units. |

## Extensions

Nimbus-specific controls for compute, retained research switches, stochastic forcing, execution policy, restart, diagnostics and timing. Research/control presence does not complete its science or backend coupling.

Prefix: `extensions`. Each name below is relative to that prefix.

| Key | Type / choices | When omitted | Meaning and restrictions |
| --- | --- | --- | --- |
| [`chaos.alpha_tend.diffusion`](../src/init/config/v1_loader.cpp#L2099) | Real ≥ 0 | `0.1` | Nonnegative dimensionless tendency-perturbation amplitude for `diffusion`. Activation follows the scheme and `apply_to_tendencies`. |
| [`chaos.alpha_tend.microphysics`](../src/init/config/v1_loader.cpp#L2099) | Real ≥ 0 | `0.3` | Nonnegative dimensionless tendency-perturbation amplitude for `microphysics`. Activation follows the scheme and `apply_to_tendencies`. |
| [`chaos.alpha_tend.pbl`](../src/init/config/v1_loader.cpp#L2099) | Real ≥ 0 | `0.2` | Nonnegative dimensionless tendency-perturbation amplitude for `pbl`. Activation follows the scheme and `apply_to_tendencies`. |
| [`chaos.apply_to_ic`](../src/init/config/v1_loader.cpp#L2035) | Inline list | `[]` | Inline list of distinct initial-state targets: `u`, `v`, `w`, `theta`, `qv`, `qc`, `qr`, `qi`, `qs`, `qg`, `qh`. Empty/omitted retains scheme defaults; selection does not bypass physical or native-profile restrictions. |
| [`chaos.apply_to_tendencies`](../src/init/config/v1_loader.cpp#L2037) | Inline list | `[]` | Inline list of distinct process targets: `microphysics`, `pbl`, `diffusion`. Empty/omitted retains scheme defaults. |
| [`chaos.member_id`](../src/init/config/v1_loader.cpp#L2031) | Integer | `0` | Integer ensemble-member identifier; distinguishes the member's stochastic stream. |
| [`chaos.native_generation_required`](../src/init/config/v1_loader.cpp#L2032) | Boolean | `false` | Require perturbation generation through the native owner instead of permitting a host-generated path. The selected backend/profile must provide it. |
| [`chaos.scheme`](../src/init/config/v1_loader.cpp#L2026) | `none`, `initial_conditions`, `boundary_layer`, `full_stochastic` | `none` | Stochastic-perturbation owner: disabled, initial conditions, boundary-layer perturbations or full stochastic forcing. Native generation and coupled-profile scope are separate requirements. |
| [`chaos.seed`](../src/init/config/v1_loader.cpp#L2030) | Integer | `0` | Integer reproducibility seed. Keep it with member ID and the spatial/temporal owner when comparing runs. |
| [`chaos.sigma_ic.qc`](../src/init/config/v1_loader.cpp#L2096) | Real ≥ 0 | `0` | Nonnegative initial-perturbation amplitude for `qc`, in kg kg⁻¹ dry air. This amplitude does not itself select the target; the scheme and `apply_to_ic` own activation. |
| [`chaos.sigma_ic.qg`](../src/init/config/v1_loader.cpp#L2096) | Real ≥ 0 | Absent from default amplitude map | Nonnegative initial-perturbation amplitude for `qg`, in kg kg⁻¹ dry air. This amplitude does not itself select the target; the scheme and `apply_to_ic` own activation. |
| [`chaos.sigma_ic.qh`](../src/init/config/v1_loader.cpp#L2096) | Real ≥ 0 | Absent from default amplitude map | Nonnegative initial-perturbation amplitude for `qh`, in kg kg⁻¹ dry air. This amplitude does not itself select the target; the scheme and `apply_to_ic` own activation. |
| [`chaos.sigma_ic.qi`](../src/init/config/v1_loader.cpp#L2096) | Real ≥ 0 | Absent from default amplitude map | Nonnegative initial-perturbation amplitude for `qi`, in kg kg⁻¹ dry air. This amplitude does not itself select the target; the scheme and `apply_to_ic` own activation. |
| [`chaos.sigma_ic.qr`](../src/init/config/v1_loader.cpp#L2096) | Real ≥ 0 | `0` | Nonnegative initial-perturbation amplitude for `qr`, in kg kg⁻¹ dry air. This amplitude does not itself select the target; the scheme and `apply_to_ic` own activation. |
| [`chaos.sigma_ic.qs`](../src/init/config/v1_loader.cpp#L2096) | Real ≥ 0 | Absent from default amplitude map | Nonnegative initial-perturbation amplitude for `qs`, in kg kg⁻¹ dry air. This amplitude does not itself select the target; the scheme and `apply_to_ic` own activation. |
| [`chaos.sigma_ic.qv`](../src/init/config/v1_loader.cpp#L2096) | Real ≥ 0 | `0.0005` | Nonnegative initial-perturbation amplitude for `qv`, in kg kg⁻¹ dry air. This amplitude does not itself select the target; the scheme and `apply_to_ic` own activation. |
| [`chaos.sigma_ic.theta`](../src/init/config/v1_loader.cpp#L2096) | Real ≥ 0 | `0.5` | Nonnegative initial-perturbation amplitude for `theta`, in K. This amplitude does not itself select the target; the scheme and `apply_to_ic` own activation. |
| [`chaos.sigma_ic.u`](../src/init/config/v1_loader.cpp#L2096) | Real ≥ 0 | `0.5` | Nonnegative initial-perturbation amplitude for `u`, in m s⁻¹. This amplitude does not itself select the target; the scheme and `apply_to_ic` own activation. |
| [`chaos.sigma_ic.v`](../src/init/config/v1_loader.cpp#L2096) | Real ≥ 0 | `0.5` | Nonnegative initial-perturbation amplitude for `v`, in m s⁻¹. This amplitude does not itself select the target; the scheme and `apply_to_ic` own activation. |
| [`chaos.sigma_ic.w`](../src/init/config/v1_loader.cpp#L2096) | Real ≥ 0 | `0.1` | Nonnegative initial-perturbation amplitude for `w`, in m s⁻¹. This amplitude does not itself select the target; the scheme and `apply_to_ic` own activation. |
| [`checkpoint.directory`](../src/init/config/v1_loader.cpp#L2122) | String | `""` | Checkpoint output directory. Empty/omitted uses the runtime's run-directory-based destination. |
| [`checkpoint.every_s`](../src/init/config/v1_loader.cpp#L2119) | Integer ≥ 0 | `0` | Nonnegative checkpoint interval in s; zero disables periodic checkpoints. Checkpoints retain continuation state, not just plot fields. |
| [`checkpoint.keep_last`](../src/init/config/v1_loader.cpp#L2120) | Integer ≥ 0 | `0` | Nonnegative checkpoint retention count; zero disables the count cap. |
| [`checkpoint.minimum_free_space_mib`](../src/init/config/v1_loader.cpp#L2121) | Integer ≥ 0 | `0` | Nonnegative free-space reserve in MiB for checkpoint publication; zero disables this reserve check. |
| [`checkpoint.resume_from`](../src/init/config/v1_loader.cpp#L2123) | String | `""` | Checkpoint path for continuation; empty starts from initialization. Configuration, executable and saved-state compatibility checks still apply. |
| [`compute.allow_fallback`](../src/init/config/v1_loader.cpp#L2015) | Boolean | `true` | Allow disclosed CPU recovery if accelerator initialization fails. Qualified execution forces effective fallback off; native-required scopes cannot use hidden substitution. |
| [`compute.backend`](../src/init/config/v1_loader.cpp#L2009) | `cpu`, `gpu`, `metal` | `cpu` | CPU, Vulkan (`gpu`, `vulkan`, `vk`) or Metal (`metal`, `mtl`). A recognized selector still needs a compiled backend, compatible device and admitted scientific combination. Aliases: `vulkan` → `gpu`; `vk` → `gpu`; `mtl` → `metal`. |
| [`compute.cuda_assist`](../src/init/config/v1_loader.cpp#L2017) | Boolean | `false` | Optional CUDA-assisted path on a compatible NVIDIA device while using Vulkan. True is rejected with other backend selections; it does not add a general CUDA solver backend. |
| [`compute.gpu_apply_microphysics`](../src/init/config/v1_loader.cpp#L2020) | Boolean | `true` | Request native state application for microphysics where the interaction policy supports it. No-op on CPU; scheme, geometry and maturity restrictions remain. |
| [`compute.gpu_apply_sgs`](../src/init/config/v1_loader.cpp#L2022) | Boolean | `false` | Request native application of SGS state changes where admitted. No-op on CPU; ownership/parity and exact-profile restrictions remain. |
| [`compute.gpu_apply_slow`](../src/init/config/v1_loader.cpp#L2021) | Boolean | `false` | Request native application of slow-source state changes where admitted. No-op on CPU; this switch is not a blanket native-coupling guarantee. |
| [`compute.memory_admission`](../src/init/config/v1_loader.cpp#L2011) | `warn`, `reject` | `warn` | Warn or reject when the phase-aware host-memory estimate exceeds its budget. This is separate from GPU memory topology and allocation policy. |
| [`compute.validate_parity`](../src/init/config/v1_loader.cpp#L2016) | Boolean | `false` | Enable sampled CPU/reference replay checks for native dispatches. This is a runtime checking cost, not a proof of full backend qualification. |
| [`diagnostics.deterministic_reductions`](../src/init/config/v1_loader.cpp#L2130) | Boolean | `false` | Use the deterministic reduction contract for scientific budgets. It does not promise bitwise identity for all fields across hardware or compilers. |
| [`diagnostics.scientific_every_steps`](../src/init/config/v1_loader.cpp#L2129) | Integer ≥ 0 | Compatibility cadence; omit key | Nonnegative scientific-diagnostic step cadence: zero disables, positive values select one-based step cadence. Omission preserves the historical output/live/debug cadence, or every step when periodic output is disabled. Internal sentinel -1 is **not** an accepted explicit YAML value. |
| [`execution.acknowledge_maturity_limits`](../src/init/config/v1_loader.cpp#L2104) | Boolean | Absent (typed `false`); omit key | Deprecated compatibility input; normally omit it. Explicit false is rejected in qualified mode, while true is rejected in exploratory mode. Ready profiles do not need an acknowledgment and this key cannot open an incomplete profile. |
| [`execution.mode`](../src/init/config/v1_loader.cpp#L2102) | `exploratory`, `qualified` | `exploratory` | `exploratory` permits disclosed recovery within remaining guards; `qualified` requests stricter admitted execution and fatal recovery/substitution behavior. The latter is a policy name, not the status Fully Qualified, and does not bypass incomplete engineering. |
| [`performance.report_every_steps`](../src/init/config/v1_loader.cpp#L2137) | Integer ≥ 0 | Inherit environment/runtime policy | Optional nonnegative timing-report cadence in steps. Zero means final summary only; omission preserves environment/runtime policy. |
| [`performance.timing`](../src/init/config/v1_loader.cpp#L2132) | Boolean | Inherit environment/runtime policy | Optional instrumentation switch. When omitted, existing performance environment policy is retained; explicit YAML overrides that policy. |
| [`performance.warmup_steps`](../src/init/config/v1_loader.cpp#L2142) | Integer ≥ 0 | Inherit environment/runtime policy | Optional nonnegative number of initial steps omitted from reported timing totals. These steps still execute the normal simulation. |
| [`solver.axis_u_m1_filter`](../src/init/config/v1_loader.cpp#L2025) | Boolean | `false` | Retained axis m=1 radial-velocity filter switch. Default off; parsed presence does not establish a supported Cartesian scientific use. |
| [`solver.lid_bc_gamma`](../src/init/config/v1_loader.cpp#L2024) | Finite real | `-1` | Lid boundary damping parameter. A finite value in (0, 1] selects the base-state-anchored form; other values retain the linear hydrostatic form. This is a solver research control with profile restrictions. |
| [`solver.slow_rk2`](../src/init/config/v1_loader.cpp#L2023) | Boolean | `true` | Use Heun/RK2 for slow tendencies. The false branch retains Forward Euler for comparison and is not an appropriate substitute for the stable retained UP5 pairing; ordinary scientific admission may reject it. |
| [`validation.guard_mode`](../src/init/config/v1_loader.cpp#L2124) | `off`, `warn`, `error` | `warn` | Field-contract guard behavior: off, warn or error. Qualified/native-required profiles can require error; lowering the guard does not make an unsupported profile supported. |

## Per-field ZFP map

`output.zfp_field_tolerance.<field>` supplies a **strictly positive finite**
absolute tolerance for a named field, in that field's native units. The map is
empty by default. A nonempty field name passes the map-key parser; the output
policy must still accept that field and its relationship to the selected
preset, format, tolerance mode and qualified scope. This is an open mapping,
not a fixed list of eight Studio controls.

For example, the syntax of one entry is:

```yaml
output:
  zfp_field_tolerance:
    theta: 0.001
```

This fragment is not a complete experiment and does not select a ZFP output
format by itself. A tolerance for potential temperature in K is not
interchangeable with a mixing-ratio tolerance in kg kg⁻¹ dry air.

## Command-line companion

Command-line overrides are separate from the YAML keys. In particular,
`--threads` configures OpenMP workers, output/checkpoint paths select filesystem
destinations, and several diagnostics/live-stream controls exist only at this
boundary. Inspect `--help` and `--version` on the executable actually being used.
The current maintained help is reproduced below. Internal qualification flags
require their special test builds; they are not ordinary configuration escape
hatches or instructions to bypass a readiness guard.

```text
Usage:
  nimbus_sim --config PATH [options]
  nimbus_sim --sounding PATH [options]
  nimbus_sim --migrate-config PATH
  nimbus_sim --scheme-catalog=json
  nimbus_sim --compute-ownership-matrix=json
  nimbus_sim --compute-process-ownership-matrix=json

Required input:
  --config PATH                 Simulation YAML. Required unless --sounding is used.
  --sounding PATH               Override the sounding; uses the public 1000 m Kessler preset as base.

Execution:
  --validate-only                Strictly load/project the config and build its
                                 1-D initial-condition sounding; allocate no
                                 simulation fields and run no timestep.
  --preflight-context=json        Emit a bounded native context receipt for
                                 --validate-only without science overrides.
  --headless                    Run without the Studio UI.
  --duration SECONDS            Override configured run duration.
  --write-every=SECONDS         Override output cadence (0 disables snapshots).
  --outdir=PATH                 Snapshot directory (default: data/exports).
  --checkpoint-every=SECONDS    Portable checkpoint cadence (0 disables).
  --checkpoint-dir=PATH         Checkpoint directory (default: OUTDIR/checkpoints).
  --resume=PATH                 Continue from a verified Nimbus checkpoint into a new OUTDIR.
  --compute-backend KIND        cpu, vulkan, gpu, or metal.
  --threads COUNT               OpenMP worker count; overrides OMP_NUM_THREADS.
  --acoustic-substeps COUNT     Override split-explicit acoustic substep count.
  --dt-large SECONDS            Override the positive large timestep.
  --qualification-hydrometeor-pulse
                                 Internal A-grid qualification binary only.
  --qualification-enable-milbrandt
                                 Internal qualification binary only; admits
                                 the maintained Milbrandt CPU C-grid campaign.
  --qualification-milbrandt-mixed-phase
                                 Same admission plus a mixed-phase seed.
  --qualification-acoustic-wave Internal A-grid qualification binary only.
  --qualification-acoustic-standing-wave
                                 Internal A-grid qualification binary only.
  --qualification-acoustic-pulse
                                 Internal A-grid qualification binary only.
  --qualification-moving-nest  Internal qualification binary only; admits the
                                 exact moving-supercell campaign profile.
  --export-ms=MILLISECONDS      Delay between live exports.

Diagnostics and output:
  --log-profile PROFILE         quiet, normal, or debug.
  --timing | --no-timing        Enable or disable performance timing.
  --timing-every=STEPS          Timing report cadence.
  --guard-mode MODE             off, sanitize, or strict.
  --guard-fail-on CONDITION     nonfinite, bounds, or both.
  --guard-scope SCOPE           required or exported.
  --guard-report PATH           Validation report path.
  --live-shm                    Publish fields to Studio shared memory.
  --live-shm-fields=LIST        Comma-separated shared-memory fields.
  --live-shm-name=NAME          Logical endpoint (default: /tmv_live); use an
                                 explicit name to isolate concurrent runs.

Trajectories:
  --trajectories
  --trajectory-seed-time=SECONDS
  --trajectory-count=COUNT

Other:
  --migrate-config PATH         Write a migrated nimbus.v1 config to stdout.
  --scheme-catalog=json         Write the exact compiled scheme catalog as JSON.
  --compute-ownership-matrix=json
                                 Write the native state-commit ownership matrix as JSON.
  --compute-process-ownership-matrix=json
                                 Write the native solver-process ownership matrix as JSON.
  -h, --help                    Show this help and exit.
  --version                     Show build/schema identity and exit.
```

Build-worker limits (`NIMBUS_BUILD_JOBS`, `CARGO_BUILD_JOBS` and CMake's
`CMAKE_BUILD_PARALLEL_LEVEL`) belong to compilation, not the namelist or
simulation thread count. Runtime environment also supplies library/driver
discovery, OpenMP policy and optional coefficient locations such as
`NIMBUS_RRTMG_LW_COEFFICIENTS` and `NIMBUS_RRTMG_SW_COEFFICIENTS`.
Use [BUILDING.md](BUILDING.md) for provisioning. These operating-environment
settings are not additional strict-v1 keys.

## Legacy compatibility

Unversioned legacy configuration loading and v0 migration still exist. Their
spellings are not additional `nimbus.v1` keys: adding an old flat key to a v1
document causes an unknown-key error. Migrate a copy with `--migrate-config`,
inspect its warnings and resulting YAML, then validate the new document.
Migration can translate, resolve, drop or reject legacy settings; it does not
certify that the old experiment had the same physical interpretation.

The maintained compatibility owners are
[the legacy runtime loader](../src/core/runtime/runtime_config.cpp) and
[the migration implementation](../src/init/config/v0_to_v1.cpp), including its
[physics](../src/init/config/v0_to_v1_physics.cpp),
[initialization](../src/init/config/v0_to_v1_initialization.cpp) and
[runtime/output](../src/init/config/v0_to_v1_runtime.cpp) translators.

The compatibility inventory below lists all 324 literal keys read by
the legacy runtime/output owners, including the schema branch selector. It is
an index for understanding old files, not a second recommended input schema.
Legacy omission defaults and accepted values can differ even when a spelling
also exists in v1. Unknown legacy keys lack v1's consumed-key rejection pass;
silently tolerating a key does not mean an operator uses it.

Only directly traced migration renames/conversions are labeled as such. Other
entries require inspecting migration diagnostics; a missing mapping in this
table does not establish that migration is impossible. Legacy `nested.enabled:
true` is rejected by the nested-grid owner, separately from modern
`domain_motion.nesting`. Legacy Milbrandt `triple_moment: false` can be consumed,
while true is rejected; strict v1 rejects the explicit key regardless of value.

<details>
<summary>Complete legacy reader inventory (324 keys)</summary>

| Legacy key | Relationship to v1 |
| --- | --- |
| [`boundary_layer.apply_surface_fluxes`](../src/core/runtime/runtime_config.cpp#L1697) | [Migrates to `surface.apply_fluxes`](../src/init/config/v0_to_v1_physics.cpp#L278). |
| [`boundary_layer.clear_sky_only`](../src/core/runtime/runtime_config.cpp#L1703) | Same spelling exists; compare semantics and omission defaults. |
| [`boundary_layer.dt`](../src/core/runtime/runtime_config.cpp#L1743) | [Migrates to `boundary_layer.cadence_s`](../src/init/config/v0_to_v1_physics.cpp#L342). |
| [`boundary_layer.dt_pbl`](../src/core/runtime/runtime_config.cpp#L1755) | [Migrates to `boundary_layer.cadence_s`](../src/init/config/v0_to_v1_physics.cpp#L337). |
| [`boundary_layer.enable_tke`](../src/core/runtime/runtime_config.cpp#L1707) | Legacy-only spelling; inspect migration diagnostics. |
| [`boundary_layer.min_ustar`](../src/core/runtime/runtime_config.cpp#L1781) | Legacy-only spelling; inspect migration diagnostics. |
| [`boundary_layer.pbl_max_height`](../src/core/runtime/runtime_config.cpp#L1768) | Legacy-only spelling; inspect migration diagnostics. |
| [`boundary_layer.pbl_top_method`](../src/core/runtime/runtime_config.cpp#L1711) | Same spelling exists; compare semantics and omission defaults. |
| [`boundary_layer.scheme`](../src/core/runtime/runtime_config.cpp#L1692) | Same spelling exists; compare semantics and omission defaults. |
| [`boundary_layer.surface_layer`](../src/core/runtime/runtime_config.cpp#L1740) | Legacy-only spelling; inspect migration diagnostics. |
| [`boundary_layer.surface_layer_id`](../src/core/runtime/runtime_config.cpp#L1741) | Legacy-only spelling; inspect migration diagnostics. |
| [`chaos.Lx`](../src/core/runtime/runtime_config.cpp#L3123) | Legacy-only spelling; inspect migration diagnostics. |
| [`chaos.Ly`](../src/core/runtime/runtime_config.cpp#L3135) | Legacy-only spelling; inspect migration diagnostics. |
| [`chaos.apply_to_ic`](../src/core/runtime/runtime_config.cpp#L3191) | Legacy-only spelling; inspect migration diagnostics. |
| [`chaos.apply_to_tendencies`](../src/core/runtime/runtime_config.cpp#L3195) | Legacy-only spelling; inspect migration diagnostics. |
| [`chaos.dt_chaos`](../src/core/runtime/runtime_config.cpp#L3099) | Legacy-only spelling; inspect migration diagnostics. |
| [`chaos.filter_id`](../src/core/runtime/runtime_config.cpp#L3147) | Legacy-only spelling; inspect migration diagnostics. |
| [`chaos.member_id`](../src/core/runtime/runtime_config.cpp#L3076) | [Migrates to `extensions.chaos.member_id`](../src/init/config/v0_to_v1_runtime.cpp#L615). |
| [`chaos.native_generation_required`](../src/core/runtime/runtime_config.cpp#L3088) | Legacy-only spelling; inspect migration diagnostics. |
| [`chaos.scheme`](../src/core/runtime/runtime_config.cpp#L3060) | [Migrates to `extensions.chaos.scheme`](../src/init/config/v0_to_v1_runtime.cpp#L599). |
| [`chaos.seed`](../src/core/runtime/runtime_config.cpp#L3064) | [Migrates to `extensions.chaos.seed`](../src/init/config/v0_to_v1_runtime.cpp#L610). |
| [`chaos.taper_id`](../src/core/runtime/runtime_config.cpp#L3163) | Legacy-only spelling; inspect migration diagnostics. |
| [`chaos.taper_z1`](../src/core/runtime/runtime_config.cpp#L3167) | Legacy-only spelling; inspect migration diagnostics. |
| [`chaos.taper_z2`](../src/core/runtime/runtime_config.cpp#L3179) | Legacy-only spelling; inspect migration diagnostics. |
| [`chaos.tau_t`](../src/core/runtime/runtime_config.cpp#L3111) | Legacy-only spelling; inspect migration diagnostics. |
| [`chaos.xi_max`](../src/core/runtime/runtime_config.cpp#L3151) | Legacy-only spelling; inspect migration diagnostics. |
| [`checkpoint.directory`](../src/core/runtime/runtime_config.cpp#L5086) | Legacy-only spelling; inspect migration diagnostics. |
| [`checkpoint.every_s`](../src/core/runtime/runtime_config.cpp#L5066) | Legacy-only spelling; inspect migration diagnostics. |
| [`checkpoint.keep_last`](../src/core/runtime/runtime_config.cpp#L5074) | Legacy-only spelling; inspect migration diagnostics. |
| [`checkpoint.minimum_free_space_mib`](../src/core/runtime/runtime_config.cpp#L5080) | Legacy-only spelling; inspect migration diagnostics. |
| [`checkpoint.resume_from`](../src/core/runtime/runtime_config.cpp#L5090) | Legacy-only spelling; inspect migration diagnostics. |
| [`coordinate_system`](../src/core/runtime/runtime_config.cpp#L3747) | [Migrates to `grid.coordinate_system`](../src/init/config/v0_to_v1_initialization.cpp#L22). |
| [`diagnostics.deterministic_reductions`](../src/core/runtime/runtime_config.cpp#L3846) | [Migrates to `extensions.diagnostics.deterministic_reductions`](../src/init/config/v0_to_v1_runtime.cpp#L732). |
| [`diagnostics.scientific_every_steps`](../src/core/runtime/runtime_config.cpp#L3832) | [Migrates to `extensions.diagnostics.scientific_every_steps`](../src/init/config/v0_to_v1_runtime.cpp#L725). |
| [`domain_motion.frame_velocity.u_ms`](../src/core/runtime/runtime_config.cpp#L3993) | Same spelling exists; compare semantics and omission defaults. |
| [`domain_motion.frame_velocity.v_ms`](../src/core/runtime/runtime_config.cpp#L3994) | Same spelling exists; compare semantics and omission defaults. |
| [`domain_motion.lifetime_planning.action`](../src/core/runtime/runtime_config.cpp#L1105) | Same spelling exists; compare semantics and omission defaults. |
| [`domain_motion.lifetime_planning.cross_track_buffer_m`](../src/core/runtime/runtime_config.cpp#L1116) | Same spelling exists; compare semantics and omission defaults. |
| [`domain_motion.lifetime_planning.downstream_buffer_m`](../src/core/runtime/runtime_config.cpp#L1115) | Same spelling exists; compare semantics and omission defaults. |
| [`domain_motion.lifetime_planning.enabled`](../src/core/runtime/runtime_config.cpp#L1071) | Same spelling exists; compare semantics and omission defaults. |
| [`domain_motion.lifetime_planning.explicit_u_ms`](../src/core/runtime/runtime_config.cpp#L1111) | Same spelling exists; compare semantics and omission defaults. |
| [`domain_motion.lifetime_planning.explicit_v_ms`](../src/core/runtime/runtime_config.cpp#L1112) | Same spelling exists; compare semantics and omission defaults. |
| [`domain_motion.lifetime_planning.motion_source`](../src/core/runtime/runtime_config.cpp#L1107) | Same spelling exists; compare semantics and omission defaults. |
| [`domain_motion.lifetime_planning.motion_uncertainty_ms`](../src/core/runtime/runtime_config.cpp#L1117) | Same spelling exists; compare semantics and omission defaults. |
| [`domain_motion.lifetime_planning.upstream_buffer_m`](../src/core/runtime/runtime_config.cpp#L1114) | Same spelling exists; compare semantics and omission defaults. |
| [`domain_motion.mode`](../src/core/runtime/runtime_config.cpp#L3979) | Same spelling exists; compare semantics and omission defaults. |
| [`domain_motion.nesting.boundary_stencil_parent_cells`](../src/core/runtime/runtime_config.cpp#L2376) | Same spelling exists; compare semantics and omission defaults. |
| [`domain_motion.nesting.child_cells_x`](../src/core/runtime/runtime_config.cpp#L2370) | Same spelling exists; compare semantics and omission defaults. |
| [`domain_motion.nesting.child_cells_y`](../src/core/runtime/runtime_config.cpp#L2372) | Same spelling exists; compare semantics and omission defaults. |
| [`domain_motion.nesting.child_origin_parent_cell_i`](../src/core/runtime/runtime_config.cpp#L2366) | Same spelling exists; compare semantics and omission defaults. |
| [`domain_motion.nesting.child_origin_parent_cell_j`](../src/core/runtime/runtime_config.cpp#L2368) | Same spelling exists; compare semantics and omission defaults. |
| [`domain_motion.nesting.enabled`](../src/core/runtime/runtime_config.cpp#L2357) | Same spelling exists; compare semantics and omission defaults. |
| [`domain_motion.nesting.feedback`](../src/core/runtime/runtime_config.cpp#L2386) | Same spelling exists; compare semantics and omission defaults. |
| [`domain_motion.nesting.parent_ratio_x`](../src/core/runtime/runtime_config.cpp#L2351) | Same spelling exists; compare semantics and omission defaults. |
| [`domain_motion.nesting.parent_ratio_y`](../src/core/runtime/runtime_config.cpp#L2353) | Same spelling exists; compare semantics and omission defaults. |
| [`domain_motion.nesting.parent_time_step_ratio`](../src/core/runtime/runtime_config.cpp#L2355) | Same spelling exists; compare semantics and omission defaults. |
| [`domain_motion.nesting.relaxation_e_folding_time_s`](../src/core/runtime/runtime_config.cpp#L2378) | Same spelling exists; compare semantics and omission defaults. |
| [`domain_motion.nesting.relaxation_width_child_cells`](../src/core/runtime/runtime_config.cpp#L2374) | Same spelling exists; compare semantics and omission defaults. |
| [`domain_motion.tracking.branch_preference`](../src/core/runtime/runtime_config.cpp#L3904) | Same spelling exists; compare semantics and omission defaults. |
| [`domain_motion.tracking.branch_velocity_uncertainty_ms`](../src/core/runtime/runtime_config.cpp#L3903) | Same spelling exists; compare semantics and omission defaults. |
| [`domain_motion.tracking.cooldown_s`](../src/core/runtime/runtime_config.cpp#L3891) | Same spelling exists; compare semantics and omission defaults. |
| [`domain_motion.tracking.enabled`](../src/core/runtime/runtime_config.cpp#L3859) | Same spelling exists; compare semantics and omission defaults. |
| [`domain_motion.tracking.inner_margin_m`](../src/core/runtime/runtime_config.cpp#L3890) | Same spelling exists; compare semantics and omission defaults. |
| [`domain_motion.tracking.interval_s`](../src/core/runtime/runtime_config.cpp#L3889) | Same spelling exists; compare semantics and omission defaults. |
| [`domain_motion.tracking.lost_after_samples`](../src/core/runtime/runtime_config.cpp#L3937) | Same spelling exists; compare semantics and omission defaults. |
| [`domain_motion.tracking.maximum_motion_acceleration_ms2`](../src/core/runtime/runtime_config.cpp#L3901) | Same spelling exists; compare semantics and omission defaults. |
| [`domain_motion.tracking.maximum_motion_speed_ms`](../src/core/runtime/runtime_config.cpp#L3900) | Same spelling exists; compare semantics and omission defaults. |
| [`domain_motion.tracking.maximum_shift_cells`](../src/core/runtime/runtime_config.cpp#L3913) | Same spelling exists; compare semantics and omission defaults. |
| [`domain_motion.tracking.minimum_absolute_updraft_helicity_m2s2`](../src/core/runtime/runtime_config.cpp#L3897) | Same spelling exists; compare semantics and omission defaults. |
| [`domain_motion.tracking.minimum_component_cells`](../src/core/runtime/runtime_config.cpp#L3921) | Same spelling exists; compare semantics and omission defaults. |
| [`domain_motion.tracking.minimum_condensate_path_kgm2`](../src/core/runtime/runtime_config.cpp#L3894) | Same spelling exists; compare semantics and omission defaults. |
| [`domain_motion.tracking.minimum_confidence`](../src/core/runtime/runtime_config.cpp#L3892) | Same spelling exists; compare semantics and omission defaults. |
| [`domain_motion.tracking.minimum_continuity_samples`](../src/core/runtime/runtime_config.cpp#L3929) | Same spelling exists; compare semantics and omission defaults. |
| [`domain_motion.tracking.minimum_core_area_m2`](../src/core/runtime/runtime_config.cpp#L3899) | Same spelling exists; compare semantics and omission defaults. |
| [`domain_motion.tracking.minimum_updraft_ms`](../src/core/runtime/runtime_config.cpp#L3893) | Same spelling exists; compare semantics and omission defaults. |
| [`domain_motion.tracking.recycle_domain`](../src/core/runtime/runtime_config.cpp#L3863) | Same spelling exists; compare semantics and omission defaults. |
| [`domain_motion.tracking.topology_confirmation_samples`](../src/core/runtime/runtime_config.cpp#L3944) | Same spelling exists; compare semantics and omission defaults. |
| [`domain_motion.tracking.updraft_helicity_bottom_m_agl`](../src/core/runtime/runtime_config.cpp#L3895) | Same spelling exists; compare semantics and omission defaults. |
| [`domain_motion.tracking.updraft_helicity_top_m_agl`](../src/core/runtime/runtime_config.cpp#L3896) | Same spelling exists; compare semantics and omission defaults. |
| [`duration_s`](../src/core/runtime/runtime_config.cpp#L5111) | [Migrates to `time.run_time`](../src/init/config/v0_to_v1_initialization.cpp#L42). |
| [`dynamics.scheme`](../src/core/runtime/runtime_config.cpp#L2092) | Same spelling exists; compare semantics and omission defaults. |
| [`environment.cape_target_jkg`](../src/core/runtime/runtime_config.cpp#L3271) | [Migrates to `initial_conditions.sounding.cape_jkg`](../src/init/config/v0_to_v1_initialization.cpp#L110). |
| [`environment.hodograph.type`](../src/core/runtime/runtime_config.cpp#L2580) | [Migrates to `initial_conditions.hodograph.type`](../src/init/config/v0_to_v1_initialization.cpp#L258). |
| [`environment.hodograph.u_1km_ms`](../src/core/runtime/runtime_config.cpp#L2509) | Legacy-only spelling; inspect migration diagnostics. |
| [`environment.hodograph.u_6km_ms`](../src/core/runtime/runtime_config.cpp#L2537) | Legacy-only spelling; inspect migration diagnostics. |
| [`environment.hodograph.u_sfc_ms`](../src/core/runtime/runtime_config.cpp#L2481) | Legacy-only spelling; inspect migration diagnostics. |
| [`environment.hodograph.v_1km_ms`](../src/core/runtime/runtime_config.cpp#L2523) | Legacy-only spelling; inspect migration diagnostics. |
| [`environment.hodograph.v_6km_ms`](../src/core/runtime/runtime_config.cpp#L2551) | Legacy-only spelling; inspect migration diagnostics. |
| [`environment.hodograph.v_sfc_ms`](../src/core/runtime/runtime_config.cpp#L2495) | Legacy-only spelling; inspect migration diagnostics. |
| [`environment.sfc_qv_kgkg`](../src/core/runtime/runtime_config.cpp#L3297) | [Migrates to `initial_conditions.sounding.moisture_qv_kgkg`](../src/init/config/v0_to_v1_initialization.cpp#L105). |
| [`environment.sfc_theta_k`](../src/core/runtime/runtime_config.cpp#L3285) | [Migrates to `initial_conditions.sounding.surface_theta_k`](../src/init/config/v0_to_v1_initialization.cpp#L100). |
| [`environment.sounding.allow_placeholder_profiles`](../src/core/runtime/runtime_config.cpp#L3428) | Legacy-only spelling; inspect migration diagnostics. |
| [`environment.sounding.enabled`](../src/core/runtime/runtime_config.cpp#L3323) | Legacy-only spelling; inspect migration diagnostics. |
| [`environment.sounding.extrapolate_above_top`](../src/core/runtime/runtime_config.cpp#L3367) | Legacy-only spelling; inspect migration diagnostics. |
| [`environment.sounding.extrapolate_below_ground`](../src/core/runtime/runtime_config.cpp#L3362) | Legacy-only spelling; inspect migration diagnostics. |
| [`environment.sounding.file_path`](../src/core/runtime/runtime_config.cpp#L3335) | [Migrates to `initial_conditions.sounding.file_path`](../src/init/config/v0_to_v1_initialization.cpp#L179). |
| [`environment.sounding.interpolation_method`](../src/core/runtime/runtime_config.cpp#L3348) | Legacy-only spelling; inspect migration diagnostics. |
| [`environment.sounding.max_pressure_hpa`](../src/core/runtime/runtime_config.cpp#L3386) | Legacy-only spelling; inspect migration diagnostics. |
| [`environment.sounding.max_temperature_k`](../src/core/runtime/runtime_config.cpp#L3414) | Legacy-only spelling; inspect migration diagnostics. |
| [`environment.sounding.min_pressure_hpa`](../src/core/runtime/runtime_config.cpp#L3372) | Legacy-only spelling; inspect migration diagnostics. |
| [`environment.sounding.min_temperature_k`](../src/core/runtime/runtime_config.cpp#L3400) | Legacy-only spelling; inspect migration diagnostics. |
| [`environment.sounding.path`](../src/core/runtime/runtime_config.cpp#L3339) | Legacy-only spelling; inspect migration diagnostics. |
| [`environment.sounding.require_winds`](../src/core/runtime/runtime_config.cpp#L3467) | Legacy-only spelling; inspect migration diagnostics. |
| [`environment.sounding.scheme`](../src/core/runtime/runtime_config.cpp#L3327) | Legacy-only spelling; inspect migration diagnostics. |
| [`environment.sounding.scheme_id`](../src/core/runtime/runtime_config.cpp#L3331) | Legacy-only spelling; inspect migration diagnostics. |
| [`environment.sounding.targets.cape_jkg`](../src/core/runtime/runtime_config.cpp#L3541) | [Migrates to `initial_conditions.sounding.cape_jkg`](../src/init/config/v0_to_v1_initialization.cpp#L191). |
| [`environment.sounding.targets.cin_jkg`](../src/core/runtime/runtime_config.cpp#L3549) | [Migrates to `initial_conditions.sounding.cin_jkg`](../src/init/config/v0_to_v1_initialization.cpp#L196). |
| [`environment.sounding.targets.el_m`](../src/core/runtime/runtime_config.cpp#L3573) | [Migrates to `initial_conditions.sounding.el_m`](../src/init/config/v0_to_v1_initialization.cpp#L211). |
| [`environment.sounding.targets.lcl_m`](../src/core/runtime/runtime_config.cpp#L3557) | [Migrates to `initial_conditions.sounding.lcl_m`](../src/init/config/v0_to_v1_initialization.cpp#L201). |
| [`environment.sounding.targets.lfc_m`](../src/core/runtime/runtime_config.cpp#L3565) | [Migrates to `initial_conditions.sounding.lfc_m`](../src/init/config/v0_to_v1_initialization.cpp#L206). |
| [`environment.sounding.targets.moisture_qv_kgkg`](../src/core/runtime/runtime_config.cpp#L3589) | [Migrates to `initial_conditions.sounding.moisture_qv_kgkg`](../src/init/config/v0_to_v1_initialization.cpp#L221). |
| [`environment.sounding.targets.surface_theta_k`](../src/core/runtime/runtime_config.cpp#L3581) | [Migrates to `initial_conditions.sounding.surface_theta_k`](../src/init/config/v0_to_v1_initialization.cpp#L216). |
| [`environment.sounding.type`](../src/core/runtime/runtime_config.cpp#L3473) | [Migrates to `initial_conditions.sounding.type`](../src/init/config/v0_to_v1_initialization.cpp#L144). |
| [`environment.sounding.use_fallback_profiles`](../src/core/runtime/runtime_config.cpp#L3343) | Legacy-only spelling; inspect migration diagnostics. |
| [`environment.tropopause_z_m`](../src/core/runtime/runtime_config.cpp#L3309) | [Migrates to `initial_conditions.sounding.tropopause_m`](../src/init/config/v0_to_v1_initialization.cpp#L115). |
| [`execution.acknowledge_maturity_limits`](../src/core/runtime/runtime_config.cpp#L5050) | Legacy-only spelling; inspect migration diagnostics. |
| [`execution.mode`](../src/core/runtime/runtime_config.cpp#L5044) | Legacy-only spelling; inspect migration diagnostics. |
| [`grid.dt`](../src/core/runtime/runtime_config.cpp#L5295) | [Migrates to `time.dt_large`](../src/init/config/v0_to_v1_initialization.cpp#L83). |
| [`grid.dx`](../src/core/runtime/runtime_config.cpp#L5252) | Same spelling exists; compare semantics and omission defaults. |
| [`grid.dy`](../src/core/runtime/runtime_config.cpp#L5269) | Same spelling exists; compare semantics and omission defaults. |
| [`grid.dz`](../src/core/runtime/runtime_config.cpp#L5282) | Same spelling exists; compare semantics and omission defaults. |
| [`grid.nx`](../src/core/runtime/runtime_config.cpp#L5201) | Same spelling exists; compare semantics and omission defaults. |
| [`grid.ny`](../src/core/runtime/runtime_config.cpp#L5218) | Same spelling exists; compare semantics and omission defaults. |
| [`grid.nz`](../src/core/runtime/runtime_config.cpp#L5235) | Same spelling exists; compare semantics and omission defaults. |
| [`grid.staggering`](../src/core/runtime/runtime_config.cpp#L3761) | Same spelling exists; compare semantics and omission defaults. |
| [`hardware.memory_admission`](../src/core/runtime/runtime_config.cpp#L5319) | [Migrates to `extensions.compute.memory_admission`](../src/init/config/v0_to_v1_runtime.cpp#L183). |
| [`logging.profile`](../src/core/runtime/runtime_config.cpp#L3778) | Legacy-only spelling; inspect migration diagnostics. |
| [`microphysics.kessler.accretion_coefficient`](../src/core/runtime/runtime_config.cpp#L1867) | Same spelling exists; compare semantics and omission defaults. |
| [`microphysics.kessler.evaporation_efficiency`](../src/core/runtime/runtime_config.cpp#L1869) | Same spelling exists; compare semantics and omission defaults. |
| [`microphysics.lin.accretion_coefficient`](../src/core/runtime/runtime_config.cpp#L1871) | Same spelling exists; compare semantics and omission defaults. |
| [`microphysics.lin.evaporation_rate_s_inv`](../src/core/runtime/runtime_config.cpp#L1872) | Same spelling exists; compare semantics and omission defaults. |
| [`microphysics.milbrandt.evaporation_rate_s_inv`](../src/core/runtime/runtime_config.cpp#L1879) | Same spelling exists; compare semantics and omission defaults. |
| [`microphysics.milbrandt.hail_processes`](../src/core/runtime/runtime_config.cpp#L1888) | Same spelling exists; compare semantics and omission defaults. |
| [`microphysics.milbrandt.triple_moment`](../src/core/runtime/runtime_config.cpp#L1881) | Same spelling exists; compare semantics and omission defaults. |
| [`microphysics.scheme`](../src/core/runtime/runtime_config.cpp#L1799) | Same spelling exists; compare semantics and omission defaults. |
| [`microphysics.shared.autoconversion_rate_s_inv`](../src/core/runtime/runtime_config.cpp#L1865) | Same spelling exists; compare semantics and omission defaults. |
| [`microphysics.shared.cloud_water_threshold_kgkg`](../src/core/runtime/runtime_config.cpp#L1863) | Same spelling exists; compare semantics and omission defaults. |
| [`microphysics.thompson.evaporation_rate_s_inv`](../src/core/runtime/runtime_config.cpp#L1873) | Same spelling exists; compare semantics and omission defaults. |
| [`microphysics.thompson.ice_friendly_aerosol_surface_m3`](../src/core/runtime/runtime_config.cpp#L1877) | Same spelling exists; compare semantics and omission defaults. |
| [`microphysics.thompson.water_friendly_aerosol_surface_m3`](../src/core/runtime/runtime_config.cpp#L1875) | Same spelling exists; compare semantics and omission defaults. |
| [`nested.enabled`](../src/core/runtime/runtime_config.cpp#L2417) | Legacy-only spelling; inspect migration diagnostics. |
| [`nested.refinement`](../src/core/runtime/runtime_config.cpp#L2423) | Legacy-only spelling; inspect migration diagnostics. |
| [`nested.size_r`](../src/core/runtime/runtime_config.cpp#L2436) | Legacy-only spelling; inspect migration diagnostics. |
| [`nested.size_th`](../src/core/runtime/runtime_config.cpp#L2449) | Legacy-only spelling; inspect migration diagnostics. |
| [`nested.size_z`](../src/core/runtime/runtime_config.cpp#L2462) | Legacy-only spelling; inspect migration diagnostics. |
| [`numerics.advection`](../src/core/runtime/runtime_config.cpp#L2101) | Legacy-only spelling; inspect migration diagnostics. |
| [`numerics.advection.scheme`](../src/core/runtime/runtime_config.cpp#L2105) | [Migrates to `advection.scheme`](../src/init/config/v0_to_v1_runtime.cpp#L233). |
| [`numerics.compute.allow_fallback`](../src/core/runtime/runtime_config.cpp#L2817) | [Migrates to `extensions.compute.allow_fallback`](../src/init/config/v0_to_v1_runtime.cpp#L205). |
| [`numerics.compute.backend`](../src/core/runtime/runtime_config.cpp#L2791) | [Migrates to `extensions.compute.backend`](../src/init/config/v0_to_v1_runtime.cpp#L167). |
| [`numerics.compute.cuda_assist`](../src/core/runtime/runtime_config.cpp#L2825) | [Migrates to `extensions.compute.cuda_assist`](../src/init/config/v0_to_v1_runtime.cpp#L213). |
| [`numerics.compute.device_index`](../src/core/runtime/runtime_config.cpp#L2804) | Legacy-only spelling; inspect migration diagnostics. |
| [`numerics.compute.gpu_apply_microphysics`](../src/core/runtime/runtime_config.cpp#L2827) | Legacy-only spelling; inspect migration diagnostics. |
| [`numerics.compute.gpu_apply_sgs`](../src/core/runtime/runtime_config.cpp#L2837) | Legacy-only spelling; inspect migration diagnostics. |
| [`numerics.compute.gpu_apply_slow`](../src/core/runtime/runtime_config.cpp#L2832) | Legacy-only spelling; inspect migration diagnostics. |
| [`numerics.compute.templates.vertical_flux`](../src/core/runtime/runtime_config.cpp#L2842) | Legacy-only spelling; inspect migration diagnostics. |
| [`numerics.compute.validate_parity`](../src/core/runtime/runtime_config.cpp#L2821) | [Migrates to `extensions.compute.validate_parity`](../src/init/config/v0_to_v1_runtime.cpp#L197). |
| [`numerics.diffusion`](../src/core/runtime/runtime_config.cpp#L2115) | Legacy-only spelling; inspect migration diagnostics. |
| [`numerics.diffusion.K_h`](../src/core/runtime/runtime_config.cpp#L2158) | [Migrates to `diffusion.K_h`](../src/init/config/v0_to_v1_runtime.cpp#L102). |
| [`numerics.diffusion.K_v`](../src/core/runtime/runtime_config.cpp#L2171) | [Migrates to `diffusion.K_v`](../src/init/config/v0_to_v1_runtime.cpp#L108). |
| [`numerics.diffusion.K_v_surface`](../src/core/runtime/runtime_config.cpp#L2227) | Legacy-only spelling; inspect migration diagnostics. |
| [`numerics.diffusion.K_v_top`](../src/core/runtime/runtime_config.cpp#L2229) | Legacy-only spelling; inspect migration diagnostics. |
| [`numerics.diffusion.apply_to`](../src/core/runtime/runtime_config.cpp#L2150) | [Migrates to `diffusion.apply_to`](../src/init/config/v0_to_v1_runtime.cpp#L117). |
| [`numerics.diffusion.coefficient_source`](../src/core/runtime/runtime_config.cpp#L2206) | Legacy-only spelling; inspect migration diagnostics. |
| [`numerics.diffusion.dt_diffusion`](../src/core/runtime/runtime_config.cpp#L2184) | Legacy-only spelling; inspect migration diagnostics. |
| [`numerics.diffusion.implicit_dim`](../src/core/runtime/runtime_config.cpp#L2154) | Legacy-only spelling; inspect migration diagnostics. |
| [`numerics.diffusion.operator_order`](../src/core/runtime/runtime_config.cpp#L2135) | [Migrates to `diffusion.operator_order`](../src/init/config/v0_to_v1_runtime.cpp#L89). |
| [`numerics.diffusion.operator_type`](../src/core/runtime/runtime_config.cpp#L2146) | Legacy-only spelling; inspect migration diagnostics. |
| [`numerics.diffusion.scheme`](../src/core/runtime/runtime_config.cpp#L2120) | [Migrates to `diffusion.scheme`](../src/init/config/v0_to_v1_runtime.cpp#L67). |
| [`numerics.diffusion.use_variable_K`](../src/core/runtime/runtime_config.cpp#L2198) | Legacy-only spelling; inspect migration diagnostics. |
| [`numerics.time_stepping`](../src/core/runtime/runtime_config.cpp#L2254) | Legacy-only spelling; inspect migration diagnostics. |
| [`numerics.time_stepping.adaptive_dt`](../src/core/runtime/runtime_config.cpp#L2301) | [Migrates to `time.adaptive_dt`](../src/init/config/v0_to_v1_runtime.cpp#L27). |
| [`numerics.time_stepping.cfl_safety`](../src/core/runtime/runtime_config.cpp#L2287) | [Migrates to `time.cfl_safety`](../src/init/config/v0_to_v1_runtime.cpp#L33). |
| [`numerics.time_stepping.dt_max`](../src/core/runtime/runtime_config.cpp#L2318) | Legacy-only spelling; inspect migration diagnostics. |
| [`numerics.time_stepping.dt_min`](../src/core/runtime/runtime_config.cpp#L2305) | Legacy-only spelling; inspect migration diagnostics. |
| [`numerics.time_stepping.n_acoustic_substeps`](../src/core/runtime/runtime_config.cpp#L2266) | [Migrates to `time.acoustic_substeps`](../src/init/config/v0_to_v1_runtime.cpp#L21). |
| [`numerics.time_stepping.physics_splitting`](../src/core/runtime/runtime_config.cpp#L2283) | Legacy-only spelling; inspect migration diagnostics. |
| [`numerics.time_stepping.scheme`](../src/core/runtime/runtime_config.cpp#L2258) | Legacy-only spelling; inspect migration diagnostics. |
| [`numerics.time_stepping.split_acoustic`](../src/core/runtime/runtime_config.cpp#L2262) | Legacy-only spelling; inspect migration diagnostics. |
| [`output.async_io`](../src/core/output/output_config.cpp#L1728) | Same spelling exists; compare semantics and omission defaults. |
| [`output.estimate_disk_budget`](../src/core/output/output_config.cpp#L1754) | Legacy-only spelling; inspect migration diagnostics. |
| [`output.fields`](../src/core/output/output_config.cpp#L1544) | [Migrates to `output.preset`](../src/init/config/v0_to_v1_runtime.cpp#L385). |
| [`output.format`](../src/core/output/output_config.cpp#L1503) | Same spelling exists; compare semantics and omission defaults. |
| [`output.keep_last_frames`](../src/core/output/output_config.cpp#L1750) | Same spelling exists; compare semantics and omission defaults. |
| [`output.live_shm`](../src/core/runtime/runtime_config.cpp#L5145) | Same spelling exists; compare semantics and omission defaults. |
| [`output.live_shm_fields`](../src/core/runtime/runtime_config.cpp#L5149) | Same spelling exists; compare semantics and omission defaults. |
| [`output.live_shm_full_rate`](../src/core/output/output_config.cpp#L1731) | Same spelling exists; compare semantics and omission defaults. |
| [`output.live_shm_interval_s`](../src/core/output/output_config.cpp#L1729) | Same spelling exists; compare semantics and omission defaults. |
| [`output.maximum_retained_history_files`](../src/core/output/output_config.cpp#L1752) | Same spelling exists; compare semantics and omission defaults. |
| [`output.minimum_free_space_mib`](../src/core/output/output_config.cpp#L1751) | Same spelling exists; compare semantics and omission defaults. |
| [`output.netcdf_deflate_level`](../src/core/output/output_config.cpp#L1740) | Same spelling exists; compare semantics and omission defaults. |
| [`output.netcdf_shuffle`](../src/core/output/output_config.cpp#L1745) | Same spelling exists; compare semantics and omission defaults. |
| [`output.netcdf_target_chunk_kib`](../src/core/output/output_config.cpp#L1746) | Same spelling exists; compare semantics and omission defaults. |
| [`output.qualification_diagnostics.interval_s`](../src/core/output/output_config.cpp#L1715) | Same spelling exists; compare semantics and omission defaults. |
| [`output.qualification_diagnostics.preset`](../src/core/output/output_config.cpp#L1694) | Same spelling exists; compare semantics and omission defaults. |
| [`output.tiered_write_cadence`](../src/core/output/output_config.cpp#L1686) | Same spelling exists; compare semantics and omission defaults. |
| [`output.trajectories`](../src/core/output/output_config.cpp#L1733) | Same spelling exists; compare semantics and omission defaults. |
| [`output.trajectory_count`](../src/core/output/output_config.cpp#L1735) | Same spelling exists; compare semantics and omission defaults. |
| [`output.trajectory_max_samples_per_parcel`](../src/core/output/output_config.cpp#L1738) | Same spelling exists; compare semantics and omission defaults. |
| [`output.trajectory_sample_interval_s`](../src/core/output/output_config.cpp#L1736) | Same spelling exists; compare semantics and omission defaults. |
| [`output.trajectory_seed_time_s`](../src/core/output/output_config.cpp#L1734) | Same spelling exists; compare semantics and omission defaults. |
| [`output.write_cadence_medium_s`](../src/core/output/output_config.cpp#L1687) | Same spelling exists; compare semantics and omission defaults. |
| [`output.write_cadence_slow_s`](../src/core/output/output_config.cpp#L1688) | Same spelling exists; compare semantics and omission defaults. |
| [`output.write_every_s`](../src/core/runtime/runtime_config.cpp#L5130) | [Migrates to `time.output_interval`](../src/init/config/v0_to_v1_runtime.cpp#L349). |
| [`output.zfp_field_execution`](../src/core/output/output_config.cpp#L1655) | Same spelling exists; compare semantics and omission defaults. |
| [`output.zfp_field_scratch_mib`](../src/core/output/output_config.cpp#L1665) | Same spelling exists; compare semantics and omission defaults. |
| [`output.zfp_field_workers`](../src/core/output/output_config.cpp#L1663) | Same spelling exists; compare semantics and omission defaults. |
| [`output.zfp_float16_prefilter`](../src/core/output/output_config.cpp#L1653) | Same spelling exists; compare semantics and omission defaults. |
| [`output.zfp_keyframe_interval`](../src/core/output/output_config.cpp#L1651) | Same spelling exists; compare semantics and omission defaults. |
| [`output.zfp_loose_tolerance`](../src/core/output/output_config.cpp#L1676) | Same spelling exists; compare semantics and omission defaults. |
| [`output.zfp_mode`](../src/core/output/output_config.cpp#L1602) | Same spelling exists; compare semantics and omission defaults. |
| [`output.zfp_moderate_tolerance`](../src/core/output/output_config.cpp#L1675) | Same spelling exists; compare semantics and omission defaults. |
| [`output.zfp_per_field_tolerances`](../src/core/output/output_config.cpp#L1673) | Same spelling exists; compare semantics and omission defaults. |
| [`output.zfp_precision_bits`](../src/core/output/output_config.cpp#L1649) | Same spelling exists; compare semantics and omission defaults. |
| [`output.zfp_predictive_delta`](../src/core/output/output_config.cpp#L1654) | Same spelling exists; compare semantics and omission defaults. |
| [`output.zfp_rate_bps`](../src/core/output/output_config.cpp#L1650) | Same spelling exists; compare semantics and omission defaults. |
| [`output.zfp_sparse_threshold`](../src/core/output/output_config.cpp#L1652) | Same spelling exists; compare semantics and omission defaults. |
| [`output.zfp_tight_tolerance`](../src/core/output/output_config.cpp#L1674) | Same spelling exists; compare semantics and omission defaults. |
| [`output.zfp_tolerance`](../src/core/output/output_config.cpp#L1647) | Same spelling exists; compare semantics and omission defaults. |
| [`performance.report_every_steps`](../src/core/runtime/runtime_config.cpp#L3802) | [Migrates to `extensions.performance.report_every_steps`](../src/init/config/v0_to_v1_runtime.cpp#L697). |
| [`performance.timing`](../src/core/runtime/runtime_config.cpp#L3798) | [Migrates to `extensions.performance.timing`](../src/init/config/v0_to_v1_runtime.cpp#L684). |
| [`performance.warmup_steps`](../src/core/runtime/runtime_config.cpp#L3815) | [Migrates to `extensions.performance.warmup_steps`](../src/init/config/v0_to_v1_runtime.cpp#L711). |
| [`radiation.clear_sky_only`](../src/core/runtime/runtime_config.cpp#L1371) | Same spelling exists; compare semantics and omission defaults. |
| [`radiation.declination_deg`](../src/core/runtime/runtime_config.cpp#L1332) | Same spelling exists; compare semantics and omission defaults. |
| [`radiation.do_lw`](../src/core/runtime/runtime_config.cpp#L1336) | Same spelling exists; compare semantics and omission defaults. |
| [`radiation.do_sw`](../src/core/runtime/runtime_config.cpp#L1341) | Same spelling exists; compare semantics and omission defaults. |
| [`radiation.dt`](../src/core/runtime/runtime_config.cpp#L1346) | Legacy-only spelling; inspect migration diagnostics. |
| [`radiation.dt_radiation`](../src/core/runtime/runtime_config.cpp#L1358) | Legacy-only spelling; inspect migration diagnostics. |
| [`radiation.enable_diagnostics`](../src/core/runtime/runtime_config.cpp#L1385) | Same spelling exists; compare semantics and omission defaults. |
| [`radiation.latitude_deg`](../src/core/runtime/runtime_config.cpp#L1331) | Same spelling exists; compare semantics and omission defaults. |
| [`radiation.local_solar_time_at_start_h`](../src/core/runtime/runtime_config.cpp#L1333) | Same spelling exists; compare semantics and omission defaults. |
| [`radiation.n_lw`](../src/core/runtime/runtime_config.cpp#L1426) | Same spelling exists; compare semantics and omission defaults. |
| [`radiation.n_sw`](../src/core/runtime/runtime_config.cpp#L1438) | Same spelling exists; compare semantics and omission defaults. |
| [`radiation.rrtmg_ccl4_vmr`](../src/core/runtime/runtime_config.cpp#L1505) | Same spelling exists; compare semantics and omission defaults. |
| [`radiation.rrtmg_cfc11_vmr`](../src/core/runtime/runtime_config.cpp#L1499) | Same spelling exists; compare semantics and omission defaults. |
| [`radiation.rrtmg_cfc12_vmr`](../src/core/runtime/runtime_config.cpp#L1501) | Same spelling exists; compare semantics and omission defaults. |
| [`radiation.rrtmg_cfc22_vmr`](../src/core/runtime/runtime_config.cpp#L1503) | Same spelling exists; compare semantics and omission defaults. |
| [`radiation.rrtmg_ch4_vmr`](../src/core/runtime/runtime_config.cpp#L1494) | Same spelling exists; compare semantics and omission defaults. |
| [`radiation.rrtmg_co2_vmr`](../src/core/runtime/runtime_config.cpp#L1493) | Same spelling exists; compare semantics and omission defaults. |
| [`radiation.rrtmg_co_vmr`](../src/core/runtime/runtime_config.cpp#L1496) | Same spelling exists; compare semantics and omission defaults. |
| [`radiation.rrtmg_ice_effective_radius_um`](../src/core/runtime/runtime_config.cpp#L1511) | Same spelling exists; compare semantics and omission defaults. |
| [`radiation.rrtmg_liquid_effective_radius_um`](../src/core/runtime/runtime_config.cpp#L1507) | Same spelling exists; compare semantics and omission defaults. |
| [`radiation.rrtmg_n2_vmr`](../src/core/runtime/runtime_config.cpp#L1498) | Same spelling exists; compare semantics and omission defaults. |
| [`radiation.rrtmg_n2o_vmr`](../src/core/runtime/runtime_config.cpp#L1495) | Same spelling exists; compare semantics and omission defaults. |
| [`radiation.rrtmg_o2_vmr`](../src/core/runtime/runtime_config.cpp#L1497) | Same spelling exists; compare semantics and omission defaults. |
| [`radiation.rrtmg_ozone_vmr`](../src/core/runtime/runtime_config.cpp#L1492) | Same spelling exists; compare semantics and omission defaults. |
| [`radiation.scheme`](../src/core/runtime/runtime_config.cpp#L1275) | Same spelling exists; compare semantics and omission defaults. |
| [`radiation.simple_grey_lw_ice_absorption_m2kg`](../src/core/runtime/runtime_config.cpp#L1469) | Same spelling exists; compare semantics and omission defaults. |
| [`radiation.simple_grey_lw_liquid_absorption_m2kg`](../src/core/runtime/runtime_config.cpp#L1466) | Same spelling exists; compare semantics and omission defaults. |
| [`radiation.simple_grey_lw_vapor_absorption_m2kg`](../src/core/runtime/runtime_config.cpp#L1463) | Same spelling exists; compare semantics and omission defaults. |
| [`radiation.simple_grey_optical_model`](../src/core/runtime/runtime_config.cpp#L1375) | Same spelling exists; compare semantics and omission defaults. |
| [`radiation.simple_grey_sw_ice_absorption_m2kg`](../src/core/runtime/runtime_config.cpp#L1478) | Same spelling exists; compare semantics and omission defaults. |
| [`radiation.simple_grey_sw_liquid_absorption_m2kg`](../src/core/runtime/runtime_config.cpp#L1475) | Same spelling exists; compare semantics and omission defaults. |
| [`radiation.simple_grey_sw_vapor_absorption_m2kg`](../src/core/runtime/runtime_config.cpp#L1472) | Same spelling exists; compare semantics and omission defaults. |
| [`radiation.solar_forcing`](../src/core/runtime/runtime_config.cpp#L1313) | Same spelling exists; compare semantics and omission defaults. |
| [`radiation.subcycle_count`](../src/core/runtime/runtime_config.cpp#L1389) | Same spelling exists; compare semantics and omission defaults. |
| [`radiation.tau_lw_ref`](../src/core/runtime/runtime_config.cpp#L1400) | Same spelling exists; compare semantics and omission defaults. |
| [`radiation.tau_sw_ref`](../src/core/runtime/runtime_config.cpp#L1413) | Same spelling exists; compare semantics and omission defaults. |
| [`radiation.terrain_sw_absolute_residual_tolerance_w`](../src/core/runtime/runtime_config.cpp#L1561) | Same spelling exists; compare semantics and omission defaults. |
| [`radiation.terrain_sw_azimuth_nodes`](../src/core/runtime/runtime_config.cpp#L1541) | Same spelling exists; compare semantics and omission defaults. |
| [`radiation.terrain_sw_cosine_nodes`](../src/core/runtime/runtime_config.cpp#L1536) | Same spelling exists; compare semantics and omission defaults. |
| [`radiation.terrain_sw_maximum_bounces`](../src/core/runtime/runtime_config.cpp#L1546) | Same spelling exists; compare semantics and omission defaults. |
| [`radiation.terrain_sw_relative_residual_tolerance`](../src/core/runtime/runtime_config.cpp#L1549) | Same spelling exists; compare semantics and omission defaults. |
| [`radiation.use_subcycling`](../src/core/runtime/runtime_config.cpp#L1381) | Same spelling exists; compare semantics and omission defaults. |
| [`schema`](../src/core/runtime/runtime_config.cpp#L3606) | Same spelling exists; compare semantics and omission defaults. |
| [`surface.Tsfc`](../src/core/runtime/runtime_config.cpp#L1644) | Legacy-only spelling; inspect migration diagnostics. |
| [`surface.albedo`](../src/core/runtime/runtime_config.cpp#L1605) | Same spelling exists; compare semantics and omission defaults. |
| [`surface.emissivity`](../src/core/runtime/runtime_config.cpp#L1618) | Same spelling exists; compare semantics and omission defaults. |
| [`surface.qsfc`](../src/core/runtime/runtime_config.cpp#L1658) | [Migrates to `surface.qsfc_kgkg`](../src/init/config/v0_to_v1_physics.cpp#L303). |
| [`surface.qv_sfc`](../src/core/runtime/runtime_config.cpp#L1671) | Legacy-only spelling; inspect migration diagnostics. |
| [`surface.scheme`](../src/core/runtime/runtime_config.cpp#L5378) | Same spelling exists; compare semantics and omission defaults. |
| [`surface.tsfc`](../src/core/runtime/runtime_config.cpp#L1631) | [Migrates to `surface.tsk_k`](../src/init/config/v0_to_v1_physics.cpp#L295). |
| [`surface.z0h`](../src/core/runtime/runtime_config.cpp#L1592) | [Migrates to `surface.z0h_m`](../src/init/config/v0_to_v1_physics.cpp#L313). |
| [`surface.z0m`](../src/core/runtime/runtime_config.cpp#L1579) | [Migrates to `surface.z0m_m`](../src/init/config/v0_to_v1_physics.cpp#L308). |
| [`terrain.bell.a`](../src/core/runtime/runtime_config.cpp#L3004) | Same spelling exists; compare semantics and omission defaults. |
| [`terrain.bell.axisymmetric`](../src/core/runtime/runtime_config.cpp#L3016) | Same spelling exists; compare semantics and omission defaults. |
| [`terrain.bell.h0`](../src/core/runtime/runtime_config.cpp#L2992) | Same spelling exists; compare semantics and omission defaults. |
| [`terrain.compute_derivatives`](../src/core/runtime/runtime_config.cpp#L2971) | Same spelling exists; compare semantics and omission defaults. |
| [`terrain.compute_metrics`](../src/core/runtime/runtime_config.cpp#L2975) | Same spelling exists; compare semantics and omission defaults. |
| [`terrain.coord_id`](../src/core/runtime/runtime_config.cpp#L2935) | [Migrates to `terrain.coordinate`](../src/init/config/v0_to_v1_runtime.cpp#L272). |
| [`terrain.feature_center_earth_x_m`](../src/core/runtime/runtime_config.cpp#L2965) | Same spelling exists; compare semantics and omission defaults. |
| [`terrain.feature_center_earth_y_m`](../src/core/runtime/runtime_config.cpp#L2968) | Same spelling exists; compare semantics and omission defaults. |
| [`terrain.horizontal_placement`](../src/core/runtime/runtime_config.cpp#L2949) | Same spelling exists; compare semantics and omission defaults. |
| [`terrain.schar.H`](../src/core/runtime/runtime_config.cpp#L3020) | Same spelling exists; compare semantics and omission defaults. |
| [`terrain.schar.a`](../src/core/runtime/runtime_config.cpp#L3032) | Same spelling exists; compare semantics and omission defaults. |
| [`terrain.schar.ell`](../src/core/runtime/runtime_config.cpp#L3044) | Same spelling exists; compare semantics and omission defaults. |
| [`terrain.scheme`](../src/core/runtime/runtime_config.cpp#L2873) | Same spelling exists; compare semantics and omission defaults. |
| [`terrain.smoothing_decay`](../src/core/runtime/runtime_config.cpp#L2979) | Same spelling exists; compare semantics and omission defaults. |
| [`terrain.ztop`](../src/core/runtime/runtime_config.cpp#L2923) | [Migrates to `grid.ztop`](../src/init/config/v0_to_v1_runtime.cpp#L254). |
| [`trigger.bubble.center_x_km`](../src/core/runtime/runtime_config.cpp#L2601) | [Unit conversion to `initial_conditions.trigger.bubble.center_x_m`](../src/init/config/v0_to_v1_initialization.cpp#L382). |
| [`trigger.bubble.center_y_km`](../src/core/runtime/runtime_config.cpp#L2614) | [Unit conversion to `initial_conditions.trigger.bubble.center_y_m`](../src/init/config/v0_to_v1_initialization.cpp#L387). |
| [`trigger.bubble.center_z_km`](../src/core/runtime/runtime_config.cpp#L2627) | [Unit conversion to `initial_conditions.trigger.bubble.center_z_m`](../src/init/config/v0_to_v1_initialization.cpp#L392). |
| [`trigger.bubble.cloud_water_kgkg`](../src/core/runtime/runtime_config.cpp#L2767) | [Migrates to `initial_conditions.trigger.bubble.cloud_water_kgkg`](../src/init/config/v0_to_v1_initialization.cpp#L370). |
| [`trigger.bubble.dtheta_k`](../src/core/runtime/runtime_config.cpp#L2755) | [Migrates to `initial_conditions.trigger.bubble.dtheta_k`](../src/init/config/v0_to_v1_initialization.cpp#L365). |
| [`trigger.bubble.radius_km`](../src/core/runtime/runtime_config.cpp#L2640) | [Migrates to `initial_conditions.trigger.bubble.radius_km`](../src/init/config/v0_to_v1_initialization.cpp#L377). |
| [`trigger.type`](../src/core/runtime/runtime_config.cpp#L2659) | [Migrates to `initial_conditions.trigger.type`](../src/init/config/v0_to_v1_initialization.cpp#L351). |
| [`trigger.vortex_seed.center_x_km`](../src/core/runtime/runtime_config.cpp#L2715) | [Unit conversion to `initial_conditions.trigger.vortex_seed.center_x_m`](../src/init/config/v0_to_v1_initialization.cpp#L414). |
| [`trigger.vortex_seed.center_y_km`](../src/core/runtime/runtime_config.cpp#L2728) | [Unit conversion to `initial_conditions.trigger.vortex_seed.center_y_m`](../src/init/config/v0_to_v1_initialization.cpp#L420). |
| [`trigger.vortex_seed.center_z_km`](../src/core/runtime/runtime_config.cpp#L2741) | [Unit conversion to `initial_conditions.trigger.vortex_seed.center_z_m`](../src/init/config/v0_to_v1_initialization.cpp#L426). |
| [`trigger.vortex_seed.r_max_m`](../src/core/runtime/runtime_config.cpp#L2676) | [Migrates to `initial_conditions.trigger.vortex_seed.r_max_m`](../src/init/config/v0_to_v1_initialization.cpp#L399). |
| [`trigger.vortex_seed.v_max_ms`](../src/core/runtime/runtime_config.cpp#L2689) | [Migrates to `initial_conditions.trigger.vortex_seed.v_max_ms`](../src/init/config/v0_to_v1_initialization.cpp#L404). |
| [`trigger.vortex_seed.z_top_m`](../src/core/runtime/runtime_config.cpp#L2702) | [Migrates to `initial_conditions.trigger.vortex_seed.z_top_m`](../src/init/config/v0_to_v1_initialization.cpp#L409). |
| [`turbulence.Cs`](../src/core/runtime/runtime_config.cpp#L1942) | Same spelling exists; compare semantics and omission defaults. |
| [`turbulence.K_max`](../src/core/runtime/runtime_config.cpp#L1998) | Legacy-only spelling; inspect migration diagnostics. |
| [`turbulence.Pr_t`](../src/core/runtime/runtime_config.cpp#L1956) | Same spelling exists; compare semantics and omission defaults. |
| [`turbulence.Sc_t`](../src/core/runtime/runtime_config.cpp#L1970) | Same spelling exists; compare semantics and omission defaults. |
| [`turbulence.dt`](../src/core/runtime/runtime_config.cpp#L1939) | Legacy-only spelling; inspect migration diagnostics. |
| [`turbulence.dt_sgs`](../src/core/runtime/runtime_config.cpp#L1940) | Legacy-only spelling; inspect migration diagnostics. |
| [`turbulence.dynamic_Cs`](../src/core/runtime/runtime_config.cpp#L2065) | Legacy-only spelling; inspect migration diagnostics. |
| [`turbulence.dynamic_cs`](../src/core/runtime/runtime_config.cpp#L2069) | Legacy-only spelling; inspect migration diagnostics. |
| [`turbulence.enable_diagnostics`](../src/core/runtime/runtime_config.cpp#L2082) | Legacy-only spelling; inspect migration diagnostics. |
| [`turbulence.enable_moist_buoyancy`](../src/core/runtime/runtime_config.cpp#L2078) | Legacy-only spelling; inspect migration diagnostics. |
| [`turbulence.filter_width`](../src/core/runtime/runtime_config.cpp#L2030) | Same spelling exists; compare semantics and omission defaults. |
| [`turbulence.mode`](../src/core/runtime/runtime_config.cpp#L1907) | Same spelling exists; compare semantics and omission defaults. |
| [`turbulence.nu_t_max`](../src/core/runtime/runtime_config.cpp#L1984) | Legacy-only spelling; inspect migration diagnostics. |
| [`turbulence.scheme`](../src/core/runtime/runtime_config.cpp#L1896) | Same spelling exists; compare semantics and omission defaults. |
| [`turbulence.smagorinsky.dynamic_procedure`](../src/core/runtime/runtime_config.cpp#L2073) | Legacy-only spelling; inspect migration diagnostics. |
| [`turbulence.smagorinsky.filter_width`](../src/core/runtime/runtime_config.cpp#L2031) | Legacy-only spelling; inspect migration diagnostics. |
| [`turbulence.smagorinsky.stability_correction`](../src/core/runtime/runtime_config.cpp#L2063) | Legacy-only spelling; inspect migration diagnostics. |
| [`turbulence.stability_correction`](../src/core/runtime/runtime_config.cpp#L2062) | Same spelling exists; compare semantics and omission defaults. |
| [`turbulence.user_filter_width_m`](../src/core/runtime/runtime_config.cpp#L2032) | Same spelling exists; compare semantics and omission defaults. |
| [`validation.guard_fail_on`](../src/core/runtime/runtime_config.cpp#L4249) | Legacy-only spelling; inspect migration diagnostics. |
| [`validation.guard_mode`](../src/core/runtime/runtime_config.cpp#L4235) | [Migrates to `extensions.validation.guard_mode`](../src/init/config/v0_to_v1_runtime.cpp#L650). |
| [`validation.guard_scope`](../src/core/runtime/runtime_config.cpp#L4262) | Legacy-only spelling; inspect migration diagnostics. |
| [`validation.report_path`](../src/core/runtime/runtime_config.cpp#L4275) | Legacy-only spelling; inspect migration diagnostics. |

</details>

Legacy dynamic families are also read:

| Pattern | Meaning |
| --- | --- |
| [`validation.field_overrides.<field>.min`](../src/core/runtime/runtime_config.cpp#L352) | Finite lower field-contract bound; field identifier is lowercased. Validation policy applies this research/compatibility override. |
| [`validation.field_overrides.<field>.max`](../src/core/runtime/runtime_config.cpp#L352) | Finite upper field-contract bound; field identifier is lowercased. |
| [`chaos.sigma_ic.<target>`](../src/core/runtime/runtime_config.cpp#L3240) | Legacy dynamic amplitude map, parsed as finite doubles; later chaos validation owns target/domain legality. |
| [`chaos.alpha_tend.<target>`](../src/core/runtime/runtime_config.cpp#L3253) | Legacy dynamic tendency-amplitude map, parsed as finite doubles; later chaos validation owns legality. |
| [`output.zfp_field_tolerance.<field>`](../src/core/output/output_config.cpp#L1619) | Positive finite absolute tolerance; field name must be in the output-field registry. |


## Keeping this list current

The [strict loader](../src/init/config/v1_loader.cpp),
[typed configuration](../include/init/config/config_v1.hpp),
[microphysics defaults](../include/microphysics/config.hpp), enum parsers and
ordinary runtime admission are authoritative. Linked key names lead to their
loader consumers. The reviewed inventory and document generator live under
`scripts/dev/docs/`; `make check-docs` checks their source identities and the
rendered reference. A source change requires reviewing affected entries rather
than silently carrying old descriptions forward.

The optional native-default check constructs `ConfigV1` and reads its fields;
it runs no atmosphere and loads no scientific data:

```sh
python3 scripts/dev/docs/generate_namelist.py --check --verify-defaults
```

Constructor-default verification establishes the omission-value inventory,
not correctness of the physical assumptions or qualification of a combination.
