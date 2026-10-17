# Internal configuration fixtures

This tree contains configuration inputs required by Nimbus automated tests,
benchmarks, and preserved qualification workflows. It is deliberately outside
the public `configs/` catalog so Studio users and external users do not mistake
specialized evidence fixtures for general simulation presets.

Keep fixture paths stable unless every executable reference is updated. These
files may encode narrow qualification contracts and should not be presented as
resolution recommendations.

`make test-ready-admission-matrix` discovers every top-level fixture whose
execution mode is `qualified` and requires ordinary `--validate-only`
admission. Every top-level fixture must declare either `qualified` or
`exploratory`; silent mode defaults are rejected. A top-level `*_ready.yaml`
file must use qualified mode. Scientific scope and neighboring-profile
rejection remain owned by each profile's dedicated bounded gate.

`qualification/moving_nested_supercell_{cpu,vulkan,metal}_v1.yaml` are
ordinary-binary profiles for the exact moving-nest CPU, Vulkan, and Metal
lanes. They are evidence-bound admission fixtures, not templates for broader
moving-nest experiments.

`qualification/domain_recycling_kessler_ready.yaml` is the exact Ready
flat D5 recycling profile shared by the bounded CPU, Vulkan, and Metal
transaction/restart gate.

`qualification/tracked_window_kessler_ready.yaml` is the exact zero-frame CPU
tracked-window form of the same flat D5 owner set. Its bounded gate requires a
real tracker-requested transaction, retiring-swath hashes, trajectory
continuity, one/four-worker identity, exact restart, and fail-closed neighboring
profiles.

`qualification/agrid_flat_cpu_core_ready.yaml` is the exact ordinary-binary
Ready profile for the flat dry Cartesian A-grid CPU core. It covers a rigid-
boundary warm bubble with no active microphysics, turbulence, PBL, radiation,
terrain, chaos, or diffusion coefficient. The bounded gate checks
one/four-worker bitwise field and ledger identity, exact midpoint restart,
finite and positive-state invariants, and fail-closed neighboring profiles.
Long campaigns, external review, and other A-grid combinations retain their
own admission and evidence requirements.

`qualification/agrid_milbrandt_rrtmg_ready.yaml` is the active collocated CPU
Milbrandt/RRTMG representative. Its gate requires a distinct mixed-phase
response versus `none`, all 15 public fields, six ledgers, byte-identical
one/four-worker private-state checkpoints, exact continuation, and rejection
of native-backend and still-C-grid-only Thompson neighbors.

`qualification/agrid_lin_rrtmg_ready.yaml` is the corresponding active
single-moment mixed-phase collocated CPU representative. Lin uses the shared
physical-cell domain and its `k_begin` fallout surface; its gate requires
distinct hydrometeor/thermal response, exact one/four-worker fields,
checkpoints, six ledgers and continuation, plus native/Thompson rejection.

`qualification/agrid_thompson_rrtmg_ready.yaml` closes the aerosol-aware
collocated CPU family. Its bounded gate covers five private number/aerosol
moments, physical-domain size-sorting sedimentation and fallout, distinct
hydrometeor/thermal response, exact one/four-worker fields, private-state
checkpoints, six ledgers and continuation, plus native/terrain rejection.

`qualification/agrid_thompson_rrtmg_implicit_ready.yaml` is the strong full
backward-Euler A-grid representative. It keeps Thompson/RRTMG active while
`K_h=K_v=4000 m²/s` diffuses scalars and momentum; the gate requires a
distinct explicit control, nonzero work, exact threading, private-state
checkpoints, six ledgers, continuation, and native/terrain rejection.

`qualification/agrid_thompson_rrtmg_implicit_weno_ready.yaml` selects the
collocated WENO5 reconstruction on the same maximal stack. Its gate requires a
distinct TVD control plus exact one/four-worker fields, private-state
checkpoints, six ledgers, continuation, and native/terrain rejection.

`qualification/agrid_thompson_rrtmg_implicit_weno_{ssprk3,rk4}_ready.yaml`
exercise both unsplit RK tableaux on the same maximal collocated stack. Their
gate requires distinct split-explicit and mutual endpoints, exact
one/four-worker fields, Thompson private-state checkpoints, six ledgers,
continuation, and native/terrain rejection.

`qualification/agrid_full_stochastic_ready.yaml` composes RK4, WENO5,
Thompson, horizontal Smagorinsky, MYNN/slab, RRTMG, full implicit diffusion,
beta-plane Coriolis, and all-process stochastic forcing on flat collocated CPU
storage. Its gate
requires non-inert initial-condition, microphysics, conserved PBL, and
conservative SGS controls, distinct f-plane/beta-plane response, pointwise
work-free Coriolis, ensemble-member diversity, exact one/four-worker
fields, checkpoints and six ledgers, exact continuation, and rejection of
unsupported native/terrain/target neighbors.

`qualification/constant_velocity_agrid_ready.yaml` is the maximal periodic flat
CPU A-grid constant-translation representative. It composes f-plane Coriolis,
WENO5, Lin, horizontal Smagorinsky, cloudy RRTMG, and full implicit
scalar/momentum diffusion. The qualification also retains 20-second dry and
Kessler/simple-grey corners, bounds FP32 stationary/translated Galilean
differences, requires exact one/four-worker fields, frame/origin provenance,
checkpoints and six-ledger continuation, accepts finite velocity variants, and
adds Thompson/Milbrandt private-state and prognostic-TKE variants. It rejects
accelerator, evolving stochastic, and terrain neighbors. F-plane and
Earth-referenced beta-plane branches are distinct and exact; all 29 supported
lateral/vertical boundary combinations pass in stationary and moving form,
while radiative and half-periodic selections reject. Bounded moving SSPRK3/RK4 branches use the runtime-v50 combined
unsplit commit owner and require nonzero `1e-12`-closed completed-step ledgers.
Separate moving slab, YSU, and MYNN branches prove non-inert
Earth-wind-relative surface/PBL response, while one-time initial-condition
chaos retains Galilean identity. Moving boundary-layer and full-stochastic
branches reconstruct their Earth lattices continuously at subcell frame
offsets, remain non-inert, and preserve exact threading, portable state,
checkpoints, complete ledgers, and restart.

`qualification/cgrid_stationary_bell_pbl_implicit_ready.yaml` and
`qualification/cgrid_translating_schar_pbl_implicit_ready.yaml` are the exact
ordinary-binary mapped-terrain CPU Ready profiles. Both use slab surface/PBL
and active implicit scalar-plus-momentum diffusion; the first is stationary Bell and the
second is grid-attached constant-velocity Schär. Their bounded gate checks
active evolution, one/four-worker bitwise identity, exact midpoint restart,
and fail-closed neighboring profiles. They are evidence fixtures, not general
terrain presets or campaign-validation claims.

`qualification/cgrid_stationary_bell_{vulkan,metal}_complete_ready.yaml` are
the exact dry stationary Bell fallback-off native mapped-core Ready profiles.
Their bounded physical gates require every mapped stage, atomic nine-field and
report publication, truthful backend-specific submission/kernel provenance,
one/four-worker identity, complete ledgers, exact midpoint continuation, and
fail-closed neighboring profiles. The final nonlinear EOS projection is
explicitly host FP64; active physics and other mapped device profiles do not
inherit this status.

`qualification/stationary_nested_cpu_two_way_ready.yaml` is the exact dry
stationary flat 2:1 Cartesian C-grid CPU two-way-feedback Ready profile. Its
bounded gate requires nonzero local reflux, all five currency closures,
one/four-worker parent, child, and report identity, synchronized midpoint
continuation, failure atomicity, and fail-closed neighboring profiles. Active
physics, movement, terrain, other ratios, and accelerators remain separate.

`qualification/earth_fixed_terrain_recycling_ready.yaml` is the exact CPU
tracked-window Earth-fixed Bell terrain-recycling Ready profile. It couples
Kessler, slab PBL/surface, simple-grey longwave, and evolving full-stochastic
chaos. Its gate requires a real terrain-aware D5 movement, destination physical-
height leading columns, atomic geometry/physics/frame/swath publication,
byte-identical one/four-worker physical fields and six ledgers, independent
owner recomposition, runtime-v50 restart identity, and fail-closed neighbors.
Other terrain shapes, combinations, backends, and nested-grid recycling do not
inherit this status.

`qualification/agrid_stationary_bell_cpu_ready.yaml` is the exact stationary
grid-attached Bell mapped Cartesian A-grid CPU Ready profile. Its real split-
explicit path owns mapped slow/fast pressure and continuity equations, retained
contravariant fluxes, scalar and completed-step ledgers, terrain-tangent rigid
boundaries, and FP64 EOS. Direct and ordinary gates require literal equations,
conservation, invalid/late-failure atomicity, one/four-worker fields and six
ledgers, independent owner recomposition, exact restart suffixes, and
fail-closed neighboring profiles. The permanent gate also constructs the exact
`7/-3 m/s` constant-velocity companion with matching Earth-relative hodograph,
proves bounded Galilean endpoints, exact moving-frame ledgers, threading,
checkpoints, and restart. Accelerators, Earth-fixed, nested, and active-physics
mapped A-grid combinations remain separate.

`qualification/cgrid_thompson_cpu_ready.yaml` is the exact flat periodic CPU
Thompson-Eidhammer/YSU/clear-sky-RRTMG Ready profile. It requires the pinned
packaged activation, cold-process, and collection identities and exercises
nonzero warm/cold processes, aerosol/private moments, CFL-substepped
sedimentation and fallout. Its gate proves one/four-worker cold-species fields,
conservation/activity, midpoint continuation, packaged hash and missing/corrupt
failure behavior, and fail-closed neighboring profiles. Accelerators and other
Thompson physics/geometry combinations do not inherit this status; archived
legacy Lin-class accelerator identities remain explicitly orphaned.

`qualification/nested_grid_recycling_ready.yaml` is the exact flat one-way CPU
nested-grid D5 recycling Ready profile. Parent movement rebases child parent-
cell placement while preserving the child Earth origin, clocks, fields,
runtime modules, five feedback arrays/registers, ledgers, and scheme identity
through one atomic owner. Its gate proves a real `[4,0]` parent move,
byte-identical one/four-worker parent/child fields, nested section v8 and five
ledgers, exact restart, owner provenance, v6 defaults, and fail-closed nearby
profiles. Two-way active-moist and active-terrain nested recycling remain
separate.

`qualification/stationary_nested_active_terrain_cpu_two_way_ready.yaml` is the
exact non-flat stationary Bell mapped two-way CPU nesting Ready profile. It
owns halo-inclusive parent/child geometry, J/Jx/Jy-aware initialization and
relaxation, scalar and native-dual momentum transfer/reflux, terrain-tangent
boundaries, endpoint storage, and checkpoint section v8. Its gate requires
five nonzero mapped currency closures, byte-identical one/four-worker root and
child fields plus six ledgers, independent owner recomposition, exact midpoint
continuation, non-flat physical-height output, and four fail-closed neighbors.
Other terrain shapes, active physics, movement, accelerators, and nested terrain
combinations remain separate.

`qualification/cgrid_ysu_theta_top_ready.yaml` and
`qualification/cgrid_mynn_tke_top_ready.yaml` are the exact CPU profiles for
potential-temperature-increase and prognostic-TKE PBL-top diagnosis. Their
bounded gate checks the typed public selectors, analytical threshold routing,
one/four-worker field and ledger identity, radar-free presentation output,
exact midpoint restart, checkpoint configuration identity, and fail-closed
neighboring profiles.
