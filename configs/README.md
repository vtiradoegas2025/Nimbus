# Configs

YAML presets for `bin/tornado_sim`. Pick one that matches what you want
to do, then optionally apply a variation snippet from this README.

```sh
./bin/tornado_sim --headless --config=configs/<dir>/<name>.yaml
```

The `--headless` flag is required for the simulation to actually run;
without it the binary exits after init. The space-separated form
`--config <path>` also works.

---

## Pick a preset

| I want to ...                          | Start from                           |
|----------------------------------------|--------------------------------------|
| Try the simulator on a laptop          | `student/student.yaml` (cylindrical) |
| Try the Cartesian backend              | `student/student_cartesian.yaml`     |
| Watch a thermal bubble rise            | `teaching/thermal_bubble.yaml`       |
| See a supercell form in 30 minutes     | `teaching/supercell_30min.yaml`      |
| Watch a tornado spin up over 2-4 hours | `teaching/tornado_genesis.yaml`      |
| Run serious science on a workstation   | `simulation/research.yaml`           |
| Push hard on a beefy machine (32-96 GB)| `simulation/production.yaml`         |
| Time the kernel paths (no I/O)         | `benchmark/benchmark.yaml`           |
| Measure ZFP compression ratios         | `benchmark/zfp_benchmark.yaml`       |

Presets in `configs/test/` are dev-side test fixtures driven by
`make test-cgrid-integration` and the module-README smoke commands;
they are not user presets and assume specific test contexts.

---

## The three big knobs

Every preset can be redirected onto a different grid + dynamics scheme
by editing two or three lines. The runtime supports the combinations
in the table below; pick the row you want and set the YAML keys
accordingly.

| Coordinate system | Staggering   | Dynamics scheme                     | Good for                                          |
|-------------------|--------------|-------------------------------------|---------------------------------------------------|
| `cartesian`       | `collocated` | `cartesian`                         | Idealized 3D box experiments, terrain studies     |
| `cylindrical`     | `collocated` | `supercell` / `tornado`             | Default cylindrical path; current production runs |
| `cylindrical`     | `c_grid`     | `supercell_cgrid` / `tornado_cgrid` | Phase C path: better hydrostatic balance, mass conservation |

YAML keys to set:

```yaml
coordinate_system: cylindrical    # or cartesian
grid:
  staggering: collocated          # or c_grid
dynamics:
  scheme: supercell               # see table; must match coordinate + stagger
```

`staggering:` defaults to `collocated` if omitted. `coordinate_system:`
defaults to `cylindrical`.

`tornado` and `tornado_cgrid` are axisymmetric tornado-vortex schemes
(no horizontal hodograph contribution to the dynamics). `supercell`
and `supercell_cgrid` are full 3D cylindrical supercell schemes.

---

## Variation recipes

Drop these snippets into any preset to change its behavior. Each
overrides the corresponding block in the YAML it's pasted into.

### Run on the C-grid

```yaml
grid:
  staggering: c_grid
dynamics:
  scheme: supercell_cgrid   # was: supercell
  # scheme: tornado_cgrid   # if the original was tornado
```

### Switch to the GPU compute backend

```yaml
numerics:
  compute:
    backend: vulkan         # falls back to CPU if Vulkan is unavailable
    allow_fallback: true
```

### Change microphysics

```yaml
microphysics:
  scheme: kessler           # warm rain only (cheapest)
  # scheme: lin             # ice + warm rain
  # scheme: thompson        # full mixed-phase
  # scheme: milbrandt       # double-moment
```

### Low-Precipitation (LP) supercell environment

LP storms have visible updrafts with sparse precipitation (Bluestein &
Parks 1983). Drier midlevel, weaker low-level shear.

```yaml
environment:
  cape_target_jkg: 3000
  shear_0_6km_ms: 35
  midlevel_drying: true
  hodograph:
    type: wk_param
    u_sfc_ms: 4
    v_sfc_ms: 1
    u_1km_ms: 14
    v_1km_ms: 6
    u_6km_ms: 28
    v_6km_ms: 22
microphysics:
  scheme: kessler
  hail_graupel_enabled: false
```

### High-Precipitation (HP) supercell environment

HP storms are precipitation-shrouded with heavy rain wrapping the
mesocyclone. Moister column, stronger shear, hail / graupel enabled.

```yaml
environment:
  cape_target_jkg: 2500
  shear_0_6km_ms: 40
  sfc_qv_kgkg: 0.016
  moist_column: true
  hodograph:
    type: wk_param
    u_sfc_ms: 6
    v_sfc_ms: 2
    u_1km_ms: 16
    v_1km_ms: 10
    u_6km_ms: 32
    v_6km_ms: 26
microphysics:
  scheme: kessler
  hail_graupel_enabled: true
```

### Elevated convection (above a low-level inversion)

Surface parcels are capped, so lifting must originate from above the
inversion. Mirrors the nocturnal severe-weather pattern after surface
cooling stabilises the boundary layer.

```yaml
environment:
  cape_target_jkg: 1500
  inversion_base_m: 1000
```

### Disable the trigger bubble (for hydrostatic / equilibrium tests)

```yaml
trigger:
  bubble:
    dtheta_k: 0.0
```

### Zero the wind (for static atmosphere tests)

```yaml
environment:
  hodograph:
    type: wk_param
    u_sfc_ms: 0
    v_sfc_ms: 0
    u_1km_ms: 0
    v_1km_ms: 0
    u_6km_ms: 0
    v_6km_ms: 0
```

### High-resolution classic supercell

Finer than `research.yaml`; resolves updraft cores cleanly. Costly.

```yaml
grid:
  nx: 256
  ny: 128
  nz: 128
  dx: 1000.0
  dy: 1000.0
  dz: 100.0
  dt: 0.1
```

### Live SHM streaming for the viewer

```yaml
output:
  live_shm: true
  live_shm_fields: w,reflectivity_dbz,vorticity_z
```

---

## Directory layout

- `configs/student/`    — small-grid presets that fit on 8 GB laptops
- `configs/teaching/`   — guided exercises with learning objectives + expected runtimes
- `configs/simulation/` — research / production workstation presets
- `configs/benchmark/`  — performance-focused, no or minimal output
- `configs/test/`       — dev-side test fixtures (not user-facing)

Each user-facing YAML opens with a scenario header explaining what
it's for and what to flip for variants. If the variation you want
isn't in this README's recipes, copy the preset and tweak directly.
