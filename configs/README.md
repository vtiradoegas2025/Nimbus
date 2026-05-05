# Configs

YAML configuration presets for `bin/tornado_sim`. Pick a preset that
matches what you want to do, then optionally tweak a knob or two.

```sh
./bin/tornado_sim --headless --config=configs/<dir>/<name>.yaml
```

The `--headless` flag is required for the simulation to run; without it
the binary exits after init. The `--config <path>` form (with a space)
also works.

---

## Pick a preset by scenario

| I want to ...                          | Start from                                          |
|----------------------------------------|-----------------------------------------------------|
| Try the simulator on a laptop          | `student/student.yaml`                              |
| Watch a thermal bubble rise            | `teaching/thermal_bubble.yaml`                      |
| See a supercell form in 30 min         | `teaching/supercell_30min.yaml`                     |
| Watch a tornado spin up (2-4 hr)       | `teaching/tornado_genesis.yaml`                     |
| Run serious science on a workstation   | `simulation/research.yaml`                          |
| Push hard on a beefy machine           | `simulation/production.yaml`                        |
| Compare a low-precip vs high-precip storm | `simulation/lp.yaml` vs `simulation/hp.yaml`     |
| Test a physical-numerics-rich setup    | `simulation/physical_supercell.yaml` (Thompson microphys, WENO5 advection, RK3 timestepping) |
| Try elevated convection (above an inversion) | `simulation/elevated.yaml`                    |
| Use the GPU compute backend            | `simulation/lp_vulkan.yaml` (mirror of `lp.yaml` + Vulkan) |
| Time the kernel paths (no I/O)         | `benchmark/benchmark.yaml`                          |
| Measure ZFP compression ratios         | `benchmark/zfp_benchmark.yaml`                      |

Presets in `configs/test/` are dev-side test fixtures driven by
`make test-cgrid-integration` and the unit test binaries -- they are
not user presets and the YAMLs there assume specific test contexts.

---

## The three big knobs

Every preset can be redirected onto a different grid + dynamics scheme
by editing two or three lines. The runtime supports the combinations
in the table below; pick the row you want and set the YAML keys
accordingly.

| Coordinate system | Staggering   | Dynamics scheme              | What it's good for                                |
|-------------------|--------------|------------------------------|---------------------------------------------------|
| `cartesian`       | `collocated` | `cartesian`                  | Idealized 3D box experiments, terrain studies     |
| `cylindrical`     | `collocated` | `supercell` / `tornado`      | Default cylindrical path; production runs         |
| `cylindrical`     | `c_grid`     | `supercell_cgrid` / `tornado_cgrid` | Phase C path: better hydrostatic balance, conservation |

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

## Common variations

**Run an existing preset on the C-grid.** Add the staggering line and
swap the scheme name; everything else (hodograph, trigger, output) stays
the same.

```yaml
# In any cylindrical preset:
grid:
  staggering: c_grid
dynamics:
  scheme: supercell_cgrid   # was: supercell
```

**Disable the trigger bubble** (for hydrostatic / equilibrium tests):

```yaml
trigger:
  bubble:
    dtheta_k: 0.0   # no perturbation
```

**Zero the wind** (for static atmosphere tests):

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

**Change the microphysics scheme:**

```yaml
microphysics:
  scheme: kessler   # warm rain only (cheapest)
  # scheme: lin       # ice + warm rain
  # scheme: thompson  # full mixed-phase
  # scheme: milbrandt # double-moment
```

**Pick GPU vs CPU compute backend:**

```yaml
numerics:
  compute:
    backend: vulkan      # GPU; falls back to CPU if Vulkan unavailable
    # backend: cpu       # explicit CPU
    allow_fallback: true
```

---

## Directory layout

- `configs/student/`     — student presets, fit on 8GB laptops
- `configs/teaching/`    — guided exercises with learning objectives
- `configs/simulation/`  — research / production presets (~10 variants)
- `configs/benchmark/`   — performance-focused, no or minimal output
- `configs/test/`        — dev-side test fixtures (not user-facing)

Each user-facing YAML opens with a 3-line scenario header explaining
what it's for and what to flip if you want a variant. If the header
doesn't match what you need, copy the preset and tweak.
