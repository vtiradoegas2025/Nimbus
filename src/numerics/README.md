# Numerics Module

This module provides the numerical kernels used by the simulation runtime and is wired through `src/core/orchestration/dynamics/numerics.cpp`.

## Current Runtime Integration

Numerics components are initialized once by:
- `src/core/runtime/tornado_sim.cpp` -> `initialize_numerics()`

Numerics components are consumed at runtime by:
- `src/numerics/advection/advection.cpp` and `advection_cartesian.cpp`
  - Directional splitting kernels for cylindrical and Cartesian coordinate systems.
  - Uses factory-selected advection schemes (TVD, WENO5) for the vertical split step.
- `src/core/orchestration/dynamics/dynamics.cpp`
  - Applies numerics diffusion tendencies (`explicit` or `implicit`) to momentum/scalars.
- `src/core/runtime/headless_runtime.cpp`
- `src/core/runtime/gui.cpp`
  - Both use `choose_runtime_timestep()` to enforce numerics stability caps.

## Directory Layout

```text
src/numerics/
├── advection/
│   ├── factory.cpp/.hpp
│   ├── advection.cpp              cylindrical directional splitting kernels
│   ├── advection_cartesian.cpp    Cartesian directional splitting kernels
│   └── schemes/
│       ├── tvd/
│       └── weno5/
├── diffusion/
│   ├── factory.cpp/.hpp
│   └── schemes/
│       ├── explicit/
│       └── implicit/
├── time_stepping/
│   ├── factory.cpp/.hpp
│   └── schemes/
│       ├── rk3/
│       ├── rk4/
│       └── split_explicit/
└── README.md
```

## Scheme Behavior

### Advection
- Implementations: `tvd`, `weno5`
- Scope today: computes vertical 1D tracer tendencies (`dqdt_adv`) used by the `z` split step.
- Includes CFL diagnostics and suggested timestep estimates.
- Positivity protection is applied as a tendency limiter (limits `dqdt` so `q + dt*dqdt >= floor`).

### Diffusion
- Implementations: `explicit`, `implicit`
- Scope today: vertical diffusion tendencies for momentum and scalar fields.
- `explicit` uses flux-form tendencies and now uses the physically diffusive sign convention.
- `implicit` uses a tridiagonal solve in the vertical (Crank-Nicolson style) and derives tendencies from updated state.
- Variable diffusivity fields are supported and sanitized to non-negative finite values at use sites.

### Time Stepping
- Implementations: `rk3`, `rk4`, `split_explicit`
- `rk3` is implemented as a 3-stage SSPRK3 update.
- `rk4` is a standard 4-stage Runge-Kutta update.
- `split_explicit` is a Klemp-Wilhelmson (1978) forward-backward scheme that separates fast acoustic modes from slow advective modes. The acoustic substep loop supports GPU-accelerated fused and batched dispatch paths.
- In current app wiring, global model stepping is handled in dynamics; numerics time-stepping schemes provide both the integration logic and timestep guidance (`suggest_dt`).

## Runtime Timestep Guardrails

`choose_runtime_timestep()` in `src/core/orchestration/dynamics/numerics.cpp` applies:
- Config bounds: `dt_min`, `dt_max`
- Advection cap: based on max resolved flow speed and grid spacing
- Explicit diffusion cap: based on `K_h`, `K_v`, and grid spacing

It enforces caps even when adaptive dt is disabled (as a stability guardrail).

## Configuration

Typical YAML usage:

```yaml
numerics:
  advection: weno5
  diffusion:
    scheme: explicit
    apply_to: all
    K_h: 20.0
    K_v: 10.0
  time_stepping: rk3
```

Factory name parsing is canonicalized (case-insensitive; ignores `_`, `-`, and spaces), so variants like `SSP-RK3` are accepted for time stepping.

## Final Check Status

Final numerics verification was run with:
- `make -j4 bin/tornado_sim`
- `bash tests/test_guards.sh`
- Headless smoke run: TVD + explicit diffusion + SSP-RK3
- Headless smoke run: WENO5 + implicit diffusion + RK4

All checks above completed successfully.

## Known Limitations

- Diffusion kernels are currently vertical-focused in both explicit and implicit schemes.
- Full 3D anisotropic diffusion is not yet implemented.
