# Boundary Conditions

Coordinate-specific boundary enforcement for the simulation domain edges.

## Why No Factory

Boundary conditions are not a user-selectable physics scheme. They are mathematical constraints determined by the coordinate system geometry. The active coordinate system (Cartesian or cylindrical) determines which BC functions are called -- there is nothing for the user to configure.

This is distinct from the `boundary_layer/` module, which is atmospheric physics (vertical mixing, surface fluxes, PBL height diagnosis). Despite the similar names, these modules are orthogonal:

| | boundary_conditions/ | boundary_layer/ |
|---|---|---|
| **What** | Mathematical edge constraints | Atmospheric physics |
| **Where** | Grid boundaries (6 faces) | Lower atmosphere (surface to PBL top) |
| **Configured by** | Coordinate system (automatic) | User (scheme selection in YAML) |
| **Pattern** | Two fixed implementations | Factory with 3 schemes |
| **Examples** | Zero-gradient, periodic, rigid-lid | Surface fluxes, K-profile mixing |

## Layout

```
src/boundary_conditions/
  boundary_conditions_cartesian.cpp    BCs for Cartesian (x, y, z) grids
  boundary_conditions_cylindrical.cpp  BCs for cylindrical (r, theta, z) grids

include/boundary_conditions/
  boundary_conditions.hpp              Public API (apply_boundary_conditions)
  boundary_conditions_base.hpp         Shared BC definitions
```

## Boundary Types

**Cartesian grids:**
- Lateral faces (x, y): zero-gradient for all prognostic fields
- Top/bottom (z): rigid-lid (w=0), zero-gradient for scalars, hydrostatic pressure extrapolation

**Cylindrical grids:**
- Radial (r=0): axis symmetry conditions (antisymmetric u_r, symmetric scalars)
- Radial (r=R): zero-gradient
- Azimuthal (theta): periodic
- Top/bottom (z): rigid-lid (w=0), zero-gradient for scalars

## Runtime Integration

BCs are applied by the split-explicit time-stepping loop after each acoustic substep, and by the main dynamics step after tendency integration. The call site is in `src/core/orchestration/dynamics/dynamics.cpp`.
