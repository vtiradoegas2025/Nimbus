/**
 * @file tornado_cgrid.hpp
 * @brief Arakawa C-grid axisymmetric tornado dynamics scheme.
 *
 * Phase C.4 of docs/CoordinateBackend_Plan.md. Computes momentum, mass and
 * pressure tendencies for the staggered cylindrical grid where u lives on
 * r-faces, v on theta-faces, w on z-faces, and scalars (rho, p, theta, q*)
 * at cell centers.
 *
 * Compared to the collocated TornadoScheme, this scheme:
 *   - replaces the antisymmetric ghost-cell hack u[0] = -u[1] with the
 *     C-grid axis condition (u[0] is interior; the implicit "u at the
 *     axis" is zero because no flow crosses r=0)
 *   - uses one-sided pressure gradients at faces (grad_r, grad_z)
 *   - uses flux-form divergence at cell centers (div_flux, axis-aware)
 *   - uses reference-state subtraction for the vertical momentum equation
 *     (perturbation pressure + perturbation density buoyancy), matching
 *     the supercell scheme; this preserves discrete hydrostatic balance
 *     to machine precision and avoids the Bug 3 double-counted-buoyancy
 *     trap (see docs/Journey.md)
 *   - uses the standard compressible pressure equation
 *     dp/dt = -gamma*p*div(u) - u . grad(p), which is exactly zero at
 *     hydrostatic and cyclostrophic balance (no spurious "centrifugal
 *     pressure source" that would drift Lamb-Oseen out of balance).
 */

#pragma once
#include "dynamics/dynamics_base.hpp"
#include "numerics/derivatives/derivative_operators.hpp"
#include <string>


/**
 * @brief Axisymmetric tornado dynamics scheme on the cylindrical C-grid.
 *
 * Implements both DynamicsScheme (unsplit / single-sweep path used by
 * Forward Euler, RK, and direct tests) AND SplitExplicitDynamics
 * (Klemp-Wilhelmson slow / fast acoustic decomposition).
 *
 * Slow / fast decomposition follows the SupercellCGridScheme convention,
 * adapted for axisymmetric:
 *   slow: advection + centrifugal/curvature + buoyancy on velocities;
 *         dp/dt = -u . grad(p); drho/dt = 0.
 *   fast pressure: dp/dt = -gamma*p*div_flux, drho/dt = -rho*div_flux.
 *   fast momentum: -grad(p)/rho on u (r-face);
 *                  dv/dt fast == 0 (axisymmetric: dp/dtheta = 0);
 *                  -(dp/dz - dp0/dz)/rho on w (z-face).
 * Sum (slow + fast) reproduces the unsplit tendencies bit-exactly.
 */
class TornadoCGridScheme : public DynamicsScheme, public SplitExplicitDynamics
{
public:
    TornadoCGridScheme();

    // --- DynamicsScheme interface (unsplit / single-step path) -----------

    void compute_momentum_tendencies(
        const Field3D& u,
        const Field3D& v,
        const Field3D& w,
        const Field3D& rho,
        const Field3D& p,
        const Field3D& theta,
        double dt,
        Field3D& du_dt,
        Field3D& dv_dt,
        Field3D& dw_dt,
        Field3D& drho_dt,
        Field3D& dp_dt
    ) override;

    void compute_angular_momentum(
        const Field3D& u,
        const Field3D& v,
        Field3D& angular_momentum,
        Field3D& angular_momentum_tendency
    ) override;

    void compute_vorticity_diagnostics(
        const Field3D& u,
        const Field3D& v,
        const Field3D& w,
        const Field3D& rho,
        const Field3D& p,
        Field3D& vorticity_r,
        Field3D& vorticity_theta,
        Field3D& vorticity_z,
        Field3D& stretching_term,
        Field3D& tilting_term,
        Field3D& baroclinic_term
    ) override;

    // --- SplitExplicitDynamics interface ---------------------------------

    void compute_slow_tendencies(
        const Field3D& u, const Field3D& v, const Field3D& w,
        const Field3D& rho, const Field3D& p, const Field3D& theta, double dt,
        Field3D& du_dt, Field3D& dv_dt, Field3D& dw_dt,
        Field3D& drho_dt, Field3D& dp_dt) override;

    void compute_fast_pressure_tendencies(
        const Field3D& u, const Field3D& v, const Field3D& w,
        const Field3D& rho, const Field3D& p,
        Field3D& drho_dt, Field3D& dp_dt) override;

    void compute_fast_momentum_tendencies(
        const Field3D& u, const Field3D& v, const Field3D& w,
        const Field3D& rho, const Field3D& p,
        Field3D& du_dt, Field3D& dv_dt, Field3D& dw_dt) override;

    // --- Scheme identity -------------------------------------------------

    std::string get_scheme_name() const override { return "tornado_cgrid"; }
    std::string get_coordinate_system() const override { return "cylindrical_cgrid"; }
    int get_num_prognostic_vars() const override { return 5; }

private:
    int NR_, NTH_, NZ_;
    double dr_, dtheta_, dz_;
    const GridGeometry& geo_;
    StaggeredCylindricalDerivatives deriv_;
};
