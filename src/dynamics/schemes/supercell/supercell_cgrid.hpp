/**
 * @file supercell_cgrid.hpp
 * @brief Arakawa C-grid non-axisymmetric supercell dynamics scheme.
 *
 * Phase C.5 of docs/CoordinateBackend_Plan.md. Computes momentum, mass and
 * pressure tendencies on the staggered cylindrical grid for a fully
 * three-dimensional (theta-dependent) flow field. Field placement matches
 * TornadoCGridScheme:
 *   u at r-face (r_{i+1/2}, theta_j,      z_k)
 *   v at theta-face(r_i,    theta_{j+1/2}, z_k)
 *   w at z-face (r_i,       theta_j,       z_{k+1/2})
 *   scalars at cell center (r_i, theta_j,  z_k)
 *
 * Differences from TornadoCGridScheme (which assumes axisymmetric, j=0
 * replicate):
 *   - Loops over all j with periodic theta indexing.
 *   - Adds the azimuthal pressure gradient at theta-face
 *     (1/r) * (p[j+1] - p[j]) / dtheta and the corresponding -dp_dtheta/rho
 *     contribution to dv/dt.
 *   - 4-point bilinear interpolation of cross-component velocities (e.g. v
 *     at r-face uses both i,j and i+1,j-1 via theta-face placement) so that
 *     advective tendencies of u, v, w are second-order accurate on the
 *     staggered grid.
 *   - Adds full azimuthal advection terms in compute_momentum_tendencies
 *     and compute_slow_tendencies.
 *
 * Inherits SplitExplicitDynamics so it can be driven by the
 * Klemp-Wilhelmson split-explicit time stepping path. The slow/fast
 * decomposition follows the same convention as the collocated supercell:
 *   - slow: advection, centrifugal/coriolis, buoyancy (NO pressure
 *     gradient on velocity, NO -gamma*p*div on pressure, drho/dt = 0).
 *   - fast pressure: -gamma*p*div_flux, -rho*div_flux.
 *   - fast momentum: -grad(p)/rho on velocities; vertical uses the
 *     reference-state perturbation gradient (dp/dz - dp0/dz)/rho for
 *     the same Bug 3 reason.
 *
 * The standard compressible pressure equation
 *   dp/dt = -gamma*p*div(u) - u . grad(p)
 * is reconstructed by summing slow + N * fast contributions inside the
 * split-explicit time stepper. The unsplit compute_momentum_tendencies
 * call (used by Forward Euler / RK paths and by tests) computes both
 * slow and fast contributions in a single sweep.
 */

#pragma once
#include "dynamics/dynamics_base.hpp"
#include "numerics/derivatives/derivative_operators.hpp"
#include <string>


/**
 * @brief Non-axisymmetric supercell dynamics scheme on the cylindrical
 *        Arakawa C-grid. Supports split-explicit acoustic time stepping.
 */
class SupercellCGridScheme : public DynamicsScheme, public SplitExplicitDynamics
{
public:
    SupercellCGridScheme();

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

    void compute_pressure_diagnostics(
        const Field3D& u,
        const Field3D& v,
        const Field3D& w,
        const Field3D& rho,
        const Field3D& theta,
        Field3D& p_prime,
        Field3D& dynamic_pressure,
        Field3D& buoyancy_pressure
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

    std::string get_scheme_name() const override { return "supercell_cgrid"; }
    std::string get_coordinate_system() const override { return "cylindrical_cgrid"; }
    int get_num_prognostic_vars() const override { return 5; }

private:
    int NR_, NTH_, NZ_;
    double dr_, dtheta_, dz_;
    const GridGeometry& geo_;
    StaggeredCylindricalDerivatives deriv_;
};
