/**
 * @file cartesian.cpp
 * @brief Implementation of the Cartesian dynamics scheme.
 *
 * Computes momentum, mass, and pressure tendencies on a regular Cartesian
 * (x, y, z) grid. This is the backend that solves Bug 7: a uniform horizontal
 * wind on a Cartesian grid produces zero spurious body force, whereas the
 * cylindrical antisymmetric-BC convention at i=0 creates a false radial
 * gradient that breaks hydrostatic balance for non-axisymmetric hodographs
 * like the WK2002 supercell case.
 *
 * Phase A (CPU only, per docs/CoordinateBackend_Plan.md). The GPU shader
 * port (cartesian_tendencies.comp) is scheduled for A.7.
 *
 * Field reuse note: the base `DynamicsScheme` interface still uses the
 * cylindrical-flavored argument names (`u_r`, `u_theta`, `u_z`). In Cartesian
 * mode we alias them locally as `u_x`, `u_y`, `w_field`. The global rename
 * (`u -> u_x`, `v_theta -> u_y`) is deferred to Phase B, after both backends
 * work end to end.
 *
 * This file is part of the src/dynamics subsystem.
 */

#include "cartesian.hpp"
#include "core/simulation.hpp"
#include "numerics/compute_kernel_template.hpp"
#include <cmath>
#include <algorithm>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

/**
 * @brief Initializes the Cartesian dynamics scheme by caching the grid
 *        dimensions and horizontal/vertical spacings.
 */
CartesianScheme::CartesianScheme()
    : NR_(NR), NTH_(NTH), NZ_(NZ),
      dr_(dr), dz_(dz)
{
}

/**
 * @brief Computes the momentum tendencies for the Cartesian scheme.
 *
 * Differences from the cylindrical scheme:
 *   - No centrifugal term (no u_θ²/r).
 *   - No azimuthal coriolis-like coupling (no -u_r u_θ/r).
 *   - Divergence is straightforward: ∂u/∂x + ∂v/∂y + ∂w/∂z (no 1/r terms).
 *   - Derivatives use literal centered differences with dx = dy = dr_
 *     and dz = dz_. There is no axis singularity, so the eps guards are
 *     removed and the loop runs over the full interior 1..N-1 in all
 *     three dimensions.
 *   - The vertical momentum equation is reused verbatim from the cylindrical
 *     schemes (post-Bug-3): dw/dt = -(1/ρ) ∂p/∂z - g + advection.
 */
void CartesianScheme::compute_momentum_tendencies(
    const Field3D& u_r,
    const Field3D& u_theta,
    const Field3D& u_z,
    const Field3D& rho,
    const Field3D& p,
    const Field3D& theta,
    double dt,
    Field3D& du_r_dt,
    Field3D& du_theta_dt,
    Field3D& du_z_dt,
    Field3D& drho_dt,
    Field3D& dp_dt)
{
    // Field-name aliasing. In Cartesian mode, the cylindrical-flavored slots
    // carry their Cartesian counterparts. The rename of the actual field
    // globals happens in Phase B (docs/CoordinateBackend_Plan.md §Phase B).
    const Field3D& u_x = u_r;        // x-velocity
    const Field3D& u_y = u_theta;    // y-velocity
    const Field3D& w_field = u_z;    // z-velocity (already the right name)
    Field3D& du_x_dt = du_r_dt;
    Field3D& du_y_dt = du_theta_dt;
    Field3D& dw_dt = du_z_dt;

    (void)dt;     // Forward-Euler update is done by the runtime coupler,
                  // not the scheme.
    (void)theta;  // theta is not used in momentum tendencies: buoyancy is
                  // implicit in (-∂p/∂z/ρ - g). See Journey.md "Bug 3:
                  // Double-counted buoyancy" for why re-introducing an
                  // explicit g·(θ-θ₀)/θ₀ term here is wrong.

    // --- GPU dispatch (A.7): attempt Vulkan compute if available -----------
    // The GPU shader includes the reference-state subtraction and matches
    // the CPU fallback path below. The terrain guard mirrors the supercell
    // scheme's convention: GPU shaders assume uniform grid spacing.
    {
        // Prepare 1D profile data as float arrays for the GPU.
        std::vector<float> p0f(static_cast<size_t>(NZ_));
        std::vector<float> rho0f(static_cast<size_t>(NZ_));
        for (int k = 0; k < NZ_; ++k)
        {
            p0f[k] = static_cast<float>(p0_base[k]);
            rho0f[k] = static_cast<float>(rho0_base[k]);
        }

        if (dispatch_cartesian_tendencies_backend(
                u_r.data(), u_theta.data(), u_z.data(),
                rho.data(), p.data(), theta.data(),
                p0f.data(), rho0f.data(),
                du_r_dt.data(), du_theta_dt.data(), du_z_dt.data(),
                drho_dt.data(), dp_dt.data(),
                NR_, NTH_, NZ_,
                static_cast<float>(dr_), static_cast<float>(dr_),
                static_cast<float>(dz_),
                static_cast<float>(dynamics_constants::g),
                static_cast<float>(dynamics_constants::gamma)))
        {
            return;  // GPU computed successfully
        }
    }

    // --- CPU fallback path --------------------------------------------------

    // Zero outputs everywhere so the boundary layer is guaranteed zero.
    // The interior loop overwrites indices [1..N-1); faces stay at 0.
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < NR_; ++i)
    {
        for (int j = 0; j < NTH_; ++j)
        {
            for (int k = 0; k < NZ_; ++k)
            {
                du_x_dt[i][j][k] = 0.0f;
                du_y_dt[i][j][k] = 0.0f;
                dw_dt[i][j][k] = 0.0f;
                drho_dt[i][j][k] = 0.0f;
                dp_dt[i][j][k] = 0.0f;
            }
        }
    }

    // CPU implementation. A.7 will add GPU dispatch here, modeled after the
    // supercell GPU pipeline (guarded by `dispatch_cartesian_tendencies_backend`).
    #pragma omp parallel for collapse(2)
    for (int i = 1; i < NR_ - 1; ++i)
    {
        for (int j = 1; j < NTH_ - 1; ++j)
        {
            for (int k = 1; k < NZ_ - 1; ++k)
            {
                const double ux = u_x[i][j][k];
                const double uy = u_y[i][j][k];
                const double wz = w_field[i][j][k];
                const double rho_val = rho[i][j][k];
                const double p_val = p[i][j][k];
                const double rho_safe = (std::isfinite(rho_val) && rho_val > 1.0e-6)
                                            ? rho_val : 1.0;

                // Centered first derivatives.
                const double dux_dx = compute_dx(u_x, i, j, k);
                const double dux_dy = compute_dy(u_x, i, j, k);
                const double dux_dz = compute_dz(u_x, i, j, k);

                const double duy_dx = compute_dx(u_y, i, j, k);
                const double duy_dy = compute_dy(u_y, i, j, k);
                const double duy_dz = compute_dz(u_y, i, j, k);

                const double dw_dx = compute_dx(w_field, i, j, k);
                const double dw_dy = compute_dy(w_field, i, j, k);
                const double dw_dz_local = compute_dz(w_field, i, j, k);

                const double dp_dx = compute_dx(p, i, j, k);
                const double dp_dy = compute_dy(p, i, j, k);
                const double dp_dz_local = compute_dz(p, i, j, k);

                // --- x-momentum: ∂u_x/∂t = -(u·∇) u_x - (1/ρ) ∂p/∂x ---
                const double advective_x = -ux * dux_dx - uy * dux_dy - wz * dux_dz;
                const double pressure_grad_x = -dp_dx / rho_safe;
                double du_x_val = advective_x + pressure_grad_x;
                if (!std::isfinite(du_x_val)) du_x_val = 0.0;
                du_x_dt[i][j][k] = static_cast<float>(du_x_val);

                // --- y-momentum: ∂u_y/∂t = -(u·∇) u_y - (1/ρ) ∂p/∂y ---
                const double advective_y = -ux * duy_dx - uy * duy_dy - wz * duy_dz;
                const double pressure_grad_y = -dp_dy / rho_safe;
                double du_y_val = advective_y + pressure_grad_y;
                if (!std::isfinite(du_y_val)) du_y_val = 0.0;
                du_y_dt[i][j][k] = static_cast<float>(du_y_val);

                // --- z-momentum with reference-state subtraction ---
                //
                // The naive form  dw/dt = -(1/ρ)∂p/∂z - g  has a discrete
                // hydrostatic imbalance: the centered stencil of the initial
                // pressure does not exactly cancel ρ₀g on a collocated grid,
                // seeding O(Δz²) spurious vertical acceleration (~0.03 m/s²
                // at Δz=500 m). Over 30 steps this grows to |w|>1 m/s.
                //
                // The standard cloud-model fix (CM1, WRF) decomposes into
                // perturbation and reference:
                //
                //   dw/dt = -(1/ρ)(∂p/∂z - ∂p₀/∂z) - g(ρ - ρ₀)/ρ
                //
                // where ∂p₀/∂z is the SAME centered stencil applied to the
                // stored base-state pressure p0_base[]. At initialization
                // p ≡ p₀ and ρ ≡ ρ₀, so both perturbation terms are exactly
                // zero — no residual, no spurious w.
                //
                // See docs/Journey.md Phase 2 "Bug 3" for why an explicit
                // g·(θ-θ₀)/θ₀ buoyancy term must NOT be added on top.
                const double advective_z = -ux * dw_dx - uy * dw_dy - wz * dw_dz_local;

                // Reference-state centered ∂p₀/∂z (same stencil as dp_dz_local).
                const double dp0_dz = (p0_base[k + 1] - p0_base[k - 1]) / (2.0 * dz_);
                const double dp_prime_dz = dp_dz_local - dp0_dz;
                const double rho0_k = rho0_base[k];
                const double buoyancy = -dynamics_constants::g * (rho_val - rho0_k) / rho_safe;
                double dw_val = advective_z - dp_prime_dz / rho_safe + buoyancy;
                if (!std::isfinite(dw_val)) dw_val = 0.0;
                dw_dt[i][j][k] = static_cast<float>(dw_val);

                // --- Mass continuity (anelastic-style): ∂ρ/∂t = -ρ ∇·u.
                //     Straight Cartesian divergence, no 1/r terms. ---
                const double divergence = dux_dx + duy_dy + dw_dz_local;
                double drho_val = -rho_safe * divergence;
                if (!std::isfinite(drho_val)) drho_val = 0.0;
                drho_dt[i][j][k] = static_cast<float>(drho_val);

                // --- Pressure tendency: ∂p/∂t = -γ p ∇·u - u·∇p ---
                const double gamma_term = dynamics_constants::gamma * p_val * divergence;
                const double advection_p = -ux * dp_dx - uy * dp_dy - wz * dp_dz_local;
                double dp_val = -gamma_term + advection_p;
                if (!std::isfinite(dp_val)) dp_val = 0.0;
                dp_dt[i][j][k] = static_cast<float>(dp_val);
            }
        }
    }
}

// =========================================================================
// Split-explicit acoustic substep methods (Klemp & Wilhelmson 1978)
// =========================================================================

void CartesianScheme::compute_slow_tendencies(
    const Field3D& u_r, const Field3D& u_theta, const Field3D& u_z,
    const Field3D& rho, const Field3D& p, const Field3D& theta,
    double /*dt*/,
    Field3D& du_r_dt, Field3D& du_theta_dt, Field3D& du_z_dt,
    Field3D& drho_dt, Field3D& dp_dt)
{
    const Field3D& u_x = u_r;
    const Field3D& u_y = u_theta;
    const Field3D& w_field = u_z;
    Field3D& du_x_dt = du_r_dt;
    Field3D& du_y_dt = du_theta_dt;
    Field3D& dw_dt = du_z_dt;

    (void)theta;

    // Zero all outputs (boundaries stay zero).
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < NR_; ++i)
        for (int j = 0; j < NTH_; ++j)
            for (int k = 0; k < NZ_; ++k)
            {
                du_x_dt[i][j][k] = 0.0f;
                du_y_dt[i][j][k] = 0.0f;
                dw_dt[i][j][k] = 0.0f;
                drho_dt[i][j][k] = 0.0f;
                dp_dt[i][j][k] = 0.0f;
            }

    #pragma omp parallel for collapse(2)
    for (int i = 1; i < NR_ - 1; ++i)
    {
        for (int j = 1; j < NTH_ - 1; ++j)
        {
            for (int k = 1; k < NZ_ - 1; ++k)
            {
                const double ux = u_x[i][j][k];
                const double uy = u_y[i][j][k];
                const double wz = w_field[i][j][k];
                const double rho_val = rho[i][j][k];
                const double rho_safe = (std::isfinite(rho_val) && rho_val > 1.0e-6)
                                            ? rho_val : 1.0;

                // Velocity derivatives for advection.
                const double dux_dx = compute_dx(u_x, i, j, k);
                const double dux_dy = compute_dy(u_x, i, j, k);
                const double dux_dz = compute_dz(u_x, i, j, k);
                const double duy_dx = compute_dx(u_y, i, j, k);
                const double duy_dy = compute_dy(u_y, i, j, k);
                const double duy_dz = compute_dz(u_y, i, j, k);
                const double dw_dx = compute_dx(w_field, i, j, k);
                const double dw_dy = compute_dy(w_field, i, j, k);
                const double dw_dz_local = compute_dz(w_field, i, j, k);

                // Pressure derivatives for advection of pressure.
                const double dp_dx = compute_dx(p, i, j, k);
                const double dp_dy = compute_dy(p, i, j, k);
                const double dp_dz_local = compute_dz(p, i, j, k);

                // x-momentum: advection only
                double du_x_val = -ux * dux_dx - uy * dux_dy - wz * dux_dz;
                if (!std::isfinite(du_x_val)) du_x_val = 0.0;
                du_x_dt[i][j][k] = static_cast<float>(du_x_val);

                // y-momentum: advection only
                double du_y_val = -ux * duy_dx - uy * duy_dy - wz * duy_dz;
                if (!std::isfinite(du_y_val)) du_y_val = 0.0;
                du_y_dt[i][j][k] = static_cast<float>(du_y_val);

                // z-momentum: advection + buoyancy (no pressure gradient)
                const double advective_z = -ux * dw_dx - uy * dw_dy - wz * dw_dz_local;
                const double rho0_k = rho0_base[k];
                const double buoyancy = -dynamics_constants::g * (rho_val - rho0_k) / rho_safe;
                double dw_val = advective_z + buoyancy;
                if (!std::isfinite(dw_val)) dw_val = 0.0;
                dw_dt[i][j][k] = static_cast<float>(dw_val);

                // density: no slow tendency (all continuity is fast)
                drho_dt[i][j][k] = 0.0f;

                // pressure: advection only (no compression)
                double dp_val = -ux * dp_dx - uy * dp_dy - wz * dp_dz_local;
                if (!std::isfinite(dp_val)) dp_val = 0.0;
                dp_dt[i][j][k] = static_cast<float>(dp_val);
            }
        }
    }
}

void CartesianScheme::compute_fast_pressure_tendencies(
    const Field3D& u_r, const Field3D& u_theta, const Field3D& u_z,
    const Field3D& rho, const Field3D& p,
    Field3D& drho_dt, Field3D& dp_dt)
{
    const Field3D& u_x = u_r;
    const Field3D& u_y = u_theta;
    const Field3D& w_field = u_z;

    // Zero outputs.
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < NR_; ++i)
        for (int j = 0; j < NTH_; ++j)
            for (int k = 0; k < NZ_; ++k)
            {
                drho_dt[i][j][k] = 0.0f;
                dp_dt[i][j][k] = 0.0f;
            }

    #pragma omp parallel for collapse(2)
    for (int i = 1; i < NR_ - 1; ++i)
    {
        for (int j = 1; j < NTH_ - 1; ++j)
        {
            for (int k = 1; k < NZ_ - 1; ++k)
            {
                const double rho_val = rho[i][j][k];
                const double p_val = p[i][j][k];
                const double rho_safe = (std::isfinite(rho_val) && rho_val > 1.0e-6)
                                            ? rho_val : 1.0;

                const double dux_dx = compute_dx(u_x, i, j, k);
                const double duy_dy = compute_dy(u_y, i, j, k);
                const double dw_dz_local = compute_dz(w_field, i, j, k);
                const double divergence = dux_dx + duy_dy + dw_dz_local;

                // Continuity: dρ/dt = -ρ ∇·u
                double drho_val = -rho_safe * divergence;
                if (!std::isfinite(drho_val)) drho_val = 0.0;
                drho_dt[i][j][k] = static_cast<float>(drho_val);

                // Compression: dp/dt = -γ p ∇·u
                double dp_val = -dynamics_constants::gamma * p_val * divergence;
                if (!std::isfinite(dp_val)) dp_val = 0.0;
                dp_dt[i][j][k] = static_cast<float>(dp_val);
            }
        }
    }
}

void CartesianScheme::compute_fast_momentum_tendencies(
    const Field3D& /*u_r*/, const Field3D& /*u_theta*/, const Field3D& /*u_z*/,
    const Field3D& rho, const Field3D& p,
    Field3D& du_r_dt, Field3D& du_theta_dt, Field3D& du_z_dt)
{
    Field3D& du_x_dt = du_r_dt;
    Field3D& du_y_dt = du_theta_dt;
    Field3D& dw_dt = du_z_dt;

    // Zero outputs.
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < NR_; ++i)
        for (int j = 0; j < NTH_; ++j)
            for (int k = 0; k < NZ_; ++k)
            {
                du_x_dt[i][j][k] = 0.0f;
                du_y_dt[i][j][k] = 0.0f;
                dw_dt[i][j][k] = 0.0f;
            }

    #pragma omp parallel for collapse(2)
    for (int i = 1; i < NR_ - 1; ++i)
    {
        for (int j = 1; j < NTH_ - 1; ++j)
        {
            for (int k = 1; k < NZ_ - 1; ++k)
            {
                const double rho_val = rho[i][j][k];
                const double rho_safe = (std::isfinite(rho_val) && rho_val > 1.0e-6)
                                            ? rho_val : 1.0;

                const double dp_dx = compute_dx(p, i, j, k);
                const double dp_dy = compute_dy(p, i, j, k);
                const double dp_dz_local = compute_dz(p, i, j, k);

                // x-momentum: pressure gradient only
                double du_x_val = -dp_dx / rho_safe;
                if (!std::isfinite(du_x_val)) du_x_val = 0.0;
                du_x_dt[i][j][k] = static_cast<float>(du_x_val);

                // y-momentum: pressure gradient only
                double du_y_val = -dp_dy / rho_safe;
                if (!std::isfinite(du_y_val)) du_y_val = 0.0;
                du_y_dt[i][j][k] = static_cast<float>(du_y_val);

                // z-momentum: perturbation pressure gradient only (no buoyancy)
                const double dp0_dz = (p0_base[k + 1] - p0_base[k - 1]) / (2.0 * dz_);
                const double dp_prime_dz = dp_dz_local - dp0_dz;
                double dw_val = -dp_prime_dz / rho_safe;
                if (!std::isfinite(dw_val)) dw_val = 0.0;
                dw_dt[i][j][k] = static_cast<float>(dw_val);
            }
        }
    }
}

/**
 * @brief Computes Cartesian vorticity diagnostics.
 *
 * Cartesian curl of velocity:
 *   ω_x = ∂w/∂y - ∂v/∂z
 *   ω_y = ∂u/∂z - ∂w/∂x
 *   ω_z = ∂v/∂x - ∂u/∂y
 *
 * Reused terms from the vertical vorticity equation:
 *   stretching = ω_z * ∂w/∂z
 *   tilting    = ω_x * ∂w/∂x + ω_y * ∂w/∂y
 *
 * Baroclinic (vertical component): (∇ρ × ∇p)_z / ρ²
 *   = (∂ρ/∂x · ∂p/∂y - ∂ρ/∂y · ∂p/∂x) / ρ²
 *
 * Fields are stored in the cylindrical-flavored slots:
 *   vorticity_r     -> ω_x
 *   vorticity_theta -> ω_y
 *   vorticity_z     -> ω_z
 */
void CartesianScheme::compute_vorticity_diagnostics(
    const Field3D& u_r,
    const Field3D& u_theta,
    const Field3D& u_z,
    const Field3D& rho,
    const Field3D& p,
    Field3D& vorticity_r,
    Field3D& vorticity_theta,
    Field3D& vorticity_z,
    Field3D& stretching_term,
    Field3D& tilting_term,
    Field3D& baroclinic_term)
{
    const Field3D& u_x = u_r;
    const Field3D& u_y = u_theta;
    const Field3D& w_field = u_z;

    #pragma omp parallel for collapse(2)
    for (int i = 1; i < NR_ - 1; ++i)
    {
        for (int j = 1; j < NTH_ - 1; ++j)
        {
            for (int k = 1; k < NZ_ - 1; ++k)
            {
                const double duy_dx = compute_dx(u_y, i, j, k);
                const double dux_dy = compute_dy(u_x, i, j, k);
                const double dux_dz = compute_dz(u_x, i, j, k);
                const double duy_dz = compute_dz(u_y, i, j, k);
                const double dw_dx = compute_dx(w_field, i, j, k);
                const double dw_dy = compute_dy(w_field, i, j, k);
                const double dw_dz_local = compute_dz(w_field, i, j, k);

                double omega_x = dw_dy - duy_dz;
                double omega_y = dux_dz - dw_dx;
                double omega_z = duy_dx - dux_dy;
                if (!std::isfinite(omega_x)) omega_x = 0.0;
                if (!std::isfinite(omega_y)) omega_y = 0.0;
                if (!std::isfinite(omega_z)) omega_z = 0.0;

                vorticity_r[i][j][k] = static_cast<float>(omega_x);
                vorticity_theta[i][j][k] = static_cast<float>(omega_y);
                vorticity_z[i][j][k] = static_cast<float>(omega_z);

                double stretch = omega_z * dw_dz_local;
                if (!std::isfinite(stretch)) stretch = 0.0;
                stretching_term[i][j][k] = static_cast<float>(stretch);

                double tilt = omega_x * dw_dx + omega_y * dw_dy;
                if (!std::isfinite(tilt)) tilt = 0.0;
                tilting_term[i][j][k] = static_cast<float>(tilt);

                const double drho_dx = compute_dx(rho, i, j, k);
                const double drho_dy = compute_dy(rho, i, j, k);
                const double dp_dx = compute_dx(p, i, j, k);
                const double dp_dy = compute_dy(p, i, j, k);

                const double rho_val = rho[i][j][k];
                const double rho_sq = rho_val * rho_val;
                double baro = 0.0;
                if (rho_sq > dynamics_constants::eps)
                {
                    baro = (drho_dx * dp_dy - drho_dy * dp_dx) / rho_sq;
                }
                if (!std::isfinite(baro)) baro = 0.0;
                baroclinic_term[i][j][k] = static_cast<float>(baro);
            }
        }
    }
}

/**
 * @brief Computes Cartesian pressure-decomposition diagnostics.
 *
 * Dynamic-pressure proxy uses the sum of squared velocity derivatives
 * (the magnitude of the deformation / velocity-gradient tensor). Buoyancy
 * pressure uses the same θ-perturbation convention as the cylindrical
 * schemes. The combined p' is the sum of the two.
 */
void CartesianScheme::compute_pressure_diagnostics(
    const Field3D& u_r,
    const Field3D& u_theta,
    const Field3D& u_z,
    const Field3D& rho,
    const Field3D& theta,
    Field3D& p_prime,
    Field3D& dynamic_pressure,
    Field3D& buoyancy_pressure)
{
    (void)rho;  // rho is not needed here; we use the 1D rho0_base profile
                // for consistency with the cylindrical schemes.

    const Field3D& u_x = u_r;
    const Field3D& u_y = u_theta;
    const Field3D& w_field = u_z;

    #pragma omp parallel for collapse(2)
    for (int i = 1; i < NR_ - 1; ++i)
    {
        for (int j = 1; j < NTH_ - 1; ++j)
        {
            for (int k = 1; k < NZ_ - 1; ++k)
            {
                const double dux_dx = compute_dx(u_x, i, j, k);
                const double dux_dy = compute_dy(u_x, i, j, k);
                const double dux_dz = compute_dz(u_x, i, j, k);
                const double duy_dx = compute_dx(u_y, i, j, k);
                const double duy_dy = compute_dy(u_y, i, j, k);
                const double duy_dz = compute_dz(u_y, i, j, k);
                const double dw_dx = compute_dx(w_field, i, j, k);
                const double dw_dy = compute_dy(w_field, i, j, k);
                const double dw_dz_local = compute_dz(w_field, i, j, k);

                double deformation =
                    dux_dx * dux_dx + duy_dy * duy_dy + dw_dz_local * dw_dz_local +
                    dux_dy * dux_dy + dux_dz * dux_dz + duy_dz * duy_dz +
                    duy_dx * duy_dx + dw_dx * dw_dx + dw_dy * dw_dy;
                if (!std::isfinite(deformation)) deformation = 0.0;

                const double rho0_k = (k >= 0 && k < static_cast<int>(rho0_base.size()))
                                          ? rho0_base[k] : 1.0;
                dynamic_pressure[i][j][k] = static_cast<float>(-rho0_k * deformation);

                const double theta_val = theta[i][j][k];
                const double theta_prime = theta_val - dynamics_constants::theta0;
                const double buoyancy = dynamics_constants::g *
                                        (theta_prime / dynamics_constants::theta0);
                buoyancy_pressure[i][j][k] = static_cast<float>(rho0_k * buoyancy);

                p_prime[i][j][k] = dynamic_pressure[i][j][k] + buoyancy_pressure[i][j][k];
            }
        }
    }
}

/**
 * @brief Centered ∂/∂x on the regular Cartesian grid (dx = dr_).
 */
double CartesianScheme::compute_dx(const Field3D& field, int i, int j, int k) const
{
    return (field[i + 1][j][k] - field[i - 1][j][k]) / (2.0 * dr_);
}

/**
 * @brief Centered ∂/∂y on the regular Cartesian grid (dy = dr_).
 */
double CartesianScheme::compute_dy(const Field3D& field, int i, int j, int k) const
{
    return (field[i][j + 1][k] - field[i][j - 1][k]) / (2.0 * dr_);
}

/**
 * @brief Centered ∂/∂z on the regular Cartesian grid.
 */
double CartesianScheme::compute_dz(const Field3D& field, int i, int j, int k) const
{
    return (field[i][j][k + 1] - field[i][j][k - 1]) / (2.0 * dz_);
}
