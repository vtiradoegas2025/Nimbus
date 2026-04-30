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
 *
 * Field naming: the interface uses generic `u`, `v`, `w`. Cartesian
 * functions alias these as `u_x`, `u_y`, `w_field` for clarity in the
 * physics equations. Phase B.1 renamed the interface parameters;
 * local aliases remain for readability.
 *
 * This file is part of the src/dynamics subsystem.
 */

#include "cartesian.hpp"
#include "core/simulation.hpp"
#include "compute/compute_kernel_template.hpp"
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
      dr_(dr), dz_(dz),
      deriv_(std::make_unique<CartesianDerivatives>(dr, dr, dz))
{
}

/**
 * @brief Computes the momentum tendencies for the Cartesian scheme.
 *
 * Differences from the cylindrical scheme:
 *   - No centrifugal term (no u_θ²/r).
 *   - No azimuthal coriolis-like coupling (no -u u_θ/r).
 *   - Divergence is straightforward: ∂u/∂x + ∂v/∂y + ∂w/∂z (no 1/r terms).
 *   - Derivatives use literal centered differences with dx = dy = dr_
 *     and dz = dz_. There is no axis singularity, so the eps guards are
 *     removed and the loop runs over the full interior 1..N-1 in all
 *     three dimensions.
 *   - The vertical momentum equation is reused verbatim from the cylindrical
 *     schemes (post-Bug-3): dw/dt = -(1/ρ) ∂p/∂z - g + advection.
 */
void CartesianScheme::compute_momentum_tendencies(
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
    Field3D& dp_dt)
{
    // Aliases removed in Phase B field rename — body now uses u, v, w directly.
    const Field3D& u_x = u;
    const Field3D& u_y = v;
    const Field3D& w_field = w;

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

        // Precompute total hydrometeor loading for GPU dispatch.
        const size_t n_total = static_cast<size_t>(NR_) * NTH_ * NZ_;
        std::vector<float> loading_buf(n_total, 0.0f);
        if (!qc.empty())
        {
            for (size_t idx = 0; idx < n_total; ++idx)
            {
                loading_buf[idx] = qc.data()[idx] + qr.data()[idx] +
                                   qi.data()[idx] + qs.data()[idx] +
                                   qg.data()[idx] + qh.data()[idx];
            }
        }

        // Base-state wind profiles for perturbation Coriolis.
        std::vector<float> u0f(static_cast<size_t>(NZ_));
        std::vector<float> v0f(static_cast<size_t>(NZ_));
        for (int k = 0; k < NZ_; ++k)
        {
            u0f[k] = static_cast<float>(u0_base[k]);
            v0f[k] = static_cast<float>(v0_base[k]);
        }

        if (dispatch_cartesian_tendencies_backend(
                u.data(), v.data(), w.data(),
                rho.data(), p.data(), theta.data(),
                p0f.data(), rho0f.data(),
                loading_buf.data(),
                u0f.data(), v0f.data(),
                du_dt.data(), dv_dt.data(), dw_dt.data(),
                drho_dt.data(), dp_dt.data(),
                NR_, NTH_, NZ_,
                static_cast<float>(dr_), static_cast<float>(dr_),
                static_cast<float>(dz_),
                static_cast<float>(dynamics_constants::g),
                static_cast<float>(dynamics_constants::gamma),
                static_cast<float>(coriolis_f)))
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
                du_dt[i][j][k] = 0.0f;
                dv_dt[i][j][k] = 0.0f;
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
                const double dux_dx = deriv_->di(u_x, i, j, k);
                const double dux_dy = deriv_->dj(u_x, i, j, k);
                const double dux_dz = deriv_->dk(u_x, i, j, k);

                const double duy_dx = deriv_->di(u_y, i, j, k);
                const double duy_dy = deriv_->dj(u_y, i, j, k);
                const double duy_dz = deriv_->dk(u_y, i, j, k);

                const double dw_dx = deriv_->di(w_field, i, j, k);
                const double dw_dy = deriv_->dj(w_field, i, j, k);
                const double dw_dz_local = deriv_->dk(w_field, i, j, k);

                const double dp_dx = deriv_->di(p, i, j, k);
                const double dp_dy = deriv_->dj(p, i, j, k);
                const double dp_dz_local = deriv_->dk(p, i, j, k);

                // --- x-momentum: ∂u_x/∂t = -(u·∇)u_x - (1/ρ)∂p/∂x + f(v-v0) ---
                const double advective_x = -ux * dux_dx - uy * dux_dy - wz * dux_dz;
                const double pressure_grad_x = -dp_dx / rho_safe;
                // Perturbation Coriolis (Rotunno & Klemp 1982):
                // Applied to (v - v0) to avoid invented forces on f-plane.
                const double coriolis_x = coriolis_f * (uy - v0_base[k]);
                double du_x_val = advective_x + pressure_grad_x + coriolis_x;
                if (!std::isfinite(du_x_val)) du_x_val = 0.0;
                du_dt[i][j][k] = static_cast<float>(du_x_val);

                // --- y-momentum: ∂u_y/∂t = -(u·∇)u_y - (1/ρ)∂p/∂y - f(u-u0) ---
                const double advective_y = -ux * duy_dx - uy * duy_dy - wz * duy_dz;
                const double pressure_grad_y = -dp_dy / rho_safe;
                const double coriolis_y = -coriolis_f * (ux - u0_base[k]);
                double du_y_val = advective_y + pressure_grad_y + coriolis_y;
                if (!std::isfinite(du_y_val)) du_y_val = 0.0;
                dv_dt[i][j][k] = static_cast<float>(du_y_val);

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

                // Virtual temperature buoyancy: moist air is lighter than
                // dry air at the same theta (Rv/Rd - 1 = 0.608).
                // Virtual temperature buoyancy: moist air is lighter than
                // dry air at the same theta (Rv/Rd - 1 = 0.608).
                double moisture_buoyancy = 0.0;
                if (!qv.empty() && !qv0_base.empty())
                {
                    const double qv_val = static_cast<double>(qv[i][j][k]);
                    const double qv0_k = qv0_base[k];
                    moisture_buoyancy = dynamics_constants::g * 0.608 * (qv_val - qv0_k);
                }

                // Precipitation loading (Klemp & Wilhelmson 1978, Bryan &
                // Fritsch 2002): hydrometeor mass acts as ballast opposing
                // the updraft and driving downdrafts.
                double loading = 0.0;
                if (!qc.empty())
                {
                    loading = -dynamics_constants::g *
                        (static_cast<double>(qc[i][j][k]) + static_cast<double>(qr[i][j][k]) +
                         static_cast<double>(qi[i][j][k]) + static_cast<double>(qs[i][j][k]) +
                         static_cast<double>(qg[i][j][k]) + static_cast<double>(qh[i][j][k]));
                }

                double dw_val = advective_z - dp_prime_dz / rho_safe + buoyancy + moisture_buoyancy + loading;
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
    const Field3D& u, const Field3D& v, const Field3D& w,
    const Field3D& rho, const Field3D& p, const Field3D& theta,
    double /*dt*/,
    Field3D& du_dt, Field3D& dv_dt, Field3D& dw_dt,
    Field3D& drho_dt, Field3D& dp_dt)
{
    const Field3D& u_x = u;
    const Field3D& u_y = v;
    const Field3D& w_field = w;

    (void)theta;

    // Zero all outputs (boundaries stay zero).
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < NR_; ++i)
        for (int j = 0; j < NTH_; ++j)
            for (int k = 0; k < NZ_; ++k)
            {
                du_dt[i][j][k] = 0.0f;
                dv_dt[i][j][k] = 0.0f;
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
                const double dux_dx = deriv_->di(u_x, i, j, k);
                const double dux_dy = deriv_->dj(u_x, i, j, k);
                const double dux_dz = deriv_->dk(u_x, i, j, k);
                const double duy_dx = deriv_->di(u_y, i, j, k);
                const double duy_dy = deriv_->dj(u_y, i, j, k);
                const double duy_dz = deriv_->dk(u_y, i, j, k);
                const double dw_dx = deriv_->di(w_field, i, j, k);
                const double dw_dy = deriv_->dj(w_field, i, j, k);
                const double dw_dz_local = deriv_->dk(w_field, i, j, k);

                // Pressure derivatives for advection of pressure.
                const double dp_dx = deriv_->di(p, i, j, k);
                const double dp_dy = deriv_->dj(p, i, j, k);
                const double dp_dz_local = deriv_->dk(p, i, j, k);

                // x-momentum: advection + Coriolis (slow terms, no pressure gradient)
                double du_x_val = -ux * dux_dx - uy * dux_dy - wz * dux_dz
                                  + coriolis_f * (uy - v0_base[k]);
                if (!std::isfinite(du_x_val)) du_x_val = 0.0;
                du_dt[i][j][k] = static_cast<float>(du_x_val);

                // y-momentum: advection + Coriolis (slow terms, no pressure gradient)
                double du_y_val = -ux * duy_dx - uy * duy_dy - wz * duy_dz
                                  - coriolis_f * (ux - u0_base[k]);
                if (!std::isfinite(du_y_val)) du_y_val = 0.0;
                dv_dt[i][j][k] = static_cast<float>(du_y_val);

                // z-momentum: advection + buoyancy + moisture buoyancy + loading (no pressure gradient)
                const double advective_z = -ux * dw_dx - uy * dw_dy - wz * dw_dz_local;
                const double rho0_k = rho0_base[k];
                const double buoyancy = -dynamics_constants::g * (rho_val - rho0_k) / rho_safe;
                // Virtual temperature buoyancy: moist air is lighter than
                // dry air at the same theta (Rv/Rd - 1 = 0.608).
                double moisture_buoyancy = 0.0;
                if (!qv.empty() && !qv0_base.empty())
                {
                    const double qv_val = static_cast<double>(qv[i][j][k]);
                    const double qv0_k = qv0_base[k];
                    moisture_buoyancy = dynamics_constants::g * 0.608 * (qv_val - qv0_k);
                }
                double loading = 0.0;
                if (!qc.empty())
                {
                    loading = -dynamics_constants::g *
                        (static_cast<double>(qc[i][j][k]) + static_cast<double>(qr[i][j][k]) +
                         static_cast<double>(qi[i][j][k]) + static_cast<double>(qs[i][j][k]) +
                         static_cast<double>(qg[i][j][k]) + static_cast<double>(qh[i][j][k]));
                }
                double dw_val = advective_z + buoyancy + moisture_buoyancy + loading;
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
    const Field3D& u, const Field3D& v, const Field3D& w,
    const Field3D& rho, const Field3D& p,
    Field3D& drho_dt, Field3D& dp_dt)
{
    // Aliases removed in Phase B field rename — body now uses u, v, w directly.
    const Field3D& u_x = u;
    const Field3D& u_y = v;
    const Field3D& w_field = w;

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

                const double dux_dx = deriv_->di(u_x, i, j, k);
                const double duy_dy = deriv_->dj(u_y, i, j, k);
                const double dw_dz_local = deriv_->dk(w_field, i, j, k);
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
    const Field3D& /*u*/, const Field3D& /*v*/, const Field3D& /*w*/,
    const Field3D& rho, const Field3D& p,
    Field3D& du_dt, Field3D& dv_dt, Field3D& dw_dt)
{
    // Zero outputs.
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < NR_; ++i)
        for (int j = 0; j < NTH_; ++j)
            for (int k = 0; k < NZ_; ++k)
            {
                du_dt[i][j][k] = 0.0f;
                dv_dt[i][j][k] = 0.0f;
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

                const double dp_dx = deriv_->di(p, i, j, k);
                const double dp_dy = deriv_->dj(p, i, j, k);
                const double dp_dz_local = deriv_->dk(p, i, j, k);

                // x-momentum: pressure gradient only
                double du_x_val = -dp_dx / rho_safe;
                if (!std::isfinite(du_x_val)) du_x_val = 0.0;
                du_dt[i][j][k] = static_cast<float>(du_x_val);

                // y-momentum: pressure gradient only
                double du_y_val = -dp_dy / rho_safe;
                if (!std::isfinite(du_y_val)) du_y_val = 0.0;
                dv_dt[i][j][k] = static_cast<float>(du_y_val);

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
    Field3D& baroclinic_term)
{
    // Aliases removed in Phase B field rename — body now uses u, v, w directly.
    const Field3D& u_x = u;
    const Field3D& u_y = v;
    const Field3D& w_field = w;

    #pragma omp parallel for collapse(2)
    for (int i = 1; i < NR_ - 1; ++i)
    {
        for (int j = 1; j < NTH_ - 1; ++j)
        {
            for (int k = 1; k < NZ_ - 1; ++k)
            {
                const double duy_dx = deriv_->di(u_y, i, j, k);
                const double dux_dy = deriv_->dj(u_x, i, j, k);
                const double dux_dz = deriv_->dk(u_x, i, j, k);
                const double duy_dz = deriv_->dk(u_y, i, j, k);
                const double dw_dx = deriv_->di(w_field, i, j, k);
                const double dw_dy = deriv_->dj(w_field, i, j, k);
                const double dw_dz_local = deriv_->dk(w_field, i, j, k);

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

                const double drho_dx = deriv_->di(rho, i, j, k);
                const double drho_dy = deriv_->dj(rho, i, j, k);
                const double dp_dx = deriv_->di(p, i, j, k);
                const double dp_dy = deriv_->dj(p, i, j, k);

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
    const Field3D& u,
    const Field3D& v,
    const Field3D& w,
    const Field3D& rho,
    const Field3D& theta,
    Field3D& p_prime,
    Field3D& dynamic_pressure,
    Field3D& buoyancy_pressure)
{
    (void)rho;  // rho is not needed here; we use the 1D rho0_base profile
                // for consistency with the cylindrical schemes.

    // Aliases removed in Phase B field rename — body now uses u, v, w directly.
    const Field3D& u_x = u;
    const Field3D& u_y = v;
    const Field3D& w_field = w;

    #pragma omp parallel for collapse(2)
    for (int i = 1; i < NR_ - 1; ++i)
    {
        for (int j = 1; j < NTH_ - 1; ++j)
        {
            for (int k = 1; k < NZ_ - 1; ++k)
            {
                const double dux_dx = deriv_->di(u_x, i, j, k);
                const double dux_dy = deriv_->dj(u_x, i, j, k);
                const double dux_dz = deriv_->dk(u_x, i, j, k);
                const double duy_dx = deriv_->di(u_y, i, j, k);
                const double duy_dy = deriv_->dj(u_y, i, j, k);
                const double duy_dz = deriv_->dk(u_y, i, j, k);
                const double dw_dx = deriv_->di(w_field, i, j, k);
                const double dw_dy = deriv_->dj(w_field, i, j, k);
                const double dw_dz_local = deriv_->dk(w_field, i, j, k);

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

// Derivative operators moved to DerivativeOperators (Phase B.2).
// CartesianDerivatives is constructed in CartesianScheme::CartesianScheme().
