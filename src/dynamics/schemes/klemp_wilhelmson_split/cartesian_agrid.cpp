#include "core/runtime/simulation_state.hpp"
#include "core/infra/physical_grid_metrics.hpp"
/**
 * @file cartesian_agrid.cpp
 * @brief Implementation of the Cartesian A-grid dynamics scheme.
 *
 * Computes momentum, mass, and pressure tendencies on a regular Cartesian
 * (x, y, z) grid. A uniform horizontal wind on a Cartesian grid produces
 * zero spurious body force, so hydrostatic balance holds for non-axisymmetric
 * hodographs like the WK2002 supercell case.
 *
 *
 * Field naming: the interface uses generic `u`, `v`, `w`. Cartesian
 * functions alias these as `u_x`, `u_y`, `w_field` for clarity in the
 * physics equations. Phase B.1 renamed the interface parameters;
 * local aliases remain for readability.
 *
 * This file is part of the src/dynamics subsystem.
 */

#include "cartesian_agrid.hpp"
#include "cartesian_common.hpp"
#include "core/runtime/process_state.hpp"
#include "core/runtime/runtime_config.hpp"
#include "compute/compute_backend.hpp"
#include "compute/compute_kernel_template.hpp"
#include "numerics/advection/cartesian_mass_flux.hpp"
#include "numerics/runtime.hpp"
#include "terrain/conservative_transport.hpp"
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace
{
bool supports_dynamics_backend(const SimulationStateView& state, DynamicsKernelOp operation)
{
    return state.compute_runtime && state.compute_runtime->backend &&
        state.compute_runtime->backend->dynamics_kernels() &&
        state.compute_runtime->backend->dynamics_kernels()->supports(operation);
}
}

namespace kw_cart = nimbus::dynamics::cartesian;

namespace
{

void validate_collocated_pressure_inputs(
    const Field3D& u, const Field3D& v, const Field3D& w,
    const Field3D& rho, const Field3D& p,
    int nx, int ny, int nz)
{
    const auto matches = [=](const Field3D& field)
    {
        return field.size_r() == nx && field.size_th() == ny
            && field.size_z() == nz;
    };
    if (nx < 4 || ny < 4 || nz < 4
        || !matches(u) || !matches(v) || !matches(w)
        || !matches(rho) || !matches(p))
    {
        throw std::invalid_argument(
            "A-grid acoustic pressure coupling requires matching fields on a grid of at least 4x4x4");
    }
    for (std::size_t n = 0; n < rho.size(); ++n)
    {
        const double rho_n = static_cast<double>(rho.data()[n]);
        if (!std::isfinite(rho_n) || rho_n <= 0.0
            || !std::isfinite(static_cast<double>(p.data()[n]))
            || !std::isfinite(static_cast<double>(u.data()[n]))
            || !std::isfinite(static_cast<double>(v.data()[n]))
            || !std::isfinite(static_cast<double>(w.data()[n])))
        {
            throw std::invalid_argument(
                "A-grid acoustic pressure coupling requires finite velocity/pressure and strictly positive finite density");
        }
    }
}

double collocated_mass_divergence(const SimulationStateView& state, const Field3D& density,
                                  const Field3D& u,
                                  const Field3D& v,
                                  const Field3D& w,
                                  int i, int j, int k,
                                  double dx, double dy, double dz,
                                  const tmv::cartesian_mass_flux::CollocatedPressureCoupling*
                                      pressure_coupling = nullptr,
                                  double frame_u_ms = 0.0,
                                  double frame_v_ms = 0.0)
{
    using tmv::cartesian_mass_flux::x_face;
    using tmv::cartesian_mass_flux::y_face;
    using tmv::cartesian_mass_flux::z_face;
    const double fx_plus = x_face(
        density, u, i, j, k, state.grid.nr, StaggerType::Collocated,
        state.configuration->lateral_bc_west, state.configuration->lateral_bc_east, pressure_coupling,
        frame_u_ms);
    const double fx_minus = x_face(
        density, u, i - 1, j, k, state.grid.nr, StaggerType::Collocated,
        state.configuration->lateral_bc_west, state.configuration->lateral_bc_east, pressure_coupling,
        frame_u_ms);
    const double fy_plus = y_face(
        density, v, i, j, k, state.grid.nth, StaggerType::Collocated,
        state.configuration->lateral_bc_south, state.configuration->lateral_bc_north, pressure_coupling,
        frame_v_ms);
    const double fy_minus = y_face(
        density, v, i, j - 1, k, state.grid.nth, StaggerType::Collocated,
        state.configuration->lateral_bc_south, state.configuration->lateral_bc_north, pressure_coupling,
        frame_v_ms);
    const double fz_plus = z_face(
        density, w, i, j, k, state.grid.nz, StaggerType::Collocated,
        state.configuration->bottom_bc, state.configuration->top_bc, pressure_coupling);
    const double fz_minus = z_face(
        density, w, i, j, k - 1, state.grid.nz, StaggerType::Collocated,
        state.configuration->bottom_bc, state.configuration->top_bc, pressure_coupling);
    return (fx_plus - fx_minus) / dx
         + (fy_plus - fy_minus) / dy
         + (fz_plus - fz_minus) / dz;
}

} // namespace

namespace tmv
{

/**
 * @brief Initializes the Cartesian dynamics scheme by caching the grid
 *        dimensions and horizontal/vertical spacings.
 */
CartesianAGridScheme::CartesianAGridScheme() : CartesianAGridScheme(global_simulation_state()) {}

CartesianAGridScheme::CartesianAGridScheme(const SimulationStateView& state)
    : state_(std::make_shared<SimulationStateView>(state)),
      NR_(state.grid.nr), NTH_(state.grid.nth), NZ_(state.grid.nz),
      dr_(state.grid.dr), dy_(state.grid.dtheta), dz_(state.grid.dz),
      deriv_(std::make_unique<CartesianDerivatives>(state.grid.dr, state.grid.dtheta, state.grid.dz))
{
}

CartesianAGridScheme::CartesianAGridScheme(
    const CartesianAGridScheme& other)
    : state_(other.state_), NR_(other.NR_), NTH_(other.NTH_), NZ_(other.NZ_),
      dr_(other.dr_), dy_(other.dy_), dz_(other.dz_),
      deriv_(std::make_unique<CartesianDerivatives>(
          other.dr_, other.dy_, other.dz_)),
      gpu_p0_(other.gpu_p0_),
      gpu_rho0_(other.gpu_rho0_),
      gpu_qv0_(other.gpu_qv0_),
      gpu_u0_(other.gpu_u0_),
      gpu_v0_(other.gpu_v0_),
      gpu_loading_(other.gpu_loading_),
      coriolis_profile_(other.coriolis_profile_),
      has_coriolis_profile_(other.has_coriolis_profile_)
{
}

void CartesianAGridScheme::bind_execution_context(const SimulationStateView& state)
{
    if (state.grid.nr != NR_ || state.grid.nth != NTH_ || state.grid.nz != NZ_ ||
        state.grid.dr != dr_ || state.grid.dtheta != dy_ || state.grid.dz != dz_)
        throw std::invalid_argument("dynamics operator geometry differs from its level");
    if (!state.configuration || !state.domain_frame || !state.compute_runtime || !state.parity_runtime)
        throw std::invalid_argument("dynamics requires complete level execution ownership");
    if (&state_->grid.nr != &state.grid.nr || &state_->fields.rho != &state.fields.rho ||
        state_->configuration != state.configuration || state_->domain_frame != state.domain_frame ||
        state_->terrain_runtime != state.terrain_runtime || state_->compute_runtime != state.compute_runtime ||
        state_->parity_runtime != state.parity_runtime)
        state_ = std::make_shared<SimulationStateView>(state);
}

std::unique_ptr<DynamicsScheme> CartesianAGridScheme::clone() const
{
    return std::make_unique<CartesianAGridScheme>(*this);
}

void CartesianAGridScheme::invalidate_generation_state_after_domain_recycling() noexcept
{
    has_coriolis_profile_ = false;
    for (std::vector<float>* values : {
             &gpu_p0_, &gpu_rho0_, &gpu_qv0_, &gpu_u0_, &gpu_v0_,
             &gpu_loading_})
        std::fill(values->begin(), values->end(), 0.0f);
}

/**
 * @brief Computes the momentum tendencies for the Cartesian scheme.
 *
 * Cartesian form:
 *   - No centrifugal term (no u_θ²/r).
 *   - No azimuthal coriolis-like coupling (no -u u_θ/r).
 *   - Divergence is straightforward: ∂u/∂x + ∂v/∂y + ∂w/∂z (no 1/r terms).
 *   - Derivatives use literal centered differences with dx = dr_, dy = dy_,
 *     and dz = dz_. There is no axis singularity, so there are no eps guards
 *     and the loop runs over the full interior 1..N-1 in all three dimensions.
 *   - Vertical momentum: dw/dt = -(1/ρ) ∂p/∂z - g + advection.
 */
void CartesianAGridScheme::compute_momentum_tendencies(const Field3D& u,
                                                       const Field3D& v,
                                                       const Field3D& w,
                                                       const Field3D& rho,
                                                       const Field3D& p,
                                                       const Field3D& theta,
                                                       double         dt,
                                                       Field3D&       du_dt,
                                                       Field3D&       dv_dt,
                                                       Field3D&       dw_dt,
                                                       Field3D&       drho_dt,
                                                       Field3D&       dp_dt)
{
    if (tmv::physical_grid_metrics::terrain_is_active(*state_))
        throw std::runtime_error("mapped A-grid dynamics requires the split-explicit operator path");
    const CartesianCoriolisProfile coriolis =
        has_coriolis_profile_ ? coriolis_profile_ : CartesianCoriolisProfile{state_->coriolis_f, 0.0};
    const Field3D& u_x        = u;
    const Field3D& u_y        = v;
    const Field3D& w_field    = w;
    const double   frame_u_ms = (*state_->domain_frame).velocity_ms.u_ms;
    const double   frame_v_ms = (*state_->domain_frame).velocity_ms.v_ms;

    (void)dt;    // Forward-Euler update is done by the runtime coupler,
                 // not the scheme.
    (void)theta; // theta is not used in momentum tendencies: buoyancy is
                 // implicit in (-∂p/∂z/ρ - g). See Journey.md "Bug 3:
                 // Double-counted buoyancy" for why re-introducing an
                 // explicit g·(θ-θ₀)/θ₀ term here is wrong.

    if (frame_u_ms == 0.0 && frame_v_ms == 0.0 &&
        supports_dynamics_backend(*state_, DynamicsKernelOp::CartesianTendencies))
    {
        for (const auto* profile : {&state_->base.p0, &state_->base.rho0, &state_->base.u0, &state_->base.v0})
            if (profile->size() < std::size_t(NZ_))
                throw std::invalid_argument("native collocated tendencies require complete base profiles");
        gpu_p0_.resize(static_cast<std::size_t>(NZ_));
        gpu_rho0_.resize(static_cast<std::size_t>(NZ_));
        const bool has_moisture =
            !state_->fields.qv.empty() && state_->base.qv0.size() >= static_cast<std::size_t>(NZ_);
        gpu_qv0_.resize(has_moisture ? static_cast<std::size_t>(NZ_) : 0);
        for (int k = 0; k < NZ_; ++k)
        {
            gpu_p0_[static_cast<std::size_t>(k)]   = static_cast<float>(state_->base.p0[static_cast<std::size_t>(k)]);
            gpu_rho0_[static_cast<std::size_t>(k)] = static_cast<float>(state_->base.rho0[static_cast<std::size_t>(k)]);
            if (has_moisture)
            {
                gpu_qv0_[static_cast<std::size_t>(k)] =
                    static_cast<float>(state_->base.qv0[static_cast<std::size_t>(k)]);
            }
        }

        const size_t n_total = static_cast<size_t>(NR_) * NTH_ * NZ_;
        kw_cart::fill_hydrometeor_loading_buffer(state_->fields.qc,
                                                 state_->fields.qr,
                                                 state_->fields.qi,
                                                 state_->fields.qs,
                                                 state_->fields.qg,
                                                 state_->fields.qh,
                                                 n_total,
                                                 /*compact_empty=*/false,
                                                 gpu_loading_);

        gpu_u0_.resize(static_cast<std::size_t>(NZ_));
        gpu_v0_.resize(static_cast<std::size_t>(NZ_));
        for (int k = 0; k < NZ_; ++k)
        {
            gpu_u0_[static_cast<std::size_t>(k)] = static_cast<float>(state_->base.u0[static_cast<std::size_t>(k)]);
            gpu_v0_[static_cast<std::size_t>(k)] = static_cast<float>(state_->base.v0[static_cast<std::size_t>(k)]);
        }

        if (state_->compute_runtime->backend->dynamics_kernels()->dispatch_cartesian_tendencies(
                {u.data(),
                 v.data(),
                 w.data(),
                 rho.data(),
                 p.data(),
                 theta.data(),
                 gpu_p0_.data(),
                 gpu_rho0_.data(),
                 gpu_loading_.data(),
                 gpu_u0_.data(),
                 gpu_v0_.data(),
                 has_moisture ? state_->fields.qv.data() : nullptr,
                 has_moisture ? gpu_qv0_.data() : nullptr,
                 du_dt.data(),
                 dv_dt.data(),
                 dw_dt.data(),
                 drho_dt.data(),
                 dp_dt.data(),
                 NR_,
                 NTH_,
                 NZ_,
                 static_cast<float>(dr_),
                 static_cast<float>(dy_),
                 static_cast<float>(dz_),
                 static_cast<float>(dynamics_constants::g),
                 static_cast<float>(dynamics_constants::gamma),
                 static_cast<float>(coriolis.f_at_grid_y0),
                 static_cast<int>(state_->configuration->lateral_bc_west),
                 static_cast<int>(state_->configuration->lateral_bc_east),
                 static_cast<int>(state_->configuration->lateral_bc_south),
                 static_cast<int>(state_->configuration->lateral_bc_north),
                 static_cast<int>(state_->configuration->top_bc),
                 has_moisture,
                 static_cast<float>(coriolis.beta)}))
        {
            return;
        }
    }

    if (state_->compute_runtime && state_->compute_runtime->active_kind != ComputeBackendKind::Cpu &&
        !state_->compute_runtime->fallback_active)
        throw std::runtime_error("native collocated RK tendency dispatch failed");

    // Zero outputs everywhere so the boundary layer is guaranteed zero.
    // The interior loop overwrites indices [1..N-1); faces stay at 0.
    kw_cart::zero_fields(NR_, NTH_, NZ_, du_dt, dv_dt, dw_dt, drho_dt, dp_dt);

#pragma omp parallel for collapse(2)
    for (int i = 1; i < NR_ - 1; ++i)
    {
        for (int j = 1; j < NTH_ - 1; ++j)
        {
            for (int k = 1; k < NZ_ - 1; ++k)
            {
                const double ux           = u_x[i][j][k];
                const double uy           = u_y[i][j][k];
                const double transport_ux = ux - frame_u_ms;
                const double transport_uy = uy - frame_v_ms;
                const double wz           = w_field[i][j][k];
                const double rho_val      = rho[i][j][k];
                const double p_val        = p[i][j][k];
                const double rho_safe     = kw_cart::safe_rho(rho_val);

                // Centered first derivatives.
                const double dux_dx = deriv_->di(u_x, i, j, k);
                const double dux_dy = deriv_->dj(u_x, i, j, k);
                const double dux_dz = deriv_->dk(u_x, i, j, k);

                const double duy_dx = deriv_->di(u_y, i, j, k);
                const double duy_dy = deriv_->dj(u_y, i, j, k);
                const double duy_dz = deriv_->dk(u_y, i, j, k);

                const double dw_dx       = deriv_->di(w_field, i, j, k);
                const double dw_dy       = deriv_->dj(w_field, i, j, k);
                const double dw_dz_local = deriv_->dk(w_field, i, j, k);

                const double dp_dx       = deriv_->di(p, i, j, k);
                const double dp_dy       = deriv_->dj(p, i, j, k);
                const double dp_dz_local = deriv_->dk(p, i, j, k);

                // --- x-momentum: ∂u_x/∂t = -(u·∇)u_x - (1/ρ)∂p/∂x + f(v-v0) ---
                const double advective_x     = -transport_ux * dux_dx - transport_uy * dux_dy - wz * dux_dz;
                const double pressure_grad_x = -dp_dx / rho_safe;
                // Perturbation Coriolis (Rotunno & Klemp 1982):
                // Applied to (v - v0) to avoid invented forces on f-plane.
                const double local_f = coriolis.is_f_plane() ? coriolis.f_at_grid_y0
                                                             : coriolis.at_grid_y((static_cast<double>(j) + 0.5) * dy_);
                const double coriolis_x = local_f * (uy - state_->base.v0[k]);
                const double du_x_val   = advective_x + pressure_grad_x + coriolis_x;
                du_dt[i][j][k]          = static_cast<float>(kw_cart::finite_or_zero(du_x_val));

                // --- y-momentum: ∂u_y/∂t = -(u·∇)u_y - (1/ρ)∂p/∂y - f(u-u0) ---
                const double advective_y     = -transport_ux * duy_dx - transport_uy * duy_dy - wz * duy_dz;
                const double pressure_grad_y = -dp_dy / rho_safe;
                const double coriolis_y      = -local_f * (ux - state_->base.u0[k]);
                const double du_y_val        = advective_y + pressure_grad_y + coriolis_y;
                dv_dt[i][j][k]               = static_cast<float>(kw_cart::finite_or_zero(du_y_val));

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
                // An explicit g·(θ-θ₀)/θ₀ buoyancy term must NOT be added on
                // top because the perturbation-pressure/density form already
                // owns that restoring force.
                const double advective_z = -transport_ux * dw_dx - transport_uy * dw_dy - wz * dw_dz_local;

                // Reference-state centered ∂p₀/∂z (same stencil as dp_dz_local).
                const double dp0_dz      = (kw_cart::stored_base_value(state_->base.p0, k + 1) -
                                            kw_cart::stored_base_value(state_->base.p0, k - 1)) /
                                           (2.0 * dz_);
                const double dp_prime_dz = dp_dz_local - dp0_dz;
                const double rho0_k      = kw_cart::stored_base_value(state_->base.rho0, k);
                const double buoyancy    = -dynamics_constants::g * (rho_val - rho0_k) / rho_safe;

                // Virtual temperature buoyancy: moist air is lighter than
                // dry air at the same theta (Rv/Rd - 1 = 0.608).
                const double moisture_buoyancy =
                    kw_cart::moisture_buoyancy_at(state_->fields.qv, state_->base.qv0, dynamics_constants::g, i, j, k);

                // Precipitation loading (Klemp & Wilhelmson 1978, Bryan &
                // Fritsch 2002): hydrometeor mass acts as ballast opposing
                // the updraft and driving downdrafts.
                double loading = 0.0;
                if (!state_->fields.qc.empty())
                {
                    loading = -dynamics_constants::g * kw_cart::hydrometeor_mixing_ratio_at(state_->fields.qc,
                                                                                            state_->fields.qr,
                                                                                            state_->fields.qi,
                                                                                            state_->fields.qs,
                                                                                            state_->fields.qg,
                                                                                            state_->fields.qh,
                                                                                            i,
                                                                                            j,
                                                                                            k);
                }

                const double dw_val = advective_z - dp_prime_dz / rho_safe + buoyancy + moisture_buoyancy + loading;
                dw_dt[i][j][k]      = static_cast<float>(kw_cart::finite_or_zero(dw_val));

                // --- Conservative mass continuity: d rho/dt = -div(rho u).
                // The face reconstruction is shared with the retained flux
                // history consumed by thermodynamics and tracers. ---
                const double divergence = dux_dx + duy_dy + dw_dz_local;
                const double drho_val   = -collocated_mass_divergence(
                    *state_, rho, u_x, u_y, w_field, i, j, k, dr_, dy_, dz_, nullptr, frame_u_ms, frame_v_ms);
                drho_dt[i][j][k] = static_cast<float>(kw_cart::finite_or_zero(drho_val));

                // --- Pressure tendency: ∂p/∂t = -γ p ∇·u - u·∇p ---
                const double gamma_term  = dynamics_constants::gamma * p_val * divergence;
                const double advection_p = -transport_ux * dp_dx - transport_uy * dp_dy - wz * dp_dz_local;
                const double dp_val      = -gamma_term + advection_p;
                dp_dt[i][j][k]           = static_cast<float>(kw_cart::finite_or_zero(dp_val));
            }
        }
    }
}

// =========================================================================
// Split-explicit acoustic substep methods (Klemp & Wilhelmson 1978)
// =========================================================================

void CartesianAGridScheme::compute_slow_tendencies(const Field3D& u,
                                                   const Field3D& v,
                                                   const Field3D& w,
                                                   const Field3D& rho,
                                                   const Field3D& p,
                                                   const Field3D& theta,
                                                   double /*dt*/,
                                                   Field3D&                              du_dt,
                                                   Field3D&                              dv_dt,
                                                   Field3D&                              dw_dt,
                                                   Field3D&                              drho_dt,
                                                   Field3D&                              dp_dt,
                                                   SlowMomentumAdvectionTendencyOutputs* advection_outputs)
{
    const Field3D& u_x = u;
    const Field3D& u_y = v;
    const Field3D& w_field = w;
    const double frame_u_ms = (*state_->domain_frame).velocity_ms.u_ms;
    const double frame_v_ms = (*state_->domain_frame).velocity_ms.v_ms;
    const CartesianCoriolisProfile coriolis = has_coriolis_profile_
        ? coriolis_profile_
        : CartesianCoriolisProfile{state_->coriolis_f, 0.0};
    const TerrainMetrics3D* terrain_metrics =
        tmv::physical_grid_metrics::terrain_is_active(*state_)
            ? tmv::physical_grid_metrics::active_terrain_metrics(*state_) : nullptr;
    const bool periodic_x =
        state_->configuration->lateral_bc_west == tmv::init::config::LateralBC::Periodic
        && state_->configuration->lateral_bc_east == tmv::init::config::LateralBC::Periodic;
    const bool periodic_y =
        state_->configuration->lateral_bc_south == tmv::init::config::LateralBC::Periodic
        && state_->configuration->lateral_bc_north == tmv::init::config::LateralBC::Periodic;
    if (terrain_metrics != nullptr)
    {
        terrain_transport::validate_geometry(*terrain_metrics);
        if (state_->base.p0.size() != static_cast<std::size_t>(NZ_)
            || state_->base.rho0.size() != static_cast<std::size_t>(NZ_)
            || state_->base.theta0.size() != static_cast<std::size_t>(NZ_)
            || state_->base.u0.size() != static_cast<std::size_t>(NZ_)
            || state_->base.v0.size() != static_cast<std::size_t>(NZ_))
            throw std::invalid_argument(
                "mapped A-grid slow dynamics requires complete reference profiles");
    }

    (void)theta;

    if (advection_outputs != nullptr)
    {
        if (!advection_outputs->complete())
            throw std::invalid_argument(
                "A-grid slow momentum attribution requires all three advection outputs");
        const auto matches = [this](const Field3D& field)
        {
            return field.size_r() == NR_ && field.size_th() == NTH_
                && field.size_z() == NZ_;
        };
        if (!matches(*advection_outputs->du_dt)
            || !matches(*advection_outputs->dv_dt)
            || !matches(*advection_outputs->dw_dt))
        {
            throw std::invalid_argument(
                "A-grid slow momentum attribution requires grid-shaped outputs");
        }
    }

    const bool native_fallback_off =
        (*state_->compute_runtime).active_kind != ComputeBackendKind::Cpu &&
        !(*state_->compute_runtime).fallback_active &&
        !(*state_->compute_runtime).config.allow_fallback;
    const bool native_candidate = terrain_metrics == nullptr
        && frame_u_ms == 0.0 && frame_v_ms == 0.0 &&
        supports_dynamics_backend(*state_, DynamicsKernelOp::CartesianTendencies);
    if (native_candidate)
    {
        if (state_->base.p0.size() < static_cast<std::size_t>(NZ_) ||
            state_->base.rho0.size() < static_cast<std::size_t>(NZ_) ||
            state_->base.u0.size() < static_cast<std::size_t>(NZ_) ||
            state_->base.v0.size() < static_cast<std::size_t>(NZ_))
        {
            throw std::invalid_argument(
                "native A-grid slow tendencies require complete base-state profiles");
        }
        const bool has_moisture =
            !state_->fields.qv.empty() && state_->base.qv0.size() >= static_cast<std::size_t>(NZ_);
        gpu_p0_.resize(static_cast<std::size_t>(NZ_));
        gpu_rho0_.resize(static_cast<std::size_t>(NZ_));
        gpu_u0_.resize(static_cast<std::size_t>(NZ_));
        gpu_v0_.resize(static_cast<std::size_t>(NZ_));
        gpu_qv0_.resize(has_moisture ? static_cast<std::size_t>(NZ_) : 0);
        for (int k = 0; k < NZ_; ++k)
        {
            const auto index = static_cast<std::size_t>(k);
            gpu_p0_[index] = static_cast<float>(state_->base.p0[index]);
            gpu_rho0_[index] = static_cast<float>(state_->base.rho0[index]);
            gpu_u0_[index] = static_cast<float>(state_->base.u0[index]);
            gpu_v0_[index] = static_cast<float>(state_->base.v0[index]);
            if (has_moisture)
                gpu_qv0_[index] = static_cast<float>(state_->base.qv0[index]);
        }
        const std::size_t cells = static_cast<std::size_t>(NR_) * NTH_ * NZ_;
        kw_cart::fill_hydrometeor_loading_buffer(
            state_->fields.qc, state_->fields.qr, state_->fields.qi, state_->fields.qs, state_->fields.qg, state_->fields.qh, cells, false, gpu_loading_);
        const bool dispatched = state_->compute_runtime->backend->dynamics_kernels()->dispatch_cartesian_tendencies({
            u.data(), v.data(), w.data(), rho.data(), p.data(), theta.data(),
            gpu_p0_.data(), gpu_rho0_.data(), gpu_loading_.data(),
            gpu_u0_.data(), gpu_v0_.data(),
            has_moisture ? state_->fields.qv.data() : nullptr,
            has_moisture ? gpu_qv0_.data() : nullptr,
            du_dt.data(), dv_dt.data(), dw_dt.data(),
            drho_dt.data(), dp_dt.data(), NR_, NTH_, NZ_,
            static_cast<float>(dr_), static_cast<float>(dy_),
            static_cast<float>(dz_), static_cast<float>(dynamics_constants::g),
            static_cast<float>(dynamics_constants::gamma),
            static_cast<float>(coriolis.f_at_grid_y0),
            static_cast<int>(state_->configuration->lateral_bc_west),
            static_cast<int>(state_->configuration->lateral_bc_east),
            static_cast<int>(state_->configuration->lateral_bc_south),
            static_cast<int>(state_->configuration->lateral_bc_north),
            static_cast<int>(state_->configuration->top_bc), has_moisture,
            static_cast<float>(coriolis.beta), true});
        if (dispatched)
        {
            if (advection_outputs != nullptr)
            {
                advection_outputs->du_dt->fill(0.0f);
                advection_outputs->dv_dt->fill(0.0f);
                advection_outputs->dw_dt->fill(0.0f);
                #pragma omp parallel for collapse(2)
                for (int i = 1; i < NR_ - 1; ++i)
                    for (int j = 1; j < NTH_ - 1; ++j)
                        for (int k = 1; k < NZ_ - 1; ++k)
                        {
                            const double local_f = coriolis.is_f_plane()
                                ? coriolis.f_at_grid_y0
                                : coriolis.at_grid_y(
                                      (static_cast<double>(j) + 0.5) * dy_);
                            const double rho_value = rho[i][j][k];
                            const double rho_safe = kw_cart::safe_rho(rho_value);
                            const double buoyancy = -dynamics_constants::g *
                                (rho_value - state_->base.rho0[static_cast<std::size_t>(k)]) /
                                rho_safe;
                            const double moisture = kw_cart::moisture_buoyancy_at(
                                state_->fields.qv, state_->base.qv0, dynamics_constants::g, i, j, k);
                            const double loading = state_->fields.qc.empty() ? 0.0 :
                                -dynamics_constants::g *
                                kw_cart::hydrometeor_mixing_ratio_at(
                                    state_->fields.qc, state_->fields.qr, state_->fields.qi, state_->fields.qs, state_->fields.qg, state_->fields.qh, i, j, k);
                            (*advection_outputs->du_dt)[i][j][k] =
                                du_dt[i][j][k] - static_cast<float>(
                                    local_f * (v[i][j][k] - state_->base.v0[k]));
                            (*advection_outputs->dv_dt)[i][j][k] =
                                dv_dt[i][j][k] + static_cast<float>(
                                    local_f * (u[i][j][k] - state_->base.u0[k]));
                            (*advection_outputs->dw_dt)[i][j][k] =
                                dw_dt[i][j][k] - static_cast<float>(
                                    buoyancy + moisture + loading);
                        }
            }
            return;
        }
        if (native_fallback_off)
            throw std::runtime_error(
                "fallback-off native A-grid slow tendency dispatch failed");
    }
    else if (native_fallback_off)
    {
        throw std::runtime_error(
            "fallback-off native A-grid slow tendencies are unavailable for this profile");
    }

    // Zero all outputs (boundaries stay zero).
    kw_cart::zero_fields(NR_, NTH_, NZ_, du_dt, dv_dt, dw_dt, drho_dt, dp_dt);
    if (advection_outputs != nullptr)
    {
        advection_outputs->du_dt->fill(0.0f);
        advection_outputs->dv_dt->fill(0.0f);
        advection_outputs->dw_dt->fill(0.0f);
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
                const double transport_ux = ux - frame_u_ms;
                const double transport_uy = uy - frame_v_ms;
                const double wz = w_field[i][j][k];
                const double rho_val = rho[i][j][k];
                const double rho_safe = kw_cart::safe_rho(rho_val);

                const auto u_gradient = terrain_metrics != nullptr
                    ? terrain_transport::agrid_scalar_gradient_center(
                        *terrain_metrics, u_x, i, j, k,
                        periodic_x, periodic_y)
                    : terrain_transport::PhysicalVector3{
                        deriv_->di(u_x, i, j, k),
                        deriv_->dj(u_x, i, j, k),
                        deriv_->dk(u_x, i, j, k)};
                const auto v_gradient = terrain_metrics != nullptr
                    ? terrain_transport::agrid_scalar_gradient_center(
                        *terrain_metrics, u_y, i, j, k,
                        periodic_x, periodic_y)
                    : terrain_transport::PhysicalVector3{
                        deriv_->di(u_y, i, j, k),
                        deriv_->dj(u_y, i, j, k),
                        deriv_->dk(u_y, i, j, k)};
                const auto w_gradient = terrain_metrics != nullptr
                    ? terrain_transport::agrid_scalar_gradient_center(
                        *terrain_metrics, w_field, i, j, k,
                        periodic_x, periodic_y)
                    : terrain_transport::PhysicalVector3{
                        deriv_->di(w_field, i, j, k),
                        deriv_->dj(w_field, i, j, k),
                        deriv_->dk(w_field, i, j, k)};
                const auto p_gradient = terrain_metrics != nullptr
                    ? terrain_transport::agrid_pressure_perturbation_gradient_center(
                        *terrain_metrics, p, state_->base.p0, i, j, k,
                        periodic_x, periodic_y)
                    : terrain_transport::PhysicalVector3{
                        deriv_->di(p, i, j, k),
                        deriv_->dj(p, i, j, k),
                        deriv_->dk(p, i, j, k)};

                // x-momentum: advection + Coriolis (slow terms, no pressure gradient)
                const double advection_x =
                    -transport_ux * u_gradient.x
                    - transport_uy * u_gradient.y - wz * u_gradient.z;
                const double local_f = coriolis.is_f_plane()
                    ? coriolis.f_at_grid_y0
                    : coriolis.at_grid_y((static_cast<double>(j) + 0.5) * dy_);
                const double du_x_val = advection_x
                                      + local_f * (uy - state_->base.v0[k]);
                du_dt[i][j][k] = static_cast<float>(kw_cart::finite_or_zero(du_x_val));

                // y-momentum: advection + Coriolis (slow terms, no pressure gradient)
                const double advection_y =
                    -transport_ux * v_gradient.x
                    - transport_uy * v_gradient.y - wz * v_gradient.z;
                const double du_y_val = advection_y
                                      - local_f * (ux - state_->base.u0[k]);
                dv_dt[i][j][k] = static_cast<float>(kw_cart::finite_or_zero(du_y_val));

                // z-momentum: advection + buoyancy + moisture buoyancy + loading (no pressure gradient)
                const double advective_z = -transport_ux * w_gradient.x
                                          - transport_uy * w_gradient.y
                                          - wz * w_gradient.z;
                const double rho0_k = terrain_metrics != nullptr
                    ? terrain_transport::reference_density_fp32(
                        state_->base.p0, state_->base.theta0,
                        terrain_metrics->z(i, j, k), dz_)
                    : kw_cart::stored_base_value(state_->base.rho0, k);
                const double buoyancy = -dynamics_constants::g * (rho_val - rho0_k) / rho_safe;
                // Virtual temperature buoyancy: moist air is lighter than
                // dry air at the same theta (Rv/Rd - 1 = 0.608).
                const double moisture_buoyancy =
                    terrain_metrics != nullptr && !state_->fields.qv.empty()
                    ? dynamics_constants::g * 0.608 * (
                        static_cast<double>(state_->fields.qv[i][j][k])
                        - terrain_transport::reference_profile_value_fp32(
                            state_->base.qv0, terrain_metrics->z(i, j, k), dz_))
                    : kw_cart::moisture_buoyancy_at(
                        state_->fields.qv, state_->base.qv0, dynamics_constants::g, i, j, k);
                double loading = 0.0;
                if (!state_->fields.qc.empty())
                {
                    loading = -dynamics_constants::g * kw_cart::hydrometeor_mixing_ratio_at(
                        state_->fields.qc, state_->fields.qr, state_->fields.qi, state_->fields.qs, state_->fields.qg, state_->fields.qh, i, j, k);
                }
                const double dw_val = advective_z + buoyancy + moisture_buoyancy + loading;
                dw_dt[i][j][k] = static_cast<float>(kw_cart::finite_or_zero(dw_val));
                if (advection_outputs != nullptr)
                {
                    (*advection_outputs->du_dt)[i][j][k] =
                        static_cast<float>(kw_cart::finite_or_zero(advection_x));
                    (*advection_outputs->dv_dt)[i][j][k] =
                        static_cast<float>(kw_cart::finite_or_zero(advection_y));
                    (*advection_outputs->dw_dt)[i][j][k] =
                        static_cast<float>(kw_cart::finite_or_zero(advective_z));
                }

                // density: no slow tendency (all continuity is fast)
                drho_dt[i][j][k] = 0.0f;

                // pressure: advection only (no compression)
                const double dp_val = -transport_ux * p_gradient.x
                                    - transport_uy * p_gradient.y
                                    - wz * p_gradient.z;
                dp_dt[i][j][k] = static_cast<float>(kw_cart::finite_or_zero(dp_val));
            }
        }
    }
}

void CartesianAGridScheme::compute_fast_pressure_tendencies(
    const Field3D& u, const Field3D& v, const Field3D& w,
    const Field3D& rho, const Field3D& p,
    double dt_small,
    Field3D& drho_dt, Field3D& dp_dt)
{
    // Aliases removed in Phase B field rename — body now uses u, v, w directly.
    const Field3D& u_x = u;
    const Field3D& u_y = v;
    const Field3D& w_field = w;
    const double frame_u_ms = (*state_->domain_frame).velocity_ms.u_ms;
    const double frame_v_ms = (*state_->domain_frame).velocity_ms.v_ms;
    const TerrainMetrics3D* terrain_metrics =
        tmv::physical_grid_metrics::terrain_is_active(*state_)
            ? tmv::physical_grid_metrics::active_terrain_metrics(*state_) : nullptr;

    if (!std::isfinite(dt_small) || dt_small <= 0.0)
    {
        throw std::invalid_argument(
            "A-grid acoustic pressure coupling requires finite dt_small > 0");
    }
    if (!std::isfinite(dr_) || dr_ <= 0.0
        || !std::isfinite(dy_) || dy_ <= 0.0
        || !std::isfinite(dz_) || dz_ <= 0.0
        || static_cast<int>(state_->base.p0.size()) < NZ_)
    {
        throw std::invalid_argument(
            "A-grid acoustic pressure coupling requires positive grid spacing and a complete base-pressure profile");
    }
    for (int k = 0; k < NZ_; ++k)
    {
        if (!std::isfinite(state_->base.p0[static_cast<std::size_t>(k)]))
        {
            throw std::invalid_argument(
                "A-grid acoustic pressure coupling requires a finite base-pressure profile");
        }
    }
    validate_collocated_pressure_inputs(
        u_x, u_y, w_field, rho, p, NR_, NTH_, NZ_);
    const cartesian_mass_flux::CollocatedPressureCoupling pressure_coupling{
        &p, &state_->base.p0, dt_small, dr_, dy_, dz_};
    const terrain_transport::mass_flux_detail::CollocatedPressureCouplingView
        mapped_pressure_coupling{&p, &state_->base.p0, dt_small};
    if (terrain_metrics != nullptr)
        terrain_transport::validate_geometry(*terrain_metrics);

    // Zero outputs.
    kw_cart::zero_fields(NR_, NTH_, NZ_, drho_dt, dp_dt);

    #pragma omp parallel for collapse(2)
    for (int i = 1; i < NR_ - 1; ++i)
    {
        for (int j = 1; j < NTH_ - 1; ++j)
        {
            for (int k = 1; k < NZ_ - 1; ++k)
            {
                const double p_val = p[i][j][k];

                const double divergence = terrain_metrics != nullptr
                    ? terrain_transport::agrid_velocity_divergence(
                        *terrain_metrics, u_x, u_y, w_field, i, j, k,
                        state_->configuration->lateral_bc_west, state_->configuration->lateral_bc_east,
                        state_->configuration->lateral_bc_south, state_->configuration->lateral_bc_north,
                        state_->configuration->bottom_bc, state_->configuration->top_bc,
                        frame_u_ms, frame_v_ms)
                    : deriv_->di(u_x, i, j, k)
                        + deriv_->dj(u_y, i, j, k)
                        + deriv_->dk(w_field, i, j, k);

                // Conservative continuity using the exact face operator
                // sampled by CartesianCGridTransportContext.
                const double drho_val = terrain_metrics != nullptr
                    ? -terrain_transport::agrid_mass_divergence(
                        *terrain_metrics, rho, u_x, u_y, w_field,
                        i, j, k,
                        state_->configuration->lateral_bc_west, state_->configuration->lateral_bc_east,
                        state_->configuration->lateral_bc_south, state_->configuration->lateral_bc_north,
                        state_->configuration->bottom_bc, state_->configuration->top_bc,
                        &mapped_pressure_coupling,
                        frame_u_ms, frame_v_ms)
                    : -collocated_mass_divergence(*state_,
                        rho, u_x, u_y, w_field, i, j, k,
                        dr_, dy_, dz_, &pressure_coupling,
                        frame_u_ms, frame_v_ms);
                drho_dt[i][j][k] = static_cast<float>(kw_cart::finite_or_zero(drho_val));

                // Compression: dp/dt = -γ p ∇·u
                const double dp_val = -dynamics_constants::gamma * p_val * divergence;
                dp_dt[i][j][k] = static_cast<float>(kw_cart::finite_or_zero(dp_val));
            }
        }
    }
}

void CartesianAGridScheme::compute_fast_momentum_tendencies(
    const Field3D& /*u*/, const Field3D& /*v*/, const Field3D& /*w*/,
    const Field3D& rho, const Field3D& p,
    double /*dt_small*/,
    Field3D& du_dt, Field3D& dv_dt, Field3D& dw_dt)
{
    const TerrainMetrics3D* terrain_metrics =
        tmv::physical_grid_metrics::terrain_is_active(*state_)
            ? tmv::physical_grid_metrics::active_terrain_metrics(*state_) : nullptr;
    const bool periodic_x =
        state_->configuration->lateral_bc_west == tmv::init::config::LateralBC::Periodic
        && state_->configuration->lateral_bc_east == tmv::init::config::LateralBC::Periodic;
    const bool periodic_y =
        state_->configuration->lateral_bc_south == tmv::init::config::LateralBC::Periodic
        && state_->configuration->lateral_bc_north == tmv::init::config::LateralBC::Periodic;
    if (terrain_metrics != nullptr)
    {
        terrain_transport::validate_geometry(*terrain_metrics);
        if (state_->base.p0.size() != static_cast<std::size_t>(NZ_))
            throw std::invalid_argument(
                "mapped A-grid fast momentum requires a complete reference pressure");
    }
    // Zero outputs.
    kw_cart::zero_fields(NR_, NTH_, NZ_, du_dt, dv_dt, dw_dt);

    #pragma omp parallel for collapse(2)
    for (int i = 1; i < NR_ - 1; ++i)
    {
        for (int j = 1; j < NTH_ - 1; ++j)
        {
            for (int k = 1; k < NZ_ - 1; ++k)
            {
                const double rho_val = rho[i][j][k];
                const double rho_safe = kw_cart::safe_rho(rho_val);

                const auto pressure_gradient = terrain_metrics != nullptr
                    ? terrain_transport::agrid_pressure_perturbation_gradient_center(
                        *terrain_metrics, p, state_->base.p0, i, j, k,
                        periodic_x, periodic_y)
                    : terrain_transport::PhysicalVector3{
                        deriv_->di(p, i, j, k),
                        deriv_->dj(p, i, j, k),
                        deriv_->dk(p, i, j, k)};

                // x-momentum: pressure gradient only
                const double du_x_val = -pressure_gradient.x / rho_safe;
                du_dt[i][j][k] = static_cast<float>(kw_cart::finite_or_zero(du_x_val));

                // y-momentum: pressure gradient only
                const double du_y_val = -pressure_gradient.y / rho_safe;
                dv_dt[i][j][k] = static_cast<float>(kw_cart::finite_or_zero(du_y_val));

                // z-momentum: perturbation pressure gradient only (no buoyancy)
                const double dp_prime_dz = terrain_metrics != nullptr
                    ? pressure_gradient.z
                    : pressure_gradient.z - (
                        kw_cart::stored_base_value(state_->base.p0, k + 1)
                        - kw_cart::stored_base_value(state_->base.p0, k - 1))
                        / (2.0 * dz_);
                const double dw_val = -dp_prime_dz / rho_safe;
                dw_dt[i][j][k] = static_cast<float>(kw_cart::finite_or_zero(dw_val));
            }
        }
    }
}

// Derivative operators moved to DerivativeOperators (Phase B.2).
// CartesianDerivatives is constructed in CartesianAGridScheme::CartesianAGridScheme().

} // namespace tmv
