#include "core/runtime/simulation_state.hpp"
#include "core/infra/physical_grid_metrics.hpp"
#include "core/field/field_pool.hpp"
/**
 * @file cartesian_cgrid_3d.cpp
 * @brief Implementation of the Cartesian C-grid (fully 3D) dynamics scheme.
 *
 * Field placement (Arakawa C-grid):
 *   u[i][j][k] (u_x) at (x_face[i], y[j],      z[k])
 *   v[i][j][k] (u_y) at (x[i],      y_face[j], z[k])
 *   w[i][j][k]       at (x[i],      y[j],      z_face[k])
 *   scalars         at (x[i],       y[j],      z[k])
 *
 * Loop ranges (interior only; the lateral/vertical boundary columns are
 * written by boundary_conditions_cartesian_cgrid after each integration step):
 *   du_x/dt at x-face:   i = 1..NX-2, j = 1..NY-2, k = 0..NZ-2
 *   du_y/dt at y-face:   i = 1..NX-2, j = 1..NY-2, k = 0..NZ-2
 *     k=0 is prognostic (SCIENCE.md M19): the ground enters via the
 *     implicit zero-normal-flux face and a free-slip mirror ghost; RigidNoSlip
 *     pins u[0]=v[0]=0 in the BC pass instead.
 *   dw/dt   at z-face:   i = 1..NX-2, j = 1..NY-2, k = 0..NZ-2
 *     k=0 is INTERIOR on the C-grid; the rigid ground normal flux is implicit
 *     in div_flux_z; k=NZ-1 is the lid (held by the BC module).
 *   drho/dt, dp/dt at cell center: continuity over i=1..NX-2, j=1..NY-2, all k
 *     (div_flux_mass_z closes the ground/lid faces); compression likewise.
 *
 * Vertical momentum uses reference-state subtraction
 *   dw/dt = -(1/rho)(dp/dz - dp0/dz) - g*(rho - rho0)/rho + advection (+moist)
 * with dp/dz and dp0/dz taken on the SAME one-sided z-face stencil so that at
 * the discrete hydrostatic state p == p0(z), rho == rho0(z) the vertical
 * pressure perturbation is bit-exactly zero.
 *
 * Generalized to distinct dx/dy and the global base state, with moisture
 * buoyancy/loading added so moist supercells run. No 1/r, no axis, no patch.
 */

#include "cartesian_cgrid_3d.hpp"
#include "cartesian_common.hpp"
#include "core/runtime/process_state.hpp"
#include "core/runtime/runtime_config.hpp"
#include "core/field/field_bounds.hpp"
#include "core/field/field_pool.hpp"    // parity-replay scratch fields
#include "dynamics/dynamics_base.hpp"   // dynamics_constants
#include "dynamics/eos_pressure.hpp"
#include "diagnostics/vorticity_budget_kernel.hpp"  // cartesian_cgrid_vorticity_budget_at (shared)
#include "compute/compute_backend.hpp"
#include "compute/conservative_momentum_contract.hpp"
#include "compute/compute_kernel_template.hpp"  // dispatch_cartesian_cgrid_slow_tendencies_backend
#include "compute/parity_check.hpp"     // GPU.md Phase 1 harness
#include "numerics/runtime.hpp"
#include "numerics/advection/cartesian_mass_flux.hpp"
#include "terrain/conservative_transport.hpp"
#include "terrain/native_mapped_transaction.hpp"
#include <cmath>
#include <limits>
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

namespace tmv
{

namespace
{

double flat_cgrid_mass_divergence_in_frame(const SimulationStateView& state,
    const Field3D& rho, const Field3D& u, const Field3D& v, const Field3D& w,
    int nx, int ny, int nz, double dx, double dy, double dz,
    int i, int j, int k, double frame_u_ms, double frame_v_ms)
{
    using tmv::cartesian_mass_flux::x_face;
    using tmv::cartesian_mass_flux::y_face;
    using tmv::cartesian_mass_flux::z_face;
    const double fx_plus = x_face(
        rho, u, i, j, k, nx, StaggerType::CGrid,
        state.configuration->lateral_bc_west, state.configuration->lateral_bc_east,
        nullptr, frame_u_ms);
    const double fx_minus = x_face(
        rho, u, i - 1, j, k, nx, StaggerType::CGrid,
        state.configuration->lateral_bc_west, state.configuration->lateral_bc_east,
        nullptr, frame_u_ms);
    const double fy_plus = y_face(
        rho, v, i, j, k, ny, StaggerType::CGrid,
        state.configuration->lateral_bc_south, state.configuration->lateral_bc_north,
        nullptr, frame_v_ms);
    const double fy_minus = y_face(
        rho, v, i, j - 1, k, ny, StaggerType::CGrid,
        state.configuration->lateral_bc_south, state.configuration->lateral_bc_north,
        nullptr, frame_v_ms);
    const double fz_plus = z_face(
        rho, w, i, j, k, nz, StaggerType::CGrid,
        state.configuration->bottom_bc, state.configuration->top_bc);
    const double fz_minus = k == 0 ? 0.0 : z_face(
        rho, w, i, j, k - 1, nz, StaggerType::CGrid,
        state.configuration->bottom_bc, state.configuration->top_bc);
    return (fx_plus - fx_minus) / dx
         + (fy_plus - fy_minus) / dy
         + (fz_plus - fz_minus) / dz;
}

double x_edge_y_mass(const Field3D& rho, const Field3D& v,
                     int i, int face, int k, double frame_v_ms)
{
    const double density = 0.25 * (
        static_cast<double>(rho[i][face][k])
      + static_cast<double>(rho[i + 1][face][k])
      + static_cast<double>(rho[i][face + 1][k])
      + static_cast<double>(rho[i + 1][face + 1][k]));
    return density * (0.5 * (
        static_cast<double>(v[i][face][k])
      + static_cast<double>(v[i + 1][face][k])) - frame_v_ms);
}

double x_edge_z_mass(const Field3D& rho, const Field3D& w,
                     int i, int j, int face)
{
    const double density = 0.25 * (
        static_cast<double>(rho[i][j][face])
      + static_cast<double>(rho[i + 1][j][face])
      + static_cast<double>(rho[i][j][face + 1])
      + static_cast<double>(rho[i + 1][j][face + 1]));
    return density * 0.5 * (
        static_cast<double>(w[i][j][face])
      + static_cast<double>(w[i + 1][j][face]));
}

double y_edge_x_mass(const Field3D& rho, const Field3D& u,
                     int face, int j, int k, double frame_u_ms)
{
    const double density = 0.25 * (
        static_cast<double>(rho[face][j][k])
      + static_cast<double>(rho[face + 1][j][k])
      + static_cast<double>(rho[face][j + 1][k])
      + static_cast<double>(rho[face + 1][j + 1][k]));
    return density * (0.5 * (
        static_cast<double>(u[face][j][k])
      + static_cast<double>(u[face][j + 1][k])) - frame_u_ms);
}

double y_edge_z_mass(const Field3D& rho, const Field3D& w,
                     int i, int j, int face)
{
    const double density = 0.25 * (
        static_cast<double>(rho[i][j][face])
      + static_cast<double>(rho[i][j + 1][face])
      + static_cast<double>(rho[i][j][face + 1])
      + static_cast<double>(rho[i][j + 1][face + 1]));
    return density * 0.5 * (
        static_cast<double>(w[i][j][face])
      + static_cast<double>(w[i][j + 1][face]));
}

double z_edge_x_mass(const Field3D& rho, const Field3D& u,
                     int face, int j, int k, double frame_u_ms)
{
    const double density = 0.25 * (
        static_cast<double>(rho[face][j][k])
      + static_cast<double>(rho[face + 1][j][k])
      + static_cast<double>(rho[face][j][k + 1])
      + static_cast<double>(rho[face + 1][j][k + 1]));
    return density * (0.5 * (
        static_cast<double>(u[face][j][k])
      + static_cast<double>(u[face][j][k + 1])) - frame_u_ms);
}

double z_edge_y_mass(const Field3D& rho, const Field3D& v,
                     int i, int face, int k, double frame_v_ms)
{
    const double density = 0.25 * (
        static_cast<double>(rho[i][face][k])
      + static_cast<double>(rho[i][face + 1][k])
      + static_cast<double>(rho[i][face][k + 1])
      + static_cast<double>(rho[i][face + 1][k + 1]));
    return density * (0.5 * (
        static_cast<double>(v[i][face][k])
      + static_cast<double>(v[i][face][k + 1])) - frame_v_ms);
}

void require_conservative_momentum_fields(
    int nx, int ny, int nz, double dt,
    const Field3D& u, const Field3D& v, const Field3D& w,
    const Field3D& rho, const Field3D& p, const Field3D& theta,
    const Field3D& du_dt, const Field3D& dv_dt, const Field3D& dw_dt,
    const Field3D& drho_dt, const Field3D& dp_dt)
{
    if (!std::isfinite(dt) || dt <= 0.0)
        throw std::invalid_argument(
            "conservative C-grid momentum capture requires finite dt > 0");
    const auto matches = [=](const Field3D& field) {
        return field.size_r() == nx && field.size_th() == ny &&
               field.size_z() == nz;
    };
    for (const Field3D* field : {&u, &v, &w, &rho, &p, &theta,
                                 &du_dt, &dv_dt, &dw_dt, &drho_dt, &dp_dt})
        if (!matches(*field))
            throw std::invalid_argument(
                "conservative C-grid momentum capture requires grid-shaped fields");
    for (const Field3D* field : {&u, &v, &w, &rho, &p, &theta})
        for (std::size_t index = 0; index < field->size(); ++index)
            if (!std::isfinite(static_cast<double>(field->data()[index])))
                throw std::runtime_error(
                    "conservative C-grid momentum input is non-finite");
    for (std::size_t index = 0; index < rho.size(); ++index)
        if (rho.data()[index] <= 0.0f)
            throw std::runtime_error(
                "conservative C-grid momentum density must be positive");
}

bool dispatch_conservative_momentum(const SimulationStateView&          state,
                                    CartesianCGridMomentumFluxRegister& destination,
                                    const Field3D&                      u,
                                    const Field3D&                      v,
                                    const Field3D&                      w,
                                    const Field3D&                      rho,
                                    const Field3D&                      pressure,
                                    double                              dt,
                                    const CartesianCoriolisProfile&     coriolis,
                                    bool                                pressure_step,
                                    const std::array<Field3D*, 8>&      tendencies)
{
    if (!state.compute_runtime)
        return false;
    if (state.compute_runtime->active_kind == ComputeBackendKind::Cpu)
    {
        if (state.compute_runtime->config.backend == "metal" || state.compute_runtime->config.backend == "vulkan" ||
            state.compute_runtime->config.backend == "gpu")
            throw std::runtime_error("native conservative momentum has no active device owner");
        return false;
    }
    const auto* metrics = tmv::physical_grid_metrics::active_terrain_metrics(state);
    const bool mapped = tmv::physical_grid_metrics::terrain_is_active(state);
    if (state.compute_runtime->fallback_active || !state.compute_runtime->backend ||
        !state.compute_runtime->backend->dynamics_kernels() ||
        !state.compute_runtime->backend->supports_dynamics(DynamicsKernelOp::ConservativeMomentum) ||
        !coriolis.is_f_plane() || (mapped && coriolis.f_at_grid_y0 != 0) ||
        state.domain_frame->velocity_ms.u_ms != 0 || state.domain_frame->velocity_ms.v_ms != 0)
        throw std::runtime_error("native conservative momentum requires its stationary device owner");
    const auto& grid = state.grid;
    for (const auto* profile : {&state.base.rho0, &state.base.p0, &state.base.qv0, &state.base.u0, &state.base.v0})
        if ((!profile->empty() || profile == &state.base.rho0 || profile == &state.base.p0) &&
            profile->size() != std::size_t(grid.nz))
            throw std::invalid_argument("native conservative momentum requires complete reference profiles");
    for (const auto* field : {&state.fields.qv,
                              &state.fields.qc,
                              &state.fields.qr,
                              &state.fields.qi,
                              &state.fields.qs,
                              &state.fields.qg,
                              &state.fields.qh})
        if (!field->empty() &&
            (field->size_r() != grid.nr || field->size_th() != grid.nth || field->size_z() != grid.nz))
            throw std::invalid_argument("native conservative momentum moisture shape mismatch");
    for (auto* field : tendencies)
        if (field && (field->size_r() != grid.nr || field->size_th() != grid.nth || field->size_z() != grid.nz))
            throw std::invalid_argument("native conservative momentum tendency shape mismatch");
    const auto split = [](double value, float& high, float& low)
    {
        high = float(value);
        low  = float(value - double(high));
    };
    ConservativeMomentumDispatch d;
    auto&                        p = d.constants;
    p.nx                           = grid.nr;
    p.ny                           = grid.nth;
    p.nz                           = grid.nz;
    p.operation                    = (pressure_step ? 1 : 0) + (mapped ? 2 : 0);
    const auto footprint           = destination.footprint();
    p.i_begin                      = footprint.i_begin;
    p.i_end                        = footprint.i_end;
    p.j_begin                      = footprint.j_begin;
    p.j_end                        = footprint.j_end;
    p.k_begin                      = footprint.k_begin;
    p.k_end                        = footprint.k_end;
    split(grid.dr, p.dx_high, p.dx_low);
    split(grid.dtheta, p.dy_high, p.dy_low);
    split(mapped ? metrics->dzeta : grid.dz, p.dz_high, p.dz_low);
    split(dynamics_constants::g, p.gravity_high, p.gravity_low);
    split(coriolis.f_at_grid_y0, p.coriolis_high, p.coriolis_low);
    const auto         cells = rho.size();
    std::vector<float> input(cells * ConservativeMomentumDispatch::input_stride);
    std::vector<float> output(nimbus::compute::conservative_momentum::output_values(d));
    for (std::size_t n = 0; n < cells; ++n)
    {
        auto*     values = input.data() + n * ConservativeMomentumDispatch::input_stride;
        const int k      = int(n % grid.nz);
        values[0]        = u.data()[n];
        values[1]        = v.data()[n];
        values[2]        = w.data()[n];
        values[3]        = rho.data()[n];
        values[4]        = pressure.data()[n];
        split(state.base.rho0[k], values[5], values[6]);
        split(state.base.p0[k], values[7], values[8]);
        values[9]      = !state.fields.qv.empty() && !state.base.qv0.empty() ? state.fields.qv.data()[n] : 0;
        values[10]     = !state.fields.qv.empty() && !state.base.qv0.empty() ? float(state.base.qv0[k]) : 0;
        double loading = 0;
        for (const auto* field : {&state.fields.qc,
                                  &state.fields.qr,
                                  &state.fields.qi,
                                  &state.fields.qs,
                                  &state.fields.qg,
                                  &state.fields.qh})
            if (!field->empty())
                loading += double(field->data()[n]);
        split(loading, values[11], values[12]);
        split(state.base.u0.empty() ? 0 : state.base.u0[k], values[13], values[14]);
        split(state.base.v0.empty() ? 0 : state.base.v0[k], values[15], values[16]);
        if (mapped)
        {
            const int i = int(n / (grid.nth * grid.nz)), j = int(n / grid.nz) % grid.nth;
            split(metrics->Jx_face(i + 1, j, k), values[17], values[18]);
            split(metrics->Jy_face(i, j + 1, k), values[19], values[20]);
        }
    }
    d.input  = input.data();
    d.output = output.data();
    std::array<Field3D, 8> mapped_rates;
    if (mapped)
        tmv::terrain_native::prepare_mapped_momentum_tendencies(state, u, v, w, rho, pressure, pressure_step, mapped_rates);
    if (!state.compute_runtime->backend->dynamics_kernels()->dispatch_conservative_momentum(d))
        throw std::runtime_error("native conservative momentum dispatch failed");
    auto candidate = destination;
    candidate.stage_native_fluxes(output.data() + cells * ConservativeMomentumDispatch::tendency_stride,
                                  nimbus::compute::conservative_momentum::strip_values(d),
                                  dt,
                                  grid.dr,
                                  grid.dtheta,
                                  mapped ? metrics->dzeta : grid.dz,
                                  pressure_step);
    if (mapped)
    {
        candidate.stage_mapped_tendency_closure(rho, mapped_rates[0], mapped_rates[1], mapped_rates[2], *metrics, dt);
        candidate.validate_staged_commit();
        candidate.commit_staged();
        for (std::size_t field = 0; field < tendencies.size(); ++field)
            if (tendencies[field])
                std::copy_n(mapped_rates[field].data(), cells, tendencies[field]->data());
        destination = std::move(candidate);
        return true;
    }
    std::array<double, 3> source{};
    const double          volume_dt = grid.dr * grid.dtheta * grid.dz * dt;
    for (int i = footprint.i_begin; i < footprint.i_end; ++i)
        for (int j = footprint.j_begin; j < footprint.j_end; ++j)
            for (int k = footprint.k_begin; k < footprint.k_end; ++k)
            {
                const auto   n      = (std::size_t(i) * grid.nth + j) * grid.nz + k;
                const float* values = output.data() + n * ConservativeMomentumDispatch::tendency_stride;
                if (pressure_step)
                    source[2] += cartesian_cgrid_momentum_flux::rho_z(rho, i, j, k) * double(values[2]) * volume_dt;
                else
                {
                    source[0] += cartesian_cgrid_momentum_flux::rho_x(rho, i, j, k) * (double(values[0]) - values[5]) *
                                 volume_dt;
                    source[1] += cartesian_cgrid_momentum_flux::rho_y(rho, i, j, k) * (double(values[1]) - values[6]) *
                                 volume_dt;
                    source[2] += cartesian_cgrid_momentum_flux::rho_z(rho, i, j, k) * (double(values[2]) - values[7]) *
                                 volume_dt;
                }
            }
    if (pressure_step)
        candidate.stage_pressure_source_impulse(source);
    else
        candidate.stage_source_impulse(source);
    candidate.validate_staged_commit();
    candidate.commit_staged();
    for (std::size_t field = 0; field < tendencies.size(); ++field)
        if (tendencies[field])
            for (std::size_t n = 0; n < cells; ++n)
                tendencies[field]->data()[n] = output[n * ConservativeMomentumDispatch::tendency_stride + field];
    destination = std::move(candidate);
    return true;
}

} // namespace

// Cartesian grid: the global NR/NTH map to NX/NY, and dr/dtheta to dx/dy
// (see GridGeometry: scalar centers are half a cell above the physical faces).
CartesianCGrid3DScheme::CartesianCGrid3DScheme() : CartesianCGrid3DScheme(global_simulation_state()) {}

CartesianCGrid3DScheme::CartesianCGrid3DScheme(const SimulationStateView& state)
    : state_(std::make_shared<SimulationStateView>(state)),
      NX_(state.grid.nr), NY_(state.grid.nth), NZ_(state.grid.nz),
      dx_(state.grid.dr), dy_(state.grid.dtheta), dz_(state.grid.dz),
      deriv_(state.grid.dr, state.grid.dtheta, state.grid.dz, state.grid.nr, state.grid.nth, state.grid.nz)
{
}

void CartesianCGrid3DScheme::bind_execution_context(const SimulationStateView& state)
{
    if (state.grid.nr != NX_ || state.grid.nth != NY_ || state.grid.nz != NZ_ ||
        state.grid.dr != dx_ || state.grid.dtheta != dy_ || state.grid.dz != dz_)
        throw std::invalid_argument("dynamics operator geometry differs from its level");
    if (!state.configuration || !state.domain_frame || !state.compute_runtime || !state.parity_runtime)
        throw std::invalid_argument("dynamics requires complete level execution ownership");
    if (&state_->grid.nr != &state.grid.nr || &state_->fields.rho != &state.fields.rho ||
        state_->configuration != state.configuration || state_->domain_frame != state.domain_frame ||
        state_->terrain_runtime != state.terrain_runtime || state_->compute_runtime != state.compute_runtime ||
        state_->parity_runtime != state.parity_runtime)
        state_ = std::make_shared<SimulationStateView>(state);
}

std::unique_ptr<DynamicsScheme> CartesianCGrid3DScheme::clone() const
{
    auto result = std::make_unique<CartesianCGrid3DScheme>(*this);
    result->set_momentum_interface_flux_register(nullptr);
    result->momentum_interface_flux_slow_weight_ = 1.0;
    return result;
}

void CartesianCGrid3DScheme::invalidate_generation_state_after_domain_recycling() noexcept
{
    has_coriolis_profile_ = false;
    parity_replay_ = false;
    slow_rk2_invariant_input_residency_ = false;
    momentum_interface_flux_register_ = nullptr;
    momentum_interface_flux_slow_weight_ = 1.0;
    for (std::vector<float>* values : {
             &gpu_loading_, &gpu_rho0_, &gpu_qv0_, &gpu_u0_, &gpu_v0_})
        std::fill(values->begin(), values->end(), 0.0f);
}

void CartesianCGrid3DScheme::set_momentum_interface_flux_slow_weight(
    double weight)
{
    if (!std::isfinite(weight) || weight <= 0.0 || weight > 1.0)
        throw std::invalid_argument(
            "momentum interface flux slow-stage weight must be finite in (0, 1]");
    momentum_interface_flux_slow_weight_ = weight;
}

void CartesianCGrid3DScheme::compute_conservative_slow_tendencies(
    const Field3D& u, const Field3D& v, const Field3D& w,
    const Field3D& rho, const Field3D& p, const Field3D& theta, double dt,
    Field3D& du_dt, Field3D& dv_dt, Field3D& dw_dt,
    Field3D& drho_dt, Field3D& dp_dt,
    SlowMomentumAdvectionTendencyOutputs* advection_outputs)
{
    require_conservative_momentum_fields(
        NX_, NY_, NZ_, dt, u, v, w, rho, p, theta,
        du_dt, dv_dt, dw_dt, drho_dt, dp_dt);
    if (tmv::physical_grid_metrics::terrain_is_active(*state_) ||
        (*state_->domain_frame).velocity_ms.u_ms != 0.0 ||
        (*state_->domain_frame).velocity_ms.v_ms != 0.0)
        throw std::runtime_error(
            "conservative C-grid momentum capture currently requires a stationary flat level");
    if (state_->base.rho0.size() != static_cast<std::size_t>(NZ_) ||
        state_->base.p0.size() != static_cast<std::size_t>(NZ_))
        throw std::runtime_error(
            "conservative C-grid momentum capture requires complete base profiles");
    for (double value : state_->base.rho0)
        if (!std::isfinite(value) || value <= 0.0)
            throw std::runtime_error(
                "conservative C-grid momentum base density is invalid");
    for (double value : state_->base.p0)
        if (!std::isfinite(value) || value <= 0.0)
            throw std::runtime_error(
                "conservative C-grid momentum base pressure is invalid");
    if (advection_outputs != nullptr)
    {
        if (!advection_outputs->complete())
            throw std::invalid_argument(
                "conservative C-grid momentum attribution requires three outputs");
        for (const Field3D* field : {advection_outputs->du_dt,
                                     advection_outputs->dv_dt,
                                     advection_outputs->dw_dt})
            if (field->size_r() != NX_ || field->size_th() != NY_ ||
                field->size_z() != NZ_)
                throw std::invalid_argument(
                    "conservative C-grid momentum attribution shape mismatch");
    }

    const double flux_dt = dt * momentum_interface_flux_slow_weight_;
    const CartesianCoriolisProfile native_coriolis = has_coriolis_profile_
        ? coriolis_profile_ : CartesianCoriolisProfile{state_->coriolis_f, 0.0};
    if (!parity_replay_ && dispatch_conservative_momentum(*state_, *momentum_interface_flux_register_,
        u, v, w, rho, p, flux_dt, native_coriolis, false,
        {&du_dt, &dv_dt, &dw_dt, &drho_dt, &dp_dt,
         advection_outputs ? advection_outputs->du_dt : nullptr,
         advection_outputs ? advection_outputs->dv_dt : nullptr,
         advection_outputs ? advection_outputs->dw_dt : nullptr}))
        return;
    momentum_interface_flux_register_->stage_advective(
        rho, u, v, w, flux_dt, dx_, dy_, dz_);

    Field3D candidate_u(NX_, NY_, NZ_, 0.0f);
    Field3D candidate_v(NX_, NY_, NZ_, 0.0f);
    Field3D candidate_w(NX_, NY_, NZ_, 0.0f);
    Field3D candidate_rho(NX_, NY_, NZ_, 0.0f);
    Field3D candidate_p(NX_, NY_, NZ_, 0.0f);
    Field3D candidate_adv_u(NX_, NY_, NZ_, 0.0f);
    Field3D candidate_adv_v(NX_, NY_, NZ_, 0.0f);
    Field3D candidate_adv_w(NX_, NY_, NZ_, 0.0f);

    namespace mf = cartesian_cgrid_momentum_flux;
    const double inv_dx = 1.0 / dx_;
    const double inv_dy = 1.0 / dy_;
    const double inv_dz = 1.0 / dz_;
    const double inv_2dx = 0.5 * inv_dx;
    const double inv_2dy = 0.5 * inv_dy;
    const double inv_2dz = 0.5 * inv_dz;
    const double g_local = dynamics_constants::g;
    const CartesianCoriolisProfile coriolis = has_coriolis_profile_
        ? coriolis_profile_ : CartesianCoriolisProfile{state_->coriolis_f, 0.0};
    if (!coriolis.is_f_plane())
        throw std::runtime_error(
            "conservative C-grid momentum capture does not yet own beta-plane source bins");

    std::atomic<bool> finite{true};
    #pragma omp parallel for collapse(2)
    for (int i = 1; i < NX_ - 1; ++i)
        for (int j = 1; j < NY_ - 1; ++j)
            for (int k = 0; k < NZ_ - 1; ++k)
            {
                const double mass_div =
                    (mf::x_normal_mass(rho, u, i + 1, j, k, 0.0)
                   - mf::x_normal_mass(rho, u, i, j, k, 0.0)) * inv_dx
                  + (x_edge_y_mass(rho, v, i, j, k, 0.0)
                   - x_edge_y_mass(rho, v, i, j - 1, k, 0.0)) * inv_dy
                  + ((k + 1 < NZ_ - 1
                        ? x_edge_z_mass(rho, w, i, j, k) : 0.0)
                   - (k == 0 ? 0.0
                             : x_edge_z_mass(rho, w, i, j, k - 1))) * inv_dz;
                const double momentum_div =
                    (mf::x_flux_x(rho, u, i + 1, j, k, 0.0)
                   - mf::x_flux_x(rho, u, i, j, k, 0.0)) * inv_dx
                  + (mf::y_flux_x(rho, u, v, i, j, k, 0.0)
                   - mf::y_flux_x(rho, u, v, i, j - 1, k, 0.0)) * inv_dy
                  + ((k + 1 < NZ_ - 1
                        ? mf::z_flux_x(rho, u, w, i, j, k) : 0.0)
                   - (k == 0 ? 0.0
                             : mf::z_flux_x(rho, u, w, i, j, k - 1))) * inv_dz;
                const double rho_dual = mf::rho_x(rho, i, j, k);
                const double advection =
                    (-momentum_div + static_cast<double>(u[i][j][k]) * mass_div)
                    / rho_dual;
                const double v_at_x = 0.25 * (
                    static_cast<double>(v[i][j - 1][k])
                  + static_cast<double>(v[i][j][k])
                  + static_cast<double>(v[i + 1][j - 1][k])
                  + static_cast<double>(v[i + 1][j][k]));
                const double v0 = static_cast<int>(state_->base.v0.size()) > k
                    ? state_->base.v0[static_cast<std::size_t>(k)] : 0.0;
                const double source = coriolis.f_at_grid_y0 * (v_at_x - v0);
                const double tendency = advection + source;
                if (!std::isfinite(tendency) ||
                    std::abs(tendency) > std::numeric_limits<float>::max())
                    finite.store(false, std::memory_order_relaxed);
                else
                    candidate_u[i][j][k] = static_cast<float>(tendency);
                candidate_adv_u[i][j][k] = static_cast<float>(advection);
            }

    #pragma omp parallel for collapse(2)
    for (int i = 1; i < NX_ - 1; ++i)
        for (int j = 1; j < NY_ - 1; ++j)
            for (int k = 0; k < NZ_ - 1; ++k)
            {
                const double mass_div =
                    (y_edge_x_mass(rho, u, i, j, k, 0.0)
                   - y_edge_x_mass(rho, u, i - 1, j, k, 0.0)) * inv_dx
                  + (mf::y_normal_mass(rho, v, i, j + 1, k, 0.0)
                   - mf::y_normal_mass(rho, v, i, j, k, 0.0)) * inv_dy
                  + ((k + 1 < NZ_ - 1
                        ? y_edge_z_mass(rho, w, i, j, k) : 0.0)
                   - (k == 0 ? 0.0
                             : y_edge_z_mass(rho, w, i, j, k - 1))) * inv_dz;
                const double momentum_div =
                    (mf::x_flux_y(rho, u, v, i, j, k, 0.0)
                   - mf::x_flux_y(rho, u, v, i - 1, j, k, 0.0)) * inv_dx
                  + (mf::y_flux_y(rho, v, i, j + 1, k, 0.0)
                   - mf::y_flux_y(rho, v, i, j, k, 0.0)) * inv_dy
                  + ((k + 1 < NZ_ - 1
                        ? mf::z_flux_y(rho, v, w, i, j, k) : 0.0)
                   - (k == 0 ? 0.0
                             : mf::z_flux_y(rho, v, w, i, j, k - 1))) * inv_dz;
                const double rho_dual = mf::rho_y(rho, i, j, k);
                const double advection =
                    (-momentum_div + static_cast<double>(v[i][j][k]) * mass_div)
                    / rho_dual;
                const double u_at_y = 0.25 * (
                    static_cast<double>(u[i - 1][j][k])
                  + static_cast<double>(u[i][j][k])
                  + static_cast<double>(u[i - 1][j + 1][k])
                  + static_cast<double>(u[i][j + 1][k]));
                const double u0 = static_cast<int>(state_->base.u0.size()) > k
                    ? state_->base.u0[static_cast<std::size_t>(k)] : 0.0;
                const double source = -coriolis.f_at_grid_y0 * (u_at_y - u0);
                const double tendency = advection + source;
                if (!std::isfinite(tendency) ||
                    std::abs(tendency) > std::numeric_limits<float>::max())
                    finite.store(false, std::memory_order_relaxed);
                else
                    candidate_v[i][j][k] = static_cast<float>(tendency);
                candidate_adv_v[i][j][k] = static_cast<float>(advection);
            }

    #pragma omp parallel for collapse(2)
    for (int i = 1; i < NX_ - 1; ++i)
        for (int j = 1; j < NY_ - 1; ++j)
            for (int k = 0; k < NZ_ - 1; ++k)
            {
                const double mass_div =
                    (z_edge_x_mass(rho, u, i, j, k, 0.0)
                   - z_edge_x_mass(rho, u, i - 1, j, k, 0.0)) * inv_dx
                  + (z_edge_y_mass(rho, v, i, j, k, 0.0)
                   - z_edge_y_mass(rho, v, i, j - 1, k, 0.0)) * inv_dy
                  + (mf::z_normal_mass(rho, w, i, j, k + 1)
                   - mf::z_normal_mass(rho, w, i, j, k)) * inv_dz;
                const double momentum_div =
                    (mf::x_flux_z(rho, u, w, i, j, k, 0.0)
                   - mf::x_flux_z(rho, u, w, i - 1, j, k, 0.0)) * inv_dx
                  + (mf::y_flux_z(rho, v, w, i, j, k, 0.0)
                   - mf::y_flux_z(rho, v, w, i, j - 1, k, 0.0)) * inv_dy
                  + (mf::z_flux_z(rho, w, i, j, k + 1)
                   - mf::z_flux_z(rho, w, i, j, k)) * inv_dz;
                const double rho_dual = mf::rho_z(rho, i, j, k);
                const double advection =
                    (-momentum_div + static_cast<double>(w[i][j][k]) * mass_div)
                    / rho_dual;
                const double rho0 = 0.5 * (
                    state_->base.rho0[static_cast<std::size_t>(k)]
                  + state_->base.rho0[static_cast<std::size_t>(k + 1)]);
                double source = -g_local * (rho_dual - rho0) / rho_dual;
                source += kw_cart::moisture_buoyancy_z_face(
                    state_->fields.qv, state_->base.qv0, g_local, i, j, k);
                if (!state_->fields.qc.empty())
                    source -= g_local * kw_cart::hydrometeor_mixing_ratio_z_face(
                        state_->fields.qc, state_->fields.qr, state_->fields.qi, state_->fields.qs, state_->fields.qg, state_->fields.qh, i, j, k);
                const double tendency = advection + source;
                if (!std::isfinite(tendency) ||
                    std::abs(tendency) > std::numeric_limits<float>::max())
                    finite.store(false, std::memory_order_relaxed);
                else
                    candidate_w[i][j][k] = static_cast<float>(tendency);
                candidate_adv_w[i][j][k] = static_cast<float>(advection);
            }

    #pragma omp parallel for collapse(2)
    for (int i = 1; i < NX_ - 1; ++i)
        for (int j = 1; j < NY_ - 1; ++j)
            for (int k = 1; k < NZ_ - 1; ++k)
            {
                const double center_u = 0.5 * (
                    static_cast<double>(u[i - 1][j][k])
                  + static_cast<double>(u[i][j][k]));
                const double center_v = 0.5 * (
                    static_cast<double>(v[i][j - 1][k])
                  + static_cast<double>(v[i][j][k]));
                const double center_w = 0.5 * (
                    static_cast<double>(w[i][j][k - 1])
                  + static_cast<double>(w[i][j][k]));
                const double tendency =
                    -center_u * (static_cast<double>(p[i + 1][j][k])
                               - static_cast<double>(p[i - 1][j][k])) * inv_2dx
                    -center_v * (static_cast<double>(p[i][j + 1][k])
                               - static_cast<double>(p[i][j - 1][k])) * inv_2dy
                    -center_w * (static_cast<double>(p[i][j][k + 1])
                               - static_cast<double>(p[i][j][k - 1])) * inv_2dz;
                if (!std::isfinite(tendency) ||
                    std::abs(tendency) > std::numeric_limits<float>::max())
                    finite.store(false, std::memory_order_relaxed);
                else
                    candidate_p[i][j][k] = static_cast<float>(tendency);
            }

    if (!finite.load(std::memory_order_relaxed))
        throw std::runtime_error(
            "conservative C-grid momentum candidate is not representable");

    std::array<double, 3> source_impulse{};
    const auto footprint = momentum_interface_flux_register_->footprint();
    const double volume_dt = dx_ * dy_ * dz_ * flux_dt;
    for (int i = footprint.i_begin; i < footprint.i_end; ++i)
        for (int j = footprint.j_begin; j < footprint.j_end; ++j)
            for (int k = footprint.k_begin; k < footprint.k_end; ++k)
            {
                const double adv_u = static_cast<double>(candidate_adv_u[i][j][k]);
                const double adv_v = static_cast<double>(candidate_adv_v[i][j][k]);
                const double adv_w = static_cast<double>(candidate_adv_w[i][j][k]);
                const double source_u = static_cast<double>(candidate_u[i][j][k]) - adv_u;
                const double source_v = static_cast<double>(candidate_v[i][j][k]) - adv_v;
                const double source_w = static_cast<double>(candidate_w[i][j][k]) - adv_w;
                source_impulse[0] += mf::rho_x(rho, i, j, k) * source_u * volume_dt;
                source_impulse[1] += mf::rho_y(rho, i, j, k) * source_v * volume_dt;
                source_impulse[2] += mf::rho_z(rho, i, j, k) * source_w * volume_dt;
            }
    momentum_interface_flux_register_->stage_source_impulse(source_impulse);
    momentum_interface_flux_register_->validate_staged_commit();

    const std::size_t bytes = u.size() * sizeof(float);
    std::memcpy(du_dt.data(), candidate_u.data(), bytes);
    std::memcpy(dv_dt.data(), candidate_v.data(), bytes);
    std::memcpy(dw_dt.data(), candidate_w.data(), bytes);
    std::memcpy(drho_dt.data(), candidate_rho.data(), bytes);
    std::memcpy(dp_dt.data(), candidate_p.data(), bytes);
    if (advection_outputs != nullptr)
    {
        std::memcpy(advection_outputs->du_dt->data(), candidate_adv_u.data(), bytes);
        std::memcpy(advection_outputs->dv_dt->data(), candidate_adv_v.data(), bytes);
        std::memcpy(advection_outputs->dw_dt->data(), candidate_adv_w.data(), bytes);
    }
    momentum_interface_flux_register_->commit_staged();
}

// =====================================================================
// Slow tendencies: advection (+ buoyancy/moisture on w), -u.grad(p).
// NO pressure gradient on velocity, NO -gamma*p*div on p, drho/dt = 0.
// =====================================================================
void CartesianCGrid3DScheme::compute_slow_tendencies(
    const Field3D& u, const Field3D& v, const Field3D& w,
    const Field3D& rho, const Field3D& p, const Field3D& theta, double dt,
    Field3D& du_dt, Field3D& dv_dt, Field3D& dw_dt,
    Field3D& drho_dt, Field3D& dp_dt,
    SlowMomentumAdvectionTendencyOutputs* advection_outputs)
{
    const bool mapped_momentum_capture =
        momentum_interface_flux_register_ != nullptr
        && tmv::physical_grid_metrics::terrain_is_active(*state_);
    if (mapped_momentum_capture && !parity_replay_)
    {
        require_conservative_momentum_fields(NX_, NY_, NZ_, dt, u, v, w, rho, p, theta,
                                             du_dt, dv_dt, dw_dt, drho_dt, dp_dt);
        const CartesianCoriolisProfile coriolis = has_coriolis_profile_ ? coriolis_profile_ : CartesianCoriolisProfile{state_->coriolis_f, 0.0};
        if (dispatch_conservative_momentum(*state_, *momentum_interface_flux_register_, u, v, w, rho, p,
                dt * momentum_interface_flux_slow_weight_, coriolis, false,
                {&du_dt, &dv_dt, &dw_dt, &drho_dt, &dp_dt,
                 advection_outputs ? advection_outputs->du_dt : nullptr,
                 advection_outputs ? advection_outputs->dv_dt : nullptr,
                 advection_outputs ? advection_outputs->dw_dt : nullptr}))
            return;
    }
    if (momentum_interface_flux_register_ != nullptr
        && !mapped_momentum_capture)
    {
        compute_conservative_slow_tendencies(
            u, v, w, rho, p, theta, dt,
            du_dt, dv_dt, dw_dt, drho_dt, dp_dt, advection_outputs);
        return;
    }
    (void)theta;  // perturbation-density buoyancy uses rho, not theta
    const double frame_u_ms = (*state_->domain_frame).velocity_ms.u_ms;
    const double frame_v_ms = (*state_->domain_frame).velocity_ms.v_ms;
    const CartesianCoriolisProfile coriolis = has_coriolis_profile_
        ? coriolis_profile_
        : CartesianCoriolisProfile{state_->coriolis_f, 0.0};

    if (advection_outputs != nullptr)
    {
        if (!advection_outputs->complete())
            throw std::invalid_argument(
                "C-grid slow momentum attribution requires all three advection outputs");
        const auto matches = [this](const Field3D& field)
        {
            return field.size_r() == NX_ && field.size_th() == NY_
                && field.size_z() == NZ_;
        };
        if (!matches(*advection_outputs->du_dt)
            || !matches(*advection_outputs->dv_dt)
            || !matches(*advection_outputs->dw_dt))
        {
            throw std::invalid_argument(
                "C-grid slow momentum attribution requires grid-shaped outputs");
        }
    }

    const TerrainMetrics3D* terrain_metrics =
        tmv::physical_grid_metrics::terrain_is_active(*state_)
        ? tmv::physical_grid_metrics::active_terrain_metrics(*state_) : nullptr;
    if (terrain_metrics != nullptr)
    {
        terrain_transport::validate_cgrid_shapes(
            *terrain_metrics, rho, u, v, w);
        if (state_->base.rho0.size() != static_cast<std::size_t>(NZ_))
            throw std::invalid_argument(
                "terrain C-grid buoyancy requires a full reference-density profile");
        for (double value : state_->base.rho0)
            if (!std::isfinite(value) || value <= 0.0)
                throw std::invalid_argument(
                    "terrain C-grid reference density must be positive and finite");
        if (state_->base.p0.size() != static_cast<std::size_t>(NZ_))
            throw std::invalid_argument(
                "terrain C-grid pressure advection requires a full reference-pressure profile");
        for (double value : state_->base.p0)
            if (!std::isfinite(value) || value <= 0.0)
                throw std::invalid_argument(
                    "terrain C-grid reference pressure must be positive and finite");
        if (state_->base.theta0.size() != static_cast<std::size_t>(NZ_))
            throw std::invalid_argument(
                "terrain C-grid buoyancy requires a full reference-theta profile");
        for (double value : state_->base.theta0)
            if (!std::isfinite(value) || value <= 0.0)
                throw std::invalid_argument(
                    "terrain C-grid reference theta must be positive and finite");
        for (const auto* wind_profile : {&state_->base.u0, &state_->base.v0})
        {
            if (!wind_profile->empty()
                && wind_profile->size() != static_cast<std::size_t>(NZ_))
            {
                throw std::invalid_argument(
                    "terrain C-grid reference wind profiles must be empty or full-column");
            }
            for (double value : *wind_profile)
                if (!std::isfinite(value))
                    throw std::invalid_argument(
                        "terrain C-grid reference wind must be finite");
        }
        if (!state_->fields.qv.empty())
        {
            if (state_->base.qv0.size() != static_cast<std::size_t>(NZ_))
                throw std::invalid_argument(
                    "terrain C-grid moisture requires a full reference-vapor profile");
            for (double value : state_->base.qv0)
                if (!std::isfinite(value) || value < 0.0)
                    throw std::invalid_argument(
                        "terrain C-grid reference vapor must be nonnegative and finite");
        }
    }

    // GPU-first dispatch. Returns false when
    // no backend supports it (CPU-only build, or the .spv pipeline failed to
    // load), in which case we fall through to the CPU path below. The shader
    // takes a single pre-summed hydrometeor `loading` buffer and broadcasts
    // rho0/qv0 1D->3D.
    // Skipped entirely during a parity replay: the replay must take the CPU
    // path below against the same inputs.
    if (terrain_metrics == nullptr && frame_u_ms == 0.0 && frame_v_ms == 0.0
        && !parity_replay_
        && supports_dynamics_backend(*state_,
            DynamicsKernelOp::CartesianCgridSlowTendencies))
    {
        const size_t n_total =
            static_cast<size_t>(NX_) * static_cast<size_t>(NY_) *
            static_cast<size_t>(NZ_);
        const bool has_moisture = !state_->fields.qv.empty() && !state_->base.qv0.empty();

        const bool has_loading = kw_cart::fill_hydrometeor_loading_buffer(
            state_->fields.qc, state_->fields.qr, state_->fields.qi, state_->fields.qs, state_->fields.qg, state_->fields.qh, n_total, /*compact_empty=*/true,
            gpu_loading_);

        if (!state_->base.rho0.empty())
        {
            gpu_rho0_.assign(state_->base.rho0.begin(), state_->base.rho0.end());
        }
        else
        {
            gpu_rho0_.clear();
        }
        if (has_moisture)
        {
            gpu_qv0_.assign(state_->base.qv0.begin(), state_->base.qv0.end());
        }
        else
        {
            gpu_qv0_.assign(1, 0.0f);
        }

        const float* qv_data_ptr =
            has_moisture ? state_->fields.qv.data() : gpu_loading_.data();
        const int rho0_len = static_cast<int>(gpu_rho0_.size());
        const int qv0_len  = static_cast<int>(gpu_qv0_.size());

        // Environmental wind profiles for the perturbation-wind Coriolis
        // terms (SCIENCE.md M1); compact fp32 like the other base
        // profiles, 1-float zeros when no sounding wind exists.
        if (static_cast<int>(state_->base.u0.size()) >= NZ_
            && static_cast<int>(state_->base.v0.size()) >= NZ_)
        {
            gpu_u0_.assign(state_->base.u0.begin(), state_->base.u0.end());
            gpu_v0_.assign(state_->base.v0.begin(), state_->base.v0.end());
        }
        else
        {
            gpu_u0_.assign(1, 0.0f);
            gpu_v0_.assign(1, 0.0f);
        }
        const int wind0_len = static_cast<int>(gpu_u0_.size());

        const auto try_dispatch = [&]() {
            return state_->compute_runtime->backend->dynamics_kernels()->dispatch_cartesian_cgrid_slow_tendencies({
                u.data(), v.data(), w.data(),
                rho.data(), p.data(),
                gpu_rho0_.data(), rho0_len,
                gpu_qv0_.data(), qv0_len,
                gpu_u0_.data(), gpu_v0_.data(), wind0_len,
                qv_data_ptr,
                gpu_loading_.data(),
                du_dt.data(), dv_dt.data(), dw_dt.data(),
                drho_dt.data(), dp_dt.data(),
                NX_, NY_, NZ_,
                static_cast<float>(dx_),
                static_cast<float>(dy_),
                static_cast<float>(dz_),
                static_cast<float>(dynamics_constants::g),
                static_cast<float>(coriolis.f_at_grid_y0),
                has_moisture, has_loading,
                slow_rk2_invariant_input_residency_,
                advection_outputs != nullptr
                    ? advection_outputs->du_dt->data() : nullptr,
                advection_outputs != nullptr
                    ? advection_outputs->dv_dt->data() : nullptr,
                advection_outputs != nullptr
                    ? advection_outputs->dw_dt->data() : nullptr,
                static_cast<float>(frame_u_ms),
                static_cast<float>(frame_v_ms),
                static_cast<float>(coriolis.beta)});
        };

        if (rho0_len >= NZ_)
        {
            // Parity harness (GPU.md Phase 1): when sampling,
            // snapshot the output fields before the GPU writes them, replay
            // this same method on the CPU path into the snapshots, and
            // compare GPU vs CPU. The GPU result stays the production
            // result; the snapshots seed the replay so cells untouched by
            // both paths compare equal.
            const bool parity_sample =
                nimbus::parity::sample(*state_->parity_runtime, state_->compute_runtime->config.validate_parity, "cartesian_cgrid_slow_tendencies");
            if (!parity_sample)
            {
                if (try_dispatch()) return;
            }
            else
            {
                auto& pool = FieldPool::instance();
                auto s_du   = pool.scoped_acquire_copy(du_dt);
                auto s_dv   = pool.scoped_acquire_copy(dv_dt);
                auto s_dw   = pool.scoped_acquire_copy(dw_dt);
                auto s_drho = pool.scoped_acquire_copy(drho_dt);
                auto s_dp   = pool.scoped_acquire_copy(dp_dt);
                if (try_dispatch())
                {
                    Field3D cpu_adv_u;
                    Field3D cpu_adv_v;
                    Field3D cpu_adv_w;
                    SlowMomentumAdvectionTendencyOutputs cpu_advection;
                    SlowMomentumAdvectionTendencyOutputs* cpu_advection_ptr =
                        nullptr;
                    if (advection_outputs != nullptr)
                    {
                        cpu_adv_u.resize(NX_, NY_, NZ_, 0.0f);
                        cpu_adv_v.resize(NX_, NY_, NZ_, 0.0f);
                        cpu_adv_w.resize(NX_, NY_, NZ_, 0.0f);
                        cpu_advection = {
                            &cpu_adv_u, &cpu_adv_v, &cpu_adv_w,
                        };
                        cpu_advection_ptr = &cpu_advection;
                    }
                    {
                        struct ParityReplayScope
                        {
                            bool& active;
                            explicit ParityReplayScope(bool& flag) noexcept
                                : active(flag)
                            {
                                active = true;
                            }
                            ~ParityReplayScope() { active = false; }
                        } parity_replay_scope(parity_replay_);
                        compute_slow_tendencies(
                            u, v, w, rho, p, theta, 0.0,
                            s_du.field, s_dv.field,
                            s_dw.field, s_drho.field,
                            s_dp.field,
                            cpu_advection_ptr);
                    }

                    using nimbus::parity::Tier;
                    const char* kernel = "cartesian_cgrid_slow_tendencies";
                    nimbus::parity::check_field(*state_->parity_runtime, kernel, "du_dt",
                        s_du.field.data(), du_dt.data(),
                        NX_, NY_, NZ_, Tier::Stencil, nimbus::parity::ComparisonDomain::StrictInterior);
                    nimbus::parity::check_field(*state_->parity_runtime, kernel, "dv_dt",
                        s_dv.field.data(), dv_dt.data(),
                        NX_, NY_, NZ_, Tier::Stencil, nimbus::parity::ComparisonDomain::StrictInterior);
                    nimbus::parity::check_field(*state_->parity_runtime, kernel, "dw_dt",
                        s_dw.field.data(), dw_dt.data(),
                        NX_, NY_, NZ_, Tier::Stencil, nimbus::parity::ComparisonDomain::StrictInterior);
                    nimbus::parity::check_field(*state_->parity_runtime, kernel, "drho_dt",
                        s_drho.field.data(), drho_dt.data(),
                        NX_, NY_, NZ_, Tier::Stencil, nimbus::parity::ComparisonDomain::StrictInterior);
                    nimbus::parity::check_field(*state_->parity_runtime, kernel, "dp_dt",
                        s_dp.field.data(), dp_dt.data(),
                        NX_, NY_, NZ_, Tier::Stencil, nimbus::parity::ComparisonDomain::StrictInterior);
                    if (advection_outputs != nullptr)
                    {
                        nimbus::parity::check_field(*state_->parity_runtime,
                            kernel, "du_advection_dt",
                            cpu_adv_u.data(),
                            advection_outputs->du_dt->data(),
                            NX_, NY_, NZ_, Tier::Stencil, nimbus::parity::ComparisonDomain::StrictInterior);
                        nimbus::parity::check_field(*state_->parity_runtime,
                            kernel, "dv_advection_dt",
                            cpu_adv_v.data(),
                            advection_outputs->dv_dt->data(),
                            NX_, NY_, NZ_, Tier::Stencil, nimbus::parity::ComparisonDomain::StrictInterior);
                        nimbus::parity::check_field(*state_->parity_runtime,
                            kernel, "dw_advection_dt",
                            cpu_adv_w.data(),
                            advection_outputs->dw_dt->data(),
                            NX_, NY_, NZ_, Tier::Stencil, nimbus::parity::ComparisonDomain::StrictInterior);
                    }
                    return;
                }
            }
        }
    }

    if (native_unsplit_ && !parity_replay_)
        throw std::runtime_error("native unsplit slow tendency dispatch failed");

    // Centered factors kept for the dp/dt advection (acoustically inert);
    // momentum advection below uses upwind-biased derivatives for the
    // scale-selective dissipation documented in the science audit.
    const double inv_2dx = 1.0 / (2.0 * dx_);
    const double inv_2dy = 1.0 / (2.0 * dy_);
    const double inv_2dz = 1.0 / (2.0 * dz_);
    const double inv_dz  = 1.0 / dz_;
    const double g_local = dynamics_constants::g;

    // One worker team owns output initialization and all four independent
    // slow-tendency components. The initialization barrier is required;
    // component passes write disjoint fields and can overlap without barriers.
    #pragma omp parallel
    {
    #pragma omp for collapse(2)
    for (int i = 0; i < NX_; ++i)
        for (int j = 0; j < NY_; ++j)
            for (int k = 0; k < NZ_; ++k)
            {
                du_dt[i][j][k] = 0.0f;
                dv_dt[i][j][k] = 0.0f;
                dw_dt[i][j][k] = 0.0f;
                drho_dt[i][j][k] = 0.0f;
                dp_dt[i][j][k] = 0.0f;
                if (advection_outputs != nullptr)
                {
                    (*advection_outputs->du_dt)[i][j][k] = 0.0f;
                    (*advection_outputs->dv_dt)[i][j][k] = 0.0f;
                    (*advection_outputs->dw_dt)[i][j][k] = 0.0f;
                }
            }

    // ===== du_x/dt at x-face: advection =====
    // k = 0 is prognostic (SCIENCE.md M19: the old k=1 floor plus
    // the RigidSlip k0:=k1 copy enslaved the lowest momentum level and
    // erased low-level shear). The ground enters through the implicit
    // zero-normal-flux face and a free-slip mirror ghost (du/dzeta = 0 below the
    // first level); RigidNoSlip still pins u[0]=v[0]=0 in the BC pass.
    #pragma omp for collapse(2) nowait
    for (int i = 1; i < NX_ - 1; ++i)
        for (int j = 1; j < NY_ - 1; ++j)
            for (int k = 0; k < NZ_ - 1; ++k)
            {
                const double ux = u[i][j][k];
                const double uy_at_xface = 0.25 * (
                      static_cast<double>(v[i    ][j - 1][k])
                    + static_cast<double>(v[i    ][j    ][k])
                    + static_cast<double>(v[i + 1][j - 1][k])
                    + static_cast<double>(v[i + 1][j    ][k]));
                const double transport_ux = ux - frame_u_ms;
                const double transport_uy_at_xface =
                    uy_at_xface - frame_v_ms;
                // At k=0, interpolate between the upper stored faces and the
                // implicit impermeable ground. Terrain's physical ground w is
                // the surface-relative wind dotted with the slope even though its
                // contravariant flux is zero.
                const double w_at_xface = (k == 0)
                    ? (terrain_metrics != nullptr
                        ? 0.25 * (
                            terrain_transport::cgrid_ground_vertical_velocity(
                                *terrain_metrics, u, v, i, j,
                                tmv::physical_grid_metrics::surface_velocity_ms(*state_).u_ms,
                                tmv::physical_grid_metrics::surface_velocity_ms(*state_).v_ms)
                            + terrain_transport::cgrid_ground_vertical_velocity(
                                *terrain_metrics, u, v, i + 1, j,
                                tmv::physical_grid_metrics::surface_velocity_ms(*state_).u_ms,
                                tmv::physical_grid_metrics::surface_velocity_ms(*state_).v_ms)
                            + static_cast<double>(w[i][j][0])
                            + static_cast<double>(w[i + 1][j][0]))
                        : 0.25 * (static_cast<double>(w[i    ][j][0])
                                + static_cast<double>(w[i + 1][j][0])))
                    : 0.25 * (
                      static_cast<double>(w[i    ][j][k - 1])
                    + static_cast<double>(w[i    ][j][k    ])
                    + static_cast<double>(w[i + 1][j][k - 1])
                    + static_cast<double>(w[i + 1][j][k    ]));
                const double zeta_velocity = terrain_metrics != nullptr
                    ? terrain_transport::cgrid_zeta_velocity_at_x_face(
                        *terrain_metrics, transport_ux,
                        transport_uy_at_xface, w_at_xface,
                        i, j, k)
                    : w_at_xface;

                const double dux_dx = deriv_.upwind_dx(
                    u, transport_ux, i, j, k);
                const double dux_dy = deriv_.upwind_dy(
                    u, transport_uy_at_xface, i, j, k);
                // k=0: free-slip mirror ghost u[-1] = u[0] - advection
                // from below carries zero gradient; from above, first
                // order one-sided.
                const double dux_dz = (k == 0)
                    ? ((zeta_velocity >= 0.0)
                        ? 0.0
                        : (static_cast<double>(u[i][j][1])
                         - static_cast<double>(u[i][j][0])) * inv_dz)
                    : deriv_.upwind_dz(u, zeta_velocity, i, j, k);

                // f-plane Coriolis on the PERTURBATION wind (SCIENCE.md
                // M1; mirrors the collocated scheme and CM1's pertcor
                // convention - full-wind f would accelerate the whole
                // domain absent a balancing geostrophic pressure gradient).
                // coriolis_f is 0 when coriolis.enabled is false.
                const double v0_k = (terrain_metrics != nullptr
                                     && state_->base.v0.size()
                                        == static_cast<std::size_t>(NZ_))
                    ? terrain_transport::reference_profile_value_fp32(
                        state_->base.v0, 0.5 * (terrain_metrics->z(i, j, k)
                                      + terrain_metrics->z(i + 1, j, k)), dz_)
                    : ((static_cast<int>(state_->base.v0.size()) > k)
                        ? state_->base.v0[k] : 0.0);
                double coriolis_x = coriolis.f_at_grid_y0
                    * (uy_at_xface - v0_k);
                if (!coriolis.is_f_plane())
                {
                    const double f_lo = coriolis.at_grid_y(
                        (static_cast<double>(j) + 0.25) * dy_);
                    const double f_hi = coriolis.at_grid_y(
                        (static_cast<double>(j) + 0.75) * dy_);
                    coriolis_x = 0.25 * (
                        f_lo * (static_cast<double>(v[i][j - 1][k])
                              + static_cast<double>(v[i + 1][j - 1][k])
                              - 2.0 * v0_k)
                      + f_hi * (static_cast<double>(v[i][j][k])
                              + static_cast<double>(v[i + 1][j][k])
                              - 2.0 * v0_k));
                }
                const double du = -transport_ux * dux_dx
                                - transport_uy_at_xface * dux_dy
                                - zeta_velocity * dux_dz
                                + coriolis_x;
                du_dt[i][j][k] = static_cast<float>(kw_cart::finite_or_zero(du));
                if (advection_outputs != nullptr)
                {
                    const double advection = -transport_ux * dux_dx
                        - transport_uy_at_xface * dux_dy
                        - zeta_velocity * dux_dz;
                    (*advection_outputs->du_dt)[i][j][k] =
                        static_cast<float>(kw_cart::finite_or_zero(advection));
                }
            }

    // ===== du_y/dt at y-face: advection (k = 0 prognostic; see du_x) =====
    #pragma omp for collapse(2) nowait
    for (int i = 1; i < NX_ - 1; ++i)
        for (int j = 1; j < NY_ - 1; ++j)
            for (int k = 0; k < NZ_ - 1; ++k)
            {
                const double uy = v[i][j][k];
                const double ux_at_yface = 0.25 * (
                      static_cast<double>(u[i - 1][j    ][k])
                    + static_cast<double>(u[i    ][j    ][k])
                    + static_cast<double>(u[i - 1][j + 1][k])
                    + static_cast<double>(u[i    ][j + 1][k]));
                const double transport_uy = uy - frame_v_ms;
                const double transport_ux_at_yface =
                    ux_at_yface - frame_u_ms;
                const double w_at_yface = (k == 0)
                    ? (terrain_metrics != nullptr
                        ? 0.25 * (
                            terrain_transport::cgrid_ground_vertical_velocity(
                                *terrain_metrics, u, v, i, j,
                                tmv::physical_grid_metrics::surface_velocity_ms(*state_).u_ms,
                                tmv::physical_grid_metrics::surface_velocity_ms(*state_).v_ms)
                            + terrain_transport::cgrid_ground_vertical_velocity(
                                *terrain_metrics, u, v, i, j + 1,
                                tmv::physical_grid_metrics::surface_velocity_ms(*state_).u_ms,
                                tmv::physical_grid_metrics::surface_velocity_ms(*state_).v_ms)
                            + static_cast<double>(w[i][j][0])
                            + static_cast<double>(w[i][j + 1][0]))
                        : 0.25 * (static_cast<double>(w[i][j    ][0])
                                + static_cast<double>(w[i][j + 1][0])))
                    : 0.25 * (
                      static_cast<double>(w[i][j    ][k - 1])
                    + static_cast<double>(w[i][j    ][k    ])
                    + static_cast<double>(w[i][j + 1][k - 1])
                    + static_cast<double>(w[i][j + 1][k    ]));
                const double zeta_velocity = terrain_metrics != nullptr
                    ? terrain_transport::cgrid_zeta_velocity_at_y_face(
                        *terrain_metrics, transport_ux_at_yface,
                        transport_uy, w_at_yface,
                        i, j, k)
                    : w_at_yface;

                const double duy_dx = deriv_.upwind_dx(
                    v, transport_ux_at_yface, i, j, k);
                const double duy_dy = deriv_.upwind_dy(
                    v, transport_uy, i, j, k);
                const double duy_dz = (k == 0)
                    ? ((zeta_velocity >= 0.0)
                        ? 0.0
                        : (static_cast<double>(v[i][j][1])
                         - static_cast<double>(v[i][j][0])) * inv_dz)
                    : deriv_.upwind_dz(v, zeta_velocity, i, j, k);

                // f-plane Coriolis on the perturbation wind (M1; see du_x).
                const double u0_k = (terrain_metrics != nullptr
                                     && state_->base.u0.size()
                                        == static_cast<std::size_t>(NZ_))
                    ? terrain_transport::reference_profile_value_fp32(
                        state_->base.u0, 0.5 * (terrain_metrics->z(i, j, k)
                                      + terrain_metrics->z(i, j + 1, k)), dz_)
                    : ((static_cast<int>(state_->base.u0.size()) > k)
                        ? state_->base.u0[k] : 0.0);
                double coriolis_y = coriolis.f_at_grid_y0
                    * (ux_at_yface - u0_k);
                if (!coriolis.is_f_plane())
                {
                    const double f_lo = coriolis.at_grid_y(
                        (static_cast<double>(j) + 0.75) * dy_);
                    const double f_hi = coriolis.at_grid_y(
                        (static_cast<double>(j) + 1.25) * dy_);
                    coriolis_y = 0.25 * (
                        f_lo * (static_cast<double>(u[i - 1][j][k])
                              + static_cast<double>(u[i][j][k])
                              - 2.0 * u0_k)
                      + f_hi * (static_cast<double>(u[i - 1][j + 1][k])
                              + static_cast<double>(u[i][j + 1][k])
                              - 2.0 * u0_k));
                }
                const double du = -transport_ux_at_yface * duy_dx
                                - transport_uy * duy_dy
                                - zeta_velocity * duy_dz
                                - coriolis_y;
                dv_dt[i][j][k] = static_cast<float>(kw_cart::finite_or_zero(du));
                if (advection_outputs != nullptr)
                {
                    const double advection = -transport_ux_at_yface * duy_dx
                        - transport_uy * duy_dy
                        - zeta_velocity * duy_dz;
                    (*advection_outputs->dv_dt)[i][j][k] =
                        static_cast<float>(kw_cart::finite_or_zero(advection));
                }
            }

    // ===== dw/dt at z-face: advection + buoyancy (+ moisture) =====
    #pragma omp for collapse(2) nowait
    for (int i = 1; i < NX_ - 1; ++i)
        for (int j = 1; j < NY_ - 1; ++j)
            for (int k = 0; k < NZ_ - 1; ++k)
            {
                const double wz = w[i][j][k];
                const double ux_at_wface = 0.25 * (
                      static_cast<double>(u[i - 1][j][k    ])
                    + static_cast<double>(u[i    ][j][k    ])
                    + static_cast<double>(u[i - 1][j][k + 1])
                    + static_cast<double>(u[i    ][j][k + 1]));
                const double uy_at_wface = 0.25 * (
                      static_cast<double>(v[i][j - 1][k    ])
                    + static_cast<double>(v[i][j    ][k    ])
                    + static_cast<double>(v[i][j - 1][k + 1])
                    + static_cast<double>(v[i][j    ][k + 1]));
                const double transport_ux_at_wface =
                    ux_at_wface - frame_u_ms;
                const double transport_uy_at_wface =
                    uy_at_wface - frame_v_ms;
                const double zeta_velocity = terrain_metrics != nullptr
                    ? terrain_transport::cgrid_zeta_velocity_at_z_face(
                        *terrain_metrics, transport_ux_at_wface,
                        transport_uy_at_wface, wz,
                        i, j, k)
                    : wz;

                const double dw_dx = deriv_.upwind_dx(
                    w, transport_ux_at_wface, i, j, k);
                const double dw_dy = deriv_.upwind_dy(
                    w, transport_uy_at_wface, i, j, k);
                // Vertical: w has a virtual impermeable ground face. Its
                // physical value is zero only for flat terrain.
                // At k=0 the wide upwind stencil has no cell below, so use a
                // 1st-order upwind that honors the ground BC; k>=1 uses upwind_dz.
                const double ground_w = terrain_metrics != nullptr
                    ? terrain_transport::cgrid_ground_vertical_velocity(
                        *terrain_metrics, u, v, i, j,
                        tmv::physical_grid_metrics::surface_velocity_ms(*state_).u_ms,
                        tmv::physical_grid_metrics::surface_velocity_ms(*state_).v_ms)
                    : 0.0;
                const double dw_dz = (k == 0)
                    ? ((zeta_velocity >= 0.0)
                        ? (static_cast<double>(w[i][j][0]) - ground_w) * inv_dz
                        : (static_cast<double>(w[i][j][1])
                         - static_cast<double>(w[i][j][0])) * inv_dz)
                    : deriv_.upwind_dz(w, zeta_velocity, i, j, k);

                const double rho_face  = 0.5 * (static_cast<double>(rho[i][j][k])
                                              + static_cast<double>(rho[i][j][k + 1]));
                const double rho_safe  = kw_cart::safe_rho(rho_face);
                const double rho0_face = terrain_metrics != nullptr
                    ? 0.5 * (
                        terrain_transport::reference_density_fp32(
                            state_->base.p0, state_->base.theta0,
                            terrain_metrics->z(i, j, k), dz_)
                        + terrain_transport::reference_density_fp32(
                            state_->base.p0, state_->base.theta0,
                            terrain_metrics->z(i, j, k + 1), dz_))
                    : 0.5 * (state_->base.rho0[k] + state_->base.rho0[k + 1]);
                const double buoyancy  = -g_local * (rho_face - rho0_face) / rho_safe;

                const double moisture_buoyancy =
                    (terrain_metrics != nullptr && !state_->fields.qv.empty())
                    ? g_local * 0.608 * (
                        0.5 * (static_cast<double>(state_->fields.qv[i][j][k])
                             + static_cast<double>(state_->fields.qv[i][j][k + 1]))
                        - 0.5 * (
                            terrain_transport::reference_profile_value_fp32(
                                state_->base.qv0, terrain_metrics->z(i, j, k), dz_)
                            + terrain_transport::reference_profile_value_fp32(
                                state_->base.qv0, terrain_metrics->z(i, j, k + 1), dz_)))
                    : kw_cart::moisture_buoyancy_z_face(
                        state_->fields.qv, state_->base.qv0, g_local, i, j, k);

                double loading = 0.0;
                if (!state_->fields.qc.empty())
                {
                    loading = -g_local * kw_cart::hydrometeor_mixing_ratio_z_face(
                        state_->fields.qc, state_->fields.qr, state_->fields.qi, state_->fields.qs, state_->fields.qg, state_->fields.qh, i, j, k);
                }

                const double dw = -transport_ux_at_wface * dw_dx
                                - transport_uy_at_wface * dw_dy
                                - zeta_velocity * dw_dz
                                + buoyancy + moisture_buoyancy + loading;
                dw_dt[i][j][k] = static_cast<float>(kw_cart::finite_or_zero(dw));
                if (advection_outputs != nullptr)
                {
                    const double advection = -transport_ux_at_wface * dw_dx
                        - transport_uy_at_wface * dw_dy
                        - zeta_velocity * dw_dz;
                    (*advection_outputs->dw_dt)[i][j][k] =
                        static_cast<float>(kw_cart::finite_or_zero(advection));
                }
            }

    // ===== dp/dt at cell center: advection only (-u.grad p) =====
    #pragma omp for collapse(2)
    for (int i = 1; i < NX_ - 1; ++i)
        for (int j = 1; j < NY_ - 1; ++j)
            for (int k = 1; k < NZ_ - 1; ++k)
            {
                const double u_center = 0.5 * (static_cast<double>(u[i][j][k])
                                             + static_cast<double>(u[i - 1][j][k]));
                const double v_center = 0.5 * (static_cast<double>(v[i][j][k])
                                             + static_cast<double>(v[i][j - 1][k]));
                const double transport_u_center = u_center - frame_u_ms;
                const double transport_v_center = v_center - frame_v_ms;
                const double w_center = 0.5 * (static_cast<double>(w[i][j][k])
                                             + static_cast<double>(w[i][j][k - 1]));

                const double dp_dx = terrain_metrics != nullptr
                    ? 0.5 * (
                        terrain_transport::cgrid_pressure_perturbation_gradient_x(
                            *terrain_metrics, p, state_->base.p0, i - 1, j, k)
                        + terrain_transport::cgrid_pressure_perturbation_gradient_x(
                            *terrain_metrics, p, state_->base.p0, i, j, k))
                    : (static_cast<double>(p[i + 1][j][k])
                      - static_cast<double>(p[i - 1][j][k])) * inv_2dx;
                const double dp_dy = terrain_metrics != nullptr
                    ? 0.5 * (
                        terrain_transport::cgrid_pressure_perturbation_gradient_y(
                            *terrain_metrics, p, state_->base.p0, i, j - 1, k)
                        + terrain_transport::cgrid_pressure_perturbation_gradient_y(
                            *terrain_metrics, p, state_->base.p0, i, j, k))
                    : (static_cast<double>(p[i][j + 1][k])
                      - static_cast<double>(p[i][j - 1][k])) * inv_2dy;
                const double dp_dz = terrain_metrics != nullptr
                    ? terrain_transport::cgrid_scalar_gradient_z_center(
                        *terrain_metrics, p, i, j, k)
                    : (static_cast<double>(p[i][j][k + 1])
                      - static_cast<double>(p[i][j][k - 1])) * inv_2dz;

                const double dp = -transport_u_center * dp_dx
                                - transport_v_center * dp_dy
                                - w_center * dp_dz;
                dp_dt[i][j][k] = static_cast<float>(kw_cart::finite_or_zero(dp));
            }
    }
    if (mapped_momentum_capture)
    {
        const double flux_dt = dt * momentum_interface_flux_slow_weight_;
        momentum_interface_flux_register_->stage_advective_mapped(
            rho, u, v, w, *terrain_metrics, flux_dt,
            frame_u_ms, frame_v_ms);
        momentum_interface_flux_register_->stage_mapped_tendency_closure(
            rho, du_dt, dv_dt, dw_dt, *terrain_metrics, flux_dt);
        momentum_interface_flux_register_->validate_staged_commit();
        momentum_interface_flux_register_->commit_staged();
    }
}

// =====================================================================
// Fast pressure: drho/dt = -div(rho u) (conservative), dp/dt = -gamma p div(u).
// =====================================================================
CartesianCGrid3DScheme::MovingDensityStep CartesianCGrid3DScheme::prepare_moving_density_step(
    const PreparedEarthFixedTerrainMotion& motion) const
{
    if (state_ == nullptr || state_->terrain_runtime == nullptr
        || state_->domain_frame == nullptr || !state_->grid.geometry.staggered
        || state_->terrain_runtime->earth_fixed_placement.sampling != EarthFixedTerrainSampling::CellAverageBilinear
        || state_->time_s != motion.begin_time_s
        || !can_commit_earth_fixed_terrain(motion.terrain, *state_->terrain_runtime)
        || motion.end_time_s <= motion.begin_time_s
        || motion.swept_volumes.interval_s != motion.end_time_s - motion.begin_time_s)
        throw std::invalid_argument("C-grid moving continuity requires a current cell-average geometry interval");
    if (state_->compute_runtime != nullptr
        && (state_->compute_runtime->active_kind != ComputeBackendKind::Cpu || state_->compute_runtime->fallback_active))
        throw std::invalid_argument("C-grid moving continuity requires CPU-owned execution");
    const auto& before = state_->terrain_runtime->metrics;
    const auto& stage = motion.stage_geometry.metrics;
    const auto& after = motion.terrain.geometry.metrics;
    const auto& swept = motion.swept_volumes;
    const auto& rho = state_->fields.rho;
    const auto& u = state_->fields.u;
    const auto& v = state_->fields.v;
    const auto& w = state_->fields.w;
    terrain_transport::validate_cgrid_state(before, rho, u, v, w);
    terrain_transport::validate_cgrid_state(stage, rho, u, v, w);
    terrain_transport::validate_cgrid_state(after, rho, u, v, w);
    const int nr = rho.size_r(), nth = rho.size_th(), nz = rho.size_z();
    const auto has_shape = [](const TerrainField3D& field, int nx, int ny, int levels) {
        return field.nr == nx && field.nth == ny && field.nz == levels
            && field.values.size() == static_cast<std::size_t>(nx)*ny*levels;
    };
    if (!has_shape(swept.x,nr+1,nth,nz) || !has_shape(swept.y,nr,nth+1,nz)
        || !has_shape(swept.z,nr,nth,nz+1)
        || !std::all_of(swept.z.values.begin(), swept.z.values.end(), [](double value) { return value == 0.0; }))
        throw std::invalid_argument("Earth-fixed continuity cannot sweep mass through stationary terrain surfaces");
    const auto expected = state_->domain_frame->earth_origin_at(motion.end_time_s);
    if (expected.x_m != motion.terrain.next_placement.grid_origin_earth_m.x_m
        || expected.y_m != motion.terrain.next_placement.grid_origin_earth_m.y_m)
        throw std::invalid_argument("moving continuity geometry and frame deadlines disagree");
    const auto domain = physical_scalar_domain(state_->grid);
    if (nr != state_->grid.nr || nth != state_->grid.nth || nz != state_->grid.nz
        || domain.empty() || domain.i_begin < 1 || domain.j_begin < 1)
        throw std::invalid_argument("moving C-grid continuity requires matching physical and closure storage");
    MovingDensityStep result;
    auto& flux = result.integrated_relative_mass_flux;
    flux.x.resize(nr+1,nth,nz,0.0);
    flux.y.resize(nr,nth+1,nz,0.0);
    flux.z.resize(nr,nth,nz+1,0.0);
    const double dt = swept.interval_s;
    for (int i = domain.i_begin; i <= domain.i_end; ++i)
        for (int j = domain.j_begin; j < domain.j_end; ++j)
            for (int k = domain.k_begin; k < domain.k_end; ++k)
            {
                const double face_density = 0.5 * (static_cast<double>(rho(i-1,j,k)) + rho(i,j,k));
                flux.x(i,j,k) = dt * stage.dy * stage.dzeta
                    * terrain_transport::cgrid_mass_flux_x(stage,rho,u,i-1,j,k)
                    - face_density * swept.x(i,j,k);
            }
    for (int i = domain.i_begin; i < domain.i_end; ++i)
        for (int j = domain.j_begin; j <= domain.j_end; ++j)
            for (int k = domain.k_begin; k < domain.k_end; ++k)
            {
                const double face_density = 0.5 * (static_cast<double>(rho(i,j-1,k)) + rho(i,j,k));
                flux.y(i,j,k) = dt * stage.dx * stage.dzeta
                    * terrain_transport::cgrid_mass_flux_y(stage,rho,v,i,j-1,k)
                    - face_density * swept.y(i,j,k);
            }
    // These surfaces are fixed in Earth coordinates. Their physical mass flux
    // uses Earth-relative wind; the horizontal mesh velocity is not subtracted.
    for (int i = domain.i_begin; i < domain.i_end; ++i)
        for (int j = domain.j_begin; j < domain.j_end; ++j)
            for (int face = domain.k_begin+1; face <= domain.k_end; ++face)
                flux.z(i,j,face) = dt * stage.dx * stage.dy
                    * terrain_transport::cgrid_mass_flux_zeta(stage,rho,u,v,w,i,j,face-1);
    result.inventory = update_moving_mesh_inventory(before, after, flux, domain, rho, result.density, 0.0);
    for (int i = domain.i_begin; i < domain.i_end; ++i)
        for (int j = domain.j_begin; j < domain.j_end; ++j)
            for (int k = domain.k_begin; k < domain.k_end; ++k)
                if (result.density(i,j,k) <= 0.0f)
                    throw std::domain_error("moving continuity exhausted physical dry mass");
    return result;
}

void CartesianCGrid3DScheme::compute_fast_pressure_tendencies(
    const Field3D& u, const Field3D& v, const Field3D& w,
    const Field3D& rho, const Field3D& p,
    double /*dt_small*/,
    Field3D& drho_dt, Field3D& dp_dt)
{
    const double gamma_local = dynamics_constants::gamma;
    const double frame_u_ms = (*state_->domain_frame).velocity_ms.u_ms;
    const double frame_v_ms = (*state_->domain_frame).velocity_ms.v_ms;

    kw_cart::zero_fields(NX_, NY_, NZ_, drho_dt, dp_dt);

    const TerrainMetrics3D* terrain_metrics =
        tmv::physical_grid_metrics::terrain_is_active(*state_)
        ? tmv::physical_grid_metrics::active_terrain_metrics(*state_) : nullptr;
    if (terrain_metrics != nullptr)
    {
        terrain_transport::validate_cgrid_shapes(
            *terrain_metrics, rho, u, v, w);
    }

    // Conservative mass continuity. The divergence closes the ground with
    // zero contravariant mass flux and the lid with w[NZ-1]=0, so k spans the
    // full column.
    #pragma omp parallel for collapse(2)
    for (int i = 1; i < NX_ - 1; ++i)
        for (int j = 1; j < NY_ - 1; ++j)
            for (int k = 0; k < NZ_; ++k)
            {
                double drho = 0.0;
                if (terrain_metrics != nullptr)
                {
                    drho = -terrain_transport::cgrid_mass_divergence(
                        *terrain_metrics, rho, u, v, w, i, j, k,
                        frame_u_ms, frame_v_ms);
                }
                else if (frame_u_ms == 0.0 && frame_v_ms == 0.0)
                {
                    // Preserve the stationary arithmetic path exactly.
                    drho = -deriv_.div_flux_mass(rho, u, v, w, i, j, k);
                }
                else
                {
                    drho = -flat_cgrid_mass_divergence_in_frame(*state_,
                        rho, u, v, w, NX_, NY_, NZ_, dx_, dy_, dz_,
                        i, j, k, frame_u_ms, frame_v_ms);
                }
                drho_dt[i][j][k] = static_cast<float>(kw_cart::finite_or_zero(drho));
            }

    // Pressure tendency -gamma*p*div(u), full column (the acoustic restoring
    // force at the boundary-adjacent w-faces requires evolving k=0 and k=NZ-1).
    #pragma omp parallel for collapse(2)
    for (int i = 1; i < NX_ - 1; ++i)
        for (int j = 1; j < NY_ - 1; ++j)
            for (int k = 0; k < NZ_; ++k)
            {
                const double div = terrain_metrics != nullptr
                    ? terrain_transport::cgrid_velocity_divergence(
                        *terrain_metrics, u, v, w, i, j, k,
                        frame_u_ms, frame_v_ms)
                    : deriv_.div_flux(u, v, w, i, j, k);
                const double dp  = -gamma_local * static_cast<double>(p[i][j][k]) * div;
                dp_dt[i][j][k] = static_cast<float>(kw_cart::finite_or_zero(dp));
            }
}

// =====================================================================
// Fast momentum: -grad(p)/rho on u,v; -(dp/dz - dp0/dz)/rho on w.
// =====================================================================
void CartesianCGrid3DScheme::compute_fast_momentum_tendencies(
    const Field3D& u, const Field3D& v, const Field3D& w,
    const Field3D& rho, const Field3D& p,
    double dt_small,
    Field3D& du_dt, Field3D& dv_dt, Field3D& dw_dt)
{
    const double inv_dz = 1.0 / dz_;
    const TerrainMetrics3D* terrain_metrics =
        tmv::physical_grid_metrics::terrain_is_active(*state_)
        ? tmv::physical_grid_metrics::active_terrain_metrics(*state_) : nullptr;

    if (momentum_interface_flux_register_ && terrain_metrics && !parity_replay_)
    {
        require_conservative_momentum_fields(NX_, NY_, NZ_, dt_small, u, v, w, rho, p, p,
                                             du_dt, dv_dt, dw_dt, du_dt, dv_dt);
        if (dispatch_conservative_momentum(*state_, *momentum_interface_flux_register_, u, v, w, rho, p,
                dt_small, CartesianCoriolisProfile{}, true,
                {&du_dt, &dv_dt, &dw_dt, nullptr, nullptr, nullptr, nullptr, nullptr}))
            return;
    }
    if (momentum_interface_flux_register_ != nullptr
        && terrain_metrics == nullptr)
    {
        require_conservative_momentum_fields(
            NX_, NY_, NZ_, dt_small, u, v, w, rho, p, p,
            du_dt, dv_dt, dw_dt, du_dt, dv_dt);
        if (tmv::physical_grid_metrics::terrain_is_active(*state_))
            throw std::runtime_error(
                "conservative C-grid pressure flux capture requires a flat level");
        if (state_->base.p0.size() != static_cast<std::size_t>(NZ_))
            throw std::runtime_error(
                "conservative C-grid pressure flux capture requires a complete base pressure");
        if (!parity_replay_ && dispatch_conservative_momentum(*state_, *momentum_interface_flux_register_,
            u, v, w, rho, p, dt_small, CartesianCoriolisProfile{}, true,
            {&du_dt, &dv_dt, &dw_dt, nullptr, nullptr, nullptr, nullptr, nullptr}))
            return;
        momentum_interface_flux_register_->stage_pressure(
            p, dt_small, dx_, dy_, dz_);
        Field3D candidate_u(NX_, NY_, NZ_, 0.0f);
        Field3D candidate_v(NX_, NY_, NZ_, 0.0f);
        Field3D candidate_w(NX_, NY_, NZ_, 0.0f);
        std::atomic<bool> finite{true};
        #pragma omp parallel for collapse(2)
        for (int i = 1; i < NX_ - 1; ++i)
            for (int j = 1; j < NY_ - 1; ++j)
                for (int k = 0; k < NZ_ - 1; ++k)
                {
                    const double rho_face =
                        cartesian_cgrid_momentum_flux::rho_x(rho, i, j, k);
                    const double value = -(
                        static_cast<double>(p[i + 1][j][k])
                      - static_cast<double>(p[i][j][k])) / (dx_ * rho_face);
                    if (!std::isfinite(value) ||
                        std::abs(value) > std::numeric_limits<float>::max())
                        finite.store(false, std::memory_order_relaxed);
                    else
                        candidate_u[i][j][k] = static_cast<float>(value);
                }
        #pragma omp parallel for collapse(2)
        for (int i = 1; i < NX_ - 1; ++i)
            for (int j = 1; j < NY_ - 1; ++j)
                for (int k = 0; k < NZ_ - 1; ++k)
                {
                    const double rho_face =
                        cartesian_cgrid_momentum_flux::rho_y(rho, i, j, k);
                    const double value = -(
                        static_cast<double>(p[i][j + 1][k])
                      - static_cast<double>(p[i][j][k])) / (dy_ * rho_face);
                    if (!std::isfinite(value) ||
                        std::abs(value) > std::numeric_limits<float>::max())
                        finite.store(false, std::memory_order_relaxed);
                    else
                        candidate_v[i][j][k] = static_cast<float>(value);
                }
        #pragma omp parallel for collapse(2)
        for (int i = 1; i < NX_ - 1; ++i)
            for (int j = 1; j < NY_ - 1; ++j)
                for (int k = 0; k < NZ_ - 1; ++k)
                {
                    const double rho_face =
                        cartesian_cgrid_momentum_flux::rho_z(rho, i, j, k);
                    const double pressure_jump =
                        static_cast<double>(p[i][j][k + 1])
                      - static_cast<double>(p[i][j][k])
                      - (state_->base.p0[static_cast<std::size_t>(k + 1)]
                       - state_->base.p0[static_cast<std::size_t>(k)]);
                    const double value = -pressure_jump / (dz_ * rho_face);
                    if (!std::isfinite(value) ||
                        std::abs(value) > std::numeric_limits<float>::max())
                        finite.store(false, std::memory_order_relaxed);
                    else
                        candidate_w[i][j][k] = static_cast<float>(value);
                }
        if (!finite.load(std::memory_order_relaxed))
            throw std::runtime_error(
                "conservative C-grid pressure momentum candidate is not representable");
        std::array<double, 3> pressure_source{};
        const auto footprint = momentum_interface_flux_register_->footprint();
        const double volume_dt = dx_ * dy_ * dz_ * dt_small;
        for (int i = footprint.i_begin; i < footprint.i_end; ++i)
            for (int j = footprint.j_begin; j < footprint.j_end; ++j)
                for (int k = footprint.k_begin; k < footprint.k_end; ++k)
                    pressure_source[2] +=
                        cartesian_cgrid_momentum_flux::rho_z(rho, i, j, k) *
                        static_cast<double>(candidate_w[i][j][k]) *
                        volume_dt;
        momentum_interface_flux_register_->stage_pressure_source_impulse(
            pressure_source);
        momentum_interface_flux_register_->validate_staged_commit();
        const std::size_t bytes = u.size() * sizeof(float);
        std::memcpy(du_dt.data(), candidate_u.data(), bytes);
        std::memcpy(dv_dt.data(), candidate_v.data(), bytes);
        std::memcpy(dw_dt.data(), candidate_w.data(), bytes);
        momentum_interface_flux_register_->commit_staged();
        return;
    }

    kw_cart::zero_fields(NX_, NY_, NZ_, du_dt, dv_dt, dw_dt);

    if (terrain_metrics != nullptr)
    {
        terrain_transport::validate_cgrid_shapes(
            *terrain_metrics, rho, u, v, w);
        if (state_->base.p0.size() != static_cast<std::size_t>(NZ_))
            throw std::invalid_argument(
                "terrain C-grid pressure force requires a full reference-pressure profile");
        for (double value : state_->base.p0)
            if (!std::isfinite(value) || value <= 0.0)
                throw std::invalid_argument(
                    "terrain C-grid reference pressure must be positive and finite");
    }

    // du_x/dt at x-face: -grad_x(p)/rho_face. Adjacent-point grad_x is what
    // removes the checkerboard from the null space. k = 0 is prognostic
    // (M19; purely horizontal gradient, no vertical dependence).
    #pragma omp parallel for collapse(2)
    for (int i = 1; i < NX_ - 1; ++i)
        for (int j = 1; j < NY_ - 1; ++j)
            for (int k = 0; k < NZ_ - 1; ++k)
            {
                const double dp_dx = terrain_metrics != nullptr
                    ? terrain_transport::cgrid_pressure_perturbation_gradient_x(
                        *terrain_metrics, p, state_->base.p0, i, j, k)
                    : deriv_.grad_x(p, i, j, k);
                const double rho_face = 0.5 * (static_cast<double>(rho[i    ][j][k])
                                             + static_cast<double>(rho[i + 1][j][k]));
                const double du = -dp_dx / kw_cart::safe_rho(rho_face);
                du_dt[i][j][k] = static_cast<float>(kw_cart::finite_or_zero(du));
            }

    // du_y/dt at y-face: -grad_y(p)/rho_face (k = 0 prognostic; M19).
    #pragma omp parallel for collapse(2)
    for (int i = 1; i < NX_ - 1; ++i)
        for (int j = 1; j < NY_ - 1; ++j)
            for (int k = 0; k < NZ_ - 1; ++k)
            {
                const double dp_dy = terrain_metrics != nullptr
                    ? terrain_transport::cgrid_pressure_perturbation_gradient_y(
                        *terrain_metrics, p, state_->base.p0, i, j, k)
                    : deriv_.grad_y(p, i, j, k);
                const double rho_face = 0.5 * (static_cast<double>(rho[i][j    ][k])
                                             + static_cast<double>(rho[i][j + 1][k]));
                const double du = -dp_dy / kw_cart::safe_rho(rho_face);
                dv_dt[i][j][k] = static_cast<float>(kw_cart::finite_or_zero(du));
            }

    // dw/dt at z-face: -(dp/dz - dp0/dz)/rho_face. Reference-state subtraction
    // keeps hydrostatic balance at fp32.
    #pragma omp parallel for collapse(2)
    for (int i = 1; i < NX_ - 1; ++i)
        for (int j = 1; j < NY_ - 1; ++j)
            for (int k = 0; k < NZ_ - 1; ++k)
            {
                const double dp_prime_dz = terrain_metrics != nullptr
                    ? terrain_transport::cgrid_pressure_perturbation_gradient_z(
                        *terrain_metrics, p, state_->base.p0, i, j, k)
                    : deriv_.grad_z(p, i, j, k)
                        - (state_->base.p0[k + 1] - state_->base.p0[k]) * inv_dz;
                const double rho_face    = 0.5 * (static_cast<double>(rho[i][j][k])
                                                + static_cast<double>(rho[i][j][k + 1]));
                const double dw = -dp_prime_dz / kw_cart::safe_rho(rho_face);
                dw_dt[i][j][k] = static_cast<float>(kw_cart::finite_or_zero(dw));
            }
    if (momentum_interface_flux_register_ != nullptr)
    {
        momentum_interface_flux_register_->stage_pressure_mapped(
            p, *terrain_metrics, dt_small);
        momentum_interface_flux_register_->stage_mapped_tendency_closure(
            rho, du_dt, dv_dt, dw_dt, *terrain_metrics, dt_small);
        momentum_interface_flux_register_->validate_staged_commit();
        momentum_interface_flux_register_->commit_staged();
    }
}

bool CartesianCGrid3DScheme::apply_fast_substep_cpu_in_place(
    Field3D& u, Field3D& v, Field3D& w,
    Field3D& rho, Field3D& p, const Field3D& theta,
    double dt_small, Field3D& drho_scratch)
{
    const double frame_u_ms = (*state_->domain_frame).velocity_ms.u_ms;
    const double frame_v_ms = (*state_->domain_frame).velocity_ms.v_ms;
    const TerrainMetrics3D* terrain_metrics =
        tmv::physical_grid_metrics::terrain_is_active(*state_)
        ? tmv::physical_grid_metrics::active_terrain_metrics(*state_) : nullptr;
    if (terrain_metrics != nullptr)
    {
        // The fused callback still contains a flat-grid momentum update.
        // Fail over before touching state so orchestration uses the separate
        // terrain-aware fast-pressure and fast-momentum callbacks.
        terrain_transport::validate_cgrid_shapes(
            *terrain_metrics, rho, u, v, w);
        return false;
    }

    if (drho_scratch.size_r() != NX_ || drho_scratch.size_th() != NY_
        || drho_scratch.size_z() != NZ_)
    {
        drho_scratch.resize(NX_, NY_, NZ_, 0.0f);
    }

    const double inv_dz = 1.0 / dz_;
    // One worker team spans continuity, the EOS barrier, and all three
    // momentum components. drho_scratch is the only required tendency field:
    // continuity must finish everywhere before any density cell changes.
    #pragma omp parallel
    {
        #pragma omp for collapse(2)
        for (int i = 0; i < NX_; ++i)
            for (int j = 0; j < NY_; ++j)
                for (int k = 0; k < NZ_; ++k)
                {
                    double drho = 0.0;
                    if (i >= 1 && i < NX_ - 1
                        && j >= 1 && j < NY_ - 1)
                    {
                        drho = (frame_u_ms == 0.0 && frame_v_ms == 0.0)
                            ? -deriv_.div_flux_mass(rho, u, v, w, i, j, k)
                            : -flat_cgrid_mass_divergence_in_frame(*state_,
                                rho, u, v, w, NX_, NY_, NZ_,
                                dx_, dy_, dz_, i, j, k,
                                frame_u_ms, frame_v_ms);
                    }
                    drho_scratch[i][j][k] = static_cast<float>(
                        kw_cart::finite_or_zero(drho));
                }

        #pragma omp for collapse(2)
        for (int i = 0; i < NX_; ++i)
            for (int j = 0; j < NY_; ++j)
                for (int k = 0; k < NZ_; ++k)
                {
                    const double drho_d = static_cast<double>(
                        drho_scratch[i][j][k]);
                    double rho_new = static_cast<double>(rho[i][j][k])
                                   + drho_d * dt_small;
                    if (primary_state_repairs_enabled
                        && (!std::isfinite(rho_new) || rho_new <= 0.0))
                    {
                        rho_new = std::max(0.1, state_->base.rho0[k]);
                    }
                    const float rho_stored = clamp_density_kgm3(
                        static_cast<float>(rho_new));
                    rho[i][j][k] = rho_stored;

                    double p_new = std::numeric_limits<double>::quiet_NaN();
                    const double th = static_cast<double>(theta[i][j][k]);
                    if (th > 0.0)
                    {
                        p_new = dynamics_eos::pressure_selected(
                            rho_stored, static_cast<float>(th));
                    }
                    if (primary_state_repairs_enabled
                        && (!std::isfinite(p_new) || p_new <= 0.0))
                    {
                        p_new = p0;
                    }
                    p[i][j][k] = clamp_pressure_pa(
                        static_cast<float>(p_new));
                }

        #pragma omp for collapse(2)
        for (int i = 0; i < NX_; ++i)
            for (int j = 0; j < NY_; ++j)
                for (int k = 0; k < NZ_; ++k)
                {
                    float du_stored = 0.0f;
                    float dv_stored = 0.0f;
                    float dw_stored = 0.0f;
                    if (i >= 1 && i < NX_ - 1
                        && j >= 1 && j < NY_ - 1 && k < NZ_ - 1)
                    {
                        const double rho_x = 0.5 * (
                            static_cast<double>(rho[i][j][k])
                            + static_cast<double>(rho[i + 1][j][k]));
                        const double rho_y = 0.5 * (
                            static_cast<double>(rho[i][j][k])
                            + static_cast<double>(rho[i][j + 1][k]));
                        const double rho_z = 0.5 * (
                            static_cast<double>(rho[i][j][k])
                            + static_cast<double>(rho[i][j][k + 1]));
                        const double dp0_dz =
                            (state_->base.p0[k + 1] - state_->base.p0[k]) * inv_dz;
                        du_stored = static_cast<float>(kw_cart::finite_or_zero(
                            -deriv_.grad_x(p, i, j, k)
                            / kw_cart::safe_rho(rho_x)));
                        dv_stored = static_cast<float>(kw_cart::finite_or_zero(
                            -deriv_.grad_y(p, i, j, k)
                            / kw_cart::safe_rho(rho_y)));
                        dw_stored = static_cast<float>(kw_cart::finite_or_zero(
                            -(deriv_.grad_z(p, i, j, k) - dp0_dz)
                            / kw_cart::safe_rho(rho_z)));
                    }

                    u[i][j][k] = clamp_wind_horizontal_ms(
                        static_cast<float>(static_cast<double>(u[i][j][k])
                                           + static_cast<double>(du_stored)
                                               * dt_small));
                    v[i][j][k] = clamp_wind_horizontal_ms(
                        static_cast<float>(static_cast<double>(v[i][j][k])
                                           + static_cast<double>(dv_stored)
                                               * dt_small));
                    w[i][j][k] = clamp_wind_vertical_ms(
                        static_cast<float>(static_cast<double>(w[i][j][k])
                                           + static_cast<double>(dw_stored)
                                               * dt_small));
                }
    }

    return true;
}

// =====================================================================
// The unsplit RHS uses direct rates; rounded acoustic state updates cannot
// recover these derivatives without cancellation error.
// =====================================================================
void CartesianCGrid3DScheme::compute_momentum_tendencies(const Field3D& u,
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
    const bool native = !parity_replay_ && state_->compute_runtime &&
                        state_->compute_runtime->active_kind != ComputeBackendKind::Cpu &&
                        !state_->compute_runtime->fallback_active;
    if (native)
    {
        if (!state_->compute_runtime->backend || !state_->compute_runtime->backend->supports_rk_tendency() ||
            tmv::physical_grid_metrics::terrain_is_active(*state_) || momentum_interface_flux_register_ != nullptr ||
            state_->domain_frame->velocity_ms.u_ms != 0 || state_->domain_frame->velocity_ms.v_ms != 0)
            throw std::runtime_error("native unsplit C-grid momentum requires stationary flat geometry");
        if (state_->base.p0.size() != std::size_t(NZ_))
            throw std::invalid_argument("native unsplit C-grid momentum requires a full base pressure");
        for (double value : state_->base.p0)
            if (!std::isfinite(value) || value <= 0 || !std::isfinite(float(value)))
                throw std::invalid_argument("native unsplit C-grid base pressure is invalid");
        for (auto& field : native_rk_base_pressure_)
            if (field.size_r() != NX_ || field.size_th() != NY_ || field.size_z() != NZ_)
                field.resize(NX_, NY_, NZ_, 0);
        for (std::size_t n = 0; n < p.size(); ++n)
        {
            const double value                    = state_->base.p0[n % NZ_];
            native_rk_base_pressure_[0].data()[n] = float(value);
            native_rk_base_pressure_[1].data()[n] = float(value - float(value));
        }
        auto&              backend = *state_->compute_runtime->backend;
        RkTendencyDispatch d;
        d.nx                 = NX_;
        d.ny                 = NY_;
        d.nz                 = NZ_;
        d.dx                 = dx_;
        d.dy                 = dy_;
        d.dz                 = dz_;
        d.dt                 = dt;
        d.gamma              = dynamics_constants::gamma;
        d.input              = {u.data(),
                                v.data(),
                                w.data(),
                                rho.data(),
                                p.data(),
                                native_rk_base_pressure_[0].data(),
                                native_rk_base_pressure_[1].data()};
        const auto& acoustic = native_rk_.run(backend, d);
        struct NativeScope
        {
            bool& active;
            explicit NativeScope(bool& flag) : active(flag)
            {
                active = true;
            }
            ~NativeScope()
            {
                active = false;
            }
        } native_scope(native_unsplit_);
        compute_slow_tendencies(u, v, w, rho, p, theta, dt, du_dt, dv_dt, dw_dt, drho_dt, dp_dt);
        const std::array<Field3D*, 5> result{&du_dt, &dv_dt, &dw_dt, &drho_dt, &dp_dt};
        d.operation = RkTendencyOperation::Add;
        for (std::size_t f = 0; f < result.size(); ++f)
        {
            d.input         = {result[f]->data(), acoustic[f].data()};
            const auto& sum = native_rk_.run(backend, d, 1);
            tmv::NativeRkTendencyWorkspace::copy(*result[f], sum[0]);
        }
        return;
    }
    // Slow pass populates du/dv/dw (advection + buoyancy), dp (advection),
    // and zeroes drho.
    compute_slow_tendencies(u, v, w, rho, p, theta, dt, du_dt, dv_dt, dw_dt, drho_dt, dp_dt);

    // Fast passes into temporaries, then add: velocity += -grad(p)/rho,
    // drho = conservative continuity, dp += -gamma*p*div(u).
    auto& pool       = FieldPool::instance();
    auto  du_guard   = pool.scoped_acquire_for_overwrite(NX_, NY_, NZ_);
    auto  dv_guard   = pool.scoped_acquire_for_overwrite(NX_, NY_, NZ_);
    auto  dw_guard   = pool.scoped_acquire_for_overwrite(NX_, NY_, NZ_);
    auto  drho_guard = pool.scoped_acquire_for_overwrite(NX_, NY_, NZ_);
    auto  dp_guard   = pool.scoped_acquire_for_overwrite(NX_, NY_, NZ_);
    auto& du_f       = du_guard.field;
    auto& dv_f       = dv_guard.field;
    auto& dw_f       = dw_guard.field;
    auto& drho_f     = drho_guard.field;
    auto& dp_f       = dp_guard.field;
    compute_fast_pressure_tendencies(u, v, w, rho, p, dt, drho_f, dp_f);
    compute_fast_momentum_tendencies(u, v, w, rho, p, dt, du_f, dv_f, dw_f);

#pragma omp parallel for collapse(2)
    for (int i = 0; i < NX_; ++i)
        for (int j = 0; j < NY_; ++j)
            for (int k = 0; k < NZ_; ++k)
            {
                du_dt[i][j][k] += du_f[i][j][k];
                dv_dt[i][j][k] += dv_f[i][j][k];
                dw_dt[i][j][k] += dw_f[i][j][k];
                drho_dt[i][j][k] = drho_f[i][j][k]; // slow drho is 0
                dp_dt[i][j][k] += dp_f[i][j][k];
            }
}

} // namespace tmv
