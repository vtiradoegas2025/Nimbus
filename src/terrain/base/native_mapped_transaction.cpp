#include "terrain/native_mapped_transaction.hpp"

#include "compute/backend_runtime.hpp"
#include "core/field/field_bounds.hpp"
#include "core/infra/physical_constants.hpp"
#include "core/infra/physical_cell_domain.hpp"
#include "core/infra/physical_grid_metrics.hpp"
#include "core/runtime/simulation_state.hpp"
#include "dynamics/device_slow_energy_attribution.hpp"
#include "dynamics/native_cgrid_boundary.hpp"
#include "diagnostics/deterministic_reduction.hpp"
#include "diagnostics/energy_budget.hpp"
#include "terrain/terrain_base.hpp"
#include "terrain/conservative_transport.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

namespace tmv::terrain_native
{
namespace
{

struct CandidateFields
{
    Field3D u;
    Field3D v;
    Field3D w;
    Field3D rho;
    Field3D p;
    Field3D theta;
    Field3D qv;
    Field3D tracer;
    Field3D tke;
};

struct MetricFields
{
    std::vector<float> height;
    std::vector<float> jacobian;
    std::vector<float> jacobian_x;
    std::vector<float> jacobian_y;
    std::vector<float> metric_x;
    std::vector<float> metric_y;
    std::vector<float> ground_metric_x;
    std::vector<float> ground_metric_y;
};

void copy_field(Field3D& destination, const Field3D& source)
{
    destination.resize(source.size_r(), source.size_th(), source.size_z(), 0.0f);
    std::copy_n(source.data(), source.size(), destination.data());
}

Field3D field_from_vector(const std::vector<float>& source, int nx, int ny, int nz)
{
    Field3D field(nx, ny, nz, 0.0f);
    std::copy(source.begin(), source.end(), field.data());
    return field;
}

SimulationStateView candidate_view(const SimulationStateView& source, CandidateFields& candidate)
{
    SimulationStateView view{source.grid,
                             {candidate.rho,
                              candidate.p,
                              candidate.u,
                              candidate.w,
                              candidate.v,
                              candidate.tracer,
                              candidate.theta,
                              candidate.qv,
                              source.fields.qc,
                              source.fields.qr,
                              source.fields.qi,
                              source.fields.qs,
                              source.fields.qh,
                              source.fields.qg,
                              candidate.tke},
                             source.base,
                             source.tendencies,
                             source.coriolis_f,
                             source.time_s,
                             source.surface};
    view.radiation_runtime          = source.radiation_runtime;
    view.turbulence_runtime         = source.turbulence_runtime;
    view.boundary_layer_runtime     = source.boundary_layer_runtime;
    view.chaos_runtime              = source.chaos_runtime;
    view.rayleigh_runtime           = source.rayleigh_runtime;
    view.microphysics_runtime       = source.microphysics_runtime;
    view.advection_runtime          = source.advection_runtime;
    view.diffusion_runtime          = source.diffusion_runtime;
    view.time_stepping_runtime      = source.time_stepping_runtime;
    view.terrain_runtime            = source.terrain_runtime;
    view.radar_runtime              = source.radar_runtime;
    view.compute_runtime            = source.compute_runtime;
    view.parity_runtime             = source.parity_runtime;
    view.configuration              = source.configuration;
    view.diagnostics                = source.diagnostics;
    view.dynamics_runtime           = source.dynamics_runtime;
    view.runtime_owner              = source.runtime_owner;
    view.domain_frame               = source.domain_frame;
    view.boundary_condition_runtime = source.boundary_condition_runtime;
    return view;
}

bool pack_metrics(const TerrainMetrics3D& metrics, int nx, int ny, int nz, MetricFields& packed)
{
    const auto shape = [](const TerrainField3D& field, int x, int y, int z)
    { return field.size_r() == x && field.size_th() == y && field.size_z() == z; };
    if (!shape(metrics.z, nx, ny, nz) || !shape(metrics.J, nx, ny, nz) || !shape(metrics.Jx_face, nx + 1, ny, nz) ||
        !shape(metrics.Jy_face, nx, ny + 1, nz) || !shape(metrics.Gx_face, nx, ny, nz + 1) ||
        !shape(metrics.Gy_face, nx, ny, nz + 1) || !std::isfinite(metrics.dx) || metrics.dx <= 0.0 ||
        !std::isfinite(metrics.dy) || metrics.dy <= 0.0 || !std::isfinite(metrics.dzeta) || metrics.dzeta <= 0.0)
        return false;
    const std::size_t sx = static_cast<std::size_t>(nx);
    const std::size_t sy = static_cast<std::size_t>(ny);
    const std::size_t sz = static_cast<std::size_t>(nz);
    if (sx > std::numeric_limits<std::size_t>::max() / sy || sx * sy > std::numeric_limits<std::size_t>::max() / sz)
        return false;
    const std::size_t count = sx * sy * sz;
    for (auto* field : {&packed.height,
                        &packed.jacobian,
                        &packed.jacobian_x,
                        &packed.jacobian_y,
                        &packed.metric_x,
                        &packed.metric_y,
                        &packed.ground_metric_x,
                        &packed.ground_metric_y})
        field->resize(count);
    for (int i = 0; i < nx; ++i)
        for (int j = 0; j < ny; ++j)
            for (int k = 0; k < nz; ++k)
            {
                const std::size_t n       = (static_cast<std::size_t>(i) * ny + j) * nz + k;
                packed.height[n]          = static_cast<float>(metrics.z(i, j, k));
                packed.jacobian[n]        = static_cast<float>(metrics.J(i, j, k));
                packed.jacobian_x[n]      = i < nx - 1 ? static_cast<float>(metrics.Jx_face(i + 1, j, k)) : 0.0f;
                packed.jacobian_y[n]      = j < ny - 1 ? static_cast<float>(metrics.Jy_face(i, j + 1, k)) : 0.0f;
                packed.metric_x[n]        = static_cast<float>(metrics.Gx_face(i, j, k + 1));
                packed.metric_y[n]        = static_cast<float>(metrics.Gy_face(i, j, k + 1));
                packed.ground_metric_x[n] = static_cast<float>(metrics.Gx_face(i, j, 0));
                packed.ground_metric_y[n] = static_cast<float>(metrics.Gy_face(i, j, 0));
            }
    if (!std::all_of(metrics.J.values.begin(),
                     metrics.J.values.end(),
                     [](double value) { return std::isfinite(value) && value > 0.0; }))
        return false;
    for (const auto* field : {&packed.height,
                              &packed.jacobian,
                              &packed.jacobian_x,
                              &packed.jacobian_y,
                              &packed.metric_x,
                              &packed.metric_y,
                              &packed.ground_metric_x,
                              &packed.ground_metric_y})
        if (!std::all_of(field->begin(), field->end(), [](float value) { return std::isfinite(value); }))
            return false;
    return true;
}

std::vector<float> canonical_reference_state(const SimulationStateView& state, const TerrainMetrics3D& metrics)
{
    const auto count  = metrics.z.size();
    const auto levels = state.base.p0.size();
    if (levels > std::numeric_limits<std::uint32_t>::max() ||
        count > (std::numeric_limits<std::uint32_t>::max() - levels) / 3)
        throw std::length_error("canonical mapped reference exceeds device indexing");
    std::vector<float> result(state.base.p0.begin(), state.base.p0.end());
    result.resize(levels + 3 * count);
    for (std::size_t n = 0; n < count; ++n)
    {
        const double height = metrics.z.values[n];
        result[levels + 3 * n] =
            float(terrain_transport::reference_profile_value_fp32(state.base.p0, height, metrics.dzeta));
        result[levels + 3 * n + 1] =
            float(terrain_transport::reference_density_fp32(state.base.p0, state.base.theta0, height, metrics.dzeta));
        result[levels + 3 * n + 2] =
            state.base.qv0.empty()
                ? 0.0f
                : float(terrain_transport::reference_profile_value_fp32(state.base.qv0, height, metrics.dzeta));
    }
    return result;
}
bool finite_field(const Field3D& field)
{
    return std::all_of(field.data(), field.data() + field.size(), [](float value) { return std::isfinite(value); });
}

bool positive_field(const Field3D& field)
{
    return std::all_of(
        field.data(), field.data() + field.size(), [](float value) { return std::isfinite(value) && value > 0.0f; });
}

bool nonnegative_field(const Field3D& field)
{
    return std::all_of(
        field.data(), field.data() + field.size(), [](float value) { return std::isfinite(value) && value >= 0.0f; });
}

bool zero_field(const Field3D& field)
{
    return std::all_of(field.data(), field.data() + field.size(), [](float value) { return value == 0.0f; });
}

bool distinct_live_fields(std::initializer_list<const Field3D*> fields, std::size_t bytes) noexcept
{
    if (bytes == 0)
        return false;
    for (auto first = fields.begin(); first != fields.end(); ++first)
    {
        const auto first_address = reinterpret_cast<std::uintptr_t>((*first)->data());
        for (auto second = std::next(first); second != fields.end(); ++second)
        {
            const auto second_address = reinterpret_cast<std::uintptr_t>((*second)->data());
            const auto distance =
                first_address < second_address ? second_address - first_address : first_address - second_address;
            if (distance < bytes)
                return false;
        }
    }
    return true;
}

std::array<double, 6> reservoir_difference(const MoistMixtureEnergyBudgetV2& before,
                                           const MoistMixtureEnergyBudgetV2& after)
{
    return {{
        after.dry_internal - before.dry_internal,
        after.water_sensible - before.water_sensible,
        after.phase_reference - before.phase_reference,
        after.resolved_kinetic - before.resolved_kinetic,
        after.gravitational - before.gravitational,
        after.subgrid_tke - before.subgrid_tke,
    }};
}

void accumulate_reservoir_difference(CartesianCGridAcousticEnergyReportV1& report, const std::array<double, 6>& change)
{
    report.continuity_eos_dry_internal_change_j += change[0];
    report.continuity_eos_water_sensible_change_j += change[1];
    report.continuity_eos_phase_reference_change_j += change[2];
    report.continuity_carrier_kinetic_change_j += change[3];
    report.continuity_gravitational_change_j += change[4];
    report.continuity_subgrid_tke_change_j += change[5];
}

double mixture_factor(const SimulationStateView&, int, int, int);

double requested_carrier_change(const SimulationStateView& state, const Field3D& drho_dt, float dt, bool moist = false)
{
    const PhysicalCellDomain domain         = physical_scalar_domain(state.grid);
    const auto               before         = compute_moist_mixture_energy_v2_deterministic(state);
    const double             total_dry_mass = tmv::physical_grid_metrics::ControlVolumeMassView(state).scalar_mass_kg(
        state.fields.rho, domain.i_begin, domain.j_begin, domain.k_begin);
    if (!std::isfinite(before.resolved_kinetic) || !std::isfinite(total_dry_mass) || total_dry_mass <= 0.0)
        throw std::runtime_error("mapped continuity carrier attribution received invalid state");

    const auto& fields  = state.fields;
    const auto* metrics = tmv::physical_grid_metrics::active_terrain_metrics(state);
    const tmv::physical_grid_metrics::ControlVolumeMassView volumes(state);
    double                                                  requested = 0.0;
    for (int i = domain.i_begin; i < domain.i_end; ++i)
        for (int j = domain.j_begin; j < domain.j_end; ++j)
            for (int k = domain.k_begin; k < domain.k_end; ++k)
            {
                const double u_w = fields.u[i - 1][j][k];
                const double u_e = fields.u[i][j][k];
                const double v_s = fields.v[i][j - 1][k];
                const double v_n = fields.v[i][j][k];
                const double w_b = k == domain.k_begin ? 0.5 * (u_w + u_e) * (-metrics->Gx_face(i, j, 0)) +
                                                             0.5 * (v_s + v_n) * (-metrics->Gy_face(i, j, 0))
                                                       : static_cast<double>(fields.w[i][j][k - 1]);
                const double w_t = fields.w[i][j][k];
                const double coefficient =
                    0.25 * (u_w * u_w + u_e * u_e + v_s * v_s + v_n * v_n + w_b * w_b + w_t * w_t);
                requested += static_cast<double>(drho_dt[i][j][k] * dt) * volumes.scalar_volume_m3(i, j, k) *
                             coefficient * (moist ? mixture_factor(state, i, j, k) : 1.0);
            }
    if (!std::isfinite(requested))
        throw std::runtime_error("mapped continuity carrier attribution overflowed");
    return requested;
}

struct ScalarAttributionContext
{
    const SimulationStateView*          state             = nullptr;
    const Field3D*                      scalar_begin      = nullptr;
    const Field3D*                      density_begin     = nullptr;
    const Field3D*                      density_end       = nullptr;
    const Field3D*                      pressure          = nullptr;
    const MetricFields*                 metrics           = nullptr;
    float                               dt                = 0.0f;
    bool                                positive_definite = false;
    ScalarMoistEnergyKind               energy_kind       = ScalarMoistEnergyKind::None;
    ScalarBoundaryTransportReport*      boundary          = nullptr;
    ScalarMoistEnergyTransportReportV1* energy            = nullptr;
};

void attribute_scalar_commit(void*        raw,
                             const float* scalar_end,
                             const float* scalar_flux_x,
                             const float* scalar_flux_y,
                             const float* scalar_flux_z,
                             const float* limiter_factors)
{
    const auto& context = *static_cast<const ScalarAttributionContext*>(raw);
    const auto& state   = *context.state;
    const auto& grid    = state.grid;
    const auto  domain  = physical_scalar_domain(grid);
    const int   ny      = grid.nth;
    const int   nz      = grid.nz;
    const auto  at      = [ny, nz](int i, int j, int k) { return (static_cast<std::size_t>(i) * ny + j) * nz + k; };
    const auto  limited_flux = [&](float raw_flux, std::size_t low, std::size_t high)
    {
        if (!context.positive_definite)
            return raw_flux;
        return raw_flux * (raw_flux > 0.0f ? limiter_factors[low] : limiter_factors[high]);
    };
    const double      dx         = grid.dr;
    const double      dy         = grid.dtheta;
    const auto*       terrain    = physical_grid_metrics::active_terrain_metrics(state);
    const double      dzeta      = terrain->dzeta;
    const long double cell_scale = static_cast<long double>(dx) * dy * dzeta;
    const long double area_x     = static_cast<long double>(dy) * dzeta;
    const long double area_y     = static_cast<long double>(dx) * dzeta;
    const long double area_z     = static_cast<long double>(dx) * dy;

    long double                outward_rate        = 0.0L;
    long double                energy_outward_rate = 0.0L;
    long double                begin_inventory     = 0.0L;
    long double                end_inventory       = 0.0L;
    std::array<long double, 6> committed_reservoirs{};
    long double                density_history  = 0.0L;
    long double                requested_change = 0.0L;

    constexpr double cv_d  = physical_constants::specific_heat_cp_jkgk - physical_constants::gas_constant_dry_air_jkgk;
    constexpr double kappa = physical_constants::gas_constant_dry_air_jkgk / physical_constants::specific_heat_cp_jkgk;
    const auto       coefficient = [&](std::size_t index)
    {
        if (context.energy_kind != ScalarMoistEnergyKind::PotentialTemperature)
            return 0.0;
        return cv_d * std::pow(static_cast<double>(context.pressure->data()[index]) /
                                   physical_constants::reference_pressure_pa,
                               kappa);
    };

    const int west_face = domain.i_begin - 1;
    const int east_face = domain.i_end - 1;
    for (int j = domain.j_begin; j < domain.j_end; ++j)
        for (int k = domain.k_begin; k < domain.k_end; ++k)
        {
            const std::size_t west      = at(west_face, j, k);
            const std::size_t east      = at(east_face, j, k);
            const float       west_flux = limited_flux(scalar_flux_x[west], west, at(west_face + 1, j, k));
            const float       east_flux = limited_flux(scalar_flux_x[east], east, at(east_face + 1, j, k));
            outward_rate += area_x * static_cast<long double>(east_flux - west_flux);
            energy_outward_rate +=
                area_x * (static_cast<long double>(east_flux) * coefficient(at(domain.i_end - 1, j, k)) -
                          static_cast<long double>(west_flux) * coefficient(at(domain.i_begin, j, k)));
        }
    const int south_face = domain.j_begin - 1;
    const int north_face = domain.j_end - 1;
    for (int i = domain.i_begin; i < domain.i_end; ++i)
        for (int k = domain.k_begin; k < domain.k_end; ++k)
        {
            const std::size_t south      = at(i, south_face, k);
            const std::size_t north      = at(i, north_face, k);
            const float       south_flux = limited_flux(scalar_flux_y[south], south, at(i, south_face + 1, k));
            const float       north_flux = limited_flux(scalar_flux_y[north], north, at(i, north_face + 1, k));
            outward_rate += area_y * static_cast<long double>(north_flux - south_flux);
            energy_outward_rate +=
                area_y * (static_cast<long double>(north_flux) * coefficient(at(i, domain.j_end - 1, k)) -
                          static_cast<long double>(south_flux) * coefficient(at(i, domain.j_begin, k)));
        }
    const int top_face = domain.k_end - 1;
    for (int i = domain.i_begin; i < domain.i_end; ++i)
        for (int j = domain.j_begin; j < domain.j_end; ++j)
        {
            const std::size_t top      = at(i, j, top_face);
            const float       top_flux = limited_flux(scalar_flux_z[top], top, top);
            outward_rate += area_z * static_cast<long double>(top_flux);
            energy_outward_rate += area_z * static_cast<long double>(top_flux) * coefficient(top);
        }

    const float inv_dx = 1.0f / static_cast<float>(dx);
    const float inv_dy = 1.0f / static_cast<float>(dy);
    const float inv_dz = 1.0f / static_cast<float>(dzeta);
    for (int i = domain.i_begin; i < domain.i_end; ++i)
        for (int j = domain.j_begin; j < domain.j_end; ++j)
            for (int k = domain.k_begin; k < domain.k_end; ++k)
            {
                const std::size_t c                    = at(i, j, k);
                const std::size_t im                   = at(i - 1, j, k);
                const std::size_t jm                   = at(i, j - 1, k);
                const std::size_t km                   = k > 0 ? at(i, j, k - 1) : c;
                const std::size_t ip                   = at(i + 1, j, k);
                const std::size_t jp                   = at(i, j + 1, k);
                const std::size_t kp                   = k + 1 < grid.nz ? at(i, j, k + 1) : c;
                const float       fxp                  = limited_flux(scalar_flux_x[c], c, ip);
                const float       fxm                  = limited_flux(scalar_flux_x[im], im, c);
                const float       fyp                  = limited_flux(scalar_flux_y[c], c, jp);
                const float       fym                  = limited_flux(scalar_flux_y[jm], jm, c);
                const float       fzp                  = limited_flux(scalar_flux_z[c], c, kp);
                const float       fzm                  = k > 0 ? limited_flux(scalar_flux_z[km], km, c) : 0.0f;
                float             requested_rho_scalar = 0.0f;
                if (context.positive_definite)
                {
                    float      outgoing     = 0.0f;
                    float      incoming     = 0.0f;
                    const auto signed_parts = [](float plus, float minus, float inv, float& out, float& in)
                    {
                        if (plus >= 0.0f)
                            out += plus * inv;
                        else
                            in -= plus * inv;
                        if (minus <= 0.0f)
                            out -= minus * inv;
                        else
                            in += minus * inv;
                    };
                    signed_parts(fxp, fxm, inv_dx, outgoing, incoming);
                    signed_parts(fyp, fym, inv_dy, outgoing, incoming);
                    signed_parts(fzp, fzm, inv_dz, outgoing, incoming);
                    requested_rho_scalar =
                        (context.density_begin->data()[c] * context.scalar_begin->data()[c] - context.dt * outgoing) +
                        context.dt * incoming;
                }
                else
                {
                    const float divergence = (fxp - fxm) * inv_dx + (fyp - fym) * inv_dy + (fzp - fzm) * inv_dz;
                    requested_rho_scalar =
                        context.density_begin->data()[c] * context.scalar_begin->data()[c] - context.dt * divergence;
                }
                const long double volume     = cell_scale * context.metrics->jacobian[c];
                const double      old_scalar = context.scalar_begin->data()[c];
                const double      new_scalar = scalar_end[c];
                const double      rho_begin  = context.density_begin->data()[c];
                const double      rho_end    = context.density_end->data()[c];
                const double      coeff      = coefficient(c);
                begin_inventory += volume * rho_begin * old_scalar;
                end_inventory += volume * rho_end * new_scalar;
                committed_reservoirs[0] += volume * rho_end * (new_scalar - old_scalar) * coeff;
                density_history += volume * coeff * (rho_begin - rho_end) * old_scalar;
                requested_change +=
                    volume * coeff * (static_cast<double>(requested_rho_scalar) - rho_begin * old_scalar);
            }

    ScalarBoundaryTransportReport boundary;
    boundary.outward_boundary_inventory = static_cast<double>(outward_rate * context.dt);
    boundary.flux_implied_state_change  = -boundary.outward_boundary_inventory;
    boundary.committed_state_change     = static_cast<double>(end_inventory - begin_inventory);
    boundary.closure_residual           = boundary.committed_state_change - boundary.flux_implied_state_change;
    boundary.complete                   = true;
    ScalarMoistEnergyTransportReportV1 energy;
    energy.dry_internal_change_j       = static_cast<double>(committed_reservoirs[0]);
    energy.density_history_remap_j     = static_cast<double>(density_history);
    energy.outward_boundary_energy_j   = static_cast<double>(energy_outward_rate * context.dt);
    energy.limited_face_state_change_j = static_cast<double>(requested_change);
    energy.interior_conversion_j       = energy.limited_face_state_change_j + energy.outward_boundary_energy_j;
    energy.requested_state_change_j    = energy.density_history_remap_j + energy.limited_face_state_change_j;
    energy.committed_state_change_j    = energy.dry_internal_change_j;
    energy.storage_rounding_j          = energy.committed_state_change_j - energy.requested_state_change_j;
    energy.closure_residual_j =
        energy.committed_state_change_j - (energy.requested_state_change_j + energy.storage_rounding_j);
    energy.complete       = true;
    const double values[] = {
        boundary.outward_boundary_inventory,
        boundary.committed_state_change,
        boundary.closure_residual,
        energy.dry_internal_change_j,
        energy.density_history_remap_j,
        energy.outward_boundary_energy_j,
        energy.limited_face_state_change_j,
        energy.storage_rounding_j,
    };
    if (std::any_of(std::begin(values), std::end(values), [](double value) { return !std::isfinite(value); }))
        throw std::runtime_error("mapped scalar attribution overflowed");
    *context.boundary = boundary;
    *context.energy   = energy;
}

std::array<double, 3> pressure_work(const SimulationStateView&               before,
                                    const CandidateFields&                   after,
                                    const std::array<std::vector<float>, 3>& tendency,
                                    float                                    dt)
{
    const PhysicalCellDomain domain  = physical_scalar_domain(before.grid);
    const auto*              metrics = tmv::physical_grid_metrics::active_terrain_metrics(before);
    const tmv::physical_grid_metrics::ControlVolumeMassView masses(before);
    double                                                  requested   = 0.0;
    double                                                  finite_step = 0.0;
    for (int i = domain.i_begin; i < domain.i_end; ++i)
        for (int j = domain.j_begin; j < domain.j_end; ++j)
            for (int k = domain.k_begin; k < domain.k_end; ++k)
            {
                const std::size_t n      = (static_cast<std::size_t>(i) * before.grid.nth + j) * before.grid.nz + k;
                const int         right  = i + 1 < domain.i_end ? i + 1 : domain.i_begin;
                const int         north  = j + 1 < domain.j_end ? j + 1 : domain.j_begin;
                const double      u_mass = masses.u_dual_mass_kg(before.fields.rho, i, j, k, right);
                const double      v_mass = masses.v_dual_mass_kg(before.fields.rho, i, j, k, north);
                const double w_mass = k + 1 < domain.k_end ? masses.w_dual_mass_kg(before.fields.rho, i, j, k)
                                                           : 0.5 * masses.scalar_mass_kg(before.fields.rho, i, j, k);
                const double du     = static_cast<double>(tendency[0][n] * dt);
                const double dv     = static_cast<double>(tendency[1][n] * dt);
                const double dw     = static_cast<double>(tendency[2][n] * dt);
                requested += u_mass * before.fields.u[i][j][k] * du + v_mass * before.fields.v[i][j][k] * dv +
                             w_mass * before.fields.w[i][j][k] * dw;
                finite_step += 0.5 * (u_mass * du * du + v_mass * dv * dv + w_mass * dw * dw);

                if (k == domain.k_begin)
                {
                    const double slope_x = -metrics->Gx_face(i, j, 0);
                    const double slope_y = -metrics->Gy_face(i, j, 0);
                    const double ground_old =
                        slope_x * 0.5 * (before.fields.u[i - 1][j][k] + before.fields.u[i][j][k]) +
                        slope_y * 0.5 * (before.fields.v[i][j - 1][k] + before.fields.v[i][j][k]);
                    const std::size_t west =
                        (static_cast<std::size_t>(i - 1) * before.grid.nth + j) * before.grid.nz + k;
                    const std::size_t south =
                        (static_cast<std::size_t>(i) * before.grid.nth + j - 1) * before.grid.nz + k;
                    const double ground_delta =
                        slope_x * 0.5 * static_cast<double>((tendency[0][west] + tendency[0][n]) * dt) +
                        slope_y * 0.5 * static_cast<double>((tendency[1][south] + tendency[1][n]) * dt);
                    const double ground_mass = 0.5 * masses.scalar_mass_kg(before.fields.rho, i, j, k);
                    requested += ground_mass * ground_old * ground_delta;
                    finite_step += 0.5 * ground_mass * ground_delta * ground_delta;
                }
            }
    SimulationStateView after_view = candidate_view(before, const_cast<CandidateFields&>(after));
    const double        committed  = compute_moist_mixture_energy_v2_deterministic(after_view).resolved_kinetic -
                                     compute_moist_mixture_energy_v2_deterministic(before).resolved_kinetic;
    const double        values[]   = {requested, finite_step, committed};
    if (std::any_of(std::begin(values), std::end(values), [](double value) { return !std::isfinite(value); }) ||
        finite_step < 0.0)
        throw std::runtime_error("mapped pressure-work attribution received invalid state");
    return {{requested, finite_step, committed}};
}

std::array<double, 6> cell_velocity(const SimulationStateView& s, int i, int j, int k)
{
    std::array<double, 6> v{s.fields.u(i - 1, j, k),
                            s.fields.u(i, j, k),
                            s.fields.v(i, j - 1, k),
                            s.fields.v(i, j, k),
                            k > 0 ? s.fields.w(i, j, k - 1) : 0,
                            s.fields.w(i, j, k)};
    if (k == 0)
    {
        const auto& g = s.terrain_runtime->metrics;
        v[4]          = -.5 * g.Gx_face(i, j, 0) * (v[0] + v[1]) - .5 * g.Gy_face(i, j, 0) * (v[2] + v[3]);
    }
    return v;
}
double mixture_factor(const SimulationStateView& s, int i, int j, int k)
{
    double result = 1;
    for (auto* q : {&s.fields.qv, &s.fields.qc, &s.fields.qr, &s.fields.qi, &s.fields.qs, &s.fields.qg, &s.fields.qh})
        result += (*q)(i, j, k);
    return result;
}
std::array<double, 3> moist_pressure_work(const SimulationStateView&               before,
                                          const CandidateFields&                   after,
                                          const std::array<std::vector<float>, 3>& tendency,
                                          float                                    dt)
{
    const auto                                         domain = physical_scalar_domain(before.grid);
    const physical_grid_metrics::ControlVolumeMassView mass(before);
    const auto&                                        g = before.terrain_runtime->metrics;
    diagnostics::CompensatedSum                        linear, finite;
    const auto                                         delta = [&](int c, int i, int j, int k)
    { return double(tendency[c][(std::size_t(i) * before.grid.nth + j) * before.grid.nz + k] * dt); };
    for (int i = domain.i_begin; i < domain.i_end; ++i)
        for (int j = domain.j_begin; j < domain.j_end; ++j)
            for (int k = domain.k_begin; k < domain.k_end; ++k)
            {
                const auto            old = cell_velocity(before, i, j, k);
                std::array<double, 6> change{delta(0, i - 1, j, k),
                                             delta(0, i, j, k),
                                             delta(1, i, j - 1, k),
                                             delta(1, i, j, k),
                                             k > 0 ? delta(2, i, j, k - 1) : 0,
                                             delta(2, i, j, k)};
                if (k == 0)
                    change[4] = -.5 * g.Gx_face(i, j, 0) * (change[0] + change[1]) -
                                .5 * g.Gy_face(i, j, 0) * (change[2] + change[3]);
                const double m = mass.scalar_mass_kg(before.fields.rho, i, j, k) * mixture_factor(before, i, j, k);
                for (std::size_t f = 0; f < 6; ++f)
                {
                    linear.add(.5 * m * old[f] * change[f]);
                    finite.add(.25 * m * change[f] * change[f]);
                }
            }
    auto         next      = candidate_view(before, const_cast<CandidateFields&>(after));
    const double committed = compute_moist_mixture_energy_v2_deterministic(next).resolved_kinetic -
                             compute_moist_mixture_energy_v2_deterministic(before).resolved_kinetic;
    return {linear.value(), finite.value(), committed};
}
double
carrier_boundary_work(const SimulationStateView& s, const std::array<std::vector<float>, 6>& continuity, float dt)
{
    const auto  d           = physical_scalar_domain(s.grid);
    const auto& g           = s.terrain_runtime->metrics;
    const auto  at          = [&](int i, int j, int k) { return (std::size_t(i) * s.grid.nth + j) * s.grid.nz + k; };
    const auto  coefficient = [&](int i, int j, int k)
    {
        double kinetic = 0;
        for (double v : cell_velocity(s, i, j, k))
            kinetic += .25 * v * v;
        return mixture_factor(s, i, j, k) * kinetic;
    };
    diagnostics::CompensatedSum result;
    using L = init::config::LateralBC;
    for (int j = d.j_begin; j < d.j_end; ++j)
        for (int k = d.k_begin; k < d.k_end; ++k)
        {
            if (s.configuration->lateral_bc_west != L::Periodic)
                result.add(-g.dy * g.dzeta * continuity[3][at(d.i_begin - 1, j, k)] * coefficient(d.i_begin, j, k) *
                           dt);
            if (s.configuration->lateral_bc_east != L::Periodic)
                result.add(g.dy * g.dzeta * continuity[3][at(d.i_end - 1, j, k)] * coefficient(d.i_end - 1, j, k) * dt);
        }
    for (int i = d.i_begin; i < d.i_end; ++i)
        for (int k = d.k_begin; k < d.k_end; ++k)
        {
            if (s.configuration->lateral_bc_south != L::Periodic)
                result.add(-g.dx * g.dzeta * continuity[4][at(i, d.j_begin - 1, k)] * coefficient(i, d.j_begin, k) *
                           dt);
            if (s.configuration->lateral_bc_north != L::Periodic)
                result.add(g.dx * g.dzeta * continuity[4][at(i, d.j_end - 1, k)] * coefficient(i, d.j_end - 1, k) * dt);
        }
    return result.value();
}

} // namespace

static bool advance_mapped_core(const SimulationStateView&  state,
                                double                      dt_large,
                                int                         acoustic_substeps,
                                MappedCoreStepReport&       report,
                                MappedCoreCompletedObserver completed_observer,
                                void*                       completed_observer_context,
                                MappedAcousticResult*       acoustic_result)
{
    const bool acoustic_only = acoustic_result != nullptr;
    if (acoustic_only &&
        (!state.configuration || !state.domain_frame || !state.boundary_condition_runtime ||
         state.configuration->nesting.enabled || state.configuration->top_bc != init::config::TopBC::Rigid ||
         state.domain_frame->velocity_ms.u_ms != 0 || state.domain_frame->velocity_ms.v_ms != 0 ||
         state.coriolis_f != 0))
        return false;
    if (!std::isfinite(dt_large) || dt_large <= 0.0 || acoustic_substeps <= 0 || state.grid.nr < 3 ||
        state.grid.nth < 3 || state.grid.nz < 2 || state.terrain_runtime == nullptr || !std::isfinite(state.grid.dr) ||
        state.grid.dr <= 0.0 || !std::isfinite(state.grid.dtheta) || state.grid.dtheta <= 0.0)
        return false;
    const float dt_large_f  = static_cast<float>(dt_large);
    const float dt_small_f  = static_cast<float>(dt_large / acoustic_substeps);
    const float time_weight = dt_small_f / dt_large_f;
    if (!std::isfinite(dt_large_f) || dt_large_f <= 0.0f || !std::isfinite(dt_small_f) || dt_small_f <= 0.0f ||
        !std::isfinite(time_weight) || time_weight <= 0.0f || time_weight > 1.0f)
        return false;
    ComputeBackend* backend =
        state.compute_runtime != nullptr ? mutable_compute_backend(*state.compute_runtime) : mutable_compute_backend();
    TerrainComputeKernels* kernels = backend != nullptr ? backend->terrain_kernels() : nullptr;
    if (kernels == nullptr || !kernels->supports_mapped_mass_continuity() ||
        !kernels->supports_mapped_fast_momentum() || !kernels->supports_mapped_slow_tendencies() ||
        !kernels->supports_mapped_scalar_transport() || !kernels->supports_mapped_state_update() ||
        !kernels->supports_mapped_flux_accumulation() || !kernels->supports_mapped_final_state())
        return false;

    const int         nx = state.grid.nr;
    const int         ny = state.grid.nth;
    const int         nz = state.grid.nz;
    const std::size_t sx = static_cast<std::size_t>(nx);
    const std::size_t sy = static_cast<std::size_t>(ny);
    const std::size_t sz = static_cast<std::size_t>(nz);
    if (sx > std::numeric_limits<std::size_t>::max() / sy || sx * sy > std::numeric_limits<std::size_t>::max() / sz)
        return false;
    const std::size_t expected_count = sx * sy * sz;
    const std::size_t count          = state.fields.rho.size();
    if (count > std::numeric_limits<std::size_t>::max() / sizeof(float))
        return false;
    if (count != expected_count || state.base.p0.size() != static_cast<std::size_t>(nz) ||
        state.base.theta0.size() != static_cast<std::size_t>(nz))
        return false;
    const auto exact_field_shape = [nx, ny, nz, count](const Field3D& field)
    { return field.size_r() == nx && field.size_th() == ny && field.size_z() == nz && field.size() == count; };
    for (const Field3D* field : {&state.fields.rho,
                                 &state.fields.p,
                                 &state.fields.u,
                                 &state.fields.w,
                                 &state.fields.v,
                                 &state.fields.tracer,
                                 &state.fields.theta,
                                 &state.fields.qv,
                                 &state.fields.qc,
                                 &state.fields.qr,
                                 &state.fields.qi,
                                 &state.fields.qs,
                                 &state.fields.qh,
                                 &state.fields.qg,
                                 &state.fields.tke})
        if (!exact_field_shape(*field))
            return false;
    if (acoustic_only &&
        (!exact_field_shape(state.tendencies.du_dt_pbl) || !exact_field_shape(state.tendencies.dv_dt_pbl) ||
         !finite_field(state.tendencies.du_dt_pbl) || !finite_field(state.tendencies.dv_dt_pbl)))
        return false;
    if (!distinct_live_fields({&state.fields.u,
                               &state.fields.v,
                               &state.fields.w,
                               &state.fields.rho,
                               &state.fields.p,
                               &state.fields.theta,
                               &state.fields.qv,
                               &state.fields.tracer,
                               &state.fields.tke},
                              count * sizeof(float)))
        return false;
    if (!finite_field(state.fields.u) || !finite_field(state.fields.v) || !finite_field(state.fields.w) ||
        !positive_field(state.fields.rho) || !positive_field(state.fields.p) || !positive_field(state.fields.theta) ||
        !nonnegative_field(state.fields.tracer) || !nonnegative_field(state.fields.tke) ||
        !(acoustic_only ? nonnegative_field(state.fields.qv) : zero_field(state.fields.qv)) ||
        !(acoustic_only ? nonnegative_field(state.fields.qc) : zero_field(state.fields.qc)) ||
        !(acoustic_only ? nonnegative_field(state.fields.qr) : zero_field(state.fields.qr)) ||
        !(acoustic_only ? nonnegative_field(state.fields.qi) : zero_field(state.fields.qi)) ||
        !(acoustic_only ? nonnegative_field(state.fields.qs) : zero_field(state.fields.qs)) ||
        !(acoustic_only ? nonnegative_field(state.fields.qg) : zero_field(state.fields.qg)) ||
        !(acoustic_only ? nonnegative_field(state.fields.qh) : zero_field(state.fields.qh)))
        return false;

    const TerrainMetrics3D& metrics         = state.terrain_runtime->metrics;
    const auto              spacing_matches = [](double a, double b)
    {
        return std::isfinite(a) && std::isfinite(b) &&
               std::abs(a - b) <=
                   8.0 * std::numeric_limits<double>::epsilon() * std::max({1.0, std::abs(a), std::abs(b)});
    };
    if (!spacing_matches(state.grid.dr, metrics.dx) || !spacing_matches(state.grid.dtheta, metrics.dy))
        return false;
    MetricFields packed;
    if (!pack_metrics(metrics, nx, ny, nz, packed))
        return false;
    std::vector<float> reference_pressure(state.base.p0.begin(), state.base.p0.end());
    std::vector<float> reference_theta(state.base.theta0.begin(), state.base.theta0.end());
    if (!std::all_of(reference_pressure.begin(),
                     reference_pressure.end(),
                     [](float value) { return std::isfinite(value) && value > 0.0f; }) ||
        !std::all_of(reference_theta.begin(),
                     reference_theta.end(),
                     [](float value) { return std::isfinite(value) && value > 0.0f; }))
        return false;

    try
    {
        reference_pressure = canonical_reference_state(state, metrics);
        CandidateFields candidate;
        copy_field(candidate.u, state.fields.u);
        copy_field(candidate.v, state.fields.v);
        copy_field(candidate.w, state.fields.w);
        copy_field(candidate.rho, state.fields.rho);
        copy_field(candidate.p, state.fields.p);
        copy_field(candidate.theta, state.fields.theta);
        copy_field(candidate.qv, state.fields.qv);
        copy_field(candidate.tracer, state.fields.tracer);
        copy_field(candidate.tke, state.fields.tke);
        BoundaryConditionRuntimeState wave_state;
        if (acoustic_only)
            wave_state = *state.boundary_condition_runtime;
        bool                            wave_seeded = false;
        BoundaryLayerMomentumWorkReport pbl_work;
        const bool active_pbl = acoustic_only && state.boundary_layer_runtime && state.boundary_layer_runtime->scheme &&
                                state.boundary_layer_runtime->scheme->name() != "none";
        if (!active_pbl && (!zero_field(state.tendencies.du_dt_pbl) || !zero_field(state.tendencies.dv_dt_pbl)))
            return false;
        Field3D density_begin;
        copy_field(density_begin, candidate.rho);

        std::array<std::vector<float>, 8> slow;
        std::array<std::vector<float>, 6> continuity;
        std::array<std::vector<float>, 3> fast;
        std::array<std::vector<float>, 3> accumulated;
        std::array<std::vector<float>, 3> accumulated_candidate;
        for (auto& field : slow)
            field.assign(count, 0.0f);
        for (auto& field : continuity)
            field.assign(count, 0.0f);
        for (auto& field : fast)
            field.assign(count, 0.0f);
        for (auto& field : accumulated)
            field.assign(count, 0.0f);
        for (auto& field : accumulated_candidate)
            field.assign(count, 0.0f);
        std::vector<float> disabled_field(count, 0.0f);
        std::vector<float> disabled_profile(static_cast<std::size_t>(nz), 0.0f);

        const auto mass_dispatch = [&]
        {
            MappedTerrainMassContinuityDispatch d;
            d.rho                         = candidate.rho.data();
            d.u                           = candidate.u.data();
            d.v                           = candidate.v.data();
            d.w                           = candidate.w.data();
            d.p                           = candidate.p.data();
            d.jacobian                    = packed.jacobian.data();
            d.jacobian_x_positive_face    = packed.jacobian_x.data();
            d.jacobian_y_positive_face    = packed.jacobian_y.data();
            d.metric_x_positive_zeta_face = packed.metric_x.data();
            d.metric_y_positive_zeta_face = packed.metric_y.data();
            d.metric_x_ground_face        = packed.ground_metric_x.data();
            d.metric_y_ground_face        = packed.ground_metric_y.data();
            d.ground_physical_w           = continuity[0].data();
            d.drho_dt                     = continuity[1].data();
            d.dp_dt                       = continuity[2].data();
            d.mass_flux_x_positive        = continuity[3].data();
            d.mass_flux_y_positive        = continuity[4].data();
            d.mass_flux_zeta_positive     = continuity[5].data();
            d.nx                          = nx;
            d.ny                          = ny;
            d.nz                          = nz;
            d.dx                          = static_cast<float>(state.grid.dr);
            d.dy                          = static_cast<float>(state.grid.dtheta);
            d.dzeta                       = static_cast<float>(metrics.dzeta);
            d.gamma                       = static_cast<float>(physical_constants::dry_air_gamma);
            d.report                      = nullptr;
            return d;
        };

        MappedCoreStepReport              next_report;
        const std::uint64_t               dispatches_before = backend->successful_dispatch_count();
        MappedTerrainMassContinuityReport seed_report;
        auto                              seed = mass_dispatch();
        seed.report                            = &seed_report;
        if (!kernels->dispatch_mapped_mass_continuity(seed) || !seed_report.complete)
            return false;
        next_report.mass_continuity_dispatches += 1;

        std::vector<float> vapor_reference(static_cast<std::size_t>(nz), 0), loading(count, 0);
        bool               moisture = false, has_loading = false;
        if (acoustic_only)
        {
            if (state.base.qv0.size() != std::size_t(nz))
                return false;
            for (int k = 0; k < nz; ++k)
            {
                if (!std::isfinite(state.base.qv0[k]) || state.base.qv0[k] < 0)
                    return false;
                vapor_reference[k] = float(state.base.qv0[k]);
                moisture           = moisture || vapor_reference[k] != 0;
            }
            for (std::size_t n = 0; n < count; ++n)
            {
                moisture     = moisture || state.fields.qv.data()[n] != 0;
                double total = 0;
                for (const Field3D* field : {&state.fields.qc,
                                             &state.fields.qr,
                                             &state.fields.qi,
                                             &state.fields.qs,
                                             &state.fields.qg,
                                             &state.fields.qh})
                    total += field->data()[n];
                loading[n]  = float(total);
                has_loading = has_loading || total != 0;
            }
        }
        MappedTerrainSlowTendenciesReport   slow_report;
        MappedTerrainSlowTendenciesDispatch slow_dispatch;
        slow_dispatch.u                           = candidate.u.data();
        slow_dispatch.v                           = candidate.v.data();
        slow_dispatch.w                           = candidate.w.data();
        slow_dispatch.rho                         = candidate.rho.data();
        slow_dispatch.p                           = candidate.p.data();
        slow_dispatch.physical_height             = packed.height.data();
        slow_dispatch.jacobian_x_positive_face    = packed.jacobian_x.data();
        slow_dispatch.jacobian_y_positive_face    = packed.jacobian_y.data();
        slow_dispatch.metric_x_positive_zeta_face = packed.metric_x.data();
        slow_dispatch.metric_y_positive_zeta_face = packed.metric_y.data();
        slow_dispatch.ground_physical_w           = continuity[0].data();
        slow_dispatch.reference_pressure          = reference_pressure.data();
        slow_dispatch.sampled_reference_state = true;
        slow_dispatch.reference_theta             = reference_theta.data();
        slow_dispatch.qv                          = moisture ? candidate.qv.data() : disabled_field.data();
        slow_dispatch.reference_qv                = vapor_reference.data();
        slow_dispatch.hydrometeor_loading         = loading.data();
        slow_dispatch.du_dt                       = slow[0].data();
        slow_dispatch.dv_dt                       = slow[1].data();
        slow_dispatch.dw_dt                       = slow[2].data();
        slow_dispatch.drho_dt                     = slow[3].data();
        slow_dispatch.dp_dt                       = slow[4].data();
        slow_dispatch.du_advection_dt             = slow[5].data();
        slow_dispatch.dv_advection_dt             = slow[6].data();
        slow_dispatch.dw_advection_dt             = slow[7].data();
        slow_dispatch.nx                          = nx;
        slow_dispatch.ny                          = ny;
        slow_dispatch.nz                          = nz;
        slow_dispatch.has_moisture                = moisture;
        slow_dispatch.has_loading                 = has_loading;
        slow_dispatch.dx                          = static_cast<float>(state.grid.dr);
        slow_dispatch.dy                          = static_cast<float>(state.grid.dtheta);
        slow_dispatch.dzeta                       = static_cast<float>(metrics.dzeta);
        slow_dispatch.nominal_dz                  = static_cast<float>(metrics.dzeta);
        slow_dispatch.gravity                     = static_cast<float>(physical_constants::gravity_ms2);
        slow_dispatch.report                      = &slow_report;
        if (!kernels->dispatch_mapped_slow_tendencies(slow_dispatch) || !slow_report.complete)
            return false;
        next_report.slow_tendency_dispatches = 1;

        const auto update = [&](MappedTerrainStateUpdateMode             mode,
                                float                                    step,
                                const std::array<std::vector<float>, 5>& tendency) -> bool
        {
            if (acoustic_only && mode == MappedTerrainStateUpdateMode::Slow)
            {
                auto* dynamics = backend->dynamics_kernels();
                if (!dynamics)
                    return false;
                SlowApplyDispatch d;
                d.u         = candidate.u.data();
                d.v         = candidate.v.data();
                d.w         = candidate.w.data();
                d.p_data    = candidate.p.data();
                d.du_dt     = tendency[0].data();
                d.dv_dt     = tendency[1].data();
                d.dw_dt     = tendency[2].data();
                d.dp_dt     = tendency[4].data();
                d.du_dt_pbl = active_pbl ? state.tendencies.du_dt_pbl.data() : disabled_field.data();
                d.dv_dt_pbl = active_pbl ? state.tendencies.dv_dt_pbl.data() : disabled_field.data();
                d.nx        = nx;
                d.ny        = ny;
                d.nz        = nz;
                d.dt_large  = step;
                d.sanitize  = false;
                if (!dynamics->dispatch_slow_apply(d))
                    return false;
                ++next_report.state_update_dispatches;
                return true;
            }
            CandidateFields output;
            for (Field3D* field : {&output.u, &output.v, &output.w, &output.rho, &output.p})
                field->resize(nx, ny, nz, 0.0f);
            MappedTerrainStateUpdateReport   update_report;
            MappedTerrainStateUpdateDispatch d;
            d.u             = candidate.u.data();
            d.v             = candidate.v.data();
            d.w             = candidate.w.data();
            d.rho           = candidate.rho.data();
            d.p             = candidate.p.data();
            d.theta         = candidate.theta.data();
            d.du_dt         = tendency[0].data();
            d.dv_dt         = tendency[1].data();
            d.dw_dt         = tendency[2].data();
            d.drho_dt       = tendency[3].data();
            d.dp_dt         = tendency[4].data();
            d.u_candidate   = output.u.data();
            d.v_candidate   = output.v.data();
            d.w_candidate   = output.w.data();
            d.rho_candidate = output.rho.data();
            d.p_candidate   = output.p.data();
            d.nx            = nx;
            d.ny            = ny;
            d.nz            = nz;
            d.mode          = mode;
            d.dt            = step;
            d.report        = &update_report;
            if (!kernels->dispatch_mapped_state_update(d) || !update_report.complete)
                return false;
            candidate.u   = std::move(output.u);
            candidate.v   = std::move(output.v);
            candidate.w   = std::move(output.w);
            candidate.rho = std::move(output.rho);
            candidate.p   = std::move(output.p);
            ++next_report.state_update_dispatches;
            return true;
        };
        const auto boundary = [&]() -> bool
        {
            if (acoustic_only)
            {
                auto view                       = candidate_view(state, candidate);
                view.boundary_condition_runtime = &wave_state;
                apply_native_cgrid_boundary(view, *kernels, !wave_seeded);
                wave_seeded = true;
                ++next_report.boundary_dispatches;
                return true;
            }

            CandidateFields output;
            for (Field3D* field : {&output.u, &output.v, &output.w, &output.rho, &output.p})
                field->resize(nx, ny, nz, 0.0f);
            MappedTerrainFinalStateReport   boundary_report;
            MappedTerrainFinalStateDispatch d;
            d.u         = candidate.u.data();
            d.v         = candidate.v.data();
            d.w         = candidate.w.data();
            d.rho       = candidate.rho.data();
            d.p         = candidate.p.data();
            d.theta     = candidate.theta.data();
            d.u_final   = output.u.data();
            d.v_final   = output.v.data();
            d.w_final   = output.w.data();
            d.rho_final = output.rho.data();
            d.p_final   = output.p.data();
            d.nx        = nx;
            d.ny        = ny;
            d.nz        = nz;
            d.report    = &boundary_report;
            d.apply_eos = false;
            if (!kernels->dispatch_mapped_final_state(d) || !boundary_report.complete ||
                boundary_report.eos_dispatch_count != 0 || boundary_report.boundary_dispatch_count != 1)
                return false;
            candidate.u   = std::move(output.u);
            candidate.v   = std::move(output.v);
            candidate.w   = std::move(output.w);
            candidate.rho = std::move(output.rho);
            candidate.p   = std::move(output.p);
            ++next_report.boundary_dispatches;
            return true;
        };

        const auto acoustic_boundary = [&](float step = 0.0f) -> bool
        {
            if (acoustic_only && step > 0)
            {
                auto* dynamics = backend->dynamics_kernels();
                if (!dynamics)
                    return false;
                auto view                       = candidate_view(state, candidate);
                view.boundary_condition_runtime = &wave_state;
                apply_native_cgrid_acoustic_boundary(view, *dynamics, step, &next_report.acoustic);
                ++next_report.boundary_dispatches;
                return true;
            }
            SimulationStateView before_view = candidate_view(state, candidate);
            const auto          before      = compute_moist_mixture_energy_v2_deterministic(before_view);
            if (!boundary())
                return false;
            SimulationStateView after_view = candidate_view(state, candidate);
            const auto          after      = compute_moist_mixture_energy_v2_deterministic(after_view);
            const double        committed  = after.resolved_kinetic - before.resolved_kinetic;
            if (!std::isfinite(committed))
                throw std::runtime_error("mapped boundary energy attribution overflowed");
            next_report.acoustic.boundary_committed_work_j += committed;
            next_report.acoustic.boundary_constraint_history_committed_work_j += committed;
            ++next_report.acoustic.boundary_application_count;
            next_report.acoustic.boundary_complete = true;
            return true;
        };

        {
            CandidateFields                         base       = candidate;
            const auto                              first_slow = slow;
            const std::array<std::vector<float>, 5> predictor  = {slow[0], slow[1], slow[2], slow[3], slow[4]};
            if (!update(MappedTerrainStateUpdateMode::Slow, dt_large_f, predictor))
                return false;
            MappedTerrainMassContinuityReport predictor_mass;
            auto                              predicted = mass_dispatch();
            predicted.report                            = &predictor_mass;
            if (!kernels->dispatch_mapped_mass_continuity(predicted) || !predictor_mass.complete)
                return false;
            ++next_report.mass_continuity_dispatches;
            slow_dispatch.u   = candidate.u.data();
            slow_dispatch.v   = candidate.v.data();
            slow_dispatch.w   = candidate.w.data();
            slow_dispatch.rho = candidate.rho.data();
            slow_dispatch.p   = candidate.p.data();
            slow_report       = {};
            if (!kernels->dispatch_mapped_slow_tendencies(slow_dispatch) || !slow_report.complete)
                return false;
            ++next_report.slow_tendency_dispatches;
            for (const std::array<std::size_t, 3> indices : {std::array<std::size_t, 3>{0, 1, 2}, {4, 5, 6}, {7, 8, 8}})
            {
                std::array<std::vector<float>, 3> half, mean;
                for (auto& field : half)
                    field.resize(count);
                for (auto& field : mean)
                    field.resize(count);
                for (int stage = 0; stage < 2; ++stage)
                {
                    const auto& source = stage == 0 ? first_slow : slow;
                    const auto  input  = [&](std::size_t slot)
                    { return indices[slot] < source.size() ? source[indices[slot]].data() : disabled_field.data(); };
                    auto&                                 output = stage == 0 ? half : mean;
                    MappedTerrainFluxAccumulationReport   average_report;
                    MappedTerrainFluxAccumulationDispatch average;
                    average.accumulated_x              = stage == 0 ? disabled_field.data() : half[0].data();
                    average.accumulated_y              = stage == 0 ? disabled_field.data() : half[1].data();
                    average.accumulated_zeta           = stage == 0 ? disabled_field.data() : half[2].data();
                    average.step_x                     = input(0);
                    average.step_y                     = input(1);
                    average.step_zeta                  = input(2);
                    average.accumulated_x_candidate    = output[0].data();
                    average.accumulated_y_candidate    = output[1].data();
                    average.accumulated_zeta_candidate = output[2].data();
                    average.nx                         = nx;
                    average.ny                         = ny;
                    average.nz                         = nz;
                    average.time_weight                = 0.5f;
                    average.report                     = &average_report;
                    if (!kernels->dispatch_mapped_flux_accumulation(average) || !average_report.complete)
                        return false;
                    ++next_report.slow_average_dispatches;
                }
                for (std::size_t slot = 0; slot < 3; ++slot)
                    if (indices[slot] < slow.size())
                        slow[indices[slot]] = std::move(mean[slot]);
            }
            candidate = std::move(base);
        }

        std::array<std::vector<float>, 5> slow_update = {slow[0], slow[1], slow[2], slow[3], slow[4]};
        if (!update(MappedTerrainStateUpdateMode::Slow, dt_large_f, slow_update))
            return false;
        Field3D                              slow_u      = field_from_vector(slow[0], nx, ny, nz);
        Field3D                              slow_v      = field_from_vector(slow[1], nx, ny, nz);
        Field3D                              slow_w      = field_from_vector(slow[2], nx, ny, nz);
        Field3D                              advection_u = field_from_vector(slow[5], nx, ny, nz);
        Field3D                              advection_v = field_from_vector(slow[6], nx, ny, nz);
        Field3D                              advection_w = field_from_vector(slow[7], nx, ny, nz);
        SlowMomentumAdvectionTendencyOutputs advection{&advection_u, &advection_v, &advection_w};
        const bool                           wrap_x =
            !acoustic_only || (state.configuration->lateral_bc_west == init::config::LateralBC::Periodic &&
                               state.configuration->lateral_bc_east == init::config::LateralBC::Periodic);
        const bool wrap_y =
            !acoustic_only || (state.configuration->lateral_bc_south == init::config::LateralBC::Periodic &&
                               state.configuration->lateral_bc_north == init::config::LateralBC::Periodic);
        const auto slow_attribution = tmv::device_slow_energy::attribute_completed_commit(state,
                                                                                          slow_u,
                                                                                          slow_v,
                                                                                          slow_w,
                                                                                          &advection,
                                                                                          dt_large_f,
                                                                                          true,
                                                                                          active_pbl,
                                                                                          wrap_x,
                                                                                          wrap_y,
                                                                                          candidate.u.data(),
                                                                                          candidate.v.data(),
                                                                                          candidate.w.data(),
                                                                                          candidate.p.data());
        next_report.slow_momentum   = slow_attribution.momentum;
        pbl_work                    = slow_attribution.pbl;
        if (!next_report.slow_momentum.complete || !acoustic_boundary())
            return false;

        for (int substep = 0; substep < acoustic_substeps; ++substep)
        {
            MappedTerrainMassContinuityReport continuity_report;
            auto                              d = mass_dispatch();
            d.report                            = &continuity_report;
            if (!kernels->dispatch_mapped_mass_continuity(d) || !continuity_report.complete)
                return false;
            ++next_report.mass_continuity_dispatches;
            if (acoustic_only)
            {
                auto view = candidate_view(state, candidate);
                next_report.acoustic.continuity_carrier_outward_boundary_transport_j +=
                    carrier_boundary_work(view, continuity, dt_small_f);
            }

            MappedTerrainFluxAccumulationReport   flux_report;
            MappedTerrainFluxAccumulationDispatch flux_dispatch;
            flux_dispatch.accumulated_x              = accumulated[0].data();
            flux_dispatch.accumulated_y              = accumulated[1].data();
            flux_dispatch.accumulated_zeta           = accumulated[2].data();
            flux_dispatch.step_x                     = continuity[3].data();
            flux_dispatch.step_y                     = continuity[4].data();
            flux_dispatch.step_zeta                  = continuity[5].data();
            flux_dispatch.accumulated_x_candidate    = accumulated_candidate[0].data();
            flux_dispatch.accumulated_y_candidate    = accumulated_candidate[1].data();
            flux_dispatch.accumulated_zeta_candidate = accumulated_candidate[2].data();
            flux_dispatch.nx                         = nx;
            flux_dispatch.ny                         = ny;
            flux_dispatch.nz                         = nz;
            flux_dispatch.time_weight                = time_weight;
            flux_dispatch.report                     = &flux_report;
            if (!kernels->dispatch_mapped_flux_accumulation(flux_dispatch) || !flux_report.complete)
                return false;
            accumulated.swap(accumulated_candidate);
            ++next_report.flux_accumulation_dispatches;
            next_report.accumulated_time_weight += time_weight;

            std::array<std::vector<float>, 5> continuity_update = {
                disabled_field, disabled_field, disabled_field, continuity[1], continuity[2]};
            CandidateFields     continuity_before      = candidate;
            SimulationStateView continuity_before_view = candidate_view(state, continuity_before);
            const auto continuity_energy_before = compute_moist_mixture_energy_v2_deterministic(continuity_before_view);
            if (!update(MappedTerrainStateUpdateMode::ContinuityEos, dt_small_f, continuity_update))
                return false;
            // Fast pressure gradients need the newly projected pressure on both sides of periodic faces.
            if (!boundary())
                return false;
            SimulationStateView continuity_after_view = candidate_view(state, candidate);
            const auto continuity_energy_after = compute_moist_mixture_energy_v2_deterministic(continuity_after_view);
            accumulate_reservoir_difference(next_report.acoustic,
                                            reservoir_difference(continuity_energy_before, continuity_energy_after));
            const Field3D continuity_drho = field_from_vector(continuity[1], nx, ny, nz);
            next_report.acoustic.continuity_carrier_requested_change_j +=
                requested_carrier_change(continuity_before_view, continuity_drho, dt_small_f, acoustic_only);

            MappedTerrainFastMomentumReport   fast_report;
            MappedTerrainFastMomentumDispatch fast_dispatch;
            fast_dispatch.rho                = candidate.rho.data();
            fast_dispatch.p                  = candidate.p.data();
            fast_dispatch.physical_height    = packed.height.data();
            fast_dispatch.reference_pressure = reference_pressure.data();
            fast_dispatch.sampled_reference_state = true;
            fast_dispatch.du_dt              = fast[0].data();
            fast_dispatch.dv_dt              = fast[1].data();
            fast_dispatch.dw_dt              = fast[2].data();
            fast_dispatch.nx                 = nx;
            fast_dispatch.ny                 = ny;
            fast_dispatch.nz                 = nz;
            fast_dispatch.dx                 = static_cast<float>(state.grid.dr);
            fast_dispatch.dy                 = static_cast<float>(state.grid.dtheta);
            fast_dispatch.nominal_dz         = static_cast<float>(metrics.dzeta);
            fast_dispatch.report             = &fast_report;
            if (!kernels->dispatch_mapped_fast_momentum(fast_dispatch) || !fast_report.complete)
                return false;
            ++next_report.fast_momentum_dispatches;
            std::array<std::vector<float>, 5> fast_update = {fast[0], fast[1], fast[2], disabled_field, disabled_field};
            CandidateFields                   fast_before = candidate;
            SimulationStateView               fast_before_view = candidate_view(state, fast_before);
            if (!update(MappedTerrainStateUpdateMode::FastMomentum, dt_small_f, fast_update))
                return false;
            const auto work = acoustic_only ? moist_pressure_work(fast_before_view, candidate, fast, dt_small_f)
                                            : pressure_work(fast_before_view, candidate, fast, dt_small_f);
            next_report.acoustic.pressure_gradient_requested_linear_work_j += work[0];
            next_report.acoustic.pressure_gradient_finite_step_correction_j += work[1];
            next_report.acoustic.pressure_gradient_committed_work_j += work[2];
            next_report.acoustic.pressure_gradient_storage_rounding_j += work[2] - work[0] - work[1];
            ++next_report.acoustic.substep_count;
            if (!acoustic_boundary(acoustic_only ? dt_small_f : 0.0f))
                return false;
        }
        next_report.acoustic_substeps = static_cast<std::uint32_t>(acoustic_substeps);
        const double weight_error     = std::abs(next_report.accumulated_time_weight - 1.0);
        if (weight_error > static_cast<double>(acoustic_substeps) * std::numeric_limits<float>::epsilon())
            return false;

        if (acoustic_only)
        {
            if (!finite_field(candidate.u) || !finite_field(candidate.v) || !finite_field(candidate.w) ||
                !positive_field(candidate.rho) || !positive_field(candidate.p))
                return false;
            auto& a = next_report.acoustic;
            a.continuity_carrier_storage_rounding_j =
                a.continuity_carrier_kinetic_change_j - a.continuity_carrier_requested_change_j;
            a.continuity_carrier_interior_conversion_j =
                a.continuity_carrier_requested_change_j + a.continuity_carrier_outward_boundary_transport_j;
            a.slow_momentum_advection_requested_linear_work_j =
                next_report.slow_momentum.advection_requested_linear_work_j;
            a.kinetic_transport_interior_conversion_j = a.continuity_carrier_requested_change_j +
                                                        a.slow_momentum_advection_requested_linear_work_j +
                                                        a.continuity_carrier_outward_boundary_transport_j;
            a.kinetic_transport_closure_residual_j    = 0;
            a.kinetic_transport_complete              = true;
            a.boundary_complete                       = true;
            a.committed_inventory_change_j =
                a.continuity_eos_dry_internal_change_j + a.continuity_eos_water_sensible_change_j +
                a.continuity_eos_phase_reference_change_j + a.continuity_carrier_kinetic_change_j +
                a.continuity_gravitational_change_j + a.continuity_subgrid_tke_change_j +
                a.pressure_gradient_committed_work_j + a.boundary_committed_work_j;
            a.complete = true;
            MappedAcousticResult result;
            result.acoustic                     = a;
            result.slow                         = next_report.slow_momentum;
            result.pbl                          = pbl_work;
            result.density_begin                = std::move(density_begin);
            result.mass_flux_x                  = field_from_vector(accumulated[0], nx, ny, nz);
            result.mass_flux_y                  = field_from_vector(accumulated[1], nx, ny, nz);
            result.mass_flux_z                  = field_from_vector(accumulated[2], nx, ny, nz);
            result.completed_backend_dispatches = backend->successful_dispatch_count() - dispatches_before;
            result.complete                     = true;
            const std::size_t bytes             = count * sizeof(float);
            for (const auto& value : {std::pair<float*, const Field3D*>(state.fields.u.data(), &candidate.u),
                                      {state.fields.v.data(), &candidate.v},
                                      {state.fields.w.data(), &candidate.w},
                                      {state.fields.rho.data(), &candidate.rho},
                                      {state.fields.p.data(), &candidate.p}})
                std::memcpy(value.first, value.second->data(), bytes);
            static_cast<BoundaryConditionWaveState&>(*state.boundary_condition_runtime) =
                std::move(static_cast<BoundaryConditionWaveState&>(wave_state));
            *acoustic_result = std::move(result);
            return true;
        }

        const auto transport_scalar = [&](Field3D&              scalar,
                                          bool                  positive_definite,
                                          std::size_t           report_index,
                                          ScalarMoistEnergyKind energy_kind) -> bool
        {
            Field3D                              output(nx, ny, nz, 0.0f);
            MappedTerrainScalarTransportReport   scalar_report;
            const bool                           record_thermodynamic = energy_kind != ScalarMoistEnergyKind::None;
            ScalarAttributionContext             attribution{&state,
                                                             &scalar,
                                                             &density_begin,
                                                             &candidate.rho,
                                                             &candidate.p,
                                                             &packed,
                                                             static_cast<float>(dt_large),
                                                             positive_definite,
                                                             energy_kind,
                                                             &next_report.thermodynamic_boundary[report_index],
                                                             &next_report.thermodynamic_moist_energy[report_index]};
            MappedTerrainScalarTransportDispatch d;
            d.scalar_begin      = scalar.data();
            d.density_begin     = density_begin.data();
            d.density_end       = candidate.rho.data();
            d.mass_flux_x       = accumulated[0].data();
            d.mass_flux_y       = accumulated[1].data();
            d.mass_flux_zeta    = accumulated[2].data();
            d.jacobian          = packed.jacobian.data();
            d.scalar_end        = output.data();
            d.nx                = nx;
            d.ny                = ny;
            d.nz                = nz;
            d.positive_definite = positive_definite;
            d.weno  = state.advection_runtime != nullptr && state.advection_runtime->config.scheme_id == "weno5";
            d.dx    = static_cast<float>(state.grid.dr);
            d.dy    = static_cast<float>(state.grid.dtheta);
            d.dzeta = static_cast<float>(metrics.dzeta);
            d.dt    = static_cast<float>(dt_large);
            d.density_floor              = density_min_kgm3;
            d.report                     = &scalar_report;
            d.completed_observer         = record_thermodynamic ? attribute_scalar_commit : nullptr;
            d.completed_observer_context = record_thermodynamic ? &attribution : nullptr;
            if (!kernels->dispatch_mapped_scalar_transport(d) || !scalar_report.complete)
                return false;
            scalar = std::move(output);
            next_report.scalar_flux_dispatches += scalar_report.scalar_flux_dispatch_count;
            next_report.scalar_limiter_dispatches += scalar_report.positive_definite_limiter_dispatch_count;
            next_report.scalar_update_dispatches += scalar_report.scalar_update_dispatch_count;
            return true;
        };
        if (!transport_scalar(
                candidate.theta, false, theta_boundary_report_index, ScalarMoistEnergyKind::PotentialTemperature) ||
            !transport_scalar(candidate.qv, true, qv_boundary_report_index, ScalarMoistEnergyKind::WaterVapor) ||
            !transport_scalar(candidate.tracer, true, qv_boundary_report_index, ScalarMoistEnergyKind::None) ||
            !transport_scalar(candidate.tke, true, qv_boundary_report_index, ScalarMoistEnergyKind::None))
            return false;

        for (std::size_t index = qc_boundary_report_index; index < thermodynamic_boundary_report_count; ++index)
        {
            next_report.thermodynamic_boundary[index].complete     = true;
            next_report.thermodynamic_moist_energy[index].complete = true;
        }

        SimulationStateView staged_view = candidate_view(state, candidate);
        next_report.final_eos           = dynamics_eos::apply_pressure_projection(staged_view);
        if (!next_report.final_eos.complete)
            return false;
        const MoistMixtureEnergyBudgetV2 before_boundary = compute_moist_mixture_energy_v2_deterministic(staged_view);
        if (!boundary())
            return false;
        const MoistMixtureEnergyBudgetV2 after_boundary  = compute_moist_mixture_energy_v2_deterministic(staged_view);
        next_report.final_boundary.dry_internal_change_j = after_boundary.dry_internal - before_boundary.dry_internal;
        next_report.final_boundary.water_sensible_change_j =
            after_boundary.water_sensible - before_boundary.water_sensible;
        next_report.final_boundary.phase_reference_change_j =
            after_boundary.phase_reference - before_boundary.phase_reference;
        next_report.final_boundary.carrier_kinetic_change_j =
            after_boundary.resolved_kinetic - before_boundary.resolved_kinetic;
        next_report.final_boundary.gravitational_change_j =
            after_boundary.gravitational - before_boundary.gravitational;
        next_report.final_boundary.subgrid_tke_change_j = after_boundary.subgrid_tke - before_boundary.subgrid_tke;
        next_report.final_boundary.constraint_committed_work_j = next_report.final_boundary.carrier_kinetic_change_j;
        next_report.final_boundary.committed_energy_change_j =
            next_report.final_boundary.dry_internal_change_j + next_report.final_boundary.water_sensible_change_j +
            next_report.final_boundary.phase_reference_change_j + next_report.final_boundary.carrier_kinetic_change_j +
            next_report.final_boundary.gravitational_change_j + next_report.final_boundary.subgrid_tke_change_j;
        next_report.final_boundary.closure_residual_j = 0.0;
        next_report.final_boundary.applied            = true;
        next_report.final_boundary.complete           = true;
        if (!valid_final_boundary_energy_report(next_report.final_boundary))
            return false;

        const PhysicalCellDomain domain                = physical_scalar_domain(state.grid);
        next_report.final_pressure_uses_canonical_fp64 = true;
        for (int i = domain.i_begin; i < domain.i_end; ++i)
            for (int j = domain.j_begin; j < domain.j_end; ++j)
                for (int k = domain.k_begin; k < domain.k_end; ++k)
                {
                    const float expected = clamp_pressure_pa(static_cast<float>(
                        dynamics_eos::pressure_fp64(candidate.rho[i][j][k], candidate.theta[i][j][k])));
                    next_report.final_pressure_uses_canonical_fp64 =
                        next_report.final_pressure_uses_canonical_fp64 && candidate.p[i][j][k] == expected;
                }
        if (!next_report.final_pressure_uses_canonical_fp64)
            return false;
        if (!finite_field(candidate.u) || !finite_field(candidate.v) || !finite_field(candidate.w) ||
            !positive_field(candidate.rho) || !positive_field(candidate.p) || !positive_field(candidate.theta) ||
            !nonnegative_field(candidate.qv) || !nonnegative_field(candidate.tracer) ||
            !nonnegative_field(candidate.tke))
            return false;
        next_report.acoustic.continuity_carrier_outward_boundary_transport_j = 0.0;
        next_report.acoustic.continuity_carrier_storage_rounding_j =
            next_report.acoustic.continuity_carrier_kinetic_change_j -
            next_report.acoustic.continuity_carrier_requested_change_j;
        next_report.acoustic.continuity_carrier_interior_conversion_j =
            next_report.acoustic.continuity_carrier_requested_change_j;
        next_report.acoustic.slow_momentum_advection_requested_linear_work_j =
            next_report.slow_momentum.advection_requested_linear_work_j;
        next_report.acoustic.kinetic_transport_interior_conversion_j =
            next_report.acoustic.continuity_carrier_requested_change_j +
            next_report.acoustic.slow_momentum_advection_requested_linear_work_j;
        next_report.acoustic.kinetic_transport_closure_residual_j = 0.0;
        next_report.acoustic.kinetic_transport_complete           = true;
        next_report.acoustic.boundary_complete                    = true;
        next_report.acoustic.committed_inventory_change_j =
            next_report.acoustic.continuity_eos_dry_internal_change_j +
            next_report.acoustic.continuity_eos_water_sensible_change_j +
            next_report.acoustic.continuity_eos_phase_reference_change_j +
            next_report.acoustic.continuity_carrier_kinetic_change_j +
            next_report.acoustic.continuity_gravitational_change_j +
            next_report.acoustic.continuity_subgrid_tke_change_j +
            next_report.acoustic.pressure_gradient_committed_work_j + next_report.acoustic.boundary_committed_work_j;
        next_report.acoustic.complete            = true;
        next_report.completed_backend_dispatches = backend->successful_dispatch_count() - dispatches_before;
        next_report.complete                     = true;
        Field3D retained_flux_x(nx, ny, nz, 0.0f);
        Field3D retained_flux_y(nx, ny, nz, 0.0f);
        Field3D retained_flux_z(nx, ny, nz, 0.0f);
        std::copy(accumulated[0].begin(), accumulated[0].end(), retained_flux_x.data());
        std::copy(accumulated[1].begin(), accumulated[1].end(), retained_flux_y.data());
        std::copy(accumulated[2].begin(), accumulated[2].end(), retained_flux_z.data());
        if (completed_observer != nullptr)
            completed_observer(completed_observer_context,
                               candidate.u,
                               candidate.v,
                               candidate.w,
                               candidate.rho,
                               candidate.p,
                               candidate.theta,
                               candidate.qv,
                               candidate.tracer,
                               candidate.tke,
                               density_begin,
                               retained_flux_x,
                               retained_flux_y,
                               retained_flux_z,
                               next_report);

        const std::size_t bytes = count * sizeof(float);
        for (const auto& publication : {std::pair<float*, const Field3D*>(state.fields.u.data(), &candidate.u),
                                        {state.fields.v.data(), &candidate.v},
                                        {state.fields.w.data(), &candidate.w},
                                        {state.fields.rho.data(), &candidate.rho},
                                        {state.fields.p.data(), &candidate.p},
                                        {state.fields.theta.data(), &candidate.theta},
                                        {state.fields.qv.data(), &candidate.qv},
                                        {state.fields.tracer.data(), &candidate.tracer},
                                        {state.fields.tke.data(), &candidate.tke}})
            std::memcpy(publication.first, publication.second->data(), bytes);
        report = next_report;
        return true;
    }
    catch (const std::bad_alloc&)
    {
        return false;
    }
    catch (const std::length_error&)
    {
        return false;
    }
}

bool prepare_mapped_momentum_tendencies(const SimulationStateView& state,
                                        const Field3D&             u,
                                        const Field3D&             v,
                                        const Field3D&             w,
                                        const Field3D&             rho,
                                        const Field3D&             pressure,
                                        bool                       pressure_only,
                                        std::array<Field3D, 8>&    result)
{
    if (!state.compute_runtime || state.compute_runtime->active_kind == ComputeBackendKind::Cpu)
        return false;
    const auto& compute = *state.compute_runtime;
    auto*       kernels = compute.backend ? compute.backend->terrain_kernels() : nullptr;
    if (compute.fallback_active || !kernels || !state.terrain_runtime || !state.domain_frame ||
        state.domain_frame->velocity_ms.u_ms != 0 || state.domain_frame->velocity_ms.v_ms != 0 ||
        state.coriolis_f != 0 || !kernels->supports_mapped_fast_momentum() ||
        !kernels->supports_mapped_slow_tendencies())
        throw std::runtime_error("native mapped momentum requires stationary geometry and its device owner");
    const int  nx = state.grid.nr, ny = state.grid.nth, nz = state.grid.nz;
    const auto count = rho.size();
    for (const auto* field : {&u, &v, &w, &rho, &pressure})
        if (field->size_r() != nx || field->size_th() != ny || field->size_z() != nz || !finite_field(*field))
            throw std::invalid_argument("native mapped momentum fields are incomplete");
    if (!positive_field(rho) || !positive_field(pressure) || state.base.p0.size() != std::size_t(nz) ||
        state.base.theta0.size() != std::size_t(nz) || state.base.qv0.size() != std::size_t(nz))
        throw std::invalid_argument("native mapped momentum reference profiles are incomplete");
    const auto&  metrics = state.terrain_runtime->metrics;
    MetricFields packed;
    if (!pack_metrics(metrics, nx, ny, nz, packed))
        throw std::invalid_argument("native mapped momentum geometry is invalid");
    std::array<Field3D, 8> candidate;
    for (auto& field : candidate)
        field.resize(nx, ny, nz, 0);
    auto base_pressure = canonical_reference_state(state, metrics);
    if (pressure_only)
    {
        MappedTerrainFastMomentumReport   report;
        MappedTerrainFastMomentumDispatch d;
        d.rho                     = rho.data();
        d.p                       = pressure.data();
        d.physical_height         = packed.height.data();
        d.reference_pressure      = base_pressure.data();
        d.du_dt                   = candidate[0].data();
        d.dv_dt                   = candidate[1].data();
        d.dw_dt                   = candidate[2].data();
        d.nx                      = nx;
        d.ny                      = ny;
        d.nz                      = nz;
        d.dx                      = float(state.grid.dr);
        d.dy                      = float(state.grid.dtheta);
        d.nominal_dz              = float(metrics.dzeta);
        d.report                  = &report;
        d.sampled_reference_state = true;
        if (!kernels->dispatch_mapped_fast_momentum(d) || !report.complete)
            throw std::runtime_error("native mapped pressure momentum dispatch failed");
    }
    else
    {
        for (const auto* field : {&state.fields.qv,
                                  &state.fields.qc,
                                  &state.fields.qr,
                                  &state.fields.qi,
                                  &state.fields.qs,
                                  &state.fields.qg,
                                  &state.fields.qh})
            if (field->size_r() != nx || field->size_th() != ny || field->size_z() != nz || !nonnegative_field(*field))
                throw std::invalid_argument("native mapped momentum moisture is incomplete");
        std::vector<float> base_theta(state.base.theta0.begin(), state.base.theta0.end()),
            base_vapor(state.base.qv0.begin(), state.base.qv0.end()), loading(count), ground(count);
        bool has_loading = false;
        for (std::size_t n = 0; n < count; ++n)
        {
            double total = 0;
            for (const auto* field : {&state.fields.qc,
                                      &state.fields.qr,
                                      &state.fields.qi,
                                      &state.fields.qs,
                                      &state.fields.qg,
                                      &state.fields.qh})
                total += field->data()[n];
            loading[n]  = float(total);
            has_loading = has_loading || total != 0;
        }
        for (int i = 1; i < nx; ++i)
            for (int j = 1; j < ny; ++j)
                ground[(std::size_t(i) * ny + j) * nz] =
                    float(terrain_transport::cgrid_ground_vertical_velocity(metrics, u, v, i, j));
        MappedTerrainSlowTendenciesReport   report;
        MappedTerrainSlowTendenciesDispatch d;
        d.u                           = u.data();
        d.v                           = v.data();
        d.w                           = w.data();
        d.rho                         = rho.data();
        d.p                           = pressure.data();
        d.physical_height             = packed.height.data();
        d.jacobian_x_positive_face    = packed.jacobian_x.data();
        d.jacobian_y_positive_face    = packed.jacobian_y.data();
        d.metric_x_positive_zeta_face = packed.metric_x.data();
        d.metric_y_positive_zeta_face = packed.metric_y.data();
        d.ground_physical_w           = ground.data();
        d.reference_pressure          = base_pressure.data();
        d.reference_theta             = base_theta.data();
        d.qv                          = state.fields.qv.data();
        d.reference_qv                = base_vapor.data();
        d.hydrometeor_loading         = loading.data();
        d.du_dt                       = candidate[0].data();
        d.dv_dt                       = candidate[1].data();
        d.dw_dt                       = candidate[2].data();
        d.drho_dt                     = candidate[3].data();
        d.dp_dt                       = candidate[4].data();
        d.du_advection_dt             = candidate[5].data();
        d.dv_advection_dt             = candidate[6].data();
        d.dw_advection_dt             = candidate[7].data();
        d.nx                          = nx;
        d.ny                          = ny;
        d.nz                          = nz;
        d.has_moisture                = true;
        d.has_loading                 = has_loading;
        d.dx                          = float(state.grid.dr);
        d.dy                          = float(state.grid.dtheta);
        d.dzeta                       = float(metrics.dzeta);
        d.nominal_dz                  = float(metrics.dzeta);
        d.gravity                     = float(physical_constants::gravity_ms2);
        d.report                      = &report;
        d.sampled_reference_state     = true;
        if (!kernels->dispatch_mapped_slow_tendencies(d) || !report.complete)
            throw std::runtime_error("native mapped slow momentum dispatch failed");
    }
    result = std::move(candidate);
    return true;
}
bool advance_dry_stationary_mapped_core(const SimulationStateView&  state,
                                        double                      dt,
                                        int                         substeps,
                                        MappedCoreStepReport&       report,
                                        MappedCoreCompletedObserver observer,
                                        void*                       context)
{
    return advance_mapped_core(state, dt, substeps, report, observer, context, nullptr);
}
bool advance_mapped_acoustic_core(const SimulationStateView& state,
                                  double                     dt,
                                  int                        substeps,
                                  MappedAcousticResult&      result)
{
    MappedCoreStepReport unused;
    return advance_mapped_core(state, dt, substeps, unused, nullptr, nullptr, &result);
}

} // namespace tmv::terrain_native
