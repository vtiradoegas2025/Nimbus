#include "dynamics/final_boundary_energy_report.hpp"
#include "dynamics/radiative_face.hpp"
#include "dynamics/native_cgrid_boundary.hpp"
#include "compute/dispatch/terrain.hpp"
#include "compute/dispatch/dynamics.hpp"
#include "compute/cgrid_boundary_contract.hpp"
#include "core/runtime/simulation_state.hpp"
#include "core/infra/physical_grid_metrics.hpp"
#include "core/infra/physical_cell_domain.hpp"
#include "dynamics/acoustic_energy_report.hpp"
#include "diagnostics/deterministic_reduction.hpp"
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
namespace tmv
{
namespace
{
std::array<Field3D*, 5> boundary_fields(const SimulationStateView& s)
{
    const int nx = s.grid.nr, ny = s.grid.nth, nz = s.grid.nz;
    if (!s.configuration || !s.domain_frame || !s.dynamics_runtime || !s.boundary_condition_runtime || nx < 3 ||
        ny < 3 || nz < 2 || !s.grid.geometry.staggered || s.grid.geometry.coord != CoordinateSystem::Cartesian ||
        s.configuration->stagger_type != StaggerType::CGrid ||
        s.dynamics_runtime->momentum_interface_flux_capture_enabled ||
        std::size_t(nx) > std::size_t(std::numeric_limits<int>::max()) / std::size_t(ny) / std::size_t(nz))
        throw std::invalid_argument("native C-grid boundary requires a complete unnested Cartesian C-grid");
    const std::size_t             count = std::size_t(nx) * ny * nz;
    const std::array<Field3D*, 5> fields{&s.fields.u, &s.fields.v, &s.fields.w, &s.fields.rho, &s.fields.p};
    for (const auto* field : {&s.fields.u, &s.fields.v, &s.fields.w, &s.fields.rho, &s.fields.p, &s.fields.theta})
        if (field->size_r() != nx || field->size_th() != ny || field->size_z() != nz || field->size() != count)
            throw std::invalid_argument("native C-grid boundary received malformed state storage");
    s.domain_frame->validate();
    return fields;
}
CartesianCGridAcousticEnergyReportV1 boundary_energy(const SimulationStateView&                  state,
                                                     const std::array<Field3D, 5>&               after,
                                                     const std::vector<float>&                   wave_before,
                                                     double                                      dt_s,
                                                     const CartesianCGridAcousticEnergyReportV1& original)
{
    if (original.boundary_application_count > 0 && !original.boundary_complete)
        throw std::invalid_argument("native boundary energy received an incomplete prior report");
    const auto& view = state;
    physical_grid_metrics::validate(view);
    const auto                                         domain = physical_scalar_domain(view.grid);
    const physical_grid_metrics::ControlVolumeMassView volume(view);
    const auto*                                        geometry = physical_grid_metrics::active_terrain_metrics(view);
    const std::array<const Field3D*, 7>                water{&state.fields.qv,
                                                             &state.fields.qc,
                                                             &state.fields.qr,
                                                             &state.fields.qi,
                                                             &state.fields.qs,
                                                             &state.fields.qg,
                                                             &state.fields.qh};
    for (auto* field : water)
        if (field->size_r() != state.grid.nr || field->size_th() != state.grid.nth || field->size_z() != state.grid.nz)
            throw std::invalid_argument("native boundary energy requires complete water storage");
    const auto old_velocity = [&](int i, int j, int k)
    {
        return std::array<double, 6>{state.fields.u(i - 1, j, k),
                                     state.fields.u(i, j, k),
                                     state.fields.v(i, j - 1, k),
                                     state.fields.v(i, j, k),
                                     k > 0 ? state.fields.w(i, j, k - 1) : 0,
                                     state.fields.w(i, j, k)};
    };
    const auto ground = [&](std::array<double, 6>& v, int i, int j, int k)
    {
        if (k == 0)
            v[4] = geometry ? -geometry->Gx_face(i, j, 0) *
                                      (.5 * (v[0] + v[1]) - physical_grid_metrics::surface_velocity_ms(state).u_ms) -
                                  geometry->Gy_face(i, j, 0) *
                                      (.5 * (v[2] + v[3]) - physical_grid_metrics::surface_velocity_ms(state).v_ms)
                            : 0;
    };
    using L = init::config::LateralBC;
    diagnostics::CompensatedSum actual, linear, finite, radiative;
    const auto yz = std::size_t(state.grid.nth) * state.grid.nz, xz = std::size_t(state.grid.nr) * state.grid.nz;
    for (int i = domain.i_begin; i < domain.i_end; ++i)
        for (int j = domain.j_begin; j < domain.j_end; ++j)
            for (int k = domain.k_begin; k < domain.k_end; ++k)
            {
                auto                  before = old_velocity(i, j, k), rad_before = before, rad_after = before;
                std::array<double, 6> next{after[0](i - 1, j, k),
                                           after[0](i, j, k),
                                           after[1](i, j - 1, k),
                                           after[1](i, j, k),
                                           k > 0 ? after[2](i, j, k - 1) : 0,
                                           after[2](i, j, k)};
                const auto            assign = [&](int component, std::size_t index)
                {
                    if (dt_s == 0)
                        return;
                    rad_before[component] = wave_before[index];
                    const float previous  = wave_before[index];
                    const bool  negative  = component == 0 || component == 2;
                    const bool  x_face    = component < 2;
                    const float neighbor  = component == 0   ? state.fields.u(1, j, k)
                                            : component == 1 ? state.fields.u(state.grid.nr - 3, j, k)
                                            : component == 2 ? state.fields.v(i, 1, k)
                                                             : state.fields.v(i, state.grid.nth - 3, k);
                    const float frame =
                        float(x_face ? state.domain_frame->velocity_ms.u_ms : state.domain_frame->velocity_ms.v_ms);
                    // Attribute the isolated radiative operator before ordered corner/constraint updates.
                    rad_after[component] = kw78_updated_face(previous,
                                                             previous - frame + (negative ? -30.0f : 30.0f),
                                                             negative ? neighbor - previous : previous - neighbor,
                                                             float(dt_s),
                                                             1.0f / float(x_face ? state.grid.dr : state.grid.dtheta),
                                                             negative);
                };
                if (i == domain.i_begin && state.configuration->lateral_bc_west == L::Radiative)
                    assign(0, std::size_t(j) * state.grid.nz + k);
                if (i == domain.i_end - 1 && state.configuration->lateral_bc_east == L::Radiative)
                    assign(1, yz + std::size_t(j) * state.grid.nz + k);
                if (j == domain.j_begin && state.configuration->lateral_bc_south == L::Radiative)
                    assign(2, 2 * yz + std::size_t(i) * state.grid.nz + k);
                if (j == domain.j_end - 1 && state.configuration->lateral_bc_north == L::Radiative)
                    assign(3, 2 * yz + xz + std::size_t(i) * state.grid.nz + k);
                ground(before, i, j, k);
                ground(next, i, j, k);
                ground(rad_before, i, j, k);
                ground(rad_after, i, j, k);
                if (before == next && rad_before == rad_after)
                    continue;
                double carrier = 1;
                for (auto* field : water)
                {
                    const double q = (*field)(i, j, k);
                    if (!std::isfinite(q) || q < 0)
                        throw std::invalid_argument("native boundary energy has invalid water content");
                    carrier += q;
                }
                const double mass = state.fields.rho(i, j, k) * volume.scalar_volume_m3(i, j, k) * carrier;
                if (!std::isfinite(mass) || mass <= 0)
                    throw std::invalid_argument("native boundary energy has invalid mass");
                for (std::size_t f = 0; f < 6; ++f)
                {
                    const double delta = rad_after[f] - rad_before[f];
                    linear.add(.5 * mass * rad_before[f] * delta);
                    finite.add(.25 * mass * delta * delta);
                    radiative.add(.25 * mass * (rad_after[f] * rad_after[f] - rad_before[f] * rad_before[f]));
                    actual.add(.25 * mass * (next[f] * next[f] - before[f] * before[f]));
                }
            }
    if (original.boundary_application_count == std::numeric_limits<std::uint64_t>::max())
        throw std::overflow_error("native boundary report counter overflowed");
    auto report = original;
    report.boundary_committed_work_j += actual.value();
    report.boundary_radiative_requested_linear_work_j += linear.value();
    report.boundary_radiative_finite_step_correction_j += finite.value();
    report.boundary_radiative_storage_rounding_j += radiative.value() - linear.value() - finite.value();
    report.boundary_radiative_committed_work_j += radiative.value();
    report.boundary_constraint_history_committed_work_j += actual.value() - radiative.value();
    report.committed_inventory_change_j += actual.value();
    ++report.boundary_application_count;
    report.boundary_complete = true;
    for (double value : {report.boundary_committed_work_j,
                         report.boundary_radiative_requested_linear_work_j,
                         report.boundary_radiative_finite_step_correction_j,
                         report.boundary_radiative_storage_rounding_j,
                         report.boundary_radiative_committed_work_j,
                         report.boundary_constraint_history_committed_work_j,
                         report.committed_inventory_change_j})
        if (!std::isfinite(value))
            throw std::overflow_error("native boundary energy report overflowed");
    return report;
}

} // namespace
void apply_native_cgrid_boundary(const SimulationStateView&   s,
                                 TerrainComputeKernels&       kernels,
                                 bool                         reseed,
                                 FinalBoundaryEnergyReportV1* energy)
{
    const auto fields = boundary_fields(s);
    const int  nx = s.grid.nr, ny = s.grid.nth, nz = s.grid.nz;
    const auto count = s.fields.rho.size();
    if (!kernels.supports_mapped_final_state())
        throw std::runtime_error("native C-grid boundary kernels are unavailable");
    std::array<Field3D, 5> candidates;
    for (auto& field : candidates)
        field.resize(nx, ny, nz, 0);
    MappedTerrainFinalStateReport   report;
    MappedTerrainFinalStateDispatch d;
    d.u         = fields[0]->data();
    d.v         = fields[1]->data();
    d.w         = fields[2]->data();
    d.rho       = fields[3]->data();
    d.p         = fields[4]->data();
    d.theta     = s.fields.theta.data();
    d.u_final   = candidates[0].data();
    d.v_final   = candidates[1].data();
    d.w_final   = candidates[2].data();
    d.rho_final = candidates[3].data();
    d.p_final   = candidates[4].data();
    d.nx        = nx;
    d.ny        = ny;
    d.nz        = nz;
    d.report    = &report;
    d.apply_eos = false;
    d.west      = int(s.configuration->lateral_bc_west);
    d.east      = int(s.configuration->lateral_bc_east);
    d.south     = int(s.configuration->lateral_bc_south);
    d.north     = int(s.configuration->lateral_bc_north);
    d.bottom    = int(s.configuration->bottom_bc);
    d.top       = int(s.configuration->top_bc);
    d.frame_u   = static_cast<float>(s.domain_frame->velocity_ms.u_ms);
    d.frame_v   = static_cast<float>(s.domain_frame->velocity_ms.v_ms);
    if (!kernels.dispatch_mapped_final_state(d) || !report.complete || report.eos_dispatch_count != 0 ||
        report.boundary_dispatch_count != 1)
        throw std::runtime_error("native C-grid boundary dispatch failed");
    for (std::size_t f = 0; f < candidates.size(); ++f)
        for (std::size_t n = 0; n < count; ++n)
            if (!std::isfinite(candidates[f].data()[n]) || (f >= 3 && candidates[f].data()[n] <= 0))
                throw std::domain_error("native C-grid boundary returned an invalid candidate");
    FinalBoundaryEnergyReportV1 next_energy;
    if (energy)
    {
        const auto acoustic                     = boundary_energy(s, candidates, {}, 0, {});
        next_energy.constraint_committed_work_j = acoustic.boundary_committed_work_j;
        next_energy.carrier_kinetic_change_j    = acoustic.boundary_committed_work_j;
        next_energy.committed_energy_change_j   = acoustic.boundary_committed_work_j;
        next_energy.applied                     = true;
        next_energy.complete                    = true;
    }
    BoundaryConditionWaveState wave;
    if (reseed)
    {
        wave.radiative_u_west.resize(std::size_t(ny) * nz);
        wave.radiative_u_east.resize(std::size_t(ny) * nz);
        wave.radiative_v_south.resize(std::size_t(nx) * nz);
        wave.radiative_v_north.resize(std::size_t(nx) * nz);
        for (int j = 0; j < ny; ++j)
            for (int k = 0; k < nz; ++k)
            {
                wave.radiative_u_west[std::size_t(j) * nz + k] = candidates[0](0, j, k);
                wave.radiative_u_east[std::size_t(j) * nz + k] = candidates[0](nx - 2, j, k);
            }
        for (int i = 0; i < nx; ++i)
            for (int k = 0; k < nz; ++k)
            {
                wave.radiative_v_south[std::size_t(i) * nz + k] = candidates[1](i, 0, k);
                wave.radiative_v_north[std::size_t(i) * nz + k] = candidates[1](i, ny - 2, k);
            }
    }
    for (std::size_t f = 0; f < fields.size(); ++f)
        *fields[f] = std::move(candidates[f]);
    if (energy)
        *energy = next_energy;
    if (reseed)
        static_cast<BoundaryConditionWaveState&>((*s.boundary_condition_runtime)) = std::move(wave);
}
void apply_native_cgrid_acoustic_boundary(const SimulationStateView&            s,
                                          DynamicsComputeKernels&               kernels,
                                          double                                dt_s,
                                          CartesianCGridAcousticEnergyReportV1* report)
{
    const auto fields = boundary_fields(s);
    const int  nx = s.grid.nr, ny = s.grid.nth, nz = s.grid.nz;
    if (!std::isfinite(dt_s) || dt_s <= 0 || !std::isfinite(s.grid.dr) || s.grid.dr <= 0 ||
        !std::isfinite(s.grid.dtheta) || s.grid.dtheta <= 0)
        throw std::invalid_argument("native acoustic boundary requires positive finite time and spacing");
    const auto         yz = std::size_t(ny) * nz, xz = std::size_t(nx) * nz;
    const auto&        old     = (*s.boundary_condition_runtime);
    const bool         history = old.radiative_u_west.size() == yz && old.radiative_u_east.size() == yz &&
                                 old.radiative_v_south.size() == xz && old.radiative_v_north.size() == xz;
    std::vector<float> wave(2 * (yz + xz)), next_wave(wave.size());
    for (int j = 0; j < ny; ++j)
        for (int k = 0; k < nz; ++k)
        {
            const auto n = std::size_t(j) * nz + k;
            wave[n]      = history ? old.radiative_u_west[n] : s.fields.u(0, j, k);
            wave[yz + n] = history ? old.radiative_u_east[n] : s.fields.u(nx - 2, j, k);
        }
    for (int i = 0; i < nx; ++i)
        for (int k = 0; k < nz; ++k)
        {
            const auto n          = std::size_t(i) * nz + k;
            wave[2 * yz + n]      = history ? old.radiative_v_south[n] : s.fields.v(i, 0, k);
            wave[2 * yz + xz + n] = history ? old.radiative_v_north[n] : s.fields.v(i, ny - 2, k);
        }
    CgridBoundaryDispatch d;
    auto&                 p = d.constants;
    p.nx                    = nx;
    p.ny                    = ny;
    p.nz                    = nz;
    p.bc_west               = int(s.configuration->lateral_bc_west);
    p.bc_east               = int(s.configuration->lateral_bc_east);
    p.bc_south              = int(s.configuration->lateral_bc_south);
    p.bc_north              = int(s.configuration->lateral_bc_north);
    p.bc_bottom             = int(s.configuration->bottom_bc);
    p.bc_top                = int(s.configuration->top_bc);
    p.dt_small              = float(dt_s);
    p.inv_dx                = 1.0f / float(s.grid.dr);
    p.inv_dy                = 1.0f / float(s.grid.dtheta);
    p.frame_velocity_u_ms   = float(s.domain_frame->velocity_ms.u_ms);
    p.frame_velocity_v_ms   = float(s.domain_frame->velocity_ms.v_ms);
    d.wave                  = wave.data();
    d.wave_output           = next_wave.data();
    std::array<Field3D, 5> candidate;
    for (std::size_t f = 0; f < 5; ++f)
    {
        candidate[f].resize(nx, ny, nz, 0);
        d.fields[f] = fields[f]->data();
        d.output[f] = candidate[f].data();
    }
    if (!kernels.dispatch_cgrid_boundary(d))
        throw std::runtime_error("native acoustic boundary dispatch failed");
    const std::array<const float*, 6> values{candidate[0].data(),
                                             candidate[1].data(),
                                             candidate[2].data(),
                                             candidate[3].data(),
                                             candidate[4].data(),
                                             next_wave.data()};
    if (!nimbus::compute::cgrid_boundary::finite_candidate(values, d))
        throw std::runtime_error("native acoustic boundary returned invalid fields");
    CartesianCGridAcousticEnergyReportV1 next_report;
    if (report)
        next_report = boundary_energy(s, candidate, wave, dt_s, *report);
    BoundaryConditionWaveState prepared;
    prepared.radiative_u_west.assign(next_wave.begin(), next_wave.begin() + yz);
    prepared.radiative_u_east.assign(next_wave.begin() + yz, next_wave.begin() + 2 * yz);
    prepared.radiative_v_south.assign(next_wave.begin() + 2 * yz, next_wave.begin() + 2 * yz + xz);
    prepared.radiative_v_north.assign(next_wave.begin() + 2 * yz + xz, next_wave.end());
    for (std::size_t f = 0; f < 5; ++f)
        *fields[f] = std::move(candidate[f]);
    static_cast<BoundaryConditionWaveState&>((*s.boundary_condition_runtime)) = std::move(prepared);
    if (report)
        *report = next_report;
}
void apply_native_cgrid_boundary(SimulationStorageState&      state,
                                 TerrainComputeKernels&       kernels,
                                 bool                         reseed,
                                 FinalBoundaryEnergyReportV1* report)
{
    const auto view = make_simulation_state_view(state);
    apply_native_cgrid_boundary(view, kernels, reseed, report);
}
void apply_native_cgrid_acoustic_boundary(SimulationStorageState&               state,
                                          DynamicsComputeKernels&               kernels,
                                          double                                dt_s,
                                          CartesianCGridAcousticEnergyReportV1* report)
{
    const auto view = make_simulation_state_view(state);
    apply_native_cgrid_acoustic_boundary(view, kernels, dt_s, report);
}
} // namespace tmv
