#include "core/runtime/earth_fixed_remap.hpp"
#include "core/runtime/held_source_remap.hpp"
#include "core/runtime/environment_column_refresh.hpp"
#include "core/runtime/native_held_sources.hpp"
#include "dynamics/native_cgrid_boundary.hpp"
#include "dynamics/native_moving_cgrid_transport.hpp"
#include "terrain/native_conservative_remap.hpp"

#include "boundary_conditions/runtime.hpp"
#include "boundary_layer/runtime.hpp"
#include "boundary_layer/column_evaluation.hpp"
#include "core/field/field_bounds.hpp"
#include "core/infra/physical_constants.hpp"
#include "core/infra/physical_grid_metrics.hpp"
#include "core/runtime/simulation_state.hpp"
#include "diagnostics/energy_budget.hpp"
#include "diagnostics/momentum_budget.hpp"
#include "dynamics/moving_energy.hpp"
#include "dynamics/moving_environment.hpp"
#include "dynamics/moving_cgrid_transport.hpp"
#include "dynamics/moving_boundary.hpp"
#include "microphysics/runtime.hpp"
#include "radiation/column_evaluation.hpp"
#include "chaos/runtime.hpp"
#include "numerics/runtime.hpp"
#include "terrain/conservative_transport.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace tmv
{
namespace
{
constexpr double rd = physical_constants::gas_constant_dry_air_jkgk;
constexpr double cp = physical_constants::specific_heat_cp_jkgk;
double           volume(const TerrainMetrics3D& g, int i, int j, int k)
{
    return g.dx * g.dy * g.dzeta * g.J(i, j, k);
}

MovingMeshInventoryUpdate remap_scalar(const Field3D&                 before,
                                       Field3D&                       after,
                                       const Field3D&                 rho0,
                                       const Field3D&                 rho1,
                                       const TerrainMetrics3D&        g0,
                                       const TerrainMetrics3D&        g1,
                                       const TerrainMeshSweptVolumes& sweep,
                                       bool                           volume_concentration,
                                       TerrainComputeKernels*         kernels = nullptr)
{
    if (before.empty())
        return {};
    if (kernels)
        return terrain_native::remap_inventory(
            *kernels, before, after, rho0, rho1, g0, g1, sweep, !volume_concentration);

    if (before.size_r() != rho0.size_r() || before.size_th() != rho0.size_th() || before.size_z() != rho0.size_z())
        throw std::invalid_argument("coordinate remapping received a malformed transported scalar");
    Field3D     result  = before;
    long double initial = 0.0L, final = 0.0L, boundary = 0.0L, rounding = 0.0L;
    const auto  currency = [&](int i, int j, int k)
    { return static_cast<double>(before(i, j, k)) * (volume_concentration ? 1.0 : rho0(i, j, k)); };
    for (int i = 1; i < before.size_r() - 1; ++i)
        for (int j = 1; j < before.size_th() - 1; ++j)
            for (int k = 0; k < before.size_z(); ++k)
            {
                long double inventory = currency(i, j, k) * volume(g0, i, j, k);
                initial += inventory;
                for (int side : {-1, 1})
                {
                    const double      fx = -sweep.x(i + (side > 0), j, k);
                    const int         ix = i + (side > 0 ? (fx < 0.0) : -(fx > 0.0));
                    const long double x  = side * static_cast<long double>(fx) * currency(ix, j, k);
                    inventory -= x;
                    if ((side < 0 && i == 1) || (side > 0 && i == before.size_r() - 2))
                        boundary += x;
                    const double      fy = -sweep.y(i, j + (side > 0), k);
                    const int         jy = j + (side > 0 ? (fy < 0.0) : -(fy > 0.0));
                    const long double y  = side * static_cast<long double>(fy) * currency(i, jy, k);
                    inventory -= y;
                    if ((side < 0 && j == 1) || (side > 0 && j == before.size_th() - 2))
                        boundary += y;
                }
                const double denominator = volume(g1, i, j, k) * (volume_concentration ? 1.0 : rho1(i, j, k));
                result(i, j, k)          = static_cast<float>(inventory / denominator);
                const long double stored = static_cast<long double>(result(i, j, k)) * denominator;
                final += stored;
                rounding += stored - inventory;
                if (!std::isfinite(result(i, j, k)))
                    throw std::overflow_error("coordinate remapping overflowed a transported scalar");
            }
    after = std::move(result);
    return {static_cast<double>(initial),
            static_cast<double>(final),
            static_cast<double>(boundary),
            static_cast<double>(rounding)};
}

double minimum_global_jacobian(const TerrainConfig& config, int nz)
{
    const double height = config.scheme_id == "bell" ? config.bell.h0 : config.schar.H;
    const auto   weight = [&](int k)
    {
        const double s = static_cast<double>(k) / nz;
        return (1.0 - s) * (config.coord_id == "smoothed" ? std::exp(-config.smoothing_decay * s) : 1.0);
    };
    double minimum = 1.0;
    for (int k = 0; k < nz; ++k)
        minimum = std::min(minimum, 1.0 + height * (weight(k + 1) - weight(k)) / (config.ztop / nz));
    if (!std::isfinite(minimum) || minimum <= 0.0)
        throw std::invalid_argument("continuous Earth-fixed motion requires globally unfolded terrain");
    return minimum;
}

} // namespace

static EarthFixedRemapReport remap_window(SimulationStorageState&              s,
                                          const domain::HorizontalDomainFrame& frame,
                                          double                               begin_time_s,
                                          double                               end_time_s,
                                          ComputeBackendRuntimeState*          compute)
{
    TerrainComputeKernels* kernels = nullptr;
    if (compute)
    {
        if (!compute->backend || compute->active_kind == ComputeBackendKind::Cpu || compute->fallback_active)
            throw std::invalid_argument("native coordinate remapping requires an active accelerator without fallback");
        kernels = compute->backend->terrain_kernels();
        if (!kernels || !kernels->supports_earth_geometry() || !kernels->supports_conservative_remap() ||
            !kernels->supports_moving_momentum() || !kernels->supports_moving_energy() ||
            !kernels->supports_reference_atmosphere() || !kernels->supports_mapped_final_state() ||
            !kernels->supports_column_remap() || !kernels->supports_held_sources())
            throw std::runtime_error("native coordinate remapping requires its complete kernel set");
    }
    EarthFixedRemapReport report;
    const double          dt = end_time_s - begin_time_s;
    const int             nx = s.grid.nr, ny = s.grid.nth, nz = s.grid.nz;
    if (!std::isfinite(begin_time_s) || !std::isfinite(end_time_s) || !std::isfinite(dt) || dt <= 0.0 ||
        s.time_s != begin_time_s || nx < 4 || ny < 4 || nz < 2 || !s.grid.geometry.staggered ||
        (!compute && s.runtime.compute.active_kind != ComputeBackendKind::Cpu) || s.runtime.compute.fallback_active ||
        !s.runtime.terrain.earth_fixed_placement.active ||
        s.runtime.terrain.config.horizontal_placement_id != "earth_fixed" ||
        s.runtime.terrain.earth_fixed_placement.sampling != EarthFixedTerrainSampling::CellAverageBilinear)
        throw std::invalid_argument("coordinate remapping requires a cell-average C-grid interval");
    frame.validate();
    terrain_transport::validate_geometry(s.runtime.terrain.metrics);
    const double norm =
        dt * (std::abs(frame.velocity_ms.u_ms) / s.grid.dx + std::abs(frame.velocity_ms.v_ms) / s.grid.dy);
    const double count = std::ceil(norm / (0.4 * minimum_global_jacobian(s.runtime.terrain.config, nz)));
    if (!std::isfinite(count) || count > 1000000.0)
        throw std::invalid_argument("coordinate remapping cannot make bounded stable progress");
    const unsigned substeps      = static_cast<unsigned>(std::max(1.0, count));
    s.configuration.domain_frame = frame;
    auto view                    = make_simulation_state_view(s);
    if (compute)
        view.compute_runtime = compute;
    const auto energy_before = compute_moist_mixture_energy_v2_deterministic(view);
    const auto domain        = physical_scalar_domain(view.grid);
    const auto index         = [&](int i, int j, int k) { return (static_cast<std::size_t>(i) * ny + j) * nz + k; };
    double     mesh_time     = begin_time_s;
    for (unsigned substep = 0; substep < substeps; ++substep)
    {
        const double next_time =
            substep + 1 == substeps ? end_time_s : begin_time_s + dt * (static_cast<double>(substep) + 1.0) / substeps;
        if (!(next_time > mesh_time))
            throw std::invalid_argument("coordinate remapping clock cannot advance");
        if (kernels)
            fill_moving_environment(s.fields, s.base, s.runtime.terrain.metrics, *kernels);
        else
            fill_moving_environment(s.fields, s.base, s.runtime.terrain.metrics);
        auto extended_fluxes =
            compute ? extend_environment_columns(s, *compute->backend) : extend_environment_columns(s);
        auto motion = kernels
                          ? prepare_earth_fixed_terrain_motion(s.runtime.terrain, frame, mesh_time, next_time, *kernels)
                          : prepare_earth_fixed_terrain_motion(s.runtime.terrain, frame, mesh_time, next_time);
        const auto&                g0              = s.runtime.terrain.metrics;
        const auto&                g1              = motion.terrain.geometry.metrics;
        const auto&                sweep           = motion.swept_volumes;
        const auto                 momentum_before = compute_physical_momentum_budget_v2(view);
        TerrainContravariantFluxes mass_flux;
        mass_flux.x.resize(nx + 1, ny, nz, 0.0);
        mass_flux.y.resize(nx, ny + 1, nz, 0.0);
        mass_flux.z.resize(nx, ny, nz + 1, 0.0);
        const auto old_energy =
            kernels ? moving_energy_density(s.fields, g0, *kernels) : moving_energy_density(s.fields, g0);
        std::vector<double> target_energy(old_energy.size());
        if (kernels)
            mass_flux = terrain_native::moving_mesh_mass_flux(*kernels, s.fields.rho, sweep);
        else
        {
            for (int i = 1; i < nx; ++i)
                for (int j = 1; j < ny - 1; ++j)
                    for (int k = 0; k < nz; ++k)
                        mass_flux.x(i, j, k) =
                            -sweep.x(i, j, k) * s.fields.rho(sweep.x(i, j, k) >= 0.0 ? i : i - 1, j, k);
            for (int i = 1; i < nx - 1; ++i)
                for (int j = 1; j < ny; ++j)
                    for (int k = 0; k < nz; ++k)
                        mass_flux.y(i, j, k) =
                            -sweep.y(i, j, k) * s.fields.rho(i, sweep.y(i, j, k) >= 0.0 ? j : j - 1, k);
        }
        Field3D    density;
        const auto mass = kernels ? terrain_native::remap_inventory(*kernels,
                                                                    s.fields.rho,
                                                                    density,
                                                                    s.fields.rho,
                                                                    s.fields.rho,
                                                                    g0,
                                                                    g1,
                                                                    sweep,
                                                                    false,
                                                                    terrain_native::RemapLowerBound::Positive)
                                  : update_moving_mesh_inventory(g0, g1, mass_flux, domain, s.fields.rho, density, 0.0);
        if (substep == 0)
            report.dry_mass.initial_inventory = mass.initial_inventory;
        report.dry_mass.final_inventory = mass.final_inventory;
        report.dry_mass.boundary_outflow += mass.boundary_outflow;
        report.dry_mass.storage_rounding += mass.storage_rounding;
        auto momentum = kernels ? prepare_moving_cgrid_momentum_transport(view, g1, density, mass_flux, *kernels, true)
                                : prepare_moving_cgrid_momentum_transport(view, g1, density, mass_flux, true);
        const auto ground_targets =
            kernels ? prepare_moving_ground_targets(g0, s.fields.rho, s.fields.u, s.fields.v, mass_flux, *kernels)
                    : prepare_moving_ground_targets(g0, s.fields.rho, s.fields.u, s.fields.v, mass_flux);
        if (kernels)
        {
            const auto energy_transport =
                terrain_native::remap_inventory(*kernels, old_energy, target_energy, g0, g1, sweep);
            report.outward_energy_j += energy_transport.boundary_outflow;
        }
        else
        {
            for (int i = 1; i < nx - 1; ++i)
                for (int j = 1; j < ny - 1; ++j)
                    for (int k = 0; k < nz; ++k)
                    {
                        long double inventory = old_energy[index(i, j, k)] * volume(g0, i, j, k);
                        for (int side : {-1, 1})
                        {
                            const double fx = -sweep.x(i + (side > 0), j, k);
                            const int    ix = i + (side > 0 ? (fx < 0.0) : -(fx > 0.0));
                            const double ex = side * fx * old_energy[index(ix, j, k)];
                            const double fy = -sweep.y(i, j + (side > 0), k);
                            const int    jy = j + (side > 0 ? (fy < 0.0) : -(fy > 0.0));
                            const double ey = side * fy * old_energy[index(i, jy, k)];
                            inventory -= static_cast<long double>(ex) + ey;
                            if ((side < 0 && i == 1) || (side > 0 && i == nx - 2))
                                report.outward_energy_j += ex;
                            if ((side < 0 && j == 1) || (side > 0 && j == ny - 2))
                                report.outward_energy_j += ey;
                        }
                        target_energy[index(i, j, k)] = static_cast<double>(inventory / volume(g1, i, j, k));
                    }
        }
        const Field3D                 old_density = s.fields.rho;
        const std::array<Field3D*, 7> water_fields{
            &s.fields.qv, &s.fields.qc, &s.fields.qr, &s.fields.qi, &s.fields.qs, &s.fields.qg, &s.fields.qh};
        for (std::size_t n = 0; n < water_fields.size(); ++n)
        {
            auto&      field  = *water_fields[n];
            const auto update = remap_scalar(field, field, old_density, density, g0, g1, sweep, false, kernels);
            if (substep == 0)
                report.water[n].initial_inventory = update.initial_inventory;
            report.water[n].final_inventory = update.final_inventory;
            report.water[n].boundary_outflow += update.boundary_outflow;
            report.water[n].storage_rounding += update.storage_rounding;
        }
        for (auto* field : {&s.fields.tke, &s.fields.tracer, &s.runtime.microphysics.theta_storage_residual})
            remap_scalar(*field, *field, old_density, density, g0, g1, sweep, false, kernels);
        if (s.runtime.microphysics.scheme)
            for (auto descriptor : s.runtime.microphysics.scheme->prognostic_scalars_for_transport())
            {
                if (descriptor.field == nullptr)
                    throw std::invalid_argument("coordinate remapping received a missing private moment");
                remap_scalar(*descriptor.field,
                             *descriptor.field,
                             old_density,
                             density,
                             g0,
                             g1,
                             sweep,
                             descriptor.storage == MicrophysicsScalarStorage::VolumeConcentration,
                             kernels);
            }
        s.fields.rho = std::move(density);
        s.fields.u   = std::move(momentum.u);
        s.fields.v   = std::move(momentum.v);
        s.fields.w   = std::move(momentum.w);
        if (!commit_earth_fixed_terrain(motion.terrain, s.runtime.terrain))
            throw std::logic_error("coordinate remapping geometry became stale");
        const auto& geometry = s.runtime.terrain.metrics;
        if (kernels)
        {
            project_moving_top_momentum(geometry, s.fields.rho, s.fields.u, s.fields.v, *kernels);
            apply_native_cgrid_boundary(s, *kernels);
            correct_moving_ground_momentum(
                geometry, s.fields.rho, s.fields.u, s.fields.v, s.fields.w, ground_targets, *kernels);
            reconstruct_moving_energy(s.fields, geometry, target_energy, *kernels);
            fill_moving_environment(s.fields, s.base, geometry, *kernels);
            apply_native_cgrid_boundary(s, *kernels, true);
        }
        else
        {
            auto& boundary = boundary_scheme_for(view);
            project_moving_top_momentum(s.runtime.terrain.metrics, s.fields.rho, s.fields.u, s.fields.v);
            boundary->apply_full();
            correct_moving_ground_momentum(geometry, s.fields.rho, s.fields.u, s.fields.v, s.fields.w, ground_targets);
            reconstruct_moving_energy(s.fields, geometry, target_energy);
            fill_moving_environment(s.fields, s.base, s.runtime.terrain.metrics);
            boundary->apply_full();
            boundary->seed_acoustic_radiation();
        }
        const auto                  momentum_after = compute_physical_momentum_budget_v2(view);
        const std::array<double, 3> p0{
            momentum_before.momentum_x, momentum_before.momentum_y, momentum_before.momentum_z};
        const std::array<double, 3> p1{momentum_after.momentum_x, momentum_after.momentum_y, momentum_after.momentum_z};
        for (std::size_t component = 0; component < 3; ++component)
        {
            const double outward =
                momentum.ledger[component].outward_impulse + (component == 2 ? ground_targets.outward_impulse : 0.0);
            const double storage = p1[component] - p0[component] + outward;
            const double scale   = (mass.initial_inventory + mass.final_inventory) *
                                   (component == 2 ? wind_vertical_abs_max_ms : wind_horizontal_abs_max_ms);
            if (!std::isfinite(storage) || std::abs(storage) > 8.0 * std::numeric_limits<float>::epsilon() * scale)
                throw std::runtime_error("coordinate remapping has an unowned momentum change");
            if (substep == 0)
                report.initial_momentum[component] = p0[component];
            report.final_momentum[component] = p1[component];
            report.outward_momentum[component] += outward;
            report.momentum_storage_rounding[component] += storage;
        }
        const double local_dt = next_time - mesh_time;
        if (kernels)
            remap_held_owners(s,
                              frame.velocity_ms.u_ms * local_dt / s.grid.dx,
                              frame.velocity_ms.v_ms * local_dt / s.grid.dy,
                              extended_fluxes,
                              *kernels);
        else
            remap_held_owners(s,
                              frame.velocity_ms.u_ms * local_dt / s.grid.dx,
                              frame.velocity_ms.v_ms * local_dt / s.grid.dy,
                              extended_fluxes);
        s.time_s = next_time;
        refresh_grid_metrics_from_terrain(view);
        if (s.runtime.chaos.scheme &&
            (s.runtime.chaos.scheme->name() == "full_stochastic" || s.runtime.chaos.scheme->name() == "boundary_layer"))
        {
            if (compute)
                s.runtime.chaos.scheme->configure_compute_backend(compute->backend.get(), true);
            const auto  origin = frame.earth_origin_at(next_time);
            std::string error;
            if (!s.runtime.chaos.scheme->set_continuous_horizontal_earth_origin(
                    origin.x_m, origin.y_m, s.runtime.chaos.config, s.process_grid_metrics, error))
                throw std::runtime_error("remapped stochastic origin could not be reconstructed: " + error);
            const bool updated                         = s.runtime.boundary_layer.updated_this_step;
            s.runtime.boundary_layer.updated_this_step = false;
            chaos::apply_chaos_tendencies(view);
            s.runtime.boundary_layer.updated_this_step = updated;
        }
        if (kernels)
            reconstruct_native_held_sources(view, *kernels);
        else
        {
            prepare_boundary_layer_tendencies_for_application(view);
            auto& radiation = s.runtime.radiation;
            if (!radiation.held_cell_power_w.empty())
                for (int i = 0; i < nx; ++i)
                    for (int j = 0; j < ny; ++j)
                        for (int k = 0; k < nz; ++k)
                        {
                            const double exner =
                                std::pow(s.fields.p(i, j, k) / physical_constants::reference_pressure_pa, rd / cp);
                            s.tendencies.dtheta_dt_rad(i, j, k) =
                                static_cast<float>(radiation.held_cell_power_w(i, j, k) /
                                                   (s.fields.rho(i, j, k) * cp * exner * volume(geometry, i, j, k)));
                        }
        }
        mesh_time = next_time;
        s.time_s  = mesh_time;
    }
    const auto energy_after                    = compute_moist_mixture_energy_v2_deterministic(view);
    auto&      energy                          = report.energy;
    energy.dry_internal_change_j               = energy_after.dry_internal - energy_before.dry_internal;
    energy.water_sensible_change_j             = energy_after.water_sensible - energy_before.water_sensible;
    energy.phase_reference_change_j            = energy_after.phase_reference - energy_before.phase_reference;
    energy.carrier_kinetic_change_j            = energy_after.resolved_kinetic - energy_before.resolved_kinetic;
    energy.gravitational_change_j              = energy_after.gravitational - energy_before.gravitational;
    energy.subgrid_tke_change_j                = energy_after.subgrid_tke - energy_before.subgrid_tke;
    energy.coordinate_remap_committed_change_j = completed_step_reservoir_sum(energy);
    energy.committed_owner_sum_j               = completed_step_owner_sum(energy);
    energy.closure_residual_j                  = energy.committed_owner_sum_j - completed_step_reservoir_sum(energy);
    energy.complete                            = true;
    report.energy_storage_rounding_j           = energy.coordinate_remap_committed_change_j + report.outward_energy_j;
    report.substeps                            = substeps;
    invalidate_microphysics_transients_after_domain_recycling(view);
    fields::visit(s.diagnostics,
                  fields::diagnostics,
                  [](const char*, Field3D& field)
                  {
                      if (!field.empty())
                          field.fill(0.0f);
                  });
    return report;
}
EarthFixedRemapReport remap_earth_fixed_window(SimulationStorageState&              state,
                                               const domain::HorizontalDomainFrame& frame,
                                               double                               begin,
                                               double                               end)
{
    return remap_window(state, frame, begin, end, nullptr);
}
EarthFixedRemapReport remap_earth_fixed_window(SimulationStorageState&              state,
                                               const domain::HorizontalDomainFrame& frame,
                                               double                               begin,
                                               double                               end,
                                               ComputeBackendRuntimeState&          compute)
{
    return remap_window(state, frame, begin, end, &compute);
}

} // namespace tmv
