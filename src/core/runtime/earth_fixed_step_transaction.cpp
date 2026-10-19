#include "core/runtime/earth_fixed_step_transaction.hpp"

#include "boundary_conditions/factory.hpp"
#include "compute/bridge/residency.hpp"
#include "core/infra/physical_grid_metrics.hpp"
#include "core/runtime/simulation_state.hpp"
#include "core/runtime/earth_fixed_remap.hpp"
#include "diagnostics/conservation_budget.hpp"
#include "diagnostics/energy_budget.hpp"
#include "diagnostics/momentum_budget.hpp"
#include "diagnostics/moisture_budget.hpp"
#include "microphysics/runtime.hpp"
#include "microphysics/grid_geometry.hpp"
#include "numerics/runtime.hpp"
#include "radar/runtime.hpp"

#include <cmath>
#include <stdexcept>
#include <type_traits>
#include <fstream>
#include <iomanip>

namespace tmv
{
namespace
{
std::ofstream remap_csv;
}

struct EarthFixedStepTransaction::Reports
{
    ConservationBudget         mass_before, mass_after;
    MomentumBudget             momentum_before, momentum_after;
    MoistureBudget             water_before, water_after;
    EnergyDiagnosticSnapshotV1 energy_before, energy_after;
    EarthFixedRemapReport      remap;
};

void open_earth_fixed_remap_csv(const std::string& path)
{
    close_earth_fixed_remap_csv();
    remap_csv.open(path, std::ios::trunc);
    if (!remap_csv)
        throw std::runtime_error("cannot open Earth-fixed remapping ledger");
    remap_csv << "time_s,dt_s,substeps,dry_mass_before_kg,dry_mass_after_kg,outward_dry_mass_kg,"
                 "dry_mass_storage_kg,energy_change_j,outward_energy_j,energy_storage_j,"
                 "water_before_kg,water_after_kg,outward_water_kg,water_storage_kg,"
                 "momentum_x_before,momentum_x_after,momentum_x_outward,momentum_x_storage,"
                 "momentum_y_before,momentum_y_after,momentum_y_outward,momentum_y_storage,"
                 "momentum_z_before,momentum_z_after,momentum_z_outward,momentum_z_storage\n";
    remap_csv << std::setprecision(17);
}

void close_earth_fixed_remap_csv()
{
    if (remap_csv.is_open())
        remap_csv.close();
}

EarthFixedStepTransaction::EarthFixedStepTransaction(SimulationStorageState& live, double dt_s)
    : live_(live), begin_time_s_(live.time_s), end_time_s_(live.time_s + dt_s)
{
    if (!std::isfinite(begin_time_s_) || !std::isfinite(dt_s) || dt_s <= 0.0 || !std::isfinite(end_time_s_) ||
        end_time_s_ <= begin_time_s_ || live.runtime.compute.fallback_active || live.configuration.nesting.enabled ||
        !live.runtime.terrain.earth_fixed_placement.active ||
        live.runtime.terrain.config.horizontal_placement_id != "earth_fixed")
        throw std::invalid_argument("Earth-fixed step staging requires a valid unnested interval without fallback");
    candidate_ = std::make_unique<SimulationStorageState>(live);
    frame_     = live.configuration.domain_frame;
    moving_    = frame_.velocity_ms.u_ms != 0.0 || frame_.velocity_ms.v_ms != 0.0;
    if (moving_)
    {
        if (!live.grid.geometry.staggered || live.configuration.storm_tracking.enabled ||
            live.runtime.turbulence.config.scheme_id != "none" ||
            (live.runtime.boundary_layer.config.scheme_id != "none" &&
             live.runtime.boundary_layer.config.scheme_id != "slab") ||
            (live.runtime.radiation.config.scheme_id != "none" &&
             (live.runtime.radiation.config.scheme_id != "simple_grey" || live.runtime.radiation.config.do_sw)))
            throw std::invalid_argument(
                "continuous Earth-fixed execution requires the maintained column-physics profile");
        candidate_->configuration.domain_frame.earth_origin_at_t0_m = frame_.earth_origin_at(begin_time_s_);
        candidate_->configuration.domain_frame.velocity_ms          = {};
        auto view                                                   = state();
        refresh_grid_metrics_from_terrain(view);
        auto& chaos = candidate_->runtime.chaos;
        if (chaos.scheme && (chaos.scheme->name() == "full_stochastic" || chaos.scheme->name() == "boundary_layer"))
        {
            const auto  origin = frame_.earth_origin_at(begin_time_s_);
            std::string error;
            chaos.scheme->configure_compute_backend(live_.runtime.compute.backend.get(),
                                                    chaos.config.native_generation_required);
            if (!chaos.scheme->set_continuous_horizontal_earth_origin(
                    origin.x_m, origin.y_m, chaos.config, candidate_->process_grid_metrics, error))
                throw std::runtime_error("Earth-fixed stochastic origin update failed: " + error);
        }
    }
}

EarthFixedStepTransaction::~EarthFixedStepTransaction()
{
    // Failed candidates may have registered their storage with the shared
    // backend. No address from discarded storage may survive this boundary.
    invalidate_all_resident_fields_backend(live_.runtime.compute);
}

SimulationStateView EarthFixedStepTransaction::state()
{
    if (published_ || prepared_)
        throw std::logic_error("sealed Earth-fixed step has no mutable candidate state");
    auto view               = make_simulation_state_view(*candidate_);
    view.compute_runtime    = &live_.runtime.compute;
    view.defer_step_reports = moving_;
    return view;
}

void EarthFixedStepTransaction::finish_evolution()
{
    if (evolved_ || prepared_ || published_)
        throw std::logic_error("Earth-fixed evolution was already finished");
    if (moving_)
    {
        reports_        = std::make_unique<Reports>();
        const auto live = make_simulation_state_view(live_);
        if (step_diagnostics_enabled())
        {
            reports_->mass_before     = compute_conservation_budget(live);
            reports_->momentum_before = compute_momentum_budget(live);
            reports_->water_before    = compute_moisture_budget(live);
            reports_->energy_before   = compute_energy_diagnostic_snapshot_v1(live);
        }
        reports_->remap =
            live_.runtime.compute.active_kind == ComputeBackendKind::Cpu
                ? remap_earth_fixed_window(*candidate_, frame_, begin_time_s_, end_time_s_)
                : remap_earth_fixed_window(*candidate_, frame_, begin_time_s_, end_time_s_, live_.runtime.compute);
        auto view = state();
        refresh_grid_metrics_from_terrain(view);
        if (step_radar_diagnostics_enabled())
            calculate_radar_observables(view);
        compute_dynamics_diagnostics(view,
                                     step_vorticity_diagnostics_enabled(),
                                     step_angular_momentum_diagnostics_enabled(),
                                     step_pressure_diagnostics_enabled());
        auto&       time = candidate_->runtime.time_stepping;
        const auto& mass = reports_->remap.dry_mass;
        time.last_boundary_mass.outward_boundary_mass_kg += mass.boundary_outflow;
        time.last_boundary_mass.flux_implied_state_change_kg -= mass.boundary_outflow;
        time.last_boundary_mass.committed_state_change_kg += mass.final_inventory - mass.initial_inventory;
        time.last_boundary_mass.closure_residual_kg =
            time.last_boundary_mass.committed_state_change_kg - time.last_boundary_mass.flux_implied_state_change_kg;
        time.cumulative_outward_boundary_mass_kg += mass.boundary_outflow;
        time.cumulative_flux_implied_state_change_kg -= mass.boundary_outflow;
        time.cumulative_committed_state_change_kg += mass.final_inventory - mass.initial_inventory;
        time.cumulative_boundary_mass_closure_residual_kg =
            time.cumulative_committed_state_change_kg - time.cumulative_flux_implied_state_change_kg;
        if (step_diagnostics_enabled())
        {
            accumulate_completed_step_energy(time.last_completed_step_energy, reports_->remap.energy);
            accumulate_completed_step_energy(time.cumulative_completed_step_energy, reports_->remap.energy);
            reports_->mass_after     = compute_conservation_budget(view);
            reports_->momentum_after = compute_momentum_budget(view);
            reports_->water_after    = compute_moisture_budget(view);
            reports_->energy_after   = compute_energy_diagnostic_snapshot_v1(view);
        }
    }
    evolved_ = true;
}

void EarthFixedStepTransaction::prepare_commit()
{
    if (published_ || prepared_ || live_.time_s != begin_time_s_)
        throw std::logic_error("Earth-fixed step publication is stale or repeated");
    if (!evolved_)
        finish_evolution();
    auto& candidate = *candidate_;
    if (candidate.time_s != begin_time_s_ && candidate.time_s != end_time_s_)
        throw std::invalid_argument("Earth-fixed step candidate has an unexpected clock");
    const auto  origin    = candidate.configuration.domain_frame.earth_origin_at(end_time_s_);
    const auto& placement = candidate.runtime.terrain.earth_fixed_placement;
    if (!placement.active || placement.grid_origin_earth_m.x_m != origin.x_m ||
        placement.grid_origin_earth_m.y_m != origin.y_m)
        throw std::invalid_argument("Earth-fixed step fields and geometry have different publication times");
    if (candidate.grid.nr != live_.grid.nr || candidate.grid.nth != live_.grid.nth ||
        candidate.grid.nz != live_.grid.nz || candidate.grid.dx != live_.grid.dx ||
        candidate.grid.dy != live_.grid.dy || candidate.grid.dz != live_.grid.dz)
        throw std::invalid_argument("Earth-fixed timestep cannot change its storage partition");
    physical_grid_metrics::validate(state());
    auto& transport = candidate.runtime.dynamics.transport_context;
    if (transport.active() && !transport.is_complete())
        throw std::invalid_argument("Earth-fixed step has an unfinished mass-flux history");

    const auto destination = make_simulation_state_view(live_);
    auto       boundary    = create_boundary_condition_scheme(destination);
    if (candidate.runtime.dynamics.scheme)
        candidate.runtime.dynamics.scheme->bind_execution_context(destination);
    if (candidate.runtime.microphysics.scheme)
    {
        candidate.runtime.microphysics.scheme->bind_execution_owners(live_.runtime.compute, &live_.runtime.parity);
        bind_microphysics_velocity(make_simulation_state_view(candidate), *candidate.runtime.microphysics.scheme);
        candidate.runtime.microphysics.scheme->rebind_vertical_velocity_field(live_.fields.w);
    }
    candidate.runtime.boundary_conditions.execution_operator = std::move(boundary);
    transport                                                = {};
    if (candidate.process_grid_metrics.terrain_metrics_active ||
        candidate.process_grid_metrics.terrain_metrics != nullptr)
    {
        candidate.process_grid_metrics.terrain_metrics    = &live_.runtime.terrain.metrics;
        candidate.process_grid_metrics.terrain_topography = &live_.runtime.terrain.topography;
    }
    candidate.time_s = end_time_s_;
    prepared_        = true;
}

bool EarthFixedStepTransaction::publish() noexcept
{
    static_assert(std::is_nothrow_move_assignable_v<SimulationStorageState>);
    if (!prepared_ || published_ || live_.time_s != begin_time_s_)
        return false;
    candidate_->runtime.compute = std::move(live_.runtime.compute);
    live_                       = std::move(*candidate_);
    published_                  = true;
    return true;
}

void EarthFixedStepTransaction::publish_reports()
{
    if (!published_)
        throw std::logic_error("Earth-fixed reports require published state");
    if (!moving_)
        return;
    auto                live = make_simulation_state_view(live_);
    SimulationStateView report_state{
        live.grid, live.fields, live.base, live.tendencies, live.coriolis_f, begin_time_s_, live.surface};
    live_.runtime.bind(report_state);
    report_state.configuration        = &live_.configuration;
    report_state.domain_frame         = &live_.configuration.domain_frame;
    report_state.diagnostics          = &live_.diagnostics;
    report_state.process_grid_metrics = &live_.process_grid_metrics;
    const double dt                   = end_time_s_ - begin_time_s_;
    write_thermodynamic_source_budget_row(report_state, begin_time_s_, dt);
    if (step_diagnostics_enabled())
    {
        report_budget_transition(report_state, "total_step", reports_->mass_before, reports_->mass_after, dt);
        report_momentum_transition(report_state, "total_step", reports_->momentum_before, reports_->momentum_after, dt);
        report_moisture_transition(report_state, "total_step", reports_->water_before, reports_->water_after, dt);
        report_energy_transition(report_state, "total_step", reports_->energy_before, reports_->energy_after, dt);
    }
    if (remap_csv.is_open())
    {
        const auto&               r = reports_->remap;
        MovingMeshInventoryUpdate water;
        for (const auto& species : r.water)
        {
            water.initial_inventory += species.initial_inventory;
            water.final_inventory += species.final_inventory;
            water.boundary_outflow += species.boundary_outflow;
            water.storage_rounding += species.storage_rounding;
        }
        remap_csv << begin_time_s_ << ',' << dt << ',' << r.substeps << ',' << r.dry_mass.initial_inventory << ','
                  << r.dry_mass.final_inventory << ',' << r.dry_mass.boundary_outflow << ','
                  << r.dry_mass.storage_rounding << ',' << r.energy.coordinate_remap_committed_change_j << ','
                  << r.outward_energy_j << ',' << r.energy_storage_rounding_j << ',' << water.initial_inventory << ','
                  << water.final_inventory << ',' << water.boundary_outflow << ',' << water.storage_rounding;
        for (std::size_t n = 0; n < 3; ++n)
            remap_csv << ',' << r.initial_momentum[n] << ',' << r.final_momentum[n] << ',' << r.outward_momentum[n]
                      << ',' << r.momentum_storage_rounding[n];
        remap_csv << '\n';
        if (!remap_csv)
            throw std::runtime_error("Earth-fixed remapping ledger write failed");
    }
}

} // namespace tmv
