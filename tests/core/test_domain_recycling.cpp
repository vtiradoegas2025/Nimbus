#include "catch2/catch.hpp"
#include "core/infra/domain_recycling.hpp"
#include "core/infra/domain_recycling_state.hpp"
#include "core/infra/physical_cell_domain.hpp"
#include "core/infra/physical_constants.hpp"
#include "core/infra/rayleigh_damping.hpp"
#include "core/runtime/runtime_config.hpp"
#include "diagnostics/deterministic_reduction.hpp"
#include "radiation/column_evaluation.hpp"
#include "radiation/schemes/simple_grey/simple_grey.hpp"
#include "boundary_layer/column_evaluation.hpp"
#include "boundary_layer/schemes/slab/slab.hpp"
#include "boundary_layer/schemes/ysu/ysu.hpp"
#include "boundary_layer/schemes/mynn/mynn.hpp"
#include "microphysics/runtime.hpp"
#include "microphysics/schemes/milbrandt/milbrandt.hpp"
#include "microphysics/schemes/thompson/thompson.hpp"
#include "chaos/schemes/full_stochastic/full_stochastic.hpp"
#include "init/hodograph/wind_profile.hpp"
#include "terrain/conservative_transport.hpp"

#include <array>
#include <cmath>
#include <limits>
#include <string>
#include <utility>
#include <vector>

void compute_wind_profile(const WindProfile&, double, double& u, double& v)
{
    u = 0.0;
    v = 0.0;
}

namespace
{

class D5DynamicsScheme final : public tmv::DynamicsScheme
{
public:
    explicit D5DynamicsScheme(
        bool supports_invalidation = true,
        std::string scheme_name =
            "klemp_wilhelmson_split_cartesian_cgrid",
        std::string coordinate = "cartesian_cgrid")
        : supports_invalidation_(supports_invalidation),
          scheme_name_(std::move(scheme_name)),
          coordinate_(std::move(coordinate))
    {
    }

    std::unique_ptr<tmv::DynamicsScheme> clone() const override
    {
        return std::make_unique<D5DynamicsScheme>(*this);
    }
    bool supports_domain_recycling_invalidation() const noexcept override
    {
        return supports_invalidation_;
    }
    void invalidate_generation_state_after_domain_recycling() noexcept override
    {
        invalidated = true;
    }
    void compute_momentum_tendencies(
        const Field3D&, const Field3D&, const Field3D&, const Field3D&,
        const Field3D&, const Field3D&, double, Field3D&, Field3D&,
        Field3D&, Field3D&, Field3D&) override
    {
    }
    std::string get_scheme_name() const override { return scheme_name_; }
    std::string get_coordinate_system() const override { return coordinate_; }
    int get_num_prognostic_vars() const override { return 5; }

    bool invalidated = false;

private:
    bool supports_invalidation_;
    std::string scheme_name_;
    std::string coordinate_;
};

tmv::domain::StormWindowMovementDecision seed_d5_storm_tracking(
    SimulationStorageState& storage)
{
    auto& continuity = storage.runtime.storm_tracking;
    continuity.status = tmv::diagnostics::StormContinuityStatus::Tracking;
    continuity.reason = tmv::diagnostics::StormContinuityReason::None;
    continuity.sample_count = 2;
    continuity.samples[0] = {10.0, 3000.0, 2000.0, 0.90};
    continuity.samples[1] = {20.0, 3200.0, 1900.0, 0.92};
    continuity.branch_id = 7;
    continuity.topology =
        tmv::diagnostics::StormBranchTopology::Single;
    continuity.selected_candidate_available = true;
    continuity.selected_candidate_rank = 0;
    continuity.selected_component_label = 11;
    continuity.motion_available = true;
    continuity.velocity_x_ms = 10.0;
    continuity.velocity_y_ms = -5.0;
    continuity.speed_ms = std::hypot(10.0, -5.0);
    continuity.velocity_uncertainty_ms = 0.5;
    continuity.continuity_confidence = 0.92;
    continuity.innovation_distance_m = 4.0;
    continuity.association_gate_radius_m = 1000.0;
    continuity.association_envelope_available = true;
    continuity.association_envelope_west_m = 2800.0;
    continuity.association_envelope_east_m = 3600.0;
    continuity.association_envelope_south_m = 1500.0;
    continuity.association_envelope_north_m = 2300.0;

    auto& movement = storage.runtime.storm_window_movement;
    movement.armed = true;
    movement.pending_i_direction = 1;
    movement.pending_j_direction = -1;
    movement.consecutive_compatible_samples = 2;
    movement.sample_time_available = true;
    movement.last_sample_time_s = 20.0;

    const auto state = make_simulation_state_view(storage);
    const PhysicalCellDomain physical = physical_scalar_domain(state.grid);
    const std::size_t nx = static_cast<std::size_t>(physical.nx());
    const std::size_t ny = static_cast<std::size_t>(physical.ny());
    storage.runtime.storm_selected_partition_available = true;
    storage.runtime.storm_selected_partition_nx = nx;
    storage.runtime.storm_selected_partition_ny = ny;
    storage.runtime.storm_selected_partition.assign(nx * ny, 0U);
    storage.runtime.storm_selected_partition[3 * ny + 1] = 1U;
    storage.runtime.storm_selected_partition[4 * ny + 2] = 1U;

    tmv::domain::StormWindowMovementDecision decision;
    decision.status = tmv::domain::StormWindowMovementStatus::Requested;
    decision.reason = tmv::domain::StormWindowMovementReason::RequestReady;
    decision.sample_time_s = 20.0;
    decision.requested_i_shift_cells = 2;
    decision.requested_j_shift_cells = -1;
    decision.confirmation_count = 2;
    return decision;
}

class D5TurbulenceScheme final : public TurbulenceSchemeBase
{
public:
    std::unique_ptr<TurbulenceSchemeBase> clone() const override
    {
        return std::make_unique<D5TurbulenceScheme>(*this);
    }
    std::string name() const override { return "d5_stub"; }
    int required_fields() const override { return 0; }
    void initialize(const TurbulenceConfig&) override {}
    void compute(const TurbulenceConfig&, const GridMetrics&,
                 const TurbulenceStateView&, TurbulenceTendencies&,
                 TurbulenceDiagnostics*) override {}
};

class D5ChaosScheme final : public tmv::chaos::ChaosScheme
{
public:
    explicit D5ChaosScheme(std::string scheme_name)
        : scheme_name_(std::move(scheme_name))
    {
    }
    std::unique_ptr<tmv::chaos::ChaosScheme> clone() const override
    {
        return std::make_unique<D5ChaosScheme>(*this);
    }
    std::string name() const override { return scheme_name_; }
    void initialize(const tmv::chaos::ChaosConfig&, const GridMetrics&) override
    {
    }

private:
    std::string scheme_name_;
};

class D5AdvectionScheme final : public tmv::AdvectionSchemeBase
{
public:
    std::unique_ptr<tmv::AdvectionSchemeBase> clone() const override
    {
        return std::make_unique<D5AdvectionScheme>(*this);
    }
    std::string name() const override { return "d5_stub"; }
    void initialize() override {}
    void initialize(const tmv::AdvectionConfig&) override {}
    void compute_flux_divergence(
        const tmv::AdvectionConfig&, const tmv::AdvectionStateView&,
        tmv::AdvectionTendencies&,
        tmv::AdvectionDiagnostics*) override {}
    double suggest_dt(
        const tmv::AdvectionConfig&,
        const tmv::AdvectionStateView&) override
    {
        return 1.0;
    }
};

class D5DiffusionScheme final : public tmv::DiffusionSchemeBase
{
public:
    std::unique_ptr<tmv::DiffusionSchemeBase> clone() const override
    {
        return std::make_unique<D5DiffusionScheme>(*this);
    }
    std::string name() const override { return "d5_stub"; }
    void initialize() override {}
    void initialize(const tmv::DiffusionConfig&) override {}
    void compute_diffusion_tendencies(
        const tmv::DiffusionConfig&, const tmv::DiffusionStateView&,
        tmv::DiffusionTendencies&,
        tmv::DiffusionDiagnostics*) override {}
    double check_stability(
        const tmv::DiffusionConfig&,
        const tmv::DiffusionStateView&) override
    {
        return 0.0;
    }
};

class D5TimeSteppingScheme final : public tmv::TimeSteppingSchemeBase
{
public:
    explicit D5TimeSteppingScheme(bool recycling_supported = true)
        : recycling_supported_(recycling_supported)
    {
    }
    std::unique_ptr<tmv::TimeSteppingSchemeBase> clone() const override
    {
        return std::make_unique<D5TimeSteppingScheme>(*this);
    }
    std::string name() const override { return "d5_stub"; }
    void initialize() override {}
    void initialize(
        const tmv::TimeSteppingConfig&, tmv::RHSFunction) override {}
    void step(
        const tmv::TimeSteppingConfig&, tmv::TimeSteppingState&,
        tmv::TimeSteppingDiagnostics*) override {}
    double suggest_dt(
        const tmv::TimeSteppingConfig&,
        const tmv::TimeSteppingState&) override
    {
        return 1.0;
    }
    bool supports_domain_recycling_invalidation() const noexcept override
    {
        return recycling_supported_;
    }
    void invalidate_generation_state_after_domain_recycling() noexcept override
    {
        generation_invalidated = true;
    }

    bool generation_invalidated = false;

private:
    bool recycling_supported_ = true;
};

class D5RadarScheme final : public tmv::RadarSchemeBase
{
public:
    explicit D5RadarScheme(
        std::string scheme_name = "d5_radar",
        bool recycling_supported = true)
        : scheme_name_(std::move(scheme_name)),
          recycling_supported_(recycling_supported)
    {
    }
    std::unique_ptr<tmv::RadarSchemeBase> clone() const override
    {
        return std::make_unique<D5RadarScheme>(*this);
    }
    void initialize(const tmv::RadarConfig&, int, int, int) override {}
    void compute(
        const tmv::RadarConfig&, const tmv::RadarStateView&,
        tmv::RadarOut&) override {}
    std::string name() const override { return scheme_name_; }
    bool supports_domain_recycling_invalidation() const noexcept override
    {
        return recycling_supported_;
    }
    void invalidate_generation_state_after_domain_recycling() noexcept override
    {
        generation_invalidated = true;
    }

    bool generation_invalidated = false;

private:
    std::string scheme_name_;
    bool recycling_supported_ = true;
};

void require_diffusion_report_equal(
    const tmv::DiffusionMomentumWorkReport& actual,
    const tmv::DiffusionMomentumWorkReport& expected)
{
    REQUIRE(actual.requested_linear_work_j ==
            expected.requested_linear_work_j);
    REQUIRE(actual.finite_step_correction_j ==
            expected.finite_step_correction_j);
    REQUIRE(actual.storage_rounding_j == expected.storage_rounding_j);
    REQUIRE(actual.committed_work_j == expected.committed_work_j);
    REQUIRE(actual.moist_dry_internal_change_j ==
            expected.moist_dry_internal_change_j);
    REQUIRE(actual.moist_water_sensible_change_j ==
            expected.moist_water_sensible_change_j);
    REQUIRE(actual.moist_phase_reference_change_j ==
            expected.moist_phase_reference_change_j);
    REQUIRE(actual.moist_carrier_kinetic_change_j ==
            expected.moist_carrier_kinetic_change_j);
    REQUIRE(actual.moist_gravitational_change_j ==
            expected.moist_gravitational_change_j);
    REQUIRE(actual.moist_subgrid_tke_change_j ==
            expected.moist_subgrid_tke_change_j);
    REQUIRE(actual.moist_committed_energy_change_j ==
            expected.moist_committed_energy_change_j);
    REQUIRE(actual.moist_energy_closure_residual_j ==
            expected.moist_energy_closure_residual_j);
    REQUIRE(actual.moist_energy_complete == expected.moist_energy_complete);
}

Field3D indexed_field(int nr, int nth, int nz, float offset = 0.0f)
{
    Field3D field(nr, nth, nz);
    for (int i = 0; i < nr; ++i)
        for (int j = 0; j < nth; ++j)
            for (int k = 0; k < nz; ++k)
                field(i, j, k) = offset + static_cast<float>(100 * i + 10 * j + k);
    return field;
}

void require_shifted(const Field3D& before,
                     const Field3D& after,
                     const tmv::domain::WholeCellShiftPlan& plan,
                     const std::vector<float>& profile)
{
    for (int i = 0; i < plan.nr; ++i)
        for (int j = 0; j < plan.nth; ++j)
            for (int k = 0; k < plan.nz; ++k)
            {
                CAPTURE(i, j, k);
                const bool retained = i >= plan.destination_i_begin && i < plan.destination_i_end &&
                                      j >= plan.destination_j_begin && j < plan.destination_j_end;
                if (retained)
                    REQUIRE(after(i, j, k) == before(i + plan.shift_i_cells, j + plan.shift_j_cells, k));
                else
                    REQUIRE(after(i, j, k) == profile[static_cast<std::size_t>(k)]);
            }
}

void require_shifted_from_field(
    const Field3D& before,
    const Field3D& after,
    const tmv::domain::WholeCellShiftPlan& plan,
    const Field3D& leading)
{
    for (int i = 0; i < plan.nr; ++i)
        for (int j = 0; j < plan.nth; ++j)
            for (int k = 0; k < plan.nz; ++k)
            {
                const bool retained =
                    i >= plan.destination_i_begin
                    && i < plan.destination_i_end
                    && j >= plan.destination_j_begin
                    && j < plan.destination_j_end;
                REQUIRE(after(i, j, k) == (retained
                    ? before(i + plan.shift_i_cells,
                             j + plan.shift_j_cells, k)
                    : leading(i, j, k)));
            }
}

void require_exported(const Field3D& before,
                      const std::vector<float>& trailing,
                      const tmv::domain::WholeCellShiftPlan& plan)
{
    std::size_t offset = 0;
    for (int i = plan.physical_i_begin; i < plan.physical_i_end; ++i)
        for (int j = plan.physical_j_begin; j < plan.physical_j_end; ++j)
        {
            const bool retained =
                i >= plan.source_i_begin && i < plan.source_i_end &&
                j >= plan.source_j_begin && j < plan.source_j_end;
            if (retained) continue;
            for (int k = 0; k < plan.nz; ++k)
                REQUIRE(trailing[offset++] == before(i, j, k));
        }
    REQUIRE(offset == trailing.size());
}

void require_surface_shifted(const std::vector<float>& before,
                             const std::vector<float>& after,
                             const tmv::domain::WholeCellShiftPlan& plan,
                             float leading_value)
{
    for (int i = 0; i < plan.nr; ++i)
        for (int j = 0; j < plan.nth; ++j)
        {
            const bool retained = i >= plan.destination_i_begin && i < plan.destination_i_end &&
                                  j >= plan.destination_j_begin && j < plan.destination_j_end;
            const std::size_t destination = static_cast<std::size_t>(i * plan.nth + j);
            if (retained)
            {
                const std::size_t source = static_cast<std::size_t>(
                    (i + plan.shift_i_cells) * plan.nth + j + plan.shift_j_cells);
                REQUIRE(after[destination] == before[source]);
            }
            else
                REQUIRE(after[destination] == leading_value);
        }
}

void require_surface_exported(
    const std::vector<float>& before,
    const std::vector<float>& trailing,
    const tmv::domain::WholeCellShiftPlan& plan)
{
    std::size_t offset = 0;
    for (int i = plan.physical_i_begin; i < plan.physical_i_end; ++i)
        for (int j = plan.physical_j_begin; j < plan.physical_j_end; ++j)
        {
            const bool retained =
                i >= plan.source_i_begin && i < plan.source_i_end
                && j >= plan.source_j_begin && j < plan.source_j_end;
            if (retained) continue;
            REQUIRE(trailing[offset++] == before[
                static_cast<std::size_t>(i * plan.nth + j)]);
        }
    REQUIRE(offset == trailing.size());
}

SimulationStorageState radiation_recycling_storage(
    double sample_time_s,
    bool longwave,
    bool shortwave)
{
    SimulationStorageState storage(
        GridStorageState{6, 7, 4, 100.0, 250.0, 0.1, 200.0});
    storage.grid.geometry.initialize(
        6, 7, 4, 100.0, 250.0, 200.0,
        CoordinateSystem::Cartesian, StaggerType::CGrid);
    storage.base.p0 = {98500.0, 95500.0, 92500.0, 89500.0};
    storage.base.theta0 = {298.0, 300.0, 302.0, 304.0};
    storage.base.qv0 = {0.012, 0.010, 0.008, 0.006};
    storage.base.u0 = {8.0, 9.0, 10.0, 11.0};
    storage.base.v0 = {-2.0, -1.0, 0.0, 1.0};
    storage.base.rho0.resize(4);
    constexpr double kappa =
        physical_constants::gas_constant_dry_air_jkgk /
        physical_constants::specific_heat_cp_jkgk;
    for (std::size_t k = 0; k < 4; ++k)
    {
        const double temperature = storage.base.theta0[k] * std::pow(
            storage.base.p0[k] /
                physical_constants::reference_pressure_pa,
            kappa);
        storage.base.rho0[k] = storage.base.p0[k] /
            (physical_constants::gas_constant_dry_air_jkgk * temperature);
    }
    storage.tendencies.dtheta_dt_rad =
        indexed_field(6, 7, 4, -1000.0f);
    auto& radiation = storage.runtime.radiation;
    radiation.config.scheme_id = "simple_grey";
    radiation.config.do_lw = longwave;
    radiation.config.do_sw = shortwave;
    radiation.config.dt_radiation = 300.0;
    radiation.config.local_solar_time_at_start_h = 0.0;
    radiation.scheme = std::make_unique<SimpleGreyScheme>();
    radiation.scheme->initialize(radiation.config);
    radiation.last_update_time_s = sample_time_s;
    radiation.next_update_time_s = sample_time_s + 300.0;
    radiation.held_cell_power_w = indexed_field(6, 7, 4, 10.0f);
    storage.runtime.boundary_layer.surface_config.Tsfc = 295.0;
    storage.runtime.boundary_layer.surface_config.albedo = 0.18;
    storage.runtime.boundary_layer.surface_config.emissivity = 0.97;

    const auto state = make_simulation_state_view(storage);
    const auto physical = physical_scalar_domain(state.grid);
    double aggregate = 0.0;
    for (int i = physical.i_begin; i < physical.i_end; ++i)
        for (int j = physical.j_begin; j < physical.j_end; ++j)
            for (int k = physical.k_begin; k < physical.k_end; ++k)
                aggregate += static_cast<double>(
                    radiation.held_cell_power_w[i][j][k]);
    radiation.held_atmospheric_power_w = aggregate;
    radiation.last_expected_atmospheric_energy_j = 11.0;
    radiation.last_applied_radiation_heat_j = 12.0;
    radiation.accumulated_expected_atmospheric_energy_j = 13.0;
    radiation.accumulated_applied_radiation_heat_j = 14.0;
    radiation.applied_step_count = 15;
    return storage;
}

SimulationStorageState pbl_recycling_storage(
    const std::string& scheme_name = "slab")
{
    SimulationStorageState storage(
        GridStorageState{6, 7, 4, 100.0, 250.0, 0.1, 200.0});
    storage.grid.geometry.initialize(
        6, 7, 4, 100.0, 250.0, 200.0,
        CoordinateSystem::Cartesian, StaggerType::CGrid);
    storage.base.p0 = {98500.0, 95500.0, 92500.0, 89500.0};
    storage.base.theta0 = {298.0, 300.0, 303.0, 307.0};
    storage.base.qv0 = {0.012, 0.010, 0.008, 0.006};
    storage.base.u0 = {6.0, 8.0, 10.0, 12.0};
    storage.base.v0 = {-2.0, -1.0, 0.0, 1.0};
    storage.base.rho0.resize(4);
    constexpr double kappa =
        physical_constants::gas_constant_dry_air_jkgk /
        physical_constants::specific_heat_cp_jkgk;
    for (std::size_t k = 0; k < 4; ++k)
    {
        const double temperature = storage.base.theta0[k] * std::pow(
            storage.base.p0[k] /
                physical_constants::reference_pressure_pa,
            kappa);
        storage.base.rho0[k] = storage.base.p0[k] /
            (physical_constants::gas_constant_dry_air_jkgk * temperature);
    }
    auto& runtime = storage.runtime.boundary_layer;
    runtime.config.scheme_id = scheme_name;
    runtime.config.dt_pbl = 60.0;
    runtime.config.enable_tke = scheme_name == "mynn";
    runtime.surface_config.Tsfc = 295.0;
    runtime.surface_config.qsfc = 0.011;
    if (scheme_name == "slab")
        runtime.scheme = std::make_unique<SlabScheme>();
    else if (scheme_name == "ysu")
        runtime.scheme = std::make_unique<YSUScheme>();
    else if (scheme_name == "mynn")
        runtime.scheme = std::make_unique<MYNNScheme>();
    else
        throw std::invalid_argument("unsupported PBL test scheme");
    runtime.scheme->initialize(runtime.config);
    runtime.last_update_time_s = 600.0;
    runtime.next_update_time_s = 660.0;
    runtime.updated_this_step = true;
    runtime.skin_nr = 6;
    runtime.skin_nth = 7;
    runtime.skin_captured = true;
    runtime.skin_theta_2d.resize(42);
    runtime.skin_qv_2d.resize(42);
    runtime.slab_nr = 6;
    runtime.slab_nth = 7;
    runtime.slab_columns.resize(42);
    for (std::size_t index = 0; index < 42; ++index)
    {
        runtime.skin_theta_2d[index] = 290.0f + static_cast<float>(index);
        runtime.skin_qv_2d[index] = 0.001f * static_cast<float>(index);
        runtime.slab_columns[index] = {
            100.0 + static_cast<double>(index),
            295.0 + static_cast<double>(index),
            0.001 * static_cast<double>(index), true};
    }
    runtime.held_u_force_n = indexed_field(6, 7, 4, 1000.0f);
    runtime.held_v_force_n = indexed_field(6, 7, 4, 2000.0f);
    runtime.held_theta_inventory_rate_kgk_s =
        indexed_field(6, 7, 4, 3000.0f);
    runtime.held_vapor_mass_rate_kg_s =
        indexed_field(6, 7, 4, 4000.0f);
    storage.tendencies.dtke_dt_pbl = indexed_field(6, 7, 4, 5000.0f);
    std::vector<float>* surfaces[] = {
        &storage.surface.ustar, &storage.surface.tau_u,
        &storage.surface.tau_v, &storage.surface.sensible_heat_flux,
        &storage.surface.latent_heat_flux, &storage.surface.moisture_flux,
        &storage.surface.pbl_height};
    for (std::size_t field = 0; field < 7; ++field)
    {
        surfaces[field]->resize(42);
        for (std::size_t index = 0; index < 42; ++index)
            (*surfaces[field])[index] = static_cast<float>(
                1000 * field + index);
    }
    runtime.last_momentum_work.committed_work_j = 9.0;
    runtime.cumulative_committed_momentum_work_j = 10.0;
    runtime.momentum_work_step_count = 11;
    return storage;
}

SimulationStorageState separately_evolving_nested_recycling_storage()
{
    SimulationStorageState storage(
        GridStorageState{14, 14, 3, 1000.0, 500.0, 1.0, 1000.0});
    storage.grid.geometry.initialize(
        14, 14, 3, 1000.0, 1000.0, 500.0,
        CoordinateSystem::Cartesian, StaggerType::CGrid);
    storage.time_s = 1.0;
    auto& nested = storage.nested_execution;
    nested.relationship.version = 1;
    nested.relationship.parent_cells_x = 12;
    nested.relationship.parent_cells_y = 12;
    nested.relationship.levels_z = 3;
    nested.relationship.parent_ratio_x = 2;
    nested.relationship.parent_ratio_y = 2;
    nested.relationship.parent_time_step_ratio = 2;
    nested.relationship.child_origin_parent_cell_i = 4;
    nested.relationship.child_origin_parent_cell_j = 4;
    nested.relationship.child_cells_x = 4;
    nested.relationship.child_cells_y = 4;
    nested.relationship.child_parent_span_x = 2;
    nested.relationship.child_parent_span_y = 2;
    nested.relationship.relaxation_width_child_cells = 1;
    nested.relationship.boundary_stencil_parent_cells = 1;
    nested.relationship.parent_dx_m = 1000.0;
    nested.relationship.parent_dy_m = 1000.0;
    nested.relationship.parent_dt_s = 1.0;
    nested.relationship.child_dx_m = 500.0;
    nested.relationship.child_dy_m = 500.0;
    nested.relationship.child_dt_s = 0.5;
    nested.parent_time_bracket = {1, true, 0.0, 1.0, 2};
    nested.child = AtmosphericLevelStorage(
        GridStorageState{6, 6, 3, 500.0, 500.0, 0.5, 500.0});
    nested.child.grid.geometry.initialize(
        6, 6, 3, 500.0, 500.0, 500.0,
        CoordinateSystem::Cartesian, StaggerType::CGrid);
    nested.child.time_s = 1.0;
    nested.child.domain_frame.earth_origin_at_t0_m = {4500.0, 4500.0};
    nested.child.domain_frame.velocity_ms = {0.0, 0.0};
    nested.initial_parent_origin_i = 4;
    nested.initial_parent_origin_j = 4;
    nested.relationship_ready = true;
    nested.reference_state_ready = true;
    nested.physical_primary_state_ready = true;
    nested.parent_boundary_state_ready = true;
    nested.interface_source.available = true;
    nested.interface_source.application_count = 3;
    nested.interface_source.last.theta.target_dry_mass_change_kg = 1.0;
    nested.interface_source.cumulative.theta.target_dry_mass_change_kg = 2.0;
    nested.two_way_interval_source_begin.theta
        .committed_dry_mass_change_kg = 3.0;
    nested.two_way_child_interval_begin_inventory = {4, 5, 6, 7, 8};
    nested.two_way_dry_mass_storage_begin = 9.0;
    nested.two_way_theta_storage_begin = 10.0;
    nested.last_two_way_feedback.complete = true;
    nested.last_two_way_feedback.interface_mismatch = {11, 12, 13, 14, 15};
    nested.last_two_way_feedback.applied_reflux = {16, 17, 18, 19, 20};
    nested.two_way_feedback_application_count = 4;
#define NIMBUS_D5_SEED_NESTED_PRIMARY(name) \
    nested.child.fields.name = indexed_field(6, 6, 3, 100.0f);
    NIMBUS_D5_PRIMARY_OWNER_LIST(NIMBUS_D5_SEED_NESTED_PRIMARY)
#undef NIMBUS_D5_SEED_NESTED_PRIMARY
    nested.child.runtime.turbulence.config.scheme_id = "d5_stub";
    nested.child.runtime.turbulence.scheme =
        std::make_unique<D5TurbulenceScheme>();
    nested.child.runtime.chaos.config.scheme_id = "none";
    nested.child.runtime.chaos.scheme =
        std::make_unique<D5ChaosScheme>("none");
    nested.child.runtime.advection.config.scheme_id = "d5_stub";
    nested.child.runtime.advection.scheme =
        std::make_unique<D5AdvectionScheme>();
    nested.child.runtime.diffusion.config.scheme_id = "d5_stub";
    nested.child.runtime.diffusion.scheme =
        std::make_unique<D5DiffusionScheme>();
    nested.child.runtime.time_stepping.config.scheme_id = "d5_stub";
    nested.child.runtime.time_stepping.scheme =
        std::make_unique<D5TimeSteppingScheme>();
    nested.child.runtime.dynamics.scheme =
        std::make_unique<D5DynamicsScheme>();
    return storage;
}

} // namespace

TEST_CASE("whole-cell shift plan defines Earth-window sign and exact overlap",
          "[core][domain_recycling][plan]")
{
    const auto plan = tmv::domain::plan_whole_cell_shift(5, 6, 3, 1, -2);
    REQUIRE(plan.destination_i_begin == 0);
    REQUIRE(plan.destination_i_end == 4);
    REQUIRE(plan.destination_j_begin == 2);
    REQUIRE(plan.destination_j_end == 6);
    REQUIRE(plan.source_i_begin == 1);
    REQUIRE(plan.source_i_end == 5);
    REQUIRE(plan.source_j_begin == 0);
    REQUIRE(plan.source_j_end == 4);
    REQUIRE(plan.retained_cells == 4 * 4 * 3);
    REQUIRE(plan.leading_cells == (5 * 6 - 4 * 4) * 3);
}

TEST_CASE("homogeneous whole-cell shift preserves overlap and initializes every leading cell",
          "[core][domain_recycling][overlap]")
{
    Field3D field = indexed_field(5, 6, 3);
    const Field3D before = field;
    const float* const original_pointer = field.data();
    const std::size_t original_capacity = field.capacity();
    const std::vector<float> profile{-1.0f, -2.0f, -3.0f};
    const auto plan = tmv::domain::plan_whole_cell_shift(5, 6, 3, 1, -2);
    const auto report = tmv::domain::execute_homogeneous_field_shift(
        plan, {{"theta", &field, &profile}});

    require_shifted(before, field, plan, profile);
    REQUIRE(field.data() == original_pointer);
    REQUIRE(field.capacity() == original_capacity);
    REQUIRE(report.fields_shifted == 1);
    REQUIRE(std::string(
                tmv::domain::terrain_aware_field_shift_algorithm_id()) ==
            "stagger_aware_destination_indexed_terrain_whole_cell_shift_v1");
    REQUIRE(report.retained_values == plan.retained_cells);
    REQUIRE(report.initialized_values == plan.leading_cells);
    REQUIRE(report.retained_bytes == plan.retained_cells * sizeof(float));
    REQUIRE(report.initialized_bytes == plan.leading_cells * sizeof(float));
    REQUIRE(std::string(tmv::domain::homogeneous_field_shift_algorithm_id()) ==
            "stagger_aware_in_place_homogeneous_whole_cell_shift_v2");
}

TEST_CASE("whole-cell shift handles the opposite diagonal without changing allocation",
          "[core][domain_recycling][overlap]")
{
    Field3D field = indexed_field(6, 5, 4);
    const Field3D before = field;
    const float* const original_pointer = field.data();
    const std::vector<float> profile{11.0f, 12.0f, 13.0f, 14.0f};
    const auto plan = tmv::domain::plan_whole_cell_shift(6, 5, 4, -2, 1);
    tmv::domain::execute_homogeneous_field_shift(plan, {{"qv", &field, &profile}});
    require_shifted(before, field, plan, profile);
    REQUIRE(field.data() == original_pointer);
}

TEST_CASE("multi-field shift preflight is atomic on invalid leading state",
          "[core][domain_recycling][transaction]")
{
    Field3D first = indexed_field(4, 5, 3);
    Field3D second = indexed_field(4, 5, 3, 10000.0f);
    const Field3D first_before = first;
    const Field3D second_before = second;
    const std::vector<float> valid{1.0f, 2.0f, 3.0f};
    const std::vector<float> invalid{4.0f, std::numeric_limits<float>::quiet_NaN(), 6.0f};
    const auto plan = tmv::domain::plan_whole_cell_shift(4, 5, 3, 1, 1);

    REQUIRE_THROWS_WITH(
        tmv::domain::execute_homogeneous_field_shift(
            plan, {{"rho", &first, &valid}, {"theta", &second, &invalid}}),
        Catch::Matchers::Contains("non-finite"));
    REQUIRE(first.size() == first_before.size());
    REQUIRE(second.size() == second_before.size());
    for (std::size_t index = 0; index < first.size(); ++index)
    {
        REQUIRE(first.data()[index] == first_before.data()[index]);
        REQUIRE(second.data()[index] == second_before.data()[index]);
    }
}

TEST_CASE("whole-cell transaction rejects aliasing and non-retaining displacement",
          "[core][domain_recycling][transaction]")
{
    Field3D field = indexed_field(4, 5, 3);
    const std::vector<float> profile{1.0f, 2.0f, 3.0f};
    const auto plan = tmv::domain::plan_whole_cell_shift(4, 5, 3, 1, 0);
    REQUIRE_THROWS_WITH(
        tmv::domain::execute_homogeneous_field_shift(
            plan, {{"rho", &field, &profile}, {"rho_duplicate", &field, &profile}}),
        Catch::Matchers::Contains("aliases"));
    REQUIRE_THROWS_WITH(tmv::domain::plan_whole_cell_shift(4, 5, 3, 4, 0),
                        Catch::Matchers::Contains("retain at least one"));
    REQUIRE_THROWS_WITH(tmv::domain::plan_whole_cell_shift(4, 5, 3, 0, 0),
                        Catch::Matchers::Contains("non-zero"));
}

TEST_CASE("volume and surface owners move under one allocation-stable transaction",
          "[core][domain_recycling][surface][transaction]")
{
    Field3D volume = indexed_field(5, 6, 3);
    const Field3D volume_before = volume;
    const std::vector<float> profile{21.0f, 22.0f, 23.0f};
    std::vector<float> surface(5 * 6);
    for (std::size_t index = 0; index < surface.size(); ++index)
        surface[index] = static_cast<float>(index + 100);
    const std::vector<float> surface_before = surface;
    const float* const volume_pointer = volume.data();
    const float* const surface_pointer = surface.data();
    const auto plan = tmv::domain::plan_whole_cell_shift(5, 6, 3, -2, 1);

    const auto report = tmv::domain::execute_homogeneous_window_shift(
        plan,
        {{"theta", &volume, &profile}},
        {{"accumulated_rainfall_mm", &surface, 0.0f}});

    require_shifted(volume_before, volume, plan, profile);
    require_surface_shifted(surface_before, surface, plan, 0.0f);
    REQUIRE(volume.data() == volume_pointer);
    REQUIRE(surface.data() == surface_pointer);
    REQUIRE(report.volumes.fields_shifted == 1);
    REQUIRE(report.surfaces_shifted == 1);
    REQUIRE(report.retained_surface_values == 3 * 5);
    REQUIRE(report.initialized_surface_values == 5 * 6 - 3 * 5);
}

TEST_CASE("D5 volume transaction accepts destination-indexed terrain leading columns",
          "[core][domain_recycling][terrain][transaction][D5]")
{
    const auto plan = tmv::domain::plan_whole_cell_shift(5, 6, 3, 1, -1);
    Field3D field = indexed_field(5, 6, 3, 1000.0f);
    const Field3D before = field;
    Field3D leading = indexed_field(5, 6, 3, -5000.0f);
    const float* const pointer = field.data();
    const auto report = tmv::domain::execute_homogeneous_field_shift(
        plan, {{"theta", &field, nullptr,
                tmv::domain::HorizontalFieldLocation::ScalarCenter,
                &leading}});
    require_shifted_from_field(before, field, plan, leading);
    REQUIRE(field.data() == pointer);
    REQUIRE(report.fields_shifted == 1);

    Field3D first = before;
    Field3D second = before;
    const Field3D first_before = first;
    Field3D malformed(1, 1, 1, 0.0f);
    REQUIRE_THROWS_WITH(
        tmv::domain::execute_homogeneous_field_shift(
            plan,
            {{"first", &first, nullptr,
              tmv::domain::HorizontalFieldLocation::ScalarCenter,
              &leading},
             {"second", &second, nullptr,
              tmv::domain::HorizontalFieldLocation::ScalarCenter,
              &malformed}}),
        Catch::Matchers::Contains("leading field shape"));
    for (std::size_t index = 0; index < first.size(); ++index)
    {
        REQUIRE(first.data()[index] == first_before.data()[index]);
        REQUIRE(second.data()[index] == before.data()[index]);
    }
}

TEST_CASE("invalid surface state aborts before any volume or surface write",
          "[core][domain_recycling][surface][transaction]")
{
    Field3D volume = indexed_field(4, 5, 3);
    const Field3D volume_before = volume;
    const std::vector<float> profile{1.0f, 2.0f, 3.0f};
    std::vector<float> valid_surface(4 * 5, 7.0f);
    std::vector<float> invalid_surface(4 * 5 - 1, 8.0f);
    const std::vector<float> surface_before = valid_surface;
    const auto plan = tmv::domain::plan_whole_cell_shift(4, 5, 3, 1, -1);

    REQUIRE_THROWS_WITH(
        tmv::domain::execute_homogeneous_window_shift(
            plan,
            {{"rho", &volume, &profile}},
            {{"skin_theta", &valid_surface, 300.0f}, {"bad", &invalid_surface, 0.0f}}),
        Catch::Matchers::Contains("shape does not match"));
    for (std::size_t index = 0; index < volume.size(); ++index)
        REQUIRE(volume.data()[index] == volume_before.data()[index]);
    REQUIRE(valid_surface == surface_before);
}

TEST_CASE("D5 split preflight and post-fence commit exactly match the atomic CPU transaction",
          "[core][domain_recycling][transaction][preflight][D5]")
{
    const auto plan = tmv::domain::plan_whole_cell_shift(5, 6, 3, -1, 2);
    const std::vector<float> profile{281.0f, 283.0f, 286.0f};

    Field3D oracle_volume = indexed_field(5, 6, 3, 50.0f);
    Field3D split_volume = oracle_volume;
    const Field3D split_before = split_volume;
    std::vector<float> oracle_surface(5U * 6U);
    for (std::size_t index = 0; index < oracle_surface.size(); ++index)
        oracle_surface[index] = static_cast<float>(1000U + index);
    std::vector<float> split_surface = oracle_surface;
    const std::vector<float> split_surface_before = split_surface;
    double oracle_origin = 0.0;
    double split_origin = 0.0;
    bool oracle_valid = true;
    bool split_valid = true;

    const auto oracle_report = tmv::domain::execute_homogeneous_window_shift(
        plan, {{"theta", &oracle_volume, &profile}},
        {{"skin_theta", &oracle_surface, 295.0f}},
        {{"domain_frame.origin_x", &oracle_origin, -100.0}}, {},
        {{"cache_valid", &oracle_valid, false}});

    const auto split_report = tmv::domain::preflight_homogeneous_window_shift(
        plan, {{"theta", &split_volume, &profile}},
        {{"skin_theta", &split_surface, 295.0f}},
        {{"domain_frame.origin_x", &split_origin, -100.0}}, {},
        {{"cache_valid", &split_valid, false}});

    for (std::size_t index = 0; index < split_volume.size(); ++index)
        REQUIRE(split_volume.data()[index] == split_before.data()[index]);
    REQUIRE(split_surface == split_surface_before);
    REQUIRE(split_origin == 0.0);
    REQUIRE(split_valid);

    // This is the host oracle for the fence-complete Vulkan volume batch.
    (void)tmv::domain::execute_homogeneous_field_shift(
        plan, {{"theta", &split_volume, &profile}});
    tmv::domain::commit_preflighted_homogeneous_window_nonvolume(
        plan, {{"skin_theta", &split_surface, 295.0f}},
        {{"domain_frame.origin_x", &split_origin, -100.0}}, {},
        {{"cache_valid", &split_valid, false}}, {}, {});

    for (std::size_t index = 0; index < split_volume.size(); ++index)
        REQUIRE(split_volume.data()[index] == oracle_volume.data()[index]);
    REQUIRE(split_surface == oracle_surface);
    REQUIRE(split_origin == oracle_origin);
    REQUIRE(split_valid == oracle_valid);
    REQUIRE(split_report.volumes.fields_shifted ==
            oracle_report.volumes.fields_shifted);
    REQUIRE(split_report.volumes.retained_values ==
            oracle_report.volumes.retained_values);
    REQUIRE(split_report.volumes.initialized_values ==
            oracle_report.volumes.initialized_values);
    REQUIRE(split_report.surfaces_shifted == oracle_report.surfaces_shifted);
    REQUIRE(split_report.scalars_committed == oracle_report.scalars_committed);
    REQUIRE(split_report.flags_committed == oracle_report.flags_committed);
}

TEST_CASE("generated primary registry builds the complete physical flat environment",
          "[core][domain_recycling][primary][registry]")
{
    SimulationStorageState storage(GridStorageState{5, 6, 3, 100.0, 100.0, 0.1, 100.0});
    storage.grid.geometry.initialize(
        5, 6, 3, 100.0, 100.0, 100.0, CoordinateSystem::Cartesian);
    storage.grid.geometry.staggered = true;
    storage.base.p0 = {100000.0, 99000.0, 98000.0};
    storage.base.theta0 = {300.0, 301.0, 302.0};
    storage.base.qv0 = {0.012, 0.010, 0.008};
    storage.base.u0 = {5.0, 7.0, 9.0};
    storage.base.v0 = {-2.0, 0.0, 2.0};
    storage.base.rho0.resize(3);
    constexpr double kappa = physical_constants::gas_constant_dry_air_jkgk /
                             physical_constants::specific_heat_cp_jkgk;
    for (std::size_t k = 0; k < 3; ++k)
    {
        const double temperature = storage.base.theta0[k] * std::pow(
            storage.base.p0[k] / physical_constants::reference_pressure_pa, kappa);
        storage.base.rho0[k] = storage.base.p0[k] /
            (physical_constants::gas_constant_dry_air_jkgk * temperature);
    }
    const std::vector<double> environmental_tke{0.1, 0.1, 0.1};

#define NIMBUS_D5_SIZE_PRIMARY(name) storage.fields.name = indexed_field(5, 6, 3, 1000.0f);
    NIMBUS_D5_PRIMARY_OWNER_LIST(NIMBUS_D5_SIZE_PRIMARY)
#undef NIMBUS_D5_SIZE_PRIMARY

    const PrognosticFieldStorage before = storage.fields;
    const auto profiles =
        tmv::domain::prepare_flat_homogeneous_primary_profiles(storage, environmental_tke);
    const auto bindings = tmv::domain::bind_homogeneous_primary_fields(storage, profiles);
    REQUIRE(bindings.size() == 15);
    REQUIRE(tmv::domain::homogeneous_primary_owner_names().size() == 15);
    REQUIRE(std::string(tmv::domain::homogeneous_primary_profile_algorithm_id()) ==
            "checkpointed_flat_base_profiles_v1");

    const auto physical = physical_scalar_domain(make_grid_state_view(storage.grid));
    const auto plan = tmv::domain::plan_whole_cell_shift_for_physical_domain(
        5, 6, 3, 1, -1,
        physical.i_begin, physical.i_end,
        physical.j_begin, physical.j_end);
    const auto report = tmv::domain::execute_homogeneous_field_shift(plan, bindings);
    REQUIRE(report.fields_shifted == 15);
#define NIMBUS_D5_CHECK_PRIMARY(name)                                                \
    do                                                                               \
    {                                                                                \
        const auto owner = tmv::domain::HomogeneousPrimaryOwner::name;              \
        const auto location = owner == tmv::domain::HomogeneousPrimaryOwner::u      \
            ? tmv::domain::HorizontalFieldLocation::XFace                           \
            : (owner == tmv::domain::HomogeneousPrimaryOwner::v                     \
                   ? tmv::domain::HorizontalFieldLocation::YFace                    \
                   : tmv::domain::HorizontalFieldLocation::ScalarCenter);           \
        const auto owner_plan =                                                      \
            tmv::domain::plan_whole_cell_shift_for_horizontal_location(             \
                plan, location);                                                     \
        require_shifted(before.name, storage.fields.name, owner_plan,                \
                        profiles[owner]);                                             \
    } while (false);
    NIMBUS_D5_PRIMARY_OWNER_LIST(NIMBUS_D5_CHECK_PRIMARY)
#undef NIMBUS_D5_CHECK_PRIMARY

    for (std::size_t k = 0; k < 3; ++k)
        REQUIRE(profiles[tmv::domain::HomogeneousPrimaryOwner::rho][k] ==
                static_cast<float>(storage.base.rho0[k]));
    REQUIRE(profiles[tmv::domain::HomogeneousPrimaryOwner::p] ==
            std::vector<float>{100000.0f, 99000.0f, 98000.0f});
    REQUIRE(profiles[tmv::domain::HomogeneousPrimaryOwner::theta] ==
            std::vector<float>{300.0f, 301.0f, 302.0f});
    REQUIRE(profiles[tmv::domain::HomogeneousPrimaryOwner::qv] ==
            std::vector<float>{0.012f, 0.010f, 0.008f});
    REQUIRE(profiles[tmv::domain::HomogeneousPrimaryOwner::u] ==
            std::vector<float>{5.0f, 7.0f, 9.0f});
    REQUIRE(profiles[tmv::domain::HomogeneousPrimaryOwner::v] ==
            std::vector<float>{-2.0f, 0.0f, 2.0f});

    for (const auto owner : {tmv::domain::HomogeneousPrimaryOwner::w,
                             tmv::domain::HomogeneousPrimaryOwner::tracer,
                             tmv::domain::HomogeneousPrimaryOwner::qc,
                             tmv::domain::HomogeneousPrimaryOwner::qr,
                             tmv::domain::HomogeneousPrimaryOwner::qi,
                             tmv::domain::HomogeneousPrimaryOwner::qs,
                             tmv::domain::HomogeneousPrimaryOwner::qh,
                             tmv::domain::HomogeneousPrimaryOwner::qg})
        REQUIRE(profiles[owner] == std::vector<float>(3, 0.0f));
    REQUIRE(profiles[tmv::domain::HomogeneousPrimaryOwner::tke] ==
            std::vector<float>(3, 0.1f));
}

TEST_CASE("D5 preparation rejects legacy nested storage",
          "[core][domain_recycling][nested][rollback][D5]")
{
    SimulationStorageState storage(
        GridStorageState{5, 6, 3, 100.0, 100.0, 0.1, 100.0});
    storage.grid.geometry.initialize(
        5, 6, 3, 100.0, 100.0, 100.0,
        CoordinateSystem::Cartesian);
    storage.grid.geometry.staggered = true;
    storage.runtime.terrain.config.scheme_id = "none";
    tmv::domain::StormWindowMovementDecision decision;

    storage.nested.config.enabled = true;

    REQUIRE_THROWS_WITH(
        tmv::domain::prepare_flat_homogeneous_domain_recycling(
            storage, decision, std::vector<double>(3, 0.1)),
        Catch::Matchers::Contains("legacy nested storage"));
}

TEST_CASE("D5 rebases separately evolving nested storage without moving child state",
          "[core][domain_recycling][nested][transaction][D5]")
{
    SimulationStorageState storage =
        separately_evolving_nested_recycling_storage();
    const auto plan = tmv::domain::plan_whole_cell_shift_for_physical_domain(
        14, 14, 3, 1, 0, 1, 13, 1, 13);
    auto post_frame = storage.configuration.domain_frame;
    (void)tmv::domain::commit_whole_cell_shift_to_domain_frame(
        post_frame, plan, storage.grid.dx, storage.grid.dy);
    auto prepared = tmv::domain::prepare_nested_grid_domain_recycling(
        storage, plan, post_frame);
    REQUIRE(prepared.active);
    REQUIRE(prepared.committed_relationship.child_origin_parent_cell_i == 3);
    REQUIRE(prepared.committed_recycling_shift_i == 1);
    REQUIRE(std::string(tmv::domain::nested_grid_domain_recycling_algorithm_id())
            == "preserve_child_earth_frame_rebase_parent_indices_v1");

    const Field3D child_rho_before = storage.nested_execution.child.fields.rho;
    const auto child_frame_before =
        storage.nested_execution.child.domain_frame;
    const auto child_bracket_before =
        storage.nested_execution.parent_time_bracket;
    const auto feedback_before =
        storage.nested_execution.last_two_way_feedback;
    const auto interface_before =
        storage.nested_execution.interface_source;
    const auto* turbulence_scheme =
        storage.nested_execution.child.runtime.turbulence.scheme.get();
    const auto* chaos_scheme =
        storage.nested_execution.child.runtime.chaos.scheme.get();
    std::vector<tmv::domain::HomogeneousRuntimeCommitBinding> commits;
    REQUIRE(tmv::domain::append_nested_grid_domain_recycling_binding(
        storage, prepared, commits));
    REQUIRE(commits.size() == 1);
    (void)tmv::domain::execute_homogeneous_window_shift(
        plan, {},
        {},
        {{"domain_frame.origin_x",
          &storage.configuration.domain_frame.earth_origin_at_t0_m.x_m,
          post_frame.earth_origin_at_t0_m.x_m},
         {"domain_frame.origin_y",
          &storage.configuration.domain_frame.earth_origin_at_t0_m.y_m,
          post_frame.earth_origin_at_t0_m.y_m}},
        {}, {}, {}, commits);

    const auto& nested = storage.nested_execution;
    REQUIRE(nested.relationship.child_origin_parent_cell_i == 3);
    REQUIRE(nested.relationship.child_origin_parent_cell_j == 4);
    REQUIRE(nested.cumulative_parent_shift_i == 0);
    REQUIRE(nested.cumulative_parent_recycling_shift_i == 1);
    REQUIRE(nested.child.time_s == storage.time_s);
    REQUIRE(nested.child.domain_frame.earth_origin_at_t0_m.x_m ==
            child_frame_before.earth_origin_at_t0_m.x_m);
    REQUIRE(nested.child.domain_frame.earth_origin_at_t0_m.y_m ==
            child_frame_before.earth_origin_at_t0_m.y_m);
    REQUIRE(nested.parent_time_bracket.previous_parent_time_s ==
            child_bracket_before.previous_parent_time_s);
    REQUIRE(nested.parent_time_bracket.current_parent_time_s ==
            child_bracket_before.current_parent_time_s);
    REQUIRE(nested.last_two_way_feedback.interface_mismatch ==
            feedback_before.interface_mismatch);
    REQUIRE(nested.last_two_way_feedback.applied_reflux ==
            feedback_before.applied_reflux);
    REQUIRE(nested.last_two_way_feedback.storage_residual ==
            feedback_before.storage_residual);
    REQUIRE(nested.last_two_way_feedback.child_interface_source ==
            feedback_before.child_interface_source);
    REQUIRE(nested.last_two_way_feedback.child_dynamics_source ==
            feedback_before.child_dynamics_source);
    REQUIRE(nested.interface_source.application_count ==
            interface_before.application_count);
    REQUIRE(nested.interface_source.last.theta.target_dry_mass_change_kg ==
            interface_before.last.theta.target_dry_mass_change_kg);
    REQUIRE(nested.child.runtime.turbulence.scheme.get() == turbulence_scheme);
    REQUIRE(nested.child.runtime.chaos.scheme.get() == chaos_scheme);
    for (std::size_t index = 0; index < child_rho_before.size(); ++index)
        REQUIRE(nested.child.fields.rho.data()[index] ==
                child_rho_before.data()[index]);
}

TEST_CASE("Parent recycling planner bounds preserve the fixed child and its stencil",
          "[core][domain_recycling][nested][planning]")
{
    auto                                     storage      = separately_evolving_nested_recycling_storage();
    const auto                               relationship = storage.nested_execution.relationship;
    tmv::domain::StormWindowMovementEvidence original;
    original.sample_time_s   = 17;
    original.confidence      = .9;
    original.distance_west_m = 350;
    const auto bounded       = tmv::domain::with_nested_parent_recycling_bounds(relationship, original);
    REQUIRE(bounded.legal_shift_bounds_available);
    REQUIRE(bounded.sample_time_s == original.sample_time_s);
    REQUIRE(bounded.confidence == original.confidence);
    REQUIRE(bounded.distance_west_m == original.distance_west_m);
    for (int i = -12; i <= 12; ++i)
    {
        for (int j = -12; j <= 12; ++j)
        {
            const auto rebased_i = relationship.child_origin_parent_cell_i - i;
            const auto rebased_j = relationship.child_origin_parent_cell_j - j;
            const int  margin    = relationship.boundary_stencil_parent_cells;
            const bool fits = rebased_i >= margin && rebased_j >= margin &&
                              rebased_i + relationship.child_parent_span_x <= relationship.parent_cells_x - margin &&
                              rebased_j + relationship.child_parent_span_y <= relationship.parent_cells_y - margin;
            const bool admitted = i >= bounded.minimum_i_shift_cells && i <= bounded.maximum_i_shift_cells &&
                                  j >= bounded.minimum_j_shift_cells && j <= bounded.maximum_j_shift_cells;
            CHECK(admitted == fits);
        }
    }
    auto invalid                       = relationship;
    invalid.child_origin_parent_cell_i = 0;
    REQUIRE_THROWS(tmv::domain::with_nested_parent_recycling_bounds(invalid, original));
    REQUIRE_FALSE(original.legal_shift_bounds_available);
}

TEST_CASE("Mapped nested D5 publishes geometry atomically and rejects stale child terrain",
          "[core][domain_recycling][nested][terrain]")
{
    const std::string scheme                       = GENERATE("bell", "schar");
    auto              storage                      = separately_evolving_nested_recycling_storage();
    auto&             nested                       = storage.nested_execution;
    auto&             terrain                      = nested.child.runtime.terrain;
    auto&             config                       = terrain.config;
    config.scheme_id                               = scheme;
    config.horizontal_placement_id                 = "earth_fixed";
    config.ztop                                    = 1500;
    config.bell.h0                                 = 100;
    config.schar.H                                 = 100;
    config.feature_center_earth_x_m                = 6000;
    config.feature_center_earth_y_m                = 6500;
    const auto&                                  r = nested.relationship;
    tmv::domain::NestedMappedConservativeStateV1 parent;
    parent.scalar.resize(r.parent_cells_x, r.parent_cells_y, r.levels_z, 1);
    parent.x_flux.resize(r.parent_cells_x + 1, r.parent_cells_y, r.levels_z, 2);
    parent.y_flux.resize(r.parent_cells_x, r.parent_cells_y + 1, r.levels_z, 3);
    parent.zeta_flux.resize(r.parent_cells_x, r.parent_cells_y, r.levels_z + 1, 0);
    nested.active_terrain =
        tmv::domain::prepare_nested_active_terrain_commit({}, config, r, {1000, 1000}, parent, storage.time_s, 0, 0)
            .candidate;
    terrain.metrics               = nested.active_terrain.geometry.child_storage.metrics;
    terrain.topography            = nested.active_terrain.geometry.child_storage.topography;
    auto& placement               = terrain.earth_fixed_placement;
    placement.active              = true;
    placement.generation          = nested.active_terrain.generation;
    placement.grid_origin_earth_m = nested.child.domain_frame.earth_origin_at_t0_m;
    const auto plan       = tmv::domain::plan_whole_cell_shift_for_physical_domain(14, 14, 3, 1, -1, 1, 13, 1, 13);
    auto       post_frame = storage.configuration.domain_frame;
    tmv::domain::commit_whole_cell_shift_to_domain_frame(post_frame, plan, storage.grid.dx, storage.grid.dy);
    const auto before   = nested.active_terrain;
    auto       prepared = tmv::domain::prepare_nested_grid_domain_recycling(storage, plan, post_frame);
    REQUIRE(prepared.mapped_rebase);
    REQUIRE(tmv::domain::can_commit_nested_grid_domain_recycling(storage, prepared));
    terrain.metrics.J.values[0] += .25;
    CHECK_FALSE(tmv::domain::can_commit_nested_grid_domain_recycling(storage, prepared));
    terrain.metrics.J = before.geometry.child_storage.metrics.J;
    config.feature_center_earth_x_m += 1;
    CHECK_FALSE(tmv::domain::can_commit_nested_grid_domain_recycling(storage, prepared));
    config.feature_center_earth_x_m -= 1;
    placement.grid_origin_earth_m.x_m += 1;
    CHECK_FALSE(tmv::domain::can_commit_nested_grid_domain_recycling(storage, prepared));
    placement.grid_origin_earth_m.x_m -= 1;
    ++placement.generation;
    CHECK_FALSE(tmv::domain::can_commit_nested_grid_domain_recycling(storage, prepared));
    --placement.generation;
    nested.active_terrain.child_state.scalar.values[0] += 1;
    CHECK_FALSE(tmv::domain::can_commit_nested_grid_domain_recycling(storage, prepared));
    nested.active_terrain.child_state.scalar = before.child_state.scalar;
    REQUIRE(tmv::domain::can_commit_nested_grid_domain_recycling(storage, prepared));
    std::vector<tmv::domain::HomogeneousRuntimeCommitBinding> commits;
    REQUIRE(tmv::domain::append_nested_grid_domain_recycling_binding(storage, prepared, commits));
    REQUIRE(commits.size() == 1);
    commits.front().commit(commits.front().prepared_state);
    CHECK(nested.relationship.child_origin_parent_cell_i == 3);
    CHECK(nested.relationship.child_origin_parent_cell_j == 5);
    CHECK(nested.active_terrain.geometry.relationship.child_origin_parent_cell_i == 3);
    CHECK(nested.active_terrain.geometry.parent_origin_earth_m.x_m == 2000);
    CHECK(nested.active_terrain.geometry.parent_origin_earth_m.y_m == 0);
    CHECK(nested.active_terrain.geometry.child_origin_earth_m.x_m == before.geometry.child_origin_earth_m.x_m);
    CHECK(nested.active_terrain.geometry.child_origin_earth_m.y_m == before.geometry.child_origin_earth_m.y_m);
    CHECK(nested.active_terrain.geometry.child_storage.metrics.J.values == terrain.metrics.J.values);
    CHECK(nested.active_terrain.child_state.scalar.values == before.child_state.scalar.values);
    CHECK(nested.active_terrain.generation == before.generation + 1);
    CHECK(placement.generation == nested.active_terrain.generation);
    CHECK_FALSE(tmv::domain::can_commit_nested_grid_domain_recycling(storage, prepared));
}

TEST_CASE("nested D5 rebase rejects stale and out-of-parent candidates atomically",
          "[core][domain_recycling][nested][rollback][D5]")
{
    SimulationStorageState storage = separately_evolving_nested_recycling_storage();
    const auto             plan = tmv::domain::plan_whole_cell_shift_for_physical_domain(14, 14, 3, 1, 0, 1, 13, 1, 13);
    auto post_frame = storage.configuration.domain_frame;
    (void)tmv::domain::commit_whole_cell_shift_to_domain_frame(
        post_frame, plan, storage.grid.dx, storage.grid.dy);
    auto prepared = tmv::domain::prepare_nested_grid_domain_recycling(
        storage, plan, post_frame);
    const auto relationship_before = storage.nested_execution.relationship;
    storage.nested_execution.child.fields.theta(2, 2, 1) += 1.0f;
    REQUIRE_FALSE(tmv::domain::can_commit_nested_grid_domain_recycling(
        storage, prepared));
    std::vector<tmv::domain::HomogeneousRuntimeCommitBinding> commits;
    REQUIRE_THROWS_WITH(
        tmv::domain::append_nested_grid_domain_recycling_binding(
            storage, prepared, commits),
        Catch::Matchers::Contains("stale"));
    REQUIRE(commits.empty());
    REQUIRE(storage.nested_execution.relationship
                .child_origin_parent_cell_i ==
            relationship_before.child_origin_parent_cell_i);
    REQUIRE(storage.configuration.domain_frame.earth_origin_at_t0_m.x_m ==
            0.0);

    storage = separately_evolving_nested_recycling_storage();
    post_frame = storage.configuration.domain_frame;
    (void)tmv::domain::commit_whole_cell_shift_to_domain_frame(
        post_frame, plan, storage.grid.dx, storage.grid.dy);
    prepared = tmv::domain::prepare_nested_grid_domain_recycling(
        storage, plan, post_frame);
    storage.nested_execution.last_two_way_feedback.applied_reflux[3] += 1.0;
    REQUIRE_FALSE(tmv::domain::can_commit_nested_grid_domain_recycling(
        storage, prepared));
    REQUIRE_THROWS_WITH(
        tmv::domain::append_nested_grid_domain_recycling_binding(
            storage, prepared, commits),
        Catch::Matchers::Contains("stale"));
    REQUIRE(commits.empty());
    REQUIRE(storage.nested_execution.relationship
                .child_origin_parent_cell_i == 4);

    storage = separately_evolving_nested_recycling_storage();
    const auto illegal =
        tmv::domain::plan_whole_cell_shift_for_physical_domain(
            14, 14, 3, 4, 0, 1, 13, 1, 13);
    post_frame = storage.configuration.domain_frame;
    (void)tmv::domain::commit_whole_cell_shift_to_domain_frame(
        post_frame, illegal, storage.grid.dx, storage.grid.dy);
    REQUIRE_THROWS_WITH(
        tmv::domain::prepare_nested_grid_domain_recycling(
            storage, illegal, post_frame),
        Catch::Matchers::Contains("stencil margin"));
    REQUIRE(storage.nested_execution.relationship
                .child_origin_parent_cell_i == 4);
}

TEST_CASE("homogeneous primary profile preparation rejects incomplete and unphysical bases",
          "[core][domain_recycling][primary][transaction]")
{
    SimulationStorageState storage(GridStorageState{4, 5, 3, 100.0, 100.0, 0.1, 100.0});
    storage.base.p0 = {100000.0, 99000.0, 98000.0};
    storage.base.theta0 = {300.0, 301.0, 302.0};
    storage.base.qv0 = {0.012, 0.010, 0.008};
    storage.base.u0 = {5.0, 7.0, 9.0};
    storage.base.v0 = {-2.0, 0.0, 2.0};
    storage.base.rho0.resize(3);
    constexpr double kappa = physical_constants::gas_constant_dry_air_jkgk /
                             physical_constants::specific_heat_cp_jkgk;
    for (std::size_t k = 0; k < 3; ++k)
    {
        const double temperature = storage.base.theta0[k] * std::pow(
            storage.base.p0[k] / physical_constants::reference_pressure_pa, kappa);
        storage.base.rho0[k] = storage.base.p0[k] /
            (physical_constants::gas_constant_dry_air_jkgk * temperature);
    }
    const std::vector<double> environmental_tke{0.1, 0.1, 0.1};

    storage.base.qv0[1] = -0.001;
    REQUIRE_THROWS_WITH(tmv::domain::prepare_flat_homogeneous_primary_profiles(
                            storage, environmental_tke),
                        Catch::Matchers::Contains("unphysical"));
    storage.base.qv0[1] = 0.010;
    storage.base.p0.pop_back();
    REQUIRE_THROWS_WITH(tmv::domain::prepare_flat_homogeneous_primary_profiles(
                            storage, environmental_tke),
                        Catch::Matchers::Contains("complete base.p0"));

    storage.base.p0 = {100000.0, 99000.0, 98000.0};
    storage.base.rho0[1] *= 1.02;
    REQUIRE_THROWS_WITH(tmv::domain::prepare_flat_homogeneous_primary_profiles(
                            storage, environmental_tke),
                        Catch::Matchers::Contains("dry EOS"));

    storage.base.rho0[1] /= 1.02;
    const std::vector<double> invalid_tke{0.1, -0.1, 0.1};
    REQUIRE_THROWS_WITH(tmv::domain::prepare_flat_homogeneous_primary_profiles(
                            storage, invalid_tke),
                        Catch::Matchers::Contains("unphysical"));
}

TEST_CASE("terrain-aware D5 leading atmosphere samples every destination physical height",
          "[core][domain_recycling][terrain][leading][D5]")
{
    SimulationStorageState storage(
        GridStorageState{6, 7, 4, 500.0, 500.0, 0.1, 500.0});
    storage.grid.geometry.initialize(
        6, 7, 4, 500.0, 500.0, 500.0,
        CoordinateSystem::Cartesian, StaggerType::CGrid);
    storage.base.p0 = {98000.0, 93000.0, 88000.0, 83000.0};
    storage.base.theta0 = {298.0, 300.0, 303.0, 307.0};
    storage.base.qv0 = {0.012, 0.010, 0.008, 0.006};
    storage.base.u0 = {2.0, 5.0, 9.0, 14.0};
    storage.base.v0 = {-3.0, -1.0, 2.0, 6.0};
    storage.base.rho0.resize(4);
    constexpr double kappa =
        physical_constants::gas_constant_dry_air_jkgk
        / physical_constants::specific_heat_cp_jkgk;
    for (std::size_t k = 0; k < 4; ++k)
    {
        const double temperature = storage.base.theta0[k] * std::pow(
            storage.base.p0[k]
                / physical_constants::reference_pressure_pa,
            kappa);
        storage.base.rho0[k] = storage.base.p0[k]
            / (physical_constants::gas_constant_dry_air_jkgk
               * temperature);
    }

    TerrainConfig config;
    config.scheme_id = "bell";
    config.coord_id = "smoothed";
    config.horizontal_placement_id = "earth_fixed";
    config.ztop = 2000.0;
    config.bell.h0 = 450.0;
    config.bell.a = 900.0;
    config.bell.axisymmetric = true;
    config.feature_center_earth_x_m = 1750.0;
    config.feature_center_earth_y_m = 1750.0;
    tmv::EarthFixedTerrainBuildRequest request;
    request.nr = 6;
    request.nth = 7;
    request.nz = 4;
    request.dx_m = 500.0;
    request.dy_m = 500.0;
    request.feature_center_earth_m = {1750.0, 1750.0};
    request.sampling = GENERATE(tmv::EarthFixedTerrainSampling::Point,
                               tmv::EarthFixedTerrainSampling::CellAverageBilinear);
    auto& terrain = storage.runtime.terrain;
    terrain.config = config;
    auto initial_commit = tmv::prepare_earth_fixed_terrain_commit(
        terrain, request);
    REQUIRE(tmv::commit_earth_fixed_terrain(initial_commit, terrain));
    const auto& geometry = terrain;
    const std::vector<double> environmental_tke{0.1, 0.2, 0.3, 0.4};
    const auto leading = tmv::domain::prepare_terrain_aware_primary_fields(
        storage, geometry.metrics, environmental_tke);

    const int low_i = 0;
    const int low_j = 0;
    const int peak_i = 3;
    const int peak_j = 3;
    REQUIRE(geometry.metrics.z(peak_i, peak_j, 0)
            > geometry.metrics.z(low_i, low_j, 0));
    REQUIRE(leading[tmv::domain::HomogeneousPrimaryOwner::p](
                peak_i, peak_j, 0)
            != leading[tmv::domain::HomogeneousPrimaryOwner::p](
                low_i, low_j, 0));
    for (int i = 0; i < request.nr; ++i)
        for (int j = 0; j < request.nth; ++j)
            for (int k = 0; k < request.nz; ++k)
            {
                const double height = geometry.metrics.z(i, j, k);
                REQUIRE(leading[tmv::domain::HomogeneousPrimaryOwner::p](
                            i, j, k)
                        == static_cast<float>(
                            terrain_transport::reference_profile_value_fp32(
                                storage.base.p0, height,
                                storage.grid.dz)));
                REQUIRE(leading[tmv::domain::HomogeneousPrimaryOwner::rho](
                            i, j, k)
                        == static_cast<float>(
                            terrain_transport::reference_density_fp32(
                                storage.base.p0, storage.base.theta0,
                                height, storage.grid.dz)));
                REQUIRE(leading[tmv::domain::HomogeneousPrimaryOwner::w](
                            i, j, k) == 0.0f);
            }

#define NIMBUS_D5_SEED_TERRAIN_OWNER(name)                                      \
    storage.fields.name = leading[tmv::domain::HomogeneousPrimaryOwner::name];
    NIMBUS_D5_PRIMARY_OWNER_LIST(NIMBUS_D5_SEED_TERRAIN_OWNER)
#undef NIMBUS_D5_SEED_TERRAIN_OWNER
    storage.runtime.microphysics.scheme = std::make_unique<tmv::MilbrandtScheme>();
    storage.runtime.microphysics.scheme->prepare_prognostic_state(storage.fields.rho,
                                                                  storage.fields.qc,
                                                                  storage.fields.qr,
                                                                  storage.fields.qi,
                                                                  storage.fields.qs,
                                                                  storage.fields.qg,
                                                                  storage.fields.qh);
    const auto private_fields = storage.runtime.microphysics.scheme->prognostic_scalars_for_transport();
    for (std::size_t f = 0; f < private_fields.size(); ++f)
        *private_fields[f].field = indexed_field(request.nr, request.nth, request.nz, float(1000 * (f + 1)));
    const SimulationStateView state = make_simulation_state_view(storage);
    const PhysicalCellDomain physical = physical_scalar_domain(state.grid);
    const auto plan =
        tmv::domain::plan_whole_cell_shift_for_physical_domain(
            state.grid.nr, state.grid.nth, state.grid.nz, 1, -1,
            physical.i_begin, physical.i_end,
            physical.j_begin, physical.j_end);
    auto post_frame = storage.configuration.domain_frame;
    (void)tmv::domain::commit_whole_cell_shift_to_domain_frame(
        post_frame, plan, storage.grid.dx, storage.grid.dy);
    auto translated_request = request;
    translated_request.grid_origin_earth_m =
        post_frame.earth_origin_at(storage.time_s);
    auto translated_terrain = tmv::prepare_earth_fixed_terrain_commit(
        terrain, translated_request);
    for (int i = plan.destination_i_begin; i < plan.destination_i_end; ++i)
        for (int j = plan.destination_j_begin;
             j < plan.destination_j_end; ++j)
        {
            const int source_i = i + plan.shift_i_cells;
            const int source_j = j + plan.shift_j_cells;
            REQUIRE(translated_terrain.geometry.topography.h[
                        static_cast<std::size_t>(i)]
                    [static_cast<std::size_t>(j)]
                    == terrain.topography.h[
                        static_cast<std::size_t>(source_i)]
                       [static_cast<std::size_t>(source_j)]);
            for (int k = 0; k < plan.nz; ++k)
            {
                REQUIRE(translated_terrain.geometry.metrics.J(i, j, k)
                        == terrain.metrics.J(source_i, source_j, k));
                REQUIRE(translated_terrain.geometry.metrics.Gx_face(i, j, k)
                        == terrain.metrics.Gx_face(source_i, source_j, k));
                REQUIRE(translated_terrain.geometry.metrics.Gy_face(i, j, k)
                        == terrain.metrics.Gy_face(source_i, source_j, k));
            }
        }
    const auto translated_leading =
        tmv::domain::prepare_terrain_aware_primary_fields(
            storage, translated_terrain.geometry.metrics,
            environmental_tke);
    const auto private_leading = tmv::domain::prepare_terrain_aware_microphysics_recycling(state, plan, translated_leading, translated_terrain.geometry.metrics);
    const auto ledger =
        tmv::domain::compute_terrain_aware_domain_recycling_control_volume_exchange_ledger(
            state, plan, translated_leading,
            translated_terrain.geometry.metrics,
            storage.configuration.domain_frame, &private_leading);
    REQUIRE(ledger.retiring.private_inventory.count == private_fields.size());
    auto bindings = tmv::domain::bind_terrain_aware_primary_fields(
        storage, translated_leading);
    REQUIRE(tmv::domain::append_prepared_microphysics_domain_recycling_bindings(
        state, plan, private_leading, bindings));
    tmv::domain::TerrainAwarePrimaryFields before_fields;
    for (std::size_t owner = 0;
         owner < tmv::domain::homogeneous_primary_owner_count(); ++owner)
        before_fields.values[owner] = *bindings[owner].field;
    (void)tmv::domain::execute_homogeneous_field_shift(plan, bindings);
    for (std::size_t owner = 0;
         owner < tmv::domain::homogeneous_primary_owner_count(); ++owner)
    {
        const auto owner_plan =
            tmv::domain::plan_whole_cell_shift_for_horizontal_location(
                plan, bindings[owner].horizontal_location);
        require_shifted_from_field(
            before_fields.values[owner], *bindings[owner].field,
            owner_plan, translated_leading.values[owner]);
    }
    storage.configuration.domain_frame = post_frame;
    REQUIRE(tmv::commit_earth_fixed_terrain(
        translated_terrain, terrain));
    REQUIRE_NOTHROW(
        tmv::domain::verify_domain_recycling_control_volume_exchange_after_commit(
            make_simulation_state_view(storage), plan, post_frame, ledger));
}

TEST_CASE("trailing export captures exactly the old cells excluded from retained overlap",
          "[core][domain_recycling][trailing][transaction]")
{
    const Field3D first = indexed_field(5, 6, 3);
    const Field3D second = indexed_field(5, 6, 3, 10000.0f);
    const auto plan = tmv::domain::plan_whole_cell_shift(5, 6, 3, 1, -2);
    std::vector<float> first_export(plan.leading_cells, -1.0f);
    std::vector<float> second_export(plan.leading_cells, -1.0f);
    const float* const first_pointer = first_export.data();
    const float* const second_pointer = second_export.data();

    const auto report = tmv::domain::export_trailing_fields(
        plan, {{"rho", &first, &first_export}, {"theta", &second, &second_export}});

    std::size_t offset = 0;
    for (int i = 0; i < plan.nr; ++i)
        for (int j = 0; j < plan.nth; ++j)
        {
            const bool retained_source = i >= plan.source_i_begin && i < plan.source_i_end &&
                                         j >= plan.source_j_begin && j < plan.source_j_end;
            if (retained_source) continue;
            for (int k = 0; k < plan.nz; ++k)
            {
                REQUIRE(first_export[offset] == first(i, j, k));
                REQUIRE(second_export[offset] == second(i, j, k));
                ++offset;
            }
        }
    REQUIRE(offset == plan.leading_cells);
    REQUIRE(first_export.data() == first_pointer);
    REQUIRE(second_export.data() == second_pointer);
    REQUIRE(report.fields_exported == 2);
    REQUIRE(report.values_exported == 2 * plan.leading_cells);
    REQUIRE(report.bytes_exported == 2 * plan.leading_cells * sizeof(float));
}

TEST_CASE("trailing export preflights every destination before writing",
          "[core][domain_recycling][trailing][transaction]")
{
    const Field3D first = indexed_field(4, 5, 3);
    const Field3D second = indexed_field(4, 5, 3, 1000.0f);
    const auto plan = tmv::domain::plan_whole_cell_shift(4, 5, 3, -1, 1);
    std::vector<float> valid(plan.leading_cells, 71.0f);
    std::vector<float> invalid(plan.leading_cells - 1, 72.0f);
    const std::vector<float> before = valid;

    REQUIRE_THROWS_WITH(
        tmv::domain::export_trailing_fields(
            plan, {{"rho", &first, &valid}, {"theta", &second, &invalid}}),
        Catch::Matchers::Contains("size does not match"));
    REQUIRE(valid == before);
    std::vector<float> alias_destination(plan.leading_cells, 73.0f);
    REQUIRE_THROWS_WITH(
        tmv::domain::export_trailing_fields(
            plan, {{"rho", &first, &valid},
                   {"rho_again", &first, &alias_destination}}),
                        Catch::Matchers::Contains("aliases"));
}

TEST_CASE("physical-domain shift never promotes closure storage into leading atmosphere",
          "[core][domain_recycling][physical-domain][closure]")
{
    Field3D field = indexed_field(6, 7, 3);
    for (int j = 0; j < 7; ++j)
        for (int k = 0; k < 3; ++k)
            field(5, j, k) = 999999.0f;
    const Field3D before = field;
    const std::vector<float> profile{-1.0f, -2.0f, -3.0f};
    const auto plan = tmv::domain::plan_whole_cell_shift_for_physical_domain(
        6, 7, 3, 1, 0, 1, 5, 1, 6);

    REQUIRE(plan.destination_i_begin == 1);
    REQUIRE(plan.destination_i_end == 4);
    REQUIRE(plan.source_i_begin == 2);
    REQUIRE(plan.source_i_end == 5);
    REQUIRE(plan.retained_cells == 3 * 5 * 3);
    REQUIRE(plan.trailing_cells == 1 * 5 * 3);
    REQUIRE(plan.leading_cells == (6 * 7 - 3 * 5) * 3);

    tmv::domain::execute_homogeneous_field_shift(
        plan, {{"rho", &field, &profile}});
    require_shifted(before, field, plan, profile);
    for (int j = 1; j < 6; ++j)
        for (int k = 0; k < 3; ++k)
            REQUIRE(field(4, j, k) == profile[static_cast<std::size_t>(k)]);

    std::vector<float> trailing(plan.trailing_cells, 0.0f);
    tmv::domain::export_trailing_fields(
        plan, {{"rho", &before, &trailing}});
    std::size_t offset = 0;
    for (int j = 1; j < 6; ++j)
        for (int k = 0; k < 3; ++k)
            REQUIRE(trailing[offset++] == before(1, j, k));
    REQUIRE(offset == trailing.size());
}

TEST_CASE("physical-domain shift is closure-safe in every signed horizontal direction",
          "[core][domain_recycling][physical-domain][closure]")
{
    const std::array<std::pair<int, int>, 8> shifts{{
        {1, 0}, {-1, 0}, {0, 1}, {0, -1},
        {1, 1}, {1, -1}, {-1, 1}, {-1, -1},
    }};
    for (const auto [shift_i, shift_j] : shifts)
    {
        CAPTURE(shift_i, shift_j);
        Field3D field = indexed_field(7, 8, 2);
        for (int i = 0; i < 7; ++i)
            for (int j = 0; j < 8; ++j)
                if (i == 0 || i == 6 || j == 0 || j == 7)
                    for (int k = 0; k < 2; ++k)
                        field(i, j, k) = 900000.0f + static_cast<float>(100 * i + 10 * j + k);
        const Field3D before = field;
        const std::vector<float> profile{-10.0f, -20.0f};
        const auto plan = tmv::domain::plan_whole_cell_shift_for_physical_domain(
            7, 8, 2, shift_i, shift_j, 1, 6, 1, 7);

        tmv::domain::execute_homogeneous_field_shift(
            plan, {{"theta", &field, &profile}});
        require_shifted(before, field, plan, profile);

        std::vector<float> trailing(plan.trailing_cells, 0.0f);
        tmv::domain::export_trailing_fields(
            plan, {{"theta", &before, &trailing}});
        std::size_t offset = 0;
        for (int i = plan.physical_i_begin; i < plan.physical_i_end; ++i)
            for (int j = plan.physical_j_begin; j < plan.physical_j_end; ++j)
            {
                const bool retained =
                    i >= plan.source_i_begin && i < plan.source_i_end &&
                    j >= plan.source_j_begin && j < plan.source_j_end;
                if (retained) continue;
                for (int k = 0; k < plan.nz; ++k)
                    REQUIRE(trailing[offset++] == before(i, j, k));
            }
        REQUIRE(offset == trailing.size());
    }
}

TEST_CASE("C-grid face owners retain and export their exact staggered domains",
          "[core][domain_recycling][physical-domain][staggering]")
{
    const auto scalar_plan =
        tmv::domain::plan_whole_cell_shift_for_physical_domain(
            7, 8, 3, -1, 1, 1, 6, 1, 7);
    const auto u_plan =
        tmv::domain::plan_whole_cell_shift_for_horizontal_location(
            scalar_plan, tmv::domain::HorizontalFieldLocation::XFace);
    const auto v_plan =
        tmv::domain::plan_whole_cell_shift_for_horizontal_location(
            scalar_plan, tmv::domain::HorizontalFieldLocation::YFace);
    REQUIRE(u_plan.physical_i_begin == 0);
    REQUIRE(u_plan.physical_i_end == 6);
    REQUIRE(u_plan.physical_j_begin == 1);
    REQUIRE(v_plan.physical_i_begin == 1);
    REQUIRE(v_plan.physical_j_begin == 0);
    REQUIRE(v_plan.physical_j_end == 7);

    Field3D u = indexed_field(7, 8, 3, 1000.0f);
    Field3D v = indexed_field(7, 8, 3, 2000.0f);
    const Field3D u_before = u;
    const Field3D v_before = v;
    const std::vector<float> u_profile{5.0f, 6.0f, 7.0f};
    const std::vector<float> v_profile{-2.0f, -1.0f, 0.0f};
    const auto report = tmv::domain::execute_homogeneous_field_shift(
        scalar_plan,
        {{"u", &u, &u_profile, tmv::domain::HorizontalFieldLocation::XFace},
         {"v", &v, &v_profile, tmv::domain::HorizontalFieldLocation::YFace}});
    require_shifted(u_before, u, u_plan, u_profile);
    require_shifted(v_before, v, v_plan, v_profile);
    REQUIRE(report.retained_values ==
            u_plan.retained_cells + v_plan.retained_cells);
    REQUIRE(report.initialized_values ==
            u_plan.leading_cells + v_plan.leading_cells);

    std::vector<float> u_trailing(u_plan.trailing_cells, 0.0f);
    std::vector<float> v_trailing(v_plan.trailing_cells, 0.0f);
    const auto export_report = tmv::domain::export_trailing_fields(
        scalar_plan,
        {{"u", &u_before, &u_trailing,
          tmv::domain::HorizontalFieldLocation::XFace},
         {"v", &v_before, &v_trailing,
          tmv::domain::HorizontalFieldLocation::YFace}});
    REQUIRE(export_report.values_exported ==
            u_plan.trailing_cells + v_plan.trailing_cells);

    require_exported(u_before, u_trailing, u_plan);
    require_exported(v_before, v_trailing, v_plan);
}

TEST_CASE("trailing physical ledger uses production mass momentum and energy bases",
          "[core][domain_recycling][trailing][ledger]")
{
    SimulationStorageState storage(
        GridStorageState{6, 7, 3, 10.0, 5.0, 0.1, 20.0});
    storage.grid.geometry.initialize(
        6, 7, 3, 10.0, 5.0, 20.0, CoordinateSystem::Cartesian);
    storage.grid.geometry.staggered = true;

    storage.fields.rho = Field3D(6, 7, 3, 2.0f);
    storage.fields.p = Field3D(6, 7, 3, 100000.0f);
    storage.fields.theta = Field3D(6, 7, 3, 300.0f);
    storage.fields.qv = Field3D(6, 7, 3, 0.010f);
    storage.fields.qc = Field3D(6, 7, 3, 0.002f);
    storage.fields.qr = Field3D(6, 7, 3, 0.003f);
    storage.fields.qi = Field3D(6, 7, 3, 0.004f);
    storage.fields.qs = Field3D(6, 7, 3, 0.005f);
    storage.fields.qg = Field3D(6, 7, 3, 0.007f);
    storage.fields.qh = Field3D(6, 7, 3, 0.006f);
    storage.fields.u = Field3D(6, 7, 3, 5.0f);
    storage.fields.v = Field3D(6, 7, 3, -2.0f);
    storage.fields.w = Field3D(6, 7, 3, 0.0f);
    storage.fields.tracer = Field3D(6, 7, 3, 0.0f);
    storage.fields.tke = Field3D(6, 7, 3, 0.1f);
    storage.runtime.radiation.config.scheme_id = "simple_grey";
    storage.runtime.microphysics.theta_storage_residual =
        Field3D(6, 7, 3, 1.0e-6f);
    auto state = make_simulation_state_view(storage);
    const auto physical = physical_scalar_domain(state.grid);
    const auto plan = tmv::domain::plan_whole_cell_shift_for_physical_domain(
        6, 7, 3, 1, 0,
        physical.i_begin, physical.i_end,
        physical.j_begin, physical.j_end);

    const auto ledger =
        tmv::domain::compute_trailing_physical_export_ledger(state, plan);
    tmv::diagnostics::set_deterministic_reductions(true);
    const auto deterministic_ledger =
        tmv::domain::compute_trailing_physical_export_ledger(state, plan);
    tmv::diagnostics::set_deterministic_reductions(false);
    const double cell_volume = 10.0 * 20.0 * 5.0;
    const double expected_dry_mass = 2.0 * cell_volume * 5.0 * 3.0;
    const double expected_water = expected_dry_mass * 0.037;
    REQUIRE(ledger.physical_columns == 5);
    REQUIRE(ledger.physical_cells == 15);
    REQUIRE(ledger.inventories.dry_mass == Approx(expected_dry_mass));
    REQUIRE(ledger.inventories.total_mass == Approx(expected_dry_mass));
    REQUIRE(ledger.inventories.theta_volume ==
            Approx(expected_dry_mass * 300.0));
    REQUIRE(ledger.inventories.total_water == Approx(expected_water));
    REQUIRE(ledger.inventories.moist_total_mass ==
            Approx(expected_dry_mass + expected_water));
    REQUIRE(ledger.inventories.water_species_mass[0] ==
            Approx(expected_dry_mass * 0.010));
    REQUIRE(ledger.inventories.water_species_mass[1] ==
            Approx(expected_dry_mass * 0.002));
    REQUIRE(ledger.inventories.water_species_mass[2] ==
            Approx(expected_dry_mass * 0.003));
    REQUIRE(ledger.inventories.water_species_mass[3] ==
            Approx(expected_dry_mass * 0.004));
    REQUIRE(ledger.inventories.water_species_mass[4] ==
            Approx(expected_dry_mass * 0.005));
    REQUIRE(ledger.inventories.water_species_mass[5] ==
            Approx(expected_dry_mass * 0.007));
    REQUIRE(ledger.inventories.water_species_mass[6] ==
            Approx(expected_dry_mass * 0.006));
    REQUIRE(ledger.momentum.total_mass == Approx(expected_dry_mass));
    REQUIRE(ledger.momentum.momentum_x ==
            Approx(expected_dry_mass * 5.0));
    REQUIRE(ledger.momentum.momentum_y ==
            Approx(expected_dry_mass * -2.0));
    REQUIRE(ledger.momentum.momentum_z == 0.0);
    REQUIRE(ledger.moist_mixture_energy.total_water_mass ==
            Approx(expected_water));
    REQUIRE(ledger.moist_mixture_energy.subgrid_tke ==
            Approx(expected_dry_mass * 0.1));
    REQUIRE(ledger.moist_mixture_energy.resolved_kinetic ==
            Approx(0.5 * (expected_dry_mass + expected_water) *
                   (5.0 * 5.0 + 2.0 * 2.0)));
    REQUIRE(std::isfinite(ledger.moist_mixture_energy.total));
    REQUIRE(ledger.theta_storage_residual_energy_j == Approx(
        expected_dry_mass *
        physical_constants::specific_heat_cp_jkgk * 1.0e-6));
    REQUIRE(deterministic_ledger.inventories.dry_mass ==
            Approx(ledger.inventories.dry_mass));
    REQUIRE(deterministic_ledger.inventories.theta_volume ==
            Approx(ledger.inventories.theta_volume));
    REQUIRE(deterministic_ledger.inventories.total_water ==
            Approx(ledger.inventories.total_water));
    REQUIRE(deterministic_ledger.momentum.momentum_x ==
            Approx(ledger.momentum.momentum_x));
    REQUIRE(deterministic_ledger.momentum.momentum_y ==
            Approx(ledger.momentum.momentum_y));
    REQUIRE(deterministic_ledger.momentum.angular_momentum_z ==
            Approx(ledger.momentum.angular_momentum_z));
    REQUIRE(deterministic_ledger.moist_mixture_energy.total ==
            Approx(ledger.moist_mixture_energy.total));
    REQUIRE(deterministic_ledger.theta_storage_residual_energy_j ==
            ledger.theta_storage_residual_energy_j);
    REQUIRE(std::string(tmv::domain::trailing_scalar_export_ledger_basis_id()) ==
            "production_conservation_budget_retiring_cells_v1");
    REQUIRE(std::string(tmv::domain::trailing_momentum_export_ledger_basis_id()) ==
            "production_scalar_centered_momentum_retiring_cells_v1");
    REQUIRE(std::string(tmv::domain::trailing_energy_export_ledger_basis_id()) ==
            "physical_moist_mixture_v2_retiring_cells_v1");
    REQUIRE(std::string(
                tmv::domain::trailing_theta_residual_export_ledger_basis_id()) ==
            "deferred_theta_storage_residual_energy_retiring_cells_v1");

    const auto storage_domain_plan =
        tmv::domain::plan_whole_cell_shift(6, 7, 3, 1, 0);
    REQUIRE_THROWS_WITH(
        tmv::domain::compute_trailing_physical_export_ledger(
            state, storage_domain_plan),
        Catch::Matchers::Contains("physical scalar domain"));

    SimulationStorageState empty_storage(
        GridStorageState{6, 7, 3, 10.0, 5.0, 0.1, 20.0});
    empty_storage.grid.geometry.initialize(
        6, 7, 3, 10.0, 5.0, 20.0, CoordinateSystem::Cartesian);
    empty_storage.grid.geometry.staggered = true;
    auto empty_state = make_simulation_state_view(empty_storage);
    REQUIRE_THROWS_WITH(
        tmv::domain::compute_trailing_physical_export_ledger(empty_state, plan),
        Catch::Matchers::Contains("allocated atmospheric state"));

    storage.runtime.microphysics.theta_storage_residual = {};
    REQUIRE_THROWS_WITH(
        tmv::domain::compute_trailing_physical_export_ledger(state, plan),
        Catch::Matchers::Contains("active radiation"));
}

TEST_CASE("D5 retiring and admitted inventories close the fixed-Earth control volume",
          "[core][domain_recycling][exchange][ledger][SCI-228]")
{
    const std::array<std::pair<int, int>, 8> shifts{{
        {1, 0}, {-1, 0}, {0, 1}, {0, -1},
        {1, 1}, {1, -1}, {-1, 1}, {-1, -1},
    }};
    for (const auto [shift_i, shift_j] : shifts)
    {
        CAPTURE(shift_i, shift_j);
        SimulationStorageState storage(
            GridStorageState{7, 8, 4, 125.0, 60.0, 0.2, 175.0});
        storage.grid.geometry.initialize(
            7, 8, 4, 125.0, 60.0, 175.0,
            CoordinateSystem::Cartesian);
        storage.grid.geometry.staggered = true;
        storage.time_s = 37.0;
        storage.configuration.domain_frame.earth_origin_at_t0_m =
            {1500.0, -2300.0};
        storage.configuration.domain_frame.velocity_ms = {11.0, -3.5};

        auto initialize = [&](Field3D& field, float base, float di,
                              float dj, float dk)
        {
            field.resize(7, 8, 4, 0.0f);
            for (int i = 0; i < 7; ++i)
                for (int j = 0; j < 8; ++j)
                    for (int k = 0; k < 4; ++k)
                        field[i][j][k] = base + di * i + dj * j + dk * k;
        };
        initialize(storage.fields.rho, 1.00f, 0.008f, 0.004f, -0.03f);
        initialize(storage.fields.p, 94000.0f, 4.0f, 2.0f, -6500.0f);
        initialize(storage.fields.theta, 296.0f, 0.03f, 0.02f, 2.5f);
        initialize(storage.fields.u, 5.0f, 0.4f, -0.08f, 0.15f);
        initialize(storage.fields.v, -3.0f, 0.06f, 0.3f, -0.1f);
        initialize(storage.fields.w, 0.2f, 0.01f, -0.015f, 0.07f);
        initialize(storage.fields.tracer, 0.0f, 0.0f, 0.0f, 0.0f);
        initialize(storage.fields.qv, 0.010f, 0.00005f, 0.00002f, -0.0008f);
        initialize(storage.fields.qc, 0.0004f, 0.00001f, 0.000005f, 0.0f);
        initialize(storage.fields.qr, 0.0003f, 0.000005f, 0.000004f, 0.0f);
        initialize(storage.fields.qi, 0.0002f, 0.000003f, 0.000002f, 0.0f);
        initialize(storage.fields.qs, 0.0001f, 0.000002f, 0.000001f, 0.0f);
        initialize(storage.fields.qg, 0.00008f, 0.000001f, 0.000001f, 0.0f);
        initialize(storage.fields.qh, 0.00004f, 0.000001f, 0.000001f, 0.0f);
        initialize(storage.fields.tke, 0.25f, 0.005f, 0.003f, 0.01f);
        storage.runtime.radiation.config.scheme_id = "simple_grey";
        initialize(storage.runtime.microphysics.theta_storage_residual,
                   2.0e-6f, 1.0e-8f, -5.0e-9f, 2.0e-8f);

        tmv::domain::HomogeneousPrimaryProfiles profiles;
        for (auto& profile : profiles.values)
            profile.assign(4, 0.0f);
        for (int k = 0; k < 4; ++k)
        {
            const std::size_t level = static_cast<std::size_t>(k);
            profiles[tmv::domain::HomogeneousPrimaryOwner::rho][level] =
                1.05f - 0.04f * k;
            profiles[tmv::domain::HomogeneousPrimaryOwner::p][level] =
                95000.0f - 7000.0f * k;
            profiles[tmv::domain::HomogeneousPrimaryOwner::theta][level] =
                298.0f + 2.0f * k;
            profiles[tmv::domain::HomogeneousPrimaryOwner::qv][level] =
                0.009f - 0.001f * k;
            profiles[tmv::domain::HomogeneousPrimaryOwner::u][level] =
                7.0f + 0.5f * k;
            profiles[tmv::domain::HomogeneousPrimaryOwner::v][level] =
                -1.0f + 0.25f * k;
            profiles[tmv::domain::HomogeneousPrimaryOwner::tke][level] =
                0.15f;
        }

        storage.runtime.microphysics.scheme = std::make_unique<tmv::MilbrandtScheme>();
        storage.runtime.microphysics.scheme->prepare_prognostic_state(storage.fields.rho,
                                                                      storage.fields.qc,
                                                                      storage.fields.qr,
                                                                      storage.fields.qi,
                                                                      storage.fields.qs,
                                                                      storage.fields.qg,
                                                                      storage.fields.qh);
        auto private_fields = storage.runtime.microphysics.scheme->prognostic_scalars_for_transport();
        for (std::size_t f = 0; f < private_fields.size(); ++f)
            initialize(*private_fields[f].field, float(1000 * (f + 1)), 10, 5, 20);
        auto                     state     = make_simulation_state_view(storage);
        const PhysicalCellDomain physical  = physical_scalar_domain(state.grid);
        const auto               plan      = tmv::domain::plan_whole_cell_shift_for_physical_domain(state.grid.nr,
                                                                                                    state.grid.nth,
                                                                                                    state.grid.nz,
                                                                                                    shift_i,
                                                                                                    shift_j,
                                                                                                    physical.i_begin,
                                                                                                    physical.i_end,
                                                                                                    physical.j_begin,
                                                                                                    physical.j_end);
        const auto               pre_frame = storage.configuration.domain_frame;
        const auto               private_leading =
            tmv::domain::prepare_flat_homogeneous_microphysics_recycling(state, plan, profiles);
        REQUIRE_THROWS(
            tmv::domain::compute_domain_recycling_control_volume_exchange_ledger(state, plan, profiles, pre_frame));
        const auto exchange = tmv::domain::compute_domain_recycling_control_volume_exchange_ledger(
            state, plan, profiles, pre_frame, &private_leading);
        REQUIRE(exchange.retiring.physical_cells == plan.trailing_cells);
        REQUIRE(exchange.admitted.physical_cells == plan.trailing_cells);
        REQUIRE(exchange.admitted.theta_storage_residual_energy_j == 0.0);

        auto volume_bindings = tmv::domain::bind_homogeneous_primary_fields(storage, profiles);
        REQUIRE(exchange.pre_shift.private_inventory.count == private_fields.size());
        REQUIRE(exchange.retiring.private_inventory.count == private_fields.size());
        REQUIRE(exchange.admitted.private_inventory.count == private_fields.size());
        REQUIRE(tmv::domain::append_prepared_microphysics_domain_recycling_bindings(
            state, plan, private_leading, volume_bindings));
        auto post_frame = pre_frame;
        tmv::domain::commit_whole_cell_shift_to_domain_frame(post_frame, plan, state.grid.dr, state.grid.dtheta);
        const std::vector<tmv::domain::HomogeneousScalarCommitBinding> frame_bindings{{
            {"domain_frame.origin_x",
             &storage.configuration.domain_frame.earth_origin_at_t0_m.x_m,
             post_frame.earth_origin_at_t0_m.x_m},
            {"domain_frame.origin_y",
             &storage.configuration.domain_frame.earth_origin_at_t0_m.y_m,
             post_frame.earth_origin_at_t0_m.y_m},
        }};
        tmv::domain::execute_homogeneous_window_shift(plan, volume_bindings, {}, frame_bindings);
        auto post_state = make_simulation_state_view(storage);
        REQUIRE_NOTHROW(tmv::domain::verify_domain_recycling_control_volume_exchange_after_commit(
            post_state, plan, storage.configuration.domain_frame, exchange));

        const float private_saved = (*private_fields[0].field)(physical.i_begin, physical.j_begin, 0);
        (*private_fields[0].field)(physical.i_begin, physical.j_begin, 0) += 1000;
        REQUIRE_THROWS_WITH(tmv::domain::verify_domain_recycling_control_volume_exchange_after_commit(
                                post_state, plan, storage.configuration.domain_frame, exchange),
                            Catch::Matchers::Contains("milbrandt.Nc"));
        (*private_fields[0].field)(physical.i_begin, physical.j_begin, 0) = private_saved;
        const float saved = storage.fields.rho[physical.i_begin][physical.j_begin][physical.k_begin];
        storage.fields.rho[physical.i_begin][physical.j_begin][physical.k_begin] = saved * 1.01f;
        REQUIRE_THROWS_WITH(tmv::domain::verify_domain_recycling_control_volume_exchange_after_commit(
                                post_state, plan, storage.configuration.domain_frame, exchange),
                            Catch::Matchers::Contains("closure"));
        storage.fields.rho[physical.i_begin][physical.j_begin][physical.k_begin] = saved;
    }
    REQUIRE(std::string(
        tmv::domain::domain_recycling_control_volume_exchange_basis_id()) ==
        "instantaneous_fixed_earth_control_volume_exchange_v1");
}

TEST_CASE("D5 root owner contract is explicit and duplicate-free",
          "[core][domain_recycling][owners][D5]")
{
    using Action = tmv::domain::DomainRecyclingOwnerAction;
    using Ready = tmv::domain::DomainRecyclingOwnerReadiness;
    const auto& owners = tmv::domain::domain_recycling_owner_contracts();

    REQUIRE(owners.size() ==
            tmv::domain::domain_recycling_owner_contract_count());
    std::size_t ready = 0;
    std::size_t partial = 0;
    std::size_t required = 0;
    std::size_t unavailable = 0;
    for (std::size_t i = 0; i < owners.size(); ++i)
    {
        CAPTURE(i, owners[i].id);
        REQUIRE_FALSE(owners[i].id.empty());
        REQUIRE_FALSE(owners[i].scope.empty());
        for (std::size_t j = i + 1; j < owners.size(); ++j)
            REQUIRE(owners[i].id != owners[j].id);
        switch (owners[i].readiness)
        {
            case Ready::ReadyForTransaction: ++ready; break;
            case Ready::Partial: ++partial; break;
            case Ready::Required: ++required; break;
            case Ready::ExplicitlyUnavailable: ++unavailable; break;
        }
    }

    REQUIRE(ready == 28);
    REQUIRE(partial == 0);
    REQUIRE(required == 0);
    REQUIRE(unavailable == 0);
    REQUIRE(tmv::domain::remaining_domain_recycling_owner_contracts() == 0);

    const auto find_owner = [&](std::string_view id)
        -> const tmv::domain::DomainRecyclingOwnerContract&
    {
        for (const auto& owner : owners)
            if (owner.id == id)
                return owner;
        FAIL("missing D5 root owner " << id);
        return owners.front();
    };
    REQUIRE(find_owner("primary_atmosphere").action ==
            Action::ShiftWithLeadingState);
    REQUIRE(find_owner("retiring_physical_ledgers").readiness ==
            Ready::ReadyForTransaction);
    REQUIRE(find_owner("boundary_closure_and_kw78_history").readiness ==
            Ready::ReadyForTransaction);
    REQUIRE(find_owner("radiation_runtime").action ==
            Action::ShiftPersistentInvalidateTransient);
    REQUIRE(find_owner("microphysics_runtime").readiness ==
            Ready::ReadyForTransaction);
    REQUIRE(find_owner("turbulence_runtime").readiness ==
            Ready::ReadyForTransaction);
    REQUIRE(find_owner("chaos_runtime").readiness ==
            Ready::ReadyForTransaction);
    REQUIRE(find_owner("rayleigh_runtime").readiness ==
            Ready::ReadyForTransaction);
    REQUIRE(find_owner("rayleigh_runtime").action == Action::Preserve);
    REQUIRE(find_owner("advection_runtime").readiness ==
            Ready::ReadyForTransaction);
    REQUIRE(find_owner("diffusion_runtime").readiness ==
            Ready::ReadyForTransaction);
    REQUIRE(find_owner("time_stepping_runtime").readiness ==
            Ready::ReadyForTransaction);
    REQUIRE(find_owner("radar_runtime").readiness ==
            Ready::ReadyForTransaction);
    REQUIRE(find_owner("derived_diagnostic_fields").readiness ==
            Ready::ReadyForTransaction);
    REQUIRE(find_owner("dynamics_operator").readiness ==
            Ready::ReadyForTransaction);
    REQUIRE(find_owner("storm_tracking_and_partition").readiness ==
            Ready::ReadyForTransaction);
    REQUIRE(find_owner("lagrangian_trajectories").action ==
            Action::TranslateEarthCoordinates);
    REQUIRE(find_owner("durable_global_swath_output").readiness ==
            Ready::ReadyForTransaction);
    REQUIRE(find_owner("checkpoint_transaction_provenance").readiness ==
            Ready::ReadyForTransaction);
    REQUIRE(find_owner("surface_accumulated_rainfall").readiness ==
            Ready::ReadyForTransaction);
    REQUIRE(find_owner("compute_residency").action ==
            Action::InvalidateDeviceResidency);
    REQUIRE(find_owner("terrain_runtime").readiness ==
            Ready::ReadyForTransaction);
    REQUIRE(find_owner("checkpoint_transaction_provenance").action ==
            Action::TranslateEarthCoordinates);
}

TEST_CASE("D5 whole-cell frame commit preserves retained Earth coordinates",
          "[core][domain_recycling][frame][D5]")
{
    const auto plan = tmv::domain::plan_whole_cell_shift(8, 9, 3, 2, -1);
    tmv::domain::HorizontalDomainFrame frame;
    frame.earth_origin_at_t0_m = {1000.0, -500.0};
    frame.velocity_ms = {15.0, -4.0};
    const auto before = frame;

    const tmv::domain::HorizontalPoint destination_grid_m{300.0, 700.0};
    const tmv::domain::HorizontalPoint source_grid_m{
        destination_grid_m.x_m + 2.0 * 100.0,
        destination_grid_m.y_m - 1.0 * 250.0,
    };
    const auto expected_earth =
        before.grid_to_earth(source_grid_m, 12.0);

    const auto report =
        tmv::domain::commit_whole_cell_shift_to_domain_frame(
            frame, plan, 100.0, 250.0);
    const auto committed_earth =
        frame.grid_to_earth(destination_grid_m, 12.0);

    REQUIRE(report.shift_i_cells == 2);
    REQUIRE(report.shift_j_cells == -1);
    REQUIRE(report.displacement_m.x_m == 200.0);
    REQUIRE(report.displacement_m.y_m == -250.0);
    REQUIRE(report.previous_earth_origin_at_t0_m.x_m == 1000.0);
    REQUIRE(report.previous_earth_origin_at_t0_m.y_m == -500.0);
    REQUIRE(report.committed_earth_origin_at_t0_m.x_m == 1200.0);
    REQUIRE(report.committed_earth_origin_at_t0_m.y_m == -750.0);
    REQUIRE(committed_earth.x_m == Approx(expected_earth.x_m));
    REQUIRE(committed_earth.y_m == Approx(expected_earth.y_m));
    REQUIRE(frame.velocity_ms.u_ms == before.velocity_ms.u_ms);
    REQUIRE(frame.velocity_ms.v_ms == before.velocity_ms.v_ms);
    REQUIRE(std::string(tmv::domain::domain_frame_shift_algorithm_id()) ==
            "integer_whole_cell_earth_origin_commit_v1");
}

TEST_CASE("D5 checkpoint provenance commits atomically with the Earth frame",
          "[core][domain_recycling][provenance][transaction][SCI-227]")
{
    SimulationStorageState storage(
        GridStorageState(8, 9, 3, 100.0, 50.0, 0.1, 250.0));
    storage.time_s = 20.0;
    storage.configuration.domain_frame.earth_origin_at_t0_m =
        {1000.0, -500.0};
    storage.configuration.domain_frame.velocity_ms = {15.0, -4.0};
    auto state = make_simulation_state_view(storage);
    const auto plan = tmv::domain::plan_whole_cell_shift(8, 9, 3, 2, -1);
    tmv::domain::DomainRecyclingControlVolumeExchangeLedger ledger;
    ledger.retiring.physical_cells = plan.trailing_cells;
    ledger.retiring.physical_columns = plan.trailing_cells
        / static_cast<std::size_t>(plan.nz);
    ledger.admitted.physical_cells = ledger.retiring.physical_cells;
    ledger.admitted.physical_columns = ledger.retiring.physical_columns;

    auto prepared =
        tmv::domain::prepare_domain_recycling_transaction_provenance(
            state, plan, storage.configuration.domain_frame, ledger,
            "recycle_000001", "run_generation_1", "npy_3d",
            "canonical_rectangular_earth_swath_capture_v1", 20.0,
            1, 0, 8192, UINT64_C(0xfedcba9876543210));
    REQUIRE(prepared.active);
    REQUIRE(prepared.committed.sequence == 1);
    REQUIRE(prepared.committed.pre_shift_origin_t0_m.x_m == 1000.0);
    REQUIRE(prepared.committed.post_shift_origin_t0_m.x_m == 1200.0);
    REQUIRE(prepared.committed.post_shift_origin_t0_m.y_m == -750.0);

    std::vector<tmv::domain::HomogeneousRuntimeCommitBinding> commits;
    REQUIRE(tmv::domain::append_domain_recycling_transaction_provenance_binding(
        state, prepared, commits));
    std::vector<tmv::domain::HomogeneousScalarCommitBinding> frame_scalars{
        {"domain_frame.origin_x", &storage.configuration.domain_frame
                                      .earth_origin_at_t0_m.x_m,
         prepared.committed.post_shift_origin_t0_m.x_m},
        {"domain_frame.origin_y", &storage.configuration.domain_frame
                                      .earth_origin_at_t0_m.y_m,
         prepared.committed.post_shift_origin_t0_m.y_m},
    };

    Field3D malformed(7, 9, 3, 1.0f);
    std::vector<float> leading(3, 0.0f);
    const std::vector<tmv::domain::HomogeneousFieldShiftBinding> bad_fields{
        {"late_failure", &malformed, &leading},
    };
    REQUIRE_THROWS(tmv::domain::execute_homogeneous_window_shift(
        plan, bad_fields, {}, frame_scalars, {}, {}, {}, commits));
    REQUIRE_FALSE(storage.runtime.domain_recycling_provenance.available);
    REQUIRE(storage.configuration.domain_frame.earth_origin_at_t0_m.x_m
            == 1000.0);
    REQUIRE(storage.configuration.domain_frame.earth_origin_at_t0_m.y_m
            == -500.0);

    const auto report = tmv::domain::execute_homogeneous_window_shift(
        plan, {}, {}, frame_scalars, {}, {}, {}, commits);
    REQUIRE(report.scalars_committed == 2);
    REQUIRE(report.runtime_commits_committed == 1);
    const auto& committed = storage.runtime.domain_recycling_provenance;
    REQUIRE(tmv::domain::valid_domain_recycling_transaction_provenance(
        committed));
    REQUIRE(committed.transaction_id.view() == "recycle_000001");
    REQUIRE(committed.event_manifest_bytes == 8192);
    REQUIRE(committed.event_manifest_fnv1a64
            == UINT64_C(0xfedcba9876543210));
    REQUIRE(committed.previous_chain_fnv1a64 == 0);
    REQUIRE(committed.committed_chain_fnv1a64
            == tmv::domain::compute_domain_recycling_transaction_chain_fnv1a64(
                   0, 1, 8192, UINT64_C(0xfedcba9876543210),
                   "recycle_000001", "run_generation_1"));
    REQUIRE(storage.configuration.domain_frame.earth_origin_at_t0_m.x_m
            == committed.post_shift_origin_t0_m.x_m);
    REQUIRE(storage.configuration.domain_frame.earth_origin_at_t0_m.y_m
            == committed.post_shift_origin_t0_m.y_m);

    REQUIRE_THROWS_WITH(
        tmv::domain::prepare_domain_recycling_transaction_provenance(
            state, plan, storage.configuration.domain_frame, ledger,
            "recycle_000001", "run_generation_1", "npy_3d",
            "canonical_rectangular_earth_swath_capture_v1", 20.0,
            2, committed.committed_chain_fnv1a64, 8192,
            UINT64_C(0xfedcba9876543210)),
        Catch::Matchers::Contains("must not be reused"));

    auto next = tmv::domain::prepare_domain_recycling_transaction_provenance(
        state, plan, storage.configuration.domain_frame, ledger,
        "recycle_000002", "run_generation_1", "netcdf",
        "canonical_rectangular_earth_swath_capture_v1", 20.0, 2,
        committed.committed_chain_fnv1a64, 9216,
        UINT64_C(0x1020304050607080));
    REQUIRE(next.committed.sequence == 2);
    REQUIRE(next.committed.previous_chain_fnv1a64
            == committed.committed_chain_fnv1a64);
    REQUIRE(next.committed.committed_chain_fnv1a64
            != committed.committed_chain_fnv1a64);

    REQUIRE_THROWS_WITH(
        tmv::domain::prepare_domain_recycling_transaction_provenance(
            state, plan, storage.configuration.domain_frame, ledger,
            "recycle_000002", "run_generation_1", "netcdf",
            "canonical_rectangular_earth_swath_capture_v1", 20.0, 2,
            committed.committed_chain_fnv1a64 ^ 1U, 9216,
            UINT64_C(0x1020304050607080)),
        Catch::Matchers::Contains("does not extend live history"));
}

TEST_CASE("D5 whole-cell frame commit rejects atomically",
          "[core][domain_recycling][frame][transaction][D5]")
{
    const auto plan = tmv::domain::plan_whole_cell_shift(8, 9, 3, 2, -1);
    tmv::domain::HorizontalDomainFrame frame;
    frame.earth_origin_at_t0_m = {1000.0, -500.0};
    frame.velocity_ms = {15.0, -4.0};

    REQUIRE_THROWS_WITH(
        tmv::domain::commit_whole_cell_shift_to_domain_frame(
            frame, plan, -100.0, 250.0),
        Catch::Matchers::Contains("positive horizontal spacing"));
    REQUIRE(frame.earth_origin_at_t0_m.x_m == 1000.0);
    REQUIRE(frame.earth_origin_at_t0_m.y_m == -500.0);

    auto malformed = plan;
    ++malformed.source_i_begin;
    REQUIRE_THROWS_WITH(
        tmv::domain::commit_whole_cell_shift_to_domain_frame(
            frame, malformed, 100.0, 250.0),
        Catch::Matchers::Contains("canonical whole-cell plan"));
    REQUIRE(frame.earth_origin_at_t0_m.x_m == 1000.0);
    REQUIRE(frame.earth_origin_at_t0_m.y_m == -500.0);

    REQUIRE_THROWS_WITH(
        tmv::domain::commit_whole_cell_shift_to_domain_frame(
            frame, plan, std::numeric_limits<double>::max(), 250.0),
        Catch::Matchers::Contains("not finite"));
    REQUIRE(frame.earth_origin_at_t0_m.x_m == 1000.0);
    REQUIRE(frame.earth_origin_at_t0_m.y_m == -500.0);
    REQUIRE(frame.velocity_ms.u_ms == 15.0);
    REQUIRE(frame.velocity_ms.v_ms == -4.0);
}

TEST_CASE("D5 microphysics compensation history shifts and exports atomically",
          "[core][domain_recycling][microphysics][D5]")
{
    SimulationStorageState storage(
        GridStorageState{6, 7, 3, 100.0, 50.0, 0.1, 200.0});
    storage.grid.geometry.initialize(
        6, 7, 3, 100.0, 50.0, 200.0,
        CoordinateSystem::Cartesian);
    storage.runtime.radiation.config.scheme_id = "simple_grey";
    storage.fields.theta = indexed_field(6, 7, 3, 300.0f);
    storage.runtime.microphysics.theta_storage_residual =
        indexed_field(6, 7, 3, -1000.0f);
    SimulationStateView state = make_simulation_state_view(storage);

    const auto plan = tmv::domain::plan_whole_cell_shift(
        6, 7, 3, 1, -1);
    const Field3D theta_before = storage.fields.theta;
    const Field3D residual_before =
        storage.runtime.microphysics.theta_storage_residual;
    const float* residual_pointer =
        storage.runtime.microphysics.theta_storage_residual.data();
    const std::vector<float> theta_leading{300.0f, 301.0f, 302.0f};
    const std::vector<float> zero_leading(3, 0.0f);

    std::vector<float> residual_trailing(plan.trailing_cells, 0.0f);
    std::vector<tmv::domain::TrailingFieldExportBinding> exports;
    exports.reserve(1);
    REQUIRE(tmv::domain::append_microphysics_trailing_export_binding(
        state, residual_trailing, exports));
    const auto export_report =
        tmv::domain::export_trailing_fields(plan, exports);

    std::vector<tmv::domain::HomogeneousFieldShiftBinding> bindings;
    bindings.reserve(2);
    bindings.push_back({
        "theta", &storage.fields.theta, &theta_leading,
        tmv::domain::HorizontalFieldLocation::ScalarCenter,
    });
    REQUIRE(tmv::domain::append_microphysics_domain_recycling_binding(
        state, zero_leading, bindings));
    const auto shift_report =
        tmv::domain::execute_homogeneous_field_shift(plan, bindings);

    REQUIRE(export_report.fields_exported == 1);
    REQUIRE(export_report.values_exported == plan.trailing_cells);
    REQUIRE(shift_report.fields_shifted == 2);
    REQUIRE(storage.runtime.microphysics.theta_storage_residual.data() ==
            residual_pointer);
    require_exported(residual_before, residual_trailing, plan);
    require_shifted(
        theta_before, storage.fields.theta, plan, theta_leading);
    require_shifted(
        residual_before,
        storage.runtime.microphysics.theta_storage_residual,
        plan, zero_leading);
}

TEST_CASE("D5 microphysics compensation binding rejects fabricated history",
          "[core][domain_recycling][microphysics][transaction][D5]")
{
    SimulationStorageState storage(
        GridStorageState{6, 7, 3, 100.0, 50.0, 0.1, 200.0});
    storage.grid.geometry.initialize(
        6, 7, 3, 100.0, 50.0, 200.0,
        CoordinateSystem::Cartesian);
    storage.runtime.radiation.config.scheme_id = "simple_grey";
    SimulationStateView state = make_simulation_state_view(storage);
    std::vector<tmv::domain::HomogeneousFieldShiftBinding> bindings;
    const std::vector<float> zero_leading(3, 0.0f);

    REQUIRE_THROWS_WITH(
        tmv::domain::append_microphysics_domain_recycling_binding(
            state, zero_leading, bindings),
        Catch::Matchers::Contains("requires allocated"));
    REQUIRE(bindings.empty());

    storage.runtime.radiation.config.scheme_id = "none";
    REQUIRE_FALSE(tmv::domain::append_microphysics_domain_recycling_binding(
        state, zero_leading, bindings));
    REQUIRE(bindings.empty());

    storage.runtime.microphysics.theta_storage_residual =
        Field3D(6, 7, 3, 0.0f);
    const std::vector<float> fabricated{0.0f, 1.0e-8f, 0.0f};
    REQUIRE_THROWS_WITH(
        tmv::domain::append_microphysics_domain_recycling_binding(
            state, fabricated, bindings),
        Catch::Matchers::Contains("exact finite zero"));
    REQUIRE(bindings.empty());
}

TEST_CASE("D5 shifts Milbrandt moments and exact precipitation history atomically",
          "[core][domain_recycling][microphysics][precipitation][transaction][D5]")
{
    SimulationStorageState storage =
        radiation_recycling_storage(12.0 * 3600.0, true, true);
    storage.time_s = 100.0;
    storage.configuration.microphysics_scheme = "milbrandt";
    storage.runtime.microphysics.scheme =
        std::make_unique<tmv::MilbrandtScheme>();
    storage.runtime.microphysics.theta_storage_residual =
        indexed_field(6, 7, 4, 0.001f);
    storage.runtime.microphysics.scheme->prepare_prognostic_state(
        storage.fields.rho, storage.fields.qc, storage.fields.qr,
        storage.fields.qi, storage.fields.qs, storage.fields.qg,
        storage.fields.qh);
    auto live = storage.runtime.microphysics.scheme
        ->prognostic_scalars_for_transport();
    REQUIRE(live.size() == 6);
    for (std::size_t scalar = 0; scalar < live.size(); ++scalar)
        *live[scalar].field = indexed_field(
            6, 7, 4, static_cast<float>(1000 * (scalar + 1)));

    const std::size_t surface_size = 6U * 7U;
    storage.surface.precipitation_rate_mm_h.resize(surface_size);
    storage.surface.accumulated_rainfall_mm.resize(surface_size);
    storage.surface.accumulated_rainfall_residual_mm.resize(surface_size);
    for (std::size_t index = 0; index < surface_size; ++index)
    {
        storage.surface.precipitation_rate_mm_h[index] =
            0.1f + static_cast<float>(index);
        storage.surface.accumulated_rainfall_mm[index] =
            10.0f + static_cast<float>(index);
        storage.surface.accumulated_rainfall_residual_mm[index] =
            1.0e-7f * static_cast<float>(index);
    }
    storage.surface.accumulated_rainfall_last_update_s = storage.time_s;

    SimulationStateView state = make_simulation_state_view(storage);
    const auto physical = physical_scalar_domain(state.grid);
    const auto plan =
        tmv::domain::plan_whole_cell_shift_for_physical_domain(
            state.grid.nr, state.grid.nth, state.grid.nz, 1, -1,
            physical.i_begin, physical.i_end,
            physical.j_begin, physical.j_end);
    const auto primary =
        tmv::domain::prepare_flat_homogeneous_primary_profiles(
            storage, std::vector<double>(4, 0.1));
    const auto prepared =
        tmv::domain::prepare_flat_homogeneous_microphysics_recycling(
            state, plan, primary);
    REQUIRE(prepared.active);
    REQUIRE(prepared.private_scalar_names.size() == 6);
    REQUIRE(prepared.leading_private_scalars.size() == 6);
    REQUIRE(std::string(
                tmv::domain::microphysics_domain_recycling_algorithm_id()) ==
            "persistent_private_moments_flat_or_mapped_environment_shift_v2");

    const Field3D residual_before =
        storage.runtime.microphysics.theta_storage_residual;
    std::vector<Field3D> moments_before;
    moments_before.reserve(live.size());
    for (const auto& scalar : live) moments_before.push_back(*scalar.field);
    const auto rate_before = storage.surface.precipitation_rate_mm_h;
    const auto rainfall_before = storage.surface.accumulated_rainfall_mm;
    const auto rainfall_residual_before =
        storage.surface.accumulated_rainfall_residual_mm;
    const float* residual_pointer =
        storage.runtime.microphysics.theta_storage_residual.data();
    std::vector<const float*> moment_pointers;
    for (const auto& scalar : live)
        moment_pointers.push_back(scalar.field->data());

    Field3D* const transaction_tendencies[] = {
        &storage.runtime.microphysics.tendencies.dtheta_dt,
        &storage.runtime.microphysics.tendencies.dqv_dt,
        &storage.runtime.microphysics.tendencies.dqc_dt,
        &storage.runtime.microphysics.tendencies.dqr_dt,
        &storage.runtime.microphysics.tendencies.dqi_dt,
        &storage.runtime.microphysics.tendencies.dqs_dt,
        &storage.runtime.microphysics.tendencies.dqg_dt,
        &storage.runtime.microphysics.tendencies.dqh_dt};
    std::array<Field3D, 8> transaction_tendencies_before;
    for (std::size_t index = 0; index < 8; ++index)
    {
        *transaction_tendencies[index] =
            indexed_field(6, 7, 4, 50.0f + static_cast<float>(index));
        transaction_tendencies_before[index] =
            *transaction_tendencies[index];
    }

    std::vector<std::vector<float>> micro_trailing(7);
    for (auto& values : micro_trailing)
        values.assign(plan.trailing_cells, 0.0f);
    std::vector<tmv::domain::TrailingFieldExportBinding> volume_exports;
    REQUIRE(
        tmv::domain::append_prepared_microphysics_trailing_export_bindings(
            state, micro_trailing, volume_exports));
    const auto volume_export_report =
        tmv::domain::export_trailing_fields(plan, volume_exports);

    const std::size_t trailing_columns =
        plan.trailing_cells / static_cast<std::size_t>(plan.nz);
    std::vector<float> rate_trailing(trailing_columns, 0.0f);
    std::vector<float> rainfall_trailing(trailing_columns, 0.0f);
    std::vector<float> rainfall_residual_trailing(
        trailing_columns, 0.0f);
    std::vector<tmv::domain::TrailingSurfaceExportBinding> surface_exports;
    REQUIRE(
        tmv::domain::append_surface_precipitation_trailing_export_bindings(
            state, rate_trailing, rainfall_trailing,
            rainfall_residual_trailing, surface_exports));
    const auto surface_export_report =
        tmv::domain::export_trailing_surfaces(plan, surface_exports);

    std::vector<tmv::domain::HomogeneousFieldShiftBinding> volumes;
    std::vector<tmv::domain::HomogeneousSurfaceShiftBinding> surfaces;
    REQUIRE(
        tmv::domain::append_prepared_microphysics_domain_recycling_bindings(
            state, plan, prepared, volumes));
    REQUIRE(
        tmv::domain::append_surface_precipitation_domain_recycling_bindings(
            state, surfaces));
    std::vector<tmv::domain::HomogeneousRuntimeInvalidationBinding>
        invalidations;
    REQUIRE(tmv::domain::
                append_microphysics_domain_recycling_invalidation_binding(
                    state, invalidations));
    double rollback_probe = 1.0;
    REQUIRE_THROWS_WITH(
        tmv::domain::execute_homogeneous_window_shift(
            plan, volumes, surfaces,
            {{"rollback_probe", &rollback_probe,
              std::numeric_limits<double>::quiet_NaN()}}, {}, {},
            invalidations),
        Catch::Matchers::Contains("non-finite"));
    REQUIRE(rollback_probe == 1.0);
    for (std::size_t index = 0; index < 8; ++index)
        for (std::size_t value = 0;
             value < transaction_tendencies[index]->size(); ++value)
            REQUIRE(transaction_tendencies[index]->data()[value] ==
                    transaction_tendencies_before[index].data()[value]);
    for (std::size_t value = 0; value < residual_before.size(); ++value)
        REQUIRE(storage.runtime.microphysics.theta_storage_residual.data()[value]
                == residual_before.data()[value]);
    const auto report = tmv::domain::execute_homogeneous_window_shift(
        plan, volumes, surfaces, {}, {}, {}, invalidations);

    REQUIRE(volume_export_report.fields_exported == 7);
    REQUIRE(surface_export_report.fields_exported == 3);
    REQUIRE(report.volumes.fields_shifted == 7);
    REQUIRE(report.surfaces_shifted == 3);
    REQUIRE(report.runtime_invalidations_committed == 1);
    for (Field3D* tendency : transaction_tendencies)
        for (std::size_t value = 0; value < tendency->size(); ++value)
            REQUIRE(tendency->data()[value] == 0.0f);
    REQUIRE(storage.runtime.microphysics.theta_storage_residual.data() ==
            residual_pointer);
    live = storage.runtime.microphysics.scheme
        ->prognostic_scalars_for_transport();
    for (std::size_t scalar = 0; scalar < live.size(); ++scalar)
    {
        REQUIRE(live[scalar].field->data() == moment_pointers[scalar]);
        require_shifted(
            moments_before[scalar], *live[scalar].field, plan,
            prepared.leading_private_scalars[scalar]);
        require_exported(
            moments_before[scalar], micro_trailing[scalar + 1], plan);
    }
    require_shifted(
        residual_before,
        storage.runtime.microphysics.theta_storage_residual,
        plan, prepared.zero_leading_profile);
    require_exported(residual_before, micro_trailing[0], plan);
    require_surface_shifted(
        rate_before, storage.surface.precipitation_rate_mm_h, plan, 0.0f);
    require_surface_shifted(
        rainfall_before, storage.surface.accumulated_rainfall_mm, plan, 0.0f);
    require_surface_shifted(
        rainfall_residual_before,
        storage.surface.accumulated_rainfall_residual_mm, plan, 0.0f);
    require_surface_exported(rate_before, rate_trailing, plan);
    require_surface_exported(rainfall_before, rainfall_trailing, plan);
    require_surface_exported(
        rainfall_residual_before, rainfall_residual_trailing, plan);
    REQUIRE(storage.surface.accumulated_rainfall_last_update_s == 100.0);
    REQUIRE(std::string(
                tmv::domain::surface_precipitation_domain_recycling_algorithm_id()) ==
            "exact_fallout_surface_history_shift_v1");

    Field3D* const runtime_tendencies[] = {
        &storage.runtime.microphysics.tendencies.dtheta_dt,
        &storage.runtime.microphysics.tendencies.dqv_dt,
        &storage.runtime.microphysics.tendencies.dqc_dt,
        &storage.runtime.microphysics.tendencies.dqr_dt,
        &storage.runtime.microphysics.tendencies.dqi_dt,
        &storage.runtime.microphysics.tendencies.dqs_dt,
        &storage.runtime.microphysics.tendencies.dqg_dt,
        &storage.runtime.microphysics.tendencies.dqh_dt};
    std::vector<const float*> tendency_pointers;
    for (Field3D* tendency : runtime_tendencies)
    {
        *tendency = indexed_field(6, 7, 4, 5.0f);
        tendency_pointers.push_back(tendency->data());
    }
    PortableSchemeState portable =
        storage.runtime.microphysics.scheme->portable_state();
    REQUIRE(portable.fields.size() == 13);
    for (std::size_t index = 6; index < 12; ++index)
        portable.fields[index].resize(6, 7, 4, 3.0f);
    portable.fields[12].resize(6, 7, 1, 3.0f);
    for (std::size_t index = 6; index < portable.fields.size(); ++index)
        portable.fields[index].fill(3.0f);
    std::string restore_error;
    REQUIRE(storage.runtime.microphysics.scheme->restore_portable_state(
        portable, restore_error));
    const auto moments_before_invalidation =
        storage.runtime.microphysics.scheme->portable_state().fields;
    tmv::invalidate_microphysics_transients_after_domain_recycling(state);
    for (std::size_t index = 0; index < 8; ++index)
    {
        REQUIRE(runtime_tendencies[index]->data() == tendency_pointers[index]);
        for (std::size_t value = 0;
             value < runtime_tendencies[index]->size(); ++value)
            REQUIRE(runtime_tendencies[index]->data()[value] == 0.0f);
    }
    const PortableSchemeState invalidated =
        storage.runtime.microphysics.scheme->portable_state();
    for (std::size_t index = 0; index < 6; ++index)
    {
        REQUIRE(invalidated.fields[index].size() ==
                moments_before_invalidation[index].size());
        for (std::size_t value = 0;
             value < invalidated.fields[index].size(); ++value)
            REQUIRE(invalidated.fields[index].data()[value] ==
                    moments_before_invalidation[index].data()[value]);
    }
    for (std::size_t index = 6; index < invalidated.fields.size(); ++index)
        for (std::size_t value = 0;
             value < invalidated.fields[index].size(); ++value)
            REQUIRE(invalidated.fields[index].data()[value] == 0.0f);
    REQUIRE(std::string(
                tmv::microphysics_recycling_invalidation_algorithm_id()) ==
            "preserve_prognostics_zero_step_products_v1");
}

TEST_CASE("Earth-fixed D5 prepares full-field private microphysics currencies",
          "[core][domain_recycling][terrain][microphysics][transaction][D5]")
{
    const std::string family = GENERATE("thompson", "milbrandt", "milbrandt_no_hail");
    CAPTURE(family);
    SimulationStorageState storage =
        radiation_recycling_storage(12.0 * 3600.0, true, false);
    const bool thompson = family == "thompson";
    storage.fields.rho.resize(6, 7, 4, 1.0f);
    for (auto* field : {&storage.fields.qc, &storage.fields.qr, &storage.fields.qi,
                        &storage.fields.qs, &storage.fields.qg, &storage.fields.qh})
    {
        field->resize(6, 7, 4, 0.0f);
    }
    storage.configuration.microphysics_scheme = thompson ? "thompson" : "milbrandt";
    if (thompson)
    {
        storage.runtime.microphysics.scheme = std::make_unique<tmv::ThompsonScheme>(.001, .001, .001, 450e6, 3e6);
    }
    else
    {
        storage.runtime.microphysics.scheme = std::make_unique<tmv::MilbrandtScheme>();
        tmv::MicrophysicsConfig config;
        config.milbrandt.hail_processes = family == "milbrandt";
        storage.runtime.microphysics.scheme->configure(config);
    }
    storage.runtime.microphysics.theta_storage_residual =
        indexed_field(6, 7, 4, 0.001f);
    storage.runtime.microphysics.scheme->prepare_prognostic_state(
        storage.fields.rho, storage.fields.qc, storage.fields.qr,
        storage.fields.qi, storage.fields.qs, storage.fields.qg,
        storage.fields.qh);
    auto live = storage.runtime.microphysics.scheme
        ->prognostic_scalars_for_transport();
    REQUIRE(live.size() == (thompson ? 5 : 6));
    std::vector<Field3D> before;
    for (std::size_t scalar = 0; scalar < live.size(); ++scalar)
    {
        *live[scalar].field = indexed_field(
            6, 7, 4, static_cast<float>(1000 * (scalar + 1)));
        before.push_back(*live[scalar].field);
    }
    const auto fresh = storage.runtime.microphysics.scheme->clone_parameters();
    REQUIRE(fresh);
    REQUIRE(fresh->parameter_identity() == storage.runtime.microphysics.scheme->parameter_identity());
    for (const auto& scalar : fresh->prognostic_scalars_for_transport())
    {
        REQUIRE(scalar.field->empty());
    }

    TerrainConfig config;
    config.scheme_id = "bell";
    config.coord_id = "smoothed";
    config.horizontal_placement_id = "earth_fixed";
    config.ztop = 800.0;
    config.bell.h0 = 120.0;
    config.bell.a = 350.0;
    config.bell.axisymmetric = true;
    config.feature_center_earth_x_m = 350.0;
    config.feature_center_earth_y_m = 625.0;
    storage.runtime.terrain.config = config;
    tmv::EarthFixedTerrainBuildRequest request;
    request.nr = storage.grid.nr;
    request.nth = storage.grid.nth;
    request.nz = storage.grid.nz;
    request.dx_m = storage.grid.dx;
    request.dy_m = storage.grid.dy;
    request.feature_center_earth_m = {350.0, 625.0};
    const auto geometry = tmv::build_earth_fixed_terrain_geometry(
        config, request);
    const auto primary = tmv::domain::prepare_terrain_aware_primary_fields(
        storage, geometry.metrics, std::vector<double>(4, 0.1));
    SimulationStateView state = make_simulation_state_view(storage);
    const auto physical = physical_scalar_domain(state.grid);
    const auto plan =
        tmv::domain::plan_whole_cell_shift_for_physical_domain(
            state.grid.nr, state.grid.nth, state.grid.nz, 1, -1,
            physical.i_begin, physical.i_end,
            physical.j_begin, physical.j_end);
    auto prepared =
        tmv::domain::prepare_terrain_aware_microphysics_recycling(
            state, plan, primary, geometry.metrics);
    REQUIRE(prepared.active);
    REQUIRE(prepared.terrain_aware);
    auto invalid_metrics = geometry.metrics;
    invalid_metrics.z_face(0, 0, 1) = invalid_metrics.z_face(0, 0, 0);
    REQUIRE_THROWS(tmv::domain::prepare_terrain_aware_microphysics_recycling(state, plan, primary, invalid_metrics));
    REQUIRE(prepared.terrain_leading_private_scalars.size() == live.size());
    for (const Field3D& field : prepared.terrain_leading_private_scalars)
        REQUIRE(field.size() == static_cast<std::size_t>(6 * 7 * 4));
    for (std::size_t f = 0; f < live.size(); ++f)
    {
        for (int i = 0; i < 6; ++i)
        {
            for (int j = 0; j < 7; ++j)
            {
                for (int k = 0; k < 4; ++k)
                {
                    CHECK((*live[f].field)(i, j, k) == before[f](i, j, k));
                    double expected = 0;
                    if (thompson && f >= 3)
                    {
                        const auto& z = geometry.metrics.z_face;
                        const int sampled = std::max(1, k);
                        const double height = .5 * (z(i, j, sampled) + z(i, j, sampled + 1) - z(i, j, 0) - z(i, j, 1));
                        const double surface = f == 3 ? 450e6 : 3e6;
                        const double background = f == 3 ? 50e6 : .5e6;
                        expected = (background + (surface - background) *
                            std::exp(std::log(background / (surface - background)) * height / 800.0)) /
                            primary[tmv::domain::HomogeneousPrimaryOwner::rho](i, j, k);
                    }
                    CHECK(prepared.terrain_leading_private_scalars[f](i, j, k) == Approx(expected).epsilon(2e-7));
                }
            }
        }
    }

    std::vector<tmv::domain::HomogeneousFieldShiftBinding> bindings;
    REQUIRE(tmv::domain::append_prepared_microphysics_domain_recycling_bindings(
        state, plan, prepared, bindings));
    REQUIRE(bindings.size() == live.size() + 1);
    const Field3D residual_before =
        storage.runtime.microphysics.theta_storage_residual;
    Field3D invalid = prepared.terrain_leading_private_scalars[0];
    invalid(0, 0, 0) = std::numeric_limits<float>::quiet_NaN();
    auto invalid_bindings = bindings;
    invalid_bindings[1].leading_field = &invalid;
    REQUIRE_THROWS_WITH(
        tmv::domain::execute_homogeneous_field_shift(
            plan, invalid_bindings),
        Catch::Matchers::Contains("non-finite"));
    for (std::size_t index = 0; index < residual_before.size(); ++index)
        REQUIRE(storage.runtime.microphysics.theta_storage_residual.data()[index]
                == residual_before.data()[index]);
    (void)tmv::domain::execute_homogeneous_field_shift(plan, bindings);
    for (std::size_t f = 0; f < live.size(); ++f)
    {
        require_shifted_from_field(before[f], *live[f].field, plan, prepared.terrain_leading_private_scalars[f]);
    }
    require_shifted_from_field(
        residual_before,
        storage.runtime.microphysics.theta_storage_residual,
        plan, prepared.zero_leading_field);
}

TEST_CASE("D5 radiation held state shifts exports and commits one aggregate",
          "[core][domain_recycling][radiation][transaction][D5]")
{
    SimulationStorageState storage =
        radiation_recycling_storage(12.0 * 3600.0, true, true);
    SimulationStateView state = make_simulation_state_view(storage);
    const auto physical = physical_scalar_domain(state.grid);
    const auto plan =
        tmv::domain::plan_whole_cell_shift_for_physical_domain(
            state.grid.nr, state.grid.nth, state.grid.nz, 1, -1,
            physical.i_begin, physical.i_end,
            physical.j_begin, physical.j_end);
    const auto primary =
        tmv::domain::prepare_flat_homogeneous_primary_profiles(
            storage, std::vector<double>(4, 0.1));
    const auto prepared =
        tmv::domain::prepare_flat_homogeneous_radiation_recycling(
            state, plan, primary);

    REQUIRE(prepared.active);
    REQUIRE(prepared.sample_time_s == 12.0 * 3600.0);
    REQUIRE(prepared.leading_theta_tendency_s.size() == 4);
    REQUIRE(prepared.leading_cell_power_w.size() == 4);
    REQUIRE(std::string(
                tmv::domain::radiation_domain_recycling_algorithm_id()) ==
            "held_sample_time_flat_or_mapped_radiation_shift_v2");
    REQUIRE(std::string(tmv::radiation_prepared_column_algorithm_id()) ==
            "radiation_column_flux_power_fp32_storage_v1");

    const Field3D tendency_before = storage.tendencies.dtheta_dt_rad;
    const Field3D power_before =
        storage.runtime.radiation.held_cell_power_w;
    const float* tendency_pointer = storage.tendencies.dtheta_dt_rad.data();
    const float* power_pointer =
        storage.runtime.radiation.held_cell_power_w.data();
    const double last_update = storage.runtime.radiation.last_update_time_s;
    const double next_update = storage.runtime.radiation.next_update_time_s;
    const double last_expected =
        storage.runtime.radiation.last_expected_atmospheric_energy_j;
    const double last_applied =
        storage.runtime.radiation.last_applied_radiation_heat_j;
    const double accumulated_expected =
        storage.runtime.radiation.accumulated_expected_atmospheric_energy_j;
    const double accumulated_applied =
        storage.runtime.radiation.accumulated_applied_radiation_heat_j;
    const std::uint64_t applied_count =
        storage.runtime.radiation.applied_step_count;

    std::vector<float> tendency_trailing(plan.trailing_cells, 0.0f);
    std::vector<float> power_trailing(plan.trailing_cells, 0.0f);
    std::vector<tmv::domain::TrailingFieldExportBinding> exports;
    REQUIRE(tmv::domain::append_radiation_trailing_export_bindings(
        state, tendency_trailing, power_trailing, exports));
    const auto export_report =
        tmv::domain::export_trailing_fields(plan, exports);

    std::vector<tmv::domain::HomogeneousFieldShiftBinding> volumes;
    std::vector<tmv::domain::HomogeneousScalarCommitBinding> scalars;
    REQUIRE(tmv::domain::append_radiation_domain_recycling_bindings(
        state, plan, prepared, volumes, scalars));
    const auto report = tmv::domain::execute_homogeneous_window_shift(
        plan, volumes, {}, scalars);

    REQUIRE(export_report.fields_exported == 2);
    REQUIRE(export_report.values_exported == 2 * plan.trailing_cells);
    REQUIRE(report.volumes.fields_shifted == 2);
    REQUIRE(report.scalars_committed == 1);
    REQUIRE(storage.tendencies.dtheta_dt_rad.data() == tendency_pointer);
    REQUIRE(storage.runtime.radiation.held_cell_power_w.data() == power_pointer);
    require_exported(tendency_before, tendency_trailing, plan);
    require_exported(power_before, power_trailing, plan);
    require_shifted(
        tendency_before, storage.tendencies.dtheta_dt_rad, plan,
        prepared.leading_theta_tendency_s);
    require_shifted(
        power_before, storage.runtime.radiation.held_cell_power_w, plan,
        prepared.leading_cell_power_w);

    double post_shift_power = 0.0;
    for (int i = physical.i_begin; i < physical.i_end; ++i)
        for (int j = physical.j_begin; j < physical.j_end; ++j)
            for (int k = physical.k_begin; k < physical.k_end; ++k)
                post_shift_power += static_cast<double>(
                    storage.runtime.radiation.held_cell_power_w[i][j][k]);
    REQUIRE(storage.runtime.radiation.held_atmospheric_power_w ==
            post_shift_power);
    REQUIRE(post_shift_power ==
            prepared.committed_held_atmospheric_power_w);
    REQUIRE(storage.runtime.radiation.last_update_time_s == last_update);
    REQUIRE(storage.runtime.radiation.next_update_time_s == next_update);
    REQUIRE(storage.runtime.radiation.last_expected_atmospheric_energy_j ==
            last_expected);
    REQUIRE(storage.runtime.radiation.last_applied_radiation_heat_j ==
            last_applied);
    REQUIRE(storage.runtime.radiation.accumulated_expected_atmospheric_energy_j ==
            accumulated_expected);
    REQUIRE(storage.runtime.radiation.accumulated_applied_radiation_heat_j ==
            accumulated_applied);
    REQUIRE(storage.runtime.radiation.applied_step_count == applied_count);
}

TEST_CASE("D5 radiation leading state preserves held solar sample time",
          "[core][domain_recycling][radiation][solar][D5]")
{
    SimulationStorageState night =
        radiation_recycling_storage(0.0, false, true);
    SimulationStorageState noon =
        radiation_recycling_storage(12.0 * 3600.0, false, true);
    SimulationStateView night_state = make_simulation_state_view(night);
    SimulationStateView noon_state = make_simulation_state_view(noon);
    const auto physical = physical_scalar_domain(night_state.grid);
    const auto plan =
        tmv::domain::plan_whole_cell_shift_for_physical_domain(
            night_state.grid.nr, night_state.grid.nth,
            night_state.grid.nz, 1, 0,
            physical.i_begin, physical.i_end,
            physical.j_begin, physical.j_end);
    const auto night_primary =
        tmv::domain::prepare_flat_homogeneous_primary_profiles(
            night, std::vector<double>(4, 0.1));
    const auto noon_primary =
        tmv::domain::prepare_flat_homogeneous_primary_profiles(
            noon, std::vector<double>(4, 0.1));
    const auto night_prepared =
        tmv::domain::prepare_flat_homogeneous_radiation_recycling(
            night_state, plan, night_primary);
    const auto noon_prepared =
        tmv::domain::prepare_flat_homogeneous_radiation_recycling(
            noon_state, plan, noon_primary);

    double night_magnitude = 0.0;
    double noon_magnitude = 0.0;
    for (std::size_t k = 0; k < 4; ++k)
    {
        night_magnitude += std::abs(static_cast<double>(
            night_prepared.leading_cell_power_w[k]));
        noon_magnitude += std::abs(static_cast<double>(
            noon_prepared.leading_cell_power_w[k]));
    }
    REQUIRE(night_magnitude == 0.0);
    REQUIRE(noon_magnitude > 0.0);
    REQUIRE(night_prepared.sample_time_s == 0.0);
    REQUIRE(noon_prepared.sample_time_s == 12.0 * 3600.0);
}

TEST_CASE("Earth-fixed D5 radiation evaluates destination-indexed mapped columns",
          "[core][domain_recycling][terrain][radiation][transaction][D5]")
{
    SimulationStorageState storage = radiation_recycling_storage(
        3600.0, true, false);
    TerrainConfig config;
    config.scheme_id = "bell";
    config.coord_id = "smoothed";
    config.horizontal_placement_id = "earth_fixed";
    config.ztop = 800.0;
    config.bell.h0 = 120.0;
    config.bell.a = 350.0;
    config.bell.axisymmetric = true;
    config.feature_center_earth_x_m = 350.0;
    config.feature_center_earth_y_m = 625.0;
    storage.runtime.terrain.config = config;
    tmv::EarthFixedTerrainBuildRequest request;
    request.nr = storage.grid.nr;
    request.nth = storage.grid.nth;
    request.nz = storage.grid.nz;
    request.dx_m = storage.grid.dx;
    request.dy_m = storage.grid.dy;
    request.feature_center_earth_m = {350.0, 625.0};
    const auto geometry = tmv::build_earth_fixed_terrain_geometry(
        config, request);
    const auto primary = tmv::domain::prepare_terrain_aware_primary_fields(
        storage, geometry.metrics, std::vector<double>(4, 0.1));
    SimulationStateView state = make_simulation_state_view(storage);
    const auto physical = physical_scalar_domain(state.grid);
    const auto plan =
        tmv::domain::plan_whole_cell_shift_for_physical_domain(
            state.grid.nr, state.grid.nth, state.grid.nz, 1, -1,
            physical.i_begin, physical.i_end,
            physical.j_begin, physical.j_end);
    auto prepared = tmv::domain::prepare_terrain_aware_radiation_recycling(
        state, plan, primary, geometry.metrics);
    REQUIRE(prepared.active);
    REQUIRE(prepared.terrain_aware);
    REQUIRE(prepared.terrain_leading_theta_tendency_s.size()
            == static_cast<std::size_t>(6 * 7 * 4));
    REQUIRE(prepared.terrain_leading_cell_power_w.size()
            == static_cast<std::size_t>(6 * 7 * 4));

    std::vector<tmv::domain::HomogeneousFieldShiftBinding> volumes;
    std::vector<tmv::domain::HomogeneousScalarCommitBinding> scalars;
    REQUIRE(tmv::domain::append_radiation_domain_recycling_bindings(
        state, plan, prepared, volumes, scalars));
    const Field3D before = storage.runtime.radiation.held_cell_power_w;
    Field3D invalid = prepared.terrain_leading_cell_power_w;
    invalid(0, 0, 0) = std::numeric_limits<float>::quiet_NaN();
    auto invalid_volumes = volumes;
    invalid_volumes[1].leading_field = &invalid;
    REQUIRE_THROWS_WITH(
        tmv::domain::execute_homogeneous_window_shift(
            plan, invalid_volumes, {}, scalars),
        Catch::Matchers::Contains("non-finite"));
    for (std::size_t index = 0; index < before.size(); ++index)
        REQUIRE(storage.runtime.radiation.held_cell_power_w.data()[index]
                == before.data()[index]);
    (void)tmv::domain::execute_homogeneous_window_shift(
        plan, volumes, {}, scalars);
    require_shifted_from_field(
        before, storage.runtime.radiation.held_cell_power_w,
        plan, prepared.terrain_leading_cell_power_w);
    REQUIRE(storage.runtime.radiation.held_atmospheric_power_w
            == prepared.committed_held_atmospheric_power_w);
}

TEST_CASE("D5 radiation preparation rejects stale or unsupported ownership",
          "[core][domain_recycling][radiation][transaction][D5]")
{
    SimulationStorageState storage =
        radiation_recycling_storage(12.0 * 3600.0, true, true);
    SimulationStateView state = make_simulation_state_view(storage);
    const auto physical = physical_scalar_domain(state.grid);
    const auto plan =
        tmv::domain::plan_whole_cell_shift_for_physical_domain(
            state.grid.nr, state.grid.nth, state.grid.nz, 1, 0,
            physical.i_begin, physical.i_end,
            physical.j_begin, physical.j_end);
    const auto primary =
        tmv::domain::prepare_flat_homogeneous_primary_profiles(
            storage, std::vector<double>(4, 0.1));
    const Field3D tendency_before = storage.tendencies.dtheta_dt_rad;
    const Field3D power_before =
        storage.runtime.radiation.held_cell_power_w;

    SECTION("aggregate mismatch")
    {
        storage.runtime.radiation.held_atmospheric_power_w += 1.0;
        REQUIRE_THROWS_WITH(
            tmv::domain::prepare_flat_homogeneous_radiation_recycling(
                state, plan, primary),
            Catch::Matchers::Contains("aggregate ledger"));
    }
    SECTION("unsampled cadence")
    {
        storage.runtime.radiation.last_update_time_s = -300.0;
        REQUIRE_THROWS_WITH(
            tmv::domain::prepare_flat_homogeneous_radiation_recycling(
                state, plan, primary),
            Catch::Matchers::Contains("sampled held cadence"));
    }
    SECTION("active terrain")
    {
        storage.runtime.terrain.config.scheme_id = "bell";
        REQUIRE_THROWS_WITH(
            tmv::domain::prepare_flat_homogeneous_radiation_recycling(
                state, plan, primary),
            Catch::Matchers::Contains("flat terrain"));
    }
    for (std::size_t index = 0; index < tendency_before.size(); ++index)
    {
        REQUIRE(storage.tendencies.dtheta_dt_rad.data()[index] ==
                tendency_before.data()[index]);
        REQUIRE(storage.runtime.radiation.held_cell_power_w.data()[index] ==
                power_before.data()[index]);
    }
}

TEST_CASE("D5 scalar preflight is atomic with field shift",
          "[core][domain_recycling][scalar][transaction][D5]")
{
    Field3D field = indexed_field(4, 5, 3);
    const Field3D before = field;
    const std::vector<float> leading{1.0f, 2.0f, 3.0f};
    double ledger = 7.0;
    const auto plan =
        tmv::domain::plan_whole_cell_shift(4, 5, 3, 1, 0);
    REQUIRE_THROWS_WITH(
        tmv::domain::execute_homogeneous_window_shift(
            plan, {{"field", &field, &leading}}, {},
            {{"ledger", &ledger,
              std::numeric_limits<double>::quiet_NaN()}}),
        Catch::Matchers::Contains("non-finite"));
    REQUIRE(ledger == 7.0);
    for (std::size_t index = 0; index < field.size(); ++index)
        REQUIRE(field.data()[index] == before.data()[index]);

    std::vector<SlabColumnState> records(20, {100.0, 300.0, 0.01, true});
    const std::vector<SlabColumnState> records_before = records;
    const SlabColumnState leading_record{200.0, 301.0, 0.009, true};
    REQUIRE_THROWS_WITH(
        tmv::domain::execute_homogeneous_window_shift(
            plan, {{"field", &field, &leading}}, {}, {},
            {{"bad_records", records.data(), records.size() - 1,
              sizeof(SlabColumnState), &leading_record}}),
        Catch::Matchers::Contains("nr*nth records"));
    for (std::size_t index = 0; index < records.size(); ++index)
    {
        REQUIRE(records[index].h_m == records_before[index].h_m);
        REQUIRE(records[index].theta_m_k == records_before[index].theta_m_k);
        REQUIRE(records[index].qv_m_kgkg == records_before[index].qv_m_kgkg);
        REQUIRE(records[index].initialized == records_before[index].initialized);
    }
    for (std::size_t index = 0; index < field.size(); ++index)
        REQUIRE(field.data()[index] == before.data()[index]);
}

TEST_CASE("D5 boundary-layer persistent state and diagnostics shift atomically",
          "[core][domain_recycling][boundary_layer][transaction][D5]")
{
    SimulationStorageState storage = pbl_recycling_storage();
    SimulationStateView state = make_simulation_state_view(storage);
    const auto physical = physical_scalar_domain(state.grid);
    const auto plan =
        tmv::domain::plan_whole_cell_shift_for_physical_domain(
            state.grid.nr, state.grid.nth, state.grid.nz, 1, -1,
            physical.i_begin, physical.i_end,
            physical.j_begin, physical.j_end);
    const auto primary =
        tmv::domain::prepare_flat_homogeneous_primary_profiles(
            storage, std::vector<double>(4, 0.1));
    const auto prepared =
        tmv::domain::prepare_flat_homogeneous_boundary_layer_recycling(
            state, plan, primary);
    REQUIRE(prepared.active);
    REQUIRE(prepared.sample_time_s == 600.0);
    REQUIRE(prepared.leading_slab_column.initialized);
    REQUIRE(std::string(
                tmv::domain::boundary_layer_domain_recycling_algorithm_id()) ==
            "held_sample_flat_or_mapped_pbl_shift_v2");
    REQUIRE(std::string(
                tmv::boundary_layer_prepared_column_algorithm_id()) ==
            "pbl_column_conserved_source_fp32_storage_v1");

    auto& runtime = storage.runtime.boundary_layer;
    const std::array<Field3D, 5> before_fields{{
        runtime.held_u_force_n, runtime.held_v_force_n,
        runtime.held_theta_inventory_rate_kgk_s,
        runtime.held_vapor_mass_rate_kg_s,
        storage.tendencies.dtke_dt_pbl}};
    const std::array<const float*, 5> field_pointers{{
        runtime.held_u_force_n.data(), runtime.held_v_force_n.data(),
        runtime.held_theta_inventory_rate_kgk_s.data(),
        runtime.held_vapor_mass_rate_kg_s.data(),
        storage.tendencies.dtke_dt_pbl.data()}};
    const std::array<std::vector<float>, 9> before_surfaces{{
        runtime.skin_theta_2d, runtime.skin_qv_2d,
        storage.surface.ustar, storage.surface.tau_u,
        storage.surface.tau_v, storage.surface.sensible_heat_flux,
        storage.surface.latent_heat_flux, storage.surface.moisture_flux,
        storage.surface.pbl_height}};
    const std::vector<SlabColumnState> slab_before = runtime.slab_columns;
    const SlabColumnState* const slab_pointer = runtime.slab_columns.data();
    const double last_update = runtime.last_update_time_s;
    const double next_update = runtime.next_update_time_s;
    const auto last_work = runtime.last_momentum_work;
    const double cumulative_work = runtime.cumulative_committed_momentum_work_j;
    const std::uint64_t work_count = runtime.momentum_work_step_count;

    std::array<std::vector<float>, 5> trailing;
    for (auto& values : trailing)
        values.assign(plan.trailing_cells, 0.0f);
    std::vector<tmv::domain::TrailingFieldExportBinding> exports;
    REQUIRE(tmv::domain::append_boundary_layer_trailing_export_bindings(
        state, trailing, exports));
    const auto export_report =
        tmv::domain::export_trailing_fields(plan, exports);

    std::vector<tmv::domain::HomogeneousFieldShiftBinding> volumes;
    std::vector<tmv::domain::HomogeneousSurfaceShiftBinding> surfaces;
    std::vector<tmv::domain::HomogeneousSurfaceRecordShiftBinding> records;
    std::vector<tmv::domain::HomogeneousBooleanCommitBinding> flags;
    REQUIRE(tmv::domain::append_boundary_layer_domain_recycling_bindings(
        state, plan, prepared, volumes, surfaces, records, flags));
    const auto report = tmv::domain::execute_homogeneous_window_shift(
        plan, volumes, surfaces, {}, records, flags);

    REQUIRE(export_report.fields_exported == 5);
    REQUIRE(export_report.values_exported == 5 * plan.trailing_cells);
    REQUIRE(report.volumes.fields_shifted == 5);
    REQUIRE(report.surfaces_shifted == 9);
    REQUIRE(report.record_surfaces_shifted == 1);
    REQUIRE(report.flags_committed == 1);
    REQUIRE_FALSE(runtime.updated_this_step);
    REQUIRE(runtime.slab_columns.data() == slab_pointer);

    const std::array<const Field3D*, 5> after_fields{{
        &runtime.held_u_force_n, &runtime.held_v_force_n,
        &runtime.held_theta_inventory_rate_kgk_s,
        &runtime.held_vapor_mass_rate_kg_s,
        &storage.tendencies.dtke_dt_pbl}};
    const std::array<const std::vector<float>*, 5> leading_profiles{{
        &prepared.leading_u_force_n, &prepared.leading_v_force_n,
        &prepared.leading_theta_inventory_rate_kgk_s,
        &prepared.leading_vapor_mass_rate_kg_s,
        &prepared.leading_dtke_s}};
    for (std::size_t field = 0; field < 5; ++field)
    {
        REQUIRE(after_fields[field]->data() == field_pointers[field]);
        require_exported(before_fields[field], trailing[field], plan);
        require_shifted(before_fields[field], *after_fields[field], plan,
                        *leading_profiles[field]);
    }

    const std::array<const std::vector<float>*, 9> after_surfaces{{
        &runtime.skin_theta_2d, &runtime.skin_qv_2d,
        &storage.surface.ustar, &storage.surface.tau_u,
        &storage.surface.tau_v, &storage.surface.sensible_heat_flux,
        &storage.surface.latent_heat_flux, &storage.surface.moisture_flux,
        &storage.surface.pbl_height}};
    const std::array<float, 9> leading_surface_values{{
        prepared.leading_skin_theta, prepared.leading_skin_qv,
        prepared.leading_ustar, prepared.leading_tau_u,
        prepared.leading_tau_v, prepared.leading_sensible_heat_flux,
        prepared.leading_latent_heat_flux, prepared.leading_moisture_flux,
        prepared.leading_pbl_height}};
    for (std::size_t field = 0; field < 9; ++field)
        require_surface_shifted(
            before_surfaces[field], *after_surfaces[field], plan,
            leading_surface_values[field]);

    for (int i = 0; i < plan.nr; ++i)
        for (int j = 0; j < plan.nth; ++j)
        {
            const bool retained = i >= plan.destination_i_begin
                && i < plan.destination_i_end
                && j >= plan.destination_j_begin
                && j < plan.destination_j_end;
            const SlabColumnState& actual = runtime.slab_columns[
                static_cast<std::size_t>(i * plan.nth + j)];
            const SlabColumnState& expected = retained
                ? slab_before[static_cast<std::size_t>(
                      (i + plan.shift_i_cells) * plan.nth
                      + j + plan.shift_j_cells)]
                : prepared.leading_slab_column;
            REQUIRE(actual.h_m == expected.h_m);
            REQUIRE(actual.theta_m_k == expected.theta_m_k);
            REQUIRE(actual.qv_m_kgkg == expected.qv_m_kgkg);
            REQUIRE(actual.initialized == expected.initialized);
        }
    REQUIRE(runtime.last_update_time_s == last_update);
    REQUIRE(runtime.next_update_time_s == next_update);
    REQUIRE(runtime.last_momentum_work.committed_work_j ==
            last_work.committed_work_j);
    REQUIRE(runtime.cumulative_committed_momentum_work_j == cumulative_work);
    REQUIRE(runtime.momentum_work_step_count == work_count);
}

TEST_CASE("D5 turbulence invalidation shares the atomic window transaction",
          "[core][domain_recycling][turbulence][transaction][D5]")
{
    SimulationStorageState storage(
        GridStorageState{6, 7, 4, 100.0, 250.0, 0.1, 200.0});
    storage.grid.geometry.initialize(
        6, 7, 4, 100.0, 250.0, 200.0,
        CoordinateSystem::Cartesian, StaggerType::CGrid);
    storage.time_s = 100.0;
    auto& runtime = storage.runtime.turbulence;
    runtime.config.scheme_id = "d5_stub";
    runtime.config.dt_sgs = 2.0;
    runtime.scheme = std::make_unique<D5TurbulenceScheme>();
    runtime.scheme->initialize(runtime.config);
    runtime.last_update_time_s = 99.0;
    runtime.next_update_time_s = 101.0;
    runtime.last_explicit_stability_limit_s = 0.25;
    runtime.last_explicit_substep_count = 8;
    runtime.held_tendencies.resident_source_unchanged = true;
    runtime.candidate_tendencies.resident_source_unchanged = true;

    const auto initialize_group = [](TurbulenceTendencies& tendencies,
                                     float offset)
    {
        tendencies.dudt_sgs = indexed_field(6, 7, 4, offset + 0.0f);
        tendencies.dvdt_sgs = indexed_field(6, 7, 4, offset + 1000.0f);
        tendencies.dwdt_sgs = indexed_field(6, 7, 4, offset + 2000.0f);
        tendencies.dthetadt_sgs = indexed_field(6, 7, 4, offset + 3000.0f);
        tendencies.dqvdt_sgs = indexed_field(6, 7, 4, offset + 4000.0f);
        tendencies.dtkedt_sgs = indexed_field(6, 7, 4, offset + 5000.0f);
    };
    initialize_group(runtime.held_tendencies, 100.0f);
    initialize_group(runtime.candidate_tendencies, 10000.0f);
    // Rejected unpublished scratch may be non-finite; recycling must safely erase it
    // without treating it as committed science state.
    runtime.candidate_tendencies.dqvdt_sgs(1, 2, 3) =
        std::numeric_limits<float>::quiet_NaN();

    runtime.grid_metrics_scratch.dx = 123.0;
    runtime.grid_metrics_scratch.dy = 456.0;
    runtime.grid_metrics_scratch.dz = {1.0, 2.0, 3.0, 4.0};
    runtime.grid_metrics_scratch.z_int = {0.0, 1.0, 3.0, 6.0, 10.0};
    runtime.grid_metrics_scratch.terrain_metrics_active = true;
    runtime.grid_metrics_scratch.terrain_metrics =
        reinterpret_cast<const TerrainMetrics3D*>(1);
    runtime.grid_metrics_scratch.terrain_topography =
        reinterpret_cast<const Topography2D*>(1);

    runtime.last_momentum_work.requested_linear_work_j = 1.0;
    runtime.last_momentum_work.finite_step_correction_j = 2.0;
    runtime.last_momentum_work.storage_rounding_j = 3.0;
    runtime.last_momentum_work.committed_work_j = 6.0;
    runtime.cumulative_moist_energy_work.requested_linear_work_j = 10.0;
    runtime.cumulative_moist_energy_work.finite_step_correction_j = 20.0;
    runtime.cumulative_moist_energy_work.storage_rounding_j = 30.0;
    runtime.cumulative_moist_energy_work.committed_work_j = 60.0;
    runtime.cumulative_requested_momentum_work_j = 10.0;
    runtime.cumulative_finite_step_momentum_correction_j = 20.0;
    runtime.cumulative_momentum_storage_rounding_j = 30.0;
    runtime.cumulative_committed_momentum_work_j = 60.0;
    runtime.momentum_work_step_count = 7;

    const auto tendency_fields = [](TurbulenceRuntimeState& owner)
    {
        return std::array<Field3D*, 12>{{
            &owner.held_tendencies.dudt_sgs,
            &owner.held_tendencies.dvdt_sgs,
            &owner.held_tendencies.dwdt_sgs,
            &owner.held_tendencies.dthetadt_sgs,
            &owner.held_tendencies.dqvdt_sgs,
            &owner.held_tendencies.dtkedt_sgs,
            &owner.candidate_tendencies.dudt_sgs,
            &owner.candidate_tendencies.dvdt_sgs,
            &owner.candidate_tendencies.dwdt_sgs,
            &owner.candidate_tendencies.dthetadt_sgs,
            &owner.candidate_tendencies.dqvdt_sgs,
            &owner.candidate_tendencies.dtkedt_sgs,
        }};
    };
    const auto fields = tendency_fields(runtime);
    std::array<const float*, 12> pointers{};
    std::array<std::size_t, 12> capacities{};
    for (std::size_t index = 0; index < fields.size(); ++index)
    {
        pointers[index] = fields[index]->data();
        capacities[index] = fields[index]->capacity();
    }
    const std::size_t dz_capacity = runtime.grid_metrics_scratch.dz.capacity();
    const std::size_t z_int_capacity =
        runtime.grid_metrics_scratch.z_int.capacity();
    TurbulenceSchemeBase* const scheme = runtime.scheme.get();
    const TurbulenceConfig config = runtime.config;
    const auto last_work = runtime.last_momentum_work;
    const auto cumulative_work = runtime.cumulative_moist_energy_work;
    const double last_update = runtime.last_update_time_s;
    const std::uint64_t work_steps = runtime.momentum_work_step_count;

    SimulationStateView state = make_simulation_state_view(storage);
    const auto plan = tmv::domain::plan_whole_cell_shift(6, 7, 4, 1, -1);
    Field3D shifted = indexed_field(6, 7, 4, -1000.0f);
    const Field3D shifted_before = shifted;
    const std::vector<float> leading{-1.0f, -2.0f, -3.0f, -4.0f};
    std::vector<tmv::domain::HomogeneousScalarCommitBinding> scalars;
    std::vector<tmv::domain::HomogeneousRuntimeInvalidationBinding>
        invalidations;
    REQUIRE(tmv::domain::append_turbulence_domain_recycling_bindings(
        state, scalars, invalidations));
    REQUIRE(std::string(
                tmv::domain::turbulence_domain_recycling_algorithm_id()) ==
            "preserve_history_invalidate_nonlinear_sgs_state_v1");

    // A later preflight error must leave the shift, cadence, and every cache
    // untouched; the no-throw invalidation is a true commit-phase action.
    double invalid_target = 4.0;
    scalars.push_back({
        "invalid", &invalid_target,
        std::numeric_limits<double>::quiet_NaN()});
    REQUIRE_THROWS_WITH(
        tmv::domain::execute_homogeneous_window_shift(
            plan, {{"shifted", &shifted, &leading}}, {}, scalars, {}, {},
            invalidations),
        Catch::Matchers::Contains("non-finite"));
    REQUIRE(shifted(2, 3, 1) == shifted_before(2, 3, 1));
    REQUIRE(runtime.held_tendencies.dudt_sgs(0, 0, 0) == 100.0f);
    REQUIRE(runtime.next_update_time_s == 101.0);
    REQUIRE(runtime.grid_metrics_scratch.dx == 123.0);

    scalars.pop_back();
    const auto report = tmv::domain::execute_homogeneous_window_shift(
        plan, {{"shifted", &shifted, &leading}}, {}, scalars, {}, {},
        invalidations);
    REQUIRE(report.runtime_invalidations_committed == 1);
    require_shifted(shifted_before, shifted, plan, leading);
    for (std::size_t field = 0; field < fields.size(); ++field)
    {
        REQUIRE(fields[field]->data() == pointers[field]);
        REQUIRE(fields[field]->capacity() == capacities[field]);
        for (std::size_t index = 0; index < fields[field]->size(); ++index)
            REQUIRE(fields[field]->data()[index] == 0.0f);
    }
    REQUIRE_FALSE(runtime.held_tendencies.resident_source_unchanged);
    REQUIRE_FALSE(runtime.candidate_tendencies.resident_source_unchanged);
    REQUIRE(std::isnan(runtime.grid_metrics_scratch.dx));
    REQUIRE(std::isnan(runtime.grid_metrics_scratch.dy));
    REQUIRE(runtime.grid_metrics_scratch.dz.capacity() == dz_capacity);
    REQUIRE(runtime.grid_metrics_scratch.z_int.capacity() == z_int_capacity);
    for (double value : runtime.grid_metrics_scratch.dz)
        REQUIRE(std::isnan(value));
    for (double value : runtime.grid_metrics_scratch.z_int)
        REQUIRE(std::isnan(value));
    REQUIRE_FALSE(runtime.grid_metrics_scratch.terrain_metrics_active);
    REQUIRE(runtime.grid_metrics_scratch.terrain_metrics == nullptr);
    REQUIRE(runtime.grid_metrics_scratch.terrain_topography == nullptr);
    REQUIRE(runtime.next_update_time_s == storage.time_s);
    REQUIRE(runtime.last_update_time_s == last_update);
    REQUIRE(std::isinf(runtime.last_explicit_stability_limit_s));
    REQUIRE(runtime.last_explicit_stability_limit_s > 0.0);
    REQUIRE(runtime.last_explicit_substep_count == 0);
    REQUIRE(runtime.scheme.get() == scheme);
    REQUIRE(runtime.config.scheme_id == config.scheme_id);
    REQUIRE(runtime.config.dt_sgs == config.dt_sgs);
    REQUIRE(runtime.last_momentum_work.committed_work_j ==
            last_work.committed_work_j);
    REQUIRE(runtime.cumulative_moist_energy_work.committed_work_j ==
            cumulative_work.committed_work_j);
    REQUIRE(runtime.cumulative_committed_momentum_work_j == 60.0);
    REQUIRE(runtime.momentum_work_step_count == work_steps);
}

TEST_CASE("D5 turbulence preflight rejects malformed published state",
          "[core][domain_recycling][turbulence][rollback][D5]")
{
    SimulationStorageState storage(
        GridStorageState{3, 4, 2, 100.0, 250.0, 0.1, 200.0});
    storage.time_s = 10.0;
    auto& runtime = storage.runtime.turbulence;
    runtime.config.scheme_id = "d5_stub";
    runtime.scheme = std::make_unique<D5TurbulenceScheme>();
    runtime.last_update_time_s = 9.0;
    runtime.next_update_time_s = 11.0;
    runtime.held_tendencies.dudt_sgs = indexed_field(3, 4, 2);
    runtime.held_tendencies.dvdt_sgs = indexed_field(3, 4, 2);
    runtime.held_tendencies.dwdt_sgs = indexed_field(3, 4, 2);
    runtime.held_tendencies.dthetadt_sgs = indexed_field(3, 4, 2);
    runtime.held_tendencies.dqvdt_sgs = indexed_field(3, 4, 2);
    runtime.held_tendencies.dtkedt_sgs = indexed_field(3, 4, 2);
    SimulationStateView state = make_simulation_state_view(storage);
    std::vector<tmv::domain::HomogeneousScalarCommitBinding> scalars;
    std::vector<tmv::domain::HomogeneousRuntimeInvalidationBinding>
        invalidations;

    SECTION("non-finite held publication")
    {
        runtime.held_tendencies.dwdt_sgs(1, 1, 1) =
            std::numeric_limits<float>::infinity();
        REQUIRE_THROWS_WITH(
            tmv::domain::append_turbulence_domain_recycling_bindings(
                state, scalars, invalidations),
            Catch::Matchers::Contains("non-finite"));
    }
    SECTION("partial held allocation")
    {
        runtime.held_tendencies.dqvdt_sgs.release_storage();
        REQUIRE_THROWS_WITH(
            tmv::domain::append_turbulence_domain_recycling_bindings(
                state, scalars, invalidations),
            Catch::Matchers::Contains("partially allocated"));
    }
    REQUIRE(runtime.next_update_time_s == 11.0);
    REQUIRE(invalidations.empty());
}

TEST_CASE("D5 non-evolving chaos cache invalidates atomically",
          "[core][domain_recycling][chaos][transaction][D5]")
{
    SimulationStorageState storage(
        GridStorageState{4, 5, 3, 100.0, 250.0, 0.1, 200.0});
    auto& runtime = storage.runtime.chaos;
    runtime.config.scheme_id = "initial_conditions";
    runtime.config.dt_chaos = 1.0;
    runtime.scheme =
        std::make_unique<D5ChaosScheme>("initial_conditions");
    runtime.noise_elapsed_s = 0.25;
    runtime.diagnostics.mean_perturbation = 0.1;
    runtime.diagnostics.variance_perturbation = 0.2;
    runtime.diagnostics.correlation_length_x = 1000.0;
    runtime.diagnostics.correlation_length_y = 2000.0;
    runtime.diagnostics.min_qv_after_perturbation = 0.003;
    runtime.diagnostics.time_apply_ic = 0.5;
    runtime.base_du_dt_pbl = indexed_field(4, 5, 3, 100.0f);
    runtime.base_dv_dt_pbl = indexed_field(4, 5, 3, 200.0f);
    runtime.base_dtheta_dt_pbl = indexed_field(4, 5, 3, 300.0f);
    runtime.base_dqv_dt_pbl = indexed_field(4, 5, 3, 400.0f);
    runtime.base_dtke_dt_pbl = indexed_field(4, 5, 3, 500.0f);
    runtime.base_sfc_ustar.assign(20, 1.0f);
    runtime.base_sfc_tau_u.assign(20, 2.0f);
    runtime.base_sfc_tau_v.assign(20, 3.0f);
    runtime.base_sfc_sensible_heat_flux.assign(20, 4.0f);
    runtime.base_sfc_latent_heat_flux.assign(20, 5.0f);
    runtime.base_sfc_moisture_flux.assign(20, 6.0f);
    runtime.pbl_base_tendencies_valid = true;
    runtime.grid_metrics_scratch.dx = 10.0;
    runtime.grid_metrics_scratch.dy = 20.0;
    runtime.grid_metrics_scratch.dz = {30.0, 30.0, 30.0};
    runtime.grid_metrics_scratch.z_int = {0.0, 30.0, 60.0, 90.0};

    const std::array<Field3D*, 5> cached{{
        &runtime.base_du_dt_pbl,
        &runtime.base_dv_dt_pbl,
        &runtime.base_dtheta_dt_pbl,
        &runtime.base_dqv_dt_pbl,
        &runtime.base_dtke_dt_pbl,
    }};
    std::array<const float*, 5> pointers{};
    std::array<std::size_t, 5> capacities{};
    for (std::size_t index = 0; index < cached.size(); ++index)
    {
        pointers[index] = cached[index]->data();
        capacities[index] = cached[index]->capacity();
    }
    const std::size_t dz_capacity = runtime.grid_metrics_scratch.dz.capacity();
    const std::size_t z_int_capacity =
        runtime.grid_metrics_scratch.z_int.capacity();
    auto* const scheme = runtime.scheme.get();
    const auto diagnostics = runtime.diagnostics;

    SimulationStateView state = make_simulation_state_view(storage);
    std::vector<tmv::domain::HomogeneousRuntimeInvalidationBinding>
        invalidations;
    REQUIRE(
        tmv::domain::append_non_evolving_chaos_domain_recycling_bindings(
            state, invalidations));
    REQUIRE(std::string(tmv::domain::chaos_domain_recycling_algorithm_id()) ==
            "earth_control_lattice_translate_or_shift_pbl_cache_v3");

    const auto plan = tmv::domain::plan_whole_cell_shift(4, 5, 3, -1, 1);
    Field3D shifted = indexed_field(4, 5, 3, -1000.0f);
    const Field3D shifted_before = shifted;
    const std::vector<float> leading{-1.0f, -2.0f, -3.0f};
    double invalid_target = 2.0;
    const std::vector<tmv::domain::HomogeneousScalarCommitBinding> bad{{
        "bad", &invalid_target,
        std::numeric_limits<double>::quiet_NaN()}};
    REQUIRE_THROWS_WITH(
        tmv::domain::execute_homogeneous_window_shift(
            plan, {{"shifted", &shifted, &leading}}, {}, bad, {}, {},
            invalidations),
        Catch::Matchers::Contains("non-finite"));
    REQUIRE(runtime.pbl_base_tendencies_valid);
    REQUIRE(runtime.base_du_dt_pbl(0, 0, 0) == 100.0f);
    REQUIRE(runtime.grid_metrics_scratch.dx == 10.0);
    REQUIRE(shifted(1, 2, 0) == shifted_before(1, 2, 0));

    const auto report = tmv::domain::execute_homogeneous_window_shift(
        plan, {{"shifted", &shifted, &leading}}, {}, {}, {}, {},
        invalidations);
    REQUIRE(report.runtime_invalidations_committed == 1);
    require_shifted(shifted_before, shifted, plan, leading);
    for (std::size_t field = 0; field < cached.size(); ++field)
    {
        REQUIRE(cached[field]->data() == pointers[field]);
        REQUIRE(cached[field]->capacity() == capacities[field]);
        for (std::size_t index = 0; index < cached[field]->size(); ++index)
            REQUIRE(cached[field]->data()[index] == 0.0f);
    }
    REQUIRE_FALSE(runtime.pbl_base_tendencies_valid);
    REQUIRE(runtime.base_sfc_ustar.empty());
    REQUIRE(runtime.base_sfc_tau_u.empty());
    REQUIRE(runtime.base_sfc_tau_v.empty());
    REQUIRE(runtime.base_sfc_sensible_heat_flux.empty());
    REQUIRE(runtime.base_sfc_latent_heat_flux.empty());
    REQUIRE(runtime.base_sfc_moisture_flux.empty());
    REQUIRE(std::isnan(runtime.grid_metrics_scratch.dx));
    REQUIRE(std::isnan(runtime.grid_metrics_scratch.dy));
    REQUIRE(runtime.grid_metrics_scratch.dz.capacity() == dz_capacity);
    REQUIRE(runtime.grid_metrics_scratch.z_int.capacity() == z_int_capacity);
    REQUIRE(runtime.noise_elapsed_s == 0.25);
    REQUIRE(runtime.scheme.get() == scheme);
    REQUIRE(runtime.config.scheme_id == "initial_conditions");
    REQUIRE(runtime.diagnostics.mean_perturbation ==
            diagnostics.mean_perturbation);
    REQUIRE(runtime.diagnostics.variance_perturbation ==
            diagnostics.variance_perturbation);
    REQUIRE(runtime.diagnostics.time_apply_ic == diagnostics.time_apply_ic);
}

TEST_CASE("D5 evolving chaos translates one Earth AR1 realization atomically",
          "[core][domain_recycling][chaos][earth_coordinates][transaction][D5]")
{
    SimulationStorageState storage(
        GridStorageState{8, 9, 3, 100.0, 250.0, 0.1, 200.0});
    storage.grid.geometry.initialize(
        8, 9, 3, 100.0, 250.0, 200.0,
        CoordinateSystem::Cartesian, StaggerType::CGrid);
    auto& runtime = storage.runtime.chaos;
    runtime.config.scheme_id = "full_stochastic";
    runtime.config.seed = 778899ULL;
    runtime.config.member_id = 7;
    runtime.config.dt_chaos = 2.0;
    runtime.config.tau_t = 20.0;
    runtime.config.Lx = 350.0;
    runtime.config.Ly = 500.0;
    runtime.config.filter_id = "recursive_gaussian";
    runtime.scheme = std::make_unique<tmv::chaos::FullStochasticScheme>();
    GridMetrics grid;
    grid.dx = storage.grid.dx;
    grid.dy = storage.grid.dy;
    grid.dz.assign(3, storage.grid.dz);
    grid.z_int = storage.grid.geometry.z_faces;
    runtime.scheme->initialize(runtime.config, grid);
    Field3D tendency(8, 9, 3, 1.0f);
    tmv::chaos::ChaosTendencies tendencies;
    tendencies.dtheta_micro_dt = &tendency;
    tmv::chaos::ChaosStateView chaos_state;
    chaos_state.grid = &grid;
    runtime.scheme->apply_tendencies(
        runtime.config, grid, chaos_state, tendencies, nullptr);
    runtime.scheme->step_noise(runtime.config, grid, 2.0);
    runtime.scheme->step_noise(runtime.config, grid, 2.0);
    const PortableSchemeState before = runtime.scheme->portable_state();
    REQUIRE(before.version == 4);
    REQUIRE(before.fields.size() == 12);
    runtime.base_du_dt_pbl = indexed_field(8, 9, 3, 100.0f);
    runtime.base_dv_dt_pbl = indexed_field(8, 9, 3, 200.0f);
    runtime.base_dtheta_dt_pbl = indexed_field(8, 9, 3, 300.0f);
    runtime.base_dqv_dt_pbl = indexed_field(8, 9, 3, 400.0f);
    runtime.base_dtke_dt_pbl = indexed_field(8, 9, 3, 500.0f);
    runtime.base_sfc_ustar.assign(72, 1.0f);
    runtime.base_sfc_tau_u.assign(72, 2.0f);
    runtime.base_sfc_tau_v.assign(72, 3.0f);
    runtime.base_sfc_sensible_heat_flux.assign(72, 4.0f);
    runtime.base_sfc_latent_heat_flux.assign(72, 5.0f);
    runtime.base_sfc_moisture_flux.assign(72, 6.0f);
    runtime.pbl_base_tendencies_valid = true;
    const Field3D base_u_before = runtime.base_du_dt_pbl;
    const std::vector<float> base_tau_before = runtime.base_sfc_tau_u;

    SimulationStateView state = make_simulation_state_view(storage);
    const PhysicalCellDomain physical = physical_scalar_domain(state.grid);
    const auto plan =
        tmv::domain::plan_whole_cell_shift_for_physical_domain(
            8, 9, 3, 2, -1,
            physical.i_begin, physical.i_end,
            physical.j_begin, physical.j_end);
    auto prepared = tmv::domain::prepare_evolving_chaos_domain_recycling(
        state, plan);
    REQUIRE(prepared.scheme);
    REQUIRE(runtime.scheme->portable_state().integers == before.integers);
    tmv::domain::PreparedBoundaryLayerDomainRecyclingState pbl;
    pbl.active = true;
    pbl.leading_u_force_n = {-1.0f, -2.0f, -3.0f};
    pbl.leading_v_force_n = {-4.0f, -5.0f, -6.0f};
    pbl.leading_theta_inventory_rate_kgk_s = {-7.0f, -8.0f, -9.0f};
    pbl.leading_vapor_mass_rate_kg_s = {-10.0f, -11.0f, -12.0f};
    pbl.leading_dtke_s = {-13.0f, -14.0f, -15.0f};
    pbl.leading_ustar = 21.0f;
    pbl.leading_tau_u = 22.0f;
    pbl.leading_tau_v = 23.0f;
    pbl.leading_sensible_heat_flux = 24.0f;
    pbl.leading_latent_heat_flux = 25.0f;
    pbl.leading_moisture_flux = 26.0f;
    std::vector<tmv::domain::HomogeneousFieldShiftBinding> volumes;
    std::vector<tmv::domain::HomogeneousSurfaceShiftBinding> surfaces;
    REQUIRE(tmv::domain::
        append_evolving_chaos_pbl_cache_domain_recycling_bindings(
            state, pbl, prepared, volumes, surfaces));
    std::vector<tmv::domain::HomogeneousRuntimeCommitBinding> commits;
    REQUIRE(tmv::domain::append_evolving_chaos_domain_recycling_bindings(
        state, prepared, commits));

    double rollback_target = 5.0;
    const std::vector<tmv::domain::HomogeneousScalarCommitBinding> bad{{
        "invalid", &rollback_target,
        std::numeric_limits<double>::quiet_NaN()}};
    REQUIRE_THROWS_WITH(
        tmv::domain::execute_homogeneous_window_shift(
            plan, volumes, surfaces, bad, {}, {}, {}, commits),
        Catch::Matchers::Contains("non-finite"));
    REQUIRE(runtime.scheme->portable_state().integers == before.integers);

    const auto report = tmv::domain::execute_homogeneous_window_shift(
        plan, volumes, surfaces, {}, {}, {}, {}, commits);
    REQUIRE(report.runtime_commits_committed == 1);
    const PortableSchemeState after = runtime.scheme->portable_state();
    REQUIRE(after.version == 4);
    REQUIRE(after.integers[0] == before.integers[0]);
    REQUIRE(after.integers[1] != before.integers[1]);
    REQUIRE(after.integers[2] != before.integers[2]);
    for (std::size_t field = 0; field < 6; ++field)
        for (int i = 0; i < 8; ++i)
            for (int j = 0; j < 9; ++j)
            {
                const int source_i = i + plan.shift_i_cells;
                const int source_j = j + plan.shift_j_cells;
                if (source_i < 0 || source_i >= 8
                    || source_j < 0 || source_j >= 9)
                    continue;
                for (int k = 0; k < 3; ++k)
                {
                    CAPTURE(field, i, j, k);
                    REQUIRE(after.fields[field](i, j, k)
                            == before.fields[field](source_i, source_j, k));
                }
            }
    REQUIRE(runtime.pbl_base_tendencies_valid);
    require_shifted(
        base_u_before, runtime.base_du_dt_pbl, plan,
        pbl.leading_u_force_n);
    require_surface_shifted(
        base_tau_before, runtime.base_sfc_tau_u, plan,
        pbl.leading_tau_u);
}

TEST_CASE("D5 Rayleigh profiles are validated translation invariants",
          "[core][domain_recycling][rayleigh][transaction][D5]")
{
    global_damping_enabled = true;
    global_damping_z_d = 500.0;
    global_damping_alpha = 0.05;
    global_w_only_top_sponge = true;
    global_w_only_depth_cells = 2;
    global_w_only_alpha_max = 0.2;
    global_lateral_sponge_enabled = true;
    global_lateral_sponge_width = 2;
    global_lateral_sponge_alpha = 0.03;

    SimulationStorageState storage(
        GridStorageState{5, 6, 3, 100.0, 250.0, 0.1, 200.0});
    storage.grid.geometry.initialize(
        5, 6, 3, 100.0, 250.0, 200.0,
        CoordinateSystem::Cartesian, StaggerType::CGrid);
    SimulationStateView state = make_simulation_state_view(storage);
    (void)rayleigh_alpha_profile(state);
    (void)rayleigh_upper_w_alpha_profile(state);
    (void)rayleigh_w_only_alpha_profile(state);
    (void)rayleigh_u_ref_profile(state);
    (void)rayleigh_v_ref_profile(state);
    (void)rayleigh_lateral_alpha_profile(state);

    auto& runtime = storage.runtime.rayleigh;
    const std::array<std::vector<double>*, 6> profiles{{
        &runtime.alpha, &runtime.u_ref, &runtime.v_ref,
        &runtime.upper_w_alpha, &runtime.w_only_alpha,
        &runtime.lateral_alpha,
    }};
    std::array<const double*, 6> pointers{};
    std::array<std::size_t, 6> capacities{};
    std::array<std::vector<double>, 6> before{};
    for (std::size_t index = 0; index < profiles.size(); ++index)
    {
        pointers[index] = profiles[index]->data();
        capacities[index] = profiles[index]->capacity();
        before[index] = *profiles[index];
    }

    std::vector<tmv::domain::HomogeneousBooleanCommitBinding> flags;
    REQUIRE(tmv::domain::append_rayleigh_domain_recycling_bindings(
        state, flags));
    REQUIRE(std::string(rayleigh_horizontal_recycling_algorithm_id()) ==
            "validated_translation_invariant_rayleigh_profiles_v1");
    REQUIRE(flags.size() == 2);

    const auto plan = tmv::domain::plan_whole_cell_shift(5, 6, 3, 1, -1);
    Field3D shifted = indexed_field(5, 6, 3, -1000.0f);
    const Field3D shifted_before = shifted;
    const std::vector<float> leading{-1.0f, -2.0f, -3.0f};
    const auto report = tmv::domain::execute_homogeneous_window_shift(
        plan, {{"shifted", &shifted, &leading}}, {}, {}, {}, flags);
    REQUIRE(report.flags_committed == 2);
    require_shifted(shifted_before, shifted, plan, leading);
    REQUIRE(runtime.upper_initialized);
    REQUIRE(runtime.lateral_initialized);
    for (std::size_t index = 0; index < profiles.size(); ++index)
    {
        REQUIRE(*profiles[index] == before[index]);
        REQUIRE(profiles[index]->data() == pointers[index]);
        REQUIRE(profiles[index]->capacity() == capacities[index]);
    }
}

TEST_CASE("D5 Rayleigh validation rejects stale profiles and accepts restart-lazy upper-w",
          "[core][domain_recycling][rayleigh][rollback][checkpoint][D5]")
{
    global_damping_enabled = true;
    global_damping_z_d = 500.0;
    global_damping_alpha = 0.05;
    global_w_only_top_sponge = true;
    global_w_only_depth_cells = 2;
    global_w_only_alpha_max = 0.2;
    global_lateral_sponge_enabled = true;
    global_lateral_sponge_width = 2;
    global_lateral_sponge_alpha = 0.03;
    SimulationStorageState storage(
        GridStorageState{5, 6, 3, 100.0, 250.0, 0.1, 200.0});
    storage.grid.geometry.initialize(
        5, 6, 3, 100.0, 250.0, 200.0,
        CoordinateSystem::Cartesian, StaggerType::CGrid);
    SimulationStateView state = make_simulation_state_view(storage);
    (void)rayleigh_alpha_profile(state);
    (void)rayleigh_lateral_alpha_profile(state);
    auto& runtime = storage.runtime.rayleigh;

    SECTION("stale initialized profile")
    {
        runtime.alpha[1] += 1.0e-6;
        std::vector<tmv::domain::HomogeneousBooleanCommitBinding> flags;
        REQUIRE_THROWS_WITH(
            tmv::domain::append_rayleigh_domain_recycling_bindings(
                state, flags),
            Catch::Matchers::Contains("disagrees"));
        REQUIRE(flags.empty());
    }
    SECTION("restart-lazy upper-w profile")
    {
        runtime.upper_w_alpha.clear();
        const double* const alpha_pointer = runtime.alpha.data();
        std::vector<tmv::domain::HomogeneousBooleanCommitBinding> flags;
        REQUIRE(tmv::domain::append_rayleigh_domain_recycling_bindings(
            state, flags));
        const auto plan =
            tmv::domain::plan_whole_cell_shift(5, 6, 3, -1, 0);
        const auto report = tmv::domain::execute_homogeneous_window_shift(
            plan, {}, {}, {}, {}, flags);
        REQUIRE(report.flags_committed == 2);
        REQUIRE(runtime.upper_w_alpha.empty());
        REQUIRE(runtime.alpha.data() == alpha_pointer);
        // Normal runtime access deterministically repairs the lazy checkpoint
        // representation before the profile can be consumed.
        REQUIRE(rayleigh_upper_w_alpha_profile(state).size() == 3);
    }
}

TEST_CASE("D5 advection invalidation preserves ledgers and allocation",
          "[core][domain_recycling][advection][transaction][D5]")
{
    SimulationStorageState storage(
        GridStorageState{5, 6, 3, 100.0, 250.0, 0.1, 200.0});
    storage.grid.geometry.initialize(
        5, 6, 3, 100.0, 250.0, 200.0,
        CoordinateSystem::Cartesian, StaggerType::CGrid);
    auto& runtime = storage.runtime.advection;
    runtime.config.scheme_id = "d5_stub";
    runtime.config.positivity_dt = 0.1;
    runtime.config.cfl_target = 0.8;
    runtime.scheme = std::make_unique<D5AdvectionScheme>();
    runtime.scalar_scratch_a = indexed_field(5, 6, 3, 100.0f);
    runtime.scalar_scratch_b = indexed_field(5, 6, 3, 200.0f);
    runtime.vertical_tendencies.dqdt_adv =
        indexed_field(5, 6, 3, 300.0f);
    runtime.thermodynamic_boundary_reports_requested = true;
    runtime.thermodynamic_moist_energy_reports_requested = true;
    runtime.thermodynamic_boundary_reports_complete = true;
    runtime.thermodynamic_moist_energy_reports_complete = true;
    for (std::size_t index = 0;
         index < tmv::thermodynamic_boundary_report_count; ++index)
    {
        const double value = static_cast<double>(index + 1);
        runtime.thermodynamic_boundary_reports[index] = {
            value, -value, -value + 0.25, 0.25, true};
        runtime.cumulative_thermodynamic_boundary_reports[index] = {
            10.0 * value, -10.0 * value,
            -10.0 * value + 0.5, 0.5, true};

        auto& current = runtime.thermodynamic_moist_energy_reports[index];
        current.dry_internal_change_j = value;
        current.requested_state_change_j = value;
        current.committed_state_change_j = value;
        current.complete = true;
        auto& cumulative =
            runtime.cumulative_thermodynamic_moist_energy_reports[index];
        cumulative.dry_internal_change_j = 10.0 * value;
        cumulative.requested_state_change_j = 10.0 * value;
        cumulative.committed_state_change_j = 10.0 * value;
        cumulative.complete = true;
    }
    runtime.thermodynamic_boundary_report_step_count = 7;
    runtime.thermodynamic_moist_energy_report_step_count = 8;
    runtime.tracer_roundoff_report = {
        2.0e-30, 0.0,
        -4.0f * std::numeric_limits<float>::denorm_min(),
        2, 1, 1, true};
    runtime.cumulative_tracer_roundoff_report = {
        7.0e-30, 0.0,
        -8.0f * std::numeric_limits<float>::denorm_min(),
        5, 3, 2, true};
    runtime.tracer_roundoff_report_step_count = 4;
    runtime.tracer_roundoff_cumulative_history_known = true;
    runtime.hydrometeor_active = {true, false, true, false, true, false};
    runtime.nr = 5;
    runtime.nth = 6;
    runtime.nz = 3;
    runtime.activity_current = true;
    runtime.metadata_consumptions = 9;
    runtime.fallback_scans = 10;

    const std::array<Field3D*, 3> scratch{{
        &runtime.scalar_scratch_a,
        &runtime.scalar_scratch_b,
        &runtime.vertical_tendencies.dqdt_adv,
    }};
    std::array<const float*, 3> pointers{};
    std::array<std::size_t, 3> capacities{};
    for (std::size_t index = 0; index < scratch.size(); ++index)
    {
        pointers[index] = scratch[index]->data();
        capacities[index] = scratch[index]->capacity();
    }
    const auto cumulative_boundary =
        runtime.cumulative_thermodynamic_boundary_reports;
    const auto cumulative_energy =
        runtime.cumulative_thermodynamic_moist_energy_reports;
    const auto tracer_roundoff = runtime.tracer_roundoff_report;
    const auto cumulative_tracer_roundoff =
        runtime.cumulative_tracer_roundoff_report;
    auto* const scheme = runtime.scheme.get();

    SimulationStateView state = make_simulation_state_view(storage);
    std::vector<tmv::domain::HomogeneousRuntimeInvalidationBinding>
        invalidations;
    REQUIRE(tmv::domain::append_advection_domain_recycling_bindings(
        state, invalidations));
    REQUIRE(std::string(
                tmv::domain::advection_domain_recycling_algorithm_id()) ==
            "preserve_transport_history_invalidate_generation_state_v1");

    const auto plan = tmv::domain::plan_whole_cell_shift(5, 6, 3, 1, -1);
    Field3D shifted = indexed_field(5, 6, 3, -1000.0f);
    const Field3D shifted_before = shifted;
    const std::vector<float> leading{-1.0f, -2.0f, -3.0f};
    double invalid_target = 1.0;
    const std::vector<tmv::domain::HomogeneousScalarCommitBinding> bad{{
        "bad", &invalid_target,
        std::numeric_limits<double>::quiet_NaN()}};
    REQUIRE_THROWS_WITH(
        tmv::domain::execute_homogeneous_window_shift(
            plan, {{"shifted", &shifted, &leading}}, {}, bad, {}, {},
            invalidations),
        Catch::Matchers::Contains("non-finite"));
    REQUIRE(runtime.scalar_scratch_a(0, 0, 0) == 100.0f);
    REQUIRE(runtime.thermodynamic_boundary_reports_complete);
    REQUIRE(runtime.activity_current);
    REQUIRE(shifted(2, 2, 1) == shifted_before(2, 2, 1));

    const auto report = tmv::domain::execute_homogeneous_window_shift(
        plan, {{"shifted", &shifted, &leading}}, {}, {}, {}, {},
        invalidations);
    REQUIRE(report.runtime_invalidations_committed == 1);
    require_shifted(shifted_before, shifted, plan, leading);
    for (std::size_t field = 0; field < scratch.size(); ++field)
    {
        REQUIRE(scratch[field]->data() == pointers[field]);
        REQUIRE(scratch[field]->capacity() == capacities[field]);
        for (std::size_t index = 0; index < scratch[field]->size(); ++index)
            REQUIRE(scratch[field]->data()[index] == 0.0f);
    }
    REQUIRE(runtime.vertical_tendencies.fused_dst == nullptr);
    REQUIRE(runtime.vertical_tendencies.fused_src == nullptr);
    REQUIRE(runtime.vertical_tendencies.fused_dt == 0.0f);
    REQUIRE_FALSE(runtime.vertical_tendencies.fused_completed);
    REQUIRE_FALSE(runtime.tracer_advected_in_batch);
    REQUIRE_FALSE(runtime.activity_current);
    for (bool active : runtime.hydrometeor_active)
        REQUIRE_FALSE(active);
    REQUIRE_FALSE(runtime.thermodynamic_boundary_reports_requested);
    REQUIRE_FALSE(runtime.thermodynamic_moist_energy_reports_requested);
    REQUIRE_FALSE(runtime.thermodynamic_boundary_reports_complete);
    REQUIRE_FALSE(runtime.thermodynamic_moist_energy_reports_complete);
    for (const auto& current : runtime.thermodynamic_boundary_reports)
        REQUIRE_FALSE(current.complete);
    for (const auto& current :
         runtime.thermodynamic_moist_energy_reports)
        REQUIRE_FALSE(current.complete);
    for (std::size_t index = 0;
         index < tmv::thermodynamic_boundary_report_count; ++index)
    {
        const auto& actual_boundary =
            runtime.cumulative_thermodynamic_boundary_reports[index];
        const auto& expected_boundary = cumulative_boundary[index];
        REQUIRE(actual_boundary.outward_boundary_inventory ==
                expected_boundary.outward_boundary_inventory);
        REQUIRE(actual_boundary.flux_implied_state_change ==
                expected_boundary.flux_implied_state_change);
        REQUIRE(actual_boundary.committed_state_change ==
                expected_boundary.committed_state_change);
        REQUIRE(actual_boundary.closure_residual ==
                expected_boundary.closure_residual);
        REQUIRE(actual_boundary.complete == expected_boundary.complete);

        const auto& actual_energy =
            runtime.cumulative_thermodynamic_moist_energy_reports[index];
        const auto& expected_energy = cumulative_energy[index];
        REQUIRE(actual_energy.dry_internal_change_j ==
                expected_energy.dry_internal_change_j);
        REQUIRE(actual_energy.water_sensible_change_j ==
                expected_energy.water_sensible_change_j);
        REQUIRE(actual_energy.phase_reference_change_j ==
                expected_energy.phase_reference_change_j);
        REQUIRE(actual_energy.carrier_kinetic_change_j ==
                expected_energy.carrier_kinetic_change_j);
        REQUIRE(actual_energy.gravitational_change_j ==
                expected_energy.gravitational_change_j);
        REQUIRE(actual_energy.subgrid_tke_change_j ==
                expected_energy.subgrid_tke_change_j);
        REQUIRE(actual_energy.density_history_remap_j ==
                expected_energy.density_history_remap_j);
        REQUIRE(actual_energy.outward_boundary_energy_j ==
                expected_energy.outward_boundary_energy_j);
        REQUIRE(actual_energy.interior_conversion_j ==
                expected_energy.interior_conversion_j);
        REQUIRE(actual_energy.limited_face_state_change_j ==
                expected_energy.limited_face_state_change_j);
        REQUIRE(actual_energy.requested_state_change_j ==
                expected_energy.requested_state_change_j);
        REQUIRE(actual_energy.storage_rounding_j ==
                expected_energy.storage_rounding_j);
        REQUIRE(actual_energy.committed_state_change_j ==
                expected_energy.committed_state_change_j);
        REQUIRE(actual_energy.closure_residual_j ==
                expected_energy.closure_residual_j);
        REQUIRE(actual_energy.complete == expected_energy.complete);
    }
    REQUIRE(runtime.thermodynamic_boundary_report_step_count == 7);
    REQUIRE(runtime.thermodynamic_moist_energy_report_step_count == 8);
    REQUIRE(runtime.tracer_roundoff_report.physical_inventory_correction_kg ==
            tracer_roundoff.physical_inventory_correction_kg);
    REQUIRE(runtime.tracer_roundoff_report.thermodynamic_energy_correction_j ==
            tracer_roundoff.thermodynamic_energy_correction_j);
    REQUIRE(runtime.tracer_roundoff_report.most_negative_repaired_value ==
            tracer_roundoff.most_negative_repaired_value);
    REQUIRE(runtime.tracer_roundoff_report.negative_subnormal_count ==
            tracer_roundoff.negative_subnormal_count);
    REQUIRE(runtime.tracer_roundoff_report.negative_zero_count ==
            tracer_roundoff.negative_zero_count);
    REQUIRE(runtime.tracer_roundoff_report.closure_canonicalization_count ==
            tracer_roundoff.closure_canonicalization_count);
    REQUIRE(runtime.tracer_roundoff_report.complete ==
            tracer_roundoff.complete);
    REQUIRE(runtime.cumulative_tracer_roundoff_report
                .physical_inventory_correction_kg ==
            cumulative_tracer_roundoff.physical_inventory_correction_kg);
    REQUIRE(runtime.cumulative_tracer_roundoff_report
                .thermodynamic_energy_correction_j ==
            cumulative_tracer_roundoff.thermodynamic_energy_correction_j);
    REQUIRE(runtime.cumulative_tracer_roundoff_report
                .most_negative_repaired_value ==
            cumulative_tracer_roundoff.most_negative_repaired_value);
    REQUIRE(runtime.cumulative_tracer_roundoff_report
                .negative_subnormal_count ==
            cumulative_tracer_roundoff.negative_subnormal_count);
    REQUIRE(runtime.cumulative_tracer_roundoff_report.negative_zero_count ==
            cumulative_tracer_roundoff.negative_zero_count);
    REQUIRE(runtime.cumulative_tracer_roundoff_report
                .closure_canonicalization_count ==
            cumulative_tracer_roundoff.closure_canonicalization_count);
    REQUIRE(runtime.cumulative_tracer_roundoff_report.complete ==
            cumulative_tracer_roundoff.complete);
    REQUIRE(runtime.tracer_roundoff_report_step_count == 4);
    REQUIRE(runtime.tracer_roundoff_cumulative_history_known);
    REQUIRE(runtime.metadata_consumptions == 9);
    REQUIRE(runtime.fallback_scans == 10);
    REQUIRE(runtime.scheme.get() == scheme);
    REQUIRE(runtime.config.scheme_id == "d5_stub");
}

TEST_CASE("D5 advection rejects mid-step and malformed published state",
          "[core][domain_recycling][advection][rollback][D5]")
{
    SimulationStorageState storage(
        GridStorageState{4, 5, 3, 100.0, 250.0, 0.1, 200.0});
    auto& runtime = storage.runtime.advection;
    runtime.config.scheme_id = "d5_stub";
    runtime.scheme = std::make_unique<D5AdvectionScheme>();
    SimulationStateView state = make_simulation_state_view(storage);
    std::vector<tmv::domain::HomogeneousRuntimeInvalidationBinding>
        invalidations;

    SECTION("outstanding tracer handoff")
    {
        runtime.tracer_advected_in_batch = true;
        REQUIRE_THROWS_WITH(
            tmv::domain::append_advection_domain_recycling_bindings(
                state, invalidations),
            Catch::Matchers::Contains("batched tracer handoff"));
    }
    SECTION("partially published report set")
    {
        runtime.thermodynamic_boundary_reports[0].complete = true;
        REQUIRE_THROWS_WITH(
            tmv::domain::append_advection_domain_recycling_bindings(
                state, invalidations),
            Catch::Matchers::Contains("partially published"));
    }
    SECTION("cumulative ledger count disagrees with publication")
    {
        runtime.thermodynamic_boundary_report_step_count = 1;
        REQUIRE_THROWS_WITH(
            tmv::domain::append_advection_domain_recycling_bindings(
                state, invalidations),
            Catch::Matchers::Contains("cumulative boundary ledger"));
    }
    SECTION("tracer roundoff report has a material minimum")
    {
        runtime.tracer_roundoff_report = {
            1.0e-30, 0.0, -std::numeric_limits<float>::min(),
            1, 0, 0, true};
        runtime.cumulative_tracer_roundoff_report =
            runtime.tracer_roundoff_report;
        runtime.tracer_roundoff_report_step_count = 1;
        REQUIRE_THROWS_WITH(
            tmv::domain::append_advection_domain_recycling_bindings(
                state, invalidations),
            Catch::Matchers::Contains("passive-tracer roundoff ledger"));
    }
    SECTION("tracer cumulative report is behind the last report")
    {
        runtime.tracer_roundoff_report = {
            2.0e-30, 0.0,
            -4.0f * std::numeric_limits<float>::denorm_min(),
            2, 1, 1, true};
        runtime.cumulative_tracer_roundoff_report = {
            1.0e-30, 0.0,
            -2.0f * std::numeric_limits<float>::denorm_min(),
            1, 1, 1, true};
        runtime.tracer_roundoff_report_step_count = 2;
        REQUIRE_THROWS_WITH(
            tmv::domain::append_advection_domain_recycling_bindings(
                state, invalidations),
            Catch::Matchers::Contains("passive-tracer roundoff ledger"));
    }
    SECTION("tracer cumulative completeness disagrees with history")
    {
        runtime.tracer_roundoff_report = {
            0.0, 0.0, 0.0f, 0, 1, 0, true};
        runtime.cumulative_tracer_roundoff_report =
            runtime.tracer_roundoff_report;
        runtime.cumulative_tracer_roundoff_report.complete = false;
        runtime.tracer_roundoff_report_step_count = 1;
        runtime.tracer_roundoff_cumulative_history_known = true;
        REQUIRE_THROWS_WITH(
            tmv::domain::append_advection_domain_recycling_bindings(
                state, invalidations),
            Catch::Matchers::Contains("passive-tracer roundoff ledger"));
    }
    SECTION("legacy tracer history preserves a partial cumulative ledger")
    {
        runtime.tracer_roundoff_report = {
            1.0e-30, 0.0,
            -2.0f * std::numeric_limits<float>::denorm_min(),
            1, 0, 0, true};
        runtime.cumulative_tracer_roundoff_report =
            runtime.tracer_roundoff_report;
        runtime.cumulative_tracer_roundoff_report.complete = false;
        runtime.tracer_roundoff_report_step_count = 1;
        runtime.tracer_roundoff_cumulative_history_known = false;
        REQUIRE(tmv::domain::append_advection_domain_recycling_bindings(
            state, invalidations));
        REQUIRE(invalidations.size() == 1);
        invalidations.clear();
    }
    REQUIRE(invalidations.empty());
}

TEST_CASE("D5 diffusion invalidation preserves cumulative work and allocation",
          "[core][domain_recycling][diffusion][transaction][D5]")
{
    SimulationStorageState storage(
        GridStorageState{5, 6, 3, 100.0, 250.0, 0.1, 200.0});
    storage.grid.geometry.initialize(
        5, 6, 3, 100.0, 250.0, 200.0,
        CoordinateSystem::Cartesian, StaggerType::CGrid);
    auto& runtime = storage.runtime.diffusion;
    runtime.config.scheme_id = "d5_stub";
    runtime.config.K_h = 2.0;
    runtime.config.K_v = 3.0;
    runtime.scheme = std::make_unique<D5DiffusionScheme>();

    const std::array<Field3D*, 5> tendencies{{
        &runtime.tendencies.dudt_diff,
        &runtime.tendencies.dvdt_diff,
        &runtime.tendencies.dwdt_diff,
        &runtime.tendencies.dthetadt_diff,
        &runtime.tendencies.dqvdt_diff,
    }};
    const std::array<Field3D*, 5> candidates{{
        &runtime.scratch_u,
        &runtime.scratch_v,
        &runtime.scratch_w,
        &runtime.scratch_theta,
        &runtime.scratch_qv,
    }};
    for (std::size_t index = 0; index < tendencies.size(); ++index)
    {
        *tendencies[index] = indexed_field(
            5, 6, 3, 1000.0f * static_cast<float>(index + 1));
        *candidates[index] = indexed_field(
            5, 6, 3, 10000.0f * static_cast<float>(index + 1));
    }
    runtime.diagnostics.max_diffusion_number = 0.5;
    runtime.diagnostics.K_effective = indexed_field(5, 6, 3, 1.0f);

    runtime.last_momentum_work = {
        -4.0, 1.0, 0.0, -3.0,
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
        18.0, 0.0, true};
    runtime.cumulative_moist_energy_work = {
        -40.0, 10.0, 0.0, -30.0,
        10.0, 20.0, 30.0, 40.0, 50.0, 60.0,
        180.0, 0.0, true};
    runtime.cumulative_requested_momentum_work_j = -40.0;
    runtime.cumulative_finite_step_momentum_correction_j = 10.0;
    runtime.cumulative_momentum_storage_rounding_j = 0.0;
    runtime.cumulative_committed_momentum_work_j = -30.0;
    runtime.momentum_work_step_count = 10;
    runtime.moist_energy_step_count = 10;

    std::array<Field3D*, 11> generation_fields{{
        tendencies[0], tendencies[1], tendencies[2], tendencies[3],
        tendencies[4], candidates[0], candidates[1], candidates[2],
        candidates[3], candidates[4], &runtime.diagnostics.K_effective,
    }};
    std::array<const float*, 11> pointers{};
    std::array<std::size_t, 11> capacities{};
    for (std::size_t index = 0; index < generation_fields.size(); ++index)
    {
        pointers[index] = generation_fields[index]->data();
        capacities[index] = generation_fields[index]->capacity();
    }
    const auto cumulative_moist = runtime.cumulative_moist_energy_work;
    auto* const scheme = runtime.scheme.get();

    SimulationStateView state = make_simulation_state_view(storage);
    std::vector<tmv::domain::HomogeneousRuntimeInvalidationBinding>
        invalidations;
    REQUIRE(tmv::domain::append_diffusion_domain_recycling_bindings(
        state, invalidations));
    REQUIRE(std::string(
                tmv::domain::diffusion_domain_recycling_algorithm_id()) ==
            "preserve_cumulative_work_invalidate_diffusion_generation_v1");

    const auto plan = tmv::domain::plan_whole_cell_shift(5, 6, 3, -1, 1);
    Field3D shifted = indexed_field(5, 6, 3, -1000.0f);
    const Field3D shifted_before = shifted;
    const std::vector<float> leading{-1.0f, -2.0f, -3.0f};
    double invalid_target = 1.0;
    const std::vector<tmv::domain::HomogeneousScalarCommitBinding> bad{{
        "bad", &invalid_target,
        std::numeric_limits<double>::quiet_NaN()}};
    REQUIRE_THROWS_WITH(
        tmv::domain::execute_homogeneous_window_shift(
            plan, {{"shifted", &shifted, &leading}}, {}, bad, {}, {},
            invalidations),
        Catch::Matchers::Contains("non-finite"));
    REQUIRE(runtime.tendencies.dudt_diff(0, 0, 0) == 1000.0f);
    REQUIRE(runtime.scratch_theta(0, 0, 0) == 40000.0f);
    REQUIRE(runtime.diagnostics.max_diffusion_number == 0.5);
    REQUIRE(runtime.last_momentum_work.committed_work_j == -3.0);
    REQUIRE(shifted(2, 2, 1) == shifted_before(2, 2, 1));

    const auto report = tmv::domain::execute_homogeneous_window_shift(
        plan, {{"shifted", &shifted, &leading}}, {}, {}, {}, {},
        invalidations);
    REQUIRE(report.runtime_invalidations_committed == 1);
    require_shifted(shifted_before, shifted, plan, leading);
    for (std::size_t field = 0; field < generation_fields.size(); ++field)
    {
        REQUIRE(generation_fields[field]->data() == pointers[field]);
        REQUIRE(generation_fields[field]->capacity() == capacities[field]);
        for (std::size_t index = 0;
             index < generation_fields[field]->size(); ++index)
            REQUIRE(generation_fields[field]->data()[index] == 0.0f);
    }
    REQUIRE(runtime.diagnostics.max_diffusion_number == 0.0);
    require_diffusion_report_equal(
        runtime.last_momentum_work,
        tmv::DiffusionMomentumWorkReport{});
    require_diffusion_report_equal(
        runtime.cumulative_moist_energy_work, cumulative_moist);
    REQUIRE(runtime.cumulative_requested_momentum_work_j == -40.0);
    REQUIRE(runtime.cumulative_finite_step_momentum_correction_j == 10.0);
    REQUIRE(runtime.cumulative_momentum_storage_rounding_j == 0.0);
    REQUIRE(runtime.cumulative_committed_momentum_work_j == -30.0);
    REQUIRE(runtime.momentum_work_step_count == 10);
    REQUIRE(runtime.moist_energy_step_count == 10);
    REQUIRE(runtime.scheme.get() == scheme);
    REQUIRE(runtime.config.scheme_id == "d5_stub");
}

TEST_CASE("D5 diffusion rejects malformed published and cumulative state",
          "[core][domain_recycling][diffusion][rollback][D5]")
{
    SimulationStorageState storage(
        GridStorageState{4, 5, 3, 100.0, 250.0, 0.1, 200.0});
    auto& runtime = storage.runtime.diffusion;
    runtime.config.scheme_id = "d5_stub";
    runtime.scheme = std::make_unique<D5DiffusionScheme>();
    SimulationStateView state = make_simulation_state_view(storage);
    std::vector<tmv::domain::HomogeneousRuntimeInvalidationBinding>
        invalidations;

    SECTION("partial tendency publication")
    {
        runtime.tendencies.dudt_diff = indexed_field(4, 5, 3);
        REQUIRE_THROWS_WITH(
            tmv::domain::append_diffusion_domain_recycling_bindings(
                state, invalidations),
            Catch::Matchers::Contains("partially allocated"));
    }
    SECTION("non-finite old-grid transaction candidate")
    {
        runtime.scratch_theta = indexed_field(4, 5, 3);
        runtime.scratch_theta(1, 2, 1) =
            std::numeric_limits<float>::quiet_NaN();
        REQUIRE_THROWS_WITH(
            tmv::domain::append_diffusion_domain_recycling_bindings(
                state, invalidations),
            Catch::Matchers::Contains("non-finite"));
    }
    SECTION("stale effective coefficient shape")
    {
        runtime.diagnostics.K_effective.resize(1, 1, 1, 1.0f);
        REQUIRE_THROWS_WITH(
            tmv::domain::append_diffusion_domain_recycling_bindings(
                state, invalidations),
            Catch::Matchers::Contains("shape"));
    }
    SECTION("invalid completed work report")
    {
        runtime.last_momentum_work.finite_step_correction_j = -1.0;
        REQUIRE_THROWS_WITH(
            tmv::domain::append_diffusion_domain_recycling_bindings(
                state, invalidations),
            Catch::Matchers::Contains("work history"));
    }
    SECTION("nonzero cumulative work with zero count")
    {
        runtime.cumulative_requested_momentum_work_j = 1.0;
        REQUIRE_THROWS_WITH(
            tmv::domain::append_diffusion_domain_recycling_bindings(
                state, invalidations),
            Catch::Matchers::Contains("cumulative work ledger"));
    }
    REQUIRE(invalidations.empty());
}

TEST_CASE("D5 time-stepping invalidation preserves completed histories",
          "[core][domain_recycling][time_stepping][transaction][D5]")
{
    SimulationStorageState storage(
        GridStorageState{5, 6, 3, 100.0, 250.0, 0.1, 200.0});
    storage.grid.geometry.initialize(
        5, 6, 3, 100.0, 250.0, 200.0,
        CoordinateSystem::Cartesian, StaggerType::CGrid);
    auto& runtime = storage.runtime.time_stepping;
    runtime.config.scheme_id = "d5_stub";
    runtime.config.max_acoustic_speed_ms = 350.0;
    runtime.config.min_acoustic_spacing_m = 100.0;
    runtime.scheme = std::make_unique<D5TimeSteppingScheme>();
    auto* const scheme =
        static_cast<D5TimeSteppingScheme*>(runtime.scheme.get());
    runtime.diagnostics.n_steps = 2;
    runtime.diagnostics.max_cfl = 0.5;
    runtime.diagnostics.cpu_time_per_step = 0.01;
    runtime.diagnostics.dt_history = {1.0, 0.5};

    const std::array<Field3D*, 11> workspace{{
        &runtime.slow_rk2.u_save,
        &runtime.slow_rk2.v_save,
        &runtime.slow_rk2.w_save,
        &runtime.slow_rk2.p_save,
        &runtime.slow_rk2.du_dt_k1,
        &runtime.slow_rk2.dv_dt_k1,
        &runtime.slow_rk2.dw_dt_k1,
        &runtime.slow_rk2.dp_dt_k1,
        &runtime.slow_rk2.du_advection_dt_k1,
        &runtime.slow_rk2.dv_advection_dt_k1,
        &runtime.slow_rk2.dw_advection_dt_k1,
    }};
    for (std::size_t index = 0; index < workspace.size(); ++index)
        *workspace[index] = indexed_field(
            5, 6, 3, 1000.0f * static_cast<float>(index + 1));
    std::array<const float*, 11> pointers{};
    std::array<std::size_t, 11> capacities{};
    for (std::size_t index = 0; index < workspace.size(); ++index)
    {
        pointers[index] = workspace[index]->data();
        capacities[index] = workspace[index]->capacity();
    }

    runtime.last_boundary_mass = {10.0, -10.0, -9.0, 1.0};
    runtime.cumulative_outward_boundary_mass_kg = 10.0;
    runtime.cumulative_flux_implied_state_change_kg = -10.0;
    runtime.cumulative_committed_state_change_kg = -9.0;
    runtime.cumulative_boundary_mass_closure_residual_kg = 1.0;
    runtime.boundary_mass_step_count = 1;

    runtime.last_acoustic_energy.complete = true;
    runtime.last_acoustic_energy.kinetic_transport_complete = true;
    runtime.last_acoustic_energy.substep_count = 2;
    runtime.last_acoustic_energy.boundary_complete = true;
    runtime.last_acoustic_energy.boundary_application_count = 1;
    runtime.cumulative_acoustic_energy = runtime.last_acoustic_energy;
    runtime.acoustic_energy_step_count = 1;

    runtime.last_slow_momentum_energy.complete = true;
    runtime.cumulative_slow_momentum_energy =
        runtime.last_slow_momentum_energy;
    runtime.slow_momentum_energy_step_count = 1;

    runtime.last_eos_pressure_energy.complete = true;
    runtime.cumulative_eos_pressure_energy =
        runtime.last_eos_pressure_energy;
    runtime.eos_pressure_energy_step_count = 1;

    const auto complete_sponge = [](tmv::SpongeEnergyReportV1& report)
    {
        report.upper.complete = true;
        report.w_only.complete = true;
        report.lateral.complete = true;
        report.complete = true;
    };
    complete_sponge(runtime.last_sponge_energy);
    runtime.cumulative_sponge_energy = runtime.last_sponge_energy;
    runtime.sponge_energy_step_count = 1;

    runtime.last_final_boundary_energy.applied = true;
    runtime.last_final_boundary_energy.complete = true;
    runtime.cumulative_final_boundary_energy =
        runtime.last_final_boundary_energy;
    runtime.final_boundary_energy_step_count = 1;

    runtime.last_completed_step_energy.complete = true;
    runtime.cumulative_completed_step_energy =
        runtime.last_completed_step_energy;
    runtime.completed_step_energy_count = 1;

    const auto cumulative_acoustic = runtime.cumulative_acoustic_energy;
    const auto cumulative_slow = runtime.cumulative_slow_momentum_energy;
    const auto cumulative_eos = runtime.cumulative_eos_pressure_energy;
    const auto cumulative_sponge = runtime.cumulative_sponge_energy;
    const auto cumulative_final = runtime.cumulative_final_boundary_energy;
    const auto cumulative_completed =
        runtime.cumulative_completed_step_energy;
    const auto diagnostics = runtime.diagnostics;

    SimulationStateView state = make_simulation_state_view(storage);
    std::vector<tmv::domain::HomogeneousRuntimeInvalidationBinding>
        invalidations;
    REQUIRE(tmv::domain::append_time_stepping_domain_recycling_bindings(
        state, invalidations));
    REQUIRE(std::string(
                tmv::domain::time_stepping_domain_recycling_algorithm_id()) ==
            "preserve_completed_ledgers_invalidate_timestep_generation_v1");

    const auto plan = tmv::domain::plan_whole_cell_shift(5, 6, 3, 1, 0);
    Field3D shifted = indexed_field(5, 6, 3, -1000.0f);
    const Field3D shifted_before = shifted;
    const std::vector<float> leading{-1.0f, -2.0f, -3.0f};
    double invalid_target = 1.0;
    const std::vector<tmv::domain::HomogeneousScalarCommitBinding> bad{{
        "bad", &invalid_target,
        std::numeric_limits<double>::quiet_NaN()}};
    REQUIRE_THROWS_WITH(
        tmv::domain::execute_homogeneous_window_shift(
            plan, {{"shifted", &shifted, &leading}}, {}, bad, {}, {},
            invalidations),
        Catch::Matchers::Contains("non-finite"));
    REQUIRE(runtime.slow_rk2.u_save(0, 0, 0) == 1000.0f);
    REQUIRE(runtime.last_acoustic_energy.complete);
    REQUIRE(runtime.config.max_acoustic_speed_ms == 350.0);
    REQUIRE_FALSE(scheme->generation_invalidated);
    REQUIRE(shifted(2, 2, 1) == shifted_before(2, 2, 1));

    const auto report = tmv::domain::execute_homogeneous_window_shift(
        plan, {{"shifted", &shifted, &leading}}, {}, {}, {}, {},
        invalidations);
    REQUIRE(report.runtime_invalidations_committed == 1);
    require_shifted(shifted_before, shifted, plan, leading);
    for (std::size_t field = 0; field < workspace.size(); ++field)
    {
        REQUIRE(workspace[field]->data() == pointers[field]);
        REQUIRE(workspace[field]->capacity() == capacities[field]);
        for (std::size_t index = 0; index < workspace[field]->size(); ++index)
            REQUIRE(workspace[field]->data()[index] == 0.0f);
    }
    REQUIRE(scheme->generation_invalidated);
    REQUIRE(runtime.config.max_acoustic_speed_ms == 0.0);
    REQUIRE(runtime.config.min_acoustic_spacing_m == 0.0);
    REQUIRE_FALSE(runtime.last_acoustic_energy.complete);
    REQUIRE_FALSE(runtime.last_slow_momentum_energy.complete);
    REQUIRE_FALSE(runtime.last_eos_pressure_energy.complete);
    REQUIRE_FALSE(runtime.last_sponge_energy.complete);
    REQUIRE_FALSE(runtime.last_final_boundary_energy.complete);
    REQUIRE_FALSE(runtime.last_completed_step_energy.complete);
    REQUIRE(runtime.last_boundary_mass.outward_boundary_mass_kg == 0.0);

    REQUIRE(runtime.cumulative_acoustic_energy.complete ==
            cumulative_acoustic.complete);
    REQUIRE(runtime.cumulative_acoustic_energy.substep_count ==
            cumulative_acoustic.substep_count);
    REQUIRE(runtime.cumulative_slow_momentum_energy.complete ==
            cumulative_slow.complete);
    REQUIRE(runtime.cumulative_eos_pressure_energy.complete ==
            cumulative_eos.complete);
    REQUIRE(runtime.cumulative_sponge_energy.complete ==
            cumulative_sponge.complete);
    REQUIRE(runtime.cumulative_final_boundary_energy.complete ==
            cumulative_final.complete);
    REQUIRE(runtime.cumulative_completed_step_energy.complete ==
            cumulative_completed.complete);
    REQUIRE(runtime.acoustic_energy_step_count == 1);
    REQUIRE(runtime.slow_momentum_energy_step_count == 1);
    REQUIRE(runtime.eos_pressure_energy_step_count == 1);
    REQUIRE(runtime.sponge_energy_step_count == 1);
    REQUIRE(runtime.final_boundary_energy_step_count == 1);
    REQUIRE(runtime.completed_step_energy_count == 1);
    REQUIRE(runtime.boundary_mass_step_count == 1);
    REQUIRE(runtime.cumulative_outward_boundary_mass_kg == 10.0);
    REQUIRE(runtime.cumulative_committed_state_change_kg == -9.0);
    REQUIRE(runtime.diagnostics.n_steps == diagnostics.n_steps);
    REQUIRE(runtime.diagnostics.dt_history == diagnostics.dt_history);
}

TEST_CASE("D5 time-stepping rejects partial, invalid, and unsupported state",
          "[core][domain_recycling][time_stepping][rollback][D5]")
{
    SimulationStorageState storage(
        GridStorageState{4, 5, 3, 100.0, 250.0, 0.1, 200.0});
    auto& runtime = storage.runtime.time_stepping;
    runtime.config.scheme_id = "d5_stub";
    runtime.scheme = std::make_unique<D5TimeSteppingScheme>();
    SimulationStateView state = make_simulation_state_view(storage);
    std::vector<tmv::domain::HomogeneousRuntimeInvalidationBinding>
        invalidations;

    SECTION("partial primary workspace")
    {
        runtime.slow_rk2.u_save = indexed_field(4, 5, 3);
        REQUIRE_THROWS_WITH(
            tmv::domain::append_time_stepping_domain_recycling_bindings(
                state, invalidations),
            Catch::Matchers::Contains("partially allocated"));
    }
    SECTION("invalid timestep history")
    {
        runtime.diagnostics.dt_history = {1.0, 0.0};
        REQUIRE_THROWS_WITH(
            tmv::domain::append_time_stepping_domain_recycling_bindings(
                state, invalidations),
            Catch::Matchers::Contains("diagnostic history"));
    }
    SECTION("scheme lacks recycling capability")
    {
        runtime.scheme = std::make_unique<D5TimeSteppingScheme>(false);
        REQUIRE_THROWS_WITH(
            tmv::domain::append_time_stepping_domain_recycling_bindings(
                state, invalidations),
            Catch::Matchers::Contains("does not implement"));
    }
    SECTION("cumulative report count is inconsistent")
    {
        runtime.acoustic_energy_step_count = 1;
        REQUIRE_THROWS_WITH(
            tmv::domain::append_time_stepping_domain_recycling_bindings(
                state, invalidations),
            Catch::Matchers::Contains("energy history"));
    }
    SECTION("last boundary-mass report violates its sign identity")
    {
        runtime.last_boundary_mass = {1.0, 1.0, 0.0, -1.0};
        REQUIRE_THROWS_WITH(
            tmv::domain::append_time_stepping_domain_recycling_bindings(
                state, invalidations),
            Catch::Matchers::Contains("last boundary-mass report"));
    }
    SECTION("cumulative boundary-mass ledger violates its sign identity")
    {
        runtime.cumulative_outward_boundary_mass_kg = 1.0;
        runtime.cumulative_flux_implied_state_change_kg = 1.0;
        runtime.boundary_mass_step_count = 1;
        REQUIRE_THROWS_WITH(
            tmv::domain::append_time_stepping_domain_recycling_bindings(
                state, invalidations),
            Catch::Matchers::Contains("cumulative boundary-mass ledger"));
    }
    REQUIRE(invalidations.empty());
}

TEST_CASE("D5 radar products and private candidates invalidate atomically",
          "[core][domain_recycling][radar][transaction][D5]")
{
    SimulationStorageState storage(
        GridStorageState{4, 5, 3, 100.0, 250.0, 0.1, 200.0});
    auto& runtime = storage.runtime.radar;
    runtime.selected_config.scheme_id = "d5_radar";
    runtime.selected_config.radar_x = 1000.0;
    runtime.selected_config.radar_y = -2000.0;
    runtime.selected_config.radar_z = 300.0;
    runtime.nr = 4;
    runtime.nth = 5;
    runtime.nz = 3;
    runtime.radar_x = 1000.0;
    runtime.radar_y = -2000.0;
    runtime.radar_z = 300.0;

    auto selected = std::make_unique<D5RadarScheme>();
    auto reflectivity = std::make_unique<D5RadarScheme>("reflectivity");
    auto velocity = std::make_unique<D5RadarScheme>("velocity");
    auto zdr = std::make_unique<D5RadarScheme>("zdr");
    D5RadarScheme* selected_ptr = selected.get();
    D5RadarScheme* reflectivity_ptr = reflectivity.get();
    D5RadarScheme* velocity_ptr = velocity.get();
    D5RadarScheme* zdr_ptr = zdr.get();
    runtime.selected_scheme = std::move(selected);
    runtime.reflectivity_operator = std::move(reflectivity);
    runtime.velocity_operator = std::move(velocity);
    runtime.zdr_operator = std::move(zdr);
    runtime.selected_output.initialize(4, 5, 3);
    runtime.polarimetric_output.initialize(4, 5, 3);
    const auto output_fields = [](tmv::RadarOut& output)
    {
        return std::array<Field3D*, 13>{{
            &output.Z_dBZ, &output.Vr, &output.ZDR_dB,
            &output.KDP_deg_km, &output.rhohv,
            &output.Ze_linear, &output.ZH_dBZ, &output.ZV_dBZ,
            &output.Ze_rain, &output.Ze_snow, &output.Ze_graupel,
            &output.Ze_hail, &output.Ze_ice,
        }};
    };
    auto selected_fields = output_fields(runtime.selected_output);
    auto polarimetric_fields = output_fields(runtime.polarimetric_output);
    std::array<const float*, 26> pointers{};
    std::array<std::size_t, 26> capacities{};
    for (std::size_t index = 0; index < selected_fields.size(); ++index)
    {
        selected_fields[index]->fill(1.0f);
        polarimetric_fields[index]->fill(2.0f);
        pointers[index] = selected_fields[index]->data();
        capacities[index] = selected_fields[index]->capacity();
        pointers[index + selected_fields.size()] = polarimetric_fields[index]->data();
        capacities[index + selected_fields.size()] =
            polarimetric_fields[index]->capacity();
    }

    SimulationStateView state = make_simulation_state_view(storage);
    std::vector<tmv::domain::HomogeneousRuntimeInvalidationBinding>
        invalidations;
    REQUIRE(tmv::domain::append_radar_domain_recycling_bindings(
        state, invalidations));
    REQUIRE(std::string(tmv::domain::radar_domain_recycling_algorithm_id()) ==
            "preserve_radar_operators_invalidate_generation_products_v1");

    const auto plan = tmv::domain::plan_whole_cell_shift(4, 5, 3, 1, 0);
    double invalid_target = 1.0;
    const std::vector<tmv::domain::HomogeneousScalarCommitBinding> bad{{
        "bad", &invalid_target,
        std::numeric_limits<double>::quiet_NaN()}};
    REQUIRE_THROWS_WITH(
        tmv::domain::execute_homogeneous_window_shift(
            plan, {}, {}, bad, {}, {}, invalidations),
        Catch::Matchers::Contains("non-finite"));
    REQUIRE(runtime.selected_output.Z_dBZ(0, 0, 0) == 1.0f);
    REQUIRE_FALSE(selected_ptr->generation_invalidated);

    const auto report = tmv::domain::execute_homogeneous_window_shift(
        plan, {}, {}, {}, {}, {}, invalidations);
    REQUIRE(report.runtime_invalidations_committed == 1);
    for (std::size_t index = 0; index < selected_fields.size(); ++index)
    {
        REQUIRE(selected_fields[index]->data() == pointers[index]);
        REQUIRE(selected_fields[index]->capacity() == capacities[index]);
        REQUIRE(polarimetric_fields[index]->data() ==
                pointers[index + selected_fields.size()]);
        REQUIRE(polarimetric_fields[index]->capacity() ==
                capacities[index + selected_fields.size()]);
        for (std::size_t cell = 0;
             cell < selected_fields[index]->size(); ++cell)
            REQUIRE(selected_fields[index]->data()[cell] ==
                    (selected_fields[index] ==
                         &runtime.selected_output.rhohv ? 1.0f : 0.0f));
        for (std::size_t cell = 0;
             cell < polarimetric_fields[index]->size(); ++cell)
            REQUIRE(polarimetric_fields[index]->data()[cell] ==
                    (polarimetric_fields[index] ==
                         &runtime.polarimetric_output.rhohv ? 1.0f : 0.0f));
    }
    REQUIRE(selected_ptr->generation_invalidated);
    REQUIRE(reflectivity_ptr->generation_invalidated);
    REQUIRE(velocity_ptr->generation_invalidated);
    REQUIRE(zdr_ptr->generation_invalidated);
    REQUIRE(runtime.selected_scheme.get() == selected_ptr);
    REQUIRE(runtime.reflectivity_operator.get() == reflectivity_ptr);
    REQUIRE(runtime.velocity_operator.get() == velocity_ptr);
    REQUIRE(runtime.zdr_operator.get() == zdr_ptr);
    REQUIRE(runtime.selected_config.scheme_id == "d5_radar");
    REQUIRE(runtime.radar_x == 1000.0);
    REQUIRE(runtime.radar_y == -2000.0);
    REQUIRE(runtime.radar_z == 300.0);
    REQUIRE(runtime.nr == 4);
    REQUIRE(runtime.nth == 5);
    REQUIRE(runtime.nz == 3);
}

TEST_CASE("D5 radar rejects malformed and unsupported runtime generations",
          "[core][domain_recycling][radar][rollback][D5]")
{
    SimulationStorageState storage(
        GridStorageState{4, 5, 3, 100.0, 250.0, 0.1, 200.0});
    auto& runtime = storage.runtime.radar;
    runtime.selected_config.scheme_id = "d5_radar";
    runtime.selected_scheme = std::make_unique<D5RadarScheme>();
    SimulationStateView state = make_simulation_state_view(storage);
    std::vector<tmv::domain::HomogeneousRuntimeInvalidationBinding>
        invalidations;

    SECTION("non-finite instrument coordinate")
    {
        runtime.radar_x = std::numeric_limits<double>::quiet_NaN();
        REQUIRE_THROWS_WITH(
            tmv::domain::append_radar_domain_recycling_bindings(
                state, invalidations),
            Catch::Matchers::Contains("instrument geometry"));
    }
    SECTION("selected output has no selected operator")
    {
        runtime.selected_scheme.reset();
        runtime.selected_output.initialize(4, 5, 3);
        REQUIRE_THROWS_WITH(
            tmv::domain::append_radar_domain_recycling_bindings(
                state, invalidations),
            Catch::Matchers::Contains("without its operator"));
    }
    SECTION("selected operator lacks recycling capability")
    {
        runtime.selected_scheme =
            std::make_unique<D5RadarScheme>("d5_radar", false);
        REQUIRE_THROWS_WITH(
            tmv::domain::append_radar_domain_recycling_bindings(
                state, invalidations),
            Catch::Matchers::Contains("does not implement"));
    }
    SECTION("product dimensions disagree with the grid")
    {
        runtime.selected_output.initialize(1, 1, 1);
        REQUIRE_THROWS_WITH(
            tmv::domain::append_radar_domain_recycling_bindings(
                state, invalidations),
            Catch::Matchers::Contains("metadata"));
    }
    SECTION("product contains a non-finite value")
    {
        runtime.selected_output.initialize(4, 5, 3);
        runtime.selected_output.Z_dBZ(1, 1, 1) =
            std::numeric_limits<float>::quiet_NaN();
        REQUIRE_THROWS_WITH(
            tmv::domain::append_radar_domain_recycling_bindings(
                state, invalidations),
            Catch::Matchers::Contains("non-finite product"));
    }
    SECTION("product exceeds its physical operator range")
    {
        runtime.selected_output.initialize(4, 5, 3);
        runtime.selected_output.Ze_linear(1, 1, 1) = 1.1e12f;
        REQUIRE_THROWS_WITH(
            tmv::domain::append_radar_domain_recycling_bindings(
                state, invalidations),
            Catch::Matchers::Contains("out-of-contract product"));
    }
    REQUIRE(invalidations.empty());
}

TEST_CASE("D5 lazy derived volumes invalidate allocation-stably and atomically",
          "[core][domain_recycling][diagnostics][transaction][D5]")
{
    SimulationStorageState storage(
        GridStorageState{4, 5, 3, 100.0, 250.0, 0.1, 200.0});
    auto fields = std::array<Field3D*, 12>{{
        &storage.diagnostics.radar_reflectivity,
        &storage.diagnostics.vorticity_r,
        &storage.diagnostics.vorticity_theta,
        &storage.diagnostics.vorticity_z,
        &storage.diagnostics.stretching_term,
        &storage.diagnostics.tilting_term,
        &storage.diagnostics.baroclinic_term,
        &storage.diagnostics.angular_momentum,
        &storage.diagnostics.angular_momentum_tendency,
        &storage.diagnostics.p_prime,
        &storage.diagnostics.dynamic_pressure,
        &storage.diagnostics.buoyancy_pressure,
    }};
    std::array<const float*, 12> pointers{};
    std::array<std::size_t, 12> capacities{};
    for (std::size_t index = 0; index < fields.size(); ++index)
    {
        fields[index]->resize(4, 5, 3, static_cast<float>(index + 1));
        pointers[index] = fields[index]->data();
        capacities[index] = fields[index]->capacity();
    }
    // A stale failed diagnostic is repairable generation-local state. The
    // recycle clears it rather than blocking atmospheric progress.
    storage.diagnostics.tilting_term(1, 1, 1) =
        std::numeric_limits<float>::quiet_NaN();

    SimulationStateView state = make_simulation_state_view(storage);
    std::vector<tmv::domain::HomogeneousRuntimeInvalidationBinding>
        invalidations;
    REQUIRE(
        tmv::domain::append_derived_diagnostic_domain_recycling_bindings(
            state, invalidations));
    REQUIRE(std::string(
                tmv::domain::derived_diagnostic_domain_recycling_algorithm_id()) ==
            "invalidate_lazy_derived_volume_generation_v1");

    const auto plan = tmv::domain::plan_whole_cell_shift(4, 5, 3, 1, 0);
    double invalid_target = 1.0;
    const std::vector<tmv::domain::HomogeneousScalarCommitBinding> bad{{
        "bad", &invalid_target,
        std::numeric_limits<double>::quiet_NaN()}};
    REQUIRE_THROWS_WITH(
        tmv::domain::execute_homogeneous_window_shift(
            plan, {}, {}, bad, {}, {}, invalidations),
        Catch::Matchers::Contains("non-finite"));
    REQUIRE(fields[0]->data()[0] == 1.0f);
    REQUIRE(std::isnan(storage.diagnostics.tilting_term(1, 1, 1)));

    const auto report = tmv::domain::execute_homogeneous_window_shift(
        plan, {}, {}, {}, {}, {}, invalidations);
    REQUIRE(report.runtime_invalidations_committed == 1);
    for (std::size_t index = 0; index < fields.size(); ++index)
    {
        REQUIRE(fields[index]->data() == pointers[index]);
        REQUIRE(fields[index]->capacity() == capacities[index]);
        for (std::size_t cell = 0; cell < fields[index]->size(); ++cell)
            REQUIRE(fields[index]->data()[cell] == 0.0f);
    }
}

TEST_CASE("D5 lazy derived-volume preflight rejects unsafe group layouts",
          "[core][domain_recycling][diagnostics][rollback][D5]")
{
    SimulationStorageState storage(
        GridStorageState{4, 5, 3, 100.0, 250.0, 0.1, 200.0});
    SimulationStateView state = make_simulation_state_view(storage);
    std::vector<tmv::domain::HomogeneousRuntimeInvalidationBinding>
        invalidations;

    SECTION("owning diagnostic storage is absent")
    {
        state.diagnostics = nullptr;
        REQUIRE_THROWS_WITH(
            tmv::domain::append_derived_diagnostic_domain_recycling_bindings(
                state, invalidations),
            Catch::Matchers::Contains("owning storage"));
    }
    SECTION("vorticity publication is partial")
    {
        storage.diagnostics.vorticity_r.resize(4, 5, 3, 1.0f);
        REQUIRE_THROWS_WITH(
            tmv::domain::append_derived_diagnostic_domain_recycling_bindings(
                state, invalidations),
            Catch::Matchers::Contains("publication is partial"));
    }
    SECTION("pressure publication has a wrong-grid member")
    {
        storage.diagnostics.p_prime.resize(4, 5, 3, 1.0f);
        storage.diagnostics.dynamic_pressure.resize(1, 1, 1, 1.0f);
        storage.diagnostics.buoyancy_pressure.resize(4, 5, 3, 1.0f);
        REQUIRE_THROWS_WITH(
            tmv::domain::append_derived_diagnostic_domain_recycling_bindings(
                state, invalidations),
            Catch::Matchers::Contains("does not match the recycling grid"));
    }
    REQUIRE(invalidations.empty());
}

TEST_CASE("D5 dynamics generation state invalidates atomically and allocation-stably",
          "[core][domain_recycling][dynamics][transaction][D5]")
{
    SimulationStorageState storage(
        GridStorageState{4, 5, 3, 100.0, 250.0, 0.1, 200.0});
    storage.grid.geometry.initialize(
        4, 5, 3, 100.0, 200.0, 250.0, CoordinateSystem::Cartesian);
    storage.grid.geometry.staggered = true;
    // The unqualified v1 selector resolves from the live staggering exactly
    // as the production factory does.
    storage.configuration.dynamics_scheme_name =
        "klemp_wilhelmson_split";
    storage.runtime.terrain.config.scheme_id = "none";
    auto scheme = std::make_unique<D5DynamicsScheme>();
    D5DynamicsScheme* scheme_ptr = scheme.get();
    auto& runtime = storage.runtime.dynamics;
    runtime.scheme = std::move(scheme);

    const std::array<Field3D*, 8> fields{{
        &runtime.du_dt, &runtime.dv_dt, &runtime.dw_dt,
        &runtime.drho_dt, &runtime.dp_dt,
        &runtime.du_advection_dt, &runtime.dv_advection_dt,
        &runtime.dw_advection_dt,
    }};
    std::array<const float*, 8> pointers{};
    std::array<std::size_t, 8> capacities{};
    for (std::size_t index = 0; index < fields.size(); ++index)
    {
        fields[index]->resize(4, 5, 3, static_cast<float>(index + 1));
        pointers[index] = fields[index]->data();
        capacities[index] = fields[index]->capacity();
    }
    fields[2]->data()[4] = std::numeric_limits<float>::quiet_NaN();
    runtime.slow_pressure_energy_columns.assign(20, {1.0, 2.0, 3.0});
    runtime.state_relaxation_profiles.assign(21, 4.0f);
    runtime.acoustic_profile_rows.assign(tmv::DynamicsExecutionWorkspace::acoustic_profile_row_count * 3, 5.0f);
    const std::array<Field3D*, 4> transport_fields{{
        &runtime.transport_context.density_begin,
        &runtime.transport_context.mass_flux_x,
        &runtime.transport_context.mass_flux_y,
        &runtime.transport_context.mass_flux_z,
    }};
    std::array<const float*, 4> transport_pointers{};
    for (std::size_t index = 0; index < transport_fields.size(); ++index)
    {
        transport_fields[index]->resize(
            4, 5, 3, static_cast<float>(10 + index));
        transport_pointers[index] = transport_fields[index]->data();
    }
    const auto* slow_pointer = runtime.slow_pressure_energy_columns.data();
    const auto* relaxation_pointer = runtime.state_relaxation_profiles.data();
    const auto* acoustic_pointer = runtime.acoustic_profile_rows.data();

    SimulationStateView state = make_simulation_state_view(storage);
    std::vector<tmv::domain::HomogeneousRuntimeInvalidationBinding>
        invalidations;
    REQUIRE(tmv::domain::append_dynamics_domain_recycling_bindings(
        state, invalidations));
    REQUIRE(std::string(tmv::domain::dynamics_domain_recycling_algorithm_id()) ==
            "preserve_operator_invalidate_dynamics_generation_state_v1");

    const auto plan = tmv::domain::plan_whole_cell_shift(4, 5, 3, 1, 0);
    double invalid_target = 1.0;
    const std::vector<tmv::domain::HomogeneousScalarCommitBinding> bad{{
        "bad", &invalid_target,
        std::numeric_limits<double>::quiet_NaN()}};
    REQUIRE_THROWS_WITH(
        tmv::domain::execute_homogeneous_window_shift(
            plan, {}, {}, bad, {}, {}, invalidations),
        Catch::Matchers::Contains("non-finite"));
    REQUIRE_FALSE(scheme_ptr->invalidated);
    REQUIRE(std::isnan(fields[2]->data()[4]));

    const auto report = tmv::domain::execute_homogeneous_window_shift(
        plan, {}, {}, {}, {}, {}, invalidations);
    REQUIRE(report.runtime_invalidations_committed == 1);
    REQUIRE(scheme_ptr->invalidated);
    for (std::size_t index = 0; index < fields.size(); ++index)
    {
        REQUIRE(fields[index]->data() == pointers[index]);
        REQUIRE(fields[index]->capacity() == capacities[index]);
        for (std::size_t cell = 0; cell < fields[index]->size(); ++cell)
            REQUIRE(fields[index]->data()[cell] == 0.0f);
    }
    REQUIRE(runtime.slow_pressure_energy_columns.data() == slow_pointer);
    REQUIRE(runtime.state_relaxation_profiles.data() == relaxation_pointer);
    REQUIRE(runtime.acoustic_profile_rows.data() == acoustic_pointer);
    for (const auto& partial : runtime.slow_pressure_energy_columns)
        REQUIRE(partial == std::array<double, 3>{});
    for (float value : runtime.state_relaxation_profiles)
        REQUIRE(value == 0.0f);
    for (float value : runtime.acoustic_profile_rows)
        REQUIRE(value == 0.0f);
    for (std::size_t index = 0; index < transport_fields.size(); ++index)
    {
        REQUIRE(transport_fields[index]->data() == transport_pointers[index]);
        for (std::size_t cell = 0; cell < transport_fields[index]->size(); ++cell)
            REQUIRE(transport_fields[index]->data()[cell] == 0.0f);
    }
}

TEST_CASE("D5 dynamics preflight rejects ambiguous ownership and stale scratch",
          "[core][domain_recycling][dynamics][rollback][D5]")
{
    SimulationStorageState storage(
        GridStorageState{4, 5, 3, 100.0, 250.0, 0.1, 200.0});
    storage.grid.geometry.initialize(
        4, 5, 3, 100.0, 200.0, 250.0, CoordinateSystem::Cartesian);
    storage.grid.geometry.staggered = true;
    storage.configuration.dynamics_scheme_name =
        "klemp_wilhelmson_split_cartesian_cgrid";
    storage.runtime.terrain.config.scheme_id = "none";
    storage.runtime.dynamics.scheme = std::make_unique<D5DynamicsScheme>();
    SimulationStateView state = make_simulation_state_view(storage);
    std::vector<tmv::domain::HomogeneousRuntimeInvalidationBinding>
        invalidations;

    SECTION("runtime binding is absent")
    {
        state.dynamics_runtime = nullptr;
        REQUIRE_THROWS_WITH(
            tmv::domain::append_dynamics_domain_recycling_bindings(
                state, invalidations),
            Catch::Matchers::Contains("bound runtime"));
    }
    SECTION("scheme capability is absent")
    {
        storage.runtime.dynamics.scheme =
            std::make_unique<D5DynamicsScheme>(false);
        REQUIRE_THROWS_WITH(
            tmv::domain::append_dynamics_domain_recycling_bindings(
                state, invalidations),
            Catch::Matchers::Contains("does not implement"));
    }
    SECTION("primary tendency allocation is partial")
    {
        storage.runtime.dynamics.du_dt.resize(4, 5, 3, 1.0f);
        REQUIRE_THROWS_WITH(
            tmv::domain::append_dynamics_domain_recycling_bindings(
                state, invalidations),
            Catch::Matchers::Contains("partially allocated"));
    }
    SECTION("compact acoustic profile has a stale shape")
    {
        storage.runtime.dynamics.acoustic_profile_rows.assign(9, 1.0f);
        REQUIRE_THROWS_WITH(
            tmv::domain::append_dynamics_domain_recycling_bindings(
                state, invalidations),
            Catch::Matchers::Contains("stale shape"));
    }
    SECTION("acoustic transport allocation is partial")
    {
        storage.runtime.dynamics.transport_context.mass_flux_x.resize(
            4, 5, 3, 1.0f);
        REQUIRE_THROWS_WITH(
            tmv::domain::append_dynamics_domain_recycling_bindings(
                state, invalidations),
            Catch::Matchers::Contains("transport scratch is only partially allocated"));
    }
    SECTION("configuration disagrees with the operator")
    {
        storage.configuration.dynamics_scheme_name =
            "klemp_wilhelmson_split_cartesian_agrid";
        REQUIRE_THROWS_WITH(
            tmv::domain::append_dynamics_domain_recycling_bindings(
                state, invalidations),
            Catch::Matchers::Contains("configuration ownership disagree"));
    }
    REQUIRE(invalidations.empty());
}

TEST_CASE("D5 storm authority translates and shifts atomically without reallocating",
          "[core][domain_recycling][storm_tracking][transaction][D5]")
{
    SimulationStorageState storage(
        GridStorageState{8, 8, 3, 100.0, 50.0, 0.1, 200.0});
    storage.grid.geometry.initialize(
        8, 8, 3, 100.0, 50.0, 200.0,
        CoordinateSystem::Cartesian);
    storage.grid.geometry.staggered = true;
    storage.runtime.terrain.config.scheme_id = "none";
    auto decision = seed_d5_storm_tracking(storage);
    decision.reason = tmv::domain::StormWindowMovementReason::EmergencyContainment;
    decision.confirmation_count = 1;
    storage.runtime.storm_window_movement.consecutive_compatible_samples = 1;
    SimulationStateView state = make_simulation_state_view(storage);
    const PhysicalCellDomain physical = physical_scalar_domain(state.grid);
    const auto plan =
        tmv::domain::plan_whole_cell_shift_for_physical_domain(
            state.grid.nr, state.grid.nth, state.grid.nz, 2, -1,
            physical.i_begin, physical.i_end,
            physical.j_begin, physical.j_end);

    auto& runtime = storage.runtime;
    const auto source_continuity = runtime.storm_tracking;
    const auto source_movement = runtime.storm_window_movement;
    const auto source_partition = runtime.storm_selected_partition;
    const auto* partition_pointer = runtime.storm_selected_partition.data();
    const std::size_t partition_capacity =
        runtime.storm_selected_partition.capacity();

    auto prepared = tmv::domain::prepare_storm_tracking_domain_recycling(
        state, plan, decision);
    REQUIRE(prepared.active);
    REQUIRE(prepared.displacement_x_m == 200.0);
    REQUIRE(prepared.displacement_y_m == -200.0);
    REQUIRE(prepared.committed_continuity.samples[0].x_m == 2800.0);
    REQUIRE(prepared.committed_continuity.samples[0].y_m == 2200.0);
    REQUIRE(prepared.committed_continuity.samples[1].x_m == 3000.0);
    REQUIRE(prepared.committed_continuity.samples[1].y_m == 2100.0);
    REQUIRE(prepared.committed_continuity.association_envelope_west_m ==
            2600.0);
    REQUIRE(prepared.committed_continuity.association_envelope_east_m ==
            3400.0);
    REQUIRE(prepared.committed_continuity.association_envelope_south_m ==
            1700.0);
    REQUIRE(prepared.committed_continuity.association_envelope_north_m ==
            2500.0);
    REQUIRE(prepared.committed_continuity.velocity_x_ms == 10.0);
    REQUIRE(prepared.committed_continuity.velocity_y_ms == -5.0);
    REQUIRE_FALSE(prepared.committed_movement.armed);
    REQUIRE(prepared.committed_movement.committed_move_available);
    REQUIRE(prepared.committed_movement.last_committed_time_s == 20.0);
    REQUIRE(prepared.committed_movement.cumulative_i_shift_cells == 2);
    REQUIRE(prepared.committed_movement.cumulative_j_shift_cells == -1);
    REQUIRE(prepared.committed_movement.pending_i_direction == 0);
    REQUIRE(prepared.committed_movement.pending_j_direction == 0);
    REQUIRE(prepared.committed_movement.consecutive_compatible_samples == 0);

    const std::size_t nx = static_cast<std::size_t>(physical.nx());
    const std::size_t ny = static_cast<std::size_t>(physical.ny());
    std::vector<std::uint8_t> expected_partition(nx * ny, 0U);
    expected_partition[1 * ny + 2] = 1U;
    expected_partition[2 * ny + 3] = 1U;
    REQUIRE(prepared.committed_partition == expected_partition);
    REQUIRE(std::string(
                tmv::domain::storm_tracking_domain_recycling_algorithm_id()) ==
            "translate_grid_track_shift_partition_commit_authority_v1");

    std::vector<tmv::domain::HomogeneousRuntimeCommitBinding> commits;
    REQUIRE(tmv::domain::append_storm_tracking_domain_recycling_binding(
        state, prepared, commits));
    REQUIRE(commits.size() == 1);

    double invalid_target = 1.0;
    const std::vector<tmv::domain::HomogeneousScalarCommitBinding> bad{{
        "bad", &invalid_target,
        std::numeric_limits<double>::quiet_NaN()}};
    REQUIRE_THROWS_WITH(
        tmv::domain::execute_homogeneous_window_shift(
            plan, {}, {}, bad, {}, {}, {}, commits),
        Catch::Matchers::Contains("non-finite"));
    REQUIRE(runtime.storm_tracking.samples[1].x_m ==
            source_continuity.samples[1].x_m);
    REQUIRE(runtime.storm_tracking.samples[1].y_m ==
            source_continuity.samples[1].y_m);
    REQUIRE(runtime.storm_window_movement.armed == source_movement.armed);
    REQUIRE(runtime.storm_window_movement.cumulative_i_shift_cells == 0);
    REQUIRE(runtime.storm_selected_partition == source_partition);
    REQUIRE(runtime.storm_selected_partition.data() == partition_pointer);

    tmv::domain::HorizontalDomainFrame frame;
    frame.earth_origin_at_t0_m = {1000.0, -500.0};
    frame.velocity_ms = {15.0, -4.0};
    const auto source_earth = frame.grid_to_earth(
        {source_continuity.samples[1].x_m,
         source_continuity.samples[1].y_m},
        source_continuity.samples[1].time_s);

    const auto report = tmv::domain::execute_homogeneous_window_shift(
        plan, {}, {}, {}, {}, {}, {}, commits);
    REQUIRE(report.runtime_commits_committed == 1);
    REQUIRE(runtime.storm_tracking.samples[0].x_m == 2800.0);
    REQUIRE(runtime.storm_tracking.samples[0].y_m == 2200.0);
    REQUIRE(runtime.storm_tracking.samples[1].x_m == 3000.0);
    REQUIRE(runtime.storm_tracking.samples[1].y_m == 2100.0);
    REQUIRE(runtime.storm_tracking.velocity_x_ms == 10.0);
    REQUIRE(runtime.storm_tracking.velocity_y_ms == -5.0);
    REQUIRE_FALSE(runtime.storm_window_movement.armed);
    REQUIRE(runtime.storm_window_movement.committed_move_available);
    REQUIRE(runtime.storm_window_movement.cumulative_i_shift_cells == 2);
    REQUIRE(runtime.storm_window_movement.cumulative_j_shift_cells == -1);
    REQUIRE(runtime.storm_selected_partition == expected_partition);
    REQUIRE(runtime.storm_selected_partition.data() == partition_pointer);
    REQUIRE(runtime.storm_selected_partition.capacity() ==
            partition_capacity);

    tmv::domain::commit_whole_cell_shift_to_domain_frame(
        frame, plan, state.grid.dr, state.grid.dtheta);
    const auto committed_earth = frame.grid_to_earth(
        {runtime.storm_tracking.samples[1].x_m,
         runtime.storm_tracking.samples[1].y_m},
        runtime.storm_tracking.samples[1].time_s);
    REQUIRE(committed_earth.x_m == Approx(source_earth.x_m));
    REQUIRE(committed_earth.y_m == Approx(source_earth.y_m));
}

TEST_CASE("D5 storm recycling rejects non-authoritative or lossy input",
          "[core][domain_recycling][storm_tracking][rollback][D5]")
{
    SimulationStorageState storage(
        GridStorageState{8, 8, 3, 100.0, 50.0, 0.1, 200.0});
    storage.grid.geometry.initialize(
        8, 8, 3, 100.0, 50.0, 200.0,
        CoordinateSystem::Cartesian);
    storage.grid.geometry.staggered = true;
    storage.runtime.terrain.config.scheme_id = "none";
    auto decision = seed_d5_storm_tracking(storage);
    SimulationStateView state = make_simulation_state_view(storage);
    const PhysicalCellDomain physical = physical_scalar_domain(state.grid);
    const auto plan =
        tmv::domain::plan_whole_cell_shift_for_physical_domain(
            state.grid.nr, state.grid.nth, state.grid.nz, 2, -1,
            physical.i_begin, physical.i_end,
            physical.j_begin, physical.j_end);
    std::vector<tmv::domain::HomogeneousRuntimeCommitBinding> commits;

    SECTION("aggregate runtime binding is absent")
    {
        state.runtime_owner = nullptr;
        REQUIRE_THROWS_WITH(
            tmv::domain::prepare_storm_tracking_domain_recycling(
                state, plan, decision),
            Catch::Matchers::Contains("aggregate"));
    }
    SECTION("continuity is not an authoritative active track")
    {
        storage.runtime.storm_tracking.motion_available = false;
        REQUIRE_THROWS_WITH(
            tmv::domain::prepare_storm_tracking_domain_recycling(
                state, plan, decision),
            Catch::Matchers::Contains("authoritative selected track"));
    }
    SECTION("movement decision does not match the exact plan")
    {
        decision.requested_i_shift_cells = 1;
        REQUIRE_THROWS_WITH(
            tmv::domain::prepare_storm_tracking_domain_recycling(
                state, plan, decision),
            Catch::Matchers::Contains("does not match"));
    }
    SECTION("selected partition is non-binary")
    {
        storage.runtime.storm_selected_partition[0] = 2U;
        REQUIRE_THROWS_WITH(
            tmv::domain::prepare_storm_tracking_domain_recycling(
                state, plan, decision),
            Catch::Matchers::Contains("non-binary"));
    }
    SECTION("selected partition shape is stale")
    {
        --storage.runtime.storm_selected_partition_nx;
        REQUIRE_THROWS_WITH(
            tmv::domain::prepare_storm_tracking_domain_recycling(
                state, plan, decision),
            Catch::Matchers::Contains("physical-domain selected partition"));
    }
    SECTION("the move would discard the complete selected partition")
    {
        const std::size_t ny = static_cast<std::size_t>(physical.ny());
        storage.runtime.storm_selected_partition.assign(
            static_cast<std::size_t>(physical.nx()) * ny, 0U);
        storage.runtime.storm_selected_partition[ny] = 1U;
        REQUIRE_THROWS_WITH(
            tmv::domain::prepare_storm_tracking_domain_recycling(
                state, plan, decision),
            Catch::Matchers::Contains("complete selected storm partition"));
    }
    SECTION("inactive prepared state cannot enter the transaction")
    {
        auto prepared = tmv::domain::prepare_storm_tracking_domain_recycling(
            state, plan, decision);
        prepared.active = false;
        REQUIRE_THROWS_WITH(
            tmv::domain::append_storm_tracking_domain_recycling_binding(
                state, prepared, commits),
            Catch::Matchers::Contains("does not match"));
    }
    REQUIRE(commits.empty());
}

TEST_CASE("D5 boundary-layer preparation rejects incomplete sampled memory",
          "[core][domain_recycling][boundary_layer][transaction][D5]")
{
    SimulationStorageState storage = pbl_recycling_storage();
    SimulationStateView state = make_simulation_state_view(storage);
    const auto physical = physical_scalar_domain(state.grid);
    const auto plan =
        tmv::domain::plan_whole_cell_shift_for_physical_domain(
            state.grid.nr, state.grid.nth, state.grid.nz, 1, 0,
            physical.i_begin, physical.i_end,
            physical.j_begin, physical.j_end);
    const auto primary =
        tmv::domain::prepare_flat_homogeneous_primary_profiles(
            storage, std::vector<double>(4, 0.1));
    const Field3D before = storage.runtime.boundary_layer.held_u_force_n;
    storage.runtime.boundary_layer.skin_captured = false;
    REQUIRE_THROWS_WITH(
        tmv::domain::prepare_flat_homogeneous_boundary_layer_recycling(
            state, plan, primary),
        Catch::Matchers::Contains("complete skin and column memory"));
    for (std::size_t index = 0; index < before.size(); ++index)
        REQUIRE(storage.runtime.boundary_layer.held_u_force_n.data()[index] ==
                before.data()[index]);
}

TEST_CASE("D5 leading PBL preparation covers every implemented active scheme",
          "[core][domain_recycling][boundary_layer][schemes][D5]")
{
    for (const std::string scheme_name : {"slab", "ysu", "mynn"})
    {
        CAPTURE(scheme_name);
        SimulationStorageState storage = pbl_recycling_storage(scheme_name);
        SimulationStateView state = make_simulation_state_view(storage);
        const auto physical = physical_scalar_domain(state.grid);
        const auto plan =
            tmv::domain::plan_whole_cell_shift_for_physical_domain(
                state.grid.nr, state.grid.nth, state.grid.nz, -1, 1,
                physical.i_begin, physical.i_end,
                physical.j_begin, physical.j_end);
        const auto primary =
            tmv::domain::prepare_flat_homogeneous_primary_profiles(
                storage, std::vector<double>(4, 0.1));
        const auto prepared =
            tmv::domain::prepare_flat_homogeneous_boundary_layer_recycling(
                state, plan, primary);
        REQUIRE(prepared.active);
        for (const std::vector<float>* profile : {
                 &prepared.leading_u_force_n,
                 &prepared.leading_v_force_n,
                 &prepared.leading_theta_inventory_rate_kgk_s,
                 &prepared.leading_vapor_mass_rate_kg_s,
                 &prepared.leading_dtke_s})
        {
            REQUIRE(profile->size() == 4);
            for (float value : *profile)
                REQUIRE(std::isfinite(value));
        }
        for (float value : {
                 prepared.leading_skin_theta,
                 prepared.leading_skin_qv,
                 prepared.leading_ustar,
                 prepared.leading_tau_u,
                 prepared.leading_tau_v,
                 prepared.leading_sensible_heat_flux,
                 prepared.leading_latent_heat_flux,
                 prepared.leading_moisture_flux,
                 prepared.leading_pbl_height})
            REQUIRE(std::isfinite(value));
    }
}

TEST_CASE("Earth-fixed D5 slab preparation owns physical-height volume and surface columns",
          "[core][domain_recycling][terrain][boundary_layer][transaction][D5]")
{
    SimulationStorageState storage = pbl_recycling_storage("slab");
    storage.configuration.domain_frame =
        tmv::domain::HorizontalDomainFrame::stationary();
    TerrainConfig config;
    config.scheme_id = "bell";
    config.coord_id = "smoothed";
    config.horizontal_placement_id = "earth_fixed";
    config.ztop = 800.0;
    config.bell.h0 = 120.0;
    config.bell.a = 350.0;
    config.bell.axisymmetric = true;
    config.feature_center_earth_x_m = 350.0;
    config.feature_center_earth_y_m = 625.0;
    storage.runtime.terrain.config = config;
    tmv::EarthFixedTerrainBuildRequest request;
    request.nr = storage.grid.nr;
    request.nth = storage.grid.nth;
    request.nz = storage.grid.nz;
    request.dx_m = storage.grid.dx;
    request.dy_m = storage.grid.dy;
    request.feature_center_earth_m = {350.0, 625.0};
    const auto geometry = tmv::build_earth_fixed_terrain_geometry(
        config, request);
    const std::vector<double> environmental_tke(4, 0.1);
    const auto primary = tmv::domain::prepare_terrain_aware_primary_fields(
        storage, geometry.metrics, environmental_tke);
    SimulationStateView state = make_simulation_state_view(storage);
    const auto physical = physical_scalar_domain(state.grid);
    const auto plan =
        tmv::domain::plan_whole_cell_shift_for_physical_domain(
            state.grid.nr, state.grid.nth, state.grid.nz, 1, -1,
            physical.i_begin, physical.i_end,
            physical.j_begin, physical.j_end);
    auto prepared =
        tmv::domain::prepare_terrain_aware_boundary_layer_recycling(
            state, plan, primary, geometry.metrics);
    REQUIRE(prepared.active);
    REQUIRE(prepared.terrain_aware);
    for (const auto& field : prepared.terrain_leading_fields)
        REQUIRE(field.size() == static_cast<std::size_t>(6 * 7 * 4));
    for (const auto& surface : prepared.terrain_leading_surfaces)
        REQUIRE(surface.size() == static_cast<std::size_t>(6 * 7));
    REQUIRE(prepared.terrain_leading_surfaces[0][0]
            != prepared.terrain_leading_surfaces[0][3 * 7 + 2]);

    std::vector<tmv::domain::HomogeneousFieldShiftBinding> volumes;
    std::vector<tmv::domain::HomogeneousSurfaceShiftBinding> surfaces;
    std::vector<tmv::domain::HomogeneousSurfaceRecordShiftBinding> records;
    std::vector<tmv::domain::HomogeneousBooleanCommitBinding> flags;
    REQUIRE(tmv::domain::append_boundary_layer_domain_recycling_bindings(
        state, plan, prepared, volumes, surfaces, records, flags));
    const Field3D before = storage.runtime.boundary_layer.held_u_force_n;
    auto invalid = prepared.terrain_leading_surfaces[0];
    invalid[0] = std::numeric_limits<float>::quiet_NaN();
    auto invalid_surfaces = surfaces;
    invalid_surfaces[0].leading_field = &invalid;
    REQUIRE_THROWS_WITH(
        tmv::domain::execute_homogeneous_window_shift(
            plan, volumes, invalid_surfaces, {}, records, flags),
        Catch::Matchers::Contains("non-finite"));
    for (std::size_t index = 0; index < before.size(); ++index)
        REQUIRE(storage.runtime.boundary_layer.held_u_force_n.data()[index]
                == before.data()[index]);

    (void)tmv::domain::execute_homogeneous_window_shift(
        plan, volumes, surfaces, {}, records, flags);
    require_shifted_from_field(
        before, storage.runtime.boundary_layer.held_u_force_n,
        plan, prepared.terrain_leading_fields[0]);
    REQUIRE_FALSE(storage.runtime.boundary_layer.updated_this_step);
}

TEST_CASE("Two-way recycling rejects unconsumed and staged interface contributions atomically",
          "[core][domain_recycling][nested][two_way][rollback]")
{
    auto storage                           = separately_evolving_nested_recycling_storage();
    storage.configuration.nesting.feedback = tmv::init::config::NestingFeedback::TwoWay;
    auto& nested                           = storage.nested_execution;
    nested.last_two_way_feedback           = {};
    nested.last_two_way_feedback.complete  = true;
    const auto plan       = tmv::domain::plan_whole_cell_shift_for_physical_domain(14, 14, 3, 1, 0, 1, 13, 1, 13);
    auto       post_frame = storage.configuration.domain_frame;
    (void)tmv::domain::commit_whole_cell_shift_to_domain_frame(post_frame, plan, storage.grid.dx, storage.grid.dy);
    auto prepared = tmv::domain::prepare_nested_grid_domain_recycling(storage, plan, post_frame);
    REQUIRE(tmv::domain::can_commit_nested_grid_domain_recycling(storage, prepared));
    auto& water = nested.child.runtime.advection.moist_interface.water[0];
    water.configure(6, 6, 3, {1, 5, 1, 5, 0, 3}, 2, 2);
    prepared = tmv::domain::prepare_nested_grid_domain_recycling(storage, plan, post_frame);
    water.stage_scalar_flux([](int, int, int, int) { return 1.0; }, .5, 500, 500, 500);
    CHECK_FALSE(tmv::domain::can_commit_nested_grid_domain_recycling(storage, prepared));
    CHECK_THROWS(tmv::domain::prepare_nested_grid_domain_recycling(storage, plan, post_frame));
    water.validate_staged_scalar_commit();
    water.commit_staged_scalar();
    CHECK_THROWS(tmv::domain::prepare_nested_grid_domain_recycling(storage, plan, post_frame));
    water.reset_interval();
    prepared = tmv::domain::prepare_nested_grid_domain_recycling(storage, plan, post_frame);
    storage.runtime.advection.moist_interface.committed_physics[0] = 1;
    CHECK_FALSE(tmv::domain::can_commit_nested_grid_domain_recycling(storage, prepared));
    storage.runtime.advection.moist_interface.committed_physics[0] = 0;
    CHECK(tmv::domain::can_commit_nested_grid_domain_recycling(storage, prepared));
    auto& momentum = storage.runtime.dynamics.momentum_interface_flux_register;
    momentum.accumulate_operator_source_impulse({1, 0, 0});
    CHECK_FALSE(tmv::domain::can_commit_nested_grid_domain_recycling(storage, prepared));
    momentum.reset_interval();
    CHECK(tmv::domain::can_commit_nested_grid_domain_recycling(storage, prepared));
    nested.last_two_way_feedback.complete = false;
    CHECK_THROWS(tmv::domain::prepare_nested_grid_domain_recycling(storage, plan, post_frame));
    CHECK(nested.relationship.child_origin_parent_cell_i == 4);
    CHECK(nested.cumulative_parent_recycling_shift_i == 0);
}
