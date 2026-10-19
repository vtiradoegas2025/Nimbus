#include <fstream>
#include "microphysics/schemes/kessler/kessler.hpp"
#include "microphysics/base/thermodynamics.hpp"
#include <catch2/catch.hpp>

#include "boundary_conditions/runtime.hpp"
#include "boundary_layer/runtime.hpp"
#include "chaos/runtime.hpp"
#include "compute/compute_backend.hpp"
#ifdef NIMBUS_NESTED_RELOCATION_BENCHMARK
#include "core/infra/allocation_telemetry.hpp"
#endif
#include "core/infra/domain_recycling_state.hpp"
#include "core/infra/nested_execution.hpp"
#include "core/runtime/runtime_config.hpp"
#include "core/runtime/checkpoint_format.hpp"
#include "core/runtime/headless_runtime.hpp"
#include "core/runtime/process_state.hpp"
#include "core/runtime/simulation_checkpoint.hpp"
#include "core/orchestration/dynamics/thermodynamic_closure.hpp"
#include "dynamics/coriolis.hpp"
#include "dynamics/dynamics_base.hpp"
#include "dynamics/eos_pressure.hpp"
#include "diagnostics/conservation_budget.hpp"
#include "diagnostics/momentum_budget.hpp"
#include "diagnostics/moisture_budget.hpp"
#include "diagnostics/storm_tracker.hpp"
#include "microphysics/runtime.hpp"
#include "microphysics/grid_geometry.hpp"
#include "numerics/advection/advection_base.hpp"
#include "numerics/diffusion/diffusion_base.hpp"
#include "numerics/runtime.hpp"
#include "numerics/time_stepping/time_stepping_base.hpp"
#include "radar/factory.hpp"
#include "radar/runtime.hpp"
#include "radiation/runtime.hpp"
#include "terrain/factory.hpp"
#include "turbulence/runtime.hpp"
#include "turbulence/schemes/smagorinsky/smagorinsky.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <iterator>
#include <limits>
#include <memory>
#include <map>
#include <regex>
#include <string>
#include <utility>
#include <vector>
#include <omp.h>

// The production executable still owns this legacy profile adapter in its
// main translation unit. This non-executable qualification artifact supplies
// the identical adapter while linking the remaining production objects.
void compute_wind_profile(
    const WindProfile& profile, double z, double& u, double& v)
{
    if (z <= 1000.0)
    {
        const double fraction = z / 1000.0;
        u = profile.u_sfc + fraction * (profile.u_1km - profile.u_sfc);
        v = profile.v_sfc + fraction * (profile.v_1km - profile.v_sfc);
    }
    else if (z <= 6000.0)
    {
        const double fraction = (z - 1000.0) / 5000.0;
        u = profile.u_1km + fraction * (profile.u_6km - profile.u_1km);
        v = profile.v_1km + fraction * (profile.v_6km - profile.v_1km);
    }
    else
    {
        u = profile.u_6km;
        v = profile.v_6km;
    }
}

namespace
{

class ProcessStateRestore
{
public:
    ProcessStateRestore()
        : storage_(std::move(global_simulation_storage)),
          boundary_scheme_(std::move(tmv::bc_scheme)),
          science_diagnostics_(step_diagnostics_enabled()),
          radar_diagnostics_(step_radar_diagnostics_enabled()),
          vorticity_diagnostics_(step_vorticity_diagnostics_enabled()),
          angular_momentum_diagnostics_(
              step_angular_momentum_diagnostics_enabled()),
          pressure_diagnostics_(step_pressure_diagnostics_enabled())
    {
    }

    ~ProcessStateRestore()
    {
        tmv::bc_scheme = std::move(boundary_scheme_);
        global_simulation_storage = std::move(storage_);
        set_step_diagnostics_enabled(science_diagnostics_);
        set_step_derived_diagnostic_requirements(
            radar_diagnostics_, vorticity_diagnostics_,
            angular_momentum_diagnostics_, pressure_diagnostics_);
    }

private:
    SimulationStorageState storage_;
    std::unique_ptr<tmv::BoundaryConditionScheme> boundary_scheme_;
    bool science_diagnostics_ = false;
    bool radar_diagnostics_ = false;
    bool vorticity_diagnostics_ = false;
    bool angular_momentum_diagnostics_ = false;
    bool pressure_diagnostics_ = false;
};

void resize_primary(PrognosticFieldStorage& fields, int nr, int nth, int nz)
{
    Field3D* all[] = {
        &fields.rho, &fields.p, &fields.u, &fields.v, &fields.w,
        &fields.theta, &fields.qv, &fields.qc, &fields.qr, &fields.qi,
        &fields.qs, &fields.qg, &fields.qh, &fields.tracer, &fields.tke};
    for (Field3D* field : all)
        field->resize(nr, nth, nz, 0.0f);
}

void fill_balanced_primary(
    PrognosticFieldStorage& fields, int nr, int nth, int nz)
{
    resize_primary(fields, nr, nth, nz);
    for (int i = 0; i < nr; ++i)
        for (int j = 0; j < nth; ++j)
            for (int k = 0; k < nz; ++k)
            {
                const float rho = k == 0 ? 1.1f : 1.0f;
                const float theta = k == 0 ? 300.0f : 302.0f;
                fields.rho[i][j][k] = rho;
                fields.theta[i][j][k] = theta;
                fields.p[i][j][k] = static_cast<float>(
                    tmv::dynamics_eos::pressure_fp64(rho, theta));
                fields.tke[i][j][k] = 0.1f;
            }
}

SimulationStorageState make_parent(int nr = 8, int nth = 8)
{
    SimulationStorageState state(
        GridStorageState(nr, nth, 2, 1000.0, 300.0, 2.0, 800.0));
    state.grid.geometry.initialize(
        nr, nth, 2, 1000.0, 300.0, 800.0,
        CoordinateSystem::Cartesian, StaggerType::CGrid);
    state.grid.geometry.z_faces = {0.0, 200.0, 550.0};
    state.grid.geometry.z = {100.0, 375.0};
    state.process_grid_metrics.dx = 1000.0;
    state.process_grid_metrics.dy = 800.0;
    state.process_grid_metrics.dz = {200.0, 350.0};
    state.process_grid_metrics.z_int = state.grid.geometry.z_faces;
    state.time_s = 2.0;
    fill_balanced_primary(state.fields, nr, nth, 2);
    state.base.rho0 = {
        static_cast<double>(state.fields.rho[1][1][0]),
        static_cast<double>(state.fields.rho[1][1][1])};
    state.base.theta0 = {
        static_cast<double>(state.fields.theta[1][1][0]),
        static_cast<double>(state.fields.theta[1][1][1])};
    state.base.p0 = {
        static_cast<double>(state.fields.p[1][1][0]),
        static_cast<double>(state.fields.p[1][1][1])};
    state.base.qv0 = {0.0, 0.0};
    state.base.u0 = {0.0, 0.0};
    state.base.v0 = {0.0, 0.0};

    state.configuration.coordinate_system = CoordinateSystem::Cartesian;
    state.configuration.stagger_type = StaggerType::CGrid;
    state.configuration.domain_motion_mode =
        tmv::init::config::DomainMotionMode::Stationary;
    state.configuration.domain_frame =
        tmv::domain::HorizontalDomainFrame::stationary();
    state.configuration.microphysics_scheme = "none";
    state.configuration.dynamics_scheme_name =
        "klemp_wilhelmson_split_cartesian_cgrid";
    state.configuration.damping_enabled = false;
    state.configuration.w_only_top_sponge = false;
    state.configuration.lateral_sponge_enabled = false;
    state.configuration.coriolis_enabled = false;
    state.configuration.scientific_diagnostics_every_steps = 0;
    state.configuration.validation_policy.mode = tmv::GuardMode::Strict;
    state.configuration.validation_policy.fail_on = tmv::GuardFailOn::Both;
    return state;
}

struct AcousticCaseSpec
{
    const char* name = "normal_ratio2_sigma600";
    int parent_nth = 10;
    int refinement_ratio = 2;
    int child_origin_parent_j = 2;
    int child_parent_span_y = 4;
    double angle_degrees = 0.0;
    double sigma_m = 600.0;
    int total_parent_intervals = 28;
};

SimulationStorageState make_acoustic_parent(
    const AcousticCaseSpec& spec = {})
{
    constexpr int nr = 34;
    const int nth = spec.parent_nth;
    constexpr int nz = 2;
    constexpr double spacing_m = 500.0;
    constexpr double step_s = 0.5;
    constexpr double density_kgm3[] = {1.1, 1.0};
    constexpr double theta_k[] = {300.0, 302.0};
    constexpr double amplitude = 2.0e-4;
    constexpr double center_m = 1000.0;
    SimulationStorageState state(
        GridStorageState(nr, nth, nz, spacing_m, 300.0,
                         step_s, spacing_m));
    state.grid.geometry.initialize(
        nr, nth, nz, spacing_m, 300.0, spacing_m,
        CoordinateSystem::Cartesian, StaggerType::CGrid);
    state.grid.geometry.z_faces = {0.0, 200.0, 550.0};
    state.grid.geometry.z = {100.0, 375.0};
    state.process_grid_metrics.dx = spacing_m;
    state.process_grid_metrics.dy = spacing_m;
    state.process_grid_metrics.dz = {200.0, 350.0};
    state.process_grid_metrics.z_int = state.grid.geometry.z_faces;
    state.time_s = 0.0;
    resize_primary(state.fields, nr, nth, nz);
    const double pressure_pa[] = {
        tmv::dynamics_eos::pressure_fp64(density_kgm3[0], theta_k[0]),
        tmv::dynamics_eos::pressure_fp64(density_kgm3[1], theta_k[1])};
    constexpr double gamma =
        physical_constants::specific_heat_cp_jkgk /
        (physical_constants::specific_heat_cp_jkgk -
         physical_constants::gas_constant_dry_air_jkgk);
    const double angle_rad = spec.angle_degrees *
        std::acos(-1.0) / 180.0;
    const double propagation_x = std::cos(angle_rad);
    const double propagation_y = std::sin(angle_rad);
    const double center_y_m = 0.5 * static_cast<double>(nth) * spacing_m;
    const auto perturbation = [&](double x_m, double y_m) {
        const double along_m =
            (x_m - center_m) * propagation_x +
            (y_m - center_y_m) * propagation_y;
        const double normalized = along_m / spec.sigma_m;
        return amplitude * std::exp(-0.5 * normalized * normalized);
    };
    for (int i = 0; i < nr; ++i)
    {
        const double scalar_x_m = (static_cast<double>(i) + 0.5) *
            spacing_m;
        for (int j = 0; j < nth; ++j)
        {
            const double scalar_y_m = (static_cast<double>(j) + 0.5) *
                spacing_m;
            for (int k = 0; k < nz; ++k)
            {
                const double density = density_kgm3[k] *
                    (1.0 + perturbation(scalar_x_m, scalar_y_m));
                state.fields.rho[i][j][k] = static_cast<float>(density);
                state.fields.theta[i][j][k] =
                    static_cast<float>(theta_k[k]);
                state.fields.p[i][j][k] = static_cast<float>(
                    tmv::dynamics_eos::pressure_fp64(
                        density, theta_k[k]));
                state.fields.tke[i][j][k] = 0.0f;
            }
        }
    }
    for (int i = 0; i < nr; ++i)
    {
        const double face_x_m = (static_cast<double>(i) + 1.0) *
            spacing_m;
        for (int j = 0; j < nth; ++j)
        {
            const double scalar_y_m = (static_cast<double>(j) + 0.5) *
                spacing_m;
            for (int k = 0; k < nz; ++k)
            {
                const double sound_speed_ms = std::sqrt(
                    gamma * pressure_pa[k] / density_kgm3[k]);
                state.fields.u[i][j][k] = static_cast<float>(
                    sound_speed_ms *
                    perturbation(face_x_m, scalar_y_m) * propagation_x);
            }
        }
    }
    for (int i = 0; i < nr; ++i)
    {
        const double scalar_x_m = (static_cast<double>(i) + 0.5) *
            spacing_m;
        for (int j = 0; j < nth; ++j)
        {
            const double face_y_m = (static_cast<double>(j) + 1.0) *
                spacing_m;
            for (int k = 0; k < nz; ++k)
            {
                const double sound_speed_ms = std::sqrt(
                    gamma * pressure_pa[k] / density_kgm3[k]);
                state.fields.v[i][j][k] = static_cast<float>(
                    sound_speed_ms *
                    perturbation(scalar_x_m, face_y_m) * propagation_y);
            }
        }
    }

    state.base.rho0 = {density_kgm3[0], density_kgm3[1]};
    state.base.theta0 = {theta_k[0], theta_k[1]};
    state.base.p0 = {pressure_pa[0], pressure_pa[1]};
    state.base.qv0 = {0.0, 0.0};
    state.base.u0 = {0.0, 0.0};
    state.base.v0 = {0.0, 0.0};
    state.configuration.coordinate_system = CoordinateSystem::Cartesian;
    state.configuration.stagger_type = StaggerType::CGrid;
    state.configuration.domain_motion_mode =
        tmv::init::config::DomainMotionMode::Stationary;
    state.configuration.domain_frame =
        tmv::domain::HorizontalDomainFrame::stationary();
    state.configuration.microphysics_scheme = "none";
    state.configuration.dynamics_scheme_name =
        "klemp_wilhelmson_split_cartesian_cgrid";
    state.configuration.damping_enabled = false;
    state.configuration.w_only_top_sponge = false;
    state.configuration.lateral_sponge_enabled = false;
    state.configuration.coriolis_enabled = false;
    state.configuration.scientific_diagnostics_every_steps = 0;
    state.configuration.validation_policy.mode = tmv::GuardMode::Strict;
    state.configuration.validation_policy.fail_on = tmv::GuardFailOn::Both;
    return state;
}

SimulationStorageState make_active_physics_parent()
{
    SimulationStorageState state = make_acoustic_parent();
    constexpr int nz = 2;
    constexpr double dz_m = 300.0;
    state.grid.dz = dz_m;
    state.grid.geometry.z_faces = {0.0, dz_m, 2.0 * dz_m};
    state.grid.geometry.z = {0.5 * dz_m, 1.5 * dz_m};
    state.process_grid_metrics.dz = {dz_m, dz_m};
    state.process_grid_metrics.z_int = state.grid.geometry.z_faces;
    state.base.qv0 = {0.014, 0.009};

    for (int i = 0; i < state.grid.nr; ++i)
        for (int j = 0; j < state.grid.nth; ++j)
            for (int k = 0; k < nz; ++k)
            {
                const float qv = k == 0 ? 0.014f : 0.009f;
                const float horizontal = static_cast<float>(
                    1.0 + 0.08 * std::sin(0.31 * i) *
                              std::cos(0.47 * j));
                state.fields.qv[i][j][k] = qv;
                state.fields.qc[i][j][k] = 1.6e-3f * horizontal;
                state.fields.qr[i][j][k] = 3.0e-4f * horizontal;
                state.fields.u[i][j][k] = static_cast<float>(
                    3.0 + 0.08 * i + 0.35 * k);
                state.fields.v[i][j][k] = static_cast<float>(
                    -1.0 + 0.06 * j - 0.2 * k);
                state.fields.w[i][j][k] = static_cast<float>(
                    0.08 * std::sin(0.2 * i) * std::cos(0.3 * j));
            }
    return state;
}

tmv::domain::StormWindowMovementDecision seed_recycling_storm_authority(
    SimulationStorageState& storage)
{
    auto& continuity = storage.runtime.storm_tracking;
    continuity.status = tmv::diagnostics::StormContinuityStatus::Tracking;
    continuity.reason = tmv::diagnostics::StormContinuityReason::None;
    continuity.sample_count = 2;
    continuity.samples[0] = {10.0, 3000.0, 2000.0, 0.90};
    continuity.samples[1] = {20.0, 3200.0, 1900.0, 0.92};
    continuity.branch_id = 7;
    continuity.topology = tmv::diagnostics::StormBranchTopology::Single;
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

    const PhysicalCellDomain physical =
        physical_scalar_domain(make_simulation_state_view(storage).grid);
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

bool initialize_parent_modules(const char* backend = "cpu")
{
    auto& state = global_simulation_storage;
    state.runtime.compute.config.backend = backend;
    state.runtime.compute.config.allow_fallback = false;
    std::string compute_error;
    const bool compute_ready = initialize_compute_backend_runtime(compute_error);
    const char* required = std::getenv("NIMBUS_GPU_REQUIRED");
    const bool optional_accelerator = std::string(backend) != "cpu" &&
        !(required != nullptr && std::string(required) == "1");
    if (!compute_ready && optional_accelerator)
        return false;
    if (!compute_ready)
    {
        INFO("Compute backend initialization error: " << compute_error);
    }
    REQUIRE(compute_ready);
    set_step_diagnostics_enabled(true);
    set_step_derived_diagnostic_requirements(false, false, false, false);

    RadiationConfig radiation;
    radiation.scheme_id = "none";
    initialize_radiation(state.runtime.radiation, "none", radiation);

    TurbulenceConfig turbulence;
    turbulence.scheme_id = "none";
    initialize_turbulence(state.runtime.turbulence, "none", turbulence);

    BoundaryLayerConfig pbl;
    pbl.scheme_id = "none";
    initialize_boundary_layer(
        state.runtime.boundary_layer, "none", pbl, SurfaceConfig{});

    state.runtime.microphysics.scheme = tmv::create_microphysics_scheme(
        "none", state.configuration.microphysics_config);
    REQUIRE(state.runtime.microphysics.scheme);

    state.runtime.terrain.config.scheme_id = "none";
    state.runtime.terrain.scheme = create_terrain_scheme("none");
    REQUIRE(state.runtime.terrain.scheme);
    state.runtime.terrain.scheme->initialize(state.runtime.terrain.config);

    AdvectionConfig advection;
    advection.scheme_id = "tvd";
    advection.positivity = true;
    initialize_advection(state.runtime.advection, "tvd", advection);

    DiffusionConfig diffusion;
    diffusion.scheme_id = "explicit";
    initialize_diffusion(state.runtime.diffusion, "explicit", diffusion);

    TimeSteppingConfig time;
    time.scheme_id = "split_explicit";
    time.split_acoustic = true;
    time.n_acoustic_substeps = 3;
    initialize_time_stepping(
        state.runtime.time_stepping, "split_explicit", time);

    tmv::chaos::ChaosConfig chaos;
    chaos.scheme_id = "none";
    tmv::chaos::initialize_chaos(global_simulation_state(), chaos);
    tmv::initialize_dynamics(
        "klemp_wilhelmson_split_cartesian_cgrid");
    return true;
}

void enable_kessler_smagorinsky_physics(SimulationStorageState& state)
{
    state.configuration.microphysics_scheme = "kessler";
    state.runtime.microphysics.scheme = tmv::create_microphysics_scheme(
        "kessler", state.configuration.microphysics_config);
    REQUIRE(state.runtime.microphysics.scheme);
    state.runtime.microphysics.scheme->prepare_prognostic_state(
        state.fields.rho, state.fields.qc, state.fields.qr,
        state.fields.qi, state.fields.qs, state.fields.qg,
        state.fields.qh);

    TurbulenceConfig turbulence;
    turbulence.scheme_id = "smagorinsky";
    turbulence.mode = "3d";
    turbulence.filter_width = "cubic_root";
    turbulence.stability_correction = "none";
    turbulence.dt_sgs = 0.25;
    initialize_turbulence(
        state.runtime.turbulence, "smagorinsky", turbulence);
}

void prepare_child(const PrognosticFieldStorage& previous_parent,
                   int                           parent_cells_x   = 6,
                   int                           parent_cells_y   = 6,
                   int                           child_cells_x    = 4,
                   int                           child_cells_y    = 4,
                   int                           ratio            = 2,
                   int                           relaxation_width = 1)
{
    auto& state = global_simulation_storage;
    tmv::domain::OneWayNestRelationshipInput input;
    input.parent_cells_x = parent_cells_x;
    input.parent_cells_y = parent_cells_y;
    input.parent_levels_z = 2;
    input.parent_dx_m = 1000.0;
    input.parent_dy_m = 800.0;
    input.parent_dt_s = 2.0;
    input.parent_ratio_x = ratio;
    input.parent_ratio_y = ratio;
    input.parent_time_step_ratio = ratio;
    input.child_origin_parent_cell_i = 2;
    input.child_origin_parent_cell_j = 2;
    input.child_cells_x = child_cells_x;
    input.child_cells_y = child_cells_y;
    input.child_levels_z = 2;
    input.relaxation_width_child_cells         = relaxation_width;
    input.boundary_stencil_parent_cells = 1;
    state.configuration.nesting.enabled = true;
    state.configuration.nesting.parent_ratio_x = input.parent_ratio_x;
    state.configuration.nesting.parent_ratio_y = input.parent_ratio_y;
    state.configuration.nesting.parent_time_step_ratio =
        input.parent_time_step_ratio;
    state.configuration.nesting.child_origin_parent_cell_i =
        input.child_origin_parent_cell_i;
    state.configuration.nesting.child_origin_parent_cell_j =
        input.child_origin_parent_cell_j;
    state.configuration.nesting.child_cells_x = input.child_cells_x;
    state.configuration.nesting.child_cells_y = input.child_cells_y;
    state.configuration.nesting.relaxation_width_child_cells =
        input.relaxation_width_child_cells;
    state.configuration.nesting.boundary_stencil_parent_cells =
        input.boundary_stencil_parent_cells;
    state.configuration.nesting.relaxation_e_folding_time_s = 4.0;
    prepare_nested_execution_relationship(
        state, tmv::domain::prepare_one_way_nest_relationship(input), 0.0);
    initialize_nested_shared_level_reference_state(state);
    NestedChildInitializationReport initialization;
    initialize_nested_child_physical_primary_state(
        state, previous_parent, initialization);
    NestedChildRelaxationReport boundary;
    apply_nested_child_physical_relaxation(
        state, previous_parent, {4.0}, 1.0, boundary);
    initialize_nested_child_level_runtime(state);
    if (state.nested_execution.child.runtime.microphysics.scheme
            ->prognostic_scalars_for_transport().empty())
        require_nested_child_level_runtime_ready(state);
}

void enable_nested_beta_plane(SimulationStorageState& state)
{
    state.configuration.coriolis_enabled = true;
    state.configuration.coriolis_mode =
        tmv::init::config::CoriolisMode::BetaPlane;
    state.configuration.coriolis_latitude_deg = 35.0;
    state.configuration.domain_motion_mode =
        tmv::init::config::DomainMotionMode::ConstantVelocity;
    state.configuration.domain_frame.velocity_ms.v_ms = 8.0;
    state.nested_execution.child.domain_frame.velocity_ms =
        state.configuration.domain_frame.velocity_ms;
    state.coriolis_f = tmv::make_cartesian_coriolis_profile(
        true, true, state.configuration.coriolis_latitude_deg,
        state.configuration.domain_frame, state.time_s).f_at_grid_y0;
    state.nested_execution.child.coriolis_f =
        tmv::make_cartesian_coriolis_profile(
            true, true, state.configuration.coriolis_latitude_deg,
            state.nested_execution.child.domain_frame,
            state.nested_execution.child.time_s).f_at_grid_y0;
    REQUIRE_NOTHROW(require_valid_nested_execution_relationship(state));
}

#ifdef NIMBUS_NESTED_RELOCATION_BENCHMARK

int benchmark_environment_integer(const char* name, int fallback)
{
    const char* const text = std::getenv(name);
    if (!text || *text == '\0')
        return fallback;
    char* end = nullptr;
    const long value = std::strtol(text, &end, 10);
    if (!end || *end != '\0' || value < 1 ||
        value > std::numeric_limits<int>::max())
        throw std::invalid_argument(std::string(name) +
                                    " must be a positive integer");
    return static_cast<int>(value);
}

std::string relocation_benchmark_profile()
{
    const char* const text = std::getenv("NIMBUS_NESTED_BENCH_PROFILE");
    const std::string profile = text && *text != '\0' ? text : "baseline";
    if (profile != "baseline" && profile != "flat_active" &&
        profile != "flat_milbrandt" &&
        profile != "flat_active_milbrandt")
        throw std::invalid_argument(
            "NIMBUS_NESTED_BENCH_PROFILE must be baseline, flat_active, "
            "flat_milbrandt, or flat_active_milbrandt");
    return profile;
}

SimulationStorageState make_relocation_benchmark_parent(
    int child_nx, int child_ny, int nz, int margin_parent_cells)
{
    constexpr int ratio = 2;
    constexpr double dx_m = 1000.0;
    constexpr double dy_m = 1000.0;
    constexpr double dz_m = 250.0;
    constexpr double parent_dt_s = 2.0;
    const int parent_physical_nx =
        child_nx / ratio + 2 * margin_parent_cells;
    const int parent_physical_ny =
        child_ny / ratio + 2 * margin_parent_cells;
    const int storage_nx = parent_physical_nx + 2;
    const int storage_ny = parent_physical_ny + 2;
    SimulationStorageState state(GridStorageState(
        storage_nx, storage_ny, nz, dx_m, dz_m, parent_dt_s, dy_m));
    state.grid.geometry.initialize(
        storage_nx, storage_ny, nz, dx_m, dz_m, dy_m,
        CoordinateSystem::Cartesian, StaggerType::CGrid);
    state.process_grid_metrics.dx = dx_m;
    state.process_grid_metrics.dy = dy_m;
    state.process_grid_metrics.dz.assign(nz, dz_m);
    state.process_grid_metrics.z_int.resize(static_cast<std::size_t>(nz) + 1);
    for (int k = 0; k <= nz; ++k)
        state.process_grid_metrics.z_int[static_cast<std::size_t>(k)] =
            static_cast<double>(k) * dz_m;
    state.time_s = parent_dt_s;
    resize_primary(state.fields, storage_nx, storage_ny, nz);
    state.base.rho0.resize(nz);
    state.base.theta0.resize(nz);
    state.base.p0.resize(nz);
    state.base.qv0.assign(nz, 0.0);
    state.base.u0.assign(nz, 0.0);
    state.base.v0.assign(nz, 0.0);
    for (int k = 0; k < nz; ++k)
    {
        const double z_m = (static_cast<double>(k) + 0.5) * dz_m;
        const float rho = static_cast<float>(1.2 * std::exp(-z_m / 9000.0));
        const float theta = static_cast<float>(300.0 + 0.003 * z_m);
        const float pressure = static_cast<float>(
            tmv::dynamics_eos::pressure_fp64(rho, theta));
        state.base.rho0[static_cast<std::size_t>(k)] = rho;
        state.base.theta0[static_cast<std::size_t>(k)] = theta;
        state.base.p0[static_cast<std::size_t>(k)] = pressure;
        for (int i = 0; i < storage_nx; ++i)
            for (int j = 0; j < storage_ny; ++j)
            {
                state.fields.rho[i][j][k] = rho;
                state.fields.theta[i][j][k] = theta;
                state.fields.p[i][j][k] = pressure;
                state.fields.tke[i][j][k] = 0.1f;
            }
    }

    state.configuration.coordinate_system = CoordinateSystem::Cartesian;
    state.configuration.stagger_type = StaggerType::CGrid;
    state.configuration.domain_motion_mode =
        tmv::init::config::DomainMotionMode::Stationary;
    state.configuration.domain_frame =
        tmv::domain::HorizontalDomainFrame::stationary();
    state.configuration.microphysics_scheme = "none";
    state.configuration.dynamics_scheme_name =
        "klemp_wilhelmson_split_cartesian_cgrid";
    state.configuration.damping_enabled = false;
    state.configuration.w_only_top_sponge = false;
    state.configuration.lateral_sponge_enabled = false;
    state.configuration.coriolis_enabled = false;
    state.configuration.scientific_diagnostics_every_steps = 0;
    state.configuration.validation_policy.mode = tmv::GuardMode::Strict;
    state.configuration.validation_policy.fail_on = tmv::GuardFailOn::Both;
    return state;
}

void no_op_nested_child_step(const SimulationStateView&, double, double, void*)
{
}

void initialize_relocation_benchmark_active_physics()
{
    auto& storage = global_simulation_storage;
    BoundaryLayerConfig pbl;
    pbl.scheme_id = "slab";
    pbl.dt_pbl = 60.0;
    SurfaceConfig surface;
    surface.Tsfc = 305.0;
    surface.qsfc = 0.014;
    surface.albedo = 0.2;
    surface.emissivity = 0.95;
    initialize_boundary_layer(
        storage.runtime.boundary_layer, "slab", pbl, surface);
    step_boundary_layer(make_simulation_state_view(storage), 0.0);

    RadiationConfig radiation;
    radiation.scheme_id = "simple_grey";
    radiation.do_lw = true;
    radiation.do_sw = false;
    radiation.dt_radiation = 300.0;
    initialize_radiation(
        storage.runtime.radiation, "simple_grey", radiation);
    step_radiation(make_simulation_state_view(storage), 0.0);

    tmv::chaos::ChaosConfig chaos;
    chaos.scheme_id = "full_stochastic";
    chaos.seed = 948273ULL;
    chaos.member_id = 7;
    chaos.dt_chaos = 1.0;
    chaos.tau_t = 20.0;
    chaos.Lx = 2000.0;
    chaos.Ly = 1800.0;
    chaos.taper_id = "none";
    chaos.apply_to_tendencies = {"pbl"};
    chaos.alpha_tend["pbl"] = 0.25;
    tmv::chaos::initialize_chaos(make_simulation_state_view(storage), chaos);
    tmv::chaos::apply_chaos_tendencies(make_simulation_state_view(storage));
    tmv::chaos::step_chaos_noise(make_simulation_state_view(storage), 2.0);
    tmv::chaos::apply_chaos_tendencies(make_simulation_state_view(storage));
}

void initialize_relocation_benchmark_milbrandt(
    NestedParentMicrophysicsSnapshotV1& previous_microphysics)
{
    auto& storage = global_simulation_storage;
    storage.configuration.microphysics_scheme = "milbrandt";
    storage.runtime.microphysics.scheme = tmv::create_microphysics_scheme(
        "milbrandt", storage.configuration.microphysics_config);
    if (!storage.runtime.microphysics.scheme)
        throw std::runtime_error("failed to construct benchmark Milbrandt");
    storage.runtime.microphysics.scheme->prepare_prognostic_state(
        storage.fields.rho, storage.fields.qc, storage.fields.qr,
        storage.fields.qi, storage.fields.qs, storage.fields.qg,
        storage.fields.qh);
    capture_nested_parent_microphysics_snapshot(
        storage, previous_microphysics);
    previous_microphysics.sample_time_s = 0.0;
}

void prepare_relocation_benchmark_child(
    int child_nx, int child_ny, int nz, int margin_parent_cells,
    const PrognosticFieldStorage& previous_parent, bool production_setup,
    const NestedParentMicrophysicsSnapshotV1* previous_microphysics)
{
    auto& state = global_simulation_storage;
    constexpr int ratio = 2;
    tmv::domain::OneWayNestRelationshipInput input;
    input.parent_cells_x = state.grid.nr - 2;
    input.parent_cells_y = state.grid.nth - 2;
    input.parent_levels_z = nz;
    input.parent_dx_m = state.grid.dx;
    input.parent_dy_m = state.grid.dy;
    input.parent_dt_s = 2.0;
    input.parent_ratio_x = ratio;
    input.parent_ratio_y = ratio;
    input.parent_time_step_ratio = ratio;
    input.child_origin_parent_cell_i = margin_parent_cells;
    input.child_origin_parent_cell_j = margin_parent_cells;
    input.child_cells_x = child_nx;
    input.child_cells_y = child_ny;
    input.child_levels_z = nz;
    input.relaxation_width_child_cells = 1;
    input.boundary_stencil_parent_cells = 1;
    state.configuration.nesting.enabled = true;
    state.configuration.nesting.parent_ratio_x = ratio;
    state.configuration.nesting.parent_ratio_y = ratio;
    state.configuration.nesting.parent_time_step_ratio = ratio;
    state.configuration.nesting.child_origin_parent_cell_i =
        margin_parent_cells;
    state.configuration.nesting.child_origin_parent_cell_j =
        margin_parent_cells;
    state.configuration.nesting.child_cells_x = child_nx;
    state.configuration.nesting.child_cells_y = child_ny;
    state.configuration.nesting.relaxation_width_child_cells = 1;
    state.configuration.nesting.boundary_stencil_parent_cells = 1;
    state.configuration.nesting.relaxation_e_folding_time_s = 4.0;
    prepare_nested_execution_relationship(
        state, tmv::domain::prepare_one_way_nest_relationship(input), 0.0);
    initialize_nested_shared_level_reference_state(state);
    NestedChildInitializationReport initialization;
    initialize_nested_child_physical_primary_state(
        state, previous_parent, initialization);
    NestedChildRelaxationReport initial_boundary;
    apply_nested_child_physical_relaxation(
        state, previous_parent, {4.0}, 1.0, initial_boundary);
    initialize_nested_child_level_runtime(state);
    if (previous_microphysics)
    {
        NestedChildInitializationReport private_initialization;
        initialize_nested_child_microphysics_private_state(
            state, previous_parent, *previous_microphysics,
            private_initialization);
    }
    require_nested_child_level_runtime_ready(state);
    for (int phase = 0; phase < ratio; ++phase)
    {
        NestedChildSubstepReport report;
        if (production_setup)
            advance_nested_child_dynamics_substep(
                state, previous_parent, {4.0}, report,
                previous_microphysics);
        else
            advance_nested_child_substep(
                state, previous_parent, {4.0}, no_op_nested_child_step,
                nullptr, report);
    }
    require_valid_nested_execution_relationship(state);
}

#endif

void prepare_acoustic_child(
    const PrognosticFieldStorage& previous_parent,
    const AcousticCaseSpec& spec = {})
{
    auto& state = global_simulation_storage;
    tmv::domain::OneWayNestRelationshipInput input;
    input.parent_cells_x = 32;
    input.parent_cells_y = spec.parent_nth - 2;
    input.parent_levels_z = 2;
    input.parent_dx_m = 500.0;
    input.parent_dy_m = 500.0;
    input.parent_dt_s = 0.5;
    input.parent_ratio_x = spec.refinement_ratio;
    input.parent_ratio_y = spec.refinement_ratio;
    input.parent_time_step_ratio = spec.refinement_ratio;
    input.child_origin_parent_cell_i = 4;
    input.child_origin_parent_cell_j = spec.child_origin_parent_j;
    input.child_cells_x = 24 * spec.refinement_ratio;
    input.child_cells_y =
        spec.child_parent_span_y * spec.refinement_ratio;
    input.child_levels_z = 2;
    input.relaxation_width_child_cells = spec.refinement_ratio;
    input.boundary_stencil_parent_cells = 1;
    state.configuration.nesting.enabled = true;
    state.configuration.nesting.parent_ratio_x = input.parent_ratio_x;
    state.configuration.nesting.parent_ratio_y = input.parent_ratio_y;
    state.configuration.nesting.parent_time_step_ratio =
        input.parent_time_step_ratio;
    state.configuration.nesting.child_origin_parent_cell_i =
        input.child_origin_parent_cell_i;
    state.configuration.nesting.child_origin_parent_cell_j =
        input.child_origin_parent_cell_j;
    state.configuration.nesting.child_cells_x = input.child_cells_x;
    state.configuration.nesting.child_cells_y = input.child_cells_y;
    state.configuration.nesting.relaxation_width_child_cells =
        input.relaxation_width_child_cells;
    state.configuration.nesting.boundary_stencil_parent_cells =
        input.boundary_stencil_parent_cells;
    state.configuration.nesting.relaxation_e_folding_time_s = 2.0;
    prepare_nested_execution_relationship(
        state, tmv::domain::prepare_one_way_nest_relationship(input), 0.0);
    initialize_nested_shared_level_reference_state(state);
    NestedChildInitializationReport initialization;
    initialize_nested_child_physical_primary_state(
        state, previous_parent, initialization);
    NestedChildRelaxationReport boundary;
    apply_nested_child_physical_relaxation(
        state, previous_parent, {2.0},
        0.5 / static_cast<double>(spec.refinement_ratio), boundary);
    initialize_nested_child_level_runtime(state);
}

void advance_parent_production_step(double step_s)
{
    auto& state = global_simulation_storage;
    const double start_time_s = state.time_s;
    const double admitted_step_s = tmv::choose_runtime_timestep();
    REQUIRE(admitted_step_s + 1.0e-12 >= step_s);
    const SimulationStateView parent_state = global_simulation_state();
    tmv::step_radiation(parent_state, start_time_s);
    tmv::step_boundary_layer(parent_state, start_time_s);
    tmv::chaos::step_chaos_noise(parent_state, step_s);
    tmv::chaos::apply_chaos_tendencies(parent_state);
    tmv::step_dynamics_new(step_s, start_time_s);
    state.time_s = start_time_s + step_s;
}

struct AcousticInterfaceMetrics
{
    double incident_pa = 0.0;
    double transmitted_pa = 0.0;
    double reflected_pa = 0.0;
    double incident_energy_j = 0.0;
    double parent_control_energy_j = 0.0;
    double parent_control_left_energy_j = 0.0;
    double transmitted_energy_j = 0.0;
    double reflected_energy_j = 0.0;
    double prior_incident_power_w = 0.0;
    double prior_parent_control_power_w = 0.0;
    double prior_parent_control_left_power_w = 0.0;
    double prior_transmitted_power_w = 0.0;
    double prior_reflected_power_w = 0.0;
    double prior_sample_time_s = 0.0;
    bool prior_power_available = false;
};

void sample_acoustic_interface(
    AcousticInterfaceMetrics& metrics,
    const PrognosticFieldStorage* incident_parent_fields = nullptr)
{
    const auto& state = global_simulation_storage;
    const auto& nested = state.nested_execution;
    const auto& parent_fields = incident_parent_fields != nullptr
        ? *incident_parent_fields
        : state.fields;
    constexpr double gamma =
        physical_constants::specific_heat_cp_jkgk /
        (physical_constants::specific_heat_cp_jkgk -
         physical_constants::gas_constant_dry_air_jkgk);
    double incident_power_w = 0.0;
    double parent_control_power_w = 0.0;
    double parent_control_left_power_w = 0.0;
    double transmitted_power_w = 0.0;
    double reflected_power_w = 0.0;
    const double parent_dy_m = nested.relationship.parent_dy_m;
    const double child_dy_m = nested.relationship.child_dy_m;

    const int transmitted_child_face =
        3 * nested.relationship.parent_ratio_x;
    REQUIRE(transmitted_child_face %
            nested.relationship.parent_ratio_x == 0);
    // The one-way parent remains an independent control. Aligning its face
    // with the child measurement removes ordinary propagation distance from
    // the interface comparison.
    const int parent_incident_face =
        nested.relationship.child_origin_parent_cell_i;
    const int parent_control_face = parent_incident_face +
        transmitted_child_face / nested.relationship.parent_ratio_x;
    const int parent_j_begin =
        1 + nested.relationship.child_origin_parent_cell_j;
    const int parent_j_end = parent_j_begin +
        nested.relationship.child_cells_y /
            nested.relationship.parent_ratio_y;
    for (int j = parent_j_begin; j < parent_j_end; ++j)
    {
        for (int k = 0; k < state.grid.nz; ++k)
        {
            const double base_pressure_pa = state.base.p0[k];
            const double base_density_kgm3 = state.base.rho0[k];
            const double impedance = base_density_kgm3 * std::sqrt(
                gamma * base_pressure_pa / base_density_kgm3);
            const auto face_characteristics = [&](int face) {
                const double pressure_prime = 0.5 *
                    (static_cast<double>(parent_fields.p[face][j][k]) +
                     static_cast<double>(parent_fields.p[face + 1][j][k])) -
                    base_pressure_pa;
                const double velocity = parent_fields.u[face][j][k];
                return std::pair<double, double>{
                    0.5 * (pressure_prime + impedance * velocity),
                    0.5 * (pressure_prime - impedance * velocity)};
            };
            const auto incident = face_characteristics(parent_incident_face);
            const auto control = face_characteristics(parent_control_face);
            metrics.incident_pa = std::max(
                metrics.incident_pa, std::abs(incident.first));
            const double layer_area_m2 = parent_dy_m *
                state.process_grid_metrics.dz[k];
            incident_power_w += incident.first * incident.first / impedance *
                layer_area_m2;
            parent_control_power_w +=
                control.first * control.first / impedance * layer_area_m2;
            parent_control_left_power_w +=
                control.second * control.second / impedance * layer_area_m2;
        }
    }

    const auto& child = nested.child.fields;
    const int verification_margin =
        2 * nested.relationship.relaxation_width_child_cells;
    for (int ci = transmitted_child_face;
         ci < nested.relationship.child_cells_x - verification_margin; ++ci)
    {
        const int i = ci + 1;
        for (int cj = nested.relationship.relaxation_width_child_cells;
             cj < nested.relationship.child_cells_y -
                 nested.relationship.relaxation_width_child_cells; ++cj)
        {
            const int j = cj + 1;
            for (int k = 0; k < nested.child.grid.nz; ++k)
            {
                const double base_pressure_pa = state.base.p0[k];
                const double base_density_kgm3 = state.base.rho0[k];
                const double impedance = base_density_kgm3 * std::sqrt(
                    gamma * base_pressure_pa / base_density_kgm3);
                const double pressure_prime =
                    static_cast<double>(child.p[i][j][k]) -
                    base_pressure_pa;
                const double centered_u = 0.5 *
                    (static_cast<double>(child.u[ci][j][k]) +
                     static_cast<double>(child.u[ci + 1][j][k]));
                const double rightgoing =
                    0.5 * (pressure_prime + impedance * centered_u);
                const double leftgoing =
                    0.5 * (pressure_prime - impedance * centered_u);
                metrics.transmitted_pa = std::max(
                    metrics.transmitted_pa, std::abs(rightgoing));
                metrics.reflected_pa = std::max(
                    metrics.reflected_pa, std::abs(leftgoing));
            }
        }
    }

    for (int cj = 0; cj < nested.relationship.child_cells_y; ++cj)
    {
        const int j = cj + 1;
        for (int k = 0; k < nested.child.grid.nz; ++k)
        {
            const double base_pressure_pa = state.base.p0[k];
            const double base_density_kgm3 = state.base.rho0[k];
            const double impedance = base_density_kgm3 * std::sqrt(
                gamma * base_pressure_pa / base_density_kgm3);
            const double layer_area_m2 = child_dy_m *
                nested.child.process_grid_metrics.dz[k];
            const auto characteristics = [&](int face) {
                const double pressure_prime = 0.5 *
                    (static_cast<double>(child.p[face][j][k]) +
                     static_cast<double>(child.p[face + 1][j][k])) -
                    base_pressure_pa;
                const double centered_u = child.u[face][j][k];
                return std::pair<double, double>{
                    0.5 * (pressure_prime + impedance * centered_u),
                    0.5 * (pressure_prime - impedance * centered_u)};
            };
            const auto transmitted = characteristics(transmitted_child_face);
            transmitted_power_w +=
                transmitted.first * transmitted.first / impedance *
                layer_area_m2;
            reflected_power_w +=
                transmitted.second * transmitted.second / impedance *
                layer_area_m2;
        }
    }

    const double sample_time_s = nested.child.time_s;
    if (metrics.prior_power_available)
    {
        const double interval_s = sample_time_s - metrics.prior_sample_time_s;
        REQUIRE(interval_s > 0.0);
        metrics.incident_energy_j += 0.5 * interval_s *
            (metrics.prior_incident_power_w + incident_power_w);
        metrics.parent_control_energy_j += 0.5 * interval_s *
            (metrics.prior_parent_control_power_w +
             parent_control_power_w);
        metrics.parent_control_left_energy_j += 0.5 * interval_s *
            (metrics.prior_parent_control_left_power_w +
             parent_control_left_power_w);
        metrics.transmitted_energy_j += 0.5 * interval_s *
            (metrics.prior_transmitted_power_w + transmitted_power_w);
        metrics.reflected_energy_j += 0.5 * interval_s *
            (metrics.prior_reflected_power_w + reflected_power_w);
    }
    metrics.prior_incident_power_w = incident_power_w;
    metrics.prior_parent_control_power_w = parent_control_power_w;
    metrics.prior_parent_control_left_power_w =
        parent_control_left_power_w;
    metrics.prior_transmitted_power_w = transmitted_power_w;
    metrics.prior_reflected_power_w = reflected_power_w;
    metrics.prior_sample_time_s = sample_time_s;
    metrics.prior_power_available = true;
}

void advance_acoustic_open_child_interval(
    const PrognosticFieldStorage& previous_parent)
{
    const int phases = global_simulation_storage.nested_execution.
        relationship.parent_time_step_ratio;
    for (int phase = 0; phase < phases; ++phase)
    {
        NestedChildSubstepReport report;
        advance_nested_child_dynamics_substep(
            global_simulation_storage, previous_parent, {2.0}, report);
    }
}

void advance_acoustic_parent_intervals(
    int interval_count, AcousticInterfaceMetrics& metrics)
{
    for (int interval = 0; interval < interval_count; ++interval)
    {
        const PrognosticFieldStorage previous_parent =
            global_simulation_storage.fields;
        advance_parent_production_step(0.5);
        roll_nested_parent_interval(global_simulation_storage);
        advance_acoustic_open_child_interval(previous_parent);
        sample_acoustic_interface(metrics);
    }
}

bool begin_acoustic_campaign(
    const AcousticCaseSpec& spec,
    AcousticInterfaceMetrics& metrics,
    const char* backend = "cpu")
{
    global_simulation_storage = make_acoustic_parent(spec);
    if (!initialize_parent_modules(backend))
        return false;
    set_step_diagnostics_enabled(false);
    const PrognosticFieldStorage previous_parent =
        global_simulation_storage.fields;
    advance_parent_production_step(0.5);
    prepare_acoustic_child(previous_parent, spec);
    sample_acoustic_interface(metrics, &previous_parent);
    advance_acoustic_open_child_interval(previous_parent);
    sample_acoustic_interface(metrics);
    return true;
}

void require_finite_primary(const PrognosticFieldStorage& fields)
{
    const Field3D* all[] = {
        &fields.rho, &fields.p, &fields.u, &fields.v, &fields.w,
        &fields.theta, &fields.qv, &fields.qc, &fields.qr, &fields.qi,
        &fields.qs, &fields.qg, &fields.qh, &fields.tracer, &fields.tke};
    for (const Field3D* field : all)
        for (std::size_t n = 0; n < field->size(); ++n)
            REQUIRE(std::isfinite(field->data()[n]));
}

void require_acoustic_metrics_exact(
    const AcousticInterfaceMetrics& actual,
    const AcousticInterfaceMetrics& expected)
{
    REQUIRE(actual.incident_pa == expected.incident_pa);
    REQUIRE(actual.transmitted_pa == expected.transmitted_pa);
    REQUIRE(actual.reflected_pa == expected.reflected_pa);
    REQUIRE(actual.incident_energy_j == expected.incident_energy_j);
    REQUIRE(actual.parent_control_energy_j ==
            expected.parent_control_energy_j);
    REQUIRE(actual.parent_control_left_energy_j ==
            expected.parent_control_left_energy_j);
    REQUIRE(actual.transmitted_energy_j == expected.transmitted_energy_j);
    REQUIRE(actual.reflected_energy_j == expected.reflected_energy_j);
    REQUIRE(actual.prior_incident_power_w ==
            expected.prior_incident_power_w);
    REQUIRE(actual.prior_parent_control_power_w ==
            expected.prior_parent_control_power_w);
    REQUIRE(actual.prior_parent_control_left_power_w ==
            expected.prior_parent_control_left_power_w);
    REQUIRE(actual.prior_transmitted_power_w ==
            expected.prior_transmitted_power_w);
    REQUIRE(actual.prior_reflected_power_w ==
            expected.prior_reflected_power_w);
    REQUIRE(actual.prior_sample_time_s == expected.prior_sample_time_s);
    REQUIRE(actual.prior_power_available == expected.prior_power_available);
}

void require_complete_interface_ledger(
    const NestedInterfaceSourceLedgerV2& interface,
    std::uint64_t expected_applications)
{
    REQUIRE(interface.available);
    REQUIRE(interface.application_count == expected_applications);
    const tmv::domain::ConservedScalarRelaxationReport* scalar_sources[] = {
        &interface.cumulative.theta, &interface.cumulative.qv,
        &interface.cumulative.qc, &interface.cumulative.qr,
        &interface.cumulative.qi, &interface.cumulative.qs,
        &interface.cumulative.qg, &interface.cumulative.qh,
        &interface.cumulative.tracer, &interface.cumulative.tke};
    for (const auto* source : scalar_sources)
    {
        REQUIRE(source->dry_storage_residual_kg ==
                source->target_dry_mass_change_kg -
                source->committed_dry_mass_change_kg);
        REQUIRE(source->carrier_storage_residual ==
                source->target_carrier_change -
                source->committed_carrier_change);
        REQUIRE(std::isfinite(source->dry_storage_residual_kg));
        REQUIRE(std::isfinite(source->carrier_storage_residual));
    }
    const auto& momentum = interface.cumulative.momentum;
    REQUIRE(momentum.x_storage_residual_kgms ==
            momentum.target_x_change_kgms -
            momentum.committed_x_change_kgms);
    REQUIRE(momentum.y_storage_residual_kgms ==
            momentum.target_y_change_kgms -
            momentum.committed_y_change_kgms);
    REQUIRE(momentum.z_storage_residual_kgms ==
            momentum.target_z_change_kgms -
            momentum.committed_z_change_kgms);
    const auto& pressure = interface.cumulative.pressure;
    REQUIRE(pressure.storage_residual_pa_m3 ==
            pressure.target_pressure_volume_integral_pa_m3 -
            pressure.committed_pressure_volume_integral_pa_m3);
    const auto& energy = interface.cumulative.energy;
    const auto& change = energy.committed_change;
    REQUIRE(energy.reservoir_sum_closure_residual_j == change.total -
            (change.dry_internal + change.water_sensible +
             change.phase_reference + change.resolved_kinetic +
             change.gravitational + change.subgrid_tke));
    const double energy_values[] = {
        change.dry_internal, change.water_sensible,
        change.phase_reference, change.resolved_kinetic,
        change.gravitational, change.subgrid_tke,
        change.total_water_mass, change.total,
        energy.reservoir_sum_closure_residual_j};
    for (double value : energy_values)
        REQUIRE(std::isfinite(value));
}

double maximum_absolute_value(const Field3D& field)
{
    double maximum = 0.0;
    for (std::size_t n = 0; n < field.size(); ++n)
        maximum = std::max(
            maximum, std::abs(static_cast<double>(field.data()[n])));
    return maximum;
}

using ActivePhysicsKernelDispatchCounts = std::array<std::uint64_t, 4>;

constexpr std::array<nimbus::compute::KernelId, 4>
    k_active_physics_kernel_ids{
        nimbus::compute::KernelId::KesslerPointwise,
        nimbus::compute::KernelId::KesslerSedimentation,
        nimbus::compute::KernelId::SmagorinskyCoefficients,
        nimbus::compute::KernelId::SmagorinskyTendencies};

ActivePhysicsKernelDispatchCounts active_physics_kernel_dispatch_counts(
    const ComputeBackend& backend)
{
    ActivePhysicsKernelDispatchCounts counts{};
    for (std::size_t index = 0; index < counts.size(); ++index)
    {
        counts[index] = backend.successful_kernel_dispatch_count(
            k_active_physics_kernel_ids[index]);
    }
    return counts;
}

void require_active_physics_kernel_dispatches(
    const ComputeBackend& backend,
    const ActivePhysicsKernelDispatchCounts& before)
{
    for (std::size_t index = 0; index < before.size(); ++index)
    {
        const auto kernel = k_active_physics_kernel_ids[index];
        const std::uint64_t after =
            backend.successful_kernel_dispatch_count(kernel);
        INFO("kernel=" <<
             nimbus::compute::kernel_manifest_entry(kernel).operation_id);
        REQUIRE(after > before[index]);
    }
}

void require_active_kessler_smagorinsky_provenance(
    const MicrophysicsRuntimeState& microphysics,
    const TurbulenceRuntimeState& turbulence)
{
    REQUIRE(microphysics.scheme);
    REQUIRE(microphysics.scheme->get_scheme_name() == "kessler");
    REQUIRE(microphysics.last_conservation_report.active);
    REQUIRE(std::isfinite(
        microphysics.last_conservation_report.tendency_imbalance_kg));
    REQUIRE(std::isfinite(
        microphysics.last_conservation_report.relative_imbalance));
    REQUIRE(microphysics.last_thermodynamic_repair_report
                .moist_energy_complete);
    REQUIRE(std::abs(
        microphysics.last_thermodynamic_repair_report
            .microphysics_heat_j) > 0.0);
    REQUIRE(std::isfinite(
        microphysics.last_thermodynamic_repair_report
            .moist_committed_energy_change_j));
    REQUIRE(std::isfinite(
        microphysics.last_thermodynamic_repair_report
            .moist_energy_closure_residual_j));

    REQUIRE(turbulence.scheme);
    REQUIRE(turbulence.scheme->name() == "smagorinsky");
    REQUIRE(turbulence.momentum_work_step_count > 0);
    REQUIRE(turbulence.last_explicit_substep_count > 0);
    REQUIRE(turbulence.last_momentum_work.moist_energy_complete);
    REQUIRE(std::isfinite(
        turbulence.last_momentum_work.committed_work_j));
    REQUIRE(std::isfinite(
        turbulence.last_momentum_work.moist_energy_closure_residual_j));
    REQUIRE(std::isfinite(
        turbulence.cumulative_irreversible_dissipation_j));
    REQUIRE(turbulence.cumulative_irreversible_dissipation_j > 0.0);
    REQUIRE(std::isfinite(
        turbulence.cumulative_stress_closure_residual_j));
    const double stress_scale = std::max({
        1.0,
        std::abs(turbulence.cumulative_requested_momentum_work_j),
        std::abs(turbulence.cumulative_open_boundary_work_j),
        turbulence.cumulative_irreversible_dissipation_j});
    REQUIRE(std::abs(turbulence.cumulative_stress_closure_residual_j)
            <= 128.0 * std::numeric_limits<float>::epsilon()
                 * stress_scale);
    const auto& tendencies = turbulence.held_tendencies;
    const double maximum_tendency = std::max({
        maximum_absolute_value(tendencies.dudt_sgs),
        maximum_absolute_value(tendencies.dvdt_sgs),
        maximum_absolute_value(tendencies.dwdt_sgs),
        maximum_absolute_value(tendencies.dthetadt_sgs),
        maximum_absolute_value(tendencies.dqvdt_sgs)});
    REQUIRE(maximum_tendency > 0.0);
}

void require_nested_device_field_parity(
    const char* name, const Field3D& cpu, const Field3D& device,
    double absolute_floor)
{
    REQUIRE(cpu.size_r() == device.size_r());
    REQUIRE(cpu.size_th() == device.size_th());
    REQUIRE(cpu.size_z() == device.size_z());
    double scale = 1.0;
    double maximum_difference = 0.0;
    for (std::size_t n = 0; n < cpu.size(); ++n)
    {
        scale = std::max({
            scale,
            std::abs(static_cast<double>(cpu.data()[n])),
            std::abs(static_cast<double>(device.data()[n]))});
        maximum_difference = std::max(
            maximum_difference,
            std::abs(static_cast<double>(device.data()[n]) -
                     static_cast<double>(cpu.data()[n])));
    }
    constexpr double operation_bound = 512.0;
    const double product = operation_bound *
        static_cast<double>(std::numeric_limits<float>::epsilon());
    const double relative_bound = product / (1.0 - product);
    const double tolerance = std::max(
        absolute_floor, relative_bound * scale);
    INFO(name << ": max_abs_difference=" << maximum_difference
              << " scale=" << scale << " tolerance=" << tolerance);
    REQUIRE(maximum_difference <= tolerance);
}

void require_nested_device_vector_parity(
    const char* name, const std::vector<float>& cpu,
    const std::vector<float>& device, double absolute_floor)
{
    REQUIRE(cpu.size() == device.size());
    double scale = 1.0;
    double maximum_difference = 0.0;
    for (std::size_t n = 0; n < cpu.size(); ++n)
    {
        REQUIRE(std::isfinite(cpu[n]));
        REQUIRE(std::isfinite(device[n]));
        scale = std::max({scale, std::abs(static_cast<double>(cpu[n])),
                          std::abs(static_cast<double>(device[n]))});
        maximum_difference = std::max(
            maximum_difference,
            std::abs(static_cast<double>(device[n]) -
                     static_cast<double>(cpu[n])));
    }
    constexpr double operation_bound = 512.0;
    const double product = operation_bound *
        static_cast<double>(std::numeric_limits<float>::epsilon());
    const double relative_bound = product / (1.0 - product);
    const double tolerance = std::max(
        absolute_floor, relative_bound * scale);
    INFO(name << ": max_abs_difference=" << maximum_difference
              << " scale=" << scale << " tolerance=" << tolerance);
    REQUIRE(maximum_difference <= tolerance);
}

bool portable_state_is_bitwise_equal(
    const PortableSchemeState& left,
    const PortableSchemeState& right)
{
    if (left.version != right.version || left.integers != right.integers ||
        left.reals != right.reals || left.fields.size() != right.fields.size())
        return false;
    for (std::size_t field = 0; field < left.fields.size(); ++field)
    {
        const Field3D& a = left.fields[field];
        const Field3D& b = right.fields[field];
        if (a.size_r() != b.size_r() || a.size_th() != b.size_th() ||
            a.size_z() != b.size_z() ||
            !std::equal(a.data(), a.data() + a.size(), b.data()))
            return false;
    }
    return true;
}

void advance_next_balanced_parent_interval(
    const tmv::domain::NestedRelaxationConfigV1& relaxation)
{
    auto& state = global_simulation_storage;
    const PrognosticFieldStorage previous_parent = state.fields;
    const double previous_parent_time_s = state.time_s;
    state.time_s += state.nested_execution.relationship.parent_dt_s;
    state.grid.dt = state.time_s - previous_parent_time_s;
    roll_nested_parent_interval(state);

    NestedChildSubstepReport first;
    NestedChildSubstepReport second;
    advance_nested_child_dynamics_substep(
        state, previous_parent, relaxation, first);
    advance_nested_child_dynamics_substep(
        state, previous_parent, relaxation, second);
    REQUIRE(second.end_time_s == state.time_s);
    REQUIRE(second.completed_child_substeps ==
            state.nested_execution.relationship.parent_time_step_ratio);
}

} // namespace

namespace
{
class LateFailingKessler final : public tmv::KesslerScheme
{
public:
    int calls = 0;
    bool reject_after_chaos = false;
    std::unique_ptr<MicrophysicsScheme> clone() const override
    { return std::make_unique<LateFailingKessler>(*this); }
    bool validate_prognostic_update(double, std::string& error) const override
    {
        if (!reject_after_chaos) return true;
        error = "late stochastic interval failure";
        return false;
    }
    void compute_tendencies(
        const Field3D& p, const Field3D& theta, const Field3D& rho, const Field3D& qv,
        const Field3D& qc, const Field3D& qr, const Field3D& qi, const Field3D& qs,
        const Field3D& qg, const Field3D& qh, double step,
        Field3D& a, Field3D& b, Field3D& c, Field3D& d,
        Field3D& e, Field3D& f, Field3D& g, Field3D& h) override
    {
        if (++calls == 2) throw std::runtime_error("late stochastic interval failure");
        tmv::KesslerScheme::compute_tendencies(p, theta, rho, qv, qc, qr, qi, qs, qg, qh, step, a, b, c, d, e, f, g, h);
    }
};

class MixedMomentProbe final : public MicrophysicsScheme
{
public:
    using MicrophysicsScheme::lower_boundary_vertical_velocity;
    using MicrophysicsScheme::vertical_velocity_at_scalar;
    using MicrophysicsScheme::vertical_velocity_field;
    Field3D ratio;
    Field3D concentration;
    std::unique_ptr<MicrophysicsScheme> clone() const override
    { return std::make_unique<MixedMomentProbe>(*this); }
    tmv::MicrophysicsPrognosticScalarList prognostic_scalars_for_transport() override
    { return {{"probe.number_per_kg", &ratio, true, tmv::MicrophysicsScalarStorage::DryAirMixingRatio},
              {"probe.number_per_m3", &concentration, true, tmv::MicrophysicsScalarStorage::VolumeConcentration}}; }
    std::string get_scheme_name() const override { return "none"; }
    int get_num_prognostic_vars() const override { return 2; }
    void compute_tendencies(
        const Field3D& p, const Field3D&, const Field3D&, const Field3D&,
        const Field3D&, const Field3D&, const Field3D&, const Field3D&,
        const Field3D&, const Field3D&, double,
        Field3D& a, Field3D& b, Field3D& c, Field3D& d,
        Field3D& e, Field3D& f, Field3D& g, Field3D& h) override
    {
        for (auto* output : {&a, &b, &c, &d, &e, &f, &g, &h})
            output->resize(p.size_r(), p.size_th(), p.size_z(), 0.0f);
    }
};
}

TEST_CASE("Thompson runtime initialization samples the owning grid before transport or snapshots",
          "[core][microphysics][geometry][private_units]")
{
    const std::string geometry = GENERATE("uniform", "nonuniform", "mapped");
    const bool        snapshot = GENERATE(false, true);
    CAPTURE(geometry, snapshot);
    ProcessStateRestore restore;
    global_simulation_storage               = make_parent();
    auto& state                             = global_simulation_storage;
    state.configuration.microphysics_scheme = "thompson";
    state.runtime.microphysics.scheme       = tmv::create_microphysics_scheme("thompson");
    if (geometry == "uniform")
    {
        state.grid.dz               = 500;
        state.grid.geometry.z_faces = {0, 500, 1000};
        state.grid.geometry.z       = {250, 750};
    }
    if (geometry == "mapped")
    {
        auto& terrain                           = state.runtime.terrain;
        terrain.config.scheme_id                = "bell";
        terrain.config.horizontal_placement_id  = "earth_fixed";
        terrain.config.ztop                     = 600;
        terrain.config.bell.h0                  = 80;
        terrain.config.bell.a                   = 1600;
        terrain.config.feature_center_earth_x_m = 4000;
        terrain.config.feature_center_earth_y_m = 3200;
        tmv::EarthFixedTerrainBuildRequest request;
        request.nr                     = 8;
        request.nth                    = 8;
        request.nz                     = 2;
        request.dx_m                   = state.grid.dx;
        request.dy_m                   = state.grid.dy;
        request.feature_center_earth_m = {4000, 3200};
        const auto built               = tmv::build_earth_fixed_terrain_geometry(terrain.config, request);
        terrain.metrics                = built.metrics;
        terrain.topography             = built.topography;
    }
    if (snapshot)
    {
        NestedParentMicrophysicsSnapshotV1 captured;
        capture_nested_parent_microphysics_snapshot(state, captured);
    }
    else
    {
        tmv::prepare_microphysics_prognostics(make_simulation_state_view(state), *state.runtime.microphysics.scheme);
    }
    const auto moments = state.runtime.microphysics.scheme->prognostic_scalars_for_transport();
    REQUIRE(moments.size() == 5);
    for (int i = 0; i < 8; ++i)
    {
        for (int j = 0; j < 8; ++j)
        {
            const double height        = geometry == "mapped"
                                             ? .5 * (state.runtime.terrain.metrics.z_face(i, j, 2) -
                                                     state.runtime.terrain.metrics.z_face(i, j, 0))
                                             : .5 * (state.grid.geometry.z_faces[2] - state.grid.geometry.z_faces[0]);
            const double concentration = 50e6 + 300e6 * std::exp(std::log(1.0 / 6.0) * height / 800.0);
            for (int k = 0; k < 2; ++k)
            {
                CHECK((*moments[3].field)(i, j, k) == Approx(concentration / state.fields.rho(i, j, k)).epsilon(2e-7));
            }
        }
    }
    (*moments[3].field)(3, 3, 1) *= .75f;
    const float evolved = (*moments[3].field)(3, 3, 1);
    tmv::prepare_microphysics_prognostics(make_simulation_state_view(state), *state.runtime.microphysics.scheme);
    REQUIRE((*moments[3].field)(3, 3, 1) == evolved);
}

TEST_CASE("Microphysics samples the implicit ground face at the first C-grid cell",
          "[core][microphysics][geometry][private_units][ground_velocity]")
{
    const std::string   geometry = GENERATE("flat", "earth_fixed", "grid_attached");
    ProcessStateRestore restore;
    global_simulation_storage                      = make_parent();
    auto& storage                                  = global_simulation_storage;
    storage.configuration.domain_frame.velocity_ms = {3, -2};
    storage.fields.u.fill(4);
    storage.fields.v.fill(-3);
    storage.fields.w.fill(2);
    for (int i = 0; i < 8; ++i)
    {
        for (int j = 0; j < 8; ++j)
        {
            storage.fields.w(i, j, 1) = 6;
        }
    }
    if (geometry != "flat")
    {
        auto& terrain                           = storage.runtime.terrain;
        terrain.config.scheme_id                = "bell";
        terrain.config.horizontal_placement_id  = "earth_fixed";
        terrain.config.ztop                     = 600;
        terrain.config.bell.h0                  = 80;
        terrain.config.bell.a                   = 1600;
        terrain.config.feature_center_earth_x_m = 4000;
        terrain.config.feature_center_earth_y_m = 3200;
        tmv::EarthFixedTerrainBuildRequest request;
        request.nr                             = 8;
        request.nth                            = 8;
        request.nz                             = 2;
        request.dx_m                           = storage.grid.dx;
        request.dy_m                           = storage.grid.dy;
        request.feature_center_earth_m         = {4000, 3200};
        const auto built                       = tmv::build_earth_fixed_terrain_geometry(terrain.config, request);
        terrain.metrics                        = built.metrics;
        terrain.topography                     = built.topography;
        terrain.config.horizontal_placement_id = geometry;
    }
    auto scheme = std::make_unique<MixedMomentProbe>();
    tmv::bind_microphysics_velocity(make_simulation_state_view(storage), *scheme);
    for (int i = 1; i < 7; ++i)
    {
        for (int j = 1; j < 7; ++j)
        {
            const double u       = geometry == "grid_attached" ? 1 : 4;
            const double v       = geometry == "grid_attached" ? -1 : -3;
            const auto&  metrics = storage.runtime.terrain.metrics;
            const double ground = geometry == "flat" ? 0 : -u * metrics.Gx_face(i, j, 0) - v * metrics.Gy_face(i, j, 0);
            CAPTURE(geometry, i, j, ground);
            CHECK(scheme->vertical_velocity_at_scalar(i, j, 0) == Approx(.5 * (2 + ground)).epsilon(1e-12));
            CHECK(scheme->vertical_velocity_at_scalar(i, j, 1) == 4);
        }
    }
    CHECK(scheme->vertical_velocity_at_scalar(0, 0, 0) == scheme->vertical_velocity_at_scalar(1, 1, 0));
    const double        before = scheme->vertical_velocity_at_scalar(3, 3, 0);
    std::vector<double> invalid(64, 0);
    invalid.back() = std::numeric_limits<double>::quiet_NaN();
    REQUIRE_THROWS(scheme->set_vertical_velocity_field(storage.fields.w, true, invalid));
    CHECK(scheme->vertical_velocity_at_scalar(3, 3, 0) == before);
    const auto clone = scheme->clone();
    CHECK(dynamic_cast<const MixedMomentProbe&>(*clone).vertical_velocity_field() == nullptr);
    Field3D relocated = storage.fields.w;
    relocated.fill(10);
    scheme->rebind_vertical_velocity_field(relocated);
    CHECK(scheme->vertical_velocity_at_scalar(3, 3, 0) == Approx(before + 4));
    Field3D incompatible(1, 1, 1, 0);
    REQUIRE_THROWS(scheme->rebind_vertical_velocity_field(incompatible));
    CHECK(scheme->vertical_velocity_field() == &relocated);
    scheme->set_vertical_velocity_field(storage.fields.w, false);
    CHECK(scheme->lower_boundary_vertical_velocity().empty());
    CHECK(scheme->vertical_velocity_at_scalar(3, 3, 0) == 2);
}

TEST_CASE("Private scalar halos preserve mixing ratios and publish atomically",
          "[core][boundary][private_units][recycling]")
{
    const bool          staggered = GENERATE(false, true);
    ProcessStateRestore restore;
    global_simulation_storage = make_parent();
    auto& state               = global_simulation_storage;
    state.grid.nz             = 4;
    state.grid.geometry.initialize(
        8, 8, 4, 1000, 300, 800, CoordinateSystem::Cartesian, staggered ? StaggerType::CGrid : StaggerType::Collocated);
    state.fields.rho.resize(8, 8, 4, 1);
    state.configuration.lateral_bc_west = state.configuration.lateral_bc_east = tmv::init::config::LateralBC::Periodic;
    state.configuration.lateral_bc_south = tmv::init::config::LateralBC::ZeroGradient;
    state.configuration.lateral_bc_north = tmv::init::config::LateralBC::Radiative;
    auto  scheme                         = std::make_unique<MixedMomentProbe>();
    auto* probe                          = scheme.get();
    probe->ratio.resize(8, 8, 4, -999);
    probe->concentration.resize(8, 8, 4, -999);
    for (int i = 0; i < 8; ++i)
    {
        for (int j = 0; j < 8; ++j)
        {
            for (int k = 0; k < 4; ++k)
            {
                state.fields.rho(i, j, k) = float(.8 + .1 * i + .02 * j + .03 * k);
                if (i > 0 && i < 7 && j > 0 && j < 7 && (staggered || (k > 0 && k < 3)))
                {
                    probe->ratio(i, j, k)         = float(100 * i + 10 * j + k + 1);
                    probe->concentration(i, j, k) = probe->ratio(i, j, k) * state.fields.rho(i, j, k);
                }
            }
        }
    }
    state.runtime.microphysics.scheme   = std::move(scheme);
    const Field3D before_ratio          = probe->ratio;
    const Field3D before_concentration  = probe->concentration;
    const int     last_k                = staggered ? 3 : 2;
    probe->concentration(6, 6, last_k)  = std::numeric_limits<float>::infinity();
    const Field3D invalid_concentration = probe->concentration;
    REQUIRE_THROWS(tmv::rebuild_microphysics_boundary_conditions(make_simulation_state_view(state)));
    REQUIRE(std::memcmp(probe->ratio.data(), before_ratio.data(), probe->ratio.size() * sizeof(float)) == 0);
    REQUIRE(std::memcmp(probe->concentration.data(),
                        invalid_concentration.data(),
                        probe->concentration.size() * sizeof(float)) == 0);
    probe->concentration = before_concentration;
    tmv::rebuild_microphysics_boundary_conditions(make_simulation_state_view(state));
    for (int i = 0; i < 8; ++i)
    {
        for (int j = 0; j < 8; ++j)
        {
            for (int k = 0; k < 4; ++k)
            {
                const int si = i == 0 ? 6 : (i == 7 ? 1 : i);
                const int sj = std::clamp(j, 1, 6);
                const int sk = staggered ? k : std::clamp(k, 1, 2);
                CHECK(probe->ratio(i, j, k) == float(100 * si + 10 * sj + sk + 1));
                CHECK(probe->concentration(i, j, k) / state.fields.rho(i, j, k) ==
                      Approx(probe->ratio(i, j, k)).epsilon(2e-7));
            }
        }
    }
}

TEST_CASE("Recycling boundary rebuild supplies matching private and primary halos",
          "[core][boundary][private_units][recycling]")
{
    ProcessStateRestore restore;
    global_simulation_storage           = make_parent();
    auto& state                         = global_simulation_storage;
    state.configuration.lateral_bc_west = state.configuration.lateral_bc_east = tmv::init::config::LateralBC::Periodic;
    state.configuration.lateral_bc_south                                      = state.configuration.lateral_bc_north =
        tmv::init::config::LateralBC::Radiative;
    auto  scheme = std::make_unique<MixedMomentProbe>();
    auto* probe  = scheme.get();
    probe->ratio.resize(8, 8, 2, 0);
    probe->concentration.resize(8, 8, 2, 0);
    state.fields.qc.fill(0);
    for (int i = 1; i < 7; ++i)
    {
        for (int j = 1; j < 7; ++j)
        {
            for (int k = 0; k < 2; ++k)
            {
                state.fields.qc(i, j, k)      = float(.001 + .0001 * i + .00001 * j);
                probe->ratio(i, j, k)         = state.fields.qc(i, j, k) * 1e8f;
                probe->concentration(i, j, k) = probe->ratio(i, j, k) * state.fields.rho(i, j, k);
            }
        }
    }
    state.runtime.microphysics.scheme = std::move(scheme);
    tmv::rebuild_boundary_conditions_after_domain_recycling(make_simulation_state_view(state));
    for (int i = 0; i < 8; ++i)
    {
        for (int j = 0; j < 8; ++j)
        {
            for (int k = 0; k < 2; ++k)
            {
                CHECK(probe->ratio(i, j, k) / state.fields.qc(i, j, k) == Approx(1e8).epsilon(2e-7));
                CHECK(probe->concentration(i, j, k) / state.fields.rho(i, j, k) ==
                      Approx(probe->ratio(i, j, k)).epsilon(2e-7));
            }
        }
    }
}

TEST_CASE("Production RK transports mixing ratios and volume concentrations consistently",
          "[core][rk][private_units][conservation]")
{
    for (const std::string integrator : {"rk3", "rk4"})
    {
        ProcessStateRestore restore;
        global_simulation_storage = make_parent();
        auto& state = global_simulation_storage;
        state.grid.geometry.z_faces = {0.0, 300.0, 600.0};
        state.grid.geometry.z = {150.0, 450.0};
        state.process_grid_metrics.dz = {300.0, 300.0};
        state.process_grid_metrics.z_int = state.grid.geometry.z_faces;
        state.configuration.lateral_bc_west = state.configuration.lateral_bc_east =
            state.configuration.lateral_bc_south = state.configuration.lateral_bc_north =
                tmv::init::config::LateralBC::Periodic;
        REQUIRE(initialize_parent_modules());
        TimeSteppingConfig time;
        time.scheme_id = integrator;
        time.split_acoustic = false;
        initialize_time_stepping(state.runtime.time_stepping, integrator, time);
        auto probe = std::make_unique<MixedMomentProbe>();
        probe->ratio.resize(state.grid.nr, state.grid.nth, state.grid.nz, 0.0f);
        probe->concentration.resize(state.grid.nr, state.grid.nth, state.grid.nz, 0.0f);
        double initial_inventory = 0.0;
        for (int i = 0; i < state.grid.nr; ++i)
            for (int j = 0; j < state.grid.nth; ++j)
                for (int k = 0; k < state.grid.nz; ++k)
                {
                    const double phase = 6.283185307179586 * (i - 1) / (state.grid.nr - 2);
                    state.fields.u[i][j][k] = 5.0 * std::sin(phase);
                    probe->ratio[i][j][k] = 1000.0 + 50.0 * std::cos(phase);
                    probe->concentration[i][j][k] = probe->ratio[i][j][k] * state.fields.rho[i][j][k];
                    if (i > 0 && i < state.grid.nr - 1 && j > 0 && j < state.grid.nth - 1)
                        initial_inventory += probe->concentration[i][j][k];
                }
        auto* observed = probe.get();
        state.runtime.microphysics.scheme = std::move(probe);
        const Field3D density_before = state.fields.rho;
        step_radiation(global_simulation_state(), state.time_s);
        step_boundary_layer(global_simulation_state(), state.time_s);
        tmv::step_dynamics_new(0.05, state.time_s);
        double inventory = 0.0;
        for (int i = 1; i < state.grid.nr - 1; ++i)
            for (int j = 1; j < state.grid.nth - 1; ++j)
                for (int k = 0; k < state.grid.nz; ++k)
                {
                    REQUIRE(observed->concentration[i][j][k] ==
                        Approx(observed->ratio[i][j][k] * state.fields.rho[i][j][k]).epsilon(5e-6));
                    inventory += observed->concentration[i][j][k];
                }
        REQUIRE(inventory == Approx(initial_inventory).epsilon(5e-6));
        REQUIRE(std::memcmp(density_before.data(), state.fields.rho.data(),
            density_before.size() * sizeof(float)) != 0);
    }
}

TEST_CASE("Sedimentation subcycling retains native process ownership and water closure",
          "[core][microphysics][subcycling][physical_backends]")
{
    std::array<Field3D, 4> reference;
    for (const char* backend : {"cpu", "vulkan", "metal"})
    {
        ProcessStateRestore restore;
        global_simulation_storage = make_parent();
        auto& state = global_simulation_storage;
        state.grid.geometry.z_faces = {0.0, 300.0, 600.0};
        state.grid.geometry.z = {150.0, 450.0};
        state.process_grid_metrics.dz = {300.0, 300.0};
        state.process_grid_metrics.z_int = state.grid.geometry.z_faces;
        if (!initialize_parent_modules(backend)) continue;
        state.configuration.microphysics_scheme = "kessler";
        state.runtime.microphysics.scheme = tmv::create_microphysics_scheme("kessler");
        state.fields.qc.fill(0.0f);
        state.fields.qr.fill(0.001f);
        for (std::size_t index = 0; index < state.fields.qv.size(); ++index)
            state.fields.qv.data()[index] = static_cast<float>(tmv::thermodynamics::saturation_mixing_ratio_water(
                tmv::thermodynamics::theta_to_temperature(state.fields.theta.data()[index], state.fields.p.data()[index]),
                state.fields.p.data()[index]));
        step_radiation(global_simulation_state(), state.time_s);
        step_boundary_layer(global_simulation_state(), state.time_s);
        const auto inventory = [&] {
            double value = 0.0;
            for (int i = 1; i < state.grid.nr - 1; ++i)
                for (int j = 1; j < state.grid.nth - 1; ++j)
                    for (int k = 0; k < state.grid.nz; ++k)
                        value += state.fields.rho[i][j][k] * 300.0 *
                            (static_cast<double>(state.fields.qv[i][j][k]) + state.fields.qc[i][j][k] + state.fields.qr[i][j][k]);
            return value;
        };
        const double before = inventory();
        const double start_time = state.time_s;
        tmv::reset_microphysics_perf_stats();
        step_microphysics(global_simulation_state(), 100.0);
        const auto substeps = tmv::microphysics_perf_snapshot().sedimentation_substeps;
        REQUIRE(substeps > 1);
        REQUIRE(state.time_s == start_time);
        double fallout = 0.0;
        for (std::size_t index = 0; index < state.surface.accumulated_rainfall_mm.size(); ++index)
            fallout += static_cast<double>(state.surface.accumulated_rainfall_mm[index]) +
                state.surface.accumulated_rainfall_residual_mm[index];
        REQUIRE(inventory() + fallout == Approx(before).epsilon(3e-6));
        REQUIRE(fallout > 0.0);
        if (std::string(backend) != "cpu")
        {
            REQUIRE(state.runtime.compute.backend->successful_kernel_dispatch_count(
                nimbus::compute::KernelId::KesslerPointwise) == substeps);
            REQUIRE(state.runtime.compute.backend->successful_kernel_dispatch_count(
                nimbus::compute::KernelId::KesslerSedimentation) == substeps);
        }
        const std::array<const Field3D*, 4> output{&state.fields.theta, &state.fields.qv, &state.fields.qc, &state.fields.qr};
        for (std::size_t field = 0; field < output.size(); ++field)
        {
            if (std::string(backend) == "cpu") reference[field] = *output[field];
            else for (std::size_t index = 0; index < output[field]->size(); ++index)
                REQUIRE(output[field]->data()[index] == Approx(reference[field].data()[index]).epsilon(5e-6).margin(1e-9));
        }
        std::cout << "[microphysics-subcycle] backend=" << backend << " interval_s=100 substeps=" << substeps << '\n';
    }
}

TEST_CASE("Microphysics interval rollback restores stochastic ownership",
          "[core][microphysics][subcycling][chaos][transaction]")
{
    const double interval = GENERATE(0.1, 100.0);
    ProcessStateRestore restore;
    global_simulation_storage = make_parent();
    REQUIRE(initialize_parent_modules());
    auto& state = global_simulation_storage;
    state.configuration.microphysics_scheme = "kessler";
    auto scheme = std::make_unique<LateFailingKessler>();
    scheme->reject_after_chaos = interval < 1.0;
    state.runtime.microphysics.scheme = std::move(scheme);
    state.fields.qr.fill(0.001f);
    tmv::chaos::ChaosConfig chaos;
    chaos.scheme_id = "full_stochastic";
    chaos.apply_to_tendencies = {"microphysics"};
    tmv::chaos::initialize_chaos(global_simulation_state(), chaos);
    step_chaos_noise(global_simulation_state(), 0.1);
    step_radiation(global_simulation_state(), state.time_s);
    step_boundary_layer(global_simulation_state(), state.time_s);
    state.runtime.chaos.diagnostics.mean_perturbation = -42.0;
    state.runtime.chaos.diagnostics.time_apply_tend = 17.0;
    REQUIRE_THROWS_WITH(step_microphysics(global_simulation_state(), interval),
        Catch::Matchers::Contains("late stochastic interval failure"));
    REQUIRE(state.fields.qr[1][1][0] == 0.001f);
    REQUIRE(state.runtime.chaos.diagnostics.mean_perturbation == -42.0);
    REQUIRE(state.runtime.chaos.diagnostics.time_apply_tend == 17.0);
    REQUIRE(dynamic_cast<LateFailingKessler&>(*state.runtime.microphysics.scheme).calls == 0);
}

TEST_CASE("required-native Smagorinsky rejects missing capability before CPU science",
          "[core][production][turbulence][backend][strict]")
{
    ProcessStateRestore restore;
    global_simulation_storage = SimulationStorageState{};

    constexpr int n = 4;
    Field3D u_field(n, n, n, 0.0f);
    Field3D v_field(n, n, n, 0.0f);
    Field3D w_field(n, n, n, 0.0f);
    Field3D rho_field(n, n, n, 1.0f);
    Field3D theta_field(n, n, n, 300.0f);
    std::vector<double> rho0_field(static_cast<std::size_t>(n), 1.0);

    tmv::GridMetrics grid;
    grid.dx = 100.0;
    grid.dy = 100.0;
    grid.dz.assign(static_cast<std::size_t>(n), 100.0);
    tmv::TurbulenceStateView state;
    state.u = &u_field;
    state.v = &v_field;
    state.w = &w_field;
    state.rho = &rho_field;
    state.theta = &theta_field;
    state.rho0 = &rho0_field;
    state.NR = n;
    state.NTH = n;
    state.NZ = n;
    state.c_grid = true;
    state.require_native_dispatch = true;

    tmv::TurbulenceConfig cfg;
    tmv::SmagorinskyScheme scheme;
    scheme.initialize(cfg);
    tmv::TurbulenceTendencies tendencies;

    REQUIRE_THROWS_WITH(
        scheme.compute(cfg, grid, state, tendencies, nullptr),
        Catch::Matchers::Contains(
            "requires native flat-grid stability-none capability"));
}

TEST_CASE("Completed velocity buffers retain native dual-cell masses and reject malformed inputs",
          "[core][native_slow][contract]")
{
    Field3D                                 rho(4, 4, 2, 2), u(4, 4, 2, 3), v(4, 4, 2, -4), w(4, 4, 2, 5);
    tmv::CartesianCGridMomentumFluxRegister flux;
    flux.configure(4, 4, 2, {1, 3, 1, 3, 0, 1});
    const auto inventory = [&]()
    { return flux.native_dual_inventory_buffers(rho, {u.data(), v.data(), w.data()}, rho.size(), 10, 20, 30); };
    REQUIRE(inventory() == std::array<double, 3>{432000, -576000, 360000});
    REQUIRE_THROWS(flux.native_dual_inventory_buffers(rho, {u.data(), v.data(), w.data()}, rho.size() - 1, 10, 20, 30));
    REQUIRE_THROWS(flux.native_dual_inventory_buffers(rho, {nullptr, v.data(), w.data()}, rho.size(), 10, 20, 30));
    w.data()[w.size() - 1] = std::numeric_limits<float>::quiet_NaN();
    REQUIRE_THROWS(inventory());
    w.fill(5);
    rho.data()[0] = 0;
    REQUIRE_THROWS(inventory());
    rho.fill(2);
    REQUIRE(inventory() == std::array<double, 3>{432000, -576000, 360000});
}

TEST_CASE("Native slow commits close captured momentum before acoustic stages",
          "[core][production][native_slow][metal][vulkan][device]")
{
    const std::string backend = GENERATE("metal", "vulkan");
    const bool        mapped  = GENERATE(false, true);
    CAPTURE(backend, mapped);
    ProcessStateRestore restore;
    global_simulation_storage = make_parent();
    REQUIRE(initialize_parent_modules(backend.c_str()));
    auto& storage                                                 = global_simulation_storage;
    storage.configuration.gpu_apply_slow                          = true;
    storage.configuration.admission.two_way_accelerator_reference = true;
    for (auto* field : {&storage.tendencies.dtheta_dt_rad, &storage.tendencies.dtheta_dt_pbl,
                        &storage.tendencies.dqv_dt_pbl, &storage.tendencies.du_dt_pbl,
                        &storage.tendencies.dv_dt_pbl, &storage.tendencies.dtke_dt_pbl,
                        &storage.tendencies.dqc_dt_pbl, &storage.tendencies.dqi_dt_pbl})
    {
        field->resize(8, 8, 2, 0);
    }
    if (mapped)
    {
        auto& terrain                           = storage.runtime.terrain;
        terrain.config.scheme_id                = "bell";
        terrain.config.horizontal_placement_id  = "earth_fixed";
        terrain.config.ztop                     = 600;
        terrain.config.bell.h0                  = 20;
        terrain.config.bell.a                   = 1600;
        terrain.config.feature_center_earth_x_m = 4000;
        terrain.config.feature_center_earth_y_m = 3200;
        tmv::EarthFixedTerrainBuildRequest request;
        request.nr                     = storage.grid.nr;
        request.nth                    = storage.grid.nth;
        request.nz                     = storage.grid.nz;
        request.dx_m                   = storage.grid.dx;
        request.dy_m                   = storage.grid.dy;
        request.feature_center_earth_m = {4000, 3200};
        const auto geometry            = tmv::build_earth_fixed_terrain_geometry(terrain.config, request);
        terrain.metrics                = geometry.metrics;
        terrain.topography             = geometry.topography;
    }
    auto& dynamics                                   = storage.runtime.dynamics;
    dynamics.momentum_interface_flux_capture_enabled = true;
    dynamics.momentum_interface_flux_register.configure(8, 8, 2, {2, 6, 2, 6, 0, 1});
    auto&      device = *storage.runtime.compute.backend;
    const auto before = device.successful_kernel_dispatch_count(nimbus::compute::KernelId::SlowApply);
    REQUIRE(tmv::choose_runtime_timestep(make_simulation_state_view(storage)) >= .2);
    REQUIRE_NOTHROW(tmv::step_dynamics_new(make_simulation_state_view(storage), .2, storage.time_s));
    REQUIRE(device.successful_kernel_dispatch_count(nimbus::compute::KernelId::SlowApply) > before);
    const auto& flux = dynamics.momentum_interface_flux_register;
    REQUIRE(flux.advective_fluxes().contribution_count > 0);
    REQUIRE(flux.pressure_fluxes().contribution_count > 0);
    REQUIRE_THROWS(flux.pending_committed_operator_impulse());
    for (const auto* stage : {&flux.advective_fluxes(), &flux.pressure_fluxes()})
    {
        for (double value : stage->source_impulse)
        {
            REQUIRE(std::isfinite(value));
        }
        for (double value : stage->storage_impulse)
        {
            REQUIRE(std::isfinite(value));
        }
    }
    require_finite_primary(storage.fields);
}

TEST_CASE("PBL momentum work rejects cumulative overflow without publishing any owner",
          "[core][production][native_slow][atomicity]")
{
    ProcessStateRestore restore;
    global_simulation_storage = make_parent();
    initialize_parent_modules();
    auto&               storage = global_simulation_storage;
    BoundaryLayerConfig config;
    config.scheme_id = "slab";
    initialize_boundary_layer(storage.runtime.boundary_layer, "slab", config, SurfaceConfig{});
    auto&                                 runtime = storage.runtime.boundary_layer;
    const BoundaryLayerMomentumWorkReport initial{1, 2, 3, 6};
    commit_boundary_layer_momentum_work(make_simulation_state_view(storage), initial);
    runtime.cumulative_committed_momentum_work_j = std::numeric_limits<double>::max();
    const BoundaryLayerMomentumWorkReport overflowing{1, 2, 3, std::numeric_limits<double>::max()};
    REQUIRE_THROWS(commit_boundary_layer_momentum_work(make_simulation_state_view(storage), overflowing));
    REQUIRE(runtime.momentum_work_step_count == 1);
    REQUIRE(runtime.last_momentum_work.committed_work_j == 6);
    REQUIRE(runtime.cumulative_requested_momentum_work_j == 1);
    REQUIRE(runtime.cumulative_finite_step_momentum_correction_j == 2);
    REQUIRE(runtime.cumulative_momentum_storage_rounding_j == 3);
    REQUIRE(runtime.cumulative_committed_momentum_work_j == std::numeric_limits<double>::max());
    runtime.momentum_work_step_count = std::numeric_limits<std::uint64_t>::max();
    REQUIRE_THROWS(commit_boundary_layer_momentum_work(make_simulation_state_view(storage), initial));
    REQUIRE(runtime.momentum_work_step_count == std::numeric_limits<std::uint64_t>::max());
    REQUIRE(runtime.cumulative_requested_momentum_work_j == 1);
}

TEST_CASE("production child dynamics reaches the parent endpoint", "[core][nesting][production][dynamics]")
{
    ProcessStateRestore restore;
    global_simulation_storage = make_parent();
    initialize_parent_modules();
    const PrognosticFieldStorage previous_parent = global_simulation_storage.fields;
    prepare_child(previous_parent);

    enable_nested_beta_plane(global_simulation_storage);

    const float              parent_rho = global_simulation_storage.fields.rho[2][2][0];
    NestedChildSubstepReport first;
    REQUIRE_NOTHROW(advance_nested_child_dynamics_substep(global_simulation_storage, previous_parent, {4.0}, first));
    REQUIRE(first.start_time_s == 0.0);
    REQUIRE(first.end_time_s == 1.0);
    REQUIRE(first.completed_child_substeps == 1);
    REQUIRE(first.interface_source.pressure.diagnosed_cells > 0);
    REQUIRE(std::isfinite(first.interface_source.energy.committed_change.total));
    require_finite_primary(global_simulation_storage.nested_execution.child.fields);
    const auto&       first_scratch = global_simulation_storage.nested_execution.boundary_scratch;
    const std::size_t target_rho_capacity =
        first_scratch.target.rho.capacity();
    const std::size_t target_u_capacity =
        first_scratch.target.u.capacity();
    const std::size_t candidate_rho_capacity =
        first_scratch.candidate.rho.capacity();
    const std::size_t candidate_qh_capacity =
        first_scratch.candidate.qh.capacity();
    REQUIRE(target_rho_capacity > 0);
    REQUIRE(target_u_capacity > 0);
    REQUIRE(candidate_rho_capacity > 0);
    REQUIRE(candidate_qh_capacity > 0);

    NestedChildSubstepReport second;
    REQUIRE_NOTHROW(advance_nested_child_dynamics_substep(
        global_simulation_storage, previous_parent, {4.0}, second));
    REQUIRE(second.start_time_s == 1.0);
    REQUIRE(second.end_time_s == 2.0);
    REQUIRE(second.completed_child_substeps == 2);
    REQUIRE(second.interface_source.pressure.diagnosed_cells > 0);
    REQUIRE(std::isfinite(
        second.interface_source.energy.committed_change.total));
    REQUIRE(global_simulation_storage.nested_execution.child.time_s ==
            global_simulation_storage.time_s);
    REQUIRE(global_simulation_storage.nested_execution.child.coriolis_f ==
            tmv::make_cartesian_coriolis_profile(
                true, true,
                global_simulation_storage.configuration.coriolis_latitude_deg,
                global_simulation_storage.nested_execution.child.domain_frame,
                global_simulation_storage.nested_execution.child.time_s)
                .f_at_grid_y0);
    const auto& second_scratch =
        global_simulation_storage.nested_execution.boundary_scratch;
    REQUIRE(second_scratch.target.rho.capacity() == target_rho_capacity);
    REQUIRE(second_scratch.target.u.capacity() == target_u_capacity);
    REQUIRE(second_scratch.candidate.rho.capacity() ==
            candidate_rho_capacity);
    REQUIRE(second_scratch.candidate.qh.capacity() ==
            candidate_qh_capacity);
    const NestedExecutionStorage copied_nested =
        global_simulation_storage.nested_execution;
    REQUIRE(copied_nested.boundary_scratch.target.rho.capacity() == 0);
    REQUIRE(copied_nested.boundary_scratch.candidate.rho.capacity() == 0);
    REQUIRE(copied_nested.boundary_scratch
                .movement_primary_discard[0].capacity() == 0);
    REQUIRE(copied_nested.boundary_scratch
                .movement_microphysics_discard[0].capacity() == 0);
    REQUIRE(global_simulation_storage.fields.rho[2][2][0] == parent_rho);
    REQUIRE_NOTHROW(require_valid_nested_execution_relationship(
        global_simulation_storage));
    const auto& child =
        global_simulation_storage.nested_execution.child.fields;
    require_finite_primary(child);
    REQUIRE(maximum_absolute_value(child.u) <= 1.0e-6);
    REQUIRE(maximum_absolute_value(child.v) <= 1.0e-6);
    REQUIRE(maximum_absolute_value(child.w) <= 1.0e-6);
}

TEST_CASE("production child rejects an interval before advancing module cadence",
          "[core][nesting][production][stability]")
{
    ProcessStateRestore restore;
    global_simulation_storage = make_parent();
    initialize_parent_modules();
    const PrognosticFieldStorage previous_parent =
        global_simulation_storage.fields;
    prepare_child(previous_parent);
    enable_nested_beta_plane(global_simulation_storage);

    auto& nested = global_simulation_storage.nested_execution;
    nested.child.fields.u[3][3][0] = 5000.0f;
    const double child_time_before = nested.child.time_s;
    const int phase_before =
        nested.parent_time_bracket.completed_child_substeps;
    const double radiation_sample_before =
        nested.child.runtime.radiation.last_update_time_s;
    const double pbl_sample_before =
        nested.child.runtime.boundary_layer.last_update_time_s;

    NestedChildSubstepReport report;
    REQUIRE_THROWS_AS(
        advance_nested_child_dynamics_substep(
            global_simulation_storage, previous_parent, {4.0}, report),
        std::runtime_error);

    REQUIRE(nested.child.time_s == child_time_before);
    REQUIRE(nested.parent_time_bracket.completed_child_substeps ==
            phase_before);
    REQUIRE(nested.child.runtime.radiation.last_update_time_s ==
            radiation_sample_before);
    REQUIRE(nested.child.runtime.boundary_layer.last_update_time_s ==
            pbl_sample_before);
}

TEST_CASE("evolved production child continues exactly across synchronization restart",
          "[core][nesting][production][checkpoint]")
{
    ProcessStateRestore restore;
    global_simulation_storage = make_parent();
    initialize_parent_modules();
    const PrognosticFieldStorage initial_previous_parent =
        global_simulation_storage.fields;
    prepare_child(initial_previous_parent);

    NestedChildSubstepReport first;
    NestedChildSubstepReport second;
    advance_nested_child_dynamics_substep(
        global_simulation_storage, initial_previous_parent, {4.0}, first);
    advance_nested_child_dynamics_substep(
        global_simulation_storage, initial_previous_parent, {4.0}, second);
    REQUIRE(global_simulation_storage.nested_execution.child.time_s ==
            global_simulation_storage.time_s);

    std::string checkpoint_error;
    std::vector<std::uint8_t> synchronized_checkpoint;
    REQUIRE(tmv::checkpoint::serialize_simulation_storage(
        global_simulation_storage, synchronized_checkpoint,
        checkpoint_error));

    advance_next_balanced_parent_interval({4.0});
    std::vector<std::uint8_t> continuous_checkpoint;
    REQUIRE(tmv::checkpoint::serialize_simulation_storage(
        global_simulation_storage, continuous_checkpoint,
        checkpoint_error));

    SimulationStorageState restarted;
    REQUIRE(tmv::checkpoint::deserialize_simulation_storage(
        synchronized_checkpoint, restarted, checkpoint_error));
    REQUIRE(restarted.nested_execution.boundary_scratch.target.rho.capacity()
            == 0);
    REQUIRE(restarted.nested_execution.boundary_scratch.candidate.rho.capacity()
            == 0);
    REQUIRE(restarted.nested_execution.boundary_scratch
                .movement_primary_discard[0].capacity() == 0);
    REQUIRE(restarted.nested_execution.boundary_scratch
                .movement_microphysics_discard[0].capacity() == 0);
    global_simulation_storage = std::move(restarted);
    tmv::refresh_grid_metrics_from_terrain();
    tmv::initialize_dynamics(
        global_simulation_storage.configuration.dynamics_scheme_name);
    rehydrate_nested_child_transient_operators(
        global_simulation_storage);
    std::string compute_error;
    REQUIRE(initialize_compute_backend_runtime(compute_error));
    REQUIRE_NOTHROW(require_nested_child_level_runtime_ready(
        global_simulation_storage));

    advance_next_balanced_parent_interval({4.0});
    std::vector<std::uint8_t> restarted_checkpoint;
    REQUIRE(tmv::checkpoint::serialize_simulation_storage(
        global_simulation_storage, restarted_checkpoint,
        checkpoint_error));
    REQUIRE(restarted_checkpoint == continuous_checkpoint);
}

TEST_CASE("integer child movement retains overlap and survives restart",
          "[core][nesting][production][movement][checkpoint]")
{
    ProcessStateRestore restore;
    global_simulation_storage = make_parent();
    initialize_parent_modules();
    const PrognosticFieldStorage previous_parent =
        global_simulation_storage.fields;
    prepare_child(previous_parent);
    enable_nested_beta_plane(global_simulation_storage);

    NestedChildSubstepReport first;
    NestedChildSubstepReport second;
    advance_nested_child_dynamics_substep(
        global_simulation_storage, previous_parent, {4.0}, first);
    const float mid_bracket_rho =
        global_simulation_storage.nested_execution.child.fields.rho[2][2][0];
    NestedChildMovementReportV5 mid_bracket_rejection;
    mid_bracket_rejection.movement_sequence = 71;
    REQUIRE_THROWS_AS(move_nested_child_integer_parent_cells(
        global_simulation_storage, 1, 0, {4.0}, mid_bracket_rejection),
        std::invalid_argument);
    REQUIRE(mid_bracket_rejection.movement_sequence == 71);
    REQUIRE(global_simulation_storage.nested_execution.movement_sequence == 0);
    REQUIRE(global_simulation_storage.nested_execution.relationship.
                child_origin_parent_cell_i == 2);
    REQUIRE(global_simulation_storage.nested_execution.parent_time_bracket.
                completed_child_substeps == 1);
    REQUIRE(global_simulation_storage.nested_execution.child.fields.
                rho[2][2][0] == mid_bracket_rho);
    advance_nested_child_dynamics_substep(
        global_simulation_storage, previous_parent, {4.0}, second);
    REQUIRE(global_simulation_storage.nested_execution.child.coriolis_f ==
            tmv::make_cartesian_coriolis_profile(
                true, true,
                global_simulation_storage.configuration.coriolis_latitude_deg,
                global_simulation_storage.nested_execution.child.domain_frame,
                global_simulation_storage.nested_execution.child.time_s)
                .f_at_grid_y0);

    auto& storage = global_simulation_storage;
    auto initialize_surface_history = [](
        SurfaceDiagnosticFieldStorage& surface, int nr, int nth,
        float rate_base, float accumulated_base, double sample_time_s) {
        const std::size_t size = static_cast<std::size_t>(nr) *
            static_cast<std::size_t>(nth);
        surface.precipitation_rate_mm_h.resize(size);
        surface.accumulated_rainfall_mm.resize(size);
        surface.accumulated_rainfall_residual_mm.resize(size);
        for (int i = 0; i < nr; ++i)
            for (int j = 0; j < nth; ++j)
            {
                const std::size_t index = static_cast<std::size_t>(i) *
                        static_cast<std::size_t>(nth) +
                    static_cast<std::size_t>(j);
                surface.precipitation_rate_mm_h[index] =
                    rate_base + 10.0f * static_cast<float>(i) +
                    static_cast<float>(j);
                surface.accumulated_rainfall_mm[index] =
                    accumulated_base + 2.0f * static_cast<float>(i) +
                    0.25f * static_cast<float>(j);
                surface.accumulated_rainfall_residual_mm[index] =
                    1.0e-5f * static_cast<float>(1 + i + j);
            }
        surface.accumulated_rainfall_last_update_s = sample_time_s;
    };
    initialize_surface_history(
        storage.surface, storage.grid.nr, storage.grid.nth,
        10.0f, 100.0f, storage.time_s);
    initialize_surface_history(
        storage.nested_execution.child.surface,
        storage.nested_execution.child.grid.nr,
        storage.nested_execution.child.grid.nth,
        200.0f, 300.0f,
        storage.nested_execution.child.time_s);
    const SurfaceDiagnosticFieldStorage original_child_surface =
        storage.nested_execution.child.surface;
    std::string checkpoint_error;
    std::vector<std::uint8_t> before_failed_transaction;
    const bool serialized_before_failed_transaction =
        tmv::checkpoint::serialize_simulation_storage(
            storage, before_failed_transaction, checkpoint_error);
    INFO(checkpoint_error);
    REQUIRE(serialized_before_failed_transaction);
    const float valid_parent_residual =
        storage.surface.accumulated_rainfall_residual_mm[0];
    storage.surface.accumulated_rainfall_residual_mm[0] =
        std::numeric_limits<float>::quiet_NaN();
    NestedChildMovementReportV5 invalid_surface;
    invalid_surface.movement_sequence = 50;
    REQUIRE_THROWS_AS(move_nested_child_integer_parent_cells(
        storage, 1, 0, {4.0}, invalid_surface), std::invalid_argument);
    REQUIRE(invalid_surface.movement_sequence == 50);
    storage.surface.accumulated_rainfall_residual_mm[0] =
        valid_parent_residual;
    std::vector<std::uint8_t> after_invalid_surface;
    REQUIRE(tmv::checkpoint::serialize_simulation_storage(
        storage, after_invalid_surface, checkpoint_error));
    REQUIRE(after_invalid_surface == before_failed_transaction);
    NestedChildMovementReportV5 failed_transaction;
    failed_transaction.movement_sequence = 51;
    const double coriolis_before_failure = storage.nested_execution.child.coriolis_f;
    REQUIRE_THROWS_AS(move_nested_child_integer_parent_cells(
        storage, 0, 1, {0.0}, failed_transaction), std::invalid_argument);
    REQUIRE(failed_transaction.movement_sequence == 51);
    CHECK(storage.nested_execution.child.coriolis_f == coriolis_before_failure);
    std::vector<std::uint8_t> after_failed_transaction;
    REQUIRE(tmv::checkpoint::serialize_simulation_storage(
        storage, after_failed_transaction, checkpoint_error));
    REQUIRE(after_failed_transaction == before_failed_transaction);
    REQUIRE(storage.nested_execution.child.surface.precipitation_rate_mm_h ==
            original_child_surface.precipitation_rate_mm_h);
    REQUIRE(storage.nested_execution.child.surface.accumulated_rainfall_mm ==
            original_child_surface.accumulated_rainfall_mm);
    REQUIRE(storage.nested_execution.child.surface.
                accumulated_rainfall_residual_mm ==
            original_child_surface.accumulated_rainfall_residual_mm);
    REQUIRE(storage.nested_execution.child.surface.
                accumulated_rainfall_last_update_s ==
            original_child_surface.accumulated_rainfall_last_update_s);

    NestedChildMovementReportV5 zero_shift;
    zero_shift.movement_sequence = 61;
    REQUIRE_THROWS_AS(move_nested_child_integer_parent_cells(
        storage, 0, 0, {4.0}, zero_shift), std::invalid_argument);
    REQUIRE(zero_shift.movement_sequence == 61);

    auto& child_turbulence =
        storage.nested_execution.child.runtime.turbulence.config.scheme_id;
    child_turbulence = "smagorinsky";
    NestedChildMovementReportV5 active_cache;
    active_cache.movement_sequence = 62;
    REQUIRE_THROWS_AS(move_nested_child_integer_parent_cells(
        storage, 1, 0, {4.0}, active_cache), std::invalid_argument);
    REQUIRE(active_cache.movement_sequence == 62);
    child_turbulence = "none";

    auto& old_child = storage.nested_execution.child.fields;
    old_child.rho[4][2][0] = 1.07f;
    old_child.theta[4][2][0] = 304.0f;
    old_child.qv[4][2][0] = 0.003f;
    old_child.tracer[4][2][0] = 0.25f;
    old_child.tke[4][2][0] = 0.2f;
    old_child.p[4][2][0] = static_cast<float>(
        tmv::dynamics_eos::pressure_fp64(
            old_child.rho[4][2][0], old_child.theta[4][2][0]));
    old_child.u[4][2][0] = 7.0f;
    old_child.v[4][2][0] = -3.0f;
    old_child.w[4][2][0] = 1.5f;

    storage.fields.rho[5][3][0] = 1.03f;
    storage.fields.theta[5][3][0] = 303.0f;
    storage.fields.qv[5][3][0] = 0.002f;
    storage.fields.p[5][3][0] = static_cast<float>(
        tmv::dynamics_eos::pressure_fp64(
            storage.fields.rho[5][3][0], storage.fields.theta[5][3][0]));

    storage.runtime.storm_selected_partition_available = true;
    storage.runtime.storm_selected_partition_nx = 4;
    storage.runtime.storm_selected_partition_ny = 4;
    storage.runtime.storm_selected_partition.assign(16, 0U);
    storage.runtime.storm_selected_partition[3 * 4 + 1] = 1U;

    NestedChildMovementReportV5 movement;
    REQUIRE_NOTHROW(move_nested_child_integer_parent_cells(
        storage, 1, 0, {4.0}, movement));
    REQUIRE(storage.nested_execution.child.coriolis_f ==
            tmv::make_cartesian_coriolis_profile(
                true, true, storage.configuration.coriolis_latitude_deg,
                storage.nested_execution.child.domain_frame,
                storage.nested_execution.child.time_s).f_at_grid_y0);
    REQUIRE(movement.movement_sequence == 1);
    REQUIRE(movement.parent_shift_i == 1);
    REQUIRE(movement.parent_shift_j == 0);
    REQUIRE(movement.child_shift_i == 2);
    REQUIRE(movement.child_shift_j == 0);
    REQUIRE(movement.previous_parent_origin_i == 2);
    REQUIRE(movement.committed_parent_origin_i == 3);
    REQUIRE(movement.overlap_scalar_cells == 16);
    REQUIRE(movement.newly_admitted_scalar_cells == 16);
    REQUIRE(movement.overlap_x_faces == 24);
    REQUIRE(movement.newly_admitted_x_faces == 16);
    REQUIRE(movement.overlap_y_faces == 20);
    REQUIRE(movement.newly_admitted_y_faces == 20);
    REQUIRE(movement.overlap_z_faces == 16);
    REQUIRE(movement.newly_admitted_z_faces == 16);
    REQUIRE(movement.rollback_journal_values == 244);
    REQUIRE(movement.rollback_journal_bytes == 244 * sizeof(float));
    REQUIRE(movement.full_primary_owner_values == 496);
    REQUIRE(movement.rollback_journal_values <
            movement.full_primary_owner_values);
    REQUIRE(movement.precipitation.active);
    REQUIRE(movement.precipitation.sample_time_s == storage.time_s);
    REQUIRE(movement.precipitation.newly_admitted_columns == 8);
    REQUIRE(movement.precipitation.previous_child_accumulated_water_m3 ==
            Approx(
                movement.precipitation.discarded_child_accumulated_water_m3 +
                movement.precipitation.retained_child_accumulated_water_m3));
    REQUIRE(movement.precipitation.committed_child_accumulated_water_m3 ==
            Approx(
                movement.precipitation.retained_child_accumulated_water_m3 +
                movement.precipitation.
                    newly_admitted_parent_accumulated_water_m3));
    REQUIRE(
        movement.precipitation.previous_child_precipitation_volume_rate_m3s ==
        Approx(
            movement.precipitation.
                discarded_child_precipitation_volume_rate_m3s +
            movement.precipitation.
                retained_child_precipitation_volume_rate_m3s));
    REQUIRE(
        movement.precipitation.committed_child_precipitation_volume_rate_m3s ==
        Approx(
            movement.precipitation.
                retained_child_precipitation_volume_rate_m3s +
            movement.precipitation.
                newly_admitted_parent_precipitation_volume_rate_m3s));

    const auto& nested = storage.nested_execution;
    const auto& child = nested.child.fields;
    REQUIRE(nested.relationship.child_origin_parent_cell_i == 3);
    REQUIRE(nested.relationship.child_origin_parent_cell_j == 2);
    REQUIRE(nested.initial_parent_origin_i == 2);
    REQUIRE(nested.cumulative_parent_shift_i == 1);
    REQUIRE(nested.movement_sequence == 1);
    REQUIRE(child.rho[2][2][0] == 1.07f);
    REQUIRE(child.theta[2][2][0] == 304.0f);
    REQUIRE(child.qv[2][2][0] == 0.003f);
    REQUIRE(child.tracer[2][2][0] == 0.25f);
    REQUIRE(child.tke[2][2][0] == 0.2f);
    REQUIRE(child.u[2][2][0] == 7.0f);
    REQUIRE(child.v[2][2][0] == -3.0f);
    REQUIRE(child.w[2][2][0] == 1.5f);
    REQUIRE(child.p[2][2][0] == static_cast<float>(
        tmv::dynamics_eos::pressure_fp64(
            child.rho[2][2][0], child.theta[2][2][0])));
    const auto& moved_surface = nested.child.surface;
    const double child_area_m2 = nested.relationship.child_dx_m *
        nested.relationship.child_dy_m;
    double expected_new_water_m3 = 0.0;
    double expected_new_rate_m3s = 0.0;
    for (int ci = 0; ci < 4; ++ci)
        for (int cj = 0; cj < 4; ++cj)
        {
            const std::size_t destination =
                static_cast<std::size_t>(1 + ci) * 6U +
                static_cast<std::size_t>(1 + cj);
            if (ci < 2)
            {
                const std::size_t source =
                    static_cast<std::size_t>(1 + ci + 2) * 6U +
                    static_cast<std::size_t>(1 + cj);
                REQUIRE(moved_surface.precipitation_rate_mm_h[destination] ==
                        original_child_surface.
                            precipitation_rate_mm_h[source]);
                REQUIRE(moved_surface.accumulated_rainfall_mm[destination] ==
                        original_child_surface.
                            accumulated_rainfall_mm[source]);
                REQUIRE(moved_surface.
                            accumulated_rainfall_residual_mm[destination] ==
                        original_child_surface.
                            accumulated_rainfall_residual_mm[source]);
                continue;
            }
            const std::size_t parent_source = 5U * 8U +
                static_cast<std::size_t>(3 + cj / 2);
            REQUIRE(moved_surface.precipitation_rate_mm_h[destination] ==
                    storage.surface.precipitation_rate_mm_h[parent_source]);
            REQUIRE(moved_surface.accumulated_rainfall_mm[destination] ==
                    storage.surface.accumulated_rainfall_mm[parent_source]);
            REQUIRE(moved_surface.
                        accumulated_rainfall_residual_mm[destination] ==
                    storage.surface.
                        accumulated_rainfall_residual_mm[parent_source]);
            expected_new_water_m3 +=
                (static_cast<double>(
                     storage.surface.accumulated_rainfall_mm[parent_source]) +
                 static_cast<double>(storage.surface.
                     accumulated_rainfall_residual_mm[parent_source])) *
                child_area_m2 * 1.0e-3;
            expected_new_rate_m3s += static_cast<double>(
                storage.surface.precipitation_rate_mm_h[parent_source]) *
                child_area_m2 * 1.0e-3 / 3600.0;
        }
    REQUIRE(movement.precipitation.
                newly_admitted_parent_accumulated_water_m3 ==
            Approx(expected_new_water_m3));
    REQUIRE(movement.precipitation.
                newly_admitted_parent_precipitation_volume_rate_m3s ==
            Approx(expected_new_rate_m3s));
    REQUIRE(moved_surface.accumulated_rainfall_last_update_s ==
            nested.child.time_s);
    REQUIRE(storage.runtime.storm_selected_partition[1 * 4 + 1] == 1U);
    REQUIRE(std::count(
                storage.runtime.storm_selected_partition.begin(),
                storage.runtime.storm_selected_partition.end(), 1U) == 1);
    storage.runtime.storm_selected_partition_available = false;
    storage.runtime.storm_selected_partition_nx = 0;
    storage.runtime.storm_selected_partition_ny = 0;
    storage.runtime.storm_selected_partition.clear();
    REQUIRE(child.rho[3][2][0] == storage.fields.rho[5][3][0]);
    REQUIRE(child.theta[3][2][0] == storage.fields.theta[5][3][0]);
    REQUIRE(child.qv[3][2][0] == storage.fields.qv[5][3][0]);
    REQUIRE(nested.child.domain_frame.earth_origin_at_t0_m.x_m ==
            3500.0);
    REQUIRE(nested.child.domain_frame.earth_origin_at_t0_m.y_m ==
            2000.0);
    REQUIRE(movement.boundary_source.pressure.diagnosed_cells == 32);
    REQUIRE_NOTHROW(require_valid_nested_execution_relationship(storage));
    REQUIRE_NOTHROW(require_nested_child_level_runtime_ready(storage));

    std::vector<std::uint8_t> moved_checkpoint;
    REQUIRE(tmv::checkpoint::serialize_simulation_storage(
        storage, moved_checkpoint, checkpoint_error));
    SimulationStorageState restarted;
    REQUIRE(tmv::checkpoint::deserialize_simulation_storage(
        moved_checkpoint, restarted, checkpoint_error));
    REQUIRE(restarted.nested_execution.child.surface.
                precipitation_rate_mm_h ==
            moved_surface.precipitation_rate_mm_h);
    REQUIRE(restarted.nested_execution.child.surface.
                accumulated_rainfall_mm ==
            moved_surface.accumulated_rainfall_mm);
    REQUIRE(restarted.nested_execution.child.surface.
                accumulated_rainfall_residual_mm ==
            moved_surface.accumulated_rainfall_residual_mm);
    REQUIRE(restarted.nested_execution.child.surface.
                accumulated_rainfall_last_update_s ==
            moved_surface.accumulated_rainfall_last_update_s);
    std::vector<std::uint8_t> restarted_checkpoint;
    REQUIRE(tmv::checkpoint::serialize_simulation_storage(
        restarted, restarted_checkpoint, checkpoint_error));
    REQUIRE(restarted_checkpoint == moved_checkpoint);

    NestedChildMovementReportV5 rejected;
    rejected.movement_sequence = 99;
    REQUIRE_THROWS_AS(move_nested_child_integer_parent_cells(
        storage, 1, 0, {4.0}, rejected), std::invalid_argument);
    REQUIRE(rejected.movement_sequence == 99);
    std::vector<std::uint8_t> after_rejection;
    REQUIRE(tmv::checkpoint::serialize_simulation_storage(
        storage, after_rejection, checkpoint_error));
    REQUIRE(after_rejection == moved_checkpoint);

    auto& once_moved = storage.nested_execution.child.fields;
    once_moved.rho[1][4][0] = 1.02f;
    once_moved.theta[1][4][0] = 301.5f;
    once_moved.qc[1][4][0] = 0.001f;
    once_moved.tracer[1][4][0] = 0.125f;
    once_moved.tke[1][4][0] = 0.15f;
    once_moved.p[1][4][0] = static_cast<float>(
        tmv::dynamics_eos::pressure_fp64(
            once_moved.rho[1][4][0], once_moved.theta[1][4][0]));
    once_moved.u[0][4][0] = 4.0f;
    once_moved.v[1][4][0] = -2.0f;
    once_moved.w[1][4][0] = 0.75f;

    NestedChildMovementReportV5 diagonal;
    REQUIRE_NOTHROW(move_nested_child_integer_parent_cells(
        storage, -1, 1, {4.0}, diagonal));
    REQUIRE(diagonal.movement_sequence == 2);
    REQUIRE(diagonal.child_shift_i == -2);
    REQUIRE(diagonal.child_shift_j == 2);
    REQUIRE(diagonal.overlap_scalar_cells == 8);
    REQUIRE(diagonal.newly_admitted_scalar_cells == 24);
    REQUIRE(diagonal.rollback_journal_values == 368);
    REQUIRE(diagonal.rollback_journal_bytes == 368 * sizeof(float));
    REQUIRE(diagonal.full_primary_owner_values == 496);
    REQUIRE(diagonal.rollback_journal_values <
            diagonal.full_primary_owner_values);
    REQUIRE(storage.nested_execution.relationship.child_origin_parent_cell_i ==
            2);
    REQUIRE(storage.nested_execution.relationship.child_origin_parent_cell_j ==
            3);
    REQUIRE(storage.nested_execution.cumulative_parent_shift_i == 0);
    REQUIRE(storage.nested_execution.cumulative_parent_shift_j == 1);
    const auto& twice_moved = storage.nested_execution.child.fields;
    REQUIRE(twice_moved.rho[3][2][0] == 1.02f);
    REQUIRE(twice_moved.theta[3][2][0] == 301.5f);
    REQUIRE(twice_moved.qc[3][2][0] == 0.001f);
    REQUIRE(twice_moved.tracer[3][2][0] == 0.125f);
    REQUIRE(twice_moved.tke[3][2][0] == 0.15f);
    REQUIRE(twice_moved.u[2][2][0] == 4.0f);
    REQUIRE(twice_moved.v[3][2][0] == -2.0f);
    REQUIRE(twice_moved.w[3][2][0] == 0.75f);
    REQUIRE_NOTHROW(require_valid_nested_execution_relationship(storage));

    std::vector<std::uint8_t> twice_moved_checkpoint;
    REQUIRE(tmv::checkpoint::serialize_simulation_storage(
        storage, twice_moved_checkpoint, checkpoint_error));
    SimulationStorageState twice_restarted;
    REQUIRE(tmv::checkpoint::deserialize_simulation_storage(
        twice_moved_checkpoint, twice_restarted, checkpoint_error));
    std::vector<std::uint8_t> twice_restarted_checkpoint;
    REQUIRE(tmv::checkpoint::serialize_simulation_storage(
        twice_restarted, twice_restarted_checkpoint, checkpoint_error));
    REQUIRE(twice_restarted_checkpoint == twice_moved_checkpoint);

    REQUIRE_NOTHROW(advance_next_balanced_parent_interval({4.0}));
    REQUIRE(storage.nested_execution.child.time_s == storage.time_s);
    REQUIRE(storage.nested_execution.parent_time_bracket.completed_child_substeps ==
            storage.nested_execution.relationship.parent_time_step_ratio);
}

TEST_CASE("tracker authority commits only successful synchronized child movement",
          "[core][nesting][production][movement][tracker][checkpoint]")
{
    ProcessStateRestore restore;
    global_simulation_storage = make_parent(10, 10);
    initialize_parent_modules();
    const PrognosticFieldStorage previous_parent =
        global_simulation_storage.fields;
    prepare_child(previous_parent, 8, 8, 8, 8);

    NestedChildSubstepReport first;
    NestedChildSubstepReport second;
    advance_nested_child_dynamics_substep(
        global_simulation_storage, previous_parent, {4.0}, first);
    advance_nested_child_dynamics_substep(
        global_simulation_storage, previous_parent, {4.0}, second);
    REQUIRE(global_simulation_storage.nested_execution.child.time_s ==
            global_simulation_storage.time_s);

    tmv::domain::StormWindowMovementPolicy policy;
    policy.inner_margin_m = 1200.0;
    policy.release_margin_m = 1300.0;
    policy.dx_m = 1000.0;
    policy.dy_m = 800.0;
    policy.maximum_shift_cells = 1;
    policy.cooldown_s = 0.0;
    policy.minimum_confidence = 0.9;
    policy.confirmation_samples = 2;
    policy.relaxation_margin_x_m = 500.0;
    policy.relaxation_margin_y_m = 400.0;
    REQUIRE(policy.relaxation_margin_x_m ==
            global_simulation_storage.nested_execution.relationship.
                relaxation_width_child_cells *
                global_simulation_storage.nested_execution.relationship.
                    child_dx_m);
    REQUIRE(policy.relaxation_margin_y_m ==
            global_simulation_storage.nested_execution.relationship.
                relaxation_width_child_cells *
                global_simulation_storage.nested_execution.relationship.
                    child_dy_m);

    const auto tracked_envelope = [](
        double sample_time_s, double west_m, double east_m) {
        NestedTrackedEnvelopeV1 envelope;
        envelope.sample_time_s = sample_time_s;
        envelope.selected_feature_available = true;
        envelope.continuity_authoritative = true;
        envelope.motion_available = true;
        envelope.confidence = 0.99;
        envelope.envelope_west_earth_m = west_m;
        envelope.envelope_east_earth_m = east_m;
        envelope.envelope_south_earth_m = 3800.0;
        envelope.envelope_north_earth_m = 4200.0;
        envelope.velocity_x_earth_ms = 12.0;
        envelope.envelope_face_motion_available = true;
        envelope.envelope_face_motion_interval_s = 30.0;
        envelope.envelope_east_earth_velocity_ms = 30.0;
        return envelope;
    };

    const auto initial_evidence = prepare_nested_child_movement_evidence(
        global_simulation_storage,
        tracked_envelope(global_simulation_storage.time_s, 5700.0, 5900.0));
    REQUIRE(initial_evidence.distance_west_m == 2700.0);
    REQUIRE(initial_evidence.distance_east_m == 1100.0);
    REQUIRE(initial_evidence.distance_south_m == 1400.0);
    REQUIRE(initial_evidence.distance_north_m == 1400.0);
    REQUIRE(initial_evidence.velocity_x_ms == 12.0);
    REQUIRE(initial_evidence.velocity_y_ms == 0.0);
    REQUIRE(initial_evidence.legal_shift_bounds_available);
    REQUIRE(initial_evidence.minimum_i_shift_cells == -1);
    REQUIRE(initial_evidence.maximum_i_shift_cells == 1);
    REQUIRE(initial_evidence.minimum_j_shift_cells == -1);
    REQUIRE(initial_evidence.maximum_j_shift_cells == 1);

    NestedChildMovementAuthorityReportV1 committed;
    REQUIRE_NOTHROW(evaluate_and_move_nested_child_from_tracker(
        global_simulation_storage,
        tracked_envelope(global_simulation_storage.time_s, 5700.0, 5900.0),
        policy, {4.0}, committed));
    REQUIRE(committed.decision.status ==
            tmv::domain::StormWindowMovementStatus::Requested);
    REQUIRE(committed.decision.reason ==
            tmv::domain::StormWindowMovementReason::ProjectedRelaxationRisk);
    REQUIRE(committed.decision.confirmation_count == 1);
    REQUIRE(committed.movement_committed);
    REQUIRE(committed.movement.parent_shift_i == 1);
    REQUIRE(committed.movement.parent_shift_j == 0);
    REQUIRE(committed.movement.movement_sequence == 1);
    REQUIRE(global_simulation_storage.nested_execution.relationship.
                child_origin_parent_cell_i == 3);
    REQUIRE(global_simulation_storage.nested_execution.movement_sequence == 1);
    const auto& committed_authority =
        global_simulation_storage.runtime.storm_window_movement;
    REQUIRE(committed_authority.armed);
    REQUIRE(committed_authority.committed_move_available);
    REQUIRE(committed_authority.last_committed_time_s == 2.0);
    REQUIRE(committed_authority.cumulative_i_shift_cells == 1);
    REQUIRE(committed_authority.cumulative_j_shift_cells == 0);

    std::string checkpoint_error;
    std::vector<std::uint8_t> committed_checkpoint;
    REQUIRE(tmv::checkpoint::serialize_simulation_storage(
        global_simulation_storage, committed_checkpoint, checkpoint_error));
    SimulationStorageState restarted;
    REQUIRE(tmv::checkpoint::deserialize_simulation_storage(
        committed_checkpoint, restarted, checkpoint_error));
    REQUIRE(restarted.nested_execution.relationship.
                child_origin_parent_cell_i == 3);
    REQUIRE(restarted.nested_execution.movement_sequence == 1);
    REQUIRE(restarted.runtime.storm_window_movement.
                cumulative_i_shift_cells == 1);
    REQUIRE(restarted.runtime.storm_window_movement.
                last_committed_time_s == 2.0);
    std::vector<std::uint8_t> restarted_checkpoint;
    REQUIRE(tmv::checkpoint::serialize_simulation_storage(
        restarted, restarted_checkpoint, checkpoint_error));
    REQUIRE(restarted_checkpoint == committed_checkpoint);

    REQUIRE_NOTHROW(advance_next_balanced_parent_interval({4.0}));
    NestedChildMovementAuthorityReportV1 rearmed;
    REQUIRE_NOTHROW(evaluate_and_move_nested_child_from_tracker(
        global_simulation_storage,
        tracked_envelope(global_simulation_storage.time_s, 5700.0, 5900.0),
        policy, {4.0}, rearmed));
    REQUIRE(rearmed.decision.status ==
            tmv::domain::StormWindowMovementStatus::Ineligible);
    REQUIRE(rearmed.decision.reason ==
            tmv::domain::StormWindowMovementReason::NoApproachingBreach);
    REQUIRE_FALSE(rearmed.movement_committed);
    REQUIRE(global_simulation_storage.runtime.storm_window_movement.armed);

    REQUIRE_NOTHROW(advance_next_balanced_parent_interval({4.0}));
    std::vector<std::uint8_t> before_stencil_limited_sample;
    REQUIRE(tmv::checkpoint::serialize_simulation_storage(
        global_simulation_storage, before_stencil_limited_sample,
        checkpoint_error));
    NestedChildMovementAuthorityReportV1 stencil_limited;
    const auto parent_infeasible_envelope = tracked_envelope(
        global_simulation_storage.time_s, 6700.0, 6900.0);
    auto diagnostic_authority =
        global_simulation_storage.runtime.storm_window_movement;
    const auto parent_infeasible_decision =
        tmv::domain::evaluate_storm_window_movement(
            diagnostic_authority,
            prepare_nested_child_movement_evidence(
                global_simulation_storage, parent_infeasible_envelope),
            policy);
    REQUIRE(parent_infeasible_decision.status ==
            tmv::domain::StormWindowMovementStatus::Ineligible);
    REQUIRE(parent_infeasible_decision.reason ==
            tmv::domain::StormWindowMovementReason::ParentStencilLimitedContained);
    REQUIRE_NOTHROW(evaluate_and_move_nested_child_from_tracker(
        global_simulation_storage,
        parent_infeasible_envelope,
        policy, {4.0}, stencil_limited));
    REQUIRE(stencil_limited.decision.status ==
            tmv::domain::StormWindowMovementStatus::Ineligible);
    REQUIRE(stencil_limited.decision.reason ==
            tmv::domain::StormWindowMovementReason::ParentStencilLimitedContained);
    REQUIRE_FALSE(stencil_limited.movement_committed);
    REQUIRE(stencil_limited.movement.movement_sequence == 0);
    REQUIRE(global_simulation_storage.nested_execution.relationship.
                child_origin_parent_cell_i == 3);
    REQUIRE(global_simulation_storage.nested_execution.movement_sequence == 1);
    REQUIRE(global_simulation_storage.runtime.storm_window_movement.
                last_sample_time_s == 6.0);
    REQUIRE(global_simulation_storage.runtime.storm_window_movement.
                consecutive_compatible_samples == 0);
    REQUIRE(global_simulation_storage.runtime.storm_window_movement.
                cumulative_i_shift_cells == 1);
    std::vector<std::uint8_t> after_stencil_limited_sample;
    REQUIRE(tmv::checkpoint::serialize_simulation_storage(
        global_simulation_storage, after_stencil_limited_sample,
        checkpoint_error));
    REQUIRE(after_stencil_limited_sample != before_stencil_limited_sample);
}

TEST_CASE("tracker authority sustains repeated eastward child movement across restart",
          "[core][nesting][production][movement][tracker][lifetime][checkpoint]")
{
    ProcessStateRestore restore;
    global_simulation_storage = make_parent(14, 8);
    initialize_parent_modules();
    const PrognosticFieldStorage previous_parent =
        global_simulation_storage.fields;
    prepare_child(previous_parent, 12, 6);

    NestedChildSubstepReport first;
    NestedChildSubstepReport second;
    advance_nested_child_dynamics_substep(
        global_simulation_storage, previous_parent, {4.0}, first);
    advance_nested_child_dynamics_substep(
        global_simulation_storage, previous_parent, {4.0}, second);
    REQUIRE(global_simulation_storage.nested_execution.child.time_s ==
            global_simulation_storage.time_s);

    tmv::domain::StormWindowMovementPolicy policy;
    policy.inner_margin_m = 500.0;
    policy.release_margin_m = 600.0;
    policy.dx_m = 1000.0;
    policy.dy_m = 800.0;
    policy.maximum_shift_cells = 1;
    policy.cooldown_s = 0.0;
    policy.minimum_confidence = 0.9;
    policy.confirmation_samples = 2;

    const auto envelope = [](double sample_time_s, double west_m,
                             double east_m) {
        NestedTrackedEnvelopeV1 result;
        result.sample_time_s = sample_time_s;
        result.selected_feature_available = true;
        result.continuity_authoritative = true;
        result.motion_available = true;
        result.confidence = 0.99;
        result.envelope_west_earth_m = west_m;
        result.envelope_east_earth_m = east_m;
        result.envelope_south_earth_m = 3000.0;
        result.envelope_north_earth_m = 3400.0;
        result.velocity_x_earth_ms = 12.0;
        return result;
    };
    const auto current_child_east_m = []() {
        const auto& relationship =
            global_simulation_storage.nested_execution.relationship;
        return static_cast<double>(
                   1 + relationship.child_origin_parent_cell_i +
                   relationship.child_parent_span_x) *
            relationship.parent_dx_m;
    };
    const auto current_child_center_m = [&]() {
        return current_child_east_m() - 0.5 *
            static_cast<double>(global_simulation_storage.nested_execution.
                                    relationship.child_parent_span_x) *
                global_simulation_storage.nested_execution.relationship.
                    parent_dx_m;
    };

    const auto primary_fields = [](PrognosticFieldStorage& fields) {
        return std::array<Field3D*, 15>{
            &fields.rho, &fields.p, &fields.u, &fields.v, &fields.w,
            &fields.theta, &fields.qv, &fields.qc, &fields.qr,
            &fields.qi, &fields.qs, &fields.qg, &fields.qh,
            &fields.tracer, &fields.tke};
    };
    const auto reusable_target_fields = primary_fields(
        global_simulation_storage.nested_execution.boundary_scratch.candidate);
    const auto reusable_live_fields = primary_fields(
        global_simulation_storage.nested_execution.child.fields);
    std::array<const float*, 15> reusable_target_addresses{};
    std::array<const float*, 15> reusable_live_addresses{};
    std::array<std::size_t, 15> reusable_target_capacities{};
    std::array<const float*, 15> reusable_discard_addresses{};
    std::array<std::size_t, 15> reusable_discard_capacities{};
    for (std::size_t index = 0; index < reusable_target_fields.size(); ++index)
    {
        REQUIRE(reusable_target_fields[index]->size() > 0);
        reusable_target_addresses[index] = reusable_target_fields[index]->data();
        reusable_live_addresses[index] = reusable_live_fields[index]->data();
        reusable_target_capacities[index] = reusable_target_fields[index]->capacity();
        REQUIRE(reusable_live_fields[index]->capacity() ==
                reusable_target_capacities[index]);
    }

    std::string checkpoint_error;
    for (std::uint64_t movement = 1; movement <= 4; ++movement)
    {
        if (reusable_target_addresses[0] == nullptr)
        {
            const auto restored_target_fields = primary_fields(
                global_simulation_storage.nested_execution.boundary_scratch.candidate);
            const auto restored_live_fields = primary_fields(
                global_simulation_storage.nested_execution.child.fields);
            for (std::size_t index = 0;
                 index < restored_target_fields.size(); ++index)
            {
                REQUIRE(restored_target_fields[index]->size() > 0);
                reusable_target_addresses[index] =
                    restored_target_fields[index]->data();
                reusable_live_addresses[index] =
                    restored_live_fields[index]->data();
                reusable_target_capacities[index] =
                    restored_target_fields[index]->capacity();
                REQUIRE(restored_live_fields[index]->capacity() ==
                        reusable_target_capacities[index]);
            }
        }
        const double east_m = current_child_east_m();
        NestedChildMovementAuthorityReportV1 committed;
        REQUIRE_NOTHROW(evaluate_and_move_nested_child_from_tracker(
            global_simulation_storage,
            envelope(global_simulation_storage.time_s,
                     east_m - 350.0, east_m - 150.0),
            policy, {4.0}, committed));
        REQUIRE(committed.decision.reason ==
                tmv::domain::StormWindowMovementReason::EmergencyContainment);
        REQUIRE(committed.decision.confirmation_count == 1);
        REQUIRE(committed.movement_committed);
        REQUIRE(committed.movement.parent_shift_i == 1);
        REQUIRE(committed.movement.parent_shift_j == 0);
        REQUIRE(committed.movement.movement_sequence == movement);
        REQUIRE(global_simulation_storage.nested_execution.relationship.
                    child_origin_parent_cell_i == 2 +
                static_cast<int>(movement));
        REQUIRE(global_simulation_storage.runtime.storm_window_movement.
                    cumulative_i_shift_cells == static_cast<int>(movement));
        REQUIRE_NOTHROW(require_valid_nested_execution_relationship(
            global_simulation_storage));
        require_finite_primary(
            global_simulation_storage.nested_execution.child.fields);
        require_complete_interface_ledger(
            global_simulation_storage.nested_execution.interface_source,
            5 * movement - 1);
        const auto retained_target_fields = primary_fields(
            global_simulation_storage.nested_execution.boundary_scratch.candidate);
        const auto retained_live_fields = primary_fields(
            global_simulation_storage.nested_execution.child.fields);
        for (std::size_t index = 0;
             index < retained_target_fields.size(); ++index)
        {
            const bool direct =
                retained_target_fields[index]->data() ==
                    reusable_target_addresses[index] &&
                retained_live_fields[index]->data() ==
                    reusable_live_addresses[index];
            const bool swapped =
                retained_target_fields[index]->data() ==
                    reusable_live_addresses[index] &&
                retained_live_fields[index]->data() ==
                    reusable_target_addresses[index];
            REQUIRE((direct || swapped));
            REQUIRE(retained_target_fields[index]->capacity() ==
                    reusable_target_capacities[index]);
            REQUIRE(retained_live_fields[index]->capacity() ==
                    reusable_target_capacities[index]);
        }
        auto& discard_storage = global_simulation_storage.nested_execution
            .boundary_scratch.movement_primary_discard;
        if (reusable_discard_addresses[0] == nullptr)
        {
            for (std::size_t index = 0; index < discard_storage.size();
                 ++index)
            {
                REQUIRE_FALSE(discard_storage[index].empty());
                reusable_discard_addresses[index] =
                    discard_storage[index].data();
                reusable_discard_capacities[index] =
                    discard_storage[index].capacity();
            }
        }
        else
            for (std::size_t index = 0; index < discard_storage.size();
                 ++index)
            {
                REQUIRE(discard_storage[index].data() ==
                        reusable_discard_addresses[index]);
                REQUIRE(discard_storage[index].capacity() ==
                        reusable_discard_capacities[index]);
            }

        if (movement == 2)
        {
            std::vector<std::uint8_t> before_restart;
            REQUIRE(tmv::checkpoint::serialize_simulation_storage(
                global_simulation_storage, before_restart, checkpoint_error));
            SimulationStorageState restarted;
            REQUIRE(tmv::checkpoint::deserialize_simulation_storage(
                before_restart, restarted, checkpoint_error));
            global_simulation_storage = std::move(restarted);
            std::vector<std::uint8_t> after_restart;
            REQUIRE(tmv::checkpoint::serialize_simulation_storage(
                global_simulation_storage, after_restart, checkpoint_error));
            REQUIRE(after_restart == before_restart);
            tmv::refresh_grid_metrics_from_terrain();
            tmv::initialize_dynamics(
                global_simulation_storage.configuration.dynamics_scheme_name);
            rehydrate_nested_child_transient_operators(
                global_simulation_storage);
            std::string compute_error;
            REQUIRE(initialize_compute_backend_runtime(compute_error));
            REQUIRE_NOTHROW(require_nested_child_level_runtime_ready(
                global_simulation_storage));
            reusable_target_addresses = {};
            reusable_live_addresses = {};
            reusable_target_capacities = {};
            reusable_discard_addresses = {};
            reusable_discard_capacities = {};
        }

        if (movement < 4)
        {
            REQUIRE_NOTHROW(advance_next_balanced_parent_interval({4.0}));
            const double center_m = current_child_center_m();
            NestedChildMovementAuthorityReportV1 rearmed;
            REQUIRE_NOTHROW(evaluate_and_move_nested_child_from_tracker(
                global_simulation_storage,
                envelope(global_simulation_storage.time_s,
                         center_m - 100.0, center_m + 100.0),
                policy, {4.0}, rearmed));
            REQUIRE(rearmed.decision.reason ==
                    tmv::domain::StormWindowMovementReason::NoApproachingBreach);
            REQUIRE(global_simulation_storage.runtime.storm_window_movement.
                        armed);
            REQUIRE_NOTHROW(advance_next_balanced_parent_interval({4.0}));
        }
    }

    REQUIRE(global_simulation_storage.nested_execution.movement_sequence == 4);
    REQUIRE(global_simulation_storage.nested_execution.relationship.
                child_origin_parent_cell_i == 6);
    REQUIRE(global_simulation_storage.nested_execution.child.time_s ==
            global_simulation_storage.time_s);
}

TEST_CASE("production detector drives repeated synchronized child movement",
          "[core][nesting][production][movement][tracker][detector][checkpoint]")
{
    ProcessStateRestore restore;
    global_simulation_storage = make_parent(22, 12);
    initialize_parent_modules();
    const PrognosticFieldStorage previous_parent =
        global_simulation_storage.fields;
    prepare_child(previous_parent, 20, 10, 12, 12);

    NestedChildSubstepReport first;
    NestedChildSubstepReport second;
    advance_nested_child_dynamics_substep(
        global_simulation_storage, previous_parent, {4.0}, first);
    advance_nested_child_dynamics_substep(
        global_simulation_storage, previous_parent, {4.0}, second);

    constexpr std::size_t signal_nx = 20;
    constexpr std::size_t signal_ny = 10;
    std::vector<double> maximum_updraft(signal_nx * signal_ny, 0.0);
    std::vector<double> condensate_path(signal_nx * signal_ny, 0.0);
    std::vector<double> updraft_helicity(signal_nx * signal_ny, 0.0);
    tmv::diagnostics::StormFeatureDetector detector;
    detector.reserve(signal_nx, signal_ny);
    tmv::diagnostics::StormFeatureDetection last_detection;
    tmv::diagnostics::StormContinuityState continuity;
    const tmv::diagnostics::StormFeatureThresholds feature_thresholds{
        5.0, 0.1, 25.0, 2, 1.0e6, 0.9};
    const tmv::diagnostics::StormContinuityThresholds continuity_thresholds{
        3, 3, 30.0, 10.0, 100.0, 2};

    const auto detect = [&](int start_i, double sample_time_s) {
        std::fill(maximum_updraft.begin(), maximum_updraft.end(), 0.0);
        std::fill(condensate_path.begin(), condensate_path.end(), 0.0);
        std::fill(updraft_helicity.begin(), updraft_helicity.end(), 0.0);
        for (int di = 0; di < 2; ++di)
            {
                const auto index = static_cast<std::size_t>(start_i + di) *
                                       signal_ny + 5U;
                maximum_updraft[index] = 12.0;
                condensate_path[index] = 1.0;
                updraft_helicity[index] = 50.0;
            }
        const tmv::diagnostics::StormColumnSignalView signals{
            signal_nx, signal_ny, 1000.0, 800.0, 1000.0, 800.0,
            &maximum_updraft, &condensate_path, &updraft_helicity};
        last_detection = detector.detect(signals, feature_thresholds);
        REQUIRE(last_detection.status ==
                tmv::diagnostics::StormDetectionStatus::Detected);
        tmv::diagnostics::update_storm_continuity(
            continuity, last_detection, sample_time_s, continuity_thresholds);
        global_simulation_storage.runtime.storm_tracking = continuity;
        if (continuity.selected_candidate_available)
        {
            auto& runtime = global_simulation_storage.runtime;
            detector.copy_partition(
                continuity.selected_component_label,
                runtime.storm_selected_partition);
            runtime.storm_selected_partition_available = true;
            runtime.storm_selected_partition_nx = signal_nx;
            runtime.storm_selected_partition_ny = signal_ny;
        }
        return prepare_nested_tracked_envelope(
            last_detection, continuity, sample_time_s,
            0.0, 0.0, 0.0, 0.0);
    };

    detect(4, -334.0);
    detect(5, -166.0);
    auto envelope = detect(6, global_simulation_storage.time_s);
    REQUIRE(envelope.selected_feature_available);
    REQUIRE(envelope.continuity_authoritative);
    REQUIRE(envelope.motion_available);
    REQUIRE(envelope.velocity_x_earth_ms > 0.0);
    const auto earth_envelope = prepare_nested_tracked_envelope_earth(
        last_detection, continuity, global_simulation_storage.time_s);
    REQUIRE(earth_envelope.envelope_west_earth_m ==
            envelope.envelope_west_earth_m);
    REQUIRE(earth_envelope.envelope_east_earth_m ==
            envelope.envelope_east_earth_m);
    REQUIRE(earth_envelope.velocity_x_earth_ms ==
            envelope.velocity_x_earth_ms);
    const tmv::diagnostics::StormEnvelopeFaceMotion face_motion{
        true, 168.0, -4.0, 6.0, -3.0, 2.0};
    const auto earth_envelope_with_face_motion =
        prepare_nested_tracked_envelope_earth(
            last_detection, continuity, face_motion,
            global_simulation_storage.time_s);
    REQUIRE(earth_envelope_with_face_motion.
                envelope_face_motion_available);
    REQUIRE(earth_envelope_with_face_motion.
                envelope_face_motion_interval_s == 168.0);
    REQUIRE(earth_envelope_with_face_motion.
                envelope_west_earth_velocity_ms == -4.0);
    REQUIRE(earth_envelope_with_face_motion.
                envelope_east_earth_velocity_ms == 6.0);
    REQUIRE(earth_envelope_with_face_motion.
                envelope_south_earth_velocity_ms == -3.0);
    REQUIRE(earth_envelope_with_face_motion.
                envelope_north_earth_velocity_ms == 2.0);
    const auto translated_grid_envelope = prepare_nested_tracked_envelope(
        last_detection, continuity, face_motion,
        global_simulation_storage.time_s,
        125.0, -75.0, 2.0, -1.0);
    REQUIRE(translated_grid_envelope.envelope_west_earth_m ==
            earth_envelope.envelope_west_earth_m + 125.0);
    REQUIRE(translated_grid_envelope.envelope_south_earth_m ==
            earth_envelope.envelope_south_earth_m - 75.0);
    REQUIRE(translated_grid_envelope.velocity_x_earth_ms ==
            earth_envelope.velocity_x_earth_ms + 2.0);
    REQUIRE(translated_grid_envelope.velocity_y_earth_ms ==
            earth_envelope.velocity_y_earth_ms - 1.0);
    REQUIRE(translated_grid_envelope.envelope_west_earth_velocity_ms ==
            -2.0);
    REQUIRE(translated_grid_envelope.envelope_east_earth_velocity_ms ==
            8.0);
    REQUIRE(translated_grid_envelope.envelope_south_earth_velocity_ms ==
            -4.0);
    REQUIRE(translated_grid_envelope.envelope_north_earth_velocity_ms ==
            1.0);

    const auto original_parent_frame =
        global_simulation_storage.configuration.domain_frame;
    const auto original_child_frame = global_simulation_storage.
        nested_execution.child.domain_frame;
    global_simulation_storage.configuration.domain_frame.velocity_ms =
        {2.0, -1.0};
    global_simulation_storage.nested_execution.child.domain_frame.velocity_ms =
        {2.0, -1.0};
    const auto child_relative_face_motion =
        prepare_nested_child_movement_evidence(
            global_simulation_storage, translated_grid_envelope);
    REQUIRE(child_relative_face_motion.envelope_face_motion_available);
    REQUIRE(child_relative_face_motion.envelope_face_motion_interval_s ==
            168.0);
    REQUIRE(child_relative_face_motion.
                envelope_west_coordinate_velocity_ms == -4.0);
    REQUIRE(child_relative_face_motion.
                envelope_east_coordinate_velocity_ms == 6.0);
    REQUIRE(child_relative_face_motion.
                envelope_south_coordinate_velocity_ms == -3.0);
    REQUIRE(child_relative_face_motion.
                envelope_north_coordinate_velocity_ms == 2.0);
    global_simulation_storage.configuration.domain_frame =
        original_parent_frame;
    global_simulation_storage.nested_execution.child.domain_frame =
        original_child_frame;

    auto invalid_face_motion = face_motion;
    invalid_face_motion.available = false;
    REQUIRE_THROWS_AS(prepare_nested_tracked_envelope(
        last_detection, continuity, invalid_face_motion,
        global_simulation_storage.time_s, 0.0, 0.0, 0.0, 0.0),
        std::invalid_argument);
    auto corrupt_continuity = continuity;
    corrupt_continuity.selected_candidate_rank =
        last_detection.reported_branch_candidate_count;
    REQUIRE_THROWS_AS(prepare_nested_tracked_envelope(
        last_detection, corrupt_continuity,
        global_simulation_storage.time_s, 0.0, 0.0, 0.0, 0.0),
        std::invalid_argument);

    const tmv::domain::StormWindowMovementPolicy policy{
        100.0, 1100.0, 1000.0, 800.0, 3, 0.0, 0.5, 2};
    const auto advance_to = [&](double target_time_s) {
        while (global_simulation_storage.time_s < target_time_s)
            advance_next_balanced_parent_interval({4.0});
        REQUIRE(global_simulation_storage.time_s == target_time_s);
    };

    std::string checkpoint_error;
    for (std::uint64_t movement = 1; movement <= 4; ++movement)
    {
        const int start_i = 6 + 2 * static_cast<int>(movement - 1);
        if (movement > 1)
        {
            const double approach_time_s =
                6.0 + 172.0 * static_cast<double>(movement - 1) - 4.0;
            advance_to(approach_time_s);
            envelope = detect(start_i, approach_time_s);
        }

        const double commit_time_s = global_simulation_storage.time_s;
        NestedChildMovementAuthorityReportV1 committed;
        REQUIRE_NOTHROW(evaluate_and_move_nested_child_from_tracker(
            global_simulation_storage, envelope, policy, {4.0}, committed));
        REQUIRE(committed.decision.reason ==
                tmv::domain::StormWindowMovementReason::EmergencyContainment);
        REQUIRE(committed.decision.confirmation_count == 1);
        REQUIRE(committed.movement_committed);
        REQUIRE(committed.movement.parent_shift_i == 2);
        REQUIRE(committed.movement.parent_shift_j == 0);
        REQUIRE(committed.movement.movement_sequence == movement);
        REQUIRE(global_simulation_storage.nested_execution.
                    cumulative_parent_shift_i ==
                2 * static_cast<int>(movement));
        require_finite_primary(
            global_simulation_storage.nested_execution.child.fields);
        REQUIRE_NOTHROW(require_valid_nested_execution_relationship(
            global_simulation_storage));

        const double rearm_time_s = commit_time_s + 2.0;
        advance_to(rearm_time_s);
        envelope = detect(start_i, rearm_time_s);
        NestedChildMovementAuthorityReportV1 rearmed;
        REQUIRE_NOTHROW(evaluate_and_move_nested_child_from_tracker(
            global_simulation_storage, envelope, policy, {4.0}, rearmed));
        REQUIRE(rearmed.decision.reason ==
                tmv::domain::StormWindowMovementReason::NoApproachingBreach);
        REQUIRE(global_simulation_storage.runtime.storm_window_movement.armed);

        if (movement == 2)
        {
            std::vector<std::uint8_t> before_restart;
            REQUIRE(tmv::checkpoint::serialize_simulation_storage(
                global_simulation_storage, before_restart, checkpoint_error));
            SimulationStorageState restarted;
            REQUIRE(tmv::checkpoint::deserialize_simulation_storage(
                before_restart, restarted, checkpoint_error));
            global_simulation_storage = std::move(restarted);
            continuity = global_simulation_storage.runtime.storm_tracking;
            std::vector<std::uint8_t> after_restart;
            REQUIRE(tmv::checkpoint::serialize_simulation_storage(
                global_simulation_storage, after_restart, checkpoint_error));
            REQUIRE(after_restart == before_restart);
            tmv::refresh_grid_metrics_from_terrain();
            tmv::initialize_dynamics(
                global_simulation_storage.configuration.dynamics_scheme_name);
            rehydrate_nested_child_transient_operators(
                global_simulation_storage);
            std::string compute_error;
            REQUIRE(initialize_compute_backend_runtime(compute_error));
        }
    }

    REQUIRE(global_simulation_storage.nested_execution.movement_sequence == 4);
    REQUIRE(global_simulation_storage.nested_execution.relationship.
                child_origin_parent_cell_i == 10);
    REQUIRE(global_simulation_storage.nested_execution.child.time_s ==
            global_simulation_storage.time_s);
}

TEST_CASE("child movement invalidates and rebuilds SGS and radar generations",
          "[core][nesting][production][movement][physics]")
{
    ProcessStateRestore restore;
    global_simulation_storage = make_parent();
    initialize_parent_modules();
    auto& storage = global_simulation_storage;

    TurbulenceConfig turbulence;
    turbulence.scheme_id = "smagorinsky";
    turbulence.mode = "3d";
    turbulence.dt_sgs = 1.0;
    initialize_turbulence(
        storage.runtime.turbulence, "smagorinsky", turbulence);

    storage.runtime.radar.selected_config = {};
    storage.runtime.radar.selected_config.scheme_id = "reflectivity";
    storage.runtime.radar.selected_config.operator_tier = "fast_da";
    storage.runtime.radar.selected_config.has_qr = true;
    storage.runtime.radar.product_selection.reflectivity = true;
    storage.runtime.radar.selected_scheme =
        tmv::create_radar_scheme("reflectivity");
    REQUIRE(storage.runtime.radar.selected_scheme);
    storage.runtime.radar.selected_scheme->initialize(
        storage.runtime.radar.selected_config,
        storage.grid.nr, storage.grid.nth, storage.grid.nz);
    storage.runtime.radar.nr = storage.grid.nr;
    storage.runtime.radar.nth = storage.grid.nth;
    storage.runtime.radar.nz = storage.grid.nz;

    const PrognosticFieldStorage previous_parent = storage.fields;
    prepare_child(previous_parent);
    NestedChildSubstepReport first;
    NestedChildSubstepReport second;
    advance_nested_child_dynamics_substep(
        storage, previous_parent, {4.0}, first);
    advance_nested_child_dynamics_substep(
        storage, previous_parent, {4.0}, second);
    REQUIRE(storage.nested_execution.child.time_s == storage.time_s);

    auto& child = storage.nested_execution.child;
    child.fields.qr[4][2][0] = 0.001f;
    {
        ScopedNestedChildResidency residency(storage);
        const auto child_state = nested_child_execution_state(storage);
        refresh_turbulence(child_state, child_state.time_s,
            child.runtime.turbulence.held_tendencies);
        calculate_radar_reflectivity(child_state);
    }
    auto& child_turbulence = child.runtime.turbulence;
    auto& child_radar = child.runtime.radar;
    REQUIRE(child_turbulence.realized_transport_tendency_current);
    REQUIRE(child_turbulence.scheme->realized_eddy_viscosity() != nullptr);
    REQUIRE(child_radar.selected_output.Ze_linear.size() > 0);
    REQUIRE(maximum_absolute_value(
                child_radar.selected_output.Ze_linear) > 0.0);
    REQUIRE(maximum_absolute_value(
                child.diagnostics.radar_reflectivity) > 0.0);
    const std::size_t held_u_capacity =
        child_turbulence.held_tendencies.dudt_sgs.capacity();
    const std::size_t radar_capacity =
        child_radar.selected_output.Ze_linear.capacity();
    REQUIRE(held_u_capacity > 0);
    REQUIRE(radar_capacity > 0);
    child_turbulence.last_update_time_s = 1.25;
    child_turbulence.next_update_time_s = 5.25;
    child_turbulence.cumulative_requested_momentum_work_j = 11.0;
    child_turbulence.momentum_work_step_count = 3;

    auto& boundary_generation = child.runtime.boundary_conditions;
    boundary_generation.radiative_u_west = {9.0f};
    auto& advection_generation = child.runtime.advection;
    advection_generation.hydrometeor_active[0] = true;
    advection_generation.activity_current = true;
    advection_generation.tracer_advected_in_batch = true;
    advection_generation.metadata_consumptions = 17;
    advection_generation.fallback_scans = 19;
    advection_generation.thermodynamic_boundary_report_step_count = 23;
    advection_generation.scalar_scratch_a.resize(
        child.grid.nr, child.grid.nth, child.grid.nz, 7.0f);
    const std::size_t advection_capacity =
        advection_generation.scalar_scratch_a.capacity();
    auto& diffusion_generation = child.runtime.diffusion;
    diffusion_generation.tendencies.dudt_diff.resize(
        child.grid.nr, child.grid.nth, child.grid.nz, 6.0f);
    diffusion_generation.cumulative_requested_momentum_work_j = 13.0;
    diffusion_generation.momentum_work_step_count = 5;
    const std::size_t diffusion_capacity =
        diffusion_generation.tendencies.dudt_diff.capacity();
    auto& stepping_generation = child.runtime.time_stepping;
    stepping_generation.slow_rk2.u_save.resize(
        child.grid.nr, child.grid.nth, child.grid.nz, 5.0f);
    stepping_generation.config.max_acoustic_speed_ms = 340.0;
    stepping_generation.config.min_acoustic_spacing_m = 200.0;
    stepping_generation.cumulative_outward_boundary_mass_kg = 17.0;
    stepping_generation.boundary_mass_step_count = 7;
    const std::size_t stepping_capacity =
        stepping_generation.slow_rk2.u_save.capacity();
    auto& dynamics_generation = child.runtime.dynamics;
    dynamics_generation.du_dt.resize(
        child.grid.nr, child.grid.nth, child.grid.nz, 4.0f);
    const std::size_t dynamics_capacity =
        dynamics_generation.du_dt.capacity();
    auto& microphysics_generation = child.runtime.microphysics;
    microphysics_generation.tendencies.dtheta_dt.resize(
        child.grid.nr, child.grid.nth, child.grid.nz, 3.0f);
    microphysics_generation.theta_storage_residual.resize(
        child.grid.nr, child.grid.nth, child.grid.nz, 2.0f);
    const std::size_t microphysics_capacity =
        microphysics_generation.tendencies.dtheta_dt.capacity();
    const std::size_t microphysics_residual_capacity =
        microphysics_generation.theta_storage_residual.capacity();

    NestedChildMovementReportV5 movement;
    REQUIRE_NOTHROW(move_nested_child_integer_parent_cells(
        storage, 1, 0, {4.0}, movement));
    REQUIRE(movement.movement_sequence == 1);
    REQUIRE_FALSE(child_turbulence.realized_transport_tendency_current);
    REQUIRE_FALSE(
        child_turbulence.stochastic_transport_applied_to_current);
    REQUIRE(std::isinf(
        child_turbulence.last_explicit_stability_limit_s));
    REQUIRE(child_turbulence.last_explicit_substep_count == 0);
    REQUIRE(child_turbulence.scheme->realized_eddy_viscosity() == nullptr);
    REQUIRE(maximum_absolute_value(
                child_turbulence.held_tendencies.dudt_sgs) == 0.0);
    REQUIRE(maximum_absolute_value(
                child_turbulence.candidate_tendencies.dudt_sgs) == 0.0);
    REQUIRE(child_turbulence.held_tendencies.dudt_sgs.capacity() ==
            held_u_capacity);
    REQUIRE(child_turbulence.last_update_time_s == 1.25);
    REQUIRE(child_turbulence.next_update_time_s == 5.25);
    REQUIRE(child_turbulence.cumulative_requested_momentum_work_j == 11.0);
    REQUIRE(child_turbulence.momentum_work_step_count == 3);
    REQUIRE(maximum_absolute_value(
                child_radar.selected_output.Ze_linear) == 0.0);
    REQUIRE(child_radar.selected_output.Ze_linear.capacity() ==
            radar_capacity);
    REQUIRE(child.diagnostics.radar_reflectivity.empty());
    REQUIRE(boundary_generation.radiative_u_west.empty());
    REQUIRE_FALSE(advection_generation.activity_current);
    REQUIRE_FALSE(advection_generation.hydrometeor_active[0]);
    REQUIRE_FALSE(advection_generation.tracer_advected_in_batch);
    REQUIRE(maximum_absolute_value(
                advection_generation.scalar_scratch_a) == 0.0);
    REQUIRE(advection_generation.scalar_scratch_a.capacity() ==
            advection_capacity);
    REQUIRE(advection_generation.metadata_consumptions == 17);
    REQUIRE(advection_generation.fallback_scans == 19);
    REQUIRE(advection_generation.thermodynamic_boundary_report_step_count ==
            23);
    REQUIRE(maximum_absolute_value(
                diffusion_generation.tendencies.dudt_diff) == 0.0);
    REQUIRE(diffusion_generation.tendencies.dudt_diff.capacity() ==
            diffusion_capacity);
    REQUIRE(diffusion_generation.cumulative_requested_momentum_work_j ==
            13.0);
    REQUIRE(diffusion_generation.momentum_work_step_count == 5);
    REQUIRE(maximum_absolute_value(
                stepping_generation.slow_rk2.u_save) == 0.0);
    REQUIRE(stepping_generation.slow_rk2.u_save.capacity() ==
            stepping_capacity);
    REQUIRE(stepping_generation.config.max_acoustic_speed_ms == 0.0);
    REQUIRE(stepping_generation.config.min_acoustic_spacing_m == 0.0);
    REQUIRE(stepping_generation.cumulative_outward_boundary_mass_kg == 17.0);
    REQUIRE(stepping_generation.boundary_mass_step_count == 7);
    REQUIRE(maximum_absolute_value(dynamics_generation.du_dt) == 0.0);
    REQUIRE(dynamics_generation.du_dt.capacity() == dynamics_capacity);
    REQUIRE(maximum_absolute_value(
                microphysics_generation.tendencies.dtheta_dt) == 0.0);
    REQUIRE(microphysics_generation.tendencies.dtheta_dt.capacity() ==
            microphysics_capacity);
    REQUIRE(maximum_absolute_value(
                microphysics_generation.theta_storage_residual) == 2.0);
    REQUIRE(microphysics_generation.theta_storage_residual.capacity() ==
            microphysics_residual_capacity);

    {
        ScopedNestedChildResidency residency(storage);
        const auto child_state = nested_child_execution_state(storage);
        refresh_turbulence(child_state, child_state.time_s,
            child.runtime.turbulence.held_tendencies);
        calculate_radar_reflectivity(child_state);
    }
    REQUIRE(child_turbulence.realized_transport_tendency_current);
    REQUIRE(child_turbulence.scheme->realized_eddy_viscosity() != nullptr);
    REQUIRE(child_turbulence.held_tendencies.dudt_sgs.capacity() ==
            held_u_capacity);
    REQUIRE(child_radar.selected_output.Ze_linear.capacity() ==
            radar_capacity);
    REQUIRE(maximum_absolute_value(
                child_radar.selected_output.Ze_linear) > 0.0);
    REQUIRE(maximum_absolute_value(
                child.diagnostics.radar_reflectivity) > 0.0);
    REQUIRE_NOTHROW(require_nested_child_level_runtime_ready(storage));
    REQUIRE_NOTHROW(advance_next_balanced_parent_interval({4.0}));
    REQUIRE(child.time_s == storage.time_s);
}

TEST_CASE("child movement preserves the exact held radiation sample",
          "[core][nesting][production][movement][radiation]")
{
    ProcessStateRestore restore;
    global_simulation_storage = make_parent();
    initialize_parent_modules();
    auto& storage = global_simulation_storage;
    const PrognosticFieldStorage previous_parent = storage.fields;
    prepare_child(previous_parent);
    NestedChildSubstepReport first;
    NestedChildSubstepReport second;
    advance_nested_child_dynamics_substep(
        storage, previous_parent, {4.0}, first);
    advance_nested_child_dynamics_substep(
        storage, previous_parent, {4.0}, second);

    RadiationConfig radiation_config;
    radiation_config.scheme_id = "simple_grey";
    radiation_config.do_lw = true;
    radiation_config.do_sw = false;
    radiation_config.dt_radiation = 300.0;
    storage.runtime.boundary_layer.surface_config.Tsfc = 300.0;
    storage.runtime.boundary_layer.surface_config.albedo = 0.2;
    storage.runtime.boundary_layer.surface_config.emissivity = 0.95;
    initialize_radiation(
        storage.runtime.radiation, "simple_grey", radiation_config);
    auto& child_level = storage.nested_execution.child;
    initialize_radiation(
        child_level.runtime.radiation, "simple_grey", radiation_config);

    auto prepare_held_state = [](AtmosphericLevelRuntimeState& runtime,
                                 PhysicsTendencyFieldStorage& tendencies,
                                 const GridStorageState& grid,
                                 float offset,
                                 float power_scale) {
        tendencies.dtheta_dt_rad.resize(
            grid.nr, grid.nth, grid.nz, 0.0f);
        runtime.radiation.held_cell_power_w.resize(
            grid.nr, grid.nth, grid.nz, 0.0f);
        runtime.radiation.last_update_time_s = 0.0;
        runtime.radiation.next_update_time_s = 300.0;
        const auto physical = physical_scalar_domain(
            grid.nr, grid.nth, grid.nz, StaggerType::CGrid);
        double aggregate = 0.0;
        for (int i = physical.i_begin; i < physical.i_end; ++i)
            for (int j = physical.j_begin; j < physical.j_end; ++j)
                for (int k = physical.k_begin; k < physical.k_end; ++k)
                {
                    tendencies.dtheta_dt_rad[i][j][k] =
                        offset + static_cast<float>(100 * i + 10 * j + k);
                    const float power = power_scale * static_cast<float>(
                        100 + 10 * i + j + k);
                    runtime.radiation.held_cell_power_w[i][j][k] = power;
                    aggregate += static_cast<double>(power);
                }
        runtime.radiation.held_atmospheric_power_w = aggregate;
    };
    prepare_held_state(
        storage.runtime, storage.tendencies, storage.grid, 1000.0f, 4.0f);
    prepare_held_state(
        child_level.runtime, child_level.tendencies, child_level.grid,
        -1000.0f, 1.0f);

    auto& child_radiation = child_level.runtime.radiation;
    child_radiation.accumulated_expected_atmospheric_energy_j = 31.0;
    child_radiation.accumulated_applied_radiation_heat_j = 29.0;
    child_radiation.applied_step_count = 7;
    child_radiation.scratch_theta_tendency.resize(
        child_level.grid.nr, child_level.grid.nth, child_level.grid.nz, 9.0f);
    child_radiation.scratch_cell_power_w.resize(
        child_level.grid.nr, child_level.grid.nth, child_level.grid.nz, 8.0f);
    child_radiation.scratch_column_power_w.assign(4, 7.0);
    child_radiation.scratch_column_errors.assign(4, 1);
    const std::size_t tendency_scratch_capacity =
        child_radiation.scratch_theta_tendency.capacity();
    const std::size_t power_scratch_capacity =
        child_radiation.scratch_cell_power_w.capacity();
    const Field3D old_child_tendency = child_level.tendencies.dtheta_dt_rad;
    const Field3D old_child_power = child_radiation.held_cell_power_w;
    const double previous_child_held_power =
        child_radiation.held_atmospheric_power_w;

    NestedChildMovementReportV5 rejected;
    rejected.movement_sequence = 99;
    child_radiation.next_update_time_s = 301.0;
    REQUIRE_THROWS_AS(move_nested_child_integer_parent_cells(
        storage, 1, 0, {4.0}, rejected), std::invalid_argument);
    REQUIRE(rejected.movement_sequence == 99);
    REQUIRE(std::equal(
        child_level.tendencies.dtheta_dt_rad.data(),
        child_level.tendencies.dtheta_dt_rad.data() +
            child_level.tendencies.dtheta_dt_rad.size(),
        old_child_tendency.data()));
    REQUIRE(std::equal(
        child_radiation.held_cell_power_w.data(),
        child_radiation.held_cell_power_w.data() +
            child_radiation.held_cell_power_w.size(),
        old_child_power.data()));
    child_radiation.next_update_time_s = 300.0;

    NestedChildMovementReportV5 rollback_rejection;
    REQUIRE_THROWS_AS(move_nested_child_integer_parent_cells(
        storage, 1, 0, {0.0}, rollback_rejection), std::invalid_argument);
    auto& radiation_movement_scratch = storage.nested_execution
        .boundary_scratch.movement_radiation_newly_admitted;
    std::array<const float*, 2> radiation_strip_addresses{};
    std::array<std::size_t, 2> radiation_strip_capacities{};
    for (std::size_t field = 0; field < radiation_movement_scratch.size();
         ++field)
    {
        REQUIRE_FALSE(radiation_movement_scratch[field].empty());
        radiation_strip_addresses[field] =
            radiation_movement_scratch[field].data();
        radiation_strip_capacities[field] =
            radiation_movement_scratch[field].capacity();
    }

    NestedChildMovementReportV5 movement;
    REQUIRE_NOTHROW(move_nested_child_integer_parent_cells(
        storage, 1, 0, {4.0}, movement));
    REQUIRE(movement.child_shift_i == 2);
    REQUIRE(movement.child_shift_j == 0);
    for (std::size_t field = 0; field < radiation_movement_scratch.size();
         ++field)
    {
        REQUIRE(radiation_movement_scratch[field].data() ==
                radiation_strip_addresses[field]);
        REQUIRE(radiation_movement_scratch[field].capacity() ==
                radiation_strip_capacities[field]);
    }
    const NestedExecutionStorage copied_nested =
        storage.nested_execution;
    for (const auto& field : copied_nested.boundary_scratch
             .movement_radiation_newly_admitted)
        REQUIRE(field.capacity() == 0);
    const auto& relationship = storage.nested_execution.relationship;
    const auto parent_physical = physical_scalar_domain(
        storage.grid.nr, storage.grid.nth, storage.grid.nz,
        StaggerType::CGrid);
    const auto child_physical = physical_scalar_domain(
        child_level.grid.nr, child_level.grid.nth, child_level.grid.nz,
        StaggerType::CGrid);
    double expected_aggregate = 0.0;
    for (int ci = 0; ci < relationship.child_cells_x; ++ci)
        for (int cj = 0; cj < relationship.child_cells_y; ++cj)
            for (int k = 0; k < relationship.levels_z; ++k)
            {
                const int i = child_physical.i_begin + ci;
                const int j = child_physical.j_begin + cj;
                float expected_tendency = 0.0f;
                float expected_power = 0.0f;
                if (ci + movement.child_shift_i <
                    relationship.child_cells_x)
                {
                    expected_tendency = old_child_tendency
                        [i + movement.child_shift_i][j][k];
                    expected_power = old_child_power
                        [i + movement.child_shift_i][j][k];
                }
                else
                {
                    const int pi = parent_physical.i_begin +
                        relationship.child_origin_parent_cell_i +
                        ci / relationship.parent_ratio_x;
                    const int pj = parent_physical.j_begin +
                        relationship.child_origin_parent_cell_j +
                        cj / relationship.parent_ratio_y;
                    expected_tendency =
                        storage.tendencies.dtheta_dt_rad[pi][pj][k];
                    expected_power = static_cast<float>(
                        storage.runtime.radiation.held_cell_power_w[pi][pj][k]
                        / 4.0f);
                }
                REQUIRE(child_level.tendencies.dtheta_dt_rad[i][j][k] ==
                        expected_tendency);
                REQUIRE(child_radiation.held_cell_power_w[i][j][k] ==
                        expected_power);
                expected_aggregate += static_cast<double>(expected_power);
            }
    REQUIRE(child_radiation.held_atmospheric_power_w == expected_aggregate);
    REQUIRE(movement.radiation.active);
    REQUIRE(movement.radiation.sample_time_s == 0.0);
    REQUIRE(movement.radiation.newly_admitted_cells == 16);
    REQUIRE(movement.radiation.previous_child_held_power_w ==
            previous_child_held_power);
    REQUIRE(movement.radiation.newly_admitted_target_power_w ==
            movement.radiation.newly_admitted_committed_power_w);
    REQUIRE(movement.radiation.newly_admitted_storage_rounding_w == 0.0);
    REQUIRE(movement.radiation.committed_child_held_power_w ==
            expected_aggregate);
    REQUIRE(child_radiation.last_update_time_s == 0.0);
    REQUIRE(child_radiation.next_update_time_s == 300.0);
    REQUIRE(child_radiation.accumulated_expected_atmospheric_energy_j == 31.0);
    REQUIRE(child_radiation.accumulated_applied_radiation_heat_j == 29.0);
    REQUIRE(child_radiation.applied_step_count == 7);
    REQUIRE(maximum_absolute_value(
                child_radiation.scratch_theta_tendency) == 0.0);
    REQUIRE(maximum_absolute_value(
                child_radiation.scratch_cell_power_w) == 0.0);
    REQUIRE(child_radiation.scratch_theta_tendency.capacity() ==
            tendency_scratch_capacity);
    REQUIRE(child_radiation.scratch_cell_power_w.capacity() ==
            power_scratch_capacity);
    REQUIRE(std::all_of(
        child_radiation.scratch_column_power_w.begin(),
        child_radiation.scratch_column_power_w.end(),
        [](double value) { return value == 0.0; }));
    REQUIRE(std::all_of(
        child_radiation.scratch_column_errors.begin(),
        child_radiation.scratch_column_errors.end(),
        [](unsigned char value) { return value == 0; }));

    std::string checkpoint_error;
    std::vector<std::uint8_t> moved_checkpoint;
    REQUIRE(tmv::checkpoint::serialize_simulation_storage(
        storage, moved_checkpoint, checkpoint_error));
    SimulationStorageState restored;
    REQUIRE(tmv::checkpoint::deserialize_simulation_storage(
        moved_checkpoint, restored, checkpoint_error));
    std::vector<std::uint8_t> restored_checkpoint;
    REQUIRE(tmv::checkpoint::serialize_simulation_storage(
        restored, restored_checkpoint, checkpoint_error));
    REQUIRE(restored_checkpoint == moved_checkpoint);
}

TEST_CASE("child movement preserves the exact held boundary-layer sample",
          "[core][nesting][production][movement][boundary_layer]")
{
    ProcessStateRestore restore;
    global_simulation_storage = make_parent();
    initialize_parent_modules();
    auto& storage = global_simulation_storage;
    const PrognosticFieldStorage previous_parent = storage.fields;
    prepare_child(previous_parent);
    NestedChildSubstepReport first;
    NestedChildSubstepReport second;
    advance_nested_child_dynamics_substep(
        storage, previous_parent, {4.0}, first);
    advance_nested_child_dynamics_substep(
        storage, previous_parent, {4.0}, second);

    BoundaryLayerConfig pbl_config;
    pbl_config.scheme_id = "slab";
    pbl_config.dt_pbl = 60.0;
    SurfaceConfig surface_config;
    surface_config.Tsfc = 302.0;
    surface_config.qsfc = 0.012;
    initialize_boundary_layer(
        storage.runtime.boundary_layer, "slab", pbl_config,
        surface_config);
    auto& child_level = storage.nested_execution.child;
    initialize_boundary_layer(
        child_level.runtime.boundary_layer, "slab", pbl_config,
        surface_config);

    auto prepare_held_state = [](
        AtmosphericLevelRuntimeState& level_runtime,
        PhysicsTendencyFieldStorage& tendencies,
        SurfaceDiagnosticFieldStorage& surface,
        const GridStorageState& grid,
        float offset,
        float extensive_scale) {
        auto& runtime = level_runtime.boundary_layer;
        runtime.last_update_time_s = 0.0;
        runtime.next_update_time_s = 60.0;
        runtime.updated_this_step = true;
        runtime.skin_captured = true;
        runtime.skin_nr = grid.nr;
        runtime.skin_nth = grid.nth;
        runtime.slab_nr = grid.nr;
        runtime.slab_nth = grid.nth;
        const std::size_t columns =
            static_cast<std::size_t>(grid.nr) *
            static_cast<std::size_t>(grid.nth);
        runtime.skin_theta_2d.assign(columns, 290.0f + offset);
        runtime.skin_qv_2d.assign(columns, 0.005f);
        runtime.slab_columns.assign(columns, SlabColumnState{});
        std::array<std::vector<float>*, 7> surface_fields = {{
            &surface.ustar, &surface.tau_u, &surface.tau_v,
            &surface.sensible_heat_flux, &surface.latent_heat_flux,
            &surface.moisture_flux, &surface.pbl_height}};
        for (std::size_t field = 0; field < surface_fields.size(); ++field)
            surface_fields[field]->assign(
                columns, offset + static_cast<float>(100 * field));
        runtime.held_u_force_n.resize(grid.nr, grid.nth, grid.nz, 0.0f);
        runtime.held_v_force_n.resize(grid.nr, grid.nth, grid.nz, 0.0f);
        runtime.held_theta_inventory_rate_kgk_s.resize(
            grid.nr, grid.nth, grid.nz, 0.0f);
        runtime.held_vapor_mass_rate_kg_s.resize(
            grid.nr, grid.nth, grid.nz, 0.0f);
        runtime.held_cloud_liquid_mass_rate_kg_s.resize(
            grid.nr, grid.nth, grid.nz, 0.0f);
        runtime.held_cloud_ice_mass_rate_kg_s.resize(
            grid.nr, grid.nth, grid.nz, 0.0f);
        tendencies.dtke_dt_pbl.resize(grid.nr, grid.nth, grid.nz, 0.0f);
        const auto physical = physical_scalar_domain(
            grid.nr, grid.nth, grid.nz, StaggerType::CGrid);
        for (int i = physical.i_begin; i < physical.i_end; ++i)
            for (int j = physical.j_begin; j < physical.j_end; ++j)
            {
                const auto column = static_cast<std::size_t>(i) *
                    static_cast<std::size_t>(grid.nth) +
                    static_cast<std::size_t>(j);
                runtime.skin_theta_2d[column] =
                    280.0f + offset + static_cast<float>(10 * i + j);
                runtime.skin_qv_2d[column] =
                    0.001f * static_cast<float>(i + j + 1);
                runtime.slab_columns[column] = {
                    100.0 + i + j, 290.0 + i + j,
                    0.001 * (i + j + 1), true};
                for (std::size_t field = 0; field < surface_fields.size();
                     ++field)
                    (*surface_fields[field])[column] =
                        offset + static_cast<float>(
                            1000 * field + 10 * i + j);
                for (int k = physical.k_begin; k < physical.k_end; ++k)
                {
                    const float base = offset +
                        static_cast<float>(100 * i + 10 * j + k);
                    runtime.held_u_force_n[i][j][k] =
                        extensive_scale * (base + 1.0f);
                    runtime.held_v_force_n[i][j][k] =
                        extensive_scale * (base + 2.0f);
                    runtime.held_theta_inventory_rate_kgk_s[i][j][k] =
                        extensive_scale * (base + 3.0f);
                    runtime.held_vapor_mass_rate_kg_s[i][j][k] =
                        extensive_scale * (base + 4.0f);
                    runtime.held_cloud_liquid_mass_rate_kg_s[i][j][k] =
                        extensive_scale * (base + 5.0f);
                    runtime.held_cloud_ice_mass_rate_kg_s[i][j][k] =
                        extensive_scale * (base + 6.0f);
                    tendencies.dtke_dt_pbl[i][j][k] = base + 5.0f;
                }
            }
    };
    prepare_held_state(
        storage.runtime, storage.tendencies, storage.surface,
        storage.grid, 1000.0f, 4.0f);
    prepare_held_state(
        child_level.runtime, child_level.tendencies, child_level.surface,
        child_level.grid, -100.0f, 1.0f);

    auto& child_pbl = child_level.runtime.boundary_layer;
    child_pbl.last_momentum_work.committed_work_j = 10.0;
    child_pbl.cumulative_requested_momentum_work_j = 11.0;
    child_pbl.cumulative_finite_step_momentum_correction_j = 12.0;
    child_pbl.cumulative_momentum_storage_rounding_j = 13.0;
    child_pbl.cumulative_committed_momentum_work_j = 14.0;
    child_pbl.momentum_work_step_count = 15;
    child_pbl.refresh_scratch.u_force.resize(
        child_level.grid.nr, child_level.grid.nth,
        child_level.grid.nz, 9.0f);
    child_pbl.refresh_scratch.skin_theta.assign(4, 8.0f);
    const std::size_t scratch_capacity =
        child_pbl.refresh_scratch.u_force.capacity();
    const Field3D old_child_u_force = child_pbl.held_u_force_n;
    const Field3D old_child_v_force = child_pbl.held_v_force_n;
    const Field3D old_child_theta_rate =
        child_pbl.held_theta_inventory_rate_kgk_s;
    const Field3D old_child_vapor_rate =
        child_pbl.held_vapor_mass_rate_kg_s;
    const Field3D old_child_cloud_liquid_rate =
        child_pbl.held_cloud_liquid_mass_rate_kg_s;
    const Field3D old_child_cloud_ice_rate =
        child_pbl.held_cloud_ice_mass_rate_kg_s;
    const Field3D old_child_dtke = child_level.tendencies.dtke_dt_pbl;
    const auto old_skin_theta = child_pbl.skin_theta_2d;
    const auto old_slab = child_pbl.slab_columns;
    const auto old_ustar = child_level.surface.ustar;

    NestedChildMovementReportV5 rejected;
    rejected.movement_sequence = 99;
    child_pbl.next_update_time_s = 61.0;
    REQUIRE_THROWS_AS(move_nested_child_integer_parent_cells(
        storage, 1, 0, {4.0}, rejected), std::invalid_argument);
    REQUIRE(rejected.movement_sequence == 99);
    REQUIRE(std::equal(
        child_pbl.held_u_force_n.data(),
        child_pbl.held_u_force_n.data() + child_pbl.held_u_force_n.size(),
        old_child_u_force.data()));
    child_pbl.next_update_time_s = 60.0;

    NestedChildMovementReportV5 rollback_rejection;
    REQUIRE_THROWS_AS(move_nested_child_integer_parent_cells(
        storage, 1, 0, {0.0}, rollback_rejection), std::invalid_argument);
    auto& pbl_cell_scratch = storage.nested_execution.boundary_scratch
        .movement_boundary_layer_cell_newly_admitted;
    auto& pbl_column_scratch = storage.nested_execution.boundary_scratch
        .movement_boundary_layer_column_newly_admitted;
    auto& pbl_slab_scratch = storage.nested_execution.boundary_scratch
        .movement_boundary_layer_slab_newly_admitted;
    std::array<const float*, 7> pbl_cell_addresses{};
    std::array<std::size_t, 7> pbl_cell_capacities{};
    std::array<const float*, 9> pbl_column_addresses{};
    std::array<std::size_t, 9> pbl_column_capacities{};
    for (std::size_t field = 0; field < pbl_cell_scratch.size(); ++field)
    {
        REQUIRE_FALSE(pbl_cell_scratch[field].empty());
        pbl_cell_addresses[field] = pbl_cell_scratch[field].data();
        pbl_cell_capacities[field] = pbl_cell_scratch[field].capacity();
    }
    for (std::size_t field = 0; field < pbl_column_scratch.size(); ++field)
    {
        REQUIRE_FALSE(pbl_column_scratch[field].empty());
        pbl_column_addresses[field] = pbl_column_scratch[field].data();
        pbl_column_capacities[field] =
            pbl_column_scratch[field].capacity();
    }
    REQUIRE_FALSE(pbl_slab_scratch.empty());
    const tmv::SlabColumnState* const pbl_slab_address =
        pbl_slab_scratch.data();
    const std::size_t pbl_slab_capacity = pbl_slab_scratch.capacity();

    NestedChildMovementReportV5 movement;
    REQUIRE_NOTHROW(move_nested_child_integer_parent_cells(
        storage, 1, 0, {4.0}, movement));
    REQUIRE(movement.child_shift_i == 2);
    REQUIRE(movement.child_shift_j == 0);
    for (std::size_t field = 0; field < pbl_cell_scratch.size(); ++field)
    {
        REQUIRE(pbl_cell_scratch[field].data() ==
                pbl_cell_addresses[field]);
        REQUIRE(pbl_cell_scratch[field].capacity() ==
                pbl_cell_capacities[field]);
    }
    for (std::size_t field = 0; field < pbl_column_scratch.size(); ++field)
    {
        REQUIRE(pbl_column_scratch[field].data() ==
                pbl_column_addresses[field]);
        REQUIRE(pbl_column_scratch[field].capacity() ==
                pbl_column_capacities[field]);
    }
    REQUIRE(pbl_slab_scratch.data() == pbl_slab_address);
    REQUIRE(pbl_slab_scratch.capacity() == pbl_slab_capacity);
    const NestedExecutionStorage copied_nested =
        storage.nested_execution;
    for (const auto& field : copied_nested.boundary_scratch
             .movement_boundary_layer_cell_newly_admitted)
        REQUIRE(field.capacity() == 0);
    for (const auto& field : copied_nested.boundary_scratch
             .movement_boundary_layer_column_newly_admitted)
        REQUIRE(field.capacity() == 0);
    REQUIRE(copied_nested.boundary_scratch
                .movement_boundary_layer_slab_newly_admitted.capacity() ==
            0);
    const auto& relationship = storage.nested_execution.relationship;
    const auto parent_physical = physical_scalar_domain(
        storage.grid.nr, storage.grid.nth, storage.grid.nz,
        StaggerType::CGrid);
    const auto child_physical = physical_scalar_domain(
        child_level.grid.nr, child_level.grid.nth, child_level.grid.nz,
        StaggerType::CGrid);
    double expected_x_force = 0.0;
    double expected_y_force = 0.0;
    double expected_theta_rate = 0.0;
    double expected_vapor_rate = 0.0;
    double expected_cloud_liquid_rate = 0.0;
    double expected_cloud_ice_rate = 0.0;
    for (int ci = 0; ci < relationship.child_cells_x; ++ci)
        for (int cj = 0; cj < relationship.child_cells_y; ++cj)
        {
            const int i = child_physical.i_begin + ci;
            const int j = child_physical.j_begin + cj;
            const bool retained =
                ci + movement.child_shift_i < relationship.child_cells_x;
            const int pi = parent_physical.i_begin +
                relationship.child_origin_parent_cell_i +
                ci / relationship.parent_ratio_x;
            const int pj = parent_physical.j_begin +
                relationship.child_origin_parent_cell_j +
                cj / relationship.parent_ratio_y;
            const auto child_column = static_cast<std::size_t>(i) *
                static_cast<std::size_t>(child_level.grid.nth) + j;
            const auto old_column = static_cast<std::size_t>(
                i + movement.child_shift_i) *
                static_cast<std::size_t>(child_level.grid.nth) + j;
            const auto parent_column = static_cast<std::size_t>(pi) *
                static_cast<std::size_t>(storage.grid.nth) + pj;
            REQUIRE(child_pbl.skin_theta_2d[child_column] ==
                    (retained ? old_skin_theta[old_column]
                              : storage.runtime.boundary_layer.
                                    skin_theta_2d[parent_column]));
            REQUIRE(child_level.surface.ustar[child_column] ==
                    (retained ? old_ustar[old_column]
                              : storage.surface.ustar[parent_column]));
            const SlabColumnState& expected_slab = retained
                ? old_slab[old_column]
                : storage.runtime.boundary_layer.slab_columns[parent_column];
            REQUIRE(child_pbl.slab_columns[child_column].initialized ==
                    expected_slab.initialized);
            REQUIRE(child_pbl.slab_columns[child_column].h_m ==
                    expected_slab.h_m);
            REQUIRE(child_pbl.slab_columns[child_column].theta_m_k ==
                    expected_slab.theta_m_k);
            REQUIRE(child_pbl.slab_columns[child_column].qv_m_kgkg ==
                    expected_slab.qv_m_kgkg);
            for (int k = 0; k < relationship.levels_z; ++k)
            {
                const int old_i = i + movement.child_shift_i;
                const float expected_u = retained
                    ? old_child_u_force[old_i][j][k]
                    : static_cast<float>(
                        storage.runtime.boundary_layer.held_u_force_n
                            [pi][pj][k] / 4.0f);
                const float expected_v = retained
                    ? old_child_v_force[old_i][j][k]
                    : static_cast<float>(
                        storage.runtime.boundary_layer.held_v_force_n
                            [pi][pj][k] / 4.0f);
                const float expected_theta = retained
                    ? old_child_theta_rate[old_i][j][k]
                    : static_cast<float>(
                        storage.runtime.boundary_layer.
                            held_theta_inventory_rate_kgk_s[pi][pj][k] /
                        4.0f);
                const float expected_vapor = retained
                    ? old_child_vapor_rate[old_i][j][k]
                    : static_cast<float>(
                        storage.runtime.boundary_layer.
                            held_vapor_mass_rate_kg_s[pi][pj][k] / 4.0f);
                const float expected_cloud_liquid = retained
                    ? old_child_cloud_liquid_rate[old_i][j][k]
                    : static_cast<float>(
                        storage.runtime.boundary_layer.
                            held_cloud_liquid_mass_rate_kg_s[pi][pj][k] /
                        4.0f);
                const float expected_cloud_ice = retained
                    ? old_child_cloud_ice_rate[old_i][j][k]
                    : static_cast<float>(
                        storage.runtime.boundary_layer.
                            held_cloud_ice_mass_rate_kg_s[pi][pj][k] /
                        4.0f);
                const float expected_dtke = retained
                    ? old_child_dtke[old_i][j][k]
                    : storage.tendencies.dtke_dt_pbl[pi][pj][k];
                REQUIRE(child_pbl.held_u_force_n[i][j][k] == expected_u);
                REQUIRE(child_pbl.held_v_force_n[i][j][k] == expected_v);
                REQUIRE(child_pbl.held_theta_inventory_rate_kgk_s[i][j][k]
                        == expected_theta);
                REQUIRE(child_pbl.held_vapor_mass_rate_kg_s[i][j][k] ==
                        expected_vapor);
                REQUIRE(child_pbl.held_cloud_liquid_mass_rate_kg_s[i][j][k]
                        == expected_cloud_liquid);
                REQUIRE(child_pbl.held_cloud_ice_mass_rate_kg_s[i][j][k]
                        == expected_cloud_ice);
                REQUIRE(child_level.tendencies.dtke_dt_pbl[i][j][k] ==
                        expected_dtke);
                expected_x_force += expected_u;
                expected_y_force += expected_v;
                expected_theta_rate += expected_theta;
                expected_vapor_rate += expected_vapor;
                expected_cloud_liquid_rate += expected_cloud_liquid;
                expected_cloud_ice_rate += expected_cloud_ice;
            }
        }
    REQUIRE(movement.boundary_layer.active);
    REQUIRE(movement.boundary_layer.sample_time_s == 0.0);
    REQUIRE(movement.boundary_layer.newly_admitted_columns == 8);
    REQUIRE(movement.boundary_layer.newly_admitted_cells == 16);
    REQUIRE(movement.boundary_layer.newly_admitted_target_x_force_n -
                movement.boundary_layer.newly_admitted_committed_x_force_n ==
            movement.boundary_layer.
                newly_admitted_x_force_storage_rounding_n);
    REQUIRE(movement.boundary_layer.newly_admitted_target_y_force_n -
                movement.boundary_layer.newly_admitted_committed_y_force_n ==
            movement.boundary_layer.
                newly_admitted_y_force_storage_rounding_n);
    REQUIRE(movement.boundary_layer.
                newly_admitted_target_theta_inventory_rate_kgk_s -
                movement.boundary_layer.
                    newly_admitted_committed_theta_inventory_rate_kgk_s ==
            movement.boundary_layer.
                newly_admitted_theta_storage_rounding_kgk_s);
    REQUIRE(movement.boundary_layer.
                newly_admitted_target_vapor_mass_rate_kg_s -
                movement.boundary_layer.
                    newly_admitted_committed_vapor_mass_rate_kg_s ==
            movement.boundary_layer.
                newly_admitted_vapor_storage_rounding_kg_s);
    REQUIRE(movement.boundary_layer.
                newly_admitted_target_cloud_liquid_mass_rate_kg_s -
                movement.boundary_layer.
                    newly_admitted_committed_cloud_liquid_mass_rate_kg_s ==
            movement.boundary_layer.
                newly_admitted_cloud_liquid_storage_rounding_kg_s);
    REQUIRE(movement.boundary_layer.
                newly_admitted_target_cloud_ice_mass_rate_kg_s -
                movement.boundary_layer.
                    newly_admitted_committed_cloud_ice_mass_rate_kg_s ==
            movement.boundary_layer.
                newly_admitted_cloud_ice_storage_rounding_kg_s);
    REQUIRE(movement.boundary_layer.committed_child_x_force_n ==
            expected_x_force);
    REQUIRE(movement.boundary_layer.committed_child_y_force_n ==
            expected_y_force);
    REQUIRE(movement.boundary_layer.
                committed_child_theta_inventory_rate_kgk_s ==
            expected_theta_rate);
    REQUIRE(movement.boundary_layer.committed_child_vapor_mass_rate_kg_s ==
            expected_vapor_rate);
    REQUIRE(movement.boundary_layer.
                committed_child_cloud_liquid_mass_rate_kg_s ==
            expected_cloud_liquid_rate);
    REQUIRE(movement.boundary_layer.committed_child_cloud_ice_mass_rate_kg_s
            == expected_cloud_ice_rate);
    REQUIRE(child_pbl.last_update_time_s == 0.0);
    REQUIRE(child_pbl.next_update_time_s == 60.0);
    REQUIRE_FALSE(child_pbl.updated_this_step);
    REQUIRE(child_pbl.last_momentum_work.committed_work_j == 10.0);
    REQUIRE(child_pbl.cumulative_requested_momentum_work_j == 11.0);
    REQUIRE(child_pbl.cumulative_finite_step_momentum_correction_j == 12.0);
    REQUIRE(child_pbl.cumulative_momentum_storage_rounding_j == 13.0);
    REQUIRE(child_pbl.cumulative_committed_momentum_work_j == 14.0);
    REQUIRE(child_pbl.momentum_work_step_count == 15);
    REQUIRE(maximum_absolute_value(child_level.tendencies.dtheta_dt_pbl) ==
            0.0);
    REQUIRE(maximum_absolute_value(child_level.tendencies.dqv_dt_pbl) ==
            0.0);
    REQUIRE(maximum_absolute_value(child_level.tendencies.du_dt_pbl) == 0.0);
    REQUIRE(maximum_absolute_value(child_level.tendencies.dv_dt_pbl) == 0.0);
    REQUIRE(maximum_absolute_value(child_pbl.refresh_scratch.u_force) == 0.0);
    REQUIRE(child_pbl.refresh_scratch.u_force.capacity() == scratch_capacity);
    REQUIRE(std::all_of(
        child_pbl.refresh_scratch.skin_theta.begin(),
        child_pbl.refresh_scratch.skin_theta.end(),
        [](float value) { return value == 0.0f; }));

    std::string checkpoint_error;
    std::vector<std::uint8_t> moved_checkpoint;
    REQUIRE(tmv::checkpoint::serialize_simulation_storage(
        storage, moved_checkpoint, checkpoint_error));
    SimulationStorageState restored;
    REQUIRE(tmv::checkpoint::deserialize_simulation_storage(
        moved_checkpoint, restored, checkpoint_error));
    for (const auto& field : restored.nested_execution.boundary_scratch
             .movement_boundary_layer_cell_newly_admitted)
        REQUIRE(field.capacity() == 0);
    for (const auto& field : restored.nested_execution.boundary_scratch
             .movement_boundary_layer_column_newly_admitted)
        REQUIRE(field.capacity() == 0);
    REQUIRE(restored.nested_execution.boundary_scratch
                .movement_boundary_layer_slab_newly_admitted.capacity() ==
            0);
    for (const auto& field : restored.nested_execution.boundary_scratch
             .movement_chaos_cache_cell_newly_admitted)
        REQUIRE(field.capacity() == 0);
    for (const auto& field : restored.nested_execution.boundary_scratch
             .movement_chaos_cache_column_newly_admitted)
        REQUIRE(field.capacity() == 0);
    std::vector<std::uint8_t> restored_checkpoint;
    REQUIRE(tmv::checkpoint::serialize_simulation_storage(
        restored, restored_checkpoint, checkpoint_error));
    REQUIRE(restored_checkpoint == moved_checkpoint);
}

TEST_CASE("moving child preserves Milbrandt number moments and theta carry",
          "[core][nesting][production][movement][microphysics][restart]")
{
    ProcessStateRestore restore;
    global_simulation_storage = make_parent();
    initialize_parent_modules();
    auto& storage = global_simulation_storage;
    storage.configuration.microphysics_scheme = "milbrandt";
    storage.runtime.microphysics.scheme = tmv::create_microphysics_scheme(
        "milbrandt", storage.configuration.microphysics_config);
    REQUIRE(storage.runtime.microphysics.scheme);
    storage.runtime.microphysics.scheme->prepare_prognostic_state(
        storage.fields.rho, storage.fields.qc, storage.fields.qr,
        storage.fields.qi, storage.fields.qs, storage.fields.qg,
        storage.fields.qh);
    PortableSchemeState parent_microphysics_state =
        storage.runtime.microphysics.scheme->portable_state();
    REQUIRE(parent_microphysics_state.fields.size() == 13);
    for (std::size_t field = 6; field < 12; ++field)
        parent_microphysics_state.fields[field].resize(
            storage.grid.nr, storage.grid.nth, storage.grid.nz, 0.0f);
    parent_microphysics_state.fields[12].resize(
        storage.grid.nr, storage.grid.nth, 1, 0.0f);
    std::string parent_microphysics_error;
    REQUIRE(storage.runtime.microphysics.scheme->restore_portable_state(
        parent_microphysics_state, parent_microphysics_error));

    const PrognosticFieldStorage previous_parent = storage.fields;
    NestedParentMicrophysicsSnapshotV1 previous_microphysics;
    capture_nested_parent_microphysics_snapshot(
        storage, previous_microphysics);
    previous_microphysics.sample_time_s = 0.0;
    auto evolving_parent_moments = storage.runtime.microphysics.scheme
        ->prognostic_scalars_for_transport();
    for (std::size_t species = 0;
         species < evolving_parent_moments.size(); ++species)
        for (int i = 1; i < storage.grid.nr - 1; ++i)
            for (int j = 1; j < storage.grid.nth - 1; ++j)
                for (int k = 0; k < storage.grid.nz; ++k)
                    (*evolving_parent_moments[species].field)[i][j][k] =
                        previous_microphysics.fields[species][i][j][k] +
                        static_cast<float>(
                            1000 * (species + 1) + 10 * i + j);
    for (int i = 1; i < storage.grid.nr - 1; ++i)
        for (int j = 1; j < storage.grid.nth - 1; ++j)
            for (int k = 0; k < storage.grid.nz; ++k)
            {
                storage.fields.rho[i][j][k] *= 1.01f;
                storage.fields.p[i][j][k] = static_cast<float>(
                    tmv::dynamics_eos::pressure_fp64(
                        storage.fields.rho[i][j][k],
                        storage.fields.theta[i][j][k]));
            }
    prepare_child(previous_parent);
    NestedChildInitializationReport private_initialization;
    initialize_nested_child_microphysics_private_state(
        storage, previous_parent, previous_microphysics,
        private_initialization);
    NestedChildSubstepReport first;
    NestedChildSubstepReport second;
    advance_nested_child_dynamics_substep(
        storage, previous_parent, {4.0}, first,
        &previous_microphysics);
    {
        auto first_phase_moments = storage.nested_execution.child.runtime
            .microphysics.scheme->prognostic_scalars_for_transport();
        const auto expected = [&](std::size_t species, int pi, int pj, int k) {
            const double previous_rho = previous_parent.rho[pi][pj][k];
            const double current_rho = storage.fields.rho[pi][pj][k];
            const double previous_value =
                previous_microphysics.fields[species][pi][pj][k];
            const double current_value =
                (*evolving_parent_moments[species].field)[pi][pj][k];
            return static_cast<float>(
                (0.5 * previous_rho * previous_value +
                 0.5 * current_rho * current_value) /
                (0.5 * previous_rho + 0.5 * current_rho));
        };
        for (std::size_t species = 0;
             species < first_phase_moments.size(); ++species)
        {
            REQUIRE((*first_phase_moments[species].field)[1][1][0] ==
                    expected(species, 3, 3, 0));
            REQUIRE((*first_phase_moments[species].field)[0][0][0] ==
                    expected(species, 2, 2, 0));
        }
    }
    advance_nested_child_dynamics_substep(
        storage, previous_parent, {4.0}, second,
        &previous_microphysics);
    REQUIRE(private_initialization.microphysics_private_scalar_count == 6);
    for (const NestedChildSubstepReport* phase : {&first, &second})
    {
        const auto& private_source =
            phase->interface_source.microphysics_private;
        REQUIRE(private_source.scalar_count == 6);
        for (std::size_t species = 0; species < private_source.scalar_count;
             ++species)
        {
            REQUIRE_FALSE(std::string(
                private_source.name[species].data()).empty());
            REQUIRE(private_source.storage_semantic[species] ==
                static_cast<std::uint8_t>(
                    tmv::MicrophysicsScalarStorage::DryAirMixingRatio));
            REQUIRE(private_source.positive_definite[species]);
            REQUIRE(private_source.carrier_storage_residual[species] ==
                private_source.target_carrier_change[species] -
                    private_source.committed_carrier_change[species]);
            REQUIRE(private_source.forced_cells[species] == 24);
        }
    }
    auto& child_level = storage.nested_execution.child;
    REQUIRE(child_level.time_s == storage.time_s);
    const std::size_t parent_surface_size =
        static_cast<std::size_t>(storage.grid.nr) *
        static_cast<std::size_t>(storage.grid.nth);
    storage.surface.precipitation_rate_mm_h.assign(
        parent_surface_size, 0.0f);
    storage.surface.accumulated_rainfall_mm.assign(
        parent_surface_size, 0.0f);
    storage.surface.accumulated_rainfall_residual_mm.assign(
        parent_surface_size, 0.0f);
    storage.surface.accumulated_rainfall_last_update_s =
        child_level.surface.accumulated_rainfall_last_update_s;

    auto parent_moments = storage.runtime.microphysics.scheme
        ->prognostic_scalars_for_transport();
    auto child_moments = child_level.runtime.microphysics.scheme
        ->prognostic_scalars_for_transport();
    REQUIRE(parent_moments.size() == 6);
    REQUIRE(child_moments.size() == parent_moments.size());
    for (std::size_t species = 0; species < parent_moments.size(); ++species)
    {
        for (int i = 1; i < storage.grid.nr - 1; ++i)
            for (int j = 1; j < storage.grid.nth - 1; ++j)
                for (int k = 0; k < storage.grid.nz; ++k)
                    (*parent_moments[species].field)[i][j][k] =
                        static_cast<float>(
                            100000 * (species + 1) + 1000 * i +
                            10 * j + k);
        for (int i = 1; i < child_level.grid.nr - 1; ++i)
            for (int j = 1; j < child_level.grid.nth - 1; ++j)
                for (int k = 0; k < child_level.grid.nz; ++k)
                    (*child_moments[species].field)[i][j][k] =
                        static_cast<float>(
                            700000 * (species + 1) + 1000 * i +
                            10 * j + k);
    }
    std::array<Field3D, 6> old_child_moments;
    for (std::size_t species = 0; species < child_moments.size(); ++species)
        old_child_moments[species] = *child_moments[species].field;

    auto& theta_residual =
        child_level.runtime.microphysics.theta_storage_residual;
    theta_residual.resize(
        child_level.grid.nr, child_level.grid.nth,
        child_level.grid.nz, -9.0f);
    for (int i = 1; i < child_level.grid.nr - 1; ++i)
        for (int j = 1; j < child_level.grid.nth - 1; ++j)
            for (int k = 0; k < child_level.grid.nz; ++k)
                theta_residual[i][j][k] = static_cast<float>(
                    1000 * i + 10 * j + k);
    const Field3D old_theta_residual = theta_residual;

    const float valid_private = (*child_moments[0].field)[1][1][0];
    auto* const child_microphysics_identity =
        child_level.runtime.microphysics.scheme.get();
    (*child_moments[0].field)[1][1][0] =
        std::numeric_limits<float>::quiet_NaN();
    NestedChildMovementReportV5 rejected;
    rejected.movement_sequence = 99;
    REQUIRE_THROWS_AS(move_nested_child_integer_parent_cells(
        storage, 1, 0, {4.0}, rejected), std::invalid_argument);
    REQUIRE(rejected.movement_sequence == 99);
    REQUIRE(storage.nested_execution.movement_sequence == 0);
    REQUIRE(storage.nested_execution.relationship
                .child_origin_parent_cell_i == 2);
    REQUIRE(child_level.runtime.microphysics.scheme.get() ==
            child_microphysics_identity);
    (*child_moments[0].field)[1][1][0] = valid_private;

    std::string checkpoint_error;
    std::vector<std::uint8_t> before_shift_rollback;
    REQUIRE(tmv::checkpoint::serialize_simulation_storage(
        storage, before_shift_rollback, checkpoint_error));
    NestedChildMovementReportV5 rollback_rejection;
    rollback_rejection.movement_sequence = 98;
    REQUIRE_THROWS_AS(move_nested_child_integer_parent_cells(
        storage, 1, 0, {0.0}, rollback_rejection), std::invalid_argument);
    REQUIRE(rollback_rejection.movement_sequence == 98);
    REQUIRE(child_level.runtime.microphysics.scheme.get() ==
            child_microphysics_identity);
    std::vector<std::uint8_t> after_shift_rollback;
    REQUIRE(tmv::checkpoint::serialize_simulation_storage(
        storage, after_shift_rollback, checkpoint_error));
    REQUIRE(after_shift_rollback == before_shift_rollback);
    auto& movement_scratch =
        storage.nested_execution.boundary_scratch;
    std::array<const float*, 6> private_new_addresses{};
    std::array<const float*, 6> private_discard_addresses{};
    std::array<std::size_t, 6> private_new_capacities{};
    std::array<std::size_t, 6> private_discard_capacities{};
    for (std::size_t species = 0; species < 6; ++species)
    {
        const auto& newly_admitted =
            movement_scratch.movement_microphysics_newly_admitted[species];
        const auto& discarded =
            movement_scratch.movement_microphysics_discard[species];
        REQUIRE_FALSE(newly_admitted.empty());
        REQUIRE_FALSE(discarded.empty());
        private_new_addresses[species] = newly_admitted.data();
        private_discard_addresses[species] = discarded.data();
        private_new_capacities[species] = newly_admitted.capacity();
        private_discard_capacities[species] = discarded.capacity();
    }
    const float* const theta_new_address = movement_scratch
        .movement_theta_residual_newly_admitted.data();
    const float* const theta_discard_address = movement_scratch
        .movement_theta_residual_discard.data();
    const std::size_t theta_new_capacity = movement_scratch
        .movement_theta_residual_newly_admitted.capacity();
    const std::size_t theta_discard_capacity = movement_scratch
        .movement_theta_residual_discard.capacity();
    REQUIRE(theta_new_address != nullptr);
    REQUIRE(theta_discard_address != nullptr);

    NestedChildMovementReportV5 movement;
    REQUIRE_NOTHROW(move_nested_child_integer_parent_cells(
        storage, 1, 0, {4.0}, movement));
    REQUIRE(movement.schema_version == 5);
    REQUIRE(movement.microphysics.active);
    REQUIRE(movement.microphysics.theta_storage_residual_active);
    REQUIRE(movement.microphysics.private_scalar_count == 6);
    REQUIRE(movement.microphysics.newly_admitted_private_values == 96);
    REQUIRE(
        movement.microphysics.newly_admitted_theta_residual_values == 16);
    REQUIRE(child_level.runtime.microphysics.scheme.get() ==
            child_microphysics_identity);
    for (std::size_t species = 0; species < 6; ++species)
    {
        REQUIRE(movement_scratch
                    .movement_microphysics_newly_admitted[species].data() ==
                private_new_addresses[species]);
        REQUIRE(movement_scratch
                    .movement_microphysics_newly_admitted[species].capacity() ==
                private_new_capacities[species]);
        REQUIRE(movement_scratch
                    .movement_microphysics_discard[species].data() ==
                private_discard_addresses[species]);
        REQUIRE(movement_scratch
                    .movement_microphysics_discard[species].capacity() ==
                private_discard_capacities[species]);
    }
    REQUIRE(movement_scratch
                .movement_theta_residual_newly_admitted.data() ==
            theta_new_address);
    REQUIRE(movement_scratch
                .movement_theta_residual_newly_admitted.capacity() ==
            theta_new_capacity);
    REQUIRE(movement_scratch.movement_theta_residual_discard.data() ==
            theta_discard_address);
    REQUIRE(movement_scratch.movement_theta_residual_discard.capacity() ==
            theta_discard_capacity);
    const NestedExecutionStorage copied_nested =
        storage.nested_execution;
    REQUIRE(copied_nested.boundary_scratch
                .movement_primary_discard[0].capacity() == 0);
    REQUIRE(copied_nested.boundary_scratch
                .movement_microphysics_newly_admitted[0].capacity() == 0);
    REQUIRE(copied_nested.boundary_scratch
                .movement_microphysics_discard[0].capacity() == 0);
    REQUIRE(copied_nested.boundary_scratch
                .movement_theta_residual_discard.capacity() == 0);
    for (std::size_t species = 0; species < 6; ++species)
    {
        REQUIRE(movement.microphysics
                    .newly_admitted_target_inventory[species] ==
                movement.microphysics
                    .newly_admitted_committed_inventory[species]);
        REQUIRE(movement.microphysics
                    .newly_admitted_storage_rounding[species] == 0.0);
    }

    child_moments = child_level.runtime.microphysics.scheme
        ->prognostic_scalars_for_transport();
    parent_moments = storage.runtime.microphysics.scheme
        ->prognostic_scalars_for_transport();
    for (std::size_t species = 0; species < child_moments.size(); ++species)
        for (int ci = 0; ci < 4; ++ci)
            for (int cj = 0; cj < 4; ++cj)
                for (int k = 0; k < 2; ++k)
                {
                    const bool prescribed_boundary =
                        ci == 0 || ci == 3 || cj == 0 || cj == 3;
                    const float expected = ci < 2 && !prescribed_boundary
                        ? old_child_moments[species][1 + ci + 2][1 + cj][k]
                        : (*parent_moments[species].field)
                            [1 + 3 + ci / 2][1 + 2 + cj / 2][k];
                    REQUIRE((*child_moments[species].field)
                                [1 + ci][1 + cj][k] == expected);
                }
    for (int ci = 0; ci < 4; ++ci)
        for (int cj = 0; cj < 4; ++cj)
            for (int k = 0; k < 2; ++k)
            {
                const bool retained_interior =
                    ci < 2 && ci != 0 && cj != 0 && cj != 3;
                const float expected = retained_interior
                    ? old_theta_residual[1 + ci + 2][1 + cj][k]
                    : 0.0f;
                REQUIRE(child_level.runtime.microphysics
                            .theta_storage_residual[1 + ci][1 + cj][k] ==
                        expected);
            }
    REQUIRE(child_level.runtime.microphysics
                .theta_storage_residual[0][0][0] == 0.0f);
    REQUIRE(movement.boundary_source.microphysics_private.scalar_count == 6);
    for (std::size_t species = 0; species < 6; ++species)
    {
        REQUIRE(movement.boundary_source.microphysics_private
                    .forced_cells[species] == 24);
        REQUIRE(movement.boundary_source.microphysics_private
                    .carrier_storage_residual[species] ==
                movement.boundary_source.microphysics_private
                    .target_carrier_change[species] -
                    movement.boundary_source.microphysics_private
                        .committed_carrier_change[species]);
    }

    std::vector<std::uint8_t> moved_checkpoint;
    REQUIRE(tmv::checkpoint::serialize_simulation_storage(
        storage, moved_checkpoint, checkpoint_error));
    SimulationStorageState restored;
    const bool microphysics_restored_ok =
        tmv::checkpoint::deserialize_simulation_storage(
            moved_checkpoint, restored, checkpoint_error);
    INFO(checkpoint_error);
    REQUIRE(microphysics_restored_ok);
    REQUIRE(restored.nested_execution.boundary_scratch
                .movement_primary_discard[0].capacity() == 0);
    REQUIRE(restored.nested_execution.boundary_scratch
                .movement_microphysics_newly_admitted[0].capacity() == 0);
    REQUIRE(restored.nested_execution.boundary_scratch
                .movement_microphysics_discard[0].capacity() == 0);
    REQUIRE(restored.nested_execution.boundary_scratch
                .movement_theta_residual_discard.capacity() == 0);
    std::vector<std::uint8_t> restored_checkpoint;
    REQUIRE(tmv::checkpoint::serialize_simulation_storage(
        restored, restored_checkpoint, checkpoint_error));
    REQUIRE(restored_checkpoint == moved_checkpoint);
    REQUIRE_NOTHROW(require_nested_child_level_runtime_ready(storage));
}

TEST_CASE("moved active radiation continues through the production child",
          "[core][nesting][production][movement][radiation][continuation]")
{
    ProcessStateRestore restore;
    global_simulation_storage = make_parent();
    initialize_parent_modules();
    auto& storage = global_simulation_storage;
    RadiationConfig radiation_config;
    radiation_config.scheme_id = "simple_grey";
    radiation_config.do_lw = true;
    radiation_config.do_sw = false;
    radiation_config.dt_radiation = 300.0;
    storage.runtime.boundary_layer.surface_config.Tsfc = 300.0;
    storage.runtime.boundary_layer.surface_config.albedo = 0.2;
    storage.runtime.boundary_layer.surface_config.emissivity = 0.95;
    initialize_radiation(
        storage.runtime.radiation, "simple_grey", radiation_config);
    step_radiation(make_simulation_state_view(storage), 0.0);
    REQUIRE(storage.runtime.radiation.last_update_time_s == 0.0);
    REQUIRE(storage.runtime.radiation.held_atmospheric_power_w != 0.0);

    const PrognosticFieldStorage previous_parent = storage.fields;
    prepare_child(previous_parent);
    NestedChildSubstepReport first;
    NestedChildSubstepReport second;
    advance_nested_child_dynamics_substep(
        storage, previous_parent, {4.0}, first);
    advance_nested_child_dynamics_substep(
        storage, previous_parent, {4.0}, second);
    auto& child = storage.nested_execution.child;
    REQUIRE(child.runtime.radiation.last_update_time_s == 0.0);
    REQUIRE(child.runtime.radiation.held_atmospheric_power_w != 0.0);
    const double accumulated_expected =
        child.runtime.radiation.accumulated_expected_atmospheric_energy_j;
    const double accumulated_applied =
        child.runtime.radiation.accumulated_applied_radiation_heat_j;
    const std::uint64_t applied_steps =
        child.runtime.radiation.applied_step_count;
    REQUIRE(accumulated_expected != 0.0);
    REQUIRE(accumulated_applied != 0.0);
    REQUIRE(applied_steps > 0);

    NestedChildMovementReportV5 movement;
    REQUIRE_NOTHROW(move_nested_child_integer_parent_cells(
        storage, 1, 0, {4.0}, movement));
    REQUIRE(movement.radiation.active);
    REQUIRE(movement.radiation.sample_time_s == 0.0);
    REQUIRE(movement.radiation.newly_admitted_cells == 16);
    REQUIRE(std::isfinite(
        movement.radiation.newly_admitted_storage_rounding_w));
    REQUIRE(movement.radiation.committed_child_held_power_w ==
            child.runtime.radiation.held_atmospheric_power_w);
    REQUIRE(child.runtime.radiation.last_update_time_s == 0.0);
    REQUIRE(child.runtime.radiation.next_update_time_s == 300.0);
    REQUIRE(child.runtime.radiation.accumulated_expected_atmospheric_energy_j ==
            accumulated_expected);
    REQUIRE(child.runtime.radiation.accumulated_applied_radiation_heat_j ==
            accumulated_applied);
    REQUIRE(child.runtime.radiation.applied_step_count == applied_steps);
    REQUIRE_NOTHROW(require_nested_child_level_runtime_ready(storage));
    REQUIRE_NOTHROW(advance_next_balanced_parent_interval({4.0}));
    REQUIRE(child.time_s == storage.time_s);
    REQUIRE(child.runtime.radiation.accumulated_expected_atmospheric_energy_j !=
            accumulated_expected);
    REQUIRE(child.runtime.radiation.accumulated_applied_radiation_heat_j !=
            accumulated_applied);
    REQUIRE(child.runtime.radiation.applied_step_count > applied_steps);
    require_finite_primary(child.fields);
}

TEST_CASE("moved active boundary layer continues through the production child",
          "[core][nesting][production][movement][boundary_layer][continuation]")
{
    ProcessStateRestore restore;
    global_simulation_storage = make_parent();
    initialize_parent_modules();
    auto& storage = global_simulation_storage;
    BoundaryLayerConfig pbl_config;
    pbl_config.scheme_id = "slab";
    pbl_config.dt_pbl = 60.0;
    SurfaceConfig surface_config;
    surface_config.Tsfc = 305.0;
    surface_config.qsfc = 0.014;
    initialize_boundary_layer(
        storage.runtime.boundary_layer, "slab", pbl_config,
        surface_config);
    step_boundary_layer(make_simulation_state_view(storage), 0.0);
    REQUIRE(storage.runtime.boundary_layer.last_update_time_s == 0.0);
    REQUIRE(maximum_absolute_value(
                storage.runtime.boundary_layer.
                    held_theta_inventory_rate_kgk_s) > 0.0);

    const PrognosticFieldStorage previous_parent = storage.fields;
    prepare_child(previous_parent);
    NestedChildSubstepReport first;
    NestedChildSubstepReport second;
    advance_nested_child_dynamics_substep(
        storage, previous_parent, {4.0}, first);
    advance_nested_child_dynamics_substep(
        storage, previous_parent, {4.0}, second);
    auto& child = storage.nested_execution.child;
    auto& child_pbl = child.runtime.boundary_layer;
    REQUIRE(child_pbl.last_update_time_s == 0.0);
    REQUIRE(maximum_absolute_value(
                child_pbl.held_theta_inventory_rate_kgk_s) > 0.0);
    const double requested_work =
        child_pbl.cumulative_requested_momentum_work_j;
    const double committed_work =
        child_pbl.cumulative_committed_momentum_work_j;
    const std::uint64_t work_steps = child_pbl.momentum_work_step_count;
    REQUIRE(work_steps > 0);

    NestedChildMovementReportV5 movement;
    REQUIRE_NOTHROW(move_nested_child_integer_parent_cells(
        storage, 1, 0, {4.0}, movement));
    REQUIRE(movement.boundary_layer.active);
    REQUIRE(movement.boundary_layer.sample_time_s == 0.0);
    REQUIRE(movement.boundary_layer.newly_admitted_cells == 16);
    REQUIRE(std::isfinite(
        movement.boundary_layer.newly_admitted_x_force_storage_rounding_n));
    REQUIRE(std::isfinite(
        movement.boundary_layer.newly_admitted_y_force_storage_rounding_n));
    REQUIRE(std::isfinite(
        movement.boundary_layer.newly_admitted_theta_storage_rounding_kgk_s));
    REQUIRE(std::isfinite(
        movement.boundary_layer.newly_admitted_vapor_storage_rounding_kg_s));
    REQUIRE(child_pbl.last_update_time_s == 0.0);
    REQUIRE(child_pbl.next_update_time_s == 60.0);
    REQUIRE(child_pbl.cumulative_requested_momentum_work_j == requested_work);
    REQUIRE(child_pbl.cumulative_committed_momentum_work_j == committed_work);
    REQUIRE(child_pbl.momentum_work_step_count == work_steps);
    REQUIRE_NOTHROW(require_nested_child_level_runtime_ready(storage));
    REQUIRE_NOTHROW(advance_next_balanced_parent_interval({4.0}));
    REQUIRE(child.time_s == storage.time_s);
    REQUIRE(child_pbl.momentum_work_step_count > work_steps);
    REQUIRE(std::isfinite(
        child_pbl.cumulative_requested_momentum_work_j));
    REQUIRE(std::isfinite(
        child_pbl.cumulative_committed_momentum_work_j));
    REQUIRE(maximum_absolute_value(
                child_pbl.held_theta_inventory_rate_kgk_s) > 0.0);
    require_finite_primary(child.fields);
}

TEST_CASE("Decimal child substeps preserve the parent stochastic epoch and exact synchronized clock",
          "[core][nesting][production][chaos][clock]")
{
    const int           ratio = GENERATE(2, 3);
    ProcessStateRestore restore;
    global_simulation_storage = make_parent();
    initialize_parent_modules();
    auto& storage   = global_simulation_storage;
    storage.time_s  = .2;
    storage.grid.dt = .2;
    BoundaryLayerConfig pbl;
    pbl.scheme_id = "slab";
    initialize_boundary_layer(storage.runtime.boundary_layer, "slab", pbl, SurfaceConfig{});
    step_boundary_layer(make_simulation_state_view(storage), 0.0);
    tmv::chaos::ChaosConfig chaos;
    chaos.scheme_id           = "full_stochastic";
    chaos.dt_chaos            = .1;
    chaos.tau_t               = 20;
    chaos.Lx                  = 2000;
    chaos.Ly                  = 1800;
    chaos.taper_id            = "none";
    chaos.apply_to_tendencies = {"pbl"};
    chaos.alpha_tend["pbl"]   = .25;
    tmv::chaos::initialize_chaos(make_simulation_state_view(storage), chaos);
    tmv::chaos::apply_chaos_tendencies(make_simulation_state_view(storage));
    tmv::chaos::step_chaos_noise(make_simulation_state_view(storage), .2);
    const auto                                 previous = storage.fields;
    tmv::chaos::HorizontalStochasticStatePhase phase;
    std::string                                phase_error;
    REQUIRE(storage.runtime.chaos.scheme->horizontal_stochastic_state_phase(phase, phase_error));
    CAPTURE(phase.current_epoch,
            phase.initialized,
            storage.time_s,
            storage.runtime.chaos.noise_elapsed_s,
            storage.runtime.chaos.config.dt_chaos);
    prepare_child(previous, 6, 6, 2 * ratio, 2 * ratio, ratio);
    for (int interval = 0; interval < 30; ++interval)
    {
        CAPTURE(interval, ratio);
        for (int step = 0; step < ratio; ++step)
        {
            NestedChildSubstepReport report;
            REQUIRE_NOTHROW(advance_nested_child_dynamics_substep(storage, previous, {4}, report));
            REQUIRE_NOTHROW(require_nested_child_level_runtime_ready(storage));
        }
        REQUIRE(storage.nested_execution.child.time_s == storage.time_s);
        REQUIRE(storage.nested_execution.child.runtime.chaos.noise_elapsed_s == storage.runtime.chaos.noise_elapsed_s);
        if (interval != 29)
        {
            tmv::chaos::step_chaos_noise(make_simulation_state_view(storage), .2);
            storage.time_s += .2;
            roll_nested_parent_interval(storage);
        }
    }
    auto& child = storage.nested_execution.child.runtime.chaos;
    child.noise_elapsed_s += .01;
    const double before = child.noise_elapsed_s;
    REQUIRE_THROWS(synchronize_nested_child_chaos_clock(storage, storage.time_s));
    REQUIRE(child.noise_elapsed_s == before);
}

TEST_CASE("moving child preserves one Earth-addressed stochastic process",
          "[core][nesting][production][movement][chaos][restart]")
{
    ProcessStateRestore restore;
    global_simulation_storage = make_parent();
    initialize_parent_modules();
    auto& storage = global_simulation_storage;

    BoundaryLayerConfig pbl_config;
    pbl_config.scheme_id = "slab";
    pbl_config.dt_pbl    = 60.0;
    SurfaceConfig surface_config;
    surface_config.Tsfc = 305.0;
    surface_config.qsfc = 0.014;
    initialize_boundary_layer(
        storage.runtime.boundary_layer, "slab", pbl_config,
        surface_config);
    step_boundary_layer(make_simulation_state_view(storage), 0.0);

    tmv::chaos::ChaosConfig chaos_config;
    chaos_config.scheme_id = "full_stochastic";
    chaos_config.seed = 948273ULL;
    chaos_config.member_id = 7;
    chaos_config.dt_chaos = 1.0;
    chaos_config.tau_t = 20.0;
    chaos_config.Lx = 2000.0;
    chaos_config.Ly = 1800.0;
    chaos_config.taper_id = "none";
    chaos_config.apply_to_tendencies = {"pbl"};
    chaos_config.alpha_tend["pbl"] = 0.25;
    tmv::chaos::initialize_chaos(
        make_simulation_state_view(storage), chaos_config);
    tmv::chaos::apply_chaos_tendencies(
        make_simulation_state_view(storage));
    tmv::chaos::step_chaos_noise(
        make_simulation_state_view(storage), 2.0);
    tmv::chaos::apply_chaos_tendencies(
        make_simulation_state_view(storage));
    REQUIRE(storage.runtime.chaos.noise_elapsed_s == 0.0);

    const PrognosticFieldStorage previous_parent = storage.fields;
    prepare_child(previous_parent);
    auto& child = storage.nested_execution.child;
    tmv::chaos::HorizontalStochasticStatePhase initial_child_phase;
    std::string phase_error;
    REQUIRE(child.runtime.chaos.scheme->horizontal_stochastic_state_phase(
        initial_child_phase, phase_error));
    REQUIRE(initial_child_phase.current_epoch == 0);
    REQUIRE(child.runtime.chaos.noise_elapsed_s == 0.0);

    NestedChildSubstepReport first;
    NestedChildSubstepReport second;
    REQUIRE_NOTHROW(advance_nested_child_dynamics_substep(
        storage, previous_parent, {4.0}, first));
    REQUIRE_NOTHROW(advance_nested_child_dynamics_substep(
        storage, previous_parent, {4.0}, second));
    tmv::chaos::HorizontalStochasticStatePhase parent_phase;
    tmv::chaos::HorizontalStochasticStatePhase child_phase;
    REQUIRE(storage.runtime.chaos.scheme->horizontal_stochastic_state_phase(
        parent_phase, phase_error));
    REQUIRE(child.runtime.chaos.scheme->horizontal_stochastic_state_phase(
        child_phase, phase_error));
    REQUIRE(parent_phase.current_epoch == 2);
    REQUIRE(child_phase.current_epoch == parent_phase.current_epoch);
    REQUIRE(child.runtime.chaos.noise_elapsed_s ==
            storage.runtime.chaos.noise_elapsed_s);
    REQUIRE(child.runtime.chaos.pbl_base_tendencies_valid);
    REQUIRE(storage.runtime.chaos.pbl_base_tendencies_valid);

    auto expected_scheme = child.runtime.chaos.scheme->clone();
    REQUIRE(expected_scheme);
    REQUIRE(expected_scheme->translate_horizontal_stochastic_state(
        2, 0, child.runtime.chaos.config,
        child.process_grid_metrics, phase_error));
    const PortableSchemeState expected_portable =
        expected_scheme->portable_state();
    const Field3D old_child_cache = child.runtime.chaos.base_du_dt_pbl;
    const std::vector<float> old_child_surface =
        child.runtime.chaos.base_sfc_tau_u;

    NestedChildMovementReportV5 rejected;
    rejected.movement_sequence = 77;
    child.runtime.chaos.noise_elapsed_s = 0.25;
    const PortableSchemeState before_rejected_move =
        child.runtime.chaos.scheme->portable_state();
    REQUIRE_THROWS_AS(move_nested_child_integer_parent_cells(
        storage, 1, 0, {4.0}, rejected), std::invalid_argument);
    REQUIRE(rejected.movement_sequence == 77);
    REQUIRE(portable_state_is_bitwise_equal(
        before_rejected_move,
        child.runtime.chaos.scheme->portable_state()));
    child.runtime.chaos.noise_elapsed_s = 0.0;

    NestedChildMovementReportV5 rollback_rejection;
    REQUIRE_THROWS_AS(move_nested_child_integer_parent_cells(
        storage, 1, 0, {0.0}, rollback_rejection), std::invalid_argument);
    auto& chaos_translation_workspace = storage.nested_execution
        .boundary_scratch.movement_chaos_translation_workspace;
    REQUIRE(chaos_translation_workspace);
    auto* const chaos_translation_workspace_identity =
        chaos_translation_workspace.get();
    tmv::chaos::HorizontalStochasticStorageIdentity live_storage_before;
    tmv::chaos::HorizontalStochasticStorageIdentity
        workspace_storage_before;
    std::string storage_identity_error;
    REQUIRE(child.runtime.chaos.scheme
        ->horizontal_stochastic_storage_identity(
            live_storage_before, storage_identity_error));
    REQUIRE(chaos_translation_workspace
        ->horizontal_stochastic_storage_identity(
            workspace_storage_before, storage_identity_error));
    REQUIRE(live_storage_before.process_count == 3);
    REQUIRE(workspace_storage_before.process_count == 3);
    for (std::size_t process = 0; process < 3; ++process)
    {
        REQUIRE(live_storage_before.current_controls[process] != nullptr);
        REQUIRE(live_storage_before.previous_controls[process] != nullptr);
        REQUIRE(workspace_storage_before.current_controls[process] !=
                nullptr);
        REQUIRE(workspace_storage_before.previous_controls[process] !=
                nullptr);
        REQUIRE(workspace_storage_before.current_rendered[process] ==
                nullptr);
        REQUIRE(workspace_storage_before.previous_rendered[process] ==
                nullptr);
    }
    auto& chaos_cell_scratch = storage.nested_execution.boundary_scratch
        .movement_chaos_cache_cell_newly_admitted;
    auto& chaos_column_scratch = storage.nested_execution.boundary_scratch
        .movement_chaos_cache_column_newly_admitted;
    std::array<const float*, 5> chaos_cell_addresses{};
    std::array<std::size_t, 5> chaos_cell_capacities{};
    std::array<const float*, 6> chaos_column_addresses{};
    std::array<std::size_t, 6> chaos_column_capacities{};
    for (std::size_t field = 0; field < chaos_cell_scratch.size(); ++field)
    {
        REQUIRE_FALSE(chaos_cell_scratch[field].empty());
        chaos_cell_addresses[field] = chaos_cell_scratch[field].data();
        chaos_cell_capacities[field] = chaos_cell_scratch[field].capacity();
    }
    for (std::size_t field = 0; field < chaos_column_scratch.size(); ++field)
    {
        REQUIRE_FALSE(chaos_column_scratch[field].empty());
        chaos_column_addresses[field] = chaos_column_scratch[field].data();
        chaos_column_capacities[field] =
            chaos_column_scratch[field].capacity();
    }

    NestedChildMovementReportV5 movement;
    REQUIRE_NOTHROW(move_nested_child_integer_parent_cells(
        storage, 1, 0, {4.0}, movement));
    REQUIRE(chaos_translation_workspace.get() ==
            chaos_translation_workspace_identity);
    tmv::chaos::HorizontalStochasticStorageIdentity live_storage_after;
    tmv::chaos::HorizontalStochasticStorageIdentity workspace_storage_after;
    REQUIRE(child.runtime.chaos.scheme
        ->horizontal_stochastic_storage_identity(
            live_storage_after, storage_identity_error));
    REQUIRE(chaos_translation_workspace
        ->horizontal_stochastic_storage_identity(
            workspace_storage_after, storage_identity_error));
    for (std::size_t process = 0; process < 3; ++process)
    {
        REQUIRE(live_storage_after.current_controls[process] ==
                workspace_storage_before.current_controls[process]);
        REQUIRE(live_storage_after.previous_controls[process] ==
                workspace_storage_before.previous_controls[process]);
        REQUIRE(workspace_storage_after.current_controls[process] ==
                live_storage_before.current_controls[process]);
        REQUIRE(workspace_storage_after.previous_controls[process] ==
                live_storage_before.previous_controls[process]);
        REQUIRE(live_storage_after.current_control_capacities[process] ==
                workspace_storage_before
                    .current_control_capacities[process]);
        REQUIRE(live_storage_after.previous_control_capacities[process] ==
                workspace_storage_before
                    .previous_control_capacities[process]);
        REQUIRE(workspace_storage_after.current_control_capacities[process] ==
                live_storage_before.current_control_capacities[process]);
        REQUIRE(workspace_storage_after.previous_control_capacities[process] ==
                live_storage_before.previous_control_capacities[process]);
        REQUIRE(live_storage_after.current_rendered[process] ==
                live_storage_before.current_rendered[process]);
        REQUIRE(live_storage_after.previous_rendered[process] ==
                live_storage_before.previous_rendered[process]);
        REQUIRE(workspace_storage_after.current_rendered[process] == nullptr);
        REQUIRE(workspace_storage_after.previous_rendered[process] == nullptr);
    }
    const NestedExecutionStorage copied_chaos_nested =
        storage.nested_execution;
    REQUIRE_FALSE(copied_chaos_nested.boundary_scratch
        .movement_chaos_translation_workspace);
    REQUIRE(movement.chaos.active);
    REQUIRE(movement.chaos.pbl_base_cache_active);
    REQUIRE(movement.chaos.stochastic_epoch == 2);
    REQUIRE(movement.chaos.noise_elapsed_s == 0.0);
    REQUIRE(movement.chaos.newly_admitted_cache_columns == 8);
    REQUIRE(movement.chaos.newly_admitted_cache_cells == 16);
    for (std::size_t field = 0; field < chaos_cell_scratch.size(); ++field)
    {
        REQUIRE(chaos_cell_scratch[field].data() ==
                chaos_cell_addresses[field]);
        REQUIRE(chaos_cell_scratch[field].capacity() ==
                chaos_cell_capacities[field]);
    }
    for (std::size_t field = 0; field < chaos_column_scratch.size(); ++field)
    {
        REQUIRE(chaos_column_scratch[field].data() ==
                chaos_column_addresses[field]);
        REQUIRE(chaos_column_scratch[field].capacity() ==
                chaos_column_capacities[field]);
    }
    const NestedExecutionStorage copied_nested =
        storage.nested_execution;
    for (const auto& field : copied_nested.boundary_scratch
             .movement_chaos_cache_cell_newly_admitted)
        REQUIRE(field.capacity() == 0);
    for (const auto& field : copied_nested.boundary_scratch
             .movement_chaos_cache_column_newly_admitted)
        REQUIRE(field.capacity() == 0);
    REQUIRE(movement.chaos.newly_admitted_target_x_force_n -
                movement.chaos.newly_admitted_committed_x_force_n ==
            movement.chaos.newly_admitted_x_force_storage_rounding_n);
    REQUIRE(movement.chaos.newly_admitted_target_y_force_n -
                movement.chaos.newly_admitted_committed_y_force_n ==
            movement.chaos.newly_admitted_y_force_storage_rounding_n);
    REQUIRE(movement.chaos.
                newly_admitted_target_theta_inventory_rate_kgk_s -
                movement.chaos.
                    newly_admitted_committed_theta_inventory_rate_kgk_s ==
            movement.chaos.newly_admitted_theta_storage_rounding_kgk_s);
    REQUIRE(movement.chaos.newly_admitted_target_vapor_mass_rate_kg_s -
                movement.chaos.
                    newly_admitted_committed_vapor_mass_rate_kg_s ==
            movement.chaos.newly_admitted_vapor_storage_rounding_kg_s);
    REQUIRE(portable_state_is_bitwise_equal(
        expected_portable, child.runtime.chaos.scheme->portable_state()));
    REQUIRE(movement.chaos.earth_origin_i_cells ==
            child_phase.earth_origin_i_cells + 2);
    REQUIRE(movement.chaos.earth_origin_j_cells ==
            child_phase.earth_origin_j_cells);

    const auto& relationship = storage.nested_execution.relationship;
    const auto parent_physical = physical_scalar_domain(
        storage.grid.nr, storage.grid.nth, storage.grid.nz,
        StaggerType::CGrid);
    for (int ci = 0; ci < relationship.child_cells_x; ++ci)
        for (int cj = 0; cj < relationship.child_cells_y; ++cj)
        {
            const int i = ci + 1;
            const int j = cj + 1;
            const bool retained = ci + movement.child_shift_i <
                relationship.child_cells_x;
            const int pi = parent_physical.i_begin +
                relationship.child_origin_parent_cell_i +
                ci / relationship.parent_ratio_x;
            const int pj = parent_physical.j_begin +
                relationship.child_origin_parent_cell_j +
                cj / relationship.parent_ratio_y;
            const auto parent_column = static_cast<std::size_t>(pi) *
                storage.grid.nth + static_cast<std::size_t>(pj);
            const auto child_column = static_cast<std::size_t>(i) *
                child.grid.nth + static_cast<std::size_t>(j);
            const float expected_surface = retained
                ? old_child_surface[static_cast<std::size_t>(i + 2) *
                      child.grid.nth + static_cast<std::size_t>(j)]
                : storage.runtime.chaos.base_sfc_tau_u[parent_column];
            REQUIRE(child.runtime.chaos.base_sfc_tau_u[child_column] ==
                    expected_surface);
            for (int k = 0; k < relationship.levels_z; ++k)
            {
                const float expected_force = retained
                    ? old_child_cache[i + 2][j][k]
                    : static_cast<float>(
                        storage.runtime.chaos.base_du_dt_pbl[pi][pj][k] /
                        4.0f);
                REQUIRE(child.runtime.chaos.base_du_dt_pbl[i][j][k] ==
                        expected_force);
            }
        }

    std::string checkpoint_error;
    std::vector<std::uint8_t> moved_checkpoint;
    REQUIRE(tmv::checkpoint::serialize_simulation_storage(
        storage, moved_checkpoint, checkpoint_error));
    SimulationStorageState restored;
    REQUIRE(tmv::checkpoint::deserialize_simulation_storage(
        moved_checkpoint, restored, checkpoint_error));
    REQUIRE_FALSE(restored.nested_execution.boundary_scratch
        .movement_chaos_translation_workspace);
    for (const auto& field : restored.nested_execution.boundary_scratch
             .movement_chaos_cache_cell_newly_admitted)
        REQUIRE(field.capacity() == 0);
    for (const auto& field : restored.nested_execution.boundary_scratch
             .movement_chaos_cache_column_newly_admitted)
        REQUIRE(field.capacity() == 0);
    std::vector<std::uint8_t> restored_checkpoint;
    REQUIRE(tmv::checkpoint::serialize_simulation_storage(
        restored, restored_checkpoint, checkpoint_error));
    REQUIRE(restored_checkpoint == moved_checkpoint);

    const PrognosticFieldStorage next_previous_parent = storage.fields;
    tmv::chaos::step_chaos_noise(
        make_simulation_state_view(storage), 2.0);
    tmv::chaos::apply_chaos_tendencies(
        make_simulation_state_view(storage));
    storage.time_s += 2.0;
    storage.grid.dt = 2.0;
    REQUIRE_NOTHROW(roll_nested_parent_interval(storage));
    REQUIRE_NOTHROW(advance_nested_child_dynamics_substep(
        storage, next_previous_parent, {4.0}, first));
    REQUIRE_NOTHROW(advance_nested_child_dynamics_substep(
        storage, next_previous_parent, {4.0}, second));
    REQUIRE(child.time_s == storage.time_s);
    REQUIRE_NOTHROW(require_nested_child_level_runtime_ready(storage));
    REQUIRE(child.runtime.chaos.scheme->horizontal_stochastic_state_phase(
        child_phase, phase_error));
    REQUIRE(storage.runtime.chaos.scheme->horizontal_stochastic_state_phase(
        parent_phase, phase_error));
    REQUIRE(child_phase.current_epoch == parent_phase.current_epoch);
    REQUIRE(child.runtime.chaos.noise_elapsed_s ==
            storage.runtime.chaos.noise_elapsed_s);
    require_finite_primary(child.fields);
}

TEST_CASE("constant-velocity recycling commits exact private and primary native transactions",
          "[core][domain_recycling][production][metal][vulkan][device][D5]")
{
    const std::string scheme  = GENERATE("none", "thompson", "milbrandt");
    const std::string backend = GENERATE("metal", "vulkan");
    CAPTURE(scheme, backend);
    ProcessStateRestore restore;
    const auto          configure_private = [&]()
    {
        auto& state                             = global_simulation_storage;
        state.configuration.microphysics_scheme = scheme;
        state.runtime.microphysics.scheme =
            tmv::create_microphysics_scheme(scheme, state.configuration.microphysics_config);
        auto& physics = *state.runtime.microphysics.scheme;
        physics.prepare_prognostic_state(state.fields.rho,
                                         state.fields.qc,
                                         state.fields.qr,
                                         state.fields.qi,
                                         state.fields.qs,
                                         state.fields.qg,
                                         state.fields.qh);
        const auto values = physics.prognostic_scalars_for_transport();
        for (std::size_t f = 0; f < values.size(); ++f)
            for (int i = 0; i < state.grid.nr; ++i)
                for (int j = 0; j < state.grid.nth; ++j)
                    for (int k = 0; k < state.grid.nz; ++k)
                        (*values[f].field)(i, j, k) = float(1000 * (f + 1) + 10 * i + 5 * j + k);
    };
    const auto configure_motion = []
    {
        auto& storage                                   = global_simulation_storage;
        auto& configuration                             = storage.configuration;
        configuration.domain_motion_mode                = tmv::init::config::DomainMotionMode::ConstantVelocity;
        configuration.domain_frame.earth_origin_at_t0_m = {1250.0, -750.0};
        configuration.domain_frame.velocity_ms          = {12.0, -3.0};
        const std::size_t surface_size                  = static_cast<std::size_t>(storage.grid.nr) * storage.grid.nth;
        storage.surface.precipitation_rate_mm_h.assign(surface_size, 0.0f);
        storage.surface.accumulated_rainfall_mm.assign(surface_size, 0.0f);
        storage.surface.accumulated_rainfall_residual_mm.assign(surface_size, 0.0f);
    };
    const auto fields = [](const PrognosticFieldStorage& values)
    {
        return std::array<const Field3D*, 15>{&values.rho,
                                              &values.p,
                                              &values.u,
                                              &values.v,
                                              &values.w,
                                              &values.theta,
                                              &values.qv,
                                              &values.qc,
                                              &values.qr,
                                              &values.qi,
                                              &values.qs,
                                              &values.qg,
                                              &values.qh,
                                              &values.tracer,
                                              &values.tke};
    };

    std::vector<Field3D>                              device_private;
    PrognosticFieldStorage                            device_primary;
    tmv::domain::HomogeneousWindowShiftReport         device_report;
    tmv::domain::HorizontalDomainFrame                device_frame;
    tmv::diagnostics::StormContinuityState            device_continuity;
    tmv::domain::StormWindowMovementState             device_movement;
    std::vector<std::uint8_t>                         device_partition;
    tmv::domain::DomainRecyclingTransactionProvenance device_provenance;

    global_simulation_storage = make_active_physics_parent();
    configure_motion();
    if (!initialize_parent_modules(backend.c_str()))
    {
        WARN("physical accelerator unavailable; required qualification uses "
             "NIMBUS_GPU_REQUIRED=1");
        return;
    }
    {
        configure_private();
        auto& state = global_simulation_storage;
        REQUIRE(state.runtime.compute.active_kind ==
                (backend == "metal" ? ComputeBackendKind::Metal : ComputeBackendKind::Vulkan));
        REQUIRE_FALSE(state.runtime.compute.fallback_active);
        REQUIRE(state.runtime.compute.backend);
        REQUIRE(state.runtime.compute.backend->supports_homogeneous_domain_field_shift());
        const auto decision = seed_recycling_storm_authority(state);
        auto       prepared = tmv::domain::prepare_flat_homogeneous_domain_recycling(state, decision, {0.1, 0.1});
        REQUIRE(prepared->authoritative_backend == state.runtime.compute.active_kind);
        REQUIRE_FALSE(prepared->volume_bindings.empty());
        const std::size_t   volume_count         = prepared->volume_bindings.size();
        const std::uint64_t command_count_before = state.runtime.compute.backend->successful_dispatch_count();
        const std::uint64_t kernel_count_before  = state.runtime.compute.backend->successful_kernel_dispatch_count(
            nimbus::compute::KernelId::DomainRecyclingHomogeneousFieldShift);
        tmv::domain::append_flat_homogeneous_domain_recycling_provenance(*prepared,
                                                                         "metal-production-recycling-1",
                                                                         "generation-1",
                                                                         "npy_3d",
                                                                         "physical-metal-test-swath-v1",
                                                                         state.time_s,
                                                                         1,
                                                                         0,
                                                                         4096,
                                                                         UINT64_C(0xd5a11e7a1));
        device_report = tmv::domain::commit_flat_homogeneous_domain_recycling(*prepared);
        REQUIRE_NOTHROW(tmv::domain::verify_flat_homogeneous_domain_recycling_after_commit(*prepared));
        REQUIRE(device_report.volumes.fields_shifted == volume_count);
        REQUIRE(state.runtime.compute.backend->successful_dispatch_count() == command_count_before + 1);
        REQUIRE(state.runtime.compute.backend->successful_kernel_dispatch_count(
                    nimbus::compute::KernelId::DomainRecyclingHomogeneousFieldShift) ==
                kernel_count_before + volume_count);
        for (const auto& field : state.runtime.microphysics.scheme->prognostic_scalars_for_transport())
            device_private.push_back(*field.field);
        device_primary    = state.fields;
        device_frame      = state.configuration.domain_frame;
        device_continuity = state.runtime.storm_tracking;
        device_movement   = state.runtime.storm_window_movement;
        device_partition  = state.runtime.storm_selected_partition;
        device_provenance = state.runtime.domain_recycling_provenance;
    }

    global_simulation_storage = make_active_physics_parent();
    configure_motion();
    REQUIRE(initialize_parent_modules("cpu"));
    {
        configure_private();
        auto&      state    = global_simulation_storage;
        const auto decision = seed_recycling_storm_authority(state);
        auto       prepared = tmv::domain::prepare_flat_homogeneous_domain_recycling(state, decision, {0.1, 0.1});
        REQUIRE(prepared->authoritative_backend == ComputeBackendKind::Cpu);
        tmv::domain::append_flat_homogeneous_domain_recycling_provenance(*prepared,
                                                                         "metal-production-recycling-1",
                                                                         "generation-1",
                                                                         "npy_3d",
                                                                         "physical-metal-test-swath-v1",
                                                                         state.time_s,
                                                                         1,
                                                                         0,
                                                                         4096,
                                                                         UINT64_C(0xd5a11e7a1));
        const auto cpu_report = tmv::domain::commit_flat_homogeneous_domain_recycling(*prepared);
        REQUIRE_NOTHROW(tmv::domain::verify_flat_homogeneous_domain_recycling_after_commit(*prepared));

        REQUIRE(cpu_report.volumes.fields_shifted == device_report.volumes.fields_shifted);
        REQUIRE(cpu_report.volumes.retained_bytes == device_report.volumes.retained_bytes);
        REQUIRE(cpu_report.volumes.initialized_bytes == device_report.volumes.initialized_bytes);
        REQUIRE(cpu_report.scalars_committed == device_report.scalars_committed);
        REQUIRE(cpu_report.runtime_invalidations_committed == device_report.runtime_invalidations_committed);
        REQUIRE(cpu_report.runtime_commits_committed == device_report.runtime_commits_committed);

        const auto cpu_fields    = fields(state.fields);
        const auto device_fields = fields(device_primary);
        for (std::size_t index = 0; index < cpu_fields.size(); ++index)
        {
            REQUIRE(cpu_fields[index]->size() == device_fields[index]->size());
            REQUIRE(std::memcmp(cpu_fields[index]->data(),
                                device_fields[index]->data(),
                                cpu_fields[index]->size() * sizeof(float)) == 0);
        }
        REQUIRE(state.configuration.domain_frame.earth_origin_at_t0_m.x_m == device_frame.earth_origin_at_t0_m.x_m);
        REQUIRE(state.configuration.domain_frame.earth_origin_at_t0_m.y_m == device_frame.earth_origin_at_t0_m.y_m);
        REQUIRE(state.configuration.domain_frame.velocity_ms.u_ms == device_frame.velocity_ms.u_ms);
        REQUIRE(state.configuration.domain_frame.velocity_ms.v_ms == device_frame.velocity_ms.v_ms);
        REQUIRE(state.runtime.storm_tracking.samples[0].x_m == device_continuity.samples[0].x_m);
        REQUIRE(state.runtime.storm_tracking.samples[0].y_m == device_continuity.samples[0].y_m);
        REQUIRE(state.runtime.storm_tracking.samples[1].x_m == device_continuity.samples[1].x_m);
        REQUIRE(state.runtime.storm_tracking.samples[1].y_m == device_continuity.samples[1].y_m);
        REQUIRE(state.runtime.storm_window_movement.cumulative_i_shift_cells ==
                device_movement.cumulative_i_shift_cells);
        REQUIRE(state.runtime.storm_window_movement.cumulative_j_shift_cells ==
                device_movement.cumulative_j_shift_cells);
        REQUIRE(state.runtime.storm_selected_partition == device_partition);
        const auto private_fields = state.runtime.microphysics.scheme->prognostic_scalars_for_transport();
        REQUIRE(private_fields.size() == device_private.size());
        for (std::size_t f = 0; f < private_fields.size(); ++f)
            CHECK(std::memcmp(private_fields[f].field->data(),
                              device_private[f].data(),
                              device_private[f].size() * sizeof(float)) == 0);
        const auto& cpu_provenance = state.runtime.domain_recycling_provenance;
        REQUIRE(cpu_provenance.private_inventory_complete);
        REQUIRE(cpu_provenance.retiring_private_inventory.count == private_fields.size());
        for (std::size_t f = 0; f < private_fields.size(); ++f)
        {
            CHECK(cpu_provenance.retiring_private_inventory.entries[f].inventory ==
                  device_provenance.retiring_private_inventory.entries[f].inventory);
            CHECK(cpu_provenance.admitted_private_inventory.entries[f].inventory ==
                  device_provenance.admitted_private_inventory.entries[f].inventory);
        }
        REQUIRE(cpu_provenance.available);
        REQUIRE(cpu_provenance.sequence == device_provenance.sequence);
        REQUIRE(cpu_provenance.committed_chain_fnv1a64 == device_provenance.committed_chain_fnv1a64);
        REQUIRE(cpu_provenance.post_shift_origin_t0_m.x_m == device_provenance.post_shift_origin_t0_m.x_m);
        REQUIRE(cpu_provenance.post_shift_origin_t0_m.y_m == device_provenance.post_shift_origin_t0_m.y_m);
    }
}

TEST_CASE("strict CPU production bypasses the repairing fused acoustic writer",
          "[core][production][cpu][strict][acoustic][guard]")
{
    ProcessStateRestore restore;
    global_simulation_storage = make_acoustic_parent();
    REQUIRE(global_simulation_storage.configuration.validation_policy.mode ==
            tmv::GuardMode::Strict);
    REQUIRE(initialize_parent_modules("cpu"));
    REQUIRE(global_simulation_storage.runtime.compute.backend);
    ComputeBackend& backend =
        *global_simulation_storage.runtime.compute.backend;
    // The manifest tracks scientific stages rather than implementation-only
    // fusion entries, so the CPU fused path reports the pressure-stage ID.
    const auto kernel =
        nimbus::compute::KernelId::AcousticPressureCartesianCgrid;
    const std::uint64_t before =
        backend.successful_kernel_dispatch_count(kernel);

    advance_parent_production_step(0.5);

    REQUIRE(backend.successful_kernel_dispatch_count(kernel) == before);
    require_finite_primary(global_simulation_storage.fields);
}

TEST_CASE("traveling acoustic pulse crosses the stationary child interface",
          "[core][nesting][production][wave]")
{
    ProcessStateRestore restore;
    global_simulation_storage = make_acoustic_parent();
    initialize_parent_modules();
    set_step_diagnostics_enabled(false);

    PrognosticFieldStorage previous_parent =
        global_simulation_storage.fields;
    advance_parent_production_step(0.5);
    prepare_acoustic_child(previous_parent);
    AcousticInterfaceMetrics metrics;
    sample_acoustic_interface(metrics, &previous_parent);
    advance_acoustic_open_child_interval(previous_parent);

    sample_acoustic_interface(metrics);
    advance_acoustic_parent_intervals(13, metrics);

    std::string checkpoint_error;
    std::vector<std::uint8_t> midpoint_checkpoint;
    REQUIRE(tmv::checkpoint::serialize_simulation_storage(
        global_simulation_storage, midpoint_checkpoint,
        checkpoint_error));
    const AcousticInterfaceMetrics midpoint_metrics = metrics;

    advance_acoustic_parent_intervals(14, metrics);
    std::vector<std::uint8_t> continuous_checkpoint;
    REQUIRE(tmv::checkpoint::serialize_simulation_storage(
        global_simulation_storage, continuous_checkpoint,
        checkpoint_error));

    SimulationStorageState restarted;
    REQUIRE(tmv::checkpoint::deserialize_simulation_storage(
        midpoint_checkpoint, restarted, checkpoint_error));
    global_simulation_storage = std::move(restarted);
    tmv::refresh_grid_metrics_from_terrain();
    tmv::initialize_dynamics(
        global_simulation_storage.configuration.dynamics_scheme_name);
    rehydrate_nested_child_transient_operators(
        global_simulation_storage);
    std::string compute_error;
    REQUIRE(initialize_compute_backend_runtime(compute_error));
    AcousticInterfaceMetrics restarted_metrics = midpoint_metrics;
    advance_acoustic_parent_intervals(14, restarted_metrics);
    std::vector<std::uint8_t> restarted_checkpoint;
    REQUIRE(tmv::checkpoint::serialize_simulation_storage(
        global_simulation_storage, restarted_checkpoint,
        checkpoint_error));
    REQUIRE(restarted_checkpoint == continuous_checkpoint);
    require_acoustic_metrics_exact(restarted_metrics, metrics);

    CAPTURE(metrics.incident_pa,
            metrics.transmitted_pa,
            metrics.reflected_pa);
    REQUIRE(metrics.incident_pa > 1.0);
    const double transmission =
        metrics.transmitted_pa / metrics.incident_pa;
    const double reflection = metrics.reflected_pa / metrics.incident_pa;
    REQUIRE(metrics.incident_energy_j > 0.0);
    REQUIRE(metrics.parent_control_energy_j > 0.0);
    const double energy_transmission =
        metrics.transmitted_energy_j / metrics.parent_control_energy_j;
    const double energy_reflection =
        metrics.reflected_energy_j / metrics.parent_control_energy_j;
    const double energy_closure =
        energy_transmission + energy_reflection;
    CAPTURE(metrics.incident_energy_j,
            metrics.parent_control_energy_j,
            metrics.transmitted_energy_j,
            metrics.reflected_energy_j,
            energy_transmission,
            energy_reflection,
            energy_closure);
    REQUIRE(transmission > 0.9);
    REQUIRE(transmission < 1.1);
    REQUIRE(reflection < 0.05);
    REQUIRE(energy_transmission > 0.98);
    REQUIRE(energy_transmission < 1.02);
    REQUIRE(energy_reflection < 0.005);
    REQUIRE(energy_closure > 0.98);
    REQUIRE(energy_closure < 1.02);
    REQUIRE(global_simulation_storage.nested_execution.child.time_s ==
            global_simulation_storage.time_s);
    const auto& interface =
        global_simulation_storage.nested_execution.interface_source;
    require_complete_interface_ledger(interface, 57);
    require_finite_primary(
        global_simulation_storage.nested_execution.child.fields);
}

TEST_CASE("stationary one-way child executes counted physical Vulkan work",
          "[core][nesting][production][vulkan][device]")
{
    ProcessStateRestore restore;
    global_simulation_storage = make_acoustic_parent();
    if (!initialize_parent_modules("vulkan"))
    {
        WARN("physical Vulkan device unavailable; required qualification uses "
             "NIMBUS_GPU_REQUIRED=1");
        return;
    }
    auto& state = global_simulation_storage;
    REQUIRE(state.runtime.compute.active_kind == ComputeBackendKind::Vulkan);
    REQUIRE_FALSE(state.runtime.compute.fallback_active);
    REQUIRE(state.runtime.compute.backend);
    set_step_diagnostics_enabled(false);

    PrognosticFieldStorage previous_parent = state.fields;
    const std::uint64_t before_parent =
        state.runtime.compute.backend->successful_dispatch_count();
    advance_parent_production_step(0.5);
    REQUIRE(state.runtime.compute.backend->successful_dispatch_count() >
            before_parent);

    prepare_acoustic_child(previous_parent);
    const std::uint64_t before_child =
        state.runtime.compute.backend->successful_dispatch_count();
    advance_acoustic_open_child_interval(previous_parent);
    const std::uint64_t after_child =
        state.runtime.compute.backend->successful_dispatch_count();
    REQUIRE(after_child > before_child);
    REQUIRE(state.nested_execution.child.time_s == state.time_s);
    require_complete_interface_ledger(
        state.nested_execution.interface_source, 3);
    require_finite_primary(state.nested_execution.child.fields);

    previous_parent = state.fields;
    const std::uint64_t before_continuation =
        state.runtime.compute.backend->successful_dispatch_count();
    advance_parent_production_step(0.5);
    roll_nested_parent_interval(state);
    advance_acoustic_open_child_interval(previous_parent);
    REQUIRE(state.runtime.compute.backend->successful_dispatch_count() >
            before_continuation);
    REQUIRE(state.nested_execution.child.time_s == state.time_s);
    require_complete_interface_ledger(
        state.nested_execution.interface_source, 5);
    require_finite_primary(state.nested_execution.child.fields);

    std::string checkpoint_error;
    std::vector<std::uint8_t> checkpoint;
    REQUIRE(tmv::checkpoint::serialize_simulation_storage(
        state, checkpoint, checkpoint_error));
    REQUIRE_FALSE(checkpoint.empty());

    previous_parent = state.fields;
    advance_parent_production_step(0.5);
    roll_nested_parent_interval(state);
    advance_acoustic_open_child_interval(previous_parent);
    require_complete_interface_ledger(
        state.nested_execution.interface_source, 7);
    std::vector<std::uint8_t> continuous_checkpoint;
    REQUIRE(tmv::checkpoint::serialize_simulation_storage(
        state, continuous_checkpoint, checkpoint_error));

    SimulationStorageState restarted;
    REQUIRE(tmv::checkpoint::deserialize_simulation_storage(
        checkpoint, restarted, checkpoint_error));
    global_simulation_storage = std::move(restarted);
    tmv::refresh_grid_metrics_from_terrain();
    tmv::initialize_dynamics(
        global_simulation_storage.configuration.dynamics_scheme_name);
    rehydrate_nested_child_transient_operators(
        global_simulation_storage);
    std::string compute_error;
    REQUIRE(initialize_compute_backend_runtime(compute_error));
    auto& restarted_state = global_simulation_storage;
    REQUIRE(restarted_state.runtime.compute.active_kind ==
            ComputeBackendKind::Vulkan);
    REQUIRE_FALSE(restarted_state.runtime.compute.fallback_active);
    REQUIRE(restarted_state.runtime.compute.backend);
    previous_parent = restarted_state.fields;
    const std::uint64_t before_restarted_child =
        restarted_state.runtime.compute.backend->successful_dispatch_count();
    advance_parent_production_step(0.5);
    roll_nested_parent_interval(restarted_state);
    advance_acoustic_open_child_interval(previous_parent);
    REQUIRE(restarted_state.runtime.compute.backend->successful_dispatch_count() >
            before_restarted_child);
    require_complete_interface_ledger(
        restarted_state.nested_execution.interface_source, 7);
    std::vector<std::uint8_t> restarted_checkpoint;
    REQUIRE(tmv::checkpoint::serialize_simulation_storage(
        restarted_state, restarted_checkpoint, checkpoint_error));
    REQUIRE(restarted_checkpoint == continuous_checkpoint);

    const PrognosticFieldStorage device_parent = restarted_state.fields;
    const PrognosticFieldStorage device_child =
        restarted_state.nested_execution.child.fields;

    global_simulation_storage = make_acoustic_parent();
    REQUIRE(initialize_parent_modules("cpu"));
    set_step_diagnostics_enabled(false);
    previous_parent = global_simulation_storage.fields;
    advance_parent_production_step(0.5);
    prepare_acoustic_child(previous_parent);
    advance_acoustic_open_child_interval(previous_parent);
    previous_parent = global_simulation_storage.fields;
    advance_parent_production_step(0.5);
    roll_nested_parent_interval(global_simulation_storage);
    advance_acoustic_open_child_interval(previous_parent);
    previous_parent = global_simulation_storage.fields;
    advance_parent_production_step(0.5);
    roll_nested_parent_interval(global_simulation_storage);
    advance_acoustic_open_child_interval(previous_parent);

    const char* names[] = {
        "rho", "p", "u", "v", "w", "theta", "qv", "qc", "qr",
        "qi", "qs", "qg", "qh", "tracer", "tke"};
    const double absolute_floors[] = {
        1.0e-7, 1.0e-3, 1.0e-7, 1.0e-7, 1.0e-7, 1.0e-6,
        1.0e-9, 1.0e-9, 1.0e-9, 1.0e-9, 1.0e-9, 1.0e-9,
        1.0e-9, 1.0e-9, 1.0e-8};
    const auto fields = [](const PrognosticFieldStorage& values) {
        return std::array<const Field3D*, 15>{
            &values.rho, &values.p, &values.u, &values.v, &values.w,
            &values.theta, &values.qv, &values.qc, &values.qr,
            &values.qi, &values.qs, &values.qg, &values.qh,
            &values.tracer, &values.tke};
    };
    const auto cpu_parent_fields = fields(global_simulation_storage.fields);
    const auto cpu_child_fields = fields(
        global_simulation_storage.nested_execution.child.fields);
    const auto device_parent_fields = fields(device_parent);
    const auto device_child_fields = fields(device_child);
    for (std::size_t n = 0; n < cpu_parent_fields.size(); ++n)
    {
        require_nested_device_field_parity(
            names[n], *cpu_parent_fields[n], *device_parent_fields[n],
            absolute_floors[n]);
        require_nested_device_field_parity(
            names[n], *cpu_child_fields[n], *device_child_fields[n],
            absolute_floors[n]);
    }
    require_complete_interface_ledger(
        global_simulation_storage.nested_execution.interface_source, 7);
    require_finite_primary(
        global_simulation_storage.nested_execution.child.fields);
}

TEST_CASE("stationary nested Kessler and Smagorinsky physics qualify physical Vulkan parity",
          "[core][nesting][production][vulkan][device][physics]")
{
    ProcessStateRestore restore;
    global_simulation_storage = make_active_physics_parent();
    if (!initialize_parent_modules("vulkan"))
    {
        WARN("physical Vulkan device unavailable; required qualification uses "
             "NIMBUS_GPU_REQUIRED=1");
        return;
    }
    auto& state = global_simulation_storage;
    enable_kessler_smagorinsky_physics(state);
    REQUIRE(state.runtime.compute.active_kind == ComputeBackendKind::Vulkan);
    REQUIRE_FALSE(state.runtime.compute.fallback_active);
    REQUIRE(state.runtime.compute.backend);
    REQUIRE(state.runtime.compute.backend->supports_microphysics(
        MicrophysicsKernelOp::KesslerPointwise));
    REQUIRE(state.runtime.compute.backend->supports_microphysics(
        MicrophysicsKernelOp::KesslerSedimentation));
    REQUIRE(state.runtime.compute.backend->supports_turbulence(
        TurbulenceKernelOp::Smagorinsky));
    set_step_diagnostics_enabled(true);

    const std::uint64_t before_device =
        state.runtime.compute.backend->successful_dispatch_count();
    const ActivePhysicsKernelDispatchCounts before_physics_device =
        active_physics_kernel_dispatch_counts(
            *state.runtime.compute.backend);
    PrognosticFieldStorage previous_parent = state.fields;
    advance_parent_production_step(0.5);
    prepare_acoustic_child(previous_parent);
    advance_acoustic_open_child_interval(previous_parent);
    require_complete_interface_ledger(
        state.nested_execution.interface_source, 3);

    previous_parent = state.fields;
    advance_parent_production_step(0.5);
    roll_nested_parent_interval(state);
    advance_acoustic_open_child_interval(previous_parent);
    require_complete_interface_ledger(
        state.nested_execution.interface_source, 5);
    REQUIRE(state.runtime.compute.backend->successful_dispatch_count() >
            before_device);
    require_active_physics_kernel_dispatches(
        *state.runtime.compute.backend, before_physics_device);
    require_active_kessler_smagorinsky_provenance(
        state.runtime.microphysics, state.runtime.turbulence);
    require_active_kessler_smagorinsky_provenance(
        state.nested_execution.child.runtime.microphysics,
        state.nested_execution.child.runtime.turbulence);

    std::string checkpoint_error;
    std::vector<std::uint8_t> midpoint_checkpoint;
    REQUIRE(tmv::checkpoint::serialize_simulation_storage(
        state, midpoint_checkpoint, checkpoint_error));
    REQUIRE_FALSE(midpoint_checkpoint.empty());

    previous_parent = state.fields;
    advance_parent_production_step(0.5);
    roll_nested_parent_interval(state);
    advance_acoustic_open_child_interval(previous_parent);
    require_complete_interface_ledger(
        state.nested_execution.interface_source, 7);
    std::vector<std::uint8_t> continuous_checkpoint;
    REQUIRE(tmv::checkpoint::serialize_simulation_storage(
        state, continuous_checkpoint, checkpoint_error));

    SimulationStorageState restarted;
    REQUIRE(tmv::checkpoint::deserialize_simulation_storage(
        midpoint_checkpoint, restarted, checkpoint_error));
    global_simulation_storage = std::move(restarted);
    tmv::refresh_grid_metrics_from_terrain();
    tmv::initialize_dynamics(
        global_simulation_storage.configuration.dynamics_scheme_name);
    rehydrate_nested_child_transient_operators(
        global_simulation_storage);
    std::string compute_error;
    REQUIRE(initialize_compute_backend_runtime(compute_error));
    auto& restarted_state = global_simulation_storage;
    REQUIRE(restarted_state.runtime.compute.active_kind ==
            ComputeBackendKind::Vulkan);
    REQUIRE_FALSE(restarted_state.runtime.compute.fallback_active);
    REQUIRE(restarted_state.runtime.compute.backend);
    const std::uint64_t before_restart_device =
        restarted_state.runtime.compute.backend->successful_dispatch_count();
    const ActivePhysicsKernelDispatchCounts before_restart_physics_device =
        active_physics_kernel_dispatch_counts(
            *restarted_state.runtime.compute.backend);
    previous_parent = restarted_state.fields;
    advance_parent_production_step(0.5);
    roll_nested_parent_interval(restarted_state);
    advance_acoustic_open_child_interval(previous_parent);
    REQUIRE(restarted_state.runtime.compute.backend->successful_dispatch_count() >
            before_restart_device);
    require_active_physics_kernel_dispatches(
        *restarted_state.runtime.compute.backend,
        before_restart_physics_device);
    require_complete_interface_ledger(
        restarted_state.nested_execution.interface_source, 7);
    require_active_kessler_smagorinsky_provenance(
        restarted_state.runtime.microphysics,
        restarted_state.runtime.turbulence);
    require_active_kessler_smagorinsky_provenance(
        restarted_state.nested_execution.child.runtime.microphysics,
        restarted_state.nested_execution.child.runtime.turbulence);
    std::vector<std::uint8_t> restarted_checkpoint;
    REQUIRE(tmv::checkpoint::serialize_simulation_storage(
        restarted_state, restarted_checkpoint, checkpoint_error));
    REQUIRE(restarted_checkpoint == continuous_checkpoint);

    const PrognosticFieldStorage device_parent = restarted_state.fields;
    const PrognosticFieldStorage device_child =
        restarted_state.nested_execution.child.fields;
    const SurfaceDiagnosticFieldStorage device_parent_surface =
        restarted_state.surface;
    const SurfaceDiagnosticFieldStorage device_child_surface =
        restarted_state.nested_execution.child.surface;

    global_simulation_storage = make_active_physics_parent();
    REQUIRE(initialize_parent_modules("cpu"));
    enable_kessler_smagorinsky_physics(global_simulation_storage);
    set_step_diagnostics_enabled(true);
    previous_parent = global_simulation_storage.fields;
    advance_parent_production_step(0.5);
    prepare_acoustic_child(previous_parent);
    advance_acoustic_open_child_interval(previous_parent);
    for (int interval = 0; interval < 2; ++interval)
    {
        previous_parent = global_simulation_storage.fields;
        advance_parent_production_step(0.5);
        roll_nested_parent_interval(global_simulation_storage);
        advance_acoustic_open_child_interval(previous_parent);
    }

    const auto fields = [](const PrognosticFieldStorage& values) {
        return std::array<const Field3D*, 15>{
            &values.rho, &values.p, &values.u, &values.v, &values.w,
            &values.theta, &values.qv, &values.qc, &values.qr,
            &values.qi, &values.qs, &values.qg, &values.qh,
            &values.tracer, &values.tke};
    };
    const char* names[] = {
        "rho", "p", "u", "v", "w", "theta", "qv", "qc", "qr",
        "qi", "qs", "qg", "qh", "tracer", "tke"};
    const double absolute_floors[] = {
        1.0e-7, 1.0e-3, 1.0e-7, 1.0e-7, 1.0e-7, 1.0e-6,
        1.0e-8, 1.0e-8, 1.0e-8, 1.0e-9, 1.0e-9, 1.0e-9,
        1.0e-9, 1.0e-9, 1.0e-8};
    const auto cpu_parent_fields = fields(global_simulation_storage.fields);
    const auto cpu_child_fields = fields(
        global_simulation_storage.nested_execution.child.fields);
    const auto device_parent_fields = fields(device_parent);
    const auto device_child_fields = fields(device_child);
    for (std::size_t n = 0; n < cpu_parent_fields.size(); ++n)
    {
        require_nested_device_field_parity(
            names[n], *cpu_parent_fields[n], *device_parent_fields[n],
            absolute_floors[n]);
        require_nested_device_field_parity(
            names[n], *cpu_child_fields[n], *device_child_fields[n],
            absolute_floors[n]);
    }
    require_nested_device_vector_parity(
        "parent precipitation rate",
        global_simulation_storage.surface.precipitation_rate_mm_h,
        device_parent_surface.precipitation_rate_mm_h, 1.0e-6);
    require_nested_device_vector_parity(
        "parent accumulated rainfall",
        global_simulation_storage.surface.accumulated_rainfall_mm,
        device_parent_surface.accumulated_rainfall_mm, 1.0e-7);
    require_nested_device_vector_parity(
        "child precipitation rate",
        global_simulation_storage.nested_execution.child.surface
            .precipitation_rate_mm_h,
        device_child_surface.precipitation_rate_mm_h, 1.0e-6);
    require_nested_device_vector_parity(
        "child accumulated rainfall",
        global_simulation_storage.nested_execution.child.surface
            .accumulated_rainfall_mm,
        device_child_surface.accumulated_rainfall_mm, 1.0e-7);
    require_complete_interface_ledger(
        global_simulation_storage.nested_execution.interface_source, 7);
    require_active_kessler_smagorinsky_provenance(
        global_simulation_storage.runtime.microphysics,
        global_simulation_storage.runtime.turbulence);
    require_active_kessler_smagorinsky_provenance(
        global_simulation_storage.nested_execution.child.runtime.microphysics,
        global_simulation_storage.nested_execution.child.runtime.turbulence);
    require_finite_primary(global_simulation_storage.fields);
    require_finite_primary(
        global_simulation_storage.nested_execution.child.fields);
}

void require_synchronized_child_relocation_accelerator(
    const char* backend_name, ComputeBackendKind expected_kind)
{
    ProcessStateRestore restore;

    const auto fields = [](const PrognosticFieldStorage& values) {
        return std::array<const Field3D*, 15>{
            &values.rho, &values.p, &values.u, &values.v, &values.w,
            &values.theta, &values.qv, &values.qc, &values.qr,
            &values.qi, &values.qs, &values.qg, &values.qh,
            &values.tracer, &values.tke};
    };
    const auto envelope_at_east_edge = [](const SimulationStorageState& state) {
        const auto& nested = state.nested_execution;
        const auto physical = physical_scalar_domain(
            nested.child.grid.nr, nested.child.grid.nth,
            nested.child.grid.nz, StaggerType::CGrid);
        const auto origin = nested.child.domain_frame.earth_origin_at(
            state.time_s);
        const double east_m = origin.x_m +
            static_cast<double>(physical.i_end) * nested.child.grid.dx;
        const double south_m = origin.y_m +
            static_cast<double>(physical.j_begin + 1) *
                nested.child.grid.dy;
        NestedTrackedEnvelopeV1 envelope;
        envelope.sample_time_s = state.time_s;
        envelope.selected_feature_available = true;
        envelope.continuity_authoritative = true;
        envelope.motion_available = true;
        envelope.confidence = 0.99;
        envelope.envelope_west_earth_m = east_m - 300.0;
        envelope.envelope_east_earth_m = east_m - 100.0;
        envelope.envelope_south_earth_m = south_m;
        envelope.envelope_north_earth_m = south_m + 250.0;
        envelope.velocity_x_earth_ms = 12.0;
        return envelope;
    };
    const tmv::domain::StormWindowMovementPolicy movement_policy{
        500.0, 600.0, 500.0, 500.0, 1, 0.0, 0.9, 2};
    const auto advance_interval = [] {
        auto& state = global_simulation_storage;
        PrognosticFieldStorage previous_parent = state.fields;
        advance_parent_production_step(0.5);
        roll_nested_parent_interval(state);
        advance_acoustic_open_child_interval(previous_parent);
    };
    const auto prepare_first_interval = [] {
        auto& state = global_simulation_storage;
        PrognosticFieldStorage previous_parent = state.fields;
        advance_parent_production_step(0.5);
        prepare_acoustic_child(previous_parent);
        advance_acoustic_open_child_interval(previous_parent);
    };
    const auto move_for_containment = [&] {
        auto& state = global_simulation_storage;
        NestedChildMovementAuthorityReportV1 committed;
        evaluate_and_move_nested_child_from_tracker(
            state, envelope_at_east_edge(state), movement_policy,
            {2.0}, committed);
        REQUIRE(committed.decision.reason ==
                tmv::domain::StormWindowMovementReason::EmergencyContainment);
        REQUIRE(committed.decision.confirmation_count == 1);
        REQUIRE(committed.movement_committed);
        REQUIRE(committed.movement.parent_shift_i == 1);
        // The synthetic envelope is only one child cell from the south
        // physical edge.  The same emergency transaction must therefore
        // restore the orthogonal release box as well as the east face.
        REQUIRE(committed.movement.parent_shift_j == -1);
        REQUIRE(committed.movement.child_shift_i == 2);
        REQUIRE(committed.movement.child_shift_j == -2);
        REQUIRE(committed.movement.movement_sequence == 1);
        REQUIRE(state.nested_execution.movement_sequence == 1);
        REQUIRE(state.nested_execution.cumulative_parent_shift_i == 1);
        REQUIRE(state.nested_execution.cumulative_parent_shift_j == -1);
        REQUIRE(state.runtime.storm_window_movement
                    .cumulative_i_shift_cells == 1);
        REQUIRE(state.runtime.storm_window_movement
                    .cumulative_j_shift_cells == -1);
        require_complete_interface_ledger(
            state.nested_execution.interface_source, 4);
        require_finite_primary(state.nested_execution.child.fields);
    };

    global_simulation_storage = make_active_physics_parent();
    if (!initialize_parent_modules(backend_name))
    {
        WARN("physical accelerator unavailable; required qualification uses "
             "NIMBUS_GPU_REQUIRED=1");
        return;
    }
    enable_kessler_smagorinsky_physics(global_simulation_storage);
    set_step_diagnostics_enabled(true);
    prepare_first_interval();
    move_for_containment();

    auto& device_state = global_simulation_storage;
    REQUIRE(device_state.runtime.compute.active_kind == expected_kind);
    REQUIRE_FALSE(device_state.runtime.compute.fallback_active);
    REQUIRE(device_state.runtime.compute.backend);
    std::string checkpoint_error;
    std::vector<std::uint8_t> moved_checkpoint;
    REQUIRE(tmv::checkpoint::serialize_simulation_storage(
        device_state, moved_checkpoint, checkpoint_error));
    REQUIRE_FALSE(moved_checkpoint.empty());

    const auto before_post_move = active_physics_kernel_dispatch_counts(
        *device_state.runtime.compute.backend);
    advance_interval();
    require_active_physics_kernel_dispatches(
        *device_state.runtime.compute.backend, before_post_move);
    require_complete_interface_ledger(
        device_state.nested_execution.interface_source, 6);
    std::vector<std::uint8_t> continuous_checkpoint;
    REQUIRE(tmv::checkpoint::serialize_simulation_storage(
        device_state, continuous_checkpoint, checkpoint_error));

    SimulationStorageState restarted;
    REQUIRE(tmv::checkpoint::deserialize_simulation_storage(
        moved_checkpoint, restarted, checkpoint_error));
    global_simulation_storage = std::move(restarted);
    tmv::refresh_grid_metrics_from_terrain();
    tmv::initialize_dynamics(
        global_simulation_storage.configuration.dynamics_scheme_name);
    rehydrate_nested_child_transient_operators(
        global_simulation_storage);
    std::string compute_error;
    REQUIRE(initialize_compute_backend_runtime(compute_error));
    auto& restarted_state = global_simulation_storage;
    REQUIRE(restarted_state.runtime.compute.active_kind == expected_kind);
    REQUIRE_FALSE(restarted_state.runtime.compute.fallback_active);
    REQUIRE(restarted_state.runtime.compute.backend);
    const auto before_restarted = active_physics_kernel_dispatch_counts(
        *restarted_state.runtime.compute.backend);
    advance_interval();
    require_active_physics_kernel_dispatches(
        *restarted_state.runtime.compute.backend, before_restarted);
    REQUIRE(restarted_state.nested_execution.movement_sequence == 1);
    REQUIRE(restarted_state.nested_execution.cumulative_parent_shift_i == 1);
    REQUIRE(restarted_state.nested_execution.cumulative_parent_shift_j == -1);
    require_complete_interface_ledger(
        restarted_state.nested_execution.interface_source, 6);
    std::vector<std::uint8_t> restarted_checkpoint;
    REQUIRE(tmv::checkpoint::serialize_simulation_storage(
        restarted_state, restarted_checkpoint, checkpoint_error));
    REQUIRE(restarted_checkpoint == continuous_checkpoint);

    const PrognosticFieldStorage device_parent = restarted_state.fields;
    const PrognosticFieldStorage device_child =
        restarted_state.nested_execution.child.fields;
    const SurfaceDiagnosticFieldStorage device_parent_surface =
        restarted_state.surface;
    const SurfaceDiagnosticFieldStorage device_child_surface =
        restarted_state.nested_execution.child.surface;

    global_simulation_storage = make_active_physics_parent();
    REQUIRE(initialize_parent_modules("cpu"));
    enable_kessler_smagorinsky_physics(global_simulation_storage);
    set_step_diagnostics_enabled(true);
    prepare_first_interval();
    move_for_containment();
    advance_interval();

    const char* names[] = {
        "rho", "p", "u", "v", "w", "theta", "qv", "qc", "qr",
        "qi", "qs", "qg", "qh", "tracer", "tke"};
    const double absolute_floors[] = {
        1.0e-7, 1.0e-3, 1.0e-7, 1.0e-7, 1.0e-7, 1.0e-6,
        1.0e-8, 1.0e-8, 1.0e-8, 1.0e-9, 1.0e-9, 1.0e-9,
        1.0e-9, 1.0e-9, 1.0e-8};
    const auto cpu_parent_fields = fields(global_simulation_storage.fields);
    const auto cpu_child_fields = fields(
        global_simulation_storage.nested_execution.child.fields);
    const auto device_parent_fields = fields(device_parent);
    const auto device_child_fields = fields(device_child);
    for (std::size_t index = 0; index < cpu_parent_fields.size(); ++index)
    {
        require_nested_device_field_parity(
            names[index], *cpu_parent_fields[index],
            *device_parent_fields[index], absolute_floors[index]);
        require_nested_device_field_parity(
            names[index], *cpu_child_fields[index],
            *device_child_fields[index], absolute_floors[index]);
    }
    require_nested_device_vector_parity(
        "parent precipitation rate",
        global_simulation_storage.surface.precipitation_rate_mm_h,
        device_parent_surface.precipitation_rate_mm_h, 1.0e-6);
    require_nested_device_vector_parity(
        "parent accumulated rainfall",
        global_simulation_storage.surface.accumulated_rainfall_mm,
        device_parent_surface.accumulated_rainfall_mm, 1.0e-7);
    require_nested_device_vector_parity(
        "child precipitation rate",
        global_simulation_storage.nested_execution.child.surface
            .precipitation_rate_mm_h,
        device_child_surface.precipitation_rate_mm_h, 1.0e-6);
    require_nested_device_vector_parity(
        "child accumulated rainfall",
        global_simulation_storage.nested_execution.child.surface
            .accumulated_rainfall_mm,
        device_child_surface.accumulated_rainfall_mm, 1.0e-7);
    REQUIRE(global_simulation_storage.nested_execution.movement_sequence == 1);
    REQUIRE(global_simulation_storage.nested_execution
                .cumulative_parent_shift_i == 1);
    REQUIRE(global_simulation_storage.nested_execution
                .cumulative_parent_shift_j == -1);
    require_complete_interface_ledger(
        global_simulation_storage.nested_execution.interface_source, 6);
    require_finite_primary(global_simulation_storage.fields);
    require_finite_primary(
        global_simulation_storage.nested_execution.child.fields);
}

TEST_CASE("synchronized child relocation resumes exact physical Vulkan work",
          "[core][nesting][production][movement][tracker][vulkan][device]")
{
    require_synchronized_child_relocation_accelerator(
        "vulkan", ComputeBackendKind::Vulkan);
}

TEST_CASE("synchronized child relocation resumes exact physical Metal work",
          "[core][nesting][production][movement][tracker][metal][device]")
{
    require_synchronized_child_relocation_accelerator(
        "metal", ComputeBackendKind::Metal);
}

TEST_CASE("stationary child transmits a physical Vulkan acoustic wave",
          "[core][nesting][production][vulkan][device][wave]")
{
    ProcessStateRestore restore;
    const AcousticCaseSpec spec{
        "physical_vulkan_normal_ratio2", 10, 2, 2, 4, 0.0, 600.0, 32};
    AcousticInterfaceMetrics metrics;
    if (!begin_acoustic_campaign(spec, metrics, "vulkan"))
    {
        WARN("physical Vulkan device unavailable; required qualification uses "
             "NIMBUS_GPU_REQUIRED=1");
        return;
    }
    auto& state = global_simulation_storage;
    REQUIRE(state.runtime.compute.active_kind == ComputeBackendKind::Vulkan);
    REQUIRE_FALSE(state.runtime.compute.fallback_active);
    REQUIRE(state.runtime.compute.backend);
    const std::uint64_t before_wave =
        state.runtime.compute.backend->successful_dispatch_count();
    advance_acoustic_parent_intervals(
        spec.total_parent_intervals - 1, metrics);
    REQUIRE(state.runtime.compute.backend->successful_dispatch_count() >
            before_wave);

    REQUIRE(metrics.parent_control_energy_j > 0.0);
    REQUIRE(metrics.parent_control_energy_j >
            metrics.parent_control_left_energy_j);
    const double parent_net_energy_j =
        metrics.parent_control_energy_j -
        metrics.parent_control_left_energy_j;
    const double child_net_energy_j =
        metrics.transmitted_energy_j - metrics.reflected_energy_j;
    const double right_transmission =
        metrics.transmitted_energy_j / metrics.parent_control_energy_j;
    const double net_transmission =
        child_net_energy_j / parent_net_energy_j;
    const double excess_left_fraction = std::abs(
        metrics.reflected_energy_j -
        metrics.parent_control_left_energy_j) /
        metrics.parent_control_energy_j;
    CAPTURE(right_transmission,
            net_transmission,
            excess_left_fraction);
    REQUIRE(right_transmission > 0.97);
    REQUIRE(right_transmission < 1.03);
    REQUIRE(net_transmission > 0.97);
    REQUIRE(net_transmission < 1.03);
    REQUIRE(excess_left_fraction < 0.005);
    REQUIRE(state.time_s == 0.5 * spec.total_parent_intervals);
    REQUIRE(state.nested_execution.child.time_s == state.time_s);
    require_complete_interface_ledger(
        state.nested_execution.interface_source,
        1 + static_cast<std::uint64_t>(spec.total_parent_intervals) *
            static_cast<std::uint64_t>(spec.refinement_ratio));
    require_finite_primary(state.nested_execution.child.fields);
}

TEST_CASE("production acoustic interface closes its wave and refinement matrix",
          "[core][nesting][production][wave][matrix]")
{
    ProcessStateRestore restore;
    const std::vector<AcousticCaseSpec> cases = {
        {"normal_ratio2_sigma600_matrix", 10, 2, 2, 4, 0.0, 600.0, 32},
        {"normal_ratio2_sigma1000", 10, 2, 2, 4, 0.0, 1000.0, 40},
        {"normal_ratio3_sigma600", 10, 3, 2, 4, 0.0, 600.0, 32},
        {"oblique_plus20_ratio2", 34, 2, 4, 24, 20.0, 600.0, 44},
        {"oblique_minus20_ratio2", 34, 2, 4, 24, -20.0, 600.0, 44}};
    std::vector<double> net_transmissions;
    std::vector<double> excess_left_fractions;

    for (const auto& spec : cases)
    {
        CAPTURE(spec.name,
                spec.refinement_ratio,
                spec.angle_degrees,
                spec.sigma_m);
        AcousticInterfaceMetrics metrics;
        begin_acoustic_campaign(spec, metrics);
        const int midpoint_intervals = spec.total_parent_intervals / 2;
        REQUIRE(midpoint_intervals >= 1);
        advance_acoustic_parent_intervals(midpoint_intervals - 1, metrics);

        std::string checkpoint_error;
        std::vector<std::uint8_t> midpoint_checkpoint;
        REQUIRE(tmv::checkpoint::serialize_simulation_storage(
            global_simulation_storage, midpoint_checkpoint,
            checkpoint_error));
        const AcousticInterfaceMetrics midpoint_metrics = metrics;
        const int remaining_intervals =
            spec.total_parent_intervals - midpoint_intervals;
        advance_acoustic_parent_intervals(remaining_intervals, metrics);
        std::vector<std::uint8_t> continuous_checkpoint;
        REQUIRE(tmv::checkpoint::serialize_simulation_storage(
            global_simulation_storage, continuous_checkpoint,
            checkpoint_error));

        SimulationStorageState restarted;
        REQUIRE(tmv::checkpoint::deserialize_simulation_storage(
            midpoint_checkpoint, restarted, checkpoint_error));
        global_simulation_storage = std::move(restarted);
        tmv::refresh_grid_metrics_from_terrain();
        tmv::initialize_dynamics(
            global_simulation_storage.configuration.dynamics_scheme_name);
        rehydrate_nested_child_transient_operators(
            global_simulation_storage);
        std::string compute_error;
        REQUIRE(initialize_compute_backend_runtime(compute_error));
        AcousticInterfaceMetrics restarted_metrics = midpoint_metrics;
        advance_acoustic_parent_intervals(
            remaining_intervals, restarted_metrics);
        std::vector<std::uint8_t> restarted_checkpoint;
        REQUIRE(tmv::checkpoint::serialize_simulation_storage(
            global_simulation_storage, restarted_checkpoint,
            checkpoint_error));
        REQUIRE(restarted_checkpoint == continuous_checkpoint);
        require_acoustic_metrics_exact(restarted_metrics, metrics);

        REQUIRE(metrics.parent_control_energy_j > 0.0);
        REQUIRE(metrics.parent_control_energy_j >
                metrics.parent_control_left_energy_j);
        const double parent_net_energy_j =
            metrics.parent_control_energy_j -
            metrics.parent_control_left_energy_j;
        const double child_net_energy_j =
            metrics.transmitted_energy_j - metrics.reflected_energy_j;
        const double right_transmission =
            metrics.transmitted_energy_j /
            metrics.parent_control_energy_j;
        const double net_transmission =
            child_net_energy_j / parent_net_energy_j;
        const double excess_left_fraction = std::abs(
            metrics.reflected_energy_j -
            metrics.parent_control_left_energy_j) /
            metrics.parent_control_energy_j;
        CAPTURE(right_transmission,
                net_transmission,
                excess_left_fraction);
        REQUIRE(right_transmission > 0.97);
        REQUIRE(right_transmission < 1.03);
        REQUIRE(net_transmission > 0.97);
        REQUIRE(net_transmission < 1.03);
        REQUIRE(excess_left_fraction < 0.005);
        net_transmissions.push_back(net_transmission);
        excess_left_fractions.push_back(excess_left_fraction);
        REQUIRE(global_simulation_storage.time_s ==
                0.5 * spec.total_parent_intervals);
        REQUIRE(global_simulation_storage.nested_execution.child.time_s ==
                global_simulation_storage.time_s);
        require_complete_interface_ledger(
            global_simulation_storage.nested_execution.interface_source,
            1 + static_cast<std::uint64_t>(spec.total_parent_intervals) *
                static_cast<std::uint64_t>(spec.refinement_ratio));
        require_finite_primary(
            global_simulation_storage.nested_execution.child.fields);
    }
    REQUIRE(net_transmissions.size() == cases.size());
    REQUIRE(excess_left_fractions.size() == cases.size());
    REQUIRE(std::abs(net_transmissions[0] - net_transmissions[2]) < 0.03);
    REQUIRE(std::abs(net_transmissions[3] - net_transmissions[4]) < 0.001);
    REQUIRE(std::abs(excess_left_fractions[3] -
                     excess_left_fractions[4]) < 0.0001);
}

#ifdef NIMBUS_NESTED_RELOCATION_BENCHMARK

TEST_CASE("warm production nested relocation benchmark",
          "[benchmark][nesting][relocation]")
{
    ProcessStateRestore restore;
    const int child_nx = benchmark_environment_integer(
        "NIMBUS_NESTED_BENCH_CHILD_NX", 128);
    const int child_ny = benchmark_environment_integer(
        "NIMBUS_NESTED_BENCH_CHILD_NY", 128);
    const int nz = benchmark_environment_integer(
        "NIMBUS_NESTED_BENCH_NZ", 64);
    const int warmup_moves = benchmark_environment_integer(
        "NIMBUS_NESTED_BENCH_WARMUP_MOVES", 4);
    const int measured_moves = benchmark_environment_integer(
        "NIMBUS_NESTED_BENCH_MEASURED_MOVES", 8);
    const std::string profile = relocation_benchmark_profile();
    constexpr int margin_parent_cells = 4;
    REQUIRE(child_nx >= 4);
    REQUIRE(child_ny >= 4);
    REQUIRE(child_nx % 2 == 0);
    REQUIRE(child_ny % 2 == 0);

    global_simulation_storage = make_relocation_benchmark_parent(
        child_nx, child_ny, nz, margin_parent_cells);
    initialize_parent_modules();
    const bool active_physics =
        profile == "flat_active" || profile == "flat_active_milbrandt";
    const bool active_milbrandt =
        profile == "flat_milbrandt" ||
        profile == "flat_active_milbrandt";
    if (active_physics)
        initialize_relocation_benchmark_active_physics();
    NestedParentMicrophysicsSnapshotV1 previous_microphysics;
    if (active_milbrandt)
        initialize_relocation_benchmark_milbrandt(previous_microphysics);
    const PrognosticFieldStorage previous_parent =
        global_simulation_storage.fields;
    prepare_relocation_benchmark_child(
        child_nx, child_ny, nz, margin_parent_cells, previous_parent,
        profile != "baseline",
        active_milbrandt ? &previous_microphysics : nullptr);

    NestedChildMovementReportV5 movement;
    for (int index = 0; index < warmup_moves; ++index)
    {
        const int shift_i = index % 2 == 0 ? 1 : -1;
        move_nested_child_integer_parent_cells(
            global_simulation_storage, shift_i, 0, {4.0}, movement);
    }
    REQUIRE_NOTHROW(require_nested_child_level_runtime_ready(
        global_simulation_storage));

    tmv::allocation_telemetry::begin_measurement();
    const auto started = std::chrono::steady_clock::now();
    for (int index = 0; index < measured_moves; ++index)
    {
        const int shift_i = (warmup_moves + index) % 2 == 0 ? 1 : -1;
        move_nested_child_integer_parent_cells(
            global_simulation_storage, shift_i, 0, {4.0}, movement);
    }
    const auto stopped = std::chrono::steady_clock::now();
    const auto allocations = tmv::allocation_telemetry::end_measurement();
    const double total_ms = std::chrono::duration<double, std::milli>(
        stopped - started).count();

    REQUIRE_NOTHROW(require_nested_child_level_runtime_ready(
        global_simulation_storage));
    if (allocations.heap_allocations != 0)
    {
        for (std::size_t bytes = 1;
             bytes <= tmv::allocation_telemetry::maximum_exact_heap_size_bytes;
             ++bytes)
        {
            const auto count = tmv::allocation_telemetry::heap_size_count(
                tmv::allocation_telemetry::Component::Unattributed, bytes);
            if (count != 0)
                std::cerr << "[NESTED RELOCATION ALLOCATION] bytes="
                          << bytes << ", count=" << count << '\n';
        }
        std::cerr << "[NESTED RELOCATION ALLOCATION] total_count="
                  << allocations.heap_allocations
                  << ", total_bytes=" << allocations.heap_requested_bytes
                  << '\n';
    }
    REQUIRE(allocations.physics_field_allocations == 0);
    REQUIRE(allocations.physics_field_allocated_bytes == 0);
    REQUIRE(allocations.heap_allocations == 0);
    REQUIRE(allocations.heap_requested_bytes == 0);
    REQUIRE(movement.rollback_journal_values <
            movement.full_primary_owner_values);
    std::cout << "[NESTED RELOCATION BENCHMARK] "
              << "profile=" << profile << ", "
              << "child_nx=" << child_nx << ", "
              << "child_ny=" << child_ny << ", "
              << "nz=" << nz << ", "
              << "warmup_moves=" << warmup_moves << ", "
              << "measured_moves=" << measured_moves << ", "
              << "total_ms=" << total_ms << ", "
              << "avg_ms=" << total_ms / measured_moves << ", "
              << "heap_allocations=" << allocations.heap_allocations << ", "
              << "heap_requested_bytes="
              << allocations.heap_requested_bytes << ", "
              << "physics_field_allocations="
              << allocations.physics_field_allocations << ", "
              << "physics_field_allocated_bytes="
              << allocations.physics_field_allocated_bytes << ", "
              << "rollback_journal_bytes="
              << movement.rollback_journal_bytes << ", "
              << "full_primary_owner_values="
              << movement.full_primary_owner_values << '\n';
}

#endif

TEST_CASE("nested rollback discards scientific publication until relaxation commits",
          "[core][nesting][production][budget][atomicity]")
{
    ProcessStateRestore restore;
    global_simulation_storage = make_parent();
    initialize_parent_modules();
    const PrognosticFieldStorage previous_parent = global_simulation_storage.fields;
    prepare_child(previous_parent);
    const auto directory = std::filesystem::temp_directory_path() /
        ("nimbus_nested_budget_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::create_directories(directory);
    struct CloseBudget
    {
        ~CloseBudget() { tmv::close_radiation_budget_csv(); }
    } close;
    tmv::open_radiation_budget_csv((directory / "radiation_budget.csv").string());
    auto& child = global_simulation_storage.nested_execution.child;
    const auto before = child.runtime.radiation.applied_step_count;
    const double before_time = child.time_s;
    const auto kernel = +[](const SimulationStateView& state, double dt, double, void* fail) {
        tmv::commit_radiation_applied_heat(state, dt, state.radiation_runtime->held_atmospheric_power_w * dt);
        if (*static_cast<bool*>(fail)) throw std::runtime_error("downstream child failure");
    };
    bool fail = true;
    NestedChildSubstepReport report;
    REQUIRE_THROWS(advance_nested_child_substep(
        global_simulation_storage, previous_parent, {4.0}, kernel, &fail, report));
    REQUIRE(child.runtime.radiation.applied_step_count == before);
    REQUIRE(child.time_s == before_time);
    const auto csv_path = directory / "nests/child_0000/radiation_budget.csv";
    REQUIRE_FALSE(std::filesystem::exists(csv_path));
    fail = false;
    REQUIRE_NOTHROW(advance_nested_child_substep(
        global_simulation_storage, previous_parent, {4.0}, kernel, &fail, report));
    REQUIRE(child.runtime.radiation.applied_step_count == before + 1);
    tmv::close_radiation_budget_csv();
    std::ifstream csv(csv_path);
    std::string row;
    REQUIRE(static_cast<bool>(std::getline(csv, row)));
    REQUIRE(static_cast<bool>(std::getline(csv, row)));
    REQUIRE(row.rfind(std::to_string(before + 1) + ",", 0) == 0);
    REQUIRE_FALSE(static_cast<bool>(std::getline(csv, row)));
    csv.close();
    std::filesystem::remove_all(directory);
}

TEST_CASE("Two-way child relocation closes inventory and checkpoints atomically",
          "[core][nesting][production][movement][two_way][checkpoint]")
{
    ProcessStateRestore restore;
    global_simulation_storage = make_parent();
    auto& storage             = global_simulation_storage;
    storage.fields.tke.fill(0);
    storage.fields.qv.fill(.003f);
    initialize_parent_modules();
    const auto previous = storage.fields;
    prepare_child(previous);
    NestedChildSubstepReport substep;
    advance_nested_child_dynamics_substep(storage, previous, {4.0}, substep);
    advance_nested_child_dynamics_substep(storage, previous, {4.0}, substep);
    auto& nested                                = storage.nested_execution;
    storage.configuration.nesting.feedback      = tmv::init::config::NestingFeedback::TwoWay;
    const auto&                          r      = nested.relationship;
    const auto&                          layers = nested.child.process_grid_metrics.dz;
    tmv::CartesianDryMassInterfaceFluxes mass;
    tmv::CartesianScalarInterfaceFluxes  scalar;
    const auto                           scalar_register = [&](auto& values)
    {
        values.west.assign(r.child_parent_span_y * r.levels_z, 0);
        values.east = values.west;
        values.south.assign(r.child_parent_span_x * r.levels_z, 0);
        values.north              = values.south;
        values.accumulated_dt_s   = r.parent_dt_s;
        values.contribution_count = 1;
    };
    scalar_register(mass);
    scalar_register(scalar);
    tmv::CartesianMomentumInterfaceFluxes momentum;
    for (std::size_t f = 0; f < 3; ++f)
    {
        momentum.west.component[f].assign(r.child_parent_span_y * (r.levels_z - 1), 0);
        momentum.east.component[f] = momentum.west.component[f];
        momentum.south.component[f].assign(r.child_parent_span_x * (r.levels_z - 1), 0);
        momentum.north.component[f] = momentum.south.component[f];
    }
    momentum.accumulated_dt_s   = r.parent_dt_s;
    momentum.contribution_count = 1;
    NestedMoistFeedbackInputs inputs;
    const auto                primary = nested_primary_inventory(r, layers, nested.child.fields);
    for (std::size_t f = 0; f < 7; ++f)
    {
        inputs.parent_flux[f] = inputs.child_flux[f] = &scalar;
        inputs.child_begin[f]                        = primary[f + 2];
    }
    const auto inventory = nested_two_way_child_inventory_flat(r, layers, nested.child.fields);
    apply_nested_two_way_feedback_flat(r,
                                       layers,
                                       mass,
                                       mass,
                                       scalar,
                                       scalar,
                                       momentum,
                                       momentum,
                                       momentum,
                                       momentum,
                                       {},
                                       {},
                                       {},
                                       inventory,
                                       storage.fields,
                                       nested.child.fields,
                                       nested.last_two_way_feedback,
                                       &inputs);
    nested.two_way_feedback_application_count = 1;
    nested.two_way_interval_source_begin      = nested.interface_source.cumulative;
    NestedChildMovementReportV5 moved;
    auto&                       pending = nested.child.runtime.advection.moist_interface.water[0];
    pending.configure(6, 6, 2, {1, 5, 1, 5, 0, 2}, 2, 2);
    pending.stage_scalar_flux([](int, int, int, int) { return 1.0; }, 1, 500, 400, 300);
    REQUIRE_THROWS(move_nested_child_integer_parent_cells(storage, 1, 0, {4.0}, moved));
    CHECK(nested.movement_sequence == 0);
    pending.reset_interval();
    move_nested_child_integer_parent_cells(storage, 1, 0, {4.0}, moved);
    REQUIRE(valid_nested_two_way_relocation_report(moved.two_way_inventory));
    CHECK(std::equal(primary.begin(), primary.end(), moved.two_way_inventory.before.begin()));
    const auto actual = nested_primary_inventory(r, layers, nested.child.fields);
    CHECK(std::equal(actual.begin(), actual.end(), moved.two_way_inventory.after.begin()));
    CHECK(moved.two_way_inventory.retired[0] > 0);
    CHECK(moved.two_way_inventory.admitted[0] > 0);
    CHECK_FALSE(storage.runtime.advection.moist_interface.enabled);
    CHECK_FALSE(nested.child.runtime.dynamics.momentum_interface_flux_capture_enabled);
    const auto valid_fields = nested.child.fields;
    nested.child.fields.p(2, 2, 0) += 100;
    const auto failure_before  = nested.child.fields;
    const auto movement_before = nested.last_two_way_feedback.relocation;
    REQUIRE_THROWS(move_nested_child_integer_parent_cells(storage, -1, 0, {4.0}, moved));
    CHECK(nested.movement_sequence == 1);
    CHECK(nested.last_two_way_feedback.relocation.after == movement_before.after);
    for (const auto pair : {std::pair<const Field3D*, const Field3D*>{&nested.child.fields.rho, &failure_before.rho},
                            {&nested.child.fields.theta, &failure_before.theta},
                            {&nested.child.fields.p, &failure_before.p},
                            {&nested.child.fields.qv, &failure_before.qv},
                            {&nested.child.fields.u, &failure_before.u}})
        CHECK(std::memcmp(pair.first->data(), pair.second->data(), pair.first->size() * sizeof(float)) == 0);
    nested.child.fields = valid_fields;
    move_nested_child_integer_parent_cells(storage, -1, 0, {4.0}, moved);
    REQUIRE(valid_nested_two_way_relocation_report(moved.two_way_inventory));
    CHECK(moved.two_way_inventory.movement_sequence == 2);
    CHECK(std::equal(primary.begin(), primary.end(), moved.two_way_inventory.before.begin()));
    std::vector<std::uint8_t> bytes;
    std::string               error;
    const bool                saved = tmv::checkpoint::serialize_simulation_storage(storage, bytes, error);
    INFO(error);
    REQUIRE(saved);
    SimulationStorageState loaded;
    REQUIRE(tmv::checkpoint::deserialize_simulation_storage(bytes, loaded, error));
    CHECK(loaded.nested_execution.last_two_way_feedback.relocation.after == moved.two_way_inventory.after);
    CHECK(loaded.nested_execution.last_two_way_feedback.relocation.movement_sequence == 2);
    const auto retained = bytes;
    nested.last_two_way_feedback.relocation.admitted[2] += 1000;
    REQUIRE_FALSE(tmv::checkpoint::serialize_simulation_storage(storage, bytes, error));
    CHECK(bytes == retained);
    nested.last_two_way_feedback.relocation.admitted[2] -= 1000;
    nested.last_two_way_feedback.relocation.after[18] += 1e9;
    REQUIRE_FALSE(tmv::checkpoint::serialize_simulation_storage(storage, bytes, error));
    CHECK(bytes == retained);
}

TEST_CASE("Mapped child relocation retains overlap and rolls back late failures",
          "[core][nesting][production][movement][mapped]")
{
    for (const std::string physics_name : {"none", "thompson", "milbrandt"})
        for (const bool two_way : {false, true})
            for (const int ratio : {2, 3})
                for (const std::string scheme : {"bell", "schar"})
                {
                    CAPTURE(physics_name, two_way, ratio, scheme);
                    ProcessStateRestore restore;
                    global_simulation_storage = make_parent(14, 14);
                    auto& storage             = global_simulation_storage;
                    storage.fields.tke.fill(0);
                    storage.fields.u.fill(7);
                    storage.fields.v.fill(-3);
                    storage.fields.w.fill(1);
                    storage.configuration.domain_motion_mode = tmv::init::config::DomainMotionMode::MovingNest;
                    storage.grid.geometry.z_faces            = {0.0, 300.0, 600.0};
                    storage.grid.geometry.z                  = {150.0, 450.0};
                    storage.process_grid_metrics.dz          = {300.0, 300.0};
                    storage.process_grid_metrics.z_int       = storage.grid.geometry.z_faces;
                    initialize_parent_modules();
                    storage.configuration.microphysics_scheme = physics_name;
                    storage.runtime.microphysics.scheme =
                        tmv::create_microphysics_scheme(physics_name, storage.configuration.microphysics_config);
                    auto& micro = *storage.runtime.microphysics.scheme;
                    micro.prepare_prognostic_state(storage.fields.rho,
                                                   storage.fields.qc,
                                                   storage.fields.qr,
                                                   storage.fields.qi,
                                                   storage.fields.qs,
                                                   storage.fields.qg,
                                                   storage.fields.qh);
                    auto parent_private = micro.prognostic_scalars_for_transport();
                    for (std::size_t f = 0; f < parent_private.size(); ++f)
                        for (int i = 0; i < storage.grid.nr; ++i)
                            for (int j = 0; j < storage.grid.nth; ++j)
                                for (int k = 0; k < storage.grid.nz; ++k)
                                    (*parent_private[f].field)(i, j, k) = float((f + 1) * (1000 + 13 * i + 7 * j + k));
                    NestedParentMicrophysicsSnapshotV1 private_snapshot;
                    if (!parent_private.empty())
                    {
                        capture_nested_parent_microphysics_snapshot(storage, private_snapshot);
                        private_snapshot.sample_time_s = 0;
                    }
                    auto& terrain                          = storage.runtime.terrain;
                    terrain.config.scheme_id               = scheme;
                    terrain.config.coord_id                = "smoothed";
                    terrain.config.horizontal_placement_id = "grid_attached";
                    terrain.config.compute_derivatives     = true;
                    terrain.config.compute_metrics         = true;
                    terrain.config.ztop                    = 600.0;
                    terrain.config.bell.h0                 = 20.0;
                    terrain.config.bell.a                  = 1600.0;
                    terrain.config.schar.H                 = 20.0;
                    terrain.config.schar.a                 = 1600.0;
                    terrain.config.schar.ell               = 2400.0;
                    terrain.scheme                         = create_terrain_scheme(scheme);
                    terrain.scheme->initialize(terrain.config);
                    tmv::EarthFixedTerrainBuildRequest request;
                    request.nr                            = 14;
                    request.nth                           = 14;
                    request.nz                            = 2;
                    request.dx_m                          = storage.grid.dx;
                    request.dy_m                          = storage.grid.dy;
                    request.feature_center_earth_m        = {7 * storage.grid.dx, 7 * storage.grid.dy};
                    auto build_config                     = terrain.config;
                    build_config.horizontal_placement_id  = "earth_fixed";
                    build_config.feature_center_earth_x_m = request.feature_center_earth_m.x_m;
                    build_config.feature_center_earth_y_m = request.feature_center_earth_m.y_m;
                    const auto built    = tmv::build_earth_fixed_terrain_geometry(build_config, request);
                    terrain.metrics     = built.metrics;
                    terrain.topography  = built.topography;
                    const auto previous = storage.fields;
                    prepare_child(previous, 12, 12, 4 * ratio, 4 * ratio, ratio);
                    auto& nested = storage.nested_execution;
                    if (!parent_private.empty())
                    {
                        NestedChildInitializationReport private_initialization;
                        initialize_nested_child_microphysics_private_state(
                            storage, previous, private_snapshot, private_initialization);
                    }
                    NestedChildSubstepReport substep;
                    const auto               idle = +[](const SimulationStateView&, double, double, void*) {};
                    for (int step = 0; step < ratio; ++step)
                        advance_nested_child_substep(storage,
                                                     previous,
                                                     {4.0},
                                                     idle,
                                                     nullptr,
                                                     substep,
                                                     parent_private.empty() ? nullptr : &private_snapshot);
                    REQUIRE(nested.active_terrain.ready);
                    REQUIRE(nested.active_terrain.geometry.sampling_version == 2);
                    if (two_way)
                    {
                        storage.configuration.nesting.feedback = tmv::init::config::NestingFeedback::TwoWay;
                        const auto& r                          = nested.relationship;
                        const auto  initialize                 = [&](auto& flux)
                        {
                            flux.west.assign(r.child_parent_span_y * r.levels_z, 0);
                            flux.east = flux.west;
                            flux.south.assign(r.child_parent_span_x * r.levels_z, 0);
                            flux.north              = flux.south;
                            flux.accumulated_dt_s   = r.parent_dt_s;
                            flux.contribution_count = 1;
                        };
                        tmv::CartesianDryMassInterfaceFluxes mass;
                        tmv::CartesianThetaInterfaceFluxes   theta;
                        initialize(mass);
                        initialize(theta);
                        tmv::CartesianMomentumInterfaceFluxes momentum;
                        for (std::size_t component = 0; component < 3; ++component)
                        {
                            momentum.west.component[component].assign(r.child_parent_span_y * (r.levels_z - 1), 0);
                            momentum.east.component[component] = momentum.west.component[component];
                            momentum.south.component[component].assign(r.child_parent_span_x * (r.levels_z - 1), 0);
                            momentum.north.component[component] = momentum.south.component[component];
                        }
                        momentum.accumulated_dt_s   = r.parent_dt_s;
                        momentum.contribution_count = 1;
                        const auto begin =
                            nested_two_way_child_inventory_mapped(nested.active_terrain.geometry, nested.child.fields);
                        const auto density_before = storage.fields.rho;
                        apply_nested_two_way_feedback_mapped(nested.active_terrain.geometry,
                                                             mass,
                                                             mass,
                                                             theta,
                                                             theta,
                                                             momentum,
                                                             momentum,
                                                             momentum,
                                                             momentum,
                                                             {},
                                                             {},
                                                             {},
                                                             begin,
                                                             storage.fields,
                                                             nested.child.fields,
                                                             nested.last_two_way_feedback);
                        if (!parent_private.empty())
                        {
                            auto child_private =
                                nested.child.runtime.microphysics.scheme->prognostic_scalars_for_transport();
                            const auto amounts = nested_private_inventory(nested_child_execution_state(storage));
                            std::vector<NestedPrivateScalarFeedbackInput> private_inputs;
                            for (std::size_t f = 0; f < parent_private.size(); ++f)
                                private_inputs.push_back({parent_private[f].name,
                                                          parent_private[f].field,
                                                          child_private[f].field,
                                                          &theta,
                                                          &theta,
                                                          amounts[f],
                                                          0,
                                                          0,
                                                          0});
                            auto prepared = prepare_nested_private_feedback(r,
                                                                            nested.child.process_grid_metrics.dz,
                                                                            &nested.active_terrain.geometry,
                                                                            density_before,
                                                                            storage.fields.rho,
                                                                            nested.child.fields.rho,
                                                                            private_inputs);
                            for (std::size_t f = 0; f < parent_private.size(); ++f)
                            {
                                std::swap(*parent_private[f].field, prepared.parent[f]);
                                if (!prepared.child[f].empty())
                                    std::swap(*child_private[f].field, prepared.child[f]);
                            }
                            nested.last_two_way_feedback.private_scalars = std::move(prepared.reports);
                        }
                        nested.two_way_feedback_application_count = 1;
                        nested.two_way_interval_source_begin      = nested.interface_source.cumulative;
                    }
                    const auto physical  = compute_physical_momentum_budget_v2(make_simulation_state_view(
                        nested.child, storage.configuration, &storage.runtime.compute, &storage.runtime.parity));
                    const auto inventory = nested_primary_inventory(nested.relationship,
                                                                    nested.child.process_grid_metrics.dz,
                                                                    nested.child.fields,
                                                                    0,
                                                                    0,
                                                                    false,
                                                                    false,
                                                                    &nested.child.runtime.terrain.metrics);
                    CHECK(inventory[9] == Approx(physical.momentum_x).epsilon(1e-12));
                    CHECK(inventory[10] == Approx(physical.momentum_y).epsilon(1e-12));
                    CHECK(inventory[11] == Approx(physical.momentum_z).epsilon(1e-12));
                    for (const auto shift : {std::pair<int, int>{1, -1}, {-1, 1}, {1, 1}, {-1, -1}})
                    {
                        CAPTURE(shift.first, shift.second);
                        const auto before = nested.child.fields;
                        auto       private_fields =
                            nested.child.runtime.microphysics.scheme->prognostic_scalars_for_transport();
                        std::vector<Field3D> private_before;
                        for (const auto& field : private_fields)
                            private_before.push_back(*field.field);
                        const auto                  geometry_before        = nested.active_terrain;
                        const auto                  metrics_before         = nested.child.runtime.terrain.metrics;
                        const auto                  process_metrics_before = nested.child.process_grid_metrics;
                        const auto                  sequence_before        = nested.movement_sequence;
                        const auto                  interface_count        = nested.interface_source.application_count;
                        NestedChildMovementReportV5 report;
                        report.movement_sequence                = 12345;
                        nested.active_terrain.publication_count = std::numeric_limits<std::uint64_t>::max();
                        REQUIRE_THROWS_WITH(
                            move_nested_child_integer_parent_cells(storage, shift.first, shift.second, {4.0}, report),
                            Catch::Matchers::Contains("publication counter is exhausted"));
                        CHECK(report.movement_sequence == 12345);
                        CHECK(nested.movement_sequence == sequence_before);
                        CHECK(nested.interface_source.application_count == interface_count);
                        CHECK(nested.active_terrain.generation == geometry_before.generation);
                        CHECK(nested.active_terrain.geometry.child_storage.metrics.z_face.values ==
                              geometry_before.geometry.child_storage.metrics.z_face.values);
                        CHECK(nested.child.runtime.terrain.metrics.J.values == metrics_before.J.values);
                        CHECK(nested.child.runtime.terrain.metrics.z.values == metrics_before.z.values);
                        CHECK(nested.child.process_grid_metrics.dz == process_metrics_before.dz);
                        CHECK(nested.child.process_grid_metrics.z_int == process_metrics_before.z_int);
                        const Field3D* expected[] = {&before.rho,
                                                     &before.theta,
                                                     &before.p,
                                                     &before.u,
                                                     &before.v,
                                                     &before.w,
                                                     &before.qv,
                                                     &before.qc,
                                                     &before.qr,
                                                     &before.qi,
                                                     &before.qs,
                                                     &before.qg,
                                                     &before.qh,
                                                     &before.tracer,
                                                     &before.tke};
                        const Field3D* actual[]   = {&nested.child.fields.rho,
                                                     &nested.child.fields.theta,
                                                     &nested.child.fields.p,
                                                     &nested.child.fields.u,
                                                     &nested.child.fields.v,
                                                     &nested.child.fields.w,
                                                     &nested.child.fields.qv,
                                                     &nested.child.fields.qc,
                                                     &nested.child.fields.qr,
                                                     &nested.child.fields.qi,
                                                     &nested.child.fields.qs,
                                                     &nested.child.fields.qg,
                                                     &nested.child.fields.qh,
                                                     &nested.child.fields.tracer,
                                                     &nested.child.fields.tke};
                        for (std::size_t f = 0; f < std::size(expected); ++f)
                            CHECK(std::memcmp(
                                      actual[f]->data(), expected[f]->data(), actual[f]->size() * sizeof(float)) == 0);
                        for (std::size_t f = 0; f < private_fields.size(); ++f)
                            CHECK(std::memcmp(private_fields[f].field->data(),
                                              private_before[f].data(),
                                              private_before[f].size() * sizeof(float)) == 0);
                        nested.active_terrain.publication_count = geometry_before.publication_count;
                        REQUIRE_NOTHROW(
                            move_nested_child_integer_parent_cells(storage, shift.first, shift.second, {4.0}, report));
                        CHECK(nested.movement_sequence == sequence_before + 1);
                        CHECK(nested.active_terrain.generation == geometry_before.generation + 1);
                        if (two_way)
                        {
                            REQUIRE(valid_nested_two_way_relocation_report(report.two_way_inventory));
                            const auto private_amounts =
                                nested_private_inventory(nested_child_execution_state(storage));
                            REQUIRE(report.two_way_inventory.private_scalars.size() == private_amounts.size());
                            for (std::size_t f = 0; f < private_amounts.size(); ++f)
                            {
                                CHECK(report.two_way_inventory.private_scalars[f].after ==
                                      Approx(private_amounts[f]).epsilon(8 * std::numeric_limits<double>::epsilon()));
                                CHECK(report.microphysics.newly_admitted_committed_inventory[f] > 0);
                            }
                            const auto committed = nested_primary_inventory(nested.relationship,
                                                                            nested.child.process_grid_metrics.dz,
                                                                            nested.child.fields,
                                                                            0,
                                                                            0,
                                                                            false,
                                                                            false,
                                                                            &nested.child.runtime.terrain.metrics);
                            CHECK(
                                std::equal(committed.begin(), committed.end(), report.two_way_inventory.after.begin()));
                        }
                        const auto& r = nested.relationship;
                        for (int i = 2; i < r.child_cells_x; ++i)
                            for (int j = 2; j < r.child_cells_y; ++j)
                            {
                                const int si = i + ratio * shift.first;
                                const int sj = j + ratio * shift.second;
                                if (si < 2 || si >= r.child_cells_x || sj < 2 || sj >= r.child_cells_y)
                                    continue;
                                for (int k = 0; k < r.levels_z; ++k)
                                {
                                    CHECK(nested.child.runtime.terrain.metrics.J(i, j, k) ==
                                          metrics_before.J(si, sj, k));
                                    for (std::size_t f = 0; f < std::size(expected); ++f)
                                    {
                                        if ((f == 3 && i + 1 >= r.child_cells_x) ||
                                            (f == 4 && j + 1 >= r.child_cells_y))
                                            continue;
                                        CAPTURE(f, i, j, k);
                                        CHECK((*actual[f])(i, j, k) == (*expected[f])(si, sj, k));
                                    }
                                }
                            }
                    }
                    if (two_way)
                    {
                        std::vector<std::uint8_t> checkpoint;
                        std::string               error;
                        INFO(error);
                        REQUIRE(tmv::checkpoint::serialize_simulation_storage(storage, checkpoint, error));
                        SimulationStorageState restored;
                        REQUIRE(tmv::checkpoint::deserialize_simulation_storage(checkpoint, restored, error));
                        CHECK(restored.nested_execution.last_two_way_feedback.relocation.after ==
                              nested.last_two_way_feedback.relocation.after);
                    }
                }
}

TEST_CASE("Mapped private forcing conserves stored moments and theta carry atomically",
          "[core][nesting][production][mapped][microphysics]")
{
    for (const std::string scheme : {"thompson", "milbrandt"})
        for (const std::string terrain_name : {"bell", "schar"})
            for (const int ratio : {2, 3})
            {
                CAPTURE(scheme, terrain_name, ratio);
                ProcessStateRestore restore;
                global_simulation_storage          = make_parent(14, 14);
                auto& storage                      = global_simulation_storage;
                storage.grid.geometry.z_faces      = {0.0, 300.0, 600.0};
                storage.grid.geometry.z            = {150.0, 450.0};
                storage.process_grid_metrics.dz    = {300.0, 300.0};
                storage.process_grid_metrics.z_int = storage.grid.geometry.z_faces;
                initialize_parent_modules();
                auto& terrain                          = storage.runtime.terrain;
                terrain.config.scheme_id               = terrain_name;
                terrain.config.coord_id                = "smoothed";
                terrain.config.horizontal_placement_id = "grid_attached";
                terrain.config.compute_derivatives     = true;
                terrain.config.compute_metrics         = true;
                terrain.config.ztop                    = 600.0;
                terrain.config.bell.h0                 = 20.0;
                terrain.config.bell.a                  = 1600.0;
                terrain.config.schar.H                 = 20.0;
                terrain.config.schar.a                 = 1600.0;
                terrain.config.schar.ell               = 2400.0;
                terrain.scheme                         = create_terrain_scheme(terrain_name);
                terrain.scheme->initialize(terrain.config);
                tmv::EarthFixedTerrainBuildRequest request;
                request.nr                            = 14;
                request.nth                           = 14;
                request.nz                            = 2;
                request.dx_m                          = storage.grid.dx;
                request.dy_m                          = storage.grid.dy;
                request.feature_center_earth_m        = {7 * storage.grid.dx, 7 * storage.grid.dy};
                auto build_config                     = terrain.config;
                build_config.horizontal_placement_id  = "earth_fixed";
                build_config.feature_center_earth_x_m = request.feature_center_earth_m.x_m;
                build_config.feature_center_earth_y_m = request.feature_center_earth_m.y_m;
                const auto built                      = tmv::build_earth_fixed_terrain_geometry(build_config, request);
                terrain.metrics                       = built.metrics;
                terrain.topography                    = built.topography;
                storage.configuration.microphysics_scheme = scheme;
                storage.runtime.microphysics.scheme =
                    tmv::create_microphysics_scheme(scheme, storage.configuration.microphysics_config);
                auto& parent_scheme = *storage.runtime.microphysics.scheme;
                parent_scheme.prepare_prognostic_state(storage.fields.rho,
                                                       storage.fields.qc,
                                                       storage.fields.qr,
                                                       storage.fields.qi,
                                                       storage.fields.qs,
                                                       storage.fields.qg,
                                                       storage.fields.qh);
                auto checkpoint_probe =
                    tmv::create_microphysics_scheme(scheme, storage.configuration.microphysics_config);
                std::string startup_error;
                const bool  startup_restored =
                    checkpoint_probe->restore_portable_state(parent_scheme.portable_state(), startup_error);
                INFO(startup_error);
                REQUIRE(startup_restored);
                auto parent_moments = parent_scheme.prognostic_scalars_for_transport();
                for (std::size_t species = 0; species < parent_moments.size(); ++species)
                    for (int i = 0; i < 14; ++i)
                        for (int j = 0; j < 14; ++j)
                            for (int k = 0; k < 2; ++k)
                                (*parent_moments[species].field)[i][j][k] =
                                    static_cast<float>(1000 * (species + 1) + 37 * i + 13 * j + 7 * k);
                const auto                         previous = storage.fields;
                NestedParentMicrophysicsSnapshotV1 previous_microphysics;
                capture_nested_parent_microphysics_snapshot(storage, previous_microphysics);
                previous_microphysics.sample_time_s = 0.0;
                prepare_child(previous, 12, 12, 4 * ratio, 4 * ratio, ratio, 2);
                auto&                           nested = storage.nested_execution;
                NestedChildInitializationReport initialization;
                initialize_nested_child_microphysics_private_state(
                    storage, previous, previous_microphysics, initialization);
                REQUIRE_NOTHROW(require_nested_child_level_runtime_ready(storage));
                auto child_moments = nested.child.runtime.microphysics.scheme->prognostic_scalars_for_transport();
                REQUIRE(initialization.microphysics_private_scalar_count == parent_moments.size());
                const auto& r               = nested.relationship;
                const auto& pm              = nested.active_terrain.geometry.parent.metrics;
                const auto& cm              = nested.active_terrain.geometry.child_storage.metrics;
                const auto  child_inventory = [&](const Field3D& rho, const Field3D* moment)
                {
                    long double total = 0;
                    for (int i = 1; i <= r.child_cells_x; ++i)
                        for (int j = 1; j <= r.child_cells_y; ++j)
                            for (int k = 0; k < r.levels_z; ++k)
                                total += static_cast<long double>(rho[i][j][k]) * (moment ? (*moment)[i][j][k] : 1.0f) *
                                         cm.J(i, j, k) * cm.dx * cm.dy * cm.dzeta;
                    return static_cast<double>(total);
                };
                const auto check_equal = [](const Field3D& actual, const Field3D& expected)
                {
                    REQUIRE(actual.size() == expected.size());
                    for (std::size_t n = 0; n < actual.size(); ++n)
                        REQUIRE(actual.data()[n] == expected.data()[n]);
                };
                for (std::size_t species = 0; species < parent_moments.size(); ++species)
                {
                    long double parent_total = 0;
                    for (int i = r.child_origin_parent_cell_i; i < r.child_origin_parent_cell_i + r.child_parent_span_x;
                         ++i)
                        for (int j = r.child_origin_parent_cell_j;
                             j < r.child_origin_parent_cell_j + r.child_parent_span_y;
                             ++j)
                            for (int k = 0; k < r.levels_z; ++k)
                                parent_total += static_cast<long double>(previous.rho[i + 1][j + 1][k]) *
                                                previous_microphysics.fields[species][i + 1][j + 1][k] * pm.J(i, j, k) *
                                                pm.dx * pm.dy * pm.dzeta;
                    const auto& report = initialization.microphysics_private[species];
                    CHECK(report.target_carrier_inventory == Approx(static_cast<double>(parent_total)).epsilon(2e-15));
                    CHECK(
                        report.committed_carrier_inventory ==
                        Approx(child_inventory(nested.child.fields.rho, child_moments[species].field)).epsilon(2e-15));
                    CHECK(report.committed_dry_mass_kg ==
                          Approx(child_inventory(nested.child.fields.rho, nullptr)).epsilon(2e-15));
                    CHECK(report.carrier_storage_residual ==
                          report.target_carrier_inventory - report.committed_carrier_inventory);
                    CHECK(report.dry_storage_residual_kg == report.target_dry_mass_kg - report.committed_dry_mass_kg);
                }
                std::vector<Field3D> retained_moments;
                for (const auto& binding : child_moments)
                    retained_moments.push_back(*binding.field);
                const auto retained_fields       = nested.child.fields;
                const auto retained_generation   = nested.active_terrain.generation;
                const auto retained_applications = nested.interface_source.application_count;
                float&     initial_endpoint =
                    previous_microphysics.fields[previous_microphysics.scalar_count - 1]
                                                [r.child_origin_parent_cell_i + 1][r.child_origin_parent_cell_j + 1][0];
                const float initial_value = initial_endpoint;
                initial_endpoint          = std::numeric_limits<float>::quiet_NaN();
                NestedChildInitializationReport rejected_initialization;
                rejected_initialization.theta.target_carrier_inventory = -123;
                REQUIRE_THROWS(initialize_nested_child_microphysics_private_state(
                    storage, previous, previous_microphysics, rejected_initialization));
                initial_endpoint = initial_value;
                CHECK(rejected_initialization.theta.target_carrier_inventory == -123);
                CHECK(nested.interface_source.application_count == retained_applications);
                for (std::size_t species = 0; species < child_moments.size(); ++species)
                    check_equal(*child_moments[species].field, retained_moments[species]);
                auto& carry = nested.child.runtime.microphysics.theta_storage_residual;
                carry.resize(nested.child.grid.nr, nested.child.grid.nth, nested.child.grid.nz, 0.00001f);
                const auto retained_carry = carry;
                for (auto& binding : parent_moments)
                    for (std::size_t n = 0; n < binding.field->size(); ++n)
                        binding.field->data()[n] *= 1.25f;
                for (std::size_t n = 0; n < storage.fields.rho.size(); ++n)
                    storage.fields.rho.data()[n] *= 1.01f;
                nested.parent_time_bracket.completed_child_substeps = 1;
                nested.child.time_s                                 = 2.0 / ratio;
                nested.active_terrain.child_time_s                  = nested.child.time_s;
                nested.active_terrain.parent_time_s                 = nested.child.time_s;
                nested.active_terrain.ledger.child_time_s           = nested.child.time_s;
                nested.active_terrain.ledger.parent_time_s          = nested.child.time_s;
                const int   pi                                      = r.child_origin_parent_cell_i + 1;
                const int   pj                                      = r.child_origin_parent_cell_j + 1;
                float&      last_endpoint  = (*parent_moments[parent_moments.size() - 1].field)[pi][pj][0];
                const float valid_endpoint = last_endpoint;
                last_endpoint              = std::numeric_limits<float>::quiet_NaN();
                NestedChildRelaxationReport relaxation;
                relaxation.theta.target_carrier_change = -123.0;
                REQUIRE_THROWS(apply_nested_child_physical_relaxation(
                    storage, previous, {4.0}, 1.0, relaxation, &previous_microphysics));
                last_endpoint = valid_endpoint;
                for (std::size_t species = 0; species < child_moments.size(); ++species)
                    check_equal(*child_moments[species].field, retained_moments[species]);
                check_equal(nested.child.fields.rho, retained_fields.rho);
                check_equal(carry, retained_carry);
                CHECK(nested.active_terrain.generation == retained_generation);
                CHECK(nested.interface_source.application_count == retained_applications);
                CHECK(relaxation.theta.target_carrier_change == -123.0);
                carry[r.child_cells_x][r.child_cells_y][r.levels_z - 1] = std::numeric_limits<float>::quiet_NaN();
                REQUIRE_THROWS(apply_nested_child_physical_relaxation(
                    storage, previous, {4.0}, 1.0, relaxation, &previous_microphysics));
                carry[r.child_cells_x][r.child_cells_y][r.levels_z - 1] = 0.00001f;
                for (std::size_t species = 0; species < child_moments.size(); ++species)
                    check_equal(*child_moments[species].field, retained_moments[species]);
                check_equal(nested.child.fields.rho, retained_fields.rho);
                check_equal(nested.child.fields.theta, retained_fields.theta);
                check_equal(nested.child.fields.u, retained_fields.u);
                check_equal(carry, retained_carry);
                CHECK(nested.active_terrain.generation == retained_generation);
                CHECK(nested.interface_source.application_count == retained_applications);
                CHECK(relaxation.theta.target_carrier_change == -123.0);
                REQUIRE_NOTHROW(apply_nested_child_physical_relaxation(
                    storage, previous, {4.0}, 1.0, relaxation, &previous_microphysics));
                REQUIRE(relaxation.microphysics_private.scalar_count == child_moments.size());
                for (std::size_t species = 0; species < child_moments.size(); ++species)
                {
                    const double change = child_inventory(nested.child.fields.rho, child_moments[species].field) -
                                          child_inventory(retained_fields.rho, &retained_moments[species]);
                    CHECK(relaxation.microphysics_private.committed_carrier_change[species] ==
                          Approx(change).epsilon(2e-13));
                    for (int i = 1; i <= r.child_cells_x; ++i)
                        for (int j = 1; j <= r.child_cells_y; ++j)
                            for (int k = 0; k < r.levels_z; ++k)
                            {
                                const int distance = std::min({i - 1, j - 1, r.child_cells_x - i, r.child_cells_y - j});
                                const double fraction =
                                    distance == 0 ? 1.0 : (distance == 1 ? -std::expm1(-0.125) : 0.0);
                                const int    source_i        = r.child_origin_parent_cell_i + (i - 1) / ratio;
                                const int    source_j        = r.child_origin_parent_cell_j + (j - 1) / ratio;
                                const double rho0            = previous.rho[source_i + 1][source_j + 1][k];
                                const double rho1            = storage.fields.rho[source_i + 1][source_j + 1][k];
                                const double blended_rho     = ((ratio - 1) * rho0 + rho1) / ratio;
                                const float  target_specific = static_cast<float>(
                                    ((ratio - 1) * rho0 *
                                         previous_microphysics.fields[species][source_i + 1][source_j + 1][k] +
                                     rho1 * (*parent_moments[species].field)[source_i + 1][source_j + 1][k]) /
                                    ((ratio - 1) * rho0 + rho1));
                                long double child_volume = 0;
                                const int   block_i      = ((i - 1) / ratio) * ratio + 1;
                                const int   block_j      = ((j - 1) / ratio) * ratio + 1;
                                for (int fi = 0; fi < ratio; ++fi)
                                    for (int fj = 0; fj < ratio; ++fj)
                                        child_volume += cm.J(block_i + fi, block_j + fj, k) * cm.dx * cm.dy * cm.dzeta;
                                const float  target_rho = static_cast<float>(blended_rho * pm.J(source_i, source_j, k) *
                                                                             pm.dx * pm.dy * pm.dzeta / child_volume);
                                const double old_rho    = retained_fields.rho[i][j][k];
                                const double new_rho    = nested.child.fields.rho[i][j][k];
                                const double old_carrier    = old_rho * retained_moments[species][i][j][k];
                                const double target_carrier = static_cast<double>(target_rho) * target_specific;
                                const double expected =
                                    (old_carrier + fraction * (target_carrier - old_carrier)) / new_rho;
                                CHECK((*child_moments[species].field)[i][j][k] == Approx(expected).epsilon(2e-7));
                                const float expected_carry =
                                    static_cast<float>((1.0 - fraction) * old_rho / new_rho * retained_carry[i][j][k]);
                                CHECK(carry[i][j][k] == expected_carry);
                            }
                }
                const auto               idle = +[](const SimulationStateView&, double, double, void*) {};
                NestedChildSubstepReport substep;
                while (nested.parent_time_bracket.completed_child_substeps < ratio)
                    advance_nested_child_substep(
                        storage, previous, {4.0}, idle, nullptr, substep, &previous_microphysics);
                std::vector<std::uint8_t> checkpoint;
                std::string               checkpoint_error;
                const bool saved = tmv::checkpoint::serialize_simulation_storage(storage, checkpoint, checkpoint_error);
                INFO(checkpoint_error);
                REQUIRE(saved);
                SimulationStorageState restarted;
                const bool             loaded =
                    tmv::checkpoint::deserialize_simulation_storage(checkpoint, restarted, checkpoint_error);
                INFO(checkpoint_error);
                REQUIRE(loaded);
                std::vector<std::uint8_t> roundtrip;
                REQUIRE(tmv::checkpoint::serialize_simulation_storage(restarted, roundtrip, checkpoint_error));
                CHECK(roundtrip == checkpoint);
                tmv::initialize_dynamics(make_simulation_state_view(restarted),
                                         restarted.configuration.dynamics_scheme_name);
                rehydrate_nested_child_transient_operators(restarted);
                const auto continue_interval = [&](SimulationStorageState& state)
                {
                    state.fields.qv.fill(.008f);
                    state.fields.qc.fill(.001f);
                    state.fields.qr.fill(.0001f);
                    state.nested_execution.child.fields.qv.fill(.008f);
                    state.nested_execution.child.fields.qc.fill(.001f);
                    state.nested_execution.child.fields.qr.fill(.0001f);

                    const auto                         previous_fields = state.fields;
                    NestedParentMicrophysicsSnapshotV1 snapshot;
                    capture_nested_parent_microphysics_snapshot(state, snapshot);
                    state.time_s += 2.0;
                    roll_nested_parent_interval(state);
                    auto& nest           = state.nested_execution;
                    auto& capture        = nest.child.runtime.advection.private_interface;
                    auto  parent_fields  = state.runtime.microphysics.scheme->prognostic_scalars_for_transport();
                    auto  child_fields   = nest.child.runtime.microphysics.scheme->prognostic_scalars_for_transport();
                    const auto& relation = nest.relationship;
                    nest.child.runtime.advection.reset_for_shape(
                        nest.child.grid.nr, nest.child.grid.nth, nest.child.grid.nz);
                    capture.configure(
                        nest.child.grid.nr,
                        nest.child.grid.nth,
                        nest.child.grid.nz,
                        {1, relation.child_cells_x + 1, 1, relation.child_cells_y + 1, 0, relation.levels_z},
                        ratio,
                        child_fields);
                    const auto          inventory_begin = nested_private_inventory(nested_child_execution_state(state));
                    std::vector<double> source(child_fields.size()), storage_rounding(child_fields.size());
                    for (int phase = 0; phase < ratio; ++phase)
                    {
                        advance_nested_child_dynamics_substep(state, previous_fields, {4.0}, substep, &snapshot);
                        REQUIRE(capture.fields.size() == child_fields.size());
                        for (std::size_t f = 0; f < child_fields.size(); ++f)
                        {
                            REQUIRE(capture.fields[f].boundary.complete);
                            source[f] += substep.interface_source.microphysics_private.committed_carrier_change[f];
                            storage_rounding[f] += capture.fields[f].boundary.closure_residual;
                        }
                    }
                    child_fields  = nest.child.runtime.microphysics.scheme->prognostic_scalars_for_transport();
                    parent_fields = state.runtime.microphysics.scheme->prognostic_scalars_for_transport();
                    tmv::CartesianScalarInterfaceFluxes coarse;
                    coarse.west.assign(relation.child_parent_span_y * relation.levels_z, 0);
                    coarse.east = coarse.west;
                    coarse.south.assign(relation.child_parent_span_x * relation.levels_z, 0);
                    coarse.north              = coarse.south;
                    coarse.accumulated_dt_s   = 2;
                    coarse.contribution_count = 1;
                    std::vector<NestedPrivateScalarFeedbackInput> inputs;
                    double                                        transported = 0;
                    double                                        physics     = 0;
                    for (std::size_t f = 0; f < child_fields.size(); ++f)
                    {
                        const auto& captured = capture.fields[f];
                        const auto& flux     = captured.flux.scalar();
                        CHECK(flux.accumulated_dt_s == Approx(2.0));
                        CHECK(flux.contribution_count >= static_cast<std::uint64_t>(ratio));
                        for (double v : flux.west)
                            transported += std::abs(v);
                        for (double v : flux.east)
                            transported += std::abs(v);
                        physics += std::abs(captured.committed_physics);
                        inputs.push_back({captured.name,
                                          parent_fields[f].field,
                                          child_fields[f].field,
                                          &coarse,
                                          &flux,
                                          inventory_begin[f],
                                          source[f],
                                          captured.committed_physics,
                                          storage_rounding[f]});
                    }
                    CHECK(transported > 0);
                    CHECK(physics > 0);
                    const auto prepare =
                        [&](const std::vector<NestedPrivateScalarFeedbackInput>& bindings, const Field3D& density)
                    {
                        return prepare_nested_private_feedback(relation,
                                                               nest.child.process_grid_metrics.dz,
                                                               &nest.active_terrain.geometry,
                                                               state.fields.rho,
                                                               density,
                                                               nest.child.fields.rho,
                                                               bindings);
                    };
                    auto malformed                  = inputs;
                    malformed.back().physics_source = std::numeric_limits<double>::quiet_NaN();
                    const Field3D retained_parent   = *parent_fields[0].field;
                    const Field3D retained_child    = *child_fields[0].field;
                    REQUIRE_THROWS(prepare(malformed, state.fields.rho));
                    check_equal(*parent_fields[0].field, retained_parent);
                    check_equal(*child_fields[0].field, retained_child);
                    Field3D changed_density = state.fields.rho;
                    for (std::size_t i = 0; i < changed_density.size(); ++i)
                        changed_density.data()[i] *= 1.125f;
                    const auto rescaled = prepare(inputs, changed_density);
                    REQUIRE(valid_nested_private_feedback_reports(rescaled.reports));
                    CHECK(rescaled.parent[0][1][1][0] == Approx(static_cast<double>(retained_parent[1][1][0]) *
                                                                state.fields.rho[1][1][0] / changed_density[1][1][0])
                                                             .epsilon(2e-7));
                    auto prepared = prepare(inputs, state.fields.rho);
                    REQUIRE(valid_nested_private_feedback_reports(prepared.reports));
                    for (std::size_t f = 0; f < child_fields.size(); ++f)
                    {
                        CHECK(prepared.reports[f].name == parent_fields[f].name);
                        std::swap(*parent_fields[f].field, prepared.parent[f]);
                        if (!prepared.child[f].empty())
                            std::swap(*child_fields[f].field, prepared.child[f]);
                    }
                    capture.reset_interval();
                    CHECK(capture.interval_empty());
                    std::vector<std::uint8_t> result;
                    const bool                serialized =
                        tmv::checkpoint::serialize_simulation_storage(state, result, checkpoint_error);
                    INFO(checkpoint_error);
                    REQUIRE(serialized);
                    return result;
                };
                const auto uninterrupted = continue_interval(storage);
                CHECK(continue_interval(restarted) == uninterrupted);
            }
}

TEST_CASE("Private feedback checkpoints bind names inventories and closed intervals",
          "[core][nesting][private-feedback][checkpoint]")
{
    for (const std::string scheme : {"thompson", "milbrandt"})
    {
        CAPTURE(scheme);
        ProcessStateRestore restore;
        global_simulation_storage = make_parent();
        auto& storage             = global_simulation_storage;
        storage.fields.tke.fill(0);
        initialize_parent_modules();
        storage.configuration.microphysics_scheme = scheme;
        storage.runtime.microphysics.scheme =
            tmv::create_microphysics_scheme(scheme, storage.configuration.microphysics_config);
        auto& physics = *storage.runtime.microphysics.scheme;
        physics.prepare_prognostic_state(storage.fields.rho,
                                         storage.fields.qc,
                                         storage.fields.qr,
                                         storage.fields.qi,
                                         storage.fields.qs,
                                         storage.fields.qg,
                                         storage.fields.qh);
        auto parent_private = physics.prognostic_scalars_for_transport();
        for (std::size_t f = 0; f < parent_private.size(); ++f)
            parent_private[f].field->fill(static_cast<float>(1000 * (f + 1)));
        const auto                         previous = storage.fields;
        NestedParentMicrophysicsSnapshotV1 snapshot;
        capture_nested_parent_microphysics_snapshot(storage, snapshot);
        snapshot.sample_time_s = 0;
        prepare_child(previous);
        auto&                           nested = storage.nested_execution;
        NestedChildInitializationReport initialization;
        initialize_nested_child_microphysics_private_state(storage, previous, snapshot, initialization);
        const auto               idle = +[](const SimulationStateView&, double, double, void*) {};
        NestedChildSubstepReport substep;
        for (int step = 0; step < 2; ++step)
            advance_nested_child_substep(storage, previous, {4.0}, idle, nullptr, substep, &snapshot);
        const auto&                          r = nested.relationship;
        tmv::CartesianDryMassInterfaceFluxes mass;
        tmv::CartesianScalarInterfaceFluxes  scalar;
        const auto                           initialize = [&](auto& flux)
        {
            flux.west.assign(r.child_parent_span_y * r.levels_z, 0);
            flux.east = flux.west;
            flux.south.assign(r.child_parent_span_x * r.levels_z, 0);
            flux.north              = flux.south;
            flux.accumulated_dt_s   = r.parent_dt_s;
            flux.contribution_count = 1;
        };
        initialize(mass);
        initialize(scalar);
        tmv::CartesianMomentumInterfaceFluxes momentum;
        for (std::size_t f = 0; f < 3; ++f)
        {
            momentum.west.component[f].assign(r.child_parent_span_y * (r.levels_z - 1), 0);
            momentum.east.component[f] = momentum.west.component[f];
            momentum.south.component[f].assign(r.child_parent_span_x * (r.levels_z - 1), 0);
            momentum.north.component[f] = momentum.south.component[f];
        }
        momentum.accumulated_dt_s   = r.parent_dt_s;
        momentum.contribution_count = 1;
        const auto density_before   = storage.fields.rho;
        const auto inventory =
            nested_two_way_child_inventory_flat(r, nested.child.process_grid_metrics.dz, nested.child.fields);
        apply_nested_two_way_feedback_flat(r,
                                           nested.child.process_grid_metrics.dz,
                                           mass,
                                           mass,
                                           scalar,
                                           scalar,
                                           momentum,
                                           momentum,
                                           momentum,
                                           momentum,
                                           {},
                                           {},
                                           {},
                                           inventory,
                                           storage.fields,
                                           nested.child.fields,
                                           nested.last_two_way_feedback);
        auto       child_private = nested.child.runtime.microphysics.scheme->prognostic_scalars_for_transport();
        const auto moments       = nested_private_inventory(nested_child_execution_state(storage));
        std::vector<NestedPrivateScalarFeedbackInput> inputs;
        for (std::size_t f = 0; f < parent_private.size(); ++f)
            inputs.push_back({parent_private[f].name,
                              parent_private[f].field,
                              child_private[f].field,
                              &scalar,
                              &scalar,
                              moments[f],
                              0,
                              0,
                              0});
        auto candidate = prepare_nested_private_feedback(r,
                                                         nested.child.process_grid_metrics.dz,
                                                         nullptr,
                                                         density_before,
                                                         storage.fields.rho,
                                                         nested.child.fields.rho,
                                                         inputs);
        for (std::size_t f = 0; f < parent_private.size(); ++f)
        {
            std::swap(*parent_private[f].field, candidate.parent[f]);
            if (!candidate.child[f].empty())
                std::swap(*child_private[f].field, candidate.child[f]);
        }
        nested.last_two_way_feedback.private_scalars = std::move(candidate.reports);
        nested.last_two_way_feedback.private_scalars.front().carrier.parent.constrained_cells = 1;
        nested.two_way_feedback_application_count                                             = 1;
        nested.two_way_interval_source_begin         = nested.interface_source.cumulative;
        storage.configuration.nesting.feedback       = tmv::init::config::NestingFeedback::TwoWay;
        std::vector<std::uint8_t> checkpoint;
        std::string               error;
        const bool                saved = tmv::checkpoint::serialize_simulation_storage(storage, checkpoint, error);
        INFO(error);
        REQUIRE(saved);
        SimulationStorageState loaded;
        REQUIRE(tmv::checkpoint::deserialize_simulation_storage(checkpoint, loaded, error));
        REQUIRE(loaded.nested_execution.last_two_way_feedback.private_scalars.size() == parent_private.size());
        CHECK(loaded.nested_execution.last_two_way_feedback.private_scalars.front().carrier.parent.constrained_cells ==
              1);
        std::vector<std::uint8_t> roundtrip;
        REQUIRE(tmv::checkpoint::serialize_simulation_storage(loaded, roundtrip, error));
        CHECK(roundtrip == checkpoint);
        const auto retained    = checkpoint;
        auto&      constraints = nested.last_two_way_feedback.private_scalars.front().carrier.parent.constrained_cells;
        constraints            = std::numeric_limits<std::size_t>::max();
        REQUIRE_FALSE(tmv::checkpoint::serialize_simulation_storage(storage, checkpoint, error));
        CHECK(checkpoint == retained);
        constraints         = 1;
        auto&      last     = nested.last_two_way_feedback.private_scalars.back();
        const auto name     = last.name;
        last.name           = nested.last_two_way_feedback.private_scalars.front().name;
        REQUIRE_FALSE(tmv::checkpoint::serialize_simulation_storage(storage, checkpoint, error));
        CHECK(checkpoint == retained);
        last.name            = name;
        float&      value    = (*child_private[0].field)[1][1][0];
        const float original = value;
        value += 100;
        REQUIRE_FALSE(tmv::checkpoint::serialize_simulation_storage(storage, checkpoint, error));
        CHECK(checkpoint == retained);
        value         = original;
        auto& capture = nested.child.runtime.advection.private_interface;
        capture.configure(nested.child.grid.nr,
                          nested.child.grid.nth,
                          nested.child.grid.nz,
                          {1, r.child_cells_x + 1, 1, r.child_cells_y + 1, 0, r.levels_z},
                          r.parent_ratio_x,
                          child_private);
        capture.fields[0].committed_physics = 1;
        REQUIRE_FALSE(tmv::checkpoint::serialize_simulation_storage(storage, checkpoint, error));
        CHECK(checkpoint == retained);
        capture.reset_interval();
        REQUIRE(tmv::checkpoint::serialize_simulation_storage(storage, checkpoint, error));
        CHECK(checkpoint == retained);
        REQUIRE_THROWS_WITH(
            tmv::domain::prepare_nested_grid_domain_recycling(storage, {}, storage.configuration.domain_frame),
            Catch::Matchers::Contains("private carriers"));
        {
            auto unreported = storage;
            unreported.nested_execution.last_two_way_feedback.private_scalars.clear();
            REQUIRE_THROWS_WITH(
                tmv::domain::prepare_nested_grid_domain_recycling(unreported, {}, unreported.configuration.domain_frame),
                Catch::Matchers::Contains("private carriers"));
        }
        {
            auto       recycled = loaded;
            auto       view     = make_simulation_state_view(recycled);
            const auto physical = physical_scalar_domain(view.grid);
            const auto plan     = tmv::domain::plan_whole_cell_shift_for_physical_domain(view.grid.nr,
                                                                                         view.grid.nth,
                                                                                         view.grid.nz,
                                                                                         1,
                                                                                         0,
                                                                                         physical.i_begin,
                                                                                         physical.i_end,
                                                                                         physical.j_begin,
                                                                                         physical.j_end);
            const auto primary  = tmv::domain::prepare_flat_homogeneous_primary_profiles(recycled, {0, 0});
            const auto private_leading =
                tmv::domain::prepare_flat_homogeneous_microphysics_recycling(view, plan, primary);
            const auto exchange = tmv::domain::compute_domain_recycling_control_volume_exchange_ledger(
                view, plan, primary, recycled.configuration.domain_frame, &private_leading);
            auto post_frame = recycled.configuration.domain_frame;
            tmv::domain::commit_whole_cell_shift_to_domain_frame(post_frame, plan, view.grid.dr, view.grid.dtheta);
            auto invalid_exchange = exchange;
            invalid_exchange.pre_shift.private_inventory.entries[0].inventory *= 2;
            REQUIRE_THROWS(
                tmv::domain::prepare_nested_grid_domain_recycling(recycled, plan, post_frame, &invalid_exchange));
            auto nested_commit =
                tmv::domain::prepare_nested_grid_domain_recycling(recycled, plan, post_frame, &exchange);
            auto provenance = tmv::domain::prepare_domain_recycling_transaction_provenance(
                view,
                plan,
                recycled.configuration.domain_frame,
                exchange,
                "private-recycle",
                "private-generation",
                "npy_3d",
                "canonical_rectangular_earth_swath_capture_v1",
                recycled.time_s,
                1,
                0,
                8192,
                12345);
            auto bindings = tmv::domain::bind_homogeneous_primary_fields(recycled, primary);
            REQUIRE(tmv::domain::append_prepared_microphysics_domain_recycling_bindings(
                view, plan, private_leading, bindings));
            std::vector<tmv::domain::HomogeneousRuntimeCommitBinding> commits;
            REQUIRE(tmv::domain::append_nested_grid_domain_recycling_binding(recycled, nested_commit, commits));
            REQUIRE(tmv::domain::append_domain_recycling_transaction_provenance_binding(view, provenance, commits));
            const auto child_inventory = nested_private_inventory(nested_child_execution_state(recycled));
            tmv::domain::execute_homogeneous_window_shift(
                plan,
                bindings,
                {},
                {{"origin_x",
                  &recycled.configuration.domain_frame.earth_origin_at_t0_m.x_m,
                  post_frame.earth_origin_at_t0_m.x_m},
                 {"origin_y",
                  &recycled.configuration.domain_frame.earth_origin_at_t0_m.y_m,
                  post_frame.earth_origin_at_t0_m.y_m}},
                {},
                {},
                {},
                commits);
            REQUIRE_NOTHROW(tmv::domain::verify_domain_recycling_control_volume_exchange_after_commit(
                make_simulation_state_view(recycled), plan, post_frame, exchange));
            CHECK(nested_private_inventory(nested_child_execution_state(recycled)) == child_inventory);
            CHECK(recycled.nested_execution.cumulative_parent_recycling_shift_i == 1);
            CHECK(recycled.runtime.domain_recycling_provenance.private_inventory_complete);
            std::vector<std::uint8_t> recycled_bytes, restored_bytes;
            REQUIRE(tmv::checkpoint::serialize_simulation_storage(recycled, recycled_bytes, error));
            SimulationStorageState restored_recycling;
            REQUIRE(tmv::checkpoint::deserialize_simulation_storage(recycled_bytes, restored_recycling, error));
            REQUIRE(tmv::checkpoint::serialize_simulation_storage(restored_recycling, restored_bytes, error));
            CHECK(recycled_bytes == restored_bytes);
            for (auto* continuation : {&recycled, &restored_recycling})
            {
                tmv::initialize_dynamics(make_simulation_state_view(*continuation),
                                         continuation->configuration.dynamics_scheme_name);
                rehydrate_nested_child_transient_operators(*continuation);
                NestedChildMovementReportV5 continued_movement;
                REQUIRE_NOTHROW(move_nested_child_integer_parent_cells(*continuation, 1, 0, {4.0}, continued_movement));
            }
            REQUIRE(tmv::checkpoint::serialize_simulation_storage(recycled, recycled_bytes, error));
            REQUIRE(tmv::checkpoint::serialize_simulation_storage(restored_recycling, restored_bytes, error));
            CHECK(recycled_bytes == restored_bytes);
        }
        NestedChildMovementReportV5 movement;
        const auto                  before_move = nested_private_inventory(nested_child_execution_state(storage));
        REQUIRE_NOTHROW(move_nested_child_integer_parent_cells(storage, 1, 0, {4.0}, movement));
        REQUIRE(valid_nested_two_way_relocation_report(movement.two_way_inventory));
        REQUIRE(movement.two_way_inventory.private_scalars.size() == before_move.size());
        const auto after_move = nested_private_inventory(nested_child_execution_state(storage));
        for (std::size_t f = 0; f < before_move.size(); ++f)
        {
            const auto& ledger = movement.two_way_inventory.private_scalars[f];
            CHECK(ledger.name == parent_private[f].name);
            CHECK(ledger.before == before_move[f]);
            CHECK(ledger.after == after_move[f]);
            CHECK(ledger.retired > 0);
            CHECK(ledger.admitted > 0);
            CHECK(std::abs(ledger.after - ledger.before + ledger.retired - ledger.admitted - ledger.boundary_source) <=
                  128 * std::numeric_limits<double>::epsilon() * ledger.arithmetic_scale);
        }
        REQUIRE(tmv::checkpoint::serialize_simulation_storage(storage, checkpoint, error));
        SimulationStorageState moved_restored;
        REQUIRE(tmv::checkpoint::deserialize_simulation_storage(checkpoint, moved_restored, error));
        const auto   serialized_move            = checkpoint;
        auto&        private_ledger             = nested.last_two_way_feedback.relocation.private_scalars;
        const double admitted_before_corruption = private_ledger.back().admitted;
        private_ledger.back().admitted += private_ledger.back().arithmetic_scale;
        REQUIRE_FALSE(tmv::checkpoint::serialize_simulation_storage(storage, checkpoint, error));
        CHECK(checkpoint == serialized_move);
        private_ledger.back().admitted = admitted_before_corruption;
        REQUIRE_NOTHROW(move_nested_child_integer_parent_cells(storage, -1, 0, {4.0}, movement));
        REQUIRE(valid_nested_two_way_relocation_report(movement.two_way_inventory));
        CHECK(movement.two_way_inventory.private_scalars.front().before == before_move.front());
        CHECK(movement.two_way_inventory.movement_sequence == 2);
        REQUIRE(tmv::checkpoint::serialize_simulation_storage(storage, checkpoint, error));
        tmv::initialize_dynamics(make_simulation_state_view(moved_restored),
                                 moved_restored.configuration.dynamics_scheme_name);
        rehydrate_nested_child_transient_operators(moved_restored);
        NestedChildMovementReportV5 resumed_movement;
        REQUIRE_NOTHROW(move_nested_child_integer_parent_cells(moved_restored, -1, 0, {4.0}, resumed_movement));
        std::vector<std::uint8_t> resumed_checkpoint;
        REQUIRE(tmv::checkpoint::serialize_simulation_storage(moved_restored, resumed_checkpoint, error));
        CHECK(checkpoint == resumed_checkpoint);
    }
}

TEST_CASE("Private WENO capture exports conservative fluxes on the selected backend",
          "[core][nesting][private-feedback][native]")
{
    const char*       selected = std::getenv("NIMBUS_PRIVATE_FEEDBACK_BACKEND");
    const std::string backend  = selected ? selected : "cpu";
    CAPTURE(backend);
    SimulationStorageState storage{{10, 9, 6, 100, 100, .2, 100}};
    storage.grid.geometry.initialize(10, 9, 6, 100, 100, 100, CoordinateSystem::Cartesian, StaggerType::CGrid);
    storage.configuration.coordinate_system = CoordinateSystem::Cartesian;
    storage.configuration.stagger_type      = StaggerType::CGrid;
    fill_balanced_primary(storage.fields, 10, 9, 6);
    storage.fields.rho.fill(1);
    storage.fields.u.fill(2);
    storage.fields.v.fill(1);
    auto& runtime            = storage.runtime.advection;
    runtime.config.scheme_id = "weno5";
    runtime.reset_for_shape(10, 9, 6);
    auto scheme = tmv::create_microphysics_scheme("milbrandt", storage.configuration.microphysics_config);
    scheme->prepare_prognostic_state(storage.fields.rho,
                                     storage.fields.qc,
                                     storage.fields.qr,
                                     storage.fields.qi,
                                     storage.fields.qs,
                                     storage.fields.qg,
                                     storage.fields.qh);
    const auto           moments = scheme->prognostic_scalars_for_transport();
    std::vector<Field3D> original;
    for (std::size_t f = 0; f < moments.size(); ++f)
    {
        for (int i = 0; i < 10; ++i)
            for (int j = 0; j < 9; ++j)
                for (int k = 0; k < 6; ++k)
                    (*moments[f].field)[i][j][k] = static_cast<float>(1000 * (f + 1) + 3 * i + 7 * j + 2 * k);
        original.push_back(*moments[f].field);
    }
    auto                                view = make_simulation_state_view(storage);
    tmv::CartesianCGridTransportContext context;
    context.begin(view, 10, 9, 6, .2);
    context.accumulate(view, storage.fields.rho, storage.fields.u, storage.fields.v, storage.fields.w, .2);
    context.complete(view);
    runtime.private_interface.configure(10, 9, 6, {1, 9, 1, 8, 0, 6}, 1, moments);
    tmv::advect_microphysics_prognostics_3d(view, *scheme, .2, &context, &runtime);
    std::vector<Field3D> reference;
    for (const auto& moment : moments)
        reference.push_back(*moment.field);
    const auto reference_capture = runtime.private_interface;
    for (std::size_t f = 0; f < moments.size(); ++f)
        *moments[f].field = original[f];
    runtime.private_interface.reset_interval();
    if (backend != "cpu")
    {
        storage.runtime.compute.config.backend        = backend;
        storage.runtime.compute.config.allow_fallback = false;
        std::string error;
        const bool  initialized =
            initialize_compute_backend_runtime(storage.runtime.compute, CoordinateSystem::Cartesian, error);
        INFO(error);
        REQUIRE(initialized);
        REQUIRE_FALSE(storage.runtime.compute.fallback_active);
        storage.configuration.admission.native_weno5 = true;
    }
    tmv::advect_microphysics_prognostics_3d(view, *scheme, .2, &context, &runtime);
    for (std::size_t f = 0; f < moments.size(); ++f)
    {
        const auto& report = runtime.private_interface.fields[f];
        REQUIRE(report.boundary.complete);
        CHECK(report.flux.scalar().contribution_count > 0);
        CHECK(report.flux.scalar().accumulated_dt_s == Approx(.2));
        long double change = 0;
        for (int i = 1; i < 9; ++i)
            for (int j = 1; j < 8; ++j)
                for (int k = 0; k < 6; ++k)
                {
                    const float actual = (*moments[f].field)[i][j][k];
                    CHECK(actual >= 0);
                    CHECK(actual == Approx(reference[f][i][j][k]).margin(.001));
                    if (i > 2 && i < 7 && j > 2 && j < 6)
                        CHECK(actual == Approx(original[f][i][j][k] - .026).margin(.001));
                    change += (static_cast<double>(actual) - original[f][i][j][k]) * 1000000.0;
                }
        CHECK(report.boundary.committed_state_change == Approx(static_cast<double>(change)).margin(.01));
        CHECK(report.boundary.flux_implied_state_change ==
              Approx(reference_capture.fields[f].boundary.flux_implied_state_change).margin(10000));
        CHECK(report.boundary.closure_residual ==
              Approx(report.boundary.committed_state_change - report.boundary.flux_implied_state_change).margin(.01));
    }
}

TEST_CASE("Completed cloud reaction provenance expires with its timestep and state owner",
          "[core][nesting][private-feedback][reaction-provenance]")
{
    ProcessStateRestore restore;
    global_simulation_storage = make_parent();
    auto& storage             = global_simulation_storage;
    REQUIRE(initialize_parent_modules());
    storage.time_s = 0;
    storage.fields.p.fill(100000);
    storage.fields.theta.fill(300);
    storage.fields.rho.fill(1);
    storage.fields.qv.fill(.001f);
    storage.fields.qc.fill(1e-6f);
    for (auto* field : {&storage.tendencies.dtheta_dt_rad,
                        &storage.tendencies.dtheta_dt_pbl,
                        &storage.tendencies.dqv_dt_pbl,
                        &storage.tendencies.dqc_dt_pbl,
                        &storage.tendencies.dqi_dt_pbl})
    {
        field->resize(storage.grid.nr, storage.grid.nth, storage.grid.nz, 0);
    }
    for (auto* field :
         {&storage.fields.qr, &storage.fields.qi, &storage.fields.qs, &storage.fields.qg, &storage.fields.qh})
    {
        field->fill(0);
    }
    tmv::initialize_microphysics(storage.runtime.microphysics, "thompson", storage.configuration.microphysics_config);
    auto& scheme = *storage.runtime.microphysics.scheme;
    scheme.prepare_prognostic_state(storage.fields.rho,
                                    storage.fields.qc,
                                    storage.fields.qr,
                                    storage.fields.qi,
                                    storage.fields.qs,
                                    storage.fields.qg,
                                    storage.fields.qh);
    auto       view   = make_simulation_state_view(storage);
    const auto before = scheme.portable_state();
    REQUIRE(tmv::completed_cloud_evaporation_return(view, .5).empty());
    tmv::step_microphysics(view, .5);
    REQUIRE(tmv::completed_cloud_evaporation_return(view, .5).empty());
    storage.time_s      = .5;
    const auto returned = tmv::completed_cloud_evaporation_return(view, .5);
    REQUIRE_FALSE(returned.empty());
    REQUIRE(returned(4, 4, 1) > 0);
    CHECK(returned(4, 4, 1) == before.fields[0](4, 4, 1));
    CHECK(tmv::completed_cloud_evaporation_return(view, .25).empty());
    storage.time_s = 1;
    CHECK(tmv::completed_cloud_evaporation_return(view, .5).empty());
    storage.time_s                 = .5;
    auto        particles          = scheme.prognostic_scalars_for_transport();
    const float aerosol            = (*particles[3].field)(4, 4, 1);
    (*particles[3].field)(4, 4, 1) = 0;
    CHECK(tmv::completed_cloud_evaporation_return(view, .5)(4, 4, 1) == 0);
    (*particles[3].field)(4, 4, 1) = std::numeric_limits<float>::max();
    CHECK(tmv::completed_cloud_evaporation_return(view, .5)(4, 4, 1) == 0);
    (*particles[3].field)(4, 4, 1)      = aerosol;
    auto original_scheme                = std::move(storage.runtime.microphysics.scheme);
    storage.runtime.microphysics.scheme = original_scheme->clone();
    REQUIRE(storage.runtime.microphysics.scheme);
    CHECK(tmv::completed_cloud_evaporation_return(view, .5).empty());
    storage.runtime.microphysics.scheme = std::move(original_scheme);
    REQUIRE_FALSE(tmv::completed_cloud_evaporation_return(view, .5).empty());
    tmv::invalidate_microphysics_reaction_snapshot(storage.runtime.microphysics);
    CHECK(tmv::completed_cloud_evaporation_return(view, .5).empty());
}

namespace
{
class LateScalarFailureBackend final : public ComputeBackend, public AdvectionComputeKernels
{
public:
    explicit LateScalarFailureBackend(std::unique_ptr<ComputeBackend> real)
        : real_(std::move(real))
    {
    }

    std::string name() const override
    {
        return real_->name();
    }

    bool initialize(std::string&) override
    {
        return true;
    }

    void shutdown() override
    {
        real_->shutdown();
    }

    ComputeMemoryProperties memory_properties() const override
    {
        return real_->memory_properties();
    }

    std::uint64_t successful_dispatch_count() const override
    {
        return real_->successful_dispatch_count();
    }

    std::uint64_t successful_kernel_dispatch_count(nimbus::compute::KernelId kernel) const override
    {
        return real_->successful_kernel_dispatch_count(kernel);
    }

    AdvectionComputeKernels* advection_kernels() override
    {
        return this;
    }

    const AdvectionComputeKernels* advection_kernels() const override
    {
        return this;
    }

    DynamicsComputeKernels* dynamics_kernels() override
    {
        return real_->dynamics_kernels();
    }

    const DynamicsComputeKernels* dynamics_kernels() const override
    {
        return real_->dynamics_kernels();
    }

    TerrainComputeKernels* terrain_kernels() override
    {
        return real_->terrain_kernels();
    }

    const TerrainComputeKernels* terrain_kernels() const override
    {
        return real_->terrain_kernels();
    }

    MicrophysicsComputeKernels* microphysics_kernels() override
    {
        return real_->microphysics_kernels();
    }

    const MicrophysicsComputeKernels* microphysics_kernels() const override
    {
        return real_->microphysics_kernels();
    }

    TurbulenceComputeKernels* turbulence_kernels() override
    {
        return real_->turbulence_kernels();
    }

    const TurbulenceComputeKernels* turbulence_kernels() const override
    {
        return real_->turbulence_kernels();
    }

    DiffusionComputeKernels* diffusion_kernels() override
    {
        return real_->diffusion_kernels();
    }

    const DiffusionComputeKernels* diffusion_kernels() const override
    {
        return real_->diffusion_kernels();
    }

    BoundaryLayerComputeKernels* boundary_layer_kernels() override
    {
        return real_->boundary_layer_kernels();
    }

    const BoundaryLayerComputeKernels* boundary_layer_kernels() const override
    {
        return real_->boundary_layer_kernels();
    }

    RadiationComputeKernels* radiation_kernels() override
    {
        return real_->radiation_kernels();
    }

    const RadiationComputeKernels* radiation_kernels() const override
    {
        return real_->radiation_kernels();
    }

    bool supports(AdvectionKernelOp operation) const override
    {
        return real_->advection_kernels()->supports(operation);
    }

    bool supports_supplied_cartesian_cgrid_mass_flux() const override
    {
        return real_->advection_kernels()->supports_supplied_cartesian_cgrid_mass_flux();
    }

    bool dispatch_vertical_flux(const VerticalFluxDispatch& dispatch) override
    {
        return real_->advection_kernels()->dispatch_vertical_flux(dispatch);
    }

    bool dispatch_advect_scalars_conservative_cartesian_cgrid(
        const ConservativeCartesianCgridTransportDispatch& dispatch) override
    {
        const bool completed = real_->advection_kernels()->dispatch_advect_scalars_conservative_cartesian_cgrid(dispatch);
        if (completed)
        {
            ++completed_scalar_batches;
            if (inject_failure)
            {
                if (completed_scalar_batches >= fail_after_batch)
                {
                    if (before_injected_throw)
                    {
                        before_injected_throw();
                    }
                    throw std::runtime_error("injected failure after native scalar dispatch");
                }
            }
        }
        return completed;
    }

    bool dispatch_advection_x(const ScalarAdvectionDispatch& dispatch) override
    {
        return real_->advection_kernels()->dispatch_advection_x(dispatch);
    }

    bool dispatch_advection_y(const ScalarAdvectionDispatch& dispatch) override
    {
        return real_->advection_kernels()->dispatch_advection_y(dispatch);
    }

    bool dispatch_advection_batch_pre_vertical(const AdvectionBatchPreVerticalDispatch& dispatch) override
    {
        return real_->advection_kernels()->dispatch_advection_batch_pre_vertical(dispatch);
    }

    bool dispatch_advection_batch_post_vertical(const AdvectionBatchPostVerticalDispatch& dispatch) override
    {
        return real_->advection_kernels()->dispatch_advection_batch_post_vertical(dispatch);
    }

    void begin_resident_step() override
    {
        real_->begin_resident_step();
    }

    void invalidate_resident_field(const char* field) noexcept override
    {
        real_->invalidate_resident_field(field);
    }

    void invalidate_all_resident_fields() noexcept override
    {
        real_->invalidate_all_resident_fields();
    }

    int completed_scalar_batches = 0;
    bool inject_failure = true;
    int fail_after_batch = 1;
    std::function<void()> before_injected_throw;

private:
    std::unique_ptr<ComputeBackend> real_;
};

SimulationStorageState flat_native_failure_state(bool east_radiative, bool mixed_open = false)
{
    constexpr int n = 18;
    constexpr double dx = 500, dz = 250, dt = .1;
    auto state = make_parent(n, n);
    state.grid = GridStorageState(n, n, n, dx, dz, dt, dx);
    state.grid.geometry.initialize(n, n, n, dx, dz, dx, CoordinateSystem::Cartesian, StaggerType::CGrid);
    state.process_grid_metrics.dx = dx;
    state.process_grid_metrics.dy = dx;
    state.process_grid_metrics.dz.assign(n, dz);
    state.process_grid_metrics.z_int.resize(n + 1);
    for (int k = 0; k <= n; ++k)
    {
        state.process_grid_metrics.z_int[k] = k * dz;
    }
    state.time_s = 0;
    resize_primary(state.fields, n, n, n);
    tmv::fields::visit(state.tendencies, tmv::fields::physics_tendencies,
                       [&](const char*, Field3D& field)
                       {
                           field.resize(n, n, n, 0);
                       });
    const float pressure = float(tmv::dynamics_eos::pressure_fp64(1.1, 300.0));
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            for (int k = 0; k < n; ++k)
            {
                const float corner = .002f * std::exp(-.5f *
                    ((i - 15) * (i - 15) + (j - 2) * (j - 2) + .25f * (k - 6) * (k - 6)));
                state.fields.rho(i, j, k) = 1.1f;
                state.fields.p(i, j, k) = pressure;
                state.fields.theta(i, j, k) = 300.0f + corner;
                state.fields.qv(i, j, k) = .004f;
                state.fields.u(i, j, k) = .05f;
            }
        }
    }
    state.base.rho0.assign(n, 1.1);
    state.base.theta0.assign(n, 300.0);
    state.base.p0.assign(n, pressure);
    state.base.qv0.assign(n, .004);
    state.base.u0.assign(n, .05);
    state.base.v0.assign(n, 0);
    for (auto* values : {&state.surface.ustar, &state.surface.tau_u, &state.surface.tau_v,
                         &state.surface.sensible_heat_flux, &state.surface.latent_heat_flux,
                         &state.surface.moisture_flux, &state.surface.pbl_height,
                         &state.surface.precipitation_rate_mm_h, &state.surface.accumulated_rainfall_mm,
                         &state.surface.accumulated_rainfall_residual_mm})
    {
        values->assign(n * n, 0);
    }
    using L = tmv::init::config::LateralBC;
    const bool open_top = east_radiative || mixed_open;
    state.configuration.lateral_bc_west = open_top ? L::RigidSlip : L::Rigid;
    state.configuration.lateral_bc_east = east_radiative ? L::Radiative : open_top ? L::RigidSlip : L::Rigid;
    state.configuration.lateral_bc_south = open_top ? L::RigidSlip : L::Rigid;
    state.configuration.lateral_bc_north = open_top ? L::RigidSlip : L::Rigid;
    state.configuration.bottom_bc = open_top
        ? tmv::init::config::BottomBC::RigidNoSlip
        : tmv::init::config::BottomBC::RigidSlip;
    state.configuration.top_bc = open_top
        ? tmv::init::config::TopBC::Open
        : tmv::init::config::TopBC::Rigid;
    state.configuration.admission.native_dry_tvd = true;
    return state;
}

std::vector<const char*> changed_primary(const PrognosticFieldStorage& left,
                                         const PrognosticFieldStorage& right)
{
    const std::array<const Field3D*, 15> lhs{
        &left.rho, &left.p, &left.u, &left.v, &left.w, &left.theta, &left.qv,
        &left.qc, &left.qr, &left.qi, &left.qs, &left.qg, &left.qh, &left.tracer, &left.tke};
    const std::array<const Field3D*, 15> rhs{
        &right.rho, &right.p, &right.u, &right.v, &right.w, &right.theta, &right.qv,
        &right.qc, &right.qr, &right.qi, &right.qs, &right.qg, &right.qh, &right.tracer, &right.tke};
    const std::array<const char*, 15> names{
        "rho", "p", "u", "v", "w", "theta", "qv", "qc", "qr", "qi", "qs", "qg", "qh", "tracer", "tke"};
    std::vector<const char*> changed;
    for (std::size_t index = 0; index < lhs.size(); ++index)
    {
        if (lhs[index]->size() != rhs[index]->size() ||
            std::memcmp(lhs[index]->data(), rhs[index]->data(), lhs[index]->size() * sizeof(float)) != 0)
        {
            changed.push_back(names[index]);
        }
    }
    return changed;
}

void exercise_flat_native_late_failure(bool mixed_open)
{
    const int previous_threads = omp_get_max_threads();
    struct RestoreThreads
    {
        int count;
        ~RestoreThreads()
        {
            omp_set_num_threads(count);
        }
    } restore_threads{previous_threads};
    omp_set_num_threads(1);
    for (bool east_radiative : {false, true})
    {
        if (mixed_open && east_radiative)
        {
            continue;
        }
        INFO("east_radiative=" << east_radiative);
        INFO("mixed_open=" << mixed_open);
        ProcessStateRestore restore;
        global_simulation_storage = flat_native_failure_state(east_radiative, mixed_open);
        auto& state = global_simulation_storage;
        const auto initialize_fixed_modules = [&]()
        {
            REQUIRE(initialize_parent_modules("metal"));
            tmv::BoundaryLayerConfig pbl;
            pbl.scheme_id = "none";
            pbl.apply_surface_fluxes = false;
            initialize_boundary_layer(state.runtime.boundary_layer, "none", pbl, SurfaceConfig{});
            tmv::TimeSteppingConfig time;
            time.scheme_id = "split_explicit";
            time.split_acoustic = true;
            time.n_acoustic_substeps = 6;
            initialize_time_stepping(state.runtime.time_stepping, "split_explicit", time);
            state.runtime.time_stepping.config.max_acoustic_speed_ms = 350;
            state.runtime.time_stepping.config.min_acoustic_spacing_m = 250;
        };
        initialize_fixed_modules();
        REQUIRE_FALSE(state.runtime.compute.fallback_active);
        REQUIRE(state.runtime.compute.backend->name() == "metal");
        if (mixed_open)
        {
            REQUIRE(state.configuration.admission.native_dry_tvd);
            REQUIRE(state.configuration.lateral_bc_west == tmv::init::config::LateralBC::RigidSlip);
            REQUIRE(state.configuration.lateral_bc_east == tmv::init::config::LateralBC::RigidSlip);
            REQUIRE(state.configuration.lateral_bc_south == tmv::init::config::LateralBC::RigidSlip);
            REQUIRE(state.configuration.lateral_bc_north == tmv::init::config::LateralBC::RigidSlip);
            REQUIRE(state.configuration.bottom_bc == tmv::init::config::BottomBC::RigidNoSlip);
            REQUIRE(state.configuration.top_bc == tmv::init::config::TopBC::Open);
        }
        auto* boundary_operator_before = state.runtime.boundary_conditions.execution_operator.get();
        REQUIRE(boundary_operator_before);
        auto real_backend = std::move(state.runtime.compute.backend);
        auto faulting = std::make_unique<LateScalarFailureBackend>(std::move(real_backend));
        auto* injected = faulting.get();
        state.runtime.compute.backend = std::move(faulting);
        const auto primary_before = state.fields;
        const auto wave_before = static_cast<const tmv::BoundaryConditionWaveState&>(state.runtime.boundary_conditions);
        const auto acoustic_steps_before = state.runtime.time_stepping.acoustic_energy_step_count;
        const auto mass_steps_before = state.runtime.time_stepping.boundary_mass_step_count;
        const auto scalar_steps_before = state.runtime.advection.thermodynamic_boundary_report_step_count;
        const auto acoustic_report_before = state.runtime.time_stepping.last_acoustic_energy;
        const auto mass_report_before = state.runtime.time_stepping.last_boundary_mass;
        const auto scalar_reports_before = state.runtime.advection.thermodynamic_boundary_reports;
        std::vector<std::uint8_t> checkpoint_before, checkpoint_after;
        std::string error;
        REQUIRE(tmv::checkpoint::serialize_simulation_storage(state, checkpoint_before, error));
        const auto budget_file = std::filesystem::temp_directory_path() /
            (std::string("nimbus_flat_native_step_failure_") +
             (east_radiative ? "radiative_" : mixed_open ? "mixed_open_" : "canonical_") +
             std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".csv");
        open_budget_csv(budget_file.string());
        CHECK_THROWS_WITH(tmv::step_dynamics(make_simulation_state_view(state), 0),
                          Catch::Matchers::Contains("injected failure after native scalar dispatch"));
        close_budget_csv();
        std::ifstream budget_stream(budget_file);
        const auto budget_rows = std::count(std::istreambuf_iterator<char>(budget_stream),
                                            std::istreambuf_iterator<char>(), '\n') - 1;
        std::filesystem::remove(budget_file);
        REQUIRE(injected->completed_scalar_batches == 1);
        REQUIRE(tmv::checkpoint::serialize_simulation_storage(state, checkpoint_after, error));
        const auto changed = changed_primary(state.fields, primary_before);
        const bool west_wave_same = state.runtime.boundary_conditions.radiative_u_west == wave_before.radiative_u_west;
        const bool east_wave_same = state.runtime.boundary_conditions.radiative_u_east == wave_before.radiative_u_east;
        const bool south_wave_same = state.runtime.boundary_conditions.radiative_v_south == wave_before.radiative_v_south;
        const bool north_wave_same = state.runtime.boundary_conditions.radiative_v_north == wave_before.radiative_v_north;
        const bool checkpoint_same = checkpoint_after == checkpoint_before;
        const auto first_checkpoint_difference = std::mismatch(checkpoint_before.begin(), checkpoint_before.end(),
                                                                checkpoint_after.begin(), checkpoint_after.end()).first -
            checkpoint_before.begin();
        const std::vector<std::uint8_t> before_payload(checkpoint_before.begin() + 28, checkpoint_before.end());
        const std::vector<std::uint8_t> after_payload(checkpoint_after.begin() + 28, checkpoint_after.end());
        std::vector<tmv::checkpoint::wire::SectionView> before_sections, after_sections;
        REQUIRE(tmv::checkpoint::wire::decode_sections(before_payload, before_sections, error));
        REQUIRE(tmv::checkpoint::wire::decode_sections(after_payload, after_sections, error));
        REQUIRE(before_sections.size() == after_sections.size());
        for (std::size_t section = 0; section < before_sections.size(); ++section)
        {
            const auto& before = before_sections[section];
            const auto& after = after_sections[section];
            REQUIRE(before.id == after.id);
            if (before.payload.size() != after.payload.size() ||
                std::memcmp(before.payload.data(), after.payload.data(),
                            std::min(before.payload.size(), after.payload.size())) != 0)
            {
                const auto payload_difference = std::mismatch(before.payload.data(),
                    before.payload.data() + std::min(before.payload.size(), after.payload.size()),
                    after.payload.data()).first - before.payload.data();
                std::cout << "[flat-native-checkpoint-drift] east_radiative=" << east_radiative
                          << " section_id=" << before.id << " version=" << before.version
                          << " before_bytes=" << before.payload.size()
                          << " after_bytes=" << after.payload.size()
                          << " first_section_byte=" << payload_difference << '\n';
            }
        }
        const bool acoustic_report_same = std::memcmp(&state.runtime.time_stepping.last_acoustic_energy,
            &acoustic_report_before, sizeof(acoustic_report_before)) == 0;
        const bool mass_report_same = std::memcmp(&state.runtime.time_stepping.last_boundary_mass,
            &mass_report_before, sizeof(mass_report_before)) == 0;
        const bool scalar_reports_same = std::memcmp(&state.runtime.advection.thermodynamic_boundary_reports,
            &scalar_reports_before, sizeof(scalar_reports_before)) == 0;
        std::cout << "[flat-native-late-failure] east_radiative=" << east_radiative
                  << " mixed_open=" << mixed_open
                  << " changed_primary=" << changed.size() << ":";
        for (const char* name : changed)
        {
            std::cout << name << ',';
        }
        std::cout
                  << " waves=" << state.runtime.boundary_conditions.radiative_u_west.size() << '/'
                  << state.runtime.boundary_conditions.radiative_u_east.size() << '/'
                  << state.runtime.boundary_conditions.radiative_v_south.size() << '/'
                  << state.runtime.boundary_conditions.radiative_v_north.size()
                  << " acoustic_steps=" << state.runtime.time_stepping.acoustic_energy_step_count
                  << " mass_steps=" << state.runtime.time_stepping.boundary_mass_step_count
                  << " scalar_steps=" << state.runtime.advection.thermodynamic_boundary_report_step_count
                  << " budget_rows=" << budget_rows
                  << " checkpoint_first_diff=" << first_checkpoint_difference << '\n';
        CHECK(changed.empty());
        CHECK(west_wave_same);
        CHECK(east_wave_same);
        CHECK(south_wave_same);
        CHECK(north_wave_same);
        CHECK(state.runtime.time_stepping.acoustic_energy_step_count == acoustic_steps_before);
        CHECK(state.runtime.time_stepping.boundary_mass_step_count == mass_steps_before);
        CHECK(state.runtime.advection.thermodynamic_boundary_report_step_count == scalar_steps_before);
        CHECK(acoustic_report_same);
        CHECK(mass_report_same);
        CHECK(scalar_reports_same);
        CHECK(checkpoint_same);
        CHECK(budget_rows == 0);
        CHECK(state.runtime.boundary_conditions.execution_operator.get() == boundary_operator_before);

        injected->inject_failure = false;
        REQUIRE_NOTHROW(tmv::step_dynamics(make_simulation_state_view(state), 0));
        CHECK(injected->completed_scalar_batches == 2);
        const auto recovered_fields = state.fields;
        const auto recovered_waves =
            static_cast<const tmv::BoundaryConditionWaveState&>(state.runtime.boundary_conditions);
        global_simulation_storage = flat_native_failure_state(east_radiative, mixed_open);
        initialize_fixed_modules();
        REQUIRE_NOTHROW(tmv::step_dynamics(make_simulation_state_view(state), 0));
        CHECK(changed_primary(state.fields, recovered_fields).empty());
        const bool west_continuation_same =
            state.runtime.boundary_conditions.radiative_u_west == recovered_waves.radiative_u_west;
        const bool east_continuation_same =
            state.runtime.boundary_conditions.radiative_u_east == recovered_waves.radiative_u_east;
        const bool south_continuation_same =
            state.runtime.boundary_conditions.radiative_v_south == recovered_waves.radiative_v_south;
        const bool north_continuation_same =
            state.runtime.boundary_conditions.radiative_v_north == recovered_waves.radiative_v_north;
        CHECK(west_continuation_same);
        CHECK(east_continuation_same);
        CHECK(south_continuation_same);
        CHECK(north_continuation_same);
    }
}

void exercise_headless_native_late_failure(bool mixed_open)
{
    ProcessStateRestore restore;
    const auto root = std::filesystem::temp_directory_path() /
        (std::string(mixed_open ? "nimbus_headless_mixed_open_failure_" : "nimbus_headless_native_failure_") +
         std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    REQUIRE(std::filesystem::create_directories(root));
    const auto fixture = root / (mixed_open ? "mixed-open.yaml" : "east-radiative.yaml");
    std::ifstream maintained("tests/fixtures/reference/catalog/flat_mixed_boundary_metal_exploratory_v1.yaml");
    REQUIRE(maintained.good());
    std::string source(std::istreambuf_iterator<char>{maintained}, {});
    const auto replace_once = [&](const std::string& previous, const std::string& next)
    {
        const auto position = source.find(previous);
        REQUIRE(position != std::string::npos);
        REQUIRE(source.find(previous, position + previous.size()) == std::string::npos);
        source.replace(position, previous.size(), next);
    };
    if (!mixed_open)
    {
        replace_once("    east: rigid_slip\n", "    east: radiative\n");
        replace_once("    center_x_m: 4500\n", "    center_x_m: 8000\n");
        replace_once("    center_y_m: 4500\n", "    center_y_m: 1000\n");
    }
    std::ofstream(fixture) << source;
    const std::array<const char*, 6> ledgers{
        "budget.csv", "energy_budget.csv", "moisture_budget.csv",
        "momentum_budget.csv", "radiation_budget.csv", "thermodynamic_source_budget.csv"};
    const auto row_counts = [&](const std::filesystem::path& outdir)
    {
        std::array<int, 6> counts{};
        for (std::size_t index = 0; index < ledgers.size(); ++index)
        {
            std::ifstream stream(outdir / ledgers[index]);
            REQUIRE(stream.good());
            counts[index] = int(std::count(std::istreambuf_iterator<char>(stream),
                                           std::istreambuf_iterator<char>(), '\n')) - 1;
        }
        return counts;
    };
    const auto durable_members = [&](const std::filesystem::path& outdir)
    {
        std::map<std::string, std::vector<char>> members;
        if (!std::filesystem::exists(outdir))
        {
            return members;
        }
        for (const auto& entry : std::filesystem::recursive_directory_iterator(outdir))
        {
            if (!entry.is_regular_file())
            {
                continue;
            }
            const std::string relative = std::filesystem::relative(entry.path(), outdir).generic_string();
            if (relative.rfind("step_", 0) != 0 && relative.rfind("checkpoints/", 0) != 0 &&
                relative != "history_index.json")
            {
                continue;
            }
            std::ifstream stream(entry.path(), std::ios::binary);
            members[relative] = std::vector<char>(std::istreambuf_iterator<char>(stream), {});
        }
        return members;
    };
    const auto run_case = [&](const std::filesystem::path& outdir, int duration,
                              std::function<void(SimulationStorageState&)> hook)
    {
        shutdown_compute_backend_runtime();
        global_simulation_storage = SimulationStorageState{};
        int selected_duration = duration;
        int write_every = 1;
        std::string path = outdir.string();
        OutputConfig output;
        load_config(fixture.string(), selected_duration, write_every, path, &output,
                    nullptr, nullptr, -1, -1, nullptr, false, nullptr, nullptr, duration);
        initialize_microphysics(global_microphysics_scheme, global_microphysics_config);
        initialize_radar_products(global_radar_product_selection);
        initialize_terrain(global_terrain_config.scheme_id, global_terrain_config);
        initialize();
        initialize_numerics();
        tmv::chaos::initialize_chaos(global_simulation_state(), tmv::chaos::global_chaos_config);
        tmv::chaos::apply_chaos_initial_conditions(global_simulation_state());
        apply_initial_thermodynamic_closure(global_simulation_state());
        tmv::initialize_dynamics(global_dynamics_scheme_name);
        initialize_radiation(global_radiation_config.scheme_id, global_radiation_config);
        initialize_boundary_layer(global_boundary_layer_config.scheme_id,
                                  global_boundary_layer_config, global_surface_config);
        initialize_turbulence(global_turbulence_config.scheme_id, global_turbulence_config);
        apply_boundary_conditions(global_simulation_state());
        HeadlessRunOptions options;
        options.duration_s = selected_duration;
        options.write_every_s = write_every;
        options.scientific_diagnostics_every_steps = 1;
        options.checkpoint_every_s = 1;
        options.outdir = path;
        options.output_config = output;
        options.before_time_loop_test_hook = std::move(hook);
        return run_headless_simulation(options);
    };
    const auto reference_out = root / "completed-one-second";
    REQUIRE(run_case(reference_out, 1, {}) == 0);
    const auto expected_rows = row_counts(reference_out);
    CHECK(expected_rows == std::array<int, 6>{90, 10, 10, 10, 10, 10});
    const auto failing_out = root / "failed-two-second";
    std::map<std::string, std::vector<char>> prior_artifacts;
    std::string ready_contract;
    LateScalarFailureBackend* injected = nullptr;
    int hook_calls = 0;
    const auto hook = [&](SimulationStorageState& state)
    {
        ++hook_calls;
        REQUIRE_FALSE(state.runtime.compute.fallback_active);
        REQUIRE(state.runtime.compute.backend->name() == "metal");
        if (mixed_open)
        {
            REQUIRE(state.configuration.admission.native_dry_tvd);
            REQUIRE(state.configuration.lateral_bc_west == tmv::init::config::LateralBC::RigidSlip);
            REQUIRE(state.configuration.lateral_bc_east == tmv::init::config::LateralBC::RigidSlip);
            REQUIRE(state.configuration.lateral_bc_south == tmv::init::config::LateralBC::RigidSlip);
            REQUIRE(state.configuration.lateral_bc_north == tmv::init::config::LateralBC::RigidSlip);
            REQUIRE(state.configuration.bottom_bc == tmv::init::config::BottomBC::RigidNoSlip);
            REQUIRE(state.configuration.top_bc == tmv::init::config::TopBC::Open);
        }
        auto real = std::move(state.runtime.compute.backend);
        REQUIRE(real);
        auto wrapper = std::make_unique<LateScalarFailureBackend>(std::move(real));
        wrapper->fail_after_batch = 11;
        wrapper->before_injected_throw = [&]()
        {
            prior_artifacts = durable_members(failing_out);
            std::ifstream contract(failing_out / "nimbus_output.json");
            ready_contract.assign(std::istreambuf_iterator<char>(contract), {});
        };
        injected = wrapper.get();
        state.runtime.compute.backend = std::move(wrapper);
    };
    CHECK(run_case(failing_out, 2, hook) == 2);
    REQUIRE(hook_calls == 1);
    REQUIRE(injected);
    CHECK(injected->completed_scalar_batches == 11);
    close_budget_csv();
    close_energy_csv();
    close_moisture_csv();
    close_momentum_csv();
    close_radiation_budget_csv();
    close_thermodynamic_source_budget_csv();
    REQUIRE_FALSE(prior_artifacts.empty());
    const bool prior_artifacts_identical = durable_members(failing_out) == prior_artifacts;
    CHECK(prior_artifacts_identical);
    CHECK(row_counts(failing_out) == expected_rows);
    const auto read_contract = [](const std::filesystem::path& outdir)
    {
        std::ifstream stream(outdir / "nimbus_output.json");
        return std::string(std::istreambuf_iterator<char>(stream), {});
    };
    const auto generation = [](const std::string& document)
    {
        std::smatch match;
        const std::regex pattern(R"GEN("publication_generation": "([^"]+)")GEN");
        return std::regex_search(document, match, pattern) ? match[1].str() : std::string{};
    };
    const std::string failed_contract = read_contract(failing_out);
    CHECK(ready_contract.find("\"publication_state\": \"ready\"") != std::string::npos);
    CHECK(failed_contract.find("\"publication_state\": \"failed\"") != std::string::npos);
    CHECK_FALSE(generation(ready_contract).empty());
    CHECK(generation(failed_contract) == generation(ready_contract));
    CHECK(failed_contract != ready_contract);
    const auto hook_out = root / "post-ready-hook-failure";
    std::string hook_ready_contract;
    const auto rejected_hook = [&](SimulationStorageState&)
    {
        hook_ready_contract = read_contract(hook_out);
        throw std::runtime_error("injected post-ready hook failure");
    };
    CHECK(run_case(hook_out, 1, rejected_hook) == 2);
    const std::string hook_failed_contract = read_contract(hook_out);
    CHECK(hook_ready_contract.find("\"publication_state\": \"ready\"") != std::string::npos);
    CHECK(hook_failed_contract.find("\"publication_state\": \"failed\"") != std::string::npos);
    CHECK_FALSE(generation(hook_ready_contract).empty());
    CHECK(generation(hook_failed_contract) == generation(hook_ready_contract));
    close_budget_csv();
    close_energy_csv();
    close_moisture_csv();
    close_momentum_csv();
    close_radiation_budget_csv();
    close_thermodynamic_source_budget_csv();
    std::cout << "[headless-flat-native-failure] path=" << root.string()
              << " mixed_open=" << mixed_open
              << " prior_artifacts=" << prior_artifacts.size()
              << " failed_step_artifacts_unchanged=" << prior_artifacts_identical
              << " budget_rows=" << row_counts(failing_out)[0]
              << " terminal_contract=failed hook_contract=failed" << '\n';
}
}

TEST_CASE("Flat canonical and east-radiative native steps roll back a late scalar failure",
          "[metal][native_dry_tvd][full_step_rollback]")
{
    exercise_flat_native_late_failure(false);
}

TEST_CASE("Flat mixed-open native step rolls back a late scalar failure",
          "[metal][native_dry_tvd][mixed_open][full_step_rollback]")
{
    exercise_flat_native_late_failure(true);
}

TEST_CASE("Headless east-radiative late scalar failure publishes no failed-step artifacts",
          "[metal][native_dry_tvd][headless_step_rollback]")
{
    exercise_headless_native_late_failure(false);
}

TEST_CASE("Headless mixed-open late scalar failure publishes no failed-step artifacts",
          "[metal][native_dry_tvd][mixed_open][headless_step_rollback]")
{
    exercise_headless_native_late_failure(true);
}

TEST_CASE("Static vertical-only CPU step rolls back a post-SGS diffusion failure",
          "[turbulence][vertical_only][cpu][late_failure]")
{
    const char* fixture_env = std::getenv("NIMBUS_TEST_VERTICAL_ONLY_STATIC_CONFIG");
    const char* root_env = std::getenv("NIMBUS_TEST_VERTICAL_ONLY_FAILURE_ROOT");
    if (fixture_env == nullptr && root_env == nullptr)
    {
        return;
    }
    REQUIRE(fixture_env != nullptr);
    REQUIRE(root_env != nullptr);
    ProcessStateRestore restore;
    omp_set_num_threads(1);
    const std::filesystem::path root(root_env);
    REQUIRE(std::filesystem::create_directory(root));
    const std::array<const char*, 6> ledgers{
        "budget.csv", "energy_budget.csv", "moisture_budget.csv",
        "momentum_budget.csv", "radiation_budget.csv",
        "thermodynamic_source_budget.csv"};
    const auto close_ledgers = []()
    {
        close_budget_csv();
        close_energy_csv();
        close_moisture_csv();
        close_momentum_csv();
        close_radiation_budget_csv();
        close_thermodynamic_source_budget_csv();
    };
    const auto read_bytes = [](const std::filesystem::path& path)
    {
        std::ifstream stream(path, std::ios::binary);
        return std::vector<char>(std::istreambuf_iterator<char>(stream), {});
    };
    const auto durable_members = [&](const std::filesystem::path& outdir)
    {
        std::map<std::string, std::vector<char>> members;
        if (!std::filesystem::exists(outdir))
        {
            return members;
        }
        for (const auto& entry : std::filesystem::recursive_directory_iterator(outdir))
        {
            if (!entry.is_regular_file())
            {
                continue;
            }
            const auto relative = std::filesystem::relative(entry.path(), outdir).generic_string();
            if (relative.rfind("step_", 0) == 0 ||
                relative.rfind("checkpoints/", 0) == 0 ||
                relative == "history_index.json")
            {
                members.emplace(relative, read_bytes(entry.path()));
            }
        }
        return members;
    };
    const auto run_case = [&](const std::filesystem::path& outdir, int duration,
                              std::function<void(SimulationStorageState&)> hook)
    {
        close_ledgers();
        shutdown_compute_backend_runtime();
        global_simulation_storage = SimulationStorageState{};
        int selected_duration = duration;
        int write_every = 1;
        std::string path = outdir.string();
        OutputConfig output;
        load_config(fixture_env, selected_duration, write_every, path, &output,
                    nullptr, nullptr, -1, -1, nullptr, false, nullptr, nullptr,
                    duration);
        initialize_microphysics(global_microphysics_scheme, global_microphysics_config);
        initialize_radar_products(global_radar_product_selection);
        initialize_terrain(global_terrain_config.scheme_id, global_terrain_config);
        initialize();
        initialize_numerics();
        tmv::chaos::initialize_chaos(global_simulation_state(),
                                      tmv::chaos::global_chaos_config);
        tmv::chaos::apply_chaos_initial_conditions(global_simulation_state());
        apply_initial_thermodynamic_closure(global_simulation_state());
        tmv::initialize_dynamics(global_dynamics_scheme_name);
        initialize_radiation(global_radiation_config.scheme_id, global_radiation_config);
        initialize_boundary_layer(global_boundary_layer_config.scheme_id,
                                  global_boundary_layer_config, global_surface_config);
        initialize_turbulence(global_turbulence_config.scheme_id,
                              global_turbulence_config);
        apply_boundary_conditions(global_simulation_state());
        HeadlessRunOptions options;
        options.duration_s = selected_duration;
        options.write_every_s = write_every;
        options.scientific_diagnostics_every_steps = 1;
        options.checkpoint_every_s = 1;
        options.outdir = path;
        options.output_config = output;
        options.before_time_loop_test_hook = std::move(hook);
        return run_headless_simulation(options);
    };

    const auto initialized = root / "initialized_only";
    REQUIRE(run_case(initialized, 0, {}) == 0);
    close_ledgers();
    std::array<std::vector<char>, 6> initial_ledgers;
    for (std::size_t index = 0; index < ledgers.size(); ++index)
    {
        initial_ledgers[index] = read_bytes(initialized / ledgers[index]);
    }
    const auto reference = root / "completed_one_second";
    REQUIRE(run_case(reference, 1, {}) == 0);
    const auto reference_work =
        global_simulation_storage.runtime.turbulence.cumulative_committed_momentum_work_j;
    const auto reference_sgs_steps =
        global_simulation_storage.runtime.turbulence.momentum_work_step_count;
    REQUIRE(reference_work != 0.0);
    REQUIRE(reference_sgs_steps > 0);
    const auto failing = root / "failed_two_second";
    PrognosticFieldStorage prior_fields;
    double prior_work = 0.0;
    std::uint64_t prior_sgs_steps = 0;
    std::map<std::string, std::vector<char>> prior_artifacts;
    std::string ready_contract;
    int hook_calls = 0;
    const auto hook = [&](SimulationStorageState& state)
    {
        ++hook_calls;
        REQUIRE(state.configuration.admission.static_vertical_only_sgs_cpu);
        prior_fields = state.fields;
        prior_work = state.runtime.turbulence.cumulative_committed_momentum_work_j;
        prior_sgs_steps = state.runtime.turbulence.momentum_work_step_count;
        prior_artifacts = durable_members(failing);
        const auto bytes = read_bytes(failing / "nimbus_output.json");
        ready_contract.assign(bytes.begin(), bytes.end());
        state.runtime.diffusion.config.K_h = 1.0;
        state.runtime.diffusion.scheme.reset();
    };
    REQUIRE(run_case(failing, 2, hook) == 2);
    REQUIRE(hook_calls == 1);
    close_ledgers();
    const auto changed = changed_primary(global_simulation_storage.fields,
                                         prior_fields);
    std::cout << "[vertical-only-post-SGS-failure] changed_primary=" << changed.size()
              << " sgs_work_delta=" <<
                    (global_simulation_storage.runtime.turbulence.cumulative_committed_momentum_work_j -
                     prior_work)
              << " sgs_step_delta=" <<
                    (global_simulation_storage.runtime.turbulence.momentum_work_step_count -
                     prior_sgs_steps) << '\n';
    CHECK(changed.empty());
    CHECK(global_simulation_storage.runtime.turbulence.cumulative_committed_momentum_work_j ==
          prior_work);
    CHECK(global_simulation_storage.runtime.turbulence.momentum_work_step_count ==
          prior_sgs_steps);
    CHECK(durable_members(failing) == prior_artifacts);
    for (std::size_t index = 0; index < ledgers.size(); ++index)
    {
        CHECK(read_bytes(failing / ledgers[index]) == initial_ledgers[index]);
    }
    const auto failed_bytes = read_bytes(failing / "nimbus_output.json");
    const std::string failed_contract(failed_bytes.begin(), failed_bytes.end());
    CHECK(ready_contract.find("\"publication_state\": \"ready\"") !=
          std::string::npos);
    CHECK(failed_contract.find("\"publication_state\": \"failed\"") !=
          std::string::npos);
    const auto generation = [](const std::string& document)
    {
        std::smatch match;
        const std::regex pattern(R"GEN("publication_generation": "([^"]+)")GEN");
        return std::regex_search(document, match, pattern)
            ? match[1].str() : std::string{};
    };
    CHECK_FALSE(generation(ready_contract).empty());
    CHECK(generation(failed_contract) == generation(ready_contract));
}

TEST_CASE("Static vertical-only CPU direct step restores state after post-SGS failure",
          "[turbulence][vertical_only][cpu][direct_late_failure]")
{
    const char* fixture_env = std::getenv("NIMBUS_TEST_VERTICAL_ONLY_STATIC_CONFIG");
    if (fixture_env == nullptr)
    {
        return;
    }
    ProcessStateRestore restore;
    omp_set_num_threads(1);
    const auto initialize_state = [&]()
    {
        close_budget_csv();
        close_energy_csv();
        close_moisture_csv();
        close_momentum_csv();
        close_radiation_budget_csv();
        close_thermodynamic_source_budget_csv();
        shutdown_compute_backend_runtime();
        global_simulation_storage = SimulationStorageState{};
        int duration = 2;
        int write_every = 1;
        std::string outdir = "unused_vertical_direct_output";
        OutputConfig output;
        load_config(fixture_env, duration, write_every, outdir, &output,
                    nullptr, nullptr, -1, -1, nullptr, false, nullptr, nullptr, 2);
        initialize_microphysics(global_microphysics_scheme, global_microphysics_config);
        initialize_radar_products(global_radar_product_selection);
        initialize_terrain(global_terrain_config.scheme_id, global_terrain_config);
        initialize();
        initialize_numerics();
        tmv::chaos::initialize_chaos(global_simulation_state(),
                                      tmv::chaos::global_chaos_config);
        tmv::chaos::apply_chaos_initial_conditions(global_simulation_state());
        apply_initial_thermodynamic_closure(global_simulation_state());
        tmv::initialize_dynamics(global_dynamics_scheme_name);
        initialize_radiation(global_radiation_config.scheme_id, global_radiation_config);
        initialize_boundary_layer(global_boundary_layer_config.scheme_id,
                                  global_boundary_layer_config, global_surface_config);
        initialize_turbulence(global_turbulence_config.scheme_id,
                              global_turbulence_config);
        apply_boundary_conditions(global_simulation_state());
        REQUIRE(global_simulation_storage.configuration.admission.static_vertical_only_sgs_cpu);
    };
    const auto prepare_step = [](SimulationStorageState& owner)
    {
        const auto view = make_simulation_state_view(owner);
        const double admitted = tmv::choose_runtime_timestep(view);
        REQUIRE(std::isfinite(admitted));
        REQUIRE(admitted > 0.0);
        owner.grid.dt = std::min(owner.grid.dt, admitted);
        tmv::step_radiation(view, 0.0);
        tmv::step_boundary_layer(view, 0.0);
    };

    initialize_state();
    auto& state = global_simulation_storage;
    prepare_step(state);
    const auto prior_fields = state.fields;
    const auto prior_work = state.runtime.turbulence.cumulative_committed_momentum_work_j;
    const auto prior_steps = state.runtime.turbulence.momentum_work_step_count;
    auto diffusion_scheme = std::move(state.runtime.diffusion.scheme);
    REQUIRE(diffusion_scheme != nullptr);
    state.runtime.diffusion.config.K_h = 1.0;
    REQUIRE_THROWS_WITH(tmv::step_dynamics(make_simulation_state_view(state), 0.0),
                        Catch::Matchers::Contains("active diffusion configuration has no initialized scheme"));
    CHECK(changed_primary(state.fields, prior_fields).empty());
    CHECK(state.runtime.turbulence.cumulative_committed_momentum_work_j == prior_work);
    CHECK(state.runtime.turbulence.momentum_work_step_count == prior_steps);
    state.runtime.diffusion.scheme = std::move(diffusion_scheme);
    state.runtime.diffusion.config.K_h = 0.0;
    REQUIRE_NOTHROW(tmv::step_dynamics(make_simulation_state_view(state), 0.0));
    const auto recovered_fields = state.fields;
    const auto recovered_work = state.runtime.turbulence.cumulative_committed_momentum_work_j;
    const auto recovered_steps = state.runtime.turbulence.momentum_work_step_count;

    initialize_state();
    prepare_step(global_simulation_storage);
    REQUIRE_NOTHROW(tmv::step_dynamics(make_simulation_state_view(global_simulation_storage), 0.0));
    CHECK(changed_primary(global_simulation_storage.fields, recovered_fields).empty());
    CHECK(global_simulation_storage.runtime.turbulence.cumulative_committed_momentum_work_j ==
          recovered_work);
    CHECK(global_simulation_storage.runtime.turbulence.momentum_work_step_count ==
          recovered_steps);
}
