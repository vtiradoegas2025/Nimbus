#include "util/physics_cadence.hpp"
#include "diagnostics/level_budget_csv.hpp"
#include <chrono>
#include "catch2/catch.hpp"
#include "core/runtime/runtime_cadence.hpp"
#include "core/runtime/simulation.hpp"
#include "core/runtime/simulation_runtime.hpp"
#include "core/runtime/simulation_state.hpp"

using tmv::runtime::make_diagnostic_step_plan;
using tmv::runtime::clamp_step_to_deadline;
using tmv::runtime::step_cadence_due;
using tmv::runtime::trajectory_requires_gridded_diagnostics;
using tmv::runtime::live_shm_publish_due;
using tmv::runtime::next_live_shm_publish_time;

namespace
{
class SnapshotDynamicsScheme final : public tmv::DynamicsScheme
{
public:
    explicit SnapshotDynamicsScheme(int generation = 0)
        : generation(generation)
    {
    }
    std::unique_ptr<tmv::DynamicsScheme> clone() const override
    {
        return std::make_unique<SnapshotDynamicsScheme>(*this);
    }
    void compute_momentum_tendencies(
        const Field3D&, const Field3D&, const Field3D&, const Field3D&,
        const Field3D&, const Field3D&, double, Field3D&, Field3D&,
        Field3D&, Field3D&, Field3D&) override
    {
    }
    std::string get_scheme_name() const override { return "snapshot_stub"; }
    std::string get_coordinate_system() const override { return "cartesian"; }
    int get_num_prognostic_vars() const override { return 5; }
    int generation;
};
} // namespace

TEST_CASE("runtime cadence uses one-based step boundaries",
          "[core][runtime][cadence]")
{
    REQUIRE_FALSE(step_cadence_due(0, 4));
    REQUIRE_FALSE(step_cadence_due(1, 4));
    REQUIRE(step_cadence_due(4, 4));
    REQUIRE(step_cadence_due(8, 4));
    REQUIRE_FALSE(step_cadence_due(9, 4));
    REQUIRE_FALSE(step_cadence_due(4, 0));
    REQUIRE_FALSE(step_cadence_due(4, -1));
}

TEST_CASE("runtime final step lands exactly on long floating deadlines",
          "[core][runtime][cadence][sci089]")
{
    const double accumulated_time = 3599.799999998917;
    const double adjusted = clamp_step_to_deadline(
        accumulated_time, 0.2, 3600.0, 17999);
    REQUIRE(adjusted > 0.2);
    REQUIRE(accumulated_time + adjusted == 3600.0);

    REQUIRE(clamp_step_to_deadline(3599.79, 0.2, 3600.0, 17999)
            == Approx(0.2));
    REQUIRE(clamp_step_to_deadline(3599.85, 0.2, 3600.0, 17999)
            == Approx(0.15));

    // Even an extreme addition count cannot classify a step with more than a
    // quarter-step remaining error as complete.
    REQUIRE(clamp_step_to_deadline(3599.74, 0.2, 3600.0, 1000000000)
            == Approx(0.2));
    REQUIRE(clamp_step_to_deadline(
                3599.74, 0.2, 3600.0,
                std::numeric_limits<std::uint64_t>::max())
            == Approx(0.2));
}

TEST_CASE("explicit scientific cadence is independent of derived output",
          "[core][runtime][cadence]")
{
    SECTION("explicit off disables science without starving required output fields")
    {
        const auto no_output = make_diagnostic_step_plan(1, 0, true, false);
        REQUIRE_FALSE(no_output.scientific);
        REQUIRE_FALSE(no_output.derived);

        const auto output_due = make_diagnostic_step_plan(1, 0, true, true);
        REQUIRE_FALSE(output_due.scientific);
        REQUIRE(output_due.derived);
    }

    SECTION("a trajectory consumer forces derived state while science stays off")
    {
        const bool trajectory_requires_vorticity = true;
        const auto trajectory_step = make_diagnostic_step_plan(
            7, 0, false, trajectory_requires_vorticity);
        REQUIRE_FALSE(trajectory_step.scientific);
        REQUIRE(trajectory_step.derived);
    }

    SECTION("positive cadence enables science without unrelated derived work")
    {
        const auto before = make_diagnostic_step_plan(2, 3, true, false);
        REQUIRE_FALSE(before.scientific);
        REQUIRE_FALSE(before.derived);

        const auto due = make_diagnostic_step_plan(3, 3, false, false);
        REQUIRE(due.scientific);
        REQUIRE_FALSE(due.derived);

        const auto due_with_consumer =
            make_diagnostic_step_plan(3, 3, false, true);
        REQUIRE(due_with_consumer.scientific);
        REQUIRE(due_with_consumer.derived);
    }

    SECTION("omitted cadence preserves the historical decision")
    {
        const auto disabled = make_diagnostic_step_plan(1, -1, false, false);
        REQUIRE_FALSE(disabled.scientific);
        REQUIRE_FALSE(disabled.derived);

        const auto enabled = make_diagnostic_step_plan(1, -1, true, false);
        REQUIRE(enabled.scientific);
        REQUIRE(enabled.derived);
    }
}

TEST_CASE("trajectory gridded diagnostics are limited to actual consumers",
          "[core][runtime][cadence][trajectory]")
{
    REQUIRE_FALSE(trajectory_requires_gridded_diagnostics(
        false, false, true, true, true));

    SECTION("a future Cartesian trajectory seed does no gridded work")
    {
        REQUIRE_FALSE(trajectory_requires_gridded_diagnostics(
            true, false, false, true, true));
    }

    SECTION("an explicit Cartesian A- or C-grid seed samples online")
    {
        REQUIRE_FALSE(trajectory_requires_gridded_diagnostics(
            true, false, true, false, true));
    }

    SECTION("automatic centering needs one gridded seed diagnostic")
    {
        REQUIRE(trajectory_requires_gridded_diagnostics(
            true, false, true, true, true));
    }

    SECTION("a seeded Cartesian A- or C-grid trajectory continues online")
    {
        REQUIRE_FALSE(trajectory_requires_gridded_diagnostics(
            true, true, false, true, true));
    }

    SECTION("unsupported grids refresh the fallback at seed and advance")
    {
        REQUIRE(trajectory_requires_gridded_diagnostics(
            true, false, true, false, false));
        REQUIRE(trajectory_requires_gridded_diagnostics(
            true, true, false, false, false));
    }
}

TEST_CASE("live shared-memory cadence is independent and explicitly full-rate",
          "[core][runtime][cadence][shm]")
{
    const double first = next_live_shm_publish_time(0.0, 1.0);
    REQUIRE(first == Approx(1.0));
    REQUIRE_FALSE(live_shm_publish_due(true, false, 0.5, first));
    REQUIRE(live_shm_publish_due(true, false, 1.0, first));
    REQUIRE(live_shm_publish_due(true, false, 1.25, first));
    REQUIRE_FALSE(live_shm_publish_due(false, true, 0.1, first));

    SECTION("full-rate is an explicit every-step override")
    {
        REQUIRE(live_shm_publish_due(true, true, 0.1, 10.0));
        REQUIRE(live_shm_publish_due(true, true, 0.2, 10.0));
    }

    SECTION("restart begins from the next absolute boundary")
    {
        REQUIRE(next_live_shm_publish_time(7.0, 2.0) == Approx(8.0));
        REQUIRE(next_live_shm_publish_time(8.0, 2.0) == Approx(10.0));
    }

    REQUIRE(next_live_shm_publish_time(0.0, 0.0) < 0.0);
}

TEST_CASE("simulation runtime snapshot restores all persistent module memory",
          "[core][runtime][restart][sci020]")
{
    SimulationRuntimeState runtime;
    runtime.compute.config.backend = "vulkan";
    runtime.compute.config.device_index = 3;
    runtime.compute.requested_kind = ComputeBackendKind::Vulkan;
    runtime.parity.call_counters["runtime_kernel"] = 4;
    runtime.parity.total_samples = 3;
    runtime.parity.any_breach = true;
    runtime.radiation.last_update_time_s = 17.0;
    runtime.turbulence.last_update_time_s = 18.0;
    runtime.boundary_layer.last_update_time_s = 19.0;
    runtime.boundary_layer.skin_theta_2d = {301.0f, 302.0f};
    runtime.boundary_layer.skin_captured = true;
    runtime.boundary_layer.slab_columns.resize(1);
    runtime.boundary_layer.slab_columns[0].h_m = 725.0;
    runtime.chaos.noise_elapsed_s = 3.5;
    runtime.chaos.pbl_base_tendencies_valid = true;
    runtime.rayleigh.upper_initialized = true;
    runtime.rayleigh.alpha = {0.1, 0.2};
    runtime.microphysics.last_water_repair_report
        .species[0].lower_bound_count = 4;
    runtime.microphysics.last_thermodynamic_repair_report
        .theta_bound_correction_j = -125.0;
    runtime.terrain.config.scheme_id = "none";
    runtime.terrain.topography.h = {{12.0}};
    runtime.radar.selected_config.scheme_id = "reflectivity";
    runtime.radar.selected_output.Z_dBZ.resize(1, 1, 1, 35.0f);
    runtime.advection.publish_hydrometeor_activity(
        {true, false, true, false, false, true}, 12, 10, 8);
    runtime.advection.metadata_consumptions = 7;
    runtime.advection.config.scheme_id = "weno5";
    runtime.advection.scalar_scratch_a.resize(2, 1, 1, 4.0f);
    runtime.advection.vertical_tendencies.dqdt_adv.resize(2, 1, 1, -2.0f);
    runtime.advection.tracer_advected_in_batch = true;
    runtime.diffusion.config.scheme_id = "implicit";
    runtime.diffusion.tendencies.dthetadt_diff.resize(2, 1, 1, 0.25f);
    runtime.diffusion.diagnostics.K_effective.resize(2, 1, 1, 4.0f);
    runtime.time_stepping.config.scheme_id = "split_explicit";
    runtime.time_stepping.diagnostics.n_steps = 9;
    runtime.time_stepping.diagnostics.dt_history = {1.0, 0.5};
    runtime.time_stepping.slow_rk2.u_save.resize(2, 1, 1, 6.0f);
    runtime.dynamics.scheme = std::make_unique<SnapshotDynamicsScheme>(17);
    runtime.dynamics.du_dt.resize(2, 1, 1, 7.0f);
    runtime.dynamics.acoustic_profile_rows = {1.0f, 2.0f, 3.0f};
    runtime.dynamics.transport_context.mass_flux_x.resize(
        2, 1, 1, 8.0f);

    // The aggregate is useful only if every state-aware physics entry point
    // is attached to the same simulation-owned lifetime. Pin the complete
    // binding contract here so adding a new persistent module cannot silently
    // leave it on the process-global compatibility path.
    SimulationStateView bound_state = global_simulation_state();
    runtime.bind(bound_state);
    REQUIRE(bound_state.compute_runtime == &runtime.compute);
    REQUIRE(bound_state.parity_runtime == &runtime.parity);
    REQUIRE(bound_state.radiation_runtime == &runtime.radiation);
    REQUIRE(bound_state.turbulence_runtime == &runtime.turbulence);
    REQUIRE(bound_state.boundary_layer_runtime == &runtime.boundary_layer);
    REQUIRE(bound_state.chaos_runtime == &runtime.chaos);
    REQUIRE(bound_state.rayleigh_runtime == &runtime.rayleigh);
    REQUIRE(bound_state.microphysics_runtime == &runtime.microphysics);
    REQUIRE(bound_state.terrain_runtime == &runtime.terrain);
    REQUIRE(bound_state.radar_runtime == &runtime.radar);
    REQUIRE(bound_state.advection_runtime == &runtime.advection);
    REQUIRE(bound_state.diffusion_runtime == &runtime.diffusion);
    REQUIRE(bound_state.time_stepping_runtime == &runtime.time_stepping);
    REQUIRE(bound_state.dynamics_runtime == &runtime.dynamics);

    const SimulationRuntimeSnapshot snapshot =
        snapshot_simulation_runtime(runtime);
    REQUIRE(snapshot.state.compute.config.backend == "vulkan");
    REQUIRE(snapshot.state.compute.backend == nullptr);
    REQUIRE(snapshot.state.parity.call_counters.at("runtime_kernel") == 4);
    REQUIRE(snapshot.state.parity.total_samples == 3);

    // Snapshots must own their field/vector storage. A shallow alias would
    // make restart dependent on mutations made after the checkpoint.
    REQUIRE(snapshot.state.boundary_layer.skin_theta_2d.data() !=
            runtime.boundary_layer.skin_theta_2d.data());
    REQUIRE(snapshot.state.rayleigh.alpha.data() != runtime.rayleigh.alpha.data());
    REQUIRE(snapshot.state.terrain.topography.h[0].data() !=
            runtime.terrain.topography.h[0].data());
    REQUIRE(snapshot.state.radar.selected_output.Z_dBZ.data() !=
            runtime.radar.selected_output.Z_dBZ.data());
    REQUIRE(snapshot.state.advection.scalar_scratch_a.data() !=
            runtime.advection.scalar_scratch_a.data());
    REQUIRE(snapshot.state.advection.vertical_tendencies.dqdt_adv.data() !=
            runtime.advection.vertical_tendencies.dqdt_adv.data());
    REQUIRE(snapshot.state.diffusion.tendencies.dthetadt_diff.data() !=
            runtime.diffusion.tendencies.dthetadt_diff.data());
    REQUIRE(snapshot.state.time_stepping.diagnostics.dt_history.data() !=
            runtime.time_stepping.diagnostics.dt_history.data());
    REQUIRE(snapshot.state.time_stepping.slow_rk2.u_save.data() !=
            runtime.time_stepping.slow_rk2.u_save.data());
    REQUIRE(snapshot.state.dynamics.scheme.get() !=
            runtime.dynamics.scheme.get());
    REQUIRE(snapshot.state.dynamics.du_dt.data() !=
            runtime.dynamics.du_dt.data());
    REQUIRE(snapshot.state.dynamics.acoustic_profile_rows.data() !=
            runtime.dynamics.acoustic_profile_rows.data());
    // The acoustic mass-flux handoff is explicitly step-scoped and never a
    // legal checkpoint-boundary owner.
    REQUIRE(snapshot.state.dynamics.transport_context.mass_flux_x.empty());

    runtime.compute.config.backend = "cpu";
    runtime.compute.config.device_index = -1;
    runtime.parity = {};
    runtime.radiation.last_update_time_s = -1.0;
    runtime.turbulence.last_update_time_s = -1.0;
    runtime.boundary_layer.skin_theta_2d.clear();
    runtime.boundary_layer.slab_columns[0].h_m = 0.0;
    runtime.chaos.noise_elapsed_s = 0.0;
    runtime.rayleigh.alpha.clear();
    runtime.microphysics.last_water_repair_report = {};
    runtime.microphysics.last_thermodynamic_repair_report = {};
    runtime.terrain = {};
    runtime.radar = {};
    runtime.advection = {};
    runtime.time_stepping = {};
    runtime.dynamics = {};

    restore_simulation_runtime(runtime, snapshot);

    REQUIRE(runtime.compute.config.backend == "vulkan");
    REQUIRE(runtime.compute.config.device_index == 3);
    REQUIRE(runtime.compute.backend == nullptr);
    REQUIRE(runtime.compute.active_kind == ComputeBackendKind::Cpu);
    REQUIRE(runtime.compute.requested_kind == ComputeBackendKind::Vulkan);
    REQUIRE(runtime.parity.call_counters.at("runtime_kernel") == 4);
    REQUIRE(runtime.parity.total_samples == 3);
    REQUIRE(runtime.parity.any_breach);
    REQUIRE(runtime.radiation.last_update_time_s == Approx(17.0));
    REQUIRE(runtime.turbulence.last_update_time_s == Approx(18.0));
    REQUIRE(runtime.boundary_layer.last_update_time_s == Approx(19.0));
    REQUIRE(runtime.boundary_layer.skin_theta_2d.size() == 2);
    REQUIRE(runtime.boundary_layer.skin_theta_2d[0] == Approx(301.0f));
    REQUIRE(runtime.boundary_layer.skin_theta_2d[1] == Approx(302.0f));
    REQUIRE(runtime.boundary_layer.skin_captured);
    REQUIRE(runtime.boundary_layer.slab_columns[0].h_m == Approx(725.0));
    REQUIRE(runtime.chaos.noise_elapsed_s == Approx(3.5));
    REQUIRE(runtime.chaos.pbl_base_tendencies_valid);
    REQUIRE(runtime.rayleigh.upper_initialized);
    REQUIRE(runtime.rayleigh.alpha.size() == 2);
    REQUIRE(runtime.rayleigh.alpha[0] == Approx(0.1));
    REQUIRE(runtime.rayleigh.alpha[1] == Approx(0.2));
    REQUIRE(runtime.microphysics.last_water_repair_report
                .species[0].lower_bound_count == 4);
    REQUIRE(runtime.microphysics.last_thermodynamic_repair_report
                .theta_bound_correction_j == Approx(-125.0));
    REQUIRE(runtime.terrain.config.scheme_id == "none");
    REQUIRE(runtime.terrain.topography.h[0][0] == Approx(12.0));
    REQUIRE(runtime.radar.selected_config.scheme_id == "reflectivity");
    REQUIRE(runtime.radar.selected_output.Z_dBZ[0][0][0] == Approx(35.0f));
    REQUIRE(runtime.advection.activity_current);
    REQUIRE(runtime.advection.hydrometeor_active[0]);
    REQUIRE_FALSE(runtime.advection.hydrometeor_active[1]);
    REQUIRE(runtime.advection.hydrometeor_active[2]);
    REQUIRE(runtime.advection.hydrometeor_active[5]);
    REQUIRE(runtime.advection.metadata_consumptions == 7);
    REQUIRE(runtime.advection.config.scheme_id == "weno5");
    REQUIRE(runtime.advection.scalar_scratch_a[0][0][0] == Approx(4.0f));
    REQUIRE(runtime.advection.vertical_tendencies.dqdt_adv[0][0][0]
            == Approx(-2.0f));
    REQUIRE(runtime.advection.tracer_advected_in_batch);
    REQUIRE(runtime.diffusion.config.scheme_id == "implicit");
    REQUIRE(runtime.diffusion.tendencies.dthetadt_diff[0][0][0]
            == Approx(0.25f));
    REQUIRE(runtime.diffusion.diagnostics.K_effective[0][0][0]
            == Approx(4.0f));
    REQUIRE(runtime.time_stepping.config.scheme_id == "split_explicit");
    REQUIRE(runtime.time_stepping.diagnostics.n_steps == 9);
    REQUIRE(runtime.time_stepping.diagnostics.dt_history[1] == Approx(0.5));
    REQUIRE(runtime.time_stepping.slow_rk2.u_save[0][0][0]
            == Approx(6.0f));
    REQUIRE(dynamic_cast<SnapshotDynamicsScheme*>(
                runtime.dynamics.scheme.get())->generation == 17);
    REQUIRE(runtime.dynamics.du_dt.empty());
    REQUIRE(runtime.dynamics.acoustic_profile_rows.empty());

    // Restoring replaces the aggregate's owned objects; callers rebind their
    // view explicitly at the checkpoint boundary.
    runtime.bind(bound_state);
    REQUIRE(bound_state.compute_runtime == &runtime.compute);
    REQUIRE(bound_state.parity_runtime == &runtime.parity);
    REQUIRE(bound_state.boundary_layer_runtime == &runtime.boundary_layer);
    REQUIRE(bound_state.chaos_runtime == &runtime.chaos);
    REQUIRE(bound_state.terrain_runtime == &runtime.terrain);
    REQUIRE(bound_state.radar_runtime == &runtime.radar);
    REQUIRE(bound_state.advection_runtime == &runtime.advection);
    REQUIRE(bound_state.diffusion_runtime == &runtime.diffusion);
    REQUIRE(bound_state.time_stepping_runtime == &runtime.time_stepping);
    REQUIRE(bound_state.dynamics_runtime == &runtime.dynamics);
}

TEST_CASE("simulation storage owns state-view fields and deep snapshots",
          "[core][runtime][restart][storage]")
{
    int nr = 2;
    int nth = 1;
    int nz = 2;
    double dx = 100.0;
    double dz_local = 50.0;
    double dt_local = 1.0;
    double dy = 200.0;
    SimulationStorageState storage{{nr, nth, nz, dx, dz_local, dt_local, dy}};
    storage.grid.geometry.initialize(nr, nth, nz, dx, dz_local, dy,
                                     CoordinateSystem::Cartesian);
    storage.configuration.coordinate_system = CoordinateSystem::Cartesian;
    storage.configuration.stagger_type = StaggerType::CGrid;
    storage.configuration.validation_report_path =
        "validation/report/path/long-enough-to-require-owned-storage.json";
    storage.fields.rho.resize(nr, nth, nz, 1.1f);
    storage.fields.theta.resize(nr, nth, nz, 301.0f);
    storage.base.rho0 = {1.1, 1.0};
    storage.tendencies.dtheta_dt_rad.resize(nr, nth, nz, 0.02f);
    storage.surface.ustar = {0.4f, 0.5f};
    storage.nested.config.enabled = true;
    storage.nested.rho.resize(1, 1, 1, 2.0f);
    storage.diagnostics.radar_reflectivity.resize(1, 1, 1, 3.0f);
    storage.diagnostics.vorticity_r.resize(1, 1, 1, 4.0f);
    storage.runtime.radiation.last_update_time_s = 17.0;
    storage.runtime.compute.config.backend = "vulkan";
    storage.runtime.compute.config.device_index = 2;
    storage.runtime.parity.call_counters["storage_kernel"] = 5;
    storage.runtime.parity.total_samples = 2;
    storage.runtime.boundary_layer.skin_theta_2d = {301.0f, 302.0f};
    storage.runtime.terrain.topography.h = {{7.0}};
    storage.runtime.radar.selected_output.Z_dBZ.resize(1, 1, 1, 42.0f);
    storage.runtime.advection.scalar_scratch_a.resize(1, 1, 1, 5.0f);
    storage.runtime.time_stepping.diagnostics.dt_history = {1.0, 0.5};
    storage.coriolis_f = 8.0e-5;
    storage.time_s = 90.0;

    SimulationStateView state = make_simulation_state_view(storage);
    REQUIRE(&state.grid.nr == &storage.grid.nr);
    REQUIRE(&state.grid.geometry == &storage.grid.geometry);
    REQUIRE(&state.fields.rho == &storage.fields.rho);
    REQUIRE(&state.fields.theta == &storage.fields.theta);
    REQUIRE(&state.base.rho0 == &storage.base.rho0);
    REQUIRE(&state.tendencies.dtheta_dt_rad ==
            &storage.tendencies.dtheta_dt_rad);
    REQUIRE(state.configuration == &storage.configuration);
    REQUIRE(state.surface.sfc_ustar == &storage.surface.ustar);
    REQUIRE(&state.coriolis_f == &storage.coriolis_f);
    REQUIRE(&state.time_s == &storage.time_s);
    REQUIRE(state.compute_runtime == &storage.runtime.compute);
    REQUIRE(state.parity_runtime == &storage.runtime.parity);
    REQUIRE(state.radiation_runtime == &storage.runtime.radiation);
    REQUIRE(state.boundary_layer_runtime == &storage.runtime.boundary_layer);
    REQUIRE(state.terrain_runtime == &storage.runtime.terrain);
    REQUIRE(state.radar_runtime == &storage.runtime.radar);
    REQUIRE(state.advection_runtime == &storage.runtime.advection);
    REQUIRE(state.time_stepping_runtime == &storage.runtime.time_stepping);
    REQUIRE(state.runtime_owner == &storage.runtime);

    const SimulationStorageSnapshot snapshot =
        snapshot_simulation_storage(storage);
    REQUIRE(snapshot.state.fields.rho.data() != storage.fields.rho.data());
    REQUIRE(snapshot.state.grid.geometry.z.data() !=
            storage.grid.geometry.z.data());
    REQUIRE(snapshot.state.configuration.validation_report_path.data() !=
            storage.configuration.validation_report_path.data());
    REQUIRE(snapshot.state.base.rho0.data() != storage.base.rho0.data());
    REQUIRE(snapshot.state.surface.ustar.data() != storage.surface.ustar.data());
    REQUIRE(snapshot.state.nested.rho.data() != storage.nested.rho.data());
    REQUIRE(snapshot.state.diagnostics.radar_reflectivity.data() !=
            storage.diagnostics.radar_reflectivity.data());
    REQUIRE(snapshot.state.diagnostics.vorticity_r.data() !=
            storage.diagnostics.vorticity_r.data());
    REQUIRE(snapshot.state.runtime.boundary_layer.skin_theta_2d.data() !=
            storage.runtime.boundary_layer.skin_theta_2d.data());
    REQUIRE(snapshot.state.runtime.compute.config.backend == "vulkan");
    REQUIRE(snapshot.state.runtime.compute.backend == nullptr);
    REQUIRE(snapshot.state.runtime.parity.call_counters.at("storage_kernel") == 5);
    REQUIRE(snapshot.state.runtime.terrain.topography.h[0].data() !=
            storage.runtime.terrain.topography.h[0].data());
    REQUIRE(snapshot.state.runtime.radar.selected_output.Z_dBZ.data() !=
            storage.runtime.radar.selected_output.Z_dBZ.data());
    REQUIRE(snapshot.state.runtime.advection.scalar_scratch_a.data() !=
            storage.runtime.advection.scalar_scratch_a.data());
    REQUIRE(snapshot.state.runtime.time_stepping.diagnostics.dt_history.data() !=
            storage.runtime.time_stepping.diagnostics.dt_history.data());

    storage.fields.rho[0][0][0] = 9.0f;
    storage.grid.dx = 9.0;
    storage.grid.geometry.z[0] = 9.0;
    storage.configuration.stagger_type = StaggerType::Collocated;
    storage.configuration.validation_report_path = "mutated";
    storage.base.rho0[0] = 9.0;
    storage.surface.ustar[0] = 9.0f;
    storage.nested.config.enabled = false;
    storage.nested.rho[0][0][0] = 9.0f;
    storage.diagnostics.radar_reflectivity[0][0][0] = 9.0f;
    storage.diagnostics.vorticity_r[0][0][0] = 9.0f;
    storage.runtime.radiation.last_update_time_s = -1.0;
    storage.runtime.compute.config.backend = "cpu";
    storage.runtime.parity = {};
    storage.runtime.boundary_layer.skin_theta_2d.clear();
    storage.runtime.terrain = {};
    storage.runtime.radar = {};
    storage.runtime.advection.scalar_scratch_a = {};
    storage.runtime.time_stepping.diagnostics.dt_history.clear();
    storage.time_s = 0.0;
    restore_simulation_storage(storage, snapshot);

    REQUIRE(storage.fields.rho[0][0][0] == Approx(1.1f));
    REQUIRE(storage.grid.dx == Approx(100.0));
    REQUIRE(storage.grid.geometry.z[0] == Approx(25.0));
    REQUIRE(storage.configuration.stagger_type == StaggerType::CGrid);
    REQUIRE(storage.configuration.validation_report_path ==
            "validation/report/path/long-enough-to-require-owned-storage.json");
    REQUIRE(storage.base.rho0[0] == Approx(1.1));
    REQUIRE(storage.surface.ustar[0] == Approx(0.4f));
    REQUIRE(storage.nested.config.enabled);
    REQUIRE(storage.nested.config.center_i == nr / 2);
    REQUIRE(storage.nested.config.center_j == nth / 2);
    REQUIRE(storage.nested.config.center_k == nz / 2);
    REQUIRE(storage.nested.rho[0][0][0] == Approx(2.0f));
    REQUIRE(storage.diagnostics.radar_reflectivity[0][0][0] == Approx(3.0f));
    REQUIRE(storage.diagnostics.vorticity_r[0][0][0] == Approx(4.0f));
    REQUIRE(storage.runtime.radiation.last_update_time_s == Approx(17.0));
    REQUIRE(storage.runtime.compute.config.backend == "vulkan");
    REQUIRE(storage.runtime.compute.config.device_index == 2);
    REQUIRE(storage.runtime.compute.backend == nullptr);
    REQUIRE(storage.runtime.parity.call_counters.at("storage_kernel") == 5);
    REQUIRE(storage.runtime.parity.total_samples == 2);
    REQUIRE(storage.runtime.boundary_layer.skin_theta_2d[0] == Approx(301.0f));
    REQUIRE(storage.runtime.terrain.topography.h[0][0] == Approx(7.0));
    REQUIRE(storage.runtime.radar.selected_output.Z_dBZ[0][0][0] ==
            Approx(42.0f));
    REQUIRE(storage.runtime.advection.scalar_scratch_a[0][0][0] == Approx(5.0f));
    REQUIRE(storage.runtime.time_stepping.diagnostics.dt_history[1] == Approx(0.5));
    REQUIRE(storage.time_s == Approx(90.0));

    SimulationStateView restored_state = make_simulation_state_view(storage);
    REQUIRE(restored_state.compute_runtime == &storage.runtime.compute);
    REQUIRE(restored_state.parity_runtime == &storage.runtime.parity);
    REQUIRE(restored_state.configuration == &storage.configuration);
    REQUIRE(restored_state.radiation_runtime == &storage.runtime.radiation);
    REQUIRE(restored_state.boundary_layer_runtime ==
            &storage.runtime.boundary_layer);
    REQUIRE(restored_state.terrain_runtime == &storage.runtime.terrain);
    REQUIRE(restored_state.radar_runtime == &storage.runtime.radar);
    REQUIRE(restored_state.advection_runtime == &storage.runtime.advection);
    REQUIRE(restored_state.time_stepping_runtime ==
            &storage.runtime.time_stepping);
    REQUIRE(restored_state.runtime_owner == &storage.runtime);

    REQUIRE(&rho == &global_simulation_storage.fields.rho);
    REQUIRE(&NR == &global_simulation_storage.grid.nr);
    REQUIRE(&dr == &global_simulation_storage.grid.dx);
    REQUIRE(&global_grid_geometry == &global_simulation_storage.grid.geometry);
    REQUIRE(&global_coordinate_system ==
            &global_simulation_storage.configuration.coordinate_system);
    REQUIRE(&global_stagger_type ==
            &global_simulation_storage.configuration.stagger_type);
    REQUIRE(&global_wind_profile ==
            &global_simulation_storage.configuration.wind_profile);
    REQUIRE(&global_damping_alpha ==
            &global_simulation_storage.configuration.damping_alpha);
    REQUIRE(&global_coriolis_enabled ==
            &global_simulation_storage.configuration.coriolis_enabled);
    REQUIRE(&global_validation_report_path ==
            &global_simulation_storage.configuration.validation_report_path);
    REQUIRE(&global_log_profile ==
            &global_simulation_storage.configuration.log_profile);
    REQUIRE(&theta == &global_simulation_storage.fields.theta);
    REQUIRE(&rho0_base == &global_simulation_storage.base.rho0);
    REQUIRE(&dtheta_dt_rad ==
            &global_simulation_storage.tendencies.dtheta_dt_rad);
    REQUIRE(&sfc_ustar == &global_simulation_storage.surface.ustar);
    REQUIRE(&simulation_time == &global_simulation_storage.time_s);
    REQUIRE(&nested_config == &global_simulation_storage.nested.config);
    REQUIRE(&nest_rho == &global_simulation_storage.nested.rho);
    REQUIRE(&nest_tracer == &global_simulation_storage.nested.tracer);
    REQUIRE(&radar_reflectivity ==
            &global_simulation_storage.diagnostics.radar_reflectivity);
    REQUIRE(&vorticity_r ==
            &global_simulation_storage.diagnostics.vorticity_r);
    REQUIRE(&angular_momentum ==
            &global_simulation_storage.diagnostics.angular_momentum);
    REQUIRE(&p_prime == &global_simulation_storage.diagnostics.p_prime);
    REQUIRE(&tmv::global_radiation_runtime_state ==
            &global_simulation_storage.runtime.radiation);
    REQUIRE(&global_compute_backend_runtime_state ==
            &global_simulation_storage.runtime.compute);
    REQUIRE(&global_compute_backend_config ==
            &global_simulation_storage.runtime.compute.config);
    REQUIRE(&nimbus::parity::global_parity_runtime_state ==
            &global_simulation_storage.runtime.parity);
    REQUIRE(&tmv::global_turbulence_runtime_state ==
            &global_simulation_storage.runtime.turbulence);
    REQUIRE(&tmv::global_boundary_layer_runtime_state ==
            &global_simulation_storage.runtime.boundary_layer);
    REQUIRE(&tmv::chaos::global_chaos_runtime_state ==
            &global_simulation_storage.runtime.chaos);
    REQUIRE(&global_rayleigh_runtime_state ==
            &global_simulation_storage.runtime.rayleigh);
    REQUIRE(&tmv::global_microphysics_runtime_state ==
            &global_simulation_storage.runtime.microphysics);
    REQUIRE(&tmv::global_terrain_runtime_state ==
            &global_simulation_storage.runtime.terrain);
    REQUIRE(&tmv::global_radar_runtime_state ==
            &global_simulation_storage.runtime.radar);
    REQUIRE(&tmv::global_advection_runtime_state ==
            &global_simulation_storage.runtime.advection);
    REQUIRE(&tmv::global_diffusion_runtime_state ==
            &global_simulation_storage.runtime.diffusion);
    REQUIRE(&tmv::global_time_stepping_runtime_state ==
            &global_simulation_storage.runtime.time_stepping);
}

TEST_CASE("physics deadlines reject unrepresentable cadences and skip missed samples in bounded work",
          "[runtime][cadence][physics]")
{
    REQUIRE(tmv::next_physics_deadline(0.0, 0.0, 60.0, 1.0e-12) == 60.0);
    REQUIRE(tmv::next_physics_deadline(0.0, 125.0, 60.0, 1.0e-12) == 180.0);
    REQUIRE(tmv::next_physics_deadline(0.0, 1.0e9, 0.5, 0.0) == 1.0e9 + 0.5);
    REQUIRE_THROWS(tmv::next_physics_deadline(0.0, 0.0, 1.0e-300, 1.0e-12));
    REQUIRE_THROWS(tmv::next_physics_deadline(1.0e20, 1.0e20, 1.0, 1.0e-12));
    REQUIRE_THROWS(tmv::next_physics_deadline(0.0, 1.0e300, 1.0, 1.0e-12));
}

TEST_CASE("budget rows follow the outer state transaction and output generation",
          "[runtime][budget][atomicity]")
{
    const auto directory = std::filesystem::temp_directory_path() /
        ("nimbus_budget_transaction_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::create_directories(directory);
    tmv::LevelBudgetCsv csv;
    std::mutex mutex;
    REQUIRE(csv.open((directory / "budget.csv").string(), "value\n", 17));
    const auto publish = [&](double value) {
        std::lock_guard<std::mutex> lock(mutex);
        *csv.stream(tmv::BudgetLevel::Child, mutex) << value << '\n';
    };
    {
        tmv::BudgetPublicationTransaction outer;
        publish(1.0);
        tmv::BudgetPublicationTransaction inner;
        publish(2.0);
        inner.commit();
    }
    const auto child = directory / "nests/child_0000/budget.csv";
    REQUIRE_FALSE(std::filesystem::exists(child));
    {
        tmv::BudgetPublicationTransaction transaction;
        publish(3.0);
        transaction.commit();
    }
    {
        tmv::BudgetPublicationTransaction outer;
        publish(4.0);
        {
            tmv::BudgetPublicationTransaction inner;
            publish(5.0);
            inner.commit();
        }
        publish(6.0);
        outer.commit();
    }
    csv.close();
    std::ifstream input(child);
    const std::string content{std::istreambuf_iterator<char>(input), {}};
    REQUIRE(content == "value\n3\n4\n5\n6\n");
    input.close();
    REQUIRE(csv.open((directory / "new.csv").string(), "value\n", 17));
    {
        tmv::BudgetPublicationTransaction transaction;
        publish(4.0);
        csv.close();
        REQUIRE(csv.open((directory / "new.csv").string(), "value\n", 17));
        transaction.commit();
    }
    REQUIRE_FALSE(std::filesystem::exists(directory / "nests/child_0000/new.csv"));
    csv.close();
    std::filesystem::remove_all(directory);
}
