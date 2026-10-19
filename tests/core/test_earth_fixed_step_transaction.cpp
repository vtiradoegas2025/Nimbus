#include "catch2/catch.hpp"
#include "core/runtime/earth_fixed_step_transaction.hpp"
#include "core/runtime/simulation_state.hpp"
#include "terrain/earth_fixed_geometry.hpp"
#include "core/runtime/earth_fixed_remap.hpp"
#include "diagnostics/energy_budget.hpp"
#include "terrain/conservative_transport.hpp"



namespace
{
SimulationStorageState initial_state(tmv::EarthFixedTerrainSampling sampling=tmv::EarthFixedTerrainSampling::Point)
{
    SimulationStorageState storage{{6,6,4,100.0,100.0,0.1,100.0}};
    storage.grid.geometry.initialize(6,6,4,100.0,100.0,100.0,
        CoordinateSystem::Cartesian, StaggerType::CGrid);
    storage.configuration.stagger_type = StaggerType::CGrid;
    using Lateral = tmv::init::config::LateralBC;
    storage.configuration.lateral_bc_west = Lateral::ZeroGradient;
    storage.configuration.lateral_bc_east = Lateral::ZeroGradient;
    storage.configuration.lateral_bc_south = Lateral::ZeroGradient;
    storage.configuration.lateral_bc_north = Lateral::ZeroGradient;
    tmv::fields::visit(storage.fields, tmv::fields::prognostic,
        [](const char*, Field3D& field) { field.resize(6,6,4,0.0f); });
    storage.fields.rho.fill(1.0f);
    storage.fields.theta.fill(300.0f);
    storage.fields.p.fill(100000.0f);
    auto& terrain = storage.runtime.terrain;
    terrain.config.scheme_id = "bell";
    terrain.config.coord_id = "smoothed";
    terrain.config.horizontal_placement_id = "earth_fixed";
    terrain.config.ztop = 400.0;
    terrain.config.bell.h0 = 10.0;
    terrain.config.bell.a = 200.0;
    terrain.config.feature_center_earth_x_m = 300.0;
    terrain.config.feature_center_earth_y_m = 300.0;
    tmv::EarthFixedTerrainBuildRequest request;
    request.nr=6; request.nth=6; request.nz=4;
    request.dx_m=request.dy_m=100.0;
    request.feature_center_earth_m={300.0,300.0};
    request.sampling=sampling;
    auto geometry = tmv::prepare_earth_fixed_terrain_commit(terrain,request);
    REQUIRE(tmv::commit_earth_fixed_terrain(geometry,terrain));
    storage.process_grid_metrics.terrain_metrics_active = true;
    storage.runtime.boundary_layer.next_update_time_s = 1.0;
    storage.runtime.boundary_layer.config.scheme_id = "none";
    storage.runtime.radiation.config.scheme_id = "none";
    storage.runtime.turbulence.config.scheme_id = "none";
    storage.base.theta0.assign(4,300.0);
    storage.base.p0.assign(4,100000.0);
    storage.base.qv0.assign(4,0.0);
    storage.base.u0.assign(4,0.0);
    storage.base.v0.assign(4,0.0);
    storage.runtime.compute.config.backend = "cpu";
    std::string error;
    REQUIRE(initialize_compute_backend_runtime(storage.runtime.compute,
        CoordinateSystem::Cartesian, error));
    return storage;
}
}

TEST_CASE("Earth-fixed timestep rejection retains every staged owner",
          "[core][earth_fixed][step][atomicity]")
{
    auto live = initial_state();
    const auto metrics = live.runtime.terrain.metrics.J.values;
    const auto* backend = live.runtime.compute.backend.get();
    {
        tmv::EarthFixedStepTransaction step(live,0.1);
        auto candidate = step.state();
        candidate.fields.theta(2,2,1) = 310.0f;
        candidate.boundary_layer_runtime->next_update_time_s = 2.0;
        candidate.terrain_runtime->earth_fixed_placement.grid_origin_earth_m.x_m = 1.0;
        REQUIRE_THROWS(step.prepare_commit());
        REQUIRE_FALSE(step.publish());
    }
    CHECK(live.time_s == 0.0);
    CHECK(live.fields.theta(2,2,1) == 300.0f);
    CHECK(live.runtime.boundary_layer.next_update_time_s == 1.0);
    CHECK(live.runtime.terrain.metrics.J.values == metrics);
    CHECK(live.runtime.terrain.earth_fixed_placement.grid_origin_earth_m.x_m == 0.0);
    CHECK(live.runtime.compute.backend.get() == backend);
}

TEST_CASE("Earth-fixed timestep publication rebinds owners and keeps the backend",
          "[core][earth_fixed][step][continuation]")
{
    auto live = initial_state();
    const auto* backend = live.runtime.compute.backend.get();
    for (int interval=0; interval<2; ++interval)
    {
        tmv::EarthFixedStepTransaction step(live,0.1);
        auto candidate = step.state();
        candidate.fields.theta(2,2,1) += 1.0f;
        candidate.fields.rho(1,2,1) = 2.0f + interval;
        candidate.boundary_layer_runtime->next_update_time_s += 1.0;
        CHECK(live.fields.theta(2,2,1) == 300.0f + interval);
        step.prepare_commit();
        CHECK(live.time_s == Approx(interval*0.1));
        CHECK_THROWS(step.state());
        REQUIRE(step.publish());
        CHECK_FALSE(step.publish());
        CHECK(live.time_s == Approx((interval+1)*0.1));
        CHECK(live.fields.theta(2,2,1) == 301.0f + interval);
        CHECK(live.runtime.boundary_layer.next_update_time_s == 2.0 + interval);
        CHECK(live.runtime.compute.backend.get() == backend);
        CHECK(live.process_grid_metrics.terrain_metrics == &live.runtime.terrain.metrics);
        REQUIRE(live.runtime.boundary_conditions.execution_operator);
        live.runtime.boundary_conditions.execution_operator->apply_full();
        CHECK(live.fields.rho(0,2,1) == 2.0f + interval);
    }
}

TEST_CASE("Earth-fixed timestep publication rejects a stale live clock",
          "[core][earth_fixed][step][atomicity]")
{
    auto live = initial_state();
    tmv::EarthFixedStepTransaction step(live,0.1);
    step.state().fields.theta(2,2,1) = 310.0f;
    step.prepare_commit();
    live.time_s = 0.05;
    CHECK_FALSE(step.publish());
    CHECK(live.time_s == 0.05);
    CHECK(live.fields.theta(2,2,1) == 300.0f);
}

TEST_CASE("Earth-fixed remapping closes physical mass and total moist energy",
          "[core][earth_fixed][remap][conservation]")
{
    auto state=initial_state(tmv::EarthFixedTerrainSampling::CellAverageBilinear);
    state.configuration.domain_frame.velocity_ms={3.0,-1.0};
    state.fields.u.fill(0.0f);
    state.fields.v.fill(0.0f);
    auto view=make_simulation_state_view(state);
    const auto before=compute_moist_mixture_energy_v2_deterministic(view);
    const auto report=tmv::remap_earth_fixed_window(state,state.configuration.domain_frame,0.0,0.2);
    const auto after=compute_moist_mixture_energy_v2_deterministic(view);
    CHECK(state.time_s == 0.2);
    CHECK(state.runtime.terrain.earth_fixed_placement.grid_origin_earth_m.x_m == Approx(0.6));
    CHECK(state.runtime.terrain.earth_fixed_placement.grid_origin_earth_m.y_m == -0.2);
    CHECK(report.substeps == 1);
    const double mass_scale=std::max(1.0,report.dry_mass.initial_inventory);
    CHECK(std::abs(report.dry_mass.final_inventory-report.dry_mass.initial_inventory
        +report.dry_mass.boundary_outflow-report.dry_mass.storage_rounding)/mass_scale < 1.0e-12);
    CHECK(report.energy.coordinate_remap_committed_change_j
        == Approx(after.total-before.total).margin(0.01));
    CHECK(std::abs(report.energy_storage_rounding_j)/std::max(1.0,std::abs(before.total)) < 1.0e-6);
    CHECK(tmv::structurally_valid_completed_step_energy_report(report.energy));
}

TEST_CASE("Late Earth-fixed remap failure discards changed geometry and atmosphere",
          "[core][earth_fixed][step][atomicity]")
{
    auto live=initial_state(tmv::EarthFixedTerrainSampling::CellAverageBilinear);
    live.configuration.domain_frame.velocity_ms={3.0,-1.0};
    const auto geometry=live.runtime.terrain.metrics.J.values;
    tmv::EarthFixedStepTransaction step(live,0.2);
    auto candidate=step.state();
    candidate.fields.theta.fill(10000.0f);
    candidate.boundary_layer_runtime->next_update_time_s=99.0;
    REQUIRE_THROWS_WITH(step.finish_evolution(),
        "coordinate remapping cannot represent its conserved energy thermodynamically");
    CHECK(step.state().terrain_runtime->earth_fixed_placement.grid_origin_earth_m.x_m == Approx(0.6));
    CHECK_FALSE(step.publish());
    CHECK(live.time_s == 0.0);
    CHECK(live.fields.theta(2,2,1) == 300.0f);
    CHECK(live.runtime.boundary_layer.next_update_time_s == 1.0);
    CHECK(live.runtime.terrain.metrics.J.values == geometry);
    CHECK(live.runtime.terrain.earth_fixed_placement.grid_origin_earth_m.x_m == 0.0);
}
