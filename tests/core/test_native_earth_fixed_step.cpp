#include "catch2/catch.hpp"
#include "core/runtime/earth_fixed_step_transaction.hpp"
#include "core/runtime/simulation_state.hpp"
#include "terrain/earth_fixed_geometry.hpp"
#include "compute/backend_runtime.hpp"
#include <algorithm>
namespace
{
SimulationStorageState initial_state(tmv::EarthFixedTerrainSampling sampling = tmv::EarthFixedTerrainSampling::Point)
{
    SimulationStorageState storage{{6, 6, 4, 100.0, 100.0, 0.1, 100.0}};
    storage.grid.geometry.initialize(6, 6, 4, 100.0, 100.0, 100.0, CoordinateSystem::Cartesian, StaggerType::CGrid);
    storage.configuration.stagger_type     = StaggerType::CGrid;
    using Lateral                          = tmv::init::config::LateralBC;
    storage.configuration.lateral_bc_west  = Lateral::ZeroGradient;
    storage.configuration.lateral_bc_east  = Lateral::ZeroGradient;
    storage.configuration.lateral_bc_south = Lateral::ZeroGradient;
    storage.configuration.lateral_bc_north = Lateral::ZeroGradient;
    tmv::fields::visit(
        storage.fields, tmv::fields::prognostic, [](const char*, Field3D& field) { field.resize(6, 6, 4, 0.0f); });
    storage.fields.rho.fill(1.0f);
    storage.fields.theta.fill(300.0f);
    storage.fields.p.fill(100000.0f);
    auto& terrain                           = storage.runtime.terrain;
    terrain.config.scheme_id                = "bell";
    terrain.config.coord_id                 = "smoothed";
    terrain.config.horizontal_placement_id  = "earth_fixed";
    terrain.config.ztop                     = 400.0;
    terrain.config.bell.h0                  = 10.0;
    terrain.config.bell.a                   = 200.0;
    terrain.config.feature_center_earth_x_m = 300.0;
    terrain.config.feature_center_earth_y_m = 300.0;
    tmv::EarthFixedTerrainBuildRequest request;
    request.nr   = 6;
    request.nth  = 6;
    request.nz   = 4;
    request.dx_m = request.dy_m    = 100.0;
    request.feature_center_earth_m = {300.0, 300.0};
    request.sampling               = sampling;
    auto geometry                  = tmv::prepare_earth_fixed_terrain_commit(terrain, request);
    REQUIRE(tmv::commit_earth_fixed_terrain(geometry, terrain));
    storage.process_grid_metrics.terrain_metrics_active = true;
    storage.runtime.boundary_layer.next_update_time_s   = 1.0;
    storage.runtime.boundary_layer.config.scheme_id     = "none";
    storage.runtime.radiation.config.scheme_id          = "none";
    storage.runtime.turbulence.config.scheme_id         = "none";
    storage.base.theta0.assign(4, 300.0);
    storage.base.p0.assign(4, 100000.0);
    storage.base.qv0.assign(4, 0.0);
    storage.base.u0.assign(4, 0.0);
    storage.base.v0.assign(4, 0.0);
    storage.runtime.compute.config.backend = "cpu";
    std::string error;
    REQUIRE(initialize_compute_backend_runtime(storage.runtime.compute, CoordinateSystem::Cartesian, error));
    return storage;
}
void exercise(const char* backend)
{
    auto live                                  = initial_state(tmv::EarthFixedTerrainSampling::CellAverageBilinear);
    live.runtime.compute.config.backend        = backend;
    live.runtime.compute.config.allow_fallback = false;
    std::string error;
    const bool  initialized =
        initialize_compute_backend_runtime(live.runtime.compute, CoordinateSystem::Cartesian, error);
    INFO(error);
    REQUIRE(initialized);
    live.configuration.domain_frame.velocity_ms = {3, -1};
    const auto* device                          = live.runtime.compute.backend.get();
    for (int interval = 0; interval < 2; ++interval)
    {
        const auto   before     = live.fields;
        const auto   geometry   = live.runtime.terrain.metrics.J.values;
        const auto   generation = live.runtime.terrain.earth_fixed_placement.generation;
        const double time       = live.time_s;
        {
            tmv::EarthFixedStepTransaction failed(live, .2);
            failed.state().fields.theta.fill(10000);
            REQUIRE_THROWS(failed.finish_evolution());
            CHECK_FALSE(failed.publish());
            CHECK(live.time_s == time);
            CHECK(live.runtime.terrain.metrics.J.values == geometry);
            CHECK(live.runtime.terrain.earth_fixed_placement.generation == generation);
            CHECK(std::equal(before.theta.data(), before.theta.data() + before.theta.size(), live.fields.theta.data()));
            CHECK(std::equal(before.u.data(), before.u.data() + before.u.size(), live.fields.u.data()));
        }
        tmv::EarthFixedStepTransaction success(live, .2);
        success.state().fields.theta(2, 2, 1) += .1f;
        success.prepare_commit();
        CHECK(live.time_s == time);
        REQUIRE(success.publish());
        CHECK_FALSE(success.publish());
        CHECK(live.time_s == Approx(time + .2));
        CHECK(live.runtime.terrain.earth_fixed_placement.grid_origin_earth_m.x_m == Approx(3 * live.time_s));
        CHECK(live.runtime.terrain.earth_fixed_placement.grid_origin_earth_m.y_m == Approx(-live.time_s));
        CHECK(live.runtime.compute.backend.get() == device);
        CHECK_FALSE(live.runtime.compute.fallback_active);
        REQUIRE(live.runtime.boundary_conditions.execution_operator);
    }
}
} // namespace
TEST_CASE("Native Earth-fixed step publishes atomically and preserves the Metal owner",
          "[metal][earth_fixed][transaction]")
{
    exercise("metal");
}
TEST_CASE("Native Earth-fixed step publishes atomically and preserves the Vulkan owner",
          "[vulkan][earth_fixed][transaction]")
{
    exercise("vulkan");
}
