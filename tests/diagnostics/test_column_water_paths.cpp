/**
 * @file test_column_water_paths.cpp
 * @brief Qualification tests for terrain-aware dry-density water paths.
 */
#include "catch2/catch.hpp"

#include "core/runtime/simulation_state.hpp"
#include "diagnostics/column_water_paths.hpp"

#include <limits>

namespace
{

void resize_water_fields(SimulationStorageState& storage)
{
    const int nr  = storage.grid.nr;
    const int nth = storage.grid.nth;
    const int nz  = storage.grid.nz;
    storage.fields.rho.resize(nr, nth, nz, 1.0f);
    storage.fields.qv.resize(nr, nth, nz, 0.01f);
    storage.fields.qc.resize(nr, nth, nz, 0.001f);
    storage.fields.qr.resize(nr, nth, nz, 0.002f);
    storage.fields.qi.resize(nr, nth, nz, 0.003f);
    storage.fields.qs.resize(nr, nth, nz, 0.004f);
    storage.fields.qg.resize(nr, nth, nz, 0.005f);
    storage.fields.qh.resize(nr, nth, nz, 0.006f);
}

tmv::diagnostics::ColumnWaterPathFields water_fields(const SimulationStorageState& storage)
{
    return {
        storage.fields.rho,
        storage.fields.qv,
        storage.fields.qc,
        storage.fields.qr,
        storage.fields.qi,
        storage.fields.qs,
        storage.fields.qg,
        storage.fields.qh,
    };
}

tmv::diagnostics::ColumnWaterPathSelection all_paths()
{
    return {true, true, true, true};
}

} // namespace

TEST_CASE("column water paths use exact terrain layer thickness and dry density",
          "[diagnostics][column_water_paths][terrain]")
{
    SimulationStorageState storage{{2, 1, 3, 1000.0, 100.0, 1.0, 2000.0}};
    storage.grid.geometry.initialize(2, 1, 3, 1000.0, 100.0, 2000.0, CoordinateSystem::Cartesian, StaggerType::CGrid);
    resize_water_fields(storage);
    for (int i = 0; i < 2; ++i)
    {
        storage.fields.rho(i, 0, 0) = 1.0f;
        storage.fields.rho(i, 0, 1) = 0.8f;
        storage.fields.rho(i, 0, 2) = 0.6f;
    }

    auto& terrain            = storage.runtime.terrain;
    terrain.config.scheme_id = "synthetic_terrain";
    terrain.metrics.z.resize(2, 1, 3, 0.0);
    terrain.metrics.z_face.resize(2, 1, 4, 0.0);
    const double faces[2][4] = {
        {100.0, 200.0, 350.0, 500.0},
        {300.0, 450.0, 500.0, 600.0},
    };
    for (int i = 0; i < 2; ++i)
        for (int k = 0; k < 3; ++k)
        {
            terrain.metrics.z_face(i, 0, k) = faces[i][k];
            terrain.metrics.z(i, 0, k)      = 0.5 * (faces[i][k] + faces[i][k + 1]);
        }
    for (int i = 0; i < 2; ++i)
        terrain.metrics.z_face(i, 0, 3) = faces[i][3];

    const SimulationStateView                         state = make_simulation_state_view(storage);
    const tmv::diagnostics::ColumnWaterPathIntegrator paths(state, water_fields(storage), all_paths());

    tmv::diagnostics::ColumnWaterPathResult first;
    REQUIRE(paths.evaluate(0, 0, 0, 3, first));
    // dry-air mass/area = 1*100 + 0.8*150 + 0.6*150 = 310 kg/m^2
    REQUIRE(first.precipitable_water_mm == Approx(3.1).epsilon(1.0e-6));
    REQUIRE(first.cloud_water_path_kgm2 == Approx(0.31).epsilon(1.0e-6));
    REQUIRE(first.liquid_water_path_kgm2 == Approx(0.93).epsilon(1.0e-6));
    REQUIRE(first.total_condensate_path_kgm2 == Approx(6.51).epsilon(1.0e-6));

    tmv::diagnostics::ColumnWaterPathResult second;
    REQUIRE(paths.evaluate(1, 0, 0, 3, second));
    // dry-air mass/area = 1*150 + 0.8*50 + 0.6*100 = 250 kg/m^2
    REQUIRE(second.precipitable_water_mm == Approx(2.5).epsilon(1.0e-6));
    REQUIRE(second.cloud_water_path_kgm2 == Approx(0.25).epsilon(1.0e-6));
    REQUIRE(second.liquid_water_path_kgm2 == Approx(0.75).epsilon(1.0e-6));
    REQUIRE(second.total_condensate_path_kgm2 == Approx(5.25).epsilon(1.0e-6));

    tmv::diagnostics::ColumnWaterPathCache cache(state, water_fields(storage), all_paths(), 0, 3);
    const auto& pwat       = cache.values(tmv::diagnostics::ColumnWaterPathKind::PrecipitableWater);
    const auto& cwp        = cache.values(tmv::diagnostics::ColumnWaterPathKind::CloudWater);
    const auto& lwp        = cache.values(tmv::diagnostics::ColumnWaterPathKind::LiquidWater);
    const auto& condensate = cache.values(tmv::diagnostics::ColumnWaterPathKind::TotalCondensate);
    REQUIRE(pwat.size() == 2);
    REQUIRE(cwp.size() == 2);
    REQUIRE(lwp.size() == 2);
    REQUIRE(condensate.size() == 2);
    REQUIRE(pwat[0] == Approx(3.1f).epsilon(1.0e-6));
    REQUIRE(pwat[1] == Approx(2.5f).epsilon(1.0e-6));
    REQUIRE(cwp[0] == Approx(0.31f).epsilon(1.0e-6));
    REQUIRE(lwp[0] == Approx(0.93f).epsilon(1.0e-6));
    REQUIRE(condensate[0] == Approx(6.51f).epsilon(1.0e-6));
}

TEST_CASE("column water paths preserve flat-grid units and ignore horizontal cell area",
          "[diagnostics][column_water_paths][flat]")
{
    SimulationStorageState storage{{1, 1, 3, 4000.0, 100.0, 1.0, 2500.0}};
    storage.grid.geometry.initialize(1, 1, 3, 4000.0, 100.0, 2500.0, CoordinateSystem::Cartesian, StaggerType::CGrid);
    resize_water_fields(storage);

    const SimulationStateView                         state = make_simulation_state_view(storage);
    const tmv::diagnostics::ColumnWaterPathIntegrator paths(state, water_fields(storage), all_paths());
    tmv::diagnostics::ColumnWaterPathResult           result;
    REQUIRE(paths.evaluate(0, 0, 0, 3, result));
    REQUIRE(result.precipitable_water_mm == Approx(3.0).epsilon(1.0e-6));
    REQUIRE(result.cloud_water_path_kgm2 == Approx(0.3).epsilon(1.0e-6));
    REQUIRE(result.liquid_water_path_kgm2 == Approx(0.9).epsilon(1.0e-6));
    REQUIRE(result.total_condensate_path_kgm2 == Approx(6.3).epsilon(1.0e-6));
}

TEST_CASE("column water paths fail closed but do not inspect unrequested species",
          "[diagnostics][column_water_paths][validation]")
{
    SimulationStorageState storage{{1, 1, 2, 1000.0, 100.0, 1.0, 1000.0}};
    storage.grid.geometry.initialize(1, 1, 2, 1000.0, 100.0, 1000.0, CoordinateSystem::Cartesian, StaggerType::CGrid);
    resize_water_fields(storage);
    storage.fields.qi(0, 0, 0) = std::numeric_limits<float>::quiet_NaN();

    const SimulationStateView               state = make_simulation_state_view(storage);
    tmv::diagnostics::ColumnWaterPathResult result;

    const tmv::diagnostics::ColumnWaterPathIntegrator vapor_only(
        state, water_fields(storage), {true, false, false, false});
    REQUIRE(vapor_only.evaluate(0, 0, 0, 2, result));
    REQUIRE(result.precipitable_water_mm == Approx(2.0).epsilon(1.0e-6));

    const tmv::diagnostics::ColumnWaterPathIntegrator every_path(state, water_fields(storage), all_paths());
    REQUIRE_FALSE(every_path.evaluate(0, 0, 0, 2, result));

    storage.fields.qi(0, 0, 0) = 0.0f;
    storage.fields.qv(0, 0, 1) = -1.0e-6f;
    const tmv::diagnostics::ColumnWaterPathIntegrator invalid_vapor(
        state, water_fields(storage), {true, false, false, false});
    REQUIRE_FALSE(invalid_vapor.evaluate(0, 0, 0, 2, result));

    tmv::diagnostics::ColumnWaterPathCache invalid_cache(
        state, water_fields(storage), {true, false, false, false}, 0, 2);
    REQUIRE(invalid_cache.values(tmv::diagnostics::ColumnWaterPathKind::PrecipitableWater).size() == 1);
    REQUIRE_FALSE(invalid_cache.valid());
}

TEST_CASE("column water path field shapes are validated once at construction",
          "[diagnostics][column_water_paths][validation]")
{
    SimulationStorageState storage{{1, 1, 2, 1000.0, 100.0, 1.0, 1000.0}};
    storage.grid.geometry.initialize(1, 1, 2, 1000.0, 100.0, 1000.0, CoordinateSystem::Cartesian, StaggerType::CGrid);
    resize_water_fields(storage);
    storage.fields.qv.resize(1, 1, 1, 0.01f);
    const SimulationStateView state = make_simulation_state_view(storage);

    REQUIRE_THROWS_WITH(
        tmv::diagnostics::ColumnWaterPathIntegrator(state, water_fields(storage), {true, false, false, false}),
        Catch::Contains("qv"));
}
