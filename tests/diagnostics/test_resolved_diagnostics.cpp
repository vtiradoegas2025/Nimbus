#include <catch2/catch.hpp>
#include "core/runtime/simulation_state.hpp"
#include "core/infra/physical_cell_domain.hpp"
#include "core/infra/physical_grid_metrics.hpp"
#include "diagnostics/resolved_diagnostics.hpp"
#include "diagnostics/physical_reference.hpp"
#include "dynamics/eos_pressure.hpp"
#include "terrain/earth_fixed_geometry.hpp"
#include "terrain/factory.hpp"

#include <cmath>
#include <cstring>
#include <limits>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace
{
SimulationStorageState state_for(StaggerType staggering, bool mapped = false)
{
    const int              halo = staggering == StaggerType::CGrid ? 2 : 0;
    SimulationStorageState state(GridStorageState(8 + halo, 7 + halo, 6, 100.0, 100.0, 1.0, 100.0));
    state.grid.geometry.initialize(
        state.grid.nr, state.grid.nth, state.grid.nz, 100.0, 100.0, 100.0, CoordinateSystem::Cartesian, staggering);
    state.configuration.coordinate_system = CoordinateSystem::Cartesian;
    state.configuration.stagger_type      = staggering;
    state.configuration.coriolis_enabled  = false;
    const float p0                        = static_cast<float>(tmv::dynamics_eos::pressure_fp64(1.0f, 300.0f));
    state.base.p0.assign(6, p0);
    state.base.theta0.assign(6, 300.0);
    const float density = static_cast<float>(
        tmv::terrain_transport::reference_density_fp32(state.base.p0, state.base.theta0, 50.0, 100.0));
    state.base.rho0.assign(6, density);
    state.base.qv0.assign(6, 0.0);
    state.base.u0.assign(6, 0.0);
    state.base.v0.assign(6, 0.0);
    for (auto* field : {&state.fields.rho,
                        &state.fields.p,
                        &state.fields.theta,
                        &state.fields.u,
                        &state.fields.v,
                        &state.fields.w,
                        &state.fields.qv,
                        &state.fields.qc,
                        &state.fields.qr,
                        &state.fields.qi,
                        &state.fields.qs,
                        &state.fields.qg,
                        &state.fields.qh,
                        &state.fields.tracer,
                        &state.fields.tke})
        field->resize(state.grid.nr, state.grid.nth, state.grid.nz, 0.0f);
    state.fields.rho.fill(density);
    state.fields.theta.fill(300.0f);
    state.fields.p.fill(p0);
    if (mapped)
    {
        auto& terrain                          = state.runtime.terrain;
        terrain.config.scheme_id               = "bell";
        terrain.config.horizontal_placement_id = "earth_fixed";
        terrain.config.coord_id                = "smoothed";
        terrain.config.compute_derivatives = terrain.config.compute_metrics = true;
        terrain.config.ztop                                                 = 600.0;
        terrain.config.bell.h0                                              = 30.0;
        terrain.config.bell.a                                               = 250.0;
        terrain.config.bell.axisymmetric                                    = true;
        terrain.config.feature_center_earth_x_m                             = .5 * state.grid.nr * state.grid.dx;
        terrain.config.feature_center_earth_y_m                             = .5 * state.grid.nth * state.grid.dy;
        terrain.scheme                                                      = tmv::create_terrain_scheme("bell");
        terrain.scheme->initialize(terrain.config);
        tmv::EarthFixedTerrainBuildRequest request;
        request.nr                     = state.grid.nr;
        request.nth                    = state.grid.nth;
        request.nz                     = state.grid.nz;
        request.dx_m                   = state.grid.dx;
        request.dy_m                   = state.grid.dy;
        request.feature_center_earth_m = {terrain.config.feature_center_earth_x_m,
                                          terrain.config.feature_center_earth_y_m};
        const auto built               = tmv::build_earth_fixed_terrain_geometry(terrain.config, request);
        terrain.metrics                = built.metrics;
        terrain.topography             = built.topography;
    }
    return state;
}
} // namespace

TEST_CASE("resolved pressure and angular diagnostics match a steady rotating flow on both grids",
          "[diagnostics][resolved][pressure][angular][reference]")
{
    for (const auto staggering : {StaggerType::CGrid, StaggerType::Collocated})
    {
        auto         storage  = state_for(staggering);
        auto         state    = make_simulation_state_view(storage);
        const auto   physical = physical_scalar_domain(state.grid);
        const double xc = .5 * (state.grid.geometry.r[physical.i_begin] + state.grid.geometry.r[physical.i_end - 1]);
        const double yc =
            .5 * (state.grid.geometry.theta[physical.j_begin] + state.grid.geometry.theta[physical.j_end - 1]);
        constexpr double rotation      = .01;
        double           mean_pressure = 0.0;
        int              cells         = 0;
        for (int i = 0; i < state.grid.nr; ++i)
            for (int j = 0; j < state.grid.nth; ++j)
                for (int k = 0; k < state.grid.nz; ++k)
                {
                    const double x            = state.grid.geometry.r[i] - xc;
                    const double y            = state.grid.geometry.theta[j] - yc;
                    storage.fields.u(i, j, k) = static_cast<float>(-rotation * y);
                    storage.fields.v(i, j, k) = static_cast<float>(rotation * x);
                    const double pressure = .5 * storage.fields.rho(i, j, k) * rotation * rotation * (x * x + y * y);
                    storage.fields.p(i, j, k) = static_cast<float>(storage.base.p0[k] + pressure);
                    if (i >= physical.i_begin && i < physical.i_end && j >= physical.j_begin && j < physical.j_end)
                    {
                        mean_pressure += pressure;
                        ++cells;
                    }
                }
        mean_pressure /= cells;
        tmv::diagnostics::ResolvedDiagnosticReport report;
        tmv::diagnostics::compute_resolved_diagnostics(state, {true, true, true, true}, &report);
        CHECK(report.dynamic_pressure.relative_residual <= 1e-10);
        for (int i = physical.i_begin; i < physical.i_end; ++i)
            for (int j = physical.j_begin; j < physical.j_end; ++j)
                for (int k = 0; k < state.grid.nz; ++k)
                {
                    const double x        = state.grid.geometry.r[i] - xc;
                    const double y        = state.grid.geometry.theta[j] - yc;
                    const double pressure = .5 * storage.fields.rho(i, j, k) * rotation * rotation * (x * x + y * y);
                    CHECK(storage.diagnostics.dynamic_pressure(i, j, k) ==
                          Approx(pressure - mean_pressure).margin(3e-6));
                    CHECK(storage.diagnostics.buoyancy_pressure(i, j, k) == 0.0f);
                    CHECK(storage.diagnostics.angular_momentum(i, j, k) ==
                          Approx(rotation * (x * x + y * y)).margin(1e-4));
                    CHECK(std::abs(storage.diagnostics.angular_momentum_tendency(i, j, k)) < .1);
                    CHECK(storage.diagnostics.vorticity_z(i, j, k) == Approx(2 * rotation).margin(1e-7));
                }
    }
}

TEST_CASE("resolved buoyancy pressure reproduces physical hydrostatics on sloping terrain",
          "[diagnostics][resolved][pressure][terrain][reference]")
{
    for (const auto staggering : {StaggerType::CGrid, StaggerType::Collocated})
    {
        auto storage = state_for(staggering, true);
        storage.fields.qv.fill(.01f);
        auto       state    = make_simulation_state_view(storage);
        const auto physical = physical_scalar_domain(state.grid);
        tmv::diagnostics::compute_resolved_diagnostics(state, {true, true, true, true});
        double weighted_height = 0.0, total_volume = 0.0;
        for (int i = physical.i_begin; i < physical.i_end; ++i)
            for (int j = physical.j_begin; j < physical.j_end; ++j)
                for (int k = physical.k_begin; k < physical.k_end; ++k)
                {
                    const double volume = tmv::physical_grid_metrics::cell_volume(state, i, j, k);
                    weighted_height += volume * tmv::physical_grid_metrics::scalar_height(state, i, j, k);
                    total_volume += volume;
                }
        const double mean_height = weighted_height / total_volume;
        const double buoyancy    = physical_constants::gravity_ms2 * .608 * static_cast<double>(.01f);
        for (int i = physical.i_begin; i < physical.i_end; ++i)
            for (int j = physical.j_begin; j < physical.j_end; ++j)
                for (int k = physical.k_begin; k < physical.k_end; ++k)
                {
                    const double expected = state.fields.rho(i, j, k) * buoyancy *
                                            (tmv::physical_grid_metrics::scalar_height(state, i, j, k) - mean_height);
                    CHECK(storage.diagnostics.buoyancy_pressure(i, j, k) == Approx(expected).margin(3e-6));
                    CHECK(storage.diagnostics.dynamic_pressure(i, j, k) == 0.0f);
                    CHECK(storage.diagnostics.p_prime(i, j, k) == 0.0f);
                }
    }
}

TEST_CASE("resolved diagnostics publish atomically and independently of worker count",
          "[diagnostics][resolved][atomic][threads]")
{
    auto storage = state_for(StaggerType::CGrid, true);
    auto state   = make_simulation_state_view(storage);
    storage.fields.qv.fill(.01f);
#ifdef _OPENMP
    omp_set_num_threads(1);
#endif
    tmv::diagnostics::compute_resolved_diagnostics(state, {true, true, true, true});
    const auto before = storage.diagnostics;
#ifdef _OPENMP
    omp_set_num_threads(4);
#endif
    tmv::diagnostics::compute_resolved_diagnostics(state, {true, true, true, true});
    CHECK(std::memcmp(before.buoyancy_pressure.data(),
                      storage.diagnostics.buoyancy_pressure.data(),
                      before.buoyancy_pressure.size() * sizeof(float)) == 0);
    storage.fields.qv(2, 2, 2) = -1.0f;
    tmv::diagnostics::ResolvedDiagnosticReport report;
    report.dynamic_pressure.iterations = -22;
    REQUIRE_THROWS(tmv::diagnostics::compute_resolved_diagnostics(state, {true, true, true, true}, &report));
    CHECK(report.dynamic_pressure.iterations == -22);
    CHECK(std::memcmp(before.buoyancy_pressure.data(),
                      storage.diagnostics.buoyancy_pressure.data(),
                      before.buoyancy_pressure.size() * sizeof(float)) == 0);
    CHECK(std::memcmp(before.vorticity_z.data(),
                      storage.diagnostics.vorticity_z.data(),
                      before.vorticity_z.size() * sizeof(float)) == 0);
}

TEST_CASE("collocated mapped vorticity removes coordinate slope from physical vertical shear",
          "[diagnostics][resolved][vorticity][terrain]")
{
    auto             storage = state_for(StaggerType::Collocated, true);
    auto             state   = make_simulation_state_view(storage);
    constexpr double shear   = .01;
    for (int i = 0; i < state.grid.nr; ++i)
        for (int j = 0; j < state.grid.nth; ++j)
            for (int k = 0; k < state.grid.nz; ++k)
                storage.fields.u(i, j, k) =
                    static_cast<float>(shear * tmv::physical_grid_metrics::scalar_height(state, i, j, k));
    tmv::diagnostics::compute_resolved_diagnostics(state, {true, false, false, false});
    const auto domain = physical_scalar_domain(state.grid);
    for (int i = domain.i_begin; i < domain.i_end; ++i)
        for (int j = domain.j_begin; j < domain.j_end; ++j)
            for (int k = domain.k_begin; k < domain.k_end; ++k)
            {
                CHECK(storage.diagnostics.vorticity_r(i, j, k) == 0.0f);
                CHECK(storage.diagnostics.vorticity_theta(i, j, k) == Approx(shear).margin(1e-7));
                CHECK(std::abs(storage.diagnostics.vorticity_z(i, j, k)) < 1e-7);
            }
}

TEST_CASE("pressure perturbation alone does not require velocity or component forcing",
          "[diagnostics][resolved][pressure][selection]")
{
    auto storage = state_for(StaggerType::CGrid, true);
    storage.fields.u.release_storage();
    storage.fields.qv.fill(-1.0f);
    storage.diagnostics.dynamic_pressure.resize(storage.grid.nr, storage.grid.nth, storage.grid.nz, 17.0f);
    auto state = make_simulation_state_view(storage);
    tmv::diagnostics::compute_resolved_diagnostics(state, {false, false, true, false});
    CHECK(storage.diagnostics.p_prime(2, 2, 2) == 0.0f);
    CHECK(storage.diagnostics.dynamic_pressure(2, 2, 2) == 17.0f);
}

TEST_CASE("presentation vorticity and pressure perturbation do not request pressure potentials",
          "[diagnostics][resolved][pressure][selection][presentation]")
{
    auto storage = state_for(StaggerType::CGrid);
    storage.fields.qv.fill(-1.0f);
    storage.fields.p(2, 2, 2) += 8.0f;
    storage.diagnostics.dynamic_pressure.resize(storage.grid.nr, storage.grid.nth, storage.grid.nz, 17.0f);
    storage.diagnostics.buoyancy_pressure.resize(storage.grid.nr, storage.grid.nth, storage.grid.nz, 19.0f);
    tmv::diagnostics::ResolvedDiagnosticReport report;
    auto state = make_simulation_state_view(storage);
    REQUIRE_NOTHROW(tmv::diagnostics::compute_resolved_diagnostics(state, {true, false, true, false}, &report));
    CHECK(storage.diagnostics.p_prime(2, 2, 2) == 8.0f);
    CHECK(storage.diagnostics.vorticity_z(2, 2, 2) == 0.0f);
    CHECK(storage.diagnostics.dynamic_pressure(2, 2, 2) == 17.0f);
    CHECK(storage.diagnostics.buoyancy_pressure(2, 2, 2) == 19.0f);
    CHECK(report.dynamic_pressure.iterations == 0);
    CHECK(report.buoyancy_pressure.iterations == 0);
    REQUIRE_THROWS(tmv::diagnostics::compute_resolved_diagnostics(state, {true, false, true, true}));
}
