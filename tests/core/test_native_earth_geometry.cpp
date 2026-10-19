#include "catch2/catch.hpp"
#include "compute/backend_runtime.hpp"
#include "compute/earth_geometry_contract.hpp"
#include "terrain/native_earth_geometry.hpp"
#include "terrain/native_conservative_remap.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <omp.h>

namespace
{
void compare(const tmv::EarthFixedTerrainGeometry& actual, const tmv::EarthFixedTerrainGeometry& expected)
{
    const auto&                                      a = actual.metrics;
    const auto&                                      e = expected.metrics;
    const std::array<const tmv::TerrainField3D*, 10> lhs{
        &a.z, &a.J, &a.mx, &a.my, &a.zeta, &a.z_face, &a.Jx_face, &a.Jy_face, &a.Gx_face, &a.Gy_face};
    const std::array<const tmv::TerrainField3D*, 10> rhs{
        &e.z, &e.J, &e.mx, &e.my, &e.zeta, &e.z_face, &e.Jx_face, &e.Jy_face, &e.Gx_face, &e.Gy_face};
    for (std::size_t field = 0; field < lhs.size(); ++field)
    {
        REQUIRE(lhs[field]->nr == rhs[field]->nr);
        REQUIRE(lhs[field]->nth == rhs[field]->nth);
        REQUIRE(lhs[field]->nz == rhs[field]->nz);
        REQUIRE(lhs[field]->size() == rhs[field]->size());
        double scale = 0, error = 0;
        for (std::size_t n = 0; n < lhs[field]->size(); ++n)
        {
            REQUIRE(std::isfinite(lhs[field]->values[n]));
            scale = std::max(scale, std::abs(rhs[field]->values[n]));
            error = std::max(error, std::abs(lhs[field]->values[n] - rhs[field]->values[n]));
        }
        CAPTURE(field, error, scale);
        CHECK(error <= 3e-12 * std::max(1.0, scale));
    }
    CHECK(a.dx == e.dx);
    CHECK(a.dy == e.dy);
    CHECK(a.dzeta == e.dzeta);
    for (std::size_t i = 0; i < actual.topography.h.size(); ++i)
        for (std::size_t j = 0; j < actual.topography.h[i].size(); ++j)
        {
            CHECK(actual.topography.h[i][j] == Approx(expected.topography.h[i][j]).margin(1e-9));
            CHECK(actual.topography.hx[i][j] == Approx(expected.topography.hx[i][j]).margin(1e-12));
            CHECK(actual.topography.hy[i][j] == Approx(expected.topography.hy[i][j]).margin(1e-12));
        }
}
void commit_ownership(TerrainComputeKernels& kernels)
{
    using namespace tmv;
    TerrainRuntimeState live;
    live.config.scheme_id               = "bell";
    live.config.coord_id                = "smoothed";
    live.config.horizontal_placement_id = "earth_fixed";
    live.config.ztop                    = 2000;
    live.config.bell.h0                 = 200;
    live.config.bell.a                  = 850;
    live.config.bell.axisymmetric       = true;
    EarthFixedTerrainBuildRequest request;
    request.nr                  = 7;
    request.nth                 = 6;
    request.nz                  = 8;
    request.dx_m                = 300;
    request.dy_m                = 250;
    request.sampling            = EarthFixedTerrainSampling::CellAverageBilinear;
    request.grid_origin_earth_m = {-300, -250};
    auto initial                = prepare_earth_fixed_terrain_commit(live, request, kernels);
    REQUIRE(live.metrics.J.empty());
    REQUIRE_FALSE(live.earth_fixed_placement.active);
    REQUIRE(commit_earth_fixed_terrain(initial, live));
    CHECK_FALSE(commit_earth_fixed_terrain(initial, live));
    CHECK(live.earth_fixed_placement.generation == 1);
    domain::HorizontalDomainFrame frame;
    frame.earth_origin_at_t0_m = request.grid_origin_earth_m;
    frame.velocity_ms          = {14, -5};
    for (int step = 0; step < 3; ++step)
    {
        const auto   before        = live.metrics.J.values;
        const auto   origin_before = live.earth_fixed_placement.grid_origin_earth_m;
        const auto   generation    = live.earth_fixed_placement.generation;
        const double begin = .5 * step, end = .5 * (step + 1);
        auto         first = prepare_earth_fixed_terrain_motion(live, frame, begin, end, kernels);
        auto         stale = prepare_earth_fixed_terrain_motion(live, frame, begin, end, kernels);
        CHECK(live.metrics.J.values == before);
        CHECK(live.earth_fixed_placement.generation == generation);
        CHECK(live.earth_fixed_placement.grid_origin_earth_m.x_m == origin_before.x_m);
        CHECK(live.earth_fixed_placement.grid_origin_earth_m.y_m == origin_before.y_m);
        REQUIRE(can_commit_earth_fixed_terrain(first.terrain, live));
        REQUIRE(commit_earth_fixed_terrain(first.terrain, live));
        const auto committed = live.metrics.J.values;
        CHECK_FALSE(commit_earth_fixed_terrain(stale.terrain, live));
        CHECK(live.metrics.J.values == committed);
        CHECK(live.earth_fixed_placement.generation == generation + 1);
        CHECK(live.earth_fixed_placement.grid_origin_earth_m.x_m == frame.earth_origin_at(end).x_m);
        CHECK(live.earth_fixed_placement.grid_origin_earth_m.y_m == frame.earth_origin_at(end).y_m);
        TerrainComputeKernels unavailable;
        REQUIRE_THROWS(prepare_earth_fixed_terrain_motion(live, frame, end, end + .5, unavailable));
        CHECK(live.metrics.J.values == committed);
        CHECK(live.earth_fixed_placement.generation == generation + 1);
        REQUIRE_THROWS(prepare_earth_fixed_terrain_motion(live, frame, begin, end, kernels));
        CHECK(live.metrics.J.values == committed);
    }
    const auto committed = live.metrics.J.values;
    auto       candidate = prepare_earth_fixed_terrain_motion(live, frame, 1.5, 2, kernels);
    live.config.bell.h0 += 1;
    CHECK_FALSE(commit_earth_fixed_terrain(candidate.terrain, live));
    CHECK(live.metrics.J.values == committed);
    live.config.bell.h0 -= 1;
    live.earth_fixed_placement.generation = std::numeric_limits<std::uint64_t>::max();
    REQUIRE_THROWS(prepare_earth_fixed_terrain_motion(live, frame, 1.5, 2, kernels));
    CHECK(live.metrics.J.values == committed);
}
void exercise(const char* backend)
{
    ComputeBackendRuntimeState runtime;
    runtime.config.backend        = backend;
    runtime.config.allow_fallback = false;
    std::string error;
    REQUIRE(initialize_compute_backend_runtime(runtime, CoordinateSystem::Cartesian, error));
    INFO(error);
    REQUIRE_FALSE(runtime.fallback_active);
    auto* kernels = runtime.backend->terrain_kernels();
    REQUIRE(kernels);
    REQUIRE(kernels->supports_earth_geometry());
    commit_ownership(*kernels);
    using namespace tmv;
    constexpr int nx = 7, ny = 6, nz = 7;
    for (int scheme : {0, 1, 2})
        for (const char* coordinate : {"btf", "smoothed"})
            for (double sx : {-1.0, -.37, 0.0, .63, 1.0})
            {
                CAPTURE(backend, scheme, coordinate, sx);
                TerrainConfig config;
                config.scheme_id                = scheme == 2 ? "schar" : "bell";
                config.coord_id                 = coordinate;
                config.horizontal_placement_id  = "earth_fixed";
                config.bell.axisymmetric        = scheme == 1;
                config.bell.h0                  = 200;
                config.bell.a                   = 850;
                config.schar.H                  = 200;
                config.schar.a                  = 850;
                config.schar.ell                = 975;
                config.ztop                     = 2400;
                config.smoothing_decay          = .25;
                config.feature_center_earth_x_m = 1e6 + 650;
                config.feature_center_earth_y_m = -1e6 + 400;
                EarthFixedTerrainBuildRequest request;
                request.nr                     = nx;
                request.nth                    = ny;
                request.nz                     = nz;
                request.dx_m                   = 333.25;
                request.dy_m                   = 271.125;
                request.grid_origin_earth_m    = {1e6 - 80, -1e6 + 100};
                request.feature_center_earth_m = {config.feature_center_earth_x_m, config.feature_center_earth_y_m};
                request.sampling               = EarthFixedTerrainSampling::CellAverageBilinear;
                auto native                    = terrain_native::build_earth_geometry(*kernels, config, request);
                auto cpu                       = build_earth_fixed_volume_geometry(config, request);
                compare(native, cpu);
                for (int step = 0; step < 3; ++step)
                {
                    const double                  sy = step == 0 ? -sx : step == 1 ? .23 : 0;
                    const domain::HorizontalPoint end{request.grid_origin_earth_m.x_m + sx * request.dx_m,
                                                      request.grid_origin_earth_m.y_m + sy * request.dy_m};
                    auto expected = build_earth_fixed_volume_geometry_interval(config, request, end, .5, &cpu.metrics);
                    auto actual   = terrain_native::build_earth_geometry_interval(
                        *kernels, config, request, end, .5, &native.metrics);
                    compare(actual.end, expected.end);
                    compare(actual.stage, expected.stage);
                    for (const auto pair : {std::make_pair(&actual.swept_volumes.x, &expected.swept_volumes.x),
                                            std::make_pair(&actual.swept_volumes.y, &expected.swept_volumes.y)})
                    {
                        REQUIRE(pair.first->size() == pair.second->size());
                        for (std::size_t n = 0; n < pair.first->size(); ++n)
                            CHECK(pair.first->values[n] == Approx(pair.second->values[n]).epsilon(3e-12).margin(1e-7));
                    }
                    Field3D    density(nx, ny, nz, 1), result;
                    const auto report = terrain_native::remap_inventory(*kernels,
                                                                        density,
                                                                        result,
                                                                        density,
                                                                        density,
                                                                        native.metrics,
                                                                        actual.end.metrics,
                                                                        actual.swept_volumes,
                                                                        false,
                                                                        terrain_native::RemapLowerBound::Positive);
                    for (std::size_t n = 0; n < result.size(); ++n)
                        CHECK(result.data()[n] == 1);
                    CHECK(std::abs(report.final_inventory - report.initial_inventory + report.boundary_outflow -
                                   report.storage_rounding) < 3e-12 * report.initial_inventory);
                    if (step == 0)
                    {
                        const int workers = omp_get_max_threads();
                        omp_set_num_threads(4);
                        const auto repeated = terrain_native::build_earth_geometry_interval(
                            *kernels, config, request, end, .5, &native.metrics);
                        omp_set_num_threads(workers);
                        CHECK(repeated.end.metrics.z_face.values == actual.end.metrics.z_face.values);
                        CHECK(repeated.swept_volumes.x.values == actual.swept_volumes.x.values);
                    }
                    native                      = std::move(actual.end);
                    cpu                         = std::move(expected.end);
                    request.grid_origin_earth_m = end;
                }
                auto wrong = native.metrics;
                wrong.z_face(1, 1, 1) += 1;
                REQUIRE_THROWS(terrain_native::build_earth_geometry_interval(
                    *kernels, config, request, request.grid_origin_earth_m, .5, &wrong));
                REQUIRE_THROWS(terrain_native::build_earth_geometry_interval(
                    *kernels, config, request, request.grid_origin_earth_m, 0));
                auto far = request.grid_origin_earth_m;
                far.x_m += 2 * request.dx_m;
                REQUIRE_THROWS(terrain_native::build_earth_geometry_interval(*kernels, config, request, far, .5));
            }
    for (int levels : {127, 128})
    {
        TerrainConfig config;
        config.scheme_id               = "bell";
        config.coord_id                = "smoothed";
        config.horizontal_placement_id = "earth_fixed";
        config.bell.h0                 = 200;
        config.bell.a                  = 700;
        config.bell.axisymmetric       = true;
        config.ztop                    = 2500;
        config.smoothing_decay         = 2;
        EarthFixedTerrainBuildRequest request;
        request.nr                  = 5;
        request.nth                 = 4;
        request.nz                  = levels;
        request.dx_m                = 333.3;
        request.dy_m                = 271.12345;
        request.grid_origin_earth_m = {-333.3, -271.12345};
        request.sampling            = EarthFixedTerrainSampling::CellAverageBilinear;
        auto before                 = terrain_native::build_earth_geometry(*kernels, config, request);
        compare(before, build_earth_fixed_volume_geometry(config, request));
        for (double shift : {0.0, 1e-6, .999999999999})
        {
            const domain::HorizontalPoint end{request.grid_origin_earth_m.x_m + shift * request.dx_m,
                                              request.grid_origin_earth_m.y_m - shift * request.dy_m};
            const auto                    cpu = build_earth_fixed_volume_geometry_interval(config, request, end, .25);
            const auto                    gpu =
                terrain_native::build_earth_geometry_interval(*kernels, config, request, end, .25, &before.metrics);
            compare(gpu.end, cpu.end);
            compare(gpu.stage, cpu.stage);
        }
        config.bell.h0         = 2200;
        config.smoothing_decay = 5;
        REQUIRE_THROWS(terrain_native::build_earth_geometry(*kernels, config, request));
    }
    EarthGeometryDispatch d;
    auto&                 p = d.constants;
    p.nx                    = 3;
    p.ny                    = 3;
    p.nz                    = 3;
    p.operation             = 0;
    p.scheme                = 1;
    p.smoothed              = 1;
    p.dx_high = p.dy_high = p.dz_high        = 100;
    p.width_high                             = 200;
    p.wavelength_high                        = 400;
    p.height_high                            = 50;
    p.decay_high                             = .1f;
    const auto                         count = nimbus::compute::earth_geometry::cells(d);
    std::array<std::vector<float>, 8>  input;
    std::array<std::vector<float>, 20> output;
    for (std::size_t f = 0; f < input.size(); ++f)
    {
        input[f].assign(count, 0);
        d.input[f] = input[f].data();
    }
    for (std::size_t f = 0; f < output.size(); ++f)
    {
        output[f].assign(count, 17);
        d.output[f] = output[f].data();
    }
    const auto rejected = [&]()
    {
        CHECK_FALSE(kernels->dispatch_earth_geometry(d));
        for (const auto& field : output)
            for (float value : field)
                CHECK(value == 17);
    };
    p.nx = std::numeric_limits<int>::max();
    rejected();
    p.nx       = 3;
    d.input[0] = nullptr;
    rejected();
    d.input[0]  = input[0].data();
    d.output[1] = d.output[0] + 1;
    rejected();
    d.output[1] = output[1].data();
    d.output[2] = input[0].data();
    rejected();
    d.output[2] = output[2].data();
    input[0][0] = std::numeric_limits<float>::quiet_NaN();
    rejected();
    input[0][0] = 0;
    p.dx_high   = 1e-14f;
    rejected();
    p.dx_high    = 100;
    p.width_high = 0;
    rejected();
    p.width_high = 200;
    p.x_high     = std::numeric_limits<float>::max();
    rejected();
    p.x_high              = 0;
    p.operation           = 2;
    const auto full_count = nimbus::compute::earth_geometry::cells(d);
    for (std::size_t f = 0; f < input.size(); ++f)
    {
        input[f].assign(full_count, f % 2 == 0 && f < 6 ? 290 : 0);
        d.input[f] = input[f].data();
    }
    for (std::size_t f = 0; f < output.size(); ++f)
    {
        output[f].assign(full_count, 17);
        d.output[f] = output[f].data();
    }
    rejected();
    CHECK(runtime.backend->successful_kernel_dispatch_count(nimbus::compute::KernelId::EarthGeometry) > 0);
}
} // namespace
TEST_CASE("Native moving Earth geometry preserves the fixed surface and geometric conservation on Metal",
          "[metal][terrain][earth_geometry]")
{
    exercise("metal");
}
TEST_CASE("Native moving Earth geometry preserves the fixed surface and geometric conservation on Vulkan",
          "[vulkan][terrain][earth_geometry]")
{
    exercise("vulkan");
}
