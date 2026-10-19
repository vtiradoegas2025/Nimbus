// Analytical terrain checks retain the exact Bell profiles:
// axisymmetric h=h0*a^3/(a^2+x^2+y^2)^1.5 and ridge
// h=h0/(1+(x/a)^2), including peaks and spatial derivatives.
#include "catch2/catch.hpp"
#include "terrain/terrain_base.hpp"
#include "terrain/earth_fixed_geometry.hpp"
#include "terrain/base/topography.hpp"
#include "terrain/conservative_transport.hpp"
#include "core/infra/grid_coordinates.hpp"
#include "core/runtime/simulation.hpp"
#include "numerics/advection/advection_cartesian.hpp"
#include "numerics/time_stepping/time_stepping_base.hpp"

#include <array>
#include <cmath>
#include <iomanip>
#include <limits>
#include <utility>

namespace
{

void setup_grid()
{
    NR = 16;
    NTH = 16;
    NZ = 16;
    dr = 1000.0;
    dz = 500.0;
    dtheta = 1000.0;
}

void bind_earth_fixed_feature(
    TerrainConfig& config,
    const tmv::EarthFixedTerrainBuildRequest& request)
{
    config.horizontal_placement_id = "earth_fixed";
    config.feature_center_earth_x_m = request.feature_center_earth_m.x_m;
    config.feature_center_earth_y_m = request.feature_center_earth_m.y_m;
}

} // namespace

TEST_CASE("dynamics_supports_terrain admits both mapped Cartesian layouts",
          "[physics][terrain][guard]")
{
    CHECK(dynamics_supports_terrain(
        CoordinateSystem::Cartesian, StaggerType::CGrid));
    CHECK(dynamics_supports_terrain(CoordinateSystem::Cartesian, StaggerType::Collocated));
}

// ---- Factory tests ----

TEST_CASE("Terrain factory creates 'none' scheme", "[physics][terrain]")
{
    setup_grid();
    auto scheme = create_terrain_scheme("none");
    REQUIRE(scheme != nullptr);
}

TEST_CASE("Terrain factory creates 'bell' scheme", "[physics][terrain]")
{
    setup_grid();
    auto scheme = create_terrain_scheme("bell");
    REQUIRE(scheme != nullptr);
}

TEST_CASE("Terrain factory creates 'schar' scheme", "[physics][terrain]")
{
    setup_grid();
    auto scheme = create_terrain_scheme("schar");
    REQUIRE(scheme != nullptr);
}

TEST_CASE("legacy terrain globals reference the owning runtime",
          "[physics][terrain][state][modularity]")
{
    REQUIRE(&terrain_scheme == &global_terrain_runtime_state.scheme);
    REQUIRE(&global_terrain_config == &global_terrain_runtime_state.config);
    REQUIRE(&global_topography == &global_terrain_runtime_state.topography);
    REQUIRE(&global_terrain_metrics == &global_terrain_runtime_state.metrics);
}

TEST_CASE("terrain runtime copies preserve independent built-in schemes",
          "[physics][terrain][state][snapshot]")
{
    TerrainRuntimeState original;
    original.config.scheme_id = "none";
    original.scheme = create_terrain_scheme("none");
    original.scheme->initialize(original.config);
    original.topography.h = {{0.0}};
    original.metrics.J.resize(1, 1, 1, 1.0);
    original.earth_fixed_placement.active = true;
    original.earth_fixed_placement.grid_origin_earth_m = {1200.0, -500.0};
    original.earth_fixed_placement.feature_center_earth_m = {2500.0, 3000.0};
    original.earth_fixed_placement.generation = 7;

    TerrainRuntimeState copy(original);
    REQUIRE(copy.scheme != nullptr);
    REQUIRE(copy.scheme.get() != original.scheme.get());
    REQUIRE(copy.scheme->name() == "none");
    REQUIRE(copy.topography.h == original.topography.h);
    REQUIRE(copy.metrics.J.values == original.metrics.J.values);
    REQUIRE(copy.earth_fixed_placement.generation == 7);
    REQUIRE(copy.earth_fixed_placement.feature_center_earth_m.y_m == 3000.0);

    copy.topography.h[0][0] = 5.0;
    copy.earth_fixed_placement.generation = 8;
    REQUIRE(original.topography.h[0][0] == Approx(0.0));
    REQUIRE(original.earth_fixed_placement.generation == 7);
}

TEST_CASE("flat terrain keeps conservative face geometry lazily unallocated",
          "[physics][terrain][memory][flat]")
{
    setup_grid();
    TerrainConfig cfg;
    cfg.scheme_id = "none";
    Topography2D topo;
    TerrainMetrics3D metrics;
    topography::initialize_topography(topo, NR, NTH);
    topography::initialize_metrics(metrics, NR, NTH, NZ);
    auto scheme = create_terrain_scheme("none");
    scheme->initialize(cfg);
    scheme->build_topography(cfg, topo);
    scheme->build_metrics(cfg, topo, metrics);

    CHECK(metrics.z_face.empty());
    CHECK(metrics.Jx_face.empty());
    CHECK(metrics.Jy_face.empty());
    CHECK(metrics.Gx_face.empty());
    CHECK(metrics.Gy_face.empty());
}

TEST_CASE("Earth-fixed terrain geometry preserves whole-cell overlap",
          "[physics][terrain][earth_fixed][translation]")
{
    TerrainConfig config;
    config.scheme_id = "schar";
    config.coord_id = "btf";
    config.ztop = 6000.0;
    config.schar.H = 800.0;
    config.schar.a = 1800.0;
    config.schar.ell = 2200.0;

    tmv::EarthFixedTerrainBuildRequest before;
    before.nr = 9;
    before.nth = 7;
    before.nz = 6;
    before.dx_m = 500.0;
    before.dy_m = 500.0;
    before.grid_origin_earth_m = {12000.0, -3000.0};
    before.feature_center_earth_m = {
        before.grid_origin_earth_m.x_m
            + tmv::grid_coordinates::scalar_center(4, before.dx_m),
        before.grid_origin_earth_m.y_m
            + tmv::grid_coordinates::scalar_center(3, before.dy_m)};
    bind_earth_fixed_feature(config, before);

    const auto old_geometry =
        tmv::build_earth_fixed_terrain_geometry(config, before);
    auto after = before;
    after.grid_origin_earth_m.x_m += before.dx_m;
    const auto new_geometry =
        tmv::build_earth_fixed_terrain_geometry(config, after);

    for (int i = 0; i + 1 < before.nr; ++i)
        for (int j = 0; j < before.nth; ++j)
        {
            const auto ui = static_cast<std::size_t>(i);
            const auto uj = static_cast<std::size_t>(j);
            CHECK(new_geometry.topography.h[ui][uj]
                  == old_geometry.topography.h[ui + 1][uj]);
            CHECK(new_geometry.topography.hx[ui][uj]
                  == old_geometry.topography.hx[ui + 1][uj]);
            for (int k = 0; k < before.nz; ++k)
            {
                CHECK(new_geometry.metrics.z(i, j, k)
                      == old_geometry.metrics.z(i + 1, j, k));
                CHECK(new_geometry.metrics.J(i, j, k)
                      == old_geometry.metrics.J(i + 1, j, k));
                CHECK(new_geometry.metrics.mx(i, j, k)
                      == old_geometry.metrics.mx(i + 1, j, k));
            }
            for (int k = 0; k <= before.nz; ++k)
                CHECK(new_geometry.metrics.z_face(i, j, k)
                      == old_geometry.metrics.z_face(i + 1, j, k));
        }

    int old_peak = 0;
    int new_peak = 0;
    for (int i = 1; i < before.nr; ++i)
    {
        if (old_geometry.topography.h[static_cast<std::size_t>(i)][3]
            > old_geometry.topography.h[static_cast<std::size_t>(old_peak)][3])
            old_peak = i;
        if (new_geometry.topography.h[static_cast<std::size_t>(i)][3]
            > new_geometry.topography.h[static_cast<std::size_t>(new_peak)][3])
            new_peak = i;
    }
    CHECK(old_peak == 4);
    CHECK(new_peak == 3);
}

TEST_CASE("Earth-fixed terrain preserves complete metrics under signed shifts",
          "[physics][terrain][earth_fixed][translation][metrics]")
{
    TerrainConfig config;
    config.scheme_id = "bell";
    config.coord_id = "smoothed";
    config.ztop = 8000.0;
    config.smoothing_decay = 0.2;
    config.bell.h0 = 700.0;
    config.bell.a = 2100.0;
    config.bell.axisymmetric = true;

    tmv::EarthFixedTerrainBuildRequest request;
    request.nr = 10;
    request.nth = 9;
    request.nz = 7;
    request.dx_m = 400.0;
    request.dy_m = 600.0;
    request.grid_origin_earth_m = {7312.5, -9125.0};
    request.feature_center_earth_m = {9120.0, -7020.0};
    bind_earth_fixed_feature(config, request);
    const auto original =
        tmv::build_earth_fixed_terrain_geometry(config, request);

    const std::array<std::pair<int, int>, 4> shifts{{
        {1, 0}, {-1, 0}, {0, 1}, {1, -1}}};
    for (const auto [shift_i, shift_j] : shifts)
    {
        auto translated_request = request;
        translated_request.grid_origin_earth_m.x_m +=
            static_cast<double>(shift_i) * request.dx_m;
        translated_request.grid_origin_earth_m.y_m +=
            static_cast<double>(shift_j) * request.dy_m;
        const auto translated = tmv::build_earth_fixed_terrain_geometry(
            config, translated_request);

        for (int i = 1; i < request.nr - 1; ++i)
            for (int j = 1; j < request.nth - 1; ++j)
            {
                const int source_i = i + shift_i;
                const int source_j = j + shift_j;
                if (source_i < 1 || source_i >= request.nr - 1
                    || source_j < 1 || source_j >= request.nth - 1)
                    continue;
                const auto ui = static_cast<std::size_t>(i);
                const auto uj = static_cast<std::size_t>(j);
                const auto usi = static_cast<std::size_t>(source_i);
                const auto usj = static_cast<std::size_t>(source_j);
                CHECK(translated.topography.h[ui][uj]
                      == original.topography.h[usi][usj]);
                CHECK(translated.topography.hx[ui][uj]
                      == original.topography.hx[usi][usj]);
                CHECK(translated.topography.hy[ui][uj]
                      == original.topography.hy[usi][usj]);
                for (int k = 0; k < request.nz; ++k)
                {
                    CHECK(translated.metrics.z(i, j, k)
                          == original.metrics.z(source_i, source_j, k));
                    CHECK(translated.metrics.J(i, j, k)
                          == original.metrics.J(source_i, source_j, k));
                    CHECK(translated.metrics.mx(i, j, k)
                          == original.metrics.mx(source_i, source_j, k));
                    CHECK(translated.metrics.my(i, j, k)
                          == original.metrics.my(source_i, source_j, k));
                    CHECK(translated.metrics.zeta(i, j, k)
                          == original.metrics.zeta(source_i, source_j, k));
                    CHECK(translated.metrics.Jx_face(i, j, k)
                          == original.metrics.Jx_face(
                              source_i, source_j, k));
                    CHECK(translated.metrics.Jy_face(i, j, k)
                          == original.metrics.Jy_face(
                              source_i, source_j, k));
                }
                for (int k = 0; k <= request.nz; ++k)
                {
                    CHECK(translated.metrics.z_face(i, j, k)
                          == original.metrics.z_face(
                              source_i, source_j, k));
                    CHECK(translated.metrics.Gx_face(i, j, k)
                          == original.metrics.Gx_face(
                              source_i, source_j, k));
                    CHECK(translated.metrics.Gy_face(i, j, k)
                          == original.metrics.Gy_face(
                              source_i, source_j, k));
                }
            }
    }
}

TEST_CASE("Earth-fixed terrain geometry rejects incomplete candidates",
          "[physics][terrain][earth_fixed][transaction]")
{
    TerrainConfig config;
    config.scheme_id = "bell";
    config.coord_id = "btf";
    config.ztop = 5000.0;
    tmv::EarthFixedTerrainBuildRequest request;
    request.nr = 6;
    request.nth = 6;
    request.nz = 5;
    request.dx_m = 500.0;
    request.dy_m = 500.0;
    bind_earth_fixed_feature(config, request);

    const auto live = tmv::build_earth_fixed_terrain_geometry(config, request);
    const auto live_height = live.topography.h;
    const auto live_jacobian = live.metrics.J.values;

    config.compute_derivatives = false;
    CHECK_THROWS_AS(
        tmv::build_earth_fixed_terrain_geometry(config, request),
        std::invalid_argument);
    CHECK(live.topography.h == live_height);
    CHECK(live.metrics.J.values == live_jacobian);

    config.compute_derivatives = true;
    config.bell.a = 0.0;
    CHECK_THROWS_AS(
        tmv::build_earth_fixed_terrain_geometry(config, request),
        std::invalid_argument);
    CHECK(live.topography.h == live_height);
    CHECK(live.metrics.J.values == live_jacobian);

    config.bell.a = 5000.0;
    config.bell.h0 = config.ztop;
    CHECK_THROWS_AS(
        tmv::build_earth_fixed_terrain_geometry(config, request),
        std::invalid_argument);
    CHECK(live.topography.h == live_height);
    CHECK(live.metrics.J.values == live_jacobian);

    config.bell.h0 = 1000.0;
    request.grid_origin_earth_m.x_m =
        std::numeric_limits<double>::quiet_NaN();
    CHECK_THROWS_AS(
        tmv::build_earth_fixed_terrain_geometry(config, request),
        std::invalid_argument);
    CHECK(live.topography.h == live_height);
    CHECK(live.metrics.J.values == live_jacobian);

    request.grid_origin_earth_m.x_m = 0.0;
    request.feature_center_earth_m.x_m =
        -std::numeric_limits<double>::max();
    request.grid_origin_earth_m.x_m =
        std::numeric_limits<double>::max();
    bind_earth_fixed_feature(config, request);
    CHECK_THROWS_AS(
        tmv::build_earth_fixed_terrain_geometry(config, request),
        std::overflow_error);
    CHECK(live.topography.h == live_height);
    CHECK(live.metrics.J.values == live_jacobian);

    request.grid_origin_earth_m.x_m = 0.0;
    request.feature_center_earth_m.x_m = 0.0;
    bind_earth_fixed_feature(config, request);
    request.nr = std::numeric_limits<int>::max();
    CHECK_THROWS_AS(
        tmv::build_earth_fixed_terrain_geometry(config, request),
        std::overflow_error);
    CHECK(live.topography.h == live_height);
    CHECK(live.metrics.J.values == live_jacobian);
}

TEST_CASE("Earth-fixed terrain placement publishes one complete generation",
          "[physics][terrain][earth_fixed][transaction][placement]")
{
    TerrainConfig config;
    config.scheme_id = "bell";
    config.coord_id = "smoothed";
    config.ztop = 6000.0;
    config.smoothing_decay = 0.15;
    config.bell.h0 = 500.0;
    config.bell.a = 1900.0;
    config.bell.axisymmetric = true;

    tmv::TerrainRuntimeState live;
    live.config = config;
    live.topography.h = {{-7.0}};
    live.metrics.J.resize(1, 1, 1, -9.0);

    tmv::EarthFixedTerrainBuildRequest request;
    request.nr = 8;
    request.nth = 7;
    request.nz = 6;
    request.dx_m = 400.0;
    request.dy_m = 500.0;
    request.grid_origin_earth_m = {1200.0, -3400.0};
    request.feature_center_earth_m = {3100.0, -1800.0};
    bind_earth_fixed_feature(config, request);
    live.config = config;

    auto initial = tmv::prepare_earth_fixed_terrain_commit(live, request);
    const auto initial_height = initial.geometry.topography.h;
    const auto initial_jacobian = initial.geometry.metrics.J.values;
    CHECK_FALSE(live.earth_fixed_placement.active);
    CHECK(live.topography.h == std::vector<std::vector<double>>{{-7.0}});
    CHECK(live.metrics.J.values == std::vector<double>{-9.0});

    REQUIRE(tmv::commit_earth_fixed_terrain(initial, live));
    CHECK(live.earth_fixed_placement.active);
    CHECK(live.earth_fixed_placement.generation == 1);
    CHECK(live.earth_fixed_placement.grid_origin_earth_m.x_m
          == request.grid_origin_earth_m.x_m);
    CHECK(live.topography.h == initial_height);
    CHECK(live.metrics.J.values == initial_jacobian);
    CHECK_FALSE(tmv::commit_earth_fixed_terrain(initial, live));

    auto translated_request = request;
    translated_request.grid_origin_earth_m.x_m += request.dx_m;
    translated_request.grid_origin_earth_m.y_m -= request.dy_m;
    auto translated =
        tmv::prepare_earth_fixed_terrain_commit(live, translated_request);
    const auto translated_height = translated.geometry.topography.h;
    const auto translated_jacobian = translated.geometry.metrics.J.values;
    const auto live_before_stale_commit = live.topography.h;

    ++live.earth_fixed_placement.generation;
    CHECK_FALSE(tmv::commit_earth_fixed_terrain(translated, live));
    CHECK(live.topography.h == live_before_stale_commit);
    --live.earth_fixed_placement.generation;
    REQUIRE(tmv::commit_earth_fixed_terrain(translated, live));
    CHECK(live.earth_fixed_placement.generation == 2);
    CHECK(live.topography.h == translated_height);
    CHECK(live.metrics.J.values == translated_jacobian);
    for (int i = 0; i + 1 < request.nr; ++i)
        for (int j = 1; j < request.nth; ++j)
        {
            CHECK(live.topography.h[static_cast<std::size_t>(i)]
                                    [static_cast<std::size_t>(j)]
                  == initial_height[static_cast<std::size_t>(i + 1)]
                                   [static_cast<std::size_t>(j - 1)]);
            for (int k = 0; k < request.nz; ++k)
            {
                const auto destination =
                    (static_cast<std::size_t>(i) * request.nth + j)
                    * request.nz + k;
                const auto source =
                    (static_cast<std::size_t>(i + 1) * request.nth + j - 1)
                    * request.nz + k;
                CHECK(live.metrics.J.values[destination]
                      == initial_jacobian[source]);
            }
        }

    auto stale_config =
        tmv::prepare_earth_fixed_terrain_commit(live, translated_request);
    const auto current_height = live.topography.h;
    live.config.smoothing_decay = 0.25;
    CHECK_FALSE(tmv::commit_earth_fixed_terrain(stale_config, live));
    CHECK(live.topography.h == current_height);
    live.config = config;

    auto changed_grid = translated_request;
    ++changed_grid.nr;
    CHECK_THROWS_AS(
        tmv::prepare_earth_fixed_terrain_commit(live, changed_grid),
        std::invalid_argument);
    auto changed_feature = translated_request;
    changed_feature.feature_center_earth_m.x_m += 1.0;
    CHECK_THROWS_AS(
        tmv::prepare_earth_fixed_terrain_commit(live, changed_feature),
        std::invalid_argument);

    live.earth_fixed_placement.generation =
        std::numeric_limits<std::uint64_t>::max();
    CHECK_THROWS_AS(
        tmv::prepare_earth_fixed_terrain_commit(live, translated_request),
        std::overflow_error);
    CHECK(live.topography.h == current_height);
}

TEST_CASE("Terrain descriptors expose both mapped Cartesian layouts",
          "[physics][terrain][descriptor]")
{
    const auto& descriptors = get_terrain_scheme_descriptors();
    const auto* flat = tmv::find_scheme_descriptor(descriptors, "flat");
    const auto* bell = tmv::find_scheme_descriptor(descriptors, "bell");
    const auto* schar = tmv::find_scheme_descriptor(descriptors, "schaer");

    REQUIRE(flat != nullptr);
    REQUIRE(flat->id == "none");
    REQUIRE(flat->maturity == tmv::SchemeMaturity::Qualified);
    REQUIRE(flat->capabilities.declared);
    REQUIRE(bell != nullptr);
    REQUIRE(bell->maturity == tmv::SchemeMaturity::Qualified);
    REQUIRE(bell->runtime_selectable);
    REQUIRE(bell->capabilities.coordinate_systems
            == std::vector<std::string>{"cartesian"});
    REQUIRE(bell->capabilities.staggerings
            == std::vector<std::string>{"collocated", "c_grid"});
    REQUIRE(bell->capabilities.config_validation_contract
            == "terrain.scheme+dynamics_supports_terrain");
    REQUIRE(schar != nullptr);
    REQUIRE(schar->maturity == tmv::SchemeMaturity::Qualified);
    REQUIRE(schar->runtime_selectable);
    REQUIRE(schar->capabilities.coordinate_systems
            == std::vector<std::string>{"cartesian"});
    REQUIRE(schar->capabilities.staggerings
            == std::vector<std::string>{"collocated", "c_grid"});
    REQUIRE(schar->capabilities.config_validation_contract
            == "terrain.scheme+dynamics_supports_terrain");
}

TEST_CASE("terrain computational levels are scalar centers between physical faces",
          "[physics][terrain][coordinates]")
{
    const auto zeta = topography::build_zeta_levels(4, 2000.0);
    REQUIRE(zeta.size() == 4);
    REQUIRE(zeta[0] == Approx(250.0));
    REQUIRE(zeta[1] == Approx(750.0));
    REQUIRE(zeta[3] == Approx(1750.0));
}

TEST_CASE("terrain primitives reject parameter substitution",
          "[physics][terrain][config-truth]")
{
    SECTION("bell scale cannot be repaired with an absolute value")
    {
        TerrainConfig::BellParams params;
        params.a = -5000.0;
        REQUIRE_THROWS_AS(
            topography::eval_bell(0.0, 0.0, params), std::invalid_argument);
    }

    SECTION("Schar wavelength cannot be replaced with a default")
    {
        TerrainConfig::ScharParams params;
        params.ell = 0.0;
        REQUIRE_THROWS_AS(
            topography::eval_schar(0.0, params), std::invalid_argument);
    }

    SECTION("terrain height cannot reach the metric top")
    {
        TerrainConfig cfg;
        cfg.coord_id = "btf";
        cfg.ztop = 1000.0;
        Topography2D topo;
        topography::initialize_topography(topo, 1, 1);
        topo.h[0][0] = 1000.0;
        REQUIRE_THROWS_AS(
            topography::TerrainFollowingCoordinate(cfg, topo),
            std::invalid_argument);
    }
}

// ---- Analytical topography function tests ----

TEST_CASE("Bell axisymmetric: peak height equals h0", "[physics][terrain][analytical]")
{
    // At r=0: h(0,0) = h0 * a^3 / (a^2)^1.5 = h0 * a^3 / a^3 = h0
    TerrainConfig::BellParams params;
    params.h0 = 500.0;
    params.a = 5000.0;
    params.axisymmetric = true;

    auto result = topography::eval_bell(0.0, 0.0, params);
    REQUIRE(result.h == Approx(500.0).epsilon(1e-12));
}

TEST_CASE("Bell axisymmetric: height decays with distance", "[physics][terrain][analytical]")
{
    TerrainConfig::BellParams params;
    params.h0 = 1000.0;
    params.a = 5000.0;
    params.axisymmetric = true;

    // At r = a: h = h0 * a^3 / (a^2 + a^2)^1.5 = h0 / (2^1.5) = h0 / 2.828
    double expected_at_a = 1000.0 / std::pow(2.0, 1.5);
    auto result_at_a = topography::eval_bell(5000.0, 0.0, params);
    REQUIRE(result_at_a.h == Approx(expected_at_a).epsilon(1e-10));

    // At r = 2a: h = h0 * a^3 / (a^2 + 4a^2)^1.5 = h0 / 5^1.5
    double expected_at_2a = 1000.0 / std::pow(5.0, 1.5);
    auto result_at_2a = topography::eval_bell(10000.0, 0.0, params);
    REQUIRE(result_at_2a.h == Approx(expected_at_2a).epsilon(1e-10));

    // Monotone decay
    REQUIRE(result_at_a.h < 1000.0);
    REQUIRE(result_at_2a.h < result_at_a.h);
}

TEST_CASE("Bell axisymmetric: derivatives are zero at peak", "[physics][terrain][analytical]")
{
    TerrainConfig::BellParams params;
    params.h0 = 500.0;
    params.a = 5000.0;
    params.axisymmetric = true;

    auto result = topography::eval_bell(0.0, 0.0, params);
    REQUIRE(result.hx == Approx(0.0).margin(1e-15));
    REQUIRE(result.hy == Approx(0.0).margin(1e-15));
}

TEST_CASE("Bell axisymmetric: derivative formula dh/dx = -3*h0*a^3*x / (a^2+r^2)^2.5", "[physics][terrain][analytical]")
{
    TerrainConfig::BellParams params;
    params.h0 = 1000.0;
    params.a = 5000.0;
    params.axisymmetric = true;

    double x = 3000.0;
    double y = 4000.0;  // r = 5000 = a
    double r2 = x * x + y * y;
    double a = params.a;

    double expected_hx = -3.0 * params.h0 * a * a * a * x / std::pow(a * a + r2, 2.5);
    double expected_hy = -3.0 * params.h0 * a * a * a * y / std::pow(a * a + r2, 2.5);

    auto result = topography::eval_bell(x, y, params);
    REQUIRE(result.hx == Approx(expected_hx).epsilon(1e-10));
    REQUIRE(result.hy == Approx(expected_hy).epsilon(1e-10));
}

TEST_CASE("Bell axisymmetric: symmetry h(x,0) == h(0,x)", "[physics][terrain][analytical]")
{
    TerrainConfig::BellParams params;
    params.h0 = 500.0;
    params.a = 5000.0;
    params.axisymmetric = true;

    auto h1 = topography::eval_bell(3000.0, 0.0, params);
    auto h2 = topography::eval_bell(0.0, 3000.0, params);
    REQUIRE(h1.h == Approx(h2.h).epsilon(1e-12));
}

TEST_CASE("Bell 2D ridge: peak height equals h0", "[physics][terrain][analytical]")
{
    // h(0) = h0 / (1 + 0) = h0
    TerrainConfig::BellParams params;
    params.h0 = 750.0;
    params.a = 5000.0;
    params.axisymmetric = false;

    auto result = topography::eval_bell(0.0, 0.0, params);
    REQUIRE(result.h == Approx(750.0).epsilon(1e-12));
}

TEST_CASE("Bell 2D ridge: h(a) = h0/2", "[physics][terrain][analytical]")
{
    // h(a) = h0 / (1 + 1) = h0/2
    TerrainConfig::BellParams params;
    params.h0 = 1000.0;
    params.a = 5000.0;
    params.axisymmetric = false;

    auto result = topography::eval_bell(5000.0, 0.0, params);
    REQUIRE(result.h == Approx(500.0).epsilon(1e-12));
}

TEST_CASE("Bell 2D ridge: y-invariant (hy = 0)", "[physics][terrain][analytical]")
{
    TerrainConfig::BellParams params;
    params.h0 = 500.0;
    params.a = 5000.0;
    params.axisymmetric = false;

    auto result = topography::eval_bell(3000.0, 7000.0, params);
    REQUIRE(result.hy == 0.0);

    // Height depends only on x
    auto result2 = topography::eval_bell(3000.0, 0.0, params);
    REQUIRE(result.h == Approx(result2.h).epsilon(1e-12));
}

TEST_CASE("Bell 2D ridge: dh/dx = -2*h0*(x/a) / (a*(1+(x/a)^2)^2)", "[physics][terrain][analytical]")
{
    TerrainConfig::BellParams params;
    params.h0 = 1000.0;
    params.a = 5000.0;
    params.axisymmetric = false;

    double x = 2500.0;
    double xa = x / params.a;
    double denom = 1.0 + xa * xa;
    double expected_hx = -2.0 * params.h0 * xa / (params.a * denom * denom);

    auto result = topography::eval_bell(x, 0.0, params);
    REQUIRE(result.hx == Approx(expected_hx).epsilon(1e-10));
}

TEST_CASE("Bell height is non-negative for positive h0", "[physics][terrain][analytical]")
{
    TerrainConfig::BellParams params;
    params.h0 = 500.0;
    params.a = 5000.0;

    for (bool axi : {true, false})
    {
        params.axisymmetric = axi;
        for (double x = -20000.0; x <= 20000.0; x += 2000.0)
        {
            auto result = topography::eval_bell(x, 0.0, params);
            REQUIRE(result.h >= 0.0);
        }
    }
}

TEST_CASE("Bell terrain uses Cartesian dy in metres on a rectangular grid",
          "[physics][terrain][analytical][rectangular-grid]")
{
    NR = 5;
    NTH = 5;
    NZ = 3;
    dr = 1000.0;
    dtheta = 500.0;

    TerrainConfig cfg;
    cfg.scheme_id = "bell";
    cfg.bell.h0 = 1000.0;
    cfg.bell.a = 5000.0;
    cfg.bell.axisymmetric = true;
    Topography2D topo;
    topography::initialize_topography(topo, NR, NTH);
    auto scheme = create_terrain_scheme("bell");
    scheme->initialize(cfg);
    scheme->build_topography(cfg, topo);

    // These cells are both exactly 1 km from the domain center: one in x,
    // one in y. Axisymmetric terrain must give identical values.
    REQUIRE(topo.h[3][2] == Approx(topo.h[2][4]).epsilon(1.0e-12));
}

TEST_CASE("Smoothed terrain coordinate round-trips and metrics match derivatives",
          "[physics][terrain][analytical][metrics]")
{
    TerrainConfig cfg;
    cfg.coord_id = "smoothed";
    cfg.ztop = 20000.0;
    cfg.smoothing_decay = 0.7;
    Topography2D topo;
    topography::initialize_topography(topo, 1, 1);
    topo.h[0][0] = 1200.0;
    topo.hx[0][0] = 0.08;
    topo.hy[0][0] = -0.03;
    topography::TerrainFollowingCoordinate coordinate(cfg, topo);

    for (double zeta : {0.0, 1000.0, 5000.0, 12000.0, 19999.0})
    {
        CAPTURE(zeta);
        const double z = coordinate.zeta_to_z(zeta, 0, 0);
        REQUIRE(coordinate.z_to_zeta(z, 0, 0)
                == Approx(zeta).margin(1.0e-7));

        double z_metric = 0.0, jacobian = 0.0, mx = 0.0, my = 0.0;
        coordinate.compute_metrics(zeta, 0, 0, z_metric, jacobian, mx, my);
        REQUIRE(z_metric == Approx(z).margin(1.0e-10));
        REQUIRE(jacobian > 0.0);

        const double decay = std::exp(-cfg.smoothing_decay * zeta / cfg.ztop);
        const double sigma = 1.0 - zeta / cfg.ztop;
        REQUIRE(mx == Approx(-sigma * topo.hx[0][0] * decay / jacobian)
                      .epsilon(1.0e-12));
        REQUIRE(my == Approx(-sigma * topo.hy[0][0] * decay / jacobian)
                      .epsilon(1.0e-12));

        if (zeta > 1.0 && zeta < cfg.ztop - 1.0)
        {
            const double eps = 0.1;
            const double numeric_j =
                (coordinate.zeta_to_z(zeta + eps, 0, 0) -
                 coordinate.zeta_to_z(zeta - eps, 0, 0)) / (2.0 * eps);
            REQUIRE(jacobian == Approx(numeric_j).epsilon(1.0e-8));
        }
    }
}

// ---- Conservative finite-volume terrain geometry (TE-001) ----

namespace
{
TerrainMetrics3D make_conservative_test_geometry(bool flat)
{
    constexpr int nr = 6;
    constexpr int nth = 5;
    constexpr int nz = 4;
    constexpr double dx_local = 700.0;
    constexpr double dy_local = 900.0;
    constexpr double ztop = 6000.0;

    TerrainConfig cfg;
    cfg.coord_id = flat ? "cartesian" : "smoothed";
    cfg.ztop = ztop;
    cfg.smoothing_decay = 0.8;
    Topography2D topo;
    topography::initialize_topography(topo, nr, nth);
    for (int i = 0; i < nr; ++i)
    {
        for (int j = 0; j < nth; ++j)
        {
            if (!flat)
            {
                // A non-separable 3-D surface exercises both mixed metric
                // identities; analytic hx/hy are intentionally irrelevant to
                // the finite-volume construction.
                topo.h[i][j] = 250.0
                    + 55.0 * std::sin(0.7 * static_cast<double>(i))
                    * std::cos(0.5 * static_cast<double>(j))
                    + 11.0 * static_cast<double>(i * j);
            }
        }
    }

    TerrainMetrics3D metrics;
    topography::initialize_metrics(metrics, nr, nth, nz);
    topography::build_conservative_face_metrics(
        cfg, topo, topography::build_zeta_interfaces(nz, ztop),
        dx_local, dy_local, metrics);
    return metrics;
}

TerrainPhysicalFluxes make_physical_fluxes(const TerrainMetrics3D& metrics)
{
    const int nr = metrics.J.size_r();
    const int nth = metrics.J.size_th();
    const int nz = metrics.J.size_z();
    TerrainPhysicalFluxes fluxes;
    fluxes.x.resize(nr + 1, nth, nz, 0.0);
    fluxes.y.resize(nr, nth + 1, nz, 0.0);
    fluxes.z_x.resize(nr, nth, nz + 1, 0.0);
    fluxes.z_y.resize(nr, nth, nz + 1, 0.0);
    fluxes.z_vertical.resize(nr, nth, nz + 1, 0.0);
    return fluxes;
}

TerrainMetrics3D make_named_terrain_geometry(const std::string& scheme_id)
{
    NR = 9;
    NTH = 7;
    NZ = 6;
    dr = 800.0;
    dtheta = 900.0;
    dz = 1000.0;

    TerrainConfig cfg;
    cfg.scheme_id = scheme_id;
    cfg.coord_id = "smoothed";
    cfg.ztop = static_cast<double>(NZ) * dz;
    cfg.smoothing_decay = 0.65;
    cfg.bell.h0 = 450.0;
    cfg.bell.a = 1800.0;
    cfg.bell.axisymmetric = true;
    cfg.schar.H = 350.0;
    cfg.schar.a = 2300.0;
    cfg.schar.ell = 1700.0;

    Topography2D topo;
    TerrainMetrics3D metrics;
    topography::initialize_topography(topo, NR, NTH);
    topography::initialize_metrics(metrics, NR, NTH, NZ);
    auto scheme = create_terrain_scheme(scheme_id);
    scheme->initialize(cfg);
    scheme->build_topography(cfg, topo);
    scheme->build_metrics(cfg, topo, metrics);
    return metrics;
}
} // namespace

TEST_CASE("terrain finite-volume metrics preserve a uniform physical flux",
          "[physics][terrain][conservative][metric-identity]")
{
    const TerrainMetrics3D metrics = make_conservative_test_geometry(false);
    TerrainPhysicalFluxes physical = make_physical_fluxes(metrics);
    std::fill(physical.x.values.begin(), physical.x.values.end(), 7.25);
    std::fill(physical.y.values.begin(), physical.y.values.end(), -3.5);
    std::fill(physical.z_x.values.begin(), physical.z_x.values.end(), 7.25);
    std::fill(physical.z_y.values.begin(), physical.z_y.values.end(), -3.5);
    std::fill(physical.z_vertical.values.begin(),
              physical.z_vertical.values.end(), 1.75);

    const auto transformed = terrain_transport::transform_fluxes(metrics, physical);
    const auto divergence =
        terrain_transport::conservative_divergence(metrics, transformed);
    for (double value : divergence.values)
    {
        CHECK(value == Approx(0.0).margin(2.0e-15));
    }
}

TEST_CASE("terrain conservative transport reduces exactly to Cartesian fluxes",
          "[physics][terrain][conservative][flat-reduction]")
{
    const TerrainMetrics3D metrics = make_conservative_test_geometry(true);
    TerrainPhysicalFluxes physical = make_physical_fluxes(metrics);
    for (std::size_t n = 0; n < physical.x.size(); ++n)
        physical.x.values[n] = 0.125 * static_cast<double>(n) - 2.0;
    for (std::size_t n = 0; n < physical.y.size(); ++n)
        physical.y.values[n] = -0.25 * static_cast<double>(n) + 1.0;
    for (std::size_t n = 0; n < physical.z_vertical.size(); ++n)
    {
        physical.z_vertical.values[n] = 0.0625 * static_cast<double>(n) - 0.5;
        physical.z_x.values[n] = 300.0 + static_cast<double>(n);
        physical.z_y.values[n] = -200.0 - static_cast<double>(n);
    }

    const auto transformed = terrain_transport::transform_fluxes(metrics, physical);
    REQUIRE(transformed.x.values == physical.x.values);
    REQUIRE(transformed.y.values == physical.y.values);
    REQUIRE(transformed.z.values == physical.z_vertical.values);

    const auto divergence =
        terrain_transport::conservative_divergence(metrics, transformed);
    const int nr = metrics.J.size_r();
    const int nth = metrics.J.size_th();
    const int nz = metrics.J.size_z();
    for (int i = 0; i < nr; ++i)
        for (int j = 0; j < nth; ++j)
            for (int k = 0; k < nz; ++k)
            {
                const double cartesian =
                    (physical.x(i + 1, j, k) - physical.x(i, j, k)) / metrics.dx
                    + (physical.y(i, j + 1, k) - physical.y(i, j, k)) / metrics.dy
                    + (physical.z_vertical(i, j, k + 1)
                       - physical.z_vertical(i, j, k)) / metrics.dzeta;
                CHECK(divergence(i, j, k) == cartesian);
            }
}

TEST_CASE("terrain Jacobian-weighted divergence telescopes to boundary flux",
          "[physics][terrain][conservative][budget]")
{
    const TerrainMetrics3D metrics = make_conservative_test_geometry(false);
    TerrainPhysicalFluxes physical = make_physical_fluxes(metrics);
    for (std::size_t n = 0; n < physical.x.size(); ++n)
        physical.x.values[n] = std::sin(0.13 * static_cast<double>(n));
    for (std::size_t n = 0; n < physical.y.size(); ++n)
        physical.y.values[n] = std::cos(0.17 * static_cast<double>(n));
    for (std::size_t n = 0; n < physical.z_vertical.size(); ++n)
    {
        physical.z_x.values[n] = 0.4 * std::sin(0.11 * static_cast<double>(n));
        physical.z_y.values[n] = -0.3 * std::cos(0.07 * static_cast<double>(n));
        physical.z_vertical.values[n] = 0.2 * std::sin(0.19 * static_cast<double>(n));
    }
    const auto flux = terrain_transport::transform_fluxes(metrics, physical);
    const auto divergence = terrain_transport::conservative_divergence(metrics, flux);
    const int nr = metrics.J.size_r();
    const int nth = metrics.J.size_th();
    const int nz = metrics.J.size_z();

    double volume_integral = 0.0;
    for (int i = 0; i < nr; ++i)
        for (int j = 0; j < nth; ++j)
            for (int k = 0; k < nz; ++k)
                volume_integral += metrics.J(i, j, k) * divergence(i, j, k)
                    * metrics.dx * metrics.dy * metrics.dzeta;

    double boundary_flux = 0.0;
    for (int j = 0; j < nth; ++j)
        for (int k = 0; k < nz; ++k)
            boundary_flux += (flux.x(nr, j, k) - flux.x(0, j, k))
                * metrics.dy * metrics.dzeta;
    for (int i = 0; i < nr; ++i)
        for (int k = 0; k < nz; ++k)
            boundary_flux += (flux.y(i, nth, k) - flux.y(i, 0, k))
                * metrics.dx * metrics.dzeta;
    for (int i = 0; i < nr; ++i)
        for (int j = 0; j < nth; ++j)
            boundary_flux += (flux.z(i, j, nz) - flux.z(i, j, 0))
                * metrics.dx * metrics.dy;

    REQUIRE(volume_integral == Approx(boundary_flux).epsilon(2.0e-13));
}

TEST_CASE("constant scalar shares the terrain mass-flux tendency exactly",
          "[physics][terrain][conservative][constant-state]")
{
    constexpr double q0 = 0.0125;
    const TerrainMetrics3D metrics = make_conservative_test_geometry(false);
    TerrainPhysicalFluxes mass = make_physical_fluxes(metrics);
    for (std::size_t n = 0; n < mass.x.size(); ++n)
        mass.x.values[n] = 1.0 + 0.01 * static_cast<double>(n);
    for (std::size_t n = 0; n < mass.y.size(); ++n)
        mass.y.values[n] = -0.5 + 0.02 * static_cast<double>(n);
    for (std::size_t n = 0; n < mass.z_vertical.size(); ++n)
    {
        mass.z_x.values[n] = 0.3 + 0.005 * static_cast<double>(n);
        mass.z_y.values[n] = -0.2 + 0.003 * static_cast<double>(n);
        mass.z_vertical.values[n] = 0.1 - 0.002 * static_cast<double>(n);
    }
    TerrainPhysicalFluxes scalar = mass;
    for (TerrainField3D* field : {&scalar.x, &scalar.y, &scalar.z_x,
                                  &scalar.z_y, &scalar.z_vertical})
        for (double& value : field->values) value *= q0;

    const auto mass_divergence = terrain_transport::conservative_divergence(
        metrics, terrain_transport::transform_fluxes(metrics, mass));
    const auto scalar_divergence = terrain_transport::conservative_divergence(
        metrics, terrain_transport::transform_fluxes(metrics, scalar));
    REQUIRE(scalar_divergence.size() == mass_divergence.size());
    for (std::size_t n = 0; n < mass_divergence.size(); ++n)
    {
        CHECK(scalar_divergence.values[n]
              == Approx(q0 * mass_divergence.values[n]).margin(2.0e-19));
    }
}

TEST_CASE("terrain conservative transport rejects corrupted geometry",
          "[physics][terrain][conservative][fail-closed]")
{
    TerrainMetrics3D metrics = make_conservative_test_geometry(false);
    metrics.Jx_face(1, 1, 1) = std::numeric_limits<double>::quiet_NaN();
    REQUIRE_THROWS_AS(terrain_transport::validate_geometry(metrics),
                      std::invalid_argument);
}

TEST_CASE("terrain physical scalar gradient removes coordinate-slope terms",
          "[physics][terrain][pressure-gradient][metric]")
{
    const TerrainMetrics3D metrics = make_conservative_test_geometry(false);
    const int nr = metrics.J.size_r();
    const int nth = metrics.J.size_th();
    const int nz = metrics.J.size_z();
    Field3D scalar(nr, nth, nz, 0.0f);
    constexpr double ax = 0.125;
    constexpr double ay = -0.0625;
    constexpr double az = 0.03125;
    for (int i = 0; i < nr; ++i)
        for (int j = 0; j < nth; ++j)
            for (int k = 0; k < nz; ++k)
                scalar[i][j][k] = static_cast<float>(
                    ax * static_cast<double>(i) * metrics.dx
                    + ay * static_cast<double>(j) * metrics.dy
                    + az * metrics.z(i, j, k));

    for (int i = 1; i < nr - 1; ++i)
        for (int j = 1; j < nth - 1; ++j)
            for (int k = 0; k < nz; ++k)
            {
                CHECK(terrain_transport::cgrid_scalar_gradient_x(
                          metrics, scalar, i, j, k)
                      == Approx(ax).margin(2.0e-7));
                CHECK(terrain_transport::cgrid_scalar_gradient_y(
                          metrics, scalar, i, j, k)
                      == Approx(ay).margin(2.0e-7));
                if (k + 1 < nz)
                    CHECK(terrain_transport::cgrid_scalar_gradient_z(
                              metrics, scalar, i, j, k)
                          == Approx(az).margin(2.0e-7));
            }
}

TEST_CASE("bell and Schar reference pressure has zero terrain acceleration",
          "[physics][terrain][pressure-gradient][well-balanced]")
{
    for (const std::string& scheme_id : {std::string("bell"),
                                         std::string("schar")})
    {
        CAPTURE(scheme_id);
        const TerrainMetrics3D metrics = make_named_terrain_geometry(scheme_id);
        const int nr = metrics.J.size_r();
        const int nth = metrics.J.size_th();
        const int nz = metrics.J.size_z();
        std::vector<double> reference_pressure(static_cast<std::size_t>(nz));
        for (int k = 0; k < nz; ++k)
        {
            const double height = (static_cast<double>(k) + 0.5)
                                * metrics.dzeta;
            reference_pressure[static_cast<std::size_t>(k)] =
                100000.0 * std::exp(-height / 8200.0);
        }

        Field3D pressure(nr, nth, nz, 0.0f);
        for (int i = 0; i < nr; ++i)
            for (int j = 0; j < nth; ++j)
                for (int k = 0; k < nz; ++k)
                    pressure[i][j][k] = static_cast<float>(
                        terrain_transport::reference_profile_value_fp32(
                            reference_pressure, metrics.z(i, j, k),
                            metrics.dzeta));

        for (int i = 1; i < nr - 1; ++i)
            for (int j = 1; j < nth - 1; ++j)
                for (int k = 0; k < nz; ++k)
                {
                    CHECK(terrain_transport::cgrid_pressure_perturbation_gradient_x(
                              metrics, pressure, reference_pressure, i, j, k)
                          == 0.0);
                    CHECK(terrain_transport::cgrid_pressure_perturbation_gradient_y(
                              metrics, pressure, reference_pressure, i, j, k)
                          == 0.0);
                    if (k + 1 < nz)
                        CHECK(terrain_transport::cgrid_pressure_perturbation_gradient_z(
                                  metrics, pressure, reference_pressure, i, j, k)
                              == 0.0);
                }
    }
}

TEST_CASE("terrain contravariant velocity vanishes along a zeta surface",
          "[physics][terrain][momentum][contravariant]")
{
    const TerrainMetrics3D metrics = make_conservative_test_geometry(false);
    constexpr int i = 2;
    constexpr int j = 2;
    constexpr int k = 1;
    constexpr double physical_u = 11.0;
    constexpr double physical_v = -4.0;

    const double slope_x_u = (metrics.z(i + 1, j, k) - metrics.z(i, j, k))
                           / metrics.dx;
    const double slope_y_u = (
        0.5 * (metrics.z(i, j + 1, k) + metrics.z(i + 1, j + 1, k))
        - 0.5 * (metrics.z(i, j - 1, k) + metrics.z(i + 1, j - 1, k)))
        / (2.0 * metrics.dy);
    CHECK(terrain_transport::cgrid_zeta_velocity_at_x_face(
              metrics, physical_u, physical_v,
              physical_u * slope_x_u + physical_v * slope_y_u,
              i, j, k)
          == Approx(0.0).margin(2.0e-15));

    const double slope_x_v = (
        0.5 * (metrics.z(i + 1, j, k) + metrics.z(i + 1, j + 1, k))
        - 0.5 * (metrics.z(i - 1, j, k) + metrics.z(i - 1, j + 1, k)))
        / (2.0 * metrics.dx);
    const double slope_y_v = (metrics.z(i, j + 1, k) - metrics.z(i, j, k))
                           / metrics.dy;
    CHECK(terrain_transport::cgrid_zeta_velocity_at_y_face(
              metrics, physical_u, physical_v,
              physical_u * slope_x_v + physical_v * slope_y_v,
              i, j, k)
          == Approx(0.0).margin(2.0e-15));

    const double slope_x_w = -metrics.Gx_face(i, j, k + 1);
    const double slope_y_w = -metrics.Gy_face(i, j, k + 1);
    CHECK(terrain_transport::cgrid_zeta_velocity_at_z_face(
              metrics, physical_u, physical_v,
              physical_u * slope_x_w + physical_v * slope_y_w,
              i, j, k)
          == Approx(0.0).margin(2.0e-15));
}

TEST_CASE("implicit terrain ground converts zero normal flow to physical w",
          "[physics][terrain][boundary][impermeable]")
{
    const TerrainMetrics3D metrics = make_conservative_test_geometry(false);
    const int nr = metrics.J.size_r();
    const int nth = metrics.J.size_th();
    const int nz = metrics.J.size_z();
    Field3D u_face(nr, nth, nz, 9.0f);
    Field3D v_face(nr, nth, nz, -5.0f);
    constexpr int i = 2;
    constexpr int j = 2;
    const double slope_x = -metrics.Gx_face(i, j, 0);
    const double slope_y = -metrics.Gy_face(i, j, 0);
    const double physical_w =
        terrain_transport::cgrid_ground_vertical_velocity(
            metrics, u_face, v_face, i, j);
    CHECK(physical_w == Approx(9.0 * slope_x - 5.0 * slope_y));
    const double     residual      = physical_w - 9.0 * slope_x - (-5.0) * slope_y;
    constexpr double unit_roundoff = std::numeric_limits<double>::epsilon() / 2.0;
    const auto upward = [](double value) { return std::nextafter(value, std::numeric_limits<double>::infinity()); };
    // w has error <= gamma_2*S. Its residual adds at most
    // (1+u)*(gamma_2*(1+u)+2u+u*u)*S <= gamma_4*S for these normal
    // binary64 operands, including either permitted FMA contraction.
    const double gamma_four     = upward(4.0 * unit_roundoff / (1.0 - 4.0 * unit_roundoff));
    const double product_scale  = upward(upward(std::abs(9.0 * slope_x)) + upward(std::abs(-5.0 * slope_y)));
    const double roundoff_bound = upward(gamma_four * product_scale);
    INFO(std::scientific << std::setprecision(17) << "normal-flow residual=" << residual
                         << ", bound=" << roundoff_bound);
    CHECK(std::abs(residual) <= roundoff_bound);

    const TerrainMetrics3D flat = make_conservative_test_geometry(true);
    CHECK(terrain_transport::cgrid_ground_vertical_velocity(flat, u_face, v_face, i, j) == 0.0);
}

TEST_CASE("terrain ground velocity preserves exact slope signs and C-grid face averages",
          "[physics][terrain][boundary][impermeable][staggering]")
{
    TerrainMetrics3D metrics = make_conservative_test_geometry(false);
    const int        nr      = metrics.J.size_r();
    const int        nth     = metrics.J.size_th();
    const int        nz      = metrics.J.size_z();
    Field3D          u_face(nr, nth, nz, 32.0f);
    Field3D          v_face(nr, nth, nz, 64.0f);
    constexpr int    i  = 2;
    constexpr int    j  = 2;
    u_face[i - 1][j][0] = 2.0f;
    u_face[i][j][0]     = 6.0f;
    v_face[i][j - 1][0] = -2.0f;
    v_face[i][j][0]     = -6.0f;
    for (const auto& sample : std::array<std::array<double, 3>, 4>{
             {{{0.25, 0.125, 0.5}}, {{0.25, -0.125, 1.5}}, {{-0.25, 0.125, -1.5}}, {{-0.25, -0.125, -0.5}}}})
    {
        const double slope_x = sample[0];
        const double slope_y = sample[1];
        CAPTURE(slope_x, slope_y);
        metrics.Gx_face(i, j, 0) = -slope_x;
        metrics.Gy_face(i, j, 0) = -slope_y;
        const double physical_w  = terrain_transport::cgrid_ground_vertical_velocity(metrics, u_face, v_face, i, j);
        CHECK(physical_w == sample[2]);
        CHECK(physical_w - 4.0 * slope_x - (-4.0) * slope_y == 0.0);
    }
    const TerrainMetrics3D flat = make_conservative_test_geometry(true);
    CHECK(terrain_transport::cgrid_ground_vertical_velocity(flat, u_face, v_face, i, j) == 0.0);
}

TEST_CASE("terrain ground surface frame closes normal flow and traction geometry",
          "[physics][terrain][boundary][stress][geometry]")
{
    const TerrainMetrics3D metrics = make_conservative_test_geometry(false);
    constexpr int i = 2;
    constexpr int j = 2;
    const auto frame = terrain_transport::ground_surface_frame(metrics, i, j);

    CHECK(frame.slope_x == Approx(-metrics.Gx_face(i, j, 0)));
    CHECK(frame.slope_y == Approx(-metrics.Gy_face(i, j, 0)));
    CHECK(std::hypot(std::hypot(frame.normal_x, frame.normal_y),
                     frame.normal_z)
          == Approx(1.0).epsilon(2.0e-15));
    CHECK(frame.area_ratio
          == Approx(std::hypot(std::hypot(frame.slope_x, frame.slope_y), 1.0)));

    const auto velocity = terrain_transport::ground_tangent_velocity(
        frame, 8.0, -3.0);
    CHECK(velocity.x * frame.normal_x + velocity.y * frame.normal_y
              + velocity.z * frame.normal_z
          == Approx(0.0).margin(2.0e-15));
    CHECK(terrain_transport::mapped_ground_normal_flux(frame, velocity)
          == Approx(0.0).margin(2.0e-15));

    const terrain_transport::PhysicalVector3 arbitrary{4.0, -2.0, 5.0};
    const auto tangent = terrain_transport::project_to_ground_tangent(
        frame, arbitrary);
    CHECK(tangent.x * frame.normal_x + tangent.y * frame.normal_y
              + tangent.z * frame.normal_z
          == Approx(0.0).margin(2.0e-15));

    const double mapped_flux =
        terrain_transport::mapped_ground_normal_flux(frame, arbitrary);
    const double unit_normal_flux = arbitrary.x * frame.normal_x
                                  + arbitrary.y * frame.normal_y
                                  + arbitrary.z * frame.normal_z;
    CHECK(mapped_flux == Approx(frame.area_ratio * unit_normal_flux));

    const auto traction = terrain_transport::area_weighted_ground_traction(
        frame, arbitrary);
    CHECK(traction.x == Approx(frame.area_ratio * arbitrary.x));
    CHECK(traction.y == Approx(frame.area_ratio * arbitrary.y));
    CHECK(traction.z == Approx(frame.area_ratio * arbitrary.z));

    const terrain_transport::SymmetricStressTensor stress{
        3.0, 5.0, 7.0, -2.0, 4.0, 6.0};
    const auto mapped_traction =
        terrain_transport::mapped_ground_stress_traction(frame, stress);
    const terrain_transport::PhysicalVector3 unit_traction{
        stress.xx * frame.normal_x + stress.xy * frame.normal_y
            + stress.xz * frame.normal_z,
        stress.xy * frame.normal_x + stress.yy * frame.normal_y
            + stress.yz * frame.normal_z,
        stress.xz * frame.normal_x + stress.yz * frame.normal_y
            + stress.zz * frame.normal_z};
    CHECK(mapped_traction.x
          == Approx(frame.area_ratio * unit_traction.x).epsilon(2.0e-15));
    CHECK(mapped_traction.y
          == Approx(frame.area_ratio * unit_traction.y).epsilon(2.0e-15));
    CHECK(mapped_traction.z
          == Approx(frame.area_ratio * unit_traction.z).epsilon(2.0e-15));
}

TEST_CASE("terrain stress tensor uses one contravariant transform on every face",
          "[physics][terrain][stress][tensor][mapped][sci099]")
{
    const terrain_transport::SymmetricStressTensor stress{
        3.0, 5.0, 7.0, -2.0, 4.0, 6.0};
    constexpr double jx = 1.3;
    constexpr double jy = 0.8;
    constexpr double gx = -0.12;
    constexpr double gy = 0.07;

    const auto x = terrain_transport::contravariant_stress_flux_x(jx, stress);
    CHECK(x.x == Approx(jx * stress.xx));
    CHECK(x.y == Approx(jx * stress.xy));
    CHECK(x.z == Approx(jx * stress.xz));

    const auto y = terrain_transport::contravariant_stress_flux_y(jy, stress);
    CHECK(y.x == Approx(jy * stress.xy));
    CHECK(y.y == Approx(jy * stress.yy));
    CHECK(y.z == Approx(jy * stress.yz));

    const auto zeta = terrain_transport::contravariant_stress_flux_zeta(
        gx, gy, stress);
    CHECK(zeta.x == Approx(stress.xz + gx * stress.xx + gy * stress.xy));
    CHECK(zeta.y == Approx(stress.yz + gx * stress.xy + gy * stress.yy));
    CHECK(zeta.z == Approx(stress.zz + gx * stress.xz + gy * stress.yz));

    const auto flat_zeta = terrain_transport::contravariant_stress_flux_zeta(
        0.0, 0.0, stress);
    CHECK(flat_zeta.x == stress.xz);
    CHECK(flat_zeta.y == stress.yz);
    CHECK(flat_zeta.z == stress.zz);

    const TerrainMetrics3D metrics = make_conservative_test_geometry(false);
    const auto frame = terrain_transport::ground_surface_frame(metrics, 2, 2);
    const auto ground = terrain_transport::mapped_ground_stress_traction(
        frame, stress);
    const auto interior = terrain_transport::contravariant_stress_flux_zeta(
        metrics.Gx_face(2, 2, 0), metrics.Gy_face(2, 2, 0), stress);
    CHECK(ground.x == interior.x);
    CHECK(ground.y == interior.y);
    CHECK(ground.z == interior.z);
}

TEST_CASE("terrain stress transform rejects invalid metrics before arithmetic",
          "[physics][terrain][stress][tensor][validation][sci099]")
{
    terrain_transport::SymmetricStressTensor stress;
    CHECK_THROWS_AS(terrain_transport::contravariant_stress_flux_x(
                        0.0, stress), std::invalid_argument);
    CHECK_THROWS_AS(terrain_transport::contravariant_stress_flux_y(
                        std::numeric_limits<double>::infinity(), stress),
                    std::invalid_argument);
    CHECK_THROWS_AS(terrain_transport::contravariant_stress_flux_zeta(
                        std::numeric_limits<double>::quiet_NaN(), 0.0, stress),
                    std::invalid_argument);
    stress.xy = std::numeric_limits<double>::quiet_NaN();
    CHECK_THROWS_AS(terrain_transport::contravariant_stress_flux_zeta(
                        0.0, 0.0, stress), std::invalid_argument);
}

TEST_CASE("terrain derivative transform recovers a Cartesian linear field",
          "[physics][terrain][gradient][mapped][sci100]")
{
    // z = h(x,y) + J*zeta, with G=(-dh/dx,-dh/dy). For
    // phi=a*x+b*y+c*z, computational differentiation gives
    // phi_x|zeta=a-c*Gx, phi_y|zeta=b-c*Gy, phi_zeta=c*J.
    constexpr double a = 2.5;
    constexpr double b = -1.75;
    constexpr double c = 0.04;
    constexpr double jacobian = 1.3;
    constexpr double gx = -0.12;
    constexpr double gy = 0.07;
    const auto gradient =
        terrain_transport::physical_gradient_from_computational(
            a - c * gx, b - c * gy, c * jacobian,
            jacobian, gx, gy);
    CHECK(gradient.x == Approx(a).epsilon(2.0e-15));
    CHECK(gradient.y == Approx(b).epsilon(2.0e-15));
    CHECK(gradient.z == Approx(c).epsilon(2.0e-15));

    const auto flat = terrain_transport::physical_gradient_from_computational(
        a, b, c * jacobian, jacobian, 0.0, 0.0);
    CHECK(flat.x == a);
    CHECK(flat.y == b);
    CHECK(flat.z == Approx(c));
}

TEST_CASE("terrain derivative transform rejects invalid metric state",
          "[physics][terrain][gradient][validation][sci100]")
{
    CHECK_THROWS_AS(terrain_transport::physical_gradient_from_computational(
                        0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                    std::invalid_argument);
    CHECK_THROWS_AS(terrain_transport::physical_gradient_from_computational(
                        0.0, 0.0, 0.0, 1.0,
                        std::numeric_limits<double>::quiet_NaN(), 0.0),
                    std::invalid_argument);
    CHECK_THROWS_AS(terrain_transport::physical_gradient_from_computational(
                        0.0, std::numeric_limits<double>::infinity(), 0.0,
                        1.0, 0.0, 0.0),
                    std::invalid_argument);
}

TEST_CASE("terrain ground surface frame reduces exactly on flat geometry",
          "[physics][terrain][boundary][stress][flat-reduction]")
{
    const TerrainMetrics3D metrics = make_conservative_test_geometry(true);
    const auto frame = terrain_transport::ground_surface_frame(metrics, 2, 2);
    CHECK(frame.slope_x == 0.0);
    CHECK(frame.slope_y == 0.0);
    CHECK(frame.normal_x == 0.0);
    CHECK(frame.normal_y == 0.0);
    CHECK(frame.normal_z == 1.0);
    CHECK(frame.area_ratio == 1.0);

    const terrain_transport::PhysicalVector3 flux{2.0, -7.0, 3.5};
    CHECK(terrain_transport::mapped_ground_normal_flux(frame, flux) == flux.z);
    const auto velocity = terrain_transport::ground_tangent_velocity(
        frame, 2.0, -7.0);
    CHECK(velocity.x == 2.0);
    CHECK(velocity.y == -7.0);
    CHECK(velocity.z == 0.0);

    const terrain_transport::SymmetricStressTensor stress{
        3.0, 5.0, 7.0, -2.0, 4.0, 6.0};
    const auto traction = terrain_transport::mapped_ground_stress_traction(
        frame, stress);
    CHECK(traction.x == stress.xz);
    CHECK(traction.y == stress.yz);
    CHECK(traction.z == stress.zz);
}

TEST_CASE("terrain ground surface geometry fails closed on invalid input",
          "[physics][terrain][boundary][stress][validation]")
{
    TerrainMetrics3D metrics = make_conservative_test_geometry(false);
    CHECK_NOTHROW(terrain_transport::ground_surface_frame(metrics, 0, 2));
    CHECK_NOTHROW(terrain_transport::ground_surface_frame(
        metrics, metrics.J.size_r() - 1, metrics.J.size_th() - 1));
    CHECK_THROWS_AS(terrain_transport::ground_surface_frame(metrics, -1, 2),
                    std::out_of_range);
    metrics.Gx_face(2, 2, 0) = std::numeric_limits<double>::quiet_NaN();
    CHECK_THROWS_AS(terrain_transport::ground_surface_frame(metrics, 2, 2),
                    std::invalid_argument);

    const int nr = metrics.J.size_r();
    const int nth = metrics.J.size_th();
    const int nz = metrics.J.size_z();
    Field3D u_face(nr, nth, nz, 1.0f);
    Field3D v_face(nr, nth, nz, 1.0f);
    CHECK_THROWS_AS(terrain_transport::cgrid_ground_vertical_velocity(
                        metrics, u_face, v_face, 0, 2),
                    std::out_of_range);

    terrain_transport::GroundSurfaceFrame inconsistent;
    inconsistent.slope_x = 1.0;
    CHECK_THROWS_AS(terrain_transport::ground_tangent_velocity(
                        inconsistent, 1.0, 1.0),
                    std::invalid_argument);
}

TEST_CASE("terrain C-grid face flux obeys the interior metric identity",
          "[physics][terrain][conservative][cgrid][metric-identity]")
{
    const TerrainMetrics3D metrics = make_conservative_test_geometry(false);
    const int nr = metrics.J.size_r();
    const int nth = metrics.J.size_th();
    const int nz = metrics.J.size_z();
    Field3D density(nr, nth, nz, 1.3f);
    Field3D u_face(nr, nth, nz, 6.0f);
    Field3D v_face(nr, nth, nz, -2.0f);
    Field3D w_face(nr, nth, nz, 0.75f);
    terrain_transport::validate_cgrid_state(
        metrics, density, u_face, v_face, w_face);

    // k=0 has an implicit impermeable terrain face and therefore is not a
    // free-stream cell for an arbitrary uniform physical velocity.  Every
    // cell above that boundary must satisfy the metric identity.
    for (int i = 1; i < nr - 1; ++i)
        for (int j = 1; j < nth - 1; ++j)
            for (int k = 1; k < nz; ++k)
            {
                CHECK(terrain_transport::cgrid_velocity_divergence(
                          metrics, u_face, v_face, w_face, i, j, k)
                      == Approx(0.0).margin(2.0e-15));
                CHECK(terrain_transport::cgrid_mass_divergence(
                          metrics, density, u_face, v_face, w_face, i, j, k)
                      == Approx(0.0).margin(4.0e-15));
            }
}

TEST_CASE("terrain C-grid transport is Galilean-consistent in a translating grid",
          "[physics][terrain][conservative][cgrid][domain-motion]")
{
    const TerrainMetrics3D metrics = make_conservative_test_geometry(false);
    const int nr = metrics.J.size_r();
    const int nth = metrics.J.size_th();
    const int nz = metrics.J.size_z();
    Field3D density(nr, nth, nz, 1.25f);
    Field3D u_relative(nr, nth, nz, 4.0f);
    Field3D v_relative(nr, nth, nz, -1.0f);
    Field3D u_physical(nr, nth, nz, 6.0f);
    Field3D v_physical(nr, nth, nz, -2.0f);
    Field3D w_face(nr, nth, nz, 0.75f);
    for (int i = 0; i < nr; ++i)
        for (int j = 0; j < nth; ++j)
            w_face[i][j][nz - 1] = 0.0f;
    constexpr double frame_u = 2.0;
    constexpr double frame_v = -1.0;

    for (int i = 1; i < nr - 1; ++i)
        for (int j = 1; j < nth - 1; ++j)
        {
            CHECK(terrain_transport::cgrid_ground_vertical_velocity(
                      metrics, u_physical, v_physical, i, j,
                      frame_u, frame_v)
                  == terrain_transport::cgrid_ground_vertical_velocity(
                      metrics, u_relative, v_relative, i, j));
            for (int k = 0; k < nz; ++k)
            {
                CHECK(terrain_transport::cgrid_mass_flux_x(
                          metrics, density, u_physical, i, j, k, frame_u)
                      == terrain_transport::cgrid_mass_flux_x(
                          metrics, density, u_relative, i, j, k));
                CHECK(terrain_transport::cgrid_mass_flux_y(
                          metrics, density, v_physical, i, j, k, frame_v)
                      == terrain_transport::cgrid_mass_flux_y(
                          metrics, density, v_relative, i, j, k));
                CHECK(terrain_transport::cgrid_mass_flux_zeta(
                          metrics, density, u_physical, v_physical, w_face,
                          i, j, k, frame_u, frame_v)
                      == terrain_transport::cgrid_mass_flux_zeta(
                          metrics, density, u_relative, v_relative, w_face,
                          i, j, k));
                CHECK(terrain_transport::cgrid_mass_divergence(
                          metrics, density, u_physical, v_physical, w_face,
                          i, j, k, frame_u, frame_v)
                      == terrain_transport::cgrid_mass_divergence(
                          metrics, density, u_relative, v_relative, w_face,
                          i, j, k));
                CHECK(terrain_transport::cgrid_velocity_divergence(
                          metrics, u_physical, v_physical, w_face,
                          i, j, k, frame_u, frame_v)
                      == terrain_transport::cgrid_velocity_divergence(
                          metrics, u_relative, v_relative, w_face,
                          i, j, k));
            }
        }

    Field3D frame_u_field(nr, nth, nz, static_cast<float>(frame_u));
    Field3D frame_v_field(nr, nth, nz, static_cast<float>(frame_v));
    Field3D zero_w(nr, nth, nz, 0.0f);
    for (int i = 1; i < nr - 1; ++i)
        for (int j = 1; j < nth - 1; ++j)
        {
            CHECK(terrain_transport::cgrid_ground_vertical_velocity(
                      metrics, frame_u_field, frame_v_field, i, j,
                      frame_u, frame_v) == 0.0);
            for (int k = 0; k < nz; ++k)
            {
                CHECK(terrain_transport::cgrid_mass_divergence(
                          metrics, density, frame_u_field, frame_v_field,
                          zero_w, i, j, k, frame_u, frame_v) == 0.0);
                CHECK(terrain_transport::cgrid_velocity_divergence(
                          metrics, frame_u_field, frame_v_field, zero_w,
                          i, j, k, frame_u, frame_v) == 0.0);
            }
        }

    NR = nr;
    NTH = nth;
    NZ = nz;
    global_stagger_type = StaggerType::CGrid;
    const auto old_west = global_lateral_bc_west;
    const auto old_east = global_lateral_bc_east;
    const auto old_south = global_lateral_bc_south;
    const auto old_north = global_lateral_bc_north;
    global_lateral_bc_west = tmv::init::config::LateralBC::Periodic;
    global_lateral_bc_east = tmv::init::config::LateralBC::Periodic;
    global_lateral_bc_south = tmv::init::config::LateralBC::Periodic;
    global_lateral_bc_north = tmv::init::config::LateralBC::Periodic;
    constexpr double dt_step = 0.25;
    CartesianCGridTransportContext stationary;
    stationary.begin(nr, nth, nz, dt_step, &metrics);
    stationary.accumulate(
        density, u_relative, v_relative, w_face, dt_step);
    stationary.complete();
    CartesianCGridTransportContext translated;
    translated.begin(
        nr, nth, nz, dt_step, &metrics, frame_u, frame_v);
    translated.accumulate(
        density, u_physical, v_physical, w_face, dt_step);
    translated.complete();
    REQUIRE(stationary.validate(nr, nth, nz, dt_step));
    REQUIRE(translated.validate(nr, nth, nz, dt_step));
    for (std::size_t n = 0; n < density.size(); ++n)
    {
        CHECK(translated.mass_flux_x.data()[n]
              == stationary.mass_flux_x.data()[n]);
        CHECK(translated.mass_flux_y.data()[n]
              == stationary.mass_flux_y.data()[n]);
        CHECK(translated.mass_flux_z.data()[n]
              == stationary.mass_flux_z.data()[n]);
    }
    global_lateral_bc_west = old_west;
    global_lateral_bc_east = old_east;
    global_lateral_bc_south = old_south;
    global_lateral_bc_north = old_north;
}

TEST_CASE("retained terrain continuity flux preserves a constant C-grid scalar",
          "[physics][terrain][conservative][cgrid][coupled][SCI-148]")
{
    TerrainMetrics3D metrics = make_conservative_test_geometry(false);
    NR = metrics.J.size_r();
    NTH = metrics.J.size_th();
    NZ = metrics.J.size_z();
    dr = metrics.dx;
    dtheta = metrics.dy;
    dz = metrics.dzeta;
    global_stagger_type = StaggerType::CGrid;

    rho.resize(NR, NTH, NZ, 1.1f);
    u.resize(NR, NTH, NZ, 0.0f);
    v.resize(NR, NTH, NZ, 0.0f);
    w.resize(NR, NTH, NZ, 0.0f);
    for (int i = 0; i < NR; ++i)
        for (int j = 0; j < NTH; ++j)
            for (int k = 0; k < NZ; ++k)
            {
                u[i][j][k] = static_cast<float>(150.0 + 8.0 * i - 3.0 * k);
                v[i][j][k] = static_cast<float>(-70.0 + 4.0 * j + 2.0 * k);
                w[i][j][k] = (k == NZ - 1)
                    ? 0.0f : static_cast<float>(i - j);
            }

    constexpr double dt_step = 0.2;
    CartesianCGridTransportContext context;
    context.begin(NR, NTH, NZ, dt_step, &metrics);
    context.accumulate(rho, u, v, w, dt_step);
    context.complete();
    context.require_valid(NR, NTH, NZ, dt_step);

    // Mirror the production continuity update from the retained FP32 face
    // history.  The scalar must recover against this exact ending density.
    for (int i = 1; i < NR - 1; ++i)
    {
        for (int j = 1; j < NTH - 1; ++j)
        {
            for (int k = 0; k < NZ; ++k)
            {
                const double x_div =
                    (static_cast<double>(context.mass_flux_x[i][j][k])
                     - static_cast<double>(context.mass_flux_x[i - 1][j][k]))
                    / metrics.dx;
                const double y_div =
                    (static_cast<double>(context.mass_flux_y[i][j][k])
                     - static_cast<double>(context.mass_flux_y[i][j - 1][k]))
                    / metrics.dy;
                const double z_top =
                    static_cast<double>(context.mass_flux_z[i][j][k]);
                const double z_bottom = k > 0
                    ? static_cast<double>(context.mass_flux_z[i][j][k - 1])
                    : 0.0;
                const double mass_divergence =
                    (x_div + y_div + (z_top - z_bottom) / metrics.dzeta)
                    / metrics.J(i, j, k);
                rho[i][j][k] = static_cast<float>(
                    static_cast<double>(rho[i][j][k])
                    - dt_step * mass_divergence);
                REQUIRE(rho[i][j][k] > 0.0f);
            }
        }
    }

    constexpr float q_constant = 0.0125f;
    Field3D scalar(NR, NTH, NZ, q_constant);
    Field3D transported;
    p.resize(NR, NTH, NZ, 90000.0f);
    theta.resize(NR, NTH, NZ, 300.0f);
    qv.resize(NR, NTH, NZ, q_constant);
    qc.resize(NR, NTH, NZ, 0.0f);
    qr.resize(NR, NTH, NZ, 0.0f);
    qi.resize(NR, NTH, NZ, 0.0f);
    qs.resize(NR, NTH, NZ, 0.0f);
    qg.resize(NR, NTH, NZ, 0.0f);
    qh.resize(NR, NTH, NZ, 0.0f);
    ScalarBoundaryTransportReport scalar_report;
    ScalarMoistEnergyTransportReportV1 energy_report;
    advect_scalar_conservative_3d_cartesian(
        scalar, transported, dt_step, false, &context, &scalar_report,
        ScalarMoistEnergyKind::WaterVapor, &energy_report);
    for (int i = 1; i < NR - 1; ++i)
        for (int j = 1; j < NTH - 1; ++j)
            for (int k = 0; k < NZ; ++k)
                CHECK(transported[i][j][k]
                      == Approx(q_constant).margin(3.0e-9));

    const tmv::DynamicsBoundaryMassReport report = context.boundary_mass_report(
        rho, metrics.dx, metrics.dy, metrics.dzeta);
    const double ledger_scale = std::max(
        1.0, std::max(std::abs(report.flux_implied_state_change_kg),
                      std::abs(report.committed_state_change_kg)));
    CHECK(std::abs(report.closure_residual_kg) / ledger_scale < 2.0e-6);
    CHECK(scalar_report.complete);
    CHECK(scalar_report.outward_boundary_inventory
          == Approx(report.outward_boundary_mass_kg * q_constant)
                 .epsilon(2.0e-7));
    CHECK(scalar_report.committed_state_change
          == Approx(report.committed_state_change_kg * q_constant)
                 .epsilon(5.0e-6));
    const double scalar_ledger_scale = std::max(
        1.0, std::max(
            std::abs(scalar_report.flux_implied_state_change),
            std::abs(scalar_report.committed_state_change)));
    CHECK(std::abs(scalar_report.closure_residual) / scalar_ledger_scale
          < 2.0e-6);
    CHECK(energy_report.complete);
    CHECK(std::isfinite(energy_report.density_history_remap_j));
    CHECK(std::isfinite(energy_report.limited_face_state_change_j));
    CHECK(energy_report.interior_conversion_j
          == energy_report.limited_face_state_change_j
              + energy_report.outward_boundary_energy_j);
    CHECK(energy_report.requested_state_change_j
          == energy_report.density_history_remap_j
              + energy_report.limited_face_state_change_j);
    CHECK(energy_report.storage_rounding_j
          == energy_report.committed_state_change_j
              - energy_report.requested_state_change_j);
    CHECK(energy_report.closure_residual_j == 0.0);
}

TEST_CASE("retained collocated terrain transport owns mapped state",
          "[physics][terrain][conservative][agrid]")
{
    const TerrainMetrics3D metrics = make_conservative_test_geometry(false);
    global_stagger_type = StaggerType::Collocated;
    CartesianCGridTransportContext context;
    REQUIRE_NOTHROW(
        context.begin(metrics.J.size_r(), metrics.J.size_th(),
                      metrics.J.size_z(), 1.0, &metrics));
    CHECK(context.active());
    CHECK(context.staggering() == StaggerType::Collocated);
    CHECK(context.terrain_metrics() == &metrics);
    global_stagger_type = StaggerType::CGrid;
}

TEST_CASE("continuous terrain motion satisfies space-time geometric conservation",
          "[physics][terrain][earth_fixed][moving_mesh][gcl]")
{
    for (const auto* shape : {"bell", "schar"})
    for (const auto* coordinate : {"btf", "smoothed"})
    for (const double direction : {-1.0, 1.0})
    {
        TerrainConfig config;
        config.scheme_id = shape;
        config.coord_id = coordinate;
        config.ztop = 8000.0;
        config.bell.h0 = 700.0;
        config.bell.a = 2100.0;
        config.bell.axisymmetric = true;
        config.schar.H = 600.0;
        config.schar.a = 2100.0;
        config.schar.ell = 1500.0;
        tmv::EarthFixedTerrainBuildRequest request;
        request.nr = 9; request.nth = 8; request.nz = 7;
        request.dx_m = 400.0; request.dy_m = 600.0;
        request.grid_origin_earth_m = {7312.5, -9125.0};
        request.feature_center_earth_m = {9120.0, -7020.0};
        bind_earth_fixed_feature(config, request);
        tmv::TerrainRuntimeState live;
        live.config = config;
        auto initial = tmv::prepare_earth_fixed_terrain_commit(live, request);
        REQUIRE(tmv::commit_earth_fixed_terrain(initial, live));
        tmv::domain::HorizontalDomainFrame frame;
        frame.earth_origin_at_t0_m = request.grid_origin_earth_m;
        frame.velocity_ms = {direction * 13.5, direction * -7.25};
        const auto old_j = live.metrics.J.values;
        const auto stationary = tmv::build_terrain_mesh_swept_volumes(live.metrics, live.metrics, {0,0}, 1.0);
        for (const auto* field : {&stationary.x, &stationary.y, &stationary.z})
            REQUIRE(std::all_of(field->values.begin(), field->values.end(), [](double value) { return value == 0.0; }));
        auto motion = tmv::prepare_earth_fixed_terrain_motion(live, frame, 0.0, 0.4);
        REQUIRE(live.metrics.J.values == old_j);
        const auto& next = motion.terrain.geometry.metrics;
        const auto& swept = motion.swept_volumes;
        const double logical_volume = request.dx_m * request.dy_m * live.metrics.dzeta;
        double physical_change = 0.0, boundary_exchange = 0.0;
        for (int i = 1; i < request.nr - 1; ++i)
            for (int j = 1; j < request.nth - 1; ++j)
                for (int k = 0; k < request.nz; ++k)
                {
                    const double volume0 = logical_volume * live.metrics.J(i,j,k);
                    const double volume1 = logical_volume * next.J(i,j,k);
                    const double change = swept.x(i+1,j,k) - swept.x(i,j,k)
                        + swept.y(i,j+1,k) - swept.y(i,j,k)
                        + swept.z(i,j,k+1) - swept.z(i,j,k);
                    const double updated_density = (1.2 * volume0 + 1.2 * change) / volume1;
                    CAPTURE(shape, coordinate, direction, i, j, k);
                    REQUIRE(updated_density == Approx(1.2).epsilon(2.0e-14));
                    physical_change += volume1 - volume0;
                    if (i == 1) boundary_exchange -= swept.x(i,j,k);
                    if (i + 2 == request.nr) boundary_exchange += swept.x(i+1,j,k);
                    if (j == 1) boundary_exchange -= swept.y(i,j,k);
                    if (j + 2 == request.nth) boundary_exchange += swept.y(i,j+1,k);
                    if (k == 0) boundary_exchange -= swept.z(i,j,k);
                    if (k + 1 == request.nz) boundary_exchange += swept.z(i,j,k+1);
                }
        REQUIRE(physical_change == Approx(boundary_exchange).margin(1.0e-4));
        const PhysicalCellDomain physical{1, request.nr - 1, 1, request.nth - 1, 0, request.nz};
        Field3D constant(request.nr, request.nth, request.nz, 1.25f), updated;
        TerrainContravariantFluxes relative;
        relative.x = swept.x; relative.y = swept.y; relative.z = swept.z;
        for (auto* field : {&relative.x, &relative.y, &relative.z})
            for (double& value : field->values) value *= -1.25;
        const auto uniform_report = tmv::update_moving_mesh_inventory(
            live.metrics, next, relative, physical, constant, updated, 0.0);
        for (std::size_t n = 0; n < updated.size(); ++n) REQUIRE(updated.data()[n] == 1.25f);
        REQUIRE(uniform_report.final_inventory - uniform_report.initial_inventory
            + uniform_report.boundary_outflow - uniform_report.storage_rounding
            == Approx(0.0).margin(1.0e-4));

        Field3D tracer(request.nr, request.nth, request.nz, 0.0f);
        for (int i = 0; i < request.nr; ++i)
            for (int j = 0; j < request.nth; ++j)
                for (int k = 0; k < request.nz; ++k)
                    tracer(i,j,k) = 1.0f + 0.1f*i + 0.05f*j + 0.01f*k;
        for (int i = 0; i <= request.nr; ++i)
            for (int j = 0; j < request.nth; ++j)
                for (int k = 0; k < request.nz; ++k)
                {
                    const double volume = -swept.x(i,j,k);
                    const int donor = std::clamp(volume >= 0.0 ? i-1 : i, 0, request.nr-1);
                    relative.x(i,j,k) = volume * tracer(donor,j,k);
                }
        for (int i = 0; i < request.nr; ++i)
            for (int j = 0; j <= request.nth; ++j)
                for (int k = 0; k < request.nz; ++k)
                {
                    const double volume = -swept.y(i,j,k);
                    const int donor = std::clamp(volume >= 0.0 ? j-1 : j, 0, request.nth-1);
                    relative.y(i,j,k) = volume * tracer(i,donor,k);
                }
        for (int i = 0; i < request.nr; ++i)
            for (int j = 0; j < request.nth; ++j)
                for (int k = 0; k <= request.nz; ++k)
                {
                    const double volume = -swept.z(i,j,k);
                    const int donor = std::clamp(volume >= 0.0 ? k-1 : k, 0, request.nz-1);
                    relative.z(i,j,k) = volume * tracer(i,j,donor);
                }
        const auto tracer_report = tmv::update_moving_mesh_inventory(
            live.metrics, next, relative, physical, tracer, updated, 0.0);
        REQUIRE(tracer_report.final_inventory - tracer_report.initial_inventory
            + tracer_report.boundary_outflow - tracer_report.storage_rounding
            == Approx(0.0).margin(1.0e-4));
        for (std::size_t n = 0; n < updated.size(); ++n)
        {
            REQUIRE(updated.data()[n] >= 1.0f);
            REQUIRE(updated.data()[n] <= 3.0f);
        }
        Field3D aliased = tracer;
        tmv::update_moving_mesh_inventory(live.metrics, next, relative, physical, aliased, aliased, 0.0);
        for (std::size_t n = 0; n < updated.size(); ++n) REQUIRE(aliased.data()[n] == updated.data()[n]);
        updated.fill(-9.0f);
        relative.x(request.nr-1,request.nth-2,request.nz-1) = std::numeric_limits<double>::quiet_NaN();
        REQUIRE_THROWS(tmv::update_moving_mesh_inventory(live.metrics, next, relative, physical, tracer, updated, 0.0));
        for (std::size_t n = 0; n < updated.size(); ++n) REQUIRE(updated.data()[n] == -9.0f);
        relative.x(request.nr-1,request.nth-2,request.nz-1) = 1.0e30;
        REQUIRE_THROWS(tmv::update_moving_mesh_inventory(live.metrics, next, relative, physical, tracer, updated, 0.0));
        for (std::size_t n = 0; n < updated.size(); ++n) REQUIRE(updated.data()[n] == -9.0f);
        const auto end = frame.earth_origin_at(0.4);
        const auto reverse = tmv::build_terrain_mesh_swept_volumes(next, live.metrics,
            {request.grid_origin_earth_m.x_m - end.x_m, request.grid_origin_earth_m.y_m - end.y_m}, 0.4);
        for (std::size_t n = 0; n < swept.z.values.size(); ++n)
            REQUIRE(reverse.z.values[n] == -swept.z.values[n]);
        REQUIRE(tmv::commit_earth_fixed_terrain(motion.terrain, live));
        REQUIRE(live.earth_fixed_placement.grid_origin_earth_m.x_m == end.x_m);
        REQUIRE(live.earth_fixed_placement.grid_origin_earth_m.y_m == end.y_m);
        REQUIRE_THROWS(tmv::prepare_earth_fixed_terrain_motion(live, frame, 0.0, 0.4));
        auto continuation = tmv::prepare_earth_fixed_terrain_motion(live, frame, 0.4, 0.8);
        REQUIRE(continuation.terrain.next_placement.generation == live.earth_fixed_placement.generation + 1);
        auto malformed = continuation.terrain.geometry.metrics;
        malformed.Gx_face(3,3,2) += 1.0;
        REQUIRE_THROWS(tmv::build_terrain_mesh_swept_volumes(live.metrics, malformed, {5.4,-2.9}, 0.4));
        REQUIRE_THROWS(tmv::build_terrain_mesh_swept_volumes(live.metrics, next, {0,0}, 0.0));
        auto invalid = live.metrics;
        invalid.Jx_face(1,1,1) = -1.0;
        REQUIRE_THROWS(tmv::build_terrain_mesh_swept_volumes(live.metrics, invalid, {0,0}, 1.0));
        invalid = live.metrics;
        invalid.zeta(1,1,1) += 1.0;
        REQUIRE_THROWS(tmv::build_terrain_mesh_swept_volumes(live.metrics, invalid, {0,0}, 1.0));
        invalid = live.metrics;
        invalid.z_face.values.pop_back();
        REQUIRE_THROWS(tmv::build_terrain_mesh_swept_volumes(live.metrics, invalid, {0,0}, 1.0));
        REQUIRE(live.earth_fixed_placement.grid_origin_earth_m.x_m == end.x_m);
    }
}

TEST_CASE("cell-averaged Earth terrain moves without sweeping through its fixed surfaces",
          "[physics][terrain][earth_fixed][volume_geometry][gcl]")
{
    for (const auto* shape : {"bell", "schar"})
    for (const auto* coordinate : {"btf", "smoothed"})
    for (const double sign : {-1.0, 1.0})
    {
        TerrainConfig config;
        config.scheme_id = shape;
        config.coord_id = coordinate;
        config.ztop = 8000.0;
        config.bell.h0 = 700.0; config.bell.a = 2100.0; config.bell.axisymmetric = true;
        config.schar.H = 600.0; config.schar.a = 2100.0; config.schar.ell = 1500.0;
        tmv::EarthFixedTerrainBuildRequest request;
        request.nr = 9; request.nth = 8; request.nz = 7;
        request.dx_m = 400.0; request.dy_m = 600.0;
        request.grid_origin_earth_m = {7312.5,-9125.0};
        request.feature_center_earth_m = {9120.0,-7020.0};
        request.sampling = tmv::EarthFixedTerrainSampling::CellAverageBilinear;
        bind_earth_fixed_feature(config, request);
        auto aligned = request;
        aligned.grid_origin_earth_m = request.feature_center_earth_m;
        const auto node_column = [&](int i) {
            if (config.scheme_id == "schar") return topography::eval_schar(i*request.dx_m,config.schar).h;
            return 0.5 * (topography::eval_bell(i*request.dx_m,0.0,config.bell).h
                + topography::eval_bell(i*request.dx_m,request.dy_m,config.bell).h);
        };
        const double h0=node_column(0), h1=node_column(1), h2=node_column(2);
        const auto aligned_geometry = tmv::build_earth_fixed_volume_geometry(config,aligned);
        REQUIRE(aligned_geometry.topography.h[0][0] == Approx(0.5*(h0+h1)).epsilon(1.0e-13));
        const auto half_cell = tmv::build_earth_fixed_volume_geometry_interval(config,aligned,
            {aligned.grid_origin_earth_m.x_m+0.5*request.dx_m, aligned.grid_origin_earth_m.y_m},1.0);
        REQUIRE(half_cell.end.topography.h[0][0] == Approx(0.125*h0+0.75*h1+0.125*h2).epsilon(1.0e-13));
        REQUIRE(half_cell.stage.topography.h[0][0] == Approx((7.0*h0+16.0*h1+h2)/24.0).epsilon(1.0e-13));
        tmv::TerrainRuntimeState live;
        live.config = config;
        auto initial = tmv::prepare_earth_fixed_terrain_commit(live, request);
        REQUIRE(tmv::commit_earth_fixed_terrain(initial, live));
        tmv::domain::HorizontalDomainFrame frame;
        frame.earth_origin_at_t0_m = request.grid_origin_earth_m;
        frame.velocity_ms = {sign * 300.0, sign * -450.0};
        auto motion = tmv::prepare_earth_fixed_terrain_motion(live, frame, 0.0, 1.0);
        REQUIRE_NOTHROW(tmv::terrain_transport::validate_geometry(motion.stage_geometry.metrics));
        const auto& swept = motion.swept_volumes;
        for (double volume : swept.z.values) REQUIRE(volume == 0.0);
        const auto& after = motion.terrain.geometry.metrics;
        const double volume = request.dx_m * request.dy_m * live.metrics.dzeta;
        for (int i = 1; i < request.nr-1; ++i)
            for (int j = 1; j < request.nth-1; ++j)
                for (int k = 0; k < request.nz; ++k)
                {
                    const double change = swept.x(i+1,j,k)-swept.x(i,j,k)
                        + swept.y(i,j+1,k)-swept.y(i,j,k);
                    const double density = (volume*live.metrics.J(i,j,k)+change)/(volume*after.J(i,j,k));
                    CAPTURE(shape,coordinate,sign,i,j,k);
                    REQUIRE(density == Approx(1.0).epsilon(2.0e-14));
                }
        auto reverse_request = request;
        reverse_request.grid_origin_earth_m = frame.earth_origin_at(1.0);
        const auto reverse = tmv::build_earth_fixed_volume_geometry_interval(config, reverse_request,
            request.grid_origin_earth_m, 1.0);
        for (std::size_t n = 0; n < swept.x.values.size(); ++n)
            REQUIRE(reverse.swept_volumes.x.values[n] == Approx(-swept.x.values[n]).epsilon(1.0e-13));
        REQUIRE(tmv::commit_earth_fixed_terrain(motion.terrain, live));
        REQUIRE(live.earth_fixed_placement.sampling == tmv::EarthFixedTerrainSampling::CellAverageBilinear);
        REQUIRE_NOTHROW(tmv::prepare_earth_fixed_terrain_motion(live, frame, 1.0, 2.0));
        request.sampling = tmv::EarthFixedTerrainSampling::Point;
        REQUIRE_THROWS(tmv::prepare_earth_fixed_terrain_commit(live, request));
    }
}
