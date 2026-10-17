/**
 * @file test_trajectories.cpp
 * @brief Analytic unit tests for the online Lagrangian parcel integrator and
 *        the vorticity-budget sampling it records along each path.
 *
 * Constant flows make RK2 exact, so the position checks are tight; the
 * Cartesian solid-body-rotation case verifies radius conservation under a
 * rotating wind; the exit case checks domain deactivation.
 * The budget tests verify the shared Cartesian C-grid kernel against analytic
 * gradients and that the trajectory sampler records a fresh, finite budget on
 * the C-grid (and NaN where the online budget is unsupported).
 */
#include "catch2/catch.hpp"
#include "diagnostics/trajectories.hpp"
#include "diagnostics/vorticity_budget_kernel.hpp"
#include "core/field/field3d.hpp"
#include "core/infra/coordinate_system.hpp"
#include "core/infra/grid_coordinates.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>

namespace
{
constexpr double kTwoPi = 6.283185307179586;

// One parcel at an explicit centre (zero box extent, single seed point).
tmv::TrajectoryConfig single_parcel(double cx, double cy, double cz)
{
    tmv::TrajectoryConfig c;
    c.enabled = true;
    c.nx = c.ny = c.nz = 1;
    c.seed_time_s      = 0.0;
    c.half_x_m = c.half_y_m = c.half_z_m = 0.0;
    c.center_x_m                         = cx;
    c.center_y_m                         = cy;
    c.center_z_m                         = cz;
    return c;
}
} // namespace

TEST_CASE("trajectory NPY publication preserves the retained samples",
          "[trajectories][output][memory]")
{
    const int nr = 8, nth = 8, nz = 6;
    Field3D u(nr, nth, nz, 1.0f), v(nr, nth, nz, 0.0f),
        w(nr, nth, nz, 0.0f), rho(nr, nth, nz, 1.0f),
        pressure(nr, nth, nz, 1.0e5f), zeta(nr, nth, nz, 0.0f);
    tmv::TrajectorySet trajectories;
    trajectories.configure(
        single_parcel(3000.0, 3000.0, 1500.0),
        CoordinateSystem::Cartesian, false, nr, nth, nz,
        1000.0, 1000.0, 500.0);
    trajectories.seed(0.0, u, v, w, rho, pressure, zeta);
    trajectories.advance(1.0, 1.0, u, v, w, rho, pressure, zeta);

    const auto output = std::filesystem::temp_directory_path() /
        "nimbus_trajectory_streaming.npy";
    std::filesystem::remove(output);
    REQUIRE(trajectories.write(output.string()));

    std::ifstream input(output, std::ios::binary | std::ios::ate);
    REQUIRE(input);
    const auto length = input.tellg();
    REQUIRE(length > 10);
    std::vector<std::uint8_t> bytes(static_cast<std::size_t>(length));
    input.seekg(0);
    input.read(reinterpret_cast<char*>(bytes.data()), length);
    REQUIRE(input);
    const std::size_t header_size =
        static_cast<std::size_t>(bytes[8]) |
        (static_cast<std::size_t>(bytes[9]) << 8U);
    const std::size_t payload_offset = 10 + header_size;
    REQUIRE(bytes.size() == payload_offset +
            trajectories.total_samples() * sizeof(tmv::TrajectorySample));
    REQUIRE(trajectories.parcels().front().path.size() == 2);
    std::size_t sample_offset = payload_offset;
    for (const auto& sample : trajectories.parcels().front().path)
    {
        CHECK(std::memcmp(bytes.data() + sample_offset, &sample,
                          sizeof(sample)) == 0);
        sample_offset += sizeof(sample);
    }
    std::filesystem::remove(output);
}

TEST_CASE("trajectory configuration rejects an oversized seed lattice",
          "[trajectories][config][memory]")
{
    tmv::TrajectoryConfig config;
    config.enabled = true;
    config.nx = 1024;
    config.ny = 1024;
    config.nz = 3;
    tmv::TrajectorySet trajectories;
    REQUIRE_THROWS_WITH(
        trajectories.configure(
            config, CoordinateSystem::Cartesian, false,
            8, 8, 6, 1000.0, 1000.0, 500.0),
        Catch::Matchers::Contains("parcel limit"));
}

TEST_CASE("Cartesian constant horizontal wind advects linearly", "[trajectories][analytical]")
{
    const int    nr = 20, nth = 20, nz = 10;
    const double dx = 1000.0, dy = 1000.0, dz = 500.0;
    Field3D      u(nr, nth, nz, 10.0f), v(nr, nth, nz, 0.0f), w(nr, nth, nz, 0.0f), zeta(nr, nth, nz, 0.0f),
        rho(nr, nth, nz, 1.0f), pres(nr, nth, nz, 1.0e5f);

    tmv::TrajectorySet ts;
    ts.configure(single_parcel(5000.0, 5000.0, 2000.0),
                 CoordinateSystem::Cartesian,
                 /*staggered=*/false,
                 nr,
                 nth,
                 nz,
                 dx,
                 dy,
                 dz);
    ts.seed(0.0, u, v, w, rho, pres, zeta);
    REQUIRE(ts.count() == 1);

    const double dt = 10.0;
    const int    N  = 10;
    for (int s = 0; s < N; ++s)
        ts.advance((s + 1) * dt, dt, u, v, w, rho, pres, zeta);

    const auto& p = ts.parcels()[0];
    REQUIRE(p.active);
    // u = 10 m/s over N*dt = 100 s -> +1000 m in x; y, z unchanged.
    CHECK(p.x == Approx(6000.0).margin(1e-3));
    CHECK(p.y == Approx(5000.0).margin(1e-3));
    CHECK(p.z == Approx(2000.0).margin(1e-3));
    CHECK(p.path.size() == static_cast<std::size_t>(N + 1)); // seed + N steps
}

TEST_CASE("trajectory checkpoint state restores exact history and continues",
          "[trajectories][checkpoint]")
{
    const int nr = 20, nth = 20, nz = 10;
    const double dx = 1000.0, dy = 1000.0, dz = 500.0;
    Field3D u(nr, nth, nz, 10.0f), v(nr, nth, nz, 0.0f),
        w(nr, nth, nz, 0.0f), zeta(nr, nth, nz, 0.0f),
        rho(nr, nth, nz, 1.0f), pres(nr, nth, nz, 1.0e5f);
    const auto config = single_parcel(5000.0, 5000.0, 2000.0);

    tmv::TrajectorySet continuous;
    continuous.configure(config, CoordinateSystem::Cartesian, false,
                         nr, nth, nz, dx, dy, dz);
    continuous.seed(0.0, u, v, w, rho, pres, zeta);
    continuous.advance(10.0, 10.0, u, v, w, rho, pres, zeta);

    tmv::TrajectoryCheckpointState saved;
    saved.configuration = continuous.configuration();
    saved.seeded = continuous.seeded();
    saved.parcels = continuous.parcels();

    tmv::TrajectorySet resumed;
    resumed.configure(config, CoordinateSystem::Cartesian, false,
                      nr, nth, nz, dx, dy, dz);
    std::string error;
    REQUIRE(resumed.restore_checkpoint_state(std::move(saved), 10.0, error));
    resumed.advance(20.0, 10.0, u, v, w, rho, pres, zeta);
    continuous.advance(20.0, 10.0, u, v, w, rho, pres, zeta);

    REQUIRE(resumed.count() == 1);
    REQUIRE(resumed.parcels()[0].path.size() == 3);
    CHECK(resumed.parcels()[0].x == continuous.parcels()[0].x);
    CHECK(resumed.parcels()[0].path.back().t ==
          continuous.parcels()[0].path.back().t);
    CHECK(resumed.parcels()[0].path.back().x ==
          continuous.parcels()[0].path.back().x);

    tmv::TrajectoryCheckpointState mismatch;
    mismatch.configuration = config;
    mismatch.configuration.half_x_m = 1.0;
    mismatch.seeded = true;
    mismatch.parcels = continuous.parcels();
    tmv::TrajectorySet refused;
    refused.configure(config, CoordinateSystem::Cartesian, false,
                      nr, nth, nz, dx, dy, dz);
    REQUIRE_FALSE(refused.restore_checkpoint_state(
        std::move(mismatch), 20.0, error));
    CHECK(refused.count() == 0);
    CHECK(error.find("configuration") != std::string::npos);
}

TEST_CASE("trajectory recording cadence bounds history without changing integration",
          "[trajectories][cadence][memory][checkpoint]")
{
    const int nr = 20, nth = 20, nz = 10;
    Field3D u(nr, nth, nz, 10.0f), v(nr, nth, nz, 0.0f),
        w(nr, nth, nz, 0.0f), zeta(nr, nth, nz, 0.0f),
        rho(nr, nth, nz, 1.0f), pres(nr, nth, nz, 1.0e5f);
    auto config = single_parcel(5000.0, 5000.0, 2000.0);
    config.sample_interval_s = 3.0;
    config.max_samples_per_parcel = 3;

    tmv::TrajectorySet continuous;
    continuous.configure(config, CoordinateSystem::Cartesian, false,
                         nr, nth, nz, 1000.0, 1000.0, 500.0);
    continuous.seed(0.0, u, v, w, rho, pres, zeta);
    for (int step = 1; step <= 4; ++step)
        continuous.advance(step, 1.0, u, v, w, rho, pres, zeta);
    CHECK(continuous.parcels()[0].x == Approx(5040.0));
    REQUIRE(continuous.parcels()[0].path.size() == 2);
    CHECK(continuous.parcels()[0].path[0].t == 0.0f);
    CHECK(continuous.parcels()[0].path[1].t == 3.0f);
    CHECK(continuous.next_sample_time_s() == Approx(6.0));

    tmv::TrajectoryCheckpointState saved;
    saved.configuration = continuous.configuration();
    saved.seeded = continuous.seeded();
    saved.next_sample_time_s = continuous.next_sample_time_s();
    saved.parcels = continuous.parcels();
    tmv::TrajectorySet resumed;
    resumed.configure(config, CoordinateSystem::Cartesian, false,
                      nr, nth, nz, 1000.0, 1000.0, 500.0);
    std::string error;
    REQUIRE(resumed.restore_checkpoint_state(std::move(saved), 4.0, error));

    for (int step = 5; step <= 9; ++step)
    {
        continuous.advance(step, 1.0, u, v, w, rho, pres, zeta);
        resumed.advance(step, 1.0, u, v, w, rho, pres, zeta);
    }
    REQUIRE(resumed.parcels()[0].path.size() == 3);
    CHECK(resumed.parcels()[0].path[0].t == 3.0f);
    CHECK(resumed.parcels()[0].path[1].t == 6.0f);
    CHECK(resumed.parcels()[0].path[2].t == 9.0f);
    CHECK(resumed.parcels()[0].x == Approx(5090.0));
    CHECK(resumed.parcels()[0].x == continuous.parcels()[0].x);
    CHECK(resumed.next_sample_time_s() == continuous.next_sample_time_s());
}

TEST_CASE("automatic trajectory seed boxes fit inside small physical domains", "[trajectories][seeding][small-domain]")
{
    const int    nr = 10, nth = 10, nz = 13;
    const double dx = 1000.0, dy = 1000.0, dz = 500.0;
    Field3D      u(nr, nth, nz, 0.0f), v(nr, nth, nz, 0.0f), w(nr, nth, nz, 0.0f), zeta(nr, nth, nz, 0.0f),
        rho(nr, nth, nz, 1.0f), pres(nr, nth, nz, 1.0e5f);

    tmv::TrajectoryConfig config;
    config.enabled = true;
    config.nx = config.ny = 2;
    config.nz             = 3;
    // Default half extents are deliberately wider than the C-grid physical
    // interior once closure cells are excluded.

    tmv::TrajectorySet trajectories;
    trajectories.configure(config, CoordinateSystem::Cartesian, true, nr, nth, nz, dx, dy, dz);
    trajectories.seed(0.0, u, v, w, rho, pres, zeta);

    REQUIRE(trajectories.seeded());
    REQUIRE(trajectories.count() == 12);
    for (const auto& parcel : trajectories.parcels())
    {
        CHECK(parcel.active);
        CHECK(parcel.x >= 1000.0);
        CHECK(parcel.x <= 9000.0);
        CHECK(parcel.y >= 1000.0);
        CHECK(parcel.y <= 9000.0);
    }
}

TEST_CASE("terrain trajectories preserve AGL in a terrain-parallel uniform flow",
          "[trajectories][terrain][analytical][budget]")
{
    constexpr int nr = 12;
    constexpr int nth = 10;
    constexpr int nz = 8;
    constexpr double dx = 1000.0;
    constexpr double dy = 800.0;
    constexpr double dz = 500.0;
    constexpr double slope_x = 0.1;
    constexpr double wind_u = 20.0;
    constexpr double wind_w = slope_x * wind_u;

    TerrainMetrics3D terrain;
    terrain.z.resize(nr, nth, nz, 0.0);
    terrain.J.resize(nr, nth, nz, 1.0);
    terrain.mx.resize(nr, nth, nz, -slope_x);
    terrain.my.resize(nr, nth, nz, 0.0);
    terrain.zeta.resize(nr, nth, nz, 0.0);
    terrain.z_face.resize(nr, nth, nz + 1, 0.0);
    terrain.Jx_face.resize(nr + 1, nth, nz, 1.0);
    terrain.Jy_face.resize(nr, nth + 1, nz, 1.0);
    terrain.Gx_face.resize(nr, nth, nz + 1, -slope_x);
    terrain.Gy_face.resize(nr, nth, nz + 1, 0.0);
    terrain.dx = dx;
    terrain.dy = dy;
    terrain.dzeta = dz;
    for (int i = 0; i < nr; ++i)
        for (int j = 0; j < nth; ++j)
        {
            const double ground = slope_x
                * tmv::grid_coordinates::scalar_center(i, dx);
            for (int k = 0; k <= nz; ++k)
                terrain.z_face(i, j, k) = ground + k * dz;
            for (int k = 0; k < nz; ++k)
            {
                terrain.z(i, j, k) = ground + (k + 0.5) * dz;
                terrain.zeta(i, j, k) = (k + 0.5) * dz;
            }
        }

    Field3D u(nr, nth, nz, static_cast<float>(wind_u));
    Field3D v(nr, nth, nz, 0.0f);
    Field3D w(nr, nth, nz, static_cast<float>(wind_w));
    Field3D rho(nr, nth, nz, 1.0f);
    Field3D pressure(nr, nth, nz, 90000.0f);
    Field3D zeta(nr, nth, nz, 0.0f);

    SECTION("Earth-fixed ground sampling is independent of the translating window")
    {
        constexpr double x = 5000.0, y = 3200.0, agl = 100.0;
        auto frame = tmv::domain::HorizontalDomainFrame::stationary();
        frame.velocity_ms = {10.0, -5.0};
        const auto seed_config = single_parcel(x, y, slope_x * x + agl);
        tmv::TrajectorySet fixed, attached;
        fixed.configure(seed_config, CoordinateSystem::Cartesian, true,
            nr, nth, nz, dx, dy, dz, frame, &terrain, true);
        attached.configure(seed_config, CoordinateSystem::Cartesian, true,
            nr, nth, nz, dx, dy, dz, frame, &terrain, false);
        fixed.seed(0.0, u, v, w, rho, pressure, zeta);
        attached.seed(0.0, u, v, w, rho, pressure, zeta);
        REQUIRE(fixed.count() == 1);
        REQUIRE(attached.count() == 1);
        CHECK(fixed.parcels().front().path.front().w == Approx(wind_w).margin(1.0e-6));
        const double attached_ground_w = slope_x * (wind_u - frame.velocity_ms.u_ms);
        const double attached_w = attached_ground_w * (1.0 - agl / dz) + wind_w * agl / dz;
        CHECK(attached.parcels().front().path.front().w == Approx(attached_w).margin(1.0e-6));
    }

    constexpr double seed_x = 5000.0;
    constexpr double seed_y = 3200.0;
    tmv::TrajectoryConfig config = single_parcel(seed_x, seed_y, 0.0);
    config.nx = 3;
    config.half_x_m = 1000.0;
    config.center_z_m = std::numeric_limits<double>::quiet_NaN();
    tmv::TrajectorySet trajectories;
    trajectories.configure(
        config, CoordinateSystem::Cartesian, true,
        nr, nth, nz, dx, dy, dz,
        tmv::domain::HorizontalDomainFrame::stationary(), &terrain);
    trajectories.seed(0.0, u, v, w, rho, pressure, zeta);
    REQUIRE(trajectories.count() == 3);
    for (const auto& parcel : trajectories.parcels())
    {
        CHECK(parcel.z - slope_x * parcel.x
              == Approx(1000.0).margin(1.0e-9));
        CHECK(parcel.path.front().w == Approx(wind_w).margin(1.0e-6));
    }

    constexpr double step_s = 10.0;
    trajectories.advance(
        step_s, step_s, u, v, w, rho, pressure, zeta);
    for (std::size_t index = 0; index < trajectories.parcels().size(); ++index)
    {
        const auto& parcel = trajectories.parcels()[index];
        const double initial_x = seed_x - config.half_x_m
            + static_cast<double>(index) * config.half_x_m;
        REQUIRE(parcel.active);
        CHECK(parcel.x == Approx(initial_x + wind_u * step_s).margin(1.0e-9));
        CHECK(parcel.z - slope_x * parcel.x
              == Approx(1000.0).margin(1.0e-8));
        REQUIRE(parcel.path.size() == 2);
        CHECK(parcel.path.back().w == Approx(wind_w).margin(1.0e-6));
        CHECK(parcel.path.back().zeta == Approx(0.0f).margin(1.0e-8));
        CHECK(parcel.path.back().tilting == Approx(0.0f).margin(1.0e-8));
        CHECK(parcel.path.back().stretching == Approx(0.0f).margin(1.0e-8));
        CHECK(parcel.path.back().baroclinic == Approx(0.0f).margin(1.0e-8));
    }
}

TEST_CASE("terrain trajectories reject incomplete physical geometry",
          "[trajectories][terrain][contract]")
{
    constexpr int nr = 4;
    constexpr int nth = 4;
    constexpr int nz = 3;
    TerrainMetrics3D terrain;
    terrain.z.resize(nr, nth, nz, 0.0);
    terrain.J.resize(nr, nth, nz, 1.0);
    terrain.mx.resize(nr, nth, nz, 0.0);
    terrain.my.resize(nr, nth, nz, 0.0);
    terrain.zeta.resize(nr, nth, nz, 0.0);
    terrain.z_face.resize(nr, nth, nz + 1, 0.0);
    terrain.Jx_face.resize(nr + 1, nth, nz, 1.0);
    terrain.Jy_face.resize(nr, nth + 1, nz, 1.0);
    terrain.Gx_face.resize(nr, nth, nz + 1, 0.0);
    terrain.Gy_face.resize(nr, nth, nz + 1, 0.0);
    terrain.dx = 1000.0;
    terrain.dy = 1000.0;
    terrain.dzeta = 500.0;
    for (int i = 0; i < nr; ++i)
        for (int j = 0; j < nth; ++j)
            for (int k = 0; k < nz; ++k)
                terrain.z(i, j, k) = 250.0 + 500.0 * k;

    tmv::TrajectorySet trajectories;
    REQUIRE_THROWS_WITH(
        trajectories.configure(
            single_parcel(1500.0, 1500.0, 1000.0),
            CoordinateSystem::Cartesian, true,
            nr, nth, nz, 1000.0, 1000.0, 500.0,
            tmv::domain::HorizontalDomainFrame::stationary(), &terrain),
        Catch::Matchers::Contains("finite increasing interface heights"));
}

TEST_CASE("Translating-frame trajectories integrate on the grid and publish Earth positions",
          "[trajectories][analytical][domain-motion]")
{
    const int    nr = 20, nth = 20, nz = 10;
    const double dx = 1000.0, dy = 1000.0, dz = 500.0;
    Field3D      u(nr, nth, nz, 17.0f), v(nr, nth, nz, 1.0f), w(nr, nth, nz, 0.0f), zeta(nr, nth, nz, 0.0f),
        rho(nr, nth, nz, 1.0f), pres(nr, nth, nz, 1.0e5f);

    tmv::domain::HorizontalDomainFrame frame;
    frame.earth_origin_at_t0_m = {1000.0, 2000.0};
    frame.velocity_ms          = {7.0, -3.0};

    tmv::TrajectorySet ts;
    ts.configure(
        single_parcel(6000.0, 6000.0, 2000.0), CoordinateSystem::Cartesian, false, nr, nth, nz, dx, dy, dz, frame);
    ts.seed(0.0, u, v, w, rho, pres, zeta);
    REQUIRE(ts.count() == 1);
    CHECK(ts.parcels()[0].x == Approx(5000.0));
    CHECK(ts.parcels()[0].y == Approx(4000.0));
    CHECK(ts.parcels()[0].path[0].x == Approx(6000.0f));
    CHECK(ts.parcels()[0].path[0].y == Approx(6000.0f));

    ts.advance(10.0, 10.0, u, v, w, rho, pres, zeta);
    const auto& parcel = ts.parcels()[0];
    CHECK(parcel.x == Approx(5100.0));
    CHECK(parcel.y == Approx(4040.0));
    REQUIRE(parcel.path.size() == 2);
    CHECK(parcel.path.back().x == Approx(6170.0f));
    CHECK(parcel.path.back().y == Approx(6010.0f));
}

TEST_CASE("D5 trajectory translation preserves Earth paths and retires only lost parcels",
          "[trajectories][analytical][domain-motion][D5]")
{
    const int    nr = 20, nth = 20, nz = 10;
    const double dx = 1000.0, dy = 1000.0, dz = 500.0;
    Field3D      u(nr, nth, nz, 17.0f), v(nr, nth, nz, 1.0f), w(nr, nth, nz, 0.0f), zeta(nr, nth, nz, 0.0f),
        rho(nr, nth, nz, 1.0f), pres(nr, nth, nz, 1.0e5f);

    tmv::domain::HorizontalDomainFrame before;
    before.earth_origin_at_t0_m              = {1000.0, 2000.0};
    before.velocity_ms                       = {7.0, -3.0};
    tmv::domain::HorizontalDomainFrame after = before;
    after.earth_origin_at_t0_m.x_m += 2000.0;
    after.earth_origin_at_t0_m.y_m -= 1000.0;

    tmv::TrajectorySet retained;
    retained.configure(
        single_parcel(6000.0, 6000.0, 2000.0), CoordinateSystem::Cartesian, false, nr, nth, nz, dx, dy, dz, before);
    retained.seed(0.0, u, v, w, rho, pres, zeta);
    const auto path_before = retained.parcels()[0].path;
    retained.preflight_domain_recycling(2000.0, -1000.0, after);
    retained.commit_domain_recycling(2000.0, -1000.0, after);
    REQUIRE(retained.parcels()[0].active);
    CHECK(retained.parcels()[0].x == Approx(3000.0));
    CHECK(retained.parcels()[0].y == Approx(5000.0));
    REQUIRE(retained.parcels()[0].path.size() == path_before.size());
    CHECK(retained.parcels()[0].path[0].x == path_before[0].x);
    CHECK(retained.parcels()[0].path[0].y == path_before[0].y);

    retained.advance(10.0, 10.0, u, v, w, rho, pres, zeta);
    REQUIRE(retained.parcels()[0].path.size() == 2);
    CHECK(retained.parcels()[0].path.back().x == Approx(6170.0f));
    CHECK(retained.parcels()[0].path.back().y == Approx(6010.0f));

    tmv::TrajectorySet retired;
    retired.configure(
        single_parcel(2500.0, 6000.0, 2000.0), CoordinateSystem::Cartesian, false, nr, nth, nz, dx, dy, dz, before);
    retired.seed(0.0, u, v, w, rho, pres, zeta);
    REQUIRE(retired.parcels()[0].active);
    retired.preflight_domain_recycling(2000.0, -1000.0, after);
    retired.commit_domain_recycling(2000.0, -1000.0, after);
    REQUIRE_FALSE(retired.parcels()[0].active);
    REQUIRE(retired.parcels()[0].path.size() == 1);
}

TEST_CASE("Constant-wind advection is staggering-invariant", "[trajectories][analytical]")
{
    const int    nr = 20, nth = 20, nz = 10;
    const double dx = 1000.0, dy = 1000.0, dz = 500.0;
    Field3D      u(nr, nth, nz, -8.0f), v(nr, nth, nz, 4.0f), w(nr, nth, nz, 0.0f), zeta(nr, nth, nz, 0.0f),
        rho(nr, nth, nz, 1.0f), pres(nr, nth, nz, 1.0e5f);

    for (bool stag : {false, true})
    {
        tmv::TrajectorySet ts;
        ts.configure(
            single_parcel(10000.0, 10000.0, 2000.0), CoordinateSystem::Cartesian, stag, nr, nth, nz, dx, dy, dz);
        ts.seed(0.0, u, v, w, rho, pres, zeta);
        const double dt = 5.0;
        for (int s = 0; s < 10; ++s)
            ts.advance((s + 1) * dt, dt, u, v, w, rho, pres, zeta);
        const auto& p = ts.parcels()[0];
        // 50 s: x += -8*50 = -400 -> 9600 ; y += 4*50 = 200 -> 10200.
        CHECK(p.x == Approx(9600.0).margin(1e-3));
        CHECK(p.y == Approx(10200.0).margin(1e-3));
    }
}

TEST_CASE("C-grid trajectory samples vertical velocity on physical upper faces",
          "[trajectories][analytical][coordinates]")
{
    const int        nr = 8, nth = 8, nz = 8;
    const double     dx = 1000.0, dy = 1000.0, dz = 500.0;
    constexpr double gradient = 1.0e-3;
    Field3D          u(nr, nth, nz, 0.0f), v(nr, nth, nz, 0.0f), w(nr, nth, nz, 0.0f), zeta(nr, nth, nz, 0.0f),
        rho(nr, nth, nz, 1.0f), pres(nr, nth, nz, 1.0e5f);
    for (int i = 0; i < nr; ++i)
        for (int j = 0; j < nth; ++j)
            for (int k = 0; k < nz; ++k)
                w[i][j][k] = static_cast<float>(gradient * (k + 1) * dz);

    constexpr double   sample_z = 1250.0;
    tmv::TrajectorySet ts;
    ts.configure(single_parcel(3500.0, 3500.0, sample_z), CoordinateSystem::Cartesian, true, nr, nth, nz, dx, dy, dz);
    ts.seed(0.0, u, v, w, rho, pres, zeta);

    REQUIRE(ts.count() == 1);
    REQUIRE(ts.parcels()[0].path.size() == 1);
    REQUIRE(ts.parcels()[0].path[0].w == Approx(gradient * sample_z).epsilon(1.0e-6));
}

TEST_CASE("Vertical wind lifts a parcel and clamps at the lid", "[trajectories][analytical]")
{
    const int    nr = 10, nth = 10, nz = 10;
    const double dx = 1000.0, dy = 1000.0, dz = 500.0;
    Field3D      u(nr, nth, nz, 0.0f), v(nr, nth, nz, 0.0f), w(nr, nth, nz, 5.0f), zeta(nr, nth, nz, 0.0f),
        rho(nr, nth, nz, 1.0f), pres(nr, nth, nz, 1.0e5f);

    tmv::TrajectorySet ts;
    ts.configure(single_parcel(5000.0, 5000.0, 1000.0), CoordinateSystem::Cartesian, false, nr, nth, nz, dx, dy, dz);
    ts.seed(0.0, u, v, w, rho, pres, zeta);
    const double dt = 10.0;
    for (int s = 0; s < 200; ++s)
        ts.advance((s + 1) * dt, dt, u, v, w, rho, pres, zeta);

    const auto& p = ts.parcels()[0];
    // Collocated k=0 and k=nz-1 are boundary-closure storage. The physical
    // column is k=[1,nz-1), whose upper face is (nz-1)*dz = 4500 m.
    CHECK(p.z == Approx(4500.0).margin(1e-6)); // clamped at top physical face
    CHECK(p.active);                           // vertical clamp must not deactivate
}

TEST_CASE("Collocated trajectory seeds honor the physical-cell domain", "[trajectories][analytical][coordinates]")
{
    const int    nr = 10, nth = 10, nz = 10;
    const double dx = 1000.0, dy = 1000.0, dz = 500.0;
    Field3D      u(nr, nth, nz, 0.0f), v(nr, nth, nz, 0.0f), w(nr, nth, nz, 0.0f), zeta(nr, nth, nz, 0.0f),
        rho(nr, nth, nz, 1.0f), pres(nr, nth, nz, 1.0e5f);

    SECTION("horizontal boundary-closure storage is not seedable volume")
    {
        tmv::TrajectorySet ts;
        ts.configure(single_parcel(500.0, 5000.0, 1000.0),
                     CoordinateSystem::Cartesian,
                     /*staggered=*/false,
                     nr,
                     nth,
                     nz,
                     dx,
                     dy,
                     dz);
        ts.seed(0.0, u, v, w, rho, pres, zeta);
        CHECK(ts.count() == 0);
    }

    SECTION("vertical seed coordinates clamp to the lower physical face")
    {
        tmv::TrajectorySet ts;
        ts.configure(single_parcel(5000.0, 5000.0, 0.0),
                     CoordinateSystem::Cartesian,
                     /*staggered=*/false,
                     nr,
                     nth,
                     nz,
                     dx,
                     dy,
                     dz);
        ts.seed(0.0, u, v, w, rho, pres, zeta);
        REQUIRE(ts.count() == 1);
        CHECK(ts.parcels()[0].z == Approx(500.0).margin(1e-6));
    }
}

TEST_CASE("Cartesian solid-body rotation orbits at constant radius", "[trajectories][analytical]")
{
    const int    nr = 41, nth = 41, nz = 5;
    const double dx = 500.0, dy = 500.0, dz = 500.0; // domain 20 km square
    const double xc = 10250.0, yc = 10250.0;         // scalar centre (index 20)
    const double omega = 1.0e-3;                     // rad/s
    Field3D      u(nr, nth, nz, 0.0f), v(nr, nth, nz, 0.0f), w(nr, nth, nz, 0.0f), zeta(nr, nth, nz, 0.0f),
        rho(nr, nth, nz, 1.0f), pres(nr, nth, nz, 1.0e5f);
    // Solid-body rotation about (xc, yc): vx = -omega*(y-yc), vy = omega*(x-xc).
    // Both wind components are linear in space, so the trilinear sampler is
    // exact and the only error is RK2's O((omega*dt)^4) per-step radius drift.
    for (int i = 0; i < nr; ++i)
        for (int j = 0; j < nth; ++j)
            for (int k = 0; k < nz; ++k)
            {
                u[i][j][k] = static_cast<float>(-omega * ((static_cast<double>(j) + 0.5) * dy - yc));
                v[i][j][k] = static_cast<float>(omega * ((static_cast<double>(i) + 0.5) * dx - xc));
            }

    tmv::TrajectorySet ts;
    ts.configure(single_parcel(xc + 5000.0, yc, 1000.0), CoordinateSystem::Cartesian, false, nr, nth, nz, dx, dy, dz);
    ts.seed(0.0, u, v, w, rho, pres, zeta);
    REQUIRE(ts.count() == 1);
    const double dt = 10.0;
    const int    N  = 10; // T = 100 s -> angle ~ omega*T = 0.1 rad
    for (int s = 0; s < N; ++s)
        ts.advance((s + 1) * dt, dt, u, v, w, rho, pres, zeta);

    const auto& p = ts.parcels()[0];
    REQUIRE(p.active);
    const double r   = std::hypot(p.x - xc, p.y - yc);
    double       ang = std::atan2(p.y - yc, p.x - xc);
    if (ang < 0.0)
        ang += kTwoPi;
    CHECK(r == Approx(5000.0).epsilon(0.02));  // radius conserved (~2%)
    CHECK(ang == Approx(0.1).epsilon(0.1));    // ~ omega * T (counter-clockwise)
    CHECK(p.z == Approx(1000.0).margin(1e-6)); // w = 0 -> z unchanged
}

TEST_CASE("Parcel leaving the domain is deactivated", "[trajectories][analytical]")
{
    const int    nr = 10, nth = 10, nz = 5;
    const double dx = 1000.0, dy = 1000.0, dz = 500.0;
    Field3D      u(nr, nth, nz, 100.0f), v(nr, nth, nz, 0.0f), w(nr, nth, nz, 0.0f), zeta(nr, nth, nz, 0.0f),
        rho(nr, nth, nz, 1.0f), pres(nr, nth, nz, 1.0e5f);

    tmv::TrajectorySet ts;
    ts.configure(single_parcel(5000.0, 5000.0, 1000.0), CoordinateSystem::Cartesian, false, nr, nth, nz, dx, dy, dz);
    ts.seed(0.0, u, v, w, rho, pres, zeta);
    const double dt = 10.0; // 100 m/s -> 1000 m/step; physical upper face is x=9000 m
    for (int s = 0; s < 10; ++s)
        ts.advance((s + 1) * dt, dt, u, v, w, rho, pres, zeta);

    CHECK_FALSE(ts.parcels()[0].active);
}

// ---------------------------------------------------------------------------
// Vorticity budget (the tornadogenesis-mechanism diagnostic)
// ---------------------------------------------------------------------------

namespace
{
// Build a C-grid flow with uniform analytic gradients:
//   v = beta * x      -> dv/dx = beta
//   u = alpha * y     -> du/dy = alpha   => omega_z = beta - alpha
//   w = Wz * z        -> dw/dz = Wz      => stretching = omega_z * Wz
//   rho = 1 + Rx * x  -> drho/dx = Rx
//   p   = p0 + Py * y -> dp/dy   = Py    => baroclinic = Rx*Py / rho^2
// All other gradients (and hence omega_x, omega_y, tilting) vanish.
struct BudgetFlow
{
    static constexpr double alpha = 1.0e-5;
    static constexpr double beta  = 2.0e-5;
    static constexpr double Wz    = 1.0e-3;
    static constexpr double Rx    = 1.0e-6;
    static constexpr double Py    = 0.1;
    static constexpr double p0    = 1.0e5;

    static void fill(Field3D& u,
                     Field3D& v,
                     Field3D& w,
                     Field3D& rho,
                     Field3D& p,
                     int      nr,
                     int      nth,
                     int      nz,
                     double   dx,
                     double   dy,
                     double   dz)
    {
        for (int i = 0; i < nr; ++i)
            for (int j = 0; j < nth; ++j)
                for (int k = 0; k < nz; ++k)
                {
                    u[i][j][k]   = static_cast<float>(alpha * (j * dy));
                    v[i][j][k]   = static_cast<float>(beta * (i * dx));
                    w[i][j][k]   = static_cast<float>(Wz * (k * dz));
                    rho[i][j][k] = static_cast<float>(1.0 + Rx * (i * dx));
                    p[i][j][k]   = static_cast<float>(p0 + Py * (j * dy));
                }
    }
};
} // namespace

TEST_CASE("Cartesian C-grid budget kernel matches analytic gradients", "[trajectories][budget]")
{
    const int    nr = 12, nth = 12, nz = 8;
    const double dx = 1000.0, dy = 1000.0, dz = 500.0;
    Field3D      u(nr, nth, nz, 0.0f), v(nr, nth, nz, 0.0f), w(nr, nth, nz, 0.0f), rho(nr, nth, nz, 1.0f),
        p(nr, nth, nz, 1.0e5f);
    BudgetFlow::fill(u, v, w, rho, p, nr, nth, nz, dx, dy, dz);

    const int                       i = 6, j = 6, k = 4;
    const tmv::VorticityBudgetPoint b = tmv::cartesian_cgrid_vorticity_budget_at(u, v, w, rho, p, i, j, k, dx, dy, dz);

    const double omega_z = BudgetFlow::beta - BudgetFlow::alpha;
    const double rho_val = 1.0 + BudgetFlow::Rx * (i * dx);
    CHECK(b.omega_z == Approx(omega_z).epsilon(1e-3));
    CHECK(b.stretching == Approx(omega_z * BudgetFlow::Wz).epsilon(1e-2));
    CHECK(b.baroclinic == Approx(BudgetFlow::Rx * BudgetFlow::Py / (rho_val * rho_val)).epsilon(1e-2));
    CHECK(b.tilting == Approx(0.0).margin(1e-12));
    CHECK(b.omega_x == Approx(0.0).margin(1e-9));
    CHECK(b.omega_y == Approx(0.0).margin(1e-9));
}

TEST_CASE("Cartesian collocated budget kernel matches analytic gradients",
          "[trajectories][budget][agrid]")
{
    const int nr = 12, nth = 12, nz = 8;
    const double dx = 1000.0, dy = 1000.0, dz = 500.0;
    Field3D u(nr, nth, nz, 0.0f), v(nr, nth, nz, 0.0f),
        w(nr, nth, nz, 0.0f), rho(nr, nth, nz, 1.0f),
        p(nr, nth, nz, 1.0e5f);
    BudgetFlow::fill(u, v, w, rho, p, nr, nth, nz, dx, dy, dz);

    constexpr int i = 6, j = 6, k = 4;
    const auto budget = tmv::cartesian_collocated_vorticity_budget_at(
        u, v, w, rho, p, i, j, k, dx, dy, dz);
    const double omega_z = BudgetFlow::beta - BudgetFlow::alpha;
    const double rho_value = 1.0 + BudgetFlow::Rx * (i * dx);
    CHECK(budget.omega_z == Approx(omega_z).epsilon(1e-3));
    CHECK(budget.stretching == Approx(omega_z * BudgetFlow::Wz).epsilon(1e-2));
    CHECK(budget.baroclinic == Approx(
        BudgetFlow::Rx * BudgetFlow::Py / (rho_value * rho_value)).epsilon(1e-2));
    CHECK(budget.tilting == Approx(0.0).margin(1e-12));
    CHECK(budget.omega_x == Approx(0.0).margin(1e-9));
    CHECK(budget.omega_y == Approx(0.0).margin(1e-9));
}

TEST_CASE("Trajectory records a fresh, finite budget on the Cartesian C-grid", "[trajectories][budget]")
{
    const int    nr = 16, nth = 16, nz = 12;
    const double dx = 1000.0, dy = 1000.0, dz = 500.0;
    Field3D      u(nr, nth, nz, 0.0f), v(nr, nth, nz, 0.0f), w(nr, nth, nz, 0.0f), rho(nr, nth, nz, 1.0f),
        p(nr, nth, nz, 1.0e5f), zeta(nr, nth, nz, 0.0f);
    BudgetFlow::fill(u, v, w, rho, p, nr, nth, nz, dx, dy, dz);

    tmv::TrajectorySet ts;
    ts.configure(single_parcel(6000.0, 6000.0, 2000.0),
                 CoordinateSystem::Cartesian,
                 /*staggered=*/true,
                 nr,
                 nth,
                 nz,
                 dx,
                 dy,
                 dz);
    ts.seed(0.0, u, v, w, rho, p, zeta);
    const double dt = 10.0;
    for (int s = 0; s < 5; ++s)
        ts.advance((s + 1) * dt, dt, u, v, w, rho, p, zeta);

    const double omega_z = BudgetFlow::beta - BudgetFlow::alpha;
    const auto&  parcel  = ts.parcels()[0];
    REQUIRE(parcel.path.size() == 6u); // seed + 5 steps
    for (const auto& smp : parcel.path)
    {
        // Uniform gradients -> the budget is the same at every interior cell, so
        // the parcel records a constant, finite budget independent of position.
        CHECK(std::isfinite(smp.tilting));
        CHECK(std::isfinite(smp.stretching));
        CHECK(std::isfinite(smp.baroclinic));
        CHECK(smp.zeta == Approx(omega_z).epsilon(1e-2));
        CHECK(smp.stretching == Approx(omega_z * BudgetFlow::Wz).epsilon(1e-2));
    }
}

TEST_CASE("Trajectory records a fresh finite budget on the collocated A-grid",
          "[trajectories][budget][agrid]")
{
    const int nr = 16, nth = 16, nz = 12;
    const double dx = 1000.0, dy = 1000.0, dz = 500.0;
    Field3D u(nr, nth, nz, 0.0f), v(nr, nth, nz, 0.0f),
        w(nr, nth, nz, 0.0f), rho(nr, nth, nz, 1.0f),
        p(nr, nth, nz, 1.0e5f), zeta(nr, nth, nz, 0.0f);
    BudgetFlow::fill(u, v, w, rho, p, nr, nth, nz, dx, dy, dz);

    tmv::TrajectorySet trajectories;
    trajectories.configure(
        single_parcel(6000.0, 6000.0, 2000.0),
        CoordinateSystem::Cartesian, false,
        nr, nth, nz, dx, dy, dz);
    trajectories.seed(0.0, u, v, w, rho, p, zeta);
    for (int step = 0; step < 5; ++step)
        trajectories.advance(
            (step + 1) * 10.0, 10.0, u, v, w, rho, p, zeta);

    const double omega_z = BudgetFlow::beta - BudgetFlow::alpha;
    REQUIRE(trajectories.parcels()[0].path.size() == 6u);
    for (const auto& sample : trajectories.parcels()[0].path)
    {
        CHECK(std::isfinite(sample.tilting));
        CHECK(std::isfinite(sample.stretching));
        CHECK(std::isfinite(sample.baroclinic));
        CHECK(sample.zeta == Approx(omega_z).epsilon(1e-2));
        CHECK(sample.stretching == Approx(
            omega_z * BudgetFlow::Wz).epsilon(1e-2));
    }
}

TEST_CASE("Trajectory budget is interpolated continuously between scalar centres",
          "[trajectories][budget][interpolation]")
{
    const int nr = 12, nth = 12, nz = 8;
    const double dx = 1000.0, dy = 1000.0, dz = 500.0;
    const double v_curvature = 1.0e-9;
    Field3D u(nr, nth, nz, 0.0f), v(nr, nth, nz, 0.0f),
        w(nr, nth, nz, 0.0f), rho(nr, nth, nz, 1.0f),
        pressure(nr, nth, nz, 90000.0f), zeta(nr, nth, nz, 0.0f);
    for (int i = 0; i < nr; ++i)
    {
        const double x = tmv::grid_coordinates::scalar_center(i, dx);
        const float value = static_cast<float>(0.5 * v_curvature * x * x);
        for (int j = 0; j < nth; ++j)
            for (int k = 0; k < nz; ++k)
                v[i][j][k] = value;
    }

    constexpr double parcel_x = 5750.0;
    tmv::TrajectorySet trajectories;
    trajectories.configure(
        single_parcel(parcel_x, 5750.0, 2250.0),
        CoordinateSystem::Cartesian, true,
        nr, nth, nz, dx, dy, dz);
    trajectories.seed(0.0, u, v, w, rho, pressure, zeta);

    REQUIRE(trajectories.count() == 1);
    REQUIRE(trajectories.parcels()[0].path.size() == 1);
    const auto& sample = trajectories.parcels()[0].path.front();
    CHECK(sample.zeta
          == Approx(v_curvature * parcel_x).epsilon(2.0e-4));
    CHECK(sample.tilting == Approx(0.0f).margin(1.0e-12));
    CHECK(sample.stretching == Approx(0.0f).margin(1.0e-12));
    CHECK(sample.baroclinic == Approx(0.0f).margin(1.0e-12));
}

TEST_CASE("Online budget is NaN where the coordinate equation is unsupported",
          "[trajectories][budget][unsupported]")
{
    const int    nr = 12, nth = 12, nz = 8;
    const double dx = 1000.0, dy = 1000.0, dz = 500.0;
    Field3D      u(nr, nth, nz, 2.0f), v(nr, nth, nz, 1.0f), w(nr, nth, nz, 0.5f), rho(nr, nth, nz, 1.0f),
        p(nr, nth, nz, 1.0e5f), zeta(nr, nth, nz, 0.0f);
    BudgetFlow::fill(u, v, w, rho, p, nr, nth, nz, dx, dy, dz);

    tmv::TrajectorySet ts;
    // Preserve the explicit fallback for an unknown enum value. Cartesian
    // C-grid and collocated A-grid both have online equations.
    ts.configure(single_parcel(6000.0, 6000.0, 2000.0),
                 static_cast<CoordinateSystem>(-1),
                 /*staggered=*/false,
                 nr,
                 nth,
                 nz,
                 dx,
                 dy,
                 dz);
    ts.seed(0.0, u, v, w, rho, p, zeta);
    const double dt = 10.0;
    for (int s = 0; s < 3; ++s)
        ts.advance((s + 1) * dt, dt, u, v, w, rho, p, zeta);

    const auto& parcel = ts.parcels()[0];
    for (const auto& smp : parcel.path)
    {
        CHECK(std::isnan(smp.tilting));
        CHECK(std::isnan(smp.stretching));
        CHECK(std::isnan(smp.baroclinic));
    }
    // zeta falls back to the (here zero) vort_z field sample, not the kernel.
    CHECK(parcel.path.back().zeta == Approx(0.0).margin(1e-9));
}

TEST_CASE("trajectory time interpolation converges for accelerating winds",
          "[trajectories][analytical][time_interpolation]")
{
    const int nr = 8, nth = 8, nz = 6;
    Field3D u(nr, nth, nz, 0.0f), zero(nr, nth, nz, 0.0f),
        rho(nr, nth, nz, 1.0f), pressure(nr, nth, nz, 1.0e5f);
    double previous_error = 0.0;
    for (const int steps : {1, 2, 4, 8})
    {
        tmv::TrajectorySet parcels;
        parcels.configure(single_parcel(3000.0, 3000.0, 1500.0),
            CoordinateSystem::Cartesian, false, nr, nth, nz, 1000.0, 1000.0, 500.0);
        u.fill(0.0f);
        parcels.seed(0.0, u, zero, zero, rho, pressure, zero);
        const double dt = 1.0 / steps;
        for (int step = 0; step < steps; ++step)
        {
            const double time = step * dt;
            u.fill(static_cast<float>(time * time));
            parcels.prepare_advance(time, dt, u, zero, zero);
            REQUIRE(parcels.parcels()[0].path.size() == static_cast<std::size_t>(step + 1));
            const double next = time + dt;
            u.fill(static_cast<float>(next * next));
            parcels.finish_advance(next, dt, u, zero, zero, rho, pressure, zero);
        }
        const double error = std::abs(parcels.parcels()[0].x - (3000.0 + 1.0 / 3.0));
        REQUIRE(error == Approx(dt * dt / 6.0).margin(1e-10));
        if (previous_error > 0.0) REQUIRE(previous_error / error == Approx(4.0).epsilon(1e-6));
        previous_error = error;
    }
}
