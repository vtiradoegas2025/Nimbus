#include "catch2/catch.hpp"
#include "core/runtime/grid_state.hpp"
#include "radar/factory.hpp"
#include "radar/radar_base.hpp"
#include "radar/schemes/velocity/velocity.hpp"

#include <cmath>
#include <limits>

namespace
{

void configure_cartesian_grid(int nx, int ny, int nz,
                              double dx, double dy, double dz_value,
                              StaggerType staggering = StaggerType::Collocated)
{
    ::NR = nx;
    ::NTH = ny;
    ::NZ = nz;
    ::dr = dx;
    ::dtheta = dy;
    ::dz = dz_value;
    ::global_grid_geometry.initialize(nx, ny, nz, dx, dz_value, dy,
                                      CoordinateSystem::Cartesian, staggering);
}

RadarOut compute_velocity(const RadarConfig& config,
                          const Field3D& u,
                          const Field3D& v,
                          const Field3D& w,
                          const Field3D* qr = nullptr,
                          StaggerType staggering = StaggerType::Collocated)
{
    RadarStateView state;
    state.NR = u.size_r();
    state.NTH = u.size_th();
    state.NZ = u.size_z();
    state.staggering = staggering;
    state.geometry = &::global_grid_geometry;
    state.u = &u;
    state.v = &v;
    state.w = &w;
    state.qr = qr;

    VelocityScheme scheme;
    scheme.initialize(config, state.NR, state.NTH, state.NZ);

    RadarOut output;
    scheme.compute(config, state, output);
    return output;
}

double analytic_radial_velocity(double u, double v, double w,
                                double radar_x, double radar_y, double radar_z,
                                int i, int j, int k)
{
    const double dx = ::global_grid_geometry.r[static_cast<std::size_t>(i)] - radar_x;
    const double dy = ::global_grid_geometry.theta[static_cast<std::size_t>(j)] - radar_y;
    const double dz_local = ::global_grid_geometry.z[static_cast<std::size_t>(k)] - radar_z;
    const double range = std::sqrt(dx * dx + dy * dy + dz_local * dz_local);
    return (u * dx + v * dy + w * dz_local) / range;
}

} // namespace

TEST_CASE("Radar descriptors separate required fields from products",
          "[radar][descriptor]")
{
    const auto& descriptors = get_radar_scheme_descriptors();
    REQUIRE(descriptors.size() == 3);
    for (const auto& descriptor : descriptors)
    {
        REQUIRE(descriptor.capabilities.declared);
        REQUIRE_FALSE(descriptor.capabilities.required_state_fields.empty());
        REQUIRE_FALSE(descriptor.capabilities.produced_state_fields.empty());
        REQUIRE(descriptor.capabilities.gpu_implementation_coverage
                == tmv::SchemeCoverage::None);
    }

    const auto* velocity = tmv::find_scheme_descriptor(descriptors, "vr");
    REQUIRE(velocity != nullptr);
    REQUIRE(velocity->id == "velocity");
    REQUIRE(velocity->capabilities.produced_state_fields
            == std::vector<std::string>{"Vr"});
}

TEST_CASE("Radar velocity reconstructs C-grid face winds to scalar centers",
          "[radar][velocity][cartesian][cgrid]")
{
    constexpr int nx = 5;
    constexpr int ny = 5;
    constexpr int nz = 4;
    constexpr double dx = 700.0;
    constexpr double dy = 1300.0;
    constexpr double dz_value = 400.0;
    configure_cartesian_grid(nx, ny, nz, dx, dy, dz_value, StaggerType::CGrid);

    Field3D u(nx, ny, nz, 0.0f);
    Field3D v(nx, ny, nz, 0.0f);
    Field3D w(nx, ny, nz, 0.0f);
    for (int i = 0; i < nx; ++i)
    {
        for (int j = 0; j < ny; ++j)
        {
            for (int k = 0; k < nz; ++k)
            {
                u(i, j, k) = 2.0f + 3.0f * i + 0.4f * j + 0.1f * k;
                v(i, j, k) = -1.0f + 0.2f * i + 4.0f * j + 0.1f * k;
                w(i, j, k) = 1.0f + 0.2f * i + 0.3f * j + 5.0f * k;
            }
        }
    }

    RadarConfig config;
    config.sampling = "point";
    config.apply_scatterer_correction = false;
    config.radar_x = -0.5 * dx;
    config.radar_y = -0.5 * dy;
    config.radar_z = -0.5 * dz_value;

    constexpr int i = 2;
    constexpr int j = 2;
    constexpr int k = 2;
    const double u_center = 0.5 * (u(i - 1, j, k) + u(i, j, k));
    const double v_center = 0.5 * (v(i, j - 1, k) + v(i, j, k));
    const double w_center = 0.5 * (w(i, j, k - 1) + w(i, j, k));

    SECTION("C-grid input uses adjacent faces")
    {
        const RadarOut output =
            compute_velocity(config, u, v, w, nullptr, StaggerType::CGrid);
        const double expected = analytic_radial_velocity(
            u_center, v_center, w_center,
            config.radar_x, config.radar_y, config.radar_z, i, j, k);
        const double raw_face_projection = analytic_radial_velocity(
            u(i, j, k), v(i, j, k), w(i, j, k),
            config.radar_x, config.radar_y, config.radar_z, i, j, k);

        REQUIRE(output.Vr(i, j, k) == Approx(expected).margin(1.0e-5));
        REQUIRE(std::abs(expected - raw_face_projection) > 0.1);
    }

    SECTION("collocated input remains unchanged")
    {
        const RadarOut output =
            compute_velocity(config, u, v, w, nullptr, StaggerType::Collocated);
        const double expected = analytic_radial_velocity(
            u(i, j, k), v(i, j, k), w(i, j, k),
            config.radar_x, config.radar_y, config.radar_z, i, j, k);
        REQUIRE(output.Vr(i, j, k) == Approx(expected).margin(1.0e-5));
    }

    SECTION("beam-volume sampling averages reconstructed center winds")
    {
        config.sampling = "beam_volume";
        const RadarOut output =
            compute_velocity(config, u, v, w, nullptr, StaggerType::CGrid);
        // These face fields are linear, so the symmetric 3x3x3 average of
        // their reconstructed center values equals the target-center value.
        const double expected = analytic_radial_velocity(
            u_center, v_center, w_center,
            config.radar_x, config.radar_y, config.radar_z, i, j, k);
        REQUIRE(output.Vr(i, j, k) == Approx(expected).margin(1.0e-5));
    }
}

TEST_CASE("Radar C-grid reconstruction matches output conventions at physical edges",
          "[radar][velocity][cartesian][cgrid][boundary]")
{
    constexpr int nx = 3;
    constexpr int ny = 3;
    constexpr int nz = 3;
    constexpr double dx = 600.0;
    constexpr double dy = 1100.0;
    constexpr double dz_value = 300.0;
    configure_cartesian_grid(nx, ny, nz, dx, dy, dz_value, StaggerType::CGrid);

    Field3D u(nx, ny, nz, 1.0f);
    Field3D v(nx, ny, nz, 2.0f);
    Field3D w(nx, ny, nz, 3.0f);
    // The first physical scalar cell is (1,1,0); indices i/j=0 hold its
    // west/south boundary faces on the C-grid.
    u(0, 1, 0) = 12.0f;
    u(1, 1, 0) = 4.0f;
    v(1, 0, 0) = -7.0f;
    v(1, 1, 0) = 3.0f;
    w(1, 1, 0) = 8.0f;

    RadarConfig config;
    config.sampling = "point";
    config.apply_scatterer_correction = false;
    config.radar_x = -dx;
    config.radar_y = -dy;
    config.radar_z = -dz_value;

    const RadarOut output =
        compute_velocity(config, u, v, w, nullptr, StaggerType::CGrid);
    // The maintained west/south faces participate in the physical cell-center
    // reconstruction. At the surface, w averages with rigid-ground w=0.
    const double expected = analytic_radial_velocity(
        8.0, -2.0, 4.0,
        config.radar_x, config.radar_y, config.radar_z, 1, 1, 0);
    REQUIRE(output.Vr(1, 1, 0) == Approx(expected).margin(1.0e-5));

    RadarStateView full_state;
    full_state.NR = nx;
    full_state.NTH = ny;
    full_state.NZ = nz;
    full_state.staggering = StaggerType::CGrid;
    full_state.geometry = &::global_grid_geometry;
    full_state.u = &u;
    full_state.v = &v;
    full_state.w = &w;
    RadarStateView point_state;
    RadarGeometry::sample_state_point(full_state, 1, 1, 0, point_state);
    REQUIRE(point_state.staggering == StaggerType::Collocated);
    REQUIRE((*point_state.u)(0, 0, 0) == Approx(8.0f));
    REQUIRE((*point_state.v)(0, 0, 0) == Approx(-2.0f));
    REQUIRE((*point_state.w)(0, 0, 0) == Approx(4.0f));
}

TEST_CASE("Radar velocity projects Cartesian wind onto Cartesian line of sight",
          "[radar][velocity][cartesian]")
{
    constexpr int nx = 5;
    constexpr int ny = 5;
    constexpr int nz = 5;
    constexpr double dx = 700.0;
    constexpr double dy = 1300.0;
    constexpr double dz_value = 400.0;
    constexpr float u_value = 7.0f;
    constexpr float v_value = -4.0f;
    constexpr float w_value = 2.0f;

    configure_cartesian_grid(nx, ny, nz, dx, dy, dz_value);
    const Field3D u(nx, ny, nz, u_value);
    const Field3D v(nx, ny, nz, v_value);
    const Field3D w(nx, ny, nz, w_value);

    RadarConfig config;
    config.sampling = "point";
    config.apply_scatterer_correction = false;

    SECTION("radar at domain corner")
    {
        config.radar_x = 0.0;
        config.radar_y = 0.0;
        config.radar_z = 0.0;
        const int i = 1;
        const int j = 2;
        const int k = 3;

        const RadarOut output = compute_velocity(config, u, v, w);
        REQUIRE(output.Vr(i, j, k) ==
                Approx(analytic_radial_velocity(u_value, v_value, w_value,
                                                config.radar_x, config.radar_y, config.radar_z,
                                                i, j, k)).margin(1.0e-5));
    }

    SECTION("radar at domain center")
    {
        config.radar_x = 2.0 * dx;
        config.radar_y = 2.0 * dy;
        config.radar_z = 2.0 * dz_value;
        const int i = 4;
        const int j = 1;
        const int k = 3;

        const RadarOut output = compute_velocity(config, u, v, w);
        REQUIRE(output.Vr(i, j, k) ==
                Approx(analytic_radial_velocity(u_value, v_value, w_value,
                                                config.radar_x, config.radar_y, config.radar_z,
                                                i, j, k)).margin(1.0e-5));
    }

    SECTION("radar outside domain")
    {
        config.radar_x = -dx;
        config.radar_y = -2.0 * dy;
        config.radar_z = -dz_value;
        const int i = 0;
        const int j = 0;
        const int k = 0;

        const RadarOut output = compute_velocity(config, u, v, w);
        REQUIRE(output.Vr(i, j, k) ==
                Approx(analytic_radial_velocity(u_value, v_value, w_value,
                                                config.radar_x, config.radar_y, config.radar_z,
                                                i, j, k)).margin(1.0e-5));
    }
}

TEST_CASE("Radar velocity is invariant to a shared Earth-frame translation",
          "[radar][velocity][cartesian][domain-motion]")
{
    constexpr int nx = 4;
    constexpr int ny = 4;
    constexpr int nz = 3;
    configure_cartesian_grid(nx, ny, nz, 600.0, 900.0, 350.0);

    const Field3D u(nx, ny, nz, 11.0f);
    const Field3D v(nx, ny, nz, -3.0f);
    const Field3D w(nx, ny, nz, 1.5f);
    RadarStateView state;
    state.NR = nx;
    state.NTH = ny;
    state.NZ = nz;
    state.geometry = &global_grid_geometry;
    state.u = &u;
    state.v = &v;
    state.w = &w;

    RadarConfig config;
    config.sampling = "point";
    config.apply_scatterer_correction = false;
    config.radar_x = -800.0;
    config.radar_y = 250.0;
    config.radar_z = 100.0;

    VelocityScheme scheme;
    scheme.initialize(config, nx, ny, nz);
    RadarOut baseline;
    scheme.compute(config, state, baseline);

    constexpr double translation_x = 18500.0;
    constexpr double translation_y = -7200.0;
    state.grid_origin_earth_x_m = translation_x;
    state.grid_origin_earth_y_m = translation_y;
    config.radar_x += translation_x;
    config.radar_y += translation_y;
    scheme.initialize(config, nx, ny, nz);
    RadarOut translated;
    scheme.compute(config, state, translated);

    for (int i = 0; i < nx; ++i)
        for (int j = 0; j < ny; ++j)
            for (int k = 0; k < nz; ++k)
                REQUIRE(translated.Vr(i, j, k)
                        == Approx(baseline.Vr(i, j, k)).margin(1.0e-6));

    RadarStateView point;
    RadarGeometry::sample_state_point(state, 2, 1, 1, point);
    REQUIRE(point.grid_origin_earth_x_m == translation_x);
    REQUIRE(point.grid_origin_earth_y_m == translation_y);

    state.grid_origin_earth_x_m =
        std::numeric_limits<double>::quiet_NaN();
    REQUIRE_THROWS_AS(
        scheme.compute(config, state, translated), std::invalid_argument);
}

TEST_CASE("Radar velocity uses caller-owned terrain height geometry",
          "[radar][velocity][terrain][state]")
{
    constexpr int nx = 3, ny = 3, nz = 3;
    GridGeometry local_geometry;
    local_geometry.initialize(
        nx, ny, nz, 100.0, 100.0, 100.0,
        CoordinateSystem::Cartesian, StaggerType::Collocated);
    TerrainMetrics3D terrain;
    terrain.z.resize(nx, ny, nz, 0.0);
    terrain.z_face.resize(nx, ny, nz + 1, 0.0);
    for (int i = 0; i < nx; ++i)
        for (int j = 0; j < ny; ++j)
        {
            for (int k = 0; k <= nz; ++k)
                terrain.z_face(i, j, k) = 1300.0 + 100.0 * k;
            for (int k = 0; k < nz; ++k)
                terrain.z(i, j, k) = 1350.0 + 100.0 * k;
        }

    // Poison the process-global geometry: a state-owned diagnostic must not
    // borrow these coordinates.
    configure_cartesian_grid(
        nx, ny, nz, 900.0, 800.0, 25.0,
        StaggerType::Collocated);

    Field3D u(nx, ny, nz, 10.0f);
    Field3D v(nx, ny, nz, 0.0f);
    Field3D w(nx, ny, nz, 0.0f);
    RadarStateView state;
    state.NR = nx;
    state.NTH = ny;
    state.NZ = nz;
    state.geometry = &local_geometry;
    state.terrain_metrics = &terrain;
    state.u = &u;
    state.v = &v;
    state.w = &w;

    RadarConfig config;
    config.sampling = "point";
    config.radar_x = 0.0;
    config.radar_y = local_geometry.theta[1];
    config.radar_z = 0.0;
    VelocityScheme scheme;
    scheme.initialize(config, nx, ny, nz);
    RadarOut output;
    scheme.compute(config, state, output);

    const double x = local_geometry.r[1];
    const double z = terrain.z(1, 1, 1);
    const double expected = 10.0 * x / std::sqrt(x * x + z * z);
    REQUIRE(output.Vr(1, 1, 1) == Approx(expected).margin(1.0e-6));

    state.geometry = nullptr;
    REQUIRE_THROWS_AS(
        scheme.compute(config, state, output), std::invalid_argument);
}

TEST_CASE("Radar velocity validates without clipping accepted CPU winds",
          "[radar][velocity][strict][cpu]")
{
    constexpr int n = 3;
    configure_cartesian_grid(
        n, n, n, 100.0, 100.0, 100.0,
        StaggerType::Collocated);
    Field3D u(n, n, n, 180.0f);
    Field3D v(n, n, n, 180.0f);
    Field3D w(n, n, n, 120.0f);
    RadarStateView state;
    state.NR = n;
    state.NTH = n;
    state.NZ = n;
    state.geometry = &global_grid_geometry;
    state.u = &u;
    state.v = &v;
    state.w = &w;

    RadarConfig config;
    config.sampling = "point";
    config.apply_scatterer_correction = false;
    config.radar_x = global_grid_geometry.r[1] - 180.0;
    config.radar_y = global_grid_geometry.theta[1] - 180.0;
    config.radar_z = global_grid_geometry.z[1] - 120.0;
    VelocityScheme scheme;
    scheme.initialize(config, n, n, n);
    RadarOut output;
    scheme.compute(config, state, output);
    const double speed = std::sqrt(180.0 * 180.0 + 180.0 * 180.0
                                   + 120.0 * 120.0);
    REQUIRE(speed > 250.0);
    REQUIRE(output.Vr(1, 1, 1) == Approx(speed).margin(1.0e-5));

    const float published = output.Vr(1, 1, 1);
    u(1, 1, 1) = std::numeric_limits<float>::quiet_NaN();
    REQUIRE_THROWS_AS(
        scheme.compute(config, state, output), std::invalid_argument);
    REQUIRE(output.Vr(1, 1, 1) == published);
}

TEST_CASE("Radar beam-volume sampling truncates at Cartesian edges without wrapping",
          "[radar][velocity][cartesian][sampling]")
{
    constexpr int nx = 3;
    constexpr int ny = 4;
    constexpr int nz = 3;
    constexpr double dx = 100.0;
    constexpr double dy = 250.0;
    constexpr double dz_value = 50.0;
    configure_cartesian_grid(nx, ny, nz, dx, dy, dz_value);

    Field3D u(nx, ny, nz, 0.0f);
    Field3D v(nx, ny, nz, 0.0f);
    Field3D w(nx, ny, nz, 0.0f);
    for (int i = 0; i < nx; ++i)
    {
        for (int k = 0; k < nz; ++k)
        {
            u(i, 0, k) = 3.0f;
            u(i, 1, k) = 9.0f;
            u(i, 2, k) = 12.0f;
            u(i, 3, k) = 99.0f;
        }
    }

    RadarConfig config;
    config.sampling = "beam_volume";
    config.apply_scatterer_correction = false;
    config.radar_x = 0.0;
    config.radar_y = global_grid_geometry.theta[1];
    config.radar_z = global_grid_geometry.z[1];

    const RadarOut output = compute_velocity(config, u, v, w);

    // Collocated j=0 and j=3 are closure storage. At the first physical row
    // j=1, the footprint contains only physical j={1,2}; poisoned closures do
    // not enter the beam average. The target beam points along +x.
    REQUIRE(output.Vr(1, 1, 1) == Approx(10.5f).margin(1.0e-6));

    RadarStateView state;
    state.NR = nx;
    state.NTH = ny;
    state.NZ = nz;
    state.u = &u;
    const Field3D density(nx, ny, nz, 1.2f);
    state.rho = &density;
    RadarStateView point;
    RadarGeometry::sample_state_point(state, 1, -1, 1, point);
    REQUIRE(point.u != nullptr);
    REQUIRE(point.rho != nullptr);
    REQUIRE((*point.u)(0, 0, 0) == Approx(9.0f));
    REQUIRE((*point.rho)(0, 0, 0) == Approx(1.2f));
}

TEST_CASE("Radar scatterer fall speed uses the Cartesian beam elevation",
          "[radar][velocity][cartesian][scatterer]")
{
    constexpr int nx = 3;
    constexpr int ny = 3;
    constexpr int nz = 3;
    constexpr double dx = 600.0;
    constexpr double dy = 1100.0;
    constexpr double dz_value = 300.0;
    configure_cartesian_grid(nx, ny, nz, dx, dy, dz_value);

    const Field3D u(nx, ny, nz, 0.0f);
    const Field3D v(nx, ny, nz, 0.0f);
    const Field3D w(nx, ny, nz, 0.0f);
    const Field3D qr(nx, ny, nz, 1.0e-3f);

    RadarConfig config;
    config.sampling = "point";
    config.apply_scatterer_correction = true;
    config.radar_x = 0.0;
    config.radar_y = 0.0;
    config.radar_z = 0.0;

    const RadarOut output = compute_velocity(config, u, v, w, &qr);
    const double range = std::sqrt(dx * dx + dy * dy + dz_value * dz_value);
    const double expected = -9.0 * dz_value / range;
    REQUIRE(output.Vr(1, 1, 1) == Approx(expected).margin(1.0e-5));
}
