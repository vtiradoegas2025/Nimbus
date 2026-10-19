#include "catch2/catch.hpp"

#include "core/infra/physical_constants.hpp"
#include "dynamics/eos_pressure.hpp"
#include "terrain/collocated_mapped_timestep.hpp"
#include "terrain/conservative_transport.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace
{

constexpr int nx = 8;
constexpr int ny = 7;
constexpr int nz = 7;
constexpr double dx = 400.0;
constexpr double dy = 500.0;
constexpr double dzeta = 600.0;
constexpr double ztop = nz * dzeta;

TerrainMetrics3D mapped_geometry()
{
    TerrainMetrics3D metrics;
    metrics.dx = dx;
    metrics.dy = dy;
    metrics.dzeta = dzeta;
    metrics.z.resize(nx, ny, nz, 0.0);
    metrics.J.resize(nx, ny, nz, 0.0);
    metrics.mx.resize(nx, ny, nz, 0.0);
    metrics.my.resize(nx, ny, nz, 0.0);
    metrics.zeta.resize(nx, ny, nz, 0.0);
    metrics.z_face.resize(nx, ny, nz + 1, 0.0);
    metrics.Jx_face.resize(nx + 1, ny, nz, 1.0);
    metrics.Jy_face.resize(nx, ny + 1, nz, 1.0);
    metrics.Gx_face.resize(nx, ny, nz + 1, 0.0);
    metrics.Gy_face.resize(nx, ny, nz + 1, 0.0);

    const auto height = [](int i, int j)
    {
        const double x = (static_cast<double>(i) - 3.5) * dx;
        const double y = (static_cast<double>(j) - 3.0) * dy;
        return 240.0 * std::exp(
            -(x * x / (1500.0 * 1500.0)
              + y * y / (1700.0 * 1700.0)));
    };
    const auto slope_x = [&](int i, int j)
    {
        const int west = std::max(0, i - 1);
        const int east = std::min(nx - 1, i + 1);
        return (height(east, j) - height(west, j))
            / (static_cast<double>(east - west) * dx);
    };
    const auto slope_y = [&](int i, int j)
    {
        const int south = std::max(0, j - 1);
        const int north = std::min(ny - 1, j + 1);
        return (height(i, north) - height(i, south))
            / (static_cast<double>(north - south) * dy);
    };

    for (int i = 0; i < nx; ++i)
        for (int j = 0; j < ny; ++j)
        {
            const double h = height(i, j);
            const double jacobian = 1.0 - h / ztop;
            for (int k = 0; k < nz; ++k)
            {
                const double computational = (k + 0.5) * dzeta;
                const double decay = 1.0 - computational / ztop;
                metrics.zeta(i, j, k) = computational;
                metrics.z(i, j, k) = computational + h * decay;
                metrics.J(i, j, k) = jacobian;
                metrics.mx(i, j, k) = -slope_x(i, j) * decay / jacobian;
                metrics.my(i, j, k) = -slope_y(i, j) * decay / jacobian;
            }
            for (int face = 0; face <= nz; ++face)
            {
                const double computational = face * dzeta;
                const double decay = 1.0 - computational / ztop;
                metrics.z_face(i, j, face) = computational + h * decay;
                metrics.Gx_face(i, j, face) = -slope_x(i, j) * decay;
                metrics.Gy_face(i, j, face) = -slope_y(i, j) * decay;
            }
        }
    for (int face = 0; face <= nx; ++face)
        for (int j = 0; j < ny; ++j)
            for (int k = 0; k < nz; ++k)
            {
                const int left = std::clamp(face - 1, 0, nx - 1);
                const int right = std::clamp(face, 0, nx - 1);
                metrics.Jx_face(face, j, k) = 0.5 * (
                    metrics.J(left, j, k) + metrics.J(right, j, k));
            }
    for (int i = 0; i < nx; ++i)
        for (int face = 0; face <= ny; ++face)
            for (int k = 0; k < nz; ++k)
            {
                const int south = std::clamp(face - 1, 0, ny - 1);
                const int north = std::clamp(face, 0, ny - 1);
                metrics.Jy_face(i, face, k) = 0.5 * (
                    metrics.J(i, south, k) + metrics.J(i, north, k));
            }
    return metrics;
}

struct State
{
    Field3D u{nx, ny, nz, 0.0f};
    Field3D v{nx, ny, nz, 0.0f};
    Field3D w{nx, ny, nz, 0.0f};
    Field3D rho{nx, ny, nz, 1.0f};
    Field3D p{nx, ny, nz, 90000.0f};
    Field3D theta{nx, ny, nz, 300.0f};
    Field3D qv{nx, ny, nz, 0.01f};
    Field3D tracer{nx, ny, nz, 0.2f};
    Field3D tke{nx, ny, nz, 0.1f};

    tmv::terrain_collocated::CollocatedMappedState mutable_view()
    {
        return {u, v, w, rho, p, theta, qv, tracer, tke};
    }
};

std::vector<double> pressure_reference()
{
    std::vector<double> profile(nz);
    for (int k = 0; k < nz; ++k)
        profile[static_cast<std::size_t>(k)] =
            100000.0 * std::exp(-((k + 0.5) * dzeta) / 8500.0);
    return profile;
}

std::vector<double> theta_reference()
{
    std::vector<double> profile(nz);
    for (int k = 0; k < nz; ++k)
        profile[static_cast<std::size_t>(k)] = 300.0 + 0.003 * (k + 0.5) * dzeta;
    return profile;
}

void initialize_reference(State& state,
                          const TerrainMetrics3D& metrics,
                          const std::vector<double>& pressure,
                          const std::vector<double>& theta)
{
    for (int i = 0; i < nx; ++i)
        for (int j = 0; j < ny; ++j)
            for (int k = 0; k < nz; ++k)
            {
                state.p[i][j][k] = static_cast<float>(
                    tmv::terrain_transport::reference_profile_value_fp32(
                        pressure, metrics.z(i, j, k), metrics.dzeta));
                state.theta[i][j][k] = static_cast<float>(
                    tmv::terrain_transport::reference_profile_value_fp32(
                        theta, metrics.z(i, j, k), metrics.dzeta));
                state.rho[i][j][k] = static_cast<float>(
                    tmv::terrain_transport::reference_density_fp32(
                        pressure, theta, metrics.z(i, j, k), metrics.dzeta));
            }
}

std::vector<float> bytes(const State& state)
{
    std::vector<float> values;
    for (const Field3D* field : std::array<const Field3D*, 9>{
             &state.u, &state.v, &state.w, &state.rho, &state.p,
             &state.theta, &state.qv, &state.tracer, &state.tke})
        values.insert(values.end(), field->data(), field->data() + field->size());
    return values;
}

void seed_transport(State& state)
{
    for (int i = 1; i < nx - 1; ++i)
        for (int j = 1; j < ny - 1; ++j)
            for (int k = 1; k < nz - 1; ++k)
            {
                state.u[i][j][k] = static_cast<float>(
                    1.2 * std::sin(0.31 * i + 0.17 * j + 0.11 * k));
                state.v[i][j][k] = static_cast<float>(
                    0.8 * std::cos(0.23 * i - 0.19 * j + 0.07 * k));
                state.w[i][j][k] = static_cast<float>(
                    0.15 * std::sin(0.13 * i + 0.29 * j - 0.21 * k));
                state.qv[i][j][k] = static_cast<float>(0.004 + 0.0001 * (i + j + k));
                state.tracer[i][j][k] = static_cast<float>(0.1 + 0.01 * i + 0.005 * j);
                state.tke[i][j][k] = static_cast<float>(0.05 + 0.002 * (i + 2 * j + k));
            }
}

double normalized(const tmv::terrain_collocated::ConservationClosure& closure)
{
    return std::abs(closure.residual)
        / std::max({1.0, std::abs(closure.before), std::abs(closure.after)});
}

void veto(void*,
          const tmv::terrain_collocated::CollocatedMappedStateView&,
          const TerrainContravariantFluxes& flux,
          const tmv::terrain_collocated::CollocatedMappedStepReport& report)
{
    REQUIRE(report.complete);
    REQUIRE(flux.x(1, 3, 3) == 0.0);
    REQUIRE(flux.x(nx - 1, 3, 3) == 0.0);
    REQUIRE(flux.z(3, 3, 1) == 0.0);
    REQUIRE(flux.z(3, 3, nz - 1) == 0.0);
    throw std::runtime_error("late mapped A-grid observer veto");
}

} // namespace

TEST_CASE("mapped collocated stage matches the literal retained-flux equation",
          "[dynamics][terrain][collocated][reference]")
{
    CHECK(dynamics_supports_terrain(
        CoordinateSystem::Cartesian, StaggerType::Collocated));
    const auto metrics = mapped_geometry();
    const auto p0 = pressure_reference();
    const auto theta0 = theta_reference();
    State state;
    initialize_reference(state, metrics, p0, theta0);
    seed_transport(state);
    const State before = state;
    constexpr double step = 0.4;
    constexpr int i = 3;
    constexpr int j = 3;
    constexpr int k = 3;

    const auto x_flux = [&](int face)
    {
        return metrics.Jx_face(face, j, k)
            * 0.5 * (before.rho[face - 1][j][k] + before.rho[face][j][k])
            * 0.5 * (before.u[face - 1][j][k] + before.u[face][j][k]);
    };
    const auto y_flux = [&](int face)
    {
        return metrics.Jy_face(i, face, k)
            * 0.5 * (before.rho[i][face - 1][k] + before.rho[i][face][k])
            * 0.5 * (before.v[i][face - 1][k] + before.v[i][face][k]);
    };
    const auto z_flux = [&](int face)
    {
        const double density = 0.5 * (
            before.rho[i][j][face - 1] + before.rho[i][j][face]);
        const double u = 0.5 * (
            before.u[i][j][face - 1] + before.u[i][j][face]);
        const double v = 0.5 * (
            before.v[i][j][face - 1] + before.v[i][j][face]);
        const double w = 0.5 * (
            before.w[i][j][face - 1] + before.w[i][j][face]);
        return density * (w + metrics.Gx_face(i, j, face) * u
                             + metrics.Gy_face(i, j, face) * v);
    };
    const double divergence = (x_flux(i + 1) - x_flux(i)) / dx
        + (y_flux(j + 1) - y_flux(j)) / dy
        + (z_flux(k + 1) - z_flux(k)) / dzeta;
    const float expected_rho = static_cast<float>(
        before.rho[i][j][k] - step * divergence / metrics.J(i, j, k));
    const auto upwind = [](double flux, float low, float high)
    {
        return flux * static_cast<double>(flux >= 0.0 ? low : high);
    };
    const double theta_divergence =
        (upwind(x_flux(i + 1), before.theta[i][j][k], before.theta[i + 1][j][k])
         - upwind(x_flux(i), before.theta[i - 1][j][k], before.theta[i][j][k])) / dx
      + (upwind(y_flux(j + 1), before.theta[i][j][k], before.theta[i][j + 1][k])
         - upwind(y_flux(j), before.theta[i][j - 1][k], before.theta[i][j][k])) / dy
      + (upwind(z_flux(k + 1), before.theta[i][j][k], before.theta[i][j][k + 1])
         - upwind(z_flux(k), before.theta[i][j][k - 1], before.theta[i][j][k])) / dzeta;
    const float expected_theta = static_cast<float>(
        (metrics.J(i, j, k) * before.rho[i][j][k] * before.theta[i][j][k]
         - step * theta_divergence)
        / (metrics.J(i, j, k) * expected_rho));

    tmv::terrain_collocated::CollocatedMappedStepReport report;
    tmv::terrain_collocated::advance_collocated_mapped_stage(
        state.mutable_view(), metrics, p0, theta0, step, report);

    CHECK(state.rho[i][j][k] == expected_rho);
    CHECK(state.theta[i][j][k] == expected_theta);
    CHECK(state.p[i][j][k] == static_cast<float>(
        tmv::dynamics_eos::pressure_fp64(
            state.rho[i][j][k], state.theta[i][j][k])));
    CHECK(report.complete);
    CHECK(report.physical_cells == static_cast<std::uint64_t>((nx - 2) * (ny - 2) * (nz - 2)));
    CHECK(report.eos_projected_cells == report.physical_cells);
    CHECK(report.rigid_closed_boundary);
    CHECK(report.final_pressure_uses_canonical_fp64);

    CHECK(0.5f * (state.u[0][j][k] + state.u[1][j][k]) == 0.0f);
    CHECK(0.5f * (state.v[i][0][k] + state.v[i][1][k]) == 0.0f);
    CHECK(0.5f * (state.w[i][j][nz - 1] + state.w[i][j][nz - 2]) == 0.0f);
    const auto ground = tmv::terrain_transport::ground_surface_frame(metrics, i, j);
    const double face_w = 0.5 * (
        static_cast<double>(state.w[i][j][0]) + state.w[i][j][1]);
    CHECK(face_w == Approx(
        ground.slope_x * state.u[i][j][1]
        + ground.slope_y * state.v[i][j][1]).margin(2e-8));
}

TEST_CASE("mapped collocated reference atmosphere is a well-balanced rest state",
          "[dynamics][terrain][collocated][well_balanced]")
{
    const auto metrics = mapped_geometry();
    const auto p0 = pressure_reference();
    const auto theta0 = theta_reference();
    State state;
    initialize_reference(state, metrics, p0, theta0);

    tmv::terrain_collocated::CollocatedMappedStepReport report;
    tmv::terrain_collocated::advance_collocated_mapped_stage(
        state.mutable_view(), metrics, p0, theta0, 1.0, report);

    for (int i = 1; i < nx - 1; ++i)
        for (int j = 1; j < ny - 1; ++j)
            for (int k = 1; k < nz - 1; ++k)
            {
                CHECK(state.u[i][j][k] == 0.0f);
                CHECK(state.v[i][j][k] == 0.0f);
                CHECK(state.w[i][j][k] == 0.0f);
            }
    CHECK(report.dry_mass.residual == 0.0);
    CHECK(report.rho_theta.residual == 0.0);
    CHECK(report.water_vapor.residual == 0.0);
    CHECK(report.tracer.residual == 0.0);
    CHECK(report.subgrid_tke.residual == 0.0);
}

TEST_CASE("mapped collocated retained flux closes mass scalars and advected momentum",
          "[dynamics][terrain][collocated][conservation]")
{
    const auto metrics = mapped_geometry();
    const auto p0 = pressure_reference();
    const auto theta0 = theta_reference();
    State state;
    initialize_reference(state, metrics, p0, theta0);
    seed_transport(state);

    tmv::terrain_collocated::CollocatedMappedStepReport report;
    tmv::terrain_collocated::advance_collocated_mapped_stage(
        state.mutable_view(), metrics, p0, theta0, 0.5, report);

    CHECK(normalized(report.dry_mass) < 2e-8);
    CHECK(normalized(report.rho_theta) < 2e-8);
    CHECK(normalized(report.water_vapor) < 2e-8);
    CHECK(normalized(report.tracer) < 2e-8);
    CHECK(normalized(report.subgrid_tke) < 2e-8);
    CHECK(normalized(report.x_momentum) < 2e-7);
    CHECK(normalized(report.y_momentum) < 2e-7);
    CHECK(normalized(report.z_momentum) < 2e-7);
    CHECK(report.dry_mass.boundary_outflow == 0.0);
    CHECK(report.max_outgoing_courant < 1.0);
}

TEST_CASE("mapped collocated invalid and late failures preserve fields and report",
          "[dynamics][terrain][collocated][atomicity]")
{
    const auto metrics = mapped_geometry();
    const auto p0 = pressure_reference();
    const auto theta0 = theta_reference();
    State state;
    initialize_reference(state, metrics, p0, theta0);
    seed_transport(state);
    const auto original = bytes(state);
    tmv::terrain_collocated::CollocatedMappedStepReport report;
    report.dt_s = 19.0;
    report.complete = true;

    CHECK_THROWS_AS(
        tmv::terrain_collocated::advance_collocated_mapped_stage(
            state.mutable_view(), metrics, p0, theta0, -1.0, report),
        std::invalid_argument);
    CHECK(bytes(state) == original);
    CHECK(report.dt_s == 19.0);
    CHECK(report.complete);

    CHECK_THROWS_AS(
        tmv::terrain_collocated::advance_collocated_mapped_stage(
            state.mutable_view(), metrics, p0, theta0, 1.0e9, report),
        std::invalid_argument);
    CHECK(bytes(state) == original);
    CHECK(report.dt_s == 19.0);

    auto malformed = metrics;
    malformed.J(3, 3, 3) = 0.0;
    CHECK_THROWS_AS(
        tmv::terrain_collocated::advance_collocated_mapped_stage(
            state.mutable_view(), malformed, p0, theta0, 0.5, report),
        std::invalid_argument);
    CHECK(bytes(state) == original);
    CHECK(report.dt_s == 19.0);

    CHECK_THROWS_WITH(
        tmv::terrain_collocated::advance_collocated_mapped_stage(
            state.mutable_view(), metrics, p0, theta0, 0.5, report, veto),
        "late mapped A-grid observer veto");
    CHECK(bytes(state) == original);
    CHECK(report.dt_s == 19.0);
    CHECK(report.complete);
}

TEST_CASE("mapped collocated transaction is bitwise identical at one and four threads",
          "[dynamics][terrain][collocated][threads]")
{
    const auto metrics = mapped_geometry();
    const auto p0 = pressure_reference();
    const auto theta0 = theta_reference();
    State one;
    initialize_reference(one, metrics, p0, theta0);
    seed_transport(one);
    State four = one;
    tmv::terrain_collocated::CollocatedMappedStepReport report_one;
    tmv::terrain_collocated::CollocatedMappedStepReport report_four;

#ifdef _OPENMP
    omp_set_dynamic(0);
    omp_set_num_threads(1);
#endif
    tmv::terrain_collocated::advance_collocated_mapped_stage(
        one.mutable_view(), metrics, p0, theta0, 0.5, report_one);
#ifdef _OPENMP
    omp_set_num_threads(4);
#endif
    tmv::terrain_collocated::advance_collocated_mapped_stage(
        four.mutable_view(), metrics, p0, theta0, 0.5, report_four);

    CHECK(bytes(one) == bytes(four));
    CHECK(report_one.dry_mass.residual == report_four.dry_mass.residual);
    CHECK(report_one.rho_theta.residual == report_four.rho_theta.residual);
    CHECK(report_one.x_momentum.residual == report_four.x_momentum.residual);
    CHECK(report_one.max_outgoing_courant == report_four.max_outgoing_courant);
}
