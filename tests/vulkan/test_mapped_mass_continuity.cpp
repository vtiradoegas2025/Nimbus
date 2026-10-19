#include "catch2/catch.hpp"
#include "boundary_conditions/factory.hpp"
#include "compute/compute_backend.hpp"
#include "core/field/field3d.hpp"
#include "core/field/field_bounds.hpp"
#include "core/infra/coordinate_system.hpp"
#include "core/infra/physical_constants.hpp"
#include "core/runtime/grid_state.hpp"
#include "core/runtime/process_state.hpp"
#include "core/runtime/runtime_config.hpp"
#include "core/runtime/simulation_state.hpp"
#include "dynamics/eos_pressure.hpp"
#include "numerics/advection/advection_cartesian.hpp"
#include "numerics/runtime.hpp"
#include "terrain/base/topography.hpp"
#include "terrain/conservative_transport.hpp"
#include "terrain/native_mapped_transaction.hpp"
#include "numerics/derivatives/derivative_operators.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <omp.h>

namespace
{
constexpr int nx = 8;
constexpr int ny = 7;
constexpr int nz = 5;
constexpr float dx = 750.0f;
constexpr float dy = 750.0f;
constexpr float gamma_value = 1.4f;
constexpr std::size_t cells = static_cast<std::size_t>(nx) * ny * nz;

std::size_t at(int i, int j, int k)
{
    return (static_cast<std::size_t>(i) * ny + j) * nz + k;
}

struct CanonicalCase
{
    TerrainMetrics3D metrics;
    Field3D rho{nx, ny, nz};
    Field3D u{nx, ny, nz};
    Field3D v{nx, ny, nz};
    Field3D w{nx, ny, nz};
    Field3D p{nx, ny, nz};
    std::vector<float> jacobian = std::vector<float>(cells);
    std::vector<float> jacobian_x = std::vector<float>(cells);
    std::vector<float> jacobian_y = std::vector<float>(cells);
    std::vector<float> metric_x = std::vector<float>(cells);
    std::vector<float> metric_y = std::vector<float>(cells);
    std::vector<float> ground_metric_x = std::vector<float>(cells);
    std::vector<float> ground_metric_y = std::vector<float>(cells);
    std::vector<float> ground_w = std::vector<float>(cells, -11.0f);
    std::vector<float> drho = std::vector<float>(cells, -12.0f);
    std::vector<float> dp = std::vector<float>(cells, -13.0f);
    std::vector<float> flux_x = std::vector<float>(cells, -14.0f);
    std::vector<float> flux_y = std::vector<float>(cells, -15.0f);
    std::vector<float> flux_zeta = std::vector<float>(cells, -16.0f);
    MappedTerrainMassContinuityReport report{7, 9, false};

    CanonicalCase()
    {
        TerrainConfig config;
        config.ztop = 5000.0;
        config.bell.h0 = 220.0;
        config.bell.a = 1600.0;
        config.bell.axisymmetric = true;
        Topography2D topo;
        topography::initialize_topography(topo, nx, ny);
        for (int i = 0; i < nx; ++i)
            for (int j = 0; j < ny; ++j)
            {
                const auto bell = topography::eval_bell((i - 0.5 * (nx - 1)) * dx,
                                                        (j - 0.5 * (ny - 1)) * dy,
                                                        config.bell);
                topo.h[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] = bell.h;
                topo.hx[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] = bell.hx;
                topo.hy[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] = bell.hy;
            }
        topography::initialize_metrics(metrics, nx, ny, nz);
        topography::build_conservative_face_metrics(
            config, topo, topography::build_zeta_interfaces(nz, config.ztop), dx, dy, metrics);
        for (int i = 0; i < nx; ++i)
            for (int j = 0; j < ny; ++j)
                for (int k = 0; k < nz; ++k)
                {
                    const auto n = at(i, j, k);
                    rho[i][j][k] = 1.12f - 0.045f * k + 8.0e-4f * i - 5.0e-4f * j;
                    u[i][j][k] = 4.0f + 0.11f * i - 0.06f * j + 0.03f * k;
                    v[i][j][k] = -1.3f - 0.04f * i + 0.09f * j - 0.02f * k;
                    w[i][j][k] = k == nz - 1 ? 0.0f :
                        0.06f * std::sin(0.4f * i + 0.7f * j + 0.2f * k);
                    p[i][j][k] = 96000.0f - 1250.0f * k + 7.0f * i - 4.0f * j;
                    jacobian[n] = static_cast<float>(metrics.J(i, j, k));
                    jacobian_x[n] = i < nx - 1 ? static_cast<float>(metrics.Jx_face(i + 1, j, k)) : 0.0f;
                    jacobian_y[n] = j < ny - 1 ? static_cast<float>(metrics.Jy_face(i, j + 1, k)) : 0.0f;
                    metric_x[n] = static_cast<float>(metrics.Gx_face(i, j, k + 1));
                    metric_y[n] = static_cast<float>(metrics.Gy_face(i, j, k + 1));
                    ground_metric_x[n] = static_cast<float>(metrics.Gx_face(i, j, 0));
                    ground_metric_y[n] = static_cast<float>(metrics.Gy_face(i, j, 0));
                }
    }

    MappedTerrainMassContinuityDispatch dispatch()
    {
        return {rho.data(), u.data(), v.data(), w.data(), p.data(), jacobian.data(), jacobian_x.data(),
            jacobian_y.data(), metric_x.data(), metric_y.data(), ground_metric_x.data(), ground_metric_y.data(),
            ground_w.data(), drho.data(), dp.data(), flux_x.data(), flux_y.data(), flux_zeta.data(),
            nx, ny, nz, dx, dy, static_cast<float>(metrics.dzeta), gamma_value, 0.0f, 0.0f, &report};
    }
};

struct Expected
{
    std::vector<float> ground_w = std::vector<float>(cells);
    std::vector<float> drho = std::vector<float>(cells);
    std::vector<float> dp = std::vector<float>(cells);
    std::vector<float> flux_x = std::vector<float>(cells);
    std::vector<float> flux_y = std::vector<float>(cells);
    std::vector<float> flux_zeta = std::vector<float>(cells);
};

Expected canonical_expected(const CanonicalCase& c)
{
    Expected result;
    for (int i = 1; i < nx; ++i)
        for (int j = 1; j < ny; ++j)
            result.ground_w[at(i, j, 0)] = static_cast<float>(
                tmv::terrain_transport::cgrid_ground_vertical_velocity(c.metrics, c.u, c.v, i, j));
    for (int i = 0; i < nx - 1; ++i)
        for (int j = 0; j < ny; ++j)
            for (int k = 0; k < nz; ++k)
                result.flux_x[at(i, j, k)] = static_cast<float>(
                    tmv::terrain_transport::cgrid_mass_flux_x(c.metrics, c.rho, c.u, i, j, k));
    for (int i = 0; i < nx; ++i)
        for (int j = 0; j < ny - 1; ++j)
            for (int k = 0; k < nz; ++k)
                result.flux_y[at(i, j, k)] = static_cast<float>(
                    tmv::terrain_transport::cgrid_mass_flux_y(c.metrics, c.rho, c.v, i, j, k));
    for (int i = 1; i < nx - 1; ++i)
        for (int j = 1; j < ny - 1; ++j)
            for (int k = 0; k < nz; ++k)
                result.flux_zeta[at(i, j, k)] = static_cast<float>(
                    tmv::terrain_transport::cgrid_mass_flux_zeta(c.metrics, c.rho, c.u, c.v, c.w, i, j, k));
    for (int i = 1; i < nx - 1; ++i)
        for (int j = 1; j < ny - 1; ++j)
            for (int k = 0; k < nz; ++k)
            {
                const auto n = at(i, j, k);
                result.drho[n] = static_cast<float>(-
                    tmv::terrain_transport::cgrid_mass_divergence(c.metrics, c.rho, c.u, c.v, c.w, i, j, k));
                result.dp[n] = static_cast<float>(-gamma_value * static_cast<double>(c.p[i][j][k]) *
                    tmv::terrain_transport::cgrid_velocity_divergence(c.metrics, c.u, c.v, c.w, i, j, k));
            }
    return result;
}

void require_close(const std::vector<float>& expected, const std::vector<float>& actual,
                   float tolerance, const char* label)
{
    float max_abs = 0.0f;
    for (std::size_t n = 0; n < expected.size(); ++n)
        max_abs = std::max(max_abs, std::abs(expected[n] - actual[n]));
    INFO(label << " max_abs=" << max_abs);
    REQUIRE(max_abs <= tolerance);
}

bool outputs_unchanged(const CanonicalCase& after, const CanonicalCase& before)
{
    const auto same = [](const std::vector<float>& a, const std::vector<float>& b)
    { return std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0; };
    return same(after.ground_w, before.ground_w) && same(after.drho, before.drho) && same(after.dp, before.dp) &&
           same(after.flux_x, before.flux_x) && same(after.flux_y, before.flux_y) &&
           same(after.flux_zeta, before.flux_zeta) &&
           std::memcmp(&after.report, &before.report, sizeof(after.report)) == 0;
}

struct FastMomentumCase
{
    CanonicalCase state;
    std::vector<float> height = std::vector<float>(cells);
    std::vector<float> reference_pressure = {96000.0f, 94750.0f, 93500.0f, 92250.0f, 91000.0f};
    std::vector<float> du = std::vector<float>(cells, -21.0f);
    std::vector<float> dv = std::vector<float>(cells, -22.0f);
    std::vector<float> dw = std::vector<float>(cells, -23.0f);
    MappedTerrainFastMomentumReport report{8, false};

    FastMomentumCase()
    {
        const std::vector<double> reference(reference_pressure.begin(), reference_pressure.end());
        for (int i = 0; i < nx; ++i)
            for (int j = 0; j < ny; ++j)
                for (int k = 0; k < nz; ++k)
                {
                    const auto n = at(i, j, k);
                    height[n] = static_cast<float>(state.metrics.z(i, j, k));
                    const double base = tmv::terrain_transport::reference_profile_value_fp32(
                        reference, state.metrics.z(i, j, k), state.metrics.dzeta);
                    state.p[i][j][k] = static_cast<float>(base +
                        24.0 * std::sin(0.31 * i + 0.43 * j + 0.27 * k));
                }
    }

    MappedTerrainFastMomentumDispatch dispatch()
    {
        return {state.rho.data(), state.p.data(), height.data(), reference_pressure.data(),
                du.data(), dv.data(), dw.data(), nx, ny, nz, dx, dy,
                static_cast<float>(state.metrics.dzeta), &report};
    }
};

struct FastMomentumExpected
{
    std::vector<float> du = std::vector<float>(cells);
    std::vector<float> dv = std::vector<float>(cells);
    std::vector<float> dw = std::vector<float>(cells);
};

FastMomentumExpected canonical_fast_momentum_expected(const FastMomentumCase& c)
{
    FastMomentumExpected result;
    const std::vector<double> reference(c.reference_pressure.begin(), c.reference_pressure.end());
    for (int i = 1; i < nx - 1; ++i)
        for (int j = 1; j < ny - 1; ++j)
            for (int k = 0; k < nz - 1; ++k)
            {
                const auto n = at(i, j, k);
                const double rho_x = 0.5 * (static_cast<double>(c.state.rho[i][j][k]) +
                                             static_cast<double>(c.state.rho[i + 1][j][k]));
                const double rho_y = 0.5 * (static_cast<double>(c.state.rho[i][j][k]) +
                                             static_cast<double>(c.state.rho[i][j + 1][k]));
                const double rho_z = 0.5 * (static_cast<double>(c.state.rho[i][j][k]) +
                                             static_cast<double>(c.state.rho[i][j][k + 1]));
                result.du[n] = static_cast<float>(-
                    tmv::terrain_transport::cgrid_pressure_perturbation_gradient_x(
                        c.state.metrics, c.state.p, reference, i, j, k) / rho_x);
                result.dv[n] = static_cast<float>(-
                    tmv::terrain_transport::cgrid_pressure_perturbation_gradient_y(
                        c.state.metrics, c.state.p, reference, i, j, k) / rho_y);
                result.dw[n] = static_cast<float>(-
                    tmv::terrain_transport::cgrid_pressure_perturbation_gradient_z(
                        c.state.metrics, c.state.p, reference, i, j, k) / rho_z);
            }
    return result;
}

bool momentum_outputs_unchanged(const FastMomentumCase& after, const FastMomentumCase& before)
{
    const auto same = [](const std::vector<float>& a, const std::vector<float>& b)
    { return std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0; };
    return same(after.du, before.du) && same(after.dv, before.dv) && same(after.dw, before.dw) &&
           std::memcmp(&after.report, &before.report, sizeof(after.report)) == 0;
}

struct SlowTendencyCase
{
    CanonicalCase state;
    std::vector<float> height = std::vector<float>(cells);
    std::vector<float> reference_pressure = {96000.0f, 94750.0f, 93500.0f, 92250.0f, 91000.0f};
    std::vector<float> reference_theta = {300.0f, 301.0f, 302.0f, 303.0f, 304.0f};
    std::vector<float> qv = std::vector<float>(cells);
    std::vector<float> reference_qv = std::vector<float>(nz);
    std::vector<float> loading = std::vector<float>(cells);
    std::array<std::vector<float>, 8> output;
    MappedTerrainSlowTendenciesReport report{6, false, false};
    bool has_moisture = false;
    bool has_loading = false;

    SlowTendencyCase()
    {
        for (auto& field : output) field.assign(cells, -31.0f);
        for (int i = 0; i < nx; ++i)
            for (int j = 0; j < ny; ++j)
                for (int k = 0; k < nz; ++k)
                    height[at(i, j, k)] = static_cast<float>(state.metrics.z(i, j, k));
    }

    MappedTerrainSlowTendenciesDispatch dispatch()
    {
        return {state.u.data(), state.v.data(), state.w.data(), state.rho.data(), state.p.data(),
            height.data(), state.jacobian_x.data(), state.jacobian_y.data(), state.metric_x.data(),
            state.metric_y.data(), state.ground_w.data(), reference_pressure.data(),
            reference_theta.data(), qv.data(), reference_qv.data(), loading.data(),
            output[0].data(), output[1].data(), output[2].data(), output[3].data(),
            output[4].data(), output[5].data(), output[6].data(), output[7].data(),
            nx, ny, nz, has_moisture, has_loading, dx, dy, static_cast<float>(state.metrics.dzeta),
            static_cast<float>(state.metrics.dzeta), 9.81f, &report};
    }
};

std::array<std::vector<float>, 8> canonical_slow_expected(const SlowTendencyCase& c)
{
    std::array<std::vector<float>, 8> result;
    for (auto& field : result) field.assign(cells, 0.0f);
    StaggeredCartesianDerivatives deriv(dx, dy, c.state.metrics.dzeta, nx, ny, nz);
    const std::vector<double> p0(c.reference_pressure.begin(), c.reference_pressure.end());
    const std::vector<double> theta0(c.reference_theta.begin(), c.reference_theta.end());
    const double inv_dz = 1.0 / c.state.metrics.dzeta;
    for (int i = 1; i < nx - 1; ++i)
        for (int j = 1; j < ny - 1; ++j)
            for (int k = 0; k < nz - 1; ++k)
            {
                const auto n = at(i, j, k);
                const double ux = c.state.u[i][j][k];
                const double vy_at_x = 0.25 * (c.state.v[i][j - 1][k] + c.state.v[i][j][k] +
                                                c.state.v[i + 1][j - 1][k] + c.state.v[i + 1][j][k]);
                const double wx = k == 0 ? 0.25 * (
                    tmv::terrain_transport::cgrid_ground_vertical_velocity(c.state.metrics, c.state.u, c.state.v, i, j) +
                    tmv::terrain_transport::cgrid_ground_vertical_velocity(c.state.metrics, c.state.u, c.state.v, i + 1, j) +
                    c.state.w[i][j][0] + c.state.w[i + 1][j][0]) :
                    0.25 * (c.state.w[i][j][k - 1] + c.state.w[i][j][k] +
                            c.state.w[i + 1][j][k - 1] + c.state.w[i + 1][j][k]);
                const double zeta_x = tmv::terrain_transport::cgrid_zeta_velocity_at_x_face(
                    c.state.metrics, ux, vy_at_x, wx, i, j, k);
                const double duz = k == 0 ? (zeta_x >= 0.0 ? 0.0 :
                    (c.state.u[i][j][1] - c.state.u[i][j][0]) * inv_dz) :
                    deriv.upwind_dz(c.state.u, zeta_x, i, j, k);
                const double adv_u = -ux * deriv.upwind_dx(c.state.u, ux, i, j, k) -
                    vy_at_x * deriv.upwind_dy(c.state.u, vy_at_x, i, j, k) - zeta_x * duz;
                result[0][n] = result[5][n] = static_cast<float>(adv_u);

                const double vy = c.state.v[i][j][k];
                const double ux_at_y = 0.25 * (c.state.u[i - 1][j][k] + c.state.u[i][j][k] +
                                                c.state.u[i - 1][j + 1][k] + c.state.u[i][j + 1][k]);
                const double wy = k == 0 ? 0.25 * (
                    tmv::terrain_transport::cgrid_ground_vertical_velocity(c.state.metrics, c.state.u, c.state.v, i, j) +
                    tmv::terrain_transport::cgrid_ground_vertical_velocity(c.state.metrics, c.state.u, c.state.v, i, j + 1) +
                    c.state.w[i][j][0] + c.state.w[i][j + 1][0]) :
                    0.25 * (c.state.w[i][j][k - 1] + c.state.w[i][j][k] +
                            c.state.w[i][j + 1][k - 1] + c.state.w[i][j + 1][k]);
                const double zeta_y = tmv::terrain_transport::cgrid_zeta_velocity_at_y_face(
                    c.state.metrics, ux_at_y, vy, wy, i, j, k);
                const double dvz = k == 0 ? (zeta_y >= 0.0 ? 0.0 :
                    (c.state.v[i][j][1] - c.state.v[i][j][0]) * inv_dz) :
                    deriv.upwind_dz(c.state.v, zeta_y, i, j, k);
                const double adv_v = -ux_at_y * deriv.upwind_dx(c.state.v, ux_at_y, i, j, k) -
                    vy * deriv.upwind_dy(c.state.v, vy, i, j, k) - zeta_y * dvz;
                result[1][n] = result[6][n] = static_cast<float>(adv_v);

                const double ux_w = 0.25 * (c.state.u[i - 1][j][k] + c.state.u[i][j][k] +
                    c.state.u[i - 1][j][k + 1] + c.state.u[i][j][k + 1]);
                const double vy_w = 0.25 * (c.state.v[i][j - 1][k] + c.state.v[i][j][k] +
                    c.state.v[i][j - 1][k + 1] + c.state.v[i][j][k + 1]);
                const double zeta_w = tmv::terrain_transport::cgrid_zeta_velocity_at_z_face(
                    c.state.metrics, ux_w, vy_w, c.state.w[i][j][k], i, j, k);
                const double ground = tmv::terrain_transport::cgrid_ground_vertical_velocity(
                    c.state.metrics, c.state.u, c.state.v, i, j);
                const double dwz = k == 0 ? (zeta_w >= 0.0 ?
                    (c.state.w[i][j][0] - ground) * inv_dz :
                    (c.state.w[i][j][1] - c.state.w[i][j][0]) * inv_dz) :
                    deriv.upwind_dz(c.state.w, zeta_w, i, j, k);
                const double adv_w = -ux_w * deriv.upwind_dx(c.state.w, ux_w, i, j, k) -
                    vy_w * deriv.upwind_dy(c.state.w, vy_w, i, j, k) - zeta_w * dwz;
                const double rho_face = 0.5 * (c.state.rho[i][j][k] + c.state.rho[i][j][k + 1]);
                const double rho0_face = 0.5 * (
                    tmv::terrain_transport::reference_density_fp32(p0, theta0, c.state.metrics.z(i, j, k), c.state.metrics.dzeta) +
                    tmv::terrain_transport::reference_density_fp32(p0, theta0, c.state.metrics.z(i, j, k + 1), c.state.metrics.dzeta));
                double source = -9.81 * (rho_face - rho0_face) / rho_face;
                if (c.has_moisture)
                {
                    const std::vector<double> qv0(c.reference_qv.begin(), c.reference_qv.end());
                    source += 9.81 * 0.608 * (0.5 * (c.qv[n] + c.qv[at(i, j, k + 1)]) -
                        0.5 * (tmv::terrain_transport::reference_profile_value_fp32(
                                   qv0, c.state.metrics.z(i, j, k), c.state.metrics.dzeta) +
                               tmv::terrain_transport::reference_profile_value_fp32(
                                   qv0, c.state.metrics.z(i, j, k + 1), c.state.metrics.dzeta)));
                }
                if (c.has_loading)
                    source -= 9.81 * 0.5 * (c.loading[n] + c.loading[at(i, j, k + 1)]);
                result[2][n] = static_cast<float>(adv_w + source);
                result[7][n] = static_cast<float>(adv_w);
            }
    for (int i = 1; i < nx - 1; ++i)
        for (int j = 1; j < ny - 1; ++j)
            for (int k = 1; k < nz - 1; ++k)
            {
                const auto n = at(i, j, k);
                const double center_u = 0.5 * (c.state.u[i][j][k] + c.state.u[i - 1][j][k]);
                const double center_v = 0.5 * (c.state.v[i][j][k] + c.state.v[i][j - 1][k]);
                const double center_w = 0.5 * (c.state.w[i][j][k] + c.state.w[i][j][k - 1]);
                const double gx = 0.5 * (
                    tmv::terrain_transport::cgrid_pressure_perturbation_gradient_x(c.state.metrics, c.state.p, p0, i - 1, j, k) +
                    tmv::terrain_transport::cgrid_pressure_perturbation_gradient_x(c.state.metrics, c.state.p, p0, i, j, k));
                const double gy = 0.5 * (
                    tmv::terrain_transport::cgrid_pressure_perturbation_gradient_y(c.state.metrics, c.state.p, p0, i, j - 1, k) +
                    tmv::terrain_transport::cgrid_pressure_perturbation_gradient_y(c.state.metrics, c.state.p, p0, i, j, k));
                const double gz = tmv::terrain_transport::cgrid_scalar_gradient_z_center(
                    c.state.metrics, c.state.p, i, j, k);
                result[4][n] = static_cast<float>(-center_u * gx - center_v * gy - center_w * gz);
            }
    return result;
}

bool slow_outputs_unchanged(const SlowTendencyCase& after, const SlowTendencyCase& before)
{
    for (std::size_t field = 0; field < after.output.size(); ++field)
        if (std::memcmp(after.output[field].data(), before.output[field].data(), cells * sizeof(float)) != 0)
            return false;
    return std::memcmp(&after.report, &before.report, sizeof(after.report)) == 0;
}

struct ScalarTransportCase
{
    std::vector<float> scalar_begin = std::vector<float>(cells);
    std::vector<float> density_begin = std::vector<float>(cells);
    std::vector<float> density_end = std::vector<float>(cells);
    std::vector<float> mass_flux_x = std::vector<float>(cells);
    std::vector<float> mass_flux_y = std::vector<float>(cells);
    std::vector<float> mass_flux_zeta = std::vector<float>(cells);
    std::vector<float> jacobian = std::vector<float>(cells);
    std::vector<float> scalar_end = std::vector<float>(cells, -41.0f);
    MappedTerrainScalarTransportReport report{7, 8, 9, false};
    bool positive_definite = false;
    float step = 0.4f;
    float dzeta = 0.0f;

    ScalarTransportCase(const CanonicalCase& carrier, bool positive)
        : positive_definite(positive)
    {
        dzeta = static_cast<float>(carrier.metrics.dzeta);
        const Expected continuity = canonical_expected(carrier);
        for (int i = 0; i < nx; ++i)
            for (int j = 0; j < ny; ++j)
                for (int k = 0; k < nz; ++k)
                {
                    const auto n = at(i, j, k);
                    const float wave = std::exp(-0.24f * ((i - 4.0f) * (i - 4.0f) +
                                                          (j - 3.0f) * (j - 3.0f) +
                                                          (k - 2.0f) * (k - 2.0f)));
                    scalar_begin[n] = positive ? 0.001f + 0.006f * wave : 299.0f + 5.0f * wave;
                    density_begin[n] = carrier.rho.data()[n];
                    density_end[n] = density_begin[n] + step * continuity.drho[n];
                    mass_flux_x[n] = carrier.flux_x[n];
                    mass_flux_y[n] = carrier.flux_y[n];
                    mass_flux_zeta[n] = carrier.flux_zeta[n];
                    jacobian[n] = carrier.jacobian[n];
                }
        if (positive_definite)
            scalar_begin[at(4, 3, 2)] = 1.0e-12f;
    }

    MappedTerrainScalarTransportDispatch dispatch()
    {
        MappedTerrainScalarTransportDispatch d;
        d.scalar_begin = scalar_begin.data();
        d.density_begin = density_begin.data();
        d.density_end = density_end.data();
        d.mass_flux_x = mass_flux_x.data();
        d.mass_flux_y = mass_flux_y.data();
        d.mass_flux_zeta = mass_flux_zeta.data();
        d.jacobian = jacobian.data();
        d.scalar_end = scalar_end.data();
        d.nx = nx;
        d.ny = ny;
        d.nz = nz;
        d.positive_definite = positive_definite;
        d.dx = dx;
        d.dy = dy;
        d.dzeta = dzeta;
        d.dt = step;
        d.density_floor = density_min_kgm3;
        d.report = &report;
        return d;
    }
};

void configure_scalar_oracle_grid(const CanonicalCase& carrier)
{
    NR = nx;
    NTH = ny;
    NZ = nz;
    dr = dx;
    dtheta = dy;
    dz = carrier.metrics.dzeta;
    global_coordinate_system = CoordinateSystem::Cartesian;
    global_stagger_type = StaggerType::CGrid;
    global_lateral_bc_west = tmv::init::config::LateralBC::LegacyOpen;
    global_lateral_bc_east = tmv::init::config::LateralBC::LegacyOpen;
    global_lateral_bc_south = tmv::init::config::LateralBC::LegacyOpen;
    global_lateral_bc_north = tmv::init::config::LateralBC::LegacyOpen;
    global_bottom_bc = tmv::init::config::BottomBC::RigidSlip;
    global_top_bc = tmv::init::config::TopBC::Rigid;
    global_grid_geometry.initialize(nx, ny, nz, dx, carrier.metrics.dzeta, dy,
                                    CoordinateSystem::Cartesian, StaggerType::CGrid);
}

std::vector<float> canonical_scalar_expected(const CanonicalCase& carrier,
                                             const ScalarTransportCase& scalar)
{
    configure_scalar_oracle_grid(carrier);
    rho.resize(nx, ny, nz, 0.0f);
    u = carrier.u;
    v = carrier.v;
    w = carrier.w;
    std::copy(scalar.density_end.begin(), scalar.density_end.end(), rho.data());
    Field3D source(nx, ny, nz), result;
    std::copy(scalar.scalar_begin.begin(), scalar.scalar_begin.end(), source.data());
    CartesianCGridTransportContext context;
    context.begin(nx, ny, nz, scalar.step, &carrier.metrics);
    context.accumulate(carrier.rho, carrier.u, carrier.v, carrier.w, scalar.step);
    context.complete();
    advect_scalar_conservative_3d_cartesian(
        source, result, scalar.step, scalar.positive_definite, &context);
    return std::vector<float>(result.data(), result.data() + result.size());
}

bool scalar_outputs_unchanged(const ScalarTransportCase& after,
                              const ScalarTransportCase& before)
{
    return std::memcmp(after.scalar_end.data(), before.scalar_end.data(), cells * sizeof(float)) == 0 &&
           std::memcmp(&after.report, &before.report, sizeof(after.report)) == 0;
}

struct StateUpdateCase
{
    std::array<std::vector<float>, 11> input;
    std::array<std::vector<float>, 5> output;
    MappedTerrainStateUpdateReport report{7, MappedTerrainStateUpdateMode::FastMomentum, false};
    MappedTerrainStateUpdateMode mode = MappedTerrainStateUpdateMode::Slow;
    float step = 0.025f;

    explicit StateUpdateCase(const CanonicalCase& carrier)
    {
        for (auto& field : input)
            field.resize(cells);
        for (auto& field : output)
            field.assign(cells, -47.0f);
        for (std::size_t n = 0; n < cells; ++n)
        {
            input[0][n] = carrier.u.data()[n];
            input[1][n] = carrier.v.data()[n];
            input[2][n] = carrier.w.data()[n];
            input[3][n] = carrier.rho.data()[n];
            input[4][n] = carrier.p.data()[n];
            input[5][n] = 299.0f + 0.015f * static_cast<float>(n);
            input[6][n] = 0.01f * std::sin(0.03f * static_cast<float>(n));
            input[7][n] = -0.012f * std::cos(0.02f * static_cast<float>(n));
            input[8][n] = 0.008f * std::sin(0.05f * static_cast<float>(n));
            input[9][n] = 0.0003f * std::cos(0.04f * static_cast<float>(n));
            input[10][n] = 2.5f * std::sin(0.06f * static_cast<float>(n));
        }
    }

    MappedTerrainStateUpdateDispatch dispatch()
    {
        MappedTerrainStateUpdateDispatch d;
        d.u = input[0].data();
        d.v = input[1].data();
        d.w = input[2].data();
        d.rho = input[3].data();
        d.p = input[4].data();
        d.theta = input[5].data();
        d.du_dt = input[6].data();
        d.dv_dt = input[7].data();
        d.dw_dt = input[8].data();
        d.drho_dt = input[9].data();
        d.dp_dt = input[10].data();
        d.u_candidate = output[0].data();
        d.v_candidate = output[1].data();
        d.w_candidate = output[2].data();
        d.rho_candidate = output[3].data();
        d.p_candidate = output[4].data();
        d.nx = nx;
        d.ny = ny;
        d.nz = nz;
        d.mode = mode;
        d.dt = step;
        d.report = &report;
        return d;
    }
};

bool state_update_outputs_unchanged(const StateUpdateCase& after,
                                    const StateUpdateCase& before)
{
    for (std::size_t index = 0; index < after.output.size(); ++index)
        if (std::memcmp(after.output[index].data(), before.output[index].data(), cells * sizeof(float)) != 0)
            return false;
    return std::memcmp(&after.report, &before.report, sizeof(after.report)) == 0;
}

struct ScalarObservation
{
    std::vector<float> scalar_end = std::vector<float>(cells);
    std::array<std::vector<float>, 3> flux = {
        std::vector<float>(cells), std::vector<float>(cells), std::vector<float>(cells)};
    std::vector<float> limiter = std::vector<float>(cells);
};

void observe_scalar(void* context, const float* scalar_end, const float* flux_x,
                    const float* flux_y, const float* flux_z, const float* limiter)
{
    auto& observed = *static_cast<ScalarObservation*>(context);
    std::copy_n(scalar_end, cells, observed.scalar_end.begin());
    std::copy_n(flux_x, cells, observed.flux[0].begin());
    std::copy_n(flux_y, cells, observed.flux[1].begin());
    std::copy_n(flux_z, cells, observed.flux[2].begin());
    std::copy_n(limiter, cells, observed.limiter.begin());
}

double mapped_scalar_closure_residual(const ScalarTransportCase& scalar,
                                      const ScalarObservation& observed)
{
    const long double volume = static_cast<long double>(dx) * dy * scalar.dzeta;
    long double beginning = 0.0L;
    long double ending = 0.0L;
    for (int i = 1; i < nx - 1; ++i)
        for (int j = 1; j < ny - 1; ++j)
            for (int k = 0; k < nz; ++k)
            {
                const auto n = at(i, j, k);
                const long double cell_volume = volume * scalar.jacobian[n];
                beginning += cell_volume * scalar.density_begin[n] * scalar.scalar_begin[n];
                ending += cell_volume * scalar.density_end[n] * observed.scalar_end[n];
            }
    const auto limited = [&](float value, std::size_t low, std::size_t high)
    {
        if (!scalar.positive_definite || value == 0.0f)
            return static_cast<long double>(value);
        return static_cast<long double>(value) * observed.limiter[value > 0.0f ? low : high];
    };
    long double outward_rate = 0.0L;
    const long double area_x = static_cast<long double>(dy) * scalar.dzeta;
    const long double area_y = static_cast<long double>(dx) * scalar.dzeta;
    for (int j = 1; j < ny - 1; ++j)
        for (int k = 0; k < nz; ++k)
        {
            const auto west = at(0, j, k);
            const auto east = at(nx - 2, j, k);
            outward_rate += area_x * (limited(observed.flux[0][east], east, at(nx - 1, j, k)) -
                                      limited(observed.flux[0][west], west, at(1, j, k)));
        }
    for (int i = 1; i < nx - 1; ++i)
        for (int k = 0; k < nz; ++k)
        {
            const auto south = at(i, 0, k);
            const auto north = at(i, ny - 2, k);
            outward_rate += area_y * (limited(observed.flux[1][north], north, at(i, ny - 1, k)) -
                                      limited(observed.flux[1][south], south, at(i, 1, k)));
        }
    return static_cast<double>(((ending - beginning) + scalar.step * outward_rate) /
                               std::max(std::abs(beginning), 1.0L));
}

struct FinalStateCase
{
    std::array<std::vector<float>, 6> input;
    std::array<std::vector<float>, 5> output;
    MappedTerrainFinalStateReport report{7, 9, false};

    FinalStateCase(const CanonicalCase& carrier)
    {
        for (auto& field : input)
            field.resize(cells);
        for (auto& field : output)
            field.assign(cells, -51.0f);
        for (int i = 0; i < nx; ++i)
            for (int j = 0; j < ny; ++j)
                for (int k = 0; k < nz; ++k)
                {
                    const auto n = at(i, j, k);
                    input[0][n] = carrier.u[i][j][k];
                    input[1][n] = carrier.v[i][j][k];
                    input[2][n] = carrier.w[i][j][k];
                    input[3][n] = carrier.rho[i][j][k];
                    input[4][n] = carrier.p[i][j][k];
                    input[5][n] = 299.0f + 0.7f * k + 0.05f * i - 0.03f * j;
                }
    }

    MappedTerrainFinalStateDispatch dispatch()
    {
        return {input[0].data(), input[1].data(), input[2].data(), input[3].data(),
                input[4].data(), input[5].data(), output[0].data(), output[1].data(),
                output[2].data(), output[3].data(), output[4].data(), nx, ny, nz, &report};
    }
};

struct FinalExpected
{
    std::array<std::vector<float>, 5> field;
    tmv::dynamics_eos::PressureProjectionEnergyReportV1 eos;
    tmv::FinalBoundaryEnergyReportV1 boundary;
};

FinalExpected canonical_final_expected(const CanonicalCase& carrier,
                                       const FinalStateCase& state)
{
    configure_scalar_oracle_grid(carrier);
    global_lateral_bc_west = tmv::init::config::LateralBC::Periodic;
    global_lateral_bc_east = tmv::init::config::LateralBC::Periodic;
    global_lateral_bc_south = tmv::init::config::LateralBC::Periodic;
    global_lateral_bc_north = tmv::init::config::LateralBC::Periodic;
    auto assign = [](Field3D& field, const std::vector<float>& values)
    {
        field.resize(nx, ny, nz, 0.0f);
        std::copy(values.begin(), values.end(), field.data());
    };
    assign(u, state.input[0]);
    assign(v, state.input[1]);
    assign(w, state.input[2]);
    assign(rho, state.input[3]);
    assign(p, state.input[4]);
    assign(theta, state.input[5]);
    for (Field3D* field : {&qv, &qc, &qr, &qi, &qs, &qg, &qh, &tke})
        field->resize(nx, ny, nz, 0.0f);
    global_grid_metrics.dx = dx;
    global_grid_metrics.dy = dy;
    global_grid_metrics.dz.assign(nz, carrier.metrics.dzeta);
    global_grid_metrics.terrain_metrics_active = true;
    global_grid_metrics.terrain_metrics = &carrier.metrics;

    FinalExpected expected;
    expected.eos = tmv::dynamics_eos::apply_pressure_projection(global_simulation_state());
    auto boundary = create_cartesian_cgrid_bc_scheme();
    expected.boundary = boundary->apply_full_attributed();
    const Field3D* fields[] = {&u, &v, &w, &rho, &p};
    for (std::size_t index = 0; index < expected.field.size(); ++index)
        expected.field[index].assign(fields[index]->data(), fields[index]->data() + fields[index]->size());
    return expected;
}

bool final_outputs_unchanged(const FinalStateCase& after, const FinalStateCase& before)
{
    for (std::size_t index = 0; index < after.output.size(); ++index)
        if (std::memcmp(after.output[index].data(), before.output[index].data(), cells * sizeof(float)) != 0)
            return false;
    return std::memcmp(&after.report, &before.report, sizeof(after.report)) == 0;
}

SimulationStorageState mapped_transaction_state()
{
    CanonicalCase canonical;
    SimulationStorageState storage{{nx, ny, nz, dx, canonical.metrics.dzeta, 0.05, dy}};
    storage.grid.geometry.initialize(
        nx, ny, nz, dx, canonical.metrics.dzeta, dy,
        CoordinateSystem::Cartesian, StaggerType::CGrid);
    storage.runtime.terrain.config.scheme_id = "bell";
    storage.runtime.terrain.metrics = canonical.metrics;

    auto initialize = [](Field3D& field, float value)
    {
        field.resize(nx, ny, nz, value);
    };
    for (Field3D* field : {&storage.fields.qv, &storage.fields.qc, &storage.fields.qr,
                           &storage.fields.qi, &storage.fields.qs,
                           &storage.fields.qg, &storage.fields.qh})
        initialize(*field, 0.0f);
    initialize(storage.fields.tracer, 0.0f);
    initialize(storage.fields.tke, 0.1f);
    for (Field3D* field : {&storage.tendencies.dtheta_dt_rad,
                           &storage.tendencies.dtheta_dt_pbl,
                           &storage.tendencies.dqv_dt_pbl,
                           &storage.tendencies.du_dt_pbl,
                           &storage.tendencies.dv_dt_pbl,
                           &storage.tendencies.dtke_dt_pbl,
                           &storage.tendencies.dqc_dt_pbl,
                           &storage.tendencies.dqi_dt_pbl})
        initialize(*field, 0.0f);
    initialize(storage.fields.rho, 0.0f);
    initialize(storage.fields.p, 0.0f);
    initialize(storage.fields.u, 0.0f);
    initialize(storage.fields.v, 0.0f);
    initialize(storage.fields.w, 0.0f);
    initialize(storage.fields.theta, 0.0f);

    storage.base.rho0.resize(nz);
    storage.base.p0.resize(nz);
    storage.base.qv0.assign(nz, 0.0);
    storage.base.theta0.resize(nz);
    storage.base.u0.assign(nz, 0.0);
    storage.base.v0.assign(nz, 0.0);
    for (int k = 0; k < nz; ++k)
    {
        const double reference_pressure = 96000.0 - 1250.0 * k;
        const double reference_theta = 300.0 + 0.65 * k;
        storage.base.p0[static_cast<std::size_t>(k)] = reference_pressure;
        storage.base.theta0[static_cast<std::size_t>(k)] = reference_theta;
        storage.base.rho0[static_cast<std::size_t>(k)] =
            physical_constants::reference_pressure_pa /
            (physical_constants::gas_constant_dry_air_jkgk * reference_theta) *
            std::pow(reference_pressure / physical_constants::reference_pressure_pa,
                     1.0 / physical_constants::dry_air_gamma);
    }
    for (int i = 0; i < nx; ++i)
        for (int j = 0; j < ny; ++j)
            for (int k = 0; k < nz; ++k)
            {
                const float theta_value = static_cast<float>(
                    storage.base.theta0[static_cast<std::size_t>(k)] +
                    0.015 * std::sin(0.3 * i + 0.2 * j + 0.4 * k));
                const float rho_value = static_cast<float>(
                    storage.base.rho0[static_cast<std::size_t>(k)] +
                    8.0e-6 * std::sin(0.2 * i - 0.25 * j + 0.35 * k));
                storage.fields.rho[i][j][k] = rho_value;
                storage.fields.theta[i][j][k] = theta_value;
                storage.fields.p[i][j][k] = clamp_pressure_pa(static_cast<float>(
                    tmv::dynamics_eos::pressure_fp64(rho_value, theta_value)));
                storage.fields.u[i][j][k] = 1.4f + 0.008f * i - 0.004f * j;
                storage.fields.v[i][j][k] = -0.5f + 0.003f * i + 0.006f * j;
                storage.fields.w[i][j][k] = k == nz - 1 ? 0.0f :
                    0.002f * std::sin(0.2f * i + 0.3f * j + 0.4f * k);
                storage.fields.tracer[i][j][k] =
                    0.01f + 0.001f * std::sin(0.4f * i + 0.3f * j + 0.2f * k);
                storage.fields.tke[i][j][k] =
                    0.1f + 0.002f * std::cos(0.3f * i - 0.2f * j + 0.4f * k);
            }
    return storage;
}

using NineFields = std::array<std::vector<float>, 9>;

NineFields mapped_public_fields(const SimulationStorageState& storage)
{
    NineFields fields;
    const Field3D* source[] = {&storage.fields.u, &storage.fields.v, &storage.fields.w,
                               &storage.fields.rho, &storage.fields.p,
                               &storage.fields.theta, &storage.fields.qv,
                               &storage.fields.tracer, &storage.fields.tke};
    for (std::size_t index = 0; index < fields.size(); ++index)
        fields[index].assign(source[index]->data(), source[index]->data() + source[index]->size());
    return fields;
}

bool same_mapped_report(const tmv::terrain_native::MappedCoreStepReport& a,
                        const tmv::terrain_native::MappedCoreStepReport& b)
{
    const auto same_acoustic = [](const auto& x, const auto& y)
    {
        return x.continuity_eos_dry_internal_change_j == y.continuity_eos_dry_internal_change_j &&
            x.continuity_eos_water_sensible_change_j == y.continuity_eos_water_sensible_change_j &&
            x.continuity_eos_phase_reference_change_j == y.continuity_eos_phase_reference_change_j &&
            x.continuity_carrier_kinetic_change_j == y.continuity_carrier_kinetic_change_j &&
            x.continuity_gravitational_change_j == y.continuity_gravitational_change_j &&
            x.continuity_subgrid_tke_change_j == y.continuity_subgrid_tke_change_j &&
            x.continuity_carrier_requested_change_j == y.continuity_carrier_requested_change_j &&
            x.continuity_carrier_outward_boundary_transport_j == y.continuity_carrier_outward_boundary_transport_j &&
            x.continuity_carrier_interior_conversion_j == y.continuity_carrier_interior_conversion_j &&
            x.continuity_carrier_storage_rounding_j == y.continuity_carrier_storage_rounding_j &&
            x.slow_momentum_advection_requested_linear_work_j == y.slow_momentum_advection_requested_linear_work_j &&
            x.kinetic_transport_interior_conversion_j == y.kinetic_transport_interior_conversion_j &&
            x.kinetic_transport_closure_residual_j == y.kinetic_transport_closure_residual_j &&
            x.kinetic_transport_complete == y.kinetic_transport_complete &&
            x.pressure_gradient_requested_linear_work_j == y.pressure_gradient_requested_linear_work_j &&
            x.pressure_gradient_finite_step_correction_j == y.pressure_gradient_finite_step_correction_j &&
            x.pressure_gradient_storage_rounding_j == y.pressure_gradient_storage_rounding_j &&
            x.pressure_gradient_committed_work_j == y.pressure_gradient_committed_work_j &&
            x.boundary_committed_work_j == y.boundary_committed_work_j &&
            x.boundary_radiative_requested_linear_work_j == y.boundary_radiative_requested_linear_work_j &&
            x.boundary_radiative_finite_step_correction_j == y.boundary_radiative_finite_step_correction_j &&
            x.boundary_radiative_storage_rounding_j == y.boundary_radiative_storage_rounding_j &&
            x.boundary_radiative_committed_work_j == y.boundary_radiative_committed_work_j &&
            x.boundary_constraint_history_committed_work_j == y.boundary_constraint_history_committed_work_j &&
            x.boundary_application_count == y.boundary_application_count &&
            x.boundary_complete == y.boundary_complete &&
            x.committed_inventory_change_j == y.committed_inventory_change_j &&
            x.substep_count == y.substep_count && x.complete == y.complete;
    };
    const auto same_slow = [](const auto& x, const auto& y)
    {
        return x.advection_requested_linear_work_j == y.advection_requested_linear_work_j &&
            x.nonadvection_requested_linear_work_j == y.nonadvection_requested_linear_work_j &&
            x.advection_finite_step_correction_j == y.advection_finite_step_correction_j &&
            x.nonadvection_finite_step_correction_j == y.nonadvection_finite_step_correction_j &&
            x.coupled_finite_step_cross_work_j == y.coupled_finite_step_cross_work_j &&
            x.storage_rounding_j == y.storage_rounding_j &&
            x.committed_kinetic_change_j == y.committed_kinetic_change_j &&
            x.closure_residual_j == y.closure_residual_j &&
            x.pressure_dry_internal_change_j == y.pressure_dry_internal_change_j &&
            x.pressure_water_sensible_change_j == y.pressure_water_sensible_change_j &&
            x.committed_pressure_energy_change_j == y.committed_pressure_energy_change_j &&
            x.complete == y.complete;
    };
    return a.acoustic_substeps == b.acoustic_substeps &&
        a.mass_continuity_dispatches == b.mass_continuity_dispatches &&
        a.slow_tendency_dispatches == b.slow_tendency_dispatches &&
        a.slow_average_dispatches == b.slow_average_dispatches &&
        a.fast_momentum_dispatches == b.fast_momentum_dispatches &&
        a.state_update_dispatches == b.state_update_dispatches &&
        a.flux_accumulation_dispatches == b.flux_accumulation_dispatches &&
        a.scalar_flux_dispatches == b.scalar_flux_dispatches &&
        a.scalar_limiter_dispatches == b.scalar_limiter_dispatches &&
        a.scalar_update_dispatches == b.scalar_update_dispatches &&
        a.boundary_dispatches == b.boundary_dispatches &&
        a.completed_backend_dispatches == b.completed_backend_dispatches &&
        a.accumulated_time_weight == b.accumulated_time_weight &&
        same_acoustic(a.acoustic, b.acoustic) &&
        same_slow(a.slow_momentum, b.slow_momentum) &&
        a.final_eos.dry_internal_change_j == b.final_eos.dry_internal_change_j &&
        a.final_eos.water_sensible_change_j == b.final_eos.water_sensible_change_j &&
        a.final_eos.committed_energy_change_j == b.final_eos.committed_energy_change_j &&
        a.final_eos.closure_residual_j == b.final_eos.closure_residual_j &&
        a.final_eos.complete == b.final_eos.complete &&
        a.final_boundary.constraint_committed_work_j == b.final_boundary.constraint_committed_work_j &&
        a.final_boundary.committed_energy_change_j == b.final_boundary.committed_energy_change_j &&
        a.final_boundary.closure_residual_j == b.final_boundary.closure_residual_j &&
        a.final_boundary.complete == b.final_boundary.complete &&
        a.final_pressure_uses_canonical_fp64 == b.final_pressure_uses_canonical_fp64 &&
        a.complete == b.complete;
}

SimulationStateView aliased_uv_view(SimulationStorageState& storage)
{
    auto source = make_simulation_state_view(storage);
    SimulationStateView view{
        source.grid,
        {source.fields.rho, source.fields.p, source.fields.u, source.fields.w,
         source.fields.u, source.fields.tracer, source.fields.theta,
         source.fields.qv, source.fields.qc, source.fields.qr,
         source.fields.qi, source.fields.qs, source.fields.qh,
         source.fields.qg, source.fields.tke},
        source.base, source.tendencies, source.coriolis_f, source.time_s,
        source.surface};
    view.terrain_runtime = source.terrain_runtime;
    view.compute_runtime = nullptr;
    view.configuration = source.configuration;
    view.diagnostics = source.diagnostics;
    view.dynamics_runtime = source.dynamics_runtime;
    view.runtime_owner = source.runtime_owner;
    view.domain_frame = source.domain_frame;
    return view;
}
} // namespace

TEST_CASE("Vulkan mapped Bell mass continuity matches canonical production operators atomically",
          "[vulkan][device][terrain][mapped_mass_continuity]")
{
    std::string error;
    global_compute_backend_config.backend = "vulkan";
    global_compute_backend_config.allow_fallback = false;
    if (!initialize_compute_backend_runtime(error))
    {
        WARN("Vulkan unavailable: " << error);
        return;
    }
    ComputeBackend* backend = mutable_compute_backend();
    TerrainComputeKernels* kernels = backend != nullptr ? backend->terrain_kernels() : nullptr;
    REQUIRE(kernels != nullptr);
    REQUIRE(kernels->supports_mapped_mass_continuity());

    CanonicalCase actual;
    const Expected expected = canonical_expected(actual);
    auto dispatch = actual.dispatch();
    REQUIRE(kernels->dispatch_mapped_mass_continuity(dispatch));
    REQUIRE(actual.report.complete);
    REQUIRE(actual.report.ground_tangency_dispatch_count == 1);
    REQUIRE(actual.report.mass_flux_continuity_dispatch_count == 1);
    require_close(expected.ground_w, actual.ground_w, 2.0e-5f, "terrain ground tangent velocity");
    for (int i = 1; i < nx - 1; ++i)
        for (int j = 1; j < ny - 1; ++j)
        {
            const auto n = at(i, j, 0);
            const float ground_u = 0.5f * (actual.u[i - 1][j][0] + actual.u[i][j][0]);
            const float ground_v = 0.5f * (actual.v[i][j - 1][0] + actual.v[i][j][0]);
            REQUIRE(std::abs(actual.ground_w[n] + actual.ground_metric_x[n] * ground_u +
                             actual.ground_metric_y[n] * ground_v) <= 2.0e-5f);
        }
    require_close(expected.drho, actual.drho, 2.0e-6f, "mapped mass continuity");
    require_close(expected.dp, actual.dp, 2.0e-2f, "mapped split pressure tendency");
    require_close(expected.flux_x, actual.flux_x, 2.0e-3f, "mapped x mass flux");
    require_close(expected.flux_y, actual.flux_y, 2.0e-3f, "mapped y mass flux");
    require_close(expected.flux_zeta, actual.flux_zeta, 2.0e-5f, "mapped zeta mass flux");
    REQUIRE(backend->successful_kernel_dispatch_count(
        nimbus::compute::KernelId::MappedTerrainMassContinuity) == 2);

    CanonicalCase invalid;
    const CanonicalCase invalid_before = invalid;
    auto invalid_dispatch = invalid.dispatch();
    invalid_dispatch.nx = 2;
    REQUIRE_FALSE(kernels->dispatch_mapped_mass_continuity(invalid_dispatch));
    REQUIRE(outputs_unchanged(invalid, invalid_before));

    CanonicalCase oversized;
    const CanonicalCase oversized_before = oversized;
    auto oversized_dispatch = oversized.dispatch();
    oversized_dispatch.nx = std::numeric_limits<int>::max();
    oversized_dispatch.ny = std::numeric_limits<int>::max();
    oversized_dispatch.nz = std::numeric_limits<int>::max();
    REQUIRE_FALSE(kernels->dispatch_mapped_mass_continuity(oversized_dispatch));
    REQUIRE(outputs_unchanged(oversized, oversized_before));

    CanonicalCase malformed;
    malformed.jacobian[at(3, 3, 2)] = std::numeric_limits<float>::quiet_NaN();
    const CanonicalCase malformed_before = malformed;
    auto malformed_dispatch = malformed.dispatch();
    REQUIRE_FALSE(kernels->dispatch_mapped_mass_continuity(malformed_dispatch));
    REQUIRE(outputs_unchanged(malformed, malformed_before));

    CanonicalCase late_failure;
    late_failure.metric_x[at(3, 3, 2)] = std::numeric_limits<float>::quiet_NaN();
    const CanonicalCase late_before = late_failure;
    auto late_dispatch = late_failure.dispatch();
    REQUIRE_FALSE(kernels->dispatch_mapped_mass_continuity(late_dispatch));
    REQUIRE(outputs_unchanged(late_failure, late_before));

    CanonicalCase observer_failure;
    const CanonicalCase observer_before = observer_failure;
    auto observer_dispatch = observer_failure.dispatch();
    observer_dispatch.completed_observer = +[](void*, const float*, const float*, const float*,
                                                const float*, const float*, const float*)
    { throw std::runtime_error("reject mapped continuity candidate"); };
    REQUIRE_THROWS_WITH(kernels->dispatch_mapped_mass_continuity(observer_dispatch),
                        "reject mapped continuity candidate");
    REQUIRE(outputs_unchanged(observer_failure, observer_before));
    REQUIRE(backend->successful_kernel_dispatch_count(
        nimbus::compute::KernelId::MappedTerrainMassContinuity) == 6);
    shutdown_compute_backend_runtime();
}

TEST_CASE("Vulkan mapped Bell fast pressure force matches canonical production operators atomically",
          "[vulkan][device][terrain][mapped_fast_momentum]")
{
    std::string error;
    global_compute_backend_config.backend = "vulkan";
    global_compute_backend_config.allow_fallback = false;
    if (!initialize_compute_backend_runtime(error))
    {
        WARN("Vulkan unavailable: " << error);
        return;
    }
    ComputeBackend* backend = mutable_compute_backend();
    TerrainComputeKernels* kernels = backend != nullptr ? backend->terrain_kernels() : nullptr;
    REQUIRE(kernels != nullptr);
    REQUIRE(kernels->supports_mapped_fast_momentum());

    FastMomentumCase actual;
    const auto expected = canonical_fast_momentum_expected(actual);
    auto dispatch = actual.dispatch();
    REQUIRE(kernels->dispatch_mapped_fast_momentum(dispatch));
    REQUIRE(actual.report.complete);
    REQUIRE(actual.report.pressure_force_dispatch_count == 1);
    require_close(expected.du, actual.du, 5.0e-5f, "mapped x pressure force");
    require_close(expected.dv, actual.dv, 5.0e-5f, "mapped y pressure force");
    require_close(expected.dw, actual.dw, 5.0e-5f, "mapped z pressure force");
    REQUIRE(backend->successful_kernel_dispatch_count(
        nimbus::compute::KernelId::MappedTerrainFastMomentum) == 1);

    FastMomentumCase oversized;
    const FastMomentumCase oversized_before = oversized;
    auto oversized_dispatch = oversized.dispatch();
    oversized_dispatch.nx = std::numeric_limits<int>::max();
    oversized_dispatch.ny = std::numeric_limits<int>::max();
    oversized_dispatch.nz = std::numeric_limits<int>::max();
    REQUIRE_FALSE(kernels->dispatch_mapped_fast_momentum(oversized_dispatch));
    REQUIRE(momentum_outputs_unchanged(oversized, oversized_before));

    FastMomentumCase malformed;
    malformed.reference_pressure[2] = std::numeric_limits<float>::quiet_NaN();
    const FastMomentumCase malformed_before = malformed;
    auto malformed_dispatch = malformed.dispatch();
    REQUIRE_FALSE(kernels->dispatch_mapped_fast_momentum(malformed_dispatch));
    REQUIRE(momentum_outputs_unchanged(malformed, malformed_before));

    FastMomentumCase late_failure;
    late_failure.height[at(3, 3, 2)] = late_failure.height[at(3, 3, 1)];
    const FastMomentumCase late_before = late_failure;
    auto late_dispatch = late_failure.dispatch();
    REQUIRE_FALSE(kernels->dispatch_mapped_fast_momentum(late_dispatch));
    REQUIRE(momentum_outputs_unchanged(late_failure, late_before));

    FastMomentumCase observer_failure;
    const FastMomentumCase observer_before = observer_failure;
    auto observer_dispatch = observer_failure.dispatch();
    observer_dispatch.completed_observer = +[](void*, const float*, const float*, const float*)
    { throw std::runtime_error("reject mapped pressure-force candidate"); };
    REQUIRE_THROWS_WITH(kernels->dispatch_mapped_fast_momentum(observer_dispatch),
                        "reject mapped pressure-force candidate");
    REQUIRE(momentum_outputs_unchanged(observer_failure, observer_before));
    REQUIRE(backend->successful_kernel_dispatch_count(
        nimbus::compute::KernelId::MappedTerrainFastMomentum) == 3);
    shutdown_compute_backend_runtime();
}

TEST_CASE("Vulkan mapped Bell slow tendencies match canonical production operators atomically",
          "[vulkan][device][terrain][mapped_slow_tendencies]")
{
    std::string error;
    global_compute_backend_config.backend = "vulkan";
    global_compute_backend_config.allow_fallback = false;
    if (!initialize_compute_backend_runtime(error))
    {
        WARN("Vulkan unavailable: " << error);
        return;
    }
    ComputeBackend* backend = mutable_compute_backend();
    TerrainComputeKernels* kernels = backend != nullptr ? backend->terrain_kernels() : nullptr;
    REQUIRE(kernels != nullptr);
    REQUIRE(kernels->supports_mapped_mass_continuity());
    REQUIRE(kernels->supports_mapped_slow_tendencies());

    SlowTendencyCase actual;
    auto ground_dispatch = actual.state.dispatch();
    REQUIRE(kernels->dispatch_mapped_mass_continuity(ground_dispatch));
    const auto expected = canonical_slow_expected(actual);
    auto dispatch = actual.dispatch();
    REQUIRE(kernels->dispatch_mapped_slow_tendencies(dispatch));
    REQUIRE(actual.report.complete);
    REQUIRE(actual.report.slow_tendency_dispatch_count == 1);
    REQUIRE(actual.report.advection_attribution_complete);
    const char* labels[] = {"mapped slow u", "mapped slow v", "mapped slow w", "mapped slow rho",
                            "mapped slow pressure", "mapped advection u", "mapped advection v", "mapped advection w"};
    const float tolerances[] = {2.0e-5f, 2.0e-5f, 2.0e-5f, 0.0f, 5.0e-2f, 2.0e-5f, 2.0e-5f, 2.0e-5f};
    for (std::size_t field = 0; field < actual.output.size(); ++field)
        require_close(expected[field], actual.output[field], tolerances[field], labels[field]);
    REQUIRE(backend->successful_kernel_dispatch_count(
        nimbus::compute::KernelId::MappedTerrainSlowTendencies) == 1);

    SlowTendencyCase disabled_currencies;
    disabled_currencies.state.ground_w = actual.state.ground_w;
    std::fill(disabled_currencies.qv.begin(), disabled_currencies.qv.end(),
              std::numeric_limits<float>::quiet_NaN());
    std::fill(disabled_currencies.reference_qv.begin(), disabled_currencies.reference_qv.end(),
              std::numeric_limits<float>::quiet_NaN());
    std::fill(disabled_currencies.loading.begin(), disabled_currencies.loading.end(),
              std::numeric_limits<float>::quiet_NaN());
    auto disabled_dispatch = disabled_currencies.dispatch();
    REQUIRE(kernels->dispatch_mapped_slow_tendencies(disabled_dispatch));
    for (std::size_t field = 0; field < actual.output.size(); ++field)
        REQUIRE(std::memcmp(disabled_currencies.output[field].data(), actual.output[field].data(),
                            cells * sizeof(float)) == 0);

    SlowTendencyCase coupled;
    coupled.has_moisture = true;
    coupled.has_loading = true;
    for (int k = 0; k < nz; ++k)
        coupled.reference_qv[static_cast<std::size_t>(k)] = 0.006f - 0.0005f * k;
    for (std::size_t n = 0; n < cells; ++n)
    {
        coupled.qv[n] = coupled.reference_qv[n % nz] + 0.0007f;
        coupled.loading[n] = 0.001f + 0.00001f * static_cast<float>(n % 7);
    }
    auto coupled_ground_dispatch = coupled.state.dispatch();
    REQUIRE(kernels->dispatch_mapped_mass_continuity(coupled_ground_dispatch));
    const auto coupled_expected = canonical_slow_expected(coupled);
    auto coupled_dispatch = coupled.dispatch();
    REQUIRE(kernels->dispatch_mapped_slow_tendencies(coupled_dispatch));
    require_close(coupled_expected[2], coupled.output[2], 2.0e-5f,
                  "mapped slow moisture buoyancy and loading");
    REQUIRE(std::memcmp(coupled.output[2].data(), actual.output[2].data(), cells * sizeof(float)) != 0);

    SlowTendencyCase oversized;
    oversized.state.ground_w = actual.state.ground_w;
    const SlowTendencyCase oversized_before = oversized;
    auto oversized_dispatch = oversized.dispatch();
    oversized_dispatch.nx = std::numeric_limits<int>::max();
    oversized_dispatch.ny = std::numeric_limits<int>::max();
    oversized_dispatch.nz = std::numeric_limits<int>::max();
    REQUIRE_FALSE(kernels->dispatch_mapped_slow_tendencies(oversized_dispatch));
    REQUIRE(slow_outputs_unchanged(oversized, oversized_before));

    SlowTendencyCase malformed_rho;
    malformed_rho.state.ground_w = actual.state.ground_w;
    malformed_rho.state.rho[3][3][2] = 0.0f;
    const SlowTendencyCase malformed_rho_before = malformed_rho;
    auto malformed_rho_dispatch = malformed_rho.dispatch();
    REQUIRE_FALSE(kernels->dispatch_mapped_slow_tendencies(malformed_rho_dispatch));
    REQUIRE(slow_outputs_unchanged(malformed_rho, malformed_rho_before));

    SlowTendencyCase malformed_base;
    malformed_base.state.ground_w = actual.state.ground_w;
    malformed_base.reference_pressure[2] = std::numeric_limits<float>::quiet_NaN();
    const SlowTendencyCase malformed_base_before = malformed_base;
    auto malformed_base_dispatch = malformed_base.dispatch();
    REQUIRE_FALSE(kernels->dispatch_mapped_slow_tendencies(malformed_base_dispatch));
    REQUIRE(slow_outputs_unchanged(malformed_base, malformed_base_before));

    SlowTendencyCase late_failure;
    late_failure.state.ground_w = actual.state.ground_w;
    late_failure.height[at(3, 3, 2)] = late_failure.height[at(3, 3, 1)];
    const SlowTendencyCase late_before = late_failure;
    auto late_dispatch = late_failure.dispatch();
    REQUIRE_FALSE(kernels->dispatch_mapped_slow_tendencies(late_dispatch));
    REQUIRE(slow_outputs_unchanged(late_failure, late_before));

    SlowTendencyCase observer_failure;
    observer_failure.state.ground_w = actual.state.ground_w;
    const SlowTendencyCase observer_before = observer_failure;
    auto observer_dispatch = observer_failure.dispatch();
    observer_dispatch.completed_observer = +[](void*, const float*, const float*, const float*,
        const float*, const float*, const float*, const float*, const float*)
    { throw std::runtime_error("reject mapped slow candidate"); };
    REQUIRE_THROWS_WITH(kernels->dispatch_mapped_slow_tendencies(observer_dispatch),
                        "reject mapped slow candidate");
    REQUIRE(slow_outputs_unchanged(observer_failure, observer_before));
    REQUIRE(backend->successful_kernel_dispatch_count(
        nimbus::compute::KernelId::MappedTerrainSlowTendencies) == 5);
    shutdown_compute_backend_runtime();
}

TEST_CASE("Vulkan mapped Bell retained scalars match canonical production transport atomically",
          "[vulkan][device][terrain][mapped_scalar_transport]")
{
    std::string error;
    global_compute_backend_config.backend = "vulkan";
    global_compute_backend_config.allow_fallback = false;
    if (!initialize_compute_backend_runtime(error))
    {
        WARN("Vulkan unavailable: " << error);
        return;
    }
    ComputeBackend* backend = mutable_compute_backend();
    TerrainComputeKernels* kernels = backend != nullptr ? backend->terrain_kernels() : nullptr;
    REQUIRE(kernels != nullptr);
    REQUIRE(kernels->supports_mapped_mass_continuity());
    REQUIRE(kernels->supports_mapped_scalar_transport());

    CanonicalCase carrier;
    auto continuity_dispatch = carrier.dispatch();
    REQUIRE(kernels->dispatch_mapped_mass_continuity(continuity_dispatch));

    ScalarTransportCase theta_case(carrier, false);
    const auto theta_expected = canonical_scalar_expected(carrier, theta_case);
    ScalarObservation theta_observed;
    auto theta_dispatch = theta_case.dispatch();
    theta_dispatch.completed_observer = observe_scalar;
    theta_dispatch.completed_observer_context = &theta_observed;
    REQUIRE(kernels->dispatch_mapped_scalar_transport(theta_dispatch));
    REQUIRE(theta_case.report.complete);
    REQUIRE(theta_case.report.scalar_flux_dispatch_count == 1);
    REQUIRE(theta_case.report.positive_definite_limiter_dispatch_count == 0);
    REQUIRE(theta_case.report.scalar_update_dispatch_count == 1);
    REQUIRE(std::memcmp(theta_case.scalar_end.data(), theta_observed.scalar_end.data(),
                        cells * sizeof(float)) == 0);
    require_close(theta_expected, theta_case.scalar_end, 1.0e-4f,
                  "mapped retained theta transport");
    const double theta_closure = mapped_scalar_closure_residual(theta_case, theta_observed);
    INFO("mapped retained theta normalized closure=" << theta_closure);
    REQUIRE(std::abs(theta_closure) <= 2.0e-7);

    ScalarTransportCase qv_case(carrier, true);
    const auto qv_expected = canonical_scalar_expected(carrier, qv_case);
    ScalarObservation qv_observed;
    auto qv_dispatch = qv_case.dispatch();
    qv_dispatch.completed_observer = observe_scalar;
    qv_dispatch.completed_observer_context = &qv_observed;
    REQUIRE(kernels->dispatch_mapped_scalar_transport(qv_dispatch));
    REQUIRE(qv_case.report.complete);
    REQUIRE(qv_case.report.scalar_flux_dispatch_count == 1);
    REQUIRE(qv_case.report.positive_definite_limiter_dispatch_count == 1);
    REQUIRE(qv_case.report.scalar_update_dispatch_count == 1);
    require_close(qv_expected, qv_case.scalar_end, 2.0e-6f,
                  "mapped retained positive-definite qv transport");
    REQUIRE(*std::min_element(qv_case.scalar_end.begin(), qv_case.scalar_end.end()) >= 0.0f);
    REQUIRE(*std::min_element(qv_observed.limiter.begin(), qv_observed.limiter.end()) < 1.0f);
    const double qv_closure = mapped_scalar_closure_residual(qv_case, qv_observed);
    INFO("mapped retained qv normalized closure=" << qv_closure);
    REQUIRE(std::abs(qv_closure) <= 2.0e-7);

    ScalarTransportCase oversized(carrier, false);
    const ScalarTransportCase oversized_before = oversized;
    auto oversized_dispatch = oversized.dispatch();
    oversized_dispatch.nx = std::numeric_limits<int>::max();
    oversized_dispatch.ny = std::numeric_limits<int>::max();
    oversized_dispatch.nz = std::numeric_limits<int>::max();
    REQUIRE_FALSE(kernels->dispatch_mapped_scalar_transport(oversized_dispatch));
    REQUIRE(scalar_outputs_unchanged(oversized, oversized_before));

    ScalarTransportCase malformed_density(carrier, false);
    malformed_density.density_end[at(3, 3, 2)] = 0.0f;
    const ScalarTransportCase malformed_density_before = malformed_density;
    auto malformed_density_dispatch = malformed_density.dispatch();
    REQUIRE_FALSE(kernels->dispatch_mapped_scalar_transport(malformed_density_dispatch));
    REQUIRE(scalar_outputs_unchanged(malformed_density, malformed_density_before));

    ScalarTransportCase malformed_metric(carrier, false);
    malformed_metric.jacobian[at(3, 3, 2)] = std::numeric_limits<float>::quiet_NaN();
    const ScalarTransportCase malformed_metric_before = malformed_metric;
    auto malformed_metric_dispatch = malformed_metric.dispatch();
    REQUIRE_FALSE(kernels->dispatch_mapped_scalar_transport(malformed_metric_dispatch));
    REQUIRE(scalar_outputs_unchanged(malformed_metric, malformed_metric_before));

    ScalarTransportCase observer_failure(carrier, false);
    const ScalarTransportCase observer_before = observer_failure;
    auto observer_dispatch = observer_failure.dispatch();
    observer_dispatch.completed_observer = +[](void*, const float*, const float*, const float*,
                                                const float*, const float*)
    { throw std::runtime_error("reject mapped scalar candidate"); };
    REQUIRE_THROWS_WITH(kernels->dispatch_mapped_scalar_transport(observer_dispatch),
                        "reject mapped scalar candidate");
    REQUIRE(scalar_outputs_unchanged(observer_failure, observer_before));

    REQUIRE(backend->successful_kernel_dispatch_count(
        nimbus::compute::KernelId::AdvectConservativeScalarFluxCartesianCgrid) == 3);
    REQUIRE(backend->successful_kernel_dispatch_count(
        nimbus::compute::KernelId::MappedTerrainScalarLimiter) == 1);
    REQUIRE(backend->successful_kernel_dispatch_count(
        nimbus::compute::KernelId::MappedTerrainScalarUpdate) == 3);
    shutdown_compute_backend_runtime();
}

TEST_CASE("Vulkan mapped Bell EOS and periodic rigid boundary match production atomically",
          "[vulkan][device][terrain][mapped_final_state]")
{
    std::string error;
    global_compute_backend_config.backend = "vulkan";
    global_compute_backend_config.allow_fallback = false;
    if (!initialize_compute_backend_runtime(error))
    {
        WARN("Vulkan unavailable: " << error);
        return;
    }
    ComputeBackend* backend = mutable_compute_backend();
    TerrainComputeKernels* kernels = backend != nullptr ? backend->terrain_kernels() : nullptr;
    REQUIRE(kernels != nullptr);
    REQUIRE(kernels->supports_mapped_final_state());

    CanonicalCase carrier;
    FinalStateCase actual(carrier);
    const FinalExpected expected = canonical_final_expected(carrier, actual);
    REQUIRE(expected.eos.complete);
    REQUIRE(expected.boundary.complete);
    auto dispatch = actual.dispatch();
    REQUIRE(kernels->dispatch_mapped_final_state(dispatch));
    REQUIRE(actual.report.complete);
    REQUIRE(actual.report.eos_dispatch_count == 1);
    REQUIRE(actual.report.boundary_dispatch_count == 1);
    require_close(expected.field[0], actual.output[0], 0.0f, "mapped periodic/top boundary u");
    require_close(expected.field[1], actual.output[1], 0.0f, "mapped periodic/top boundary v");
    require_close(expected.field[2], actual.output[2], 0.0f, "mapped periodic/top boundary w");
    require_close(expected.field[3], actual.output[3], 0.0f, "mapped periodic boundary rho");
    require_close(expected.field[4], actual.output[4], 5.0e-2f, "mapped native EOS and periodic pressure");
    for (int i = 0; i < nx; ++i)
        for (int j = 0; j < ny; ++j)
        {
            REQUIRE(actual.output[2][at(i, j, nz - 1)] == 0.0f);
            REQUIRE(actual.output[0][at(i, j, nz - 1)] == actual.output[0][at(i, j, nz - 2)]);
            REQUIRE(actual.output[1][at(i, j, nz - 1)] == actual.output[1][at(i, j, nz - 2)]);
        }

    FinalStateCase oversized(carrier);
    const FinalStateCase oversized_before = oversized;
    auto oversized_dispatch = oversized.dispatch();
    oversized_dispatch.nx = std::numeric_limits<int>::max();
    oversized_dispatch.ny = std::numeric_limits<int>::max();
    oversized_dispatch.nz = std::numeric_limits<int>::max();
    REQUIRE_FALSE(kernels->dispatch_mapped_final_state(oversized_dispatch));
    REQUIRE(final_outputs_unchanged(oversized, oversized_before));

    FinalStateCase malformed(carrier);
    malformed.input[5][at(3, 3, 2)] = std::numeric_limits<float>::quiet_NaN();
    const FinalStateCase malformed_before = malformed;
    auto malformed_dispatch = malformed.dispatch();
    REQUIRE_FALSE(kernels->dispatch_mapped_final_state(malformed_dispatch));
    REQUIRE(final_outputs_unchanged(malformed, malformed_before));

    FinalStateCase aliased(carrier);
    const FinalStateCase aliased_before = aliased;
    auto aliased_dispatch = aliased.dispatch();
    aliased_dispatch.v_final = aliased_dispatch.u_final;
    REQUIRE_FALSE(kernels->dispatch_mapped_final_state(aliased_dispatch));
    REQUIRE(final_outputs_unchanged(aliased, aliased_before));

    FinalStateCase observer_failure(carrier);
    const FinalStateCase observer_before = observer_failure;
    auto observer_dispatch = observer_failure.dispatch();
    observer_dispatch.completed_observer = +[](void*, const float*, const float*, const float*,
                                                const float*, const float*)
    { throw std::runtime_error("reject mapped final-state candidate"); };
    REQUIRE_THROWS_WITH(kernels->dispatch_mapped_final_state(observer_dispatch),
                        "reject mapped final-state candidate");
    REQUIRE(final_outputs_unchanged(observer_failure, observer_before));

    FinalStateCase boundary_only(carrier);
    for (std::size_t n = 0; n < cells; ++n)
        boundary_only.input[4][n] = static_cast<float>(tmv::dynamics_eos::pressure_fp64(
            boundary_only.input[3][n], boundary_only.input[5][n]));
    auto boundary_only_dispatch = boundary_only.dispatch();
    boundary_only_dispatch.apply_eos = false;
    REQUIRE(kernels->dispatch_mapped_final_state(boundary_only_dispatch));
    REQUIRE(boundary_only.report.eos_dispatch_count == 0);
    REQUIRE(boundary_only.report.boundary_dispatch_count == 1);
    for (int i = 1; i < nx - 1; ++i)
        for (int j = 1; j < ny - 1; ++j)
            for (int k = 0; k < nz; ++k)
                REQUIRE(boundary_only.output[4][at(i, j, k)] ==
                        boundary_only.input[4][at(i, j, k)]);
    REQUIRE(backend->successful_kernel_dispatch_count(
        nimbus::compute::KernelId::MappedTerrainEos) == 2);
    REQUIRE(backend->successful_kernel_dispatch_count(
        nimbus::compute::KernelId::MappedTerrainBoundary) == 3);
    shutdown_compute_backend_runtime();
}

TEST_CASE("Vulkan mapped Bell state commits preserve split-explicit production ordering atomically",
          "[vulkan][device][terrain][mapped_state_update]")
{
    std::string error;
    global_compute_backend_config.backend = "vulkan";
    global_compute_backend_config.allow_fallback = false;
    if (!initialize_compute_backend_runtime(error))
    {
        WARN("Vulkan unavailable: " << error);
        return;
    }
    ComputeBackend* backend = mutable_compute_backend();
    TerrainComputeKernels* kernels = backend->terrain_kernels();
    REQUIRE(kernels != nullptr);
    REQUIRE(kernels->supports_mapped_state_update());
    CanonicalCase carrier;
    for (const auto mode : {MappedTerrainStateUpdateMode::Slow,
                            MappedTerrainStateUpdateMode::ContinuityEos,
                            MappedTerrainStateUpdateMode::FastMomentum})
    {
        StateUpdateCase state(carrier);
        state.mode = mode;
        const auto before_dispatches = backend->successful_dispatch_count();
        REQUIRE(kernels->dispatch_mapped_state_update(state.dispatch()));
        REQUIRE(state.report.complete);
        REQUIRE(state.report.dispatch_count == 1);
        REQUIRE(state.report.mode == mode);
        REQUIRE(backend->successful_dispatch_count() == before_dispatches + 1);
        for (std::size_t n = 0; n < cells; ++n)
        {
            const auto expected_update = [&](std::size_t field, std::size_t tendency)
            {
                return static_cast<float>(static_cast<double>(state.input[field][n]) +
                    static_cast<double>(state.step) * state.input[tendency][n]);
            };
            const float expected_u = mode == MappedTerrainStateUpdateMode::ContinuityEos
                ? state.input[0][n] : expected_update(0, 6);
            const float expected_v = mode == MappedTerrainStateUpdateMode::ContinuityEos
                ? state.input[1][n] : expected_update(1, 7);
            const float expected_w = mode == MappedTerrainStateUpdateMode::ContinuityEos
                ? state.input[2][n] : expected_update(2, 8);
            const float expected_rho = mode == MappedTerrainStateUpdateMode::ContinuityEos
                ? expected_update(3, 9) : state.input[3][n];
            const float expected_p = mode == MappedTerrainStateUpdateMode::Slow
                ? expected_update(4, 10)
                : (mode == MappedTerrainStateUpdateMode::ContinuityEos
                    ? static_cast<float>(tmv::dynamics_eos::pressure_fp64(expected_rho, state.input[5][n]))
                    : state.input[4][n]);
            CHECK(std::abs(state.output[0][n] - expected_u) <= 2.0e-6f);
            CHECK(std::abs(state.output[1][n] - expected_v) <= 2.0e-6f);
            CHECK(std::abs(state.output[2][n] - expected_w) <= 2.0e-6f);
            CHECK(std::abs(state.output[3][n] - expected_rho) <= 2.0e-7f);
            CHECK(std::abs(state.output[4][n] - expected_p) <= 0.05f);
        }
    }

    StateUpdateCase unchanged(carrier);
    const StateUpdateCase before = unchanged;
    auto oversized = unchanged.dispatch();
    oversized.nx = std::numeric_limits<int>::max();
    REQUIRE_FALSE(kernels->dispatch_mapped_state_update(oversized));
    REQUIRE(state_update_outputs_unchanged(unchanged, before));
    auto aliased = unchanged.dispatch();
    aliased.v_candidate = aliased.u_candidate;
    REQUIRE_FALSE(kernels->dispatch_mapped_state_update(aliased));
    REQUIRE(state_update_outputs_unchanged(unchanged, before));
    auto malformed = unchanged.dispatch();
    unchanged.input[3][at(2, 2, 2)] = std::numeric_limits<float>::quiet_NaN();
    REQUIRE_FALSE(kernels->dispatch_mapped_state_update(malformed));
    REQUIRE(state_update_outputs_unchanged(unchanged, before));

    unchanged = before;
    unchanged.mode = MappedTerrainStateUpdateMode::Slow;
    unchanged.input[10][at(2, 2, 2)] = -1.0e8f;
    auto negative_pressure = unchanged.dispatch();
    REQUIRE_FALSE(kernels->dispatch_mapped_state_update(negative_pressure));
    REQUIRE(state_update_outputs_unchanged(unchanged, before));

    unchanged = before;
    unchanged.mode = MappedTerrainStateUpdateMode::ContinuityEos;
    unchanged.input[9][at(2, 2, 2)] = -100.0f;
    auto negative_density = unchanged.dispatch();
    REQUIRE_FALSE(kernels->dispatch_mapped_state_update(negative_density));
    REQUIRE(state_update_outputs_unchanged(unchanged, before));

    unchanged = before;
    auto veto = unchanged.dispatch();
    veto.completed_observer = [](void*, const float*, const float*, const float*, const float*, const float*)
    { throw std::runtime_error("late mapped state veto"); };
    REQUIRE_THROWS_WITH(kernels->dispatch_mapped_state_update(veto), "late mapped state veto");
    REQUIRE(state_update_outputs_unchanged(unchanged, before));
    shutdown_compute_backend_runtime();
}

TEST_CASE("Vulkan mapped Bell retained mass flux uses checked device time accumulation atomically",
          "[vulkan][device][terrain][mapped_flux_accumulation]")
{
    std::string error;
    global_compute_backend_config.backend = "vulkan";
    global_compute_backend_config.allow_fallback = false;
    if (!initialize_compute_backend_runtime(error))
    {
        WARN("Vulkan unavailable: " << error);
        return;
    }
    ComputeBackend* backend = mutable_compute_backend();
    TerrainComputeKernels* kernels = backend->terrain_kernels();
    REQUIRE(kernels != nullptr);
    REQUIRE(kernels->supports_mapped_flux_accumulation());
    std::array<std::vector<float>, 3> accumulated;
    std::array<std::vector<float>, 3> step_flux;
    std::array<std::vector<float>, 3> candidate;
    for (std::size_t field = 0; field < 3; ++field)
    {
        accumulated[field].assign(cells, 0.0f);
        step_flux[field].resize(cells);
        candidate[field].assign(cells, -71.0f);
        for (std::size_t n = 0; n < cells; ++n)
            step_flux[field][n] = static_cast<float>(field + 1) *
                std::sin(0.013f * static_cast<float>(n + 1));
    }
    MappedTerrainFluxAccumulationReport report;
    auto dispatch = [&]
    {
        MappedTerrainFluxAccumulationDispatch d;
        d.accumulated_x = accumulated[0].data();
        d.accumulated_y = accumulated[1].data();
        d.accumulated_zeta = accumulated[2].data();
        d.step_x = step_flux[0].data();
        d.step_y = step_flux[1].data();
        d.step_zeta = step_flux[2].data();
        d.accumulated_x_candidate = candidate[0].data();
        d.accumulated_y_candidate = candidate[1].data();
        d.accumulated_zeta_candidate = candidate[2].data();
        d.nx = nx;
        d.ny = ny;
        d.nz = nz;
        d.time_weight = 1.0f / 6.0f;
        d.report = &report;
        return d;
    };
    for (int substep = 0; substep < 6; ++substep)
    {
        const auto before_dispatches = backend->successful_dispatch_count();
        REQUIRE(kernels->dispatch_mapped_flux_accumulation(dispatch()));
        REQUIRE(report.complete);
        REQUIRE(report.dispatch_count == 1);
        REQUIRE(backend->successful_dispatch_count() == before_dispatches + 1);
        accumulated = candidate;
    }
    for (std::size_t field = 0; field < 3; ++field)
        for (std::size_t n = 0; n < cells; ++n)
            CHECK(std::abs(accumulated[field][n] - step_flux[field][n]) <= 4.0e-7f);

    const auto unchanged_candidate = candidate;
    const auto unchanged_report = report;
    auto invalid = dispatch();
    invalid.time_weight = std::nextafter(1.0f, 2.0f);
    REQUIRE_FALSE(kernels->dispatch_mapped_flux_accumulation(invalid));
    REQUIRE(candidate == unchanged_candidate);
    REQUIRE(std::memcmp(&report, &unchanged_report, sizeof(report)) == 0);
    auto aliased = dispatch();
    aliased.accumulated_y_candidate = aliased.accumulated_x_candidate;
    REQUIRE_FALSE(kernels->dispatch_mapped_flux_accumulation(aliased));
    REQUIRE(candidate == unchanged_candidate);
    auto veto = dispatch();
    veto.completed_observer = [](void*, const float*, const float*, const float*)
    { throw std::runtime_error("late mapped flux veto"); };
    REQUIRE_THROWS_WITH(kernels->dispatch_mapped_flux_accumulation(veto),
                        "late mapped flux veto");
    REQUIRE(candidate == unchanged_candidate);
    REQUIRE(backend->successful_kernel_dispatch_count(
        nimbus::compute::KernelId::MappedTerrainFluxAccumulation) == 7);
    shutdown_compute_backend_runtime();
}

TEST_CASE("Vulkan dry stationary Bell mapped core commits one canonical transaction",
          "[vulkan][device][terrain][mapped_core_transaction]")
{
    std::string error;
    global_compute_backend_config.backend = "vulkan";
    global_compute_backend_config.allow_fallback = false;
    if (!initialize_compute_backend_runtime(error))
    {
        WARN("Vulkan unavailable: " << error);
        return;
    }

    const int previous_threads = omp_get_max_threads();
    SimulationStorageState one_thread = mapped_transaction_state();
    auto one_view = make_simulation_state_view(one_thread);
    one_view.compute_runtime = nullptr;
    tmv::terrain_native::MappedCoreStepReport one_report;
    omp_set_num_threads(1);
    REQUIRE(tmv::terrain_native::advance_dry_stationary_mapped_core(
        one_view, 0.03, 3, one_report));
    REQUIRE(one_report.complete);
    REQUIRE(one_report.final_pressure_uses_canonical_fp64);
    REQUIRE(one_report.acoustic_substeps == 3);
    REQUIRE(one_report.mass_continuity_dispatches == 5);
    REQUIRE(one_report.slow_tendency_dispatches == 2);
    REQUIRE(one_report.slow_average_dispatches == 6);
    REQUIRE(one_report.fast_momentum_dispatches == 3);
    REQUIRE(one_report.state_update_dispatches == 8);
    REQUIRE(one_report.flux_accumulation_dispatches == 3);
    REQUIRE(one_report.scalar_flux_dispatches == 4);
    REQUIRE(one_report.scalar_limiter_dispatches == 3);
    REQUIRE(one_report.scalar_update_dispatches == 4);
    REQUIRE(one_report.boundary_dispatches == 8);
    REQUIRE(one_report.accumulated_time_weight == Approx(1.0).margin(4.0e-7));
    REQUIRE(one_report.completed_backend_dispatches > 0);
    REQUIRE(one_report.acoustic.complete);
    REQUIRE(one_report.acoustic.kinetic_transport_complete);
    REQUIRE(one_report.acoustic.boundary_complete);
    REQUIRE(one_report.acoustic.substep_count == 3);
    REQUIRE(one_report.acoustic.boundary_application_count == 4);
    REQUIRE(one_report.slow_momentum.complete);
    REQUIRE(one_report.final_eos.complete);
    REQUIRE(one_report.final_boundary.complete);
    const auto domain = physical_scalar_domain(one_view.grid);
    for (int i = domain.i_begin; i < domain.i_end; ++i)
        for (int j = domain.j_begin; j < domain.j_end; ++j)
            for (int k = domain.k_begin; k < domain.k_end; ++k)
                REQUIRE(one_thread.fields.p[i][j][k] == clamp_pressure_pa(static_cast<float>(
                    tmv::dynamics_eos::pressure_fp64(one_thread.fields.rho[i][j][k],
                                                     one_thread.fields.theta[i][j][k]))));

    SimulationStorageState four_threads = mapped_transaction_state();
    auto four_view = make_simulation_state_view(four_threads);
    four_view.compute_runtime = nullptr;
    tmv::terrain_native::MappedCoreStepReport four_report;
    omp_set_num_threads(4);
    REQUIRE(tmv::terrain_native::advance_dry_stationary_mapped_core(
        four_view, 0.03, 3, four_report));
    REQUIRE(mapped_public_fields(four_threads) == mapped_public_fields(one_thread));
    REQUIRE(same_mapped_report(four_report, one_report));

    SimulationStorageState invalid = mapped_transaction_state();
    invalid.fields.qv[3][3][2] = 1.0e-6f;
    auto invalid_view = make_simulation_state_view(invalid);
    invalid_view.compute_runtime = nullptr;
    const NineFields invalid_before = mapped_public_fields(invalid);
    tmv::terrain_native::MappedCoreStepReport invalid_report;
    invalid_report.acoustic_substeps = 91;
    const auto invalid_report_before = invalid_report;
    REQUIRE_FALSE(tmv::terrain_native::advance_dry_stationary_mapped_core(
        invalid_view, 0.03, 3, invalid_report));
    REQUIRE(mapped_public_fields(invalid) == invalid_before);
    REQUIRE(std::memcmp(&invalid_report, &invalid_report_before,
                        sizeof(invalid_report)) == 0);

    SimulationStorageState malformed_metrics = mapped_transaction_state();
    malformed_metrics.runtime.terrain.metrics.Jx_face.resize(
        nx, ny, nz, 1.0);
    auto malformed_view = make_simulation_state_view(malformed_metrics);
    malformed_view.compute_runtime = nullptr;
    const NineFields malformed_before = mapped_public_fields(malformed_metrics);
    tmv::terrain_native::MappedCoreStepReport malformed_report;
    malformed_report.fast_momentum_dispatches = 51;
    const auto malformed_report_before = malformed_report;
    REQUIRE_FALSE(tmv::terrain_native::advance_dry_stationary_mapped_core(
        malformed_view, 0.03, 3, malformed_report));
    REQUIRE(mapped_public_fields(malformed_metrics) == malformed_before);
    REQUIRE(std::memcmp(&malformed_report, &malformed_report_before,
                        sizeof(malformed_report)) == 0);

    SimulationStorageState wrong_shape = mapped_transaction_state();
    wrong_shape.fields.tracer.resize(nx * ny, 1, nz, 0.1f);
    auto wrong_shape_view = make_simulation_state_view(wrong_shape);
    wrong_shape_view.compute_runtime = nullptr;
    const NineFields wrong_shape_before = mapped_public_fields(wrong_shape);
    tmv::terrain_native::MappedCoreStepReport wrong_shape_report;
    wrong_shape_report.scalar_update_dispatches = 53;
    const auto wrong_shape_report_before = wrong_shape_report;
    REQUIRE_FALSE(tmv::terrain_native::advance_dry_stationary_mapped_core(
        wrong_shape_view, 0.03, 3, wrong_shape_report));
    REQUIRE(mapped_public_fields(wrong_shape) == wrong_shape_before);
    REQUIRE(std::memcmp(&wrong_shape_report, &wrong_shape_report_before,
                        sizeof(wrong_shape_report)) == 0);

    SimulationStorageState spacing_mismatch = mapped_transaction_state();
    spacing_mismatch.runtime.terrain.metrics.dx += 1.0;
    auto spacing_view = make_simulation_state_view(spacing_mismatch);
    spacing_view.compute_runtime = nullptr;
    const NineFields spacing_before = mapped_public_fields(spacing_mismatch);
    tmv::terrain_native::MappedCoreStepReport spacing_report;
    spacing_report.flux_accumulation_dispatches = 55;
    const auto spacing_report_before = spacing_report;
    REQUIRE_FALSE(tmv::terrain_native::advance_dry_stationary_mapped_core(
        spacing_view, 0.03, 3, spacing_report));
    REQUIRE(mapped_public_fields(spacing_mismatch) == spacing_before);
    REQUIRE(std::memcmp(&spacing_report, &spacing_report_before,
                        sizeof(spacing_report)) == 0);

    SimulationStorageState huge_dt = mapped_transaction_state();
    auto huge_dt_view = make_simulation_state_view(huge_dt);
    huge_dt_view.compute_runtime = nullptr;
    const NineFields huge_dt_before = mapped_public_fields(huge_dt);
    tmv::terrain_native::MappedCoreStepReport huge_dt_report;
    huge_dt_report.state_update_dispatches = 57;
    const auto huge_dt_report_before = huge_dt_report;
    REQUIRE_FALSE(tmv::terrain_native::advance_dry_stationary_mapped_core(
        huge_dt_view, std::numeric_limits<double>::max(), 3, huge_dt_report));
    REQUIRE(mapped_public_fields(huge_dt) == huge_dt_before);
    REQUIRE(std::memcmp(&huge_dt_report, &huge_dt_report_before,
                        sizeof(huge_dt_report)) == 0);

    SimulationStorageState nonfinite_candidate = mapped_transaction_state();
    std::fill_n(nonfinite_candidate.fields.u.data(), cells,
                std::numeric_limits<float>::max() / 16.0f);
    auto nonfinite_view = make_simulation_state_view(nonfinite_candidate);
    nonfinite_view.compute_runtime = nullptr;
    const NineFields nonfinite_before = mapped_public_fields(nonfinite_candidate);
    tmv::terrain_native::MappedCoreStepReport nonfinite_report;
    nonfinite_report.slow_tendency_dispatches = 59;
    const auto nonfinite_report_before = nonfinite_report;
    REQUIRE_FALSE(tmv::terrain_native::advance_dry_stationary_mapped_core(
        nonfinite_view, 0.03, 3, nonfinite_report));
    REQUIRE(mapped_public_fields(nonfinite_candidate) == nonfinite_before);
    REQUIRE(std::memcmp(&nonfinite_report, &nonfinite_report_before,
                        sizeof(nonfinite_report)) == 0);

    SimulationStorageState aliased = mapped_transaction_state();
    auto alias_view = aliased_uv_view(aliased);
    const NineFields alias_before = mapped_public_fields(aliased);
    tmv::terrain_native::MappedCoreStepReport alias_report;
    alias_report.boundary_dispatches = 63;
    const auto alias_report_before = alias_report;
    REQUIRE_FALSE(tmv::terrain_native::advance_dry_stationary_mapped_core(
        alias_view, 0.03, 3, alias_report));
    REQUIRE(mapped_public_fields(aliased) == alias_before);
    REQUIRE(std::memcmp(&alias_report, &alias_report_before,
                        sizeof(alias_report)) == 0);

    SimulationStorageState vetoed = mapped_transaction_state();
    auto vetoed_view = make_simulation_state_view(vetoed);
    vetoed_view.compute_runtime = nullptr;
    const NineFields vetoed_before = mapped_public_fields(vetoed);
    tmv::terrain_native::MappedCoreStepReport vetoed_report;
    vetoed_report.mass_continuity_dispatches = 77;
    const auto vetoed_report_before = vetoed_report;
    REQUIRE_THROWS_WITH(tmv::terrain_native::advance_dry_stationary_mapped_core(
        vetoed_view, 0.03, 3, vetoed_report,
        +[](void*, const Field3D&, const Field3D&, const Field3D&, const Field3D&,
            const Field3D&, const Field3D&, const Field3D&, const Field3D&, const Field3D&,
            const Field3D&, const Field3D&, const Field3D&, const Field3D&,
            const tmv::terrain_native::MappedCoreStepReport&)
        { throw std::runtime_error("reject complete mapped core candidate"); }),
        "reject complete mapped core candidate");
    REQUIRE(mapped_public_fields(vetoed) == vetoed_before);
    REQUIRE(std::memcmp(&vetoed_report, &vetoed_report_before,
                        sizeof(vetoed_report)) == 0);

    omp_set_num_threads(previous_threads);
    shutdown_compute_backend_runtime();
}
