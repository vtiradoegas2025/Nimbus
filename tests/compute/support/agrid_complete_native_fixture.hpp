#pragma once

#include "catch2/catch.hpp"
#include "compute/compute_backend.hpp"
#include "core/field/field3d.hpp"
#include "dynamics/collocated_velocity_constraint.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

namespace nimbus::test
{

struct AgridCompleteNativeMetrics
{
    double pressure_max_abs = 0.0;
    double momentum_max_abs = 0.0;
    double constraint_max_abs = 0.0;
    double slow_max_abs = 0.0;
};

inline AgridCompleteNativeMetrics run_agrid_complete_native_fixture(
    DynamicsComputeKernels& kernels)
{
    constexpr int nx = 8;
    constexpr int ny = 7;
    constexpr int nz = 6;
    constexpr float dx = 500.0f;
    constexpr float dy = 500.0f;
    constexpr float dz = 250.0f;
    constexpr float dt = 0.1f;
    constexpr float p_ref = 100000.0f;
    constexpr float rd = 287.05f;
    constexpr float gamma = 1.4f;
    constexpr float rho_min = 0.2f;
    constexpr float rho_max = 2.0f;
    constexpr float p_min = 10000.0f;
    constexpr float p_max = 120000.0f;
    constexpr std::size_t cells = static_cast<std::size_t>(nx) * ny * nz;
    const auto at = [](int i, int j, int k)
    {
        return (static_cast<std::size_t>(i) * ny + j) * nz + k;
    };

    REQUIRE(kernels.supports(DynamicsKernelOp::AcousticPressureEosCartesianAgrid));
    REQUIRE(kernels.supports(DynamicsKernelOp::AcousticMomentumCartesianAgrid));
    REQUIRE(kernels.supports(DynamicsKernelOp::AcousticNullModeConstraintCartesianAgrid));
    REQUIRE(kernels.supports(DynamicsKernelOp::AcousticBoundaryCartesianAgrid));

    std::vector<float> rho(cells), theta(cells), drho(cells);
    std::vector<float> p0(nz), rho0(nz), u0(nz), v0(nz);
    std::vector<float> u(cells), v(cells), w(cells);
    std::array<std::vector<float>, 7> water;
    for (auto& field : water) field.assign(cells, 0.0f);
    for (int k = 0; k < nz; ++k)
    {
        p0[static_cast<std::size_t>(k)] = 100000.0f - 2100.0f * k;
        rho0[static_cast<std::size_t>(k)] = 1.16f - 0.035f * k;
        u0[static_cast<std::size_t>(k)] = 4.0f + 0.2f * k;
        v0[static_cast<std::size_t>(k)] = -1.0f + 0.1f * k;
    }
    for (int i = 0; i < nx; ++i)
        for (int j = 0; j < ny; ++j)
            for (int k = 0; k < nz; ++k)
            {
                const std::size_t n = at(i, j, k);
                rho[n] = rho0[static_cast<std::size_t>(k)] +
                         0.0012f * i - 0.0007f * j;
                theta[n] = 300.0f + 0.5f * k + 0.03f * i - 0.02f * j;
                drho[n] = 0.0004f * (i - 3) - 0.0002f * (j - 2) +
                          0.0001f * (k - 2);
                u[n] = u0[static_cast<std::size_t>(k)] +
                       0.03f * i - 0.02f * j + (((i & 1) == 0) ? 0.7f : -0.7f);
                v[n] = v0[static_cast<std::size_t>(k)] -
                       0.01f * i + 0.04f * j + (((j & 1) == 0) ? 0.5f : -0.5f);
                w[n] = 0.02f * i - 0.015f * j + 0.01f * k +
                       (((k & 1) == 0) ? 0.35f : -0.35f);
                water[0][n] = 0.008f + 0.00001f * (i + j + k);
                water[1][n] = 0.0001f * ((i + j + k) % 3);
            }

    std::vector<float> expected_rho(cells), expected_p(cells);
    for (std::size_t n = 0; n < cells; ++n)
    {
        expected_rho[n] = std::clamp(rho[n] + dt * drho[n], rho_min, rho_max);
        expected_p[n] = std::clamp(
            p_ref * std::pow(expected_rho[n] * rd * theta[n] / p_ref, gamma),
            p_min, p_max);
    }
    std::vector<float> actual_rho(cells, -91.0f), actual_p(cells, -92.0f);
    AcousticPressureEosCartesianAgridDispatch pressure{};
    pressure.rho_in = rho.data();
    pressure.theta = theta.data();
    pressure.drho_dt = drho.data();
    pressure.rho_out = actual_rho.data();
    pressure.p_out = actual_p.data();
    pressure.nx = nx;
    pressure.ny = ny;
    pressure.nz = nz;
    pressure.dt_small = dt;
    pressure.reference_pressure = p_ref;
    pressure.gas_constant_dry_air = rd;
    pressure.gamma = gamma;
    pressure.rho_min = rho_min;
    pressure.rho_max = rho_max;
    pressure.p_min = p_min;
    pressure.p_max = p_max;
    const auto pressure_before = actual_rho;
    const auto reject_pressure = +[](void*, const float*, const float*)
    { throw std::runtime_error("reject native A-grid pressure candidate"); };
    pressure.completed_observer = reject_pressure;
    REQUIRE_THROWS_WITH(
        kernels.dispatch_acoustic_pressure_eos_cartesian_agrid(pressure),
        "reject native A-grid pressure candidate");
    REQUIRE(actual_rho == pressure_before);
    pressure.completed_observer = nullptr;
    REQUIRE(kernels.dispatch_acoustic_pressure_eos_cartesian_agrid(pressure));

    AgridCompleteNativeMetrics metrics;
    for (std::size_t n = 0; n < cells; ++n)
    {
        metrics.pressure_max_abs = std::max(
            metrics.pressure_max_abs,
            std::abs(static_cast<double>(actual_p[n]) - expected_p[n]));
        REQUIRE(std::abs(actual_rho[n] - expected_rho[n]) <= 2.0e-6f);
        REQUIRE(std::abs(actual_p[n] - expected_p[n]) <= 5.0e-2f);
    }

    std::vector<float> expected_u = u, expected_v = v, expected_w = w;
    for (int i = 1; i < nx - 1; ++i)
        for (int j = 1; j < ny - 1; ++j)
            for (int k = 1; k < nz - 1; ++k)
            {
                const std::size_t n = at(i, j, k);
                const float px = (actual_p[at(i + 1, j, k)] - actual_p[at(i - 1, j, k)]) /
                                 (2.0f * dx);
                const float py = (actual_p[at(i, j + 1, k)] - actual_p[at(i, j - 1, k)]) /
                                 (2.0f * dy);
                const float pz = (actual_p[at(i, j, k + 1)] - actual_p[at(i, j, k - 1)]) /
                                 (2.0f * dz);
                const float p0z = (p0[static_cast<std::size_t>(k + 1)] -
                                    p0[static_cast<std::size_t>(k - 1)]) /
                                   (2.0f * dz);
                expected_u[n] = std::clamp(u[n] - dt * px / actual_rho[n], -120.0f, 120.0f);
                expected_v[n] = std::clamp(v[n] - dt * py / actual_rho[n], -120.0f, 120.0f);
                expected_w[n] = std::clamp(w[n] - dt * (pz - p0z) / actual_rho[n], -90.0f, 90.0f);
            }
    std::vector<float> actual_u(cells), actual_v(cells), actual_w(cells);
    AcousticMomentumCartesianAgridDispatch momentum{};
    momentum.rho = actual_rho.data();
    momentum.p = actual_p.data();
    momentum.p0_base_data = p0.data();
    momentum.base_len = nz;
    momentum.u_in = u.data();
    momentum.v_in = v.data();
    momentum.w_in = w.data();
    momentum.u_out = actual_u.data();
    momentum.v_out = actual_v.data();
    momentum.w_out = actual_w.data();
    momentum.nx = nx;
    momentum.ny = ny;
    momentum.nz = nz;
    momentum.dx = dx;
    momentum.dy = dy;
    momentum.dz = dz;
    momentum.dt_small = dt;
    momentum.wind_clamp_h = 120.0f;
    momentum.wind_clamp_v = 90.0f;
    REQUIRE(kernels.dispatch_acoustic_momentum_cartesian_agrid(momentum));
    for (std::size_t n = 0; n < cells; ++n)
    {
        metrics.momentum_max_abs = std::max({
            metrics.momentum_max_abs,
            std::abs(static_cast<double>(actual_u[n]) - expected_u[n]),
            std::abs(static_cast<double>(actual_v[n]) - expected_v[n]),
            std::abs(static_cast<double>(actual_w[n]) - expected_w[n])});
        REQUIRE(std::abs(actual_u[n] - expected_u[n]) <= 2.0e-5f);
        REQUIRE(std::abs(actual_v[n] - expected_v[n]) <= 2.0e-5f);
        REQUIRE(std::abs(actual_w[n] - expected_w[n]) <= 2.0e-5f);
    }

    Field3D expected_u_field(nx, ny, nz), expected_v_field(nx, ny, nz);
    Field3D expected_w_field(nx, ny, nz), rho_field(nx, ny, nz);
    std::array<Field3D, 7> water_fields;
    for (Field3D& field : water_fields) field.resize(nx, ny, nz);
    std::memcpy(expected_u_field.data(), actual_u.data(), cells * sizeof(float));
    std::memcpy(expected_v_field.data(), actual_v.data(), cells * sizeof(float));
    std::memcpy(expected_w_field.data(), actual_w.data(), cells * sizeof(float));
    std::memcpy(rho_field.data(), actual_rho.data(), cells * sizeof(float));
    std::array<const Field3D*, 7> water_views{};
    for (int q = 0; q < 7; ++q)
    {
        std::memcpy(water_fields[static_cast<std::size_t>(q)].data(),
                    water[static_cast<std::size_t>(q)].data(), cells * sizeof(float));
        water_views[static_cast<std::size_t>(q)] = &water_fields[static_cast<std::size_t>(q)];
    }
    tmv::collocated_velocity::constrain_directional_null_modes(
        expected_u_field, expected_v_field, expected_w_field, rho_field,
        nx, ny, nz, true, true, true, water_views);

    std::vector<float> constrained_u(cells), constrained_v(cells), constrained_w(cells);
    AcousticNullModeConstraintCartesianAgridDispatch constraint{};
    constraint.rho = actual_rho.data();
    constraint.qv = water[0].data();
    constraint.qc = water[1].data();
    constraint.qr = water[2].data();
    constraint.qi = water[3].data();
    constraint.qs = water[4].data();
    constraint.qg = water[5].data();
    constraint.qh = water[6].data();
    constraint.u_in = actual_u.data();
    constraint.v_in = actual_v.data();
    constraint.w_in = actual_w.data();
    constraint.u_out = constrained_u.data();
    constraint.v_out = constrained_v.data();
    constraint.w_out = constrained_w.data();
    constraint.nx = nx;
    constraint.ny = ny;
    constraint.nz = nz;
    REQUIRE(kernels.dispatch_acoustic_null_mode_constraint_cartesian_agrid(constraint));
    for (std::size_t n = 0; n < cells; ++n)
    {
        metrics.constraint_max_abs = std::max({
            metrics.constraint_max_abs,
            std::abs(static_cast<double>(constrained_u[n]) - expected_u_field.data()[n]),
            std::abs(static_cast<double>(constrained_v[n]) - expected_v_field.data()[n]),
            std::abs(static_cast<double>(constrained_w[n]) - expected_w_field.data()[n])});
        REQUIRE(std::abs(constrained_u[n] - expected_u_field.data()[n]) <= 2.0e-5f);
        REQUIRE(std::abs(constrained_v[n] - expected_v_field.data()[n]) <= 2.0e-5f);
        REQUIRE(std::abs(constrained_w[n] - expected_w_field.data()[n]) <= 2.0e-5f);
    }

    std::vector<float> loading(cells), qv0(nz);
    for (int k = 0; k < nz; ++k) qv0[static_cast<std::size_t>(k)] = 0.008f;
    for (std::size_t n = 0; n < cells; ++n)
        for (int q = 1; q < 7; ++q)
            loading[n] += water[static_cast<std::size_t>(q)][n];
    std::array<std::vector<float>, 5> slow_expected, slow_actual;
    for (auto& field : slow_expected) field.assign(cells, 0.0f);
    for (auto& field : slow_actual) field.assign(cells, 73.0f);
    constexpr float coriolis_f = 1.0e-4f;
    constexpr float coriolis_beta = 2.0e-11f;
    for (int i = 1; i < nx - 1; ++i)
        for (int j = 1; j < ny - 1; ++j)
            for (int k = 1; k < nz - 1; ++k)
            {
                const std::size_t n = at(i, j, k);
                const float ix = 1.0f / (2.0f * dx);
                const float iy = 1.0f / (2.0f * dy);
                const float iz = 1.0f / (2.0f * dz);
                const float local_f = coriolis_f +
                    coriolis_beta * (static_cast<float>(j) + 0.5f) * dy;
                const float adv_u = -u[n] * (u[at(i + 1, j, k)] - u[at(i - 1, j, k)]) * ix
                    - v[n] * (u[at(i, j + 1, k)] - u[at(i, j - 1, k)]) * iy
                    - w[n] * (u[at(i, j, k + 1)] - u[at(i, j, k - 1)]) * iz;
                const float adv_v = -u[n] * (v[at(i + 1, j, k)] - v[at(i - 1, j, k)]) * ix
                    - v[n] * (v[at(i, j + 1, k)] - v[at(i, j - 1, k)]) * iy
                    - w[n] * (v[at(i, j, k + 1)] - v[at(i, j, k - 1)]) * iz;
                const float adv_w = -u[n] * (w[at(i + 1, j, k)] - w[at(i - 1, j, k)]) * ix
                    - v[n] * (w[at(i, j + 1, k)] - w[at(i, j - 1, k)]) * iy
                    - w[n] * (w[at(i, j, k + 1)] - w[at(i, j, k - 1)]) * iz;
                const float p_adv = -u[n] * (actual_p[at(i + 1, j, k)] - actual_p[at(i - 1, j, k)]) * ix
                    - v[n] * (actual_p[at(i, j + 1, k)] - actual_p[at(i, j - 1, k)]) * iy
                    - w[n] * (actual_p[at(i, j, k + 1)] - actual_p[at(i, j, k - 1)]) * iz;
                const float buoyancy = -9.81f * (rho[n] - rho0[static_cast<std::size_t>(k)]) / rho[n];
                const float moisture = 9.81f * 0.608f *
                    (water[0][n] - qv0[static_cast<std::size_t>(k)]);
                slow_expected[0][n] = adv_u + local_f * (v[n] - v0[static_cast<std::size_t>(k)]);
                slow_expected[1][n] = adv_v - local_f * (u[n] - u0[static_cast<std::size_t>(k)]);
                slow_expected[2][n] = adv_w + buoyancy + moisture - 9.81f * loading[n];
                slow_expected[4][n] = p_adv;
            }
    CartesianTendenciesDispatch slow{};
    slow.u_x_data = u.data();
    slow.u_y_data = v.data();
    slow.w_data = w.data();
    slow.rho_data = rho.data();
    slow.p_data = actual_p.data();
    slow.theta_data = theta.data();
    slow.p0_base_data = p0.data();
    slow.rho0_base_data = rho0.data();
    slow.loading_data = loading.data();
    slow.u0_base_data = u0.data();
    slow.v0_base_data = v0.data();
    slow.qv_data = water[0].data();
    slow.qv0_base_data = qv0.data();
    slow.du_x_dt_data = slow_actual[0].data();
    slow.du_y_dt_data = slow_actual[1].data();
    slow.dw_dt_data = slow_actual[2].data();
    slow.drho_dt_data = slow_actual[3].data();
    slow.dp_dt_data = slow_actual[4].data();
    slow.nr = nx;
    slow.nth = ny;
    slow.nz = nz;
    slow.dx = dx;
    slow.dy = dy;
    slow.dz = dz;
    slow.g = 9.81f;
    slow.gamma_val = gamma;
    slow.coriolis_f_val = coriolis_f;
    slow.lateral_bc_west = 2;
    slow.lateral_bc_east = 2;
    slow.lateral_bc_south = 2;
    slow.lateral_bc_north = 2;
    slow.top_bc = 0;
    slow.has_moisture = true;
    slow.coriolis_beta = coriolis_beta;
    slow.split_slow_only = true;
    const auto slow_before = slow_actual;
    const float saved_rho = rho[at(2, 2, 2)];
    rho[at(2, 2, 2)] = std::numeric_limits<float>::quiet_NaN();
    REQUIRE_FALSE(kernels.dispatch_cartesian_tendencies(slow));
    REQUIRE(slow_actual == slow_before);
    rho[at(2, 2, 2)] = saved_rho;
    REQUIRE(kernels.dispatch_cartesian_tendencies(slow));
    for (int field = 0; field < 5; ++field)
        for (std::size_t n = 0; n < cells; ++n)
        {
            metrics.slow_max_abs = std::max(
                metrics.slow_max_abs,
                std::abs(static_cast<double>(slow_actual[field][n]) -
                         slow_expected[field][n]));
            REQUIRE(std::abs(slow_actual[field][n] - slow_expected[field][n]) <= 3.0e-5f);
        }

    constexpr int balanced_i = 2;
    constexpr int balanced_j = 2;
    for (int k = 0; k < nz; ++k)
    {
        const auto n     = at(balanced_i, balanced_j, k);
        constrained_u[n] = u0[k];
        constrained_v[n] = v0[k];
        constrained_w[n] = 0.0f;
        actual_rho[n]    = rho0[k];
        actual_p[n]      = p0[k];
    }

    auto expected_boundary_u = constrained_u;
    auto expected_boundary_v = constrained_v;
    auto expected_boundary_w = constrained_w;
    auto expected_boundary_rho = actual_rho;
    auto expected_boundary_p = actual_p;
    for (int j = 0; j < ny; ++j)
        for (int k = 0; k < nz; ++k)
            for (int face = 0; face < 2; ++face)
            {
                const int i = face == 0 ? 0 : nx - 1;
                const int source = face == 0 ? 1 : nx - 2;
                const std::size_t n = at(i, j, k), s = at(source, j, k);
                expected_boundary_u[n] = 0.0f;
                expected_boundary_v[n] = 0.0f;
                expected_boundary_w[n] = 0.0f;
                expected_boundary_rho[n] = expected_boundary_rho[s];
                expected_boundary_p[n] = expected_boundary_p[s];
            }
    for (int i = 0; i < nx; ++i)
        for (int k = 0; k < nz; ++k)
            for (int face = 0; face < 2; ++face)
            {
                const int j = face == 0 ? 0 : ny - 1;
                const int source = face == 0 ? 1 : ny - 2;
                const std::size_t n = at(i, j, k), s = at(i, source, k);
                expected_boundary_u[n] = 0.0f;
                expected_boundary_v[n] = 0.0f;
                expected_boundary_w[n] = 0.0f;
                expected_boundary_rho[n] = expected_boundary_rho[s];
                expected_boundary_p[n] = expected_boundary_p[s];
            }
    for (int i = 0; i < nx; ++i)
        for (int j = 0; j < ny; ++j)
        {
            const std::size_t bottom = at(i, j, 0), bottom_source = at(i, j, 1);
            const std::size_t top = at(i, j, nz - 1), top_source = at(i, j, nz - 2);
            expected_boundary_w[bottom] = 0.0f;
            expected_boundary_w[top] = 0.0f;
            expected_boundary_u[bottom]   = u0[0] + (expected_boundary_u[bottom_source] - u0[1]);
            expected_boundary_v[bottom]   = v0[0] + (expected_boundary_v[bottom_source] - v0[1]);
            expected_boundary_u[top]      = u0[nz - 1] + (expected_boundary_u[top_source] - u0[nz - 2]);
            expected_boundary_v[top]      = v0[nz - 1] + (expected_boundary_v[top_source] - v0[nz - 2]);
            expected_boundary_rho[bottom] = rho0[0] + (expected_boundary_rho[bottom_source] - rho0[1]);
            expected_boundary_rho[top]    = rho0[nz - 1] + (expected_boundary_rho[top_source] - rho0[nz - 2]);
            expected_boundary_p[bottom]   = p0[0] + (expected_boundary_p[bottom_source] - p0[1]);
            expected_boundary_p[top]      = p0[nz - 1] + (expected_boundary_p[top_source] - p0[nz - 2]);
        }

    AcousticBoundaryCartesianAgridDispatch boundary{};
    boundary.p0_base_data = p0.data();
    boundary.rho0_base_data = rho0.data();
    boundary.u0_base_data = u0.data();
    boundary.v0_base_data = v0.data();
    boundary.base_len = nz;
    boundary.u = constrained_u.data();
    boundary.v = constrained_v.data();
    boundary.w = constrained_w.data();
    boundary.rho = actual_rho.data();
    boundary.p = actual_p.data();
    boundary.nx = nx;
    boundary.ny = ny;
    boundary.nz = nz;
    const auto boundary_before = constrained_u;
    p0[2] = std::numeric_limits<float>::quiet_NaN();
    REQUIRE_FALSE(kernels.dispatch_acoustic_boundary_cartesian_agrid(boundary));
    REQUIRE(constrained_u == boundary_before);
    p0[2] = 100000.0f - 2100.0f * 2.0f;
    REQUIRE(kernels.dispatch_acoustic_boundary_cartesian_agrid(boundary));
    REQUIRE(constrained_u == expected_boundary_u);
    REQUIRE(constrained_v == expected_boundary_v);
    REQUIRE(constrained_w == expected_boundary_w);
    REQUIRE(actual_rho == expected_boundary_rho);
    REQUIRE(actual_p == expected_boundary_p);
    for (int k : {0, nz - 1})
    {
        const auto n = at(balanced_i, balanced_j, k);
        REQUIRE(constrained_u[n] == u0[k]);
        REQUIRE(constrained_v[n] == v0[k]);
        REQUIRE(constrained_w[n] == 0.0f);
        REQUIRE(actual_rho[n] == rho0[k]);
        REQUIRE(actual_p[n] == p0[k]);
    }
    return metrics;
}

} // namespace nimbus::test
