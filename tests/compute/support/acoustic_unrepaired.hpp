#pragma once

#include "compute/compute_kernel_interfaces.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

inline void exercise_acoustic_unrepaired(DynamicsComputeKernels& kernels)
{
    constexpr int      n      = 4;
    constexpr int      cells  = n * n * n;
    constexpr int      center = (n + 1) * n + 1;
    std::vector<float> zero(cells, 0.0f), rho(cells, 1.0f), pressure(cells, 100000.0f);
    std::vector<float> theta(cells, 300.0f), base_rho(n, 1.0f), base_p(n, 100000.0f), gradient(n, 0.0f);
    std::vector<float> density_out(cells), pressure_out(cells), u(cells), v(cells), w(cells);
    AcousticPressureCartesianCgridDispatch pd;
    pd.u = pd.v = pd.w = zero.data();
    pd.rho_in          = rho.data();
    pd.p_in            = pressure.data();
    pd.rho0_base_data  = base_rho.data();
    pd.p0_base_data    = base_p.data();
    pd.base_len        = n;
    pd.theta_data      = theta.data();
    pd.rho_out         = density_out.data();
    pd.p_out           = pressure_out.data();
    pd.nx = pd.ny = pd.nz = n;
    pd.dx = pd.dy = pd.dz = 1.0f;
    pd.gamma_val          = 1.4f;
    pd.dt_small           = 0.1f;
    pd.rho_min            = 0.1f;
    pd.rho_max            = 2.0f;
    pd.p_min              = 100.0f;
    pd.p_max              = 110000.0f;
    pd.p_ref_fallback     = 100000.0f;

    for (bool repair : {false, true})
    {
        CAPTURE(repair);
        rho.assign(cells, 1.0f);
        theta.assign(cells, 300.0f);
        rho[0]                  = 3.0f;
        rho[1]                  = -1.0f;
        rho[2]                  = std::numeric_limits<float>::quiet_NaN();
        theta[3]                = 0.0f;
        pd.repair_primary_state = repair;
        REQUIRE(kernels.dispatch_acoustic_pressure_cartesian_cgrid(pd));
        CHECK(density_out[0] == (repair ? 2.0f : 3.0f));
        CHECK(density_out[1] == (repair ? 1.0f : -1.0f));
        CHECK(std::isnan(density_out[2]) == !repair);
        CHECK(std::isnan(pressure_out[3]) == !repair);
        CHECK((pressure_out[0] > pd.p_max) == !repair);
        CHECK(std::abs(pressure_out[center] - 100000.0f * std::pow(0.861f, 1.4f)) < 0.1f);
    }

    rho.assign(cells, 1.0f);
    theta.assign(cells, 300.0f);
    pd.repair_primary_state = false;
    REQUIRE(kernels.dispatch_acoustic_pressure_cartesian_cgrid(pd));
    pd.rho_out = rho.data();
    pd.p_out   = pressure.data();
    REQUIRE(kernels.dispatch_acoustic_pressure_cartesian_cgrid(pd));
    CHECK(rho == density_out);
    CHECK(pressure == pressure_out);
    const auto saved_rho      = rho;
    const auto saved_pressure = pressure;
    pd.dx                     = 0.0f;
    REQUIRE_FALSE(kernels.dispatch_acoustic_pressure_cartesian_cgrid(pd));
    CHECK(rho == saved_rho);
    CHECK(pressure == saved_pressure);

    pd.dx = 1.0f;
    pd.rho_out = density_out.data();
    pd.p_out = pressure_out.data();
    pd.u = u.data();
    pd.v = v.data();
    pd.frame_velocity_u_ms = 5.25f;
    pd.frame_velocity_v_ms = -2.125f;
    std::fill(u.begin(), u.end(), pd.frame_velocity_u_ms);
    std::fill(v.begin(), v.end(), pd.frame_velocity_v_ms);
    for (int cell = 0; cell < cells; ++cell)
        rho[cell] = 1.0f + 0.01f * cell;
    REQUIRE(kernels.dispatch_acoustic_pressure_cartesian_cgrid(pd));
    CHECK(density_out == rho);

    AcousticMomentumCartesianCgridDispatch md;
    md.rho_data    = rho.data();
    md.p_data      = pressure.data();
    md.dp0_dz_data = gradient.data();
    md.base_len    = n;
    md.u_in        = u.data();
    md.v_in        = v.data();
    md.w_in        = w.data();
    md.u_out       = u.data();
    md.v_out       = v.data();
    md.w_out       = w.data();
    md.nx = md.ny = md.nz = n;
    md.dx = md.dy = md.dz = 1.0f;
    md.dt_small           = 0.1f;
    md.wind_clamp_h = md.wind_clamp_v = 10.0f;
    for (bool repair : {false, true})
    {
        CAPTURE(repair);
        pressure.assign(cells, 100000.0f);
        u.assign(cells, 20.0f);
        v.assign(cells, -20.0f);
        w.assign(cells, std::numeric_limits<float>::quiet_NaN());
        pressure[center]        = std::numeric_limits<float>::quiet_NaN();
        md.repair_primary_state = repair;
        REQUIRE(kernels.dispatch_acoustic_momentum_cartesian_cgrid(md));
        CHECK(u[0] == (repair ? 10.0f : 20.0f));
        CHECK(v[0] == (repair ? -10.0f : -20.0f));
        CHECK(std::isnan(w[0]) == !repair);
        CHECK(std::isnan(u[center]) == !repair);
        CHECK(std::isnan(v[center]) == !repair);
        CHECK(std::isnan(w[center]) == !repair);
    }
    const auto saved_u = u;
    const auto saved_v = v;
    const auto saved_w = w;
    md.dz              = -1.0f;
    REQUIRE_FALSE(kernels.dispatch_acoustic_momentum_cartesian_cgrid(md));
    CHECK(u == saved_u);
    CHECK(v == saved_v);
    CHECK(w == saved_w);

    pd.u = pd.v = pd.w = zero.data();
    pd.frame_velocity_u_ms = pd.frame_velocity_v_ms = 0.0f;
    pd.gamma_val = 1004.0 / 717.0;
    for (int cell = 0; cell < cells; ++cell)
    {
        rho[cell] = 0.1f + 0.023f * cell;
        theta[cell] = 240.0f + 2.0f * cell;
    }
    REQUIRE(kernels.dispatch_acoustic_pressure_cartesian_cgrid(pd));
    for (int cell = 0; cell < cells; ++cell)
    {
        const float reference = static_cast<float>(100000.0 * std::pow(
            static_cast<double>(rho[cell]) * 287.0 * theta[cell] / 100000.0, pd.gamma_val));
        const float ulp = std::nextafter(reference, std::numeric_limits<float>::infinity()) - reference;
        CHECK(std::abs(pressure_out[cell] - reference) <= ulp);
    }

    std::vector<float> gradient_low(n, -std::ldexp(1.0f, -22));
    std::fill(gradient.begin(), gradient.end(), 10.0f);
    std::fill(rho.begin(), rho.end(), 1.0f);
    std::fill(u.begin(), u.end(), 0.0f);
    std::fill(v.begin(), v.end(), 0.0f);
    std::fill(w.begin(), w.end(), 0.0f);
    for (int cell = 0; cell < cells; ++cell)
        pressure[cell] = 100000.0f + 2500.0f * (cell % n);
    md.dz = 250.0f;
    md.dt_small = 0.25f;
    md.repair_primary_state = false;
    md.dp0_dz_low_data = gradient_low.data();
    REQUIRE(kernels.dispatch_acoustic_momentum_cartesian_cgrid(md));
    CHECK(w[center] == -std::ldexp(1.0f, -24));
    CHECK(u[center] == 0.0f);
    CHECK(v[center] == 0.0f);
    const auto residual_w = w;
    md.dp0_dz_low_data = w.data();
    REQUIRE_FALSE(kernels.dispatch_acoustic_momentum_cartesian_cgrid(md));
    CHECK(w == residual_w);

    pd.u = u.data();
    pd.v = v.data();
    pd.w = w.data();
    pd.frame_velocity_u_ms = 5.25f;
    pd.frame_velocity_v_ms = -2.125f;
    pd.dx = 1000.0f;
    pd.dy = 750.0f;
    pd.dz = 250.0f;
    pd.dt_small = 1.0f / 30.0f;
    for (int cell = 0; cell < cells; ++cell)
    {
        rho[cell] = 0.7f + 0.0023f * cell;
        u[cell] = 7.0f + 0.011f * cell;
        v[cell] = -1.0f + 0.023f * cell;
        w[cell] = 0.005f * (cell % n);
    }
    REQUIRE(kernels.dispatch_acoustic_pressure_cartesian_cgrid(pd));
    const auto flux = [&](int a, int b, float velocity, float frame)
    {
        return 0.5 * (static_cast<double>(rho[a]) + rho[b]) * (static_cast<double>(velocity) - frame);
    };
    for (int i = 1; i < n - 1; ++i)
        for (int j = 1; j < n - 1; ++j)
            for (int k = 0; k < n; ++k)
            {
                const int cell = (i * n + j) * n + k;
                const double x = (flux(cell, cell + n * n, u[cell], pd.frame_velocity_u_ms) -
                                  flux(cell - n * n, cell, u[cell - n * n], pd.frame_velocity_u_ms)) / pd.dx;
                const double y = (flux(cell, cell + n, v[cell], pd.frame_velocity_v_ms) -
                                  flux(cell - n, cell, v[cell - n], pd.frame_velocity_v_ms)) / pd.dy;
                const double upper = k + 1 < n ? flux(cell, cell + 1, w[cell], 0)
                                               : static_cast<double>(rho[cell]) * w[cell];
                const double lower = k > 0 ? flux(cell - 1, cell, w[cell - 1], 0) : 0.0;
                const float rate = static_cast<float>(-(x + y + (upper - lower) / pd.dz));
                const float reference = static_cast<float>(rho[cell] + static_cast<double>(rate) * pd.dt_small);
                const float ulp = std::nextafter(reference, std::numeric_limits<float>::infinity()) - reference;
                CHECK(std::abs(density_out[cell] - reference) <= ulp);
            }
}
