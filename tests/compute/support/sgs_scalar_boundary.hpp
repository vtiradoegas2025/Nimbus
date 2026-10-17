#pragma once

#include "compute/compute_kernel_interfaces.hpp"
#include "core/field/field3d.hpp"
#include <array>
#include <cmath>
#include <vector>

inline void exercise_sgs_scalar_boundary(TurbulenceComputeKernels& kernels)
{
    constexpr int n = 6;
    Field3D zero(n, n, n, 0.0f), rho(n, n, n, 1.0f), diffusivity(n, n, n, 2.0f);
    Field3D theta(n, n, n, 308.0f), vapor(n, n, n, 0.03f), energy(n, n, n, 11.0f);
    std::array<Field3D, 5> rates;
    for (auto& field : rates)
        field.resize(n, n, n, -99.0f);
    Field3D energy_rate(n, n, n, -99.0f);
    for (int i = 1; i < n - 1; ++i)
        for (int j = 1; j < n - 1; ++j)
            for (int k = 0; k < n; ++k)
            {
                theta(i, j, k) = 300.0f;
                vapor(i, j, k) = 0.01f;
                energy(i, j, k) = 3.0f;
            }
    std::vector<float> dz(n, 4.0f), dzc(n, 8.0f);
    dzc.front() = dzc.back() = 4.0f;
    SmagorinskyTendenciesDispatch transport;
    transport.u = transport.v = transport.w = zero.data();
    transport.theta_data = theta.data();
    transport.qv_data = vapor.data();
    transport.nu_t = transport.K_theta = transport.K_q = diffusivity.data();
    transport.rho = rho.data();
    transport.dz_prof = dz.data();
    transport.prof_len = n;
    transport.nx = transport.ny = transport.nz = n;
    transport.dx = transport.dy = 4.0f;
    transport.use_horizontal = true;
    transport.has_qv = true;
    transport.dudt_sgs = rates[0].data();
    transport.dvdt_sgs = rates[1].data();
    transport.dwdt_sgs = rates[2].data();
    transport.dthetadt_sgs = rates[3].data();
    transport.dqvdt_sgs = rates[4].data();
    TkeTendencyDispatch tke;
    tke.rho = rho.data();
    tke.tke = energy.data();
    tke.K_tke = diffusivity.data();
    tke.shear = tke.buoyancy = tke.dissipation = zero.data();
    tke.dz = dz.data();
    tke.tendency = energy_rate.data();
    tke.nx = tke.ny = tke.nz = tke.profile_length = n;
    tke.dx = tke.dy = 4.0f;
    tke.use_horizontal = true;
    tke.use_vertical = false;
    for (int mode = 0; mode < 3; ++mode)
    {
        CAPTURE(mode);
        transport.open_lateral = {mode != 0, false, false, false};
        transport.periodic_x = mode == 2;
        tke.open_lateral = transport.open_lateral;
        tke.periodic_x = transport.periodic_x;
        REQUIRE(kernels.dispatch_smagorinsky_tendencies(transport));
        REQUIRE(kernels.dispatch_tke_tendency(tke));
        for (int i = 1; i < n - 1; ++i)
            for (int j = 1; j < n - 1; ++j)
                for (int k = 0; k < n; ++k)
                {
                    const bool inflow = mode == 1 && i == 1;
                    CHECK(rates[3](i, j, k) == (inflow ? 1.0f : 0.0f));
                    CHECK(std::abs(rates[4](i, j, k) - (inflow ? 0.0025f : 0.0f)) < 1e-9f);
                    CHECK(energy_rate(i, j, k) == (inflow ? 1.0f : 0.0f));
                }
    }
    Field3D wind(n, n, n), nu(n, n, n), kth(n, n, n), kq(n, n, n);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            for (int k = 0; k < n; ++k)
                wind(i, j, k) = 0.1f * j;
    SmagorinskyFusedDispatch fused;
    fused.u_data = wind.data();
    fused.v_data = fused.w_data = zero.data();
    fused.theta_data = theta.data();
    fused.qv_data = vapor.data();
    fused.dz_prof = dz.data();
    fused.dzc_prof = dzc.data();
    fused.rho = rho.data();
    fused.prof_len = fused.nx = fused.ny = fused.nz = n;
    fused.nu_t = nu.data();
    fused.K_theta = kth.data();
    fused.K_q = kq.data();
    fused.dudt_sgs = rates[0].data();
    fused.dvdt_sgs = rates[1].data();
    fused.dwdt_sgs = rates[2].data();
    fused.dthetadt_sgs = rates[3].data();
    fused.dqvdt_sgs = rates[4].data();
    fused.dx_val = fused.dy_val = 4.0f;
    fused.use_horizontal = fused.has_qv = true;
    fused.Cs = 0.18f;
    fused.Pr_t = 0.7f;
    fused.Sc_t = 1.0f;
    fused.nu_t_max = fused.K_max = 10.0f;
    fused.open_lateral = {false, false, false, false};
    REQUIRE(kernels.dispatch_smagorinsky_fused(fused));
    REQUIRE(kth(2, 2, 2) > 0.0f);
    for (std::size_t index = 0; index < theta.size(); ++index)
    {
        CHECK(rates[3].data()[index] == 0.0f);
        CHECK(rates[4].data()[index] == 0.0f);
    }
}
