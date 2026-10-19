#include "catch2/catch.hpp"
#include "compute/backend_runtime.hpp"
#include "compute/kernel_manifest.hpp"
#include "turbulence/schemes/tke/tke.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <omp.h>

namespace
{
void compare(const Field3D& actual, const Field3D& expected, const char* field)
{
    INFO(field);
    REQUIRE(actual.size() == expected.size());
    for (std::size_t n = 0; n < actual.size(); ++n)
    {
        INFO(n);
        REQUIRE(std::isfinite(actual.data()[n]));
        CHECK(std::abs(actual.data()[n] - expected.data()[n]) <= 1e-6 + 1e-4 * std::abs(expected.data()[n]));
    }
}

void exercise(const char* backend_name)
{
    constexpr int              nx = 7, ny = 6, nz = 5;
    ComputeBackendRuntimeState runtime;
    runtime.config.backend        = backend_name;
    runtime.config.allow_fallback = false;
    std::string error;
    REQUIRE(initialize_compute_backend_runtime(runtime, CoordinateSystem::Cartesian, error));
    INFO(error);
    REQUIRE_FALSE(runtime.fallback_active);
    auto* kernels = runtime.backend->turbulence_kernels();
    REQUIRE(kernels != nullptr);
    REQUIRE(kernels->supports(TurbulenceKernelOp::Tke));
    Field3D u(nx, ny, nz), v(nx, ny, nz), w(nx, ny, nz), theta(nx, ny, nz), rho(nx, ny, nz), qv(nx, ny, nz),
        energy(nx, ny, nz);
    std::vector<double> rho0(nz, 1.0);
    GridMetrics         grid;
    grid.dx = 120;
    grid.dy = 90;
    grid.dz = {40, 50, 65, 80, 100};
    TurbulenceConfig cfg;
    cfg.scheme_id            = "tke";
    cfg.mode                 = "3d";
    cfg.filter_width         = "cubic_root";
    cfg.stability_correction = "tke_based";
    cfg.Pr_t                 = 1.0 / 3.0;
    cfg.Sc_t                 = 0.5;
    cfg.nu_t_max             = 1000;
    cfg.K_max                = 1000;
    TurbulenceStateView state;
    state.u          = &u;
    state.v          = &v;
    state.w          = &w;
    state.theta      = &theta;
    state.rho        = &rho;
    state.qv         = &qv;
    state.tke        = &energy;
    state.rho0       = &rho0;
    state.NR         = nx;
    state.NTH        = ny;
    state.NZ         = nz;
    state.c_grid     = true;
    state.periodic_x = true;
    state.periodic_y = true;
    state.open_lateral.fill(false);
    for (const char* correction : {"none", "tke_based"})
        for (const char* filter : {"cubic_root", "dx"})
            for (bool moist : {false, true})
                for (double slope : {-0.2, 0.0, 0.2})
                    for (const char* mode : {"3d", "horizontal_only", "vertical_only"})
                    {
                        INFO(correction);
                        INFO(filter);
                        cfg.stability_correction = correction;
                        cfg.filter_width         = filter;
                        INFO(moist);
                        INFO(slope);
                        INFO(mode);
                        cfg.enable_moist_buoyancy = moist;
                        cfg.mode                  = mode;
                        for (int i = 0; i < nx; ++i)
                            for (int j = 0; j < ny; ++j)
                                for (int k = 0; k < nz; ++k)
                                {
                                    u(i, j, k)      = 2.0f + 0.1f * k + 0.2f * std::sin(i + 0.4 * j);
                                    v(i, j, k)      = -1.0f + 0.05f * k + 0.1f * std::cos(j + 0.3 * i);
                                    w(i, j, k)      = k == nz - 1 ? 0.0f : 0.02f * std::sin(i + j + k);
                                    theta(i, j, k)  = static_cast<float>(300 + slope * k + 0.01 * i);
                                    rho(i, j, k)    = 1.2f - 0.1f * k + 0.01f * i;
                                    qv(i, j, k)     = 0.008f - 0.0001f * k;
                                    energy(i, j, k) = (i == 3 && j == 2) ? 0.0f : 0.4f + 0.02f * k + 0.01f * i;
                                }
                        TKEScheme             cpu, gpu;
                        TurbulenceTendencies  expected, actual;
                        TurbulenceDiagnostics cpu_diag, gpu_diag;
                        state.compute_runtime         = nullptr;
                        state.require_native_dispatch = false;
                        cpu.compute(cfg, grid, state, expected, &cpu_diag);
                        state.compute_runtime         = &runtime;
                        state.require_native_dispatch = true;
                        gpu.compute(cfg, grid, state, actual, &gpu_diag);
                        compare(actual.dudt_sgs, expected.dudt_sgs, "u");
                        compare(actual.dvdt_sgs, expected.dvdt_sgs, "v");
                        compare(actual.dwdt_sgs, expected.dwdt_sgs, "w");
                        compare(actual.dthetadt_sgs, expected.dthetadt_sgs, "theta");
                        compare(actual.dqvdt_sgs, expected.dqvdt_sgs, "qv");
                        compare(actual.dtkedt_sgs, expected.dtkedt_sgs, "tke");
                        compare(gpu_diag.nu_t, cpu_diag.nu_t, "nu");
                        compare(gpu_diag.K_theta, cpu_diag.K_theta, "heat");
                        compare(gpu_diag.K_q, cpu_diag.K_q, "vapor");
                        compare(gpu_diag.K_tke, cpu_diag.K_tke, "energy diffusivity");
                        compare(gpu_diag.shear_prod, cpu_diag.shear_prod, "shear");
                        compare(gpu_diag.buoy_prod, cpu_diag.buoy_prod, "buoyancy");
                        compare(gpu_diag.dissipation, cpu_diag.dissipation, "dissipation");
                        compare(gpu_diag.Pr_t_eff, cpu_diag.Pr_t_eff, "Prandtl");
                        double transport = 0;
                        for (int i = 1; i < nx - 1; ++i)
                            for (int j = 1; j < ny - 1; ++j)
                                for (int k = 0; k < nz; ++k)
                                    transport += rho(i, j, k) * grid.dz[k] *
                                                 (actual.dtkedt_sgs(i, j, k) - gpu_diag.shear_prod(i, j, k) -
                                                  gpu_diag.buoy_prod(i, j, k) + gpu_diag.dissipation(i, j, k));
                        CHECK(std::abs(transport) < 2e-4);
                        const auto saved             = actual.dtkedt_sgs;
                        const auto saved_coefficient = *gpu.realized_eddy_viscosity();
                        energy(2, 2, 2)              = -1.0f;
                        CHECK_THROWS(gpu.compute(cfg, grid, state, actual, &gpu_diag));
                        CHECK(std::equal(saved.data(), saved.data() + saved.size(), actual.dtkedt_sgs.data()));
                        CHECK(std::equal(saved_coefficient.data(),
                                         saved_coefficient.data() + saved_coefficient.size(),
                                         gpu.realized_eddy_viscosity()->data()));
                    }
    const auto count = runtime.backend->successful_kernel_dispatch_count(nimbus::compute::KernelId::TkeClosure);
    CHECK(count == 72);
    CHECK(runtime.backend->successful_kernel_dispatch_count(nimbus::compute::KernelId::TkeTendency) == count);
    energy(2, 2, 2)            = 0.4f;
    const int previous_threads = omp_get_max_threads();
    struct RestoreThreads
    {
        int count;
        ~RestoreThreads()
        {
            omp_set_num_threads(count);
        }
    } restore{previous_threads};
    TKEScheme            threaded;
    TurbulenceTendencies one, four;
    omp_set_num_threads(1);
    threaded.compute(cfg, grid, state, one, nullptr);
    omp_set_num_threads(4);
    threaded.compute(cfg, grid, state, four, nullptr);
    const std::array<const Field3D*, 6> first{
        &one.dudt_sgs, &one.dvdt_sgs, &one.dwdt_sgs, &one.dthetadt_sgs, &one.dqvdt_sgs, &one.dtkedt_sgs};
    const std::array<const Field3D*, 6> second{
        &four.dudt_sgs, &four.dvdt_sgs, &four.dwdt_sgs, &four.dthetadt_sgs, &four.dqvdt_sgs, &four.dtkedt_sgs};
    for (std::size_t f = 0; f < first.size(); ++f)
        CHECK(std::equal(first[f]->data(), first[f]->data() + first[f]->size(), second[f]->data()));
    std::array<std::vector<float>, 8> outputs;
    for (auto& field : outputs)
        field.assign(u.size(), -91.0f);
    std::vector<float> dz(grid.dz.begin(), grid.dz.end());
    TkeClosureDispatch dispatch;
    dispatch.u                 = u.data();
    dispatch.v                 = v.data();
    dispatch.w                 = w.data();
    dispatch.theta             = theta.data();
    dispatch.qv                = qv.data();
    dispatch.tke               = energy.data();
    dispatch.dz                = dz.data();
    dispatch.nu                = outputs[0].data();
    dispatch.K_theta           = outputs[1].data();
    dispatch.K_q               = outputs[2].data();
    dispatch.K_tke             = outputs[3].data();
    dispatch.shear             = outputs[4].data();
    dispatch.buoyancy          = outputs[5].data();
    dispatch.dissipation       = outputs[6].data();
    dispatch.effective_prandtl = outputs[7].data();
    dispatch.nx                = nx;
    dispatch.ny                = ny;
    dispatch.nz                = nz;
    dispatch.profile_length    = nz;
    dispatch.dx                = 120;
    dispatch.dy                = 90;
    dispatch.Pr_t              = 1.0f / 3.0f;
    dispatch.Sc_t              = 0.5f;
    dispatch.nu_max            = 1000;
    dispatch.K_max             = 1000;
    const auto before          = outputs;
    const auto original        = dispatch;
    dispatch.nu                = dispatch.K_theta + 1;
    CHECK_FALSE(kernels->dispatch_tke_closure(dispatch));
    CHECK(outputs == before);
    dispatch    = original;
    dispatch.nu = u.data();
    CHECK_FALSE(kernels->dispatch_tke_closure(dispatch));
    CHECK(outputs == before);
    dispatch = original;
    dz[1]    = 0;
    CHECK_FALSE(kernels->dispatch_tke_closure(dispatch));
    CHECK(outputs == before);
    dz[1]      = 50;
    u(2, 2, 2) = std::numeric_limits<float>::quiet_NaN();
    CHECK_FALSE(kernels->dispatch_tke_closure(dispatch));
    CHECK(outputs == before);
    u(2, 2, 2)  = 2;
    dispatch.nx = std::numeric_limits<int>::max();
    CHECK_FALSE(kernels->dispatch_tke_closure(dispatch));
    CHECK(outputs == before);
    auto* dynamics = runtime.backend->dynamics_kernels();
    REQUIRE(dynamics != nullptr);
    Field3D pressure(nx, ny, nz, 100000.0f), rate(nx, ny, nz, 0.0f);
    rate(0, 0, 0) = std::numeric_limits<float>::quiet_NaN();
    SlowApplyDispatch slow;
    slow.u      = u.data();
    slow.v      = v.data();
    slow.w      = w.data();
    slow.p_data = pressure.data();
    slow.du_dt  = rate.data();
    slow.dv_dt  = rate.data();
    slow.dw_dt  = rate.data();
    slow.dp_dt  = rate.data();
    std::vector<float> zero(u.size(), 0.0f);
    slow.du_dt_pbl         = zero.data();
    slow.dv_dt_pbl         = zero.data();
    slow.nx                = nx;
    slow.ny                = ny;
    slow.nz                = nz;
    slow.dt_large          = 0.1f;
    slow.sanitize          = false;
    const Field3D before_u = u, before_v = v, before_w = w, before_p = pressure;
    slow.completed_observer = +[](void*, const float* cu, const float* cv, const float* cw, const float* cp)
    {
        REQUIRE(std::isnan(cu[0]));
        REQUIRE(std::isnan(cv[0]));
        REQUIRE(std::isnan(cw[0]));
        REQUIRE(std::isnan(cp[0]));
        throw std::runtime_error("reject raw slow candidate");
    };
    REQUIRE_THROWS_WITH(dynamics->dispatch_slow_apply(slow), "reject raw slow candidate");
    for (const auto pair : {std::pair<const Field3D*, const Field3D*>{&u, &before_u},
                            {&v, &before_v},
                            {&w, &before_w},
                            {&pressure, &before_p}})
        CHECK(std::equal(pair.first->data(), pair.first->data() + pair.first->size(), pair.second->data()));
    slow.completed_observer = nullptr;
    slow.sanitize           = true;
    REQUIRE(dynamics->dispatch_slow_apply(slow));
    CHECK(u(0, 0, 0) == before_u(0, 0, 0));
    CHECK(pressure(0, 0, 0) == before_p(0, 0, 0));

    grid.dx = grid.dy = 1000;
    grid.dz.assign(nz, 250);
    cfg.mode                  = "3d";
    cfg.filter_width          = "cubic_root";
    cfg.stability_correction  = "tke_based";
    cfg.enable_moist_buoyancy = false;
    cfg.Pr_t = cfg.Sc_t   = 1.0 / 3.0;
    const float theta_ulp = std::nextafter(300.0f, 301.0f) - 300.0f;
    for (int slope_ulps : {32, 36, 40})
    {
        for (int i = 0; i < nx; ++i)
            for (int j = 0; j < ny; ++j)
                for (int k = 0; k < nz; ++k)
                {
                    u(i, j, k)      = 5 + .75f * k;
                    v(i, j, k)      = 2 + .5f * k;
                    w(i, j, k)      = 0;
                    theta(i, j, k)  = 300.0f + slope_ulps * theta_ulp * k;
                    rho(i, j, k)    = 1.1f;
                    qv(i, j, k)     = .008f;
                    energy(i, j, k) = .1f;
                }
        TKEScheme             cpu, gpu;
        TurbulenceTendencies  expected, actual;
        TurbulenceDiagnostics expected_diag, actual_diag;
        state.compute_runtime         = nullptr;
        state.require_native_dispatch = false;
        cpu.compute(cfg, grid, state, expected, &expected_diag);
        state.compute_runtime         = &runtime;
        state.require_native_dispatch = true;
        gpu.compute(cfg, grid, state, actual, &actual_diag);
        for (std::size_t n = 0; n < energy.size(); ++n)
        {
            CAPTURE(slope_ulps, n);
            REQUIRE(std::isfinite(actual.dtkedt_sgs.data()[n]));
            CHECK(std::abs(actual.dtkedt_sgs.data()[n] - expected.dtkedt_sgs.data()[n]) <=
                  64 * std::numeric_limits<float>::epsilon() *
                      std::max(1e-5f, std::abs(expected.dtkedt_sgs.data()[n])));
            CHECK(std::abs(actual_diag.K_theta.data()[n] - expected_diag.K_theta.data()[n]) <=
                  64 * std::numeric_limits<float>::epsilon() *
                      std::max(1e-5f, std::abs(expected_diag.K_theta.data()[n])));
        }
    }
}
} // namespace

TEST_CASE("Metal prognostic TKE agrees with the CPU closure", "[native-tke][metal]")
{
    exercise("metal");
}
TEST_CASE("Vulkan prognostic TKE agrees with the CPU closure", "[native-tke][vulkan]")
{
    exercise("vulkan");
}
