#include "catch2/catch.hpp"
#include "compute/backend_runtime.hpp"
#include "compute/collocated_sgs_contract.hpp"
#include "turbulence/schemes/tke/tke.hpp"
#include "turbulence/schemes/smagorinsky/smagorinsky.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>

namespace
{
void exercise(const char* backend)
{
    ComputeBackendRuntimeState runtime;
    runtime.config.backend        = backend;
    runtime.config.allow_fallback = false;
    std::string error;
    const bool  ready = initialize_compute_backend_runtime(runtime, CoordinateSystem::Cartesian, error);
    INFO(error);
    REQUIRE(ready);
    REQUIRE_FALSE(runtime.fallback_active);
    REQUIRE(runtime.backend->turbulence_kernels()->supports(TurbulenceKernelOp::CollocatedSgs));
    constexpr int nx = 7, ny = 6, nz = 7;
    Field3D       u(nx, ny, nz), v(nx, ny, nz), w(nx, ny, nz), rho(nx, ny, nz), theta(nx, ny, nz), qv(nx, ny, nz),
        energy(nx, ny, nz);
    GridMetrics grid;
    grid.dx         = 120;
    grid.dy         = 90;
    grid.dz         = {40, 50, 65, 80, 100, 110, 120};
    grid.staggering = StaggerType::Collocated;
    std::vector<double> rho0(nz, 1.0);
    TurbulenceStateView state;
    state.u      = &u;
    state.v      = &v;
    state.w      = &w;
    state.rho    = &rho;
    state.theta  = &theta;
    state.qv     = &qv;
    state.tke    = &energy;
    state.rho0   = &rho0;
    state.NR     = nx;
    state.NTH    = ny;
    state.NZ     = nz;
    state.c_grid = false;
    state.open_lateral.fill(false);
    TurbulenceConfig cfg;
    cfg.Cs              = .125;
    cfg.Pr_t            = .5;
    cfg.Sc_t            = .75;
    std::uint64_t calls = 0;
    for (bool tke : {false, true})
        for (bool periodic : {false, true})
            for (bool moist : {false, true})
                for (const char* mode : {"3d", "horizontal_only", "vertical_only"})
                    for (double slope : {-.2, .2})
                    {
                        CAPTURE(backend, tke, periodic, moist, mode, slope);
                        cfg.scheme_id             = tke ? "tke" : "smagorinsky";
                        cfg.mode                  = mode;
                        cfg.enable_moist_buoyancy = moist;
                        cfg.stability_correction  = tke ? "tke_based" : "none";
                        state.periodic_x = state.periodic_y = periodic;
                        for (int i = 0; i < nx; ++i)
                            for (int j = 0; j < ny; ++j)
                                for (int k = 0; k < nz; ++k)
                                {
                                    bool ghost =
                                        i == 0 || j == 0 || k == 0 || i == nx - 1 || j == ny - 1 || k == nz - 1;
                                    u(i, j, k)      = 2 + .1f * k + .2f * std::sin(i + .4 * j);
                                    v(i, j, k)      = -1 + .05f * k + .1f * std::cos(j + .3 * i);
                                    w(i, j, k)      = .03f * std::sin(i + j + k);
                                    rho(i, j, k)    = 1.2f - .1f * k + .01f * i;
                                    theta(i, j, k)  = float(300 + slope * k + .01 * i);
                                    qv(i, j, k)     = .008f - .0001f * k;
                                    energy(i, j, k) = i == 3 && j == 2 ? 0 : .4f + .02f * k + .01f * i;
                                    if (ghost)
                                        for (Field3D* field : {&u, &v, &w, &rho, &theta, &qv, &energy})
                                            (*field)(i, j, k) = std::numeric_limits<float>::quiet_NaN();
                                }
                        auto create = [&]() -> std::unique_ptr<TurbulenceSchemeBase>
                        {
                            if (tke)
                                return std::make_unique<TKEScheme>();
                            return std::make_unique<tmv::SmagorinskyScheme>();
                        };
                        auto cpu = create(), gpu = create();
                        cpu->initialize(cfg);
                        gpu->initialize(cfg);
                        TurbulenceTendencies  expected, actual;
                        TurbulenceDiagnostics reference, observed;
                        state.compute_runtime         = nullptr;
                        state.require_native_dispatch = false;
                        cpu->compute(cfg, grid, state, expected, &reference);
                        state.compute_runtime         = &runtime;
                        state.require_native_dispatch = true;
                        gpu->compute(cfg, grid, state, actual, &observed);
                        calls += 2;
                        const std::array<const Field3D*, 13> lhs{&actual.dudt_sgs,
                                                                 &actual.dvdt_sgs,
                                                                 &actual.dwdt_sgs,
                                                                 &actual.dthetadt_sgs,
                                                                 &actual.dqvdt_sgs,
                                                                 &actual.dtkedt_sgs,
                                                                 &observed.nu_t,
                                                                 &observed.K_theta,
                                                                 &observed.K_q,
                                                                 &observed.K_tke,
                                                                 &observed.shear_prod,
                                                                 &observed.buoy_prod,
                                                                 &observed.dissipation};
                        const std::array<const Field3D*, 13> rhs{&expected.dudt_sgs,
                                                                 &expected.dvdt_sgs,
                                                                 &expected.dwdt_sgs,
                                                                 &expected.dthetadt_sgs,
                                                                 &expected.dqvdt_sgs,
                                                                 &expected.dtkedt_sgs,
                                                                 &reference.nu_t,
                                                                 &reference.K_theta,
                                                                 &reference.K_q,
                                                                 &reference.K_tke,
                                                                 &reference.shear_prod,
                                                                 &reference.buoy_prod,
                                                                 &reference.dissipation};
                        for (std::size_t f = 0; f < lhs.size(); ++f)
                        {
                            REQUIRE(lhs[f]->size() == rhs[f]->size());
                            double scale = 0, max_error = 0;
                            for (std::size_t n = 0; n < lhs[f]->size(); ++n)
                            {
                                REQUIRE(std::isfinite(lhs[f]->data()[n]));
                                max_error =
                                    std::max(max_error, std::abs(double(lhs[f]->data()[n]) - rhs[f]->data()[n]));
                                scale = std::max(scale, std::abs(double(rhs[f]->data()[n])));
                            }
                            CAPTURE(f, max_error, scale);
                            CHECK(max_error <= 2e-6 * std::max(scale, 1e-12));
                        }
                        double momentum[3]{}, scalar = 0, work = 0, shear = 0, work_scale = 0;
                        for (int i = 1; i < nx - 1; ++i)
                            for (int j = 1; j < ny - 1; ++j)
                                for (int k = 1; k < nz - 1; ++k)
                                {
                                    double mass = double(rho(i, j, k)) * grid.dx * grid.dy * grid.dz[k];
                                    double a    = mass * (double(u(i, j, k)) * actual.dudt_sgs(i, j, k) +
                                                          double(v(i, j, k)) * actual.dvdt_sgs(i, j, k) +
                                                          double(w(i, j, k)) * actual.dwdt_sgs(i, j, k));
                                    work += a;
                                    work_scale += std::abs(a);
                                    shear += mass * observed.shear_prod(i, j, k);
                                    momentum[0] += mass * actual.dudt_sgs(i, j, k);
                                    momentum[1] += mass * actual.dvdt_sgs(i, j, k);
                                    momentum[2] += mass * actual.dwdt_sgs(i, j, k);
                                    scalar += mass * actual.dthetadt_sgs(i, j, k);
                                }
                        for (double residual : momentum)
                            CHECK(std::abs(residual) <= 2e-6 * std::max(work_scale, 1.));
                        CHECK(std::abs(scalar) <= 2e-6 * std::max(work_scale, 1.));
                        if (tke)
                            CHECK(std::abs(work + shear) <= 2e-6 * std::max(work_scale, 1.));
                    }
    std::array<Field3D, 8> output;
    Field3D                zero(nx, ny, nz, 0), dz(nx, ny, nz, 50);
    CollocatedSgsDispatch  dispatch;
    dispatch.nx     = nx;
    dispatch.ny     = ny;
    dispatch.nz     = nz;
    dispatch.dx     = 120;
    dispatch.dy     = 90;
    dispatch.Cs     = .125;
    dispatch.Pr     = .5;
    dispatch.Sc     = .75;
    dispatch.nu_max = dispatch.K_max = 1000;
    dispatch.has_qv                  = true;
    dispatch.input.fill(zero.data());
    dispatch.input[0] = u.data();
    dispatch.input[1] = v.data();
    dispatch.input[2] = w.data();
    dispatch.input[3] = theta.data();
    dispatch.input[4] = qv.data();
    dispatch.input[5] = energy.data();
    dispatch.input[6] = rho.data();
    dispatch.input[7] = dz.data();
    for (int f = 0; f < 8; ++f)
    {
        output[f].resize(nx, ny, nz, 17);
        dispatch.output[f] = output[f].data();
    }
    const auto rejected = [&]()
    {
        CHECK_FALSE(runtime.backend->turbulence_kernels()->dispatch_collocated_sgs(dispatch));
        for (auto& field : output)
            for (std::size_t n = 0; n < field.size(); ++n)
                CHECK(field.data()[n] == 17);
    };
    const float old_density = rho(2, 2, 2);
    rho(2, 2, 2)            = 0;
    rejected();
    rho(2, 2, 2)       = old_density;
    dispatch.output[1] = dispatch.output[0] + 1;
    rejected();
    dispatch.output[1] = output[1].data();
    dispatch.nx        = std::numeric_limits<int>::max();
    rejected();
    dispatch.nx = nx;
    u(2, 2, 2)  = std::numeric_limits<float>::max();
    u(4, 2, 2)  = -std::numeric_limits<float>::max();
    rejected();
    ++calls;
    std::array<Field3D, 6>     committed, rates, before;
    const std::array<float, 6> initial{2, -1, .1f, 300, .01f, .2f};
    const std::array<float, 6> rate{.1f, .2f, .05f, .01f, .000001f, .001f};
    for (int f = 0; f < 6; ++f)
    {
        committed[f].resize(nx, ny, nz, initial[f]);
        rates[f].resize(nx, ny, nz, 10000);
        for (int i = 1; i < nx - 1; ++i)
            for (int j = 1; j < ny - 1; ++j)
                for (int k = 1; k < nz - 1; ++k)
                    rates[f](i, j, k) = rate[f];
        before[f] = committed[f];
    }
    SgsApplyDispatch apply;
    apply.u              = committed[0].data();
    apply.v              = committed[1].data();
    apply.w              = committed[2].data();
    apply.theta_data     = committed[3].data();
    apply.qv_data        = committed[4].data();
    apply.tke_data       = committed[5].data();
    apply.dudt           = rates[0].data();
    apply.dvdt           = rates[1].data();
    apply.dwdt           = rates[2].data();
    apply.dthetadt       = rates[3].data();
    apply.dqvdt          = rates[4].data();
    apply.dtkedt         = rates[5].data();
    apply.dtkedt_pbl     = zero.data();
    apply.nx             = nx;
    apply.ny             = ny;
    apply.nz             = nz;
    apply.has_qv         = true;
    apply.has_tke        = true;
    apply.dt_step        = .25f;
    apply.theta_fallback = 300;
    apply.theta_min      = 100;
    apply.theta_max      = 400;
    apply.qv_max_val     = .05f;
    apply.wind_h_max     = 100;
    apply.wind_v_max     = 100;
    apply.sanitize       = false;
    apply.collocated     = true;
    REQUIRE(runtime.backend->turbulence_kernels()->dispatch_sgs_apply(apply));
    for (int f = 0; f < 6; ++f)
        for (int i = 0; i < nx; ++i)
            for (int j = 0; j < ny; ++j)
                for (int k = 0; k < nz; ++k)
                {
                    if (i == 0 || i == nx - 1 || j == 0 || j == ny - 1 || k == 0 || k == nz - 1)
                        CHECK(committed[f](i, j, k) == before[f](i, j, k));
                    else
                        CHECK(committed[f](i, j, k) == Approx(initial[f] + .25f * rate[f]).margin(1e-7));
                }
    CHECK(runtime.backend->successful_kernel_dispatch_count(nimbus::compute::KernelId::CollocatedSgs) == calls);
}
} // namespace
TEST_CASE("Native collocated SGS preserves stress and TKE transfer on Metal", "[metal][sgs]")
{
    exercise("metal");
}
TEST_CASE("Native collocated SGS preserves stress and TKE transfer on Vulkan", "[vulkan][sgs]")
{
    exercise("vulkan");
}
