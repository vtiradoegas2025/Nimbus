#include "catch2/catch.hpp"
#include "compute/backend_runtime.hpp"
#include "microphysics/base/fall_speeds.hpp"
#include <cmath>
#include <limits>
#include <vector>

namespace
{
void exercise(const char* backend)
{
    ComputeBackendRuntimeState runtime;
    runtime.config.backend        = backend;
    runtime.config.allow_fallback = false;
    std::string error;
    const bool  initialized = initialize_compute_backend_runtime(runtime, CoordinateSystem::Cartesian, error);
    INFO(error);
    REQUIRE(initialized);
    auto* kernels = runtime.backend->microphysics_kernels();
    REQUIRE(kernels);
    constexpr int         nx = 5, ny = 4, nz = 7;
    constexpr std::size_t count = nx * ny * nz;
    std::vector<float>    rain(count), zero(count, 0), rho(count), dz(count), rate(count), unused1(count, 0),
        unused2(count, 0);
    for (std::size_t n = 0; n < count; ++n)
    {
        rain[n] = .0001f + float(n % 11) * .00003f;
        rho[n]  = .8f + float(n % 7) * .05f;
        dz[n]   = 20 + float(n % 7) * 11 + float(n % 3);
        rate[n] = float(n % 3) * 1e-9f;
    }
    KesslerSedimentationDispatch d;
    d.qr_data     = rain.data();
    d.qg_data     = zero.data();
    d.qh_data     = zero.data();
    d.rho_d_data  = rho.data();
    d.dqr_dt_data = rate.data();
    d.dqg_dt_data = unused1.data();
    d.dqh_dt_data = unused2.data();
    d.nr          = nx;
    d.nth         = ny;
    d.nz          = nz;
    d.i_begin     = 1;
    d.i_end       = nx - 1;
    d.j_begin     = 1;
    d.j_end       = ny - 1;
    d.k_begin     = 1;
    d.k_end       = nz - 1;
    d.dz_val      = 100;
    d.Vt_max_rain = 9;
    d.vt_cap      = std::numeric_limits<float>::max();
    d.dt_step     = .1f;
    for (bool variable : {false, true})
    {
        CAPTURE(backend, variable);
        d.layer_thickness_m        = variable ? dz.data() : nullptr;
        const auto          before = rate;
        std::vector<double> expected(before.begin(), before.end());
        double              surface_loss = 0;
        for (int i = d.i_begin; i < d.i_end; ++i)
            for (int j = d.j_begin; j < d.j_end; ++j)
                for (int k = d.k_begin; k < d.k_end; ++k)
                {
                    const auto   n = (i * ny + j) * nz + k;
                    const double speed =
                        std::min(double(d.Vt_max_rain),
                                 tmv::fall_speeds::kessler_rain(rain[n], rho[n]) * std::sqrt(1.2 / rho[n]));
                    const double flux = speed * rain[n];
                    expected[n] -= flux / (variable ? dz[n] : d.dz_val);
                    if (k > d.k_begin)
                        expected[n - 1] += rho[n] / double(rho[n - 1]) * flux / (variable ? dz[n - 1] : d.dz_val);
                    else
                        surface_loss += rho[n] * flux;
                }
        REQUIRE(kernels->dispatch_kessler_sedimentation(d));
        double change = 0;
        for (std::size_t n = 0; n < count; ++n)
        {
            CHECK(rate[n] == Approx(expected[n]).margin(4e-12));
            CHECK(unused1[n] == 0);
            CHECK(unused2[n] == 0);
            change += rho[n] * double(variable ? dz[n] : d.dz_val) * (double(rate[n]) - before[n]);
        }
        CHECK(std::abs(change + surface_loss) < 3e-7 * surface_loss);
    }
    const auto before         = rate;
    dz[(1 * ny + 1) * nz + 2] = 0;
    CHECK_FALSE(kernels->dispatch_kessler_sedimentation(d));
    CHECK(rate == before);
    dz[(1 * ny + 1) * nz + 2] = std::numeric_limits<float>::quiet_NaN();
    CHECK_FALSE(kernels->dispatch_kessler_sedimentation(d));
    CHECK(rate == before);
    dz[(1 * ny + 1) * nz + 2]   = 1;
    rain[(1 * ny + 1) * nz + 2] = std::numeric_limits<float>::max();
    CHECK_FALSE(kernels->dispatch_kessler_sedimentation(d));
    CHECK(rate == before);
    d.layer_thickness_m = rate.data();
    CHECK_FALSE(kernels->dispatch_kessler_sedimentation(d));
    CHECK(rate == before);
}
} // namespace
TEST_CASE("Native Kessler sedimentation conserves rain across unequal layers on Metal", "[metal][kessler][layers]")
{
    exercise("metal");
}
TEST_CASE("Native Kessler sedimentation conserves rain across unequal layers on Vulkan", "[vulkan][kessler][layers]")
{
    exercise("vulkan");
}
