#include "catch2/catch.hpp"
#include "compute/backend_runtime.hpp"
#include "compute/kernel_manifest.hpp"
#include "compute/milbrandt_warm_contract.hpp"
#include "microphysics/schemes/milbrandt/warm_rain.hpp"
#include <array>
#include <cmath>
#include <cstring>
#include <vector>

namespace
{
void exercise(const char* name)
{
    using namespace tmv::milbrandt_detail;
    constexpr int                     nx = 6, ny = 5, nz = 8, cells = nx * ny * nz;
    std::array<std::vector<float>, 8> state;
    std::array<std::vector<float>, 6> rates, actual, expected;
    for (auto& f : state)
        f.resize(cells);
    for (auto* fields : {&rates, &actual, &expected})
        for (auto& f : *fields)
            f.resize(cells);
    const auto identical = [&](const auto& reference)
    {
        for (int f = 0; f < 6; ++f)
            CHECK(std::memcmp(actual[f].data(), reference[f].data(), cells * sizeof(float)) == 0);
    };
    MilbrandtWarmRainDispatch dispatch;
    dispatch.nx      = nx;
    dispatch.ny      = ny;
    dispatch.nz      = nz;
    dispatch.i_begin = 1;
    dispatch.i_end   = nx - 1;
    dispatch.j_begin = 1;
    dispatch.j_end   = ny - 1;
    dispatch.k_end   = nz;
    for (int f = 0; f < 8; ++f)
        dispatch.state[f] = state[f].data();
    for (int f = 0; f < 6; ++f)
    {
        dispatch.rates[f]  = rates[f].data();
        dispatch.output[f] = actual[f].data();
    }
    ComputeBackendRuntimeState runtime;
    runtime.config.backend        = name;
    runtime.config.allow_fallback = false;
    std::string error;
    REQUIRE(initialize_compute_backend_runtime(runtime, CoordinateSystem::Cartesian, error));
    REQUIRE_FALSE(runtime.fallback_active);
    auto* kernels = runtime.backend->microphysics_kernels();
    REQUIRE(kernels);
    REQUIRE(kernels->supports(MicrophysicsKernelOp::MilbrandtWarmRain));
    const auto    parameters = warm_rain_parameters(0, 523.598775598, 3, 149.1, .5);
    std::uint64_t calls      = 0;
    for (double dt : {.2, 1.0, 10.0})
    {
        dispatch.dt = dt;
        for (int n = 0; n < cells; ++n)
        {
            const float  temperature = 250.0f + 10.0f * (n % 7), pressure = 60000.0f + 5000.0f * (n % 8);
            const float  cloud    = n % 3 == 0 ? 0.0f : (n % 3 == 1 ? .001f : .004f);
            const float  rain     = n % 5 == 0 ? 0.0f : .001f;
            const double diameter = std::array<double, 6>{60e-6, 90e-6, 100e-6, 500e-6, 3.5e-3, 6e-3}[n % 6];
            state[0][n]           = temperature;
            state[1][n]           = pressure;
            state[2][n]           = .3f + .15f * (n % 7);
            const double qs       = thermodynamics::saturation_mixing_ratio_water(temperature, pressure);
            state[3][n]           = float(qs * (n % 4 == 0 ? 0 : n % 4 == 1 ? .5 : n % 4 == 2 ? 1.0 : 1.05));
            state[4][n]           = cloud;
            state[5][n]           = rain;
            state[6][n]           = n % 3 == 0 ? 0 : n % 3 == 1 ? 2e8f : 1e5f;
            state[7][n]           = float(rain / (parameters.c_r * std::pow(diameter, 3)));
            for (int f = 0; f < 6; ++f)
                rates[f][n] = 0;
            rates[0][n] = 1e-6f;
            rates[2][n] = -1e-6f;
            rates[3][n] = .01f;
            rates[4][n] = .125f;
            if (n % 3 == 2 && n % 2 == 0)
            {
                state[6][n] = 0;
                rates[4][n] = float(1e5 / dt);
            }
            if (n == (2 * ny + 2) * nz + 1)
                state[7][n] = 1e-4f / state[2][n];
            if (n == (2 * ny + 2) * nz + 2)
            {
                state[4][n] = state[5][n] = state[7][n] = 0;
                state[6][n]                             = 123456784.0f;
                rates[4][n] = std::nextafter(float(-double(state[6][n]) / dt), -std::numeric_limits<float>::infinity());
            }
            WarmRainCell cell{
                state[0][n], state[1][n], state[2][n], state[3][n], state[4][n], state[5][n], state[6][n], state[7][n]};
            WarmRainTendencies result{rates[0][n], rates[1][n], rates[2][n], rates[3][n], rates[4][n], rates[5][n]};
            const int          i = n / (ny * nz), j = n / nz % ny;
            if (i >= 1 && i < nx - 1 && j >= 1 && j < ny - 1)
                warm_rain_cell(parameters, cell, dt, result);
            const std::array<float, 6> packed{
                result.cloud, result.rain, result.vapor, result.temperature, result.cloud_number, result.rain_number};
            for (int f = 0; f < 6; ++f)
            {
                expected[f][n] = packed[f];
                actual[f][n]   = 17;
            }
        }
        REQUIRE(kernels->dispatch_milbrandt_warm_rain(dispatch));
        ++calls;
        for (int f = 0; f < 6; ++f)
            for (int n = 0; n < cells; ++n)
            {
                CAPTURE(name, dt, f, n, actual[f][n], expected[f][n]);
                REQUIRE(std::isfinite(actual[f][n]));
                const double floor = f < 3 ? 2e-11 : f == 3 ? 5e-8 : .05;
                CHECK(std::abs(double(actual[f][n]) - expected[f][n]) <=
                      floor + 2e-4 * std::abs(double(expected[f][n])));
            }
        const auto first = actual;
        REQUIRE(kernels->dispatch_milbrandt_warm_rain(dispatch));
        ++calls;
        identical(first);
    }
    const auto accepted         = actual;
    dispatch.dt                 = 1e-36;
    const int active            = (2 * ny + 2) * nz;
    state[4][active]            = 0;
    state[5][active]            = .001f;
    state[7][active]            = float(.001 / (parameters.c_r * std::pow(60e-6, 3)));
    rates[0][active]            = std::numeric_limits<float>::max();
    std::size_t validated_cells = 0;
    REQUIRE(nimbus::compute::milbrandt_warm::validate(dispatch, validated_cells));
    CHECK_FALSE(kernels->dispatch_milbrandt_warm_rain(dispatch));
    ++calls;
    identical(accepted);
    dispatch.dt            = 10;
    rates[0][active]       = 1e-6f;
    const auto before      = actual;
    const auto unchanged   = [&] { identical(before); };
    dispatch.parameters[0] = 1;
    CHECK_FALSE(kernels->dispatch_milbrandt_warm_rain(dispatch));
    unchanged();
    dispatch.parameters[0] = 0;
    dispatch.output[5]     = actual[0].data();
    CHECK_FALSE(kernels->dispatch_milbrandt_warm_rain(dispatch));
    unchanged();
    dispatch.output[5] = actual[5].data();
    dispatch.output[5] = state[0].data();
    CHECK_FALSE(kernels->dispatch_milbrandt_warm_rain(dispatch));
    unchanged();
    dispatch.output[5] = actual[5].data();
    dispatch.nx        = std::numeric_limits<int>::max();
    dispatch.ny        = std::numeric_limits<int>::max();
    CHECK_FALSE(kernels->dispatch_milbrandt_warm_rain(dispatch));
    unchanged();
    dispatch.nx                                           = nx;
    dispatch.ny                                           = ny;
    state[0][cells - nz - 1]                              = std::numeric_limits<float>::quiet_NaN();
    state[0][(nx - 2) * ny * nz + (ny - 2) * nz + nz - 1] = std::numeric_limits<float>::quiet_NaN();
    CHECK_FALSE(kernels->dispatch_milbrandt_warm_rain(dispatch));
    unchanged();
    CHECK(runtime.backend->successful_kernel_dispatch_count(nimbus::compute::KernelId::MilbrandtWarmRain) == calls);
}
} // namespace
TEST_CASE("Metal Milbrandt warm-rain stage matches the CPU reference", "[native-milbrandt-warm][metal]")
{
    exercise("metal");
}
TEST_CASE("Vulkan Milbrandt warm-rain stage matches the CPU reference", "[native-milbrandt-warm][vulkan]")
{
    exercise("vulkan");
}
