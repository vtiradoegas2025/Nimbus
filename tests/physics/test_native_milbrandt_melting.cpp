#include "catch2/catch.hpp"
#include "compute/backend_runtime.hpp"
#include "compute/kernel_manifest.hpp"
#include "compute/milbrandt_melting_contract.hpp"
#include "microphysics/schemes/milbrandt/melting.hpp"
#include <array>
#include <cmath>
#include <cstring>
#include <vector>

namespace
{
void exercise(const char* name)
{
    constexpr int                      nx = 6, ny = 5, nz = 16, cells = nx * ny * nz;
    std::array<std::vector<float>, 12> state;
    std::array<std::vector<float>, 11> actual, expected;
    for (auto& f : state)
        f.resize(cells);
    for (auto* fields : {&actual, &expected})
        for (auto& f : *fields)
            f.resize(cells);
    MilbrandtMeltingDispatch d;
    d.nx      = nx;
    d.ny      = ny;
    d.nz      = nz;
    d.i_begin = 1;
    d.i_end   = nx - 1;
    d.j_begin = 1;
    d.j_end   = ny - 1;
    d.k_end   = nz;
    for (int f = 0; f < 12; ++f)
        d.state[f] = state[f].data();
    for (int f = 0; f < 11; ++f)
        d.output[f] = actual[f].data();
    ComputeBackendRuntimeState runtime;
    runtime.config.backend        = name;
    runtime.config.allow_fallback = false;
    std::string error;
    REQUIRE(initialize_compute_backend_runtime(runtime, CoordinateSystem::Cartesian, error));
    REQUIRE_FALSE(runtime.fallback_active);
    auto* kernels = runtime.backend->microphysics_kernels();
    REQUIRE(kernels);
    REQUIRE(kernels->supports(MicrophysicsKernelOp::MilbrandtMelting));
    std::uint64_t calls     = 0;
    const auto    identical = [&](const auto& reference)
    {
        for (int f = 0; f < 11; ++f)
            CHECK(std::memcmp(actual[f].data(), reference[f].data(), cells * sizeof(float)) == 0);
    };
    for (double dt : {.2, 1.0, 1000.0})
    {
        d.dt      = dt;
        d.k_begin = dt == 1 ? 1 : 0;
        d.k_end   = dt == 1 ? nz - 1 : nz;
        for (int n = 0; n < cells; ++n)
        {
            state[0][n] = std::array<float, 12>{150,
                                                250,
                                                273.15f,
                                                std::nextafter(273.15f, 300.0f),
                                                275,
                                                278.15f,
                                                std::nextafter(278.15f, 300.0f),
                                                280,
                                                290,
                                                300,
                                                320,
                                                350}[n % 12];
            state[1][n] = std::array<float, 5>{1000, 20000, 60000, 95000, 110000}[n % 5];
            state[2][n] = std::array<float, 5>{.05f, .3f, .7f, 1.2f, 2}[n % 5];
            state[3][n] = std::array<float, 5>{0, .001f, .005f, .02f, .1f}[(n / 5) % 5];
            for (int f = 0; f < 4; ++f)
            {
                state[4 + f][n] = std::array<float, 8>{0, 1e-30f, 1e-20f, 1e-14f, 1e-8f, .001f, .01f, .1f}[(n + f) % 8];
                state[8 + f][n] = std::array<float, 8>{0, 1e-30f, 1e-12f, 1e-6f, 1, 1e3f, 1e7f, 1e12f}[(n / 6 + f) % 8];
            }
            std::array<double, 12> cell;
            for (int f = 0; f < 12; ++f)
                cell[f] = state[f][n];
            std::array<float, 11> rates{};
            const int             i = n / (ny * nz), j = n / nz % ny, k = n % nz;
            if (i >= d.i_begin && i < d.i_end && j >= d.j_begin && j < d.j_end && k >= d.k_begin && k < d.k_end)
                tmv::milbrandt_detail::melting_cell(d.parameters, cell, dt, rates);
            for (int f = 0; f < 11; ++f)
            {
                expected[f][n] = rates[f];
                actual[f][n]   = 17;
            }
        }
        REQUIRE(kernels->dispatch_milbrandt_melting(d));
        ++calls;
        for (int f = 0; f < 11; ++f)
            for (int n = 0; n < cells; ++n)
            {
                CAPTURE(name, dt, f, n, actual[f][n], expected[f][n]);
                const double floor = f < 5 ? 2e-12 : f == 5 ? 2e-9 : 1e-4;
                CHECK(std::abs(double(actual[f][n]) - expected[f][n]) <=
                      floor + 2e-5 * std::abs(double(expected[f][n])));
            }
        const auto first = actual;
        REQUIRE(kernels->dispatch_milbrandt_melting(d));
        ++calls;
        identical(first);
    }
    const auto accepted = actual;
    d.dt                = 1e-36;
    std::size_t checked = 0;
    REQUIRE(nimbus::compute::milbrandt_melting::validate(d, checked));
    CHECK_FALSE(kernels->dispatch_milbrandt_melting(d));
    ++calls;
    identical(accepted);
    d.dt = 1;
    for (double invalid :
         {0.0, -1.0, std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN()})
    {
        d.dt = invalid;
        CHECK_FALSE(kernels->dispatch_milbrandt_melting(d));
        identical(accepted);
    }
    d.dt            = 1;
    d.parameters[0] = 1;
    CHECK_FALSE(kernels->dispatch_milbrandt_melting(d));
    identical(accepted);
    d.parameters = tmv::milbrandt_detail::default_melting_parameters;
    d.output[10] = actual[0].data();
    CHECK_FALSE(kernels->dispatch_milbrandt_melting(d));
    identical(accepted);
    d.output[10] = state[0].data();
    CHECK_FALSE(kernels->dispatch_milbrandt_melting(d));
    identical(accepted);
    d.output[10] = nullptr;
    CHECK_FALSE(kernels->dispatch_milbrandt_melting(d));
    identical(accepted);
    d.output[10] = actual[10].data();
    d.nx         = std::numeric_limits<int>::max();
    d.ny         = std::numeric_limits<int>::max();
    CHECK_FALSE(kernels->dispatch_milbrandt_melting(d));
    identical(accepted);
    d.nx    = nx;
    d.ny    = ny;
    d.k_end = nz + 1;
    CHECK_FALSE(kernels->dispatch_milbrandt_melting(d));
    identical(accepted);
    d.k_end          = nz;
    const int active = (2 * ny + 2) * nz;
    for (int f = 0; f < 12; ++f)
    {
        const float original = state[f][active];
        state[f][active]     = -1;
        CHECK_FALSE(kernels->dispatch_milbrandt_melting(d));
        identical(accepted);
        state[f][active] = original;
    }
    state[0][0] = std::numeric_limits<float>::quiet_NaN();
    CHECK_FALSE(kernels->dispatch_milbrandt_melting(d));
    identical(accepted);
    CHECK(runtime.backend->successful_kernel_dispatch_count(nimbus::compute::KernelId::MilbrandtMelting) == calls);
}
} // namespace
TEST_CASE("Metal Milbrandt melting matches the CPU reference", "[native-milbrandt-melting][metal]")
{
    exercise("metal");
}
TEST_CASE("Vulkan Milbrandt melting matches the CPU reference", "[native-milbrandt-melting][vulkan]")
{
    exercise("vulkan");
}
