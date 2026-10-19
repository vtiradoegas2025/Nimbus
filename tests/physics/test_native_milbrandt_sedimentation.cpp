#include "catch2/catch.hpp"
#include "compute/backend_runtime.hpp"
#include "compute/kernel_manifest.hpp"
#include "compute/milbrandt_sedimentation_contract.hpp"
#include "microphysics/schemes/milbrandt/sedimentation.hpp"
#include <array>
#include <cstring>
#include <vector>

namespace
{
void exercise(const char* name)
{
    constexpr int                     nx = 6, ny = 5, nz = 16, cells = nx * ny * nz;
    std::array<std::vector<float>, 7> state;
    std::array<std::vector<float>, 3> actual, expected;
    for (auto& f : state)
        f.resize(cells);
    for (auto* fields : {&actual, &expected})
        for (auto& f : *fields)
            f.resize(cells);
    MilbrandtSedimentationDispatch d;
    d.nx      = nx;
    d.ny      = ny;
    d.nz      = nz;
    d.i_begin = 1;
    d.i_end   = nx - 1;
    d.j_begin = 1;
    d.j_end   = ny - 1;
    d.k_end   = nz;
    for (int f = 0; f < 7; ++f)
        d.state[f] = state[f].data();
    for (int f = 0; f < 3; ++f)
        d.output[f] = actual[f].data();
    ComputeBackendRuntimeState runtime;
    runtime.config.backend        = name;
    runtime.config.allow_fallback = false;
    std::string error;
    const bool  initialized = initialize_compute_backend_runtime(runtime, CoordinateSystem::Cartesian, error);
    INFO(error);
    REQUIRE(initialized);
    REQUIRE_FALSE(runtime.fallback_active);
    auto* kernels = runtime.backend->microphysics_kernels();
    REQUIRE(kernels);
    REQUIRE(kernels->supports(MicrophysicsKernelOp::MilbrandtSedimentation));
    std::uint64_t calls     = 0;
    const auto    identical = [&](const auto& reference)
    {
        for (int f = 0; f < 3; ++f)
            CHECK(std::memcmp(actual[f].data(), reference[f].data(), cells * sizeof(float)) == 0);
    };
    tmv::MilbrandtSedimentationWorkspace workspace;
    workspace.resize(nz);
    for (int species = 0; species < 5; ++species)
        for (int mode = 0; mode < 3; ++mode)
        {
            d.species = species;
            d.dt      = mode == 0 ? .2 : mode == 1 ? 10 : 1;
            d.k_begin = mode == 1 ? 1 : 0;
            d.k_end   = mode == 1 ? nz - 1 : nz;
            for (int n = 0; n < cells; ++n)
            {
                state[0][n]     = std::array<float, 5>{0, 1e-15f, 1e-8f, .0001f, .002f}[n % 5];
                state[1][n]     = std::array<float, 5>{0, 1, 100, 1e4f, 1e6f}[(n / 5) % 5];
                state[2][n]     = .3f + .1f * (n % nz);
                const double dz = mode == 2 ? ((n % nz) % 3 == 0 ? 1e-4 : 1.75) : 250.0 / 3.0 + 2 * (n % nz);
                state[3][n]     = float(dz);
                state[4][n]     = float(dz - state[3][n]);
                state[5][n]     = n % 3 == 0 ? float(.1 * state[0][n] / d.dt) : 0;
                state[6][n]     = float(-.1 * state[1][n] / d.dt);
                expected[0][n]  = state[5][n];
                expected[1][n]  = state[6][n];
                expected[2][n]  = 0;
                for (int f = 0; f < 3; ++f)
                    actual[f][n] = 17;
            }
            for (int i = d.i_begin; i < d.i_end; ++i)
                for (int j = d.j_begin; j < d.j_end; ++j)
                {
                    const int base = (i * ny + j) * nz, first = base + d.k_begin;
                    for (int k = d.k_begin; k < d.k_end; ++k)
                        workspace.thickness[k - d.k_begin] = double(state[3][base + k]) + state[4][base + k];
                    double loss = 0;
                    REQUIRE(tmv::milbrandt_detail::sediment_column(
                        tmv::milbrandt_detail::default_sedimentation_parameters[species],
                        d.k_end - d.k_begin,
                        state[0].data() + first,
                        state[1].data() + first,
                        state[2].data() + first,
                        workspace.thickness.data(),
                        d.dt,
                        workspace,
                        expected[0].data() + first,
                        expected[1].data() + first,
                        loss));
                    expected[2][base] = float(loss / d.dt);
                }
            REQUIRE(kernels->dispatch_milbrandt_sedimentation(d));
            ++calls;
            for (int f = 0; f < 3; ++f)
                for (int n = 0; n < cells; ++n)
                {
                    CAPTURE(name, species, mode, f, n, actual[f][n], expected[f][n]);
                    const double floor = f == 1 ? .001 : 2e-11;
                    CHECK(std::abs(double(actual[f][n]) - expected[f][n]) <=
                          floor + 2e-5 * std::abs(double(expected[f][n])));
                }
            const auto first = actual;
            REQUIRE(kernels->dispatch_milbrandt_sedimentation(d));
            ++calls;
            identical(first);
        }
    const auto accepted = actual;
    for (int n = 0; n < cells; ++n)
    {
        state[3][n] = 1;
        state[4][n] = 0;
    }
    d.dt                = 10;
    d.maximum_substeps  = 1;
    std::size_t checked = 0;
    REQUIRE(nimbus::compute::milbrandt_sedimentation::validate(d, checked));
    CHECK_FALSE(kernels->dispatch_milbrandt_sedimentation(d));
    ++calls;
    identical(accepted);
    d.maximum_substeps = 100000;
    d.species          = 5;
    CHECK_FALSE(kernels->dispatch_milbrandt_sedimentation(d));
    identical(accepted);
    d.species   = 4;
    d.output[2] = actual[0].data();
    CHECK_FALSE(kernels->dispatch_milbrandt_sedimentation(d));
    identical(accepted);
    d.output[2] = state[0].data();
    CHECK_FALSE(kernels->dispatch_milbrandt_sedimentation(d));
    identical(accepted);
    d.output[2] = actual[2].data();
    d.nx        = std::numeric_limits<int>::max();
    d.ny        = std::numeric_limits<int>::max();
    CHECK_FALSE(kernels->dispatch_milbrandt_sedimentation(d));
    identical(accepted);
    d.nx                        = nx;
    d.ny                        = ny;
    state[3][(2 * ny + 2) * nz] = -1;
    CHECK_FALSE(kernels->dispatch_milbrandt_sedimentation(d));
    identical(accepted);
    state[3][(2 * ny + 2) * nz] = 1;
    state[0][0]                 = std::numeric_limits<float>::quiet_NaN();
    CHECK_FALSE(kernels->dispatch_milbrandt_sedimentation(d));
    identical(accepted);
    CHECK(runtime.backend->successful_kernel_dispatch_count(nimbus::compute::KernelId::MilbrandtSedimentation) ==
          calls);
}
} // namespace
TEST_CASE("Metal Milbrandt sedimentation matches the CPU reference", "[native-milbrandt-sedimentation][metal]")
{
    exercise("metal");
}
TEST_CASE("Vulkan Milbrandt sedimentation matches the CPU reference", "[native-milbrandt-sedimentation][vulkan]")
{
    exercise("vulkan");
}
