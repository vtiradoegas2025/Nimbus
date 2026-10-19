#include "catch2/catch.hpp"
#include "compute/backend_runtime.hpp"
#include "compute/kernel_manifest.hpp"
#include "microphysics/base/thermodynamics.hpp"
#include "microphysics/schemes/thompson/thompson.hpp"
#include "numerics/diffusion/schemes/implicit/implicit.hpp"

#include <array>
#include <cmath>

namespace
{
struct ColdColumn
{
    static constexpr int         n = 8;
    Field3D                      p{n, n, n, 100000}, rho{n, n, n, 1.1f}, theta{n, n, n, 260}, zero{n, n, n, 0};
    std::array<Field3D, 7>       water;
    tmv::ThompsonScheme          scheme;
    tmv::ImplicitDiffusionScheme diffusion;
    GridMetrics                  grid;
    tmv::DiffusionConfig         config;
    ComputeBackendRuntimeState*  runtime = nullptr;

    explicit ColdColumn(ComputeBackendRuntimeState* compute) : runtime(compute)
    {
        const std::array<float, 7> initial{.001f, .001f, .0003f, .00001f, .0002f, .0001f, 0};
        for (std::size_t f = 0; f < water.size(); ++f)
            water[f] = Field3D(n, n, n, initial[f]);
        for (int i = 1; i < n - 1; ++i)
            for (int j = 1; j < n - 1; ++j)
                for (int k = 0; k < n; ++k)
                {
                    theta(i, j, k)    = 245 + 5 * k + .25f * i;
                    water[0](i, j, k) = .9f * thermodynamics::saturation_mixing_ratio_water(theta(i, j, k), 100000);
                }
        grid.dx = grid.dy = 100;
        grid.dz.assign(n, 100);
        config.scheme_id = "implicit";
        config.K_h = config.K_v = 20;
        config.dt_diffusion     = .2;
        config.apply_to         = "scalars_and_momentum";
        diffusion.initialize(config);
        scheme.set_physical_cell_domain(n, n, n, {1, n - 1, 1, n - 1, 0, n});
        scheme.set_vertical_spacing_m(100);
        scheme.set_vertical_velocity_field(zero, false);
        scheme.prepare_prognostic_state(rho, water[1], water[2], water[3], water[4], water[5], water[6]);
        if (runtime)
        {
            scheme.bind_execution_owners(*runtime, nullptr);
            scheme.set_require_native_dispatch(true);
        }
    }

    void advance()
    {
        std::array<Field3D, 8> rates;
        scheme.compute_tendencies(p,
                                  theta,
                                  rho,
                                  water[0],
                                  water[1],
                                  water[2],
                                  water[3],
                                  water[4],
                                  water[5],
                                  water[6],
                                  .2,
                                  rates[0],
                                  rates[1],
                                  rates[2],
                                  rates[3],
                                  rates[4],
                                  rates[5],
                                  rates[6],
                                  rates[7]);
        std::string error;
        REQUIRE(scheme.validate_prognostic_update(.2, error));
        scheme.commit_prognostic_update(.2);
        for (int i = 1; i < n - 1; ++i)
            for (int j = 1; j < n - 1; ++j)
                for (int k = 0; k < n; ++k)
                {
                    theta(i, j, k) += .2f * rates[0](i, j, k);
                    for (std::size_t f = 0; f < water.size(); ++f)
                    {
                        const double value = water[f](i, j, k) + .2 * rates[f + 1](i, j, k);
                        REQUIRE(value >= -2e-10);
                        water[f](i, j, k) = static_cast<float>(std::max(0.0, value));
                    }
                }
        tmv::DiffusionStateView state;
        state.theta = &theta;
        state.qv    = &water[0];
        state.u = state.v = state.w = &zero;
        state.rho                   = &rho;
        state.grid                  = &grid;
        state.c_grid                = true;
        state.periodic_x = state.periodic_y = true;
        state.compute_runtime               = runtime;
        tmv::DiffusionTendencies result;
        diffusion.compute_diffusion_tendencies(config, state, result, nullptr);
        for (int i = 1; i < n - 1; ++i)
            for (int j = 1; j < n - 1; ++j)
                for (int k = 0; k < n; ++k)
                {
                    theta(i, j, k) += .2f * result.dthetadt_diff(i, j, k);
                    water[0](i, j, k) += .2f * result.dqvdt_diff(i, j, k);
                    REQUIRE(water[0](i, j, k) >= 0);
                }
    }
};

void exercise(const char* backend)
{
    ComputeBackendRuntimeState runtime;
    runtime.config.backend        = backend;
    runtime.config.allow_fallback = false;
    std::string error;
    REQUIRE(initialize_compute_backend_runtime(runtime, CoordinateSystem::Cartesian, error));
    ColdColumn cpu(nullptr), native(&runtime);
    const auto initial = cpu.water;
    for (int step = 0; step < 3; ++step)
    {
        INFO(step);
        REQUIRE_NOTHROW(cpu.advance());
        REQUIRE_NOTHROW(native.advance());
        for (int i = 1; i < ColdColumn::n - 1; ++i)
            for (int j = 1; j < ColdColumn::n - 1; ++j)
                for (int k = 0; k < ColdColumn::n; ++k)
                {
                    CHECK(std::abs(cpu.theta(i, j, k) - native.theta(i, j, k)) < 3e-4);
                    for (std::size_t f = 0; f < cpu.water.size(); ++f)
                    {
                        INFO(f);
                        CHECK(std::abs(cpu.water[f](i, j, k) - native.water[f](i, j, k)) <=
                              2e-9 + 1e-4 * std::abs(cpu.water[f](i, j, k)));
                    }
                }
        const auto reference = cpu.scheme.portable_state(), actual = native.scheme.portable_state();
        for (std::size_t f = 0; f < 5; ++f)
            for (std::size_t n = 0; n < reference.fields[f].size(); ++n)
                CHECK(std::abs(reference.fields[f].data()[n] - actual.fields[f].data()[n]) <=
                      .01 + 1e-5 * std::abs(reference.fields[f].data()[n]));
    }
    for (std::size_t f : {std::size_t(2), std::size_t(3), std::size_t(4), std::size_t(5)})
    {
        float change = 0;
        for (std::size_t n = 0; n < initial[f].size(); ++n)
            change = std::max(change, std::abs(initial[f].data()[n] - native.water[f].data()[n]));
        CHECK(change > 1e-8);
    }
    CHECK(runtime.backend->successful_kernel_dispatch_count(nimbus::compute::KernelId::ThompsonCompletion) == 3);
    CHECK(runtime.backend->successful_kernel_dispatch_count(nimbus::compute::KernelId::ImplicitDiffusionStencil) > 0);
}
} // namespace

TEST_CASE("Metal Thompson cold processes couple to implicit diffusion", "[thompson-implicit][metal]")
{
    exercise("metal");
}
TEST_CASE("Vulkan Thompson cold processes couple to implicit diffusion", "[thompson-implicit][vulkan]")
{
    exercise("vulkan");
}
