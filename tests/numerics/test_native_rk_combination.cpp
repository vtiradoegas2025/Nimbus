#include "catch2/catch.hpp"
#include "compute/backend_runtime.hpp"
#include "compute/rk_combination_contract.hpp"
#include "numerics/time_stepping/rk_combination.hpp"
#include "numerics/time_stepping/factory.hpp"
#include "numerics/time_stepping/time_stepping_base.hpp"
#include <array>
#include <cmath>
#include <cstring>
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
    REQUIRE_FALSE(runtime.fallback_active);
    REQUIRE(runtime.backend->supports_rk_combination());
    constexpr int                     cells = 257;
    std::array<std::vector<float>, 9> input;
    std::array<std::vector<float>, 4> out;
    for (auto& f : input)
        f.resize(cells);
    for (auto& f : out)
        f.resize(cells, 17);
    RkCombinationDispatch d;
    d.base              = input[0].data();
    d.current           = input[1].data();
    d.base_density      = input[2].data();
    d.current_density   = input[3].data();
    d.candidate_density = input[4].data();
    for (int f = 0; f < 4; ++f)
    {
        d.rhs[f]    = input[5 + f].data();
        d.output[f] = out[f].data();
    }
    d.cells                 = cells;
    std::uint64_t calls     = 0;
    const auto    identical = [&](const auto& reference)
    {
        for (int f = 0; f < 4; ++f)
            CHECK(std::memcmp(out[f].data(), reference[f].data(), cells * sizeof(float)) == 0);
    };
    for (bool conserved : {false, true})
        for (bool ssp : {false, true})
            for (int stages = 1; stages <= (ssp ? 3 : 4); ++stages)
                for (double dt : {.2, 1.0, 10.0})
                {
                    d.conserved_mixing_ratio = conserved;
                    d.ssp                    = ssp;
                    d.stages                 = stages;
                    d.dt                     = dt;
                    d.weights                = {1.0 / 6, 1.0 / 3, 1.0 / 3, 1.0 / 6};
                    for (int n = 0; n < cells; ++n)
                    {
                        input[0][n] = std::array<float, 6>{0, .0001f, -.01f, 300, 123456784, 1e-20f}[n % 6];
                        input[1][n] = n % 3 == 0 ? 0 : input[0][n] * .75f;
                        input[2][n] = .3f + .125f * (n % 7);
                        input[3][n] = .7f + .0625f * (n % 5);
                        input[4][n] = std::nextafter(.8f + .125f * (n % 3), 2.0f);
                        for (int stage = 0; stage < 4; ++stage)
                            input[5 + stage][n] = float((stage % 2 == 0 ? -1 : 1) * (double(input[0][n]) + .0125) / dt);
                    }
                    REQUIRE(runtime.backend->dispatch_rk_combination(d) == RkCombinationStatus::Completed);
                    ++calls;
                    for (int n = 0; n < cells; ++n)
                    {
                        double base = input[0][n], current = input[1][n];
                        if (conserved)
                        {
                            base *= input[2][n];
                            current *= input[3][n];
                        }
                        double scale    = 0;
                        double expected = tmv::rk::combine(
                            base,
                            current,
                            [&](std::size_t s) { return double(input[5 + s][n]); },
                            stages,
                            d.weights,
                            dt,
                            ssp,
                            scale);
                        if (conserved)
                        {
                            expected /= input[4][n];
                            scale /= input[4][n];
                        }
                        const double actual       = double(out[0][n]) + out[1][n],
                                     actual_scale = double(out[2][n]) + out[3][n];
                        CAPTURE(backend, conserved, ssp, stages, dt, n, expected, actual, scale, actual_scale);
                        CHECK(std::abs(actual - expected) <= 2e-13 * std::max({1.0, std::abs(expected), scale}));
                        CHECK(std::abs(actual_scale - scale) <= 2e-13 * std::max(1.0, scale));
                    }
                    const auto first = out;
                    REQUIRE(runtime.backend->dispatch_rk_combination(d) == RkCombinationStatus::Completed);
                    ++calls;
                    identical(first);
                }
    const auto accepted      = out;
    d.ssp                    = false;
    d.conserved_mixing_ratio = false;
    d.stages                 = 1;
    d.dt                     = 10;
    d.weights                = {1, 0, 0, 0};
    input[0].back()          = std::numeric_limits<float>::max();
    input[5].back()          = std::numeric_limits<float>::max();
    REQUIRE(nimbus::compute::rk_combination::validate(d));
    CHECK(runtime.backend->dispatch_rk_combination(d) == RkCombinationStatus::NonFinite);
    ++calls;
    identical(accepted);
    input[5].back() = std::numeric_limits<float>::quiet_NaN();
    CHECK(runtime.backend->dispatch_rk_combination(d) == RkCombinationStatus::InvalidInput);
    identical(accepted);
    input[0].back() = 0;
    input[5].back() = 0;
    d.output[3]     = out[0].data();
    CHECK(runtime.backend->dispatch_rk_combination(d) == RkCombinationStatus::InvalidInput);
    identical(accepted);
    d.output[3] = input[0].data();
    CHECK(runtime.backend->dispatch_rk_combination(d) == RkCombinationStatus::InvalidInput);
    identical(accepted);
    d.output[3] = out[3].data();
    d.cells     = std::numeric_limits<std::size_t>::max();
    CHECK(runtime.backend->dispatch_rk_combination(d) == RkCombinationStatus::InvalidInput);
    identical(accepted);
    d.cells                  = cells;
    d.conserved_mixing_ratio = true;
    input[4].back()          = 0;
    CHECK(runtime.backend->dispatch_rk_combination(d) == RkCombinationStatus::InvalidInput);
    identical(accepted);
    input[4].back() = 1;
    d.stages        = 4;
    d.ssp           = true;
    CHECK(runtime.backend->dispatch_rk_combination(d) == RkCombinationStatus::InvalidInput);
    identical(accepted);
    d.stages                 = 1;
    d.ssp                    = false;
    d.conserved_mixing_ratio = false;
    d.dt                     = 1e-20;
    d.weights                = {1e-30, 0, 0, 0};
    CHECK(runtime.backend->dispatch_rk_combination(d) == RkCombinationStatus::InvalidInput);
    identical(accepted);
    d.ssp     = true;
    d.stages  = 2;
    d.weights = {1e300, 1e300, 0, 0};
    REQUIRE(runtime.backend->dispatch_rk_combination(d) == RkCombinationStatus::Completed);
    ++calls;
    d.weights                = {1, 0, 0, 0};
    d.stages                 = 1;
    d.ssp                    = false;
    d.conserved_mixing_ratio = false;
    d.dt                     = .2;
    for (int s = 1; s < 4; ++s)
    {
        d.rhs[s]     = nullptr;
        d.weights[s] = std::numeric_limits<double>::quiet_NaN();
    }
    REQUIRE(runtime.backend->dispatch_rk_combination(d) == RkCombinationStatus::Completed);
    ++calls;

    for (const char* name : {"rk3", "rk4"})
    {
        std::array<double, 3> errors{};
        for (int resolution = 0; resolution < 3; ++resolution)
        {
            const int             steps = 2 << resolution;
            const double          dt    = 1.0 / steps;
            float                 base = 1, value = 1;
            std::array<float, 4>  rhs{}, result{};
            RkCombinationDispatch step_dispatch;
            step_dispatch.base    = &base;
            step_dispatch.current = &value;
            step_dispatch.cells   = 1;
            step_dispatch.ssp     = std::string(name) == "rk3";
            for (int f = 0; f < 4; ++f)
            {
                step_dispatch.rhs[f]    = &rhs[f];
                step_dispatch.output[f] = &result[f];
            }
            TimeSteppingConfig config;
            config.scheme_id          = name;
            auto               scheme = create_time_stepping_scheme(config.scheme_id);
            UnsplitRkCallbacks callbacks;
            callbacks.capture_base       = [&] { base = value; };
            callbacks.compute_tendencies = [&](std::size_t s, double t) { rhs[s] = float(t * value); };
            callbacks.publish_combination =
                [&](std::size_t stages, const std::array<double, 4>& weights, double dt_stage)
            {
                step_dispatch.stages  = int(stages);
                step_dispatch.weights = weights;
                step_dispatch.dt      = dt_stage;
                REQUIRE(runtime.backend->dispatch_rk_combination(step_dispatch) == RkCombinationStatus::Completed);
                ++calls;
                value = float(double(result[0]) + result[1]);
            };
            callbacks.restore_base = [&] { value = base; };
            for (int step = 0; step < steps; ++step)
                scheme->step_coupled_unsplit_rk(config, dt, step * dt, callbacks, nullptr);
            errors[resolution] = std::abs(double(value) - std::exp(.5));
        }
        CAPTURE(name, errors[0], errors[1], errors[2]);
        const double minimum_ratio = std::string(name) == "rk3" ? 5.0 : 8.0;
        CHECK(errors[0] / errors[1] > minimum_ratio);
        CHECK(errors[1] / errors[2] > minimum_ratio);
    }
    CHECK(runtime.backend->successful_kernel_dispatch_count(nimbus::compute::KernelId::RkCombination) == calls);
}
} // namespace
TEST_CASE("Metal RK combinations preserve stored stages and conservative reconstruction", "[native-rk][metal]")
{
    exercise("metal");
}
TEST_CASE("Vulkan RK combinations preserve stored stages and conservative reconstruction", "[native-rk][vulkan]")
{
    exercise("vulkan");
}
