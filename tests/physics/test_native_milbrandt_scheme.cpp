#include "catch2/catch.hpp"
#include "compute/backend_runtime.hpp"
#include "compute/kernel_manifest.hpp"
#include "microphysics/base/thermodynamics.hpp"
#include "microphysics/schemes/milbrandt/milbrandt.hpp"
#include <array>
#include <cstring>

namespace
{
void exercise(const char* name)
{
    constexpr int           nx = 6, ny = 5, nz = 12;
    std::array<Field3D, 10> state;
    std::array<Field3D, 8>  cpu_rates, native_rates;
    for (auto& f : state)
        f.resize(nx, ny, nz, 0);
    Field3D wind(nx, ny, nz, 1);
    state[0].fill(270);
    state[1].fill(90000);
    state[2].fill(1);
    for (int i = 0; i < nx; ++i)
        for (int j = 0; j < ny; ++j)
            for (int k = 0; k < nz; ++k)
            {
                const double T    = 245 + 4 * k;
                state[0](i, j, k) = float(T / std::pow(.9, microphysics_constants::R_d / microphysics_constants::cp));
                state[3](i, j, k) =
                    float(thermodynamics::saturation_mixing_ratio_water(T, 90000) * (k % 2 == 0 ? 1.05 : .8));
                for (int f = 4; f < 10; ++f)
                    state[f](i, j, k) = f == 4 ? .001f : .0001f;
            }
    ComputeBackendRuntimeState runtime;
    runtime.config.backend        = name;
    runtime.config.allow_fallback = false;
    std::string error;
    const bool  initialized = initialize_compute_backend_runtime(runtime, CoordinateSystem::Cartesian, error);
    INFO(error);
    REQUIRE(initialized);
    tmv::MilbrandtScheme cpu, native;
    const auto           configure = [&](tmv::MilbrandtScheme& scheme, bool device)
    {
        scheme.set_physical_cell_domain(nx, ny, nz, {1, nx - 1, 1, ny - 1, 0, nz});
        scheme.set_vertical_spacing_m(100);
        scheme.set_vertical_velocity_field(wind, true);
        scheme.prepare_prognostic_state(state[2], state[4], state[5], state[6], state[7], state[8], state[9]);
        if (device)
        {
            scheme.bind_execution_owners(runtime, nullptr);
            scheme.set_require_native_dispatch(true);
        }
    };
    configure(cpu, false);
    configure(native, true);
    const auto compute = [&](tmv::MilbrandtScheme& scheme, std::array<Field3D, 8>& rates, double dt)
    {
        scheme.compute_tendencies(state[1],
                                  state[0],
                                  state[2],
                                  state[3],
                                  state[4],
                                  state[5],
                                  state[6],
                                  state[7],
                                  state[8],
                                  state[9],
                                  dt,
                                  rates[0],
                                  rates[1],
                                  rates[2],
                                  rates[3],
                                  rates[4],
                                  rates[5],
                                  rates[6],
                                  rates[7]);
    };
    for (int step = 0; step < 3; ++step)
    {
        constexpr double dt = .2;
        compute(cpu, cpu_rates, dt);
        compute(native, native_rates, dt);
        for (int f = 0; f < 8; ++f)
            for (std::size_t n = 0; n < state[0].size(); ++n)
            {
                CAPTURE(name, step, f, n, cpu_rates[f].data()[n], native_rates[f].data()[n]);
                const double floor = f == 0 ? 5e-6 : 5e-10;
                CHECK(std::abs(double(native_rates[f].data()[n]) - cpu_rates[f].data()[n]) <=
                      floor + 5e-4 * std::abs(double(cpu_rates[f].data()[n])));
            }
        REQUIRE(cpu.validate_prognostic_update(dt, error));
        REQUIRE(native.validate_prognostic_update(dt, error));
        cpu.commit_prognostic_update(dt);
        native.commit_prognostic_update(dt);
        const auto a = cpu.prognostic_scalars_for_transport(), b = native.prognostic_scalars_for_transport();
        for (std::size_t f = 0; f < a.size(); ++f)
            for (std::size_t n = 0; n < state[0].size(); ++n)
            {
                CAPTURE(name, step, f, n);
                CHECK(std::abs(double(a[f].field->data()[n]) - b[f].field->data()[n]) <=
                      .05 + 5e-4 * std::abs(double(a[f].field->data()[n])));
            }
        for (int i = 1; i < nx - 1; ++i)
            for (int j = 1; j < ny - 1; ++j)
            {
                const double reference =
                    cpu.bottom_sedimentation_sink_rate(state[2], state[5], state[7], state[8], state[9], i, j, dt);
                const double actual =
                    native.bottom_sedimentation_sink_rate(state[2], state[5], state[7], state[8], state[9], i, j, dt);
                CHECK(std::abs(actual - reference) <= 2e-10 + 5e-4 * std::abs(reference));
            }
    }
    tmv::MilbrandtScheme resumed;
    configure(resumed, true);
    REQUIRE(resumed.restore_portable_state(native.portable_state(), error));
    std::array<Field3D, 8> resumed_rates;
    compute(resumed, resumed_rates, .2);
    compute(native, native_rates, .2);
    for (int f = 0; f < 8; ++f)
        CHECK(std::memcmp(resumed_rates[f].data(), native_rates[f].data(), state[0].size() * sizeof(float)) == 0);
    const auto optional_calls =
        runtime.backend->successful_kernel_dispatch_count(nimbus::compute::KernelId::MilbrandtWarmRain);
    native.set_require_native_dispatch(false);
    runtime.config.allow_fallback = true;
    compute(native, native_rates, .2);
    CHECK(runtime.backend->successful_kernel_dispatch_count(nimbus::compute::KernelId::MilbrandtWarmRain) ==
          optional_calls + 1);
    runtime.config.allow_fallback = false;
    native.set_require_native_dispatch(true);
    const auto accepted    = native_rates;
    const auto wind_before = wind;
    REQUIRE_THROWS_AS(native.compute_tendencies(state[1],
                                                state[0],
                                                state[2],
                                                state[3],
                                                state[4],
                                                state[5],
                                                state[6],
                                                state[7],
                                                state[8],
                                                state[9],
                                                .2,
                                                wind,
                                                native_rates[1],
                                                native_rates[2],
                                                native_rates[3],
                                                native_rates[4],
                                                native_rates[5],
                                                native_rates[6],
                                                native_rates[7]),
                      std::invalid_argument);
    CHECK(std::memcmp(wind.data(), wind_before.data(), wind.size() * sizeof(float)) == 0);
    const auto moments = native.portable_state();
    const auto calls  = runtime.backend->successful_kernel_dispatch_count(nimbus::compute::KernelId::MilbrandtWarmRain);
    state[9](2, 2, 4) = .2f;
    REQUIRE_THROWS_WITH(compute(native, native_rates, .2),
                        Catch::Matchers::Contains("native Milbrandt cold processes failed"));
    for (int f = 0; f < 8; ++f)
        CHECK(std::memcmp(accepted[f].data(), native_rates[f].data(), state[0].size() * sizeof(float)) == 0);
    CHECK(runtime.backend->successful_kernel_dispatch_count(nimbus::compute::KernelId::MilbrandtWarmRain) == calls + 1);
    const auto after = native.portable_state();
    REQUIRE(after.fields.size() == moments.fields.size());
    for (std::size_t f = 0; f < after.fields.size(); ++f)
        CHECK(std::memcmp(after.fields[f].data(), moments.fields[f].data(), after.fields[f].size() * sizeof(float)) ==
              0);
}
} // namespace
TEST_CASE("Metal complete Milbrandt process chain matches CPU and resumes atomically",
          "[native-milbrandt-scheme][metal]")
{
    exercise("metal");
}
TEST_CASE("Vulkan complete Milbrandt process chain matches CPU and resumes atomically",
          "[native-milbrandt-scheme][vulkan]")
{
    exercise("vulkan");
}
