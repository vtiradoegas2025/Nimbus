#include "catch2/catch.hpp"
#include "compute/backend_runtime.hpp"
#include "compute/kernel_manifest.hpp"
#include "compute/milbrandt_cloud_contract.hpp"
#include "microphysics/base/thermodynamics.hpp"
#include "microphysics/schemes/milbrandt/activation.hpp"
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
    constexpr int                     nx = 6, ny = 5, nz = 16, cells = nx * ny * nz;
    std::array<std::vector<float>, 7> state, actual, expected;
    std::vector<double>               lower_boundary(nx * ny);
    for (std::size_t n = 0; n < lower_boundary.size(); ++n)
    {
        lower_boundary[n] = std::array<double, 5>{-2.0, 0.0, .00100000001, .0176543210987, 3.0}[n % 5];
    }
    for (auto* fields : {&state, &actual, &expected})
        for (auto& field : *fields)
            field.resize(cells);
    MilbrandtCloudAdjustmentDispatch d;
    d.nx      = nx;
    d.ny      = ny;
    d.nz      = nz;
    d.i_begin = 1;
    d.i_end   = nx - 1;
    d.j_begin = 1;
    d.j_end   = ny - 1;
    d.k_end   = nz;
    for (int f = 0; f < 7; ++f)
    {
        d.state[f]  = state[f].data();
        d.output[f] = actual[f].data();
    }
    ComputeBackendRuntimeState runtime;
    runtime.config.backend        = name;
    runtime.config.allow_fallback = false;
    std::string error;
    REQUIRE(initialize_compute_backend_runtime(runtime, CoordinateSystem::Cartesian, error));
    REQUIRE_FALSE(runtime.fallback_active);
    auto* kernels = runtime.backend->microphysics_kernels();
    REQUIRE(kernels);
    REQUIRE(kernels->supports(MicrophysicsKernelOp::MilbrandtCloudAdjustment));
    std::uint64_t calls     = 0;
    const auto    identical = [&](const auto& reference)
    {
        for (int f = 0; f < 7; ++f)
            CHECK(std::memcmp(actual[f].data(), reference[f].data(), cells * sizeof(float)) == 0);
    };
    for (int velocity_layout : {0, 1, 2})
        for (double dt : {.2, 1.0, 10.0})
        {
            const bool faces                   = velocity_layout != 0;
            d.face_staggered_velocity          = faces;
            d.lower_boundary_vertical_velocity = velocity_layout == 2 ? lower_boundary.data() : nullptr;
            d.dt                               = dt;
            d.k_begin                          = dt == 1.0 ? 1 : 0;
            d.k_end                            = dt == 1.0 ? nz - 1 : nz;
            for (int n = 0; n < cells; ++n)
            {
                const float  T  = std::array<float, 9>{150, 230, 250, 270, 273.15f, 290, 310, 350, 400}[n % 9];
                const float  p  = 60000 + 5000 * (n % 8);
                const double qs = thermodynamics::saturation_mixing_ratio_water(T, p);
                state[0][n]     = T;
                state[1][n]     = p;
                state[2][n]     = .3f + .15f * (n % 7);
                state[3][n]     = float(std::min(.2, qs * std::array<double, 6>{0, .5, 1, 1.000001, 1.1, 2}[n % 6]));
                state[4][n] =
                    std::array<float, 8>{0, 1e-39f, 2.528586384847599e-31f, 1e-30f, 2e-30f, 1e-8f, .001f, .02f}[n % 8];
                state[5][n] = std::array<float, 4>{0, 1e5f, 2e8f, 1e12f}[n % 4];
                state[6][n] = std::array<float, 8>{-1, 0, std::nextafter(.001f, 0.0f), .001f, .01f, .2f, 1, 10}[n % 8];
                if (n == (3 * ny + 3) * nz)
                {
                    state[0][n] = 290;
                    state[1][n] = 90000;
                    state[2][n] = .9f;
                    state[3][n] = float(1.1 * thermodynamics::saturation_mixing_ratio_water(290, 90000));
                    state[4][n] = 0;
                    state[5][n] = 0;
                    state[6][n] = std::nextafter(.001f, 0.0f);
                    // Dropping the boundary's low word changes the 1 mm/s activation branch.
                    lower_boundary[3 * ny + 3] = .00100000007;
                }
            }
            for (int n = 0; n < cells; ++n)
            {
                const int i = n / (ny * nz), j = n / nz % ny, k = n % nz;
                double    T = state[0][n], qv = state[3][n], qc = state[4][n], Nc = state[5][n];
                expected[0][n] = T;
                expected[1][n] = qv;
                expected[2][n] = qc;
                for (int f = 3; f < 7; ++f)
                    expected[f][n] = 0;
                if (i >= d.i_begin && i < d.i_end && j >= d.j_begin && j < d.j_end && k >= d.k_begin && k < d.k_end)
                {
                    T                         = thermodynamics::saturation_adjustment(T, state[1][n], qv, qc);
                    expected[0][n]            = float(T);
                    expected[1][n]            = float(qv);
                    expected[2][n]            = float(qc);
                    expected[3][n]            = float((T - state[0][n]) / dt);
                    expected[4][n]            = float((qv - state[3][n]) / dt);
                    expected[5][n]            = float((qc - state[4][n]) / dt);
                    const double stored_cloud = expected[2][n];
                    if (stored_cloud > state[4][n])
                    {
                        const double lower = k > 0                                ? double(state[6][n - 1])
                                             : d.lower_boundary_vertical_velocity ? lower_boundary[i * ny + j]
                                                                                  : 0.0;
                        const double w     = faces ? .5 * (double(state[6][n]) + lower) : state[6][n];
                        Nc = std::max(Nc, continental_activated_ccn(w, expected[0][n], state[1][n]) / state[2][n]);
                    }
                    else if (stored_cloud < state[4][n])
                        Nc = stored_cloud > 0 ? Nc * stored_cloud / std::max(double(state[4][n]), 1e-30) : 0;
                    expected[6][n] = float((Nc - state[5][n]) / dt);
                }
            }
            REQUIRE(kernels->dispatch_milbrandt_cloud_adjustment(d));
            ++calls;
            for (int f = 0; f < 7; ++f)
                for (int n = 0; n < cells; ++n)
                {
                    CAPTURE(name, velocity_layout, dt, f, n, actual[f][n], expected[f][n]);
                    const double floor = f == 0 ? 3e-5 : f == 3 ? 2e-8 : f == 6 ? .05 : 2e-11;
                    CHECK(std::abs(double(actual[f][n]) - expected[f][n]) <=
                          floor + 2e-5 * std::abs(double(expected[f][n])));
                }
            const auto first = actual;
            REQUIRE(kernels->dispatch_milbrandt_cloud_adjustment(d));
            ++calls;
            identical(first);
        }
    for (int n = 0; n < cells; ++n)
    {
        state[0][n]     = 250 + 5 * (n % 10);
        const double qs = thermodynamics::saturation_mixing_ratio_water(state[0][n], state[1][n]);
        state[3][n]     = float(qs * (n % 2 == 0 ? .5 : 1.1));
    }
    REQUIRE(kernels->dispatch_milbrandt_cloud_adjustment(d));
    ++calls;
    std::array<std::vector<float>, 8> warm_state;
    std::array<std::vector<float>, 6> warm_rates, warm_output;
    for (auto& field : warm_state)
        field.resize(cells);
    for (auto* fields : {&warm_rates, &warm_output})
        for (auto& field : *fields)
            field.resize(cells);
    MilbrandtWarmRainDispatch warm;
    warm.nx      = nx;
    warm.ny      = ny;
    warm.nz      = nz;
    warm.i_begin = d.i_begin;
    warm.i_end   = d.i_end;
    warm.j_begin = d.j_begin;
    warm.j_end   = d.j_end;
    warm.k_end   = nz;
    warm.dt      = d.dt;
    for (int f = 0; f < 8; ++f)
        warm.state[f] = warm_state[f].data();
    for (int f = 0; f < 6; ++f)
    {
        warm.rates[f]  = warm_rates[f].data();
        warm.output[f] = warm_output[f].data();
    }
    for (int n = 0; n < cells; ++n)
    {
        const std::array<float, 8> values{
            actual[0][n], state[1][n], state[2][n], actual[1][n], actual[2][n], .001f, state[5][n], 1e5f};
        for (int f = 0; f < 8; ++f)
            warm_state[f][n] = values[f];
        warm_rates[0][n] = actual[5][n];
        warm_rates[2][n] = actual[4][n];
        warm_rates[3][n] = actual[3][n];
        warm_rates[4][n] = actual[6][n];
    }
    REQUIRE(kernels->dispatch_milbrandt_warm_rain(warm));
    const auto parameters = warm_rain_parameters(0, 523.598775598, 3, 149.1, .5);
    for (int n = 0; n < cells; ++n)
    {
        WarmRainCell       cell{warm_state[0][n],
                                warm_state[1][n],
                                warm_state[2][n],
                                warm_state[3][n],
                                warm_state[4][n],
                                warm_state[5][n],
                                warm_state[6][n],
                                warm_state[7][n]};
        WarmRainTendencies rates{
            warm_rates[0][n], warm_rates[1][n], warm_rates[2][n], warm_rates[3][n], warm_rates[4][n], warm_rates[5][n]};
        const int i = n / (ny * nz), j = n / nz % ny;
        if (i >= d.i_begin && i < d.i_end && j >= d.j_begin && j < d.j_end)
            warm_rain_cell(parameters, cell, d.dt, rates);
        const std::array<float, 6> values{
            rates.cloud, rates.rain, rates.vapor, rates.temperature, rates.cloud_number, rates.rain_number};
        for (int f = 0; f < 6; ++f)
        {
            CAPTURE(name, n, f, warm_output[f][n], values[f]);
            const double floor = f < 3 ? 2e-11 : f == 3 ? 5e-8 : .05;
            CHECK(std::abs(double(warm_output[f][n]) - values[f]) <= floor + 2e-4 * std::abs(double(values[f])));
        }
    }
    const auto accepted     = actual;
    const auto lower_before = lower_boundary;
    lower_boundary.back()   = std::numeric_limits<double>::quiet_NaN();
    CHECK_FALSE(kernels->dispatch_milbrandt_cloud_adjustment(d));
    identical(accepted);
    lower_boundary                     = lower_before;
    d.lower_boundary_vertical_velocity = reinterpret_cast<const double*>(actual[6].data());
    CHECK_FALSE(kernels->dispatch_milbrandt_cloud_adjustment(d));
    identical(accepted);
    d.lower_boundary_vertical_velocity = lower_boundary.data();
    d.face_staggered_velocity          = false;
    CHECK_FALSE(kernels->dispatch_milbrandt_cloud_adjustment(d));
    identical(accepted);
    d.face_staggered_velocity   = true;
    d.dt                        = 1e-36;
    std::size_t validated_cells = 0;
    REQUIRE(nimbus::compute::milbrandt_cloud::validate(d, validated_cells));
    CHECK_FALSE(kernels->dispatch_milbrandt_cloud_adjustment(d));
    ++calls;
    identical(accepted);
    d.dt        = 10;
    d.output[6] = actual[0].data();
    CHECK_FALSE(kernels->dispatch_milbrandt_cloud_adjustment(d));
    identical(accepted);
    d.output[6] = state[0].data();
    CHECK_FALSE(kernels->dispatch_milbrandt_cloud_adjustment(d));
    identical(accepted);
    d.output[6] = actual[6].data();
    d.nx        = std::numeric_limits<int>::max();
    d.ny        = std::numeric_limits<int>::max();
    CHECK_FALSE(kernels->dispatch_milbrandt_cloud_adjustment(d));
    identical(accepted);
    d.nx        = nx;
    d.ny        = ny;
    state[6][0] = std::numeric_limits<float>::quiet_NaN();
    CHECK_FALSE(kernels->dispatch_milbrandt_cloud_adjustment(d));
    identical(accepted);
    CHECK(runtime.backend->successful_kernel_dispatch_count(nimbus::compute::KernelId::MilbrandtCloudAdjustment) ==
          calls);
}
} // namespace
TEST_CASE("Metal Milbrandt cloud adjustment matches the CPU reference", "[native-milbrandt-cloud][metal]")
{
    exercise("metal");
}
TEST_CASE("Vulkan Milbrandt cloud adjustment matches the CPU reference", "[native-milbrandt-cloud][vulkan]")
{
    exercise("vulkan");
}
