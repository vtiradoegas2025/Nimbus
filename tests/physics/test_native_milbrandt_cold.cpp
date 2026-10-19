#include "catch2/catch.hpp"
#include "compute/backend_runtime.hpp"
#include "compute/kernel_manifest.hpp"
#include "compute/milbrandt_cold_contract.hpp"
#include "microphysics/schemes/milbrandt/cold.hpp"
#include "support/milbrandt_double_riming_reference.hpp"
#include <array>
#include <cmath>
#include <cstring>
#include <vector>

namespace
{
template <class Kernels>
std::uint64_t independent_collection(Kernels* kernels, const char* name, double duration)
{
    namespace reference = milbrandt_double_riming_reference;
    constexpr int cells = 12;
    std::array<std::array<float, cells>, 16> state{};
    std::array<std::array<float, cells>, 14> incoming{}, output{};
    MilbrandtColdDispatch dispatch;
    dispatch.nx = cells;
    dispatch.ny = dispatch.nz = 1;
    dispatch.i_end = cells;
    dispatch.j_end = dispatch.k_end = 1;
    dispatch.dt = duration;
    for (std::size_t field = 0; field < state.size(); ++field)
    {
        dispatch.state[field] = state[field].data();
    }
    for (std::size_t field = 0; field < incoming.size(); ++field)
    {
        dispatch.rates[field] = incoming[field].data();
        dispatch.output[field] = output[field].data();
    }
    for (int cell = 0; cell < cells; ++cell)
    {
        state[0][cell] = 253;
        state[1][cell] = 80000;
        state[2][cell] = std::array<float, 3>{.4f, 1.0f, 1.6f}[cell / 4];
        state[3][cell] = static_cast<float>(.99L * reference::coupled::saturation(253, 80000).mass);
        state[4][cell] = 2e-4f;
        state[10][cell] = 8e7f;
        for (int species = 0; species < 3; ++species)
        {
            if (cell % 4 == 3 || cell % 4 == species)
            {
                state[7 + species][cell] = std::array<float, 3>{1e-4f, 2e-4f, 3e-4f}[species];
                state[13 + species][cell] = std::array<float, 3>{5000, 600, 80}[species];
            }
        }
    }
    for (bool hail : {false, true})
    {
        dispatch.hail_processes = hail;
        REQUIRE(kernels->dispatch_milbrandt_cold(dispatch));
        for (int index = 0; index < cells; ++index)
        {
            std::array<double, 16> cell{};
            for (std::size_t field = 0; field < cell.size(); ++field)
            {
                cell[field] = state[field][index];
            }
            const auto expected = reference::collection::evaluate(reference::collection_source(cell), dispatch.dt);
            const auto check = [&](int field, long double value)
            {
                CAPTURE(name, hail, index, field, output[field][index], static_cast<double>(value));
                REQUIRE(std::isfinite(output[field][index]));
                CHECK(std::abs(static_cast<long double>(output[field][index]) - value)
                    <= 2e-6L * std::max(std::abs(value), 1e-28L));
            };
            check(0, expected.cloud_mass_increment / dispatch.dt);
            check(8, expected.cloud_number_increment / dispatch.dt);
            if (index % 4 == 0)
            {
                const auto snow = reference::snow_diffusion(cell, dispatch.dt, expected.channels[0].mass);
                const long double diffusion = snow.environmental + snow.heating;
                const long double converted = 400.0L / (400.0L - 100.0L) * expected.channels[0].mass;
                check(4, converted / dispatch.dt);
                check(12, cell[13] * converted / (cell[7] + expected.channels[0].mass) / dispatch.dt);
                check(3, (expected.channels[0].mass + diffusion - converted) / dispatch.dt);
                check(6, -diffusion / dispatch.dt);
                check(7, (reference::coupled::latent_heat_sublimation * diffusion
                    + reference::collection::latent_heat_fusion * expected.channels[0].mass)
                    / (reference::collection::heat_capacity * dispatch.dt));
            }
        }
    }
    return 2;
}

template <class Kernels>
std::uint64_t snow_admission_fixtures(Kernels* kernels, const char* name)
{
    const std::array<std::array<double, 16>, 2> cells{{
        {256.3509521484375, 92997.296875, 1.0825551748275757,
            .00032474694307893515, 6.7725557073572418e-7, 1.9364311043545968e-8,
            3.595914677134715e-8, .023380322381854057, .00021004240261390805,
            .0018277277704328299, 50090.95703125, 5.1465916633605957,
            4.5703465730184689e-5, 65093.734375, 378.01608276367188, 2746596864},
        {265.80990600585938, 88899.6875, .94878780841827393,
            .022537695243954659, 3.4392439118846596e-8, 1.3518631931219716e-5,
            .014146114699542522, .015282873064279556, 2.7798416340374388e-5,
            .010872337035834789, 10540.935546875, 24819536, 4956.21484375,
            676.070068359375, 264328.34375, 52801356}}};
    for (std::size_t index = 0; index < cells.size(); ++index)
    {
        CAPTURE(name, index);
        std::array<float, 16> state{};
        std::array<float, 14> incoming{}, output{}, expected{};
        incoming[0] = 1e-7f;
        incoming[6] = -1e-7f;
        incoming[7] = .001f;
        incoming[8] = index == 0 ? 24543230.0f : 176859.203125f;
        expected = incoming;
        output.fill(17);
        MilbrandtColdDispatch dispatch;
        dispatch.nx = dispatch.ny = dispatch.nz = 1;
        dispatch.i_end = dispatch.j_end = dispatch.k_end = 1;
        dispatch.dt = index == 0 ? .040744432778729599 : 5.6542151130055904;
        dispatch.hail_processes = true;
        for (std::size_t field = 0; field < state.size(); ++field)
        {
            state[field] = static_cast<float>(cells[index][field]);
            REQUIRE(static_cast<double>(state[field]) == cells[index][field]);
            dispatch.state[field] = &state[field];
        }
        for (std::size_t field = 0; field < incoming.size(); ++field)
        {
            dispatch.rates[field] = &incoming[field];
            dispatch.output[field] = &output[field];
        }
        const auto& p = dispatch.parameters;
        const tmv::milbrandt_detail::ColdParameters parameters{
            p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9],
            p[10], p[11], p[12], p[13], p[14], p[15], p[16], p[17], p[18],
            p[19], p[20], p[21], p[22], p[23], p[24], true};
        tmv::milbrandt_detail::ColdSnowRimingAdmissionObservation observation;
        if (index == 0)
        {
            REQUIRE_NOTHROW(tmv::milbrandt_detail::cold_cell(parameters, cells[index], dispatch.dt,
                expected, nullptr, &observation));
            REQUIRE(observation.admission_passes == 2);
            REQUIRE(kernels->dispatch_milbrandt_cold(dispatch));
            for (std::size_t field = 0; field < output.size(); ++field)
            {
                CAPTURE(field, output[field], expected[field]);
                const double floor = field < 7 ? 2e-10 : field == 7 ? 2e-7 : .05;
                CHECK(std::abs(double(output[field]) - expected[field]) <=
                    floor + 2e-4 * std::abs(double(expected[field])));
            }
        }
        else
        {
            REQUIRE_THROWS_AS(tmv::milbrandt_detail::cold_cell(parameters, cells[index], dispatch.dt,
                expected, nullptr, &observation), tmv::milbrandt_detail::ColdSnowRimingAdmissionError);
            REQUIRE_FALSE(kernels->dispatch_milbrandt_cold(dispatch));
            for (float value : output)
            {
                CHECK(value == 17);
            }
        }
    }
    return 2;
}

void exercise(const char* name)
{
    constexpr int                      nx = 6, ny = 5, nz = 16, cells = nx * ny * nz;
    std::array<std::vector<float>, 16> state;
    std::array<std::vector<float>, 14> rates, actual, expected;
    for (auto& f : state)
        f.resize(cells);
    for (auto* fields : {&rates, &actual, &expected})
        for (auto& f : *fields)
            f.resize(cells);
    MilbrandtColdDispatch d;
    d.nx      = nx;
    d.ny      = ny;
    d.nz      = nz;
    d.i_begin = 1;
    d.i_end   = nx - 1;
    d.j_begin = 1;
    d.j_end   = ny - 1;
    d.k_end   = nz;
    for (int f = 0; f < 16; ++f)
        d.state[f] = state[f].data();
    for (int f = 0; f < 14; ++f)
    {
        d.rates[f]  = rates[f].data();
        d.output[f] = actual[f].data();
    }
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
    REQUIRE(kernels->supports(MicrophysicsKernelOp::MilbrandtCold));
    std::uint64_t calls = 0;
    for (double duration : {.004, .001, .00025})
    {
        calls += independent_collection(kernels, name, duration);
    }
    calls += snow_admission_fixtures(kernels, name);
    const auto    identical = [&](const auto& reference)
    {
        for (int f = 0; f < 14; ++f)
            CHECK(std::memcmp(actual[f].data(), reference[f].data(), cells * sizeof(float)) == 0);
    };
    for (bool hail : {false, true})
        for (double dt : {.2, 1.0, 10.0})
        {
            std::vector<int> rejected_cells;
            d.hail_processes                              = hail;
            d.dt                                          = dt;
            d.k_begin                                     = dt == 1 ? 1 : 0;
            d.k_end                                       = dt == 1 ? nz - 1 : nz;
            const auto&                                 p = d.parameters;
            const tmv::milbrandt_detail::ColdParameters parameters{
                p[0],  p[1],  p[2],  p[3],  p[4],  p[5],  p[6],  p[7],  p[8],  p[9],  p[10], p[11], p[12],
                p[13], p[14], p[15], p[16], p[17], p[18], p[19], p[20], p[21], p[22], p[23], p[24], hail};
            for (int n = 0; n < cells; ++n)
            {
                state[0][n] = std::array<float, 12>{
                    210, 223, 233, 238, 243, 253, 263, 268, 270, 273.15f, std::nextafter(273.15f, 300.0f), 280}[n % 12];
                state[1][n]      = 50000 + 5000 * (n % 10);
                state[2][n]      = .3f + .15f * (n % 7);
                const double qsi = thermodynamics::saturation_mixing_ratio_ice(state[0][n], state[1][n]);
                state[3][n]      = float(qsi * std::array<double, 5>{0, .5, 1, 1.1, 1.5}[(n / 12) % 5]);
                for (int f = 0; f < 6; ++f)
                {
                    state[4 + f][n]  = std::array<float, 5>{0, 1e-10f, 1e-5f, .0005f, .003f}[(n + f) % 5];
                    state[10 + f][n] = f == 0 ? 2e8f : std::array<float, 5>{0, 1, 100, 1e4f, 1e6f}[(n / 5 + f) % 5];
                }
                for (int f = 0; f < 14; ++f)
                    rates[f][n] = 0;
                rates[0][n] = 1e-7f;
                rates[6][n] = -1e-7f;
                rates[7][n] = .001f;
                rates[8][n] = n % 2 == 0 ? float(1e6 / dt) : 0;
                std::array<double, 16> cell;
                std::array<float, 14>  reference;
                for (int f = 0; f < 16; ++f)
                    cell[f] = state[f][n];
                for (int f = 0; f < 14; ++f)
                    reference[f] = rates[f][n];
                const int i = n / (ny * nz), j = n / nz % ny, k = n % nz;
                if (i >= d.i_begin && i < d.i_end && j >= d.j_begin && j < d.j_end && k >= d.k_begin && k < d.k_end)
                {
                    try
                    {
                        tmv::milbrandt_detail::cold_cell(parameters, cell, dt, reference);
                    }
                    catch (const tmv::milbrandt_detail::ColdSnowRimingAdmissionError& failure)
                    {
                        using Failure = tmv::milbrandt_detail::ColdSnowRimingAdmissionFailure;
                        REQUIRE((failure.kind() == Failure::BudgetCap || failure.kind() == Failure::InvalidEndpoint));
                        rejected_cells.push_back(n);
                        for (int field = 0; field < 14; ++field)
                        {
                            CHECK(reference[field] == rates[field][n]);
                        }
                    }
                }
                for (int f = 0; f < 14; ++f)
                {
                    expected[f][n] = reference[f];
                    actual[f][n]   = 17;
                }
            }
            if (!rejected_cells.empty())
            {
                const auto before = actual;
                REQUIRE_FALSE(kernels->dispatch_milbrandt_cold(d));
                ++calls;
                identical(before);
                for (int rejected : rejected_cells)
                {
                    std::array<float, 16> bad_state{};
                    std::array<float, 14> bad_rates{}, sentinel{};
                    sentinel.fill(17);
                    MilbrandtColdDispatch single = d;
                    single.nx = single.ny = single.nz = 1;
                    single.i_begin = single.j_begin = single.k_begin = 0;
                    single.i_end = single.j_end = single.k_end = 1;
                    for (int field = 0; field < 16; ++field)
                    {
                        bad_state[field] = state[field][rejected];
                        single.state[field] = &bad_state[field];
                    }
                    for (int field = 0; field < 14; ++field)
                    {
                        bad_rates[field] = rates[field][rejected];
                        single.rates[field] = &bad_rates[field];
                        single.output[field] = &sentinel[field];
                    }
                    CAPTURE(name, hail, dt, rejected);
                    REQUIRE_FALSE(kernels->dispatch_milbrandt_cold(single));
                    ++calls;
                    for (float value : sentinel)
                    {
                        CHECK(value == 17);
                    }
                    for (int field = 3; field < 16; ++field)
                    {
                        state[field][rejected] = 0;
                    }
                    for (int field = 0; field < 14; ++field)
                    {
                        rates[field][rejected] = 0;
                        expected[field][rejected] = 0;
                    }
                }
            }
            const bool admitted_batch = kernels->dispatch_milbrandt_cold(d);
            ++calls;
            if (!admitted_batch)
            {
                for (int n = 0; n < cells; ++n)
                {
                    const int i = n / (ny * nz), j = n / nz % ny, k = n % nz;
                    if (i < d.i_begin || i >= d.i_end || j < d.j_begin || j >= d.j_end
                        || k < d.k_begin || k >= d.k_end)
                    {
                        continue;
                    }
                    std::array<float, 16> point{};
                    std::array<float, 14> point_rates{}, point_output{};
                    MilbrandtColdDispatch single = d;
                    single.nx = single.ny = single.nz = 1;
                    single.i_begin = single.j_begin = single.k_begin = 0;
                    single.i_end = single.j_end = single.k_end = 1;
                    for (int field = 0; field < 16; ++field)
                    {
                        point[field] = state[field][n];
                        single.state[field] = &point[field];
                    }
                    for (int field = 0; field < 14; ++field)
                    {
                        point_rates[field] = rates[field][n];
                        single.rates[field] = &point_rates[field];
                        single.output[field] = &point_output[field];
                    }
                    CAPTURE(name, hail, dt, n);
                    REQUIRE(kernels->dispatch_milbrandt_cold(single));
                    ++calls;
                }
            }
            REQUIRE(admitted_batch);
            for (int f = 0; f < 14; ++f)
                for (int n = 0; n < cells; ++n)
                {
                    CAPTURE(name, hail, dt, f, n, actual[f][n], expected[f][n]);
                    const double floor = f < 7 ? 2e-10 : f == 7 ? 2e-7 : .05;
                    CHECK(std::abs(double(actual[f][n]) - expected[f][n]) <=
                          floor + 2e-4 * std::abs(double(expected[f][n])));
                }
            const auto first = actual;
            REQUIRE(kernels->dispatch_milbrandt_cold(d));
            ++calls;
            identical(first);
        }
    const auto accepted = actual;
    d.dt                = 1e-36;
    std::size_t checked = 0;
    REQUIRE(nimbus::compute::milbrandt_cold::validate(d, checked));
    CHECK_FALSE(kernels->dispatch_milbrandt_cold(d));
    ++calls;
    identical(accepted);
    d.dt = 10;

    d.parameters[0] = 20;
    CHECK_FALSE(kernels->dispatch_milbrandt_cold(d));
    identical(accepted);
    d.parameters = tmv::milbrandt_detail::default_cold_parameters;
    d.output[13] = actual[0].data();
    CHECK_FALSE(kernels->dispatch_milbrandt_cold(d));
    identical(accepted);
    d.output[13] = state[0].data();
    CHECK_FALSE(kernels->dispatch_milbrandt_cold(d));
    identical(accepted);
    d.output[13] = actual[13].data();
    d.nx         = std::numeric_limits<int>::max();
    d.ny         = std::numeric_limits<int>::max();
    CHECK_FALSE(kernels->dispatch_milbrandt_cold(d));
    identical(accepted);
    d.nx        = nx;
    d.ny        = ny;
    state[0][0] = std::numeric_limits<float>::quiet_NaN();
    CHECK_FALSE(kernels->dispatch_milbrandt_cold(d));
    identical(accepted);
    CHECK(runtime.backend->successful_kernel_dispatch_count(nimbus::compute::KernelId::MilbrandtCold) == calls);
}
} // namespace
TEST_CASE("Metal Milbrandt cold processes match the CPU reference", "[native-milbrandt-cold][metal]")
{
    exercise("metal");
}
TEST_CASE("Vulkan Milbrandt cold processes match the CPU reference", "[native-milbrandt-cold][vulkan]")
{
    exercise("vulkan");
}
