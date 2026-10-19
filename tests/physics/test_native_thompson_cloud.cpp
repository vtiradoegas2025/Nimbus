#include "catch2/catch.hpp"
#include "compute/backend_runtime.hpp"
#include "compute/thompson_cloud_contract.hpp"
#include "microphysics/base/thermodynamics.hpp"
#include "microphysics/schemes/thompson/native_cloud_tables.hpp"
#include "microphysics/schemes/thompson/nucleation.hpp"
#include "microphysics/schemes/thompson/thermodynamic_helpers.hpp"
#include "microphysics/schemes/thompson/required_packaged_data.hpp"

#include <array>
#include <cmath>
#include <limits>

namespace
{
struct Tables
{
    std::shared_ptr<const tmv::thompson::ActivationTable> activation =
        tmv::thompson::require_packaged_activation_table();
    tmv::thompson::CloudEvaporationTable evaporation = tmv::thompson::generate_cloud_evaporation_table();
    std::shared_ptr<const nimbus::compute::ThompsonCloudTables> native =
        tmv::thompson::pack_native_cloud_tables(*activation, evaporation);
};

std::array<double, 5> reference(const ThompsonCloudAdjustmentDispatch& d, const Tables& tables, int n)
{
    using namespace tmv::thompson;
    const double saturation     = tmv::thermodynamics::saturation_mixing_ratio_water(d.temperature[n], d.pressure[n]);
    const double vapor          = d.vapor[n];
    double       adjusted_vapor = vapor;
    double       adjusted_cloud = d.cloud[n];
    const auto   decision       = [&](float evaporated_cloud)
    {
        return d.potential_temperature
                   ? liquid_saturation_resolution_from_theta(
                         d.potential_temperature[n], d.pressure[n], d.vapor[n], evaporated_cloud)
                   : liquid_saturation_resolution(d.temperature[n], d.pressure[n], d.vapor[n], evaporated_cloud);
    };
    const auto resolution = decision(0);
    if (resolution.condensation || resolution.evaporation)
        tmv::thermodynamics::saturation_adjustment(d.temperature[n], d.pressure[n], adjusted_vapor, adjusted_cloud);
    double       change                = adjusted_vapor - vapor;
    const double exhausted_temperature = d.temperature[n] - (2.5e6 / 1004.0) * d.cloud[n];
    const bool   exhausted_resolved    = exhausted_temperature >= 150 && !decision(d.cloud[n]).condensation;
    const bool   fully_evaporated      = change > 0 && (d.density[n] * adjusted_cloud <= 1e-12 || exhausted_resolved);
    if (fully_evaporated)
        change = d.cloud[n];
    double exchange = 0;
    double index    = -1;
    if (fully_evaporated)
    {
        exchange = -d.cloud_number[n];
    }
    else if (change < -1e-15 && resolution.condensation)
    {
        exchange = activate_cloud_droplets(*tables.activation,
                                           d.temperature[n],
                                           d.velocity[n],
                                           d.density[n],
                                           d.cloud_number[n],
                                           d.water_aerosol[n],
                                           d.density[n] * adjusted_cloud)
                       .hydrometeor_kg_dry;
        index    = -2 - std::lround(std::clamp((d.temperature[n] - 243.15) * 0.1, 0.0, 6.0));
    }
    else if (change > 1e-15 && resolution.evaporation && vapor / saturation - 1 < -1e-6)
    {
        const double diameter = maximum_fully_evaporated_cloud_diameter(
            d.temperature[n], d.pressure[n], d.density[n], vapor, saturation, d.dt);
        const double mass   = static_cast<double>(d.cloud[n]) * d.density[n];
        const double number = static_cast<double>(d.cloud_number[n]) * d.density[n];
        exchange = return_evaporated_cloud_number(tables.evaporation, diameter, mass, d.cloud_number[n], d.density[n])
                       .cloud_number_kg_dry;
        index    = cloud_evaporation_index(cloud_evaporation_diameter_index(diameter),
                                           cloud_evaporation_content_index(mass),
                                           cloud_evaporation_number_index(number));
    }
    return {change, -2.5e6 / 1004.0 * change, exchange / d.dt, -exchange / d.dt, index};
}

void exercise(const char* backend)
{
    Tables                     tables;
    ComputeBackendRuntimeState runtime;
    runtime.config.backend        = backend;
    runtime.config.allow_fallback = false;
    std::string error;
    REQUIRE(initialize_compute_backend_runtime(runtime, CoordinateSystem::Cartesian, error));
    auto* kernels = runtime.backend->microphysics_kernels();
    REQUIRE(kernels != nullptr);
    REQUIRE(kernels->supports(MicrophysicsKernelOp::ThompsonCloudAdjustment));
    constexpr int          nx = 8, ny = 5, nz = 4, size = nx * ny * nz;
    const float            nan = std::numeric_limits<float>::quiet_NaN();
    std::array<Field3D, 8> fields;
    Field3D theta(nx, ny, nz, nan);
    std::array<Field3D, 7> result;
    for (auto& field : fields)
    {
        field = Field3D(nx, ny, nz, nan);
    }
    for (auto& field : result)
    {
        field = Field3D(nx, ny, nz, 123.0f);
    }
    ThompsonCloudAdjustmentDispatch d;
    d.temperature           = fields[0].data();
    d.pressure              = fields[1].data();
    d.density               = fields[2].data();
    d.vapor                 = fields[3].data();
    d.cloud                 = fields[4].data();
    d.cloud_number          = fields[5].data();
    d.water_aerosol         = fields[6].data();
    d.velocity              = fields[7].data();
    d.tables                = tables.native;
    d.vapor_change          = result[0].data();
    d.temperature_change    = result[1].data();
    d.cloud_number_tendency = result[2].data();
    d.aerosol_tendency      = result[3].data();
    d.lookup_index          = result[4].data();
    d.vapor_change_low       = result[5].data();
    d.temperature_change_low = result[6].data();
    d.nx                    = nx;
    d.ny                    = ny;
    d.nz                    = nz;
    d.i_begin               = 1;
    d.i_end                 = nx - 1;
    d.j_begin               = 1;
    d.j_end                 = ny - 1;
    d.k_begin               = 1;
    d.k_end                 = nz - 1;
    REQUIRE(tables.native->evaporation().size() > size * 2000);
    for (double dt : {0.01, 0.2, 10.0})
    {
        d.dt = dt;
        for (int scenario : {0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 5})
        {
            for (int i = 1; i < nx - 1; ++i)
            {
                for (int j = 1; j < ny - 1; ++j)
                {
                    for (int k = 1; k < nz - 1; ++k)
                    {
                        const int   n       = (i * ny + j) * nz + k;
                        const float t       = 238.15f + 10.0f * i;
                        fields[0].data()[n] = k == 1 ? t : std::nextafter(t, std::numeric_limits<float>::infinity());
                        fields[1].data()[n] = 70000.0f + 5000.0f * j;
                        fields[2].data()[n] = 0.75f;
                        const float qsat    = static_cast<float>(tmv::thermodynamics::saturation_mixing_ratio_water(
                            fields[0].data()[n], fields[1].data()[n]));
                        fields[3].data()[n] = scenario == 0   ? qsat + 0.003f
                                              : scenario == 1 ? qsat * 0.7f
                                              : scenario == 2 ? qsat
                                                              : std::nextafter(qsat, scenario == 3 ? 0.0f : 1.0f);
                        if (scenario == 5)
                        {
                            fields[3].data()[n] = qsat * 0.7f;
                        }
                        fields[4].data()[n] = static_cast<float>((1 + i) * std::pow(10.0, -6 + j));
                        if (k == 2)
                        {
                            fields[4].data()[n] = std::nextafter(fields[4].data()[n], 0.0f);
                        }
                        fields[5].data()[n] = scenario == 5 ? 2.4e9f : 5e7f;
                        fields[6].data()[n] = 350e6f;
                        fields[7].data()[n] = j == 1 ? -0.2f : j == 2 ? 0.1f : 120.0f;
                        if (scenario == 6 || scenario == 7)
                        {
                            fields[0].data()[n] = 236.40225219726562f;
                            fields[1].data()[n] = 32419.28125f;
                            fields[4].data()[n] = scenario == 6 ? 1e-6f : 0;
                            fields[5].data()[n] = scenario == 6 ? 5e7f : 0;
                            if (scenario == 6)
                            {
                                const double amount               = double(fields[4].data()[n]) - 2e-10;
                                const double endpoint_temperature = fields[0].data()[n] - (2.5e6 / 1004.0) * amount;
                                fields[3].data()[n] = float(tmv::thermodynamics::saturation_mixing_ratio_water(
                                                                endpoint_temperature, fields[1].data()[n]) -
                                                            amount);
                            }
                            else
                                fields[3].data()[n] = .0005077312234789133f;
                        }
                    }
                }
            }
            d.potential_temperature = scenario >= 8 ? theta.data() : nullptr;
            if (scenario >= 8)
                for (int i = 1; i < nx - 1; ++i)
                    for (int j = 1; j < ny - 1; ++j)
                        for (int k = 1; k < nz - 1; ++k)
                        {
                            const int n         = (i * ny + j) * nz + k;
                            theta.data()[n]     = k == 1 ? 336.94647216796875f : 336.9464416503906f;
                            fields[1].data()[n] = 26041.193359375f;
                            fields[0].data()[n] =
                                float(tmv::thermodynamics::theta_to_temperature(theta.data()[n], fields[1].data()[n]));
                            fields[3].data()[n] = k == 1 ? .000304150307783857f : .00030415027868002653f;
                            fields[3].data()[n] *= scenario == 9 ? 1.001f : scenario == 10 ? .999f : 1;
                            fields[4].data()[n] = scenario == 10 ? 1e-6f : 0;
                            fields[5].data()[n] = scenario == 10 ? 1e6f : 0;
                            if (scenario == 11)
                            {
                                fields[4].data()[n] = 1e-6f;
                                fields[5].data()[n] = 5e7f;
                                const double amount = double(fields[4].data()[n]) - 2e-10;
                                fields[3].data()[n] =
                                    float(tmv::thermodynamics::saturation_mixing_ratio_water(
                                              fields[0].data()[n] - (2.5e6 / 1004.0) * amount, fields[1].data()[n]) -
                                          amount);
                            }
                            if (scenario == 12 || scenario == 13)
                            {
                                const bool cpu_input = scenario == 12;
                                theta.data()[n]      = cpu_input ? 305.5755920410156f : 305.5749206542969f;
                                fields[1].data()[n]  = cpu_input ? 81576.328125f : 81576.375f;
                                fields[0].data()[n]  = float(
                                    tmv::thermodynamics::theta_to_temperature(theta.data()[n], fields[1].data()[n]));
                                fields[2].data()[n] = cpu_input ? .9859173893928528f : .9859198927879333f;
                                fields[3].data()[n] = cpu_input ? .013406486250460148f : .013406498357653618f;
                                fields[4].data()[n] = cpu_input ? 1.7000461127736344e-7f : 1.7045836386841984e-7f;
                                fields[5].data()[n] = cpu_input ? 950517.6875f : 950510.125f;
                                fields[6].data()[n] = cpu_input ? 216105072.0f : 216105312.0f;
                                fields[7].data()[n] = 2.178f;
                            }
                        }
            INFO(backend);
            INFO(dt);
            INFO(scenario);
            REQUIRE(kernels->dispatch_thompson_cloud_adjustment(d));
            const auto first = result;
            REQUIRE(kernels->dispatch_thompson_cloud_adjustment(d));
            for (int f = 0; f < 7; ++f)
            {
                CHECK(std::equal(first[f].data(), first[f].data() + size, result[f].data()));
            }
            for (int i = 0; i < nx; ++i)
            {
                for (int j = 0; j < ny; ++j)
                {
                    for (int k = 0; k < nz; ++k)
                    {
                        const int n = (i * ny + j) * nz + k;
                        INFO(n);
                        if (i == 0 || i == nx - 1 || j == 0 || j == ny - 1 || k == 0 || k == nz - 1)
                        {
                            for (int f = 0; f < 4; ++f)
                            {
                                CHECK(result[f].data()[n] == 0.0f);
                            }
                            CHECK(result[4].data()[n] == -1.0f);
                            continue;
                        }
                        if (scenario == 11)
                        {
                            CHECK(result[0].data()[n] == fields[4].data()[n]);
                            CHECK(std::abs(double(fields[5].data()[n]) + dt * result[2].data()[n]) < 8);
                        }
                        if (scenario == 8)
                        {
                            CHECK(result[0].data()[n] == 0);
                            CHECK(result[2].data()[n] == 0);
                        }
                        if (scenario == 6)
                        {
                            CHECK(result[0].data()[n] == d.cloud[n]);
                            CHECK(d.cloud_number[n] + dt * result[2].data()[n] <=
                                  8 * std::numeric_limits<float>::epsilon() * d.cloud_number[n]);
                        }
                        if (scenario == 7)
                        {
                            CHECK(result[0].data()[n] == 0);
                            CHECK(result[1].data()[n] == 0);
                            CHECK(result[2].data()[n] == 0);
                            CHECK(result[3].data()[n] == 0);
                        }
                        const auto expected = reference(d, tables, n);
                        if (scenario == 12)
                        {
                            CHECK(result[0].data()[n] > 0);
                            CHECK(result[2].data()[n] <= 0);
                        }
                        if (scenario == 13)
                        {
                            CHECK(result[0].data()[n] < 0);
                            CHECK(result[2].data()[n] > 0);
                        }
                        CHECK(result[4].data()[n] == expected[4]);
                        CHECK(std::abs((double(result[0].data()[n]) + result[5].data()[n]) - expected[0]) <=
                              2e-6 * std::abs(expected[0]) + 1e-13);
                        CHECK(std::abs((double(result[1].data()[n]) + result[6].data()[n]) - expected[1]) <=
                              3e-6 * std::abs(expected[1]) + 3e-10);
                        const double number_tolerance =
                            5e-5 * std::abs(expected[2]) + 2e-7 * (d.cloud_number[n] + d.water_aerosol[n]) / dt + 1e-6;
                        CHECK(std::abs(result[2].data()[n] - expected[2]) <= number_tolerance);
                        CHECK(result[2].data()[n] == -result[3].data()[n]);
                        CHECK(static_cast<double>(d.cloud_number[n]) + dt * result[2].data()[n] >= 0.0);
                        CHECK(static_cast<double>(d.water_aerosol[n]) + dt * result[3].data()[n] >= 0.0);
                        CHECK(static_cast<double>(d.vapor[n]) + double(result[0].data()[n]) + result[5].data()[n] >=
                              0.0);
                        CHECK(static_cast<double>(d.cloud[n]) - (double(result[0].data()[n]) + result[5].data()[n]) >=
                              0.0);
                    }
                }
            }
        }
    }
    fields[5](1, 1, 1) = 0.0f;
    fields[6](1, 1, 1) = 1e-5f;
    fields[3](1, 1, 1) += 0.003f;
    d.dt = 0.01;
    REQUIRE(kernels->dispatch_thompson_cloud_adjustment(d));
    CHECK(result[2](1, 1, 1) > 0.0f);
    CHECK(static_cast<double>(fields[6](1, 1, 1)) + d.dt * result[3](1, 1, 1) >= 0.0);
    for (int i = 1; i < nx - 1; ++i)
        for (int j = 1; j < ny - 1; ++j)
            for (int k = 1; k < nz - 1; ++k)
            {
                fields[0](i, j, k) = 240.20846557617188f;
                fields[1](i, j, k) = 35043.56640625f;
                fields[2](i, j, k) = .5083209276199341f;
                fields[3](i, j, k) = .00014423704124055803f;
                fields[4](i, j, k) = .0009458515560254455f;
                fields[5](i, j, k) = 2530718;
                fields[6](i, j, k) = 114829304;
                fields[7](i, j, k) = .01540290703997016f;
            }
    d.dt = .25;
    REQUIRE(kernels->dispatch_thompson_cloud_adjustment(d));
    const int  captured       = (ny + 1) * nz + 1;
    const auto expected_cloud = reference(d, tables, captured);
    for (int f = 0; f < 2; ++f)
    {
        const double pair      = double(result[f].data()[captured]) + result[5 + f].data()[captured];
        const double tolerance = 2e-12 * std::abs(expected_cloud[f]) + 1e-16;
        CHECK(std::abs(pair - expected_cloud[f]) <= tolerance);
        CHECK(std::abs(double(result[f].data()[captured]) - expected_cloud[f]) > 100 * tolerance);
    }
    d.potential_temperature = nullptr;
    d.dt = .5;
    for (int i = 1; i < nx - 1; ++i)
        for (int j = 1; j < ny - 1; ++j)
            for (int k = 1; k < nz - 1; ++k)
            {
                fields[0](i, j, k) = 280;
                fields[1](i, j, k) = 95000;
                fields[2](i, j, k) = 1;
                fields[3](i, j, k) = 0;
                fields[4](i, j, k) = 1e-8f;
                fields[5](i, j, k) = 1e6f;
                fields[6](i, j, k) = 1e6f;
                fields[7](i, j, k) = 0;
            }
    REQUIRE(kernels->dispatch_thompson_cloud_adjustment(d));
    for (int i = 1; i < nx - 1; ++i)
        for (int j = 1; j < ny - 1; ++j)
            for (int k = 1; k < nz - 1; ++k)
            {
                CHECK(fields[5](i, j, k) + d.dt * result[2](i, j, k) == 0);
                CHECK(result[2](i, j, k) == -result[3](i, j, k));
            }
    for (int i = 1; i < nx - 1; ++i)
        for (int j = 1; j < ny - 1; ++j)
            for (int k = 1; k < nz - 1; ++k)
            {
                fields[5](i, j, k) = 1;
                fields[6](i, j, k) = 1;
            }
    const double exact_tiny_step = std::ldexp(1.0, -120);
    for (double step : {exact_tiny_step, std::nextafter(exact_tiny_step, std::numeric_limits<double>::infinity())})
    {
        d.dt = step;
        REQUIRE(kernels->dispatch_thompson_cloud_adjustment(d));
        const double remaining = 1 + step * result[2](1, 1, 1);
        CHECK(remaining >= 0);
        if (step == exact_tiny_step)
            CHECK(remaining == 0);
    }
    const auto before    = result;
    const auto unchanged = [&]()
    {
        for (int f = 0; f < 7; ++f)
        {
            CHECK(std::equal(before[f].data(), before[f].data() + size, result[f].data()));
        }
    };
    d.vapor_change_low = result[0].data() + 1;
    CHECK_FALSE(kernels->dispatch_thompson_cloud_adjustment(d));
    unchanged();
    d.vapor_change_low       = result[5].data();
    d.temperature_change_low = nullptr;
    CHECK_FALSE(kernels->dispatch_thompson_cloud_adjustment(d));
    unchanged();
    d.temperature_change_low = result[6].data();
    d.potential_temperature = result[0].data();
    CHECK_FALSE(kernels->dispatch_thompson_cloud_adjustment(d));
    unchanged();
    for (int i = 1; i < nx - 1; ++i)
        for (int j = 1; j < ny - 1; ++j)
            for (int k = 1; k < nz - 1; ++k)
            {
                theta(i, j, k) = 300;
                fields[0](i, j, k) =
                    float(tmv::thermodynamics::theta_to_temperature(theta(i, j, k), fields[1](i, j, k)));
            }
    d.potential_temperature       = theta.data();
    theta(nx - 2, ny - 2, nz - 2) = 350;
    CHECK_FALSE(kernels->dispatch_thompson_cloud_adjustment(d));
    unchanged();
    theta(nx - 2, ny - 2, nz - 2) = nan;
    CHECK_FALSE(kernels->dispatch_thompson_cloud_adjustment(d));
    unchanged();
    d.potential_temperature = nullptr;
    d.cloud_number_tendency = result[0].data() + 1;
    CHECK_FALSE(kernels->dispatch_thompson_cloud_adjustment(d));
    unchanged();
    d.cloud_number_tendency = result[2].data();
    d.vapor_change          = fields[0].data() + 1;
    CHECK_FALSE(kernels->dispatch_thompson_cloud_adjustment(d));
    unchanged();
    d.vapor_change = const_cast<float*>(d.tables->evaporation().data() + 200000);
    CHECK_FALSE(kernels->dispatch_thompson_cloud_adjustment(d));
    unchanged();
    d.vapor_change            = result[0].data();
    const float number_before = fields[5](1, 1, 1);
    fields[5](1, 1, 1)        = 0.0f;
    fields[3](1, 1, 1)        = 0.0f;
    CHECK_FALSE(kernels->dispatch_thompson_cloud_adjustment(d));
    unchanged();
    fields[5](1, 1, 1) = number_before;
    fields[2](1, 1, 1) = nan;
    CHECK_FALSE(kernels->dispatch_thompson_cloud_adjustment(d));
    unchanged();
}
} // namespace

TEST_CASE("Thompson native tables reject unrepresentable payloads and edges", "[native-thompson-cloud][contract]")
{
    Tables tables;
    using Native = nimbus::compute::ThompsonCloudTables;
    std::array<double, Native::edge_count> edges;
    for (std::size_t i = 0; i < edges.size(); ++i)
    {
        edges[i] = static_cast<double>(tables.native->evaporation()[Native::evaporation_count + 2 * i]) +
                   tables.native->evaporation()[Native::evaporation_count + 2 * i + 1];
    }
    const std::vector<float> evaporation(tables.evaporation.evaporated_number_m3.begin(),
                                         tables.evaporation.evaporated_number_m3.end());
    REQUIRE(Native::create(tables.activation->fractions, evaporation, edges));
    auto invalid   = edges;
    invalid.back() = std::numeric_limits<double>::max();
    CHECK_FALSE(Native::create(tables.activation->fractions, evaporation, invalid));
    invalid         = edges;
    invalid.front() = std::numeric_limits<double>::denorm_min();
    CHECK_FALSE(Native::create(tables.activation->fractions, evaporation, invalid));
    tables.evaporation.evaporated_number_m3.front() = std::numeric_limits<double>::max();
    CHECK_THROWS_AS(tmv::thompson::pack_native_cloud_tables(*tables.activation, tables.evaporation),
                    std::invalid_argument);
}

TEST_CASE("Metal Thompson cloud tables preserve saturation and aerosol exchange", "[native-thompson-cloud][metal]")
{
    exercise("metal");
}

TEST_CASE("Vulkan Thompson cloud tables preserve saturation and aerosol exchange", "[native-thompson-cloud][vulkan]")
{
    exercise("vulkan");
}

TEST_CASE("Thompson number exchange requires resolved liquid supersaturation", "[thompson][cloud][resolution]")
{
    for (const auto state : {std::array<float, 3>{236.40225219726562f, 32419.28125f, .0005077312234789133f},
                             std::array<float, 3>{236.40228271484375f, 32419.28515625f, .0005077315727248788f}})
    {
        const auto decision = tmv::thompson::liquid_saturation_resolution(state[0], state[1], state[2]);
        CHECK_FALSE(decision.condensation);
        CHECK_FALSE(decision.evaporation);
        CHECK(tmv::thompson::liquid_saturation_resolution(state[0], state[1], state[2] * 1.001f).condensation);
        CHECK(tmv::thompson::liquid_saturation_resolution(state[0], state[1], state[2] * .999f).evaporation);
    }
}

TEST_CASE("Thompson liquid resolution propagates stored potential temperature", "[thompson][cloud][resolution]")
{
    for (float theta : {336.94647216796875f, 336.9464416503906f})
    {
        const float pressure = 26041.193359375f, vapor = .00030415027868002653f;
        const auto  state = tmv::thompson::liquid_saturation_resolution_from_theta(theta, pressure, vapor);
        CHECK_FALSE(state.condensation);
        CHECK_FALSE(state.evaporation);
        CHECK(tmv::thompson::liquid_saturation_resolution_from_theta(theta, pressure, vapor * 1.001f).condensation);
        CHECK(tmv::thompson::liquid_saturation_resolution_from_theta(theta, pressure, vapor * .999f).evaporation);
    }
}
