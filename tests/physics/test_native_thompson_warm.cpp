#include "catch2/catch.hpp"
#include "compute/backend_runtime.hpp"
#include "core/field/field3d.hpp"
#include "compute/thompson_warm_contract.hpp"
#include "microphysics/schemes/thompson/moment_initialization.hpp"
#include "microphysics/schemes/thompson/thompson.hpp"
#include "microphysics/schemes/thompson/thermodynamic_helpers.hpp"
#include "microphysics/schemes/thompson/warm_rain_processes.hpp"

#include <array>
#include <cmath>
#include <limits>

namespace
{
float stored(double initial, double candidate, double dt)
{
    float rate = static_cast<float>((candidate - initial) / dt);
    return initial + dt * rate < 0 ? std::nextafter(rate, std::numeric_limits<float>::infinity()) : rate;
}

std::array<double, 7> reference(const ThompsonWarmRainDispatch& d, int n)
{
    using namespace tmv::thompson;
    const double rho        = d.density[n];
    double       q = static_cast<double>(d.cloud[n]) - (double(d.cloud_vapor_change[n]) + d.cloud_vapor_change_low[n]);
    double       r          = d.rain[n];
    const double v = static_cast<double>(d.vapor[n]) + (double(d.cloud_vapor_change[n]) + d.cloud_vapor_change_low[n]);
    float        nc_rate    = d.cloud_number_source[n];
    float        nr_rate    = 0;
    const auto   conversion = warm_rain_conversion_rates(d.temperature[n],
                                                         rho,
                                                         rho * q,
                                                         rho * r,
                                                         rho * (d.cloud_number[n] + d.dt * nc_rate),
                                                         rho * r > 1e-12
                                                             ? normalize_rain_number_m3(rho * r, rho * d.rain_number[n])
                                                             : rho * d.rain_number[n],
                                                         d.dt);
    const double converted  = d.dt * conversion.rain_water_kg_m3_s / rho;
    q -= converted;
    r += converted;
    nc_rate += static_cast<float>(conversion.cloud_number_m3_s / rho);
    nr_rate += static_cast<float>(conversion.rain_number_m3_s / rho);
    double nc = std::max(0.0, d.cloud_number[n] + d.dt * nc_rate);
    double nr = std::max(0.0, d.rain_number[n] + d.dt * nr_rate);
    if (rho * q > 1e-12)
    {
        nc = normalize_cloud_number_m3(rho * q, rho * nc) / rho;
    }
    if (rho * r > 1e-12)
    {
        nr = normalize_rain_number_m3(rho * r, rho * nr) / rho;
    }
    const float  unprojected_cloud_rate = nc_rate;
    const double raw_nc                 = d.cloud_number[n] + d.dt * nc_rate;
    if (raw_nc < 0 || rho * raw_nc > 1999e6 ||
        std::abs(nc - raw_nc) >
            8 * std::numeric_limits<float>::epsilon() * std::max({1.0, std::abs(nc), std::abs(raw_nc)}))
        nc_rate = stored(d.cloud_number[n], nc, d.dt);
    nr_rate                = stored(d.rain_number[n], nr, d.dt);
    const auto evaporation = rain_evaporation_rates(
        d.pressure[n], d.temperature[n], rho, v, rho * r, rho * (d.rain_number[n] + d.dt * nr_rate), d.dt);
    nr_rate =
        stored(d.rain_number[n], d.rain_number[n] + d.dt * nr_rate + d.dt * evaporation.rain_number_m3_s / rho, d.dt);
    const float  baseline_rate      = float(conversion.cloud_number_m3_s / rho);
    const double baseline_content   = rho * std::max(0.0, double(d.cloud[n]) - converted);
    const double baseline_candidate = d.cloud_number[n] + d.dt * baseline_rate;
    const double baseline_normalized =
        baseline_content > 1e-12
            ? normalize_cloud_number_m3(baseline_content, rho * std::max(0.0, baseline_candidate)) / rho
            : std::max(0.0, baseline_candidate);
    const double baseline_rounding      = 8 * std::numeric_limits<float>::epsilon() *
                                          std::max({1.0, std::abs(baseline_candidate), std::abs(baseline_normalized)});
    const float baseline_projected_rate = baseline_candidate < 0 || rho * baseline_candidate > 1999e6 ||
                                                  std::abs(baseline_normalized - baseline_candidate) > baseline_rounding
                                              ? stored(d.cloud_number[n], baseline_normalized, d.dt)
                                              : baseline_rate;
    const double baseline_loss          = std::max(0.0, double(baseline_rate) - baseline_projected_rate);
    const double actual_loss            = std::max(0.0, double(unprojected_cloud_rate) - nc_rate);
    const float  aerosol_source =
        (double(d.cloud_vapor_change[n]) + d.cloud_vapor_change_low[n]) > 0
            ? float(double(d.cloud_aerosol_source[n]) + std::max(0.0, actual_loss - baseline_loss))
            : d.cloud_aerosol_source[n];
    const float na_rate = aerosol_source + static_cast<float>(evaporation.water_aerosol_m3_s / rho);
    return {converted,
            d.dt * evaporation.vapor_kg_m3_s / rho,
            d.dt * evaporation.temperature_k_s,
            nc_rate,
            nr_rate,
            na_rate,
            (conversion.cloud_number_m3_s + evaporation.water_aerosol_m3_s) / rho};
}

void exercise(const char* backend)
{
    ComputeBackendRuntimeState runtime;
    runtime.config.backend        = backend;
    runtime.config.allow_fallback = false;
    std::string error;
    REQUIRE(initialize_compute_backend_runtime(runtime, CoordinateSystem::Cartesian, error));
    auto* kernels = runtime.backend->microphysics_kernels();
    REQUIRE(kernels);
    REQUIRE(kernels->supports(MicrophysicsKernelOp::ThompsonWarmRain));
    constexpr int           nx = 8, ny = 5, nz = 4, size = nx * ny * nz;
    const float             nan = std::numeric_limits<float>::quiet_NaN();
    std::array<Field3D, 12> fields;
    std::array<Field3D, 10> result;
    for (auto& field : fields)
    {
        field = Field3D(nx, ny, nz, nan);
    }
    for (auto& field : result)
    {
        field = Field3D(nx, ny, nz, 123.0f);
    }
    Field3D                  cloud_low(nx, ny, nz, 0);
    ThompsonWarmRainDispatch d;
    d.temperature           = fields[0].data();
    d.pressure              = fields[1].data();
    d.density               = fields[2].data();
    d.vapor                 = fields[3].data();
    d.cloud                 = fields[4].data();
    d.rain                  = fields[5].data();
    d.cloud_number          = fields[6].data();
    d.rain_number           = fields[7].data();
    d.water_aerosol         = fields[8].data();
    d.cloud_vapor_change    = fields[9].data();
    d.cloud_vapor_change_low = cloud_low.data();
    d.cloud_number_source   = fields[10].data();
    d.cloud_aerosol_source  = fields[11].data();
    d.coefficients          = tmv::thompson::native_warm_rain_coefficients();
    d.cloud_to_rain_high    = result[0].data();
    d.cloud_to_rain_low     = result[1].data();
    d.rain_to_vapor_high    = result[2].data();
    d.rain_to_vapor_low     = result[3].data();
    d.temperature_high      = result[4].data();
    d.temperature_low       = result[5].data();
    d.cloud_number_tendency = result[6].data();
    d.rain_number_tendency  = result[7].data();
    d.aerosol_tendency      = result[8].data();
    d.status                = result[9].data();
    d.nx                    = nx;
    d.ny                    = ny;
    d.nz                    = nz;
    d.i_begin               = 1;
    d.i_end                 = nx - 1;
    d.j_begin               = 1;
    d.j_end                 = ny - 1;
    d.k_begin               = 1;
    d.k_end                 = nz - 1;
    for (double dt : {0.01, 0.03, 0.2, 1.3, 10.0})
    {
        d.dt = dt;
        for (int scenario = 0; scenario < 50; ++scenario)
        {
            CAPTURE(backend, dt, scenario);
            for (int i = 1; i < nx - 1; ++i)
            {
                for (int j = 1; j < ny - 1; ++j)
                {
                    for (int k = 1; k < nz - 1; ++k)
                    {
                        const int n            = (i * ny + j) * nz + k;
                        fields[0].data()[n]    = 253.15f + 8.0f * i;
                        fields[1].data()[n]    = 60000.0f + 10000.0f * j;
                        fields[2].data()[n]    = k == 1 ? 0.75f : 1.125f;
                        const float saturation = static_cast<float>(
                            tmv::thompson::liquid_saturation_mixing_ratio(fields[1].data()[n], fields[0].data()[n]));
                        fields[3].data()[n]  = scenario == 0   ? saturation * 1.01f
                                               : scenario == 1 ? saturation * 0.5f
                                               : scenario == 2 ? saturation
                                                               : std::nextafter(saturation, 0.0f);
                        fields[4].data()[n]  = scenario == 4 ? 0.0f : scenario == 5 ? 1e-13f : 0.001f * j;
                        fields[5].data()[n]  = scenario == 0 ? 0.0f : scenario == 6 ? 5e-9f : 0.0001f * j;
                        fields[6].data()[n]  = 1e7f * i;
                        fields[7].data()[n]  = scenario == 0 ? 0.0f : scenario == 7 ? 20.0f : 2e4f;
                        fields[8].data()[n]  = 1e8f;
                        fields[9].data()[n]  = scenario == 3 ? -1e-4f : 0.0f;
                        fields[10].data()[n] = scenario == 3 ? 1e4f : 0.0f;
                        fields[11].data()[n] = -fields[10].data()[n];
                        if (scenario >= 8)
                        {
                            const int    bin    = (scenario * 3 + i) % 100;
                            const double edge   = static_cast<double>((*d.coefficients)[20000 + 2 * bin]) +
                                                  (*d.coefficients)[20001 + 2 * bin];
                            const double slope  = 3.672 / edge;
                            const float  number = static_cast<float>(fields[5].data()[n] * slope * slope * slope /
                                                                     (3.1415926536 * 1000.0));
                            fields[7].data()[n] =
                                std::nextafter(number, k == 1 ? 0.0f : std::numeric_limits<float>::infinity());
                            fields[6].data()[n] =
                                static_cast<float>(1e9 / ((scenario % 13 + 1.5) * fields[2].data()[n]));
                            fields[3].data()[n] = saturation * 0.7f;
                            if (scenario >= 36)
                            {
                                fields[4].data()[n] = 0;
                                fields[5].data()[n] = scenario == 36 ? 0 : scenario == 37 ? 1e-13f : 5e-9f;
                                if (scenario >= 40)
                                {
                                    fields[7].data()[n] = scenario == 40 ? 1e7f : 8e11f;
                                }
                            }
                        }
                        if (scenario >= 42)
                        {
                            fields[4].data()[n]  = 2.4559442974851221e-36f;
                            fields[5].data()[n]  = 0;
                            fields[6].data()[n]  = 6.1398607437128051e-37f;
                            fields[7].data()[n]  = 0;
                            fields[9].data()[n]  = fields[4].data()[n];
                            fields[10].data()[n] = stored(fields[6].data()[n], 0, dt);
                            fields[11].data()[n] = -fields[10].data()[n];
                        }
                    }
                }
            }
            if (scenario == 44)
                for (int i = 1; i < nx - 1; ++i)
                    for (int j = 1; j < ny - 1; ++j)
                        for (int k = 1; k < nz - 1; ++k)
                        {
                            const int n          = (i * ny + j) * nz + k;
                            fields[4].data()[n]  = 5.7700958677742165e-6f;
                            fields[5].data()[n]  = 0;
                            fields[6].data()[n]  = 82468808.0f;
                            fields[7].data()[n]  = 0;
                            fields[9].data()[n]  = fields[4].data()[n] - 2.9378e-8f;
                            fields[10].data()[n] = -1000;
                            fields[11].data()[n] = 1000;
                        }
            if (scenario >= 45)
                for (int i = 1; i < nx - 1; ++i)
                    for (int j = 1; j < ny - 1; ++j)
                        for (int k = 1; k < nz - 1; ++k)
                        {
                            const int n          = (i * ny + j) * nz + k;
                            fields[2].data()[n]  = 1;
                            fields[4].data()[n]  = 1e-8f;
                            fields[5].data()[n]  = 0;
                            fields[6].data()[n]  = 8e7f;
                            fields[7].data()[n]  = 0;
                            fields[9].data()[n]  = scenario == 45 ? 0 : scenario == 46 ? 1e-20f : 1e-12f;
                            fields[10].data()[n] = scenario == 48 ? float(-10000 / dt) : 0;
                            fields[11].data()[n] = -fields[10].data()[n];
                            if (scenario == 49)
                            {
                                fields[3].data()[n] = tmv::thermodynamics::saturation_mixing_ratio_water(
                                                          fields[0].data()[n], fields[1].data()[n]) *
                                                      1.01f;
                                fields[4].data()[n] = 1e-3f;
                                fields[5].data()[n] = 1e-3f;
                                fields[7].data()[n] = 2e4f;
                            }
                        }
            std::size_t cells = 0;
            REQUIRE(nimbus::compute::thompson_warm::validate(d, cells));
            REQUIRE(kernels->dispatch_thompson_warm_rain(d));
            for (int n = 0; n < size; ++n)
            {
                const int i = n / (ny * nz), j = (n / nz) % ny, k = n % nz;
                if (i == 0 || i == nx - 1 || j == 0 || j == ny - 1 || k == 0 || k == nz - 1)
                {
                    for (const auto& field : result)
                    {
                        CHECK(field.data()[n] == 0);
                    }
                    continue;
                }
                CAPTURE(n);
                const auto expected = reference(d, n);
                for (int f = 0; f < 6; ++f)
                {
                    const double actual =
                        f < 3 ? static_cast<double>(result[2 * f].data()[n]) + result[2 * f + 1].data()[n]
                              : result[f + 3].data()[n];
                    CAPTURE(f, actual, expected[f]);
                    CHECK(actual == Approx(expected[f]).epsilon(2e-5).margin(f < 3 ? 1e-12 : 0.005));
                }
                const double converted  = static_cast<double>(result[0].data()[n]) + result[1].data()[n];
                const double evaporated = static_cast<double>(result[2].data()[n]) + result[3].data()[n];
                CHECK(converted >= 0);
                CHECK(evaporated >= 0);
                CHECK(converted <= static_cast<double>(d.cloud[n]) -
                                       (double(d.cloud_vapor_change[n]) + d.cloud_vapor_change_low[n]) + 1e-15);
                CHECK(evaporated <= d.rain[n] + converted + 1e-15);
                CHECK(d.cloud_number[n] + dt * result[6].data()[n] >= 0);
                CHECK(d.rain_number[n] + dt * result[7].data()[n] >= 0);
                CHECK(d.water_aerosol[n] + dt * result[8].data()[n] >= 0);
                if (scenario == 44)
                {
                    const double physical_change = expected[6];
                    const double storage =
                        16 * std::numeric_limits<float>::epsilon() *
                        std::max({1.0, std::abs(double(result[6].data()[n])), std::abs(double(result[8].data()[n]))});
                    CHECK(std::abs(double(result[6].data()[n]) + result[8].data()[n] - physical_change) <= storage);
                    CHECK(result[8].data()[n] > d.cloud_aerosol_source[n]);
                }
                if (scenario >= 45)
                {
                    const double additional = dt * (double(result[8].data()[n]) - d.cloud_aerosol_source[n]);
                    CHECK(additional == Approx(scenario == 47 ? 2675.59744 : 0).margin(32));
                    if (scenario == 49)
                        CHECK(converted > 0);
                    else
                        CHECK(converted == 0);
                    CHECK(evaporated == 0);
                }
                CHECK(result[9].data()[n] == 0);
            }
        }
    }
    {
        const auto   saved_fields = fields;
        const double saved_dt     = d.dt;
        for (auto& field : fields)
            field.fill(0);
        fields[0].fill(280);
        fields[1].fill(100000);
        fields[2].fill(1);
        fields[3].fill(float(tmv::thermodynamics::saturation_mixing_ratio_water(280, 100000)));
        fields[4].fill(.001f);
        fields[5].fill(.001f);
        fields[6].fill(1e8f);
        fields[7].fill(1e4f);
        fields[8].fill(1e8f);
        fields[9].fill(.001f);
        cloud_low.fill(-4e-11f);
        d.dt = .25;
        REQUIRE(kernels->dispatch_thompson_warm_rain(d));
        const int    cell          = (ny + 1) * nz + 1;
        const auto   expected_pair = reference(d, cell);
        const double converted     = double(result[0].data()[cell]) + result[1].data()[cell];
        CHECK(converted == 0);
        CHECK(expected_pair[3] < 0);
        CHECK(result[6].data()[cell] == Approx(expected_pair[3]).epsilon(2e-6));
        for (std::size_t f = 0; f < fields.size(); ++f)
            std::copy_n(saved_fields[f].data(), size, fields[f].data());
        cloud_low.fill(0);
        d.dt = saved_dt;
        REQUIRE(kernels->dispatch_thompson_warm_rain(d));
    }
    {
        const auto saved_fields = fields;
        const double saved_dt = d.dt;
        for (auto& field : fields)
            field.fill(0);
        fields[0].fill(280);
        fields[1].fill(100000);
        fields[2].fill(1);
        fields[3].fill(float(tmv::thermodynamics::saturation_mixing_ratio_water(280, 100000)));
        fields[5].fill(1.3387687788486869e-12f);
        d.dt = .25;
        for (float number : {0.0f, 1e-20f, 1e-11f})
        {
            fields[7].fill(number);
            REQUIRE(kernels->dispatch_thompson_warm_rain(d));
            const int cell = (ny + 1) * nz + 1;
            const auto expected = reference(d, cell);
            CHECK(std::isfinite(result[7].data()[cell]));
            CHECK(result[7].data()[cell] == Approx(expected[4]).epsilon(2e-6));
            CHECK(number + d.dt * result[7].data()[cell] > 0);
        }
        for (std::size_t f = 0; f < fields.size(); ++f)
            std::copy_n(saved_fields[f].data(), size, fields[f].data());
        d.dt = saved_dt;
        REQUIRE(kernels->dispatch_thompson_warm_rain(d));
    }
    const auto before    = result;
    const auto unchanged = [&]()
    {
        for (int f = 0; f < 10; ++f)
        {
            CHECK(std::equal(before[f].data(), before[f].data() + size, result[f].data()));
        }
    };
    cloud_low(nx - 2, ny - 2, nz - 2) = nan;
    CHECK_FALSE(kernels->dispatch_thompson_warm_rain(d));
    unchanged();
    cloud_low(nx - 2, ny - 2, nz - 2) = 0;
    d.cloud_number_tendency = result[0].data() + 1;
    CHECK_FALSE(kernels->dispatch_thompson_warm_rain(d));
    unchanged();
    d.cloud_number_tendency = result[6].data();
    d.cloud_to_rain_high    = const_cast<float*>(d.coefficients->data() + 20000);
    CHECK_FALSE(kernels->dispatch_thompson_warm_rain(d));
    unchanged();
    d.cloud_to_rain_high    = result[0].data();
    const auto coefficients = d.coefficients;
    d.coefficients          = nullptr;
    CHECK_FALSE(kernels->dispatch_thompson_warm_rain(d));
    unchanged();
    d.coefficients = coefficients;
    auto invalid   = std::make_shared<std::vector<float>>(*coefficients);
    d.coefficients = invalid;
    (*invalid)[0]  = 1.0f;
    CHECK_FALSE(kernels->dispatch_thompson_warm_rain(d));
    unchanged();
    *invalid          = *coefficients;
    (*invalid)[20002] = 0.0f;
    CHECK_FALSE(kernels->dispatch_thompson_warm_rain(d));
    unchanged();
    d.coefficients                    = coefficients;
    fields[4](nx - 2, ny - 2, nz - 2) = 0.001f;
    fields[6](nx - 2, ny - 2, nz - 2) = 0;
    CHECK_FALSE(kernels->dispatch_thompson_warm_rain(d));
    unchanged();
    fields[2](1, 1, 1) = nan;
    CHECK_FALSE(kernels->dispatch_thompson_warm_rain(d));
    unchanged();
}
} // namespace

TEST_CASE("Metal Thompson warm rain preserves conversion and evaporation", "[native-thompson-warm][metal]")
{
    exercise("metal");
}

TEST_CASE("Vulkan Thompson warm rain preserves conversion and evaporation", "[native-thompson-warm][vulkan]")
{
    exercise("vulkan");
}

TEST_CASE("Thompson evaporation cannot overdraw a normalized rain population", "[native-thompson-warm][cpu]")
{
    for (double dt : {0.01, 0.03, 1.3})
    {
        CAPTURE(dt);
        constexpr int       nx = 3, ny = 3, nz = 3;
        Field3D             p(nx, ny, nz, 100000), theta(nx, ny, nz, 290), rho(nx, ny, nz, 1);
        Field3D             zero(nx, ny, nz, 0), rain(nx, ny, nz, 5e-9f);
        tmv::ThompsonScheme scheme;
        scheme.set_physical_cell_domain(nx, ny, nz, {1, 2, 1, 2, 1, 2});
        scheme.set_vertical_spacing_m(100);
        scheme.set_vertical_velocity_field(zero, false);
        scheme.prepare_prognostic_state(rho, zero, rain, zero, zero, zero, zero);
        auto state               = scheme.portable_state();
        state.fields[1](1, 1, 1) = 1e12f;
        std::string error;
        REQUIRE(scheme.restore_portable_state(state, error));
        std::array<Field3D, 8> t;
        REQUIRE_NOTHROW(scheme.compute_tendencies(p,
                                                  theta,
                                                  rho,
                                                  zero,
                                                  zero,
                                                  rain,
                                                  zero,
                                                  zero,
                                                  zero,
                                                  zero,
                                                  dt,
                                                  t[0],
                                                  t[1],
                                                  t[2],
                                                  t[3],
                                                  t[4],
                                                  t[5],
                                                  t[6],
                                                  t[7]));
        REQUIRE(scheme.validate_prognostic_update(dt, error));
        scheme.commit_prognostic_update(dt);
        const auto result = scheme.portable_state();
        CHECK(result.fields[1](1, 1, 1) >= 0);
        CHECK(result.fields[3](1, 1, 1) >= state.fields[3](1, 1, 1));
        CHECK(t[1](1, 1, 1) == -t[3](1, 1, 1));
        CHECK(t[1](1, 1, 1) > 0);
    }
}

TEST_CASE("Thompson split timestep preserves the host donor bound", "[native-thompson-warm][contract]")
{
    for (double dt : {0.01, 0.03, 0.2, std::nextafter(0.2, 1.0), 1.3, 10.0, 1e-38, 1e37})
    {
        const double split =
            static_cast<float>(dt) + static_cast<double>(nimbus::compute::thompson_cloud::timestep_low(dt));
        CAPTURE(dt, split);
        REQUIRE(split >= dt);
        CHECK(split - dt <= std::max(double(std::numeric_limits<float>::denorm_min()),
                                     2 * std::pow(std::numeric_limits<float>::epsilon(), 2) * dt));
    }
}
