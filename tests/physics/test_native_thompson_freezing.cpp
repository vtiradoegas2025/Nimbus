#include "catch2/catch.hpp"
#include "compute/backend_runtime.hpp"
#include "compute/thompson_freezing_contract.hpp"
#include "core/field/field3d.hpp"
#include "microphysics/schemes/thompson/cold_processes.hpp"
#include "microphysics/schemes/thompson/nucleation.hpp"
#include "microphysics/schemes/thompson/native_freezing_tables.hpp"
#include "microphysics/schemes/thompson/required_packaged_data.hpp"

#include <array>
#include <cmath>
#include <limits>

namespace
{
float stored(float initial, double candidate, double dt)
{
    float rate = static_cast<float>((candidate - initial) / dt);
    if (initial + dt * rate < 0)
        rate = std::nextafter(rate, std::numeric_limits<float>::infinity());
    return rate;
}
std::array<double, 9>
reference(const ThompsonFreezingDispatch& d, const tmv::thompson::ColdProcessTables& tables, int n)
{
    using namespace tmv::thompson;
    if (d.temperature[n] >= 273.15)
        return {0, 0, 0, d.cloud_number_source[n], d.rain_number_source[n], 0, 0, -1, -1};
    const double       rho = d.density[n], t = d.temperature[n];
    const double       converted  = static_cast<double>(d.conversion_high[n]) + d.conversion_low[n];
    const double       evaporated = static_cast<double>(d.evaporation_high[n]) + d.evaporation_low[n];
    const double       q = std::max(0.0,
                                    static_cast<double>(d.cloud[n]) -
                                        (double(d.cloud_vapor_change[n]) + d.cloud_vapor_change_low[n]) - converted);
    const double       r      = std::max(0.0, d.rain[n] + converted - evaporated);
    const double       nc     = d.cloud_number[n] + d.dt * d.cloud_number_source[n];
    const double       nr     = d.rain_number[n] + d.dt * d.rain_number_source[n];
    const double       nuclei = demott_ice_number_m3(t - 273.15, rho, rho * d.ice_aerosol[n]);
    RainFreezingState  rain{t, rho * r, rho * nr, nuclei};
    CloudFreezingState cloud{t, rho * q, rho * nc, nuclei};
    const auto         rf = rain_freezing_exchange(tables, rain);
    const auto         cf = cloud_freezing_exchange(tables, cloud);
    const double       ri = rho * r <= 1e-12  ? -1
                            : rho * r <= 1e-6 ? -2
                                              : static_cast<double>(flatten(rain_freezing_lookup_index(rain)));
    const double       ci = rho * q <= 1e-12 ? -1
                            : rho * q <= 1e-6 && t < 235.16
                                ? -2
                                : static_cast<double>(flatten(cloud_freezing_lookup_index(cloud)));
    return {cf.ice_kg_m3 / rho,
            rf.ice_kg_m3 / rho,
            rf.graupel_kg_m3 / rho,
            stored(d.cloud_number[n], std::max(0.0, nc + cf.cloud_number_m3 / rho), d.dt),
            stored(d.rain_number[n], std::max(0.0, nr + rf.rain_number_m3 / rho), d.dt),
            rf.ice_number_m3 / d.dt / rho,
            cf.ice_number_m3 / d.dt / rho,
            ri,
            ci};
}
void exercise(const char* backend)
{
    const auto                 tables = tmv::thompson::require_packaged_cold_process_tables();
    ComputeBackendRuntimeState runtime;
    runtime.config.backend        = backend;
    runtime.config.allow_fallback = false;
    std::string error;
    REQUIRE(initialize_compute_backend_runtime(runtime, CoordinateSystem::Cartesian, error));
    auto* kernels = runtime.backend->microphysics_kernels();
    REQUIRE(kernels);
    REQUIRE(kernels->supports(MicrophysicsKernelOp::ThompsonFreezing));
    constexpr int           nx = 10, ny = 5, nz = 4, size = nx * ny * nz;
    const float             nan = std::numeric_limits<float>::quiet_NaN();
    std::array<Field3D, 14> fields;
    std::array<Field3D, 13> result;
    for (auto& f : fields)
        f = Field3D(nx, ny, nz, nan);
    for (auto& f : result)
        f = Field3D(nx, ny, nz, 123);
    Field3D                  cloud_low(nx, ny, nz, 0);
    ThompsonFreezingDispatch d;
    d.temperature              = fields[0].data();
    d.density                  = fields[1].data();
    d.cloud                    = fields[2].data();
    d.rain                     = fields[3].data();
    d.cloud_number             = fields[4].data();
    d.rain_number              = fields[5].data();
    d.ice_aerosol              = fields[6].data();
    d.cloud_vapor_change       = fields[7].data();
    d.cloud_vapor_change_low   = cloud_low.data();
    d.conversion_high          = fields[8].data();
    d.conversion_low           = fields[9].data();
    d.evaporation_high         = fields[10].data();
    d.evaporation_low          = fields[11].data();
    d.cloud_number_source      = fields[12].data();
    d.rain_number_source       = fields[13].data();
    d.tables                   = tmv::thompson::native_freezing_tables();
    d.cloud_frozen_high        = result[0].data();
    d.cloud_frozen_low         = result[1].data();
    d.rain_ice_high            = result[2].data();
    d.rain_ice_low             = result[3].data();
    d.rain_graupel_high        = result[4].data();
    d.rain_graupel_low         = result[5].data();
    d.next_cloud_number_source = result[6].data();
    d.next_rain_number_source  = result[7].data();
    d.rain_ice_number_source   = result[8].data();
    d.cloud_ice_number_source  = result[9].data();
    d.rain_lookup              = result[10].data();
    d.cloud_lookup             = result[11].data();
    d.status                   = result[12].data();
    d.nx                       = nx;
    d.ny                       = ny;
    d.nz                       = nz;
    d.i_begin                  = 1;
    d.i_end                    = nx - 1;
    d.j_begin                  = 1;
    d.j_end                    = ny - 1;
    d.k_begin                  = 1;
    d.k_end                    = nz - 1;
    for (double dt : {.01, .03, .2, 1.3})
        for (int scenario = 0; scenario < 22; ++scenario)
        {
            d.dt = dt;
            CAPTURE(backend, dt, scenario);
            for (int i = 1; i < nx - 1; ++i)
                for (int j = 1; j < ny - 1; ++j)
                    for (int k = 1; k < nz - 1; ++k)
                    {
                        const int n         = (i * ny + j) * nz + k;
                        fields[0].data()[n] = i == 1   ? 273.15f
                                              : i == 2 ? 235.16f
                                              : i == 3 ? 235.155f
                                              : i == 4 ? 230.f
                                                       : 237.f + 9.f * (i - 5);
                        fields[1].data()[n] = k == 1 ? .75f : 1.125f;
                        fields[2].data()[n] = scenario == 1   ? 5e-7f
                                              : scenario == 2 ? 1e-13f
                                              : scenario == 3 ? 0
                                                              : 1e-4f * j;
                        fields[3].data()[n] = scenario == 1   ? 5e-7f
                                              : scenario == 2 ? 1e-13f
                                              : scenario == 3 ? 0
                                                              : 1e-4f * j;
                        fields[4].data()[n] = static_cast<float>(1e6 * std::pow(2.0, scenario));
                        const double slope  = 3.672 / (.0001 * (i + 1));
                        fields[5].data()[n] =
                            static_cast<float>(fields[3].data()[n] * slope * slope * slope / (3.1415926536 * 1000));
                        fields[6].data()[n]  = static_cast<float>(100 * std::pow(10.0, j));
                        fields[7].data()[n]  = scenario == 4 ? -1e-5f : 0;
                        fields[8].data()[n]  = scenario == 5 ? 3e-5f : 0;
                        fields[9].data()[n]  = 0;
                        fields[10].data()[n] = scenario == 6 ? 1e-5f : 0;
                        fields[11].data()[n] = 0;
                        fields[12].data()[n] = scenario == 7 ? static_cast<float>(-.1 * fields[4].data()[n] / dt) : 0;
                        fields[13].data()[n] = scenario == 7 ? static_cast<float>(-.2 * fields[5].data()[n] / dt) : 0;
                        if (scenario == 12)
                        {
                            fields[4].data()[n]  = 3;
                            fields[5].data()[n]  = 3;
                            fields[12].data()[n] = dt == .03 ? -100 : 0;
                            fields[13].data()[n] = dt == .03 ? -100 : 0;
                        }
                    }
            if (scenario >= 13)
                for (int i = 1; i < nx - 1; ++i)
                    for (int j = 1; j < ny - 1; ++j)
                        for (int k = 1; k < nz - 1; ++k)
                        {
                            const int n         = (i * ny + j) * nz + k;
                            fields[0].data()[n] = 236.04f;
                            fields[1].data()[n] = 1;
                            fields[2].data()[n] = scenario == 13   ? std::nextafter(1e-6f, 0.0f)
                                                  : scenario == 14 ? std::nextafter(1e-6f, 1.0f)
                                                                   : 1e-8f;
                            fields[3].data()[n] = 0;
                            fields[4].data()[n] = 2e6f;
                            fields[5].data()[n] = 0;
                            fields[6].data()[n] = 1e6f;
                            for (int f = 7; f < 14; ++f)
                                fields[f].data()[n] = 0;
                        }
            if (scenario >= 16)
                for (int i = 1; i < nx - 1; ++i)
                    for (int j = 1; j < ny - 1; ++j)
                        for (int k = 1; k < nz - 1; ++k)
                        {
                            const int   n       = (i * ny + j) * nz + k;
                            const float node    = scenario < 18 ? 6e-6f : 1e-5f;
                            fields[0].data()[n] = 237.9857f;
                            fields[2].data()[n] = std::nextafter(node, scenario % 2 ? 1.0f : 0.0f);
                            fields[4].data()[n] = 15787790;
                        }
            if (scenario >= 20)
                for (int i = 1; i < nx - 1; ++i)
                    for (int j = 1; j < ny - 1; ++j)
                        for (int k = 1; k < nz - 1; ++k)
                        {
                            const int n = (i * ny + j) * nz + k;
                            fields[0].data()[n] = 238.22554f;
                            fields[2].data()[n] = 1.3351961e-5f;
                            fields[4].data()[n] = scenario == 20 ? 14810615 : 14810546;
                        }
            REQUIRE(kernels->dispatch_thompson_freezing(d));
            for (int n = 0; n < size; ++n)
            {
                const int i = n / (ny * nz), j = (n / nz) % ny, k = n % nz;
                if (i == 0 || i == nx - 1 || j == 0 || j == ny - 1 || k == 0 || k == nz - 1)
                {
                    for (auto& f : result)
                        CHECK(f.data()[n] == 0);
                    continue;
                }
                CAPTURE(n);
                const auto expected = reference(d, *tables, n);
                for (int f = 0; f < 9; ++f)
                {
                    CAPTURE(f);
                    const double actual =
                        f < 3 ? static_cast<double>(result[2 * f].data()[n]) + result[2 * f + 1].data()[n]
                              : result[f + 3].data()[n];
                    if (f >= 7)
                        CHECK(actual == expected[f]);
                    else
                        CHECK(actual == Approx(expected[f]).epsilon(3e-5).margin(f < 3 ? 1e-14 : .005));
                }
                CHECK(d.cloud_number[n] + dt * result[6].data()[n] >= 0);
                CHECK(d.rain_number[n] + dt * result[7].data()[n] >= 0);
                CHECK(result[12].data()[n] == 0);
                const double frozen_cloud = static_cast<double>(result[0].data()[n]) + result[1].data()[n];
                const double frozen_rain  = static_cast<double>(result[2].data()[n]) + result[3].data()[n] +
                                            result[4].data()[n] + result[5].data()[n];
                CHECK(frozen_cloud >= 0);
                CHECK(frozen_rain >= 0);
                CHECK(frozen_cloud <=
                      d.cloud[n] -
                          static_cast<double>((double(d.cloud_vapor_change[n]) + d.cloud_vapor_change_low[n])) -
                          d.conversion_high[n] - d.conversion_low[n] + 1e-14);
                CHECK(frozen_rain <= d.rain[n] + static_cast<double>(d.conversion_high[n]) + d.conversion_low[n] -
                                         d.evaporation_high[n] - d.evaporation_low[n] + 1e-14);
            }
        }
    {
        const auto   saved_fields = fields;
        const double saved_dt     = d.dt;
        for (auto& field : fields)
            field.fill(0);
        fields[0].fill(230);
        fields[1].fill(1);
        fields[2].fill(.001f);
        fields[4].fill(1e5f);
        fields[6].fill(1e6f);
        fields[7].fill(.001f);
        cloud_low.fill(-4e-11f);
        d.dt = .25;
        REQUIRE(kernels->dispatch_thompson_freezing(d));
        const int    cell          = (ny + 1) * nz + 1;
        const auto   expected_pair = reference(d, *tables, cell);
        const double frozen        = double(result[0].data()[cell]) + result[1].data()[cell];
        CHECK(expected_pair[0] > 0);
        CHECK(frozen == Approx(expected_pair[0]).epsilon(2e-5).margin(1e-24));
        for (std::size_t f = 0; f < fields.size(); ++f)
            std::copy_n(saved_fields[f].data(), size, fields[f].data());
        cloud_low.fill(0);
        d.dt = saved_dt;
        REQUIRE(kernels->dispatch_thompson_freezing(d));
    }
    const auto before    = result;
    const auto unchanged = [&]()
    {
        for (int f = 0; f < 13; ++f)
            CHECK(std::equal(before[f].data(), before[f].data() + size, result[f].data()));
    };
    cloud_low(nx - 2, ny - 2, nz - 2) = nan;
    CHECK_FALSE(kernels->dispatch_thompson_freezing(d));
    unchanged();
    cloud_low(nx - 2, ny - 2, nz - 2) = 0;
    d.status = result[0].data() + 1;
    CHECK_FALSE(kernels->dispatch_thompson_freezing(d));
    unchanged();
    d.status            = result[12].data();
    d.cloud_frozen_high = const_cast<float*>(d.tables->values().data() + 1000000);
    CHECK_FALSE(kernels->dispatch_thompson_freezing(d));
    unchanged();
    d.cloud_frozen_high               = result[0].data();
    fields[1](nx - 2, ny - 2, nz - 2) = std::numeric_limits<float>::max();
    CHECK_FALSE(kernels->dispatch_thompson_freezing(d));
    unchanged();
    fields[0](1, 1, 1) = nan;
    CHECK_FALSE(kernels->dispatch_thompson_freezing(d));
    unchanged();
}
} // namespace
TEST_CASE("Thompson compact freezing tables reject malformed factors", "[native-thompson-freezing][cpu]")
{
    using Tables      = nimbus::compute::ThompsonFreezingTables;
    const auto source = tmv::thompson::native_freezing_tables();
    REQUIRE(source);
    CHECK(source->values().size() * sizeof(float) == 8018648);
    CHECK(source->values().size() * sizeof(float) < tmv::thompson::checked_cold_payload_bytes() / 30);
    auto values                          = source->values();
    values[2 * Tables::rain_probability] = 2;
    CHECK_FALSE(Tables::create(std::move(values)));
    values                                   = source->values();
    values[2 * Tables::rain_probability]     = .5f;
    values[2 * Tables::rain_probability + 1] = .5f;
    CHECK_FALSE(Tables::create(std::move(values)));
    values                                  = source->values();
    values[2 * Tables::rain_intercepts + 2] = 0;
    CHECK_FALSE(Tables::create(std::move(values)));
}
TEST_CASE("Metal Thompson freezing preserves the tabulated exchanges", "[native-thompson-freezing][metal]")
{
    exercise("metal");
}
TEST_CASE("Vulkan Thompson freezing preserves the tabulated exchanges", "[native-thompson-freezing][vulkan]")
{
    exercise("vulkan");
}
