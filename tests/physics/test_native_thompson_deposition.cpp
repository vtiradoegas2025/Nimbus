#include "catch2/catch.hpp"
#include "compute/backend_runtime.hpp"
#include "compute/thompson_deposition_contract.hpp"
#include "core/field/field3d.hpp"
#include "microphysics/base/thermodynamics.hpp"
#include "microphysics/schemes/thompson/cold_processes.hpp"
#include "microphysics/schemes/thompson/thompson.hpp"
#include "microphysics/schemes/thompson/native_deposition_tables.hpp"
#include "microphysics/schemes/thompson/required_packaged_data.hpp"

#include <array>
#include <cmath>
#include <limits>

namespace
{
float stored(float initial, double candidate, double dt)
{
    float rate = static_cast<float>((candidate - initial) / dt);
    if (std::fma(dt, static_cast<double>(rate), static_cast<double>(initial)) < 0)
        rate = std::nextafter(rate, std::numeric_limits<float>::infinity());
    return rate;
}
std::array<double, 7>
reference(const ThompsonDepositionDispatch& d, const tmv::thompson::ColdProcessTables& tables, int n)
{
    using namespace tmv::thompson;
    const double             rho = d.density[n], v = static_cast<double>(d.vapor[n]) +
                                                     (double(d.cloud_vapor_change[n]) + d.cloud_vapor_change_low[n]) +
                                                     d.evaporation_high[n] + d.evaporation_low[n];
    const IceDepositionState s{d.pressure[n],
                               d.temperature[n],
                               rho,
                               v,
                               rho * d.ice[n],
                               rho * d.ice_number[n],
                               rho * d.snow[n],
                               rho * d.graupel[n]};
    const auto               r   = ice_deposition_rates(tables, s, d.dt);
    const double             sat = tmv::thermodynamics::saturation_mixing_ratio_ice(d.temperature[n], d.pressure[n]);
    double                   ss  = v / sat - 1;
    if (std::abs(ss) < 1e-9)
        ss = 0;
    const double index = d.temperature[n] < 273.15 && rho * d.ice[n] > 1e-12 && ss >= 0
                             ? static_cast<double>(flatten(ice_deposition_lookup_index(s)))
                             : -1;
    return {d.dt * r.ice_kg_m3_s / rho,
            d.dt * r.snow_kg_m3_s / rho,
            d.dt * r.graupel_kg_m3_s / rho,
            d.dt * r.temperature_k_s,
            stored(d.ice_number[n], std::max(0.0, d.ice_number[n] + d.dt * r.ice_number_m3_s / rho), d.dt),
            r.snow_vapor_kg_m3_s,
            index};
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
    REQUIRE(kernels->supports(MicrophysicsKernelOp::ThompsonDeposition));
    constexpr int           nx = 8, ny = 5, nz = 4, size = nx * ny * nz;
    const float             nan = std::numeric_limits<float>::quiet_NaN();
    std::array<Field3D, 11> fields;
    std::array<Field3D, 13> result;
    for (auto& f : fields)
        f = Field3D(nx, ny, nz, nan);
    for (auto& f : result)
        f = Field3D(nx, ny, nz, 123);
    Field3D                    cloud_low(nx, ny, nz, 0);
    ThompsonDepositionDispatch d;
    d.temperature             = fields[0].data();
    d.pressure                = fields[1].data();
    d.density                 = fields[2].data();
    d.vapor                   = fields[3].data();
    d.ice                     = fields[4].data();
    d.snow                    = fields[5].data();
    d.graupel                 = fields[6].data();
    d.ice_number              = fields[7].data();
    d.cloud_vapor_change      = fields[8].data();
    d.cloud_vapor_change_low  = cloud_low.data();
    d.evaporation_high        = fields[9].data();
    d.evaporation_low         = fields[10].data();
    d.tables                  = tmv::thompson::native_deposition_tables();
    d.ice_change_high         = result[0].data();
    d.ice_change_low          = result[1].data();
    d.snow_change_high        = result[2].data();
    d.snow_change_low         = result[3].data();
    d.graupel_change_high     = result[4].data();
    d.graupel_change_low      = result[5].data();
    d.temperature_change_high = result[6].data();
    d.temperature_change_low  = result[7].data();
    d.ice_number_tendency     = result[8].data();
    d.snow_vapor_rate_high    = result[9].data();
    d.snow_vapor_rate_low     = result[10].data();
    d.lookup_index            = result[11].data();
    d.status                  = result[12].data();
    d.nx                      = nx;
    d.ny                      = ny;
    d.nz                      = nz;
    d.i_begin                 = 1;
    d.i_end                   = nx - 1;
    d.j_begin                 = 1;
    d.j_end                   = ny - 1;
    d.k_begin                 = 1;
    d.k_end                   = nz - 1;
    for (double dt : {.01, .03, .2, 10.0, 1e6})
        for (int scenario = 0; scenario < 10; ++scenario)
        {
            d.dt = dt;
            CAPTURE(backend, dt, scenario);
            for (int i = 1; i < nx - 1; ++i)
                for (int j = 1; j < ny - 1; ++j)
                    for (int k = 1; k < nz - 1; ++k)
                    {
                        const int n         = (i * ny + j) * nz + k;
                        fields[0].data()[n] = i == 1   ? 273.15f
                                              : i == 2 ? std::nextafter(273.15f, 300.f)
                                                       : 230.f + 8.f * (i - 3);
                        fields[1].data()[n] = 60000.f + 10000.f * j;
                        fields[2].data()[n] = k == 1 ? .75f : 1.125f;
                        const float sat     = static_cast<float>(
                            tmv::thermodynamics::saturation_mixing_ratio_ice(fields[0].data()[n], fields[1].data()[n]));
                        fields[3].data()[n]  = scenario == 0   ? sat * 1.1f
                                               : scenario == 1 ? sat * .7f
                                               : scenario == 2 ? sat
                                               : scenario == 3 ? std::nextafter(sat, 0.f)
                                               : scenario == 4 ? std::nextafter(sat, 1.f)
                                                               : 0.f;
                        fields[4].data()[n]  = scenario == 7 ? 0 : scenario == 8 ? 1e-13f : 1e-5f * j;
                        fields[5].data()[n]  = scenario >= 5 ? 0 : 1e-4f * j;
                        fields[6].data()[n]  = scenario >= 5 ? 0 : 1e-4f * j;
                        fields[7].data()[n]  = scenario == 5 ? 0 : scenario == 6 ? 1 : scenario == 9 ? 1e10f : 1e4f * j;
                        fields[8].data()[n]  = scenario == 4 ? 1e-6f : 0;
                        fields[9].data()[n]  = scenario == 4 ? 2e-7f : 0;
                        fields[10].data()[n] = 0;
                    }
            REQUIRE(kernels->dispatch_thompson_deposition(d));
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
                for (int f = 0; f < 7; ++f)
                {
                    const double actual =
                        f < 4    ? static_cast<double>(result[2 * f].data()[n]) + result[2 * f + 1].data()[n]
                        : f == 4 ? result[8].data()[n]
                        : f == 5 ? static_cast<double>(result[9].data()[n]) + result[10].data()[n]
                                 : result[11].data()[n];
                    CAPTURE(f, actual, expected[f]);
                    if (f == 6)
                        CHECK(actual == expected[f]);
                    else
                        CHECK(actual == Approx(expected[f])
                                            .epsilon(3e-5)
                                            .margin(f == 4   ? .005
                                                    : f == 3 ? 1e-9
                                                             : 1e-14));
                }
                double deposited = 0;
                for (int f = 0; f < 3; ++f)
                {
                    const double change = static_cast<double>(result[2 * f].data()[n]) + result[2 * f + 1].data()[n];
                    deposited += change;
                    CHECK(change >= -fields[4 + f].data()[n] * (1 + 2e-12));
                }
                const double available = static_cast<double>(d.vapor[n]) +
                                         (double(d.cloud_vapor_change[n]) + d.cloud_vapor_change_low[n]) +
                                         d.evaporation_high[n] + d.evaporation_low[n];
                CHECK(deposited <= available + 1e-13);
                CHECK(d.ice_number[n] + dt * result[8].data()[n] >= 0);
                const double heat = static_cast<double>(result[6].data()[n]) + result[7].data()[n];
                CHECK(heat == Approx(2.834e6 / 1004 * deposited).epsilon(1e-6).margin(1e-10));
                CHECK(result[12].data()[n] == 0);
            }
        }
    {
        const auto   saved_fields = fields;
        const double saved_dt     = d.dt;
        for (auto& field : fields)
            field.fill(0);
        fields[0].fill(258.15f);
        fields[1].fill(80000);
        fields[2].fill(1);
        fields[4].fill(1e-4f);
        fields[7].fill(1e5f);
        const float saturated = float(tmv::thermodynamics::saturation_mixing_ratio_ice(258.15f, 80000));
        fields[8].fill(saturated);
        cloud_low.fill(.5f * (std::nextafter(saturated, 1.0f) - saturated));
        d.dt = .25;
        REQUIRE(kernels->dispatch_thompson_deposition(d));
        const int    cell          = (ny + 1) * nz + 1;
        const auto   expected_pair = reference(d, *tables, cell);
        const double deposited     = double(result[0].data()[cell]) + result[1].data()[cell];
        CHECK(expected_pair[0] > 0);
        CHECK(deposited == Approx(expected_pair[0]).epsilon(1e-3).margin(1e-24));
        for (std::size_t f = 0; f < fields.size(); ++f)
            std::copy_n(saved_fields[f].data(), size, fields[f].data());
        cloud_low.fill(0);
        d.dt = saved_dt;
        REQUIRE(kernels->dispatch_thompson_deposition(d));
    }
    const auto before    = result;
    const auto unchanged = [&]()
    {
        for (int f = 0; f < 13; ++f)
            CHECK(std::equal(before[f].data(), before[f].data() + size, result[f].data()));
    };
    cloud_low(nx - 2, ny - 2, nz - 2) = nan;
    CHECK_FALSE(kernels->dispatch_thompson_deposition(d));
    unchanged();
    cloud_low(nx - 2, ny - 2, nz - 2) = 0;
    d.status = result[0].data() + 1;
    CHECK_FALSE(kernels->dispatch_thompson_deposition(d));
    unchanged();
    d.status          = result[12].data();
    d.ice_change_high = const_cast<float*>(d.tables->values().data() + 5000);
    CHECK_FALSE(kernels->dispatch_thompson_deposition(d));
    unchanged();
    d.ice_change_high                 = result[0].data();
    fields[2](nx - 2, ny - 2, nz - 2) = std::numeric_limits<float>::max();
    CHECK_FALSE(kernels->dispatch_thompson_deposition(d));
    unchanged();
    fields[0](1, 1, 1) = nan;
    CHECK_FALSE(kernels->dispatch_thompson_deposition(d));
    unchanged();
}
} // namespace
TEST_CASE("Thompson deposition tables reject invalid and unrepresentable fractions",
          "[native-thompson-deposition][cpu]")
{
    using Tables      = nimbus::compute::ThompsonDepositionTables;
    const auto tables = tmv::thompson::native_deposition_tables();
    REQUIRE(tables);
    CHECK(tables->values().size() * sizeof(float) == 29120);
    std::vector<double> values(Tables::fractions, .5);
    REQUIRE(Tables::create(values));
    values.back() = 2;
    CHECK_FALSE(Tables::create(values));
    values.back() = 1e-100;
    CHECK_FALSE(Tables::create(values));
    values.back() = std::numeric_limits<double>::quiet_NaN();
    CHECK_FALSE(Tables::create(values));
}
TEST_CASE("Metal Thompson deposition preserves the CPU process rates", "[native-thompson-deposition][metal]")
{
    exercise("metal");
}
TEST_CASE("Vulkan Thompson deposition preserves the CPU process rates", "[native-thompson-deposition][vulkan]")
{
    exercise("vulkan");
}

TEST_CASE("Thompson small ice populations survive a complete sublimating step without overdraft",
          "[native-thompson-deposition][cpu][coupling]")
{
    for (double dt : {.03, .2})
        for (float number : {0.f, 1.f})
        {
            CAPTURE(dt, number);
            constexpr int nx = 3, ny = 3, nz = 4;
            Field3D       p(nx, ny, nz, 100000), theta(nx, ny, nz, 258.15f), rho(nx, ny, nz, 1), zero(nx, ny, nz, 0),
                ice(nx, ny, nz, 1e-4f);
            tmv::ThompsonScheme scheme;
            scheme.set_physical_cell_domain(nx, ny, nz, {1, 2, 1, 2, 1, 3});
            scheme.set_vertical_spacing_m(100);
            scheme.set_vertical_velocity_field(zero, false);
            scheme.prepare_prognostic_state(rho, zero, zero, ice, zero, zero, zero);
            auto state               = scheme.portable_state();
            state.fields[2](1, 1, 1) = number;
            state.fields[2](1, 1, 2) = number;
            std::string error;
            REQUIRE(scheme.restore_portable_state(state, error));
            std::array<Field3D, 8> t;
            REQUIRE_NOTHROW(scheme.compute_tendencies(p,
                                                      theta,
                                                      rho,
                                                      zero,
                                                      zero,
                                                      zero,
                                                      ice,
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
            for (int k = 1; k < 3; ++k)
            {
                CHECK(result.fields[2](1, 1, k) >= 0);
                CHECK(t[1](1, 1, k) > 0);
                CHECK(t[4](1, 1, k) < 0);
            }
        }
}
