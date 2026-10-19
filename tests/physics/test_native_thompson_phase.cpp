#include "catch2/catch.hpp"
#include "compute/backend_runtime.hpp"
#include "compute/thompson_phase_contract.hpp"
#include "microphysics/schemes/thompson/cloud_phase.hpp"
#include "microphysics/schemes/thompson/moment_initialization.hpp"
#include "microphysics/schemes/thompson/cold_processes.hpp"
#include "microphysics/schemes/thompson/thompson.hpp"

#include <array>
#include <cmath>
#include <limits>

namespace
{
void exercise(const char* backend)
{
    ComputeBackendRuntimeState runtime;
    runtime.config.backend        = backend;
    runtime.config.allow_fallback = false;
    std::string error;
    REQUIRE(initialize_compute_backend_runtime(runtime, CoordinateSystem::Cartesian, error));
    auto* kernels = runtime.backend->microphysics_kernels();
    REQUIRE(kernels);
    REQUIRE(kernels->supports(MicrophysicsKernelOp::ThompsonCloudPhase));
    constexpr int           nx = 8, ny = 4, nz = 4, size = nx * ny * nz;
    const float             nan = std::numeric_limits<float>::quiet_NaN();
    std::array<Field3D, 12> fields;
    std::array<Field3D, 6>  result;
    for (auto& f : fields)
        f = Field3D(nx, ny, nz, nan);
    for (auto& f : result)
        f = Field3D(nx, ny, nz, 123);
    ThompsonCloudPhaseDispatch d;
    d.temperature               = fields[0].data();
    d.vapor                     = fields[1].data();
    d.density                   = fields[2].data();
    d.cloud                     = fields[3].data();
    d.ice                       = fields[4].data();
    d.cloud_number              = fields[5].data();
    d.ice_number                = fields[6].data();
    d.cloud_tendency            = fields[7].data();
    d.ice_tendency              = fields[8].data();
    d.cloud_number_source       = fields[9].data();
    d.ice_number_source         = fields[10].data();
    d.temperature_tendency      = fields[11].data();
    d.next_cloud_tendency       = result[0].data();
    d.next_ice_tendency         = result[1].data();
    d.next_cloud_number_source  = result[2].data();
    d.next_ice_number_source    = result[3].data();
    d.next_temperature_tendency = result[4].data();
    d.status                    = result[5].data();
    d.nx                        = nx;
    d.ny                        = ny;
    d.nz                        = nz;
    d.i_begin                   = 1;
    d.i_end                     = nx - 1;
    d.j_begin                   = 1;
    d.j_end                     = ny - 1;
    d.k_begin                   = 1;
    d.k_end                     = nz - 1;
    for (double dt : {0.01, 0.03, 0.2, 1.3, 10.0})
        for (int scenario = 0; scenario < 11; ++scenario)
        {
            CAPTURE(backend, dt, scenario);
            d.dt = dt;
            for (int i = 1; i < nx - 1; ++i)
                for (int j = 1; j < ny - 1; ++j)
                    for (int k = 1; k < nz - 1; ++k)
                    {
                        const int n          = (i * ny + j) * nz + k;
                        fields[0].data()[n]  = i == 1   ? 273.15f
                                               : i == 2 ? std::nextafter(273.15f, 300.0f)
                                               : i == 3 ? 235.16f
                                               : i == 4 ? std::nextafter(235.16f, 0.0f)
                                               : i == 5 ? 230.0f
                                                        : 290.0f;
                        fields[1].data()[n]  = .004f;
                        fields[2].data()[n]  = j == 1 ? .75f : 1.125f;
                        fields[3].data()[n]  = scenario == 1 ? 0 : scenario == 2 ? 1e-13f : .0002f;
                        fields[4].data()[n]  = scenario == 1 ? 0 : scenario == 2 ? 1e-13f : .0001f;
                        fields[5].data()[n]  = scenario == 3 ? 1e9f : 1e8f;
                        fields[6].data()[n]  = scenario == 3 ? 1e9f : 1e4f;
                        fields[7].data()[n]  = scenario == 4 ? static_cast<float>(-.25 * fields[3].data()[n] / dt) : 0;
                        fields[8].data()[n]  = scenario == 4 ? static_cast<float>(-.5 * fields[4].data()[n] / dt) : 0;
                        fields[9].data()[n]  = scenario == 4 ? static_cast<float>(-.25 * fields[5].data()[n] / dt) : 0;
                        fields[10].data()[n] = scenario == 4 ? static_cast<float>(-.5 * fields[6].data()[n] / dt) : 0;
                        fields[11].data()[n] = .0001f;
                        if (scenario == 5)
                        {
                            fields[0].data()[n] = 269.080810546875f;
                            fields[3].data()[n] = 1.9391205569263548e-05f;
                            fields[7].data()[n] = static_cast<float>(-fields[3].data()[n] / dt);
                        }
                    }
            if (scenario >= 6)
                for (int i = 1; i < nx - 1; ++i)
                    for (int j = 1; j < ny - 1; ++j)
                        for (int k = 1; k < nz - 1; ++k)
                        {
                            const int n          = (i * ny + j) * nz + k;
                            fields[0].data()[n]  = 229.41578674316406f;
                            fields[1].data()[n]  = .00030575221171602607f;
                            fields[2].data()[n]  = .3957296311855316f;
                            fields[3].data()[n]  = 0;
                            fields[4].data()[n]  = .0005474676145240664f;
                            fields[5].data()[n]  = 0;
                            fields[6].data()[n]  = 2204100.75f;
                            fields[7].data()[n]  = scenario == 6 ? 0 : 1e-20f;
                            fields[8].data()[n]  = 0;
                            fields[9].data()[n]  = 0;
                            fields[10].data()[n] = 167086160.0f;
                            fields[11].data()[n] = 0;
                        }
            if (scenario >= 8)
                for (int i = 1; i < nx - 1; ++i)
                    for (int j = 1; j < ny - 1; ++j)
                        for (int k = 1; k < nz - 1; ++k)
                        {
                            const int n          = (i * ny + j) * nz + k;
                            fields[0].data()[n]  = 226.87233f;
                            fields[2].data()[n]  = .3712649f;
                            fields[4].data()[n]  = scenario == 10 ? 0 : .000195f;
                            fields[6].data()[n]  = scenario == 10 ? 0 : 1163579.875f;
                            fields[7].data()[n]  = scenario == 9 ? 1e-20f : 0;
                            fields[9].data()[n]  = 85605760;
                            fields[10].data()[n] = 0;
                        }
            REQUIRE(kernels->dispatch_thompson_cloud_phase(d));
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
                const auto expected = tmv::thompson::cloud_phase_tendencies({fields[0].data()[n],
                                                                             fields[1].data()[n],
                                                                             fields[2].data()[n],
                                                                             fields[3].data()[n],
                                                                             fields[4].data()[n],
                                                                             fields[5].data()[n],
                                                                             fields[6].data()[n],
                                                                             {fields[7].data()[n],
                                                                              fields[8].data()[n],
                                                                              fields[9].data()[n],
                                                                              fields[10].data()[n],
                                                                              fields[11].data()[n]}},
                                                                            dt);
                for (int f = 0; f < 5; ++f)
                {
                    CAPTURE(f);
                    CHECK(result[f].data()[n] == Approx(expected[f])
                                                     .epsilon(2e-6)
                                                     .margin(f < 2   ? 1e-12
                                                             : f < 4 ? .001
                                                                     : 1e-8));
                    if (f < 4)
                        CHECK(static_cast<double>(fields[f + 3].data()[n]) + dt * result[f].data()[n] >= 0);
                }
                if (scenario == 8 || scenario == 9)
                    CHECK(result[2].data()[n] == 0);
                if (scenario == 10)
                {
                    CHECK(result[2].data()[n] == fields[9].data()[n]);
                    CHECK(result[3].data()[n] == 0);
                }
                CHECK(result[5].data()[n] == 0);
                if (fields[0].data()[n] > 273.15 && fields[4].data()[n] + dt * fields[8].data()[n] > 0)
                {
                    CHECK((fields[5].data()[n] + dt * result[2].data()[n]) * fields[2].data()[n] <= 1999e6);
                }
                if (fields[0].data()[n] < 235.16 && fields[3].data()[n] + dt * fields[7].data()[n] > 0)
                {
                    CHECK((fields[6].data()[n] + dt * result[3].data()[n]) * fields[2].data()[n] <= 999e3);
                }
                CHECK(static_cast<double>(result[0].data()[n]) + result[1].data()[n] ==
                      Approx(static_cast<double>(fields[7].data()[n]) + fields[8].data()[n]).margin(2e-9));
            }
            const auto previous = result;
            REQUIRE(kernels->dispatch_thompson_cloud_phase(d));
            for (int f = 0; f < 6; ++f)
                CHECK(std::equal(previous[f].data(), previous[f].data() + size, result[f].data()));
        }
    const auto before    = result;
    const auto unchanged = [&]()
    {
        for (int f = 0; f < 6; ++f)
            CHECK(std::equal(before[f].data(), before[f].data() + size, result[f].data()));
    };
    d.next_cloud_tendency = result[1].data() + 1;
    CHECK_FALSE(kernels->dispatch_thompson_cloud_phase(d));
    unchanged();
    d.next_cloud_tendency             = result[0].data();
    fields[3](nx - 2, ny - 2, nz - 2) = 1e-6f;
    fields[0](nx - 2, ny - 2, nz - 2) = 10;
    CHECK_FALSE(kernels->dispatch_thompson_cloud_phase(d));
    unchanged();
    fields[0](nx - 2, ny - 2, nz - 2) = nan;
    CHECK_FALSE(kernels->dispatch_thompson_cloud_phase(d));
    unchanged();
}
} // namespace

TEST_CASE("Thompson phase completion follows melting and homogeneous freezing limits", "[native-thompson-phase][cpu]")
{
    using namespace tmv::thompson;
    CloudPhaseState s{274, .004f, 1, .0002f, .0001f, 1e8f, 1e4f, {0, 0, 0, 0, 0}};
    const double    dt   = .2;
    const auto      melt = cloud_phase_tendencies(s, dt);
    CHECK(s.ice + dt * melt[1] >= 0);
    CHECK(s.ice + dt * melt[1] < 1e-10);
    CHECK(melt[0] == -melt[1]);
    CHECK(melt[4] == Approx(-334000 / (1004 * (1 + .887 * s.vapor)) * melt[0]).epsilon(1e-7));
    s.temperature      = 235.16f;
    const auto neutral = cloud_phase_tendencies(s, dt);
    CHECK(neutral == s.tendencies);
    s.temperature     = std::nextafter(235.16f, 0.f);
    const auto frozen = cloud_phase_tendencies(s, dt);
    CHECK(s.cloud + dt * frozen[0] >= 0);
    CHECK(s.cloud + dt * frozen[0] < 1e-10);
    CHECK(frozen[0] == -frozen[1]);
    CHECK(frozen[4] > 0);
    s.ice = -1;
    CHECK_THROWS_AS(cloud_phase_tendencies(s, dt), std::invalid_argument);
}

TEST_CASE("Thompson removes cloud ice from a warm column after sedimentation", "[native-thompson-phase][cpu][coupling]")
{
    constexpr int nx = 3, ny = 3, nz = 4;
    Field3D       p(nx, ny, nz, 100000), theta(nx, ny, nz, 275), rho(nx, ny, nz, 1), zero(nx, ny, nz, 0),
        ice(nx, ny, nz, .0001f);
    tmv::ThompsonScheme scheme;
    scheme.set_physical_cell_domain(nx, ny, nz, {1, 2, 1, 2, 1, 3});
    scheme.set_vertical_spacing_m(100);
    scheme.set_vertical_velocity_field(zero, false);
    scheme.prepare_prognostic_state(rho, zero, zero, ice, zero, zero, zero);
    std::array<Field3D, 8> t;
    const double           dt = .2;
    REQUIRE_NOTHROW(scheme.compute_tendencies(
        p, theta, rho, zero, zero, zero, ice, zero, zero, zero, dt, t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7]));
    for (int k = 1; k < 3; ++k)
    {
        CHECK(ice(1, 1, k) + dt * t[4](1, 1, k) >= 0);
        CHECK(ice(1, 1, k) + dt * t[4](1, 1, k) < 1e-10);
        CHECK(t[2](1, 1, k) > 0);
        CHECK(t[0](1, 1, k) < 0);
    }
    std::string error;
    REQUIRE(scheme.validate_prognostic_update(dt, error));
}
TEST_CASE("Metal Thompson cloud phase matches the CPU", "[native-thompson-phase][metal]")
{
    exercise("metal");
}
TEST_CASE("Vulkan Thompson cloud phase matches the CPU", "[native-thompson-phase][vulkan]")
{
    exercise("vulkan");
}

TEST_CASE("Thompson trace liquid freezing uses the pinned homogeneous threshold", "[native-thompson-phase][cpu]")
{
    using namespace tmv::thompson;
    const auto rain = rain_freezing_exchange(RainFreezingCell{}, {235.155, 5e-7, 40, 0});
    CHECK(rain.rain_kg_m3 == -5e-7);
    CHECK(rain.ice_kg_m3 == 5e-7);
    CHECK(rain.rain_number_m3 == -40);
    const auto cloud = cloud_freezing_exchange(CloudFreezingCell{}, {235.155, 5e-7, 100, 0});
    CHECK(cloud.cloud_kg_m3 == -5e-7);
    CHECK(cloud.ice_number_m3 == 100);
    CHECK(rain_freezing_exchange(RainFreezingCell{}, {235.16, 5e-7, 40, 0}).ice_kg_m3 == 0);
    CHECK(cloud_freezing_exchange(CloudFreezingCell{}, {235.16, 5e-7, 100, 0}).ice_kg_m3 == 0);
}

TEST_CASE("Thompson final cloud population represents liquid remaining after other processes",
          "[native-thompson-phase][cpu][coupling]")
{
    tmv::thompson::CloudPhaseState state{240, .0005, .5, 2.9e-6f, 0, 0, 0, {}};
    const auto                     rates = tmv::thompson::cloud_phase_tendencies(state, 1.0 / 6.0);
    CHECK(rates[0] == 0);
    CHECK(rates[1] == 0);
    CHECK(rates[4] == 0);
    CHECK(rates[2] > 0);
    const double number            = rates[2] / 6.0;
    const double mass_per_particle = state.cloud / number;
    CHECK(std::isfinite(mass_per_particle));
    CHECK(mass_per_particle > 0);
    auto continued         = state;
    continued.cloud_number = static_cast<float>(number);
    CHECK(tmv::thompson::cloud_phase_tendencies(continued, 1.0 / 6.0)[2] == 0);
}

TEST_CASE("Thompson ice population bounds do not depend on a roundoff cloud remnant", "[thompson][phase][population]")
{
    tmv::thompson::CloudPhaseState state{229.41578674316406f,
                                         .00030575221171602607f,
                                         .3957296311855316f,
                                         0,
                                         .0005474676145240664f,
                                         0,
                                         2204100.75f,
                                         {0, 0, 0, 167086160.0f, 0}};
    const auto                     clear = tmv::thompson::cloud_phase_tendencies(state, .25);
    state.tendencies[0]                  = 1e-20f;
    const auto remnant                   = tmv::thompson::cloud_phase_tendencies(state, .25);
    CHECK(clear[3] == remnant[3]);
    CHECK(state.density * (state.ice_number + .25 * clear[3]) <= 999e3);
    CHECK(state.density * (state.ice_number + .25 * clear[3]) == Approx(999e3).epsilon(2e-7));
    CHECK(clear[0] == 0);
    CHECK(clear[1] == 0);
    CHECK(clear[4] == 0);
}

TEST_CASE("Thompson cloud population projection is idempotent across shape changes", "[thompson][population]")
{
    for (double mass : {1e-11, 1e-8, 1e-4, .001, .01, .1})
        for (double supplied : {2.0, 1e4, 1e6, 1e8, 1e9})
        {
            const double number   = tmv::thompson::normalize_cloud_number_m3(mass, supplied);
            const double repeated = tmv::thompson::normalize_cloud_number_m3(mass, number);
            CAPTURE(mass, supplied, number, repeated);
            CHECK(repeated == number);
            const int    shape = std::min(15L, std::lround(1e9 / number) + 2L);
            const double ratio = double((shape + 1) * (shape + 2) * (shape + 3));
            const double diameter =
                std::cbrt(mass * std::pow(double(shape + 4), 3) / (number * ratio * (3.1415926536 * 1000 / 6)));
            CHECK(diameter >= 1e-6 * (1 - 2e-12));
            CHECK(diameter <= 100e-6 * (1 + 2e-12));
        }
}

TEST_CASE("Thompson terminal freezing completes particle transfer at an exhausted cloud endpoint", "[thompson][phase][population]")
{
    tmv::thompson::CloudPhaseState state{226.87233f, .0002528f, .3712649f, 0, .000195f, 0, 1163579.875f,
        {0, 0, 85605760, 0, 0}};
    const auto clear = tmv::thompson::cloud_phase_tendencies(state, .5);
    state.tendencies[0]                  = 1e-20f;
    const auto remnant                   = tmv::thompson::cloud_phase_tendencies(state, .5);
    CHECK(clear[2] == 0);
    CHECK(clear[3] == remnant[3]);
    CHECK(clear[0] == 0);
    CHECK(clear[1] == 0);
    CHECK(clear[4] == 0);
    CHECK(state.density * (state.ice_number + .5 * clear[3]) == Approx(999e3).epsilon(2e-7));
    state.tendencies[0] = 0;
    state.temperature   = 260;
    CHECK(tmv::thompson::cloud_phase_tendencies(state, .5)[2] == state.tendencies[2]);
    state.temperature = 226.87233f;
    state.ice         = 0;
    state.ice_number  = 0;
    const auto empty  = tmv::thompson::cloud_phase_tendencies(state, .5);
    CHECK(empty[3] == 0);
    CHECK(empty[2] == state.tendencies[2]);
    state.tendencies[2] = 0;
    CHECK(tmv::thompson::cloud_phase_tendencies(state, .5) == state.tendencies);
}
