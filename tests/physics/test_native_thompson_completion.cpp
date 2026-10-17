#include "catch2/catch.hpp"
#include "compute/backend_runtime.hpp"
#include "compute/thompson_completion_contract.hpp"
#include "microphysics/base/thermodynamics.hpp"
#include "microphysics/schemes/thompson/completion_tables.hpp"
#include "microphysics/schemes/thompson/collection_processes.hpp"
#include "microphysics/schemes/thompson/cold_processes.hpp"
#include "microphysics/schemes/thompson/thermodynamic_helpers.hpp"
#include "microphysics/schemes/thompson/thompson.hpp"
#include "microphysics/schemes/thompson/moment_initialization.hpp"
#include "microphysics/schemes/thompson/required_packaged_data.hpp"

#include <array>
#include <limits>

namespace
{
void completion(const char* backend)
{
    ComputeBackendRuntimeState runtime;
    runtime.config.backend        = backend;
    runtime.config.allow_fallback = false;
    std::string error;
    const bool  initialized = initialize_compute_backend_runtime(runtime, CoordinateSystem::Cartesian, error);
    INFO(error);
    REQUIRE(initialized);
    auto* kernels = runtime.backend->microphysics_kernels();
    REQUIRE(kernels);
    REQUIRE(kernels->supports(MicrophysicsKernelOp::ThompsonCompletion));
    auto               tables = tmv::thompson::native_completion_tables();
    std::vector<float> input(nimbus::compute::thompson_completion_input_stride * 65, 0), output(18 * 65, -777);
    for (std::size_t n = 0; n < 65; ++n)
    {
        input[n * nimbus::compute::thompson_completion_input_stride]     = 280;
        input[n * nimbus::compute::thompson_completion_input_stride + 1] = 100000;
        input[n * nimbus::compute::thompson_completion_input_stride + 2] = 1;
    }
    ThompsonCompletionDispatch d{input.data(), output.data(), 65, .2, tables};
    REQUIRE(kernels->dispatch_thompson_completion(d));
    const auto expected = output;
    for (std::size_t n = 0; n < 65; ++n)
    {
        for (std::size_t f = 0; f < 13; ++f)
            CHECK(output[n * 18 + f] == 0);
        CHECK(output[n * 18 + 13] == 1);
        for (std::size_t f = 14; f < 17; ++f)
            CHECK(output[n * 18 + f] == -1);
        CHECK(output[n * 18 + 17] == 0);
    }
    for (int repeat = 0; repeat < 2; ++repeat)
    {
        d.tables = repeat == 0 ? tables : nimbus::compute::ThompsonCompletionTables::create(tables->values());
        REQUIRE(d.tables);
        REQUIRE(kernels->dispatch_thompson_completion(d));
        CHECK(output == expected);
    }
    {
        auto        paired = input;
        auto*       row    = paired.data();
        const float low    = 4e-11f;
        row[3] = row[5]         = .001f;
        row[15]                 = -.001f;
        row[49]                 = low;
        row[21]                 = .001f;
        const double latent     = 2.5e6 / 1004.0;
        const double cloud_heat = -latent * (double(row[15]) + low);
        const double rain_heat  = -latent * row[21];
        row[16]                 = float(cloud_heat);
        row[50]                 = float(cloud_heat - row[16]);
        row[23]                 = float(rain_heat);
        row[24]                 = float(rain_heat - row[23]);
        d.input                 = paired.data();
        REQUIRE(kernels->dispatch_thompson_completion(d));
        CHECK(output[1] == Approx(double(low) / d.dt).epsilon(2e-6));
        CHECK(output[0] == Approx(-latent * low / d.dt).epsilon(2e-6));
        const double water_rate = double(output[1]) + output[2] + output[3];
        CHECK(std::abs(water_rate) <= 2 * std::numeric_limits<float>::epsilon() * std::abs(output[3]));
        d.input = input.data();
    }
    auto active_input = input;
    for (std::size_t n = 0; n < 65; ++n)
    {
        auto* row = active_input.data() + n * nimbus::compute::thompson_completion_input_stride;
        row[0]    = 268;
        row[3]    = thermodynamics::saturation_mixing_ratio_water(268, 100000);
        row[4]    = .001f;
        row[5]    = .0003f;
        row[6]    = .00001f;
        row[7]    = .0002f;
        row[8]    = .0001f;
        row[10]   = 2e8f;
        row[11]   = 1e4f;
        row[12]   = 1e5f;
        row[13]   = 2e8f;
        row[14]   = 1e6f;
    }
    d.input  = active_input.data();
    d.tables = tables;
    REQUIRE(kernels->dispatch_thompson_completion(d));
    const auto active_result = output;
    auto       changed       = tables->values();
    std::fill(changed.begin(), changed.begin() + 2 * nimbus::compute::ThompsonCompletionTables::riming, 0);
    d.tables = nimbus::compute::ThompsonCompletionTables::create(std::move(changed));
    REQUIRE(d.tables);
    REQUIRE(kernels->dispatch_thompson_completion(d));
    CHECK(output != active_result);
    d.tables = tables;
    REQUIRE(kernels->dispatch_thompson_completion(d));
    CHECK(output == active_result);
    std::array<float, nimbus::compute::thompson_completion_input_stride> lookup{};
    lookup[0]  = 270;
    lookup[1]  = 100000;
    lookup[2]  = 1;
    lookup[5]  = .0003f;
    lookup[7]  = .0002f;
    lookup[8]  = .0001f;
    lookup[11] = 1e4f;
    d.input    = lookup.data();
    d.cells    = 1;
    REQUIRE(kernels->dispatch_thompson_completion(d));
    const double wet_bulb = tmv::thompson::wet_bulb_temperature(lookup[1], lookup[0], lookup[3]);
    CHECK(output[14] == -1);
    CHECK(output[15] ==
          tmv::thompson::flatten(tmv::thompson::rain_snow_lookup_index(lookup[5], lookup[11], lookup[7], wet_bulb)));
    CHECK(output[16] ==
          tmv::thompson::flatten(tmv::thompson::rain_graupel_lookup_index(lookup[5], lookup[11], lookup[8])));
    lookup[0] = 265;
    lookup[3] = thermodynamics::saturation_mixing_ratio_ice(265, 100000);
    lookup[5] = lookup[7] = lookup[8] = lookup[11] = 0;
    lookup[6]                                      = 1e-5f;
    lookup[12]                                     = 2e5f;
    REQUIRE(kernels->dispatch_thompson_completion(d));
    CHECK(output[14] == tmv::thompson::flatten(tmv::thompson::ice_deposition_lookup_index(
                            {lookup[1], lookup[0], lookup[2], lookup[3], lookup[6], lookup[12], 0, 0})));
    CHECK(output[15] == -1);
    CHECK(output[16] == -1);
    const auto cold_tables = tmv::thompson::require_packaged_cold_process_tables();
    for (float population : {999990.984f, 1000008.778f})
    {
        lookup[6]  = .00021191394f;
        lookup[12] = population;
        REQUIRE(kernels->dispatch_thompson_completion(d));
        const auto conversion = tmv::thompson::ice_to_snow_rates(*cold_tables, {lookup[6], lookup[12]}, d.dt);
        CHECK(output[4] == Approx(conversion.ice_kg_m3_s).epsilon(2e-6));
        CHECK(output[5] == Approx(conversion.snow_kg_m3_s).epsilon(2e-6));
    }
    d.cells = 65;
    d.input = input.data();
    REQUIRE(kernels->dispatch_thompson_completion(d));
    CHECK(output == expected);
    const auto rejected = [&]()
    {
        CHECK_FALSE(kernels->dispatch_thompson_completion(d));
        CHECK(output == expected);
    };
    for (std::size_t slot : {49, 50})
    {
        const auto last = 64 * nimbus::compute::thompson_completion_input_stride + slot;
        input[last] = std::numeric_limits<float>::quiet_NaN();
        rejected();
        input[last] = 0;
    }
    d.input          = active_input.data();
    active_input[10] = std::numeric_limits<float>::max();
    rejected();
    d.input  = input.data();
    input[0] = std::numeric_limits<float>::quiet_NaN();
    rejected();
    input[0] = 280;
    input[2] = 0;
    rejected();
    input[2] = 1;
    d.cells  = 0;
    rejected();
    d.cells = nimbus::compute::thompson_completion_chunk_cells + 1;
    rejected();
    d.cells = 65;
    d.dt    = -1;
    rejected();
    d.dt    = .2;
    d.input = output.data();
    rejected();
    d.input = input.data();
    d.tables.reset();
    rejected();
}

void coupled(const char* backend, bool upper_air = false)
{
    ComputeBackendRuntimeState runtime;
    runtime.config.backend        = backend;
    runtime.config.allow_fallback = false;
    std::string error;
    const bool  initialized = initialize_compute_backend_runtime(runtime, CoordinateSystem::Cartesian, error);
    INFO(error);
    REQUIRE(initialized);
    constexpr int            nx = 20, ny = 20, nz = 30;
    const PhysicalCellDomain domain{1, nx - 1, 1, ny - 1, 1, nz - 1};
    Field3D                  p(nx, ny, nz, 100000), theta(nx, ny, nz, 270), rho(nx, ny, nz, 1.1f), w(nx, ny, nz, .5f);
    std::array<Field3D, 7>   q;
    for (auto& field : q)
        field = Field3D(nx, ny, nz, 0);
    const std::array<float, 14> temperatures{
        220, 235.15f, 235.17f, 240, 250, 260, 265.14f, 268.14f, 268.16f, 270.14f, 273.14f, 273.16f, 280, 300};
    for (int i = 1; i < nx - 1; ++i)
        for (int j = 1; j < ny - 1; ++j)
            for (int k = 1; k < nz - 1; ++k)
            {
                const auto  n   = (i * ny + j) * nz + k;
                const float t   = temperatures[(k - 1) % temperatures.size()];
                theta.data()[n] = t;
                q[0].data()[n] = thermodynamics::saturation_mixing_ratio_water(t, p.data()[n]) * (j % 2 ? 1.01f : .75f);
                const float scale = i % 3 == 0 ? 0 : i % 3 == 1 ? .01f : 1.f;
                q[1].data()[n]    = 1e-3f * scale;
                q[2].data()[n]    = 3e-4f * scale;
                q[3].data()[n]    = 1e-5f * scale;
                q[4].data()[n]    = 2e-4f * scale;
                q[5].data()[n]    = 1e-4f * scale;
                if (upper_air)
                {
                    const float cold = 228.0f + .5f * (k - 1);
                    p.data()[n]      = i % 2 ? 28152.087890625f : 50000.0f;
                    rho.data()[n]    = p.data()[n] / (287.0f * cold);
                    theta.data()[n]  = float(thermodynamics::temperature_to_theta(cold, p.data()[n]));
                    q[0].data()[n] =
                        thermodynamics::saturation_mixing_ratio_water(cold, p.data()[n]) * (j % 2 ? .1f : .5f);
                    q[1].data()[n] = .000847363960929215f;
                    for (int species = 2; species < 7; ++species)
                        q[species].data()[n] = 0;
                }
            }
    for (double dt : (upper_air ? std::array<double, 2>{.25, .5} : std::array<double, 2>{.03, .2}))
    {
        INFO(dt);
        tmv::ThompsonScheme cpu, native;
        for (auto* scheme : {&cpu, &native})
        {
            scheme->set_physical_cell_domain(nx, ny, nz, domain);
            scheme->set_vertical_spacing_m(100);
            scheme->set_vertical_velocity_field(w, false);
            scheme->prepare_prognostic_state(rho, q[1], q[2], q[3], q[4], q[5], q[6]);
        }
        const auto initial = cpu.portable_state();
        native.bind_execution_owners(runtime, nullptr);
        native.set_require_native_dispatch(true);
        std::array<Field3D, 8> reference, actual;
        const auto             compute = [&](tmv::ThompsonScheme& scheme, std::array<Field3D, 8>& result)
        {
            scheme.compute_tendencies(p,
                                      theta,
                                      rho,
                                      q[0],
                                      q[1],
                                      q[2],
                                      q[3],
                                      q[4],
                                      q[5],
                                      q[6],
                                      dt,
                                      result[0],
                                      result[1],
                                      result[2],
                                      result[3],
                                      result[4],
                                      result[5],
                                      result[6],
                                      result[7]);
        };
        REQUIRE_NOTHROW(compute(cpu, reference));
        REQUIRE_NOTHROW(compute(native, actual));
        for (std::size_t f = 0; f < reference.size(); ++f)
            for (std::size_t n = 0; n < reference[f].size(); ++n)
            {
                INFO(f);
                INFO(n);
                CHECK(std::abs(reference[f].data()[n] - actual[f].data()[n]) <=
                      1e-4 * std::abs(reference[f].data()[n]) + 2e-9);
            }
        REQUIRE(cpu.validate_prognostic_update(dt, error));
        REQUIRE(native.validate_prognostic_update(dt, error));
        cpu.commit_prognostic_update(dt);
        native.commit_prognostic_update(dt);
        const auto lhs = cpu.portable_state(), rhs = native.portable_state();
        REQUIRE(lhs.fields.size() == rhs.fields.size());
        for (std::size_t f = 0; f < lhs.fields.size(); ++f)
            for (std::size_t n = 0; n < lhs.fields[f].size(); ++n)
            {
                INFO(f);
                INFO(n);
                CHECK(std::abs(lhs.fields[f].data()[n] - rhs.fields[f].data()[n]) <=
                      1e-4 * std::abs(lhs.fields[f].data()[n]) +
                          (2e-6 + 8 * std::numeric_limits<float>::epsilon() *
                                      std::max(initial.fields[f % 5].data()[n], lhs.fields[f % 5].data()[n])) /
                              (f >= 5 ? dt : 1));
            }
    }
    CHECK(runtime.backend->successful_kernel_dispatch_count(nimbus::compute::KernelId::ThompsonCompletion) == 4);
}
} // namespace

TEST_CASE("Metal Thompson completion validates transactions and table cache ownership",
          "[native-thompson-completion][metal][contract]")
{
    completion("metal");
}
TEST_CASE("Vulkan Thompson completion validates transactions and table cache ownership",
          "[native-thompson-completion][vulkan][contract]")
{
    completion("vulkan");
}
TEST_CASE("Metal Thompson full cold chain matches CPU across chunks and phase boundaries",
          "[native-thompson-completion][metal][coupling]")
{
    coupled("metal");
}
TEST_CASE("Vulkan Thompson full cold chain matches CPU across chunks and phase boundaries",
          "[native-thompson-completion][vulkan][coupling]")
{
    coupled("vulkan");
}

TEST_CASE("Thompson transport excludes exhausted populations within tendency precision",
          "[native-thompson-completion][cpu]")
{
    using tmv::thompson::transport_number_candidate;
    CHECK(transport_number_candidate(47.28021240234375f, -1576.007080078125f, .03) == 0);
    CHECK(transport_number_candidate(47.28021240234375f, -1576.0069580078125f, .03) == 0);
    CHECK(transport_number_candidate(47.28021240234375f, -1576.0067138671875f, .03) > 0);
    CHECK(transport_number_candidate(1e-8f, 0, .03) == 1e-8f);
}

TEST_CASE("Native Thompson activation preserves its aerosol donor through the complete chain",
          "[native-thompson-completion][activation][conservation]")
{
    const char* backend = GENERATE("metal", "vulkan");
    CAPTURE(backend);
    ComputeBackendRuntimeState runtime;
    runtime.config.backend        = backend;
    runtime.config.allow_fallback = false;
    std::string error;
    REQUIRE(initialize_compute_backend_runtime(runtime, CoordinateSystem::Cartesian, error));
    constexpr int          nx = 8, ny = 8, nz = 8;
    Field3D                p(nx, ny, nz, 81549.5078125f), theta(nx, ny, nz, 305.29608154296875f);
    Field3D                rho(nx, ny, nz, .9865851402282715f), velocity(nx, ny, nz, 2.137484312057495f);
    std::array<Field3D, 7> water;
    for (auto& field : water)
    {
        field = Field3D(nx, ny, nz, 0);
    }
    water[0].fill(.013157349079847336f);
    water[1].fill(4.304698542068053e-10f);
    tmv::ThompsonScheme scheme;
    scheme.set_physical_cell_domain(nx, ny, nz, {1, nx - 1, 1, ny - 1, 0, nz});
    scheme.set_vertical_spacing_m(500);
    scheme.set_vertical_velocity_field(velocity, false);
    scheme.prepare_prognostic_state(rho, water[1], water[2], water[3], water[4], water[5], water[6]);
    auto initial = scheme.portable_state();
    initial.fields[0].fill(111067.1015625f);
    initial.fields[3].fill(209394096.0f);
    REQUIRE(scheme.restore_portable_state(initial, error));
    scheme.bind_execution_owners(runtime, nullptr);
    scheme.set_require_native_dispatch(true);
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
                              .25,
                              rates[0],
                              rates[1],
                              rates[2],
                              rates[3],
                              rates[4],
                              rates[5],
                              rates[6],
                              rates[7]);
    const auto   after        = scheme.portable_state();
    const double droplet_rate = after.fields[5](4, 4, 4), aerosol_rate = after.fields[8](4, 4, 4);
    REQUIRE(rates[2](4, 4, 4) > 0);
    REQUIRE(rates[3](4, 4, 4) == 0);
    REQUIRE(droplet_rate > 0);
    CHECK(std::abs(droplet_rate + aerosol_rate) <= 8 * std::numeric_limits<float>::epsilon() * std::abs(droplet_rate));
    REQUIRE(scheme.validate_prognostic_update(.25, error));
    scheme.commit_prognostic_update(.25);
    const auto   final            = scheme.portable_state();
    const double before_particles = double(initial.fields[0](4, 4, 4)) + initial.fields[3](4, 4, 4);
    const double after_particles  = double(final.fields[0](4, 4, 4)) + final.fields[3](4, 4, 4);
    CHECK(std::abs(after_particles - before_particles) <= 8 * std::numeric_limits<float>::epsilon() * before_particles);
    for (const auto kernel : {nimbus::compute::KernelId::ThompsonCloudAdjustment,
                              nimbus::compute::KernelId::ThompsonWarmRain,
                              nimbus::compute::KernelId::ThompsonCompletion})
    {
        CHECK(runtime.backend->successful_kernel_dispatch_count(kernel) > 0);
    }
}

TEST_CASE("Metal Thompson upper-air seeded clouds preserve the CPU process chain",
          "[native-thompson-completion][metal][upper-air]")
{
    coupled("metal", true);
}
TEST_CASE("Vulkan Thompson upper-air seeded clouds preserve the CPU process chain",
          "[native-thompson-completion][vulkan][upper-air]")
{
    coupled("vulkan", true);
}
