#include "catch2/catch.hpp"
#include "compute/backend_runtime.hpp"
#include "compute/conservative_remap_contract.hpp"
#include "terrain/native_conservative_remap.hpp"
#include "terrain/conservative_transport.hpp"
#include <omp.h>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <vector>

namespace
{
constexpr int nx = 7, ny = 6, nz = 4, count = nx * ny * nz;
struct Fields
{
    std::array<std::vector<float>, 16> input;
    std::array<std::vector<float>, 10> output;
    ConservativeRemapDispatch          dispatch;
    Fields()
    {
        dispatch.nx = nx;
        dispatch.ny = ny;
        dispatch.nz = nz;
        for (int f = 0; f < 16; ++f)
        {
            input[f].assign(count, 0);
            dispatch.input[f] = input[f].data();
        }
        for (int f = 0; f < 10; ++f)
        {
            output[f].assign(count, 17);
            dispatch.output[f] = output[f].data();
        }
    }
    void pair(int f, int n, double value)
    {
        input[f][n]     = static_cast<float>(value);
        input[f + 1][n] = static_cast<float>(value - input[f][n]);
    }
    long double in(int f, int n) const
    {
        return static_cast<long double>(input[f][n]) + input[f + 1][n];
    }
    long double out(int f, int n) const
    {
        return static_cast<long double>(output[f][n]) + output[f + 1][n];
    }
    void initialize(int sx, int sy, bool signed_value)
    {
        for (int i = 0; i < nx; ++i)
            for (int j = 0; j < ny; ++j)
                for (int k = 0; k < nz; ++k)
                {
                    const int n = (i * ny + j) * nz + k;
                    pair(0, n, (signed_value ? -1.2 : 2.4) + .35 * i - .17 * j + .07 * k);
                    input[2][n] = 1.3f - .07f * k + .03f * i;
                    input[3][n] = input[2][n] + .013f;
                    pair(4, n, 1000000.0 + 12345.6789 * i + 7654.321 * j + 3456.789 * k);
                    pair(6, n, double(in(4, n)) + 913.123);
                    pair(8, n, sx * (31000.123 + 45.2 * i + 33.1 * j));
                    pair(10, n, sx * (31000.123 + 45.2 * (i + 1) + 33.1 * j));
                    pair(12, n, sy * (23000.987 + 67.3 * i + 27.2 * j));
                    pair(14, n, sy * (23000.987 + 67.3 * i + 27.2 * (j + 1)));
                }
    }
};

void geometry_reference(TerrainComputeKernels& kernels)
{
    using namespace tmv;
    using namespace tmv::terrain_native;
    for (const char* scheme : {"bell", "schar"})
        for (int sx : {-1, 1})
            for (int sy : {-1, 1})
            {
                CAPTURE(scheme, sx, sy);
                TerrainConfig config;
                config.feature_center_earth_x_m = 750;
                config.feature_center_earth_y_m = 500;
                config.scheme_id                = scheme;
                config.horizontal_placement_id  = "earth_fixed";
                config.coord_id                 = "smoothed";
                config.ztop                     = 2000;
                config.smoothing_decay          = .1;
                config.bell.h0                  = 150;
                config.bell.a                   = 900;
                config.bell.axisymmetric        = true;
                config.schar.H                  = 120;
                config.schar.a                  = 900;
                config.schar.ell                = 1200;
                EarthFixedTerrainBuildRequest request;
                request.nr                     = nx;
                request.nth                    = ny;
                request.nz                     = nz;
                request.dx_m                   = 300;
                request.dy_m                   = 250;
                request.grid_origin_earth_m    = {-300, -250};
                request.feature_center_earth_m = {750, 500};
                request.sampling               = EarthFixedTerrainSampling::CellAverageBilinear;
                auto    geometry               = build_earth_fixed_volume_geometry(config, request);
                Field3D density(nx, ny, nz, 1), scalar(nx, ny, nz, .005f), number(nx, ny, nz, 2e5f);
                Field3D varying_density(nx, ny, nz), cpu_density;
                for (int i = 0; i < nx; ++i)
                    for (int j = 0; j < ny; ++j)
                        for (int k = 0; k < nz; ++k)
                            varying_density(i, j, k) = 1.2f - .1f * k + .01f * i - .005f * j;
                std::vector<double> energy(count);
                for (int n = 0; n < count; ++n)
                    energy[n] = 3e5 + 17.23456789 * n;
                for (int step = 0; step < 3; ++step)
                {
                    const domain::HorizontalPoint end{request.grid_origin_earth_m.x_m + sx * 13.25,
                                                      request.grid_origin_earth_m.y_m + sy * 17.75};
                    auto                          interval =
                        build_earth_fixed_volume_geometry_interval(config, request, end, .5, &geometry.metrics);
                    const auto&                g0    = geometry.metrics;
                    const auto&                g1    = interval.end.metrics;
                    const auto&                sweep = interval.swept_volumes;
                    TerrainContravariantFluxes flux;
                    flux.x.resize(nx + 1, ny, nz, 0);
                    flux.y.resize(nx, ny + 1, nz, 0);
                    flux.z.resize(nx, ny, nz + 1, 0);
                    for (int i = 1; i < nx; ++i)
                        for (int j = 1; j < ny - 1; ++j)
                            for (int k = 0; k < nz; ++k)
                                flux.x(i, j, k) =
                                    -sweep.x(i, j, k) * varying_density(sweep.x(i, j, k) >= 0 ? i : i - 1, j, k);
                    for (int i = 1; i < nx - 1; ++i)
                        for (int j = 1; j < ny; ++j)
                            for (int k = 0; k < nz; ++k)
                                flux.y(i, j, k) =
                                    -sweep.y(i, j, k) * varying_density(i, sweep.y(i, j, k) >= 0 ? j : j - 1, k);
                    const auto expected_mass =
                        update_moving_mesh_inventory(g0,
                                                     g1,
                                                     flux,
                                                     physical_scalar_domain(nx, ny, nz, StaggerType::CGrid),
                                                     varying_density,
                                                     cpu_density,
                                                     0);
                    const auto native_mass = remap_inventory(kernels,
                                                             varying_density,
                                                             varying_density,
                                                             varying_density,
                                                             varying_density,
                                                             g0,
                                                             g1,
                                                             sweep,
                                                             false,
                                                             RemapLowerBound::Positive);
                    for (int n = 0; n < count; ++n)
                        CHECK(varying_density.data()[n] == Approx(cpu_density.data()[n]).epsilon(1e-7));
                    CHECK(native_mass.initial_inventory == Approx(expected_mass.initial_inventory).epsilon(3e-12));
                    CHECK(native_mass.boundary_outflow ==
                          Approx(expected_mass.boundary_outflow).margin(3e-12 * expected_mass.initial_inventory));
                    CHECK(native_mass.final_inventory == Approx(expected_mass.final_inventory).epsilon(1e-7));
                    CHECK(std::abs(native_mass.final_inventory - native_mass.initial_inventory +
                                   native_mass.boundary_outflow - native_mass.storage_rounding) <
                          1e-11 * native_mass.initial_inventory);
                    Field3D candidate;
                    auto    mass = remap_inventory(
                        kernels, density, candidate, density, density, g0, g1, sweep, false, RemapLowerBound::Positive);
                    for (std::size_t n = 0; n < candidate.size(); ++n)
                        CHECK(candidate.data()[n] == 1);
                    CHECK(std::abs(mass.final_inventory - mass.initial_inventory + mass.boundary_outflow -
                                   mass.storage_rounding) < 1e-11 * mass.initial_inventory);
                    Field3D ratio, concentration;
                    auto    water = remap_inventory(
                        kernels, scalar, ratio, density, candidate, g0, g1, sweep, true, RemapLowerBound::Nonnegative);
                    remap_inventory(kernels,
                                    number,
                                    concentration,
                                    density,
                                    candidate,
                                    g0,
                                    g1,
                                    sweep,
                                    false,
                                    RemapLowerBound::Positive);
                    for (int n = 0; n < count; ++n)
                    {
                        CHECK(ratio.data()[n] == scalar.data()[n]);
                        CHECK(concentration.data()[n] == number.data()[n]);
                    }
                    CHECK(std::abs(water.final_inventory - water.initial_inventory + water.boundary_outflow -
                                   water.storage_rounding) < 1e-11 * water.initial_inventory);
                    const auto initial_energy = energy;
                    auto report = remap_inventory(kernels, energy, energy, g0, g1, sweep, RemapLowerBound::Positive);
                    long double reference_out = 0;
                    for (int i = 0; i < nx; ++i)
                        for (int j = 0; j < ny; ++j)
                            for (int k = 0; k < nz; ++k)
                            {
                                const int n = (i * ny + j) * nz + k;
                                if (i == 0 || i == nx - 1 || j == 0 || j == ny - 1)
                                {
                                    CHECK(energy[n] == initial_energy[n]);
                                    continue;
                                }
                                long double inventory = initial_energy[n] * (g0.J(i, j, k) * g0.dx * g0.dy * g0.dzeta);
                                for (int side : {-1, 1})
                                {
                                    const double      fx = -sweep.x(i + (side > 0), j, k),
                                                      fy = -sweep.y(i, j + (side > 0), k);
                                    const int         ix = i + (side > 0 ? (fx < 0) : -(fx > 0)),
                                                      jy = j + (side > 0 ? (fy < 0) : -(fy > 0));
                                    const long double x =
                                        side * static_cast<long double>(fx) * initial_energy[(ix * ny + j) * nz + k];
                                    const long double y =
                                        side * static_cast<long double>(fy) * initial_energy[(i * ny + jy) * nz + k];
                                    inventory -= x + y;
                                    if ((side < 0 && i == 1) || (side > 0 && i == nx - 2))
                                        reference_out += x;
                                    if ((side < 0 && j == 1) || (side > 0 && j == ny - 2))
                                        reference_out += y;
                                }
                                const double expected = inventory / (g1.J(i, j, k) * g1.dx * g1.dy * g1.dzeta);
                                CHECK(energy[n] == Approx(expected).epsilon(3e-12));
                            }
                    CHECK(report.boundary_outflow ==
                          Approx(double(reference_out)).margin(3e-12 * report.initial_inventory));
                    CHECK(std::abs(report.final_inventory - report.initial_inventory + report.boundary_outflow -
                                   report.storage_rounding) < 1e-11 * report.initial_inventory);
                    std::vector<double> repeated;
                    const int           threads = omp_get_max_threads();
                    omp_set_num_threads(4);
                    const auto repeated_report =
                        remap_inventory(kernels, initial_energy, repeated, g0, g1, sweep, RemapLowerBound::Positive);
                    omp_set_num_threads(threads);
                    CHECK(repeated == energy);
                    CHECK(repeated_report.final_inventory == report.final_inventory);
                    auto malformed       = sweep;
                    malformed.z(1, 1, 1) = 1;
                    REQUIRE_THROWS(remap_inventory(kernels, initial_energy, repeated, g0, g1, malformed));
                    CHECK(repeated == energy);
                    malformed = sweep;
                    malformed.x.values.pop_back();
                    REQUIRE_THROWS(
                        remap_inventory(kernels, scalar, ratio, density, candidate, g0, g1, malformed, true));
                    CHECK(std::memcmp(ratio.data(), scalar.data(), count * sizeof(float)) == 0);
                    density                     = std::move(candidate);
                    scalar                      = std::move(ratio);
                    number                      = std::move(concentration);
                    geometry                    = std::move(interval.end);
                    request.grid_origin_earth_m = end;
                }
            }
}

void exercise(const char* backend)
{
    ComputeBackendRuntimeState runtime;
    runtime.config.backend        = backend;
    runtime.config.allow_fallback = false;
    std::string error;
    REQUIRE(initialize_compute_backend_runtime(runtime, CoordinateSystem::Cartesian, error));
    INFO(error);
    REQUIRE_FALSE(runtime.fallback_active);
    auto* kernels = runtime.backend->terrain_kernels();
    REQUIRE(kernels);
    REQUIRE(kernels->supports_conservative_remap());
    std::uint64_t calls = 0;
    for (int sx : {-1, 0, 1})
        for (int sy : {-1, 0, 1})
            for (bool mixing : {false, true})
                for (bool store_float : {false, true})
                    for (bool signed_value : {false, true})
                    {
                        CAPTURE(backend, sx, sy, mixing, store_float, signed_value);
                        Fields f;
                        f.initialize(sx, sy, signed_value);
                        f.dispatch.mixing_ratio = mixing;
                        f.dispatch.store_float  = store_float;
                        f.dispatch.lower_bound  = signed_value ? 0 : 2;
                        REQUIRE(kernels->dispatch_conservative_remap(f.dispatch));
                        ++calls;
                        long double initial = 0, final = 0, boundary = 0, rounding = 0, scale = 0;
                        const auto  currency = [&](int n)
                        {
                            return f.in(0, n) * (mixing ? f.input[2][n] : 1);
                        };
                        for (int i = 0; i < nx; ++i)
                            for (int j = 0; j < ny; ++j)
                                for (int k = 0; k < nz; ++k)
                                {
                                    const int n = (i * ny + j) * nz + k;
                                    if (i == 0 || i == nx - 1 || j == 0 || j == ny - 1)
                                    {
                                        CHECK(f.output[0][n] == f.input[0][n]);
                                        CHECK(f.output[1][n] == f.input[1][n]);
                                        for (int a = 2; a < 10; ++a)
                                            CHECK(f.output[a][n] == 0);
                                        continue;
                                    }
                                    const long double start = currency(n) * f.in(4, n);
                                    long double inventory = start, exported = 0, arithmetic_scale = std::abs(start);
                                    for (int axis = 0; axis < 2; ++axis)
                                        for (int side : {-1, 1})
                                        {
                                            const long double flux   = -f.in(8 + 4 * axis + (side > 0 ? 2 : 0), n);
                                            const int         offset = axis == 0 ? ny * nz : nz;
                                            const int donor = n + offset * (side > 0 ? (flux < 0) : -(flux > 0));
                                            const long double outward = side * flux * currency(donor);
                                            inventory -= outward;
                                            arithmetic_scale += std::abs(outward);
                                            const int c = axis == 0 ? i : j, extent = axis == 0 ? nx : ny;
                                            if ((side < 0 && c == 1) || (side > 0 && c == extent - 2))
                                                exported += outward;
                                        }
                                    const long double denominator = f.in(6, n) * (mixing ? f.input[3][n] : 1);
                                    const long double expected    = inventory / denominator;
                                    const long double actual      = f.out(0, n);
                                    const long double stored      = actual * denominator;
                                    const long double tolerance   = 3e-12L * std::max(1.0L, arithmetic_scale);
                                    CHECK(std::abs(actual - expected) <=
                                          (store_float ? 8e-8L : 3e-12L) * std::max(1.0L, std::abs(expected)));
                                    CHECK(std::abs(f.out(2, n) - start) <= tolerance);
                                    CHECK(std::abs(f.out(4, n) - stored) <= tolerance);
                                    CHECK(std::abs(f.out(6, n) - exported) <= tolerance);
                                    CHECK(std::abs(f.out(8, n) - (stored - inventory)) <= tolerance);
                                    initial += f.out(2, n);
                                    final += f.out(4, n);
                                    boundary += f.out(6, n);
                                    rounding += f.out(8, n);
                                    scale += std::abs(start);
                                }
                        CHECK(std::abs(final - initial + boundary - rounding) <= 3e-12L * scale);
                    }
    Fields f;
    f.initialize(0, 0, false);
    for (int n = 0; n < count; ++n)
    {
        f.input[6][n] = f.input[4][n];
        f.input[7][n] = f.input[5][n];
        f.input[0][n] = n % 3 == 0 ? std::numeric_limits<float>::denorm_min() : (n % 3 == 1 ? -0.0f : 3.25f);
        f.input[1][n] = 0;
    }
    REQUIRE(kernels->dispatch_conservative_remap(f.dispatch));
    ++calls;
    CHECK(std::memcmp(f.input[0].data(), f.output[0].data(), count * sizeof(float)) == 0);
    for (float tiny : {std::numeric_limits<float>::denorm_min(), 0x1p-140f, 0x1p-125f})
    {
        Fields small;
        small.initialize(1, -1, false);
        for (int n = 0; n < count; ++n)
        {
            small.input[0][n] = tiny;
            small.input[1][n] = 0;
            small.pair(4, n, 1000000);
            small.pair(6, n, 1000000);
            small.pair(8, n, 20000);
            small.pair(10, n, 20000);
            small.pair(12, n, -30000);
            small.pair(14, n, -30000);
        }
        REQUIRE(kernels->dispatch_conservative_remap(small.dispatch));
        ++calls;
        const int n = (2 * ny + 2) * nz;
        CAPTURE(tiny);
        CHECK(small.output[0][n] == tiny);
        CHECK(small.out(2, n) == Approx(double(tiny) * 1000000).epsilon(3e-6).margin(1e-45));
    }
    for (int sign : {-1, 1})
        for (bool store_float : {false, true})
        {
            Fields small;
            small.initialize(0, 0, true);
            small.dispatch.store_float = store_float;
            for (int n = 0; n < count; ++n)
            {
                small.input[0][n] = std::ldexp(float(sign * (1 + n % 7)), -149);
                small.input[1][n] = 0;
                small.pair(4, n, 1000000);
                small.pair(6, n, 1000000);
                small.pair(8, n, -500000);
                small.pair(10, n, -500000);
            }
            REQUIRE(kernels->dispatch_conservative_remap(small.dispatch));
            ++calls;
            for (int i = 1; i < nx - 1; ++i)
                for (int j = 1; j < ny - 1; ++j)
                    for (int k = 0; k < nz; ++k)
                    {
                        const int   n = (i * ny + j) * nz + k;
                        const float expected =
                            static_cast<float>(.5 * (double(small.input[0][n]) + small.input[0][n - ny * nz]));
                        CHECK(small.output[0][n] == expected);
                        const double inventory =
                            .5 * (double(small.input[0][n]) + small.input[0][n - ny * nz]) * 1000000;
                        const double stored = double(expected) * 1000000;
                        CHECK(small.out(4, n) == Approx(stored).margin(1e-45));
                        CHECK(small.out(8, n) == Approx(stored - inventory).margin(1e-45));
                    }
        }
    for (auto& field : f.output)
        std::fill(field.begin(), field.end(), 17);
    const auto rejected = [&]()
    {
        CHECK_FALSE(kernels->dispatch_conservative_remap(f.dispatch));
        for (auto& field : f.output)
            for (float v : field)
                CHECK(v == 17);
    };
    f.dispatch.input[0] = nullptr;
    rejected();
    f.dispatch.input[0]  = f.input[0].data();
    f.dispatch.output[9] = nullptr;
    rejected();
    f.dispatch.output[9] = f.output[9].data();
    f.input[1][0]        = 1;
    rejected();
    f.input[1][0] = 0;
    f.dispatch.nx = std::numeric_limits<int>::max();
    rejected();
    f.dispatch.nx        = nx;
    f.dispatch.output[1] = f.dispatch.output[0] + 1;
    rejected();
    f.dispatch.output[1] = f.output[1].data();
    f.dispatch.output[1] = f.input[0].data();
    rejected();
    f.dispatch.output[1] = f.output[1].data();
    f.input[4][0]        = 0;
    f.input[5][0]        = 0;
    rejected();
    f.pair(4, 0, 1000000);
    f.dispatch.mixing_ratio = true;
    f.input[3][1]           = 0;
    rejected();
    f.input[3][1] = 1;
    f.input[0][1] = std::numeric_limits<float>::quiet_NaN();
    rejected();
    f.input[0][1]          = 1;
    f.dispatch.lower_bound = 2;
    rejected();
    f.dispatch.lower_bound = 0;
    f.initialize(1, 1, false);
    f.dispatch.mixing_ratio = false;
    f.dispatch.lower_bound  = 1;
    f.pair(10, (2 * ny + 2) * nz, -1e20);
    rejected();
    ++calls;
    f.dispatch.lower_bound = 0;
    f.pair(8, (2 * ny + 2) * nz, 1e38);
    f.pair(0, (2 * ny + 2) * nz, 1e38);
    rejected();
    ++calls;
    CHECK(runtime.backend->successful_kernel_dispatch_count(nimbus::compute::KernelId::ConservativeRemap) == calls);
    geometry_reference(*kernels);
}
} // namespace
TEST_CASE("Conservative remapping closes inventories atomically on Metal", "[metal][terrain][remap]")
{
    exercise("metal");
}
TEST_CASE("Conservative remapping closes inventories atomically on Vulkan", "[vulkan][terrain][remap]")
{
    exercise("vulkan");
}
