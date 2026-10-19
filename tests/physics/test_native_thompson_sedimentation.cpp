#include "catch2/catch.hpp"
#include "compute/backend_runtime.hpp"
#include "compute/thompson_sedimentation_contract.hpp"
#include "microphysics/schemes/thompson/particle_distributions.hpp"
#include "microphysics/schemes/thompson/thompson.hpp"
#include "compute/kernel_manifest.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace
{
struct Reference
{
    std::vector<double> mass;
    std::vector<double> number;
    std::vector<double> mass_turnover;
    double              surface = 0.0;
};

Reference reference(const ThompsonMomentSedimentationDispatch& d, int column)
{
    const int           levels = d.k_end - d.k_begin;
    Reference           result{std::vector<double>(levels), std::vector<double>(levels), std::vector<double>(levels)};
    std::vector<double> initial_mass(levels), initial_number(levels), vm(levels), vn(levels), dz(levels);
    double              height  = 0.0;
    const int           species = static_cast<int>(d.species);
    for (int k = 0; k < levels; ++k)
    {
        const int    n   = column * d.nz + d.k_begin + k;
        const double rho = d.density[n];
        result.mass[k] = initial_mass[k] = d.mass[n] * rho;
        result.number[k] = initial_number[k] = species < 3 ? d.number[n] * rho : 0.0;
        dz[k]                                = d.thickness[n];
        tmv::thompson::MassNumberFallSpeeds speeds;
        if (species == 0 && height <= 500 && d.vertical_velocity[n] < 0.1)
        {
            speeds = tmv::thompson::cloud_fall_speeds(result.mass[k], result.number[k], rho);
        }
        else if (species == 1)
        {
            speeds = tmv::thompson::rain_fall_speeds(result.mass[k], result.number[k], rho);
        }
        else if (species == 2)
        {
            speeds = tmv::thompson::ice_fall_speeds(result.mass[k], result.number[k], rho);
        }
        else if (species == 3)
        {
            speeds.mass_m_s =
                tmv::thompson::snow_distribution(result.mass[k], d.temperature[n], rho).mass_fall_speed_m_s *
                d.snow_speed_multiplier[n];
        }
        else if (species == 4)
        {
            speeds.mass_m_s = tmv::thompson::graupel_fall_speeds(result.mass[k], rho).mass_m_s;
        }
        vm[k] = speeds.mass_m_s;
        vn[k] = species < 3 ? speeds.number_m_s : 0.0;
        height += dz[k];
    }
    int steps = 1;
    if (species != 0)
    {
        for (int k = levels - 2; k >= 0; --k)
        {
            if (initial_mass[k] <= 1e-12)
            {
                vm[k] = vm[k + 1];
                vn[k] = vn[k + 1];
            }
        }
        for (int k = 0; k < levels; ++k)
        {
            const double speed = species == 1 ? std::max(vm[k], vn[k]) : vm[k];
            if (speed > 1e-3)
            {
                steps = std::max(steps, static_cast<int>(d.dt * speed / dz[k]) + 1);
            }
        }
    }
    const double dt = static_cast<double>(d.dt) / steps;
    for (int step = 0; step < steps; ++step)
    {
        auto next_mass   = result.mass;
        auto next_number = result.number;
        for (int k = 0; k < levels; ++k)
        {
            const double flux        = vm[k] * result.mass[k] * dt;
            const double number_flux = vn[k] * result.number[k] * dt;
            result.mass_turnover[k] += flux / dz[k];
            next_mass[k]   = std::max(0.0, next_mass[k] - flux / dz[k]);
            next_number[k] = std::max(0.0, next_number[k] - number_flux / dz[k]);
            if (k > 0)
            {
                result.mass_turnover[k - 1] += flux / dz[k - 1];
                next_mass[k - 1] += flux / dz[k - 1];
                next_number[k - 1] += number_flux / dz[k - 1];
            }
            else
            {
                result.surface += flux;
            }
        }
        result.mass.swap(next_mass);
        result.number.swap(next_number);
    }
    for (int k = 0; k < levels; ++k)
    {
        const double rho = d.density[column * d.nz + d.k_begin + k];
        result.mass_turnover[k] /= rho * d.dt;
        result.mass[k]   = (result.mass[k] - initial_mass[k]) / rho / d.dt;
        result.number[k] = (result.number[k] - initial_number[k]) / rho / d.dt;
    }
    return result;
}

void exercise(const char* backend)
{
    ComputeBackendRuntimeState runtime;
    runtime.config.backend        = backend;
    runtime.config.allow_fallback = false;
    std::string error;
    REQUIRE(initialize_compute_backend_runtime(runtime, CoordinateSystem::Cartesian, error));
    auto* kernels = runtime.backend->microphysics_kernels();
    REQUIRE(kernels != nullptr);
    REQUIRE(kernels->supports(MicrophysicsKernelOp::ThompsonMomentSedimentation));
    constexpr int                       nx = 5, ny = 4, nz = 9, cells = nx * ny * nz;
    const float                         nan = std::numeric_limits<float>::quiet_NaN();
    std::vector<float>                  mass(cells, nan), number(cells, nan), rho(cells, nan), temperature(cells, nan);
    std::vector<float>                  dz(cells, nan), w(cells, nan), multiplier(cells, nan);
    std::vector<float>                  dm(cells, 123), dn(cells, 456), surface(nx * ny, 789);
    ThompsonMomentSedimentationDispatch d;
    d.mass                  = mass.data();
    d.number                = number.data();
    d.density               = rho.data();
    d.temperature           = temperature.data();
    d.thickness             = dz.data();
    d.vertical_velocity     = w.data();
    d.snow_speed_multiplier = multiplier.data();
    d.mass_tendency         = dm.data();
    d.number_tendency       = dn.data();
    d.surface_mass          = surface.data();
    d.nx                    = nx;
    d.ny                    = ny;
    d.nz                    = nz;
    d.i_begin               = 1;
    d.i_end                 = nx - 1;
    d.j_begin               = 1;
    d.j_end                 = ny - 1;
    d.k_begin               = 1;
    d.k_end                 = nz - 1;
    for (int species = 0; species < 5; ++species)
    {
        for (float dt : {0.25f, 120.0f})
        {
            for (bool empty_layers : {false, true})
            {
                INFO(backend);
                INFO(species);
                INFO(dt);
                INFO(empty_layers);
                d.species = static_cast<ThompsonSedimentationSpecies>(species);
                d.dt      = dt;
                for (int i = d.i_begin; i < d.i_end; ++i)
                {
                    for (int j = d.j_begin; j < d.j_end; ++j)
                    {
                        for (int k = d.k_begin; k < d.k_end; ++k)
                        {
                            const int n    = (i * ny + j) * nz + k;
                            mass[n]        = empty_layers && k % 2 ? 0.0f : 1e-4f * (1 + k + i);
                            number[n]      = mass[n] == 0 ? 0.0f : species == 0 ? 2e8f : species == 1 ? 1e4f : 2e5f;
                            rho[n]         = 0.6f + 0.1f * k;
                            temperature[n] = 225.0f + 8.0f * k;
                            dz[n]          = 60.0f + 20.0f * k;
                            w[n]           = (j == 1 && k == 2) ? 0.2f : 0.0f;
                            multiplier[n]  = 1.0f + 0.1f * k;
                        }
                    }
                }
                REQUIRE(kernels->dispatch_thompson_moment_sedimentation(d));
                const auto first_dm = dm, first_dn = dn, first_surface = surface;
                REQUIRE(kernels->dispatch_thompson_moment_sedimentation(d));
                CHECK(dm == first_dm);
                CHECK(dn == first_dn);
                CHECK(surface == first_surface);
                for (int i = 0; i < nx; ++i)
                {
                    for (int j = 0; j < ny; ++j)
                    {
                        const int  column   = i * ny + j;
                        const bool physical = i >= d.i_begin && i < d.i_end && j >= d.j_begin && j < d.j_end;
                        if (!physical)
                        {
                            CHECK(surface[column] == 0);
                        }
                        Reference expected;
                        if (physical)
                        {
                            expected = reference(d, column);
                            CHECK(std::abs(surface[column] - expected.surface) <= 3e-5 * expected.surface + 1e-9);
                        }
                        double budget  = surface[column];
                        double initial = 0;
                        for (int k = 0; k < nz; ++k)
                        {
                            const int n = column * nz + k;
                            if (!physical || k < d.k_begin || k >= d.k_end)
                            {
                                CHECK(dm[n] == 0);
                                CHECK(dn[n] == 0);
                                continue;
                            }
                            const int local = k - d.k_begin;
                            INFO("column=" << column << " k=" << k << " nano tendency=" << dm[n] * 1e9
                                           << " nano reference=" << expected.mass[local] * 1e9
                                           << " nano error=" << (dm[n] - expected.mass[local]) * 1e9);
                            CHECK(std::abs(dm[n] - expected.mass[local]) <=
                                  5e-5 * expected.mass_turnover[local] + 2e-10 / dt);
                            CHECK(std::abs(dn[n] - expected.number[local]) <=
                                  5e-5 * std::abs(expected.number[local]) + 4e-7 *
                                                                                (species == 0   ? 2e8
                                                                                 : species == 1 ? 1e4
                                                                                                : 2e5) /
                                                                                dt);
                            CHECK(mass[n] + dt * dm[n] >= -2e-10);
                            CHECK(number[n] + dt * dn[n] >= -4e-7 * (species == 0 ? 2e8 : species == 1 ? 1e4 : 2e5));
                            budget += static_cast<double>(dm[n]) * rho[n] * dz[n] * dt;
                            initial += static_cast<double>(mass[n]) * rho[n] * dz[n];
                        }
                        CHECK(std::abs(budget) <= 3e-6 * initial + 1e-10);
                    }
                }
            }
        }
    }
    const auto saved_dm = dm, saved_dn = dn, saved_surface = surface;
    const auto unchanged = [&]()
    {
        CHECK(dm == saved_dm);
        CHECK(dn == saved_dn);
        CHECK(surface == saved_surface);
    };
    d.maximum_substeps = 1;
    CHECK_FALSE(kernels->dispatch_thompson_moment_sedimentation(d));
    unchanged();
    d.maximum_substeps         = 4096;
    const int   physical_index = (ny + 1) * nz + 1;
    const float saved_density  = rho[physical_index];
    rho[physical_index]        = nan;
    CHECK_FALSE(kernels->dispatch_thompson_moment_sedimentation(d));
    unchanged();
    rho[physical_index]      = saved_density;
    const float saved_number = number[physical_index];
    d.species                = ThompsonSedimentationSpecies::Rain;
    number[physical_index]   = 1e38f;
    CHECK_FALSE(kernels->dispatch_thompson_moment_sedimentation(d));
    unchanged();
    number[physical_index] = saved_number;
    d.mass_tendency        = mass.data() + 1;
    CHECK_FALSE(kernels->dispatch_thompson_moment_sedimentation(d));
    unchanged();
    d.mass_tendency   = dm.data();
    d.number_tendency = dm.data() + 1;
    CHECK_FALSE(kernels->dispatch_thompson_moment_sedimentation(d));
    unchanged();
    d.number_tendency = dn.data();
    d.dt              = std::numeric_limits<float>::max();
    CHECK_FALSE(kernels->dispatch_thompson_moment_sedimentation(d));
    unchanged();
}
void integrated(const char* backend)
{
    ComputeBackendRuntimeState runtime;
    runtime.config.backend        = backend;
    runtime.config.allow_fallback = false;
    std::string error;
    const bool  initialized = initialize_compute_backend_runtime(runtime, CoordinateSystem::Cartesian, error);
    INFO(error);
    REQUIRE(initialized);
    constexpr int            nx = 4, ny = 4, nz = 6;
    constexpr double         dt = 0.2;
    Field3D                  p(nx, ny, nz, 100000), theta(nx, ny, nz, 265), rho(nx, ny, nz, 1.1f), w(nx, ny, nz, 0.0f);
    Field3D                  qv(nx, ny, nz, 0.0015f), qc(nx, ny, nz, 1e-4f), qr(nx, ny, nz, 3e-4f);
    Field3D                  qi(nx, ny, nz, 1e-5f), qs(nx, ny, nz, 2e-4f), qg(nx, ny, nz, 1e-4f), qh(nx, ny, nz, 0);
    const PhysicalCellDomain domain{1, nx - 1, 1, ny - 1, 1, nz - 1};
    for (int i = 1; i < nx - 1; ++i)
    {
        for (int j = 1; j < ny - 1; ++j)
        {
            for (int k = 1; k < nz - 1; ++k)
            {
                theta(i, j, k) = 240.0f + 13.0f * k;
                qv(i, j, k)    = thermodynamics::saturation_mixing_ratio_water(theta(i, j, k), p(i, j, k));
            }
        }
    }
    tmv::ThompsonScheme cpu, native;
    for (auto* scheme : {&cpu, &native})
    {
        scheme->set_physical_cell_domain(nx, ny, nz, domain);
        scheme->set_vertical_spacing_m(100.0);
        scheme->set_vertical_velocity_field(w, false);
        scheme->prepare_prognostic_state(rho, qc, qr, qi, qs, qg, qh);
    }
    native.bind_execution_owners(runtime, nullptr);
    native.set_require_native_dispatch(true);
    using Tendencies   = std::array<Field3D, 8>;
    const auto compute = [&](tmv::ThompsonScheme& scheme, Tendencies& t)
    {
        scheme.compute_tendencies(
            p, theta, rho, qv, qc, qr, qi, qs, qg, qh, dt, t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7]);
    };
    const auto exact_fields = [](const std::vector<Field3D>& a, const std::vector<Field3D>& b)
    {
        REQUIRE(a.size() == b.size());
        for (std::size_t f = 0; f < a.size(); ++f)
        {
            REQUIRE(a[f].size() == b[f].size());
            CHECK(std::equal(a[f].data(), a[f].data() + a[f].size(), b[f].data()));
        }
    };
    Tendencies cpu_result, native_result;
    REQUIRE_NOTHROW(compute(cpu, cpu_result));
    REQUIRE_NOTHROW(compute(native, native_result));
    CHECK(runtime.backend->successful_kernel_dispatch_count(nimbus::compute::KernelId::ThompsonCloudAdjustment) == 1);
    CHECK(runtime.backend->successful_kernel_dispatch_count(nimbus::compute::KernelId::ThompsonWarmRain) == 1);
    CHECK(runtime.backend->successful_kernel_dispatch_count(nimbus::compute::KernelId::ThompsonCloudPhase) == 1);
    CHECK(runtime.backend->successful_kernel_dispatch_count(nimbus::compute::KernelId::ThompsonFreezing) == 1);
    CHECK(runtime.backend->successful_kernel_dispatch_count(nimbus::compute::KernelId::ThompsonDeposition) == 1);
    CHECK(runtime.backend->successful_kernel_dispatch_count(nimbus::compute::KernelId::ThompsonCompletion) == 1);
    CHECK(runtime.backend->successful_kernel_dispatch_count(nimbus::compute::KernelId::ThompsonMomentSedimentation) ==
          5);
    for (std::size_t f = 0; f < cpu_result.size(); ++f)
    {
        for (std::size_t n = 0; n < cpu_result[f].size(); ++n)
        {
            INFO(f);
            INFO(n);
            CHECK(std::abs(cpu_result[f].data()[n] - native_result[f].data()[n]) <=
                  1e-4 * std::abs(cpu_result[f].data()[n]) + 2e-9);
        }
    }
    REQUIRE(cpu.validate_prognostic_update(dt, error));
    REQUIRE(native.validate_prognostic_update(dt, error));
    tmv::ThompsonScheme restored;
    restored.set_physical_cell_domain(nx, ny, nz, domain);
    restored.set_vertical_spacing_m(100.0);
    restored.set_vertical_velocity_field(w, false);
    REQUIRE(restored.restore_portable_state(native.portable_state(), error));
    restored.bind_execution_owners(runtime, nullptr);
    restored.set_require_native_dispatch(true);
    native.commit_prognostic_update(dt);
    restored.commit_prognostic_update(dt);
    exact_fields(native.portable_state().fields, restored.portable_state().fields);
    Tendencies resumed, continued;
#ifdef _OPENMP
    struct RestoreThreads
    {
        int count = omp_get_max_threads();
        ~RestoreThreads()
        {
            omp_set_num_threads(count);
        }
    } restore_threads;
    omp_set_num_threads(1);
#endif
    REQUIRE_NOTHROW(compute(native, continued));
#ifdef _OPENMP
    omp_set_num_threads(4);
#endif
    REQUIRE_NOTHROW(compute(restored, resumed));
    for (std::size_t f = 0; f < continued.size(); ++f)
    {
        CHECK(std::equal(continued[f].data(), continued[f].data() + continued[f].size(), resumed[f].data()));
    }
    exact_fields(native.portable_state().fields, restored.portable_state().fields);
    const auto before   = native.portable_state();
    const auto previous = continued;
    ComputeBackendRuntimeState unavailable;
    native.bind_execution_owners(unavailable, nullptr);
    CHECK_THROWS_WITH(compute(native, continued),
                      "native Thompson requires prepared state and native process ownership");
    exact_fields(before.fields, native.portable_state().fields);
    for (std::size_t f = 0; f < continued.size(); ++f)
    {
        CHECK(std::equal(continued[f].data(), continued[f].data() + continued[f].size(), previous[f].data()));
    }
    native.bind_execution_owners(runtime, nullptr);
    for (std::size_t n = 0; n < theta.size(); ++n)
    {
        theta.data()[n] = 285.0f;
        qv.data()[n]    = thermodynamics::saturation_mixing_ratio_water(285.0f, 100000.0f);
        qc.data()[n] = qi.data()[n] = qs.data()[n] = qg.data()[n] = 0.0f;
    }
    for (auto* scheme : {&cpu, &native})
    {
        const auto state_before  = scheme->portable_state();
        const auto result_before = continued;
        scheme->set_vertical_spacing_m(1e-10);
        CHECK_THROWS(compute(*scheme, continued));
        exact_fields(state_before.fields, scheme->portable_state().fields);
        for (std::size_t f = 0; f < continued.size(); ++f)
        {
            CHECK(std::equal(continued[f].data(), continued[f].data() + continued[f].size(), result_before[f].data()));
        }
    }
}
} // namespace

TEST_CASE("Metal Thompson sedimentation transports mass and number conservatively", "[native-thompson][metal]")
{
    exercise("metal");
}

TEST_CASE("Vulkan Thompson sedimentation transports mass and number conservatively", "[native-thompson][vulkan]")
{
    exercise("vulkan");
}

TEST_CASE("Metal Thompson sedimentation couples to private process state", "[native-thompson][metal][coupling]")
{
    integrated("metal");
}

TEST_CASE("Vulkan Thompson sedimentation couples to private process state", "[native-thompson][vulkan][coupling]")
{
    integrated("vulkan");
}
