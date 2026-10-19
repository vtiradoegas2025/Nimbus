#include "catch2/catch.hpp"
#include "compute/backend_runtime.hpp"
#include "core/runtime/simulation_state.hpp"
#include "dynamics/schemes/klemp_wilhelmson_split/cartesian_cgrid_3d.hpp"
#include "terrain/earth_fixed_geometry.hpp"
#include "terrain/conservative_transport.hpp"

#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <omp.h>

namespace
{
void compare_flux(const tmv::CartesianMomentumInterfaceFluxes& actual,
                  const tmv::CartesianMomentumInterfaceFluxes& expected,
                  bool                                         exact,
                  bool                                         mapped = false)
{
    CHECK(actual.contribution_count == expected.contribution_count);
    CHECK(actual.accumulated_dt_s == expected.accumulated_dt_s);
    const std::array<const tmv::CartesianMomentumInterfaceSide*, 4> a{
        &actual.west, &actual.east, &actual.south, &actual.north};
    const std::array<const tmv::CartesianMomentumInterfaceSide*, 4> e{
        &expected.west, &expected.east, &expected.south, &expected.north};
    for (int side = 0; side < 4; ++side)
        for (int component = 0; component < 3; ++component)
        {
            REQUIRE(a[side]->component[component].size() == e[side]->component[component].size());
            for (std::size_t n = 0; n < a[side]->component[component].size(); ++n)
            {
                const double av = a[side]->component[component][n], ev = e[side]->component[component][n];
                if (exact)
                    CHECK(av == ev);
                else
                    CHECK(std::abs(av - ev) <= 2e-12 * std::max(1.0, std::abs(ev)));
            }
        }
    for (int component = 0; component < 3; ++component)
    {
        if (exact)
            CHECK(actual.source_impulse[component] == expected.source_impulse[component]);
        else
        {
            double cancellation_bound = 0;
            if (mapped)
                for (int side = 0; side < 4; ++side)
                    for (std::size_t n = 0; n < e[side]->component[component].size(); ++n)
                    {
                        const double av = a[side]->component[component][n], ev = e[side]->component[component][n];
                        cancellation_bound += std::abs(av - ev) + 8 * std::numeric_limits<double>::epsilon() *
                                                                      (std::abs(av) + std::abs(ev));
                    }
            CHECK(actual.source_impulse[component] ==
                  Approx(expected.source_impulse[component]).epsilon(2e-6).margin(1e-6 + cancellation_bound));
        }
    }
}
void exercise(const char* name, bool mapped = false)
{
    constexpr int          nx = 14, ny = 14, nz = 8;
    constexpr double       dx = 123.456789, dy = 234.567891, dz = 87.654321, dt = .37;
    SimulationStorageState storage{{nx, ny, nz, dx, dz, dt, dy}};
    storage.grid.geometry.initialize(nx, ny, nz, dx, dz, dy, CoordinateSystem::Cartesian, StaggerType::CGrid);
    storage.configuration.coordinate_system = CoordinateSystem::Cartesian;
    storage.configuration.stagger_type      = StaggerType::CGrid;
    auto& f                                 = storage.fields;
    for (auto* field : {&f.rho, &f.p, &f.u, &f.v, &f.w, &f.theta, &f.qv, &f.qc, &f.qr, &f.qi, &f.qs, &f.qg, &f.qh})
        field->resize(nx, ny, nz, 0);
    for (auto* profile : {&storage.base.rho0, &storage.base.p0, &storage.base.qv0, &storage.base.u0, &storage.base.v0})
        profile->resize(nz);
    for (int k = 0; k < nz; ++k)
    {
        storage.base.rho0[k] = 1.2 - .025 * k;
        storage.base.p0[k]   = 100000 * std::exp(-.0125 * k);
        storage.base.qv0[k]  = .01 - .00025 * k;
        storage.base.u0[k]   = 3.7;
        storage.base.v0[k]   = -2.3;
    }
    for (int i = 0; i < nx; ++i)
        for (int j = 0; j < ny; ++j)
            for (int k = 0; k < nz; ++k)
            {
                f.rho(i, j, k)   = storage.base.rho0[k] + .01 * std::sin(.3 * i + .2 * j + .1 * k);
                f.p(i, j, k)     = storage.base.p0[k] + 20 * std::sin(.2 * i) * std::cos(.3 * j) * (k + 1);
                f.theta(i, j, k) = 300;
                f.u(i, j, k)     = 3.7 + std::sin(.2 * i + .3 * j + .15 * k);
                f.v(i, j, k)     = -2.3 + std::cos(.3 * i + .1 * j - .2 * k);
                f.w(i, j, k)     = k == nz - 1 ? 0 : .15 * std::sin(.4 * i + .3 * j + .2 * k);
                f.qv(i, j, k)    = storage.base.qv0[k] + .0001 * std::sin(.3 * i + .2 * j);
                f.qc(i, j, k)    = .00001 * (1 + std::cos(.2 * i + .3 * j));
                f.qr(i, j, k)    = .00002;
                f.qi(i, j, k)    = .00003;
            }
    storage.base.theta0.assign(nz, 300.0);
    if (mapped)
    {
        auto& terrain                          = storage.runtime.terrain;
        terrain.config.scheme_id               = "bell";
        terrain.config.coord_id                = "smoothed";
        terrain.config.horizontal_placement_id = "earth_fixed";
        terrain.config.ztop                    = nz * dz;
        terrain.config.bell.h0                 = 10;
        terrain.config.bell.a                  = 600;
        tmv::EarthFixedTerrainBuildRequest request;
        request.nr       = nx;
        request.nth      = ny;
        request.nz       = nz;
        request.dx_m     = dx;
        request.dy_m     = dy;
        request.sampling = tmv::EarthFixedTerrainSampling::CellAverageBilinear;
        auto prepared    = tmv::prepare_earth_fixed_terrain_commit(terrain, request);
        REQUIRE(tmv::commit_earth_fixed_terrain(prepared, terrain));
        terrain.config.horizontal_placement_id = "grid_attached";
    }
    auto                        view = make_simulation_state_view(storage);
    tmv::CartesianCGrid3DScheme scheme(view);
    scheme.set_coriolis_profile({mapped ? 0.0 : 1e-4, 0});
    scheme.set_momentum_interface_flux_slow_weight(.5);
    ComputeBackendRuntimeState native;
    native.config.backend        = name;
    native.config.allow_fallback = false;
    std::string error;
    const bool  initialized = initialize_compute_backend_runtime(native, CoordinateSystem::Cartesian, error);
    INFO(error);
    REQUIRE(initialized);
    REQUIRE(native.backend->supports_dynamics(DynamicsKernelOp::ConservativeMomentum));
    std::array<Field3D, 8> rates;
    for (auto& field : rates)
        field.resize(nx, ny, nz, 0);
    tmv::SlowMomentumAdvectionTendencyOutputs adv{&rates[5], &rates[6], &rates[7]};
    const auto                                calculate = [&](bool pressure)
    {
        if (pressure)
            scheme.compute_fast_momentum_tendencies(f.u, f.v, f.w, f.rho, f.p, dt, rates[0], rates[1], rates[2]);
        else
            scheme.compute_slow_tendencies(
                f.u, f.v, f.w, f.rho, f.p, f.theta, dt, rates[0], rates[1], rates[2], rates[3], rates[4], &adv);
    };
    for (bool exterior : {false, true})
        for (int ratio : {1, 2, 3})
            for (bool pressure : {false, true})
            {
                CAPTURE(name, mapped, exterior, ratio, pressure);
                const tmv::CartesianInterfaceFootprint footprint =
                    exterior ? tmv::CartesianInterfaceFootprint{1, 13, 1, 13, 0, nz - 1}
                             : tmv::CartesianInterfaceFootprint{3, 9, 3, 9, 1, nz - 1};
                CartesianCGridMomentumFluxRegister reference;
                reference.configure(nx, ny, nz, footprint, ratio, ratio);
                scheme.set_momentum_interface_flux_register(&reference);
                storage.runtime.compute.active_kind = ComputeBackendKind::Cpu;
                calculate(pressure);
                const auto expected = rates;
                auto       actual   = reference;
                actual.reset_interval();
                scheme.set_momentum_interface_flux_register(&actual);
                storage.runtime.compute = std::move(native);
                calculate(pressure);
                const auto first          = rates;
                const auto first_register = actual;
                for (std::size_t field = 0; field < (pressure ? 3 : rates.size()); ++field)
                    for (std::size_t n = 0; n < f.rho.size(); ++n)
                    {
                        CAPTURE(field, n);
                        CHECK(rates[field].data()[n] == Approx(expected[field].data()[n])
                                                            .epsilon(mapped ? 3e-5 : 3e-6)
                                                            .margin(mapped ? 8e-5 : 1e-9));
                    }
                if (mapped)
                {
                    reference.reset_interval();
                    const auto&  metrics = storage.runtime.terrain.metrics;
                    const double weight  = pressure ? dt : .5 * dt;
                    if (pressure)
                        reference.stage_pressure_mapped(f.p, metrics, weight);
                    else
                        reference.stage_advective_mapped(f.rho, f.u, f.v, f.w, metrics, weight);
                    reference.stage_mapped_tendency_closure(f.rho, rates[0], rates[1], rates[2], metrics, weight);
                    reference.validate_staged_commit();
                    reference.commit_staged();
                }
                compare_flux(pressure ? actual.pressure_fluxes() : actual.advective_fluxes(),
                             pressure ? reference.pressure_fluxes() : reference.advective_fluxes(),
                             false,
                             mapped);
                omp_set_num_threads(4);
                actual.reset_interval();
                calculate(pressure);
                for (std::size_t field = 0; field < (pressure ? 3 : rates.size()); ++field)
                    CHECK(std::memcmp(first[field].data(), rates[field].data(), f.rho.size() * sizeof(float)) == 0);
                compare_flux(pressure ? actual.pressure_fluxes() : actual.advective_fluxes(),
                             pressure ? first_register.pressure_fluxes() : first_register.advective_fluxes(),
                             true);
                omp_set_num_threads(1);
                const float density = f.rho(3, 3, 2);
                f.rho(3, 3, 2)      = std::numeric_limits<float>::quiet_NaN();
                CHECK_THROWS(calculate(pressure));
                f.rho(3, 3, 2) = density;
                for (std::size_t field = 0; field < (pressure ? 3 : rates.size()); ++field)
                    CHECK(std::memcmp(first[field].data(), rates[field].data(), f.rho.size() * sizeof(float)) == 0);
                compare_flux(pressure ? actual.pressure_fluxes() : actual.advective_fluxes(),
                             pressure ? first_register.pressure_fluxes() : first_register.advective_fluxes(),
                             true);
                native = std::move(storage.runtime.compute);
            }
    if (mapped)
    {
        storage.runtime.compute = std::move(native);
        const auto& metrics     = storage.runtime.terrain.metrics;
        for (std::size_t n = 0; n < f.p.size(); ++n)
            f.p.data()[n] = float(tmv::terrain_transport::reference_profile_value_fp32(
                storage.base.p0, metrics.z.values[n], metrics.dzeta));
        CartesianCGridMomentumFluxRegister hydrostatic;
        hydrostatic.configure(nx, ny, nz, {1, nx - 1, 1, ny - 1, 0, nz - 1});
        scheme.set_momentum_interface_flux_register(&hydrostatic);
        calculate(true);
        for (int field = 0; field < 3; ++field)
            for (std::size_t n = 0; n < f.p.size(); ++n)
                CHECK(rates[field].data()[n] == 0);
        for (std::size_t n = 0; n < f.p.size(); ++n)
        {
            f.rho.data()[n] = float(tmv::terrain_transport::reference_density_fp32(
                storage.base.p0, storage.base.theta0, metrics.z.values[n], metrics.dzeta));
            f.qv.data()[n] = float(tmv::terrain_transport::reference_profile_value_fp32(
                storage.base.qv0, metrics.z.values[n], metrics.dzeta));
        }
        for (auto* field : {&f.u, &f.v, &f.w, &f.qc, &f.qr, &f.qi, &f.qs, &f.qg, &f.qh})
            field->fill(0);
        hydrostatic.reset_interval();
        calculate(false);
        for (int field = 0; field < 3; ++field)
            for (std::size_t n = 0; n < f.p.size(); ++n)
                CHECK(rates[field].data()[n] == 0);
        native = std::move(storage.runtime.compute);
    }
    auto*                         kernels = native.backend->dynamics_kernels();
    std::array<float, 16>         input{0, 1, 1, 1, 3, 0, 2, 0, 4, 0, 2, 0, 1, 0, 3, 0};
    std::array<float, 5>          output{};
    ConservativeReductionDispatch reduction{input.data(), output.data(), {1, 2}};
    REQUIRE(kernels->dispatch_conservative_reduction(reduction));
    CHECK(double(output[0]) + output[1] == 24);
    CHECK(output[2] == 4);
    CHECK(double(output[3]) + output[4] == 0);
    const auto valid_input = input;
    const auto unchanged   = output;
    for (int fault = 0; fault < 5; ++fault)
    {
        input            = valid_input;
        reduction.output = output.data();
        if (fault == 0)
            input[1] = 3;
        if (fault == 1)
            input[0] = .5f;
        if (fault == 2)
            input[10] = -2;
        if (fault == 3)
            input[4] = std::numeric_limits<float>::quiet_NaN();
        if (fault == 4)
            reduction.output = input.data() + 1;
        CHECK_FALSE(kernels->dispatch_conservative_reduction(reduction));
        CHECK(output == unchanged);
    }
}
} // namespace
TEST_CASE("Metal conservative momentum matches the CPU operator and interface ledgers",
          "[metal][conservative_momentum]")
{
    exercise("metal");
}
TEST_CASE("Vulkan conservative momentum matches the CPU operator and interface ledgers",
          "[vulkan][conservative_momentum]")
{
    exercise("vulkan");
}

TEST_CASE("Metal mapped momentum supplies metric-weighted native interface ledgers", "[metal][mapped_momentum]")
{
    exercise("metal", true);
}
TEST_CASE("Vulkan mapped momentum supplies metric-weighted native interface ledgers", "[vulkan][mapped_momentum]")
{
    exercise("vulkan", true);
}
