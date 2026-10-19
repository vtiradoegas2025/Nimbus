#include "numerics/advection/advection_cartesian.hpp"
#include "catch2/catch.hpp"
#include "core/runtime/simulation_state.hpp"
#include "terrain/earth_fixed_geometry.hpp"
#include "terrain/native_mapped_transaction.hpp"
#include "dynamics/moving_environment.hpp"
#include "dynamics/schemes/klemp_wilhelmson_split/cartesian_cgrid_3d.hpp"
#include "boundary_conditions/factory.hpp"
#include "compute/backend_runtime.hpp"
#include "dynamics/eos_pressure.hpp"
#include <algorithm>
#include <cmath>
#include <omp.h>
#include <cstring>
#include <limits>
namespace
{
using namespace tmv;
SimulationStorageState initial()
{
    constexpr int          nx = 8, ny = 7, nz = 8;
    SimulationStorageState s{{nx, ny, nz, 200, 100, .1, 200}};
    s.grid.geometry.initialize(nx, ny, nz, 200, 100, 200, CoordinateSystem::Cartesian, StaggerType::CGrid);
    s.configuration.stagger_type = StaggerType::CGrid;
    tmv::fields::visit(s.fields, tmv::fields::prognostic, [&](const char*, Field3D& f) { f.resize(nx, ny, nz, 0); });
    for (int k = 0; k < nz; ++k)
    {
        s.base.theta0.push_back(300 + .1 * k);
        s.base.p0.push_back(100000 * std::exp(-.012 * k));
        s.base.qv0.push_back(.012 - .0001 * k);
        s.base.u0.push_back(5 + .5 * k);
        s.base.v0.push_back(-2 + .1 * k);
        for (int i = 0; i < nx; ++i)
            for (int j = 0; j < ny; ++j)
            {
                s.fields.theta(i, j, k) = float(s.base.theta0[k]);
                s.fields.p(i, j, k)     = float(s.base.p0[k]);
                s.fields.qv(i, j, k)    = float(s.base.qv0[k]);
                s.fields.rho(i, j, k)   = 1;
                s.fields.u(i, j, k)     = float(s.base.u0[k]);
                s.fields.v(i, j, k)     = float(s.base.v0[k]);
            }
    }
    auto& terrain                          = s.runtime.terrain;
    terrain.config.scheme_id               = "bell";
    terrain.config.coord_id                = "smoothed";
    terrain.config.horizontal_placement_id = "earth_fixed";
    terrain.config.ztop                    = 800;
    terrain.config.bell.h0                 = 40;
    terrain.config.bell.a                  = 600;
    EarthFixedTerrainBuildRequest request;
    request.nr       = nx;
    request.nth      = ny;
    request.nz       = nz;
    request.dx_m     = 200;
    request.dy_m     = 200;
    request.sampling = EarthFixedTerrainSampling::CellAverageBilinear;
    auto prepared    = prepare_earth_fixed_terrain_commit(terrain, request);
    REQUIRE(commit_earth_fixed_terrain(prepared, terrain));
    fill_moving_environment(s.fields, s.base, terrain.metrics);
    s.base.rho0.assign(nz, 1.0);
    s.configuration.domain_frame.velocity_ms = {0, 0};
    s.configuration.coriolis_enabled         = false;
    s.configuration.top_bc                   = init::config::TopBC::Rigid;
    s.configuration.bottom_bc                = init::config::BottomBC::RigidSlip;
    s.configuration.lateral_bc_west = s.configuration.lateral_bc_east = s.configuration.lateral_bc_south =
        s.configuration.lateral_bc_north                              = init::config::LateralBC::Radiative;
    s.tendencies.du_dt_pbl.resize(nx, ny, nz, 0);
    s.tendencies.dv_dt_pbl.resize(nx, ny, nz, 0);
    s.fields.qr.fill(.0001f);
    s.fields.qc.fill(.0002f);
    s.fields.qi.fill(.00003f);
    auto view     = make_simulation_state_view(s);
    auto boundary = create_cartesian_cgrid_bc_scheme(view);
    boundary->apply_full();
    return s;
}

void reference(SimulationStorageState& s, double dt, int substeps)
{
    auto                   view = make_simulation_state_view(s);
    CartesianCGrid3DScheme dynamics(view);
    auto                   bc = create_cartesian_cgrid_bc_scheme(view);
    std::array<Field3D, 5> slow;
    for (auto& f : slow)
        f.resize(s.grid.nr, s.grid.nth, s.grid.nz, 0);
    const auto compute = [&]
    {
        dynamics.compute_slow_tendencies(s.fields.u,
                                         s.fields.v,
                                         s.fields.w,
                                         s.fields.rho,
                                         s.fields.p,
                                         s.fields.theta,
                                         dt,
                                         slow[0],
                                         slow[1],
                                         slow[2],
                                         slow[3],
                                         slow[4]);
    };
    const auto update = [&](double step)
    {
        std::array<Field3D*, 4> fields{&s.fields.u, &s.fields.v, &s.fields.w, &s.fields.p};
        for (std::size_t f = 0; f < fields.size(); ++f)
            for (std::size_t n = 0; n < s.fields.rho.size(); ++n)
                fields[f]->data()[n] = float(double(fields[f]->data()[n]) + step * slow[f == 3 ? 4 : f].data()[n]);
    };
    const auto original = s.fields;
    compute();
    const auto first = slow;
    update(dt);
    compute();
    for (std::size_t f = 0; f < slow.size(); ++f)
        for (std::size_t n = 0; n < slow[f].size(); ++n)
            slow[f].data()[n] = .5f * (first[f].data()[n] + slow[f].data()[n]);
    s.fields = original;
    update(dt);
    bc->apply_acoustic(0);
    bc->seed_acoustic_radiation();
    const double step = dt / substeps;
    for (int iteration = 0; iteration < substeps; ++iteration)
    {
        dynamics.compute_fast_pressure_tendencies(
            s.fields.u, s.fields.v, s.fields.w, s.fields.rho, s.fields.p, step, slow[3], slow[4]);
        for (std::size_t n = 0; n < s.fields.rho.size(); ++n)
        {
            s.fields.rho.data()[n] = float(double(s.fields.rho.data()[n]) + step * slow[3].data()[n]);
            s.fields.p.data()[n] = float(dynamics_eos::pressure_fp64(s.fields.rho.data()[n], s.fields.theta.data()[n]));
        }
        bc->apply_acoustic(0);
        dynamics.compute_fast_momentum_tendencies(
            s.fields.u, s.fields.v, s.fields.w, s.fields.rho, s.fields.p, step, slow[0], slow[1], slow[2]);
        slow[4].fill(0);
        update(step);
        bc->apply_acoustic(step);
    }
}
void exercise(const char* backend)
{
    const int threads = omp_get_max_threads();
    omp_set_num_threads(1);
    ComputeBackendRuntimeState runtime;
    runtime.config.backend        = backend;
    runtime.config.allow_fallback = false;
    std::string error;
    const bool  initialized = initialize_compute_backend_runtime(runtime, CoordinateSystem::Cartesian, error);
    INFO(error);
    REQUIRE(initialized);
    auto cpu = initial(), gpu = initial();
    auto view            = make_simulation_state_view(gpu);
    view.compute_runtime = &runtime;
    for (int step = 0; step < 3; ++step)
    {
        CAPTURE(backend, step);
        terrain_native::MappedAcousticResult result;
        REQUIRE(terrain_native::advance_mapped_acoustic_core(view, .025, 3, result));
        REQUIRE(result.complete);
        REQUIRE(result.acoustic.complete);
        REQUIRE(result.slow.complete);
        CartesianCGridTransportContext transport;
        transport.begin(view, gpu.grid.nr, gpu.grid.nth, gpu.grid.nz, .025, &gpu.runtime.terrain.metrics);
        transport.capture_density_begin(result.density_begin);
        transport.mass_flux_x = result.mass_flux_x;
        transport.mass_flux_y = result.mass_flux_y;
        transport.mass_flux_z = result.mass_flux_z;
        transport.record_external_accumulation(.025, 3, true);
        transport.complete(view);
        gpu.configuration.admission.earth_fixed_terrain_motion = true;
        for (const auto reconstruction :
             {CartesianFaceReconstruction::WENO5, CartesianFaceReconstruction::MonotonicCentered})
        {
            Field3D                            expected_scalar, actual_scalar;
            ScalarBoundaryTransportReport      expected_boundary, actual_boundary;
            ScalarMoistEnergyTransportReportV1 expected_energy, actual_energy;
            ComputeBackendRuntimeState         host;
            auto                               host_view = view;
            host_view.compute_runtime                    = &host;
            advect_scalar_conservative_3d_cartesian(host_view,
                                                    gpu.fields.qv,
                                                    expected_scalar,
                                                    .025,
                                                    true,
                                                    &transport,
                                                    &expected_boundary,
                                                    ScalarMoistEnergyKind::WaterVapor,
                                                    &expected_energy,
                                                    reconstruction);
            advect_scalar_conservative_3d_cartesian(view,
                                                    gpu.fields.qv,
                                                    actual_scalar,
                                                    .025,
                                                    true,
                                                    &transport,
                                                    &actual_boundary,
                                                    ScalarMoistEnergyKind::WaterVapor,
                                                    &actual_energy,
                                                    reconstruction);
            CHECK(actual_boundary.complete);
            CHECK(actual_energy.complete);
            double max_scalar_error = 0;
            for (std::size_t n = 0; n < expected_scalar.size(); ++n)
                max_scalar_error =
                    std::max(max_scalar_error, std::abs(double(actual_scalar.data()[n]) - expected_scalar.data()[n]));
            CHECK(max_scalar_error < 5e-9);
            CHECK(std::abs(actual_energy.closure_residual_j) < 1e-3);
        }
        reference(cpu, .025, 3);
        std::array<const Field3D*, 5> lhs{&cpu.fields.u, &cpu.fields.v, &cpu.fields.w, &cpu.fields.rho, &cpu.fields.p};
        std::array<const Field3D*, 5> rhs{&gpu.fields.u, &gpu.fields.v, &gpu.fields.w, &gpu.fields.rho, &gpu.fields.p};
        for (std::size_t f = 0; f < lhs.size(); ++f)
        {
            double maximum = 0;
            for (std::size_t n = 0; n < lhs[f]->size(); ++n)
                maximum = std::max(maximum, std::abs(double(lhs[f]->data()[n]) - rhs[f]->data()[n]));
            CAPTURE(f, maximum);
            CHECK(maximum <= (step + 1) * (f == 4 ? .0625 : f == 3 ? 5e-7 : 5e-5));
        }
    }
    auto atomic                                               = initial();
    atomic.configuration.admission.earth_fixed_terrain_motion = true;
    auto atomic_view                                          = make_simulation_state_view(atomic);
    atomic_view.compute_runtime                               = &runtime;
    CartesianCGridTransportContext atomic_transport;
    atomic_transport.begin(
        atomic_view, atomic.grid.nr, atomic.grid.nth, atomic.grid.nz, .025, &atomic.runtime.terrain.metrics);
    atomic_transport.accumulate(
        atomic_view, atomic.fields.rho, atomic.fields.u, atomic.fields.v, atomic.fields.w, .025);
    atomic_transport.complete(atomic_view);
    CartesianInterfaceFluxRegister capture;
    capture.configure(atomic.grid.nr,
                      atomic.grid.nth,
                      atomic.grid.nz,
                      {1, atomic.grid.nr - 1, 1, atomic.grid.nth - 1, 0, atomic.grid.nz});
    Field3D changed;
    advect_scalar_conservative_3d_cartesian(atomic_view,
                                            atomic.fields.qv,
                                            changed,
                                            .025,
                                            true,
                                            &atomic_transport,
                                            nullptr,
                                            ScalarMoistEnergyKind::None,
                                            nullptr,
                                            CartesianFaceReconstruction::MonotonicCentered);
    REQUIRE(std::memcmp(changed.data(), atomic.fields.qv.data(), changed.size() * sizeof(float)) != 0);
    const auto saved              = atomic.fields.qv;
    atomic.fields.tracer(3, 3, 2) = std::numeric_limits<float>::quiet_NaN();
    ScalarBoundaryTransportReport preserved_report;
    preserved_report.complete               = true;
    preserved_report.committed_state_change = 17;
    CartesianScalarTransport scalars[]{
        {&atomic.fields.qv, 0, true, &preserved_report, ScalarMoistEnergyKind::None, nullptr, &capture},
        {&atomic.fields.tracer, 0, true, nullptr, ScalarMoistEnergyKind::None, nullptr, nullptr}};
    REQUIRE_THROWS(advect_scalars_conservative_3d_cartesian(
        atomic_view, scalars, 2, .025, &atomic_transport, CartesianFaceReconstruction::MonotonicCentered));
    CHECK(std::memcmp(saved.data(), atomic.fields.qv.data(), saved.size() * sizeof(float)) == 0);
    CHECK(preserved_report.complete);
    CHECK(preserved_report.committed_state_change == 17);
    CHECK(capture.interval_empty());
    const auto                           before = gpu.fields;
    terrain_native::MappedAcousticResult rejected;
    CHECK_FALSE(terrain_native::advance_mapped_acoustic_core(view, -.1, 3, rejected));
    CHECK_FALSE(rejected.complete);
    CHECK(std::equal(before.u.data(), before.u.data() + before.u.size(), gpu.fields.u.data()));
    omp_set_num_threads(threads);
}
} // namespace
TEST_CASE("Mapped moist acoustic core matches CPU evolution on Metal", "[metal][mapped][moist]")
{
    exercise("metal");
}
TEST_CASE("Mapped moist acoustic core matches CPU evolution on Vulkan", "[vulkan][mapped][moist]")
{
    exercise("vulkan");
}
