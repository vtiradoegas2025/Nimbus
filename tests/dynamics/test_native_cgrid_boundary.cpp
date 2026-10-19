#include "dynamics/final_boundary_energy_report.hpp"
#include "catch2/catch.hpp"
#include "compute/backend_runtime.hpp"
#include "core/runtime/simulation_state.hpp"
#include "boundary_conditions/factory.hpp"
#include "dynamics/native_cgrid_boundary.hpp"
#include "core/runtime/native_earth_fixed_transport.hpp"
#include "dynamics/moving_environment.hpp"
#include "terrain/native_conservative_remap.hpp"
#include "diagnostics/deterministic_reduction.hpp"
#include <array>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <omp.h>
namespace
{
using namespace tmv;
SimulationStorageState initial(int nx, int ny, int nz, float scale = 1)
{
    SimulationStorageState s{{nx, ny, nz, 100, 100, .1, 100}};
    s.grid.geometry.initialize(nx, ny, nz, 100, 100, 100, CoordinateSystem::Cartesian, StaggerType::CGrid);
    s.configuration.stagger_type             = StaggerType::CGrid;
    s.configuration.domain_frame.velocity_ms = {2, -3};
    fields::visit(s.fields, fields::prognostic, [&](const char*, Field3D& field) { field.resize(nx, ny, nz, 0); });
    for (std::size_t n = 0; n < s.fields.u.size(); ++n)
    {
        s.fields.u.data()[n]     = scale * float(int(n % 19) - 9);
        s.fields.v.data()[n]     = scale * float(int(n % 17) - 8);
        s.fields.w.data()[n]     = scale * float(int(n % 13) - 6);
        s.fields.rho.data()[n]   = 1 + .001f * n;
        s.fields.p.data()[n]     = 80000 + float(n);
        s.fields.theta.data()[n] = 300 + .001f * n;
        s.fields.qv.data()[n]    = .005f + .000001f * n;
    }
    s.runtime.boundary_conditions.radiative_u_west  = {19};
    s.runtime.boundary_conditions.radiative_u_east  = {20};
    s.runtime.boundary_conditions.radiative_v_south = {21};
    s.runtime.boundary_conditions.radiative_v_north = {22};
    return s;
}
void equal(const PrognosticFieldStorage& a, const PrognosticFieldStorage& b)
{
    const std::array<const Field3D*, 8> lhs{&a.u, &a.v, &a.w, &a.rho, &a.p, &a.theta, &a.qv, &a.tke};
    const std::array<const Field3D*, 8> rhs{&b.u, &b.v, &b.w, &b.rho, &b.p, &b.theta, &b.qv, &b.tke};
    for (std::size_t f = 0; f < lhs.size(); ++f)
    {
        CAPTURE(f);
        REQUIRE(lhs[f]->size() == rhs[f]->size());
        CHECK(std::memcmp(lhs[f]->data(), rhs[f]->data(), lhs[f]->size() * sizeof(float)) == 0);
    }
}
void equal(const BoundaryConditionWaveState& a, const BoundaryConditionWaveState& b)
{
    CHECK(a.radiative_u_west == b.radiative_u_west);
    CHECK(a.radiative_u_east == b.radiative_u_east);
    CHECK(a.radiative_v_south == b.radiative_v_south);
    CHECK(a.radiative_v_north == b.radiative_v_north);
}
class Reject final : public TerrainComputeKernels
{
public:
    bool supports_mapped_final_state() const override
    {
        return true;
    }
    bool dispatch_mapped_final_state(const MappedTerrainFinalStateDispatch& d) override
    {
        std::fill_n(d.u_final, std::size_t(d.nx) * d.ny * d.nz, 19);
        return false;
    }
};
void composed(TerrainComputeKernels& kernels)
{
    auto state                          = initial(6, 5, 4);
    using L                             = init::config::LateralBC;
    state.configuration.lateral_bc_west = state.configuration.lateral_bc_east = L::Radiative;
    state.configuration.lateral_bc_south = state.configuration.lateral_bc_north = L::Radiative;
    state.configuration.bottom_bc                                               = init::config::BottomBC::RigidSlip;
    state.configuration.top_bc                                                  = init::config::TopBC::Rigid;
    state.base.theta0.assign(4, 300);
    state.base.p0.assign(4, 100000);
    state.base.qv0.assign(4, .005);
    state.base.u0.assign(4, 8);
    state.base.v0.assign(4, -3);
    auto& terrain                          = state.runtime.terrain;
    terrain.config.scheme_id               = "bell";
    terrain.config.coord_id                = "smoothed";
    terrain.config.horizontal_placement_id = "earth_fixed";
    terrain.config.ztop                    = 400;
    terrain.config.bell.h0                 = 20;
    terrain.config.bell.a                  = 200;
    EarthFixedTerrainBuildRequest request;
    request.nr       = 6;
    request.nth      = 5;
    request.nz       = 4;
    request.dx_m     = 100;
    request.dy_m     = 100;
    request.sampling = EarthFixedTerrainSampling::CellAverageBilinear;
    auto geometry    = prepare_earth_fixed_terrain_commit(terrain, request, kernels);
    REQUIRE(commit_earth_fixed_terrain(geometry, terrain));
    fill_moving_environment(state.fields, state.base, terrain.metrics, kernels);
    apply_native_cgrid_boundary(state, kernels, true);
    const auto                       original      = state.fields;
    const BoundaryConditionWaveState original_wave = state.runtime.boundary_conditions;
    auto                             view          = make_simulation_state_view(state);
    const auto                       transport     = prepare_native_earth_fixed_transport(view, .1, kernels);
    const auto                       energy        = prepare_native_earth_fixed_energy(state, transport, kernels);
    const auto&                      g0            = terrain.metrics;
    const auto&                      g1            = transport.motion.terrain.geometry.metrics;
    auto                             next          = initial(6, 5, 4);
    next.configuration                             = state.configuration;
    next.base                                      = state.base;
    next.fields                                    = state.fields;
    for (auto* field : {&next.fields.qv,
                        &next.fields.qc,
                        &next.fields.qr,
                        &next.fields.qi,
                        &next.fields.qs,
                        &next.fields.qg,
                        &next.fields.qh,
                        &next.fields.tke,
                        &next.fields.tracer})
        terrain_native::remap_inventory(
            kernels, *field, *field, state.fields.rho, transport.density, g0, g1, transport.motion.swept_volumes, true);
    next.fields.rho = transport.density;
    next.fields.u   = transport.projected_u;
    next.fields.v   = transport.projected_v;
    next.fields.w   = transport.momentum.w;
    apply_native_cgrid_boundary(next, kernels);
    correct_moving_ground_momentum(
        g1, next.fields.rho, next.fields.u, next.fields.v, next.fields.w, transport.ground_targets, kernels);
    const auto storage   = reconstruct_moving_energy(next.fields, g1, energy.target_density, kernels);
    const auto inventory = [&]()
    {
        const auto                  density = moving_energy_density(next.fields, g1);
        diagnostics::CompensatedSum sum;
        for (int i = 1; i < 5; ++i)
            for (int j = 1; j < 4; ++j)
                for (int k = 0; k < 4; ++k)
                    sum.add(density[(std::size_t(i) * 5 + j) * 4 + k] * g1.J(i, j, k) * g1.dx * g1.dy * g1.dzeta);
        return sum.value();
    };
    const double before = inventory();
    CHECK(std::abs(before - storage.stored_energy_j) <= 5e-12 * std::abs(before));
    CHECK(std::abs(storage.stored_energy_j - storage.requested_energy_j - storage.storage_rounding_j) <=
          5e-12 * std::abs(before));
    CHECK(std::abs(energy.transport.final_inventory - energy.transport.initial_inventory +
                   energy.transport.boundary_outflow - energy.transport.storage_rounding) <= 5e-12 * std::abs(before));
    fill_moving_environment(next.fields, next.base, g1, kernels);
    auto reference          = initial(6, 5, 4);
    reference.configuration = next.configuration;
    reference.fields        = next.fields;
    auto reference_view     = make_simulation_state_view(reference);
    auto scheme             = create_cartesian_cgrid_bc_scheme(reference_view);
    scheme->apply_full();
    scheme->seed_acoustic_radiation();
    apply_native_cgrid_boundary(next, kernels, true);
    equal(reference.fields, next.fields);
    equal(reference.runtime.boundary_conditions, next.runtime.boundary_conditions);
    CHECK(inventory() == before);
    equal(original, state.fields);
    equal(original_wave, state.runtime.boundary_conditions);
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
    composed(*kernels);
    const int threads = omp_get_max_threads();
    omp_set_num_threads(1);
    using L = init::config::LateralBC;
    using B = init::config::BottomBC;
    using T = init::config::TopBC;
    for (int encoded = 0; encoded < 625; ++encoded)
        for (int vertical = 0; vertical < 6; ++vertical)
        {
            CAPTURE(backend, encoded, vertical);
            auto cpu = initial(3, 3, 2), gpu = initial(3, 3, 2);
            int  code                         = encoded;
            cpu.configuration.lateral_bc_west = L(code % 5);
            code /= 5;
            cpu.configuration.lateral_bc_east = L(code % 5);
            code /= 5;
            cpu.configuration.lateral_bc_south = L(code % 5);
            code /= 5;
            cpu.configuration.lateral_bc_north          = L(code % 5);
            cpu.configuration.bottom_bc                 = B(vertical % 2);
            cpu.configuration.top_bc                    = T(vertical / 2);
            gpu.configuration                           = cpu.configuration;
            auto                        view            = make_simulation_state_view(cpu);
            auto                        scheme          = create_cartesian_cgrid_bc_scheme(view);
            const auto                  expected_energy = scheme->apply_full_attributed();
            FinalBoundaryEnergyReportV1 actual_energy;
            const bool                  reseed = (encoded + vertical) % 2 == 0;
            if (reseed)
                scheme->seed_acoustic_radiation();
            apply_native_cgrid_boundary(gpu, *kernels, reseed, &actual_energy);
            REQUIRE(valid_final_boundary_energy_report(actual_energy));
            CHECK(actual_energy.complete);
            CHECK(std::abs(actual_energy.committed_energy_change_j - expected_energy.committed_energy_change_j) <=
                  1e-6 + 3e-12 * std::abs(expected_energy.committed_energy_change_j));
            equal(cpu.fields, gpu.fields);
            equal(cpu.runtime.boundary_conditions, gpu.runtime.boundary_conditions);
        }
    for (int nz : {3, 17, 128})
        for (int bc = 0; bc < 5; ++bc)
        {
            auto cpu                           = initial(7, 6, nz, std::numeric_limits<float>::denorm_min()),
                 gpu                           = initial(7, 6, nz, std::numeric_limits<float>::denorm_min());
            cpu.configuration.lateral_bc_west  = L(bc);
            cpu.configuration.lateral_bc_east  = L((bc + 1) % 5);
            cpu.configuration.lateral_bc_south = L((bc + 2) % 5);
            cpu.configuration.lateral_bc_north = L((bc + 3) % 5);
            cpu.configuration.bottom_bc        = B::RigidSlip;
            cpu.configuration.top_bc           = T::Open;
            gpu.configuration                  = cpu.configuration;
            auto view                          = make_simulation_state_view(cpu);
            auto scheme                        = create_cartesian_cgrid_bc_scheme(view);
            scheme->apply_full();
            scheme->seed_acoustic_radiation();
            apply_native_cgrid_boundary(gpu, *kernels, true);
            equal(cpu.fields, gpu.fields);
            equal(cpu.runtime.boundary_conditions, gpu.runtime.boundary_conditions);
            omp_set_num_threads(4);
            scheme->apply_full();
            scheme->seed_acoustic_radiation();
            apply_native_cgrid_boundary(gpu, *kernels, true);
            omp_set_num_threads(1);
            equal(cpu.fields, gpu.fields);
            equal(cpu.runtime.boundary_conditions, gpu.runtime.boundary_conditions);
        }
    auto                             s       = initial(4, 5, 3);
    const auto                       saved   = s.fields;
    const BoundaryConditionWaveState history = s.runtime.boundary_conditions;
    Reject                           reject;
    REQUIRE_THROWS(apply_native_cgrid_boundary(s, reject, true));
    equal(saved, s.fields);
    equal(history, s.runtime.boundary_conditions);
    s.configuration.lateral_bc_west = L::LegacyOpen;
    REQUIRE_THROWS(apply_native_cgrid_boundary(s, *kernels, true));
    equal(saved, s.fields);
    equal(history, s.runtime.boundary_conditions);
    s.configuration.lateral_bc_west = L::Radiative;
    s.fields.p(3, 4, 2)             = std::numeric_limits<float>::quiet_NaN();
    const auto invalid              = s.fields;
    REQUIRE_THROWS(apply_native_cgrid_boundary(s, *kernels, true));
    equal(invalid, s.fields);
    equal(history, s.runtime.boundary_conditions);
    omp_set_num_threads(threads);
    CHECK(runtime.backend->successful_kernel_dispatch_count(nimbus::compute::KernelId::MappedTerrainBoundary) > 0);
    CHECK(runtime.backend->successful_kernel_dispatch_count(nimbus::compute::KernelId::MappedTerrainEos) == 0);
}
} // namespace
TEST_CASE("Native full-step C-grid boundary closures preserve CPU face ownership on Metal",
          "[metal][native_cgrid_boundary]")
{
    exercise("metal");
}
TEST_CASE("Native full-step C-grid boundary closures preserve CPU face ownership on Vulkan",
          "[vulkan][native_cgrid_boundary]")
{
    exercise("vulkan");
}
