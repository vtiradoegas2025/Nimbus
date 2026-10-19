#include "catch2/catch.hpp"
#include "compute/backend_runtime.hpp"
#include "compute/reference_atmosphere_contract.hpp"
#include "core/runtime/native_earth_fixed_transport.hpp"
#include "core/runtime/simulation_state.hpp"
#include "dynamics/moving_environment.hpp"
#include "terrain/conservative_transport.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <omp.h>

namespace
{
using namespace tmv;
struct Fixture
{
    PrognosticFieldStorage fields;
    BaseStateStorage       base;
    TerrainMetrics3D       geometry;
    Fixture(int nx, int ny, int nz)
    {
        geometry.dzeta = 100;
        geometry.z.resize(nx, ny, nz, 0);
        for (auto* field : {&fields.rho,
                            &fields.theta,
                            &fields.p,
                            &fields.qv,
                            &fields.u,
                            &fields.v,
                            &fields.w,
                            &fields.qc,
                            &fields.qr,
                            &fields.qi,
                            &fields.qs,
                            &fields.qg,
                            &fields.qh,
                            &fields.tke,
                            &fields.tracer})
            field->resize(nx, ny, nz, 7);
        for (int k = 0; k < nz; ++k)
        {
            base.theta0.push_back(280 + 2 * k);
            base.p0.push_back(100000 - 500 * k);
            base.qv0.push_back(.008 - .00005 * k);
            base.u0.push_back(-8 + .3 * k);
            base.v0.push_back(3 - .2 * k);
        }
        for (int i = 0; i < nx; ++i)
            for (int j = 0; j < ny; ++j)
                for (int k = 0; k < nz; ++k)
                    geometry.z(i, j, k) = (k + .5) * 100 + 11 * i - 7 * j;
    }
};
bool identical(const Field3D& a, const Field3D& b)
{
    return a.size() == b.size() && (a.empty() || std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0);
}
void unchanged(const PrognosticFieldStorage& a, const PrognosticFieldStorage& b)
{
    const std::array<const Field3D*, 15> lhs{
        &a.rho, &a.theta, &a.p, &a.qv, &a.u, &a.v, &a.w, &a.qc, &a.qr, &a.qi, &a.qs, &a.qg, &a.qh, &a.tke, &a.tracer};
    const std::array<const Field3D*, 15> rhs{
        &b.rho, &b.theta, &b.p, &b.qv, &b.u, &b.v, &b.w, &b.qc, &b.qr, &b.qi, &b.qs, &b.qg, &b.qh, &b.tke, &b.tracer};
    for (std::size_t n = 0; n < lhs.size(); ++n)
        CHECK(identical(*lhs[n], *rhs[n]));
}
void fill(Fixture& f, TerrainComputeKernels* kernels)
{
    if (kernels)
        fill_moving_environment(f.fields, f.base, f.geometry, *kernels);
    else
        fill_moving_environment(f.fields, f.base, f.geometry);
}
void analytic(TerrainComputeKernels* kernels)
{
    for (int nz : {1, 2, 17, 128})
    {
        Fixture    f(7, 6, nz);
        const auto before = f.fields;
        fill(f, kernels);
        for (int i = 0; i < 7; ++i)
            for (int j = 0; j < 6; ++j)
                for (int k = 0; k < nz; ++k)
                {
                    CAPTURE(nz, i, j, k);
                    const bool   ghost = i == 0 || i == 6 || j == 0 || j == 5;
                    const double x     = nz == 1 ? 0 : f.geometry.z(i, j, k) / 100 - .5;
                    const float  theta = float(280 + 2 * x), p = float(100000 - 500 * x),
                                 vapor = float(.008 - .00005 * x);
                    const float  u = float(-8 + .3 * x), v = float(3 - .2 * x);
                    const double rho = p / (287 * theta * std::pow(p / 100000.0, 287.0 / 1004.0));
                    CHECK(f.fields.theta(i, j, k) == (ghost ? theta : 7));
                    CHECK(f.fields.p(i, j, k) == (ghost ? p : 7));
                    CHECK(f.fields.qv(i, j, k) == Approx(ghost ? vapor : 7).epsilon(2e-7));
                    CHECK(f.fields.rho(i, j, k) == Approx(ghost ? rho : 7).epsilon(2e-7));
                    CHECK(f.fields.u(i, j, k) == Approx(ghost && i != 0 ? u : 7).margin(3e-6));
                    CHECK(f.fields.v(i, j, k) == Approx(ghost && j != 0 ? v : 7).margin(3e-6));
                    for (const auto* field : {&f.fields.w,
                                              &f.fields.qc,
                                              &f.fields.qr,
                                              &f.fields.qi,
                                              &f.fields.qs,
                                              &f.fields.qg,
                                              &f.fields.qh,
                                              &f.fields.tke,
                                              &f.fields.tracer})
                        CHECK((*field)(i, j, k) == (ghost ? 0 : 7));
                }
        auto cpu   = f;
        cpu.fields = before;
        fill(cpu, nullptr);
        for (std::size_t n = 0; n < f.fields.theta.size(); ++n)
        {
            CHECK(f.fields.theta.data()[n] == cpu.fields.theta.data()[n]);
            CHECK(f.fields.p.data()[n] == cpu.fields.p.data()[n]);
            CHECK(f.fields.rho.data()[n] == Approx(cpu.fields.rho.data()[n]).epsilon(2e-7));
        }
        const auto saved           = f.fields;
        f.geometry.z(6, 5, nz - 1) = std::numeric_limits<double>::quiet_NaN();
        REQUIRE_THROWS(fill(f, kernels));
        unchanged(f.fields, saved);
    }
    Fixture    bad(4, 4, 3);
    const auto saved        = bad.fields;
    bad.geometry.z(3, 3, 2) = 1e8;
    REQUIRE_THROWS(fill(bad, kernels));
    unchanged(bad.fields, saved);
    bad.geometry.z(3, 3, 2) = 250;
    bad.base.qv0[1]         = -1;
    REQUIRE_THROWS(fill(bad, kernels));
    unchanged(bad.fields, saved);
    bad.base.qv0[1] = .005;
    bad.base.theta0.pop_back();
    REQUIRE_THROWS(fill(bad, kernels));
    unchanged(bad.fields, saved);
    Fixture optional(3, 3, 1);
    optional.fields.tke    = Field3D{};
    optional.fields.tracer = Field3D{};
    optional.fields.qc     = Field3D{};
    optional.fields.qh     = Field3D{};
    fill(optional, kernels);
    CHECK(optional.fields.tke.empty());
    CHECK(optional.fields.tracer.empty());
    CHECK(optional.fields.qc.empty());
    CHECK(optional.fields.qh.empty());
}
class Capture final : public TerrainComputeKernels
{
public:
    TerrainComputeKernels&                                 backend;
    nimbus::compute::abi::ReferenceAtmospherePushConstants constants{};
    std::vector<float>                                     input;
    bool                                                   fail = false;
    explicit Capture(TerrainComputeKernels& kernels) : backend(kernels)
    {
    }
    bool supports_reference_atmosphere() const override
    {
        return true;
    }
    bool dispatch_reference_atmosphere(const ReferenceAtmosphereDispatch& d) override
    {
        constants = d.constants;
        input.assign(d.input, d.input + nimbus::compute::reference_atmosphere::input_count(d));
        if (fail)
            return false;
        return backend.dispatch_reference_atmosphere(d);
    }
};
void composed(TerrainComputeKernels& kernels)
{
    for (const char* scheme : {"bell", "schar"})
    {
        CAPTURE(scheme);
        constexpr int          nx = 7, ny = 6, nz = 5;
        Fixture                fixture(nx, ny, nz);
        SimulationStorageState state{{nx, ny, nz, 400, 350, .2, 275}};
        state.grid.geometry.initialize(nx, ny, nz, 400, 350, 275, CoordinateSystem::Cartesian, StaggerType::CGrid);
        state.configuration.stagger_type = StaggerType::CGrid;
        state.fields                     = fixture.fields;
        state.base                       = fixture.base;
        state.fields.rho.fill(1);
        state.fields.theta.fill(300);
        state.fields.p.fill(100000);
        state.fields.qv.fill(.008f);
        state.fields.u.fill(8);
        state.fields.v.fill(-3);
        for (auto* field : {&state.fields.w,
                            &state.fields.qc,
                            &state.fields.qr,
                            &state.fields.qi,
                            &state.fields.qs,
                            &state.fields.qg,
                            &state.fields.qh,
                            &state.fields.tke,
                            &state.fields.tracer})
            field->fill(0);
        auto& terrain                          = state.runtime.terrain;
        terrain.config.scheme_id               = scheme;
        terrain.config.coord_id                = "smoothed";
        terrain.config.horizontal_placement_id = "earth_fixed";
        terrain.config.ztop                    = nz * 350;
        terrain.config.bell.h0                 = 200;
        terrain.config.bell.a                  = 850;
        terrain.config.bell.axisymmetric       = true;
        terrain.config.schar.H                 = 150;
        terrain.config.schar.a                 = 850;
        terrain.config.schar.ell               = 500;
        EarthFixedTerrainBuildRequest request;
        request.nr                  = nx;
        request.nth                 = ny;
        request.nz                  = nz;
        request.dx_m                = 400;
        request.dy_m                = 275;
        request.grid_origin_earth_m = {-800, -550};
        request.sampling            = EarthFixedTerrainSampling::CellAverageBilinear;
        auto prepared               = prepare_earth_fixed_terrain_commit(terrain, request, kernels);
        REQUIRE(commit_earth_fixed_terrain(prepared, terrain));
        state.configuration.domain_frame.earth_origin_at_t0_m = request.grid_origin_earth_m;
        state.configuration.domain_frame.velocity_ms          = {8, -5};
        const auto compare                                    = [&](const TerrainMetrics3D& geometry)
        {
            auto cpu = state.fields;
            fill_moving_environment(cpu, state.base, geometry);
            fill_moving_environment(state.fields, state.base, geometry, kernels);
            for (std::size_t n = 0; n < cpu.rho.size(); ++n)
            {
                CHECK(state.fields.theta.data()[n] == Approx(cpu.theta.data()[n]).epsilon(2e-7));
                CHECK(state.fields.p.data()[n] == Approx(cpu.p.data()[n]).epsilon(2e-7));
                CHECK(state.fields.rho.data()[n] == Approx(cpu.rho.data()[n]).epsilon(2e-7));
                CHECK(state.fields.qv.data()[n] == Approx(cpu.qv.data()[n]).epsilon(2e-7));
                CHECK(state.fields.u.data()[n] == Approx(cpu.u.data()[n]).margin(2e-6));
                CHECK(state.fields.v.data()[n] == Approx(cpu.v.data()[n]).margin(2e-6));
            }
        };
        compare(terrain.metrics);
        const auto saved     = state.fields;
        auto       view      = make_simulation_state_view(state);
        const auto transport = prepare_native_earth_fixed_transport(view, .2, kernels);
        const auto energy    = prepare_native_earth_fixed_energy(state, transport, kernels);
        CHECK(std::abs(energy.transport.final_inventory - energy.transport.initial_inventory +
                       energy.transport.boundary_outflow - energy.transport.storage_rounding) <=
              5e-12 * std::max(std::abs(energy.transport.initial_inventory), 1.0));
        unchanged(state.fields, saved);
        compare(transport.motion.terrain.geometry.metrics);
    }
}

void hardware(const char* backend)
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
    REQUIRE(kernels->supports_reference_atmosphere());
    analytic(kernels);
    composed(*kernels);
    Fixture f(5, 4, 3);
    Capture capture(*kernels);
    fill(f, &capture);
    const auto first   = f.fields;
    const int  threads = omp_get_max_threads();
    omp_set_num_threads(4);
    fill(f, kernels);
    omp_set_num_threads(threads);
    unchanged(first, f.fields);
    capture.fail = true;
    REQUIRE_THROWS(fill(f, &capture));
    unchanged(first, f.fields);
    TerrainComputeKernels unavailable;
    REQUIRE_THROWS(fill(f, &unavailable));
    unchanged(first, f.fields);
    for (double magnitude : {std::ldexp(1.0, -149), std::ldexp(1.0, -140), std::ldexp(1.0, -126), 1e-100})
    {
        Fixture tiny(3, 3, 2);
        tiny.base.qv0 = {magnitude, 3 * magnitude};
        tiny.base.u0  = {-magnitude, -3 * magnitude};
        tiny.base.v0  = {magnitude, 3 * magnitude};
        tiny.geometry.z.resize(3, 3, 2, 100);
        fill(tiny, kernels);
        CHECK(tiny.fields.qv(0, 0, 0) == float(2 * magnitude));
        CHECK(tiny.fields.u(2, 0, 0) == float(-2 * magnitude));
        CHECK(tiny.fields.v(0, 2, 0) == float(2 * magnitude));
    }
    ReferenceAtmosphereDispatch d;
    d.constants = capture.constants;
    d.input     = capture.input.data();
    std::vector<float> output(std::size_t(d.constants.count) * 6, 19);
    d.output            = output.data();
    const auto rejected = [&]()
    {
        CHECK_FALSE(kernels->dispatch_reference_atmosphere(d));
        for (float value : output)
            CHECK(value == 19);
    };
    d.constants.count = std::numeric_limits<int>::max();
    rejected();
    d.constants = capture.constants;
    d.input     = nullptr;
    rejected();
    d.input  = capture.input.data();
    d.output = capture.input.data() + 1;
    rejected();
    d.output         = output.data();
    capture.input[0] = -1;
    rejected();
    capture.input[0]     = 1;
    const auto n         = 10 * std::size_t(d.constants.levels) + 2 * std::size_t(d.constants.count - 1);
    capture.input[n]     = 1e8;
    capture.input[n + 1] = 0;
    rejected();
    CHECK(runtime.backend->successful_kernel_dispatch_count(nimbus::compute::KernelId::ReferenceAtmosphere) > 0);
}
} // namespace
TEST_CASE("Moving environment fills only physical ghost owners atomically on CPU", "[cpu][moving_environment]")
{
    analytic(nullptr);
    using terrain_transport::reference_profile_value_fp32;
    CHECK(reference_profile_value_fp32({3, 3}, 1e100, 1) == 3);
    REQUIRE_THROWS(reference_profile_value_fp32({3, 4}, 1e100, 1));
    REQUIRE_THROWS(reference_profile_value_fp32({3, 4}, 1e308, 1e-100));
    REQUIRE_THROWS(reference_profile_value_fp32({1e100}, 0, 1));
}
TEST_CASE("Moving environment samples native terrain heights on Metal", "[metal][moving_environment]")
{
    hardware("metal");
}
TEST_CASE("Moving environment samples native terrain heights on Vulkan", "[vulkan][moving_environment]")
{
    hardware("vulkan");
}
