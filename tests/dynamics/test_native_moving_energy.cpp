#include "catch2/catch.hpp"
#include "compute/backend_runtime.hpp"
#include "compute/moving_energy_contract.hpp"
#include "core/runtime/native_earth_fixed_transport.hpp"
#include "core/runtime/simulation_state.hpp"
#include "core/infra/physical_constants.hpp"
#include "dynamics/eos_pressure.hpp"
#include "dynamics/moving_energy.hpp"
#include "terrain/native_conservative_remap.hpp"
#include "diagnostics/deterministic_reduction.hpp"
#include <cmath>
#include <cstring>
#include <limits>
#include <omp.h>

namespace
{
void allocate(PrognosticFieldStorage& f, int nx, int ny, int nz)
{
    for (auto* field :
         {&f.rho, &f.theta, &f.p, &f.u, &f.v, &f.w, &f.qv, &f.qc, &f.qr, &f.qi, &f.qs, &f.qg, &f.qh, &f.tke})
        field->resize(nx, ny, nz, 0);
}
class Capture final : public TerrainComputeKernels
{
public:
    explicit Capture(TerrainComputeKernels& backend) : backend_(backend)
    {
    }
    bool supports_moving_energy() const override
    {
        return backend_.supports_moving_energy();
    }
    bool dispatch_moving_energy(const MovingEnergyDispatch& d) override
    {
        constants = d.constants;
        input.assign(d.input, d.input + nimbus::compute::moving_energy::cells(d) * MovingEnergyDispatch::input_stride);
        return backend_.dispatch_moving_energy(d);
    }
    nimbus::compute::abi::MovingEnergyPushConstants constants{};
    std::vector<float>                              input;

private:
    TerrainComputeKernels& backend_;
};
void analytic(TerrainComputeKernels* kernels)
{
    using namespace tmv;
    constexpr int    nx = 4, ny = 4, nz = 3;
    TerrainMetrics3D g;
    g.dx    = 2;
    g.dy    = 3;
    g.dzeta = 4;
    g.J.resize(nx, ny, nz, 1);
    g.z.resize(nx, ny, nz, 123.5);
    g.Gx_face.resize(nx, ny, nz + 1, -.1);
    g.Gy_face.resize(nx, ny, nz + 1, .2);
    PrognosticFieldStorage f;
    allocate(f, nx, ny, nz);
    f.rho.fill(1);
    f.theta.fill(300);
    f.p.fill(100000);
    f.u.fill(8);
    f.v.fill(-4);
    f.w.fill(.5);
    f.qv.fill(.012f);
    f.qc.fill(.001f);
    f.qr.fill(.002f);
    f.qi.fill(.0003f);
    f.qs.fill(.0004f);
    f.qg.fill(.0005f);
    f.qh.fill(.0006f);
    f.tke.fill(.4f);
    const auto   values = kernels ? moving_energy_density(f, g, *kernels) : moving_energy_density(f, g);
    const double vapor = f.qv(0, 0, 0), liquid = double(f.qc(0, 0, 0)) + f.qr(0, 0, 0);
    const double ice      = double(f.qi(0, 0, 0)) + f.qs(0, 0, 0) + f.qg(0, 0, 0) + f.qh(0, 0, 0);
    const double capacity = 717 + vapor * (1850 - 461.5) + liquid * 4186 + ice * 2106;
    for (int i = 0; i < nx; ++i)
        for (int j = 0; j < ny; ++j)
            for (int k = 0; k < nz; ++k)
            {
                const double kinetic  = k == 0 ? 40.7025 : 40.125;
                const double expected = capacity * (300 - 273.15) +
                                        (1 + vapor + liquid + ice) * (kinetic + 9.81 * 123.5) +
                                        vapor * (2.5e6 - 461.5 * 273.15) - ice * 3.34e5 + f.tke(i, j, k);
                CHECK(values[(i * ny + j) * nz + k] == Approx(expected).margin(3e-6));
            }
    auto target       = values;
    target[0]         = std::numeric_limits<double>::quiet_NaN();
    const auto before = f;
    const auto report =
        kernels ? reconstruct_moving_energy(f, g, target, *kernels) : reconstruct_moving_energy(f, g, target);
    CHECK(std::abs(report.stored_energy_j - report.requested_energy_j - report.storage_rounding_j) < 1e-5);
    for (int i = 0; i < nx; ++i)
        for (int j = 0; j < ny; ++j)
            for (int k = 0; k < nz; ++k)
            {
                if (i == 0 || j == 0 || i == nx - 1 || j == ny - 1)
                {
                    CHECK(f.theta(i, j, k) == before.theta(i, j, k));
                    CHECK(f.p(i, j, k) == before.p(i, j, k));
                }
                else
                {
                    const double temperature = f.theta(i, j, k) * std::pow(f.p(i, j, k) / 100000.0, 287.0 / 1004.0);
                    CHECK(temperature == Approx(300).epsilon(3e-7));
                    CHECK(f.p(i, j, k) == Approx(86100).epsilon(3e-7));
                }
            }
    const auto saved              = f;
    target[(2 * ny + 2) * nz + 1] = -1e8;
    if (kernels)
        REQUIRE_THROWS(reconstruct_moving_energy(f, g, target, *kernels));
    else
        REQUIRE_THROWS(reconstruct_moving_energy(f, g, target));
    CHECK(std::memcmp(f.theta.data(), saved.theta.data(), f.theta.size() * sizeof(float)) == 0);
    CHECK(std::memcmp(f.p.data(), saved.p.data(), f.p.size() * sizeof(float)) == 0);
    f.qi(1, 1, 0) = -.01f;
    if (kernels)
        REQUIRE_THROWS(moving_energy_density(f, g, *kernels));
    else
        REQUIRE_THROWS(moving_energy_density(f, g));
    f    = saved;
    g.dx = std::numeric_limits<double>::min();
    g.J.resize(nx, ny, nz, std::numeric_limits<double>::min());
    if (kernels)
        REQUIRE_THROWS(moving_energy_density(f, g, *kernels));
    else
        REQUIRE_THROWS(moving_energy_density(f, g));
}
void exercise(const char* backend)
{
    using namespace tmv;
    ComputeBackendRuntimeState runtime;
    runtime.config.backend        = backend;
    runtime.config.allow_fallback = false;
    std::string error;
    REQUIRE(initialize_compute_backend_runtime(runtime, CoordinateSystem::Cartesian, error));
    INFO(error);
    REQUIRE_FALSE(runtime.fallback_active);
    auto* kernels = runtime.backend->terrain_kernels();
    REQUIRE(kernels);
    REQUIRE(kernels->supports_moving_energy());
    analytic(kernels);
    Capture       capture(*kernels);
    constexpr int nx = 7, ny = 6, nz = 5;
    for (int phase : {0, 1, 2, 3})
        for (double theta : {210.0, 300.0, 650.0})
            for (double height : {0.0, 200.0})
            {
                CAPTURE(backend, phase, theta, height);
                SimulationStorageState s{{nx, ny, nz, 400, 350, .2, 275}};
                s.grid.geometry.initialize(nx, ny, nz, 400, 350, 275, CoordinateSystem::Cartesian, StaggerType::CGrid);
                s.configuration.stagger_type           = StaggerType::CGrid;
                auto& terrain                          = s.runtime.terrain;
                terrain.config.scheme_id               = "bell";
                terrain.config.coord_id                = "smoothed";
                terrain.config.horizontal_placement_id = "earth_fixed";
                terrain.config.ztop                    = nz * 350;
                terrain.config.bell.h0                 = height;
                terrain.config.bell.a                  = 850;
                terrain.config.bell.axisymmetric       = true;
                EarthFixedTerrainBuildRequest request;
                request.nr                  = nx;
                request.nth                 = ny;
                request.nz                  = nz;
                request.dx_m                = 400;
                request.dy_m                = 275;
                request.grid_origin_earth_m = {-800, -550};
                request.sampling            = EarthFixedTerrainSampling::CellAverageBilinear;
                auto geometry               = prepare_earth_fixed_terrain_commit(terrain, request, *kernels);
                REQUIRE(commit_earth_fixed_terrain(geometry, terrain));
                s.configuration.domain_frame.earth_origin_at_t0_m = request.grid_origin_earth_m;
                s.configuration.domain_frame.velocity_ms          = {8, -5};
                allocate(s.fields, nx, ny, nz);
                for (int i = 0; i < nx; ++i)
                    for (int j = 0; j < ny; ++j)
                        for (int k = 0; k < nz; ++k)
                        {
                            auto& f          = s.fields;
                            f.rho(i, j, k)   = 1.2f - .08f * k + .004f * i - .002f * j;
                            f.theta(i, j, k) = float(theta + .1 * i - .2 * k);
                            f.p(i, j, k)     = float(dynamics_eos::pressure_fp64(f.rho(i, j, k), f.theta(i, j, k)));
                            f.u(i, j, k)     = 8 + .2f * i - .03f * k;
                            f.v(i, j, k)     = -3 + .1f * j;
                            f.w(i, j, k)     = k == nz - 1 ? 0 : .03f * (i + j + k);
                            f.qv(i, j, k)    = phase == 0 ? 0 : .005f + .0001f * k;
                            f.qc(i, j, k)    = phase < 2 ? 0 : .001f + .00001f * i;
                            f.qr(i, j, k)    = phase < 2 ? 0 : .002f + .00001f * j;
                            f.qi(i, j, k)    = phase < 3 ? 0 : .0003f;
                            f.qs(i, j, k)    = phase < 3 ? 0 : .0004f;
                            f.qg(i, j, k)    = phase < 3 ? 0 : .0005f;
                            f.qh(i, j, k)    = phase < 3 ? 0 : .0006f;
                            f.tke(i, j, k)   = .2f + .01f * k;
                        }
                const auto expected = moving_energy_density(s.fields, terrain.metrics);
                const auto observed = moving_energy_density(s.fields, terrain.metrics, capture);
                for (std::size_t n = 0; n < expected.size(); ++n)
                    CHECK(std::abs(observed[n] - expected[n]) <= 3e-12 * std::max(std::abs(expected[n]), 1e6));
                const auto source    = s.fields;
                auto       view      = make_simulation_state_view(s);
                auto       transport = prepare_native_earth_fixed_transport(view, .2, *kernels);
                auto       energy    = prepare_native_earth_fixed_energy(s, transport, *kernels);
                CHECK(std::abs(energy.transport.final_inventory - energy.transport.initial_inventory +
                               energy.transport.boundary_outflow - energy.transport.storage_rounding) <=
                      5e-12 * std::max(std::abs(energy.transport.initial_inventory), 1.0));
                auto        next = s.fields;
                const auto& g0   = terrain.metrics;
                const auto& g1   = transport.motion.terrain.geometry.metrics;
                for (auto* field : {&next.qv, &next.qc, &next.qr, &next.qi, &next.qs, &next.qg, &next.qh, &next.tke})
                    terrain_native::remap_inventory(*kernels,
                                                    *field,
                                                    *field,
                                                    s.fields.rho,
                                                    transport.density,
                                                    g0,
                                                    g1,
                                                    transport.motion.swept_volumes,
                                                    true);
                next.rho = transport.density;
                next.u   = transport.projected_u;
                next.v   = transport.projected_v;
                next.w   = transport.momentum.w;
                correct_moving_ground_momentum(
                    g1, next.rho, next.u, next.v, next.w, transport.ground_targets, *kernels);
                auto                        reference = next;
                const auto                  cpu       = reconstruct_moving_energy(reference, g1, energy.target_density);
                const auto                  gpu = reconstruct_moving_energy(next, g1, energy.target_density, *kernels);
                const auto                  actual = moving_energy_density(next, g1);
                diagnostics::CompensatedSum inventory, scale;
                for (int i = 0; i < nx; ++i)
                    for (int j = 0; j < ny; ++j)
                        for (int k = 0; k < nz; ++k)
                        {
                            CHECK(next.theta(i, j, k) == Approx(reference.theta(i, j, k)).epsilon(2e-7));
                            CHECK(next.p(i, j, k) == Approx(reference.p(i, j, k)).epsilon(3e-7));
                            if (i > 0 && i < nx - 1 && j > 0 && j < ny - 1)
                            {
                                const auto   n      = (std::size_t(i) * ny + j) * nz + k;
                                const double volume = g1.dx * g1.dy * g1.dzeta * g1.J(i, j, k);
                                inventory.add(actual[n] * volume);
                                scale.add(std::max(std::abs(actual[n]), 1e6) * volume);
                            }
                            else
                            {
                                CHECK(next.theta(i, j, k) == source.theta(i, j, k));
                                CHECK(next.p(i, j, k) == source.p(i, j, k));
                            }
                        }
                CHECK(std::abs(inventory.value() - gpu.stored_energy_j) <= 5e-12 * scale.value());
                CHECK(std::abs(gpu.stored_energy_j - gpu.requested_energy_j - gpu.storage_rounding_j) <=
                      5e-12 * scale.value());
                CHECK(std::abs(gpu.stored_energy_j - cpu.stored_energy_j) <= 3e-7 * scale.value());
                CHECK(std::memcmp(s.fields.theta.data(), source.theta.data(), source.theta.size() * sizeof(float)) ==
                      0);
                if (phase == 3 && height > 0)
                {
                    auto repeated     = s.fields;
                    repeated.rho      = next.rho;
                    repeated.u        = next.u;
                    repeated.v        = next.v;
                    repeated.w        = next.w;
                    repeated.qv       = next.qv;
                    repeated.qc       = next.qc;
                    repeated.qr       = next.qr;
                    repeated.qi       = next.qi;
                    repeated.qs       = next.qs;
                    repeated.qg       = next.qg;
                    repeated.qh       = next.qh;
                    repeated.tke      = next.tke;
                    const int threads = omp_get_max_threads();
                    omp_set_num_threads(4);
                    const auto report = reconstruct_moving_energy(repeated, g1, energy.target_density, *kernels);
                    omp_set_num_threads(threads);
                    CHECK(report.stored_energy_j == gpu.stored_energy_j);
                    CHECK(std::memcmp(repeated.theta.data(), next.theta.data(), next.theta.size() * sizeof(float)) ==
                          0);
                    auto stale = transport;
                    stale.motion.terrain.expected_placement.generation++;
                    REQUIRE_THROWS(prepare_native_earth_fixed_energy(s, stale, *kernels));
                }
            }
    MovingEnergyDispatch d;
    d.constants              = capture.constants;
    d.input                  = capture.input.data();
    const auto         count = nimbus::compute::moving_energy::cells(d);
    std::vector<float> output(count * MovingEnergyDispatch::output_stride, 17);
    d.output            = output.data();
    const auto rejected = [&]()
    {
        CHECK_FALSE(kernels->dispatch_moving_energy(d));
        for (float value : output)
            CHECK(value == 17);
    };
    const int saved = d.constants.nx;
    d.constants.nx  = std::numeric_limits<int>::max();
    rejected();
    d.constants.nx = saved;
    d.input        = nullptr;
    rejected();
    d.input  = capture.input.data();
    d.output = capture.input.data() + 1;
    rejected();
    d.output         = output.data();
    const float rho  = capture.input[0];
    capture.input[0] = 0;
    rejected();
    capture.input[0] = rho;
    capture.input[6] = -1;
    rejected();
    capture.input[6] = 0;
    capture.input[3] = std::numeric_limits<float>::max();
    rejected();
    CHECK(runtime.backend->successful_kernel_dispatch_count(nimbus::compute::KernelId::MovingEnergy) > 0);
}
} // namespace
TEST_CASE("Moving moist energy reconstruction is native and atomic on Metal", "[metal][moving_energy]")
{
    exercise("metal");
}
TEST_CASE("Moving moist energy reconstruction is native and atomic on Vulkan", "[vulkan][moving_energy]")
{
    exercise("vulkan");
}
TEST_CASE("CPU moving moist energy matches the analytic reference and rejects atomically", "[cpu][moving_energy]")
{
    analytic(nullptr);
}
