#include "catch2/catch.hpp"
#include "compute/backend_runtime.hpp"
#include "core/runtime/simulation_state.hpp"
#include "core/runtime/native_earth_fixed_transport.hpp"
#include "diagnostics/momentum_budget.hpp"
#include "diagnostics/deterministic_reduction.hpp"
#include "dynamics/moving_boundary.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <omp.h>

namespace
{
std::array<double, 7> values(const tmv::MovingBoundaryLedger& l)
{
    return {l.initial_impulse,
            l.requested_impulse,
            l.final_impulse,
            l.storage_rounding,
            l.initial_kinetic,
            l.exact_kinetic,
            l.final_kinetic};
}
void ledger(const tmv::MovingBoundaryLedger& actual, const tmv::MovingBoundaryLedger& reference)
{
    const auto   a = values(actual), b = values(reference);
    const double momentum = std::max({std::abs(b[0]), std::abs(b[1]), std::abs(b[2]), 1e-100});
    const double energy   = std::max({std::abs(b[4]), std::abs(b[5]), std::abs(b[6]), 1e-100});
    for (int f = 0; f < 7; ++f)
    {
        CAPTURE(f, a[f], b[f], momentum, energy);
        const bool stored = f == 2 || f == 3 || f == 6;
        CHECK(std::abs(a[f] - b[f]) <= (stored ? 4e-7 : 3e-12) * (f < 4 ? momentum : energy) + 1e-100);
    }
    CHECK(std::abs(a[2] - a[1] - a[3]) <= 5e-12 * momentum + 1e-100);
}
void fields(const Field3D& actual, const Field3D& expected, double scale)
{
    REQUIRE(actual.size() == expected.size());
    for (std::size_t n = 0; n < actual.size(); ++n)
        CHECK(std::abs(double(actual.data()[n]) - expected.data()[n]) <=
              3e-7 * scale + std::numeric_limits<float>::denorm_min());
}
class RejectTopV final : public TerrainComputeKernels
{
public:
    explicit RejectTopV(TerrainComputeKernels& backend) : backend_(backend)
    {
    }
    bool supports_moving_momentum() const override
    {
        return backend_.supports_moving_momentum();
    }
    bool dispatch_moving_momentum(const MovingMomentumDispatch& d) override
    {
        if (d.constants.operation == 2 && d.constants.component == 1)
            return false;
        const bool ok = backend_.dispatch_moving_momentum(d);
        if (ok)
            ++calls;
        return ok;
    }
    int calls = 0;

private:
    TerrainComputeKernels& backend_;
};
void analytic(TerrainComputeKernels* kernels)
{
    using namespace tmv;
    constexpr int    nx = 6, ny = 5, nz = 64;
    TerrainMetrics3D g;
    g.dx = g.dy = g.dzeta = 1;
    g.J.resize(nx, ny, nz, 1);
    g.Gx_face.resize(nx, ny, nz + 1, 0);
    g.Gy_face.resize(nx, ny, nz + 1, 0);
    Field3D rho(nx, ny, nz, 1), u(nx, ny, nz, 2), v(nx, ny, nz, -3), w(nx, ny, nz, 2);
    for (int i = 0; i < nx; ++i)
        for (int j = 0; j < ny; ++j)
        {
            rho(i, j, nz - 1) = 3;
            u(i, j, nz - 1)   = 6;
            v(i, j, nz - 1)   = 1;
        }
    const float nan = std::numeric_limits<float>::quiet_NaN();
    rho(2, 2, 20) = u(2, 2, 20) = v(2, 2, 20) = w(2, 2, 20) = nan;
    g.J(2, 2, 20)                                           = std::numeric_limits<double>::quiet_NaN();
    const auto top =
        kernels ? project_moving_top_momentum(g, rho, u, v, *kernels) : project_moving_top_momentum(g, rho, u, v);
    for (int c = 0; c < 2; ++c)
    {
        const auto& field = c == 0 ? u : v;
        for (int i = c == 0 ? 0 : 1; i < nx - 1; ++i)
            for (int j = c == 1 ? 0 : 1; j < ny - 1; ++j)
            {
                CHECK(field(i, j, nz - 2) == (c == 0 ? 5 : 0));
                CHECK(field(i, j, nz - 1) == field(i, j, nz - 2));
            }
        CHECK(top[c].exact_kinetic - top[c].initial_kinetic == Approx(-6.0 * (nx - 2) * (ny - 2)).margin(1e-10));
        CHECK(top[c].storage_rounding == 0);
    }
    CHECK(std::isnan(u(2, 2, 20)));
    CHECK(std::isnan(v(2, 2, 20)));
    MovingGroundTargets targets;
    targets.impulse.resize(nx, ny, 1, 1);
    const auto correction = kernels ? correct_moving_ground_momentum(g, rho, u, v, w, targets, *kernels)
                                    : correct_moving_ground_momentum(g, rho, u, v, w, targets);
    for (int i = 1; i < nx - 1; ++i)
        for (int j = 1; j < ny - 1; ++j)
            CHECK(w(i, j, 0) == 3);
    CHECK(correction.final_impulse - correction.initial_impulse == Approx((nx - 2) * (ny - 2)).margin(1e-10));
    CHECK(correction.final_kinetic - correction.initial_kinetic == Approx(2.5 * (nx - 2) * (ny - 2)).margin(1e-10));
    CHECK(std::isnan(w(2, 2, 20)));
    TerrainContravariantFluxes flux;
    flux.x.resize(nx + 1, ny, nz, 0);
    flux.y.resize(nx, ny + 1, nz, 0);
    flux.z.resize(nx, ny, nz + 1, 0);
    flux.z(2, 2, 20) = std::numeric_limits<double>::quiet_NaN();
    for (double amount : {-2.0, 2.0})
    {
        for (int i = 0; i <= nx; ++i)
            for (int j = 0; j < ny; ++j)
                flux.x(i, j, 0) = amount;
        if (kernels)
            REQUIRE_THROWS(prepare_moving_ground_targets(g, rho, u, v, flux, *kernels));
        else
            REQUIRE_THROWS(prepare_moving_ground_targets(g, rho, u, v, flux));
    }
    for (int i = 0; i <= nx; ++i)
        for (int j = 0; j < ny; ++j)
            flux.x(i, j, 0) = 1;
    const auto zero = kernels ? prepare_moving_ground_targets(g, rho, u, v, flux, *kernels)
                              : prepare_moving_ground_targets(g, rho, u, v, flux);
    CHECK(zero.initial_impulse == 0);
    CHECK(zero.outward_impulse == 0);
    for (double value : zero.impulse.values)
        CHECK(value == 0);
    flux.z(1, 1, 1) = 1;
    if (kernels)
        REQUIRE_THROWS(prepare_moving_ground_targets(g, rho, u, v, flux, *kernels));
    else
        REQUIRE_THROWS(prepare_moving_ground_targets(g, rho, u, v, flux));
    const auto saved_w       = w;
    targets.impulse(2, 2, 0) = std::numeric_limits<double>::infinity();
    if (kernels)
        REQUIRE_THROWS(correct_moving_ground_momentum(g, rho, u, v, w, targets, *kernels));
    else
        REQUIRE_THROWS(correct_moving_ground_momentum(g, rho, u, v, w, targets));
    CHECK(std::memcmp(saved_w.data(), w.data(), w.size() * sizeof(float)) == 0);
    const auto saved_u = u, saved_v = v;
    auto       invalid    = rho;
    invalid(2, 2, nz - 1) = 0;
    if (kernels)
        REQUIRE_THROWS(project_moving_top_momentum(g, invalid, u, v, *kernels));
    else
        REQUIRE_THROWS(project_moving_top_momentum(g, invalid, u, v));
    CHECK(std::memcmp(saved_u.data(), u.data(), u.size() * sizeof(float)) == 0);
    CHECK(std::memcmp(saved_v.data(), v.data(), v.size() * sizeof(float)) == 0);
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
    analytic(kernels);
    constexpr int nx = 7, ny = 6;
    for (int nz : {2, 7, 128})
        for (double height : {0.0, 200.0})
            for (int sign : {-1, 1})
                for (bool tiny : {false, true})
                {
                    CAPTURE(backend, nz, height, sign, tiny);
                    SimulationStorageState s{{nx, ny, nz, 400, 2400.0 / nz, .2, 275}};
                    s.grid.geometry.initialize(
                        nx, ny, nz, 400, 2400.0 / nz, 275, CoordinateSystem::Cartesian, StaggerType::CGrid);
                    s.configuration.stagger_type     = StaggerType::CGrid;
                    auto& t                          = s.runtime.terrain;
                    t.config.scheme_id               = "bell";
                    t.config.coord_id                = "smoothed";
                    t.config.horizontal_placement_id = "earth_fixed";
                    t.config.ztop                    = 2400;
                    t.config.bell.h0                 = height;
                    t.config.bell.a                  = 850;
                    t.config.bell.axisymmetric       = true;
                    EarthFixedTerrainBuildRequest request;
                    request.nr                  = nx;
                    request.nth                 = ny;
                    request.nz                  = nz;
                    request.dx_m                = 400;
                    request.dy_m                = 275;
                    request.grid_origin_earth_m = {-800, -550};
                    request.sampling            = EarthFixedTerrainSampling::CellAverageBilinear;
                    auto initial                = prepare_earth_fixed_terrain_commit(t, request, *kernels);
                    REQUIRE(commit_earth_fixed_terrain(initial, t));
                    s.configuration.domain_frame.earth_origin_at_t0_m = request.grid_origin_earth_m;
                    s.configuration.domain_frame.velocity_ms          = {sign * 12.0, -sign * 8.0};
                    for (auto* f : {&s.fields.rho, &s.fields.u, &s.fields.v, &s.fields.w})
                        f->resize(nx, ny, nz, 0);
                    double speed = 0;
                    for (int i = 0; i < nx; ++i)
                        for (int j = 0; j < ny; ++j)
                            for (int k = 0; k < nz; ++k)
                            {
                                s.fields.rho(i, j, k) = 1.3f - .003f * k + .01f * i + .004f * j;
                                s.fields.u(i, j, k)   = tiny ? std::ldexp(float(1 + (i + j + k) % 7), -145)
                                                             : 8 + .3f * i + .2f * k + .1f * j;
                                s.fields.v(i, j, k)   = tiny ? -std::ldexp(float(1 + (2 * i + j + k) % 7), -145)
                                                             : -4 + .1f * i - .2f * k + .2f * j;
                                s.fields.w(i, j, k) =
                                    k == nz - 1
                                        ? 0
                                        : (tiny ? std::ldexp(float(1 + (i + j + 2 * k) % 7), -145) : .05f * (1 + k));
                                speed = std::max({speed,
                                                  std::abs(double(s.fields.u(i, j, k))),
                                                  std::abs(double(s.fields.v(i, j, k))),
                                                  std::abs(double(s.fields.w(i, j, k)))});
                            }
                    auto       state       = make_simulation_state_view(s);
                    const auto before      = compute_physical_momentum_budget_v2(state);
                    auto       prepared    = prepare_native_earth_fixed_transport(state, .2, *kernels);
                    const auto cpu_targets = prepare_moving_ground_targets(
                        t.metrics, s.fields.rho, s.fields.u, s.fields.v, prepared.mass_flux);
                    diagnostics::CompensatedSum target_sum, target_scale;
                    for (std::size_t n = 0; n < cpu_targets.impulse.size(); ++n)
                    {
                        const double a = prepared.ground_targets.impulse.values[n], b = cpu_targets.impulse.values[n];
                        CHECK(std::abs(a - b) <= 3e-12 * std::max(std::abs(b), 1e-100) + 1e-100);
                        target_sum.add(a);
                        target_scale.add(std::abs(b));
                    }
                    const double ground_scale = std::max({target_scale.value(),
                                                          std::abs(cpu_targets.initial_impulse),
                                                          std::abs(cpu_targets.outward_impulse),
                                                          1e-100});
                    CHECK(std::abs(prepared.ground_targets.outward_impulse - cpu_targets.outward_impulse) <=
                          3e-12 * ground_scale);
                    CHECK(std::abs(target_sum.value() - prepared.ground_targets.initial_impulse +
                                   prepared.ground_targets.outward_impulse) <= 5e-12 * ground_scale);
                    const auto& end   = prepared.motion.terrain.geometry.metrics;
                    auto        cpu_u = prepared.momentum.u, cpu_v = prepared.momentum.v;
                    const auto  top = project_moving_top_momentum(end, prepared.density, cpu_u, cpu_v);
                    fields(prepared.projected_u, cpu_u, speed);
                    fields(prepared.projected_v, cpu_v, speed);
                    for (int c = 0; c < 2; ++c)
                    {
                        ledger(prepared.top_projection[c], top[c]);
                        const auto& l = prepared.top_projection[c];
                        CHECK(std::abs(l.requested_impulse - l.initial_impulse) <=
                              3e-12 * std::max(std::abs(l.initial_impulse), 1e-100));
                        CHECK(l.exact_kinetic <= l.initial_kinetic * (1 + 3e-12) + 1e-100);
                    }
                    auto       gpu_w = prepared.momentum.w, cpu_w = gpu_w;
                    const auto correction = correct_moving_ground_momentum(end,
                                                                           prepared.density,
                                                                           prepared.projected_u,
                                                                           prepared.projected_v,
                                                                           gpu_w,
                                                                           prepared.ground_targets,
                                                                           *kernels);
                    const auto ref        = correct_moving_ground_momentum(end,
                                                                           prepared.density,
                                                                           prepared.projected_u,
                                                                           prepared.projected_v,
                                                                           cpu_w,
                                                                           prepared.ground_targets);
                    fields(gpu_w, cpu_w, speed);
                    ledger(correction, ref);
                    for (int i = 0; i < nx; ++i)
                        for (int j = 0; j < ny; ++j)
                            for (int k = 0; k < nz; ++k)
                            {
                                if (i == 0 || j == 0 || i == nx - 1 || j == ny - 1 || k != 0)
                                    CHECK(gpu_w(i, j, k) == prepared.momentum.w(i, j, k));
                                if (k < nz - 2)
                                {
                                    CHECK(prepared.projected_u(i, j, k) == prepared.momentum.u(i, j, k));
                                    CHECK(prepared.projected_v(i, j, k) == prepared.momentum.v(i, j, k));
                                }
                            }
                    if (nz == 7)
                    {
                        auto       u = prepared.momentum.u, v = prepared.momentum.v;
                        RejectTopV failure(*kernels);
                        REQUIRE_THROWS(project_moving_top_momentum(end, prepared.density, u, v, failure));
                        CHECK(failure.calls == 1);
                        CHECK(std::memcmp(u.data(), prepared.momentum.u.data(), u.size() * sizeof(float)) == 0);
                        CHECK(std::memcmp(v.data(), prepared.momentum.v.data(), v.size() * sizeof(float)) == 0);
                        auto bad = prepared.ground_targets;
                        bad.impulse.values.pop_back();
                        auto w = prepared.momentum.w;
                        REQUIRE_THROWS(correct_moving_ground_momentum(end, prepared.density, u, v, w, bad, *kernels));
                        CHECK(std::memcmp(w.data(), prepared.momentum.w.data(), w.size() * sizeof(float)) == 0);
                        bad                  = prepared.ground_targets;
                        bad.impulse(2, 2, 0) = std::numeric_limits<double>::max();
                        REQUIRE_THROWS(correct_moving_ground_momentum(end, prepared.density, u, v, w, bad, *kernels));
                        CHECK(std::memcmp(w.data(), prepared.momentum.w.data(), w.size() * sizeof(float)) == 0);
                        REQUIRE_THROWS(project_moving_top_momentum(end, prepared.density, u, u, *kernels));
                        const int threads = omp_get_max_threads();
                        omp_set_num_threads(4);
                        auto       repeat   = prepared.momentum.w;
                        const auto repeated = correct_moving_ground_momentum(end,
                                                                             prepared.density,
                                                                             prepared.projected_u,
                                                                             prepared.projected_v,
                                                                             repeat,
                                                                             prepared.ground_targets,
                                                                             *kernels);
                        omp_set_num_threads(threads);
                        CHECK(values(repeated) == values(correction));
                        CHECK(std::memcmp(repeat.data(), gpu_w.data(), repeat.size() * sizeof(float)) == 0);
                    }
                    REQUIRE(commit_earth_fixed_terrain(prepared.motion.terrain, t));
                    s.fields.rho                      = prepared.density;
                    s.fields.u                        = prepared.projected_u;
                    s.fields.v                        = prepared.projected_v;
                    s.fields.w                        = gpu_w;
                    const auto                  after = compute_physical_momentum_budget_v2(state);
                    const std::array<double, 3> p0{before.momentum_x, before.momentum_y, before.momentum_z},
                        p1{after.momentum_x, after.momentum_y, after.momentum_z};
                    const double scale =
                        (prepared.dry_mass.initial_inventory + prepared.dry_mass.final_inventory) * speed;
                    for (int c = 0; c < 3; ++c)
                    {
                        const double outward = prepared.momentum.ledger[c].outward_impulse +
                                               (c == 2 ? prepared.ground_targets.outward_impulse : 0);
                        const double rounding =
                            prepared.momentum.ledger[c].impulse_storage_rounding +
                            (c == 2 ? correction.storage_rounding : prepared.top_projection[c].storage_rounding);
                        CAPTURE(c, p0[c], p1[c], outward, rounding, scale);
                        CHECK(std::abs(p1[c] - p0[c] + outward - rounding) <= 2e-11 * scale + 1e-100);
                    }
                }
}
} // namespace
TEST_CASE("Native moving boundary projection and ground correction conserve momentum on Metal",
          "[metal][moving_boundary]")
{
    exercise("metal");
}
TEST_CASE("Native moving boundary projection and ground correction conserve momentum on Vulkan",
          "[vulkan][moving_boundary]")
{
    exercise("vulkan");
}

TEST_CASE("CPU moving boundary operators preserve analytic invariants and their plane ownership",
          "[cpu][moving_boundary]")
{
    analytic(nullptr);
}
