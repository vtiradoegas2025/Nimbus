#include "catch2/catch.hpp"
#include "compute/backend_runtime.hpp"
#include "compute/implicit_diffusion_solve.hpp"
#include "compute/kernel_manifest.hpp"
#include "numerics/diffusion/schemes/implicit/implicit.hpp"
#include "numerics/diffusion/schemes/explicit/explicit.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace
{
void exercise(const char* backend_name)
{
    ComputeBackendRuntimeState runtime;
    runtime.config.backend        = backend_name;
    runtime.config.allow_fallback = false;
    std::string error;
    REQUIRE(initialize_compute_backend_runtime(runtime, CoordinateSystem::Cartesian, error));
    auto* kernels = runtime.backend->diffusion_kernels();
    REQUIRE(kernels != nullptr);
    REQUIRE(kernels->supports(DiffusionKernelOp::Implicit));
    constexpr int      nx = 8, ny = 7, nz = 6;
    constexpr double   pi = 3.14159265358979323846;
    Field3D            field(nx, ny, nz), rho(nx, ny, nz, 1.0f), tendency(nx, ny, nz, 0.0f);
    std::vector<float> dz(nz, 2.0f);
    DiffusionDispatch  dispatch;
    dispatch.field    = field.data();
    dispatch.rho      = rho.data();
    dispatch.dz       = dz.data();
    dispatch.tendency = tendency.data();
    dispatch.nx       = nx;
    dispatch.ny       = ny;
    dispatch.nz       = nz;
    dispatch.dx       = 2;
    dispatch.dy       = 3;
    dispatch.c_grid   = true;
    for (bool periodic : {false, true})
        for (int location = 0; location < 4; ++location)
            for (float strength : {0.05f, 10.0f, 100.0f})
                for (float amplitude : {1.0f, 1e-12f})
                {
                    INFO(backend_name);
                    INFO(periodic);
                    INFO(location);
                    INFO(strength);
                    INFO(amplitude);
                    dispatch.periodic_x     = periodic;
                    dispatch.periodic_y     = periodic;
                    dispatch.location       = static_cast<DiffusionFieldLocation>(location);
                    dispatch.K_h            = strength * 4;
                    dispatch.K_v            = strength;
                    const int    physical_x = nx - (location == 1 && !periodic ? 3 : 2);
                    const double frequency  = (periodic ? 2 : 1) * pi / physical_x;
                    for (int i = 0; i < nx; ++i)
                        for (int j = 0; j < ny; ++j)
                            for (int k = 0; k < nz; ++k)
                                field(i, j, k) = static_cast<float>(amplitude * std::cos(frequency * (i - 0.5)));
                    nimbus::compute::ImplicitDiffusionSolveReport report;
                    REQUIRE(nimbus::compute::solve_implicit_diffusion(*kernels, dispatch, 1.0f, report));
                    REQUIRE(report.iterations > 0);
                    CHECK(report.final_residual_squared <= report.initial_residual_squared * 1e-12);
                    const double decay       = 1.0 / (1.0 + 4.0 * strength * std::pow(std::sin(frequency / 2), 2));
                    const int    physical_y  = ny - (location == 2 && !periodic ? 3 : 2);
                    const int    levels      = location == 0 ? nz : nz - 1;
                    double       mass_change = 0.0;
                    for (int i = 1; i <= physical_x; ++i)
                        for (int j = 1; j <= physical_y; ++j)
                            for (int k = 0; k < levels; ++k)
                            {
                                const double result = field(i, j, k) + tendency(i, j, k);
                                CHECK(std::abs(result - decay * field(i, j, k)) <= 3e-5 * amplitude);
                                mass_change += tendency(i, j, k);
                            }
                    CHECK(std::abs(mass_change) <= 2e-4 * amplitude);
                }

    const auto                                    before = tendency;
    nimbus::compute::ImplicitDiffusionSolveReport report{99, 42, 43};
    dispatch.tendency = field.data() + 1;
    CHECK_FALSE(nimbus::compute::solve_implicit_diffusion(*kernels, dispatch, 1, report));
    CHECK(report.iterations == 99);
    dispatch.tendency = tendency.data();
    rho(2, 2, 2)      = -1;
    CHECK_FALSE(nimbus::compute::solve_implicit_diffusion(*kernels, dispatch, 1, report));
    CHECK(std::equal(before.data(), before.data() + before.size(), tendency.data()));
    CHECK(report.iterations == 99);
    rho(2, 2, 2) = 1;
    for (int i = 0; i < nx; ++i)
        for (int j = 0; j < ny; ++j)
            for (int k = 0; k < nz; ++k)
                field(i, j, k) = static_cast<float>(std::sin(i + 0.7 * j + 0.3 * k));
    CHECK_FALSE(nimbus::compute::solve_implicit_diffusion(*kernels, dispatch, 1, report, 1));
    CHECK(std::equal(before.data(), before.data() + before.size(), tendency.data()));
    CHECK(report.iterations == 99);

    GridMetrics grid;
    grid.dx               = 2;
    grid.dy               = 3;
    grid.dz               = {1, 1.5, 2, 2.5, 3, 4};
    Field3D            qv = field, u = field, v = field, w = field;
    DiffusionStateView state;
    state.grid       = &grid;
    state.rho        = &rho;
    state.theta      = &field;
    state.qv         = &qv;
    state.u          = &u;
    state.v          = &v;
    state.w          = &w;
    state.periodic_x = true;
    state.periodic_y = true;
    DiffusionConfig config;
    config.scheme_id    = "implicit";
    config.apply_to     = "scalars_and_momentum";
    config.K_h          = 4;
    config.K_v          = 2;
    config.dt_diffusion = 0.1;
    for (std::size_t n = 0; n < field.size(); ++n)
        rho.data()[n] = 0.8f + 0.2f * std::cos(n * 0.1);
    ImplicitDiffusionScheme cpu, gpu;
    DiffusionTendencies     expected, actual;
    DiffusionDiagnostics    diagnostics;
    Field3D                 coefficient_scalar(nx, ny, nz, 1), coefficient_momentum(nx, ny, nz, 20);
    const Field3D           reference_field   = field;
    const Field3D           reference_density = rho;
    for (bool c_grid : {false, true})
        for (bool variable : {false, true})
        {
            INFO(c_grid);
            INFO(variable);
            state.c_grid = c_grid;
            field = qv = u = v = w = reference_field;
            rho                    = reference_density;
            const std::array<Field3D*, 5> fields{&field, &qv, &u, &v, &w};
            for (int i = 0; i < nx; ++i)
                for (int j = 0; j < ny; ++j)
                    for (int k = 0; k < nz; ++k)
                    {
                        const bool scalar_closure =
                            i == 0 || i == nx - 1 || j == 0 || j == ny - 1 || (!c_grid && (k == 0 || k == nz - 1));
                        if (scalar_closure)
                            rho(i, j, k) = std::numeric_limits<float>::quiet_NaN();
                        for (std::size_t f = 0; f < fields.size(); ++f)
                            if (scalar_closure || (c_grid && f >= 2 && k == nz - 1))
                                (*fields[f])(i, j, k) = std::numeric_limits<float>::quiet_NaN();
                    }

            config.use_variable_K = variable;
            state.K_scalar        = variable ? &coefficient_scalar : nullptr;
            state.K_momentum      = variable ? &coefficient_momentum : nullptr;
            state.compute_runtime = nullptr;
            cpu.compute_diffusion_tendencies(config, state, expected, nullptr);
            state.compute_runtime = &runtime;
            gpu.compute_diffusion_tendencies(config, state, actual, &diagnostics);
            const Field3D* reference[] = {&expected.dudt_diff,
                                          &expected.dvdt_diff,
                                          &expected.dwdt_diff,
                                          &expected.dthetadt_diff,
                                          &expected.dqvdt_diff};
            const Field3D* device[]    = {
                &actual.dudt_diff, &actual.dvdt_diff, &actual.dwdt_diff, &actual.dthetadt_diff, &actual.dqvdt_diff};
            for (int f = 0; f < 5; ++f)
                for (std::size_t n = 0; n < field.size(); ++n)
                    CHECK(std::abs(device[f]->data()[n] - reference[f]->data()[n]) <= 2e-5);
            CHECK(diagnostics.K_effective(2, 2, 2) == (variable ? coefficient_momentum(2, 2, 2) : 4));
            CHECK(gpu.check_stability(config, state) == 0);
        }
    const auto saved = actual.dthetadt_diff;
    qv(2, 2, 2)      = std::numeric_limits<float>::quiet_NaN();
    CHECK_THROWS(gpu.compute_diffusion_tendencies(config, state, actual, &diagnostics));
    CHECK(std::equal(saved.data(), saved.data() + saved.size(), actual.dthetadt_diff.data()));
}
} // namespace

TEST_CASE("Metal implicit diffusion solves the conservative backward equation", "[native-implicit][metal]")
{
    exercise("metal");
}

TEST_CASE("Vulkan implicit diffusion solves the conservative backward equation", "[native-implicit][vulkan]")
{
    exercise("vulkan");
}

TEST_CASE("CPU implicit diffusion preserves scale invariance", "[implicit][scaling]")
{
    GridMetrics grid;
    grid.dx = 1;
    grid.dy = 1;
    grid.dz.assign(4, 1);
    Field3D field(6, 5, 4, 0), rho(6, 5, 4, 1);
    for (int i = 0; i < 6; ++i)
        for (int j = 0; j < 5; ++j)
            for (int k = 0; k < 4; ++k)
                field(i, j, k) = i % 2 ? 1.0f : -1.0f;
    DiffusionStateView state;
    state.grid       = &grid;
    state.theta      = &field;
    state.rho        = &rho;
    state.periodic_x = state.periodic_y = true;
    DiffusionConfig cfg;
    cfg.K_h          = 1;
    cfg.dt_diffusion = 1;
    ImplicitDiffusionScheme solver;
    DiffusionTendencies     large, small;
    for (bool momentum : {false, true})
    {
        INFO(momentum);
        cfg.apply_to = momentum ? "momentum" : "scalars";
        state.theta  = momentum ? nullptr : &field;
        state.u = state.v = state.w = momentum ? &field : nullptr;
        for (int i = 0; i < 6; ++i)
            for (int j = 0; j < 5; ++j)
                for (int k = 0; k < 4; ++k)
                    field(i, j, k) = i % 2 ? 1.0f : -1.0f;
        solver.compute_diffusion_tendencies(cfg, state, large, nullptr);
        for (std::size_t n = 0; n < field.size(); ++n)
            field.data()[n] *= 1e-12f;
        solver.compute_diffusion_tendencies(cfg, state, small, nullptr);
        const auto& big  = momentum ? large.dudt_diff : large.dthetadt_diff;
        const auto& tiny = momentum ? small.dudt_diff : small.dthetadt_diff;
        for (std::size_t n = 0; n < field.size(); ++n)
            CHECK(std::abs(tiny.data()[n] * 1e12 - big.data()[n]) < 1e-5);
    }
}
