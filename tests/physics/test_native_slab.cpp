#include "catch2/catch.hpp"
#include "boundary_layer/schemes/slab/slab.hpp"
#include "compute/backend_runtime.hpp"
#include "core/runtime/simulation_state.hpp"
#include "boundary_layer/runtime.hpp"
#include "compute/slab_column_contract.hpp"
#include <array>
#include <cmath>
#include <limits>
namespace
{
struct Column
{
    std::vector<double>  theta, qv, u, v, p, rho, z;
    tmv::SlabColumnState slab;
    explicit Column(int levels, double gradient)
    {
        for (int k = 0; k <= levels; ++k)
            z.push_back(50.0 * k);
        for (int k = 0; k < levels; ++k)
        {
            theta.push_back(300 + gradient * k);
            qv.push_back(.012 - .0001 * k);
            u.push_back(2 + .5 * k);
            v.push_back(-1 + .2 * k);
            p.push_back(100000 * std::exp(-.007 * k));
            rho.push_back(1 - .005 * k);
        }
    }
    tmv::BoundaryLayerColumnStateView view(double delta, double slope)
    {
        tmv::BoundaryLayerColumnStateView c;
        c.theta           = &theta;
        c.qv              = &qv;
        c.u               = &u;
        c.v               = &v;
        c.p               = &p;
        c.rho             = &rho;
        c.z_int           = &z;
        c.slab_state      = &slab;
        c.u_sfc           = u[0];
        c.v_sfc           = v[0];
        c.theta_sfc       = theta[0] + delta;
        c.qv_sfc          = qv[0];
        c.z_sfc           = 25;
        c.surface_slope_x = slope;
        c.surface_slope_y = -.4 * slope;
        return c;
    }
};
class Capture final : public BoundaryLayerComputeKernels
{
public:
    BoundaryLayerComputeKernels& backend;
    SlabColumnDispatch           saved;
    std::vector<float>           input;
    bool                         fail = false;
    explicit Capture(BoundaryLayerComputeKernels& k) : backend(k)
    {
    }
    bool supports(BoundaryLayerKernelOp op) const override
    {
        return backend.supports(op);
    }
    bool dispatch_slab_column(const SlabColumnDispatch& d) override
    {
        saved = d;
        input.assign(d.input, d.input + nimbus::compute::slab_column::input_count(d));
        return !fail && backend.dispatch_slab_column(d);
    }
};
void compare(const std::vector<double>& a, const std::vector<double>& b, double tolerance)
{
    REQUIRE(a.size() == b.size());
    for (std::size_t n = 0; n < a.size(); ++n)
    {
        CAPTURE(n, a[n], b[n]);
        CHECK(std::abs(a[n] - b[n]) <= tolerance);
    }
}
void runtime_columns(ComputeBackendRuntimeState& runtime)
{
    constexpr int          nx = 9, ny = 8, nz = 8;
    SimulationStorageState cpu{{nx, ny, nz, 200, 50, .1, 200}}, gpu{{nx, ny, nz, 200, 50, .1, 200}};
    for (auto* s : {&cpu, &gpu})
    {
        s->grid.geometry.initialize(nx, ny, nz, 200, 50, 200, CoordinateSystem::Cartesian, StaggerType::CGrid);
        s->configuration.stagger_type = StaggerType::CGrid;
        tmv::fields::visit(
            s->fields, tmv::fields::prognostic, [&](const char*, Field3D& f) { f.resize(nx, ny, nz, 0); });
        s->fields.rho.fill(1);
        s->fields.qv.fill(.01f);
        for (int i = 0; i < nx; ++i)
            for (int j = 0; j < ny; ++j)
                for (int k = 0; k < nz; ++k)
                {
                    s->fields.theta(i, j, k) = 300 + .1f * k;
                    s->fields.p(i, j, k)     = float(100000 * std::exp(-.007 * k));
                    s->fields.u(i, j, k)     = 2 + .5f * k;
                    s->fields.v(i, j, k)     = -1 + .2f * k;
                }
    }
    auto a = make_simulation_state_view(cpu), b = make_simulation_state_view(gpu);
    b.compute_runtime = &runtime;
    tmv::BoundaryLayerConfig cfg;
    cfg.scheme_id      = "slab";
    cfg.dt_pbl         = 2;
    cfg.pbl_max_height = 400;
    tmv::SurfaceConfig surface;
    surface.Tsfc = 303;
    surface.qsfc = .01;
    tmv::initialize_boundary_layer(a, "slab", cfg, surface);
    tmv::initialize_boundary_layer(b, "slab", cfg, surface);
    const auto start = runtime.backend->successful_kernel_dispatch_count(nimbus::compute::KernelId::SlabColumn);
    tmv::step_boundary_layer(a, 0);
    tmv::step_boundary_layer(b, 0);
    CHECK(runtime.backend->successful_kernel_dispatch_count(nimbus::compute::KernelId::SlabColumn) > start);
    for (std::size_t n = 0; n < cpu.tendencies.dtheta_dt_pbl.size(); ++n)
        CHECK(std::abs(double(cpu.tendencies.dtheta_dt_pbl.data()[n]) - gpu.tendencies.dtheta_dt_pbl.data()[n]) < 3e-9);
    const auto held       = gpu.runtime.boundary_layer.held_theta_inventory_rate_kgk_s;
    const auto dispatches = runtime.backend->successful_kernel_dispatch_count(nimbus::compute::KernelId::SlabColumn);
    tmv::step_boundary_layer(b, 1);
    CHECK(runtime.backend->successful_kernel_dispatch_count(nimbus::compute::KernelId::SlabColumn) == dispatches);
    for (std::size_t n = 0; n < held.size(); ++n)
        CHECK(held.data()[n] == gpu.runtime.boundary_layer.held_theta_inventory_rate_kgk_s.data()[n]);
    const auto next                          = gpu.runtime.boundary_layer.next_update_time_s;
    gpu.fields.theta(nx - 2, ny - 2, nz - 1) = std::numeric_limits<float>::quiet_NaN();
    REQUIRE_THROWS(tmv::step_boundary_layer(b, 2));
    CHECK(gpu.runtime.boundary_layer.next_update_time_s == next);
    for (std::size_t n = 0; n < held.size(); ++n)
        CHECK(held.data()[n] == gpu.runtime.boundary_layer.held_theta_inventory_rate_kgk_s.data()[n]);
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
    auto* kernels = runtime.backend->boundary_layer_kernels();
    REQUIRE(kernels);
    Capture capture(*kernels);
    runtime_columns(runtime);
    for (double height : {100.0, 249.999, 250.001, 499.999, 500.001, 749.999, 750.001, 999.999, 1000.001})
    {
        Column cpu(4, -4), gpu(4, -4);
        for (auto* column : {&cpu, &gpu})
        {
            column->z     = {0, 500, 1000, 1500, 2000};
            column->theta = {304, 300, 296, 292};
            column->qv    = {.014, .01, .008, .005};
            column->u     = {4, 10, 12, 14};
            column->rho   = {1.2, .8, .5, .3};
        }
        tmv::BoundaryLayerConfig cfg;
        cfg.apply_surface_fluxes = false;
        cfg.pbl_top_method       = "theta";
        cfg.pbl_max_height       = height;
        tmv::SlabScheme scheme;
        scheme.initialize(cfg);
        tmv::BoundaryLayerTendencies  expected, actual;
        tmv::BoundaryLayerDiagnostics diagnostic;
        auto                          cpu_view = cpu.view(0, 0), gpu_view = gpu.view(0, 0);
        scheme.compute_column(cfg, {}, cpu_view, expected);
        REQUIRE(scheme.compute_columns_native(cfg, {}, &gpu_view, 1, &actual, &diagnostic, capture));
        CHECK(gpu.slab.h_m == Approx(height).margin(1e-10));
        CHECK(gpu.slab.theta_m_k == Approx(cpu.slab.theta_m_k).epsilon(3e-12));
        compare(actual.dudt_pbl, expected.dudt_pbl, 3e-10);
        compare(actual.dthetadt_pbl, expected.dthetadt_pbl, 3e-10);
        compare(actual.dqvdt_pbl, expected.dqvdt_pbl, 3e-12);
    }
    int completed = 0;
    for (int nz : {1, 8, 64})
        for (const char* method : {"ri", "theta"})
            for (const char* surface : {"bulk", "monin_obukhov"})
                for (double gradient : {-.05, .1, 1.5})
                    for (double delta : {-3.0, 0.0, 3.0})
                        for (double slope : {0.0, .25})
                            for (bool apply : {false, true})
                            {
                                CAPTURE(backend, nz, method, surface, gradient, delta, slope, apply);
                                tmv::BoundaryLayerConfig cfg;
                                cfg.pbl_top_method       = method;
                                cfg.surface_layer_id     = surface;
                                cfg.apply_surface_fluxes = apply;
                                cfg.pbl_max_height       = 500;
                                cfg.min_ustar            = .2;
                                tmv::SurfaceConfig sfc;
                                tmv::SlabScheme    scheme;
                                scheme.initialize(cfg);
                                Column                        cpu(nz, gradient), gpu(nz, gradient);
                                auto                          a = cpu.view(delta, slope), b = gpu.view(delta, slope);
                                tmv::BoundaryLayerTendencies  expected, actual;
                                tmv::BoundaryLayerDiagnostics da, db;
                                da.include_diffusivity_profiles = db.include_diffusivity_profiles = true;
                                bool valid                                                        = true;
                                try
                                {
                                    scheme.compute_column(cfg, sfc, a, expected, &da);
                                }
                                catch (const std::exception&)
                                {
                                    valid = false;
                                }
                                if (!valid)
                                {
                                    CHECK_FALSE(scheme.compute_columns_native(cfg, sfc, &b, 1, &actual, &db, capture));
                                    CHECK_FALSE(gpu.slab.initialized);
                                    continue;
                                }
                                ++completed;
                                REQUIRE(scheme.compute_columns_native(cfg, sfc, &b, 1, &actual, &db, capture));
                                CHECK(gpu.slab.h_m == Approx(cpu.slab.h_m).epsilon(3e-11));
                                CHECK(gpu.slab.theta_m_k == Approx(cpu.slab.theta_m_k).epsilon(3e-12));
                                CHECK(gpu.slab.qv_m_kgkg == Approx(cpu.slab.qv_m_kgkg).epsilon(3e-12));
                                compare(actual.dudt_pbl, expected.dudt_pbl, 3e-10);
                                compare(actual.dvdt_pbl, expected.dvdt_pbl, 3e-10);
                                compare(actual.dthetadt_pbl, expected.dthetadt_pbl, 3e-10);
                                compare(actual.dqvdt_pbl, expected.dqvdt_pbl, 3e-12);
                                compare(db.K_m, da.K_m, 0);
                                compare(db.K_h, da.K_h, 0);
                                compare(db.K_e, da.K_e, 0);
                                const std::array<double, 8> diagnostics_a{
                                    da.ustar, da.tau_u, da.tau_v, da.H, da.E, da.Cd, da.Ch, da.Ce},
                                    diagnostics_b{db.ustar, db.tau_u, db.tau_v, db.H, db.E, db.Cd, db.Ch, db.Ce};
                                for (std::size_t n = 0; n < diagnostics_a.size(); ++n)
                                    CHECK(std::abs(diagnostics_a[n] - diagnostics_b[n]) <=
                                          3e-9 * std::max(1.0, std::abs(diagnostics_a[n])));
                                double iu = 0, iv = 0, it = 0, iq = 0;
                                for (int k = 0; k < nz; ++k)
                                {
                                    double m = gpu.rho[k] * 50;
                                    iu += m * actual.dudt_pbl[k];
                                    iv += m * actual.dvdt_pbl[k];
                                    it += m * actual.dthetadt_pbl[k];
                                    iq += m * actual.dqvdt_pbl[k];
                                }
                                CHECK(std::abs(iu - db.tau_u) < 3e-9);
                                CHECK(std::abs(iv - db.tau_v) < 3e-9);
                                CHECK(std::abs(it - db.H / (1004 * std::pow(gpu.p[0] / 100000, 287.0 / 1004))) < 3e-9);
                                CHECK(std::abs(iq - db.E) < 3e-12);
                            }
    CHECK(completed > 0);

    tmv::BoundaryLayerConfig cfg;
    cfg.pbl_max_height       = 50;
    cfg.apply_surface_fluxes = false;
    tmv::SurfaceConfig sfc;
    tmv::SlabScheme    scheme;
    scheme.initialize(cfg);
    Column                        column(8, .5);
    auto                          view = column.view(0, 0);
    tmv::BoundaryLayerTendencies  tendency;
    tmv::BoundaryLayerDiagnostics diag;
    REQUIRE(scheme.compute_columns_native(cfg, sfc, &view, 1, &tendency, &diag, capture));
    CHECK(column.slab.h_m <= 50);
    CHECK(column.slab.h_m > 0);
    const auto original = column.slab;
    const auto saved    = tendency.dthetadt_pbl;
    capture.fail        = true;
    CHECK_FALSE(scheme.compute_columns_native(cfg, sfc, &view, 1, &tendency, &diag, capture));
    capture.fail = false;
    CHECK(column.slab.h_m == original.h_m);
    CHECK(tendency.dthetadt_pbl == saved);
    tmv::BoundaryLayerColumnStateView aliases[2]{view, view};
    tmv::BoundaryLayerTendencies      t[2];
    tmv::BoundaryLayerDiagnostics     f[2];
    REQUIRE_THROWS(scheme.compute_columns_native(cfg, sfc, aliases, 2, t, f, capture));
    CHECK(column.slab.h_m == original.h_m);
    auto d  = capture.saved;
    d.input = capture.input.data();
    std::vector<float> output(nimbus::compute::slab_column::output_count(d), 17);
    d.output            = output.data();
    const auto rejected = [&]()
    {
        CHECK_FALSE(kernels->dispatch_slab_column(d));
        for (float v : output)
            CHECK(v == 17);
    };
    d.constants.columns = std::numeric_limits<int>::max();
    rejected();
    d.constants = capture.saved.constants;
    d.input     = nullptr;
    rejected();
    d.input  = capture.input.data();
    d.output = capture.input.data() + 1;
    rejected();
    d.output         = output.data();
    capture.input[0] = -1;
    rejected();
    CHECK(runtime.backend->successful_kernel_dispatch_count(nimbus::compute::KernelId::SlabColumn) > 0);
}
} // namespace
TEST_CASE("Native slab height, surface fluxes and mixing close on Metal", "[metal][slab]")
{
    exercise("metal");
}
TEST_CASE("Native slab height, surface fluxes and mixing close on Vulkan", "[vulkan][slab]")
{
    exercise("vulkan");
}
