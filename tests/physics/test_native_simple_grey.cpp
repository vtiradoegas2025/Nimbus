#include "catch2/catch.hpp"
#include "compute/backend_runtime.hpp"
#include "core/runtime/simulation_state.hpp"
#include "radiation/runtime.hpp"
#include "compute/simple_grey_contract.hpp"
#include "radiation/schemes/simple_grey/simple_grey.hpp"
#include <cmath>
#include <limits>
#include <omp.h>
namespace
{
struct Column
{
    std::vector<double> t, p, pi, rho, dz, z, qv, qc, qi;
    explicit Column(int nz)
    {
        for (int k = 0; k <= nz; ++k)
        {
            pi.push_back(100000.0 * std::exp(-.1 * k));
            z.push_back(200.0 * k);
        }
        for (int k = 0; k < nz; ++k)
        {
            t.push_back(290 - 1.1 * k);
            p.push_back(.5 * (pi[k] + pi[k + 1]));
            rho.push_back(1.1 - .01 * k);
            dz.push_back(200);
            qv.push_back(.012 - .0001 * k);
            qc.push_back(.0001 * (k % 3));
            qi.push_back(.0001 * (k % 2));
        }
    }
    RadiationColumnStateView view() const
    {
        RadiationColumnStateView c;
        c.T             = &t;
        c.p             = &p;
        c.p_int         = &pi;
        c.rho           = &rho;
        c.dz            = &dz;
        c.z_int         = &z;
        c.qv            = &qv;
        c.qc            = &qc;
        c.qi            = &qi;
        c.Tsfc          = 300;
        c.emissivity_lw = .8;
        c.mu0           = .6;
        c.S0            = 1361;
        c.albedo_sw     = .3;
        return c;
    }
};
class Capture final : public RadiationComputeKernels
{
public:
    RadiationComputeKernels& backend;
    SimpleGreyDispatch       saved;
    std::vector<float>       input;
    bool                     fail = false;
    explicit Capture(RadiationComputeKernels& k) : backend(k)
    {
    }
    bool supports(RadiationKernelOp op) const override
    {
        return backend.supports(op);
    }
    bool dispatch_simple_grey(const SimpleGreyDispatch& d) override
    {
        saved = d;
        input.assign(d.input, d.input + nimbus::compute::simple_grey::input_count(d));
        return !fail && backend.dispatch_simple_grey(d);
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
    constexpr int          nx = 11, ny = 10, nz = 8;
    SimulationStorageState cpu{{nx, ny, nz, 200, 200, .1, 200}}, gpu{{nx, ny, nz, 200, 200, .1, 200}};
    for (auto* s : {&cpu, &gpu})
    {
        s->grid.geometry.initialize(nx, ny, nz, 200, 200, 200, CoordinateSystem::Cartesian, StaggerType::CGrid);
        s->configuration.stagger_type = StaggerType::CGrid;
        tmv::fields::visit(
            s->fields, tmv::fields::prognostic, [&](const char*, Field3D& field) { field.resize(nx, ny, nz, 0); });
        s->fields.theta.fill(300);
        s->fields.rho.fill(1);
        s->fields.qv.fill(.01f);
        for (int i = 0; i < nx; ++i)
            for (int j = 0; j < ny; ++j)
                for (int k = 0; k < nz; ++k)
                    s->fields.p(i, j, k) = float(95000 * std::exp(-.07 * k));
        s->runtime.boundary_layer.surface_config.Tsfc       = 290;
        s->runtime.boundary_layer.surface_config.albedo     = .3;
        s->runtime.boundary_layer.surface_config.emissivity = .8;
    }
    auto cpu_view = make_simulation_state_view(cpu), gpu_view = make_simulation_state_view(gpu);
    gpu_view.compute_runtime = &runtime;
    tmv::RadiationConfig cfg;
    cfg.scheme_id = "simple_grey";
    cfg.do_lw = cfg.do_sw           = true;
    cfg.enable_diagnostics          = true;
    cfg.dt_radiation                = 4;
    cfg.use_subcycling              = true;
    cfg.subcycle_count              = 2;
    cfg.local_solar_time_at_start_h = 12;
    tmv::initialize_radiation(cpu_view, "simple_grey", cfg);
    tmv::initialize_radiation(gpu_view, "simple_grey", cfg);
    const auto start = runtime.backend->successful_kernel_dispatch_count(nimbus::compute::KernelId::SimpleGrey);
    tmv::step_radiation(cpu_view, 0);
    tmv::step_radiation(gpu_view, 0);
    CHECK(runtime.backend->successful_kernel_dispatch_count(nimbus::compute::KernelId::SimpleGrey) == start + 2);
    const auto& a = cpu.runtime.radiation.held_flux_diagnostics;
    const auto& b = gpu.runtime.radiation.held_flux_diagnostics;
    compare(b.upward_longwave_wm2, a.upward_longwave_wm2, 3e-8);
    compare(b.downward_longwave_wm2, a.downward_longwave_wm2, 3e-8);
    compare(b.upward_shortwave_wm2, a.upward_shortwave_wm2, 3e-8);
    compare(b.downward_shortwave_wm2, a.downward_shortwave_wm2, 3e-8);
    REQUIRE_FALSE(b.empty());
    CHECK(b.columns_x == 9);
    CHECK(b.columns_y == 8);
    CHECK(b.levels == 8);
    CHECK(gpu.runtime.radiation.held_shortwave_incoming_power_w ==
          Approx(gpu.runtime.radiation.held_shortwave_atmospheric_absorbed_power_w +
                 gpu.runtime.radiation.held_surface_absorbed_power_w + gpu.runtime.radiation.held_top_escape_power_w)
              .epsilon(3e-12));
    const auto held = gpu.runtime.radiation.held_flux_diagnostics.upward_longwave_wm2;
    tmv::step_radiation(gpu_view, 1);
    CHECK(gpu.runtime.radiation.held_flux_diagnostics.upward_longwave_wm2 == held);
    CHECK(runtime.backend->successful_kernel_dispatch_count(nimbus::compute::KernelId::SimpleGrey) == start + 2);
    tmv::step_radiation(gpu_view, 2);
    CHECK(gpu.runtime.radiation.last_update_time_s == 2);
    CHECK(gpu.runtime.radiation.next_update_time_s == 4);
    const auto before                    = gpu.runtime.radiation.held_flux_diagnostics.upward_longwave_wm2;
    gpu.fields.p(nx - 2, ny - 2, nz - 1) = std::numeric_limits<float>::quiet_NaN();
    REQUIRE_THROWS(tmv::step_radiation(gpu_view, 4));
    CHECK(gpu.runtime.radiation.held_flux_diagnostics.upward_longwave_wm2 == before);
    CHECK(gpu.runtime.radiation.last_update_time_s == 2);
    CHECK(gpu.runtime.radiation.next_update_time_s == 4);
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
    auto* kernels = runtime.backend->radiation_kernels();
    REQUIRE(kernels);
    Capture capture(*kernels);
    runtime_columns(runtime);
    for (int nz : {1, 3, 16, 64})
        for (int bands : {1, 2, 3})
            for (bool water : {false, true})
                for (double depth : {0.0, 1e-6, .3, 3.0, 100.0})
                {
                    CAPTURE(backend, nz, bands, water, depth);
                    tmv::RadiationConfig cfg;
                    cfg.do_lw                     = (bands & 1) != 0;
                    cfg.do_sw                     = (bands & 2) != 0;
                    cfg.tau_lw_ref                = depth;
                    cfg.tau_sw_ref                = depth;
                    cfg.n_lw                      = 1.3;
                    cfg.n_sw                      = 2.1;
                    cfg.clear_sky_only            = !water;
                    cfg.simple_grey_optical_model = water ? "absorbing_water_cloud_path_v1" : "pressure_clear_sky_v1";
                    tmv::SimpleGreyScheme scheme;
                    scheme.initialize(cfg);
                    Column                    column(nz);
                    auto                      view = column.view();
                    RadiationColumnTendencies cpu, gpu;
                    RadiationColumnFluxes     cpu_flux, gpu_flux;
                    scheme.compute_column(cfg, view, cpu, &cpu_flux);
                    REQUIRE(scheme.compute_columns_native_diagnosed(cfg, &view, 1, &gpu, capture, &gpu_flux));
                    compare(gpu.dTdt_rad, cpu.dTdt_rad, 3e-11);
                    compare(gpu.dTdt_lw, cpu.dTdt_lw, 3e-11);
                    compare(gpu.dTdt_sw, cpu.dTdt_sw, 3e-11);
                    compare(gpu_flux.Fup_lw, cpu_flux.Fup_lw, 3e-8);
                    compare(gpu_flux.Fdn_lw, cpu_flux.Fdn_lw, 3e-8);
                    compare(gpu_flux.Fup_sw, cpu_flux.Fup_sw, 3e-8);
                    compare(gpu_flux.Fdn_sw, cpu_flux.Fdn_sw, 3e-8);
                    double heating = 0;
                    for (int k = 0; k < nz; ++k)
                        heating += gpu.dTdt_rad[k] * column.rho[k] * column.dz[k] * 1004;
                    CHECK(std::abs(heating - gpu.atmospheric_power_total_wm2) < 3e-8);
                    CHECK(std::abs(gpu.atmospheric_power_total_wm2 -
                                   (gpu_flux.Fnet_total.front() - gpu_flux.Fnet_total.back())) < 3e-8);
                    if (depth == .3 && water)
                    {
                        auto bad  = view;
                        auto p    = column.pi;
                        p[1]      = p[0] + 1;
                        bad.p_int = &p;
                        RadiationColumnStateView  columns[2]{view, bad};
                        RadiationColumnTendencies t[2]{gpu, gpu};
                        RadiationColumnFluxes     f[2]{gpu_flux, gpu_flux};
                        REQUIRE_THROWS(scheme.compute_columns_native_diagnosed(cfg, columns, 2, t, capture, f));
                        CHECK(t[0].dTdt_rad == gpu.dTdt_rad);
                        CHECK(f[0].Fup_lw == gpu_flux.Fup_lw);
                        capture.fail = true;
                        CHECK_FALSE(scheme.compute_columns_native_diagnosed(cfg, &view, 1, &gpu, capture, &gpu_flux));
                        capture.fail = false;
                        CHECK(gpu.dTdt_rad == t[0].dTdt_rad);
                        CHECK(gpu_flux.Fup_lw == f[0].Fup_lw);
                    }
                }
    auto d  = capture.saved;
    d.input = capture.input.data();
    std::vector<float> out(nimbus::compute::simple_grey::output_count(d), 17);
    d.output            = out.data();
    const auto rejected = [&]()
    {
        CHECK_FALSE(kernels->dispatch_simple_grey(d));
        for (float v : out)
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
    d.output         = out.data();
    capture.input[0] = -1;
    rejected();
    CHECK(runtime.backend->successful_kernel_dispatch_count(nimbus::compute::KernelId::SimpleGrey) > 0);
}
} // namespace
TEST_CASE("Native simple-grey optics and transfer preserve column energy on Metal", "[metal][simple_grey]")
{
    exercise("metal");
}
TEST_CASE("Native simple-grey optics and transfer preserve column energy on Vulkan", "[vulkan][simple_grey]")
{
    exercise("vulkan");
}
