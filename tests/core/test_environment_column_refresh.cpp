#include "catch2/catch.hpp"
#include "core/runtime/environment_column_refresh.hpp"
#include "core/runtime/earth_fixed_remap.hpp"
#include "core/runtime/simulation_state.hpp"
#include "terrain/earth_fixed_geometry.hpp"
#include "dynamics/moving_environment.hpp"
#include "boundary_layer/runtime.hpp"
#include "radiation/runtime.hpp"
#include "compute/backend_runtime.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
namespace
{
using namespace tmv;
std::vector<Field3D*> fields(SimulationStorageState& s)
{
    auto& p = s.runtime.boundary_layer;
    auto& c = s.runtime.chaos;
    return {&p.held_u_force_n,
            &p.held_v_force_n,
            &p.held_theta_inventory_rate_kgk_s,
            &p.held_vapor_mass_rate_kg_s,
            &p.held_cloud_liquid_mass_rate_kg_s,
            &p.held_cloud_ice_mass_rate_kg_s,
            &s.tendencies.dtke_dt_pbl,
            &s.runtime.radiation.held_cell_power_w,
            &c.base_du_dt_pbl,
            &c.base_dv_dt_pbl,
            &c.base_dtheta_dt_pbl,
            &c.base_dqv_dt_pbl,
            &c.base_dtke_dt_pbl};
}
std::vector<std::vector<float>*> surfaces(SimulationStorageState& s)
{
    auto& p = s.runtime.boundary_layer;
    auto& c = s.runtime.chaos;
    return {&p.skin_theta_2d,
            &p.skin_qv_2d,
            &s.surface.ustar,
            &s.surface.tau_u,
            &s.surface.tau_v,
            &s.surface.sensible_heat_flux,
            &s.surface.latent_heat_flux,
            &s.surface.moisture_flux,
            &s.surface.pbl_height,
            &c.base_sfc_ustar,
            &c.base_sfc_tau_u,
            &c.base_sfc_tau_v,
            &c.base_sfc_sensible_heat_flux,
            &c.base_sfc_latent_heat_flux,
            &c.base_sfc_moisture_flux};
}
SimulationStorageState initial()
{
    constexpr int          nx = 20, ny = 19, nz = 8;
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
    auto                view = make_simulation_state_view(s);
    BoundaryLayerConfig pbl;
    pbl.scheme_id      = "slab";
    pbl.pbl_max_height = 500;
    SurfaceConfig surface;
    surface.Tsfc = 303;
    surface.qsfc = .013;
    initialize_boundary_layer(view, "slab", pbl, surface);
    step_boundary_layer(view, 0);
    RadiationConfig rad;
    rad.scheme_id          = "simple_grey";
    rad.do_lw              = true;
    rad.do_sw              = false;
    rad.enable_diagnostics = true;
    initialize_radiation(view, "simple_grey", rad);
    step_radiation(view, 0);
    auto& c                       = s.runtime.chaos;
    auto& b                       = s.runtime.boundary_layer;
    c.pbl_base_tendencies_valid   = true;
    c.base_du_dt_pbl              = b.held_u_force_n;
    c.base_dv_dt_pbl              = b.held_v_force_n;
    c.base_dtheta_dt_pbl          = b.held_theta_inventory_rate_kgk_s;
    c.base_dqv_dt_pbl             = b.held_vapor_mass_rate_kg_s;
    c.base_dtke_dt_pbl            = s.tendencies.dtke_dt_pbl;
    c.base_sfc_ustar              = s.surface.ustar;
    c.base_sfc_tau_u              = s.surface.tau_u;
    c.base_sfc_tau_v              = s.surface.tau_v;
    c.base_sfc_sensible_heat_flux = s.surface.sensible_heat_flux;
    c.base_sfc_latent_heat_flux   = s.surface.latent_heat_flux;
    c.base_sfc_moisture_flux      = s.surface.moisture_flux;
    return s;
}
std::vector<double> snapshot(SimulationStorageState& s)
{
    std::vector<double> v;
    for (auto* f : fields(s))
        v.insert(v.end(), f->data(), f->data() + f->size());
    for (auto* f : surfaces(s))
        v.insert(v.end(), f->begin(), f->end());
    for (const auto& c : s.runtime.boundary_layer.slab_columns)
    {
        v.push_back(c.h_m);
        v.push_back(c.theta_m_k);
        v.push_back(c.qv_m_kgkg);
        v.push_back(c.initialized);
    }
    return v;
}
class RejectRadiation final : public RadiationComputeKernels
{
public:
    RadiationComputeKernels& backend;
    int                      calls = 0;
    explicit RejectRadiation(RadiationComputeKernels& b) : backend(b)
    {
    }
    bool supports(RadiationKernelOp op) const override
    {
        return backend.supports(op);
    }
    bool dispatch_simple_grey(const SimpleGreyDispatch& d) override
    {
        ++calls;
        if (calls == 2)
            return false;
        return backend.dispatch_simple_grey(d);
    }
};
class Backend final : public ComputeBackend
{
public:
    ComputeBackend& backend;
    RejectRadiation radiation;
    explicit Backend(ComputeBackend& b) : backend(b), radiation(*b.radiation_kernels())
    {
    }
    std::string name() const override
    {
        return backend.name();
    }
    bool initialize(std::string&) override
    {
        return false;
    }
    void shutdown() override
    {
    }
    TerrainComputeKernels* terrain_kernels() override
    {
        return backend.terrain_kernels();
    }
    BoundaryLayerComputeKernels* boundary_layer_kernels() override
    {
        return backend.boundary_layer_kernels();
    }
    RadiationComputeKernels* radiation_kernels() override
    {
        return &radiation;
    }
};
void exercise(const char* name)
{
    ComputeBackendRuntimeState runtime;
    runtime.config.backend        = name;
    runtime.config.allow_fallback = false;
    std::string error;
    REQUIRE(initialize_compute_backend_runtime(runtime, CoordinateSystem::Cartesian, error));
    INFO(error);
    REQUIRE_FALSE(runtime.fallback_active);
    auto       cpu = initial(), gpu = initial();
    const auto original  = snapshot(gpu);
    const auto a         = extend_environment_columns(cpu);
    const auto b         = extend_environment_columns(gpu, *runtime.backend);
    const auto reference = snapshot(cpu), actual = snapshot(gpu);
    REQUIRE(reference.size() == actual.size());
    for (std::size_t n = 0; n < reference.size(); ++n)
    {
        CAPTURE(n, reference[n], actual[n]);
        CHECK(std::abs(reference[n] - actual[n]) <= 5e-6 * std::max(1.0, std::abs(reference[n])));
    }
    for (int f = 0; f < 4; ++f)
    {
        REQUIRE(a[f].size() == b[f].size());
        for (std::size_t n = 0; n < a[f].size(); ++n)
            CHECK(std::abs(a[f][n] - b[f][n]) < 3e-8);
    }
    auto       baseline  = initial();
    const auto untouched = fields(baseline), observed = fields(gpu);
    for (std::size_t f = 0; f < untouched.size(); ++f)
        for (int i = 1; i < 19; ++i)
            for (int j = 1; j < 18; ++j)
                for (int k = 0; k < 8; ++k)
                    CHECK((*untouched[f])(i, j, k) == (*observed[f])(i, j, k));
    CHECK(gpu.runtime.boundary_layer.last_update_time_s == baseline.runtime.boundary_layer.last_update_time_s);
    CHECK(gpu.runtime.radiation.last_update_time_s == baseline.runtime.radiation.last_update_time_s);
    auto remapped_cpu = initial(), remapped_gpu = initial();
    auto frame            = remapped_cpu.configuration.domain_frame;
    frame.velocity_ms     = {5, -2};
    const auto cpu_report = remap_earth_fixed_window(remapped_cpu, frame, 0, .2);
    const auto gpu_report = remap_earth_fixed_window(remapped_gpu, frame, 0, .2, runtime);
    CHECK(gpu_report.substeps == cpu_report.substeps);
    CHECK(gpu_report.dry_mass.final_inventory == Approx(cpu_report.dry_mass.final_inventory).epsilon(3e-7));
    CHECK(std::abs(gpu_report.dry_mass.final_inventory - gpu_report.dry_mass.initial_inventory +
                   gpu_report.dry_mass.boundary_outflow - gpu_report.dry_mass.storage_rounding) <
          3e-12 * gpu_report.dry_mass.initial_inventory);
    for (std::size_t species = 0; species < gpu_report.water.size(); ++species)
    {
        const auto& value = gpu_report.water[species];
        CHECK(value.final_inventory == Approx(cpu_report.water[species].final_inventory).epsilon(3e-7));
        CHECK(std::abs(value.final_inventory - value.initial_inventory + value.boundary_outflow -
                       value.storage_rounding) < 3e-12 * std::max(value.initial_inventory, 1.0));
    }
    CHECK(remapped_gpu.time_s == .2);
    const std::array<const Field3D*, 8> cpuf{&remapped_cpu.fields.rho,
                                             &remapped_cpu.fields.theta,
                                             &remapped_cpu.fields.p,
                                             &remapped_cpu.fields.qv,
                                             &remapped_cpu.fields.u,
                                             &remapped_cpu.fields.v,
                                             &remapped_cpu.fields.w,
                                             &remapped_cpu.tendencies.dtheta_dt_rad};
    const std::array<const Field3D*, 8> gpuf{&remapped_gpu.fields.rho,
                                             &remapped_gpu.fields.theta,
                                             &remapped_gpu.fields.p,
                                             &remapped_gpu.fields.qv,
                                             &remapped_gpu.fields.u,
                                             &remapped_gpu.fields.v,
                                             &remapped_gpu.fields.w,
                                             &remapped_gpu.tendencies.dtheta_dt_rad};
    for (std::size_t field = 0; field < cpuf.size(); ++field)
        for (std::size_t n = 0; n < cpuf[field]->size(); ++n)
        {
            CAPTURE(field, n, cpuf[field]->data()[n], gpuf[field]->data()[n]);
            CHECK(std::abs(double(cpuf[field]->data()[n]) - gpuf[field]->data()[n]) <
                  3e-6 * std::max(1.0, std::abs(double(cpuf[field]->data()[n]))));
        }
    auto       failed = initial();
    const auto before = snapshot(failed);
    Backend    reject(*runtime.backend);
    REQUIRE_THROWS(extend_environment_columns(failed, reject));
    CHECK(reject.radiation.calls == 2);
    CHECK(snapshot(failed) == before);
    failed.fields.theta(19, 18, 7) = std::numeric_limits<float>::quiet_NaN();
    REQUIRE_THROWS(extend_environment_columns(failed, *runtime.backend));
    CHECK(snapshot(failed) == before);
    CHECK(runtime.backend->successful_kernel_dispatch_count(nimbus::compute::KernelId::SlabColumn) >= 2);
    CHECK(runtime.backend->successful_kernel_dispatch_count(nimbus::compute::KernelId::SimpleGrey) >= 2);
}
} // namespace
TEST_CASE("Native environmental physics fills ghost owners atomically on Metal", "[metal][environment_physics]")
{
    exercise("metal");
}
TEST_CASE("Native environmental physics fills ghost owners atomically on Vulkan", "[vulkan][environment_physics]")
{
    exercise("vulkan");
}
