// Explicit-state boundary-layer runtime regressions.

#include "catch2/catch.hpp"
#include "boundary_layer/runtime.hpp"
#include "boundary_layer/base/surface_fluxes.hpp"
#include "boundary_layer/schemes/slab/slab.hpp"
#include "boundary_layer/schemes/mynn/mynn.hpp"
#include "boundary_layer/schemes/mynn/mynn_stability.hpp"
#include "compute/backend_runtime.hpp"
#include "core/runtime/boundary_layer_checkpoint.hpp"
#include "core/runtime/checkpoint_format.hpp"
#include "core/infra/physical_grid_metrics.hpp"
#include "core/runtime/simulation.hpp"
#include "core/runtime/simulation_state.hpp"
#include "core/runtime/simulation_runtime.hpp"
#include "util/log_profile.hpp"

#include <cmath>
#include <atomic>
#include <array>
#include <cstring>
#include <filesystem>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif
SimulationStorageState          global_simulation_storage{{3, 2, 6, 1000.0, 100.0, 1.0, 1000.0}};
int&                            NR                                   = global_simulation_storage.grid.nr;
int&                            NTH                                  = global_simulation_storage.grid.nth;
int&                            NZ                                   = global_simulation_storage.grid.nz;
double&                         dr                                   = global_simulation_storage.grid.dx;
double&                         dz                                   = global_simulation_storage.grid.dz;
double&                         dt                                   = global_simulation_storage.grid.dt;
double&                         dtheta                               = global_simulation_storage.grid.dy;
GridGeometry&                   global_grid_geometry                 = global_simulation_storage.grid.geometry;
tmv::RadiationRuntimeState&     tmv::global_radiation_runtime_state  = global_simulation_storage.runtime.radiation;
tmv::TurbulenceRuntimeState&    tmv::global_turbulence_runtime_state = global_simulation_storage.runtime.turbulence;
tmv::BoundaryLayerRuntimeState& tmv::global_boundary_layer_runtime_state =
    global_simulation_storage.runtime.boundary_layer;
tmv::chaos::ChaosRuntimeState& tmv::chaos::global_chaos_runtime_state = global_simulation_storage.runtime.chaos;
RayleighDampingRuntimeState&   global_rayleigh_runtime_state          = global_simulation_storage.runtime.rayleigh;
tmv::MicrophysicsRuntimeState& tmv::global_microphysics_runtime_state = global_simulation_storage.runtime.microphysics;
tmv::AdvectionRuntimeState&    tmv::global_advection_runtime_state    = global_simulation_storage.runtime.advection;
tmv::DiffusionRuntimeState&    tmv::global_diffusion_runtime_state    = global_simulation_storage.runtime.diffusion;
tmv::TimeSteppingRuntimeState& tmv::global_time_stepping_runtime_state =
    global_simulation_storage.runtime.time_stepping;
Field3D& rho    = global_simulation_storage.fields.rho;
Field3D& p      = global_simulation_storage.fields.p;
Field3D& u      = global_simulation_storage.fields.u;
Field3D& w      = global_simulation_storage.fields.w;
Field3D& v      = global_simulation_storage.fields.v;
Field3D& tracer = global_simulation_storage.fields.tracer;
Field3D& theta  = global_simulation_storage.fields.theta;
Field3D& qv     = global_simulation_storage.fields.qv;
Field3D& qc     = global_simulation_storage.fields.qc;
Field3D& qr     = global_simulation_storage.fields.qr;
Field3D& qi     = global_simulation_storage.fields.qi;
Field3D& qs     = global_simulation_storage.fields.qs;
Field3D& qh     = global_simulation_storage.fields.qh;
Field3D& qg     = global_simulation_storage.fields.qg;
Field3D& tke    = global_simulation_storage.fields.tke;

Field3D& dtheta_dt_rad = global_simulation_storage.tendencies.dtheta_dt_rad;
Field3D& dtheta_dt_pbl = global_simulation_storage.tendencies.dtheta_dt_pbl;
Field3D& dqv_dt_pbl    = global_simulation_storage.tendencies.dqv_dt_pbl;
Field3D& du_dt_pbl     = global_simulation_storage.tendencies.du_dt_pbl;
Field3D& dv_dt_pbl     = global_simulation_storage.tendencies.dv_dt_pbl;
Field3D& dtke_dt_pbl   = global_simulation_storage.tendencies.dtke_dt_pbl;

std::vector<float>& sfc_ustar              = global_simulation_storage.surface.ustar;
std::vector<float>& sfc_tau_u              = global_simulation_storage.surface.tau_u;
std::vector<float>& sfc_tau_v              = global_simulation_storage.surface.tau_v;
std::vector<float>& sfc_sensible_heat_flux = global_simulation_storage.surface.sensible_heat_flux;
std::vector<float>& sfc_latent_heat_flux   = global_simulation_storage.surface.latent_heat_flux;
std::vector<float>& sfc_moisture_flux      = global_simulation_storage.surface.moisture_flux;
std::vector<float>& pbl_height_2d          = global_simulation_storage.surface.pbl_height;

std::vector<double>& rho0_base   = global_simulation_storage.base.rho0;
std::vector<double>& p0_base     = global_simulation_storage.base.p0;
std::vector<double>& qv0_base    = global_simulation_storage.base.qv0;
std::vector<double>& theta0_base = global_simulation_storage.base.theta0;
std::vector<double>& u0_base     = global_simulation_storage.base.u0;
std::vector<double>& v0_base     = global_simulation_storage.base.v0;

double&     coriolis_f         = global_simulation_storage.coriolis_f;
double&     simulation_time    = global_simulation_storage.time_s;
LogProfile& global_log_profile = (global_simulation_storage.configuration.log_profile = LogProfile::quiet);

namespace
{
} // namespace

TEST_CASE("post-recycle PBL tendencies rebuild from shifted conserved sources",
          "[boundary_layer][runtime][domain_recycling]")
{
    SimulationStorageState storage{{5, 5, 3, 100.0, 50.0, 1.0, 100.0}};
    storage.grid.geometry.initialize(5, 5, 3, 100.0, 50.0, 100.0, CoordinateSystem::Cartesian, StaggerType::CGrid);
    storage.fields.rho       = Field3D(5, 5, 3, 2.0f);
    auto& runtime            = storage.runtime.boundary_layer;
    runtime.scheme           = std::make_unique<SlabScheme>();
    runtime.config.scheme_id = "slab";
    runtime.config.dt_pbl    = 60.0;
    runtime.scheme->initialize(runtime.config);
    runtime.held_u_force_n                  = Field3D(5, 5, 3, 1.0e6f);
    runtime.held_v_force_n                  = Field3D(5, 5, 3, -2.0e6f);
    runtime.held_theta_inventory_rate_kgk_s = Field3D(5, 5, 3, 3.0e6f);
    runtime.held_vapor_mass_rate_kg_s       = Field3D(5, 5, 3, 4.0e3f);
    runtime.updated_this_step               = false;
    storage.tendencies.dtke_dt_pbl          = Field3D(5, 5, 3, 0.125f);
    auto state                              = make_simulation_state_view(storage);

    rebuild_boundary_layer_tendencies_after_domain_recycling(state);
    const double cell_mass = 2.0 * 100.0 * 100.0 * 50.0;
    REQUIRE(storage.tendencies.dtheta_dt_pbl[1][1][0] == Approx(3.0e6 / cell_mass));
    REQUIRE(storage.tendencies.dqv_dt_pbl[1][1][0] == Approx(4.0e3 / cell_mass));
    REQUIRE(storage.tendencies.du_dt_pbl[1][1][0] == Approx(1.0e6 / cell_mass));
    REQUIRE(storage.tendencies.dv_dt_pbl[1][1][0] == Approx(-2.0e6 / cell_mass));
    REQUIRE(storage.tendencies.dtke_dt_pbl[1][1][0] == 0.125f);
    REQUIRE(std::string(tmv::boundary_layer_domain_recycling_finalize_algorithm_id()) ==
            "live_density_pbl_source_reconstruction_v1");

    const Field3D theta_before = storage.tendencies.dtheta_dt_pbl;
    runtime.updated_this_step  = true;
    REQUIRE_THROWS_WITH(rebuild_boundary_layer_tendencies_after_domain_recycling(state),
                        Catch::Contains("clear its cadence event flag"));
    for (std::size_t index = 0; index < theta_before.size(); ++index)
        REQUIRE(storage.tendencies.dtheta_dt_pbl.data()[index] == theta_before.data()[index]);
}
tmv::init::config::LateralBC& global_lateral_bc_west  = global_simulation_storage.configuration.lateral_bc_west;
tmv::init::config::LateralBC& global_lateral_bc_east  = global_simulation_storage.configuration.lateral_bc_east;
tmv::init::config::LateralBC& global_lateral_bc_south = global_simulation_storage.configuration.lateral_bc_south;
tmv::init::config::LateralBC& global_lateral_bc_north = global_simulation_storage.configuration.lateral_bc_north;

namespace
{

class ThrowingBoundaryLayerScheme final : public BoundaryLayerSchemeBase
{
public:
    std::unique_ptr<BoundaryLayerSchemeBase> clone() const override
    {
        return std::make_unique<ThrowingBoundaryLayerScheme>();
    }

    std::string name() const override
    {
        return "throwing_test_scheme";
    }
    int required_fields() const override
    {
        return static_cast<int>(BoundaryLayerRequirements::BASIC);
    }
    void initialize(const BoundaryLayerConfig&) override
    {
    }
    void compute_column(const BoundaryLayerConfig&,
                        const SurfaceConfig&,
                        const BoundaryLayerColumnStateView& col,
                        BoundaryLayerTendencies&            tend,
                        BoundaryLayerDiagnostics*) override
    {
        tend.dudt_pbl.assign(col.u.size(), 123.0);
        if (col.slab_state)
        {
            col.slab_state->h_m         = 9999.0;
            col.slab_state->initialized = true;
        }
        throw std::runtime_error("injected PBL column failure");
    }
};

class GeometryProbeBoundaryLayerScheme final : public BoundaryLayerSchemeBase
{
public:
    std::unique_ptr<BoundaryLayerSchemeBase> clone() const override
    {
        return std::make_unique<GeometryProbeBoundaryLayerScheme>(*this);
    }
    std::string name() const override
    {
        return "geometry_probe";
    }
    int required_fields() const override
    {
        return static_cast<int>(BoundaryLayerRequirements::BASIC);
    }
    void initialize(const BoundaryLayerConfig&) override
    {
    }
    void compute_column(const BoundaryLayerConfig&,
                        const SurfaceConfig&,
                        const BoundaryLayerColumnStateView& col,
                        BoundaryLayerTendencies&            tend,
                        BoundaryLayerDiagnostics*           diagnostics) override
    {
        if (col.z_int == nullptr)
            throw std::invalid_argument("geometry probe received no interfaces");
        const std::size_t nz_col = col.theta.size();
        if (col.z_int.size() != nz_col + 1)
            throw std::invalid_argument("geometry probe received malformed interfaces");
        tend.dthetadt_pbl.resize(nz_col);
        tend.dqvdt_pbl.assign(nz_col, 0.0);
        tend.dqcdt_pbl.assign(nz_col, 0.0);
        tend.dqidt_pbl.assign(nz_col, 0.0);
        tend.dudt_pbl.assign(nz_col, col.surface_slope_x);
        tend.dvdt_pbl.assign(nz_col, col.surface_slope_y);
        tend.dtkedt_pbl.assign(nz_col, 0.0);
        for (std::size_t k = 0; k < nz_col; ++k)
            tend.dthetadt_pbl[k] = col.z_int[k + 1] - col.z_int[k]
                + col.u_sfc + col.v_sfc;
        if (diagnostics)
            *diagnostics = {};
    }
};

struct MynnRuntimeBatchStats
{
    std::atomic<int> dispatches{0};
    bool             reject = false;
};

class MynnRuntimeBatchKernels final : public BoundaryLayerComputeKernels
{
public:
    explicit MynnRuntimeBatchKernels(std::shared_ptr<MynnRuntimeBatchStats> stats) : stats_(std::move(stats))
    {
    }
    bool supports(BoundaryLayerKernelOp op) const override
    {
        return op == BoundaryLayerKernelOp::MynnLevel25Stability;
    }
    bool dispatch_mynn_level25_stability(const MynnLevel25StabilityDispatch& dispatch) override
    {
        ++stats_->dispatches;
        if (stats_->reject)
            return false;
        for (int index = 0; index < dispatch.count; ++index)
        {
            const auto result  = tmv::mynn::level25_stability_functions({dispatch.gm_s2[index],
                                                                         dispatch.gh_s2[index],
                                                                         dispatch.mixing_length_m[index],
                                                                         dispatch.q_squared_m2s2[index]});
            dispatch.sm[index] = static_cast<float>(result.sm);
            dispatch.sh[index] = static_cast<float>(result.sh);
            dispatch.gradient_richardson[index] = static_cast<float>(result.gradient_richardson);
            dispatch.flux_richardson[index]     = static_cast<float>(result.flux_richardson);
            dispatch.equilibrium_limiter[index] = static_cast<float>(result.equilibrium_limiter);
        }
        return true;
    }

private:
    std::shared_ptr<MynnRuntimeBatchStats> stats_;
};

class MynnRuntimeBatchBackend final : public ComputeBackend
{
public:
    explicit MynnRuntimeBatchBackend(std::shared_ptr<MynnRuntimeBatchStats> stats) : kernels_(std::move(stats))
    {
    }
    std::string name() const override
    {
        return "mynn_batch_probe";
    }
    bool initialize(std::string& error) override
    {
        error.clear();
        return true;
    }
    void shutdown() override
    {
    }
    BoundaryLayerComputeKernels* boundary_layer_kernels() override
    {
        return &kernels_;
    }
    const BoundaryLayerComputeKernels* boundary_layer_kernels() const override
    {
        return &kernels_;
    }

private:
    MynnRuntimeBatchKernels kernels_;
};

void initialize_test_grid()
{
    NR     = 3;
    NTH    = 2;
    NZ     = 6;
    dr     = 1000.0;
    dz     = 100.0;
    dt     = 1.0;
    dtheta = 1000.0;
    global_grid_geometry.initialize(NR, NTH, NZ, dr, dz, dtheta, CoordinateSystem::Cartesian);
}

void resize_global_fields(float u_value, float theta_value)
{
    p.resize(NR, NTH, NZ, 100000.0f);
    theta.resize(NR, NTH, NZ, theta_value);
    rho.resize(NR, NTH, NZ, 1.1f);
    qv.resize(NR, NTH, NZ, 0.010f);
    u.resize(NR, NTH, NZ, u_value);
    v.resize(NR, NTH, NZ, 0.0f);
    w.resize(NR, NTH, NZ, 0.0f);
    tracer.resize(NR, NTH, NZ, 0.0f);
    qc.resize(NR, NTH, NZ, 0.0f);
    qr.resize(NR, NTH, NZ, 0.0f);
    qi.resize(NR, NTH, NZ, 0.0f);
    qs.resize(NR, NTH, NZ, 0.0f);
    qh.resize(NR, NTH, NZ, 0.0f);
    qg.resize(NR, NTH, NZ, 0.0f);
    tke = Field3D();
}

void copy_local_to_global(const Field3D& local_p,
                          const Field3D& local_theta,
                          const Field3D& local_rho,
                          const Field3D& local_qv,
                          const Field3D& local_u,
                          const Field3D& local_v)
{
    p     = local_p;
    theta = local_theta;
    rho   = local_rho;
    qv    = local_qv;
    u     = local_u;
    v     = local_v;
}

bool any_nonzero(const Field3D& field)
{
    for (int i = 0; i < static_cast<int>(field.size_r()); ++i)
    {
        for (int j = 0; j < static_cast<int>(field.size_th()); ++j)
        {
            for (int k = 0; k < static_cast<int>(field.size_z()); ++k)
            {
                if (std::abs(static_cast<double>(field[i][j][k])) > 0.0)
                {
                    return true;
                }
            }
        }
    }
    return false;
}

bool any_nonzero(const std::vector<float>& values)
{
    for (float value : values)
    {
        if (std::abs(static_cast<double>(value)) > 0.0)
        {
            return true;
        }
    }
    return false;
}

bool fields_match(const Field3D& lhs, const Field3D& rhs, double tolerance)
{
    if (lhs.size() != rhs.size())
    {
        return false;
    }
    for (int i = 0; i < static_cast<int>(lhs.size_r()); ++i)
    {
        for (int j = 0; j < static_cast<int>(lhs.size_th()); ++j)
        {
            for (int k = 0; k < static_cast<int>(lhs.size_z()); ++k)
            {
                if (std::abs(static_cast<double>(lhs[i][j][k] - rhs[i][j][k])) > tolerance)
                {
                    return false;
                }
            }
        }
    }
    return true;
}

bool vectors_match(const std::vector<float>& lhs, const std::vector<float>& rhs, double tolerance)
{
    if (lhs.size() != rhs.size())
    {
        return false;
    }
    for (std::size_t idx = 0; idx < lhs.size(); ++idx)
    {
        if (std::abs(static_cast<double>(lhs[idx] - rhs[idx])) > tolerance)
        {
            return false;
        }
    }
    return true;
}

BoundaryLayerConfig runtime_test_config()
{
    BoundaryLayerConfig cfg;
    cfg.scheme_id            = "slab";
    cfg.dt_pbl               = 1.0;
    cfg.apply_surface_fluxes = true;
    cfg.pbl_max_height       = 2000.0;
    return cfg;
}

SurfaceConfig runtime_test_surface()
{
    SurfaceConfig sfc;
    sfc.Tsfc = 303.0;
    sfc.qsfc = 0.014;
    return sfc;
}

void reset_boundary_layer_runtime(const BoundaryLayerConfig& cfg, const SurfaceConfig& sfc)
{
    initialize_boundary_layer("slab", cfg, sfc);
}

std::shared_ptr<MynnRuntimeBatchStats> prepare_mynn_batch_run(int columns_x, bool allow_fallback, bool reject)
{
    NR     = columns_x;
    NTH    = 2;
    NZ     = 6;
    dr     = 1000.0;
    dz     = 100.0;
    dt     = 1.0;
    dtheta = 1000.0;
    global_grid_geometry.initialize(NR, NTH, NZ, dr, dz, dtheta, CoordinateSystem::Cartesian);
    resize_global_fields(7.0f, 300.0f);
    for (int i = 0; i < NR; ++i)
        for (int j = 0; j < NTH; ++j)
            for (int k = 0; k < NZ; ++k)
            {
                p[i][j][k] = static_cast<float>(100000.0 * std::exp(-0.015 * k));
                theta[i][j][k] += static_cast<float>(0.35 * k);
                u[i][j][k] += static_cast<float>(0.6 * k + 0.01 * i);
                qv[i][j][k]  = static_cast<float>(0.010 * std::exp(-0.1 * k));
                rho[i][j][k] = static_cast<float>(1.1 - 0.06 * k);
            }
    tke.resize(NR, NTH, NZ, 0.15f);
    BoundaryLayerConfig config = runtime_test_config();
    config.scheme_id           = "mynn";
    config.dt_pbl              = 30.0;
    config.enable_tke          = true;
    initialize_boundary_layer("mynn", config, runtime_test_surface());

    auto stats    = std::make_shared<MynnRuntimeBatchStats>();
    stats->reject = reject;
    auto& compute = global_simulation_storage.runtime.compute;
    compute.reset();
    compute.backend               = std::make_unique<MynnRuntimeBatchBackend>(stats);
    compute.active_kind           = ComputeBackendKind::Metal;
    compute.requested_kind        = ComputeBackendKind::Metal;
    compute.fallback_active       = false;
    compute.config.backend        = "metal";
    compute.config.allow_fallback = allow_fallback;
    return stats;
}

} // namespace

namespace
{
struct ComputeRuntimeReset
{
    ~ComputeRuntimeReset()
    {
        global_simulation_storage.runtime.compute.reset();
    }
};
}

TEST_CASE("runtime MYNN batching is bounded and obeys strict fallback policy",
          "[boundary_layer][runtime][mynn][batch][backend]")
{
    const ComputeRuntimeReset reset_compute;
    SECTION("bounded native batches publish complete tendencies")
    {
        const auto stats = prepare_mynn_batch_run(260, false, false);
        REQUIRE_NOTHROW(step_boundary_layer(0.0));
        const int physical_columns = (NR - 2) * NTH;
        REQUIRE(stats->dispatches.load() == (physical_columns + 255) / 256);
        REQUIRE(any_nonzero(dtheta_dt_pbl));
        REQUIRE(any_nonzero(sfc_ustar));
    }

    SECTION("strict dispatch rejection leaves published state unchanged")
    {
        const auto stats = prepare_mynn_batch_run(3, false, true);
        dtheta_dt_pbl.resize(NR, NTH, NZ, 81.0f);
        REQUIRE_THROWS_WITH(step_boundary_layer(0.0), "native boundary-layer batch dispatch was rejected");
        REQUIRE(stats->dispatches.load() == 1);
        for (std::size_t index = 0; index < dtheta_dt_pbl.size(); ++index)
            REQUIRE(dtheta_dt_pbl.data()[index] == 81.0f);
    }

    SECTION("permitted fallback recomputes through CPU columns")
    {
        const auto stats = prepare_mynn_batch_run(3, true, true);
        REQUIRE_NOTHROW(step_boundary_layer(0.0));
        REQUIRE(stats->dispatches.load() == 1);
        REQUIRE(any_nonzero(dtheta_dt_pbl));
    }
}

TEST_CASE("native MYNN runtime batching is bitwise invariant across OpenMP workers",
          "[boundary_layer][runtime][mynn][batch][parallel][determinism]")
{
    const ComputeRuntimeReset reset_compute;
#ifdef _OPENMP
    const int original_threads = omp_get_max_threads();
    const int original_dynamic = omp_get_dynamic();
    struct OpenMpRestore
    {
        int threads;
        int dynamic;
        ~OpenMpRestore()
        {
            omp_set_dynamic(dynamic);
            omp_set_num_threads(threads);
        }
    } restore{original_threads, original_dynamic};
    omp_set_dynamic(0);
    omp_set_num_threads(1);
#endif

    const auto serial_stats = prepare_mynn_batch_run(260, false, false);
    REQUIRE_NOTHROW(step_boundary_layer(0.0));
    const Field3D            serial_dtheta     = dtheta_dt_pbl;
    const Field3D            serial_dqv        = dqv_dt_pbl;
    const Field3D            serial_du         = du_dt_pbl;
    const Field3D            serial_dv         = dv_dt_pbl;
    const Field3D            serial_dtke       = dtke_dt_pbl;
    const std::vector<float> serial_ustar      = sfc_ustar;
    const int                serial_dispatches = serial_stats->dispatches.load();

#ifdef _OPENMP
    omp_set_num_threads(4);
#endif
    const auto parallel_stats = prepare_mynn_batch_run(260, false, false);
    REQUIRE_NOTHROW(step_boundary_layer(0.0));
    REQUIRE(parallel_stats->dispatches.load() == serial_dispatches);
    REQUIRE(fields_match(dtheta_dt_pbl, serial_dtheta, 0.0));
    REQUIRE(fields_match(dqv_dt_pbl, serial_dqv, 0.0));
    REQUIRE(fields_match(du_dt_pbl, serial_du, 0.0));
    REQUIRE(fields_match(dv_dt_pbl, serial_dv, 0.0));
    REQUIRE(fields_match(dtke_dt_pbl, serial_dtke, 0.0));
    REQUIRE(vectors_match(sfc_ustar, serial_ustar, 0.0));
}

TEST_CASE("legacy PBL globals reference the owning runtime", "[physics][boundary_layer][ownership]")
{
    REQUIRE(&boundary_layer_scheme == &tmv::global_boundary_layer_runtime_state.scheme);
    REQUIRE(&global_boundary_layer_config == &tmv::global_boundary_layer_runtime_state.config);
    REQUIRE(&global_surface_config == &tmv::global_boundary_layer_runtime_state.surface_config);

    const BoundaryLayerConfig cfg     = runtime_test_config();
    const SurfaceConfig       surface = runtime_test_surface();
    initialize_boundary_layer("slab", cfg, surface);
    REQUIRE(tmv::global_boundary_layer_runtime_state.scheme != nullptr);
    REQUIRE(tmv::global_boundary_layer_runtime_state.scheme->name() == "slab");
}

TEST_CASE("boundary-layer runtime initialization rejects invalid configuration",
          "[physics][boundary_layer][config-truth]")
{
    BoundaryLayerRuntimeState runtime;
    BoundaryLayerConfig       cfg = runtime_test_config();
    SurfaceConfig             sfc = runtime_test_surface();

    SECTION("cadence")
    {
        cfg.dt_pbl = 0.0;
        REQUIRE_THROWS_AS(initialize_boundary_layer(runtime, "slab", cfg, sfc), std::invalid_argument);
    }
    SECTION("PBL height")
    {
        cfg.pbl_max_height = std::numeric_limits<double>::quiet_NaN();
        REQUIRE_THROWS_AS(initialize_boundary_layer(runtime, "slab", cfg, sfc), std::invalid_argument);
    }
    SECTION("surface layer")
    {
        cfg.surface_layer_id = "invented";
        REQUIRE_THROWS_AS(initialize_boundary_layer(runtime, "slab", cfg, sfc), std::invalid_argument);
    }
    SECTION("potential-temperature PBL-top diagnostic")
    {
        cfg.pbl_top_method = "theta";
        REQUIRE_NOTHROW(initialize_boundary_layer(runtime, "slab", cfg, sfc));
    }
    SECTION("TKE PBL-top diagnostic requires a prognostic-TKE scheme")
    {
        cfg.pbl_top_method = "tke";
        REQUIRE_THROWS_AS(initialize_boundary_layer(runtime, "slab", cfg, sfc), std::invalid_argument);
    }
    SECTION("TKE PBL-top diagnostic is accepted by MYNN")
    {
        cfg.enable_tke = true;
        cfg.pbl_top_method = "tke";
        REQUIRE_NOTHROW(initialize_boundary_layer(runtime, "mynn", cfg, sfc));
    }
    SECTION("unknown PBL-top diagnostic")
    {
        cfg.pbl_top_method = "invented";
        REQUIRE_THROWS_AS(initialize_boundary_layer(runtime, "slab", cfg, sfc), std::invalid_argument);
    }
    SECTION("unimplemented cloud-aware closure")
    {
        cfg.clear_sky_only = false;
        REQUIRE_THROWS_AS(initialize_boundary_layer(runtime, "slab", cfg, sfc), std::invalid_argument);
    }
    SECTION("TKE switch does not match selected scheme")
    {
        cfg.enable_tke = true;
        REQUIRE_THROWS_AS(initialize_boundary_layer(runtime, "slab", cfg, sfc), std::invalid_argument);
    }
    SECTION("roughness")
    {
        sfc.z0m = -0.1;
        REQUIRE_THROWS_AS(initialize_boundary_layer(runtime, "slab", cfg, sfc), std::invalid_argument);
    }
    SECTION("surface temperature")
    {
        sfc.Tsfc = 500.0;
        REQUIRE_THROWS_AS(initialize_boundary_layer(runtime, "slab", cfg, sfc), std::invalid_argument);
    }
    SECTION("surface moisture")
    {
        sfc.qsfc = 0.5;
        REQUIRE_THROWS_AS(initialize_boundary_layer(runtime, "slab", cfg, sfc), std::invalid_argument);
    }
}

TEST_CASE("first C-grid slab call captures an unset surface skin before use",
          "[physics][boundary_layer][surface][cgrid][regression]")
{
    initialize_test_grid();
    global_grid_geometry.initialize(NR, NTH, NZ, dr, dz, dtheta, CoordinateSystem::Cartesian, StaggerType::CGrid);
    resize_global_fields(5.0f, 300.0f);

    BoundaryLayerConfig cfg = runtime_test_config();
    SurfaceConfig       surface;
    REQUIRE(surface.Tsfc < 0.0);
    REQUIRE(surface.qsfc < 0.0);

    initialize_boundary_layer("slab", cfg, surface);
    REQUIRE_NOTHROW(step_boundary_layer(0.0));

    const auto&       runtime      = tmv::global_boundary_layer_runtime_state;
    const std::size_t surface_size = static_cast<std::size_t>(NR) * static_cast<std::size_t>(NTH);
    REQUIRE(runtime.skin_captured);
    REQUIRE(runtime.skin_theta_2d.size() == surface_size);
    REQUIRE(runtime.skin_qv_2d.size() == surface_size);
    const std::size_t physical_column = static_cast<std::size_t>(NTH + 1);
    REQUIRE(runtime.skin_theta_2d[physical_column] == Approx(300.0f));
    REQUIRE(runtime.skin_qv_2d[physical_column] == Approx(0.010f));
    REQUIRE(std::isfinite(sfc_sensible_heat_flux[physical_column]));
    REQUIRE(std::isfinite(sfc_moisture_flux[physical_column]));

    global_grid_geometry.initialize(NR, NTH, NZ, dr, dz, dtheta, CoordinateSystem::Cartesian, StaggerType::Collocated);
}

TEST_CASE("C-grid PBL centers winds and converts held forces on dual volumes",
          "[physics][boundary_layer][cgrid][momentum][terrain-ready]")
{
    NR     = 5;
    NTH    = 5;
    NZ     = 3;
    dr     = 1000.0;
    dz     = 100.0;
    dt     = 1.0;
    dtheta = 800.0;
    global_grid_geometry.initialize(NR, NTH, NZ, dr, dz, dtheta, CoordinateSystem::Cartesian, StaggerType::CGrid);
    resize_global_fields(0.0f, 300.0f);
    for (int i = 0; i < NR; ++i)
        for (int j = 0; j < NTH; ++j)
            for (int k = 0; k < NZ; ++k)
            {
                u[i][j][k] = static_cast<float>(2.0 * i);
                v[i][j][k] = static_cast<float>(j);
            }

    global_lateral_bc_west  = tmv::init::config::LateralBC::Periodic;
    global_lateral_bc_east  = tmv::init::config::LateralBC::Periodic;
    global_lateral_bc_south = tmv::init::config::LateralBC::Periodic;
    global_lateral_bc_north = tmv::init::config::LateralBC::Periodic;
    BoundaryLayerConfig cfg = runtime_test_config();
    SurfaceConfig       surface;
    initialize_boundary_layer("slab", cfg, surface);
    REQUIRE_NOTHROW(step_boundary_layer(0.0));

    constexpr int     ci = 2, cj = 2;
    const auto        expected_surface = surface_fluxes::compute_surface_fluxes(cfg,
                                                                                surface,
                                                                                /*u_center=*/3.0,
                                                                                /*v_center=*/1.5,
                                                                                /*theta_skin=*/300.0,
                                                                                /*qv_skin=*/0.01,
                                                                                /*theta_air=*/300.0,
                                                                                /*qv_air=*/0.01,
                                                                                /*p_air=*/100000.0,
                                                                                /*rho_air=*/1.1,
                                                                                /*first-center AGL=*/50.0);
    const std::size_t center_index     = static_cast<std::size_t>(ci) * NTH + cj;
    REQUIRE(sfc_tau_u[center_index] == Approx(expected_surface.tau_u).epsilon(2.0e-6));
    REQUIRE(sfc_tau_v[center_index] == Approx(expected_surface.tau_v).epsilon(2.0e-6));

    const auto&  runtime             = tmv::global_boundary_layer_runtime_state;
    const double uniform_mass        = 1.1 * dr * dtheta * dz;
    const double expected_internal_u = 0.5 *
                                       (static_cast<double>(runtime.held_u_force_n[ci][cj][0]) +
                                        static_cast<double>(runtime.held_u_force_n[ci + 1][cj][0])) /
                                       uniform_mass;
    REQUIRE(du_dt_pbl[ci][cj][0] == Approx(expected_internal_u).epsilon(2.0e-6));
    const double expected_seam_u = 0.5 *
                                   (static_cast<double>(runtime.held_u_force_n[3][cj][0]) +
                                    static_cast<double>(runtime.held_u_force_n[1][cj][0])) /
                                   uniform_mass;
    REQUIRE(du_dt_pbl[3][cj][0] == Approx(expected_seam_u).epsilon(2.0e-6));
    REQUIRE(du_dt_pbl[0][cj][0] == 0.0f);

    // Held scalar-column forces are converted with the current dual mass.
    // Storage-only scalar closures and unused outer velocity closures must not
    // participate in that conversion.
    for (int i = 1; i < NR - 1; ++i)
        for (int j = 1; j < NTH - 1; ++j)
            for (int k = 0; k < NZ; ++k)
                rho[i][j][k] = static_cast<float>(0.8 + 0.1 * i + 0.05 * j + 0.02 * k);
    const float poison = std::numeric_limits<float>::quiet_NaN();
    for (int k = 0; k < NZ; ++k)
    {
        for (int j = 0; j < NTH; ++j)
        {
            rho[0][j][k]      = poison;
            rho[NR - 1][j][k] = poison;
        }
        for (int i = 0; i < NR; ++i)
        {
            rho[i][0][k]       = poison;
            rho[i][NTH - 1][k] = poison;
        }
    }
    REQUIRE_NOTHROW(prepare_boundary_layer_tendencies_for_application(global_simulation_state()));
    const double left_mass       = static_cast<double>(rho[ci][cj][0]) * dr * dtheta * dz;
    const double right_mass      = static_cast<double>(rho[ci + 1][cj][0]) * dr * dtheta * dz;
    const double expected_live_u = (static_cast<double>(runtime.held_u_force_n[ci][cj][0]) +
                                    static_cast<double>(runtime.held_u_force_n[ci + 1][cj][0])) /
                                   (left_mass + right_mass);
    REQUIRE(du_dt_pbl[ci][cj][0] == Approx(expected_live_u).epsilon(2.0e-6));
    REQUIRE(du_dt_pbl[0][cj][0] == 0.0f);

    global_lateral_bc_west  = tmv::init::config::LateralBC::ZeroGradient;
    global_lateral_bc_east  = tmv::init::config::LateralBC::ZeroGradient;
    global_lateral_bc_south = tmv::init::config::LateralBC::ZeroGradient;
    global_lateral_bc_north = tmv::init::config::LateralBC::ZeroGradient;
}

TEST_CASE("active A-grid PBL schemes consume only physical columns and preserve closure planes",
          "[physics][boundary_layer][agrid][ownership][closure]")
{
    constexpr int nx = 6, ny = 6, nz_local = 7;
    const float poison = std::numeric_limits<float>::quiet_NaN();
    for (const std::string& scheme_id : {
             std::string("slab"), std::string("ysu"), std::string("mynn")})
    {
        DYNAMIC_SECTION("scheme=" << scheme_id)
        {
            SimulationStorageState storage{{
                nx, ny, nz_local, 1000.0, 100.0, 1.0, 800.0}};
            storage.grid.geometry.initialize(
                nx, ny, nz_local, 1000.0, 100.0, 800.0,
                CoordinateSystem::Cartesian, StaggerType::Collocated);
            auto& fields = storage.fields;
            fields.p = Field3D(nx, ny, nz_local, poison);
            fields.theta = Field3D(nx, ny, nz_local, poison);
            fields.qv = Field3D(nx, ny, nz_local, poison);
            fields.rho = Field3D(nx, ny, nz_local, poison);
            fields.u = Field3D(nx, ny, nz_local, poison);
            fields.v = Field3D(nx, ny, nz_local, poison);
            if (scheme_id == "mynn")
                fields.tke = Field3D(nx, ny, nz_local, poison);

            const PhysicalCellDomain domain = physical_scalar_domain(
                nx, ny, nz_local, StaggerType::Collocated);
            REQUIRE(domain.k_begin == 1);
            REQUIRE(domain.k_end == nz_local - 1);
            for (int i = domain.i_begin; i < domain.i_end; ++i)
                for (int j = domain.j_begin; j < domain.j_end; ++j)
                    for (int k = domain.k_begin; k < domain.k_end; ++k)
                    {
                        const int physical_k = k - domain.k_begin;
                        fields.p(i, j, k) = static_cast<float>(
                            100000.0 * std::exp(-0.012 * physical_k));
                        fields.theta(i, j, k) = static_cast<float>(
                            299.0 + 0.45 * physical_k);
                        fields.qv(i, j, k) = static_cast<float>(
                            0.011 * std::exp(-0.08 * physical_k));
                        fields.rho(i, j, k) = static_cast<float>(
                            1.15 - 0.06 * physical_k);
                        fields.u(i, j, k) = static_cast<float>(
                            4.0 + 0.8 * physical_k);
                        fields.v(i, j, k) = static_cast<float>(
                            -1.0 + 0.2 * physical_k);
                        if (!fields.tke.empty())
                            fields.tke(i, j, k) = static_cast<float>(
                                0.12 + 0.02 * physical_k);
                    }

            BoundaryLayerConfig cfg = runtime_test_config();
            cfg.scheme_id = scheme_id;
            cfg.enable_tke = scheme_id == "mynn";
            cfg.dt_pbl = 30.0;
            initialize_boundary_layer(
                storage.runtime.boundary_layer, scheme_id, cfg,
                runtime_test_surface());
            auto state = make_simulation_state_view(storage);
            REQUIRE_NOTHROW(step_boundary_layer(state, 0.0));

            bool physical_force = false;
            for (int i = 0; i < nx; ++i)
                for (int j = 0; j < ny; ++j)
                    for (int k = 0; k < nz_local; ++k)
                    {
                        const bool physical =
                            i >= domain.i_begin && i < domain.i_end &&
                            j >= domain.j_begin && j < domain.j_end &&
                            k >= domain.k_begin && k < domain.k_end;
                        const Field3D* outputs[] = {
                            &storage.tendencies.dtheta_dt_pbl,
                            &storage.tendencies.dqv_dt_pbl,
                            &storage.tendencies.du_dt_pbl,
                            &storage.tendencies.dv_dt_pbl,
                            &storage.tendencies.dtke_dt_pbl,
                            &storage.runtime.boundary_layer.held_u_force_n,
                            &storage.runtime.boundary_layer.held_v_force_n,
                            &storage.runtime.boundary_layer
                                 .held_theta_inventory_rate_kgk_s,
                            &storage.runtime.boundary_layer
                                 .held_vapor_mass_rate_kg_s};
                        for (const Field3D* output : outputs)
                        {
                            REQUIRE(std::isfinite((*output)(i, j, k)));
                            if (!physical)
                                REQUIRE((*output)(i, j, k) == 0.0f);
                        }
                        if (physical &&
                            (storage.runtime.boundary_layer.held_u_force_n(
                                 i, j, k) != 0.0f ||
                             storage.runtime.boundary_layer.held_v_force_n(
                                 i, j, k) != 0.0f))
                            physical_force = true;
                    }
            REQUIRE(physical_force);

            const Field3D du_before = storage.tendencies.du_dt_pbl;
            REQUIRE_NOTHROW(
                prepare_boundary_layer_tendencies_for_application(state));
            for (int i = 0; i < nx; ++i)
                for (int j = 0; j < ny; ++j)
                    for (int k : {0, nz_local - 1})
                    {
                        REQUIRE(storage.tendencies.du_dt_pbl(i, j, k) ==
                                0.0f);
                        REQUIRE(storage.tendencies.dv_dt_pbl(i, j, k) ==
                                0.0f);
                    }
            REQUIRE(fields_match(
                du_before, storage.tendencies.du_dt_pbl, 2.0e-6));
        }
    }
}

TEST_CASE("failed boundary-layer columns do not publish partial runtime state",
          "[physics][boundary_layer][transactional]")
{
    initialize_test_grid();
    resize_global_fields(5.0f, 300.0f);
    const BoundaryLayerConfig cfg = runtime_test_config();
    const SurfaceConfig       sfc = runtime_test_surface();
    reset_boundary_layer_runtime(cfg, sfc);

    dtheta_dt_pbl                  = Field3D(NR, NTH, NZ, 1.0f);
    dqv_dt_pbl                     = Field3D(NR, NTH, NZ, 2.0f);
    du_dt_pbl                      = Field3D(NR, NTH, NZ, 3.0f);
    dv_dt_pbl                      = Field3D(NR, NTH, NZ, 4.0f);
    dtke_dt_pbl                    = Field3D(NR, NTH, NZ, 5.0f);
    const std::size_t surface_size = static_cast<std::size_t>(NR) * static_cast<std::size_t>(NTH);
    sfc_ustar.assign(surface_size, 6.0f);
    sfc_tau_u.assign(surface_size, 7.0f);
    sfc_tau_v.assign(surface_size, 8.0f);
    sfc_sensible_heat_flux.assign(surface_size, 9.0f);
    sfc_latent_heat_flux.assign(surface_size, 10.0f);
    sfc_moisture_flux.assign(surface_size, 11.0f);
    pbl_height_2d.assign(surface_size, 12.0f);

    auto& runtime              = tmv::global_boundary_layer_runtime_state;
    runtime.scheme             = std::make_unique<ThrowingBoundaryLayerScheme>();
    runtime.last_update_time_s = 2.0;
    runtime.next_update_time_s = 3.0;
    runtime.updated_this_step  = true;
    runtime.skin_nr            = NR;
    runtime.skin_nth           = NTH;
    runtime.skin_theta_2d.assign(surface_size, 301.0f);
    runtime.skin_qv_2d.assign(surface_size, 0.012f);
    runtime.skin_captured = true;
    runtime.slab_nr       = NR;
    runtime.slab_nth      = NTH;
    runtime.slab_columns.assign(surface_size, SlabColumnState{});
    runtime.slab_columns.front().h_m = 321.0;

    const Field3D                      before_dtheta     = dtheta_dt_pbl;
    const Field3D                      before_dqv        = dqv_dt_pbl;
    const Field3D                      before_du         = du_dt_pbl;
    const Field3D                      before_dv         = dv_dt_pbl;
    const Field3D                      before_dtke       = dtke_dt_pbl;
    const std::vector<float>           before_ustar      = sfc_ustar;
    const std::vector<float>           before_tau_u      = sfc_tau_u;
    const std::vector<float>           before_tau_v      = sfc_tau_v;
    const std::vector<float>           before_sensible   = sfc_sensible_heat_flux;
    const std::vector<float>           before_latent     = sfc_latent_heat_flux;
    const std::vector<float>           before_moisture   = sfc_moisture_flux;
    const std::vector<float>           before_height     = pbl_height_2d;
    const std::vector<float>           before_skin_theta = runtime.skin_theta_2d;
    const std::vector<float>           before_skin_qv    = runtime.skin_qv_2d;
    const std::vector<SlabColumnState> before_slab       = runtime.slab_columns;

    REQUIRE_THROWS_WITH(step_boundary_layer(3.0), Catch::Contains("injected PBL column failure"));
    REQUIRE(fields_match(before_dtheta, dtheta_dt_pbl, 0.0));
    REQUIRE(fields_match(before_dqv, dqv_dt_pbl, 0.0));
    REQUIRE(fields_match(before_du, du_dt_pbl, 0.0));
    REQUIRE(fields_match(before_dv, dv_dt_pbl, 0.0));
    REQUIRE(fields_match(before_dtke, dtke_dt_pbl, 0.0));
    REQUIRE(vectors_match(before_ustar, sfc_ustar, 0.0));
    REQUIRE(vectors_match(before_tau_u, sfc_tau_u, 0.0));
    REQUIRE(vectors_match(before_tau_v, sfc_tau_v, 0.0));
    REQUIRE(vectors_match(before_sensible, sfc_sensible_heat_flux, 0.0));
    REQUIRE(vectors_match(before_latent, sfc_latent_heat_flux, 0.0));
    REQUIRE(vectors_match(before_moisture, sfc_moisture_flux, 0.0));
    REQUIRE(vectors_match(before_height, pbl_height_2d, 0.0));
    REQUIRE(vectors_match(before_skin_theta, runtime.skin_theta_2d, 0.0));
    REQUIRE(vectors_match(before_skin_qv, runtime.skin_qv_2d, 0.0));
    REQUIRE(runtime.slab_columns.size() == before_slab.size());
    REQUIRE(runtime.slab_columns.front().h_m == before_slab.front().h_m);
    REQUIRE(runtime.last_update_time_s == 2.0);
    REQUIRE(runtime.next_update_time_s == 3.0);
    REQUIRE(runtime.updated_this_step);
}

TEST_CASE("cloud-top YSU runtime is threaded deterministic and failure atomic",
          "[physics][boundary_layer][ysu][cloud-top][runtime][transactional]")
{
#ifdef _OPENMP
    const int original_threads = omp_get_max_threads();
    const int original_dynamic = omp_get_dynamic();
    struct OpenMpRestore
    {
        int threads;
        int dynamic;
        ~OpenMpRestore()
        {
            omp_set_dynamic(dynamic);
            omp_set_num_threads(threads);
        }
    } restore{original_threads, original_dynamic};
    omp_set_dynamic(0);
#endif

    struct Result
    {
        Field3D dtheta;
        Field3D dqv;
        Field3D dqc;
        Field3D dqi;
        Field3D du;
        Field3D dv;
        Field3D held_theta;
        Field3D held_vapor;
        Field3D held_cloud_liquid;
        Field3D held_cloud_ice;
    };

    const auto prepare = []
    {
        SimulationStorageState storage{{
            12, 12, 10, 1000.0, 100.0, 1.0, 1000.0}};
        storage.grid.geometry.initialize(
            12, 12, 10, 1000.0, 100.0, 1000.0,
            CoordinateSystem::Cartesian, StaggerType::CGrid);
        auto& fields = storage.fields;
        fields.p = Field3D(12, 12, 10, 0.0f);
        fields.theta = Field3D(12, 12, 10, 0.0f);
        fields.qv = Field3D(12, 12, 10, 0.0f);
        fields.rho = Field3D(12, 12, 10, 0.0f);
        fields.u = Field3D(12, 12, 10, 0.0f);
        fields.v = Field3D(12, 12, 10, 0.0f);
        fields.qc = Field3D(12, 12, 10, 0.0f);
        fields.qi = Field3D(12, 12, 10, 0.0f);
        storage.tendencies.dtheta_dt_rad =
            Field3D(12, 12, 10, 0.0f);
        for (int i = 0; i < 12; ++i)
            for (int j = 0; j < 12; ++j)
                for (int k = 0; k < 10; ++k)
                {
                    fields.p[i][j][k] = static_cast<float>(
                        100000.0 * std::exp(-0.013 * k));
                    fields.theta[i][j][k] = static_cast<float>(
                        299.0 + 0.02 * k + 0.001 * i);
                    fields.qv[i][j][k] = static_cast<float>(
                        0.012 - 0.0004 * k + 1.0e-6 * j);
                    fields.rho[i][j][k] = static_cast<float>(
                        1.18 - 0.03 * k);
                    fields.u[i][j][k] = static_cast<float>(
                        4.0 + 0.25 * k + 0.01 * i);
                    fields.v[i][j][k] = static_cast<float>(
                        -1.0 + 0.1 * k - 0.01 * j);
                    if (k == 5 || k == 6)
                        fields.qc[i][j][k] = static_cast<float>(
                            2.0e-4 - 5.0e-5 * (k - 5));
                    if (k == 5)
                        fields.qi[i][j][k] = 7.5e-5f;
                    if (k == 6)
                        storage.tendencies.dtheta_dt_rad[i][j][k] =
                            -2.0e-3f;
                }

        BoundaryLayerConfig config;
        config.scheme_id = "ysu";
        config.dt_pbl = 10.0;
        config.apply_surface_fluxes = false;
        config.clear_sky_only = false;
        config.pbl_top_method = "theta";
        config.pbl_max_height = 1000.0;
        initialize_boundary_layer(
            storage.runtime.boundary_layer, "ysu", config,
            SurfaceConfig{});
        return storage;
    };

    const auto run = [&](int threads)
    {
#ifdef _OPENMP
        omp_set_num_threads(threads);
#else
        (void)threads;
#endif
        auto storage = prepare();
        auto state = make_simulation_state_view(storage);
        step_boundary_layer(state, 0.0);
        REQUIRE(any_nonzero(storage.tendencies.dtheta_dt_pbl));
        REQUIRE(any_nonzero(storage.tendencies.dqv_dt_pbl));
        REQUIRE(any_nonzero(storage.tendencies.dqc_dt_pbl));
        REQUIRE(any_nonzero(storage.tendencies.dqi_dt_pbl));
        return Result{
            storage.tendencies.dtheta_dt_pbl,
            storage.tendencies.dqv_dt_pbl,
            storage.tendencies.dqc_dt_pbl,
            storage.tendencies.dqi_dt_pbl,
            storage.tendencies.du_dt_pbl,
            storage.tendencies.dv_dt_pbl,
            storage.runtime.boundary_layer
                .held_theta_inventory_rate_kgk_s,
            storage.runtime.boundary_layer.held_vapor_mass_rate_kg_s,
            storage.runtime.boundary_layer
                .held_cloud_liquid_mass_rate_kg_s,
            storage.runtime.boundary_layer.held_cloud_ice_mass_rate_kg_s};
    };

    const Result serial = run(1);
    const Result parallel = run(4);
    REQUIRE(fields_match(serial.dtheta, parallel.dtheta, 0.0));
    REQUIRE(fields_match(serial.dqv, parallel.dqv, 0.0));
    REQUIRE(fields_match(serial.dqc, parallel.dqc, 0.0));
    REQUIRE(fields_match(serial.dqi, parallel.dqi, 0.0));
    REQUIRE(fields_match(serial.du, parallel.du, 0.0));
    REQUIRE(fields_match(serial.dv, parallel.dv, 0.0));
    REQUIRE(fields_match(serial.held_theta, parallel.held_theta, 0.0));
    REQUIRE(fields_match(serial.held_vapor, parallel.held_vapor, 0.0));
    REQUIRE(fields_match(serial.held_cloud_liquid,
                         parallel.held_cloud_liquid, 0.0));
    REQUIRE(fields_match(serial.held_cloud_ice,
                         parallel.held_cloud_ice, 0.0));

    auto failed = prepare();
    auto failed_state = make_simulation_state_view(failed);
    step_boundary_layer(failed_state, 0.0);
    const Field3D before_dtheta = failed.tendencies.dtheta_dt_pbl;
    const Field3D before_dqv = failed.tendencies.dqv_dt_pbl;
    const Field3D before_dqc = failed.tendencies.dqc_dt_pbl;
    const Field3D before_dqi = failed.tendencies.dqi_dt_pbl;
    const Field3D before_du = failed.tendencies.du_dt_pbl;
    const Field3D before_dv = failed.tendencies.dv_dt_pbl;
    const Field3D before_held_theta = failed.runtime.boundary_layer
        .held_theta_inventory_rate_kgk_s;
    const Field3D before_held_vapor = failed.runtime.boundary_layer
        .held_vapor_mass_rate_kg_s;
    const Field3D before_held_cloud_liquid = failed.runtime.boundary_layer
        .held_cloud_liquid_mass_rate_kg_s;
    const Field3D before_held_cloud_ice = failed.runtime.boundary_layer
        .held_cloud_ice_mass_rate_kg_s;
    const double before_last = failed.runtime.boundary_layer.last_update_time_s;
    const double before_next = failed.runtime.boundary_layer.next_update_time_s;
    failed.fields.qc[1][1][5] = std::numeric_limits<float>::quiet_NaN();
    REQUIRE_THROWS_AS(step_boundary_layer(failed_state, 10.0),
                      std::invalid_argument);
    REQUIRE(fields_match(before_dtheta, failed.tendencies.dtheta_dt_pbl, 0.0));
    REQUIRE(fields_match(before_dqv, failed.tendencies.dqv_dt_pbl, 0.0));
    REQUIRE(fields_match(before_dqc, failed.tendencies.dqc_dt_pbl, 0.0));
    REQUIRE(fields_match(before_dqi, failed.tendencies.dqi_dt_pbl, 0.0));
    REQUIRE(fields_match(before_du, failed.tendencies.du_dt_pbl, 0.0));
    REQUIRE(fields_match(before_dv, failed.tendencies.dv_dt_pbl, 0.0));
    REQUIRE(fields_match(before_held_theta, failed.runtime.boundary_layer
        .held_theta_inventory_rate_kgk_s, 0.0));
    REQUIRE(fields_match(before_held_vapor, failed.runtime.boundary_layer
        .held_vapor_mass_rate_kg_s, 0.0));
    REQUIRE(fields_match(before_held_cloud_liquid,
        failed.runtime.boundary_layer.held_cloud_liquid_mass_rate_kg_s, 0.0));
    REQUIRE(fields_match(before_held_cloud_ice,
        failed.runtime.boundary_layer.held_cloud_ice_mass_rate_kg_s, 0.0));
    REQUIRE(failed.runtime.boundary_layer.last_update_time_s == before_last);
    REQUIRE(failed.runtime.boundary_layer.next_update_time_s == before_next);

    failed.fields.qc[1][1][5] = 2.0e-4f;
    failed.fields.rho[5][5][4] = 0.0f;
    REQUIRE_THROWS_AS(
        prepare_boundary_layer_tendencies_for_application(failed_state),
        std::invalid_argument);
    REQUIRE(fields_match(before_dtheta, failed.tendencies.dtheta_dt_pbl, 0.0));
    REQUIRE(fields_match(before_dqv, failed.tendencies.dqv_dt_pbl, 0.0));
    REQUIRE(fields_match(before_dqc, failed.tendencies.dqc_dt_pbl, 0.0));
    REQUIRE(fields_match(before_dqi, failed.tendencies.dqi_dt_pbl, 0.0));
    REQUIRE(fields_match(before_du, failed.tendencies.du_dt_pbl, 0.0));
    REQUIRE(fields_match(before_dv, failed.tendencies.dv_dt_pbl, 0.0));
}

TEST_CASE("boundary-layer refresh reuses transactional candidate storage",
          "[physics][boundary_layer][allocation][transactional]")
{
    initialize_test_grid();
    resize_global_fields(5.0f, 300.0f);
    const BoundaryLayerConfig cfg = runtime_test_config();
    const SurfaceConfig       sfc = runtime_test_surface();
    reset_boundary_layer_runtime(cfg, sfc);

    auto& runtime = tmv::global_boundary_layer_runtime_state;
    REQUIRE(runtime.refresh_scratch.dtheta.empty());
    REQUIRE(runtime.refresh_scratch.u_force.empty());
    REQUIRE(runtime.refresh_scratch.ustar.empty());
    REQUIRE(runtime.refresh_scratch.skin_theta.empty());
    REQUIRE(runtime.refresh_scratch.slab_columns.empty());

    step_boundary_layer(0.0);
    const float* const           first_dtheta_buffer  = dtheta_dt_pbl.data();
    const float* const           first_u_force_buffer = runtime.held_u_force_n.data();
    const float* const           first_surface_buffer = sfc_ustar.data();
    const float* const           first_skin_buffer    = runtime.skin_theta_2d.data();
    const SlabColumnState* const first_slab_buffer    = runtime.slab_columns.data();
    REQUIRE(runtime.worker_contexts.size() == 1);
    const BoundaryLayerSchemeBase* const first_worker_scheme   = runtime.worker_contexts.front().scheme.get();
    const auto&                          first_worker          = runtime.worker_contexts.front();
    const double* const                  first_geometry_buffer = first_worker.workspace.z_interface.data();
    const double* const                  first_du_buffer       = first_worker.tendencies.dudt_pbl.data();
    REQUIRE(first_dtheta_buffer != nullptr);
    REQUIRE(first_u_force_buffer != nullptr);
    REQUIRE(first_surface_buffer != nullptr);
    REQUIRE(first_skin_buffer != nullptr);
    REQUIRE(first_slab_buffer != nullptr);
    REQUIRE(first_worker_scheme != nullptr);
    REQUIRE(first_geometry_buffer != nullptr);
    REQUIRE(first_du_buffer != nullptr);

    // Two successful publications cycle each committed allocation back through
    // scratch. The third sample must therefore return to the first buffers.
    step_boundary_layer(1.0);
    step_boundary_layer(2.0);
    REQUIRE(dtheta_dt_pbl.data() == first_dtheta_buffer);
    REQUIRE(runtime.held_u_force_n.data() == first_u_force_buffer);
    REQUIRE(sfc_ustar.data() == first_surface_buffer);
    REQUIRE(runtime.skin_theta_2d.data() == first_skin_buffer);
    REQUIRE(runtime.slab_columns.data() == first_slab_buffer);
    REQUIRE(runtime.worker_contexts.size() == 1);
    REQUIRE(runtime.worker_contexts.front().scheme.get() == first_worker_scheme);
    REQUIRE(runtime.worker_contexts.front().workspace.z_interface.data() == first_geometry_buffer);
    REQUIRE(runtime.worker_contexts.front().tendencies.dudt_pbl.data() == first_du_buffer);

    const BoundaryLayerRuntimeState snapshot = runtime;
    REQUIRE(snapshot.refresh_scratch.dtheta.empty());
    REQUIRE(snapshot.refresh_scratch.u_force.empty());
    REQUIRE(snapshot.refresh_scratch.ustar.empty());
    REQUIRE(snapshot.refresh_scratch.skin_theta.empty());
    REQUIRE(snapshot.refresh_scratch.slab_columns.empty());
    REQUIRE(snapshot.worker_contexts.empty());
    REQUIRE(fields_match(snapshot.held_u_force_n, runtime.held_u_force_n, 0.0));
    REQUIRE(snapshot.skin_theta_2d == runtime.skin_theta_2d);
    REQUIRE(snapshot.slab_columns.size() == runtime.slab_columns.size());
}

TEST_CASE("runtime boundary-layer step samples explicit simulation state", "[physics][boundary_layer][state]")
{
    initialize_test_grid();

    Field3D            local_p(NR, NTH, NZ, 0.0f);
    Field3D            local_theta(NR, NTH, NZ, 0.0f);
    Field3D            local_rho(NR, NTH, NZ, 0.0f);
    Field3D            local_qv(NR, NTH, NZ, 0.0f);
    Field3D            local_u(NR, NTH, NZ, 0.0f);
    Field3D            local_w(NR, NTH, NZ, 0.0f);
    Field3D            local_v(NR, NTH, NZ, 0.0f);
    Field3D            local_tracer(NR, NTH, NZ, 0.0f);
    Field3D            local_qc(NR, NTH, NZ, 0.0f);
    Field3D            local_qr(NR, NTH, NZ, 0.0f);
    Field3D            local_qi(NR, NTH, NZ, 0.0f);
    Field3D            local_qs(NR, NTH, NZ, 0.0f);
    Field3D            local_qh(NR, NTH, NZ, 0.0f);
    Field3D            local_qg(NR, NTH, NZ, 0.0f);
    Field3D            local_tke(NR, NTH, NZ, 0.0f);
    Field3D            local_dtheta_dt_rad(NR, NTH, NZ, 0.0f);
    Field3D            local_dtheta_dt_pbl(NR, NTH, NZ, 0.0f);
    Field3D            local_dqv_dt_pbl(NR, NTH, NZ, 0.0f);
    Field3D            local_du_dt_pbl(NR, NTH, NZ, 0.0f);
    Field3D            local_dv_dt_pbl(NR, NTH, NZ, 0.0f);
    Field3D            local_dtke_dt_pbl(NR, NTH, NZ, 0.0f);
    std::vector<float> local_sfc_ustar;
    std::vector<float> local_sfc_tau_u;
    std::vector<float> local_sfc_tau_v;
    std::vector<float> local_sfc_sensible_heat_flux;
    std::vector<float> local_sfc_latent_heat_flux;
    std::vector<float> local_sfc_moisture_flux;
    std::vector<float> local_pbl_height_2d;

    for (int i = 0; i < NR; ++i)
    {
        for (int j = 0; j < NTH; ++j)
        {
            for (int k = 0; k < NZ; ++k)
            {
                local_p[i][j][k]     = static_cast<float>(100000.0 * std::exp(-0.02 * k));
                local_theta[i][j][k] = static_cast<float>(300.0 + 0.5 * k);
                local_rho[i][j][k]   = static_cast<float>(1.15 - 0.03 * k);
                local_qv[i][j][k]    = 0.010f;
                local_u[i][j][k]     = 6.0f;
                local_v[i][j][k]     = 0.0f;
            }
        }
    }

    std::vector<double>       local_rho0;
    std::vector<double>       local_p0;
    std::vector<double>       local_qv0;
    std::vector<double>       local_theta0;
    std::vector<double>       local_u0;
    std::vector<double>       local_v0;
    double                    local_coriolis = 0.0;
    double                    local_time     = 0.0;
    BoundaryLayerRuntimeState local_boundary_layer_runtime;

    SimulationStateView local_state{{NR, NTH, NZ, dr, dz, dt, dtheta, global_grid_geometry},
                                    {local_rho,
                                     local_p,
                                     local_u,
                                     local_w,
                                     local_v,
                                     local_tracer,
                                     local_theta,
                                     local_qv,
                                     local_qc,
                                     local_qr,
                                     local_qi,
                                     local_qs,
                                     local_qh,
                                     local_qg,
                                     local_tke},
                                    {local_rho0, local_p0, local_qv0, local_theta0, local_u0, local_v0},
                                    {local_dtheta_dt_rad,
                                     local_dtheta_dt_pbl,
                                     local_dqv_dt_pbl,
                                     local_du_dt_pbl,
                                     local_dv_dt_pbl,
                                     local_dtke_dt_pbl},
                                    local_coriolis,
                                    local_time,
                                    {&local_sfc_ustar,
                                     &local_sfc_tau_u,
                                     &local_sfc_tau_v,
                                     &local_sfc_sensible_heat_flux,
                                     &local_sfc_latent_heat_flux,
                                     &local_sfc_moisture_flux,
                                     &local_pbl_height_2d},
                                    nullptr,
                                    nullptr,
                                    &local_boundary_layer_runtime};

    const BoundaryLayerConfig cfg           = runtime_test_config();
    const SurfaceConfig       sfc           = runtime_test_surface();
    const double              sample_time_s = 1.0;

    resize_global_fields(0.0f, 280.0f);
    reset_boundary_layer_runtime(cfg, sfc);
    step_boundary_layer(sample_time_s);
    const Field3D            bad_global_output  = du_dt_pbl;
    const std::vector<float> bad_global_surface = sfc_tau_u;

    resize_global_fields(0.0f, 280.0f);
#ifdef _OPENMP
    const int previous_dynamic = omp_get_dynamic();
    const int previous_threads = omp_get_max_threads();
    omp_set_dynamic(0);
    omp_set_num_threads(1);
#endif
    initialize_boundary_layer(local_boundary_layer_runtime, "slab", cfg, sfc);
    step_boundary_layer(local_state, sample_time_s);
    const Field3D            explicit_output  = local_du_dt_pbl;
    const Field3D            serial_dtheta    = local_dtheta_dt_pbl;
    const Field3D            serial_dqv       = local_dqv_dt_pbl;
    const Field3D            serial_dv        = local_dv_dt_pbl;
    const Field3D            serial_dtke      = local_dtke_dt_pbl;
    const std::vector<float> explicit_surface = local_sfc_tau_u;

#ifdef _OPENMP
    const std::vector<float>           serial_ustar    = local_sfc_ustar;
    const std::vector<float>           serial_tau_v    = local_sfc_tau_v;
    const std::vector<float>           serial_sensible = local_sfc_sensible_heat_flux;
    const std::vector<float>           serial_latent   = local_sfc_latent_heat_flux;
    const std::vector<float>           serial_moisture = local_sfc_moisture_flux;
    const std::vector<float>           serial_height   = local_pbl_height_2d;
    const std::vector<SlabColumnState> serial_slab     = local_boundary_layer_runtime.slab_columns;

    initialize_boundary_layer(local_boundary_layer_runtime, "slab", cfg, sfc);
    omp_set_num_threads(4);
    step_boundary_layer(local_state, sample_time_s);
    const bool threaded_fields_match =
        fields_match(explicit_output, local_du_dt_pbl, 0.0) && fields_match(serial_dtheta, local_dtheta_dt_pbl, 0.0) &&
        fields_match(serial_dqv, local_dqv_dt_pbl, 0.0) && fields_match(serial_dv, local_dv_dt_pbl, 0.0) &&
        fields_match(serial_dtke, local_dtke_dt_pbl, 0.0);
    const bool threaded_surface_matches = vectors_match(explicit_surface, local_sfc_tau_u, 0.0) &&
                                          vectors_match(serial_ustar, local_sfc_ustar, 0.0) &&
                                          vectors_match(serial_tau_v, local_sfc_tau_v, 0.0) &&
                                          vectors_match(serial_sensible, local_sfc_sensible_heat_flux, 0.0) &&
                                          vectors_match(serial_latent, local_sfc_latent_heat_flux, 0.0) &&
                                          vectors_match(serial_moisture, local_sfc_moisture_flux, 0.0) &&
                                          vectors_match(serial_height, local_pbl_height_2d, 0.0);
    bool       threaded_slab_matches    = serial_slab.size() == local_boundary_layer_runtime.slab_columns.size();
    for (std::size_t index = 0; threaded_slab_matches && index < serial_slab.size(); ++index)
    {
        const SlabColumnState& lhs = serial_slab[index];
        const SlabColumnState& rhs = local_boundary_layer_runtime.slab_columns[index];
        threaded_slab_matches      = lhs.h_m == rhs.h_m && lhs.theta_m_k == rhs.theta_m_k &&
                                     lhs.qv_m_kgkg == rhs.qv_m_kgkg && lhs.initialized == rhs.initialized;
    }

    bool stateless_schemes_match = true;
    for (const std::string scheme_name : {"ysu", "mynn"})
    {
        BoundaryLayerConfig scheme_cfg = cfg;
        scheme_cfg.scheme_id           = scheme_name;
        scheme_cfg.enable_tke          = scheme_name == "mynn";
        omp_set_num_threads(1);
        initialize_boundary_layer(local_boundary_layer_runtime, scheme_name, scheme_cfg, sfc);
        step_boundary_layer(local_state, sample_time_s);
        const Field3D scheme_du     = local_du_dt_pbl;
        const Field3D scheme_dv     = local_dv_dt_pbl;
        const Field3D scheme_dtheta = local_dtheta_dt_pbl;
        const Field3D scheme_dqv    = local_dqv_dt_pbl;
        const Field3D scheme_dtke   = local_dtke_dt_pbl;
        if (scheme_name == "mynn")
            REQUIRE(any_nonzero(scheme_dtke));
        const std::vector<float> scheme_ustar    = local_sfc_ustar;
        const std::vector<float> scheme_tau_u    = local_sfc_tau_u;
        const std::vector<float> scheme_tau_v    = local_sfc_tau_v;
        const std::vector<float> scheme_sensible = local_sfc_sensible_heat_flux;
        const std::vector<float> scheme_latent   = local_sfc_latent_heat_flux;
        const std::vector<float> scheme_moisture = local_sfc_moisture_flux;
        const std::vector<float> scheme_height   = local_pbl_height_2d;

        omp_set_num_threads(4);
        initialize_boundary_layer(local_boundary_layer_runtime, scheme_name, scheme_cfg, sfc);
        step_boundary_layer(local_state, sample_time_s);
        stateless_schemes_match =
            stateless_schemes_match && fields_match(scheme_du, local_du_dt_pbl, 0.0) &&
            fields_match(scheme_dv, local_dv_dt_pbl, 0.0) && fields_match(scheme_dtheta, local_dtheta_dt_pbl, 0.0) &&
            fields_match(scheme_dqv, local_dqv_dt_pbl, 0.0) && fields_match(scheme_dtke, local_dtke_dt_pbl, 0.0) &&
            vectors_match(scheme_ustar, local_sfc_ustar, 0.0) && vectors_match(scheme_tau_u, local_sfc_tau_u, 0.0) &&
            vectors_match(scheme_tau_v, local_sfc_tau_v, 0.0) &&
            vectors_match(scheme_sensible, local_sfc_sensible_heat_flux, 0.0) &&
            vectors_match(scheme_latent, local_sfc_latent_heat_flux, 0.0) &&
            vectors_match(scheme_moisture, local_sfc_moisture_flux, 0.0) &&
            vectors_match(scheme_height, local_pbl_height_2d, 0.0);
    }
    omp_set_num_threads(previous_threads);
    omp_set_dynamic(previous_dynamic);
    REQUIRE(threaded_fields_match);
    REQUIRE(threaded_surface_matches);
    REQUIRE(threaded_slab_matches);
    REQUIRE(stateless_schemes_match);

    // Preserve the slab-runtime state expected by the cadence assertions below.
    initialize_boundary_layer(local_boundary_layer_runtime, "slab", cfg, sfc);
    step_boundary_layer(local_state, sample_time_s);
#endif

    const std::vector<float> global_surface_after_explicit = sfc_tau_u;

    copy_local_to_global(local_p, local_theta, local_rho, local_qv, local_u, local_v);
    reset_boundary_layer_runtime(cfg, sfc);
    step_boundary_layer(sample_time_s);
    const Field3D            legacy_local_output  = du_dt_pbl;
    const std::vector<float> legacy_local_surface = sfc_tau_u;

    REQUIRE(any_nonzero(explicit_output));
    REQUIRE(fields_match(explicit_output, legacy_local_output, 1.0e-8));
    REQUIRE_FALSE(fields_match(explicit_output, bad_global_output, 1.0e-8));
    REQUIRE(any_nonzero(explicit_surface));
    REQUIRE(vectors_match(explicit_surface, legacy_local_surface, 1.0e-8));
    REQUIRE_FALSE(vectors_match(explicit_surface, bad_global_surface, 1.0e-8));
    REQUIRE_FALSE(any_nonzero(global_surface_after_explicit));

    BoundaryLayerConfig slow_global_cfg = cfg;
    slow_global_cfg.dt_pbl              = 100.0;
    reset_boundary_layer_runtime(slow_global_cfg, SurfaceConfig{});
    step_boundary_layer(2.0);

    local_du_dt_pbl.fill(0.0f);
    step_boundary_layer(local_state, 2.0);
    REQUIRE(boundary_layer_updated_this_step(local_state));
    REQUIRE(any_nonzero(local_du_dt_pbl));

    // A skipped global call clears only the global update marker. The local
    // cadence and its held-skin/slab memory remain independently owned.
    step_boundary_layer(3.0);
    REQUIRE_FALSE(boundary_layer_updated_this_step());
    REQUIRE(boundary_layer_updated_this_step(local_state));

    step_boundary_layer(local_state, 2.5);
    REQUIRE_FALSE(boundary_layer_updated_this_step(local_state));

    BoundaryLayerConfig cadence_cfg = cfg;
    cadence_cfg.dt_pbl              = 10.0;
    initialize_boundary_layer(local_boundary_layer_runtime, "slab", cadence_cfg, sfc);
    step_boundary_layer(local_state, 0.0);
    REQUIRE(local_boundary_layer_runtime.next_update_time_s == Approx(10.0));
    step_boundary_layer(local_state, 7.0);
    REQUIRE_FALSE(boundary_layer_updated_this_step(local_state));
    step_boundary_layer(local_state, 14.0);
    REQUIRE(boundary_layer_updated_this_step(local_state));
    REQUIRE(local_boundary_layer_runtime.last_update_time_s == Approx(14.0));
    REQUIRE(local_boundary_layer_runtime.next_update_time_s == Approx(20.0));
    step_boundary_layer(local_state, 21.0);
    REQUIRE(boundary_layer_updated_this_step(local_state));
    REQUIRE(local_boundary_layer_runtime.next_update_time_s == Approx(30.0));

    BoundaryLayerConfig none_cfg = cfg;
    none_cfg.scheme_id           = "none";
    initialize_boundary_layer(local_boundary_layer_runtime, "none", none_cfg, SurfaceConfig{});
    local_du_dt_pbl.fill(1.0f);
    std::fill(local_sfc_tau_u.begin(), local_sfc_tau_u.end(), 1.0f);
    step_boundary_layer(local_state, 4.0);
    REQUIRE(local_boundary_layer_runtime.slab_columns.empty());
    REQUIRE(local_boundary_layer_runtime.held_u_force_n.empty());
    REQUIRE(local_boundary_layer_runtime.held_v_force_n.empty());
    REQUIRE(local_boundary_layer_runtime.held_theta_inventory_rate_kgk_s.empty());
    REQUIRE(local_boundary_layer_runtime.held_vapor_mass_rate_kg_s.empty());
    REQUIRE(local_boundary_layer_runtime.refresh_scratch.dtheta.empty());
    REQUIRE(local_boundary_layer_runtime.refresh_scratch.u_force.empty());
    REQUIRE(local_boundary_layer_runtime.refresh_scratch.ustar.empty());
    REQUIRE(local_boundary_layer_runtime.worker_contexts.empty());
    REQUIRE_FALSE(any_nonzero(local_du_dt_pbl));
    REQUIRE_FALSE(any_nonzero(local_sfc_tau_u));
}

TEST_CASE("portable boundary-layer checkpoint round-trips slab continuation",
          "[physics][boundary_layer][checkpoint][cpu003]")
{
    BoundaryLayerRuntimeState original;
    BoundaryLayerConfig       cfg;
    cfg.scheme_id            = "slab";
    cfg.dt_pbl               = 37.5;
    cfg.apply_surface_fluxes = true;
    cfg.surface_layer_id     = "bulk";
    cfg.clear_sky_only       = true;
    cfg.enable_tke           = false;
    cfg.pbl_top_method       = "ri";
    cfg.pbl_max_height       = 2345.25;
    cfg.min_ustar            = 2.5e-4;
    SurfaceConfig surface;
    surface.z0m        = 0.35;
    surface.z0h        = 0.025;
    surface.albedo     = 0.17;
    surface.emissivity = 0.93;
    surface.Tsfc       = 302.75;
    surface.qsfc       = 0.01425;
    initialize_boundary_layer(original, "slab", cfg, surface);

    original.last_update_time_s = 123.75;
    original.next_update_time_s = 150.0;
    original.updated_this_step  = true;
    original.skin_nr            = 2;
    original.skin_nth           = 2;
    original.skin_theta_2d      = {300.25f, 301.5f, 302.75f, 304.0f};
    original.skin_qv_2d         = {0.010f, 0.011f, 0.012f, 0.013f};
    original.skin_captured      = true;
    original.slab_nr            = 2;
    original.slab_nth           = 2;
    original.slab_columns.resize(4);
    for (std::size_t index = 0; index < original.slab_columns.size(); ++index)
    {
        SlabColumnState& column = original.slab_columns[index];
        column.h_m              = 500.0 + 25.0 * static_cast<double>(index);
        column.theta_m_k        = 298.0 + 0.5 * static_cast<double>(index);
        column.qv_m_kgkg        = 0.009 + 0.001 * static_cast<double>(index);
        column.initialized      = index != 1;
    }
    original.held_u_force_n.resize(2, 2, 1, -3.25f);
    original.held_v_force_n.resize(2, 2, 1, 1.5f);
    original.held_theta_inventory_rate_kgk_s.resize(2, 2, 1, 7.75f);
    original.held_vapor_mass_rate_kg_s.resize(2, 2, 1, 0.0125f);
    original.held_cloud_liquid_mass_rate_kg_s.resize(2, 2, 1, -0.003f);
    original.held_cloud_ice_mass_rate_kg_s.resize(2, 2, 1, 0.004f);
    original.last_momentum_work                           = {-17.0, 0.25, -0.002, -16.752};
    original.cumulative_requested_momentum_work_j         = -170.0;
    original.cumulative_finite_step_momentum_correction_j = 2.5;
    original.cumulative_momentum_storage_rounding_j       = -0.02;
    original.cumulative_committed_momentum_work_j         = -167.52;
    original.momentum_work_step_count                     = 10;

    std::vector<std::uint8_t> first;
    std::vector<std::uint8_t> second;
    std::string               error;
    REQUIRE(tmv::checkpoint::serialize_boundary_layer_runtime(original, first, error));
    REQUIRE(error.empty());
    REQUIRE(tmv::checkpoint::serialize_boundary_layer_runtime(original, second, error));
    REQUIRE(first == second);

    BoundaryLayerRuntimeState restored;
    REQUIRE(tmv::checkpoint::deserialize_boundary_layer_runtime(first, restored, error));
    REQUIRE(restored.scheme != nullptr);
    REQUIRE(restored.scheme->name() == "slab");
    REQUIRE(restored.config.scheme_id == original.config.scheme_id);
    REQUIRE(restored.config.dt_pbl == original.config.dt_pbl);
    REQUIRE(restored.surface_config.Tsfc == original.surface_config.Tsfc);
    REQUIRE(restored.last_update_time_s == original.last_update_time_s);
    REQUIRE(restored.next_update_time_s == original.next_update_time_s);
    REQUIRE(restored.updated_this_step == original.updated_this_step);
    REQUIRE(restored.skin_theta_2d == original.skin_theta_2d);
    REQUIRE(restored.skin_qv_2d == original.skin_qv_2d);
    REQUIRE(restored.skin_captured == original.skin_captured);
    REQUIRE(restored.skin_nr == original.skin_nr);
    REQUIRE(restored.skin_nth == original.skin_nth);
    REQUIRE(restored.slab_nr == original.slab_nr);
    REQUIRE(restored.slab_nth == original.slab_nth);
    REQUIRE(restored.slab_columns.size() == original.slab_columns.size());
    REQUIRE(fields_match(restored.held_u_force_n, original.held_u_force_n, 0.0));
    REQUIRE(fields_match(restored.held_v_force_n, original.held_v_force_n, 0.0));
    REQUIRE(fields_match(restored.held_theta_inventory_rate_kgk_s, original.held_theta_inventory_rate_kgk_s, 0.0));
    REQUIRE(fields_match(restored.held_vapor_mass_rate_kg_s, original.held_vapor_mass_rate_kg_s, 0.0));
    REQUIRE(fields_match(restored.held_cloud_liquid_mass_rate_kg_s,
                         original.held_cloud_liquid_mass_rate_kg_s, 0.0));
    REQUIRE(fields_match(restored.held_cloud_ice_mass_rate_kg_s,
                         original.held_cloud_ice_mass_rate_kg_s, 0.0));
    REQUIRE(restored.last_momentum_work.requested_linear_work_j == original.last_momentum_work.requested_linear_work_j);
    REQUIRE(restored.last_momentum_work.finite_step_correction_j ==
            original.last_momentum_work.finite_step_correction_j);
    REQUIRE(restored.last_momentum_work.storage_rounding_j == original.last_momentum_work.storage_rounding_j);
    REQUIRE(restored.last_momentum_work.committed_work_j == original.last_momentum_work.committed_work_j);
    REQUIRE(restored.cumulative_requested_momentum_work_j == original.cumulative_requested_momentum_work_j);
    REQUIRE(restored.cumulative_finite_step_momentum_correction_j ==
            original.cumulative_finite_step_momentum_correction_j);
    REQUIRE(restored.cumulative_momentum_storage_rounding_j == original.cumulative_momentum_storage_rounding_j);
    REQUIRE(restored.cumulative_committed_momentum_work_j == original.cumulative_committed_momentum_work_j);
    REQUIRE(restored.momentum_work_step_count == original.momentum_work_step_count);
    for (std::size_t index = 0; index < original.slab_columns.size(); ++index)
    {
        REQUIRE(restored.slab_columns[index].h_m == original.slab_columns[index].h_m);
        REQUIRE(restored.slab_columns[index].theta_m_k == original.slab_columns[index].theta_m_k);
        REQUIRE(restored.slab_columns[index].qv_m_kgkg == original.slab_columns[index].qv_m_kgkg);
        REQUIRE(restored.slab_columns[index].initialized == original.slab_columns[index].initialized);
    }

    const auto checkpoint_path = std::filesystem::temp_directory_path() / "nimbus_boundary_layer_checkpoint_v2.bin";
    std::filesystem::remove(checkpoint_path);
    REQUIRE(tmv::checkpoint::write_boundary_layer_checkpoint(checkpoint_path, original, error));
    BoundaryLayerRuntimeState file_restored;
    REQUIRE(tmv::checkpoint::read_boundary_layer_checkpoint(checkpoint_path, file_restored, error));
    REQUIRE(file_restored.slab_columns[3].theta_m_k == original.slab_columns[3].theta_m_k);
    std::filesystem::remove(checkpoint_path);
}

TEST_CASE("held PBL physical sources close and survive density evolution",
          "[physics][boundary_layer][runtime][budget][cadence]")
{
    initialize_test_grid();
    resize_global_fields(/*u=*/6.0f, /*theta=*/300.0f);
    BoundaryLayerConfig cfg = runtime_test_config();
    cfg.dt_pbl              = 60.0;
    const SurfaceConfig sfc = runtime_test_surface();
    reset_boundary_layer_runtime(cfg, sfc);
    step_boundary_layer(0.0);

    const double area_m2            = dr * dtheta;
    double       surface_u_force_n  = 0.0;
    double       surface_theta_rate = 0.0;
    double       surface_vapor_rate = 0.0;
    double       held_u_force_n     = 0.0;
    double       held_theta_rate    = 0.0;
    double       held_vapor_rate    = 0.0;
    for (int i = 0; i < NR; ++i)
    {
        for (int j = 0; j < NTH; ++j)
        {
            const std::size_t surface_index =
                static_cast<std::size_t>(i) * static_cast<std::size_t>(NTH) + static_cast<std::size_t>(j);
            const double exner =
                std::pow(static_cast<double>(p[i][j][0]) / physical_constants::reference_pressure_pa,
                         physical_constants::gas_constant_dry_air_jkgk / physical_constants::specific_heat_cp_jkgk);
            surface_u_force_n += sfc_tau_u[surface_index] * area_m2;
            surface_theta_rate +=
                sfc_sensible_heat_flux[surface_index] * area_m2 / (physical_constants::specific_heat_cp_jkgk * exner);
            surface_vapor_rate += sfc_moisture_flux[surface_index] * area_m2;
            for (int k = 0; k < NZ; ++k)
            {
                held_u_force_n += tmv::global_boundary_layer_runtime_state.held_u_force_n[i][j][k];
                held_theta_rate += tmv::global_boundary_layer_runtime_state.held_theta_inventory_rate_kgk_s[i][j][k];
                held_vapor_rate += tmv::global_boundary_layer_runtime_state.held_vapor_mass_rate_kg_s[i][j][k];
            }
        }
    }
    REQUIRE(held_u_force_n == Approx(surface_u_force_n).epsilon(2.0e-7));
    REQUIRE(held_theta_rate == Approx(surface_theta_rate).epsilon(2.0e-7));
    REQUIRE(held_vapor_rate == Approx(surface_vapor_rate).epsilon(2.0e-7));

    const float tendency_before = du_dt_pbl[0][0][0];
    rho.fill(0.55f);
    prepare_boundary_layer_tendencies_for_application(global_simulation_state());
    REQUIRE(du_dt_pbl[0][0][0] == Approx(2.0 * tendency_before).epsilon(2.0e-7));

    double       reconstructed_u_force_n  = 0.0;
    double       reconstructed_theta_rate = 0.0;
    double       reconstructed_vapor_rate = 0.0;
    const double volume_m3                = dr * dtheta * dz;
    for (int i = 0; i < NR; ++i)
        for (int j = 0; j < NTH; ++j)
            for (int k = 0; k < NZ; ++k)
            {
                const double mass = rho[i][j][k] * volume_m3;
                reconstructed_u_force_n += mass * du_dt_pbl[i][j][k];
                reconstructed_theta_rate += mass * dtheta_dt_pbl[i][j][k];
                reconstructed_vapor_rate += mass * dqv_dt_pbl[i][j][k];
            }
    REQUIRE(reconstructed_u_force_n == Approx(held_u_force_n).epsilon(2.0e-7));
    REQUIRE(reconstructed_theta_rate == Approx(held_theta_rate).epsilon(2.0e-7));
    REQUIRE(reconstructed_vapor_rate == Approx(held_vapor_rate).epsilon(2.0e-7));
}

TEST_CASE("portable boundary-layer checkpoint rejects corruption transactionally",
          "[physics][boundary_layer][checkpoint][cpu003]")
{
    BoundaryLayerRuntimeState original;
    BoundaryLayerConfig       cfg;
    cfg.scheme_id = "slab";
    initialize_boundary_layer(original, "slab", cfg, SurfaceConfig{});
    original.skin_nr       = 1;
    original.skin_nth      = 1;
    original.skin_theta_2d = {301.0f};
    original.skin_qv_2d    = {0.012f};
    original.slab_nr       = 1;
    original.slab_nth      = 1;
    original.slab_columns  = {{800.0, 300.0, 0.01, true}};

    std::vector<std::uint8_t> bytes;
    std::string               error;
    REQUIRE(tmv::checkpoint::serialize_boundary_layer_runtime(original, bytes, error));

    BoundaryLayerRuntimeState destination;
    destination.last_update_time_s = 999.0;
    destination.slab_columns       = {{42.0, 0.0, 0.0, false}};
    bytes.back() ^= 0x80U;
    REQUIRE_FALSE(tmv::checkpoint::deserialize_boundary_layer_runtime(bytes, destination, error));
    REQUIRE(error.find("checksum") != std::string::npos);
    REQUIRE(destination.last_update_time_s == 999.0);
    REQUIRE(destination.slab_columns[0].h_m == 42.0);

    bytes.resize(12);
    REQUIRE_FALSE(tmv::checkpoint::deserialize_boundary_layer_runtime(bytes, destination, error));
    REQUIRE(destination.last_update_time_s == 999.0);

    original.held_u_force_n.resize(1, 1, 1, 0.0f);
    REQUIRE_FALSE(tmv::checkpoint::serialize_boundary_layer_runtime(original, bytes, error));
    REQUIRE(error.find("held-source") != std::string::npos);

    original.held_u_force_n.release_storage();
    original.config.scheme_id  = "mynn";
    original.config.enable_tke = false;
    REQUIRE_FALSE(tmv::checkpoint::serialize_boundary_layer_runtime(original, bytes, error));
    REQUIRE(error.find("MYNN requires") != std::string::npos);

    original.config.scheme_id    = "slab";
    original.surface_config.Tsfc = std::numeric_limits<double>::quiet_NaN();
    REQUIRE_FALSE(tmv::checkpoint::serialize_boundary_layer_runtime(original, bytes, error));
    REQUIRE(error.find("surface Tsfc") != std::string::npos);
}

TEST_CASE("portable checkpoint sections are deterministic and fail closed", "[core][runtime][checkpoint][format]")
{
    using tmv::checkpoint::wire::Section;
    const std::vector<Section> sections{{101, 1, tmv::checkpoint::wire::section_required, {1, 2, 3}},
                                        {205, 2, 0, {4, 5}}};
    std::vector<std::uint8_t>  first;
    std::vector<std::uint8_t>  second;
    std::string                error;
    REQUIRE(tmv::checkpoint::wire::encode_sections(sections, first, error));
    REQUIRE(tmv::checkpoint::wire::encode_sections(sections, second, error));
    REQUIRE(first == second);

    std::vector<Section> restored;
    REQUIRE(tmv::checkpoint::wire::decode_sections(first, restored, error));
    REQUIRE(restored.size() == 2);
    REQUIRE(restored[0].id == 101);
    REQUIRE(restored[0].version == 1);
    REQUIRE(restored[0].payload == std::vector<std::uint8_t>{1, 2, 3});
    REQUIRE(restored[1].flags == 0);

    std::vector<Section> duplicate = sections;
    duplicate.push_back(sections.front());
    REQUIRE_FALSE(tmv::checkpoint::wire::encode_sections(duplicate, second, error));
    REQUIRE(error.find("duplicate") != std::string::npos);

    std::vector<std::uint8_t> corrupt = first;
    corrupt.back() ^= 0x80U;
    REQUIRE_FALSE(tmv::checkpoint::wire::decode_sections(corrupt, restored, error));
    REQUIRE(error.find("checksum") != std::string::npos);

    std::vector<std::uint8_t> trailing = first;
    trailing.push_back(0);
    REQUIRE_FALSE(tmv::checkpoint::wire::decode_sections(trailing, restored, error));
    REQUIRE(error.find("trailing") != std::string::npos);
}

TEST_CASE("PBL terrain columns publish mapped conserved sources and work",
          "[physics][boundary_layer][terrain][coupling][work]")
{
    constexpr int nx = 4, ny = 4, nz_local = 3;
    constexpr double dx = 1000.0, dy = 800.0, dz_local = 100.0;
    constexpr double slope_x = 0.10, slope_y = -0.05;
    for (const bool earth_fixed : {false, true})
    for (const StaggerType stagger :
         {StaggerType::CGrid, StaggerType::Collocated})
    {
    INFO("stagger=" << (stagger == StaggerType::CGrid
        ? "cgrid" : "collocated"));
    SimulationStorageState storage{{nx, ny, nz_local, dx, dz_local, 1.0, dy}};
    storage.grid.geometry.initialize(
        nx, ny, nz_local, dx, dz_local, dy,
        CoordinateSystem::Cartesian, stagger);
    auto& fields = storage.fields;
    fields.p.resize(nx, ny, nz_local, 0.0f);
    fields.theta.resize(nx, ny, nz_local, 300.0f);
    fields.qv.resize(nx, ny, nz_local, 0.01f);
    fields.rho.resize(nx, ny, nz_local, 1.0f);
    fields.u.resize(nx, ny, nz_local, 5.0f);
    fields.v.resize(nx, ny, nz_local, -2.0f);
    // The terrain map translates with the computational grid. Air whose
    // Earth-relative horizontal velocity matches that translation is at rest
    // relative to the surface and must not acquire spurious surface exchange.
    storage.configuration.domain_frame.velocity_ms = {5.0, -2.0};
    for (int i = 0; i < nx; ++i)
        for (int j = 0; j < ny; ++j)
            for (int k = 0; k < nz_local; ++k)
                fields.p[i][j][k] = 100000.0f - 5000.0f * k;

    auto& terrain            = storage.runtime.terrain;
    terrain.config.scheme_id = "synthetic_terrain";
    terrain.config.horizontal_placement_id = earth_fixed ? "earth_fixed" : "grid_attached";
    auto& metrics = terrain.metrics;
    metrics.dx = dx;
    metrics.dy = dy;
    metrics.dzeta = dz_local;
    metrics.z.resize(nx, ny, nz_local, 0.0);
    metrics.J.resize(nx, ny, nz_local, 1.0);
    metrics.mx.resize(nx, ny, nz_local, -slope_x);
    metrics.my.resize(nx, ny, nz_local, -slope_y);
    metrics.zeta.resize(nx, ny, nz_local, 0.0);
    metrics.z_face.resize(nx, ny, nz_local + 1, 0.0);
    metrics.Jx_face.resize(nx + 1, ny, nz_local, 1.0);
    metrics.Jy_face.resize(nx, ny + 1, nz_local, 1.0);
    metrics.Gx_face.resize(nx, ny, nz_local + 1, -slope_x);
    metrics.Gy_face.resize(nx, ny, nz_local + 1, -slope_y);
    for (int i = 0; i < nx; ++i)
        for (int j = 0; j < ny; ++j)
        {
            const double ground = 500.0 + slope_x * i * dx
                + slope_y * j * dy;
            for (int k = 0; k <= nz_local; ++k)
                metrics.z_face(i, j, k) = ground + k * dz_local;
            for (int k = 0; k < nz_local; ++k)
            {
                metrics.z(i, j, k) = ground + (k + 0.5) * dz_local;
                metrics.zeta(i, j, k) = (k + 0.5) * dz_local;
            }
        }

    auto& pbl              = storage.runtime.boundary_layer;
    pbl.scheme             = std::make_unique<GeometryProbeBoundaryLayerScheme>();
    pbl.config.scheme_id   = "geometry_probe";
    pbl.config.dt_pbl      = 60.0;
    pbl.last_update_time_s = std::numeric_limits<double>::quiet_NaN();
    pbl.next_update_time_s = 0.0;
    pbl.surface_config.z0m = 0.1;
    pbl.surface_config.z0h = 0.01;

    auto         state = make_simulation_state_view(storage);
    const PhysicalCellDomain domain = physical_scalar_domain(state.grid);
    const double area  = storage.grid.dx * storage.grid.dy;
    for (int i = 1; i < nx - 1; ++i)
        for (int j = 1; j < ny - 1; ++j)
    {
        const auto column = tmv::physical_grid_metrics::column_geometry(
            state, i, j, domain.k_begin, domain.k_end, true);
        REQUIRE(column.z_interface.front() == 0.0);
        for (int k = domain.k_begin; k < domain.k_end; ++k)
        {
            REQUIRE(column.layer_thickness[
                        static_cast<std::size_t>(k - domain.k_begin)]
                    == Approx(dz_local));
            REQUIRE(tmv::physical_grid_metrics::cell_volume(state, i, j, k)
                    == Approx(area * dz_local));
        }
    }

    REQUIRE_NOTHROW(step_boundary_layer(state, 0.0));
    REQUIRE(pbl.updated_this_step);
    REQUIRE(pbl.next_update_time_s == Approx(60.0));
    for (int i = 1; i < nx - 1; ++i)
        for (int j = 1; j < ny - 1; ++j)
            for (int k = domain.k_begin; k < domain.k_end; ++k)
            {
                REQUIRE(storage.tendencies.dtheta_dt_pbl(i, j, k)
                        == Approx(dz_local + (earth_fixed ? 3.0 : 0.0)));
                const float expected_u = stagger == StaggerType::Collocated
                    || i == 1
                    ? static_cast<float>(slope_x) : 0.0f;
                const float expected_v = stagger == StaggerType::Collocated
                    || j == 1
                    ? static_cast<float>(slope_y) : 0.0f;
                REQUIRE(storage.tendencies.du_dt_pbl(i, j, k)
                        == Approx(expected_u));
                REQUIRE(storage.tendencies.dv_dt_pbl(i, j, k)
                        == Approx(expected_v));
            }
    REQUIRE_NOTHROW(prepare_boundary_layer_tendencies_for_application(state));
    BoundaryLayerMomentumWorkReport report;
    REQUIRE_NOTHROW(commit_boundary_layer_momentum_work(state, report));
    REQUIRE(pbl.momentum_work_step_count == 1);
    }
}
