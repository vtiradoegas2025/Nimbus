// Regression tests for explicit-state radiation stepping and ownership.

#include "catch2/catch.hpp"
#include "core/runtime/simulation.hpp"
#include "core/runtime/simulation_state.hpp"
#include "core/runtime/simulation_runtime.hpp"
#include "core/infra/physical_constants.hpp"
#include "core/infra/physical_grid_metrics.hpp"
#include "compute/backend_runtime.hpp"
#include "radiation/factory.hpp"
#include "radiation/native_batch_runtime.hpp"
#include "radiation/radiation_base.hpp"
#include "util/log_profile.hpp"

#include <cmath>
#include <atomic>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

SimulationStorageState          global_simulation_storage{{3, 2, 4, 1000.0, 500.0, 1.0, 1000.0}};
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
tmv::SurfaceConfig&            tmv::global_surface_config = tmv::global_boundary_layer_runtime_state.surface_config;
tmv::chaos::ChaosRuntimeState& tmv::chaos::global_chaos_runtime_state = global_simulation_storage.runtime.chaos;
RayleighDampingRuntimeState&   global_rayleigh_runtime_state          = global_simulation_storage.runtime.rayleigh;
tmv::MicrophysicsRuntimeState& tmv::global_microphysics_runtime_state = global_simulation_storage.runtime.microphysics;
tmv::AdvectionRuntimeState&    tmv::global_advection_runtime_state    = global_simulation_storage.runtime.advection;
tmv::DiffusionRuntimeState&    tmv::global_diffusion_runtime_state    = global_simulation_storage.runtime.diffusion;
tmv::TimeSteppingRuntimeState& tmv::global_time_stepping_runtime_state =
    global_simulation_storage.runtime.time_stepping;
Field3D& rho           = global_simulation_storage.fields.rho;
Field3D& p             = global_simulation_storage.fields.p;
Field3D& u             = global_simulation_storage.fields.u;
Field3D& w             = global_simulation_storage.fields.w;
Field3D& v             = global_simulation_storage.fields.v;
Field3D& tracer        = global_simulation_storage.fields.tracer;
Field3D& theta         = global_simulation_storage.fields.theta;
Field3D& qv            = global_simulation_storage.fields.qv;
Field3D& qc            = global_simulation_storage.fields.qc;
Field3D& qr            = global_simulation_storage.fields.qr;
Field3D& qi            = global_simulation_storage.fields.qi;
Field3D& qs            = global_simulation_storage.fields.qs;
Field3D& qh            = global_simulation_storage.fields.qh;
Field3D& qg            = global_simulation_storage.fields.qg;
Field3D& tke           = global_simulation_storage.fields.tke;
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

struct BatchProbeStats
{
    std::atomic<int> native_calls{0};
    std::atomic<int> cpu_calls{0};
    bool             reject_native = false;
};

struct ComputeRuntimeReset
{
    ~ComputeRuntimeReset()
    {
        global_simulation_storage.runtime.compute.reset();
    }
};

void fill_probe_tendency(const RadiationColumnStateView& column, RadiationColumnTendencies& tendency)
{
    constexpr double  heating_rate = 1.0e-5;
    const std::size_t layers       = column.rho->size();
    tendency.dTdt_rad.assign(layers, heating_rate);
    tendency.dTdt_lw.assign(layers, heating_rate);
    tendency.dTdt_sw.assign(layers, 0.0);
    double power = 0.0;
    for (std::size_t layer = 0; layer < layers; ++layer)
        power += (*column.rho)[layer] * physical_constants::specific_heat_cp_jkgk * (*column.dz)[layer] * heating_rate;
    tendency.atmospheric_power_lw_wm2    = power;
    tendency.atmospheric_power_sw_wm2    = 0.0;
    tendency.atmospheric_power_total_wm2 = power;
}

class BatchProbeKernels final : public RadiationComputeKernels
{
public:
    bool supports(RadiationKernelOp op) const override
    {
        return op == RadiationKernelOp::RrtmgLongwaveTransferBatch ||
               op == RadiationKernelOp::RrtmgShortwaveTransferBatch;
    }
};

class BatchProbeBackend final : public ComputeBackend
{
public:
    std::string name() const override
    {
        return "batch_probe";
    }
    bool initialize(std::string& error) override
    {
        error.clear();
        return true;
    }
    void shutdown() override
    {
    }
    RadiationComputeKernels* radiation_kernels() override
    {
        return &kernels_;
    }
    const RadiationComputeKernels* radiation_kernels() const override
    {
        return &kernels_;
    }

private:
    BatchProbeKernels kernels_;
};

class BatchProbeRadiationScheme final : public RadiationSchemeBase
{
public:
    explicit BatchProbeRadiationScheme(std::shared_ptr<BatchProbeStats> stats) : stats_(std::move(stats))
    {
    }
    std::unique_ptr<RadiationSchemeBase> clone() const override
    {
        return std::make_unique<BatchProbeRadiationScheme>(*this);
    }
    std::string name() const override
    {
        return "rrtmg";
    }
    int required_fields() const override
    {
        return 0;
    }
    void initialize(const RadiationConfig&) override
    {
    }
    void compute_column(const RadiationConfig&,
                        const RadiationColumnStateView& column,
                        RadiationColumnTendencies&      tendency,
                        RadiationColumnFluxes*) override
    {
        ++stats_->cpu_calls;
        fill_probe_tendency(column, tendency);
    }
    bool supports_native_column_batch(const RadiationConfig&         config,
                                      const RadiationComputeKernels& kernels) const override
    {
        return (!config.do_lw || kernels.supports(RadiationKernelOp::RrtmgLongwaveTransferBatch)) &&
               (!config.do_sw || kernels.supports(RadiationKernelOp::RrtmgShortwaveTransferBatch));
    }
    bool compute_columns_native(const RadiationConfig&,
                                const RadiationColumnStateView* columns,
                                std::size_t                     count,
                                RadiationColumnTendencies*      tendencies,
                                RadiationComputeKernels&) override
    {
        ++stats_->native_calls;
        if (stats_->reject_native)
            return false;
        std::vector<RadiationColumnTendencies> candidates(count);
        for (std::size_t column = 0; column < count; ++column)
            fill_probe_tendency(columns[column], candidates[column]);
        for (std::size_t column = 0; column < count; ++column)
            tendencies[column] = std::move(candidates[column]);
        return true;
    }

private:
    std::shared_ptr<BatchProbeStats> stats_;
};

class ThrowingRadiationScheme final : public RadiationSchemeBase
{
public:
    std::unique_ptr<RadiationSchemeBase> clone() const override
    {
        return std::make_unique<ThrowingRadiationScheme>(*this);
    }
    std::string name() const override
    {
        return "throwing_test_scheme";
    }
    int required_fields() const override
    {
        return 0;
    }
    void initialize(const RadiationConfig&) override
    {
    }
    void compute_column(const RadiationConfig&,
                        const RadiationColumnStateView&,
                        RadiationColumnTendencies&,
                        RadiationColumnFluxes*) override
    {
        throw std::runtime_error("injected column failure");
    }
};

class GeometryProbeRadiationScheme final : public RadiationSchemeBase
{
public:
    std::unique_ptr<RadiationSchemeBase> clone() const override
    {
        return std::make_unique<GeometryProbeRadiationScheme>(*this);
    }
    std::string name() const override
    {
        return "simple_grey";
    }
    int required_fields() const override
    {
        return 0;
    }
    void initialize(const RadiationConfig&) override
    {
    }
    void compute_column(const RadiationConfig&,
                        const RadiationColumnStateView& col,
                        RadiationColumnTendencies&      tend,
                        RadiationColumnFluxes*) override
    {
        if (col.rho == nullptr || col.dz == nullptr || col.z_int == nullptr)
            throw std::invalid_argument("geometry probe received a null column");
        const std::size_t nz_col = col.rho->size();
        if (col.dz->size() != nz_col || col.z_int->size() != nz_col + 1)
            throw std::invalid_argument("geometry probe received malformed geometry");
        constexpr double heating_rate = 1.0e-5;
        tend.dTdt_rad.assign(nz_col, heating_rate);
        tend.dTdt_lw.assign(nz_col, heating_rate);
        tend.dTdt_sw.assign(nz_col, 0.0);
        double power = 0.0;
        for (std::size_t k = 0; k < nz_col; ++k)
            power += (*col.rho)[k] * physical_constants::specific_heat_cp_jkgk * (*col.dz)[k] * heating_rate;
        tend.atmospheric_power_lw_wm2    = power;
        tend.atmospheric_power_total_wm2 = power;
    }
};

void initialize_test_grid()
{
    NR     = 3;
    NTH    = 2;
    NZ     = 4;
    dr     = 1000.0;
    dz     = 500.0;
    dt     = 1.0;
    dtheta = 1000.0;
    global_grid_geometry.initialize(NR, NTH, NZ, dr, dz, dtheta, CoordinateSystem::Cartesian);
}

void resize_global_fields(float p_value, float theta_value, float rho_value)
{
    p.resize(NR, NTH, NZ, 0.0f);
    theta.resize(NR, NTH, NZ, theta_value);
    rho.resize(NR, NTH, NZ, rho_value);
    qv.resize(NR, NTH, NZ, 0.0f);
    u.resize(NR, NTH, NZ, 0.0f);
    w.resize(NR, NTH, NZ, 0.0f);
    v.resize(NR, NTH, NZ, 0.0f);
    tracer.resize(NR, NTH, NZ, 0.0f);
    qc.resize(NR, NTH, NZ, 0.0f);
    qr.resize(NR, NTH, NZ, 0.0f);
    qi.resize(NR, NTH, NZ, 0.0f);
    qs.resize(NR, NTH, NZ, 0.0f);
    qh.resize(NR, NTH, NZ, 0.0f);
    qg.resize(NR, NTH, NZ, 0.0f);
    for (int i = 0; i < NR; ++i)
        for (int j = 0; j < NTH; ++j)
            for (int k = 0; k < NZ; ++k)
                p[i][j][k] = static_cast<float>(static_cast<double>(p_value) * std::exp(-0.05 * k));
}

void copy_local_to_global(const Field3D& local_p,
                          const Field3D& local_theta,
                          const Field3D& local_rho,
                          const Field3D& local_qv)
{
    p     = local_p;
    theta = local_theta;
    rho   = local_rho;
    qv    = local_qv;
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

RadiationConfig runtime_test_config()
{
    RadiationConfig cfg;
    cfg.scheme_id    = "simple_grey";
    cfg.do_lw        = true;
    cfg.do_sw        = true;
    cfg.dt_radiation = 1.0;
    cfg.tau_lw_ref   = 3.0;
    cfg.tau_sw_ref   = 0.1;
    cfg.n_lw         = 1.0;
    cfg.n_sw         = 1.0;
    return cfg;
}

void reset_radiation_runtime(const RadiationConfig& cfg)
{
    initialize_radiation("simple_grey", cfg);
    dtheta_dt_rad.resize(NR, NTH, NZ, 0.0f);
}

std::shared_ptr<BatchProbeStats> prepare_batch_probe_run(int columns_x, bool allow_fallback, bool reject_native)
{
    NR     = columns_x;
    NTH    = 2;
    NZ     = 4;
    dr     = 1000.0;
    dz     = 500.0;
    dt     = 1.0;
    dtheta = 1000.0;
    global_grid_geometry.initialize(NR, NTH, NZ, dr, dz, dtheta, CoordinateSystem::Cartesian);
    resize_global_fields(95000.0f, 300.0f, 1.0f);
    global_surface_config.albedo     = 0.2;
    global_surface_config.emissivity = 0.95;
    global_surface_config.Tsfc       = 288.0;

    RadiationConfig config       = runtime_test_config();
    config.scheme_id             = "rrtmg";
    config.do_sw                 = false;
    auto stats                   = std::make_shared<BatchProbeStats>();
    stats->reject_native         = reject_native;
    auto& radiation              = global_simulation_storage.runtime.radiation;
    radiation.scheme             = std::make_unique<BatchProbeRadiationScheme>(stats);
    radiation.config             = config;
    radiation.last_update_time_s = -config.dt_radiation;
    radiation.next_update_time_s = 0.0;
    radiation.batch_workspace.reset();
    dtheta_dt_rad.resize(NR, NTH, NZ, 0.0f);

    auto& compute = global_simulation_storage.runtime.compute;
    compute.reset();
    compute.backend               = std::make_unique<BatchProbeBackend>();
    compute.active_kind           = ComputeBackendKind::Metal;
    compute.requested_kind        = ComputeBackendKind::Metal;
    compute.fallback_active       = false;
    compute.config.backend        = "metal";
    compute.config.allow_fallback = allow_fallback;
    return stats;
}

} // namespace

TEST_CASE("runtime RRTMG batches are bounded, strict, and transactionally fallback-safe",
          "[physics][radiation][rrtmg][batch][backend]")
{
    const ComputeRuntimeReset reset_compute_runtime;

    SECTION("bounded native chunks publish the complete update")
    {
        const auto stats = prepare_batch_probe_run(130, false, false);
        REQUIRE_NOTHROW(step_radiation(0.0));
        const int physical_columns = (NR - 2) * NTH;
        REQUIRE(stats->native_calls.load() == (physical_columns + 63) / 64);
        REQUIRE(stats->cpu_calls.load() == 0);
        REQUIRE(any_nonzero(dtheta_dt_rad));
        REQUIRE(global_radiation_runtime_state.batch_workspace != nullptr);
    }

    SECTION("strict rejection preserves the previously published field")
    {
        const auto stats = prepare_batch_probe_run(3, false, true);
        dtheta_dt_rad.fill(77.0f);
        REQUIRE_THROWS_WITH(step_radiation(0.0), "native radiation batch dispatch was rejected");
        REQUIRE(stats->native_calls.load() == 1);
        REQUIRE(stats->cpu_calls.load() == 0);
        for (std::size_t index = 0; index < dtheta_dt_rad.size(); ++index)
            REQUIRE(dtheta_dt_rad.data()[index] == 77.0f);
    }

    SECTION("explicit fallback recomputes every column on CPU")
    {
        const auto stats = prepare_batch_probe_run(3, true, true);
        REQUIRE_NOTHROW(step_radiation(0.0));
        REQUIRE(stats->native_calls.load() == 1);
        REQUIRE(stats->cpu_calls.load() == (NR - 2) * NTH);
        REQUIRE(any_nonzero(dtheta_dt_rad));
    }
}

TEST_CASE("native RRTMG runtime batching is bitwise invariant across OpenMP workers",
          "[physics][radiation][rrtmg][batch][parallel][determinism]")
{
    const ComputeRuntimeReset reset_compute_runtime;

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

    const auto serial_stats = prepare_batch_probe_run(130, false, false);
    REQUIRE_NOTHROW(step_radiation(0.0));
    const Field3D serial_tendency   = dtheta_dt_rad;
    const Field3D serial_cell_power = global_radiation_runtime_state.held_cell_power_w;
    const double  serial_power      = global_radiation_runtime_state.held_atmospheric_power_w;
    const int     serial_dispatches = serial_stats->native_calls.load();

#ifdef _OPENMP
    omp_set_num_threads(4);
#endif
    const auto parallel_stats = prepare_batch_probe_run(130, false, false);
    REQUIRE_NOTHROW(step_radiation(0.0));
    REQUIRE(parallel_stats->native_calls.load() == serial_dispatches);
    REQUIRE(fields_match(dtheta_dt_rad, serial_tendency, 0.0));
    REQUIRE(fields_match(global_radiation_runtime_state.held_cell_power_w, serial_cell_power, 0.0));
    REQUIRE(global_radiation_runtime_state.held_atmospheric_power_w == serial_power);
}

TEST_CASE("runtime radiation initialization rejects unknown schemes", "[physics][radiation][configuration]")
{
    const RadiationConfig cfg = runtime_test_config();
    REQUIRE_THROWS_AS(initialize_radiation("nonexistent_scheme_xyz", cfg), std::runtime_error);
}

TEST_CASE("runtime radiation initialization rejects malformed active physics",
          "[physics][radiation][configuration][guard]")
{
    RadiationConfig cfg = runtime_test_config();
    cfg.tau_lw_ref      = -1.0;
    REQUIRE_THROWS_AS(initialize_radiation("simple_grey", cfg), std::invalid_argument);

    cfg                = runtime_test_config();
    cfg.clear_sky_only = false;
    REQUIRE_THROWS_AS(initialize_radiation("simple_grey", cfg), std::invalid_argument);

    cfg              = runtime_test_config();
    cfg.latitude_deg = 100.0;
    REQUIRE_THROWS_AS(initialize_radiation("simple_grey", cfg), std::invalid_argument);

    cfg                = runtime_test_config();
    cfg.use_subcycling = true;
    REQUIRE_THROWS_AS(initialize_radiation("simple_grey", cfg), std::invalid_argument);

    cfg                    = runtime_test_config();
    cfg.enable_diagnostics = true;
    REQUIRE_NOTHROW(initialize_radiation("simple_grey", cfg));

    cfg = runtime_test_config();
    cfg.simple_grey_optical_model = "unsupported_model";
    REQUIRE_THROWS_AS(initialize_radiation("simple_grey", cfg), std::invalid_argument);
}

TEST_CASE("absorbing water-cloud simple-grey runtime retains diagnostics and subcycles",
          "[physics][radiation][runtime][diagnostics][cadence][sci352]")
{
    initialize_test_grid();
    resize_global_fields(95000.0f, 300.0f, 1.0f);
    for (int i = 0; i < NR; ++i)
        for (int j = 0; j < NTH; ++j)
            for (int k = 0; k < NZ; ++k)
            {
                qv[i][j][k] = 0.01f;
                qc[i][j][k] = k == 1 ? 1.0e-4f : 0.0f;
                qi[i][j][k] = k == 2 ? 8.0e-5f : 0.0f;
            }
    global_surface_config.albedo = 0.2;
    global_surface_config.emissivity = 0.95;
    global_surface_config.Tsfc = 290.0;

    RadiationConfig cfg = runtime_test_config();
    cfg.clear_sky_only = false;
    cfg.simple_grey_optical_model = "absorbing_water_cloud_path_v1";
    cfg.enable_diagnostics = true;
    cfg.use_subcycling = true;
    cfg.subcycle_count = 2;
    cfg.dt_radiation = 4.0;
    cfg.local_solar_time_at_start_h = 12.0;
    reset_radiation_runtime(cfg);
    REQUIRE(global_radiation_runtime_state.last_update_time_s == -2.0);
    REQUIRE_NOTHROW(step_radiation(0.0));
    REQUIRE(global_radiation_runtime_state.next_update_time_s == 2.0);
    REQUIRE_FALSE(global_radiation_runtime_state.held_flux_diagnostics.empty());
    REQUIRE(global_radiation_runtime_state.held_flux_diagnostics.columns_x == 1);
    REQUIRE(global_radiation_runtime_state.held_flux_diagnostics.columns_y == 2);
    REQUIRE(global_radiation_runtime_state.held_flux_diagnostics.levels == 2);
    REQUIRE(global_radiation_runtime_state.held_shortwave_incoming_power_w > 0.0);
    REQUIRE(global_radiation_runtime_state.held_shortwave_atmospheric_absorbed_power_w > 0.0);
    REQUIRE(global_radiation_runtime_state.held_surface_absorbed_power_w > 0.0);
    REQUIRE(global_radiation_runtime_state.held_top_escape_power_w > 0.0);
    REQUIRE(global_radiation_runtime_state.held_shortwave_incoming_power_w ==
            Approx(global_radiation_runtime_state.held_shortwave_atmospheric_absorbed_power_w +
                   global_radiation_runtime_state.held_surface_absorbed_power_w +
                   global_radiation_runtime_state.held_top_escape_power_w)
                .epsilon(2.0e-14));

    const Field3D held_before = dtheta_dt_rad;
    const auto diagnostics_before =
        global_radiation_runtime_state.held_flux_diagnostics;
    REQUIRE_NOTHROW(step_radiation(1.0));
    REQUIRE(fields_match(dtheta_dt_rad, held_before, 0.0));
    REQUIRE(global_radiation_runtime_state.held_flux_diagnostics
                .upward_longwave_wm2 ==
            diagnostics_before.upward_longwave_wm2);
    REQUIRE(global_radiation_runtime_state.last_update_time_s == 0.0);
    REQUIRE_NOTHROW(step_radiation(2.0));
    REQUIRE(global_radiation_runtime_state.last_update_time_s == 2.0);
    REQUIRE(global_radiation_runtime_state.next_update_time_s == 4.0);
}

TEST_CASE("absorbing water-cloud simple-grey publication is failure atomic",
          "[physics][radiation][runtime][transactional][sci352]")
{
    initialize_test_grid();
    resize_global_fields(95000.0f, 300.0f, 1.0f);
    qv.fill(0.01f);
    qc.fill(1.0e-4f);
    qi.fill(5.0e-5f);
    global_surface_config.albedo = 0.2;
    global_surface_config.emissivity = 0.95;
    global_surface_config.Tsfc = 290.0;
    RadiationConfig cfg = runtime_test_config();
    cfg.clear_sky_only = false;
    cfg.simple_grey_optical_model = "absorbing_water_cloud_path_v1";
    cfg.enable_diagnostics = true;
    cfg.local_solar_time_at_start_h = 12.0;
    reset_radiation_runtime(cfg);
    REQUIRE_NOTHROW(step_radiation(0.0));

    const Field3D tendency_before = dtheta_dt_rad;
    const Field3D power_field_before =
        global_radiation_runtime_state.held_cell_power_w;
    const tmv::RadiationFluxDiagnostics diagnostics_before =
        global_radiation_runtime_state.held_flux_diagnostics;
    const double power_before =
        global_radiation_runtime_state.held_atmospheric_power_w;
    const double incoming_before =
        global_radiation_runtime_state.held_shortwave_incoming_power_w;
    const double last_before =
        global_radiation_runtime_state.last_update_time_s;
    const double next_before =
        global_radiation_runtime_state.next_update_time_s;
    qc[1][0][1] = std::numeric_limits<float>::quiet_NaN();
    REQUIRE_THROWS_AS(step_radiation(1.0), std::runtime_error);
    REQUIRE(fields_match(dtheta_dt_rad, tendency_before, 0.0));
    REQUIRE(fields_match(global_radiation_runtime_state.held_cell_power_w,
                         power_field_before, 0.0));
    REQUIRE(global_radiation_runtime_state.held_atmospheric_power_w ==
            power_before);
    REQUIRE(global_radiation_runtime_state.held_shortwave_incoming_power_w ==
            incoming_before);
    REQUIRE(global_radiation_runtime_state.held_flux_diagnostics
                .upward_longwave_wm2 ==
            diagnostics_before.upward_longwave_wm2);
    REQUIRE(global_radiation_runtime_state.held_flux_diagnostics
                .downward_longwave_wm2 ==
            diagnostics_before.downward_longwave_wm2);
    REQUIRE(global_radiation_runtime_state.held_flux_diagnostics
                .upward_shortwave_wm2 ==
            diagnostics_before.upward_shortwave_wm2);
    REQUIRE(global_radiation_runtime_state.held_flux_diagnostics
                .downward_shortwave_wm2 ==
            diagnostics_before.downward_shortwave_wm2);
    REQUIRE(global_radiation_runtime_state.last_update_time_s == last_before);
    REQUIRE(global_radiation_runtime_state.next_update_time_s == next_before);
}

TEST_CASE("legacy radiation globals reference the owning runtime", "[physics][radiation][state][modularity]")
{
    REQUIRE(&radiation_scheme == &global_radiation_runtime_state.scheme);
    REQUIRE(&global_radiation_config == &global_radiation_runtime_state.config);

    const RadiationConfig cfg = runtime_test_config();
    initialize_radiation("simple_grey", cfg);

    REQUIRE(global_radiation_runtime_state.scheme != nullptr);
    REQUIRE(global_radiation_runtime_state.scheme->name() == "simple_grey");
    REQUIRE(global_radiation_runtime_state.config.scheme_id == "simple_grey");
    REQUIRE(global_radiation_runtime_state.last_update_time_s == Approx(-cfg.dt_radiation));
    REQUIRE(global_radiation_runtime_state.next_update_time_s == Approx(0.0));
}

TEST_CASE("runtime radiation step samples explicit simulation state", "[physics][radiation][state]")
{
    initialize_test_grid();
    global_surface_config.albedo     = 0.2;
    global_surface_config.emissivity = 0.95;
    global_surface_config.Tsfc       = 288.0;

    Field3D local_p(NR, NTH, NZ, 0.0f);
    Field3D local_theta(NR, NTH, NZ, 0.0f);
    Field3D local_rho(NR, NTH, NZ, 0.0f);
    Field3D local_qv(NR, NTH, NZ, 0.0f);
    Field3D local_u(NR, NTH, NZ, 0.0f);
    Field3D local_w(NR, NTH, NZ, 0.0f);
    Field3D local_v(NR, NTH, NZ, 0.0f);
    Field3D local_tracer(NR, NTH, NZ, 0.0f);
    Field3D local_qc(NR, NTH, NZ, 0.0f);
    Field3D local_qr(NR, NTH, NZ, 0.0f);
    Field3D local_qi(NR, NTH, NZ, 0.0f);
    Field3D local_qs(NR, NTH, NZ, 0.0f);
    Field3D local_qh(NR, NTH, NZ, 0.0f);
    Field3D local_qg(NR, NTH, NZ, 0.0f);
    Field3D local_tke(NR, NTH, NZ, 0.0f);
    Field3D local_dtheta_dt_rad(NR, NTH, NZ, 0.0f);
    Field3D local_dtheta_dt_pbl(NR, NTH, NZ, 0.0f);
    Field3D local_dqv_dt_pbl(NR, NTH, NZ, 0.0f);
    Field3D local_du_dt_pbl(NR, NTH, NZ, 0.0f);
    Field3D local_dv_dt_pbl(NR, NTH, NZ, 0.0f);
    Field3D local_dtke_dt_pbl(NR, NTH, NZ, 0.0f);

    for (int i = 0; i < NR; ++i)
    {
        for (int j = 0; j < NTH; ++j)
        {
            for (int k = 0; k < NZ; ++k)
            {
                local_p[i][j][k]     = static_cast<float>(100000.0 * std::exp(-0.08 * k));
                local_theta[i][j][k] = static_cast<float>(300.0 + 2.0 * k);
                local_rho[i][j][k]   = static_cast<float>(1.1 - 0.05 * k);
                local_qv[i][j][k]    = 0.004f;
            }
        }
    }

    std::vector<double>   local_rho0;
    std::vector<double>   local_p0;
    std::vector<double>   local_qv0;
    std::vector<double>   local_theta0;
    std::vector<double>   local_u0;
    std::vector<double>   local_v0;
    double                local_coriolis = 0.0;
    double                local_time     = 0.0;
    RadiationRuntimeState local_radiation_runtime;

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
                                    {},
                                    &local_radiation_runtime};
    BoundaryLayerRuntimeState local_boundary_layer;
    local_boundary_layer.surface_config = global_simulation_storage.runtime.boundary_layer.surface_config;
    local_state.boundary_layer_runtime = &local_boundary_layer;


    const RadiationConfig cfg           = runtime_test_config();
    const double          sample_time_s = 43200.0;

    resize_global_fields(5000.0f, 900.0f, 5.0f);
    reset_radiation_runtime(cfg);
    step_radiation(sample_time_s);
    const Field3D bad_global_output = dtheta_dt_rad;

    resize_global_fields(5000.0f, 900.0f, 5.0f);
    initialize_radiation(local_radiation_runtime, "simple_grey", cfg);
    step_radiation(local_state, sample_time_s);
    const Field3D explicit_output  = local_dtheta_dt_rad;
    const double  explicit_power_w = local_radiation_runtime.held_atmospheric_power_w;

    copy_local_to_global(local_p, local_theta, local_rho, local_qv);
    reset_radiation_runtime(cfg);
    step_radiation(sample_time_s);
    const Field3D legacy_local_output = dtheta_dt_rad;

    REQUIRE(any_nonzero(explicit_output));
    REQUIRE(fields_match(explicit_output, legacy_local_output, 1.0e-8));
    REQUIRE(explicit_power_w != Approx(0.0));
    REQUIRE(global_radiation_runtime_state.held_atmospheric_power_w == Approx(explicit_power_w).epsilon(1.0e-12));
    REQUIRE_FALSE(fields_match(explicit_output, bad_global_output, 1.0e-8));
}

TEST_CASE("interleaved radiation states own scheme configuration and cadence", "[physics][radiation][state][sci020]")
{
    initialize_test_grid();
    global_surface_config.albedo     = 0.2;
    global_surface_config.emissivity = 0.95;
    global_surface_config.Tsfc       = 288.0;
    resize_global_fields(90000.0f, 300.0f, 1.0f);

    Field3D               local_p     = p;
    Field3D               local_theta = theta;
    Field3D               local_rho   = rho;
    Field3D               local_qv(NR, NTH, NZ, 0.004f);
    Field3D               local_u = u, local_w = w, local_v = v, local_tracer = tracer;
    Field3D               local_qc = qc, local_qr = qr, local_qi = qi, local_qs = qs;
    Field3D               local_qh = qh, local_qg = qg, local_tke(NR, NTH, NZ, 0.0f);
    Field3D               local_rad(NR, NTH, NZ, 0.0f);
    Field3D               local_pbl(NR, NTH, NZ, 0.0f), local_qv_pbl(NR, NTH, NZ, 0.0f);
    Field3D               local_u_pbl(NR, NTH, NZ, 0.0f), local_v_pbl(NR, NTH, NZ, 0.0f);
    Field3D               local_tke_pbl(NR, NTH, NZ, 0.0f);
    std::vector<double>   empty_base;
    double                local_coriolis = 0.0;
    double                local_time     = 0.0;
    RadiationRuntimeState local_runtime;
    SimulationStateView   local_state{{NR, NTH, NZ, dr, dz, dt, dtheta, global_grid_geometry},
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
                                      {empty_base, empty_base, empty_base, empty_base, empty_base, empty_base},
                                      {local_rad, local_pbl, local_qv_pbl, local_u_pbl, local_v_pbl, local_tke_pbl},
                                      local_coriolis,
                                      local_time,
                                      {},
                                      &local_runtime};
    BoundaryLayerRuntimeState local_boundary_layer;
    local_boundary_layer.surface_config = global_simulation_storage.runtime.boundary_layer.surface_config;
    local_state.boundary_layer_runtime = &local_boundary_layer;


    RadiationConfig local_cfg = runtime_test_config();
    local_cfg.dt_radiation    = 10.0;
    initialize_radiation(local_runtime, "simple_grey", local_cfg);

    RadiationConfig global_cfg = runtime_test_config();
    global_cfg.dt_radiation    = 100.0;
    initialize_radiation("none", global_cfg);
    dtheta_dt_rad.resize(NR, NTH, NZ, 0.0f);

    step_radiation(local_state, 0.0);
    REQUIRE(any_nonzero(local_rad));
    local_rad.fill(0.0f);

    step_radiation(0.0);
    REQUIRE_FALSE(any_nonzero(dtheta_dt_rad));

    // Reconfiguring and stepping the process-global runtime must neither
    // replace the local scheme/config nor consume its independent cadence.
    step_radiation(local_state, 10.0);
    REQUIRE(any_nonzero(local_rad));
    const Field3D held_tendency = local_rad;
    const double  held_power_w  = local_runtime.held_atmospheric_power_w;
    step_radiation(local_state, 15.0);
    REQUIRE(fields_match(local_rad, held_tendency, 0.0));
    REQUIRE(local_runtime.held_atmospheric_power_w == Approx(held_power_w));
    REQUIRE(local_runtime.last_update_time_s == Approx(10.0));
}

TEST_CASE("runtime radiation consumes valid upper-atmosphere theta and pressure",
          "[physics][radiation][state][regression]")
{
    initialize_test_grid();
    global_surface_config.albedo     = 0.2;
    global_surface_config.emissivity = 0.95;
    global_surface_config.Tsfc       = 288.0;

    RadiationConfig cfg = runtime_test_config();
    cfg.do_sw           = false;

    resize_global_fields(8000.0f, 300.0f, 0.12f);
    for (int i = 0; i < NR; ++i)
        for (int j = 0; j < NTH; ++j)
            for (int k = 0; k < NZ; ++k)
                p[i][j][k] = 8000.0f - 1000.0f * static_cast<float>(k);
    reset_radiation_runtime(cfg);
    step_radiation(43200.0);
    const Field3D cool_column = dtheta_dt_rad;

    theta.fill(340.0f);
    reset_radiation_runtime(cfg);
    step_radiation(43200.0);
    const Field3D warm_column = dtheta_dt_rad;

    REQUIRE(any_nonzero(cool_column));
    REQUIRE(any_nonzero(warm_column));
    REQUIRE_FALSE(fields_match(cool_column, warm_column, 1.0e-10));
}

TEST_CASE("radiation cadence follows the nominal clock without phase drift",
          "[physics][radiation][cadence][regression]")
{
    initialize_test_grid();
    resize_global_fields(90000.0f, 300.0f, 1.0f);
    global_surface_config.albedo     = 0.2;
    global_surface_config.emissivity = 0.95;
    global_surface_config.Tsfc       = 288.0;
    RadiationConfig cfg              = runtime_test_config();
    cfg.dt_radiation                 = 10.0;
    reset_radiation_runtime(cfg);

    step_radiation(0.0);
    REQUIRE(global_radiation_runtime_state.last_update_time_s == Approx(0.0));
    REQUIRE(global_radiation_runtime_state.next_update_time_s == Approx(10.0));

    // A 7 s dynamics step first crosses the t=10 due time at t=14. The next
    // due time remains the nominal t=20, not t=24 based on the late sample.
    step_radiation(14.0);
    REQUIRE(global_radiation_runtime_state.last_update_time_s == Approx(14.0));
    REQUIRE(global_radiation_runtime_state.next_update_time_s == Approx(20.0));
    step_radiation(21.0);
    REQUIRE(global_radiation_runtime_state.last_update_time_s == Approx(21.0));
    REQUIRE(global_radiation_runtime_state.next_update_time_s == Approx(30.0));
}

TEST_CASE("runtime radiation rejects malformed thermodynamic input", "[physics][radiation][state][guard]")
{
    initialize_test_grid();
    resize_global_fields(90000.0f, 300.0f, 1.0f);
    p[1][1][2] = -1.0f;
    reset_radiation_runtime(runtime_test_config());
    REQUIRE_THROWS_AS(step_radiation(43200.0), std::invalid_argument);
}

TEST_CASE("runtime radiation publication is transactional on column failure",
          "[physics][radiation][state][transactional]")
{
    initialize_test_grid();
    resize_global_fields(90000.0f, 300.0f, 1.0f);
    global_surface_config.albedo     = 0.2;
    global_surface_config.emissivity = 0.95;
    global_surface_config.Tsfc       = 288.0;
    reset_radiation_runtime(runtime_test_config());

    radiation_scheme = std::make_unique<ThrowingRadiationScheme>();
    dtheta_dt_rad.fill(7.0f);
    const Field3D published_before = dtheta_dt_rad;
    const double  timestamp_before = global_radiation_runtime_state.last_update_time_s;
    const double  power_before     = global_radiation_runtime_state.held_atmospheric_power_w;

    REQUIRE_THROWS_AS(step_radiation(43200.0), std::runtime_error);
    REQUIRE(fields_match(dtheta_dt_rad, published_before, 0.0));
    REQUIRE(global_radiation_runtime_state.last_update_time_s == timestamp_before);
    REQUIRE(global_radiation_runtime_state.held_atmospheric_power_w == power_before);
}

TEST_CASE("radiation reuses full-volume transactional publication buffers", "[physics][radiation][state][allocation]")
{
    initialize_test_grid();
    resize_global_fields(90000.0f, 300.0f, 1.0f);
    global_surface_config.albedo     = 0.2;
    global_surface_config.emissivity = 0.95;
    global_surface_config.Tsfc       = 288.0;

    RadiationConfig cfg = runtime_test_config();
    cfg.dt_radiation    = 1.0;
    reset_radiation_runtime(cfg);

    REQUIRE(global_radiation_runtime_state.scratch_theta_tendency.empty());
    REQUIRE(global_radiation_runtime_state.scratch_cell_power_w.empty());

    step_radiation(0.0);
    const float* const first_tendency_buffer = dtheta_dt_rad.data();
    const float* const first_power_buffer    = global_radiation_runtime_state.held_cell_power_w.data();
    REQUIRE(first_tendency_buffer != nullptr);
    REQUIRE(first_power_buffer != nullptr);
    REQUIRE(global_radiation_runtime_state.worker_contexts.size() == 1);
    const RadiationWorkerContext&    first_context         = global_radiation_runtime_state.worker_contexts.front();
    const RadiationSchemeBase* const first_worker_scheme   = first_context.scheme.get();
    const double* const              first_pressure_buffer = first_context.workspace.p.data();
    const double* const              first_interface_pressure_buffer = first_context.workspace.p_interface.data();
    const double* const              first_flux_buffer               = first_context.workspace.flux_net.data();
    const double* const              first_tendency_column_buffer    = first_context.tendencies.dTdt_rad.data();
    const double* const        first_column_power_buffer = global_radiation_runtime_state.scratch_column_power_w.data();
    const unsigned char* const first_column_error_buffer = global_radiation_runtime_state.scratch_column_errors.data();
    REQUIRE(first_worker_scheme != nullptr);
    REQUIRE(first_pressure_buffer != nullptr);
    REQUIRE(first_interface_pressure_buffer != nullptr);
    REQUIRE(first_flux_buffer != nullptr);
    REQUIRE(first_tendency_column_buffer != nullptr);
    REQUIRE(first_column_power_buffer != nullptr);
    REQUIRE(first_column_error_buffer != nullptr);

    // Transactional publication alternates the two owned buffers. By the
    // third sample both outputs must reuse their first-sample allocations.
    step_radiation(1.0);
    step_radiation(2.0);
    REQUIRE(dtheta_dt_rad.data() == first_tendency_buffer);
    REQUIRE(global_radiation_runtime_state.held_cell_power_w.data() == first_power_buffer);
    const RadiationWorkerContext& reused_context = global_radiation_runtime_state.worker_contexts.front();
    REQUIRE(reused_context.scheme.get() == first_worker_scheme);
    REQUIRE(reused_context.workspace.p.data() == first_pressure_buffer);
    REQUIRE(reused_context.workspace.p_interface.data() == first_interface_pressure_buffer);
    REQUIRE(reused_context.workspace.flux_net.data() == first_flux_buffer);
    REQUIRE(reused_context.tendencies.dTdt_rad.data() == first_tendency_column_buffer);
    REQUIRE(global_radiation_runtime_state.scratch_column_power_w.data() == first_column_power_buffer);
    REQUIRE(global_radiation_runtime_state.scratch_column_errors.data() == first_column_error_buffer);

    // Runtime copies/checkpoints contain committed state, not replaceable
    // scratch contents or their full-volume allocations.
    const RadiationRuntimeState snapshot = global_radiation_runtime_state;
    REQUIRE(snapshot.scratch_theta_tendency.empty());
    REQUIRE(snapshot.scratch_cell_power_w.empty());
    REQUIRE(snapshot.scratch_column_power_w.empty());
    REQUIRE(snapshot.scratch_column_errors.empty());
    REQUIRE(snapshot.worker_contexts.empty());
    REQUIRE(fields_match(snapshot.held_cell_power_w, global_radiation_runtime_state.held_cell_power_w, 0.0));
}

TEST_CASE("inactive radiation rejects invalid time transactionally", "[physics][radiation][state][transactional][none]")
{
    initialize_test_grid();
    resize_global_fields(90000.0f, 300.0f, 1.0f);
    RadiationConfig cfg = runtime_test_config();
    initialize_radiation("none", cfg);
    dtheta_dt_rad.fill(9.0f);
    const Field3D published_before = dtheta_dt_rad;
    const double  timestamp_before = global_radiation_runtime_state.last_update_time_s;

    REQUIRE_THROWS_AS(step_radiation(std::numeric_limits<double>::quiet_NaN()), std::invalid_argument);
    REQUIRE(fields_match(dtheta_dt_rad, published_before, 0.0));
    REQUIRE(global_radiation_runtime_state.last_update_time_s == timestamp_before);

    global_radiation_runtime_state.held_cell_power_w.resize(NR, NTH, NZ, 123.0f);
    global_radiation_runtime_state.held_atmospheric_power_w = 456.0;
    step_radiation(1.0);
    REQUIRE(global_radiation_runtime_state.held_cell_power_w.empty());
    REQUIRE(global_radiation_runtime_state.scratch_theta_tendency.empty());
    REQUIRE(global_radiation_runtime_state.scratch_cell_power_w.empty());
    REQUIRE(global_radiation_runtime_state.scratch_column_power_w.empty());
    REQUIRE(global_radiation_runtime_state.scratch_column_errors.empty());
    REQUIRE(global_radiation_runtime_state.worker_contexts.empty());
    REQUIRE(global_radiation_runtime_state.held_atmospheric_power_w == 0.0);
    REQUIRE_FALSE(any_nonzero(dtheta_dt_rad));
    REQUIRE(global_radiation_runtime_state.last_update_time_s == 1.0);
}

TEST_CASE("radiation held power is invariant to CPU thread count", "[physics][radiation][cpu][determinism]")
{
    NR     = 12;
    NTH    = 10;
    NZ     = 6;
    dr     = 700.0;
    dz     = 350.0;
    dt     = 1.0;
    dtheta = 900.0;
    global_grid_geometry.initialize(NR, NTH, NZ, dr, dz, dtheta, CoordinateSystem::Cartesian);
    resize_global_fields(95000.0f, 300.0f, 1.0f);
    for (int i = 0; i < NR; ++i)
        for (int j = 0; j < NTH; ++j)
            for (int k = 0; k < NZ; ++k)
            {
                theta[i][j][k] += static_cast<float>(0.17 * i + 0.03 * j + 0.11 * k);
                rho[i][j][k] += static_cast<float>(0.007 * i + 0.002 * j - 0.01 * k);
            }
    global_surface_config.albedo     = 0.23;
    global_surface_config.emissivity = 0.94;
    global_surface_config.Tsfc       = 291.0;
    const RadiationConfig cfg        = runtime_test_config();

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
    reset_radiation_runtime(cfg);
    step_radiation(43200.0);
    const double  serial_power    = global_radiation_runtime_state.held_atmospheric_power_w;
    const Field3D serial_tendency = dtheta_dt_rad;

#ifdef _OPENMP
    omp_set_num_threads(4);
#endif
    reset_radiation_runtime(cfg);
    step_radiation(43200.0);
    REQUIRE(global_radiation_runtime_state.held_atmospheric_power_w == serial_power);
    REQUIRE(fields_match(dtheta_dt_rad, serial_tendency, 0.0));
#ifdef _OPENMP
    REQUIRE(global_radiation_runtime_state.worker_contexts.size() == 4);
    for (std::size_t worker = 0; worker < global_radiation_runtime_state.worker_contexts.size(); ++worker)
    {
        const RadiationSchemeBase* const scheme = global_radiation_runtime_state.worker_contexts[worker].scheme.get();
        REQUIRE(scheme != nullptr);
        for (std::size_t previous = 0; previous < worker; ++previous)
            REQUIRE(scheme != global_radiation_runtime_state.worker_contexts[previous].scheme.get());
    }
#endif
}

TEST_CASE("absorbing water-cloud simple-grey is bitwise invariant across CPU workers",
          "[physics][radiation][cpu][determinism][diagnostics][sci352]")
{
    NR = 12;
    NTH = 10;
    NZ = 6;
    dr = 700.0;
    dz = 350.0;
    dt = 1.0;
    dtheta = 900.0;
    global_grid_geometry.initialize(
        NR, NTH, NZ, dr, dz, dtheta, CoordinateSystem::Cartesian);
    resize_global_fields(95000.0f, 300.0f, 1.0f);
    for (int i = 0; i < NR; ++i)
        for (int j = 0; j < NTH; ++j)
            for (int k = 0; k < NZ; ++k)
            {
                theta[i][j][k] += static_cast<float>(
                    0.17 * i + 0.03 * j + 0.11 * k);
                rho[i][j][k] += static_cast<float>(
                    0.004 * i + 0.002 * j - 0.01 * k);
                qv[i][j][k] = static_cast<float>(
                    0.012 * std::exp(-0.15 * k));
                qc[i][j][k] = k == 2
                    ? static_cast<float>(1.0e-4 * (1 + (i + j) % 3))
                    : 0.0f;
                qi[i][j][k] = k == 3
                    ? static_cast<float>(7.0e-5 * (1 + (2 * i + j) % 2))
                    : 0.0f;
            }
    global_surface_config.albedo = 0.23;
    global_surface_config.emissivity = 0.94;
    global_surface_config.Tsfc = 291.0;
    RadiationConfig cfg = runtime_test_config();
    cfg.clear_sky_only = false;
    cfg.simple_grey_optical_model = "absorbing_water_cloud_path_v1";
    cfg.enable_diagnostics = true;
    cfg.local_solar_time_at_start_h = 12.0;

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
    reset_radiation_runtime(cfg);
    step_radiation(0.0);
    const Field3D serial_tendency = dtheta_dt_rad;
    const Field3D serial_cell_power =
        global_radiation_runtime_state.held_cell_power_w;
    const double serial_power =
        global_radiation_runtime_state.held_atmospheric_power_w;
    const double serial_incoming =
        global_radiation_runtime_state.held_shortwave_incoming_power_w;
    const tmv::RadiationFluxDiagnostics serial_diagnostics =
        global_radiation_runtime_state.held_flux_diagnostics;

#ifdef _OPENMP
    omp_set_num_threads(4);
#endif
    reset_radiation_runtime(cfg);
    step_radiation(0.0);
    REQUIRE(fields_match(dtheta_dt_rad, serial_tendency, 0.0));
    REQUIRE(fields_match(global_radiation_runtime_state.held_cell_power_w,
                         serial_cell_power, 0.0));
    REQUIRE(global_radiation_runtime_state.held_atmospheric_power_w ==
            serial_power);
    REQUIRE(global_radiation_runtime_state.held_shortwave_incoming_power_w ==
            serial_incoming);
    REQUIRE(global_radiation_runtime_state.held_flux_diagnostics
                .upward_longwave_wm2 ==
            serial_diagnostics.upward_longwave_wm2);
    REQUIRE(global_radiation_runtime_state.held_flux_diagnostics
                .downward_longwave_wm2 ==
            serial_diagnostics.downward_longwave_wm2);
    REQUIRE(global_radiation_runtime_state.held_flux_diagnostics
                .upward_shortwave_wm2 ==
            serial_diagnostics.upward_shortwave_wm2);
    REQUIRE(global_radiation_runtime_state.held_flux_diagnostics
                .downward_shortwave_wm2 ==
            serial_diagnostics.downward_shortwave_wm2);
#ifdef _OPENMP
    REQUIRE(global_radiation_runtime_state.worker_contexts.size() == 4);
#endif
}

TEST_CASE("native RRTMG production columns are bitwise invariant across OpenMP workers",
          "[physics][radiation][rrtmg][cpu][parallel][determinism][external]")
{
    if (std::getenv("NIMBUS_RRTMG_LW_COEFFICIENTS") == nullptr ||
        std::getenv("NIMBUS_RRTMG_SW_COEFFICIENTS") == nullptr)
    {
        SUCCEED("official RRTMG coefficient paths were not supplied");
        return;
    }

    NR     = 12;
    NTH    = 10;
    NZ     = 8;
    dr     = 700.0;
    dz     = 500.0;
    dt     = 1.0;
    dtheta = 900.0;
    global_grid_geometry.initialize(NR, NTH, NZ, dr, dz, dtheta, CoordinateSystem::Cartesian);
    resize_global_fields(98000.0f, 300.0f, 1.05f);
    for (int i = 0; i < NR; ++i)
        for (int j = 0; j < NTH; ++j)
            for (int k = 0; k < NZ; ++k)
            {
                theta[i][j][k] += static_cast<float>(0.07 * i + 0.03 * j + 0.6 * k);
                rho[i][j][k] -= static_cast<float>(0.045 * k);
                qv[i][j][k] = static_cast<float>(0.009 * std::exp(-0.22 * k));
                if (k >= 2 && k <= 5)
                {
                    qc[i][j][k] = static_cast<float>(2.0e-5 * (1 + (i + j + k) % 3));
                    qi[i][j][k] = static_cast<float>(1.0e-5 * (1 + (2 * i + j + k) % 2));
                }
            }
    global_surface_config.albedo     = 0.21;
    global_surface_config.emissivity = 0.96;
    global_surface_config.Tsfc       = 292.0;

    RadiationConfig config;
    config.scheme_id       = "rrtmg";
    config.dt_radiation    = 300.0;
    config.rrtmg_ozone_vmr = 3.0e-6;

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
    for (const bool clear_sky_only : {true, false})
    {
        CAPTURE(clear_sky_only);
        config.clear_sky_only = clear_sky_only;
#ifdef _OPENMP
        omp_set_num_threads(1);
#endif
        initialize_radiation("rrtmg", config);
        step_radiation(43200.0);
        const Field3D serial_tendency   = dtheta_dt_rad;
        const double  serial_power      = global_radiation_runtime_state.held_atmospheric_power_w;
        double        serial_cell_power = 0.0;
        for (std::size_t index = 0; index < global_radiation_runtime_state.held_cell_power_w.size(); ++index)
            serial_cell_power += global_radiation_runtime_state.held_cell_power_w.data()[index];
        REQUIRE(any_nonzero(serial_tendency));
        REQUIRE(serial_cell_power == serial_power);

#ifdef _OPENMP
        omp_set_num_threads(4);
#endif
        initialize_radiation("rrtmg", config);
        step_radiation(43200.0);
        REQUIRE(fields_match(dtheta_dt_rad, serial_tendency, 0.0));
        REQUIRE(global_radiation_runtime_state.held_atmospheric_power_w == serial_power);
        double parallel_cell_power = 0.0;
        for (std::size_t index = 0; index < global_radiation_runtime_state.held_cell_power_w.size(); ++index)
            parallel_cell_power += global_radiation_runtime_state.held_cell_power_w.data()[index];
        REQUIRE(parallel_cell_power == serial_cell_power);
#ifdef _OPENMP
        REQUIRE(global_radiation_runtime_state.worker_contexts.size() == 4);
#endif
    }
}

TEST_CASE("runtime radiation excludes collocated closure storage from columns",
          "[physics][radiation][state][ownership]")
{
    initialize_test_grid();
    resize_global_fields(90000.0f, 300.0f, 1.0f);
    global_surface_config.albedo     = 0.2;
    global_surface_config.emissivity = 0.95;
    global_surface_config.Tsfc       = 288.0;

    // k=0 and k=NZ-1 are boundary-closure planes on the collocated grid.
    // Poisoning them must neither invalidate the atmospheric column nor
    // receive a radiative source tendency.
    for (int i = 0; i < NR; ++i)
        for (int j = 0; j < NTH; ++j)
        {
            p[i][j][0]          = -1.0f;
            theta[i][j][NZ - 1] = std::numeric_limits<float>::quiet_NaN();
        }

    reset_radiation_runtime(runtime_test_config());
    REQUIRE_NOTHROW(step_radiation(43200.0));
    REQUIRE(any_nonzero(dtheta_dt_rad));
    for (int i = 0; i < NR; ++i)
        for (int j = 0; j < NTH; ++j)
        {
            REQUIRE(dtheta_dt_rad[i][j][0] == 0.0f);
            REQUIRE(dtheta_dt_rad[i][j][NZ - 1] == 0.0f);
        }
}

TEST_CASE("runtime radiation rejects an implicit or invalid radiative surface", "[physics][radiation][surface][guard]")
{
    initialize_test_grid();
    resize_global_fields(90000.0f, 300.0f, 1.0f);
    global_surface_config.albedo     = 0.2;
    global_surface_config.emissivity = 0.95;
    global_surface_config.Tsfc       = -1.0;
    reset_radiation_runtime(runtime_test_config());
    REQUIRE_THROWS_AS(step_radiation(43200.0), std::invalid_argument);

    global_surface_config.Tsfc   = 300.0;
    global_surface_config.albedo = 1.1;
    reset_radiation_runtime(runtime_test_config());
    REQUIRE_THROWS_AS(step_radiation(43200.0), std::invalid_argument);
}

TEST_CASE("terrain longwave radiation uses exact physical columns and power",
          "[physics][radiation][terrain][longwave][sci115]")
{
    SimulationStorageState storage{{2, 1, 3, 1000.0, 100.0, 1.0, 800.0}};
    storage.grid.geometry.initialize(2, 1, 3, 1000.0, 100.0, 800.0, CoordinateSystem::Cartesian, StaggerType::CGrid);
    auto& fields = storage.fields;
    fields.p.resize(2, 1, 3, 0.0f);
    fields.theta.resize(2, 1, 3, 300.0f);
    fields.rho.resize(2, 1, 3, 1.0f);
    for (int i = 0; i < 2; ++i)
        for (int k = 0; k < 3; ++k)
            fields.p[i][0][k] = 100000.0f - 5000.0f * k;

    auto& terrain            = storage.runtime.terrain;
    terrain.config.scheme_id = "synthetic_terrain";
    terrain.metrics.z.resize(2, 1, 3, 0.0);
    terrain.metrics.z_face.resize(2, 1, 4, 0.0);
    terrain.metrics.Gx_face.resize(2, 1, 4, -0.2);
    terrain.metrics.Gy_face.resize(2, 1, 4, 0.0);
    const double faces[2][4] = {{100.0, 200.0, 350.0, 600.0}, {300.0, 450.0, 500.0, 600.0}};
    for (int i = 0; i < 2; ++i)
    {
        for (int k = 0; k <= 3; ++k)
            terrain.metrics.z_face(i, 0, k) = faces[i][k];
        for (int k = 0; k < 3; ++k)
            terrain.metrics.z(i, 0, k) = 0.5 * (faces[i][k] + faces[i][k + 1]);
    }

    auto& radiation                                          = storage.runtime.radiation;
    radiation.scheme                                         = std::make_unique<GeometryProbeRadiationScheme>();
    radiation.config.scheme_id                               = "simple_grey";
    radiation.config.dt_radiation                            = 60.0;
    radiation.config.do_lw                                   = true;
    radiation.config.do_sw                                   = false;
    radiation.last_update_time_s                             = -60.0;
    radiation.next_update_time_s                             = 0.0;
    storage.runtime.boundary_layer.surface_config.Tsfc       = 288.0;
    storage.runtime.boundary_layer.surface_config.albedo     = 0.2;
    storage.runtime.boundary_layer.surface_config.emissivity = 0.95;

    auto         state = make_simulation_state_view(storage);
    const double area  = storage.grid.dx * storage.grid.dy;
    for (int i = 0; i < 2; ++i)
    {
        const auto column = tmv::physical_grid_metrics::column_geometry(state, i, 0, 0, 3);
        for (int k = 0; k < 3; ++k)
        {
            const double thickness = faces[i][k + 1] - faces[i][k];
            REQUIRE(column.layer_thickness[static_cast<std::size_t>(k)] == Approx(thickness));
            REQUIRE(tmv::physical_grid_metrics::cell_volume(state, i, 0, k) == Approx(area * thickness));
        }
    }

    storage.tendencies.dtheta_dt_rad.resize(2, 1, 3, 23.0f);
    radiation.held_atmospheric_power_w = 42.0;
    REQUIRE_NOTHROW(step_radiation(state, 0.0));
    constexpr double heating_rate = 1.0e-5;
    constexpr double kappa = physical_constants::gas_constant_dry_air_jkgk / physical_constants::specific_heat_cp_jkgk;
    double           expected_power = 0.0;
    for (int i = 0; i < 2; ++i)
        for (int k = 0; k < 3; ++k)
        {
            const double theta_rate =
                heating_rate *
                std::pow(physical_constants::reference_pressure_pa / static_cast<double>(fields.p(i, 0, k)), kappa);
            REQUIRE(storage.tendencies.dtheta_dt_rad(i, 0, k) == Approx(theta_rate));
            const double cell_power =
                physical_constants::specific_heat_cp_jkgk * area * (faces[i][k + 1] - faces[i][k]) * heating_rate;
            REQUIRE(radiation.held_cell_power_w(i, 0, k) == Approx(cell_power));
            expected_power += cell_power;
        }
    REQUIRE(radiation.held_atmospheric_power_w == Approx(expected_power));
    REQUIRE(radiation.last_update_time_s == 0.0);
    REQUIRE(radiation.next_update_time_s == 60.0);

    REQUIRE_NOTHROW(commit_radiation_applied_heat(state, 1.0, expected_power));
    REQUIRE(radiation.applied_step_count == 1);
    REQUIRE(radiation.last_expected_atmospheric_energy_j == Approx(expected_power));
    REQUIRE(radiation.last_applied_radiation_heat_j == Approx(expected_power));

    const auto require_ledger_unchanged = [&]
    {
        REQUIRE(radiation.applied_step_count == 1);
        REQUIRE(radiation.last_expected_atmospheric_energy_j
                == Approx(expected_power));
        REQUIRE(radiation.last_applied_radiation_heat_j
                == Approx(expected_power));
        REQUIRE(radiation.accumulated_expected_atmospheric_energy_j
                == Approx(expected_power));
        REQUIRE(radiation.accumulated_applied_radiation_heat_j
                == Approx(expected_power));
    };
    radiation.held_shortwave_incoming_power_w = -1.0;
    REQUIRE_THROWS_AS(
        commit_radiation_applied_heat(state, 1.0, expected_power),
        std::invalid_argument);
    require_ledger_unchanged();
    radiation.held_shortwave_incoming_power_w = 0.0;
    radiation.held_shortwave_atmospheric_absorbed_power_w =
        std::numeric_limits<double>::max();
    radiation.held_surface_absorbed_power_w =
        std::numeric_limits<double>::max();
    REQUIRE_THROWS_AS(
        commit_radiation_applied_heat(state, 1.0, expected_power),
        std::overflow_error);
    require_ledger_unchanged();
    radiation.held_shortwave_atmospheric_absorbed_power_w = 0.0;
    radiation.held_surface_absorbed_power_w = 0.0;

    radiation.config.do_sw                       = true;
    radiation.config.local_solar_time_at_start_h = 12.0;
    radiation.last_update_time_s                 = -60.0;
    radiation.next_update_time_s                 = 0.0;
    storage.tendencies.dtheta_dt_rad.fill(23.0f);
    REQUIRE_NOTHROW(step_radiation(state, 0.0));
    REQUIRE(radiation.held_shortwave_incoming_power_w > 0.0);
    REQUIRE(radiation.held_shortwave_atmospheric_absorbed_power_w > 0.0);
    REQUIRE(radiation.held_surface_absorbed_power_w > 0.0);
    REQUIRE(radiation.held_reflection_converged);
    REQUIRE(radiation.held_unresolved_reflected_power_w <= 1.0e-8 * radiation.held_shortwave_incoming_power_w);
    REQUIRE(radiation.held_shortwave_incoming_power_w ==
            Approx(radiation.held_shortwave_atmospheric_absorbed_power_w + radiation.held_surface_absorbed_power_w +
                   radiation.held_top_escape_power_w + radiation.held_lateral_escape_power_w +
                   radiation.held_unresolved_reflected_power_w)
                .epsilon(4.0e-14));
    REQUIRE(radiation.terrain_shortwave_workspace != nullptr);
    for (std::size_t n = 0; n < storage.tendencies.dtheta_dt_rad.size(); ++n)
        REQUIRE(storage.tendencies.dtheta_dt_rad.data()[n] != 23.0f);

    const auto   committed_tendency  = storage.tendencies.dtheta_dt_rad;
    const double committed_incoming  = radiation.held_shortwave_incoming_power_w;
    terrain.metrics.Gx_face(1, 0, 0) = std::numeric_limits<double>::quiet_NaN();
    radiation.last_update_time_s     = -60.0;
    radiation.next_update_time_s     = 0.0;
    REQUIRE_THROWS_AS(step_radiation(state, 0.0), std::invalid_argument);
    REQUIRE(fields_match(storage.tendencies.dtheta_dt_rad, committed_tendency, 0.0));
    REQUIRE(radiation.held_shortwave_incoming_power_w == committed_incoming);
}

TEST_CASE("mapped RRTMG shortwave commits spectral heating and ledgers atomically",
          "[physics][radiation][rrtmg][terrain][shortwave][spectral][runtime][transactional][external]")
{
    if (std::getenv("NIMBUS_RRTMG_LW_COEFFICIENTS") == nullptr ||
        std::getenv("NIMBUS_RRTMG_SW_COEFFICIENTS") == nullptr)
    {
        SUCCEED("official RRTMG coefficient paths were not supplied");
        return;
    }
    SimulationStorageState storage{{2, 1, 3, 1000.0, 100.0, 1.0, 800.0}};
    storage.grid.geometry.initialize(
        2, 1, 3, 1000.0, 100.0, 800.0,
        CoordinateSystem::Cartesian, StaggerType::CGrid);
    auto& fields = storage.fields;
    fields.p.resize(2, 1, 3, 0.0f);
    fields.theta.resize(2, 1, 3, 300.0f);
    fields.rho.resize(2, 1, 3, 1.0f);
    fields.qv.resize(2, 1, 3, 0.005f);
    fields.qc.resize(2, 1, 3, 0.0f);
    fields.qi.resize(2, 1, 3, 0.0f);
    for (int i = 0; i < 2; ++i)
        for (int k = 0; k < 3; ++k)
            fields.p(i, 0, k) = 95000.0f - 15000.0f * k;

    auto& terrain = storage.runtime.terrain;
    terrain.config.scheme_id = "synthetic_terrain";
    terrain.metrics.z.resize(2, 1, 3, 0.0);
    terrain.metrics.z_face.resize(2, 1, 4, 0.0);
    terrain.metrics.Gx_face.resize(2, 1, 4, 0.0);
    terrain.metrics.Gy_face.resize(2, 1, 4, 0.0);
    const double faces[2][4] = {
        {0.0, 250.0, 500.0, 800.0},
        {100.0, 300.0, 550.0, 800.0}};
    for (int i = 0; i < 2; ++i)
    {
        terrain.metrics.Gx_face(i, 0, 0) = -0.1 * i;
        for (int k = 0; k <= 3; ++k)
            terrain.metrics.z_face(i, 0, k) = faces[i][k];
        for (int k = 0; k < 3; ++k)
            terrain.metrics.z(i, 0, k) =
                0.5 * (faces[i][k] + faces[i][k + 1]);
    }

    auto& radiation = storage.runtime.radiation;
    radiation.config.scheme_id = "rrtmg";
    radiation.config.do_lw = true;
    radiation.config.do_sw = true;
    radiation.config.clear_sky_only = true;
    radiation.config.dt_radiation = 60.0;
    radiation.config.local_solar_time_at_start_h = 12.0;
    radiation.config.rrtmg_ozone_vmr = 4.0e-6;
    radiation.config.terrain_sw_cosine_nodes = 1;
    radiation.config.terrain_sw_azimuth_nodes = 4;
    radiation.config.terrain_sw_maximum_bounces = 64;
    radiation.config.terrain_sw_relative_residual_tolerance = 1.0e-8;
    radiation.config.terrain_sw_absolute_residual_tolerance_w = 1.0e-6;
    radiation.scheme = create_radiation_scheme("rrtmg");
    radiation.scheme->initialize(radiation.config);
    radiation.last_update_time_s = -60.0;
    radiation.next_update_time_s = 0.0;
    storage.runtime.boundary_layer.surface_config.Tsfc = 288.0;
    storage.runtime.boundary_layer.surface_config.albedo = 0.2;
    storage.runtime.boundary_layer.surface_config.emissivity = 0.95;
    storage.configuration.lateral_bc_west =
        tmv::init::config::LateralBC::Periodic;
    storage.configuration.lateral_bc_east =
        tmv::init::config::LateralBC::Periodic;
    storage.configuration.lateral_bc_south =
        tmv::init::config::LateralBC::Periodic;
    storage.configuration.lateral_bc_north =
        tmv::init::config::LateralBC::Periodic;
    auto state = make_simulation_state_view(storage);
    REQUIRE_NOTHROW(step_radiation(state, 0.0));
    REQUIRE(radiation.held_shortwave_incoming_power_w > 0.0);
    REQUIRE(radiation.held_shortwave_atmospheric_absorbed_power_w > 0.0);
    REQUIRE(radiation.held_surface_absorbed_power_w > 0.0);
    REQUIRE(radiation.held_reflection_converged);
    REQUIRE(radiation.held_shortwave_incoming_power_w == Approx(
        radiation.held_shortwave_atmospheric_absorbed_power_w +
        radiation.held_surface_absorbed_power_w +
        radiation.held_top_escape_power_w +
        radiation.held_lateral_escape_power_w +
        radiation.held_unresolved_reflected_power_w).epsilon(2.0e-12));
    REQUIRE(radiation.terrain_shortwave_workspace != nullptr);
    REQUIRE(radiation.terrain_shortwave_workspace
                ->spectral_absorption_optical_depth.size() == 2 * 3 * 112);
    REQUIRE(any_nonzero(storage.tendencies.dtheta_dt_rad));

#ifdef _OPENMP
    const int saved_threads = omp_get_max_threads();
    omp_set_num_threads(1);
#endif
    radiation.last_update_time_s = -60.0;
    radiation.next_update_time_s = 0.0;
    REQUIRE_NOTHROW(step_radiation(state, 0.0));
    const auto one_thread_tendency = storage.tendencies.dtheta_dt_rad;
    const double one_thread_incoming = radiation.held_shortwave_incoming_power_w;
    const double one_thread_absorbed =
        radiation.held_shortwave_atmospheric_absorbed_power_w;
#ifdef _OPENMP
    omp_set_num_threads(4);
#endif
    radiation.last_update_time_s = -60.0;
    radiation.next_update_time_s = 0.0;
    REQUIRE_NOTHROW(step_radiation(state, 0.0));
    REQUIRE(fields_match(
        storage.tendencies.dtheta_dt_rad, one_thread_tendency, 0.0));
    REQUIRE(radiation.held_shortwave_incoming_power_w == one_thread_incoming);
    REQUIRE(radiation.held_shortwave_atmospheric_absorbed_power_w ==
            one_thread_absorbed);
#ifdef _OPENMP
    omp_set_num_threads(saved_threads);
#endif

    const auto held_tendency = storage.tendencies.dtheta_dt_rad;
    const double held_incoming = radiation.held_shortwave_incoming_power_w;
    fields.p(1, 0, 1) = std::numeric_limits<float>::quiet_NaN();
    radiation.last_update_time_s = -60.0;
    radiation.next_update_time_s = 0.0;
    REQUIRE_THROWS(step_radiation(state, 0.0));
    REQUIRE(fields_match(storage.tendencies.dtheta_dt_rad, held_tendency, 0.0));
    REQUIRE(radiation.held_shortwave_incoming_power_w == held_incoming);
}
