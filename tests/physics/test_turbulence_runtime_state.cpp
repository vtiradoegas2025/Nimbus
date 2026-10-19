#include "catch2/catch.hpp"
#include "core/runtime/simulation_state.hpp"
#include "core/runtime/simulation_runtime.hpp"
#include "numerics/advection/advection_base.hpp"
#include "numerics/diffusion/diffusion_base.hpp"
#include "numerics/numerics_base.hpp"
#include "numerics/time_stepping/time_stepping_base.hpp"
#include "turbulence/turbulence_base.hpp"
#include "util/log_profile.hpp"

#include <cmath>
#include <limits>
#include <memory>
#include <string>
#include <vector>

SimulationStorageState global_simulation_storage{{2, 2, 2,
                                                  100.0, 50.0, 1.0, 100.0}};
int& NR = global_simulation_storage.grid.nr;
int& NTH = global_simulation_storage.grid.nth;
int& NZ = global_simulation_storage.grid.nz;
double& dr = global_simulation_storage.grid.dx;
double& dz = global_simulation_storage.grid.dz;
double& dt = global_simulation_storage.grid.dt;
double& dtheta = global_simulation_storage.grid.dy;
GridGeometry& global_grid_geometry = global_simulation_storage.grid.geometry;
tmv::GridMetrics turbulence_test_grid_metrics;
tmv::GridMetrics& tmv::global_grid_metrics = turbulence_test_grid_metrics;
tmv::RadiationRuntimeState& tmv::global_radiation_runtime_state = global_simulation_storage.runtime.radiation;
tmv::TurbulenceRuntimeState& tmv::global_turbulence_runtime_state = global_simulation_storage.runtime.turbulence;
tmv::BoundaryLayerRuntimeState& tmv::global_boundary_layer_runtime_state = global_simulation_storage.runtime.boundary_layer;
tmv::chaos::ChaosRuntimeState& tmv::chaos::global_chaos_runtime_state = global_simulation_storage.runtime.chaos;
RayleighDampingRuntimeState& global_rayleigh_runtime_state = global_simulation_storage.runtime.rayleigh;
tmv::MicrophysicsRuntimeState& tmv::global_microphysics_runtime_state = global_simulation_storage.runtime.microphysics;
tmv::AdvectionRuntimeState& tmv::global_advection_runtime_state = global_simulation_storage.runtime.advection;
tmv::DiffusionRuntimeState& tmv::global_diffusion_runtime_state = global_simulation_storage.runtime.diffusion;
tmv::TimeSteppingRuntimeState& tmv::global_time_stepping_runtime_state = global_simulation_storage.runtime.time_stepping;

Field3D& rho = global_simulation_storage.fields.rho;
Field3D& p = global_simulation_storage.fields.p;
Field3D& u = global_simulation_storage.fields.u;
Field3D& w = global_simulation_storage.fields.w;
Field3D& v = global_simulation_storage.fields.v;
Field3D& tracer = global_simulation_storage.fields.tracer;
Field3D& theta = global_simulation_storage.fields.theta;
Field3D& qv = global_simulation_storage.fields.qv;
Field3D& qc = global_simulation_storage.fields.qc;
Field3D& qr = global_simulation_storage.fields.qr;
Field3D& qi = global_simulation_storage.fields.qi;
Field3D& qs = global_simulation_storage.fields.qs;
Field3D& qh = global_simulation_storage.fields.qh;
Field3D& qg = global_simulation_storage.fields.qg;
Field3D& tke = global_simulation_storage.fields.tke;
Field3D& dtheta_dt_rad = global_simulation_storage.tendencies.dtheta_dt_rad;
Field3D& dtheta_dt_pbl = global_simulation_storage.tendencies.dtheta_dt_pbl;
Field3D& dqv_dt_pbl = global_simulation_storage.tendencies.dqv_dt_pbl;
Field3D& du_dt_pbl = global_simulation_storage.tendencies.du_dt_pbl;
Field3D& dv_dt_pbl = global_simulation_storage.tendencies.dv_dt_pbl;
Field3D& dtke_dt_pbl = global_simulation_storage.tendencies.dtke_dt_pbl;

std::vector<float>& sfc_ustar = global_simulation_storage.surface.ustar;
std::vector<float>& sfc_tau_u = global_simulation_storage.surface.tau_u;
std::vector<float>& sfc_tau_v = global_simulation_storage.surface.tau_v;
std::vector<float>& sfc_sensible_heat_flux = global_simulation_storage.surface.sensible_heat_flux;
std::vector<float>& sfc_latent_heat_flux = global_simulation_storage.surface.latent_heat_flux;
std::vector<float>& sfc_moisture_flux = global_simulation_storage.surface.moisture_flux;
std::vector<float>& pbl_height_2d = global_simulation_storage.surface.pbl_height;

std::vector<double>& rho0_base = global_simulation_storage.base.rho0;
std::vector<double>& p0_base = global_simulation_storage.base.p0;
std::vector<double>& qv0_base = global_simulation_storage.base.qv0;
std::vector<double>& theta0_base = global_simulation_storage.base.theta0;
std::vector<double>& u0_base = global_simulation_storage.base.u0;
std::vector<double>& v0_base = global_simulation_storage.base.v0;

TEST_CASE("Turbulence capabilities are complete without concrete built-ins",
          "[physics][turbulence][descriptor]")
{
    const auto& descriptors = get_turbulence_scheme_descriptors();
    for (const auto& descriptor : descriptors)
    {
        REQUIRE(tmv::has_complete_scheme_capabilities(descriptor));
    }

    const auto* smag = tmv::find_scheme_descriptor(descriptors, "smag");
    const auto* tke_descriptor = tmv::find_scheme_descriptor(descriptors, "1.5-order");
    REQUIRE(smag != nullptr);
    REQUIRE(smag->capabilities.staggerings
            == std::vector<std::string>{"collocated", "c_grid"});
    REQUIRE(smag->capabilities.solver_high_water_delta_fields == 0);
    REQUIRE(smag->capabilities.cpu_gpu_parity_coverage == tmv::SchemeCoverage::Covered);
    REQUIRE(tke_descriptor != nullptr);
    REQUIRE(tke_descriptor->capabilities.staggerings ==
            std::vector<std::string>{"collocated", "c_grid"});
    REQUIRE(tke_descriptor->capabilities.solver_high_water_delta_fields == 5);
    REQUIRE(tke_descriptor->capabilities.gpu_implementation_coverage == tmv::SchemeCoverage::Covered);
}

double& coriolis_f = global_simulation_storage.coriolis_f;
double& simulation_time = global_simulation_storage.time_s;
LogProfile& global_log_profile =
    (global_simulation_storage.configuration.log_profile = LogProfile::quiet);

// This isolated runtime-state fixture exercises the host path. Physical native
// dispatch and backend ownership are covered by the accelerator parity suites.
const ComputeBackend* active_compute_backend()
{
    return nullptr;
}

namespace
{

class StubTurbulenceScheme : public TurbulenceSchemeBase
{
public:
    std::string name() const override { return "stub"; }
    int required_fields() const override { return 0; }
    void initialize(const TurbulenceConfig&) override {}

    void compute(const TurbulenceConfig&,
                 const GridMetrics& grid,
                 const TurbulenceStateView& state,
                 TurbulenceTendencies& tend,
                 TurbulenceDiagnostics* diagnostics) override
    {
        tend.dudt_sgs.resize(state.NR, state.NTH, state.NZ, static_cast<float>(grid.dx / 100.0));
        tend.dvdt_sgs.resize(state.NR, state.NTH, state.NZ, 0.0f);
        tend.dwdt_sgs.resize(state.NR, state.NTH, state.NZ, 0.0f);
        tend.dthetadt_sgs.resize(state.NR, state.NTH, state.NZ, 0.0f);
        tend.dqvdt_sgs.resize(state.NR, state.NTH, state.NZ, 0.0f);
        tend.dtkedt_sgs.resize(state.NR, state.NTH, state.NZ, 0.0f);

        for (int i = 0; i < state.NR; ++i)
        {
            for (int j = 0; j < state.NTH; ++j)
            {
                for (int k = 0; k < state.NZ; ++k)
                {
                    tend.dthetadt_sgs[i][j][k] = (*state.theta)[i][j][k] - 300.0f;
                    tend.dqvdt_sgs[i][j][k] = (*state.qv)[i][j][k];
                    tend.dtkedt_sgs[i][j][k] =
                        state.tke ? (*state.tke)[i][j][k] : -1.0f;
                }
            }
        }
        if (diagnostics != nullptr)
        {
            Field3D* fields[] = {
                &diagnostics->nu_t,
                &diagnostics->K_theta,
                &diagnostics->K_q,
                &diagnostics->K_tke,
                &diagnostics->shear_prod,
                &diagnostics->buoy_prod,
                &diagnostics->dissipation,
                &diagnostics->Cs_eff,
                &diagnostics->Pr_t_eff,
            };
            for (std::size_t index = 0; index < std::size(fields); ++index)
                fields[index]->resize(
                    state.NR, state.NTH, state.NZ,
                    static_cast<float>(index + 1));
        }
    }
};

class ActiveTurbulenceStub final : public StubTurbulenceScheme
{
public:
    std::string name() const override { return "smagorinsky"; }
    std::unique_ptr<TurbulenceSchemeBase> clone() const override
    {
        return std::make_unique<ActiveTurbulenceStub>(*this);
    }
};

class NonFiniteTurbulenceScheme final : public TurbulenceSchemeBase
{
public:
    std::string name() const override { return "nonfinite_stub"; }
    int required_fields() const override { return 0; }
    void initialize(const TurbulenceConfig&) override {}

    void compute(const TurbulenceConfig&,
                 const GridMetrics&,
                 const TurbulenceStateView& state,
                 TurbulenceTendencies& tend,
                 TurbulenceDiagnostics*) override
    {
        tend.dudt_sgs.resize(state.NR, state.NTH, state.NZ, 1.0f);
        tend.dvdt_sgs.resize(state.NR, state.NTH, state.NZ, 2.0f);
        tend.dwdt_sgs.resize(state.NR, state.NTH, state.NZ, 3.0f);
        tend.dthetadt_sgs.resize(state.NR, state.NTH, state.NZ, 4.0f);
        tend.dqvdt_sgs.resize(state.NR, state.NTH, state.NZ, 5.0f);
        tend.dtkedt_sgs.resize(state.NR, state.NTH, state.NZ, 6.0f);
        tend.dthetadt_sgs[0][0][0] = NAN;
    }
};

bool all_equal(const Field3D& field, float expected)
{
    for (int i = 0; i < static_cast<int>(field.size_r()); ++i)
    {
        for (int j = 0; j < static_cast<int>(field.size_th()); ++j)
        {
            for (int k = 0; k < static_cast<int>(field.size_z()); ++k)
            {
                if (std::abs(field[i][j][k] - expected) > 1.0e-6f)
                {
                    return false;
                }
            }
        }
    }
    return true;
}

void initialize_process_globals()
{
    NR = 2;
    NTH = 2;
    NZ = 2;
    dr = 100.0;
    dtheta = 100.0;
    dz = 50.0;
    dt = 1.0;
    global_grid_geometry.initialize(NR, NTH, NZ, dr, dz, dtheta,
                                    CoordinateSystem::Cartesian);
    global_grid_metrics.dx = 100.0;
    global_grid_metrics.dy = 100.0;
    global_grid_metrics.dz.assign(static_cast<std::size_t>(NZ), dz);
    global_grid_metrics.z_int = {0.0, 50.0, 100.0};

    p.resize(NR, NTH, NZ, 90000.0f);
    rho.resize(NR, NTH, NZ, 1.0f);
    u.resize(NR, NTH, NZ, 0.0f);
    w.resize(NR, NTH, NZ, 0.0f);
    v.resize(NR, NTH, NZ, 0.0f);
    tracer.resize(NR, NTH, NZ, 0.0f);
    theta.resize(NR, NTH, NZ, 280.0f);
    qv.resize(NR, NTH, NZ, 0.001f);
    qc.resize(NR, NTH, NZ, 0.0f);
    qr.resize(NR, NTH, NZ, 0.0f);
    qi.resize(NR, NTH, NZ, 0.0f);
    qs.resize(NR, NTH, NZ, 0.0f);
    qh.resize(NR, NTH, NZ, 0.0f);
    qg.resize(NR, NTH, NZ, 0.0f);
    tke.resize(NR, NTH, NZ, 0.0f);
}

} // namespace

std::unique_ptr<tmv::TurbulenceSchemeBase> tmv::create_turbulence_scheme(const std::string&)
{
    return std::make_unique<StubTurbulenceScheme>();
}

TEST_CASE("legacy turbulence globals reference the owning runtime",
          "[turbulence][runtime][state][modularity]")
{
    REQUIRE(&turbulence_scheme == &global_turbulence_runtime_state.scheme);
    REQUIRE(&global_turbulence_config == &global_turbulence_runtime_state.config);

    TurbulenceConfig config;
    config.scheme_id = "stub";
    config.dt_sgs = 7.5;
    initialize_turbulence("stub", config);

    REQUIRE(global_turbulence_runtime_state.scheme != nullptr);
    REQUIRE(global_turbulence_runtime_state.config.scheme_id == "stub");
    REQUIRE(global_turbulence_runtime_state.last_update_time_s == Approx(-7.5));
    REQUIRE(global_turbulence_runtime_state.next_update_time_s == Approx(0.0));
}

TEST_CASE("turbulence initialization preserves supported mode semantics",
          "[turbulence][runtime][config][sci021]")
{
    TurbulenceConfig config;
    config.scheme_id = "smagorinsky";

    SECTION("strict-v1 2d token normalizes to horizontal-only SGS")
    {
        config.mode = "2d";
        REQUIRE_NOTHROW(initialize_turbulence(config.scheme_id, config));
        REQUIRE(global_turbulence_config.mode == "horizontal_only");
    }

    SECTION("invalid mode throws instead of changing the experiment to 3d")
    {
        config.mode = "definitely_not_a_turbulence_mode";
        REQUIRE_THROWS(initialize_turbulence(config.scheme_id, config));
    }
}

TEST_CASE("turbulence initialization rejects parameter substitution",
          "[turbulence][runtime][config][config-truth]")
{
    TurbulenceConfig config;
    config.scheme_id = "smagorinsky";

    SECTION("invalid numerical coefficient")
    {
        config.Pr_t = 0.0;
        REQUIRE_THROWS_AS(
            initialize_turbulence(config.scheme_id, config), std::invalid_argument);
    }

    SECTION("invalid filter width")
    {
        config.filter_width = "magic_width";
        REQUIRE_THROWS_AS(
            initialize_turbulence(config.scheme_id, config), std::invalid_argument);
    }

    SECTION("user filter width requires an explicit physical length")
    {
        config.filter_width = "user";
        REQUIRE_THROWS_AS(
            initialize_turbulence(config.scheme_id, config), std::invalid_argument);
        config.user_filter_width_m = 125.0;
        REQUIRE_NOTHROW(initialize_turbulence(config.scheme_id, config));
        REQUIRE(global_turbulence_config.filter_width == "user");
        REQUIRE(global_turbulence_config.user_filter_width_m == 125.0);
    }

    SECTION("user width is rejected for a metric-derived selector")
    {
        config.user_filter_width_m = 125.0;
        REQUIRE_THROWS_AS(
            initialize_turbulence(config.scheme_id, config), std::invalid_argument);
    }

    SECTION("invalid user width fails closed")
    {
        config.filter_width = "user";
        config.user_filter_width_m =
            std::numeric_limits<double>::infinity();
        REQUIRE_THROWS_AS(
            initialize_turbulence(config.scheme_id, config), std::invalid_argument);
    }

    SECTION("invalid stability correction")
    {
        config.stability_correction = "magic_stability";
        REQUIRE_THROWS_AS(
            initialize_turbulence(config.scheme_id, config), std::invalid_argument);
    }

    SECTION("TKE-based stability selector is restricted to TKE")
    {
        config.stability_correction = "tke_based";
        REQUIRE_THROWS_AS(
            initialize_turbulence(config.scheme_id, config), std::invalid_argument);
        config.scheme_id = "tke";
        REQUIRE_NOTHROW(initialize_turbulence(config.scheme_id, config));
        REQUIRE(global_turbulence_config.stability_correction == "tke_based");
    }

    SECTION("dynamic coefficient is accepted only by Smagorinsky")
    {
        config.dynamic_Cs = true;
        REQUIRE_NOTHROW(initialize_turbulence(config.scheme_id, config));
        REQUIRE(global_turbulence_config.dynamic_Cs);
        config.scheme_id = "tke";
        REQUIRE_THROWS_AS(initialize_turbulence(
            config.scheme_id, config), std::invalid_argument);
    }

    SECTION("moist buoyancy is accepted")
    {
        config.enable_moist_buoyancy = true;
        REQUIRE_NOTHROW(initialize_turbulence(config.scheme_id, config));
        REQUIRE(global_turbulence_config.enable_moist_buoyancy);
    }

    SECTION("runtime diagnostics publication is accepted")
    {
        config.enable_diagnostics = true;
        REQUIRE_NOTHROW(initialize_turbulence(config.scheme_id, config));
        REQUIRE(global_turbulence_config.enable_diagnostics);
    }
}

TEST_CASE("turbulence diagnostics publish transactionally into runtime state",
          "[turbulence][runtime][diagnostics][state]")
{
    initialize_process_globals();
    auto& runtime = global_turbulence_runtime_state;
    runtime = TurbulenceRuntimeState{};
    runtime.config.dt_sgs = 1.0;
    runtime.config.enable_diagnostics = true;
    runtime.scheme = std::make_unique<ActiveTurbulenceStub>();
    runtime.last_update_time_s = -1.0;
    runtime.next_update_time_s = 0.0;

    REQUIRE(step_turbulence(0.0, runtime.held_tendencies));
    REQUIRE(all_equal(runtime.held_diagnostics.nu_t, 1.0f));
    REQUIRE(all_equal(runtime.held_diagnostics.K_tke, 4.0f));
    REQUIRE(all_equal(runtime.held_diagnostics.dissipation, 7.0f));
    REQUIRE(all_equal(runtime.held_diagnostics.Pr_t_eff, 9.0f));

    REQUIRE(step_turbulence(1.0, runtime.held_tendencies));
    const TurbulenceRuntimeState snapshot(runtime);
    REQUIRE(all_equal(snapshot.held_diagnostics.nu_t, 1.0f));
    REQUIRE(snapshot.candidate_diagnostics.nu_t.empty());
    runtime = TurbulenceRuntimeState{};
}

TEST_CASE("turbulence runtime step uses explicit SimulationStateView")
{
    initialize_process_globals();
    global_turbulence_config.dt_sgs = 1.0;
    turbulence_scheme = std::make_unique<StubTurbulenceScheme>();

    int local_nr = 2;
    int local_nth = 2;
    int local_nz = 2;
    double local_dr = 200.0;
    double local_dtheta = 250.0;
    double local_dz = 75.0;
    double local_dt = 1.0;
    GridGeometry local_geometry;
    local_geometry.initialize(local_nr, local_nth, local_nz,
                              local_dr, local_dz, local_dtheta,
                              CoordinateSystem::Cartesian);

    Field3D local_rho(local_nr, local_nth, local_nz, 1.0f);
    Field3D local_p(local_nr, local_nth, local_nz, 90000.0f);
    Field3D local_u(local_nr, local_nth, local_nz, 0.0f);
    Field3D local_w(local_nr, local_nth, local_nz, 0.0f);
    Field3D local_v(local_nr, local_nth, local_nz, 0.0f);
    Field3D local_tracer(local_nr, local_nth, local_nz, 0.0f);
    Field3D local_theta(local_nr, local_nth, local_nz, 310.0f);
    Field3D local_qv(local_nr, local_nth, local_nz, 0.012f);
    Field3D local_qc(local_nr, local_nth, local_nz, 0.0f);
    Field3D local_qr(local_nr, local_nth, local_nz, 0.0f);
    Field3D local_qi(local_nr, local_nth, local_nz, 0.0f);
    Field3D local_qs(local_nr, local_nth, local_nz, 0.0f);
    Field3D local_qh(local_nr, local_nth, local_nz, 0.0f);
    Field3D local_qg(local_nr, local_nth, local_nz, 0.0f);
    Field3D local_tke(local_nr, local_nth, local_nz, 0.5f);
    Field3D local_dtheta_dt_rad(local_nr, local_nth, local_nz, 0.5f);
    Field3D local_dtheta_dt_pbl(local_nr, local_nth, local_nz, 0.5f);
    Field3D local_dqv_dt_pbl(local_nr, local_nth, local_nz, 0.5f);
    Field3D local_du_dt_pbl(local_nr, local_nth, local_nz, 0.5f);
    Field3D local_dv_dt_pbl(local_nr, local_nth, local_nz, 0.5f);
    Field3D local_dtke_dt_pbl(local_nr, local_nth, local_nz, 0.5f);

    std::vector<double> local_rho0(local_nz, 1.0);
    std::vector<double> local_p0(local_nz, 90000.0);
    std::vector<double> local_qv0(local_nz, 0.012);
    std::vector<double> local_theta0(local_nz, 310.0);
    std::vector<double> local_u0(local_nz, 0.0);
    std::vector<double> local_v0(local_nz, 0.0);
    double local_coriolis = 0.0;
    double local_time = 0.0;

    SimulationStateView local_state{
        {local_nr, local_nth, local_nz, local_dr, local_dz,
         local_dt, local_dtheta, local_geometry},
        {local_rho, local_p, local_u, local_w, local_v, local_tracer,
         local_theta, local_qv, local_qc, local_qr, local_qi, local_qs,
         local_qh, local_qg, local_tke},
        {local_rho0, local_p0, local_qv0, local_theta0, local_u0, local_v0},
        {local_dtheta_dt_rad, local_dtheta_dt_pbl, local_dqv_dt_pbl,
         local_du_dt_pbl, local_dv_dt_pbl, local_dtke_dt_pbl},
        local_coriolis,
        local_time};

    local_state.turbulence_runtime = &global_turbulence_runtime_state;
    GridMetrics local_process_metrics;
    local_process_metrics.dx = local_dr;
    local_process_metrics.dy = local_dtheta;
    local_process_metrics.dz.assign(static_cast<std::size_t>(local_nz), local_dz);
    local_process_metrics.z_int = local_geometry.z_faces;
    local_process_metrics.staggering = local_geometry.staggered ? StaggerType::CGrid : StaggerType::Collocated;
    local_state.process_grid_metrics = &local_process_metrics;
    local_state.configuration = &global_simulation_storage.configuration;
    local_state.domain_frame = &global_simulation_storage.configuration.domain_frame;
    TurbulenceTendencies tendencies;
    REQUIRE(step_turbulence(local_state, 0.0, tendencies));
    REQUIRE_FALSE(tendencies.resident_source_unchanged);

    REQUIRE(all_equal(tendencies.dudt_sgs, 2.0f));
    REQUIRE(all_equal(tendencies.dthetadt_sgs, 10.0f));
    REQUIRE(all_equal(tendencies.dqvdt_sgs, 0.012f));
    REQUIRE(all_equal(tendencies.dtkedt_sgs, 0.5f));
    REQUIRE(all_equal(theta, 280.0f));
    REQUIRE(all_equal(qv, 0.001f));
    REQUIRE(all_equal(tke, 0.0f));

    // The caller uses this flag to avoid applying stochastic multipliers to
    // the same cached SGS tendency more than once between refreshes.
    tendencies.dudt_sgs.fill(7.0f);
    REQUIRE_FALSE(step_turbulence(local_state, 0.5, tendencies));
    REQUIRE(all_equal(tendencies.dudt_sgs, 7.0f));
    REQUIRE(step_turbulence(local_state, 1.0, tendencies));
    REQUIRE(all_equal(tendencies.dudt_sgs, 2.0f));
}

TEST_CASE("caller-owned terrain cannot bypass the turbulence stress guard",
          "[turbulence][runtime][terrain][state][fail-closed]")
{
    constexpr int nr = 4, nth = 4, nz = 3;
    SimulationStorageState storage{{nr, nth, nz, 100.0, 80.0, 1.0, 120.0}};
    storage.grid.geometry.initialize(
        nr, nth, nz, 100.0, 80.0, 120.0,
        CoordinateSystem::Cartesian, StaggerType::CGrid);
    storage.fields.rho.resize(nr, nth, nz, 1.0f);
    storage.fields.u.resize(nr, nth, nz, 0.0f);
    storage.fields.v.resize(nr, nth, nz, 0.0f);
    storage.fields.w.resize(nr, nth, nz, 0.0f);
    storage.fields.theta.resize(nr, nth, nz, 300.0f);
    storage.fields.qv.resize(nr, nth, nz, 0.01f);
    storage.fields.tke.resize(nr, nth, nz, 0.1f);
    storage.base.rho0.assign(nz, 1.0);

    auto& terrain = storage.runtime.terrain;
    terrain.config.scheme_id = "bell";
    terrain.metrics.z.resize(nr, nth, nz, 0.0);
    terrain.metrics.z_face.resize(nr, nth, nz + 1, 0.0);
    for (int i = 0; i < nr; ++i)
        for (int j = 0; j < nth; ++j)
        {
            double lower = 5.0 * i;
            terrain.metrics.z_face(i, j, 0) = lower;
            for (int k = 0; k < nz; ++k)
            {
                const double upper = lower + 60.0 + i + k;
                terrain.metrics.z(i, j, k) = 0.5 * (lower + upper);
                terrain.metrics.z_face(i, j, k + 1) = upper;
                lower = upper;
            }
        }

    auto& runtime = storage.runtime.turbulence;
    runtime.config.scheme_id = "stub";
    runtime.config.dt_sgs = 1.0;
    runtime.scheme = std::make_unique<StubTurbulenceScheme>();
    runtime.last_update_time_s = -1.0;
    runtime.next_update_time_s = 0.0;
    SimulationStateView state = make_simulation_state_view(storage);
    TurbulenceTendencies tendencies;

    REQUIRE_THROWS_WITH(
        step_turbulence(state, 0.0, tendencies),
        Catch::Contains("unavailable with terrain"));
    REQUIRE(runtime.held_tendencies.dudt_sgs.empty());
    REQUIRE(runtime.last_update_time_s == Approx(-1.0));
    REQUIRE(runtime.next_update_time_s == Approx(0.0));
}

TEST_CASE("interleaved turbulence states own scheme configuration and cadence",
          "[turbulence][runtime][state][sci020]")
{
    initialize_process_globals();

    int local_nr = NR, local_nth = NTH, local_nz = NZ;
    double local_dr = 200.0, local_dtheta = 250.0, local_dz = 75.0;
    double local_dt = 1.0;
    GridGeometry local_geometry;
    local_geometry.initialize(local_nr, local_nth, local_nz,
                              local_dr, local_dz, local_dtheta,
                              CoordinateSystem::Cartesian);
    Field3D local_rho(local_nr, local_nth, local_nz, 1.0f);
    Field3D local_p(local_nr, local_nth, local_nz, 90000.0f);
    Field3D local_u(local_nr, local_nth, local_nz, 0.0f);
    Field3D local_w = local_u, local_v = local_u, local_tracer = local_u;
    Field3D local_theta(local_nr, local_nth, local_nz, 310.0f);
    Field3D local_qv(local_nr, local_nth, local_nz, 0.012f);
    Field3D local_qc = local_u, local_qr = local_u, local_qi = local_u;
    Field3D local_qs = local_u, local_qh = local_u, local_qg = local_u;
    Field3D local_tke(local_nr, local_nth, local_nz, 0.5f);
    Field3D local_rad = local_u, local_pbl = local_u, local_qv_pbl = local_u;
    Field3D local_u_pbl = local_u, local_v_pbl = local_u, local_tke_pbl = local_u;
    std::vector<double> empty_base;
    double local_coriolis = 0.0, local_time = 0.0;
    TurbulenceRuntimeState local_runtime;
    SimulationStateView local_state{
        {local_nr, local_nth, local_nz, local_dr, local_dz,
         local_dt, local_dtheta, local_geometry},
        {local_rho, local_p, local_u, local_w, local_v, local_tracer,
         local_theta, local_qv, local_qc, local_qr, local_qi, local_qs,
         local_qh, local_qg, local_tke},
        {empty_base, empty_base, empty_base, empty_base, empty_base, empty_base},
        {local_rad, local_pbl, local_qv_pbl, local_u_pbl, local_v_pbl,
         local_tke_pbl},
        local_coriolis, local_time, {}, nullptr, &local_runtime};

    RuntimeConfigurationState local_configuration;
    tmv::domain::HorizontalDomainFrame local_frame;
    GridMetrics local_process_metrics;
    local_process_metrics.dx = local_dr;
    local_process_metrics.dy = local_dtheta;
    local_process_metrics.dz.assign(static_cast<std::size_t>(local_nz), local_dz);
    local_process_metrics.z_int = local_geometry.z_faces;
    local_process_metrics.staggering = local_geometry.staggered ? StaggerType::CGrid : StaggerType::Collocated;
    local_state.process_grid_metrics = &local_process_metrics;
    local_state.configuration = &local_configuration;
    local_state.domain_frame = &local_frame;
    TurbulenceConfig local_cfg;
    local_cfg.scheme_id = "stub";
    local_cfg.dt_sgs = 10.0;
    initialize_turbulence(local_runtime, "stub", local_cfg);

    TurbulenceConfig global_cfg = local_cfg;
    global_cfg.dt_sgs = 100.0;
    initialize_turbulence("stub", global_cfg);

    TurbulenceTendencies local_tendencies;
    TurbulenceTendencies global_tendencies;
    REQUIRE(step_turbulence(local_state, 0.0, local_tendencies));
    REQUIRE(all_equal(local_tendencies.dudt_sgs, 2.0f));
    REQUIRE(local_runtime.next_update_time_s == Approx(10.0));
    REQUIRE(step_turbulence(0.0, global_tendencies));
    REQUIRE(all_equal(global_tendencies.dudt_sgs, 1.0f));

    REQUIRE_FALSE(step_turbulence(local_state, 7.0, local_tendencies));
    REQUIRE(step_turbulence(local_state, 14.0, local_tendencies));
    REQUIRE(local_runtime.last_update_time_s == Approx(14.0));
    REQUIRE(local_runtime.next_update_time_s == Approx(20.0));
    REQUIRE_FALSE(step_turbulence(14.0, global_tendencies));
    REQUIRE_FALSE(step_turbulence(local_state, 19.0, local_tendencies));
    REQUIRE(step_turbulence(local_state, 21.0, local_tendencies));
    REQUIRE(local_runtime.next_update_time_s == Approx(30.0));
    REQUIRE(all_equal(local_runtime.held_tendencies.dudt_sgs, 2.0f));
}

TEST_CASE("turbulence runtime rejects invalid candidate transactionally",
          "[turbulence][runtime][state][rollback]")
{
    initialize_process_globals();
    global_turbulence_config.dt_sgs = 1.0;
    global_turbulence_runtime_state.last_update_time_s = -1.0;
    global_turbulence_runtime_state.next_update_time_s = 0.0;
    turbulence_scheme = std::make_unique<NonFiniteTurbulenceScheme>();

    TurbulenceTendencies published;
    published.dudt_sgs.resize(NR, NTH, NZ, 11.0f);
    published.dvdt_sgs.resize(NR, NTH, NZ, 12.0f);
    published.dwdt_sgs.resize(NR, NTH, NZ, 13.0f);
    published.dthetadt_sgs.resize(NR, NTH, NZ, 14.0f);
    published.dqvdt_sgs.resize(NR, NTH, NZ, 15.0f);
    published.dtkedt_sgs.resize(NR, NTH, NZ, 16.0f);
    published.resident_source_unchanged = true;

    REQUIRE_THROWS_AS(step_turbulence(0.0, published), std::runtime_error);
    REQUIRE(all_equal(published.dudt_sgs, 11.0f));
    REQUIRE(all_equal(published.dvdt_sgs, 12.0f));
    REQUIRE(all_equal(published.dwdt_sgs, 13.0f));
    REQUIRE(all_equal(published.dthetadt_sgs, 14.0f));
    REQUIRE(all_equal(published.dqvdt_sgs, 15.0f));
    REQUIRE(all_equal(published.dtkedt_sgs, 16.0f));
    REQUIRE(published.resident_source_unchanged);
    REQUIRE(global_turbulence_runtime_state.last_update_time_s == Approx(-1.0));
    REQUIRE(global_turbulence_runtime_state.next_update_time_s == Approx(0.0));
}

TEST_CASE("turbulence refresh reuses two transactional field sets",
          "[turbulence][runtime][state][allocation]")
{
    initialize_process_globals();
    global_grid_geometry.initialize(
        NR, NTH, NZ, dr, dz, dtheta,
        CoordinateSystem::Cartesian, StaggerType::CGrid);
    auto& runtime = global_turbulence_runtime_state;
    runtime = TurbulenceRuntimeState{};
    runtime.config.dt_sgs = 1.0;
    runtime.scheme = std::make_unique<ActiveTurbulenceStub>();
    runtime.last_update_time_s = -1.0;
    runtime.next_update_time_s = 0.0;

    TurbulenceTendencies& published = runtime.held_tendencies;
    REQUIRE(step_turbulence(0.0, published));
    const float* first = runtime.held_tendencies.dudt_sgs.data();
    REQUIRE(first != nullptr);

    REQUIRE(step_turbulence(1.0, published));
    const float* second = runtime.held_tendencies.dudt_sgs.data();
    REQUIRE(second != nullptr);
    REQUIRE(second != first);

    REQUIRE(step_turbulence(2.0, published));
    REQUIRE(runtime.held_tendencies.dudt_sgs.data() == first);
    REQUIRE(runtime.candidate_tendencies.dudt_sgs.data() == second);

    runtime.grid_metrics_scratch.dz = {1.0, 2.0};
    runtime.grid_metrics_scratch.z_int = {0.0, 1.0, 3.0};
    runtime.scheme.reset();
    const TurbulenceRuntimeState snapshot(runtime);
    REQUIRE(snapshot.candidate_tendencies.dudt_sgs.empty());
    REQUIRE(snapshot.grid_metrics_scratch.dz.empty());
    REQUIRE(snapshot.grid_metrics_scratch.z_int.empty());
    REQUIRE_FALSE(snapshot.held_tendencies.dudt_sgs.empty());
}

TEST_CASE("forced SGS refresh never reuses a changed nonlinear state",
          "[turbulence][runtime][cadence][sci086]")
{
    initialize_process_globals();
    TurbulenceConfig cfg;
    cfg.scheme_id = "stub";
    cfg.dt_sgs = 10.0;
    initialize_turbulence("stub", cfg);

    TurbulenceTendencies tendencies;
    REQUIRE(step_turbulence(0.0, tendencies));
    REQUIRE(all_equal(tendencies.dthetadt_sgs, -20.0f));

    // Ordinary diagnostic cadence still reports no scheduled refresh, while
    // the production integration primitive must sample the new state now.
    theta.fill(315.0f);
    REQUIRE_FALSE(step_turbulence(0.5, tendencies));
    REQUIRE(all_equal(tendencies.dthetadt_sgs, -20.0f));
    refresh_turbulence(global_simulation_state(), 0.5, tendencies);
    REQUIRE(all_equal(tendencies.dthetadt_sgs, 15.0f));
    REQUIRE(global_turbulence_runtime_state.last_update_time_s == Approx(0.5));
    REQUIRE(global_turbulence_runtime_state.next_update_time_s == Approx(10.5));
}

TEST_CASE("real closure step API cannot expose a stale cadence sample",
          "[turbulence][runtime][cadence][direct][sci086]")
{
    initialize_process_globals();
    global_grid_geometry.initialize(
        NR, NTH, NZ, dr, dz, dtheta,
        CoordinateSystem::Cartesian, StaggerType::CGrid);
    global_turbulence_config = TurbulenceConfig{};
    global_turbulence_config.scheme_id = "smagorinsky";
    global_turbulence_config.dt_sgs = 10.0;
    global_turbulence_runtime_state.last_update_time_s = -10.0;
    global_turbulence_runtime_state.next_update_time_s = 0.0;
    turbulence_scheme = std::make_unique<ActiveTurbulenceStub>();

    TurbulenceTendencies tendencies;
    REQUIRE(step_turbulence(0.0, tendencies));
    REQUIRE(all_equal(tendencies.dthetadt_sgs, -20.0f));
    theta.fill(315.0f);
    REQUIRE(step_turbulence(0.5, tendencies));
    REQUIRE(all_equal(tendencies.dthetadt_sgs, 15.0f));
    REQUIRE(global_turbulence_runtime_state.last_update_time_s == Approx(0.5));
    REQUIRE(global_turbulence_runtime_state.next_update_time_s == Approx(10.5));
}

TEST_CASE("SGS cadence rejects unrepresentable progress before publication",
          "[turbulence][runtime][cadence][atomic]")
{
    initialize_process_globals();
    TurbulenceConfig cfg;
    cfg.scheme_id = "stub";
    cfg.dt_sgs = 1.0;
    initialize_turbulence("stub", cfg);
    TurbulenceTendencies tendencies;
    REQUIRE(step_turbulence(0.0, tendencies));
    cfg.dt_sgs = 1.0e-25;
    REQUIRE_THROWS(initialize_turbulence("stub", cfg));
    REQUIRE(global_turbulence_config.dt_sgs == 1.0);
    auto& runtime = global_turbulence_runtime_state;
    const double last = runtime.last_update_time_s;
    const double next = runtime.next_update_time_s;
    REQUIRE_THROWS(step_turbulence(1.0e20, tendencies));
    REQUIRE(runtime.last_update_time_s == last);
    REQUIRE(runtime.next_update_time_s == next);
    REQUIRE(all_equal(tendencies.dthetadt_sgs, -20.0f));
    REQUIRE(step_turbulence(1000000.5, tendencies));
    REQUIRE(runtime.next_update_time_s == Approx(1000001.0));
}
