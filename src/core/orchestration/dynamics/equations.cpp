#include <algorithm>
#include <iostream>
#include <cmath>
#include <cstdlib>
#include <exception>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include "core/runtime/simulation_runtime.hpp"
#include "core/runtime/runtime_config.hpp"
#include "core/runtime/simulation_state.hpp"
#include "core/field/field_pool.hpp"
#include "core/infra/nested_grid.hpp"
#include "core/orchestration/dynamics/initial_conditions.hpp"
#include "core/orchestration/dynamics/thermodynamic_closure.hpp"
#include "init/hodograph/factory.hpp"
#include "init/hodograph/hodograph_source.hpp"
#include "init/sounding/diagnostics.hpp"
#include "init/sounding/factory.hpp"
#include "init/sounding/sounding_source.hpp"
#include "init/trigger/factory.hpp"
#include "init/trigger/trigger_source.hpp"
#include "util/simd_utils.hpp"
#include "util/log.hpp"
#include "numerics/advection/advection.hpp"
#include "dynamics/dynamics_base.hpp"
#include "numerics/advection/advection_base.hpp"
#include "numerics/diffusion/diffusion_base.hpp"
#include "numerics/time_stepping/time_stepping_base.hpp"
#include "terrain/conservative_transport.hpp"
#include "terrain/runtime.hpp"

#include "radar/radar_base.hpp"
#include "radar/factory.hpp"
#ifdef _OPENMP
#include <omp.h>
#endif

void compute_wind_profile(const WindProfile& profile, double z, double& u, double& v);

SimulationStorageState global_simulation_storage{{200, 128, 150, 100.0, 100.0, 0.1, 100.0}};
int&                   NR                   = global_simulation_storage.grid.nr;
int&                   NTH                  = global_simulation_storage.grid.nth;
int&                   NZ                   = global_simulation_storage.grid.nz;
double&                dr                   = global_simulation_storage.grid.dx;
double&                dz                   = global_simulation_storage.grid.dz;
double&                dt                   = global_simulation_storage.grid.dt;
double&                dtheta               = global_simulation_storage.grid.dy;
GridGeometry&          global_grid_geometry = global_simulation_storage.grid.geometry;
double&                simulation_time      = global_simulation_storage.time_s;

// dtheta is resolved Cartesian y-spacing; invalid direct callers use square cells.
void update_grid_resolution()
{
    if (!std::isfinite(dtheta) || dtheta <= 0.0)
    {
        dtheta = dr;
    }

    global_grid_geometry.initialize(NR, NTH, NZ, dr, dz, dtheta, global_coordinate_system, global_stagger_type);
}

Field3D& rho    = global_simulation_storage.fields.rho;
Field3D& p      = global_simulation_storage.fields.p;
Field3D& u      = global_simulation_storage.fields.u;
Field3D& w      = global_simulation_storage.fields.w;
Field3D& v      = global_simulation_storage.fields.v;
Field3D& tracer = global_simulation_storage.fields.tracer;

Field3D& theta = global_simulation_storage.fields.theta;
Field3D& qv    = global_simulation_storage.fields.qv;
Field3D& qc    = global_simulation_storage.fields.qc;
Field3D& qr    = global_simulation_storage.fields.qr;
Field3D& qi    = global_simulation_storage.fields.qi;
Field3D& qs    = global_simulation_storage.fields.qs;
Field3D& qh    = global_simulation_storage.fields.qh;
Field3D& qg    = global_simulation_storage.fields.qg;
Field3D& tke   = global_simulation_storage.fields.tke;

Field3D& radar_reflectivity        = global_simulation_storage.diagnostics.radar_reflectivity;
Field3D& vorticity_r               = global_simulation_storage.diagnostics.vorticity_r;
Field3D& vorticity_theta           = global_simulation_storage.diagnostics.vorticity_theta;
Field3D& vorticity_z               = global_simulation_storage.diagnostics.vorticity_z;
Field3D& stretching_term           = global_simulation_storage.diagnostics.stretching_term;
Field3D& tilting_term              = global_simulation_storage.diagnostics.tilting_term;
Field3D& baroclinic_term           = global_simulation_storage.diagnostics.baroclinic_term;
Field3D& angular_momentum          = global_simulation_storage.diagnostics.angular_momentum;
Field3D& angular_momentum_tendency = global_simulation_storage.diagnostics.angular_momentum_tendency;
Field3D& p_prime                   = global_simulation_storage.diagnostics.p_prime;
Field3D& dynamic_pressure          = global_simulation_storage.diagnostics.dynamic_pressure;
Field3D& buoyancy_pressure         = global_simulation_storage.diagnostics.buoyancy_pressure;

Field3D& dtheta_dt_pbl = global_simulation_storage.tendencies.dtheta_dt_pbl;
Field3D& dqv_dt_pbl    = global_simulation_storage.tendencies.dqv_dt_pbl;
Field3D& dqc_dt_pbl    = global_simulation_storage.tendencies.dqc_dt_pbl;
Field3D& dqi_dt_pbl    = global_simulation_storage.tendencies.dqi_dt_pbl;
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

Field3D& dtheta_dt_rad = global_simulation_storage.tendencies.dtheta_dt_rad;

std::vector<double>& rho0_base   = global_simulation_storage.base.rho0;
std::vector<double>& p0_base     = global_simulation_storage.base.p0;
std::vector<double>& qv0_base    = global_simulation_storage.base.qv0;
std::vector<double>& theta0_base = global_simulation_storage.base.theta0;
std::vector<double>& u0_base     = global_simulation_storage.base.u0;
std::vector<double>& v0_base     = global_simulation_storage.base.v0;
double&              coriolis_f  = global_simulation_storage.coriolis_f;

NestedGridConfig& nested_config = global_simulation_storage.nested.config;

Field3D& nest_rho    = global_simulation_storage.nested.rho;
Field3D& nest_p      = global_simulation_storage.nested.p;
Field3D& nest_u      = global_simulation_storage.nested.u;
Field3D& nest_w      = global_simulation_storage.nested.w;
Field3D& nest_v      = global_simulation_storage.nested.v;
Field3D& nest_theta  = global_simulation_storage.nested.theta;
Field3D& nest_qv     = global_simulation_storage.nested.qv;
Field3D& nest_qc     = global_simulation_storage.nested.qc;
Field3D& nest_qr     = global_simulation_storage.nested.qr;
Field3D& nest_qi     = global_simulation_storage.nested.qi;
Field3D& nest_qs     = global_simulation_storage.nested.qs;
Field3D& nest_qh     = global_simulation_storage.nested.qh;
Field3D& nest_qg     = global_simulation_storage.nested.qg;
Field3D& nest_tracer = global_simulation_storage.nested.tracer;

void resize_fields()
{
    update_grid_resolution();

    // A new problem/grid is a phase boundary. Scratch retained for the prior
    // shape must not remain part of this run's resident high-water footprint.
    FieldPool::instance().clear();

    // Prognostic fields — required for initialize() and the integration loop.
    rho.resize(NR, NTH, NZ, 0.0f);
    p.resize(NR, NTH, NZ, 0.0f);
    u.resize(NR, NTH, NZ, 0.0f);
    w.resize(NR, NTH, NZ, 0.0f);
    v.resize(NR, NTH, NZ, 0.0f);
    tracer.resize(NR, NTH, NZ, 0.0f);
    theta.resize(NR, NTH, NZ, 0.0f);
    qv.resize(NR, NTH, NZ, 0.0f);
    qc.resize(NR, NTH, NZ, 0.0f);
    qr.resize(NR, NTH, NZ, 0.0f);
    qi.resize(NR, NTH, NZ, 0.0f);
    qs.resize(NR, NTH, NZ, 0.0f);
    qh.resize(NR, NTH, NZ, 0.0f);
    qg.resize(NR, NTH, NZ, 0.0f);
    tke.resize(NR, NTH, NZ, 0.0f);

    // Derived/diagnostic fields are consumer-owned lazy storage. The t=0
    // output path refreshes requested diagnostics before publication, and the
    // step cadence refreshes them before later output/live/trajectory use.
    // Explicitly release prior-run capacity here: merely omitting resize would
    // leave a previous same-process run's full diagnostic working set resident.
    radar_reflectivity.release_storage();
    auto& diagnostics = global_simulation_storage.diagnostics;
    diagnostics.vorticity_r.release_storage();
    diagnostics.vorticity_theta.release_storage();
    diagnostics.vorticity_z.release_storage();
    diagnostics.stretching_term.release_storage();
    diagnostics.tilting_term.release_storage();
    diagnostics.baroclinic_term.release_storage();
    diagnostics.p_prime.release_storage();
    diagnostics.dynamic_pressure.release_storage();
    diagnostics.buoyancy_pressure.release_storage();
    diagnostics.angular_momentum.release_storage();
    diagnostics.angular_momentum_tendency.release_storage();

    // PBL/radiation tendency buffers are zero-initialized by their
    // respective initialize_*() functions before the first timestep.
}

/**
 * @brief Coordinate-aware wind initialization from explicit (u_x, u_y) columns.
 *
 * Used when the active SoundingSource carries its own hodograph (e.g.
 * FileSoundingSource extracting winds from a SHARPY column). Mirrors the
 * coordinate / stagger dispatch that the parametric apply_*_wind_*()
 * helpers use, but reads from a per-level Cartesian column rather than
 * sampling compute_wind_profile.
 */
static void apply_wind_initialization_from_column(const SimulationStateView& state,
                                                  const std::vector<double>& u_x_column,
                                                  const std::vector<double>& u_y_column)
{
    const auto&             grid   = state.grid;
    auto&                   fields = state.fields;
    const TerrainMetrics3D* terrain_metrics =
        (state.terrain_runtime != nullptr && state.terrain_runtime->config.scheme_id != "none")
            ? &state.terrain_runtime->metrics
            : nullptr;
    if (terrain_metrics != nullptr)
    {
        terrain_transport::validate_geometry(*terrain_metrics);
        if (u_x_column.size() != static_cast<std::size_t>(grid.nz) ||
            u_y_column.size() != static_cast<std::size_t>(grid.nz))
        {
            throw std::invalid_argument("terrain wind initialization requires full reference columns");
        }
        for (int k = 0; k < grid.nz; ++k)
        {
            if (!std::isfinite(u_x_column[static_cast<std::size_t>(k)]) ||
                !std::isfinite(u_y_column[static_cast<std::size_t>(k)]))
                throw std::invalid_argument("terrain wind initialization requires finite reference columns");
        }
        for (int i = 0; i < grid.nr; ++i)
            for (int j = 0; j < grid.nth; ++j)
                for (int k = 0; k < grid.nz; ++k)
                {
                    const double u_height = (i + 1 < grid.nr)
                                                ? 0.5 * (terrain_metrics->z(i, j, k) + terrain_metrics->z(i + 1, j, k))
                                                : terrain_metrics->z(i, j, k);
                    const double v_height = (j + 1 < grid.nth)
                                                ? 0.5 * (terrain_metrics->z(i, j, k) + terrain_metrics->z(i, j + 1, k))
                                                : terrain_metrics->z(i, j, k);
                    (void)terrain_transport::reference_profile_value_fp32(u_x_column, u_height, grid.dz);
                    (void)terrain_transport::reference_profile_value_fp32(u_y_column, v_height, grid.dz);
                }
    }

#pragma omp parallel for collapse(2)
    for (int i = 0; i < grid.nr; ++i)
    {
        for (int j = 0; j < grid.nth; ++j)
        {
            for (int k = 0; k < grid.nz; ++k)
            {
                const std::size_t kz = static_cast<std::size_t>(k);
                if (terrain_metrics != nullptr)
                {
                    const double u_height = (i + 1 < grid.nr)
                                                ? 0.5 * (terrain_metrics->z(i, j, k) + terrain_metrics->z(i + 1, j, k))
                                                : terrain_metrics->z(i, j, k);
                    const double v_height = (j + 1 < grid.nth)
                                                ? 0.5 * (terrain_metrics->z(i, j, k) + terrain_metrics->z(i, j + 1, k))
                                                : terrain_metrics->z(i, j, k);
                    fields.u[i][j][k]     = static_cast<float>(
                        terrain_transport::reference_profile_value_fp32(u_x_column, u_height, grid.dz));
                    fields.v[i][j][k] = static_cast<float>(
                        terrain_transport::reference_profile_value_fp32(u_y_column, v_height, grid.dz));
                }
                else
                {
                    fields.u[i][j][k] = static_cast<float>(u_x_column[kz]);
                    fields.v[i][j][k] = static_cast<float>(u_y_column[kz]);
                }
                fields.w[i][j][k] = 0.0f;
            }
        }
    }
}

struct InitialFieldSummary
{
    float theta_min = 1e10f;
    float theta_max = -1e10f;
    float p_min     = 1e10f;
    float p_max     = -1e10f;
    float rho_min   = 1e10f;
    float rho_max   = -1e10f;
    int   nan_count = 0;
    int   inf_count = 0;
};

static std::vector<double> build_grid_heights(const GridStateView& grid)
{
    std::vector<double> z_m(static_cast<std::size_t>(grid.nz));
    for (int k = 0; k < grid.nz; ++k)
    {
        z_m[static_cast<std::size_t>(k)] = grid.geometry.z[k];
    }
    return z_m;
}

static tmv::init::Sounding
build_base_sounding(const std::vector<double>& z_m, double dz_m, std::string& source_description)
{
    auto sounding_source         = tmv::init::make_sounding_source(global_sounding_source_config);
    source_description           = sounding_source->describe();
    tmv::init::Sounding sounding = sounding_source->build(z_m, dz_m);
    sounding.verify_self_consistent();
    return sounding;
}

static void assign_base_state_vectors(const BaseStateView& base, const tmv::init::Sounding& sounding)
{
    // Base-state vectors are read elsewhere (perturbation Coriolis, GPU
    // reference-state subtraction, virtual-T buoyancy). Keep the global
    // names and shapes the same so consumers don't move.
    base.rho0.assign(sounding.rho_kgm3.begin(), sounding.rho_kgm3.end());
    base.p0.assign(sounding.p_pa.begin(), sounding.p_pa.end());
    base.qv0.assign(sounding.qv_kgkg.begin(), sounding.qv_kgkg.end());
    base.theta0.assign(sounding.theta_k.begin(), sounding.theta_k.end());
}

static void log_base_sounding(const GridStateView&       grid,
                              const BaseStateView&       base,
                              const tmv::init::Sounding& sounding,
                              const std::string&         source_description)
{
    tmv::log_info("Base state initialized via ",
                  source_description,
                  ": rho0_base[0]=",
                  base.rho0[0],
                  ", rho0_base[",
                  grid.nz - 1,
                  "]=",
                  base.rho0[grid.nz - 1],
                  ", p0_base[0]=",
                  base.p0[0],
                  "Pa",
                  ", p0_base[",
                  grid.nz - 1,
                  "]=",
                  base.p0[grid.nz - 1],
                  "Pa");

    for (std::size_t k = 0; k < std::min<std::size_t>(5, sounding.size()); ++k)
    {
        tmv::log_debug("[INIT SOUNDING] k=",
                       k,
                       ", z=",
                       sounding.z_m[k],
                       "m: T=",
                       sounding.T_k[k],
                       "K, p=",
                       sounding.p_pa[k],
                       "Pa, theta=",
                       sounding.theta_k[k],
                       "K");
    }
}

static void broadcast_thermodynamic_column(const SimulationStateView& state, const tmv::init::Sounding& sounding)
{
    const auto&             grid   = state.grid;
    auto&                   fields = state.fields;
    const TerrainMetrics3D* terrain_metrics =
        (state.terrain_runtime != nullptr && state.terrain_runtime->config.scheme_id != "none")
            ? &state.terrain_runtime->metrics
            : nullptr;
    if (terrain_metrics != nullptr)
    {
        terrain_transport::validate_geometry(*terrain_metrics);
        if (sounding.size() != static_cast<std::size_t>(grid.nz))
            throw std::invalid_argument("terrain thermodynamic initialization requires a full sounding");
        apply_cartesian_terrain_thermodynamic_initialization(state, sounding.p_pa, sounding.theta_k, sounding.qv_kgkg);
    }

    const float initial_tke = 0.1f;
#pragma omp parallel for collapse(2)
    for (int i = 0; i < grid.nr; ++i)
    {
        for (int j = 0; j < grid.nth; ++j)
        {
            for (int k = 0; k < grid.nz; ++k)
            {
                const std::size_t kz = static_cast<std::size_t>(k);
                if (terrain_metrics == nullptr)
                {
                    fields.p[i][j][k]     = static_cast<float>(sounding.p_pa[kz]);
                    fields.rho[i][j][k]   = static_cast<float>(sounding.rho_kgm3[kz]);
                    fields.theta[i][j][k] = static_cast<float>(sounding.theta_k[kz]);
                    fields.qv[i][j][k]    = static_cast<float>(sounding.qv_kgkg[kz]);
                }
                fields.qc[i][j][k] = 0.0f;
                fields.qr[i][j][k] = 0.0f;
                fields.qi[i][j][k] = 0.0f;
                fields.qs[i][j][k] = 0.0f;
                fields.qh[i][j][k] = 0.0f;
                fields.qg[i][j][k] = 0.0f;
                tke[i][j][k]       = initial_tke;

                fields.u[i][j][k]      = 0.0f;
                fields.v[i][j][k]      = 0.0f;
                fields.w[i][j][k]      = 0.0f;
                fields.tracer[i][j][k] = 0.0f;
            }
        }
    }
}

static void initialize_base_state_from_sounding(const SimulationStateView& state,
                                                const tmv::init::Sounding& sounding,
                                                const std::string&         source_description)
{
    assign_base_state_vectors(state.base, sounding);
    log_base_sounding(state.grid, state.base, sounding, source_description);
    broadcast_thermodynamic_column(state, sounding);
}

static std::string format_height_optional(double x)
{
    if (!std::isfinite(x))
    {
        return std::string("(none)");
    }
    std::ostringstream os;
    os.setf(std::ios::fixed);
    os.precision(0);
    os << x << "m";
    return os.str();
}

static void
report_sounding_diagnostics(const tmv::init::Sounding& sounding, const tmv::init::WindColumn& wind, double grid_dz_m)
{
    const auto diags = tmv::init::compute_sounding_diagnostics(sounding, wind);
    tmv::log_info("Sounding diagnostics: CAPE=",
                  static_cast<int>(diags.cape_jkg),
                  " J/kg, CIN=",
                  static_cast<int>(diags.cin_jkg),
                  " J/kg, LCL=",
                  static_cast<int>(diags.lcl_m),
                  "m",
                  ", LFC=",
                  format_height_optional(diags.lfc_m),
                  ", EL=",
                  format_height_optional(diags.el_m),
                  ", PWAT=",
                  static_cast<int>(diags.pwat_mm),
                  "mm",
                  ", shear_0_6km=",
                  diags.has_kinematic ? std::to_string(static_cast<int>(diags.bulk_shear_0_6km_ms))
                                      : std::string("(n/a)"),
                  diags.has_kinematic ? " m/s" : "");

    // SCI-026: accepted source parameters and realized diagnostics must be
    // visible together. ParametricCAPE has one diagnostic target (CAPE); its
    // surface state and tropopause are procedural inputs. ParametricTargets
    // exposes the full diagnostic table, so publish every requested value and
    // an explicit resolution-aware acceptance policy. The target source uses
    // a documented heuristic for CIN/LFC/EL, hence these are qualification
    // tolerances rather than claims of exact inversion.
    const auto& source = global_sounding_source_config;
    if (source.type == tmv::init::SoundingSourceConfig::Type::ParametricCAPE)
    {
        const double requested = source.parametric.cape_target_jkg;
        const double tolerance = std::max(100.0, 0.05 * std::abs(requested));
        const double error     = std::abs(diags.cape_jkg - requested);
        const bool   within    = std::isfinite(diags.cape_jkg) && error <= tolerance;
        tmv::log_info("[SOUNDING TARGETS] source=parametric_cape",
                      " requested_cape_jkg=",
                      requested,
                      " realized_cape_jkg=",
                      diags.cape_jkg,
                      " tolerance_jkg=",
                      tolerance,
                      " within_tolerance=",
                      within ? "yes" : "no",
                      " surface_theta_k=",
                      source.parametric.surface_theta_k,
                      " surface_qv_kgkg=",
                      source.parametric.surface_qv_kgkg,
                      " tropopause_m=",
                      source.parametric.tropopause_z_m);
        if (!within)
        {
            tmv::log_warn("[SOUNDING TARGET MISMATCH] parametric_cape CAPE error=",
                          error,
                          " J/kg exceeds tolerance=",
                          tolerance,
                          " J/kg");
            if (global_execution_mode == tmv::init::config::ExecutionMode::Qualified)
            {
                throw std::runtime_error("qualified initialization rejected parametric_cape: "
                                         "realized CAPE is outside its declared tolerance");
            }
        }
    }
    else if (source.type == tmv::init::SoundingSourceConfig::Type::ParametricTargets)
    {
        const auto&  requested         = source.targets;
        const double cape_tolerance    = std::max(100.0, 0.05 * std::abs(requested.target_cape_jkg));
        const double cin_tolerance     = std::max(75.0, 0.5 * std::abs(requested.target_cin_jkg));
        const double height_tolerance  = std::max(250.0, grid_dz_m);
        const bool   lcl_target_active = requested.lcl_is_targeted();
        const auto   within            = [](double realized, double target, double tolerance)
        { return std::isfinite(realized) && std::abs(realized - target) <= tolerance; };

        const bool cape_ok = within(diags.cape_jkg, requested.target_cape_jkg, cape_tolerance);
        const bool cin_ok  = within(diags.cin_jkg, requested.target_cin_jkg, cin_tolerance);
        const bool lcl_ok  = !lcl_target_active || within(diags.lcl_m, requested.target_lcl_m, height_tolerance);
        const bool lfc_ok  = within(diags.lfc_m, requested.target_lfc_m, height_tolerance);
        const bool el_ok   = within(diags.el_m, requested.target_el_m, height_tolerance);
        const bool all_ok  = cape_ok && cin_ok && lcl_ok && lfc_ok && el_ok;

        tmv::log_info("[SOUNDING TARGETS] source=parametric_targets",
                      " requested={cape_jkg:",
                      requested.target_cape_jkg,
                      ",cin_jkg:",
                      requested.target_cin_jkg,
                      ",lcl_m:",
                      lcl_target_active ? format_height_optional(requested.target_lcl_m)
                                        : std::string("(inactive:qv_override)"),
                      ",lfc_m:",
                      requested.target_lfc_m,
                      ",el_m:",
                      requested.target_el_m,
                      "}",
                      " surface_qv_override_kgkg=",
                      requested.moisture_qv_kgkg_override > 0.0 ? std::to_string(requested.moisture_qv_kgkg_override)
                                                                : std::string("(none)"),
                      " realized={cape_jkg:",
                      diags.cape_jkg,
                      ",cin_jkg:",
                      diags.cin_jkg,
                      ",lcl_m:",
                      diags.lcl_m,
                      ",lfc_m:",
                      format_height_optional(diags.lfc_m),
                      ",el_m:",
                      format_height_optional(diags.el_m),
                      "}",
                      " tolerances={cape_jkg:",
                      cape_tolerance,
                      ",cin_jkg:",
                      cin_tolerance,
                      ",height_m:",
                      height_tolerance,
                      "}",
                      " within_tolerance=",
                      all_ok ? "yes" : "no");
        if (!all_ok)
        {
            tmv::log_warn("[SOUNDING TARGET MISMATCH] cape=",
                          cape_ok ? "ok" : "fail",
                          ", cin=",
                          cin_ok ? "ok" : "fail",
                          ", lcl=",
                          lcl_ok ? "ok" : "fail",
                          ", lfc=",
                          lfc_ok ? "ok" : "fail",
                          ", el=",
                          el_ok ? "ok" : "fail");
            if (global_execution_mode == tmv::init::config::ExecutionMode::Qualified)
            {
                throw std::runtime_error("qualified initialization rejected parametric_targets: "
                                         "one or more realized sounding diagnostics are outside "
                                         "their declared tolerances");
            }
        }
    }

    if (!std::isfinite(diags.lfc_m))
    {
        tmv::log_warn("[SOUNDING] No LFC found: surface parcel never becomes "
                      "positively buoyant. No deep convection will develop.");
    }
    else if (diags.cape_jkg < 500.0)
    {
        tmv::log_warn(
            "[SOUNDING] Marginal CAPE (", static_cast<int>(diags.cape_jkg), " J/kg). Storm development unlikely.");
    }
    if (diags.cin_jkg > 200.0)
    {
        tmv::log_warn("[SOUNDING] Strong cap (CIN=",
                      static_cast<int>(diags.cin_jkg),
                      " J/kg). The trigger may not break through.");
    }
    if (diags.has_kinematic && diags.bulk_shear_0_6km_ms < 15.0)
    {
        tmv::log_warn("[SOUNDING] Weak 0-6 km shear (",
                      static_cast<int>(diags.bulk_shear_0_6km_ms),
                      " m/s). Storm-mode organization unlikely.");
    }
}

static void apply_configured_trigger(const SimulationStateView& state)
{
    auto trigger_source = tmv::init::make_trigger_source(global_trigger_source_config);
    trigger_source->apply(state);
    tmv::log_info("Trigger source: ", trigger_source->describe());
}

static void populate_base_wind_profiles(const SimulationStateView& state, const tmv::init::WindColumn& wind)
{
    state.base.u0.resize(static_cast<std::size_t>(state.grid.nz));
    state.base.v0.resize(static_cast<std::size_t>(state.grid.nz));
    for (int k = 0; k < state.grid.nz; ++k)
    {
        const std::size_t kz = static_cast<std::size_t>(k);
        state.base.u0[kz]    = wind.u_ms[kz];
        state.base.v0[kz]    = wind.v_ms[kz];
    }
}

static void configure_coriolis_parameter(const SimulationStateView& state)
{
    const auto& frame = state.domain_frame != nullptr
        ? *state.domain_frame : global_domain_frame;
    const bool beta_plane =
        global_coriolis_mode == tmv::init::config::CoriolisMode::BetaPlane;
    const auto profile = tmv::make_cartesian_coriolis_profile(
        global_coriolis_enabled, beta_plane, global_coriolis_latitude_deg,
        frame, state.time_s);
    state.coriolis_f = profile.f_at_grid_y0;
    if (!global_coriolis_enabled)
    {
        tmv::log_info("Coriolis disabled (coriolis.enabled=false)");
        return;
    }
    if (beta_plane)
        tmv::log_info("Coriolis f(y=0) = ", state.coriolis_f,
                      " /s (latitude = ", global_coriolis_latitude_deg,
                      " deg, mode = beta_plane, beta = ", profile.beta,
                      " m^-1 s^-1)");
    else
        tmv::log_info("Coriolis f = ", state.coriolis_f,
                      " /s (latitude = ", global_coriolis_latitude_deg,
                      " deg, mode = f_plane)");
}

static void seed_tracer_marker(const SimulationStateView& state)
{
    const int ic = state.grid.nr / 4;
    const int kc = state.grid.nz / 4;
    for (int j = 0; j < state.grid.nth; ++j)
    {
        state.fields.tracer[ic][j][kc] = 1.0f;
    }
}

static InitialFieldSummary summarize_initial_fields(const SimulationStateView& state)
{
    InitialFieldSummary summary;
    const auto&         grid   = state.grid;
    const auto&         fields = state.fields;
    for (int i = 0; i < grid.nr; ++i)
    {
        for (int j = 0; j < grid.nth; ++j)
        {
            for (int k = 0; k < grid.nz; ++k)
            {
                const float theta_val = fields.theta[i][j][k];
                const float p_val     = fields.p[i][j][k];
                const float rho_val   = fields.rho[i][j][k];

                if (std::isnan(theta_val) || std::isnan(p_val) || std::isnan(rho_val))
                {
                    ++summary.nan_count;
                }
                if (std::isinf(theta_val) || std::isinf(p_val) || std::isinf(rho_val))
                {
                    ++summary.inf_count;
                }

                if (theta_val < summary.theta_min)
                {
                    summary.theta_min = theta_val;
                }
                if (theta_val > summary.theta_max)
                {
                    summary.theta_max = theta_val;
                }
                if (p_val < summary.p_min)
                {
                    summary.p_min = p_val;
                }
                if (p_val > summary.p_max)
                {
                    summary.p_max = p_val;
                }
                if (rho_val < summary.rho_min)
                {
                    summary.rho_min = rho_val;
                }
                if (rho_val > summary.rho_max)
                {
                    summary.rho_max = rho_val;
                }
            }
        }
    }
    return summary;
}

static void log_initial_field_summary(const InitialFieldSummary& summary)
{
    tmv::log_info("\n[INIT SUMMARY] After initialization:");
    tmv::log_info("  Theta: min=",
                  summary.theta_min,
                  "K, max=",
                  summary.theta_max,
                  "K, expected ~250-450K (deep domains reach ~400-430K near the model top)");
    tmv::log_info("  Pressure: min=", summary.p_min, "Pa, max=", summary.p_max, "Pa, expected ~1000-110000Pa");
    tmv::log_info("  Density: min=", summary.rho_min, "kg/m³, max=", summary.rho_max, "kg/m³, expected ~0.5-1.5kg/m³");
    tmv::log_info("  NaN count: ", summary.nan_count, ", Inf count: ", summary.inf_count);

    if (summary.theta_min < 0 || summary.theta_max > 700) // matches simulation.hpp theta_max_k
    {
        tmv::log_warn("  WARNING: Theta values are outside expected range!");
    }
    if (summary.p_min < 500 || summary.p_max > 120000)
    {
        tmv::log_warn("  WARNING: Pressure values are outside expected range!");
    }
}

/**
 * @brief Initializes the base state.
 *
 * The base state is built from a single consistent thermodynamic profile:
 *   1. T_actual(z) — a multi-layer profile (mixed layer + unstable layer +
 *      stable above unstable_top) — is the single source of truth for T.
 *   2. Pressure is integrated upward from p0 using the analytical hydrostatic
 *      solution over each layer with that T:
 *          p[k] = p[k-1] * exp( -g * dz / (R_d * T_avg) )
 *      so that ∂p/∂z = -ρg holds in the discrete sense the dynamics uses.
 *   3. Density follows from the equation of state: ρ = p / (R_d * T).
 *   4. Potential temperature follows from its definition: θ = T * (p₀/p)^κ.
 *
 * This guarantees ρ, p, T, θ are mutually consistent and the base state is
 * in true hydrostatic balance, so the dynamics sees no spurious unbalanced
 * pressure-gradient force on the first step. Previously, p and ρ were
 * derived from the US Standard Atmosphere (T = T_sfc - 6.5 K/km), while θ
 * was computed from T_actual ≠ T_std. The mismatch grew to ~40 K at the
 * model top and produced a ~2 m/s² unbalanced force at z=15 km, which
 * exploded vertical velocity within ~10 timesteps for any config.
 */
void initialize()
{
    auto                      state = global_simulation_state();
    const std::vector<double> z_m   = build_grid_heights(state.grid);

    std::string               sounding_source_description;
    const tmv::init::Sounding sounding = build_base_sounding(z_m, state.grid.dz, sounding_source_description);
    initialize_base_state_from_sounding(state, sounding, sounding_source_description);

    const tmv::init::WindColumn wind = tmv::init::resolve_wind_column(z_m, sounding, global_hodograph_source_config);
    apply_wind_initialization_from_column(state, wind.u_ms, wind.v_ms);
    report_sounding_diagnostics(sounding, wind, state.grid.dz);

    apply_configured_trigger(state);
    populate_base_wind_profiles(state, wind);
    configure_coriolis_parameter(state);
    apply_initial_thermodynamic_closure(state);

    seed_tracer_marker(state);
    initialize_nested_grid(global_simulation_storage.nested, state);
    log_initial_field_summary(summarize_initial_fields(state));
}

/**
 * @brief Advects the tracer field.
 */
void advect_tracer(double                                     dt_advect,
                   const tmv::CartesianCGridTransportContext* context,
                   tmv::AdvectionRuntimeState*                runtime_state)
{
    advect_tracer_3d(dt_advect, 0.0, context, runtime_state);
}

/**
 * @brief Advects the thermodynamics.
 */
void advect_thermodynamics(double                                     dt_advect,
                           const tmv::CartesianCGridTransportContext* context,
                           tmv::AdvectionRuntimeState*                runtime_state)
{
    advect_thermodynamics_3d(dt_advect, 0.0, 0.0, context, runtime_state);
}
