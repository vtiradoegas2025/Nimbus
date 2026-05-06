#pragma once

#include <algorithm>
#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include "core/coordinate_system.hpp"
#include "core/field3d.hpp"
#include "core/grid_geometry.hpp"
#include "core/physical_constants.hpp"

/**
 * @file simulation.hpp
 * @brief Central runtime API and shared state declarations for the model.
 *
 * Exposes global grid/state variables and subsystem entry points used by
 * the main integration loop.
 * This header is the integration surface connecting dynamics, physics,
 * numerics, diagnostics, and optional GUI execution.
 */

class DynamicsScheme;
class AdvectionSchemeBase;
class DiffusionSchemeBase;
class TimeSteppingSchemeBase;

struct RadiationConfig;
struct BoundaryLayerConfig;
struct SurfaceConfig;
struct TurbulenceConfig;
struct TurbulenceTendencies;
struct GridMetrics;
struct AdvectionConfig;
struct DiffusionConfig;
struct TimeSteppingConfig;
struct TerrainConfig;

namespace chaos
{
struct ChaosConfig;
struct ChaosDiagnostics;
}

extern int NR;
extern int NTH;
extern int NZ;

extern double dr;
extern double dz;
extern double dt;
extern double dtheta;

extern GridGeometry global_grid_geometry;

inline constexpr double gamma = physical_constants::dry_air_gamma;
inline constexpr double theta0 = physical_constants::theta_reference_k;
inline constexpr float theta_min_k = 200.0f;
inline constexpr float theta_max_k = 700.0f;  // Stratospheric theta easily exceeds 500K
inline constexpr float pressure_min_pa = 1000.0f;
inline constexpr float pressure_max_pa = 200000.0f;
inline constexpr float density_min_kgm3 = 0.05f;
inline constexpr float density_max_kgm3 = 2.0f;
inline constexpr float wind_horizontal_abs_max_ms = 180.0f;
inline constexpr float wind_vertical_abs_max_ms = 120.0f;
inline constexpr float qv_max_kgkg = 0.05f;
inline constexpr float hydrometeor_max_kgkg = 0.05f;

enum class LogProfile : int
{
    quiet = 0,
    normal = 1,
    debug = 2
};

extern LogProfile global_log_profile;

/**
 * @brief Returns whether current logging level includes the target level.
 * @param level Minimum desired logging level.
 * @return True when logging at the requested level is enabled.
 */
inline bool log_at_least(LogProfile level)
{
    return static_cast<int>(global_log_profile) >= static_cast<int>(level);
}

/**
 * @brief Returns whether normal logging output is enabled.
 */
inline bool log_normal_enabled()
{
    return log_at_least(LogProfile::normal);
}

/**
 * @brief Returns whether debug logging output is enabled.
 */
inline bool log_debug_enabled()
{
    return log_at_least(LogProfile::debug);
}

extern bool global_perf_timing_enabled;
extern int global_perf_report_every_steps;

/**
 * @brief Sanitizes a scalar value: replaces non-finite with a fallback, then clamps.
 * @param value   Candidate scalar.
 * @param fallback Value to substitute when input is NaN or Inf.
 * @param lo      Lower bound for clamping.
 * @param hi      Upper bound for clamping.
 * @return Sanitized and clamped value.
 *
 * Every field clamp function below delegates to this so that NaN/Inf never
 * propagates silently through any prognostic variable.
 */
inline float sanitize_value(float value, float fallback, float lo, float hi)
{
    if (!std::isfinite(value))
    {
        return fallback;
    }
    return std::clamp(value, lo, hi);
}

/**
 * @brief Clamps potential temperature into physically allowed bounds.
 * Non-finite input is replaced with the reference potential temperature.
 */
inline float clamp_theta_k(float value)
{
    return sanitize_value(value, static_cast<float>(theta0), theta_min_k, theta_max_k);
}

/**
 * @brief Clamps pressure into physically allowed bounds.
 * Non-finite input is replaced with standard sea-level pressure (101325 Pa).
 */
inline float clamp_pressure_pa(float value)
{
    return sanitize_value(value, 101325.0f, pressure_min_pa, pressure_max_pa);
}

/**
 * @brief Sanitizes non-finite values and enforces non-negativity.
 * @param value Candidate scalar.
 * @return Zero for non-finite input, otherwise max(value, 0).
 */
inline float clamp_non_negative(float value)
{
    if (!std::isfinite(value))
    {
        return 0.0f;
    }
    return std::max(0.0f, value);
}

/**
 * @brief Clamps density into physically allowed bounds.
 * Non-finite input is replaced with a standard tropospheric density (1.0 kg/m^3).
 */
inline float clamp_density_kgm3(float value)
{
    return sanitize_value(value, 1.0f, density_min_kgm3, density_max_kgm3);
}

/**
 * @brief Clamps horizontal wind speed magnitude.
 * Non-finite input is replaced with zero (calm).
 */
inline float clamp_wind_horizontal_ms(float value)
{
    return sanitize_value(value, 0.0f, -wind_horizontal_abs_max_ms, wind_horizontal_abs_max_ms);
}

/**
 * @brief Clamps vertical wind speed magnitude.
 * Non-finite input is replaced with zero (calm).
 */
inline float clamp_wind_vertical_ms(float value)
{
    return sanitize_value(value, 0.0f, -wind_vertical_abs_max_ms, wind_vertical_abs_max_ms);
}

/**
 * @brief Clamps water-vapor mixing ratio and sanitizes non-finite input.
 * Non-finite input is replaced with zero.
 */
inline float clamp_qv_kgkg(float value)
{
    return sanitize_value(value, 0.0f, 0.0f, qv_max_kgkg);
}

/**
 * @brief Clamps hydrometeor mixing ratio and sanitizes non-finite input.
 * Non-finite input is replaced with zero.
 */
inline float clamp_hydrometeor_kgkg(float value)
{
    return sanitize_value(value, 0.0f, 0.0f, hydrometeor_max_kgkg);
}

extern std::vector<double> rho0_base;
extern std::vector<double> p0_base;
extern std::vector<double> qv0_base;

// Base-state wind profiles for perturbation Coriolis (Rotunno & Klemp 1982).
// Stored at initialization so Coriolis is applied to (u-u0, v-v0) rather than
// the full wind, avoiding the "invented forces" problem (Davies-Jones 2021).
extern std::vector<double> u0_base;
extern std::vector<double> v0_base;

// Coriolis parameter f = 2*Omega*sin(lat). Set from the runtime config
// latitude; defaults to 35N (f ~ 8.39e-5 /s), a mid-latitude supercell value.
extern double coriolis_f;

inline constexpr double qc0 = 1.0e-3;
inline constexpr double c_auto = 1.0e-3;
inline constexpr double c_accr = 2.2;
inline constexpr double c_evap = 3.0e-3;
inline constexpr double L_v = physical_constants::latent_heat_vaporization_jkg;
inline constexpr double a_term = 65.0;
inline constexpr double b_term = 0.125;
inline constexpr double Vt_max = 20.0;

inline constexpr double T0 = physical_constants::freezing_temperature_k;
inline constexpr double L_f = physical_constants::latent_heat_fusion_jkg;
inline constexpr double L_s = physical_constants::latent_heat_sublimation_jkg;
inline constexpr double c_rime = 1.0;
inline constexpr double c_freeze = 1.0e-3;
inline constexpr double c_melt = 1.0e-3;
inline constexpr double c_subl = 1.0e-3;
inline constexpr double a_hail = 114.0;
inline constexpr double b_hail = 0.5;
inline constexpr double a_grau = 40.0;
inline constexpr double b_grau = 0.37;
inline constexpr double Vt_max_hail = 40.0;
inline constexpr double Vt_max_grau = 15.0;

inline constexpr double sigma_sb = physical_constants::stefan_boltzmann_wm2k4;
inline constexpr double S0_solar = physical_constants::solar_constant_wm2;

extern double simulation_time;

extern Field3D rho;
extern Field3D p;
extern Field3D u;
extern Field3D w;
extern Field3D v;
extern Field3D tracer;

extern Field3D theta;
extern Field3D qv;
extern Field3D qc;
extern Field3D qr;
extern Field3D qi;
extern Field3D qs;
extern Field3D qh;
extern Field3D qg;

struct WindProfile
{
    double u_sfc;
    double v_sfc;
    double u_1km;
    double v_1km;
    double u_6km;
    double v_6km;
};

extern WindProfile global_wind_profile;
extern double global_cape_target;
extern double global_sfc_theta_k;
extern double global_sfc_qv_kgkg;
extern double global_tropopause_z_m;

extern double global_bubble_center_x_m;
// `global_bubble_center_y_m` is only consumed by the Cartesian initialize()
// path (Phase A.4 of the Coordinate Backend Plan). The cylindrical path
// historically reuses `global_bubble_center_x_m` as a radial distance from
// the singular axis and ignores y entirely; that behavior is preserved.
extern double global_bubble_center_y_m;
extern double global_bubble_center_z_m;
extern double global_bubble_radius_m;
extern double global_bubble_dtheta_k;

struct NestedGridConfig
{
    bool enabled = false;
    int center_i = NR / 2;
    int center_j = NTH / 2;
    int center_k = NZ / 2;
    int nest_size_r = 32;
    int nest_size_th = 32;
    int nest_size_z = 32;
    double refinement = 3.0;
};

extern NestedGridConfig nested_config;

extern Field3D nest_rho;
extern Field3D nest_p;
extern Field3D nest_u;
extern Field3D nest_w;
extern Field3D nest_v;
extern Field3D nest_theta;
extern Field3D nest_qv;
extern Field3D nest_qc;
extern Field3D nest_qr;
extern Field3D nest_qh;
extern Field3D nest_qg;
extern Field3D nest_tracer;

extern Field3D radar_reflectivity;

extern std::unique_ptr<DynamicsScheme> dynamics_scheme;

extern Field3D vorticity_r;
extern Field3D vorticity_theta;
extern Field3D vorticity_z;
extern Field3D stretching_term;
extern Field3D tilting_term;
extern Field3D baroclinic_term;

extern Field3D angular_momentum;
extern Field3D angular_momentum_tendency;

extern Field3D p_prime;
extern Field3D dynamic_pressure;
extern Field3D buoyancy_pressure;

extern Field3D dtheta_dt_rad;

extern Field3D dtheta_dt_pbl;
extern Field3D dqv_dt_pbl;
extern Field3D du_dt_pbl;
extern Field3D dv_dt_pbl;
extern Field3D dtke_dt_pbl;

extern Field3D tke;
extern RadiationConfig global_radiation_config;

extern BoundaryLayerConfig global_boundary_layer_config;
extern SurfaceConfig global_surface_config;

extern TurbulenceConfig global_turbulence_config;

extern GridMetrics global_grid_metrics;
extern AdvectionConfig global_advection_config;
extern DiffusionConfig global_diffusion_config;
extern TimeSteppingConfig global_time_stepping_config;
extern std::unique_ptr<AdvectionSchemeBase> advection_scheme;
extern std::unique_ptr<DiffusionSchemeBase> diffusion_scheme;
extern std::unique_ptr<TimeSteppingSchemeBase> time_stepping_scheme;

extern chaos::ChaosConfig global_chaos_config;
extern TerrainConfig global_terrain_config;

// ---------------------------------------------------------------------------
// Grouped configuration structs
//
// These aggregate related globals so that new code can pass around a single
// reference instead of reaching for individual globals.  Existing extern
// variables remain valid — they are *not* aliases but the canonical storage.
// Migration of call sites is incremental across later phases.
// ---------------------------------------------------------------------------

/**
 * @brief Groups grid parameters and environment configuration that are set
 *        once at startup and read throughout the simulation.
 *
 * Members mirror the corresponding globals declared above.  The canonical
 * global instance is `global_sim_config`, defined in tornado_sim.cpp.
 */
struct SimulationConfig
{
    // Grid dimensions
    int NR = 0;
    int NTH = 0;
    int NZ = 0;
    double dr = 0.0;
    double dz = 0.0;
    double dt = 0.0;
    double dtheta = 0.0;

    // Environment / sounding
    WindProfile wind_profile{};
    double cape_target = 0.0;
    double sfc_theta_k = 0.0;
    double sfc_qv_kgkg = 0.0;
    double tropopause_z_m = 0.0;

    // Warm bubble
    double bubble_center_x_m = 0.0;
    double bubble_center_z_m = 0.0;
    double bubble_radius_m = 0.0;
    double bubble_dtheta_k = 0.0;

    // Runtime behaviour
    LogProfile log_profile = LogProfile::normal;
    bool perf_timing_enabled = false;
    int perf_report_every_steps = 100;
};

/**
 * @brief Groups the owning pointers for pluggable physics / numerics schemes.
 *
 * Consolidates the four scheme unique_ptrs that are currently individual
 * globals.  The canonical global instance is `global_scheme_registry`,
 * defined in tornado_sim.cpp.
 */
struct SchemeRegistry
{
    std::unique_ptr<DynamicsScheme> dynamics;
    std::unique_ptr<AdvectionSchemeBase> advection;
    std::unique_ptr<DiffusionSchemeBase> diffusion;
    std::unique_ptr<TimeSteppingSchemeBase> time_stepping;

    SchemeRegistry() = default;

    // Destructor and move ops must be defined in a TU where the pointed-to
    // types are complete (equations.cpp).
    ~SchemeRegistry();
    SchemeRegistry(SchemeRegistry&&) noexcept;
    SchemeRegistry& operator=(SchemeRegistry&&) noexcept;
};

extern SimulationConfig global_sim_config;
extern SchemeRegistry global_scheme_registry;

/**
 * @brief Initializes global simulation state and subsystem defaults.
 */
void initialize();

/**
 * @brief Resizes global fields to current grid dimensions.
 */
void resize_fields();

/**
 * @brief Advances passive tracer by advection for one step.
 * @param dt_advect Advection step in seconds.
 */
void advect_tracer(double dt_advect);

/**
 * @brief Advances thermodynamic tracers by advection for one step.
 * @param dt_advect Advection step in seconds.
 */
void advect_thermodynamics(double dt_advect);

/**
 * @brief Applies microphysics tendencies for one step.
 * @param dt_micro Microphysics step in seconds.
 */
void step_microphysics(double dt_micro);

/**
 * @brief Applies model boundary conditions.
 */
void apply_boundary_conditions();

/**
 * @brief Computes radar reflectivity diagnostic fields.
 */
void calculate_radar_reflectivity();

/**
 * @brief Initializes the microphysics subsystem.
 * @param scheme_name Microphysics scheme identifier.
 */
void initialize_microphysics(const std::string& scheme_name = "kessler");

/**
 * @brief Initializes the dynamics subsystem.
 * @param scheme_name Dynamics scheme identifier.
 */
void initialize_dynamics(const std::string& scheme_name = "tornado");

/**
 * @brief Advances dynamics using the active integration path.
 * @param current_time Current simulation time in seconds.
 */
void step_dynamics(double current_time = 0.0);

/**
 * @brief Advances dynamics using legacy dynamics implementation.
 * @param current_time Current simulation time in seconds.
 */
void step_dynamics_old(double current_time = 0.0);

/**
 * @brief Advances dynamics using the modular dynamics implementation.
 * @param dt_dynamics Dynamics step length in seconds.
 * @param current_time Current simulation time in seconds.
 */
void step_dynamics_new(double dt_dynamics, double current_time = 0.0);

/**
 * @brief Computes dynamics diagnostic fields.
 */
void compute_dynamics_diagnostics();

/**
 * @brief Applies radiation tendencies for the current time.
 * @param current_time Current simulation time in seconds.
 */
void step_radiation(double current_time);

/**
 * @brief Initializes the boundary-layer subsystem.
 * @param scheme_name Boundary-layer scheme identifier.
 * @param cfg Boundary-layer configuration.
 * @param sfc Surface-layer configuration.
 */
void initialize_boundary_layer(const std::string& scheme_name,
                               const BoundaryLayerConfig& cfg,
                               const SurfaceConfig& sfc);

/**
 * @brief Steps boundary-layer physics for the current time.
 * @param current_time Current simulation time in seconds.
 */
void step_boundary_layer(double current_time);

/**
 * @brief Reports whether boundary-layer physics updated this step.
 * @return True if boundary-layer tendencies were applied.
 */
bool boundary_layer_updated_this_step();

/**
 * @brief Initializes the turbulence subsystem.
 * @param scheme_name Turbulence scheme identifier.
 * @param cfg Turbulence configuration.
 */
void initialize_turbulence(const std::string& scheme_name, const TurbulenceConfig& cfg);

/**
 * @brief Steps turbulence tendencies for the current time.
 * @param current_time Current simulation time in seconds.
 * @param tendencies Output turbulence tendencies.
 */
void step_turbulence(double current_time, TurbulenceTendencies& tendencies);

/**
 * @brief Initializes the chaos subsystem.
 * @param cfg Chaos configuration.
 */
void initialize_chaos(const chaos::ChaosConfig& cfg);

/**
 * @brief Applies configured chaos perturbations to initial conditions.
 */
void apply_chaos_initial_conditions();

/**
 * @brief Applies configured chaos perturbations to tendency pathways.
 */
void apply_chaos_tendencies();

/**
 * @brief Applies chaos multipliers to microphysics tendencies.
 */
void apply_chaos_to_microphysics_tendencies(Field3D& dtheta_dt,
                                            Field3D& dqv_dt,
                                            Field3D& dqc_dt,
                                            Field3D& dqr_dt,
                                            Field3D& dqi_dt,
                                            Field3D& dqs_dt,
                                            Field3D& dqg_dt,
                                            Field3D& dqh_dt);

/**
 * @brief Applies chaos multipliers to turbulence tendencies.
 * @param tendencies Turbulence tendency container.
 */
void apply_chaos_to_turbulence_tendencies(TurbulenceTendencies& tendencies);

/**
 * @brief Advances stochastic chaos noise fields.
 * @param dt Step length in seconds.
 */
void step_chaos_noise(double dt);

/**
 * @brief Returns current chaos diagnostics.
 * @return Immutable reference to chaos diagnostics.
 */
const chaos::ChaosDiagnostics& get_chaos_diagnostics();

/**
 * @brief Resets accumulated chaos diagnostics.
 */
void reset_chaos_diagnostics();

/**
 * @brief Initializes numerical subsystems.
 */
void initialize_numerics();

/**
 * @brief Rebuilds grid metrics from active terrain fields.
 */
void refresh_grid_metrics_from_terrain();

/**
 * @brief Computes a stable runtime timestep recommendation.
 * @return Recommended timestep in seconds.
 */
double choose_runtime_timestep();

/**
 * @brief Builds terrain fields for the current configuration.
 */
void build_terrain_fields();

/**
 * @brief Initializes nested-grid storage and metadata.
 */
void initialize_nested_grid();

/**
 * @brief Feeds nested-grid updates back to the parent grid.
 */
void feedback_to_parent();
