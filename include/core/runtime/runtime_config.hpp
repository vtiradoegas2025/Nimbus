#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>

#include "core/infra/coordinate_system.hpp"
#include "core/infra/domain_motion.hpp"
#include "core/output/qualified_radar_zfp_scope.hpp"
#include "diagnostics/validation_policy.hpp"
#include "soundings/soundings_base.hpp"
#include "init/config/config_v1.hpp"
#include "init/hodograph/hodograph_source_config.hpp"
#include "init/hodograph/wind_profile.hpp"
#include "init/sounding/sounding_source_config.hpp"
#include "init/trigger/trigger_source_config.hpp"
#include "microphysics/config.hpp"
#include "util/log_profile.hpp"

struct RuntimeScienceAdmission
{
    bool domain_recycling = false;
    bool earth_fixed_terrain_recycling = false;
    bool earth_fixed_terrain_motion = false;
    bool nested_grid_recycling = false;
    bool moving_nest = false;
    bool two_way_nested = false;
    bool two_way_accelerator_reference = false;
    bool active_terrain_nested = false;
    bool mapped_terrain = false;
    bool native_mapped_terrain = false;
    bool lin_rrtmg = false;
    bool lin_storm = false;
    bool thompson_cpu = false;
    bool milbrandt_reference           = false;
    bool thompson_native               = false;
    bool cloud_top_radiative_ysu = false;
    bool radar = false;
    bool user_filter_width = false;
    bool dynamic_smagorinsky = false;
    bool static_vertical_only_sgs_cpu = false;
    bool richardson_smagorinsky = false;
    bool tke_stability = false;
    bool native_implicit_diffusion = false;
    bool native_weno5                  = false;
    bool native_dry_tvd = false;
    bool generated_metal = false;
    bool generated_metal_east_npy = false;
    bool generated_metal_mixed_npy = false;
    bool coupled_rk = false;
    bool variable_diffusion = false;
    bool agrid_core = false;
    bool simple_grey_absorbing = false;
};

inline constexpr const char* kQualifiedMetalScopeError =
    "qualified Metal execution is limited to exact public moving-nest, domain-recycling, "
    "Lin/Thompson/YSU/RRTMG, variable-diffusion, complete A-grid, native RK, mapped-terrain, "
    "and versioned dry C-grid canonical profiles";

inline bool admits_qualified_metal(const RuntimeScienceAdmission& admission) noexcept
{
    return admission.domain_recycling || admission.moving_nest || admission.lin_rrtmg || admission.lin_storm ||
           admission.thompson_native || admission.milbrandt_reference ||
           admission.variable_diffusion || admission.tke_stability ||
           admission.native_implicit_diffusion || admission.native_weno5 ||
           admission.two_way_accelerator_reference || admission.coupled_rk ||
           admission.agrid_core || admission.mapped_terrain || admission.generated_metal;
}

RuntimeScienceAdmission evaluate_runtime_science_admission(const tmv::init::config::ConfigV1& config);
bool is_public_qualified_generated_metal_profile(const tmv::init::config::ConfigV1& config);
bool is_public_qualified_generated_metal_east_npy_profile(const tmv::init::config::ConfigV1& config);

struct RuntimeConfigurationState
{
    RuntimeScienceAdmission admission;
    tmv::init::config::ExecutionMode execution_mode = tmv::init::config::ExecutionMode::Exploratory;
    WindProfile                      wind_profile{0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    double                           cape_target       = 2500.0;
    double                           sfc_theta_k       = 300.0;
    double                           sfc_qv_kgkg       = 0.014;
    double                           tropopause_z_m    = 12000.0;
    double                           bubble_center_x_m = 50000.0;
    double                           bubble_center_y_m = 0.0;
    double                           bubble_center_z_m = 1500.0;
    double                           bubble_radius_m   = 10000.0;
    double                           bubble_dtheta_k   = 2.0;
    double                           bubble_cloud_water_kgkg = 0.0;

    bool                             sounding_enabled                    = false;
    bool                             sounding_allow_placeholder_profiles = false;
    SoundingConfig                   sounding_config{};
    tmv::init::SoundingSourceConfig  sounding_source{};
    tmv::init::HodographSourceConfig hodograph_source{};
    tmv::init::TriggerSourceConfig   trigger_source{};
    std::string                      microphysics_scheme = "kessler";
    tmv::MicrophysicsConfig          microphysics_config{};
    std::string                      dynamics_scheme_name;

    CoordinateSystem             coordinate_system = CoordinateSystem::Cartesian;
    StaggerType                  stagger_type      = StaggerType::Collocated;
    tmv::init::config::LateralBC lateral_bc_west   = tmv::init::config::LateralBC::ZeroGradient;
    tmv::init::config::LateralBC lateral_bc_east   = tmv::init::config::LateralBC::ZeroGradient;
    tmv::init::config::LateralBC lateral_bc_south  = tmv::init::config::LateralBC::ZeroGradient;
    tmv::init::config::LateralBC lateral_bc_north  = tmv::init::config::LateralBC::ZeroGradient;
    tmv::init::config::BottomBC  bottom_bc         = tmv::init::config::BottomBC::RigidSlip;
    tmv::init::config::TopBC     top_bc            = tmv::init::config::TopBC::Rigid;

    // The broad explicit sponge retains the historical top-30% fallback.
    bool                           damping_enabled      = true;
    double                         damping_z_d          = -1.0;
    double                         damping_alpha        = 0.05;
    int                            lateral_sponge_width = 20;
    double                         lateral_sponge_alpha = 0.05;
    tmv::init::config::DampingMode damping_mode         = tmv::init::config::DampingMode::Explicit;
    // The narrow w-only layer absorbs lid-generated vertical motion without
    // relaxing the balanced horizontal vortex.
    bool   w_only_top_sponge      = true;
    int    w_only_depth_cells     = 8;
    double w_only_alpha_max       = 0.20;
    bool   lateral_sponge_enabled = true;

    bool   slow_rk2               = true;
    double lid_bc_gamma           = -1.0;
    bool   axis_u_m1_filter       = false;
    bool   gpu_apply_microphysics = true;
    bool   gpu_apply_slow         = false;
    bool   gpu_apply_sgs          = false;

    bool                            coriolis_enabled      = true;
    tmv::init::config::CoriolisMode coriolis_mode         = tmv::init::config::CoriolisMode::FPlane;
    double                          coriolis_latitude_deg = 35.0;

    LogProfile                          log_profile                        = LogProfile::normal;
    bool                                perf_timing_enabled                = false;
    int                                 perf_report_every_steps            = 0;
    int                                 perf_warmup_steps                  = 0;
    int                                 scientific_diagnostics_every_steps = -1;
    double                              configured_dt_large_s              = 0.1;
    tmv::init::config::DomainMotionMode domain_motion_mode = tmv::init::config::DomainMotionMode::Stationary;
    tmv::domain::HorizontalDomainFrame  domain_frame{};
    tmv::init::config::DomainMotionSection::Tracking storm_tracking{};
    tmv::init::config::DomainMotionSection::Nesting nesting{};
    tmv::ValidationPolicy                            validation_policy{};
    std::string                                      validation_report_path;
    int                                              checkpoint_every_s = 0;
    int                                              checkpoint_keep_last = 0;
    int                                              checkpoint_minimum_free_space_mib = 0;
    std::string                                      checkpoint_directory;
    std::string                                      checkpoint_resume_from;
};

extern bool&                             global_sounding_enabled;
extern tmv::init::config::ExecutionMode& global_execution_mode;
extern bool&                             global_sounding_allow_placeholder_profiles;
extern SoundingConfig&                   global_runtime_sounding_config;

/**
 * @brief Configured initial-condition sounding source.
 *
 * Populated from `environment.sounding.*` YAML keys in load_config().
 * Read by `equations.cpp::initialize()` to construct the SoundingSource
 * that builds the base-state column. Defaults to ParametricCAPE so YAML
 * configs that omit `environment.sounding.type` get the historical
 * procedural base state.
 */
extern tmv::init::SoundingSourceConfig& global_sounding_source_config;

/**
 * @brief Configured initial-condition hodograph source.
 *
 * Populated from `environment.hodograph.*` YAML keys. Default Type::Auto
 * means "use sounding winds when the SoundingSource provides them, fall
 * back to WK 3-point parametric otherwise" — preserves today's behavior
 * for both legacy parametric configs and SHARPY file overlays.
 */
extern tmv::init::HodographSourceConfig& global_hodograph_source_config;

/**
 * @brief Configured initial-condition trigger source.
 *
 * Populated from `trigger.type` (and the legacy trigger.bubble.* knobs
 * when type=warm_bubble). Default Type::WarmBubble preserves today's
 * behavior; type=none replaces the legacy `dtheta_k = 0` workaround.
 */
extern tmv::init::TriggerSourceConfig& global_trigger_source_config;

/** Immutable horizontal frame contract for the current run. Stored winds are
 * Earth-relative; only explicitly frame-aware transport may consume the
 * grid-relative velocity derived from this object. */
extern tmv::init::config::DomainMotionMode& global_domain_motion_mode;
extern tmv::domain::HorizontalDomainFrame&  global_domain_frame;

extern std::string&             global_microphysics_scheme;
extern tmv::MicrophysicsConfig& global_microphysics_config;
extern std::string&             global_dynamics_scheme_name;
extern tmv::ValidationPolicy&   global_validation_policy;
extern std::string&             global_validation_report_path;
extern int&                     global_checkpoint_every_s;
extern int&                     global_checkpoint_keep_last;
extern int&                     global_checkpoint_minimum_free_space_mib;
extern std::string&             global_checkpoint_directory;
extern std::string&             global_checkpoint_resume_from;

extern double& global_cape_target;
extern double& global_sfc_theta_k;
extern double& global_sfc_qv_kgkg;
extern double& global_tropopause_z_m;
extern double& global_bubble_center_x_m;
extern double& global_bubble_center_y_m;
extern double& global_bubble_center_z_m;
extern double& global_bubble_radius_m;
extern double& global_bubble_dtheta_k;
extern double& global_bubble_cloud_water_kgkg;

extern bool& global_perf_timing_enabled;
extern int&  global_perf_report_every_steps;
extern int&  global_perf_warmup_steps;
// -1 preserves the output/live/debug policy; 0 disables; positive values are
// a one-based step cadence.
extern int&                                              global_scientific_diagnostics_every_steps;
extern double&                                           global_configured_dt_large_s;
extern tmv::init::config::DomainMotionSection::Tracking& global_storm_tracking_config;

/**
 * @brief Active horizontal coordinate system for the run.
 *
 * Cartesian is the only supported value; the global is retained so the
 * `coordinate_system` YAML key and the grid-metric dispatch helpers keep a
 * stable call surface. Set from the top-level `coordinate_system` YAML key in
 * load_config().
 */
extern CoordinateSystem& global_coordinate_system;

/**
 * @brief Active grid staggering arrangement for the run.
 *
 * Defaults to StaggerType::Collocated so existing configurations are
 * unchanged. Set from the `grid.staggering` YAML key in load_config().
 * Used by the dynamics factory to select collocated vs C-grid scheme
 * variants (Phase C of the Coordinate Backend Plan).
 */
extern StaggerType& global_stagger_type;

/**
 * @brief Per-face lateral boundary condition selection (Phase C1).
 *
 * Populated from v1 `boundary_conditions.lateral.{west,east,south,north}`
 * YAML keys. v0 configs (no schema stamp) leave these at the default
 * (Open / zero-gradient) so existing tests preserve their behavior.
 *
 * Read by Cartesian BC dispatch (`boundary_conditions_cartesian.cpp`)
 * to choose periodic / open / rigid / rigid-slip per face.
 */
extern tmv::init::config::LateralBC& global_lateral_bc_west;
extern tmv::init::config::LateralBC& global_lateral_bc_east;
extern tmv::init::config::LateralBC& global_lateral_bc_south;
extern tmv::init::config::LateralBC& global_lateral_bc_north;

/**
 * @brief Vertical boundary condition selection (Phase C2).
 *
 * Populated from v1 `boundary_conditions.{bottom,top}` YAML keys.
 *
 * Defaults are Nimbus' historical Cartesian behavior — `RigidSlip` at the
 * bottom (free-slip horizontal: w=0, u/v zero-gradient) and `Rigid` at the
 * top (w=0, u/v zero-gradient, no sponge). This is a deviation from CM1's
 * default (`RigidNoSlip` + `RigidWithSponge`); the v1 schema in §20 keeps
 * the CM1 defaults at the YAML level, but the migrator (`v0_to_v1.cpp`)
 * explicitly emits the legacy Nimbus values when translating v0 configs
 * so existing test behavior is preserved. A v0 config (no schema stamp)
 * also leaves these globals at the legacy defaults.
 *
 * Read by Cartesian BC dispatch in `apply_vertical_bcs_momentum`.
 *
 * `TopBC::RigidWithSponge` produces the same BC behavior as `Rigid` at
 * the C2 layer; the sponge itself is the C3 (Rayleigh damping) deliverable.
 */
extern tmv::init::config::BottomBC& global_bottom_bc;
extern tmv::init::config::TopBC&    global_top_bc;

/**
 * @brief Upper-boundary Rayleigh damping (sponge) configuration (Phase C3).
 *
 * Populated from v1 `damping.{enabled, z_d, alpha, mode}` YAML keys. v0
 * configs (no schema stamp) leave these at runtime-computed defaults that
 * preserve the legacy depth-fraction (top 30% of NZ*dz) behavior — see
 * load_config for the v0 fallback.
 *
 * Read by `core/infra/rayleigh_damping.cpp::ensure_initialized()`:
 *   - `global_damping_enabled = false` short-circuits the entire sponge
 *   - `global_damping_z_d`     is the height (m) above which the sponge
 *                              becomes active; `z_d >= NZ*dz` disables it
 *   - `global_damping_alpha`   is the peak damping rate (1/s)
 *   - `global_damping_mode`    selects the established exact-exponential
 *                              writer or backward-Euler relaxation
 *
 * The sponge is the primary absorber for the lid-BC residual in cgrid
 * runs — `[[scientific-fidelity-status]]` notes alpha=0.05 is load-bearing
 * for `cgrid_tornado_genesis`. Lowering alpha (or disabling) below the
 * load-bearing point destabilises the run within ~200 steps.
 */
extern bool&   global_damping_enabled;
extern double& global_damping_z_d;
extern double& global_damping_alpha;
/// Lateral sponge band (SCIENCE.md M18): grid points per face and
/// peak rate at the face. Width 0 or alpha 0 disables the band.
extern int&                            global_lateral_sponge_width;
extern double&                         global_lateral_sponge_alpha;
extern tmv::init::config::DampingMode& global_damping_mode;

/**
 * @brief w-only upper Rayleigh sponge configuration (Smell #8 closure).
 *
 * When `global_w_only_top_sponge` is true, `apply_rayleigh_damping` replaces
 * its legacy all-field damping with a narrow w-only layer in the top
 * `global_w_only_depth_cells` cells. Only the vertical velocity is relaxed
 * (toward 0); u, v, theta, rho, p are left untouched.
 *
 * Motivation: the legacy all-field top sponge at alpha=0.05 was load-bearing
 * for cgrid_tornado_genesis because it absorbed the lid-BC O(alpha^3)
 * residual that drives a steady secondary circulation. The same damping
 * also relaxes the LEGITIMATE cyclostrophic v(r) perturbation toward the
 * environmental wind, breaking cyclostrophic balance in the upper
 * troposphere. A w-only sponge targets the residual mechanism (upward
 * w-signal from the lid) without distorting horizontal balance --
 * standard CM1/WRF treatment for rigid-lid tornado-genesis simulations.
 *
 * Defaults preserve legacy behavior: off by default until verified across
 * cgrid_tornado_genesis, cgrid_hydrostatic, and the full integration suite.
 * Set via the damping.w_only_top_sponge config key.
 *
 * See [[reference_cyclostrophic_plan]] §5b option (b) and
 * docs/CyclostrophicConservativePlan.md §8 (Phase 5b future-session path).
 */
extern bool&   global_w_only_top_sponge;
extern int&    global_w_only_depth_cells;
extern double& global_w_only_alpha_max;

/// Master on/off for the lateral sponge band (modularity cleanup 6a: was the
/// hidden NIMBUS_DISABLE_LATERAL_SPONGE env knob). Populated from
/// damping.lateral_sponge_enabled; default true preserves the legacy band.
extern bool& global_lateral_sponge_enabled;

/**
 * @brief Solver internals + GPU pointwise-apply toggles promoted from the
 *        hidden NIMBUS_* env knobs (modularity cleanup 6a).
 *
 * Populated from the v1 `extensions.{solver,compute}` blocks; v0 configs lack
 * those keys and keep the defaults below, which reproduce the prior env-unset
 * runtime exactly. The experimental toggles (lid gamma, m=1 filter) stay off
 * by default.
 */
extern bool&   global_slow_rk2;
extern double& global_lid_bc_gamma;
extern bool&   global_axis_u_m1_filter;
extern bool&   global_gpu_apply_microphysics;
extern bool&   global_gpu_apply_slow;
extern bool&   global_gpu_apply_sgs;

/**
 * @brief Coriolis configuration (Phase C4).
 *
 * Populated from v1 `coriolis.{enabled, mode, latitude_deg}` YAML keys.
 *
 * Read by `equations.cpp::initialize()` to compute the f-plane Coriolis
 * parameter `coriolis_f = 2 * Omega * sin(latitude_rad)` when enabled, or
 * to leave `coriolis_f = 0` when disabled. The cartesian dynamics scheme
 * applies `coriolis_f * (uy - v0_base[k])` and `-coriolis_f * (ux -
 * u0_base[k])` in its momentum tendency unconditionally — gating happens
 * by setting `coriolis_f = 0`.
 *
 * Defaults preserve legacy behavior: enabled=true, latitude=35°N (Weisman
 * & Klemp 1982 standard mid-latitude supercell), mode=f_plane. The v1
 * schema CM1 default is `enabled=false` (CM1 &param12:icor=0); the
 * migrator overrides to enabled=true so re-migrated v0 configs preserve
 * the legacy 35°N Coriolis they had pre-C4.
 *
 * Beta-plane uses an Earth-referenced first-order tangent-plane profile and
 * follows constant-translation/recycling origins. Each nested level derives
 * its grid-origin value from its own Earth frame, including after an integer
 * child-window relocation. Production never substitutes f-plane behavior.
 */
extern bool&                            global_coriolis_enabled;
extern tmv::init::config::CoriolisMode& global_coriolis_mode;
extern double&                          global_coriolis_latitude_deg;

/**
 * @brief Returns a label for the configured sounding interpolation method.
 * @param method Numeric interpolation selector.
 * @return Method name string.
 */
const char* sounding_interpolation_method_name(double method);

/**
 * @brief Converts temperature and pressure to potential temperature.
 * @param temperature_k Air temperature in kelvin.
 * @param pressure_hpa Pressure in hectopascals.
 * @return Potential temperature in kelvin.
 */
double potential_temperature_from_temperature_pressure(double temperature_k, double pressure_hpa);

/**
 * @brief Returns a string label for a log profile.
 * @param profile Log profile enum value.
 * @return Profile name string.
 */
const char* log_profile_name(LogProfile profile);

/**
 * @brief Parses a log profile string.
 * @param value Input profile string.
 * @param valid Optional parse-success output flag.
 * @return Parsed log profile.
 */
LogProfile parse_log_profile(const std::string& value, bool* valid = nullptr);

/**
 * @brief Escapes a string for JSON output.
 * @param value Input string.
 * @return Escaped JSON string.
 */
std::string json_escape_local(const std::string& value);

struct OutputConfig;

nimbus::output::QualifiedRadarZfpV4Context
make_qualified_radar_zfp_v4_context(
    const tmv::init::config::ConfigV1& config,
    const OutputConfig& output,
    double write_every_s,
    bool live_shm,
    bool trajectories_enabled);

struct StationaryNestingLiveExecution
{
    bool        initialized        = false;
    std::string active_backend;
    bool        fallback_active    = false;
    bool        terrain_ready = false;
    bool        native_dry_cgrid_ready = false;
    bool        native_smagorinsky_ready = false;
    bool        native_kessler_ready = false;
};

/**
 * Validate the effective stationary one-way nesting execution profile.
 * A null live state checks the requested pre-allocation profile. A live state
 * additionally verifies the realized backend and flat terrain runtime.
 */
void validate_stationary_nesting_execution(
    const StationaryNestingLiveExecution* live = nullptr);

bool exact_two_way_nested_profile_selected();
bool exact_native_mapped_terrain_profile_selected();

/**
 * Enable the internal CPU C-grid Milbrandt qualification exception.
 * Ordinary simulator builds never call this function.
 */
void set_internal_milbrandt_qualification(bool enabled);

/**
 * Enable the internal exploratory moving-nest campaign path. Ordinary
 * simulator builds never call this function; public admission is independently
 * restricted to the exact qualified fallback-off Vulkan profile. The
 * internal exploratory lane may additionally exercise the same ownership
 * tuple on native Metal without broadening public admission.
 */
void set_internal_moving_nest_qualification(bool enabled);

/**
 * @brief Loads runtime configuration from disk.
 * @param config_path Path to configuration file.
 * @param duration_s Output simulation duration in seconds.
 * @param write_every_s Output cadence in seconds.
 * @param outdir Output directory path.
 * @param output_config Optional output pipeline configuration.
 * @param write_every_override Effective CLI cadence, or -1 when absent.
 * @param live_shm_override Effective CLI live-publishing flag, or -1 when absent.
 * @param live_shm_fields_override Effective CLI live field list, or null when absent.
 * @param trajectories_enabled Whether a CLI trajectory override can consume
 *        gridded vorticity diagnostics without file/live output. YAML
 *        output.trajectories is combined with this before memory admission.
 * @param compute_backend_override Effective canonical CLI backend, or null
 *        when the configuration owns backend selection. The override is
 *        injected before memory admission and science-profile validation.
 * @param sounding_file_override Effective CLI file-sounding path, or null
 *        when the configuration owns its sounding source. File-reader and
 *        fail-closed fallback policy are projected before science validation.
 * @param duration_override Effective CLI run duration, or -1 when the
 *        configuration owns duration. Applied before lifetime and memory
 *        preflight.
 */
void load_config(const std::string&              config_path,
                 int&                            duration_s,
                 int&                            write_every_s,
                 std::string&                    outdir,
                 OutputConfig*                   output_config            = nullptr,
                 bool*                           live_shm                 = nullptr,
                 std::vector<std::string>*       live_shm_fields          = nullptr,
                 int                             write_every_override     = -1,
                 int                             live_shm_override        = -1,
                 const std::vector<std::string>* live_shm_fields_override = nullptr,
                 bool                            trajectories_enabled     = false,
                 const std::string*              compute_backend_override = nullptr,
                 const std::string*              sounding_file_override   = nullptr,
                 int                             duration_override        = -1);
