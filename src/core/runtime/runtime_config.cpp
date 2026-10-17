#include "core/runtime/flat_lin_storm_profile.hpp"
#include "configuration_compiler.hpp"
#include "chaos/control_grid_geometry.hpp"
#include "core/runtime/runtime_config.hpp"
#include "core/runtime/generated_metal_scope_v1.hpp"
#include "core/runtime/generated_metal_unserialized_controls.hpp"
#include "core/runtime/flat_mixed_metal_admission.hpp"
#include "core/runtime/generated_metal_east_npy_scope_v1.hpp"
#include "core/runtime/generated_metal_zfp_typed_scope_v1.hpp"
#include "core/runtime/simulation_state.hpp"
#include "core/infra/sha256.hpp"

#include "core/infra/nested_grid.hpp"
#include "core/infra/domain_motion.hpp"
#include "core/infra/grid_coordinates.hpp"
#include "core/infra/physical_cell_domain.hpp"
#include "core/infra/physical_constants.hpp"
#include "core/output/output_config.hpp"
#include "core/output/derived_slice_scratch.hpp"
#include "core/output/velocity_staging.hpp"
#include "core/output/field_registry.hpp"
#include "core/field/field_pool.hpp"
#include "core/runtime/yaml_helpers.hpp"
#include "core/runtime/grid_state.hpp"
#include "core/runtime/qualified_agrid_core_profile.hpp"
#include "core/runtime/qualified_variable_diffusion_profile.hpp"
#include "core/runtime/science_profile.hpp"
#include "diagnostics/field_validation.hpp"
#include "diagnostics/deterministic_reduction.hpp"
#include "diagnostics/storm_tracker.hpp"
#include "init/config/config_v1_serialization_common.hpp"
#include "init/config/qualified_terrain_profile.hpp"
#include "init/config/v0_to_v1.hpp"
#include "init/hodograph/factory.hpp"
#include "init/scheme_profile.hpp"
#include "init/sounding/factory.hpp"
#include "init/trigger/factory.hpp"
#include "microphysics/milbrandt_triple_scope.hpp"
#include "radiation/rrtmg_config_limits.hpp"
#include "radar/runtime.hpp"

#include <algorithm>
#include <cmath>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <unordered_set>
#include <vector>

#include "numerics/advection/advection_base.hpp"
#include "boundary_layer/boundary_layer_base.hpp"
#include "chaos/runtime.hpp"
#include "compute/compute_backend.hpp"
#include "compute/compute_memory_admission.hpp"
#include "compute/compute_kernel_template.hpp"
#include "numerics/diffusion/diffusion_base.hpp"
#include "radiation/radiation_base.hpp"
#include "core/infra/hardware_info.hpp"
#include "core/infra/runtime_memory_profile.hpp"
#include "util/log.hpp"
#include "util/string_utils.hpp"
#include "terrain/terrain_base.hpp"
#include "numerics/time_stepping/time_stepping_base.hpp"
#include "turbulence/turbulence_base.hpp"
#include "boundary_layer/scheme_descriptor.hpp"
#include "chaos/scheme_descriptor.hpp"
#include "dynamics/scheme_descriptor.hpp"
#include "microphysics/scheme_descriptor.hpp"
#include "numerics/advection/scheme_descriptor.hpp"
#include "numerics/diffusion/scheme_descriptor.hpp"
#include "numerics/time_stepping/scheme_descriptor.hpp"
#include "radiation/scheme_descriptor.hpp"
#include "soundings/scheme_descriptor.hpp"
#include "terrain/scheme_descriptor.hpp"
#include "turbulence/scheme_descriptor.hpp"

/**
 * @brief Reject an explicit invalid science value instead of substituting state.
 *
 * This legacy helper used to warn and retain a previous/default value. That
 * made accepted configurations lie about the experiment actually executed and
 * also allowed state from an earlier in-process load to survive. Keep the
 * call sites centralized, but make every explicit invalid value fatal.
 */
[[noreturn]] void warn_invalid_config_value(const std::string& key, const std::string& value, const char* expected)
{
    throw std::runtime_error("Invalid " + key + " '" + value + "'; expected " + expected);
}

/**
 * @brief Parses sounding interpolation mode from textual input.
 */
bool parse_sounding_interpolation_method(const std::string& value, double& method_out)
{
    const std::string normalized = to_lower_copy(value);
    if (normalized == "linear" || normalized == "0")
    {
        method_out = 0.0;
        return true;
    }
    if (normalized == "spline" || normalized == "1")
    {
        method_out = 1.0;
        return true;
    }
    if (normalized == "log-linear" || normalized == "log_linear" || normalized == "loglinear" || normalized == "2")
    {
        method_out = 2.0;
        return true;
    }

    double parsed = 0.0;
    if (try_parse_double_value(value, parsed) && (parsed == 0.0 || parsed == 1.0 || parsed == 2.0))
    {
        method_out = parsed;
        return true;
    }

    return false;
}

/**
 * @brief Returns human-readable name for sounding interpolation mode.
 */
const char* sounding_interpolation_method_name(double method)
{
    if (method == 1.0)
    {
        return "spline";
    }
    if (method == 2.0)
    {
        return "log-linear";
    }
    return "linear";
}

/**
 * @brief Converts temperature and pressure to potential temperature.
 */
double potential_temperature_from_temperature_pressure(double temperature_k, double pressure_hpa)
{
    if (!std::isfinite(temperature_k) || !std::isfinite(pressure_hpa) || pressure_hpa <= 0.0)
    {
        return std::numeric_limits<double>::quiet_NaN();
    }
    const double pressure_pa = pressure_hpa * 100.0;
    if (pressure_pa <= 0.0)
    {
        return std::numeric_limits<double>::quiet_NaN();
    }
    const double kappa = R_d / cp;
    return temperature_k * std::pow(p0 / pressure_pa, kappa);
}

/**
 * @brief Returns string label for runtime logging profile.
 */
const char* log_profile_name(LogProfile profile)
{
    switch (profile)
    {
    case LogProfile::quiet:
        return "quiet";
    case LogProfile::debug:
        return "debug";
    case LogProfile::normal:
    default:
        return "normal";
    }
}

/**
 * @brief Parses runtime logging profile from text.
 */
LogProfile parse_log_profile(const std::string& value, bool* valid)
{
    const std::string normalized = to_lower_copy(value);
    if (normalized == "quiet")
    {
        if (valid)
            *valid = true;
        return LogProfile::quiet;
    }
    if (normalized == "normal")
    {
        if (valid)
            *valid = true;
        return LogProfile::normal;
    }
    if (normalized == "debug")
    {
        if (valid)
            *valid = true;
        return LogProfile::debug;
    }

    if (valid)
        *valid = false;
    return LogProfile::normal;
}

/**
 * @brief Parses boundary-layer surface-layer identifier aliases.
 */
bool parse_boundary_layer_surface_layer_id(const std::string& value, std::string& id_out)
{
    const std::string normalized = to_lower_copy(value);
    if (normalized == "bulk")
    {
        id_out = "bulk";
        return true;
    }
    if (normalized == "monin_obukhov" || normalized == "monin-obukhov" || normalized == "moninobukhov" ||
        normalized == "most" || normalized == "mo")
    {
        id_out = "monin_obukhov";
        return true;
    }
    return false;
}

/**
 * @brief Parses boundary-layer top-diagnosis method aliases.
 */
bool parse_boundary_layer_top_method(const std::string& value, std::string& method_out)
{
    const std::string normalized = to_lower_copy(value);
    if (normalized == "ri")
    {
        method_out = "ri";
        return true;
    }
    if (normalized == "theta" || normalized == "thetav" || normalized == "theta_v")
    {
        method_out = "theta";
        return true;
    }
    if (normalized == "tke")
    {
        method_out = "tke";
        return true;
    }
    return false;
}

/**
 * @brief Parses turbulence operating mode aliases.
 */
bool parse_turbulence_mode(const std::string& value, std::string& mode_out)
{
    const std::string normalized = to_lower_copy(value);
    if (normalized == "3d" || normalized == "full")
    {
        mode_out = "3d";
        return true;
    }
    if (normalized == "2d" || normalized == "horizontal_only" || normalized == "horizontal-only" ||
        normalized == "horizontal")
    {
        mode_out = "horizontal_only";
        return true;
    }
    if (normalized == "vertical_only" || normalized == "vertical-only" || normalized == "vertical")
    {
        mode_out = "vertical_only";
        return true;
    }
    return false;
}

/**
 * @brief Parses turbulence filter-width selection.
 */
bool parse_turbulence_filter_width(const std::string& value, std::string& filter_width_out)
{
    const std::string normalized = to_lower_copy(value);
    if (normalized == "dx" || normalized == "cubic_root" || normalized == "user")
    {
        filter_width_out = normalized;
        return true;
    }
    return false;
}

/**
 * @brief Parses turbulence stability-correction selector.
 */
bool parse_turbulence_stability_correction(const std::string& value, std::string& stability_out)
{
    const std::string normalized = to_lower_copy(value);
    if (normalized == "none" || normalized == "false" || normalized == "off" || normalized == "0")
    {
        stability_out = "none";
        return true;
    }
    if (normalized == "ri" || normalized == "true" || normalized == "on" || normalized == "1")
    {
        stability_out = "ri";
        return true;
    }
    if (normalized == tmv::kLillyRichardsonModel)
    {
        stability_out = tmv::kLillyRichardsonModel;
        return true;
    }
    if (normalized == "tke" || normalized == "tke_based" || normalized == "tke-based")
    {
        stability_out = "tke_based";
        return true;
    }
    return false;
}

/**
 * @brief Parses terrain coordinate-system aliases.
 */
bool parse_terrain_coord_id(const std::string& value, std::string& coord_out)
{
    const std::string normalized = to_lower_copy(value);
    if (normalized == "btf" || normalized == "terrain_following" || normalized == "terrain-following" ||
        normalized == "sigma")
    {
        coord_out = "btf";
        return true;
    }
    if (normalized == "smoothed" || normalized == "smooth")
    {
        coord_out = "smoothed";
        return true;
    }
    if (normalized == "cartesian" || normalized == "flat" || normalized == "none")
    {
        coord_out = "cartesian";
        return true;
    }
    return false;
}

/**
 * @brief Parses a YAML file.
 */

/**
 * @brief Parses validation override keys of form `field_overrides.<field>.(min|max)`.
 */
bool parse_override_key(const std::string& key, std::string& field_id_out, bool& is_min_out)
{
    const std::string prefix = "validation.field_overrides.";
    if (key.rfind(prefix, 0) != 0)
    {
        return false;
    }

    const std::string tail = key.substr(prefix.size());
    const std::size_t dot  = tail.rfind('.');
    if (dot == std::string::npos || dot == 0 || dot + 1 >= tail.size())
    {
        return false;
    }

    field_id_out                = tail.substr(0, dot);
    const std::string bound_key = tail.substr(dot + 1);
    if (bound_key == "min")
    {
        is_min_out = true;
        return true;
    }
    if (bound_key == "max")
    {
        is_min_out = false;
        return true;
    }

    return false;
}

/**
 * @brief Escapes a string for JSON output.
 */
std::string json_escape_local(const std::string& value)
{
    return tmv::strutil::json_escape(value);
}

static RuntimeConfigurationState& runtime_configuration            = global_simulation_storage.configuration;
static bool                       internal_milbrandt_qualification = false;
static bool                       internal_moving_nest_qualification = false;

bool exact_two_way_nested_profile_selected()
{
    return runtime_configuration.admission.two_way_nested;
}

bool exact_native_mapped_terrain_profile_selected()
{
    return runtime_configuration.admission.native_mapped_terrain;
}

void set_internal_milbrandt_qualification(bool enabled)
{
    internal_milbrandt_qualification = enabled;
}

void set_internal_moving_nest_qualification(bool enabled)
{
    internal_moving_nest_qualification = enabled;
    tmv::init::config::set_internal_moving_nest_qualification(enabled);
}

WindProfile& global_wind_profile      = runtime_configuration.wind_profile;
double&      global_cape_target       = runtime_configuration.cape_target;
double&      global_sfc_theta_k       = runtime_configuration.sfc_theta_k;
double&      global_sfc_qv_kgkg       = runtime_configuration.sfc_qv_kgkg;
double&      global_tropopause_z_m    = runtime_configuration.tropopause_z_m;
double&      global_bubble_center_x_m = runtime_configuration.bubble_center_x_m;
double&      global_bubble_center_y_m = runtime_configuration.bubble_center_y_m;
double&      global_bubble_center_z_m = runtime_configuration.bubble_center_z_m;
double&      global_bubble_radius_m   = runtime_configuration.bubble_radius_m;
double&      global_bubble_dtheta_k   = runtime_configuration.bubble_dtheta_k;
double&      global_bubble_cloud_water_kgkg =
    runtime_configuration.bubble_cloud_water_kgkg;

bool&                             global_sounding_enabled  = runtime_configuration.sounding_enabled;
tmv::init::config::ExecutionMode& global_execution_mode    = runtime_configuration.execution_mode;
bool&           global_sounding_allow_placeholder_profiles = runtime_configuration.sounding_allow_placeholder_profiles;
SoundingConfig& global_runtime_sounding_config             = runtime_configuration.sounding_config;
tmv::init::SoundingSourceConfig&  global_sounding_source_config  = runtime_configuration.sounding_source;
tmv::init::HodographSourceConfig& global_hodograph_source_config = runtime_configuration.hodograph_source;
tmv::init::TriggerSourceConfig&   global_trigger_source_config   = runtime_configuration.trigger_source;
std::string&                      global_microphysics_scheme     = runtime_configuration.microphysics_scheme;
tmv::MicrophysicsConfig&          global_microphysics_config     = runtime_configuration.microphysics_config;
std::string&                      global_dynamics_scheme_name    = runtime_configuration.dynamics_scheme_name;

CoordinateSystem&             global_coordinate_system = runtime_configuration.coordinate_system;
StaggerType&                  global_stagger_type      = runtime_configuration.stagger_type;
tmv::init::config::LateralBC& global_lateral_bc_west   = runtime_configuration.lateral_bc_west;
tmv::init::config::LateralBC& global_lateral_bc_east   = runtime_configuration.lateral_bc_east;
tmv::init::config::LateralBC& global_lateral_bc_south  = runtime_configuration.lateral_bc_south;
tmv::init::config::LateralBC& global_lateral_bc_north  = runtime_configuration.lateral_bc_north;
tmv::init::config::BottomBC&  global_bottom_bc         = runtime_configuration.bottom_bc;
tmv::init::config::TopBC&     global_top_bc            = runtime_configuration.top_bc;

bool&                           global_damping_enabled        = runtime_configuration.damping_enabled;
double&                         global_damping_z_d            = runtime_configuration.damping_z_d;
double&                         global_damping_alpha          = runtime_configuration.damping_alpha;
int&                            global_lateral_sponge_width   = runtime_configuration.lateral_sponge_width;
double&                         global_lateral_sponge_alpha   = runtime_configuration.lateral_sponge_alpha;
tmv::init::config::DampingMode& global_damping_mode           = runtime_configuration.damping_mode;
bool&                           global_w_only_top_sponge      = runtime_configuration.w_only_top_sponge;
int&                            global_w_only_depth_cells     = runtime_configuration.w_only_depth_cells;
double&                         global_w_only_alpha_max       = runtime_configuration.w_only_alpha_max;
bool&                           global_lateral_sponge_enabled = runtime_configuration.lateral_sponge_enabled;

bool&   global_slow_rk2               = runtime_configuration.slow_rk2;
double& global_lid_bc_gamma           = runtime_configuration.lid_bc_gamma;
bool&   global_axis_u_m1_filter       = runtime_configuration.axis_u_m1_filter;
bool&   global_gpu_apply_microphysics = runtime_configuration.gpu_apply_microphysics;
bool&   global_gpu_apply_slow         = runtime_configuration.gpu_apply_slow;
bool&   global_gpu_apply_sgs          = runtime_configuration.gpu_apply_sgs;

bool&                            global_coriolis_enabled      = runtime_configuration.coriolis_enabled;
tmv::init::config::CoriolisMode& global_coriolis_mode         = runtime_configuration.coriolis_mode;
double&                          global_coriolis_latitude_deg = runtime_configuration.coriolis_latitude_deg;

LogProfile& global_log_profile                        = runtime_configuration.log_profile;
bool&       global_perf_timing_enabled                = runtime_configuration.perf_timing_enabled;
int&        global_perf_report_every_steps            = runtime_configuration.perf_report_every_steps;
int&        global_perf_warmup_steps                  = runtime_configuration.perf_warmup_steps;
int&        global_scientific_diagnostics_every_steps = runtime_configuration.scientific_diagnostics_every_steps;
double&     global_configured_dt_large_s              = runtime_configuration.configured_dt_large_s;
tmv::init::config::DomainMotionMode&              global_domain_motion_mode = runtime_configuration.domain_motion_mode;
tmv::domain::HorizontalDomainFrame&               global_domain_frame       = runtime_configuration.domain_frame;
tmv::init::config::DomainMotionSection::Tracking& global_storm_tracking_config = runtime_configuration.storm_tracking;
tmv::ValidationPolicy&                            global_validation_policy = runtime_configuration.validation_policy;
std::string& global_validation_report_path = runtime_configuration.validation_report_path;
int&         global_checkpoint_every_s     = runtime_configuration.checkpoint_every_s;
int&         global_checkpoint_keep_last   = runtime_configuration.checkpoint_keep_last;
int&         global_checkpoint_minimum_free_space_mib = runtime_configuration.checkpoint_minimum_free_space_mib;
std::string& global_checkpoint_directory   = runtime_configuration.checkpoint_directory;
std::string& global_checkpoint_resume_from = runtime_configuration.checkpoint_resume_from;

namespace
{
// Re-establish the defaults/environment baseline before every config load.
// Embedded callers can load multiple files in one process; an explicit value
// in file A must not leak into file B when B omits that key. Config values are
// applied after this reset and the executable reapplies CLI values afterward,
// preserving CLI > explicit config > environment > defaults.
void reset_performance_and_diagnostics_baseline()
{
    global_perf_timing_enabled                = false;
    global_perf_report_every_steps            = 0;
    global_perf_warmup_steps                  = 0;
    global_scientific_diagnostics_every_steps = -1;
    tmv::diagnostics::set_deterministic_reductions(false);

    if (const char* env_perf = std::getenv("NIMBUS_PERF_TIMING"))
    {
        global_perf_timing_enabled = parse_bool_value(env_perf);
    }
    if (const char* env_every = std::getenv("NIMBUS_PERF_EVERY_STEPS"))
    {
        int parsed = 0;
        if (try_parse_non_negative_int_value(env_every, parsed))
        {
            global_perf_report_every_steps = parsed;
        }
    }
}

enum class QualifiedStationaryNestingProfile
{
    None,
    Dry,
    TwoWayDry,
    TwoWayMoist,
    Smagorinsky,
    KesslerSmagorinsky,
    DrySimpleGreyLongwave,
};

bool canonical_qualified_parameter(double value, double reference)
{
    return std::isfinite(value) &&
        std::abs(value - reference) <=
            8.0 * std::numeric_limits<double>::epsilon() *
                std::max(1.0, std::abs(reference));
}

QualifiedStationaryNestingProfile classify_qualified_stationary_nesting_profile()
{
    const auto& nesting = runtime_configuration.nesting;
    ComputeBackendKind backend;
    if (!parse_compute_backend_kind(global_compute_backend_config.backend, backend))
        return QualifiedStationaryNestingProfile::None;
    const bool supported_ratio =
        (nesting.parent_ratio_x == 2 && nesting.parent_ratio_y == 2 &&
         nesting.parent_time_step_ratio == 2 &&
         nesting.child_cells_x == 8 && nesting.child_cells_y == 8) ||
        (runtime_configuration.admission.active_terrain_nested &&
         nesting.parent_ratio_x == 3 && nesting.parent_ratio_y == 3 &&
         nesting.parent_time_step_ratio == 3 &&
         nesting.child_cells_x == 9 && nesting.child_cells_y == 9);
    const bool common =
        nesting.enabled &&
        global_execution_mode ==
            tmv::init::config::ExecutionMode::Qualified &&
        (backend == ComputeBackendKind::Cpu || backend == ComputeBackendKind::Vulkan ||
         (backend == ComputeBackendKind::Metal && runtime_configuration.admission.two_way_accelerator_reference)) &&
        !global_compute_backend_config.allow_fallback &&
        global_domain_motion_mode ==
            tmv::init::config::DomainMotionMode::Stationary &&
        global_coordinate_system == CoordinateSystem::Cartesian &&
        global_stagger_type == StaggerType::CGrid &&
        (global_terrain_config.scheme_id == "none" ||
         runtime_configuration.admission.active_terrain_nested) &&
        !global_simulation_storage.nested.config.enabled &&
        global_dynamics_scheme_name == tmv::kCartesianCGridDynamicsId &&
        (to_lower_copy(global_advection_config.scheme_id) == "tvd" ||
         (to_lower_copy(global_advection_config.scheme_id) == "weno5" && runtime_configuration.admission.two_way_accelerator_reference)) &&
        to_lower_copy(global_time_stepping_config.scheme_id) ==
            "split_explicit" &&
        global_time_stepping_config.split_acoustic &&
        to_lower_copy(global_diffusion_config.scheme_id) == "explicit" &&
        global_diffusion_config.operator_order == 2 &&
        global_diffusion_config.K_h == 0.0 &&
        global_diffusion_config.K_v == 0.0 &&
        !global_diffusion_config.use_variable_K &&
        (nesting.feedback == tmv::init::config::NestingFeedback::OneWay ||
         runtime_configuration.admission.two_way_nested) &&
        supported_ratio &&
        !global_coriolis_enabled && !global_damping_enabled &&
        !global_w_only_top_sponge && !global_lateral_sponge_enabled &&
        !global_storm_tracking_config.enabled &&
        !global_storm_tracking_config.recycle_domain &&
        global_lateral_bc_west ==
            tmv::init::config::LateralBC::ZeroGradient &&
        global_lateral_bc_east ==
            tmv::init::config::LateralBC::ZeroGradient &&
        global_lateral_bc_south ==
            tmv::init::config::LateralBC::ZeroGradient &&
        global_lateral_bc_north ==
            tmv::init::config::LateralBC::ZeroGradient &&
        global_bottom_bc == tmv::init::config::BottomBC::RigidSlip &&
        global_top_bc == tmv::init::config::TopBC::Rigid &&
        global_boundary_layer_config.scheme_id == "none" &&
        !global_boundary_layer_config.apply_surface_fluxes &&
        global_chaos_config.scheme_id == "none";
    if (!common)
        return QualifiedStationaryNestingProfile::None;

    const bool no_turbulence =
        global_turbulence_config.scheme_id == "none";
    const bool qualified_smagorinsky =
        global_turbulence_config.scheme_id == "smagorinsky" &&
        global_turbulence_config.dt_sgs == 1.0 &&
        global_turbulence_config.mode == "3d" &&
        global_turbulence_config.filter_width == "cubic_root" &&
        canonical_qualified_parameter(
            global_turbulence_config.Cs,
            tmv::turbulence_constants::C_s_default) &&
        canonical_qualified_parameter(
            global_turbulence_config.Pr_t,
            tmv::turbulence_constants::Pr_t_default) &&
        canonical_qualified_parameter(
            global_turbulence_config.Sc_t,
            tmv::turbulence_constants::Sc_t_default) &&
        !global_turbulence_config.dynamic_Cs &&
        global_turbulence_config.stability_correction == "none" &&
        !global_turbulence_config.enable_moist_buoyancy &&
        !global_turbulence_config.enable_diagnostics &&
        global_turbulence_config.nu_t_max == 1000.0 &&
        global_turbulence_config.K_max == 1000.0 &&
        global_gpu_apply_sgs;
    const bool qualified_kessler =
        global_microphysics_scheme == "kessler" &&
        canonical_qualified_parameter(global_microphysics_config.shared.cloud_water_threshold_kgkg, 1.0e-3) &&
        canonical_qualified_parameter(global_microphysics_config.shared.autoconversion_rate_s_inv, 1.0e-3) &&
        canonical_qualified_parameter(global_microphysics_config.kessler.accretion_coefficient, 2.2) &&
        canonical_qualified_parameter(global_microphysics_config.kessler.evaporation_efficiency, 1.0) &&
        global_gpu_apply_microphysics;
    const bool no_radiation = global_radiation_config.scheme_id == "none";
    if (runtime_configuration.admission.two_way_nested &&
        (backend == ComputeBackendKind::Cpu || runtime_configuration.admission.two_way_accelerator_reference) &&
        (global_microphysics_scheme == "kessler" || global_microphysics_scheme == "lin" ||
         global_microphysics_scheme == "thompson" || global_microphysics_scheme == "milbrandt") &&
        no_turbulence && no_radiation)
        return QualifiedStationaryNestingProfile::TwoWayMoist;
    const bool qualified_simple_grey_longwave = tmv::runtime::is_exact_qualified_simple_grey_longwave({
        global_radiation_config.scheme_id,
        global_radiation_config.solar_forcing,
        global_radiation_config.do_lw,
        global_radiation_config.do_sw,
        global_radiation_config.clear_sky_only,
        global_radiation_config.use_subcycling,
        global_radiation_config.enable_diagnostics,
        global_radiation_config.dt_radiation,
        global_radiation_config.tau_lw_ref,
        global_radiation_config.n_lw,
        global_surface_config.emissivity,
        global_surface_config.Tsfc,
    });

    if (qualified_simple_grey_longwave &&
        global_microphysics_scheme == "none" && no_turbulence)
    {
        return QualifiedStationaryNestingProfile::DrySimpleGreyLongwave;
    }
    if (!no_radiation)
        return QualifiedStationaryNestingProfile::None;
    if (global_microphysics_scheme == "none" && no_turbulence)
        return nesting.feedback ==
                tmv::init::config::NestingFeedback::TwoWay
            ? QualifiedStationaryNestingProfile::TwoWayDry
            : QualifiedStationaryNestingProfile::Dry;
    if (global_microphysics_scheme == "none" && qualified_smagorinsky)
        return QualifiedStationaryNestingProfile::Smagorinsky;
    if (qualified_kessler && qualified_smagorinsky)
        return QualifiedStationaryNestingProfile::KesslerSmagorinsky;
    return QualifiedStationaryNestingProfile::None;
}

void validate_runtime_science_profile(tmv::init::config::ExecutionMode execution_mode,
                                      const OutputConfig*              output_config,
                                      const std::string&               surface_scheme_id,
                                      const tmv::init::config::ConfigV1* typed_config,
                                      int                              write_every_s,
                                      bool                             live_shm,
                                      bool                             trajectories_enabled)
{
    const bool        qualified = execution_mode == tmv::init::config::ExecutionMode::Qualified;
    const std::string backend   = to_lower_copy(global_compute_backend_config.backend) == "cpu" ? "cpu" : "gpu";
    const auto*       dynamics_descriptors     = &tmv::get_dynamics_scheme_descriptors();
    const auto*       microphysics_descriptors = &tmv::get_microphysics_scheme_descriptors();
    std::vector<tmv::SchemeDescriptor> qualification_microphysics_descriptors;
    if (internal_milbrandt_qualification && !qualified && global_microphysics_scheme == "milbrandt" &&
        global_coordinate_system == CoordinateSystem::Cartesian && global_stagger_type == StaggerType::CGrid &&
        backend == "cpu")
    {
        qualification_microphysics_descriptors = *microphysics_descriptors;
        for (auto& descriptor : qualification_microphysics_descriptors)
        {
            if (descriptor.id == "milbrandt")
            {
                descriptor.runtime_selectable = true;
                break;
            }
        }
        microphysics_descriptors = &qualification_microphysics_descriptors;
    }
    const auto* radiation_descriptors =
        &tmv::get_radiation_scheme_descriptors();
    const auto* time_stepping_descriptors =
        &tmv::get_time_stepping_scheme_descriptors();
    std::vector<tmv::SchemeDescriptor> qualified_time_stepping_descriptors;
    if (qualified && runtime_configuration.admission.coupled_rk)
    {
        qualified_time_stepping_descriptors = *time_stepping_descriptors;
        for (auto& descriptor : qualified_time_stepping_descriptors)
        {
            if (descriptor.id == "rk3" || descriptor.id == "rk4")
            {
                descriptor.maturity = tmv::SchemeMaturity::Qualified;
                descriptor.runtime_selectable = true;
            }
        }
        time_stepping_descriptors = &qualified_time_stepping_descriptors;
    }
    std::vector<tmv::runtime::SelectedScheme> selections = {
        {"dynamics.scheme", global_dynamics_scheme_name, dynamics_descriptors},
        {"microphysics.scheme", global_microphysics_scheme, microphysics_descriptors},
        {"radiation.scheme", global_radiation_config.scheme_id, radiation_descriptors},
        {"boundary_layer.scheme",
         global_boundary_layer_config.scheme_id,
         &tmv::get_boundary_layer_scheme_descriptors()},
        {"turbulence.scheme", global_turbulence_config.scheme_id, &tmv::get_turbulence_scheme_descriptors()},
        {"numerics.advection.scheme", global_advection_config.scheme_id, &tmv::get_advection_scheme_descriptors()},
        {"numerics.diffusion.scheme", global_diffusion_config.scheme_id, &tmv::get_diffusion_scheme_descriptors()},
        {"numerics.time_stepping.scheme",
         global_time_stepping_config.scheme_id,
         time_stepping_descriptors},
        {"chaos.scheme", global_chaos_config.scheme_id, &tmv::chaos::get_chaos_scheme_descriptors()},
        {"terrain.scheme", global_terrain_config.scheme_id, &tmv::get_terrain_scheme_descriptors()},
    };
    if (global_sounding_source_config.type == tmv::init::SoundingSourceConfig::Type::File)
    {
        selections.push_back({
            "environment.sounding.file.scheme_id",
            global_sounding_source_config.file.scheme_id,
            &tmv::get_sounding_scheme_descriptors(),
        });
    }

    auto errors = tmv::runtime::validate_science_profile(selections,
                                                         qualified,
                                                         coordinate_system_name(global_coordinate_system),
                                                         stagger_type_name(global_stagger_type),
                                                         backend,
                                                         runtime_configuration.admission.mapped_terrain &&
                                                             global_stagger_type == StaggerType::Collocated);
    if (global_turbulence_config.scheme_id == "smagorinsky" &&
        global_turbulence_config.dynamic_Cs &&
        global_turbulence_config.mode != "3d")
    {
        errors.push_back("dynamic Smagorinsky requires turbulence.mode=3d in every execution mode");
    }
    if (qualified && global_turbulence_config.scheme_id == "smagorinsky" &&
        global_turbulence_config.mode == "vertical_only" &&
        !runtime_configuration.admission.static_vertical_only_sgs_cpu)
    {
        errors.push_back("qualified static vertical_only Smagorinsky is guarded pending whole-step failure atomicity");
    }
    ComputeBackendKind requested_kind = ComputeBackendKind::Cpu;
    if (qualified && global_diffusion_config.use_variable_K &&
        !runtime_configuration.admission.variable_diffusion)
    {
        errors.push_back(
            "qualified linear-height variable diffusion is limited to the exact flat native reference profiles");
    }
    if (qualified && global_turbulence_config.filter_width == "user" &&
        !runtime_configuration.admission.user_filter_width)
    {
        errors.push_back(
            "turbulence.filter_width=user is qualified only for the exact flat periodic C-grid CPU Smagorinsky profile");
    }
    if (qualified && global_turbulence_config.dynamic_Cs &&
        !runtime_configuration.admission.dynamic_smagorinsky)
    {
        errors.push_back(
            "dynamic Smagorinsky is qualified only for the exact flat periodic C-grid CPU reference profile");
    }
    if (qualified && global_turbulence_config.scheme_id == "tke" &&
        !runtime_configuration.admission.tke_stability &&
        !runtime_configuration.admission.coupled_rk &&
        !runtime_configuration.admission.agrid_core &&
        !runtime_configuration.admission.mapped_terrain)
    {
        errors.push_back(
            "turbulence.scheme=tke is outside the qualified flat, RK, A-grid, and mapped CPU families");
    }
    if (qualified && (global_turbulence_config.stability_correction == "ri"
        || global_turbulence_config.stability_correction == tmv::kLillyRichardsonModel)
        && !runtime_configuration.admission.richardson_smagorinsky)
    {
        errors.push_back(
            "Richardson-corrected Smagorinsky requires the maintained Lilly CPU reference profile");
    }
    if (qualified && !global_boundary_layer_config.clear_sky_only &&
        !runtime_configuration.admission.cloud_top_radiative_ysu)
    {
        errors.push_back(
            "cloud-top radiative YSU mixing is qualified only for the exact surface-coupled CPU reference profile");
    }
    if (qualified && !admits_qualified_metal(runtime_configuration.admission) &&
        parse_compute_backend_kind(global_compute_backend_config.backend, requested_kind) &&
        requested_kind == ComputeBackendKind::Metal)
    {
        errors.push_back(kQualifiedMetalScopeError);
    }
    if (qualified && runtime_configuration.admission.generated_metal_mixed_npy)
    {
        if (write_every_s != 1)
        {
            errors.push_back("qualified mixed-open Metal NPY scope requires --write-every=1");
        }
        if (live_shm)
        {
            errors.push_back("qualified generated Metal scope does not admit live-SHM CLI overrides");
        }
        if (trajectories_enabled || (output_config != nullptr && output_config->trajectories))
        {
            errors.push_back("qualified generated Metal scope does not admit trajectory CLI overrides");
        }
    }
    tmv::runtime::append_qualified_kessler_scope_error(
        errors,
        qualified,
        global_microphysics_scheme == "kessler",
        runtime_configuration.nesting.enabled,
        runtime_configuration.admission.domain_recycling || runtime_configuration.admission.radar ||
            runtime_configuration.admission.simple_grey_absorbing || runtime_configuration.admission.coupled_rk ||
            runtime_configuration.admission.agrid_core || runtime_configuration.admission.mapped_terrain);
    tmv::runtime::append_qualified_lin_scope_error(
        errors,
        qualified,
        global_microphysics_scheme == "lin",
        runtime_configuration.admission.two_way_nested || runtime_configuration.admission.lin_rrtmg || runtime_configuration.admission.lin_storm ||
            runtime_configuration.admission.coupled_rk || runtime_configuration.admission.agrid_core ||
            runtime_configuration.admission.mapped_terrain);
    if (qualified && global_microphysics_scheme == "thompson" && !runtime_configuration.admission.two_way_nested &&
        !runtime_configuration.admission.thompson_cpu && !runtime_configuration.admission.thompson_native &&
        !runtime_configuration.admission.coupled_rk && !runtime_configuration.admission.agrid_core &&
        !runtime_configuration.admission.mapped_terrain)
        errors.push_back(
            "microphysics.scheme=thompson requires a maintained CPU profile or the native C-grid Thompson reference");
    tmv::runtime::append_qualified_tke_scope_error(
        errors,
        qualified && runtime_configuration.admission.tke_stability,
        {
            global_turbulence_config.scheme_id,
            backend,
            coordinate_system_name(global_coordinate_system),
            stagger_type_name(global_stagger_type),
            global_turbulence_config.mode,
            global_microphysics_scheme,
            global_boundary_layer_config.scheme_id,
            surface_scheme_id,
            global_terrain_config.scheme_id,
            global_chaos_config.scheme_id,
            global_diffusion_config.scheme_id,
            global_diffusion_config.apply_to,
            tmv::init::config::lateral_bc_name(global_lateral_bc_west),
            tmv::init::config::lateral_bc_name(global_lateral_bc_east),
            tmv::init::config::lateral_bc_name(global_lateral_bc_south),
            tmv::init::config::lateral_bc_name(global_lateral_bc_north),
            tmv::init::config::bottom_bc_name(global_bottom_bc),
            tmv::init::config::top_bc_name(global_top_bc),
            global_boundary_layer_config.apply_surface_fluxes,
            global_diffusion_config.K_h > 0.0
                || global_diffusion_config.K_v > 0.0
                || global_diffusion_config.use_variable_K,
            global_damping_enabled,
            global_lateral_sponge_enabled,
            global_w_only_top_sponge,
            global_turbulence_config.Pr_t,
            global_turbulence_config.Sc_t,
            global_turbulence_config.stability_correction,
        });
    tmv::runtime::append_qualified_weno5_scope_error(
        errors,
        qualified && !runtime_configuration.admission.two_way_accelerator_reference && !runtime_configuration.admission.native_weno5 && !runtime_configuration.admission.tke_stability &&
            !runtime_configuration.admission.thompson_cpu && !runtime_configuration.admission.coupled_rk &&
            !runtime_configuration.admission.agrid_core && !runtime_configuration.admission.mapped_terrain,
        {
            global_advection_config.scheme_id,
            backend,
            coordinate_system_name(global_coordinate_system),
            stagger_type_name(global_stagger_type),
            global_dynamics_scheme_name,
            global_microphysics_scheme,
            global_turbulence_config.scheme_id,
            global_boundary_layer_config.scheme_id,
            surface_scheme_id,
            global_radiation_config.scheme_id,
            global_terrain_config.scheme_id,
            global_chaos_config.scheme_id,
            global_diffusion_config.scheme_id,
            global_diffusion_config.apply_to,
            tmv::init::config::lateral_bc_name(global_lateral_bc_west),
            tmv::init::config::lateral_bc_name(global_lateral_bc_east),
            tmv::init::config::lateral_bc_name(global_lateral_bc_south),
            tmv::init::config::lateral_bc_name(global_lateral_bc_north),
            tmv::init::config::bottom_bc_name(global_bottom_bc),
            tmv::init::config::top_bc_name(global_top_bc),
            tmv::init::config::domain_motion_mode_name(global_domain_motion_mode),
            global_boundary_layer_config.apply_surface_fluxes,
            global_diffusion_config.K_h > 0.0 || global_diffusion_config.K_v > 0.0 ||
                global_diffusion_config.use_variable_K,
            global_damping_enabled,
            global_lateral_sponge_enabled,
            global_w_only_top_sponge,
            runtime_configuration.nesting.enabled,
        });
    tmv::runtime::append_qualified_implicit_diffusion_scope_error(
        errors,
        qualified,
        {
            global_diffusion_config.scheme_id,
            backend,
            coordinate_system_name(global_coordinate_system),
            stagger_type_name(global_stagger_type),
            global_dynamics_scheme_name,
            global_advection_config.scheme_id,
            global_microphysics_scheme,
            global_turbulence_config.scheme_id,
            global_boundary_layer_config.scheme_id,
            surface_scheme_id,
            global_radiation_config.scheme_id,
            global_terrain_config.scheme_id,
            global_chaos_config.scheme_id,
            global_diffusion_config.apply_to,
            tmv::init::config::lateral_bc_name(global_lateral_bc_west),
            tmv::init::config::lateral_bc_name(global_lateral_bc_east),
            tmv::init::config::lateral_bc_name(global_lateral_bc_south),
            tmv::init::config::lateral_bc_name(global_lateral_bc_north),
            tmv::init::config::bottom_bc_name(global_bottom_bc),
            tmv::init::config::top_bc_name(global_top_bc),
            tmv::init::config::domain_motion_mode_name(global_domain_motion_mode),
            global_boundary_layer_config.apply_surface_fluxes,
            global_diffusion_config.use_variable_K,
            global_damping_enabled,
            global_lateral_sponge_enabled,
            global_w_only_top_sponge,
            runtime_configuration.nesting.enabled,
            global_diffusion_config.K_h,
            global_diffusion_config.K_v,
        },
        runtime_configuration.admission.mapped_terrain,
        runtime_configuration.admission.native_implicit_diffusion || runtime_configuration.admission.coupled_rk);
    tmv::runtime::append_qualified_terrain_scope_error(
        errors,
        qualified,
        global_terrain_config.scheme_id,
        runtime_configuration.admission.mapped_terrain);
    const tmv::runtime::ScienceInteractionSelection interactions{
        global_boundary_layer_config.scheme_id,
        global_turbulence_config.scheme_id,
        global_turbulence_config.mode,
        surface_scheme_id,
        global_boundary_layer_config.apply_surface_fluxes,
        global_terrain_config.scheme_id,
        global_radiation_config.scheme_id,
        global_radiation_config.do_sw,
        global_chaos_config.scheme_id,
        global_diffusion_config.scheme_id,
        global_diffusion_config.apply_to,
        global_diffusion_config.K_h > 0.0 || global_diffusion_config.K_v > 0.0 ||
            global_diffusion_config.use_variable_K,
        output_config == nullptr ? std::string{} : std::string(output_format_name(output_config->format)),
        qualified && !runtime_configuration.admission.mapped_terrain,
        stagger_type_name(global_stagger_type),
        tmv::init::config::lateral_bc_name(global_lateral_bc_west),
        tmv::init::config::lateral_bc_name(global_lateral_bc_east),
        tmv::init::config::lateral_bc_name(global_lateral_bc_south),
        tmv::init::config::lateral_bc_name(global_lateral_bc_north),
        tmv::init::config::bottom_bc_name(global_bottom_bc),
        tmv::init::config::top_bc_name(global_top_bc),
        global_scientific_diagnostics_every_steps,
        backend,
        global_dynamics_scheme_name,
        global_microphysics_scheme,
        global_gpu_apply_microphysics,
        global_gpu_apply_slow,
        global_gpu_apply_sgs,
    };
    const auto interaction_errors = tmv::runtime::validate_science_interactions(interactions);
    errors.insert(errors.end(), interaction_errors.begin(), interaction_errors.end());

    // Initialization compatibility belongs to the same pre-allocation
    // decision. The old full-runtime call happened only after the sounding,
    // wind, trigger, chaos, and thermodynamic state had already been built.
    tmv::init::ValidationInputs initialization;
    initialization.scheme_id             = global_dynamics_scheme_name;
    initialization.turbulence_scheme_id  = global_turbulence_config.scheme_id;
    initialization.coordinate            = global_coordinate_system;
    initialization.stagger               = global_stagger_type;
    initialization.sounding              = global_sounding_source_config;
    initialization.hodograph             = global_hodograph_source_config;
    initialization.trigger               = global_trigger_source_config;
    const auto initialization_validation = tmv::init::validate_initial_condition_config(initialization);
    for (const auto& warning : initialization_validation.warnings)
        tmv::log_warn(warning);
    errors.insert(errors.end(), initialization_validation.errors.begin(), initialization_validation.errors.end());

    if (qualified && global_microphysics_scheme == "milbrandt" &&
        parse_compute_backend_kind(global_compute_backend_config.backend, requested_kind) &&
        requested_kind != ComputeBackendKind::Cpu && !runtime_configuration.admission.milbrandt_reference &&
        !runtime_configuration.admission.two_way_nested && !runtime_configuration.admission.coupled_rk)
        errors.push_back("microphysics.scheme=milbrandt requires the maintained native C-grid reference profile");
    if (qualified && output_config != nullptr && output_config->format == OutputFormat::zfp)
    {
        OutputConfig qualified_output = *output_config;
        resolve_output_fields(qualified_output);
        std::vector<std::string> selected_radar_products;
        if (global_radar_product_selection.reflectivity)
            selected_radar_products.push_back("reflectivity_dbz");
        if (global_radar_product_selection.radial_velocity)
            selected_radar_products.push_back("radial_velocity");
        if (global_radar_product_selection.zdr)
            selected_radar_products.push_back("zdr");
        if (global_radar_product_selection.kdp)
            selected_radar_products.push_back("kdp");
        if (global_radar_product_selection.rhohv)
            selected_radar_products.push_back("rhohv");
        merge_selected_radar_output_fields(
            qualified_output, selected_radar_products);
        nimbus::output::QualifiedRadarZfpV4Context radar_context;
        if (typed_config != nullptr)
        {
            radar_context = make_qualified_radar_zfp_v4_context(
                *typed_config, qualified_output, write_every_s,
                live_shm, trajectories_enabled);
        }
        const auto zfp_errors = validate_qualified_zfp_history(
            qualified_output, &radar_context);
        for (const std::string& error : zfp_errors)
        {
            errors.push_back("output.format=zfp: " + error);
        }
    }

    if (!errors.empty())
    {
        std::ostringstream message;
        message << "Science profile rejected before initialization:";
        for (const auto& error : errors)
        {
            message << "\n  - " << error;
        }
        throw std::runtime_error(message.str());
    }
}

void preflight_file_sounding_before_allocation()
{
    if (global_sounding_source_config.type != tmv::init::SoundingSourceConfig::Type::File)
    {
        return;
    }
    std::vector<double> z_m(static_cast<std::size_t>(NZ));
    for (int k = 0; k < NZ; ++k)
    {
        z_m[static_cast<std::size_t>(k)] = tmv::grid_coordinates::scalar_center(k, dz);
    }
    auto source = tmv::init::make_sounding_source(global_sounding_source_config);
    auto column = source->build(z_m, dz);
    column.verify_self_consistent();
}

using LifetimePlanning = tmv::init::config::DomainMotionSection::LifetimePlanning;

LifetimePlanning parse_legacy_lifetime_planning(const std::unordered_map<std::string, std::string>& config)
{
    LifetimePlanning result;
    const auto enabled = config.find("domain_motion.lifetime_planning.enabled");
    result.enabled = enabled != config.end() && parse_bool_value(enabled->second);
    if (!result.enabled) return result;
    const auto required_string = [&config](const char* key) -> const std::string&
    {
        const auto it = config.find(key);
        if (it == config.end())
            throw std::runtime_error(std::string("Missing required ") + key);
        return it->second;
    };
    const auto required_finite = [&required_string](const char* key)
    {
        double             value = 0.0;
        const std::string& raw   = required_string(key);
        if (!try_parse_double_value(raw, value))
            throw std::runtime_error(std::string("Invalid ") + key + " '" + raw + "'; expected a finite number");
        return value;
    };
    const auto required_positive = [&required_finite](const char* key)
    {
        const double value = required_finite(key);
        if (value <= 0.0)
            throw std::runtime_error(std::string(key) + " must be positive when lifetime planning is enabled");
        return value;
    };
    const auto required_non_negative = [&required_finite](const char* key)
    {
        const double value = required_finite(key);
        if (value < 0.0)
            throw std::runtime_error(std::string(key) + " must be non-negative");
        return value;
    };

    if (!tmv::init::config::parse_lifetime_planning_action(
            required_string("domain_motion.lifetime_planning.action"), result.action) ||
        !tmv::init::config::parse_storm_motion_source(
            required_string("domain_motion.lifetime_planning.motion_source"), result.motion_source))
        throw std::invalid_argument("invalid storm lifetime planning selection");
    if (result.motion_source == tmv::init::config::StormMotionSource::Explicit)
    {
        result.explicit_u_ms = required_finite("domain_motion.lifetime_planning.explicit_u_ms");
        result.explicit_v_ms = required_finite("domain_motion.lifetime_planning.explicit_v_ms");
    }
    result.upstream_buffer_m = required_positive("domain_motion.lifetime_planning.upstream_buffer_m");
    result.downstream_buffer_m = required_positive("domain_motion.lifetime_planning.downstream_buffer_m");
    result.cross_track_buffer_m = required_positive("domain_motion.lifetime_planning.cross_track_buffer_m");
    result.motion_uncertainty_ms = required_non_negative("domain_motion.lifetime_planning.motion_uncertainty_ms");
    return result;
}

bool preflight_storm_lifetime_before_allocation(const LifetimePlanning& planning,
                                                int                                                 duration_s)
{
    const bool lifetime_planning_enabled = planning.enabled;
    const bool branch_motion_required =
        global_storm_tracking_config.enabled &&
        global_storm_tracking_config.branch_preference != tmv::init::config::StormBranchPreference::Continuity;
    if (!lifetime_planning_enabled && !branch_motion_required)
        return false;
    if (lifetime_planning_enabled &&
        (!std::isfinite(planning.upstream_buffer_m) || planning.upstream_buffer_m <= 0.0 ||
         !std::isfinite(planning.downstream_buffer_m) || planning.downstream_buffer_m <= 0.0 ||
         !std::isfinite(planning.cross_track_buffer_m) || planning.cross_track_buffer_m <= 0.0 ||
         !std::isfinite(planning.motion_uncertainty_ms) || planning.motion_uncertainty_ms < 0.0))
        throw std::invalid_argument("enabled storm lifetime planning requires positive buffers and non-negative uncertainty");

    std::vector<double> z_m(static_cast<std::size_t>(NZ));
    for (int k = 0; k < NZ; ++k)
    {
        z_m[static_cast<std::size_t>(k)] = tmv::grid_coordinates::scalar_center(k, dz);
    }
    auto sounding_source = tmv::init::make_sounding_source(global_sounding_source_config);
    auto sounding        = sounding_source->build(z_m, dz);
    sounding.verify_self_consistent();
    const tmv::init::WindColumn wind = tmv::init::resolve_wind_column(z_m, sounding, global_hodograph_source_config);

    if (branch_motion_required)
    {
        const auto right = tmv::domain::estimate_bunkers_storm_motion(
            z_m, wind.u_ms, wind.v_ms, tmv::domain::StormMotionMethod::BunkersRight);
        const auto left = tmv::domain::estimate_bunkers_storm_motion(
            z_m, wind.u_ms, wind.v_ms, tmv::domain::StormMotionMethod::BunkersLeft);
        tmv::log_info("Storm branch-motion preflight [",
                      tmv::domain::storm_motion_algorithm_id(right.method),
                      "]: right=(",
                      right.velocity_ms.u_ms,
                      ", ",
                      right.velocity_ms.v_ms,
                      ") m/s, left=(",
                      left.velocity_ms.u_ms,
                      ", ",
                      left.velocity_ms.v_ms,
                      ") m/s; PASS");
    }
    if (!lifetime_planning_enabled)
        return true;

    tmv::domain::StormMotionEstimate estimate;
    const std::string motion_source = tmv::init::config::storm_motion_source_name(planning.motion_source);
    if (motion_source == "explicit")
    {
        estimate.method      = tmv::domain::StormMotionMethod::Explicit;
        estimate.velocity_ms = {
            planning.explicit_u_ms,
            planning.explicit_v_ms,
        };
    }
    else
    {
        const tmv::domain::StormMotionMethod method =
            motion_source == "bunkers_right" ? tmv::domain::StormMotionMethod::BunkersRight
            : motion_source == "bunkers_left"
                ? tmv::domain::StormMotionMethod::BunkersLeft
                : throw std::runtime_error("domain_motion.lifetime_planning.motion_source must be "
                                           "bunkers_right, bunkers_left, or explicit");
        estimate = tmv::domain::estimate_bunkers_storm_motion(z_m, wind.u_ms, wind.v_ms, method);
    }

    tmv::domain::HorizontalPoint storm_center;
    switch (global_trigger_source_config.type)
    {
    case tmv::init::TriggerSourceConfig::Type::WarmBubble:
        storm_center = {global_bubble_center_x_m, global_bubble_center_y_m};
        break;
    case tmv::init::TriggerSourceConfig::Type::VortexSeed:
        storm_center = {
            global_trigger_source_config.vortex_seed.center_x_m,
            global_trigger_source_config.vortex_seed.center_y_m,
        };
        break;
    case tmv::init::TriggerSourceConfig::Type::None:
        throw std::runtime_error("storm-lifetime planning requires a warm_bubble or vortex_seed "
                                 "trigger so the initial storm center is defined");
    }

    const PhysicalCellDomain physical = physical_scalar_domain(NR, NTH, NZ, global_stagger_type);
    if (physical.nx() <= 0 || physical.ny() <= 0)
        throw std::runtime_error("storm-lifetime planning requires a non-empty physical horizontal domain");
    const double west_face_m  = tmv::grid_coordinates::face(physical.i_begin, dr);
    const double south_face_m = tmv::grid_coordinates::face(physical.j_begin, dtheta);

    tmv::domain::StormLifetimePlanInput input;
    input.nx                     = static_cast<std::size_t>(physical.nx());
    input.ny                     = static_cast<std::size_t>(physical.ny());
    input.dx_m                   = dr;
    input.dy_m                   = dtheta;
    input.run_time_s             = static_cast<double>(duration_s);
    input.initial_storm_center_m = {
        storm_center.x_m - west_face_m,
        storm_center.y_m - south_face_m,
    };
    input.earth_relative_storm_motion_ms = estimate.velocity_ms;
    input.frame                          = global_domain_frame;
    input.upstream_buffer_m              = planning.upstream_buffer_m;
    input.downstream_buffer_m            = planning.downstream_buffer_m;
    input.cross_track_buffer_m           = planning.cross_track_buffer_m;
    input.motion_uncertainty_ms = planning.motion_uncertainty_ms;

    const tmv::domain::StormLifetimePlan plan = tmv::domain::plan_storm_lifetime(input);
    std::ostringstream                   report;
    report << std::fixed << std::setprecision(1) << "Storm-lifetime preflight ["
           << tmv::domain::storm_motion_method_name(estimate.method) << "/"
           << tmv::domain::storm_motion_algorithm_id(estimate.method) << "]: motion=(" << estimate.velocity_ms.u_ms
           << ", " << estimate.velocity_ms.v_ms << ") m/s, physical domain=" << physical.nx() << "x" << physical.ny()
           << " cells, required=" << plan.required_nx << "x" << plan.required_ny
           << " physical cells (configure grid.nx>=" << plan.required_nx + static_cast<std::size_t>(NR - physical.nx())
           << ", grid.ny>=" << plan.required_ny + static_cast<std::size_t>(NTH - physical.ny())
           << " for current halo topology), recommended trigger center=("
           << (plan.recommended_initial_center_m.x_m + west_face_m) / 1000.0 << ", "
           << (plan.recommended_initial_center_m.y_m + south_face_m) / 1000.0 << ") km";
    if (estimate.method != tmv::domain::StormMotionMethod::Explicit)
    {
        report << ", 0--6 km mean=(" << estimate.mean_wind_0_6km_ms.u_ms << ", " << estimate.mean_wind_0_6km_ms.v_ms
               << ") m/s, shear=(" << estimate.shear_0_6km_ms.u_ms << ", " << estimate.shear_0_6km_ms.v_ms << ") m/s";
    }

    if (plan.contained_for_run)
    {
        tmv::log_info(report.str(), "; PASS");
        return true;
    }

    report << "; FAIL: first envelope risk at " << tmv::domain::boundary_face_name(plan.first_risk_face)
           << " boundary, t=" << plan.first_risk_time_s
           << " s. Increase grid.nx/grid.ny (physical-cell requirements above), "
              "move the trigger to the recommended center, shorten the run, or "
              "supply better motion/buffer evidence.";
    const std::string action = tmv::init::config::lifetime_planning_action_name(planning.action);
    if (action == "warn")
    {
        tmv::log_warn(report.str());
        return true;
    }
    if (action != "error")
        throw std::runtime_error("domain_motion.lifetime_planning.action must be warn or error");
    throw std::runtime_error(report.str());
}

// Legacy documents are normalized here; typed v1 input bypasses these parsers.
void load_radiation(std::unordered_map<std::string, std::string>& config)
{
    // Config loads may share a process. Radiation forcing must never inherit
    // an active scheme or solar geometry from the previous document.
    global_radiation_config = tmv::RadiationConfig{};
    if (config.count("radiation.scheme"))
    {
        const std::string              requested_scheme = config["radiation.scheme"];
        const std::string              canonical        = canonicalize_radiation_scheme_id(requested_scheme);
        const std::vector<std::string> known_schemes    = get_known_radiation_schemes();

        bool known = false;
        for (const auto& scheme : known_schemes)
        {
            if (canonical == scheme)
            {
                known = true;
                break;
            }
        }

        if (known && is_radiation_scheme_implemented(canonical))
        {
            global_radiation_config.scheme_id = canonical;
        }
        else if (known)
        {
            throw std::runtime_error("Requested radiation scheme '" + requested_scheme + "' maps to '" + canonical +
                                     "', which is recognized but not implemented in this build");
        }
        else
        {
            std::string opts;
            for (size_t i = 0; i < known_schemes.size(); ++i)
            {
                if (i > 0)
                    opts += ", ";
                opts += known_schemes[i];
            }
            throw std::runtime_error("Invalid radiation scheme '" + requested_scheme + "'. Known options: " + opts);
        }
    }

    if (config.count("radiation.solar_forcing"))
    {
        global_radiation_config.solar_forcing = to_lower_copy(config["radiation.solar_forcing"]);
        if (global_radiation_config.solar_forcing != "idealized_diurnal")
            throw std::runtime_error("radiation.solar_forcing must be idealized_diurnal");
    }
    const auto load_bounded_solar_value =
        [&](const char* key, double& target, double minimum, double maximum, bool upper_exclusive)
    {
        if (!config.count(key))
            return;
        double     parsed = 0.0;
        const bool within = try_parse_double_value(config[key], parsed) && parsed >= minimum &&
                            (upper_exclusive ? parsed < maximum : parsed <= maximum);
        if (!within)
            throw std::runtime_error(std::string(key) + " is outside its physical range");
        target = parsed;
    };
    load_bounded_solar_value("radiation.latitude_deg", global_radiation_config.latitude_deg, -90.0, 90.0, false);
    load_bounded_solar_value("radiation.declination_deg", global_radiation_config.declination_deg, -23.5, 23.5, false);
    load_bounded_solar_value(
        "radiation.local_solar_time_at_start_h", global_radiation_config.local_solar_time_at_start_h, 0.0, 24.0, true);

    if (config.count("radiation.do_lw"))
    {
        global_radiation_config.do_lw = parse_bool_value(config["radiation.do_lw"]);
    }

    if (config.count("radiation.do_sw"))
    {
        global_radiation_config.do_sw = parse_bool_value(config["radiation.do_sw"]);
    }

    if (config.count("radiation.dt"))
    {
        double parsed = 0.0;
        if (try_parse_double_value(config["radiation.dt"], parsed) && parsed > 0.0)
        {
            global_radiation_config.dt_radiation = parsed;
        }
        else
        {
            throw std::runtime_error("Invalid radiation.dt: expected a positive finite number");
        }
    }
    else if (config.count("radiation.dt_radiation"))
    {
        double parsed = 0.0;
        if (try_parse_double_value(config["radiation.dt_radiation"], parsed) && parsed > 0.0)
        {
            global_radiation_config.dt_radiation = parsed;
        }
        else
        {
            throw std::runtime_error("Invalid radiation.dt_radiation: expected a positive finite number");
        }
    }

    if (config.count("radiation.clear_sky_only"))
    {
        global_radiation_config.clear_sky_only = parse_bool_value(config["radiation.clear_sky_only"]);
    }
    if (config.count("radiation.simple_grey_optical_model"))
    {
        global_radiation_config.simple_grey_optical_model =
            to_lower_copy(strip_wrapping_quotes(
                config["radiation.simple_grey_optical_model"]));
    }
    if (config.count("radiation.use_subcycling"))
    {
        global_radiation_config.use_subcycling = parse_bool_value(config["radiation.use_subcycling"]);
    }
    if (config.count("radiation.enable_diagnostics"))
    {
        global_radiation_config.enable_diagnostics = parse_bool_value(config["radiation.enable_diagnostics"]);
    }
    if (config.count("radiation.subcycle_count"))
    {
        int parsed = 0;
        if (!try_parse_positive_int_value(
                config["radiation.subcycle_count"], parsed) || parsed > 64)
            throw std::runtime_error(
                "Invalid radiation.subcycle_count: expected an integer in [1,64]");
        global_radiation_config.subcycle_count =
            static_cast<std::size_t>(parsed);
    }

    if (config.count("radiation.tau_lw_ref"))
    {
        double parsed = 0.0;
        if (try_parse_double_value(config["radiation.tau_lw_ref"], parsed) && parsed >= 0.0)
        {
            global_radiation_config.tau_lw_ref = parsed;
        }
        else
        {
            throw std::runtime_error("Invalid radiation.tau_lw_ref: expected a non-negative finite number");
        }
    }

    if (config.count("radiation.tau_sw_ref"))
    {
        double parsed = 0.0;
        if (try_parse_double_value(config["radiation.tau_sw_ref"], parsed) && parsed >= 0.0)
        {
            global_radiation_config.tau_sw_ref = parsed;
        }
        else
        {
            throw std::runtime_error("Invalid radiation.tau_sw_ref: expected a non-negative finite number");
        }
    }

    if (config.count("radiation.n_lw"))
    {
        double parsed = 0.0;
        if (try_parse_double_value(config["radiation.n_lw"], parsed) && parsed > 0.0)
        {
            global_radiation_config.n_lw = parsed;
        }
        else
        {
            throw std::runtime_error("Invalid radiation.n_lw: expected a positive finite number");
        }
    }
    if (config.count("radiation.n_sw"))
    {
        double parsed = 0.0;
        if (try_parse_double_value(config["radiation.n_sw"], parsed) && parsed > 0.0)
        {
            global_radiation_config.n_sw = parsed;
        }
        else
        {
            throw std::runtime_error("Invalid radiation.n_sw: expected a positive finite number");
        }
    }

    const auto load_simple_grey_absorption = [&](const char* key,
                                                 double& target)
    {
        if (!config.count(key))
            return;
        double parsed = 0.0;
        if (!try_parse_double_value(config[key], parsed) || parsed < 0.0)
            throw std::runtime_error(
                std::string("Invalid ") + key +
                ": expected a non-negative finite number in m2/kg");
        target = parsed;
    };
    load_simple_grey_absorption(
        "radiation.simple_grey_lw_vapor_absorption_m2kg",
        global_radiation_config.simple_grey_lw_vapor_absorption_m2kg);
    load_simple_grey_absorption(
        "radiation.simple_grey_lw_liquid_absorption_m2kg",
        global_radiation_config.simple_grey_lw_liquid_absorption_m2kg);
    load_simple_grey_absorption(
        "radiation.simple_grey_lw_ice_absorption_m2kg",
        global_radiation_config.simple_grey_lw_ice_absorption_m2kg);
    load_simple_grey_absorption(
        "radiation.simple_grey_sw_vapor_absorption_m2kg",
        global_radiation_config.simple_grey_sw_vapor_absorption_m2kg);
    load_simple_grey_absorption(
        "radiation.simple_grey_sw_liquid_absorption_m2kg",
        global_radiation_config.simple_grey_sw_liquid_absorption_m2kg);
    load_simple_grey_absorption(
        "radiation.simple_grey_sw_ice_absorption_m2kg",
        global_radiation_config.simple_grey_sw_ice_absorption_m2kg);

    const auto load_rrtmg_value = [&](const char* key, double& target, double minimum, double maximum)
    {
        if (!config.count(key))
            return;
        double parsed = 0.0;
        if (!try_parse_double_value(config[key], parsed) || parsed < minimum || parsed > maximum)
            throw std::runtime_error(std::string("Invalid ") + key + ": outside the RRTMG physical range");
        target = parsed;
    };
    using namespace tmv::rrtmg_config_limits;
    load_rrtmg_value("radiation.rrtmg_ozone_vmr", global_radiation_config.rrtmg_ozone_vmr, 0.0, ozone_vmr_max);
    load_rrtmg_value("radiation.rrtmg_co2_vmr", global_radiation_config.rrtmg_co2_vmr, 0.0, co2_vmr_max);
    load_rrtmg_value("radiation.rrtmg_ch4_vmr", global_radiation_config.rrtmg_ch4_vmr, 0.0, ch4_vmr_max);
    load_rrtmg_value("radiation.rrtmg_n2o_vmr", global_radiation_config.rrtmg_n2o_vmr, 0.0, n2o_vmr_max);
    load_rrtmg_value("radiation.rrtmg_co_vmr", global_radiation_config.rrtmg_co_vmr, 0.0, co_vmr_max);
    load_rrtmg_value("radiation.rrtmg_o2_vmr", global_radiation_config.rrtmg_o2_vmr, 0.0, o2_vmr_max);
    load_rrtmg_value("radiation.rrtmg_n2_vmr", global_radiation_config.rrtmg_n2_vmr, 0.0, n2_vmr_max);
    load_rrtmg_value(
        "radiation.rrtmg_cfc11_vmr", global_radiation_config.rrtmg_cfc11_vmr, 0.0, halocarbon_vmr_max);
    load_rrtmg_value(
        "radiation.rrtmg_cfc12_vmr", global_radiation_config.rrtmg_cfc12_vmr, 0.0, halocarbon_vmr_max);
    load_rrtmg_value(
        "radiation.rrtmg_cfc22_vmr", global_radiation_config.rrtmg_cfc22_vmr, 0.0, halocarbon_vmr_max);
    load_rrtmg_value(
        "radiation.rrtmg_ccl4_vmr", global_radiation_config.rrtmg_ccl4_vmr, 0.0, halocarbon_vmr_max);
    load_rrtmg_value("radiation.rrtmg_liquid_effective_radius_um",
                     global_radiation_config.rrtmg_liquid_effective_radius_um,
                     liquid_effective_radius_min_um,
                     liquid_effective_radius_max_um);
    load_rrtmg_value("radiation.rrtmg_ice_effective_radius_um",
                     global_radiation_config.rrtmg_ice_effective_radius_um,
                     ice_effective_radius_min_um,
                     ice_effective_radius_max_um);
    if (global_radiation_config.scheme_id == "rrtmg" && !config.count("radiation.rrtmg_ozone_vmr"))
        throw std::runtime_error(
            "radiation.rrtmg_ozone_vmr is required for RRTMG; Nimbus does not supply a hidden ozone climatology");

    const auto load_terrain_sw_count = [&](const char* key,
                                            std::size_t& target,
                                            int minimum,
                                            int maximum,
                                            bool multiple_of_four)
    {
        if (!config.count(key))
            return;
        int parsed = 0;
        if (!try_parse_int_value(config[key], parsed)
            || parsed < minimum || parsed > maximum
            || (multiple_of_four && parsed % 4 != 0))
            throw std::runtime_error(
                std::string("Invalid ") + key
                + ": outside the qualified bounded integer range");
        target = static_cast<std::size_t>(parsed);
    };
    load_terrain_sw_count(
        "radiation.terrain_sw_cosine_nodes",
        global_radiation_config.terrain_sw_cosine_nodes,
        global_radiation_config.scheme_id == "rrtmg" ? 1 : 4,
        32, false);
    load_terrain_sw_count(
        "radiation.terrain_sw_azimuth_nodes",
        global_radiation_config.terrain_sw_azimuth_nodes,
        global_radiation_config.scheme_id == "rrtmg" ? 4 : 8,
        128, true);
    load_terrain_sw_count(
        "radiation.terrain_sw_maximum_bounces",
        global_radiation_config.terrain_sw_maximum_bounces, 1, 256, false);
    if (config.count("radiation.terrain_sw_relative_residual_tolerance"))
    {
        double parsed = 0.0;
        if (!try_parse_double_value(
                config["radiation.terrain_sw_relative_residual_tolerance"],
                parsed)
            || parsed < 0.0 || parsed > 1.0e-6)
            throw std::runtime_error(
                "Invalid radiation.terrain_sw_relative_residual_tolerance");
        global_radiation_config.terrain_sw_relative_residual_tolerance =
            parsed;
    }
    if (config.count("radiation.terrain_sw_absolute_residual_tolerance_w"))
    {
        double parsed = 0.0;
        if (!try_parse_double_value(
                config["radiation.terrain_sw_absolute_residual_tolerance_w"],
                parsed)
            || parsed < 0.0)
            throw std::runtime_error(
                "Invalid radiation.terrain_sw_absolute_residual_tolerance_w");
        global_radiation_config.terrain_sw_absolute_residual_tolerance_w =
            parsed;
    }
}

// surface.* -> global_surface_config. Leaf: no throw, no cross-section ordering
// dependency. (surface.scheme / apply_fluxes are applied on the v1 path.)
void load_surface(std::unordered_map<std::string, std::string>& config)
{
    if (config.count("surface.z0m"))
    {
        double parsed = 0.0;
        if (try_parse_double_value(config["surface.z0m"], parsed) && parsed > 0.0)
        {
            global_surface_config.z0m = parsed;
        }
        else
        {
            throw std::runtime_error("Invalid surface.z0m: expected a positive finite number");
        }
    }

    if (config.count("surface.z0h"))
    {
        double parsed = 0.0;
        if (try_parse_double_value(config["surface.z0h"], parsed) && parsed > 0.0)
        {
            global_surface_config.z0h = parsed;
        }
        else
        {
            throw std::runtime_error("Invalid surface.z0h: expected a positive finite number");
        }
    }

    if (config.count("surface.albedo"))
    {
        double parsed = 0.0;
        if (try_parse_double_value(config["surface.albedo"], parsed) && parsed >= 0.0 && parsed <= 1.0)
        {
            global_surface_config.albedo = parsed;
        }
        else
        {
            throw std::runtime_error("Invalid surface.albedo: expected a finite number in [0,1]");
        }
    }

    if (config.count("surface.emissivity"))
    {
        double parsed = 0.0;
        if (try_parse_double_value(config["surface.emissivity"], parsed) && parsed >= 0.0 && parsed <= 1.0)
        {
            global_surface_config.emissivity = parsed;
        }
        else
        {
            throw std::runtime_error("Invalid surface.emissivity: expected a finite number in [0,1]");
        }
    }

    if (config.count("surface.tsfc"))
    {
        double parsed = 0.0;
        if (try_parse_double_value(config["surface.tsfc"], parsed) &&
            (parsed == -1.0 || (parsed >= 150.0 && parsed <= 400.0)))
        {
            global_surface_config.Tsfc = parsed;
        }
        else
        {
            throw std::runtime_error("Invalid surface.tsfc: expected -1 or a finite temperature in [150,400] K");
        }
    }
    else if (config.count("surface.Tsfc"))
    {
        double parsed = 0.0;
        if (try_parse_double_value(config["surface.Tsfc"], parsed) &&
            (parsed == -1.0 || (parsed >= 150.0 && parsed <= 400.0)))
        {
            global_surface_config.Tsfc = parsed;
        }
        else
        {
            throw std::runtime_error("Invalid surface.Tsfc: expected -1 or a finite temperature in [150,400] K");
        }
    }

    if (config.count("surface.qsfc"))
    {
        double parsed = 0.0;
        if (try_parse_double_value(config["surface.qsfc"], parsed) &&
            (parsed == -1.0 || (parsed >= 0.0 && parsed <= 0.1)))
        {
            global_surface_config.qsfc = parsed;
        }
        else
        {
            throw std::runtime_error("Invalid surface.qsfc: expected -1 or a finite mixing ratio in [0,0.1]");
        }
    }
    else if (config.count("surface.qv_sfc"))
    {
        double parsed = 0.0;
        if (try_parse_double_value(config["surface.qv_sfc"], parsed) &&
            (parsed == -1.0 || (parsed >= 0.0 && parsed <= 0.1)))
        {
            global_surface_config.qsfc = parsed;
        }
        else
        {
            throw std::runtime_error("Invalid surface.qv_sfc: expected -1 or a finite mixing ratio in [0,0.1]");
        }
    }
}

// boundary_layer.* -> global_boundary_layer_config. Leaf: no throw, no
// cross-section ordering dependency. The parse_surface_layer_id_key lambda
// captures [&], so it re-captures this function's `config` parameter and moves
// verbatim.
void load_boundary_layer(std::unordered_map<std::string, std::string>& config)
{
    if (config.count("boundary_layer.scheme"))
    {
        global_boundary_layer_config.scheme_id = to_lower_copy(config["boundary_layer.scheme"]);
    }

    if (config.count("boundary_layer.apply_surface_fluxes"))
    {
        global_boundary_layer_config.apply_surface_fluxes =
            parse_bool_value(config["boundary_layer.apply_surface_fluxes"]);
    }

    if (config.count("boundary_layer.clear_sky_only"))
    {
        global_boundary_layer_config.clear_sky_only = parse_bool_value(config["boundary_layer.clear_sky_only"]);
    }
    if (config.count("boundary_layer.enable_tke"))
    {
        global_boundary_layer_config.enable_tke = parse_bool_value(config["boundary_layer.enable_tke"]);
    }
    if (config.count("boundary_layer.pbl_top_method"))
    {
        std::string method;
        if (parse_boundary_layer_top_method(config["boundary_layer.pbl_top_method"], method))
        {
            global_boundary_layer_config.pbl_top_method = method;
        }
        else
        {
            throw std::runtime_error("Invalid boundary_layer.pbl_top_method: expected one of ri, theta, tke");
        }
    }

    auto parse_surface_layer_id_key = [&](const char* key)
    {
        if (!config.count(key))
        {
            return;
        }
        std::string parsed;
        if (parse_boundary_layer_surface_layer_id(config[key], parsed))
        {
            global_boundary_layer_config.surface_layer_id = parsed;
        }
        else
        {
            throw std::runtime_error(std::string("Invalid ") + key + ": expected one of bulk, monin_obukhov");
        }
    };
    parse_surface_layer_id_key("boundary_layer.surface_layer");
    parse_surface_layer_id_key("boundary_layer.surface_layer_id");

    if (config.count("boundary_layer.dt"))
    {
        double parsed = 0.0;
        if (try_parse_double_value(config["boundary_layer.dt"], parsed) && parsed > 0.0)
        {
            global_boundary_layer_config.dt_pbl = parsed;
        }
        else
        {
            throw std::runtime_error("Invalid boundary_layer.dt: expected a positive finite number");
        }
    }
    else if (config.count("boundary_layer.dt_pbl"))
    {
        double parsed = 0.0;
        if (try_parse_double_value(config["boundary_layer.dt_pbl"], parsed) && parsed > 0.0)
        {
            global_boundary_layer_config.dt_pbl = parsed;
        }
        else
        {
            throw std::runtime_error("Invalid boundary_layer.dt_pbl: expected a positive finite number");
        }
    }

    if (config.count("boundary_layer.pbl_max_height"))
    {
        double parsed = 0.0;
        if (try_parse_double_value(config["boundary_layer.pbl_max_height"], parsed) && parsed > 0.0)
        {
            global_boundary_layer_config.pbl_max_height = parsed;
        }
        else
        {
            throw std::runtime_error("Invalid boundary_layer.pbl_max_height: expected a positive finite number");
        }
    }

    if (config.count("boundary_layer.min_ustar"))
    {
        double parsed = 0.0;
        if (try_parse_double_value(config["boundary_layer.min_ustar"], parsed) && parsed >= 0.0)
        {
            global_boundary_layer_config.min_ustar = parsed;
        }
        else
        {
            throw std::runtime_error("Invalid boundary_layer.min_ustar: expected a non-negative finite number");
        }
    }
}

// microphysics.* -> selected scheme and its validated factory configuration.
void load_microphysics(std::unordered_map<std::string, std::string>& config)
{
    global_microphysics_config = {};
    if (config.count("microphysics.scheme"))
    {
        const std::string& requested  = config["microphysics.scheme"];
        const auto*        descriptor = tmv::find_scheme_descriptor(get_microphysics_scheme_descriptors(), requested);
        if (descriptor && descriptor->constructible)
        {
            global_microphysics_scheme = descriptor->id;
        }
        else
        {
            std::string opts;
            const auto  valid_schemes = get_available_microphysics_schemes();
            for (size_t i = 0; i < valid_schemes.size(); ++i)
            {
                if (i > 0)
                    opts += ", ";
                opts += valid_schemes[i];
            }
            throw std::runtime_error("Invalid microphysics scheme '" + requested + "'. Valid options: " + opts);
        }
    }

    const auto load_non_negative = [&](const char* key, double& destination)
    {
        if (!config.count(key))
            return;
        double parsed = 0.0;
        if (try_parse_double_value(config[key], parsed) && parsed >= 0.0)
        {
            destination = parsed;
            return;
        }
        throw std::runtime_error(std::string("Invalid ") + key + ": expected a finite non-negative number");
    };
    const auto load_positive = [&](const char* key, double& destination)
    {
        if (!config.count(key)) return;
        double parsed = 0.0;
        if (try_parse_double_value(config[key], parsed) && parsed > 0.0)
        {
            destination = parsed;
            return;
        }
        throw std::runtime_error(
            std::string("Invalid ") + key + ": expected a finite positive number");
    };
    const auto load_boolean = [&](const char* key, bool& destination)
    {
        if (!config.count(key))
            return;
        const std::string normalized = to_lower_copy(strip_wrapping_quotes(config[key]));
        if (normalized == "true" || normalized == "1" || normalized == "yes" || normalized == "on")
        {
            destination = true;
            return;
        }
        if (normalized == "false" || normalized == "0" || normalized == "no" || normalized == "off")
        {
            destination = false;
            return;
        }
        throw std::runtime_error(std::string("Invalid ") + key + ": expected true or false");
    };

    load_non_negative("microphysics.shared.cloud_water_threshold_kgkg",
                      global_microphysics_config.shared.cloud_water_threshold_kgkg);
    load_non_negative("microphysics.shared.autoconversion_rate_s_inv",
                      global_microphysics_config.shared.autoconversion_rate_s_inv);
    load_non_negative("microphysics.kessler.accretion_coefficient",
                      global_microphysics_config.kessler.accretion_coefficient);
    load_non_negative("microphysics.kessler.evaporation_efficiency",
                      global_microphysics_config.kessler.evaporation_efficiency);
    load_non_negative("microphysics.lin.accretion_coefficient", global_microphysics_config.lin.accretion_coefficient);
    load_non_negative("microphysics.lin.evaporation_rate_s_inv", global_microphysics_config.lin.evaporation_rate_s_inv);
    load_non_negative("microphysics.thompson.evaporation_rate_s_inv",
                      global_microphysics_config.thompson.evaporation_rate_s_inv);
    load_positive("microphysics.thompson.water_friendly_aerosol_surface_m3",
                  global_microphysics_config.thompson.water_friendly_aerosol_surface_m3);
    load_positive("microphysics.thompson.ice_friendly_aerosol_surface_m3",
                  global_microphysics_config.thompson.ice_friendly_aerosol_surface_m3);
    load_non_negative("microphysics.milbrandt.evaporation_rate_s_inv",
                      global_microphysics_config.milbrandt.evaporation_rate_s_inv);
    load_boolean("microphysics.milbrandt.triple_moment", global_microphysics_config.milbrandt.triple_moment);
    if (global_microphysics_config.milbrandt.triple_moment &&
        !tmv::microphysics::kMilbrandtTripleMomentPublicAdmissionOpen)
    {
        throw std::runtime_error(
            "microphysics.milbrandt.triple_moment is retained but engineering incomplete and not admitted");
    }
    load_boolean("microphysics.milbrandt.hail_processes", global_microphysics_config.milbrandt.hail_processes);
}

// turbulence.* -> global_turbulence_config. Leaf with three inline capture-by-ref
// lambdas (dt / filter_width / stability_correction key parsers) that re-capture
// this function's `config`; no ordering dependency.
void load_turbulence(std::unordered_map<std::string, std::string>& config)
{
    if (config.count("turbulence.scheme"))
    {
        const std::string& requested  = config["turbulence.scheme"];
        const auto*        descriptor = tmv::find_scheme_descriptor(get_turbulence_scheme_descriptors(), requested);
        if (!descriptor || !descriptor->constructible)
        {
            throw std::runtime_error("Invalid turbulence scheme '" + requested + "'.");
        }
        global_turbulence_config.scheme_id = descriptor->id;
    }

    if (config.count("turbulence.mode"))
    {
        std::string mode;
        if (parse_turbulence_mode(config["turbulence.mode"], mode))
        {
            global_turbulence_config.mode = mode;
        }
        else
        {
            throw std::runtime_error("Invalid turbulence.mode '" + config["turbulence.mode"] +
                                     "'. Valid values: 2d, 3d, horizontal_only, vertical_only");
        }
    }

    auto parse_turbulence_dt_key = [&](const char* key)
    {
        if (!config.count(key))
        {
            return;
        }

        double parsed = 0.0;
        if (try_parse_double_value(config[key], parsed) && parsed > 0.0)
        {
            global_turbulence_config.dt_sgs = parsed;
        }
        else
        {
            throw std::runtime_error(std::string("Invalid ") + key + " '" + config[key] +
                                     "': expected a positive finite number");
        }
    };
    parse_turbulence_dt_key("turbulence.dt");
    parse_turbulence_dt_key("turbulence.dt_sgs");

    if (config.count("turbulence.Cs"))
    {
        double parsed = 0.0;
        if (try_parse_double_value(config["turbulence.Cs"], parsed) && parsed >= 0.0)
        {
            global_turbulence_config.Cs = parsed;
        }
        else
        {
            throw std::runtime_error("Invalid turbulence.Cs '" + config["turbulence.Cs"] +
                                     "': expected a non-negative finite number");
        }
    }

    if (config.count("turbulence.Pr_t"))
    {
        double parsed = 0.0;
        if (try_parse_double_value(config["turbulence.Pr_t"], parsed) && parsed > 0.0)
        {
            global_turbulence_config.Pr_t = parsed;
        }
        else
        {
            throw std::runtime_error("Invalid turbulence.Pr_t '" + config["turbulence.Pr_t"] +
                                     "': expected a positive finite number");
        }
    }

    if (config.count("turbulence.Sc_t"))
    {
        double parsed = 0.0;
        if (try_parse_double_value(config["turbulence.Sc_t"], parsed) && parsed > 0.0)
        {
            global_turbulence_config.Sc_t = parsed;
        }
        else
        {
            throw std::runtime_error("Invalid turbulence.Sc_t '" + config["turbulence.Sc_t"] +
                                     "': expected a positive finite number");
        }
    }

    if (config.count("turbulence.nu_t_max"))
    {
        double parsed = 0.0;
        if (try_parse_double_value(config["turbulence.nu_t_max"], parsed) && parsed >= 0.0)
        {
            global_turbulence_config.nu_t_max = parsed;
        }
        else
        {
            throw std::runtime_error("Invalid turbulence.nu_t_max '" + config["turbulence.nu_t_max"] +
                                     "': expected a non-negative finite number");
        }
    }

    if (config.count("turbulence.K_max"))
    {
        double parsed = 0.0;
        if (try_parse_double_value(config["turbulence.K_max"], parsed) && parsed >= 0.0)
        {
            global_turbulence_config.K_max = parsed;
        }
        else
        {
            throw std::runtime_error("Invalid turbulence.K_max '" + config["turbulence.K_max"] +
                                     "': expected a non-negative finite number");
        }
    }

    auto parse_filter_width_key = [&](const char* key)
    {
        if (!config.count(key))
        {
            return;
        }

        std::string parsed;
        if (parse_turbulence_filter_width(config[key], parsed))
        {
            global_turbulence_config.filter_width = parsed;
        }
        else
        {
            throw std::runtime_error(std::string("Invalid ") + key + " '" + config[key] +
                                     "': expected one of dx, cubic_root, user");
        }
    };
    parse_filter_width_key("turbulence.filter_width");
    parse_filter_width_key("turbulence.smagorinsky.filter_width");
    if (config.count("turbulence.user_filter_width_m"))
    {
        double parsed = 0.0;
        if (!try_parse_double_value(
                config["turbulence.user_filter_width_m"], parsed) ||
            parsed < 0.0)
            throw std::runtime_error(
                "Invalid turbulence.user_filter_width_m: expected a "
                "non-negative finite number");
        global_turbulence_config.user_filter_width_m = parsed;
    }

    auto parse_stability_correction_key = [&](const char* key)
    {
        if (!config.count(key))
        {
            return;
        }

        std::string parsed;
        if (parse_turbulence_stability_correction(config[key], parsed))
        {
            global_turbulence_config.stability_correction = parsed;
        }
        else
        {
            throw std::runtime_error(std::string("Invalid ") + key + " '" + config[key] +
                                     "': expected one of none, ri, tke_based");
        }
    };
    parse_stability_correction_key("turbulence.stability_correction");
    parse_stability_correction_key("turbulence.smagorinsky.stability_correction");

    if (config.count("turbulence.dynamic_Cs"))
    {
        global_turbulence_config.dynamic_Cs = parse_bool_value(config["turbulence.dynamic_Cs"]);
    }
    else if (config.count("turbulence.dynamic_cs"))
    {
        global_turbulence_config.dynamic_Cs = parse_bool_value(config["turbulence.dynamic_cs"]);
    }
    else if (config.count("turbulence.smagorinsky.dynamic_procedure"))
    {
        global_turbulence_config.dynamic_Cs = parse_bool_value(config["turbulence.smagorinsky.dynamic_procedure"]);
    }

    if (config.count("turbulence.enable_moist_buoyancy"))
    {
        global_turbulence_config.enable_moist_buoyancy = parse_bool_value(config["turbulence.enable_moist_buoyancy"]);
    }
    if (config.count("turbulence.enable_diagnostics"))
    {
        global_turbulence_config.enable_diagnostics = parse_bool_value(config["turbulence.enable_diagnostics"]);
    }
}

// dynamics.scheme -> global_dynamics_scheme_name (the dynamics factory owns alias
// resolution; the loader just records the requested string). Leaf.
void load_dynamics(std::unordered_map<std::string, std::string>& config)
{
    if (config.count("dynamics.scheme"))
    {
        global_dynamics_scheme_name = config["dynamics.scheme"];
    }
}

// numerics.advection[.scheme] -> global_advection_config. Leaf.
void load_advection(std::unordered_map<std::string, std::string>& config)
{
    if (config.count("numerics.advection"))
    {
        global_advection_config.scheme_id = config["numerics.advection"];
    }
    if (config.count("numerics.advection.scheme"))
    {
        global_advection_config.scheme_id = config["numerics.advection.scheme"];
    }
}

// numerics.diffusion.* -> global_diffusion_config. Leaf.
void load_diffusion(std::unordered_map<std::string, std::string>& config)
{
    bool scheme_configured = false;
    if (config.count("numerics.diffusion"))
    {
        global_diffusion_config.scheme_id = config["numerics.diffusion"];
        scheme_configured                 = true;
    }
    if (config.count("numerics.diffusion.scheme"))
    {
        global_diffusion_config.scheme_id = config["numerics.diffusion.scheme"];
        scheme_configured                 = true;
    }
    if (scheme_configured)
    {
        global_diffusion_config.scheme_id = to_lower_copy(global_diffusion_config.scheme_id);
        if (global_diffusion_config.scheme_id != "explicit" && global_diffusion_config.scheme_id != "implicit")
        {
            throw std::runtime_error("Invalid numerics.diffusion.scheme '" + global_diffusion_config.scheme_id +
                                     "'. Registered runtime schemes: explicit, implicit. To "
                                     "disable diffusion, select explicit with K_h=0 and K_v=0.");
        }
    }
    if (config.count("numerics.diffusion.operator_order"))
    {
        int parsed = 0;
        if (!try_parse_positive_int_value(config["numerics.diffusion.operator_order"], parsed) || parsed != 2)
        {
            throw std::runtime_error("numerics.diffusion.operator_order must be the implemented "
                                     "runtime order 2; got '" +
                                     config["numerics.diffusion.operator_order"] + "'");
        }
        global_diffusion_config.operator_order = parsed;
    }
    if (config.count("numerics.diffusion.operator_type"))
    {
        global_diffusion_config.operator_type = config["numerics.diffusion.operator_type"];
    }
    if (config.count("numerics.diffusion.apply_to"))
    {
        global_diffusion_config.apply_to = config["numerics.diffusion.apply_to"];
    }
    if (config.count("numerics.diffusion.implicit_dim"))
    {
        global_diffusion_config.implicit_dim = to_lower_copy(config["numerics.diffusion.implicit_dim"]);
    }
    if (config.count("numerics.diffusion.K_h"))
    {
        double parsed = 0.0;
        if (try_parse_double_value(config["numerics.diffusion.K_h"], parsed) && parsed >= 0.0)
        {
            global_diffusion_config.K_h = parsed;
        }
        else
        {
            warn_invalid_config_value(
                "numerics.diffusion.K_h", config["numerics.diffusion.K_h"], "a non-negative finite number");
        }
    }
    if (config.count("numerics.diffusion.K_v"))
    {
        double parsed = 0.0;
        if (try_parse_double_value(config["numerics.diffusion.K_v"], parsed) && parsed >= 0.0)
        {
            global_diffusion_config.K_v = parsed;
        }
        else
        {
            warn_invalid_config_value(
                "numerics.diffusion.K_v", config["numerics.diffusion.K_v"], "a non-negative finite number");
        }
    }
    if (config.count("numerics.diffusion.dt_diffusion"))
    {
        double parsed = 0.0;
        if (try_parse_double_value(config["numerics.diffusion.dt_diffusion"], parsed) && parsed > 0.0)
        {
            global_diffusion_config.dt_diffusion = parsed;
        }
        else
        {
            warn_invalid_config_value("numerics.diffusion.dt_diffusion",
                                      config["numerics.diffusion.dt_diffusion"],
                                      "a positive finite number");
        }
    }
    if (config.count("numerics.diffusion.use_variable_K"))
    {
        const bool requested = parse_bool_value(config["numerics.diffusion.use_variable_K"]);
        if (requested)
            throw std::invalid_argument(
                "numerics.diffusion.use_variable_K is obsolete; select "
                "numerics.diffusion.coefficient_source=linear_height");
    }
    if (config.count("numerics.diffusion.coefficient_source"))
    {
        global_diffusion_config.coefficient_source =
            to_lower_copy(config["numerics.diffusion.coefficient_source"]);
    }
    const bool linear_height =
        global_diffusion_config.coefficient_source == "linear_height";
    if (global_diffusion_config.coefficient_source != "constant" &&
        !linear_height)
        throw std::invalid_argument(
            "numerics.diffusion.coefficient_source must be constant or linear_height");
    const auto load_profile_value = [&](const char* key, double& destination)
    {
        if (!config.count(key)) return;
        double parsed = 0.0;
        if (!try_parse_double_value(config[key], parsed) ||
            !std::isfinite(parsed) || parsed < 0.0)
            throw std::invalid_argument(std::string(key) +
                " must be a non-negative finite number");
        destination = parsed;
    };
    load_profile_value("numerics.diffusion.K_v_surface",
                       global_diffusion_config.K_v_surface);
    load_profile_value("numerics.diffusion.K_v_top",
                       global_diffusion_config.K_v_top);
    if (linear_height)
    {
        if (to_lower_copy(global_diffusion_config.scheme_id) != "explicit" ||
            global_diffusion_config.K_v != 0.0)
            throw std::invalid_argument(
                "linear-height variable diffusion requires explicit scheme "
                "and K_v=0");
        if (!config.count("numerics.diffusion.K_v_surface") ||
            !config.count("numerics.diffusion.K_v_top"))
            throw std::invalid_argument(
                "linear-height variable diffusion requires K_v_surface and K_v_top");
    }
    else if (config.count("numerics.diffusion.K_v_surface") ||
             config.count("numerics.diffusion.K_v_top"))
        throw std::invalid_argument(
            "K_v_surface and K_v_top require linear_height coefficient source");
    global_diffusion_config.use_variable_K = linear_height;
}

// numerics.time_stepping.* -> global_time_stepping_config, including the
// dt_min/dt_max guard-swap. Leaf.
void load_time_stepping(std::unordered_map<std::string, std::string>& config)
{
    if (config.count("numerics.time_stepping"))
    {
        global_time_stepping_config.scheme_id = config["numerics.time_stepping"];
    }
    if (config.count("numerics.time_stepping.scheme"))
    {
        global_time_stepping_config.scheme_id = config["numerics.time_stepping.scheme"];
    }
    if (config.count("numerics.time_stepping.split_acoustic"))
    {
        global_time_stepping_config.split_acoustic = parse_bool_value(config["numerics.time_stepping.split_acoustic"]);
    }
    if (config.count("numerics.time_stepping.n_acoustic_substeps"))
    {
        int parsed = 0;
        if (try_parse_non_negative_int_value(config["numerics.time_stepping.n_acoustic_substeps"], parsed))
        {
            // 0 means fully automatic. A positive value is a requested
            // minimum; SplitExplicitScheme promotes it when the runtime sound
            // speed and grid spacing require more substeps for acoustic CFL.
            global_time_stepping_config.n_acoustic_substeps = parsed;
        }
        else
        {
            warn_invalid_config_value("numerics.time_stepping.n_acoustic_substeps",
                                      config["numerics.time_stepping.n_acoustic_substeps"],
                                      "a non-negative integer (0 = auto; positive = requested minimum)");
        }
    }
    if (config.count("numerics.time_stepping.physics_splitting"))
    {
        global_time_stepping_config.physics_splitting = config["numerics.time_stepping.physics_splitting"];
    }
    if (config.count("numerics.time_stepping.cfl_safety"))
    {
        double parsed = 0.0;
        if (try_parse_double_value(config["numerics.time_stepping.cfl_safety"], parsed) && parsed > 0.0)
        {
            global_time_stepping_config.cfl_safety = parsed;
        }
        else
        {
            warn_invalid_config_value("numerics.time_stepping.cfl_safety",
                                      config["numerics.time_stepping.cfl_safety"],
                                      "a positive finite number");
        }
    }
    if (config.count("numerics.time_stepping.adaptive_dt"))
    {
        global_time_stepping_config.adaptive_dt = parse_bool_value(config["numerics.time_stepping.adaptive_dt"]);
    }
    if (config.count("numerics.time_stepping.dt_min"))
    {
        double parsed = 0.0;
        if (try_parse_double_value(config["numerics.time_stepping.dt_min"], parsed) && parsed > 0.0)
        {
            global_time_stepping_config.dt_min = parsed;
        }
        else
        {
            warn_invalid_config_value(
                "numerics.time_stepping.dt_min", config["numerics.time_stepping.dt_min"], "a positive finite number");
        }
    }
    if (config.count("numerics.time_stepping.dt_max"))
    {
        double parsed = 0.0;
        if (try_parse_double_value(config["numerics.time_stepping.dt_max"], parsed) && parsed > 0.0)
        {
            global_time_stepping_config.dt_max = parsed;
        }
        else
        {
            warn_invalid_config_value(
                "numerics.time_stepping.dt_max", config["numerics.time_stepping.dt_max"], "a positive finite number");
        }
    }
    if (global_time_stepping_config.dt_min > global_time_stepping_config.dt_max)
    {
        std::swap(global_time_stepping_config.dt_min, global_time_stepping_config.dt_max);
    }
}

// nested.* -> nested_config (the dormant nested-grid feature; preserved). Leaf.
void load_nested(std::unordered_map<std::string, std::string>& config)
{
    auto& nested = global_simulation_storage.nested.config;
    runtime_configuration.nesting = {};
    auto& relationship = runtime_configuration.nesting;

    const auto take_positive_integer = [&](const char* key, int& destination)
    {
        if (!config.count(key))
            return;
        if (!try_parse_positive_int_value(config[key], destination))
            throw std::runtime_error(std::string(key) + " must be a positive integer");
    };
    take_positive_integer("domain_motion.nesting.parent_ratio_x",
                          relationship.parent_ratio_x);
    take_positive_integer("domain_motion.nesting.parent_ratio_y",
                          relationship.parent_ratio_y);
    take_positive_integer("domain_motion.nesting.parent_time_step_ratio",
                          relationship.parent_time_step_ratio);
    if (config.count("domain_motion.nesting.enabled"))
        relationship.enabled = parse_bool_value(config["domain_motion.nesting.enabled"]);
    const auto take_nonnegative_integer = [&](const char* key, int& destination)
    {
        if (!config.count(key))
            return;
        if (!try_parse_non_negative_int_value(config[key], destination))
            throw std::runtime_error(std::string(key) + " must be a non-negative integer");
    };
    take_nonnegative_integer("domain_motion.nesting.child_origin_parent_cell_i",
                             relationship.child_origin_parent_cell_i);
    take_nonnegative_integer("domain_motion.nesting.child_origin_parent_cell_j",
                             relationship.child_origin_parent_cell_j);
    take_nonnegative_integer("domain_motion.nesting.child_cells_x",
                             relationship.child_cells_x);
    take_nonnegative_integer("domain_motion.nesting.child_cells_y",
                             relationship.child_cells_y);
    take_nonnegative_integer("domain_motion.nesting.relaxation_width_child_cells",
                             relationship.relaxation_width_child_cells);
    take_positive_integer("domain_motion.nesting.boundary_stencil_parent_cells",
                          relationship.boundary_stencil_parent_cells);
    if (config.count("domain_motion.nesting.relaxation_e_folding_time_s") &&
        (!try_parse_double_value(config["domain_motion.nesting.relaxation_e_folding_time_s"],
                                 relationship.relaxation_e_folding_time_s) ||
         relationship.relaxation_e_folding_time_s <= 0.0))
    {
        throw std::runtime_error(
            "domain_motion.nesting.relaxation_e_folding_time_s must be finite and positive");
    }
    if (config.count("domain_motion.nesting.feedback") &&
        !tmv::init::config::parse_nesting_feedback(
            config["domain_motion.nesting.feedback"], relationship.feedback))
    {
        throw std::runtime_error(
            "domain_motion.nesting.feedback must be one_way or two_way");
    }
    if (relationship.parent_ratio_x < 2 || relationship.parent_ratio_y < 2)
        throw std::runtime_error(
            "domain_motion.nesting horizontal parent ratios must be at least 2");
    if (relationship.parent_time_step_ratio <
        std::max(relationship.parent_ratio_x, relationship.parent_ratio_y))
    {
        throw std::runtime_error(
            "domain_motion.nesting.parent_time_step_ratio must be at least the largest horizontal parent ratio");
    }
    if (relationship.enabled &&
        (relationship.child_cells_x <= 0 || relationship.child_cells_y <= 0 ||
         relationship.relaxation_width_child_cells <= 0 ||
         relationship.boundary_stencil_parent_cells <= 0 ||
         !std::isfinite(relationship.relaxation_e_folding_time_s) ||
         relationship.relaxation_e_folding_time_s <= 0.0 ||
         relationship.child_cells_x % relationship.parent_ratio_x != 0 ||
         relationship.child_cells_y % relationship.parent_ratio_y != 0 ||
         2LL * relationship.relaxation_width_child_cells >=
             std::min(relationship.child_cells_x, relationship.child_cells_y)))
    {
        throw std::runtime_error(
            "enabled domain_motion.nesting has an invalid child footprint or relaxation contract");
    }

    if (config.count("nested.enabled"))
    {
        nested.enabled = parse_bool_value(config["nested.enabled"]);
        require_supported_nested_grid_runtime(nested.enabled);
    }

    if (config.count("nested.refinement"))
    {
        double parsed = 0.0;
        if (try_parse_double_value(config["nested.refinement"], parsed))
        {
            nested.refinement = parsed;
        }
        else
        {
            warn_invalid_config_value("nested.refinement", config["nested.refinement"], "a finite number");
        }
    }

    if (config.count("nested.size_r"))
    {
        int parsed = 0;
        if (try_parse_int_value(config["nested.size_r"], parsed))
        {
            nested.nest_size_r = parsed;
        }
        else
        {
            warn_invalid_config_value("nested.size_r", config["nested.size_r"], "an integer");
        }
    }

    if (config.count("nested.size_th"))
    {
        int parsed = 0;
        if (try_parse_int_value(config["nested.size_th"], parsed))
        {
            nested.nest_size_th = parsed;
        }
        else
        {
            warn_invalid_config_value("nested.size_th", config["nested.size_th"], "an integer");
        }
    }

    if (config.count("nested.size_z"))
    {
        int parsed = 0;
        if (try_parse_int_value(config["nested.size_z"], parsed))
        {
            nested.nest_size_z = parsed;
        }
        else
        {
            warn_invalid_config_value("nested.size_z", config["nested.size_z"], "an integer");
        }
    }
}

// environment.hodograph.* -> global_wind_profile (the WK 3-point anchors) then
// composed into global_hodograph_source_config. Leaf; the anchors-before-compose
// order is preserved within the helper (the compose reads the anchors set above).
void load_hodograph(std::unordered_map<std::string, std::string>& config)
{
    if (config.count("environment.hodograph.u_sfc_ms"))
    {
        double parsed = 0.0;
        if (try_parse_double_value(config["environment.hodograph.u_sfc_ms"], parsed))
        {
            global_wind_profile.u_sfc = parsed;
        }
        else
        {
            warn_invalid_config_value(
                "environment.hodograph.u_sfc_ms", config["environment.hodograph.u_sfc_ms"], "a finite number");
        }
    }

    if (config.count("environment.hodograph.v_sfc_ms"))
    {
        double parsed = 0.0;
        if (try_parse_double_value(config["environment.hodograph.v_sfc_ms"], parsed))
        {
            global_wind_profile.v_sfc = parsed;
        }
        else
        {
            warn_invalid_config_value(
                "environment.hodograph.v_sfc_ms", config["environment.hodograph.v_sfc_ms"], "a finite number");
        }
    }

    if (config.count("environment.hodograph.u_1km_ms"))
    {
        double parsed = 0.0;
        if (try_parse_double_value(config["environment.hodograph.u_1km_ms"], parsed))
        {
            global_wind_profile.u_1km = parsed;
        }
        else
        {
            warn_invalid_config_value(
                "environment.hodograph.u_1km_ms", config["environment.hodograph.u_1km_ms"], "a finite number");
        }
    }

    if (config.count("environment.hodograph.v_1km_ms"))
    {
        double parsed = 0.0;
        if (try_parse_double_value(config["environment.hodograph.v_1km_ms"], parsed))
        {
            global_wind_profile.v_1km = parsed;
        }
        else
        {
            warn_invalid_config_value(
                "environment.hodograph.v_1km_ms", config["environment.hodograph.v_1km_ms"], "a finite number");
        }
    }

    if (config.count("environment.hodograph.u_6km_ms"))
    {
        double parsed = 0.0;
        if (try_parse_double_value(config["environment.hodograph.u_6km_ms"], parsed))
        {
            global_wind_profile.u_6km = parsed;
        }
        else
        {
            warn_invalid_config_value(
                "environment.hodograph.u_6km_ms", config["environment.hodograph.u_6km_ms"], "a finite number");
        }
    }

    if (config.count("environment.hodograph.v_6km_ms"))
    {
        double parsed = 0.0;
        if (try_parse_double_value(config["environment.hodograph.v_6km_ms"], parsed))
        {
            global_wind_profile.v_6km = parsed;
        }
        else
        {
            warn_invalid_config_value(
                "environment.hodograph.v_6km_ms", config["environment.hodograph.v_6km_ms"], "a finite number");
        }
    }

    // ── HodographSourceConfig surface ─────────────────────────────────
    //
    // Composed from the WK 3-point anchors above (always populated) plus
    // the new `environment.hodograph.type` selector. Default Type::Auto
    // means "use sounding winds when the SoundingSource provides them
    // (file path), fall back to WK 3-point parametric otherwise" — which
    // is exactly today's behavior. Explicit type=zero|wk_param overrides
    // any sounding-supplied winds.
    global_hodograph_source_config                     = tmv::init::HodographSourceConfig{};
    global_hodograph_source_config.wk_anchors.u_sfc_ms = global_wind_profile.u_sfc;
    global_hodograph_source_config.wk_anchors.v_sfc_ms = global_wind_profile.v_sfc;
    global_hodograph_source_config.wk_anchors.u_1km_ms = global_wind_profile.u_1km;
    global_hodograph_source_config.wk_anchors.v_1km_ms = global_wind_profile.v_1km;
    global_hodograph_source_config.wk_anchors.u_6km_ms = global_wind_profile.u_6km;
    global_hodograph_source_config.wk_anchors.v_6km_ms = global_wind_profile.v_6km;
    if (config.count("environment.hodograph.type"))
    {
        const std::string                      raw = config["environment.hodograph.type"];
        tmv::init::HodographSourceConfig::Type parsed_type{};
        if (tmv::init::parse_hodograph_source_type(raw, parsed_type))
        {
            global_hodograph_source_config.type = parsed_type;
        }
        else
        {
            warn_invalid_config_value("environment.hodograph.type", raw, "one of: auto, wk_param, zero");
        }
    }
}

// trigger.* -> global_bubble_* (warm-bubble center/radius/dtheta) +
// global_trigger_source_config (type, vortex_seed). The seed/vortex knobs are
// read regardless of the active trigger.type so the resolved configuration is
// deterministic, and every explicitly supplied value is validated.
void load_trigger(std::unordered_map<std::string, std::string>& config)
{
    if (config.count("trigger.bubble.center_x_km"))
    {
        double parsed = 0.0;
        if (try_parse_double_value(config["trigger.bubble.center_x_km"], parsed))
        {
            global_bubble_center_x_m = parsed * 1000.0;
        }
        else
        {
            warn_invalid_config_value(
                "trigger.bubble.center_x_km", config["trigger.bubble.center_x_km"], "a finite number");
        }
    }
    if (config.count("trigger.bubble.center_y_km"))
    {
        double parsed = 0.0;
        if (try_parse_double_value(config["trigger.bubble.center_y_km"], parsed))
        {
            global_bubble_center_y_m = parsed * 1000.0;
        }
        else
        {
            warn_invalid_config_value(
                "trigger.bubble.center_y_km", config["trigger.bubble.center_y_km"], "a finite number");
        }
    }
    if (config.count("trigger.bubble.center_z_km"))
    {
        double parsed = 0.0;
        if (try_parse_double_value(config["trigger.bubble.center_z_km"], parsed))
        {
            global_bubble_center_z_m = parsed * 1000.0;
        }
        else
        {
            warn_invalid_config_value(
                "trigger.bubble.center_z_km", config["trigger.bubble.center_z_km"], "a finite number");
        }
    }
    if (config.count("trigger.bubble.radius_km"))
    {
        double parsed = 0.0;
        if (try_parse_double_value(config["trigger.bubble.radius_km"], parsed))
        {
            global_bubble_radius_m = parsed * 1000.0;
        }
        else
        {
            warn_invalid_config_value(
                "trigger.bubble.radius_km", config["trigger.bubble.radius_km"], "a finite number");
        }
    }
    // trigger.type — selects which TriggerSource the runtime applies after
    // the sounding column is broadcast. Default is warm_bubble (today's
    // behavior); type=none disables the trigger explicitly; type=vortex_seed
    // imposes a Rankine vortex (used by tornado / tornado_cgrid schemes).
    // Legacy YAML that just sets trigger.bubble.dtheta_k = 0 keeps working
    // as a no-op since WarmBubbleTrigger reads global_bubble_dtheta_k.
    if (config.count("trigger.type"))
    {
        const std::string                    raw = config["trigger.type"];
        tmv::init::TriggerSourceConfig::Type parsed_type{};
        if (tmv::init::parse_trigger_source_type(raw, parsed_type))
        {
            global_trigger_source_config.type = parsed_type;
        }
        else
        {
            warn_invalid_config_value("trigger.type", raw, "one of: warm_bubble, none, vortex_seed");
        }
    }

    // trigger.vortex_seed.* — Rankine vortex parameters. Read regardless
    // of the active trigger.type so a user can flip between warm_bubble
    // and vortex_seed without losing their seed knobs.
    if (config.count("trigger.vortex_seed.r_max_m"))
    {
        double parsed = 0.0;
        if (try_parse_double_value(config["trigger.vortex_seed.r_max_m"], parsed))
        {
            global_trigger_source_config.vortex_seed.r_max_m = parsed;
        }
        else
        {
            warn_invalid_config_value(
                "trigger.vortex_seed.r_max_m", config["trigger.vortex_seed.r_max_m"], "a finite number");
        }
    }
    if (config.count("trigger.vortex_seed.v_max_ms"))
    {
        double parsed = 0.0;
        if (try_parse_double_value(config["trigger.vortex_seed.v_max_ms"], parsed))
        {
            global_trigger_source_config.vortex_seed.v_max_ms = parsed;
        }
        else
        {
            warn_invalid_config_value(
                "trigger.vortex_seed.v_max_ms", config["trigger.vortex_seed.v_max_ms"], "a finite number");
        }
    }
    if (config.count("trigger.vortex_seed.z_top_m"))
    {
        double parsed = 0.0;
        if (try_parse_double_value(config["trigger.vortex_seed.z_top_m"], parsed))
        {
            global_trigger_source_config.vortex_seed.z_top_m = parsed;
        }
        else
        {
            warn_invalid_config_value(
                "trigger.vortex_seed.z_top_m", config["trigger.vortex_seed.z_top_m"], "a finite number");
        }
    }
    if (config.count("trigger.vortex_seed.center_x_km"))
    {
        double parsed = 0.0;
        if (try_parse_double_value(config["trigger.vortex_seed.center_x_km"], parsed))
        {
            global_trigger_source_config.vortex_seed.center_x_m = parsed * 1000.0;
        }
        else
        {
            warn_invalid_config_value(
                "trigger.vortex_seed.center_x_km", config["trigger.vortex_seed.center_x_km"], "a finite number");
        }
    }
    if (config.count("trigger.vortex_seed.center_y_km"))
    {
        double parsed = 0.0;
        if (try_parse_double_value(config["trigger.vortex_seed.center_y_km"], parsed))
        {
            global_trigger_source_config.vortex_seed.center_y_m = parsed * 1000.0;
        }
        else
        {
            warn_invalid_config_value(
                "trigger.vortex_seed.center_y_km", config["trigger.vortex_seed.center_y_km"], "a finite number");
        }
    }
    if (config.count("trigger.vortex_seed.center_z_km"))
    {
        double parsed = 0.0;
        if (try_parse_double_value(config["trigger.vortex_seed.center_z_km"], parsed))
        {
            global_trigger_source_config.vortex_seed.center_z_m = parsed * 1000.0;
        }
        else
        {
            warn_invalid_config_value(
                "trigger.vortex_seed.center_z_km", config["trigger.vortex_seed.center_z_km"], "a finite number");
        }
    }

    if (config.count("trigger.bubble.dtheta_k"))
    {
        double parsed = 0.0;
        if (try_parse_double_value(config["trigger.bubble.dtheta_k"], parsed))
        {
            global_bubble_dtheta_k = parsed;
        }
        else
        {
            warn_invalid_config_value("trigger.bubble.dtheta_k", config["trigger.bubble.dtheta_k"], "a finite number");
        }
    }
    if (config.count("trigger.bubble.cloud_water_kgkg"))
    {
        double parsed = 0.0;
        if (try_parse_double_value(
                config["trigger.bubble.cloud_water_kgkg"], parsed)
            && parsed >= 0.0)
        {
            global_bubble_cloud_water_kgkg = parsed;
        }
        else
        {
            warn_invalid_config_value(
                "trigger.bubble.cloud_water_kgkg",
                config["trigger.bubble.cloud_water_kgkg"],
                "a finite non-negative number");
        }
    }
}

// numerics.compute.* -> global_compute_backend_config (+ the active vertical-flux
// template). THROWS std::runtime_error on an invalid backend (the caller, the
// load_config branch, propagates it), so this is NOT a no-throw leaf.
void load_numerics_compute(std::unordered_map<std::string, std::string>& config)
{
    if (config.count("numerics.compute.backend"))
    {
        ComputeBackendKind parsed = ComputeBackendKind::Cpu;
        if (parse_compute_backend_kind(config["numerics.compute.backend"], parsed))
        {
            global_compute_backend_config.backend = compute_backend_kind_name(parsed);
        }
        else
        {
            throw std::runtime_error("Invalid numerics.compute.backend '" + config["numerics.compute.backend"] +
                                     "'; expected one of: cpu, vulkan, gpu, metal");
        }
    }
    if (config.count("numerics.compute.device_index"))
    {
        int parsed = 0;
        if (try_parse_int_value(config["numerics.compute.device_index"], parsed) && parsed >= -1)
        {
            global_compute_backend_config.device_index = parsed;
        }
        else
        {
            warn_invalid_config_value(
                "numerics.compute.device_index", config["numerics.compute.device_index"], "an integer >= -1");
        }
    }
    if (config.count("numerics.compute.allow_fallback"))
    {
        global_compute_backend_config.allow_fallback = parse_bool_value(config["numerics.compute.allow_fallback"]);
    }
    if (config.count("numerics.compute.validate_parity"))
    {
        global_compute_backend_config.validate_parity = parse_bool_value(config["numerics.compute.validate_parity"]);
    }
    if (config.count("numerics.compute.cuda_assist"))
        global_compute_backend_config.cuda_assist = parse_bool_value(config["numerics.compute.cuda_assist"]);
    if (config.count("numerics.compute.gpu_apply_microphysics"))
    {
        global_gpu_apply_microphysics =
            parse_bool_value(config["numerics.compute.gpu_apply_microphysics"]);
    }
    if (config.count("numerics.compute.gpu_apply_slow"))
    {
        global_gpu_apply_slow =
            parse_bool_value(config["numerics.compute.gpu_apply_slow"]);
    }
    if (config.count("numerics.compute.gpu_apply_sgs"))
    {
        global_gpu_apply_sgs =
            parse_bool_value(config["numerics.compute.gpu_apply_sgs"]);
    }
    if (config.count("numerics.compute.templates.vertical_flux"))
    {
        const std::string template_id = config["numerics.compute.templates.vertical_flux"];
        if (has_vertical_flux_template(template_id))
        {
            set_active_vertical_flux_template_id(template_id);
        }
        else
        {
            std::string supported;
            const auto  templates = list_vertical_flux_templates();
            for (std::size_t i = 0; i < templates.size(); ++i)
            {
                if (i > 0)
                {
                    supported += ", ";
                }
                supported += templates[i].id;
            }
            warn_invalid_config_value("numerics.compute.templates.vertical_flux",
                                      template_id,
                                      supported.empty() ? "a known template id" : supported.c_str());
        }
    }
}

void load_terrain(std::unordered_map<std::string, std::string>& config)
{
    global_terrain_config.horizontal_placement_id = "grid_attached";
    global_terrain_config.feature_center_earth_x_m = 0.0;
    global_terrain_config.feature_center_earth_y_m = 0.0;
    if (config.count("terrain.scheme"))
    {
        const std::string& raw_requested = config["terrain.scheme"];
        const auto*        descriptor    = tmv::find_scheme_descriptor(get_terrain_scheme_descriptors(), raw_requested);
        const std::string  requested     = descriptor ? descriptor->id : std::string{};
        const std::vector<std::string> valid_schemes = get_available_terrain_schemes();
        const bool                     valid         = descriptor && descriptor->constructible;

        if (valid)
        {
            // Guard: refuse a non-flat terrain scheme on a dynamics solver that
            // has no terrain-following metric wiring (currently the Cartesian
            // A-grid). Without this the terrain metrics are computed and then
            // silently ignored by the dynamics, producing physically meaningless
            // output. coordinate_system + staggering are parsed earlier in
            // load_config, so they are resolved here. See
            // dynamics_supports_terrain() in terrain_base.hpp.
            if (requested != "none" && !dynamics_supports_terrain(global_coordinate_system, global_stagger_type))
            {
                throw std::runtime_error(std::string("Unsupported configuration: terrain.scheme '") + requested +
                                         "' with the " + coordinate_system_name(global_coordinate_system) + " " +
                                         stagger_type_name(global_stagger_type) +
                                         " dynamics solver, which has no terrain-following metric wiring "
                                         "(Jacobian-weighted continuity, G13/G23 pressure-gradient correction, "
                                         "sloped lower boundary). The terrain would be computed and then silently "
                                         "ignored. Set terrain.scheme: none on this path until terrain-following "
                                         "coordinates are implemented for it.");
            }

            global_terrain_config.scheme_id = requested;
        }
        else
        {
            if (descriptor != nullptr)
            {
                throw std::runtime_error("terrain.scheme '" + descriptor->id +
                                         "' is a development-only coordinate mapping and is unavailable "
                                         "for simulation until terrain-following dynamics are qualified");
            }
            std::string opts;
            for (std::size_t i = 0; i < valid_schemes.size(); ++i)
            {
                if (i > 0)
                    opts += ", ";
                opts += valid_schemes[i];
            }
            throw std::runtime_error("Invalid terrain.scheme '" + config["terrain.scheme"] +
                                     "'. Available options: " + opts);
        }
    }
    if (config.count("terrain.ztop"))
    {
        double parsed = 0.0;
        if (try_parse_double_value(config["terrain.ztop"], parsed) && parsed > 0.0)
        {
            global_terrain_config.ztop = parsed;
        }
        else
        {
            warn_invalid_config_value("terrain.ztop", config["terrain.ztop"], "a positive finite number");
        }
    }
    if (config.count("terrain.coord_id"))
    {
        std::string parsed_coord;
        if (parse_terrain_coord_id(config["terrain.coord_id"], parsed_coord))
        {
            global_terrain_config.coord_id = parsed_coord;
        }
        else
        {
            warn_invalid_config_value("terrain.coord_id",
                                      config["terrain.coord_id"],
                                      "\"btf\", \"smoothed\", or \"cartesian\" (aliases supported)");
        }
    }
    if (config.count("terrain.horizontal_placement"))
    {
        const std::string placement =
            tmv::strutil::trim_and_lower(config["terrain.horizontal_placement"]);
        if (placement != "grid_attached" && placement != "earth_fixed")
            throw std::runtime_error(
                "terrain.horizontal_placement must be grid_attached or earth_fixed");
        global_terrain_config.horizontal_placement_id = placement;
    }
    const auto load_feature_center = [&](const char* key, double& destination)
    {
        if (!config.count(key))
            return;
        if (!try_parse_double_value(config[key], destination))
            throw std::runtime_error(std::string(key) + " must be finite");
    };
    load_feature_center(
        "terrain.feature_center_earth_x_m",
        global_terrain_config.feature_center_earth_x_m);
    load_feature_center(
        "terrain.feature_center_earth_y_m",
        global_terrain_config.feature_center_earth_y_m);
    if (config.count("terrain.compute_derivatives"))
    {
        global_terrain_config.compute_derivatives = parse_bool_value(config["terrain.compute_derivatives"]);
    }
    if (config.count("terrain.compute_metrics"))
    {
        global_terrain_config.compute_metrics = parse_bool_value(config["terrain.compute_metrics"]);
    }
    if (config.count("terrain.smoothing_decay"))
    {
        double parsed = 0.0;
        if (try_parse_double_value(config["terrain.smoothing_decay"], parsed) && parsed >= 0.0)
        {
            global_terrain_config.smoothing_decay = parsed;
        }
        else
        {
            warn_invalid_config_value(
                "terrain.smoothing_decay", config["terrain.smoothing_decay"], "a non-negative finite number");
        }
    }
    if (config.count("terrain.bell.h0"))
    {
        double parsed = 0.0;
        if (try_parse_double_value(config["terrain.bell.h0"], parsed) && parsed >= 0.0)
        {
            global_terrain_config.bell.h0 = parsed;
        }
        else
        {
            warn_invalid_config_value("terrain.bell.h0", config["terrain.bell.h0"], "a non-negative finite number");
        }
    }
    if (config.count("terrain.bell.a"))
    {
        double parsed = 0.0;
        if (try_parse_double_value(config["terrain.bell.a"], parsed) && parsed > 0.0)
        {
            global_terrain_config.bell.a = parsed;
        }
        else
        {
            warn_invalid_config_value("terrain.bell.a", config["terrain.bell.a"], "a positive finite number");
        }
    }
    if (config.count("terrain.bell.axisymmetric"))
    {
        global_terrain_config.bell.axisymmetric = parse_bool_value(config["terrain.bell.axisymmetric"]);
    }
    if (config.count("terrain.schar.H"))
    {
        double parsed = 0.0;
        if (try_parse_double_value(config["terrain.schar.H"], parsed) && parsed >= 0.0)
        {
            global_terrain_config.schar.H = parsed;
        }
        else
        {
            warn_invalid_config_value("terrain.schar.H", config["terrain.schar.H"], "a non-negative finite number");
        }
    }
    if (config.count("terrain.schar.a"))
    {
        double parsed = 0.0;
        if (try_parse_double_value(config["terrain.schar.a"], parsed) && parsed > 0.0)
        {
            global_terrain_config.schar.a = parsed;
        }
        else
        {
            warn_invalid_config_value("terrain.schar.a", config["terrain.schar.a"], "a positive finite number");
        }
    }
    if (config.count("terrain.schar.ell"))
    {
        double parsed = 0.0;
        if (try_parse_double_value(config["terrain.schar.ell"], parsed) && parsed > 0.0)
        {
            global_terrain_config.schar.ell = parsed;
        }
        else
        {
            warn_invalid_config_value("terrain.schar.ell", config["terrain.schar.ell"], "a positive finite number");
        }
    }
}

void load_chaos(std::unordered_map<std::string, std::string>& config)
{
    if (config.count("chaos.scheme"))
    {
        global_chaos_config.scheme_id = to_lower_copy(config["chaos.scheme"]);
    }
    if (config.count("chaos.seed"))
    {
        std::uint64_t parsed = 0;
        if (try_parse_uint64_value(config["chaos.seed"], parsed))
        {
            global_chaos_config.seed = parsed;
        }
        else
        {
            warn_invalid_config_value("chaos.seed", config["chaos.seed"], "a non-negative integer");
        }
    }
    if (config.count("chaos.member_id"))
    {
        int parsed = 0;
        if (try_parse_int_value(config["chaos.member_id"], parsed))
        {
            global_chaos_config.member_id = parsed;
        }
        else
        {
            warn_invalid_config_value("chaos.member_id", config["chaos.member_id"], "an integer");
        }
    }
    if (config.count("chaos.native_generation_required"))
    {
        bool parsed = false;
        if (try_parse_bool_value(
                config["chaos.native_generation_required"], parsed))
            global_chaos_config.native_generation_required = parsed;
        else
            warn_invalid_config_value(
                "chaos.native_generation_required",
                config["chaos.native_generation_required"], "true or false");
    }
    if (config.count("chaos.dt_chaos"))
    {
        double parsed = 0.0;
        if (try_parse_double_value(config["chaos.dt_chaos"], parsed))
        {
            global_chaos_config.dt_chaos = parsed;
        }
        else
        {
            warn_invalid_config_value("chaos.dt_chaos", config["chaos.dt_chaos"], "a finite number");
        }
    }
    if (config.count("chaos.tau_t"))
    {
        double parsed = 0.0;
        if (try_parse_double_value(config["chaos.tau_t"], parsed))
        {
            global_chaos_config.tau_t = parsed;
        }
        else
        {
            warn_invalid_config_value("chaos.tau_t", config["chaos.tau_t"], "a finite number");
        }
    }
    if (config.count("chaos.Lx"))
    {
        double parsed = 0.0;
        if (try_parse_double_value(config["chaos.Lx"], parsed))
        {
            global_chaos_config.Lx = parsed;
        }
        else
        {
            warn_invalid_config_value("chaos.Lx", config["chaos.Lx"], "a finite number");
        }
    }
    if (config.count("chaos.Ly"))
    {
        double parsed = 0.0;
        if (try_parse_double_value(config["chaos.Ly"], parsed))
        {
            global_chaos_config.Ly = parsed;
        }
        else
        {
            warn_invalid_config_value("chaos.Ly", config["chaos.Ly"], "a finite number");
        }
    }
    if (config.count("chaos.filter_id"))
    {
        global_chaos_config.filter_id = config["chaos.filter_id"];
    }
    if (config.count("chaos.xi_max"))
    {
        double parsed = 0.0;
        if (try_parse_double_value(config["chaos.xi_max"], parsed))
        {
            global_chaos_config.xi_max = parsed;
        }
        else
        {
            warn_invalid_config_value("chaos.xi_max", config["chaos.xi_max"], "a finite number");
        }
    }
    if (config.count("chaos.taper_id"))
    {
        global_chaos_config.taper_id = config["chaos.taper_id"];
    }
    if (config.count("chaos.taper_z1"))
    {
        double parsed = 0.0;
        if (try_parse_double_value(config["chaos.taper_z1"], parsed))
        {
            global_chaos_config.taper_z1 = parsed;
        }
        else
        {
            warn_invalid_config_value("chaos.taper_z1", config["chaos.taper_z1"], "a finite number");
        }
    }
    if (config.count("chaos.taper_z2"))
    {
        double parsed = 0.0;
        if (try_parse_double_value(config["chaos.taper_z2"], parsed))
        {
            global_chaos_config.taper_z2 = parsed;
        }
        else
        {
            warn_invalid_config_value("chaos.taper_z2", config["chaos.taper_z2"], "a finite number");
        }
    }
    if (config.count("chaos.apply_to_ic"))
    {
        global_chaos_config.apply_to_ic = parse_string_list(config["chaos.apply_to_ic"]);
    }
    if (config.count("chaos.apply_to_tendencies"))
    {
        global_chaos_config.apply_to_tendencies = parse_string_list(config["chaos.apply_to_tendencies"]);
    }
}

void load_dynamic_key_overrides(std::unordered_map<std::string, std::string>& config)
{
    for (const auto& kv : config)
    {
        const std::string& key = kv.first;
        std::string        override_field;
        bool               override_is_min = true;
        if (parse_override_key(key, override_field, override_is_min))
        {
            auto& bounds = global_validation_policy.field_overrides[to_lower_copy(override_field)];
            if (override_is_min)
            {
                double parsed = 0.0;
                if (try_parse_double_value(kv.second, parsed))
                {
                    bounds.has_min   = true;
                    bounds.min_value = parsed;
                }
                else
                {
                    warn_invalid_config_value(key, kv.second, "a finite number");
                }
            }
            else
            {
                double parsed = 0.0;
                if (try_parse_double_value(kv.second, parsed))
                {
                    bounds.has_max   = true;
                    bounds.max_value = parsed;
                }
                else
                {
                    warn_invalid_config_value(key, kv.second, "a finite number");
                }
            }
            continue;
        }

        if (key.rfind("chaos.sigma_ic.", 0) == 0)
        {
            const std::string var    = key.substr(std::string("chaos.sigma_ic.").size());
            double            parsed = 0.0;
            if (try_parse_double_value(kv.second, parsed))
            {
                global_chaos_config.sigma_ic[var] = parsed;
            }
            else
            {
                warn_invalid_config_value(key, kv.second, "a finite number");
            }
        }
        else if (key.rfind("chaos.alpha_tend.", 0) == 0)
        {
            const std::string block  = key.substr(std::string("chaos.alpha_tend.").size());
            double            parsed = 0.0;
            if (try_parse_double_value(kv.second, parsed))
            {
                global_chaos_config.alpha_tend[block] = parsed;
            }
            else
            {
                warn_invalid_config_value(key, kv.second, "a finite number");
            }
        }
    }
}

void load_environment(std::unordered_map<std::string, std::string>& config)
{
    if (config.count("environment.cape_target_jkg"))
    {
        double parsed = 0.0;
        if (try_parse_double_value(config["environment.cape_target_jkg"], parsed))
        {
            global_cape_target = parsed;
        }
        else
        {
            warn_invalid_config_value(
                "environment.cape_target_jkg", config["environment.cape_target_jkg"], "a finite number");
        }
    }

    if (config.count("environment.sfc_theta_k"))
    {
        double parsed = 0.0;
        if (try_parse_double_value(config["environment.sfc_theta_k"], parsed))
        {
            global_sfc_theta_k = parsed;
        }
        else
        {
            warn_invalid_config_value("environment.sfc_theta_k", config["environment.sfc_theta_k"], "a finite number");
        }
    }
    if (config.count("environment.sfc_qv_kgkg"))
    {
        double parsed = 0.0;
        if (try_parse_double_value(config["environment.sfc_qv_kgkg"], parsed))
        {
            global_sfc_qv_kgkg = parsed;
        }
        else
        {
            warn_invalid_config_value("environment.sfc_qv_kgkg", config["environment.sfc_qv_kgkg"], "a finite number");
        }
    }
    if (config.count("environment.tropopause_z_m"))
    {
        double parsed = 0.0;
        if (try_parse_double_value(config["environment.tropopause_z_m"], parsed))
        {
            global_tropopause_z_m = parsed;
        }
        else
        {
            warn_invalid_config_value(
                "environment.tropopause_z_m", config["environment.tropopause_z_m"], "a finite number");
        }
    }

    if (config.count("environment.sounding.enabled"))
    {
        global_sounding_enabled = parse_bool_value(config["environment.sounding.enabled"]);
    }
    if (config.count("environment.sounding.scheme"))
    {
        global_runtime_sounding_config.scheme_id = to_lower_copy(config["environment.sounding.scheme"]);
    }
    if (config.count("environment.sounding.scheme_id"))
    {
        global_runtime_sounding_config.scheme_id = to_lower_copy(config["environment.sounding.scheme_id"]);
    }
    if (config.count("environment.sounding.file_path"))
    {
        global_runtime_sounding_config.file_path = config["environment.sounding.file_path"];
    }
    if (config.count("environment.sounding.path"))
    {
        global_runtime_sounding_config.file_path = config["environment.sounding.path"];
    }
    if (config.count("environment.sounding.use_fallback_profiles"))
    {
        global_runtime_sounding_config.use_fallback_profiles =
            parse_bool_value(config["environment.sounding.use_fallback_profiles"]);
    }
    if (config.count("environment.sounding.interpolation_method"))
    {
        double method = global_runtime_sounding_config.interpolation_method;
        if (parse_sounding_interpolation_method(config["environment.sounding.interpolation_method"], method))
        {
            global_runtime_sounding_config.interpolation_method = method;
        }
        else
        {
            warn_invalid_config_value("environment.sounding.interpolation_method",
                                      config["environment.sounding.interpolation_method"],
                                      "one of: linear|0, spline|1, log-linear|2");
        }
    }
    if (config.count("environment.sounding.extrapolate_below_ground"))
    {
        global_runtime_sounding_config.extrapolate_below_ground =
            parse_bool_value(config["environment.sounding.extrapolate_below_ground"]);
    }
    if (config.count("environment.sounding.extrapolate_above_top"))
    {
        global_runtime_sounding_config.extrapolate_above_top =
            parse_bool_value(config["environment.sounding.extrapolate_above_top"]);
    }
    if (config.count("environment.sounding.min_pressure_hpa"))
    {
        double parsed = 0.0;
        if (try_parse_double_value(config["environment.sounding.min_pressure_hpa"], parsed))
        {
            global_runtime_sounding_config.min_pressure_hpa = parsed;
        }
        else
        {
            warn_invalid_config_value("environment.sounding.min_pressure_hpa",
                                      config["environment.sounding.min_pressure_hpa"],
                                      "a finite number");
        }
    }
    if (config.count("environment.sounding.max_pressure_hpa"))
    {
        double parsed = 0.0;
        if (try_parse_double_value(config["environment.sounding.max_pressure_hpa"], parsed))
        {
            global_runtime_sounding_config.max_pressure_hpa = parsed;
        }
        else
        {
            warn_invalid_config_value("environment.sounding.max_pressure_hpa",
                                      config["environment.sounding.max_pressure_hpa"],
                                      "a finite number");
        }
    }
    if (config.count("environment.sounding.min_temperature_k"))
    {
        double parsed = 0.0;
        if (try_parse_double_value(config["environment.sounding.min_temperature_k"], parsed))
        {
            global_runtime_sounding_config.min_temperature_k = parsed;
        }
        else
        {
            warn_invalid_config_value("environment.sounding.min_temperature_k",
                                      config["environment.sounding.min_temperature_k"],
                                      "a finite number");
        }
    }
    if (config.count("environment.sounding.max_temperature_k"))
    {
        double parsed = 0.0;
        if (try_parse_double_value(config["environment.sounding.max_temperature_k"], parsed))
        {
            global_runtime_sounding_config.max_temperature_k = parsed;
        }
        else
        {
            warn_invalid_config_value("environment.sounding.max_temperature_k",
                                      config["environment.sounding.max_temperature_k"],
                                      "a finite number");
        }
    }
    if (config.count("environment.sounding.allow_placeholder_profiles"))
    {
        global_sounding_allow_placeholder_profiles =
            parse_bool_value(config["environment.sounding.allow_placeholder_profiles"]);
    }

    if (global_sounding_enabled && global_runtime_sounding_config.scheme_id.empty())
    {
        global_runtime_sounding_config.scheme_id = "auto";
    }

    // ── New SoundingSourceConfig surface ───────────────────────────────
    //
    // The legacy globals above (cape_target_jkg, sfc_*, tropopause_z_m,
    // sounding.scheme_id, sounding.file_path) drive both the legacy
    // SHARPY overlay path AND the new SoundingSource factory. This block
    // composes a SoundingSourceConfig from those globals plus the new
    // `environment.sounding.type` selector and stores it in
    // global_sounding_source_config for equations.cpp::initialize() to
    // consume. Defaults to ParametricCAPE so configs with no explicit
    // `type` get the historical procedural base state.
    global_sounding_source_config                            = tmv::init::SoundingSourceConfig{};
    global_sounding_source_config.parametric.cape_target_jkg = global_cape_target;
    global_sounding_source_config.parametric.surface_theta_k = global_sfc_theta_k;
    global_sounding_source_config.parametric.surface_qv_kgkg = global_sfc_qv_kgkg;
    global_sounding_source_config.parametric.tropopause_z_m  = global_tropopause_z_m;
    global_sounding_source_config.file.path                  = global_runtime_sounding_config.file_path;
    global_sounding_source_config.file.interpolation_method  = global_runtime_sounding_config.interpolation_method;
    global_sounding_source_config.file.extrapolate_below_ground =
        global_runtime_sounding_config.extrapolate_below_ground;
    global_sounding_source_config.file.extrapolate_above_top = global_runtime_sounding_config.extrapolate_above_top;
    global_sounding_source_config.file.min_pressure_hpa      = global_runtime_sounding_config.min_pressure_hpa;
    global_sounding_source_config.file.max_pressure_hpa      = global_runtime_sounding_config.max_pressure_hpa;
    global_sounding_source_config.file.min_temperature_k     = global_runtime_sounding_config.min_temperature_k;
    global_sounding_source_config.file.max_temperature_k     = global_runtime_sounding_config.max_temperature_k;
    if (!global_runtime_sounding_config.scheme_id.empty() && global_runtime_sounding_config.scheme_id != "none")
    {
        global_sounding_source_config.file.scheme_id = global_runtime_sounding_config.scheme_id;
    }
    if (config.count("environment.sounding.require_winds"))
    {
        global_sounding_source_config.file.require_winds =
            parse_bool_value(config["environment.sounding.require_winds"]);
    }
    bool sounding_type_explicit = false;
    if (config.count("environment.sounding.type"))
    {
        const std::string                     raw = config["environment.sounding.type"];
        tmv::init::SoundingSourceConfig::Type parsed_type{};
        if (tmv::init::parse_sounding_source_type(raw, parsed_type))
        {
            global_sounding_source_config.type = parsed_type;
            sounding_type_explicit             = true;
            if (parsed_type == tmv::init::SoundingSourceConfig::Type::File &&
                global_sounding_source_config.file.path.empty())
            {
                tmv::log_warn("environment.sounding.type=file but no file_path given. "
                              "Set environment.sounding.file_path to the sounding file.");
            }
        }
        else
        {
            throw std::runtime_error("Unknown environment.sounding.type '" + raw +
                                     "'; expected parametric_cape, file, or parametric_targets");
        }
    }

    // Legacy translation: a YAML that pre-dates `environment.sounding.type`
    // can still ask for a file-based sounding by setting just
    //   environment.sounding.scheme_id: sharpy
    //   environment.sounding.file_path: ...
    // We detect that pattern here and promote it to the new File source
    // path. Without this promotion, deleting the legacy SHARPY overlay
    // (apply_soundings_to_initial_state) would silently turn those configs
    // into parametric_cape runs.
    //
    // We also propagate the legacy `use_fallback_profiles` semantics so a
    // missing file silently reverts to the procedural base state, matching
    // historical behavior. Users who explicitly set type=file in the new
    // YAML get hard errors on load failures (no fallback) by default.
    if (!sounding_type_explicit && !global_sounding_source_config.file.path.empty() &&
        !global_sounding_source_config.file.scheme_id.empty() && global_sounding_source_config.file.scheme_id != "none")
    {
        global_sounding_source_config.type                       = tmv::init::SoundingSourceConfig::Type::File;
        global_sounding_source_config.file.use_fallback_profiles = global_runtime_sounding_config.use_fallback_profiles;
    }
    else if (sounding_type_explicit &&
             global_sounding_source_config.type == tmv::init::SoundingSourceConfig::Type::File &&
             config.count("environment.sounding.use_fallback_profiles"))
    {
        // Honor an explicit use_fallback_profiles when type=file is set.
        global_sounding_source_config.file.use_fallback_profiles = global_runtime_sounding_config.use_fallback_profiles;
    }

    // The legacy SHARPY overlay path (apply_soundings_to_initial_state in
    // nimbus_sim.cpp) has been retired. Whenever a file-based source is
    // active, ensure the overlay flag stays off so no future re-introduction
    // double-applies.
    if (global_sounding_source_config.type == tmv::init::SoundingSourceConfig::Type::File)
    {
        global_sounding_enabled = false;
    }

    // environment.sounding.targets.* — diagnostic-table inputs for the
    // ParametricTargets source. Read regardless of the active type so a
    // user can flip between parametric_cape and parametric_targets without
    // losing their values. Surface-anchor defaults fall back to the
    // existing environment.* knobs if not specified separately.
    {
        auto& tg           = global_sounding_source_config.targets;
        tg.surface_theta_k = global_sfc_theta_k;
        tg.target_el_m     = global_tropopause_z_m;
        tg.target_cape_jkg = global_cape_target;
        if (config.count("environment.sounding.targets.cape_jkg"))
        {
            double parsed = 0.0;
            if (try_parse_double_value(config["environment.sounding.targets.cape_jkg"], parsed))
            {
                tg.target_cape_jkg = parsed;
            }
        }
        if (config.count("environment.sounding.targets.cin_jkg"))
        {
            double parsed = 0.0;
            if (try_parse_double_value(config["environment.sounding.targets.cin_jkg"], parsed))
            {
                tg.target_cin_jkg = parsed;
            }
        }
        if (config.count("environment.sounding.targets.lcl_m"))
        {
            double parsed = 0.0;
            if (try_parse_double_value(config["environment.sounding.targets.lcl_m"], parsed))
            {
                tg.target_lcl_m = parsed;
            }
        }
        if (config.count("environment.sounding.targets.lfc_m"))
        {
            double parsed = 0.0;
            if (try_parse_double_value(config["environment.sounding.targets.lfc_m"], parsed))
            {
                tg.target_lfc_m = parsed;
            }
        }
        if (config.count("environment.sounding.targets.el_m"))
        {
            double parsed = 0.0;
            if (try_parse_double_value(config["environment.sounding.targets.el_m"], parsed))
            {
                tg.target_el_m = parsed;
            }
        }
        if (config.count("environment.sounding.targets.surface_theta_k"))
        {
            double parsed = 0.0;
            if (try_parse_double_value(config["environment.sounding.targets.surface_theta_k"], parsed))
            {
                tg.surface_theta_k = parsed;
            }
        }
        if (config.count("environment.sounding.targets.moisture_qv_kgkg"))
        {
            double parsed = 0.0;
            if (try_parse_double_value(config["environment.sounding.targets.moisture_qv_kgkg"], parsed))
            {
                tg.moisture_qv_kgkg_override = parsed;
            }
        }
    }
}

std::optional<tmv::init::config::ConfigV1> ingest_v1_config(std::unordered_map<std::string, std::string>& config, const std::string& config_path,
    const std::unordered_map<std::string, YamlField>& document,
    const std::string* compute_override, const std::string* sounding_override)
{
    runtime_configuration.admission = {};
    global_radar_product_selection = {};
    if (auto schema_it = config.find("schema"); schema_it != config.end() && schema_it->second == "nimbus.v1")
    {
        auto v1_result = tmv::init::config::load_v1_config_fields(config_path, document);
        if (!v1_result.ok())
        {
            tmv::log_error("v1 config load failed for: ", config_path);
            for (const auto& e : v1_result.errors)
            {
                std::ostringstream loc;
                loc << e.file;
                if (e.line > 0)
                    loc << ':' << e.line;
                tmv::log_error("  ", loc.str(), " ", e.key, ": ", e.message);
            }
            // Fatal: continuing here would run the built-in defaults in
            // place of the user's config (and crash downstream on the
            // half-applied state). The caller catches and exits non-zero.
            throw std::runtime_error("v1 config load failed for '" + config_path + "' (" +
                                     std::to_string(v1_result.errors.size()) + " error(s), listed above)");
        }
        if (compute_override && !tmv::init::config::parse_compute_backend(
                *compute_override, v1_result.config->extensions.compute.backend))
            throw std::invalid_argument("invalid compute backend override: " + *compute_override);
        if (sounding_override)
        {
            v1_result.config->initial_conditions.sounding = {};
            v1_result.config->initial_conditions.sounding.type = tmv::init::config::SoundingType::File;
            v1_result.config->initial_conditions.sounding.file_path = *sounding_override;
        }
        tmv::init::config::require_runtime_supported_configuration(*v1_result.config);
        runtime_configuration.admission = evaluate_runtime_science_admission(*v1_result.config);
        const auto& radar = v1_result.config->radar;
        const auto selected = [&](const char* product)
        {
            return std::find(
                radar.products.begin(), radar.products.end(), product) !=
                radar.products.end();
        };
        global_radar_product_selection.reflectivity =
            radar.enabled && selected("reflectivity_dbz");
        global_radar_product_selection.radial_velocity =
            radar.enabled && selected("radial_velocity");
        global_radar_product_selection.zdr =
            radar.enabled && selected("zdr");
        global_radar_product_selection.kdp =
            radar.enabled && selected("kdp");
        global_radar_product_selection.rhohv =
            radar.enabled && selected("rhohv");
        global_radar_product_selection.reflectivity_tier =
            radar.reflectivity_tier;
        global_radar_product_selection.velocity_sampling =
            radar.velocity_sampling;
        global_radar_product_selection.zdr_tier = radar.zdr_tier;
        global_radar_product_selection.radar_x = radar.radar_x_m;
        global_radar_product_selection.radar_y = radar.radar_y_m;
        global_radar_product_selection.radar_z = radar.radar_z_m;
        global_radar_product_selection.apply_scatterer_correction =
            radar.apply_scatterer_correction;
        // C1: propagate per-face lateral BC selection. These globals have
        // no v0 equivalent (v0 hard-codes BCs per scheme), so they remain
        // at their defaults when loading a v0 config.
        global_lateral_bc_west  = v1_result.config->boundary_conditions.lateral.west;
        global_lateral_bc_east  = v1_result.config->boundary_conditions.lateral.east;
        global_lateral_bc_south = v1_result.config->boundary_conditions.lateral.south;
        global_lateral_bc_north = v1_result.config->boundary_conditions.lateral.north;
        // C2: bottom/top BC selection. Same no-v0-equivalent caveat; v0
        // configs leave these at the legacy Cartesian defaults (RigidSlip
        // + Rigid). The migrator writes those legacy values explicitly so
        // round-tripped configs preserve behavior.
        global_bottom_bc = v1_result.config->boundary_conditions.bottom;
        global_top_bc    = v1_result.config->boundary_conditions.top;
        // C3: Rayleigh damping (top sponge). v1 schema defaults to CM1
        // (enabled, z_d=12000, alpha=0.05, explicit); migrator overrides
        // z_d per-config to NZ*dz*0.7 so re-migrated v0 configs preserve
        // their legacy depth-fraction behavior.
        global_damping_enabled      = v1_result.config->damping.enabled;
        global_damping_z_d          = v1_result.config->damping.z_d;
        global_damping_alpha        = v1_result.config->damping.alpha;
        global_lateral_sponge_width = v1_result.config->damping.lateral_width;
        global_lateral_sponge_alpha = v1_result.config->damping.lateral_alpha;
        // Additive w-only lid absorber (science audit: now YAML-visible, was
        // always-on with hardcoded defaults). modularity cleanup 6a removed the
        // NIMBUS_W_ONLY_TOP_SPONGE override; the config field is authoritative.
        global_w_only_top_sponge      = v1_result.config->damping.w_only_top_sponge;
        global_w_only_depth_cells     = v1_result.config->damping.w_only_depth_cells;
        global_w_only_alpha_max       = v1_result.config->damping.w_only_alpha_max;
        global_damping_mode           = v1_result.config->damping.mode;
        global_lateral_sponge_enabled = v1_result.config->damping.lateral_sponge_enabled;

        // modularity cleanup 6a: solver internals + GPU pointwise-apply toggles,
        // formerly the hidden NIMBUS_* env knobs. v0 configs lack the
        // extensions.{solver,compute} blocks, so they keep the defaults set
        // above (= prior env-unset runtime).
        global_slow_rk2               = v1_result.config->extensions.solver.slow_rk2;
        global_lid_bc_gamma           = v1_result.config->extensions.solver.lid_bc_gamma;
        global_axis_u_m1_filter       = v1_result.config->extensions.solver.axis_u_m1_filter;
        global_gpu_apply_microphysics = v1_result.config->extensions.compute.gpu_apply_microphysics;
        global_gpu_apply_slow         = v1_result.config->extensions.compute.gpu_apply_slow;
        global_gpu_apply_sgs          = v1_result.config->extensions.compute.gpu_apply_sgs;

        // A slab surface paired with a non-slab PBL scheme means YSU/MYNN takes
        // the single runtime boundary-layer slot
        // and consumes the slab surface fluxes - the slab MIXED-LAYER scheme is
        // NOT run. Say so loudly, since "slab" historically selected the
        // mixed-layer scheme and a user may expect that behavior. (Before this
        // fix, surface:slab silently overrode the requested PBL with slab.)
        if (v1_result.config->surface.scheme == tmv::init::config::SurfaceScheme::Slab &&
            v1_result.config->boundary_layer.scheme != tmv::init::config::BoundaryLayerScheme::None &&
            v1_result.config->boundary_layer.scheme != tmv::init::config::BoundaryLayerScheme::Slab)
        {
            const char* pbl_name =
                tmv::init::config::boundary_layer_scheme_name(v1_result.config->boundary_layer.scheme);
            tmv::log_info("[CONFIG] surface.scheme: slab + boundary_layer.scheme: ",
                          pbl_name,
                          " -> the ",
                          pbl_name,
                          " PBL takes the boundary-layer slot and consumes the "
                          "slab surface fluxes; the slab mixed-layer scheme is "
                          "not run.");
        }
        // C4: Coriolis. v1 schema CM1 default is enabled=false; migrator
        // overrides to enabled=true (latitude=35°N) to preserve the legacy
        // Weisman-Klemp Coriolis applied unconditionally by every v0 run.
        global_coriolis_enabled      = v1_result.config->coriolis.enabled;
        global_coriolis_mode         = v1_result.config->coriolis.mode;
        global_coriolis_latitude_deg = v1_result.config->coriolis.latitude_deg;

        // Time integration and advection were projected from their typed
        // selectors. Replacing either here would execute a different method
        // from the one admitted by strict v1 validation.
        return std::move(*v1_result.config);
    }
    return std::nullopt;
}

// Selects the grid's coordinate system (cartesian) and variable staggering
// (collocated/C-grid) - the spatial-discretization geometry, distinct from the
// grid dimensions/spacing parsed later (NR/NTH/NZ, dx/dy/dz). The coordinate/
// staggering cross-validation lives in the dynamics-scheme registry, not here.
void load_grid_geometry(std::unordered_map<std::string, std::string>& config)
{
    if (config.count("coordinate_system"))
    {
        const std::string& raw    = config["coordinate_system"];
        CoordinateSystem   parsed = global_coordinate_system;
        if (parse_coordinate_system(raw, parsed))
        {
            global_coordinate_system = parsed;
        }
        else
        {
            warn_invalid_config_value("coordinate_system", raw, "cartesian");
        }
    }

    if (config.count("grid.staggering"))
    {
        const std::string& raw    = config["grid.staggering"];
        StaggerType        parsed = global_stagger_type;
        if (parse_stagger_type(raw, parsed))
        {
            global_stagger_type = parsed;
        }
        else
        {
            warn_invalid_config_value("grid.staggering", raw, "one of: collocated, c_grid");
        }
    }
}

void load_logging(std::unordered_map<std::string, std::string>& config)
{
    if (config.count("logging.profile"))
    {
        bool             valid  = false;
        const LogProfile parsed = parse_log_profile(config["logging.profile"], &valid);
        if (valid)
        {
            global_log_profile = parsed;
        }
        else
        {
            tmv::log_warn("Invalid logging.profile '",
                          config["logging.profile"],
                          "'. Valid values: quiet, normal, debug. Using normal.");
            global_log_profile = LogProfile::normal;
        }
    }
}

void load_performance(std::unordered_map<std::string, std::string>& config)
{
    if (config.count("performance.timing"))
    {
        global_perf_timing_enabled = parse_bool_value(config["performance.timing"]);
    }
    if (config.count("performance.report_every_steps"))
    {
        int parsed = 0;
        if (try_parse_non_negative_int_value(config["performance.report_every_steps"], parsed))
        {
            global_perf_report_every_steps = parsed;
        }
        else
        {
            warn_invalid_config_value(
                "performance.report_every_steps", config["performance.report_every_steps"], "a non-negative integer");
        }
    }
    if (config.count("performance.warmup_steps"))
    {
        int parsed = 0;
        if (try_parse_non_negative_int_value(config["performance.warmup_steps"], parsed))
        {
            global_perf_warmup_steps = parsed;
        }
        else
        {
            warn_invalid_config_value(
                "performance.warmup_steps", config["performance.warmup_steps"], "a non-negative integer");
        }
    }
}

void load_diagnostics_cadence(std::unordered_map<std::string, std::string>& config)
{
    if (config.count("diagnostics.scientific_every_steps"))
    {
        int parsed = 0;
        if (try_parse_non_negative_int_value(config["diagnostics.scientific_every_steps"], parsed))
        {
            global_scientific_diagnostics_every_steps = parsed;
        }
        else
        {
            warn_invalid_config_value("diagnostics.scientific_every_steps",
                                      config["diagnostics.scientific_every_steps"],
                                      "a non-negative integer");
        }
    }
    if (config.count("diagnostics.deterministic_reductions"))
    {
        tmv::diagnostics::set_deterministic_reductions(
            parse_bool_value(config["diagnostics.deterministic_reductions"]));
    }
}

void load_storm_tracking(std::unordered_map<std::string, std::string>& config)
{
    using Tracking               = tmv::init::config::DomainMotionSection::Tracking;
    global_storm_tracking_config = Tracking{};
    auto& tracking               = global_storm_tracking_config;

    if (config.count("domain_motion.tracking.enabled"))
    {
        tracking.enabled = parse_bool_value(config["domain_motion.tracking.enabled"]);
    }
    if (config.count("domain_motion.tracking.recycle_domain"))
    {
        tracking.recycle_domain = parse_bool_value(config["domain_motion.tracking.recycle_domain"]);
    }
    const auto positive_double = [&](const char* key, double& destination)
    {
        if (!config.count(key))
            return;
        double parsed = 0.0;
        if (!try_parse_double_value(config[key], parsed) || parsed <= 0.0)
        {
            warn_invalid_config_value(key, config[key], "a positive finite number");
        }
        destination = parsed;
    };
    const auto non_negative_double = [&](const char* key, double& destination)
    {
        if (!config.count(key))
            return;
        double parsed = 0.0;
        if (!try_parse_double_value(config[key], parsed) || parsed < 0.0)
        {
            warn_invalid_config_value(key, config[key], "a non-negative finite number");
        }
        destination = parsed;
    };
    positive_double("domain_motion.tracking.interval_s", tracking.interval_s);
    non_negative_double("domain_motion.tracking.inner_margin_m", tracking.inner_margin_m);
    non_negative_double("domain_motion.tracking.cooldown_s", tracking.cooldown_s);
    non_negative_double("domain_motion.tracking.minimum_confidence", tracking.minimum_confidence);
    positive_double("domain_motion.tracking.minimum_updraft_ms", tracking.minimum_updraft_ms);
    positive_double("domain_motion.tracking.minimum_condensate_path_kgm2", tracking.minimum_condensate_path_kgm2);
    non_negative_double("domain_motion.tracking.updraft_helicity_bottom_m_agl", tracking.updraft_helicity_bottom_m_agl);
    positive_double("domain_motion.tracking.updraft_helicity_top_m_agl", tracking.updraft_helicity_top_m_agl);
    positive_double("domain_motion.tracking.minimum_absolute_updraft_helicity_m2s2",
                    tracking.minimum_absolute_updraft_helicity_m2s2);
    positive_double("domain_motion.tracking.minimum_core_area_m2", tracking.minimum_core_area_m2);
    positive_double("domain_motion.tracking.maximum_motion_speed_ms", tracking.maximum_motion_speed_ms);
    non_negative_double("domain_motion.tracking.maximum_motion_acceleration_ms2",
                        tracking.maximum_motion_acceleration_ms2);
    positive_double("domain_motion.tracking.branch_velocity_uncertainty_ms", tracking.branch_velocity_uncertainty_ms);
    if (config.count("domain_motion.tracking.branch_preference") &&
        !tmv::init::config::parse_storm_branch_preference(config["domain_motion.tracking.branch_preference"],
                                                          tracking.branch_preference))
    {
        warn_invalid_config_value("domain_motion.tracking.branch_preference",
                                  config["domain_motion.tracking.branch_preference"],
                                  "continuity, bunkers_right, or bunkers_left");
    }

    if (config.count("domain_motion.tracking.maximum_shift_cells") &&
        !try_parse_non_negative_int_value(config["domain_motion.tracking.maximum_shift_cells"],
                                          tracking.maximum_shift_cells))
    {
        warn_invalid_config_value("domain_motion.tracking.maximum_shift_cells",
                                  config["domain_motion.tracking.maximum_shift_cells"],
                                  "a non-negative integer");
    }
    if (config.count("domain_motion.tracking.minimum_component_cells") &&
        !try_parse_positive_int_value(config["domain_motion.tracking.minimum_component_cells"],
                                      tracking.minimum_component_cells))
    {
        warn_invalid_config_value("domain_motion.tracking.minimum_component_cells",
                                  config["domain_motion.tracking.minimum_component_cells"],
                                  "a positive integer");
    }
    if (config.count("domain_motion.tracking.minimum_continuity_samples") &&
        !try_parse_positive_int_value(config["domain_motion.tracking.minimum_continuity_samples"],
                                      tracking.minimum_continuity_samples))
    {
        warn_invalid_config_value("domain_motion.tracking.minimum_continuity_samples",
                                  config["domain_motion.tracking.minimum_continuity_samples"],
                                  "an integer from 2 through 5");
    }
    if (config.count("domain_motion.tracking.lost_after_samples") &&
        !try_parse_positive_int_value(config["domain_motion.tracking.lost_after_samples"], tracking.lost_after_samples))
    {
        warn_invalid_config_value("domain_motion.tracking.lost_after_samples",
                                  config["domain_motion.tracking.lost_after_samples"],
                                  "a positive integer");
    }
    if (config.count("domain_motion.tracking.topology_confirmation_samples") &&
        !try_parse_positive_int_value(config["domain_motion.tracking.topology_confirmation_samples"],
                                      tracking.topology_confirmation_samples))
    {
        warn_invalid_config_value("domain_motion.tracking.topology_confirmation_samples",
                                  config["domain_motion.tracking.topology_confirmation_samples"],
                                  "a positive integer");
    }

    if (tracking.minimum_confidence > 1.0)
    {
        throw std::runtime_error("domain_motion.tracking.minimum_confidence must be <= 1");
    }
    if (tracking.minimum_continuity_samples < 2 ||
        tracking.minimum_continuity_samples > static_cast<int>(tmv::diagnostics::storm_continuity_history_capacity))
    {
        throw std::runtime_error("domain_motion.tracking.minimum_continuity_samples must be in [2, 5]");
    }
    if (tracking.enabled &&
        (tracking.inner_margin_m <= 0.0 || tracking.minimum_confidence <= 0.0 ||
         tracking.maximum_motion_speed_ms <= 0.0 || tracking.maximum_motion_acceleration_ms2 < 0.0 ||
         tracking.lost_after_samples <= 0 || tracking.topology_confirmation_samples <= 0 ||
         tracking.branch_velocity_uncertainty_ms <= 0.0))
    {
        throw std::runtime_error("enabled domain_motion.tracking requires positive inner margin, "
                                 "confidence, maximum motion speed, and lost-track count");
    }
}

void load_domain_frame(std::unordered_map<std::string, std::string>& config)
{
    using tmv::init::config::DomainMotionMode;
    global_domain_motion_mode = DomainMotionMode::Stationary;
    global_domain_frame       = tmv::domain::HorizontalDomainFrame::stationary();

    if (config.count("domain_motion.mode") &&
        !tmv::init::config::parse_domain_motion_mode(config["domain_motion.mode"], global_domain_motion_mode))
    {
        warn_invalid_config_value("domain_motion.mode",
                                  config["domain_motion.mode"],
                                  "stationary, constant_velocity, tracked_window, or moving_nest");
    }
    const auto finite_double = [&](const char* key, double& destination)
    {
        if (!config.count(key))
            return;
        if (!try_parse_double_value(config[key], destination))
            warn_invalid_config_value(key, config[key], "a finite number");
    };
    finite_double("domain_motion.frame_velocity.u_ms", global_domain_frame.velocity_ms.u_ms);
    finite_double("domain_motion.frame_velocity.v_ms", global_domain_frame.velocity_ms.v_ms);
    global_domain_frame.validate();

    if (global_domain_motion_mode == DomainMotionMode::TrackedWindow
        && (global_domain_frame.velocity_ms.u_ms != 0.0
            || global_domain_frame.velocity_ms.v_ms != 0.0))
        throw std::runtime_error(
            "domain_motion.mode=tracked_window requires zero continuous frame velocity");
    if (global_domain_motion_mode == DomainMotionMode::Stationary &&
        (global_domain_frame.velocity_ms.u_ms != 0.0 || global_domain_frame.velocity_ms.v_ms != 0.0))
        throw std::runtime_error("domain_motion.mode=stationary requires zero frame velocity");
}

void validate_domain_motion_execution()
{
    using tmv::init::config::DomainMotionMode;
    if (global_terrain_config.scheme_id != "none"
        && global_terrain_config.horizontal_placement_id == "earth_fixed"
        && (global_domain_frame.velocity_ms.u_ms != 0.0
            || global_domain_frame.velocity_ms.v_ms != 0.0)
        && !runtime_configuration.admission.earth_fixed_terrain_motion)
        throw std::runtime_error(
            "continuous Earth-fixed motion requires the maintained Bell remapping profile");
    if (global_domain_motion_mode != DomainMotionMode::ConstantVelocity)
        return;
    if (global_coordinate_system != CoordinateSystem::Cartesian)
        throw std::runtime_error(
            "domain_motion.mode=constant_velocity currently requires grid.coordinate_system=cartesian");
    ComputeBackendKind backend = ComputeBackendKind::Cpu;
    if (!parse_compute_backend_kind(
            global_compute_backend_config.backend, backend))
        throw std::runtime_error("domain_motion.mode=constant_velocity currently requires an "
                                 "explicit supported numerics.compute.backend");
    if (backend != ComputeBackendKind::Cpu
        && global_stagger_type != StaggerType::CGrid)
        throw std::runtime_error("domain_motion.mode=constant_velocity with accelerator execution currently "
                                 "requires grid.staggering=c_grid");
}

void validate_stationary_nesting_effective_profile(
    const StationaryNestingLiveExecution* live)
{
    const auto& nesting = runtime_configuration.nesting;
    const bool moving_requested =
        global_domain_motion_mode ==
            tmv::init::config::DomainMotionMode::MovingNest;
    if (moving_requested && !nesting.enabled)
        throw std::runtime_error(
            "domain_motion.mode=moving_nest requires domain_motion.nesting.enabled=true");
    if (!nesting.enabled)
        return;
    if (nesting.feedback == tmv::init::config::NestingFeedback::TwoWay &&
        !runtime_configuration.admission.two_way_nested)
    {
        throw std::runtime_error(
            "domain_motion.nesting.feedback=two_way is admitted only for the exact dry/Kessler/Lin "
            "flat 2:1, grid-attached Bell/Schär 2:1/3:1, or tracked-window recycling or moving-child CPU profiles with complete interface-register, native-dual "
            "restriction, and local reflux ownership");
    }
    ComputeBackendKind requested_kind = ComputeBackendKind::Cpu;
    const bool requested_backend_valid = parse_compute_backend_kind(
        global_compute_backend_config.backend, requested_kind);
    const std::string requested_backend =
        requested_backend_valid
            ? compute_backend_kind_name(requested_kind)
            : std::string{};
    const bool internal_moving_campaign =
        internal_moving_nest_qualification && moving_requested;
    const bool public_qualified_moving_profile =
        !internal_moving_nest_qualification && moving_requested &&
        global_execution_mode ==
            tmv::init::config::ExecutionMode::Qualified;
    const bool moving_campaign =
        internal_moving_campaign || public_qualified_moving_profile;
    const bool nested_recycling_campaign =
        runtime_configuration.admission.nested_grid_recycling &&
        (global_domain_motion_mode ==
             tmv::init::config::DomainMotionMode::TrackedWindow ||
         global_domain_motion_mode ==
             tmv::init::config::DomainMotionMode::ConstantVelocity);
    const bool common =
        (global_domain_motion_mode ==
             tmv::init::config::DomainMotionMode::Stationary ||
         moving_campaign || nested_recycling_campaign) &&
        global_coordinate_system == CoordinateSystem::Cartesian &&
        global_stagger_type == StaggerType::CGrid &&
        (global_terrain_config.scheme_id == "none" ||
         runtime_configuration.admission.active_terrain_nested) &&
        (live != nullptr || !global_simulation_storage.nested.config.enabled);
    if (!common || !requested_backend_valid)
    {
        throw std::runtime_error(
            "enabled domain_motion.nesting requires stationary flat Cartesian C-grid execution "
            "on an explicit cpu, vulkan, or metal backend without the legacy nested-grid initializer");
    }

    if (moving_campaign)
    {
        const bool exact_vulkan_backend =
            requested_kind == ComputeBackendKind::Vulkan &&
            !global_compute_backend_config.allow_fallback &&
            !global_gpu_apply_microphysics &&
            !global_gpu_apply_slow && global_gpu_apply_sgs;
        const bool exact_metal_backend =
            requested_kind == ComputeBackendKind::Metal &&
            !global_compute_backend_config.allow_fallback &&
            !global_gpu_apply_microphysics &&
            !global_gpu_apply_slow && global_gpu_apply_sgs;
        const bool exact_backend = internal_moving_campaign
            ? (requested_kind == ComputeBackendKind::Cpu ||
               exact_vulkan_backend || exact_metal_backend)
            : (requested_kind == ComputeBackendKind::Cpu ||
               exact_vulkan_backend || exact_metal_backend);
        const bool exact_execution = internal_moving_campaign
            ? global_execution_mode ==
                  tmv::init::config::ExecutionMode::Exploratory
            : global_execution_mode ==
                  tmv::init::config::ExecutionMode::Qualified;
        const bool exact_campaign =
            exact_backend && exact_execution &&
            NR == 135 && NTH == 108 && NZ == 50 &&
            canonical_qualified_parameter(dr, 1200.0) &&
            canonical_qualified_parameter(dtheta, 1200.0) &&
            canonical_qualified_parameter(dz, 320.0) &&
            nesting.parent_ratio_x == 2 &&
            nesting.parent_ratio_y == 2 &&
            nesting.parent_time_step_ratio == 2 &&
            nesting.child_origin_parent_cell_i == 2 &&
            nesting.child_origin_parent_cell_j == 1 &&
            nesting.child_cells_x == 156 &&
            nesting.child_cells_y == 166 &&
            nesting.relaxation_width_child_cells == 8 &&
            nesting.boundary_stencil_parent_cells == 1 &&
            canonical_qualified_parameter(
                nesting.relaxation_e_folding_time_s, 30.0) &&
            global_storm_tracking_config.enabled &&
            !global_storm_tracking_config.recycle_domain &&
            canonical_qualified_parameter(
                global_storm_tracking_config.interval_s, 30.0) &&
            canonical_qualified_parameter(
                global_storm_tracking_config.inner_margin_m, 15600.0) &&
            global_storm_tracking_config.maximum_shift_cells == 14 &&
            canonical_qualified_parameter(
                global_storm_tracking_config.cooldown_s, 300.0) &&
            canonical_qualified_parameter(
                global_storm_tracking_config.minimum_confidence, 0.65) &&
            global_microphysics_scheme == "kessler" &&
            global_turbulence_config.scheme_id == "smagorinsky" &&
            global_turbulence_config.mode == "3d" &&
            global_radiation_config.scheme_id == "none" &&
            global_boundary_layer_config.scheme_id == "none" &&
            !global_boundary_layer_config.apply_surface_fluxes &&
            global_chaos_config.scheme_id == "none" &&
            global_terrain_config.scheme_id == "none" &&
            global_coriolis_enabled && global_damping_enabled &&
            global_lateral_sponge_enabled &&
            global_lateral_sponge_width == 10 &&
            canonical_qualified_parameter(
                global_lateral_sponge_alpha, 0.05) &&
            global_w_only_top_sponge &&
            global_w_only_depth_cells == 8 &&
            canonical_qualified_parameter(
                global_w_only_alpha_max, 0.2) &&
            global_lateral_bc_west ==
                tmv::init::config::LateralBC::Radiative &&
            global_lateral_bc_east ==
                tmv::init::config::LateralBC::Radiative &&
            global_lateral_bc_south ==
                tmv::init::config::LateralBC::Radiative &&
            global_lateral_bc_north ==
                tmv::init::config::LateralBC::Radiative &&
            global_bottom_bc == tmv::init::config::BottomBC::RigidSlip &&
            global_top_bc == tmv::init::config::TopBC::Rigid;
        const bool exact_two_way = !internal_moving_campaign &&
            (requested_kind == ComputeBackendKind::Cpu || runtime_configuration.admission.two_way_accelerator_reference) &&
            runtime_configuration.admission.moving_nest && runtime_configuration.admission.two_way_nested;
        if (!exact_campaign && !exact_two_way)
        {
            throw std::runtime_error(internal_moving_campaign
                ? "the internal moving-nest campaign accepts only its exact versioned CPU or fallback-off Vulkan/Metal exploratory 1.2-km-parent/600-m-child supercell profile"
                : "public moving_nest accepts only the exact maintained CPU or fallback-off Vulkan/Metal profiles");
        }
    }

    const QualifiedStationaryNestingProfile qualified_profile =
        classify_qualified_stationary_nesting_profile();
    if (!moving_campaign && !nested_recycling_campaign &&
        (requested_backend == "vulkan" ||
         global_execution_mode ==
             tmv::init::config::ExecutionMode::Qualified) &&
        qualified_profile == QualifiedStationaryNestingProfile::None)
    {
        throw std::runtime_error(
            "domain_motion.nesting requires a maintained fallback-off reference: the named one-way profiles or stationary dry/Kessler/Lin and cloud-seeded Milbrandt flat 2:1 and Bell/Schär 2:1/3:1 two-way profiles on CPU/Metal/Vulkan with TVD/MC or WENO. These profiles use KW C-grid dynamics, split-explicit stepping, inactive explicit "
            "diffusion, profile-scoped column physics, disabled Coriolis/sponges/tracking, and zero-gradient/rigid-slip/rigid boundaries");
    }

    const bool native_device_profile_ready =
        requested_kind == ComputeBackendKind::Cpu ||
        (live != nullptr && live->native_dry_cgrid_ready &&
         (global_turbulence_config.scheme_id != "smagorinsky" ||
          live->native_smagorinsky_ready) &&
         (global_microphysics_scheme != "kessler" ||
          live->native_kessler_ready));
    if (live != nullptr &&
        (!live->initialized || live->active_backend != requested_backend ||
         live->fallback_active || !live->terrain_ready ||
         !native_device_profile_ready))
    {
        throw std::runtime_error(
            "stationary nesting live execution does not match its requested backend, fallback, terrain, native C-grid, or selected native physics capability profile");
    }
    const PhysicalCellDomain physical = physical_scalar_domain(
        NR, NTH, NZ, StaggerType::CGrid);
    tmv::domain::OneWayNestRelationshipInput input;
    input.parent_cells_x = physical.nx();
    input.parent_cells_y = physical.ny();
    input.parent_levels_z = NZ;
    input.parent_dx_m = dr;
    input.parent_dy_m = dtheta;
    input.parent_dt_s = std::max(
        global_configured_dt_large_s,
        global_time_stepping_config.adaptive_dt
            ? global_time_stepping_config.dt_max : dt);
    input.parent_ratio_x = nesting.parent_ratio_x;
    input.parent_ratio_y = nesting.parent_ratio_y;
    input.parent_time_step_ratio = nesting.parent_time_step_ratio;
    input.child_origin_parent_cell_i = nesting.child_origin_parent_cell_i;
    input.child_origin_parent_cell_j = nesting.child_origin_parent_cell_j;
    input.child_cells_x = nesting.child_cells_x;
    input.child_cells_y = nesting.child_cells_y;
    input.child_levels_z = NZ;
    input.relaxation_width_child_cells =
        nesting.relaxation_width_child_cells;
    input.boundary_stencil_parent_cells =
        nesting.boundary_stencil_parent_cells;
    (void)tmv::domain::prepare_one_way_nest_relationship(input);
}

void load_validation(std::unordered_map<std::string, std::string>& config)
{
    if (config.count("validation.guard_mode"))
    {
        tmv::GuardMode mode = global_validation_policy.mode;
        if (tmv::parse_guard_mode(config["validation.guard_mode"], mode))
        {
            global_validation_policy.mode = mode;
        }
        else
        {
            warn_invalid_config_value("validation.guard_mode",
                                      config["validation.guard_mode"],
                                      "one of: off, warn, error (config), sanitize, strict (CLI)");
        }
    }
    if (config.count("validation.guard_fail_on"))
    {
        tmv::GuardFailOn fail_on = global_validation_policy.fail_on;
        if (tmv::parse_guard_fail_on(config["validation.guard_fail_on"], fail_on))
        {
            global_validation_policy.fail_on = fail_on;
        }
        else
        {
            warn_invalid_config_value(
                "validation.guard_fail_on", config["validation.guard_fail_on"], "one of: nonfinite, bounds, both");
        }
    }
    if (config.count("validation.guard_scope"))
    {
        tmv::StrictGuardScope scope = global_validation_policy.strict_scope;
        if (tmv::parse_strict_guard_scope(config["validation.guard_scope"], scope))
        {
            global_validation_policy.strict_scope = scope;
        }
        else
        {
            warn_invalid_config_value(
                "validation.guard_scope", config["validation.guard_scope"], "one of: required, exported");
        }
    }
    if (config.count("validation.report_path"))
    {
        global_validation_report_path = config["validation.report_path"];
    }
}

struct RuntimeMemoryProfile
{
    int field_equivalents = 0;
    // Small-grid calibration shows process/runtime/allocator overhead that
    // does not scale with Field3D volume. The worst three-repeat deficit was
    // 6.1 MiB and a later allocator outlier consumed another 4 MiB; reserve a
    // rounded 16 MiB independently of scheme field equivalents.
    uint64_t                  fixed_overhead_bytes = uint64_t(16) << 20;
    RuntimeMemoryPhaseProfile phases;
    std::string               label;
};

constexpr int kFlatNativeTvdStepRollbackFieldEquivalents = 176;
constexpr int kNativeAcousticTransportCandidateFields = 4;

std::string
memory_profile_value(const std::unordered_map<std::string, std::string>& config, const char* key, const char* fallback)
{
    const auto it = config.find(key);
    return it == config.end() ? std::string(fallback) : to_lower_copy(it->second);
}

std::vector<std::string> selected_output_fields_for_memory(const OutputConfig* output_config)
{
    if (output_config == nullptr)
    {
        return {};
    }
    if (output_config->preset == FieldPreset::custom)
    {
        std::vector<std::string> valid;
        valid.reserve(output_config->custom_fields.size());
        std::unordered_set<std::string> seen;
        for (const std::string& name : output_config->custom_fields)
        {
            if (find_output_field(name) != nullptr && seen.insert(name).second)
            {
                valid.push_back(name);
            }
        }
        if (!valid.empty())
        {
            return valid;
        }
        return get_preset_fields(FieldPreset::minimal);
    }
    return get_preset_fields(output_config->preset);
}

struct OutputMemoryFieldCounts
{
    int                   core_volume_fields       = 0;
    int                   derived_volume_fields    = 0;
    int                   map_fields               = 0;
    int                   centered_velocity_fields = 0;
    int                   slice_scratch_arrays     = 0;
    OutputFieldDependency dependencies             = OutputFieldDependency::None;
};

OutputMemoryFieldCounts count_output_memory_fields(const std::vector<std::string>& selected_fields)
{
    OutputMemoryFieldCounts         counts;
    std::unordered_set<std::string> active_fields;
    active_fields.reserve(selected_fields.size());
    for (const std::string& name : selected_fields)
    {
        active_fields.insert(name);
        const OutputFieldDescriptor* field = find_output_field(name);
        if (field == nullptr)
        {
            continue;
        }
        counts.dependencies |= field->dependencies;
        if (field->kind == OutputFieldKind::Core3D)
        {
            ++counts.core_volume_fields;
        }
        else if (field->kind == OutputFieldKind::DerivedVolume)
        {
            ++counts.derived_volume_fields;
        }
        else
        {
            ++counts.map_fields;
        }
    }
    const CenteredVelocityStagingPlan staging =
        centered_velocity_staging_plan(true, output_dependencies_for_fields(selected_fields));
    counts.centered_velocity_fields = staging.field_count();
    counts.slice_scratch_arrays     = derived_slice_scratch_array_count(active_fields);
    return counts;
}

OutputFieldDependency live_field_dependencies(bool live_shm, const std::vector<std::string>* live_shm_fields)
{
    if (!live_shm || live_shm_fields == nullptr)
    {
        return OutputFieldDependency::None;
    }
    return output_dependencies_for_fields(*live_shm_fields);
}

int checkpoint_durable_field_equivalents(
    const SimulationStorageState& state)
{
    int fields = kCheckpointCoreDurableFieldEquivalents;
    const std::string microphysics =
        to_lower_copy(state.configuration.microphysics_scheme);
    const std::string turbulence =
        to_lower_copy(state.runtime.turbulence.config.scheme_id);
    const std::string radiation =
        to_lower_copy(state.runtime.radiation.config.scheme_id);
    const std::string boundary_layer =
        to_lower_copy(state.runtime.boundary_layer.config.scheme_id);
    const std::string chaos =
        to_lower_copy(state.runtime.chaos.config.scheme_id);
    const std::string terrain =
        to_lower_copy(state.runtime.terrain.config.scheme_id);

    // Runtime sections carry these additional full-volume continuation
    // fields beyond the 15 prognostics, six tendencies, and 12 diagnostics.
    if (radiation != "none") fields += 1;
    if (turbulence != "none") fields += 6;
    if (microphysics != "none") fields += 1;
    if (microphysics == "milbrandt") fields += 13;
    if (boundary_layer != "none") fields += 8;
    if (chaos == "boundary_layer") fields += 9;
    else if (chaos == "full_stochastic") fields += 17;
    if (terrain != "none") fields += 20;
    return fields;
}

RuntimeMemoryProfile build_runtime_memory_profile(const SimulationStorageState& state,
                                                  const OutputConfig*                                 output_config,
                                                  int                                                 write_every_s,
                                                  bool                                                live_shm,
                                                  const std::vector<std::string>*                     live_shm_fields,
                                                  bool trajectories_enabled,
                                                  bool atomic_lin_outputs = false)
{
    // The post-CPU-016 diagnostic-free Kessler + Smagorinsky artifact measured
    // 309,739,520 bytes at 128x128x64, or 73.85 full-field equivalents. Start
    // from the audited 65-field solver inventory plus a conservative 10-field
    // process/allocator reserve. Lazy diagnostics are added only for a real
    // output/live consumer. Scheme deltas deliberately err high; this remains
    // a warning model rather than an allocator admission promise.
    RuntimeMemoryProfile profile;
    const std::string    microphysics   = to_lower_copy(state.configuration.microphysics_scheme);
    const std::string    turbulence     = to_lower_copy(state.runtime.turbulence.config.scheme_id);
    const std::string    boundary_layer = to_lower_copy(state.runtime.boundary_layer.config.scheme_id);
    const std::string    radiation      = to_lower_copy(state.runtime.radiation.config.scheme_id);
    const std::string    terrain        = to_lower_copy(state.runtime.terrain.config.scheme_id);
    std::string          advection      = to_lower_copy(state.runtime.advection.config.scheme_id);


    const std::string integrator = to_lower_copy(state.runtime.time_stepping.config.scheme_id);
    int rk_workspace_fields = 0;
    if (integrator == "rk3" || integrator == "ssprk3" || integrator == "rk4")
    {
        const int stages = integrator == "rk4" ? 4 : 3;
        const int private_fields = microphysics == "thompson" ? 5 :
            microphysics == "milbrandt" ? 6 : 0;
        rk_workspace_fields = (stages + 3) *
            tmv::TimeSteppingUnsplitRkWorkspace::field_count + 10 +
            (stages + 4) * private_fields + (state.runtime.compute.config.backend != "cpu" ? 32 : 0);
    }
    // Atomic native publication removes the scheme's eight duplicate transaction fields.
    const int native_lin_workspace_fields = microphysics == "lin" && state.runtime.compute.config.backend != "cpu"
        ? (atomic_lin_outputs ? 11 : 19) : 0;
    const int solver_resident_fields =
        estimate_runtime_solver_fields({microphysics, turbulence, advection, boundary_layer, radiation}) +
        rk_workspace_fields + native_lin_workspace_fields +
        (state.runtime.compute.config.backend != "cpu" ? 9 + kNativeAcousticTransportCandidateFields : 0) +
        (state.configuration.admission.two_way_accelerator_reference && state.runtime.compute.config.backend != "cpu" ? 96 : 0) + 48 + (turbulence == "tke" && state.runtime.compute.config.backend != "cpu" ? 64 : 0) +
        (state.runtime.diffusion.config.scheme_id == "implicit" && state.runtime.compute.config.backend != "cpu" ? 64
                                                                                                                 : 0) +
        (state.configuration.admission.native_weno5 || state.configuration.admission.native_dry_tvd ? 9 : 0) +
        ((state.configuration.admission.native_dry_tvd ||
          state.configuration.admission.static_vertical_only_sgs_cpu)
             ? kFlatNativeTvdStepRollbackFieldEquivalents : 0) +
        (microphysics == "thompson"    ? 5
         : microphysics == "milbrandt" ? 6
                                       : 0);

    const bool                     output_enabled = write_every_s > 0 && output_config != nullptr;
    const std::vector<std::string> selected_fields =
        output_enabled ? selected_output_fields_for_memory(output_config) : std::vector<std::string>{};
    const OutputMemoryFieldCounts output_counts = count_output_memory_fields(selected_fields);
    OutputFieldDependency         diagnostic_dependencies =
        output_enabled ? output_counts.dependencies : OutputFieldDependency::None;
    diagnostic_dependencies |= live_field_dependencies(live_shm, live_shm_fields);
    if (trajectories_enabled)
    {
        // CLI trajectories default to automatic centering, which needs the
        // six-field gridded vorticity closure at seed time. Unsupported grids
        // also retain that fallback after seeding. Account for the allocation
        // even when file output and live publishing are disabled.
        diagnostic_dependencies |= OutputFieldDependency::VorticityDiagnostics;
    }
    const int lazy_diagnostic_fields =
        (output_field_has_dependency(diagnostic_dependencies, OutputFieldDependency::RadarDiagnostic) ? 1 : 0) +
        (output_field_has_dependency(diagnostic_dependencies, OutputFieldDependency::VorticityDiagnostics) ? 6 : 0) +
        (output_field_has_dependency(diagnostic_dependencies, OutputFieldDependency::AngularMomentumDiagnostics) ? 2
                                                                                                                 : 0) +
        (output_field_has_dependency(diagnostic_dependencies, OutputFieldDependency::PressureDiagnostics) ? 3 : 0);

    RuntimeMemoryPhaseInputs phase_inputs;
    phase_inputs.solver_resident_fields = solver_resident_fields;
    phase_inputs.lazy_diagnostic_fields = lazy_diagnostic_fields;
    phase_inputs.output_enabled         = output_enabled;
    phase_inputs.volume_output =
        output_enabled &&
        (output_config->format == OutputFormat::npy_3d || output_config->format == OutputFormat::netcdf ||
         output_config->format == OutputFormat::csv || output_config->format == OutputFormat::zfp);
    phase_inputs.c_grid                   = global_stagger_type == StaggerType::CGrid;
    phase_inputs.centered_velocity_fields = output_counts.centered_velocity_fields;
    // Container and terrain output retain derived fields and geometry through
    // publication; synchronous core fields borrow stable solver/frame storage.
    phase_inputs.terrain_geometry = output_enabled && phase_inputs.volume_output && terrain != "none";
    phase_inputs.async_io =
        output_enabled && phase_inputs.volume_output &&
        (requires_snapshot_assembly(output_config->format, output_config->async_io) || phase_inputs.terrain_geometry);
    phase_inputs.borrow_core_output_fields = output_config != nullptr && !output_config->async_io;
    phase_inputs.core_volume_output_fields    = output_counts.core_volume_fields;
    phase_inputs.derived_volume_output_fields = output_counts.derived_volume_fields;
    phase_inputs.map_output_fields            = output_counts.map_fields;
    phase_inputs.slice_scratch_arrays         = output_counts.slice_scratch_arrays;
    phase_inputs.vertical_levels              = state.grid.nz;
    phase_inputs.horizontal_rows              = state.grid.nth;
    if (output_enabled && output_config->format == OutputFormat::zfp && output_config->zfp_keyframe_interval > 0)
    {
        phase_inputs.zfp_history_depth = output_config->zfp_predictive_delta ? 2 : 1;
    }
    profile.phases            = estimate_runtime_memory_phases(phase_inputs);
    profile.field_equivalents = profile.phases.high_water_field_equivalents;
    if (output_enabled && output_config->format == OutputFormat::zfp &&
        output_config->zfp_field_execution == "field_parallel")
    {
        const std::uint64_t parallel_scratch =
            static_cast<std::uint64_t>(output_config->zfp_field_scratch_mib) *
            1024ULL * 1024ULL;
        if (profile.fixed_overhead_bytes >
            std::numeric_limits<std::uint64_t>::max() - parallel_scratch)
            throw std::runtime_error(
                "ZFP field-parallel scratch exceeds the representable memory plan");
        profile.fixed_overhead_bytes += parallel_scratch;
        profile.label += ", zfp_field_parallel_scratch=" +
            std::to_string(parallel_scratch) + " bytes";
    }
    if (terrain != "none")
    {
        using namespace tmv::radiative_transfer;
        const auto nx = static_cast<std::uint64_t>(state.grid.nr);
        const auto ny = static_cast<std::uint64_t>(state.grid.nth);
        const auto nz = static_cast<std::uint64_t>(state.grid.nz);
        profile.fixed_overhead_bytes = memory_sum(profile.fixed_overhead_bytes,
            terrain_geometry_memory_bytes(nx, ny, nz));
        if (state.configuration.admission.native_mapped_terrain ||
            (state.configuration.admission.earth_fixed_terrain_motion && state.runtime.compute.config.backend != "cpu"))
        {
            // Private candidates, packed metrics, Heun stages, transport and
            // device upload/output buffers coexist with the ordinary solver.
            const auto native_bytes = memory_product(128 * sizeof(float),
                memory_product(memory_product(nx, ny), nz));
            profile.fixed_overhead_bytes = memory_sum(profile.fixed_overhead_bytes, native_bytes);
            profile.label += ", native_mapped_workspace=" + std::to_string(native_bytes) + " bytes";
        }
        if (radiation != "none" && state.runtime.radiation.config.do_sw)
        {
            const auto& cfg = state.runtime.radiation.config;
            const auto physical = physical_scalar_domain(state.grid.nr, state.grid.nth,
                state.grid.nz, state.configuration.stagger_type);
            const auto bytes = mapped_shortwave_workspace_bytes(
                memory_product(physical.nx(), physical.ny()), physical.nz(),
                radiation == "rrtmg", cfg.terrain_sw_cosine_nodes, cfg.terrain_sw_azimuth_nodes);
            profile.fixed_overhead_bytes = memory_sum(profile.fixed_overhead_bytes, bytes);
            profile.label += ", mapped_shortwave_workspace=" + std::to_string(bytes) + " bytes";
        }

    }
    const bool terrain_ground_velocity_required =
        state.configuration.storm_tracking.enabled ||
        output_field_has_dependency(diagnostic_dependencies, OutputFieldDependency::CenteredW);
    const std::uint64_t terrain_ground_velocity_bytes =
        (terrain != "none" && phase_inputs.c_grid && terrain_ground_velocity_required)
            ? static_cast<std::uint64_t>(state.grid.nr) * static_cast<std::uint64_t>(state.grid.nth) * sizeof(double)
            : 0;
    profile.fixed_overhead_bytes += terrain_ground_velocity_bytes;
    std::uint64_t storm_tracker_bytes = 0;
    if (state.configuration.storm_tracking.enabled)
    {
        const std::uint64_t physical_nx        = static_cast<std::uint64_t>(std::max(0, state.grid.nr - (state.grid.nr >= 3 ? 2 : 0)));
        const std::uint64_t physical_ny        = static_cast<std::uint64_t>(std::max(0, state.grid.nth - (state.grid.nth >= 3 ? 2 : 0)));
        const std::uint64_t columns            = physical_nx * physical_ny;
        const std::uint64_t component_capacity = tmv::diagnostics::storm_feature_component_capacity(
            static_cast<std::size_t>(physical_nx), static_cast<std::size_t>(physical_ny));
        // Three column signals, condensate-parent labels, watershed labels,
        // BFS queue entries, and the restart-exact selected-partition mask.
        const std::uint64_t per_column_bytes =
            3 * sizeof(double) + 2 * sizeof(int) + sizeof(std::size_t) + sizeof(std::uint8_t);
        storm_tracker_bytes =
            columns * per_column_bytes + component_capacity * sizeof(tmv::diagnostics::StormFeatureComponentWorkspace);
        profile.fixed_overhead_bytes += storm_tracker_bytes;
    }

    std::uint64_t stochastic_snapshots = 0;
    const auto& chaos = state.runtime.chaos.config;
    if (chaos.scheme_id != "none")
    {
        const bool micro_snapshot =
            std::find(chaos.apply_to_tendencies.begin(), chaos.apply_to_tendencies.end(),
                "microphysics") != chaos.apply_to_tendencies.end();
        const int processes = chaos.scheme_id == "boundary_layer" ? 1 : 3;
        const auto add_snapshot = [&](int nx, int ny, double dx, double dy) {
            const auto bytes = tmv::chaos::stochastic_snapshot_bytes(
                nx, ny, state.grid.nz, dx, dy, chaos.Lx, chaos.Ly, processes);
            if (bytes > std::numeric_limits<std::uint64_t>::max() - stochastic_snapshots)
                throw std::overflow_error("stochastic transaction memory overflow");
            stochastic_snapshots += bytes;
        };
        if (micro_snapshot) add_snapshot(state.grid.nr, state.grid.nth, state.grid.dx, state.grid.dy);
        const auto& nesting = state.configuration.nesting;
        if (nesting.enabled)
        {
            add_snapshot(nesting.child_cells_x + 2, nesting.child_cells_y + 2,
                state.grid.dx / nesting.parent_ratio_x, state.grid.dy / nesting.parent_ratio_y);
            if (micro_snapshot) add_snapshot(nesting.child_cells_x + 2, nesting.child_cells_y + 2,
                state.grid.dx / nesting.parent_ratio_x, state.grid.dy / nesting.parent_ratio_y);
        }
        if (stochastic_snapshots > std::numeric_limits<std::uint64_t>::max() - profile.fixed_overhead_bytes)
            throw std::overflow_error("stochastic transaction memory exceeds host budget range");
        profile.fixed_overhead_bytes += stochastic_snapshots;
    }

    if (terrain != "none")
    {
        using namespace tmv::radiative_transfer;
        const auto nx=static_cast<std::uint64_t>(state.grid.nr);
        const auto ny=static_cast<std::uint64_t>(state.grid.nth);
        const auto nz=static_cast<std::uint64_t>(state.grid.nz);
        if (state.runtime.terrain.config.horizontal_placement_id == "earth_fixed"
            && !state.configuration.nesting.enabled)
        {
            const auto cells = memory_product(memory_product(nx, ny), nz);
            const auto candidate = memory_sum(profile.fixed_overhead_bytes,
                memory_product(memory_product(cells, sizeof(float)), profile.phases.resident_fields));
            const auto remap = (state.configuration.domain_frame.velocity_ms.u_ms != 0.0
                || state.configuration.domain_frame.velocity_ms.v_ms != 0.0)
                ? earth_fixed_remap_scratch_bytes(nx, ny, nz) : 0;
            profile.fixed_overhead_bytes = memory_sum(profile.fixed_overhead_bytes, memory_sum(candidate, remap));
            profile.label += ", earth_step_candidate=" + std::to_string(candidate)
                + " bytes, earth_remap_scratch=" + std::to_string(remap) + " bytes";
        }
    }

    if(state.configuration.nesting.enabled && state.configuration.nesting.feedback==tmv::init::config::NestingFeedback::TwoWay && microphysics!="none")
    {
        using namespace tmv::radiative_transfer;
        const auto& nest = state.configuration.nesting;
        if (nest.parent_ratio_x <= 0 || nest.parent_ratio_y <= 0)
            throw std::invalid_argument("moist nesting memory requires valid refinement");
        const auto perimeter = memory_product(2,
                                              memory_sum(std::uint64_t(nest.child_cells_x / nest.parent_ratio_x),
                                                         std::uint64_t(nest.child_cells_y / nest.parent_ratio_y)));
        const auto strips =
            memory_product(memory_product(perimeter, std::uint64_t(state.grid.nz)), 7 * 2 * sizeof(double));
        const auto scalar =
            memory_product(memory_product(memory_product(std::uint64_t(state.grid.nr), std::uint64_t(state.grid.nth)),
                                          std::uint64_t(state.grid.nz)),
                           sizeof(float) + sizeof(double));
        const auto child_candidates = memory_product(
            memory_product(memory_product(std::uint64_t(nest.child_cells_x) + 2, std::uint64_t(nest.child_cells_y) + 2),
                           std::uint64_t(state.grid.nz)),
            7 * sizeof(float));
        const auto workspace         = memory_sum(memory_product(6, strips), memory_sum(scalar, child_candidates));
        profile.fixed_overhead_bytes = memory_sum(profile.fixed_overhead_bytes, workspace);
        profile.label += ", moist_feedback_workspace=" + std::to_string(workspace) + " bytes";
    }

    std::ostringstream label;
    label << "microphysics=" << microphysics << ", turbulence=" << turbulence << ", advection=" << advection
          << ", boundary_layer=" << boundary_layer << ", radiation=" << radiation
          << ", solver_resident=" << profile.phases.solver_resident_fields
          << " fields (65 audited + 10 calibration reserve before scheme deltas"
          << ", RK workspace=" << rk_workspace_fields << ")"
          << ", stochastic_snapshot_bytes=" << stochastic_snapshots
          << ", lazy_diagnostics=" << profile.phases.lazy_diagnostic_fields
          << " fields, output_working=" << profile.phases.output_working_field_equivalents
          << " fields [centered=" << profile.phases.centered_velocity_fields
          << ", slice_scratch=" << profile.phases.slice_scratch_field_equivalents
          << ", current=" << profile.phases.current_output_field_equivalents
          << ", concurrent=" << profile.phases.concurrent_snapshot_field_equivalents
          << ", compression_history=" << profile.phases.compression_history_field_equivalents
          << ", terrain_geometry=" << (phase_inputs.terrain_geometry ? "float64 center+interface" : "none")
          << ", terrain_ground_velocity_2d_bytes=" << terrain_ground_velocity_bytes
          << ", storm_tracker_2d_bytes=" << storm_tracker_bytes << "], high_water=" << profile.field_equivalents
          << " fields";
    profile.label = label.str() + profile.label;
    return profile;
}

} // namespace

nimbus::output::QualifiedRadarZfpV4Context
make_qualified_radar_zfp_v4_context(
    const tmv::init::config::ConfigV1& config,
    const OutputConfig& output,
    double write_every_s,
    bool live_shm,
    bool trajectories_enabled)
{
    nimbus::output::QualifiedRadarZfpV4Context context;
    context.qualified = config.extensions.execution.mode ==
        tmv::init::config::ExecutionMode::Qualified;
    context.cpu = config.extensions.compute.backend ==
        tmv::init::config::ComputeBackend::CPU;
    context.fallback_off = !config.extensions.compute.allow_fallback;
    context.cartesian = config.grid.coordinate_system == CoordinateSystem::Cartesian;
    context.c_grid = config.grid.staggering == StaggerType::CGrid;
    context.flat = config.terrain.scheme == tmv::init::config::TerrainScheme::None;
    context.stationary = config.domain_motion.mode ==
        tmv::init::config::DomainMotionMode::Stationary &&
        config.domain_motion.frame_velocity.u_ms == 0.0 &&
        config.domain_motion.frame_velocity.v_ms == 0.0;
    context.no_nesting = !config.domain_motion.nesting.enabled;
    context.no_tracking = !config.domain_motion.tracking.enabled &&
        !config.domain_motion.tracking.recycle_domain;
    context.no_side_stream = output.qualification_diagnostics_preset ==
        QualificationDiagnosticsPreset::none;
    context.no_tiered_cadence = !output.tiered_write_cadence;
    context.no_live_shm = !live_shm;
    context.no_trajectories = !output.trajectories && !trajectories_enabled;
    context.deterministic_reductions =
        config.extensions.diagnostics.deterministic_reductions;
    context.reviewed_source_profile =
        tmv::init::config::is_public_qualified_radar_profile(config);
    context.selected_radar_products = config.radar.products;
    context.format_tier = output.zfp_requested_format_tier;
    context.nx = config.grid.nx;
    context.ny = config.grid.ny;
    context.nz = config.grid.nz;
    context.dx_m = config.grid.dx;
    context.dy_m = config.grid.dy;
    context.dz_m = config.grid.dz;
    context.dt_large_s = config.time.dt_large;
    context.acoustic_substeps = config.time.acoustic_substeps;
    context.write_every_s = write_every_s;
    return context;
}

bool is_public_qualified_generated_metal_east_npy_profile(const tmv::init::config::ConfigV1& config)
{
    if (config.extensions.execution.mode != tmv::init::config::ExecutionMode::Qualified ||
        config.extensions.compute.backend != tmv::init::config::ComputeBackend::Metal ||
        config.output.format != tmv::init::config::OutputFormat::Npy3D ||
        !tmv::runtime::matches_generated_metal_unserialized_defaults(config))
    {
        return false;
    }
    try
    {
        const std::string serialized = tmv::init::config::serialize_v1_to_yaml(config);
        return nimbus::crypto::sha256_hex(serialized) ==
            tmv::runtime::kGeneratedMetalEastNpyCanonicalYamlSha256;
    }
    catch (const std::exception&)
    {
        return false;
    }
}

bool is_public_qualified_generated_metal_profile(const tmv::init::config::ConfigV1& config)
{
    if (config.extensions.execution.mode != tmv::init::config::ExecutionMode::Qualified ||
        config.extensions.compute.backend != tmv::init::config::ComputeBackend::Metal ||
        !tmv::runtime::matches_generated_metal_unserialized_defaults(config))
    {
        return false;
    }
    try
    {
        auto canonical = config;
        if (canonical.output.format == tmv::init::config::OutputFormat::ZfpHigh)
        {
#ifndef HAVE_ZFP
            return false;
#endif
            const std::string serialized = tmv::init::config::serialize_v1_to_yaml(config);
            const std::string_view text(serialized);
            constexpr std::string_view output_marker = "\noutput:\n";
            constexpr std::string_view extension_marker = "\n# Nimbus extensions";
            const auto marker = text.find(output_marker);
            if (marker == std::string_view::npos ||
                text.find(output_marker, marker + output_marker.size()) != std::string_view::npos)
            {
                return false;
            }
            const auto begin = marker + 1;
            const auto end = text.find(extension_marker, begin);
            if (end == std::string_view::npos)
            {
                return false;
            }
            const std::string digest = nimbus::crypto::sha256_hex(text.substr(begin, end - begin));
            if (std::string_view(digest) != tmv::runtime::kGeneratedMetalZfpTypedOutputBlockSha256)
            {
                return false;
            }
            canonical.output = tmv::init::config::OutputSection{};
            canonical.output.async_io = false;
        }
        else if (canonical.output.format == tmv::init::config::OutputFormat::Netcdf)
        {
            if (canonical.output.netcdf_deflate_level != 2 || !canonical.output.netcdf_shuffle ||
                canonical.output.netcdf_target_chunk_kib != 1024)
            {
                return false;
            }
            canonical.output.format = tmv::init::config::OutputFormat::Npy3D;
        }
        else if (canonical.output.format != tmv::init::config::OutputFormat::Npy3D)
        {
            return false;
        }
        const std::string serialized = tmv::init::config::serialize_v1_to_yaml(canonical);
        const std::string digest = nimbus::crypto::sha256_hex(serialized);
        return std::binary_search(
            tmv::runtime::kGeneratedMetalCanonicalYamlSha256.begin(),
            tmv::runtime::kGeneratedMetalCanonicalYamlSha256.end(),
            std::string_view(digest));
    }
    catch (const std::exception&)
    {
        return false;
    }
}

RuntimeScienceAdmission evaluate_runtime_science_admission(const tmv::init::config::ConfigV1& config)
{
    RuntimeScienceAdmission admission;
    admission.earth_fixed_terrain_motion =
        tmv::init::config::is_public_qualified_earth_fixed_terrain_motion_profile(config);
    admission.earth_fixed_terrain_recycling =
        tmv::init::config::
            is_public_qualified_earth_fixed_terrain_recycling_profile(
                config);
    admission.nested_grid_recycling =
        tmv::init::config::
            is_public_qualified_nested_grid_recycling_profile(
                config);
    admission.domain_recycling =
        tmv::init::config::is_public_qualified_domain_recycling_profile(
            config)
        || admission.earth_fixed_terrain_recycling ||
        admission.nested_grid_recycling;
    admission.moving_nest =
        tmv::init::config::is_public_qualified_moving_nest_profile(
            config);
    admission.two_way_accelerator_reference = tmv::init::config::is_public_qualified_accelerated_two_way_profile(config);
    admission.two_way_nested =
        tmv::init::config::is_public_qualified_two_way_nested_profile(
            config);
    admission.active_terrain_nested =
        tmv::init::config::
            is_public_qualified_active_terrain_nested_profile(
                config);
    admission.mapped_terrain =
        tmv::init::config::is_public_qualified_mapped_terrain_profile(
            config)
        || tmv::init::config::
            is_public_qualified_collocated_mapped_terrain_profile(
                config)
        || admission.earth_fixed_terrain_recycling
        || admission.earth_fixed_terrain_motion
        || admission.active_terrain_nested;
    admission.native_mapped_terrain =
        tmv::init::config::
            is_public_qualified_native_mapped_terrain_profile(
                config);
    admission.lin_rrtmg =
        tmv::init::config::is_public_qualified_lin_rrtmg_profile(
            config);
    admission.lin_storm = tmv::runtime::is_flat_lin_storm_profile(config);
    admission.thompson_cpu =
        tmv::init::config::is_public_qualified_thompson_cpu_profile(
            config);
    admission.thompson_native = tmv::init::config::is_public_ready_thompson_native_profile(config);
    admission.milbrandt_reference = tmv::init::config::is_public_ready_milbrandt_reference_profile(config);
    admission.cloud_top_radiative_ysu =
        tmv::init::config::
            is_public_qualified_cloud_top_radiative_ysu_profile(
                config);
    admission.radar =
        tmv::init::config::is_public_qualified_radar_profile(
            config);
    admission.user_filter_width =
        tmv::init::config::
            is_public_qualified_user_filter_width_profile(
                config);
    admission.richardson_smagorinsky =
        tmv::init::config::is_public_qualified_richardson_smagorinsky_profile(config);
    admission.dynamic_smagorinsky =
        tmv::init::config::
            is_public_qualified_dynamic_smagorinsky_profile(
                config);
    admission.static_vertical_only_sgs_cpu =
        tmv::init::config::is_transactional_static_vertical_only_cpu_candidate(config);
    admission.native_weno5 = tmv::init::config::is_public_ready_native_weno5_profile(config);
    const bool canonical_closed_boundaries =
        config.boundary_conditions.lateral.west == tmv::init::config::LateralBC::Rigid &&
        config.boundary_conditions.lateral.east == tmv::init::config::LateralBC::Rigid &&
        config.boundary_conditions.lateral.south == tmv::init::config::LateralBC::Rigid &&
        config.boundary_conditions.lateral.north == tmv::init::config::LateralBC::Rigid &&
        config.boundary_conditions.bottom == tmv::init::config::BottomBC::RigidSlip &&
        config.boundary_conditions.top == tmv::init::config::TopBC::Rigid;
    const bool mixed_open_boundaries =
        config.boundary_conditions.lateral.west == tmv::init::config::LateralBC::RigidSlip &&
        config.boundary_conditions.lateral.east == tmv::init::config::LateralBC::RigidSlip &&
        config.boundary_conditions.lateral.south == tmv::init::config::LateralBC::RigidSlip &&
        config.boundary_conditions.lateral.north == tmv::init::config::LateralBC::RigidSlip &&
        config.boundary_conditions.bottom == tmv::init::config::BottomBC::RigidNoSlip &&
        config.boundary_conditions.top == tmv::init::config::TopBC::Open;
    const bool east_radiative_open_boundaries =
        config.boundary_conditions.lateral.west == tmv::init::config::LateralBC::RigidSlip &&
        config.boundary_conditions.lateral.east == tmv::init::config::LateralBC::Radiative &&
        config.boundary_conditions.lateral.south == tmv::init::config::LateralBC::RigidSlip &&
        config.boundary_conditions.lateral.north == tmv::init::config::LateralBC::RigidSlip &&
        config.boundary_conditions.bottom == tmv::init::config::BottomBC::RigidNoSlip &&
        config.boundary_conditions.top == tmv::init::config::TopBC::Open;
    admission.native_dry_tvd =
        config.extensions.compute.backend == tmv::init::config::ComputeBackend::Metal &&
        !config.extensions.compute.allow_fallback &&
        config.grid.coordinate_system == CoordinateSystem::Cartesian &&
        config.grid.staggering == StaggerType::CGrid &&
        config.grid.nx == 18 && config.grid.ny == 18 && config.grid.nz == 18 &&
        config.grid.dx == 500.0 && config.grid.dy == 500.0 && config.grid.dz == 250.0 &&
        config.dynamics.scheme == "klemp_wilhelmson_split_cartesian_cgrid" &&
        config.time.integrator == tmv::init::config::TimeIntegrator::SplitExplicit &&
        config.time.dt_large == 0.1 && config.time.acoustic_substeps == 6 &&
        !config.time.adaptive_dt &&
        config.advection.scheme == tmv::init::config::AdvectionScheme::TVD &&
        config.microphysics.scheme == tmv::init::config::MicrophysicsScheme::None &&
        config.terrain.scheme == tmv::init::config::TerrainScheme::None &&
        config.domain_motion.mode == tmv::init::config::DomainMotionMode::Stationary &&
        !config.domain_motion.nesting.enabled && !config.domain_motion.tracking.enabled &&
        (canonical_closed_boundaries || mixed_open_boundaries || east_radiative_open_boundaries) &&
        !config.damping.enabled && !config.coriolis.enabled;
    admission.generated_metal_east_npy = admission.native_dry_tvd &&
        is_public_qualified_generated_metal_east_npy_profile(config);
    admission.generated_metal_mixed_npy = admission.native_dry_tvd &&
        tmv::runtime::is_public_qualified_flat_mixed_metal_cadence1_profile(config);
    admission.generated_metal = admission.native_dry_tvd &&
        (admission.generated_metal_east_npy || admission.generated_metal_mixed_npy ||
         is_public_qualified_generated_metal_profile(config));
    admission.native_implicit_diffusion =
        tmv::init::config::is_public_qualified_native_implicit_diffusion_profile(config);
    admission.tke_stability =
        tmv::init::config::is_public_qualified_tke_stability_profile(
            config);
    admission.coupled_rk =
        tmv::init::config::is_public_qualified_coupled_rk_profile(
            config);
    admission.variable_diffusion =
        tmv::runtime::is_exact_qualified_variable_diffusion_profile(
            config);
    admission.agrid_core =
        tmv::runtime::is_exact_qualified_agrid_core_profile(
            config) ||
        tmv::runtime::is_public_ready_flat_agrid_cpu_family(
            config);
    admission.simple_grey_absorbing =
        tmv::init::config::
            is_public_qualified_simple_grey_absorbing_water_cloud_profile(
                config) || admission.mapped_terrain;
    return admission;
}


void validate_stationary_nesting_execution(
    const StationaryNestingLiveExecution* live)
{
    validate_stationary_nesting_effective_profile(live);
}

/**
 * @brief Loads the configuration from a YAML file.
 */
void load_config(const std::string&              config_path,
                 int&                            duration_s,
                 int&                            write_every_s,
                 std::string&                    outdir,
                 OutputConfig*                   output_config,
                 bool*                           live_shm,
                 std::vector<std::string>*       live_shm_fields,
                 int                             write_every_override,
                 int                             live_shm_override,
                 const std::vector<std::string>* live_shm_fields_override,
                 bool                            trajectories_enabled,
                 const std::string*              compute_backend_override,
                 const std::string*              sounding_file_override,
                 int                             duration_override)
{
    if (config_path.empty())
    {
        return;
    }
    reset_performance_and_diagnostics_baseline();
    (void)outdir;
    const bool quiet_override = (global_log_profile == LogProfile::quiet);

    // An explicitly-provided config that cannot be read is fatal: running
    // the built-in defaults instead would silently simulate something the
    // user never asked for (and the stale defaults crash downstream). The
    // caller catches and exits non-zero.
    auto with_lines = parse_yaml_with_lines(config_path);
    if (!with_lines.has_value())
    {
        throw std::runtime_error("could not open config file: " + config_path);
    }
    std::unordered_map<std::string, std::string> config;
    config.reserve(with_lines->size());
    for (const auto& [key, field] : *with_lines)
    {
        config.emplace(key, field.value);
    }

    const auto typed_config = ingest_v1_config(config, config_path, *with_lines, compute_backend_override, sounding_file_override);

    // CLI backend selection changes the executed equation path and therefore
    // must be part of memory admission and the single pre-allocation science
    // profile decision. Applying it after this function would validate the
    // YAML backend and execute a different one.
    if (compute_backend_override != nullptr)
    {
        config["numerics.compute.backend"] = *compute_backend_override;
    }
    if (sounding_file_override != nullptr)
    {
        config["environment.sounding.type"]                  = "file";
        config["environment.sounding.file_path"]             = *sounding_file_override;
        config["environment.sounding.scheme_id"]             = "auto";
        config["environment.sounding.require_winds"]         = "true";
        config["environment.sounding.use_fallback_profiles"] = "false";
        config["environment.sounding.enabled"]               = "false";
    }

    tmv::init::config::ExecutionMode execution_mode;
    bool maturity_limits_acknowledged = false;
    if (typed_config)
    {
        auto controls = tmv::configuration::compile_controls(*typed_config, runtime_configuration);
        execution_mode = runtime_configuration.execution_mode;
        maturity_limits_acknowledged = typed_config->extensions.execution.acknowledge_maturity_limits;
        duration_s = controls.duration_s;
        write_every_s = controls.write_every_s;
        if (output_config) *output_config = std::move(controls.output);
        if (live_shm) *live_shm = controls.live_shm;
        if (live_shm_fields) *live_shm_fields = std::move(controls.live_shm_fields);
        tmv::diagnostics::set_deterministic_reductions(typed_config->extensions.diagnostics.deterministic_reductions);
    }
    else
    {
        execution_mode = tmv::init::config::ExecutionMode::Exploratory;
        if (const auto it = config.find("execution.mode");
            it != config.end() && !tmv::init::config::parse_execution_mode(it->second, execution_mode))
        {
            throw std::runtime_error("Invalid execution.mode '" + it->second + "'; expected exploratory or qualified");
        }
        const bool maturity_ack_present =
            config.count("execution.acknowledge_maturity_limits") != 0;
        maturity_limits_acknowledged = maturity_ack_present &&
            parse_bool_value(config.at("execution.acknowledge_maturity_limits"));
        if (execution_mode == tmv::init::config::ExecutionMode::Qualified &&
            maturity_ack_present && !maturity_limits_acknowledged)
        {
            throw std::runtime_error(
                "Qualified execution rejects an explicit false execution.acknowledge_maturity_limits; remove the obsolete key");
        }
        global_execution_mode = execution_mode;

        global_checkpoint_every_s = 0;
        global_checkpoint_keep_last = 0;
        global_checkpoint_minimum_free_space_mib = 0;
        global_checkpoint_directory.clear();
        global_checkpoint_resume_from.clear();
        if (const auto it = config.find("checkpoint.every_s"); it != config.end())
        {
            if (!try_parse_non_negative_int_value(it->second, global_checkpoint_every_s))
            {
                throw std::runtime_error("Invalid checkpoint.every_s '" + it->second +
                                         "'; expected a non-negative integer");
            }
        }
        if (const auto it = config.find("checkpoint.keep_last"); it != config.end())
        {
            if (!try_parse_non_negative_int_value(it->second, global_checkpoint_keep_last))
                throw std::runtime_error("Invalid checkpoint.keep_last '" + it->second +
                                         "'; expected a non-negative integer");
        }
        if (const auto it = config.find("checkpoint.minimum_free_space_mib"); it != config.end())
        {
            if (!try_parse_non_negative_int_value(it->second, global_checkpoint_minimum_free_space_mib))
                throw std::runtime_error("Invalid checkpoint.minimum_free_space_mib '" + it->second +
                                         "'; expected a non-negative integer");
        }
        if (const auto it = config.find("checkpoint.directory"); it != config.end())
        {
            global_checkpoint_directory = it->second;
        }
        if (const auto it = config.find("checkpoint.resume_from"); it != config.end())
        {
            global_checkpoint_resume_from = it->second;
        }

        load_grid_geometry(config);

        load_logging(config);

        if (!quiet_override && log_normal_enabled())
        {
            tmv::log_info("Loaded config with ", config.size(), " keys");
        }

        load_performance(config);

        load_diagnostics_cadence(config);

        load_validation(config);
        FieldPool::instance().set_overwrite_poisoning(global_validation_policy.mode == tmv::GuardMode::Strict);

        if (config.count("duration_s"))
        {
            int parsed = 0;
            if (try_parse_non_negative_int_value(config["duration_s"], parsed))
            {
                duration_s = parsed;
            }
            else
            {
                throw std::runtime_error("duration_s must be a non-negative whole number of seconds "
                                         "within the runtime integer range; got '" +
                                         config["duration_s"] + "'");
            }
        }
        if (duration_override >= 0)
        {
            duration_s = duration_override;
        }

        if (config.count("output.write_every_s"))
        {
            int parsed = 0;
            if (try_parse_non_negative_int_value(config["output.write_every_s"], parsed))
            {
                write_every_s = parsed;
            }
            else
            {
                throw std::runtime_error("output.write_every_s must be a non-negative whole number of "
                                         "seconds within the runtime integer range; got '" +
                                         config["output.write_every_s"] + "'");
            }
        }

        if (live_shm && config.count("output.live_shm"))
        {
            *live_shm = parse_bool_value(config["output.live_shm"]);
        }
        if (live_shm_fields && config.count("output.live_shm_fields"))
        {
            *live_shm_fields =
                parse_string_list(config["output.live_shm_fields"]);
        }

        if (output_config)
        {
            *output_config = parse_output_config(config);
        }

    }
    if (duration_override >= 0) duration_s = duration_override;
    FieldPool::instance().set_overwrite_poisoning(global_validation_policy.mode == tmv::GuardMode::Strict);

    // Admission must describe the effective run, including CLI overrides.
    // Applying these after allocation can undercount an output/live workload
    // that the YAML itself disables.
    if (write_every_override >= 0)
    {
        write_every_s = write_every_override;
    }
    if (live_shm != nullptr && live_shm_override >= 0)
    {
        *live_shm = live_shm_override != 0;
    }
    if (live_shm_fields != nullptr && live_shm_fields_override != nullptr)
    {
        *live_shm_fields = *live_shm_fields_override;
    }

    if (typed_config)
        tmv::configuration::compile_initialization(*typed_config, runtime_configuration);
    else
    {
        load_environment(config);
        load_trigger(config);
        load_nested(config);
    }

    if (typed_config)
    {
        NR = typed_config->grid.nx;
        NTH = typed_config->grid.ny;
        NZ = typed_config->grid.nz;
        dr = typed_config->grid.dx;
        dtheta = typed_config->grid.dy;
        dz = typed_config->grid.dz;
        dt = typed_config->time.dt_large;
    }
    else
    {
        if (config.count("grid.nx"))
        {
            int new_NR = 0;
            if (!try_parse_positive_int_value(config["grid.nx"], new_NR))
            {
                warn_invalid_config_value("grid.nx", config["grid.nx"], "a positive integer");
            }
            else
            {

                if (new_NR != NR)
                {
                    NR = new_NR;
                }
            }
        }

        if (config.count("grid.ny"))
        {
            int new_NTH = 0;
            if (!try_parse_positive_int_value(config["grid.ny"], new_NTH))
            {
                warn_invalid_config_value("grid.ny", config["grid.ny"], "a positive integer");
            }
            else
            {

                if (new_NTH != NTH)
                {
                    NTH = new_NTH;
                }
            }
        }

        if (config.count("grid.nz"))
        {
            int new_NZ = 0;
            if (!try_parse_positive_int_value(config["grid.nz"], new_NZ))
            {
                warn_invalid_config_value("grid.nz", config["grid.nz"], "a positive integer");
            }
            else
            {

                if (new_NZ != NZ)
                {
                    NZ = new_NZ;
                }
            }
        }

        if (config.count("grid.dx"))
        {
            double parsed = 0.0;
            if (try_parse_double_value(config["grid.dx"], parsed) && parsed > 0.0)
            {
                dr = parsed;
            }
            else
            {
                warn_invalid_config_value("grid.dx", config["grid.dx"], "a positive finite number");
            }
        }

        // Resolve dy before any allocation/geometry initialization. A missing or
        // invalid dy means square horizontal cells for this load (and cannot leak
        // a rectangular value from a previously loaded configuration).
        dtheta = dr;
        if (config.count("grid.dy"))
        {
            double parsed = 0.0;
            if (try_parse_double_value(config["grid.dy"], parsed) && parsed > 0.0)
            {
                dtheta = parsed;
            }
            else
            {
                warn_invalid_config_value("grid.dy", config["grid.dy"], "a positive finite number");
            }
        }

        if (config.count("grid.dz"))
        {
            double parsed = 0.0;
            if (try_parse_double_value(config["grid.dz"], parsed) && parsed > 0.0)
            {
                dz = parsed;
            }
            else
            {
                warn_invalid_config_value("grid.dz", config["grid.dz"], "a positive finite number");
            }
        }

        if (config.count("grid.dt"))
        {
            double parsed = 0.0;
            if (try_parse_double_value(config["grid.dt"], parsed) && parsed > 0.0)
            {
                dt = parsed;
            }
            else
            {
                warn_invalid_config_value("grid.dt", config["grid.dt"], "a positive finite number");
            }
        }
    }
    global_configured_dt_large_s = dt;

    // Filled after every loader and science preflight has resolved the active
    // backend and staggering, but still before any 3-D field allocation.
    HardwareInfo            grid_hardware;
    RuntimeMemoryProfile    memory_profile;
    GridMemoryPreflight     memory_preflight;
    ComputeMemoryPreflight  compute_memory_preflight;
    ComputeMemoryProperties compute_properties;
    const std::string memory_admission = typed_config
        ? tmv::init::config::memory_admission_policy_name(typed_config->extensions.compute.memory_admission)
        : memory_profile_value(config, "hardware.memory_admission", "warn");

    if (!typed_config)
    {
        load_hodograph(config);
        load_domain_frame(config);
        load_storm_tracking(config);
    }

    if (typed_config)
        tmv::configuration::compile_physics(*typed_config).apply_to(global_simulation_storage);
    else
    {
        load_microphysics(config);

        load_radiation(config);

        load_boundary_layer(config);

        load_surface(config);

        load_turbulence(config);

        load_dynamics(config);

        load_numerics_compute(config);

        load_advection(config);
        load_diffusion(config);
        load_time_stepping(config);

        load_chaos(config);

        load_terrain(config);

    }
    if (!typed_config) load_dynamic_key_overrides(config);

    validate_domain_motion_execution();
    validate_stationary_nesting_execution();

    if (execution_mode == tmv::init::config::ExecutionMode::Qualified)
    {
        // Qualified recovery policy must precede every preflight operation.
        // In particular, a legacy file source must not pass the file-column
        // check via fallback and have fallback disabled only after allocation.
        global_compute_backend_config.allow_fallback             = false;
        global_runtime_sounding_config.use_fallback_profiles     = false;
        global_sounding_source_config.file.use_fallback_profiles = false;
        global_validation_policy.mode                            = tmv::GuardMode::Strict;
        global_validation_policy.fail_on                         = tmv::GuardFailOn::Both;
        FieldPool::instance().set_overwrite_poisoning(true);
    }

    // One descriptor-backed decision for runtime, CLI metadata, and Studio.
    // This executes after every loader has canonicalized its selection and
    // before initialization/allocation-dependent module construction.
    const std::string surface_scheme_id = typed_config
        ? tmv::init::config::surface_scheme_name(typed_config->surface.scheme)
        : config.count("surface.scheme") != 0 ? to_lower_copy(config.at("surface.scheme")) : "none";
    validate_runtime_science_profile(
        execution_mode, output_config, surface_scheme_id,
        typed_config ? &*typed_config : nullptr, write_every_s,
        live_shm != nullptr ? *live_shm
                            : typed_config && typed_config->output.live_shm,
        trajectories_enabled);

    // File selection is part of the scientific experiment. Parse, interpolate,
    // and self-check it before allocating the 3-D state so a missing,
    // malformed, incomplete, or non-finite column cannot fail only after
    // memory admission has already committed the run.
    const bool sounding_already_built = preflight_storm_lifetime_before_allocation(
        typed_config ? typed_config->domain_motion.lifetime_planning : parse_legacy_lifetime_planning(config), duration_s);
    if (!sounding_already_built)
        preflight_file_sounding_before_allocation();

    // A requested grid and spacing define the scientific problem. Resolve the
    // actual backend first (including declared CPU fallback), then admit CPU
    // state, output working sets, accelerator staging/shared residency, and
    // device-local capacity without changing the requested workload.
    grid_hardware = detect_hardware();
    const bool effective_trajectories =
        trajectories_enabled || (output_config != nullptr && output_config->trajectories);
    memory_profile = build_runtime_memory_profile(global_simulation_storage,
                                                  output_config,
                                                  write_every_s,
                                                  live_shm != nullptr && *live_shm,
                                                  live_shm_fields,
                                                  effective_trajectories);

    std::string compute_backend_error;
    if (!initialize_compute_backend_runtime(compute_backend_error))
    {
        throw std::runtime_error("compute backend pre-allocation initialization failed: " + compute_backend_error);
    }
    const ComputeBackend* initialized_backend = active_compute_backend();
    const auto* native_microphysics = initialized_backend != nullptr ? initialized_backend->microphysics_kernels() : nullptr;
    if (to_lower_copy(runtime_configuration.microphysics_scheme) == "lin" &&
        !global_compute_backend_runtime_state.config.validate_parity && native_microphysics != nullptr &&
        native_microphysics->supports_atomic_lin_fused())
    {
        memory_profile = build_runtime_memory_profile(global_simulation_storage, output_config, write_every_s,
            live_shm != nullptr && *live_shm, live_shm_fields, effective_trajectories, true);
    }
    if (runtime_configuration.admission.coupled_rk && active_compute_backend_kind() != ComputeBackendKind::Cpu)
    {
        const auto* dynamics   = initialized_backend ? initialized_backend->dynamics_kernels() : nullptr;
        const bool  collocated = global_stagger_type == StaggerType::Collocated;
        if (!initialized_backend || !initialized_backend->supports_rk_combination() ||
            !initialized_backend->supports_rk_tendency() || !dynamics ||
            !dynamics->supports(collocated ? DynamicsKernelOp::CartesianTendencies
                                           : DynamicsKernelOp::CartesianCgridSlowTendencies))
            throw std::runtime_error(
                "native RK requires direct tendencies, transport, boundaries and stage combination");
        const auto* turbulence = initialized_backend->turbulence_kernels();
        if (!turbulence || !turbulence->supports(TurbulenceKernelOp::SgsApply))
            throw std::runtime_error("native RK requires native physics state application");
        if (global_turbulence_config.scheme_id != "none" &&
            !turbulence->supports(collocated ? TurbulenceKernelOp::CollocatedSgs
                                             : (global_turbulence_config.scheme_id == "tke"
                                                    ? TurbulenceKernelOp::Tke
                                                    : TurbulenceKernelOp::Smagorinsky)))
            throw std::runtime_error("native RK requires the selected grid's SGS closure and transport");
        if (global_diffusion_config.scheme_id == "implicit" &&
            (!initialized_backend->diffusion_kernels() ||
             !initialized_backend->diffusion_kernels()->supports(DiffusionKernelOp::Implicit)))
            throw std::runtime_error("native RK requires native implicit diffusion");
    }
    if (runtime_configuration.admission.two_way_accelerator_reference && active_compute_backend_kind() != ComputeBackendKind::Cpu)
    {
        if (!initialized_backend ||
            !initialized_backend->supports_dynamics(DynamicsKernelOp::ConservativeMomentum) ||
            !initialized_backend->supports_dynamics(DynamicsKernelOp::ConservativeReduction) ||
            (global_chaos_config.native_generation_required && !initialized_backend->supports_earth_control_evolution()))
            throw std::runtime_error("native two-way nesting requires conservative momentum and feedback kernels");
        tmv::log_info("[NESTED_NATIVE] momentum_tendencies=device, scalar_transport=device, restriction_reflux=device, "
                      "positivity_control=host_fp64, acoustic_integration=host_fp64, conservation_reports=host_fp64");
    }
    if (runtime_configuration.admission.native_weno5 || runtime_configuration.admission.native_dry_tvd ||
        (runtime_configuration.admission.two_way_accelerator_reference && active_compute_backend_kind() != ComputeBackendKind::Cpu))
    {
        const auto* kernels = initialized_backend != nullptr ? initialized_backend->advection_kernels() : nullptr;
        if (kernels == nullptr || !kernels->supports(AdvectionKernelOp::ConservativeCartesianCgridTransport) ||
            !kernels->supports_supplied_cartesian_cgrid_mass_flux())
            throw std::runtime_error("native conservative transport requires retained mass-flux ownership");
    }
    if (runtime_configuration.admission.native_implicit_diffusion)
    {
        const auto* kernels = initialized_backend != nullptr ? initialized_backend->diffusion_kernels() : nullptr;
        if (kernels == nullptr || !kernels->supports(DiffusionKernelOp::Implicit))
            throw std::runtime_error("native implicit diffusion requires both solver kernels");
    }

    if (runtime_configuration.admission.tke_stability &&
        active_compute_backend_kind() != ComputeBackendKind::Cpu)
    {
        const auto* kernels = initialized_backend != nullptr ? initialized_backend->turbulence_kernels() : nullptr;
        if (kernels == nullptr || !kernels->supports(TurbulenceKernelOp::Tke) ||
            !kernels->supports(TurbulenceKernelOp::SgsApply))
            throw std::runtime_error("native TKE requires complete closure, transport and state-application kernels");
    }
    if (runtime_configuration.admission.agrid_core && initialized_backend != nullptr &&
        active_compute_backend_kind() != ComputeBackendKind::Cpu)
    {
        const DynamicsComputeKernels* kernels =
            initialized_backend->dynamics_kernels();
        const DynamicsKernelOp required[] = {
            DynamicsKernelOp::CartesianTendencies,
            DynamicsKernelOp::AcousticContinuityMassFluxCartesianAgrid,
            DynamicsKernelOp::AcousticPressureEosCartesianAgrid,
            DynamicsKernelOp::AcousticMomentumCartesianAgrid,
            DynamicsKernelOp::AcousticNullModeConstraintCartesianAgrid,
            DynamicsKernelOp::AcousticBoundaryCartesianAgrid,
        };
        if (kernels == nullptr || std::any_of(
                std::begin(required), std::end(required),
                [&](DynamicsKernelOp op) { return !kernels->supports(op); }))
        {
            shutdown_compute_backend_runtime();
            throw std::runtime_error(
                "qualified native A-grid profile requires every complete A-grid device stage");
        }
    }
    if (runtime_configuration.admission.native_mapped_terrain)
    {
        const TerrainComputeKernels* kernels = initialized_backend != nullptr
            ? initialized_backend->terrain_kernels()
            : nullptr;
        const ComputeBackendKind mapped_backend = active_compute_backend_kind();
        if ((mapped_backend != ComputeBackendKind::Vulkan &&
             mapped_backend != ComputeBackendKind::Metal) ||
            compute_backend_fallback_active() || kernels == nullptr ||
            !kernels->supports_mapped_mass_continuity() ||
            !kernels->supports_mapped_fast_momentum() ||
            !kernels->supports_mapped_slow_tendencies() ||
            !kernels->supports_mapped_scalar_transport() ||
            !kernels->supports_mapped_state_update() ||
            !kernels->supports_mapped_flux_accumulation() ||
            !kernels->supports_mapped_final_state())
        {
            shutdown_compute_backend_runtime();
            throw std::runtime_error(
                "qualified native mapped terrain requires every selected-backend mapped-core device stage with fallback off");
        }
    }
    if (runtime_configuration.admission.earth_fixed_terrain_motion &&
        active_compute_backend_kind() != ComputeBackendKind::Cpu)
    {
        const auto* terrain = initialized_backend ? initialized_backend->terrain_kernels() : nullptr;
        if (!terrain || compute_backend_fallback_active() || !terrain->supports_earth_geometry() ||
            !terrain->supports_conservative_remap() || !terrain->supports_moving_momentum() ||
            !terrain->supports_moving_energy() || !terrain->supports_reference_atmosphere() ||
            !terrain->supports_column_remap() || !terrain->supports_held_sources() ||
            !terrain->supports_mapped_mass_continuity() || !terrain->supports_mapped_fast_momentum() ||
            !terrain->supports_mapped_slow_tendencies() || !terrain->supports_mapped_scalar_transport() ||
            !terrain->supports_mapped_state_update() || !terrain->supports_mapped_flux_accumulation() ||
            !terrain->supports_mapped_final_state() ||
            !initialized_backend->supports_boundary_layer(BoundaryLayerKernelOp::SlabColumn) ||
            !initialized_backend->supports_radiation(RadiationKernelOp::SimpleGrey) ||
            !initialized_backend->supports_microphysics(global_microphysics_scheme == "lin"
                                                            ? MicrophysicsKernelOp::LinPointwise
                                                            : MicrophysicsKernelOp::KesslerPointwise) ||
            !initialized_backend->supports_microphysics(global_microphysics_scheme == "lin"
                                                            ? MicrophysicsKernelOp::LinSedimentation
                                                            : MicrophysicsKernelOp::KesslerSedimentation) ||
            !initialized_backend->supports_dynamics(DynamicsKernelOp::SlowApply) ||
            !initialized_backend->supports_dynamics(DynamicsKernelOp::AcousticBoundaryCartesianCgrid) ||
            !initialized_backend->supports_earth_control_evolution())
        {
            shutdown_compute_backend_runtime();
            throw std::runtime_error(
                "native Earth-fixed execution requires every selected terrain and physics device owner");
        }
        tmv::log_info("[EARTH_NATIVE] spatial_operators=device, column_physics=device, stochastic_generation=device, "
                      "saturation=",
                      global_microphysics_scheme == "lin" ? "device" : "host_fp64",
                      ", thermodynamic_state_application=host_fp64, final_eos=host_fp64");
    }
    compute_properties =
        initialized_backend != nullptr ? initialized_backend->memory_properties() : ComputeMemoryProperties{};
    ComputeMemoryLevelDimensions accelerator_levels[2]   = {{NR, NTH, NZ}, {0, 0, 0}};
    std::size_t                  accelerator_level_count = 1;
    if (runtime_configuration.nesting.enabled)
    {
        const auto& nesting     = runtime_configuration.nesting;
        accelerator_levels[1]   = {nesting.child_cells_x + 2, nesting.child_cells_y + 2, NZ};
        accelerator_level_count = 2;
    }
    const int retained_kessler_fields =
        active_compute_backend_kind() == ComputeBackendKind::Metal && global_microphysics_scheme == "kessler" ? 13 : 0;
    compute_memory_preflight =
        assess_compute_memory_hierarchy(compute_properties,
                                        accelerator_levels,
                                        accelerator_level_count,
                                        kComputeMaximumRetainedFieldEquivalents + retained_kessler_fields);

    if (compute_memory_preflight.exceeds_storage_buffer_range)
    {
        shutdown_compute_backend_runtime();
        throw std::runtime_error("requested full field exceeds the active accelerator's maximum "
                                 "storage-buffer range before field allocation");
    }

    memory_profile.field_equivalents += compute_memory_preflight.field_equivalents;
    if (compute_memory_preflight.accelerator)
    {
        std::ostringstream accelerator_label;
        accelerator_label << ", accelerator_resident=" << compute_memory_preflight.field_equivalents
                          << " fields/level across "
                          << compute_memory_preflight.resident_level_count
                          << " level(s), accelerator_memory="
                          << (compute_memory_preflight.unified_host_device_memory ? "unified-counted-once"
                                                                                  : "staging+device-separated");
        memory_profile.label += accelerator_label.str();
    }

    if (runtime_configuration.nesting.enabled)
    {
        constexpr int nested_boundary_scratch_fields = 20;
        const int nested_rollback_fields = memory_profile.phases.resident_fields +
            (global_terrain_config.scheme_id == "none" ? 0 : 20);
        const auto& nesting = runtime_configuration.nesting;
        const long double root_cells = static_cast<long double>(NR) *
            static_cast<long double>(NTH) * static_cast<long double>(NZ);
        const long double child_cells =
            static_cast<long double>(nesting.child_cells_x + 2) *
            static_cast<long double>(nesting.child_cells_y + 2) *
            static_cast<long double>(NZ);
        // Fifteen root prognostics retain the previous parent endpoint for
        // temporal forcing. The child uses the same conservative solver
        // high-water profile on its smaller native grid, plus its retained
        // five-field forcing target and transaction candidate.
        // This intentionally overestimates run-global owners rather than
        // under-admitting memory.
        const long double nested_bytes = sizeof(float) *
            (15.0L * root_cells +
             static_cast<long double>(
                 memory_profile.field_equivalents +
                 nested_boundary_scratch_fields + nested_rollback_fields) *
                 child_cells);
        if (!std::isfinite(nested_bytes) || nested_bytes < 0.0L ||
            nested_bytes > static_cast<long double>(
                std::numeric_limits<std::uint64_t>::max()) ||
            memory_profile.fixed_overhead_bytes >
                std::numeric_limits<std::uint64_t>::max() -
                    static_cast<std::uint64_t>(nested_bytes))
        {
            shutdown_compute_backend_runtime();
            throw std::runtime_error(
                "stationary nesting memory estimate exceeds the representable host budget");
        }
        memory_profile.fixed_overhead_bytes +=
            static_cast<std::uint64_t>(std::ceil(nested_bytes));
        std::ostringstream nesting_label;
        nesting_label << ", stationary_nest=" << nesting.child_cells_x
                      << "x" << nesting.child_cells_y << "x" << NZ
                      << ", parent_history=15 fields"
                      << ", child_boundary_scratch="
                      << nested_boundary_scratch_fields << " fields"
                      << ", child_rollback=" << nested_rollback_fields << " fields";
        memory_profile.label += nesting_label.str();
    }

    if (global_checkpoint_every_s > 0)
    {
        const long double root_cells_ld = static_cast<long double>(NR) *
            static_cast<long double>(NTH) * static_cast<long double>(NZ);
        long double child_cells_ld = 0.0L;
        if (runtime_configuration.nesting.enabled)
        {
            const auto& nesting = runtime_configuration.nesting;
            child_cells_ld =
                static_cast<long double>(nesting.child_cells_x + 2) *
                static_cast<long double>(nesting.child_cells_y + 2) *
                static_cast<long double>(NZ);
        }
        const long double maximum_cells = static_cast<long double>(
            std::numeric_limits<std::uint64_t>::max());
        if (!std::isfinite(root_cells_ld) || root_cells_ld < 0.0L ||
            root_cells_ld > maximum_cells ||
            !std::isfinite(child_cells_ld) || child_cells_ld < 0.0L ||
            child_cells_ld > maximum_cells)
        {
            shutdown_compute_backend_runtime();
            throw std::runtime_error(
                "checkpoint memory estimate exceeds the representable host budget");
        }
        const int checkpoint_fields =
            checkpoint_durable_field_equivalents(global_simulation_storage);
        const std::uint64_t checkpoint_transient =
            estimate_checkpoint_encoding_transient_bytes(
                static_cast<std::uint64_t>(root_cells_ld),
                static_cast<std::uint64_t>(child_cells_ld),
                checkpoint_fields);
        if (checkpoint_transient == std::numeric_limits<std::uint64_t>::max() ||
            memory_profile.fixed_overhead_bytes >
                std::numeric_limits<std::uint64_t>::max() -
                    checkpoint_transient)
        {
            shutdown_compute_backend_runtime();
            throw std::runtime_error(
                "checkpoint memory estimate exceeds the representable host budget");
        }
        memory_profile.fixed_overhead_bytes += checkpoint_transient;
        std::ostringstream checkpoint_label;
        checkpoint_label << ", streamed_checkpoint_transient="
                         << checkpoint_transient << " bytes ("
                         << checkpoint_fields
                         << " durable fields/level, cadence="
                         << global_checkpoint_every_s << "s)";
        memory_profile.label += checkpoint_label.str();
    }

    check_grid_memory_safety(grid_hardware,
                             NR,
                             NTH,
                             NZ,
                             memory_profile.field_equivalents,
                             memory_profile.label,
                             memory_profile.fixed_overhead_bytes);
    memory_preflight = assess_grid_memory(
        grid_hardware, NR, NTH, NZ, memory_profile.field_equivalents, memory_profile.fixed_overhead_bytes);
    const bool reject_memory = memory_admission == "reject";
    if (memory_admission_should_reject(memory_preflight, reject_memory))
    {
        shutdown_compute_backend_runtime();
        throw std::runtime_error("phase-aware host/shared-memory estimate exceeds the 80% "
                                 "physical-RAM planning budget and "
                                 "hardware.memory_admission=reject");
    }
    if (compute_memory_admission_should_reject(compute_memory_preflight, reject_memory))
    {
        shutdown_compute_backend_runtime();
        throw std::runtime_error(compute_memory_preflight.device_budget_is_live
            ? "accelerator working-set estimate exceeds 80% of the currently available device-local memory budget; "
              "free GPU memory or reduce the requested working set (hardware.memory_admission=reject)"
            : "accelerator working-set estimate exceeds the 80% device-local "
              "heap planning budget and hardware.memory_admission=reject");
    }
    if (compute_memory_preflight.exceeds_device_budget)
    {
        tmv::log_warn("[HARDWARE] Accelerator working-set estimate ",
                      compute_memory_preflight.device_residency_bytes,
                      " bytes exceeds its 80% device-local ",
                      compute_memory_preflight.device_budget_is_live ? "available-memory budget " : "heap planning budget ",
                      compute_memory_preflight.device_budget_bytes,
                      " bytes; the requested grid is unchanged.");
    }

    {
        extern void resize_fields();
        resize_fields();
    }

    if (!quiet_override && log_normal_enabled())
    {
        tmv::log_info(std::string(60, '='));
        tmv::log_info("CONFIGURATION LOADED");
        tmv::log_info(std::string(60, '='));
        tmv::log_info("Config file: ", config_path);
        tmv::log_info("Duration: ", duration_s, "s");
        tmv::log_info("Write every: ", write_every_s, "s");
        tmv::log_info("Grid parameters:");
        tmv::log_info("  NR=", NR, ", NTH=", NTH, ", NZ=", NZ);
        tmv::log_info("  dx=", dr, "m, dy=", dtheta, "m, dz=", dz, "m, dt=", dt, "s");
        tmv::log_info("  Physical extent=",
                      static_cast<double>(NR) * dr / 1000.0,
                      "x",
                      static_cast<double>(NTH) * dtheta / 1000.0,
                      "x",
                      static_cast<double>(NZ) * dz / 1000.0,
                      " km (requested grid and spacing preserved)");
        {
            const uint64_t     grid_bytes = memory_preflight.estimated_bytes;
            const uint64_t     ram_bytes  = grid_hardware.total_ram_bytes;
            const double       grid_gb    = static_cast<double>(grid_bytes) / (uint64_t(1) << 30);
            std::ostringstream memory;
            memory << "  Memory planning estimate: ~" << std::fixed << std::setprecision(1) << grid_gb << " GB ("
                   << memory_profile.field_equivalents << " full-field equivalents)";
            if (ram_bytes > 0)
            {
                const double ram_gb = static_cast<double>(ram_bytes) / (uint64_t(1) << 30);
                const double pct    = 100.0 * static_cast<double>(grid_bytes) / ram_bytes;
                memory << " / " << ram_gb << " GB"
                       << " (" << pct << "%)";
            }
            tmv::log_info(memory.str());
            const uint64_t cells = static_cast<uint64_t>(NR) * static_cast<uint64_t>(NTH) * static_cast<uint64_t>(NZ);
            const uint64_t field_bytes = cells * sizeof(float);
            tmv::log_info("[MEMORY PLAN] cells=",
                          cells,
                          ", field_bytes=",
                          field_bytes,
                          ", predicted_fields=",
                          memory_profile.field_equivalents,
                          ", fixed_overhead_bytes=",
                          memory_profile.fixed_overhead_bytes,
                          ", predicted_bytes=",
                          grid_bytes);
            if (compute_memory_preflight.accelerator)
            {
                tmv::log_info("[MEMORY PLAN] accelerator_mode=",
                              compute_memory_preflight.unified_host_device_memory ? "unified" : "discrete",
                              ", accelerator_fields=",
                              compute_memory_preflight.field_equivalents,
                              ", accelerator_levels=",
                              compute_memory_preflight.resident_level_count,
                              ", host_shared_or_staging_bytes=",
                              compute_memory_preflight.host_residency_bytes,
                              ", device_residency_bytes=",
                              compute_memory_preflight.device_residency_bytes,
                              ", device_budget_bytes=",
                              compute_memory_preflight.device_budget_bytes,
                              ", device_budget_source=",
                              compute_properties.device_budget_is_live ? "live" : "heap_capacity",
                              ", device_available_budget_bytes=",
                              compute_properties.device_available_budget_bytes,
                              ", max_storage_buffer_bytes=",
                              compute_properties.max_storage_buffer_bytes);
            }
            tmv::log_info("  Memory profile: ", memory_profile.label);
        }
        tmv::log_info("Environment:");
        tmv::log_info("  Sounding source: ", tmv::init::sounding_source_type_name(global_sounding_source_config.type));
        switch (global_sounding_source_config.type)
        {
        case tmv::init::SoundingSourceConfig::Type::ParametricCAPE:
        {
            const auto& p = global_sounding_source_config.parametric;
            tmv::log_info("    CAPE target: ", p.cape_target_jkg, " J/kg");
            tmv::log_info("    Surface theta: ", p.surface_theta_k, " K");
            tmv::log_info("    Surface qv: ", p.surface_qv_kgkg, " kg/kg");
            tmv::log_info("    Tropopause: ", p.tropopause_z_m, " m");
            break;
        }
        case tmv::init::SoundingSourceConfig::Type::ParametricTargets:
        {
            const auto& t = global_sounding_source_config.targets;
            tmv::log_info("    Targets: CAPE=",
                          t.target_cape_jkg,
                          " J/kg, CIN=",
                          t.target_cin_jkg,
                          " J/kg, LFC=",
                          t.target_lfc_m,
                          " m, EL=",
                          t.target_el_m,
                          " m");
            tmv::log_info("    Surface theta: ", t.surface_theta_k, " K");
            if (!t.lcl_is_targeted())
            {
                tmv::log_info("    Surface qv override: ",
                              t.moisture_qv_kgkg_override,
                              " kg/kg (LCL is diagnostic, not targeted)");
            }
            else
            {
                tmv::log_info("    LCL target: ", t.target_lcl_m, " m");
            }
            break;
        }
        case tmv::init::SoundingSourceConfig::Type::File:
        {
            const auto& f = global_sounding_source_config.file;
            tmv::log_info("    File: ",
                          f.path,
                          " (scheme: ",
                          f.scheme_id,
                          ", require_winds: ",
                          f.require_winds ? "yes" : "no",
                          ", fallback: ",
                          f.use_fallback_profiles ? "on" : "off",
                          ")");
            break;
        }
        }
        tmv::log_info("  Wind profile: SFC(",
                      global_wind_profile.u_sfc,
                      ",",
                      global_wind_profile.v_sfc,
                      ") 1km(",
                      global_wind_profile.u_1km,
                      ",",
                      global_wind_profile.v_1km,
                      ") 6km(",
                      global_wind_profile.u_6km,
                      ",",
                      global_wind_profile.v_6km,
                      ")");
        switch (global_trigger_source_config.type)
        {
        case tmv::init::TriggerSourceConfig::Type::WarmBubble:
            tmv::log_info("  Trigger: warm_bubble center_x=",
                          global_bubble_center_x_m / 1000.0,
                          " km, center_y=",
                          global_bubble_center_y_m / 1000.0,
                          " km, center_z=",
                          global_bubble_center_z_m / 1000.0,
                          " km, radius=",
                          global_bubble_radius_m / 1000.0,
                          " km, dtheta=",
                          global_bubble_dtheta_k,
                          " K, cloud_water=",
                          global_bubble_cloud_water_kgkg,
                          " kg/kg");
            break;
        case tmv::init::TriggerSourceConfig::Type::None:
            tmv::log_info("  Trigger: none (no initial perturbation)");
            break;
        case tmv::init::TriggerSourceConfig::Type::VortexSeed:
        {
            const auto& vortex = global_trigger_source_config.vortex_seed;
            tmv::log_info("  Trigger: vortex_seed center_x=",
                          vortex.center_x_m / 1000.0,
                          " km, center_y=",
                          vortex.center_y_m / 1000.0,
                          " km, center_z=",
                          vortex.center_z_m / 1000.0,
                          " km, r_max=",
                          vortex.r_max_m / 1000.0,
                          " km, v_max=",
                          vortex.v_max_ms,
                          " m/s, z_top=",
                          vortex.z_top_m / 1000.0,
                          " km");
            break;
        }
        }
        tmv::log_info("Physics schemes:");
        const std::string effective_dynamics_scheme = global_dynamics_scheme_name.empty()
                                                          ? "klemp_wilhelmson_split_cartesian_cgrid (default)"
                                                          : global_dynamics_scheme_name;
        tmv::log_info("  Dynamics: ", effective_dynamics_scheme);
        tmv::log_info("  Microphysics: ", global_microphysics_scheme);
        if (global_radiation_config.scheme_id == "none")
        {
            tmv::log_info("  Radiation: none (inactive; no radiative forcing)");
        }
        else
        {
            tmv::log_info("  Radiation: ",
                          global_radiation_config.scheme_id,
                          " (LW: ",
                          global_radiation_config.do_lw ? "on" : "off",
                          ", SW: ",
                          global_radiation_config.do_sw ? "on" : "off",
                          ", dt: ",
                          global_radiation_config.dt_radiation,
                          "s, solar: ",
                          global_radiation_config.solar_forcing,
                          ", latitude: ",
                          global_radiation_config.latitude_deg,
                          " deg, declination: ",
                          global_radiation_config.declination_deg,
                          " deg, local_start: ",
                          global_radiation_config.local_solar_time_at_start_h,
                          " h)");
            if (global_radiation_config.scheme_id == "rrtmg")
            {
                tmv::log_info("    RRTMG dry-air VMR: O3=",
                              global_radiation_config.rrtmg_ozone_vmr,
                              ", CO2=",
                              global_radiation_config.rrtmg_co2_vmr,
                              ", CH4=",
                              global_radiation_config.rrtmg_ch4_vmr,
                              ", N2O=",
                              global_radiation_config.rrtmg_n2o_vmr,
                              ", CO=",
                              global_radiation_config.rrtmg_co_vmr,
                              ", O2=",
                              global_radiation_config.rrtmg_o2_vmr,
                              ", N2=",
                              global_radiation_config.rrtmg_n2_vmr,
                              ", CFC11=",
                              global_radiation_config.rrtmg_cfc11_vmr,
                              ", CFC12=",
                              global_radiation_config.rrtmg_cfc12_vmr,
                              ", CFC22=",
                              global_radiation_config.rrtmg_cfc22_vmr,
                              ", CCl4=",
                              global_radiation_config.rrtmg_ccl4_vmr);
                tmv::log_info("    RRTMG cloud effective radii: liquid=",
                              global_radiation_config.rrtmg_liquid_effective_radius_um,
                              " um, ice=",
                              global_radiation_config.rrtmg_ice_effective_radius_um,
                              " um");
            }
        }
        tmv::log_info("  Boundary Layer: ",
                      global_boundary_layer_config.scheme_id,
                      " (surface fluxes: ",
                      global_boundary_layer_config.apply_surface_fluxes ? "on" : "off",
                      ", layer: ",
                      global_boundary_layer_config.surface_layer_id,
                      ", dt: ",
                      global_boundary_layer_config.dt_pbl,
                      "s",
                      ", pbl_top: ",
                      global_boundary_layer_config.pbl_top_method,
                      ", min_ustar: ",
                      global_boundary_layer_config.min_ustar,
                      " m/s)");
        tmv::log_info("  Turbulence: ",
                      global_turbulence_config.scheme_id,
                      " (Cs: ",
                      global_turbulence_config.Cs,
                      ", max_substep: ",
                      global_turbulence_config.dt_sgs,
                      "s)");
        tmv::log_info("  Numerics-Advection: ",
                      global_advection_config.scheme_id,
                      global_time_stepping_config.scheme_id == "rk3" ||
                              global_time_stepping_config.scheme_id == "rk4"
                          ? " (bounded_scalars=tvd_mc, theta=weno5_when_selected)"
                          : "");
        tmv::log_info("  Numerics-Diffusion: ",
                      global_diffusion_config.scheme_id,
                      " (apply_to: ",
                      global_diffusion_config.apply_to,
                      ", K_h: ",
                      global_diffusion_config.K_h,
                      ", K_v: ",
                      global_diffusion_config.K_v,
                      ")");
        tmv::log_info("  Numerics-TimeStepping: ",
                      global_time_stepping_config.scheme_id,
                      " (adaptive_dt: ",
                      global_time_stepping_config.adaptive_dt ? "on" : "off",
                      ", dt_min: ",
                      global_time_stepping_config.dt_min,
                      "s, dt_max: ",
                      global_time_stepping_config.dt_max,
                      "s, cfl_safety: ",
                      global_time_stepping_config.cfl_safety,
                      ")");
        tmv::log_info("  Numerics-Compute: ",
                      global_compute_backend_config.backend,
                      " (device_index: ",
                      global_compute_backend_config.device_index,
                      ", fallback: ",
                      global_compute_backend_config.allow_fallback ? "on" : "off",
                      ", parity: ",
                      global_compute_backend_config.validate_parity ? "on" : "off",
                      ")");
        tmv::log_info("  Numerics-KernelTemplate: vertical_flux=", active_vertical_flux_template_id());
        tmv::log_info("  Execution mode: ", tmv::init::config::execution_mode_name(execution_mode));
        tmv::log_info("  Scheme maturity limits acknowledged: ",
                      maturity_limits_acknowledged ? "yes (legacy explicit key)" :
                      execution_mode == tmv::init::config::ExecutionMode::Qualified
                          ? "not required for exact Ready selection"
                          : "no (exploratory run)");
        tmv::log_info("  Chaos: ", global_chaos_config.scheme_id);
        tmv::log_info("  Terrain: ",
                      global_terrain_config.scheme_id,
                      " (coord: ",
                      global_terrain_config.coord_id,
                      ", ztop: ",
                      global_terrain_config.ztop,
                      "m)");
        std::ostringstream validation;
        validation << "  Validation: mode=" << tmv::to_string(global_validation_policy.mode)
                   << ", fail_on=" << tmv::to_string(global_validation_policy.fail_on)
                   << ", scope=" << tmv::to_string(global_validation_policy.strict_scope);
        if (!global_validation_report_path.empty())
        {
            validation << ", report_path=" << global_validation_report_path;
        }
        tmv::log_info(validation.str());
        tmv::log_info("  Logging profile: ", log_profile_name(global_log_profile));
        std::ostringstream perf;
        perf << "  Perf timing: " << (global_perf_timing_enabled ? "on" : "off");
        if (global_perf_report_every_steps > 0)
        {
            perf << " (report every " << global_perf_report_every_steps << " steps)";
        }
        else if (global_perf_timing_enabled)
        {
            perf << " (final summary only)";
        }
        if (global_perf_timing_enabled && global_perf_warmup_steps > 0)
        {
            perf << ", warmup=" << global_perf_warmup_steps << " steps";
        }
        tmv::log_info(perf.str());
        if (global_scientific_diagnostics_every_steps < 0)
        {
            tmv::log_info("  Scientific diagnostics: compatibility cadence");
        }
        else if (global_scientific_diagnostics_every_steps == 0)
        {
            tmv::log_info("  Scientific diagnostics: off");
        }
        else
        {
            tmv::log_info("  Scientific diagnostics: every ", global_scientific_diagnostics_every_steps, " steps");
        }
        if (global_storm_tracking_config.enabled)
        {
            tmv::log_info("  Storm tracking: diagnostic evidence every ",
                          global_storm_tracking_config.interval_s,
                          " s (inner margin ",
                          global_storm_tracking_config.inner_margin_m,
                          " m, continuity ",
                          global_storm_tracking_config.minimum_continuity_samples,
                          " samples, lost after ",
                          global_storm_tracking_config.lost_after_samples,
                          ", speed/acceleration gates ",
                          global_storm_tracking_config.maximum_motion_speed_ms,
                          " m/s and ",
                          global_storm_tracking_config.maximum_motion_acceleration_ms2,
                          " m/s^2, UH ",
                          global_storm_tracking_config.updraft_helicity_bottom_m_agl,
                          "-",
                          global_storm_tracking_config.updraft_helicity_top_m_agl,
                          " m AGL, movement mode ",
                          global_storm_tracking_config.recycle_domain
                              ? (global_terrain_config.scheme_id != "none"
                                     && global_terrain_config.horizontal_placement_id
                                         == "earth_fixed"
                                     ? "live_earth_fixed_terrain_cgrid_recycling"
                                     : "live_flat_cgrid_recycling")
                              : (global_storm_tracking_config.maximum_shift_cells > 0 ? "diagnostic_request_only"
                                                                                      : "diagnostic_only"),
                          ")");
        }
        else
        {
            tmv::log_info("  Storm tracking: off");
        }
        tmv::log_info("  Diagnostic reductions: ",
                      tmv::diagnostics::deterministic_reductions_enabled() ? "deterministic fixed-slab"
                                                                           : "native OpenMP");
        tmv::log_info(std::string(60, '='));
    }
}
