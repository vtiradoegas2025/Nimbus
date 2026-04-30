/**
 * @file runtime_config.cpp
 * @brief Core runtime implementation for the tornado model.
 *
 * Provides simulation orchestration and subsystem integration
 * for dynamics, numerics, physics, and runtime execution paths.
 * This file belongs to the primary src/core execution layer.
 */

#include "core/runtime_config.hpp"

#include "core/output/output_config.hpp"

#include <algorithm>
#include <cmath>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "numerics/advection/advection_base.hpp"
#include "boundary_layer/boundary_layer_base.hpp"
#include "chaos/chaos_base.hpp"
#include "compute/compute_backend.hpp"
#include "compute/compute_kernel_template.hpp"
#include "numerics/diffusion/diffusion_base.hpp"
#include "radiation/radiation_base.hpp"
#include "core/simulation.hpp"
#include "core/hardware_info.hpp"
#include "util/log.hpp"
#include "util/string_utils.hpp"
#include "terrain/terrain_base.hpp"
#include "numerics/time_stepping/time_stepping_base.hpp"
#include "turbulence/turbulence_base.hpp"

/**
 * @brief Returns a lowercase copy of the input string.
 */
std::string to_lower_copy(std::string value)
{
    return tmv::strutil::lower_copy(value);
}

/**
 * @brief Removes matching single or double quotes around a string value.
 */
std::string strip_wrapping_quotes(std::string value)
{
    if (value.size() >= 2)
    {
        const char first = value.front();
        const char last = value.back();
        if ((first == '"' && last == '"') || (first == '\'' && last == '\''))
        {
            return value.substr(1, value.size() - 2);
        }
    }
    return value;
}

/**
 * @brief Parses boolean-like configuration values.
 */
bool parse_bool_value(const std::string& value)
{
    return tmv::strutil::parse_bool(value);
}

/**
 * @brief Parses an integer value.
 */
bool try_parse_int_value(const std::string& value, int& out)
{
    try
    {
        size_t consumed = 0;
        const long long parsed = std::stoll(value, &consumed);
        if (consumed != value.size() ||
            parsed < static_cast<long long>(std::numeric_limits<int>::min()) ||
            parsed > static_cast<long long>(std::numeric_limits<int>::max()))
        {
            return false;
        }
        out = static_cast<int>(parsed);
        return true;
    }
    catch (const std::exception&)
    {
        return false;
    }
}

/**
 * @brief Parses a non-negative integer value.
 */
bool try_parse_non_negative_int_value(const std::string& value, int& out)
{
    int parsed = 0;
    if (!try_parse_int_value(value, parsed))
    {
        return false;
    }
    if (parsed < 0)
    {
        return false;
    }
    out = parsed;
    return true;
}

/**
 * @brief Parses a strictly positive integer value.
 */
bool try_parse_positive_int_value(const std::string& value, int& out)
{
    int parsed = 0;
    if (!try_parse_int_value(value, parsed))
    {
        return false;
    }
    if (parsed <= 0)
    {
        return false;
    }
    out = parsed;
    return true;
}

/**
 * @brief Parses an unsigned 64-bit integer value.
 */
bool try_parse_uint64_value(const std::string& value, std::uint64_t& out)
{
    try
    {
        size_t consumed = 0;
        const unsigned long long parsed = std::stoull(value, &consumed);
        if (consumed != value.size())
        {
            return false;
        }
        out = static_cast<std::uint64_t>(parsed);
        return true;
    }
    catch (const std::exception&)
    {
        return false;
    }
}

/**
 * @brief Parses a finite floating-point value.
 */
bool try_parse_double_value(const std::string& value, double& out)
{
    try
    {
        size_t consumed = 0;
        const double parsed = std::stod(value, &consumed);
        if (consumed != value.size() || !std::isfinite(parsed))
        {
            return false;
        }
        out = parsed;
        return true;
    }
    catch (const std::exception&)
    {
        return false;
    }
}

/**
 * @brief Emits a standardized warning for invalid configuration values.
 */
void warn_invalid_config_value(const std::string& key,
                               const std::string& value,
                               const char* expected)
{
    tmv::log_warn("Invalid ", key, " '", value, "'; expected ", expected,
                  ". Keeping previous/default value.");
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
    if (try_parse_double_value(value, parsed) &&
        (parsed == 0.0 || parsed == 1.0 || parsed == 2.0))
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
        if (valid) *valid = true;
        return LogProfile::quiet;
    }
    if (normalized == "normal")
    {
        if (valid) *valid = true;
        return LogProfile::normal;
    }
    if (normalized == "debug")
    {
        if (valid) *valid = true;
        return LogProfile::debug;
    }

    if (valid) *valid = false;
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
    if (normalized == "monin_obukhov" || normalized == "monin-obukhov" ||
        normalized == "moninobukhov" ||  normalized == "most" || normalized == "mo")
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
 * @brief Normalizes turbulence scheme aliases to canonical identifiers.
 */
std::string normalize_turbulence_scheme_name(std::string value)
{
    value = to_lower_copy(value);
    if (value == "smag" ||
        value == "smagorinsky_lilly" ||
        value == "smagorinsky-lilly")
    {
        return "smagorinsky";
    }
    if (value == "1.5" ||
        value == "1.5order" ||
        value == "1.5-order")
    {
        return "tke";
    }
    return value;
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
    if (normalized == "horizontal_only" || normalized == "horizontal-only" || normalized == "horizontal")
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
    if (normalized == "tke" || normalized == "tke_based" || normalized == "tke-based")
    {
        stability_out = "tke_based";
        return true;
    }
    return false;
}

/**
 * @brief Normalizes terrain scheme aliases to canonical identifiers.
 */
std::string normalize_terrain_scheme_name(std::string value)
{
    value = to_lower_copy(value);

    if (value == "flat"){ return "none";}
    if (value == "schaer"){ return "schar";}

    return value;
}

/**
 * @brief Parses terrain coordinate-system aliases.
 */
bool parse_terrain_coord_id(const std::string& value, std::string& coord_out)
{
    const std::string normalized = to_lower_copy(value);
    if (normalized == "btf" || normalized == "terrain_following" || normalized == "terrain-following" || normalized == "sigma")
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
std::unordered_map<std::string, std::string> parse_yaml_simple(const std::string& filename)
{
    std::unordered_map<std::string, std::string> config;
    std::ifstream file(filename);
    if (!file.is_open())
    {
        tmv::log_error("Could not open config file: ", filename);
        return config;
    }

    std::string line;
    std::vector<std::string> section_stack;

    while (std::getline(file, line))
    {
        size_t comment_pos = line.find('#');
        if (comment_pos != std::string::npos)
        {
            line = line.substr(0, comment_pos);
        }

        size_t indent = 0;
        while (indent < line.size() && line[indent] == ' ') indent++;

        size_t indent_level = indent / 2;

        std::string trimmed_line = line;
        trimmed_line.erase(trimmed_line.begin(), std::find_if(trimmed_line.begin(), trimmed_line.end(), [](unsigned char ch) 
        {
            return !std::isspace(ch);
        }));
        trimmed_line.erase(std::find_if(trimmed_line.rbegin(), trimmed_line.rend(), [](unsigned char ch) {
            return !std::isspace(ch);
        }).base(), trimmed_line.end());

        if (trimmed_line.empty()) continue;

        line = trimmed_line;

        if (line.back() == ':')
        {
            std::string section_name = line.substr(0, line.size() - 1);

            while (section_stack.size() > indent_level) 
            {
                section_stack.pop_back();
            }

            if (section_stack.size() == indent_level) 
            {
                section_stack.push_back(section_name);
            } 
            else 
            {
                section_stack[indent_level] = section_name;
            }

            continue;
        }

        size_t colon_pos = line.find(':');

        if (colon_pos != std::string::npos)
        {
            while (section_stack.size() > indent_level)
            {
                section_stack.pop_back();
            }

            std::string key = line.substr(0, colon_pos);
            std::string value = line.substr(colon_pos + 1);

            key.erase(key.begin(), std::find_if(key.begin(), key.end(), [](unsigned char ch) 
            {
                return !std::isspace(ch);
            }));

            key.erase(std::find_if(key.rbegin(), key.rend(), [](unsigned char ch) 
            {
                return !std::isspace(ch);
            }).base(), key.end());

            value.erase(value.begin(), std::find_if(value.begin(), value.end(), [](unsigned char ch) 
            {
                return !std::isspace(ch);
            }));

            value.erase(std::find_if(value.rbegin(), value.rend(), [](unsigned char ch) 
            {
                return !std::isspace(ch);
            }).base(), value.end());
            value = strip_wrapping_quotes(value);

            std::string full_key;
            for (const auto& section : section_stack) 
            {
                if (!full_key.empty()) full_key += ".";
                full_key += section;
            }
            if (!full_key.empty()) full_key += ".";
            full_key += key;
            config[full_key] = value;
        }
    }

    return config;
}

/**
 * @brief Parses a comma-separated (or YAML-like bracketed) string list.
 */
std::vector<std::string> parse_string_list(const std::string& value)
{
    std::string cleaned = value;
    cleaned.erase(std::remove(cleaned.begin(), cleaned.end(), '['), cleaned.end());
    cleaned.erase(std::remove(cleaned.begin(), cleaned.end(), ']'), cleaned.end());
    cleaned.erase(std::remove(cleaned.begin(), cleaned.end(), '"'), cleaned.end());
    cleaned.erase(std::remove(cleaned.begin(), cleaned.end(), '\''), cleaned.end());

    std::vector<std::string> out;
    std::stringstream ss(cleaned);
    std::string item;
    while (std::getline(ss, item, ','))
    {
        item.erase(item.begin(), std::find_if(item.begin(), item.end(), [](unsigned char ch) {
            return !std::isspace(ch);
        }));
        item.erase(std::find_if(item.rbegin(), item.rend(), [](unsigned char ch) {
            return !std::isspace(ch);
        }).base(), item.end());
        if (!item.empty())
        {
            out.push_back(item);
        }
    }
    return out;
}

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
    const std::size_t dot = tail.rfind('.');
    if (dot == std::string::npos || dot == 0 || dot + 1 >= tail.size())
    {
        return false;
    }

    field_id_out = tail.substr(0, dot);
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

WindProfile global_wind_profile = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

double global_cape_target = 2500.0;

double global_sfc_theta_k = 300.0;
double global_sfc_qv_kgkg = 0.014;
double global_tropopause_z_m = 12000.0;

double global_bubble_center_x_m = 50000.0;
// Default y = 0 keeps the Cartesian path behaving like the cylindrical path
// when the user has not yet added an explicit `trigger.bubble.center_y_km`
// to their config: the bubble sits on the line y = 0 in Cartesian, which
// is geometrically equivalent to "on the cylindrical radial axis" for
// configs that pre-date the Y-axis option.
double global_bubble_center_y_m = 0.0;
double global_bubble_center_z_m = 1500.0;
double global_bubble_radius_m = 10000.0;
double global_bubble_dtheta_k = 2.0;

bool global_sounding_enabled = false;
bool global_sounding_allow_placeholder_profiles = false;
SoundingConfig global_runtime_sounding_config{};

std::string global_microphysics_scheme = "kessler";

std::string global_dynamics_scheme_name = "";

CoordinateSystem global_coordinate_system = CoordinateSystem::Cylindrical;

StaggerType global_stagger_type = StaggerType::Collocated;

LogProfile global_log_profile = LogProfile::normal;
bool global_perf_timing_enabled = false;
int global_perf_report_every_steps = 0;

tmv::ValidationPolicy global_validation_policy{};
std::string global_validation_report_path;

/**
 * @brief Loads the configuration from a YAML file.
 */
void load_config(const std::string& config_path, int& duration_s, int& write_every_s, std::string& outdir, OutputConfig* output_config, bool* live_shm, std::vector<std::string>* live_shm_fields)
{
    if (config_path.empty()) 
    {
        return;
    }
    (void)outdir;
    const bool quiet_override = (global_log_profile == LogProfile::quiet);

    auto config = parse_yaml_simple(config_path);

    if (config.count("coordinate_system"))
    {
        const std::string& raw = config["coordinate_system"];
        CoordinateSystem parsed = global_coordinate_system;
        if (parse_coordinate_system(raw, parsed))
        {
            global_coordinate_system = parsed;
        }
        else
        {
            tmv::log_warn("Invalid coordinate_system '", raw,
                         "'. Valid values: cylindrical, cartesian. Falling back to '",
                         coordinate_system_name(global_coordinate_system), "'.");
        }
    }

    if (config.count("grid.staggering"))
    {
        const std::string& raw = config["grid.staggering"];
        StaggerType parsed = global_stagger_type;
        if (parse_stagger_type(raw, parsed))
        {
            global_stagger_type = parsed;
        }
        else
        {
            tmv::log_warn("Invalid grid.staggering '", raw,
                         "'. Valid values: collocated, c_grid. Falling back to '",
                         stagger_type_name(global_stagger_type), "'.");
        }
    }

    if (config.count("logging.profile"))
    {
        bool valid = false;
        const LogProfile parsed = parse_log_profile(config["logging.profile"], &valid);
        if (valid)
        {
            global_log_profile = parsed;
        }
        else
        {
            tmv::log_warn("Invalid logging.profile '", config["logging.profile"],
                         "'. Valid values: quiet, normal, debug. Using normal.");
            global_log_profile = LogProfile::normal;
        }
    }

    if (!quiet_override && log_normal_enabled())
    {
        tmv::log_info("Loaded config with ", config.size(), " keys");
    }

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
                "performance.report_every_steps",
                config["performance.report_every_steps"],
                "a non-negative integer");
        }
    }

    if (config.count("validation.guard_mode"))
    {
        tmv::GuardMode mode = global_validation_policy.mode;
        if (tmv::parse_guard_mode(config["validation.guard_mode"], mode))
        {
            global_validation_policy.mode = mode;
        }
        else
        {
            tmv::log_warn("Invalid validation.guard_mode '",
                         config["validation.guard_mode"],
                         "'. Valid values: off, sanitize, strict.");
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
            tmv::log_warn("Invalid validation.guard_fail_on '",
                         config["validation.guard_fail_on"],
                         "'. Valid values: nonfinite, bounds, both.");
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
            tmv::log_warn("Invalid validation.guard_scope '",
                         config["validation.guard_scope"],
                         "'. Valid values: required, exported.");
        }
    }
    if (config.count("validation.report_path"))
    {
        global_validation_report_path = config["validation.report_path"];
    }

    if (config.count("duration_s"))
    {
        int parsed = 0;
        if (try_parse_int_value(config["duration_s"], parsed))
        {
            duration_s = parsed;
        }
        else
        {
            warn_invalid_config_value("duration_s", config["duration_s"], "an integer");
        }
    }

    if (config.count("output.write_every_s"))
    {
        int parsed = 0;
        if (try_parse_int_value(config["output.write_every_s"], parsed))
        {
            write_every_s = parsed;
        }
        else
        {
            warn_invalid_config_value("output.write_every_s", config["output.write_every_s"], "an integer");
        }
    }

    if (live_shm && config.count("output.live_shm"))
    {
        *live_shm = parse_bool_value(config["output.live_shm"]);
    }
    if (live_shm_fields && config.count("output.live_shm_fields"))
    {
        const std::string& raw = config["output.live_shm_fields"];
        live_shm_fields->clear();
        std::istringstream stream(raw);
        std::string token;
        while (std::getline(stream, token, ','))
        {
            // Trim whitespace from each field name
            auto start = token.find_first_not_of(" \t");
            auto end = token.find_last_not_of(" \t");
            if (start != std::string::npos)
            {
                live_shm_fields->push_back(token.substr(start, end - start + 1));
            }
        }
    }

    if (output_config)
    {
        *output_config = parse_output_config(config);
    }

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
                "environment.cape_target_jkg",
                config["environment.cape_target_jkg"],
                "a finite number");
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
                "environment.tropopause_z_m",
                config["environment.tropopause_z_m"],
                "a finite number");
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
            tmv::log_warn("Invalid environment.sounding.interpolation_method '",
                         config["environment.sounding.interpolation_method"],
                         "'. Valid values: linear|0, spline|1, log-linear|2. Using ",
                         sounding_interpolation_method_name(global_runtime_sounding_config.interpolation_method), ".");
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
            warn_invalid_config_value(
                "environment.sounding.min_pressure_hpa",
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
            warn_invalid_config_value(
                "environment.sounding.max_pressure_hpa",
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
            warn_invalid_config_value(
                "environment.sounding.min_temperature_k",
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
            warn_invalid_config_value(
                "environment.sounding.max_temperature_k",
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
        global_runtime_sounding_config.scheme_id = "sharpy";
    }

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
                "trigger.bubble.center_x_km",
                config["trigger.bubble.center_x_km"],
                "a finite number");
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
                "trigger.bubble.center_y_km",
                config["trigger.bubble.center_y_km"],
                "a finite number");
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
                "trigger.bubble.center_z_km",
                config["trigger.bubble.center_z_km"],
                "a finite number");
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
                "trigger.bubble.radius_km",
                config["trigger.bubble.radius_km"],
                "a finite number");
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

    if (config.count("nested.enabled"))
    {
        nested_config.enabled = parse_bool_value(config["nested.enabled"]);
    }

    if (config.count("nested.refinement"))
    {
        double parsed = 0.0;
        if (try_parse_double_value(config["nested.refinement"], parsed))
        {
            nested_config.refinement = parsed;
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
            nested_config.nest_size_r = parsed;
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
            nested_config.nest_size_th = parsed;
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
            nested_config.nest_size_z = parsed;
        }
        else
        {
            warn_invalid_config_value("nested.size_z", config["nested.size_z"], "an integer");
        }
    }

    bool grid_changed = false;

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
                grid_changed = true;
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
                grid_changed = true;
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
                grid_changed = true;
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

    // Auto-scale grid to fit available RAM before allocating fields.
    // This prevents OOM on memory-constrained machines (e.g., 8GB
    // MacBook Air running a production config). Must happen BEFORE
    // resize_fields() -- that's the allocation point.
    {
        HardwareInfo hw = detect_hardware();
        if (apply_hardware_defaults(hw, NR, NTH, NZ))
        {
            tmv::log_info("Grid auto-scaled to ", NR, "x", NTH, "x", NZ,
                         " to fit available RAM");
        }
    }

    {
        extern void resize_fields();
        resize_fields();
    }

    // grid.dy must be applied AFTER resize_fields(), which calls
    // update_grid_resolution(). That function sets the default dtheta:
    //   Cylindrical → 2π/NTH
    //   Cartesian   → dr (square cells)
    // If the config specifies a non-square dy, override here.
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
                "environment.hodograph.u_sfc_ms",
                config["environment.hodograph.u_sfc_ms"],
                "a finite number");
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
                "environment.hodograph.v_sfc_ms",
                config["environment.hodograph.v_sfc_ms"],
                "a finite number");
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
                "environment.hodograph.u_1km_ms",
                config["environment.hodograph.u_1km_ms"],
                "a finite number");
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
                "environment.hodograph.v_1km_ms",
                config["environment.hodograph.v_1km_ms"],
                "a finite number");
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
                "environment.hodograph.u_6km_ms",
                config["environment.hodograph.u_6km_ms"],
                "a finite number");
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
                "environment.hodograph.v_6km_ms",
                config["environment.hodograph.v_6km_ms"],
                "a finite number");
        }
    }

    if (config.count("microphysics.scheme"))
    {
        std::string requested_scheme = config["microphysics.scheme"];
        std::vector<std::string> valid_schemes = {"kessler", "lin", "thompson", "milbrandt"};
        bool valid = false;

        for (const auto& scheme : valid_schemes)
        {
            if (requested_scheme == scheme) 
            {
                valid = true;
                break;
            }
        }

        if (valid) 
        {
            global_microphysics_scheme = requested_scheme;
        } 
        else
        {
            std::string opts;
            for (size_t i = 0; i < valid_schemes.size(); ++i)
            {
                if (i > 0) opts += ", ";
                opts += valid_schemes[i];
            }
            tmv::log_warn("Invalid microphysics scheme '", requested_scheme,
                          "'. Valid options: ", opts, ". Using default: kessler");
            global_microphysics_scheme = "kessler";
        }
    }

    if (config.count("radiation.scheme"))
    {
        const std::string requested_scheme = config["radiation.scheme"];
        const std::string canonical = canonicalize_radiation_scheme_id(requested_scheme);
        const std::vector<std::string> known_schemes = get_known_radiation_schemes();

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
            tmv::log_warn("Requested radiation scheme '", requested_scheme,
                         "' maps to '", canonical,
                         "', which is recognized but not implemented in this build. "
                         "Using default: simple_grey");
            global_radiation_config.scheme_id = "simple_grey";
        }
        else
        {
            std::string opts;
            for (size_t i = 0; i < known_schemes.size(); ++i)
            {
                if (i > 0) opts += ", ";
                opts += known_schemes[i];
            }
            tmv::log_warn("Invalid radiation scheme '", requested_scheme,
                          "'. Known options: ", opts, ". Using default: simple_grey");
            global_radiation_config.scheme_id = "simple_grey";
        }
    }

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
            warn_invalid_config_value("radiation.dt", config["radiation.dt"], "a positive finite number");
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
            warn_invalid_config_value(
                "radiation.dt_radiation",
                config["radiation.dt_radiation"],
                "a positive finite number");
        }
    }

    if (config.count("radiation.clear_sky_only"))
    {
        global_radiation_config.clear_sky_only = parse_bool_value(config["radiation.clear_sky_only"]);
    }
    if (config.count("radiation.use_subcycling"))
    {
        global_radiation_config.use_subcycling = parse_bool_value(config["radiation.use_subcycling"]);
    }
    if (config.count("radiation.enable_diagnostics"))
    {
        global_radiation_config.enable_diagnostics = parse_bool_value(config["radiation.enable_diagnostics"]);
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
            warn_invalid_config_value("radiation.tau_lw_ref", config["radiation.tau_lw_ref"], "a non-negative finite number");
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
            warn_invalid_config_value("radiation.tau_sw_ref", config["radiation.tau_sw_ref"], "a non-negative finite number");
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
            warn_invalid_config_value("radiation.n_lw", config["radiation.n_lw"], "a positive finite number");
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
            warn_invalid_config_value("radiation.n_sw", config["radiation.n_sw"], "a positive finite number");
        }
    }

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
        global_boundary_layer_config.clear_sky_only =
            parse_bool_value(config["boundary_layer.clear_sky_only"]);
    }
    if (config.count("boundary_layer.enable_nonlocal_transport"))
    {
        global_boundary_layer_config.enable_nonlocal_transport =
            parse_bool_value(config["boundary_layer.enable_nonlocal_transport"]);
    }
    if (config.count("boundary_layer.enable_tke"))
    {
        global_boundary_layer_config.enable_tke =
            parse_bool_value(config["boundary_layer.enable_tke"]);
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
            warn_invalid_config_value(
                "boundary_layer.pbl_top_method",
                config["boundary_layer.pbl_top_method"],
                "one of: ri, theta, tke");
        }
    }

    auto parse_surface_layer_id_key = [&](const char* key) {
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
            warn_invalid_config_value(
                key,
                config[key],
                "one of: bulk, monin_obukhov");
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
            warn_invalid_config_value("boundary_layer.dt", config["boundary_layer.dt"], "a positive finite number");
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
            warn_invalid_config_value("boundary_layer.dt_pbl", config["boundary_layer.dt_pbl"], "a positive finite number");
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
            warn_invalid_config_value(
                "boundary_layer.pbl_max_height",
                config["boundary_layer.pbl_max_height"],
                "a positive finite number");
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
            warn_invalid_config_value(
                "boundary_layer.min_ustar",
                config["boundary_layer.min_ustar"],
                "a non-negative finite number");
        }
    }

    if (config.count("surface.z0m"))
    {
        double parsed = 0.0;
        if (try_parse_double_value(config["surface.z0m"], parsed) && parsed > 0.0)
        {
            global_surface_config.z0m = parsed;
        }
        else
        {
            warn_invalid_config_value("surface.z0m", config["surface.z0m"], "a positive finite number");
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
            warn_invalid_config_value("surface.z0h", config["surface.z0h"], "a positive finite number");
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
            warn_invalid_config_value("surface.albedo", config["surface.albedo"], "a finite number in [0, 1]");
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
            warn_invalid_config_value("surface.emissivity", config["surface.emissivity"], "a finite number in [0, 1]");
        }
    }

    if (config.count("surface.tsfc"))
    {
        double parsed = 0.0;
        if (try_parse_double_value(config["surface.tsfc"], parsed))
        {
            global_surface_config.Tsfc = parsed;
        }
        else
        {
            warn_invalid_config_value("surface.tsfc", config["surface.tsfc"], "a finite number");
        }
    }
    else if (config.count("surface.Tsfc"))
    {
        double parsed = 0.0;
        if (try_parse_double_value(config["surface.Tsfc"], parsed))
        {
            global_surface_config.Tsfc = parsed;
        }
        else
        {
            warn_invalid_config_value("surface.Tsfc", config["surface.Tsfc"], "a finite number");
        }
    }

    if (config.count("surface.qsfc"))
    {
        double parsed = 0.0;
        if (try_parse_double_value(config["surface.qsfc"], parsed) && parsed >= 0.0)
        {
            global_surface_config.qsfc = parsed;
        }
        else
        {
            warn_invalid_config_value("surface.qsfc", config["surface.qsfc"], "a non-negative finite number");
        }
    }
    else if (config.count("surface.qv_sfc"))
    {
        double parsed = 0.0;
        if (try_parse_double_value(config["surface.qv_sfc"], parsed) && parsed >= 0.0)
        {
            global_surface_config.qsfc = parsed;
        }
        else
        {
            warn_invalid_config_value("surface.qv_sfc", config["surface.qv_sfc"], "a non-negative finite number");
        }
    }

    if (config.count("turbulence.scheme"))
    {
        global_turbulence_config.scheme_id =
            normalize_turbulence_scheme_name(config["turbulence.scheme"]);
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
            warn_invalid_config_value(
                "turbulence.mode",
                config["turbulence.mode"],
                "one of: 3d, horizontal_only, vertical_only");
        }
    }

    auto parse_turbulence_dt_key = [&](const char* key) {
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
            warn_invalid_config_value(key, config[key], "a positive finite number");
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
            warn_invalid_config_value("turbulence.Cs", config["turbulence.Cs"], "a non-negative finite number");
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
            warn_invalid_config_value("turbulence.Pr_t", config["turbulence.Pr_t"], "a positive finite number");
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
            warn_invalid_config_value("turbulence.Sc_t", config["turbulence.Sc_t"], "a positive finite number");
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
            warn_invalid_config_value("turbulence.nu_t_max", config["turbulence.nu_t_max"], "a non-negative finite number");
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
            warn_invalid_config_value("turbulence.K_max", config["turbulence.K_max"], "a non-negative finite number");
        }
    }

    auto parse_filter_width_key = [&](const char* key) {
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
            warn_invalid_config_value(key, config[key], "one of: dx, cubic_root, user");
        }
    };
    parse_filter_width_key("turbulence.filter_width");
    parse_filter_width_key("turbulence.smagorinsky.filter_width");

    auto parse_stability_correction_key = [&](const char* key) {
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
            warn_invalid_config_value(key, config[key], "one of: none, ri, tke_based");
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
        global_turbulence_config.dynamic_Cs =
            parse_bool_value(config["turbulence.smagorinsky.dynamic_procedure"]);
    }

    if (config.count("turbulence.enable_moist_buoyancy"))
    {
        global_turbulence_config.enable_moist_buoyancy =
            parse_bool_value(config["turbulence.enable_moist_buoyancy"]);
    }
    if (config.count("turbulence.enable_diagnostics"))
    {
        global_turbulence_config.enable_diagnostics =
            parse_bool_value(config["turbulence.enable_diagnostics"]);
    }

    if (config.count("dynamics.scheme"))
    {
        global_dynamics_scheme_name = config["dynamics.scheme"];
    }

    if (config.count("numerics.compute.backend"))
    {
        ComputeBackendKind parsed = ComputeBackendKind::Cpu;
        if (parse_compute_backend_kind(config["numerics.compute.backend"], parsed))
        {
            global_compute_backend_config.backend = compute_backend_kind_name(parsed);
        }
        else
        {
            throw std::runtime_error(
                "Invalid numerics.compute.backend '" + config["numerics.compute.backend"] +
                "'; expected one of: cpu, vulkan");
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
                "numerics.compute.device_index",
                config["numerics.compute.device_index"],
                "an integer >= -1");
        }
    }
    if (config.count("numerics.compute.allow_fallback"))
    {
        global_compute_backend_config.allow_fallback =
            parse_bool_value(config["numerics.compute.allow_fallback"]);
    }
    if (config.count("numerics.compute.validate_parity"))
    {
        global_compute_backend_config.validate_parity =
            parse_bool_value(config["numerics.compute.validate_parity"]);
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
            const auto templates = list_vertical_flux_templates();
            for (std::size_t i = 0; i < templates.size(); ++i)
            {
                if (i > 0)
                {
                    supported += ", ";
                }
                supported += templates[i].id;
            }
            warn_invalid_config_value(
                "numerics.compute.templates.vertical_flux",
                template_id,
                supported.empty() ? "a known template id" : supported.c_str());
        }
    }

    if (config.count("numerics.advection"))
    {
        global_advection_config.scheme_id = config["numerics.advection"];
    }
    if (config.count("numerics.advection.scheme"))
    {
        global_advection_config.scheme_id = config["numerics.advection.scheme"];
    }
    if (config.count("numerics.diffusion"))
    {
        global_diffusion_config.scheme_id = config["numerics.diffusion"];
    }
    if (config.count("numerics.diffusion.scheme"))
    {
        global_diffusion_config.scheme_id = config["numerics.diffusion.scheme"];
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
        global_diffusion_config.implicit_dim = config["numerics.diffusion.implicit_dim"];
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
            warn_invalid_config_value("numerics.diffusion.K_h", config["numerics.diffusion.K_h"], "a non-negative finite number");
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
            warn_invalid_config_value("numerics.diffusion.K_v", config["numerics.diffusion.K_v"], "a non-negative finite number");
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
            warn_invalid_config_value("numerics.diffusion.dt_diffusion", config["numerics.diffusion.dt_diffusion"], "a positive finite number");
        }
    }
    if (config.count("numerics.diffusion.use_variable_K"))
    {
        global_diffusion_config.use_variable_K = parse_bool_value(config["numerics.diffusion.use_variable_K"]);
    }
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
        if (try_parse_non_negative_int_value(config["numerics.time_stepping.n_acoustic_substeps"], parsed) && parsed > 0)
        {
            global_time_stepping_config.n_acoustic_substeps = parsed;
        }
        else
        {
            warn_invalid_config_value("numerics.time_stepping.n_acoustic_substeps", config["numerics.time_stepping.n_acoustic_substeps"], "a positive integer");
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
            warn_invalid_config_value("numerics.time_stepping.cfl_safety", config["numerics.time_stepping.cfl_safety"], "a positive finite number");
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
            warn_invalid_config_value("numerics.time_stepping.dt_min", config["numerics.time_stepping.dt_min"], "a positive finite number");
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
            warn_invalid_config_value("numerics.time_stepping.dt_max", config["numerics.time_stepping.dt_max"], "a positive finite number");
        }
    }
    if (global_time_stepping_config.dt_min > global_time_stepping_config.dt_max)
    {
        std::swap(global_time_stepping_config.dt_min, global_time_stepping_config.dt_max);
    }

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
    for (const auto& kv : config)
    {
        const std::string& key = kv.first;
        std::string override_field;
        bool override_is_min = true;
        if (parse_override_key(key, override_field, override_is_min))
        {
            auto& bounds = global_validation_policy.field_overrides[to_lower_copy(override_field)];
            if (override_is_min)
            {
                double parsed = 0.0;
                if (try_parse_double_value(kv.second, parsed))
                {
                    bounds.has_min = true;
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
                    bounds.has_max = true;
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
            const std::string var = key.substr(std::string("chaos.sigma_ic.").size());
            double parsed = 0.0;
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
            const std::string block = key.substr(std::string("chaos.alpha_tend.").size());
            double parsed = 0.0;
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

    if (config.count("terrain.scheme"))
    {
        const std::string requested = normalize_terrain_scheme_name(config["terrain.scheme"]);
        const std::vector<std::string> valid_schemes = get_available_terrain_schemes();
        bool valid = false;
        for (const auto& scheme : valid_schemes)
        {
            if (requested == scheme)
            {
                valid = true;
                break;
            }
        }

        if (valid)
        {
            global_terrain_config.scheme_id = requested;
        }
        else
        {
            std::string opts;
            for (std::size_t i = 0; i < valid_schemes.size(); ++i)
            {
                if (i > 0) opts += ", ";
                opts += valid_schemes[i];
            }
            tmv::log_warn("Invalid terrain.scheme '", config["terrain.scheme"],
                          "'. Valid options: ", opts, ". Keeping previous/default value.");
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
            warn_invalid_config_value(
                "terrain.coord_id",
                config["terrain.coord_id"],
                "\"btf\", \"smoothed\", or \"cartesian\" (aliases supported)");
        }
    }
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
            warn_invalid_config_value("terrain.smoothing_decay", config["terrain.smoothing_decay"], "a non-negative finite number");
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

    if (!quiet_override && log_normal_enabled())
    {
        std::cout << "\n" << "=" << std::string(60, '=') << std::endl;
        std::cout << "CONFIGURATION LOADED" << std::endl;
        std::cout << "=" << std::string(60, '=') << std::endl;
        std::cout << "Config file: " << config_path << std::endl;
        std::cout << "Duration: " << duration_s << "s" << std::endl;
        std::cout << "Write every: " << write_every_s << "s" << std::endl;
        std::cout << "\nGrid parameters:" << std::endl;
        std::cout << "  NR=" << NR << ", NTH=" << NTH << ", NZ=" << NZ << std::endl;
        std::cout << "  dr=" << dr << "m, dz=" << dz << "m, dt=" << dt << "s" << std::endl;
        {
            const uint64_t grid_bytes = estimate_grid_memory_bytes(NR, NTH, NZ);
            const uint64_t ram_bytes = detect_hardware().total_ram_bytes;
            const double grid_gb = static_cast<double>(grid_bytes) / (uint64_t(1) << 30);
            const auto prev_flags = std::cout.flags();
            const auto prev_prec = std::cout.precision();
            if (ram_bytes > 0)
            {
                const double ram_gb = static_cast<double>(ram_bytes) / (uint64_t(1) << 30);
                const double pct = 100.0 * static_cast<double>(grid_bytes) / ram_bytes;
                std::cout << "  Memory: ~" << std::fixed << std::setprecision(1)
                          << grid_gb << " GB / " << ram_gb << " GB"
                          << " (" << pct << "%)" << std::endl;
            }
            else
            {
                std::cout << "  Memory: ~" << std::fixed << std::setprecision(1)
                          << grid_gb << " GB" << std::endl;
            }
            std::cout.flags(prev_flags);
            std::cout.precision(prev_prec);
        }
        std::cout << "\nEnvironment:" << std::endl;
        std::cout << "  CAPE target: " << global_cape_target << " J/kg" << std::endl;
        std::cout << "  Surface theta: " << global_sfc_theta_k << " K" << std::endl;
        std::cout << "  Surface qv: " << global_sfc_qv_kgkg << " kg/kg" << std::endl;
        std::cout << "  Tropopause: " << global_tropopause_z_m << " m" << std::endl;
        std::cout << "  Wind profile: SFC(" << global_wind_profile.u_sfc << "," << global_wind_profile.v_sfc
                  << ") 1km(" << global_wind_profile.u_1km << "," << global_wind_profile.v_1km
                  << ") 6km(" << global_wind_profile.u_6km << "," << global_wind_profile.v_6km << ")" << std::endl;
        // The "center_x" field has two interpretations: a radial distance
        // from the singular axis (cylindrical) or a literal Cartesian
        // x-coordinate (Cartesian). Print whichever the active coordinate
        // system actually uses so the operator can sanity-check the IC.
        if (global_coordinate_system == CoordinateSystem::Cartesian)
        {
            std::cout << "  Trigger bubble: center_x=" << global_bubble_center_x_m / 1000.0
                      << " km, center_y=" << global_bubble_center_y_m / 1000.0
                      << " km, center_z=" << global_bubble_center_z_m / 1000.0
                      << " km, radius=" << global_bubble_radius_m / 1000.0
                      << " km, dtheta=" << global_bubble_dtheta_k << " K" << std::endl;
        }
        else
        {
            std::cout << "  Trigger bubble: center_r=" << global_bubble_center_x_m / 1000.0
                      << " km, center_z=" << global_bubble_center_z_m / 1000.0
                      << " km, radius=" << global_bubble_radius_m / 1000.0
                      << " km, dtheta=" << global_bubble_dtheta_k << " K" << std::endl;
        }
        if (global_sounding_enabled)
        {
            std::cout << "  Sounding: enabled"
                      << " (scheme: " << (global_runtime_sounding_config.scheme_id.empty() ? "none" : global_runtime_sounding_config.scheme_id)
                      << ", interpolation: " << sounding_interpolation_method_name(global_runtime_sounding_config.interpolation_method)
                      << ", fallback: " << (global_runtime_sounding_config.use_fallback_profiles ? "on" : "off")
                      << ", placeholder_profiles: " << (global_sounding_allow_placeholder_profiles ? "allowed" : "blocked")
                      << ")" << std::endl;
            if (!global_runtime_sounding_config.file_path.empty())
            {
                std::cout << "    file_path: " << global_runtime_sounding_config.file_path << std::endl;
            }
        }
        else
        {
            std::cout << "  Sounding: disabled" << std::endl;
        }
        std::cout << "\nPhysics schemes:" << std::endl;
        const std::string effective_dynamics_scheme =
            global_dynamics_scheme_name.empty() ? "tornado (default)" : global_dynamics_scheme_name;
        std::cout << "  Dynamics: " << effective_dynamics_scheme << std::endl;
        std::cout << "  Microphysics: " << global_microphysics_scheme << std::endl;
        std::cout << "  Radiation: " << global_radiation_config.scheme_id
                  << " (LW: " << (global_radiation_config.do_lw ? "on" : "off")
                  << ", SW: " << (global_radiation_config.do_sw ? "on" : "off")
                  << ", dt: " << global_radiation_config.dt_radiation << "s)" << std::endl;
        std::cout << "  Boundary Layer: " << global_boundary_layer_config.scheme_id
                  << " (surface fluxes: " << (global_boundary_layer_config.apply_surface_fluxes ? "on" : "off")
                  << ", layer: " << global_boundary_layer_config.surface_layer_id
                  << ", dt: " << global_boundary_layer_config.dt_pbl << "s"
                  << ", pbl_top: " << global_boundary_layer_config.pbl_top_method
                  << ", min_ustar: " << global_boundary_layer_config.min_ustar << " m/s)" << std::endl;
        std::cout << "  Turbulence: " << global_turbulence_config.scheme_id
                  << " (Cs: " << global_turbulence_config.Cs
                  << ", dt: " << global_turbulence_config.dt_sgs << "s)" << std::endl;
        std::cout << "  Numerics-Advection: " << global_advection_config.scheme_id << std::endl;
        std::cout << "  Numerics-Diffusion: " << global_diffusion_config.scheme_id
                  << " (apply_to: " << global_diffusion_config.apply_to
                  << ", K_h: " << global_diffusion_config.K_h
                  << ", K_v: " << global_diffusion_config.K_v << ")" << std::endl;
        std::cout << "  Numerics-TimeStepping: " << global_time_stepping_config.scheme_id
                  << " (adaptive_dt: " << (global_time_stepping_config.adaptive_dt ? "on" : "off")
                  << ", dt_min: " << global_time_stepping_config.dt_min
                  << "s, dt_max: " << global_time_stepping_config.dt_max
                  << "s, cfl_safety: " << global_time_stepping_config.cfl_safety << ")" << std::endl;
        std::cout << "  Numerics-Compute: " << global_compute_backend_config.backend
                  << " (device_index: " << global_compute_backend_config.device_index
                  << ", fallback: " << (global_compute_backend_config.allow_fallback ? "on" : "off")
                  << ", parity: " << (global_compute_backend_config.validate_parity ? "on" : "off")
                  << ")" << std::endl;
        std::cout << "  Numerics-KernelTemplate: vertical_flux=" << active_vertical_flux_template_id()
                  << std::endl;
        std::cout << "  Chaos: " << global_chaos_config.scheme_id << std::endl;
        std::cout << "  Terrain: " << global_terrain_config.scheme_id
                  << " (coord: " << global_terrain_config.coord_id
                  << ", ztop: " << global_terrain_config.ztop << "m)" << std::endl;
        std::cout << "  Validation: mode=" << tmv::to_string(global_validation_policy.mode)
                  << ", fail_on=" << tmv::to_string(global_validation_policy.fail_on)
                  << ", scope=" << tmv::to_string(global_validation_policy.strict_scope);
        if (!global_validation_report_path.empty())
        {
            std::cout << ", report_path=" << global_validation_report_path;
        }
        std::cout << std::endl;
        std::cout << "  Logging profile: " << log_profile_name(global_log_profile) << std::endl;
        std::cout << "  Perf timing: " << (global_perf_timing_enabled ? "on" : "off");
        if (global_perf_report_every_steps > 0)
        {
            std::cout << " (report every " << global_perf_report_every_steps << " steps)";
        }
        std::cout << std::endl;
        std::cout << "=" << std::string(60, '=') << "\n" << std::endl;
    }
}
