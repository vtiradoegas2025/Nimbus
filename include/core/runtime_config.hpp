#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>

#include "core/coordinate_system.hpp"
#include "diagnostics/field_validation.hpp"
#include "data/soundings.hpp"

enum class LogProfile : int;

/**
 * @file runtime_config.hpp
 * @brief Runtime configuration globals and parsing helpers.
 *
 * Declares the process-wide configuration state used by startup code.
 * Provides parsing and conversion helpers used while reading YAML-like
 * configuration inputs for physics and diagnostics modules.
 */

extern bool global_sounding_enabled;
extern bool global_sounding_allow_placeholder_profiles;
extern SoundingConfig global_runtime_sounding_config;
extern std::string global_microphysics_scheme;
extern std::string global_dynamics_scheme_name;
extern tmv::ValidationPolicy global_validation_policy;
extern std::string global_validation_report_path;

/**
 * @brief Active horizontal coordinate system for the run.
 *
 * Defaults to CoordinateSystem::Cylindrical so existing tornado configurations
 * are unchanged. Set from the top-level `coordinate_system` YAML key in
 * load_config(). Used by the dynamics, advection, boundary-condition, and
 * initial-condition dispatchers introduced in Phase A of the Coordinate
 * Backend Plan (see docs/CoordinateBackend_Plan.md).
 */
extern CoordinateSystem global_coordinate_system;

/**
 * @brief Active grid staggering arrangement for the run.
 *
 * Defaults to StaggerType::Collocated so existing configurations are
 * unchanged. Set from the `grid.staggering` YAML key in load_config().
 * Used by the dynamics factory to select collocated vs C-grid scheme
 * variants (Phase C of the Coordinate Backend Plan).
 */
extern StaggerType global_stagger_type;

/**
 * @brief Returns a lowercased copy of the input.
 * @param value Input string.
 * @return Lowercased string.
 */
std::string to_lower_copy(std::string value);

/**
 * @brief Parses common truthy boolean spellings.
 * @param value Input string.
 * @return Parsed boolean value.
 */
bool parse_bool_value(const std::string& value);

/**
 * @brief Parses an integer value.
 * @param value Input string.
 * @param out Parsed integer output.
 * @return True on successful parse.
 */
bool try_parse_int_value(const std::string& value, int& out);

/**
 * @brief Parses a non-negative integer value.
 * @param value Input string.
 * @param out Parsed integer output.
 * @return True on successful parse and non-negative result.
 */
bool try_parse_non_negative_int_value(const std::string& value, int& out);

/**
 * @brief Parses an unsigned 64-bit integer value.
 * @param value Input string.
 * @param out Parsed unsigned output.
 * @return True on successful parse.
 */
bool try_parse_uint64_value(const std::string& value, std::uint64_t& out);

/**
 * @brief Parses a floating-point value.
 * @param value Input string.
 * @param out Parsed double output.
 * @return True on successful parse.
 */
bool try_parse_double_value(const std::string& value, double& out);

/**
 * @brief Parses a comma-separated or YAML bracketed string list.
 * @param value Input string, e.g. "[u, v, w]" or "u, v, w".
 * @return Vector of trimmed, non-empty tokens.
 */
std::vector<std::string> parse_string_list(const std::string& value);

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
double potential_temperature_from_temperature_pressure(double temperature_k,
                                                       double pressure_hpa);

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

/**
 * @brief Parses a simple key-value YAML file.
 * @param filename Input file path.
 * @return Parsed key-value map.
 */
std::unordered_map<std::string, std::string> parse_yaml_simple(const std::string& filename);

struct OutputConfig;

/**
 * @brief Loads runtime configuration from disk.
 * @param config_path Path to configuration file.
 * @param duration_s Output simulation duration in seconds.
 * @param write_every_s Output cadence in seconds.
 * @param outdir Output directory path.
 * @param output_config Optional output pipeline configuration.
 */
void load_config(const std::string& config_path,
                 int& duration_s,
                 int& write_every_s,
                 std::string& outdir,
                 OutputConfig* output_config = nullptr,
                 bool* live_shm = nullptr,
                 std::vector<std::string>* live_shm_fields = nullptr);
