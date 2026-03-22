#pragma once

#include <cstddef>
#include <filesystem>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/field3d.hpp"
#include "diagnostics/field_contract.hpp"

/**
 * @file field_validation.hpp
 * @brief Runtime validation interfaces for physical and diagnostic fields.
 *
 * Defines guard policy, validation results, report structures,
 * and parsing helpers used to enforce field-quality constraints.
 * Supports sanitize and strict guard modes.
 */

namespace tmv
{

enum class GuardMode
{
    Off,
    Sanitize,
    Strict,
};

enum class GuardFailOn
{
    NonFinite,
    Bounds,
    Both,
};

enum class StrictGuardScope
{
    RequiredOnly,
    ExportedNow,
};

struct ValidationPolicy
{
    GuardMode mode = GuardMode::Sanitize;
    GuardFailOn fail_on = GuardFailOn::Both;
    StrictGuardScope strict_scope = StrictGuardScope::RequiredOnly;
    std::unordered_map<std::string, FieldBounds> field_overrides;
};

struct FieldStats
{
    std::size_t total_count = 0;
    std::size_t finite_count = 0;
    std::size_t nan_count = 0;
    std::size_t inf_count = 0;
    std::size_t below_min_count = 0;
    std::size_t above_max_count = 0;
    std::size_t sanitized_nonfinite_count = 0;
    std::size_t sanitized_bounds_count = 0;
    double min_value = 0.0;
    double max_value = 0.0;
    double mean_value = 0.0;
    double p01 = 0.0;
    double p50 = 0.0;
    double p99 = 0.0;
    bool has_finite = false;
};

struct FieldViolation
{
    std::string field_id;
    std::string reason;
    std::size_t count = 0;
    bool critical = false;
};

struct FieldValidationResult
{
    FieldStats stats;
    std::vector<FieldViolation> violations;
    bool failed = false;
};

struct FieldValidationReport
{
    std::string field_id;
    std::string status;
    std::string requirement;
    FieldValidationResult result;
};

struct ValidationReport
{
    std::string context;
    int step_index = -1;
    std::string guard_mode;
    std::string guard_fail_on;
    std::string guard_scope;
    bool failed = false;
    std::vector<FieldValidationReport> fields;
    std::vector<std::string> missing_required;
    std::vector<std::string> missing_exported;
    std::vector<std::string> missing_not_implemented;
    std::vector<std::string> known_not_implemented;
};

/**
 * @brief Parses guard mode from text.
 */
bool parse_guard_mode(const std::string& value, GuardMode& out_mode);

/**
 * @brief Parses guard failure mode from text.
 */
bool parse_guard_fail_on(const std::string& value, GuardFailOn& out_mode);

/**
 * @brief Parses strict-guard scope from text.
 */
bool parse_strict_guard_scope(const std::string& value, StrictGuardScope& out_scope);

/**
 * @brief Converts guard mode to text.
 */
const char* to_string(GuardMode mode);

/**
 * @brief Converts guard-fail mode to text.
 */
const char* to_string(GuardFailOn fail_on);

/**
 * @brief Converts strict-scope mode to text.
 */
const char* to_string(StrictGuardScope scope);

/**
 * @brief Resolves field bounds including policy overrides.
 */
FieldBounds effective_bounds_for_field(const FieldContract& contract, const ValidationPolicy& policy);

/**
 * @brief Validates and optionally sanitizes a raw float buffer.
 */
FieldValidationResult validate_buffer_inplace(float* values,
                                              std::size_t count,
                                              const FieldContract& contract,
                                              const ValidationPolicy& policy,
                                              bool include_percentiles);

/**
 * @brief Validates and optionally sanitizes a Field3D in place.
 */
FieldValidationResult validate_field3d_inplace(Field3D& field,
                                               const FieldContract& contract,
                                               const ValidationPolicy& policy,
                                               bool include_percentiles);

/**
 * @brief Serializes a validation report to JSON.
 */
std::string validation_report_to_json(const ValidationReport& report);

/**
 * @brief Writes a validation report to a JSON file.
 */
bool write_validation_report_json(const ValidationReport& report,
                                  const std::filesystem::path& path,
                                  std::string& error);

} // namespace tmv
