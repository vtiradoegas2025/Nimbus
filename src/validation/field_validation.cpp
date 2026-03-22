/**
 * @file field_validation.cpp
 * @brief Implementation for the validation module.
 *
 * Provides executable logic for the validation runtime path,
 * including initialization, stepping, and diagnostics helpers.
 * This file is part of the src/validation subsystem.
 */

#include "diagnostics/field_validation.hpp"
#include "util/string_utils.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <utility>

namespace tmv 
{namespace 
{

/**
 * @brief Returns true when strict mode should fail on non-finite values.
 */
bool should_fail_on_nonfinite(const ValidationPolicy& policy) 
{
    return policy.fail_on == GuardFailOn::NonFinite || policy.fail_on == GuardFailOn::Both;
}

/**
 * @brief Returns true when strict mode should fail on bounds violations.
 */
bool should_fail_on_bounds(const ValidationPolicy& policy) 
{
    return policy.fail_on == GuardFailOn::Bounds || policy.fail_on == GuardFailOn::Both;
}

/**
 * @brief Checks whether a contract participates in current strict scope.
 */
bool in_strict_scope(const FieldContract& contract, const ValidationPolicy& policy) 
{
    switch (policy.strict_scope) 
    {
        case StrictGuardScope::RequiredOnly:return contract.requirement == FieldRequirementTier::RequiredNow;

        case StrictGuardScope::ExportedNow:return contract.status == FieldImplementationStatus::ExportedNow;

        default:return contract.requirement == FieldRequirementTier::RequiredNow;
    }
}

/**
 * @brief Finite check wrapper for single-precision samples.
 */
bool is_finite(float value) {return std::isfinite(static_cast<double>(value));}

/**
 * @brief Selects a replacement value for non-finite entries during sanitization.
 */
float replacement_value_for_nonfinite(const FieldBounds& bounds) 
{
    if (bounds.has_min && bounds.has_max) 
    {
        const double clamped_zero = std::max(bounds.min_value, std::min(bounds.max_value, 0.0));
        return static_cast<float>(clamped_zero);
    }
    if (bounds.has_min) { return static_cast<float>(bounds.min_value);}
    if (bounds.has_max) {return static_cast<float>(std::min(bounds.max_value, 0.0));}
    return 0.0f;
}

/**
 * @brief Returns true when at least one bound limit is active.
 */
bool has_bounds(const FieldBounds& bounds) {return bounds.has_min || bounds.has_max;}

/**
 * @brief Computes representative percentiles from sampled finite values.
 */
void compute_percentiles(std::vector<float>& samples, FieldStats& stats) {if (samples.empty()) {return;}

    std::sort(samples.begin(), samples.end());

    const auto pick = [&samples](double q) -> double 
    {
        const double clamped_q = std::max(0.0, std::min(1.0, q));
        const double idx = clamped_q * static_cast<double>(samples.size() - 1);
        const std::size_t i = static_cast<std::size_t>(idx);
        return static_cast<double>(samples[i]);
    };

    stats.p01 = pick(0.01);
    stats.p50 = pick(0.50);
    stats.p99 = pick(0.99);
}

}

/**
 * @brief Parses guard mode text into enum representation.
 */
bool parse_guard_mode(const std::string& value, GuardMode& out_mode) 
{
    const std::string v = strutil::lower_copy(value);
    if (v == "off") 
    {
        out_mode = GuardMode::Off;
        return true;
    }
    if (v == "sanitize") 
    {
        out_mode = GuardMode::Sanitize;
        return true;
    }
    if (v == "strict") 
    {
        out_mode = GuardMode::Strict;
        return true;
    }
    return false;
}

/**
 * @brief Parses strict-failure criterion text into enum representation.
 */
bool parse_guard_fail_on(const std::string& value, GuardFailOn& out_mode) 
{
    const std::string v = strutil::lower_copy(value);
    if (v == "nonfinite") 
    {
        out_mode = GuardFailOn::NonFinite;
        return true;
    }
    if (v == "bounds") 
    {
        out_mode = GuardFailOn::Bounds;
        return true;
    }
    if (v == "both") 
    {
        out_mode = GuardFailOn::Both;
        return true;
    }
    return false;
}

/**
 * @brief Parses strict scope text into enum representation.
 */
bool parse_strict_guard_scope(const std::string& value, StrictGuardScope& out_scope) 
{
    const std::string v = strutil::lower_copy(value);
    if (v == "required" || v == "required_only") 
    {
        out_scope = StrictGuardScope::RequiredOnly;
        return true;
    }
    if (v == "exported" || v == "exported_now" || v == "all_exported") 
    {
        out_scope = StrictGuardScope::ExportedNow;
        return true;
    }
    return false;
}

/**
 * @brief Converts guard mode enum to stable string id.
 */
const char* to_string(GuardMode mode) 
{
    switch (mode) 
    {
        case GuardMode::Off: return "off";

        case GuardMode::Sanitize: return "sanitize";

        case GuardMode::Strict: return "strict";

        default:return "off";
    }
}

/**
 * @brief Converts guard failure criterion enum to stable string id.
 */
const char* to_string(GuardFailOn fail_on) 
{
    switch (fail_on) 
    {
        case GuardFailOn::NonFinite: return "nonfinite";

        case GuardFailOn::Bounds: return "bounds";

        case GuardFailOn::Both: return "both";

        default: return "both";
    }
}

/**
 * @brief Converts strict scope enum to stable string id.
 */
const char* to_string(StrictGuardScope scope) 
{
    switch (scope) 
    {
        case StrictGuardScope::RequiredOnly: return "required_only";
        
        case StrictGuardScope::ExportedNow: return "exported_now";

        default: return "required_only";
    }
}

/**
 * @brief Resolves effective bounds after applying policy overrides.
 */
FieldBounds effective_bounds_for_field(const FieldContract& contract, const ValidationPolicy& policy) {
    const std::string key = strutil::lower_copy(contract.id);

    auto it = policy.field_overrides.find(key);
    if (it != policy.field_overrides.end()) {
        return it->second;
    }

    it = policy.field_overrides.find(contract.id);
    if (it != policy.field_overrides.end()) {
        return it->second;
    }

    return contract.default_bounds;
}

/**
 * @brief Validates and optionally sanitizes a contiguous field buffer.
 */
FieldValidationResult validate_buffer_inplace(float* values, std::size_t count, const FieldContract& contract,
                                              const ValidationPolicy& policy, bool include_percentiles) 
{
    FieldValidationResult out;
    FieldStats& stats = out.stats;

    stats.total_count = count;

    const FieldBounds bounds = effective_bounds_for_field(contract, policy);
    const bool bounds_enabled = contract.severity.check_bounds && has_bounds(bounds);
    const bool nonfinite_enabled = contract.severity.check_nonfinite;
    const bool sanitize = policy.mode == GuardMode::Sanitize;
    const float nonfinite_replacement = replacement_value_for_nonfinite(bounds);

    std::size_t finite_count = 0;
    std::size_t nan_count = 0;
    std::size_t inf_count = 0;
    std::size_t below_min_count = 0;
    std::size_t above_max_count = 0;
    std::size_t sanitized_nonfinite_count = 0;
    std::size_t sanitized_bounds_count = 0;
    double finite_sum = 0.0;
    double finite_min = std::numeric_limits<double>::infinity();
    double finite_max = -std::numeric_limits<double>::infinity();

    std::vector<float> percentile_samples;
    if (include_percentiles && count > 0) 
    {
        constexpr std::size_t kMaxSamples = 4096;
        const std::size_t stride = std::max<std::size_t>(1, count / kMaxSamples);
        percentile_samples.reserve((count / stride) + 1);

        for (std::size_t i = 0; i < count; i += stride) 
        {
            const float v = values[i];
            if (is_finite(v)) 
            {
                percentile_samples.push_back(v);
            }
        }
    }

    #pragma omp parallel for reduction(+:finite_count,nan_count,inf_count,below_min_count,above_max_count,sanitized_nonfinite_count,sanitized_bounds_count,finite_sum) reduction(min:finite_min) reduction(max:finite_max)
    for (long long i = 0; i < static_cast<long long>(count); ++i) 
    {
        float value = values[static_cast<std::size_t>(i)];

        if (!is_finite(value)) {
            if (std::isnan(static_cast<double>(value))) 
            {
                ++nan_count;
            } else 
            {
                ++inf_count;
            }

            if (sanitize && nonfinite_enabled) 
            {
                values[static_cast<std::size_t>(i)] = nonfinite_replacement;
                ++sanitized_nonfinite_count;
            }
            continue;
        }

        ++finite_count;
        finite_sum += static_cast<double>(value);
        finite_min = std::min(finite_min, static_cast<double>(value));
        finite_max = std::max(finite_max, static_cast<double>(value));

        if (bounds_enabled) 
        {
            bool clamped = false;
            float clamped_value = value;

            if (bounds.has_min && static_cast<double>(value) < bounds.min_value) 
            {
                ++below_min_count;
                clamped_value = static_cast<float>(bounds.min_value);
                clamped = true;
            }
            if (bounds.has_max && static_cast<double>(value) > bounds.max_value) 
            {
                ++above_max_count;
                clamped_value = static_cast<float>(bounds.max_value);
                clamped = true;
            }

            if (sanitize && clamped) 
            {
                values[static_cast<std::size_t>(i)] = clamped_value;
                ++sanitized_bounds_count;
            }
        }
    }

    stats.finite_count = finite_count;
    stats.nan_count = nan_count;
    stats.inf_count = inf_count;
    stats.below_min_count = below_min_count;
    stats.above_max_count = above_max_count;
    stats.sanitized_nonfinite_count = sanitized_nonfinite_count;
    stats.sanitized_bounds_count = sanitized_bounds_count;

    stats.has_finite = stats.finite_count > 0;
    if (stats.has_finite) 
    {
        stats.min_value = finite_min;
        stats.max_value = finite_max;
        stats.mean_value = finite_sum / static_cast<double>(stats.finite_count);
        if (include_percentiles) 
        {
            compute_percentiles(percentile_samples, stats);
        }
    }

    const std::size_t nonfinite_count = stats.nan_count + stats.inf_count;
    if (nonfinite_enabled && nonfinite_count > 0) 
    {
        FieldViolation violation;
        violation.field_id = contract.id;
        violation.reason = "non_finite";
        violation.count = nonfinite_count;
        violation.critical =
            (policy.mode == GuardMode::Strict) &&
            in_strict_scope(contract, policy) &&
            should_fail_on_nonfinite(policy);
        out.violations.push_back(violation);
    }

    const std::size_t bounds_count = stats.below_min_count + stats.above_max_count;
    if (bounds_enabled && bounds_count > 0) 
    {
        FieldViolation violation;
        violation.field_id = contract.id;
        violation.reason = "out_of_bounds";
        violation.count = bounds_count;
        violation.critical =
            (policy.mode == GuardMode::Strict) &&
            in_strict_scope(contract, policy) &&
            should_fail_on_bounds(policy);
        out.violations.push_back(violation);
    }

    if (policy.mode == GuardMode::Strict && in_strict_scope(contract, policy)) 
    {
        if ((nonfinite_enabled && should_fail_on_nonfinite(policy) && nonfinite_count > 0) ||
            (bounds_enabled && should_fail_on_bounds(policy) && bounds_count > 0)) {
            out.failed = true;
        }
    }

    return out;
}

/**
 * @brief Validates and optionally sanitizes a `Field3D` in place.
 */
FieldValidationResult validate_field3d_inplace(Field3D& field, const FieldContract& contract,
                                               const ValidationPolicy& policy, bool include_percentiles) 
{
    return validate_buffer_inplace(field.data(), field.size(), contract, policy, include_percentiles);
}

/**
 * @brief Serializes a validation report to formatted JSON.
 */
std::string validation_report_to_json(const ValidationReport& report) 
{
    std::ostringstream oss;
    oss << "{\n";
    oss << "  \"context\": \"" << strutil::json_escape(report.context) << "\",\n";
    oss << "  \"step_index\": " << report.step_index << ",\n";
    oss << "  \"guard_mode\": \"" << strutil::json_escape(report.guard_mode) << "\",\n";
    oss << "  \"guard_fail_on\": \"" << strutil::json_escape(report.guard_fail_on) << "\",\n";
    oss << "  \"guard_scope\": \"" << strutil::json_escape(report.guard_scope) << "\",\n";
    oss << "  \"failed\": " << (report.failed ? "true" : "false") << ",\n";

    oss << "  \"missing_required\": [";
    for (std::size_t i = 0; i < report.missing_required.size(); ++i) 
    {
        if (i > 0) 
        {
            oss << ", ";
        }
        oss << "\"" << strutil::json_escape(report.missing_required[i]) << "\"";
    }
    oss << "],\n";

    oss << "  \"missing_exported\": [";
    for (std::size_t i = 0; i < report.missing_exported.size(); ++i) 
    {
        if (i > 0) 
        {
            oss << ", ";
        }
        oss << "\"" << strutil::json_escape(report.missing_exported[i]) << "\"";
    }
    oss << "],\n";

    oss << "  \"missing_not_implemented\": [";
    for (std::size_t i = 0; i < report.missing_not_implemented.size(); ++i) 
    {
        if (i > 0) 
        {
            oss << ", ";
        }
        oss << "\"" << strutil::json_escape(report.missing_not_implemented[i]) << "\"";
    }
    oss << "],\n";

    oss << "  \"known_not_implemented\": [";
    for (std::size_t i = 0; i < report.known_not_implemented.size(); ++i) 
    {
        if (i > 0) 
        {
            oss << ", ";
        }
        oss << "\"" << strutil::json_escape(report.known_not_implemented[i]) << "\"";
    }
    oss << "],\n";

    oss << "  \"fields\": [\n";
    for (std::size_t i = 0; i < report.fields.size(); ++i) 
    {
        const auto& field = report.fields[i];
        const auto& stats = field.result.stats;
        oss << "    {\n";
        oss << "      \"field_id\": \"" << strutil::json_escape(field.field_id) << "\",\n";
        oss << "      \"status\": \"" << strutil::json_escape(field.status) << "\",\n";
        oss << "      \"requirement\": \"" << strutil::json_escape(field.requirement) << "\",\n";
        oss << "      \"failed\": " << (field.result.failed ? "true" : "false") << ",\n";
        oss << "      \"stats\": {\n";
        oss << "        \"total_count\": " << stats.total_count << ",\n";
        oss << "        \"finite_count\": " << stats.finite_count << ",\n";
        oss << "        \"nan_count\": " << stats.nan_count << ",\n";
        oss << "        \"inf_count\": " << stats.inf_count << ",\n";
        oss << "        \"below_min_count\": " << stats.below_min_count << ",\n";
        oss << "        \"above_max_count\": " << stats.above_max_count << ",\n";
        oss << "        \"sanitized_nonfinite_count\": " << stats.sanitized_nonfinite_count << ",\n";
        oss << "        \"sanitized_bounds_count\": " << stats.sanitized_bounds_count << ",\n";
        oss << "        \"has_finite\": " << (stats.has_finite ? "true" : "false") << ",\n";
        oss << std::fixed << std::setprecision(6);
        oss << "        \"min\": " << stats.min_value << ",\n";
        oss << "        \"max\": " << stats.max_value << ",\n";
        oss << "        \"mean\": " << stats.mean_value << ",\n";
        oss << "        \"p01\": " << stats.p01 << ",\n";
        oss << "        \"p50\": " << stats.p50 << ",\n";
        oss << "        \"p99\": " << stats.p99 << "\n";
        oss << "      },\n";
        oss << "      \"violations\": [";
        for (std::size_t j = 0; j < field.result.violations.size(); ++j) {
            if (j > 0) {
                oss << ", ";
            }
            const auto& violation = field.result.violations[j];
            oss << "{\"reason\":\"" << strutil::json_escape(violation.reason)
                << "\",\"count\":" << violation.count
                << ",\"critical\":" << (violation.critical ? "true" : "false") << "}";
        }
        oss << "]\n";
        oss << "    }";
        if (i + 1 < report.fields.size()) 
        {
            oss << ",";
        }
        oss << "\n";
    }
    oss << "  ]\n";
    oss << "}\n";
    return oss.str();
}

/**
 * @brief Writes a validation report JSON file to disk.
 */
bool write_validation_report_json(const ValidationReport& report, const std::filesystem::path& path, std::string& error) 
{
    std::error_code ec;
    const auto parent = path.parent_path();
    if (!parent.empty()) 
    {
        std::filesystem::create_directories(parent, ec);
        if (ec) 
        {
            error = "failed to create report directory '" + parent.string() + "': " + ec.message();
            return false;
        }
    }

    std::ofstream out(path);
    if (!out) 
    {
        error = "failed to open report file for writing: " + path.string();
        return false;
    }

    out << validation_report_to_json(report);
    if (!out.good()) 
    {
        error = "failed to write report file: " + path.string();
        return false;
    }

    return true;
}

}
