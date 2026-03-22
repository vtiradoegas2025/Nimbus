/**
 * @file field_validator.cpp
 * @brief Implementation for the tools module.
 *
 * Provides executable logic for the tools runtime path,
 * including initialization, stepping, and diagnostics helpers.
 * This file is part of the src/tools subsystem.
 */

#include "diagnostics/field_contract.hpp"
#include "diagnostics/field_validation.hpp"
#include "util/string_utils.hpp"
#include "data/npy_reader.hpp"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <regex>
#include <sstream>
#include <set>
#include <string>
#include <vector>

namespace {

/**
 * @brief Parsed command-line options for the field validator tool.
 */
struct Options {
    std::string input_dir;
    std::string contract = "cm1";
    std::string mode = "report";
    std::string scope = "required";
    std::string json_path;
};

/**
 * @brief Parse outcomes for command-line argument processing.
 */
enum class ParseArgsResult {
    Ok,
    Help,
    Error,
};

/**
 * @brief Prints CLI usage help for the field validator.
 */
void print_usage() {
    std::cout << "Field Validator\n"
              << "Usage:\n"
              << "  bin/field_validator --input <dir> [--contract cm1] [--mode report|strict]\n"
              << "      [--scope required|exported] [--json <path>]\n";
}

/**
 * @brief Parses CLI arguments into an `Options` structure.
 */
ParseArgsResult parse_args(int argc, char** argv, Options& out) {
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto require_value = [&](const std::string& option) -> const char* {
            if (i + 1 >= argc) {
                std::cerr << "Missing value for " << option << "\n";
                return nullptr;
            }
            return argv[++i];
        };

        if (arg == "--help" || arg == "-h") {
            print_usage();
            return ParseArgsResult::Help;
        }
        if (arg == "--input") {
            const char* value = require_value(arg);
            if (value == nullptr) {
                return ParseArgsResult::Error;
            }
            out.input_dir = value;
            continue;
        }
        if (arg == "--contract") {
            const char* value = require_value(arg);
            if (value == nullptr) {
                return ParseArgsResult::Error;
            }
            out.contract = value;
            continue;
        }
        if (arg == "--mode") {
            const char* value = require_value(arg);
            if (value == nullptr) {
                return ParseArgsResult::Error;
            }
            out.mode = value;
            continue;
        }
        if (arg == "--scope") {
            const char* value = require_value(arg);
            if (value == nullptr) {
                return ParseArgsResult::Error;
            }
            out.scope = value;
            continue;
        }
        if (arg == "--json") {
            const char* value = require_value(arg);
            if (value == nullptr) {
                return ParseArgsResult::Error;
            }
            out.json_path = value;
            continue;
        }

        std::cerr << "Unknown argument: " << arg << "\n";
        return ParseArgsResult::Error;
    }

    if (out.input_dir.empty()) {
        std::cerr << "--input is required\n";
        return ParseArgsResult::Error;
    }

    if (out.contract != "cm1") {
        std::cerr << "Only --contract cm1 is supported in this revision\n";
        return ParseArgsResult::Error;
    }

    if (out.mode != "report" && out.mode != "strict") {
        std::cerr << "--mode must be report or strict\n";
        return ParseArgsResult::Error;
    }
    if (out.scope != "required" && out.scope != "exported") {
        std::cerr << "--scope must be required or exported\n";
        return ParseArgsResult::Error;
    }

    return ParseArgsResult::Ok;
}

/**
 * @brief Parses a non-negative integer from string input.
 */
bool try_parse_non_negative_int(const std::string& value, int& out) {
    try {
        std::size_t consumed = 0;
        const long long parsed = std::stoll(value, &consumed);
        if (consumed != value.size() ||
            parsed < 0 ||
            parsed > static_cast<long long>(std::numeric_limits<int>::max())) {
            return false;
        }
        out = static_cast<int>(parsed);
        return true;
    } catch (const std::exception&) {
        return false;
    }
}

}

/**
 * @brief Entry point for validation against exported CM1-style field contracts.
 */
int main(int argc, char** argv) {
    Options options;
    const ParseArgsResult parse_result = parse_args(argc, argv, options);
    if (parse_result == ParseArgsResult::Help) {
        return 0;
    }
    if (parse_result == ParseArgsResult::Error) {
        return 1;
    }

    std::filesystem::path input_root(options.input_dir);
    if (!std::filesystem::exists(input_root)) {
        std::cerr << "Input directory does not exist: " << input_root.string() << "\n";
        return 1;
    }
    if (!std::filesystem::is_directory(input_root)) {
        std::cerr << "Input path is not a directory: " << input_root.string() << "\n";
        return 1;
    }

    tmv::ValidationPolicy policy;
    policy.mode = (options.mode == "strict") ? tmv::GuardMode::Strict : tmv::GuardMode::Off;
    policy.fail_on = tmv::GuardFailOn::Both;
    policy.strict_scope = (options.scope == "exported")
        ? tmv::StrictGuardScope::ExportedNow
        : tmv::StrictGuardScope::RequiredOnly;

    const auto required_contracts = tmv::contracts_with_requirement(tmv::FieldRequirementTier::RequiredNow);
    const auto exported_contracts = tmv::contracts_with_status(tmv::FieldImplementationStatus::ExportedNow);
    const auto not_implemented_contracts = tmv::contracts_with_status(tmv::FieldImplementationStatus::NotImplemented);
    const auto not_implemented_required_now_inventory =
        tmv::known_not_implemented_required_now_inventory();
    const auto not_implemented_backlog_inventory =
        tmv::known_not_implemented_backlog_inventory();

    std::regex step_re("^step_([0-9]+)$");
    std::regex file_re("^th([0-9]+)_(.+)\\.npy$");

    std::vector<std::pair<int, std::filesystem::path>> steps;
    for (const auto& entry : std::filesystem::directory_iterator(input_root)) {
        if (!entry.is_directory()) {
            continue;
        }
        const std::string name = entry.path().filename().string();
        std::smatch match;
        if (std::regex_match(name, match, step_re) && match.size() >= 2) {
            int step_index = 0;
            if (!try_parse_non_negative_int(match[1].str(), step_index)) {
                std::cerr << "Skipping step directory with invalid index: " << name << "\n";
                continue;
            }
            steps.emplace_back(step_index, entry.path());
        }
    }
    std::sort(steps.begin(), steps.end(), [](const auto& a, const auto& b) {
        return a.first < b.first;
    });

    if (steps.empty()) {
        std::cerr << "No step_XXXXXX directories found in " << input_root.string() << "\n";
        return 1;
    }

    std::vector<tmv::ValidationReport> step_reports;
    bool overall_failed = false;

    for (const auto& step : steps) {
        std::map<std::string, std::vector<std::pair<int, std::filesystem::path>>> files_by_field;
        std::set<int> all_theta_indices;

        for (const auto& file_entry : std::filesystem::directory_iterator(step.second)) {
            if (!file_entry.is_regular_file()) {
                continue;
            }

            const std::string name = file_entry.path().filename().string();
            std::smatch match;
            if (!std::regex_match(name, match, file_re) || match.size() < 3) {
                continue;
            }

            int theta_idx = 0;
            if (!try_parse_non_negative_int(match[1].str(), theta_idx)) {
                std::cerr << "Skipping file with invalid theta index: " << name << "\n";
                continue;
            }
            const std::string file_field = match[2].str();
            const tmv::FieldContract* contract = tmv::find_field_contract(file_field);
            const std::string canonical_field = (contract != nullptr) ? contract->id : file_field;
            files_by_field[canonical_field].push_back({theta_idx, file_entry.path()});
            all_theta_indices.insert(theta_idx);
        }

        for (auto& kv : files_by_field) {
            std::sort(kv.second.begin(), kv.second.end(), [](const auto& a, const auto& b) {
                return a.first < b.first;
            });
        }

        tmv::ValidationReport report;
        report.context = step.second.filename().string();
        report.step_index = step.first;
        report.guard_mode = tmv::to_string(policy.mode);
        report.guard_fail_on = tmv::to_string(policy.fail_on);
        report.guard_scope = tmv::to_string(policy.strict_scope);

        for (const tmv::FieldContract* contract : required_contracts) {
            if (contract->status != tmv::FieldImplementationStatus::ExportedNow) {
                continue;
            }
            if (files_by_field.find(contract->id) == files_by_field.end()) {
                report.missing_required.push_back(contract->id);
            }
        }

        for (const tmv::FieldContract* contract : exported_contracts) {
            if (files_by_field.find(contract->id) == files_by_field.end()) {
                if (contract->requirement == tmv::FieldRequirementTier::RequiredNow) {
                    if (std::find(report.missing_required.begin(),
                                  report.missing_required.end(),
                                  contract->id) == report.missing_required.end()) {
                        report.missing_required.push_back(contract->id);
                    }
                } else {
                    report.missing_exported.push_back(contract->id);
                }
            }
        }

        std::set<int> expected_theta_indices;
        const auto theta_it = files_by_field.find("theta");
        if (theta_it != files_by_field.end()) {
            for (const auto& slice : theta_it->second) {
                expected_theta_indices.insert(slice.first);
            }
        } else if (!all_theta_indices.empty()) {
            expected_theta_indices = all_theta_indices;
        }

        report.known_not_implemented.reserve(not_implemented_contracts.size());
        for (const tmv::FieldContract* contract : not_implemented_contracts) {
            report.known_not_implemented.push_back(contract->id);
            if (contract->requirement == tmv::FieldRequirementTier::RequiredNow) {
                report.missing_not_implemented.push_back(contract->id);
            }
        }
        const std::size_t known_not_implemented_required_now =
            not_implemented_required_now_inventory.size();
        const std::size_t known_not_implemented_backlog =
            not_implemented_backlog_inventory.size();

        for (const auto& kv : files_by_field) {
            const tmv::FieldContract* contract = tmv::find_field_contract(kv.first);
            if (contract == nullptr) {
                continue;
            }
            if (contract->status != tmv::FieldImplementationStatus::ExportedNow) {
                continue;
            }

            std::vector<float> values;
            bool load_failed = false;
            std::string load_error;

            for (const auto& slice : kv.second) {
                oglcpp::NpyArray2D npy;
                std::string error;
                if (!oglcpp::load_npy_float32_2d(slice.second, npy, error)) {
                    load_failed = true;
                    load_error = "Failed to load " + slice.second.string() + ": " + error;
                    break;
                }
                values.insert(values.end(), npy.data.begin(), npy.data.end());
            }

            tmv::FieldValidationReport field_report;
            field_report.field_id = contract->id;
            field_report.status = tmv::to_string(contract->status);
            field_report.requirement = tmv::to_string(contract->requirement);

            if (load_failed) {
                tmv::FieldViolation violation;
                violation.field_id = contract->id;
                violation.reason = load_error;
                violation.count = 1;
                violation.critical = true;
                field_report.result.violations.push_back(std::move(violation));
                field_report.result.failed = true;
                report.failed = true;
                report.fields.push_back(std::move(field_report));
                continue;
            }

            if (!values.empty()) {
                field_report.result = tmv::validate_buffer_inplace(
                    values.data(),
                    values.size(),
                    *contract,
                    policy,
                    true);
                if (field_report.result.failed) {
                    report.failed = true;
                }
            }

            std::set<int> field_theta_indices;
            bool duplicate_theta_index = false;
            int prev_theta = std::numeric_limits<int>::min();
            for (const auto& slice : kv.second) {
                field_theta_indices.insert(slice.first);
                if (slice.first == prev_theta) {
                    duplicate_theta_index = true;
                }
                prev_theta = slice.first;
            }

            if (duplicate_theta_index) {
                tmv::FieldViolation violation;
                violation.field_id = contract->id;
                violation.reason = "duplicate_theta_index";
                violation.count = 1;
                violation.critical = (policy.mode == tmv::GuardMode::Strict);
                field_report.result.violations.push_back(std::move(violation));
                if (policy.mode == tmv::GuardMode::Strict) {
                    field_report.result.failed = true;
                    report.failed = true;
                }
            }

            if (!expected_theta_indices.empty() && field_theta_indices != expected_theta_indices) {
                std::size_t missing_theta = 0;
                for (int theta_idx : expected_theta_indices) {
                    if (field_theta_indices.find(theta_idx) == field_theta_indices.end()) {
                        ++missing_theta;
                    }
                }
                std::size_t extra_theta = 0;
                for (int theta_idx : field_theta_indices) {
                    if (expected_theta_indices.find(theta_idx) == expected_theta_indices.end()) {
                        ++extra_theta;
                    }
                }
                tmv::FieldViolation violation;
                violation.field_id = contract->id;
                violation.reason = "incomplete_theta_coverage";
                violation.count = missing_theta + extra_theta;
                violation.critical = (policy.mode == tmv::GuardMode::Strict);
                field_report.result.violations.push_back(std::move(violation));
                if (policy.mode == tmv::GuardMode::Strict) {
                    field_report.result.failed = true;
                    report.failed = true;
                }
            }

            report.fields.push_back(std::move(field_report));
        }

        if (!report.missing_required.empty() && policy.mode == tmv::GuardMode::Strict) {
            report.failed = true;
        }
        if (!report.missing_exported.empty() &&
            policy.mode == tmv::GuardMode::Strict &&
            policy.strict_scope == tmv::StrictGuardScope::ExportedNow) {
            report.failed = true;
        }

        if (report.failed) {
            overall_failed = true;
        }

        std::size_t nonfinite_count = 0;
        std::size_t bounds_count = 0;
        for (const auto& field : report.fields) {
            nonfinite_count += field.result.stats.nan_count + field.result.stats.inf_count;
            bounds_count += field.result.stats.below_min_count + field.result.stats.above_max_count;
        }

        std::cout << "[field-validator] " << report.context
                  << " nonfinite=" << nonfinite_count
                  << " bounds=" << bounds_count
                  << " missing_required=" << report.missing_required.size()
                  << " missing_exported=" << report.missing_exported.size()
                  << " missing_not_implemented=" << report.missing_not_implemented.size()
                  << " known_not_implemented=" << report.known_not_implemented.size()
                  << " known_not_implemented_required_now=" << known_not_implemented_required_now
                  << " known_not_implemented_backlog=" << known_not_implemented_backlog
                  << " scope=" << options.scope
                  << " failed=" << (report.failed ? "yes" : "no")
                  << "\n";

        step_reports.push_back(std::move(report));
    }

    if (!options.json_path.empty()) {
        std::filesystem::path out_path(options.json_path);
        std::error_code ec;
        if (!out_path.parent_path().empty()) {
            std::filesystem::create_directories(out_path.parent_path(), ec);
            if (ec) {
                std::cerr << "Failed creating JSON output directory: " << ec.message() << "\n";
                return 1;
            }
        }

        std::ofstream out(options.json_path);
        if (!out) {
            std::cerr << "Failed opening JSON output path: " << options.json_path << "\n";
            return 1;
        }

        out << "{\n";
        out << "  \"input_dir\": \"" << tmv::strutil::json_escape(options.input_dir) << "\",\n";
        out << "  \"contract\": \"cm1\",\n";
        out << "  \"mode\": \"" << tmv::strutil::json_escape(options.mode) << "\",\n";
        out << "  \"scope\": \"" << tmv::strutil::json_escape(options.scope) << "\",\n";
        out << "  \"failed\": " << (overall_failed ? "true" : "false") << ",\n";
        out << "  \"reports\": [\n";
        for (std::size_t i = 0; i < step_reports.size(); ++i) {
            out << tmv::validation_report_to_json(step_reports[i]);
            if (i + 1 < step_reports.size()) {
                out << ",";
            }
            out << "\n";
        }
        out << "  ]\n";
        out << "}\n";

        if (!out.good()) {
            std::cerr << "Failed writing JSON output path: " << options.json_path << "\n";
            return 1;
        }
    }

    return overall_failed ? 1 : 0;
}
