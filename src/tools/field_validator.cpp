// Standalone validation CLI over published Nimbus fields.

#include "diagnostics/field_contract.hpp"
#include "diagnostics/field_validation.hpp"
#include "util/string_utils.hpp"
#include "core/io/npy_reader.hpp"

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

using StepPath = std::pair<int, std::filesystem::path>;
using FieldSlice = std::pair<int, std::filesystem::path>;
using FieldSliceMap = std::map<std::string, std::vector<FieldSlice>>;

struct ContractInventory {
    std::vector<const tmv::FieldContract*> required_contracts;
    std::vector<const tmv::FieldContract*> exported_contracts;
    std::vector<const tmv::FieldContract*> not_implemented_contracts;
    std::size_t known_not_implemented_required_now = 0;
    std::size_t known_not_implemented_backlog = 0;
};

struct StepFileInventory {
    FieldSliceMap files_by_field;
    std::set<int> all_theta_indices;
};

struct ReportSummaryCounts {
    std::size_t nonfinite = 0;
    std::size_t bounds = 0;
};

bool validate_input_root(const std::filesystem::path& input_root) {
    if (!std::filesystem::exists(input_root)) {
        std::cerr << "Input directory does not exist: " << input_root.string() << "\n";
        return false;
    }
    if (!std::filesystem::is_directory(input_root)) {
        std::cerr << "Input path is not a directory: " << input_root.string() << "\n";
        return false;
    }
    return true;
}

tmv::ValidationPolicy make_validation_policy(const Options& options) {
    tmv::ValidationPolicy policy;
    policy.mode = (options.mode == "strict") ? tmv::GuardMode::Strict : tmv::GuardMode::Off;
    policy.fail_on = tmv::GuardFailOn::Both;
    policy.strict_scope = (options.scope == "exported")
        ? tmv::StrictGuardScope::ExportedNow
        : tmv::StrictGuardScope::RequiredOnly;
    return policy;
}

ContractInventory load_contract_inventory() {
    ContractInventory inventory;
    inventory.required_contracts =
        tmv::contracts_with_requirement(tmv::FieldRequirementTier::RequiredNow);
    inventory.exported_contracts =
        tmv::contracts_with_status(tmv::FieldImplementationStatus::ExportedNow);
    inventory.not_implemented_contracts =
        tmv::contracts_with_status(tmv::FieldImplementationStatus::NotImplemented);
    inventory.known_not_implemented_required_now =
        tmv::known_not_implemented_required_now_inventory().size();
    inventory.known_not_implemented_backlog =
        tmv::known_not_implemented_backlog_inventory().size();
    return inventory;
}

std::vector<StepPath> discover_step_directories(const std::filesystem::path& input_root,
                                                const std::regex& step_re) {
    std::vector<StepPath> steps;
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
    return steps;
}

StepFileInventory scan_step_files(const std::filesystem::path& step_path,
                                  const std::regex& file_re) {
    StepFileInventory inventory;
    for (const auto& file_entry : std::filesystem::directory_iterator(step_path)) {
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
        inventory.files_by_field[canonical_field].push_back({theta_idx, file_entry.path()});
        inventory.all_theta_indices.insert(theta_idx);
    }

    for (auto& kv : inventory.files_by_field) {
        std::sort(kv.second.begin(), kv.second.end(), [](const auto& a, const auto& b) {
            return a.first < b.first;
        });
    }

    return inventory;
}

tmv::ValidationReport make_step_report(const StepPath& step,
                                       const tmv::ValidationPolicy& policy) {
    tmv::ValidationReport report;
    report.context = step.second.filename().string();
    report.step_index = step.first;
    report.guard_mode = tmv::to_string(policy.mode);
    report.guard_fail_on = tmv::to_string(policy.fail_on);
    report.guard_scope = tmv::to_string(policy.strict_scope);
    return report;
}

void add_missing_contract_reports(tmv::ValidationReport& report,
                                  const ContractInventory& contracts,
                                  const FieldSliceMap& files_by_field) {
    for (const tmv::FieldContract* contract : contracts.required_contracts) {
        if (contract->status != tmv::FieldImplementationStatus::ExportedNow) {
            continue;
        }
        if (files_by_field.find(contract->id) == files_by_field.end()) {
            report.missing_required.push_back(contract->id);
        }
    }

    for (const tmv::FieldContract* contract : contracts.exported_contracts) {
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
}

std::set<int> infer_expected_theta_indices(const StepFileInventory& inventory) {
    std::set<int> expected_theta_indices;
    const auto theta_it = inventory.files_by_field.find("theta");
    if (theta_it != inventory.files_by_field.end()) {
        for (const auto& slice : theta_it->second) {
            expected_theta_indices.insert(slice.first);
        }
    } else if (!inventory.all_theta_indices.empty()) {
        expected_theta_indices = inventory.all_theta_indices;
    }
    return expected_theta_indices;
}

void add_known_not_implemented(tmv::ValidationReport& report,
                               const ContractInventory& contracts) {
    report.known_not_implemented.reserve(contracts.not_implemented_contracts.size());
    for (const tmv::FieldContract* contract : contracts.not_implemented_contracts) {
        report.known_not_implemented.push_back(contract->id);
        if (contract->requirement == tmv::FieldRequirementTier::RequiredNow) {
            report.missing_not_implemented.push_back(contract->id);
        }
    }
}

bool load_field_values(const std::vector<FieldSlice>& slices,
                       std::vector<float>& values,
                       std::string& load_error) {
    for (const auto& slice : slices) {
        oglcpp::NpyArray2D npy;
        std::string error;
        if (!oglcpp::load_npy_float32_2d(slice.second, npy, error)) {
            load_error = "Failed to load " + slice.second.string() + ": " + error;
            return false;
        }
        values.insert(values.end(), npy.data.begin(), npy.data.end());
    }
    return true;
}

tmv::FieldValidationReport make_field_report(const tmv::FieldContract& contract) {
    tmv::FieldValidationReport field_report;
    field_report.field_id = contract.id;
    field_report.status = tmv::to_string(contract.status);
    field_report.requirement = tmv::to_string(contract.requirement);
    return field_report;
}

void record_load_failure(tmv::ValidationReport& report,
                         tmv::FieldValidationReport& field_report,
                         const tmv::FieldContract& contract,
                         const std::string& load_error) {
    tmv::FieldViolation violation;
    violation.field_id = contract.id;
    violation.reason = load_error;
    violation.count = 1;
    violation.critical = true;
    field_report.result.violations.push_back(std::move(violation));
    field_report.result.failed = true;
    report.failed = true;
}

void validate_theta_coverage(tmv::ValidationReport& report,
                             tmv::FieldValidationReport& field_report,
                             const std::vector<FieldSlice>& slices,
                             const std::set<int>& expected_theta_indices,
                             const tmv::ValidationPolicy& policy) {
    std::set<int> field_theta_indices;
    bool duplicate_theta_index = false;
    int prev_theta = std::numeric_limits<int>::min();
    for (const auto& slice : slices) {
        field_theta_indices.insert(slice.first);
        if (slice.first == prev_theta) {
            duplicate_theta_index = true;
        }
        prev_theta = slice.first;
    }

    if (duplicate_theta_index) {
        tmv::FieldViolation violation;
        violation.field_id = field_report.field_id;
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
        violation.field_id = field_report.field_id;
        violation.reason = "incomplete_theta_coverage";
        violation.count = missing_theta + extra_theta;
        violation.critical = (policy.mode == tmv::GuardMode::Strict);
        field_report.result.violations.push_back(std::move(violation));
        if (policy.mode == tmv::GuardMode::Strict) {
            field_report.result.failed = true;
            report.failed = true;
        }
    }
}

void validate_exported_fields(tmv::ValidationReport& report,
                              const FieldSliceMap& files_by_field,
                              const std::set<int>& expected_theta_indices,
                              const tmv::ValidationPolicy& policy) {
    for (const auto& kv : files_by_field) {
        const tmv::FieldContract* contract = tmv::find_field_contract(kv.first);
        if (contract == nullptr) {
            continue;
        }
        if (contract->status != tmv::FieldImplementationStatus::ExportedNow) {
            continue;
        }

        tmv::FieldValidationReport field_report = make_field_report(*contract);

        std::vector<float> values;
        std::string load_error;
        if (!load_field_values(kv.second, values, load_error)) {
            record_load_failure(report, field_report, *contract, load_error);
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

        validate_theta_coverage(report, field_report, kv.second, expected_theta_indices, policy);
        report.fields.push_back(std::move(field_report));
    }
}

void apply_strict_missing_field_failures(tmv::ValidationReport& report,
                                         const tmv::ValidationPolicy& policy) {
    if (!report.missing_required.empty() && policy.mode == tmv::GuardMode::Strict) {
        report.failed = true;
    }
    if (!report.missing_exported.empty() &&
        policy.mode == tmv::GuardMode::Strict &&
        policy.strict_scope == tmv::StrictGuardScope::ExportedNow) {
        report.failed = true;
    }
}

ReportSummaryCounts count_report_violations(const tmv::ValidationReport& report) {
    ReportSummaryCounts counts;
    for (const auto& field : report.fields) {
        counts.nonfinite += field.result.stats.nan_count + field.result.stats.inf_count;
        counts.bounds += field.result.stats.below_min_count + field.result.stats.above_max_count;
    }
    return counts;
}

void print_step_summary(const tmv::ValidationReport& report,
                        const ContractInventory& contracts,
                        const std::string& scope) {
    const ReportSummaryCounts counts = count_report_violations(report);
    std::cout << "[field-validator] " << report.context
              << " nonfinite=" << counts.nonfinite
              << " bounds=" << counts.bounds
              << " missing_required=" << report.missing_required.size()
              << " missing_exported=" << report.missing_exported.size()
              << " missing_not_implemented=" << report.missing_not_implemented.size()
              << " known_not_implemented=" << report.known_not_implemented.size()
              << " known_not_implemented_required_now="
              << contracts.known_not_implemented_required_now
              << " known_not_implemented_backlog="
              << contracts.known_not_implemented_backlog
              << " scope=" << scope
              << " failed=" << (report.failed ? "yes" : "no")
              << "\n";
}

tmv::ValidationReport validate_step(const StepPath& step,
                                    const ContractInventory& contracts,
                                    const tmv::ValidationPolicy& policy,
                                    const std::regex& file_re) {
    const StepFileInventory inventory = scan_step_files(step.second, file_re);
    tmv::ValidationReport report = make_step_report(step, policy);

    add_missing_contract_reports(report, contracts, inventory.files_by_field);
    const std::set<int> expected_theta_indices = infer_expected_theta_indices(inventory);
    add_known_not_implemented(report, contracts);
    validate_exported_fields(report, inventory.files_by_field, expected_theta_indices, policy);
    apply_strict_missing_field_failures(report, policy);

    return report;
}

bool write_json_reports(const Options& options,
                        bool overall_failed,
                        const std::vector<tmv::ValidationReport>& step_reports) {
    if (options.json_path.empty()) {
        return true;
    }

    std::filesystem::path out_path(options.json_path);
    std::error_code ec;
    if (!out_path.parent_path().empty()) {
        std::filesystem::create_directories(out_path.parent_path(), ec);
        if (ec) {
            std::cerr << "Failed creating JSON output directory: " << ec.message() << "\n";
            return false;
        }
    }

    std::ofstream out(options.json_path);
    if (!out) {
        std::cerr << "Failed opening JSON output path: " << options.json_path << "\n";
        return false;
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
        return false;
    }
    return true;
}

int run_field_validator(const Options& options) {
    const std::filesystem::path input_root(options.input_dir);
    if (!validate_input_root(input_root)) {
        return 1;
    }

    const tmv::ValidationPolicy policy = make_validation_policy(options);
    const ContractInventory contracts = load_contract_inventory();
    const std::regex step_re("^step_([0-9]+)$");
    const std::regex file_re("^th([0-9]+)_(.+)\\.npy$");

    const std::vector<StepPath> steps = discover_step_directories(input_root, step_re);
    if (steps.empty()) {
        std::cerr << "No step_XXXXXX directories found in " << input_root.string() << "\n";
        return 1;
    }

    std::vector<tmv::ValidationReport> step_reports;
    bool overall_failed = false;
    for (const auto& step : steps) {
        tmv::ValidationReport report = validate_step(step, contracts, policy, file_re);
        if (report.failed) {
            overall_failed = true;
        }
        print_step_summary(report, contracts, options.scope);
        step_reports.push_back(std::move(report));
    }

    if (!write_json_reports(options, overall_failed, step_reports)) {
        return 1;
    }

    return overall_failed ? 1 : 0;
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

    return run_field_validator(options);
}
