// Runtime CPU/GPU parity measurement and reporting.
#include "compute/parity_check.hpp"

#include "compute/compute_backend.hpp"
#include "util/log.hpp"

#include <charconv>
#include <cstdlib>
#include <iomanip>
#include <mutex>
#include <sstream>
#include <string_view>
#include <tuple>

namespace nimbus::parity {

namespace {

bool env_flag_enabled(ParityRuntimeState& s)
{
    if (s.env_enabled < 0)
    {
        const char* env = std::getenv("NIMBUS_VALIDATE_PARITY");
        s.env_enabled = (env != nullptr && env[0] == '1') ? 1 : 0;
    }
    return s.env_enabled == 1;
}

int sample_every(ParityRuntimeState& s)
{
    if (s.env_every < 0)
    {
        s.env_every = 1;
        if (const char* env = std::getenv("NIMBUS_PARITY_EVERY"))
        {
            const std::string_view text(env);
            int parsed = 0;
            const auto result = std::from_chars(
                text.data(), text.data() + text.size(), parsed);
            if (result.ec == std::errc{}
                && result.ptr == text.data() + text.size()
                && parsed >= 1)
            {
                s.env_every = parsed;
            }
        }
    }
    return s.env_every;
}

}  // namespace

bool enabled()
{
    return enabled(global_parity_runtime_state,
                   global_compute_backend_config.validate_parity);
}

bool enabled(ParityRuntimeState& runtime, bool config_enabled)
{
    std::lock_guard<std::mutex> lock(runtime.mutex);
    return config_enabled || env_flag_enabled(runtime);
}

bool sample(const char* kernel)
{
    return sample(global_parity_runtime_state,
                  global_compute_backend_config.validate_parity, kernel);
}

bool sample(ParityRuntimeState& runtime, bool config_enabled,
            const char* kernel)
{
    std::lock_guard<std::mutex> lock(runtime.mutex);
    if (kernel == nullptr || kernel[0] == '\0'
        || !(config_enabled || env_flag_enabled(runtime)))
    {
        return false;
    }
    const std::uint64_t count = runtime.call_counters[kernel]++;
    return (count % static_cast<std::uint64_t>(sample_every(runtime))) == 0;
}

bool check_field(const char* kernel, const char* field,
                 const float* cpu, const float* gpu,
                 int nr, int nth, int nz, Tier tier,
                 ComparisonDomain domain, int horizon)
{
    return check_field(global_parity_runtime_state, kernel, field, cpu, gpu,
                       nr, nth, nz, tier, domain, horizon);
}

bool check_field(ParityRuntimeState& runtime,
                 const char* kernel, const char* field,
                 const float* cpu, const float* gpu,
                 int nr, int nth, int nz, Tier tier,
                 ComparisonDomain domain, int horizon)
{
    Tolerance tol = tolerance_for(tier);
    const double horizon_scale =
        static_cast<double>(horizon < 1 ? 1 : horizon);
    tol.abs *= horizon_scale;
    tol.rel *= horizon_scale;
    FieldComparison comparison =
        compare_field(cpu, gpu, nr, nth, nz, tol, domain);
    const char* kernel_name =
        kernel != nullptr && kernel[0] != '\0' ? kernel : "<invalid-kernel>";
    const char* field_name =
        field != nullptr && field[0] != '\0' ? field : "<invalid-field>";
    const char* identifier_error = nullptr;
    if (kernel == nullptr) identifier_error = "null_kernel";
    else if (kernel[0] == '\0') identifier_error = "empty_kernel";
    else if (field == nullptr) identifier_error = "null_field";
    else if (field[0] == '\0') identifier_error = "empty_field";
    if (identifier_error != nullptr)
    {
        comparison.invalid_reason = identifier_error;
    }

    // Branch-discontinuous physics allowance: microphysics switches whole
    // process sets on knife-edge thresholds (the T0 = 273.15 K freezing
    // branch), so a cell whose fp64-CPU temperature lands on one side and
    // whose fp32-GPU temperature lands on the other legitimately differs
    // by a full process amount. Allow <= 0.01% of compared cells per
    // sample for the microphysics tiers ONLY; a real scheme bug breaches
    // percent-scale cell counts (observed: a wrong base-state subtraction
    // hit 2.6-26%), three orders above this allowance.
    std::uint64_t allowed = 0;
    if (tier == Tier::Microphysics || tier == Tier::MicrophysicsHeating)
    {
        allowed = comparison.cells_compared / 10000;
    }
    const std::uint64_t breached =
        comparison.tolerance_breaches + comparison.nonfinite_cells;
    const bool empty_domain = comparison.cells_compared == 0;
    // The microphysics knife-edge allowance applies only to finite numerical
    // divergence. It must never hide non-finite state, and a comparison that
    // visited no cells is invalid rather than a vacuous pass.
    const bool check_failed =
        comparison.invalid_reason != nullptr
        || comparison.nonfinite_cells > 0
        || comparison.tolerance_breaches > allowed;

    std::lock_guard<std::mutex> lock(runtime.mutex);
    ParityFieldStats& fs = runtime.stats[{kernel_name, field_name, domain}];
    const bool first_failure_for_contract =
        fs.failed_samples == 0 && check_failed;
    fs.tier = tier;
    fs.domain = domain;
    fs.samples += 1;
    fs.failed_samples += check_failed ? 1 : 0;
    fs.empty_samples += empty_domain ? 1 : 0;
    fs.cells_compared += comparison.cells_compared;
    fs.breach_cells += breached;
    fs.nonfinite_cells += comparison.nonfinite_cells;
    if (comparison.worst_i >= 0
        && (fs.worst_i < 0 || comparison.max_abs > fs.worst_abs))
    {
        fs.worst_abs = comparison.max_abs;
        fs.worst_rel = comparison.max_rel;
        fs.worst_i = comparison.worst_i;
        fs.worst_j = comparison.worst_j;
        fs.worst_k = comparison.worst_k;
        fs.worst_cpu = comparison.worst_cpu;
        fs.worst_gpu = comparison.worst_gpu;
    }
    runtime.total_samples += 1;
    if (check_failed)
    {
        runtime.any_breach = true;
        if (first_failure_for_contract)
        {
            std::ostringstream msg;
            msg << "[PARITY] BREACH kernel=" << kernel_name
                << " field=" << field_name
                << " tier=" << tier_name(tier)
                << " domain=" << domain_name(domain);
            if (comparison.invalid_reason != nullptr)
            {
                msg << " reason=" << comparison.invalid_reason
                    << " dims=(" << nr << "," << nth << "," << nz << ")";
            }
            else
            {
                msg << " worst_abs=" << std::scientific
                    << std::setprecision(3) << comparison.max_abs
                    << " worst_rel=" << comparison.max_rel
                    << " tol(abs=" << tol.abs << ",rel=" << tol.rel << ")"
                    << " at (" << comparison.worst_i << ","
                    << comparison.worst_j << "," << comparison.worst_k << ")"
                    << " cpu=" << comparison.worst_cpu
                    << " gpu=" << comparison.worst_gpu
                    << " breach_cells=" << breached << "/"
                    << comparison.cells_compared
                    << " nonfinite_cells=" << comparison.nonfinite_cells;
            }
            tmv::log_warn(msg.str());
        }
    }
    return !check_failed;
}

bool any_breach()
{
    return any_breach(global_parity_runtime_state);
}

bool any_breach(const ParityRuntimeState& runtime)
{
    std::lock_guard<std::mutex> lock(runtime.mutex);
    return runtime.any_breach;
}

std::uint64_t total_samples()
{
    return total_samples(global_parity_runtime_state);
}

std::uint64_t total_samples(const ParityRuntimeState& runtime)
{
    std::lock_guard<std::mutex> lock(runtime.mutex);
    return runtime.total_samples;
}

void print_report()
{
    print_report(global_parity_runtime_state);
}

void print_report(const ParityRuntimeState& runtime)
{
    std::lock_guard<std::mutex> lock(runtime.mutex);
    if (runtime.stats.empty())
    {
        return;
    }

    tmv::log_info("[PARITY] CPU-vs-GPU divergence report "
                  "(tolerance: |gpu-cpu| <= abs + rel*|cpu|, declared domains)");

    {
        std::ostringstream line;
        line << "[PARITY] "
             << std::left << std::setw(36) << "kernel"
             << std::setw(12) << "field"
             << std::setw(11) << "tier"
             << std::setw(13) << "domain"
             << std::right << std::setw(8) << "samples"
             << std::setw(9) << "failed"
             << std::setw(13) << "breach_cells"
             << std::setw(12) << "nonfinite"
             << std::setw(8) << "empty"
             << std::setw(12) << "worst_abs"
             << std::setw(12) << "worst_rel"
             << "  worst_at (cpu vs gpu)";
        tmv::log_info(line.str());
    }

    std::size_t breached_pairs = 0;
    std::size_t failed_pairs = 0;
    for (const auto& [key, fs] : runtime.stats)
    {
        if (fs.breach_cells > 0) ++breached_pairs;
        if (fs.failed_samples > 0) ++failed_pairs;
        std::ostringstream at;
        at << "(" << fs.worst_i << "," << fs.worst_j << "," << fs.worst_k
           << ") (" << std::scientific << std::setprecision(3) << fs.worst_cpu
           << " vs " << fs.worst_gpu << ")";
        std::ostringstream line;
        line << "[PARITY] "
             << std::left << std::setw(36) << std::get<0>(key)
             << std::setw(12) << std::get<1>(key)
             << std::setw(11) << tier_name(fs.tier)
             << std::setw(13) << domain_name(fs.domain)
             << std::right << std::setw(8) << fs.samples
             << std::setw(9) << fs.failed_samples
             << std::setw(13) << fs.breach_cells
             << std::setw(12) << fs.nonfinite_cells
             << std::setw(8) << fs.empty_samples
             << std::scientific << std::setprecision(3)
             << std::setw(12) << fs.worst_abs
             << std::setw(12) << fs.worst_rel
             << "  " << at.str();
        tmv::log_info(line.str());
    }

    tmv::log_info("[PARITY] RESULT: ", runtime.any_breach ? "FAIL" : "PASS",
                  " (", runtime.stats.size(), " kernel-field-domain records, ",
                  runtime.total_samples, " samples, ",
                  failed_pairs, " failed records",
                  (!runtime.any_breach && breached_pairs > 0
                       ? ", all within the knife-edge allowance" : ""),
                  ")");
}

void reset()
{
    reset(global_parity_runtime_state);
}

void reset(ParityRuntimeState& runtime)
{
    std::lock_guard<std::mutex> lock(runtime.mutex);
    runtime.stats.clear();
    runtime.call_counters.clear();
    runtime.total_samples = 0;
    runtime.any_breach = false;
    runtime.env_enabled = -1;
    runtime.env_every = -1;
}

}  // namespace nimbus::parity
