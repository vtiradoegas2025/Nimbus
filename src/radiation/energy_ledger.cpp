#include "diagnostics/level_budget_csv.hpp"
// Production radiation boundary-power to theta-application energy ledger.

#include "radiation/runtime.hpp"

#include "core/infra/physical_grid_metrics.hpp"
#include "core/runtime/simulation_state.hpp"
#include "util/log.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <mutex>
#include <stdexcept>

namespace
{
tmv::LevelBudgetCsv radiation_budget_files;
std::mutex radiation_budget_csv_mutex;

constexpr const char* radiation_budget_csv_header =
    "step,sim_time_s,last_radiation_sample_s,next_radiation_sample_s,"
    "held_atmospheric_power_w,shortwave_incoming_power_w,"
    "shortwave_atmospheric_absorbed_power_w,"
    "shortwave_surface_absorbed_power_w,shortwave_top_escape_power_w,"
    "shortwave_lateral_escape_power_w,"
    "shortwave_unresolved_reflected_power_w,"
    "shortwave_reflection_converged,shortwave_power_residual_w,"
    "shortwave_relative_power_residual,"
    "expected_atmospheric_energy_j,booked_radiation_heat_j,booking_residual_j,"
    "relative_booking_residual,cumulative_expected_energy_j,"
    "cumulative_booked_heat_j,cumulative_booking_residual_j\n";
}

namespace tmv
{

void commit_radiation_applied_heat(const SimulationStateView& state,
                                   double dt_s,
                                   double applied_radiation_heat_j)
{
    RadiationRuntimeState& runtime = state.radiation_runtime
        ? *state.radiation_runtime : tmv::require_level_runtime(state.radiation_runtime, "radiation_runtime");
    if (!std::isfinite(dt_s) || dt_s <= 0.0
        || !std::isfinite(runtime.held_atmospheric_power_w)
        || !std::isfinite(runtime.held_shortwave_incoming_power_w)
        || !std::isfinite(
            runtime.held_shortwave_atmospheric_absorbed_power_w)
        || !std::isfinite(runtime.held_surface_absorbed_power_w)
        || !std::isfinite(runtime.held_top_escape_power_w)
        || !std::isfinite(runtime.held_lateral_escape_power_w)
        || !std::isfinite(runtime.held_unresolved_reflected_power_w)
        || runtime.held_shortwave_incoming_power_w < 0.0
        || runtime.held_shortwave_atmospheric_absorbed_power_w < 0.0
        || runtime.held_surface_absorbed_power_w < 0.0
        || runtime.held_top_escape_power_w < 0.0
        || runtime.held_lateral_escape_power_w < 0.0
        || runtime.held_unresolved_reflected_power_w < 0.0
        || !runtime.held_reflection_converged
        || runtime.applied_step_count
            == std::numeric_limits<std::uint64_t>::max()
        || !std::isfinite(applied_radiation_heat_j))
    {
        throw std::invalid_argument(
            "radiation energy ledger requires finite power/heat and positive timestep");
    }

    const double shortwave_accounted_power_w =
        runtime.held_shortwave_atmospheric_absorbed_power_w
        + runtime.held_surface_absorbed_power_w
        + runtime.held_top_escape_power_w
        + runtime.held_lateral_escape_power_w
        + runtime.held_unresolved_reflected_power_w;
    const double shortwave_residual_w = shortwave_accounted_power_w
        - runtime.held_shortwave_incoming_power_w;
    const double shortwave_relative_residual =
        std::abs(shortwave_residual_w) /
        std::max(1.0, std::abs(runtime.held_shortwave_incoming_power_w));
    const double expected_energy_j = runtime.held_atmospheric_power_w * dt_s;
    const double residual_j = applied_radiation_heat_j - expected_energy_j;
    const double relative_residual = std::abs(residual_j)
        / std::max(1.0, std::abs(expected_energy_j));
    const double next_accumulated_expected =
        runtime.accumulated_expected_atmospheric_energy_j
        + expected_energy_j;
    const double next_accumulated_applied =
        runtime.accumulated_applied_radiation_heat_j
        + applied_radiation_heat_j;
    const double cumulative_residual_j =
        next_accumulated_applied - next_accumulated_expected;
    const double candidates[] = {
        shortwave_accounted_power_w,
        shortwave_residual_w,
        shortwave_relative_residual,
        expected_energy_j,
        residual_j,
        relative_residual,
        next_accumulated_expected,
        next_accumulated_applied,
        cumulative_residual_j,
    };
    if (std::any_of(
            std::begin(candidates), std::end(candidates),
            [](double value) { return !std::isfinite(value); }))
        throw std::overflow_error(
            "radiation energy ledger candidate overflowed");

    runtime.last_expected_atmospheric_energy_j = expected_energy_j;
    runtime.last_applied_radiation_heat_j = applied_radiation_heat_j;
    runtime.accumulated_expected_atmospheric_energy_j =
        next_accumulated_expected;
    runtime.accumulated_applied_radiation_heat_j =
        next_accumulated_applied;
    ++runtime.applied_step_count;

    std::lock_guard<std::mutex> lock(radiation_budget_csv_mutex);
    if (auto* selected = radiation_budget_files.stream(state.budget_level, radiation_budget_csv_mutex))
    {
        auto& radiation_budget_csv = *selected;
        radiation_budget_csv
            << runtime.applied_step_count << ',' << state.time_s << ','
            << runtime.last_update_time_s << ',' << runtime.next_update_time_s << ','
            << runtime.held_atmospheric_power_w << ','
            << runtime.held_shortwave_incoming_power_w << ','
            << runtime.held_shortwave_atmospheric_absorbed_power_w << ','
            << runtime.held_surface_absorbed_power_w << ','
            << runtime.held_top_escape_power_w << ','
            << runtime.held_lateral_escape_power_w << ','
            << runtime.held_unresolved_reflected_power_w << ','
            << (runtime.held_reflection_converged ? 1 : 0) << ','
            << shortwave_residual_w << ','
            << shortwave_relative_residual << ',' << expected_energy_j
            << ',' << applied_radiation_heat_j << ',' << residual_j << ','
            << relative_residual << ','
            << runtime.accumulated_expected_atmospheric_energy_j << ','
            << runtime.accumulated_applied_radiation_heat_j << ','
            << cumulative_residual_j << '\n';
    }
}

void open_radiation_budget_csv(const std::string& path)
{
    std::lock_guard<std::mutex> lock(radiation_budget_csv_mutex);
    if (!radiation_budget_files.open(path, radiation_budget_csv_header, 17))
    {
        tmv::log_warn("[RADIATION CSV] failed to open '", path,
                      "' for writing; coupled energy ledger output disabled");
        return;
    }
    tmv::log_info("[RADIATION CSV] writing coupled energy ledger to ", path);
}

void close_radiation_budget_csv()
{
    std::lock_guard<std::mutex> lock(radiation_budget_csv_mutex);
    radiation_budget_files.close();
}

} // namespace tmv
