#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

namespace tmv::runtime
{

/**
 * @brief Per-step decision for scientific reductions and derived fields.
 *
 * Scientific diagnostics are conservation/budget work selected by the user.
 * Derived diagnostics are fields such as reflectivity and vorticity that may
 * additionally be required by an output or live-view publication deadline.
 */
struct DiagnosticStepPlan
{
    bool scientific = false;
    bool derived = false;
};

/**
 * @brief Returns whether a positive one-based step reaches a cadence.
 *
 * A cadence of zero is explicitly off. Negative values are reserved for
 * compatibility policy and therefore never count as a numeric cadence.
 */
constexpr bool step_cadence_due(std::uint64_t one_based_step,
                                int every_steps) noexcept
{
    return one_based_step > 0 && every_steps > 0
        && one_based_step % static_cast<std::uint64_t>(every_steps) == 0;
}

/** Decide whether a completed state is a live-SHM publication consumer. */
inline bool live_shm_publish_due(bool enabled,
                                 bool full_rate,
                                 double completed_time_s,
                                 double next_publish_time_s) noexcept
{
    if (!enabled || !std::isfinite(completed_time_s))
        return false;
    return full_rate ||
        (std::isfinite(next_publish_time_s) && next_publish_time_s >= 0.0 &&
         completed_time_s + 1.0e-9 >= next_publish_time_s);
}

/** Return the first cadence boundary strictly after a completed state. */
inline double next_live_shm_publish_time(double completed_time_s,
                                         double interval_s) noexcept
{
    if (!std::isfinite(completed_time_s) || !std::isfinite(interval_s) ||
        interval_s <= 0.0)
        return -1.0;
    return (std::floor((completed_time_s + 1.0e-9) / interval_s) + 1.0) *
        interval_s;
}

/**
 * @brief Adjusts only the final integration step to land on an exact deadline.
 *
 * Repeated floating-point additions can leave a mathematically exact landing
 * a few accumulated ulps short of its deadline. The allowance grows with the
 * number of additions but is capped at one quarter of the selected step, so it
 * cannot turn a materially incomplete step into a deadline hit.
 */
inline double clamp_step_to_deadline(double current_time_s,
                                     double selected_step_s,
                                     double deadline_s,
                                     std::uint64_t completed_steps) noexcept
{
    if (!std::isfinite(current_time_s) || !std::isfinite(selected_step_s)
        || !std::isfinite(deadline_s) || selected_step_s <= 0.0)
    {
        return selected_step_s;
    }
    const double remaining_s = deadline_s - current_time_s;
    if (remaining_s <= 0.0)
    {
        return selected_step_s;
    }
    const double scale_s = std::max({
        1.0, std::abs(current_time_s), std::abs(deadline_s)});
    const double accumulated_roundoff_s =
        8.0 * std::numeric_limits<double>::epsilon() * scale_s
        * (static_cast<double>(completed_steps) + 1.0);
    const double allowance_s = std::min(
        0.25 * selected_step_s, accumulated_roundoff_s);
    return remaining_s <= selected_step_s + allowance_s
        ? remaining_s
        : selected_step_s;
}

/**
 * @brief Builds the independent science/derived diagnostic plan for a step.
 *
 * @param one_based_step Current step number, starting at one.
 * @param scientific_every_steps -1 for compatibility policy, 0 for off, or a
 *        positive step cadence.
 * @param compatibility_scientific_due Historical output/live/debug-derived
 *        decision, consulted only when the cadence is omitted (-1).
 * @param derived_required Whether output/live publication needs fresh derived
 *        fields on this step.
 */
constexpr DiagnosticStepPlan make_diagnostic_step_plan(
    std::uint64_t one_based_step,
    int scientific_every_steps,
    bool compatibility_scientific_due,
    bool derived_required) noexcept
{
    const bool compatibility_mode = scientific_every_steps < 0;
    const bool scientific = compatibility_mode
        ? compatibility_scientific_due
        : step_cadence_due(one_based_step, scientific_every_steps);
    // Explicit v1 diagnostic cadence controls reductions/budgets only. Keep
    // the historical science->derived coupling solely for an omitted cadence;
    // output/live/trajectory consumers always remain authoritative.
    const bool derived = derived_required
        || (compatibility_mode && scientific);
    return {scientific, derived};
}

/**
 * @brief Whether a trajectory step needs the full gridded vorticity product.
 *
 * Cartesian C-grid and collocated A-grid trajectories compute their sampled
 * vorticity/budget online from prognostic fields. They need the gridded diagnostic only once when an
 * automatic seed-centre search is due. Unsupported grids use the gridded
 * fallback for both seed and subsequent samples.
 */
constexpr bool trajectory_requires_gridded_diagnostics(
    bool enabled,
    bool seeded,
    bool seed_due,
    bool automatic_center,
    bool online_sampling_supported) noexcept
{
    if (!enabled)
    {
        return false;
    }
    if (!online_sampling_supported)
    {
        return seeded || seed_due;
    }
    return !seeded && seed_due && automatic_center;
}

}  // namespace tmv::runtime
