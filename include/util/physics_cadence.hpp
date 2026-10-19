#pragma once

#include <cmath>
#include <stdexcept>
#include <string>

namespace tmv
{
inline constexpr double physics_clock_tolerance_s = 1.0e-12;

inline bool valid_physics_cadence(double interval_s) noexcept
{
    return std::isfinite(interval_s) && interval_s > physics_clock_tolerance_s;
}

inline void require_physics_cadence(double interval_s, const char* label)
{
    if (!valid_physics_cadence(interval_s))
        throw std::invalid_argument(std::string(label)
            + " must be finite and greater than the 1e-12 s scheduling tolerance");
}

inline double next_physics_deadline(double previous_next_s, double current_s,
                                   double interval_s, double tolerance_s)
{
    require_physics_cadence(interval_s, "physics cadence");
    const double threshold = current_s + tolerance_s;
    if (!std::isfinite(previous_next_s) || previous_next_s < 0.0 ||
        !std::isfinite(current_s) || current_s < 0.0 ||
        !std::isfinite(tolerance_s) || tolerance_s < 0.0 || !std::isfinite(threshold) ||
        current_s + interval_s <= current_s)
        throw std::invalid_argument("physics cadence cannot advance the current clock");
    if (previous_next_s > threshold) return previous_next_s;

    double candidate = previous_next_s + interval_s;
    if (candidate <= threshold)
    {
        const double count = std::floor((threshold - previous_next_s) / interval_s) + 1.0;
        if (!std::isfinite(count) || count >= 0x1p53)
            throw std::overflow_error("physics cadence exceeds exact deadline-count precision");
        candidate = previous_next_s + count * interval_s;
        if (candidate <= threshold) candidate += interval_s;
    }
    if (!std::isfinite(candidate) || candidate <= threshold || candidate <= previous_next_s)
        throw std::overflow_error("physics cadence cannot publish a later finite deadline");
    return candidate;
}
}
