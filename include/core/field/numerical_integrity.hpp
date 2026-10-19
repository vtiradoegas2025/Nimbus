#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>

namespace tmv
{

/**
 * Monotonic count of numerical non-finite incidents in this process.
 *
 * Sanitizers can replace an invalid candidate before a later field sweep.
 * Recording at detection preserves that failed-step evidence without coupling
 * numerical kernels to a particular runtime driver.
 */
inline std::atomic<std::uint64_t> numerical_nonfinite_incident_count{0};

inline void record_numerical_nonfinite_incidents(std::size_t count = 1) noexcept
{
    if (count > 0)
    {
        numerical_nonfinite_incident_count.fetch_add(
            static_cast<std::uint64_t>(count), std::memory_order_relaxed);
    }
}

inline std::uint64_t numerical_nonfinite_incidents() noexcept
{
    return numerical_nonfinite_incident_count.load(std::memory_order_relaxed);
}

inline void reset_numerical_nonfinite_incidents() noexcept
{
    numerical_nonfinite_incident_count.store(0, std::memory_order_relaxed);
}

} // namespace tmv
