#pragma once

#include <cmath>
#include <limits>

namespace tmv::diagnostics
{
struct ColumnBracket
{
    int lower = 0;
    int upper = 0;
    double fraction = 0.0;
    bool valid = false;

    template <class ValueAt>
    double sample(const ValueAt& value_at) const
    {
        if (!valid) return std::numeric_limits<double>::quiet_NaN();
        const double a = value_at(lower), b = value_at(upper);
        return a + fraction * (b - a);
    }
};

// Coordinates must increase: use physical height or -log(pressure).
// Only near-surface height sampling permits endpoint extrapolation.
template <class CoordinateAt>
ColumnBracket bracket_column(int begin, int end, double target,
                             const CoordinateAt& coordinate_at, bool clamp = false)
{
    if (begin < 0 || end <= begin || !std::isfinite(target)) return {};
    const double first = coordinate_at(begin);
    if (!std::isfinite(first)) return {};
    ColumnBracket result;
    if (target == first || (clamp && target < first))
        result = {begin, begin, 0.0, true};
    double previous = first;
    for (int k = begin + 1; k < end; ++k)
    {
        const double current = coordinate_at(k);
        if (!std::isfinite(current) || current <= previous) return {};
        if (target > previous && target <= current)
            result = {k - 1, k, (target - previous) / (current - previous), true};
        previous = current;
    }
    if (clamp && target > previous) return {end - 1, end - 1, 0.0, true};
    return result;
}
}
