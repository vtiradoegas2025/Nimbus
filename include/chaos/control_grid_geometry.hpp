#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace tmv::chaos
{
inline double earth_control_spacing(double spacing, double correlation_length)
{
    // Preserve the established cubic-B-spline calibration of Lx/Ly.
    const double result = std::max(0.25 * spacing, correlation_length * (1.0 / 0.844));
    if (!std::isfinite(spacing) || spacing <= 0.0 ||
        !std::isfinite(correlation_length) || correlation_length <= 0.0 || !std::isfinite(result))
        throw std::invalid_argument("invalid Earth-control spacing");
    return result;
}

inline std::uint64_t stochastic_snapshot_bytes(int nr, int nth, int nz,
    double dx, double dy, double lx, double ly, int processes)
{
    if (nr <= 0 || nth <= 0 || nz <= 0 || processes < 1 || processes > 3)
        throw std::invalid_argument("invalid stochastic snapshot dimensions");
    const long double x = std::ceil((static_cast<long double>(nr) - 1) * dx /
        earth_control_spacing(dx, lx)) + 8;
    const long double y = std::ceil((static_cast<long double>(nth) - 1) * dy /
        earth_control_spacing(dy, ly)) + 8;
    const auto multiply = [](std::uint64_t a, std::uint64_t b) {
        if (b && a > std::numeric_limits<std::uint64_t>::max() / b)
            throw std::overflow_error("stochastic snapshot memory overflow");
        return a * b;
    };
    const auto add = [](std::uint64_t a, std::uint64_t b) {
        if (b > std::numeric_limits<std::uint64_t>::max() - a)
            throw std::overflow_error("stochastic snapshot memory overflow");
        return a + b;
    };
    const auto columns = multiply(static_cast<std::uint64_t>(nr), nth);
    const auto controls = multiply(static_cast<std::uint64_t>(x), static_cast<std::uint64_t>(y));
    // Current/previous controls retain growth capacity; rendered fields and
    // held PBL sources use the native grid. Padding covers cubic supports.
    const auto volumes = add(multiply(4 * processes, controls), multiply(2 * processes + 5, columns));
    return multiply(sizeof(float), add(multiply(volumes, nz), multiply(6, columns)));
}
}
