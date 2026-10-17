#pragma once

#include "microphysics/base/thermodynamics.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

namespace tmv::milbrandt_detail
{
inline bool finalize_cloud_storage(float                 initial_cloud,
                                   float                 warm_cloud,
                                   float                 initial_vapor,
                                   float                 density,
                                   float                 initial_number,
                                   double                dt,
                                   std::array<float, 4>& rates) noexcept
{
    if (!std::isfinite(initial_cloud) || initial_cloud < 0 || !std::isfinite(warm_cloud) || warm_cloud < 0 ||
        !std::isfinite(initial_vapor) || initial_vapor < 0 || !std::isfinite(density) || density <= 0 ||
        !std::isfinite(initial_number) || initial_number < 0 || !std::isfinite(dt) || dt <= 0 ||
        !std::all_of(rates.begin(), rates.end(), [](float value) { return std::isfinite(value); }))
        return false;
    const double cloud  = initial_cloud + dt * rates[0];
    const double number = initial_number + dt * rates[3];
    if (!std::isfinite(cloud) || !std::isfinite(number))
        return false;
    if (cloud >= 0 && (density * cloud <= 1e-12 || number > 0))
        return true;
    const double scale = std::max({double(initial_cloud), double(warm_cloud), std::abs(dt * rates[0])});
    if (std::abs(cloud) > 8 * std::numeric_limits<float>::epsilon() * scale)
        return false;

    // Complete phase removal rounds mass and number independently. Close an
    // exactly representable empty-cloud endpoint, or retain a representable
    // population with the remaining liquid.
    auto candidate = rates;
    candidate[0]   = static_cast<float>(-double(initial_cloud) / dt);
    if (initial_cloud + dt * candidate[0] < 0)
        candidate[0] = std::nextafter(candidate[0], std::numeric_limits<float>::infinity());
    if (density * (initial_cloud + dt * candidate[0]) > 1e-12 && number <= 0)
    {
        if (initial_number <= 0)
            return false;
        candidate[3] = static_cast<float>(-double(initial_number) / dt);
        if (initial_number + dt * candidate[3] <= 0)
            candidate[3] = std::nextafter(candidate[3], std::numeric_limits<float>::infinity());
        const double number_scale    = std::max(double(initial_number), std::abs(dt * rates[3]));
        const double number_rounding = dt * (double(candidate[3]) - rates[3]);
        if (initial_number + dt * candidate[3] <= 0 ||
            std::abs(number_rounding) > 8 * std::numeric_limits<float>::epsilon() * number_scale +
                                            dt * std::numeric_limits<float>::denorm_min())
            return false;
    }
    const double condensation = double(candidate[0]) - rates[0];
    candidate[1]              = static_cast<float>(double(rates[1]) - condensation);
    candidate[2] =
        static_cast<float>(double(rates[2]) + condensation * microphysics_constants::L_v / microphysics_constants::cp);
    if (initial_vapor + dt * candidate[1] < 0 ||
        !std::all_of(candidate.begin(), candidate.end(), [](float value) { return std::isfinite(value); }))
        return false;
    rates = candidate;
    return true;
}
} // namespace tmv::milbrandt_detail
