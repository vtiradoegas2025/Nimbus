#pragma once

#include "core/infra/physical_constants.hpp"

#include <algorithm>
#include <cmath>

namespace tmv::init
{

/**
 * @brief Surface dry-air water mixing ratio implied by an LCL target.
 *
 * Uses the same Lawrence dewpoint-depression approximation and Magnus
 * saturation relation as ParametricTargetsSoundingSource. Keeping this small
 * calculation shared lets strict configuration validation reject an LCL whose
 * translated qv would later be clamped by the delegated CAPE source.
 */
inline double parametric_target_qv_from_lcl(double lcl_m,
                                             double surface_temperature_k,
                                             double surface_pressure_pa)
{
    constexpr double qv_floor = 1.0e-5;
    if (!std::isfinite(lcl_m) || lcl_m <= 0.0)
    {
        return qv_floor;
    }

    constexpr double lcl_per_dewpoint_depression_m_per_k = 125.0;
    const double dewpoint_depression_k =
        lcl_m / lcl_per_dewpoint_depression_m_per_k;
    const double dewpoint_k = surface_temperature_k - dewpoint_depression_k;
    if (!std::isfinite(dewpoint_k) || dewpoint_k <= 0.0 ||
        dewpoint_k > surface_temperature_k)
    {
        return qv_floor;
    }

    const double dewpoint_c =
        dewpoint_k - physical_constants::freezing_temperature_k;
    const double vapor_pressure_pa =
        dewpoint_k >= physical_constants::freezing_temperature_k
            ? 611.21 * std::exp((18.678 - dewpoint_c / 234.5) *
                                dewpoint_c / (257.14 + dewpoint_c))
            : 611.15 * std::exp((23.036 - dewpoint_c / 333.7) *
                                dewpoint_c / (279.82 + dewpoint_c));
    const double qv = 0.622 * vapor_pressure_pa /
        std::max(surface_pressure_pa - vapor_pressure_pa, 1.0);
    return std::max(qv, qv_floor);
}

}  // namespace tmv::init
