#pragma once

#include "microphysics/schemes/milbrandt/triple_homogeneous_cloud_freezing.hpp"

#include <algorithm>
#include <cmath>

namespace milbrandt_homogeneous_cloud_freezing_reference
{

using Cell = tmv::milbrandt_detail::HomogeneousCloudFreezingCell;

struct Reference
{
    long double log10_rate = 0;
    long double mean_volume_cm3 = 0;
    long double source_mass_squared_per_kg = 0;
    long double hazard = 0;
    long double fraction = 0;
    long double mass_increment = 0;
    long double number_increment = 0;
    long double z_increment = 0;
};

inline Reference evaluate(const Cell& cell, long double duration)
{
    Reference result;
    const long double tc = static_cast<long double>(cell.temperature_k) - static_cast<long double>(273.15);
    result.log10_rate = std::max(-20.0L, -606.3952L - 52.6611L * tc - 1.7439L * tc * tc
        - .0265L * tc * tc * tc - 1.536e-4L * tc * tc * tc * tc);
    if (cell.cloud_mass_per_kg == 0)
    {
        return result;
    }
    const long double q = cell.cloud_mass_per_kg;
    const long double n = cell.cloud_number_per_kg;
    const long double mean_mass = q / n;
    constexpr long double rain_mass_coefficient = static_cast<long double>(523.598775598);
    constexpr long double ice_mass_coefficient = 440;
    result.mean_volume_cm3 = std::acos(-1.0L) / 6 * 1e6L * mean_mass / rain_mass_coefficient;
    const long double generalized_second_mass_ratio = std::tgamma(4.0L) * std::tgamma(2.0L)
        / (std::tgamma(3.0L) * std::tgamma(3.0L));
    result.source_mass_squared_per_kg = generalized_second_mass_ratio * q * mean_mass;
    if (cell.cloud_mass_per_kg <= 1e-14 || tc > -30)
    {
        return result;
    }
    result.hazard = std::pow(10.0L, result.log10_rate) * result.mean_volume_cm3 * duration;
    result.fraction = tc < -50 ? 1 : -std::expm1(-result.hazard);
    result.mass_increment = result.fraction * q;
    result.number_increment = result.fraction * n;
    result.z_increment = result.fraction * result.source_mass_squared_per_kg
        / (ice_mass_coefficient * ice_mass_coefficient);
    return result;
}

inline long double cloud_mass_squared_quadrature(long double mass, long double number)
{
    constexpr int intervals = 16000;
    constexpr long double upper = 80;
    const long double step = upper / intervals;
    long double normalization = 0;
    long double moment = 0;
    for (int index = 1; index <= intervals; ++index)
    {
        const long double x = index * step;
        const long double weight = (index == intervals ? 1 : index % 2 ? 4 : 2)
            * x * std::exp(-x);
        normalization += weight;
        moment += weight * x * x;
    }
    const long double mass_scale = mass / (2 * number);
    return number * mass_scale * mass_scale * moment / normalization;
}

} // namespace milbrandt_homogeneous_cloud_freezing_reference
