#pragma once

#include "microphysics/schemes/milbrandt/triple_graupel_hail_conversion.hpp"
#include "milbrandt_frozen_collection_reference.hpp"

#include <limits>

namespace milbrandt_graupel_hail_conversion_reference
{

using Cell = tmv::milbrandt_detail::GraupelHailConversionCell;

struct Reference
{
    long double loading = 0;
    long double diameter = 0;
    long double embryo_diameter = 0;
    long double ratio = 0;
    long double raw = 0;
    long double admitted = 0;
    long double fraction = 0;
    std::array<long double, 4> mass{}, number{}, reflectivity{};
};

inline Reference evaluate(const Cell& cell, bool hail = true)
{
    constexpr long double cg = 209.439510239L;
    constexpr long double ch = 471.238898038L;
    const auto& source = cell.source;
    const auto& initial = source.initial;
    const auto& g = initial.frozen[2];
    Reference result;
    result.loading = 1.1e4L * (static_cast<long double>(initial.cloud_mass_per_kg) + cell.rain_mass_per_kg)
        - 1.3e3L * initial.frozen[0].mass_per_kg + .001L;
    if (g.mass_per_kg == 0)
    {
        return result;
    }
    result.diameter = std::cbrt(static_cast<long double>(g.mass_per_kg) / (cg * g.number_per_kg));
    const long double accreted = static_cast<long double>(cell.admitted_accretion_mass_per_kg[0])
        + cell.admitted_accretion_mass_per_kg[1] + cell.admitted_accretion_mass_per_kg[2];
    if (!hail || accreted == 0 || result.loading <= 0)
    {
        return result;
    }
    const long double x = (static_cast<long double>(273.15) - source.temperature_k) / result.loading;
    if (x > std::log(std::numeric_limits<long double>::max()) - 1)
    {
        return result;
    }
    result.embryo_diameter = .01L * std::expm1(x);
    result.ratio = result.diameter / result.embryo_diameter;
    if (result.ratio < .1L)
    {
        return result;
    }
    result.raw = .5L * result.ratio * accreted;
    result.admitted = std::min(result.raw, static_cast<long double>(g.mass_per_kg) + accreted);
    result.fraction = result.admitted / g.mass_per_kg;
    result.mass[2] = -result.admitted;
    result.mass[3] = result.admitted;
    result.number[2] = -result.fraction * g.number_per_kg;
    result.number[3] = -result.number[2];
    result.reflectivity[2] = -result.fraction * g.reflectivity_per_kg;
    result.reflectivity[3] = -result.reflectivity[2] * (cg / ch) * (cg / ch);
    return result;
}

} // namespace milbrandt_graupel_hail_conversion_reference
