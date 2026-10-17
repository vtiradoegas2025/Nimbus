#pragma once

#include "microphysics/schemes/milbrandt/triple_cloud_adjustment.hpp"
#include "microphysics/base/thermodynamics.hpp"

#include <algorithm>
#include <cmath>

namespace cloud_adjustment_reference
{

inline long double saturation(long double temperature, long double pressure)
{
    const long double tc = temperature - 273.15L;
    const long double vapor_pressure = 611.21L * std::exp((18.678L - tc / 234.5L) * tc / (257.14L + tc));
    return std::min(.2L, .62198L * vapor_pressure / std::max(pressure - vapor_pressure, 1.0L));
}

inline long double activated(long double velocity, long double temperature, long double pressure)
{
    // Preserve the binary policy boundary while evaluating the fit in extended precision.
    if (!(velocity > static_cast<long double>(.001)))
    {
        return 2e8L;
    }
    const long double x = std::log10(100 * velocity);
    const long double t = temperature - 273.15L;
    const long double p = pressure / 100;
    const long double c = (((-2.112e-9L * t + 3.9836e-8L) * t + 2.3703e-6L) * t - 1.4542e-4L) * t - .0698L;
    const long double d = (((-4.210e-8L * t + 5.5745e-7L) * t + 1.8460e-5L) * t + 9.6078e-4L) * t + .7120L;
    const long double e = (((1.434e-7L * t - 1.6455e-6L) * t - 4.3334e-5L) * t - 7.6720e-3L) * t + 1.0056L;
    const long double f = (1.340e-6L * p - 3.5114e-3L) * p + 1.9453L;
    const long double g = (((4.226e-3L * x - 5.6012e-3L) * x - 8.7846e-2L) * x + 2.7435e-2L) * x + .9932L;
    const long double h = (((5.811e-9L * t + 1.5589e-7L) * t - 3.8623e-5L) * t + 1.4471e-3L) * t + .1496L;
    return 1e6L * std::pow(10.0L, std::max(0.0L, c * x * x + d * x + e + f * g * h));
}

struct Result
{
    long double transfer = 0;
    long double vapor = 0;
    long double cloud = 0;
    long double number = 0;
    long double heat = 0;
    long double temperature = 0;
    long double target = 0;
};

inline Result evaluate(const tmv::milbrandt_detail::TripleCloudAdjustmentCell& source)
{
    using namespace tmv;
    const long double factor = static_cast<long double>(microphysics_constants::L_v) / microphysics_constants::cp;
    const bool condensation = source.vapor_per_kg > thermodynamics::saturation_mixing_ratio_water(source.temperature_k, source.pressure_pa);
    const bool evaporation = source.vapor_per_kg < thermodynamics::saturation_mixing_ratio_water(source.temperature_k, source.pressure_pa)
        && source.cloud_per_kg > 0;
    long double transfer = 0;
    if (condensation || evaporation)
    {
        long double lower = condensation ? 0 : -static_cast<long double>(source.cloud_per_kg);
        long double upper = condensation ? static_cast<long double>(source.vapor_per_kg) : 0;
        const auto residual = [&](long double value)
        {
            return source.vapor_per_kg - value - saturation(source.temperature_k + factor * value, source.pressure_pa);
        };
        if (evaporation && residual(lower) < 0)
        {
            transfer = lower;
        }
        else
        {
            for (int iteration = 0; iteration < 160; ++iteration)
            {
                const long double midpoint = (lower + upper) / 2;
                if (midpoint == lower || midpoint == upper)
                {
                    break;
                }
                if (residual(midpoint) > 0)
                {
                    lower = midpoint;
                }
                else
                {
                    upper = midpoint;
                }
            }
            transfer = (lower + upper) / 2;
        }
    }
    Result result;
    result.transfer = transfer;
    result.vapor = source.vapor_per_kg - transfer;
    result.cloud = source.cloud_per_kg + transfer;
    result.number = source.cloud_number_per_kg;
    result.heat = factor * transfer;
    result.temperature = source.temperature_k + result.heat;
    if (condensation)
    {
        result.target = activated(source.vertical_velocity_ms, result.temperature, source.pressure_pa);
        result.number = std::max(static_cast<long double>(source.cloud_number_per_kg), result.target / source.dry_density_kg_m3);
    }
    else if (evaporation)
    {
        result.number *= result.cloud / source.cloud_per_kg;
    }
    return result;
}

} // namespace cloud_adjustment_reference
