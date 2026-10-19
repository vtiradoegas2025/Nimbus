#pragma once

#include "milbrandt_cold_ice_graupel_reference.hpp"

namespace milbrandt_cold_graupel_conversions_reference
{

namespace ice = milbrandt_cold_ice_graupel_reference;
namespace riming = milbrandt_cold_riming_reference;
using Cell = riming::Cell;
using State = riming::State;

struct Reference : ice::Reference
{
    long double snow_conversion_mass = 0;
    long double snow_conversion_number = 0;
    long double snow_z_loss = 0;
    long double snow_to_graupel_z = 0;
    long double snow_mass_squared = 0;
};

inline Reference evaluate(const Cell& cell, long double duration, bool instantaneous = false)
{
    Reference result;
    static_cast<ice::Reference&>(result) = ice::evaluate(cell, duration, instantaneous);
    const auto& snow = cell.initial.frozen[1];
    if (snow.mass_per_kg > 0 && result.collected[1] > result.diffusion[1])
    {
        const auto spectrum = riming::collection::frozen::spectrum(snow, 1, cell.dry_density_kg_m3);
        const long double d = riming::collection::frozen::mass_exponent[1];
        const long double ratio = std::exp(std::lgamma(spectrum.alpha + 1 + 2 * d)
            + std::lgamma(spectrum.alpha + 1) - 2 * std::lgamma(spectrum.alpha + 1 + d));
        result.snow_mass_squared = static_cast<long double>(snow.mass_per_kg) * snow.mass_per_kg / snow.number_per_kg * ratio;
        result.snow_conversion_mass = (400.0L / 300.0L) * result.collected[1];
        const long double fraction = result.snow_conversion_mass / snow.mass_per_kg;
        result.snow_conversion_number = fraction * snow.number_per_kg;
        result.snow_z_loss = fraction * snow.reflectivity_per_kg;
        constexpr long double cg = 209.439510239L;
        result.snow_to_graupel_z = fraction * result.snow_mass_squared / (cg * cg);
        result.mass_increment[1] -= result.snow_conversion_mass;
        result.mass_increment[2] += result.snow_conversion_mass;
        result.number_increment[1] -= result.snow_conversion_number;
        result.number_increment[2] += result.snow_conversion_number;
        result.z_increment[1] -= result.snow_z_loss;
        result.z_increment[2] += result.snow_to_graupel_z;
    }
    return result;
}

inline State derivative(const Cell& context, const State& values)
{
    const auto rate = evaluate(riming::from_state(context, values), 1, true);
    State result{rate.temperature_increment, rate.vapor_increment, rate.cloud_increment, rate.cloud_number_increment};
    for (std::size_t index = 0; index < 4; ++index)
    {
        result[4 + index * 3] = rate.mass_increment[index];
        result[5 + index * 3] = rate.number_increment[index];
        result[6 + index * 3] = rate.z_increment[index];
    }
    return result;
}

inline State rk4(const Cell& source, long double duration, int steps)
{
    auto values = riming::state(source);
    const long double h = duration / steps;
    for (int step = 0; step < steps; ++step)
    {
        const auto stage = [&](const State& slope, long double fraction)
        {
            auto candidate = values;
            for (std::size_t index = 0; index < values.size(); ++index)
            {
                candidate[index] += fraction * h * slope[index];
            }
            return candidate;
        };
        const auto a = derivative(source, values);
        const auto b = derivative(source, stage(a, .5L));
        const auto c = derivative(source, stage(b, .5L));
        const auto d = derivative(source, stage(c, 1));
        for (std::size_t index = 0; index < values.size(); ++index)
        {
            values[index] += h / 6 * (a[index] + 2 * b[index] + 2 * c[index] + d[index]);
        }
    }
    return values;
}

} // namespace milbrandt_cold_graupel_conversions_reference
