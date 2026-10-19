#pragma once

#include "milbrandt_cold_riming_reference.hpp"

namespace milbrandt_cold_ice_graupel_reference
{

namespace riming = milbrandt_cold_riming_reference;
using Cell = riming::Cell;
using State = riming::State;

struct Reference : riming::Reference
{
    long double conversion_mass = 0;
    long double ice_number_loss = 0;
    long double ice_z_loss = 0;
    long double graupel_number_gain = 0;
    long double graupel_z_gain = 0;
};

inline Reference evaluate(const Cell& cell, long double duration, bool instantaneous = false)
{
    Reference result;
    static_cast<riming::Reference&>(result) = riming::evaluate(cell, duration, instantaneous);
    const auto& ice = cell.initial.frozen[0];
    if (ice.mass_per_kg > 0)
    {
        result.conversion_mass = 2 * std::max(result.collected[0] - std::max(result.diffusion[0], 0.0L), 0.0L);
        const long double fraction = result.conversion_mass / ice.mass_per_kg;
        result.ice_number_loss = fraction * ice.number_per_kg;
        result.ice_z_loss = fraction * ice.reflectivity_per_kg;
        result.graupel_number_gain = result.conversion_mass / 1.6e-10L;
        constexpr long double cg = 209.439510239L;
        result.graupel_z_gain = 20 * 1.6e-10L * result.conversion_mass / (cg * cg);
        result.mass_increment[0] -= result.conversion_mass;
        result.mass_increment[2] += result.conversion_mass;
        result.number_increment[0] -= result.ice_number_loss;
        result.number_increment[2] += result.graupel_number_gain;
        result.z_increment[0] -= result.ice_z_loss;
        result.z_increment[2] += result.graupel_z_gain;
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

} // namespace milbrandt_cold_ice_graupel_reference
