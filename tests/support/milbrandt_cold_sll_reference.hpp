#pragma once

#include "milbrandt_cold_homogeneous_reference.hpp"
#include "milbrandt_graupel_hail_conversion_reference.hpp"

namespace milbrandt_cold_sll_reference
{

namespace homogeneous = milbrandt_cold_homogeneous_reference;
namespace riming = milbrandt_cold_riming_reference;
namespace hail_transfer = milbrandt_graupel_hail_conversion_reference;
using Cell = riming::Cell;
using State = riming::State;

struct Reference : homogeneous::Reference
{
    hail_transfer::Reference sll;
};

inline Reference evaluate(const Cell& cell, long double duration, bool hail = true,
                          bool instantaneous = false)
{
    Reference result;
    static_cast<homogeneous::Reference&>(result) = homogeneous::evaluate(cell, duration, instantaneous);
    hail_transfer::Cell transfer;
    transfer.source = cell;
    transfer.admitted_accretion_mass_per_kg[0] = static_cast<double>(result.collected[2]);
    transfer.graupel_diffusion_mass_per_kg = static_cast<double>(result.diffusion[2]);
    transfer.graupel_diffusion_number_per_kg = static_cast<double>(result.diffusion_number[2]);
    result.sll = hail_transfer::evaluate(transfer, hail);
    if (instantaneous && result.sll.raw > result.sll.admitted)
    {
        const auto& graupel = cell.initial.frozen[2];
        constexpr long double cg = 209.439510239L;
        constexpr long double ch = 471.238898038L;
        result.sll.admitted = result.sll.raw;
        result.sll.fraction = result.sll.raw / graupel.mass_per_kg;
        result.sll.mass[2] = -result.sll.raw;
        result.sll.mass[3] = result.sll.raw;
        result.sll.number[2] = -result.sll.fraction * graupel.number_per_kg;
        result.sll.number[3] = -result.sll.number[2];
        result.sll.reflectivity[2] = -result.sll.fraction * graupel.reflectivity_per_kg;
        result.sll.reflectivity[3] = -result.sll.reflectivity[2] * (cg / ch) * (cg / ch);
    }
    for (std::size_t species = 0; species < 4; ++species)
    {
        result.mass_increment[species] += result.sll.mass[species];
        result.number_increment[species] += result.sll.number[species];
        result.z_increment[species] += result.sll.reflectivity[species];
    }
    return result;
}

inline State derivative(const Cell& context, const State& values, bool hail = true)
{
    const auto rate = evaluate(riming::from_state(context, values), 1, hail, true);
    State result{rate.temperature_increment, rate.vapor_increment, rate.cloud_increment, rate.cloud_number_increment};
    for (std::size_t index = 0; index < 4; ++index)
    {
        result[4 + index * 3] = rate.mass_increment[index];
        result[5 + index * 3] = rate.number_increment[index];
        result[6 + index * 3] = rate.z_increment[index];
    }
    return result;
}

inline State rk4(const Cell& source, long double duration, int steps, bool hail = true)
{
    auto values = riming::state(source);
    State accumulation_roundoff{};
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
        const auto a = derivative(source, values, hail);
        const auto b = derivative(source, stage(a, .5L), hail);
        const auto c = derivative(source, stage(b, .5L), hail);
        const auto d = derivative(source, stage(c, 1), hail);
        for (std::size_t index = 0; index < values.size(); ++index)
        {
            const long double change = h / 6 * (a[index] + 2 * b[index] + 2 * c[index] + d[index])
                - accumulation_roundoff[index];
            const long double next = values[index] + change;
            accumulation_roundoff[index] = (next - values[index]) - change;
            values[index] = next;
        }
    }
    return values;
}

} // namespace milbrandt_cold_sll_reference
