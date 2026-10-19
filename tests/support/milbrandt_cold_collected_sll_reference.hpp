#pragma once

#include "milbrandt_cold_sll_reference.hpp"
#include "milbrandt_frozen_collection_reference.hpp"

namespace milbrandt_cold_collected_sll_reference
{

namespace prefix = milbrandt_cold_sll_reference;
namespace homogeneous = milbrandt_cold_homogeneous_reference;
namespace riming = milbrandt_cold_riming_reference;
namespace frozen = milbrandt_frozen_collection_reference;
namespace hail_transfer = milbrandt_graupel_hail_conversion_reference;
using Cell = riming::Cell;
using State = riming::State;

struct Reference : homogeneous::Reference
{
    frozen::Reference collection;
    hail_transfer::Reference sll;
    std::array<long double, 3> original_mass_sink{};
    std::array<long double, 3> original_number_sink{};
};

inline frozen::Reference collection_rates(const Cell& source, long double duration,
                                         bool instantaneous, bool quadrature, bool wet_hail = false)
{
    const frozen::Cell cell{source.temperature_k, source.pressure_pa, source.dry_density_kg_m3,
        {source.initial.frozen}};
    auto result = frozen::evaluate(cell, duration, quadrature, wet_hail);
    if (!instantaneous)
    {
        return result;
    }
    const auto& snow = source.initial.frozen[1];
    if (snow.mass_per_kg > 1e-14)
    {
        const long double density = source.dry_density_kg_m3;
        const long double efficiency = std::min(1.0L,
            .05L * std::exp(.1L * (static_cast<long double>(source.temperature_k) - 273.15L)));
        result.channels[4].raw_number = duration * .93952L * efficiency
            * std::pow(density * snow.mass_per_kg, (2 + frozen::fall_exponent[1]) / 3)
            * std::pow(density * snow.number_per_kg, (4 - frozen::fall_exponent[1]) / 3) / density;
    }
    result.mass_increment = {};
    result.number_increment = {};
    result.reflectivity_increment = {};
    for (std::size_t channel = 0; channel < 5; ++channel)
    {
        auto& event = result.channels[channel];
        event.factor = 1;
        event.mass = event.raw_mass;
        event.number = event.raw_number;
        if (channel < 4)
        {
            const std::size_t donor = channel == 3 ? 1 : 0;
            const std::size_t collector = channel == 3 ? 3 : channel + 1;
            result.mass_increment[donor] -= event.mass;
            result.mass_increment[collector] += event.mass;
            result.number_increment[donor] -= event.number;
        }
        else
        {
            result.number_increment[1] -= event.number;
        }
    }
    for (std::size_t species = 0; species < 4; ++species)
    {
        const auto& population = source.initial.frozen[species];
        if (population.mass_per_kg > 0)
        {
            const long double power = 6 / frozen::mass_exponent[species];
            result.reflectivity_increment[species] = population.reflectivity_per_kg
                * (power * result.mass_increment[species] / population.mass_per_kg
                    + (1 - power) * result.number_increment[species] / population.number_per_kg);
        }
        result.endpoint[species] = {population.mass_per_kg + result.mass_increment[species],
            population.number_per_kg + result.number_increment[species],
            population.reflectivity_per_kg + result.reflectivity_increment[species]};
    }
    return result;
}

inline Reference evaluate(const Cell& cell, long double duration, bool hail = true,
                          bool instantaneous = false, bool quadrature = false,
                          const std::array<long double, 4>& rain_collection = {}, long double rain_mass = 0,
                          bool graupel_rain_stays_graupel = true, bool wet_hail = false)
{
    Reference result;
    static_cast<homogeneous::Reference&>(result) = homogeneous::evaluate(cell, duration, instantaneous, rain_collection, wet_hail);
    result.collection = collection_rates(cell, duration, instantaneous, quadrature, wet_hail);
    hail_transfer::Cell transfer;
    transfer.source = cell;
    transfer.rain_mass_per_kg = static_cast<double>(rain_mass);
    transfer.admitted_accretion_mass_per_kg[1] = graupel_rain_stays_graupel ? static_cast<double>(rain_collection[2]) : 0;
    transfer.admitted_accretion_mass_per_kg[0] = static_cast<double>(result.collected[2]);
    transfer.admitted_accretion_mass_per_kg[2] = static_cast<double>(result.collection.channels[1].mass);
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
        result.mass_increment[species] += result.collection.mass_increment[species] + result.sll.mass[species];
        result.number_increment[species] += result.collection.number_increment[species] + result.sll.number[species];
        result.z_increment[species] += result.collection.reflectivity_increment[species] + result.sll.reflectivity[species];
    }
    for (std::size_t channel = 0; channel < 3; ++channel)
    {
        result.original_mass_sink[0] += result.collection.channels[channel].mass;
        result.original_number_sink[0] += result.collection.channels[channel].number;
    }
    result.original_mass_sink[0] += result.conversion_mass + std::max(-result.diffusion[0], 0.0L);
    result.original_number_sink[0] += result.ice_number_loss + std::max(-result.diffusion_number[0], 0.0L);
    result.original_mass_sink[1] = result.collection.channels[3].mass + result.snow_conversion_mass
        + std::max(-result.diffusion[1], 0.0L);
    result.original_number_sink[1] = result.collection.channels[3].number + result.collection.channels[4].number
        + result.snow_conversion_number + std::max(-result.diffusion_number[1], 0.0L);
    result.original_mass_sink[2] = result.sll.admitted + std::max(-result.diffusion[2], 0.0L);
    result.original_number_sink[2] = -result.sll.number[2] + std::max(-result.diffusion_number[2], 0.0L);
    return result;
}

inline bool original_donors_admissible(const Cell& cell, const Reference& result)
{
    for (std::size_t species = 0; species < 3; ++species)
    {
        if (result.original_mass_sink[species] > cell.initial.frozen[species].mass_per_kg
            || result.original_number_sink[species] > cell.initial.frozen[species].number_per_kg)
        {
            return false;
        }
    }
    return true;
}

inline State derivative(const Cell& context, const State& values, bool hail = true)
{
    const auto rate = evaluate(riming::from_state(context, values), 1, hail, true);
    State result{rate.temperature_increment, rate.vapor_increment, rate.cloud_increment, rate.cloud_number_increment};
    for (std::size_t species = 0; species < 4; ++species)
    {
        result[4 + species * 3] = rate.mass_increment[species];
        result[5 + species * 3] = rate.number_increment[species];
        result[6 + species * 3] = rate.z_increment[species];
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
            for (std::size_t species = 0; species < values.size(); ++species)
            {
                candidate[species] += fraction * h * slope[species];
            }
            return candidate;
        };
        const auto a = derivative(source, values, hail);
        const auto b = derivative(source, stage(a, .5L), hail);
        const auto c = derivative(source, stage(b, .5L), hail);
        const auto d = derivative(source, stage(c, 1), hail);
        for (std::size_t species = 0; species < values.size(); ++species)
        {
            const long double change = h / 6 * (a[species] + 2 * b[species] + 2 * c[species] + d[species])
                - accumulation_roundoff[species];
            const long double next = values[species] + change;
            accumulation_roundoff[species] = (next - values[species]) - change;
            values[species] = next;
        }
    }
    return values;
}

} // namespace milbrandt_cold_collected_sll_reference
