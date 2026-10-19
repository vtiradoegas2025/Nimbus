#pragma once

#include "microphysics/schemes/milbrandt/triple_cold_riming.hpp"
#include "milbrandt_cold_cloud_diffusion_reference.hpp"
#include "milbrandt_ice_cloud_collection_reference.hpp"

namespace milbrandt_cold_riming_reference
{

namespace thermal = milbrandt_cold_cloud_diffusion_reference;
namespace collection = milbrandt_cold_cloud_collection_reference;
namespace ice_collection = milbrandt_ice_cloud_collection_reference;
using Cell = tmv::milbrandt_detail::ColdRimingCell;

struct Reference
{
    std::array<long double, 4> raw_mass{}, raw_number{}, collected{}, cloud_number_loss{};
    std::array<long double, 4> environment{}, diffusion{}, diffusion_number{};
    std::array<long double, 4> splinter_mass{}, splinter_number{}, splinter_z{};
    std::array<long double, 4> mass_increment{}, number_increment{}, z_increment{};
    long double factor = 1;
    long double cloud_increment = 0;
    long double cloud_number_increment = 0;
    long double vapor_increment = 0;
    long double temperature_increment = 0;
    long double relaxation = 0;
    long double feedback = 0;
};

inline Reference evaluate(const Cell& cell, long double duration, bool instantaneous = false)
{
    tmv::milbrandt_detail::ColdCloudDiffusionCell secondary;
    secondary.temperature_k = cell.temperature_k;
    secondary.pressure_pa = cell.pressure_pa;
    secondary.dry_density_kg_m3 = cell.dry_density_kg_m3;
    secondary.vapor_mass_per_kg = cell.initial.vapor_mass_per_kg;
    secondary.initial.cloud_mass_per_kg = cell.initial.cloud_mass_per_kg;
    secondary.initial.cloud_number_per_kg = cell.initial.cloud_number_per_kg;
    std::copy(cell.initial.frozen.begin() + 1, cell.initial.frozen.end(), secondary.initial.frozen.begin());
    const auto secondary_reference = thermal::evaluate(secondary, duration);
    const ice_collection::Cell ice{cell.temperature_k, cell.pressure_pa, cell.dry_density_kg_m3,
        {cell.initial.cloud_mass_per_kg, cell.initial.cloud_number_per_kg, cell.initial.frozen[0]}};
    const auto ice_reference = ice_collection::evaluate(ice, duration);
    Reference result;
    result.feedback = secondary_reference.feedback;
    result.raw_mass[0] = ice_reference.raw_mass;
    result.raw_number[0] = ice_reference.raw_number;
    long double sum_mass = result.raw_mass[0];
    long double sum_number = result.raw_number[0];
    for (std::size_t index = 1; index < 4; ++index)
    {
        result.raw_mass[index] = secondary_reference.collection.channels[index - 1].raw_mass;
        result.raw_number[index] = secondary_reference.collection.channels[index - 1].raw_number;
        result.environment[index] = secondary_reference.environmental[index - 1];
        sum_mass += result.raw_mass[index];
        sum_number += result.raw_number[index];
    }
    if (!instantaneous)
    {
        if (sum_mass > cell.initial.cloud_mass_per_kg)
        {
            result.factor = cell.initial.cloud_mass_per_kg / sum_mass;
        }
        if (sum_number > cell.initial.cloud_number_per_kg)
        {
            result.factor = std::min(result.factor, cell.initial.cloud_number_per_kg / sum_number);
        }
        if (result.factor < 1)
        {
            result.factor *= 1 - std::ldexp(1.0L, -40);
        }
    }
    result.relaxation = secondary_reference.relaxation;
    if (cell.initial.frozen[0].mass_per_kg > 0)
    {
        const auto& spectrum = ice_reference.spectrum;
        const long double ventilation = .78L * collection::frozen::moment(spectrum, 1)
            + .30L * secondary_reference.schmidt_third
            * std::sqrt(secondary_reference.density_factor * 71.34L / secondary_reference.viscosity)
            * collection::frozen::moment(spectrum, 1.5L + .5L * .6635L);
        const long double conductance = std::acos(-1.0L) * cell.initial.frozen[0].number_per_kg
            * ventilation / secondary_reference.resistance;
        result.environment[0] = duration * conductance * (secondary_reference.saturation_ratio - 1);
        result.relaxation += duration * conductance / secondary_reference.saturation.mass
            * (1 + secondary_reference.saturation.derivative * thermal::latent_heat_sublimation / collection::heat_capacity);
    }
    const long double tc = static_cast<long double>(cell.temperature_k) - static_cast<long double>(273.15);
    long double yield = 0;
    if (tc >= -8 && tc <= -5)
    {
        yield = 3.5e8L * (tc + 8) / 3;
    }
    else if (tc > -5 && tc < -3)
    {
        yield = 3.5e8L * (-3 - tc) / 2;
    }
    for (std::size_t index = 0; index < 4; ++index)
    {
        const auto& source = cell.initial.frozen[index];
        result.collected[index] = result.factor * result.raw_mass[index];
        result.cloud_number_loss[index] = result.factor * result.raw_number[index];
        result.diffusion[index] = result.environment[index] - result.feedback * result.collected[index];
        if (source.mass_per_kg > 0)
        {
            result.diffusion_number[index] = source.number_per_kg * std::min(result.diffusion[index], 0.0L) / source.mass_per_kg;
            const long double power = 6 / collection::frozen::mass_exponent[index];
            result.z_increment[index] = source.reflectivity_per_kg *
                (power * (result.collected[index] + result.diffusion[index]) / source.mass_per_kg
                    + (1 - power) * result.diffusion_number[index] / source.number_per_kg);
        }
        result.mass_increment[index] = result.collected[index] + result.diffusion[index];
        result.number_increment[index] = result.diffusion_number[index];
        result.cloud_increment -= result.collected[index];
        result.cloud_number_increment -= result.cloud_number_loss[index];
        result.vapor_increment -= result.diffusion[index];
        if (index < 3)
        {
            result.splinter_number[index] = yield * result.collected[index];
            result.splinter_mass[index] = 1e-12L * result.splinter_number[index];
            result.splinter_z[index] = 20e-24L * result.splinter_number[index] / (440 * 440);
        }
    }
    for (std::size_t donor = 0; donor < 3; ++donor)
    {
        result.number_increment[0] += result.splinter_number[donor];
        result.z_increment[0] += result.splinter_z[donor];
        if (donor != 0)
        {
            result.mass_increment[donor] -= result.splinter_mass[donor];
            result.mass_increment[0] += result.splinter_mass[donor];
            result.z_increment[donor] -= result.splinter_z[donor];
        }
    }
    result.temperature_increment = (-collection::latent_heat_fusion * result.cloud_increment
        - thermal::latent_heat_sublimation * result.vapor_increment) / collection::heat_capacity;
    return result;
}

using State = std::array<long double, 16>;

inline State state(const Cell& cell)
{
    State result{cell.temperature_k, cell.initial.vapor_mass_per_kg,
        cell.initial.cloud_mass_per_kg, cell.initial.cloud_number_per_kg};
    for (std::size_t index = 0; index < 4; ++index)
    {
        result[4 + index * 3] = cell.initial.frozen[index].mass_per_kg;
        result[5 + index * 3] = cell.initial.frozen[index].number_per_kg;
        result[6 + index * 3] = cell.initial.frozen[index].reflectivity_per_kg;
    }
    return result;
}

inline Cell from_state(const Cell& context, const State& values)
{
    auto cell = context;
    cell.temperature_k = static_cast<double>(values[0]);
    cell.initial.vapor_mass_per_kg = static_cast<double>(values[1]);
    cell.initial.cloud_mass_per_kg = static_cast<double>(values[2]);
    cell.initial.cloud_number_per_kg = static_cast<double>(values[3]);
    for (std::size_t index = 0; index < 4; ++index)
    {
        cell.initial.frozen[index] = {static_cast<double>(values[4 + index * 3]),
            static_cast<double>(values[5 + index * 3]), static_cast<double>(values[6 + index * 3])};
    }
    return cell;
}

inline State derivative(const Cell& context, const State& values)
{
    const auto rate = evaluate(from_state(context, values), 1, true);
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
    auto values = state(source);
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

} // namespace milbrandt_cold_riming_reference
