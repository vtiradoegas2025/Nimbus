#pragma once

#include "milbrandt_cold_graupel_conversions_reference.hpp"

namespace milbrandt_cold_homogeneous_reference
{

namespace conversions = milbrandt_cold_graupel_conversions_reference;
namespace riming = milbrandt_cold_riming_reference;
using Cell = riming::Cell;
using State = riming::State;

struct Reference : conversions::Reference
{
    long double log10_rate = 0;
    long double mean_volume_cm3 = 0;
    long double source_cloud_mass_squared = 0;
    long double hazard = 0;
    long double raw_fraction = 0;
    std::array<long double, 3> raw_freezing{};
    std::array<long double, 3> admitted_freezing{};
};

inline Reference evaluate(const Cell& cell, long double duration, bool instantaneous = false,
                          const std::array<long double, 4>& rain_collection = {}, bool wet_hail = false)
{
    Reference result;
    static_cast<riming::Reference&>(result) = riming::evaluate(cell, duration, true);
    const long double tc = static_cast<long double>(cell.temperature_k) - static_cast<long double>(273.15);
    result.log10_rate = std::max(-20.0L, -606.3952L - 52.6611L * tc - 1.7439L * tc * tc
        - .0265L * tc * tc * tc - 1.536e-4L * tc * tc * tc * tc);
    const long double qc = cell.initial.cloud_mass_per_kg;
    const long double nc = cell.initial.cloud_number_per_kg;
    if (qc > 0)
    {
        constexpr long double cloud_c = static_cast<long double>(523.598775598);
        result.mean_volume_cm3 = std::acos(-1.0L) / 6 * 1e6L * qc / (cloud_c * nc);
        result.source_cloud_mass_squared = 1.5L * qc * qc / nc;
        if (qc > 1e-14 && tc <= -30)
        {
            result.hazard = std::pow(10.0L, result.log10_rate) * result.mean_volume_cm3 * duration;
            result.raw_fraction = instantaneous ? result.hazard : tc < -50 ? 1 : -std::expm1(-result.hazard);
            result.raw_freezing = {result.raw_fraction * qc, result.raw_fraction * nc,
                result.raw_fraction * result.source_cloud_mass_squared / (440 * 440)};
        }
    }
    if (wet_hail)
    {
        riming::collection::Cell secondary{cell.temperature_k, cell.pressure_pa, cell.dry_density_kg_m3,
            {cell.initial.cloud_mass_per_kg, cell.initial.cloud_number_per_kg, {}}};
        std::copy(cell.initial.frozen.begin() + 1, cell.initial.frozen.end(), secondary.initial.frozen.begin());
        const auto unit = riming::collection::evaluate(secondary, duration, false, true);
        result.raw_mass[3] = unit.channels[2].raw_mass;
        result.raw_number[3] = unit.channels[2].raw_number;
    }
    long double sum_mass = result.raw_freezing[0];
    long double sum_number = result.raw_freezing[1];
    for (std::size_t index = 0; index < 4; ++index)
    {
        sum_mass += result.raw_mass[index];
        sum_number += result.raw_number[index];
    }
    result.factor = 1;
    if (!instantaneous)
    {
        if (sum_mass > qc)
        {
            result.factor = qc / sum_mass;
        }
        if (sum_number > nc)
        {
            result.factor = std::min(result.factor, nc / sum_number);
        }
        if (result.factor < 1)
        {
            result.factor *= 1 - std::ldexp(1.0L, -40);
        }
    }
    result.cloud_increment = 0;
    result.cloud_number_increment = 0;
    result.vapor_increment = 0;
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
        result.diffusion[index] = result.environment[index] - result.feedback * (result.collected[index] + rain_collection[index]);
        result.diffusion_number[index] = source.mass_per_kg > 0
            ? source.number_per_kg * std::min(result.diffusion[index], 0.0L) / source.mass_per_kg : 0;
        result.z_increment[index] = 0;
        if (source.mass_per_kg > 0)
        {
            const long double power = 6 / riming::collection::frozen::mass_exponent[index];
            result.z_increment[index] = source.reflectivity_per_kg *
                (power * (result.collected[index] + rain_collection[index] + result.diffusion[index]) / source.mass_per_kg
                    + (1 - power) * result.diffusion_number[index] / source.number_per_kg);
        }
        result.mass_increment[index] = result.collected[index] + rain_collection[index] + result.diffusion[index];
        result.number_increment[index] = result.diffusion_number[index];
        result.cloud_increment -= result.collected[index];
        result.cloud_number_increment -= result.cloud_number_loss[index];
        result.vapor_increment -= result.diffusion[index];
        result.splinter_number[index] = index < 3 ? yield * result.collected[index] : 0;
        result.splinter_mass[index] = 1e-12L * result.splinter_number[index];
        result.splinter_z[index] = 20e-24L * result.splinter_number[index] / (440 * 440);
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
    for (std::size_t moment = 0; moment < 3; ++moment)
    {
        result.admitted_freezing[moment] = result.factor * result.raw_freezing[moment];
    }
    result.cloud_increment -= result.admitted_freezing[0];
    result.cloud_number_increment -= result.admitted_freezing[1];
    result.mass_increment[0] += result.admitted_freezing[0];
    result.number_increment[0] += result.admitted_freezing[1];
    result.z_increment[0] += result.admitted_freezing[2];
    const auto& ice = cell.initial.frozen[0];
    constexpr long double cg = 209.439510239L;
    if (ice.mass_per_kg > 0)
    {
        result.conversion_mass = 2 * std::max(result.collected[0] - std::max(result.diffusion[0], 0.0L), 0.0L);
        const long double fraction = result.conversion_mass / ice.mass_per_kg;
        result.ice_number_loss = fraction * ice.number_per_kg;
        result.ice_z_loss = fraction * ice.reflectivity_per_kg;
        result.graupel_number_gain = result.conversion_mass / 1.6e-10L;
        result.graupel_z_gain = 20 * 1.6e-10L * result.conversion_mass / (cg * cg);
        result.mass_increment[0] -= result.conversion_mass;
        result.mass_increment[2] += result.conversion_mass;
        result.number_increment[0] -= result.ice_number_loss;
        result.number_increment[2] += result.graupel_number_gain;
        result.z_increment[0] -= result.ice_z_loss;
        result.z_increment[2] += result.graupel_z_gain;
    }
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
        result.snow_to_graupel_z = fraction * result.snow_mass_squared / (cg * cg);
        result.mass_increment[1] -= result.snow_conversion_mass;
        result.mass_increment[2] += result.snow_conversion_mass;
        result.number_increment[1] -= result.snow_conversion_number;
        result.number_increment[2] += result.snow_conversion_number;
        result.z_increment[1] -= result.snow_z_loss;
        result.z_increment[2] += result.snow_to_graupel_z;
    }
    result.temperature_increment = (-riming::collection::latent_heat_fusion * result.cloud_increment
        - riming::thermal::latent_heat_sublimation * result.vapor_increment) / riming::collection::heat_capacity;
    for (const auto mass : rain_collection)
        result.temperature_increment += riming::collection::latent_heat_fusion * mass / riming::collection::heat_capacity;
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
        const auto a = derivative(source, values);
        const auto b = derivative(source, stage(a, .5L));
        const auto c = derivative(source, stage(b, .5L));
        const auto d = derivative(source, stage(c, 1));
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

} // namespace milbrandt_cold_homogeneous_reference
