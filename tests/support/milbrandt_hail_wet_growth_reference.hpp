#pragma once

#include "microphysics/schemes/milbrandt/triple_hail_wet_growth.hpp"
#include "milbrandt_rain_collision_reference.hpp"

namespace milbrandt_hail_wet_growth_reference
{

namespace collision = milbrandt_rain_collision_reference;
namespace frozen = milbrandt_frozen_collection_reference;
namespace cold = milbrandt_cold_rain_sll_reference;
using Cell = tmv::milbrandt_detail::ColdRimingCell;
using WetCell = tmv::milbrandt_detail::HailWetGrowthCell;

struct Thermal
{
    bool wet = false;
    bool unsupported = false;
    long double ventilation = 0, atmosphere = 0, frozen = 0, capacity = 0;
    long double retained = 0, shed = 0, number = 0, z = 0;
};

inline Thermal thermal(const Cell& source, long double dt,
                       const std::array<long double, 4>& dry, const std::array<long double, 4>& wet,
                       bool quadrature = false)
{
    Thermal result;
    result.retained = dry[0] + dry[1];
    const auto& hail = source.initial.frozen[3];
    const long double t = source.temperature_k;
    const long double tc = t - static_cast<long double>(273.15);
    if (hail.mass_per_kg == 0 || tc <= -40) return result;
    const auto spectrum = frozen::spectrum(hail, 3, source.dry_density_kg_m3);
    const auto moment = [&](long double order)
    { return quadrature ? frozen::quadrature_moment(spectrum, order) : frozen::moment(spectrum, order); };
    const long double diffusivity = (2.2157e-5L + .0155e-5L * tc) * 1e5L / source.pressure_pa;
    const long double conductivity = .023971L + .000078L * tc;
    const long double viscosity = 1.72e-5L * 393 / (t + 120) * std::pow(t / 273.15L, 1.5L) / source.dry_density_kg_m3;
    const long double density_factor = std::sqrt(1.225L / source.dry_density_kg_m3);
    result.ventilation = .78L * moment(1) + .30L * std::cbrt(viscosity / diffusivity) *
        std::sqrt(density_factor * 206.89L / viscosity) * moment(1.5L + .5L * .6384L);
    const long double saturation = std::clamp(.62198L * 611.21L / std::max(static_cast<long double>(source.pressure_pa) - 611.21L, 1.L), 0.L, .2L);
    const long double denominator = 334000 + 4218 * tc;
    result.atmosphere = dt * 2 * std::acos(-1.L) * hail.number_per_kg * result.ventilation / denominator *
        (source.dry_density_kg_m3 * 2500000.L * diffusivity * (saturation - source.initial.vapor_mass_per_kg) - conductivity * tc);
    const long double solids = wet[2] + wet[3];
    result.frozen = solids * (1 - 2115.3L * tc / denominator);
    result.capacity = std::max(0.L, result.atmosphere + result.frozen);
    result.wet = result.capacity < dry[0] + dry[1] + dry[2] + dry[3];
    if (result.wet)
    {
        result.unsupported = result.capacity < solids;
        result.retained = std::min(wet[0] + wet[1], std::max(0.L, result.capacity - solids));
        result.shed = wet[0] + wet[1] - result.retained;
        result.number = result.shed / (523.598775598L * 1e-9L);
        if (result.shed > 0)
        {
            const long double lambda = std::cbrt(frozen::gamma_ratio(2, 3) / 1e-9L);
            const long double sixth = quadrature
                ? frozen::scaled_gamma_integral(2, 6) / frozen::scaled_gamma_integral(2, 0)
                : frozen::gamma_ratio(2, 6);
            result.z = result.number * sixth / std::pow(lambda, 6);
        }
    }
    return result;
}

struct Reference : cold::Reference
{
    Thermal thermal;
    std::array<long double, 4> dry_collected{}, wet_collected{};
};

inline Reference evaluate(const Cell& cell, long double dt, bool hail = true)
{
    const collision::Cell collision_cell{{cell.temperature_k, cell.pressure_pa, cell.dry_density_kg_m3, {cell.initial.frozen}}, cell.initial.rain};
    const auto pairs = collision::evaluate(collision_cell, dt, hail);
    const auto dry = collision::evaluate_coupled(cell, dt, hail);
    const bool eligible = cell.initial.frozen[3].mass_per_kg > 0 &&
        static_cast<long double>(cell.temperature_k) - static_cast<long double>(273.15) > -40;
    const auto wet = eligible ? collision::evaluate_coupled(cell, dt, hail, true) : dry;
    Reference result;
    result.dry_collected = {dry.collected[3], pairs.pairs[3].rain_mass, dry.collection.channels[2].mass, dry.collection.channels[3].mass};
    result.wet_collected = {wet.collected[3], pairs.pairs[3].rain_mass, wet.collection.channels[2].mass, wet.collection.channels[3].mass};
    result.thermal = thermal(cell, dt, result.dry_collected, result.wet_collected);
    static_cast<cold::Reference&>(result) = result.thermal.wet ? wet : dry;
    if (result.thermal.wet)
    {
        const auto& hail_source = cell.initial.frozen[3];
        const long double diffusion = result.environment[3] - result.feedback * result.thermal.retained;
        const long double number = hail_source.number_per_kg * std::min(diffusion, 0.L) / hail_source.mass_per_kg;
        const long double delta_q = diffusion - result.diffusion[3];
        const long double delta_n = number - result.diffusion_number[3];
        result.mass_increment[3] += delta_q - result.thermal.shed;
        result.number_increment[3] += delta_n;
        result.z_increment[3] += hail_source.reflectivity_per_kg *
            (2 * (delta_q - result.thermal.shed) / hail_source.mass_per_kg - delta_n / hail_source.number_per_kg);
        result.vapor_increment -= delta_q;
        result.diffusion[3] = diffusion;
        result.diffusion_number[3] = number;
        result.rain_mass += result.thermal.shed;
        result.rain_number += result.thermal.number;
        result.rain_z += result.thermal.z;
        long double frozen_liquid = result.admitted_freezing[0] + result.thermal.retained;
        for (std::size_t index = 0; index < 3; ++index) frozen_liquid += result.collected[index] + pairs.pairs[index].rain_mass;
        result.temperature_increment = (334000.L * frozen_liquid - 2834000.L * result.vapor_increment) / 1004;
    }
    return result;
}

inline cold::State derivative(const Cell& cell)
{
    const auto rate = evaluate(cell, 1);
    cold::State result{rate.temperature_increment, rate.vapor_increment, rate.cloud_increment, rate.cloud_number_increment};
    for (std::size_t index = 0; index < 4; ++index)
    {
        result[4 + 3 * index] = rate.mass_increment[index];
        result[5 + 3 * index] = rate.number_increment[index];
        result[6 + 3 * index] = rate.z_increment[index];
    }
    result[16] = rate.rain_mass;
    result[17] = rate.rain_number;
    result[18] = rate.rain_z;
    return result;
}

inline cold::State rk4(const Cell& cell, long double duration, int steps)
{
    auto values = cold::state(cell);
    cold::State roundoff{};
    const auto h = duration / steps;
    for (int step = 0; step < steps; ++step)
    {
        const auto stage = [&](const cold::State& slope, long double fraction)
        {
            auto next = values;
            for (std::size_t index = 0; index < next.size(); ++index) next[index] += fraction * h * slope[index];
            return derivative(cold::from_state(cell, next));
        };
        const auto a = derivative(cold::from_state(cell, values));
        const auto b = stage(a, .5L), c = stage(b, .5L), d = stage(c, 1);
        for (std::size_t index = 0; index < values.size(); ++index)
        {
            const auto change = h / 6 * (a[index] + 2 * b[index] + 2 * c[index] + d[index]) - roundoff[index];
            const auto next = values[index] + change;
            roundoff[index] = (next - values[index]) - change;
            values[index] = next;
        }
    }
    return values;
}

} // namespace milbrandt_hail_wet_growth_reference
