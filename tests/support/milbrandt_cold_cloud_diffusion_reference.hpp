#pragma once

#include "microphysics/schemes/milbrandt/triple_cold_cloud_diffusion.hpp"
#include "milbrandt_cold_cloud_collection_reference.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>

namespace milbrandt_cold_cloud_diffusion_reference
{

namespace collection = milbrandt_cold_cloud_collection_reference;
using Cell = tmv::milbrandt_detail::ColdCloudDiffusionCell;
inline constexpr long double latent_heat_sublimation = 2834000.0L;
inline constexpr long double vapor_gas_constant = 461.5L;

struct Saturation
{
    long double mass = 0;
    long double derivative = 0;
};

inline Saturation saturation(long double temperature, long double pressure)
{
    const long double t = std::clamp(temperature, 150.0L, 400.0L) - 273.15L;
    const long double exponent = (23.036L - t / 333.7L) * t / (279.82L + t);
    const long double vapor_pressure = 611.15L * std::exp(exponent);
    const long double exponent_derivative = (23.036L * 279.82L
        - (2 * 279.82L * t + t * t) / 333.7L) / std::pow(279.82L + t, 2);
    const long double vapor_pressure_derivative = vapor_pressure * exponent_derivative;
    const long double p = std::clamp(pressure, 100.0L, 200000.0L);
    const long double denominator = std::max(p - vapor_pressure, 1.0L);
    const long double raw = .62198L * vapor_pressure / denominator;
    Saturation result;
    result.mass = std::clamp(raw, 0.0L, .2L);
    if (raw > 0 && raw < .2L)
    {
        result.derivative = p - vapor_pressure > 1
            ? .62198L * p * vapor_pressure_derivative / (denominator * denominator)
            : .62198L * vapor_pressure_derivative;
    }
    return result;
}

inline tmv::milbrandt_detail::ColdCloudCollectionCell collection_cell(const Cell& cell)
{
    return {cell.temperature_k, cell.pressure_pa, cell.dry_density_kg_m3, cell.initial};
}

struct Reference
{
    collection::Reference collection;
    Saturation saturation;
    long double saturation_ratio = 0;
    long double diffusivity = 0;
    long double conductivity = 0;
    long double viscosity = 0;
    long double schmidt_third = 0;
    long double density_factor = 0;
    long double resistance = 0;
    long double feedback = 0;
    std::array<long double, 3> thermal_lambda{};
    std::array<long double, 3> ventilation{};
    std::array<long double, 3> coefficients{};
    long double relaxation = 0;
    std::array<long double, 3> environmental{};
    std::array<long double, 3> riming{};
    std::array<long double, 3> diffusion{};
    std::array<long double, 3> mass_increment{};
    std::array<long double, 3> number_increment{};
    std::array<long double, 3> reflectivity_increment{};
    std::array<std::array<long double, 3>, 3> endpoint{};
    long double vapor_increment = 0;
    long double vapor_after = 0;
    long double temperature_increment = 0;
};

inline Reference evaluate(const Cell& cell, long double duration,
                          bool quadrature = false, bool instantaneous = false)
{
    Reference result;
    result.collection = collection::evaluate(collection_cell(cell), duration);
    if (instantaneous)
    {
        result.collection.cloud_mass_increment = 0;
        result.collection.cloud_number_increment = 0;
        for (auto& channel : result.collection.channels)
        {
            channel.factor = 1;
            channel.mass = channel.raw_mass;
            channel.number = channel.raw_number;
            result.collection.cloud_mass_increment -= channel.mass;
            result.collection.cloud_number_increment -= channel.number;
        }
    }
    const long double temperature = cell.temperature_k;
    const long double tc = temperature - 273.15L;
    const long double density = cell.dry_density_kg_m3;
    result.saturation = saturation(temperature, cell.pressure_pa);
    result.saturation_ratio = cell.vapor_mass_per_kg / result.saturation.mass;
    result.diffusivity = (2.2157e-5L + .0155e-5L * tc) * 1e5L / cell.pressure_pa;
    result.conductivity = .023971L + .000078L * tc;
    result.viscosity = collection::dynamic_viscosity(temperature) / density;
    result.schmidt_third = std::cbrt(result.viscosity / result.diffusivity);
    result.density_factor = std::sqrt(1.225L / density);
    const long double thermal_resistance = latent_heat_sublimation * latent_heat_sublimation
        / (result.conductivity * vapor_gas_constant * temperature * temperature);
    result.resistance = thermal_resistance + 1 / (density * result.saturation.mass * result.diffusivity);
    result.feedback = latent_heat_sublimation * collection::latent_heat_fusion
        / (result.conductivity * vapor_gas_constant * temperature * temperature * result.resistance);
    long double total_coefficient = 0;
    long double total_diffusion = 0;
    for (std::size_t species = 0; species < 3; ++species)
    {
        const auto& population = cell.initial.frozen[species];
        if (population.mass_per_kg == 0)
        {
            continue;
        }
        auto spectrum = result.collection.spectra[species];
        if (species == 0)
        {
            const long double coefficient = std::acos(-1.0L) * 100 / 6;
            spectrum.lambda = std::cbrt(coefficient * population.number_per_kg
                * collection::frozen::gamma_ratio(spectrum.alpha, 3) / population.mass_per_kg);
        }
        result.thermal_lambda[species] = spectrum.lambda;
        const long double fall_exponent = collection::frozen::fall_exponent[species + 1];
        const auto moment = [&](long double order)
        {
            return quadrature ? collection::frozen::quadrature_moment(spectrum, order)
                              : collection::frozen::moment(spectrum, order);
        };
        result.ventilation[species] = .78L * moment(1) + .30L * result.schmidt_third
            * std::sqrt(result.density_factor * collection::frozen::fall_coefficient[species + 1]
                / result.viscosity) * moment(1.5L + .5L * fall_exponent);
        const long double capacitance = species == 0 ? .5L : 1;
        result.coefficients[species] = capacitance * 2 * std::acos(-1.0L)
            * population.number_per_kg * result.ventilation[species]
            / (result.resistance * result.saturation.mass);
        result.environmental[species] = duration * result.coefficients[species]
            * (static_cast<long double>(cell.vapor_mass_per_kg) - result.saturation.mass);
        result.riming[species] = -result.feedback * result.collection.channels[species].mass;
        result.diffusion[species] = result.environmental[species] + result.riming[species];
        result.mass_increment[species] = result.collection.channels[species].mass + result.diffusion[species];
        result.number_increment[species] = population.number_per_kg
            * std::min(result.diffusion[species], 0.0L) / population.mass_per_kg;
        const long double power = 6 / collection::frozen::mass_exponent[species + 1];
        result.reflectivity_increment[species] = population.reflectivity_per_kg
            * (power * result.mass_increment[species] / population.mass_per_kg
                + (1 - power) * result.number_increment[species] / population.number_per_kg);
        result.endpoint[species] = {population.mass_per_kg + result.mass_increment[species],
            population.number_per_kg + result.number_increment[species],
            population.reflectivity_per_kg + result.reflectivity_increment[species]};
        total_coefficient += result.coefficients[species];
        total_diffusion += result.diffusion[species];
    }
    result.relaxation = duration * total_coefficient
        * (1 + result.saturation.derivative * latent_heat_sublimation / collection::heat_capacity);
    result.vapor_increment = -total_diffusion;
    result.vapor_after = cell.vapor_mass_per_kg + result.vapor_increment;
    result.temperature_increment = (-collection::latent_heat_fusion * result.collection.cloud_mass_increment
        + latent_heat_sublimation * total_diffusion) / collection::heat_capacity;
    return result;
}

using StateVector = std::array<long double, 13>;

inline StateVector state_vector(const Cell& cell)
{
    StateVector result{cell.temperature_k, cell.vapor_mass_per_kg,
        cell.initial.cloud_mass_per_kg, cell.initial.cloud_number_per_kg};
    for (std::size_t species = 0; species < 3; ++species)
    {
        result[4 + 3 * species] = cell.initial.frozen[species].mass_per_kg;
        result[5 + 3 * species] = cell.initial.frozen[species].number_per_kg;
        result[6 + 3 * species] = cell.initial.frozen[species].reflectivity_per_kg;
    }
    return result;
}

inline Cell cell_from_vector(const Cell& context, const StateVector& state)
{
    Cell result = context;
    result.temperature_k = static_cast<double>(state[0]);
    result.vapor_mass_per_kg = static_cast<double>(state[1]);
    result.initial.cloud_mass_per_kg = static_cast<double>(state[2]);
    result.initial.cloud_number_per_kg = static_cast<double>(state[3]);
    for (std::size_t species = 0; species < 3; ++species)
    {
        result.initial.frozen[species] = {static_cast<double>(state[4 + 3 * species]),
            static_cast<double>(state[5 + 3 * species]), static_cast<double>(state[6 + 3 * species])};
    }
    return result;
}

inline StateVector derivative(const Cell& context, const StateVector& state)
{
    const auto rate = evaluate(cell_from_vector(context, state), 1, false, true);
    StateVector result{rate.temperature_increment, rate.vapor_increment,
        rate.collection.cloud_mass_increment, rate.collection.cloud_number_increment};
    for (std::size_t species = 0; species < 3; ++species)
    {
        result[4 + 3 * species] = rate.mass_increment[species];
        result[5 + 3 * species] = rate.number_increment[species];
        result[6 + 3 * species] = rate.reflectivity_increment[species];
    }
    return result;
}

inline StateVector rk4(const Cell& source, long double duration, int steps)
{
    auto state = state_vector(source);
    const long double h = duration / steps;
    const auto stage = [&](const StateVector& slope, long double scale)
    {
        auto next = state;
        for (std::size_t index = 0; index < next.size(); ++index)
        {
            next[index] += h * scale * slope[index];
        }
        return next;
    };
    for (int step = 0; step < steps; ++step)
    {
        const auto k1 = derivative(source, state);
        const auto k2 = derivative(source, stage(k1, .5L));
        const auto k3 = derivative(source, stage(k2, .5L));
        const auto k4 = derivative(source, stage(k3, 1));
        for (std::size_t index = 0; index < state.size(); ++index)
        {
            state[index] += h / 6 * (k1[index] + 2 * k2[index] + 2 * k3[index] + k4[index]);
        }
    }
    return state;
}

} // namespace milbrandt_cold_cloud_diffusion_reference
