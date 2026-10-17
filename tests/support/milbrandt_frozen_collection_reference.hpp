#pragma once

#include "microphysics/schemes/milbrandt/triple_frozen_collection.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>

namespace milbrandt_frozen_collection_reference
{

using Population = tmv::milbrandt_detail::FrozenCollectionPopulation;
using State = tmv::milbrandt_detail::FrozenCollectionState;
using Cell = tmv::milbrandt_detail::FrozenCollectionCell;

inline constexpr std::array<long double, 4> mass_coefficient{
    440.0L, 0.1597L, 209.439510239L, 471.238898038L};
inline constexpr std::array<long double, 4> mass_exponent{3.0L, 2.078L, 3.0L, 3.0L};
inline constexpr std::array<long double, 4> fall_coefficient{71.34L, 11.72L, 19.3L, 206.89L};
inline constexpr std::array<long double, 4> fall_exponent{0.6635L, 0.41L, 0.37L, 0.6384L};

inline long double gamma_ratio(long double alpha, long double order)
{
    if (order == std::floor(order) && order >= 0 && order <= 8)
    {
        long double value = 1;
        for (int factor = 1; factor <= static_cast<int>(order); ++factor)
        {
            value *= alpha + factor;
        }
        return value;
    }
    return std::exp(std::lgamma(alpha + 1 + order) - std::lgamma(alpha + 1));
}

inline long double log_shape_ratio(long double alpha, std::size_t species)
{
    const long double d = mass_exponent[species];
    return std::log(gamma_ratio(alpha, 6)) - 6 / d * std::log(gamma_ratio(alpha, d));
}

inline Population population(std::size_t species, double mass, double number,
                             long double alpha)
{
    if (mass == 0 && number == 0)
    {
        return {};
    }
    const long double d = mass_exponent[species];
    const long double lambda = std::pow(mass_coefficient[species] * number
        * gamma_ratio(alpha, d) / mass, 1 / d);
    return {mass, number, static_cast<double>(number * gamma_ratio(alpha, 6)
                                             / std::pow(lambda, 6))};
}

inline long double represented_alpha(const Population& population, std::size_t species)
{
    const long double d = mass_exponent[species];
    const long double log_ratio = std::log(static_cast<long double>(population.reflectivity_per_kg)
        / population.number_per_kg) - 6 / d * std::log(
            static_cast<long double>(population.mass_per_kg)
            / (mass_coefficient[species] * population.number_per_kg));
    long double lower = -1 + 1e-9L;
    long double upper = 1e6L;
    for (int iteration = 0; iteration < 160; ++iteration)
    {
        const long double middle = (lower + upper) / 2;
        if (log_shape_ratio(middle, species) > log_ratio)
        {
            lower = middle;
        }
        else
        {
            upper = middle;
        }
    }
    return (lower + upper) / 2;
}

inline long double scaled_gamma_integral(long double alpha, long double order)
{
    constexpr int intervals = 12000;
    const long double upper = std::max(120.0L,
        alpha + 1 + 30 * std::sqrt(alpha + 1) + 80);
    const long double step = std::pow(upper, .25L) / intervals;
    const long double shift = alpha > 0 ? alpha * std::log(alpha) - alpha : 0;
    long double sum = 0;
    for (int index = 1; index <= intervals; ++index)
    {
        const long double u = index * step;
        const long double x = u * u * u * u;
        const long double integrand = 4 * std::exp(
            (4 * (alpha + order) + 3) * std::log(u) - x - shift);
        sum += (index == intervals ? 1 : index % 2 == 0 ? 2 : 4) * integrand;
    }
    return sum * step / 3;
}

struct Spectrum
{
    long double alpha = 0;
    long double lambda = 0;
    long double velocity = 0;
};

inline Spectrum spectrum(const Population& source, std::size_t species,
                         long double density)
{
    if (source.mass_per_kg == 0)
    {
        return {};
    }
    Spectrum result;
    result.alpha = represented_alpha(source, species);
    const long double d = mass_exponent[species];
    result.lambda = std::pow(mass_coefficient[species] * source.number_per_kg
        * gamma_ratio(result.alpha, d) / source.mass_per_kg, 1 / d);
    result.velocity = std::sqrt(1.225L / density) * fall_coefficient[species]
        * gamma_ratio(result.alpha, d + fall_exponent[species])
        / gamma_ratio(result.alpha, d)
        / std::pow(result.lambda, fall_exponent[species]);
    return result;
}

inline long double moment(const Spectrum& spectrum, long double order)
{
    return gamma_ratio(spectrum.alpha, order) / std::pow(spectrum.lambda, order);
}

inline long double quadrature_moment(const Spectrum& spectrum, long double order)
{
    return scaled_gamma_integral(spectrum.alpha, order)
        / scaled_gamma_integral(spectrum.alpha, 0)
        / std::pow(spectrum.lambda, order);
}

struct Channel
{
    long double raw_mass = 0;
    long double raw_number = 0;
    long double mass = 0;
    long double number = 0;
    long double factor = 1;
};

inline Channel pair_channel(const Cell& cell, const std::array<Spectrum, 4>& spectra,
                            std::size_t donor, std::size_t collector,
                            long double duration, bool quadrature = false, bool wet_hail = false)
{
    Channel result;
    const auto& source = cell.initial.frozen[donor];
    const auto& destination = cell.initial.frozen[collector];
    if (!(source.mass_per_kg > 1e-14) || !(destination.mass_per_kg > 1e-14))
    {
        return result;
    }
    const auto diameter_moment = [&](std::size_t species, long double order)
    {
        return quadrature ? quadrature_moment(spectra[species], order)
                          : moment(spectra[species], order);
    };
    const long double vd = spectra[donor].velocity;
    const long double vc = spectra[collector].velocity;
    const long double relative_velocity = std::sqrt((vd - vc) * (vd - vc) + .04L * vd * vc);
    const long double efficiency = wet_hail && collector == 3 ? 1 : std::min(1.0L,
        (donor == 0 && collector == 1 ? .05L : .01L)
        * std::exp(.1L * (static_cast<long double>(cell.temperature_k) - 273.15L)));
    const long double common = duration * std::acos(-1.0L) / 4 * efficiency
        * relative_velocity * cell.dry_density_kg_m3
        * source.number_per_kg * destination.number_per_kg;
    const long double d = mass_exponent[donor];
    const long double c1 = diameter_moment(collector, 1);
    const long double c2 = diameter_moment(collector, 2);
    result.raw_mass = common * mass_coefficient[donor]
        * (c2 * diameter_moment(donor, d)
           + 2 * c1 * diameter_moment(donor, d + 1)
           + diameter_moment(donor, d + 2));
    result.raw_number = common * (c2 + 2 * c1 * diameter_moment(donor, 1)
                                  + diameter_moment(donor, 2));
    result.mass = result.raw_mass;
    result.number = result.raw_number;
    return result;
}

struct Reference
{
    std::array<Spectrum, 4> spectra{};
    std::array<Channel, 5> channels{};
    std::array<long double, 4> mass_increment{};
    std::array<long double, 4> number_increment{};
    std::array<long double, 4> reflectivity_increment{};
    std::array<std::array<long double, 3>, 4> endpoint{};
};

inline Reference evaluate(const Cell& cell, long double duration,
                          bool quadrature = false, bool wet_hail = false)
{
    Reference result;
    for (std::size_t species = 0; species < 4; ++species)
    {
        result.spectra[species] = spectrum(cell.initial.frozen[species], species,
                                            cell.dry_density_kg_m3);
    }
    for (std::size_t channel = 0; channel < 3; ++channel)
    {
        result.channels[channel] = pair_channel(cell, result.spectra, 0,
                                                 channel + 1, duration, quadrature, wet_hail);
    }
    result.channels[3] = pair_channel(cell, result.spectra, 1, 3, duration, quadrature, wet_hail);
    const auto& snow = cell.initial.frozen[1];
    if (snow.mass_per_kg > 1e-14)
    {
        const long double density = cell.dry_density_kg_m3;
        const long double efficiency = std::min(1.0L,
            .05L * std::exp(.1L * (static_cast<long double>(cell.temperature_k) - 273.15L)));
        result.channels[4].raw_number = std::min(.5L * snow.number_per_kg,
            duration * .93952L * efficiency
            * std::pow(density * snow.mass_per_kg, (2 + fall_exponent[1]) / 3)
            * std::pow(density * snow.number_per_kg, (4 - fall_exponent[1]) / 3) / density);
        result.channels[4].number = result.channels[4].raw_number;
    }
    const long double ice_mass_sink = result.channels[0].raw_mass
        + result.channels[1].raw_mass + result.channels[2].raw_mass;
    const long double ice_number_sink = result.channels[0].raw_number
        + result.channels[1].raw_number + result.channels[2].raw_number;
    const auto& ice = cell.initial.frozen[0];
    long double ice_factor = 1;
    if (ice_mass_sink > ice.mass_per_kg)
    {
        ice_factor = ice.mass_per_kg / ice_mass_sink;
    }
    if (ice_number_sink > ice.number_per_kg)
    {
        ice_factor = std::min(ice_factor, ice.number_per_kg / ice_number_sink);
    }
    for (std::size_t channel = 0; channel < 3; ++channel)
    {
        auto& event = result.channels[channel];
        event.factor = ice_factor;
        event.mass *= ice_factor;
        event.number *= ice_factor;
    }
    const long double available_snow_mass = snow.mass_per_kg + (wet_hail ? 0 : result.channels[0].mass);
    const long double snow_number_sink = result.channels[3].raw_number
        + result.channels[4].raw_number;
    long double snow_factor = 1;
    if (result.channels[3].raw_mass > available_snow_mass)
    {
        snow_factor = available_snow_mass / result.channels[3].raw_mass;
    }
    if (snow_number_sink > snow.number_per_kg)
    {
        snow_factor = std::min(snow_factor, snow.number_per_kg / snow_number_sink);
    }
    for (std::size_t channel = 3; channel < 5; ++channel)
    {
        auto& event = result.channels[channel];
        event.factor = snow_factor;
        event.mass *= snow_factor;
        event.number *= snow_factor;
    }
    for (std::size_t channel = 0; channel < 4; ++channel)
    {
        const std::size_t donor = channel == 3 ? 1 : 0;
        const std::size_t collector = channel == 3 ? 3 : channel + 1;
        result.mass_increment[donor] -= result.channels[channel].mass;
        result.mass_increment[collector] += result.channels[channel].mass;
        result.number_increment[donor] -= result.channels[channel].number;
    }
    result.number_increment[1] -= result.channels[4].number;
    for (std::size_t species = 0; species < 4; ++species)
    {
        const auto& source = cell.initial.frozen[species];
        const long double exponent = 6 / mass_exponent[species];
        if (source.mass_per_kg > 0)
        {
            result.reflectivity_increment[species] = source.reflectivity_per_kg
                * (exponent * result.mass_increment[species] / source.mass_per_kg
                   + (1 - exponent) * result.number_increment[species] / source.number_per_kg);
        }
        result.endpoint[species] = {
            source.mass_per_kg + result.mass_increment[species],
            source.number_per_kg + result.number_increment[species],
            source.reflectivity_per_kg + result.reflectivity_increment[species]};
    }
    return result;
}

} // namespace milbrandt_frozen_collection_reference
