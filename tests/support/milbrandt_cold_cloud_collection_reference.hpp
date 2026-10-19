#pragma once

#include "microphysics/schemes/milbrandt/triple_cold_cloud_collection.hpp"
#include "milbrandt_frozen_collection_reference.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>

namespace milbrandt_cold_cloud_collection_reference
{

namespace frozen = milbrandt_frozen_collection_reference;
using Population = tmv::milbrandt_detail::FrozenCollectionPopulation;
using Cell = tmv::milbrandt_detail::ColdCloudCollectionCell;
inline constexpr long double cloud_mass_coefficient = 523.598775598L;
inline constexpr long double latent_heat_fusion = 334000.0L;
inline constexpr long double heat_capacity = 1004.0L;
inline constexpr long double snow_projected_area_factor = .6366L;

inline Population population(std::size_t species, double mass, double number,
                             long double alpha)
{
    return frozen::population(species + 1, mass, number, alpha);
}

inline long double cloud_inverse_slope(const Cell& cell)
{
    if (cell.initial.cloud_mass_per_kg == 0)
    {
        return 0;
    }
    return std::cbrt(static_cast<long double>(cell.initial.cloud_mass_per_kg)
        / (2 * cloud_mass_coefficient * cell.initial.cloud_number_per_kg));
}

inline long double cloud_moment(long double inverse_slope, long double order,
                               bool quadrature = false)
{
    const long double ratio = quadrature
        ? frozen::scaled_gamma_integral(1, order / 3)
            / frozen::scaled_gamma_integral(1, 0)
        : std::tgamma(2 + order / 3);
    return ratio * std::pow(inverse_slope, order);
}

inline long double cloud_mean_mass_diameter(const Cell& cell)
{
    if (cell.initial.cloud_mass_per_kg == 0)
    {
        return 0;
    }
    return std::cbrt(static_cast<long double>(cell.initial.cloud_mass_per_kg)
        / (cloud_mass_coefficient * cell.initial.cloud_number_per_kg));
}

inline long double collector_mean_mass_diameter(const Population& source, std::size_t species)
{
    return source.mass_per_kg == 0 ? 0 : std::pow(
        static_cast<long double>(source.mass_per_kg)
        / (frozen::mass_coefficient[species + 1] * source.number_per_kg),
        1 / frozen::mass_exponent[species + 1]);
}

inline long double dynamic_viscosity(long double temperature)
{
    return 1.72e-5L * 393 / (temperature + 120) * std::pow(temperature / 273.15L, 1.5L);
}

inline long double graupel_stokes_number(const Cell& cell, const frozen::Spectrum& spectrum)
{
    const long double cloud_diameter = cloud_mean_mass_diameter(cell);
    return 1000 * spectrum.velocity * cloud_diameter * cloud_diameter
        / (9 * dynamic_viscosity(cell.temperature_k)
            * collector_mean_mass_diameter(cell.initial.frozen[1], 1));
}

inline long double efficiency(const Cell& cell, const frozen::Spectrum& spectrum,
                              std::size_t species)
{
    if (cell.initial.cloud_mass_per_kg == 0 || cell.initial.frozen[species].mass_per_kg == 0)
    {
        return 0;
    }
    const long double cloud_diameter = cloud_mean_mass_diameter(cell);
    const long double collector_diameter = collector_mean_mass_diameter(cell.initial.frozen[species], species);
    long double result = 0;
    if (species == 0)
    {
        result = std::min(cloud_diameter, 30e-6L) * 3.333e4L
            * std::sqrt(std::min(collector_diameter, 1e-3L) * 1e3L);
    }
    else if (species == 1)
    {
        result = .55L * std::log10(2.51L * std::clamp(
            graupel_stokes_number(cell, spectrum), 1.5L, 10.0L));
    }
    else
    {
        result = std::exp(-8.68e-7L * std::pow(cloud_diameter, -1.6L) * collector_diameter);
    }
    return std::clamp(result, 0.0L, 1.0L);
}

struct Channel
{
    long double raw_mass = 0;
    long double raw_number = 0;
    long double mass = 0;
    long double number = 0;
    long double factor = 1;
};

struct Reference
{
    std::array<frozen::Spectrum, 3> spectra{};
    std::array<long double, 3> efficiencies{};
    std::array<Channel, 3> channels{};
    std::array<long double, 3> mass_increment{};
    std::array<long double, 3> reflectivity_increment{};
    std::array<std::array<long double, 3>, 3> endpoint{};
    long double cloud_inverse_lambda = 0;
    long double cloud_mass_increment = 0;
    long double cloud_number_increment = 0;
    long double cloud_mass_after = 0;
    long double cloud_number_after = 0;
    long double temperature_increment = 0;
};

inline Reference evaluate(const Cell& cell, long double duration,
                          bool quadrature = false, bool wet_hail = false)
{
    Reference result;
    result.cloud_inverse_lambda = cloud_inverse_slope(cell);
    const auto mc = [&](long double order)
    {
        return cloud_moment(result.cloud_inverse_lambda, order, quadrature);
    };
    long double total_mass = 0;
    long double total_number = 0;
    for (std::size_t species = 0; species < 3; ++species)
    {
        const auto& source = cell.initial.frozen[species];
        result.spectra[species] = frozen::spectrum(source, species + 1, cell.dry_density_kg_m3);
        if (cell.initial.cloud_mass_per_kg == 0 || source.mass_per_kg == 0)
        {
            continue;
        }
        if (!(cell.initial.cloud_mass_per_kg > 1e-14) || !(source.mass_per_kg > 1e-14))
        {
            continue;
        }
        result.efficiencies[species] = wet_hail && species == 2 ? 1 : efficiency(cell, result.spectra[species], species);
        const auto mx = [&](long double order)
        {
            return quadrature ? frozen::quadrature_moment(result.spectra[species], order)
                              : frozen::moment(result.spectra[species], order);
        };
        const long double b = frozen::fall_exponent[species + 1];
        const long double common = duration * std::acos(-1.0L) / 4
            * result.efficiencies[species] * std::sqrt(1.225L / cell.dry_density_kg_m3)
            * frozen::fall_coefficient[species + 1] * cell.dry_density_kg_m3
            * cell.initial.cloud_number_per_kg * source.number_per_kg
            * (species == 0 ? snow_projected_area_factor : 1);
        auto& channel = result.channels[species];
        channel.raw_mass = common * cloud_mass_coefficient
            * (mx(b + 2) * mc(3) + 2 * mx(b + 1) * mc(4) + mx(b) * mc(5));
        channel.raw_number = common * (mx(b + 2) + 2 * mx(b + 1) * mc(1) + mx(b) * mc(2));
        total_mass += channel.raw_mass;
        total_number += channel.raw_number;
    }
    long double factor = 1;
    if (total_mass > cell.initial.cloud_mass_per_kg)
    {
        factor = cell.initial.cloud_mass_per_kg / total_mass;
    }
    if (total_number > cell.initial.cloud_number_per_kg)
    {
        factor = std::min(factor, cell.initial.cloud_number_per_kg / total_number);
    }
    for (std::size_t species = 0; species < 3; ++species)
    {
        auto& channel = result.channels[species];
        channel.factor = factor;
        channel.mass = channel.raw_mass * factor;
        channel.number = channel.raw_number * factor;
        result.mass_increment[species] = channel.mass;
        const auto& source = cell.initial.frozen[species];
        if (source.mass_per_kg > 0)
        {
            result.reflectivity_increment[species] = 6 / frozen::mass_exponent[species + 1]
                * source.reflectivity_per_kg * channel.mass / source.mass_per_kg;
        }
        result.endpoint[species] = {source.mass_per_kg + channel.mass,
            source.number_per_kg, source.reflectivity_per_kg + result.reflectivity_increment[species]};
        result.cloud_mass_increment -= channel.mass;
        result.cloud_number_increment -= channel.number;
    }
    result.cloud_mass_after = cell.initial.cloud_mass_per_kg + result.cloud_mass_increment;
    result.cloud_number_after = cell.initial.cloud_number_per_kg + result.cloud_number_increment;
    result.temperature_increment = -latent_heat_fusion / heat_capacity * result.cloud_mass_increment;
    return result;
}

inline std::array<long double, 2> missing_fall_exponent_kernel(
    const Cell& cell, const Reference& expected, std::size_t species, long double duration)
{
    const auto& s = expected.spectra[species];
    const long double b = frozen::fall_exponent[species + 1];
    const auto wrong_moment = [&](int order)
    {
        return frozen::gamma_ratio(s.alpha, order) / std::pow(s.lambda, b + order);
    };
    const long double common = duration * std::acos(-1.0L) / 4
        * expected.efficiencies[species] * std::sqrt(1.225L / cell.dry_density_kg_m3)
        * frozen::fall_coefficient[species + 1] * cell.dry_density_kg_m3
        * cell.initial.cloud_number_per_kg * cell.initial.frozen[species].number_per_kg
        * (species == 0 ? snow_projected_area_factor : 1);
    const auto mc = [&](int order)
    {
        return cloud_moment(expected.cloud_inverse_lambda, order);
    };
    return {common * cloud_mass_coefficient
        * (wrong_moment(2) * mc(3) + 2 * wrong_moment(1) * mc(4) + wrong_moment(0) * mc(5)),
        common * (wrong_moment(2) + 2 * wrong_moment(1) * mc(1) + wrong_moment(0) * mc(2))};
}

} // namespace milbrandt_cold_cloud_collection_reference
