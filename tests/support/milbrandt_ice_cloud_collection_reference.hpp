#pragma once

#include "microphysics/schemes/milbrandt/triple_ice_cloud_collection.hpp"
#include "milbrandt_cold_cloud_collection_reference.hpp"

namespace milbrandt_ice_cloud_collection_reference
{

namespace cloud = milbrandt_cold_cloud_collection_reference;
using Cell = tmv::milbrandt_detail::IceCloudCollectionCell;

struct Reference
{
    cloud::frozen::Spectrum spectrum;
    long double cloud_inverse_lambda = 0;
    long double raw_mass = 0;
    long double raw_number = 0;
    long double factor = 1;
    long double mass = 0;
    long double number = 0;
    long double reflectivity_increment = 0;
    long double temperature_increment = 0;
};

inline Reference evaluate(const Cell& cell, long double duration,
                          bool quadrature = false, bool instantaneous = false)
{
    Reference result;
    result.spectrum = cloud::frozen::spectrum(cell.initial.ice, 0, cell.dry_density_kg_m3);
    if (cell.initial.cloud_mass_per_kg > 0)
    {
        result.cloud_inverse_lambda = std::cbrt(
            static_cast<long double>(cell.initial.cloud_mass_per_kg)
            / (2 * cloud::cloud_mass_coefficient * cell.initial.cloud_number_per_kg));
    }
    if (cell.initial.cloud_mass_per_kg > 1e-14 && cell.initial.ice.mass_per_kg > 1e-14)
    {
        const auto mc = [&](int order)
        {
            return cloud::cloud_moment(result.cloud_inverse_lambda, order, quadrature);
        };
        const auto mi = [&](long double order)
        {
            return quadrature ? cloud::frozen::quadrature_moment(result.spectrum, order)
                              : cloud::frozen::moment(result.spectrum, order);
        };
        constexpr long double b = .6635L;
        const long double common = duration * std::acos(-1.0L) / 4
            * std::sqrt(1.225L / cell.dry_density_kg_m3) * 71.34L
            * cell.dry_density_kg_m3 * cell.initial.cloud_number_per_kg
            * cell.initial.ice.number_per_kg;
        result.raw_mass = common * cloud::cloud_mass_coefficient
            * (mi(b + 2) * mc(3) + 2 * mi(b + 1) * mc(4) + mi(b) * mc(5));
        result.raw_number = common * (mi(b + 2) + 2 * mi(b + 1) * mc(1) + mi(b) * mc(2));
        if (!instantaneous && result.raw_mass > cell.initial.cloud_mass_per_kg)
        {
            result.factor = cell.initial.cloud_mass_per_kg / result.raw_mass;
        }
        if (!instantaneous && result.raw_number > cell.initial.cloud_number_per_kg)
        {
            result.factor = std::min(result.factor,
                cell.initial.cloud_number_per_kg / result.raw_number);
        }
    }
    result.mass = result.raw_mass * result.factor;
    result.number = result.raw_number * result.factor;
    if (cell.initial.ice.mass_per_kg > 0)
    {
        result.reflectivity_increment = 2 * cell.initial.ice.reflectivity_per_kg
            * result.mass / cell.initial.ice.mass_per_kg;
    }
    result.temperature_increment = cloud::latent_heat_fusion / cloud::heat_capacity * result.mass;
    return result;
}

using Vector = std::array<long double, 6>;

inline Vector vector(const Cell& cell)
{
    return {cell.temperature_k, cell.initial.cloud_mass_per_kg,
        cell.initial.cloud_number_per_kg, cell.initial.ice.mass_per_kg,
        cell.initial.ice.number_per_kg, cell.initial.ice.reflectivity_per_kg};
}

inline Cell from_vector(const Cell& context, const Vector& state)
{
    auto cell = context;
    cell.temperature_k = static_cast<double>(state[0]);
    cell.initial.cloud_mass_per_kg = static_cast<double>(state[1]);
    cell.initial.cloud_number_per_kg = static_cast<double>(state[2]);
    cell.initial.ice = {static_cast<double>(state[3]), static_cast<double>(state[4]),
        static_cast<double>(state[5])};
    return cell;
}

inline Vector derivative(const Cell& context, const Vector& state)
{
    const auto rate = evaluate(from_vector(context, state), 1, false, true);
    return {rate.temperature_increment, -rate.mass, -rate.number, rate.mass, 0,
        rate.reflectivity_increment};
}

inline Vector rk4(const Cell& source, long double duration, int steps)
{
    auto state = vector(source);
    const long double h = duration / steps;
    const auto stage = [&](const Vector& slope, long double scale)
    {
        auto candidate = state;
        for (std::size_t index = 0; index < state.size(); ++index)
        {
            candidate[index] += h * scale * slope[index];
        }
        return candidate;
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

} // namespace milbrandt_ice_cloud_collection_reference
