#pragma once

#include "milbrandt_cold_collected_sll_reference.hpp"

namespace milbrandt_cold_rain_sll_reference
{

namespace prefix = milbrandt_cold_collected_sll_reference;
namespace frozen = milbrandt_frozen_collection_reference;
namespace riming = milbrandt_cold_riming_reference;
using Cell = riming::Cell;
using State = std::array<long double, 19>;
inline constexpr long double rain_coefficient = 523.598775598L;

inline frozen::Population rain_population(double mass, double number, long double alpha)
{
    const long double lambda = std::cbrt(rain_coefficient * number * frozen::gamma_ratio(alpha, 3) / mass);
    return {mass, number, static_cast<double>(number * frozen::gamma_ratio(alpha, 6) / std::pow(lambda, 6))};
}

inline frozen::Spectrum rain_spectrum(const frozen::Population& population, long double density)
{
    if (population.mass_per_kg == 0) return {};
    const long double ratio = static_cast<long double>(population.reflectivity_per_kg) * population.number_per_kg *
        rain_coefficient * rain_coefficient / (static_cast<long double>(population.mass_per_kg) * population.mass_per_kg);
    long double lo = -1 + 1e-9L, hi = 1e6L;
    for (int step = 0; step < 160; ++step)
    {
        const auto mid = (lo + hi) / 2;
        if (frozen::gamma_ratio(mid, 6) / std::pow(frozen::gamma_ratio(mid, 3), 2) > ratio) lo = mid;
        else hi = mid;
    }
    frozen::Spectrum result;
    result.alpha = (lo + hi) / 2;
    result.lambda = std::cbrt(rain_coefficient * population.number_per_kg *
        frozen::gamma_ratio(result.alpha, 3) / population.mass_per_kg);
    result.velocity = std::sqrt(1.225L / density) * 149.1L * frozen::moment(result, 3.5L) / frozen::moment(result, 3);
    return result;
}

struct Reference : prefix::Reference
{
    std::array<frozen::Channel, 4> rain;
    long double rain_factor = 1;
    long double rain_mass = 0, rain_number = 0, rain_z = 0;
};

inline Reference evaluate(const Cell& cell, long double duration, bool hail = true,
                          bool instantaneous = false, bool quadrature = false)
{
    Reference result;
    const auto& rain = cell.initial.rain;
    const auto spectrum = rain_spectrum(rain, cell.dry_density_kg_m3);
    const auto moment = [&](const frozen::Spectrum& psd, long double order)
    {
        return quadrature ? frozen::quadrature_moment(psd, order) : frozen::moment(psd, order);
    };
    long double mass = 0, number = 0;
    for (std::size_t species = 0; species < 4; ++species)
    {
        const auto& collector = cell.initial.frozen[species];
        if (rain.mass_per_kg <= 1e-14 || collector.mass_per_kg <= 1e-14) continue;
        const auto psd = frozen::spectrum(collector, species, cell.dry_density_kg_m3);
        const long double relative = std::sqrt(std::pow(spectrum.velocity - psd.velocity, 2)
            + .04L * spectrum.velocity * psd.velocity);
        const long double kernel = duration * std::acos(-1.0L) / 4 * cell.dry_density_kg_m3 *
            rain.number_per_kg * collector.number_per_kg * relative;
        auto& channel = result.rain[species];
        for (int term = 0; term <= 2; ++term)
        {
            const long double common = kernel * (term == 1 ? 2 : 1) * moment(psd, 2 - term);
            channel.raw_mass += common * rain_coefficient * moment(spectrum, 3 + term);
            channel.raw_number += common * moment(spectrum, term);
        }
        mass += channel.raw_mass;
        number += channel.raw_number;
    }
    if (!instantaneous)
    {
        if (mass > rain.mass_per_kg) result.rain_factor = rain.mass_per_kg / mass;
        if (number > rain.number_per_kg) result.rain_factor = std::min(result.rain_factor, rain.number_per_kg / number);
        if (result.rain_factor < 1) result.rain_factor *= 1 - std::ldexp(1.0L, -40);
    }
    std::array<long double, 4> accepted{};
    for (std::size_t species = 0; species < 4; ++species)
    {
        auto& channel = result.rain[species];
        channel.factor = result.rain_factor;
        channel.mass = channel.raw_mass * result.rain_factor;
        channel.number = channel.raw_number * result.rain_factor;
        accepted[species] = channel.mass;
        result.rain_mass -= channel.mass;
        result.rain_number -= channel.number;
    }
    static_cast<prefix::Reference&>(result) = prefix::evaluate(cell, duration, hail, instantaneous,
        quadrature, accepted, rain.mass_per_kg);
    if (rain.mass_per_kg > 0)
        result.rain_z = rain.reflectivity_per_kg * (2 * result.rain_mass / rain.mass_per_kg - result.rain_number / rain.number_per_kg);
    return result;
}

inline State state(const Cell& cell)
{
    State result{};
    const auto cold = riming::state(cell);
    std::copy(cold.begin(), cold.end(), result.begin());
    result[16] = cell.initial.rain.mass_per_kg;
    result[17] = cell.initial.rain.number_per_kg;
    result[18] = cell.initial.rain.reflectivity_per_kg;
    return result;
}

inline Cell from_state(const Cell& context, const State& values)
{
    riming::State cold{};
    std::copy_n(values.begin(), 16, cold.begin());
    auto cell = riming::from_state(context, cold);
    cell.initial.rain = {static_cast<double>(values[16]), static_cast<double>(values[17]), static_cast<double>(values[18])};
    return cell;
}

inline State derivative(const Cell& cell)
{
    const auto rate = evaluate(cell, 1, true, true);
    State result{rate.temperature_increment, rate.vapor_increment, rate.cloud_increment, rate.cloud_number_increment};
    for (std::size_t species = 0; species < 4; ++species)
    {
        result[4 + 3 * species] = rate.mass_increment[species];
        result[5 + 3 * species] = rate.number_increment[species];
        result[6 + 3 * species] = rate.z_increment[species];
    }
    result[16] = rate.rain_mass;
    result[17] = rate.rain_number;
    result[18] = rate.rain_z;
    return result;
}

inline State rk4(const Cell& source, long double duration, int steps)
{
    auto values = state(source);
    State roundoff{};
    const long double h = duration / steps;
    for (int step = 0; step < steps; ++step)
    {
        const auto stage = [&](const State& slope, long double fraction)
        {
            auto next = values;
            for (std::size_t index = 0; index < next.size(); ++index) next[index] += fraction * h * slope[index];
            return from_state(source, next);
        };
        const auto a = derivative(from_state(source, values));
        const auto b = derivative(stage(a, .5L));
        const auto c = derivative(stage(b, .5L));
        const auto d = derivative(stage(c, 1));
        for (std::size_t index = 0; index < values.size(); ++index)
        {
            const long double change = h / 6 * (a[index] + 2 * b[index] + 2 * c[index] + d[index]) - roundoff[index];
            const long double next = values[index] + change;
            roundoff[index] = (next - values[index]) - change;
            values[index] = next;
        }
    }
    return values;
}

} // namespace milbrandt_cold_rain_sll_reference
