#pragma once

#include "microphysics/schemes/milbrandt/triple_rain_collision.hpp"
#include "milbrandt_cold_rain_sll_reference.hpp"

namespace milbrandt_rain_collision_reference
{

namespace frozen = milbrandt_frozen_collection_reference;
namespace rain = milbrandt_cold_rain_sll_reference;
using Cell = tmv::milbrandt_detail::RainCollectionCell;
using Population = frozen::Population;

struct Pair
{
    long double raw_rain_mass = 0, raw_frozen_mass = 0, raw_number = 0;
    long double factor = 1;
    long double rain_mass = 0, frozen_mass = 0, number = 0;
    long double rain_diameter = 0, frozen_diameter = 0, mixed_density = 0, destination_density = 0;
    std::size_t destination = 2;
    long double product_mass = 0, product_number = 0, product_z = 0;
};

struct Reference
{
    std::array<Pair, 4> pairs{};
    long double rain_factor = 1;
    std::array<long double, 5> mass{}, number{}, z{};
    long double temperature = 0;
};

inline Reference evaluate(const Cell& cell, long double dt, bool hail = true, bool quadrature = false, bool instantaneous = false)
{
    Reference result;
    const auto r = rain::rain_spectrum(cell.rain, cell.frozen.dry_density_kg_m3);
    const auto moment = [&](const frozen::Spectrum& s, long double order)
    { return quadrature ? frozen::quadrature_moment(s, order) : frozen::moment(s, order); };
    long double rain_mass = 0, rain_number = 0;
    for (std::size_t species = 0; species < 4; ++species)
    {
        const auto& population = cell.frozen.initial.frozen[species];
        auto& pair = result.pairs[species];
        if (population.mass_per_kg <= 1e-14 || cell.rain.mass_per_kg <= 1e-14) continue;
        const auto f = frozen::spectrum(population, species, cell.frozen.dry_density_kg_m3);
        const long double common = dt * std::acos(-1.L) / 4 * cell.frozen.dry_density_kg_m3 *
            cell.rain.number_per_kg * population.number_per_kg * std::sqrt(std::pow(r.velocity - f.velocity, 2) + .04L * r.velocity * f.velocity);
        for (int term = 0; term < 3; ++term)
        {
            const long double coefficient = common * (term == 1 ? 2 : 1);
            pair.raw_rain_mass += coefficient * rain::rain_coefficient * moment(r, 3 + term) * moment(f, 2 - term);
            pair.raw_number += coefficient * moment(r, term) * moment(f, 2 - term);
            if (species < 3)
                pair.raw_frozen_mass += coefficient * frozen::mass_coefficient[species] *
                    moment(r, 2 - term) * moment(f, frozen::mass_exponent[species] + term);
        }
        rain_mass += pair.raw_rain_mass;
        rain_number += pair.raw_number;
    }
    if (!instantaneous)
    {
        if (rain_mass > cell.rain.mass_per_kg) result.rain_factor = cell.rain.mass_per_kg / rain_mass;
        if (rain_number > cell.rain.number_per_kg) result.rain_factor = std::min(result.rain_factor, cell.rain.number_per_kg / rain_number);
        if (result.rain_factor < 1) result.rain_factor *= 1 - std::ldexp(1.L, -40);
    }
    for (std::size_t species = 0; species < 4; ++species)
    {
        auto& pair = result.pairs[species];
        if (pair.raw_rain_mass == 0) continue;
        const auto& population = cell.frozen.initial.frozen[species];
        long double factor = 1;
        if (!instantaneous && species < 3)
        {
            if (pair.raw_frozen_mass > population.mass_per_kg) factor = population.mass_per_kg / pair.raw_frozen_mass;
            if (pair.raw_number > population.number_per_kg) factor = std::min(factor, population.number_per_kg / pair.raw_number);
            if (factor < 1) factor *= 1 - std::ldexp(1.L, -40);
        }
        pair.factor = std::min(result.rain_factor, factor);
        pair.rain_mass = pair.raw_rain_mass * pair.factor;
        pair.frozen_mass = pair.raw_frozen_mass * pair.factor;
        pair.number = pair.raw_number * pair.factor;
        result.mass[0] -= pair.rain_mass;
        result.number[0] -= pair.number;
        if (species == 3)
        {
            result.mass[4] += pair.rain_mass;
            result.z[4] += 2 * population.reflectivity_per_kg * pair.rain_mass / population.mass_per_kg;
            continue;
        }
        pair.rain_diameter = std::cbrt(cell.rain.mass_per_kg / (rain::rain_coefficient * cell.rain.number_per_kg));
        pair.frozen_diameter = std::pow(population.mass_per_kg /
            (frozen::mass_coefficient[species] * population.number_per_kg), 1 / frozen::mass_exponent[species]);
        const long double diameter = std::max(pair.rain_diameter, pair.frozen_diameter);
        const long double density = species == 0 ? 500.L : species == 2 ? 400.L :
            std::min(500.L, 6 * frozen::mass_coefficient[species] / std::acos(-1.L) *
                std::pow(pair.frozen_diameter, frozen::mass_exponent[species] - 3));
        pair.mixed_density = density * std::pow(pair.frozen_diameter / diameter, 3) + 1000 * std::pow(pair.rain_diameter / diameter, 3);
        if (species == 1 && pair.mixed_density <= (density + 400) / 2) pair.destination = 1;
        else if (hail && pair.rain_diameter > .0025L && pair.mixed_density >= 650) pair.destination = 3;
        pair.destination_density = pair.destination == 1 ? density : pair.destination == 2 ? 400 : 900;
        pair.product_mass = pair.rain_mass + pair.frozen_mass;
        pair.product_number = std::min(pair.number, pair.product_mass /
            (std::acos(-1.L) / 6 * pair.destination_density * diameter * diameter * diameter));
        const long double d = frozen::mass_exponent[pair.destination];
        const long double c = frozen::mass_coefficient[pair.destination];
        pair.product_z = pair.product_number * frozen::gamma_ratio(0, 6) *
            std::pow(pair.product_mass / (c * pair.product_number * frozen::gamma_ratio(0, d)), 6 / d);
        const long double power = 6 / frozen::mass_exponent[species];
        result.mass[species + 1] -= pair.frozen_mass;
        result.number[species + 1] -= pair.number;
        result.z[species + 1] += population.reflectivity_per_kg *
            (-power * pair.frozen_mass / population.mass_per_kg + (power - 1) * pair.number / population.number_per_kg);
        result.mass[pair.destination + 1] += pair.product_mass;
        result.number[pair.destination + 1] += pair.product_number;
        result.z[pair.destination + 1] += pair.product_z;
    }
    if (cell.rain.mass_per_kg > 0)
        result.z[0] = cell.rain.reflectivity_per_kg * (2 * result.mass[0] / cell.rain.mass_per_kg - result.number[0] / cell.rain.number_per_kg);
    result.temperature = -334000.L * result.mass[0] / 1004.L;
    return result;
}

inline rain::Reference evaluate_coupled(const rain::Cell& cell, long double dt, bool hail = true, bool wet_hail = false)
{
    const Cell collision_cell{{cell.temperature_k, cell.pressure_pa, cell.dry_density_kg_m3, {cell.initial.frozen}}, cell.initial.rain};
    const auto collision = evaluate(collision_cell, dt, hail);
    std::array<long double, 4> accepted{};
    for (std::size_t index = 0; index < 4; ++index) accepted[index] = collision.pairs[index].rain_mass;
    rain::Reference result;
    static_cast<rain::prefix::Reference&>(result) = rain::prefix::evaluate(cell, dt, hail, false, false,
        accepted, cell.initial.rain.mass_per_kg, collision.pairs[2].destination == 2, wet_hail);
    for (std::size_t index = 0; index < 4; ++index)
    {
        const auto& source = cell.initial.frozen[index];
        result.mass_increment[index] += collision.mass[index + 1] - accepted[index];
        result.number_increment[index] += collision.number[index + 1];
        result.z_increment[index] += collision.z[index + 1];
        if (source.mass_per_kg > 0)
            result.z_increment[index] -= 6 / frozen::mass_exponent[index] * source.reflectivity_per_kg * accepted[index] / source.mass_per_kg;
        if (index < 3)
        {
            result.original_mass_sink[index] += collision.pairs[index].frozen_mass;
            result.original_number_sink[index] += collision.pairs[index].number;
        }
    }
    result.rain_factor = collision.rain_factor;
    result.rain_mass = collision.mass[0];
    result.rain_number = collision.number[0];
    result.rain_z = collision.z[0];
    return result;
}

using State = std::array<long double, 16>;

inline State state(const Cell& cell)
{
    State result{cell.frozen.temperature_k, cell.rain.mass_per_kg, cell.rain.number_per_kg, cell.rain.reflectivity_per_kg};
    for (std::size_t species = 0; species < 4; ++species)
    {
        const auto& p = cell.frozen.initial.frozen[species];
        result[4 + 3 * species] = p.mass_per_kg;
        result[5 + 3 * species] = p.number_per_kg;
        result[6 + 3 * species] = p.reflectivity_per_kg;
    }
    return result;
}

inline Cell from_state(const Cell& context, const State& state)
{
    auto result = context;
    result.frozen.temperature_k = static_cast<double>(state[0]);
    result.rain = {static_cast<double>(state[1]), static_cast<double>(state[2]), static_cast<double>(state[3])};
    for (std::size_t species = 0; species < 4; ++species)
        result.frozen.initial.frozen[species] = {static_cast<double>(state[4 + 3 * species]),
            static_cast<double>(state[5 + 3 * species]), static_cast<double>(state[6 + 3 * species])};
    return result;
}

inline State derivative(const Cell& cell)
{
    const auto result = evaluate(cell, 1, true, false, true);
    State out{};
    out[0] = result.temperature;
    for (std::size_t species = 0; species < 5; ++species)
    {
        out[1 + 3 * species] = result.mass[species];
        out[2 + 3 * species] = result.number[species];
        out[3 + 3 * species] = result.z[species];
    }
    return out;
}

inline State rk4(const Cell& cell, long double duration, int steps)
{
    auto values = state(cell);
    const auto h = duration / steps;
    for (int step = 0; step < steps; ++step)
    {
        const auto stage = [&](const State& slope, long double fraction)
        {
            auto next = values;
            for (std::size_t index = 0; index < next.size(); ++index) next[index] += fraction * h * slope[index];
            return derivative(from_state(cell, next));
        };
        const auto a = derivative(from_state(cell, values));
        const auto b = stage(a, .5L), c = stage(b, .5L), d = stage(c, 1);
        for (std::size_t index = 0; index < values.size(); ++index)
            values[index] += h / 6 * (a[index] + 2 * b[index] + 2 * c[index] + d[index]);
    }
    return values;
}

} // namespace milbrandt_rain_collision_reference
