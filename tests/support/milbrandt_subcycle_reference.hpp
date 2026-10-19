#pragma once

#include "microphysics/schemes/milbrandt/triple_warm_phase.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>

namespace milbrandt_subcycle_reference
{

constexpr long double mass_coefficient = 523.598775598L;

struct BreakupPopulation
{
    long double q = 0;
    long double n = 0;
    long double z = 0;
};

inline long double moment_ratio(long double alpha)
{
    return (alpha + 4) * (alpha + 5) * (alpha + 6) /
        ((alpha + 1) * (alpha + 2) * (alpha + 3));
}

inline tmv::milbrandt_detail::RainTripleMomentState rain(float mass, float number,
                                                        long double alpha)
{
    return {mass, number, static_cast<float>(moment_ratio(alpha) * mass * mass /
        (mass_coefficient * mass_coefficient * number))};
}

inline BreakupPopulation population(const tmv::milbrandt_detail::RainTripleMomentState& source)
{
    return {source.mass_per_kg, source.number_per_kg, source.reflectivity_per_kg};
}

inline long double mean_diameter(const BreakupPopulation& state)
{
    return std::cbrt(state.q / (mass_coefficient * state.n));
}

inline BreakupPopulation regularize(BreakupPopulation state)
{
    const long double diameter = mean_diameter(state);
    if (diameter > .003L)
    {
        const long double excess = diameter - .003L;
        const long double factor = std::max(1 + 2.0e4L * excess * excess,
                                            std::pow(diameter / .005L, 3));
        state.n *= factor;
        state.z /= factor;
    }
    return state;
}

inline long double breakup_number_rate(long double mass, long double number,
                                       long double density)
{
    const long double diameter = std::cbrt(mass / (mass_coefficient * number));
    const long double equivalent = diameter / std::cbrt(6.0L);
    const long double efficiency = equivalent > 300e-6L
        ? 2 - std::exp(2300 * (equivalent - 300e-6L)) : 1;
    return -efficiency * 3.03e3L * density * mass / mass_coefficient * number;
}

inline BreakupPopulation euler_breakup(BreakupPopulation state,
                                       long double density, long double duration)
{
    const long double number_rate = breakup_number_rate(state.q, state.n, density);
    return {state.q, state.n + duration * number_rate,
            state.z - duration * state.z * number_rate / state.n};
}

inline BreakupPopulation second_order_breakup(BreakupPopulation state,
                                              long double density, long double duration)
{
    const auto coarse = euler_breakup(state, density, duration);
    const auto middle = euler_breakup(state, density, duration / 2);
    const auto fine = euler_breakup(middle, density, duration / 2);
    return {state.q + 2 * (fine.q - state.q) - (coarse.q - state.q),
            state.n + 2 * (fine.n - state.n) - (coarse.n - state.n),
            state.z + 2 * (fine.z - state.z) - (coarse.z - state.z)};
}

inline BreakupPopulation integrate_breakup(
    const tmv::milbrandt_detail::RainTripleMomentState& source,
    long double density, long double duration, std::size_t intervals = 32768)
{
    BreakupPopulation state = regularize(population(source));
    const long double invariant = state.n * state.z;
    const long double step = duration / intervals;
    for (std::size_t index = 0; index < intervals; ++index)
    {
        const long double k1 = breakup_number_rate(state.q, state.n, density);
        const long double k2 = breakup_number_rate(state.q, state.n + step * k1 / 2, density);
        const long double k3 = breakup_number_rate(state.q, state.n + step * k2 / 2, density);
        const long double k4 = breakup_number_rate(state.q, state.n + step * k3, density);
        state.n += step * (k1 + 2 * k2 + 2 * k3 + k4) / 6;
    }
    state.z = invariant / state.n;
    return regularize(state);
}

}
