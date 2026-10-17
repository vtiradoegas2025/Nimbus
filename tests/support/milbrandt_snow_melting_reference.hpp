#pragma once

#include "microphysics/schemes/milbrandt/triple_melting.hpp"

#include <algorithm>
#include <array>
#include <cmath>

namespace snow_melting_reference
{

using Moment = tmv::milbrandt_detail::MeltingTripleMomentState;
constexpr long double coefficient = .1597L;
constexpr long double exponent = 2.078L;
constexpr long double rain_coefficient = 523.598775598L;

Moment population(long double alpha, double q = 5e-4f, double n = 10.0f)
{
    if (q == 0 && n == 0)
    {
        return {};
    }
    const long double a = alpha + 1;
    const long double log_lambda = (std::log(coefficient) + std::log(n)
        + std::lgamma(a + exponent) - std::lgamma(a) - std::log(q)) / exponent;
    return {q, n, static_cast<float>(n * std::exp(std::lgamma(a + 6)
        - std::lgamma(a) - 6 * log_lambda))};
}

long double alpha(const Moment& source)
{
    if (source.mass_per_kg == 0)
    {
        return 0;
    }
    const long double target = std::log(static_cast<long double>(source.reflectivity_per_kg)
        / source.number_per_kg) - 6 / exponent
        * std::log(source.mass_per_kg / (coefficient * source.number_per_kg));
    long double lower = -1 + 1e-9L;
    long double upper = 1e6L;
    for (int iteration = 0; iteration < 120; ++iteration)
    {
        const long double middle = (lower + upper) / 2;
        const long double a = middle + 1;
        const long double ratio = std::lgamma(a + 6) - std::lgamma(a)
            + 6 / exponent * (std::lgamma(a) - std::lgamma(a + exponent));
        if (ratio > target)
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

long double moment(long double a, long double order)
{
    constexpr int intervals = 32768;
    const long double end = std::max(160.0L, a + order + 16 * std::sqrt(a + order));
    const long double width = std::pow(end, .25L) / intervals;
    const long double log_normalization = std::lgamma(a);
    const auto integrand = [&](long double y)
    {
        if (y == 0)
        {
            return 0.0L;
        }
        return 4 * std::exp((4 * (a + order) - 1) * std::log(y)
            - std::pow(y, 4) - log_normalization);
    };
    long double sum = integrand(intervals * width);
    for (int index = 1; index < intervals; ++index)
    {
        sum += (index % 2 ? 4 : 2) * integrand(index * width);
    }
    return sum * width / 3;
}

struct Result
{
    long double shape = 0;
    long double lambda = 0;
    long double mass_squared = 0;
    long double melted_mass = 0;
    long double melted_number = 0;
    long double snow_z_loss = 0;
    long double rain_z_gain = 0;
};

Result evaluate(const Moment& source, long double temperature, long double pressure,
                long double density, long double vapor, long double duration)
{
    Result result;
    if (source.mass_per_kg == 0)
    {
        return result;
    }
    result.shape = alpha(source);
    const long double a = result.shape + 1;
    const long double physical_mass_moment = moment(a, exponent);
    result.lambda = std::pow(coefficient * source.number_per_kg * physical_mass_moment
        / source.mass_per_kg, 1 / exponent);
    result.mass_squared = static_cast<long double>(source.mass_per_kg) * source.mass_per_kg
        / source.number_per_kg * moment(a, 2 * exponent)
        / (physical_mass_moment * physical_mass_moment);
    const long double tc = temperature - microphysics_constants::T0;
    if (tc <= 0)
    {
        return result;
    }
    const long double diffusivity = (2.2157e-5L + .0155e-5L * tc) * 1e5L / pressure;
    const long double conductivity = 2.3971e-2L + .0078e-2L * tc;
    const long double viscosity = 1.72e-5L * 393 / (temperature + 120)
        * std::pow(temperature / microphysics_constants::T0, 1.5L) / density;
    const long double equivalent_coefficient = std::acos(-1.0L) / 6 * 100;
    const long double inverse_lambda = std::cbrt(source.mass_per_kg
        / (equivalent_coefficient * source.number_per_kg * moment(a, 3)));
    const long double ventilation = .78L * moment(a, 1) * inverse_lambda
        + .30L * std::cbrt(viscosity / diffusivity)
        * std::sqrt(std::sqrt(1.225L / density) * 11.72L / viscosity)
        * moment(a, 1.705L) * std::pow(inverse_lambda, 1.705L);
    const long double saturation = .62198L * 611.21L / (pressure - 611.21L);
    const long double heat = conductivity * tc - density * microphysics_constants::L_v
        * diffusivity * (saturation - vapor);
    result.melted_mass = std::clamp(duration * 2 * std::acos(-1.0L)
        / microphysics_constants::L_f * source.number_per_kg * ventilation * heat,
        0.0L, static_cast<long double>(source.mass_per_kg));
    const long double fraction = result.melted_mass / source.mass_per_kg;
    result.melted_number = source.number_per_kg * fraction;
    result.snow_z_loss = source.reflectivity_per_kg * fraction;
    result.rain_z_gain = fraction * result.mass_squared
        / (rain_coefficient * rain_coefficient);
    return result;
}

} // namespace snow_melting_reference
