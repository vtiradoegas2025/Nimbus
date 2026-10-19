#pragma once

#include "microphysics/schemes/milbrandt/warm_rain.hpp"
#include "microphysics/schemes/milbrandt/triple_warm_phase.hpp"

#include <algorithm>
#include <cmath>

namespace milbrandt_dynamic_warm_reference
{

using namespace tmv::milbrandt_detail;
constexpr long double mass_coefficient = 523.598775598L;
constexpr long double pi = 3.141592653589793238462643383279502884L;

struct Population
{
    long double q = 0;
    long double n = 0;
    long double z = 0;
};

inline long double rising(long double alpha, int order)
{
    long double value = 1;
    for (int index = 1; index <= order; ++index)
    {
        value *= alpha + index;
    }
    return value;
}

inline RainTripleMomentState source_rain(float mass, double mean_diameter,
                                         long double nominal_alpha)
{
    const float number = static_cast<float>(mass /
        (mass_coefficient * std::pow(static_cast<long double>(mean_diameter), 3)));
    const long double shape = rising(nominal_alpha, 6) /
        std::pow(rising(nominal_alpha, 3), 2);
    return {mass, number, static_cast<float>(shape * mass * mass /
        (mass_coefficient * mass_coefficient * number))};
}

inline Population population(const RainTripleMomentState& state)
{
    return {state.mass_per_kg, state.number_per_kg, state.reflectivity_per_kg};
}

inline long double alpha(const Population& state)
{
    const long double ratio = state.z * state.n * mass_coefficient * mass_coefficient /
        (state.q * state.q);
    long double lower = -1.0L + 1.0e-9L;
    long double upper = 1.0e6L;
    for (int iteration = 0; iteration < 80; ++iteration)
    {
        const long double middle = (lower + upper) / 2;
        const long double closure = (middle + 4) * (middle + 5) * (middle + 6) /
            ((middle + 1) * (middle + 2) * (middle + 3));
        if (closure > ratio)
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

inline long double lambda(const Population& state, long double shape)
{
    return std::cbrt(mass_coefficient * state.n * rising(shape, 3) / state.q);
}

inline long double scaled_integral(long double shape, long double order)
{
    constexpr int intervals = 12000;
    const long double upper_x = std::max(120.0L,
        shape + 1 + 30 * std::sqrt(shape + 1) + 80);
    const long double width = std::pow(upper_x, .25L) / intervals;
    const long double shift = shape > 0 ? shape * std::log(shape) - shape : 0;
    long double sum = 0;
    for (int index = 1; index <= intervals; ++index)
    {
        const long double u = index * width;
        const long double x = u * u * u * u;
        const long double integrand = 4 * std::exp(
            (4 * (shape + order) + 3) * std::log(u) - x - shift);
        sum += (index == intervals ? 1 : index % 2 ? 4 : 2) * integrand;
    }
    return sum * width / 3;
}

inline long double integer_moment(const Population& state, int order)
{
    const long double shape = alpha(state);
    return rising(shape, order) / std::pow(lambda(state, shape), order);
}

inline long double fractional_moment(const Population& state, long double order)
{
    const long double shape = alpha(state);
    return scaled_integral(shape, order) / scaled_integral(shape, 0) /
        std::pow(lambda(state, shape), order);
}

inline long double mean_diameter(const Population& state)
{
    return std::cbrt(state.q / (mass_coefficient * state.n));
}

inline Population regularize(Population state, long double new_number)
{
    state.z *= state.n / new_number;
    state.n = new_number;
    return state;
}

inline Population replay_prefix(const RainTripleMomentState& initial,
                                const WarmRainAcceptedEventBuffer& events,
                                std::size_t count, long double dt,
                                long double density)
{
    Population state = population(initial);
    Population physical_source = state;
    long double mass_rate = 0;
    long double number_rate = 0;
    long double reflectivity_rate = 0;
    bool physical_started = false;
    bool physical_applied = false;
    const auto finish_physical = [&]()
    {
        if (physical_started && !physical_applied)
        {
            state = {physical_source.q + dt * mass_rate,
                     physical_source.n + dt * number_rate,
                     physical_source.z + dt * reflectivity_rate};
            physical_applied = true;
        }
    };
    for (std::size_t index = 0; index < count; ++index)
    {
        const auto& event = events.events[index];
        const long double dq = event.rain_mass_rate_per_kg_s;
        const long double dn = event.rain_number_rate_per_m3_s / density;
        if (event.kind == WarmRainAcceptedEventKind::PreRainNumberRegularization)
        {
            state = regularize(state, state.n + dt * dn);
            physical_source = state;
        }
        else if (event.kind == WarmRainAcceptedEventKind::Autoconversion ||
                 event.kind == WarmRainAcceptedEventKind::Accretion ||
                 event.kind == WarmRainAcceptedEventKind::RainSelfCollection ||
                 event.kind == WarmRainAcceptedEventKind::RainBreakup)
        {
            physical_started = true;
            mass_rate += dq;
            number_rate += dn;
            if (event.kind == WarmRainAcceptedEventKind::Autoconversion)
            {
                constexpr long double b4 = 28.0L / 5.0L;
                reflectivity_rate += b4 * dq * dq /
                    (mass_coefficient * mass_coefficient * dn);
            }
            else
            {
                reflectivity_rate += physical_source.z *
                    (2 * dq / physical_source.q - dn / physical_source.n);
            }
        }
        else if (event.kind == WarmRainAcceptedEventKind::PostRainNumberRegularization)
        {
            finish_physical();
            state = regularize(state, state.n + dt * dn);
        }
        else if (event.kind == WarmRainAcceptedEventKind::RainToCloud)
        {
            finish_physical();
            state = {};
        }
    }
    finish_physical();
    return state;
}

struct PhysicalRates
{
    long double autoconversion_mass = 0;
    long double autoconversion_number = 0;
    long double accretion_mass = 0;
    long double accretion_number = 0;
    long double cloud_self = 0;
    long double rain_self = 0;
    long double donor_scale = 1;
    long double seed_diameter = 0;
};

inline PhysicalRates physical_rates(const DoubleWarmRainCell& cell,
                                    const Population& rain,
                                    long double dt)
{
    PhysicalRates rates;
    const long double density = cell.density;
    const long double nc = density * cell.cloud_number;
    const long double nr = density * rain.n;
    const long double cloud = cell.cloud;
    long double inverse_cloud = 0;
    long double liquid_scale = 0;
    long double initial_diameter = 0;
    if (cloud > 1e-14L && nc > 1e-3L)
    {
        inverse_cloud = std::cbrt(density * cloud / (nc * (1000 * pi / 3)));
        const long double mean_cloud = inverse_cloud * std::cbrt(2.0L);
        const long double sigma = inverse_cloud * std::pow(2.0L, 1.0L / 6);
        liquid_scale = .027L * density * cloud *
            (6.25e18L * std::pow(sigma, 3) * mean_cloud - .4L);
        if (sigma > 15e-6L)
        {
            const long double tau = 3.7L /
                (density * cloud * (.5e6L * sigma - 7.5L));
            rates.autoconversion_mass = std::min(std::max(liquid_scale / tau, 0.0L), cloud / dt);
            initial_diameter = std::max(83e-6L, 12.6e-4L / (.5e6L * sigma - 3.5L));
            rates.seed_diameter = std::max(initial_diameter,
                rain.q > 0 ? mean_diameter(rain) : 0.0L);
            rates.autoconversion_number = rates.autoconversion_mass * density /
                (mass_coefficient * std::pow(rates.seed_diameter, 3));
            rates.cloud_self = std::min(2.59e15L * nc * nc * 6 *
                std::pow(inverse_cloud, 6), nc / dt);
        }
    }
    if (rain.q > 1e-14L && nr > 1e-3L)
    {
        const long double diameter = mean_diameter(rain);
        if (rain.q > 1.2L * std::max(liquid_scale, 0.0L) / density ||
            diameter > std::max(5e-6L, initial_diameter))
        {
            const long double rain_moment = integer_moment(rain, diameter >= 100e-6L ? 3 : 6);
            const long double coefficient = diameter >= 100e-6L ? 3.03e3L : 2.59e15L;
            const long double ic3 = std::pow(inverse_cloud, 3);
            if (cloud > 1e-14L && liquid_scale > 0)
            {
                rates.accretion_number = coefficient * nc * nr *
                    (diameter >= 100e-6L ? 2 * ic3 + rain_moment : 6 * ic3 * ic3 + rain_moment);
                rates.accretion_mass = mass_coefficient / density * coefficient * nc * nr * ic3 *
                    (diameter >= 100e-6L ? 6 * ic3 + 2 * rain_moment : 24 * ic3 * ic3 + 2 * rain_moment);
                rates.accretion_number = std::min(rates.accretion_number, nc / dt);
                rates.accretion_mass = std::min(rates.accretion_mass, cloud / dt);
            }
            const long double equivalent = diameter / std::cbrt(6.0L);
            const long double efficiency = equivalent > 300e-6L
                ? 2 - std::exp(2300 * (equivalent - 300e-6L)) : 1;
            rates.rain_self = std::min(efficiency * coefficient * nr * nr * rain_moment,
                                      .5L * nr / dt);
        }
    }
    const long double sink = dt * (rates.autoconversion_mass + rates.accretion_mass);
    if (sink > cloud && sink > 0)
    {
        rates.donor_scale = cloud / sink;
        rates.autoconversion_mass *= rates.donor_scale;
        rates.autoconversion_number *= rates.donor_scale;
        rates.accretion_mass *= rates.donor_scale;
        rates.accretion_number *= rates.donor_scale;
    }
    return rates;
}

inline long double saturation(long double temperature, long double pressure)
{
    const long double celsius = temperature - 273.15L;
    const long double vapor_pressure = 611.21L * std::exp(
        (18.678L - celsius / 234.5L) * celsius / (257.14L + celsius));
    return .62198L * vapor_pressure / (pressure - vapor_pressure);
}

inline long double evaporated(const DoubleWarmRainCell& cell,
                              const Population& rain, long double dt)
{
    if (!(rain.q > 1e-14L && rain.n * cell.density > 1e-3L))
    {
        return 0;
    }
    const long double saturation_water = saturation(cell.temperature, cell.pressure);
    if (cell.vapor >= saturation_water)
    {
        return 0;
    }
    const long double celsius = static_cast<long double>(cell.temperature) - 273.15L;
    const long double density = cell.density;
    const long double diffusivity = std::max(1.62e-5L, 2.2157e-5L + .0155e-5L * celsius) * 1e5L / cell.pressure;
    const long double viscosity = std::max(1.51e-5L, 1.7153e-5L + .0050e-5L * celsius);
    const long double conductivity = std::max(2.07e-2L, 2.3971e-2L + .0078e-2L * celsius);
    const long double nu = viscosity / density;
    const long double ventilation = density * rain.n *
        (.78L * integer_moment(rain, 1) + .30L * std::cbrt(nu / diffusivity) *
         std::sqrt(std::sqrt(1.225L / density) * 149.1L / nu) *
         fractional_moment(rain, 1.75L));
    const long double resistance = 2.5e6L * 2.5e6L /
        (conductivity * 461.5L * cell.temperature * cell.temperature) +
        1 / (density * saturation_water * diffusivity);
    return std::min(rain.q, std::max(0.0L, -dt / density * 2 * pi *
        (cell.vapor / saturation_water - 1) * ventilation / resistance));
}

} // namespace milbrandt_dynamic_warm_reference
