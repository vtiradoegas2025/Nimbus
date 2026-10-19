#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <type_traits>

namespace tmv::milbrandt_detail
{

enum class ColdRainFreezingDestination
{
    Graupel,
    Hail
};

// This records the evaluator's equal-fraction admitted q/N subset. It does
// not imply the higher moments of a size-selective Bigg freezing kernel.
struct ColdRainFreezingAcceptedEvent
{
    ColdRainFreezingDestination destination = ColdRainFreezingDestination::Graupel;
    double entry_rain_mass_per_kg = 0.0;
    double entry_rain_number_per_kg = 0.0;
    double admitted_mass_increment_per_kg = 0.0;
    double admitted_number_increment_per_kg = 0.0;
    double admitted_fraction = 0.0;
    double dry_density_kg_m3 = 0.0;
    double duration_s = 0.0;
};

struct ColdRainFreezingAcceptedEventBuffer
{
    std::array<ColdRainFreezingAcceptedEvent, 1> events{};
    std::size_t size = 0;
    std::size_t capacity = 1;
};

static_assert(std::is_trivially_copyable_v<ColdRainFreezingAcceptedEvent>);

inline ColdRainFreezingAcceptedEvent make_cold_rain_freezing_accepted_event(
    ColdRainFreezingDestination destination,
    double entry_rain_mass_per_kg,
    double entry_rain_number_per_kg,
    double admitted_mass_increment_per_kg,
    double admitted_number_increment_per_kg,
    double dry_density_kg_m3,
    double duration_s)
{
    const std::array<double, 6> inputs{
        entry_rain_mass_per_kg, entry_rain_number_per_kg,
        admitted_mass_increment_per_kg, admitted_number_increment_per_kg,
        dry_density_kg_m3, duration_s};
    if (std::any_of(inputs.begin(), inputs.end(),
                    [](double value) { return !std::isfinite(value) || !(value > 0.0); }) ||
        (destination != ColdRainFreezingDestination::Graupel &&
         destination != ColdRainFreezingDestination::Hail) ||
        admitted_mass_increment_per_kg > entry_rain_mass_per_kg ||
        admitted_number_increment_per_kg > entry_rain_number_per_kg)
    {
        throw std::runtime_error("Milbrandt rain-freezing observer received invalid event inputs");
    }
    const double mass_fraction = admitted_mass_increment_per_kg / entry_rain_mass_per_kg;
    const double number_fraction = admitted_number_increment_per_kg / entry_rain_number_per_kg;
    const double scale = std::max({std::abs(mass_fraction), std::abs(number_fraction), 1.0e-30});
    if (std::abs(mass_fraction - number_fraction) >
        64.0 * std::numeric_limits<double>::epsilon() * scale)
    {
        throw std::runtime_error("Milbrandt rain-freezing observer q/N fractions differ");
    }
    return {destination, entry_rain_mass_per_kg, entry_rain_number_per_kg,
            admitted_mass_increment_per_kg, admitted_number_increment_per_kg,
            number_fraction, dry_density_kg_m3, duration_s};
}

} // namespace tmv::milbrandt_detail
