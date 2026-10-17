#pragma once

#include "microphysics/base/thermodynamics.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace tmv::milbrandt_detail
{

enum class MeltingFrozenSpecies
{
    Graupel,
    Hail
};

struct MeltingAcceptedEvent
{
    MeltingFrozenSpecies species = MeltingFrozenSpecies::Graupel;
    double entry_mass_per_kg = 0.0;
    double entry_number_per_kg = 0.0;
    double source_alpha = 0.0;
    double admitted_mass_transfer_per_kg = 0.0;
    double admitted_number_transfer_per_kg = 0.0;
    double admitted_fraction = 0.0;
    double temperature_k = 0.0;
    double pressure_pa = 0.0;
    double dry_density_kg_m3 = 0.0;
    double vapor_per_kg = 0.0;
    double duration_s = 0.0;
};

struct MeltingAcceptedEventBuffer
{
    std::array<MeltingAcceptedEvent, 2> events{};
    std::size_t size = 0;
    std::size_t capacity = 2;
};

static_assert(std::is_trivially_copyable_v<MeltingAcceptedEvent>);

struct MeltingNoCollectionProfile
{
    std::array<double, 8> unowned_collection_increments{};
};

inline constexpr std::array<const char*, 8> melting_unowned_collection_channel_names{
    "cloud_to_graupel", "rain_collected_by_graupel",
    "cloud_to_hail", "rain_collected_by_hail",
    "cloud_number_to_graupel", "rain_number_collected_by_graupel",
    "cloud_number_to_hail", "rain_number_collected_by_hail"};

inline constexpr std::array<const char*, 4> melting_unowned_material_channel_names{
    "melted_ice", "melted_snow", "melted_ice_number", "melted_snow_number"};

inline bool validate_melting_no_collection_profile(
    const MeltingNoCollectionProfile& profile,
    std::string& error)
{
    for (std::size_t index = 0; index < profile.unowned_collection_increments.size(); ++index)
    {
        if (profile.unowned_collection_increments[index] != 0.0)
        {
            error = std::string("published triple melting observed unowned collection channel: ")
                + melting_unowned_collection_channel_names[index];
            return false;
        }
    }
    error.clear();
    return true;
}

inline bool melting_rate_representable(double value) noexcept
{
    const double magnitude = std::abs(value);
    return std::isfinite(value) && magnitude <= std::numeric_limits<float>::max() &&
        (magnitude == 0.0 || magnitude >= std::numeric_limits<float>::denorm_min());
}

inline bool validate_melting_unowned_material_channels(
    const std::array<double, 4>& channels,
    std::string& error)
{
    for (std::size_t index = 0; index < channels.size(); ++index)
    {
        if (channels[index] != 0.0)
        {
            error = std::string("published triple melting observed unowned material channel: ")
                + melting_unowned_material_channel_names[index];
            return false;
        }
    }
    error.clear();
    return true;
}

inline bool validate_triple_melting_context(
    const std::array<double, 12>& cell,
    double duration,
    const MeltingNoCollectionProfile& profile,
    std::string& error)
{
    if (!std::isfinite(duration) || duration <= 0.0 ||
        !std::isfinite(cell[0]) || cell[0] < 150.0 ||
        !std::isfinite(cell[1]) || cell[1] < 100.0 || cell[1] > 200000.0 ||
        !std::isfinite(cell[2]) || cell[2] <= 0.0)
    {
        error = "published triple melting context is invalid";
        return false;
    }
    for (std::size_t index = 3; index < cell.size(); ++index)
    {
        if (!std::isfinite(cell[index]) || cell[index] < 0.0)
        {
            error = "published triple melting water or number input is invalid";
            return false;
        }
    }
    if (!validate_melting_no_collection_profile(profile, error) ||
        !validate_melting_unowned_material_channels(
            {cell[4], cell[5], cell[8], cell[9]}, error))
    {
        return false;
    }
    error.clear();
    return true;
}

inline MeltingAcceptedEvent make_melting_accepted_event(
    MeltingFrozenSpecies species,
    double mass,
    double number,
    double alpha,
    double mass_transfer,
    double number_transfer,
    double temperature,
    double pressure,
    double density,
    double vapor,
    double duration)
{
    if ((species != MeltingFrozenSpecies::Graupel && species != MeltingFrozenSpecies::Hail) ||
        !std::isfinite(mass) || mass < 0.0 ||
        !std::isfinite(number) || number < 0.0 || (mass == 0.0) != (number == 0.0) ||
        !std::isfinite(alpha) || alpha <= -1.0 ||
        !std::isfinite(mass_transfer) || mass_transfer < 0.0 || mass_transfer > mass ||
        !std::isfinite(number_transfer) || number_transfer < 0.0 || number_transfer > number ||
        !std::isfinite(temperature) || temperature < 150.0 ||
        !std::isfinite(pressure) || pressure < 100.0 || pressure > 200000.0 ||
        !std::isfinite(density) || density <= 0.0 || !std::isfinite(vapor) || vapor < 0.0 ||
        !std::isfinite(duration) || duration <= 0.0 ||
        ((mass_transfer == 0.0) != (number_transfer == 0.0)) ||
        (mass_transfer > 0.0 &&
         (temperature <= microphysics_constants::T0 ||
          (species == MeltingFrozenSpecies::Hail && temperature - microphysics_constants::T0 <= 5.0))))
    {
        throw std::runtime_error("published triple melting event inputs are invalid");
    }
    const double fraction = mass > 0.0 ? mass_transfer / mass : 0.0;
    if (mass_transfer > 0.0)
    {
        const double number_fraction = number_transfer / number;
        if (std::abs(fraction - number_fraction) >
            64.0 * std::numeric_limits<double>::epsilon() * std::max(fraction, 1.0e-30))
        {
            throw std::runtime_error("published triple melting q/N fractions differ");
        }
    }
    return {species, mass, number, alpha, mass_transfer, number_transfer, fraction,
            temperature, pressure, density, vapor, duration};
}

} // namespace tmv::milbrandt_detail
