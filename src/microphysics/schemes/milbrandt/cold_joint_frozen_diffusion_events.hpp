#pragma once

#include "cold_frozen_diffusion_events.hpp"

#include <array>
#include <cmath>
#include <limits>
#include <string>
#include <type_traits>

namespace tmv::milbrandt_detail
{

struct ColdJointFrozenDiffusionAcceptedEventBuffer
{
    std::array<ColdFrozenDiffusionAcceptedEvent, 2> events{};
    std::size_t size = 0;
    std::size_t capacity = 2;
};

struct ColdTerminalHailToGraupelAcceptedEvent
{
    bool transferred = false;
    double entry_mass_per_kg = 0.0;
    double entry_number_per_kg = 0.0;
    double entry_reflectivity_per_kg = 0.0;
    double source_alpha = 0.0;
    double source_mean_mass_diameter_m = 0.0;
    double graupel_reflectivity_increment_per_kg = 0.0;
    double dry_density_kg_m3 = 0.0;
    double duration_s = 0.0;
};

static_assert(std::is_trivially_copyable_v<ColdTerminalHailToGraupelAcceptedEvent>);

struct ColdJointFrozenDiffusionMaterialChannel
{
    const char* name;
    bool owned;
};

inline constexpr std::array<ColdJointFrozenDiffusionMaterialChannel, 64>
    published_joint_frozen_diffusion_material_channels{{
        {"vapor_to_ice", false}, {"cloud_to_ice", false},
        {"rain_to_frozen", false}, {"ice_to_snow", false},
        {"nc_to_ice", false}, {"nr_to_frozen", false},
        {"new_ice_number", false},
        {"ice_number_sublimated", false},
        {"snow_number_sublimated", false},
        {"graupel_number_sublimated", true},
        {"hail_number_sublimated", true},
        {"ice_number_to_snow", false},
        {"cloud_to_snow", false}, {"cloud_to_graupel", false},
        {"cloud_to_hail", false},
        {"cloud_number_to_snow", false},
        {"cloud_number_to_graupel", false},
        {"cloud_number_to_hail", false},
        {"ice_collected_by_snow", false},
        {"ice_collected_by_graupel", false},
        {"ice_collected_by_hail", false},
        {"ice_number_collected_by_snow", false},
        {"ice_number_collected_by_graupel", false},
        {"ice_number_collected_by_hail", false},
        {"snow_self_collection", false},
        {"splinter_mass_from_snow", false},
        {"splinter_mass_from_graupel", false},
        {"splinter_number_from_snow", false},
        {"splinter_number_from_graupel", false},
        {"rain_collected_by_ice", false},
        {"ice_collected_by_rain", false},
        {"rain_number_collected_by_ice", false},
        {"ice_number_collected_by_rain", false},
        {"rain_collected_by_snow", false},
        {"snow_collected_by_rain", false},
        {"rain_number_collected_by_snow", false},
        {"snow_number_collected_by_rain", false},
        {"rain_collected_by_graupel", false},
        {"graupel_collected_by_rain", false},
        {"rain_number_collected_by_graupel", false},
        {"graupel_number_collected_by_rain", false},
        {"collision_mass_to_snow", false},
        {"collision_mass_to_graupel", false},
        {"collision_mass_to_hail", false},
        {"collision_number_to_snow", false},
        {"collision_number_to_graupel", false},
        {"collision_number_to_hail", false},
        {"snow_to_graupel", false},
        {"snow_number_to_graupel", false},
        {"graupel_to_hail", false},
        {"graupel_number_to_hail", false},
        {"hail_number_from_graupel", false},
        {"hail_to_graupel", false},
        {"hail_number_to_graupel", false},
        {"rain_collected_by_hail", false},
        {"rain_number_collected_by_hail", false},
        {"snow_collected_by_hail", false},
        {"snow_number_collected_by_hail", false},
        {"shed_rain_from_hail", false},
        {"shed_rain_number", false},
        {"ice_deposition", false}, {"snow_deposition", false},
        {"graupel_deposition", true}, {"hail_deposition", true}}};

inline bool validate_published_joint_frozen_diffusion_channels(
    const std::array<double, 64>& values,
    std::string& error)
{
    for (std::size_t index = 0; index < values.size(); ++index)
    {
        const auto& channel = published_joint_frozen_diffusion_material_channels[index];
        if (!std::isfinite(values[index]) ||
            (!channel.owned && values[index] != 0.0))
        {
            error = std::string("published joint frozen diffusion observed unowned or invalid cold channel: ")
                + channel.name;
            return false;
        }
    }
    error.clear();
    return true;
}

inline bool published_joint_frozen_diffusion_rate_representable(
    double value) noexcept
{
    const double magnitude = std::abs(value);
    return std::isfinite(value) &&
        magnitude <= std::numeric_limits<float>::max() &&
        (magnitude == 0.0 || magnitude >= std::numeric_limits<float>::denorm_min());
}

} // namespace tmv::milbrandt_detail
