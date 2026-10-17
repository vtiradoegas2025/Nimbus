#pragma once

#include <cmath>

namespace tmv::rrtmg_config_limits
{

inline constexpr double ozone_vmr_max = 1.0e-2;
inline constexpr double co2_vmr_max   = 1.0e-2;
inline constexpr double ch4_vmr_max   = 1.0e-3;
inline constexpr double n2o_vmr_max   = 1.0e-3;
inline constexpr double co_vmr_max    = 1.0e-3;
inline constexpr double o2_vmr_max    = 1.0;
inline constexpr double n2_vmr_max    = 1.0;
inline constexpr double halocarbon_vmr_max = 1.0e-4;

inline constexpr double liquid_effective_radius_min_um = 2.5;
inline constexpr double liquid_effective_radius_max_um = 60.0;
inline constexpr double ice_effective_radius_min_um    = 13.0;
inline constexpr double ice_effective_radius_max_um    = 130.0;

[[nodiscard]] inline bool valid_mole_fraction(double value, double maximum) noexcept
{
    return std::isfinite(value) && value >= 0.0 && value <= maximum;
}

[[nodiscard]] inline bool valid_liquid_effective_radius(double value) noexcept
{
    return std::isfinite(value) && value >= liquid_effective_radius_min_um &&
           value <= liquid_effective_radius_max_um;
}

[[nodiscard]] inline bool valid_ice_effective_radius(double value) noexcept
{
    return std::isfinite(value) && value >= ice_effective_radius_min_um && value <= ice_effective_radius_max_um;
}

} // namespace tmv::rrtmg_config_limits
