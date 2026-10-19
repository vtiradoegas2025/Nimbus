#pragma once

#include "soundings/soundings_base.hpp"

#include <vector>

namespace tmv::soundings::sharpy
{

void apply_log_linear_pressure_profile(const SoundingData& source,
                                       const std::vector<double>& target_heights_m,
                                       const SoundingConfig& config,
                                       SoundingData& interpolated);

SoundingData interpolate_sounding_monotone_spline(const SoundingData& source,
                                                  const std::vector<double>& target_heights_m,
                                                  const SoundingConfig& config);

}  // namespace tmv::soundings::sharpy
