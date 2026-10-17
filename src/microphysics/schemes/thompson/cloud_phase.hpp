#pragma once

#include <array>

namespace tmv::thompson
{
inline constexpr double cloud_melting_temperature        = 273.15;
inline constexpr double homogeneous_freezing_temperature = 235.16;
struct CloudPhaseState
{
    float                temperature;
    float                vapor;
    float                density;
    float                cloud;
    float                ice;
    float                cloud_number;
    float                ice_number;
    std::array<float, 5> tendencies;
};

std::array<float, 5> cloud_phase_tendencies(const CloudPhaseState& state, double dt);
} // namespace tmv::thompson
