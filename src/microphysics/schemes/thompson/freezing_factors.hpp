#pragma once

#include <array>
#include <vector>

namespace tmv::thompson
{
struct FreezingFactors
{
    std::array<double, 37>  water_content;
    std::array<double, 37>  rain_intercept;
    std::array<double, 55>  ice_nuclei;
    std::array<double, 100> cloud_number;
    std::array<double, 100> rain_mass;
    std::array<double, 100> cloud_mass;
    std::vector<double>     rain_bin_number;
    std::vector<double>     cloud_bin_number;
    std::vector<double>     rain_probability;
    std::vector<double>     cloud_probability;
};

FreezingFactors generate_freezing_factors();
} // namespace tmv::thompson
