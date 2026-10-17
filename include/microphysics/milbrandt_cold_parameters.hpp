#pragma once

#include <array>

namespace tmv::milbrandt_detail
{
inline constexpr std::array<double, 25> default_cold_parameters{
    19.3, 206.89,        71.34,         149.1, 11.72,         0,      0, 0, 0, 0, 0.37, 0.6384, 0.6635, 0.5,
    0.41, 209.439510239, 471.238898038, 440,   523.598775598, 0.1597, 3, 3, 3, 3, 2.078};
}
