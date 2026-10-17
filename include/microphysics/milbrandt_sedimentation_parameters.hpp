#pragma once
#include <array>

namespace tmv::milbrandt_detail
{
inline constexpr std::array<std::array<double, 7>, 5> default_sedimentation_parameters{
    {{0, 523.598775598, 3, 149.1, .5, 16, .005},
     {0, 440, 3, 71.34, .6635, 2, .005},
     {0, .1597, 2.078, 11.72, .41, 4, .005},
     {0, 209.439510239, 3, 19.3, .37, 6, .002},
     {0, 471.238898038, 3, 206.89, .6384, 25, .08}}};
}
