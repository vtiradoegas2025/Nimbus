#pragma once

#include <algorithm>
#include <cmath>
#include <limits>

namespace tmv::thompson
{

inline double transport_number_candidate(float initial, float rate, double dt)
{
    const double candidate = std::fma(dt, static_cast<double>(rate), static_cast<double>(initial));
    // An exhausted FP32 tendency can leave a one-ULP population. It must not
    // activate finite fall speeds for a particle distribution that has vanished.
    const double resolution =
        rate < 0 ? dt * (static_cast<double>(std::nextafter(rate, std::numeric_limits<float>::infinity())) - rate) : 0;
    return candidate <= resolution + 2 * std::numeric_limits<double>::epsilon() * initial ? 0 : candidate;
}

struct InitialNumberMoments
{
    double cloud_kg_dry = 0.0;
    double rain_kg_dry  = 0.0;
    double ice_kg_dry   = 0.0;
};

double normalize_cloud_number_m3(double cloud_water_kg_m3, double supplied_number_m3);
double normalize_rain_number_m3(double rain_water_kg_m3, double supplied_number_m3);
double normalize_ice_number_m3(double ice_water_kg_m3, double supplied_number_m3);

InitialNumberMoments initialize_number_moments(double cloud_water_kg_kg_dry,
                                               double rain_water_kg_kg_dry,
                                               double ice_water_kg_kg_dry,
                                               double dry_air_density_kg_m3);

} // namespace tmv::thompson
