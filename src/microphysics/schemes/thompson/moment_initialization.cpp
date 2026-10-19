#include "moment_initialization.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <iomanip>
#include <sstream>

namespace tmv::thompson
{
namespace
{

constexpr double pi = 3.1415926536;
constexpr double rain_mass_coefficient = pi * 1000.0 / 6.0;
constexpr double ice_mass_coefficient = pi * 890.0 / 6.0;
constexpr double minimum_mixing_ratio = 1.0e-12;
constexpr double minimum_number_m3 = 1.0e-6;
constexpr double maximum_cloud_number_m3 = 1999.0e6;
constexpr double maximum_ice_number_m3 = 999.0e3;

void require_valid(double mass, double number, const char* species)
{
    if (!std::isfinite(mass) || mass <= 0.0
        || !std::isfinite(number) || number < 0.0)
    {
        std::ostringstream message;
        message << std::setprecision(17) << "invalid Thompson distribution initialization: mass_kg_m3=" << mass
                << ", number_m3=" << number << ", species=" << species;
        throw std::invalid_argument(message.str());
    }
}

double number_for_spherical_mass(double mass_kg_m3,
                                 double mass_coefficient,
                                 double slope_m_inv)
{
    return mass_kg_m3 * slope_m_inv * slope_m_inv * slope_m_inv
           / (6.0 * mass_coefficient);
}

} // namespace

double normalize_cloud_number_m3(double cloud_water_kg_m3, double supplied_number_m3)
{
    require_valid(cloud_water_kg_m3, supplied_number_m3, "cloud");
    double number = std::clamp(supplied_number_m3, 2.0, maximum_cloud_number_m3);
    // Each size-bound correction moves monotonically through the finite diagnosed shapes.
    for (int iteration = 0; iteration < 16; ++iteration)
    {
        const double coordinate = 1000.0e6 / number;
        const int    shape      = coordinate >= 13 ? 15 : static_cast<int>(std::lround(coordinate)) + 2;
        const double ratio      = double((shape + 1) * (shape + 2) * (shape + 3));
        const double slope      = std::cbrt(number * rain_mass_coefficient * ratio / cloud_water_kg_m3);
        const double diameter   = double(shape + 4) / slope;
        if (diameter >= 1e-6 && diameter <= 100e-6)
            return number;
        const double bounded_slope = double(shape + 4) / std::clamp(diameter, 1e-6, 100e-6);
        const double next          = std::min(maximum_cloud_number_m3,
                                              cloud_water_kg_m3 * bounded_slope * bounded_slope * bounded_slope /
                                                  (ratio * rain_mass_coefficient));
        if (next == number)
            return number;
        number = next;
    }
    return number;
}

double normalize_rain_number_m3(double rain_water_kg_m3, double supplied_number_m3)
{
    require_valid(rain_water_kg_m3, supplied_number_m3, "rain");
    double number = std::max(minimum_number_m3, supplied_number_m3);
    if (number <= minimum_number_m3)
        number = number_for_spherical_mass(
            rain_water_kg_m3, rain_mass_coefficient, 3.672 / 1.0e-3);
    double slope = std::cbrt(
        rain_mass_coefficient * 6.0 * number / rain_water_kg_m3);
    double median_volume_diameter = 3.672 / slope;
    if (median_volume_diameter > 2.5e-3)
        median_volume_diameter = 2.5e-3;
    else if (median_volume_diameter < 37.5e-6)
        median_volume_diameter = 37.5e-6;
    slope = 3.672 / median_volume_diameter;
    return number_for_spherical_mass(
        rain_water_kg_m3, rain_mass_coefficient, slope);
}

double normalize_ice_number_m3(double ice_water_kg_m3,
                               double supplied_number_m3)
{
    require_valid(ice_water_kg_m3, supplied_number_m3, "ice");
    double number = std::max(minimum_number_m3, supplied_number_m3);
    if (number <= minimum_number_m3)
        number = std::min(
            maximum_ice_number_m3,
            number_for_spherical_mass(
                ice_water_kg_m3, ice_mass_coefficient, 4.0 / 5.0e-6));
    double slope = std::cbrt(
        ice_mass_coefficient * 6.0 * number / ice_water_kg_m3);
    const double mean_mass_diameter = 4.0 / slope;
    if (mean_mass_diameter < 5.0e-6)
        number = std::min(
            maximum_ice_number_m3,
            number_for_spherical_mass(
                ice_water_kg_m3, ice_mass_coefficient, 4.0 / 5.0e-6));
    else if (mean_mass_diameter > 300.0e-6)
        number = number_for_spherical_mass(
            ice_water_kg_m3, ice_mass_coefficient, 4.0 / 300.0e-6);
    return number;
}

InitialNumberMoments initialize_number_moments(double cloud_water_kg_kg_dry,
                                               double rain_water_kg_kg_dry,
                                               double ice_water_kg_kg_dry,
                                               double dry_air_density_kg_m3)
{
    if (!std::isfinite(cloud_water_kg_kg_dry) || cloud_water_kg_kg_dry < 0.0
        || !std::isfinite(rain_water_kg_kg_dry) || rain_water_kg_kg_dry < 0.0
        || !std::isfinite(ice_water_kg_kg_dry) || ice_water_kg_kg_dry < 0.0
        || !std::isfinite(dry_air_density_kg_m3)
        || dry_air_density_kg_m3 <= 0.0)
    {
        throw std::invalid_argument("invalid Thompson moment initialization state");
    }
    InitialNumberMoments result;
    if (cloud_water_kg_kg_dry > minimum_mixing_ratio)
        result.cloud_kg_dry = normalize_cloud_number_m3(
            cloud_water_kg_kg_dry * dry_air_density_kg_m3, 0.0)
            / dry_air_density_kg_m3;
    if (rain_water_kg_kg_dry > minimum_mixing_ratio)
        result.rain_kg_dry = normalize_rain_number_m3(
            rain_water_kg_kg_dry * dry_air_density_kg_m3, 0.0)
            / dry_air_density_kg_m3;
    if (ice_water_kg_kg_dry > minimum_mixing_ratio)
        result.ice_kg_dry = normalize_ice_number_m3(
            ice_water_kg_kg_dry * dry_air_density_kg_m3, 0.0)
            / dry_air_density_kg_m3;
    return result;
}

} // namespace tmv::thompson
