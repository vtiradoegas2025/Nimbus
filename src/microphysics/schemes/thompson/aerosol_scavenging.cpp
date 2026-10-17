#include "microphysics/schemes/thompson/aerosol_scavenging.hpp"

#include "microphysics/schemes/thompson/particle_distributions.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace tmv::thompson
{
namespace
{

constexpr double pi = 3.1415926536;
constexpr double rho_reference = 101325.0 / (287.05 * 298.0);
constexpr double water_mass_coefficient = pi * 1000.0 / 6.0;
constexpr double graupel_mass_coefficient = pi * 400.0 / 6.0;

void require_nonnegative(double value, const char* name)
{
    if (!std::isfinite(value) || value < 0.0)
        throw std::invalid_argument(std::string("invalid Thompson ") + name);
}

double viscosity(double temperature_k)
{
    const double temperature_c = temperature_k - 273.15;
    return temperature_c >= 0.0
        ? (1.718 + 0.0049 * temperature_c) * 1.0e-5
        : (1.718 + 0.0049 * temperature_c
           - 1.2e-5 * temperature_c * temperature_c) * 1.0e-5;
}

double limited_rate(double rate_m3_s, double available_kg_dry,
                    double density, double dt)
{
    if (!std::isfinite(rate_m3_s) || rate_m3_s < 0.0)
        throw std::overflow_error(
            "nonfinite Thompson aerosol scavenging rate");
    return std::min(available_kg_dry / dt, rate_m3_s / density);
}

} // namespace

double aerosol_collection_efficiency(double collector_diameter_m,
                                     double aerosol_diameter_m,
                                     double dynamic_viscosity_pa_s,
                                     double air_density_kg_m3,
                                     double temperature_k,
                                     AerosolCollector collector)
{
    if (!std::isfinite(collector_diameter_m) || collector_diameter_m <= 0.0
        || !std::isfinite(aerosol_diameter_m) || aerosol_diameter_m <= 0.0
        || !std::isfinite(dynamic_viscosity_pa_s)
        || dynamic_viscosity_pa_s <= 0.0
        || !std::isfinite(air_density_kg_m3) || air_density_kg_m3 <= 0.0
        || !std::isfinite(temperature_k) || temperature_k <= 0.0)
        throw std::invalid_argument("invalid Thompson aerosol collection state");
    double terminal_velocity = 0.0;
    switch (collector)
    {
    case AerosolCollector::Rain:
    {
        const double d = collector_diameter_m;
        terminal_velocity = -0.1021 + 4.932e3 * d - 0.9551e6 * d * d
            + 0.07934e9 * d * d * d
            - 0.002362e12 * d * d * d * d;
        break;
    }
    case AerosolCollector::Snow:
        terminal_velocity = 40.0 * std::pow(collector_diameter_m, 0.55);
        break;
    case AerosolCollector::Graupel:
        terminal_velocity = 442.0 * std::pow(collector_diameter_m, 0.89);
        break;
    default:
        throw std::invalid_argument("invalid Thompson aerosol collector");
    }
    if (!std::isfinite(terminal_velocity) || terminal_velocity <= 0.0)
        throw std::invalid_argument(
            "invalid Thompson aerosol collector velocity");

    constexpr double boltzmann = 1.3806503e-23;
    constexpr double mean_path = 0.0256e-6;
    const double slip = 1.0 + 2.0 * mean_path / aerosol_diameter_m
        * (1.257 + 0.4 * std::exp(-0.55 * aerosol_diameter_m / mean_path));
    const double diffusion = boltzmann * temperature_k * slip
        / (3.0 * pi * dynamic_viscosity_pa_s * aerosol_diameter_m);
    const double reynolds = 0.5 * air_density_kg_m3
        * collector_diameter_m * terminal_velocity / dynamic_viscosity_pa_s;
    const double schmidt = dynamic_viscosity_pa_s
        / (air_density_kg_m3 * diffusion);
    const double stokes = aerosol_diameter_m * aerosol_diameter_m
        * terminal_velocity * 1000.0
        / (9.0 * dynamic_viscosity_pa_s * collector_diameter_m);
    const double a = 1.0 + std::log(1.0 + reynolds);
    const double threshold = (1.2 + a / 12.0) / (1.0 + a);
    double efficiency = 4.0 / (reynolds * schmidt)
        * (1.0 + 0.4 * std::sqrt(reynolds) * std::cbrt(schmidt)
           + 0.16 * std::sqrt(reynolds) * std::sqrt(schmidt))
        + 4.0 * aerosol_diameter_m / collector_diameter_m
            * (0.02 + aerosol_diameter_m / collector_diameter_m
                * (1.0 + 2.0 * std::sqrt(reynolds)));
    if (stokes > threshold)
        efficiency += std::pow(
            (stokes - threshold) / (stokes - threshold + 0.666667), 1.5);
    if (!std::isfinite(diffusion) || diffusion <= 0.0
        || !std::isfinite(reynolds) || reynolds <= 0.0
        || !std::isfinite(schmidt) || schmidt <= 0.0
        || !std::isfinite(stokes) || !std::isfinite(threshold)
        || !std::isfinite(efficiency))
        throw std::overflow_error(
            "nonfinite Thompson aerosol collection calculation");
    return std::clamp(efficiency, 1.0e-5, 1.0);
}

AerosolScavengingTendencies aerosol_scavenging_tendencies(
    double temperature_k,
    double dry_air_density_kg_m3,
    double rain_water_kg_dry,
    double rain_number_kg_dry,
    double snow_water_kg_dry,
    double graupel_water_kg_dry,
    double water_aerosol_kg_dry,
    double ice_aerosol_kg_dry,
    double dt_s)
{
    if (!std::isfinite(temperature_k) || temperature_k <= 0.0
        || !std::isfinite(dry_air_density_kg_m3)
        || dry_air_density_kg_m3 <= 0.0
        || !std::isfinite(dt_s) || dt_s <= 0.0)
        throw std::invalid_argument("invalid Thompson aerosol scavenging state");
    require_nonnegative(rain_water_kg_dry, "rain water");
    require_nonnegative(rain_number_kg_dry, "rain number");
    require_nonnegative(snow_water_kg_dry, "snow water");
    require_nonnegative(graupel_water_kg_dry, "graupel water");
    require_nonnegative(water_aerosol_kg_dry, "water aerosol");
    require_nonnegative(ice_aerosol_kg_dry, "ice aerosol");
    const double density_factor = std::sqrt(
        rho_reference / dry_air_density_kg_m3);
    const double dynamic_viscosity = viscosity(temperature_k);
    const double water_aerosol_m3 = water_aerosol_kg_dry
        * dry_air_density_kg_m3;
    const double ice_aerosol_m3 = ice_aerosol_kg_dry
        * dry_air_density_kg_m3;
    double water_rate_m3_s = 0.0;
    double ice_rate_m3_s = 0.0;

    const double rain_mass = rain_water_kg_dry * dry_air_density_kg_m3;
    const double rain_number = rain_number_kg_dry * dry_air_density_kg_m3;
    if (rain_mass > 1.0e-12 && rain_number > 1.0e-6)
    {
        const double slope = std::cbrt(
            water_mass_coefficient * 6.0 * rain_number / rain_mass);
        const double diameter = 3.672 / slope;
        if (diameter > 50.0e-6)
        {
            const double intercept = rain_number * slope;
            const double factor = density_factor * pi * 0.25 * 4854.0
                * std::tgamma(4.0) * intercept
                / std::pow(slope + 195.0, 4.0);
            water_rate_m3_s += factor * water_aerosol_m3
                * aerosol_collection_efficiency(
                    diameter, 0.04e-6, dynamic_viscosity,
                    dry_air_density_kg_m3, temperature_k,
                    AerosolCollector::Rain);
            ice_rate_m3_s += factor * ice_aerosol_m3
                * aerosol_collection_efficiency(
                    diameter, 0.8e-6, dynamic_viscosity,
                    dry_air_density_kg_m3, temperature_k,
                    AerosolCollector::Rain);
        }
    }

    const double snow_mass = snow_water_kg_dry * dry_air_density_kg_m3;
    if (snow_mass > 1.0e-6)
    {
        const auto snow = snow_distribution(
            snow_mass, temperature_k, dry_air_density_kg_m3);
        const double diameter = snow.third_moment / snow.second_moment;
        const double riming_moment = [&] {
            constexpr double order = 2.55;
            constexpr double a[10] = {5.065339, -0.062659, -3.032362,
                0.029469, -0.000285, 0.31255, 0.000204, 0.003199,
                0.0, -0.015952};
            constexpr double b[10] = {0.476221, -0.015896, 0.165977,
                0.007468, -0.000141, 0.060366, 0.000079, 0.000594,
                0.0, -0.003577};
            const double t = std::min(-0.1, temperature_k - 273.15);
            const auto polynomial = [&](const double* c) {
                return c[0] + c[1] * t + c[2] * order
                    + c[3] * t * order + c[4] * t * t
                    + c[5] * order * order + c[6] * t * t * order
                    + c[7] * t * order * order + c[8] * t * t * t
                    + c[9] * order * order * order;
            };
            return std::pow(10.0, polynomial(a))
                * std::pow(snow.second_moment, polynomial(b));
        }();
        const double factor = density_factor * pi * 0.25 * 40.0
            * riming_moment;
        water_rate_m3_s += factor * water_aerosol_m3
            * aerosol_collection_efficiency(
                diameter, 0.04e-6, dynamic_viscosity,
                dry_air_density_kg_m3, temperature_k,
                AerosolCollector::Snow);
        ice_rate_m3_s += factor * ice_aerosol_m3
            * aerosol_collection_efficiency(
                diameter, 0.8e-6, dynamic_viscosity,
                dry_air_density_kg_m3, temperature_k,
                AerosolCollector::Snow);
    }

    const double graupel_mass = graupel_water_kg_dry
        * dry_air_density_kg_m3;
    if (graupel_mass > 1.0e-6)
    {
        const double logarithmic_content = std::log10(
            std::max(1.0e-9, graupel_mass));
        const double intercept_seed = std::pow(
            10.0, std::clamp(3.0 + 2.0 / 7.0
                             * (logarithmic_content + 8.0), 2.0, 6.0));
        const double slope = std::pow(
            intercept_seed * graupel_mass_coefficient / graupel_mass, 0.25)
            * std::cbrt(6.0);
        const double number = graupel_mass * std::pow(slope, 3.0)
            / (6.0 * graupel_mass_coefficient);
        const double intercept = number * slope;
        const double diameter = 4.0 / slope;
        const double factor = density_factor * pi * 0.25 * 442.0
            * std::tgamma(3.89) * intercept / std::pow(slope, 3.89);
        water_rate_m3_s += factor * water_aerosol_m3
            * aerosol_collection_efficiency(
                diameter, 0.04e-6, dynamic_viscosity,
                dry_air_density_kg_m3, temperature_k,
                AerosolCollector::Graupel);
        ice_rate_m3_s += factor * ice_aerosol_m3
            * aerosol_collection_efficiency(
                diameter, 0.8e-6, dynamic_viscosity,
                dry_air_density_kg_m3, temperature_k,
                AerosolCollector::Graupel);
    }

    return {
        -limited_rate(water_rate_m3_s, water_aerosol_kg_dry,
                      dry_air_density_kg_m3, dt_s),
        -limited_rate(ice_rate_m3_s, ice_aerosol_kg_dry,
                      dry_air_density_kg_m3, dt_s)};
}

} // namespace tmv::thompson
