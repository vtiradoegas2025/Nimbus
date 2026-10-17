#include "nucleation.hpp"
#include "microphysics/schemes/thompson/moment_initialization.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace tmv::thompson
{
namespace
{

constexpr double pi = 3.1415926536;
constexpr double universal_gas_constant = 8.314;
constexpr double reference_density = 101325.0 / (287.05 * 273.15);
constexpr double aqueous_aerosol_volume_cm3 =
    4.0 / 3.0 * pi * 2.5e-6 * 2.5e-6 * 2.5e-6;

void require_nonnegative_finite(double value)
{
    if (!std::isfinite(value) || value < 0.0)
        throw std::invalid_argument("invalid Thompson aerosol-number state");
}

} // namespace

NumberExchange activate_cloud_droplets(const ActivationTable& table,
                                       double                 temperature_k,
                                       double                 updraft_m_s,
                                       double                 dry_air_density_kg_m3,
                                       double                 cloud_number_kg_dry,
                                       double                 water_aerosol_kg_dry,
                                       double                 cloud_water_kg_m3)
{
    if (!std::isfinite(dry_air_density_kg_m3) || dry_air_density_kg_m3 <= 0.0)
        throw std::invalid_argument("invalid Thompson dry-air density");
    require_nonnegative_finite(cloud_number_kg_dry);
    require_nonnegative_finite(water_aerosol_kg_dry);
    require_nonnegative_finite(cloud_water_kg_m3);
    const double available_m3 = water_aerosol_kg_dry * dry_air_density_kg_m3;
    const double target_m3 = std::max(2.0, activated_cloud_number_m3(table, temperature_k, updraft_m_s, available_m3));
    const double requested_kg_dry = std::max(0.0, target_m3 / dry_air_density_kg_m3 - cloud_number_kg_dry);
    const double requested        = std::min(requested_kg_dry, water_aerosol_kg_dry);
    if (cloud_water_kg_m3 == 0)
    {
        return {};
    }
    // A size-bound projection must not discard newly activated particles after
    // their aerosol donor has already been consumed.
    const double supported =
        normalize_cloud_number_m3(cloud_water_kg_m3, dry_air_density_kg_m3 * (cloud_number_kg_dry + requested)) /
        dry_air_density_kg_m3;
    const double admitted = std::min(requested, std::max(0.0, supported - cloud_number_kg_dry));
    return {admitted, -admitted};
}

NumberExchange return_evaporated_cloud_droplets(double cloud_number_removed_kg_dry)
{
    require_nonnegative_finite(cloud_number_removed_kg_dry);
    return {-cloud_number_removed_kg_dry, cloud_number_removed_kg_dry};
}

double demott_ice_number_m3(double temperature_c,
                            double dry_air_density_kg_m3,
                            double ice_aerosol_m3)
{
    if (!std::isfinite(temperature_c) || temperature_c > 0.0
        || !std::isfinite(dry_air_density_kg_m3)
        || dry_air_density_kg_m3 <= 0.0)
        throw std::invalid_argument("invalid Thompson DeMott input");
    require_nonnegative_finite(ice_aerosol_m3);
    const double aerosol_cm3 = std::max(
        0.5,
        ice_aerosol_m3 * reference_density * 1.0e-6
            / dry_air_density_kg_m3);
    const double number = 5.94e-5 * std::pow(-temperature_c, 3.33)
                          * std::pow(
                              aerosol_cm3,
                              -0.0264 * temperature_c + 0.0033)
                          * dry_air_density_kg_m3 / reference_density
                          * 1000.0;
    return std::max(0.0, number);
}

double koop_frozen_aerosol_number_m3(double temperature_k,
                                     double vapor_mixing_ratio,
                                     double saturation_mixing_ratio_water,
                                     double water_aerosol_m3,
                                     double dt_s)
{
    if (!std::isfinite(temperature_k) || temperature_k <= 0.0
        || !std::isfinite(vapor_mixing_ratio) || vapor_mixing_ratio < 0.0
        || !std::isfinite(saturation_mixing_ratio_water)
        || saturation_mixing_ratio_water <= 0.0
        || !std::isfinite(dt_s) || dt_s <= 0.0)
        throw std::invalid_argument("invalid Thompson Koop input");
    require_nonnegative_finite(water_aerosol_m3);

    const double saturation =
        vapor_mixing_ratio / saturation_mixing_ratio_water;
    const double chemical_potential =
        210368.0 + 131.438 * temperature_k - 3.32373e6 / temperature_k
        - 41729.1 * std::log(temperature_k);
    const double ice_water_activity = std::exp(
        chemical_potential / (universal_gas_constant * temperature_k));
    const double activity_difference = saturation - ice_water_activity;
    const double log_rate = std::min(
        20.0,
        -906.7 + 8502.0 * activity_difference
            - 26924.0 * activity_difference * activity_difference
            + 29180.0 * activity_difference * activity_difference
                  * activity_difference);
    const double rate_cm3_s = std::pow(10.0, log_rate);
    const double probability = std::min(
        1.0 - std::exp(-rate_cm3_s * aqueous_aerosol_volume_cm3 * dt_s),
        1.0);
    if (probability <= 0.0) return 0.0;
    return std::max(
        0.0, std::min(probability * water_aerosol_m3, 1000.0e3));
}

IceNucleationExchange nucleate_aerosol_ice(
    double temperature_k,
    double vapor_mixing_ratio,
    double saturation_mixing_ratio_water,
    double saturation_mixing_ratio_ice,
    double dry_air_density_kg_m3,
    double ice_number_kg_dry,
    double snow_number_m3,
    double water_aerosol_kg_dry,
    double ice_aerosol_kg_dry,
    double dt_s)
{
    if (!std::isfinite(temperature_k) || temperature_k <= 0.0
        || !std::isfinite(vapor_mixing_ratio) || vapor_mixing_ratio < 0.0
        || !std::isfinite(saturation_mixing_ratio_water)
        || saturation_mixing_ratio_water <= 0.0
        || !std::isfinite(saturation_mixing_ratio_ice)
        || saturation_mixing_ratio_ice <= 0.0
        || !std::isfinite(dry_air_density_kg_m3)
        || dry_air_density_kg_m3 <= 0.0
        || !std::isfinite(snow_number_m3) || snow_number_m3 < 0.0
        || !std::isfinite(dt_s) || dt_s <= 0.0)
    {
        throw std::invalid_argument("invalid Thompson ice-nucleation state");
    }
    require_nonnegative_finite(ice_number_kg_dry);
    require_nonnegative_finite(water_aerosol_kg_dry);
    require_nonnegative_finite(ice_aerosol_kg_dry);
    if (temperature_k >= 273.15) return {};

    constexpr double epsilon = 1.0e-15;
    constexpr double initial_ice_mass_kg = 1.0e-12;
    const double water_supersaturation =
        vapor_mixing_ratio / saturation_mixing_ratio_water - 1.0;
    const double ice_supersaturation =
        vapor_mixing_ratio / saturation_mixing_ratio_ice - 1.0;
    const double ice_number_m3 =
        ice_number_kg_dry * dry_air_density_kg_m3;
    const double ice_aerosol_m3 =
        ice_aerosol_kg_dry * dry_air_density_kg_m3;
    const double water_aerosol_m3 =
        water_aerosol_kg_dry * dry_air_density_kg_m3;

    double demott_number_m3 = 0.0;
    if (ice_supersaturation >= 0.25
        || (water_supersaturation > epsilon && temperature_k < 253.15))
    {
        const double target = demott_ice_number_m3(
            temperature_k - 273.15,
            dry_air_density_kg_m3,
            ice_aerosol_m3);
        demott_number_m3 = std::min(ice_aerosol_m3, std::max(0.0, target - ice_number_m3));
    }

    double koop_number_m3 = 0.0;
    if (snow_number_m3 + ice_number_m3 + demott_number_m3 <= 999.0e3
        && temperature_k < 238.0 && ice_supersaturation >= 0.4)
    {
        koop_number_m3 = koop_frozen_aerosol_number_m3(
            temperature_k,
            vapor_mixing_ratio,
            saturation_mixing_ratio_water,
            water_aerosol_m3,
            dt_s);
    }

    const double requested_demott_mass =
        initial_ice_mass_kg * demott_number_m3;
    const double requested_koop_mass =
        0.1 * initial_ice_mass_kg * koop_number_m3;
    const double requested_mass =
        requested_demott_mass + requested_koop_mass;
    const double available_vapor_mass = std::max(
        0.0,
        (vapor_mixing_ratio - saturation_mixing_ratio_ice)
            * dry_air_density_kg_m3 * 0.999);
    const double admitted = requested_mass > 0.0
        ? std::min(1.0, available_vapor_mass / requested_mass)
        : 0.0;
    const double demott_admitted_m3 = demott_number_m3 * admitted;
    const double koop_admitted_m3 = koop_number_m3 * admitted;
    const double mass_m3 =
        initial_ice_mass_kg * demott_admitted_m3
        + 0.1 * initial_ice_mass_kg * koop_admitted_m3;
    const double inverse_density = 1.0 / dry_air_density_kg_m3;

    IceNucleationExchange result;
    result.vapor_mass_kg_dry = -mass_m3 * inverse_density;
    result.ice_mass_kg_dry = mass_m3 * inverse_density;
    result.ice_number_kg_dry =
        (demott_admitted_m3 + koop_admitted_m3) * inverse_density;
    result.water_aerosol_kg_dry = -koop_admitted_m3 * inverse_density;
    result.ice_aerosol_kg_dry = -demott_admitted_m3 * inverse_density;
    return result;
}

} // namespace tmv::thompson
