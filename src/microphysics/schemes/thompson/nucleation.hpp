#pragma once

#include "microphysics/schemes/thompson/activation_table.hpp"

namespace tmv::thompson
{

struct NumberExchange
{
    double hydrometeor_kg_dry = 0.0;
    double aerosol_kg_dry = 0.0;
};

struct IceNucleationExchange
{
    double vapor_mass_kg_dry = 0.0;
    double ice_mass_kg_dry = 0.0;
    double ice_number_kg_dry = 0.0;
    double water_aerosol_kg_dry = 0.0;
    double ice_aerosol_kg_dry = 0.0;
};

NumberExchange activate_cloud_droplets(const ActivationTable& table,
                                       double                 temperature_k,
                                       double                 updraft_m_s,
                                       double                 dry_air_density_kg_m3,
                                       double                 cloud_number_kg_dry,
                                       double                 water_aerosol_kg_dry,
                                       double                 cloud_water_kg_m3);

NumberExchange return_evaporated_cloud_droplets(
    double cloud_number_removed_kg_dry);

double demott_ice_number_m3(double temperature_c,
                            double dry_air_density_kg_m3,
                            double ice_aerosol_m3);

double koop_frozen_aerosol_number_m3(double temperature_k,
                                     double vapor_mixing_ratio,
                                     double saturation_mixing_ratio_water,
                                     double water_aerosol_m3,
                                     double dt_s);

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
    double dt_s);

} // namespace tmv::thompson
