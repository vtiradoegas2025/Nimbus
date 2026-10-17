#pragma once

namespace tmv::thompson
{

enum class AerosolCollector
{
    Rain,
    Snow,
    Graupel
};

struct AerosolScavengingTendencies
{
    double water_aerosol_kg_dry_s = 0.0;
    double ice_aerosol_kg_dry_s = 0.0;
};

double aerosol_collection_efficiency(double collector_diameter_m,
                                     double aerosol_diameter_m,
                                     double dynamic_viscosity_pa_s,
                                     double air_density_kg_m3,
                                     double temperature_k,
                                     AerosolCollector collector);
AerosolScavengingTendencies aerosol_scavenging_tendencies(
    double temperature_k,
    double dry_air_density_kg_m3,
    double rain_water_kg_dry,
    double rain_number_kg_dry,
    double snow_water_kg_dry,
    double graupel_water_kg_dry,
    double water_aerosol_kg_dry,
    double ice_aerosol_kg_dry,
    double dt_s);

} // namespace tmv::thompson
