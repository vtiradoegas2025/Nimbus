#pragma once

#include <memory>
#include <vector>

namespace tmv::thompson
{

std::shared_ptr<const std::vector<float>> native_warm_rain_coefficients();

struct WarmRainConversionRates
{
    double cloud_water_kg_m3_s = 0.0;
    double rain_water_kg_m3_s = 0.0;
    double cloud_number_m3_s = 0.0;
    double rain_number_m3_s = 0.0;
};

struct RainEvaporationRates
{
    double vapor_kg_m3_s = 0.0;
    double rain_water_kg_m3_s = 0.0;
    double rain_number_m3_s = 0.0;
    double water_aerosol_m3_s = 0.0;
    double temperature_k_s = 0.0;
};

WarmRainConversionRates warm_rain_conversion_rates(
    double temperature_k,
    double dry_air_density_kg_m3,
    double cloud_water_kg_m3,
    double rain_water_kg_m3,
    double cloud_number_m3,
    double rain_number_m3,
    double dt_s);

RainEvaporationRates rain_evaporation_rates(
    double pressure_pa,
    double temperature_k,
    double dry_air_density_kg_m3,
    double vapor_kg_kg_dry,
    double rain_water_kg_m3,
    double rain_number_m3,
    double dt_s);

}
