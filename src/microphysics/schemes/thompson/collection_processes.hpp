#pragma once

#include "microphysics/schemes/thompson/collection_tables.hpp"

namespace tmv::thompson
{

struct CollectionProcessRates
{
    double rain_kg_m3_s = 0.0;
    double snow_kg_m3_s = 0.0;
    double graupel_kg_m3_s = 0.0;
    double rain_number_m3_s = 0.0;
};

RainSnowIndex rain_snow_lookup_index(double rain_water_kg_m3,
                                     double rain_number_m3,
                                     double snow_water_kg_m3,
                                     double wet_bulb_temperature_k);
RainGraupelIndex rain_graupel_lookup_index(double rain_water_kg_m3,
                                           double rain_number_m3,
                                           double graupel_water_kg_m3);
CollectionProcessRates collection_process_rates(
    const CollectionTables& tables,
    double rain_water_kg_m3,
    double rain_number_m3,
    double snow_water_kg_m3,
    double graupel_water_kg_m3,
    double wet_bulb_temperature_k,
    double dt_s);

}
