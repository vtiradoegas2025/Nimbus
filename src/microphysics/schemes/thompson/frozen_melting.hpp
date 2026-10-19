#pragma once

namespace tmv::thompson
{

struct FrozenMeltingRates
{
    double rain_kg_m3_s = 0.0;
    double snow_kg_m3_s = 0.0;
    double graupel_kg_m3_s = 0.0;
    double rain_number_m3_s = 0.0;
    double temperature_k_s = 0.0;
};

FrozenMeltingRates frozen_melting_rates(
    double pressure_pa,
    double temperature_k,
    double dry_air_density_kg_m3,
    double vapor_kg_kg_dry,
    double snow_kg_m3,
    double graupel_kg_m3,
    double dt_s);

}
