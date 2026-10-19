#pragma once

namespace tmv::thompson
{

struct MassNumberFallSpeeds
{
    double mass_m_s = 0.0;
    double number_m_s = 0.0;
};

struct SnowDistribution
{
    double second_moment = 0.0;
    double third_moment = 0.0;
    double number_m3 = 0.0;
    double mass_fall_speed_m_s = 0.0;
};

MassNumberFallSpeeds cloud_fall_speeds(double cloud_water_kg_m3,
                                       double cloud_number_m3,
                                       double air_density_kg_m3);
MassNumberFallSpeeds rain_fall_speeds(double rain_water_kg_m3,
                                      double rain_number_m3,
                                      double air_density_kg_m3);
MassNumberFallSpeeds ice_fall_speeds(double ice_water_kg_m3,
                                     double ice_number_m3,
                                     double air_density_kg_m3);
SnowDistribution snow_distribution(double snow_water_kg_m3,
                                   double temperature_k,
                                   double air_density_kg_m3);
MassNumberFallSpeeds graupel_fall_speeds(double graupel_water_kg_m3,
                                         double air_density_kg_m3);

} // namespace tmv::thompson
