#pragma once

namespace tmv::thompson
{
struct LiquidSaturationResolution
{
    bool condensation;
    bool evaporation;
};
LiquidSaturationResolution
liquid_saturation_resolution(float temperature, float pressure, float vapor, float evaporated_cloud = 0);
LiquidSaturationResolution
liquid_saturation_resolution_from_theta(float theta, float pressure, float vapor, float evaporated_cloud = 0);

double liquid_saturation_mixing_ratio(double pressure_pa, double temperature_k);
double dewpoint_temperature(double pressure_pa, double mixing_ratio);
double lifting_condensation_temperature(double temperature_k, double dewpoint_k);
double equivalent_potential_temperature(double pressure_pa,
                                        double temperature_k,
                                        double mixing_ratio,
                                        double lcl_temperature_k);
double wet_bulb_temperature(double pressure_pa, double temperature_k, double mixing_ratio);

} // namespace tmv::thompson
