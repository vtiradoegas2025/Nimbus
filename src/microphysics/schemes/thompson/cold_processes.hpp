#pragma once

#include "microphysics/schemes/thompson/cold_process_tables.hpp"

namespace tmv::thompson
{

struct RainFreezingState
{
    double temperature_k = 0.0;
    double rain_kg_m3 = 0.0;
    double rain_number_m3 = 0.0;
    double ice_nuclei_m3 = 0.0;
};

struct RainFreezingExchange
{
    double rain_kg_m3 = 0.0;
    double ice_kg_m3 = 0.0;
    double graupel_kg_m3 = 0.0;
    double rain_number_m3 = 0.0;
    double ice_number_m3 = 0.0;
    double graupel_number_m3 = 0.0;
};

struct CloudFreezingState
{
    double temperature_k = 0.0;
    double cloud_kg_m3 = 0.0;
    double cloud_number_m3 = 0.0;
    double ice_nuclei_m3 = 0.0;
};

struct CloudFreezingExchange
{
    double cloud_kg_m3 = 0.0;
    double ice_kg_m3 = 0.0;
    double cloud_number_m3 = 0.0;
    double ice_number_m3 = 0.0;
};

struct IceDepositionState
{
    double pressure_pa = 0.0;
    double temperature_k = 0.0;
    double dry_air_density_kg_m3 = 0.0;
    double vapor_kg_kg_dry = 0.0;
    double ice_kg_m3 = 0.0;
    double ice_number_m3 = 0.0;
    double snow_kg_m3 = 0.0;
    double graupel_kg_m3 = 0.0;
};

struct IceDepositionExchange
{
    double vapor_kg_m3_s = 0.0;
    double ice_kg_m3_s = 0.0;
    double snow_kg_m3_s = 0.0;
    double ice_number_m3_s = 0.0;
    double temperature_k_s = 0.0;
    double graupel_kg_m3_s = 0.0;
    double snow_vapor_kg_m3_s = 0.0;
};

struct SnowRimingState
{
    double temperature_k = 0.0;
    double dry_air_density_kg_m3 = 0.0;
    double cloud_kg_m3 = 0.0;
    double cloud_number_m3 = 0.0;
    double snow_kg_m3 = 0.0;
    double graupel_kg_m3 = 0.0;
    double snow_deposition_kg_m3_s = 0.0;
};

struct SnowRimingExchange
{
    double cloud_kg_m3_s = 0.0;
    double snow_kg_m3_s = 0.0;
    double cloud_number_m3_s = 0.0;
    double temperature_k_s = 0.0;
    double graupel_kg_m3_s = 0.0;
    double ice_kg_m3_s = 0.0;
    double ice_number_m3_s = 0.0;
    double snow_fall_speed_multiplier = 1.0;
};

struct IceSnowConversionState
{
    double ice_kg_m3 = 0.0;
    double ice_number_m3 = 0.0;
};

struct IceSnowConversionExchange
{
    double ice_kg_m3_s = 0.0;
    double snow_kg_m3_s = 0.0;
    double ice_number_m3_s = 0.0;
};

std::vector<double> native_riming_coefficients();

RainFreezingIndex rain_freezing_lookup_index(const RainFreezingState& state);

RainFreezingExchange rain_freezing_exchange(
    const RainFreezingCell& cell,
    const RainFreezingState& state);

RainFreezingExchange rain_freezing_exchange(
    const ColdProcessTables& tables,
    const RainFreezingState& state);

CloudFreezingIndex cloud_freezing_lookup_index(
    const CloudFreezingState& state);

CloudFreezingExchange cloud_freezing_exchange(
    const CloudFreezingCell& cell,
    const CloudFreezingState& state);

CloudFreezingExchange cloud_freezing_exchange(
    const ColdProcessTables& tables,
    const CloudFreezingState& state);

IceSnowIndex ice_deposition_lookup_index(const IceDepositionState& state);

IceDepositionExchange ice_deposition_rates(
    const ColdProcessTables& tables,
    const IceDepositionState& state,
    double dt_s);

SnowRimingExchange snow_riming_rates(
    const SnowRimingState& state,
    double dt_s);

IceSnowConversionExchange ice_to_snow_rates(
    const ColdProcessTables& tables,
    const IceSnowConversionState& state,
    double dt_s);

} // namespace tmv::thompson
