#pragma once

#include "triple_melting.hpp"

namespace tmv::milbrandt_detail
{

struct RetainedSnowMeltingTripleState
{
    MeltingTripleMomentState rain;
    MeltingTripleMomentState snow;
};

struct RetainedSnowMeltingAcceptedEvent
{
    double entry_mass_per_kg = 0.0;
    double entry_number_per_kg = 0.0;
    double entry_reflectivity_per_kg = 0.0;
    double source_alpha = 0.0;
    double source_lambda_per_m = 0.0;
    double source_mass_squared_per_kg = 0.0;
    double admitted_mass_transfer_per_kg = 0.0;
    double admitted_number_transfer_per_kg = 0.0;
    double admitted_fraction = 0.0;
    double admitted_rain_reflectivity_per_kg = 0.0;
    double temperature_k = 0.0;
    double pressure_pa = 0.0;
    double dry_density_kg_m3 = 0.0;
    double vapor_per_kg = 0.0;
    double duration_s = 0.0;
};

struct RetainedSnowMeltingTripleResult
{
    RetainedSnowMeltingTripleState unrounded;
    MeltingTripleMomentEndpoint rain;
    MeltingTripleMomentEndpoint snow;
    RetainedSnowMeltingAcceptedEvent event;
};

inline constexpr std::array<const char*, 8> retained_snow_melting_unowned_channel_names{
    "cloud_mass", "ice_mass", "graupel_mass", "hail_mass",
    "cloud_number", "ice_number", "graupel_number", "hail_number"};

bool validate_retained_snow_melting_unowned_channels(
    const std::array<double, 8>& channels,
    std::string& error);

bool apply_retained_triple_snow_melting(
    const std::array<double, 16>& cell,
    double duration_s,
    const MeltingNoCollectionProfile& profile,
    const RetainedSnowMeltingTripleState& initial,
    const RetainedSnowMeltingAcceptedEvent& event,
    RetainedSnowMeltingTripleResult& result,
    std::string& error);

bool evaluate_retained_triple_snow_melting_cell(
    const std::array<double, 16>& cell,
    double duration_s,
    const MeltingNoCollectionProfile& profile,
    const RetainedSnowMeltingTripleState& initial,
    std::array<float, 11>& rates,
    RetainedSnowMeltingTripleResult& result,
    std::string& error);

} // namespace tmv::milbrandt_detail
