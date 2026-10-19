#pragma once

#include "triple_melting.hpp"

namespace tmv::milbrandt_detail
{

struct RetainedIceMeltingTripleState
{
    MeltingTripleMomentState rain;
    MeltingTripleMomentState ice;
};

struct RetainedIceMeltingAcceptedEvent
{
    double entry_mass_per_kg = 0.0;
    double entry_number_per_kg = 0.0;
    double entry_reflectivity_per_kg = 0.0;
    double source_alpha = 0.0;
    double admitted_mass_transfer_per_kg = 0.0;
    double admitted_number_transfer_per_kg = 0.0;
    double admitted_fraction = 0.0;
    double temperature_k = 0.0;
    double pressure_pa = 0.0;
    double dry_density_kg_m3 = 0.0;
    double vapor_per_kg = 0.0;
    double duration_s = 0.0;
};

struct RetainedIceMeltingTripleResult
{
    RetainedIceMeltingTripleState unrounded;
    MeltingTripleMomentEndpoint rain;
    MeltingTripleMomentEndpoint ice;
    RetainedIceMeltingAcceptedEvent event;
};

inline constexpr std::array<const char*, 8> retained_ice_melting_unowned_channel_names{
    "cloud_mass", "snow_mass", "graupel_mass", "hail_mass",
    "cloud_number", "snow_number", "graupel_number", "hail_number"};

bool validate_retained_ice_melting_unowned_channels(
    const std::array<double, 8>& channels,
    std::string& error);

bool apply_retained_triple_ice_melting(
    const std::array<double, 16>& cell,
    double duration_s,
    const MeltingNoCollectionProfile& profile,
    const RetainedIceMeltingTripleState& initial,
    const RetainedIceMeltingAcceptedEvent& event,
    RetainedIceMeltingTripleResult& result,
    std::string& error);

bool evaluate_retained_triple_ice_melting_cell(
    const std::array<double, 16>& cell,
    double duration_s,
    const MeltingNoCollectionProfile& profile,
    const RetainedIceMeltingTripleState& initial,
    std::array<float, 11>& rates,
    RetainedIceMeltingTripleResult& result,
    std::string& error);

} // namespace tmv::milbrandt_detail
