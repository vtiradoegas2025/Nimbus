#pragma once

#include "cold_rain_freezing_events.hpp"
#include "triple_moment.hpp"

#include <string>

namespace tmv::milbrandt_detail
{

struct ColdFreezingTripleMomentState
{
    double mass_per_kg = 0.0;
    double number_per_kg = 0.0;
    double reflectivity_per_kg = 0.0;
};

struct ColdFreezingTripleMomentEndpoint
{
    float mass_per_kg = 0.0f;
    float number_per_kg = 0.0f;
    float reflectivity_per_kg = 0.0f;
};

struct ColdFreezingTripleMomentResult
{
    ColdFreezingTripleMomentEndpoint rain;
    ColdFreezingTripleMomentEndpoint frozen;
};

bool apply_equal_fraction_rain_freezing(
    const ColdFreezingTripleMomentState& rain,
    const ColdFreezingTripleMomentState& frozen,
    double dry_density_kg_m3,
    double duration_s,
    const ColdRainFreezingAcceptedEvent& event,
    TripleMomentMassLaw rain_mass_law,
    TripleMomentMassLaw frozen_mass_law,
    ColdFreezingTripleMomentResult& result,
    std::string& error);

} // namespace tmv::milbrandt_detail
