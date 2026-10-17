#pragma once

#include "melting_accepted_events.hpp"

#include <array>
#include <string>

namespace tmv::milbrandt_detail
{

struct MeltingTripleMomentState
{
    double mass_per_kg = 0.0;
    double number_per_kg = 0.0;
    double reflectivity_per_kg = 0.0;
};

struct MeltingTripleState
{
    MeltingTripleMomentState rain;
    MeltingTripleMomentState graupel;
    MeltingTripleMomentState hail;
};

struct MeltingTripleMomentEndpoint
{
    float mass_per_kg = 0.0f;
    float number_per_kg = 0.0f;
    float reflectivity_per_kg = 0.0f;
};

struct MeltingTripleResult
{
    MeltingTripleState unrounded;
    MeltingTripleMomentEndpoint rain;
    MeltingTripleMomentEndpoint graupel;
    MeltingTripleMomentEndpoint hail;
    std::array<MeltingAcceptedEvent, 2> events{};
};

bool apply_published_triple_melting(
    const std::array<double, 12>& cell,
    double duration_s,
    const MeltingNoCollectionProfile& profile,
    const MeltingTripleState& initial,
    const std::array<MeltingAcceptedEvent, 2>& events,
    MeltingTripleResult& result,
    std::string& error);

bool evaluate_published_triple_melting_cell(
    const std::array<double, 12>& cell,
    double duration_s,
    const MeltingNoCollectionProfile& profile,
    const MeltingTripleState& initial,
    std::array<float, 11>& rates,
    MeltingTripleResult& result,
    std::string& error);

} // namespace tmv::milbrandt_detail
