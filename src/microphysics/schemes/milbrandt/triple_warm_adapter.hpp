#pragma once

#include "triple_warm_phase.hpp"
#include "warm_rain.hpp"

#include <string>

namespace tmv::milbrandt_detail
{

bool map_observed_rain_event(
    const WarmRainAcceptedEvent& event,
    double dt,
    double expected_density,
    AcceptedRainPhaseBatch& batch,
    std::string& error);

bool map_triple_produced_rain_event(
    const WarmRainAcceptedEvent& event,
    double dt,
    double expected_density,
    AcceptedRainPhaseBatch& batch,
    std::string& error);

bool evaluate_observed_triple_warm_cell(
    const WarmRainParameters& parameters,
    const WarmRainCell& cell,
    double dt,
    const WarmRainTendencies& incoming_rates,
    const RainTripleMomentState& initial_rain,
    WarmRainTendencies& output_rates,
    RainTripleMomentState& output_rain,
    std::string& error);

bool evaluate_triple_only_warm_cell(
    const WarmRainParameters& parameters,
    const WarmRainCell& cell,
    double dt,
    const WarmRainTendencies& incoming_rates,
    const RainTripleMomentState& initial_rain,
    WarmRainTendencies& output_rates,
    RainTripleMomentState& output_rain,
    std::string& error);

bool map_dynamic_triple_produced_rain_event(
    const WarmRainAcceptedEvent& event,
    double dt,
    double expected_density,
    AcceptedRainPhaseBatch& batch,
    std::string& error);

} // namespace tmv::milbrandt_detail
