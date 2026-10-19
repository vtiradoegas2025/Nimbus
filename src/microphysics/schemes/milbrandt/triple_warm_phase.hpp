#pragma once

#include "triple_warm_events.hpp"

#include <array>
#include <cstddef>
#include <string>

namespace tmv::milbrandt_detail
{

enum class RainEventPhase
{
    PreRegularization,
    SimultaneousPhysical,
    NumericalCorrection,
    Evaporation
};

struct PhasedAcceptedRainEvent
{
    RainEventPhase phase = RainEventPhase::PreRegularization;
    AcceptedRainEvent event{};
};

struct AcceptedRainPhaseBatch
{
    std::array<PhasedAcceptedRainEvent, 16> events{};
    std::size_t size = 0;
    double duration_s = 0.0;
};

struct ExactRainTripleMomentState
{
    double mass_per_kg = 0.0;
    double number_per_kg = 0.0;
    double reflectivity_per_kg = 0.0;
};

enum class DynamicWarmStepFailure
{
    None,
    InvalidSource,
    IntervalRejected,
    Resource
};

struct AdmittedRainEvaporationEndpoint
{
    bool present = false;
    double mass_before_per_kg = 0.0;
    double number_before_per_kg = 0.0;
    double admitted_mass_per_kg = 0.0;
    double mass_after_per_kg = 0.0;
    double number_after_per_kg = 0.0;
    bool donor_exhausted = false;
};

bool append_admitted_rain_phase_event(
    AcceptedRainPhaseBatch& batch,
    RainEventPhase phase,
    const AcceptedRainEvent& event,
    std::string& error);

bool apply_admitted_rain_phases(
    const RainTripleMomentState& initial,
    const AcceptedRainPhaseBatch& batch,
    RainTripleMomentState& result,
    std::string& error);

bool apply_admitted_rain_phases_exact(
    const RainTripleMomentState& initial,
    const AcceptedRainPhaseBatch& batch,
    RainTripleMomentState& rounded,
    ExactRainTripleMomentState& exact,
    std::string& error);

bool apply_admitted_rain_phases_with_evaporation_endpoint(
    const RainTripleMomentState& initial,
    const AcceptedRainPhaseBatch& batch,
    const AdmittedRainEvaporationEndpoint& endpoint,
    RainTripleMomentState& rounded,
    ExactRainTripleMomentState& exact,
    std::string& error);

bool finalize_rain_phase_endpoint(
    double mass_per_kg,
    double number_per_kg,
    double reflectivity_per_kg,
    RainTripleMomentState& result,
    std::string& error);

bool apply_admitted_rain_phase_prefix_exact(
    const RainTripleMomentState& initial,
    const AcceptedRainPhaseBatch& batch,
    ExactRainTripleMomentState& result,
    std::string& error);

bool apply_admitted_rain_phase_prefix_fp64(
    const ExactRainTripleMomentState& initial,
    const AcceptedRainPhaseBatch& batch,
    ExactRainTripleMomentState& result,
    std::string& error,
    DynamicWarmStepFailure* failure = nullptr);

bool apply_admitted_rain_phases_with_evaporation_endpoint_fp64(
    const ExactRainTripleMomentState& initial,
    const AcceptedRainPhaseBatch& batch,
    const AdmittedRainEvaporationEndpoint& endpoint,
    ExactRainTripleMomentState& result,
    std::string& error,
    DynamicWarmStepFailure* failure = nullptr);

} // namespace tmv::milbrandt_detail
