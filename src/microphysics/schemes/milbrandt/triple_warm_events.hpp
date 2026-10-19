#pragma once

#include <string>
#include <vector>

namespace tmv::milbrandt_detail
{

enum class AcceptedRainEventKind
{
    Autoconversion,
    Evaporation,
    SelfCollection,
    Accretion,
    PreRainNumberRegularization,
    RainNumberClip,
    PostRainNumberRegularization,
    RainToCloud,
    RainReset,
    Breakup
};

struct AcceptedRainEvent
{
    AcceptedRainEventKind kind = AcceptedRainEventKind::Autoconversion;
    double admitted_mass_rate_per_kg_s = 0.0;
    double admitted_number_rate_per_kg_s = 0.0;
    double duration_s = 0.0;
    double source_number_rate_per_m3_s = 0.0;
    double source_dry_density_kg_m3 = 0.0;
    double source_cloud_mass_rate_per_kg_s = 0.0;
    double source_cloud_number_rate_per_m3_s = 0.0;
};

struct RainTripleMomentState
{
    float mass_per_kg = 0.0f;
    float number_per_kg = 0.0f;
    float reflectivity_per_kg = 0.0f;
};

bool make_admitted_autoconversion_event(
    double admitted_mass_rate_per_kg_s,
    double admitted_number_rate_per_m3_s,
    double dry_density_kg_m3,
    double duration_s,
    AcceptedRainEvent& event,
    std::string& error);

bool make_admitted_same_category_event(
    AcceptedRainEventKind kind,
    double admitted_mass_rate_per_kg_s,
    double admitted_number_rate_per_kg_s,
    double duration_s,
    AcceptedRainEvent& event,
    std::string& error);

bool make_admitted_rain_number_correction_event(
    AcceptedRainEventKind kind,
    double admitted_rain_number_rate_per_m3_s,
    double dry_density_kg_m3,
    double duration_s,
    AcceptedRainEvent& event,
    std::string& error);

bool make_admitted_rain_removal_event(
    AcceptedRainEventKind kind,
    double admitted_cloud_mass_rate_per_kg_s,
    double admitted_rain_mass_rate_per_kg_s,
    double admitted_cloud_number_rate_per_m3_s,
    double admitted_rain_number_rate_per_m3_s,
    double dry_density_kg_m3,
    double duration_s,
    AcceptedRainEvent& event,
    std::string& error);

bool admitted_rain_reflectivity_rate(
    const RainTripleMomentState& state,
    const AcceptedRainEvent& event,
    double& rate_per_kg_s,
    std::string& error);

bool apply_admitted_rain_events(
    const RainTripleMomentState& initial,
    const std::vector<AcceptedRainEvent>& events,
    RainTripleMomentState& result,
    std::string& error);

} // namespace tmv::milbrandt_detail
