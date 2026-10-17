#pragma once

#include "cold_frozen_diffusion_events.hpp"
#include "cold_joint_frozen_diffusion_events.hpp"
#include "triple_moment.hpp"

#include <array>
#include <string>

namespace tmv::milbrandt_detail
{

struct ColdFrozenDiffusionTripleState
{
    double mass_per_kg = 0.0;
    double number_per_kg = 0.0;
    double reflectivity_per_kg = 0.0;
};

struct ColdFrozenDiffusionTripleEndpoint
{
    float mass_per_kg = 0.0f;
    float number_per_kg = 0.0f;
    float reflectivity_per_kg = 0.0f;
};

struct ColdJointFrozenDiffusionTripleState
{
    ColdFrozenDiffusionTripleState graupel;
    ColdFrozenDiffusionTripleState hail;
};

struct ColdJointFrozenDiffusionTripleResult
{
    ColdJointFrozenDiffusionTripleState diffusion_endpoint;
    ColdJointFrozenDiffusionTripleState unrounded;
    ColdFrozenDiffusionTripleEndpoint graupel;
    ColdFrozenDiffusionTripleEndpoint hail;
    std::array<ColdFrozenDiffusionAcceptedEvent, 2> diffusion_events{};
    ColdTerminalHailToGraupelAcceptedEvent terminal_conversion;
};

bool apply_published_frozen_diffusion(
    const ColdFrozenDiffusionTripleState& initial,
    double dry_density_kg_m3,
    double duration_s,
    const ColdFrozenDiffusionAcceptedEvent& event,
    TripleMomentMassLaw mass_law,
    ColdFrozenDiffusionTripleEndpoint& result,
    std::string& error);

bool evaluate_published_triple_frozen_diffusion_cell(
    const std::array<double, 16>& cell,
    double duration_s,
    ColdFrozenDiffusionSpecies species,
    const ColdFrozenDiffusionTripleState& initial,
    std::array<float, 14>& rates,
    ColdFrozenDiffusionTripleEndpoint& result,
    std::string& error);

bool apply_published_joint_frozen_diffusion(
    const ColdJointFrozenDiffusionTripleState& initial,
    double dry_density_kg_m3,
    double duration_s,
    const std::array<ColdFrozenDiffusionAcceptedEvent, 2>& events,
    ColdJointFrozenDiffusionTripleResult& result,
    std::string& error);

bool evaluate_published_joint_triple_frozen_diffusion_cell(
    const std::array<double, 16>& cell,
    double duration_s,
    const ColdJointFrozenDiffusionTripleState& initial,
    std::array<float, 14>& rates,
    ColdJointFrozenDiffusionTripleResult& result,
    std::string& error);

} // namespace tmv::milbrandt_detail
