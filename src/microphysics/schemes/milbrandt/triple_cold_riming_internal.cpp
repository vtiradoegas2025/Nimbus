#include "milbrandt.hpp"
#include "cold.hpp"
#include "triple_cold_riming.hpp"
#include "triple_cold_ice_graupel.hpp"
#include "triple_cold_graupel_conversions.hpp"
#include "triple_cold_sll.hpp"
#include "triple_cold_collected_sll.hpp"

#include <cmath>
#include <limits>

namespace tmv
{

bool MilbrandtScheme::enable_internal_triple_cold_riming_for_test(
    const SimulationStateView& state,
    const Field3D& cloud_number,
    const std::array<const Field3D*, 4>& frozen_number,
    const std::array<const Field3D*, 4>& frozen_reflectivity,
    std::string& error)
{
    return enable_internal_triple_primary_moments_for_test(state,
        {nullptr, frozen_number[0], frozen_number[1], frozen_number[2], frozen_number[3]},
        {nullptr, frozen_reflectivity[0], frozen_reflectivity[1], frozen_reflectivity[2], frozen_reflectivity[3]},
        InternalTripleMode::ColdRimingPrimaryOnly, error, &cloud_number);
}

bool MilbrandtScheme::enable_internal_triple_cold_ice_graupel_for_test(
    const SimulationStateView& state,
    const Field3D& cloud_number,
    const std::array<const Field3D*, 4>& frozen_number,
    const std::array<const Field3D*, 4>& frozen_reflectivity,
    std::string& error)
{
    return enable_internal_triple_primary_moments_for_test(state,
        {nullptr, frozen_number[0], frozen_number[1], frozen_number[2], frozen_number[3]},
        {nullptr, frozen_reflectivity[0], frozen_reflectivity[1], frozen_reflectivity[2], frozen_reflectivity[3]},
        InternalTripleMode::ColdIceGraupelPrimaryOnly, error, &cloud_number);
}

bool MilbrandtScheme::enable_internal_triple_cold_graupel_conversions_for_test(
    const SimulationStateView& state,
    const Field3D& cloud_number,
    const std::array<const Field3D*, 4>& frozen_number,
    const std::array<const Field3D*, 4>& frozen_reflectivity,
    std::string& error)
{
    return enable_internal_triple_primary_moments_for_test(state,
        {nullptr, frozen_number[0], frozen_number[1], frozen_number[2], frozen_number[3]},
        {nullptr, frozen_reflectivity[0], frozen_reflectivity[1], frozen_reflectivity[2], frozen_reflectivity[3]},
        InternalTripleMode::ColdGraupelConversionsPrimaryOnly, error, &cloud_number);
}

bool MilbrandtScheme::enable_internal_triple_cold_homogeneous_for_test(
    const SimulationStateView& state,
    const Field3D& cloud_number,
    const std::array<const Field3D*, 4>& frozen_number,
    const std::array<const Field3D*, 4>& frozen_reflectivity,
    std::string& error)
{
    return enable_internal_triple_primary_moments_for_test(state,
        {nullptr, frozen_number[0], frozen_number[1], frozen_number[2], frozen_number[3]},
        {nullptr, frozen_reflectivity[0], frozen_reflectivity[1], frozen_reflectivity[2], frozen_reflectivity[3]},
        InternalTripleMode::ColdHomogeneousPrimaryOnly, error, &cloud_number);
}

bool MilbrandtScheme::enable_internal_triple_cold_sll_for_test(
    const SimulationStateView& state,
    const Field3D& cloud_number,
    const std::array<const Field3D*, 4>& frozen_number,
    const std::array<const Field3D*, 4>& frozen_reflectivity,
    std::string& error)
{
    return enable_internal_triple_primary_moments_for_test(state,
        {nullptr, frozen_number[0], frozen_number[1], frozen_number[2], frozen_number[3]},
        {nullptr, frozen_reflectivity[0], frozen_reflectivity[1], frozen_reflectivity[2], frozen_reflectivity[3]},
        InternalTripleMode::ColdSllPrimaryOnly, error, &cloud_number);
}

bool MilbrandtScheme::enable_internal_triple_cold_collected_sll_for_test(
    const SimulationStateView& state,
    const Field3D& cloud_number,
    const std::array<const Field3D*, 4>& frozen_number,
    const std::array<const Field3D*, 4>& frozen_reflectivity,
    std::string& error)
{
    return enable_internal_triple_primary_moments_for_test(state,
        {nullptr, frozen_number[0], frozen_number[1], frozen_number[2], frozen_number[3]},
        {nullptr, frozen_reflectivity[0], frozen_reflectivity[1], frozen_reflectivity[2], frozen_reflectivity[3]},
        InternalTripleMode::ColdCollectedSllPrimaryOnly, error, &cloud_number);
}

bool MilbrandtScheme::enable_internal_triple_cold_rain_sll_for_test(
    const SimulationStateView& state, const Field3D& cloud_number,
    const std::array<const Field3D*, 5>& number,
    const std::array<const Field3D*, 5>& reflectivity, std::string& error)
{
    return enable_internal_triple_primary_moments_for_test(state, number, reflectivity,
        InternalTripleMode::ColdRainSllPrimaryOnly, error, &cloud_number);
}

bool MilbrandtScheme::enable_internal_triple_cold_collision_sll_for_test(
    const SimulationStateView& state, const Field3D& cloud_number,
    const std::array<const Field3D*, 5>& number,
    const std::array<const Field3D*, 5>& reflectivity, std::string& error)
{
    return enable_internal_triple_primary_moments_for_test(state, number, reflectivity,
        InternalTripleMode::ColdCollisionSllPrimaryOnly, error, &cloud_number);
}

bool MilbrandtScheme::enable_internal_triple_cold_wet_growth_sll_for_test(
    const SimulationStateView& state, const Field3D& cloud_number,
    const std::array<const Field3D*, 5>& number,
    const std::array<const Field3D*, 5>& reflectivity, std::string& error)
{
    return enable_internal_triple_primary_moments_for_test(state, number, reflectivity,
        InternalTripleMode::ColdWetGrowthSllPrimaryOnly, error, &cloud_number);
}

bool MilbrandtScheme::evaluate_internal_triple_cold_riming_primary_cell(
    const std::array<double, 16>& cell,
    const std::array<float, 5>& reflectivity,
    double dt,
    InternalTriplePrimaryCellCandidate& result,
    std::string& error) const
{
    const milbrandt_detail::ColdParameters parameters{
        a_g_, a_h_, a_i_, a_r_, a_s_, alpha_g_, alpha_h_, alpha_i_, alpha_r_, alpha_s_,
        b_g_, b_h_, b_i_, b_r_, b_s_, c_g_, c_h_, c_i_, c_r_, c_s_, d_g_, d_h_, d_i_, d_r_, d_s_, hail_processes_};
    milbrandt_detail::ColdRimingCell source;
    source.temperature_k = cell[0];
    source.pressure_pa = cell[1];
    source.dry_density_kg_m3 = cell[2];
    source.initial.vapor_mass_per_kg = cell[3];
    source.initial.cloud_mass_per_kg = cell[4];
    source.initial.cloud_number_per_kg = cell[10];
    for (std::size_t species = 0; species < source.initial.frozen.size(); ++species)
    {
        source.initial.frozen[species] = {cell[species + 6], cell[species + 12], reflectivity[species + 1]};
    }
    milbrandt_detail::ColdRimingResult combined;
    const bool wet_sll = internal_triple_ && internal_triple_->mode == InternalTripleMode::ColdWetGrowthSllPrimaryOnly;
    const bool collision_sll = wet_sll || (internal_triple_ && internal_triple_->mode == InternalTripleMode::ColdCollisionSllPrimaryOnly);
    const bool rain_sll = collision_sll || (internal_triple_ && internal_triple_->mode == InternalTripleMode::ColdRainSllPrimaryOnly);
    if (rain_sll)
    {
        source.initial.rain = {cell[5], cell[11], reflectivity[0]};
        milbrandt_detail::ColdCollectedSllResult converted;
        const bool accepted = wet_sll
            ? milbrandt_detail::evaluate_exact_triple_cold_wet_growth_sll_cell(parameters, source, dt, converted, error)
            : collision_sll
            ? milbrandt_detail::evaluate_exact_triple_cold_collision_sll_cell(parameters, source, dt, converted, error)
            : milbrandt_detail::evaluate_exact_triple_cold_rain_sll_cell(parameters, source, dt, converted, error);
        if (!accepted)
            return false;
        combined = converted.combined;
    }
    else if (internal_triple_ && internal_triple_->mode == InternalTripleMode::ColdCollectedSllPrimaryOnly)
    {
        milbrandt_detail::ColdCollectedSllResult converted;
        if (!milbrandt_detail::evaluate_exact_triple_cold_collected_sll_cell(parameters, source, dt, converted, error))
        {
            return false;
        }
        combined = converted.combined;
    }
    else if (internal_triple_ && internal_triple_->mode == InternalTripleMode::ColdSllPrimaryOnly)
    {
        milbrandt_detail::ColdSllResult converted;
        if (!milbrandt_detail::evaluate_exact_triple_cold_sll_cell(parameters, source, dt, converted, error))
        {
            return false;
        }
        combined = converted.combined;
    }
    else if (internal_triple_ && (internal_triple_->mode == InternalTripleMode::ColdGraupelConversionsPrimaryOnly ||
        internal_triple_->mode == InternalTripleMode::ColdHomogeneousPrimaryOnly))
    {
        milbrandt_detail::ColdGraupelConversionsResult converted;
        if (!milbrandt_detail::evaluate_exact_triple_cold_graupel_conversions_cell(parameters, source, dt, converted, error,
                2, nullptr, internal_triple_->mode == InternalTripleMode::ColdHomogeneousPrimaryOnly))
        {
            return false;
        }
        combined = converted.combined;
    }
    else if (internal_triple_ && internal_triple_->mode == InternalTripleMode::ColdIceGraupelPrimaryOnly)
    {
        milbrandt_detail::ColdIceGraupelResult converted;
        if (!milbrandt_detail::evaluate_exact_triple_cold_ice_graupel_cell(parameters, source, dt, converted, error))
        {
            return false;
        }
        combined = converted.combined;
    }
    else if (!milbrandt_detail::evaluate_exact_triple_cold_riming_cell(parameters, source, dt, combined, error))
    {
        return false;
    }
    InternalTriplePrimaryCellCandidate candidate;
    for (std::size_t species = 0; species < candidate.water_after.size(); ++species)
    {
        candidate.water_after[species] = cell[species + 3];
    }
    for (std::size_t species = 0; species < candidate.number_after.size(); ++species)
    {
        candidate.number_after[species] = static_cast<float>(cell[species + 11]);
    }
    candidate.reflectivity_after = reflectivity;
    candidate.water_after[0] = combined.unrounded.vapor_mass_per_kg;
    candidate.water_after[1] = combined.unrounded.cloud_mass_per_kg;
    candidate.cloud_number_after = combined.unrounded.cloud_number_per_kg;
    candidate.temperature_increment = combined.temperature_increment_k;
    const auto store_rate = [&](double increment, std::size_t slot)
    {
        const double rate = increment / dt;
        if (!std::isfinite(rate) || std::abs(rate) > std::numeric_limits<float>::max() ||
            (rate != 0.0 && static_cast<float>(rate) == 0.0f))
        {
            error = "cold riming rate is not FP32 representable";
            return false;
        }
        candidate.rates[slot] = static_cast<float>(rate);
        return true;
    };
    if (!store_rate(combined.vapor_mass_increment_per_kg, 1) ||
        !store_rate(combined.cloud_mass_increment_per_kg, 2))
    {
        return false;
    }
    if (rain_sll)
    {
        if (!store_rate(combined.rain_mass_increment_per_kg, 3)) return false;
        candidate.water_after[2] = combined.unrounded.rain.mass_per_kg;
        candidate.number_after[0] = combined.rounded_rain.number_per_kg;
        candidate.reflectivity_after[0] = combined.rounded_rain.reflectivity_per_kg;
    }
    for (std::size_t species = 0; species < combined.rounded.size(); ++species)
    {
        if (!store_rate(combined.mass_increment_per_kg[species], species + 4))
        {
            return false;
        }
        candidate.water_after[species + 3] = combined.unrounded.frozen[species].mass_per_kg;
        candidate.number_after[species + 1] = combined.rounded[species].number_per_kg;
        candidate.reflectivity_after[species + 1] = combined.rounded[species].reflectivity_per_kg;
    }
    result = candidate;
    error.clear();
    return true;
}

} // namespace tmv
