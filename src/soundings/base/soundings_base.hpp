/**
 * @file soundings_base.hpp
 * @brief Declarations for the soundings module.
 *
 * Defines interfaces, data structures, and contracts used by
 * the soundings runtime and scheme implementations.
 * This file is part of the src/soundings subsystem.
 */

#pragma once
#include <data/soundings_base.hpp>

/**
 * @brief Quality control for sounding data
 * @param data Sounding data to validate and filter
 * @param config Configuration with QC thresholds
 * @return true if data passes QC and was filtered
 */
bool quality_control_sounding(SoundingData& data, const SoundingConfig& config);

/**
 * @brief Calculate derived thermodynamic quantities from basic sounding data
 * @param data Sounding data with temperature, dewpoint, and pressure
 */
void calculate_derived_quantities(SoundingData& data);

/**
 * @brief Interpolate sounding data to target heights using linear interpolation
 * @param source Source sounding data
 * @param target_heights Target heights in meters
 * @param config Sounding configuration
 * @return Interpolated sounding data
 */
SoundingData interpolate_sounding_linear(
    const SoundingData& source,
    const std::vector<double>& target_heights,
    const SoundingConfig& config);
