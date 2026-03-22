#pragma once

#include "data/soundings_base.hpp"

/**
 * @file soundings.hpp
 * @brief Public API for sounding initialization and interpolation.
 *
 * Provides convenience wrappers around the configured sounding scheme.
 * These functions are used during model initialization to load and
 * remap atmospheric profiles to model vertical levels.
 */

/**
 * @brief Initializes the sounding subsystem.
 * @param config Runtime sounding configuration.
 */
void initialize_soundings(const SoundingConfig& config);

/**
 * @brief Loads raw sounding data from the configured source.
 * @return Loaded sounding profile and metadata.
 */
SoundingData load_sounding_data();

/**
 * @brief Interpolates a sounding profile onto model heights.
 * @param sounding Input sounding profile.
 * @param model_heights Target model heights in meters.
 * @return Interpolated sounding profile.
 */
SoundingData interpolate_sounding_to_grid(const SoundingData& sounding,
                                          const std::vector<double>& model_heights);

/**
 * @brief Returns the active sounding configuration.
 * @return Immutable reference to current sounding config.
 */
const SoundingConfig& get_sounding_config();

/**
 * @brief Reports whether the sounding subsystem has been initialized.
 * @return True when initialization has completed successfully.
 */
bool is_soundings_initialized();

/**
 * @brief Resets sounding subsystem state.
 */
void reset_soundings();
