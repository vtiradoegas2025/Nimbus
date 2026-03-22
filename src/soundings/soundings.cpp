/**
 * @file soundings.cpp
 * @brief Implementation for the soundings module.
 *
 * Provides executable logic for the soundings runtime path,
 * including initialization, stepping, and diagnostics helpers.
 * This file is part of the src/soundings subsystem.
 */

#include "data/soundings_base.hpp"
#include "factory.hpp"
#include <iostream>
#include <memory>


std::unique_ptr<SoundingScheme> sounding_scheme = nullptr;

SoundingConfig global_sounding_config;

/**
 * @brief Initialize the atmospheric sounding scheme
 * @param config Sounding configuration from YAML or defaults
 */
void initialize_soundings(const SoundingConfig& config) 
{
    try 
    {
        global_sounding_config = config;
        sounding_scheme = create_sounding_scheme(config.scheme_id);

        if (sounding_scheme) 
        {
            sounding_scheme->initialize(config);
        }

        std::cout << "Initialized sounding scheme: " << config.scheme_id << std::endl;

        if (config.scheme_id != "none" && !config.scheme_id.empty()) {
            std::cout << "  File path: " << config.file_path << std::endl;
            std::cout << "  Interpolation: " <<
                (config.interpolation_method == 0 ? "linear" :
                 config.interpolation_method == 1 ? "spline" : "log-linear") << std::endl;
            std::cout << "  Fallback profiles: " << (config.use_fallback_profiles ? "enabled" : "disabled") << std::endl;
        }

    } catch (const std::exception& e) {
        std::cerr << "Error initializing soundings: " << e.what() << std::endl;
        throw;
    }
}

/**
 * @brief Load sounding data for model initialization
 * @return SoundingData structure, or empty if failed and fallback disabled
 */
SoundingData load_sounding_data() {
    if (!sounding_scheme) {
        std::cerr << "Warning: Sounding scheme not initialized" << std::endl;
        return SoundingData{};
    }

    try {
        return sounding_scheme->load_sounding(global_sounding_config.file_path);
    } catch (const std::exception& e) {
        std::cerr << "Error loading sounding data: " << e.what() << std::endl;
        if (!global_sounding_config.use_fallback_profiles) {
            throw;
        }
        std::cerr << "Using fallback profiles as configured" << std::endl;
        return SoundingData{};
    }
}

/**
 * @brief Interpolate sounding data to model grid heights
 * @param sounding Original sounding data
 * @param model_heights Vector of model grid heights (m)
 * @return Interpolated SoundingData at model levels
 */
SoundingData interpolate_sounding_to_grid(
    const SoundingData& sounding,
    const std::vector<double>& model_heights) {

    if (!sounding_scheme) {
        throw std::runtime_error("Sounding scheme not initialized");
    }

    try {
        return sounding_scheme->interpolate_to_heights(sounding, model_heights);
    } catch (const std::exception& e) {
        std::cerr << "Error interpolating sounding: " << e.what() << std::endl;
        throw;
    }
}

/**
 * @brief Get current sounding configuration
 * @return Reference to current configuration
 */
const SoundingConfig& get_sounding_config() {
    return global_sounding_config;
}

/**
 * @brief Check if sounding scheme is initialized
 * @return true if ready to use
 */
bool is_soundings_initialized() {
    return sounding_scheme != nullptr && sounding_scheme->is_initialized();
}

/**
 * @brief Reset sounding system (for cleanup or reinitialization)
 */
void reset_soundings() {
    sounding_scheme.reset();
    global_sounding_config = SoundingConfig{};
}
