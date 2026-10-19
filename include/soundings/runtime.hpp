#pragma once

#include "soundings/soundings_base.hpp"

#include <memory>

struct SoundingRuntimeState
{
    std::unique_ptr<SoundingScheme> scheme;
    SoundingConfig config;
};

extern SoundingRuntimeState global_sounding_runtime_state;

void initialize_soundings(SoundingRuntimeState& runtime,
                          const SoundingConfig& config);
SoundingData load_sounding_data(SoundingRuntimeState& runtime);
SoundingData interpolate_sounding_to_grid(
    SoundingRuntimeState& runtime,
    const SoundingData& sounding,
    const std::vector<double>& model_heights);
const SoundingConfig& get_sounding_config(const SoundingRuntimeState& runtime);
bool is_soundings_initialized(const SoundingRuntimeState& runtime);
void reset_soundings(SoundingRuntimeState& runtime);

// Process-global compatibility adapters.
void initialize_soundings(const SoundingConfig& config);
SoundingData load_sounding_data();
SoundingData interpolate_sounding_to_grid(
    const SoundingData& sounding,
    const std::vector<double>& model_heights);
const SoundingConfig& get_sounding_config();
bool is_soundings_initialized();
void reset_soundings();
