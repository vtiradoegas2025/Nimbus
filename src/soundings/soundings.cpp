#include "soundings/runtime.hpp"
#include "util/log.hpp"

#include "factory.hpp"

SoundingRuntimeState global_sounding_runtime_state;

void initialize_soundings(SoundingRuntimeState& runtime,
                          const SoundingConfig& config)
{
    try
    {
        runtime.config = config;
        runtime.scheme = create_sounding_scheme(config.scheme_id);

        if (runtime.scheme)
        {
            runtime.scheme->initialize(config);
        }

        tmv::log_info("Initialized sounding scheme: ", config.scheme_id);

        if (config.scheme_id != "none" && !config.scheme_id.empty())
        {
            tmv::log_info("  File path: ", config.file_path);
            tmv::log_info("  Interpolation: ",
                          config.interpolation_method == 0 ? "linear" :
                          config.interpolation_method == 1 ? "spline" : "log-linear");
            tmv::log_info("  Fallback profiles: ",
                          config.use_fallback_profiles ? "enabled" : "disabled");
        }

    }
    catch (const std::exception& e)
    {
        runtime = {};
        tmv::log_error("Error initializing soundings: ", e.what());
        throw;
    }
}

SoundingData load_sounding_data(SoundingRuntimeState& runtime)
{
    if (!runtime.scheme)
    {
        tmv::log_warn("Sounding scheme not initialized");
        return SoundingData{};
    }

    try
    {
        return runtime.scheme->load_sounding(runtime.config.file_path);
    }
    catch (const std::exception& e)
    {
        tmv::log_error("Error loading sounding data: ", e.what());
        if (!runtime.config.use_fallback_profiles)
        {
            throw;
        }
        tmv::log_warn("Using fallback profiles as configured");
        return SoundingData{};
    }
}

SoundingData interpolate_sounding_to_grid(
    SoundingRuntimeState& runtime,
    const SoundingData& sounding,
    const std::vector<double>& model_heights)
{
    if (!runtime.scheme)
    {
        throw std::runtime_error("Sounding scheme not initialized");
    }

    try
    {
        return runtime.scheme->interpolate_to_heights(sounding, model_heights);
    }
    catch (const std::exception& e)
    {
        tmv::log_error("Error interpolating sounding: ", e.what());
        throw;
    }
}

const SoundingConfig& get_sounding_config(const SoundingRuntimeState& runtime)
{
    return runtime.config;
}

bool is_soundings_initialized(const SoundingRuntimeState& runtime)
{
    return runtime.scheme != nullptr && runtime.scheme->is_initialized();
}

void reset_soundings(SoundingRuntimeState& runtime)
{
    runtime = {};
}

void initialize_soundings(const SoundingConfig& config)
{
    initialize_soundings(global_sounding_runtime_state, config);
}

SoundingData load_sounding_data()
{
    return load_sounding_data(global_sounding_runtime_state);
}

SoundingData interpolate_sounding_to_grid(
    const SoundingData& sounding,
    const std::vector<double>& model_heights)
{
    return interpolate_sounding_to_grid(
        global_sounding_runtime_state, sounding, model_heights);
}

const SoundingConfig& get_sounding_config()
{
    return get_sounding_config(global_sounding_runtime_state);
}

bool is_soundings_initialized()
{
    return is_soundings_initialized(global_sounding_runtime_state);
}

void reset_soundings()
{
    reset_soundings(global_sounding_runtime_state);
}
