#pragma once

#include "soundings/base/sounding_algorithms.hpp"

/**
 * Direct reader for CM1's native input_sounding text contract:
 *
 *   p_sfc_hPa theta_sfc_K qv_sfc_gkg
 *   z_m       theta_K     qv_gkg       u_ms v_ms
 *   ...
 *
 * Potential temperature, water-vapor mixing ratio, and Cartesian wind
 * components remain primary quantities. Pressure and temperature are
 * reconstructed hydrostatically from the surface header.
 */
class Cm1SoundingScheme final : public SoundingScheme
{
public:
    void initialize(const SoundingConfig& config) override;
    SoundingData load_sounding(const std::string& file_path) override;
    SoundingData interpolate_to_heights(
        const SoundingData& sounding,
        const std::vector<double>& target_heights_m) override;

    const SoundingConfig& get_config() const override { return config_; }
    bool is_initialized() const override { return initialized_; }

private:
    SoundingConfig config_;
    bool initialized_ = false;
};
