/**
 * @file radar.cpp
 * @brief Core runtime implementation for the tornado model.
 *
 * Provides simulation orchestration and subsystem integration
 * for dynamics, numerics, physics, and runtime execution paths.
 * This file belongs to the primary src/core execution layer.
 */

#include "radar/radar.hpp"
#include "radar/factory.hpp"
#include <algorithm>
#include <cmath>

namespace
{

constexpr float kRadarLinearMin = 0.0f;
constexpr float kRadarLinearMax = 1.0e12f;
constexpr float kRadarDbzMin = -40.0f;
constexpr float kRadarDbzMax = 100.0f;
constexpr float kRadarVrAbsMax = 250.0f;
constexpr float kRadarZdrMin = -12.0f;
constexpr float kRadarZdrMax = 12.0f;

void ensure_output_initialized(const RadarStateView& state, RadarOut& output)
{
    if (state.NR <= 0 || state.NTH <= 0 || state.NZ <= 0)
    {
        return;
    }

    if (output.NR != state.NR || output.NTH != state.NTH || output.NZ != state.NZ)
    {
        output.initialize(state.NR, state.NTH, state.NZ);
    }
}

void sanitize_radar_field(Field3D& field, float min_value, float max_value)
{
    if (field.empty())
    {
        return;
    }

    float* const data = field.data();
    const std::size_t count = field.size();
    for (std::size_t idx = 0; idx < count; ++idx)
    {
        float value = data[idx];
        if (!std::isfinite(static_cast<double>(value)))
        {
            value = min_value;
        }
        value = std::clamp(value, min_value, max_value);
        data[idx] = value;
    }
}

void sanitize_radar_output(RadarOut& output)
{
    sanitize_radar_field(output.Ze_linear, kRadarLinearMin, kRadarLinearMax);
    sanitize_radar_field(output.Ze_rain, kRadarLinearMin, kRadarLinearMax);
    sanitize_radar_field(output.Ze_snow, kRadarLinearMin, kRadarLinearMax);
    sanitize_radar_field(output.Ze_graupel, kRadarLinearMin, kRadarLinearMax);
    sanitize_radar_field(output.Ze_hail, kRadarLinearMin, kRadarLinearMax);
    sanitize_radar_field(output.Ze_ice, kRadarLinearMin, kRadarLinearMax);

    sanitize_radar_field(output.Z_dBZ, kRadarDbzMin, kRadarDbzMax);
    sanitize_radar_field(output.ZH_dBZ, kRadarDbzMin, kRadarDbzMax);
    sanitize_radar_field(output.ZV_dBZ, kRadarDbzMin, kRadarDbzMax);
    sanitize_radar_field(output.ZDR_dB, kRadarZdrMin, kRadarZdrMax);
    sanitize_radar_field(output.Vr, -kRadarVrAbsMax, kRadarVrAbsMax);
}

}



/**
 * @brief Computes all the radar observables.
 */
void RadarSystem::compute_all_observables(const RadarStateView& state, RadarOut& output,
                                         double radar_x, double radar_y, double radar_z) {
    ensure_output_initialized(state, output);

    auto reflectivity = create_radar_scheme(RadarSchemes::REFLECTIVITY);
    auto velocity = create_radar_scheme(RadarSchemes::VELOCITY);
    auto zdr = create_radar_scheme(RadarSchemes::ZDR);

    RadarConfig refl_config;
    refl_config.scheme_id = RadarSchemes::REFLECTIVITY;
    refl_config.operator_tier = RadarTiers::Reflectivity::FAST_DA;
    refl_config.has_qr = (state.qr != nullptr);
    refl_config.has_qs = (state.qs != nullptr);
    refl_config.has_qg = (state.qg != nullptr);
    refl_config.has_qh = (state.qh != nullptr);
    refl_config.has_qi = (state.qi != nullptr);

    RadarConfig vel_config;
    vel_config.scheme_id = RadarSchemes::VELOCITY;
    vel_config.radar_x = radar_x;
    vel_config.radar_y = radar_y;
    vel_config.radar_z = radar_z;
    vel_config.apply_scatterer_correction =
        (state.qr != nullptr) || (state.qs != nullptr) || (state.qg != nullptr) ||
        (state.qh != nullptr) || (state.qi != nullptr);
    vel_config.has_qr = (state.qr != nullptr);
    vel_config.has_qs = (state.qs != nullptr);
    vel_config.has_qg = (state.qg != nullptr);
    vel_config.has_qh = (state.qh != nullptr);
    vel_config.has_qi = (state.qi != nullptr);

    RadarConfig zdr_config;
    zdr_config.scheme_id = RadarSchemes::ZDR;
    zdr_config.operator_tier = RadarTiers::ZDR::POLARIMETRIC_FO;
    zdr_config.has_qr = (state.qr != nullptr);
    zdr_config.has_qs = (state.qs != nullptr);
    zdr_config.has_qg = (state.qg != nullptr);
    zdr_config.has_qh = (state.qh != nullptr);
    zdr_config.has_qi = (state.qi != nullptr);

    reflectivity->initialize(refl_config, state.NR, state.NTH, state.NZ);
    velocity->initialize(vel_config, state.NR, state.NTH, state.NZ);
    zdr->initialize(zdr_config, state.NR, state.NTH, state.NZ);

    reflectivity->compute(refl_config, state, output);
    velocity->compute(vel_config, state, output);
    zdr->compute(zdr_config, state, output);
    sanitize_radar_output(output);
}

/**
 * @brief Computes the radar reflectivity.
 */
void RadarSystem::compute_reflectivity(const RadarStateView& state, RadarOut& output) 
{
    ensure_output_initialized(state, output);

    auto reflectivity = create_radar_scheme(RadarSchemes::REFLECTIVITY);

    RadarConfig config;
    config.scheme_id = RadarSchemes::REFLECTIVITY;
    config.operator_tier = RadarTiers::Reflectivity::FAST_DA;
    config.has_qr = (state.qr != nullptr);
    config.has_qs = (state.qs != nullptr);
    config.has_qg = (state.qg != nullptr);
    config.has_qh = (state.qh != nullptr);
    config.has_qi = (state.qi != nullptr);

    reflectivity->initialize(config, state.NR, state.NTH, state.NZ);
    reflectivity->compute(config, state, output);
    sanitize_radar_output(output);
}

/**
 * @brief Computes the radar velocity.
 */
void RadarSystem::compute_velocity(const RadarStateView& state, RadarOut& output,
                                  double radar_x, double radar_y, double radar_z) 
                                  {
    ensure_output_initialized(state, output);

    auto velocity = create_radar_scheme(RadarSchemes::VELOCITY);

    RadarConfig config;
    config.scheme_id = RadarSchemes::VELOCITY;
    config.radar_x = radar_x;
    config.radar_y = radar_y;
    config.radar_z = radar_z;
    config.apply_scatterer_correction =
        (state.qr != nullptr) || (state.qs != nullptr) || (state.qg != nullptr) ||
        (state.qh != nullptr) || (state.qi != nullptr);
    config.has_qr = (state.qr != nullptr);
    config.has_qs = (state.qs != nullptr);
    config.has_qg = (state.qg != nullptr);
    config.has_qh = (state.qh != nullptr);
    config.has_qi = (state.qi != nullptr);

    velocity->initialize(config, state.NR, state.NTH, state.NZ);
    velocity->compute(config, state, output);
    sanitize_radar_output(output);
}

/**
 * @brief Computes the radar ZDR.
 */
void RadarSystem::compute_zdr(const RadarStateView& state, RadarOut& output) 
{
    ensure_output_initialized(state, output);

    auto zdr = create_radar_scheme(RadarSchemes::ZDR);

    RadarConfig config;
    config.scheme_id = RadarSchemes::ZDR;
    config.operator_tier = RadarTiers::ZDR::POLARIMETRIC_FO;
    config.has_qr = (state.qr != nullptr);
    config.has_qs = (state.qs != nullptr);
    config.has_qg = (state.qg != nullptr);
    config.has_qh = (state.qh != nullptr);
    config.has_qi = (state.qi != nullptr);

    zdr->initialize(config, state.NR, state.NTH, state.NZ);
    zdr->compute(config, state, output);
    sanitize_radar_output(output);
}
