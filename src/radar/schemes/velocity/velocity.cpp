/**
 * @file velocity.cpp
 * @brief Implementation for the radar module.
 *
 * Provides executable logic for the radar runtime path,
 * including initialization, stepping, and diagnostics helpers.
 * This file is part of the src/radar subsystem.
 */

#include "velocity.hpp"
#include <iostream>
#include <algorithm>
#include <cmath>

namespace
{

constexpr float kRadarVrAbsMax = 250.0f;

void ensure_velocity_output(RadarOut& out, int nr, int nth, int nz)
{
    out.NR = nr;
    out.NTH = nth;
    out.NZ = nz;

    if (out.Vr.size_r() != nr || out.Vr.size_th() != nth || out.Vr.size_z() != nz)
    {
        out.Vr.resize(nr, nth, nz, 0.0f);
    }
    else
    {
        out.Vr.fill(0.0f);
    }
}

bool has_matching_dimensions(const RadarStateView& state, int nr, int nth, int nz)
{
    return state.NR == nr && state.NTH == nth && state.NZ == nz;
}

int wrap_theta_index(int j, int nth)
{
    if (nth <= 0)
    {
        return 0;
    }
    int wrapped = j % nth;
    if (wrapped < 0)
    {
        wrapped += nth;
    }
    return wrapped;
}

float sample_component(const Field3D* field,
                       int i,
                       int j,
                       int k,
                       int nr,
                       int nth,
                       int nz,
                       bool beam_volume)
{
    if (!field || nr <= 0 || nth <= 0 || nz <= 0)
    {
        return 0.0f;
    }

    const int ic = std::max(0, std::min(i, nr - 1));
    const int jc = wrap_theta_index(j, nth);
    const int kc = std::max(0, std::min(k, nz - 1));

    if (!beam_volume)
    {
        const float value = (*field)[ic][jc][kc];
        return std::isfinite(static_cast<double>(value)) ? value : 0.0f;
    }

    double sum = 0.0;
    int count = 0;
    for (int di = -1; di <= 1; ++di)
    {
        const int ii = std::max(0, std::min(ic + di, nr - 1));
        for (int dj = -1; dj <= 1; ++dj)
        {
            const int jj = wrap_theta_index(jc + dj, nth);
            for (int dk = -1; dk <= 1; ++dk)
            {
                const int kk = std::max(0, std::min(kc + dk, nz - 1));
                const float value = (*field)[ii][jj][kk];
                if (std::isfinite(static_cast<double>(value)))
                {
                    sum += static_cast<double>(value);
                    ++count;
                }
            }
        }
    }
    return (count > 0) ? static_cast<float>(sum / static_cast<double>(count)) : 0.0f;
}

}

/**
 * @brief Initializes the velocity scheme.
 */
void VelocityScheme::initialize(const RadarConfig& config, int NR, int NTH, int NZ)
{
    NR_ = NR;
    NTH_ = NTH;
    NZ_ = NZ;

    radar_x_ = config.radar_x;
    radar_y_ = config.radar_y;
    radar_z_ = config.radar_z;

    std::cout << "Initialized velocity radar scheme (" << NR << "x" << NTH << "x" << NZ << ")" << std::endl;
    std::cout << "Radar location: (" << radar_x_ << ", " << radar_y_ << ", " << radar_z_ << ")" << std::endl;
}

/**
 * @brief Computes the velocity scheme.
 */
void VelocityScheme::compute(const RadarConfig& config, const RadarStateView& state, RadarOut& out) 
{
    ensure_velocity_output(out, NR_, NTH_, NZ_);

    if (!has_matching_dimensions(state, NR_, NTH_, NZ_))
    {
        static bool warned_dimension_mismatch = false;
        if (!warned_dimension_mismatch)
        {
            std::cerr << "Warning: Velocity scheme state/output dimensions mismatch. "
                      << "Returning zeroed radial velocity output." << std::endl;
            warned_dimension_mismatch = true;
        }
        return;
    }

    compute_radial_velocity(config, state, out);

    if (!config.apply_scatterer_correction)
    {
        return;
    }

    const bool has_hydrometeors =
        (state.qr != nullptr) || (state.qs != nullptr) || (state.qg != nullptr) ||
        (state.qh != nullptr) || (state.qi != nullptr);
    if (!has_hydrometeors)
    {
        static bool warned_missing_hydrometeors = false;
        if (!warned_missing_hydrometeors)
        {
            std::cerr << "Warning: Radar scatterer correction requested, but no hydrometeor fields are available. "
                      << "Skipping correction." << std::endl;
            warned_missing_hydrometeors = true;
        }
        return;
    }

    apply_scatterer_correction(config, state, out);
}

/**
 * @brief Computes the radial velocity.
 */
void VelocityScheme::compute_radial_velocity(const RadarConfig& config, const RadarStateView& state, RadarOut& out) 
{
    if (!state.u || !state.v || !state.w) 
    {
        std::cerr << "Warning: Velocity scheme requires wind fields (u, v, w)" << std::endl;
        return;
    }

    extern double dr, dz, dtheta;

    const bool beam_volume = (config.sampling == "beam_volume");
    if (!beam_volume && config.sampling != "point")
    {
        static bool warned_unknown_sampling = false;
        if (!warned_unknown_sampling)
        {
            std::cerr << "Warning: Unknown radar sampling mode '" << config.sampling
                      << "'. Falling back to point sampling." << std::endl;
            warned_unknown_sampling = true;
        }
    }

    for (int i = 0; i < NR_; ++i) 
    {
        for (int j = 0; j < NTH_; ++j) 
        {
            for (int k = 0; k < NZ_; ++k) 
            {
                const float u_cyl = sample_component(state.u, i, j, k, NR_, NTH_, NZ_, beam_volume);
                const float v = sample_component(state.v, i, j, k, NR_, NTH_, NZ_, beam_volume);
                const float w = sample_component(state.w, i, j, k, NR_, NTH_, NZ_, beam_volume);

                double r = static_cast<double>(i) * dr;
                double theta = static_cast<double>(j) * dtheta;
                double z = static_cast<double>(k) * dz;

                double x = r * cos(theta);
                double y = r * sin(theta);

                double u_cart = u_cyl * cos(theta) - v * sin(theta);
                double v_cart = u_cyl * sin(theta) + v * cos(theta);

                double e_r_x, e_r_y, e_r_z, R;
                RadarGeometry::compute_line_of_sight(radar_x_, radar_y_, radar_z_,
                                                   x, y, z, e_r_x, e_r_y, e_r_z, R);

                float Vr = static_cast<float>(u_cart * e_r_x + v_cart * e_r_y + w * e_r_z);
                if (!std::isfinite(static_cast<double>(Vr)))
                {
                    Vr = 0.0f;
                }
                Vr = std::clamp(Vr, -kRadarVrAbsMax, kRadarVrAbsMax);

                out.Vr[i][j][k] = Vr;
            }
        }
    }
}

/**
 * @brief Applies the scatterer correction.
 */
void VelocityScheme::apply_scatterer_correction(const RadarConfig& config, const RadarStateView& state, RadarOut& out) 
{

    extern double dr, dz, dtheta;
    const bool beam_volume = (config.sampling == "beam_volume");

    for (int i = 0; i < NR_; ++i) 
    {
        for (int j = 0; j < NTH_; ++j) 
        {
            for (int k = 0; k < NZ_; ++k) 
            {
                const float q_rain = sample_component(state.qr, i, j, k, NR_, NTH_, NZ_, beam_volume);
                const float q_snow = sample_component(state.qs, i, j, k, NR_, NTH_, NZ_, beam_volume);
                const float q_graupel = sample_component(state.qg, i, j, k, NR_, NTH_, NZ_, beam_volume);
                const float q_hail = sample_component(state.qh, i, j, k, NR_, NTH_, NZ_, beam_volume);
                const float q_ice = sample_component(state.qi, i, j, k, NR_, NTH_, NZ_, beam_volume);

                float v_terminal = estimate_fall_speed(q_rain, q_snow, q_graupel, q_hail, q_ice);

                double r = static_cast<double>(i) * dr;
                double theta = static_cast<double>(j) * dtheta;
                double z = static_cast<double>(k) * dz;

                double x = r * cos(theta);
                double y = r * sin(theta);

                double e_r_x, e_r_y, e_r_z, R;
                RadarGeometry::compute_line_of_sight(radar_x_, radar_y_, radar_z_,
                                                   x, y, z, e_r_x, e_r_y, e_r_z, R);

                float correction = -v_terminal * static_cast<float>(e_r_z);
                if (!std::isfinite(static_cast<double>(correction)))
                {
                    correction = 0.0f;
                }
                float vr_corrected = out.Vr[i][j][k] + correction;
                if (!std::isfinite(static_cast<double>(vr_corrected)))
                {
                    vr_corrected = 0.0f;
                }
                out.Vr[i][j][k] = std::clamp(vr_corrected, -kRadarVrAbsMax, kRadarVrAbsMax);
            }
        }
    }
}

/**
 * @brief Estimates the fall speed.
 */
float VelocityScheme::estimate_fall_speed(float q_rain, float q_snow, float q_graupel, float q_hail, float q_ice) 
{
    const float qr = std::max(0.0f, q_rain);
    const float qs = std::max(0.0f, q_snow);
    const float qg = std::max(0.0f, q_graupel);
    const float qh = std::max(0.0f, q_hail);
    const float qi = std::max(0.0f, q_ice);
    const float q_total = qr + qs + qg + qh + qi;
    if (q_total <= 0.0f)
    {
        return 0.0f;
    }

    auto bulk_speed = [](float q, float a, float b) -> float
    {
        if (q <= 0.0f)
        {
            return 0.0f;
        }
        return a * std::pow(q * 1000.0f, b);
    };

    const float v_rain = bulk_speed(qr, 9.0f, 0.20f);
    const float v_snow = bulk_speed(qs, 1.5f, 0.16f);
    const float v_graupel = bulk_speed(qg, 11.0f, 0.18f);
    const float v_hail = bulk_speed(qh, 15.0f, 0.16f);
    const float v_ice = bulk_speed(qi, 0.8f, 0.14f);

    const float v_weighted =
        (qr * v_rain + qs * v_snow + qg * v_graupel + qh * v_hail + qi * v_ice) / q_total;
    if (!std::isfinite(static_cast<double>(v_weighted)))
    {
        return 0.0f;
    }

    return std::clamp(v_weighted, 0.0f, 40.0f);
}
