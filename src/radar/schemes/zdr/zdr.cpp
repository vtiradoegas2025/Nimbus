/**
 * @file zdr.cpp
 * @brief Implementation for the radar module.
 *
 * Provides executable logic for the radar runtime path,
 * including initialization, stepping, and diagnostics helpers.
 * This file is part of the src/radar subsystem.
 */

#include "zdr.hpp"
#include "core/field3d.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace
{

constexpr float kRadarDbzMin = -40.0f;
constexpr float kRadarDbzMax = 100.0f;
constexpr float kRadarZdrMin = -12.0f;
constexpr float kRadarZdrMax = 12.0f;
constexpr float kRadarLinearMin = 1.0e-10f;
constexpr float kRadarLinearMax = 1.0e12f;

void ensure_field_shape(Field3D& field, int nr, int nth, int nz, float init_value)
{
    if (field.size_r() != nr || field.size_th() != nth || field.size_z() != nz)
    {
        field.resize(nr, nth, nz, init_value);
        return;
    }
    field.fill(init_value);
}

void prepare_zdr_output(RadarOut& out, int nr, int nth, int nz)
{
    out.NR = nr;
    out.NTH = nth;
    out.NZ = nz;

    ensure_field_shape(out.ZH_dBZ, nr, nth, nz, kRadarDbzMin);
    ensure_field_shape(out.ZV_dBZ, nr, nth, nz, kRadarDbzMin);
    ensure_field_shape(out.ZDR_dB, nr, nth, nz, 0.0f);
    ensure_field_shape(out.Z_dBZ, nr, nth, nz, kRadarDbzMin);
    ensure_field_shape(out.Ze_linear, nr, nth, nz, 0.0f);
}

bool has_matching_dimensions(const RadarStateView& state, int nr, int nth, int nz)
{
    return state.NR == nr && state.NTH == nth && state.NZ == nz;
}

inline float clamp_dbz(float value)
{
    if (!std::isfinite(static_cast<double>(value)))
    {
        return kRadarDbzMin;
    }
    return std::clamp(value, kRadarDbzMin, kRadarDbzMax);
}

inline float clamp_zdr(float value)
{
    if (!std::isfinite(static_cast<double>(value)))
    {
        return 0.0f;
    }
    return std::clamp(value, kRadarZdrMin, kRadarZdrMax);
}

inline float dbz_from_linear(float linear_value)
{
    if (!std::isfinite(static_cast<double>(linear_value)) || linear_value <= kRadarLinearMin)
    {
        return kRadarDbzMin;
    }
    const float clamped_linear = std::clamp(linear_value, kRadarLinearMin, kRadarLinearMax);
    const float dbz = 10.0f * std::log10(clamped_linear);
    return clamp_dbz(dbz);
}

}

/**
 * @brief Initializes the ZDR scheme.
 */
void ZDRScheme::initialize(const RadarConfig& config, int NR, int NTH, int NZ) 
{
    NR_ = NR;
    NTH_ = NTH;
    NZ_ = NZ;

    std::cout << "Initialized ZDR radar scheme (" << NR << "x" << NTH << "x" << NZ << ")" << std::endl;
    std::cout << "Operator tier: " << config.operator_tier << std::endl;
}

/**
 * @brief Computes the ZDR scheme.
 */
void ZDRScheme::compute(const RadarConfig& config, const RadarStateView& state, RadarOut& out) 
{
    prepare_zdr_output(out, NR_, NTH_, NZ_);

    if (!has_matching_dimensions(state, NR_, NTH_, NZ_))
    {
        static bool warned_dimension_mismatch = false;
        if (!warned_dimension_mismatch)
        {
            std::cerr << "Warning: ZDR scheme state/output dimensions mismatch. "
                      << "Returning zeroed polarimetric output." << std::endl;
            warned_dimension_mismatch = true;
        }
        return;
    }

    if (config.operator_tier == "simple") 
    {
        compute_simple_zdr(config, state, out);
    } 

    else if (config.operator_tier == "polarimetric_fo") 
    {
        compute_polarimetric_fo(config, state, out);
    }
    else 
    {
        throw std::runtime_error("Unknown ZDR operator tier: " + config.operator_tier);
    }
}

/**
 * @brief Computes the simple ZDR.
 */
void ZDRScheme::compute_simple_zdr(const RadarConfig& config, const RadarStateView& state, RadarOut& out) 
{

    for (int i = 0; i < NR_; ++i) 
    {
        for (int j = 0; j < NTH_; ++j) 
        {
            for (int k = 0; k < NZ_; ++k) 
            {
                float q_rain = state.qr ? (*state.qr)[i][j][k] : 0.0f;
                if (!std::isfinite(static_cast<double>(q_rain)))
                {
                    q_rain = 0.0f;
                }

                if (q_rain <= 0.0f) {
                    out.ZDR_dB[i][j][k] = 0.0f;
                    out.ZH_dBZ[i][j][k] = kRadarDbzMin;
                    out.ZV_dBZ[i][j][k] = kRadarDbzMin;
                    out.Ze_linear[i][j][k] = 0.0f;
                    out.Z_dBZ[i][j][k] = kRadarDbzMin;
                    continue;
                }

                float axis_ratio = estimate_rain_axis_ratio(q_rain);
                axis_ratio = std::clamp(axis_ratio, 0.4f, 1.0f);

                float Z_H, Z_V;
                compute_rain_polarization(q_rain, axis_ratio, Z_H, Z_V);

                float Z_H_dBZ = dbz_from_linear(Z_H);
                float Z_V_dBZ = dbz_from_linear(Z_V);


                if (Z_V > 1e-10 && Z_H > 1e-10) 
                {
                    float Z_DR_linear = Z_H / Z_V;
                    if (std::isfinite(static_cast<double>(Z_DR_linear)) && Z_DR_linear > 1.0e-10f)
                    {
                        out.ZDR_dB[i][j][k] = clamp_zdr(10.0f * std::log10(Z_DR_linear));
                    }
                    else
                    {
                        out.ZDR_dB[i][j][k] = 0.0f;
                    }
                } 
                else 
                {
                    out.ZDR_dB[i][j][k] = 0.0f;
                }

                out.ZH_dBZ[i][j][k] = clamp_dbz(Z_H_dBZ);
                out.ZV_dBZ[i][j][k] = clamp_dbz(Z_V_dBZ);
                const float z_total_linear = 0.5f * (std::max(0.0f, Z_H) + std::max(0.0f, Z_V));
                out.Ze_linear[i][j][k] = std::clamp(z_total_linear, 0.0f, kRadarLinearMax);
                out.Z_dBZ[i][j][k] = dbz_from_linear(out.Ze_linear[i][j][k]);
            }
        }
    }
}

/**
 * @brief Computes the polarimetric forward operator.
 */
void ZDRScheme::compute_polarimetric_fo(const RadarConfig& config, const RadarStateView& state, RadarOut& out) 
{

    for (int i = 0; i < NR_; ++i) 
    {
        for (int j = 0; j < NTH_; ++j) 
        {
            for (int k = 0; k < NZ_; ++k) 
            {
                out.ZH_dBZ[i][j][k] = kRadarDbzMin;
                out.ZV_dBZ[i][j][k] = kRadarDbzMin;
                out.ZDR_dB[i][j][k] = 0.0f;
            }
        }
    }

    
    if (state.qr && config.has_qr) 
    {
        compute_polarimetric_rain(*state.qr, out);
    }

    if (state.qs && config.has_qs) 
    {
        compute_polarimetric_ice(*state.qs, "snow", out);
    }

    if (state.qg && config.has_qg) 
    {
        compute_polarimetric_ice(*state.qg, "graupel", out);
    }

    if (state.qh && config.has_qh) 
    {
        compute_polarimetric_ice(*state.qh, "hail", out);
    }

    if (state.qi && config.has_qi) 
    {
        compute_polarimetric_ice(*state.qi, "ice", out);
    }

    for (int i = 0; i < NR_; ++i) 
    {
        for (int j = 0; j < NTH_; ++j) 
        {
            for (int k = 0; k < NZ_; ++k) 
            {
                float Z_H_linear = dbz_to_linear(out.ZH_dBZ[i][j][k]);
                float Z_V_linear = dbz_to_linear(out.ZV_dBZ[i][j][k]);

                float Z_total_linear = (Z_H_linear + Z_V_linear) / 2.0f;
                out.Ze_linear[i][j][k] = std::clamp(Z_total_linear, 0.0f, kRadarLinearMax);
                out.Z_dBZ[i][j][k] = dbz_from_linear(Z_total_linear);


                if (Z_V_linear > 1e-10 && Z_H_linear > 1e-10) 
                {
                    const float zdr = 10.0f * std::log10(Z_H_linear / Z_V_linear);
                    out.ZDR_dB[i][j][k] = clamp_zdr(zdr);
                } 
                else 
                {
                    out.ZDR_dB[i][j][k] = 0.0f;
                }
            }
        }
    }
}


/**
 * @brief Computes the rain polarization.
 */
void ZDRScheme::compute_rain_polarization(float q_rain, float axis_ratio, float& Z_H, float& Z_V) 
{

    const float q_safe = (std::isfinite(static_cast<double>(q_rain)) && q_rain > 0.0f) ? q_rain : 0.0f;
    const float axis_safe = std::clamp(axis_ratio, 0.4f, 1.0f);
    double rho_ratio = 1000.0 / 1000.0;
    double Ze_base = 3.63e9 * rho_ratio * rho_ratio * q_safe * q_safe;
    if (!std::isfinite(Ze_base))
    {
        Ze_base = 0.0;
    }
    Ze_base = std::clamp(Ze_base, static_cast<double>(kRadarLinearMin), static_cast<double>(kRadarLinearMax));

    double f_h = 1.0 / (axis_safe * axis_safe);
    double f_v = axis_safe * axis_safe;

    const double zh_linear = Ze_base * f_h;
    const double zv_linear = Ze_base * f_v;
    Z_H = static_cast<float>(std::clamp(std::isfinite(zh_linear) ? zh_linear : 0.0, 0.0, static_cast<double>(kRadarLinearMax)));
    Z_V = static_cast<float>(std::clamp(std::isfinite(zv_linear) ? zv_linear : 0.0, 0.0, static_cast<double>(kRadarLinearMax)));
}

/**
 * @brief Estimates the rain axis ratio.
 */
float ZDRScheme::estimate_rain_axis_ratio(float q_rain) 
{
    if (!std::isfinite(static_cast<double>(q_rain)) || q_rain <= 0.0f)
    {
        return 0.95f;
    }


    float rain_rate = q_rain * 3600.0f;

    if (rain_rate < 1.0f) 
    {
        return 0.95f;
    } 

    else if (rain_rate < 10.0f) 
    {
        return 0.85f;
    } 
    else 
    {
        return 0.75f;
    }
}

/**
 * @brief Computes the polarimetric rain.
 */
void ZDRScheme::compute_polarimetric_rain(const Field3D& q_rain, RadarOut& out) 
{

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < NR_; ++i) 
    {
        for (int j = 0; j < NTH_; ++j) 
        {
            for (int k = 0; k < NZ_; ++k) 
            {
                float q_val = static_cast<float>(q_rain[i][j][k]);
                if (!std::isfinite(static_cast<double>(q_val)))
                {
                    continue;
                }

                if (q_val <= 0.0f) 
                {
                    continue;
                }

                float axis_ratio = estimate_rain_axis_ratio(q_val);

                double rho_ratio = 1000.0 / 1000.0;
                double Ze_base = 3.63e9 * rho_ratio * rho_ratio * q_val * q_val;


                double f_h = 1.0 / (axis_ratio * axis_ratio);
                double f_v = axis_ratio * axis_ratio * axis_ratio * axis_ratio;

                double Z_H_linear = Ze_base * f_h;
                double Z_V_linear = Ze_base * f_v;

                float Z_H_dBZ_existing = out.ZH_dBZ[i][j][k];
                float Z_V_dBZ_existing = out.ZV_dBZ[i][j][k];

                float Z_H_linear_existing = dbz_to_linear(Z_H_dBZ_existing);
                float Z_V_linear_existing = dbz_to_linear(Z_V_dBZ_existing);

                float Z_H_linear_total = Z_H_linear_existing + Z_H_linear;
                float Z_V_linear_total = Z_V_linear_existing + Z_V_linear;

                out.ZH_dBZ[i][j][k] = dbz_from_linear(Z_H_linear_total);
                out.ZV_dBZ[i][j][k] = dbz_from_linear(Z_V_linear_total);
            }
        }
    }
}

/**
 * @brief Computes the polarimetric ice.
 */
void ZDRScheme::compute_polarimetric_ice(const Field3D& q_ice,
                                         const std::string& species, RadarOut& out) {

    double rho_hyd;
    if (species == "snow") rho_hyd = 100.0;
    else if (species == "graupel") rho_hyd = 400.0;
    else if (species == "hail") rho_hyd = 900.0;
    else rho_hyd = 917.0;

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < NR_; ++i) 
    {
        for (int j = 0; j < NTH_; ++j) 
        {

            for (int k = 0; k < NZ_; ++k) 
            {
                float q_val = static_cast<float>(q_ice[i][j][k]);
                if (!std::isfinite(static_cast<double>(q_val)))
                {
                    continue;
                }

                if (q_val <= 0.0f) 
                {
                    continue;
                }

                double rho_ratio = rho_hyd / 1000.0;
                double Ze_base = 3.63e9 * rho_ratio * rho_ratio * q_val * q_val;

                double orientation_factor = 0.9;

                if (species == "hail") {
                    orientation_factor = 0.8;
                }

                double Z_H_linear = Ze_base * orientation_factor;
                double Z_V_linear = Ze_base / orientation_factor;

                float Z_H_dBZ_existing = out.ZH_dBZ[i][j][k];
                float Z_V_dBZ_existing = out.ZV_dBZ[i][j][k];

                float Z_H_linear_existing = dbz_to_linear(Z_H_dBZ_existing);
                float Z_V_linear_existing = dbz_to_linear(Z_V_dBZ_existing);

                float Z_H_linear_total = Z_H_linear_existing + Z_H_linear;
                float Z_V_linear_total = Z_V_linear_existing + Z_V_linear;

                out.ZH_dBZ[i][j][k] = dbz_from_linear(Z_H_linear_total);
                out.ZV_dBZ[i][j][k] = dbz_from_linear(Z_V_linear_total);
            }
        }
    }
}
