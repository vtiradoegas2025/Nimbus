/**
 * @file zdr.cpp
 * @brief Implementation for the radar module.
 *
 * Provides executable logic for the radar runtime path,
 * including initialization, stepping, and diagnostics helpers.
 * This file is part of the src/radar subsystem.
 */

#include "zdr.hpp"
#include "core/field/field3d.hpp"
#include "radar/polarimetric/rain_mp.hpp"
#include "util/log.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
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

float local_dry_air_density(const Field3D* rho_dry, int i, int j, int k)
{
    if (rho_dry == nullptr)
    {
        throw std::invalid_argument(
            "ZDR requires prognostic dry-air density");
    }
    return (*rho_dry)[i][j][k];
}

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
    ensure_field_shape(out.KDP_deg_km, nr, nth, nz, 0.0f);
    ensure_field_shape(out.rhohv, nr, nth, nz, 1.0f);
    ensure_field_shape(out.Z_dBZ, nr, nth, nz, kRadarDbzMin);
    ensure_field_shape(out.Ze_linear, nr, nth, nz, 0.0f);
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

namespace tmv
{

/**
 * @brief Initializes the ZDR scheme.
 */
void ZDRScheme::initialize(const RadarConfig& config, int NR, int NTH, int NZ)
{
    NR_ = NR;
    NTH_ = NTH;
    NZ_ = NZ;

    tmv::log_info("Initialized ZDR radar scheme (", NR, "x", NTH, "x", NZ, ")");
    tmv::log_info("Operator tier: ", config.operator_tier);
    if (config.operator_tier == "zhang2021_rain_mp")
    {
        tmv::log_info("Polarimetric scope: Zhang-Gao-Du (2021) "
                      "fixed-N0 rain-only forward operator");
    }
    else
    {
        tmv::log_warn("ZDR is a synthetic bulk diagnostic; its hydrometeor "
                      "shape/reflectivity relations are not yet DSD-qualified.");
    }
}

void ZDRScheme::invalidate_generation_state_after_domain_recycling() noexcept
{
    candidate_output_.invalidate_generation_values();
}

/**
 * @brief Computes the ZDR scheme.
 */
void ZDRScheme::compute(const RadarConfig& config, const RadarStateView& state, RadarOut& out) 
{
    validate_radar_dimensions(state, NR_, NTH_, NZ_, "ZDR");
    if (config.operator_tier != "simple"
        && config.operator_tier != "polarimetric_fo"
        && config.operator_tier != "zhang2021_rain_mp")
        throw std::invalid_argument(
            "Unknown ZDR operator tier: " + config.operator_tier);
    validate_radar_hydrometeor_state(
        config, state, "ZDR", /*require_density=*/true);
    prepare_zdr_output(candidate_output_, NR_, NTH_, NZ_);

    if (config.operator_tier == "simple") 
    {
        compute_simple_zdr(config, state, candidate_output_);
    } 

    else if (config.operator_tier == "polarimetric_fo") 
    {
        compute_polarimetric_fo(config, state, candidate_output_);
    }
    else
    {
        compute_zhang2021_rain_mp(state, candidate_output_);
    }
    validate_radar_output_field(candidate_output_.Ze_linear,
        NR_, NTH_, NZ_, 0.0f, kRadarLinearMax, "Ze_linear");
    validate_radar_output_field(candidate_output_.Z_dBZ,
        NR_, NTH_, NZ_, kRadarDbzMin, kRadarDbzMax, "Z_dBZ");
    validate_radar_output_field(candidate_output_.ZH_dBZ,
        NR_, NTH_, NZ_, kRadarDbzMin, kRadarDbzMax, "ZH_dBZ");
    validate_radar_output_field(candidate_output_.ZV_dBZ,
        NR_, NTH_, NZ_, kRadarDbzMin, kRadarDbzMax, "ZV_dBZ");
    validate_radar_output_field(candidate_output_.ZDR_dB,
        NR_, NTH_, NZ_, kRadarZdrMin, kRadarZdrMax, "ZDR_dB");
    validate_radar_output_field(candidate_output_.KDP_deg_km,
        NR_, NTH_, NZ_, 0.0f, 4.0f, "KDP_deg_km");
    validate_radar_output_field(candidate_output_.rhohv,
        NR_, NTH_, NZ_, 0.8f, 1.0f, "rhohv");
    out.NR = NR_;
    out.NTH = NTH_;
    out.NZ = NZ_;
    std::swap(out.Ze_linear, candidate_output_.Ze_linear);
    std::swap(out.Z_dBZ, candidate_output_.Z_dBZ);
    std::swap(out.ZH_dBZ, candidate_output_.ZH_dBZ);
    std::swap(out.ZV_dBZ, candidate_output_.ZV_dBZ);
    std::swap(out.ZDR_dB, candidate_output_.ZDR_dB);
    std::swap(out.KDP_deg_km, candidate_output_.KDP_deg_km);
    std::swap(out.rhohv, candidate_output_.rhohv);
}

void ZDRScheme::compute_zhang2021_rain_mp(const RadarStateView& state,
                                          RadarOut& out)
{
    for (int i = 0; i < NR_; ++i)
    {
        for (int j = 0; j < NTH_; ++j)
        {
            for (int k = 0; k < NZ_; ++k)
            {
                for (const Field3D* frozen : {state.qs, state.qg, state.qh, state.qi})
                {
                    if (frozen != nullptr)
                    {
                        const float mixing_ratio = (*frozen)[i][j][k];
                        if (!std::isfinite(static_cast<double>(mixing_ratio)) ||
                            mixing_ratio != 0.0f)
                        {
                            throw std::domain_error(
                                "zhang2021_rain_mp supports rain-only columns");
                        }
                    }
                }
                (void)radar::polarimetric::zhang2021_rain_mp(
                    state.qr == nullptr ? 0.0 : (*state.qr)[i][j][k],
                    local_dry_air_density(state.rho, i, j, k));
            }
        }
    }

    #pragma omp parallel for collapse(2) if(static_cast<std::size_t>(NR_) * NTH_ * NZ_ >= 32768)
    for (int i = 0; i < NR_; ++i)
    {
        for (int j = 0; j < NTH_; ++j)
        {
            for (int k = 0; k < NZ_; ++k)
            {
                const auto value = radar::polarimetric::zhang2021_rain_mp(
                    state.qr == nullptr ? 0.0 : (*state.qr)[i][j][k],
                    local_dry_air_density(state.rho, i, j, k));
                out.Ze_linear[i][j][k] = static_cast<float>(value.zh_linear);
                out.Z_dBZ[i][j][k] = dbz_from_linear(out.Ze_linear[i][j][k]);
                out.ZH_dBZ[i][j][k] = out.Z_dBZ[i][j][k];
                const double zv_linear = value.zh_linear /
                    std::pow(10.0, value.zdr_db / 10.0);
                out.ZV_dBZ[i][j][k] = dbz_from_linear(static_cast<float>(zv_linear));
                out.ZDR_dB[i][j][k] = static_cast<float>(value.zdr_db);
                out.KDP_deg_km[i][j][k] = static_cast<float>(value.kdp_deg_km);
                out.rhohv[i][j][k] = static_cast<float>(value.rhohv);
            }
        }
    }
}

/**
 * @brief Computes the simple ZDR.
 */
void ZDRScheme::compute_simple_zdr(const RadarConfig& config, const RadarStateView& state, RadarOut& out) 
{
    (void)config;

    #pragma omp parallel for collapse(2) if(static_cast<std::size_t>(NR_) * NTH_ * NZ_ >= 32768)
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

                const float rho_dry_for_shape = state.rho ? (*state.rho)[i][j][k] : 0.0f;
                const float rho_dry = local_dry_air_density(state.rho, i, j, k);
                float axis_ratio = estimate_rain_axis_ratio(q_rain, rho_dry_for_shape);
                axis_ratio = std::clamp(axis_ratio, 0.4f, 1.0f);

                float Z_H, Z_V;
                compute_rain_polarization(rho_dry * q_rain, axis_ratio, Z_H, Z_V);

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

    #pragma omp parallel for collapse(2) if(static_cast<std::size_t>(NR_) * NTH_ * NZ_ >= 32768)
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
        compute_polarimetric_rain(*state.qr, state.rho, out);
    }

    if (state.qs && config.has_qs) 
    {
        compute_polarimetric_ice(*state.qs, state.rho, "snow", out);
    }

    if (state.qg && config.has_qg) 
    {
        compute_polarimetric_ice(*state.qg, state.rho, "graupel", out);
    }

    if (state.qh && config.has_qh) 
    {
        compute_polarimetric_ice(*state.qh, state.rho, "hail", out);
    }

    if (state.qi && config.has_qi) 
    {
        compute_polarimetric_ice(*state.qi, state.rho, "ice", out);
    }

    #pragma omp parallel for collapse(2) if(static_cast<std::size_t>(NR_) * NTH_ * NZ_ >= 32768)
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
void ZDRScheme::compute_rain_polarization(float rain_mass_concentration,
                                          float axis_ratio,
                                          float& Z_H,
                                          float& Z_V)
{
    const float mass_safe =
        (std::isfinite(static_cast<double>(rain_mass_concentration)) &&
         rain_mass_concentration > 0.0f)
        ? rain_mass_concentration
        : 0.0f;
    const float axis_safe = std::clamp(axis_ratio, 0.4f, 1.0f);
    double rho_ratio = 1000.0 / 1000.0;
    double Ze_base = 3.63e9 * rho_ratio * rho_ratio * mass_safe * mass_safe;
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
float ZDRScheme::estimate_rain_axis_ratio(float q_rain, float rho_dry)
{
    if (!std::isfinite(static_cast<double>(q_rain)) || q_rain <= 0.0f ||
        !std::isfinite(static_cast<double>(rho_dry)) || rho_dry <= 0.0f)
    {
        return 0.95f;
    }

    // rho*q*Vt is a precipitation mass flux (kg m^-2 s^-1); multiplying by
    // 3600 converts it to the numerically equivalent liquid depth in mm/h.
    // This remains a bulk shape heuristic, but no longer treats kg/kg as a
    // precipitation rate.
    const float rain_rate = radar_bulk_rain_rate_mmph(q_rain, rho_dry);

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
void ZDRScheme::compute_polarimetric_rain(const Field3D& q_rain,
                                          const Field3D* rho_dry,
                                          RadarOut& out)
{

    #pragma omp parallel for collapse(2) if(static_cast<std::size_t>(NR_) * NTH_ * NZ_ >= 32768)
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

                const float rho_dry_for_shape = rho_dry ? (*rho_dry)[i][j][k] : 0.0f;
                const float rho_dry_val = local_dry_air_density(rho_dry, i, j, k);
                float axis_ratio = estimate_rain_axis_ratio(q_val, rho_dry_for_shape);

                double rho_ratio = 1000.0 / 1000.0;
                const double mass_concentration =
                    static_cast<double>(rho_dry_val) * static_cast<double>(q_val);
                double Ze_base = 3.63e9 * rho_ratio * rho_ratio *
                                 mass_concentration * mass_concentration;


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
                                         const Field3D* rho_dry,
                                         const std::string& species,
                                         RadarOut& out) {

    double rho_hyd;
    if (species == "snow") rho_hyd = 100.0;
    else if (species == "graupel") rho_hyd = 400.0;
    else if (species == "hail") rho_hyd = 900.0;
    else rho_hyd = 917.0;

    #pragma omp parallel for collapse(2) if(static_cast<std::size_t>(NR_) * NTH_ * NZ_ >= 32768)
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

                const double rho_dry_val = local_dry_air_density(rho_dry, i, j, k);
                const double mass_concentration =
                    rho_dry_val * static_cast<double>(q_val);
                double rho_ratio = rho_hyd / 1000.0;
                double Ze_base = 3.63e9 * rho_ratio * rho_ratio *
                                 mass_concentration * mass_concentration;

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

}  // namespace tmv
