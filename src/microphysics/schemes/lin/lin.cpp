/**
 * @file lin.cpp
 * @brief Implementation for the microphysics module.
 *
 * Provides executable logic for the microphysics runtime path,
 * including initialization, stepping, and diagnostics helpers.
 * This file is part of the src/microphysics subsystem.
 */

#include "lin.hpp"
#include "core/simulation.hpp"
#include <algorithm>
#include <cmath>
#ifdef _OPENMP
#include <omp.h>
#endif

/**
 * @brief Initializes Lin scheme coefficients and hydrometeor parameters.
 */
LinScheme::LinScheme(
    double qc0, double c_auto, double c_accr, double c_evap,
    double c_ihom, double c_rime, double c_agg, double c_melt, double c_subl,
    double N0r, double N0s, double N0g, double N0h,
    double a_r, double b_r, double a_s, double b_s, double a_g, double b_g, double a_h, double b_h
) : qc0_(qc0), c_auto_(c_auto), c_accr_(c_accr), c_evap_(c_evap),
    c_ihom_(c_ihom), c_rime_(c_rime), c_agg_(c_agg), c_melt_(c_melt), c_subl_(c_subl),
    N0r_(N0r), N0s_(N0s), N0g_(N0g), N0h_(N0h),
    a_r_(a_r), b_r_(b_r), a_s_(a_s), b_s_(b_s),
    a_g_(a_g), b_g_(b_g), a_h_(a_h), b_h_(b_h) {}

/**
 * @brief Computes the tendencies for the Lin scheme.
 */
void LinScheme::compute_tendencies(
    const Field3D& p,
    const Field3D& theta,
    const Field3D& qv,
    const Field3D& qc,
    const Field3D& qr,
    const Field3D& qi,
    const Field3D& qs,
    const Field3D& qg,
    const Field3D& qh,
    double dt,
    Field3D& dtheta_dt,
    Field3D& dqv_dt,
    Field3D& dqc_dt,
    Field3D& dqr_dt,
    Field3D& dqi_dt,
    Field3D& dqs_dt,
    Field3D& dqg_dt,
    Field3D& dqh_dt
) 
{
    int NR = p.size_r();
    if (NR == 0) return;
    int NTH = p.size_th();
    int NZ = p.size_z();

    Field3D temperature(NR, NTH, NZ);
    thermodynamics::convert_theta_to_temperature_field(theta, p, temperature);

    init_tendency_fields(NR, NTH, NZ, dtheta_dt, dqv_dt, dqc_dt, dqr_dt,
                         dqi_dt, dqs_dt, dqg_dt, dqh_dt);

    Field3D qv_temp = qv;
    Field3D qc_temp = qc;
    saturation_adjustment(temperature, p, qv_temp, qc_temp);

    compute_warm_rain_processes(temperature, p, qv_temp, qc_temp, qr, dqc_dt, dqr_dt, dqv_dt, dtheta_dt);
    compute_ice_processes(temperature, p, qv_temp, qc_temp, qi, qs, qg, qh,
                         dqc_dt, dqi_dt, dqs_dt, dqg_dt, dqh_dt, dqv_dt, dtheta_dt);
    compute_melting_processes(temperature, qs, qg, qh, dqs_dt, dqg_dt, dqh_dt, dqr_dt, dtheta_dt);
    compute_sedimentation(qr, qs, qg, qh, dqr_dt, dqs_dt, dqg_dt, dqh_dt);

    for (int i = 0; i < NR; ++i) 
    {
        for (int j = 0; j < NTH; ++j) 
        {
            for (int k = 0; k < NZ; ++k) 
            {
                dtheta_dt[i][j][k] = static_cast<float>(
                    thermodynamics::temperature_tendency_to_theta(
                        static_cast<float>(dtheta_dt[i][j][k]), static_cast<float>(theta[i][j][k]), static_cast<float>(p[i][j][k])
                    )
                );
            }
        }
    }
}


/**
 * @brief Computes the saturation adjustment for the Lin scheme.
 */
void LinScheme::saturation_adjustment(
    const Field3D& temperature,
    const Field3D& p,
    Field3D& qv,
    Field3D& qc
) 
{
    int NR = temperature.size_r();
    int NTH = temperature.size_th();
    int NZ = temperature.size_z();

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < NR; ++i) 
    {
        for (int j = 0; j < NTH; ++j) 
        {
            for (int k = 0; k < NZ; ++k) 
            {
                float T = static_cast<float>(temperature[i][j][k]);
                float P = static_cast<float>(p[i][j][k]);
                float qvsat = thermodynamics::saturation_mixing_ratio_water(T, P);

                float qv_val = static_cast<float>(qv[i][j][k]);
                if (qv_val > qvsat) 
                {
                    float excess = qv_val - qvsat;
                    qc[i][j][k] += excess;
                    qv[i][j][k] = qvsat;
                }
            }
        }
    }
}


/**
 * @brief Computes the warm rain processes for the Lin scheme.
 */
void LinScheme::compute_warm_rain_processes(
    const Field3D& temperature,
    const Field3D& p,
    const Field3D& qv,
    const Field3D& qc,
    const Field3D& qr,
    Field3D& dqc_dt,
    Field3D& dqr_dt,
    Field3D& dqv_dt,
    Field3D& dtheta_dt
) 
{
    int NR = temperature.size_r();
    int NTH = temperature.size_th();
    int NZ = temperature.size_z();

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < NR; ++i) 
    {
        for (int j = 0; j < NTH; ++j) 
        {
            for (int k = 0; k < NZ; ++k) 
            {
                float qc_val = static_cast<float>(qc[i][j][k]);
                float qr_val = static_cast<float>(qr[i][j][k]);
                float qv_val = static_cast<float>(qv[i][j][k]);
                float T = static_cast<float>(temperature[i][j][k]);
                float P = static_cast<float>(p[i][j][k]);

                if (qc_val > qc0_) 
                {
                    float auto_rate = c_auto_ * (qc_val - qc0_);
                    dqc_dt[i][j][k] -= auto_rate;
                    dqr_dt[i][j][k] += auto_rate;
                    dtheta_dt[i][j][k] += microphysics_constants::L_v / microphysics_constants::cp * auto_rate / T;
                }

                if (qc_val > 0.0f && qr_val > 0.0f) 
                {
                    float accr_rate = c_accr_ * qc_val * qr_val;
                    dqc_dt[i][j][k] -= accr_rate;
                    dqr_dt[i][j][k] += accr_rate;
                    dtheta_dt[i][j][k] += microphysics_constants::L_v / microphysics_constants::cp * accr_rate / T;
                }

                if (qr_val > 0.0f) 
                {
                    float qvsat = thermodynamics::saturation_mixing_ratio_water(T, P);
                    float qvsat_safe = std::max(qvsat, 1.0e-12f);
                    float RH = (qv_val > 0.0f) ? qv_val / qvsat_safe : 0.0f;

                    if (RH < 1.0f) 
                    {
                        float evap_rate = c_evap_ * (1.0f - RH) * qr_val;
                        dqv_dt[i][j][k] += evap_rate;
                        dqr_dt[i][j][k] -= evap_rate;
                        dtheta_dt[i][j][k] -= microphysics_constants::L_v / microphysics_constants::cp * evap_rate / T;
                    }
                }
            }
        }
    }
}


/**
 * @brief Computes the ice processes for the Lin scheme.
 */
void LinScheme::compute_ice_processes(
    const Field3D& temperature,
    const Field3D& p,
    const Field3D& qv,
    const Field3D& qc,
    const Field3D& qi,
    const Field3D& qs,
    const Field3D& qg,
    const Field3D& qh,
    Field3D& dqc_dt,
    Field3D& dqi_dt,
    Field3D& dqs_dt,
    Field3D& dqg_dt,
    Field3D& dqh_dt,
    Field3D& dqv_dt,
    Field3D& dtheta_dt
) 
{
    int NR = temperature.size_r();
    int NTH = temperature.size_th();
    int NZ = temperature.size_z();

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < NR; ++i) 
    {
        for (int j = 0; j < NTH; ++j) 
        {
            for (int k = 0; k < NZ; ++k) 
            {
                float qc_val = static_cast<float>(qc[i][j][k]);
                float qi_val = static_cast<float>(qi[i][j][k]);
                float qs_val = static_cast<float>(qs[i][j][k]);
                float qg_val = static_cast<float>(qg[i][j][k]);
                float qh_val = static_cast<float>(qh[i][j][k]);
                float qv_val = static_cast<float>(qv[i][j][k]);
                float T = static_cast<float>(temperature[i][j][k]);
                float P = static_cast<float>(p[i][j][k]);
                float T_celsius = T - microphysics_constants::T0;

                if (T < microphysics_constants::T0 && qc_val > 0.0f) 
                {
                    float nuc_rate = c_ihom_ * qc_val;
                    dqc_dt[i][j][k] -= nuc_rate;
                    dqi_dt[i][j][k] += nuc_rate;
                }

                if (qi_val > 0.0f && T < microphysics_constants::T0) 
                {
                    float qvsat_ice = thermodynamics::saturation_mixing_ratio_ice(T, P);

                    if (qv_val > qvsat_ice) 
                    {
                        float dep_rate = c_subl_ * (qv_val - qvsat_ice);
                        dqv_dt[i][j][k] -= dep_rate;
                        dqi_dt[i][j][k] += dep_rate;
                        dtheta_dt[i][j][k] += microphysics_constants::L_s / microphysics_constants::cp * dep_rate / T;
                    }
                }

                double Ni = ice_number_concentration(qi_val, T_celsius);
                double E_ac = collection_efficiency(T_celsius);

                if (qc_val > 0.0f && qi_val > 0.0f) 
                {
                    double rime_rate = M_PI/4.0 * 0.1 * E_ac * Ni * qc_val;
                    dqc_dt[i][j][k] -= rime_rate;
                    dqg_dt[i][j][k] += rime_rate;
                    dtheta_dt[i][j][k] += microphysics_constants::L_f / microphysics_constants::cp * rime_rate / T;
                }

                if (qi_val > 0.0f) 
                {
                    double q_I0 = ice_threshold_temperature(qi_val);

                    if (qi_val > q_I0) 
                    {
                        double agg_rate = c_agg_ * (qi_val - q_I0) * exp(0.025 * T_celsius);
                        dqi_dt[i][j][k] -= agg_rate;
                        dqs_dt[i][j][k] += agg_rate;
                    }
                }

                if (qi_val > 0.0f) 
                {
                    double auto_rate = c_auto_ * qi_val * qi_val;
                    dqi_dt[i][j][k] -= auto_rate;
                    dqs_dt[i][j][k] += auto_rate;
                }

                if (T < microphysics_constants::T0) 
                {
                    float qvsat_ice = thermodynamics::saturation_mixing_ratio_ice(T, P);
                    float qvsat_ice_safe = std::max(qvsat_ice, 1.0e-12f);
                    float RH_ice = (qv_val > 0.0f) ? qv_val / qvsat_ice_safe : 0.0f;

                    if (RH_ice < 1.0f) 
                    {
                        if (qi_val > 0.0f) 
                        {
                            float subl_rate = c_subl_ * (1.0f - RH_ice) * qi_val;
                            dqv_dt[i][j][k] += subl_rate;
                            dqi_dt[i][j][k] -= subl_rate;
                            dtheta_dt[i][j][k] -= microphysics_constants::L_s / microphysics_constants::cp * subl_rate / T;
                        }

                        if (qs_val > 0.0f) 
                        {
                            float subl_rate = c_subl_ * (1.0f - RH_ice) * qs_val;
                            dqv_dt[i][j][k] += subl_rate;
                            dqs_dt[i][j][k] -= subl_rate;
                            dtheta_dt[i][j][k] -= microphysics_constants::L_s / microphysics_constants::cp * subl_rate / T;
                        }
                    }
                }
            }
        }
    }
}


/**
 * @brief Computes the melting processes for the Lin scheme.
 */
void LinScheme::compute_melting_processes(
    const Field3D& temperature,
    const Field3D& qs,
    const Field3D& qg,
    const Field3D& qh,
    Field3D& dqs_dt,
    Field3D& dqg_dt,
    Field3D& dqh_dt,
    Field3D& dqr_dt,
    Field3D& dtheta_dt
) 
{
    int NR = temperature.size_r();
    int NTH = temperature.size_th();
    int NZ = temperature.size_z();

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < NR; ++i) 
    {
        for (int j = 0; j < NTH; ++j) 
        {
            for (int k = 0; k < NZ; ++k) 
            {
                float qs_val = static_cast<float>(qs[i][j][k]);
                float qg_val = static_cast<float>(qg[i][j][k]);
                float qh_val = static_cast<float>(qh[i][j][k]);
                float T = static_cast<float>(temperature[i][j][k]);

                if (T > microphysics_constants::T0) 
                {
                    if (qs_val > 0.0f) 
                    {
                        float melt_rate = c_melt_ * qs_val;
                        dqs_dt[i][j][k] -= melt_rate;
                        dqr_dt[i][j][k] += melt_rate;
                        dtheta_dt[i][j][k] -= microphysics_constants::L_f / microphysics_constants::cp * melt_rate / T;
                    }

                    if (qg_val > 0.0f) 
                    {
                        float melt_rate = c_melt_ * qg_val;
                        dqg_dt[i][j][k] -= melt_rate;
                        dqr_dt[i][j][k] += melt_rate;
                        dtheta_dt[i][j][k] -= microphysics_constants::L_f / microphysics_constants::cp * melt_rate / T;
                    }

                    if (qh_val > 0.0f) 
                    {
                        float melt_rate = c_melt_ * qh_val;
                        dqh_dt[i][j][k] -= melt_rate;
                        dqr_dt[i][j][k] += melt_rate;
                        dtheta_dt[i][j][k] -= microphysics_constants::L_f / microphysics_constants::cp * melt_rate / T;
                    }
                }
            }
        }
    }
}


/**
 * @brief Computes the sedimentation for the Lin scheme.
 */
void LinScheme::compute_sedimentation(
    const Field3D& qr,
    const Field3D& qs,
    const Field3D& qg,
    const Field3D& qh,
    Field3D& dqr_dt,
    Field3D& dqs_dt,
    Field3D& dqg_dt,
    Field3D& dqh_dt
) 
{
    int NR = qr.size_r();
    int NTH = qr.size_th();
    int NZ = qr.size_z();
    const double dz_local = std::max(::dz, 1.0e-6);

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < NR; ++i) 
    {
        for (int j = 0; j < NTH; ++j) 
        {
            for (int k = 0; k < NZ; ++k) 
            {
                float qr_val = static_cast<float>(qr[i][j][k]);
                if (qr_val > 0.0f && k > 0) 
                {
                    float Vt = a_r_ * std::pow(std::max(qr_val, 1e-6f), b_r_);
                    float sed_flux = Vt * qr_val;
                    dqr_dt[i][j][k] -= sed_flux / dz_local;
                    dqr_dt[i][j][k - 1] += sed_flux / dz_local;
                }

                float qs_val = static_cast<float>(qs[i][j][k]);
                if (qs_val > 0.0f && k > 0) 
                {
                    float Vt = a_s_ * std::pow(std::max(qs_val, 1e-6f), b_s_);
                    float sed_flux = Vt * qs_val;
                    dqs_dt[i][j][k] -= sed_flux / dz_local;
                    dqs_dt[i][j][k - 1] += sed_flux / dz_local;
                }

                float qg_val = static_cast<float>(qg[i][j][k]);
                if (qg_val > 0.0f && k > 0) 
                {
                    float Vt = a_g_ * std::pow(std::max(qg_val, 1e-6f), b_g_);
                    float sed_flux = Vt * qg_val;
                    dqg_dt[i][j][k] -= sed_flux / dz_local;
                    dqg_dt[i][j][k - 1] += sed_flux / dz_local;
                }

                float qh_val = static_cast<float>(qh[i][j][k]);
                if (qh_val > 0.0f && k > 0) 
                {
                    float Vt = a_h_ * std::pow(std::max(qh_val, 1e-6f), b_h_);
                    float sed_flux = Vt * qh_val;
                    dqh_dt[i][j][k] -= sed_flux / dz_local;
                    dqh_dt[i][j][k - 1] += sed_flux / dz_local;
                }
            }
        }
    }
}


/**
 * @brief Computes the radar reflectivity for the Lin scheme.
 */
void LinScheme::compute_radar_reflectivity(
    const Field3D& qc,
    const Field3D& qr,
    const Field3D& qi,
    const Field3D& qs,
    const Field3D& qg,
    const Field3D& qh,
    Field3D& reflectivity_dbz
) 
{
    int NR = qc.size_r();
    if (NR == 0) return;
    int NTH = qc.size_th();
    int NZ = qc.size_z();

    reflectivity_dbz.resize(NR, NTH, NZ, 0.0f);

    const float K_qc = 0.1e-3f;
    const float K_qr = 4.0e-3f;
    const float K_qi = 0.1e-3f;
    const float K_qs = 1.0e-3f;
    const float K_qg = 1.0e-2f;
    const float K_qh = 2.0e-2f;
    const float alpha = 1.5f;
    const float Z_min = 1e-10f;
    const float Z_max = 1.0e12f;
    const float Z_dbz_min = -30.0f;
    const float Z_dbz_max = 120.0f;

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < NR; ++i) 
    {
        for (int j = 0; j < NTH; ++j) 
        {
            for (int k = 0; k < NZ; ++k) 
            {
                float qc_val = static_cast<float>(qc[i][j][k]);
                float qr_val = static_cast<float>(qr[i][j][k]);
                float qi_val = static_cast<float>(qi[i][j][k]);
                float qs_val = static_cast<float>(qs[i][j][k]);
                float qg_val = static_cast<float>(qg[i][j][k]);
                float qh_val = static_cast<float>(qh[i][j][k]);

                float Z_linear = K_qc * std::pow(std::max(qc_val, 0.0f), alpha) +
                                K_qr * std::pow(std::max(qr_val, 0.0f), alpha) +
                                K_qi * std::pow(std::max(qi_val, 0.0f), alpha) +
                                K_qs * std::pow(std::max(qs_val, 0.0f), alpha) +
                                K_qg * std::pow(std::max(qg_val, 0.0f), alpha) +
                                K_qh * std::pow(std::max(qh_val, 0.0f), alpha);

                if (!std::isfinite(static_cast<double>(Z_linear)))
                {
                    Z_linear = Z_min;
                }
                Z_linear = std::clamp(Z_linear, Z_min, Z_max);
                float Z_dBZ = 10.0f * std::log10(Z_linear);
                if (!std::isfinite(static_cast<double>(Z_dBZ)))
                {
                    Z_dBZ = Z_dbz_min;
                }
                reflectivity_dbz[i][j][k] = std::clamp(Z_dBZ, Z_dbz_min, Z_dbz_max);
            }
        }
    }
}


/**
 * @brief Computes the ice number concentration for the Lin scheme.
 */
double LinScheme::ice_number_concentration(double qi, double T_celsius) 
{
    if (qi < 1e-6) 
    {
        return 1e3;
    } 

    else if (qi < 1e-3) 
    {
        return 1e3 * std::pow(qi / 1e-6, 0.15);
    } 
    else 
    {
        return 1e3 * std::pow(10.0, 0.45);
    }
}


/**
 * @brief Computes the collection efficiency for the Lin scheme.
 */
double LinScheme::collection_efficiency(double T_celsius) 
{
    if (T_celsius <= -5.0) 
    {
        return 1.0;
    } 
    else if (T_celsius >= 0.0) 
    {
        return 0.0;
    }
    else
    {
        return std::clamp((-T_celsius) / 5.0, 0.0, 1.0);
    }
}

/**
 * @brief Returns the empirical ice threshold used for aggregation onset.
 */
double LinScheme::ice_threshold_temperature(double qi) 
{
    return 1e-3 * exp(0.1 * 0.0);
}
