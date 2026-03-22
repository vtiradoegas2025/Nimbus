/**
 * @file kessler.cpp
 * @brief Implementation for the microphysics module.
 *
 * Provides executable logic for the microphysics runtime path,
 * including initialization, stepping, and diagnostics helpers.
 * This file is part of the src/microphysics subsystem.
 */

#include "kessler.hpp"
#include "core/field_pool.hpp"
#include "core/simulation.hpp"
#include "numerics/compute_kernel_template.hpp"
#include <algorithm>
#include <cmath>
#ifdef _OPENMP
#include <omp.h>
#endif

/**
 * @brief The constructor for the KesslerScheme class.
 */
KesslerScheme::KesslerScheme(
    double qc0, double c_auto, double c_accr, double c_evap,
    double c_freeze, double c_rime, double c_melt, double c_subl,
    double a_term, double b_term, double Vt_max,
    double a_hail, double b_hail, double a_grau, double b_grau,
    double Vt_max_hail, double Vt_max_grau
) : qc0_(qc0), c_auto_(c_auto), c_accr_(c_accr), c_evap_(c_evap),
    c_freeze_(c_freeze), c_rime_(c_rime), c_melt_(c_melt), c_subl_(c_subl),
    a_term_(a_term), b_term_(b_term), Vt_max_(Vt_max),
    a_hail_(a_hail), b_hail_(b_hail), a_grau_(a_grau), b_grau_(b_grau),
    Vt_max_hail_(Vt_max_hail), Vt_max_grau_(Vt_max_grau) {}


/**
 * @brief Computes the tendencies for the Kessler scheme.
 */
void KesslerScheme::compute_tendencies(
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
    if (NTH == 0) return;
    int NZ = p.size_z();

    auto temp_guard = FieldPool::instance().scoped_acquire(NR, NTH, NZ);
    Field3D& temperature = temp_guard.field;
    thermodynamics::convert_theta_to_temperature_field(theta, p, temperature);

    init_tendency_fields(NR, NTH, NZ, dtheta_dt, dqv_dt, dqc_dt, dqr_dt,
                         dqi_dt, dqs_dt, dqg_dt, dqh_dt);

    // Try GPU dispatch for fused point-wise processes (warm rain + ice + melting)
    const float Lv_cp = static_cast<float>(microphysics_constants::L_v / microphysics_constants::cp);
    const float Lf_cp = static_cast<float>(microphysics_constants::L_f / microphysics_constants::cp);
    const float Ls_cp = static_cast<float>(microphysics_constants::L_s / microphysics_constants::cp);

    bool gpu_pointwise_ok = dispatch_kessler_pointwise_backend(
        temperature.data(), qv.data(), qc.data(), qr.data(), qg.data(), qh.data(),
        dtheta_dt.data(), dqv_dt.data(), dqc_dt.data(), dqr_dt.data(),
        dqg_dt.data(), dqh_dt.data(),
        NR, NTH, NZ,
        static_cast<float>(qc0_), static_cast<float>(c_auto_),
        static_cast<float>(c_accr_), static_cast<float>(c_evap_),
        static_cast<float>(c_freeze_), static_cast<float>(c_rime_),
        static_cast<float>(c_melt_), static_cast<float>(c_subl_),
        Lv_cp, Lf_cp, Ls_cp,
        static_cast<float>(microphysics_constants::T0));

    if (gpu_pointwise_ok)
    {
        // Try GPU sedimentation (operates on tendencies from point-wise pass)
        const double dz_local = std::max(::dz, 1.0e-6);
        bool gpu_sed_ok = dispatch_kessler_sedimentation_backend(
            qr.data(), qg.data(), qh.data(),
            dqr_dt.data(), dqg_dt.data(), dqh_dt.data(),
            NR, NTH, NZ,
            static_cast<float>(dz_local),
            static_cast<float>(a_term_), static_cast<float>(b_term_),
            static_cast<float>(Vt_max_),
            static_cast<float>(a_grau_), static_cast<float>(b_grau_),
            static_cast<float>(Vt_max_grau_),
            static_cast<float>(a_hail_), static_cast<float>(b_hail_),
            static_cast<float>(Vt_max_hail_));

        if (!gpu_sed_ok)
        {
            compute_sedimentation(qr, qg, qh, dqr_dt, dqg_dt, dqh_dt);
        }
    }
    else
    {
        // CPU fallback for all processes
        compute_warm_rain_processes(temperature, qv, qc, qr, dqc_dt, dqr_dt, dqv_dt, dtheta_dt);
        compute_ice_processes(temperature, qv, qc, qr, qg, qh, dqc_dt, dqg_dt, dqh_dt, dqv_dt, dtheta_dt);
        compute_melting_processes(temperature, qg, qh, dqg_dt, dqh_dt, dqr_dt, dtheta_dt);
        compute_sedimentation(qr, qg, qh, dqr_dt, dqg_dt, dqh_dt);
    }

    #pragma omp parallel for collapse(2)
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
 * @brief Computes the warm rain processes for the Kessler scheme.
 */
void KesslerScheme::compute_warm_rain_processes(
    const Field3D& temperature,
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
                    float qvsat = 0.001f;
                    float RH = (qv_val > 0.0f) ? qv_val / qvsat : 0.0f;

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
 * @brief Computes the ice processes for the Kessler scheme.
 */
void KesslerScheme::compute_ice_processes(
    const Field3D& temperature,
    const Field3D& qv,
    const Field3D& qc,
    const Field3D& qr,
    const Field3D& qg,
    const Field3D& qh,
    Field3D& dqc_dt,
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
                float qc_val = qc[i][j][k];
                float qr_val = qr[i][j][k];
                float qg_val = qg[i][j][k];
                float qh_val = qh[i][j][k];
                float qv_val = qv[i][j][k];
                float T = temperature[i][j][k];

                bool is_cold = (T < microphysics_constants::T0);

                if (is_cold) 
                {
                    if (qc_val > 0.0f) 
                    {
                        float freeze_rate = c_freeze_ * qc_val;
                        dqc_dt[i][j][k] -= freeze_rate;
                        dqg_dt[i][j][k] += freeze_rate;
                        dtheta_dt[i][j][k] += microphysics_constants::L_f / microphysics_constants::cp * freeze_rate / T;
                    }

                    if (qc_val > 0.0f && qg_val > 0.0f) 
                    {
                        float rime_rate = c_rime_ * qc_val * qg_val;
                        dqc_dt[i][j][k] -= rime_rate;
                        dqg_dt[i][j][k] += rime_rate;
                        dtheta_dt[i][j][k] += microphysics_constants::L_f / microphysics_constants::cp * rime_rate / T;
                    }

                    if (qc_val > 0.0f && qh_val > 0.0f) 
                    {
                        float rime_rate = c_rime_ * qc_val * qh_val;
                        dqc_dt[i][j][k] -= rime_rate;
                        dqh_dt[i][j][k] += rime_rate;
                        dtheta_dt[i][j][k] += microphysics_constants::L_f / microphysics_constants::cp * rime_rate / T;
                    }

                    if (qg_val > qc0_) 
                    {
                        float conv_rate = c_auto_ * (qg_val - qc0_);
                        dqg_dt[i][j][k] -= conv_rate;
                        dqh_dt[i][j][k] += conv_rate;
                    }

                    float qvsat_ice = 0.001f;
                    float RH_ice = (qv_val > 0.0f) ? qv_val / qvsat_ice : 0.0f;

                    if (RH_ice < 1.0f) 
                    {
                        if (qg_val > 0.0f) 
                        {
                            float subl_rate = c_subl_ * (1.0f - RH_ice) * qg_val;
                            dqv_dt[i][j][k] += subl_rate;
                            dqg_dt[i][j][k] -= subl_rate;
                            dtheta_dt[i][j][k] -= microphysics_constants::L_s / microphysics_constants::cp * subl_rate / T;
                        }

                        if (qh_val > 0.0f)
                        {
                            float subl_rate = c_subl_ * (1.0f - RH_ice) * qh_val;
                            dqv_dt[i][j][k] += subl_rate;
                            dqh_dt[i][j][k] -= subl_rate;
                            dtheta_dt[i][j][k] -= microphysics_constants::L_s / microphysics_constants::cp * subl_rate / T;
                        }
                    }
                }
            }
        }
    }
}

/**
 * @brief Computes the melting processes for the Kessler scheme.
 */
void KesslerScheme::compute_melting_processes(
    const Field3D& temperature,
    const Field3D& qg,
    const Field3D& qh,
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
                float qg_val = static_cast<float>(qg[i][j][k]);
                float qh_val = static_cast<float>(qh[i][j][k]);
                float T = static_cast<float>(temperature[i][j][k]);

                if (T > microphysics_constants::T0) 
                {
                    if (qh_val > 0.0f) 
                    {
                        float melt_rate = c_melt_ * qh_val;
                        dqh_dt[i][j][k] -= melt_rate;
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
                }
            }
        }
    }
}

/**
 * @brief Computes the sedimentation for the Kessler scheme.
 */
void KesslerScheme::compute_sedimentation(
    const Field3D& qr,
    const Field3D& qg,
    const Field3D& qh,
    Field3D& dqr_dt,
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
                    float Vt_val = a_term_ * std::pow(std::max(qr_val, 1e-6f), b_term_);
                    float Vt = std::min(static_cast<float>(Vt_max_), Vt_val);
                    float sed_flux = Vt * qr_val;
                    dqr_dt[i][j][k] -= sed_flux / dz_local;
                    dqr_dt[i][j][k - 1] += sed_flux / dz_local;
                }

                float qg_val = static_cast<float>(qg[i][j][k]);
                if (qg_val > 0.0f && k > 0) 
                {
                    float Vt_val = a_grau_ * std::pow(std::max(qg_val, 1e-6f), b_grau_);
                    float Vt = std::min(static_cast<float>(Vt_max_grau_), Vt_val);
                    float sed_flux = Vt * qg_val;
                    dqg_dt[i][j][k] -= sed_flux / dz_local;
                    dqg_dt[i][j][k - 1] += sed_flux / dz_local;
                }

                float qh_val = static_cast<float>(qh[i][j][k]);
                if (qh_val > 0.0f && k > 0) 
                {
                    float Vt_val = a_hail_ * std::pow(std::max(qh_val, 1e-6f), b_hail_);
                    float Vt = std::min(static_cast<float>(Vt_max_hail_), Vt_val);
                    float sed_flux = Vt * qh_val;
                    dqh_dt[i][j][k] -= sed_flux / dz_local;
                    dqh_dt[i][j][k - 1] += sed_flux / dz_local;
                }
            }
        }
    }
}

/**
 * @brief Computes the radar reflectivity for the Kessler scheme.
 */
void KesslerScheme::compute_radar_reflectivity(
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
    if (NTH == 0) return;
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
