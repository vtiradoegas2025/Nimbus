/**
 * @file milbrandt.cpp
 * @brief Implementation for the microphysics module.
 *
 * Provides executable logic for the microphysics runtime path,
 * including initialization, stepping, and diagnostics helpers.
 * This file is part of the src/microphysics subsystem.
 */

#include "milbrandt.hpp"
#include "microphysics/base/thermodynamics.hpp"
#include "core/runtime/simulation.hpp"
#include <algorithm>
#include <cmath>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <limits>

/**
 * @brief Initializes the Milbrandt scheme with default parameters.
 */
MilbrandtScheme::MilbrandtScheme(
    double qc0, double c_auto,
    double alpha_r, double alpha_i, double alpha_s, double alpha_g, double alpha_h,
    double c_r, double d_r, double c_i, double d_i, double c_s, double d_s,
    double c_g, double d_g, double c_h, double d_h,
    double a_r, double b_r, double a_i, double b_i, double a_s, double b_s,
    double a_g, double b_g, double a_h, double b_h,
    bool triple_moment, bool hail_processes
) : qc0_(qc0), c_auto_(c_auto),
    alpha_r_(alpha_r), alpha_i_(alpha_i), alpha_s_(alpha_s), alpha_g_(alpha_g), alpha_h_(alpha_h),
    c_r_(c_r), d_r_(d_r), c_i_(c_i), d_i_(d_i), c_s_(c_s), d_s_(d_s),
    c_g_(c_g), d_g_(d_g), c_h_(c_h), d_h_(d_h),
    a_r_(a_r), b_r_(b_r), a_i_(a_i), b_i_(b_i), a_s_(a_s), b_s_(b_s),
    a_g_(a_g), b_g_(b_g), a_h_(a_h), b_h_(b_h),
    triple_moment_(triple_moment), hail_processes_(hail_processes) {}

/**
 * @brief Computes the tendencies for the Milbrandt scheme.
 */
void MilbrandtScheme::compute_tendencies(
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

    if (Nr_.size_r() != NR || Nr_.size_th() != NTH || Nr_.size_z() != NZ ||
        Ni_.size_r() != NR || Ni_.size_th() != NTH || Ni_.size_z() != NZ ||
        Ns_.size_r() != NR || Ns_.size_th() != NTH || Ns_.size_z() != NZ ||
        Ng_.size_r() != NR || Ng_.size_th() != NTH || Ng_.size_z() != NZ ||
        Nh_.size_r() != NR || Nh_.size_th() != NTH || Nh_.size_z() != NZ)
    {
        initialize_prognostic_fields(NR, NTH, NZ);
    }

    Field3D temperature(NR, NTH, NZ);
    thermodynamics::convert_theta_to_temperature_field(theta, p, temperature);

    Field3D rho(NR, NTH, NZ);

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < NR; ++i) 
    {
        for (int j = 0; j < NTH; ++j) 
        {
            for (int k = 0; k < NZ; ++k) 
            {
                rho[i][j][k] = static_cast<float>(p[i][j][k]) / (287.0f * static_cast<float>(temperature[i][j][k]) * (1.0f + 0.608f * static_cast<float>(qv[i][j][k])));
            }
        }
    }

    init_tendency_fields(NR, NTH, NZ, dtheta_dt, dqv_dt, dqc_dt, dqr_dt,
                         dqi_dt, dqs_dt, dqg_dt, dqh_dt);

    // Saturation adjustment: convert supersaturated vapor to cloud water
    // before processing microphysics. Without this, Milbrandt-Yau only
    // acts on existing hydrometeors and never initiates condensation.
    Field3D qv_adj = qv;
    Field3D qc_adj = qc;
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < NR; ++i)
    {
        for (int j = 0; j < NTH; ++j)
        {
            for (int k = 0; k < NZ; ++k)
            {
                double T_val = static_cast<double>(temperature[i][j][k]);
                double p_val = static_cast<double>(p[i][j][k]);
                double qv_val = static_cast<double>(qv[i][j][k]);
                double qc_val = static_cast<double>(qc[i][j][k]);
                thermodynamics::saturation_adjustment(T_val, p_val, qv_val, qc_val);
                qv_adj[i][j][k] = static_cast<float>(qv_val);
                qc_adj[i][j][k] = static_cast<float>(qc_val);
            }
        }
    }

    dNr_dt_.resize(NR, NTH, NZ, 0.0f);
    dNi_dt_.resize(NR, NTH, NZ, 0.0f);
    dNs_dt_.resize(NR, NTH, NZ, 0.0f);
    dNg_dt_.resize(NR, NTH, NZ, 0.0f);
    dNh_dt_.resize(NR, NTH, NZ, 0.0f);

    compute_warm_rain_processes(temperature, p, rho, qv_adj, qc_adj, qr,
                               dqc_dt, dqr_dt, dqv_dt, dtheta_dt, dNr_dt_);

    compute_ice_processes(temperature, p, rho, qv_adj, qc_adj, qi, qs, qg, qh,
                         dqc_dt, dqi_dt, dqs_dt, dqg_dt, dqh_dt, dqv_dt, dtheta_dt,
                         dNi_dt_, dNs_dt_, dNg_dt_, dNh_dt_);

    compute_melting_processes(temperature, qs, qg, qh,
                            dqs_dt, dqg_dt, dqh_dt, dqr_dt, dtheta_dt, dNr_dt_,
                            dNs_dt_, dNg_dt_, dNh_dt_);

    compute_sedimentation(qr, qs, qg, qh, Nr_, Ns_, Ng_, Nh_,
                         dqr_dt, dqs_dt, dqg_dt, dqh_dt,
                         dNr_dt_, dNs_dt_, dNg_dt_, dNh_dt_);

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < NR; ++i) 
    {
        for (int j = 0; j < NTH; ++j) 
        {
            for (int k = 0; k < NZ; ++k) 
            {
                Nr_[i][j][k] += dNr_dt_[i][j][k] * dt;
                Ni_[i][j][k] += dNi_dt_[i][j][k] * dt;
                Ns_[i][j][k] += dNs_dt_[i][j][k] * dt;
                Ng_[i][j][k] += dNg_dt_[i][j][k] * dt;
                Nh_[i][j][k] += dNh_dt_[i][j][k] * dt;

                Nr_[i][j][k] = std::max(static_cast<float>(Nr_[i][j][k]), 1.0f);
                Ni_[i][j][k] = std::max(static_cast<float>(Ni_[i][j][k]), 1.0f);
                Ns_[i][j][k] = std::max(static_cast<float>(Ns_[i][j][k]), 1.0f);
                Ng_[i][j][k] = std::max(static_cast<float>(Ng_[i][j][k]), 1.0f);
                Nh_[i][j][k] = std::max(static_cast<float>(Nh_[i][j][k]), 1.0f);
            }
        }
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
 * @brief Initializes the prognostic number concentrations for the Milbrandt scheme.
 */
void MilbrandtScheme::initialize_prognostic_fields(int NR, int NTH, int NZ) 
{
    Nr_.resize(NR, NTH, NZ, 1e6f);
    Ni_.resize(NR, NTH, NZ, 1e5f);
    Ns_.resize(NR, NTH, NZ, 1e5f);
    Ng_.resize(NR, NTH, NZ, 1e4f);
    Nh_.resize(NR, NTH, NZ, 1e3f);

    if (triple_moment_) 
    {
        Zr_.resize(NR, NTH, NZ, 1.0f);
        Zi_.resize(NR, NTH, NZ, 1.0f);
        Zs_.resize(NR, NTH, NZ, 1.0f);
        Zg_.resize(NR, NTH, NZ, 1.0f);
        Zh_.resize(NR, NTH, NZ, 1.0f);
    }
}

/**
 * @brief Calculates the lambda parameter for the Milbrandt scheme.
 */
double MilbrandtScheme::calculate_lambda(double q, double Nt, double alpha, double c, double d) 
{
    if (q <= 0.0 || Nt <= 0.0) return 1.0;

    double rho = 1.0;
    double lambda = pow(gamma_function(1.0 + d + alpha) /
                       (gamma_function(1.0 + alpha) * c * Nt / (rho * q)), 1.0 / d);
    return std::max(lambda, 1.0);
}

/**
 * @brief Computes intercept parameter for a generalized gamma distribution.
 */
double MilbrandtScheme::calculate_N0(double Nt, double alpha, double lambda) 
{
    return Nt * pow(lambda, 1.0 + alpha) / gamma_function(1.0 + alpha);
}

/**
 * @brief Computes a requested distribution moment for hydrometeor species.
 */
double MilbrandtScheme::calculate_moment(double q, double Nt, double alpha, double c, double d, int moment_order) 
{
    double lambda = calculate_lambda(q, Nt, alpha, c, d);
    return Nt * pow(lambda, -moment_order) * gamma_function(1.0 + alpha + moment_order);
}

/**
 * @brief Computes linear radar reflectivity contribution for one species.
 */
double MilbrandtScheme::calculate_reflectivity(double q, double Nt, double alpha, double c, double d) 
{
    double moment6 = calculate_moment(q, Nt, alpha, c, d, 6);
    return moment6 * 1e-18;
}


/**
 * @brief Computes warm-rain source and sink tendencies.
 */
void MilbrandtScheme::compute_warm_rain_processes(
    const Field3D& temperature,
    const Field3D& p,
    const Field3D& rho,
    const Field3D& qv,
    const Field3D& qc,
    const Field3D& qr,
    Field3D& dqc_dt,
    Field3D& dqr_dt,
    Field3D& dqv_dt,
    Field3D& dtheta_dt,
    Field3D& dNr_dt
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
                float rho_val = static_cast<float>(rho[i][j][k]);
                float Nr_val = static_cast<float>(Nr_[i][j][k]);

                if (qc_val > qc0_) 
                {
                    double auto_rate = autoconversion_rate(qc_val, rho_val);
                    dqc_dt[i][j][k] -= auto_rate;
                    dqr_dt[i][j][k] += auto_rate;
                    dtheta_dt[i][j][k] += microphysics_constants::L_v / microphysics_constants::cp * auto_rate / T;

                    dNr_dt[i][j][k] += auto_rate / (c_r_ * pow(1e-3, d_r_));
                }

                if (qc_val > 0.0f && qr_val > 0.0f) 
                {
                    double accr_rate = accretion_rate(qc_val, qr_val, 1e8, Nr_val,
                                                    0.0, alpha_r_, c_r_, d_r_, c_r_, d_r_, rho_val);
                    dqc_dt[i][j][k] -= accr_rate;
                    dqr_dt[i][j][k] += accr_rate;
                    dtheta_dt[i][j][k] += microphysics_constants::L_v / microphysics_constants::cp * accr_rate / T;
                }

                if (Nr_val > 0.0f) 
                {
                    double self_coll = rain_selfcollection_rate(Nr_val, rho_val);
                    dNr_dt[i][j][k] -= self_coll;
                }

                if (qr_val > 0.0f) 
                {
                    float qvsat = thermodynamics::saturation_mixing_ratio_water(T, P);
                    float qvsat_safe = std::max(qvsat, 1.0e-12f);
                    float RH = (qv_val > 0.0f) ? qv_val / qvsat_safe : 0.0f;

                    if (RH < 1.0f) 
                    {
                        double evap_rate = 1.0e-3 * (1.0f - RH) * qr_val;
                        dqv_dt[i][j][k] += evap_rate;
                        dqr_dt[i][j][k] -= evap_rate;
                        dtheta_dt[i][j][k] -= microphysics_constants::L_v / microphysics_constants::cp * evap_rate / T;

                        dNr_dt[i][j][k] -= evap_rate / (c_r_ * pow(1e-3, d_r_));
                    }
                }
            }
        }
    }
}

/**
 * @brief Computes the ice processes for the Milbrandt scheme.
 */
void MilbrandtScheme::compute_ice_processes(
    const Field3D& temperature,
    const Field3D& p,
    const Field3D& rho,
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
    Field3D& dtheta_dt,
    Field3D& dNi_dt,
    Field3D& dNs_dt,
    Field3D& dNg_dt,
    Field3D& dNh_dt
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
                float rho_val = static_cast<float>(rho[i][j][k]);

                float Ni_val = static_cast<float>(Ni_[i][j][k]);
                float Ns_val = static_cast<float>(Ns_[i][j][k]);
                float Ng_val = static_cast<float>(Ng_[i][j][k]);
                float Nh_val = static_cast<float>(Nh_[i][j][k]);

                
                if (qc_val > 0.0f && T < microphysics_constants::T0 && Ni_val < 1e6) 
                {
                    double nuc_rate = 1e-3 * qc_val;
                    dqc_dt[i][j][k] -= nuc_rate;
                    dqi_dt[i][j][k] += nuc_rate;
                    dNi_dt[i][j][k] += nuc_rate / (c_i_ * pow(1e-4, d_i_));
                }

                if (qi_val > 0.0f) 
                {
                    float qvsat_ice = thermodynamics::saturation_mixing_ratio_ice(T, P);

                    if (qv_val > qvsat_ice && T < microphysics_constants::T0) 
                    {
                        double dep_rate = 1e-3 * (qv_val - qvsat_ice);
                        dqv_dt[i][j][k] -= dep_rate;
                        dqi_dt[i][j][k] += dep_rate;
                        dtheta_dt[i][j][k] += microphysics_constants::L_s / microphysics_constants::cp * dep_rate / T;
                    } 

                    else if (qv_val < qvsat_ice) 
                    {
                        double subl_rate = 1e-3 * (qvsat_ice - qv_val);
                        dqv_dt[i][j][k] += subl_rate;
                        dqi_dt[i][j][k] -= subl_rate;
                        dtheta_dt[i][j][k] -= microphysics_constants::L_s / microphysics_constants::cp * subl_rate / T;
                    }
                }

                if (qc_val > 0.0f && qi_val > 0.0f) 
                {
                    double rime_rate = accretion_rate(qc_val, qi_val, 1e8, Ni_val,
                                                    0.0, alpha_i_, c_r_, d_r_, c_i_, d_i_, rho_val);
                    dqc_dt[i][j][k] -= rime_rate;
                    dqg_dt[i][j][k] += rime_rate;
                    dtheta_dt[i][j][k] += microphysics_constants::L_f / microphysics_constants::cp * rime_rate / T;
                }

                if (qi_val > 1e-6 && Ni_val > 1e3) 
                {
                    double agg_rate = 1e-3 * qi_val;
                    dqi_dt[i][j][k] -= agg_rate;
                    dqs_dt[i][j][k] += agg_rate;
                    dNi_dt[i][j][k] -= agg_rate / (c_i_ * pow(1e-4, d_i_));
                    dNs_dt[i][j][k] += agg_rate / (c_s_ * pow(1e-3, d_s_));
                }
            }
        }
    }
}

/**
 * @brief Computes melting exchanges between frozen and liquid hydrometeors.
 */
void MilbrandtScheme::compute_melting_processes(
    const Field3D& temperature,
    const Field3D& qs,
    const Field3D& qg,
    const Field3D& qh,
    Field3D& dqs_dt,
    Field3D& dqg_dt,
    Field3D& dqh_dt,
    Field3D& dqr_dt,
    Field3D& dtheta_dt,
    Field3D& dNr_dt,
    Field3D& dNs_dt,
    Field3D& dNg_dt,
    Field3D& dNh_dt
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
                float Ns_val = static_cast<float>(Ns_[i][j][k]);
                float Ng_val = static_cast<float>(Ng_[i][j][k]);
                float Nh_val = static_cast<float>(Nh_[i][j][k]);

                if (T > microphysics_constants::T0) 
                {
                    if (qs_val > 0.0f) 
                    {
                        double melt_rate = 1e-3 * qs_val;
                        dqs_dt[i][j][k] -= melt_rate;
                        dqr_dt[i][j][k] += melt_rate;
                        dtheta_dt[i][j][k] -= microphysics_constants::L_f / microphysics_constants::cp * melt_rate / T;
                        dNr_dt[i][j][k] += melt_rate / (c_r_ * pow(1e-3, d_r_));
                        dNs_dt[i][j][k] -= melt_rate / (c_s_ * pow(1e-3, d_s_));
                    }

                    if (qg_val > 0.0f) 
                    {
                        double melt_rate = 1e-3 * qg_val;
                        dqg_dt[i][j][k] -= melt_rate;
                        dqr_dt[i][j][k] += melt_rate;
                        dtheta_dt[i][j][k] -= microphysics_constants::L_f / microphysics_constants::cp * melt_rate / T;
                        dNr_dt[i][j][k] += melt_rate / (c_r_ * pow(1e-3, d_r_));
                        dNg_dt[i][j][k] -= melt_rate / (c_g_ * pow(1e-3, d_g_));
                    }

                    if (qh_val > 0.0f && hail_processes_) 
                    {
                        double melt_rate = 1e-3 * qh_val;
                        dqh_dt[i][j][k] -= melt_rate;
                        dqr_dt[i][j][k] += melt_rate;
                        dtheta_dt[i][j][k] -= microphysics_constants::L_f / microphysics_constants::cp * melt_rate / T;
                        dNr_dt[i][j][k] += melt_rate / (c_r_ * pow(1e-3, d_r_));
                        dNh_dt[i][j][k] -= melt_rate / (c_h_ * pow(1e-3, d_h_));
                    }
                }
            }
        }
    }
}

/**
 * @brief Computes the sedimentation for the Milbrandt scheme.
 */
void MilbrandtScheme::compute_sedimentation(
    const Field3D& qr,
    const Field3D& qs,
    const Field3D& qg,
    const Field3D& qh,
    const Field3D& Nr,
    const Field3D& Ns,
    const Field3D& Ng,
    const Field3D& Nh,
    Field3D& dqr_dt,
    Field3D& dqs_dt,
    Field3D& dqg_dt,
    Field3D& dqh_dt,
    Field3D& dNr_dt,
    Field3D& dNs_dt,
    Field3D& dNg_dt,
    Field3D& dNh_dt
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
                float Nr_val = static_cast<float>(Nr[i][j][k]);
                if (qr_val > 0.0f && k > 0) 
                {
                    double Vt = sedimentation_rate(qr_val, Nr_val, alpha_r_, c_r_, d_r_, a_r_, b_r_, dz_local);
                    dqr_dt[i][j][k] -= Vt;
                    dqr_dt[i][j][k - 1] += Vt;
                    dNr_dt[i][j][k] -= Vt / (c_r_ * pow(1e-3, d_r_));
                    dNr_dt[i][j][k - 1] += Vt / (c_r_ * pow(1e-3, d_r_));
                }

                float qs_val = static_cast<float>(qs[i][j][k]);
                float Ns_val = static_cast<float>(Ns[i][j][k]);
                if (qs_val > 0.0f && k > 0) 
                {
                    double Vt = sedimentation_rate(qs_val, Ns_val, alpha_s_, c_s_, d_s_, a_s_, b_s_, dz_local);
                    dqs_dt[i][j][k] -= Vt;
                    dqs_dt[i][j][k - 1] += Vt;
                    dNs_dt[i][j][k] -= Vt / (c_s_ * pow(1e-3, d_s_));
                    dNs_dt[i][j][k - 1] += Vt / (c_s_ * pow(1e-3, d_s_));
                }

                float qg_val = static_cast<float>(qg[i][j][k]);
                float Ng_val = static_cast<float>(Ng[i][j][k]);
                if (qg_val > 0.0f && k > 0) 
                {
                    double Vt = sedimentation_rate(qg_val, Ng_val, alpha_g_, c_g_, d_g_, a_g_, b_g_, dz_local);
                    dqg_dt[i][j][k] -= Vt;
                    dqg_dt[i][j][k - 1] += Vt;
                    dNg_dt[i][j][k] -= Vt / (c_g_ * pow(1e-3, d_g_));
                    dNg_dt[i][j][k - 1] += Vt / (c_g_ * pow(1e-3, d_g_));
                }

                float qh_val = static_cast<float>(qh[i][j][k]);
                float Nh_val = static_cast<float>(Nh[i][j][k]);
                if (qh_val > 0.0f && k > 0 && hail_processes_) 
                {
                    double Vt = sedimentation_rate(qh_val, Nh_val, alpha_h_, c_h_, d_h_, a_h_, b_h_, dz_local);
                    dqh_dt[i][j][k] -= Vt;
                    dqh_dt[i][j][k - 1] += Vt;
                    dNh_dt[i][j][k] -= Vt / (c_h_ * pow(1e-3, d_h_));
                    dNh_dt[i][j][k - 1] += Vt / (c_h_ * pow(1e-3, d_h_));
                }
            }
        }
    }
}

/**
 * @brief Computes the radar reflectivity for the Milbrandt scheme.
 */
void MilbrandtScheme::compute_radar_reflectivity(
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

    if (Nr_.size_r() != NR || Nr_.size_th() != NTH || Nr_.size_z() != NZ ||
        Ni_.size_r() != NR || Ni_.size_th() != NTH || Ni_.size_z() != NZ ||
        Ns_.size_r() != NR || Ns_.size_th() != NTH || Ns_.size_z() != NZ ||
        Ng_.size_r() != NR || Ng_.size_th() != NTH || Ng_.size_z() != NZ ||
        Nh_.size_r() != NR || Nh_.size_th() != NTH || Nh_.size_z() != NZ)
    {
        initialize_prognostic_fields(NR, NTH, NZ);
    }

    reflectivity_dbz.resize(NR, NTH, NZ, 0.0f);
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
                float Z_total = 0.0f;

                float qr_val = static_cast<float>(qr[i][j][k]);
                float Nr_val = static_cast<float>(Nr_[i][j][k]);
                if (qr_val > 0.0f && Nr_val > 0.0f) 
                {
                    Z_total += calculate_reflectivity(qr_val, Nr_val, alpha_r_, c_r_, d_r_);
                }

                float qi_val = static_cast<float>(qi[i][j][k]);
                float Ni_val = static_cast<float>(Ni_[i][j][k]);
                if (qi_val > 0.0f && Ni_val > 0.0f) 
                {
                    Z_total += calculate_reflectivity(qi_val, Ni_val, alpha_i_, c_i_, d_i_);
                }

                float qs_val = static_cast<float>(qs[i][j][k]);
                float Ns_val = static_cast<float>(Ns_[i][j][k]);
                if (qs_val > 0.0f && Ns_val > 0.0f) 
                {
                    Z_total += calculate_reflectivity(qs_val, Ns_val, alpha_s_, c_s_, d_s_);
                }

                float qg_val = static_cast<float>(qg[i][j][k]);
                float Ng_val = static_cast<float>(Ng_[i][j][k]);
                if (qg_val > 0.0f && Ng_val > 0.0f) 
                {
                    Z_total += calculate_reflectivity(qg_val, Ng_val, alpha_g_, c_g_, d_g_);
                }

                float qh_val = static_cast<float>(qh[i][j][k]);
                float Nh_val = static_cast<float>(Nh_[i][j][k]);
                if (qh_val > 0.0f && Nh_val > 0.0f && hail_processes_) 
                {
                    Z_total += calculate_reflectivity(qh_val, Nh_val, alpha_h_, c_h_, d_h_);
                }

                if (!std::isfinite(static_cast<double>(Z_total)))
                {
                    Z_total = Z_min;
                }
                Z_total = std::clamp(Z_total, Z_min, Z_max);
                float Z_dbz = 10.0f * std::log10(Z_total);
                if (!std::isfinite(static_cast<double>(Z_dbz)))
                {
                    Z_dbz = Z_dbz_min;
                }
                reflectivity_dbz[i][j][k] = std::clamp(Z_dbz, Z_dbz_min, Z_dbz_max);
            }
        }
    }
}


/**
 * @brief Computes the autoconversion rate for the Milbrandt scheme.
 */
double MilbrandtScheme::autoconversion_rate(double qc, double rho) 
{
    return c_auto_ * std::max(qc - qc0_, 0.0);
}

/**
 * @brief Computes the accretion rate for the Milbrandt scheme.
 */
double MilbrandtScheme::accretion_rate(double q1, double q2, double Nt1, double Nt2,
                                     double alpha1, double alpha2, double c1, double d1,
                                     double c2, double d2, double rho) 
{
    return 2.2 * q1 * q2;
}

/**
 * @brief Computes the rain self-collection rate for the Milbrandt scheme.
 */
double MilbrandtScheme::rain_selfcollection_rate(double Nr, double rho) 
{
    return 5.78e-4 * Nr * Nr / std::max(rho, 1.0e-6);
}

/**
 * @brief Computes the sedimentation rate for the Milbrandt scheme.
 */
double MilbrandtScheme::sedimentation_rate(
    double q,
    double Nt,
    double alpha,
    double c,
    double d,
    double a_term,
    double b_term,
    double dz
) 
{
    double lambda = calculate_lambda(q, Nt, alpha, c, d);
    double Vt_mean = a_term * pow(lambda, -b_term);
    return Vt_mean * q / std::max(dz, 1.0e-6);
}

/**
 * @brief Computes the gamma function for the Milbrandt scheme.
 */
double MilbrandtScheme::gamma_function(double x) 
{
    if (std::abs(x - 1.0) < 1e-6) return 1.0;
    if (std::abs(x - 2.0) < 1e-6) return 1.0;
    if (std::abs(x - 3.0) < 1e-6) return 2.0;
    if (std::abs(x - 4.0) < 1e-6) return 6.0;
    if (std::abs(x - 5.0) < 1e-6) return 24.0;
    if (std::abs(x - 6.0) < 1e-6) return 120.0;
    return std::tgamma(x);
}

/**
 * @brief Computes the incomplete gamma function for the Milbrandt scheme.
 */
double MilbrandtScheme::incomplete_gamma(double a, double x) 
{
    return gamma_function(a) * std::exp(-x);
}
