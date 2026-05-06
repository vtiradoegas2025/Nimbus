/**
 * @file milbrandt.hpp
 * @brief Declarations for the microphysics module.
 *
 * Defines interfaces, data structures, and contracts used by
 * the microphysics runtime and scheme implementations.
 * This file is part of the src/microphysics subsystem.
 */

#pragma once
#include "microphysics/base/thermodynamics.hpp"
#include "microphysics/microphysics_base.hpp"
#include "core/field/field3d.hpp"


/**
 * @brief Multi-moment Milbrandt-Yau style bulk microphysics scheme.
 */
class MilbrandtScheme : public MicrophysicsScheme 
{
private:
    double qc0_;
    double c_auto_;

    double alpha_r_;
    double alpha_i_;
    double alpha_s_;
    double alpha_g_;
    double alpha_h_;

    double c_r_, d_r_;
    double c_i_, d_i_;
    double c_s_, d_s_;
    double c_g_, d_g_;
    double c_h_, d_h_;

    double a_r_, b_r_;
    double a_i_, b_i_;
    double a_s_, b_s_;
    double a_g_, b_g_;
    double a_h_, b_h_;

    bool triple_moment_;
    bool hail_processes_;

    Field3D Nr_;
    Field3D Ni_;
    Field3D Ns_;
    Field3D Ng_;
    Field3D Nh_;

    Field3D Zr_;
    Field3D Zi_;
    Field3D Zs_;
    Field3D Zg_;
    Field3D Zh_;

    Field3D dNr_dt_, dNi_dt_, dNs_dt_, dNg_dt_, dNh_dt_;
    Field3D dZr_dt_, dZi_dt_, dZs_dt_, dZg_dt_, dZh_dt_;

public:
    /**
 * @brief Initializes the Milbrandt scheme with default parameters.
 */
    MilbrandtScheme(
        double qc0 = 1.0e-3,
        double c_auto = 1.0e-3,
        double alpha_r = 2.5, double alpha_i = 0.0, double alpha_s = 0.0,
        double alpha_g = 1.0, double alpha_h = 1.0,
        double c_r = 524.0, double d_r = 3.0,
        double c_i = 110.8, double d_i = 2.93,
        double c_s = 2.77e-2, double d_s = 2.0,
        double c_g = 24.0, double d_g = 2.8,
        double c_h = 114.0, double d_h = 2.8,
        double a_r = 65.0, double b_r = 0.125,
        double a_i = 40.0, double b_i = 0.37,
        double a_s = 2.77, double b_s = 0.1,
        double a_g = 114.0, double b_g = 0.5,
        double a_h = 114.0, double b_h = 0.5,
        bool triple_moment = false,
        bool hail_processes = true
    );

    /**
 * @brief Computes the tendencies for the Milbrandt scheme.
 */
    void compute_tendencies(
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
    ) override;

    /**
 * @brief Computes the radar reflectivity for the Milbrandt scheme.
 */
    void compute_radar_reflectivity(
        const Field3D& qc,
        const Field3D& qr,
        const Field3D& qi,
        const Field3D& qs,
        const Field3D& qg,
        const Field3D& qh,
        Field3D& reflectivity_dbz
    ) override;

/**
 * @brief Returns the name of the Milbrandt scheme.
 */
    std::string get_scheme_name() const override { return "milbrandt"; }
    int get_num_prognostic_vars() const override 
    {
        int vars = 7;
        vars += 5;
        if (triple_moment_) vars += 5;
        return vars;
    }

/**
 * @brief Returns the number of prognostic variables for the Milbrandt scheme.
 */
private:
    /**
     * @brief Computes generalized-gamma slope parameter for a species.
     */
    double calculate_lambda(double q, double Nt, double alpha, double c, double d);

    /**
     * @brief Computes intercept parameter from moments and slope.
     */
    double calculate_N0(double Nt, double alpha, double lambda);

    /**
     * @brief Computes requested distribution moment order.
     */
    double calculate_moment(double q, double Nt, double alpha, double c, double d, int moment_order);

    /**
     * @brief Computes linear reflectivity contribution from moments.
     */
    double calculate_reflectivity(double q, double Nt, double alpha, double c, double d);
    
    /**
     * @brief Computes warm-rain autoconversion source term.
     */
    double autoconversion_rate(double qc, double rho);

    /**
     * @brief Computes accretion transfer rate between two species.
     */
    double accretion_rate(double q1, double q2, double Nt1, double Nt2, double alpha1, double alpha2,
                         double c1, double d1, double c2, double d2, double rho);

    /**
     * @brief Computes sedimentation tendency from terminal fall speed.
     */
    double sedimentation_rate(
        double q,
        double Nt,
        double alpha,
        double c,
        double d,
        double a_term,
        double b_term,
        double dz
    );

    /**
     * @brief Computes rain self-collection tendency on number concentration.
     */
    double rain_selfcollection_rate(double Nr, double rho);

    /**
     * @brief Computes ice collection efficiency for pairwise interactions.
     */
    double ice_collection_efficiency(double T_celsius, double D1, double D2);


    /**
     * @brief Allocates and seeds prognostic number-concentration fields.
     */
    void initialize_prognostic_fields(int NR, int NTH, int NZ);

/**
 * @brief Computes the warm rain processes for the Milbrandt scheme.
 */
    void compute_warm_rain_processes(
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
    );

/**
 * @brief Computes the ice processes for the Milbrandt scheme.
 */
    void compute_ice_processes(
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
    );

/**
 * @brief Computes the melting processes for the Milbrandt scheme.
 */
    void compute_melting_processes(
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
    );

/**
 * @brief Computes the sedimentation for the Milbrandt scheme.
 */
    void compute_sedimentation(
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
    );

    /**
     * @brief Evaluates gamma function for integer and non-integer arguments.
     */
    double gamma_function(double x);
    
    /**
     * @brief Evaluates a simplified incomplete gamma approximation.
     */
    double incomplete_gamma(double a, double x);
};
