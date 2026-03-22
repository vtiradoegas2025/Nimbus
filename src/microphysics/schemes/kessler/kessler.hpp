/**
 * @file kessler.hpp
 * @brief Declarations for the microphysics module.
 *
 * Defines interfaces, data structures, and contracts used by
 * the microphysics runtime and scheme implementations.
 * This file is part of the src/microphysics subsystem.
 */

#pragma once
#include "microphysics/base/thermodynamics.hpp"
#include "physics/microphysics_base.hpp"

/**
 * @brief Implements the Kessler scheme for microphysics.
 */
class KesslerScheme : public MicrophysicsScheme 
{
private:
    double qc0_;
    double c_auto_;
    double c_accr_;
    double c_evap_;

    double c_freeze_;
    double c_rime_;
    double c_melt_;
    double c_subl_;

    double a_term_;
    double b_term_;
    double Vt_max_;
    double a_hail_;
    double b_hail_;
    double a_grau_;
    double b_grau_;
    double Vt_max_hail_;
    double Vt_max_grau_;

public:
    KesslerScheme(
        double qc0 = 1.0e-3,
        double c_auto = 1.0e-3,
        double c_accr = 2.2,
        double c_evap = 3.0e-3,
        double c_freeze = 1.0e-3,
        double c_rime = 1.0,
        double c_melt = 1.0e-3,
        double c_subl = 1.0e-3,
        double a_term = 65.0,
        double b_term = 0.125,
        double Vt_max = 20.0,
        double a_hail = 114.0,
        double b_hail = 0.5,
        double a_grau = 40.0,
        double b_grau = 0.37,
        double Vt_max_hail = 40.0,
        double Vt_max_grau = 15.0
    );

    /**
 * @brief Computes the tendencies for the Kessler scheme.
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
 * @brief Computes the radar reflectivity for the Kessler scheme.
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

    std::string get_scheme_name() const override { return "kessler"; }
    int get_num_prognostic_vars() const override { return 5; }

private:
    
    /**
 * @brief Computes the warm rain processes for the Kessler scheme.
 */
    void compute_warm_rain_processes(
        const Field3D& temperature,
        const Field3D& qv,
        const Field3D& qc,
        const Field3D& qr,
        Field3D& dqc_dt,
        Field3D& dqr_dt,
        Field3D& dqv_dt,
        Field3D& dtheta_dt
    );

    /**
 * @brief Computes the ice processes for the Kessler scheme.
 */
    void compute_ice_processes(
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
    );

    /**
 * @brief Computes the melting processes for the Kessler scheme.
 */
    void compute_melting_processes(
        const Field3D& temperature,
        const Field3D& qg,
        const Field3D& qh,
        Field3D& dqg_dt,
        Field3D& dqh_dt,
        Field3D& dqr_dt,
        Field3D& dtheta_dt
    );

    /**
 * @brief Computes the sedimentation for the Kessler scheme.
 */    
    void compute_sedimentation(
        const Field3D& qr,
        const Field3D& qg,
        const Field3D& qh,
        Field3D& dqr_dt,
        Field3D& dqg_dt,
        Field3D& dqh_dt
    );
};
