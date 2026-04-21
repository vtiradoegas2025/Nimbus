/**
 * @file lin.hpp
 * @brief Declarations for the microphysics module.
 *
 * Defines interfaces, data structures, and contracts used by
 * the microphysics runtime and scheme implementations.
 * This file is part of the src/microphysics subsystem.
 */

#pragma once
#include "microphysics/base/thermodynamics.hpp"
#include "microphysics/microphysics_base.hpp"
#include "core/field3d.hpp"

/**
 * @brief Single-moment Lin microphysics parameterization.
 */

class LinScheme : public MicrophysicsScheme 
{
private:
    double qc0_;
    double c_auto_;
    double c_accr_;
    double c_evap_;

    double c_ihom_;
    double c_rime_;
    double c_agg_;
    double c_melt_;
    double c_subl_;

    double N0r_;
    double N0s_;
    double N0g_;
    double N0h_;

    double a_r_, b_r_;
    double a_s_, b_s_;
    double a_g_, b_g_;
    double a_h_, b_h_;

public:

    /**
 * @brief Initializes the Lin scheme with default parameters.
 */
    LinScheme(
        double qc0 = 1.0e-3,
        double c_auto = 1.0e-3,
        double c_accr = 2.2,
        double c_evap = 1.0e-3,
        double c_ihom = 1.0e-3,
        double c_rime = 1.0,
        double c_agg = 1.0e-3,
        double c_melt = 1.0e-3,
        double c_subl = 1.0e-3,
        double N0r = 8.0e6,
        double N0s = 3.0e6,
        double N0g = 4.0e5,
        double N0h = 1.0e4,
        double a_r = 65.0, double b_r = 0.125,
        double a_s = 40.0, double b_s = 0.37,
        double a_g = 114.0, double b_g = 0.5,
        double a_h = 114.0, double b_h = 0.5
    );

    /**
 * @brief Computes the tendencies for the Lin scheme.
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
 * @brief Computes the radar reflectivity for the Lin scheme.
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

    std::string get_scheme_name() const override { return "lin"; }
    int get_num_prognostic_vars() const override { return 7; }

private:

    /**
 * @brief Computes the saturation adjustment for the Lin scheme.
 */
    void saturation_adjustment(
        const Field3D& temperature,
        const Field3D& p,
        Field3D& qv,
        Field3D& qc
    );

    /**
 * @brief Computes the warm rain processes for the Lin scheme.
 */
    void compute_warm_rain_processes(
        const Field3D& temperature,
        const Field3D& p,
        const Field3D& qv,
        const Field3D& qc,
        const Field3D& qr,
        Field3D& dqc_dt,
        Field3D& dqr_dt,
        Field3D& dqv_dt,
        Field3D& dtheta_dt
    );

    /**
 * @brief Computes the ice processes for the Lin scheme.
 */
    void compute_ice_processes(
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
    );

    /**
 * @brief Computes the melting processes for the Lin scheme.
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
        Field3D& dtheta_dt
    );

    /**
 * @brief Computes the sedimentation for the Lin scheme.
 */
    void compute_sedimentation(
        const Field3D& qr,
        const Field3D& qs,
        const Field3D& qg,
        const Field3D& qh,
        Field3D& dqr_dt,
        Field3D& dqs_dt,
        Field3D& dqg_dt,
        Field3D& dqh_dt
    );

    /**
     * @brief Estimates ice number concentration from mass and temperature.
     */
    double ice_number_concentration(double qi, double T_celsius);
    /**
     * @brief Returns collection efficiency for mixed-phase interactions.
     */
    double collection_efficiency(double T_celsius);
    /**
     * @brief Returns ice-threshold value used by aggregation logic.
     */
    double ice_threshold_temperature(double qi);
};
