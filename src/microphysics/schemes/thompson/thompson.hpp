/**
 * @file thompson.hpp
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
 * @brief Thompson bulk microphysics parameterization with prognostic ice number.
 */
class ThompsonScheme : public MicrophysicsScheme 
{
private:
    double qc0_;
    double c_auto_;

    double ccn_conc_;
    double in_conc_;

    double a_r_, b_r_;
    double a_s_, b_s_;
    double a_g_, b_g_;
    double a_h_, b_h_;

    Field3D Ni_;

    Field3D dNi_dt_;

public:
/**
 * @brief Initializes the Thompson scheme with default parameters.
 */
    ThompsonScheme(
        double qc0 = 1.0e-3,
        double c_auto = 1.0e-3,
        double ccn_conc = 100e6,
        double in_conc = 100e3,
        double a_r = 65.0, double b_r = 0.125,
        double a_s = 40.0, double b_s = 0.37,
        double a_g = 114.0, double b_g = 0.5,
        double a_h = 114.0, double b_h = 0.5
    );

/**
 * @brief Computes the tendencies for the Thompson scheme.
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
 * @brief Computes the radar reflectivity for the Thompson scheme.
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

    std::string get_scheme_name() const override { return "thompson"; }
    int get_num_prognostic_vars() const override { return 8; }

private:

/**
 * @brief Performs the saturation adjustment for the Thompson scheme.
 */
    void saturation_adjustment(
        const Field3D& temperature,
        const Field3D& p,
        Field3D& qv,
        Field3D& qc
    );

/**
 * @brief Computes the warm rain processes for the Thompson scheme.
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
 * @brief Computes the ice processes for the Thompson scheme.
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
        Field3D& dtheta_dt,
        Field3D& dNi_dt
    );

    /**
 * @brief Computes the melting processes for the Thompson scheme.
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
 * @brief Computes the sedimentation for the Thompson scheme.
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
     * @brief Estimates freezing probability from temperature and cloud water.
     */
    double freezing_probability(double T, double qc_vol);
    
    /**
     * @brief Computes ice-size threshold used by Thompson process switches.
     */
    double ice_size_threshold(double Ni);

    /**
     * @brief Computes collection efficiency for mixed-phase interactions.
     */
    double collection_efficiency(double T_celsius, double qi, double qr);
};
