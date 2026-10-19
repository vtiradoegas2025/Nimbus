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
#include "microphysics/microphysics_base.hpp"

namespace tmv
{

/**
 * @brief Three-species Kessler warm-rain microphysics (qv, qc, qr).
 *
 * Kessler is deliberately liquid-only. Frozen hydrometeors are accepted by
 * the common scheme interface but are never consumed, produced, diagnosed,
 * or sedimented by this implementation.
 */
class KesslerScheme : public MicrophysicsScheme 
{
private:
    double qc0_;
    double c_auto_;
    double c_accr_;
    double evaporation_efficiency_;

    double Vt_max_;

public:
    std::unique_ptr<MicrophysicsScheme> clone() const override
    {
        return std::make_unique<KesslerScheme>(*this);
    }

    KesslerScheme(
        double qc0 = 1.0e-3,
        double c_auto = 1.0e-3,
        double c_accr = 2.2,
        double evaporation_efficiency = 1.0,
        double Vt_max = 20.0
    );

    void configure(const MicrophysicsConfig& config) override;
    double explicit_terminal_velocity_mps(
        std::size_t species, double mixing_ratio, double dry_density) const override;
    PortableSchemeState portable_state() const override;
    MicrophysicsParameterIdentity parameter_identity() const noexcept override;
    bool restore_portable_state(const PortableSchemeState& state,
                                std::string& error) override;

    /**
 * @brief Computes the tendencies for the Kessler scheme.
 */
    void compute_tendencies(
        const Field3D& p,
        const Field3D& theta,
        const Field3D& rho_d,
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
    double bottom_sedimentation_sink_rate(
        const Field3D& rho_d,
        const Field3D& qr, const Field3D& qs,
        const Field3D& qg, const Field3D& qh,
        int i, int j, double dt_micro) const override;

    std::string get_scheme_name() const override { return "kessler"; }
    bool requires_runtime_column_donor_admission() const noexcept override
    {
        return true;
    }
    int get_num_prognostic_vars() const override { return 3; }

private:
    
    /**
 * @brief Computes the warm rain processes for the Kessler scheme.
 */
    void compute_warm_rain_processes(
        const Field3D& temperature,
        const Field3D& p,
        const Field3D& rho_d,
        const Field3D& qv,
        const Field3D& qc,
        const Field3D& qr,
        Field3D& dqc_dt,
        Field3D& dqr_dt,
        Field3D& dqv_dt,
        Field3D& dtheta_dt,
        double dt_micro
    );

    /**
 * @brief Computes the sedimentation for the Kessler scheme.
 */    
    void compute_sedimentation(
        const Field3D& rho_d,
        const Field3D& qr,
        Field3D& dqr_dt,
        double dt_micro);
};

} // namespace tmv

using tmv::KesslerScheme;
