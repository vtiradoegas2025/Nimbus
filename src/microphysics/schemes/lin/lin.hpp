/**
 * @file lin.hpp
 * @brief Declarations for the microphysics module.
 *
 * Defines interfaces, data structures, and contracts used by
 * the microphysics runtime and scheme implementations.
 * This file is part of the src/microphysics subsystem.
 */

#pragma once
#include <array>
#include "microphysics/base/thermodynamics.hpp"
#include "microphysics/microphysics_base.hpp"
#include "core/field/field3d.hpp"

namespace tmv
{

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
    double c_melt_s_;
    double c_melt_g_;
    double c_melt_h_;
    double c_subl_;


    // True while compute_tendencies replays itself on the CPU path for the
    // parity harness (GPU.md Phase 1); suppresses the GPU
    // dispatch so the replay measures the CPU reference.
    bool lin_parity_replay_ = false;
    bool lin_native_staging_ = false;
    std::array<Field3D, 8> native_tendency_staging_;

public:

    std::unique_ptr<MicrophysicsScheme> clone() const override
    {
        // Native output staging is disposable scratch, not rollback state.
        auto copy = std::make_unique<LinScheme>(
            qc0_, c_auto_, c_accr_, c_evap_, c_ihom_, c_rime_, c_agg_,
            c_melt_s_, c_melt_g_, c_melt_h_, c_subl_);
        static_cast<MicrophysicsScheme&>(*copy) = *this;
        return copy;
    }

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
        double c_melt_s = microphysics_constants::c_melt_snow_per_K,
        double c_melt_g = microphysics_constants::c_melt_graupel_per_K,
        double c_melt_h = microphysics_constants::c_melt_hail_per_K,
        double c_subl = 1.0e-3
    );

    void configure(const MicrophysicsConfig& config) override;
    double explicit_terminal_velocity_mps(
        std::size_t species, double mixing_ratio, double dry_density) const override;
    PortableSchemeState portable_state() const override;
    MicrophysicsParameterIdentity parameter_identity() const noexcept override;
    bool restore_portable_state(const PortableSchemeState& state,
                                std::string& error) override;

    /**
 * @brief Computes the tendencies for the Lin scheme.
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
 * @brief Computes the radar reflectivity for the Lin scheme.
 */
    double bottom_sedimentation_sink_rate(
        const Field3D& rho_d,
        const Field3D& qr, const Field3D& qs,
        const Field3D& qg, const Field3D& qh,
        int i, int j, double dt_micro) const override;

    bool requires_runtime_column_donor_admission() const noexcept override
    {
        return true;
    }

    std::string get_scheme_name() const override { return "lin"; }
    int get_num_prognostic_vars() const override { return 7; }

private:

    /**
 * @brief Computes the sedimentation for the Lin scheme.
 */
    void compute_sedimentation(
        const Field3D& rho_d,
        const Field3D& qr,
        const Field3D& qs,
        const Field3D& qg,
        const Field3D& qh,
        Field3D& dqr_dt,
        Field3D& dqs_dt,
        Field3D& dqg_dt,
        Field3D& dqh_dt,
        double dt_micro
    );

    /**
     * @brief Returns collection efficiency for mixed-phase interactions.
     */
    double collection_efficiency(double T_celsius);
};

} // namespace tmv

using tmv::LinScheme;
