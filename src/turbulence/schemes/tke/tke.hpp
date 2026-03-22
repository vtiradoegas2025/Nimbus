/**
 * @file tke.hpp
 * @brief Declarations for the turbulence module.
 *
 * Defines interfaces, data structures, and contracts used by
 * the turbulence runtime and scheme implementations.
 * This file is part of the src/turbulence subsystem.
 */

#pragma once
#include "turbulence/base/eddy_viscosity.hpp"
#include "physics/turbulence_base.hpp"

/**
 * @brief 1.5-order TKE turbulence closure with prognostic SGS energy.
 */
class TKEScheme : public TurbulenceSchemeBase 
{
private:
    double Ce1_ = 1.0;
    double Ce2_ = 1.33;
    double c_k_ = 0.1;
    double c_eps_ = 0.19;
    double Pr_t_ = 0.7;
    double Sc_t_ = 0.7;

    double c_l_ = 0.15;
    double l_max_ = 500.0;

    Field3D tke_;

    Field3D nu_t_;
    Field3D K_theta_;
    Field3D K_q_;
    Field3D K_tke_;

public:
    /**
     * @brief Constructs the TKE turbulence scheme.
     */
    TKEScheme();

    std::string name() const override { return "tke"; }
    /**
     * @brief Returns required state-field mask for this scheme.
     */
    int required_fields() const override;

    /**
 * @brief Initializes the TKE turbulence scheme.
 */
    void initialize(const TurbulenceConfig& cfg) override;

    /**
 * @brief Computes the TKE turbulence scheme.
 */
    void compute(
        const TurbulenceConfig& cfg,
        const GridMetrics& grid,
        const TurbulenceStateView& state,
        TurbulenceTendencies& tend,
        TurbulenceDiagnostics* diag_opt = nullptr) override;

private:
    
    /**
 * @brief Computes the mixing length.
 */
    double compute_mixing_length(
        const TurbulenceConfig& cfg,
        const GridMetrics& grid,
        double e,
        double N,
        int i, int j, int k
    );

    /**
 * @brief Computes the eddy coefficients from the TKE.
 */
    void compute_eddy_coefficients_from_tke(
        const TurbulenceConfig& cfg,
        const GridMetrics& grid,
        const TurbulenceStateView& state
    );

    /**
 * @brief Updates the TKE prognostic equation.
 */
    void update_tke_prognostic(
        const TurbulenceConfig& cfg,
        const GridMetrics& grid,
        const TurbulenceStateView& state,
        Field3D& dtke_dt
    );

    /**
 * @brief Computes the TKE production.
 */
    void compute_tke_production(
        const TurbulenceStateView& state,
        const GridMetrics& grid,
        const Field3D& nu_t,
        Field3D& shear_prod,
        Field3D& buoy_prod
    );
};
