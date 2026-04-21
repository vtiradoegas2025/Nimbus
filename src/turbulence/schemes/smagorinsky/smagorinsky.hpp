/**
 * @file smagorinsky.hpp
 * @brief Declarations for the turbulence module.
 *
 * Defines interfaces, data structures, and contracts used by
 * the turbulence runtime and scheme implementations.
 * This file is part of the src/turbulence subsystem.
 */

#pragma once
#include "turbulence/base/eddy_viscosity.hpp"
#include "turbulence/turbulence_base.hpp"


/**
 * @brief Smagorinsky-Lilly turbulence closure with diagnostic eddy viscosity.
 */
class SmagorinskyScheme : public TurbulenceSchemeBase {
private:
    double Cs_;
    double Pr_t_;
    double Sc_t_;
    double nu_t_max_;
    double K_max_;

    Field3D nu_t_;
    Field3D K_theta_;
    Field3D K_q_;

public:
    /**
 * @brief Initializes the Smagorinsky turbulence scheme.
 */
    SmagorinskyScheme();

    /**
 * @brief Gets the name of the Smagorinsky turbulence scheme.
 */
    std::string name() const override { return "smagorinsky"; }

    /**
 * @brief Gets the required fields.
 */
    int required_fields() const override;

    /**
 * @brief Initializes the Smagorinsky turbulence scheme.
 */
    void initialize(const TurbulenceConfig& cfg) override;

    void compute(
        const TurbulenceConfig& cfg,
        const GridMetrics& grid,
        const TurbulenceStateView& state,
        TurbulenceTendencies& tend,
        TurbulenceDiagnostics* diag_opt = nullptr) override;

private:

    /**
 * @brief Computes the eddy coefficients.
 */
   
    void compute_eddy_coefficients(
        const TurbulenceConfig& cfg,
        const GridMetrics& grid,
        const TurbulenceStateView& state
    );

    /**
 * @brief Applies the stability correction.
 */
    double apply_stability_correction(
        const TurbulenceConfig& cfg,
        const GridMetrics& grid,
        const TurbulenceStateView& state,
        int i, int j, int k
    );
};
