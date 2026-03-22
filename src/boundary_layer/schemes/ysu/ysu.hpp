/**
 * @file ysu.hpp
 * @brief Declarations for the boundary_layer module.
 *
 * Defines interfaces, data structures, and contracts used by
 * the boundary_layer runtime and scheme implementations.
 * This file is part of the src/boundary_layer subsystem.
 */

#pragma once
#include "boundary_layer/base/surface_fluxes.hpp"
#include "physics/boundary_layer_base.hpp"

/**
 * @brief Implements the YSU boundary layer scheme.
 */
class YSUScheme : public BoundaryLayerSchemeBase {
private:
    double pblfac_ = 1.0;
    double cn_ = 0.75;
    double ck_ = 0.1;
    double ce_ = 0.5;
    double c0_ = 0.15;

    std::vector<double> K_m_;
    std::vector<double> K_h_;

public:
    /**
     * @brief Constructs the YSU boundary-layer scheme.
     */
    YSUScheme();

    std::string name() const override { return "ysu"; }
    /**
     * @brief Returns required state-field mask for this scheme.
     */
    int required_fields() const override;

    /**
     * @brief Initializes scheme parameters from runtime configuration.
     */
    void initialize(const BoundaryLayerConfig& cfg) override;

    /**
     * @brief Computes one-column YSU turbulence tendencies.
     */
    void compute_column(
        const BoundaryLayerConfig& cfg,
        const SurfaceConfig& sfc,
        const BoundaryLayerColumnStateView& col,
        BoundaryLayerTendencies& tend,
        BoundaryLayerDiagnostics* diag_opt = nullptr) override;

private:
    /**
     * @brief Diagnoses boundary-layer depth from bulk Richardson criteria.
     */
    double diagnose_pbl_height(
        const BoundaryLayerColumnStateView& col,
        const BoundaryLayerConfig& cfg
    );

    /**
     * @brief Computes eddy-diffusivity vertical profiles.
     */
    void compute_k_profile(
        const BoundaryLayerColumnStateView& col,
        double h, double ustar,
        std::vector<double>& K_m,
        std::vector<double>& K_h
    );

    /**
     * @brief Computes nonlocal transport terms for scalars.
     */
    void compute_nonlocal_transport(
        const BoundaryLayerColumnStateView& col,
        const surface_fluxes::BulkFluxes& fluxes,
        double h,
        std::vector<double>& nonlocal_theta,
        std::vector<double>& nonlocal_qv
    );

    /**
     * @brief Applies vertical diffusion tendencies to column fields.
     */
    void apply_diffusion_tendencies(
        const BoundaryLayerColumnStateView& col,
        const std::vector<double>& K_m,
        const std::vector<double>& K_h,
        BoundaryLayerTendencies& tend
    );
};
