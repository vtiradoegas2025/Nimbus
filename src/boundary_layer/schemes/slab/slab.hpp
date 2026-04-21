/**
 * @file slab.hpp
 * @brief Declarations for the boundary_layer module.
 *
 * Defines interfaces, data structures, and contracts used by
 * the boundary_layer runtime and scheme implementations.
 * This file is part of the src/boundary_layer subsystem.
 */

#pragma once
#include "boundary_layer/base/surface_fluxes.hpp"
#include "boundary_layer/boundary_layer_base.hpp"

/**
 * @brief Implements the slab boundary layer scheme.
 */
class SlabScheme : public BoundaryLayerSchemeBase 
{
private:
    double h_ = 500.0;
    double theta_m_ = 300.0;
    double qv_m_ = 0.01;

    double entrainment_coeff_ = 0.2;
    double min_h_ = 100.0;
    double max_h_ = 2000.0;

public:
    /**
     * @brief Constructs the slab mixed-layer scheme.
     */
    SlabScheme();

    std::string name() const override { return "slab"; }
    /**
     * @brief Returns required state-field mask for this scheme.
     */
    int required_fields() const override;

    /**
     * @brief Initializes slab-state parameters from configuration.
     */
    void initialize(const BoundaryLayerConfig& cfg) override;

    /**
     * @brief Computes one-column slab boundary-layer tendencies.
     */
    void compute_column(
        const BoundaryLayerConfig& cfg,
        const SurfaceConfig& sfc,
        const BoundaryLayerColumnStateView& col,
        BoundaryLayerTendencies& tend,
        BoundaryLayerDiagnostics* diag_opt = nullptr) override;

private:
    /**
     * @brief Diagnoses boundary-layer depth using bulk stability.
     */
    double diagnose_pbl_height(
        const BoundaryLayerColumnStateView& col,
        const BoundaryLayerConfig& cfg
    );

    /**
     * @brief Updates mixed-layer state variables from surface forcing.
     */
    void update_slab_state(
        const BoundaryLayerColumnStateView& col,
        const SurfaceConfig& sfc,
        const surface_fluxes::BulkFluxes& fluxes,
        double dt
    );

    /**
     * @brief Applies mixed-layer relaxation tendencies to the column.
     */
    void apply_slab_tendencies(
        const BoundaryLayerColumnStateView& col,
        BoundaryLayerTendencies& tend,
        const BoundaryLayerConfig& cfg
    );
};
