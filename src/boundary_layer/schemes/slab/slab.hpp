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

namespace tmv
{

/**
 * @brief Implements the slab boundary layer scheme.
 */
class SlabScheme : public BoundaryLayerSchemeBase 
{
private:
    double min_h_ = 100.0;
    double max_h_ = 2000.0;

public:
    std::unique_ptr<BoundaryLayerSchemeBase> clone() const override
    {
        return std::make_unique<SlabScheme>(*this);
    }

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

    bool supports_native_column_batch(const BoundaryLayerConfig&,const BoundaryLayerComputeKernels&)const override;
    bool compute_columns_native(const BoundaryLayerConfig&,const SurfaceConfig&,const BoundaryLayerColumnStateView*,std::size_t,
        BoundaryLayerTendencies*,BoundaryLayerDiagnostics*,BoundaryLayerComputeKernels&)override;
private:
    /**
     * @brief Diagnoses boundary-layer depth using bulk stability.
     */
    double diagnose_pbl_height(
        const BoundaryLayerColumnStateView& col,
        const BoundaryLayerConfig& cfg
    );

    /**
     * @brief Synchronizes diagnostic mixed-layer state from the resolved column.
     */
    void update_slab_state(
        SlabColumnState& state,
        const BoundaryLayerColumnStateView& col,
        const BoundaryLayerConfig& cfg
    );

    /**
     * @brief Applies mixed-layer relaxation tendencies to the column.
     */
    void apply_slab_tendencies(
        const SlabColumnState& state,
        const BoundaryLayerColumnStateView& col,
        const surface_fluxes::BulkFluxes& fluxes,
        BoundaryLayerTendencies& tend,
        const BoundaryLayerConfig& cfg
    );
};

} // namespace tmv

using tmv::SlabScheme;
