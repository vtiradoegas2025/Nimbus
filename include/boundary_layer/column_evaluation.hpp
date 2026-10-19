#pragma once

#include "boundary_layer/boundary_layer_base.hpp"

namespace tmv
{

struct BoundaryLayerStoredSurfaceDiagnostics
{
    float ustar              = 0.0f;
    float tau_u              = 0.0f;
    float tau_v              = 0.0f;
    float sensible_heat_flux = 0.0f;
    float latent_heat_flux   = 0.0f;
    float moisture_flux      = 0.0f;
    float pbl_height         = 0.0f;
};

void validate_prepared_boundary_layer_column(const BoundaryLayerColumnStateView& column,
                                             double                              horizontal_cell_area_m2);

BoundaryLayerStoredSurfaceDiagnostics store_prepared_boundary_layer_result(BoundaryLayerColumnStateView& column,
                                                                           double horizontal_cell_area_m2,
                                                                           const BoundaryLayerTendencies&  tendencies,
                                                                           const BoundaryLayerDiagnostics& diagnostics);

/**
 * Evaluate and validate one reconstructed PBL column, then convert its
 * specific tendencies into Nimbus's exact FP32 publication and conserved
 * force/inventory-rate currencies. The caller supplies reusable workspace,
 * tendencies, and diagnostics so normal cadence execution allocates nothing
 * per column after warm-up.
 */
BoundaryLayerStoredSurfaceDiagnostics evaluate_prepared_boundary_layer_column(BoundaryLayerSchemeBase&      scheme,
                                                                              const BoundaryLayerConfig&    config,
                                                                              const SurfaceConfig&          surface,
                                                                              BoundaryLayerColumnStateView& column,
                                                                              double horizontal_cell_area_m2,
                                                                              BoundaryLayerTendencies&  tendencies,
                                                                              BoundaryLayerDiagnostics& diagnostics);

const char* boundary_layer_prepared_column_algorithm_id() noexcept;

} // namespace tmv
