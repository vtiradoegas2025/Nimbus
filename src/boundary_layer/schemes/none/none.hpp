#pragma once

#include "boundary_layer/boundary_layer_base.hpp"

namespace tmv::boundary_layer {

/**
 * @brief No-op boundary-layer scheme.
 *
 * Returns zero tendencies for all prognostic variables (u, v, theta, qv,
 * tke). Used by the cgrid_hydrostatic integration test where PBL must
 * be fully out of the loop so the test exercises only dynamics. The
 * existing `apply_surface_fluxes=false` knob only disables the surface
 * flux contribution inside the slab scheme; it does not bypass the
 * vertical-mixing operator that fires every step on whatever state is
 * present (and that vertical mixing was identified during the smell #6
 * audit as one source of cgrid_hydrostatic theta+qv drift).
 *
 * Wrapped in `tmv::boundary_layer` to avoid colliding with
 * `::NoneScheme` in the terrain/chaos modules.
 */
class NoneScheme : public BoundaryLayerSchemeBase
{
public:
    std::unique_ptr<BoundaryLayerSchemeBase> clone() const override
    {
        return std::make_unique<NoneScheme>(*this);
    }

    std::string name() const override { return "none"; }

    int required_fields() const override { return 0; }

    void initialize(const BoundaryLayerConfig& /*cfg*/) override {}

    void compute_column(const BoundaryLayerConfig& cfg,
                        const SurfaceConfig& sfc,
                        const BoundaryLayerColumnStateView& col,
                        BoundaryLayerTendencies& tend,
                        BoundaryLayerDiagnostics* diag_opt = nullptr) override;
};

}  // namespace tmv::boundary_layer

#ifndef NIMBUS_BOUNDARY_LAYER_NAMESPACE_ALIAS
#define NIMBUS_BOUNDARY_LAYER_NAMESPACE_ALIAS
namespace boundary_layer = tmv::boundary_layer;
#endif
