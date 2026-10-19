#pragma once

#include "turbulence/turbulence_base.hpp"

namespace tmv::turbulence {

/**
 * @brief No-op turbulence scheme.
 *
 * Returns zero subgrid-scale tendencies for all prognostic variables
 * (u, v, w, theta, qv, tke). Used by the cgrid_hydrostatic integration
 * test where SGS must be fully out of the loop so the test exercises
 * only dynamics. The `Cs=0` YAML knob does NOT fully disable
 * Smagorinsky -- the dispatcher still computes strain-rate tensors and
 * writes diffusion tendencies sized by other coefficients; the smell #6
 * audit confirmed that `Cs=0` is not equivalent to "no turbulence".
 *
 * Wrapped in `tmv::turbulence` to avoid colliding with
 * `::NoneScheme` in the terrain/chaos modules.
 */
class NoneScheme : public TurbulenceSchemeBase
{
public:
    std::unique_ptr<TurbulenceSchemeBase> clone() const override
    {
        return std::make_unique<NoneScheme>(*this);
    }

    std::string name() const override { return "none"; }

    int required_fields() const override { return 0; }

    void initialize(const TurbulenceConfig& /*cfg*/) override {}

    void compute(const TurbulenceConfig& cfg,
                 const GridMetrics& grid,
                 const TurbulenceStateView& state,
                 TurbulenceTendencies& tend,
                 TurbulenceDiagnostics* diag_opt = nullptr) override;
};

}  // namespace tmv::turbulence

#ifndef NIMBUS_TURBULENCE_NAMESPACE_ALIAS
#define NIMBUS_TURBULENCE_NAMESPACE_ALIAS
namespace turbulence = tmv::turbulence;
#endif
