#pragma once

#include "radiation/radiation_base.hpp"

namespace tmv::radiation
{

/**
 * @brief No-op radiation scheme.
 *
 * Returns zero radiative heating tendencies. Used by the cgrid_hydrostatic
 * integration test where radiation must be fully out of the loop so the
 * test exercises only dynamics. Also useful for dry-dynamics smoke runs,
 * CFD-style benchmarks, and any scenario where the existing `do_lw=false`
 * / `do_sw=false` knobs are insufficient (those flags only zero individual
 * band contributions inside simple_grey; they do not bypass the column
 * loop or guarantee bit-exact zero tendencies).
 *
 * Wrapped in `namespace radiation` to avoid colliding with `::NoneScheme`
 * in the terrain/chaos modules.
 */
class NoneScheme : public RadiationSchemeBase
{
public:
    std::unique_ptr<RadiationSchemeBase> clone() const override
    {
        return std::make_unique<NoneScheme>(*this);
    }

    std::string name() const override { return "none"; }

    int required_fields() const override { return 0; }

    void initialize(const RadiationConfig& /*cfg*/) override {}

    void compute_column(const RadiationConfig& cfg,
                        const RadiationColumnStateView& col,
                        RadiationColumnTendencies& tend,
                        RadiationColumnFluxes* fluxes_opt = nullptr) override;
};

}  // namespace tmv::radiation

namespace radiation = tmv::radiation;
