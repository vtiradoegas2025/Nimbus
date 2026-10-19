#pragma once

#include "numerics/advection/advection_base.hpp"

namespace tmv
{

class NoAdvectionScheme final : public AdvectionSchemeBase
{
public:
    std::string name() const override { return "none"; }
    std::unique_ptr<AdvectionSchemeBase> clone() const override
    {
        return std::make_unique<NoAdvectionScheme>(*this);
    }

    void initialize() override;
    void initialize(const AdvectionConfig& cfg) override;
    void compute_flux_divergence(
        const AdvectionConfig& cfg,
        const AdvectionStateView& state,
        AdvectionTendencies& tendencies,
        AdvectionDiagnostics* diag_opt = nullptr) override;
    double suggest_dt(
        const AdvectionConfig& cfg,
        const AdvectionStateView& state) override;
};

} // namespace tmv
