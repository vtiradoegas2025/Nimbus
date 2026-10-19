// Klemp-Wilhelmson split-explicit stepping separates slow advective/buoyancy
// tendencies from fast acoustic modes. Each acoustic substep updates p/rho
// from divergence before momentum reads the new pressure gradient. This
// forward-backward ordering is stable for acoustic CFL <= 1.

#pragma once
#include "numerics/time_stepping/time_stepping_base.hpp"
#include "core/field/field3d.hpp"

namespace tmv
{

class SplitExplicitScheme : public TimeSteppingSchemeBase
{
public:
    SplitExplicitScheme();

    std::string name() const override { return "split_explicit"; }
    std::unique_ptr<TimeSteppingSchemeBase> clone() const override
    {
        return std::make_unique<SplitExplicitScheme>(*this);
    }
    void initialize() override;
    void initialize(const TimeSteppingConfig& cfg, RHSFunction rhs_func) override;

    void step(const TimeSteppingConfig& cfg,
              TimeSteppingState& state,
              TimeSteppingDiagnostics* diag_opt = nullptr) override;

    double suggest_dt(const TimeSteppingConfig& cfg,
                      const TimeSteppingState& state) override;

    // Split-explicit interface.
    bool supports_split_acoustic() const override { return true; }
    bool supports_domain_recycling_invalidation() const noexcept override
    {
        return true;
    }
    void invalidate_generation_state_after_domain_recycling() noexcept override;
    void step_split_acoustic(
        const TimeSteppingConfig& cfg,
        double dt_large,
        const SplitExplicitCallbacks& callbacks) override;

    int resolve_substep_count(const TimeSteppingConfig& cfg, double dt_large) override;

private:
    TimeSteppingConfig config_;

    // Avoid repeating the same promotion warning on every model step.
    int last_warned_requested_substeps_ = -1;
    int last_warned_effective_substeps_ = -1;

    // Persistent fast-tendency buffers (allocated once, reused each step).
    Field3D du_fast_buf_, dv_fast_buf_, dw_fast_buf_;
    Field3D drho_fast_buf_, dp_fast_buf_;
    bool buffers_allocated_ = false;

    void ensure_buffers(int nr, int nth, int nz);
};

} // namespace tmv

using tmv::SplitExplicitScheme;
