/**
 * @file split_explicit.hpp
 * @brief Split-explicit acoustic time stepping scheme (Klemp & Wilhelmson 1978).
 *
 * Separates slow (advective + buoyancy) and fast (acoustic) tendencies.
 * The slow tendencies are applied once per large time step. The fast
 * tendencies are subcycled N times with forward-backward integration:
 *   1. FORWARD:  divergence → update p, ρ  (pressure/density respond to flow)
 *   2. BACKWARD: pressure gradient using NEW p, ρ → update u, v, w
 *
 * The forward-backward ordering is semi-implicit for acoustics and stable
 * for acoustic CFL ≤ 1 per substep.
 *
 * This file is part of the src/numerics subsystem.
 */

#pragma once
#include "numerics/time_stepping/time_stepping_base.hpp"
#include "core/field3d.hpp"

class SplitExplicitScheme : public TimeSteppingSchemeBase
{
public:
    SplitExplicitScheme();

    std::string name() const override { return "split_explicit"; }
    void initialize() override;
    void initialize(const TimeSteppingConfig& cfg, RHSFunction rhs_func) override;

    void step(const TimeSteppingConfig& cfg,
              TimeSteppingState& state,
              TimeSteppingDiagnostics* diag_opt = nullptr) override;

    double suggest_dt(const TimeSteppingConfig& cfg,
                      const TimeSteppingState& state) override;

    // Split-explicit interface.
    bool supports_split_acoustic() const override { return true; }
    void step_split_acoustic(
        const TimeSteppingConfig& cfg,
        double dt_large,
        const SplitExplicitCallbacks& callbacks) override;

private:
    TimeSteppingConfig config_;

    /// Resolve N: use config value if > 0, otherwise auto-compute.
    static int resolve_substep_count(const TimeSteppingConfig& cfg, double dt_large);

    // Persistent fast-tendency buffers (allocated once, reused each step).
    Field3D du_fast_buf_, dv_fast_buf_, dw_fast_buf_;
    Field3D drho_fast_buf_, dp_fast_buf_;
    bool buffers_allocated_ = false;

    void ensure_buffers(int nr, int nth, int nz);
};
