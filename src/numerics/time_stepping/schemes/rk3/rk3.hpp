#pragma once
#include "numerics/time_stepping/time_stepping_base.hpp"

namespace tmv
{

// Three-stage, third-order strong-stability-preserving Runge-Kutta.
class RK3Scheme : public TimeSteppingSchemeBase
{
private:
    RHSFunction rhs_function_;

    std::vector<NumericalState> stage1_;
    std::vector<NumericalState> stage2_;
    std::vector<NumericalTendencies> rhs_stage1_;
    std::vector<NumericalTendencies> rhs_stage2_;

public:
    RK3Scheme();

    std::string name() const override { return "rk3"; }
    std::unique_ptr<TimeSteppingSchemeBase> clone() const override
    {
        return std::make_unique<RK3Scheme>(*this);
    }
    void initialize() override;
    void initialize(const TimeSteppingConfig& cfg, RHSFunction rhs_func) override;

    void step(
        const TimeSteppingConfig& cfg,
        TimeSteppingState& state,
        TimeSteppingDiagnostics* diag_opt = nullptr) override;

    double suggest_dt(
        const TimeSteppingConfig& cfg,
        const TimeSteppingState& state) override;

    bool supports_coupled_unsplit_rk() const noexcept override { return true; }
    void step_coupled_unsplit_rk(
        const TimeSteppingConfig& cfg,
        double dt,
        double time,
        const UnsplitRkCallbacks& callbacks,
        TimeSteppingDiagnostics* diagnostics = nullptr) override;

    bool supports_domain_recycling_invalidation() const noexcept override
    {
        return true;
    }
    void invalidate_generation_state_after_domain_recycling() noexcept override;

private:
    void compute_stage1(const TimeSteppingState& state, std::vector<NumericalTendencies>& tendencies);

    void compute_stage2(const TimeSteppingState& state, std::vector<NumericalTendencies>& tendencies);

    void compute_final(TimeSteppingState& state, std::vector<NumericalTendencies>& tendencies);

    void add_states(const std::vector<NumericalState>& state1, double coef1,
                   const std::vector<NumericalTendencies>& tend, double coef2, double dt,
                   std::vector<NumericalState>& result);
};

} // namespace tmv

using tmv::RK3Scheme;
