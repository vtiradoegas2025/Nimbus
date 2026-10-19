#pragma once
#include "numerics/time_stepping/time_stepping_base.hpp"

namespace tmv
{

// Classical four-stage, fourth-order Runge-Kutta.
class RK4Scheme : public TimeSteppingSchemeBase
{
private:
    RHSFunction rhs_function_;

    std::vector<NumericalState> stage1_;
    std::vector<NumericalState> stage2_;
    std::vector<NumericalState> stage3_;
    std::vector<NumericalTendencies> k1_;
    std::vector<NumericalTendencies> k2_;
    std::vector<NumericalTendencies> k3_;
    std::vector<NumericalTendencies> k4_;

public:
    RK4Scheme();

    std::string name() const override { return "rk4"; }
    std::unique_ptr<TimeSteppingSchemeBase> clone() const override
    {
        return std::make_unique<RK4Scheme>(*this);
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
    void compute_k1(const TimeSteppingState& state);
    void compute_k2(const TimeSteppingState& state);
    void compute_k3(const TimeSteppingState& state);
    void compute_k4(const TimeSteppingState& state);
    void update_final(TimeSteppingState& state);

    void add_states(const std::vector<NumericalState>& state1, double coef1,
                   const std::vector<NumericalTendencies>& tend, double coef2, double dt,
                   std::vector<NumericalState>& result);
};

} // namespace tmv

using tmv::RK4Scheme;
