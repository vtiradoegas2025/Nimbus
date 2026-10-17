#pragma once

#include "numerics/advection/advection_base.hpp"

namespace tmv
{

class WENO5Scheme final : public AdvectionSchemeBase
{
public:
    WENO5Scheme() = default;

    std::string name() const override { return "weno5"; }
    std::unique_ptr<AdvectionSchemeBase> clone() const override
    {
        return std::make_unique<WENO5Scheme>(*this);
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

private:
    static double reconstruct_left(const std::vector<double>& q, int i);
    static double reconstruct_right(const std::vector<double>& q, int i);
    static double rusanov_flux(double q_left, double q_right, double velocity);
    static void advect_column(
        const std::vector<double>& q,
        const std::vector<double>& velocity,
        const std::vector<double>& dz,
        double dt,
        bool positivity,
        std::vector<double>& dqdt);
    static void apply_positivity(
        const std::vector<double>& q,
        std::vector<double>& dqdt,
        double dt,
        double minimum = 0.0);
};

} // namespace tmv

using tmv::WENO5Scheme;
