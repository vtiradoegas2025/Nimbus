/**
 * @file initial_conditions.hpp
 * @brief Declarations for the chaos module.
 *
 * Defines interfaces, data structures, and contracts used by
 * the chaos runtime and scheme implementations.
 * This file is part of the src/chaos subsystem.
 */

#pragma once
#include "chaos/scheme.hpp"

namespace tmv::chaos
{

/**
 * @brief Initial conditions perturbation scheme
 *
 * Applies stochastic perturbations to initial conditions (IC) only.
 * Represents uncertainty in the initial storm environment without ongoing
 * model error forcing. Based on AMS ensemble practices.
 */
class InitialConditionsScheme : public ChaosScheme {
public:
    InitialConditionsScheme() = default;
    InitialConditionsScheme(const InitialConditionsScheme&) = default;
    std::unique_ptr<ChaosScheme> clone() const override
    {
        return std::make_unique<InitialConditionsScheme>(*this);
    }

    std::string name() const override { return "initial_conditions"; }
    void configure_compute_backend(ComputeBackend* backend, bool required) noexcept override;

    void initialize(const ChaosConfig& cfg, const GridMetrics& grid) override;

    void apply_initial_conditions(
        const ChaosConfig& cfg,
        const GridMetrics& grid,
        SimulationState& state,
        ChaosDiagnostics* diag = nullptr
    ) override;

    void apply_tendencies(
        const ChaosConfig& cfg,
        const GridMetrics& grid,
        const ChaosStateView& state_view,
        ChaosTendencies& tendencies,
        ChaosDiagnostics* diag = nullptr
    ) override;

    void step_noise(
        const ChaosConfig& cfg,
        const GridMetrics& grid,
        double dt
    ) override;

private:
    ComputeBackend* native_backend_ = nullptr;
    bool native_generation_required_ = false;

};

}
