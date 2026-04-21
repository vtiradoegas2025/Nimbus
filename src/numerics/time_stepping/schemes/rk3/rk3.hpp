/**
 * @file rk3.hpp
 * @brief Declarations for the numerics module.
 *
 * Defines interfaces, data structures, and contracts used by
 * the numerics runtime and scheme implementations.
 * This file is part of the src/numerics subsystem.
 */

#pragma once
#include "numerics/time_stepping/time_stepping_base.hpp"

/**
 * @brief Third-order Runge-Kutta time stepping scheme.
 */
class RK3Scheme : public TimeSteppingSchemeBase {
private:
    TimeSteppingConfig config_;
    RHSFunction rhs_function_;

    std::vector<NumericalState> stage1_;
    std::vector<NumericalState> stage2_;
    std::vector<NumericalTendencies> rhs_stage1_;
    std::vector<NumericalTendencies> rhs_stage2_;

public:
    /**
     * @brief Constructs the RK3 time stepping scheme.
     */
    RK3Scheme();

    /**
 * @brief Returns the name of the RK3 time stepping scheme.
 */
    std::string name() const override { return "rk3"; }
    /**
     * @brief Initializes with default RK3 configuration.
     */
    void initialize() override;
    /**
     * @brief Initializes with runtime config and RHS callback.
     */
    void initialize(const TimeSteppingConfig& cfg, RHSFunction rhs_func) override;

    /**
 * @brief Steps the RK3 time stepping scheme.
 */
    void step(
        const TimeSteppingConfig& cfg,
        TimeSteppingState& state,
        TimeSteppingDiagnostics* diag_opt = nullptr) override;

    /**
 * @brief Suggests the time step.
 */
    double suggest_dt(
        const TimeSteppingConfig& cfg,
        const TimeSteppingState& state) override;

private:
    /**
 * @brief Computes the first stage.
 */
    void compute_stage1(const TimeSteppingState& state, std::vector<NumericalTendencies>& tendencies);

    /**
 * @brief Computes the second stage.
 */
    void compute_stage2(const TimeSteppingState& state, std::vector<NumericalTendencies>& tendencies);

    /**
     * @brief Computes the final RK3 stage and combines updates.
     */
    void compute_final(TimeSteppingState& state, std::vector<NumericalTendencies>& tendencies);

    /**
 * @brief Copies the state.
 */
    void copy_state(const std::vector<NumericalState>& src, std::vector<NumericalState>& dst);

    /**
 * @brief Adds the states.
 */
    void add_states(const std::vector<NumericalState>& state1, double coef1,
                   const std::vector<NumericalTendencies>& tend, double coef2, double dt,
                   std::vector<NumericalState>& result);
    /**
     * @brief Applies tendency increments to the active state vector.
     */
    void update_state(std::vector<NumericalState>& state,
                     const std::vector<NumericalTendencies>& tendencies, double dt);
};
