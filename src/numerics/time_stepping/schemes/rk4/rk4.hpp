/**
 * @file rk4.hpp
 * @brief Declarations for the numerics module.
 *
 * Defines interfaces, data structures, and contracts used by
 * the numerics runtime and scheme implementations.
 * This file is part of the src/numerics subsystem.
 */

#pragma once
#include "numerics/time_stepping/time_stepping_base.hpp"

/**
 * @brief Classical fourth-order Runge-Kutta time stepping scheme.
 */
class RK4Scheme : public TimeSteppingSchemeBase {
private:
    TimeSteppingConfig config_;
    RHSFunction rhs_function_;

    std::vector<NumericalState> stage1_;
    std::vector<NumericalState> stage2_;
    std::vector<NumericalState> stage3_;
    std::vector<NumericalTendencies> k1_;
    std::vector<NumericalTendencies> k2_;
    std::vector<NumericalTendencies> k3_;
    std::vector<NumericalTendencies> k4_;

public:
    /**
     * @brief Constructs the RK4 time stepping scheme.
     */
    RK4Scheme();

    /**
 * @brief Returns the name of the RK4 time stepping scheme.
 */
    std::string name() const override { return "rk4"; }
    /**
     * @brief Initializes with default RK4 configuration.
     */
    void initialize() override;
    /**
     * @brief Initializes with runtime config and RHS callback.
     */
    void initialize(const TimeSteppingConfig& cfg, RHSFunction rhs_func) override;

    /**
 * @brief Steps the RK4 time stepping scheme.
 */
    void step(
        const TimeSteppingConfig& cfg,
        TimeSteppingState& state,
        TimeSteppingDiagnostics* diag_opt = nullptr) override;

    /**
     * @brief Suggests a stable timestep using configured CFL constraints.
     */
    double suggest_dt(
        const TimeSteppingConfig& cfg,
        const TimeSteppingState& state) override;

private:
    /**
     * @brief Computes first RK4 stage tendency.
     */
    void compute_k1(const TimeSteppingState& state);
    /**
     * @brief Computes second RK4 stage tendency.
     */
    void compute_k2(const TimeSteppingState& state);
    /**
     * @brief Computes third RK4 stage tendency.
     */
    void compute_k3(const TimeSteppingState& state);
    /**
     * @brief Computes fourth RK4 stage tendency.
     */
    void compute_k4(const TimeSteppingState& state);
    /**
     * @brief Applies weighted RK4 combination to update state.
     */
    void update_final(TimeSteppingState& state);

    /**
     * @brief Copies state vectors between stage buffers.
     */
    void copy_state(const std::vector<NumericalState>& src, std::vector<NumericalState>& dst);
    /**
     * @brief Forms linear combination of state and tendency vectors.
     */
    void add_states(const std::vector<NumericalState>& state1, double coef1,
                   const std::vector<NumericalTendencies>& tend, double coef2, double dt,
                   std::vector<NumericalState>& result);
};
