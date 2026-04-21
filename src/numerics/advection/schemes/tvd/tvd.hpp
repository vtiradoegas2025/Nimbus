/**
 * @file tvd.hpp
 * @brief Declarations for the numerics module.
 *
 * Defines interfaces, data structures, and contracts used by
 * the numerics runtime and scheme implementations.
 * This file is part of the src/numerics subsystem.
 */

#pragma once
#include "numerics/advection/advection_base.hpp"

/**
 * @brief Implements the TVD advection scheme.
 */
class TVDScheme : public AdvectionSchemeBase 
{
private:
    AdvectionConfig config_;
    double (*limiter_function_)(double) = nullptr;


    /**
     * @brief Minmod slope limiter.
     */
    static double minmod_limiter(double r);

    /**
     * @brief Van Leer slope limiter.
     */
    static double vanleer_limiter(double r);

    /**
     * @brief Superbee slope limiter.
     */
    static double superbee_limiter(double r);

    /**
     * @brief Monotonized-central slope limiter.
     */
    static double mc_limiter(double r);

    /**
     * @brief Universal limiter dispatch helper.
     */
    static double universal_limiter(double r);

public:
    /**
     * @brief Constructs the TVD advection scheme.
     */
    TVDScheme();

    std::string name() const override { return "tvd"; }

    /**
     * @brief Initializes with default TVD configuration.
     */
    void initialize() override;

    /**
     * @brief Initializes from explicit advection configuration.
     */
    void initialize(const AdvectionConfig& cfg) override;

 /**
 * @brief Computes the flux divergence.
 */
    void compute_flux_divergence(
        const AdvectionConfig& cfg,
        const AdvectionStateView& state,
        AdvectionTendencies& tendencies,
        AdvectionDiagnostics* diag_opt = nullptr) override;

 /**
 * @brief Suggests the time step.
 */
    double suggest_dt(const AdvectionConfig& cfg, const AdvectionStateView& state) override;

private:
    /**
     * @brief Reconstructs face states with MUSCL and selected limiter.
     */
    void muscl_reconstruct(
        const std::vector<double>& q,
        std::vector<double>& q_left,
        std::vector<double>& q_right,
        double (*limiter)(double));

    /**
     * @brief Evaluates numerical face flux for one scalar pair.
     */
    double numerical_flux(double q_left, double q_right, double velocity);
    

    /**
     * @brief Computes 1D advection tendency using reconstructed fluxes.
     */
    void advect_1d(
        const std::vector<double>& q,
        const std::vector<double>& velocity,
        const std::vector<double>& dx,
        double dt,
        std::vector<double>& dqdt);

    /**
     * @brief Enforces lower-bound positivity for scalar tendencies.
     */
    void apply_positivity_limiter(
        const std::vector<double>& q,
        std::vector<double>& dqdt,
        double dt,
        double min_value = 0.0);
};
