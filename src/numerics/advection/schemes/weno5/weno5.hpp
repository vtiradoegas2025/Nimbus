/**
 * @file weno5.hpp
 * @brief Declarations for the numerics module.
 *
 * Defines interfaces, data structures, and contracts used by
 * the numerics runtime and scheme implementations.
 * This file is part of the src/numerics subsystem.
 */

#pragma once
#include "numerics/advection_base.hpp"

/**
 * @brief Implements the WENO5 advection scheme.
 */
class WENO5Scheme : public AdvectionSchemeBase 
{
private:
    AdvectionConfig config_;
    double weno_epsilon_;

public:
    /**
     * @brief Constructs the WENO5 advection scheme.
     */
    WENO5Scheme();

    std::string name() const override { return "weno5"; }

    /**
     * @brief Initializes with default WENO5 configuration.
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
    double suggest_dt(
        const AdvectionConfig& cfg,
        const AdvectionStateView& state) override;

private:
    /**
     * @brief Reconstructs left-biased interface value with WENO5.
     */
    double weno5_reconstruct_left(const std::vector<double>& q, int i);
    /**
     * @brief Reconstructs right-biased interface value with WENO5.
     */
    double weno5_reconstruct_right(const std::vector<double>& q, int i);

    /**
     * @brief Computes smoothness indicator over a WENO stencil.
     */
    double smoothness_indicator(const std::vector<double>& q, int start, int end);

    /**
     * @brief Computes first candidate polynomial at interface.
     */
    double q_tilde_0(const std::vector<double>& q, int i);

    /**
     * @brief Computes second candidate polynomial at interface.
     */
    double q_tilde_1(const std::vector<double>& q, int i);
    
    /**
     * @brief Computes third candidate polynomial at interface.
     */
    double q_tilde_2(const std::vector<double>& q, int i);


    static constexpr double d0 = 1.0/10.0;
    static constexpr double d1 = 6.0/10.0;
    static constexpr double d2 = 3.0/10.0;

    /**
     * @brief Evaluates Rusanov numerical flux for scalar transport.
     */
    double rusanov_flux(double q_left, double q_right, double velocity);

    /**
     * @brief Computes 1D advection tendency using WENO5 reconstruction.
     */
    void weno5_advect_1d(
        const std::vector<double>& q,
        const std::vector<double>& velocity,
        const std::vector<double>& dx,
        double dt,
        std::vector<double>& dqdt);

    /**
     * @brief Applies positivity preservation to scalar tendency updates.
     */
    void apply_positivity_preservation(
        const std::vector<double>& q,
        std::vector<double>& dqdt,
        double dt,
        double min_value = 0.0);
};
