/**
 * @file implicit.hpp
 * @brief Declarations for the numerics module.
 *
 * Defines interfaces, data structures, and contracts used by
 * the numerics runtime and scheme implementations.
 * This file is part of the src/numerics subsystem.
 */

#pragma once
#include "numerics/diffusion_base.hpp"

/**
 * @brief Implicit vertical diffusion scheme for scalar and momentum fields.
 */
class ImplicitDiffusionScheme : public DiffusionSchemeBase {
private:
    DiffusionConfig config_;

    /**
 * @brief Solves the tridiagonal system.
 */
    void solve_tridiagonal(
        const std::vector<double>& a,
        const std::vector<double>& b,
        const std::vector<double>& c,
        const std::vector<double>& rhs,
        std::vector<double>& x);

public:
    /**
     * @brief Constructs the implicit diffusion scheme.
     */
    ImplicitDiffusionScheme();

    std::string name() const override { return "implicit"; }
    /**
     * @brief Initializes with default diffusion configuration.
     */
    void initialize() override;
    /**
     * @brief Initializes from explicit diffusion configuration.
     */
    void initialize(const DiffusionConfig& cfg) override;

    /**
 * @brief Computes the diffusion tendencies.
 */
    void compute_diffusion_tendencies(
        const DiffusionConfig& cfg,
        const DiffusionStateView& state,
        DiffusionTendencies& tendencies,
        DiffusionDiagnostics* diag_opt = nullptr) override;

    /**
 * @brief Checks the stability of the diffusion scheme.
 */
    double check_stability(
        const DiffusionConfig& cfg,
        const DiffusionStateView& state) override;

private:

    /**
 * @brief Computes the vertical diffusion.
 */
    void implicit_vertical_diffusion(
        const Field3D& field,
        const Field3D* K_field,
        double K_default,
        const GridMetrics& grid,
        double dt,
        Field3D& field_new);

    /**
 * @brief Computes the vertical momentum diffusion.
 */
    void implicit_vertical_momentum_diffusion(
        const Field3D& u,
        const Field3D& v,
        const Field3D& w,
        const Field3D* nu_t,
        double nu_default,
        const GridMetrics& grid,
        double dt,
        Field3D& u_new,
        Field3D& v_new,
        Field3D& w_new);
};
