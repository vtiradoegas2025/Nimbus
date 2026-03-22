/**
 * @file explicit.hpp
 * @brief Declarations for the numerics module.
 *
 * Defines interfaces, data structures, and contracts used by
 * the numerics runtime and scheme implementations.
 * This file is part of the src/numerics subsystem.
 */

#pragma once
#include "numerics/diffusion_base.hpp"

/**
 * @brief Implements the explicit diffusion scheme.
 */
class ExplicitDiffusionScheme : public DiffusionSchemeBase 
{
private:
    DiffusionConfig config_;

public:
    ExplicitDiffusionScheme();

    std::string name() const override { return "explicit"; }
    void initialize() override;
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
 * @brief Computes the scalar diffusion.
 */
    void compute_scalar_diffusion(
        const Field3D& field,
        const Field3D* K_field,
        double K_default,
        const GridMetrics& grid,
        Field3D& tendency);

    /**
 * @brief Computes the momentum diffusion.
 */
    void compute_momentum_diffusion(
        const Field3D& u,
        const Field3D& v,
        const Field3D& w,
        const Field3D* nu_t,
        double nu_default,
        const GridMetrics& grid,
        Field3D& du_dt,
        Field3D& dv_dt,
        Field3D& dw_dt);
};
