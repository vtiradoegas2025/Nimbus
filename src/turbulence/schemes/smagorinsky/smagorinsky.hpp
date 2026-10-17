/**
 * @file smagorinsky.hpp
 * @brief Declarations for the turbulence module.
 *
 * Defines interfaces, data structures, and contracts used by
 * the turbulence runtime and scheme implementations.
 * This file is part of the src/turbulence subsystem.
 */

#pragma once
#include "turbulence/base/eddy_viscosity.hpp"
#include "turbulence/turbulence_base.hpp"
#include <vector>


namespace tmv
{

/**
 * @brief Smagorinsky-Lilly turbulence closure with diagnostic eddy viscosity.
 */
class SmagorinskyScheme : public TurbulenceSchemeBase {
private:
    double Cs_;
    double Pr_t_;
    double Sc_t_;
    double nu_t_max_;
    double K_max_;

    Field3D nu_t_;
    Field3D K_theta_;
    Field3D K_q_;
    // Flat-grid filter width is horizontally invariant. This compact profile
    // avoids repeating sqrt/cbrt at every (i,j) while mapped terrain retains
    // its fully local physical-volume evaluation.
    std::vector<double> filter_width_profile_;

    // True while compute() replays itself on the CPU path for the parity
    // harness (GPU.md Phase 1); suppresses the GPU dispatch so
    // the replay measures the CPU reference.
    bool smag_parity_replay_ = false;
    bool nu_t_host_current_ = false;

public:
    std::unique_ptr<TurbulenceSchemeBase> clone() const override
    {
        return std::make_unique<SmagorinskyScheme>(*this);
    }

    /**
 * @brief Initializes the Smagorinsky turbulence scheme.
 */
    SmagorinskyScheme();

    /**
 * @brief Gets the name of the Smagorinsky turbulence scheme.
 */
    std::string name() const override { return "smagorinsky"; }

    /**
 * @brief Gets the required fields.
 */
    int required_fields() const override;

    /**
 * @brief Initializes the Smagorinsky turbulence scheme.
 */
    void initialize(const TurbulenceConfig& cfg) override;

    void compute(
        const TurbulenceConfig& cfg,
        const GridMetrics& grid,
        const TurbulenceStateView& state,
        TurbulenceTendencies& tend,
        TurbulenceDiagnostics* diag_opt = nullptr) override;

    const Field3D* realized_eddy_viscosity() const noexcept override
    {
        return nu_t_host_current_ ? &nu_t_ : nullptr;
    }

    bool mutable_realized_transport_fields(
        TurbulenceTransportFields& fields) noexcept override;

    bool rebuild_from_realized_transport(
        const TurbulenceConfig& cfg,
        const GridMetrics& grid,
        const TurbulenceStateView& state,
        TurbulenceTendencies& tend) override;

    bool supports_spatial_generation_invalidation() const noexcept override
    {
        return true;
    }

    void invalidate_spatial_generation() noexcept override
    {
        nu_t_host_current_ = false;
    }

private:

    /**
 * @brief Computes the eddy coefficients.
 */

    void compute_eddy_coefficients(
        const TurbulenceConfig& cfg,
        const GridMetrics& grid,
        const TurbulenceStateView& state
    );

    void compute_dynamic(
        const TurbulenceConfig& cfg,
        const GridMetrics& grid,
        const TurbulenceStateView& state,
        TurbulenceTendencies& tendencies,
        TurbulenceDiagnostics* diagnostics);

    /**
 * @brief Applies the stability correction.
 */
    double apply_stability_correction(
        const TurbulenceConfig& cfg,
        const GridMetrics& grid,
        const TurbulenceStateView& state,
        int i, int j, int k
    );
};

} // namespace tmv

using tmv::SmagorinskyScheme;
