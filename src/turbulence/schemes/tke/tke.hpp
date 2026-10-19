#pragma once

#include "turbulence/base/eddy_viscosity.hpp"
#include "turbulence/turbulence_base.hpp"

namespace tmv
{

class TKEScheme final : public TurbulenceSchemeBase
{
public:
    TKEScheme() = default;

    std::unique_ptr<TurbulenceSchemeBase> clone() const override;
    std::string name() const override { return "tke"; }
    int required_fields() const override;
    void initialize(const TurbulenceConfig& cfg) override;
    void compute(const TurbulenceConfig& cfg,
                 const GridMetrics& grid,
                 const TurbulenceStateView& state,
                 TurbulenceTendencies& tendencies,
                 TurbulenceDiagnostics* diagnostics = nullptr) override;

    const Field3D* realized_eddy_viscosity() const noexcept override;
    bool mutable_realized_transport_fields(
        TurbulenceTransportFields& fields) noexcept override;
    bool rebuild_from_realized_transport(
        const TurbulenceConfig& cfg,
        const GridMetrics& grid,
        const TurbulenceStateView& state,
        TurbulenceTendencies& tendencies) override;

private:
    bool compute_native(const TurbulenceConfig&, const GridMetrics&,
                        const TurbulenceStateView&, TurbulenceTendencies&,
                        TurbulenceDiagnostics*);
    static constexpr double momentum_coefficient_ = 0.1;
    static constexpr double stable_length_coefficient_ = 0.76;
    static constexpr double dissipation_intercept_ = 0.19;
    static constexpr double dissipation_slope_ = 0.51;

    double mixing_length(const TurbulenceConfig& cfg,
                         const GridMetrics& grid,
                         double tke,
                         double buoyancy_frequency,
                         int i, int j, int k) const;
    double dissipation(const TurbulenceConfig& cfg,
                       const GridMetrics& grid,
                       const TurbulenceStateView& state,
                       double tke,
                       int i, int j, int k) const;
    void build_coefficients(const TurbulenceConfig& cfg,
                            const GridMetrics& grid,
                            const TurbulenceStateView& state,
                            const Field3D& tke,
                            Field3D& eddy_viscosity,
                            Field3D& theta_diffusivity,
                            Field3D& moisture_diffusivity,
                            Field3D& tke_diffusivity,
                            Field3D* effective_prandtl) const;
    void build_tke_tendency(const TurbulenceConfig& cfg,
                            const GridMetrics& grid,
                            const TurbulenceStateView& state,
                            const Field3D& tke,
                            const Field3D& eddy_viscosity,
                            const Field3D& theta_diffusivity,
                            const Field3D& tke_diffusivity,
                            Field3D& tendency) const;
    void build_production(const TurbulenceConfig& cfg,
                          const TurbulenceStateView& state,
                          const GridMetrics& grid,
                          const Field3D& eddy_viscosity,
                          const Field3D& theta_diffusivity,
                          Field3D& shear,
                          Field3D& buoyancy) const;

    Field3D eddy_viscosity_;
    Field3D theta_diffusivity_;
    Field3D moisture_diffusivity_;
    Field3D tke_diffusivity_;
};

} // namespace tmv

using tmv::TKEScheme;
