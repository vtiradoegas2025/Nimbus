#pragma once
#include "boundary_layer/base/surface_fluxes.hpp"
#include "boundary_layer/boundary_layer_base.hpp"

namespace tmv
{

class YSUScheme : public BoundaryLayerSchemeBase {
public:
    std::unique_ptr<BoundaryLayerSchemeBase> clone() const override
    {
        return std::make_unique<YSUScheme>(*this);
    }

    YSUScheme();

    std::string name() const override { return "ysu"; }
    int required_fields() const override;

    void initialize(const BoundaryLayerConfig& cfg) override;

    void compute_column(
        const BoundaryLayerConfig& cfg,
        const SurfaceConfig& sfc,
        const BoundaryLayerColumnStateView& col,
        BoundaryLayerTendencies& tend,
        BoundaryLayerDiagnostics* diag_opt = nullptr) override;

private:
    double diagnose_pbl_height(
        const BoundaryLayerColumnStateView& col,
        const BoundaryLayerConfig& cfg,
        double surface_virtual_heat_flux,
        double mixing_velocity) const;

    void compute_k_profile(
        const BoundaryLayerColumnStateView& col,
        double h, double ustar, double wstar,
        double monin_obukhov_length,
        std::vector<double>& K_m,
        std::vector<double>& K_h
    ) const;

    void apply_diffusion_tendencies(
        const BoundaryLayerColumnStateView& col,
        const std::vector<double>& K_m,
        const std::vector<double>& K_h,
        surface_fluxes::BulkFluxes& fluxes,
        double interval_s,
        double h,
        double wstar,
        BoundaryLayerColumnWorkspace& workspace,
        BoundaryLayerTendencies& tend
    ) const;

    void apply_cloud_top_diffusion_tendencies(
        const BoundaryLayerColumnStateView& col,
        const std::vector<double>& cloud_diffusivity,
        double interval_s,
        BoundaryLayerColumnWorkspace& workspace,
        BoundaryLayerTendencies& tend
    ) const;
};

} // namespace tmv

using tmv::YSUScheme;
