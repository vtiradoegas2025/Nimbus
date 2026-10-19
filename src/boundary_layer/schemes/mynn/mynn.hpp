/**
 * @file mynn.hpp
 * @brief Declarations for the boundary_layer module.
 *
 * Defines interfaces, data structures, and contracts used by
 * the boundary_layer runtime and scheme implementations.
 * This file is part of the src/boundary_layer subsystem.
 */

#pragma once
#include "boundary_layer/base/surface_fluxes.hpp"
#include "boundary_layer/boundary_layer_base.hpp"

namespace tmv
{

/**
 * @brief Implements the MYNN boundary layer scheme.
 */
class MYNNScheme : public BoundaryLayerSchemeBase
{
private:
    double b1_ = 24.0;

    double lmax_ = 500.0;

    /**
     * @brief Initializes the MYNN boundary layer scheme.
     */
public:
    std::unique_ptr<BoundaryLayerSchemeBase> clone() const override
    {
        return std::make_unique<MYNNScheme>(*this);
    }

    MYNNScheme();

    std::string name() const override
    {
        return "mynn";
    }
    int required_fields() const override;

    void initialize(const BoundaryLayerConfig& cfg) override;

    void compute_column(const BoundaryLayerConfig&          cfg,
                        const SurfaceConfig&                sfc,
                        const BoundaryLayerColumnStateView& col,
                        BoundaryLayerTendencies&            tend,
                        BoundaryLayerDiagnostics*           diag_opt = nullptr) override;
    bool supports_native_column_batch(const BoundaryLayerConfig&         cfg,
                                      const BoundaryLayerComputeKernels& kernels) const override;
    bool compute_columns_native(const BoundaryLayerConfig&          cfg,
                                const SurfaceConfig&                surface,
                                const BoundaryLayerColumnStateView* columns,
                                std::size_t                         column_count,
                                BoundaryLayerTendencies*            tendencies,
                                BoundaryLayerDiagnostics*           diagnostics,
                                BoundaryLayerComputeKernels&        kernels) override;

    /**
     * @brief Returns the name of the MYNN boundary layer scheme.
     */
private:
    double diagnose_pbl_height(const BoundaryLayerColumnStateView& col,
                               const std::vector<double>&          tke_col,
                               const BoundaryLayerConfig&          cfg);

    void compute_mixing_length(const BoundaryLayerColumnStateView& col, double h, std::vector<double>& l_mix);

    void compute_eddy_diffusivities(const BoundaryLayerColumnStateView& col,
                                    const std::vector<double>&          tke_col,
                                    const std::vector<double>&          l_mix,
                                    std::vector<double>&                K_m,
                                    std::vector<double>&                K_h,
                                    std::vector<double>&                K_e);

    void update_tke(const BoundaryLayerColumnStateView& col,
                    const std::vector<double>&          tke_col,
                    const std::vector<double>&          l_mix,
                    const surface_fluxes::BulkFluxes&   fluxes,
                    const std::vector<double>&          K_m,
                    const std::vector<double>&          K_h,
                    const std::vector<double>&          K_e,
                    double                              dt,
                    std::vector<double>&                tke_tend);

    void apply_diffusion_tendencies(double interval_s, const BoundaryLayerColumnStateView& col,
                                    const std::vector<double>&          K_m,
                                    const std::vector<double>&          K_h,
                                    surface_fluxes::BulkFluxes&         fluxes,
                                    BoundaryLayerTendencies&            tend);
};

} // namespace tmv

using tmv::MYNNScheme;
