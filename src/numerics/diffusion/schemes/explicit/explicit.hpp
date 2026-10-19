#pragma once
#include "numerics/diffusion/diffusion_base.hpp"

#include <vector>

namespace tmv
{

class ExplicitDiffusionScheme : public DiffusionSchemeBase
{
private:
    DiffusionConfig config_;
    // Packed float32 device metric profile. Capacity is retained across
    // timesteps; it is orchestration scratch, not scientific/checkpoint state.
    std::vector<float> device_dz_scratch_;

public:
    ExplicitDiffusionScheme();

    std::string name() const override { return "explicit"; }
    std::unique_ptr<DiffusionSchemeBase> clone() const override
    {
        auto copy = std::make_unique<ExplicitDiffusionScheme>();
        copy->config_ = config_;
        return copy;
    }
    void initialize() override;
    void initialize(const DiffusionConfig& cfg) override;

    void compute_diffusion_tendencies(
        const DiffusionConfig& cfg,
        const DiffusionStateView& state,
        DiffusionTendencies& tendencies,
        DiffusionDiagnostics* diag_opt = nullptr) override;

    double check_stability(
        const DiffusionConfig& cfg,
        const DiffusionStateView& state) override;

private:
    void compute_scalar_diffusion(
        const Field3D& field,
        const Field3D& rho,
        const Field3D* K_field,
        double K_default,
        const GridMetrics& grid,
        bool c_grid,
        Field3D& tendency);

    void add_horizontal_scalar_diffusion(
        const Field3D& field,
        const Field3D& rho,
        double K_h,
        const GridMetrics& grid,
        bool c_grid,
        bool periodic_x,
        bool periodic_y,
        Field3D& tendency);

    void add_horizontal_momentum_diffusion(
        const Field3D& u,
        const Field3D& v,
        const Field3D& w,
        const Field3D& rho,
        double K_h,
        const GridMetrics& grid,
        bool c_grid,
        bool periodic_x,
        bool periodic_y,
        Field3D& du_dt,
        Field3D& dv_dt,
        Field3D& dw_dt);

    void compute_momentum_diffusion(
        const Field3D& u,
        const Field3D& v,
        const Field3D& w,
        const Field3D& rho,
        const Field3D* nu_t,
        double nu_default,
        const GridMetrics& grid,
        bool c_grid,
        bool periodic_x,
        bool periodic_y,
        Field3D& du_dt,
        Field3D& dv_dt,
        Field3D& dw_dt);

};

} // namespace tmv

using tmv::ExplicitDiffusionScheme;
