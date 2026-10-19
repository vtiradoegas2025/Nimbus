#pragma once
#include "numerics/diffusion/diffusion_base.hpp"

namespace tmv
{

enum class ImplicitVerticalComponent
{
    Scalar,
    U,
    V,
    W
};

class ImplicitDiffusionScheme final : public DiffusionSchemeBase
{
    static void solve_tridiagonal(
        const std::vector<double>& a,
        const std::vector<double>& b,
        const std::vector<double>& c,
        const std::vector<double>& rhs,
        std::vector<double>& x);

public:
    ImplicitDiffusionScheme() = default;
    std::string name() const override { return "implicit"; }
    std::unique_ptr<DiffusionSchemeBase> clone() const override
    {
        return std::make_unique<ImplicitDiffusionScheme>(*this);
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
    void implicit_scalar_diffusion(
        const Field3D& field,
        const Field3D& rho,
        const Field3D* K_field,
        const DiffusionConfig& cfg,
        const DiffusionStateView& state,
        Field3D& field_new);

    void implicit_momentum_diffusion(
        const Field3D& u,
        const Field3D& v,
        const Field3D& w,
        const Field3D& rho,
        const Field3D* K_field,
        const DiffusionConfig& cfg,
        const DiffusionStateView& state,
        Field3D& u_new,
        Field3D& v_new,
        Field3D& w_new);

    void implicit_vertical_diffusion(
        const Field3D& field,
        const Field3D& rho,
        const Field3D* K_field,
        double K_default,
        const GridMetrics& grid,
        double dt,
        bool c_grid,
        bool periodic_x,
        bool periodic_y,
        ImplicitVerticalComponent component,
        Field3D& field_new);
};

} // namespace tmv

using tmv::ImplicitDiffusionScheme;
