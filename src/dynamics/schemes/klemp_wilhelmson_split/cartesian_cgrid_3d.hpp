/**
 * @file cartesian_cgrid_3d.hpp
 * @brief Arakawa C-grid Cartesian (fully 3D) dynamics scheme.
 *        v1 canonical id: `klemp_wilhelmson_split_cartesian_cgrid`.
 *
 * Full-domain staggered Cartesian split-explicit scheme on the global
 * Field3D grid using StaggeredCartesianDerivatives. Cartesian has no polar
 * axis, so a 1/r geometry, centrifugal/curvature terms, azimuthal
 * periodicity, and any axis singularity are all absent -- this is the
 * simplest robust core in the tree.
 *
 * Field placement (Arakawa C-grid; the x/y axes use the retained r/theta
 * grid-array aliases, see GridGeometry):
 *   u[i][j][k] (= u_x) at x-face (x_{i+1/2}, y_j,       z_k)
 *   v[i][j][k] (= u_y) at y-face (x_i,       y_{j+1/2}, z_k)
 *   w[i][j][k]         at z-face (x_i,       y_j,       z_{k+1/2})
 *   scalars            at cell center (x_i,  y_j,       z_k)
 * The ground face has implicit zero contravariant flux (physical w equals
 * u*dh/dx+v*dh/dy on terrain, and is zero only in the flat limit); the lid
 * face w[NZ-1]=0 is held by the boundary-condition module under a rigid top.
 *
 * Slow/fast split (Klemp-Wilhelmson):
 *   slow: advection + buoyancy on w (+ moisture buoyancy/loading);
 *         dp/dt = -u.grad(p) (advection only); drho/dt = 0.
 *   fast pressure: drho/dt = -div(rho u) (conservative), dp/dt = -gamma*p*div(u).
 *   fast momentum: -grad(p)/rho on u,v; -(dp/dz - dp0/dz)/rho on w
 *     (reference-state subtraction so calm p=p0(z) gives bit-exact zero).
 * The unsplit compute_momentum_tendencies is slow + fast in one call (the
 * Forward-Euler / test path).
 *
 * Lateral + vertical boundary conditions are applied externally by the
 * Cartesian C-grid BC module (boundary_conditions_cartesian_cgrid); the
 * scheme computes interior tendencies only.
 */
#pragma once
#include "numerics/time_stepping/native_rk.hpp"
#include "dynamics/cartesian_cgrid_momentum_flux.hpp"
#include "dynamics/dynamics_base.hpp"
#include "terrain/earth_fixed_geometry.hpp"
#include "numerics/derivatives/derivative_operators.hpp"
#include <string>
#include <vector>

struct SimulationStateView;

namespace tmv
{

/**
 * @brief Cartesian dynamics scheme on the Arakawa C-grid. Supports
 *        split-explicit acoustic time stepping.
 */
class CartesianCGrid3DScheme : public DynamicsScheme, public SplitExplicitDynamics
{
public:
    CartesianCGrid3DScheme();
    explicit CartesianCGrid3DScheme(const SimulationStateView& state);

    std::unique_ptr<DynamicsScheme> clone() const override;
    void bind_execution_context(const SimulationStateView& state) override;
    bool supports_domain_recycling_invalidation() const noexcept override
    {
        return true;
    }
    void invalidate_generation_state_after_domain_recycling() noexcept override;
    void set_coriolis_profile(const CartesianCoriolisProfile& profile) override
    {
        coriolis_profile_ = profile;
        has_coriolis_profile_ = true;
    }

    struct MovingDensityStep
    {
        Field3D density;
        TerrainContravariantFluxes integrated_relative_mass_flux;
        MovingMeshInventoryUpdate inventory;
    };

    MovingDensityStep prepare_moving_density_step(const PreparedEarthFixedTerrainMotion& motion) const;

    // --- DynamicsScheme interface (unsplit / single-step path) -----------

    void compute_momentum_tendencies(
        const Field3D& u,
        const Field3D& v,
        const Field3D& w,
        const Field3D& rho,
        const Field3D& p,
        const Field3D& theta,
        double dt,
        Field3D& du_dt,
        Field3D& dv_dt,
        Field3D& dw_dt,
        Field3D& drho_dt,
        Field3D& dp_dt
    ) override;

    // --- SplitExplicitDynamics interface ---------------------------------

    void compute_slow_tendencies(
        const Field3D& u, const Field3D& v, const Field3D& w,
        const Field3D& rho, const Field3D& p, const Field3D& theta, double dt,
        Field3D& du_dt, Field3D& dv_dt, Field3D& dw_dt,
        Field3D& drho_dt, Field3D& dp_dt,
        SlowMomentumAdvectionTendencyOutputs* advection_outputs = nullptr) override;

    void set_slow_rk2_invariant_input_residency(bool enabled) override
    {
        slow_rk2_invariant_input_residency_ = enabled;
    }

    void set_momentum_interface_flux_register(
        CartesianCGridMomentumFluxRegister* flux_register) noexcept override
    {
        momentum_interface_flux_register_ = flux_register;
    }
    void set_momentum_interface_flux_slow_weight(double weight) override;
    [[nodiscard]] bool momentum_interface_flux_capture_active() const noexcept override
    {
        return momentum_interface_flux_register_ != nullptr;
    }

    void compute_fast_pressure_tendencies(
        const Field3D& u, const Field3D& v, const Field3D& w,
        const Field3D& rho, const Field3D& p,
        double dt_small,
        Field3D& drho_dt, Field3D& dp_dt) override;

    void compute_fast_momentum_tendencies(
        const Field3D& u, const Field3D& v, const Field3D& w,
        const Field3D& rho, const Field3D& p,
        double dt_small,
        Field3D& du_dt, Field3D& dv_dt, Field3D& dw_dt) override;

    bool apply_fast_substep_cpu_in_place(
        Field3D& u, Field3D& v, Field3D& w,
        Field3D& rho, Field3D& p, const Field3D& theta,
        double dt_small, Field3D& drho_scratch) override;

    // --- Scheme identity -------------------------------------------------

    std::string get_scheme_name() const override { return "klemp_wilhelmson_split_cartesian_cgrid"; }
    std::string get_coordinate_system() const override { return "cartesian_cgrid"; }
    int get_num_prognostic_vars() const override { return 5; }

private:
    std::shared_ptr<const SimulationStateView> state_;
    void compute_conservative_slow_tendencies(
        const Field3D& u, const Field3D& v, const Field3D& w,
        const Field3D& rho, const Field3D& p, const Field3D& theta,
        double dt, Field3D& du_dt, Field3D& dv_dt, Field3D& dw_dt,
        Field3D& drho_dt, Field3D& dp_dt,
        SlowMomentumAdvectionTendencyOutputs* advection_outputs);

    int NX_, NY_, NZ_;
    double dx_, dy_, dz_;
    StaggeredCartesianDerivatives deriv_;
    // True while compute_slow_tendencies replays itself on the CPU path for
    // the parity harness (GPU.md Phase 1); suppresses the GPU
    // dispatch so the replay measures the CPU reference.
    bool parity_replay_ = false;
    bool native_unsplit_ = false;
    tmv::NativeRkTendencyWorkspace native_rk_;
    std::array<Field3D, 2> native_rk_base_pressure_;
    // Enabled only by the orchestration-owned two-stage Heun transaction.
    // It is deliberately false for arbitrary direct tendency calls.
    bool slow_rk2_invariant_input_residency_ = false;

    // Replaceable accelerator packing scratch. These vectors are not
    // scientific state and do not participate in checkpoint state. Retaining
    // capacity removes two RK2-stage allocation clusters per outer step
    // without changing the packed values.
    std::vector<float> gpu_loading_;
    std::vector<float> gpu_rho0_;
    std::vector<float> gpu_qv0_;
    std::vector<float> gpu_u0_;
    std::vector<float> gpu_v0_;
    CartesianCoriolisProfile coriolis_profile_{};
    bool has_coriolis_profile_ = false;
    CartesianCGridMomentumFluxRegister* momentum_interface_flux_register_ = nullptr;
    double momentum_interface_flux_slow_weight_ = 1.0;
};

} // namespace tmv

using tmv::CartesianCGrid3DScheme;
