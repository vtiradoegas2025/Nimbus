/**
 * @file cartesian_agrid.hpp
 * @brief Declarations for the Cartesian A-grid dynamics scheme.
 *
 * Cartesian (x, y, z), collocated (A-grid) implementation of the
 * DynamicsScheme interface. Selected when grid is cartesian + collocated.
 * v1 canonical id: `klemp_wilhelmson_split_cartesian_agrid`.
 *
 * Field reuse for the scoped Cartesian A-grid flat-CPU path:
 *   - The first field argument (formal name `u`) carries u_x.
 *   - The second field argument (formal name `v`) carries u_y.
 *   - The third field argument (formal name `w`) carries w (unchanged).
 *
 * Will rename the field globals (u -> u_x, v -> u_y) and update
 * the base interface accordingly. Until then we keep the retained-alias
 * signature so the existing orchestration call sites do not have
 * to change.
 *
 * This file is part of the src/dynamics subsystem.
 */

#pragma once
#include "dynamics/dynamics_base.hpp"
#include "numerics/derivatives/derivative_operators.hpp"
#include <memory>
#include <vector>

struct SimulationStateView;

namespace tmv
{

/**
 * @brief Cartesian-grid dynamics scheme (Phase A of the Coordinate Backend
 *        Plan). Computes momentum, mass, and pressure tendencies using pure
 *        (x, y, z) finite differences. Has no centrifugal, 1/r, azimuthal
 *        periodic wrap, or axis-singularity terms, so non-axisymmetric base
 *        states such as the WK2002 supercell hodograph are evaluated correctly.
 */
class CartesianAGridScheme : public DynamicsScheme, public SplitExplicitDynamics
{
public:
    /**
     * @brief Constructs the Cartesian dynamics scheme, caching grid dims.
     */
    CartesianAGridScheme();
    explicit CartesianAGridScheme(const SimulationStateView& state);
    CartesianAGridScheme(const CartesianAGridScheme& other);

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

    /**
     * @brief Computes the momentum tendencies on a Cartesian grid.
     *        See the class-level note on field aliasing.
     */
    void compute_momentum_tendencies(const Field3D& u,
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
                                     Field3D& dp_dt) override;

    // SplitExplicitDynamics interface
    void compute_slow_tendencies(
        const Field3D& u, const Field3D& v, const Field3D& w,
        const Field3D& rho, const Field3D& p, const Field3D& theta, double dt,
        Field3D& du_dt, Field3D& dv_dt, Field3D& dw_dt,
        Field3D& drho_dt, Field3D& dp_dt,
        SlowMomentumAdvectionTendencyOutputs* advection_outputs = nullptr) override;
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

    std::string get_scheme_name() const override { return "klemp_wilhelmson_split_cartesian_agrid"; }
    std::string get_coordinate_system() const override { return "cartesian"; }
    int get_num_prognostic_vars() const override { return 5; }

private:
    std::shared_ptr<const SimulationStateView> state_;
    int NR_;
    int NTH_;
    int NZ_;
    double dr_;
    double dy_;
    double dz_;
    std::unique_ptr<DerivativeOperators> deriv_;
    // Reused host upload storage for the optional Vulkan full-tendency path.
    // These are execution scratch, not prognostic/checkpoint state.
    std::vector<float> gpu_p0_;
    std::vector<float> gpu_rho0_;
    std::vector<float> gpu_qv0_;
    std::vector<float> gpu_u0_;
    std::vector<float> gpu_v0_;
    std::vector<float> gpu_loading_;
    CartesianCoriolisProfile coriolis_profile_{};
    bool has_coriolis_profile_ = false;
};

} // namespace tmv

using tmv::CartesianAGridScheme;
