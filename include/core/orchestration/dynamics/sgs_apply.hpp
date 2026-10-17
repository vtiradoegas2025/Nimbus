#pragma once

#include "core/runtime/simulation_state.hpp"
#include "turbulence/turbulence_base.hpp"

#include <limits>

struct SgsExplicitStability
{
    double maximum_timestep_s = std::numeric_limits<double>::infinity();
    double momentum_energy_timestep_s = std::numeric_limits<double>::infinity();
    double momentum_irreversible_dissipation_w = 0.0;
    double momentum_boundary_work_w = 0.0;
    double momentum_stress_closure_residual_w = 0.0;
    bool open_boundary_work_accounted = false;
    // Flat orthogonal diffusion owns a discrete maximum principle. Mapped
    // tensor diffusion instead combines these physical-bound limits with the
    // two energy limits below; its non-orthogonal cross flux is not an M-matrix.
    double theta_monotonic_timestep_s = std::numeric_limits<double>::infinity();
    double qv_monotonic_timestep_s = std::numeric_limits<double>::infinity();
    double theta_energy_timestep_s = std::numeric_limits<double>::infinity();
    double qv_energy_timestep_s = std::numeric_limits<double>::infinity();
    double tke_positivity_timestep_s = std::numeric_limits<double>::infinity();
};

// Evaluates the final, realized tendency buffer.  This deliberately happens
// after device production and stochastic modification, so CPU and GPU paths
// are admitted by the same physical bound.
SgsExplicitStability compute_sgs_explicit_stability(
    const SimulationStateView& state,
    const TurbulenceTendencies& turb_tend);

TurbulenceMomentumWorkReport apply_sgs_tendencies(
    const SimulationStateView& state,
    double dt_dynamics,
    const TurbulenceTendencies& turb_tend);

void accumulate_sgs_energy_work(
    TurbulenceMomentumWorkReport& destination,
    const TurbulenceMomentumWorkReport& source);

void commit_sgs_momentum_work(
    const SimulationStateView& state,
    const TurbulenceMomentumWorkReport& report);
