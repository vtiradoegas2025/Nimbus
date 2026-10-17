#pragma once

#include "chaos/config.hpp"
#include "chaos/scheme.hpp"
#include "chaos/state.hpp"

#include <memory>
#include <stdexcept>
#include <utility>

struct SimulationStateView;

namespace tmv::chaos
{

// Per-simulation scheme, stochastic history, caches, and diagnostics.
struct ChaosRuntimeState
{
    std::unique_ptr<ChaosScheme> scheme;
    ChaosConfig config;
    ChaosDiagnostics diagnostics;
    double noise_elapsed_s = 0.0;

    // Deterministic PBL publication cached at the last closure refresh. The
    // four historical field names are retained for checkpoint compatibility,
    // but now contain conserved u/v force, theta-inventory rate, and vapor
    // mass rate respectively—not density-divided tendencies.
    Field3D base_du_dt_pbl;
    Field3D base_dv_dt_pbl;
    Field3D base_dtheta_dt_pbl;
    Field3D base_dqv_dt_pbl;
    Field3D base_dtke_dt_pbl;
    std::vector<float> base_sfc_ustar;
    std::vector<float> base_sfc_tau_u;
    std::vector<float> base_sfc_tau_v;
    std::vector<float> base_sfc_sensible_heat_flux;
    std::vector<float> base_sfc_latent_heat_flux;
    std::vector<float> base_sfc_moisture_flux;
    bool pbl_base_tendencies_valid = false;
    // Reused only when immutable process-global grid metrics cannot be
    // borrowed. Copies/checkpoints omit this replaceable geometry cache.
    GridMetrics grid_metrics_scratch;

    ChaosRuntimeState() = default;
    ChaosRuntimeState(const ChaosRuntimeState& other)
        : config(other.config),
          diagnostics(other.diagnostics),
          noise_elapsed_s(other.noise_elapsed_s),
          base_du_dt_pbl(other.base_du_dt_pbl),
          base_dv_dt_pbl(other.base_dv_dt_pbl),
          base_dtheta_dt_pbl(other.base_dtheta_dt_pbl),
          base_dqv_dt_pbl(other.base_dqv_dt_pbl),
          base_dtke_dt_pbl(other.base_dtke_dt_pbl),
          base_sfc_ustar(other.base_sfc_ustar),
          base_sfc_tau_u(other.base_sfc_tau_u),
          base_sfc_tau_v(other.base_sfc_tau_v),
          base_sfc_sensible_heat_flux(other.base_sfc_sensible_heat_flux),
          base_sfc_latent_heat_flux(other.base_sfc_latent_heat_flux),
          base_sfc_moisture_flux(other.base_sfc_moisture_flux),
          pbl_base_tendencies_valid(other.pbl_base_tendencies_valid)
    {
        if (other.scheme)
        {
            scheme = other.scheme->clone();
            if (!scheme)
            {
                throw std::runtime_error(
                    "chaos scheme does not support runtime snapshots: "
                    + other.scheme->name());
            }
        }
    }
    ChaosRuntimeState& operator=(const ChaosRuntimeState& other)
    {
        if (this != &other)
        {
            ChaosRuntimeState copy(other);
            *this = std::move(copy);
        }
        return *this;
    }
    ChaosRuntimeState(ChaosRuntimeState&&) noexcept = default;
    ChaosRuntimeState& operator=(ChaosRuntimeState&&) noexcept = default;
};

extern ChaosRuntimeState& global_chaos_runtime_state;

// Compatibility aliases for the process-owned runtime.
extern std::unique_ptr<ChaosScheme>& chaos_scheme;
extern ChaosConfig& global_chaos_config;
extern ChaosDiagnostics& global_chaos_diagnostics;

void bind_chaos_compute_backend(const SimulationStateView& state);
void initialize_chaos(const SimulationStateView& state, const ChaosConfig& cfg);
void initialize_chaos(const ChaosConfig& cfg);

void apply_chaos_initial_conditions(const SimulationStateView& state);
void apply_chaos_initial_conditions();

void apply_chaos_tendencies(const SimulationStateView& state);
void apply_chaos_tendencies();

void apply_chaos_to_microphysics_tendencies(const SimulationStateView& state,
                                            Field3D& dtheta_dt,
                                            Field3D& dqv_dt,
                                            Field3D& dqc_dt,
                                            Field3D& dqr_dt,
                                            Field3D& dqi_dt,
                                            Field3D& dqs_dt,
                                            Field3D& dqg_dt,
                                            Field3D& dqh_dt);
bool apply_chaos_to_microphysics_tendencies(
    const SimulationStateView& state,
    Field3D& dtheta_dt,
    Field3D& dqv_dt,
    Field3D& dqc_dt,
    Field3D& dqr_dt,
    Field3D& dqi_dt,
    Field3D& dqs_dt,
    Field3D& dqg_dt,
    Field3D& dqh_dt,
    std::vector<float>& column_multipliers);
void apply_chaos_to_microphysics_tendencies(Field3D& dtheta_dt,
                                            Field3D& dqv_dt,
                                            Field3D& dqc_dt,
                                            Field3D& dqr_dt,
                                            Field3D& dqi_dt,
                                            Field3D& dqs_dt,
                                            Field3D& dqg_dt,
                                            Field3D& dqh_dt);

void apply_chaos_to_turbulence_tendencies(const SimulationStateView& state,
                                          TurbulenceTendencies& tendencies);
void apply_chaos_to_turbulence_tendencies(TurbulenceTendencies& tendencies);

void step_chaos_noise(const SimulationStateView& state, double dt);
void step_chaos_noise(double dt);

const ChaosDiagnostics& get_chaos_diagnostics();
const ChaosDiagnostics& get_chaos_diagnostics(const SimulationStateView& state);

} // namespace tmv::chaos

using tmv::chaos::ChaosRuntimeState;
using tmv::chaos::global_chaos_runtime_state;
using tmv::chaos::chaos_scheme;
using tmv::chaos::global_chaos_config;
using tmv::chaos::global_chaos_diagnostics;
using tmv::chaos::initialize_chaos;
using tmv::chaos::apply_chaos_initial_conditions;
using tmv::chaos::apply_chaos_tendencies;
using tmv::chaos::apply_chaos_to_microphysics_tendencies;
using tmv::chaos::apply_chaos_to_turbulence_tendencies;
using tmv::chaos::step_chaos_noise;
using tmv::chaos::get_chaos_diagnostics;
