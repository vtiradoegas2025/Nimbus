#pragma once

#include "chaos/runtime.hpp"
#include "core/runtime/simulation_state.hpp"

#include <memory>
#include <string>

namespace tmv::chaos::runtime_detail
{

struct ChaosRuntimeView
{
    std::unique_ptr<ChaosScheme>& scheme;
    ChaosConfig& config;
    ChaosDiagnostics& diagnostics;
    double& noise_elapsed_s;
    Field3D& base_du_dt_pbl;
    Field3D& base_dv_dt_pbl;
    Field3D& base_dtheta_dt_pbl;
    Field3D& base_dqv_dt_pbl;
    Field3D& base_dtke_dt_pbl;
    std::vector<float>& base_sfc_ustar;
    std::vector<float>& base_sfc_tau_u;
    std::vector<float>& base_sfc_tau_v;
    std::vector<float>& base_sfc_sensible_heat_flux;
    std::vector<float>& base_sfc_latent_heat_flux;
    std::vector<float>& base_sfc_moisture_flux;
    bool& pbl_base_tendencies_valid;
    GridMetrics& grid_metrics_scratch;
};

ChaosRuntimeView chaos_runtime_for(const SimulationStateView& state);
std::string normalize_tendency_block_name(const std::string& block_raw);
void invalidate_pbl_base_tendencies(ChaosRuntimeView& runtime);
const GridMetrics& grid_metrics_from_state(
    const SimulationStateView& state);

} // namespace tmv::chaos::runtime_detail
