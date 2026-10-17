#pragma once

#include "core/field/field3d.hpp"

#include <string>
#include <unordered_map>
#include <vector>

namespace tmv
{
struct GridMetrics;
}

namespace tmv::chaos
{

struct SimulationState
{
    Field3D* u = nullptr;
    Field3D* v = nullptr;
    Field3D* w = nullptr;
    Field3D* theta = nullptr;
    Field3D* qv = nullptr;
    Field3D* qc = nullptr;
    Field3D* qr = nullptr;
    Field3D* qi = nullptr;
    Field3D* qs = nullptr;
    Field3D* qg = nullptr;
    Field3D* qh = nullptr;

    bool valid() const
    {
        return u && v && w && theta && qv && qc && qr;
    }
};

struct ChaosStateView
{
    const GridMetrics* grid = nullptr;

    const Field3D* u = nullptr;
    const Field3D* v = nullptr;
    const Field3D* w = nullptr;
    const Field3D* theta = nullptr;
    const Field3D* qv = nullptr;
    const Field3D* qc = nullptr;
    const Field3D* qr = nullptr;

    double pbl_height = 1000.0;
    const std::vector<std::vector<double>>* terrain_mask = nullptr;
};

struct ChaosTendencies
{
    Field3D* dtheta_micro_dt = nullptr;
    Field3D* dqv_micro_dt = nullptr;
    Field3D* dqc_micro_dt = nullptr;
    Field3D* dqr_micro_dt = nullptr;
    Field3D* dqi_micro_dt = nullptr;
    Field3D* dqs_micro_dt = nullptr;
    Field3D* dqg_micro_dt = nullptr;
    Field3D* dqh_micro_dt = nullptr;

    // Production microphysics requests one multiplier per complete vertical
    // column. The microphysics transaction subsequently admits that factor
    // against every mass and scheme-private number reservoir before applying
    // it to tendencies and the matching bottom sedimentation flux. Scaling
    // an already formed sedimentation divergence independently by height
    // would not conserve water.
    std::vector<float>* microphysics_column_multipliers = nullptr;

    Field3D* du_pbl_dt = nullptr;
    Field3D* dv_pbl_dt = nullptr;
    Field3D* dw_pbl_dt = nullptr;
    Field3D* dtheta_pbl_dt = nullptr;
    Field3D* dqv_pbl_dt = nullptr;

    // The production PBL hook perturbs the closure's conserved, held source
    // currency rather than its already density-divided tendencies. One
    // multiplier is used for the entire column, including the matching
    // surface flux diagnostics, so the lower-boundary flux and column
    // inventory/force changes remain an exact pair.
    Field3D* pbl_u_force_n = nullptr;
    Field3D* pbl_v_force_n = nullptr;
    Field3D* pbl_theta_inventory_rate_kgk_s = nullptr;
    Field3D* pbl_vapor_mass_rate_kg_s = nullptr;
    Field3D* pbl_tke_tendency = nullptr;
    std::vector<float>* pbl_ustar = nullptr;
    std::vector<float>* pbl_tau_u = nullptr;
    std::vector<float>* pbl_tau_v = nullptr;
    std::vector<float>* pbl_sensible_heat_flux = nullptr;
    std::vector<float>* pbl_latent_heat_flux = nullptr;
    std::vector<float>* pbl_moisture_flux = nullptr;

    Field3D* du_diff_dt = nullptr;
    Field3D* dv_diff_dt = nullptr;
    Field3D* dw_diff_dt = nullptr;
    Field3D* dtheta_diff_dt = nullptr;
    Field3D* dqv_diff_dt = nullptr;

    // Conservative SGS uncertainty acts on these nonnegative coefficients
    // before the turbulence scheme rebuilds its flux divergences. They are
    // mutually exclusive with the legacy diffusion-tendency pointers in the
    // production turbulence hook.
    Field3D* eddy_viscosity = nullptr;
    Field3D* theta_diffusivity = nullptr;
    Field3D* moisture_diffusivity = nullptr;
    Field3D* tke_diffusivity = nullptr;
};

struct ChaosEffects
{
    using Volume = std::vector<std::vector<std::vector<float>>>;

    std::unordered_map<std::string, Volume> delta_state;
    std::unordered_map<std::string, Volume> multipliers;
    Volume xi_field;
    double realized_variance = 0.0;
    double min_multiplier = 1.0;
    double max_multiplier = 1.0;
};

struct ChaosDiagnostics
{
    double mean_perturbation = 0.0;
    double variance_perturbation = 0.0;
    double correlation_length_x = 0.0;
    double correlation_length_y = 0.0;

    bool negative_moisture_detected = false;
    int negative_moisture_count = 0;
    double min_qv_after_perturbation = 1.0;

    std::vector<std::string> warnings;
    std::vector<std::string> errors;

    double time_noise_update = 0.0;
    double time_apply_ic = 0.0;
    double time_apply_tend = 0.0;
};

} // namespace tmv::chaos

using tmv::chaos::SimulationState;
using tmv::chaos::ChaosStateView;
using tmv::chaos::ChaosTendencies;
using tmv::chaos::ChaosEffects;
using tmv::chaos::ChaosDiagnostics;
