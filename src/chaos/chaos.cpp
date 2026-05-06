/**
 * @file chaos.cpp
 * @brief Implementation for the chaos module.
 *
 * Provides executable logic for the chaos runtime path,
 * including initialization, stepping, and diagnostics helpers.
 * This file is part of the src/chaos subsystem.
 */

#include "core/runtime/simulation.hpp"
#include "boundary_layer/boundary_layer_base.hpp"
#include "factory.hpp"
#include "turbulence/turbulence_base.hpp"
#include "util/string_utils.hpp"
#include "util/simd_utils.hpp"
#include "util/log.hpp"
#include <iostream>
#include <memory>
#include <sstream>
#include <cmath>
#include <unordered_map>
#include <unordered_set>


namespace {

constexpr double k_default_dt_chaos = 60.0;
constexpr double k_default_tau_t = 21600.0;
constexpr double k_default_xi_max = 2.0;
constexpr double k_default_taper_z1 = 1000.0;
constexpr double k_default_taper_z2 = 3000.0;

uint64_t pbl_chaos_call_count = 0;
uint64_t microphysics_chaos_call_count = 0;
uint64_t turbulence_chaos_call_count = 0;

Field3D base_du_dt_pbl;
Field3D base_dv_dt_pbl;
Field3D base_dtheta_dt_pbl;
Field3D base_dqv_dt_pbl;
bool pbl_base_tendencies_valid = false;

std::string normalize_tendency_block_name(const std::string& block_raw)
{
    std::string block = tmv::strutil::trim_and_lower(block_raw);
    if (block == "pbl" || block == "boundary_layer" || block == "boundarylayer" || block == "bl")
    {
        return "pbl";
    }
    if (block == "micro" || block == "microphysics")
    {
        return "microphysics";
    }
    if (block == "diffusion" || block == "turbulence" || block == "sgs" || block == "subgrid")
    {
        return "diffusion";
    }
    return block;
}

void canonicalize_tendency_blocks(chaos::ChaosConfig& cfg)
{
    std::vector<std::string> canonical_blocks;
    canonical_blocks.reserve(cfg.apply_to_tendencies.size());
    std::unordered_set<std::string> seen;
    for (const std::string& raw_block : cfg.apply_to_tendencies)
    {
        const std::string canonical = normalize_tendency_block_name(raw_block);
        if (!canonical.empty() && seen.insert(canonical).second)
        {
            canonical_blocks.push_back(canonical);
        }
    }
    cfg.apply_to_tendencies = std::move(canonical_blocks);

    std::unordered_map<std::string, double> canonical_alpha;
    canonical_alpha.reserve(cfg.alpha_tend.size());
    for (const auto& kv : cfg.alpha_tend)
    {
        const std::string canonical = normalize_tendency_block_name(kv.first);
        if (!canonical.empty())
        {
            canonical_alpha[canonical] = kv.second;
        }
    }
    cfg.alpha_tend = std::move(canonical_alpha);
}

void reset_cadence_counters()
{
    pbl_chaos_call_count = 0;
    microphysics_chaos_call_count = 0;
    turbulence_chaos_call_count = 0;
}

void invalidate_pbl_base_tendencies()
{
    base_du_dt_pbl = Field3D();
    base_dv_dt_pbl = Field3D();
    base_dtheta_dt_pbl = Field3D();
    base_dqv_dt_pbl = Field3D();
    pbl_base_tendencies_valid = false;
}

void refresh_pbl_base_tendencies()
{
    if (du_dt_pbl.empty() || dv_dt_pbl.empty() || dtheta_dt_pbl.empty() || dqv_dt_pbl.empty())
    {
        invalidate_pbl_base_tendencies();
        return;
    }

    base_du_dt_pbl = du_dt_pbl;
    base_dv_dt_pbl = dv_dt_pbl;
    base_dtheta_dt_pbl = dtheta_dt_pbl;
    base_dqv_dt_pbl = dqv_dt_pbl;
    pbl_base_tendencies_valid = true;
}

void restore_pbl_tendencies_from_base()
{
    if (!pbl_base_tendencies_valid)
    {
        return;
    }

    const bool shape_match =
        base_du_dt_pbl.size_r() == du_dt_pbl.size_r() &&
        base_du_dt_pbl.size_th() == du_dt_pbl.size_th() &&
        base_du_dt_pbl.size_z() == du_dt_pbl.size_z() &&
        base_dv_dt_pbl.size_r() == dv_dt_pbl.size_r() &&
        base_dv_dt_pbl.size_th() == dv_dt_pbl.size_th() &&
        base_dv_dt_pbl.size_z() == dv_dt_pbl.size_z() &&
        base_dtheta_dt_pbl.size_r() == dtheta_dt_pbl.size_r() &&
        base_dtheta_dt_pbl.size_th() == dtheta_dt_pbl.size_th() &&
        base_dtheta_dt_pbl.size_z() == dtheta_dt_pbl.size_z() &&
        base_dqv_dt_pbl.size_r() == dqv_dt_pbl.size_r() &&
        base_dqv_dt_pbl.size_th() == dqv_dt_pbl.size_th() &&
        base_dqv_dt_pbl.size_z() == dqv_dt_pbl.size_z();
    if (!shape_match)
    {
        invalidate_pbl_base_tendencies();
        return;
    }

    du_dt_pbl = base_du_dt_pbl;
    dv_dt_pbl = base_dv_dt_pbl;
    dtheta_dt_pbl = base_dtheta_dt_pbl;
    dqv_dt_pbl = base_dqv_dt_pbl;
}

GridMetrics fallback_grid_metrics()
{
    GridMetrics grid;
    grid.dx = dr;
    grid.dy = std::max(dr, dtheta * dr);
    grid.dz.assign(NZ, dz);
    grid.z_int.resize(NZ + 1);
    grid.z_int[0] = 0.0;
    for (int k = 1; k <= NZ; ++k)
    {
        grid.z_int[k] = grid.z_int[k - 1] + dz;
    }
    return grid;
}

void apply_default_chaos_targets(chaos::ChaosConfig& cfg)
{
    if (cfg.scheme_id == "initial_conditions" && cfg.apply_to_ic.empty())
    {
        cfg.apply_to_ic = {"u", "v", "w", "theta", "qv"};
    }
    if (cfg.scheme_id == "initial_conditions")
    {
        cfg.apply_to_tendencies.clear();
    }
    if (cfg.scheme_id == "boundary_layer" && cfg.apply_to_tendencies.empty())
    {
        cfg.apply_to_tendencies = {"pbl"};
    }
    if (cfg.scheme_id == "full_stochastic" && cfg.apply_to_tendencies.empty())
    {
        cfg.apply_to_tendencies = {"microphysics", "pbl", "diffusion"};
    }
}

bool block_enabled(const chaos::ChaosConfig& cfg, const std::string& block)
{
    if (cfg.scheme_id == "none")
    {
        return false;
    }
    if (cfg.apply_to_tendencies.empty())
    {
        return false;
    }
    const std::string canonical_block = normalize_tendency_block_name(block);
    return std::find(cfg.apply_to_tendencies.begin(), cfg.apply_to_tendencies.end(), canonical_block) != cfg.apply_to_tendencies.end();
}

bool cadence_due(uint64_t& counter)
{
    ++counter;
    const double dt_eff = std::max(dt, 1e-6);
    const double dt_chaos_eff =
        (std::isfinite(global_chaos_config.dt_chaos) && global_chaos_config.dt_chaos > 0.0)
            ? global_chaos_config.dt_chaos
            : k_default_dt_chaos;
    const uint64_t stride = std::max<uint64_t>(1, static_cast<uint64_t>(std::llround(dt_chaos_eff / dt_eff)));
    return (counter % stride) == 0;
}

chaos::ChaosStateView make_state_view(const GridMetrics& active_grid)
{
    chaos::ChaosStateView state_view;
    state_view.grid = &active_grid;
    state_view.u = &u;
    state_view.v = &v;
    state_view.w = &w;
    state_view.theta = &theta;
    state_view.qv = &qv;
    state_view.qc = &qc;
    state_view.qr = &qr;
    state_view.pbl_height = 1000.0;
    return state_view;
}

int sanitize_nonfinite_field(Field3D* field)
{
    if (field == nullptr || field->empty())
    {
        return 0;
    }

    float* const data = field->data();
    const int count = static_cast<int>(field->size());
    return simd_utils::sanitize_nonfinite(data, 0.0f, data, count);
}

int sanitize_field_with_clamp(Field3D* field, float (*clamp_fn)(float))
{
    if (field == nullptr || field->empty())
    {
        return 0;
    }

    int sanitized = 0;
    float* const data = field->data();
    const std::size_t count = field->size();
    #pragma omp parallel for reduction(+:sanitized)
    for (long long idx = 0; idx < static_cast<long long>(count); ++idx)
    {
        const float old_value = data[idx];
        const float new_value = clamp_fn(old_value);
        if (new_value != old_value)
        {
            ++sanitized;
        }
        data[idx] = new_value;
    }
    return sanitized;
}

int sanitize_chaos_tendencies(chaos::ChaosTendencies& tendencies)
{
    int sanitized = 0;
    sanitized += sanitize_nonfinite_field(tendencies.dtheta_micro_dt);
    sanitized += sanitize_nonfinite_field(tendencies.dqv_micro_dt);
    sanitized += sanitize_nonfinite_field(tendencies.dqc_micro_dt);
    sanitized += sanitize_nonfinite_field(tendencies.dqr_micro_dt);
    sanitized += sanitize_nonfinite_field(tendencies.dqi_micro_dt);
    sanitized += sanitize_nonfinite_field(tendencies.dqs_micro_dt);
    sanitized += sanitize_nonfinite_field(tendencies.dqg_micro_dt);
    sanitized += sanitize_nonfinite_field(tendencies.dqh_micro_dt);

    sanitized += sanitize_nonfinite_field(tendencies.du_pbl_dt);
    sanitized += sanitize_nonfinite_field(tendencies.dv_pbl_dt);
    sanitized += sanitize_nonfinite_field(tendencies.dw_pbl_dt);
    sanitized += sanitize_nonfinite_field(tendencies.dtheta_pbl_dt);
    sanitized += sanitize_nonfinite_field(tendencies.dqv_pbl_dt);

    sanitized += sanitize_nonfinite_field(tendencies.du_diff_dt);
    sanitized += sanitize_nonfinite_field(tendencies.dv_diff_dt);
    sanitized += sanitize_nonfinite_field(tendencies.dw_diff_dt);
    sanitized += sanitize_nonfinite_field(tendencies.dtheta_diff_dt);
    sanitized += sanitize_nonfinite_field(tendencies.dqv_diff_dt);
    return sanitized;
}

int sanitize_chaos_initial_state(SimulationState& state)
{
    int sanitized = 0;
    sanitized += sanitize_field_with_clamp(state.u, clamp_wind_horizontal_ms);
    sanitized += sanitize_field_with_clamp(state.v, clamp_wind_horizontal_ms);
    sanitized += sanitize_field_with_clamp(state.w, clamp_wind_vertical_ms);
    sanitized += sanitize_field_with_clamp(state.theta, clamp_theta_k);
    sanitized += sanitize_field_with_clamp(state.qv, clamp_qv_kgkg);
    sanitized += sanitize_field_with_clamp(state.qc, clamp_hydrometeor_kgkg);
    sanitized += sanitize_field_with_clamp(state.qr, clamp_hydrometeor_kgkg);
    sanitized += sanitize_field_with_clamp(state.qi, clamp_hydrometeor_kgkg);
    sanitized += sanitize_field_with_clamp(state.qs, clamp_hydrometeor_kgkg);
    sanitized += sanitize_field_with_clamp(state.qg, clamp_hydrometeor_kgkg);
    sanitized += sanitize_field_with_clamp(state.qh, clamp_hydrometeor_kgkg);
    return sanitized;
}

}

std::unique_ptr<chaos::ChaosScheme> chaos_scheme = nullptr;

chaos::ChaosConfig global_chaos_config;

chaos::ChaosDiagnostics global_chaos_diagnostics;

/**
 * @brief Initialize the chaos perturbation scheme
 * @param cfg Chaos configuration from YAML or defaults
 */
void initialize_chaos(const chaos::ChaosConfig& cfg) 
{
    try {
        global_chaos_config = cfg;
        if (global_chaos_config.scheme_id.empty())
        {
            global_chaos_config.scheme_id = "none";
        }
        std::transform(
            global_chaos_config.scheme_id.begin(),
            global_chaos_config.scheme_id.end(),
            global_chaos_config.scheme_id.begin(),
            [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        chaos_scheme = create_chaos_scheme(global_chaos_config.scheme_id);
        global_chaos_config.scheme_id = chaos_scheme->name();
        apply_default_chaos_targets(global_chaos_config);
        canonicalize_tendency_blocks(global_chaos_config);
        if (!std::isfinite(global_chaos_config.dt_chaos) || global_chaos_config.dt_chaos <= 0.0)
        {
            tmv::log_info("[CHAOS CONFIG] invalid dt_chaos=", global_chaos_config.dt_chaos,
                         "; using ", k_default_dt_chaos, " s");
            global_chaos_config.dt_chaos = k_default_dt_chaos;
        }
        if (!std::isfinite(global_chaos_config.tau_t) || global_chaos_config.tau_t < 0.0)
        {
            tmv::log_info("[CHAOS CONFIG] invalid tau_t=", global_chaos_config.tau_t,
                         "; using ", k_default_tau_t, " s");
            global_chaos_config.tau_t = k_default_tau_t;
        }
        if (!std::isfinite(global_chaos_config.xi_max) || global_chaos_config.xi_max <= 0.0)
        {
            tmv::log_info("[CHAOS CONFIG] invalid xi_max=", global_chaos_config.xi_max,
                         "; using ", k_default_xi_max);
            global_chaos_config.xi_max = k_default_xi_max;
        }
        if (!std::isfinite(global_chaos_config.taper_z1))
        {
            tmv::log_info("[CHAOS CONFIG] invalid taper_z1=", global_chaos_config.taper_z1,
                         "; using ", k_default_taper_z1, " m");
            global_chaos_config.taper_z1 = k_default_taper_z1;
        }
        if (!std::isfinite(global_chaos_config.taper_z2))
        {
            tmv::log_info("[CHAOS CONFIG] invalid taper_z2=", global_chaos_config.taper_z2,
                         "; using ", k_default_taper_z2, " m");
            global_chaos_config.taper_z2 = k_default_taper_z2;
        }
        if (global_chaos_config.taper_z2 <= global_chaos_config.taper_z1)
        {
            tmv::log_info("[CHAOS CONFIG] taper_z2 (", global_chaos_config.taper_z2,
                         ") must be greater than taper_z1 (", global_chaos_config.taper_z1,
                         "); expanding taper_z2 by 1 m");
            global_chaos_config.taper_z2 = global_chaos_config.taper_z1 + 1.0;
        }
        reset_cadence_counters();
        invalidate_pbl_base_tendencies();
        global_chaos_diagnostics = chaos::ChaosDiagnostics{};

        GridMetrics init_grid = (global_grid_metrics.dx > 0.0) ? global_grid_metrics : fallback_grid_metrics();
        chaos_scheme->initialize(global_chaos_config, init_grid);

        tmv::log_info("Initialized chaos scheme: ", global_chaos_config.scheme_id);
        if (global_chaos_config.scheme_id == "boundary_layer")
        {
            tmv::log_info("[CHAOS NOTE] 'boundary_layer' here means stochastic PBL tendency perturbations.\n"
                          "  It does not replace the physical PBL solver (boundary_layer.scheme=",
                          global_boundary_layer_config.scheme_id, ").");
        }

        if (global_chaos_config.scheme_id != "none") {
            tmv::log_info("  Perturbation amplitudes:");
            if (!global_chaos_config.apply_to_ic.empty()) {
                std::ostringstream oss;
                oss << "    IC variables: ";
                for (const auto& var : global_chaos_config.apply_to_ic) {
                    auto it = global_chaos_config.sigma_ic.find(var);
                    if (it != global_chaos_config.sigma_ic.end()) {
                        oss << var << "(σ=" << it->second << ") ";
                    }
                }
                tmv::log_info(oss.str());
            }
            if (!global_chaos_config.apply_to_tendencies.empty()) {
                std::ostringstream oss;
                oss << "    Tendency blocks: ";
                for (const auto& block : global_chaos_config.apply_to_tendencies) {
                    auto it = global_chaos_config.alpha_tend.find(block);
                    if (it != global_chaos_config.alpha_tend.end()) {
                        oss << block << "(α=" << it->second << ") ";
                    }
                }
                tmv::log_info(oss.str());
            }
            tmv::log_info("  Correlation: Lx=", global_chaos_config.Lx, "m, Ly=", global_chaos_config.Ly, "m");
            tmv::log_info("  Temporal decorrelation: τ=", global_chaos_config.tau_t, "s");
        }

    } catch (const std::exception& e) {
        tmv::log_error("Error initializing chaos: ", e.what());
        throw;
    }
}

/**
 * @brief Apply initial condition perturbations
 * Called once at simulation startup (t=0)
 */
void apply_chaos_initial_conditions() 
{
    if (!chaos_scheme) 
    {
        tmv::log_warn("Chaos scheme not initialized");
        return;
    }

    try {
        SimulationState sim_state;
        sim_state.u = &u;
        sim_state.v = &v;
        sim_state.w = &w;
        sim_state.theta = &theta;
        sim_state.qv = &qv;
        sim_state.qc = &qc;
        sim_state.qr = &qr;
        sim_state.qi = &qi;
        sim_state.qs = &qs;
        sim_state.qg = &qg;
        sim_state.qh = &qh;

        GridMetrics active_grid = (global_grid_metrics.dx > 0.0) ? global_grid_metrics : fallback_grid_metrics();
        
        chaos_scheme->apply_initial_conditions(
            global_chaos_config,
            active_grid,
            sim_state,
            &global_chaos_diagnostics
        );

        const int sanitized_state = sanitize_chaos_initial_state(sim_state);
        if (sanitized_state > 0)
        {
            tmv::log_warn("[CHAOS GUARD] sanitized initial-condition state values: ",
                         sanitized_state);
        }

        if (!global_chaos_diagnostics.warnings.empty())
        {
            tmv::log_info("Chaos IC warnings:");
            for (const auto& warning : global_chaos_diagnostics.warnings)
            {
                tmv::log_info("  ", warning);
            }
        }
        
        if (log_debug_enabled())
        {
            float theta_min = 1e10, theta_max = -1e10;
            int nan_count = 0, inf_count = 0;
            for (int i = 0; i < NR; ++i) {
                for (int j = 0; j < NTH; ++j) {
                    for (int k = 0; k < NZ; ++k) {
                        float theta_val = theta[i][j][k];
                        if (std::isnan(theta_val)) nan_count++;
                        if (std::isinf(theta_val)) inf_count++;
                        if (theta_val < theta_min) theta_min = theta_val;
                        if (theta_val > theta_max) theta_max = theta_val;
                    }
                }
            }
            tmv::log_debug("[CHAOS DEBUG] After chaos perturbations:");
            tmv::log_debug("  Theta: min=", theta_min, "K, max=", theta_max, "K");
            tmv::log_debug("  NaN count: ", nan_count, ", Inf count: ", inf_count);
            if (theta_min < 0 || theta_max > 500) {
                tmv::log_warn("  Theta corrupted by chaos perturbations!");
            }
        }

    }
    catch (const std::exception& e) 
    {
        tmv::log_error("Error applying chaos IC perturbations: ", e.what());
        throw;
    }
}

/**
 * @brief Apply tendency perturbations
 * Called each physics timestep before tendencies are applied
 */
void apply_chaos_tendencies() 
{
    if (!chaos_scheme) 
    {
        return;
    }

    try 
    {
        if (!block_enabled(global_chaos_config, "pbl"))
        {
            return;
        }

        if (boundary_layer_updated_this_step())
        {
            refresh_pbl_base_tendencies();
        }
        if (!pbl_base_tendencies_valid)
        {
            return;
        }

        if (!cadence_due(pbl_chaos_call_count))
        {
            return;
        }

        restore_pbl_tendencies_from_base();
        if (!pbl_base_tendencies_valid)
        {
            return;
        }

        GridMetrics active_grid = (global_grid_metrics.dx > 0.0) ? global_grid_metrics : fallback_grid_metrics();
        chaos::ChaosStateView state_view = make_state_view(active_grid);

        chaos::ChaosTendencies tendencies;
        tendencies.du_pbl_dt = &du_dt_pbl;
        tendencies.dv_pbl_dt = &dv_dt_pbl;
        tendencies.dtheta_pbl_dt = &dtheta_dt_pbl;
        tendencies.dqv_pbl_dt = &dqv_dt_pbl;

        chaos_scheme->apply_tendencies(
            global_chaos_config,
            active_grid,
            state_view,
            tendencies,
            &global_chaos_diagnostics
        );

        const int sanitized_tendencies = sanitize_chaos_tendencies(tendencies);
        if (sanitized_tendencies > 0)
        {
            tmv::log_warn("[CHAOS GUARD] sanitized non-finite PBL tendencies: ",
                         sanitized_tendencies);
        }

    } 
    catch (const std::exception& e) 
    {
        tmv::log_error("Error applying chaos tendency perturbations: ", e.what());
	}
}

/**
 * @brief Applies chaos perturbations to microphysics tendency fields.
 */
void apply_chaos_to_microphysics_tendencies(
    Field3D& dtheta_dt,
    Field3D& dqv_dt,
    Field3D& dqc_dt,
    Field3D& dqr_dt,
    Field3D& dqi_dt,
    Field3D& dqs_dt,
    Field3D& dqg_dt,
    Field3D& dqh_dt
)
{
    if (!chaos_scheme || !block_enabled(global_chaos_config, "microphysics"))
    {
        return;
    }

    try
    {
        if (!cadence_due(microphysics_chaos_call_count))
        {
            return;
        }

        GridMetrics active_grid = (global_grid_metrics.dx > 0.0) ? global_grid_metrics : fallback_grid_metrics();
        chaos::ChaosStateView state_view = make_state_view(active_grid);

        chaos::ChaosTendencies tendencies;
        tendencies.dtheta_micro_dt = &dtheta_dt;
        tendencies.dqv_micro_dt = &dqv_dt;
        tendencies.dqc_micro_dt = &dqc_dt;
        tendencies.dqr_micro_dt = &dqr_dt;
        tendencies.dqi_micro_dt = &dqi_dt;
        tendencies.dqs_micro_dt = &dqs_dt;
        tendencies.dqg_micro_dt = &dqg_dt;
        tendencies.dqh_micro_dt = &dqh_dt;

        chaos_scheme->apply_tendencies(
            global_chaos_config,
            active_grid,
            state_view,
            tendencies,
            &global_chaos_diagnostics
        );

        const int sanitized_tendencies = sanitize_chaos_tendencies(tendencies);
        if (sanitized_tendencies > 0)
        {
            tmv::log_warn("[CHAOS GUARD] sanitized non-finite microphysics tendencies: ",
                         sanitized_tendencies);
        }
    }
    catch (const std::exception& e)
    {
        tmv::log_error("Error applying chaos microphysics perturbations: ", e.what());
	}
}

/**
 * @brief Applies chaos perturbations to turbulence/diffusion tendencies.
 */
void apply_chaos_to_turbulence_tendencies(TurbulenceTendencies& tendencies_in)
{
    if (!chaos_scheme || !block_enabled(global_chaos_config, "diffusion"))
    {
        return;
    }

    try
    {
        if (!cadence_due(turbulence_chaos_call_count))
        {
            return;
        }

        GridMetrics active_grid = (global_grid_metrics.dx > 0.0) ? global_grid_metrics : fallback_grid_metrics();
        chaos::ChaosStateView state_view = make_state_view(active_grid);

        chaos::ChaosTendencies tendencies;
        tendencies.du_diff_dt = &tendencies_in.dudt_sgs;
        tendencies.dv_diff_dt = &tendencies_in.dvdt_sgs;
        tendencies.dw_diff_dt = &tendencies_in.dwdt_sgs;
        tendencies.dtheta_diff_dt = &tendencies_in.dthetadt_sgs;
        tendencies.dqv_diff_dt = &tendencies_in.dqvdt_sgs;

        chaos_scheme->apply_tendencies(
            global_chaos_config,
            active_grid,
            state_view,
            tendencies,
            &global_chaos_diagnostics
        );

        const int sanitized_tendencies = sanitize_chaos_tendencies(tendencies);
        if (sanitized_tendencies > 0)
        {
            tmv::log_warn("[CHAOS GUARD] sanitized non-finite turbulence tendencies: ",
                         sanitized_tendencies);
        }
    }
    catch (const std::exception& e)
    {
        tmv::log_error("Error applying chaos turbulence perturbations: ", e.what());
    }
}

/**
 * @brief Evolve chaos noise fields
 * Called each timestep for schemes with temporal evolution
 * @param dt Timestep size
 */
void step_chaos_noise(double dt) 
{
    if(!chaos_scheme) 
    {
        return;
    }

    try 
    {
        GridMetrics active_grid = (global_grid_metrics.dx > 0.0) ? global_grid_metrics : fallback_grid_metrics();
        chaos_scheme->step_noise(global_chaos_config, active_grid, dt);
    } 
    catch (const std::exception& e) 
    {
        tmv::log_error("Error stepping chaos noise: ", e.what());
    }
}

/**
 * @brief Get current chaos diagnostics
 */
const chaos::ChaosDiagnostics& get_chaos_diagnostics() 
{
    return global_chaos_diagnostics;
}

/**
 * @brief Reset chaos diagnostics
 */
void reset_chaos_diagnostics() 
{
    global_chaos_diagnostics = chaos::ChaosDiagnostics{};
}
