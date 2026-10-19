#include "util/physics_cadence.hpp"
// Stochastic initialization, tendency perturbation, and noise stepping.

#include "core/field/field_bounds.hpp"
#include "core/field/numerical_integrity.hpp"
#include "core/runtime/simulation_state.hpp"
#include "chaos/runtime_internal.hpp"
#include "boundary_layer/runtime.hpp"
#include "turbulence/turbulence_base.hpp"
#include "util/simd_utils.hpp"
#include "util/log.hpp"
#include <iostream>
#include <memory>
#include <stdexcept>
#include <cmath>

namespace tmv::chaos
{

using runtime_detail::ChaosRuntimeView;
using runtime_detail::chaos_runtime_for;
using runtime_detail::grid_metrics_from_state;
using runtime_detail::invalidate_pbl_base_tendencies;
using runtime_detail::normalize_tendency_block_name;

namespace {

Field3D** resolve_initial_state_field(SimulationState& state,
                                      const std::string& name) noexcept
{
    if (name == "u") return &state.u;
    if (name == "v") return &state.v;
    if (name == "w") return &state.w;
    if (name == "theta") return &state.theta;
    if (name == "qv") return &state.qv;
    if (name == "qc") return &state.qc;
    if (name == "qr") return &state.qr;
    if (name == "qi") return &state.qi;
    if (name == "qs") return &state.qs;
    if (name == "qg") return &state.qg;
    if (name == "qh") return &state.qh;
    return nullptr;
}

void refresh_pbl_base_tendencies(ChaosRuntimeView& runtime,
                                 const PhysicsTendencyFieldsView& tendencies)
{
    if (tendencies.du_dt_pbl.empty() ||
        tendencies.dv_dt_pbl.empty() ||
        tendencies.dtheta_dt_pbl.empty() ||
        tendencies.dqv_dt_pbl.empty())
    {
        invalidate_pbl_base_tendencies(runtime);
        return;
    }

    runtime.base_du_dt_pbl = tendencies.du_dt_pbl;
    runtime.base_dv_dt_pbl = tendencies.dv_dt_pbl;
    runtime.base_dtheta_dt_pbl = tendencies.dtheta_dt_pbl;
    runtime.base_dqv_dt_pbl = tendencies.dqv_dt_pbl;
    runtime.pbl_base_tendencies_valid = true;
}

void restore_pbl_tendencies_from_base(ChaosRuntimeView& runtime,
                                      const PhysicsTendencyFieldsView& tendencies)
{
    if (!runtime.pbl_base_tendencies_valid)
    {
        return;
    }

    const bool shape_match =
        runtime.base_du_dt_pbl.size_r() == tendencies.du_dt_pbl.size_r() &&
        runtime.base_du_dt_pbl.size_th() == tendencies.du_dt_pbl.size_th() &&
        runtime.base_du_dt_pbl.size_z() == tendencies.du_dt_pbl.size_z() &&
        runtime.base_dv_dt_pbl.size_r() == tendencies.dv_dt_pbl.size_r() &&
        runtime.base_dv_dt_pbl.size_th() == tendencies.dv_dt_pbl.size_th() &&
        runtime.base_dv_dt_pbl.size_z() == tendencies.dv_dt_pbl.size_z() &&
        runtime.base_dtheta_dt_pbl.size_r() == tendencies.dtheta_dt_pbl.size_r() &&
        runtime.base_dtheta_dt_pbl.size_th() == tendencies.dtheta_dt_pbl.size_th() &&
        runtime.base_dtheta_dt_pbl.size_z() == tendencies.dtheta_dt_pbl.size_z() &&
        runtime.base_dqv_dt_pbl.size_r() == tendencies.dqv_dt_pbl.size_r() &&
        runtime.base_dqv_dt_pbl.size_th() == tendencies.dqv_dt_pbl.size_th() &&
        runtime.base_dqv_dt_pbl.size_z() == tendencies.dqv_dt_pbl.size_z();
    if (!shape_match)
    {
        invalidate_pbl_base_tendencies(runtime);
        return;
    }

    tendencies.du_dt_pbl = runtime.base_du_dt_pbl;
    tendencies.dv_dt_pbl = runtime.base_dv_dt_pbl;
    tendencies.dtheta_dt_pbl = runtime.base_dtheta_dt_pbl;
    tendencies.dqv_dt_pbl = runtime.base_dqv_dt_pbl;
}

struct PblSurfaceFields
{
    std::vector<float>& ustar;
    std::vector<float>& tau_u;
    std::vector<float>& tau_v;
    std::vector<float>& sensible;
    std::vector<float>& latent;
    std::vector<float>& moisture;
};

PblSurfaceFields pbl_surface_fields(const SimulationStateView& state)
{
    auto& process = global_simulation_storage.surface;
    return {
        state.surface.sfc_ustar ? *state.surface.sfc_ustar : process.ustar,
        state.surface.sfc_tau_u ? *state.surface.sfc_tau_u : process.tau_u,
        state.surface.sfc_tau_v ? *state.surface.sfc_tau_v : process.tau_v,
        state.surface.sfc_sensible_heat_flux
            ? *state.surface.sfc_sensible_heat_flux : process.sensible_heat_flux,
        state.surface.sfc_latent_heat_flux
            ? *state.surface.sfc_latent_heat_flux : process.latent_heat_flux,
        state.surface.sfc_moisture_flux
            ? *state.surface.sfc_moisture_flux : process.moisture_flux};
}

BoundaryLayerRuntimeState& pbl_runtime_for(const SimulationStateView& state)
{
    return state.boundary_layer_runtime
        ? *state.boundary_layer_runtime : tmv::require_level_runtime(state.boundary_layer_runtime, "boundary_layer_runtime");
}

bool pbl_conserved_publication_available(const SimulationStateView& state)
{
    const auto& runtime = pbl_runtime_for(state);
    const auto matches = [&](const Field3D& field) {
        return field.size_r() == state.grid.nr &&
               field.size_th() == state.grid.nth &&
               field.size_z() == state.grid.nz;
    };
    const std::size_t surface_size =
        static_cast<std::size_t>(state.grid.nr) *
        static_cast<std::size_t>(state.grid.nth);
    const PblSurfaceFields surface = pbl_surface_fields(state);
    return matches(runtime.held_u_force_n) &&
           matches(runtime.held_v_force_n) &&
           matches(runtime.held_theta_inventory_rate_kgk_s) &&
           matches(runtime.held_vapor_mass_rate_kg_s) &&
           matches(state.tendencies.dtke_dt_pbl) &&
           surface.ustar.size() == surface_size &&
           surface.tau_u.size() == surface_size &&
           surface.tau_v.size() == surface_size &&
           surface.sensible.size() == surface_size &&
           surface.latent.size() == surface_size &&
           surface.moisture.size() == surface_size;
}

void refresh_pbl_conserved_base(ChaosRuntimeView& chaos_runtime,
                                const SimulationStateView& state)
{
    if (!pbl_conserved_publication_available(state))
    {
        invalidate_pbl_base_tendencies(chaos_runtime);
        return;
    }
    const auto& pbl_runtime = pbl_runtime_for(state);
    const PblSurfaceFields surface = pbl_surface_fields(state);
    chaos_runtime.base_du_dt_pbl = pbl_runtime.held_u_force_n;
    chaos_runtime.base_dv_dt_pbl = pbl_runtime.held_v_force_n;
    chaos_runtime.base_dtheta_dt_pbl =
        pbl_runtime.held_theta_inventory_rate_kgk_s;
    chaos_runtime.base_dqv_dt_pbl = pbl_runtime.held_vapor_mass_rate_kg_s;
    chaos_runtime.base_dtke_dt_pbl = state.tendencies.dtke_dt_pbl;
    chaos_runtime.base_sfc_ustar = surface.ustar;
    chaos_runtime.base_sfc_tau_u = surface.tau_u;
    chaos_runtime.base_sfc_tau_v = surface.tau_v;
    chaos_runtime.base_sfc_sensible_heat_flux = surface.sensible;
    chaos_runtime.base_sfc_latent_heat_flux = surface.latent;
    chaos_runtime.base_sfc_moisture_flux = surface.moisture;
    chaos_runtime.pbl_base_tendencies_valid = true;
}

bool restore_pbl_conserved_base(ChaosRuntimeView& chaos_runtime,
                                const SimulationStateView& state)
{
    if (!chaos_runtime.pbl_base_tendencies_valid ||
        !pbl_conserved_publication_available(state))
        return false;
    auto& pbl_runtime = pbl_runtime_for(state);
    PblSurfaceFields surface = pbl_surface_fields(state);
    const auto same_shape = [](const Field3D& lhs, const Field3D& rhs) {
        return lhs.size_r() == rhs.size_r() &&
               lhs.size_th() == rhs.size_th() &&
               lhs.size_z() == rhs.size_z();
    };
    const std::size_t surface_size = surface.ustar.size();
    if (!same_shape(chaos_runtime.base_du_dt_pbl,
                    pbl_runtime.held_u_force_n) ||
        !same_shape(chaos_runtime.base_dv_dt_pbl,
                    pbl_runtime.held_v_force_n) ||
        !same_shape(chaos_runtime.base_dtheta_dt_pbl,
                    pbl_runtime.held_theta_inventory_rate_kgk_s) ||
        !same_shape(chaos_runtime.base_dqv_dt_pbl,
                    pbl_runtime.held_vapor_mass_rate_kg_s) ||
        !same_shape(chaos_runtime.base_dtke_dt_pbl,
                    state.tendencies.dtke_dt_pbl) ||
        chaos_runtime.base_sfc_ustar.size() != surface_size ||
        chaos_runtime.base_sfc_tau_u.size() != surface_size ||
        chaos_runtime.base_sfc_tau_v.size() != surface_size ||
        chaos_runtime.base_sfc_sensible_heat_flux.size() != surface_size ||
        chaos_runtime.base_sfc_latent_heat_flux.size() != surface_size ||
        chaos_runtime.base_sfc_moisture_flux.size() != surface_size)
    {
        invalidate_pbl_base_tendencies(chaos_runtime);
        return false;
    }
    pbl_runtime.held_u_force_n = chaos_runtime.base_du_dt_pbl;
    pbl_runtime.held_v_force_n = chaos_runtime.base_dv_dt_pbl;
    pbl_runtime.held_theta_inventory_rate_kgk_s =
        chaos_runtime.base_dtheta_dt_pbl;
    pbl_runtime.held_vapor_mass_rate_kg_s = chaos_runtime.base_dqv_dt_pbl;
    state.tendencies.dtke_dt_pbl = chaos_runtime.base_dtke_dt_pbl;
    surface.ustar = chaos_runtime.base_sfc_ustar;
    surface.tau_u = chaos_runtime.base_sfc_tau_u;
    surface.tau_v = chaos_runtime.base_sfc_tau_v;
    surface.sensible = chaos_runtime.base_sfc_sensible_heat_flux;
    surface.latent = chaos_runtime.base_sfc_latent_heat_flux;
    surface.moisture = chaos_runtime.base_sfc_moisture_flux;
    return true;
}

SimulationState chaos_initial_state_from_view(const SimulationStateView& state)
{
    SimulationState sim_state;
    sim_state.u = &state.fields.u;
    sim_state.v = &state.fields.v;
    sim_state.w = &state.fields.w;
    sim_state.theta = &state.fields.theta;
    sim_state.qv = &state.fields.qv;
    sim_state.qc = &state.fields.qc;
    sim_state.qr = &state.fields.qr;
    sim_state.qi = &state.fields.qi;
    sim_state.qs = &state.fields.qs;
    sim_state.qg = &state.fields.qg;
    sim_state.qh = &state.fields.qh;
    return sim_state;
}

bool block_enabled(const ChaosConfig& cfg, const std::string& block)
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

ChaosStateView make_state_view(const GridMetrics& active_grid,
                                      const SimulationStateView& state)
{
    ChaosStateView state_view;
    state_view.grid = &active_grid;
    state_view.u = &state.fields.u;
    state_view.v = &state.fields.v;
    state_view.w = &state.fields.w;
    state_view.theta = &state.fields.theta;
    state_view.qv = &state.fields.qv;
    state_view.qc = &state.fields.qc;
    state_view.qr = &state.fields.qr;
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
    if (!primary_state_repairs_enabled)
    {
        int invalid = 0;
        #pragma omp parallel for reduction(+:invalid)
        for (int index = 0; index < count; ++index)
        {
            if (!std::isfinite(static_cast<double>(data[index])))
            {
                ++invalid;
            }
        }
        record_numerical_nonfinite_incidents(
            static_cast<std::size_t>(invalid));
        return invalid;
    }
    const int sanitized = simd_utils::sanitize_nonfinite(data, 0.0f, data, count);
    record_numerical_nonfinite_incidents(
        static_cast<std::size_t>(sanitized));
    return sanitized;
}

int validate_or_sanitize_bounded_field(Field3D* field,
                                       float fallback,
                                       float lower,
                                       float upper)
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
        float new_value;
        if (!std::isfinite(old_value))
        {
            new_value = fallback;
            record_numerical_nonfinite_incidents();
        }
        else
        {
            new_value = std::clamp(old_value, lower, upper);
        }
        if (new_value != old_value)
        {
            ++sanitized;
        }
        if (primary_state_repairs_enabled)
        {
            data[idx] = new_value;
        }
    }
    return sanitized;
}

int sanitize_chaos_tendencies(ChaosTendencies& tendencies)
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
    sanitized += sanitize_nonfinite_field(tendencies.pbl_u_force_n);
    sanitized += sanitize_nonfinite_field(tendencies.pbl_v_force_n);
    sanitized += sanitize_nonfinite_field(
        tendencies.pbl_theta_inventory_rate_kgk_s);
    sanitized += sanitize_nonfinite_field(tendencies.pbl_vapor_mass_rate_kg_s);
    sanitized += sanitize_nonfinite_field(tendencies.pbl_tke_tendency);

    sanitized += sanitize_nonfinite_field(tendencies.du_diff_dt);
    sanitized += sanitize_nonfinite_field(tendencies.dv_diff_dt);
    sanitized += sanitize_nonfinite_field(tendencies.dw_diff_dt);
    sanitized += sanitize_nonfinite_field(tendencies.dtheta_diff_dt);
    sanitized += sanitize_nonfinite_field(tendencies.dqv_diff_dt);
    sanitized += sanitize_nonfinite_field(tendencies.eddy_viscosity);
    sanitized += sanitize_nonfinite_field(tendencies.theta_diffusivity);
    sanitized += sanitize_nonfinite_field(tendencies.moisture_diffusivity);
    sanitized += sanitize_nonfinite_field(tendencies.tke_diffusivity);
    return sanitized;
}

int sanitize_chaos_initial_state(SimulationState& state)
{
    int sanitized = 0;
    sanitized += validate_or_sanitize_bounded_field(
        state.u, 0.0f, -wind_horizontal_abs_max_ms, wind_horizontal_abs_max_ms);
    sanitized += validate_or_sanitize_bounded_field(
        state.v, 0.0f, -wind_horizontal_abs_max_ms, wind_horizontal_abs_max_ms);
    sanitized += validate_or_sanitize_bounded_field(
        state.w, 0.0f, -wind_vertical_abs_max_ms, wind_vertical_abs_max_ms);
    sanitized += validate_or_sanitize_bounded_field(
        state.theta, static_cast<float>(theta0), theta_min_k, theta_max_k);
    sanitized += validate_or_sanitize_bounded_field(state.qv, 0.0f, 0.0f, qv_max_kgkg);
    sanitized += validate_or_sanitize_bounded_field(state.qc, 0.0f, 0.0f, hydrometeor_max_kgkg);
    sanitized += validate_or_sanitize_bounded_field(state.qr, 0.0f, 0.0f, hydrometeor_max_kgkg);
    sanitized += validate_or_sanitize_bounded_field(state.qi, 0.0f, 0.0f, hydrometeor_max_kgkg);
    sanitized += validate_or_sanitize_bounded_field(state.qs, 0.0f, 0.0f, hydrometeor_max_kgkg);
    sanitized += validate_or_sanitize_bounded_field(state.qg, 0.0f, 0.0f, hydrometeor_max_kgkg);
    sanitized += validate_or_sanitize_bounded_field(state.qh, 0.0f, 0.0f, hydrometeor_max_kgkg);
    return sanitized;
}

}

// Applied once after deterministic state initialization.
void apply_chaos_initial_conditions(const SimulationStateView& state)
{
    ChaosRuntimeView runtime = chaos_runtime_for(state);
    if (!runtime.scheme)
    {
        throw std::runtime_error("Chaos scheme is not initialized");
    }

    try {
        SimulationState sim_state = chaos_initial_state_from_view(state);
        bind_chaos_compute_backend(state);
        const GridMetrics& active_grid = grid_metrics_from_state(state);

        // Initial-condition perturbations are a transaction. A strict guard
        // must never discover an invalid late field after earlier live fields
        // have already been modified. Stage only the explicitly selected
        // volumes, keeping the one-time startup cost proportional to the
        // configured target list rather than to the complete model state.
        struct StagedField
        {
            std::string name;
            Field3D* live = nullptr;
            Field3D candidate;
        };
        std::vector<StagedField> staged_fields;
        staged_fields.reserve(runtime.config.apply_to_ic.size());
        for (const std::string& field_name : runtime.config.apply_to_ic)
        {
            const bool already_staged = std::any_of(
                staged_fields.begin(), staged_fields.end(),
                [&](const StagedField& staged)
                { return staged.name == field_name; });
            if (already_staged)
            {
                continue;
            }
            Field3D** slot = resolve_initial_state_field(sim_state, field_name);
            if (slot == nullptr || *slot == nullptr)
            {
                continue;
            }
            staged_fields.push_back({field_name, *slot, **slot});
            *slot = &staged_fields.back().candidate;
        }
        ChaosDiagnostics staged_diagnostics = runtime.diagnostics;

        runtime.scheme->apply_initial_conditions(
            runtime.config,
            active_grid,
            sim_state,
            &staged_diagnostics
        );

        const int sanitized_state = sanitize_chaos_initial_state(sim_state);
        if (sanitized_state > 0)
        {
            if (primary_state_repairs_fail_closed)
            {
                throw std::runtime_error(
                    "Chaos initial-condition perturbation produced " +
                    std::to_string(sanitized_state) +
                    " nonphysical state values");
            }
            tmv::log_warn("[CHAOS GUARD] sanitized initial-condition state values: ",
                         sanitized_state);
        }

        for (StagedField& staged : staged_fields)
        {
            *staged.live = std::move(staged.candidate);
        }
        runtime.diagnostics = std::move(staged_diagnostics);

        if (!runtime.diagnostics.warnings.empty())
        {
            tmv::log_info("Chaos IC warnings:");
            for (const auto& warning : runtime.diagnostics.warnings)
            {
                tmv::log_info("  ", warning);
            }
        }

        if (log_debug_enabled())
        {
            float theta_min = 1e10, theta_max = -1e10;
            int nan_count = 0, inf_count = 0;
            for (int i = 0; i < state.grid.nr; ++i) {
                for (int j = 0; j < state.grid.nth; ++j) {
                    for (int k = 0; k < state.grid.nz; ++k) {
                        float theta_val = state.fields.theta[i][j][k];
                        if (std::isnan(theta_val)) nan_count++;
                        if (std::isinf(theta_val)) inf_count++;
                        if (theta_val < theta_min) theta_min = theta_val;
                        if (theta_val > theta_max) theta_max = theta_val;
                    }
                }
            }
            tmv::log_debug("[CHAOS DIAGNOSTIC] theta_min_k=", theta_min,
                           " theta_max_k=", theta_max,
                           " nan_count=", nan_count,
                           " inf_count=", inf_count);
            if (theta_min < theta_min_k || theta_max > theta_max_k) {
                tmv::log_warn("[CHAOS GUARD] theta bounds violated after perturbation");
            }
        }

    }
    catch (const std::exception& e)
    {
        tmv::log_error("Error applying chaos IC perturbations: ", e.what());
        throw;
    }
}

void apply_chaos_initial_conditions()
{
    apply_chaos_initial_conditions(global_simulation_state());
}

// Rebuilds stochastic PBL tendencies from the latest deterministic baseline.
void apply_chaos_tendencies(const SimulationStateView& state)
{
    ChaosRuntimeView runtime = chaos_runtime_for(state);
    if (!runtime.scheme)
    {
        return;
    }

    try
    {
        if (!block_enabled(runtime.config, "pbl"))
        {
            return;
        }

        auto& tendency_fields = state.tendencies;
        const bool conservative_publication =
            pbl_conserved_publication_available(state);
        if (conservative_publication)
        {
            if (boundary_layer_updated_this_step(state) ||
                !runtime.pbl_base_tendencies_valid)
                refresh_pbl_conserved_base(runtime, state);
            if (!restore_pbl_conserved_base(runtime, state)) return;
        }
        else
        {
            // Compatibility for direct unit/legacy callers that do not own
            // the production boundary-layer conserved-source publication.
            if (boundary_layer_updated_this_step(state))
                refresh_pbl_base_tendencies(runtime, tendency_fields);
            if (!runtime.pbl_base_tendencies_valid) return;
            restore_pbl_tendencies_from_base(runtime, tendency_fields);
            if (!runtime.pbl_base_tendencies_valid) return;
        }

        bind_chaos_compute_backend(state);
        const GridMetrics& active_grid = grid_metrics_from_state(state);
        ChaosStateView state_view = make_state_view(active_grid, state);

        ChaosTendencies tendencies;
        if (conservative_publication)
        {
            auto& pbl_runtime = pbl_runtime_for(state);
            PblSurfaceFields surface = pbl_surface_fields(state);
            tendencies.pbl_u_force_n = &pbl_runtime.held_u_force_n;
            tendencies.pbl_v_force_n = &pbl_runtime.held_v_force_n;
            tendencies.pbl_theta_inventory_rate_kgk_s =
                &pbl_runtime.held_theta_inventory_rate_kgk_s;
            tendencies.pbl_vapor_mass_rate_kg_s =
                &pbl_runtime.held_vapor_mass_rate_kg_s;
            tendencies.pbl_tke_tendency = &tendency_fields.dtke_dt_pbl;
            tendencies.pbl_ustar = &surface.ustar;
            tendencies.pbl_tau_u = &surface.tau_u;
            tendencies.pbl_tau_v = &surface.tau_v;
            tendencies.pbl_sensible_heat_flux = &surface.sensible;
            tendencies.pbl_latent_heat_flux = &surface.latent;
            tendencies.pbl_moisture_flux = &surface.moisture;
        }
        else
        {
            tendencies.du_pbl_dt = &tendency_fields.du_dt_pbl;
            tendencies.dv_pbl_dt = &tendency_fields.dv_dt_pbl;
            tendencies.dtheta_pbl_dt = &tendency_fields.dtheta_dt_pbl;
            tendencies.dqv_pbl_dt = &tendency_fields.dqv_dt_pbl;
        }

        runtime.scheme->apply_tendencies(
            runtime.config,
            active_grid,
            state_view,
            tendencies,
            &runtime.diagnostics
        );

        const int sanitized_tendencies = sanitize_chaos_tendencies(tendencies);
        if (sanitized_tendencies > 0)
        {
            if (primary_state_repairs_fail_closed)
            {
                throw std::runtime_error("Chaos produced non-finite PBL tendencies");
            }
            tmv::log_warn("[CHAOS GUARD] sanitized non-finite PBL tendencies: ",
                         sanitized_tendencies);
        }

    }
    catch (const std::exception& e)
    {
        tmv::log_error("Error applying chaos tendency perturbations: ", e.what());
        throw;
	}
}

void apply_chaos_tendencies()
{
    apply_chaos_tendencies(global_simulation_state());
}

void apply_chaos_to_microphysics_tendencies(
    const SimulationStateView& state,
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
    ChaosRuntimeView runtime = chaos_runtime_for(state);
    if (!runtime.scheme || !block_enabled(runtime.config, "microphysics"))
    {
        return;
    }

    try
    {
        bind_chaos_compute_backend(state);
        const GridMetrics& active_grid = grid_metrics_from_state(state);
        ChaosStateView state_view = make_state_view(active_grid, state);

        ChaosTendencies tendencies;
        tendencies.dtheta_micro_dt = &dtheta_dt;
        tendencies.dqv_micro_dt = &dqv_dt;
        tendencies.dqc_micro_dt = &dqc_dt;
        tendencies.dqr_micro_dt = &dqr_dt;
        tendencies.dqi_micro_dt = &dqi_dt;
        tendencies.dqs_micro_dt = &dqs_dt;
        tendencies.dqg_micro_dt = &dqg_dt;
        tendencies.dqh_micro_dt = &dqh_dt;

        runtime.scheme->apply_tendencies(
            runtime.config,
            active_grid,
            state_view,
            tendencies,
            &runtime.diagnostics
        );

        const int sanitized_tendencies = sanitize_chaos_tendencies(tendencies);
        if (sanitized_tendencies > 0)
        {
            if (primary_state_repairs_fail_closed)
            {
                throw std::runtime_error(
                    "Chaos produced non-finite microphysics tendencies");
            }
            tmv::log_warn("[CHAOS GUARD] sanitized non-finite microphysics tendencies: ",
                         sanitized_tendencies);
        }
    }
    catch (const std::exception& e)
    {
        tmv::log_error("Error applying chaos microphysics perturbations: ", e.what());
        throw;
	}
}

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
    std::vector<float>& column_multipliers)
{
    const std::size_t surface_size =
        static_cast<std::size_t>(state.grid.nr)
        * static_cast<std::size_t>(state.grid.nth);
    column_multipliers.assign(surface_size, 1.0f);

    ChaosRuntimeView runtime = chaos_runtime_for(state);
    if (!runtime.scheme || !block_enabled(runtime.config, "microphysics"))
    {
        return false;
    }

    try
    {
        bind_chaos_compute_backend(state);
        const GridMetrics& active_grid = grid_metrics_from_state(state);
        ChaosStateView state_view = make_state_view(active_grid, state);
        ChaosTendencies tendencies;
        tendencies.dtheta_micro_dt = &dtheta_dt;
        tendencies.dqv_micro_dt = &dqv_dt;
        tendencies.dqc_micro_dt = &dqc_dt;
        tendencies.dqr_micro_dt = &dqr_dt;
        tendencies.dqi_micro_dt = &dqi_dt;
        tendencies.dqs_micro_dt = &dqs_dt;
        tendencies.dqg_micro_dt = &dqg_dt;
        tendencies.dqh_micro_dt = &dqh_dt;
        tendencies.microphysics_column_multipliers = &column_multipliers;
        runtime.scheme->apply_tendencies(
            runtime.config, active_grid, state_view, tendencies,
            &runtime.diagnostics);

        if (column_multipliers.size() != surface_size)
        {
            throw std::runtime_error(
                "Chaos produced a microphysics column multiplier with the wrong shape");
        }
        for (float multiplier : column_multipliers)
        {
            if (!std::isfinite(multiplier) || multiplier < 0.0f)
            {
                throw std::runtime_error(
                    "Chaos produced an invalid microphysics column multiplier");
            }
        }
        return true;
    }
    catch (const std::exception& e)
    {
        tmv::log_error(
            "Error preparing chaos microphysics column multipliers: ",
            e.what());
        throw;
    }
}

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
    apply_chaos_to_microphysics_tendencies(
        global_simulation_state(),
        dtheta_dt,
        dqv_dt,
        dqc_dt,
        dqr_dt,
        dqi_dt,
        dqs_dt,
        dqg_dt,
        dqh_dt);
}

void apply_chaos_to_turbulence_tendencies(const SimulationStateView& state,
                                          TurbulenceTendencies& tendencies_in)
{
    ChaosRuntimeView runtime = chaos_runtime_for(state);
    if (!runtime.scheme || !block_enabled(runtime.config, "diffusion"))
    {
        return;
    }

    // From this point the scheme may mutate host transport coefficients or
    // (for a non-SGS compatibility caller) tendency fields. Invalidate the
    // reuse hint before calling user-selectable chaos code so exceptions
    // cannot leave stale device ownership.
    tendencies_in.resident_source_unchanged = false;

    try
    {
        bind_chaos_compute_backend(state);
        const GridMetrics& active_grid = grid_metrics_from_state(state);
        ChaosStateView state_view = make_state_view(active_grid, state);

        TurbulenceRuntimeState& active_turbulence = state.turbulence_runtime
            ? *state.turbulence_runtime : tmv::require_level_runtime(state.turbulence_runtime, "turbulence_runtime");
        TurbulenceTransportFields transport;
        const bool active_conservative_sgs = active_turbulence.scheme
            && (active_turbulence.scheme->name() == "smagorinsky"
                || active_turbulence.scheme->name() == "tke");
        const bool coefficient_space = active_conservative_sgs
            && active_turbulence.scheme->mutable_realized_transport_fields(
                transport);
        if (active_conservative_sgs && !coefficient_space)
        {
            throw std::runtime_error(
                "stochastic SGS transport requires host-current realized "
                "viscosity and diffusivity fields");
        }
        if (active_conservative_sgs
            && (&tendencies_in != &active_turbulence.held_tendencies
                || !active_turbulence.realized_transport_tendency_current))
        {
            throw std::runtime_error(
                "stochastic SGS transport requires the current closure-owned "
                "tendency publication");
        }
        if (active_conservative_sgs
            && active_turbulence.stochastic_transport_applied_to_current)
        {
            throw std::runtime_error(
                "stochastic SGS transport was already applied to the current "
                "closure generation");
        }
        if (active_conservative_sgs)
            active_turbulence.realized_transport_tendency_current = false;

        ChaosTendencies tendencies;
        if (coefficient_space)
        {
            tendencies.eddy_viscosity = transport.eddy_viscosity;
            tendencies.theta_diffusivity = transport.theta_diffusivity;
            tendencies.moisture_diffusivity = transport.moisture_diffusivity;
            tendencies.tke_diffusivity = transport.tke_diffusivity;
        }
        else
        {
            // Compatibility for callers without an active SGS closure. No
            // accepted nonlinear turbulence path is allowed through this
            // non-conservative representation.
            tendencies.du_diff_dt = &tendencies_in.dudt_sgs;
            tendencies.dv_diff_dt = &tendencies_in.dvdt_sgs;
            tendencies.dw_diff_dt = &tendencies_in.dwdt_sgs;
            tendencies.dtheta_diff_dt = &tendencies_in.dthetadt_sgs;
            tendencies.dqv_diff_dt = &tendencies_in.dqvdt_sgs;
        }

        runtime.scheme->apply_tendencies(
            runtime.config,
            active_grid,
            state_view,
            tendencies,
            &runtime.diagnostics
        );

        const int sanitized_tendencies = sanitize_chaos_tendencies(tendencies);
        if (sanitized_tendencies > 0)
        {
            if (primary_state_repairs_fail_closed)
            {
                throw std::runtime_error(
                    "Chaos produced non-finite turbulence tendencies");
            }
            tmv::log_warn("[CHAOS GUARD] sanitized non-finite turbulence tendencies: ",
                         sanitized_tendencies);
        }
        TurbulenceStateView turbulence_state;
        tmv::bind_flow_execution(turbulence_state, state);
        turbulence_state.u = &state.fields.u;
        turbulence_state.v = &state.fields.v;
        turbulence_state.w = &state.fields.w;
        turbulence_state.rho = &state.fields.rho;
        turbulence_state.theta = &state.fields.theta;
        turbulence_state.qv = &state.fields.qv;
        turbulence_state.tke = &state.fields.tke;
        turbulence_state.rho0 = &state.base.rho0;
        turbulence_state.c_grid = state.grid.geometry.staggered;
        turbulence_state.NR = state.grid.nr;
        turbulence_state.NTH = state.grid.nth;
        turbulence_state.NZ = state.grid.nz;
        if (coefficient_space
            && !active_turbulence.scheme->rebuild_from_realized_transport(
                active_turbulence.config, active_grid, turbulence_state,
                tendencies_in))
        {
            throw std::runtime_error(
                "turbulence scheme could not rebuild conservative tendencies "
                "from stochastic realized transport coefficients");
        }
        if (coefficient_space)
        {
            active_turbulence.realized_transport_tendency_current = true;
            active_turbulence.stochastic_transport_applied_to_current = true;
        }
    }
    catch (const std::exception& e)
    {
        tmv::log_error("Error applying chaos turbulence perturbations: ", e.what());
        throw;
    }
}

void apply_chaos_to_turbulence_tendencies(TurbulenceTendencies& tendencies_in)
{
    apply_chaos_to_turbulence_tendencies(global_simulation_state(), tendencies_in);
}

// Advances stochastic noise on its configured cadence.
void step_chaos_noise(const SimulationStateView& state, double dt)
{
    ChaosRuntimeView runtime = chaos_runtime_for(state);
    if (!runtime.scheme)
    {
        throw std::runtime_error("Chaos scheme is not initialized");
    }
    if (!std::isfinite(dt) || dt <= 0.0)
    {
        throw std::invalid_argument("Chaos timestep must be positive and finite");
    }
    if (!std::isfinite(runtime.config.dt_chaos) || runtime.config.dt_chaos <= 0.0)
    {
        throw std::invalid_argument("Chaos cadence must be positive and finite");
    }

    const double interval = runtime.config.dt_chaos;
    require_physics_cadence(interval, "chaos cadence");
    if (!std::isfinite(runtime.noise_elapsed_s) || runtime.noise_elapsed_s < -physics_clock_tolerance_s)
        throw std::invalid_argument("chaos elapsed time must be finite and nonnegative");
    double candidate_elapsed = runtime.noise_elapsed_s + dt;
    const double due = std::floor((candidate_elapsed + physics_clock_tolerance_s) / interval);
    if (!std::isfinite(candidate_elapsed) || candidate_elapsed <= runtime.noise_elapsed_s ||
        !std::isfinite(due) || due < 0.0 || due > 1000000.0)
        throw std::invalid_argument("chaos interval cannot make bounded clock progress");
    const bool move_origin = state.domain_frame != nullptr &&
        (state.domain_frame->velocity_ms.u_ms != 0.0 || state.domain_frame->velocity_ms.v_ms != 0.0) &&
        (runtime.scheme->name() == "boundary_layer" || runtime.scheme->name() == "full_stochastic");
    bind_chaos_compute_backend(state);
    const GridMetrics& active_grid = grid_metrics_from_state(state);
    std::unique_ptr<ChaosScheme> previous;
    if (move_origin || due > 0.0)
    {
        previous = runtime.scheme->clone();
        if (!previous) throw std::runtime_error("chaos scheme does not support transactional evolution");
    }
    try
    {
        if (move_origin)
        {
            const auto origin = state.domain_frame->earth_origin_at(state.time_s);
            std::string error;
            if (!runtime.scheme->set_continuous_horizontal_earth_origin(
                    origin.x_m, origin.y_m, runtime.config, active_grid, error))
                throw std::runtime_error("continuous moving-frame chaos reconstruction failed: " + error);
        }
        for (std::uint64_t step = 0; step < static_cast<std::uint64_t>(due); ++step)
        {
            runtime.scheme->step_noise(runtime.config, active_grid, interval);
            candidate_elapsed -= interval;
        }
        runtime.noise_elapsed_s = std::abs(candidate_elapsed) <= physics_clock_tolerance_s ? 0.0 : candidate_elapsed;
    }
    catch (...)
    {
        if (previous) runtime.scheme = std::move(previous);
        throw;
    }
}

void step_chaos_noise(double dt)
{
    step_chaos_noise(global_simulation_state(), dt);
}

const ChaosDiagnostics& get_chaos_diagnostics()
{
    return global_chaos_runtime_state.diagnostics;
}

const ChaosDiagnostics& get_chaos_diagnostics(const SimulationStateView& state)
{
    return state.chaos_runtime
        ? state.chaos_runtime->diagnostics
        : tmv::require_level_runtime(state.chaos_runtime, "chaos_runtime").diagnostics;
}

} // namespace tmv::chaos
