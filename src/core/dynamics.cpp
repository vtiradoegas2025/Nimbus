/**
 * @file dynamics.cpp
 * @brief Core runtime implementation for the tornado model.
 *
 * Provides simulation orchestration and subsystem integration
 * for dynamics, numerics, physics, and runtime execution paths.
 * This file belongs to the primary src/core execution layer.
 */

#include "core/simulation.hpp"
#include "numerics/diffusion_base.hpp"
#include "physics/turbulence_base.hpp"
#include "dynamics/factory.hpp"
#include "diagnostics/field_contract.hpp"
#include "util/log.hpp"
#include <algorithm>
#include <cctype>
#include <cmath>
#include <exception>
#include <iostream>
#include <limits>
#include <string>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif



std::unique_ptr<DynamicsScheme> dynamics_scheme = nullptr;

Field3D vorticity_r;
Field3D vorticity_theta;
Field3D vorticity_z;
Field3D stretching_term;
Field3D tilting_term;
Field3D baroclinic_term;

Field3D angular_momentum;
Field3D angular_momentum_tendency;

Field3D p_prime;
Field3D dynamic_pressure;
Field3D buoyancy_pressure;

namespace
{
Field3D du_r_dt_buf;
Field3D du_theta_dt_buf;
Field3D du_z_dt_buf;
Field3D drho_dt_buf;
Field3D dp_dt_buf;
TurbulenceTendencies turb_tend_buf;
DiffusionTendencies diffusion_tend_buf;
DiffusionDiagnostics diffusion_diag_buf;

inline bool field_matches_domain(const Field3D& f)
{
    return f.size_r() == NR && f.size_th() == NTH && f.size_z() == NZ;
}

void ensure_dynamics_tendency_buffers()
{
    if (!field_matches_domain(du_r_dt_buf)) du_r_dt_buf.resize(NR, NTH, NZ, 0.0f);
    if (!field_matches_domain(du_theta_dt_buf)) du_theta_dt_buf.resize(NR, NTH, NZ, 0.0f);
    if (!field_matches_domain(du_z_dt_buf)) du_z_dt_buf.resize(NR, NTH, NZ, 0.0f);
    if (!field_matches_domain(drho_dt_buf)) drho_dt_buf.resize(NR, NTH, NZ, 0.0f);
    if (!field_matches_domain(dp_dt_buf)) dp_dt_buf.resize(NR, NTH, NZ, 0.0f);
}

void ensure_diffusion_tendency_buffers()
{
    auto ensure = [](Field3D& field)
    {
        if (field.size_r() != NR || field.size_th() != NTH || field.size_z() != NZ)
        {
            field.resize(NR, NTH, NZ, 0.0f);
        }
        else
        {
            field.fill(0.0f);
        }
    };

    ensure(diffusion_tend_buf.dudt_diff);
    ensure(diffusion_tend_buf.dvdt_diff);
    ensure(diffusion_tend_buf.dwdt_diff);
    ensure(diffusion_tend_buf.dthetadt_diff);
    ensure(diffusion_tend_buf.dqvdt_diff);
}

std::string lower_copy(std::string value)
{
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return value;
}

bool diffusion_applies_to_momentum()
{
    const std::string apply_to = lower_copy(global_diffusion_config.apply_to);
    return apply_to == "momentum" || apply_to == "all";
}

bool diffusion_applies_to_scalars()
{
    const std::string apply_to = lower_copy(global_diffusion_config.apply_to);
    return apply_to == "scalars" || apply_to == "all";
}

bool diffusion_runtime_enabled()
{
    if (!diffusion_scheme)
    {
        return false;
    }
    const double K_h = std::max(0.0, global_diffusion_config.K_h);
    const double K_v = std::max(0.0, global_diffusion_config.K_v);
    if (K_h <= 0.0 && K_v <= 0.0 && !global_diffusion_config.use_variable_K)
    {
        return false;
    }

    return diffusion_applies_to_momentum() || diffusion_applies_to_scalars();
}

void log_runtime_diffusion_path_once()
{
    static bool logged = false;
    if (logged)
    {
        return;
    }
    logged = true;

    if (!diffusion_scheme)
    {
        tmv::log_info("[DIFFUSION] Runtime path: disabled (no diffusion scheme).");
        return;
    }

    tmv::log_info("[DIFFUSION] Runtime path: src/numerics/diffusion/", diffusion_scheme->name(),
                  " (apply_to=", global_diffusion_config.apply_to,
                  ", K_h=", global_diffusion_config.K_h,
                  ", K_v=", global_diffusion_config.K_v, ")");
}

void apply_runtime_diffusion(double dt_dynamics)
{
    if (!diffusion_runtime_enabled())
    {
        return;
    }

    ensure_diffusion_tendency_buffers();
    log_runtime_diffusion_path_once();

    const bool do_momentum = diffusion_applies_to_momentum();
    const bool do_scalars = diffusion_applies_to_scalars();

    DiffusionConfig runtime_cfg = global_diffusion_config;
    runtime_cfg.dt_diffusion = std::max(1.0e-6, dt_dynamics);

    DiffusionStateView state{};
    state.grid = &global_grid_metrics;
    state.rho = &rho;
    if (do_momentum)
    {
        state.u = &u;
        state.v = &v_theta;
        state.w = &w;
    }
    if (do_scalars)
    {
        state.theta = &theta;
        state.qv = &qv;
    }

    try
    {
        diffusion_scheme->compute_diffusion_tendencies(runtime_cfg, state, diffusion_tend_buf, &diffusion_diag_buf);
    }
    catch (const std::exception& e)
    {
        tmv::log_error("[DIFFUSION] tendency computation failed: ", e.what());
        return;
    }

    const bool shape_ok =
        field_matches_domain(diffusion_tend_buf.dudt_diff) &&
        field_matches_domain(diffusion_tend_buf.dvdt_diff) &&
        field_matches_domain(diffusion_tend_buf.dwdt_diff) &&
        field_matches_domain(diffusion_tend_buf.dthetadt_diff) &&
        field_matches_domain(diffusion_tend_buf.dqvdt_diff);
    if (!shape_ok)
    {
        tmv::log_error("[DIFFUSION] tendency buffers have invalid shape; skipping diffusion update.");
        return;
    }

    if (do_momentum)
    {
        #pragma omp parallel for collapse(2)
        for (int i = 0; i < NR; ++i)
        {
            for (int j = 0; j < NTH; ++j)
            {
                for (int k = 0; k < NZ; ++k)
                {
                    float dudt = diffusion_tend_buf.dudt_diff[i][j][k];
                    float dvdt = diffusion_tend_buf.dvdt_diff[i][j][k];
                    float dwdt = diffusion_tend_buf.dwdt_diff[i][j][k];
                    if (!std::isfinite(dudt)) dudt = 0.0f;
                    if (!std::isfinite(dvdt)) dvdt = 0.0f;
                    if (!std::isfinite(dwdt)) dwdt = 0.0f;

                    float u_new = u[i][j][k] + dudt * static_cast<float>(dt_dynamics);
                    float v_new = v_theta[i][j][k] + dvdt * static_cast<float>(dt_dynamics);
                    float w_new = w[i][j][k] + dwdt * static_cast<float>(dt_dynamics);

                    if (!std::isfinite(u_new)) u_new = 0.0f;
                    if (!std::isfinite(v_new)) v_new = 0.0f;
                    if (!std::isfinite(w_new)) w_new = 0.0f;

                    u[i][j][k] = clamp_wind_horizontal_ms(u_new);
                    v_theta[i][j][k] = clamp_wind_horizontal_ms(v_new);
                    w[i][j][k] = clamp_wind_vertical_ms(w_new);
                }
            }
        }
    }

    if (do_scalars)
    {
        #pragma omp parallel for collapse(2)
        for (int i = 0; i < NR; ++i)
        {
            for (int j = 0; j < NTH; ++j)
            {
                for (int k = 0; k < NZ; ++k)
                {
                    float dthetadt = diffusion_tend_buf.dthetadt_diff[i][j][k];
                    float dqvdt = diffusion_tend_buf.dqvdt_diff[i][j][k];
                    if (!std::isfinite(dthetadt)) dthetadt = 0.0f;
                    if (!std::isfinite(dqvdt)) dqvdt = 0.0f;

                    float theta_new = theta[i][j][k] + dthetadt * static_cast<float>(dt_dynamics);
                    float qv_new = qv[i][j][k] + dqvdt * static_cast<float>(dt_dynamics);

                    if (!std::isfinite(theta_new)) theta_new = static_cast<float>(theta0);
                    if (!std::isfinite(qv_new)) qv_new = 0.0f;

                    theta[i][j][k] = clamp_theta_k(theta_new);
                    qv[i][j][k] = clamp_qv_kgkg(qv_new);
                }
            }
        }
    }

    if(lower_copy(global_diffusion_config.scheme_id) == "explicit" && diffusion_diag_buf.max_diffusion_number > 1.0)
    {
        tmv::log_warn("[DIFFUSION] max diffusion number exceeds 1 (",
                      diffusion_diag_buf.max_diffusion_number,
                      "); reduce dt or diffusivity for stronger explicit stability margin.");
    }
}

int sanitize_field_nonfinite_and_bounds(Field3D& field, float min_value, float max_value)
{
    if (field.empty())
    {
        return 0;
    }

    int sanitized = 0;
    float* const data = field.data();
    const std::size_t count = field.size();
    #pragma omp parallel for reduction(+:sanitized)
    for (long long idx = 0; idx < static_cast<long long>(count); ++idx)
    {
        const float old_value = data[idx];
        float new_value = old_value;
        if (!std::isfinite(static_cast<double>(new_value)))
        {
            new_value = 0.0f;
        }
        new_value = std::clamp(new_value, min_value, max_value);
        if (new_value != old_value)
        {
            ++sanitized;
        }
        data[idx] = new_value;
    }
    return sanitized;
}

int sanitize_field_nonfinite_and_contract_bounds(Field3D& field, const char* field_id)
{
    const tmv::FieldContract* contract = tmv::find_field_contract(field_id);
    const bool has_min = (contract != nullptr) && contract->default_bounds.has_min;
    const bool has_max = (contract != nullptr) && contract->default_bounds.has_max;
    const float min_value = has_min
        ? static_cast<float>(contract->default_bounds.min_value)
        : -std::numeric_limits<float>::infinity();
    const float max_value = has_max
        ? static_cast<float>(contract->default_bounds.max_value)
        : std::numeric_limits<float>::infinity();

    if (!has_min && !has_max)
    {
        if (field.empty())
        {
            return 0;
        }

        int sanitized = 0;
        float* const data = field.data();
        const std::size_t count = field.size();
        #pragma omp parallel for reduction(+:sanitized)
        for (long long idx = 0; idx < static_cast<long long>(count); ++idx)
        {
            const float old_value = data[idx];
            if (!std::isfinite(static_cast<double>(old_value)))
            {
                data[idx] = 0.0f;
                ++sanitized;
            }
        }
        return sanitized;
    }

    return sanitize_field_nonfinite_and_bounds(field, min_value, max_value);
}

/**
 * @brief Domain-integrated mass/water/energy budget diagnostics.
 */
struct ConservationBudget
{
    double dry_mass = 0.0;
    double total_water = 0.0;
    double thermal_energy = 0.0;
};

std::vector<double> radial_cell_volume_weights;
int radial_volume_nr = -1;
double radial_volume_dr = std::numeric_limits<double>::quiet_NaN();
double radial_volume_dtheta = std::numeric_limits<double>::quiet_NaN();
double radial_volume_dz = std::numeric_limits<double>::quiet_NaN();

/**
 * @brief Recomputes radial cell-volume weights for conservation diagnostics.
 */
void ensure_radial_cell_volume_weights()
{
    if (radial_volume_nr == NR &&
        radial_volume_dr == dr &&
        radial_volume_dtheta == dtheta &&
        radial_volume_dz == dz &&
        radial_cell_volume_weights.size() == static_cast<std::size_t>(NR))
    {
        return;
    }

    radial_volume_nr = NR;
    radial_volume_dr = dr;
    radial_volume_dtheta = dtheta;
    radial_volume_dz = dz;
    radial_cell_volume_weights.assign(static_cast<std::size_t>(NR), 0.0);

    for (int i = 0; i < NR; ++i)
    {
        const double r_center = std::max((static_cast<double>(i) + 0.5) * dr, 0.5 * dr);
        radial_cell_volume_weights[static_cast<std::size_t>(i)] = r_center * dr * dtheta * dz;
    }
}

/**
 * @brief Computes dry-mass, water, and thermal-energy integrals.
 * @return Domain-integrated conservation budget snapshot.
 */
ConservationBudget compute_conservation_budget()
{
    ConservationBudget budget{};
    if (rho.empty() || p.empty() || theta.empty() || qv.empty() ||
        qc.empty() || qr.empty() || qi.empty() || qs.empty() || qg.empty() || qh.empty())
    {
        return budget;
    }

    ensure_radial_cell_volume_weights();
    constexpr double kappa = R_d / cp;
    double dry_mass = 0.0;
    double total_water = 0.0;
    double thermal_energy = 0.0;

    #pragma omp parallel for reduction(+:dry_mass,total_water,thermal_energy) collapse(2)
    for (int i = 0; i < NR; ++i)
    {
        for (int j = 0; j < NTH; ++j)
        {
            const double cell_volume = radial_cell_volume_weights[static_cast<std::size_t>(i)];
            for (int k = 0; k < NZ; ++k)
            {
                const double rho_val = std::max(0.0, static_cast<double>(rho[i][j][k]));
                const double qv_val = std::max(0.0, static_cast<double>(qv[i][j][k]));
                const double qc_val = std::max(0.0, static_cast<double>(qc[i][j][k]));
                const double qr_val = std::max(0.0, static_cast<double>(qr[i][j][k]));
                const double qi_val = std::max(0.0, static_cast<double>(qi[i][j][k]));
                const double qs_val = std::max(0.0, static_cast<double>(qs[i][j][k]));
                const double qg_val = std::max(0.0, static_cast<double>(qg[i][j][k]));
                const double qh_val = std::max(0.0, static_cast<double>(qh[i][j][k]));
                const double water = qv_val + qc_val + qr_val + qi_val + qs_val + qg_val + qh_val;
                const double dry_fraction = std::max(0.0, 1.0 - water);

                dry_mass += rho_val * dry_fraction * cell_volume;
                total_water += rho_val * water * cell_volume;

                const double p_val = static_cast<double>(p[i][j][k]);
                const double theta_val = static_cast<double>(theta[i][j][k]);
                if (std::isfinite(p_val) && std::isfinite(theta_val) && p_val > 0.0)
                {
                    const double temperature = theta_val * std::pow(p_val / p0, kappa);
                    if (std::isfinite(temperature))
                    {
                        thermal_energy += rho_val * cp * temperature * cell_volume;
                    }
                }
            }
        }
    }

    budget.dry_mass = dry_mass;
    budget.total_water = total_water;
    budget.thermal_energy = thermal_energy;
    return budget;
}

/**
 * @brief Logs per-stage conservation changes when thresholds are exceeded.
 */
void report_budget_transition(const char* stage, const ConservationBudget& before, const ConservationBudget& after, double dt_stage)
{
    if (!std::isfinite(dt_stage) || dt_stage <= 0.0)
    {
        return;
    }

    const double d_dry_mass = after.dry_mass - before.dry_mass;
    const double d_total_water = after.total_water - before.total_water;
    const double d_thermal_energy = after.thermal_energy - before.thermal_energy;

    const double rel_dry = std::abs(d_dry_mass) / std::max(1.0, std::abs(before.dry_mass));
    const double rel_water = std::abs(d_total_water) / std::max(1.0, std::abs(before.total_water));
    const double rel_energy = std::abs(d_thermal_energy) / std::max(1.0, std::abs(before.thermal_energy));

    const bool warn = rel_dry > 5.0e-4 || rel_water > 5.0e-4 || rel_energy > 2.0e-3;

    if (!warn && !log_debug_enabled())
    {
        return;
    }

    if (warn)
    {
        tmv::log_warn("[PHYSICS BUDGET WARN] stage=", stage,
                      " d_dry_mass=", d_dry_mass,
                      " d_total_water=", d_total_water,
                      " d_thermal_energy=", d_thermal_energy,
                      " dry_tendency=", (d_dry_mass / dt_stage),
                      " water_tendency=", (d_total_water / dt_stage),
                      " energy_tendency=", (d_thermal_energy / dt_stage));
    }
    else
    {
        tmv::log_debug("[PHYSICS BUDGET] stage=", stage,
                       " d_dry_mass=", d_dry_mass,
                       " d_total_water=", d_total_water,
                       " d_thermal_energy=", d_thermal_energy,
                       " dry_tendency=", (d_dry_mass / dt_stage),
                       " water_tendency=", (d_total_water / dt_stage),
                       " energy_tendency=", (d_thermal_energy / dt_stage));
    }
}

/**
 * @brief Clamps primary prognostic fields into valid physical ranges.
 * @param stage Label used in guard diagnostics.
 * @return Number of corrected samples.
 */
int enforce_primary_state_bounds(const char* stage)
{
    if (rho.empty() || p.empty() || theta.empty() || qv.empty() ||
        qc.empty() || qr.empty() || qi.empty() || qs.empty() || qg.empty() || qh.empty())
    {
        return 0;
    }

    const std::size_t count = rho.size();
    if (p.size() != count || theta.size() != count ||
        qv.size() != count || qc.size() != count || qr.size() != count ||
        qi.size() != count || qs.size() != count || qg.size() != count || qh.size() != count)
    {
        return 0;
    }

    float* rho_data = rho.data();
    float* p_data = p.data();
    float* theta_data = theta.data();
    float* qv_data = qv.data();
    float* qc_data = qc.data();
    float* qr_data = qr.data();
    float* qi_data = qi.data();
    float* qs_data = qs.data();
    float* qg_data = qg.data();
    float* qh_data = qh.data();

    int corrected = 0;
    #pragma omp parallel for reduction(+:corrected)
    for (long long idx = 0; idx < static_cast<long long>(count); ++idx)
    {
        auto apply = [&](float& value, float bounded) 
        {
            if (bounded != value)
            {
                ++corrected;
                value = bounded;
            }
        };

        apply(rho_data[idx], clamp_density_kgm3(rho_data[idx]));
        apply(p_data[idx], clamp_pressure_pa(p_data[idx]));
        apply(theta_data[idx], clamp_theta_k(theta_data[idx]));
        apply(qv_data[idx], clamp_qv_kgkg(qv_data[idx]));
        apply(qc_data[idx], clamp_hydrometeor_kgkg(qc_data[idx]));
        apply(qr_data[idx], clamp_hydrometeor_kgkg(qr_data[idx]));
        apply(qi_data[idx], clamp_hydrometeor_kgkg(qi_data[idx]));
        apply(qs_data[idx], clamp_hydrometeor_kgkg(qs_data[idx]));
        apply(qg_data[idx], clamp_hydrometeor_kgkg(qg_data[idx]));
        apply(qh_data[idx], clamp_hydrometeor_kgkg(qh_data[idx]));
    }

    if (corrected > 0)
    {
        tmv::log_warn("[PHYSICS GUARD] stage=", stage,
                      " corrected_primary_state_samples=", corrected);
    }
    return corrected;
}
}

/**
 * @brief Initializes the dynamics scheme.
 */
void initialize_dynamics(const std::string& scheme_name) 
{
    try 
    {
        dynamics_scheme = create_dynamics_scheme(scheme_name);
        const std::string active_scheme_name = dynamics_scheme ? dynamics_scheme->get_scheme_name() : scheme_name;
        tmv::log_info("Initialized dynamics scheme: ", active_scheme_name);

        vorticity_r.resize(NR, NTH, NZ, 0.0f);
        vorticity_theta.resize(NR, NTH, NZ, 0.0f);
        vorticity_z.resize(NR, NTH, NZ, 0.0f);
        stretching_term.resize(NR, NTH, NZ, 0.0f);
        tilting_term.resize(NR, NTH, NZ, 0.0f);
        baroclinic_term.resize(NR, NTH, NZ, 0.0f);
        angular_momentum.resize(NR, NTH, NZ, 0.0f);
        angular_momentum_tendency.resize(NR, NTH, NZ, 0.0f);
        p_prime.resize(NR, NTH, NZ, 0.0f);
        dynamic_pressure.resize(NR, NTH, NZ, 0.0f);
        buoyancy_pressure.resize(NR, NTH, NZ, 0.0f);
        ensure_dynamics_tendency_buffers();
        ensure_diffusion_tendency_buffers();

    } 
    catch (const std::runtime_error& e) 
    {
        tmv::log_error("Error initializing dynamics scheme: ", e.what());
        throw;
    }
}

/**
 * @brief Steps the dynamics forward in time.
 */
void step_dynamics_new(double dt_dynamics, double current_time) 
{
    if (!dynamics_scheme) 
    {
        tmv::log_warn("No dynamics scheme initialized, using old dynamics");
        step_dynamics_old(current_time);
        return;
    }
    
    ensure_dynamics_tendency_buffers();
    Field3D& du_r_dt = du_r_dt_buf;
    Field3D& du_theta_dt = du_theta_dt_buf;
    Field3D& du_z_dt = du_z_dt_buf;
    Field3D& drho_dt = drho_dt_buf;
    Field3D& dp_dt = dp_dt_buf;
    const ConservationBudget budget_start = compute_conservation_budget();
    
    dynamics_scheme->compute_momentum_tendencies(
        u, v_theta, w, rho, p, theta, dt_dynamics,
        du_r_dt, du_theta_dt, du_z_dt, drho_dt, dp_dt
    );

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < NR; ++i) 
    {
        for (int j = 0; j < NTH; ++j) 
        {
            for (int k = 0; k < NZ; ++k) 
            {
                float du_total = du_r_dt[i][j][k] + du_dt_pbl[i][j][k];
                float dv_total = du_theta_dt[i][j][k] + dv_dt_pbl[i][j][k];
                float dw_total = du_z_dt[i][j][k];
                float drho_total = drho_dt[i][j][k];
                float dp_total = dp_dt[i][j][k];

                if (!std::isfinite(du_total)) du_total = 0.0f;
                if (!std::isfinite(dv_total)) dv_total = 0.0f;
                if (!std::isfinite(dw_total)) dw_total = 0.0f;
                if (!std::isfinite(drho_total)) drho_total = 0.0f;
                if (!std::isfinite(dp_total)) dp_total = 0.0f;

                float u_new = u[i][j][k] + du_total * dt_dynamics;
                float v_new = v_theta[i][j][k] + dv_total * dt_dynamics;
                float w_new = w[i][j][k] + dw_total * dt_dynamics;
                float rho_new = rho[i][j][k] + drho_total * dt_dynamics;
                float p_new = p[i][j][k] + dp_total * dt_dynamics;

                if (!std::isfinite(u_new)) u_new = 0.0f;
                if (!std::isfinite(v_new)) v_new = 0.0f;
                if (!std::isfinite(w_new)) w_new = 0.0f;
                if (!std::isfinite(rho_new) || rho_new <= 0.0f)
                {
                    rho_new = static_cast<float>(std::max(0.1, rho0_base[k]));
                }
                if (!std::isfinite(p_new) || p_new <= 0.0f)
                {
                    p_new = static_cast<float>(p0);
                }

                u[i][j][k] = clamp_wind_horizontal_ms(u_new);
                v_theta[i][j][k] = clamp_wind_horizontal_ms(v_new);
                w[i][j][k] = clamp_wind_vertical_ms(w_new);
                rho[i][j][k] = clamp_density_kgm3(rho_new);
                p[i][j][k] = clamp_pressure_pa(p_new);
            }
        }
    }
    enforce_primary_state_bounds("dynamics");
    const ConservationBudget budget_after_dynamics = compute_conservation_budget();
    report_budget_transition("dynamics", budget_start, budget_after_dynamics, dt_dynamics);


    advect_tracer(dt_dynamics);
    advect_thermodynamics(dt_dynamics);

    step_microphysics(dt_dynamics);
    enforce_primary_state_bounds("microphysics");
    const ConservationBudget budget_after_microphysics = compute_conservation_budget();
    report_budget_transition("microphysics", budget_after_dynamics, budget_after_microphysics, dt_dynamics);

    calculate_radar_reflectivity();

    TurbulenceTendencies& turb_tend = turb_tend_buf;
    step_turbulence(current_time, turb_tend);
    apply_chaos_to_turbulence_tendencies(turb_tend);
    

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < NR; ++i) 
    {
        for (int j = 0; j < NTH; ++j) 
        {
            for (int k = 0; k < NZ; ++k) 
            {
                float dudt_sgs = turb_tend.dudt_sgs[i][j][k];
                float dvdt_sgs = turb_tend.dvdt_sgs[i][j][k];
                float dwdt_sgs = turb_tend.dwdt_sgs[i][j][k];
                float dthetadt_sgs = turb_tend.dthetadt_sgs[i][j][k];

                if (!std::isfinite(dudt_sgs)) dudt_sgs = 0.0f;
                if (!std::isfinite(dvdt_sgs)) dvdt_sgs = 0.0f;
                if (!std::isfinite(dwdt_sgs)) dwdt_sgs = 0.0f;
                if (!std::isfinite(dthetadt_sgs)) dthetadt_sgs = 0.0f;

                float u_new = u[i][j][k] + dudt_sgs * dt_dynamics;
                float v_new = v_theta[i][j][k] + dvdt_sgs * dt_dynamics;
                float w_new = w[i][j][k] + dwdt_sgs * dt_dynamics;
                if (!std::isfinite(u_new)) u_new = 0.0f;
                if (!std::isfinite(v_new)) v_new = 0.0f;
                if (!std::isfinite(w_new)) w_new = 0.0f;
                u[i][j][k] = clamp_wind_horizontal_ms(u_new);
                v_theta[i][j][k] = clamp_wind_horizontal_ms(v_new);
                w[i][j][k] = clamp_wind_vertical_ms(w_new);

                float theta_new = theta[i][j][k] + dthetadt_sgs * dt_dynamics;
                if (!std::isfinite(theta_new))
                {
                    theta_new = static_cast<float>(theta0);
                }
                theta[i][j][k] = clamp_theta_k(theta_new);

                if (!qv.empty()) 
                {
                    float dqvdt_sgs = turb_tend.dqvdt_sgs[i][j][k];
                    if (!std::isfinite(dqvdt_sgs)) dqvdt_sgs = 0.0f;
                    float qv_new = qv[i][j][k] + dqvdt_sgs * dt_dynamics;
                    if (!std::isfinite(qv_new))
                    {
                        qv_new = 0.0f;
                    }
                    qv[i][j][k] = clamp_qv_kgkg(qv_new);
                }

                if (!tke.empty()) {
                    float dtkedt_sgs = turb_tend.dtkedt_sgs[i][j][k];
                    float dtkedt_pbl = dtke_dt_pbl[i][j][k];
                    if (!std::isfinite(dtkedt_sgs)) dtkedt_sgs = 0.0f;
                    if (!std::isfinite(dtkedt_pbl)) dtkedt_pbl = 0.0f;
                    float tke_new = tke[i][j][k] + (dtkedt_sgs + dtkedt_pbl) * dt_dynamics;
                    float tke_val = tke_new;
                    if (!std::isfinite(tke_val)) tke_val = 0.001f;
                    tke[i][j][k] = std::max(0.001f, tke_val);
                }
            }
        }
    }
    enforce_primary_state_bounds("turbulence");
    const ConservationBudget budget_after_turbulence = compute_conservation_budget();
    report_budget_transition("turbulence", budget_after_microphysics, budget_after_turbulence, dt_dynamics);

    apply_runtime_diffusion(dt_dynamics);
    enforce_primary_state_bounds("diffusion");
    const ConservationBudget budget_after_diffusion = compute_conservation_budget();
    report_budget_transition("diffusion", budget_after_turbulence, budget_after_diffusion, dt_dynamics);

    compute_dynamics_diagnostics();

    apply_boundary_conditions();

    const ConservationBudget budget_final = compute_conservation_budget();
    report_budget_transition("total_step", budget_start, budget_final, dt_dynamics);
}

/**
 * @brief Computes the dynamics diagnostics.
 */
void compute_dynamics_diagnostics() 
{
    if (!dynamics_scheme){return;}

    vorticity_r.fill(0.0f);
    vorticity_theta.fill(0.0f);
    vorticity_z.fill(0.0f);
    stretching_term.fill(0.0f);
    tilting_term.fill(0.0f);
    baroclinic_term.fill(0.0f);
    angular_momentum.fill(0.0f);
    angular_momentum_tendency.fill(0.0f);
    p_prime.fill(0.0f);
    dynamic_pressure.fill(0.0f);
    buoyancy_pressure.fill(0.0f);

    dynamics_scheme->compute_vorticity_diagnostics(u, v_theta, w, rho, p, vorticity_r, vorticity_theta, vorticity_z,
        stretching_term, tilting_term, baroclinic_term);

    dynamics_scheme->compute_angular_momentum(u, v_theta, angular_momentum, angular_momentum_tendency);

    dynamics_scheme->compute_pressure_diagnostics(u, v_theta, w, rho, theta,p_prime, dynamic_pressure, buoyancy_pressure);

    int sanitized = 0;
    sanitized += sanitize_field_nonfinite_and_contract_bounds(vorticity_r, "vorticity_r");
    sanitized += sanitize_field_nonfinite_and_contract_bounds(vorticity_theta, "vorticity_theta");
    sanitized += sanitize_field_nonfinite_and_contract_bounds(vorticity_z, "vorticity_z");
    sanitized += sanitize_field_nonfinite_and_contract_bounds(stretching_term, "stretching_term");
    sanitized += sanitize_field_nonfinite_and_contract_bounds(tilting_term, "tilting_term");
    sanitized += sanitize_field_nonfinite_and_contract_bounds(baroclinic_term, "baroclinic_term");
    sanitized += sanitize_field_nonfinite_and_contract_bounds(angular_momentum, "angular_momentum");
    sanitized += sanitize_field_nonfinite_and_contract_bounds(angular_momentum_tendency, "angular_momentum_tendency");
    sanitized += sanitize_field_nonfinite_and_contract_bounds(p_prime, "p_prime");
    sanitized += sanitize_field_nonfinite_and_contract_bounds(dynamic_pressure, "dynamic_pressure");
    sanitized += sanitize_field_nonfinite_and_contract_bounds(buoyancy_pressure, "buoyancy_pressure");

    if (sanitized > 0)
    {
        tmv::log_debug("[DYNAMICS GUARD] sanitized diagnostic values: ", sanitized);
    }
}

/**
 * @brief Applies radial and vertical boundary conditions to state fields.
 */
void apply_boundary_conditions()
{

    for (int j = 0; j < NTH; ++j) 
    {
        for (int k = 0; k < NZ; ++k) 
        {
            u[0][j][k] = -u[1][j][k];
            u[NR-1][j][k] = -u[NR-2][j][k];
            w[0][j][k] = w[1][j][k];
            w[NR-1][j][k] = w[NR-2][j][k];
            rho[0][j][k] = rho[1][j][k];
            rho[NR-1][j][k] = rho[NR-2][j][k];
            p[0][j][k] = p[1][j][k];
            p[NR-1][j][k] = p[NR-2][j][k];
        }
    }

    for (int i = 0; i < NR; ++i) 
    {
        for (int j = 0; j < NTH; ++j) 
        {
            w[i][j][0] = 0.0f; w[i][j][NZ-1] = 0.0f;
            u[i][j][0] = u[i][j][1]; u[i][j][NZ-1] = u[i][j][NZ-2];
            rho[i][j][0] = rho[i][j][1]; rho[i][j][NZ-1] = rho[i][j][NZ-2];
            p[i][j][0] = p[i][j][1]; p[i][j][NZ-1] = p[i][j][NZ-2];
            theta[i][j][0] = theta[i][j][1]; theta[i][j][NZ-1] = theta[i][j][NZ-2];
            qv[i][j][0] = qv[i][j][1]; qv[i][j][NZ-1] = qv[i][j][NZ-2];
            qc[i][j][0] = qc[i][j][1]; qc[i][j][NZ-1] = qc[i][j][NZ-2];
            qr[i][j][0] = qr[i][j][1]; qr[i][j][NZ-1] = qr[i][j][NZ-2];
            qi[i][j][0] = qi[i][j][1]; qi[i][j][NZ-1] = qi[i][j][NZ-2];
            qs[i][j][0] = qs[i][j][1]; qs[i][j][NZ-1] = qs[i][j][NZ-2];
            qg[i][j][0] = qg[i][j][1]; qg[i][j][NZ-1] = qg[i][j][NZ-2];
            qh[i][j][0] = qh[i][j][1]; qh[i][j][NZ-1] = qh[i][j][NZ-2];
        }
    }

    for (int j = 0; j < NTH; ++j) 
    {
        for (int k = 0; k < NZ; ++k) 
        {
            theta[0][j][k] = theta[1][j][k]; theta[NR-1][j][k] = theta[NR-2][j][k];
            qv[0][j][k] = qv[1][j][k]; qv[NR-1][j][k] = qv[NR-2][j][k];
            qc[0][j][k] = qc[1][j][k]; qc[NR-1][j][k] = qc[NR-2][j][k];
            qr[0][j][k] = qr[1][j][k]; qr[NR-1][j][k] = qr[NR-2][j][k];
            qi[0][j][k] = qi[1][j][k]; qi[NR-1][j][k] = qi[NR-2][j][k];
            qs[0][j][k] = qs[1][j][k]; qs[NR-1][j][k] = qs[NR-2][j][k];
            qg[0][j][k] = qg[1][j][k]; qg[NR-1][j][k] = qg[NR-2][j][k];
            qh[0][j][k] = qh[1][j][k]; qh[NR-1][j][k] = qh[NR-2][j][k];
        }
    }

    enforce_primary_state_bounds("boundary_conditions");
}



/**
 * @brief Steps the dynamics forward in time.
 */
void step_dynamics(double current_time)
{
    if (dynamics_scheme)
    {
        step_dynamics_new(dt, current_time);
        return;
    }

    step_dynamics_old(current_time);
}


/**
 * @brief Steps the dynamics forward in time using the old dynamics system.
 */
void step_dynamics_old(double current_time)
{
    if (dynamics_scheme)
    {
        step_dynamics_new(dt, current_time);
        return;
    }


    double max_speed = 1e-6;

    #pragma omp parallel for collapse(2) reduction(max:max_speed)
    for (int i = 1; i < NR - 1; ++i) 
    {
        for (int j = 0; j < NTH; ++j) 
        {
            for (int k = 1; k < NZ - 1; ++k) 
            {
                double sp = std::max({std::abs((double)u[i][j][k]), std::abs((double)w[i][j][k]), std::abs((double)v_theta[i][j][k])});
                if (sp > max_speed) max_speed = sp; 
            }
        }
    }

    double cfl_num = 0.5;
    double dt_eff = dt;

    if (max_speed > 1e-6) 
    {
        double dl = std::min(dr, dz);
        double cfl_dt = cfl_num * dl / max_speed;
        dt_eff = std::min(dt, cfl_dt);
    }

    for (int i = 0; i < NR; ++i) 
    {
        for (int j = 0; j < NTH; ++j) 
        {
            for (int k = 0; k < NZ; ++k) 
            {
                if (k >= 0 && k < NZ) 
                {
                    float base_rho = rho0_base[k];

                    if (!std::isnan(base_rho) && base_rho > 0 && std::isfinite(base_rho)) 
                    {
                        rho[i][j][k] = base_rho;
                    } 

                    else 
                    {
                        rho[i][j][k] = 1.0f;
                    }
                } 

                else 
                {
                    rho[i][j][k] = 1.0f;
                }

                if (std::isnan(p[i][j][k]) || std::isinf(p[i][j][k])) 
                {
                    p[i][j][k] = p0;
                }
                if (std::isnan(theta[i][j][k]) || std::isinf(theta[i][j][k])) 
                {
                    theta[i][j][k] = theta0;
                }

                if (std::isnan(u[i][j][k]) || std::isinf(u[i][j][k])) 
                {
                    u[i][j][k] = 0.0f;
                }

                if (std::isnan(w[i][j][k]) || std::isinf(w[i][j][k])) 
                {
                    w[i][j][k] = 0.0f;
                }

                if (std::isnan(v_theta[i][j][k]) || std::isinf(v_theta[i][j][k])) 
                {
                    v_theta[i][j][k] = 0.0f;
                }

                p[i][j][k] = clamp_pressure_pa(p[i][j][k]);
                theta[i][j][k] = clamp_theta_k(theta[i][j][k]);
                u[i][j][k] = clamp_wind_horizontal_ms(u[i][j][k]);
                w[i][j][k] = clamp_wind_vertical_ms(w[i][j][k]);
                v_theta[i][j][k] = clamp_wind_horizontal_ms(v_theta[i][j][k]);
            }
        }
    }

    tmv::log_debug("[DYNAMICS] advect_thermodynamics dt_eff=", dt_eff);
    advect_tracer(dt_eff);
    advect_thermodynamics(dt_eff);
    tmv::log_debug("[DYNAMICS] advection complete");
    step_microphysics(dt_eff);
    apply_runtime_diffusion(dt_eff);


    for (int j = 0; j < NTH; ++j) 
    {
        for (int k = 0; k < NZ; ++k) 
        {
            if (!std::isnan(u[1][j][k]) && std::isfinite(u[1][j][k])) 
            {
                u[0][j][k] = -u[1][j][k];
            }

            if (!std::isnan(u[NR-2][j][k]) && std::isfinite(u[NR-2][j][k])) 
            {
                u[NR-1][j][k] = -u[NR-2][j][k];
            }

            if (!std::isnan(w[1][j][k]) && std::isfinite(w[1][j][k])) 
            {
                w[0][j][k] = w[1][j][k];
            }

            if (!std::isnan(w[NR-2][j][k]) && std::isfinite(w[NR-2][j][k])) 
            {
                w[NR-1][j][k] = w[NR-2][j][k];
            }

            rho[0][j][k] = (!std::isnan(rho[1][j][k]) && std::isfinite(rho[1][j][k])) ? rho[1][j][k] : rho0_base[k];
            rho[NR-1][j][k] = (!std::isnan(rho[NR-2][j][k]) && std::isfinite(rho[NR-2][j][k])) ? rho[NR-2][j][k] : rho0_base[k];

            p[0][j][k] = (!std::isnan(p[1][j][k]) && std::isfinite(p[1][j][k])) ? p[1][j][k] : p0;
            p[NR-1][j][k] = (!std::isnan(p[NR-2][j][k]) && std::isfinite(p[NR-2][j][k])) ? p[NR-2][j][k] : p0;
        }
    }

    for (int i = 0; i < NR; ++i) 
    {
        for (int j = 0; j < NTH; ++j) 
        {
            w[i][j][0] = 0.0f;
            w[i][j][NZ-1] = 0.0f;

            u[i][j][0] = (!std::isnan(u[i][j][1]) && std::isfinite(u[i][j][1])) ? u[i][j][1] : 0.0f;
            u[i][j][NZ-1] = (!std::isnan(u[i][j][NZ-2]) && std::isfinite(u[i][j][NZ-2])) ? u[i][j][NZ-2] : 0.0f;

            rho[i][j][0] = (!std::isnan(rho[i][j][1]) && std::isfinite(rho[i][j][1])) ? rho[i][j][1] : rho0_base[0];
            rho[i][j][NZ-1] = (!std::isnan(rho[i][j][NZ-2]) && std::isfinite(rho[i][j][NZ-2])) ? rho[i][j][NZ-2] : rho0_base[NZ-1];

            p[i][j][0] = (!std::isnan(p[i][j][1]) && std::isfinite(p[i][j][1])) ? p[i][j][1] : p0;
            p[i][j][NZ-1] = (!std::isnan(p[i][j][NZ-2]) && std::isfinite(p[i][j][NZ-2])) ? p[i][j][NZ-2] : p0;

            theta[i][j][0] = (!std::isnan(theta[i][j][1]) && std::isfinite(theta[i][j][1])) ? theta[i][j][1] : theta0;
            theta[i][j][NZ-1] = (!std::isnan(theta[i][j][NZ-2]) && std::isfinite(theta[i][j][NZ-2])) ? theta[i][j][NZ-2] : theta0;

            qv[i][j][0] = (!std::isnan(qv[i][j][1]) && std::isfinite(qv[i][j][1])) ? qv[i][j][1] : 0.0f;
            qv[i][j][NZ-1] = (!std::isnan(qv[i][j][NZ-2]) && std::isfinite(qv[i][j][NZ-2])) ? qv[i][j][NZ-2] : 0.0f;

            qc[i][j][0] = (!std::isnan(qc[i][j][1]) && std::isfinite(qc[i][j][1])) ? qc[i][j][1] : 0.0f;
            qc[i][j][NZ-1] = (!std::isnan(qc[i][j][NZ-2]) && std::isfinite(qc[i][j][NZ-2])) ? qc[i][j][NZ-2] : 0.0f;

            qr[i][j][0] = (!std::isnan(qr[i][j][1]) && std::isfinite(qr[i][j][1])) ? qr[i][j][1] : 0.0f;
            qr[i][j][NZ-1] = (!std::isnan(qr[i][j][NZ-2]) && std::isfinite(qr[i][j][NZ-2])) ? qr[i][j][NZ-2] : 0.0f;

            qi[i][j][0] = (!std::isnan(qi[i][j][1]) && std::isfinite(qi[i][j][1])) ? qi[i][j][1] : 0.0f;
            qi[i][j][NZ-1] = (!std::isnan(qi[i][j][NZ-2]) && std::isfinite(qi[i][j][NZ-2])) ? qi[i][j][NZ-2] : 0.0f;

            qs[i][j][0] = (!std::isnan(qs[i][j][1]) && std::isfinite(qs[i][j][1])) ? qs[i][j][1] : 0.0f;
            qs[i][j][NZ-1] = (!std::isnan(qs[i][j][NZ-2]) && std::isfinite(qs[i][j][NZ-2])) ? qs[i][j][NZ-2] : 0.0f;

            qg[i][j][0] = (!std::isnan(qg[i][j][1]) && std::isfinite(qg[i][j][1])) ? qg[i][j][1] : 0.0f;
            qg[i][j][NZ-1] = (!std::isnan(qg[i][j][NZ-2]) && std::isfinite(qg[i][j][NZ-2])) ? qg[i][j][NZ-2] : 0.0f;

            qh[i][j][0] = (!std::isnan(qh[i][j][1]) && std::isfinite(qh[i][j][1])) ? qh[i][j][1] : 0.0f;
            qh[i][j][NZ-1] = (!std::isnan(qh[i][j][NZ-2]) && std::isfinite(qh[i][j][NZ-2])) ? qh[i][j][NZ-2] : 0.0f;
        }
    }


    for (int j = 0; j < NTH; ++j) 
    {
        for (int k = 0; k < NZ; ++k) 
        {
            theta[0][j][k] = (!std::isnan(theta[1][j][k]) && std::isfinite(theta[1][j][k])) ? theta[1][j][k] : theta0;
            theta[NR-1][j][k] = (!std::isnan(theta[NR-2][j][k]) && std::isfinite(theta[NR-2][j][k])) ? theta[NR-2][j][k] : theta0;

            qv[0][j][k] = (!std::isnan(qv[1][j][k]) && std::isfinite(qv[1][j][k])) ? qv[1][j][k] : 0.0f;
            qv[NR-1][j][k] = (!std::isnan(qv[NR-2][j][k]) && std::isfinite(qv[NR-2][j][k])) ? qv[NR-2][j][k] : 0.0f;

            qc[0][j][k] = (!std::isnan(qc[1][j][k]) && std::isfinite(qc[1][j][k])) ? qc[1][j][k] : 0.0f;
            qc[NR-1][j][k] = (!std::isnan(qc[NR-2][j][k]) && std::isfinite(qc[NR-2][j][k])) ? qc[NR-2][j][k] : 0.0f;

            qr[0][j][k] = (!std::isnan(qr[1][j][k]) && std::isfinite(qr[1][j][k])) ? qr[1][j][k] : 0.0f;
            qr[NR-1][j][k] = (!std::isnan(qr[NR-2][j][k]) && std::isfinite(qr[NR-2][j][k])) ? qr[NR-2][j][k] : 0.0f;

            qi[0][j][k] = (!std::isnan(qi[1][j][k]) && std::isfinite(qi[1][j][k])) ? qi[1][j][k] : 0.0f;
            qi[NR-1][j][k] = (!std::isnan(qi[NR-2][j][k]) && std::isfinite(qi[NR-2][j][k])) ? qi[NR-2][j][k] : 0.0f;

            qs[0][j][k] = (!std::isnan(qs[1][j][k]) && std::isfinite(qs[1][j][k])) ? qs[1][j][k] : 0.0f;
            qs[NR-1][j][k] = (!std::isnan(qs[NR-2][j][k]) && std::isfinite(qs[NR-2][j][k])) ? qs[NR-2][j][k] : 0.0f;

            qg[0][j][k] = (!std::isnan(qg[1][j][k]) && std::isfinite(qg[1][j][k])) ? qg[1][j][k] : 0.0f;
            qg[NR-1][j][k] = (!std::isnan(qg[NR-2][j][k]) && std::isfinite(qg[NR-2][j][k])) ? qg[NR-2][j][k] : 0.0f;

            qh[0][j][k] = (!std::isnan(qh[1][j][k]) && std::isfinite(qh[1][j][k])) ? qh[1][j][k] : 0.0f;
            qh[NR-1][j][k] = (!std::isnan(qh[NR-2][j][k]) && std::isfinite(qh[NR-2][j][k])) ? qh[NR-2][j][k] : 0.0f;
        }
    }

    enforce_primary_state_bounds("old_dynamics_final");
}
