/**
 * @file dynamics.cpp
 * @brief Core runtime implementation for the tornado model.
 *
 * Provides simulation orchestration and subsystem integration
 * for dynamics, numerics, physics, and runtime execution paths.
 * This file belongs to the primary src/core execution layer.
 */

#include "core/simulation.hpp"
#include "boundary_conditions/boundary_conditions.hpp"
#include "core/diffusion_step.hpp"
#include "core/field_sanitization.hpp"
#include "core/runtime_config.hpp"
#include "diagnostics/conservation_budget.hpp"
#include "numerics/diffusion/diffusion_base.hpp"
#include "numerics/time_stepping/time_stepping_base.hpp"
#include "turbulence/turbulence_base.hpp"
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
std::unique_ptr<BoundaryConditionScheme> bc_scheme = nullptr;

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

        // Initialize coordinate-matched boundary condition scheme.
        if (global_coordinate_system == CoordinateSystem::Cartesian)
            bc_scheme = create_cartesian_bc_scheme();
        else
            bc_scheme = create_cylindrical_bc_scheme();
        tmv::log_info("Initialized BC scheme: ", bc_scheme->get_scheme_name());

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

    } 
    catch (const std::runtime_error& e) 
    {
        tmv::log_error("Error initializing dynamics scheme: ", e.what());
        throw;
    }
}

/**
 * @brief Split-explicit dynamics step — constructs callbacks and delegates
 *        to the time stepping scheme in src/numerics/time_stepping/.
 *
 * The callbacks know about the global fields (u, v, w, rho, p) and
 * the dynamics scheme. The time stepping scheme owns the algorithm.
 */
void step_dynamics_split_explicit(
    double dt_dynamics,
    SplitExplicitDynamics& split_scheme,
    Field3D& du_dt, Field3D& dv_dt, Field3D& dw_dt,
    Field3D& drho_dt, Field3D& dp_dt)
{
    SplitExplicitCallbacks callbacks;

    callbacks.compute_slow_tendencies = [&]()
    {
        split_scheme.compute_slow_tendencies(
            u, v, w, rho, p, theta, dt_dynamics,
            du_dt, dv_dt, dw_dt, drho_dt, dp_dt);
    };

    callbacks.apply_slow_tendencies = [&](double dt_large)
    {
        const float dt_f = static_cast<float>(dt_large);
        #pragma omp parallel for collapse(2)
        for (int i = 0; i < NR; ++i)
            for (int j = 0; j < NTH; ++j)
                for (int k = 0; k < NZ; ++k)
                {
                    float du_slow = du_dt[i][j][k] + du_dt_pbl[i][j][k];
                    float dv_slow = dv_dt[i][j][k] + dv_dt_pbl[i][j][k];
                    float dw_slow = dw_dt[i][j][k];
                    float dp_slow = dp_dt[i][j][k];
                    if (!std::isfinite(du_slow)) du_slow = 0.0f;
                    if (!std::isfinite(dv_slow)) dv_slow = 0.0f;
                    if (!std::isfinite(dw_slow)) dw_slow = 0.0f;
                    if (!std::isfinite(dp_slow)) dp_slow = 0.0f;

                    u[i][j][k]       += du_slow * dt_f;
                    v[i][j][k] += dv_slow * dt_f;
                    w[i][j][k]       += dw_slow * dt_f;
                    p[i][j][k]       += dp_slow * dt_f;
                }
    };

    // Fast-tendency buffers: allocated once inside the scheme.
    // We re-use the same drho_dt / dp_dt buffers since slow step is done.
    callbacks.apply_fast_pressure = [&](double dt_small)
    {
        split_scheme.compute_fast_pressure_tendencies(
            u, v, w, rho, p, drho_dt, dp_dt);

        const float dt_s = static_cast<float>(dt_small);
        #pragma omp parallel for collapse(2)
        for (int i = 0; i < NR; ++i)
            for (int j = 0; j < NTH; ++j)
                for (int k = 0; k < NZ; ++k)
                {
                    float drho_f = drho_dt[i][j][k];
                    float dp_f = dp_dt[i][j][k];
                    if (!std::isfinite(drho_f)) drho_f = 0.0f;
                    if (!std::isfinite(dp_f)) dp_f = 0.0f;

                    float rho_new = rho[i][j][k] + drho_f * dt_s;
                    float p_new = p[i][j][k] + dp_f * dt_s;
                    if (!std::isfinite(rho_new) || rho_new <= 0.0f)
                        rho_new = static_cast<float>(std::max(0.1, rho0_base[k]));
                    if (!std::isfinite(p_new) || p_new <= 0.0f)
                        p_new = static_cast<float>(p0);

                    rho[i][j][k] = clamp_density_kgm3(rho_new);
                    p[i][j][k] = clamp_pressure_pa(p_new);
                }
    };

    callbacks.apply_fast_momentum = [&](double dt_small)
    {
        split_scheme.compute_fast_momentum_tendencies(
            u, v, w, rho, p,
            du_dt, dv_dt, dw_dt);

        const float dt_s = static_cast<float>(dt_small);
        #pragma omp parallel for collapse(2)
        for (int i = 0; i < NR; ++i)
            for (int j = 0; j < NTH; ++j)
                for (int k = 0; k < NZ; ++k)
                {
                    float du_f = du_dt[i][j][k];
                    float dv_f = dv_dt[i][j][k];
                    float dw_f = dw_dt[i][j][k];
                    if (!std::isfinite(du_f)) du_f = 0.0f;
                    if (!std::isfinite(dv_f)) dv_f = 0.0f;
                    if (!std::isfinite(dw_f)) dw_f = 0.0f;

                    u[i][j][k]       = clamp_wind_horizontal_ms(u[i][j][k] + du_f * dt_s);
                    v[i][j][k] = clamp_wind_horizontal_ms(v[i][j][k] + dv_f * dt_s);
                    w[i][j][k]       = clamp_wind_vertical_ms(w[i][j][k] + dw_f * dt_s);
                }
    };

    callbacks.acoustic_bcs = [&]() { bc_scheme->apply_acoustic(); };

    // Delegate to the time stepping scheme.
    time_stepping_scheme->step_split_acoustic(
        global_time_stepping_config, dt_dynamics, callbacks);
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
    Field3D& du_dt = du_r_dt_buf;
    Field3D& dv_dt = du_theta_dt_buf;
    Field3D& dw_dt = du_z_dt_buf;
    Field3D& drho_dt = drho_dt_buf;
    Field3D& dp_dt = dp_dt_buf;
    const ConservationBudget budget_start = compute_conservation_budget();

    auto* split_dynamics = dynamic_cast<SplitExplicitDynamics*>(dynamics_scheme.get());
    const bool use_split = global_time_stepping_config.split_acoustic
                           && split_dynamics != nullptr;

    if (!use_split)
    {
        // === UNSPLIT PATH ===
        // Delegate to the time stepping scheme's step_unsplit() method.
        // Default implementation is Forward Euler (compute once, apply once).
        auto compute = [&]() {
            dynamics_scheme->compute_momentum_tendencies(
                u, v, w, rho, p, theta, dt_dynamics,
                du_dt, dv_dt, dw_dt, drho_dt, dp_dt);
        };

        auto apply = [&](double dt_step) {
            const float dt_s = static_cast<float>(dt_step);
            #pragma omp parallel for collapse(2)
            for (int i = 0; i < NR; ++i)
            {
                for (int j = 0; j < NTH; ++j)
                {
                    for (int k = 0; k < NZ; ++k)
                    {
                        float du_f = du_dt[i][j][k] + du_dt_pbl[i][j][k];
                        float dv_f = dv_dt[i][j][k] + dv_dt_pbl[i][j][k];
                        float dw_f = dw_dt[i][j][k];
                        float drho_f = drho_dt[i][j][k];
                        float dp_f = dp_dt[i][j][k];

                        if (!std::isfinite(du_f)) du_f = 0.0f;
                        if (!std::isfinite(dv_f)) dv_f = 0.0f;
                        if (!std::isfinite(dw_f)) dw_f = 0.0f;
                        if (!std::isfinite(drho_f)) drho_f = 0.0f;
                        if (!std::isfinite(dp_f)) dp_f = 0.0f;

                        float u_new = u[i][j][k] + du_f * dt_s;
                        float v_new = v[i][j][k] + dv_f * dt_s;
                        float w_new = w[i][j][k] + dw_f * dt_s;
                        float rho_new = rho[i][j][k] + drho_f * dt_s;
                        float p_new = p[i][j][k] + dp_f * dt_s;

                        if (!std::isfinite(u_new)) u_new = 0.0f;
                        if (!std::isfinite(v_new)) v_new = 0.0f;
                        if (!std::isfinite(w_new)) w_new = 0.0f;
                        if (!std::isfinite(rho_new) || rho_new <= 0.0f)
                            rho_new = static_cast<float>(std::max(0.1, rho0_base[k]));
                        if (!std::isfinite(p_new) || p_new <= 0.0f)
                            p_new = static_cast<float>(p0);

                        u[i][j][k] = clamp_wind_horizontal_ms(u_new);
                        v[i][j][k] = clamp_wind_horizontal_ms(v_new);
                        w[i][j][k] = clamp_wind_vertical_ms(w_new);
                        rho[i][j][k] = clamp_density_kgm3(rho_new);
                        p[i][j][k] = clamp_pressure_pa(p_new);
                    }
                }
            }
        };

        time_stepping_scheme->step_unsplit(dt_dynamics, compute, apply);
    }
    else
    {
        // === SPLIT-EXPLICIT PATH ===
        // Delegate to the time stepping scheme in src/numerics/time_stepping/.
        step_dynamics_split_explicit(dt_dynamics, *split_dynamics, du_dt, dv_dt, dw_dt, drho_dt, dp_dt);
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
                float v_new = v[i][j][k] + dvdt_sgs * dt_dynamics;
                float w_new = w[i][j][k] + dwdt_sgs * dt_dynamics;
                if (!std::isfinite(u_new)) u_new = 0.0f;
                if (!std::isfinite(v_new)) v_new = 0.0f;
                if (!std::isfinite(w_new)) w_new = 0.0f;
                u[i][j][k] = clamp_wind_horizontal_ms(u_new);
                v[i][j][k] = clamp_wind_horizontal_ms(v_new);
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

    dynamics_scheme->compute_vorticity_diagnostics(u, v, w, rho, p, vorticity_r, vorticity_theta, vorticity_z,
        stretching_term, tilting_term, baroclinic_term);

    dynamics_scheme->compute_angular_momentum(u, v, angular_momentum, angular_momentum_tendency);

    dynamics_scheme->compute_pressure_diagnostics(u, v, w, rho, theta,p_prime, dynamic_pressure, buoyancy_pressure);

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
 * @brief Applies boundary conditions to all prognostic state fields.
 *
 * Dispatches by the active coordinate system: the cylindrical body below
 * (axis-reflection at i=0, periodic theta wraparound implicit in the
 * dynamics, vertical rigid lid + surface) is the historical default. The
 * Delegates to the active BoundaryConditionScheme (Phase B.3).
 * (declared in `core/boundary_conditions.hpp`), which uses open lateral
 * BCs on all four x/y faces. See Phase A.3 of the Coordinate Backend Plan
 * (`docs/CoordinateBackend_Plan.md`) and Bug 7 in `docs/Journey.md` for the
 * motivation.
 *
 * `enforce_primary_state_bounds` is called once at the end for *both*
 * branches as a defensive clamp on any field that fell outside its
 * physical range.
 */
void apply_boundary_conditions()
{
    bc_scheme->apply_full();
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

    // DEPRECATED: This legacy path is only reached when no dynamics scheme
    // is initialized. All production configs initialize a scheme via
    // initialize_dynamics(). Scheduled for removal in Phase B.
    static bool warned = false;
    if (!warned)
    {
        tmv::log_warn("[DYNAMICS] Using legacy step_dynamics_old — no dynamics scheme initialized. "
                      "This code path is deprecated and will be removed.");
        warned = true;
    }

    double max_speed = 1e-6;

    #pragma omp parallel for collapse(2) reduction(max:max_speed)
    for (int i = 1; i < NR - 1; ++i) 
    {
        for (int j = 0; j < NTH; ++j) 
        {
            for (int k = 1; k < NZ - 1; ++k) 
            {
                double sp = std::max({std::abs((double)u[i][j][k]), std::abs((double)w[i][j][k]), std::abs((double)v[i][j][k])});
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

                if (std::isnan(v[i][j][k]) || std::isinf(v[i][j][k])) 
                {
                    v[i][j][k] = 0.0f;
                }

                p[i][j][k] = clamp_pressure_pa(p[i][j][k]);
                theta[i][j][k] = clamp_theta_k(theta[i][j][k]);
                u[i][j][k] = clamp_wind_horizontal_ms(u[i][j][k]);
                w[i][j][k] = clamp_wind_vertical_ms(w[i][j][k]);
                v[i][j][k] = clamp_wind_horizontal_ms(v[i][j][k]);
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
