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
#include "core/field/field_sanitization.hpp"
#include "core/runtime_config.hpp"
#include "diagnostics/conservation_budget.hpp"
#include "numerics/diffusion/diffusion_base.hpp"
#include "numerics/time_stepping/time_stepping_base.hpp"
#include "turbulence/turbulence_base.hpp"
#include "dynamics/factory.hpp"
#include "compute/compute_kernel_template.hpp"
#include "terrain/terrain_base.hpp"

// Forward-declare EOS update. Full thermodynamics.hpp cannot be included
// alongside simulation.hpp due to `using namespace` constant collisions.
class Field3D;
namespace thermodynamics { void update_density_from_eos(const Field3D& p, const Field3D& theta, Field3D& rho); }

// Forward-declare Rayleigh damping sponge layers (src/core/infra/rayleigh_damping.cpp).
void apply_rayleigh_damping(double dt_damp);
void apply_lateral_damping(double dt_damp);
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

        // Boundary-condition scheme matched to (coordinate, stagger).
        // Dispatch lives in src/boundary_conditions/factory.cpp.
        bc_scheme = create_boundary_condition_scheme(
            global_coordinate_system, global_stagger_type);
        tmv::log_info("Initialized BC scheme: ", bc_scheme->get_scheme_name());

        // Diagnostic fields (vorticity, angular momentum, pressure decomposition)
        // are allocated on first use in compute_dynamics_diagnostics().
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
        #pragma omp parallel for collapse(2)
        for (int i = 0; i < NR; ++i)
            for (int j = 0; j < NTH; ++j)
                for (int k = 0; k < NZ; ++k)
                {
                    double du_slow = static_cast<double>(du_dt[i][j][k]) + static_cast<double>(du_dt_pbl[i][j][k]);
                    double dv_slow = static_cast<double>(dv_dt[i][j][k]) + static_cast<double>(dv_dt_pbl[i][j][k]);
                    double dw_slow = static_cast<double>(dw_dt[i][j][k]);
                    double dp_slow = static_cast<double>(dp_dt[i][j][k]);
                    if (!std::isfinite(du_slow)) du_slow = 0.0;
                    if (!std::isfinite(dv_slow)) dv_slow = 0.0;
                    if (!std::isfinite(dw_slow)) dw_slow = 0.0;
                    if (!std::isfinite(dp_slow)) dp_slow = 0.0;

                    u[i][j][k] = static_cast<float>(static_cast<double>(u[i][j][k]) + du_slow * dt_large);
                    v[i][j][k] = static_cast<float>(static_cast<double>(v[i][j][k]) + dv_slow * dt_large);
                    w[i][j][k] = static_cast<float>(static_cast<double>(w[i][j][k]) + dw_slow * dt_large);
                    p[i][j][k] = static_cast<float>(static_cast<double>(p[i][j][k]) + dp_slow * dt_large);
                }
    };

    // Fast-tendency buffers: allocated once inside the scheme.
    // We re-use the same drho_dt / dp_dt buffers since slow step is done.
    const bool flat_terrain = (global_terrain_config.scheme_id == "none");
    const float dr_f = static_cast<float>(dr);
    const float dtheta_f = static_cast<float>(dtheta);
    const float dz_f = static_cast<float>(dz);

    callbacks.apply_fast_pressure = [&, flat_terrain, dr_f, dtheta_f, dz_f](double dt_small)
    {
        const float dt_s = static_cast<float>(dt_small);

        // GPU path: fused divergence + integration
        if (flat_terrain && dispatch_acoustic_pressure_backend(
                u.data(), v.data(), w.data(),
                rho.data(), p.data(),
                rho.data(), p.data(),
                NR, NTH, NZ, dr_f, dtheta_f, dz_f,
                static_cast<float>(dynamics_constants::gamma), dt_s,
                density_min_kgm3, pressure_min_pa))
        {
            return;
        }

        // CPU fallback
        split_scheme.compute_fast_pressure_tendencies(
            u, v, w, rho, p, drho_dt, dp_dt);

        #pragma omp parallel for collapse(2)
        for (int i = 0; i < NR; ++i)
            for (int j = 0; j < NTH; ++j)
                for (int k = 0; k < NZ; ++k)
                {
                    double drho_d = static_cast<double>(drho_dt[i][j][k]);
                    double dp_d = static_cast<double>(dp_dt[i][j][k]);
                    if (!std::isfinite(drho_d)) drho_d = 0.0;
                    if (!std::isfinite(dp_d)) dp_d = 0.0;

                    double rho_new = static_cast<double>(rho[i][j][k]) + drho_d * dt_small;
                    double p_new = static_cast<double>(p[i][j][k]) + dp_d * dt_small;
                    if (!std::isfinite(rho_new) || rho_new <= 0.0)
                        rho_new = std::max(0.1, rho0_base[k]);
                    if (!std::isfinite(p_new) || p_new <= 0.0)
                        p_new = p0;

                    rho[i][j][k] = clamp_density_kgm3(static_cast<float>(rho_new));
                    p[i][j][k] = clamp_pressure_pa(static_cast<float>(p_new));
                }
    };

    callbacks.apply_fast_momentum = [&, flat_terrain, dr_f, dtheta_f, dz_f](double dt_small)
    {
        const float dt_s = static_cast<float>(dt_small);

        // GPU path: fused pressure gradient + integration
        if (flat_terrain && dispatch_acoustic_momentum_backend(
                rho.data(), p.data(),
                u.data(), v.data(), w.data(),
                u.data(), v.data(), w.data(),
                NR, NTH, NZ, dr_f, dtheta_f, dz_f,
                dt_s, wind_horizontal_abs_max_ms, wind_vertical_abs_max_ms))
        {
            return;
        }

        // CPU fallback
        split_scheme.compute_fast_momentum_tendencies(
            u, v, w, rho, p,
            du_dt, dv_dt, dw_dt);

        #pragma omp parallel for collapse(2)
        for (int i = 0; i < NR; ++i)
            for (int j = 0; j < NTH; ++j)
                for (int k = 0; k < NZ; ++k)
                {
                    double du_d = static_cast<double>(du_dt[i][j][k]);
                    double dv_d = static_cast<double>(dv_dt[i][j][k]);
                    double dw_d = static_cast<double>(dw_dt[i][j][k]);
                    if (!std::isfinite(du_d)) du_d = 0.0;
                    if (!std::isfinite(dv_d)) dv_d = 0.0;
                    if (!std::isfinite(dw_d)) dw_d = 0.0;

                    u[i][j][k] = clamp_wind_horizontal_ms(static_cast<float>(static_cast<double>(u[i][j][k]) + du_d * dt_small));
                    v[i][j][k] = clamp_wind_horizontal_ms(static_cast<float>(static_cast<double>(v[i][j][k]) + dv_d * dt_small));
                    w[i][j][k] = clamp_wind_vertical_ms(static_cast<float>(static_cast<double>(w[i][j][k]) + dw_d * dt_small));
                }
    };

    callbacks.apply_fast_fused = [&, flat_terrain, dr_f, dtheta_f, dz_f](double dt_small) -> bool
    {
        if (!flat_terrain) return false;
        const float dt_s = static_cast<float>(dt_small);
        return dispatch_acoustic_substep_fused_backend(
            u.data(), v.data(), w.data(),
            rho.data(), p.data(),
            NR, NTH, NZ, dr_f, dtheta_f, dz_f,
            static_cast<float>(dynamics_constants::gamma), dt_s,
            density_min_kgm3, pressure_min_pa,
            wind_horizontal_abs_max_ms, wind_vertical_abs_max_ms);
    };

    callbacks.apply_fast_batched = [&, flat_terrain, dr_f, dtheta_f, dz_f](double dt_small, int n_substeps) -> bool
    {
        if (!flat_terrain) return false;
        const float dt_s = static_cast<float>(dt_small);
        return dispatch_acoustic_substeps_batched_backend(
            u.data(), v.data(), w.data(),
            rho.data(), p.data(),
            NR, NTH, NZ, dr_f, dtheta_f, dz_f,
            static_cast<float>(dynamics_constants::gamma), dt_s, n_substeps,
            density_min_kgm3, pressure_min_pa,
            wind_horizontal_abs_max_ms, wind_vertical_abs_max_ms);
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
            #pragma omp parallel for collapse(2)
            for (int i = 0; i < NR; ++i)
            {
                for (int j = 0; j < NTH; ++j)
                {
                    for (int k = 0; k < NZ; ++k)
                    {
                        double du_d = static_cast<double>(du_dt[i][j][k]) + static_cast<double>(du_dt_pbl[i][j][k]);
                        double dv_d = static_cast<double>(dv_dt[i][j][k]) + static_cast<double>(dv_dt_pbl[i][j][k]);
                        double dw_d = static_cast<double>(dw_dt[i][j][k]);
                        double drho_d = static_cast<double>(drho_dt[i][j][k]);
                        double dp_d = static_cast<double>(dp_dt[i][j][k]);

                        if (!std::isfinite(du_d)) du_d = 0.0;
                        if (!std::isfinite(dv_d)) dv_d = 0.0;
                        if (!std::isfinite(dw_d)) dw_d = 0.0;
                        if (!std::isfinite(drho_d)) drho_d = 0.0;
                        if (!std::isfinite(dp_d)) dp_d = 0.0;

                        double u_new = static_cast<double>(u[i][j][k]) + du_d * dt_step;
                        double v_new = static_cast<double>(v[i][j][k]) + dv_d * dt_step;
                        double w_new = static_cast<double>(w[i][j][k]) + dw_d * dt_step;
                        double rho_new = static_cast<double>(rho[i][j][k]) + drho_d * dt_step;
                        double p_new = static_cast<double>(p[i][j][k]) + dp_d * dt_step;

                        if (!std::isfinite(u_new)) u_new = 0.0;
                        if (!std::isfinite(v_new)) v_new = 0.0;
                        if (!std::isfinite(w_new)) w_new = 0.0;
                        if (!std::isfinite(rho_new) || rho_new <= 0.0)
                            rho_new = std::max(0.1, rho0_base[k]);
                        if (!std::isfinite(p_new) || p_new <= 0.0)
                            p_new = p0;

                        u[i][j][k] = clamp_wind_horizontal_ms(static_cast<float>(u_new));
                        v[i][j][k] = clamp_wind_horizontal_ms(static_cast<float>(v_new));
                        w[i][j][k] = clamp_wind_vertical_ms(static_cast<float>(w_new));
                        rho[i][j][k] = clamp_density_kgm3(static_cast<float>(rho_new));
                        p[i][j][k] = clamp_pressure_pa(static_cast<float>(p_new));
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
                double dudt_sgs = static_cast<double>(turb_tend.dudt_sgs[i][j][k]);
                double dvdt_sgs = static_cast<double>(turb_tend.dvdt_sgs[i][j][k]);
                double dwdt_sgs = static_cast<double>(turb_tend.dwdt_sgs[i][j][k]);
                double dthetadt_sgs = static_cast<double>(turb_tend.dthetadt_sgs[i][j][k]);

                if (!std::isfinite(dudt_sgs)) dudt_sgs = 0.0;
                if (!std::isfinite(dvdt_sgs)) dvdt_sgs = 0.0;
                if (!std::isfinite(dwdt_sgs)) dwdt_sgs = 0.0;
                if (!std::isfinite(dthetadt_sgs)) dthetadt_sgs = 0.0;

                double u_new = static_cast<double>(u[i][j][k]) + dudt_sgs * dt_dynamics;
                double v_new = static_cast<double>(v[i][j][k]) + dvdt_sgs * dt_dynamics;
                double w_new = static_cast<double>(w[i][j][k]) + dwdt_sgs * dt_dynamics;
                if (!std::isfinite(u_new)) u_new = 0.0;
                if (!std::isfinite(v_new)) v_new = 0.0;
                if (!std::isfinite(w_new)) w_new = 0.0;
                u[i][j][k] = clamp_wind_horizontal_ms(static_cast<float>(u_new));
                v[i][j][k] = clamp_wind_horizontal_ms(static_cast<float>(v_new));
                w[i][j][k] = clamp_wind_vertical_ms(static_cast<float>(w_new));

                double theta_new = static_cast<double>(theta[i][j][k]) + dthetadt_sgs * dt_dynamics;
                if (!std::isfinite(theta_new))
                {
                    theta_new = theta0;
                }
                theta[i][j][k] = clamp_theta_k(static_cast<float>(theta_new));

                if (!qv.empty())
                {
                    double dqvdt_sgs = static_cast<double>(turb_tend.dqvdt_sgs[i][j][k]);
                    if (!std::isfinite(dqvdt_sgs)) dqvdt_sgs = 0.0;
                    double qv_new = static_cast<double>(qv[i][j][k]) + dqvdt_sgs * dt_dynamics;
                    if (!std::isfinite(qv_new))
                    {
                        qv_new = 0.0;
                    }
                    qv[i][j][k] = clamp_qv_kgkg(static_cast<float>(qv_new));
                }

                if (!tke.empty()) {
                    double dtkedt_sgs = static_cast<double>(turb_tend.dtkedt_sgs[i][j][k]);
                    double dtkedt_pbl = static_cast<double>(dtke_dt_pbl[i][j][k]);
                    if (!std::isfinite(dtkedt_sgs)) dtkedt_sgs = 0.0;
                    if (!std::isfinite(dtkedt_pbl)) dtkedt_pbl = 0.0;
                    double tke_new = static_cast<double>(tke[i][j][k]) + (dtkedt_sgs + dtkedt_pbl) * dt_dynamics;
                    if (!std::isfinite(tke_new)) tke_new = 0.001;
                    tke[i][j][k] = static_cast<float>(std::max(0.001, tke_new));
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

    // EOS density closure: theta was modified by advection, microphysics,
    // radiation, turbulence, and diffusion during this step. Recompute
    // rho from the equation of state so that density-based buoyancy
    // (-g (rho-rho0)/rho) is thermodynamically consistent at the start
    // of the next dynamics step.
    thermodynamics::update_density_from_eos(p, theta, rho);

    // Rayleigh damping sponge layers: relax fields toward the base state
    // near boundaries to absorb outgoing gravity waves and prevent reflection.
    apply_rayleigh_damping(dt_dynamics);     // upper boundary (all grids)
    apply_lateral_damping(dt_dynamics);      // lateral boundaries (Cartesian only)

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

    // Lazy allocation: diagnostic fields are only needed when this function runs.
    if (!field_matches_domain(vorticity_r))
    {
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
    }

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
