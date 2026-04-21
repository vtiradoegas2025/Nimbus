/**
 * @file equations.cpp
 * @brief Core runtime implementation for the tornado model.
 *
 * Provides simulation orchestration and subsystem integration
 * for dynamics, numerics, physics, and runtime execution paths.
 * This file belongs to the primary src/core execution layer.
 */

#include <iostream>
#include <cmath>
#include <exception>
#include <memory>
#include "core/simulation.hpp"
#include "core/runtime_config.hpp"
#include "core/initial_conditions.hpp"
#include "util/simd_utils.hpp"
#include "util/log.hpp"
#include "numerics/advection/advection.hpp"
#include "dynamics/dynamics_base.hpp"
#include "numerics/advection/advection_base.hpp"
#include "numerics/diffusion/diffusion_base.hpp"
#include "numerics/time_stepping/time_stepping_base.hpp"
#include "microphysics/factory.hpp"

// Forward-declare EOS update. Full thermodynamics.hpp cannot be included
// alongside simulation.hpp due to `using namespace` constant collisions.
class Field3D;
namespace thermodynamics { void update_density_from_eos(const Field3D& p, const Field3D& theta, Field3D& rho); }
#include "radar/factory.hpp"
#ifdef _OPENMP
#include <omp.h>
#endif



void compute_wind_profile(const WindProfile& profile, double z, double& u, double& v);


int NR = 200;
int NTH = 128;
int NZ = 150;

double dr = 100.0;
double dz = 100.0;
double dt = 0.1;
double dtheta = 2.0 * 3.14159265358979323846 / NTH;

double simulation_time = 0.0;

/**
 * @brief Updates the grid resolution parameters.
 *
 * Cylindrical: dtheta = 2π / NTH (angular spacing in radians).
 * Cartesian:   dtheta = dr (square horizontal cells). The config parser
 *              may override this afterwards from grid.dy for rectangular cells.
 */
void update_grid_resolution()
{
    if (global_coordinate_system == CoordinateSystem::Cartesian)
    {
        dtheta = dr;
    }
    else
    {
        const double pi = 3.14159265358979323846;
        dtheta = 2.0 * pi / NTH;
    }
}

Field3D rho;
Field3D p;
Field3D u;
Field3D w;
Field3D v;
Field3D tracer;

Field3D theta;
Field3D qv;
Field3D qc;
Field3D qr;
Field3D qi;
Field3D qs;
Field3D qh;
Field3D qg;
Field3D tke;

Field3D radar_reflectivity;

Field3D dtheta_dt_pbl;
Field3D dqv_dt_pbl;
Field3D du_dt_pbl;
Field3D dv_dt_pbl;
Field3D dtke_dt_pbl;

Field3D dtheta_dt_rad;

std::vector<double> rho0_base;
std::vector<double> p0_base;
std::vector<double> u0_base;
std::vector<double> v0_base;
double coriolis_f = 0.0;

std::unique_ptr<MicrophysicsScheme> microphysics_scheme;
std::unique_ptr<RadarSchemeBase> radar_scheme;

NestedGridConfig nested_config;

SimulationConfig global_sim_config;
SchemeRegistry global_scheme_registry;

SchemeRegistry::~SchemeRegistry() = default;
SchemeRegistry::SchemeRegistry(SchemeRegistry&&) noexcept = default;
SchemeRegistry& SchemeRegistry::operator=(SchemeRegistry&&) noexcept = default;
Field3D nest_rho;
Field3D nest_p;
Field3D nest_u;
Field3D nest_w;
Field3D nest_v;
Field3D nest_theta;
Field3D nest_qv;
Field3D nest_qc;
Field3D nest_qr;
Field3D nest_qh;
Field3D nest_qg;
Field3D nest_tracer;

/**
 * @brief Resizes the fields.
 */
void resize_fields() 
{
    update_grid_resolution();

    rho.resize(NR, NTH, NZ, 0.0f);
    p.resize(NR, NTH, NZ, 0.0f);
    u.resize(NR, NTH, NZ, 0.0f);
    w.resize(NR, NTH, NZ, 0.0f);
    v.resize(NR, NTH, NZ, 0.0f);
    tracer.resize(NR, NTH, NZ, 0.0f);
    theta.resize(NR, NTH, NZ, 0.0f);
    qv.resize(NR, NTH, NZ, 0.0f);
    qc.resize(NR, NTH, NZ, 0.0f);
    qr.resize(NR, NTH, NZ, 0.0f);
    qi.resize(NR, NTH, NZ, 0.0f);
    qs.resize(NR, NTH, NZ, 0.0f);
    qh.resize(NR, NTH, NZ, 0.0f);
    qg.resize(NR, NTH, NZ, 0.0f);
    tke.resize(NR, NTH, NZ, 0.0f);
    radar_reflectivity.resize(NR, NTH, NZ, 0.0f);

    dtheta_dt_pbl.resize(NR, NTH, NZ, 0.0f);
    dqv_dt_pbl.resize(NR, NTH, NZ, 0.0f);
    du_dt_pbl.resize(NR, NTH, NZ, 0.0f);
    dv_dt_pbl.resize(NR, NTH, NZ, 0.0f);
    dtke_dt_pbl.resize(NR, NTH, NZ, 0.0f);

    dtheta_dt_rad.resize(NR, NTH, NZ, 0.0f);
}

/**
 * @brief Cylindrical wind initialization: project Cartesian hodograph onto (r, θ).
 */
static void apply_cylindrical_wind_initialization()
{
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < NR; ++i)
    {
        for (int j = 0; j < NTH; ++j)
        {
            const double th = j * dtheta;
            const double cos_th = std::cos(th);
            const double sin_th = std::sin(th);

            for (int k = 0; k < NZ; ++k)
            {
                const double z = k * dz;
                double wind_u_cart, wind_v_cart;
                compute_wind_profile(global_wind_profile, z, wind_u_cart, wind_v_cart);

                u[i][j][k] = static_cast<float>( wind_u_cart * cos_th + wind_v_cart * sin_th);
                v[i][j][k] = static_cast<float>(-wind_u_cart * sin_th + wind_v_cart * cos_th);
            }
        }
    }
}

/**
 * @brief Cylindrical bubble: 2D ring in the (r, z) plane, uniform around all θ.
 */
static void apply_cylindrical_bubble_initialization()
{
    const double bubble_center_r = std::max(0.0, global_bubble_center_x_m);
    const double bubble_center_z = std::max(0.0, global_bubble_center_z_m);
    const double bubble_radius   = std::max(100.0, global_bubble_radius_m);
    const double bubble_dtheta   = global_bubble_dtheta_k;

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < NR; ++i)
    {
        for (int j = 0; j < NTH; ++j)
        {
            const double r_dist = i * dr;
            for (int k = 0; k < NZ; ++k)
            {
                const double z_dist = k * dz;
                const double dist = std::sqrt(
                    (r_dist - bubble_center_r) * (r_dist - bubble_center_r) +
                    (z_dist - bubble_center_z) * (z_dist - bubble_center_z));

                if (dist <= bubble_radius)
                {
                    const double factor = std::exp(
                        -(dist / (bubble_radius / 3.0)) * (dist / (bubble_radius / 3.0)));
                    theta[i][j][k] += static_cast<float>(bubble_dtheta * factor);
                }
            }
        }
    }
}

/**
 * @brief Initializes the base state.
 *
 * The base state is built from a single consistent thermodynamic profile:
 *   1. T_actual(z) — a multi-layer profile (mixed layer + unstable layer +
 *      stable above unstable_top) — is the single source of truth for T.
 *   2. Pressure is integrated upward from p0 using the analytical hydrostatic
 *      solution over each layer with that T:
 *          p[k] = p[k-1] * exp( -g * dz / (R_d * T_avg) )
 *      so that ∂p/∂z = -ρg holds in the discrete sense the dynamics uses.
 *   3. Density follows from the equation of state: ρ = p / (R_d * T).
 *   4. Potential temperature follows from its definition: θ = T * (p₀/p)^κ.
 *
 * This guarantees ρ, p, T, θ are mutually consistent and the base state is
 * in true hydrostatic balance, so the dynamics sees no spurious unbalanced
 * pressure-gradient force on the first step. Previously, p and ρ were
 * derived from the US Standard Atmosphere (T = T_sfc - 6.5 K/km), while θ
 * was computed from T_actual ≠ T_std. The mismatch grew to ~40 K at the
 * model top and produced a ~2 m/s² unbalanced force at z=15 km, which
 * exploded vertical velocity within ~10 timesteps for any config.
 */
void initialize()
{
    const double cape_scaling = global_cape_target / 2500.0;
    const double surface_theta = std::max(250.0, global_sfc_theta_k);
    const double surface_qv = std::max(1.0e-5, global_sfc_qv_kgkg);
    const double tropopause_z = std::max(8000.0, global_tropopause_z_m);
    const double unstable_top_z = std::max(2500.0, std::min(7000.0, 0.5 * tropopause_z));
    const double unstable_lapse_rate = 0.004 + 0.002 * cape_scaling;
    const double kappa = R_d / cp;

    // Multi-layer T_actual(z) profile — the single source of truth for
    // base-state temperature. Used both for the 1D hydrostatic integration
    // below and the 3D field initialization in the nested loop.
    //
    // Layer structure (Weisman & Klemp 1982 style):
    //   1. Mixed layer     (0 to 1 km):        well-mixed, constant T
    //   2. Unstable layer  (1 km to unstable_top): steep lapse rate (CAPE-driven)
    //   3. Upper troposphere (unstable_top to tropopause): standard 6.5 K/km
    //   4. Tropopause      (tropopause to tropo+1 km):   isothermal lid
    //   5. Stratosphere    (above tropo+1 km):            warming +2 K/km
    //
    // The isothermal tropopause provides the "lid" that caps CAPE and
    // prevents unbounded convective growth. The stratospheric warming
    // creates strong static stability that stops any overshooting top.
    // Previous profile used only 3 K/km above the unstable layer with
    // no tropopause, producing theta of 302-315K over 16km (~0.8 K/km).
    // The new profile produces ~302-370K in the troposphere with a sharp
    // jump to ~450K+ in the stratosphere, matching real soundings.
    const double std_lapse_rate = 0.005;   // 5.0 K/km — stable upper troposphere (less than MALR at those temps)
    const double strat_warming  = 0.002;   // +2 K/km — standard stratospheric warming
    const double tropopause_depth = 1000.0; // 1 km isothermal tropopause layer

    auto T_actual_at = [&](double z) -> double
    {
        // Layer 1: Mixed layer — warm, well-mixed boundary layer.
        if (z < 1000.0)
        {
            return surface_theta + 1.0;
        }

        // Layer 2: Conditionally unstable — lapse rate driven by CAPE target.
        // Steeper than the moist adiabat to produce environmental instability.
        if (z < unstable_top_z)
        {
            return surface_theta + 1.0 - unstable_lapse_rate * (z - 1000.0);
        }

        const double T_at_unstable_top =
            surface_theta + 1.0 - unstable_lapse_rate * (unstable_top_z - 1000.0);

        // Layer 3: Upper troposphere — standard environmental lapse rate.
        // Less steep than the unstable layer, reducing CAPE accumulation.
        if (z < tropopause_z)
        {
            return T_at_unstable_top - std_lapse_rate * (z - unstable_top_z);
        }

        const double T_at_tropopause =
            T_at_unstable_top - std_lapse_rate * (tropopause_z - unstable_top_z);

        // Layer 4: Tropopause — isothermal. This is the lid that stops
        // convection. An ascending parcel cools along the moist adiabat
        // while the environment stays constant, so buoyancy goes sharply
        // negative and CAPE terminates.
        const double strat_base_z = tropopause_z + tropopause_depth;
        if (z < strat_base_z)
        {
            return T_at_tropopause;
        }

        // Layer 5: Stratosphere — temperature increases with height.
        // Strong static stability prevents any overshooting updraft from
        // penetrating far above the tropopause.
        return T_at_tropopause + strat_warming * (z - strat_base_z);
    };

    // 1D hydrostatic profile: integrate p upward from p0 using T_actual.
    // The exponential form is exact for an isothermal layer; using T_avg
    // between adjacent levels gives second-order accuracy for arbitrary T(z).
    std::vector<double> p_base(NZ);
    std::vector<double> T_base(NZ);
    rho0_base.resize(NZ);
    p0_base.resize(NZ);

    T_base[0] = T_actual_at(0.0);
    p_base[0] = p0;
    rho0_base[0] = std::max(p_base[0] / (R_d * T_base[0]), 0.1);

    for (int k = 1; k < NZ; ++k)
    {
        const double z_k = k * dz;
        T_base[k] = T_actual_at(z_k);
        const double T_avg = 0.5 * (T_base[k] + T_base[k - 1]);
        p_base[k] = p_base[k - 1] * std::exp(-g * dz / (R_d * T_avg));
        rho0_base[k] = std::max(p_base[k] / (R_d * T_base[k]), 0.1);
    }

    // Store base-state pressure globally for reference-state subtraction in
    // the dynamics schemes. The centered ∂p₀/∂z stencil does not exactly
    // equal -ρ₀g on a collocated grid, so subtracting the discrete reference
    // from the full pressure gradient removes the O(Δz²) hydrostatic
    // imbalance that otherwise seeds spurious vertical velocity.
    for (int k = 0; k < NZ; ++k)
    {
        p0_base[k] = p_base[k];
    }

    tmv::log_info("Base state initialized (hydrostatic): rho0_base[0]=", rho0_base[0],
                  ", rho0_base[", NZ-1, "]=", rho0_base[NZ-1],
                  ", p_base[0]=", p_base[0], "Pa, p_base[", NZ-1, "]=", p_base[NZ-1], "Pa");

    // ── Shared thermodynamic initialization ──
    //
    // Fills p, rho, theta, moisture, and scalars from the 1D hydrostatic
    // profile. Wind is zeroed here and set by the coordinate-specific
    // wind initializer below.

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < NR; ++i)
    {
        for (int j = 0; j < NTH; ++j)
        {
            for (int k = 0; k < NZ; ++k)
            {
                const double z = k * dz;
                const double T_actual = T_base[k];
                const double p_local = p_base[k];
                const double rho_local = p_local / (R_d * T_actual);
                const double theta_potential = T_actual * std::pow(p0 / p_local, kappa);

                p[i][j][k] = static_cast<float>(p_local);
                rho[i][j][k] = static_cast<float>(std::max(rho_local, 0.1));
                theta[i][j][k] = static_cast<float>(theta_potential);

                if (i == 0 && j == 0 && k < 5) {
                    tmv::log_debug("[INIT DEBUG] i=", i, ", j=", j, ", k=", k,
                                   ", z=", z, "m: T_actual=", T_actual,
                                   "K, p_local=", p_local, "Pa, theta=", theta_potential, "K");
                }

                double base_moisture = std::clamp(surface_qv * (0.85 + 0.15 * cape_scaling), 0.004, 0.024);
                const double moisture_scale_height = std::max(1500.0, 0.30 * tropopause_z);
                double qv_base;

                if (z < 2000.0)
                    qv_base = base_moisture;
                else
                    qv_base = base_moisture * exp(-(z - 2000.0) / moisture_scale_height);

                qv[i][j][k] = static_cast<float>(qv_base);
                qc[i][j][k] = 0.0f;
                qr[i][j][k] = 0.0f;
                qi[i][j][k] = 0.0f;
                qs[i][j][k] = 0.0f;
                qh[i][j][k] = 0.0f;
                qg[i][j][k] = 0.0f;
                tke[i][j][k] = 0.1f;

                u[i][j][k] = 0.0f;
                v[i][j][k] = 0.0f;
                w[i][j][k] = 0.0f;
                tracer[i][j][k] = 0.0f;
            }
        }
    }

    // ── Coordinate-specific wind initialization ──
    //
    // Cylindrical: project the Cartesian (u_x, u_y) hodograph onto the
    // local (r, θ) basis with cos θ / sin θ rotation.
    // Cartesian: store (u_x, u_y) directly — no rotation. This is the
    // entire reason the Cartesian backend exists (Bug 7).
    if (global_coordinate_system == CoordinateSystem::Cartesian)
    {
        apply_cartesian_wind_initialization();
    }
    else
    {
        apply_cylindrical_wind_initialization();
    }

    // ── Coordinate-specific trigger bubble ──
    //
    // Cylindrical: 2D ring in the (r, z) plane, uniform around all θ.
    // Cartesian: 3D sphere at literal (x, y, z) center.
    if (global_coordinate_system == CoordinateSystem::Cartesian)
    {
        apply_cartesian_bubble_initialization();
    }
    else
    {
        apply_cylindrical_bubble_initialization();
    }

    // ── Base-state wind profiles for perturbation Coriolis ──
    //
    // Following Rotunno & Klemp (1982), Coriolis is applied to wind
    // perturbations (u - u0, v - v0) rather than the full wind. This
    // avoids the "invented forces" problem documented by Davies-Jones
    // (2021): applying Coriolis to the full wind on an f-plane without
    // a matching body force creates a spurious ageostrophic acceleration
    // that contaminates the solution.
    //
    // The base-state wind is the horizontally homogeneous environmental
    // hodograph used to initialize the domain, stored as 1D profiles.
    u0_base.resize(NZ);
    v0_base.resize(NZ);
    for (int k = 0; k < NZ; ++k)
    {
        const double z = k * dz;
        double wind_u, wind_v;
        compute_wind_profile(global_wind_profile, z, wind_u, wind_v);

        if (global_coordinate_system == CoordinateSystem::Cartesian)
        {
            // Cartesian: base-state winds are the literal (u_x, u_y) profile.
            u0_base[k] = wind_u;
            v0_base[k] = wind_v;
        }
        else
        {
            // Cylindrical: base-state is ambiguous because the projection
            // varies with theta. Store zero -- Coriolis on cylindrical grids
            // uses the geometric -ur*uth/r term instead of planetary f.
            u0_base[k] = 0.0;
            v0_base[k] = 0.0;
        }
    }

    // Coriolis parameter: f = 2 * Omega * sin(latitude).
    // Default latitude 35N matches the radiation module and is standard
    // for mid-latitude supercell simulations (Weisman & Klemp 1982).
    constexpr double omega_earth = 7.292e-5;  // Earth's angular velocity (rad/s)
    constexpr double default_latitude_deg = 35.0;
    constexpr double pi = 3.14159265358979323846;
    const double latitude_rad = default_latitude_deg * pi / 180.0;
    coriolis_f = 2.0 * omega_earth * std::sin(latitude_rad);
    tmv::log_info("Coriolis f = ", coriolis_f, " /s (latitude = ", default_latitude_deg, " deg N)");

    // EOS density update: the bubble modified theta while rho stayed at
    // rho0. Without this, density-based buoyancy (-g (rho-rho0)/rho) is
    // zero despite the theta perturbation. The EOS closure ensures rho
    // reflects the new theta so the first dynamics step sees real buoyancy.
    thermodynamics::update_density_from_eos(p, theta, rho);

    int ic = NR / 4;
    int kc = NZ / 4;
    for (int j = 0; j < NTH; ++j)
    {
        tracer[ic][j][kc] = 1.0f;
    }

    initialize_nested_grid();
    
    float theta_min = 1e10, theta_max = -1e10;
    float p_min = 1e10, p_max = -1e10;
    float rho_min = 1e10, rho_max = -1e10;
    int nan_count = 0, inf_count = 0;
    
    for (int i = 0; i < NR; ++i) {
        for (int j = 0; j < NTH; ++j) {
            for (int k = 0; k < NZ; ++k) {
                float theta_val = theta[i][j][k];
                float p_val = p[i][j][k];
                float rho_val = rho[i][j][k];
                
                if (std::isnan(theta_val) || std::isnan(p_val) || std::isnan(rho_val)) nan_count++;
                if (std::isinf(theta_val) || std::isinf(p_val) || std::isinf(rho_val)) inf_count++;
                
                if (theta_val < theta_min) theta_min = theta_val;
                if (theta_val > theta_max) theta_max = theta_val;
                if (p_val < p_min) p_min = p_val;
                if (p_val > p_max) p_max = p_val;
                if (rho_val < rho_min) rho_min = rho_val;
                if (rho_val > rho_max) rho_max = rho_val;
            }
        }
    }
    
    tmv::log_info("\n[INIT SUMMARY] After initialization:");
    tmv::log_info("  Theta: min=", theta_min, "K, max=", theta_max, "K, expected ~250-350K");
    tmv::log_info("  Pressure: min=", p_min, "Pa, max=", p_max, "Pa, expected ~1000-110000Pa");
    tmv::log_info("  Density: min=", rho_min, "kg/m³, max=", rho_max, "kg/m³, expected ~0.5-1.5kg/m³");
    tmv::log_info("  NaN count: ", nan_count, ", Inf count: ", inf_count);
    
    if (theta_min < 0 || theta_max > 500) 
    {
        tmv::log_warn("  WARNING: Theta values are outside expected range!");
    }
    if (p_min < 500 || p_max > 120000) 
    {
        tmv::log_warn("  WARNING: Pressure values are outside expected range!");
    }
}

/**
 * @brief Initializes the microphysics scheme.
 */
void initialize_microphysics(const std::string& scheme_name) 
{
    try 
    {
        microphysics_scheme = create_microphysics_scheme(scheme_name);
        tmv::log_info("Initialized microphysics scheme: ", scheme_name);
    } 
    catch (const std::runtime_error& e)
    {
        tmv::log_error("Error initializing microphysics: ", e.what());
        microphysics_scheme = create_microphysics_scheme("kessler");
        tmv::log_info("Falling back to Kessler microphysics scheme");
    }
}


/**
 * @brief Backward-compatible scalar advection wrapper.
 *
 * Redirects legacy calls to the newer advection component.
 */
void advect_scalar(Field3D& scalar, double dt_advect, double kappa)
{
    advect_scalar_3d(scalar, dt_advect, kappa);
}

/**
 * @brief Advects the tracer field.
 */
void advect_tracer(double dt_advect) {advect_tracer_3d(dt_advect, 0.01);}

/**
 * @brief Advects the thermodynamics.
 */
void advect_thermodynamics(double dt_advect) {advect_thermodynamics_3d(dt_advect, 0.01, 0.01);}

