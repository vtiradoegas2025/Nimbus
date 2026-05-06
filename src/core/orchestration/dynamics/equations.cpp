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
#include <sstream>
#include <string>
#include "core/runtime/simulation.hpp"
#include "core/runtime/runtime_config.hpp"
#include "core/orchestration/dynamics/initial_conditions.hpp"
#include "init/hodograph/factory.hpp"
#include "init/hodograph/hodograph_source.hpp"
#include "init/sounding/diagnostics.hpp"
#include "init/sounding/factory.hpp"
#include "init/sounding/sounding_source.hpp"
#include "init/trigger/factory.hpp"
#include "init/trigger/trigger_source.hpp"
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

GridGeometry global_grid_geometry;

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

    global_grid_geometry.initialize(NR, NTH, NZ, dr, dz, dtheta,
                                    global_coordinate_system,
                                    global_stagger_type);
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
std::vector<double> qv0_base;
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

    // Prognostic fields — required for initialize() and the integration loop.
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

    // PBL tendencies, radiation tendency, and radar are allocated by their
    // respective initialize_*() functions before the first timestep.
    // Diagnostic fields are allocated on first use in compute_dynamics_diagnostics().
}

/**
 * @brief Coordinate-aware wind initialization from explicit (u_x, u_y) columns.
 *
 * Used when the active SoundingSource carries its own hodograph (e.g.
 * FileSoundingSource extracting winds from a SHARPY column). Mirrors the
 * coordinate / stagger dispatch that the parametric apply_*_wind_*()
 * helpers use, but reads from a per-level Cartesian column rather than
 * sampling compute_wind_profile.
 */
static void apply_wind_initialization_from_column(
    const std::vector<double>& u_x_column,
    const std::vector<double>& u_y_column)
{
    const auto& geo = global_grid_geometry;

    if (global_coordinate_system == CoordinateSystem::Cartesian)
    {
        #pragma omp parallel for collapse(2)
        for (int i = 0; i < NR; ++i)
        {
            for (int j = 0; j < NTH; ++j)
            {
                for (int k = 0; k < NZ; ++k)
                {
                    const std::size_t kz = static_cast<std::size_t>(k);
                    u[i][j][k] = static_cast<float>(u_x_column[kz]);
                    v[i][j][k] = static_cast<float>(u_y_column[kz]);
                    w[i][j][k] = 0.0f;
                }
            }
        }
        return;
    }

    if (global_stagger_type == StaggerType::CGrid)
    {
        // C-grid: u (radial) lives on r-faces at theta[j]; v (azimuthal)
        // lives on theta-faces at theta_{j+1/2}. Project at the
        // appropriate angle for each face. Same identity-based half-cell
        // trig step as apply_cylindrical_cgrid_wind_initialization().
        const double half_dtheta = 0.5 * dtheta;
        const double cos_half = std::cos(half_dtheta);
        const double sin_half = std::sin(half_dtheta);

        #pragma omp parallel for collapse(2)
        for (int i = 0; i < NR; ++i)
        {
            for (int j = 0; j < NTH; ++j)
            {
                const double cos_th_center = geo.cos_theta[j];
                const double sin_th_center = geo.sin_theta[j];
                const double cos_th_face = cos_th_center * cos_half
                                         - sin_th_center * sin_half;
                const double sin_th_face = sin_th_center * cos_half
                                         + cos_th_center * sin_half;

                for (int k = 0; k < NZ; ++k)
                {
                    const std::size_t kz = static_cast<std::size_t>(k);
                    const double u_x = u_x_column[kz];
                    const double u_y = u_y_column[kz];
                    u[i][j][k] = static_cast<float>( u_x * cos_th_center
                                                    + u_y * sin_th_center);
                    v[i][j][k] = static_cast<float>(-u_x * sin_th_face
                                                    + u_y * cos_th_face);
                    w[i][j][k] = 0.0f;
                }
            }
        }
        return;
    }

    // Cylindrical collocated: both u and v at cell-center theta.
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < NR; ++i)
    {
        for (int j = 0; j < NTH; ++j)
        {
            const double cos_th = geo.cos_theta[j];
            const double sin_th = geo.sin_theta[j];
            for (int k = 0; k < NZ; ++k)
            {
                const std::size_t kz = static_cast<std::size_t>(k);
                const double u_x = u_x_column[kz];
                const double u_y = u_y_column[kz];
                u[i][j][k] = static_cast<float>( u_x * cos_th + u_y * sin_th);
                v[i][j][k] = static_cast<float>(-u_x * sin_th + u_y * cos_th);
                w[i][j][k] = 0.0f;
            }
        }
    }
}

/**
 * @brief Cylindrical wind initialization: project Cartesian hodograph onto (r, θ).
 */
static void apply_cylindrical_wind_initialization()
{
    const auto& geo = global_grid_geometry;

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < NR; ++i)
    {
        for (int j = 0; j < NTH; ++j)
        {
            const double cos_th = geo.cos_theta[j];
            const double sin_th = geo.sin_theta[j];

            for (int k = 0; k < NZ; ++k)
            {
                const double z = geo.z[k];
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
 *
 * External-linkage so the trigger module's WarmBubbleTrigger can call it.
 * Declaration in `include/core/initial_conditions.hpp`.
 */
void apply_cylindrical_bubble_initialization()
{
    const double bubble_center_r = std::max(0.0, global_bubble_center_x_m);
    const double bubble_center_z = std::max(0.0, global_bubble_center_z_m);
    const double bubble_radius   = std::max(100.0, global_bubble_radius_m);
    const double bubble_dtheta   = global_bubble_dtheta_k;

    const auto& geo = global_grid_geometry;

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < NR; ++i)
    {
        for (int j = 0; j < NTH; ++j)
        {
            const double r_dist = geo.r[i];
            for (int k = 0; k < NZ; ++k)
            {
                const double z_dist = geo.z[k];
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
    // ── Build base-state column from configured SoundingSource ──
    //
    // The runtime populates global_sounding_source_config from YAML in
    // load_config(); equations.cpp reads it here and dispatches via the
    // factory. Default is ParametricCAPE so configs with no explicit
    // `environment.sounding.type` get the historical procedural base
    // state. type=file routes to FileSoundingSource (SHARPY/HDF5/NetCDF).
    // The returned column is self-consistent: hydrostatic p, theta from
    // p+T, rho from EOS, qv capped at 95% saturation, and (for sources
    // that carry winds) Cartesian (u, v) per level.
    std::vector<double> z_m(static_cast<std::size_t>(NZ));
    for (int k = 0; k < NZ; ++k)
    {
        z_m[static_cast<std::size_t>(k)] = global_grid_geometry.z[k];
    }

    auto sounding_source = tmv::init::make_sounding_source(global_sounding_source_config);
    const tmv::init::Sounding sounding = sounding_source->build(z_m, dz);
    sounding.verify_self_consistent();

    // Base-state vectors are read elsewhere (perturbation Coriolis, GPU
    // reference-state subtraction, virtual-T buoyancy). Keep the global
    // names and shapes the same so consumers don't move.
    rho0_base.assign(sounding.rho_kgm3.begin(), sounding.rho_kgm3.end());
    p0_base.assign(sounding.p_pa.begin(), sounding.p_pa.end());
    qv0_base.assign(sounding.qv_kgkg.begin(), sounding.qv_kgkg.end());

    tmv::log_info("Base state initialized via ", sounding_source->describe(),
                  ": rho0_base[0]=", rho0_base[0],
                  ", rho0_base[", NZ - 1, "]=", rho0_base[NZ - 1],
                  ", p0_base[0]=", p0_base[0], "Pa",
                  ", p0_base[", NZ - 1, "]=", p0_base[NZ - 1], "Pa");

    for (std::size_t k = 0; k < std::min<std::size_t>(5, sounding.size()); ++k)
    {
        tmv::log_debug("[INIT DEBUG] k=", k,
                       ", z=", sounding.z_m[k], "m: T=", sounding.T_k[k],
                       "K, p=", sounding.p_pa[k], "Pa, theta=", sounding.theta_k[k], "K");
    }

    // ── Shared thermodynamic broadcast ──
    //
    // Fills p, rho, theta, qv, and scalars from the sounding column. Wind
    // is zeroed here and set by the coordinate-specific wind initializer
    // below.
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < NR; ++i)
    {
        for (int j = 0; j < NTH; ++j)
        {
            for (int k = 0; k < NZ; ++k)
            {
                const std::size_t kz = static_cast<std::size_t>(k);
                p[i][j][k] = static_cast<float>(sounding.p_pa[kz]);
                rho[i][j][k] = static_cast<float>(sounding.rho_kgm3[kz]);
                theta[i][j][k] = static_cast<float>(sounding.theta_k[kz]);
                qv[i][j][k] = static_cast<float>(sounding.qv_kgkg[kz]);
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

    // ── Wind initialization via HodographSource ──
    //
    // Precedence:
    //   1. Explicit hodograph type (zero, wk_param) ALWAYS wins, even when
    //      the active SoundingSource supplied winds. This lets a user run
    //      a SHARPY thermo column with `hodograph: { type: zero }` to test
    //      pressure-driven flow on a real-world atmosphere.
    //   2. type=Auto + sounding has winds: use sounding's winds.
    //   3. type=Auto + sounding has no winds: build the WK 3-point default
    //      from environment.hodograph.* anchors (today's behavior).
    //
    // apply_wind_initialization_from_column applies coordinate / stagger
    // aware projection (cartesian: store directly; cylindrical collocated:
    // project at theta[j]; cylindrical c-grid: project at theta_{j+1/2}
    // for v). Same math the legacy apply_*_wind_initialization() helpers
    // use; those helpers are now unreachable from initialize() but kept in
    // place for the dynamics tests that link them directly.
    auto hodograph_source = tmv::init::make_hodograph_source(global_hodograph_source_config);
    tmv::init::WindColumn wind;
    const bool explicit_hodograph =
        (global_hodograph_source_config.type
            != tmv::init::HodographSourceConfig::Type::Auto);
    if (explicit_hodograph)
    {
        wind = hodograph_source->build(z_m);
    }
    else if (sounding.has_winds())
    {
        wind.u_ms = sounding.u_ms;
        wind.v_ms = sounding.v_ms;
    }
    else
    {
        wind = hodograph_source->build(z_m);
    }
    apply_wind_initialization_from_column(wind.u_ms, wind.v_ms);

    // ── Sounding diagnostics ──
    //
    // Surface-parcel lift over the final column produces CAPE, CIN, LCL,
    // LFC, EL, precipitable water, and (with the resolved hodograph) the
    // 0-6 km bulk shear. Logged once at INFO so the user can see what
    // their config actually built before the time loop starts. Marginal-
    // environment warnings fire at well-known thresholds — useful when a
    // file-based or parametric_targets sounding produces less convection
    // than the user expected.
    {
        const auto diags = tmv::init::compute_sounding_diagnostics(sounding, wind);
        const auto fmt_optional = [](double x) -> std::string {
            if (!std::isfinite(x))
            {
                return std::string("(none)");
            }
            std::ostringstream os;
            os.setf(std::ios::fixed);
            os.precision(0);
            os << x << "m";
            return os.str();
        };
        tmv::log_info("Sounding diagnostics: CAPE=", static_cast<int>(diags.cape_jkg),
                      " J/kg, CIN=", static_cast<int>(diags.cin_jkg),
                      " J/kg, LCL=", static_cast<int>(diags.lcl_m), "m",
                      ", LFC=", fmt_optional(diags.lfc_m),
                      ", EL=", fmt_optional(diags.el_m),
                      ", PWAT=", static_cast<int>(diags.pwat_mm), "mm",
                      ", shear_0_6km=",
                      diags.has_kinematic
                          ? std::to_string(static_cast<int>(diags.bulk_shear_0_6km_ms))
                          : std::string("(n/a)"),
                      diags.has_kinematic ? " m/s" : "");

        if (!std::isfinite(diags.lfc_m))
        {
            tmv::log_warn("[SOUNDING] No LFC found: surface parcel never becomes "
                          "positively buoyant. No deep convection will develop.");
        }
        else if (diags.cape_jkg < 500.0)
        {
            tmv::log_warn("[SOUNDING] Marginal CAPE (", static_cast<int>(diags.cape_jkg),
                          " J/kg). Storm development unlikely.");
        }
        if (diags.cin_jkg > 200.0)
        {
            tmv::log_warn("[SOUNDING] Strong cap (CIN=",
                          static_cast<int>(diags.cin_jkg),
                          " J/kg). The trigger may not break through.");
        }
        if (diags.has_kinematic && diags.bulk_shear_0_6km_ms < 15.0)
        {
            tmv::log_warn("[SOUNDING] Weak 0-6 km shear (",
                          static_cast<int>(diags.bulk_shear_0_6km_ms),
                          " m/s). Storm-mode organization unlikely.");
        }
    }

    // ── Trigger via TriggerSource ──
    //
    // Default WarmBubbleTrigger dispatches to the same Cartesian / cylindrical
    // bubble helpers the legacy code called inline; NoOpTrigger is the
    // explicit no-op for hydrostatic / equilibrium tests, replacing the
    // legacy `trigger.bubble.dtheta_k = 0` workaround.
    auto trigger_source = tmv::init::make_trigger_source(global_trigger_source_config);
    trigger_source->apply();
    tmv::log_info("Trigger source: ", trigger_source->describe());

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
    // u0_base / v0_base come from the same resolved hodograph the wind
    // initialization just used, so the base-state perturbation Coriolis
    // sees the same reference flow that's actually in the field.
    for (int k = 0; k < NZ; ++k)
    {
        const std::size_t kz = static_cast<std::size_t>(k);
        const double wind_u = wind.u_ms[kz];
        const double wind_v = wind.v_ms[kz];

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

