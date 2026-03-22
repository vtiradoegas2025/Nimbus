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
#include "util/simd_utils.hpp"
#include "util/log.hpp"
#include "numerics/advection.hpp"
#include "physics/dynamics_base.hpp"
#include "numerics/advection_base.hpp"
#include "numerics/diffusion_base.hpp"
#include "numerics/time_stepping_base.hpp"
#include "microphysics/factory.hpp"
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
 */
void update_grid_resolution() 
{
    const double pi = 3.14159265358979323846;
    dtheta = 2.0 * pi / NTH;
}

Field3D rho;
Field3D p;
Field3D u;
Field3D w;
Field3D v_theta;
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
Field3D nest_v_theta;
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
    v_theta.resize(NR, NTH, NZ, 0.0f);
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
 * @brief Initializes the base state.
 */
void initialize()
{
    double cape_scaling = global_cape_target / 2500.0;
    const double surface_theta = std::max(250.0, global_sfc_theta_k);
    const double surface_qv = std::max(1.0e-5, global_sfc_qv_kgkg);
    const double tropopause_z = std::max(8000.0, global_tropopause_z_m);
    const double unstable_top_z = std::max(2500.0, std::min(7000.0, 0.5 * tropopause_z));
    const double unstable_lapse_rate = 0.004 + 0.002 * cape_scaling;

    rho0_base.resize(NZ);

    for (int k = 0; k < NZ; ++k)
    {
        double z = k * dz;
        double T = surface_theta - 0.0065 * z;
        double p_local = p0 * pow(1 - (0.0065 * z / surface_theta), 5.255);
        rho0_base[k] = std::max(p_local / (R_d * T), 0.1);
    }
    tmv::log_info("Base state density initialized: rho0_base[0]=", rho0_base[0],
                  ", rho0_base[", NZ-1, "]=", rho0_base[NZ-1]);

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < NR; ++i)
    {

        for (int j = 0; j < NTH; ++j)
        {

            for (int k = 0; k < NZ; ++k)
            {
                double z = k * dz;
                double T = surface_theta - 0.0065 * z;
                double p_local = p0 * pow(1 - (0.0065 * z / surface_theta), 5.255);
                double rho_local = p_local / (R_d * T);

                p[i][j][k] = static_cast<float>(p_local);
                rho[i][j][k] = static_cast<float>(std::max(rho_local, 0.1));

                double T_actual;
                
                if (z < 1000.0) 
                {
                    T_actual = surface_theta + 1.0;
                } 

                else if (z < unstable_top_z) 
                {
                    T_actual = surface_theta + 1.0 - unstable_lapse_rate * (z - 1000.0);
                } 
                else 
                {
                    const double T_at_unstable_top = surface_theta + 1.0 - unstable_lapse_rate * (unstable_top_z - 1000.0);
                    T_actual = T_at_unstable_top - 0.003 * (z - unstable_top_z);
                }

                double kappa = R_d / cp;
                double theta_potential = T_actual * pow(p0 / p_local, kappa);
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
                {
                    qv_base = base_moisture;
                } 
                else 
                {
                    qv_base = base_moisture * exp(-(z - 2000.0) / moisture_scale_height);
                }
                qv[i][j][k] = static_cast<float>(qv_base);

                qc[i][j][k] = 0.0f;
                qr[i][j][k] = 0.0f;
                qi[i][j][k] = 0.0f;
                qs[i][j][k] = 0.0f;
                qh[i][j][k] = 0.0f;
                qg[i][j][k] = 0.0f;
                tke[i][j][k] = 0.1f;

                double wind_u_cart, wind_v_cart;
                compute_wind_profile(global_wind_profile, z, wind_u_cart, wind_v_cart);

                double th = j * dtheta;
                double u_r = wind_u_cart * cos(th) + wind_v_cart * sin(th);
                double v_th = -wind_u_cart * sin(th) + wind_v_cart * cos(th);

                u[i][j][k] = static_cast<float>(u_r);
                v_theta[i][j][k] = static_cast<float>(v_th);
                w[i][j][k] = 0.0f;
                tracer[i][j][k] = 0.0f;
            }
        }
    }

    const double bubble_center_r = std::max(0.0, global_bubble_center_x_m);
    const double bubble_center_z = std::max(0.0, global_bubble_center_z_m);
    const double bubble_radius = std::max(100.0, global_bubble_radius_m);
    const double bubble_dtheta = global_bubble_dtheta_k;

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < NR; ++i)
    {
        double r_dist = i * dr;

        for (int j = 0; j < NTH; ++j)
        {
            for (int k = 0; k < NZ; ++k)
            {
                double z_dist = k * dz;
                double dist_from_center = sqrt(pow(r_dist - bubble_center_r, 2) + pow(z_dist - bubble_center_z, 2));

                if (dist_from_center <= bubble_radius)
                {
                    double bubble_factor = exp(-pow(dist_from_center / (bubble_radius / 3.0), 2));
                    theta[i][j][k] += bubble_dtheta * bubble_factor;
                }
            }
        }
    }

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
 * @brief Initializes the radar scheme.
 */    
void initialize_radar(const std::string& scheme_name) 
{
    try 
    {
        radar_scheme = create_radar_scheme(scheme_name);

        RadarConfig config;
        config.scheme_id = scheme_name;
        config.operator_tier = "fast_da";
        if (scheme_name == "zdr")
        {
            config.operator_tier = "polarimetric_fo";
        }

        config.has_qr = true;
        config.has_qs = true;
        config.has_qg = true;
        config.has_qh = true;
        config.has_qi = true;

        radar_scheme->initialize(config, NR, NTH, NZ);
        tmv::log_info("Initialized radar scheme: ", scheme_name);
    } 
    catch (const std::runtime_error& e) 
    {
        tmv::log_error("Error initializing radar: ", e.what());
        tmv::log_info("Radar scheme initialization failed, radar calculations disabled");
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

/**
 * @brief Steps the microphysics forward in time.
 */
void step_microphysics(double dt_micro)
{
    if (!std::isfinite(dt_micro) || dt_micro <= 0.0)
    {
        tmv::log_warn("[MICROPHYSICS GUARD] invalid microphysics timestep: ", dt_micro);
        return;
    }

    if (!microphysics_scheme) 
    {
        tmv::log_warn("Microphysics scheme not initialized, using default Kessler");
        initialize_microphysics("kessler");
    }

    static Field3D dtheta_dt;
    static Field3D dqv_dt;
    static Field3D dqc_dt;
    static Field3D dqr_dt;
    static Field3D dqi_dt;
    static Field3D dqs_dt;
    static Field3D dqg_dt;
    static Field3D dqh_dt;

    auto ensure_shape = [](Field3D& f) 
    {
        if (f.size_r() != NR || f.size_th() != NTH || f.size_z() != NZ)
        {
            f.resize(NR, NTH, NZ, 0.0f);
        }
    };
    ensure_shape(dtheta_dt);
    ensure_shape(dqv_dt);
    ensure_shape(dqc_dt);
    ensure_shape(dqr_dt);
    ensure_shape(dqi_dt);
    ensure_shape(dqs_dt);
    ensure_shape(dqg_dt);
    ensure_shape(dqh_dt);

    try
    {
        microphysics_scheme->compute_tendencies(p, theta, qv, qc, qr, qi, qs, qg, qh,
            dt_micro, dtheta_dt, dqv_dt, dqc_dt, dqr_dt, dqi_dt, dqs_dt, dqg_dt, dqh_dt);
    }
    catch (const std::exception& e)
    {
        tmv::log_warn("[MICROPHYSICS GUARD] tendency computation failed: ", e.what(),
                      ". Continuing with zero microphysics tendencies for this step.");
        dtheta_dt.fill(0.0f);
        dqv_dt.fill(0.0f);
        dqc_dt.fill(0.0f);
        dqr_dt.fill(0.0f);
        dqi_dt.fill(0.0f);
        dqs_dt.fill(0.0f);
        dqg_dt.fill(0.0f);
        dqh_dt.fill(0.0f);
    }

    apply_chaos_to_microphysics_tendencies(dtheta_dt, dqv_dt, dqc_dt, dqr_dt, dqi_dt, dqs_dt, dqg_dt, dqh_dt);

    auto sanitize_nonfinite_tendency = [](Field3D& field) -> int
    {
        if (field.empty()) { return 0; }
        float* const data = field.data();
        const int count = static_cast<int>(field.size());
        return simd_utils::sanitize_nonfinite(data, 0.0f, data, count);
    };

    const int non_finite_tendency_sanitized =
        sanitize_nonfinite_tendency(dtheta_dt) +
        sanitize_nonfinite_tendency(dqv_dt) +
        sanitize_nonfinite_tendency(dqc_dt) +
        sanitize_nonfinite_tendency(dqr_dt) +
        sanitize_nonfinite_tendency(dqi_dt) +
        sanitize_nonfinite_tendency(dqs_dt) +
        sanitize_nonfinite_tendency(dqg_dt) +
        sanitize_nonfinite_tendency(dqh_dt);

    constexpr float max_theta_step_change_k = 50.0f;
    int non_finite_theta_tendency_count = 0;
    int theta_tendency_limited_count = 0;
    int theta_bounds_clamp_count = 0;

    #pragma omp parallel for collapse(2) reduction(+:non_finite_theta_tendency_count, theta_tendency_limited_count, theta_bounds_clamp_count)
    for (int i = 0; i < NR; ++i)
    {
        for (int j = 0; j < NTH; ++j)
        {
    
            for (int k = 0; k < NZ; ++k)
            {
                float theta_old = theta[i][j][k];
                float dtheta_total = (dtheta_dt[i][j][k] + dtheta_dt_rad[i][j][k] + dtheta_dt_pbl[i][j][k]) * dt_micro;
                if (!std::isfinite(dtheta_total))
                {
                    dtheta_total = 0.0f;
                    ++non_finite_theta_tendency_count;
                }

                float dtheta_limited = std::clamp(dtheta_total, -max_theta_step_change_k, max_theta_step_change_k);
                if (dtheta_limited != dtheta_total)
                {
                    ++theta_tendency_limited_count;
                }

                float theta_new = theta_old + dtheta_limited;
                float theta_bounded = clamp_theta_k(theta_new);
                if (theta_bounded != theta_new)
                {
                    ++theta_bounds_clamp_count;
                }
                theta[i][j][k] = theta_bounded;
                
                if (std::abs(dtheta_total) > 100.0f && i == 0 && j == 0 && k < 5) 
                {
                    tmv::log_debug("[MICRO DEBUG] Large theta change at i=", i, ",j=", j, ",k=", k,
                                   ": ", theta_old, " -> ", theta[i][j][k],
                                   " (raw_delta=", dtheta_total,
                                   ", applied_delta=", dtheta_limited, ")");
                    tmv::log_debug("  dtheta_dt=", dtheta_dt[i][j][k],
                                   ", dtheta_dt_rad=", dtheta_dt_rad[i][j][k],
                                   ", dtheta_dt_pbl=", dtheta_dt_pbl[i][j][k]);
                }
                qv[i][j][k] += (dqv_dt[i][j][k] + dqv_dt_pbl[i][j][k]) * dt_micro;
                qc[i][j][k] += dqc_dt[i][j][k] * dt_micro;
                qr[i][j][k] += dqr_dt[i][j][k] * dt_micro;
                qi[i][j][k] += dqi_dt[i][j][k] * dt_micro;
                qs[i][j][k] += dqs_dt[i][j][k] * dt_micro;
                qg[i][j][k] += dqg_dt[i][j][k] * dt_micro;
                qh[i][j][k] += dqh_dt[i][j][k] * dt_micro;

                qv[i][j][k] = clamp_qv_kgkg(qv[i][j][k]);
                qc[i][j][k] = clamp_hydrometeor_kgkg(qc[i][j][k]);
                qr[i][j][k] = clamp_hydrometeor_kgkg(qr[i][j][k]);
                qi[i][j][k] = clamp_hydrometeor_kgkg(qi[i][j][k]);
                qs[i][j][k] = clamp_hydrometeor_kgkg(qs[i][j][k]);
                qg[i][j][k] = clamp_hydrometeor_kgkg(qg[i][j][k]);
                qh[i][j][k] = clamp_hydrometeor_kgkg(qh[i][j][k]);
            }
        }
    }

    if (non_finite_theta_tendency_count > 0 ||
        non_finite_tendency_sanitized > 0 ||
        theta_tendency_limited_count > 0 ||
        theta_bounds_clamp_count > 0)
    {
        tmv::log_warn("[MICROPHYSICS GUARD] non_finite_dtheta=",
                      non_finite_theta_tendency_count,
                      ", non_finite_tendency_sanitized=", non_finite_tendency_sanitized,
                      ", limited_dtheta=", theta_tendency_limited_count,
                      ", theta_bounds_clamped=", theta_bounds_clamp_count);
    }
}


/**
 * @brief Initializes the nested grid.
 */
void initialize_nested_grid()
{
    if (!nested_config.enabled) return;

    int nr_nest = nested_config.nest_size_r;
    int nth_nest = nested_config.nest_size_th;
    int nz_nest = nested_config.nest_size_z;

    nest_rho.resize(nr_nest, nth_nest, nz_nest, 0.0f);
    nest_p.resize(nr_nest, nth_nest, nz_nest, 0.0f);
    nest_u.resize(nr_nest, nth_nest, nz_nest, 0.0f);
    nest_w.resize(nr_nest, nth_nest, nz_nest, 0.0f);
    nest_v_theta.resize(nr_nest, nth_nest, nz_nest, 0.0f);
    nest_theta.resize(nr_nest, nth_nest, nz_nest, 0.0f);
    nest_qv.resize(nr_nest, nth_nest, nz_nest, 0.0f);
    nest_qc.resize(nr_nest, nth_nest, nz_nest, 0.0f);
    nest_qr.resize(nr_nest, nth_nest, nz_nest, 0.0f);
    nest_qh.resize(nr_nest, nth_nest, nz_nest, 0.0f);
    nest_qg.resize(nr_nest, nth_nest, nz_nest, 0.0f);
    nest_tracer.resize(nr_nest, nth_nest, nz_nest, 0.0f);

    double ref = nested_config.refinement;
    int ci = nested_config.center_i;
    int cj = nested_config.center_j;
    int ck = nested_config.center_k;

    #pragma omp parallel for collapse(2)
    for (int i_nest = 0; i_nest < nr_nest; ++i_nest)
    {
        for (int j_nest = 0; j_nest < nth_nest; ++j_nest)
        {
            for (int k_nest = 0; k_nest < nz_nest; ++k_nest)
            {
                double i_parent = ci + (i_nest - nr_nest/2.0) / ref;
                double j_parent = cj + (j_nest - nth_nest/2.0) / ref;
                double k_parent = ck + (k_nest - nz_nest/2.0) / ref;

                int i0 = std::max(0, std::min(NR-2, (int)std::floor(i_parent)));
                int j0 = std::max(0, std::min(NTH-2, (int)std::floor(j_parent)));
                int k0 = std::max(0, std::min(NZ-2, (int)std::floor(k_parent)));

                double fi = i_parent - i0;
                double fj = j_parent - j0;
                double fk = k_parent - k0;

                auto interpolate = [&](const Field3D& field, int i0, int j0, int k0, double fi, double fj, double fk) {
                    double v000 = static_cast<float>(field[i0][j0][k0]);
                    double v001 = static_cast<float>(field[i0][j0][k0+1]);
                    double v010 = static_cast<float>(field[i0][j0+1][k0]);
                    double v011 = static_cast<float>(field[i0][j0+1][k0+1]);
                    double v100 = static_cast<float>(field[i0+1][j0][k0]);
                    double v101 = static_cast<float>(field[i0+1][j0][k0+1]);
                    double v110 = static_cast<float>(field[i0+1][j0+1][k0]);
                    double v111 = static_cast<float>(field[i0+1][j0+1][k0+1]);

                    return v000 * (1-fi)*(1-fj)*(1-fk) +
                           v001 * (1-fi)*(1-fj)*fk +
                           v010 * (1-fi)*fj*(1-fk) +
                           v011 * (1-fi)*fj*fk +
                           v100 * fi*(1-fj)*(1-fk) +
                           v101 * fi*(1-fj)*fk +
                           v110 * fi*fj*(1-fk) +
                           v111 * fi*fj*fk;
                };

                nest_rho[i_nest][j_nest][k_nest] = interpolate(rho, i0, j0, k0, fi, fj, fk);
                nest_p[i_nest][j_nest][k_nest] = interpolate(p, i0, j0, k0, fi, fj, fk);
                nest_u[i_nest][j_nest][k_nest] = interpolate(u, i0, j0, k0, fi, fj, fk);
                nest_w[i_nest][j_nest][k_nest] = interpolate(w, i0, j0, k0, fi, fj, fk);
                nest_v_theta[i_nest][j_nest][k_nest] = interpolate(v_theta, i0, j0, k0, fi, fj, fk);
                nest_theta[i_nest][j_nest][k_nest] = interpolate(theta, i0, j0, k0, fi, fj, fk);
                nest_qv[i_nest][j_nest][k_nest] = interpolate(qv, i0, j0, k0, fi, fj, fk);
                nest_qc[i_nest][j_nest][k_nest] = interpolate(qc, i0, j0, k0, fi, fj, fk);
                nest_qr[i_nest][j_nest][k_nest] = interpolate(qr, i0, j0, k0, fi, fj, fk);
                nest_qh[i_nest][j_nest][k_nest] = interpolate(qh, i0, j0, k0, fi, fj, fk);
                nest_qg[i_nest][j_nest][k_nest] = interpolate(qg, i0, j0, k0, fi, fj, fk);
                nest_tracer[i_nest][j_nest][k_nest] = interpolate(tracer, i0, j0, k0, fi, fj, fk);
            }
        }
    }
}

/**
 * @brief Feedbacks the nested grid to the parent grid.
 */
void feedback_to_parent()
{
    if (!nested_config.enabled) return;

    double ref = nested_config.refinement;
    int ci = nested_config.center_i;
    int cj = nested_config.center_j;
    int ck = nested_config.center_k;
    int nr_nest = nested_config.nest_size_r;
    int nth_nest = nested_config.nest_size_th;
    int nz_nest = nested_config.nest_size_z;


    #pragma omp parallel for collapse(2)
    for (int i_nest = 0; i_nest < nr_nest; ++i_nest)
    {
        for (int j_nest = 0; j_nest < nth_nest; ++j_nest)   
        {
            for (int k_nest = 0; k_nest < nz_nest; ++k_nest)
            {
                int i_parent = ci + (int)std::round((i_nest - nr_nest/2.0) / ref);
                int j_parent = cj + (int)std::round((j_nest - nth_nest/2.0) / ref);
                int k_parent = ck + (int)std::round((k_nest - nz_nest/2.0) / ref);

                if (i_parent >= 0 && i_parent < NR &&
                    j_parent >= 0 && j_parent < NTH &&
                    k_parent >= 0 && k_parent < NZ)
                {
                    rho[i_parent][j_parent][k_parent] = nest_rho[i_nest][j_nest][k_nest];
                    p[i_parent][j_parent][k_parent] = nest_p[i_nest][j_nest][k_nest];
                    u[i_parent][j_parent][k_parent] = nest_u[i_nest][j_nest][k_nest];
                    w[i_parent][j_parent][k_parent] = nest_w[i_nest][j_nest][k_nest];
                    v_theta[i_parent][j_parent][k_parent] = nest_v_theta[i_nest][j_nest][k_nest];
                    theta[i_parent][j_parent][k_parent] = nest_theta[i_nest][j_nest][k_nest];
                    qv[i_parent][j_parent][k_parent] = nest_qv[i_nest][j_nest][k_nest];
                    qc[i_parent][j_parent][k_parent] = nest_qc[i_nest][j_nest][k_nest];
                    qr[i_parent][j_parent][k_parent] = nest_qr[i_nest][j_nest][k_nest];
                    qh[i_parent][j_parent][k_parent] = nest_qh[i_nest][j_nest][k_nest];
                    qg[i_parent][j_parent][k_parent] = nest_qg[i_nest][j_nest][k_nest];
                    tracer[i_parent][j_parent][k_parent] = nest_tracer[i_nest][j_nest][k_nest];
                }
            }
        }
    }
}


/**
 * @brief Calculates the radar reflectivity.
 */
void calculate_radar_reflectivity()
{
    constexpr float radar_linear_min = 0.0f;
    constexpr float radar_linear_max = 1.0e12f;

    auto sanitize_linear_reflectivity_field = [&](Field3D& field, const char* source_tag) {
        if (field.empty())
        {
            return;
        }
        int sanitized = 0;
        float* const data = field.data();
        const std::size_t count = field.size();
        #pragma omp parallel for reduction(+:sanitized)
        for (long long idx = 0; idx < static_cast<long long>(count); ++idx)
        {
            const float old_value = data[idx];
            float new_value = old_value;
            if (!std::isfinite(static_cast<double>(new_value)) || new_value < radar_linear_min)
            {
                new_value = radar_linear_min;
            }
            else if (new_value > radar_linear_max)
            {
                new_value = radar_linear_max;
            }
            if (new_value != old_value)
            {
                ++sanitized;
            }
            data[idx] = new_value;
        }
        if (sanitized > 0)
        {
            tmv::log_warn("[RADAR GUARD] sanitized ", sanitized, " reflectivity samples from ", source_tag);
        }
    };

    auto apply_microphysics_fallback = [&]() -> bool
    {
        if (!microphysics_scheme)
        {
            tmv::log_warn("[RADAR GUARD] no microphysics scheme available for radar fallback.");
            return false;
        }

        Field3D radar_dbz;
        try
        {
            microphysics_scheme->compute_radar_reflectivity(
                qc, qr, qi, qs, qg, qh, radar_dbz
            );
        }
        catch (const std::exception& e)
        {
            tmv::log_warn("[RADAR GUARD] microphysics fallback reflectivity failed: ", e.what(),
                         ". Keeping previous reflectivity field for this step.");
            return false;
        }

        if (radar_dbz.size_r() != NR || radar_dbz.size_th() != NTH || radar_dbz.size_z() != NZ)
        {
            tmv::log_warn("[RADAR GUARD] microphysics fallback returned unexpected dBZ field shape; "
                         "leaving reflectivity unchanged for this step.");
            return false;
        }

        #pragma omp parallel for collapse(2)
        for (int i = 0; i < NR; ++i)
        {
            for (int j = 0; j < NTH; ++j)
            {
                for (int k = 0; k < NZ; ++k)
                {
                    float z_dbz = static_cast<float>(radar_dbz[i][j][k]);
                    if (!std::isfinite(static_cast<double>(z_dbz)))
                    {
                        radar_reflectivity[i][j][k] = radar_linear_min;
                        continue;
                    }

                    z_dbz = std::clamp(z_dbz, -120.0f, 120.0f);
                    float z_linear = std::pow(10.0f, z_dbz / 10.0f);
                    if (!std::isfinite(static_cast<double>(z_linear)))
                    {
                        z_linear = radar_linear_max;
                    }
                    radar_reflectivity[i][j][k] =
                        std::clamp(z_linear, radar_linear_min, radar_linear_max);
                }
            }
        }

        sanitize_linear_reflectivity_field(radar_reflectivity, "microphysics_fallback");
        return true;
    };

    if (!radar_scheme)
    {
        tmv::log_warn("Radar scheme not initialized, using microphysics fallback");
        apply_microphysics_fallback();
        return;
    }

    RadarStateView state_view;
    state_view.NR = NR;
    state_view.NTH = NTH;
    state_view.NZ = NZ;

    state_view.u = &u;
    state_view.v = &v_theta;
    state_view.w = &w;

    state_view.qr = &qr;
    state_view.qs = &qs;
    state_view.qg = &qg;
    state_view.qh = &qh;
    state_view.qi = &qi;

    RadarConfig config;
    config.scheme_id = "reflectivity";
    config.operator_tier = "fast_da";
    config.has_qr = true;
    config.has_qs = true;
    config.has_qg = true;
    config.has_qh = true;
    config.has_qi = true;

    RadarOut radar_out;
    radar_out.initialize(NR, NTH, NZ);

    try
    {
        radar_scheme->compute(config, state_view, radar_out);
    }
    catch (const std::exception& e)
    {
        tmv::log_warn("[RADAR GUARD] radar scheme compute failed: ", e.what(),
                      ". Attempting microphysics fallback.");
        apply_microphysics_fallback();
        return;
    }

    int sanitized_reflectivity = 0;

    #pragma omp parallel for collapse(2) reduction(+:sanitized_reflectivity)
    for (int i = 0; i < NR; ++i) 
    {
        for (int j = 0; j < NTH; ++j) 
        {
            for (int k = 0; k < NZ; ++k) 
            {   
                float z_linear = static_cast<float>(radar_out.Ze_linear[i][j][k]);
                if (!std::isfinite(static_cast<double>(z_linear)) || z_linear < radar_linear_min)
                {
                    if (z_linear != radar_linear_min)
                    {
                        ++sanitized_reflectivity;
                    }
                    z_linear = radar_linear_min;
                }
                else if (z_linear > radar_linear_max)
                {
                    ++sanitized_reflectivity;
                    z_linear = radar_linear_max;
                }
                radar_reflectivity[i][j][k] = z_linear;
            }
        }
    }

    if (sanitized_reflectivity > 0)
    {
        tmv::log_warn("[RADAR GUARD] sanitized ", sanitized_reflectivity, " reflectivity samples from radar scheme output");
    }
}
