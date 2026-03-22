/**
 * @file surface_fluxes.cpp
 * @brief Implementation for the boundary_layer module.
 *
 * Provides executable logic for the boundary_layer runtime path,
 * including initialization, stepping, and diagnostics helpers.
 * This file is part of the src/boundary_layer subsystem.
 */

#include "surface_fluxes.hpp"
#include <cmath>
#include <algorithm>
#include "util/string_utils.hpp"



namespace surface_fluxes 
{

namespace
{
constexpr double k_min_positive = 1.0e-6;

double sanitize_non_negative(double value, double fallback = 0.0)
{
    if (!std::isfinite(value))
    {
        return fallback;
    }
    return std::max(value, 0.0);
}

double sanitize_positive(double value, double fallback = k_min_positive)
{
    if (!std::isfinite(value) || value <= k_min_positive)
    {
        return fallback;
    }
    return value;
}

double sanitize_reference_height(double z_sfc, double z0m, double z0h)
{
    const double min_height = std::max(z0m, z0h) * 1.01;
    return std::max(sanitize_positive(z_sfc, min_height), min_height);
}

void enforce_minimum_ustar(BulkFluxes& fluxes, double min_ustar, double U)
{
    const double min_ustar_safe = sanitize_non_negative(min_ustar, 0.0);
    fluxes.Cd = sanitize_non_negative(fluxes.Cd, 0.0);
    fluxes.Ch = sanitize_non_negative(fluxes.Ch, 0.0);
    fluxes.Ce = sanitize_non_negative(fluxes.Ce, 0.0);

    if (!std::isfinite(fluxes.ustar))
    {
        fluxes.ustar = 0.0;
    }
    if (min_ustar_safe <= 0.0 || U <= 0.0 || !std::isfinite(U))
    {
        return;
    }
    if (fluxes.ustar >= min_ustar_safe)
    {
        return;
    }

    const double required_cd = (min_ustar_safe / U) * (min_ustar_safe / U);
    const double cd_floor = std::max(fluxes.Cd, 1.0e-12);
    if (required_cd > cd_floor)
    {
        const double scale = required_cd / cd_floor;
        fluxes.Cd = required_cd;
        fluxes.Ch *= scale;
        fluxes.Ce *= scale;
    }
    fluxes.ustar = min_ustar_safe;
}

void enforce_minimum_ustar(MoninObukhovFluxes& fluxes, double min_ustar, double U)
{
    const double min_ustar_safe = sanitize_non_negative(min_ustar, 0.0);
    fluxes.Cd = sanitize_non_negative(fluxes.Cd, 0.0);
    fluxes.Ch = sanitize_non_negative(fluxes.Ch, 0.0);
    fluxes.Ce = sanitize_non_negative(fluxes.Ce, 0.0);

    if (!std::isfinite(fluxes.ustar))
    {
        fluxes.ustar = 0.0;
    }
    if (min_ustar_safe <= 0.0 || U <= 0.0 || !std::isfinite(U))
    {
        return;
    }
    if (fluxes.ustar >= min_ustar_safe)
    {
        return;
    }

    const double required_cd = (min_ustar_safe / U) * (min_ustar_safe / U);
    const double cd_floor = std::max(fluxes.Cd, 1.0e-12);
    if (required_cd > cd_floor)
    {
        const double scale = required_cd / cd_floor;
        fluxes.Cd = required_cd;
        fluxes.Ch *= scale;
        fluxes.Ce *= scale;
    }
    fluxes.ustar = min_ustar_safe;
}

std::string normalize_surface_layer_id(std::string id)
{
    id = tmv::strutil::lower_copy(id);
    if (id == "monin_obukhov" ||
        id == "monin-obukhov" ||
        id == "moninobukhov" ||
        id == "most" ||
        id == "mo")
    {
        return "monin_obukhov";
    }
    return "bulk";
}
}

/**
 * @brief Computes the bulk fluxes of the simulation.
 */
BulkFluxes compute_bulk_fluxes(
    const SurfaceConfig& sfc_cfg,
    double u_sfc, double v_sfc,
    double theta_sfc, double qv_sfc,
    double theta_air, double qv_air,
    double p_air, double rho_air,
    double z_sfc,
    double min_ustar
) 
{
    (void)p_air;

    BulkFluxes fluxes;

    const double u_sfc_safe = std::isfinite(u_sfc) ? u_sfc : 0.0;
    const double v_sfc_safe = std::isfinite(v_sfc) ? v_sfc : 0.0;
    const double theta_sfc_safe = sanitize_positive(theta_sfc, 300.0);
    const double theta_air_safe = sanitize_positive(theta_air, theta_sfc_safe);
    const double qv_sfc_safe = sanitize_non_negative(qv_sfc, 0.0);
    const double qv_air_safe = sanitize_non_negative(qv_air, 0.0);
    const double rho_air_safe = sanitize_positive(rho_air, 1.0);
    const double z0m = sanitize_positive(sfc_cfg.z0m);
    const double z0h = sanitize_positive(sfc_cfg.z0h);
    const double z_ref = sanitize_reference_height(z_sfc, z0m, z0h);

    double U = std::sqrt(u_sfc_safe * u_sfc_safe + v_sfc_safe * v_sfc_safe);
    U = std::max(U, 0.1);

    double theta_v_sfc = virtual_potential_temperature(theta_sfc_safe, qv_sfc_safe);
    double theta_v_air = virtual_potential_temperature(theta_air_safe, qv_air_safe);

    double Ri_b = bulk_richardson_number(theta_v_sfc, theta_v_air, u_sfc_safe, v_sfc_safe, z_ref);

    double stability_factor = 1.0;

    if (Ri_b > 0.0) 
    {
        stability_factor = std::max(0.1, 1.0 - 5.0 * Ri_b);
    } 
    else 
    {
        stability_factor = 1.0 - 10.0 * Ri_b;
    }

    const double ln_z_z0m = std::max(std::log(z_ref / z0m), k_min_positive);
    double Cd_neutral = std::pow(boundary_layer_constants::kappa / ln_z_z0m, 2);
    if (!std::isfinite(Cd_neutral) || Cd_neutral < 0.0)
    {
        Cd_neutral = 0.0;
    }

    fluxes.Cd = Cd_neutral * stability_factor;
    fluxes.Ch = 0.75 * Cd_neutral * stability_factor;
    fluxes.Ce = fluxes.Ch;
    fluxes.Cd = sanitize_non_negative(fluxes.Cd, 0.0);
    fluxes.Ch = sanitize_non_negative(fluxes.Ch, 0.0);
    fluxes.Ce = sanitize_non_negative(fluxes.Ce, 0.0);

    fluxes.ustar = std::sqrt(fluxes.Cd) * U;
    enforce_minimum_ustar(fluxes, min_ustar, U);

    fluxes.tau_u = -rho_air_safe * fluxes.Cd * U * u_sfc_safe;
    fluxes.tau_v = -rho_air_safe * fluxes.Cd * U * v_sfc_safe;

    double delta_theta = theta_sfc_safe - theta_air_safe;
    fluxes.H = rho_air_safe * boundary_layer_constants::cp * fluxes.Ch * U * delta_theta;

    double delta_qv = qv_sfc_safe - qv_air_safe;
    fluxes.E = rho_air_safe * fluxes.Ce * U * delta_qv;

    return fluxes;
}

/**
 * @brief Computes the Monin-Obukhov fluxes of the simulation.
 */
MoninObukhovFluxes compute_monin_obukhov_fluxes(
    const SurfaceConfig& sfc_cfg,
    double u_sfc, double v_sfc,
    double theta_sfc, double qv_sfc,
    double theta_air, double qv_air,
    double p_air, double rho_air,
    double z_sfc,
    double min_ustar
) 
{
    (void)p_air;

    MoninObukhovFluxes fluxes;

    const double u_sfc_safe = std::isfinite(u_sfc) ? u_sfc : 0.0;
    const double v_sfc_safe = std::isfinite(v_sfc) ? v_sfc : 0.0;
    const double theta_sfc_safe = sanitize_positive(theta_sfc, 300.0);
    const double theta_air_safe = sanitize_positive(theta_air, theta_sfc_safe);
    const double qv_sfc_safe = sanitize_non_negative(qv_sfc, 0.0);
    const double qv_air_safe = sanitize_non_negative(qv_air, 0.0);
    const double rho_air_safe = sanitize_positive(rho_air, 1.0);
    const double z0m = sanitize_positive(sfc_cfg.z0m);
    const double z0h = sanitize_positive(sfc_cfg.z0h);
    const double z_ref = sanitize_reference_height(z_sfc, z0m, z0h);

    double U = std::sqrt(u_sfc_safe * u_sfc_safe + v_sfc_safe * v_sfc_safe);
    U = std::max(U, 0.1);

    double theta_v_sfc = virtual_potential_temperature(theta_sfc_safe, qv_sfc_safe);
    double theta_v_air = virtual_potential_temperature(theta_air_safe, qv_air_safe);

    double L_prev = 1e6;
    double L = L_prev;
    const int max_iter = 10;

    for (int iter = 0; iter < max_iter; ++iter) 
    {
        fluxes.zeta = z_ref / L;

        double psi_m_val = psi_m(fluxes.zeta);
        double psi_h_val = psi_h(fluxes.zeta);

        double ln_z_z0m = std::log(z_ref / z0m);
        double ln_z_z0h = std::log(z_ref / z0h);
        ln_z_z0m = std::max(ln_z_z0m, k_min_positive);
        ln_z_z0h = std::max(ln_z_z0h, k_min_positive);
        double denom_m = ln_z_z0m - psi_m_val;
        double denom_h = ln_z_z0h - psi_h_val;
        if (std::abs(denom_m) < k_min_positive)
        {
            denom_m = (denom_m >= 0.0) ? k_min_positive : -k_min_positive;
        }
        if (std::abs(denom_h) < k_min_positive)
        {
            denom_h = (denom_h >= 0.0) ? k_min_positive : -k_min_positive;
        }

        fluxes.Cd = std::pow(boundary_layer_constants::kappa / denom_m, 2);
        fluxes.Ch = boundary_layer_constants::kappa * boundary_layer_constants::kappa /
                   (denom_m * denom_h);
        fluxes.Cd = sanitize_non_negative(fluxes.Cd, 0.0);
        fluxes.Ch = sanitize_non_negative(fluxes.Ch, 0.0);

        fluxes.ustar = std::sqrt(fluxes.Cd) * U;
        fluxes.Ce = fluxes.Ch;
        enforce_minimum_ustar(fluxes, min_ustar, U);

        double delta_theta_v = theta_v_sfc - theta_v_air;
        fluxes.H = rho_air_safe * boundary_layer_constants::cp * fluxes.Ch * U * delta_theta_v;

        double delta_qv = qv_sfc_safe - qv_air_safe;
        fluxes.E = rho_air_safe * fluxes.Ce * U * delta_qv;

        const double ustar_safe = std::max(fluxes.ustar, k_min_positive);
        double theta_star = -fluxes.H / (rho_air_safe * boundary_layer_constants::cp * ustar_safe);
        double qv_star = -fluxes.E / (rho_air_safe * ustar_safe);

        double buoyancy_flux = boundary_layer_constants::g / theta_v_air *
                              (theta_star + 0.61 * theta_air_safe * qv_star);

        if (std::abs(buoyancy_flux) > 1e-10) 
        {
            L = -fluxes.ustar * fluxes.ustar * fluxes.ustar *
                theta_v_air / (boundary_layer_constants::kappa * boundary_layer_constants::g * buoyancy_flux);
        } 
        else 
        {
            L = 1e6;
        }

        if (std::abs(L - L_prev) / std::max(std::abs(L_prev), k_min_positive) < 1e-3) 
        {
            break;
        }
        L_prev = L;
    }

    fluxes.L = L;

    fluxes.tau_u = -rho_air_safe * fluxes.Cd * U * u_sfc_safe;
    fluxes.tau_v = -rho_air_safe * fluxes.Cd * U * v_sfc_safe;

    return fluxes;
}

BulkFluxes compute_surface_fluxes(
    const BoundaryLayerConfig& bl_cfg,
    const SurfaceConfig& sfc_cfg,
    double u_sfc,
    double v_sfc,
    double theta_sfc,
    double qv_sfc,
    double theta_air,
    double qv_air,
    double p_air,
    double rho_air,
    double z_sfc
)
{
    if (normalize_surface_layer_id(bl_cfg.surface_layer_id) == "monin_obukhov")
    {
        const MoninObukhovFluxes mo_fluxes = compute_monin_obukhov_fluxes(
            sfc_cfg,
            u_sfc,
            v_sfc,
            theta_sfc,
            qv_sfc,
            theta_air,
            qv_air,
            p_air,
            rho_air,
            z_sfc,
            bl_cfg.min_ustar);
        BulkFluxes fluxes;
        fluxes.ustar = mo_fluxes.ustar;
        fluxes.tau_u = mo_fluxes.tau_u;
        fluxes.tau_v = mo_fluxes.tau_v;
        fluxes.H = mo_fluxes.H;
        fluxes.E = mo_fluxes.E;
        fluxes.Cd = mo_fluxes.Cd;
        fluxes.Ch = mo_fluxes.Ch;
        fluxes.Ce = mo_fluxes.Ce;
        return fluxes;
    }

    return compute_bulk_fluxes(
        sfc_cfg,
        u_sfc,
        v_sfc,
        theta_sfc,
        qv_sfc,
        theta_air,
        qv_air,
        p_air,
        rho_air,
        z_sfc,
        bl_cfg.min_ustar);
}

/**
 * @brief Computes the stability function for momentum.
 */
double psi_m(double zeta) 
{
    if (zeta >= 0.0) 
    {
        double x = (1.0 + 5.0 * zeta);
        return -5.0 * zeta - 5.0 * zeta * zeta / x;
    } 
    else 
    {
        double x = std::pow(1.0 - 16.0 * zeta, 0.25);
        return 2.0 * std::log((1.0 + x) / 2.0) + std::log((1.0 + x * x) / 2.0) -
               2.0 * std::atan(x) + boundary_layer_constants::pi / 2.0;
    }
}

/**
 * @brief Computes the stability function for heat.
 */
double psi_h(double zeta) 
{
    if (zeta >= 0.0) 
    {
        return -5.0 * zeta;
    } 
    else 
    {
        double x = std::pow(1.0 - 16.0 * zeta, 0.25);
        return 2.0 * std::log((1.0 + x * x) / 2.0);
    }
}

/**
 * @brief Computes the bulk Richardson number.
 */
double bulk_richardson_number(double theta_v_sfc, double theta_v_air,double u_sfc, double v_sfc, double z_sfc) 
{
    double delta_theta_v = theta_v_sfc - theta_v_air;
    double U = std::sqrt(u_sfc * u_sfc + v_sfc * v_sfc);
    U = std::max(U, 0.1);

    return boundary_layer_constants::g * delta_theta_v * z_sfc /
           (theta_v_air * U * U);
}

/**
 * @brief Computes the virtual potential temperature.
 */
double virtual_potential_temperature(double theta, double qv) 
{
    return theta * (1.0 + 0.61 * qv);
}

/**
 * @brief Converts the potential temperature tendency to the temperature tendency.
 */
void convert_theta_tendency_to_temperature(
    const std::vector<double>& dthetadt,
    const std::vector<double>& theta,
    const std::vector<double>& p,
    std::vector<double>& dTdt
) 
{
    const size_t nz = dthetadt.size();
    dTdt.resize(nz);

    const double kappa = R_d / boundary_layer_constants::cp;

    for (size_t k = 0; k < nz; ++k) 
    {
        dTdt[k] = dthetadt[k] * std::pow(p[k] / p0, -kappa);
    }
}

}
