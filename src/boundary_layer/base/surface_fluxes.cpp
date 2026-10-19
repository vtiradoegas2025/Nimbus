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
#include <stdexcept>
#include "util/string_utils.hpp"



namespace tmv::surface_fluxes
{

namespace
{
constexpr double k_min_positive = 1.0e-6;

void require_non_negative_finite(double value, const char* label)
{
    if (!std::isfinite(value) || value < 0.0)
    {
        throw std::runtime_error(
            std::string("surface flux calculation produced invalid ") + label);
    }
}

template <typename Fluxes>
void validate_fluxes(const Fluxes& fluxes)
{
    require_non_negative_finite(fluxes.ustar, "ustar");
    require_non_negative_finite(fluxes.Cd, "Cd");
    require_non_negative_finite(fluxes.Ch, "Ch");
    require_non_negative_finite(fluxes.Ce, "Ce");
    if (!std::isfinite(fluxes.tau_u) || !std::isfinite(fluxes.tau_v) ||
        !std::isfinite(fluxes.H) || !std::isfinite(fluxes.E))
    {
        throw std::runtime_error(
            "surface flux calculation produced a non-finite physical flux");
    }
}

void enforce_minimum_ustar(BulkFluxes& fluxes, double min_ustar, double U)
{
    require_non_negative_finite(min_ustar, "minimum ustar");
    require_non_negative_finite(U, "wind magnitude");
    require_non_negative_finite(fluxes.ustar, "ustar");
    require_non_negative_finite(fluxes.Cd, "Cd");
    require_non_negative_finite(fluxes.Ch, "Ch");
    require_non_negative_finite(fluxes.Ce, "Ce");
    if (min_ustar <= 0.0 || U <= 0.0)
    {
        return;
    }
    if (fluxes.ustar >= min_ustar)
    {
        return;
    }

    const double required_cd = (min_ustar / U) * (min_ustar / U);
    const double cd_floor = std::max(fluxes.Cd, 1.0e-12);
    if (required_cd > cd_floor)
    {
        const double scale = required_cd / cd_floor;
        fluxes.Cd = required_cd;
        fluxes.Ch *= scale;
        fluxes.Ce *= scale;
    }
    fluxes.ustar = min_ustar;
    validate_fluxes(fluxes);
}

void enforce_minimum_ustar(MoninObukhovFluxes& fluxes, double min_ustar, double U)
{
    require_non_negative_finite(min_ustar, "minimum ustar");
    require_non_negative_finite(U, "wind magnitude");
    require_non_negative_finite(fluxes.ustar, "ustar");
    require_non_negative_finite(fluxes.Cd, "Cd");
    require_non_negative_finite(fluxes.Ch, "Ch");
    require_non_negative_finite(fluxes.Ce, "Ce");
    if (min_ustar <= 0.0 || U <= 0.0)
    {
        return;
    }
    if (fluxes.ustar >= min_ustar)
    {
        return;
    }

    const double required_cd = (min_ustar / U) * (min_ustar / U);
    const double cd_floor = std::max(fluxes.Cd, 1.0e-12);
    if (required_cd > cd_floor)
    {
        const double scale = required_cd / cd_floor;
        fluxes.Cd = required_cd;
        fluxes.Ch *= scale;
        fluxes.Ce *= scale;
    }
    fluxes.ustar = min_ustar;
    validate_fluxes(fluxes);
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
    if (id == "bulk" || id == "louis")
    {
        return "bulk";
    }
    return {};
}

void validate_surface_flux_inputs(
    const SurfaceConfig& sfc_cfg,
    double u_sfc, double v_sfc,
    double theta_sfc, double qv_sfc,
    double theta_air, double qv_air,
    double p_air, double rho_air,
    double z_sfc, double min_ustar)
{
    if (!std::isfinite(u_sfc) || !std::isfinite(v_sfc) ||
        !std::isfinite(theta_sfc) || theta_sfc <= 0.0 ||
        !std::isfinite(theta_air) || theta_air <= 0.0 ||
        !std::isfinite(qv_sfc) || qv_sfc < 0.0 || qv_sfc > 0.1 ||
        !std::isfinite(qv_air) || qv_air < 0.0 || qv_air > 0.1 ||
        !std::isfinite(p_air) || p_air <= 0.0 ||
        !std::isfinite(rho_air) || rho_air <= 0.0 ||
        !std::isfinite(sfc_cfg.z0m) || sfc_cfg.z0m <= 0.0 ||
        !std::isfinite(sfc_cfg.z0h) || sfc_cfg.z0h <= 0.0 ||
        !std::isfinite(z_sfc) ||
        z_sfc <= std::max(sfc_cfg.z0m, sfc_cfg.z0h) ||
        !std::isfinite(min_ustar) || min_ustar < 0.0)
    {
        throw std::invalid_argument(
            "surface fluxes require finite wind, positive theta/pressure/density/roughness, qv in [0,0.1], non-negative min_ustar, and a reference height above roughness");
    }
}

// Louis (1979) bulk surface-layer stability functions (BLM 17, 187-202; b=c=d=5).
// Multiplies the NEUTRAL transfer coefficient. Correctly signed and bounded:
//   stable   (Ri_b > 0): f in (0, 1], suppresses fluxes (-> 0 as Ri_b -> inf);
//   unstable (Ri_b < 0): f > 1, enhances fluxes, with the sqrt(z/z0 * |Ri_b|)
//                        denominator capping the growth (~sqrt(|Ri_b|), not the
//                        old unbounded linear 1 - 10*Ri_b).
// num_coeff is 2b for momentum and 3b for heat/moisture; c_neutral is the
// matching neutral coefficient (Cd_neutral for momentum, Ch_neutral for heat).
constexpr double k_louis_b = 5.0;
constexpr double k_louis_c = 5.0;
constexpr double k_louis_d = 5.0;

double louis_stability_factor(double Ri_b, double z_over_z0, double c_neutral,
                              double num_coeff)
{
    if (!std::isfinite(Ri_b) || !std::isfinite(z_over_z0) || z_over_z0 <= 1.0 ||
        !std::isfinite(c_neutral) || c_neutral < 0.0 ||
        !std::isfinite(num_coeff) || num_coeff <= 0.0)
    {
        throw std::invalid_argument("invalid Louis surface-layer inputs");
    }
    if (Ri_b >= 0.0)
    {
        const double denom = 1.0 + num_coeff * Ri_b /
                             std::sqrt(1.0 + k_louis_d * Ri_b);
        return 1.0 / denom;
    }
    const double denom = 1.0 + 3.0 * k_louis_b * k_louis_c * c_neutral *
                         std::sqrt(z_over_z0 * (-Ri_b));
    return 1.0 - (num_coeff * Ri_b) / denom;
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
    validate_surface_flux_inputs(
        sfc_cfg, u_sfc, v_sfc, theta_sfc, qv_sfc, theta_air,
        qv_air, p_air, rho_air, z_sfc, min_ustar);

    BulkFluxes fluxes;

    const double u_sfc_safe = u_sfc;
    const double v_sfc_safe = v_sfc;
    const double theta_sfc_safe = theta_sfc;
    const double theta_air_safe = theta_air;
    const double qv_sfc_safe = qv_sfc;
    const double qv_air_safe = qv_air;
    const double rho_air_safe = rho_air;
    const double z0m = sfc_cfg.z0m;
    const double z0h = sfc_cfg.z0h;
    const double z_ref = z_sfc;

    double U = std::sqrt(u_sfc_safe * u_sfc_safe + v_sfc_safe * v_sfc_safe);
    U = std::max(U, 0.1);

    double theta_v_sfc = virtual_potential_temperature(theta_sfc_safe, qv_sfc_safe);
    double theta_v_air = virtual_potential_temperature(theta_air_safe, qv_air_safe);

    double Ri_b = bulk_richardson_number(theta_v_sfc, theta_v_air, u_sfc_safe, v_sfc_safe, z_ref);

    // Neutral transfer coefficients. Heat uses the SEPARATE thermal roughness
    // z0h (Monin-Obukhov bulk transfer); the previous ad-hoc 0.75*Cd ignored z0h.
    const double ln_z_z0m = std::log(z_ref / z0m);
    const double ln_z_z0h = std::log(z_ref / z0h);
    const double Cd_neutral = std::pow(boundary_layer_constants::kappa / ln_z_z0m, 2);
    const double Ch_neutral = boundary_layer_constants::kappa * boundary_layer_constants::kappa /
                        (ln_z_z0m * ln_z_z0h);
    require_non_negative_finite(Cd_neutral, "neutral Cd");
    require_non_negative_finite(Ch_neutral, "neutral Ch");

    // Louis (1979) stability correction (replaces the unbounded ad-hoc factor):
    // warm/unstable (Ri_b < 0) enhances, stable (Ri_b > 0) suppresses, bounded.
    const double z_over_z0m = z_ref / z0m;
    const double f_m = louis_stability_factor(Ri_b, z_over_z0m, Cd_neutral, 2.0 * k_louis_b);
    const double f_h = louis_stability_factor(Ri_b, z_over_z0m, Ch_neutral, 3.0 * k_louis_b);

    fluxes.Cd = Cd_neutral * f_m;
    fluxes.Ch = Ch_neutral * f_h;
    fluxes.Ce = fluxes.Ch;

    fluxes.ustar = std::sqrt(fluxes.Cd) * U;
    enforce_minimum_ustar(fluxes, min_ustar, U);

    fluxes.tau_u = -rho_air_safe * fluxes.Cd * U * u_sfc_safe;
    fluxes.tau_v = -rho_air_safe * fluxes.Cd * U * v_sfc_safe;

    const double exner = std::pow(
        p_air / physical_constants::reference_pressure_pa,
        physical_constants::gas_constant_dry_air_jkgk /
            physical_constants::specific_heat_cp_jkgk);
    const double delta_theta = theta_sfc_safe - theta_air_safe;
    // H is true sensible-heat power per area [W m-2]. The transfer relation
    // operates on potential temperature, so convert its kinematic theta flux
    // to temperature/enthalpy currency with the local Exner function.
    fluxes.H = rho_air_safe * boundary_layer_constants::cp * exner *
               fluxes.Ch * U * delta_theta;

    double delta_qv = qv_sfc_safe - qv_air_safe;
    fluxes.E = rho_air_safe * fluxes.Ce * U * delta_qv;

    validate_fluxes(fluxes);
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
    validate_surface_flux_inputs(
        sfc_cfg, u_sfc, v_sfc, theta_sfc, qv_sfc, theta_air,
        qv_air, p_air, rho_air, z_sfc, min_ustar);

    MoninObukhovFluxes fluxes;

    const double u_sfc_safe = u_sfc;
    const double v_sfc_safe = v_sfc;
    const double theta_sfc_safe = theta_sfc;
    const double theta_air_safe = theta_air;
    const double qv_sfc_safe = qv_sfc;
    const double qv_air_safe = qv_air;
    const double rho_air_safe = rho_air;
    const double z0m = sfc_cfg.z0m;
    const double z0h = sfc_cfg.z0h;
    const double z_ref = z_sfc;

    double U = std::sqrt(u_sfc_safe * u_sfc_safe + v_sfc_safe * v_sfc_safe);
    U = std::max(U, 0.1);

    double theta_v_air = virtual_potential_temperature(theta_air_safe, qv_air_safe);

    double L_prev = 1e6;
    double L = L_prev;
    const int max_iter = 50;
    bool converged = false;

    for (int iter = 0; iter < max_iter; ++iter) 
    {
        fluxes.zeta = z_ref / L;

        double psi_m_val = psi_m(fluxes.zeta);
        double psi_h_val = psi_h(fluxes.zeta);

        double ln_z_z0m = std::log(z_ref / z0m);
        double ln_z_z0h = std::log(z_ref / z0h);
        double denom_m = ln_z_z0m - psi_m_val;
        double denom_h = ln_z_z0h - psi_h_val;
        if (std::abs(denom_m) < k_min_positive)
        {
            throw std::runtime_error(
                "Monin-Obukhov momentum denominator is singular");
        }
        if (std::abs(denom_h) < k_min_positive)
        {
            throw std::runtime_error(
                "Monin-Obukhov heat denominator is singular");
        }

        fluxes.Cd = std::pow(boundary_layer_constants::kappa / denom_m, 2);
        fluxes.Ch = boundary_layer_constants::kappa * boundary_layer_constants::kappa /
                   (denom_m * denom_h);
        require_non_negative_finite(fluxes.Cd, "Cd");
        require_non_negative_finite(fluxes.Ch, "Ch");

        fluxes.ustar = std::sqrt(fluxes.Cd) * U;
        fluxes.Ce = fluxes.Ch;
        enforce_minimum_ustar(fluxes, min_ustar, U);

        // Sensible heat uses the DRY potential-temperature difference; the virtual
        // (moisture) contribution belongs in the buoyancy flux used for L below,
        // not in the reported H. (Previously H used delta_theta_v.)
        const double exner = std::pow(
            p_air / physical_constants::reference_pressure_pa,
            physical_constants::gas_constant_dry_air_jkgk /
                physical_constants::specific_heat_cp_jkgk);
        const double delta_theta = theta_sfc_safe - theta_air_safe;
        fluxes.H = rho_air_safe * boundary_layer_constants::cp * exner *
                   fluxes.Ch * U * delta_theta;

        double delta_qv = qv_sfc_safe - qv_air_safe;
        fluxes.E = rho_air_safe * fluxes.Ce * U * delta_qv;

        // Monin-Obukhov length: L = -u*^3 theta_v / (kappa g w'theta_v'_s), where
        // w'theta_v'_s is the surface kinematic VIRTUAL heat flux. A warm/unstable
        // surface has H>0 -> w'theta_v'>0 -> L<0 -> zeta<0 (unstable psi branch).
        // (The previous form double-counted g/theta_v and flipped the sign, so an
        // unstable surface wrongly produced L>0 and applied stable corrections.)
        const double w_theta = fluxes.H /
            (rho_air_safe * boundary_layer_constants::cp * exner);
        const double w_qv = fluxes.E / rho_air_safe;
        const double w_theta_v = w_theta + 0.61 * theta_air_safe * w_qv;

        if (std::abs(w_theta_v) > 1e-10)
        {
            L = -fluxes.ustar * fluxes.ustar * fluxes.ustar * theta_v_air /
                (boundary_layer_constants::kappa * boundary_layer_constants::g * w_theta_v);
        }
        else
        {
            L = 1e6;
        }

        if (std::abs(L - L_prev) / std::max(std::abs(L_prev), k_min_positive) < 1e-3) 
        {
            converged = true;
            break;
        }
        L_prev = L;
    }

    if (!converged)
    {
        throw std::runtime_error(
            "Monin-Obukhov surface-layer iteration did not converge");
    }
    fluxes.L = L;
    fluxes.zeta = z_ref / L;

    fluxes.tau_u = -rho_air_safe * fluxes.Cd * U * u_sfc_safe;
    fluxes.tau_v = -rho_air_safe * fluxes.Cd * U * v_sfc_safe;

    validate_fluxes(fluxes);
    if (!std::isfinite(fluxes.L) || !std::isfinite(fluxes.zeta))
    {
        throw std::runtime_error(
            "Monin-Obukhov iteration produced non-finite similarity diagnostics");
    }
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
    double z_sfc,
    double surface_slope_x,
    double surface_slope_y
)
{
    if (!std::isfinite(surface_slope_x)
        || !std::isfinite(surface_slope_y))
        throw std::invalid_argument(
            "surface-layer terrain slopes must be finite");

    // A horizontal velocity pair q=(u,v) represents the terrain-tangent
    // physical velocity A q=(u,v,hx*u+hy*v). Factor G=A^T A=L^T L and run the
    // established surface equation on r=Lq. Transforming the traction back as
    // L^T*tau_r gives the generalized horizontal forces whose power is the
    // complete three-component tangent work. Surface heat/moisture fluxes are
    // converted from actual terrain area to horizontal column area.
    const double g11 = 1.0 + surface_slope_x * surface_slope_x;
    const double g12 = surface_slope_x * surface_slope_y;
    const double g22 = 1.0 + surface_slope_y * surface_slope_y;
    const double l11 = std::sqrt(g11);
    const double l12 = g12 / l11;
    const double l22_sq = g22 - l12 * l12;
    if (!std::isfinite(l22_sq) || l22_sq <= 0.0)
        throw std::invalid_argument(
            "surface-layer terrain metric is not positive definite");
    const double l22 = std::sqrt(l22_sq);
    const double tangent_u = l11 * u_sfc + l12 * v_sfc;
    const double tangent_v = l22 * v_sfc;
    const double area_scale = std::sqrt(
        1.0 + surface_slope_x * surface_slope_x
            + surface_slope_y * surface_slope_y);

    const std::string layer_id = normalize_surface_layer_id(
        bl_cfg.surface_layer_id);
    if (layer_id.empty())
        throw std::invalid_argument(
            "unknown surface-layer scheme: " + bl_cfg.surface_layer_id);
    if (layer_id == "monin_obukhov")
    {
        const MoninObukhovFluxes mo_fluxes = compute_monin_obukhov_fluxes(
            sfc_cfg,
            tangent_u,
            tangent_v,
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
        fluxes.tau_u = area_scale * l11 * mo_fluxes.tau_u;
        fluxes.tau_v = area_scale * (
            l12 * mo_fluxes.tau_u + l22 * mo_fluxes.tau_v);
        fluxes.H = area_scale * mo_fluxes.H;
        fluxes.E = area_scale * mo_fluxes.E;
        fluxes.Cd = mo_fluxes.Cd;
        fluxes.Ch = mo_fluxes.Ch;
        fluxes.Ce = mo_fluxes.Ce;
        return fluxes;
    }

    BulkFluxes fluxes = compute_bulk_fluxes(
        sfc_cfg,
        tangent_u,
        tangent_v,
        theta_sfc,
        qv_sfc,
        theta_air,
        qv_air,
        p_air,
        rho_air,
        z_sfc,
        bl_cfg.min_ustar);
    const double tangent_tau_u = fluxes.tau_u;
    const double tangent_tau_v = fluxes.tau_v;
    fluxes.tau_u = area_scale * l11 * tangent_tau_u;
    fluxes.tau_v = area_scale * (
        l12 * tangent_tau_u + l22 * tangent_tau_v);
    fluxes.H *= area_scale;
    fluxes.E *= area_scale;
    validate_fluxes(fluxes);
    return fluxes;
}

/**
 * @brief Computes the stability function for momentum.
 */
double psi_m(double zeta) 
{
    if (!std::isfinite(zeta))
    {
        throw std::invalid_argument("psi_m requires finite zeta");
    }
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
    if (!std::isfinite(zeta))
    {
        throw std::invalid_argument("psi_h requires finite zeta");
    }
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
    if (!std::isfinite(theta_v_sfc) || theta_v_sfc <= 0.0 ||
        !std::isfinite(theta_v_air) || theta_v_air <= 0.0 ||
        !std::isfinite(u_sfc) || !std::isfinite(v_sfc) ||
        !std::isfinite(z_sfc) || z_sfc <= 0.0)
    {
        throw std::invalid_argument(
            "bulk Richardson number requires finite positive thermodynamics/height and finite wind");
    }
    // Standard bulk Richardson convention: theta_v(upper/air) - theta_v(surface).
    // A warm/unstable surface (theta_v_sfc > theta_v_air) gives Ri_b < 0; a stable
    // surface gives Ri_b > 0. (Previously sfc - air, which inverted the sign and
    // made warm surfaces suppress fluxes + broke the Ri_crit PBL-top search.)
    double delta_theta_v = theta_v_air - theta_v_sfc;
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
    if (!std::isfinite(theta) || theta <= 0.0 || !std::isfinite(qv) ||
        qv < 0.0 || qv > 0.1)
    {
        throw std::invalid_argument(
            "virtual potential temperature requires positive theta and qv in [0,0.1]");
    }
    return theta * (1.0 + 0.61 * qv);
}

} // namespace tmv::surface_fluxes
