#include "boundary_layer/base/implicit_column_exchange.hpp"
#include "ysu.hpp"

#include "boundary_layer/base/cloud_top_radiative_mixing.hpp"
#include "boundary_layer/base/pbl_height.hpp"
#include "boundary_layer/base/vertical_diffusion.hpp"
#include "util/log.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace tmv
{
namespace
{

// Hong, Noh, and Dudhia (2006), Appendix A; constants match NCAR bl_ysu.F90.
constexpr double countergradient_coefficient = 6.8;
constexpr double convective_velocity_weight = 8.0;
constexpr double entrainment_flux_ratio = 0.15;
constexpr double maximum_diffusivity = 1000.0;

std::size_t validate_column(
    const BoundaryLayerConfig& cfg,
    const BoundaryLayerColumnStateView& col)
{
    if (!std::isfinite(cfg.dt_pbl) || cfg.dt_pbl <= 0.0
        || !std::isfinite(cfg.pbl_max_height) || cfg.pbl_max_height <= 0.0)
        throw std::invalid_argument(
            "YSU requires positive finite cadence and maximum PBL height");

    const std::size_t nz = col.theta.size();
    if (nz < 2 || !col.u || !col.v || !col.theta || !col.qv || !col.p
        || !col.rho || !col.z_int || col.u.size() != nz
        || col.v.size() != nz || col.qv.size() != nz
        || col.p.size() != nz || col.rho.size() != nz
        || col.z_int.size() != nz + 1)
        throw std::invalid_argument(
            "YSU requires consistently shaped column profiles");

    for (std::size_t k = 0; k < nz; ++k)
    {
        if (!std::isfinite(col.u[k]) || !std::isfinite(col.v[k])
            || !std::isfinite(col.theta[k]) || col.theta[k] <= 0.0
            || !std::isfinite(col.qv[k]) || col.qv[k] < 0.0
            || !std::isfinite(col.p[k]) || col.p[k] <= 0.0
            || !std::isfinite(col.rho[k]) || col.rho[k] <= 0.0
            || !std::isfinite(col.z_int[k])
            || !std::isfinite(col.z_int[k + 1])
            || col.z_int[k + 1] <= col.z_int[k]
            || (k > 0 && col.p[k] > col.p[k - 1]))
            throw std::invalid_argument(
                "YSU column contains non-finite or unphysical state");
    }
    return nz;
}

double virtual_theta(const BoundaryLayerColumnStateView& col, std::size_t k)
{
    return surface_fluxes::virtual_potential_temperature(
        col.theta[k], col.qv[k]);
}

double center_height(const BoundaryLayerColumnStateView& col, std::size_t k)
{
    return 0.5 * (col.z_int[k] + col.z_int[k + 1]);
}

double surface_exner(const BoundaryLayerColumnStateView& col)
{
    return std::pow(
        col.p[0] / physical_constants::reference_pressure_pa,
        physical_constants::gas_constant_dry_air_jkgk
            / physical_constants::specific_heat_cp_jkgk);
}

double bounded_mixed_layer_speed(double ustar, double wstar)
{
    double speed = std::cbrt(
        ustar * ustar * ustar
        + convective_velocity_weight * boundary_layer_constants::kappa
            * wstar * wstar * wstar * 0.5);
    if (ustar > 0.0)
        speed = std::clamp(speed, ustar / 5.0, ustar * 16.0);
    return speed;
}

} // namespace

YSUScheme::YSUScheme() = default;

int YSUScheme::required_fields() const
{
    return static_cast<int>(BoundaryLayerRequirements::BASIC)
        | static_cast<int>(BoundaryLayerRequirements::MOISTURE);
}

void YSUScheme::initialize(const BoundaryLayerConfig& cfg)
{
    if (!std::isfinite(cfg.pbl_max_height) || cfg.pbl_max_height <= 0.0)
        throw std::invalid_argument(
            "YSU requires a positive finite maximum PBL height");
    tmv::log_info(
        "Initialized YSU Boundary Layer (nonlocal K-profile with explicit entrainment)");
}

void YSUScheme::compute_column(
    const BoundaryLayerConfig& cfg,
    const SurfaceConfig& sfc,
    const BoundaryLayerColumnStateView& col,
    BoundaryLayerTendencies& tend,
    BoundaryLayerDiagnostics* diag_opt)
{
    const std::size_t nz = validate_column(cfg, col);

    BoundaryLayerTendencies candidate;
    candidate.dtkedt_pbl.assign(nz, 0.0);
    BoundaryLayerDiagnostics candidate_diagnostics;
    if (diag_opt)
        candidate_diagnostics.include_diffusivity_profiles =
            diag_opt->include_diffusivity_profiles;

    BoundaryLayerColumnWorkspace local_workspace;
    BoundaryLayerColumnWorkspace& workspace = col.workspace
        ? *col.workspace
        : local_workspace;

    surface_fluxes::BulkFluxes fluxes;
    if (cfg.apply_surface_fluxes)
    {
        fluxes = surface_fluxes::compute_surface_fluxes(
            cfg, sfc, col.u_sfc, col.v_sfc, col.theta_sfc, col.qv_sfc,
            col.theta[0], col.qv[0], col.p[0], col.rho[0], col.z_sfc,
            col.surface_slope_x, col.surface_slope_y);
    }

    const double theta_flux = fluxes.H
        / (boundary_layer_constants::cp * surface_exner(col));
    const double virtual_heat_flux = theta_flux / col.rho[0]
        + 0.61 * col.theta[0] * fluxes.E / col.rho[0];
    const double initial_h = boundary_layer::pbl::diagnose_pbl_height_ri(
        col.theta, col.qv, col.u, col.v, col.z_int,
        0.25, 100.0, cfg.pbl_max_height);
    double wstar = virtual_heat_flux > 0.0
        ? std::cbrt(boundary_layer_constants::g / virtual_theta(col, 0)
                       * virtual_heat_flux * initial_h)
        : 0.0;
    double mixing_velocity = bounded_mixed_layer_speed(fluxes.ustar, wstar);
    const double h = diagnose_pbl_height(
        col, cfg, virtual_heat_flux, mixing_velocity);
    wstar = virtual_heat_flux > 0.0
        ? std::cbrt(boundary_layer_constants::g / virtual_theta(col, 0)
                       * virtual_heat_flux * h)
        : 0.0;

    double monin_obukhov_length = 1.0e30;
    if (std::abs(virtual_heat_flux) > 1.0e-14
        && fluxes.ustar > 0.0)
    {
        monin_obukhov_length =
            -fluxes.ustar * fluxes.ustar * fluxes.ustar
            * virtual_theta(col, 0)
            / (boundary_layer_constants::kappa
               * boundary_layer_constants::g * virtual_heat_flux);
    }

    workspace.K_m.assign(nz + 1, 0.0);
    workspace.K_h.assign(nz + 1, 0.0);
    compute_k_profile(
        col, h, fluxes.ustar, wstar, monin_obukhov_length,
        workspace.K_m, workspace.K_h);
    if (!cfg.clear_sky_only)
    {
        auto cloud_profile =
            boundary_layer::cloud_top::diagnose_surface_coupled_profile(
                col.theta, col.qc, col.qi,
                col.radiative_theta_tendency, col.z_int, h,
                maximum_diffusivity);
        workspace.cloud_top_diffusivity =
            std::move(cloud_profile.diffusivity_m2s);
        candidate_diagnostics.cloud_top_radiative_mixing_active =
            cloud_profile.active;
        candidate_diagnostics.cloud_top_height =
            cloud_profile.cloud_top_height_m;
        candidate_diagnostics.cloud_top_cooling_flux_kms =
            cloud_profile.cooling_flux_kms;
        candidate_diagnostics.cloud_top_velocity_scale =
            cloud_profile.velocity_scale_ms;
    }
    apply_diffusion_tendencies(
        col, workspace.K_m, workspace.K_h, fluxes, cfg.dt_pbl, h, wstar,
        workspace, candidate);
    if (!cfg.clear_sky_only)
    {
        apply_cloud_top_diffusion_tendencies(
            col, workspace.cloud_top_diffusivity, cfg.dt_pbl, workspace, candidate);
        for (std::size_t face = 1; face < nz; ++face)
        {
            workspace.K_m[face] = std::clamp(
                workspace.K_m[face]
                    + workspace.cloud_top_diffusivity[face],
                0.0, maximum_diffusivity);
            workspace.K_h[face] = std::clamp(
                workspace.K_h[face]
                    + workspace.cloud_top_diffusivity[face],
                0.0, maximum_diffusivity);
        }
    }

    if (diag_opt)
    {
        candidate_diagnostics.pbl_height = h;
        candidate_diagnostics.ustar = fluxes.ustar;
        candidate_diagnostics.tau_u = fluxes.tau_u;
        candidate_diagnostics.tau_v = fluxes.tau_v;
        candidate_diagnostics.H = fluxes.H;
        candidate_diagnostics.E = fluxes.E;
        candidate_diagnostics.Cd = fluxes.Cd;
        candidate_diagnostics.Ch = fluxes.Ch;
        candidate_diagnostics.Ce = fluxes.Ce;
        if (candidate_diagnostics.include_diffusivity_profiles)
        {
            candidate_diagnostics.K_m = workspace.K_m;
            candidate_diagnostics.K_h = workspace.K_h;
            candidate_diagnostics.K_e.assign(nz + 1, 0.0);
        }
    }

    tend = std::move(candidate);
    if (diag_opt)
        *diag_opt = std::move(candidate_diagnostics);
}

double YSUScheme::diagnose_pbl_height(
    const BoundaryLayerColumnStateView& col,
    const BoundaryLayerConfig& cfg,
    double surface_virtual_heat_flux,
    double mixing_velocity) const
{
    if (cfg.pbl_top_method == "theta")
        return boundary_layer::pbl::diagnose_pbl_height_theta(
            col.theta, col.z_int,
            boundary_layer::pbl::theta_increase_threshold_k,
            100.0, cfg.pbl_max_height);
    if (surface_virtual_heat_flux <= 0.0 || mixing_velocity <= 0.0)
    {
        return boundary_layer::pbl::diagnose_pbl_height_ri(
            col.theta, col.qv, col.u, col.v, col.z_int,
            0.25, 100.0, cfg.pbl_max_height);
    }

    const double theta_v0 = virtual_theta(col, 0);
    const double thermal_excess = countergradient_coefficient
        * surface_virtual_heat_flux / mixing_velocity;
    double previous_ri = -1.0;
    double previous_z = center_height(col, 0);
    for (std::size_t k = 1; k < col.theta.size(); ++k)
    {
        const double z = center_height(col, k);
        const double speed2 = std::max(
            col.u[k] * col.u[k] + col.v[k] * col.v[k], 1.0);
        const double ri = boundary_layer_constants::g
            * (virtual_theta(col, k) - theta_v0 - thermal_excess)
            * z / (theta_v0 * speed2);
        if (std::isfinite(ri) && ri >= 0.0)
        {
            double h = z;
            if (ri > previous_ri)
                h = previous_z + (z - previous_z)
                    * (-previous_ri) / (ri - previous_ri);
            return std::clamp(h, 100.0, cfg.pbl_max_height);
        }
        previous_ri = ri;
        previous_z = z;
    }
    return cfg.pbl_max_height;
}

void YSUScheme::compute_k_profile(
    const BoundaryLayerColumnStateView& col,
    double h,
    double ustar,
    double wstar,
    double monin_obukhov_length,
    std::vector<double>& K_m,
    std::vector<double>& K_h) const
{
    const std::size_t nz = col.theta.size();
    const double ustar3 = ustar * ustar * ustar;
    const double wstar3 = wstar * wstar * wstar;
    const double surface_layer_stability = 0.1 * h / monin_obukhov_length;
    const bool unstable = wstar > 0.0;
    const double phi_m = unstable
        ? std::pow(1.0 - 16.0 * surface_layer_stability, -0.25)
        : std::max(1.0 + 5.0 * surface_layer_stability, 1.0e-3);
    const double phi_h = unstable
        ? std::pow(1.0 - 16.0 * surface_layer_stability, -0.5)
        : phi_m;
    const double prandtl_surface = std::clamp(
        phi_h / phi_m
            + (unstable ? countergradient_coefficient
                              * boundary_layer_constants::kappa * 0.1
                        : 0.0),
        0.25, 4.0);

    for (std::size_t face = 1; face < nz; ++face)
    {
        const double z = col.z_int[face];
        if (z < h)
        {
            const double eta = std::clamp(z / h, 0.0, 1.0);
            double velocity = unstable
                ? std::cbrt(
                      ustar3 + convective_velocity_weight
                          * boundary_layer_constants::kappa * wstar3 * eta)
                : ustar / std::max(
                      1.0 + 5.0 * z / monin_obukhov_length, 1.0e-3);
            velocity = std::max(velocity, 0.001);
            const double shape = (1.0 - eta) * (1.0 - eta);
            const double km = boundary_layer_constants::kappa
                * velocity * z * shape;
            const double height_above_surface_layer =
                std::max(z - 0.1 * h, 0.0);
            const double prandtl = 1.0 + (prandtl_surface - 1.0)
                * std::exp(-3.0 * height_above_surface_layer
                           * height_above_surface_layer / (h * h));
            K_m[face] = std::clamp(km, 0.0, maximum_diffusivity);
            K_h[face] = std::clamp(
                km / prandtl, 0.0, maximum_diffusivity);
            continue;
        }

        const double dz = center_height(col, face)
            - center_height(col, face - 1);
        const double du_dz = (col.u[face] - col.u[face - 1]) / dz;
        const double dv_dz = (col.v[face] - col.v[face - 1]) / dz;
        const double shear2 = du_dz * du_dz + dv_dz * dv_dz + 1.0e-9;
        const double theta_v_mean = 0.5
            * (virtual_theta(col, face) + virtual_theta(col, face - 1));
        double ri = boundary_layer_constants::g / theta_v_mean
            * (virtual_theta(col, face) - virtual_theta(col, face - 1))
            / (shear2 * dz);
        const double outer_length = std::min(
            dz, std::clamp(0.1 * dz, 30.0, 300.0));
        const double wall_length = boundary_layer_constants::kappa * z;
        const double mixing_length = wall_length * outer_length
            / (outer_length + wall_length);
        const double neutral_diffusivity = mixing_length * mixing_length
            * std::sqrt(shear2);
        double km;
        double kh;
        if (ri < 0.0)
        {
            ri = std::max(ri, -100.0);
            const double root = std::sqrt(-ri);
            km = neutral_diffusivity
                * (1.0 + 8.0 * (-ri) / (1.0 + 1.746 * root));
            kh = neutral_diffusivity
                * (1.0 + 8.0 * (-ri) / (1.0 + 1.286 * root));
        }
        else
        {
            kh = neutral_diffusivity / std::pow(1.0 + 5.0 * ri, 2);
            km = kh * std::min(1.0 + 2.1 * ri, 4.0);
        }
        K_m[face] = std::clamp(km + 0.1, 0.0, maximum_diffusivity);
        K_h[face] = std::clamp(kh + 0.01, 0.0, maximum_diffusivity);
    }
}

void YSUScheme::apply_diffusion_tendencies(
    const BoundaryLayerColumnStateView& col,
    const std::vector<double>& K_m,
    const std::vector<double>& K_h,
    surface_fluxes::BulkFluxes& fluxes,
    double interval_s,
    double h,
    double wstar,
    BoundaryLayerColumnWorkspace& workspace,
    BoundaryLayerTendencies& tend) const
{
    const std::size_t nz = col.theta.size();
    const double theta_surface_flux = fluxes.H
        / (boundary_layer_constants::cp * surface_exner(col));
    const double mixing_velocity = bounded_mixed_layer_speed(
        fluxes.ustar, wstar);

    workspace.nonlocal_u_flux.assign(nz + 1, 0.0);
    workspace.nonlocal_v_flux.assign(nz + 1, 0.0);
    workspace.nonlocal_theta_flux.assign(nz + 1, 0.0);
    workspace.nonlocal_qv_flux.assign(nz + 1, 0.0);

    std::size_t top_face = nz;
    for (std::size_t face = 1; face < nz; ++face)
    {
        if (col.z_int[face] >= h)
        {
            top_face = face;
            break;
        }
    }

    double entrainment_u = 0.0;
    double entrainment_v = 0.0;
    double entrainment_theta = 0.0;
    double entrainment_qv = 0.0;
    if (top_face < nz && wstar > 0.0 && mixing_velocity > 0.0)
    {
        const std::size_t below = top_face - 1;
        const std::size_t above = top_face;
        const double wm3 = wstar * wstar * wstar
            + 5.0 * fluxes.ustar * fluxes.ustar * fluxes.ustar;
        const double buoyancy_flux_at_top = -entrainment_flux_ratio
            * virtual_theta(col, 0) / boundary_layer_constants::g * wm3 / h;
        const double inversion = std::max(
            virtual_theta(col, above) - virtual_theta(col, below), 0.01);
        const double entrainment_velocity = std::max(
            buoyancy_flux_at_top / inversion, -std::cbrt(wm3));
        entrainment_theta = entrainment_velocity
            * std::max(col.theta[above] - col.theta[below], 0.01);
        entrainment_qv = entrainment_velocity
            * std::min(col.qv[above] - col.qv[below], 0.0);
        entrainment_u = std::clamp(
            entrainment_velocity * (col.u[above] - col.u[below]),
            -fluxes.ustar * fluxes.ustar,
            fluxes.ustar * fluxes.ustar);
        entrainment_v = std::clamp(
            entrainment_velocity * (col.v[above] - col.v[below]),
            -fluxes.ustar * fluxes.ustar,
            fluxes.ustar * fluxes.ustar);
    }

    double gamma_theta = 0.0;
    double gamma_u = 0.0;
    double gamma_v = 0.0;
    if (wstar > 0.0 && mixing_velocity > 0.0)
    {
        gamma_theta = std::max(
            std::min(countergradient_coefficient * theta_surface_flux
                         / (col.rho[0] * mixing_velocity),
                     3.0),
            0.0) / h;
        const double wind = std::max(
            std::hypot(col.u[0], col.v[0]), 0.1);
        const double momentum_factor = -15.9
            * fluxes.ustar * fluxes.ustar / wind
            * (wstar * wstar * wstar)
            / std::pow(mixing_velocity, 4);
        gamma_u = momentum_factor * col.u[0] / h;
        gamma_v = momentum_factor * col.v[0] / h;
    }

    for (std::size_t face = 1; face <= top_face && face < nz; ++face)
    {
        const double rho_face = 0.5 * (col.rho[face - 1] + col.rho[face]);
        const double eta = std::clamp(col.z_int[face] / h, 0.0, 1.0);
        const double entrainment_shape = eta * eta * eta;
        workspace.nonlocal_u_flux[face] = rho_face
            * (K_m[face] * gamma_u + entrainment_u * entrainment_shape);
        workspace.nonlocal_v_flux[face] = rho_face
            * (K_m[face] * gamma_v + entrainment_v * entrainment_shape);
        workspace.nonlocal_theta_flux[face] = rho_face
            * (K_h[face] * gamma_theta
               + entrainment_theta * entrainment_shape);
        workspace.nonlocal_qv_flux[face] = rho_face
            * entrainment_qv * entrainment_shape;
    }

    // Limit shared nonlocal vapor transfers by the donor's available mass.
    // The remaining diffusion and surface exchange form an M-matrix solve.
    std::vector<double> donor_fraction(nz, 1.0);
    for (std::size_t k = 0; k < nz; ++k)
    {
        const double outgoing = std::max(0.0, workspace.nonlocal_qv_flux[k + 1])
            + std::max(0.0, -workspace.nonlocal_qv_flux[k]);
        if (outgoing > 0.0)
            donor_fraction[k] = std::min(1.0, col.rho[k]
                * (col.z_int[k + 1] - col.z_int[k]) * col.qv[k]
                / (interval_s * outgoing));
    }
    for (std::size_t face = 1; face < nz; ++face)
    {
        double& flux = workspace.nonlocal_qv_flux[face];
        flux *= donor_fraction[flux >= 0.0 ? face - 1 : face];
    }

    boundary_layer::mix_column_surface_exchange(
        col, K_m, K_h, fluxes, interval_s, tend,
        &workspace.nonlocal_u_flux, &workspace.nonlocal_v_flux,
        &workspace.nonlocal_theta_flux, &workspace.nonlocal_qv_flux);
    tend.dqcdt_pbl.assign(nz, 0.0);
    tend.dqidt_pbl.assign(nz, 0.0);
}

void YSUScheme::apply_cloud_top_diffusion_tendencies(
    const BoundaryLayerColumnStateView& col,
    const std::vector<double>& cloud_diffusivity,
    double interval_s,
    BoundaryLayerColumnWorkspace& workspace,
    BoundaryLayerTendencies& tend) const
{
    const std::size_t nz = col.theta.size();
    std::vector<double> mixed_u(nz), mixed_v(nz), mixed_qv(nz);
    workspace.liquid_water_potential_temperature.resize(nz);
    for (std::size_t k = 0; k < nz; ++k)
    {
        const double exner = std::pow(
            col.p[k] / physical_constants::reference_pressure_pa,
            physical_constants::gas_constant_dry_air_jkgk
                / physical_constants::specific_heat_cp_jkgk);
        const double liquid_factor =
            physical_constants::latent_heat_vaporization_jkg
            / (physical_constants::specific_heat_cp_jkgk * exner);
        const double ice_factor =
            physical_constants::latent_heat_sublimation_jkg
            / (physical_constants::specific_heat_cp_jkgk * exner);
        mixed_u[k] = col.u[k] + interval_s * tend.dudt_pbl[k];
        mixed_v[k] = col.v[k] + interval_s * tend.dvdt_pbl[k];
        mixed_qv[k] = col.qv[k] + interval_s * tend.dqvdt_pbl[k];
        workspace.liquid_water_potential_temperature[k] = col.theta[k]
            + interval_s * tend.dthetadt_pbl[k] - liquid_factor * col.qc[k] - ice_factor * col.qi[k];
    }

    namespace vd = boundary_layer::vdiff;
    vd::diffuse_column_implicit(mixed_u, cloud_diffusivity, col.z_int, col.rho,
                       0.0, interval_s, workspace.cloud_top_u_tendency);
    vd::diffuse_column_implicit(mixed_v, cloud_diffusivity, col.z_int, col.rho,
                       0.0, interval_s, workspace.cloud_top_v_tendency);
    vd::diffuse_column_implicit(workspace.liquid_water_potential_temperature,
                       cloud_diffusivity, col.z_int, col.rho,
                       0.0, interval_s, workspace.cloud_top_theta_l_tendency);
    vd::diffuse_column_implicit(mixed_qv, cloud_diffusivity, col.z_int, col.rho,
                       0.0, interval_s, workspace.cloud_top_qv_tendency);
    vd::diffuse_column_implicit(col.qc, cloud_diffusivity, col.z_int, col.rho,
                       0.0, interval_s, tend.dqcdt_pbl);
    vd::diffuse_column_implicit(col.qi, cloud_diffusivity, col.z_int, col.rho,
                       0.0, interval_s, tend.dqidt_pbl);

    for (std::size_t k = 0; k < nz; ++k)
    {
        const double exner = std::pow(
            col.p[k] / physical_constants::reference_pressure_pa,
            physical_constants::gas_constant_dry_air_jkgk
                / physical_constants::specific_heat_cp_jkgk);
        const double liquid_factor =
            physical_constants::latent_heat_vaporization_jkg
            / (physical_constants::specific_heat_cp_jkgk * exner);
        const double ice_factor =
            physical_constants::latent_heat_sublimation_jkg
            / (physical_constants::specific_heat_cp_jkgk * exner);
        tend.dudt_pbl[k] += workspace.cloud_top_u_tendency[k];
        tend.dvdt_pbl[k] += workspace.cloud_top_v_tendency[k];
        tend.dqvdt_pbl[k] += workspace.cloud_top_qv_tendency[k];
        tend.dthetadt_pbl[k] += workspace.cloud_top_theta_l_tendency[k]
            + liquid_factor * tend.dqcdt_pbl[k]
            + ice_factor * tend.dqidt_pbl[k];
    }
}

} // namespace tmv
