#include "boundary_layer/base/implicit_column_exchange.hpp"
/**
 * @file mynn.cpp
 * @brief Implementation for the boundary_layer module.
 *
 * Provides executable logic for the boundary_layer runtime path,
 * including initialization, stepping, and diagnostics helpers.
 * This file is part of the src/boundary_layer subsystem.
 */

#include "mynn.hpp"
#include "mynn_stability.hpp"
#include "mynn_column_geometry.hpp"
#include "boundary_layer/base/vertical_diffusion.hpp"
#include "boundary_layer/base/pbl_height.hpp"
#include "util/log.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace tmv
{
namespace
{

std::size_t validate_column(
    const BoundaryLayerConfig& cfg,
    const BoundaryLayerColumnStateView& col)
{
    if (!std::isfinite(cfg.dt_pbl) || cfg.dt_pbl <= 0.0
        || !std::isfinite(cfg.pbl_max_height) || cfg.pbl_max_height <= 0.0)
        throw std::invalid_argument(
            "MYNN requires positive finite cadence and maximum PBL height");

    if (!col.theta)
        throw std::invalid_argument("MYNN requires a non-empty theta profile");
    const std::size_t nz = col.theta.size();
    if (nz == 0 || !col.u || !col.v || !col.qv || !col.p || !col.rho
        || !col.z_int || col.u.size() != nz || col.v.size() != nz
        || col.qv.size() != nz || col.p.size() != nz
        || col.rho.size() != nz || col.z_int.size() != nz + 1
        || (col.tke && col.tke.size() != nz))
        throw std::invalid_argument(
            "MYNN requires consistently shaped column profiles");

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
            || (k > 0 && col.p[k] > col.p[k - 1])
            || (col.tke
                && (!std::isfinite(col.tke[k]) || col.tke[k] < 0.0)))
            throw std::invalid_argument(
                "MYNN column contains non-finite or unphysical state");
    }
    return nz;
}

} // namespace

/**
 * @brief Initializes the MYNN boundary layer scheme.
 */
MYNNScheme::MYNNScheme() = default;

/**
 * @brief Returns the required fields for the MYNN boundary layer scheme.
 */
int MYNNScheme::required_fields() const 
{
    return static_cast<int>(BoundaryLayerRequirements::BASIC) |
           static_cast<int>(BoundaryLayerRequirements::MOISTURE) |
           static_cast<int>(BoundaryLayerRequirements::TKE);
}

/**
 * @brief Initializes the MYNN boundary layer scheme Takes in the boundary layer configuration.
 */
void MYNNScheme::initialize(const BoundaryLayerConfig& cfg) 
{
    if (!std::isfinite(cfg.pbl_max_height) || cfg.pbl_max_height <= 0.0)
        throw std::invalid_argument(
            "MYNN requires a positive finite maximum PBL height");
    lmax_ = cfg.pbl_max_height * 0.5;

    tmv::log_info("Initialized MYNN Boundary Layer:");
    tmv::log_info("  Prognostic TKE: enabled");
    tmv::log_info("  Max mixing length: ", lmax_, " m");
    tmv::log_info("  Stability closure: Nakanishi-Niino level 2.5");
}

/**
 * @brief Computes the column for the MYNN boundary layer scheme.
 */
void MYNNScheme::compute_column(
    const BoundaryLayerConfig& cfg,
    const SurfaceConfig& sfc,
    const BoundaryLayerColumnStateView& col,
    BoundaryLayerTendencies& tend,
    BoundaryLayerDiagnostics* diag_opt
) 
{
    const size_t nz = validate_column(cfg, col);

    BoundaryLayerTendencies candidate;
    BoundaryLayerDiagnostics candidate_diagnostics;
    if (diag_opt)
        candidate_diagnostics.include_diffusivity_profiles =
            diag_opt->include_diffusivity_profiles;

    BoundaryLayerColumnWorkspace local_workspace;
    BoundaryLayerColumnWorkspace& workspace = col.workspace
        ? *col.workspace
        : local_workspace;
    workspace.tke.assign(nz, 0.1);
    std::vector<double>& tke_col = workspace.tke;
    if (col.tke && col.tke->size() == nz)
    {
        for (size_t k = 0; k < nz; ++k)
        {
            const double tke_k = (*col.tke)[k];
            tke_col[k] = tke_k;
        }
    }

    candidate.dtkedt_pbl.assign(nz, 0.0);
    candidate.dqcdt_pbl.assign(nz, 0.0);
    candidate.dqidt_pbl.assign(nz, 0.0);

    surface_fluxes::BulkFluxes fluxes;

    if (cfg.apply_surface_fluxes) 
    {
        fluxes = surface_fluxes::compute_surface_fluxes(
            cfg,
            sfc, col.u_sfc, col.v_sfc, col.theta_sfc, col.qv_sfc,
            (*col.theta)[0], (*col.qv)[0], (*col.p)[0], (*col.rho)[0],
            col.z_sfc, col.surface_slope_x, col.surface_slope_y
        );
    }

    double h = diagnose_pbl_height(col, tke_col, cfg);

    workspace.mixing_length.resize(nz);
    std::vector<double>& l_mix = workspace.mixing_length;
    compute_mixing_length(col, h, l_mix);

    workspace.K_m.assign(nz + 1, 0.0);
    workspace.K_h.assign(nz + 1, 0.0);
    workspace.K_e.assign(nz + 1, 0.0);
    std::vector<double>& K_m = workspace.K_m;
    std::vector<double>& K_h = workspace.K_h;
    std::vector<double>& K_e = workspace.K_e;
    compute_eddy_diffusivities(col, tke_col, l_mix, K_m, K_h, K_e);

    update_tke(col, tke_col, l_mix, fluxes, K_m, K_h, K_e,
               cfg.dt_pbl, candidate.dtkedt_pbl);

    apply_diffusion_tendencies(cfg.dt_pbl, col, K_m, K_h, fluxes, candidate);

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
            candidate_diagnostics.K_m = K_m;
            candidate_diagnostics.K_h = K_h;
            candidate_diagnostics.K_e = K_e;
        }
    }

    tend = std::move(candidate);
    if (diag_opt)
        *diag_opt = std::move(candidate_diagnostics);
}

/**
 * @brief Diagnoses the PBL height.
 */
double MYNNScheme::diagnose_pbl_height(
    const BoundaryLayerColumnStateView& col,
    const std::vector<double>& tke_col,
    const BoundaryLayerConfig& cfg
) 
{
    if (cfg.pbl_top_method == "theta")
        return boundary_layer::pbl::diagnose_pbl_height_theta(
            *col.theta, *col.z_int,
            boundary_layer::pbl::theta_increase_threshold_k,
            100.0, cfg.pbl_max_height);
    if (cfg.pbl_top_method == "tke")
        return boundary_layer::pbl::diagnose_pbl_height_tke(
            tke_col, *col.z_int,
            boundary_layer::pbl::tke_diagnostic_threshold_m2s2(tke_col),
            100.0, cfg.pbl_max_height);
    // Shared bulk-Richardson method (virtual theta + bulk shear vs the lowest
    // level), consistent with slab/YSU and honoring the default pbl_top_method=ri
    // (the previous TKE-threshold ignored it).
    return boundary_layer::pbl::diagnose_pbl_height_ri(
        *col.theta, *col.qv, *col.u, *col.v, *col.z_int,
        /*Ri_crit=*/0.25, /*min_h=*/100.0, cfg.pbl_max_height);
}

/**
 * @brief Computes the mixing length.
 */
void MYNNScheme::compute_mixing_length(
    const BoundaryLayerColumnStateView& col,
    double h,
    std::vector<double>& l_mix
) 
{
    const size_t nz = col.theta->size();
    const double kappa = boundary_layer_constants::kappa;
    const double h_safe = std::max(h, 1.0);

    for (size_t k = 0; k < nz; ++k) 
    {
        double z_level = 0.5 * ((*col.z_int)[k] + (*col.z_int)[k + 1]);

        double l_blackadar = kappa * z_level / (1.0 + kappa * z_level / lmax_);

        double l_wall = kappa * z_level;
        double l_top = kappa * std::max(h_safe - z_level, 0.0);

        l_mix[k] = std::min({l_blackadar, l_wall, l_top, lmax_});
        if (!std::isfinite(l_mix[k]))
        {
            l_mix[k] = 1.0;
        }
        l_mix[k] = std::max(l_mix[k], 1.0);
    }
}

/**
 * @brief Computes the eddy diffusivities.
 */
void MYNNScheme::compute_eddy_diffusivities(
    const BoundaryLayerColumnStateView& col,
    const std::vector<double>& tke_col,
    const std::vector<double>& l_mix,
    std::vector<double>& K_m,
    std::vector<double>& K_h,
    std::vector<double>& K_e
) {
    const size_t nz = col.theta->size();

    for (size_t face = 1; face < nz; ++face)
    {
        const size_t lower = face - 1;
        const size_t upper = face;
        const auto gradients = mynn::interface_gradients(col, face);
        const double lower_weight = gradients.lower_weight;
        const double upper_weight = gradients.upper_weight;
        const double q_squared = 2.0 * (
            lower_weight * std::max(tke_col[lower], 1.0e-8)
            + upper_weight * std::max(tke_col[upper], 1.0e-8));
        const double length = std::max(
            lower_weight * l_mix[lower] + upper_weight * l_mix[upper], 1.0);
        const auto stability = mynn::level25_stability_functions(
            {gradients.shear_s2, gradients.buoyancy_s2, length, q_squared});
        const double length_times_q = length * std::sqrt(q_squared);
        K_m[face] = std::clamp(
            length_times_q * stability.sm, 0.0, 1000.0);
        K_h[face] = std::clamp(
            length_times_q * stability.sh, 0.0, 1000.0);
        K_e[face] = K_h[face];
    }

    K_m[0] = 0.0;
    K_h[0] = 0.0;
    K_e[0] = 0.0;
    K_m[nz] = 0.0;
    K_h[nz] = 0.0;
    K_e[nz] = 0.0;
}

/**
 * @brief Updates the TKE field.
 */
void MYNNScheme::update_tke(
    const BoundaryLayerColumnStateView& col,
    const std::vector<double>& tke_col,
    const std::vector<double>& l_mix,
    const surface_fluxes::BulkFluxes& fluxes,
    const std::vector<double>& K_m,
    const std::vector<double>& K_h,
    const std::vector<double>& K_e,
    double dt,
    std::vector<double>& tke_tend
) 
{
    const std::size_t nz = tke_col.size();
    std::vector<double> mass(nz), conductance(nz + 1, 0.0), source(nz),
        sink_rate(nz), dissipation_rate(nz), energy(nz), upper(nz), candidate(nz);
    double lower_production = 0.0;
    double maximum_energy = 0.0;
    for (std::size_t k = 0; k < nz; ++k)
    {
        mass[k] = col.rho[k] * (col.z_int[k + 1] - col.z_int[k]);
        if (k > 0)
            conductance[k] = 0.5 * (col.rho[k - 1] + col.rho[k]) * K_e[k]
                / (0.5 * (col.z_int[k + 1] - col.z_int[k - 1]));
        double upper_production = 0.0;
        if (k + 1 < nz)
        {
            const auto gradients = mynn::interface_gradients(col, k + 1);
            upper_production = K_m[k + 1] * gradients.shear_s2
                             + K_h[k + 1] * gradients.buoyancy_s2;
        }
        double production = 0.5 * (lower_production + upper_production);
        lower_production = upper_production;
        if (k == 0)
            production += std::pow(std::max(fluxes.ustar, 0.0), 3)
                / (boundary_layer_constants::kappa * std::max(col.z_sfc, 1.0));
        if (!std::isfinite(production))
            throw std::runtime_error("MYNN TKE production is not finite");
        source[k] = std::max(production, 0.0);
        sink_rate[k] = std::max(-production, 0.0) / std::max(tke_col[k], 1.0e-8);
        dissipation_rate[k] = std::sqrt(8.0) / (b1_ * std::max(l_mix[k], 1.0));
        maximum_energy = std::max(maximum_energy, tke_col[k] + dt * source[k]);
    }
    // A constant upper solution and the M-matrix Jacobian give monotone
    // Newton convergence to nonnegative TKE for the implicit diffusion/sinks.
    // Solve absolute iterates with a nonnegative RHS to avoid cancellation at zero.
    std::fill(energy.begin(), energy.end(), maximum_energy);
    bool converged = false;
    for (int iteration = 0; iteration < 64; ++iteration)
    {
        for (std::size_t k = 0; k < nz; ++k)
        {
            const double root = std::sqrt(energy[k]);
            const double lower = -dt * conductance[k];
            const double above = -dt * conductance[k + 1];
            const double diagonal = mass[k] * (1.0 + dt * (sink_rate[k]
                + 1.5 * dissipation_rate[k] * root)) - lower - above;
            const double pivot = diagonal - (k > 0 ? lower * upper[k - 1] : 0.0);
            const double rhs = mass[k] * (tke_col[k] + dt * source[k]
                + 0.5 * dt * dissipation_rate[k] * energy[k] * root)
                - (k > 0 ? lower * candidate[k - 1] : 0.0);
            if (!std::isfinite(pivot) || pivot <= 0.0 || !std::isfinite(rhs))
                throw std::runtime_error("MYNN implicit TKE solve is not representable");
            upper[k] = above / pivot;
            candidate[k] = rhs / pivot;
        }
        for (std::size_t k = nz - 1; k > 0; --k)
            candidate[k - 1] -= upper[k - 1] * candidate[k];
        double change = 0.0;
        for (std::size_t k = 0; k < nz; ++k)
        {
            if (!std::isfinite(candidate[k]) || candidate[k] < 0.0)
                throw std::runtime_error("MYNN implicit TKE solve produced an invalid iterate");
            change = std::max(change, std::abs(candidate[k] - energy[k]));
            energy[k] = candidate[k];
        }
        if (change <= 1.0e-12 * std::max(maximum_energy, 1.0e-8))
        {
            converged = true;
            break;
        }
    }
    if (!converged)
        throw std::runtime_error("MYNN implicit TKE solve did not converge");
    tke_tend.resize(nz);
    double lower_flux = 0.0;
    for (std::size_t k = 0; k < nz; ++k)
    {
        const double upper_flux = k + 1 < nz
            ? conductance[k + 1] * (energy[k] - energy[k + 1]) : 0.0;
        tke_tend[k] = (lower_flux - upper_flux) / mass[k]
            + source[k] - sink_rate[k] * energy[k]
            - dissipation_rate[k] * energy[k] * std::sqrt(energy[k]);
        lower_flux = upper_flux;
    }
}

/**
 * @brief Applies the diffusion tendencies.
 */
void MYNNScheme::apply_diffusion_tendencies(
    double interval_s, const BoundaryLayerColumnStateView& col,
    const std::vector<double>& K_m,
    const std::vector<double>& K_h,
    surface_fluxes::BulkFluxes& fluxes,
    BoundaryLayerTendencies& tend
)
{
    boundary_layer::mix_column_surface_exchange(
        col, K_m, K_h, fluxes, interval_s, tend);

}

} // namespace tmv
