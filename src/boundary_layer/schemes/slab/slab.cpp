/**
 * @file slab.cpp
 * @brief Implementation for the boundary_layer module.
 *
 * Provides executable logic for the boundary_layer runtime path,
 * including initialization, stepping, and diagnostics helpers.
 * This file is part of the src/boundary_layer subsystem.
 */

#include "slab.hpp"
#include "boundary_layer/base/pbl_height.hpp"
#include "util/log.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace tmv
{

/**
 * @brief Initializes the slab boundary layer scheme.
 */

SlabScheme::SlabScheme()
    : min_h_(100.0), max_h_(2000.0)
      {}

/**
 * @brief Returns the required fields for the slab boundary layer scheme.
 */
int SlabScheme::required_fields() const 
{
    return static_cast<int>(BoundaryLayerRequirements::BASIC) |
           static_cast<int>(BoundaryLayerRequirements::MOISTURE);
}

/**
 * @brief Initializes the slab boundary layer scheme.
 */
void SlabScheme::initialize(const BoundaryLayerConfig& cfg) 
{
    if (!std::isfinite(cfg.pbl_max_height) || cfg.pbl_max_height <= 0)
        throw std::invalid_argument("SlabScheme requires a positive finite maximum height");
    max_h_ = cfg.pbl_max_height;
    min_h_ = std::min(100.0, max_h_);

    tmv::log_info("Initialized Slab Boundary Layer:");
    tmv::log_info("  Max PBL height: ", max_h_, " m");
}

/**
 * @brief Computes the column for the slab boundary layer scheme.
 */
void SlabScheme::compute_column(
    const BoundaryLayerConfig& cfg,
    const SurfaceConfig& sfc,
    const BoundaryLayerColumnStateView& col,
    BoundaryLayerTendencies& tend,
    BoundaryLayerDiagnostics* diag_opt
) 
{
    const size_t nz = col.theta->size();

    tend.dudt_pbl.assign(nz, 0.0);
    tend.dvdt_pbl.assign(nz, 0.0);
    tend.dthetadt_pbl.assign(nz, 0.0);
    tend.dqvdt_pbl.assign(nz, 0.0);
    tend.dqcdt_pbl.assign(nz, 0.0);
    tend.dqidt_pbl.assign(nz, 0.0);
    tend.dtkedt_pbl.assign(nz, 0.0);

    surface_fluxes::BulkFluxes fluxes;

    if (col.slab_state == nullptr)
    {
        throw std::invalid_argument(
            "SlabScheme::compute_column requires per-column SlabColumnState");
    }
    SlabColumnState& state = *col.slab_state;
    if (cfg.apply_surface_fluxes) 
    {
        fluxes = surface_fluxes::compute_surface_fluxes(
            cfg,
            sfc, col.u_sfc, col.v_sfc, col.theta_sfc, col.qv_sfc,
            (*col.theta)[0], (*col.qv)[0], (*col.p)[0], (*col.rho)[0],
            col.z_sfc, col.surface_slope_x, col.surface_slope_y
        );
    }

    // The slab state is a diagnostic view of the resolved column, not a second
    // prognostic reservoir. Surface sources are applied below exactly once via
    // the returned tendencies.
    update_slab_state(state, col, cfg);

    apply_slab_tendencies(state, col, fluxes, tend, cfg);

    if (diag_opt) 
    {
        diag_opt->pbl_height = state.h_m;
        diag_opt->ustar = fluxes.ustar;
        diag_opt->tau_u = fluxes.tau_u;
        diag_opt->tau_v = fluxes.tau_v;
        diag_opt->H = fluxes.H;
        diag_opt->E = fluxes.E;
        diag_opt->Cd = fluxes.Cd;
        diag_opt->Ch = fluxes.Ch;
        diag_opt->Ce = fluxes.Ce;

        if (diag_opt->include_diffusivity_profiles)
        {
            diag_opt->K_m.assign(nz + 1, 0.0);
            diag_opt->K_h.assign(nz + 1, 0.0);
            diag_opt->K_e.assign(nz + 1, 0.0);
        }
    }
}

/**
 * @brief Updates the slab state.
 */
void SlabScheme::update_slab_state(
    SlabColumnState& state,
    const BoundaryLayerColumnStateView& col,
    const BoundaryLayerConfig& cfg
)
{
    state.h_m = diagnose_pbl_height(col, cfg);
    state.h_m = std::clamp(state.h_m, min_h_, max_h_);

    double mass = 0.0;
    double theta_inventory = 0.0;
    double qv_inventory = 0.0;
    for (std::size_t k = 0; k < col.theta->size(); ++k)
    {
        const double overlap = std::max(0.0, std::min(state.h_m, (*col.z_int)[k + 1]) - (*col.z_int)[k]);
        if (overlap == 0.0)
            break;
        const double layer_mass = (*col.rho)[k] * overlap;
        mass += layer_mass;
        theta_inventory += layer_mass * (*col.theta)[k];
        qv_inventory += layer_mass * (*col.qv)[k];
    }
    if (!(mass > 0.0) || !std::isfinite(mass))
        throw std::invalid_argument("SlabScheme requires positive finite column mass");

    state.theta_m_k = theta_inventory / mass;
    state.qv_m_kgkg = qv_inventory / mass;
    state.initialized = true;
}

/**
 * @brief Diagnoses the PBL height.
 */
double SlabScheme::diagnose_pbl_height(
    const BoundaryLayerColumnStateView& col,
    const BoundaryLayerConfig& cfg
)
{
    if (cfg.pbl_top_method == "theta")
        return boundary_layer::pbl::diagnose_pbl_height_theta(
            *col.theta, *col.z_int,
            boundary_layer::pbl::theta_increase_threshold_k,
            min_h_, max_h_);
    // Shared bulk-Richardson method (virtual theta + bulk shear vs the lowest
    // level), so slab/YSU/MYNN agree. Uses the resolved column, not the relaxing
    // mixed-layer state.
    return boundary_layer::pbl::diagnose_pbl_height_ri(
        *col.theta, *col.qv, *col.u, *col.v, *col.z_int,
        /*Ri_crit=*/0.25, min_h_, max_h_);
}

/**
 * @brief Applies slab-mixed-layer tendencies to column prognostic fields.
 */
void SlabScheme::apply_slab_tendencies(
    const SlabColumnState& state,
    const BoundaryLayerColumnStateView& col,
    const surface_fluxes::BulkFluxes& fluxes,
    BoundaryLayerTendencies& tend,
    const BoundaryLayerConfig& cfg
)
{
    (void)cfg;
    const size_t nz = col.theta->size();
    const double surface_exner = std::pow(
        (*col.p)[0] / physical_constants::reference_pressure_pa,
        physical_constants::gas_constant_dry_air_jkgk /
            physical_constants::specific_heat_cp_jkgk);
    std::size_t active_levels = 0;
    double column_mass = 0.0;
    double u_inventory = 0.0;
    double v_inventory = 0.0;
    for (std::size_t k = 0; k < nz; ++k)
    {
        const double overlap = std::max(0.0, std::min(state.h_m, (*col.z_int)[k + 1]) - (*col.z_int)[k]);
        if (overlap == 0.0)
            break;
        const double dz = (*col.z_int)[k + 1] - (*col.z_int)[k];
        const double layer_mass = (*col.rho)[k] * overlap;
        if (!(dz > 0.0) || !(layer_mass > 0.0) || !std::isfinite(layer_mass))
            throw std::invalid_argument(
                "SlabScheme requires increasing interfaces and positive finite density");
        column_mass += layer_mass;
        u_inventory += layer_mass * (*col.u)[k];
        v_inventory += layer_mass * (*col.v)[k];
        ++active_levels;
    }
    if (active_levels == 0 || !(column_mass > 0.0))
        throw std::invalid_argument("SlabScheme resolved mixed layer is empty");

    const double u_mean = u_inventory / column_mass;
    const double v_mean = v_inventory / column_mass;
    constexpr double inverse_mixing_time_s = 1.0 / 300.0;

    // Earlier active cells are fully covered. Closing the final tendency against
    // its full cell mass incorporates partial overlap and preserves surface forcing.
    double accumulated_u = 0.0;
    double accumulated_v = 0.0;
    double accumulated_theta = 0.0;
    double accumulated_qv = 0.0;
    for (std::size_t k = 0; k < active_levels; ++k)
    {
        const double layer_mass =
            (*col.rho)[k] * ((*col.z_int)[k + 1] - (*col.z_int)[k]);
        if (k + 1 == active_levels)
        {
            tend.dudt_pbl[k] = (fluxes.tau_u - accumulated_u) / layer_mass;
            tend.dvdt_pbl[k] = (fluxes.tau_v - accumulated_v) / layer_mass;
            tend.dthetadt_pbl[k] =
                (fluxes.H /
                     (boundary_layer_constants::cp * surface_exner) -
                 accumulated_theta) /
                layer_mass;
            tend.dqvdt_pbl[k] = (fluxes.E - accumulated_qv) / layer_mass;
            continue;
        }

        tend.dudt_pbl[k] =
            inverse_mixing_time_s * (u_mean - (*col.u)[k]) +
            fluxes.tau_u / column_mass;
        tend.dvdt_pbl[k] =
            inverse_mixing_time_s * (v_mean - (*col.v)[k]) +
            fluxes.tau_v / column_mass;
        tend.dthetadt_pbl[k] =
            inverse_mixing_time_s * (state.theta_m_k - (*col.theta)[k]) +
            (fluxes.H /
             (boundary_layer_constants::cp * surface_exner)) / column_mass;
        tend.dqvdt_pbl[k] =
            inverse_mixing_time_s * (state.qv_m_kgkg - (*col.qv)[k]) +
            fluxes.E / column_mass;

        accumulated_u += layer_mass * tend.dudt_pbl[k];
        accumulated_v += layer_mass * tend.dvdt_pbl[k];
        accumulated_theta += layer_mass * tend.dthetadt_pbl[k];
        accumulated_qv += layer_mass * tend.dqvdt_pbl[k];
    }
}

} // namespace tmv
