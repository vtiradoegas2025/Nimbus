/**
 * @file conservation_budget.cpp
 * @brief Domain-integrated mass/water/energy budget diagnostics.
 *
 * Extracted from src/core/dynamics.cpp to give conservation diagnostics
 * their own compilation unit. The dynamics orchestrator calls these after
 * each physics stage to detect and log conservation drift.
 */

#include "diagnostics/conservation_budget.hpp"
#include "core/coordinate_system.hpp"
#include "core/runtime_config.hpp"
#include "core/simulation.hpp"
#include "util/log.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace
{

std::vector<double> cell_volume_weights;
int cached_nr = -1;
double cached_dr = std::numeric_limits<double>::quiet_NaN();
double cached_dtheta = std::numeric_limits<double>::quiet_NaN();
double cached_dz = std::numeric_limits<double>::quiet_NaN();
CoordinateSystem cached_coord = CoordinateSystem::Cylindrical;

/**
 * @brief Recomputes cell-volume weights for conservation diagnostics.
 *
 * Cylindrical: volume of annular cell = r_center * dr * dtheta * dz
 *              (varies with radial index i).
 * Cartesian:   volume = dx * dy * dz (uniform for all cells).
 *              In Cartesian mode dr stores dx and dtheta stores dy.
 */
void ensure_cell_volume_weights()
{
    if (cached_nr == NR &&
        cached_dr == dr &&
        cached_dtheta == dtheta &&
        cached_dz == dz &&
        cached_coord == global_coordinate_system &&
        cell_volume_weights.size() == static_cast<std::size_t>(NR))
    {
        return;
    }

    cached_nr = NR;
    cached_dr = dr;
    cached_dtheta = dtheta;
    cached_dz = dz;
    cached_coord = global_coordinate_system;
    cell_volume_weights.assign(static_cast<std::size_t>(NR), 0.0);

    if (global_coordinate_system == CoordinateSystem::Cartesian)
    {
        // dx * dy * dz — uniform for every cell.
        const double cart_vol = dr * dtheta * dz;
        for (int i = 0; i < NR; ++i)
        {
            cell_volume_weights[static_cast<std::size_t>(i)] = cart_vol;
        }
    }
    else
    {
        // Cylindrical annular cell: r_center * dr * dtheta * dz.
        for (int i = 0; i < NR; ++i)
        {
            const double r_center = std::max((static_cast<double>(i) + 0.5) * dr, 0.5 * dr);
            cell_volume_weights[static_cast<std::size_t>(i)] = r_center * dr * dtheta * dz;
        }
    }
}

}  // namespace

ConservationBudget compute_conservation_budget()
{
    ConservationBudget budget{};
    if (rho.empty() || p.empty() || theta.empty() || qv.empty() ||
        qc.empty() || qr.empty() || qi.empty() || qs.empty() || qg.empty() || qh.empty())
    {
        return budget;
    }

    ensure_cell_volume_weights();
    constexpr double kappa = R_d / cp;
    double dry_mass = 0.0;
    double total_water = 0.0;
    double thermal_energy = 0.0;

    #pragma omp parallel for reduction(+:dry_mass,total_water,thermal_energy) collapse(2)
    for (int i = 0; i < NR; ++i)
    {
        for (int j = 0; j < NTH; ++j)
        {
            const double cell_volume = cell_volume_weights[static_cast<std::size_t>(i)];
            for (int k = 0; k < NZ; ++k)
            {
                const double rho_val = std::max(0.0, static_cast<double>(rho[i][j][k]));
                const double qv_val = std::max(0.0, static_cast<double>(qv[i][j][k]));
                const double qc_val = std::max(0.0, static_cast<double>(qc[i][j][k]));
                const double qr_val = std::max(0.0, static_cast<double>(qr[i][j][k]));
                const double qi_val = std::max(0.0, static_cast<double>(qi[i][j][k]));
                const double qs_val = std::max(0.0, static_cast<double>(qs[i][j][k]));
                const double qg_val = std::max(0.0, static_cast<double>(qg[i][j][k]));
                const double qh_val = std::max(0.0, static_cast<double>(qh[i][j][k]));
                const double water = qv_val + qc_val + qr_val + qi_val + qs_val + qg_val + qh_val;
                const double dry_fraction = std::max(0.0, 1.0 - water);

                dry_mass += rho_val * dry_fraction * cell_volume;
                total_water += rho_val * water * cell_volume;

                const double p_val = static_cast<double>(p[i][j][k]);
                const double theta_val = static_cast<double>(theta[i][j][k]);
                if (std::isfinite(p_val) && std::isfinite(theta_val) && p_val > 0.0)
                {
                    const double temperature = theta_val * std::pow(p_val / p0, kappa);
                    if (std::isfinite(temperature))
                    {
                        thermal_energy += rho_val * cp * temperature * cell_volume;
                    }
                }
            }
        }
    }

    budget.dry_mass = dry_mass;
    budget.total_water = total_water;
    budget.thermal_energy = thermal_energy;
    return budget;
}

void report_budget_transition(const char* stage, const ConservationBudget& before, const ConservationBudget& after, double dt_stage)
{
    if (!std::isfinite(dt_stage) || dt_stage <= 0.0)
    {
        return;
    }

    const double d_dry_mass = after.dry_mass - before.dry_mass;
    const double d_total_water = after.total_water - before.total_water;
    const double d_thermal_energy = after.thermal_energy - before.thermal_energy;

    const double rel_dry = std::abs(d_dry_mass) / std::max(1.0, std::abs(before.dry_mass));
    const double rel_water = std::abs(d_total_water) / std::max(1.0, std::abs(before.total_water));
    const double rel_energy = std::abs(d_thermal_energy) / std::max(1.0, std::abs(before.thermal_energy));

    const bool warn = rel_dry > 5.0e-4 || rel_water > 5.0e-4 || rel_energy > 2.0e-3;

    if (!warn && !log_debug_enabled())
    {
        return;
    }

    if (warn)
    {
        tmv::log_warn("[PHYSICS BUDGET WARN] stage=", stage,
                      " d_dry_mass=", d_dry_mass,
                      " d_total_water=", d_total_water,
                      " d_thermal_energy=", d_thermal_energy,
                      " dry_tendency=", (d_dry_mass / dt_stage),
                      " water_tendency=", (d_total_water / dt_stage),
                      " energy_tendency=", (d_thermal_energy / dt_stage));
    }
    else
    {
        tmv::log_debug("[PHYSICS BUDGET] stage=", stage,
                       " d_dry_mass=", d_dry_mass,
                       " d_total_water=", d_total_water,
                       " d_thermal_energy=", d_thermal_energy,
                       " dry_tendency=", (d_dry_mass / dt_stage),
                       " water_tendency=", (d_total_water / dt_stage),
                       " energy_tendency=", (d_thermal_energy / dt_stage));
    }
}
