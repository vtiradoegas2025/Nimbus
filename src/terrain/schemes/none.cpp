/**
 * @file none.cpp
 * @brief Implementation for the terrain module.
 *
 * Provides executable logic for the terrain runtime path,
 * including initialization, stepping, and diagnostics helpers.
 * This file is part of the src/terrain subsystem.
 */

#include "none.hpp"
#include "core/simulation.hpp"
#include <iostream>


NoneScheme::NoneScheme() {}


/**
 * @brief Initializes the none terrain scheme.
 */

void NoneScheme::initialize(const TerrainConfig& cfg)
{
    std::cout << "Initialized None terrain scheme (flat terrain)" << std::endl;
}


/**
 * @brief Builds the topography.
 */
void NoneScheme::build_topography(const TerrainConfig& cfg, Topography2D& topo)
{
    const int NR = topo.h.size();
    const int NTH = NR > 0 ? topo.h[0].size() : 0;

    for (int i = 0; i < NR; ++i)
    {
        for (int j = 0; j < NTH; ++j)
        {
            topo.h[i][j] = 0.0;
            if (!topo.hx.empty()) topo.hx[i][j] = 0.0;
            if (!topo.hy.empty()) topo.hy[i][j] = 0.0;
        }
    }
}

/**
 * @brief Builds the metrics for a flat (no-terrain) Cartesian grid.
 *
 * For "none" terrain the vertical levels MUST be the simulation's actual
 * uniform spacing z[k] = k * dz. Previously this scheme called
 * topography::build_zeta_levels(NZ, cfg.ztop), which stretches NZ levels
 * across cfg.ztop using z[k] = k * ztop / (NZ-1). With the default
 * ztop=20000 m and student.yaml's dz=500 m, NZ=32, that produced
 * z[k] = k * 645.16 m — a 29% mismatch with the simulation's intended
 * 500 m spacing. The dynamics' centered_dz_span queries the terrain z
 * directly, so it discretized ∂p/∂z with 1290 m denominators while the
 * IC built pressure on 1000 m denominators. The result was a constant
 * −2.2 m/s² spurious downward force at every interior cell that exploded
 * w within ~10 timesteps. See docs/Journey.md Phase 2 "Bug 4: Terrain
 * z mismatch with simulation dz".
 */
void NoneScheme::build_metrics(const TerrainConfig& cfg,
                              const Topography2D& topo,
                              TerrainMetrics3D& metrics,
                              TerrainDiagnostics* diag_opt)
{
    const int NR = metrics.z.size_r();
    const int NTH = metrics.z.size_th();
    const int NZ_metrics = metrics.z.size_z();

    // Use the simulation's uniform vertical spacing directly. cfg.ztop is
    // intentionally ignored for flat terrain — the user's grid (dz, NZ)
    // is the source of truth.
    const double sim_dz = ::dz;

    for (int i = 0; i < NR; ++i)
    {
        for (int j = 0; j < NTH; ++j)
        {
            for (int k = 0; k < NZ_metrics; ++k)
            {
                const double z_level = static_cast<double>(k) * sim_dz;
                metrics.z(i, j, k) = z_level;
                metrics.J(i, j, k) = 1.0;
                metrics.mx(i, j, k) = 0.0;
                metrics.my(i, j, k) = 0.0;
                metrics.zeta(i, j, k) = z_level;
            }
        }
    }

    if (diag_opt)
    {
        diag_opt->max_height = 0.0;
        diag_opt->max_slope_x = 0.0;
        diag_opt->max_slope_y = 0.0;
        diag_opt->min_jacobian = 1.0;
        diag_opt->max_jacobian = 1.0;
        diag_opt->coordinate_folding = false;
    }
}
