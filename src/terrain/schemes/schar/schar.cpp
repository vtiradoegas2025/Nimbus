/**
 * @file schar.cpp
 * @brief Implementation for the terrain module.
 *
 * Provides executable logic for the terrain runtime path,
 * including initialization, stepping, and diagnostics helpers.
 * This file is part of the src/terrain subsystem.
 */

#include "schar.hpp"
#include "core/runtime/simulation.hpp"
#include <iostream>
#include <cmath>


/**
 * @brief Constructs the Schar terrain scheme.
 */
ScharScheme::ScharScheme() {}

/**
 * @brief Initializes the Schär terrain scheme.
 */
void ScharScheme::initialize(const TerrainConfig& cfg) 
{
    config_ = cfg;

    std::cout << "Initialized Schär terrain scheme" << std::endl;
    std::cout << "  H = " << cfg.schar.H << " m" << std::endl;
    std::cout << "  a = " << cfg.schar.a << " m" << std::endl;
    std::cout << "  ell = " << cfg.schar.ell << " m" << std::endl;
    std::cout << "  coordinate type = " << cfg.coord_id << std::endl;
}

/**
 * @brief Builds the topography.
 */
void ScharScheme::build_topography(const TerrainConfig& cfg, Topography2D& topo) 
{
    const int NR = topo.h.size();
    const int NTH = NR > 0 ? topo.h[0].size() : 0;

    const double center_i = (NR - 1) / 2.0;

    for (int i = 0; i < NR; ++i) 
    {
        for (int j = 0; j < NTH; ++j) 
        {
            double x, y;
            get_coordinates(i, j, x, y);

            double x_centered = x - center_i * dr;

            auto result = topography::eval_schar(x_centered, cfg.schar);

            topo.h[i][j] = result.h;

            if (cfg.compute_derivatives) 
            {
                topo.hx[i][j] = result.hx;
                topo.hy[i][j] = 0.0;
            }
        }
    }
}

/**
 * @brief Builds the metrics.
 */
void ScharScheme::build_metrics(const TerrainConfig& cfg,
                               const Topography2D& topo,
                               TerrainMetrics3D& metrics,
                               TerrainDiagnostics* diag_opt) {
    const int NR = metrics.z.size_r();
    const int NTH = metrics.z.size_th();
    const int NZ = metrics.z.size_z();

    auto zeta_levels = topography::build_zeta_levels(NZ, cfg.ztop);

    topography::TerrainFollowingCoordinate coord(cfg, topo);

    for (int i = 0; i < NR; ++i) 
    {
        for (int j = 0; j < NTH; ++j) 
        {
            for (int k = 0; k < NZ; ++k) 
            {
                double zeta = zeta_levels[k];
                double z, J, mx, my;

                coord.compute_metrics(zeta, i, j, z, J, mx, my);

                metrics.z(i, j, k) = z;
                metrics.J(i, j, k) = J;
                metrics.mx(i, j, k) = mx;
                metrics.my(i, j, k) = my;
                metrics.zeta(i, j, k) = zeta;
            }
        }
    }

    if (diag_opt) 
    {
        diag_opt->max_height = cfg.schar.H;
        topography::check_coordinate_validity(metrics, *diag_opt);
    }
}

/**
 * @brief Gets the coordinates.
 */
void ScharScheme::get_coordinates(int i, int j, double& x, double& y) const 
{
    x = i * dr;
    y = j * dtheta * 1000.0;
}
