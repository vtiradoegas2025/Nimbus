/**
 * @file bell.cpp
 * @brief Implementation for the terrain module.
 *
 * Provides executable logic for the terrain runtime path,
 * including initialization, stepping, and diagnostics helpers.
 * This file is part of the src/terrain subsystem.
 */

#include "bell.hpp"
#include "core/simulation.hpp"
#include <iostream>
#include <cmath>


BellScheme::BellScheme() 
{
}

/**
 * @brief Initializes the bell terrain scheme.
 */
void BellScheme::initialize(const TerrainConfig& cfg) 
{
    config_ = cfg;

    std::cout << "Initialized Bell terrain scheme" << std::endl;
    std::cout << "  h0 = " << cfg.bell.h0 << " m" << std::endl;
    std::cout << "  a = " << cfg.bell.a << " m" << std::endl;
    std::cout << "  axisymmetric = " << (cfg.bell.axisymmetric ? "true" : "false") << std::endl;
    std::cout << "  coordinate type = " << cfg.coord_id << std::endl;
}

/**
 * @brief Builds the topography.
 */
void BellScheme::build_topography(const TerrainConfig& cfg, Topography2D& topo) 
{
    const int NR = topo.h.size();
    const int NTH = NR > 0 ? topo.h[0].size() : 0;

    const double center_i = (NR - 1) / 2.0;
    const double center_j = (NTH - 1) / 2.0;


    for (int i = 0; i < NR; ++i) 
    {
        for (int j = 0; j < NTH; ++j) 
        {
            double x, y;
            get_coordinates(i, j, x, y);

            double x_centered = x - center_i * dr;
            double y_centered = y - center_j * dtheta * 1000.0;

            auto result = topography::eval_bell(x_centered, y_centered, cfg.bell);

            topo.h[i][j] = result.h;

            if (cfg.compute_derivatives) 
            {
                topo.hx[i][j] = result.hx;
                topo.hy[i][j] = result.hy;
            }
        }
    }
}

/**
 * @brief Builds the metrics.
 */
void BellScheme::build_metrics(const TerrainConfig& cfg,
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
        diag_opt->max_height = cfg.bell.h0;
        topography::check_coordinate_validity(metrics, *diag_opt);
    }
}

/**
 * @brief Gets the coordinates.
 */
void BellScheme::get_coordinates(int i, int j, double& x, double& y) const 
{
    x = i * dr;
    y = j * dtheta * 1000.0;
}
