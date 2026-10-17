// Analytic bell topography and terrain-following metrics.

#include "bell.hpp"
#include "core/infra/grid_coordinates.hpp"
#include "core/runtime/grid_state.hpp"
#include "util/log.hpp"

#include <cmath>

namespace tmv
{

BellScheme::BellScheme() 
{
}

void BellScheme::initialize(const TerrainConfig& cfg) 
{
    config_ = cfg;

    tmv::log_info("Initialized Bell terrain scheme");
    tmv::log_info("  h0 = ", cfg.bell.h0, " m");
    tmv::log_info("  a = ", cfg.bell.a, " m");
    tmv::log_info("  axisymmetric = ", cfg.bell.axisymmetric ? "true" : "false");
    tmv::log_info("  coordinate type = ", cfg.coord_id);
}

void BellScheme::build_topography(const TerrainConfig& cfg, Topography2D& topo) 
{
    const int NR = topo.h.size();
    const int NTH = NR > 0 ? topo.h[0].size() : 0;

    const double center_x = 0.5 * static_cast<double>(NR) * dr;
    const double center_y = 0.5 * static_cast<double>(NTH) * dtheta;


    for (int i = 0; i < NR; ++i) 
    {
        for (int j = 0; j < NTH; ++j) 
        {
            double x, y;
            get_coordinates(i, j, x, y);

            double x_centered = x - center_x;
            double y_centered = y - center_y;

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

    topography::build_conservative_face_metrics(
        cfg, topo, topography::build_zeta_interfaces(NZ, cfg.ztop),
        dr, dtheta, metrics);

    if (diag_opt) 
    {
        diag_opt->max_height = cfg.bell.h0;
        topography::check_coordinate_validity(metrics, *diag_opt);
    }
}

void BellScheme::get_coordinates(int i, int j, double& x, double& y) const 
{
    x = grid_coordinates::scalar_center(i, dr);
    y = grid_coordinates::scalar_center(j, dtheta);
}

}  // namespace tmv
