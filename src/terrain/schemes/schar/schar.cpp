// Analytic Schar topography and terrain-following metrics.

#include "schar.hpp"
#include "core/infra/grid_coordinates.hpp"
#include "core/runtime/grid_state.hpp"
#include "util/log.hpp"

#include <cmath>

namespace tmv
{

ScharScheme::ScharScheme() {}

void ScharScheme::initialize(const TerrainConfig& cfg) 
{
    config_ = cfg;

    tmv::log_info("Initialized Schär terrain scheme");
    tmv::log_info("  H = ", cfg.schar.H, " m");
    tmv::log_info("  a = ", cfg.schar.a, " m");
    tmv::log_info("  ell = ", cfg.schar.ell, " m");
    tmv::log_info("  coordinate type = ", cfg.coord_id);
}

void ScharScheme::build_topography(const TerrainConfig& cfg, Topography2D& topo) 
{
    const int NR = topo.h.size();
    const int NTH = NR > 0 ? topo.h[0].size() : 0;

    const double center_x = 0.5 * static_cast<double>(NR) * dr;

    for (int i = 0; i < NR; ++i) 
    {
        for (int j = 0; j < NTH; ++j) 
        {
            double x, y;
            get_coordinates(i, j, x, y);

            double x_centered = x - center_x;

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

    topography::build_conservative_face_metrics(
        cfg, topo, topography::build_zeta_interfaces(NZ, cfg.ztop),
        dr, dtheta, metrics);

    if (diag_opt) 
    {
        diag_opt->max_height = cfg.schar.H;
        topography::check_coordinate_validity(metrics, *diag_opt);
    }
}

void ScharScheme::get_coordinates(int i, int j, double& x, double& y) const 
{
    x = grid_coordinates::scalar_center(i, dr);
    y = grid_coordinates::scalar_center(j, dtheta);
}

}  // namespace tmv
