/**
 * @file vortex_seed.cpp
 * @brief VortexSeedTrigger implementation. Adds a Rankine vortex to the
 *        wind field. Used by the tornado / tornado_cgrid schemes for
 *        spin-up — the axisymmetric tornado dynamics has no realistic
 *        non-zero solution without an initial vorticity source.
 */

#include "init/trigger/vortex_seed.hpp"

#include "core/coordinate_system.hpp"
#include "core/field3d.hpp"
#include "core/grid_geometry.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>

extern int NR;
extern int NTH;
extern int NZ;

extern Field3D u;
extern Field3D v;

extern GridGeometry global_grid_geometry;
extern CoordinateSystem global_coordinate_system;

namespace tmv::init
{

namespace
{

/// Rankine combined-profile tangential velocity.
double rankine_vtheta(double r_m, double r_max_m, double v_max_ms)
{
    if (r_m <= 0.0)
    {
        return 0.0;
    }
    if (r_m <= r_max_m)
    {
        return v_max_ms * (r_m / r_max_m);
    }
    return v_max_ms * (r_max_m / r_m);
}

}  // namespace

VortexSeedTrigger::VortexSeedTrigger(VortexSeedParams params)
    : params_(params)
{
}

void VortexSeedTrigger::apply() const
{
    const double r_max = std::max(1.0, params_.r_max_m);
    const double v_max = params_.v_max_ms;
    const double z_top = params_.z_top_m;
    const auto& geo = global_grid_geometry;

    if (global_coordinate_system == CoordinateSystem::Cartesian)
    {
        // Cartesian: vortex axis at (center_x, center_y). Project the
        // tangential v_theta(r) onto (u_x, u_y) at every grid cell using
        // the local azimuth from the axis. Additive so the seed composes
        // with any environmental hodograph already in the field.
        const double cx = params_.center_x_m;
        const double cy = params_.center_y_m;

        #pragma omp parallel for collapse(2)
        for (int i = 0; i < NR; ++i)
        {
            for (int j = 0; j < NTH; ++j)
            {
                const double dx = geo.r[i] - cx;
                const double dy = geo.theta[j] - cy;
                const double r = std::sqrt(dx * dx + dy * dy);
                if (r <= 0.0)
                {
                    continue;
                }
                const double inv_r = 1.0 / r;

                for (int k = 0; k < NZ; ++k)
                {
                    if (geo.z[k] > z_top)
                    {
                        break;
                    }
                    const double v_theta = rankine_vtheta(r, r_max, v_max);
                    // (u, v) Cartesian projection of an azimuthal vector:
                    //   u_x = -v_theta * dy / r
                    //   u_y = +v_theta * dx / r
                    u[i][j][k] += static_cast<float>(-v_theta * dy * inv_r);
                    v[i][j][k] += static_cast<float>( v_theta * dx * inv_r);
                }
            }
        }
        return;
    }

    // Cylindrical (collocated and C-grid): vortex axis is at r = 0. The
    // tangential perturbation is the azimuthal component v at every
    // grid point. For collocated grids v is at cell-center theta; for
    // C-grid, v is at the theta-face theta_{j+1/2}. The Rankine profile
    // depends only on r so the placement difference is absorbed by the
    // discrete scheme — we just write to v[i][j][k]. (For c-grid the
    // r used is geo.r[i] which is the cell-center radius; the v lives
    // at (r_center[i], theta_face[j+1/2], z_face[k]) but we approximate
    // by using cell-center r since the vortex is axisymmetric.)
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < NR; ++i)
    {
        for (int j = 0; j < NTH; ++j)
        {
            const double r = geo.r[i];
            for (int k = 0; k < NZ; ++k)
            {
                if (geo.z[k] > z_top)
                {
                    break;
                }
                const double v_theta = rankine_vtheta(r, r_max, v_max);
                v[i][j][k] += static_cast<float>(v_theta);
            }
        }
    }
}

std::string VortexSeedTrigger::describe() const
{
    return "vortex_seed/rankine";
}

}  // namespace tmv::init
