/**
 * @file initial_conditions_cylindrical_cgrid.cpp
 * @brief Cylindrical C-grid wind initialization (Phase C.3).
 *
 * Implements `apply_cylindrical_cgrid_wind_initialization()` declared in
 * `core/initial_conditions.hpp`. The function projects the Cartesian
 * hodograph (u_x(z), u_y(z)) onto the staggered cylindrical face positions:
 *
 *   u (radial)    at r-face       (r_face[i], theta[j],          z[k])
 *   v (azimuthal) at theta-face   (r[i],      theta_{j+1/2},     z[k])
 *   w (vertical)  at z-face       always zero from the hodograph
 *
 * The structural difference vs the collocated cylindrical wind init is
 * confined to v: its trig argument is the half-cell-shifted azimuth
 * theta_{j+1/2} = (j + 0.5) * dtheta, not the cell-center theta[j].
 *
 * The collocated wind init lives inline in `equations.cpp` as a static
 * helper. We split the C-grid variant into its own translation unit for
 * the same reason the Cartesian variant lives in
 * `initial_conditions_cartesian.cpp`: the unit test for this gate links
 * only this file plus the test_harness, without dragging in advection,
 * microphysics, radiation, etc. via the rest of `equations.cpp`.
 */

#include "core/orchestration/dynamics/initial_conditions.hpp"
#include "core/runtime/simulation.hpp"

#include <cmath>
#include <cstddef>
#include <vector>

// Forward-declared in the same way `initial_conditions_cartesian.cpp` does:
// the production definition lives in `tornado_sim.cpp` and the test files
// supply a stub. Including the full hodograph header here would pull in
// extra link dependencies the unit test does not need.
void compute_wind_profile(const WindProfile& profile, double z,
                          double& u, double& v);

void apply_cylindrical_cgrid_wind_initialization()
{
    const auto& geo = global_grid_geometry;

    std::vector<double> u_x_by_k(static_cast<std::size_t>(NZ));
    std::vector<double> u_y_by_k(static_cast<std::size_t>(NZ));
    for (int k = 0; k < NZ; ++k)
    {
        compute_wind_profile(global_wind_profile, geo.z[k],
                             u_x_by_k[static_cast<std::size_t>(k)],
                             u_y_by_k[static_cast<std::size_t>(k)]);
    }

    // Half-cell trig step for theta_{j+1/2} = theta[j] + 0.5 * dtheta.
    // Using the angle-addition identities lets us reuse the precomputed
    // cell-center sin/cos lookups instead of calling sin/cos in the inner
    // loop:
    //   sin(theta + h) = sin(theta) cos(h) + cos(theta) sin(h)
    //   cos(theta + h) = cos(theta) cos(h) - sin(theta) sin(h)
    const double half_dtheta = 0.5 * dtheta;
    const double cos_half = std::cos(half_dtheta);
    const double sin_half = std::sin(half_dtheta);

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < NR; ++i)
    {
        for (int j = 0; j < NTH; ++j)
        {
            const double cos_th_center = geo.cos_theta[j];
            const double sin_th_center = geo.sin_theta[j];
            const double cos_th_face   = cos_th_center * cos_half
                                       - sin_th_center * sin_half;
            const double sin_th_face   = sin_th_center * cos_half
                                       + cos_th_center * sin_half;

            for (int k = 0; k < NZ; ++k)
            {
                const std::size_t kz = static_cast<std::size_t>(k);
                const double u_x = u_x_by_k[kz];
                const double u_y = u_y_by_k[kz];

                u[i][j][k] = static_cast<float>( u_x * cos_th_center
                                               + u_y * sin_th_center);
                v[i][j][k] = static_cast<float>(-u_x * sin_th_face
                                               + u_y * cos_th_face);
                w[i][j][k] = 0.0f;
            }
        }
    }
}
