/**
 * @file initial_conditions_cartesian.cpp
 * @brief Cartesian initial-condition implementation (Phase A.4).
 *
 * Implements the two helpers declared in `core/initial_conditions.hpp`:
 *
 *   - apply_cartesian_wind_initialization():
 *       Fills (u, v_theta, w) with (u_x(z), u_y(z), 0) — no `cos θ`/`sin θ`
 *       projection. The cylindrical-named globals are aliased to Cartesian
 *       components while the Cartesian dynamics scheme is active.
 *
 *   - apply_cartesian_bubble_initialization():
 *       Adds a 3D Gaussian Δθ patch centered at the configured
 *       (x_c, y_c, z_c). The cylindrical path uses a 2D ring in (r, z); the
 *       Cartesian path uses a literal sphere in (x, y, z).
 *
 * Both helpers operate on the global Field3D state declared in
 * `core/simulation.hpp` and the bubble configuration globals declared in
 * the same header. This file is linked by both the production runtime
 * (via `equations.cpp::initialize`) and the A.4 unit test (via the
 * `tests/dynamics/test_cartesian_initial_conditions.cpp` driver), so the
 * helpers must remain free of advection / microphysics / radiation
 * dependencies.
 *
 * The cylindrical implementation lives inline in
 * `src/core/equations.cpp::initialize`. The dispatcher in that same
 * function calls into this file when
 * `global_coordinate_system == CoordinateSystem::Cartesian`.
 */

#include "core/initial_conditions.hpp"
#include "core/simulation.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

// Forward declaration: defined in `src/core/tornado_sim.cpp`. The function
// returns the (u_x, u_y) Cartesian wind components at altitude `z` from the
// `WindProfile` hodograph. We do not include `tornado_sim.hpp` (no such
// header exists for this signature) because we want to keep the link
// closure of this file as small as possible.
void compute_wind_profile(const WindProfile& profile, double z, double& u, double& v);

void apply_cartesian_wind_initialization()
{
    // Precompute the (u_x, u_y) profile by k. This is a tiny vector (NZ
    // entries) but it lets us hoist the wind-profile evaluation out of the
    // (i, j) loop, which dominates work for large grids. The same hoist is
    // worth doing in the cylindrical path too, but that's a Phase B cleanup.
    std::vector<double> u_x_by_k(static_cast<std::size_t>(NZ));
    std::vector<double> u_y_by_k(static_cast<std::size_t>(NZ));
    for (int k = 0; k < NZ; ++k)
    {
        const double z = static_cast<double>(k) * dz;
        compute_wind_profile(global_wind_profile, z,
                             u_x_by_k[static_cast<std::size_t>(k)],
                             u_y_by_k[static_cast<std::size_t>(k)]);
    }

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < NR; ++i)
    {
        for (int j = 0; j < NTH; ++j)
        {
            for (int k = 0; k < NZ; ++k)
            {
                const std::size_t kz = static_cast<std::size_t>(k);
                u[i][j][k]       = static_cast<float>(u_x_by_k[kz]);
                v_theta[i][j][k] = static_cast<float>(u_y_by_k[kz]);
                w[i][j][k]       = 0.0f;
            }
        }
    }
}

void apply_cartesian_bubble_initialization()
{
    // Read every per-call parameter into a local before the parallel loop
    // so the OpenMP firstprivate semantics are explicit and we don't reread
    // a global on every iteration. The 100 m radius floor matches the
    // cylindrical helper — the rest of the code assumes a positive radius.
    const double bubble_center_x  = std::max(0.0, global_bubble_center_x_m);
    const double bubble_center_y  = std::max(0.0, global_bubble_center_y_m);
    const double bubble_center_z  = std::max(0.0, global_bubble_center_z_m);
    const double bubble_radius    = std::max(100.0, global_bubble_radius_m);
    const double bubble_dtheta    = global_bubble_dtheta_k;
    const double bubble_radius_sq = bubble_radius * bubble_radius;
    // The Gaussian is exp(−(dist / (radius / 3))²) which equals
    // exp(−(dist * 3 / radius)²). Caching `3 / radius` avoids two divisions
    // per cell in the inner loop.
    const double inv_third_radius = 3.0 / bubble_radius;

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < NR; ++i)
    {
        for (int j = 0; j < NTH; ++j)
        {
            const double x_dist = static_cast<double>(i) * dr - bubble_center_x;
            const double y_dist = static_cast<double>(j) * dr - bubble_center_y;
            const double xy_sq  = x_dist * x_dist + y_dist * y_dist;

            // Cull entire columns whose horizontal distance already exceeds
            // the bubble radius — saves the inner k loop entirely for cells
            // far from the bubble center.
            if (xy_sq > bubble_radius_sq)
            {
                continue;
            }

            for (int k = 0; k < NZ; ++k)
            {
                const double z_dist = static_cast<double>(k) * dz - bubble_center_z;
                const double dist_sq = xy_sq + z_dist * z_dist;
                if (dist_sq > bubble_radius_sq)
                {
                    continue;
                }
                const double dist = std::sqrt(dist_sq);
                const double scaled = dist * inv_third_radius;
                const double bubble_factor = std::exp(-(scaled * scaled));
                theta[i][j][k] += static_cast<float>(bubble_dtheta * bubble_factor);
            }
        }
    }
}
