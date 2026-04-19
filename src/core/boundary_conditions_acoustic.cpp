/**
 * @file boundary_conditions_acoustic.cpp
 * @brief Lightweight boundary conditions for acoustic substeps.
 *
 * Applies BCs to the momentum and thermodynamic fields (u, v, w, rho, p)
 * only — skips theta and all moisture fields. Used as a callback by the
 * split-explicit time stepping scheme between acoustic substeps.
 *
 * Dispatches internally by coordinate system: cylindrical uses antisymmetric
 * u_r at i=0; Cartesian uses zero-gradient on all lateral faces.
 *
 * Extracted from src/core/dynamics.cpp.
 */

#include "core/boundary_conditions.hpp"
#include "core/runtime_config.hpp"
#include "core/simulation.hpp"
#include "physics/dynamics_base.hpp"

#include <algorithm>
#include <cmath>

void apply_acoustic_boundary_conditions()
{
    const double g_val = dynamics_constants::g;
    const double dz_local = (std::isfinite(dz) && dz > 0.0) ? dz : 1.0;

    if (global_coordinate_system == CoordinateSystem::Cartesian)
    {
        for (int j = 0; j < NTH; ++j)
            for (int k = 0; k < NZ; ++k)
            {
                u[0][j][k]       = u[1][j][k];       u[NR-1][j][k]       = u[NR-2][j][k];
                v_theta[0][j][k] = v_theta[1][j][k]; v_theta[NR-1][j][k] = v_theta[NR-2][j][k];
                w[0][j][k]       = w[1][j][k];       w[NR-1][j][k]       = w[NR-2][j][k];
                rho[0][j][k]     = rho[1][j][k];     rho[NR-1][j][k]     = rho[NR-2][j][k];
                p[0][j][k]       = p[1][j][k];       p[NR-1][j][k]       = p[NR-2][j][k];
            }
        for (int i = 0; i < NR; ++i)
            for (int k = 0; k < NZ; ++k)
            {
                u[i][0][k]       = u[i][1][k];       u[i][NTH-1][k]       = u[i][NTH-2][k];
                v_theta[i][0][k] = v_theta[i][1][k]; v_theta[i][NTH-1][k] = v_theta[i][NTH-2][k];
                w[i][0][k]       = w[i][1][k];       w[i][NTH-1][k]       = w[i][NTH-2][k];
                rho[i][0][k]     = rho[i][1][k];     rho[i][NTH-1][k]     = rho[i][NTH-2][k];
                p[i][0][k]       = p[i][1][k];       p[i][NTH-1][k]       = p[i][NTH-2][k];
            }
    }
    else
    {
        for (int j = 0; j < NTH; ++j)
            for (int k = 0; k < NZ; ++k)
            {
                u[0][j][k]   = -u[1][j][k];      u[NR-1][j][k]   = -u[NR-2][j][k];
                w[0][j][k]   = w[1][j][k];        w[NR-1][j][k]   = w[NR-2][j][k];
                rho[0][j][k] = rho[1][j][k];      rho[NR-1][j][k] = rho[NR-2][j][k];
                p[0][j][k]   = p[1][j][k];        p[NR-1][j][k]   = p[NR-2][j][k];
            }
    }
    for (int i = 0; i < NR; ++i)
        for (int j = 0; j < NTH; ++j)
        {
            w[i][j][0] = 0.0f;  w[i][j][NZ-1] = 0.0f;
            u[i][j][0] = u[i][j][1];             u[i][j][NZ-1] = u[i][j][NZ-2];
            v_theta[i][j][0] = v_theta[i][j][1]; v_theta[i][j][NZ-1] = v_theta[i][j][NZ-2];
            rho[i][j][0] = rho[i][j][1];         rho[i][j][NZ-1] = rho[i][j][NZ-2];
            const double rho_top = std::max(static_cast<double>(rho[i][j][NZ-2]), 1.0e-3);
            const double rho_bot = std::max(static_cast<double>(rho[i][j][1]),    1.0e-3);
            p[i][j][NZ-1] = static_cast<float>(static_cast<double>(p[i][j][NZ-2]) - rho_top * g_val * dz_local);
            p[i][j][0]    = static_cast<float>(static_cast<double>(p[i][j][1])    + rho_bot * g_val * dz_local);
        }
}
