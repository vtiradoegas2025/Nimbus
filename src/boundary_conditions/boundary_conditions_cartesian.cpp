/**
 * @file boundary_conditions_cartesian.cpp
 * @brief Cartesian boundary condition scheme (Phase B.3).
 *
 * Open lateral (zero-gradient) boundaries on the four x/y faces and
 * rigid-lid + rigid-surface boundaries on the two vertical faces, with
 * hydrostatic pressure extrapolation.
 */

#include "boundary_conditions/boundary_conditions_base.hpp"
#include "core/simulation.hpp"
#include "dynamics/dynamics_base.hpp"

#include <algorithm>
#include <cmath>

namespace
{

inline void copy_lateral_x_face(int j, int k, int dst_i, int src_i)
{
    u[dst_i][j][k]     = u[src_i][j][k];
    v[dst_i][j][k]     = v[src_i][j][k];
    w[dst_i][j][k]     = w[src_i][j][k];
    rho[dst_i][j][k]   = rho[src_i][j][k];
    p[dst_i][j][k]     = p[src_i][j][k];
    theta[dst_i][j][k] = theta[src_i][j][k];
    qv[dst_i][j][k]    = qv[src_i][j][k];
    qc[dst_i][j][k]    = qc[src_i][j][k];
    qr[dst_i][j][k]    = qr[src_i][j][k];
    qi[dst_i][j][k]    = qi[src_i][j][k];
    qs[dst_i][j][k]    = qs[src_i][j][k];
    qg[dst_i][j][k]    = qg[src_i][j][k];
    qh[dst_i][j][k]    = qh[src_i][j][k];
}

inline void copy_lateral_y_face(int i, int k, int dst_j, int src_j)
{
    u[i][dst_j][k]     = u[i][src_j][k];
    v[i][dst_j][k]     = v[i][src_j][k];
    w[i][dst_j][k]     = w[i][src_j][k];
    rho[i][dst_j][k]   = rho[i][src_j][k];
    p[i][dst_j][k]     = p[i][src_j][k];
    theta[i][dst_j][k] = theta[i][src_j][k];
    qv[i][dst_j][k]    = qv[i][src_j][k];
    qc[i][dst_j][k]    = qc[i][src_j][k];
    qr[i][dst_j][k]    = qr[i][src_j][k];
    qi[i][dst_j][k]    = qi[i][src_j][k];
    qs[i][dst_j][k]    = qs[i][src_j][k];
    qg[i][dst_j][k]    = qg[i][src_j][k];
    qh[i][dst_j][k]    = qh[i][src_j][k];
}

/// Vertical BCs shared by full and acoustic paths (both coordinate systems).
void apply_vertical_bcs_momentum()
{
    const double g_val  = dynamics_constants::g;
    const double dz_local = (std::isfinite(dz) && dz > 0.0) ? dz : 1.0;

    for (int i = 0; i < NR; ++i)
        for (int j = 0; j < NTH; ++j)
        {
            w[i][j][0] = 0.0f;  w[i][j][NZ - 1] = 0.0f;

            u[i][j][0]      = u[i][j][1];
            u[i][j][NZ - 1] = u[i][j][NZ - 2];
            v[i][j][0]      = v[i][j][1];
            v[i][j][NZ - 1] = v[i][j][NZ - 2];

            rho[i][j][0]      = rho[i][j][1];
            rho[i][j][NZ - 1] = rho[i][j][NZ - 2];

            const double rho_top = std::max(static_cast<double>(rho[i][j][NZ - 2]), 1.0e-3);
            const double rho_bot = std::max(static_cast<double>(rho[i][j][1]),      1.0e-3);
            p[i][j][NZ - 1] = static_cast<float>(
                static_cast<double>(p[i][j][NZ - 2]) - rho_top * g_val * dz_local);
            p[i][j][0] = static_cast<float>(
                static_cast<double>(p[i][j][1]) + rho_bot * g_val * dz_local);
        }
}

}  // namespace

class CartesianBCScheme : public BoundaryConditionScheme
{
public:
    void apply_full() override
    {
        // Lateral x faces: zero-gradient
        for (int j = 0; j < NTH; ++j)
            for (int k = 0; k < NZ; ++k)
            {
                copy_lateral_x_face(j, k, 0, 1);
                copy_lateral_x_face(j, k, NR - 1, NR - 2);
            }

        // Lateral y faces: zero-gradient
        for (int i = 0; i < NR; ++i)
            for (int k = 0; k < NZ; ++k)
            {
                copy_lateral_y_face(i, k, 0, 1);
                copy_lateral_y_face(i, k, NTH - 1, NTH - 2);
            }

        // Vertical faces: rigid lid/surface + hydrostatic p
        apply_vertical_bcs_momentum();

        // Vertical scalars (theta + moisture): zero-gradient
        for (int i = 0; i < NR; ++i)
            for (int j = 0; j < NTH; ++j)
            {
                theta[i][j][0] = theta[i][j][1];  theta[i][j][NZ - 1] = theta[i][j][NZ - 2];
                qv[i][j][0] = qv[i][j][1]; qv[i][j][NZ - 1] = qv[i][j][NZ - 2];
                qc[i][j][0] = qc[i][j][1]; qc[i][j][NZ - 1] = qc[i][j][NZ - 2];
                qr[i][j][0] = qr[i][j][1]; qr[i][j][NZ - 1] = qr[i][j][NZ - 2];
                qi[i][j][0] = qi[i][j][1]; qi[i][j][NZ - 1] = qi[i][j][NZ - 2];
                qs[i][j][0] = qs[i][j][1]; qs[i][j][NZ - 1] = qs[i][j][NZ - 2];
                qg[i][j][0] = qg[i][j][1]; qg[i][j][NZ - 1] = qg[i][j][NZ - 2];
                qh[i][j][0] = qh[i][j][1]; qh[i][j][NZ - 1] = qh[i][j][NZ - 2];
            }
    }

    void apply_acoustic() override
    {
        // Lateral x faces: zero-gradient (momentum fields only)
        for (int j = 0; j < NTH; ++j)
            for (int k = 0; k < NZ; ++k)
            {
                u[0][j][k]   = u[1][j][k];       u[NR-1][j][k]   = u[NR-2][j][k];
                v[0][j][k]   = v[1][j][k];       v[NR-1][j][k]   = v[NR-2][j][k];
                w[0][j][k]   = w[1][j][k];       w[NR-1][j][k]   = w[NR-2][j][k];
                rho[0][j][k] = rho[1][j][k];     rho[NR-1][j][k] = rho[NR-2][j][k];
                p[0][j][k]   = p[1][j][k];       p[NR-1][j][k]   = p[NR-2][j][k];
            }

        // Lateral y faces: zero-gradient (momentum fields only)
        for (int i = 0; i < NR; ++i)
            for (int k = 0; k < NZ; ++k)
            {
                u[i][0][k]   = u[i][1][k];       u[i][NTH-1][k]   = u[i][NTH-2][k];
                v[i][0][k]   = v[i][1][k];       v[i][NTH-1][k]   = v[i][NTH-2][k];
                w[i][0][k]   = w[i][1][k];       w[i][NTH-1][k]   = w[i][NTH-2][k];
                rho[i][0][k] = rho[i][1][k];     rho[i][NTH-1][k] = rho[i][NTH-2][k];
                p[i][0][k]   = p[i][1][k];       p[i][NTH-1][k]   = p[i][NTH-2][k];
            }

        // Vertical: shared momentum BCs
        apply_vertical_bcs_momentum();
    }

    std::string get_scheme_name() const override { return "cartesian"; }
};

// Factory access
std::unique_ptr<BoundaryConditionScheme> create_cartesian_bc_scheme()
{
    return std::make_unique<CartesianBCScheme>();
}
