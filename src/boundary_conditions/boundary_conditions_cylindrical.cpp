/**
 * @file boundary_conditions_cylindrical.cpp
 * @brief Cylindrical boundary condition scheme (Phase B.3).
 *
 * Axis-reflection at i=0 (antisymmetric u, symmetric w/rho/p/scalars),
 * zero-gradient at i=NR-1, vertical rigid lid + rigid surface with
 * hydrostatic pressure extrapolation.
 */

#include "boundary_conditions/boundary_conditions_base.hpp"
#include "core/simulation.hpp"
#include "dynamics/dynamics_base.hpp"

#include <algorithm>
#include <cmath>

class CylindricalBCScheme : public BoundaryConditionScheme
{
public:
    void apply_full() override
    {
        // Radial faces: antisymmetric u at axis (i=0) and outer edge (i=NR-1),
        // symmetric/zero-gradient for all other fields.
        for (int j = 0; j < NTH; ++j)
            for (int k = 0; k < NZ; ++k)
            {
                u[0][j][k]    = -u[1][j][k];
                u[NR-1][j][k] = -u[NR-2][j][k];
                w[0][j][k]    =  w[1][j][k];
                w[NR-1][j][k] =  w[NR-2][j][k];
                rho[0][j][k]    = rho[1][j][k];
                rho[NR-1][j][k] = rho[NR-2][j][k];
                p[0][j][k]    = p[1][j][k];
                p[NR-1][j][k] = p[NR-2][j][k];
            }

        // Vertical faces: rigid lid/surface + hydrostatic pressure
        const double g_val  = dynamics_constants::g;
        const double dz_local = (std::isfinite(dz) && dz > 0.0) ? dz : 1.0;

        for (int i = 0; i < NR; ++i)
            for (int j = 0; j < NTH; ++j)
            {
                w[i][j][0] = 0.0f;  w[i][j][NZ - 1] = 0.0f;

                u[i][j][0]      = u[i][j][1];
                u[i][j][NZ - 1] = u[i][j][NZ - 2];

                rho[i][j][0]      = rho[i][j][1];
                rho[i][j][NZ - 1] = rho[i][j][NZ - 2];

                const double rho_top = std::max(static_cast<double>(rho[i][j][NZ - 2]), 1.0e-3);
                const double rho_bot = std::max(static_cast<double>(rho[i][j][1]),      1.0e-3);
                p[i][j][NZ - 1] = static_cast<float>(
                    static_cast<double>(p[i][j][NZ - 2]) - rho_top * g_val * dz_local);
                p[i][j][0] = static_cast<float>(
                    static_cast<double>(p[i][j][1]) + rho_bot * g_val * dz_local);

                theta[i][j][0] = theta[i][j][1];  theta[i][j][NZ - 1] = theta[i][j][NZ - 2];
                qv[i][j][0] = qv[i][j][1]; qv[i][j][NZ - 1] = qv[i][j][NZ - 2];
                qc[i][j][0] = qc[i][j][1]; qc[i][j][NZ - 1] = qc[i][j][NZ - 2];
                qr[i][j][0] = qr[i][j][1]; qr[i][j][NZ - 1] = qr[i][j][NZ - 2];
                qi[i][j][0] = qi[i][j][1]; qi[i][j][NZ - 1] = qi[i][j][NZ - 2];
                qs[i][j][0] = qs[i][j][1]; qs[i][j][NZ - 1] = qs[i][j][NZ - 2];
                qg[i][j][0] = qg[i][j][1]; qg[i][j][NZ - 1] = qg[i][j][NZ - 2];
                qh[i][j][0] = qh[i][j][1]; qh[i][j][NZ - 1] = qh[i][j][NZ - 2];
            }

        // Radial scalar BCs (theta + moisture at i = 0, NR-1)
        for (int j = 0; j < NTH; ++j)
            for (int k = 0; k < NZ; ++k)
            {
                theta[0][j][k] = theta[1][j][k]; theta[NR-1][j][k] = theta[NR-2][j][k];
                qv[0][j][k] = qv[1][j][k]; qv[NR-1][j][k] = qv[NR-2][j][k];
                qc[0][j][k] = qc[1][j][k]; qc[NR-1][j][k] = qc[NR-2][j][k];
                qr[0][j][k] = qr[1][j][k]; qr[NR-1][j][k] = qr[NR-2][j][k];
                qi[0][j][k] = qi[1][j][k]; qi[NR-1][j][k] = qi[NR-2][j][k];
                qs[0][j][k] = qs[1][j][k]; qs[NR-1][j][k] = qs[NR-2][j][k];
                qg[0][j][k] = qg[1][j][k]; qg[NR-1][j][k] = qg[NR-2][j][k];
                qh[0][j][k] = qh[1][j][k]; qh[NR-1][j][k] = qh[NR-2][j][k];
            }
    }

    void apply_acoustic() override
    {
        // Radial faces: antisymmetric u, symmetric others (momentum fields only)
        for (int j = 0; j < NTH; ++j)
            for (int k = 0; k < NZ; ++k)
            {
                u[0][j][k]    = -u[1][j][k];      u[NR-1][j][k]    = -u[NR-2][j][k];
                w[0][j][k]    =  w[1][j][k];       w[NR-1][j][k]    =  w[NR-2][j][k];
                rho[0][j][k]  =  rho[1][j][k];     rho[NR-1][j][k]  =  rho[NR-2][j][k];
                p[0][j][k]    =  p[1][j][k];       p[NR-1][j][k]    =  p[NR-2][j][k];
            }

        // Vertical: rigid lid/surface + hydrostatic pressure (momentum only)
        const double g_val  = dynamics_constants::g;
        const double dz_local = (std::isfinite(dz) && dz > 0.0) ? dz : 1.0;

        for (int i = 0; i < NR; ++i)
            for (int j = 0; j < NTH; ++j)
            {
                w[i][j][0] = 0.0f;  w[i][j][NZ - 1] = 0.0f;
                u[i][j][0] = u[i][j][1];           u[i][j][NZ - 1] = u[i][j][NZ - 2];
                v[i][j][0] = v[i][j][1];           v[i][j][NZ - 1] = v[i][j][NZ - 2];
                rho[i][j][0] = rho[i][j][1];       rho[i][j][NZ - 1] = rho[i][j][NZ - 2];
                const double rho_top = std::max(static_cast<double>(rho[i][j][NZ - 2]), 1.0e-3);
                const double rho_bot = std::max(static_cast<double>(rho[i][j][1]),      1.0e-3);
                p[i][j][NZ - 1] = static_cast<float>(
                    static_cast<double>(p[i][j][NZ - 2]) - rho_top * g_val * dz_local);
                p[i][j][0] = static_cast<float>(
                    static_cast<double>(p[i][j][1]) + rho_bot * g_val * dz_local);
            }
    }

    std::string get_scheme_name() const override { return "cylindrical"; }
};

// Factory access
std::unique_ptr<BoundaryConditionScheme> create_cylindrical_bc_scheme()
{
    return std::make_unique<CylindricalBCScheme>();
}
