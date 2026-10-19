#pragma once

#include <algorithm>

#include "core/field/field3d.hpp"

/**
 * @file derivative_operators.hpp
 * @brief Cartesian finite-difference operators.
 *
 * Provides a uniform interface for computing spatial derivatives on the
 * Cartesian grid. Each implementation handles its own spacing model (uniform
 * vs terrain-aware) and boundary topology (non-periodic in all directions).
 *
 * Collocated operators (DerivativeOperators hierarchy): centered 2-point
 * stencils on cell-center data. Used by the collocated dynamics scheme.
 *
 * Staggered operators (StaggeredCartesianDerivatives): one-sided stencils
 * between cell centers and faces, plus flux-form divergence. Used by the
 * C-grid dynamics scheme.
 */

namespace tmv
{

class DerivativeOperators
{
public:
    virtual ~DerivativeOperators() = default;

    /// Centered derivative along the i-direction (x).
    virtual double di(const Field3D& field, int i, int j, int k) const = 0;

    /// Centered derivative along the j-direction (y).
    virtual double dj(const Field3D& field, int i, int j, int k) const = 0;

    /// Centered derivative along the k-direction (z).
    virtual double dk(const Field3D& field, int i, int j, int k) const = 0;
};

// =========================================================================
// Cartesian: uniform spacing, non-periodic in all directions
// =========================================================================

class CartesianDerivatives : public DerivativeOperators
{
public:
    CartesianDerivatives(double dx, double dy, double dz)
        : dx_(dx), dy_(dy), dz_(dz) {}

    double di(const Field3D& f, int i, int j, int k) const override
    {
        return (f[i + 1][j][k] - f[i - 1][j][k]) / (2.0 * dx_);
    }

    double dj(const Field3D& f, int i, int j, int k) const override
    {
        return (f[i][j + 1][k] - f[i][j - 1][k]) / (2.0 * dy_);
    }

    double dk(const Field3D& f, int i, int j, int k) const override
    {
        return (f[i][j][k + 1] - f[i][j][k - 1]) / (2.0 * dz_);
    }

private:
    double dx_, dy_, dz_;
};

// =========================================================================
// Staggered Cartesian (Arakawa C-grid): one-sided stencils between cell
// centers and faces, plus flux-form divergence. Uniform spacing, no 1/r
// geometry, and non-periodic boundaries in all three directions.
//
// Convention:
//   u_x[i][j][k] lives at x-face (x_{i+1/2}, y_j,       z_k)
//   u_y[i][j][k] lives at y-face (x_i,       y_{j+1/2}, z_k)
//   w  [i][j][k] lives at z-face (x_i,       y_j,       z_{k+1/2})
//   scalars      live at cell center(x_i,    y_j,       z_k)
//
// So u_x[i] is the +x face of cell i; u_x[i-1] is its -x face. w[k] is the
// top (+z) face of cell k; the ground face w[-1] and the lid face are held
// at zero by the no-penetration BC (w[-1]=0 implicit in div_flux_z;
// w[NZ-1]=0 stored). All methods are inline for tight dynamics loops.
//
// Why staggered: the centered 2-point collocated operators
// (CartesianDerivatives) annihilate the 2dx/2dy/2dz checkerboard field
// (f[i+1]-f[i-1] = 0 for an alternating field), leaving it in the null
// space of the gradient/divergence operators -- an undamped computational
// mode. The adjacent-point staggered stencils below (f[i+1]-f[i]) do NOT
// annihilate it, so the pressure-velocity coupling acts on it and the
// checkerboard cannot grow unchecked.
// =========================================================================

class StaggeredCartesianDerivatives
{
public:
    StaggeredCartesianDerivatives(double dx, double dy, double dz,
                                  int NX, int NY, int NZ)
        : inv_dx_(1.0 / dx), inv_dy_(1.0 / dy), inv_dz_(1.0 / dz),
          NX_(NX), NY_(NY), NZ_(NZ) {}

    // -----------------------------------------------------------------
    // One-sided gradients: cell center -> face (adjacent-point)
    // -----------------------------------------------------------------

    /// x-gradient at x-face (i+1/2): (f[i+1] - f[i]) / dx. Valid i in [0, NX-2].
    double grad_x(const Field3D& f, int i, int j, int k) const
    {
        return (static_cast<double>(f[i + 1][j][k])
              - static_cast<double>(f[i][j][k])) * inv_dx_;
    }

    /// y-gradient at y-face (j+1/2): (f[j+1] - f[j]) / dy. Valid j in [0, NY-2].
    double grad_y(const Field3D& f, int i, int j, int k) const
    {
        return (static_cast<double>(f[i][j + 1][k])
              - static_cast<double>(f[i][j][k])) * inv_dy_;
    }

    /// z-gradient at z-face (k+1/2): (f[k+1] - f[k]) / dz. Valid k in [0, NZ-2].
    double grad_z(const Field3D& f, int i, int j, int k) const
    {
        return (static_cast<double>(f[i][j][k + 1])
              - static_cast<double>(f[i][j][k])) * inv_dz_;
    }

    // -----------------------------------------------------------------
    // Flux-form velocity divergence at cell center (adjacent faces)
    // -----------------------------------------------------------------

    /// x-divergence at cell center: (u_x[i] - u_x[i-1]) / dx. Valid i >= 1.
    double div_flux_x(const Field3D& u_x, int i, int j, int k) const
    {
        return (static_cast<double>(u_x[i][j][k])
              - static_cast<double>(u_x[i - 1][j][k])) * inv_dx_;
    }

    /// y-divergence at cell center: (u_y[j] - u_y[j-1]) / dy. Valid j >= 1.
    double div_flux_y(const Field3D& u_y, int i, int j, int k) const
    {
        return (static_cast<double>(u_y[i][j][k])
              - static_cast<double>(u_y[i][j - 1][k])) * inv_dy_;
    }

    /// z-divergence at cell center: (w[k] - w[k-1]) / dz, with the ground
    /// face w[-1] = 0 by the no-penetration BC. Caller enforces the lid
    /// (w[NZ-1] = 0). Valid all k.
    double div_flux_z(const Field3D& w, int i, int j, int k) const
    {
        const double w_top    = static_cast<double>(w[i][j][k]);
        const double w_bottom = (k > 0) ? static_cast<double>(w[i][j][k - 1]) : 0.0;
        return (w_top - w_bottom) * inv_dz_;
    }

    /// Total flux-form velocity divergence at cell center. Valid i,j >= 1.
    double div_flux(const Field3D& u_x, const Field3D& u_y,
                    const Field3D& w, int i, int j, int k) const
    {
        return div_flux_x(u_x, i, j, k)
             + div_flux_y(u_y, i, j, k)
             + div_flux_z(w, i, j, k);
    }

    // -----------------------------------------------------------------
    // Flux-form mass divergence div(rho u) at cell center
    //
    // Each face flux is weighted by the arithmetic-mean rho interpolated
    // from the two cell centers adjacent to that face. The conservative
    // form telescopes face fluxes exactly, so the non-conservative
    // -rho*div(u) mass leak on a stratified atmosphere is avoided. Wall
    // short-circuits at the +x/+y domain faces (u_x[NX-1]=0, u_y[NY-1]=0 by
    // rigid-wall BC) and the lid avoid out-of-bounds rho reads.
    // -----------------------------------------------------------------

    /// x mass-flux divergence at cell center. Valid i >= 1.
    double div_flux_mass_x(const Field3D& rho, const Field3D& u_x,
                           int i, int j, int k) const
    {
        double flux_pos = 0.0;
        if (i + 1 < NX_)
        {
            const double rho_face_pos = 0.5 * (static_cast<double>(rho[i][j][k])
                                             + static_cast<double>(rho[i + 1][j][k]));
            flux_pos = rho_face_pos * static_cast<double>(u_x[i][j][k]);
        }
        // i == NX-1: +x face is rigid wall, u_x[NX-1] = 0 BC -> flux_pos = 0.

        const double rho_face_neg = 0.5 * (static_cast<double>(rho[i - 1][j][k])
                                         + static_cast<double>(rho[i][j][k]));
        const double flux_neg = rho_face_neg * static_cast<double>(u_x[i - 1][j][k]);
        return (flux_pos - flux_neg) * inv_dx_;
    }

    /// y mass-flux divergence at cell center. Valid j >= 1.
    double div_flux_mass_y(const Field3D& rho, const Field3D& u_y,
                           int i, int j, int k) const
    {
        double flux_pos = 0.0;
        if (j + 1 < NY_)
        {
            const double rho_face_pos = 0.5 * (static_cast<double>(rho[i][j][k])
                                             + static_cast<double>(rho[i][j + 1][k]));
            flux_pos = rho_face_pos * static_cast<double>(u_y[i][j][k]);
        }
        // j == NY-1: +y face is rigid wall, u_y[NY-1] = 0 BC -> flux_pos = 0.

        const double rho_face_neg = 0.5 * (static_cast<double>(rho[i][j - 1][k])
                                         + static_cast<double>(rho[i][j][k]));
        const double flux_neg = rho_face_neg * static_cast<double>(u_y[i][j - 1][k]);
        return (flux_pos - flux_neg) * inv_dy_;
    }

    /// z mass-flux divergence at cell center. The surface is impermeable
    /// (w[-1]=0). At the top, rho[NZ-1]*w[NZ-1] is the boundary flux: rigid
    /// closures set w to zero, while an open C-grid closure extrapolates w
    /// from the adjacent interior face.
    double div_flux_mass_z(const Field3D& rho, const Field3D& w,
                           int i, int j, int k) const
    {
        double mass_flux_top = 0.0;
        if (k + 1 < NZ_)
        {
            const double rho_face_top = 0.5 * (static_cast<double>(rho[i][j][k])
                                             + static_cast<double>(rho[i][j][k + 1]));
            mass_flux_top = rho_face_top * static_cast<double>(w[i][j][k]);
        }
        else
        {
            mass_flux_top = static_cast<double>(rho[i][j][k])
                          * static_cast<double>(w[i][j][k]);
        }

        double mass_flux_bottom = 0.0;
        if (k > 0)
        {
            const double rho_face_bot = 0.5 * (static_cast<double>(rho[i][j][k - 1])
                                             + static_cast<double>(rho[i][j][k]));
            mass_flux_bottom = rho_face_bot * static_cast<double>(w[i][j][k - 1]);
        }
        // k == 0: surface, w[-1] = 0 BC -> mass_flux_bottom = 0.

        return (mass_flux_top - mass_flux_bottom) * inv_dz_;
    }

    /// Total flux-form mass divergence at cell center: div(rho u). Valid i,j >= 1.
    double div_flux_mass(const Field3D& rho,
                         const Field3D& u_x, const Field3D& u_y, const Field3D& w,
                         int i, int j, int k) const
    {
        return div_flux_mass_x(rho, u_x, i, j, k)
             + div_flux_mass_y(rho, u_y, i, j, k)
             + div_flux_mass_z(rho, w, i, j, k);
    }

    // -----------------------------------------------------------------
    // Interpolation: cell center <-> face (arithmetic mean)
    // -----------------------------------------------------------------

    /// Scalar to x-face (i+1/2): 0.5*(f[i] + f[i+1]). Valid i in [0, NX-2].
    double interp_to_x_face(const Field3D& f, int i, int j, int k) const
    {
        return 0.5 * (static_cast<double>(f[i][j][k])
                    + static_cast<double>(f[i + 1][j][k]));
    }

    /// Scalar to y-face (j+1/2): 0.5*(f[j] + f[j+1]). Valid j in [0, NY-2].
    double interp_to_y_face(const Field3D& f, int i, int j, int k) const
    {
        return 0.5 * (static_cast<double>(f[i][j][k])
                    + static_cast<double>(f[i][j + 1][k]));
    }

    /// Scalar to z-face (k+1/2): 0.5*(f[k] + f[k+1]). Valid k in [0, NZ-2].
    double interp_to_z_face(const Field3D& f, int i, int j, int k) const
    {
        return 0.5 * (static_cast<double>(f[i][j][k])
                    + static_cast<double>(f[i][j][k + 1]));
    }

    /// x-face quantity to cell center: 0.5*(f[i-1] + f[i]). Valid i >= 1.
    double interp_from_x_face(const Field3D& f, int i, int j, int k) const
    {
        return 0.5 * (static_cast<double>(f[i][j][k])
                    + static_cast<double>(f[i - 1][j][k]));
    }

    /// y-face quantity to cell center: 0.5*(f[j-1] + f[j]). Valid j >= 1.
    double interp_from_y_face(const Field3D& f, int i, int j, int k) const
    {
        return 0.5 * (static_cast<double>(f[i][j][k])
                    + static_cast<double>(f[i][j - 1][k]));
    }

    /// z-face quantity to cell center: 0.5*(f[k-1] + f[k]). Bottom (k=0):
    /// 0.5*f[0] (the surface face w[-1] is the rigid ground, w=0).
    double interp_from_z_face(const Field3D& f, int i, int j, int k) const
    {
        if (k == 0)
        {
            return 0.5 * static_cast<double>(f[i][j][0]);
        }
        return 0.5 * (static_cast<double>(f[i][j][k])
                    + static_cast<double>(f[i][j][k - 1]));
    }

    // -----------------------------------------------------------------
    // Upwind-biased first derivatives for momentum advection
    //
    // Centered (2nd-order) advection leaves the 2dx grid-scale mode in the
    // null space of the operator with NO dissipation -- on a free-slip
    // surface under active convection that undamped mode runs away. Odd-order
    // upwind-biased advection carries an
    // implicit, scale-selective even-order dissipation (Wicker & Skamarock
    // 2002; the scheme CM1/WRF use for stability): negligible on resolved
    // smooth flow, strongly damping at 2dx. The advected field `f` and the
    // result share the same index space; the advecting velocity `a` only
    // selects the upwind bias (its magnitude multiplies the derivative at
    // the call site as -a*df/dx). Order auto-degrades to 3rd then 1st within
    // 3 / 2 cells of the boundary, where the wide stencil is unavailable.
    // -----------------------------------------------------------------

    /// Upwind-biased d/dx at index i (same index space as f).
    double upwind_dx(const Field3D& f, double a, int i, int j, int k) const
    {
        const int half = std::min(i, NX_ - 1 - i);
        const double c = static_cast<double>(f[i][j][k]);
        if (half <= 0) return 0.0;
        const double m1 = static_cast<double>(f[i - 1][j][k]);
        const double p1 = static_cast<double>(f[i + 1][j][k]);
        if (half == 1) return upwind_fd1(m1, c, p1, a, inv_dx_);
        const double m2 = static_cast<double>(f[i - 2][j][k]);
        const double p2 = static_cast<double>(f[i + 2][j][k]);
        if (half == 2) return upwind_fd3(m2, m1, c, p1, p2, a, inv_dx_);
        const double m3 = static_cast<double>(f[i - 3][j][k]);
        const double p3 = static_cast<double>(f[i + 3][j][k]);
        return upwind_fd5(m3, m2, m1, c, p1, p2, p3, a, inv_dx_);
    }

    /// Upwind-biased d/dy at index j (same index space as f).
    double upwind_dy(const Field3D& f, double a, int i, int j, int k) const
    {
        const int half = std::min(j, NY_ - 1 - j);
        const double c = static_cast<double>(f[i][j][k]);
        if (half <= 0) return 0.0;
        const double m1 = static_cast<double>(f[i][j - 1][k]);
        const double p1 = static_cast<double>(f[i][j + 1][k]);
        if (half == 1) return upwind_fd1(m1, c, p1, a, inv_dy_);
        const double m2 = static_cast<double>(f[i][j - 2][k]);
        const double p2 = static_cast<double>(f[i][j + 2][k]);
        if (half == 2) return upwind_fd3(m2, m1, c, p1, p2, a, inv_dy_);
        const double m3 = static_cast<double>(f[i][j - 3][k]);
        const double p3 = static_cast<double>(f[i][j + 3][k]);
        return upwind_fd5(m3, m2, m1, c, p1, p2, p3, a, inv_dy_);
    }

    /// Upwind-biased d/dz at index k (same index space as f). Operates purely
    /// on stored cells; fields with a virtual ground/lid face (w) handle k=0
    /// / k=NZ-1 at the call site.
    double upwind_dz(const Field3D& f, double a, int i, int j, int k) const
    {
        const int half = std::min(k, NZ_ - 1 - k);
        const double c = static_cast<double>(f[i][j][k]);
        if (half <= 0) return 0.0;
        const double m1 = static_cast<double>(f[i][j][k - 1]);
        const double p1 = static_cast<double>(f[i][j][k + 1]);
        if (half == 1) return upwind_fd1(m1, c, p1, a, inv_dz_);
        const double m2 = static_cast<double>(f[i][j][k - 2]);
        const double p2 = static_cast<double>(f[i][j][k + 2]);
        if (half == 2) return upwind_fd3(m2, m1, c, p1, p2, a, inv_dz_);
        const double m3 = static_cast<double>(f[i][j][k - 3]);
        const double p3 = static_cast<double>(f[i][j][k + 3]);
        return upwind_fd5(m3, m2, m1, c, p1, p2, p3, a, inv_dz_);
    }

private:
    // 1st-order upwind: a>=0 backward (c - m1), a<0 forward (p1 - c).
    static double upwind_fd1(double m1, double c, double p1,
                             double a, double inv_d)
    {
        return (a >= 0.0) ? (c - m1) * inv_d : (p1 - c) * inv_d;
    }

    // 3rd-order upwind-biased first derivative.
    static double upwind_fd3(double m2, double m1, double c, double p1, double p2,
                             double a, double inv_d)
    {
        return (a >= 0.0)
            ? (2.0 * p1 + 3.0 * c - 6.0 * m1 + m2) * (inv_d / 6.0)
            : (-p2 + 6.0 * p1 - 3.0 * c - 2.0 * m1) * (inv_d / 6.0);
    }

    // 5th-order upwind-biased first derivative (Wicker & Skamarock 2002).
    static double upwind_fd5(double m3, double m2, double m1, double c,
                             double p1, double p2, double p3,
                             double a, double inv_d)
    {
        return (a >= 0.0)
            ? (-2.0 * m3 + 15.0 * m2 - 60.0 * m1 + 20.0 * c + 30.0 * p1 - 3.0 * p2) * (inv_d / 60.0)
            : ( 2.0 * p3 - 15.0 * p2 + 60.0 * p1 - 20.0 * c - 30.0 * m1 + 3.0 * m2) * (inv_d / 60.0);
    }

    double inv_dx_, inv_dy_, inv_dz_;
    int NX_, NY_, NZ_;
};

} // namespace tmv

using tmv::DerivativeOperators;
using tmv::CartesianDerivatives;
using tmv::StaggeredCartesianDerivatives;
