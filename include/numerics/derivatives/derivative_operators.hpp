#pragma once

#include <algorithm>
#include <cmath>

#include "core/runtime/simulation.hpp"
#include "core/infra/grid_geometry.hpp"
#include "util/grid_metric_utils.hpp"

/**
 * @file derivative_operators.hpp
 * @brief Coordinate-system-specific finite-difference operators.
 *
 * Provides a uniform interface for computing spatial derivatives across
 * Cartesian and cylindrical coordinate systems. Each implementation handles
 * its own spacing model (uniform vs terrain-aware) and boundary topology
 * (non-periodic vs periodic azimuthal).
 *
 * Collocated operators (DerivativeOperators hierarchy): centered 2-point
 * stencils on cell-center data. Used by collocated dynamics schemes
 * (Phase B.2).
 *
 * Staggered operators (StaggeredCylindricalDerivatives): one-sided stencils
 * between cell centers and faces, plus flux-form divergence with axis
 * treatment. Used by C-grid dynamics schemes (Phase C).
 */

class DerivativeOperators
{
public:
    virtual ~DerivativeOperators() = default;

    /// Centered derivative along the i-direction (x or r).
    virtual double di(const Field3D& field, int i, int j, int k) const = 0;

    /// Centered derivative along the j-direction (y or theta).
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
// Cylindrical: terrain-aware spacing, periodic theta wrapping
// =========================================================================

class CylindricalDerivatives : public DerivativeOperators
{
public:
    CylindricalDerivatives(const GridMetrics& metrics, double dtheta, int NTH, int NZ)
        : metrics_(metrics), dtheta_(dtheta), NTH_(NTH), NZ_(NZ) {}

    double di(const Field3D& f, int i, int j, int k) const override
    {
        const double dx_local = std::max(
            grid_metric::local_dx(metrics_, i, j, k), 1.0e-6);
        return (f[i + 1][j][k] - f[i - 1][j][k]) / (2.0 * dx_local);
    }

    double dj(const Field3D& f, int i, int j, int k) const override
    {
        const int j_prev = (j - 1 + NTH_) % NTH_;
        const int j_next = (j + 1) % NTH_;
        return (f[i][j_next][k] - f[i][j_prev][k]) / (2.0 * dtheta_);
    }

    double dk(const Field3D& f, int i, int j, int k) const override
    {
        const double denom = std::max(
            grid_metric::centered_dz_span(metrics_, i, j, k, NZ_), 1.0e-6);
        return (f[i][j][k + 1] - f[i][j][k - 1]) / denom;
    }

private:
    const GridMetrics& metrics_;
    double dtheta_;
    int NTH_;
    int NZ_;
};

// =========================================================================
// Staggered cylindrical (Arakawa C-grid): one-sided stencils, flux-form
// divergence with axis singularity treatment.
//
// Convention (following CoordinateBackend_Plan.md):
//   u[i][j][k] lives at r-face   (r_{i+1/2}, theta_j,      z_k)
//   v[i][j][k] lives at theta-face(r_i,       theta_{j+1/2}, z_k)
//   w[i][j][k] lives at z-face   (r_i,       theta_j,       z_{k+1/2})
//   scalars    live at cell center(r_i,       theta_j,       z_k)
//
// All methods are inline for use in tight dynamics loops.
// =========================================================================

class StaggeredCylindricalDerivatives
{
public:
    StaggeredCylindricalDerivatives(const GridGeometry& geo, int NTH)
        : geo_(geo), NTH_(NTH) {}

    // -----------------------------------------------------------------
    // One-sided gradients: cell center -> face
    // -----------------------------------------------------------------

    /// Radial gradient at r-face (i+1/2): (f[i+1] - f[i]) / dr.
    /// Valid for i in [0, NR-2].
    double grad_r(const Field3D& f, int i, int j, int k) const
    {
        return (static_cast<double>(f[i + 1][j][k])
              - static_cast<double>(f[i][j][k])) * geo_.inv_dr;
    }

    /// Azimuthal gradient at theta-face (j+1/2): (f[j+1] - f[j]) / (r_i * dtheta).
    /// Periodic in j. Valid for i >= 1 (r_inv[0] = 0 at the axis).
    double grad_theta(const Field3D& f, int i, int j, int k) const
    {
        const int j_next = (j + 1) % NTH_;
        return (static_cast<double>(f[i][j_next][k])
              - static_cast<double>(f[i][j][k])) * geo_.inv_dtheta * geo_.r_inv[i];
    }

    /// Vertical gradient at z-face (k+1/2): (f[k+1] - f[k]) / dz.
    /// Valid for k in [0, NZ-2].
    double grad_z(const Field3D& f, int i, int j, int k) const
    {
        return (static_cast<double>(f[i][j][k + 1])
              - static_cast<double>(f[i][j][k])) * geo_.inv_dz;
    }

    // -----------------------------------------------------------------
    // Flux-form divergence components at cell center
    // -----------------------------------------------------------------

    /// Radial flux divergence at cell center i:
    ///   i > 0: (r_face[i]*u[i] - r_face[i-1]*u[i-1]) / (r[i] * dr)
    ///   i = 0: 2 * u[0] / dr   (control-volume axis formula)
    double div_flux_r(const Field3D& u_face, int i, int j, int k) const
    {
        if (i == 0)
        {
            return 2.0 * static_cast<double>(u_face[0][j][k]) * geo_.inv_dr;
        }
        const double flux_outer = geo_.r_face[i]     * static_cast<double>(u_face[i][j][k]);
        const double flux_inner = geo_.r_face[i - 1]  * static_cast<double>(u_face[i - 1][j][k]);
        return (flux_outer - flux_inner) * geo_.inv_dr * geo_.r_inv[i];
    }

    /// Azimuthal flux divergence at cell center:
    ///   (v[j] - v[j-1]) / (r_i * dtheta)
    /// Periodic in j. At axis (i=0) returns 0 (r_inv[0] = 0).
    double div_flux_theta(const Field3D& v_face, int i, int j, int k) const
    {
        const int j_prev = (j - 1 + NTH_) % NTH_;
        return (static_cast<double>(v_face[i][j][k])
              - static_cast<double>(v_face[i][j_prev][k])) * geo_.inv_dtheta * geo_.r_inv[i];
    }

    /// Vertical flux divergence at cell center:
    ///   (w[k] - w[k-1]) / dz
    /// Caller must enforce w boundary conditions (w[-1] = 0 at surface,
    /// w[NZ-1] = 0 at top).
    double div_flux_z(const Field3D& w_face, int i, int j, int k) const
    {
        const double w_top    = static_cast<double>(w_face[i][j][k]);
        const double w_bottom = (k > 0) ? static_cast<double>(w_face[i][j][k - 1]) : 0.0;
        return (w_top - w_bottom) * geo_.inv_dz;
    }

    /// Total flux-form divergence at cell center.
    double div_flux(const Field3D& u_face, const Field3D& v_face,
                    const Field3D& w_face, int i, int j, int k) const
    {
        return div_flux_r(u_face, i, j, k)
             + div_flux_theta(v_face, i, j, k)
             + div_flux_z(w_face, i, j, k);
    }

    // -----------------------------------------------------------------
    // Interpolation: cell center -> face (arithmetic mean)
    // -----------------------------------------------------------------

    /// Scalar to r-face (i+1/2): 0.5*(f[i] + f[i+1]).
    double interp_to_r_face(const Field3D& f, int i, int j, int k) const
    {
        return 0.5 * (static_cast<double>(f[i][j][k])
                    + static_cast<double>(f[i + 1][j][k]));
    }

    /// Scalar to theta-face (j+1/2): 0.5*(f[j] + f[j+1]).  Periodic.
    double interp_to_theta_face(const Field3D& f, int i, int j, int k) const
    {
        const int j_next = (j + 1) % NTH_;
        return 0.5 * (static_cast<double>(f[i][j][k])
                    + static_cast<double>(f[i][j_next][k]));
    }

    /// Scalar to z-face (k+1/2): 0.5*(f[k] + f[k+1]).
    double interp_to_z_face(const Field3D& f, int i, int j, int k) const
    {
        return 0.5 * (static_cast<double>(f[i][j][k])
                    + static_cast<double>(f[i][j][k + 1]));
    }

    // -----------------------------------------------------------------
    // Interpolation: face -> cell center (arithmetic mean)
    // -----------------------------------------------------------------

    /// r-face quantity to cell center: 0.5*(f[i] + f[i-1]).
    /// Axis (i=0): 0.5*f[0] (only the right face exists; left is the axis).
    double interp_from_r_face(const Field3D& f, int i, int j, int k) const
    {
        if (i == 0)
        {
            return 0.5 * static_cast<double>(f[0][j][k]);
        }
        return 0.5 * (static_cast<double>(f[i][j][k])
                    + static_cast<double>(f[i - 1][j][k]));
    }

    /// theta-face quantity to cell center: 0.5*(f[j] + f[j-1]).  Periodic.
    double interp_from_theta_face(const Field3D& f, int i, int j, int k) const
    {
        const int j_prev = (j - 1 + NTH_) % NTH_;
        return 0.5 * (static_cast<double>(f[i][j][k])
                    + static_cast<double>(f[i][j_prev][k]));
    }

    /// z-face quantity to cell center: 0.5*(f[k] + f[k-1]).
    /// Bottom (k=0): 0.5*f[0] (surface face; below-surface is rigid w=0).
    double interp_from_z_face(const Field3D& f, int i, int j, int k) const
    {
        if (k == 0)
        {
            return 0.5 * static_cast<double>(f[i][j][0]);
        }
        return 0.5 * (static_cast<double>(f[i][j][k])
                    + static_cast<double>(f[i][j][k - 1]));
    }

private:
    const GridGeometry& geo_;
    int NTH_;
};
