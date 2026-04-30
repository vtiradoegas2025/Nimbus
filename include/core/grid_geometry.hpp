#pragma once

#include <cmath>
#include <vector>

#include "core/coordinate_system.hpp"

/**
 * @file grid_geometry.hpp
 * @brief Precomputed 1D coordinate arrays and spacing constants.
 *
 * Eliminates per-grid-point coordinate recomputation in hot loops.
 * All arrays are computed once at startup from grid dimensions and
 * spacings. Dynamics, advection, and initialization code references
 * these arrays by index instead of computing i*dr, j*dtheta, k*dz,
 * sin(theta), cos(theta), 1/r inline.
 *
 * For cylindrical grids:
 *   r[i]         = radial position of cell center i = i * dr
 *   r_inv[i]     = 1/r[i] (0 at axis; dynamics loops skip i=0)
 *   theta[j]     = azimuthal angle j * dtheta
 *   sin_theta[j] = sin(theta[j])
 *   cos_theta[j] = cos(theta[j])
 *   z[k]         = vertical position k * dz
 *
 * For Cartesian grids:
 *   r[i]         = x-position i * dx (r_inv left empty)
 *   theta[j]     = y-position j * dy (sin/cos left empty)
 *   z[k]         = vertical position k * dz
 *
 * For C-grid staggering (Phase C), face-position arrays are also
 * populated:
 *   r_face[i]     = (i + 0.5) * dr   -- radial face position
 *   r_face_inv[i] = 1 / r_face[i]
 *   z_face[k]     = (k + 0.5) * dz   -- vertical face position
 */
struct GridGeometry
{
    // -- 1D coordinate arrays --

    /// Cell-center positions along the i-axis (NR elements).
    /// Cylindrical: physical radial position.  Cartesian: x-position.
    std::vector<double> r;

    /// Precomputed 1/r[i] for cylindrical (NR elements).
    /// r_inv[0] = 0 (axis; inner loops start at i=1).
    /// Empty for Cartesian.
    std::vector<double> r_inv;

    /// Cell-center positions along the j-axis (NTH elements).
    /// Cylindrical: azimuthal angle (radians).  Cartesian: y-position.
    std::vector<double> theta;

    /// sin(theta[j]) for cylindrical wind projection.  Empty for Cartesian.
    std::vector<double> sin_theta;

    /// cos(theta[j]) for cylindrical wind projection.  Empty for Cartesian.
    std::vector<double> cos_theta;

    /// Cell-center heights (NZ elements).  z[k] = k * dz for both systems.
    std::vector<double> z;

    // -- Face-position arrays (C-grid only; empty for collocated) --

    /// Radial face positions (NR elements).  r_face[i] = (i + 0.5) * dr.
    /// u[i][j][k] lives at this radial position on C-grid.
    std::vector<double> r_face;

    /// Precomputed 1/r_face[i] (NR elements).
    std::vector<double> r_face_inv;

    /// Vertical face positions (NZ elements).  z_face[k] = (k + 0.5) * dz.
    /// w[i][j][k] lives at this vertical position on C-grid.
    std::vector<double> z_face;

    /// True when using Arakawa C-grid staggering.
    bool staggered = false;

    // -- Precomputed spacing reciprocals --

    double inv_dr      = 0.0;   ///< 1 / dr
    double inv_dz      = 0.0;   ///< 1 / dz
    double inv_dtheta  = 0.0;   ///< 1 / dtheta
    double inv_2dr     = 0.0;   ///< 1 / (2 * dr)
    double inv_2dz     = 0.0;   ///< 1 / (2 * dz)
    double inv_2dtheta = 0.0;   ///< 1 / (2 * dtheta)
    double inv_dr2     = 0.0;   ///< 1 / (dr * dr)
    double inv_dz2     = 0.0;   ///< 1 / (dz * dz)

    /// Active coordinate system.
    CoordinateSystem coord = CoordinateSystem::Cylindrical;

    /**
     * @brief Populates all arrays from grid parameters.
     *
     * Call after NR/NTH/NZ/dr/dz/dtheta are finalized (after
     * update_grid_resolution() in the startup path).
     *
     * @param stagger  Grid staggering arrangement. When CGrid, face-position
     *                 arrays (r_face, r_face_inv, z_face) are populated.
     *                 Defaults to Collocated for backward compatibility.
     */
    void initialize(int nr, int nth, int nz,
                    double dr_val, double dz_val, double dtheta_val,
                    CoordinateSystem coordinate_system,
                    StaggerType stagger = StaggerType::Collocated)
    {
        coord = coordinate_system;
        staggered = (stagger == StaggerType::CGrid);

        // Spacing reciprocals
        inv_dr      = 1.0 / dr_val;
        inv_dz      = 1.0 / dz_val;
        inv_dtheta  = 1.0 / dtheta_val;
        inv_2dr     = 0.5 / dr_val;
        inv_2dz     = 0.5 / dz_val;
        inv_2dtheta = 0.5 / dtheta_val;
        inv_dr2     = 1.0 / (dr_val * dr_val);
        inv_dz2     = 1.0 / (dz_val * dz_val);

        // Vertical positions (identical for both coordinate systems)
        z.resize(static_cast<size_t>(nz));
        for (int k = 0; k < nz; ++k)
            z[k] = k * dz_val;

        // i-direction positions
        r.resize(static_cast<size_t>(nr));
        for (int i = 0; i < nr; ++i)
            r[i] = i * dr_val;

        // j-direction positions
        theta.resize(static_cast<size_t>(nth));
        for (int j = 0; j < nth; ++j)
            theta[j] = j * dtheta_val;

        if (coordinate_system == CoordinateSystem::Cylindrical)
        {
            // Precomputed 1/r with safe axis value
            r_inv.resize(static_cast<size_t>(nr));
            r_inv[0] = 0.0;
            for (int i = 1; i < nr; ++i)
                r_inv[i] = 1.0 / (i * dr_val);

            // Trig lookups for cylindrical wind projection
            sin_theta.resize(static_cast<size_t>(nth));
            cos_theta.resize(static_cast<size_t>(nth));
            for (int j = 0; j < nth; ++j)
            {
                const double th = j * dtheta_val;
                sin_theta[j] = std::sin(th);
                cos_theta[j] = std::cos(th);
            }
        }
        else
        {
            r_inv.clear();
            sin_theta.clear();
            cos_theta.clear();
        }

        // Face-position arrays for C-grid staggering
        if (staggered)
        {
            r_face.resize(static_cast<size_t>(nr));
            r_face_inv.resize(static_cast<size_t>(nr));
            for (int i = 0; i < nr; ++i)
            {
                r_face[i] = (i + 0.5) * dr_val;
                r_face_inv[i] = 1.0 / r_face[i];
            }

            z_face.resize(static_cast<size_t>(nz));
            for (int k = 0; k < nz; ++k)
                z_face[k] = (k + 0.5) * dz_val;
        }
        else
        {
            r_face.clear();
            r_face_inv.clear();
            z_face.clear();
        }
    }
};
