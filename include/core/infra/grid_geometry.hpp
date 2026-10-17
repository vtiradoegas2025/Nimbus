#pragma once

#include <vector>

#include "core/infra/coordinate_system.hpp"
#include "core/infra/grid_coordinates.hpp"

/**
 * @file grid_geometry.hpp
 * @brief Precomputed 1D coordinate arrays and spacing constants.
 *
 * Eliminates per-grid-point coordinate recomputation in hot loops. All arrays
 * are computed once at startup from grid dimensions and spacings. Dynamics,
 * advection, and initialization code references these arrays by index instead
 * of recomputing i*dx, j*dy, k*dz inline.
 *
 *   r[i]     = scalar x-center (i + 1/2) * dx
 *   theta[j] = scalar y-center (j + 1/2) * dy
 *   z[k]     = scalar z-center (k + 1/2) * dz
 *
 * Face arrays contain N+1 entries at integer multiples of the spacing. For a
 * C-grid field, u[i], v[j], and w[k] correspond to r_faces[i+1],
 * theta_faces[j+1], and z_faces[k+1], respectively.
 *
 * (The r/theta/dr/dtheta names are retained as Cartesian aliases pending a
 * dedicated rename pass.)
 */
struct GridGeometry
{
    // -- 1D coordinate arrays --

    /// Cell-center x-positions along the i-axis (NR elements).
    std::vector<double> r;

    /// Cell-center y-positions along the j-axis (NTH elements).
    std::vector<double> theta;

    /// Cell-center heights (NZ elements). z[k] = (k + 1/2) * dz.
    std::vector<double> z;

    /// Physical x-faces (NR + 1 elements), from 0 through NR * dx.
    std::vector<double> r_faces;

    /// Physical y-faces (NTH + 1 elements), from 0 through NTH * dy.
    std::vector<double> theta_faces;

    /// Physical z-faces (NZ + 1 elements), from 0 through NZ * dz.
    std::vector<double> z_faces;

    /// True when using Arakawa C-grid staggering.
    bool staggered = false;

    // -- Precomputed spacing reciprocals --

    double inv_dr      = 0.0;   ///< 1 / dx
    double inv_dz      = 0.0;   ///< 1 / dz
    double inv_dtheta  = 0.0;   ///< 1 / dy
    double inv_2dr     = 0.0;   ///< 1 / (2 * dx)
    double inv_2dz     = 0.0;   ///< 1 / (2 * dz)
    double inv_2dtheta = 0.0;   ///< 1 / (2 * dy)
    double inv_dr2     = 0.0;   ///< 1 / (dx * dx)
    double inv_dz2     = 0.0;   ///< 1 / (dz * dz)

    /// Active coordinate system (always Cartesian).
    CoordinateSystem coord = CoordinateSystem::Cartesian;

    /**
     * @brief Populates all arrays from grid parameters.
     *
     * Call after NR/NTH/NZ/dr/dz/dtheta are finalized (after
     * update_grid_resolution() in the startup path).
     *
     * @param stagger Grid staggering arrangement; recorded in `staggered`.
     *                Defaults to Collocated for backward compatibility.
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

        // Vertical scalar centers and physical faces.
        z.resize(static_cast<size_t>(nz));
        for (int k = 0; k < nz; ++k)
            z[k] = tmv::grid_coordinates::scalar_center(k, dz_val);
        z_faces.resize(static_cast<size_t>(nz + 1));
        for (int k = 0; k <= nz; ++k)
            z_faces[k] = tmv::grid_coordinates::face(k, dz_val);

        // i-direction (x) scalar centers and physical faces.
        r.resize(static_cast<size_t>(nr));
        for (int i = 0; i < nr; ++i)
            r[i] = tmv::grid_coordinates::scalar_center(i, dr_val);
        r_faces.resize(static_cast<size_t>(nr + 1));
        for (int i = 0; i <= nr; ++i)
            r_faces[i] = tmv::grid_coordinates::face(i, dr_val);

        // j-direction (y) scalar centers and physical faces.
        theta.resize(static_cast<size_t>(nth));
        for (int j = 0; j < nth; ++j)
            theta[j] = tmv::grid_coordinates::scalar_center(j, dtheta_val);
        theta_faces.resize(static_cast<size_t>(nth + 1));
        for (int j = 0; j <= nth; ++j)
            theta_faces[j] = tmv::grid_coordinates::face(j, dtheta_val);
    }
};
