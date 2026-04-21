#pragma once

#include <algorithm>
#include <cmath>

#include "core/simulation.hpp"
#include "util/grid_metric_utils.hpp"

/**
 * @file derivative_operators.hpp
 * @brief Coordinate-system-specific centered finite-difference operators.
 *
 * Provides a uniform interface for computing spatial derivatives across
 * Cartesian and cylindrical coordinate systems. Each implementation handles
 * its own spacing model (uniform vs terrain-aware) and boundary topology
 * (non-periodic vs periodic azimuthal).
 *
 * Used by dynamics schemes via composition to eliminate per-scheme
 * derivative operator duplication (Phase B.2).
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
