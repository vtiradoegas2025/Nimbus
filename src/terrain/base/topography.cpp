/**
 * @file topography.cpp
 * @brief Implementation for the terrain module.
 *
 * Provides executable logic for the terrain runtime path,
 * including initialization, stepping, and diagnostics helpers.
 * This file is part of the src/terrain subsystem.
 */

#include "topography.hpp"
#include <cmath>
#include <algorithm>
#include <limits>


namespace topography 
{

namespace
{
double sanitize_positive(double value, double fallback)
{
    if (!std::isfinite(value) || value <= terrain_constants::epsilon)
    {
        return fallback;
    }
    return value;
}
}

/**
 * @brief Evaluates the bell topography.
 */
BellResult eval_bell(double x, double y, const TerrainConfig::BellParams& params) 
{
    BellResult result{0.0, 0.0, 0.0};
    const double a = sanitize_positive(std::abs(params.a), 5000.0);
    const double h0 = std::isfinite(params.h0) ? params.h0 : 0.0;

    if (params.axisymmetric) 
    {
        double r2 = x*x + y*y;
        double denom = a*a + r2;
        double denom_32 = std::pow(denom, 1.5);

        result.h = h0 * a*a*a / denom_32;

        if (denom_32 > terrain_constants::epsilon) 
        {
            double factor = -3.0 * h0 * a*a*a / std::pow(denom, 2.5);
            result.hx = factor * x;
            result.hy = factor * y;
        }
    } 
    else 
    {
        double xa_ratio = x / a;
        double denom = 1.0 + xa_ratio*xa_ratio;

        result.h = h0 / denom;

        result.hx = -2.0 * h0 * xa_ratio / (a * denom*denom);
        result.hy = 0.0;
    }

    return result;
}

/**
 * @brief Evaluates the Schär topography.
 */
ScharResult eval_schar(double x, const TerrainConfig::ScharParams& params) 
{
    ScharResult result{0.0, 0.0};
    const double a = sanitize_positive(std::abs(params.a), 5000.0);
    const double ell = sanitize_positive(std::abs(params.ell), 4000.0);
    const double H = std::isfinite(params.H) ? params.H : 0.0;

    double xa_ratio = x / a;
    double envelope = std::exp(-xa_ratio*xa_ratio);
    double k = M_PI / ell;
    double cos_term = std::cos(k * x);
    double cos2_term = cos_term * cos_term;

    result.h = H * envelope * cos2_term;

    double d_envelope_dx = -2.0 * xa_ratio / a * envelope;
    double d_cos2_dx = -2.0 * k * cos_term * std::sin(k * x);

    result.hx = H * (d_envelope_dx * cos2_term + envelope * d_cos2_dx);

    return result;
}

TerrainFollowingCoordinate::TerrainFollowingCoordinate(const TerrainConfig& cfg, const Topography2D& topo)
    : config_(cfg), topography_(topo) 
{
}

double TerrainFollowingCoordinate::zeta_to_z(double zeta, int i, int j) const 
{
    double h = topography_.h[i][j];
    double ztop = sanitize_positive(config_.ztop, 20000.0);

    if (config_.coord_id == "btf") 
    {
        return zeta + h * (1.0 - zeta / ztop);
    }
    else if (config_.coord_id == "smoothed") 
    {
        double h_eff = apply_coordinate_smoothing(h, zeta);
        return zeta + h_eff * (1.0 - zeta / ztop);
    } 
    else 
    {
        return zeta;
    }
}

/**
 * @brief Converts the z coordinate to the zeta coordinate.
 */
double TerrainFollowingCoordinate::z_to_zeta(double z, int i, int j) const 
{
    double h = topography_.h[i][j];
    double ztop = sanitize_positive(config_.ztop, 20000.0);

    if (config_.coord_id == "btf") {
        if (std::abs(ztop - h) > terrain_constants::epsilon) 
        {
            return ztop * (z - h) / (ztop - h);
        }
         else 
        {
            return z;
        }
    } 
    else if (config_.coord_id == "smoothed") 
    {
        double h_eff_avg = h * 0.5;
        if (std::abs(ztop - h_eff_avg) > terrain_constants::epsilon) 
        {
            return ztop * (z - h_eff_avg) / (ztop - h_eff_avg);
        } 
        else 
        {
            return z;
        }
    } 
    else 
    {
        return z;
    }
}

void TerrainFollowingCoordinate::compute_metrics(double zeta, int i, int j,
                                                double& z_out, double& J, double& mx, double& my) const 
{
    double h = topography_.h[i][j];
    double hx = topography_.hx.empty() ? 0.0 : topography_.hx[i][j];
    double hy = topography_.hy.empty() ? 0.0 : topography_.hy[i][j];
    double ztop = sanitize_positive(config_.ztop, 20000.0);

    if (config_.coord_id == "btf")
    {
        z_out = zeta + h * (1.0 - zeta/ztop);

        J = 1.0 - h/ztop;

        double zeta_factor = 1.0 - zeta/ztop;

        if (std::abs(J) > terrain_constants::epsilon) 
        {
            mx = -zeta_factor * hx / J;
            my = -zeta_factor * hy / J;
        }
        else 
        {
            mx = 0.0;
            my = 0.0;
        }
    }
    else if (config_.coord_id == "smoothed") 
    {
        double h_eff = apply_coordinate_smoothing(h, zeta);
        double hx_eff = hx;
        double hy_eff = hy;

        z_out = zeta + h_eff * (1.0 - zeta/ztop);

        J = 1.0 - h_eff/ztop;

        double zeta_factor = 1.0 - zeta/ztop;
        if (std::abs(J) > terrain_constants::epsilon) 
        {
            mx = -zeta_factor * hx_eff / J;
            my = -zeta_factor * hy_eff / J;
        } 
        else 
        {
            mx = 0.0;
            my = 0.0;
        }
    } else {
        z_out = zeta;
        J = 1.0;
        mx = 0.0;
        my = 0.0;
    }
}

double TerrainFollowingCoordinate::apply_coordinate_smoothing(double h, double zeta) const 
{
    const double ztop = sanitize_positive(config_.ztop, 20000.0);
    const double decay_rate = std::isfinite(config_.smoothing_decay)
        ? std::max(0.0, config_.smoothing_decay)
        : 0.1;
    double decay_factor = std::exp(-decay_rate * zeta / ztop);
    return h * decay_factor;
}

/**
 * @brief Initializes the topography.
 */

void initialize_topography(Topography2D& topo, int NR, int NTH) 
{
    topo.h.assign(NR, std::vector<double>(NTH, 0.0));
    topo.hx.assign(NR, std::vector<double>(NTH, 0.0));
    topo.hy.assign(NR, std::vector<double>(NTH, 0.0));
}

/**
 * @brief Initializes the metrics.
 */
void initialize_metrics(TerrainMetrics3D& metrics, int NR, int NTH, int NZ) 
{
    metrics.z.resize(NR, NTH, NZ, 0.0);
    metrics.J.resize(NR, NTH, NZ, 1.0);
    metrics.mx.resize(NR, NTH, NZ, 0.0);
    metrics.my.resize(NR, NTH, NZ, 0.0);
    metrics.zeta.resize(NR, NTH, NZ, 0.0);
}

/**
 * @brief Builds the zeta levels.
 */
std::vector<double> build_zeta_levels(int NZ, double ztop) 
{
    if (NZ <= 0)
    {
        return {};
    }

    std::vector<double> zeta(NZ, 0.0);
    const double ztop_safe = sanitize_positive(ztop, 20000.0);

    if (NZ == 1)
    {
        return zeta;
    }

    for (int k = 0; k < NZ; ++k) 
    {
        zeta[k] = k * ztop_safe / (NZ - 1);
    }
    return zeta;
}

/**
 * @brief Checks the coordinate validity.
 */
bool check_coordinate_validity(const TerrainMetrics3D& metrics, TerrainDiagnostics& diag) 
{
    const int NR = metrics.J.size_r();
    const int NTH = metrics.J.size_th();
    const int NZ = metrics.J.size_z();

    diag.min_jacobian = std::numeric_limits<double>::infinity();
    diag.max_jacobian = -std::numeric_limits<double>::infinity();
    diag.coordinate_folding = false;
    diag.warnings.clear();

    for (int i = 0; i < NR; ++i) 
    {
        for (int j = 0; j < NTH; ++j) 
        {
            for (int k = 0; k < NZ; ++k) 
            {
                double J = metrics.J(i, j, k);
                if (std::isfinite(J))
                {
                    diag.min_jacobian = std::min(diag.min_jacobian, J);
                    diag.max_jacobian = std::max(diag.max_jacobian, J);
                }

                if (!std::isfinite(J) || J <= terrain_constants::epsilon) 
                {
                    diag.coordinate_folding = true;
                    diag.warnings.push_back("Coordinate folding detected at (i,j,k)=(" +
                                          std::to_string(i) + "," + std::to_string(j) + "," +
                                          std::to_string(k) + "), J=" + std::to_string(J));
                }
            }
        }
    }

    if (!std::isfinite(diag.min_jacobian))
    {
        diag.min_jacobian = 1.0;
    }
    if (!std::isfinite(diag.max_jacobian))
    {
        diag.max_jacobian = 1.0;
    }

    return !diag.coordinate_folding;
}

}
