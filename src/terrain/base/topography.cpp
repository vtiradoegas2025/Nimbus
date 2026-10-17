/**
 * @file topography.cpp
 * @brief Implementation for the terrain module.
 *
 * Provides executable logic for the terrain runtime path,
 * including initialization, stepping, and diagnostics helpers.
 * This file is part of the src/terrain subsystem.
 */

#include "topography.hpp"
#include "core/infra/grid_coordinates.hpp"
#include <cmath>
#include <algorithm>
#include <limits>
#include <stdexcept>


namespace tmv::topography
{

namespace
{
double require_positive(double value, const char* label)
{
    if (!std::isfinite(value) || value <= terrain_constants::epsilon)
    {
        throw std::invalid_argument(std::string(label) + " must be positive and finite");
    }
    return value;
}

double require_non_negative(double value, const char* label)
{
    if (!std::isfinite(value) || value < 0.0)
    {
        throw std::invalid_argument(std::string(label) + " must be non-negative and finite");
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
    const double a = require_positive(params.a, "terrain.bell.a");
    const double h0 = require_non_negative(params.h0, "terrain.bell.h0");

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
} // namespace tmv::topography

/**
 * @brief Evaluates the Schär topography.
 */
ScharResult eval_schar(double x, const TerrainConfig::ScharParams& params) 
{
    ScharResult result{0.0, 0.0};
    const double a = require_positive(params.a, "terrain.schar.a");
    const double ell = require_positive(params.ell, "terrain.schar.ell");
    const double H = require_non_negative(params.H, "terrain.schar.H");

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
    require_positive(config_.ztop, "terrain.ztop");
    require_non_negative(config_.smoothing_decay, "terrain.smoothing_decay");
    if (config_.coord_id != "btf" && config_.coord_id != "smoothed" &&
        config_.coord_id != "cartesian")
    {
        throw std::invalid_argument("Unknown terrain coordinate '" + config_.coord_id + "'");
    }
    for (const auto& row : topography_.h)
    {
        for (double height : row)
        {
            if (!std::isfinite(height) || height < 0.0)
            {
                throw std::invalid_argument(
                    "Terrain height must be non-negative and finite");
            }
            if (config_.coord_id != "cartesian" && height >= config_.ztop)
            {
                throw std::invalid_argument(
                    "Terrain height must remain below terrain.ztop");
            }
        }
    }
}

double TerrainFollowingCoordinate::zeta_to_z(double zeta, int i, int j) const 
{
    double h = topography_.h[i][j];
    const double ztop = require_positive(config_.ztop, "terrain.ztop");

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
    const double ztop = require_positive(config_.ztop, "terrain.ztop");

    if (config_.coord_id == "btf") {
        if (std::abs(ztop - h) > terrain_constants::epsilon) 
        {
            return ztop * (z - h) / (ztop - h);
        }
        throw std::domain_error("Terrain coordinate is singular because terrain height reaches ztop");
    } 
    else if (config_.coord_id == "smoothed") 
    {
        // The smoothed transform is nonlinear in zeta. Invert it with a
        // bracketed Newton iteration on the physical domain [0, ztop], where
        // z(zeta) is monotone for a valid coordinate (J > 0).
        if (z <= h)
        {
            return 0.0;
        }
        if (z >= ztop)
        {
            return ztop;
        }

        double lo = 0.0;
        double hi = ztop;
        double zeta = std::clamp(
            ztop * (z - h) / std::max(ztop - h, terrain_constants::epsilon),
            lo, hi);
        const double decay_rate =
            require_non_negative(config_.smoothing_decay, "terrain.smoothing_decay");

        for (int iteration = 0; iteration < 40; ++iteration)
        {
            const double decay = std::exp(-decay_rate * zeta / ztop);
            const double sigma = 1.0 - zeta / ztop;
            const double mapped = zeta + h * decay * sigma;
            const double residual = mapped - z;
            if (std::abs(residual) <= 1.0e-11 * std::max(ztop, 1.0))
            {
                break;
            }

            if (residual > 0.0) hi = zeta;
            else lo = zeta;

            const double jacobian =
                1.0 - (h * decay / ztop) * (1.0 + decay_rate * sigma);
            const double newton = (jacobian > terrain_constants::epsilon)
                ? zeta - residual / jacobian
                : 0.5 * (lo + hi);
            zeta = (newton > lo && newton < hi)
                ? newton
                : 0.5 * (lo + hi);
        }
        return zeta;
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
    const double ztop = require_positive(config_.ztop, "terrain.ztop");

    if (config_.coord_id == "btf")
    {
        z_out = zeta + h * (1.0 - zeta/ztop);

        J = 1.0 - h/ztop;

        double zeta_factor = 1.0 - zeta/ztop;

        if (J > terrain_constants::epsilon)
        {
            mx = -zeta_factor * hx / J;
            my = -zeta_factor * hy / J;
        }
        else
        {
            throw std::domain_error("Terrain-following coordinate folded in BTF metrics");
        }
    }
    else if (config_.coord_id == "smoothed") 
    {
        const double decay_rate =
            require_non_negative(config_.smoothing_decay, "terrain.smoothing_decay");
        const double decay = std::exp(-decay_rate * zeta / ztop);
        const double h_eff = h * decay;
        const double hx_eff = hx * decay;
        const double hy_eff = hy * decay;

        z_out = zeta + h_eff * (1.0 - zeta/ztop);

        const double zeta_factor = 1.0 - zeta/ztop;
        J = 1.0 - (h_eff / ztop) *
            (1.0 + decay_rate * zeta_factor);
        if (J > terrain_constants::epsilon)
        {
            mx = -zeta_factor * hx_eff / J;
            my = -zeta_factor * hy_eff / J;
        } 
        else
        {
            throw std::domain_error("Terrain-following coordinate folded in smoothed metrics");
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
    const double ztop = require_positive(config_.ztop, "terrain.ztop");
    const double decay_rate =
        require_non_negative(config_.smoothing_decay, "terrain.smoothing_decay");
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
    // Conservative face geometry is intentionally lazy.  Ordinary flat runs
    // use the existing Cartesian kernels and must not carry five additional
    // double-precision 3-D arrays merely because the terrain module exists.
    metrics.z_face = TerrainField3D{};
    metrics.Jx_face = TerrainField3D{};
    metrics.Jy_face = TerrainField3D{};
    metrics.Gx_face = TerrainField3D{};
    metrics.Gy_face = TerrainField3D{};
    metrics.dx = 0.0;
    metrics.dy = 0.0;
    metrics.dzeta = 0.0;
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
    const double ztop_safe = require_positive(ztop, "terrain.ztop");
    const double spacing = ztop_safe / static_cast<double>(NZ);
    for (int k = 0; k < NZ; ++k)
    {
        zeta[k] = grid_coordinates::scalar_center(k, spacing);
    }
    return zeta;
}

std::vector<double> build_zeta_interfaces(int NZ, double ztop)
{
    if (NZ <= 0)
    {
        return {};
    }
    const double ztop_safe = require_positive(ztop, "terrain.ztop");
    const double spacing = ztop_safe / static_cast<double>(NZ);
    std::vector<double> interfaces(static_cast<std::size_t>(NZ + 1), 0.0);
    for (int k = 0; k <= NZ; ++k)
    {
        interfaces[static_cast<std::size_t>(k)] =
            static_cast<double>(k) * spacing;
    }
    return interfaces;
}

void build_conservative_face_metrics(const TerrainConfig& cfg,
                                     const Topography2D& topo,
                                     const std::vector<double>& zeta_interfaces,
                                     double dx,
                                     double dy,
                                     TerrainMetrics3D& metrics)
{
    const int nr = metrics.J.size_r();
    const int nth = metrics.J.size_th();
    const int nz = metrics.J.size_z();
    require_positive(dx, "terrain computational dx");
    require_positive(dy, "terrain computational dy");
    if (nr <= 0 || nth <= 0 || nz <= 0
        || zeta_interfaces.size() != static_cast<std::size_t>(nz + 1))
    {
        throw std::invalid_argument(
            "terrain conservative metrics require a non-empty grid and NZ+1 interfaces");
    }
    if (topo.h.size() != static_cast<std::size_t>(nr))
    {
        throw std::invalid_argument("terrain height field does not match metric grid");
    }
    for (const auto& row : topo.h)
    {
        if (row.size() != static_cast<std::size_t>(nth))
            throw std::invalid_argument("terrain height field is not rectangular");
    }

    const double dzeta = zeta_interfaces[1] - zeta_interfaces[0];
    require_positive(dzeta, "terrain computational dzeta");
    for (int k = 0; k <= nz; ++k)
    {
        const double value = zeta_interfaces[static_cast<std::size_t>(k)];
        if (!std::isfinite(value)
            || (k > 0
                && std::abs((value - zeta_interfaces[static_cast<std::size_t>(k - 1)])
                            - dzeta)
                    > 1.0e-12 * std::max(1.0, std::abs(dzeta))))
        {
            throw std::invalid_argument(
                "terrain computational interfaces must be finite and uniformly spaced");
        }
    }

    metrics.z_face.resize(nr, nth, nz + 1, 0.0);
    metrics.Jx_face.resize(nr + 1, nth, nz, 0.0);
    metrics.Jy_face.resize(nr, nth + 1, nz, 0.0);
    metrics.Gx_face.resize(nr, nth, nz + 1, 0.0);
    metrics.Gy_face.resize(nr, nth, nz + 1, 0.0);
    metrics.dx = dx;
    metrics.dy = dy;
    metrics.dzeta = dzeta;

    TerrainFollowingCoordinate coordinate(cfg, topo);
    for (int i = 0; i < nr; ++i)
        for (int j = 0; j < nth; ++j)
        {
            for (int k = 0; k <= nz; ++k)
                metrics.z_face(i, j, k) = coordinate.zeta_to_z(
                    zeta_interfaces[static_cast<std::size_t>(k)], i, j);
            for (int k = 0; k < nz; ++k)
            {
                const double zeta_center = 0.5 * (
                    zeta_interfaces[static_cast<std::size_t>(k)]
                    + zeta_interfaces[static_cast<std::size_t>(k + 1)]);
                double point_jacobian = 0.0;
                coordinate.compute_metrics(
                    zeta_center, i, j,
                    metrics.z(i, j, k), point_jacobian,
                    metrics.mx(i, j, k), metrics.my(i, j, k));
                metrics.zeta(i, j, k) = zeta_center;
            }
        }

    // The same interpolated face-height maps feed both side Jacobians and
    // zeta-face slopes.  This makes the two mixed differences commute, which
    // is the discrete geometric conservation law needed for free-stream
    // preservation.  Domain edges use a closed, zero-normal-gradient metric
    // closure by copying the nearest column.
    TerrainField3D x_face_height;
    TerrainField3D y_face_height;
    x_face_height.resize(nr + 1, nth, nz + 1, 0.0);
    y_face_height.resize(nr, nth + 1, nz + 1, 0.0);
    for (int i = 0; i <= nr; ++i)
    {
        const int left = (i == 0) ? 0 : i - 1;
        const int right = (i == nr) ? nr - 1 : i;
        const double right_weight = (i == 0 || i == nr) ? 0.0 : 0.5;
        for (int j = 0; j < nth; ++j)
            for (int k = 0; k <= nz; ++k)
                x_face_height(i, j, k) =
                    (1.0 - right_weight) * metrics.z_face(left, j, k)
                    + right_weight * metrics.z_face(right, j, k);
    }
    for (int i = 0; i < nr; ++i)
    {
        for (int j = 0; j <= nth; ++j)
        {
            const int south = (j == 0) ? 0 : j - 1;
            const int north = (j == nth) ? nth - 1 : j;
            const double north_weight = (j == 0 || j == nth) ? 0.0 : 0.5;
            for (int k = 0; k <= nz; ++k)
                y_face_height(i, j, k) =
                    (1.0 - north_weight) * metrics.z_face(i, south, k)
                    + north_weight * metrics.z_face(i, north, k);
        }
    }

    for (int i = 0; i < nr; ++i)
    {
        for (int j = 0; j < nth; ++j)
        {
            for (int k = 0; k < nz; ++k)
            {
                const double jacobian =
                    (metrics.z_face(i, j, k + 1) - metrics.z_face(i, j, k))
                    / dzeta;
                if (!std::isfinite(jacobian)
                    || jacobian <= terrain_constants::epsilon)
                    throw std::domain_error(
                        "terrain conservative cell geometry is folded");
                metrics.J(i, j, k) = jacobian;
            }
            for (int k = 0; k <= nz; ++k)
            {
                metrics.Gx_face(i, j, k) =
                    -(x_face_height(i + 1, j, k) - x_face_height(i, j, k)) / dx;
                metrics.Gy_face(i, j, k) =
                    -(y_face_height(i, j + 1, k) - y_face_height(i, j, k)) / dy;
            }
        }
    }
    for (int i = 0; i <= nr; ++i)
        for (int j = 0; j < nth; ++j)
            for (int k = 0; k < nz; ++k)
            {
                const double jacobian =
                    (x_face_height(i, j, k + 1) - x_face_height(i, j, k))
                    / dzeta;
                if (!std::isfinite(jacobian)
                    || jacobian <= terrain_constants::epsilon)
                    throw std::domain_error("terrain conservative x face is folded");
                metrics.Jx_face(i, j, k) = jacobian;
            }
    for (int i = 0; i < nr; ++i)
        for (int j = 0; j <= nth; ++j)
            for (int k = 0; k < nz; ++k)
            {
                const double jacobian =
                    (y_face_height(i, j, k + 1) - y_face_height(i, j, k))
                    / dzeta;
                if (!std::isfinite(jacobian)
                    || jacobian <= terrain_constants::epsilon)
                    throw std::domain_error("terrain conservative y face is folded");
                metrics.Jy_face(i, j, k) = jacobian;
            }
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

    double local_min_j = diag.min_jacobian;
    double local_max_j = diag.max_jacobian;

    #pragma omp parallel for collapse(3) reduction(min:local_min_j) reduction(max:local_max_j)
    for (int i = 0; i < NR; ++i)
    {
        for (int j = 0; j < NTH; ++j)
        {
            for (int k = 0; k < NZ; ++k)
            {
                double J = metrics.J(i, j, k);
                if (std::isfinite(J))
                {
                    local_min_j = std::min(local_min_j, J);
                    local_max_j = std::max(local_max_j, J);
                }

                if (!std::isfinite(J) || J <= terrain_constants::epsilon)
                {
                    #pragma omp critical
                    {
                        diag.coordinate_folding = true;
                        diag.warnings.push_back("Coordinate folding detected at (i,j,k)=(" +
                                              std::to_string(i) + "," + std::to_string(j) + "," +
                                              std::to_string(k) + "), J=" + std::to_string(J));
                    }
                }
            }
        }
    }

    diag.min_jacobian = local_min_j;
    diag.max_jacobian = local_max_j;

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

} // namespace tmv::topography
