/**
 * @file eddy_viscosity.cpp
 * @brief Implementation for the turbulence module.
 *
 * Provides executable logic for the turbulence runtime path,
 * including initialization, stepping, and diagnostics helpers.
 * This file is part of the src/turbulence subsystem.
 */

#include "eddy_viscosity.hpp"
#include "util/grid_metric_utils.hpp"
#include <cmath>
#include <algorithm>
#ifdef _OPENMP
#include <omp.h>
#endif


namespace eddy_viscosity
{

namespace
{
/**
 * @brief Boundary-aware finite difference in a horizontal dimension.
 *
 * Uses central differencing when not at a boundary, otherwise falls
 * back to one-sided (forward or backward) differencing.
 *
 * @param f_minus Value at index-1 (only valid when idx > 0).
 * @param f_center Value at current index.
 * @param f_plus Value at index+1 (only valid when idx < n-1).
 * @param dx Grid spacing.
 * @param idx Current index in this dimension.
 * @param n Total extent in this dimension.
 */
double horiz_diff(double f_minus, double f_center, double f_plus,
                  double dx, int idx, int n)
{
    if (idx > 0 && idx < n - 1)
    {
        return (f_plus - f_minus) / (2.0 * dx);
    }
    if (idx == 0)
    {
        return (f_plus - f_center) / dx;
    }
    return (f_center - f_minus) / dx;
}

/**
 * @brief Boundary-aware vertical finite difference.
 *
 * Uses centered differencing over the actual vertical span in the
 * interior, one-sided at top/bottom.
 */
double vert_diff(double f_minus, double f_center, double f_plus,
                 double dz_one_sided, double dz_centered,
                 int k, int nz)
{
    if (k > 0 && k < nz - 1)
    {
        return (f_plus - f_minus) / dz_centered;
    }
    if (k == 0)
    {
        return (f_plus - f_center) / dz_one_sided;
    }
    return (f_center - f_minus) / dz_one_sided;
}
} // anonymous namespace

/**
 * @brief Computes the strain rate.
 */
StrainRate compute_strain_rate_3d(
    const TurbulenceStateView& state,
    const GridMetrics& grid,
    int i, int j, int k
)
{
    StrainRate S;

    const double dx = std::max(grid_metric::local_dx(grid, i, j, k), 1.0);
    const double dy = std::max(grid_metric::local_dy(grid, i, j, k), 1.0);
    const double dz_k = std::max(grid_metric::local_dz(grid, i, j, k, state.NZ), 1.0);
    const double dz_centered = std::max(grid_metric::centered_dz_span(grid, i, j, k, state.NZ), 1.0);

    // Horizontal derivatives (radial direction, index i)
    const auto& u = *state.u;
    const auto& v = *state.v;
    const auto& w = *state.w;

    const double dudx = horiz_diff(u[std::max(i-1,0)][j][k], u[i][j][k], u[std::min(i+1,state.NR-1)][j][k], dx, i, state.NR);
    const double dvdx = horiz_diff(v[std::max(i-1,0)][j][k], v[i][j][k], v[std::min(i+1,state.NR-1)][j][k], dx, i, state.NR);
    const double dwdx = horiz_diff(w[std::max(i-1,0)][j][k], w[i][j][k], w[std::min(i+1,state.NR-1)][j][k], dx, i, state.NR);

    // Horizontal derivatives (azimuthal direction, index j)
    const double dudy = horiz_diff(u[i][std::max(j-1,0)][k], u[i][j][k], u[i][std::min(j+1,state.NTH-1)][k], dy, j, state.NTH);
    const double dvdy = horiz_diff(v[i][std::max(j-1,0)][k], v[i][j][k], v[i][std::min(j+1,state.NTH-1)][k], dy, j, state.NTH);
    const double dwdy = horiz_diff(w[i][std::max(j-1,0)][k], w[i][j][k], w[i][std::min(j+1,state.NTH-1)][k], dy, j, state.NTH);

    // Vertical derivatives (index k)
    const double dudz = vert_diff(u[i][j][std::max(k-1,0)], u[i][j][k], u[i][j][std::min(k+1,state.NZ-1)], dz_k, dz_centered, k, state.NZ);
    const double dvdz = vert_diff(v[i][j][std::max(k-1,0)], v[i][j][k], v[i][j][std::min(k+1,state.NZ-1)], dz_k, dz_centered, k, state.NZ);
    const double dwdz = vert_diff(w[i][j][std::max(k-1,0)], w[i][j][k], w[i][j][std::min(k+1,state.NZ-1)], dz_k, dz_centered, k, state.NZ);

    S.S11 = dudx;
    S.S12 = 0.5 * (dudy + dvdx);
    S.S13 = 0.5 * (dudz + dwdx);

    S.S21 = S.S12;
    S.S22 = dvdy;
    S.S23 = 0.5 * (dvdz + dwdy);

    S.S31 = S.S13;
    S.S32 = S.S23;
    S.S33 = dwdz;

    S.magnitude = std::sqrt(2.0 * (
        S.S11*S.S11 + S.S22*S.S22 + S.S33*S.S33 +
        2.0*(S.S12*S.S12 + S.S13*S.S13 + S.S23*S.S23)
    ));

    return S;
}

/**
 * @brief Computes the filter width.
 */
double compute_filter_width(
    const TurbulenceConfig& cfg,
    const GridMetrics& grid,
    int i, int j, int k
) 
{
    int nz_hint = static_cast<int>(grid.dz.size());
    if (nz_hint <= 0 && grid.terrain_metrics != nullptr)
    {
        nz_hint = grid.terrain_metrics->z.size_z();
    }
    if (nz_hint <= 0)
    {
        nz_hint = std::max(k + 1, 1);
    }

    const double dx_local = std::max(grid_metric::local_dx(grid, i, j, k), 1.0);
    const double dy_local = std::max(grid_metric::local_dy(grid, i, j, k), 1.0);
    const double dz_local = std::max(grid_metric::local_dz(grid, i, j, k, nz_hint), 1.0);

    if (cfg.filter_width == "dx") 
    {
        return std::sqrt(dx_local * dy_local);
    }
    
    else if (cfg.filter_width == "cubic_root") 
    {
        return std::cbrt(dx_local * dy_local * dz_local);
    }
    else 
    {
        return std::cbrt(dx_local * dy_local * dz_local);
    }
}

/**
 * @brief Computes the Smagorinsky viscosity.
 */

double compute_smagorinsky_viscosity(
    double Cs,
    double Delta,
    double strain_mag,
    double stability_factor
) 
{
    double nu_t = Cs * Cs * Delta * Delta * strain_mag;

    nu_t *= stability_factor;

    return std::max(nu_t, 0.0);
}

/**
 * @brief Computes the eddy diffusivities.
 */
void compute_eddy_diffusivities(
    double nu_t,
    double Pr_t,
    double Sc_t,
    double& K_theta,
    double& K_q,
    double& K_tke
) 
{
    const double Pr_safe = std::max(std::abs(Pr_t), 1.0e-6);
    const double Sc_safe = std::max(std::abs(Sc_t), 1.0e-6);
    K_theta = nu_t / Pr_safe;
    K_q = nu_t / Sc_safe;
    K_tke = nu_t / Pr_safe;
}

/**
 * @brief Computes the scalar diffusion tendency.
 */
double compute_scalar_diffusion_tendency(
    const TurbulenceStateView& state,
    const GridMetrics& grid,
    const Field3D& K_field,
    const Field3D& phi,
    int i, int j, int k,
    int var_index
) 
{

    return 0.0;
}

/**
 * @brief Computes the momentum diffusion tendencies.
 */
void compute_momentum_diffusion_tendencies(
    const TurbulenceConfig& cfg,
    const TurbulenceStateView& state,
    const GridMetrics& grid,
    const Field3D& nu_t,
    Field3D& dudt_sgs,
    Field3D& dvdt_sgs,
    Field3D& dwdt_sgs
) 
{
    const bool use_horizontal = cfg.mode != "vertical_only";
    const bool use_vertical = cfg.mode != "horizontal_only";

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < state.NR; ++i)
    {
        for (int j = 0; j < state.NTH; ++j)
        {
            int j_prev = (j - 1 + state.NTH) % state.NTH;
            int j_next = (j + 1) % state.NTH;

            for (int k = 0; k < state.NZ; ++k)
            {
                double nu_local = static_cast<double>(nu_t[i][j][k]);
                if (nu_local <= 0.0)
                {
                    dudt_sgs[i][j][k] = 0.0f;
                    dvdt_sgs[i][j][k] = 0.0f;
                    dwdt_sgs[i][j][k] = 0.0f;
                    continue;
                }

                const double dx = std::max(grid_metric::local_dx(grid, i, j, k), 1.0);
                const double dy = std::max(grid_metric::local_dy(grid, i, j, k), 1.0);
                const double dz_k = std::max(grid_metric::local_dz(grid, i, j, k, state.NZ), 1.0);

                const bool needs_radial_neighbors = use_horizontal;
                const bool needs_vertical_neighbors = use_vertical;
                if ((needs_radial_neighbors && (i == 0 || i == state.NR - 1)) ||
                    (needs_vertical_neighbors && (k == 0 || k == state.NZ - 1)))
                {
                    dudt_sgs[i][j][k] = 0.0f;
                    dvdt_sgs[i][j][k] = 0.0f;
                    dwdt_sgs[i][j][k] = 0.0f;
                    continue;
                }

                auto laplacian = [&](const Field3D& field) {
                    const double center = static_cast<double>(field[i][j][k]);
                    const double d2_dx2 = use_horizontal
                        ? (static_cast<double>(field[i + 1][j][k]) - 2.0 * center +
                           static_cast<double>(field[i - 1][j][k])) / (dx * dx)
                        : 0.0;
                    const double d2_dy2 = use_horizontal
                        ? (static_cast<double>(field[i][j_next][k]) - 2.0 * center +
                           static_cast<double>(field[i][j_prev][k])) / (dy * dy)
                        : 0.0;
                    const double d2_dz2 = use_vertical
                        ? (static_cast<double>(field[i][j][k + 1]) - 2.0 * center +
                           static_cast<double>(field[i][j][k - 1])) / (dz_k * dz_k)
                        : 0.0;
                    return d2_dx2 + d2_dy2 + d2_dz2;
                };

                dudt_sgs[i][j][k] = static_cast<float>(nu_local * laplacian(*state.u));
                dvdt_sgs[i][j][k] = static_cast<float>(nu_local * laplacian(*state.v));
                dwdt_sgs[i][j][k] = static_cast<float>(nu_local * laplacian(*state.w));
            }
        }
    }
}

/**
 * @brief Computes the scalar diffusion tendencies.
 */
void compute_scalar_diffusion_tendencies(
    const TurbulenceConfig& cfg,
    const TurbulenceStateView& state,
    const GridMetrics& grid,
    const Field3D& K_field,
    const Field3D& phi,
    Field3D& dphi_dt_sgs
) 
{
    const bool use_horizontal = cfg.mode != "vertical_only";
    const bool use_vertical = cfg.mode != "horizontal_only";

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < state.NR; ++i)
    {
        for (int j = 0; j < state.NTH; ++j)
        {
            int j_prev = (j - 1 + state.NTH) % state.NTH;
            int j_next = (j + 1) % state.NTH;

            for (int k = 0; k < state.NZ; ++k)
            {
                double K_local = static_cast<double>(K_field[i][j][k]);
                if (K_local <= 0.0)
                {
                    dphi_dt_sgs[i][j][k] = 0.0f;
                    continue;
                }

                const double dx = std::max(grid_metric::local_dx(grid, i, j, k), 1.0);
                const double dy = std::max(grid_metric::local_dy(grid, i, j, k), 1.0);
                const double dz_k = std::max(grid_metric::local_dz(grid, i, j, k, state.NZ), 1.0);

                const bool needs_radial_neighbors = use_horizontal;
                const bool needs_vertical_neighbors = use_vertical;
                if ((needs_radial_neighbors && (i == 0 || i == state.NR - 1)) ||
                    (needs_vertical_neighbors && (k == 0 || k == state.NZ - 1)))
                {
                    dphi_dt_sgs[i][j][k] = 0.0f;
                    continue;
                }

                const double center = static_cast<double>(phi[i][j][k]);
                const double d2_dx2 = use_horizontal
                    ? (static_cast<double>(phi[i + 1][j][k]) - 2.0 * center +
                       static_cast<double>(phi[i - 1][j][k])) / (dx * dx)
                    : 0.0;
                const double d2_dy2 = use_horizontal
                    ? (static_cast<double>(phi[i][j_next][k]) - 2.0 * center +
                       static_cast<double>(phi[i][j_prev][k])) / (dy * dy)
                    : 0.0;
                const double d2_dz2 = use_vertical
                    ? (static_cast<double>(phi[i][j][k + 1]) - 2.0 * center +
                       static_cast<double>(phi[i][j][k - 1])) / (dz_k * dz_k)
                    : 0.0;

                dphi_dt_sgs[i][j][k] = static_cast<float>(K_local * (d2_dx2 + d2_dy2 + d2_dz2));
            }
        }
    }
}

/**
 * @brief Applies Richardson-number stability correction to SGS coefficients.
 */
double stability_correction_ri(double Ri, double Ri_crit) {
    if (Ri > 0.0) {
        return std::max(0.1, 1.0 - Ri / Ri_crit);
    } else {
        return 1.0 - 0.5 * Ri;
    }
}

/**
 * @brief Computes the TKE mixing length.
 */
double compute_tke_mixing_length(
    double Delta,
    double e,
    double N,
    double c_l
) 
{
    const double l_grid = std::max(Delta, 1.0);
    const double e_safe = std::max(e, 0.0);
    double l_stability = (N > 1e-6) ? c_l * std::sqrt(e_safe) / N : l_grid;

    return std::max(1.0e-3, std::min(l_grid, l_stability));
}

/**
 * @brief Computes the Brunt-Väisälä frequency.
 */
double compute_brunt_vaisala_frequency(
    const TurbulenceStateView& state,
    const GridMetrics& grid,
    int i, int j, int k
)
{
    if (!state.theta || k < 0 || k >= state.NZ - 1)
    {
        return 0.0;
    }

    const double theta_k = static_cast<double>((*state.theta)[i][j][k]);
    const double theta_kp1 = static_cast<double>((*state.theta)[i][j][k + 1]);
    if (!std::isfinite(theta_k) || !std::isfinite(theta_kp1) || theta_k <= 1.0e-6)
    {
        return 0.0;
    }

    const int nz_hint = std::max(state.NZ, 1);
    const double dz_local = std::max(grid_metric::local_dz(grid, i, j, k, nz_hint), 1.0);
    const double dtheta_dz = (theta_kp1 - theta_k) / dz_local;

    const double N2 = (turbulence_constants::g / theta_k) * dtheta_dz;
    return (N2 > 0.0) ? std::sqrt(N2) : 0.0;
}

double compute_brunt_vaisala_frequency(
    const TurbulenceStateView& state,
    int i, int j, int k
)
{
    GridMetrics legacy_grid;
    legacy_grid.dx = 1000.0;
    legacy_grid.dy = 1000.0;
    legacy_grid.dz.assign(static_cast<std::size_t>(std::max(state.NZ, 1)), 100.0);
    return compute_brunt_vaisala_frequency(state, legacy_grid, i, j, k);
}

/**
 * @brief Applies the positivity limits.
 */
void apply_positivity_limits(
    Field3D& field,
    double min_value,
    double max_value
) 
{
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < field.size_r(); ++i) 
    {
        for (int j = 0; j < field.size_th(); ++j) 
        {
            for (int k = 0; k < field.size_z(); ++k) 
            {
                const float old_value = static_cast<float>(field[i][j][k]);
                float new_value = old_value;
                if (!std::isfinite(static_cast<double>(new_value)))
                {
                    new_value = static_cast<float>(min_value);
                }
                new_value = std::max(static_cast<float>(min_value),
                    std::min(static_cast<float>(max_value), new_value));
                field[i][j][k] = new_value;
            }
        }
    }
}

/**
 * @brief Initializes the 3D field.
 */
void initialize_3d_field(
    std::vector<std::vector<std::vector<float>>>& field,
    int NR, int NTH, int NZ,
    float value
) 
{
    field.assign(NR, std::vector<std::vector<float>>(
                NTH, std::vector<float>(NZ, value)));
}

}
