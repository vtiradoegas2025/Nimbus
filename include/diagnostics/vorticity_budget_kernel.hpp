#pragma once

#include <algorithm>
#include <cmath>

#include "core/field/field3d.hpp"
#include "diagnostics/physical_derivative_kernel.hpp"
#include "dynamics/dynamics_base.hpp"  // dynamics_constants::eps
#include "terrain/conservative_transport.hpp"

/**
 * @file vorticity_budget_kernel.hpp
 * @brief Pointwise Cartesian C-grid vertical-vorticity budgets.
 *
 * These kernels evaluate the online Lagrangian budget at its interpolation
 * points. The mapped kernel uses physical gradients and interface depth.
 *
 * The terms depend only on the prognostic fields u, v, w, rho, p (all current
 * every step), independently of output cadence. Published gridded diagnostics
 * use the resolved-diagnostics physical-domain stencils.
 */

namespace tmv
{

/// Vorticity vector and vertical-vorticity budget terms at one cell.
/// Vorticity components are 1/s; budget terms are 1/s^2.
struct VorticityBudgetPoint
{
    double omega_x;     ///< dw/dy - dv/dz   (vorticity_r slot)
    double omega_y;     ///< du/dz - dw/dx   (vorticity_theta slot)
    double omega_z;     ///< dv/dx - du/dy   (vertical vorticity)
    double stretching;  ///< omega_z * dw/dz
    double tilting;     ///< omega_x * dw/dx + omega_y * dw/dy
    double baroclinic;  ///< (drho/dx * dp/dy - drho/dy * dp/dx) / rho^2
};

/**
 * @brief Cartesian C-grid vertical-vorticity budget at interior cell (i, j, k).
 *
 * Central differences of face-averaged (-> cell-centre) velocities, with the
 * raw face values used for the one-sided dw/dz. Non-finite intermediates
 * collapse to 0, and the baroclinic term is guarded against rho -> 0, matching
 * the flat field version exactly (so the trajectory sample is consistent with
 * the flat gridded diagnostic).
 *
 * @pre 1 <= i <= NX-2, 1 <= j <= NY-2, 1 <= k <= NZ-2 — the stencil reads the
 *      i+/-1, j+/-1, k+/-1 neighbours. The caller is responsible for clamping.
 */
inline VorticityBudgetPoint cartesian_cgrid_vorticity_budget_at(
    const Field3D& u, const Field3D& v, const Field3D& w,
    const Field3D& rho, const Field3D& p,
    int i, int j, int k,
    double dx, double dy, double dz)
{
    const double inv_2dx = 1.0 / (2.0 * dx);
    const double inv_2dy = 1.0 / (2.0 * dy);
    const double inv_2dz = 1.0 / (2.0 * dz);
    const double inv_dz  = 1.0 / dz;

    // Face -> cell-centre interpolation of each velocity component.
    auto u_c = [&](int ii, int jj, int kk) {
        return 0.5 * (static_cast<double>(u[ii - 1][jj][kk]) + static_cast<double>(u[ii][jj][kk]));
    };
    auto v_c = [&](int ii, int jj, int kk) {
        return 0.5 * (static_cast<double>(v[ii][jj - 1][kk]) + static_cast<double>(v[ii][jj][kk]));
    };
    auto w_c = [&](int ii, int jj, int kk) {
        return 0.5 * (static_cast<double>(w[ii][jj][kk - 1]) + static_cast<double>(w[ii][jj][kk]));
    };

    const double dux_dy = (u_c(i, j + 1, k) - u_c(i, j - 1, k)) * inv_2dy;
    const double dux_dz = (u_c(i, j, k + 1) - u_c(i, j, k - 1)) * inv_2dz;
    const double dv_dx  = (v_c(i + 1, j, k) - v_c(i - 1, j, k)) * inv_2dx;
    const double dv_dz  = (v_c(i, j, k + 1) - v_c(i, j, k - 1)) * inv_2dz;
    const double dw_dx  = (w_c(i + 1, j, k) - w_c(i - 1, j, k)) * inv_2dx;
    const double dw_dy  = (w_c(i, j + 1, k) - w_c(i, j - 1, k)) * inv_2dy;
    const double dw_dz  = (static_cast<double>(w[i][j][k])
                         - static_cast<double>(w[i][j][k - 1])) * inv_dz;

    double omega_x = dw_dy - dv_dz;    // vorticity_r slot
    double omega_y = dux_dz - dw_dx;   // vorticity_theta slot
    double omega_z = dv_dx - dux_dy;   // vorticity_z slot
    if (!std::isfinite(omega_x)) omega_x = 0.0;
    if (!std::isfinite(omega_y)) omega_y = 0.0;
    if (!std::isfinite(omega_z)) omega_z = 0.0;

    double stretch = omega_z * dw_dz;
    if (!std::isfinite(stretch)) stretch = 0.0;

    double tilt = omega_x * dw_dx + omega_y * dw_dy;
    if (!std::isfinite(tilt)) tilt = 0.0;

    const double drho_dx = (static_cast<double>(rho[i + 1][j][k])
                          - static_cast<double>(rho[i - 1][j][k])) * inv_2dx;
    const double drho_dy = (static_cast<double>(rho[i][j + 1][k])
                          - static_cast<double>(rho[i][j - 1][k])) * inv_2dy;
    const double dp_dx   = (static_cast<double>(p[i + 1][j][k])
                          - static_cast<double>(p[i - 1][j][k])) * inv_2dx;
    const double dp_dy   = (static_cast<double>(p[i][j + 1][k])
                          - static_cast<double>(p[i][j - 1][k])) * inv_2dy;
    const double rho_val = static_cast<double>(rho[i][j][k]);
    const double rho_sq  = rho_val * rho_val;
    double baro = 0.0;
    if (rho_sq > dynamics_constants::eps)
        baro = (drho_dx * dp_dy - drho_dy * dp_dx) / rho_sq;
    if (!std::isfinite(baro)) baro = 0.0;

    return VorticityBudgetPoint{omega_x, omega_y, omega_z, stretch, tilt, baro};
}

/**
 * Cartesian collocated-grid counterpart. All five inputs live at scalar
 * centres, so one physical-gradient operator owns curl, stretching, tilting,
 * and baroclinic production without a de-staggering pass.
 */
inline VorticityBudgetPoint cartesian_collocated_vorticity_budget_at(
    const Field3D& u, const Field3D& v, const Field3D& w,
    const Field3D& rho, const Field3D& p,
    int i, int j, int k,
    double dx, double dy, double dz)
{
    const int nr = rho.size_r();
    const int nth = rho.size_th();
    const int nz = rho.size_z();
    const auto value = [](const Field3D& field)
    {
        return [&field](int ii, int jj, int kk)
        {
            return static_cast<double>(field(ii, jj, kk));
        };
    };
    const auto grad_u = diagnostics::physical_gradient_at(
        value(u), i, j, k, nr, nth, nz, dx, dy, dz, nullptr);
    const auto grad_v = diagnostics::physical_gradient_at(
        value(v), i, j, k, nr, nth, nz, dx, dy, dz, nullptr);
    const auto grad_w = diagnostics::physical_gradient_at(
        value(w), i, j, k, nr, nth, nz, dx, dy, dz, nullptr);
    const auto grad_rho = diagnostics::physical_gradient_at(
        value(rho), i, j, k, nr, nth, nz, dx, dy, dz, nullptr);
    const auto grad_p = diagnostics::physical_gradient_at(
        value(p), i, j, k, nr, nth, nz, dx, dy, dz, nullptr);

    double omega_x = grad_w.y - grad_v.z;
    double omega_y = grad_u.z - grad_w.x;
    double omega_z = grad_v.x - grad_u.y;
    if (!std::isfinite(omega_x)) omega_x = 0.0;
    if (!std::isfinite(omega_y)) omega_y = 0.0;
    if (!std::isfinite(omega_z)) omega_z = 0.0;

    double stretch = omega_z * grad_w.z;
    if (!std::isfinite(stretch)) stretch = 0.0;
    double tilt = omega_x * grad_w.x + omega_y * grad_w.y;
    if (!std::isfinite(tilt)) tilt = 0.0;

    const double rho_value = static_cast<double>(rho(i, j, k));
    const double rho_sq = rho_value * rho_value;
    double baro = 0.0;
    if (rho_sq > dynamics_constants::eps)
        baro = (grad_rho.x * grad_p.y - grad_rho.y * grad_p.x) / rho_sq;
    if (!std::isfinite(baro)) baro = 0.0;
    return {omega_x, omega_y, omega_z, stretch, tilt, baro};
}

/**
 * Terrain-following counterpart of `cartesian_cgrid_vorticity_budget_at`.
 * Face velocities are first collocated at scalar centres, then differentiated
 * in physical Cartesian coordinates. Scalar pressure/density gradients use
 * the same mapping, so coordinate slope cannot masquerade as curl or a
 * solenoidal source.
 */
inline VorticityBudgetPoint cartesian_cgrid_mapped_vorticity_budget_at(
    const Field3D& u, const Field3D& v, const Field3D& w,
    const Field3D& rho, const Field3D& p,
    const TerrainMetrics3D& terrain,
    int i, int j, int k,
    double dx, double dy, double dz,
    double frame_u_ms = 0.0,
    double frame_v_ms = 0.0)
{
    const int nr = rho.size_r();
    const int nth = rho.size_th();
    const int nz = rho.size_z();
    const auto u_center = [&](int ii, int jj, int kk)
    {
        return ii == 0
            ? static_cast<double>(u[0][jj][kk])
            : 0.5 * (static_cast<double>(u[ii - 1][jj][kk])
                   + static_cast<double>(u[ii][jj][kk]));
    };
    const auto v_center = [&](int ii, int jj, int kk)
    {
        return jj == 0
            ? static_cast<double>(v[ii][0][kk])
            : 0.5 * (static_cast<double>(v[ii][jj - 1][kk])
                   + static_cast<double>(v[ii][jj][kk]));
    };
    const auto w_center = [&](int ii, int jj, int kk)
    {
        if (kk > 0)
            return 0.5 * (static_cast<double>(w[ii][jj][kk - 1])
                        + static_cast<double>(w[ii][jj][kk]));
        const int physical_i = std::max(ii, 1);
        const int physical_j = std::max(jj, 1);
        const double ground_w = terrain_transport::cgrid_ground_vertical_velocity(
            terrain, u, v, physical_i, physical_j,
            frame_u_ms, frame_v_ms);
        return 0.5 * (ground_w + static_cast<double>(w[ii][jj][0]));
    };
    const auto rho_at = [&](int ii, int jj, int kk)
    {
        return static_cast<double>(rho[ii][jj][kk]);
    };
    const auto p_at = [&](int ii, int jj, int kk)
    {
        return static_cast<double>(p[ii][jj][kk]);
    };

    const auto grad_u = diagnostics::physical_gradient_at(
        u_center, i, j, k, nr, nth, nz, dx, dy, dz, &terrain);
    const auto grad_v = diagnostics::physical_gradient_at(
        v_center, i, j, k, nr, nth, nz, dx, dy, dz, &terrain);
    const auto grad_w = diagnostics::physical_gradient_at(
        w_center, i, j, k, nr, nth, nz, dx, dy, dz, &terrain);
    const auto grad_rho = diagnostics::physical_gradient_at(
        rho_at, i, j, k, nr, nth, nz, dx, dy, dz, &terrain);
    const auto grad_p = diagnostics::physical_gradient_at(
        p_at, i, j, k, nr, nth, nz, dx, dy, dz, &terrain);

    double omega_x = grad_w.y - grad_v.z;
    double omega_y = grad_u.z - grad_w.x;
    double omega_z = grad_v.x - grad_u.y;
    if (!std::isfinite(omega_x)) omega_x = 0.0;
    if (!std::isfinite(omega_y)) omega_y = 0.0;
    if (!std::isfinite(omega_z)) omega_z = 0.0;

    const double layer_depth = terrain.z_face(i, j, k + 1)
                             - terrain.z_face(i, j, k);
    double dw_dz = layer_depth > 0.0
        ? (static_cast<double>(w[i][j][k])
           - static_cast<double>(w[i][j][k - 1])) / layer_depth
        : std::numeric_limits<double>::quiet_NaN();
    if (!std::isfinite(dw_dz)) dw_dz = 0.0;

    double stretch = omega_z * dw_dz;
    if (!std::isfinite(stretch)) stretch = 0.0;
    double tilt = omega_x * grad_w.x + omega_y * grad_w.y;
    if (!std::isfinite(tilt)) tilt = 0.0;

    const double rho_value = rho_at(i, j, k);
    const double rho_sq = rho_value * rho_value;
    double baro = 0.0;
    if (rho_sq > dynamics_constants::eps)
        baro = (grad_rho.x * grad_p.y - grad_rho.y * grad_p.x) / rho_sq;
    if (!std::isfinite(baro)) baro = 0.0;

    return VorticityBudgetPoint{
        omega_x, omega_y, omega_z, stretch, tilt, baro};
}

}  // namespace tmv
