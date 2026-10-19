/**
 * @file device_acoustic_energy_attribution.hpp
 * @brief FP32 operator-time helpers shared by device acoustic attribution.
 */

#pragma once

#include "core/field/field3d.hpp"

#include <cmath>

struct CartesianCGridAcousticEnergyReportV1;
struct SimulationStateView;

namespace tmv::device_acoustic_energy
{

inline float x_face_mass_flux(
    const Field3D& rho, const Field3D& u,
    int i, int j, int k, float frame_u_ms = 0.0f)
{
    return 0.5f * (rho[i][j][k] + rho[i + 1][j][k])
        * (u[i][j][k] - frame_u_ms);
}

inline float y_face_mass_flux(
    const Field3D& rho, const Field3D& v,
    int i, int j, int k, float frame_v_ms = 0.0f)
{
    return 0.5f * (rho[i][j][k] + rho[i][j + 1][k])
        * (v[i][j][k] - frame_v_ms);
}

inline float z_face_mass_flux(
    const Field3D& rho, const Field3D& w,
    int nz, int i, int j, int k)
{
    return k + 1 < nz
        ? 0.5f * (rho[i][j][k] + rho[i][j][k + 1])
            * w[i][j][k]
        : rho[i][j][k] * w[i][j][k];
}

inline double x_face_mass_flux_precise(const Field3D& rho, const Field3D& u,
                                     int i, int j, int k, float frame_u_ms)
{
    return 0.5 * (static_cast<double>(rho[i][j][k]) + rho[i + 1][j][k]) *
        (static_cast<double>(u[i][j][k]) - frame_u_ms);
}

inline double y_face_mass_flux_precise(const Field3D& rho, const Field3D& v,
                                     int i, int j, int k, float frame_v_ms)
{
    return 0.5 * (static_cast<double>(rho[i][j][k]) + rho[i][j + 1][k]) *
        (static_cast<double>(v[i][j][k]) - frame_v_ms);
}

inline double z_face_mass_flux_precise(const Field3D& rho, const Field3D& w,
                                     int nz, int i, int j, int k)
{
    return k + 1 < nz
        ? 0.5 * (static_cast<double>(rho[i][j][k]) + rho[i][j][k + 1]) * w[i][j][k]
        : static_cast<double>(rho[i][j][k]) * w[i][j][k];
}

inline float density_tendency_precise(const Field3D& rho, const Field3D& u, const Field3D& v, const Field3D& w,
                                     int nz, int i, int j, int k, float dx, float dy, float dz,
                                     float frame_u_ms, float frame_v_ms)
{
    const double x = (x_face_mass_flux_precise(rho, u, i, j, k, frame_u_ms) -
                      x_face_mass_flux_precise(rho, u, i - 1, j, k, frame_u_ms)) / dx;
    const double y = (y_face_mass_flux_precise(rho, v, i, j, k, frame_v_ms) -
                      y_face_mass_flux_precise(rho, v, i, j - 1, k, frame_v_ms)) / dy;
    const double z = (z_face_mass_flux_precise(rho, w, nz, i, j, k) -
                      (k > 0 ? z_face_mass_flux_precise(rho, w, nz, i, j, k - 1) : 0.0)) / dz;
    return static_cast<float>(-(x + y + z));
}

/**
 * Reproduce the flat Cartesian C-grid pressure shader's conservative density
 * rate for one physical scalar cell. Inputs must use the shader's stored
 * pre-commit FP32 state, positive spacings/timestep, and a physical cell with
 * i/j halo neighbours.
 */
inline float density_tendency(
    const Field3D& rho, const Field3D& u, const Field3D& v,
    const Field3D& w,
    int nx, int ny, int nz,
    int i, int j, int k,
    float dx, float dy, float dz,
    float frame_u_ms = 0.0f, float frame_v_ms = 0.0f,
    bool repair_primary_state = true)
{
    const float inv_dx = 1.0f / dx;
    const float inv_dy = 1.0f / dy;
    const float inv_dz = 1.0f / dz;

    float flux_xp = 0.0f;
    if (i + 1 < nx)
        flux_xp = x_face_mass_flux(rho, u, i, j, k, frame_u_ms);
    const float flux_xm = x_face_mass_flux(rho, u, i - 1, j, k, frame_u_ms);
    const float divm_x = (flux_xp - flux_xm) * inv_dx;

    float flux_yp = 0.0f;
    if (j + 1 < ny)
        flux_yp = y_face_mass_flux(rho, v, i, j, k, frame_v_ms);
    const float flux_ym = y_face_mass_flux(rho, v, i, j - 1, k, frame_v_ms);
    const float divm_y = (flux_yp - flux_ym) * inv_dy;

    const float flux_zp =
        z_face_mass_flux(rho, w, nz, i, j, k);
    float flux_zm = 0.0f;
    if (k > 0)
        flux_zm = z_face_mass_flux(rho, w, nz, i, j, k - 1);
    const float divm_z = (flux_zp - flux_zm) * inv_dz;

    float drho = -(divm_x + divm_y + divm_z);
    if (repair_primary_state && !std::isfinite(drho)) drho = 0.0f;
    return drho;
}

inline float requested_density_change(
    const Field3D& rho, const Field3D& u, const Field3D& v, const Field3D& w,
    int nx, int ny, int nz, int i, int j, int k,
    float dt_small, float dx, float dy, float dz,
    float frame_u_ms = 0.0f, float frame_v_ms = 0.0f,
    bool repair_primary_state = true)
{
    return density_tendency(rho, u, v, w, nx, ny, nz, i, j, k,
                            dx, dy, dz, frame_u_ms, frame_v_ms, repair_primary_state) * dt_small;
}

/**
 * Attribute one completed fused flat Cartesian C-grid device acoustic
 * substep from its exact pre-commit FP32 fields and actual committed state.
 *
 * The state view supplies the committed u/v/w/rho/p fields, unchanged moist
 * scalars/TKE, flat geometry, and base pressure. The five `*_before` fields
 * must be the inputs passed to the shader. The function adds one complete
 * continuity/EOS plus backward-pressure-gradient owner to `report`; boundary
 * closure remains the boundary scheme's separate owner.
 */
void attribute_flat_cgrid_fused_commit(
    const SimulationStateView& state,
    const Field3D& u_before,
    const Field3D& v_before,
    const Field3D& w_before,
    const Field3D& rho_before,
    const Field3D& p_before,
    float dt_small,
    float dx,
    float dy,
    float dz,
    CartesianCGridAcousticEnergyReportV1& report);

} // namespace tmv::device_acoustic_energy
