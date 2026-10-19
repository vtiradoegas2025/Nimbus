#pragma once

#include "core/field/field3d.hpp"
#include "core/infra/coordinate_system.hpp"
#include "init/config/config_v1.hpp"

#include <vector>

namespace tmv::cartesian_mass_flux
{

/**
 * Momentum-weighted pressure coupling for collocated face fluxes.
 *
 * A centered cell-to-cell pressure gradient is blind to the Nyquist
 * checkerboard on an A-grid.  The face velocity therefore uses the direct
 * pressure jump at that face minus the interpolated cell-centered pressure
 * acceleration.  The difference is high-order for a smooth pressure field,
 * but nonzero for the odd/even mode.  Supplying the same object to continuity
 * and retained scalar transport keeps one authoritative mass flux.
 *
 * This is deliberately inactive for C-grid fluxes, whose face velocity and
 * adjacent-cell pressure jump are already colocated.
 */
struct CollocatedPressureCoupling
{
    const Field3D* pressure = nullptr;
    const std::vector<double>* base_pressure = nullptr;
    double dt = 0.0;
    double dx = 0.0;
    double dy = 0.0;
    double dz = 0.0;
};

inline double average_product(float rho_lo, float rho_hi,
                              float velocity_lo, float velocity_hi)
{
    return 0.25 * (static_cast<double>(rho_lo) + static_cast<double>(rho_hi))
                * (static_cast<double>(velocity_lo)
                   + static_cast<double>(velocity_hi));
}

inline double average_product_grid_relative(
    float rho_lo, float rho_hi,
    float earth_velocity_lo, float earth_velocity_hi,
    double frame_velocity_ms)
{
    if (frame_velocity_ms == 0.0)
        return average_product(
            rho_lo, rho_hi, earth_velocity_lo, earth_velocity_hi);
    return 0.5 * (static_cast<double>(rho_lo) + static_cast<double>(rho_hi))
         * (0.5 * (static_cast<double>(earth_velocity_lo)
                 + static_cast<double>(earth_velocity_hi))
            - frame_velocity_ms);
}

inline double cgrid_product(float rho_lo, float rho_hi, float velocity_face)
{
    return 0.5 * (static_cast<double>(rho_lo) + static_cast<double>(rho_hi))
               * static_cast<double>(velocity_face);
}

inline double cgrid_product_grid_relative(
    float rho_lo, float rho_hi, float earth_velocity_face,
    double frame_velocity_ms)
{
    if (frame_velocity_ms == 0.0)
        return cgrid_product(rho_lo, rho_hi, earth_velocity_face);
    return 0.5 * (static_cast<double>(rho_lo) + static_cast<double>(rho_hi))
         * (static_cast<double>(earth_velocity_face) - frame_velocity_ms);
}

inline int periodic_physical_index(int index, int extent)
{
    if (index < 1) return extent - 2;
    if (index > extent - 2) return 1;
    return index;
}

inline double pressure_x(const Field3D& p, int i, int j, int k)
{
    return static_cast<double>(p[i][j][k]);
}

inline double pressure_z_perturbation(
    const CollocatedPressureCoupling& coupling, int i, int j, int k)
{
    return static_cast<double>((*coupling.pressure)[i][j][k])
         - static_cast<double>(static_cast<float>(
               (*coupling.base_pressure)[static_cast<std::size_t>(k)]));
}

inline double corrected_x_face_velocity(
    const Field3D& density, const Field3D& velocity,
    int i, int j, int k, int nx,
    tmv::init::config::LateralBC west,
    tmv::init::config::LateralBC east,
    const CollocatedPressureCoupling& coupling)
{
    using tmv::init::config::LateralBC;
    int lo = i;
    int hi = i + 1;
    const bool periodic_face = (i == 0 && west == LateralBC::Periodic)
        || (i == nx - 2 && east == LateralBC::Periodic);
    if (periodic_face)
    {
        lo = nx - 2;
        hi = 1;
    }

    const double rho_lo = static_cast<double>(density[lo][j][k]);
    const double rho_hi = static_cast<double>(density[hi][j][k]);
    const double rho_face = 0.5 * (rho_lo + rho_hi);
    const double velocity_face = 0.5 * (
        static_cast<double>(velocity[lo][j][k])
        + static_cast<double>(velocity[hi][j][k]));
    const Field3D& p = *coupling.pressure;
    const auto neighbor = [nx, periodic_face](int cell, int offset)
    {
        const int raw = cell + offset;
        return periodic_face ? periodic_physical_index(raw, nx) : raw;
    };
    const double grad_lo = (
        pressure_x(p, neighbor(lo, 1), j, k)
        - pressure_x(p, neighbor(lo, -1), j, k)) / (2.0 * coupling.dx);
    const double grad_hi = (
        pressure_x(p, neighbor(hi, 1), j, k)
        - pressure_x(p, neighbor(hi, -1), j, k)) / (2.0 * coupling.dx);
    const double accel_cells = -0.5 * (grad_lo / rho_lo + grad_hi / rho_hi);
    const double accel_face = -(
        pressure_x(p, hi, j, k) - pressure_x(p, lo, j, k))
        / (coupling.dx * rho_face);
    return velocity_face + coupling.dt * (accel_face - accel_cells);
}

inline double corrected_y_face_velocity(
    const Field3D& density, const Field3D& velocity,
    int i, int j, int k, int ny,
    tmv::init::config::LateralBC south,
    tmv::init::config::LateralBC north,
    const CollocatedPressureCoupling& coupling)
{
    using tmv::init::config::LateralBC;
    int lo = j;
    int hi = j + 1;
    const bool periodic_face = (j == 0 && south == LateralBC::Periodic)
        || (j == ny - 2 && north == LateralBC::Periodic);
    if (periodic_face)
    {
        lo = ny - 2;
        hi = 1;
    }

    const double rho_lo = static_cast<double>(density[i][lo][k]);
    const double rho_hi = static_cast<double>(density[i][hi][k]);
    const double rho_face = 0.5 * (rho_lo + rho_hi);
    const double velocity_face = 0.5 * (
        static_cast<double>(velocity[i][lo][k])
        + static_cast<double>(velocity[i][hi][k]));
    const Field3D& p = *coupling.pressure;
    const auto neighbor = [ny, periodic_face](int cell, int offset)
    {
        const int raw = cell + offset;
        return periodic_face ? periodic_physical_index(raw, ny) : raw;
    };
    const double grad_lo = (
        static_cast<double>(p[i][neighbor(lo, 1)][k])
        - static_cast<double>(p[i][neighbor(lo, -1)][k])) / (2.0 * coupling.dy);
    const double grad_hi = (
        static_cast<double>(p[i][neighbor(hi, 1)][k])
        - static_cast<double>(p[i][neighbor(hi, -1)][k])) / (2.0 * coupling.dy);
    const double accel_cells = -0.5 * (grad_lo / rho_lo + grad_hi / rho_hi);
    const double accel_face = -(
        static_cast<double>(p[i][hi][k]) - static_cast<double>(p[i][lo][k]))
        / (coupling.dy * rho_face);
    return velocity_face + coupling.dt * (accel_face - accel_cells);
}

inline double corrected_z_face_velocity(
    const Field3D& density, const Field3D& velocity,
    int i, int j, int k, const CollocatedPressureCoupling& coupling)
{
    const int lo = k;
    const int hi = k + 1;
    const double rho_lo = static_cast<double>(density[i][j][lo]);
    const double rho_hi = static_cast<double>(density[i][j][hi]);
    const double rho_face = 0.5 * (rho_lo + rho_hi);
    const double velocity_face = 0.5 * (
        static_cast<double>(velocity[i][j][lo])
        + static_cast<double>(velocity[i][j][hi]));
    const double grad_lo = (
        pressure_z_perturbation(coupling, i, j, lo + 1)
        - pressure_z_perturbation(coupling, i, j, lo - 1))
        / (2.0 * coupling.dz);
    const double grad_hi = (
        pressure_z_perturbation(coupling, i, j, hi + 1)
        - pressure_z_perturbation(coupling, i, j, hi - 1))
        / (2.0 * coupling.dz);
    const double accel_cells = -0.5 * (grad_lo / rho_lo + grad_hi / rho_hi);
    const double accel_face = -(
        pressure_z_perturbation(coupling, i, j, hi)
        - pressure_z_perturbation(coupling, i, j, lo))
        / (coupling.dz * rho_face);
    return velocity_face + coupling.dt * (accel_face - accel_cells);
}

inline bool is_closed_lateral(tmv::init::config::LateralBC bc)
{
    using tmv::init::config::LateralBC;
    return bc == LateralBC::Rigid || bc == LateralBC::RigidSlip;
}

inline double x_face(const Field3D& density, const Field3D& velocity,
                     int i, int j, int k, int nx, StaggerType staggering,
                     tmv::init::config::LateralBC west,
                     tmv::init::config::LateralBC east,
                     const CollocatedPressureCoupling* coupling = nullptr,
                     double frame_velocity_ms = 0.0)
{
    if (staggering == StaggerType::CGrid)
    {
        return cgrid_product_grid_relative(
            density[i][j][k], density[i + 1][j][k], velocity[i][j][k],
            frame_velocity_ms);
    }

    using tmv::init::config::LateralBC;
    if (i == 0)
    {
        if (is_closed_lateral(west)) return 0.0;
        if (west == LateralBC::Periodic)
        {
            if (coupling != nullptr)
            {
                return 0.5 * (
                           static_cast<double>(density[nx - 2][j][k])
                           + static_cast<double>(density[1][j][k]))
                     * (corrected_x_face_velocity(
                           density, velocity, i, j, k, nx, west, east,
                           *coupling) - frame_velocity_ms);
            }
            return average_product_grid_relative(
                density[nx - 2][j][k], density[1][j][k],
                velocity[nx - 2][j][k], velocity[1][j][k],
                frame_velocity_ms);
        }
        return static_cast<double>(density[0][j][k])
             * (static_cast<double>(velocity[0][j][k]) - frame_velocity_ms);
    }
    if (i == nx - 2)
    {
        if (is_closed_lateral(east)) return 0.0;
        if (east == LateralBC::Periodic)
        {
            if (coupling != nullptr)
            {
                return 0.5 * (
                           static_cast<double>(density[nx - 2][j][k])
                           + static_cast<double>(density[1][j][k]))
                     * (corrected_x_face_velocity(
                           density, velocity, i, j, k, nx, west, east,
                           *coupling) - frame_velocity_ms);
            }
            return average_product_grid_relative(
                density[nx - 2][j][k], density[1][j][k],
                velocity[nx - 2][j][k], velocity[1][j][k],
                frame_velocity_ms);
        }
        return static_cast<double>(density[nx - 1][j][k])
             * (static_cast<double>(velocity[nx - 1][j][k]) - frame_velocity_ms);
    }
    if (coupling != nullptr)
    {
        return 0.5 * (static_cast<double>(density[i][j][k])
                    + static_cast<double>(density[i + 1][j][k]))
             * (corrected_x_face_velocity(
                   density, velocity, i, j, k, nx, west, east, *coupling)
                   - frame_velocity_ms);
    }
    return average_product_grid_relative(
        density[i][j][k], density[i + 1][j][k],
        velocity[i][j][k], velocity[i + 1][j][k], frame_velocity_ms);
}

inline double y_face(const Field3D& density, const Field3D& velocity,
                     int i, int j, int k, int ny, StaggerType staggering,
                     tmv::init::config::LateralBC south,
                     tmv::init::config::LateralBC north,
                     const CollocatedPressureCoupling* coupling = nullptr,
                     double frame_velocity_ms = 0.0)
{
    if (staggering == StaggerType::CGrid)
    {
        return cgrid_product_grid_relative(
            density[i][j][k], density[i][j + 1][k], velocity[i][j][k],
            frame_velocity_ms);
    }

    using tmv::init::config::LateralBC;
    if (j == 0)
    {
        if (is_closed_lateral(south)) return 0.0;
        if (south == LateralBC::Periodic)
        {
            if (coupling != nullptr)
            {
                return 0.5 * (
                           static_cast<double>(density[i][ny - 2][k])
                           + static_cast<double>(density[i][1][k]))
                     * (corrected_y_face_velocity(
                           density, velocity, i, j, k, ny, south, north,
                           *coupling) - frame_velocity_ms);
            }
            return average_product_grid_relative(
                density[i][ny - 2][k], density[i][1][k],
                velocity[i][ny - 2][k], velocity[i][1][k],
                frame_velocity_ms);
        }
        return static_cast<double>(density[i][0][k])
             * (static_cast<double>(velocity[i][0][k]) - frame_velocity_ms);
    }
    if (j == ny - 2)
    {
        if (is_closed_lateral(north)) return 0.0;
        if (north == LateralBC::Periodic)
        {
            if (coupling != nullptr)
            {
                return 0.5 * (
                           static_cast<double>(density[i][ny - 2][k])
                           + static_cast<double>(density[i][1][k]))
                     * (corrected_y_face_velocity(
                           density, velocity, i, j, k, ny, south, north,
                           *coupling) - frame_velocity_ms);
            }
            return average_product_grid_relative(
                density[i][ny - 2][k], density[i][1][k],
                velocity[i][ny - 2][k], velocity[i][1][k],
                frame_velocity_ms);
        }
        return static_cast<double>(density[i][ny - 1][k])
             * (static_cast<double>(velocity[i][ny - 1][k]) - frame_velocity_ms);
    }
    if (coupling != nullptr)
    {
        return 0.5 * (static_cast<double>(density[i][j][k])
                    + static_cast<double>(density[i][j + 1][k]))
             * (corrected_y_face_velocity(
                   density, velocity, i, j, k, ny, south, north, *coupling)
                   - frame_velocity_ms);
    }
    return average_product_grid_relative(
        density[i][j][k], density[i][j + 1][k],
        velocity[i][j][k], velocity[i][j + 1][k], frame_velocity_ms);
}

inline double z_face(const Field3D& density, const Field3D& velocity,
                     int i, int j, int k, int nz, StaggerType staggering,
                     tmv::init::config::BottomBC bottom,
                     tmv::init::config::TopBC top,
                     const CollocatedPressureCoupling* coupling = nullptr)
{
    if (staggering == StaggerType::CGrid)
    {
        if (k == nz - 1)
        {
            if (top != tmv::init::config::TopBC::Open) return 0.0;
            // The exterior density uses the same zero-gradient closure as
            // the open normal velocity, so the boundary face density is the
            // top physical cell density.
            return static_cast<double>(density[i][j][k])
                 * static_cast<double>(velocity[i][j][k]);
        }
        return cgrid_product(density[i][j][k], density[i][j][k + 1],
                             velocity[i][j][k]);
    }

    using tmv::init::config::TopBC;
    if (k == 0)
    {
        // Both supported lower-boundary choices are impermeable. Surface
        // exchange is represented by physics tendencies, never bulk mass
        // crossing the dynamics boundary.
        (void)bottom;
        return 0.0;
    }
    if (k == nz - 2)
    {
        if (top != TopBC::Open) return 0.0;
        return static_cast<double>(density[i][j][nz - 1])
             * static_cast<double>(velocity[i][j][nz - 1]);
    }
    if (coupling != nullptr)
    {
        return 0.5 * (static_cast<double>(density[i][j][k])
                    + static_cast<double>(density[i][j][k + 1]))
             * corrected_z_face_velocity(
                   density, velocity, i, j, k, *coupling);
    }
    return average_product(density[i][j][k], density[i][j][k + 1],
                           velocity[i][j][k], velocity[i][j][k + 1]);
}

} // namespace tmv::cartesian_mass_flux
