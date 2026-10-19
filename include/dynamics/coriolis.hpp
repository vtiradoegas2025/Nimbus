#pragma once

#include "core/infra/domain_motion.hpp"

#include <cmath>
#include <stdexcept>

namespace tmv
{

/**
 * Earth-referenced tangent-plane Coriolis profile.
 *
 * `f_at_grid_y0` is the parameter at the grid's lower y face at the
 * evaluation time. `beta` is df/dy in Earth coordinates. Keeping the time-
 * dependent origin in this derived value makes the hot-loop expression just
 * `f_at_grid_y0 + beta * grid_y_m`, while preserving the same Earth-relative
 * physics for stationary, translating, and recycled domains.
 */
struct CartesianCoriolisProfile
{
    double f_at_grid_y0 = 0.0;
    double beta         = 0.0;

    double at_grid_y(double grid_y_m) const
    {
        const double value = f_at_grid_y0 + beta * grid_y_m;
        if (!std::isfinite(grid_y_m) || !std::isfinite(value))
            throw std::invalid_argument("Coriolis evaluation requires a finite grid y coordinate");
        return value;
    }

    bool is_f_plane() const noexcept { return beta == 0.0; }
};

inline constexpr double earth_rotation_rate_rad_s = 7.292e-5;
inline constexpr double earth_mean_radius_m       = 6'371'000.0;

/**
 * Construct the local f- or beta-plane profile. `latitude_deg` defines the
 * tangent-plane reference latitude at Earth y=0. The beta-plane follows the
 * standard first-order approximation
 *
 *   f(y) = 2 Omega sin(phi0) + 2 Omega cos(phi0) y / R_earth.
 */
inline CartesianCoriolisProfile make_cartesian_coriolis_profile(
    bool enabled,
    bool beta_plane,
    double latitude_deg,
    const tmv::domain::HorizontalDomainFrame& frame,
    double simulation_time_s)
{
    if (!std::isfinite(latitude_deg) || latitude_deg < -90.0 || latitude_deg > 90.0)
        throw std::invalid_argument("Coriolis latitude must be finite and in [-90, 90]");
    if (!enabled)
        return {};

    constexpr double pi = 3.14159265358979323846;
    const double latitude_rad = latitude_deg * pi / 180.0;
    const double f_reference = 2.0 * earth_rotation_rate_rad_s * std::sin(latitude_rad);
    const double beta = beta_plane
        ? 2.0 * earth_rotation_rate_rad_s * std::cos(latitude_rad) / earth_mean_radius_m
        : 0.0;
    if (!std::isfinite(simulation_time_s)
        || !std::isfinite(frame.earth_origin_at_t0_m.x_m)
        || !std::isfinite(frame.earth_origin_at_t0_m.y_m)
        || !std::isfinite(frame.velocity_ms.u_ms)
        || !std::isfinite(frame.velocity_ms.v_ms))
        throw std::invalid_argument("Coriolis domain frame and evaluation time must be finite");
    const double origin_y_m = frame.earth_origin_at_t0_m.y_m
        + frame.velocity_ms.v_ms * simulation_time_s;
    const CartesianCoriolisProfile profile{
        f_reference + beta * origin_y_m,
        beta,
    };
    if (!std::isfinite(profile.f_at_grid_y0) || !std::isfinite(profile.beta))
        throw std::invalid_argument("Coriolis profile is not finite");
    return profile;
}

} // namespace tmv
