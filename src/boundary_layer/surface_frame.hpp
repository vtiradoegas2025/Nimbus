#pragma once

#include "core/infra/domain_motion.hpp"

#include <cmath>
#include <stdexcept>
#include <utility>

namespace tmv
{

// A non-flat terrain map that is attached to a translating computational
// grid represents a translating lower surface. Surface exchange therefore
// owns air velocity relative to that surface. Flat-domain surface physics
// retains its established Earth-relative contract.
inline std::pair<double, double> boundary_layer_surface_relative_wind(
    double earth_u_ms,
    double earth_v_ms,
    bool terrain_is_grid_attached,
    const domain::HorizontalDomainFrame* frame)
{
    if (!std::isfinite(earth_u_ms) || !std::isfinite(earth_v_ms))
        throw std::invalid_argument(
            "boundary-layer surface wind must be finite");
    if (!terrain_is_grid_attached)
        return {earth_u_ms, earth_v_ms};
    if (frame == nullptr || !std::isfinite(frame->velocity_ms.u_ms)
        || !std::isfinite(frame->velocity_ms.v_ms))
        throw std::invalid_argument(
            "grid-attached terrain surface exchange requires a finite domain frame");
    return {earth_u_ms - frame->velocity_ms.u_ms,
            earth_v_ms - frame->velocity_ms.v_ms};
}

} // namespace tmv
