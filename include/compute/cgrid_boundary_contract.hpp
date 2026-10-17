#pragma once
#include "compute/dispatch/cgrid_boundary.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
namespace nimbus::compute::cgrid_boundary
{
inline std::size_t cells(const CgridBoundaryDispatch& d)
{
    return std::size_t(d.constants.nx) * d.constants.ny * d.constants.nz;
}
inline std::size_t waves(const CgridBoundaryDispatch& d)
{
    return 2 * (std::size_t(d.constants.nx) + d.constants.ny) * d.constants.nz;
}
inline bool overlap(const float* a, std::size_t na, const float* b, std::size_t nb)
{
    const auto x = reinterpret_cast<std::uintptr_t>(a), y = reinterpret_cast<std::uintptr_t>(b);
    return x <= y ? y - x < na * sizeof(float) : x - y < nb * sizeof(float);
}
inline bool validate(const CgridBoundaryDispatch& d)
{
    const auto& p = d.constants;
    if (p.nx < 3 || p.ny < 3 || p.nz < 2 || !d.wave || !d.wave_output ||
        std::size_t(p.nx) > std::size_t(std::numeric_limits<int>::max()) / std::size_t(p.ny) / std::size_t(p.nz) ||
        2 * (std::size_t(p.nx) + p.ny) > std::size_t(std::numeric_limits<int>::max()) / std::size_t(p.nz))
        return false;
    for (int bc : {p.bc_west, p.bc_east, p.bc_south, p.bc_north})
        if (bc < 0 || bc > 4)
            return false;
    if (p.bc_bottom < 0 || p.bc_bottom > 1 || p.bc_top < 0 || p.bc_top > 2 || !std::isfinite(p.dt_small) ||
        p.dt_small <= 0 || !std::isfinite(p.inv_dx) || p.inv_dx <= 0 || !std::isfinite(p.inv_dy) || p.inv_dy <= 0 ||
        !std::isfinite(p.frame_velocity_u_ms) || !std::isfinite(p.frame_velocity_v_ms))
        return false;
    for (std::size_t f = 0; f < 5; ++f)
    {
        if (!d.fields[f] || !d.output[f] || overlap(d.output[f], cells(d), d.wave_output, waves(d)))
            return false;
        for (std::size_t other = 0; other < f; ++other)
            if (overlap(d.output[f], cells(d), d.output[other], cells(d)))
                return false;
        for (std::size_t n = 0; n < cells(d); ++n)
            if (!std::isfinite(d.fields[f][n]) || (f >= 3 && d.fields[f][n] <= 0))
                return false;
    }
    for (std::size_t n = 0; n < waves(d); ++n)
        if (!std::isfinite(d.wave[n]))
            return false;
    return true;
}
inline bool finite_candidate(const std::array<const float*, 6>& values, const CgridBoundaryDispatch& d)
{
    for (std::size_t f = 0; f < values.size(); ++f)
        for (std::size_t n = 0; n < (f == 5 ? waves(d) : cells(d)); ++n)
            if (!std::isfinite(values[f][n]) || ((f == 3 || f == 4) && values[f][n] <= 0))
                return false;
    return true;
}
} // namespace nimbus::compute::cgrid_boundary
