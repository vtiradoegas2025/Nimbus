#pragma once
#include "compute/dispatch/terrain.hpp"
#include <cmath>
#include <initializer_list>
namespace nimbus::compute::mapped_boundary
{
inline bool valid_options(const MappedTerrainFinalStateDispatch& d)
{
    for (int face : {d.west, d.east, d.south, d.north})
        if (face < 0 || face > 4)
            return false;
    return d.bottom >= 0 && d.bottom <= 1 && d.top >= 0 && d.top <= 2 && std::isfinite(d.frame_u) &&
           std::isfinite(d.frame_v);
}
inline abi::MappedTerrainBoundaryPushConstants constants(const MappedTerrainFinalStateDispatch& d)
{
    return {d.nx, d.ny, d.nz, d.west, d.east, d.south, d.north, d.bottom, d.top, d.frame_u, d.frame_v, 0};
}
} // namespace nimbus::compute::mapped_boundary
