#pragma once

#include "core/infra/physical_constants.hpp"
#include "terrain/conservative_transport.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace tmv::diagnostics
{
inline double reference_at_cell(const std::vector<double>& profile,
                                const TerrainMetrics3D* terrain, double dz,
                                int i, int j, int k)
{
    if (terrain)
        return terrain_transport::reference_profile_value_fp32(profile, terrain->z(i, j, k), dz);
    if (k < 0 || static_cast<std::size_t>(k) >= profile.size() || !std::isfinite(profile[k]))
        throw std::invalid_argument("diagnostic requires a complete finite reference profile");
    return static_cast<double>(static_cast<float>(profile[k]));
}

inline double density_buoyancy(double density, double reference_density,
                               double vapor, double reference_vapor, double condensate)
{
    if (!std::isfinite(density) || density <= 0.0 ||
        !std::isfinite(reference_density) || reference_density <= 0.0 ||
        !std::isfinite(vapor) || vapor < 0.0 || !std::isfinite(reference_vapor) || reference_vapor < 0.0 ||
        !std::isfinite(condensate) || condensate < 0.0)
        return std::numeric_limits<double>::quiet_NaN();
    return -physical_constants::gravity_ms2 * (density - reference_density) / density
        + physical_constants::gravity_ms2 * 0.608 * (vapor - reference_vapor)
        - physical_constants::gravity_ms2 * condensate;
}
}
