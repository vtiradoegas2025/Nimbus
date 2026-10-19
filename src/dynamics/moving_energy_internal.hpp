#pragma once
#include "dynamics/moving_energy.hpp"
#include "core/runtime/simulation_state.hpp"

namespace tmv::moving_energy_detail
{
void validate(const PrognosticFieldStorage& fields, const TerrainMetrics3D& geometry);
void validate_target(const TerrainMetrics3D& geometry, const std::vector<double>& target);
} // namespace tmv::moving_energy_detail
