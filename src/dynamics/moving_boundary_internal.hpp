#pragma once
#include "dynamics/moving_boundary.hpp"
#include "diagnostics/deterministic_reduction.hpp"

namespace tmv::moving_boundary_detail
{
void                 validate(const TerrainMetrics3D& geometry,
                              const Field3D&          density,
                              const Field3D&          u,
                              const Field3D&          v,
                              const Field3D*          w,
                              bool                    ground);
void                 validate_flux(const TerrainMetrics3D& geometry, const TerrainContravariantFluxes& flux);
void                 validate_targets(const TerrainMetrics3D& geometry, const MovingGroundTargets& targets);
void                 validate_top_outputs(const Field3D& density, const Field3D& u, const Field3D& v);
void                 publish_top(const std::array<Field3D, 2>& means, Field3D& u, Field3D& v);
MovingBoundaryLedger ledger(const std::array<diagnostics::CompensatedSum, 7>& sums);
} // namespace tmv::moving_boundary_detail
