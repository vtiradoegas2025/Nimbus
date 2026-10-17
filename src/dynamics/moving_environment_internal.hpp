#pragma once
#include "dynamics/moving_environment.hpp"
#include "core/runtime/simulation_state.hpp"
#include <array>
#include <vector>
namespace tmv::moving_environment_detail
{
using Profiles = std::array<const std::vector<double>*, 5>;
Profiles           profiles(const BaseStateStorage& base);
void               validate_base(const BaseStateStorage& base);
std::vector<float> sample_reference(const BaseStateStorage&    base,
                                    const std::vector<double>& heights,
                                    double                     dz,
                                    TerrainComputeKernels&     kernels);
void validate(const PrognosticFieldStorage& fields, const BaseStateStorage& base, const TerrainMetrics3D& geometry);
std::vector<std::size_t> boundary_cells(const TerrainMetrics3D& geometry);
void                     publish(PrognosticFieldStorage&         fields,
                                 const TerrainMetrics3D&         geometry,
                                 const std::vector<std::size_t>& cells,
                                 const std::vector<float>&       candidates);
} // namespace tmv::moving_environment_detail
