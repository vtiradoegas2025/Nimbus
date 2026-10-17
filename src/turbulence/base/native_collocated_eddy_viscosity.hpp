#pragma once
#include "turbulence/turbulence_base.hpp"
#include <array>
namespace tmv::collocated_eddy_viscosity
{
struct NativeResult
{
    std::array<Field3D, 8> closure;
    std::array<Field3D, 8> tendency;
};
bool compute_native(const TurbulenceConfig&, const GridMetrics&, const TurbulenceStateView&, bool tke, NativeResult&);
void publish_tendencies(NativeResult&, TurbulenceTendencies&);
} // namespace tmv::collocated_eddy_viscosity
