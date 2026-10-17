#pragma once

#include "turbulence/turbulence_base.hpp"

namespace tmv::dynamic_smagorinsky
{

void compute_coefficient(const TurbulenceConfig&    config,
                         const GridMetrics&         grid,
                         const TurbulenceStateView& state,
                         Field3D&                   effective_cs);

} // namespace tmv::dynamic_smagorinsky
