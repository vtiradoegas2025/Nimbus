#pragma once

#include "numerics/diffusion/diffusion_base.hpp"

#include <vector>

namespace tmv
{

Field3D build_linear_height_diffusivity(
    int nr, int nth, const std::vector<double>& dz,
    double surface_value, double top_value);

const Field3D* prescribed_vertical_diffusivity(
    DiffusionRuntimeState& runtime, int nr, int nth,
    const GridMetrics& grid);

} // namespace tmv
