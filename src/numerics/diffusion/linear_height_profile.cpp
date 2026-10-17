#include "numerics/diffusion/linear_height_profile.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <stdexcept>

namespace tmv
{

Field3D build_linear_height_diffusivity(
    int nr, int nth, const std::vector<double>& dz,
    double surface_value, double top_value)
{
    if (nr < 1 || nth < 1 || dz.size() < 2)
        throw std::invalid_argument(
            "linear-height diffusivity requires positive horizontal extents "
            "and at least two vertical cells");
    if (!std::isfinite(surface_value) || surface_value < 0.0 ||
        !std::isfinite(top_value) || top_value < 0.0)
        throw std::invalid_argument(
            "linear-height diffusivity endpoints must be non-negative and finite");

    std::vector<double> centers(dz.size());
    double interface_height = 0.0;
    for (std::size_t k = 0; k < dz.size(); ++k)
    {
        if (!std::isfinite(dz[k]) || dz[k] <= 0.0)
            throw std::invalid_argument(
                "linear-height diffusivity requires positive finite layer thicknesses");
        centers[k] = interface_height + 0.5 * dz[k];
        interface_height += dz[k];
    }
    const double span = centers.back() - centers.front();
    if (!std::isfinite(span) || span <= 0.0)
        throw std::invalid_argument(
            "linear-height diffusivity has a degenerate cell-center span");

    Field3D result(nr, nth, static_cast<int>(dz.size()), 0.0f);
    for (std::size_t k = 0; k < dz.size(); ++k)
    {
        const double fraction = (centers[k] - centers.front()) / span;
        const float value = static_cast<float>(
            surface_value + fraction * (top_value - surface_value));
        if (!std::isfinite(value) || value < 0.0f)
            throw std::invalid_argument(
                "linear-height diffusivity is not representable in float32");
        for (int i = 0; i < nr; ++i)
            for (int j = 0; j < nth; ++j)
                result(i, j, static_cast<int>(k)) = value;
    }
    return result;
}

const Field3D* prescribed_vertical_diffusivity(
    DiffusionRuntimeState& runtime, int nr, int nth,
    const GridMetrics& grid)
{
    if (!runtime.config.use_variable_K)
        return nullptr;

    std::string source = runtime.config.coefficient_source;
    std::transform(source.begin(), source.end(), source.begin(),
                   [](unsigned char c) {
                       return static_cast<char>(std::tolower(c));
                   });
    if (source != "linear_height")
        throw std::invalid_argument(
            "variable diffusion requires the linear_height coefficient source");
    if (grid.terrain_metrics_active)
        throw std::invalid_argument(
            "linear-height variable diffusion requires flat Cartesian geometry");
    if (!std::isfinite(runtime.config.K_v_surface) ||
        runtime.config.K_v_surface < 0.0 ||
        !std::isfinite(runtime.config.K_v_top) ||
        runtime.config.K_v_top < 0.0)
        throw std::invalid_argument(
            "linear-height diffusion endpoints must be non-negative and finite");
    if (nr < 1 || nth < 1 || grid.dz.size() < 2)
        throw std::invalid_argument(
            "linear-height variable diffusion requires at least two vertical cells");

    const int nz = static_cast<int>(grid.dz.size());
    const bool cache_current =
        runtime.prescribed_vertical_nr == nr &&
        runtime.prescribed_vertical_nth == nth &&
        runtime.prescribed_vertical_nz == nz &&
        runtime.prescribed_vertical_surface == runtime.config.K_v_surface &&
        runtime.prescribed_vertical_top == runtime.config.K_v_top &&
        runtime.prescribed_vertical_dz == grid.dz &&
        runtime.prescribed_vertical_K.size_r() == nr &&
        runtime.prescribed_vertical_K.size_th() == nth &&
        runtime.prescribed_vertical_K.size_z() == nz;
    if (cache_current)
        return &runtime.prescribed_vertical_K;

    Field3D candidate = build_linear_height_diffusivity(
        nr, nth, grid.dz, runtime.config.K_v_surface,
        runtime.config.K_v_top);
    runtime.prescribed_vertical_K = std::move(candidate);
    runtime.prescribed_vertical_dz = grid.dz;
    runtime.prescribed_vertical_nr = nr;
    runtime.prescribed_vertical_nth = nth;
    runtime.prescribed_vertical_nz = nz;
    runtime.prescribed_vertical_surface = runtime.config.K_v_surface;
    runtime.prescribed_vertical_top = runtime.config.K_v_top;
    return &runtime.prescribed_vertical_K;
}

} // namespace tmv
