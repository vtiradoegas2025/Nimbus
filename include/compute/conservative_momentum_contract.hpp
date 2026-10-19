#pragma once

#include "compute/dispatch/conservative_momentum.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>

namespace nimbus::compute::conservative_momentum
{
inline std::size_t cells(const ConservativeMomentumDispatch& d)
{
    const auto& p = d.constants;
    return std::size_t(p.nx) * p.ny * p.nz;
}
inline std::size_t strip_values(const ConservativeMomentumDispatch& d)
{
    const auto& p = d.constants;
    return 12 * (std::size_t(p.i_end - p.i_begin) + std::size_t(p.j_end - p.j_begin)) *
           std::size_t(p.k_end - p.k_begin);
}
inline std::size_t output_values(const ConservativeMomentumDispatch& d)
{
    return ConservativeMomentumDispatch::tendency_stride * cells(d) + strip_values(d);
}
inline bool validate(const ConservativeMomentumDispatch& d)
{
    const auto& p = d.constants;
    if (!d.input || !d.output || p.nx < 3 || p.ny < 3 || p.nz < 2 || p.operation < 0 || p.operation > 3 ||
        p.i_begin < 1 || p.i_end >= p.nx || p.i_begin >= p.i_end || p.j_begin < 1 || p.j_end >= p.ny ||
        p.j_begin >= p.j_end || p.k_begin < 0 || p.k_end >= p.nz || p.k_begin >= p.k_end ||
        std::size_t(p.nx) > std::size_t(std::numeric_limits<int>::max()) / ConservativeMomentumDispatch::input_stride /
                                std::size_t(p.ny) / std::size_t(p.nz))
        return false;
    if (output_values(d) > std::size_t(std::numeric_limits<int>::max()))
        return false;
    const std::array<float, 10> parameters{p.dx_high,
                                           p.dx_low,
                                           p.dy_high,
                                           p.dy_low,
                                           p.dz_high,
                                           p.dz_low,
                                           p.gravity_high,
                                           p.gravity_low,
                                           p.coriolis_high,
                                           p.coriolis_low};
    for (std::size_t f = 0; f < parameters.size(); f += 2)
        if (!std::isfinite(parameters[f]) || !std::isfinite(parameters[f + 1]) ||
            std::abs(double(parameters[f + 1])) >
                std::abs(double(parameters[f])) * std::numeric_limits<float>::epsilon() ||
            (f < 6 && double(parameters[f]) + parameters[f + 1] <= 0))
            return false;
    const auto a = reinterpret_cast<std::uintptr_t>(d.input), b = reinterpret_cast<std::uintptr_t>(d.output);
    if ((a <= b && b - a < cells(d) * ConservativeMomentumDispatch::input_stride * sizeof(float)) ||
        (b < a && a - b < output_values(d) * sizeof(float)))
        return false;
    for (std::size_t n = 0; n < cells(d); ++n)
    {
        const auto* value = d.input + n * ConservativeMomentumDispatch::input_stride;
        for (std::size_t f = 0; f < ConservativeMomentumDispatch::input_stride; ++f)
            if (!std::isfinite(value[f]))
                return false;
        for (int f : {5, 7, 11, 13, 15, 17, 19})
            if (std::abs(double(value[f + 1])) > std::abs(double(value[f])) * std::numeric_limits<float>::epsilon())
                return false;
        if (p.operation >= 2 && (double(value[17]) + value[18] <= 0 || double(value[19]) + value[20] <= 0))
            return false;
        if (value[3] <= 0 || value[4] <= 0 || double(value[5]) + value[6] <= 0 || double(value[7]) + value[8] <= 0)
            return false;
    }
    return true;
}
inline bool finite_candidate(const float* candidate, const ConservativeMomentumDispatch& d)
{
    return std::all_of(candidate, candidate + output_values(d), [](float value) { return std::isfinite(value); });
}
} // namespace nimbus::compute::conservative_momentum
