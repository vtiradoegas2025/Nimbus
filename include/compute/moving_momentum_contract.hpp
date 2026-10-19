#pragma once
#include "compute/dispatch/moving_momentum.hpp"
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
namespace nimbus::compute::moving_momentum
{
inline std::size_t cells(const MovingMomentumDispatch& d)
{
    const auto& p = d.constants;
    return (std::size_t(p.nx) + 1) * (std::size_t(p.ny) + 1) * (std::size_t(p.nz) + 1);
}
inline bool validate(const MovingMomentumDispatch& d)
{
    const auto& p = d.constants;
    if (p.nx < 4 || p.ny < 4 || p.nz < 2 || p.component < 0 || p.component > 2 || p.include_boundary < 0 ||
        p.include_boundary > 1 || p.operation < 0 || p.operation > 4 || !d.input || !d.output)
        return false;
    if (p.operation == 2 && p.component > 1)
        return false;
    if (p.operation >= 2 && p.nz != 2)
        return false;
    if (std::size_t(p.nx) + 1 > std::size_t(std::numeric_limits<int>::max()) / MovingMomentumDispatch::output_stride /
                                    (std::size_t(p.ny) + 1) / (std::size_t(p.nz) + 1))
        return false;
    const std::array<float, 10> parameters{p.dx_high,
                                           p.dx_low,
                                           p.dy_high,
                                           p.dy_low,
                                           p.dz_high,
                                           p.dz_low,
                                           p.surface_u_high,
                                           p.surface_u_low,
                                           p.surface_v_high,
                                           p.surface_v_low};
    for (float v : parameters)
        if (!std::isfinite(v))
            return false;
    for (std::size_t f = 0; f < parameters.size(); f += 2)
        if (std::abs(double(parameters[f + 1])) >
            std::abs(double(parameters[f])) * std::numeric_limits<float>::epsilon())
            return false;
    for (int f : {0, 2, 4})
        if (double(parameters[f]) + parameters[f + 1] <= 0)
            return false;
    const auto a = reinterpret_cast<std::uintptr_t>(d.input), b = reinterpret_cast<std::uintptr_t>(d.output);
    if ((a <= b && b - a < cells(d) * MovingMomentumDispatch::input_stride * sizeof(float)) ||
        (b < a && a - b < cells(d) * MovingMomentumDispatch::output_stride * sizeof(float)))
        return false;
    for (std::size_t n = 0; n < cells(d) * MovingMomentumDispatch::input_stride; ++n)
        if (!std::isfinite(d.input[n]))
            return false;
    for (int i = 0; i <= p.nx; ++i)
        for (int j = 0; j <= p.ny; ++j)
            for (int k = 0; k <= p.nz; ++k)
            {
                const auto* value = d.input + ((std::size_t(i) * (p.ny + 1) + j) * (p.nz + 1) + k) *
                                                  MovingMomentumDispatch::input_stride;
                for (int f : {2, 4, 9, 11, 13, 15, 17})
                    if (std::abs(double(value[f + 1])) >
                        std::abs(double(value[f])) * std::numeric_limits<float>::epsilon())
                        return false;
                if (p.operation == 4)
                {
                    const double fraction = double(value[9]) + value[10];
                    if (value[11] < -1073 || value[11] > 1024 || value[11] != std::floor(value[11]) || value[12] != 0 ||
                        (fraction != 0 && (std::abs(fraction) < .5 || std::abs(fraction) > 1)) ||
                        (fraction == 0 && value[11] != 0))
                        return false;
                }
                if (i < p.nx && j < p.ny && k < p.nz)
                {
                    if (value[0] <= 0)
                        return false;
                    if (p.operation != 1 &&
                        (value[1] <= 0 || double(value[2]) + value[3] <= 0 || double(value[4]) + value[5] <= 0))
                        return false;
                }
                if (i < p.nx && j < p.ny && k == 0 && (value[13] != 0 || value[14] != 0))
                    return false;
                if ((p.operation == 1 || p.operation >= 2) && (value[13] != 0 || value[14] != 0))
                    return false;
            }
    return true;
}
inline bool finite_candidate(const float* candidate, const MovingMomentumDispatch& d)
{
    for (std::size_t n = 0; n < cells(d) * MovingMomentumDispatch::output_stride; ++n)
        if (!std::isfinite(candidate[n]))
            return false;
    for (std::size_t n = 0; n < cells(d); ++n)
    {
        const float exponent = candidate[n * MovingMomentumDispatch::output_stride + 23];
        if (exponent < 0 || exponent > 149 || exponent != std::floor(exponent))
            return false;
    }
    if (d.constants.operation == 1)
    {
        const auto& p = d.constants;
        for (int direction = 0; direction < 2; ++direction)
            for (int i = 1; i < (direction == 0 ? p.nx : p.nx - 1); ++i)
                for (int j = 1; j < (direction == 1 ? p.ny : p.ny - 1); ++j)
                    for (int k = 0; k < p.nz; ++k)
                    {
                        const auto  n      = (std::size_t(i) * (p.ny + 1) + j) * (p.nz + 1) + k;
                        const auto* input  = d.input + n * MovingMomentumDispatch::input_stride;
                        const auto* output = candidate + n * MovingMomentumDispatch::output_stride;
                        if ((input[9 + 2 * direction] != 0 || input[10 + 2 * direction] != 0) &&
                            double(output[2 * direction]) + output[2 * direction + 1] == 0)
                            return false;
                    }
    }
    return true;
}
} // namespace nimbus::compute::moving_momentum
