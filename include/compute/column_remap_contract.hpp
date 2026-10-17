#pragma once
#include "compute/dispatch/column_remap.hpp"
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
namespace nimbus::compute::column_remap
{
inline std::size_t cells(const ColumnRemapDispatch& d)
{
    return std::size_t(d.constants.nx) * d.constants.ny * d.constants.nz;
}
inline bool validate(const ColumnRemapDispatch& d)
{
    const auto& p = d.constants;
    if (p.nx < 1 || p.ny < 1 || p.nz < 1 || p.mode < 0 || p.mode > 3 || !d.input || !d.output ||
        std::size_t(p.nx) > std::size_t(std::numeric_limits<int>::max()) / ColumnRemapDispatch::output_stride /
                                std::size_t(p.ny) / std::size_t(p.nz))
        return false;
    const std::array<float, 4> shifts{p.sx_high, p.sx_low, p.sy_high, p.sy_low};
    for (std::size_t n = 0; n < 4; n += 2)
        if (!std::isfinite(shifts[n]) || !std::isfinite(shifts[n + 1]) ||
            std::abs(double(shifts[n + 1])) > std::abs(double(shifts[n])) * std::numeric_limits<float>::epsilon())
            return false;
    if (std::abs(double(p.sx_high) + p.sx_low) + std::abs(double(p.sy_high) + p.sy_low) > .5)
        return false;
    const auto a = reinterpret_cast<std::uintptr_t>(d.input), b = reinterpret_cast<std::uintptr_t>(d.output);
    if ((a <= b && b - a < cells(d) * ColumnRemapDispatch::input_stride * sizeof(float)) ||
        (b < a && a - b < cells(d) * ColumnRemapDispatch::output_stride * sizeof(float)))
        return false;
    for (std::size_t n = 0; n < cells(d); ++n)
    {
        const auto* v = d.input + ColumnRemapDispatch::input_stride * n;
        for (int c = 0; c < 6; c += 2)
            if (!std::isfinite(v[c]) || !std::isfinite(v[c + 1]) ||
                std::abs(double(v[c + 1])) > std::abs(double(v[c])) * std::numeric_limits<float>::epsilon())
                return false;
        for (int c = 0; c < 3; ++c)
        {
            const float exponent = v[6 + c];
            if (!std::isfinite(exponent) || exponent != std::trunc(exponent) || exponent < -1073 ||
                exponent > ((p.mode == 0 || p.mode == 3) ? 128 : 1024))
                return false;
            if (v[2 * c] != 0 && (std::abs(v[2 * c]) < .5f || std::abs(v[2 * c]) > 1.0f))
                return false;
        }
        if (p.mode == 2 && (double(v[0]) + v[1] <= 0 || double(v[2]) + v[3] <= 0 || double(v[4]) + v[5] < 0))
            return false;
        if (p.mode == 3 && double(v[0]) + v[1] < 0)
            return false;
    }
    return true;
}
inline bool finite_candidate(const float* values, const ColumnRemapDispatch& d)
{
    for (std::size_t n = 0; n < cells(d); ++n)
    {
        const auto* v = values + ColumnRemapDispatch::output_stride * n;
        for (int c = 0; c < 11; ++c)
            if (!std::isfinite(v[c]))
                return false;
        for (int c = 8; c < 11; ++c)
            if (v[c] != std::trunc(v[c]) || v[c] < -4096 || v[c] > 4096)
                return false;
        if (d.constants.mode == 2 && (double(v[0]) + v[1] <= 0 || double(v[2]) + v[3] <= 0 || double(v[4]) + v[5] < 0))
            return false;
        if (d.constants.mode == 3 && (double(v[0]) + v[1] < 0 || double(v[6]) + v[7] < 0))
            return false;
    }
    return true;
}
} // namespace nimbus::compute::column_remap
