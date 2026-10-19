#pragma once
#include "compute/dispatch/slab_column.hpp"
#include <cmath>
#include <cstdint>
#include <limits>
namespace nimbus::compute::slab_column
{
inline std::size_t input_stride(const SlabColumnDispatch& d)
{
    return 46 + 14 * (std::size_t(d.constants.layers) + 1);
}
inline std::size_t output_stride(const SlabColumnDispatch& d)
{
    return 22 + 8 * std::size_t(d.constants.layers);
}
inline std::size_t input_count(const SlabColumnDispatch& d)
{
    return std::size_t(d.constants.columns) * input_stride(d);
}
inline std::size_t output_count(const SlabColumnDispatch& d)
{
    return std::size_t(d.constants.columns) * output_stride(d);
}
inline bool validate(const SlabColumnDispatch& d)
{
    const auto& p = d.constants;
    if (p.columns < 1 || p.layers < 1 || p.layers > 1000000 || p.top_method < 0 || p.top_method > 1 ||
        p.surface_method < 0 || p.surface_method > 1 || p.apply_flux < 0 || p.apply_flux > 1 || p.padding0 ||
        p.padding1 || p.padding2 || !d.input || !d.output ||
        std::size_t(p.columns) > std::size_t(std::numeric_limits<int>::max()) / (46 + 14 * (std::size_t(p.layers) + 1)))
        return false;
    const auto a = reinterpret_cast<std::uintptr_t>(d.input), b = reinterpret_cast<std::uintptr_t>(d.output);
    if ((a <= b && b - a < input_count(d) * sizeof(float)) || (b < a && a - b < output_count(d) * sizeof(float)))
        return false;
    for (std::size_t n = 0; n < input_count(d); n += 2)
        if (!std::isfinite(d.input[n]) || !std::isfinite(d.input[n + 1]) ||
            std::abs(double(d.input[n + 1])) > std::abs(double(d.input[n])) * std::numeric_limits<float>::epsilon())
            return false;
    for (int column = 0; column < p.columns; ++column)
    {
        const auto* v     = d.input + std::size_t(column) * input_stride(d);
        const auto  param = [&](int f) { return double(v[2 * f]) + v[2 * f + 1]; };
        if (param(0) <= 0 || param(1) < param(0) || param(11) < 0)
            return false;
        if (p.apply_flux && (param(4) <= 0 || param(5) < 0 || param(5) > .1 || param(9) <= 0 || param(10) <= 0 ||
                             param(6) <= param(9) || param(6) <= param(10)))
            return false;
        for (int f = 12; f < 23; ++f)
            if (param(f) <= 0)
                return false;
        for (int k = 0; k <= p.layers; ++k)
        {
            const auto* r = v + 46 + 14 * std::size_t(k);
            if (k > 0 && double(r[12]) + r[13] <= double(r[-2]) + r[-1])
                return false;
            if (k < p.layers && (double(r[0]) + r[1] <= 0 || double(r[2]) + r[3] < 0 || double(r[2]) + r[3] > .1 ||
                                 double(r[8]) + r[9] <= 0 || double(r[10]) + r[11] <= 0))
                return false;
        }
    }
    return true;
}
inline bool finite_candidate(const float* values, const SlabColumnDispatch& d)
{
    for (std::size_t n = 0; n < output_count(d); ++n)
        if (!std::isfinite(values[n]))
            return false;
    for (int c = 0; c < d.constants.columns; ++c)
    {
        const auto* v = values + std::size_t(c) * output_stride(d);
        if (double(v[0]) + v[1] <= 0 || double(v[2]) + v[3] <= 0 || double(v[4]) + v[5] < 0 || double(v[6]) + v[7] < 0)
            return false;
        for (int f = 16; f < 22; f += 2)
            if (double(v[f]) + v[f + 1] < 0)
                return false;
    }
    return true;
}
} // namespace nimbus::compute::slab_column
