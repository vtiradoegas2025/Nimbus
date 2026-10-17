#pragma once
#include "compute/dispatch/simple_grey.hpp"
#include <cmath>
#include <cstdint>
#include <limits>
namespace nimbus::compute::simple_grey
{
inline std::size_t input_stride(const SimpleGreyDispatch& d)
{
    return 34 + 14 * (std::size_t(d.constants.layers) + 1);
}
inline std::size_t input_count(const SimpleGreyDispatch& d)
{
    return std::size_t(d.constants.columns) * input_stride(d);
}
inline std::size_t output_count(const SimpleGreyDispatch& d)
{
    return std::size_t(d.constants.columns) * 18 * (std::size_t(d.constants.layers) + 1);
}
inline bool validate(const SimpleGreyDispatch& d)
{
    const auto& p = d.constants;
    if (p.columns < 1 || p.layers < 1 || p.layers > 1000000 || p.longwave < 0 || p.longwave > 1 || p.shortwave < 0 ||
        p.shortwave > 1 || p.longwave + p.shortwave == 0 || p.water < 0 || p.water > 1 || p.padding0 || p.padding1 ||
        p.padding2 || !d.input || !d.output ||
        std::size_t(p.columns) > std::size_t(std::numeric_limits<int>::max()) / (34 + 18 * (std::size_t(p.layers) + 1)))
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
        if (p.longwave && (param(0) <= 0 || param(1) < 0 || param(1) > 1))
            return false;
        if (p.shortwave && (param(2) < 0 || param(2) > 1 || param(3) < 0 || param(4) < 0 || param(4) > 1))
            return false;
        if (param(5) < 0 || param(6) <= 0 || param(7) < 0 || param(8) <= 0 || param(9) <= 0 || param(10) <= 0)
            return false;
        for (int f = 11; f < 17; ++f)
            if (param(f) < 0)
                return false;
        for (int k = 0; k <= p.layers; ++k)
        {
            const auto*  r        = v + 34 + 14 * std::size_t(k);
            const double pressure = double(r[2]) + r[3];
            if (pressure < 0 || (k == 0 && pressure <= 0) || (k > 0 && pressure > double(r[-12]) + r[-11]))
                return false;
            if (k < p.layers)
            {
                if (double(r[0]) + r[1] <= 0 || double(r[4]) + r[5] <= 0 || double(r[6]) + r[7] <= 0)
                    return false;
                for (int f = 8; f < 14; f += 2)
                    if (double(r[f]) + r[f + 1] < 0)
                        return false;
            }
        }
    }
    return true;
}
inline bool finite_candidate(const float* values, const SimpleGreyDispatch& d)
{
    for (std::size_t n = 0; n < output_count(d); ++n)
        if (!std::isfinite(values[n]))
            return false;
    for (std::size_t n = 0; n < output_count(d); n += 18)
        for (int f = 0; f < 8; f += 2)
            if (double(values[n + f]) + values[n + f + 1] < 0)
                return false;
    return true;
}
} // namespace nimbus::compute::simple_grey
