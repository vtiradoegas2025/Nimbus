#pragma once
#include "compute/dispatch/conservative_reduction.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
namespace nimbus::compute::conservative_reduction
{
inline std::size_t input_values(const ConservativeReductionDispatch& d)
{
    return std::size_t(d.constants.rows) * 4 + std::size_t(d.constants.terms) * 6;
}
inline std::size_t output_values(const ConservativeReductionDispatch& d)
{
    return std::size_t(d.constants.rows) * 5;
}
inline bool validate(const ConservativeReductionDispatch& d)
{
    const auto& p = d.constants;
    if (!d.input || !d.output || p.rows < 1 || p.rows > 16777216 || p.terms < 1 || p.terms > 16777216)
        return false;
    const auto a = reinterpret_cast<std::uintptr_t>(d.input), b = reinterpret_cast<std::uintptr_t>(d.output);
    if ((a <= b && b - a < input_values(d) * sizeof(float)) || (b < a && a - b < output_values(d) * sizeof(float)))
        return false;
    for (int row = 0; row < p.rows; ++row)
        for (int field : {0, 2})
        {
            const double begin = d.input[row * 4 + field], count = d.input[row * 4 + field + 1];
            if (!std::isfinite(begin) || !std::isfinite(count) || begin < 0 || count < 0 ||
                begin != std::floor(begin) || count != std::floor(count) || begin + count > p.terms)
                return false;
        }
    for (std::size_t n = std::size_t(p.rows) * 4; n < input_values(d); n += 2)
        if (!std::isfinite(d.input[n]) || !std::isfinite(d.input[n + 1]) ||
            std::abs(double(d.input[n + 1])) > std::abs(double(d.input[n])) * std::numeric_limits<float>::epsilon())
            return false;
    return true;
}
inline bool finite_candidate(const float* values, const ConservativeReductionDispatch& d)
{
    return std::all_of(values, values + output_values(d), [](float value) { return std::isfinite(value); });
}
} // namespace nimbus::compute::conservative_reduction
