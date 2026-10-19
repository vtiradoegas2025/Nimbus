#pragma once

#include "compute/dispatch/rk_combination.hpp"
#include "compute/kernel_abi.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

namespace nimbus::compute::rk_combination
{
inline std::array<const float*, 9> inputs(const RkCombinationDispatch& d)
{
    std::array<const float*, 9> result{d.base, d.current, d.base, d.base, d.base, d.base, d.base, d.base, d.base};
    if (d.conserved_mixing_ratio)
    {
        result[2] = d.base_density;
        result[3] = d.current_density;
        result[4] = d.candidate_density;
    }
    for (int s = 0; s < d.stages && s < 4; ++s)
        result[5 + s] = d.rhs[s];
    return result;
}
inline bool overlap(const float* a, const float* b, std::size_t cells)
{
    const auto x = reinterpret_cast<std::uintptr_t>(a), y = reinterpret_cast<std::uintptr_t>(b);
    return (x <= y ? y - x : x - y) < cells * sizeof(float);
}
inline bool validate(const RkCombinationDispatch& d)
{
    if (d.cells == 0 || d.cells > std::size_t(std::numeric_limits<int>::max()) || d.stages < 1 || d.stages > 4 ||
        (d.ssp && d.stages > 3) || !std::isfinite(d.dt) || d.dt <= 0 || !std::isfinite(float(d.dt)) || float(d.dt) <= 0)
        return false;
    for (int s = 0; s < d.stages; ++s)
        if (!std::isfinite(d.weights[s]) ||
            ((!d.ssp || d.stages == 1) && (!std::isfinite(float(d.dt * d.weights[s])) ||
                                           (d.dt * d.weights[s] != 0 && float(d.dt * d.weights[s]) == 0))))
            return false;
    const auto in = inputs(d);
    for (auto p : in)
        if (!p)
            return false;
    for (std::size_t f = 0; f < 4; ++f)
    {
        if (!d.output[f])
            return false;
        for (auto p : in)
            if (overlap(p, d.output[f], d.cells))
                return false;
        for (std::size_t g = 0; g < f; ++g)
            if (overlap(d.output[g], d.output[f], d.cells))
                return false;
    }
    for (std::size_t n = 0; n < d.cells; ++n)
    {
        for (auto p : in)
            if (!std::isfinite(p[n]))
                return false;
        if (d.conserved_mixing_ratio && (in[2][n] <= 0 || in[3][n] <= 0 || in[4][n] <= 0))
            return false;
    }
    return true;
}
inline abi::RkCombinationPushConstants constants(const RkCombinationDispatch& d)
{
    abi::RkCombinationPushConstants result{};
    result.cells     = static_cast<int>(d.cells);
    result.stages    = d.stages;
    result.ssp       = d.ssp ? 1 : 0;
    result.conserved = d.conserved_mixing_ratio ? 1 : 0;
    result.dt_high   = float(d.dt);
    result.dt_low    = float(d.dt - result.dt_high);
    float* high[]    = {&result.weight0_high, &result.weight1_high, &result.weight2_high, &result.weight3_high};
    float* low[]     = {&result.weight0_low, &result.weight1_low, &result.weight2_low, &result.weight3_low};
    for (int s = 0; s < ((!d.ssp || d.stages == 1) ? d.stages : 0); ++s)
    {
        const double scale = d.dt * d.weights[s];
        *high[s]           = float(scale);
        *low[s]            = float(scale - *high[s]);
    }
    const double fraction = d.stages == 2 ? .25 : 2.0 / 3.0;
    result.fraction_high  = float(fraction);
    result.fraction_low   = float(fraction - result.fraction_high);
    return result;
}
inline bool finite_candidate(const std::array<const float*, 4>& out, std::size_t cells)
{
    for (std::size_t n = 0; n < cells; ++n)
    {
        for (auto p : out)
            if (!std::isfinite(p[n]))
                return false;
        if (double(out[2][n]) + out[3][n] < 0)
            return false;
    }
    return true;
}
} // namespace nimbus::compute::rk_combination
