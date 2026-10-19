#pragma once
#include "compute/dispatch/conservative_remap.hpp"
#include "compute/kernel_abi.hpp"
#include <cmath>
#include <cstdint>
#include <limits>
namespace nimbus::compute::conservative_remap
{
inline std::size_t cells(const ConservativeRemapDispatch& d)
{
    return std::size_t(d.nx) * std::size_t(d.ny) * std::size_t(d.nz);
}
inline bool validate(const ConservativeRemapDispatch& d)
{
    if (d.nx < 3 || d.ny < 3 || d.nz < 1 || d.lower_bound < 0 || d.lower_bound > 2 ||
        std::size_t(d.nx) > std::size_t(std::numeric_limits<int>::max()) / std::size_t(d.ny) / std::size_t(d.nz))
        return false;
    const auto overlaps = [&](const float* a, const float* b)
    {
        const auto x = reinterpret_cast<std::uintptr_t>(a), y = reinterpret_cast<std::uintptr_t>(b);
        return (x <= y ? y - x : x - y) < cells(d) * sizeof(float);
    };
    for (auto p : d.input)
        if (!p)
            return false;
    for (std::size_t f = 0; f < d.output.size(); ++f)
    {
        if (!d.output[f])
            return false;
        for (auto p : d.input)
            if (overlaps(p, d.output[f]))
                return false;
        for (std::size_t g = 0; g < f; ++g)
            if (overlaps(d.output[g], d.output[f]))
                return false;
    }
    for (std::size_t n = 0; n < cells(d); ++n)
    {
        for (auto p : d.input)
            if (!std::isfinite(p[n]))
                return false;
        if (double(d.input[4][n]) + d.input[5][n] <= 0 || double(d.input[6][n]) + d.input[7][n] <= 0 ||
            (d.mixing_ratio && (d.input[2][n] <= 0 || d.input[3][n] <= 0)))
            return false;
        // Donor selection uses the leading component's sign; pairs must be normalized.
        for (int f : {0, 4, 6, 8, 10, 12, 14})
            if (std::abs(double(d.input[f + 1][n])) >
                std::abs(double(d.input[f][n])) * std::numeric_limits<float>::epsilon())
                return false;
        const double value = double(d.input[0][n]) + d.input[1][n];
        if ((d.lower_bound > 0 && value < 0) || (d.lower_bound == 2 && value == 0))
            return false;
    }
    return true;
}
inline abi::ConservativeRemapPushConstants constants(const ConservativeRemapDispatch& d)
{
    return {d.nx, d.ny, d.nz, int(d.mixing_ratio), int(d.store_float), d.lower_bound};
}
inline bool finite_candidate(const std::array<const float*, 10>& out, const ConservativeRemapDispatch& d)
{
    for (std::size_t n = 0; n < cells(d); ++n)
    {
        for (auto p : out)
            if (!std::isfinite(p[n]))
                return false;
        double value = double(out[0][n]) + out[1][n];
        if ((d.lower_bound > 0 && value < 0) || (d.lower_bound == 2 && value == 0))
            return false;
    }
    return true;
}
} // namespace nimbus::compute::conservative_remap
