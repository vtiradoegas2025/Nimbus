#pragma once
#include "compute/dispatch/collocated_sgs.hpp"
#include "compute/kernel_abi.hpp"
#include <cmath>
#include <cstdint>
#include <limits>

namespace nimbus::compute::collocated_sgs
{
inline std::size_t cells(const CollocatedSgsDispatch& d)
{
    return std::size_t(d.nx) * std::size_t(d.ny) * std::size_t(d.nz);
}
inline bool validate(const CollocatedSgsDispatch& d)
{
    if (d.nx < 5 || d.ny < 5 || d.nz < 5 || d.phase < 0 || d.phase > 1 || d.filter_mode < 0 || d.filter_mode > 1 ||
        std::size_t(d.nx) > std::size_t(std::numeric_limits<int>::max()) / std::size_t(d.ny) / std::size_t(d.nz) ||
        (!d.horizontal && !d.vertical) || (d.moist && !d.has_qv))
        return false;
    for (float x : {d.dx, d.dy, d.Pr, d.Sc})
        if (!std::isfinite(x) || x <= 0)
            return false;
    for (float x : {d.Cs, d.nu_max, d.K_max})
        if (!std::isfinite(x) || x < 0)
            return false;
    const auto overlap = [&](const float* a, const float* b)
    {
        auto x = reinterpret_cast<std::uintptr_t>(a), y = reinterpret_cast<std::uintptr_t>(b);
        return (x < y ? y - x : x - y) < cells(d) * sizeof(float);
    };
    for (auto p : d.input)
        if (!p)
            return false;
    for (std::size_t f = 0; f < d.output.size(); ++f)
    {
        if (!d.output[f])
            return false;
        for (auto p : d.input)
            if (overlap(p, d.output[f]))
                return false;
        for (std::size_t g = 0; g < f; ++g)
            if (overlap(d.output[g], d.output[f]))
                return false;
    }
    for (int i = 1; i < d.nx - 1; ++i)
        for (int j = 1; j < d.ny - 1; ++j)
            for (int k = 1; k < d.nz - 1; ++k)
            {
                auto n = (std::size_t(i) * d.ny + j) * d.nz + k;
                for (int f = 0; f < (d.phase == 0 ? 8 : 15); ++f)
                    if (!std::isfinite(d.input[f][n]))
                        return false;
                if (d.input[3][n] <= 1e-6f || d.input[6][n] <= 0 || d.input[7][n] <= 0 ||
                    (d.has_qv && d.input[4][n] < 0) || (d.tke && d.input[5][n] < 0))
                    return false;
                if (d.phase == 1)
                {
                    for (int f = 8; f < 12; ++f)
                        if (d.input[f][n] < 0)
                            return false;
                    if (d.input[13][n] < 0)
                        return false;
                }
            }
    return true;
}
inline abi::CollocatedSgsPushConstants constants(const CollocatedSgsDispatch& d)
{
    const int flags = int(d.horizontal) | (int(d.vertical) << 1) | (int(d.periodic_x) << 2) | (int(d.periodic_y) << 3) |
                      (int(d.tke) << 4) | (int(d.stability) << 5) | (int(d.moist) << 6) | (int(d.has_qv) << 7);
    return {d.nx, d.ny, d.nz, d.phase, flags, d.filter_mode, d.dx, d.dy, d.Cs, d.Pr, d.Sc, d.nu_max, d.K_max, 0, 0, 0};
}
inline bool finite_candidate(const std::array<const float*, 8>& out, std::size_t n)
{
    for (auto p : out)
        for (std::size_t i = 0; i < n; ++i)
            if (!std::isfinite(p[i]))
                return false;
    return true;
}
} // namespace nimbus::compute::collocated_sgs
