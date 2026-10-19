#pragma once
#include "compute/dispatch/held_sources.hpp"
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
namespace nimbus::compute::held_sources
{
inline std::size_t cells(const HeldSourcesDispatch& d)
{
    return std::size_t(d.constants.nx) * d.constants.ny * d.constants.nz;
}
inline bool owned(const HeldSourcesDispatch& d, std::size_t n)
{
    const auto& p = d.constants;
    const int   i = int(n) / (p.ny * p.nz), j = (int(n) / p.nz) % p.ny, k = int(n) % p.nz;
    return i >= p.i_begin && i < p.i_end && j >= p.j_begin && j < p.j_end && k >= p.k_begin && k < p.k_end;
}
inline bool validate(const HeldSourcesDispatch& d)
{
    const auto& p = d.constants;
    if (p.nx < 1 || p.ny < 1 || p.nz < 1 || !d.input || !d.output ||
        std::size_t(p.nx) > std::size_t(std::numeric_limits<int>::max()) / HeldSourcesDispatch::input_stride /
                                std::size_t(p.ny) / std::size_t(p.nz))
        return false;
    if (p.i_begin < 0 || p.i_end < p.i_begin || p.i_end > p.nx || p.j_begin < 0 || p.j_end < p.j_begin ||
        p.j_end > p.ny || p.k_begin < 0 || p.k_end < p.k_begin || p.k_end > p.nz)
        return false;
    for (int flag : {p.staggered, p.periodic_x, p.periodic_y, p.pbl_enabled, p.radiation_enabled, p.cloud_enabled})
        if (flag < 0 || flag > 1)
            return false;
    if (p.pbl_enabled + p.radiation_enabled == 0 || (p.cloud_enabled && !p.pbl_enabled))
        return false;
    const std::array<float, 6> parameters{p.cp_high, p.cp_low, p.p0_high, p.p0_low, p.kappa_high, p.kappa_low};
    for (std::size_t n = 0; n < parameters.size(); n += 2)
        if (!std::isfinite(parameters[n]) || !std::isfinite(parameters[n + 1]) ||
            double(parameters[n]) + parameters[n + 1] <= 0 ||
            std::abs(double(parameters[n + 1])) >
                std::abs(double(parameters[n])) * std::numeric_limits<float>::epsilon())
            return false;
    const auto a = reinterpret_cast<std::uintptr_t>(d.input), b = reinterpret_cast<std::uintptr_t>(d.output);
    if ((a <= b && b - a < cells(d) * 11 * sizeof(float)) || (b < a && a - b < cells(d) * 7 * sizeof(float)))
        return false;
    for (std::size_t n = 0; n < cells(d); ++n)
    {
        const auto* v = d.input + 11 * n;
        for (int f = 0; f < 11; ++f)
            if (!std::isfinite(v[f]))
                return false;
        if ((p.radiation_enabled || (p.pbl_enabled && owned(d, n))) && (v[0] <= 0 || double(v[2]) + v[3] <= 0))
            return false;
        if (p.radiation_enabled && v[1] <= 0)
            return false;
        if (std::abs(double(v[3])) > std::abs(double(v[2])) * std::numeric_limits<float>::epsilon())
            return false;
    }
    return true;
}
inline bool finite_candidate(const float* values, const HeldSourcesDispatch& d)
{
    for (std::size_t n = 0; n < cells(d) * 7; ++n)
        if (!std::isfinite(values[n]))
            return false;
    return true;
}
} // namespace nimbus::compute::held_sources
