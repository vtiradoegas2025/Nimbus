#pragma once

#include "compute/dispatch/turbulence.hpp"
#include "compute/kernel_abi.hpp"

#include <array>
#include <cmath>
#include <cstdint>
#include <limits>

namespace nimbus::compute::tke_contract
{
inline bool
dimensions(int nx, int ny, int nz, int profile, float dx, float dy, bool horizontal, bool vertical, std::size_t& cells)
{
    if (nx < 3 || ny < 3 || nz < 2 || profile < nz || !std::isfinite(dx) || dx <= 0 || !std::isfinite(dy) || dy <= 0 ||
        (!horizontal && !vertical))
        return false;
    const auto limit   = static_cast<std::uint64_t>(std::numeric_limits<int>::max());
    const auto columns = static_cast<std::uint64_t>(nx) * ny;
    if (columns > limit / static_cast<std::uint64_t>(nz))
        return false;
    cells = static_cast<std::size_t>(columns * nz);
    return true;
}

template <std::size_t N>
bool destinations(const std::array<const float*, 7>& inputs,
                  const std::array<float*, N>&       outputs,
                  std::size_t                        cells,
                  int                                nz)
{
    const auto overlaps = [](const float* a, std::size_t na, const float* b, std::size_t nb)
    {
        const auto x = reinterpret_cast<std::uintptr_t>(a);
        const auto y = reinterpret_cast<std::uintptr_t>(b);
        return x <= y ? y - x < na * sizeof(float) : x - y < nb * sizeof(float);
    };
    for (const auto* field : inputs)
        if (field == nullptr)
            return false;
    for (std::size_t o = 0; o < N; ++o)
    {
        if (outputs[o] == nullptr)
            return false;
        for (std::size_t p = 0; p < o; ++p)
            if (overlaps(outputs[o], cells, outputs[p], cells))
                return false;
        for (std::size_t i = 0; i < inputs.size(); ++i)
            if (overlaps(outputs[o], cells, inputs[i], i == 6 ? nz : cells))
                return false;
    }
    for (int k = 0; k < nz; ++k)
        if (!std::isfinite(inputs[6][k]) || inputs[6][k] <= 0)
            return false;
    for (std::size_t n = 0; n < cells; ++n)
        for (std::size_t f = 0; f < 6; ++f)
            if (!std::isfinite(inputs[f][n]))
                return false;
    return true;
}

inline bool validate(const TkeClosureDispatch& d, std::size_t& cells)
{
    if (!dimensions(d.nx, d.ny, d.nz, d.profile_length, d.dx, d.dy, d.use_horizontal, d.use_vertical, cells) ||
        d.filter_mode < 0 || d.filter_mode > 1 || !std::isfinite(d.Pr_t) || d.Pr_t <= 0 || !std::isfinite(d.Sc_t) ||
        d.Sc_t <= 0 || !std::isfinite(d.nu_max) || d.nu_max < 0 || !std::isfinite(d.K_max) || d.K_max < 0 ||
        !destinations(std::array<const float*, 7>{d.u, d.v, d.w, d.theta, d.qv, d.tke, d.dz},
                      std::array<float*, 8>{
                          d.nu, d.K_theta, d.K_q, d.K_tke, d.shear, d.buoyancy, d.dissipation, d.effective_prandtl},
                      cells,
                      d.nz))
        return false;
    for (std::size_t n = 0; n < cells; ++n)
        if (d.theta[n] <= 1e-6f || d.qv[n] < 0 || d.tke[n] < 0)
            return false;
    return true;
}

inline bool validate(const TkeTendencyDispatch& d, std::size_t& cells)
{
    if (!dimensions(d.nx, d.ny, d.nz, d.profile_length, d.dx, d.dy, d.use_horizontal, d.use_vertical, cells) ||
        !destinations(std::array<const float*, 7>{d.rho, d.tke, d.K_tke, d.shear, d.buoyancy, d.dissipation, d.dz},
                      std::array<float*, 1>{d.tendency},
                      cells,
                      d.nz))
        return false;
    for (std::size_t n = 0; n < cells; ++n)
        if (d.rho[n] <= 0 || d.tke[n] < 0 || d.K_tke[n] < 0 || d.shear[n] < 0 || d.dissipation[n] < 0)
            return false;
    return true;
}

inline abi::TkeClosurePushConstants constants(const TkeClosureDispatch& d)
{
    return {d.nx,
            d.ny,
            d.nz,
            d.filter_mode,
            d.use_horizontal,
            d.use_vertical,
            d.periodic_x,
            d.periodic_y,
            d.dx,
            d.dy,
            d.Pr_t,
            d.Sc_t,
            d.nu_max,
            d.K_max,
            d.stability,
            d.moist_buoyancy};
}
inline abi::TkeTendencyPushConstants constants(const TkeTendencyDispatch& d)
{
    return {d.nx, d.ny, d.nz, d.use_horizontal, d.use_vertical, d.periodic_x, d.periodic_y, turbulence_closed_lateral_mask(d.open_lateral), d.dx, d.dy};
}
} // namespace nimbus::compute::tke_contract
