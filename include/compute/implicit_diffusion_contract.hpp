#pragma once
#include "compute/dispatch/diffusion.hpp"
#include "compute/kernel_abi.hpp"
#include <array>
#include <cmath>
#include <cstdint>

namespace nimbus::compute::implicit_contract
{
inline bool dimensions(int nx, int ny, int nz, std::size_t& cells, std::size_t& columns)
{
    if (nx < 3 || ny < 3 || nz < 2)
        return false;
    const auto xy = static_cast<std::uint64_t>(nx) * ny;
    if (xy > static_cast<std::uint64_t>(std::numeric_limits<int>::max()) / nz)
        return false;
    columns = static_cast<std::size_t>(xy);
    cells   = columns * nz;
    return true;
}
inline bool overlap(const float* a, std::size_t na, const float* b, std::size_t nb)
{
    auto x = reinterpret_cast<std::uintptr_t>(a), y = reinterpret_cast<std::uintptr_t>(b);
    return x <= y ? y - x < na * sizeof(float) : x - y < nb * sizeof(float);
}
inline bool physical(const DiffusionDispatch& d, int i, int j, int k, bool scalar = false)
{
    const int location = scalar ? 0 : static_cast<int>(d.location);
    const int i_end    = d.nx - (d.c_grid && location == 1 && !d.periodic_x ? 2 : 1);
    const int j_end    = d.ny - (d.c_grid && location == 2 && !d.periodic_y ? 2 : 1);
    const int k_begin  = d.c_grid ? 0 : 1;
    const int k_end    = d.c_grid && location == 0 ? d.nz : d.nz - 1;
    return i >= 1 && i < i_end && j >= 1 && j < j_end && k >= k_begin && k < k_end;
}
template <std::size_t I, std::size_t O>
bool buffers(const std::array<const float*, I>& in,
             const std::array<std::size_t, I>&  ni,
             const std::array<float*, O>&       out,
             const std::array<std::size_t, O>&  no,
             bool                               check_values = true)
{
    for (std::size_t i = 0; i < I; ++i)
    {
        if (!in[i])
            return false;
        for (std::size_t n = 0; n < ni[i]; ++n)
            if (check_values && !std::isfinite(in[i][n]))
                return false;
    }
    for (std::size_t i = 0; i < O; ++i)
    {
        if (!out[i])
            return false;
        for (std::size_t j = 0; j < i; ++j)
            if (overlap(out[i], no[i], out[j], no[j]))
                return false;
        for (std::size_t j = 0; j < I; ++j)
            if (overlap(out[i], no[i], in[j], ni[j]))
                return false;
    }
    return true;
}
inline bool validate(const ImplicitDiffusionStencilDispatch& d, std::size_t& n, std::size_t& c)
{
    const auto& a        = d.diffusion;
    const int   location = static_cast<int>(a.location);
    if (!dimensions(a.nx, a.ny, a.nz, n, c) || location < 0 || location > 3 || (!a.c_grid && a.nz < 3) ||
        (a.c_grid && location == 1 && !a.periodic_x && a.nx < 4) ||
        (a.c_grid && location == 2 && !a.periodic_y && a.ny < 4))
        return false;
    for (float value : {a.dx, a.dy, d.dt, d.field_scale, d.density_scale, d.thickness_scale})
        if (!std::isfinite(value) || value <= 0)
            return false;
    for (float value : {a.K_h, a.K_v})
        if (!std::isfinite(value) || value < 0)
            return false;
    if (!std::isfinite(a.dx * a.dx) || a.dx * a.dx <= 0 || !std::isfinite(a.dy * a.dy) || a.dy * a.dy <= 0)
        return false;
    const float* vk = a.vertical_K ? a.vertical_K : a.rho;
    if (!buffers(std::array<const float*, 4>{a.field, a.rho, a.dz, vk},
                 std::array<std::size_t, 4>{n, n, static_cast<std::size_t>(a.nz), n},
                 std::array<float*, 5>{a.tendency, d.diagonal, d.dot, d.preconditioned_norm, d.norm},
                 std::array<std::size_t, 5>{n, n, c, c, c},
                 false))
        return false;
    for (int k = 0; k < a.nz; ++k)
        if (!std::isfinite(a.dz[k]) || a.dz[k] <= 0 || !std::isfinite(a.dz[k] * a.dz[k]) || a.dz[k] * a.dz[k] <= 0)
            return false;
    for (int i = 0; i < a.nx; ++i)
        for (int j = 0; j < a.ny; ++j)
            for (int k = 0; k < a.nz; ++k)
            {
                const auto index = (static_cast<std::size_t>(i) * a.ny + j) * a.nz + k;
                if (physical(a, i, j, k) && !std::isfinite(a.field[index]))
                    return false;
                if (physical(a, i, j, k, true) &&
                    (!std::isfinite(a.rho[index]) || a.rho[index] <= 0 || a.rho[index] / d.density_scale <= 0 ||
                     (a.vertical_K && (!std::isfinite(a.vertical_K[index]) || a.vertical_K[index] < 0))))
                    return false;
            }
    return true;
}
inline bool validate(const ImplicitDiffusionVectorDispatch& d, std::size_t& n, std::size_t& c)
{
    const int stage = static_cast<int>(d.stage);
    if (!dimensions(d.nx, d.ny, d.nz, n, c) || stage < 0 || stage > 4 || !std::isfinite(d.alpha) ||
        !std::isfinite(d.beta) || d.alpha < 0 || d.beta < 0 || !std::isfinite(d.dt) || d.dt <= 0 ||
        !std::isfinite(d.field_scale) || d.field_scale <= 0)
        return false;
    if (!buffers(
            std::array<const float*, 7>{d.original, d.rhs, d.diagonal, d.solution, d.residual, d.direction, d.image},
            std::array<std::size_t, 7>{n, n, n, n, n, n, n},
            std::array<float*, 5>{d.next_solution, d.next_residual, d.next_direction, d.preconditioned_norm, d.norm},
            std::array<std::size_t, 5>{n, n, n, c, c},
            false))
        return false;
    for (std::size_t i = 0; i < n; ++i)
    {
        if (!std::isfinite(d.diagonal[i]) || d.diagonal[i] < 0)
            return false;
        if (d.diagonal[i] > 0)
            for (const float* field : {d.original, d.rhs, d.solution, d.residual, d.direction, d.image})
                if (!std::isfinite(field[i]))
                    return false;
    }
    return true;
}
inline abi::ImplicitDiffusionStencilPushConstants constants(const ImplicitDiffusionStencilDispatch& d)
{
    const auto& a = d.diffusion;
    return {a.nx,
            a.ny,
            a.nz,
            static_cast<int>(a.location),
            a.c_grid,
            a.periodic_x,
            a.periodic_y,
            a.vertical_K != nullptr,
            1 / (a.dx * a.dx),
            1 / (a.dy * a.dy),
            a.K_h,
            a.K_v,
            d.dt,
            d.form_rhs,
            d.field_scale,
            d.density_scale,
            d.thickness_scale,
            0,
            0,
            0};
}
inline abi::ImplicitDiffusionVectorPushConstants constants(const ImplicitDiffusionVectorDispatch& d)
{
    return {d.nx, d.ny, d.nz, static_cast<int>(d.stage), d.alpha, d.beta, d.dt, d.field_scale};
}
} // namespace nimbus::compute::implicit_contract
