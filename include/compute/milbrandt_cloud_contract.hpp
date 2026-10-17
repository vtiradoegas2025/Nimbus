#pragma once

#include "compute/dispatch/microphysics.hpp"
#include "compute/kernel_abi.hpp"
#include "core/infra/physical_constants.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace nimbus::compute::milbrandt_cloud
{
inline bool overlap(const float* a, const float* b, std::size_t cells)
{
    const auto x = reinterpret_cast<std::uintptr_t>(a), y = reinterpret_cast<std::uintptr_t>(b);
    return (x <= y ? y - x : x - y) < cells * sizeof(float);
}

inline bool validate(const MilbrandtCloudAdjustmentDispatch& d, std::size_t& cells)
{
    if (d.nx <= 0 || d.ny <= 0 || d.nz <= 0 || !std::isfinite(d.dt) || d.dt <= 0 || !std::isfinite(float(d.dt)) ||
        float(d.dt) <= 0 || d.i_begin < 0 || d.i_begin >= d.i_end || d.i_end > d.nx || d.j_begin < 0 ||
        d.j_begin >= d.j_end || d.j_end > d.ny || d.k_begin < 0 || d.k_begin >= d.k_end || d.k_end > d.nz)
        return false;
    const auto columns = std::uint64_t(d.nx) * d.ny;
    if (columns > std::uint64_t(std::numeric_limits<int>::max()) / d.nz)
        return false;
    cells = columns * d.nz;
    if (d.lower_boundary_vertical_velocity &&
        (!d.face_staggered_velocity || 2 * columns > std::numeric_limits<std::uint32_t>::max() - cells))
    {
        return false;
    }
    for (auto p : d.state)
        if (!p)
            return false;
    for (std::size_t f = 0; f < d.output.size(); ++f)
    {
        if (!d.output[f])
            return false;
        if (d.lower_boundary_vertical_velocity)
        {
            const auto output   = reinterpret_cast<std::uintptr_t>(d.output[f]);
            const auto boundary = reinterpret_cast<std::uintptr_t>(d.lower_boundary_vertical_velocity);
            if (output <= boundary ? boundary - output < cells * sizeof(float)
                                   : output - boundary < columns * sizeof(double))
            {
                return false;
            }
        }
        for (auto p : d.state)
            if (overlap(d.output[f], p, cells))
                return false;
        for (std::size_t q = 0; q < f; ++q)
            if (overlap(d.output[f], d.output[q], cells))
                return false;
    }
    for (std::size_t n = 0; n < cells; ++n)
        for (auto p : d.state)
            if (!std::isfinite(p[n]))
                return false;
    if (d.lower_boundary_vertical_velocity)
    {
        for (std::size_t n = 0; n < columns; ++n)
        {
            const double velocity = d.lower_boundary_vertical_velocity[n];
            if (!std::isfinite(velocity) || std::abs(velocity) > 100)
            {
                return false;
            }
        }
    }
    for (int i = d.i_begin; i < d.i_end; ++i)
        for (int j = d.j_begin; j < d.j_end; ++j)
            for (int k = d.k_begin; k < d.k_end; ++k)
            {
                const std::size_t n = (std::size_t(i) * d.ny + j) * d.nz + k;
                if (d.state[0][n] < 150 || d.state[0][n] > 400 || d.state[1][n] < 1000 || d.state[1][n] > 110000 ||
                    d.state[2][n] < .05f || d.state[2][n] > 2 || d.state[3][n] < 0 || d.state[3][n] > .2f ||
                    d.state[4][n] < 0 || d.state[4][n] > .2f || d.state[5][n] < 0 || d.state[5][n] > 1e12f ||
                    std::abs(d.state[6][n]) > 100)
                    return false;
            }
    return true;
}

inline abi::MilbrandtCloudAdjustmentPushConstants constants(const MilbrandtCloudAdjustmentDispatch& d)
{
    return {d.nx,
            d.ny,
            d.nz,
            d.i_begin,
            d.i_end,
            d.j_begin,
            d.j_end,
            d.k_begin,
            d.k_end,
            d.face_staggered_velocity ? (d.lower_boundary_vertical_velocity ? 2 : 1) : 0,
            float(d.dt),
            float(d.dt - float(d.dt))};
}

inline std::size_t velocity_value_count(const MilbrandtCloudAdjustmentDispatch& d, std::size_t cells)
{
    return cells + (d.lower_boundary_vertical_velocity ? 2 * (cells / d.nz) : 0);
}

inline std::array<const float*, 7>
inputs(const MilbrandtCloudAdjustmentDispatch& d, std::size_t cells, std::vector<float>& velocity)
{
    auto result = d.state;
    if (d.lower_boundary_vertical_velocity)
    {
        velocity.resize(velocity_value_count(d, cells));
        std::copy_n(d.state[6], cells, velocity.data());
        for (std::size_t column = 0; column < cells / d.nz; ++column)
        {
            const double boundary            = d.lower_boundary_vertical_velocity[column];
            velocity[cells + 2 * column]     = static_cast<float>(boundary);
            velocity[cells + 2 * column + 1] = static_cast<float>(boundary - velocity[cells + 2 * column]);
        }
        result[6] = velocity.data();
    }
    return result;
}

inline bool
valid_candidate(const MilbrandtCloudAdjustmentDispatch& d, const std::array<const float*, 7>& out, std::size_t cells)
{
    constexpr double latent =
        physical_constants::latent_heat_vaporization_jkg / physical_constants::specific_heat_cp_jkgk;
    constexpr double eps = std::numeric_limits<float>::epsilon();
    for (std::size_t n = 0; n < cells; ++n)
    {
        const int k = n % d.nz, j = n / d.nz % d.ny, i = n / (std::size_t(d.ny) * d.nz);
        for (auto p : out)
            if (!std::isfinite(p[n]))
                return false;
        if (i < d.i_begin || i >= d.i_end || j < d.j_begin || j >= d.j_end || k < d.k_begin || k >= d.k_end)
        {
            if (out[0][n] != d.state[0][n] || out[1][n] != d.state[3][n] || out[2][n] != d.state[4][n])
                return false;
            for (int f = 3; f < 7; ++f)
                if (out[f][n] != 0)
                    return false;
            continue;
        }
        if (out[0][n] < 150 || out[0][n] > 400 || out[1][n] < 0 || out[2][n] < 0)
            return false;
        const double water = double(d.state[3][n]) + d.state[4][n];
        if (std::abs(double(out[1][n]) + out[2][n] - water) > 4 * eps * std::max(1e-12, water))
            return false;
        if (std::abs(double(out[4][n]) + out[5][n]) > 4 * eps * std::max(1e-12, std::abs(double(out[5][n]))))
            return false;
        const double heat = out[3][n], condensation = out[5][n];
        if (std::abs(heat - latent * condensation) > 8 * eps * std::max(1e-12, std::abs(heat)))
            return false;
        constexpr int states[3] = {0, 3, 4};
        for (int f = 0; f < 3; ++f)
        {
            const double recovered = double(d.state[states[f]][n]) + d.dt * out[f + 3][n];
            if (!std::isfinite(recovered) ||
                std::abs(recovered - out[f][n]) >
                    8 * eps * std::max({1e-12, std::abs(recovered), double(d.state[states[f]][n])}))
                return false;
        }
        const double number = double(d.state[5][n]) + d.dt * out[6][n];
        if (!std::isfinite(number) || number > std::numeric_limits<float>::max() ||
            number < -4 * eps * std::max(1.0, double(d.state[5][n])))
            return false;
        if (out[2][n] == d.state[4][n] && out[6][n] != 0)
            return false;
        if (out[2][n] > d.state[4][n] && out[6][n] < 0)
            return false;
        if (out[2][n] < d.state[4][n] && out[6][n] > 0)
            return false;
    }
    return true;
}
} // namespace nimbus::compute::milbrandt_cloud
