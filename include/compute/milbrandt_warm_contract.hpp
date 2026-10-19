#pragma once

#include "compute/dispatch/microphysics.hpp"
#include "compute/kernel_abi.hpp"
#include "microphysics/milbrandt_warm_parameters.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>

namespace nimbus::compute::milbrandt_warm
{
inline std::array<float, 17> coefficients(const MilbrandtWarmRainDispatch& d)
{
    const auto p = tmv::milbrandt_detail::warm_rain_parameters(
        d.parameters[0], d.parameters[1], d.parameters[2], d.parameters[3], d.parameters[4]);
    return {float(p.alpha_r),
            float(p.c_r),
            float(p.d_r),
            float(p.a_r),
            float(p.b_r),
            float(p.inv_mu_c),
            float(p.gc1),
            float(p.gc2),
            float(p.gc3),
            float(p.gc4),
            float(p.gr31),
            float(p.gr34),
            float(p.gr37),
            float(p.inv_cloud_mass_factor),
            float(p.inv_rain_mass_factor),
            float(p.gamma_ventilation),
            float(p.gamma_fall_ventilation)};
}

inline std::array<const float*, 15> inputs(const MilbrandtWarmRainDispatch& d, const float* coefficients)
{
    std::array<const float*, 15> result{};
    std::copy(d.state.begin(), d.state.end(), result.begin());
    std::copy(d.rates.begin(), d.rates.end(), result.begin() + 8);
    result.back() = coefficients;
    return result;
}

inline bool overlap(const float* a, const float* b, std::size_t cells)
{
    const auto x = reinterpret_cast<std::uintptr_t>(a), y = reinterpret_cast<std::uintptr_t>(b);
    const auto bytes = cells * sizeof(float);
    return (x <= y ? y - x : x - y) < bytes;
}

inline bool validate(const MilbrandtWarmRainDispatch& d, std::size_t& cells)
{
    constexpr std::array<double, 5> reference{0, 523.598775598, 3, 149.1, .5};
    if (d.parameters != reference || d.nx <= 0 || d.ny <= 0 || d.nz <= 0 || !std::isfinite(d.dt) || d.dt <= 0 ||
        !std::isfinite(float(d.dt)) || float(d.dt) <= 0 || d.i_begin < 0 || d.i_begin >= d.i_end || d.i_end > d.nx ||
        d.j_begin < 0 || d.j_begin >= d.j_end || d.j_end > d.ny || d.k_begin < 0 || d.k_begin >= d.k_end ||
        d.k_end > d.nz)
        return false;
    const auto columns = std::uint64_t(d.nx) * d.ny;
    if (columns > std::uint64_t(std::numeric_limits<int>::max()) / d.nz)
        return false;
    cells = columns * d.nz;
    for (auto p : d.state)
        if (!p)
            return false;
    for (auto p : d.rates)
        if (!p)
            return false;
    for (std::size_t f = 0; f < d.output.size(); ++f)
    {
        if (!d.output[f])
            return false;
        for (auto p : d.state)
            if (overlap(d.output[f], p, cells))
                return false;
        for (auto p : d.rates)
            if (overlap(d.output[f], p, cells))
                return false;
        for (std::size_t q = 0; q < f; ++q)
            if (overlap(d.output[f], d.output[q], cells))
                return false;
    }
    for (int i = d.i_begin; i < d.i_end; ++i)
        for (int j = d.j_begin; j < d.j_end; ++j)
            for (int k = d.k_begin; k < d.k_end; ++k)
            {
                const std::size_t n = (std::size_t(i) * d.ny + j) * d.nz + k;
                for (auto p : d.state)
                    if (!std::isfinite(p[n]))
                        return false;
                for (auto p : d.rates)
                    if (!std::isfinite(p[n]))
                        return false;
                if (d.state[0][n] < 150 || d.state[0][n] > 350 || d.state[1][n] < 1000 || d.state[1][n] > 110000 ||
                    d.state[2][n] < .05f || d.state[2][n] > 2 || d.state[3][n] < 0 || d.state[3][n] > .1f ||
                    d.state[4][n] < 0 || d.state[4][n] > .1f || d.state[5][n] < 0 || d.state[5][n] > .1f ||
                    d.state[6][n] < 0 || d.state[7][n] < 0 || d.state[6][n] > 1e12f || d.state[7][n] > 1e12f ||
                    double(d.state[6][n]) + d.dt * d.rates[4][n] <
                        -4 * std::numeric_limits<float>::epsilon() * std::max(1.0, double(d.state[6][n])) ||
                    double(d.state[6][n]) + d.dt * d.rates[4][n] > 1e12 ||
                    double(d.state[7][n]) + d.dt * d.rates[5][n] < 0)
                    return false;
            }
    return true;
}

inline abi::ThompsonCloudAdjustmentPushConstants constants(const MilbrandtWarmRainDispatch& d)
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
            float(d.dt),
            float(d.dt - float(d.dt))};
}

inline bool
valid_candidate(const MilbrandtWarmRainDispatch& d, const std::array<const float*, 6>& out, std::size_t cells)
{
    for (int i = 0; i < d.nx; ++i)
        for (int j = 0; j < d.ny; ++j)
            for (int k = 0; k < d.nz; ++k)
            {
                const std::size_t n = (std::size_t(i) * d.ny + j) * d.nz + k;
                const bool        active =
                    i >= d.i_begin && i < d.i_end && j >= d.j_begin && j < d.j_end && k >= d.k_begin && k < d.k_end;
                for (std::size_t f = 0; f < out.size(); ++f)
                {
                    if (!active)
                    {
                        if (out[f][n] != d.rates[f][n])
                            return false;
                        continue;
                    }
                    if (!std::isfinite(out[f][n]))
                        return false;
                }
                if (!active)
                    continue;
                const double total = (double(out[0][n]) - d.rates[0][n]) + (double(out[1][n]) - d.rates[1][n]) +
                                     (double(out[2][n]) - d.rates[2][n]);
                const double scale = std::max(1e-12, double(d.state[3][n]) + d.state[4][n] + d.state[5][n]);
                if (std::abs(d.dt * total) > std::max(1.1e-14, 2e-6 * scale))
                    return false;
                constexpr double latent_factor =
                    physical_constants::latent_heat_vaporization_jkg / physical_constants::specific_heat_cp_jkgk;
                const double heat        = double(out[3][n]) - d.rates[3][n];
                const double evaporation = double(out[2][n]) - d.rates[2][n];
                const double heat_scale =
                    std::abs(double(out[3][n])) + std::abs(double(d.rates[3][n])) +
                    latent_factor * (std::abs(double(out[2][n])) + std::abs(double(d.rates[2][n])));
                if (std::abs(heat + latent_factor * evaporation) >
                    1e-10 + 8 * std::numeric_limits<float>::epsilon() * heat_scale)
                    return false;
                constexpr std::array<int, 5> state_fields{4, 5, 3, 6, 7};
                constexpr std::array<int, 5> rate_fields{0, 1, 2, 4, 5};
                for (std::size_t f = 0; f < state_fields.size(); ++f)
                {
                    const int    r         = rate_fields[f];
                    const double old       = d.state[state_fields[f]][n];
                    const double increment = f < 3 ? double(out[r][n]) - d.rates[r][n] : out[r][n];
                    const double candidate = old + d.dt * increment;
                    if (!std::isfinite(candidate) || candidate > std::numeric_limits<float>::max() ||
                        candidate < -2e-6 * std::max(1e-12, old))
                        return false;
                }
            }
    (void)cells;
    return true;
}
} // namespace nimbus::compute::milbrandt_warm
