#pragma once

#include "compute/milbrandt_cloud_contract.hpp"
#include <vector>

namespace nimbus::compute::milbrandt_cold
{
inline bool validate(const MilbrandtColdDispatch& d, std::size_t& cells)
{
    if (d.parameters != tmv::milbrandt_detail::default_cold_parameters || d.nx <= 0 || d.ny <= 0 || d.nz <= 0 ||
        !std::isfinite(d.dt) || d.dt <= 0 || !std::isfinite(float(d.dt)) || float(d.dt) <= 0 || d.i_begin < 0 ||
        d.i_begin >= d.i_end || d.i_end > d.nx || d.j_begin < 0 || d.j_begin >= d.j_end || d.j_end > d.ny ||
        d.k_begin < 0 || d.k_begin >= d.k_end || d.k_end > d.nz)
        return false;
    const auto columns = std::uint64_t(d.nx) * d.ny;
    if (columns > std::uint64_t(std::numeric_limits<int>::max() / 16) / d.nz)
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
            if (milbrandt_cloud::overlap(p, d.output[f], cells))
                return false;
        for (auto p : d.rates)
            if (milbrandt_cloud::overlap(p, d.output[f], cells))
                return false;
        for (std::size_t q = 0; q < f; ++q)
            if (milbrandt_cloud::overlap(d.output[q], d.output[f], cells))
                return false;
    }
    for (std::size_t n = 0; n < cells; ++n)
    {
        for (auto p : d.state)
            if (!std::isfinite(p[n]))
                return false;
        for (auto p : d.rates)
            if (!std::isfinite(p[n]))
                return false;
    }
    for (int i = d.i_begin; i < d.i_end; ++i)
        for (int j = d.j_begin; j < d.j_end; ++j)
            for (int k = d.k_begin; k < d.k_end; ++k)
            {
                const std::size_t n = (std::size_t(i) * d.ny + j) * d.nz + k;
                if (d.state[0][n] < 150 || d.state[0][n] > 350 || d.state[1][n] < 1000 || d.state[1][n] > 110000 ||
                    d.state[2][n] < .05f || d.state[2][n] > 2)
                    return false;
                for (int f = 3; f < 16; ++f)
                    if (d.state[f][n] < 0 || d.state[f][n] > (f < 10 ? .1f : 1e12f))
                        return false;
                for (int f = 0; f < 6; ++f)
                    if (double(d.state[10 + f][n]) + d.dt * d.rates[8 + f][n] > 1e12)
                        return false;
            }
    return true;
}
inline std::array<std::vector<float>, 2> pack(const MilbrandtColdDispatch& d, std::size_t cells)
{
    std::array<std::vector<float>, 2> packed{std::vector<float>(16 * cells), std::vector<float>(14 * cells)};
    for (std::size_t f = 0; f < 16; ++f)
        std::copy_n(d.state[f], cells, packed[0].data() + f * cells);
    for (std::size_t f = 0; f < 14; ++f)
        std::copy_n(d.rates[f], cells, packed[1].data() + f * cells);
    return packed;
}
inline abi::MilbrandtColdPushConstants constants(const MilbrandtColdDispatch& d)
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
            d.hail_processes ? 1 : 0,
            float(d.dt),
            float(d.dt - float(d.dt))};
}
inline bool valid_candidate(const MilbrandtColdDispatch& d, const std::array<const float*, 14>& out, std::size_t cells)
{
    constexpr double eps = std::numeric_limits<float>::epsilon();
    for (std::size_t n = 0; n < cells; ++n)
    {
        const int  k = n % d.nz, j = n / d.nz % d.ny, i = n / (std::size_t(d.ny) * d.nz);
        const bool active = i >= d.i_begin && i < d.i_end && j >= d.j_begin && j < d.j_end && k >= d.k_begin &&
                            k < d.k_end && double(d.state[0][n]) < 273.15;
        for (std::size_t f = 0; f < 14; ++f)
            if (!std::isfinite(out[f][n]) || (!active && out[f][n] != d.rates[f][n]))
                return false;
        if (!active)
            continue;
        double water = 0, scale = 0, frozen = 0;
        for (int f = 0; f < 7; ++f)
        {
            const double change = double(out[f][n]) - d.rates[f][n];
            water += change;
            if (f >= 2 && f < 6)
                frozen += change;
            scale += std::abs(double(out[f][n])) + std::abs(double(d.rates[f][n]));
        }
        if (std::abs(water) > 1e-12 + 16 * eps * scale)
            return false;
        const double heat  = (double(out[7][n]) - d.rates[7][n]) * 1004;
        const double vapor = double(out[6][n]) - d.rates[6][n];
        if (std::abs(heat + 2500000 * vapor - 334000 * frozen) >
            1e-7 +
                32 * eps * (1004 * (std::abs(double(out[7][n])) + std::abs(double(d.rates[7][n]))) + 2834000 * scale))
            return false;
        for (int f = 0; f < 7; ++f)
        {
            const int    state     = f == 6 ? 3 : 4 + f;
            const double candidate = double(d.state[state][n]) + d.dt * (double(out[f][n]) - d.rates[f][n]);
            if (!std::isfinite(candidate) || candidate < -1e-12 - 32 * eps * (double(d.state[state][n]) + d.dt * scale))
                return false;
        }
    }
    return true;
}
} // namespace nimbus::compute::milbrandt_cold
