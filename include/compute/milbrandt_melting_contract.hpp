#pragma once

#include "compute/milbrandt_cloud_contract.hpp"
#include "microphysics/milbrandt_melting_parameters.hpp"

namespace nimbus::compute::milbrandt_melting
{
inline std::array<float, 32> coefficients(const MilbrandtMeltingDispatch& d)
{
    std::array<float, 32> result{};
    for (int i = 0; i < 16; ++i)
    {
        const double value =
            i < 13 ? d.parameters[i] : std::tgamma(2.5 + .5 * d.parameters[std::array<int, 3>{2, 7, 12}[i - 13]]);
        result[2 * i]     = float(value);
        result[2 * i + 1] = float(value - result[2 * i]);
    }
    return result;
}

inline std::array<const float*, 13> inputs(const MilbrandtMeltingDispatch& d, const float* table)
{
    std::array<const float*, 13> result{};
    std::copy(d.state.begin(), d.state.end(), result.begin());
    result.back() = table;
    return result;
}

inline bool validate(const MilbrandtMeltingDispatch& d, std::size_t& cells)
{
    if (d.parameters != tmv::milbrandt_detail::default_melting_parameters || d.nx <= 0 || d.ny <= 0 || d.nz <= 0 ||
        !std::isfinite(d.dt) || d.dt <= 0 || !std::isfinite(float(d.dt)) || float(d.dt) <= 0 || d.i_begin < 0 ||
        d.i_begin >= d.i_end || d.i_end > d.nx || d.j_begin < 0 || d.j_begin >= d.j_end || d.j_end > d.ny ||
        d.k_begin < 0 || d.k_begin >= d.k_end || d.k_end > d.nz)
        return false;
    const auto columns = std::uint64_t(d.nx) * d.ny;
    if (columns > std::uint64_t(std::numeric_limits<int>::max()) / d.nz)
        return false;
    cells = columns * d.nz;
    for (auto p : d.state)
        if (!p)
            return false;
    for (std::size_t f = 0; f < d.output.size(); ++f)
    {
        if (!d.output[f])
            return false;
        for (auto p : d.state)
            if (milbrandt_cloud::overlap(d.output[f], p, cells))
                return false;
        for (std::size_t q = 0; q < f; ++q)
            if (milbrandt_cloud::overlap(d.output[f], d.output[q], cells))
                return false;
    }
    for (std::size_t n = 0; n < cells; ++n)
        for (auto p : d.state)
            if (!std::isfinite(p[n]))
                return false;
    for (int i = d.i_begin; i < d.i_end; ++i)
        for (int j = d.j_begin; j < d.j_end; ++j)
            for (int k = d.k_begin; k < d.k_end; ++k)
            {
                const std::size_t n = (std::size_t(i) * d.ny + j) * d.nz + k;
                if (d.state[0][n] < 150 || d.state[0][n] > 350 || d.state[1][n] < 1000 || d.state[1][n] > 110000 ||
                    d.state[2][n] < .05f || d.state[2][n] > 2 || d.state[3][n] < 0 || d.state[3][n] > .1f)
                    return false;
                for (int f = 4; f < 12; ++f)
                    if (d.state[f][n] < 0 || d.state[f][n] > (f < 8 ? .1f : 1e12f))
                        return false;
            }
    return true;
}

inline abi::ThompsonCloudAdjustmentPushConstants constants(const MilbrandtMeltingDispatch& d)
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
valid_candidate(const MilbrandtMeltingDispatch& d, const std::array<const float*, 11>& out, std::size_t cells)
{
    constexpr double eps    = std::numeric_limits<float>::epsilon();
    constexpr double latent = physical_constants::latent_heat_fusion_jkg / physical_constants::specific_heat_cp_jkgk;
    for (std::size_t n = 0; n < cells; ++n)
    {
        const int  k = n % d.nz, j = n / d.nz % d.ny, i = n / (std::size_t(d.ny) * d.nz);
        const bool active = i >= d.i_begin && i < d.i_end && j >= d.j_begin && j < d.j_end && k >= d.k_begin &&
                            k < d.k_end && double(d.state[0][n]) > 273.15;
        for (auto p : out)
            if (!std::isfinite(p[n]) || (!active && p[n] != 0))
                return false;
        if (!active)
            continue;
        double mass = out[4][n], number = out[6][n];
        if (mass < 0 || number < 0 || out[5][n] > 0)
            return false;
        for (int f = 0; f < 4; ++f)
        {
            const double dq = out[f][n], dn = out[7 + f][n], q = d.state[4 + f][n], N = d.state[8 + f][n];
            if (dq > 0 || dn > 0 || q + d.dt * dq < -4 * eps * q || N + d.dt * dn < -4 * eps * N)
                return false;
            if (std::abs(d.dt * (dn * q - dq * N)) > 8 * eps * std::max(1e-30, N * q))
                return false;
            mass += dq;
            number += dn;
        }
        if (std::abs(mass) > 8 * eps * std::max(1e-30, double(out[4][n])) ||
            std::abs(number) > 8 * eps * std::max(1e-30, double(out[6][n])) ||
            std::abs(double(out[5][n]) + latent * out[4][n]) > 8 * eps * std::max(1e-30, std::abs(double(out[5][n]))))
            return false;
    }
    return true;
}
} // namespace nimbus::compute::milbrandt_melting
