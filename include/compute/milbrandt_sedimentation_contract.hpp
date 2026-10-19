#pragma once

#include "compute/milbrandt_cloud_contract.hpp"
#include "microphysics/milbrandt_sedimentation_parameters.hpp"

namespace nimbus::compute::milbrandt_sedimentation
{
inline std::array<float, 22> coefficients(const MilbrandtSedimentationDispatch& d)
{
    const auto&            p = tmv::milbrandt_detail::default_sedimentation_parameters[d.species];
    std::array<double, 11> values{};
    std::copy(p.begin(), p.end(), values.begin());
    values[7]  = std::tgamma(1 + p[0]);
    values[8]  = std::tgamma(1 + p[0] + p[2]);
    values[9]  = std::tgamma(1 + p[0] + p[2] + p[4]);
    values[10] = std::tgamma(1 + p[0] + p[4]);
    std::array<float, 22> result{};
    for (int f = 0; f < 11; ++f)
    {
        result[2 * f]     = float(values[f]);
        result[2 * f + 1] = float(values[f] - result[2 * f]);
    }
    return result;
}
inline std::array<const float*, 9>
inputs(const MilbrandtSedimentationDispatch& d, const float* scratch, const float* table)
{
    std::array<const float*, 9> result{};
    std::copy(d.state.begin(), d.state.end(), result.begin());
    result[7] = scratch;
    result[8] = table;
    return result;
}
inline bool validate(const MilbrandtSedimentationDispatch& d, std::size_t& cells)
{
    if (d.species < 0 || d.species >= 5 || d.maximum_substeps <= 0 || d.maximum_substeps > 100000 || d.nx <= 0 ||
        d.ny <= 0 || d.nz <= 0 || !std::isfinite(d.dt) || d.dt <= 0 || !std::isfinite(float(d.dt)) ||
        float(d.dt) <= 0 || d.i_begin < 0 || d.i_begin >= d.i_end || d.i_end > d.nx || d.j_begin < 0 ||
        d.j_begin >= d.j_end || d.j_end > d.ny || d.k_begin < 0 || d.k_begin >= d.k_end || d.k_end > d.nz)
        return false;
    const auto columns = std::uint64_t(d.nx) * d.ny;
    if (columns > std::uint64_t(std::numeric_limits<int>::max() / 12) / d.nz)
        return false;
    cells = columns * d.nz;
    for (auto p : d.state)
        if (!p)
            return false;
    for (std::size_t f = 0; f < 3; ++f)
    {
        if (!d.output[f])
            return false;
        for (auto p : d.state)
            if (milbrandt_cloud::overlap(p, d.output[f], cells))
                return false;
        for (std::size_t q = 0; q < f; ++q)
            if (milbrandt_cloud::overlap(d.output[q], d.output[f], cells))
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
                const std::size_t n  = (std::size_t(i) * d.ny + j) * d.nz + k;
                const double      dz = double(d.state[3][n]) + d.state[4][n];
                if (d.state[0][n] < 0 || d.state[0][n] > .1f || d.state[1][n] < 0 || d.state[1][n] > 1e12f ||
                    d.state[2][n] < .05f || d.state[2][n] > 2 || !std::isfinite(dz) || dz < 1e-9 || dz > 1e9 ||
                    double(d.state[0][n]) + d.dt * d.state[5][n] > .1 ||
                    double(d.state[1][n]) + d.dt * d.state[6][n] > 1e12)
                    return false;
            }
    return true;
}
inline abi::MilbrandtSedimentationPushConstants constants(const MilbrandtSedimentationDispatch& d)
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
            d.maximum_substeps,
            float(d.dt),
            float(d.dt - float(d.dt))};
}
inline bool
valid_candidate(const MilbrandtSedimentationDispatch& d, const std::array<const float*, 3>& out, std::size_t cells)
{
    constexpr double eps = std::numeric_limits<float>::epsilon();
    for (std::size_t n = 0; n < cells; ++n)
        for (auto p : out)
            if (!std::isfinite(p[n]))
                return false;
    for (int i = 0; i < d.nx; ++i)
        for (int j = 0; j < d.ny; ++j)
        {
            const std::size_t base    = (std::size_t(i) * d.ny + j) * d.nz;
            const bool        column  = i >= d.i_begin && i < d.i_end && j >= d.j_begin && j < d.j_end;
            double            initial = 0, final = 0, cutoff = 0;
            for (int k = 0; k < d.nz; ++k)
            {
                const std::size_t n = base + k;
                if (k != 0 && out[2][n] != 0)
                    return false;
                if (!column || k < d.k_begin || k >= d.k_end)
                {
                    if (out[0][n] != d.state[5][n] || out[1][n] != d.state[6][n])
                        return false;
                    continue;
                }
                const double mass   = double(d.state[0][n]) + d.dt * out[0][n];
                const double number = double(d.state[1][n]) + d.dt * out[1][n];
                if (!std::isfinite(mass) || !std::isfinite(number) ||
                    mass < -8 * eps * std::max(1e-14, double(d.state[0][n])) ||
                    number < -8 * eps * std::max(1e-6, double(d.state[1][n])))
                    return false;
                const double weight = double(d.state[2][n]) * (double(d.state[3][n]) + d.state[4][n]);
                initial += std::max(0.0, double(d.state[0][n]) + d.dt * d.state[5][n]) * weight;
                final += mass * weight;
                cutoff += 1e-14 * weight;
            }
            if (!column)
            {
                if (out[2][base] != 0)
                    return false;
                continue;
            }
            if (out[2][base] < 0)
                return false;
            const std::size_t bottom = base + d.k_begin;
            const double      surface =
                d.dt * out[2][base] * double(d.state[2][bottom]) * (double(d.state[3][bottom]) + d.state[4][bottom]);
            if (std::abs(final + surface - initial) >
                cutoff + 32 * eps * std::max(1e-20, initial + std::abs(final) + surface))
                return false;
        }
    return true;
}
} // namespace nimbus::compute::milbrandt_sedimentation
