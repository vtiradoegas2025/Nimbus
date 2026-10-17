#pragma once
#include "compute/thompson_cloud_contract.hpp"

namespace nimbus::compute::thompson_phase
{
inline std::array<const float*, 12> inputs(const ThompsonCloudPhaseDispatch& d)
{
    return {d.temperature,
            d.vapor,
            d.density,
            d.cloud,
            d.ice,
            d.cloud_number,
            d.ice_number,
            d.cloud_tendency,
            d.ice_tendency,
            d.cloud_number_source,
            d.ice_number_source,
            d.temperature_tendency};
}
inline std::array<float*, 6> outputs(const ThompsonCloudPhaseDispatch& d)
{
    return {d.next_cloud_tendency,
            d.next_ice_tendency,
            d.next_cloud_number_source,
            d.next_ice_number_source,
            d.next_temperature_tendency,
            d.status};
}
inline bool validate(const ThompsonCloudPhaseDispatch& d, std::size_t& cells)
{
    if (d.nx <= 0 || d.ny <= 0 || d.nz <= 0 || !std::isfinite(d.dt) || d.dt <= 0 ||
        !std::isfinite(static_cast<float>(d.dt)) || static_cast<float>(d.dt) <= 0 || d.i_begin < 0 ||
        d.i_begin >= d.i_end || d.i_end > d.nx || d.j_begin < 0 || d.j_begin >= d.j_end || d.j_end > d.ny ||
        d.k_begin < 0 || d.k_begin >= d.k_end || d.k_end > d.nz)
    {
        return false;
    }
    const auto columns = static_cast<std::uint64_t>(d.nx) * d.ny;
    if (columns > static_cast<std::uint64_t>(std::numeric_limits<int>::max()) / d.nz)
    {
        return false;
    }
    cells          = columns * d.nz;
    const auto in  = inputs(d);
    const auto out = outputs(d);
    for (const auto* field : in)
    {
        if (!field)
            return false;
    }
    for (std::size_t f = 0; f < out.size(); ++f)
    {
        if (!out[f])
            return false;
        for (auto* field : in)
        {
            if (thompson_cloud::overlap(out[f], cells, field, cells))
                return false;
        }
        for (std::size_t j = 0; j < f; ++j)
        {
            if (thompson_cloud::overlap(out[f], cells, out[j], cells))
                return false;
        }
    }
    for (int i = d.i_begin; i < d.i_end; ++i)
        for (int j = d.j_begin; j < d.j_end; ++j)
            for (int k = d.k_begin; k < d.k_end; ++k)
            {
                const auto n = (static_cast<std::size_t>(i) * d.ny + j) * d.nz + k;
                for (std::size_t f = 0; f < in.size(); ++f)
                {
                    if (!std::isfinite(in[f][n]) || (f < 7 && in[f][n] < 0))
                        return false;
                }
                if (d.temperature[n] <= 0 || d.density[n] <= 0)
                    return false;
                for (std::size_t f = 0; f < 4; ++f)
                {
                    const double value = in[f + 3][n], rate = in[f + 7][n];
                    const double candidate = value + d.dt * rate;
                    const double envelope  = 4e-7 * (value + std::abs(d.dt * rate)) + (f < 2 ? 2e-12 : 0);
                    if (!std::isfinite(candidate) || candidate < -envelope)
                        return false;
                }
            }
    return true;
}
inline abi::ThompsonCloudAdjustmentPushConstants constants(const ThompsonCloudPhaseDispatch& d)
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
            static_cast<float>(d.dt),
            thompson_cloud::timestep_low(d.dt)};
}
inline bool
valid_candidate(const ThompsonCloudPhaseDispatch& d, const std::array<const float*, 6>& fields, std::size_t cells)
{
    for (std::size_t n = 0; n < cells; ++n)
    {
        for (auto* field : fields)
        {
            if (!std::isfinite(field[n]))
                return false;
        }
        if (fields[5][n] != 0)
            return false;
    }
    const auto in = inputs(d);
    for (int i = d.i_begin; i < d.i_end; ++i)
        for (int j = d.j_begin; j < d.j_end; ++j)
            for (int k = d.k_begin; k < d.k_end; ++k)
            {
                const auto n = (static_cast<std::size_t>(i) * d.ny + j) * d.nz + k;
                for (std::size_t f = 0; f < 4; ++f)
                {
                    if (in[f + 3][n] + d.dt * fields[f][n] < 0)
                        return false;
                }
            }
    return true;
}
} // namespace nimbus::compute::thompson_phase
