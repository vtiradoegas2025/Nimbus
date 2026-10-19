#pragma once
#include "compute/thompson_cloud_contract.hpp"

namespace nimbus::compute::thompson_deposition
{
inline std::array<const float*, 13> inputs(const ThompsonDepositionDispatch& d)
{
    return {d.temperature,
            d.pressure,
            d.density,
            d.vapor,
            d.ice,
            d.snow,
            d.graupel,
            d.ice_number,
            d.cloud_vapor_change,
            d.evaporation_high,
            d.evaporation_low,
            d.cloud_vapor_change_low,
            d.tables->values().data()};
}
inline std::array<float*, 13> outputs(const ThompsonDepositionDispatch& d)
{
    return {d.ice_change_high,
            d.ice_change_low,
            d.snow_change_high,
            d.snow_change_low,
            d.graupel_change_high,
            d.graupel_change_low,
            d.temperature_change_high,
            d.temperature_change_low,
            d.ice_number_tendency,
            d.snow_vapor_rate_high,
            d.snow_vapor_rate_low,
            d.lookup_index,
            d.status};
}
inline bool validate(const ThompsonDepositionDispatch& d, std::size_t& cells)
{
    if (!d.tables || d.nx <= 0 || d.ny <= 0 || d.nz <= 0 || !std::isfinite(d.dt) || d.dt <= 0 ||
        !std::isfinite(static_cast<float>(d.dt)) || static_cast<float>(d.dt) <= 0 || d.i_begin < 0 ||
        d.i_begin >= d.i_end || d.i_end > d.nx || d.j_begin < 0 || d.j_begin >= d.j_end || d.j_end > d.ny ||
        d.k_begin < 0 || d.k_begin >= d.k_end || d.k_end > d.nz)
        return false;
    const auto columns = static_cast<std::uint64_t>(d.nx) * d.ny;
    if (columns > static_cast<std::uint64_t>(std::numeric_limits<int>::max()) / d.nz)
        return false;
    cells          = columns * d.nz;
    const auto in  = inputs(d);
    const auto out = outputs(d);
    for (auto* p : in)
        if (!p)
            return false;
    for (std::size_t f = 0; f < out.size(); ++f)
    {
        if (!out[f])
            return false;
        for (std::size_t j = 0; j < in.size(); ++j)
            if (thompson_cloud::overlap(out[f], cells, in[j], j < 12 ? cells : d.tables->values().size()))
                return false;
        for (std::size_t j = 0; j < f; ++j)
            if (thompson_cloud::overlap(out[f], cells, out[j], cells))
                return false;
    }
    for (int i = d.i_begin; i < d.i_end; ++i)
        for (int j = d.j_begin; j < d.j_end; ++j)
            for (int k = d.k_begin; k < d.k_end; ++k)
            {
                const auto n = (static_cast<std::size_t>(i) * d.ny + j) * d.nz + k;
                for (std::size_t f = 0; f < 12; ++f)
                    if (!std::isfinite(in[f][n]) || (f < 8 && in[f][n] < 0))
                        return false;
                const double evaporation = static_cast<double>(d.evaporation_high[n]) + d.evaporation_low[n];
                if (d.temperature[n] <= 0 || d.pressure[n] <= 0 || d.density[n] <= 0 || evaporation < 0 ||
                    static_cast<double>(d.vapor[n]) + (double(d.cloud_vapor_change[n]) + d.cloud_vapor_change_low[n]) +
                            evaporation <
                        0)
                    return false;
            }
    return true;
}
inline abi::ThompsonCloudAdjustmentPushConstants constants(const ThompsonDepositionDispatch& d)
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
valid_candidate(const ThompsonDepositionDispatch& d, const std::array<const float*, 13>& fields, std::size_t cells)
{
    for (std::size_t n = 0; n < cells; ++n)
    {
        for (auto* f : fields)
            if (!std::isfinite(f[n]))
                return false;
        if (fields[12][n] != 0 || fields[11][n] < -1 || fields[11][n] >= 3520 ||
            fields[11][n] != std::floor(fields[11][n]))
            return false;
    }
    for (int i = d.i_begin; i < d.i_end; ++i)
        for (int j = d.j_begin; j < d.j_end; ++j)
            for (int k = d.k_begin; k < d.k_end; ++k)
            {
                const auto   n       = (static_cast<std::size_t>(i) * d.ny + j) * d.nz + k;
                const double ice     = static_cast<double>(fields[0][n]) + fields[1][n];
                const double snow    = static_cast<double>(fields[2][n]) + fields[3][n];
                const double graupel = static_cast<double>(fields[4][n]) + fields[5][n];
                const double vapor   = static_cast<double>(d.vapor[n]) +
                                       (double(d.cloud_vapor_change[n]) + d.cloud_vapor_change_low[n]) +
                                       d.evaporation_high[n] + d.evaporation_low[n];
                if (ice < -d.ice[n] * (1 + 2e-12) || snow < -d.snow[n] * (1 + 2e-12) ||
                    graupel < -d.graupel[n] * (1 + 2e-12) ||
                    ice + snow + graupel >
                        vapor + 2e-12 * (std::abs(vapor) + std::abs(ice) + std::abs(snow) + std::abs(graupel)) +
                            1e-30 ||
                    d.ice_number[n] + d.dt * fields[8][n] < 0)
                    return false;
            }
    return true;
}
} // namespace nimbus::compute::thompson_deposition
