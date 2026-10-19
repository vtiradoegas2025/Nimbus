#pragma once
#include "compute/thompson_cloud_contract.hpp"

namespace nimbus::compute::thompson_freezing
{
inline std::array<const float*, 16> inputs(const ThompsonFreezingDispatch& d)
{
    return {d.temperature,
            d.density,
            d.cloud,
            d.rain,
            d.cloud_number,
            d.rain_number,
            d.ice_aerosol,
            d.cloud_vapor_change,
            d.conversion_high,
            d.conversion_low,
            d.evaporation_high,
            d.evaporation_low,
            d.cloud_number_source,
            d.rain_number_source,
            d.cloud_vapor_change_low,
            d.tables->values().data()};
}
inline std::array<float*, 13> outputs(const ThompsonFreezingDispatch& d)
{
    return {d.cloud_frozen_high,
            d.cloud_frozen_low,
            d.rain_ice_high,
            d.rain_ice_low,
            d.rain_graupel_high,
            d.rain_graupel_low,
            d.next_cloud_number_source,
            d.next_rain_number_source,
            d.rain_ice_number_source,
            d.cloud_ice_number_source,
            d.rain_lookup,
            d.cloud_lookup,
            d.status};
}
inline bool validate(const ThompsonFreezingDispatch& d, std::size_t& cells)
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
        {
            if (thompson_cloud::overlap(out[f], cells, in[j], j < 15 ? cells : d.tables->values().size()))
                return false;
        }
        for (std::size_t j = 0; j < f; ++j)
            if (thompson_cloud::overlap(out[f], cells, out[j], cells))
                return false;
    }
    for (int i = d.i_begin; i < d.i_end; ++i)
        for (int j = d.j_begin; j < d.j_end; ++j)
            for (int k = d.k_begin; k < d.k_end; ++k)
            {
                const auto n = (static_cast<std::size_t>(i) * d.ny + j) * d.nz + k;
                for (std::size_t f = 0; f < 15; ++f)
                {
                    if (!std::isfinite(in[f][n]) || (f < 7 && in[f][n] < 0))
                        return false;
                }
                const double converted  = static_cast<double>(d.conversion_high[n]) + d.conversion_low[n];
                const double evaporated = static_cast<double>(d.evaporation_high[n]) + d.evaporation_low[n];
                const double cloud      = static_cast<double>(d.cloud[n]) -
                                          (double(d.cloud_vapor_change[n]) + d.cloud_vapor_change_low[n]) - converted;
                const double rain       = d.rain[n] + converted - evaporated;
                if (d.temperature[n] <= 0 || d.density[n] <= 0 || converted < 0 || evaporated < 0 || cloud < -2e-12 ||
                    rain < -2e-12 || d.cloud_number[n] + d.dt * d.cloud_number_source[n] < 0 ||
                    d.rain_number[n] + d.dt * d.rain_number_source[n] < 0)
                    return false;
            }
    return true;
}
inline abi::ThompsonCloudAdjustmentPushConstants constants(const ThompsonFreezingDispatch& d)
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
valid_candidate(const ThompsonFreezingDispatch& d, const std::array<const float*, 13>& fields, std::size_t cells)
{
    for (std::size_t n = 0; n < cells; ++n)
    {
        for (auto* f : fields)
            if (!std::isfinite(f[n]))
                return false;
        if (fields[12][n] != 0)
            return false;
        for (std::size_t f = 0; f < 6; f += 2)
            if (static_cast<double>(fields[f][n]) + fields[f + 1][n] < 0)
                return false;
    }
    for (int i = d.i_begin; i < d.i_end; ++i)
        for (int j = d.j_begin; j < d.j_end; ++j)
            for (int k = d.k_begin; k < d.k_end; ++k)
            {
                const auto   n          = (static_cast<std::size_t>(i) * d.ny + j) * d.nz + k;
                const double converted  = static_cast<double>(d.conversion_high[n]) + d.conversion_low[n];
                const double evaporated = static_cast<double>(d.evaporation_high[n]) + d.evaporation_low[n];
                const double cloud      = std::max(
                    0.0, d.cloud[n] - (double(d.cloud_vapor_change[n]) + d.cloud_vapor_change_low[n]) - converted);
                const double rain = std::max(0.0, d.rain[n] + converted - evaporated);
                const double cloud_error =
                    2e-13 * (d.cloud[n] + std::abs(double(d.cloud_vapor_change[n]) + d.cloud_vapor_change_low[n]) +
                             converted) +
                    1e-30;
                const double rain_error = 2e-13 * (d.rain[n] + converted + evaporated) + 1e-30;
                if (static_cast<double>(fields[0][n]) + fields[1][n] > cloud + cloud_error ||
                    static_cast<double>(fields[2][n]) + fields[3][n] + fields[4][n] + fields[5][n] >
                        rain + rain_error ||
                    fields[10][n] < -2 || fields[10][n] >= 3388275 || fields[10][n] != std::floor(fields[10][n]) ||
                    fields[11][n] < -2 || fields[11][n] >= 9157500 || fields[11][n] != std::floor(fields[11][n]))
                {
                    return false;
                }
                if (d.cloud_number[n] + d.dt * fields[6][n] < 0 || d.rain_number[n] + d.dt * fields[7][n] < 0 ||
                    fields[8][n] < 0 || fields[9][n] < 0)
                    return false;
            }
    return true;
}
} // namespace nimbus::compute::thompson_freezing
