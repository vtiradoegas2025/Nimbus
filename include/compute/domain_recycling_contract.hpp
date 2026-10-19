#pragma once

#include "compute/dispatch/domain_recycling.hpp"
#include <cstddef>
#include <cstdint>

namespace nimbus::compute::domain_shift
{
inline const float* leading_values(const HomogeneousDomainFieldShiftDispatch& d) noexcept
{
    return d.leading_field ? d.leading_field : d.leading_profile;
}

inline std::size_t leading_count(const HomogeneousDomainFieldShiftDispatch& d, std::size_t cells) noexcept
{
    return d.leading_field ? cells : static_cast<std::size_t>(d.nz);
}

inline bool valid_region(const HomogeneousDomainFieldShiftDispatch& d) noexcept
{
    const auto shifted = [](int source, int destination, int shift)
    { return std::int64_t(source) == std::int64_t(destination) + shift; };
    return d.source && d.destination && ((d.leading_profile != nullptr) != (d.leading_field != nullptr)) && d.nr > 0 &&
           d.nth > 0 && d.nz > 0 && (d.shift_i_cells != 0 || d.shift_j_cells != 0) && d.destination_i_begin >= 0 &&
           d.destination_i_begin < d.destination_i_end && d.destination_i_end <= d.nr && d.destination_j_begin >= 0 &&
           d.destination_j_begin < d.destination_j_end && d.destination_j_end <= d.nth && d.source_i_begin >= 0 &&
           d.source_i_begin < d.source_i_end && d.source_i_end <= d.nr && d.source_j_begin >= 0 &&
           d.source_j_begin < d.source_j_end && d.source_j_end <= d.nth &&
           shifted(d.source_i_begin, d.destination_i_begin, d.shift_i_cells) &&
           shifted(d.source_i_end, d.destination_i_end, d.shift_i_cells) &&
           shifted(d.source_j_begin, d.destination_j_begin, d.shift_j_cells) &&
           shifted(d.source_j_end, d.destination_j_end, d.shift_j_cells);
}
} // namespace nimbus::compute::domain_shift
