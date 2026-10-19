#pragma once

#include "common.hpp"
#include "compute/domain_recycling_contract.hpp"
#include "compute/compute_backend.hpp"

#include <cstdint>

namespace nimbus::metal
{

inline bool
valid_dispatch(const HomogeneousDomainFieldShiftDispatch& dispatch, std::size_t& cells, std::size_t& bytes) noexcept
{
    if (!nimbus::compute::domain_shift::valid_region(dispatch))
    {
        return false;
    }
    return checked_bytes(dispatch.nr, dispatch.nth, dispatch.nz, cells, bytes) &&
           cells <= std::numeric_limits<std::uint32_t>::max();
}

} // namespace nimbus::metal
