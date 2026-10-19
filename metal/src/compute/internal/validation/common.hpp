#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>

namespace nimbus::metal
{

inline bool checked_bytes(int nr, int nth, int nz, std::size_t& cells, std::size_t& bytes) noexcept
{
    if (nr <= 0 || nth <= 0 || nz <= 0)
        return false;
    const auto  max   = std::numeric_limits<std::size_t>::max();
    std::size_t total = static_cast<std::size_t>(nr);
    if (total > max / static_cast<std::size_t>(nth))
        return false;
    total *= static_cast<std::size_t>(nth);
    if (total > max / static_cast<std::size_t>(nz))
        return false;
    total *= static_cast<std::size_t>(nz);
    if (total > static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max()) || total > max / sizeof(float))
        return false;
    cells = total;
    bytes = total * sizeof(float);
    return true;
}

} // namespace nimbus::metal
