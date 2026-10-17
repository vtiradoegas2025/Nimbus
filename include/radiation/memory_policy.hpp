#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace tmv::radiative_transfer
{
inline constexpr std::size_t mapped_geometry_cache_max_bytes = 256U * 1024U * 1024U;

inline std::uint64_t memory_product(std::uint64_t a, std::uint64_t b)
{
    if (b && a > std::numeric_limits<std::uint64_t>::max() / b)
        throw std::overflow_error("radiation memory plan overflow");
    return a * b;
}

inline std::uint64_t memory_sum(std::uint64_t a, std::uint64_t b)
{
    if (a > std::numeric_limits<std::uint64_t>::max() - b)
        throw std::overflow_error("radiation memory plan overflow");
    return a + b;
}

inline std::uint64_t mapped_shortwave_workspace_bytes(
    std::uint64_t columns, std::uint64_t levels, bool spectral,
    std::uint64_t cosine_nodes, std::uint64_t azimuth_nodes)
{
    const auto cells = memory_product(columns, levels);
    const auto directions = memory_product(2, memory_product(cosine_nodes, azimuth_nodes));
    const auto points = spectral ? 112U : 0U;
    // Optical arrays and directional accumulation coexist with direct-beam,
    // interface, reflection, ray and transactional result workspaces.
    const auto volume_arrays = memory_sum(memory_product(3, points),
                                         memory_sum(spectral ? directions : 0, 64));
    const auto surface_arrays = memory_sum(memory_product(8, points), 64);
    const auto arrays = memory_sum(memory_product(cells, volume_arrays),
                                  memory_product(columns, surface_arrays));
    return memory_sum(memory_product(arrays, sizeof(double)),
                      spectral ? mapped_geometry_cache_max_bytes : 0);
}
}
