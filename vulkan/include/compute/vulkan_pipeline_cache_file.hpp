#pragma once

/**
 * @file vulkan_pipeline_cache_file.hpp
 * @brief Pure helpers for naming and validating persisted Vulkan pipeline caches.
 *
 * Pipeline-cache bytes are an opaque driver acceleration hint.  These helpers
 * deliberately validate only the standard Vulkan cache header and a strict
 * size bound; successful validation never replaces shader/ABI verification.
 */

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

namespace tmv_vulkan::pipeline_cache_file
{

constexpr std::size_t kHeaderBytes = 32;
constexpr std::size_t kMaximumBytes = 64u * 1024u * 1024u;
constexpr std::uint32_t kHeaderVersionOne = 1;

struct Identity
{
    std::uint32_t vendor_id = 0;
    std::uint32_t device_id = 0;
    std::uint32_t driver_version = 0;
    std::array<std::uint8_t, 16> uuid{};
    std::string manifest_signature;
};

inline std::string hex_u32(std::uint32_t value)
{
    std::ostringstream stream;
    stream << std::hex << std::setfill('0') << std::setw(8) << value;
    return stream.str();
}

inline std::string uuid_hex(const std::array<std::uint8_t, 16>& uuid)
{
    std::ostringstream stream;
    stream << std::hex << std::setfill('0');
    for (const std::uint8_t byte : uuid)
    {
        stream << std::setw(2) << static_cast<unsigned int>(byte);
    }
    return stream.str();
}

inline std::string filename(const Identity& identity)
{
    return "compute-v1-" + hex_u32(identity.vendor_id) + "-" +
           hex_u32(identity.device_id) + "-" +
           hex_u32(identity.driver_version) + "-" +
           uuid_hex(identity.uuid) + "-" +
           identity.manifest_signature + ".bin";
}

inline std::uint32_t read_u32(const std::vector<std::uint8_t>& bytes,
                              std::size_t offset)
{
    std::uint32_t value = 0;
    std::memcpy(&value, bytes.data() + offset, sizeof(value));
    return value;
}

inline bool valid_vulkan_header(const std::vector<std::uint8_t>& bytes,
                                const Identity& identity)
{
    if (bytes.size() < kHeaderBytes || bytes.size() > kMaximumBytes)
    {
        return false;
    }

    const std::uint32_t header_size = read_u32(bytes, 0);
    if (header_size < kHeaderBytes || header_size > bytes.size() ||
        read_u32(bytes, 4) != kHeaderVersionOne ||
        read_u32(bytes, 8) != identity.vendor_id ||
        read_u32(bytes, 12) != identity.device_id)
    {
        return false;
    }

    return std::memcmp(bytes.data() + 16, identity.uuid.data(),
                       identity.uuid.size()) == 0;
}

} // namespace tmv_vulkan::pipeline_cache_file
