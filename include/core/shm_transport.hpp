#pragma once

/**
 * @file shm_transport.hpp
 * @brief POSIX shared-memory transport for live simulation → viewer streaming.
 *
 * Enables zero-disk-I/O visualization: the simulation writes field data to
 * a named POSIX shared-memory region, and the Vulkan viewer reads it directly.
 *
 * Protocol:
 *   - The simulation creates the SHM region and writes a ShmHeader + field data.
 *   - The viewer opens the same region read-only and polls the sequence number.
 *   - A monotonically increasing sequence_number signals new data availability.
 *   - The viewer compares its last-seen sequence against the header's sequence;
 *     if different, it copies the field data and updates its display.
 *
 * Memory layout:
 *   [ShmHeader]
 *   [field_names: max_fields * 64 bytes, null-terminated C strings]
 *   [field_data[0]: nx * ny * nz * sizeof(float)]
 *   [field_data[1]: nx * ny * nz * sizeof(float)]
 *   ...
 *   [field_data[field_count-1]: ...]
 *
 * Thread safety: single writer, single reader. The sequence_number acts as
 * a release/acquire fence. The reader may observe a partially written frame
 * (torn read) but will never crash — the worst case is one frame of visual
 * artifacts before the next consistent frame arrives.
 */

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace tmv_shm
{

/// Default POSIX shared memory name (appears as /dev/shm/tmv_live on Linux,
/// or a Mach port on macOS).
static constexpr const char* kDefaultShmName = "/tmv_live";

/// Maximum number of fields that can be streamed simultaneously.
static constexpr int kMaxFields = 16;

/// Maximum field name length (including null terminator).
static constexpr int kMaxFieldNameLen = 64;

/// Magic number for header validation.
static constexpr uint32_t kShmMagic = 0x544D5653; // "TMVS"

/// Protocol version.
static constexpr uint32_t kShmVersion = 1;

/**
 * @brief Header at the start of the shared memory region.
 *
 * All fields except sequence_number are written once at creation and
 * are immutable for the lifetime of the region. sequence_number is
 * updated atomically after each frame write.
 */
struct ShmHeader
{
    uint32_t magic;            ///< Must be kShmMagic
    uint32_t version;          ///< Must be kShmVersion
    int32_t  nx;               ///< Volume width (radial, viewer X)
    int32_t  ny;               ///< Volume height (azimuthal, viewer Y)
    int32_t  nz;               ///< Volume depth (vertical, viewer Z)
    int32_t  field_count;      ///< Number of active fields (1..kMaxFields)
    double   simulation_time;  ///< Current simulation time in seconds

    /// Monotonically increasing counter. Bumped after each frame write.
    /// The reader polls this to detect new data.
    alignas(8) uint64_t sequence_number;
};

/// Byte offset of the field name table (immediately after the header).
inline std::size_t field_names_offset()
{
    return sizeof(ShmHeader);
}

/// Byte size of the field name table.
inline std::size_t field_names_size()
{
    return static_cast<std::size_t>(kMaxFields) * kMaxFieldNameLen;
}

/// Byte offset of field data for field index `i`.
inline std::size_t field_data_offset(int i, int nx, int ny, int nz)
{
    const std::size_t voxels = static_cast<std::size_t>(nx) *
                               static_cast<std::size_t>(ny) *
                               static_cast<std::size_t>(nz);
    return field_names_offset() + field_names_size() +
           static_cast<std::size_t>(i) * voxels * sizeof(float);
}

/// Total byte size of the shared memory region.
inline std::size_t total_shm_size(int nx, int ny, int nz, int field_count)
{
    return field_data_offset(field_count, nx, ny, nz);
}

/// Read the field name at index `i` from the name table.
inline std::string read_field_name(const void* shm_base, int i)
{
    const char* table = reinterpret_cast<const char*>(shm_base) + field_names_offset();
    const char* name = table + static_cast<std::size_t>(i) * kMaxFieldNameLen;
    return std::string(name); // stops at null terminator
}

/// Write a field name at index `i` into the name table.
inline void write_field_name(void* shm_base, int i, const char* name)
{
    char* table = reinterpret_cast<char*>(shm_base) + field_names_offset();
    char* dst = table + static_cast<std::size_t>(i) * kMaxFieldNameLen;
    std::strncpy(dst, name, kMaxFieldNameLen - 1);
    dst[kMaxFieldNameLen - 1] = '\0';
}

/// Get a mutable pointer to field data for field index `i`.
inline float* field_data_ptr(void* shm_base, int i, int nx, int ny, int nz)
{
    auto* base = reinterpret_cast<char*>(shm_base);
    return reinterpret_cast<float*>(base + field_data_offset(i, nx, ny, nz));
}

/// Get a const pointer to field data for field index `i`.
inline const float* field_data_ptr(const void* shm_base, int i, int nx, int ny, int nz)
{
    auto* base = reinterpret_cast<const char*>(shm_base);
    return reinterpret_cast<const float*>(base + field_data_offset(i, nx, ny, nz));
}

} // namespace tmv_shm
