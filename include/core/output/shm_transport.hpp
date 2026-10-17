#pragma once

/**
 * @file shm_transport.hpp
 * @brief Cross-platform shared-memory transport for live simulation → viewer streaming.
 *
 * Enables zero-disk-I/O visualization: the simulation writes field data to
 * a named shared-memory region, and the Vulkan viewer reads it directly. Unix
 * uses shm_open/mmap; Windows uses a named paging-file mapping.
 *
 * Protocol:
 *   - The simulation creates the SHM region and writes a ShmHeader + field data.
 *   - The viewer opens the same region read-only and polls the sequence number.
 *   - Fields are transported as raw float32 values. Display normalization is a
 *     reader concern and is described by per-field metadata.
 *   - sequence_number is a process-shared seqlock: odd means a write is in
 *     progress, even means the payload and simulation_time form a publication.
 *   - A reader accepts a copy only when equal, even sequence values bracket it.
 *
 * Memory layout:
 *   [ShmHeader]
 *   [field_names: max_fields * 64 bytes, null-terminated C strings]
 *   [field_metadata: max_fields * sizeof(ShmFieldMetadata)]
 *   [field_data[0]: nx * ny * nz * sizeof(float)]
 *   [field_data[1]: nx * ny * nz * sizeof(float)]
 *   ...
 *   [field_data[field_count-1]: ...]
 *
 * Synchronization: one writer, any number of readers. ShmHeader::sequence_number
 * is required to be an always-lock-free uint64 atomic because library-backed
 * atomics are not a valid process-shared primitive. The writer brackets the
 * complete field bundle and simulation time; readers retry instead of accepting
 * a torn field. Independent per-field readers can still select different valid
 * commits, so composite consumers must additionally coordinate one sequence.
 */

#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <string_view>
#include <type_traits>

namespace tmv_shm
{

/// Stable logical transport name. Unix exposes it through POSIX SHM; Windows
/// converts it to a per-session Local\\Nimbus_* file-mapping object.
static constexpr const char* kDefaultShmName = "/tmv_live";

inline bool valid_logical_shm_name(std::string_view name)
{
    if (name.size() < 2 || name.size() > 30 || name.front() != '/')
        return false;
    for (std::size_t index = 1; index < name.size(); ++index)
    {
        const unsigned char byte = static_cast<unsigned char>(name[index]);
        if (!((byte >= 'a' && byte <= 'z') ||
              (byte >= 'A' && byte <= 'Z') ||
              (byte >= '0' && byte <= '9') || byte == '_' || byte == '-'))
            return false;
    }
    return true;
}

#if defined(_WIN32)
inline std::wstring windows_mapping_name(std::string_view logical_name)
{
    std::wstring result = L"Local\\Nimbus_";
    for (const unsigned char byte : logical_name)
    {
        if ((byte >= 'a' && byte <= 'z') || (byte >= 'A' && byte <= 'Z') ||
            (byte >= '0' && byte <= '9') || byte == '_' || byte == '-')
        {
            result.push_back(static_cast<wchar_t>(byte));
        }
    }
    return result;
}
#endif

/// Maximum number of fields that can be streamed simultaneously.
static constexpr int kMaxFields = 16;

/// Maximum field name length (including null terminator).
static constexpr int kMaxFieldNameLen = 64;

/// Magic number for header validation.
static constexpr uint32_t kShmMagic = 0x544D5653; // "TMVS"

/// Protocol version. Version 5 adds an immutable publication generation.
static constexpr uint32_t kShmVersion = 5;
static constexpr std::size_t kPublicationGenerationLength = 33;

/** @brief Inclusive physical scalar-cell bounds within the stored payload. */
struct ShmPhysicalDomain
{
    int32_t lower[3] = {0, 0, 0};
    int32_t upper[3] = {-1, -1, -1};
};

/** @brief Cartesian scalar-cell spacing in metres. */
struct ShmGridSpacing
{
    double x_m = 0.0;
    double y_m = 0.0;
    double z_m = 0.0;
};

enum class ShmDomainMotionMode : uint32_t
{
    Stationary = 0,
    ConstantVelocity = 1,
};

/**
 * Horizontal coordinate and wind semantics for one live publication.
 *
 * Winds in field payloads remain Earth-relative. Horizontal transport inside
 * the model uses Earth wind minus frame_velocity_earth_ms.
 */
struct ShmHorizontalFrame
{
    uint32_t struct_size = sizeof(ShmHorizontalFrame);
    ShmDomainMotionMode motion_mode = ShmDomainMotionMode::Stationary;
    double grid_origin_t0_earth_m[2] = {0.0, 0.0};
    double grid_origin_earth_m[2] = {0.0, 0.0};
    double frame_velocity_earth_ms[2] = {0.0, 0.0};
};

inline bool horizontal_frame_valid(const ShmHorizontalFrame& frame)
{
    if (frame.struct_size != sizeof(ShmHorizontalFrame) ||
        (frame.motion_mode != ShmDomainMotionMode::Stationary &&
         frame.motion_mode != ShmDomainMotionMode::ConstantVelocity))
        return false;
    for (double value : {frame.grid_origin_t0_earth_m[0],
                         frame.grid_origin_t0_earth_m[1],
                         frame.grid_origin_earth_m[0],
                         frame.grid_origin_earth_m[1],
                         frame.frame_velocity_earth_ms[0],
                         frame.frame_velocity_earth_ms[1]})
        if (!std::isfinite(value)) return false;
    return frame.motion_mode != ShmDomainMotionMode::Stationary ||
           (frame.frame_velocity_earth_ms[0] == 0.0 &&
            frame.frame_velocity_earth_ms[1] == 0.0);
}

inline ShmPhysicalDomain full_physical_domain(int nx, int ny, int nz)
{
    return {{0, 0, 0}, {nx - 1, ny - 1, nz - 1}};
}

inline bool physical_domain_valid(const ShmPhysicalDomain& domain,
                                  int nx, int ny, int nz)
{
    const int dimensions[3] = {nx, ny, nz};
    for (int axis = 0; axis < 3; ++axis)
    {
        if (dimensions[axis] <= 0 || domain.lower[axis] < 0 ||
            domain.upper[axis] < domain.lower[axis] ||
            domain.upper[axis] >= dimensions[axis])
            return false;
    }
    return true;
}

inline bool grid_spacing_valid(const ShmGridSpacing& spacing)
{
    return spacing.x_m > 0.0 && spacing.y_m > 0.0 && spacing.z_m > 0.0 &&
           spacing.x_m < std::numeric_limits<double>::infinity() &&
           spacing.y_m < std::numeric_limits<double>::infinity() &&
           spacing.z_m < std::numeric_limits<double>::infinity();
}

/// On-wire scalar representation.
enum class ShmValueEncoding : uint8_t
{
    RawFloat32 = 1,
};

/// Whether zero is a neutral midpoint or the lower end of the display scale.
enum class ShmFieldPolarity : uint8_t
{
    Unipolar = 0,
    Bipolar = 1,
};

/// Grid location of transported voxels. The runtime publishes cell-centred fields.
enum class ShmGridLocation : uint8_t
{
    CellCenter = 1,
};

/// Treatment of non-finite source samples.
enum class ShmNonfinitePolicy : uint8_t
{
    PreserveAndSanitizeInReader = 1,
};

/// Display scaling requested by the producer.
enum class ShmDisplayScaling : uint8_t
{
    RobustPerFrame = 1,
};

static constexpr int kMaxUnitLen = 24;

/**
 * @brief Immutable descriptor for one raw field payload.
 *
 * A zero display_min/display_max pair means that the reader computes the
 * robust display range. Physical values themselves are never rewritten by the
 * transport.
 */
struct ShmFieldMetadata
{
    uint32_t struct_size = sizeof(ShmFieldMetadata);
    ShmValueEncoding encoding = ShmValueEncoding::RawFloat32;
    ShmFieldPolarity polarity = ShmFieldPolarity::Unipolar;
    ShmGridLocation grid_location = ShmGridLocation::CellCenter;
    ShmNonfinitePolicy nonfinite_policy =
        ShmNonfinitePolicy::PreserveAndSanitizeInReader;
    ShmDisplayScaling display_scaling = ShmDisplayScaling::RobustPerFrame;
    uint8_t reserved_flags[3] = {};
    float display_min = 0.0f;
    float display_max = 0.0f;
    char units[kMaxUnitLen] = {};
};

static_assert(std::is_trivially_copyable<ShmFieldMetadata>::value,
              "SHM field metadata must have a stable byte representation");

/**
 * @brief Header at the start of the shared memory region.
 *
 * Geometry and field descriptors are immutable after creation. The atomic
 * sequence brackets all mutable field data and simulation_time.
 */
struct ShmHeader
{
    uint32_t magic;            ///< Must be kShmMagic
    uint32_t version;          ///< Must be kShmVersion
    int32_t  nx;               ///< Volume width (radial, viewer X)
    int32_t  ny;               ///< Volume height (azimuthal, viewer Y)
    int32_t  nz;               ///< Volume depth (vertical, viewer Z)
    int32_t  field_count;      ///< Number of active fields (1..kMaxFields)
    uint32_t header_size;      ///< sizeof(ShmHeader) for ABI validation
    uint32_t reserved;         ///< Must be zero
    char publication_generation[kPublicationGenerationLength] = {};
    ShmPhysicalDomain physical_domain; ///< Inclusive atmospheric-cell ownership
    ShmGridSpacing grid_spacing_m; ///< Cartesian scalar-cell spacing
    ShmHorizontalFrame horizontal_frame; ///< Protected live Earth/grid transform

    /// Process-shared seqlock. Odd = writer active; even = coherent snapshot.
    alignas(8) std::atomic<uint64_t> sequence_number;
    double simulation_time;    ///< Time protected by sequence_number
};

inline bool publication_generation_valid(std::string_view generation)
{
    if (generation.size() + 1 != kPublicationGenerationLength)
        return false;
    for (const unsigned char byte : generation)
    {
        if (!((byte >= '0' && byte <= '9') ||
              (byte >= 'a' && byte <= 'f') ||
              (byte >= 'A' && byte <= 'F')))
            return false;
    }
    return true;
}

static_assert(std::atomic<uint64_t>::is_always_lock_free,
              "SHM v2 requires an always-lock-free 64-bit interprocess atomic");
static_assert(std::is_standard_layout<ShmHeader>::value,
              "SHM header must remain standard-layout across producer and consumer");

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

/// Byte offset of the field metadata table.
inline std::size_t field_metadata_offset()
{
    return field_names_offset() + field_names_size();
}

/// Byte size of the fixed-capacity metadata table.
inline std::size_t field_metadata_size()
{
    return static_cast<std::size_t>(kMaxFields) * sizeof(ShmFieldMetadata);
}

/// Byte offset of field data for field index `i`.
inline std::size_t field_data_offset(int i, int nx, int ny, int nz)
{
    const std::size_t voxels = static_cast<std::size_t>(nx) *
                               static_cast<std::size_t>(ny) *
                               static_cast<std::size_t>(nz);
    return field_metadata_offset() + field_metadata_size() +
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
    const void* terminator = std::memchr(name, '\0', kMaxFieldNameLen);
    const std::size_t length = terminator == nullptr
        ? static_cast<std::size_t>(kMaxFieldNameLen)
        : static_cast<std::size_t>(static_cast<const char*>(terminator) - name);
    return std::string(name, length);
}

/// Whether field index `i` has a terminator inside its fixed-width name slot.
inline bool field_name_is_terminated(const void* shm_base, int i)
{
    const char* table = reinterpret_cast<const char*>(shm_base) + field_names_offset();
    const char* name = table + static_cast<std::size_t>(i) * kMaxFieldNameLen;
    return std::memchr(name, '\0', kMaxFieldNameLen) != nullptr;
}

/// Write a field name at index `i` into the name table.
inline void write_field_name(void* shm_base, int i, const char* name)
{
    char* table = reinterpret_cast<char*>(shm_base) + field_names_offset();
    char* dst = table + static_cast<std::size_t>(i) * kMaxFieldNameLen;
    std::strncpy(dst, name, kMaxFieldNameLen - 1);
    dst[kMaxFieldNameLen - 1] = '\0';
}

/// Get immutable metadata for field index `i`.
inline const ShmFieldMetadata* field_metadata_ptr(const void* shm_base, int i)
{
    const auto* base = reinterpret_cast<const char*>(shm_base);
    return reinterpret_cast<const ShmFieldMetadata*>(
        base + field_metadata_offset() +
        static_cast<std::size_t>(i) * sizeof(ShmFieldMetadata));
}

/// Get mutable metadata for field index `i`.
inline ShmFieldMetadata* field_metadata_ptr(void* shm_base, int i)
{
    auto* base = reinterpret_cast<char*>(shm_base);
    return reinterpret_cast<ShmFieldMetadata*>(
        base + field_metadata_offset() +
        static_cast<std::size_t>(i) * sizeof(ShmFieldMetadata));
}

/** @brief Shared name-based signedness contract for disk and live readers. */
inline bool field_name_is_bipolar(std::string_view name)
{
    if (name == "w" || name == "u" || name == "v") return true;
    if (name.find("magnitude") != std::string_view::npos) return false;

    constexpr std::string_view kBipolarSubstrings[] = {
        "vorticity", "divergence", "buoyancy", "_prime",
        "pressure_gradient_force", "helicity", "stretching",
        "tilting", "baroclinic", "okubo",
    };
    for (const std::string_view token : kBipolarSubstrings)
    {
        if (name.find(token) != std::string_view::npos) return true;
    }
    return false;
}

/** @brief Best-known physical units for live field metadata. */
inline const char* field_units(std::string_view name)
{
    if (name == "u" || name == "v" || name == "w") return "m s-1";
    if (name == "theta" || name == "temperature") return "K";
    if (name == "p" || name == "pressure") return "Pa";
    if (name == "rho" || name == "density") return "kg m-3";
    // `radar` is the model's linear reflectivity factor Z.  Only the
    // separately derived `reflectivity_dbz` field has undergone 10*log10.
    // Keeping these units distinct prevents a live consumer from presenting
    // linear values as an apparently plausible dBZ product.
    if (name == "reflectivity_dbz") return "dBZ";
    if (name == "radar") return "mm^6 m-3";
    if (name.find("vorticity") != std::string_view::npos ||
        name.find("divergence") != std::string_view::npos)
    {
        return "s-1";
    }
    if (name.find("buoyancy") != std::string_view::npos) return "m s-2";
    if (name == "qc" || name == "qr" || name == "qi" || name == "qs" ||
        name == "qg" || name == "qh" || name == "qv")
    {
        return "kg kg-1";
    }
    return "unknown";
}

/** @brief Construct the immutable v2 descriptor for a named field. */
inline ShmFieldMetadata make_field_metadata(std::string_view name)
{
    ShmFieldMetadata metadata{};
    metadata.polarity = field_name_is_bipolar(name)
        ? ShmFieldPolarity::Bipolar
        : ShmFieldPolarity::Unipolar;
    std::strncpy(metadata.units, field_units(name), kMaxUnitLen - 1);
    metadata.units[kMaxUnitLen - 1] = '\0';
    return metadata;
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
