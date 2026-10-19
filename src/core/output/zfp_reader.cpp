/**
 * @file zfp_reader.cpp
 * @brief ZFP v2 file reader with delta frame reconstruction.
 *
 * Reads .zfp3d files written by AsyncOutputWriter and reconstructs
 * original field values from delta-encoded frame sequences.
 */

#include "core/output/zfp_reader.hpp"
#include "core/infra/sha256.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>

#ifdef HAVE_ZFP
#include <zfp.h>
#endif

namespace {

/// v1 header size (no delta flags).
constexpr std::size_t k_header_size_v1 = 36;

/// v2 header size (with delta flags + reserved).
constexpr std::size_t k_header_size_v2 = 40;

/// v3 header size and directory-entry size.
constexpr std::size_t k_header_size_v3 = 44;
constexpr uint32_t k_brick_entry_bytes_v3 = 8;
constexpr uint32_t k_brick_entry_bytes_v4 = 40;
constexpr uint32_t k_brick_dim0 = 64;
constexpr uint32_t k_brick_dim1 = 64;
constexpr uint32_t k_brick_dim2 = 32;
constexpr uint64_t k_max_zfp_brick_compressed_bytes = 4ULL * 1024ULL * 1024ULL;

bool checked_element_count(const std::array<std::size_t, 3>& extent,
                           std::size_t& count);

uint32_t ceil_div_u32(uint32_t value, uint32_t divisor)
{
    return value / divisor + (value % divisor != 0 ? 1U : 0U);
}

struct ZfpBrickEntry
{
    uint32_t origin0 = 0;
    uint32_t origin1 = 0;
    uint32_t origin2 = 0;
    uint32_t extent0 = 0;
    uint32_t extent1 = 0;
    uint32_t extent2 = 0;
    uint64_t payload_offset = 0;
    uint64_t compressed_size = 0;
    std::array<std::uint8_t, 32> digest{};
};

uint64_t decode_u64_le(const unsigned char* bytes)
{
    uint64_t value = 0;
    for (unsigned int shift = 0; shift < 64; shift += 8)
        value |= static_cast<uint64_t>(bytes[shift / 8]) << shift;
    return value;
}

uint32_t decode_u32_le(const unsigned char* bytes)
{
    uint32_t value = 0;
    for (unsigned int shift = 0; shift < 32; shift += 8)
        value |= static_cast<uint32_t>(bytes[shift / 8]) << shift;
    return value;
}

bool read_u64_le(std::ifstream& in, uint64_t& value)
{
    unsigned char bytes[8];
    in.read(reinterpret_cast<char*>(bytes), sizeof(bytes));
    if (!in.good())
        return false;
    value = decode_u64_le(bytes);
    return true;
}

template <typename T>
bool read_binary(std::ifstream& in, T& value)
{
    in.read(reinterpret_cast<char*>(&value), sizeof(value));
    return in.good();
}

/// Parse the common header fields present in both v1 and v2 files.
/// Reads the first 36 bytes. Caller must handle the v2 flags separately.
bool parse_common_header(std::ifstream& in, Zfp3dHeader& header,
                         std::string& error)
{
    char magic[4];
    in.read(magic, 4);
    if (!in.good() || magic[0] != 'Z' || magic[1] != 'F' ||
        magic[2] != 'P' || magic[3] != '3')
    {
        error = "ZFP reader: invalid magic (expected ZFP3)";
        return false;
    }

    unsigned char version_bytes[2];
    in.read(reinterpret_cast<char*>(version_bytes), sizeof(version_bytes));
    if (in.good() && version_bytes[0] == 4 && version_bytes[1] == 0)
        header.version = 4;
    else
        std::memcpy(&header.version, version_bytes, sizeof(header.version));
    if (!in.good() || (header.version != 1 && header.version != 2 &&
                       header.version != 3 && header.version != 4))
    {
        error = "ZFP reader: unsupported version " +
                std::to_string(header.version);
        return false;
    }

    if (header.version == 4)
    {
        unsigned char fields[30];
        in.read(reinterpret_cast<char*>(fields), sizeof(fields));
        if (in.good())
        {
            header.mode = static_cast<uint16_t>(fields[0] | (fields[1] << 8));
            header.dim0 = static_cast<int32_t>(decode_u32_le(fields + 2));
            header.dim1 = static_cast<int32_t>(decode_u32_le(fields + 6));
            header.dim2 = static_cast<int32_t>(decode_u32_le(fields + 10));
            const uint64_t parameter_bits = decode_u64_le(fields + 14);
            std::memcpy(&header.mode_param, &parameter_bits, sizeof(parameter_bits));
            header.compressed_size = decode_u64_le(fields + 22);
        }
    }
    else
    {
        in.read(reinterpret_cast<char*>(&header.mode), sizeof(header.mode));
        in.read(reinterpret_cast<char*>(&header.dim0), sizeof(header.dim0));
        in.read(reinterpret_cast<char*>(&header.dim1), sizeof(header.dim1));
        in.read(reinterpret_cast<char*>(&header.dim2), sizeof(header.dim2));
        in.read(reinterpret_cast<char*>(&header.mode_param), sizeof(header.mode_param));
        in.read(reinterpret_cast<char*>(&header.compressed_size),
                sizeof(header.compressed_size));
    }

    if (!in.good())
    {
        error = "ZFP reader: truncated header";
        return false;
    }

    // Validate dimensions
    if (header.dim0 <= 0 || header.dim1 <= 0 || header.dim2 <= 0)
    {
        error = "ZFP reader: invalid dimensions (" +
                std::to_string(header.dim0) + ", " +
                std::to_string(header.dim1) + ", " +
                std::to_string(header.dim2) + ")";
        return false;
    }

    // Validate mode
    if (header.mode > 3 || (header.mode == 3) != (header.version == 4))
    {
        error = "ZFP reader: unknown mode " + std::to_string(header.mode);
        return false;
    }
    uint64_t parameter_bits = 0;
    std::memcpy(&parameter_bits, &header.mode_param, sizeof(parameter_bits));
    if (header.version == 4 ? parameter_bits != 0 :
        (!std::isfinite(header.mode_param) || header.mode_param <= 0.0))
    {
        error = "ZFP reader: invalid mode parameter";
        return false;
    }
    if (header.mode == 1 &&
        (header.mode_param > 32.0 || std::floor(header.mode_param) != header.mode_param))
    {
        error = "ZFP reader: float32 precision must be an integer in [1, 32]";
        return false;
    }

    std::size_t element_count = 0;
    if (!checked_element_count(
            {static_cast<std::size_t>(header.dim0),
             static_cast<std::size_t>(header.dim1),
             static_cast<std::size_t>(header.dim2)},
            element_count))
    {
        error = "ZFP reader: field element count overflow";
        return false;
    }

    // Read v2 flags or default for v1
    if (header.version >= 2)
    {
        uint8_t reserved[3];
        in.read(reinterpret_cast<char*>(&header.flags), 1);
        in.read(reinterpret_cast<char*>(reserved), 3);
        if (!in.good())
        {
            error = "ZFP reader: truncated v2 header";
            return false;
        }

        // Validate flags
        if (header.flags != 0x00 && header.flags != 0x01 &&
            header.flags != 0x03)
        {
            error = "ZFP reader: unknown delta flags 0x" +
                    std::to_string(header.flags);
            return false;
        }
        if (header.version == 4 &&
            (header.flags != 0 || reserved[0] != 0 || reserved[1] != 0 || reserved[2] != 0))
        {
            error = "ZFP reader: v4 requires canonical keyframe flags and reserved bytes";
            return false;
        }
    }
    else
    {
        header.flags = 0x00; // v1 files are implicitly keyframes
    }

    if (header.compressed_size == 0)
    {
        error = "ZFP reader: zero compressed size";
        return false;
    }

    if (header.version >= 3)
    {
        char brick_magic[4];
        in.read(brick_magic, 4);
        const char* expected_magic = header.version == 4 ? "ZB4D" : "ZB3D";
        if (!in.good() || std::memcmp(brick_magic, expected_magic, 4) != 0)
        {
            error = "ZFP reader: invalid v3 brick magic";
            return false;
        }
        header.brick_dim0 = k_brick_dim0;
        header.brick_dim1 = k_brick_dim1;
        header.brick_dim2 = k_brick_dim2;
        header.brick_count0 = ceil_div_u32(
            static_cast<uint32_t>(header.dim0), header.brick_dim0);
        header.brick_count1 = ceil_div_u32(
            static_cast<uint32_t>(header.dim1), header.brick_dim1);
        header.brick_count2 = ceil_div_u32(
            static_cast<uint32_t>(header.dim2), header.brick_dim2);
        header.brick_entry_bytes = header.version == 4 ?
            k_brick_entry_bytes_v4 : k_brick_entry_bytes_v3;
    }

    return true;
}

bool checked_element_count(const std::array<std::size_t, 3>& extent,
                           std::size_t& count)
{
    if (extent[0] == 0 || extent[1] == 0 || extent[2] == 0 ||
        extent[0] > std::numeric_limits<std::size_t>::max() / extent[1])
        return false;
    const std::size_t xy = extent[0] * extent[1];
    if (xy > std::numeric_limits<std::size_t>::max() / extent[2])
        return false;
    count = xy * extent[2];
    return true;
}

bool file_size(std::ifstream& in, uint64_t& size)
{
    in.clear();
    in.seekg(0, std::ios::end);
    const std::streampos end = in.tellg();
    if (end < 0)
        return false;
    size = static_cast<uint64_t>(static_cast<std::streamoff>(end));
    return true;
}

bool brick_count(const Zfp3dHeader& header, uint64_t& count)
{
    const uint64_t c01 = static_cast<uint64_t>(header.brick_count0) * header.brick_count1;
    if (header.brick_count1 != 0 && c01 / header.brick_count1 != header.brick_count0)
        return false;
    if (header.brick_count2 != 0 && c01 > std::numeric_limits<uint64_t>::max() / header.brick_count2)
        return false;
    count = c01 * header.brick_count2;
    return count > 0;
}

bool read_brick_entry(std::ifstream& in, const Zfp3dHeader& header,
                      uint64_t index, uint64_t container_size,
                      ZfpBrickEntry& entry, std::string& error)
{
    const uint64_t entry_bytes = header.brick_entry_bytes;
    uint64_t count = 0;
    if (!brick_count(header, count) || index >= count ||
        count > (std::numeric_limits<uint64_t>::max() - k_header_size_v3) /
                    entry_bytes)
    {
        error = "ZFP reader: v3 brick index out of range";
        return false;
    }
    if (header.compressed_size >
        std::numeric_limits<uint64_t>::max() - k_header_size_v3)
    {
        error = "ZFP reader: v3 directory offset overflow";
        return false;
    }
    const uint64_t directory_start = k_header_size_v3 + header.compressed_size;
    if (directory_start > container_size ||
        count > (std::numeric_limits<uint64_t>::max() - directory_start) /
                    entry_bytes)
    {
        error = "ZFP reader: v3 directory range overflow";
        return false;
    }
    const uint64_t directory_offset =
        directory_start + index * entry_bytes;
    in.clear();
    in.seekg(static_cast<std::streamoff>(directory_offset));
    uint64_t next_payload_offset = directory_start;
    if (!(header.version == 4 ? read_u64_le(in, entry.payload_offset) :
          read_binary(in, entry.payload_offset)))
    {
        error = "ZFP reader: truncated brick directory";
        return false;
    }
    if (header.version == 4)
    {
        in.read(reinterpret_cast<char*>(entry.digest.data()), entry.digest.size());
        if (!in.good())
        {
            error = "ZFP reader: truncated v4 brick digest";
            return false;
        }
    }
    if (index + 1 < count)
    {
        in.seekg(static_cast<std::streamoff>(directory_offset + entry_bytes));
        if (!(header.version == 4 ? read_u64_le(in, next_payload_offset) :
              read_binary(in, next_payload_offset)))
        {
            error = "ZFP reader: truncated brick directory";
            return false;
        }
    }
    const uint64_t plane = static_cast<uint64_t>(header.brick_count1) * header.brick_count2;
    const uint32_t bi = static_cast<uint32_t>(index / plane);
    const uint64_t remainder = index % plane;
    const uint32_t bj = static_cast<uint32_t>(remainder / header.brick_count2);
    const uint32_t bk = static_cast<uint32_t>(remainder % header.brick_count2);
    entry.origin0 = bi * header.brick_dim0;
    entry.origin1 = bj * header.brick_dim1;
    entry.origin2 = bk * header.brick_dim2;
    entry.extent0 = std::min(header.brick_dim0,
        static_cast<uint32_t>(header.dim0) - entry.origin0);
    entry.extent1 = std::min(header.brick_dim1,
        static_cast<uint32_t>(header.dim1) - entry.origin1);
    entry.extent2 = std::min(header.brick_dim2,
        static_cast<uint32_t>(header.dim2) - entry.origin2);
    const uint64_t directory_end = directory_start + count * entry_bytes;
    if (directory_end != container_size ||
        entry.payload_offset < k_header_size_v3 ||
        next_payload_offset <= entry.payload_offset ||
        next_payload_offset > directory_start)
    {
        error = "ZFP reader: invalid v3 brick directory entry";
        return false;
    }
    entry.compressed_size = next_payload_offset - entry.payload_offset;
    if (entry.compressed_size == 0 ||
        entry.compressed_size > k_max_zfp_brick_compressed_bytes ||
        entry.compressed_size > directory_start - entry.payload_offset)
    {
        error = "ZFP reader: invalid v3 brick directory entry";
        return false;
    }
    return true;
}

bool validate_brick_container(std::ifstream& in, const Zfp3dHeader& header,
                              uint64_t container_size, std::string& error)
{
    uint64_t count = 0;
    if (!brick_count(header, count) ||
        count > (std::numeric_limits<uint64_t>::max() - k_header_size_v3) / header.brick_entry_bytes)
    {
        error = "ZFP reader: v3 brick count overflow";
        return false;
    }
    if (header.compressed_size >
            std::numeric_limits<uint64_t>::max() - k_header_size_v3 ||
        count > (std::numeric_limits<uint64_t>::max() -
                 (k_header_size_v3 + header.compressed_size)) /
                    header.brick_entry_bytes)
    {
        error = "ZFP reader: v3 directory range overflow";
        return false;
    }
    const uint64_t directory_start = k_header_size_v3 + header.compressed_size;
    if (directory_start + count * header.brick_entry_bytes != container_size)
    {
        error = "ZFP reader: v3 directory/file size mismatch";
        return false;
    }
    uint64_t expected_offset = k_header_size_v3;
    uint64_t compressed_sum = 0;
    for (uint64_t index = 0; index < count; ++index)
    {
        ZfpBrickEntry entry;
        if (!read_brick_entry(in, header, index, container_size, entry, error))
            return false;
        if (entry.payload_offset != expected_offset ||
            compressed_sum > std::numeric_limits<uint64_t>::max() - entry.compressed_size)
        {
            error = "ZFP reader: non-canonical v3 brick payload layout";
            return false;
        }
        compressed_sum += entry.compressed_size;
        expected_offset += entry.compressed_size;
    }
    if (compressed_sum != header.compressed_size || expected_offset != directory_start)
    {
        error = "ZFP reader: v3 payload total/file size mismatch";
        return false;
    }
    return true;
}

} // anonymous namespace

bool read_zfp_3d_header(const std::string& path,
                        Zfp3dHeader& header,
                        std::string& error)
{
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open())
    {
        error = "ZFP reader: failed to open " + path;
        return false;
    }

    Zfp3dHeader candidate;
    if (!parse_common_header(in, candidate, error))
        return false;
    header = candidate;
    return true;
}

bool read_zfp_3d(const std::string& path,
                 Zfp3dHeader& header,
                 std::vector<float>& data,
                 std::string& error)
{
#ifndef HAVE_ZFP
    error = "ZFP reader: built without ZFP support (HAVE_ZFP not defined)";
    return false;
#else
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open())
    {
        error = "ZFP reader: failed to open " + path;
        return false;
    }

    Zfp3dHeader candidate;
    if (!parse_common_header(in, candidate, error))
    {
        return false;
    }

    if (candidate.is_bricked())
    {
        const std::array<std::size_t, 3> origin{0, 0, 0};
        const std::array<std::size_t, 3> extent{
            static_cast<std::size_t>(candidate.dim0),
            static_cast<std::size_t>(candidate.dim1),
            static_cast<std::size_t>(candidate.dim2)};
        return read_zfp_3d_window(path, origin, extent, header, data, error);
    }

    const std::size_t payload_offset = candidate.version == 1
        ? k_header_size_v1 : k_header_size_v2;
    uint64_t container_size = 0;
    if (!file_size(in, container_size) || container_size < payload_offset ||
        candidate.compressed_size > container_size - payload_offset ||
        payload_offset + candidate.compressed_size != container_size)
    {
        error = "ZFP reader: legacy payload size mismatch in " + path;
        return false;
    }
    std::vector<unsigned char> compressed(static_cast<std::size_t>(candidate.compressed_size));
    in.clear();
    in.seekg(static_cast<std::streamoff>(payload_offset));
    in.read(reinterpret_cast<char*>(compressed.data()),
            static_cast<std::streamsize>(compressed.size()));
    if (!in.good())
    {
        error = "ZFP reader: truncated payload in " + path;
        return false;
    }

    std::vector<float> decoded(candidate.total_elements());
    const zfp_type type = zfp_type_float;
    zfp_field* field = zfp_field_3d(decoded.data(), type,
                                    candidate.dim2, candidate.dim1, candidate.dim0);
    zfp_stream* zfp = zfp_stream_open(nullptr);
    if (!field || !zfp)
    {
        if (field) zfp_field_free(field);
        if (zfp) zfp_stream_close(zfp);
        error = "ZFP reader: failed to create legacy decoder";
        return false;
    }
    switch (candidate.mode)
    {
    case 0: zfp_stream_set_accuracy(zfp, candidate.mode_param); break;
    case 1: zfp_stream_set_precision(zfp, static_cast<uint>(candidate.mode_param)); break;
    case 2: zfp_stream_set_rate(zfp, candidate.mode_param, type, 3, 0); break;
    }
    bitstream* stream = stream_open(compressed.data(), compressed.size());
    if (!stream)
    {
        zfp_stream_close(zfp);
        zfp_field_free(field);
        error = "ZFP reader: failed to open legacy bitstream";
        return false;
    }
    zfp_stream_set_bit_stream(zfp, stream);
    zfp_stream_rewind(zfp);
    const std::size_t decompressed_size = zfp_decompress(zfp, field);
    stream_close(stream);
    zfp_stream_close(zfp);
    zfp_field_free(field);
    if (decompressed_size == 0)
    {
        error = "ZFP reader: decompression failed for " + path;
        return false;
    }
    data.swap(decoded);
    header = candidate;
    return true;
#endif // HAVE_ZFP
}

bool read_zfp_3d_window(const std::string& path,
                        const std::array<std::size_t, 3>& origin,
                        const std::array<std::size_t, 3>& extent,
                        Zfp3dHeader& header,
                        std::vector<float>& data,
                        std::string& error)
{
#ifndef HAVE_ZFP
    (void)path; (void)origin; (void)extent; (void)header; (void)data;
    error = "ZFP reader: built without ZFP support (HAVE_ZFP not defined)";
    return false;
#else
    std::ifstream in(path, std::ios::binary);
    Zfp3dHeader candidate;
    if (!in.is_open() || !parse_common_header(in, candidate, error))
    {
        if (error.empty()) error = "ZFP reader: failed to open " + path;
        return false;
    }
    const std::array<std::size_t, 3> shape{
        static_cast<std::size_t>(candidate.dim0),
        static_cast<std::size_t>(candidate.dim1),
        static_cast<std::size_t>(candidate.dim2)};
    std::size_t output_elements = 0;
    for (std::size_t axis = 0; axis < 3; ++axis)
        if (origin[axis] > shape[axis] || extent[axis] > shape[axis] - origin[axis])
        {
            error = "ZFP reader: requested window exceeds field shape";
            return false;
        }
    if (!checked_element_count(extent, output_elements))
    {
        error = "ZFP reader: invalid/overflowing window extent";
        return false;
    }

    if (!candidate.is_bricked())
    {
        std::vector<float> full;
        if (!read_zfp_3d(path, candidate, full, error))
            return false;
        std::vector<float> cropped(output_elements);
        for (std::size_t i = 0; i < extent[0]; ++i)
            for (std::size_t j = 0; j < extent[1]; ++j)
                for (std::size_t k = 0; k < extent[2]; ++k)
                    cropped[(i * extent[1] + j) * extent[2] + k] =
                        full[((origin[0] + i) * shape[1] + origin[1] + j) * shape[2]
                             + origin[2] + k];
        data.swap(cropped);
        header = candidate;
        return true;
    }

    uint64_t container_size = 0;
    if (!file_size(in, container_size) ||
        !validate_brick_container(in, candidate, container_size, error))
    {
        if (error.empty()) error = "ZFP reader: failed to size v3 container";
        return false;
    }
    std::vector<float> window(output_elements);
    std::vector<unsigned char> compressed;
    std::vector<float> brick;
    const uint32_t first0 = static_cast<uint32_t>(origin[0] / candidate.brick_dim0);
    const uint32_t first1 = static_cast<uint32_t>(origin[1] / candidate.brick_dim1);
    const uint32_t first2 = static_cast<uint32_t>(origin[2] / candidate.brick_dim2);
    const uint32_t last0 = static_cast<uint32_t>((origin[0] + extent[0] - 1) / candidate.brick_dim0);
    const uint32_t last1 = static_cast<uint32_t>((origin[1] + extent[1] - 1) / candidate.brick_dim1);
    const uint32_t last2 = static_cast<uint32_t>((origin[2] + extent[2] - 1) / candidate.brick_dim2);
    for (uint32_t bi = first0; bi <= last0; ++bi)
        for (uint32_t bj = first1; bj <= last1; ++bj)
            for (uint32_t bk = first2; bk <= last2; ++bk)
            {
                const uint64_t index =
                    (static_cast<uint64_t>(bi) * candidate.brick_count1 + bj)
                    * candidate.brick_count2 + bk;
                ZfpBrickEntry entry;
                if (!read_brick_entry(in, candidate, index, container_size, entry, error))
                    return false;
                compressed.resize(static_cast<std::size_t>(entry.compressed_size));
                in.clear();
                in.seekg(static_cast<std::streamoff>(entry.payload_offset));
                in.read(reinterpret_cast<char*>(compressed.data()),
                        static_cast<std::streamsize>(compressed.size()));
                if (!in.good())
                {
                    error = "ZFP reader: truncated v3 brick payload";
                    return false;
                }
                if (candidate.version == 4)
                {
                    nimbus::crypto::Sha256 hash;
                    hash.update(compressed.data(), compressed.size());
                    if (hash.digest() != entry.digest)
                    {
                        error = "ZFP reader: v4 brick payload digest mismatch";
                        return false;
                    }
                }
                const std::size_t brick_elements =
                    static_cast<std::size_t>(entry.extent0) * entry.extent1 * entry.extent2;
                brick.resize(brick_elements);
                const zfp_type type = zfp_type_float;
                zfp_field* field = zfp_field_3d(brick.data(), type,
                                                entry.extent2, entry.extent1, entry.extent0);
                zfp_stream* zfp = zfp_stream_open(nullptr);
                if (!field || !zfp)
                {
                    if (field) zfp_field_free(field);
                    if (zfp) zfp_stream_close(zfp);
                    error = "ZFP reader: failed to create v3 brick decoder";
                    return false;
                }
                switch (candidate.mode)
                {
                case 0: zfp_stream_set_accuracy(zfp, candidate.mode_param); break;
                case 1: zfp_stream_set_precision(zfp, static_cast<uint>(candidate.mode_param)); break;
                case 2: zfp_stream_set_rate(zfp, candidate.mode_param, type, 3, 0); break;
                case 3: zfp_stream_set_reversible(zfp); break;
                }
                bitstream* stream = stream_open(compressed.data(), compressed.size());
                if (!stream)
                {
                    zfp_stream_close(zfp);
                    zfp_field_free(field);
                    error = "ZFP reader: failed to open v3 brick bitstream";
                    return false;
                }
                zfp_stream_set_bit_stream(zfp, stream);
                zfp_stream_rewind(zfp);
                const std::size_t decoded = zfp_decompress(zfp, field);
                stream_close(stream);
                zfp_stream_close(zfp);
                zfp_field_free(field);
                if (decoded == 0)
                {
                    error = "ZFP reader: v3 brick decompression failed";
                    return false;
                }

                const std::size_t copy_begin0 = std::max<std::size_t>(origin[0], entry.origin0);
                const std::size_t copy_begin1 = std::max<std::size_t>(origin[1], entry.origin1);
                const std::size_t copy_begin2 = std::max<std::size_t>(origin[2], entry.origin2);
                const std::size_t copy_end0 = std::min<std::size_t>(origin[0] + extent[0], entry.origin0 + entry.extent0);
                const std::size_t copy_end1 = std::min<std::size_t>(origin[1] + extent[1], entry.origin1 + entry.extent1);
                const std::size_t copy_end2 = std::min<std::size_t>(origin[2] + extent[2], entry.origin2 + entry.extent2);
                for (std::size_t gi = copy_begin0; gi < copy_end0; ++gi)
                    for (std::size_t gj = copy_begin1; gj < copy_end1; ++gj)
                        for (std::size_t gk = copy_begin2; gk < copy_end2; ++gk)
                        {
                            const std::size_t source =
                                ((gi - entry.origin0) * entry.extent1 + (gj - entry.origin1))
                                * entry.extent2 + (gk - entry.origin2);
                            const std::size_t destination =
                                ((gi - origin[0]) * extent[1] + (gj - origin[1]))
                                * extent[2] + (gk - origin[2]);
                            window[destination] = brick[source];
                        }
            }
    data.swap(window);
    header = candidate;
    return true;
#endif
}

bool read_zfp_3d_brick(const std::string& path,
                       const std::array<std::size_t, 3>& brick_index,
                       Zfp3dHeader& header,
                       std::vector<float>& data,
                       std::string& error)
{
    Zfp3dHeader candidate;
    if (!read_zfp_3d_header(path, candidate, error))
        return false;
    if (!candidate.is_bricked() ||
        brick_index[0] >= candidate.brick_count0 ||
        brick_index[1] >= candidate.brick_count1 ||
        brick_index[2] >= candidate.brick_count2)
    {
        error = "ZFP reader: requested brick is out of range";
        return false;
    }
    const std::array<std::size_t, 3> origin{
        brick_index[0] * candidate.brick_dim0,
        brick_index[1] * candidate.brick_dim1,
        brick_index[2] * candidate.brick_dim2};
    const std::array<std::size_t, 3> extent{
        std::min<std::size_t>(candidate.brick_dim0, candidate.dim0 - origin[0]),
        std::min<std::size_t>(candidate.brick_dim1, candidate.dim1 - origin[1]),
        std::min<std::size_t>(candidate.brick_dim2, candidate.dim2 - origin[2])};
    return read_zfp_3d_window(path, origin, extent, header, data, error);
}

// --- ZfpFrameReader implementation ---

bool ZfpFrameReader::read_frame(const std::string& field_name,
                                const std::string& path,
                                std::vector<float>& data,
                                std::string& error)
{
    Zfp3dHeader header;
    return read_frame(field_name, path, header, data, error);
}

bool ZfpFrameReader::read_frame(const std::string& field_name,
                                const std::string& path,
                                Zfp3dHeader& header,
                                std::vector<float>& data,
                                std::string& error)
{
    if (!read_zfp_3d(path, header, data, error))
        return false;
    const std::array<std::size_t, 3> origin{0, 0, 0};
    const std::array<std::size_t, 3> extent{
        static_cast<std::size_t>(header.dim0),
        static_cast<std::size_t>(header.dim1),
        static_cast<std::size_t>(header.dim2)};
    return apply_decoded_frame(field_name, header, origin, extent,
                               false, data, error);
}

bool ZfpFrameReader::read_frame_window(
    const std::string& field_name,
    const std::string& path,
    const std::array<std::size_t, 3>& origin,
    const std::array<std::size_t, 3>& extent,
    Zfp3dHeader& header,
    std::vector<float>& data,
    std::string& error)
{
    if (!read_zfp_3d_window(path, origin, extent, header, data, error))
        return false;
    return apply_decoded_frame(field_name, header, origin, extent,
                               true, data, error);
}

bool ZfpFrameReader::apply_decoded_frame(
    const std::string& field_name,
    const Zfp3dHeader& header,
    const std::array<std::size_t, 3>& origin,
    const std::array<std::size_t, 3>& extent,
    bool windowed,
    std::vector<float>& data,
    std::string& error)
{

    const std::size_t n = data.size();

    if (header.is_keyframe())
    {
        if (header.version == 4)
        {
            history_.erase(field_name);
            return true;
        }
        // Keyframe: data is the absolute field values.
        // Store as new baseline for subsequent delta frames.
        auto& hist = history_[field_name];
        hist.prev = data;
        hist.prev2.clear();
        hist.origin = origin;
        hist.extent = extent;
        hist.windowed = windowed;
        return true;
    }

    // Delta frame: need history to reconstruct
    auto it = history_.find(field_name);
    if (it == history_.end() || it->second.prev.empty())
    {
        error = "ZFP reader: delta frame for '" + field_name +
                "' but no preceding keyframe";
        return false;
    }

    auto& hist = it->second;

    if (hist.prev.size() != n || hist.origin != origin ||
        hist.extent != extent || hist.windowed != windowed)
    {
        error = "ZFP reader: history window/shape mismatch for '" +
                field_name + "'";
        return false;
    }

    if (header.is_simple_delta())
    {
        // Reconstruct: current = residual + previous
        for (std::size_t i = 0; i < n; ++i)
        {
            data[i] = static_cast<float>(
                static_cast<double>(data[i]) +
                static_cast<double>(hist.prev[i]));
        }
    }
    else if (header.is_predictive_delta())
    {
        if (hist.prev2.empty())
        {
            error = "ZFP reader: predictive delta for '" + field_name +
                    "' has no second decoded reference";
            return false;
        }
        else
        {
            if (hist.prev2.size() != n)
            {
                error = "ZFP reader: prev2 dimension mismatch for '" +
                        field_name + "'";
                return false;
            }

            // Reconstruct: current = residual + 2*prev - prev2
            for (std::size_t i = 0; i < n; ++i)
            {
                data[i] = static_cast<float>(
                    static_cast<double>(data[i])
                    + 2.0 * static_cast<double>(hist.prev[i])
                    - static_cast<double>(hist.prev2[i]));
            }
        }
    }
    else
    {
        error = "ZFP reader: unknown delta flags 0x" +
                std::to_string(header.flags) + " for '" + field_name + "'";
        return false;
    }

    // Rotate history: prev -> prev2, reconstructed -> prev
    hist.prev2 = std::move(hist.prev);
    hist.prev = data;

    return true;
}

void ZfpFrameReader::reset()
{
    history_.clear();
}

void ZfpFrameReader::reset_field(const std::string& field_name)
{
    history_.erase(field_name);
}

bool ZfpFrameReader::has_history(const std::string& field_name) const
{
    auto it = history_.find(field_name);
    return it != history_.end() && !it->second.prev.empty();
}
