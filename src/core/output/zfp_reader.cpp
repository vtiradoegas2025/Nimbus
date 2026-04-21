/**
 * @file zfp_reader.cpp
 * @brief ZFP v2 file reader with delta frame reconstruction.
 *
 * Reads .zfp3d files written by AsyncOutputWriter and reconstructs
 * original field values from delta-encoded frame sequences.
 */

#include "core/output/zfp_reader.hpp"

#include <cstring>
#include <fstream>

#ifdef HAVE_ZFP
#include <zfp.h>
#endif

namespace {

/// v1 header size (no delta flags).
constexpr std::size_t k_header_size_v1 = 36;

/// v2 header size (with delta flags + reserved).
constexpr std::size_t k_header_size_v2 = 40;

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

    in.read(reinterpret_cast<char*>(&header.version), sizeof(header.version));
    if (!in.good() || (header.version != 1 && header.version != 2))
    {
        error = "ZFP reader: unsupported version " +
                std::to_string(header.version);
        return false;
    }

    in.read(reinterpret_cast<char*>(&header.mode), sizeof(header.mode));
    in.read(reinterpret_cast<char*>(&header.dim0), sizeof(header.dim0));
    in.read(reinterpret_cast<char*>(&header.dim1), sizeof(header.dim1));
    in.read(reinterpret_cast<char*>(&header.dim2), sizeof(header.dim2));
    in.read(reinterpret_cast<char*>(&header.mode_param), sizeof(header.mode_param));
    in.read(reinterpret_cast<char*>(&header.compressed_size),
            sizeof(header.compressed_size));

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
    if (header.mode > 2)
    {
        error = "ZFP reader: unknown mode " + std::to_string(header.mode);
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
    }
    else
    {
        header.flags = 0x00; // v1 files are implicitly keyframes
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

    return parse_common_header(in, header, error);
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

    if (!parse_common_header(in, header, error))
    {
        return false;
    }

    // Read compressed payload
    if (header.compressed_size == 0)
    {
        error = "ZFP reader: zero compressed size in " + path;
        return false;
    }

    std::vector<unsigned char> compressed(header.compressed_size);
    in.read(reinterpret_cast<char*>(compressed.data()),
            static_cast<std::streamsize>(header.compressed_size));
    if (!in.good())
    {
        error = "ZFP reader: truncated payload in " + path +
                " (expected " + std::to_string(header.compressed_size) +
                " bytes)";
        return false;
    }

    // Set up ZFP decompression
    const std::size_t total = header.total_elements();
    data.resize(total);

    zfp_type type = zfp_type_float;
    zfp_field* field = zfp_field_3d(data.data(), type,
                                     header.dim2, header.dim1, header.dim0);
    if (!field)
    {
        error = "ZFP reader: failed to create field";
        return false;
    }

    zfp_stream* zfp = zfp_stream_open(nullptr);
    if (!zfp)
    {
        zfp_field_free(field);
        error = "ZFP reader: failed to open stream";
        return false;
    }

    // Configure decompression mode to match what was used for compression
    switch (header.mode)
    {
    case 0: // accuracy
        zfp_stream_set_accuracy(zfp, header.mode_param);
        break;
    case 1: // precision
        zfp_stream_set_precision(zfp, static_cast<uint>(header.mode_param));
        break;
    case 2: // rate
        zfp_stream_set_rate(zfp, header.mode_param, type, 3, 0);
        break;
    }

    bitstream* stream = stream_open(compressed.data(),
                                     header.compressed_size);
    if (!stream)
    {
        zfp_stream_close(zfp);
        zfp_field_free(field);
        error = "ZFP reader: failed to open bitstream";
        return false;
    }
    zfp_stream_set_bit_stream(zfp, stream);
    zfp_stream_rewind(zfp);

    // Decompress
    std::size_t decompressed_size = zfp_decompress(zfp, field);

    stream_close(stream);
    zfp_stream_close(zfp);
    zfp_field_free(field);

    if (decompressed_size == 0)
    {
        error = "ZFP reader: decompression failed for " + path;
        return false;
    }

    return true;
#endif // HAVE_ZFP
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
    // Read and decompress the raw frame data
    if (!read_zfp_3d(path, header, data, error))
    {
        return false;
    }

    const std::size_t n = data.size();

    if (header.is_keyframe())
    {
        // Keyframe: data is the absolute field values.
        // Store as new baseline for subsequent delta frames.
        auto& hist = history_[field_name];
        hist.prev = data;
        hist.prev2.clear();
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

    if (hist.prev.size() != n)
    {
        error = "ZFP reader: dimension mismatch for '" + field_name +
                "' (prev=" + std::to_string(hist.prev.size()) +
                ", current=" + std::to_string(n) + ")";
        return false;
    }

    if (header.is_simple_delta())
    {
        // Reconstruct: current = residual + previous
        for (std::size_t i = 0; i < n; ++i)
        {
            data[i] += hist.prev[i];
        }
    }
    else if (header.is_predictive_delta())
    {
        if (hist.prev2.empty())
        {
            // Only one previous frame available -- fall back to simple
            // delta reconstruction. This happens for the first delta frame
            // after a keyframe when predictive delta is enabled.
            for (std::size_t i = 0; i < n; ++i)
            {
                data[i] += hist.prev[i];
            }
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
                data[i] += 2.0f * hist.prev[i] - hist.prev2[i];
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
