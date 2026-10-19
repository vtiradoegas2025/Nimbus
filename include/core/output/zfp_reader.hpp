#pragma once

/**
 * @file zfp_reader.hpp
 * @brief ZFP v1-v4 reader with delta and bounded brick-window reconstruction.
 *
 * Provides two levels of API:
 *
 *   1. read_zfp_3d() -- single-frame read: reads a .zfp3d file, decompresses
 *      the payload, and returns the raw data (keyframe) or delta residual.
 *
 *   2. ZfpFrameReader -- stateful reader that tracks per-field frame history
 *      and automatically reconstructs original values from delta-encoded
 *      sequences. This is the correct way to read a time series of frames.
 *
 * File format (v1/v2 common prefix, v3 seekable-brick extension):
 *   Bytes 0-3:   magic "ZFP3"
 *   Bytes 4-5:   version (uint16_t: 1 or 2)
 *   Bytes 6-7:   mode (uint16_t: 0=accuracy, 1=precision, 2=rate)
 *   Bytes 8-11:  dim0 (int32_t)
 *   Bytes 12-15: dim1 (int32_t)
 *   Bytes 16-19: dim2 (int32_t)
 *   Bytes 20-27: tolerance/precision/rate (double)
 *   Bytes 28-35: compressed_size (uint64_t)
 *   --- v2 additions ---
 *   Bytes 36:    flags (uint8_t: 0x00=keyframe, 0x01=simple delta, 0x03=predictive)
 *   Bytes 37-39: reserved (3 bytes)
 *   Bytes 40+:   compressed payload
 *
 * For v1 files, payload starts at byte 36 and flags are implicitly 0x00. V3
 * retains the v2 40-byte prefix, then stores the four-byte `ZB3D` declaration,
 * independently compressed fixed 64x64x32 brick payloads, and an EOF table
 * containing one absolute eight-byte payload offset per brick. V4 uses mode 3
 * reversible keyframes and a 40-byte offset/SHA-256 entry per brick.
 */

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

/// Delta encoding flags stored in the file header.
enum class ZfpDeltaFlags : uint8_t
{
    keyframe = 0x00,         ///< Absolute values (no reconstruction needed)
    simple_delta = 0x01,     ///< current - previous
    predictive_delta = 0x03  ///< current - 2*previous + previous_2
};

/**
 * @brief Parsed header from a .zfp3d file.
 */
struct Zfp3dHeader
{
    uint16_t version = 0;
    uint16_t mode = 0;       ///< 0=accuracy, 1=precision, 2=rate, 3=v4 reversible
    int32_t dim0 = 0;
    int32_t dim1 = 0;
    int32_t dim2 = 0;
    double mode_param = 0.0; ///< tolerance, precision bits, or rate bps
    uint64_t compressed_size = 0;
    uint8_t flags = 0;       ///< Delta flags (0x00 for v1 files)
    uint32_t brick_dim0 = 0;
    uint32_t brick_dim1 = 0;
    uint32_t brick_dim2 = 0;
    uint32_t brick_count0 = 0;
    uint32_t brick_count1 = 0;
    uint32_t brick_count2 = 0;
    uint32_t brick_entry_bytes = 0;

    /// Total element count.
    std::size_t total_elements() const
    {
        return static_cast<std::size_t>(dim0) *
               static_cast<std::size_t>(dim1) *
               static_cast<std::size_t>(dim2);
    }

    /// True if this frame is a keyframe (no delta reconstruction needed).
    bool is_keyframe() const { return flags == 0x00; }

    /// True if this is a simple delta frame (current - previous).
    bool is_simple_delta() const { return flags == 0x01; }

    /// True if this is a predictive delta frame (current - 2*prev + prev2).
    bool is_predictive_delta() const { return flags == 0x03; }
    bool is_bricked() const { return version == 3 || version == 4; }
};

/**
 * @brief Read and decompress a single .zfp3d file.
 *
 * Returns the raw decompressed data. For keyframes, this is the original
 * field values. For delta frames, this is the residual -- use ZfpFrameReader
 * for automatic reconstruction.
 *
 * @param path       Path to the .zfp3d file.
 * @param[out] header  Parsed file header.
 * @param[out] data    Decompressed float data (resized to dim0*dim1*dim2).
 * @param[out] error   Error message on failure.
 * @return true on success.
 */
bool read_zfp_3d(const std::string& path,
                 Zfp3dHeader& header,
                 std::vector<float>& data,
                 std::string& error);

/**
 * @brief Decode only a bounded C-order spatial window.
 *
 * V3/v4 read and allocate only intersecting bricks. V1/v2 remain compatible
 * but require a full decode before cropping and therefore do not satisfy the
 * bounded-memory contract.
 */
bool read_zfp_3d_window(const std::string& path,
                        const std::array<std::size_t, 3>& origin,
                        const std::array<std::size_t, 3>& extent,
                        Zfp3dHeader& header,
                        std::vector<float>& data,
                        std::string& error);

/** Decode one canonical 64x64x32 spatial brick without reading other payloads. */
bool read_zfp_3d_brick(const std::string& path,
                       const std::array<std::size_t, 3>& brick_index,
                       Zfp3dHeader& header,
                       std::vector<float>& data,
                       std::string& error);

/**
 * @brief Read only the header from a .zfp3d file (no decompression).
 *
 * Useful for inspecting file metadata without the cost of decompression.
 *
 * @param path       Path to the .zfp3d file.
 * @param[out] header  Parsed file header.
 * @param[out] error   Error message on failure.
 * @return true on success.
 */
bool read_zfp_3d_header(const std::string& path,
                        Zfp3dHeader& header,
                        std::string& error);

/**
 * @brief Stateful ZFP frame reader with delta reconstruction.
 *
 * Maintains per-field frame history so that delta-encoded sequences are
 * transparently reconstructed to original values. Frames must be read in
 * order (keyframe first, then deltas). A keyframe resets the history.
 *
 * Usage:
 *   ZfpFrameReader reader;
 *   for (const auto& path : frame_paths) {
 *       std::vector<float> data;
 *       std::string error;
 *       if (!reader.read_frame("theta", path, data, error)) { ... }
 *       // data contains the reconstructed original values
 *   }
 */
class ZfpFrameReader
{
public:
    /**
     * @brief Read a frame and reconstruct the original field values.
     *
     * For keyframes: stores the data as the new baseline and returns it.
     * For simple deltas: adds the residual to the previous frame.
     * For predictive deltas: reconstructs via current = residual + 2*prev - prev2.
     *
     * @param field_name  Logical field name (used to track per-field history).
     * @param path        Path to the .zfp3d file.
     * @param[out] data   Reconstructed field data.
     * @param[out] error  Error message on failure.
     * @return true on success.
     */
    bool read_frame(const std::string& field_name,
                    const std::string& path,
                    std::vector<float>& data,
                    std::string& error);

    /** Decode/reconstruct one fixed spatial window across a frame sequence. */
    bool read_frame_window(const std::string& field_name,
                           const std::string& path,
                           const std::array<std::size_t, 3>& origin,
                           const std::array<std::size_t, 3>& extent,
                           Zfp3dHeader& header,
                           std::vector<float>& data,
                           std::string& error);

    /**
     * @brief Read a frame and return the header alongside reconstructed data.
     *
     * Same as read_frame() but also populates the header struct.
     */
    bool read_frame(const std::string& field_name,
                    const std::string& path,
                    Zfp3dHeader& header,
                    std::vector<float>& data,
                    std::string& error);

    /// Reset all per-field history. Subsequent reads must start from keyframes.
    void reset();

    /// Reset history for a single field.
    void reset_field(const std::string& field_name);

    /// Returns true if the reader has stored history for a given field.
    bool has_history(const std::string& field_name) const;

private:
    struct FieldHistory
    {
        std::vector<float> prev;   ///< Most recent reconstructed frame
        std::vector<float> prev2;  ///< Frame before prev (for predictive delta)
        std::array<std::size_t, 3> origin{};
        std::array<std::size_t, 3> extent{};
        bool windowed = false;
    };

    bool apply_decoded_frame(const std::string& field_name,
                             const Zfp3dHeader& header,
                             const std::array<std::size_t, 3>& origin,
                             const std::array<std::size_t, 3>& extent,
                             bool windowed,
                             std::vector<float>& data,
                             std::string& error);

    std::unordered_map<std::string, FieldHistory> history_;
};
