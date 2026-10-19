/**
 * @file output_writer.cpp
 * @brief Async output writer implementation.
 */

#include "core/output/output_writer.hpp"
#include "core/output/history_index.hpp"

#include "core/output/npy_writer.hpp"
#include "core/output/netcdf_history_writer.hpp"
#include "core/output/durable_file.hpp"
#include "core/output/field_brick_hierarchy.hpp"
#include "core/output/step_manifest.hpp"
#include "core/output/zfp_field_storage.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cctype>
#include <cstddef>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <map>
#include <sstream>

#include <cmath>
#include <cstdint>

#ifdef HAVE_ZFP
#include <zfp.h>
#endif

namespace
{
void assign_output_error(std::string& error, const char* message) noexcept
{
    try { error = message; }
    catch (...) { error.clear(); }
}


bool is_3d_field(const FieldSnapshotEntry& entry)
{
    return entry.is_3d && entry.dim1 > 0 && entry.dim2 > 0;
}

std::string file_sha256(const std::filesystem::path& path)
{
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("failed to open output payload for SHA-256");
    nimbus::crypto::Sha256 hash;
    std::array<char, 64 * 1024> buffer{};
    while (input)
    {
        input.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
        const auto count = input.gcount();
        if (count > 0) hash.update(buffer.data(), static_cast<std::size_t>(count));
    }
    if (!input.eof()) throw std::runtime_error("failed to read output payload for SHA-256");
    return hash.hex_digest();
}

int two_dimensional_cols(const FieldSnapshotEntry& entry)
{
    return (entry.dim1 > 0) ? entry.dim1 : entry.dim2;
}

bool is_zfp_encodable_field(const FieldSnapshotEntry& entry)
{
    return nimbus::output::classify_zfp_field_storage(entry) ==
        nimbus::output::ZfpFieldStorage::Compressed;
}

bool is_reversible_radar_field(const std::string& field_name)
{
    return field_name == "reflectivity_dbz" || field_name == "radial_velocity" ||
        field_name == "zdr" || field_name == "kdp" || field_name == "rhohv";
}

bool validate_reversible_radar_source(const FieldSnapshotEntry& entry,
                                      std::string& error)
{
    const auto* contract = tmv::find_field_contract(entry.name);
    if (!contract || !contract->default_bounds.has_min ||
        !contract->default_bounds.has_max)
    {
        error = "ZFP: reversible radar field lacks physical bounds";
        return false;
    }
    for (float value : entry.values())
        if (!std::isfinite(value) ||
            value < contract->default_bounds.min_value ||
            value > contract->default_bounds.max_value)
        {
            error = "ZFP: reversible radar source is outside the field contract for '" +
                entry.name + "'";
            return false;
        }
    return true;
}

int zfp_dim1(const FieldSnapshotEntry& entry)
{
    return is_3d_field(entry) ? entry.dim1 : two_dimensional_cols(entry);
}

int zfp_dim2(const FieldSnapshotEntry& entry)
{
    return is_3d_field(entry) ? entry.dim2 : 1;
}

/// Quantize a float32 value to float16 precision (IEEE 754 binary16).
/// The result is returned as float32 but with only 10 mantissa bits of
/// precision, reducing entropy for downstream compression.
inline float quantize_half(float val)
{
    // Handle special cases
    if (!std::isfinite(val))
    {
        return val;
    }

    // IEEE 754 bit manipulation: float32 has 23 mantissa bits,
    // float16 has 10. We zero the bottom 13 bits with rounding.
    uint32_t bits;
    std::memcpy(&bits, &val, sizeof(bits));

    // Round-to-nearest-even on bit 13
    constexpr uint32_t round_bit = 1u << 12;
    constexpr uint32_t mask      = 0xFFFFE000u; // keep top 19 bits (1 sign + 8 exp + 10 mantissa)
    bits                         = (bits + round_bit) & mask;

    float result;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

/// Quantize a float32 buffer to float16 precision in-place.
/// @param buf   Buffer (may be larger than count).
/// @param count Number of elements to process.
[[maybe_unused]] inline void quantize_buffer_to_half(std::vector<float>& buf, std::size_t count)
{
    for (std::size_t i = 0; i < count; ++i)
    {
        buf[i] = quantize_half(buf[i]);
    }
}

/// Zero out values below a threshold in a buffer (for sparse fields).
/// Also zeros negative values for fields like mixing ratios where
/// negative values are physically meaningless noise.
/// @param buf       Buffer (may be larger than count).
/// @param count     Number of elements to process.
/// @param threshold Absolute value threshold below which values are zeroed.
[[maybe_unused]] inline void threshold_sparse_buffer(std::vector<float>& buf,
                                                     std::size_t count,
                                                     float threshold)
{
    for (std::size_t i = 0; i < count; ++i)
    {
        if (std::abs(buf[i]) < threshold)
        {
            buf[i] = 0.0f;
        }
    }
}

} // anonymous namespace

#ifdef HAVE_ZFP
/**
 * @brief Writer-lifetime ZFP control state and high-water bitstream binding.
 *
 * zfp_stream, zfp_field, and bitstream are small opaque control objects, but
 * creating them in every field/brick puts allocator traffic in the hottest
 * output loop. The backing bytes remain owned by AsyncOutputWriter; this
 * workspace rebinds only when that vector actually moves or grows.
 */
struct ZfpWriterWorkspace
{
    zfp_stream* stream        = nullptr;
    zfp_field*  source_field  = nullptr;
    zfp_field*  decoded_field = nullptr;
    bitstream*  bit_stream    = nullptr;

    const void* bound_data          = nullptr;
    std::size_t bound_capacity      = 0;
    std::size_t initializations     = 0;
    std::size_t bitstream_bindings  = 0;
    double      control_time_s      = 0.0;
    double      codec_time_s        = 0.0;
    double      container_io_time_s = 0.0;

    ~ZfpWriterWorkspace()
    {
        release();
    }

    ZfpWriterWorkspace()                                     = default;
    ZfpWriterWorkspace(const ZfpWriterWorkspace&)            = delete;
    ZfpWriterWorkspace& operator=(const ZfpWriterWorkspace&) = delete;

    void release_bitstream() noexcept
    {
        if (bit_stream)
        {
            stream_close(bit_stream);
            bit_stream = nullptr;
        }
        bound_data     = nullptr;
        bound_capacity = 0;
    }

    void release() noexcept
    {
        release_bitstream();
        if (decoded_field)
        {
            zfp_field_free(decoded_field);
            decoded_field = nullptr;
        }
        if (source_field)
        {
            zfp_field_free(source_field);
            source_field = nullptr;
        }
        if (stream)
        {
            zfp_stream_close(stream);
            stream = nullptr;
        }
    }

    bool ensure_codec(std::string& error)
    {
        if (stream && source_field && decoded_field)
            return true;

        release();
        stream        = zfp_stream_open(nullptr);
        source_field  = zfp_field_3d(nullptr, zfp_type_float, 1, 1, 1);
        decoded_field = zfp_field_3d(nullptr, zfp_type_float, 1, 1, 1);
        if (!stream || !source_field || !decoded_field)
        {
            release();
            error = "ZFP: failed to create persistent codec workspace";
            return false;
        }
        if (!zfp_stream_set_execution(stream, zfp_exec_serial) || zfp_stream_execution(stream) != zfp_exec_serial)
        {
            release();
            error = "ZFP: failed to select the qualified serial execution policy";
            return false;
        }
        ++initializations;
        return true;
    }

    bool ensure_bitstream(std::vector<unsigned char>& buffer, std::size_t minimum_capacity, std::string& error)
    {
        if (minimum_capacity == 0)
        {
            error = "ZFP: codec reported a zero-size bitstream";
            return false;
        }
        if (buffer.size() < minimum_capacity)
            buffer.resize(minimum_capacity);

        if (bit_stream && bound_data == buffer.data() && bound_capacity >= minimum_capacity)
        {
            return true;
        }

        release_bitstream();
        bit_stream = stream_open(buffer.data(), buffer.size());
        if (!bit_stream)
        {
            error = "ZFP: failed to bind persistent brick bitstream";
            return false;
        }
        bound_data     = buffer.data();
        bound_capacity = buffer.size();
        ++bitstream_bindings;
        return true;
    }
};

struct ZfpParallelWorkerState
{
    ZfpWriterWorkspace workspace;
    std::vector<float> input_scratch;
    std::vector<float> brick_input;
    std::vector<float> brick_decoded;
    std::vector<unsigned char> compressed_buffer;
    std::vector<std::uint64_t> brick_offsets;

    std::size_t scratch_capacity_bytes() const noexcept
    {
        return input_scratch.capacity() * sizeof(float) +
            brick_input.capacity() * sizeof(float) +
            brick_decoded.capacity() * sizeof(float) +
            compressed_buffer.capacity() * sizeof(unsigned char) +
            brick_offsets.capacity() * sizeof(std::uint64_t);
    }
};
#endif

AsyncOutputWriter::AsyncOutputWriter(const OutputConfig& config) : config_(config)
{
    if (config_.async_io)
    {
        writer_thread_ = std::thread(&AsyncOutputWriter::writer_loop, this);
    }
}

AsyncOutputWriter::~AsyncOutputWriter()
{
    // Ensure the thread is joined even if flush() was not called
    if (writer_thread_.joinable())
    {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            shutdown_ = true;
        }
        cv_consumer_.notify_one();
        writer_thread_.join();
    }
}

std::size_t AsyncOutputWriter::zfp_codec_workspace_initializations() const noexcept
{
#ifdef HAVE_ZFP
    std::size_t total = zfp_workspace_ ? zfp_workspace_->initializations : 0;
    for (const auto& worker : zfp_parallel_workers_)
        total += worker->workspace.initializations;
    return total;
#else
    return 0;
#endif
}

std::size_t AsyncOutputWriter::zfp_bitstream_bindings() const noexcept
{
#ifdef HAVE_ZFP
    std::size_t total = zfp_workspace_ ? zfp_workspace_->bitstream_bindings : 0;
    for (const auto& worker : zfp_parallel_workers_)
        total += worker->workspace.bitstream_bindings;
    return total;
#else
    return 0;
#endif
}

std::size_t AsyncOutputWriter::zfp_bitstream_capacity_bytes() const noexcept
{
#ifdef HAVE_ZFP
    std::size_t total = zfp_workspace_ ? zfp_workspace_->bound_capacity : 0;
    for (const auto& worker : zfp_parallel_workers_)
        total += worker->workspace.bound_capacity;
    return total;
#else
    return 0;
#endif
}

std::string AsyncOutputWriter::zfp_codec_version() const
{
#ifdef HAVE_ZFP
    return zfp_version_string ? zfp_version_string : "unknown";
#else
    return "unavailable";
#endif
}

int AsyncOutputWriter::zfp_execution_policy_id() const noexcept
{
#ifdef HAVE_ZFP
    if (zfp_workspace_ && zfp_workspace_->stream)
        return static_cast<int>(zfp_stream_execution(zfp_workspace_->stream));
    for (const auto& worker : zfp_parallel_workers_)
        if (worker->workspace.stream)
            return static_cast<int>(
                zfp_stream_execution(worker->workspace.stream));
    return -1;
#else
    return -1;
#endif
}

double AsyncOutputWriter::zfp_control_time_s() const noexcept
{
#ifdef HAVE_ZFP
    double total = zfp_workspace_ ? zfp_workspace_->control_time_s : 0.0;
    for (const auto& worker : zfp_parallel_workers_)
        total += worker->workspace.control_time_s;
    return total;
#else
    return 0.0;
#endif
}

double AsyncOutputWriter::zfp_codec_time_s() const noexcept
{
#ifdef HAVE_ZFP
    double total = zfp_workspace_ ? zfp_workspace_->codec_time_s : 0.0;
    for (const auto& worker : zfp_parallel_workers_)
        total += worker->workspace.codec_time_s;
    return total;
#else
    return 0.0;
#endif
}

double AsyncOutputWriter::zfp_container_io_time_s() const noexcept
{
#ifdef HAVE_ZFP
    double total = zfp_workspace_ ? zfp_workspace_->container_io_time_s : 0.0;
    for (const auto& worker : zfp_parallel_workers_)
        total += worker->workspace.container_io_time_s;
    return total;
#else
    return 0.0;
#endif
}

bool AsyncOutputWriter::submit(ExportSnapshot snapshot)
{
    if (error_flag_.load(std::memory_order_acquire))
    {
        return false;
    }

    if (config_.async_io)
    {
        for (const auto& entry : snapshot.fields)
        {
            if (entry.is_borrowed())
            {
                std::lock_guard<std::mutex> lock(mutex_);
                error_msg_ = "Background output requires owned field storage: " + entry.name;
                error_flag_.store(true, std::memory_order_release);
                cv_producer_.notify_all();
                cv_consumer_.notify_all();
                return false;
            }
        }
    }

    if (!config_.async_io)
    {
        // Synchronous path — write directly on the calling thread
        std::string error;
        if (!write_snapshot(snapshot, error))
        {
            error_msg_ = std::move(error);
            error_flag_.store(true, std::memory_order_release);
            return false;
        }
        return true;
    }

    // Async path — hand off to writer thread
    {
        std::unique_lock<std::mutex> lock(mutex_);
        // Defend the one-frame bound even if a caller omitted the pre-assembly
        // wait. The normal runtime calls wait_until_ready() before allocating
        // the frame, so this wait should not hold a second complete snapshot.
        cv_producer_.wait(lock, [this] {
            return (!pending_ && !writing_) || shutdown_ || error_flag_.load(std::memory_order_acquire);
        });

        if (shutdown_ || error_flag_.load(std::memory_order_acquire))
        {
            return false;
        }

        pending_ = std::make_unique<ExportSnapshot>(std::move(snapshot));
    }
    cv_consumer_.notify_one();
    return true;
}

bool AsyncOutputWriter::wait_until_ready()
{
    if (!config_.async_io)
    {
        return !error_flag_.load(std::memory_order_acquire);
    }

    const auto wait_started = std::chrono::steady_clock::now();
    std::unique_lock<std::mutex> lock(mutex_);
    cv_producer_.wait(
        lock, [this] { return (!pending_ && !writing_) || shutdown_ || error_flag_.load(std::memory_order_acquire); });
    const double elapsed = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - wait_started).count();
    double old_wait = total_backpressure_wait_s_.load(std::memory_order_relaxed);
    while (!total_backpressure_wait_s_.compare_exchange_weak(
        old_wait, old_wait + elapsed, std::memory_order_relaxed))
    {
    }
    return !shutdown_ && !error_flag_.load(std::memory_order_acquire);
}

bool AsyncOutputWriter::flush()
{
    if (!config_.async_io)
    {
        return !error_flag_.load(std::memory_order_acquire);
    }

    {
        std::lock_guard<std::mutex> lock(mutex_);
        shutdown_ = true;
    }
    cv_consumer_.notify_one();

    if (writer_thread_.joinable())
    {
        writer_thread_.join();
    }

    return !error_flag_.load(std::memory_order_acquire);
}

std::string AsyncOutputWriter::error_message() const
{
    // Not thread-safe for concurrent reads, but only called after flush()
    return error_msg_;
}

void AsyncOutputWriter::writer_loop()
{
    while (true)
    {
        std::unique_ptr<ExportSnapshot> snap;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            cv_consumer_.wait(lock, [this] { return pending_ != nullptr || shutdown_; });

            if (pending_)
            {
                snap     = std::move(pending_);
                writing_ = true;
            }

            if (!snap && shutdown_)
            {
                break;
            }
        }

        if (snap)
        {
            std::string error;
            if (!write_snapshot(*snap, error))
            {
                snap.reset();
                std::lock_guard<std::mutex> lock(mutex_);
                writing_ = false;
                error_msg_ = std::move(error);
                error_flag_.store(true, std::memory_order_release);
                // Signal producer in case it's waiting
                cv_producer_.notify_all();
                break;
            }
            // Reclaim owned fields before the producer can assemble the next frame.
            snap.reset();
            {
                std::lock_guard<std::mutex> lock(mutex_);
                writing_ = false;
            }
            cv_producer_.notify_one();
        }
    }
}

#ifdef HAVE_ZFP
constexpr uint16_t k_zfp_bricked_version   = 3;
constexpr uint32_t k_zfp_brick_dim0        = 64;
constexpr uint32_t k_zfp_brick_dim1        = 64;
constexpr uint32_t k_zfp_brick_dim2        = 32;
constexpr uint32_t k_zfp_brick_entry_bytes = 8;

uint32_t ceil_div_u32(uint32_t value, uint32_t divisor)
{
    return value / divisor + (value % divisor != 0 ? 1U : 0U);
}

template <typename T>
void write_binary(std::ofstream& out, const T& value)
{
    out.write(reinterpret_cast<const char*>(&value), sizeof(value));
}

/**
 * @brief Write a seekable v3 ZFP container made of independent spatial bricks.
 *
 * The first 40 bytes retain the v2 common header and frame flags. Bytes 40-43
 * identify the fixed-layout `ZB3D` container. Version 3 defines 64x64x32
 * bricks and an EOF table of 8-byte entries, so those redundant values are not
 * repeated in every artifact. Each entry is the absolute offset of one independently
 * compressed brick; its end is the next entry (or EOF). Brick origin/extent
 * are derived from the canonical `(brick0,brick1,brick2)` order with brick2
 * fastest, avoiding redundant directory metadata.
 */
static bool write_zfp_3d(const float*                data,
                         int                         dim0,
                         int                         dim1,
                         int                         dim2,
                         const std::string&          mode_str,
                         double                      tolerance,
                         int                         precision_bits,
                         int                         rate_bps,
                         uint8_t                     delta_flags,
                         const std::string&          path,
                         float*                      decoded_output,
                         [[maybe_unused]] std::vector<float>& brick_input,
                         [[maybe_unused]] std::vector<float>& brick_decoded,
                         std::vector<unsigned char>& compressed_buffer,
                         std::vector<uint64_t>&      brick_offsets,
                         ZfpWriterWorkspace&         workspace,
                         std::size_t&                bytes_out,
                         std::string&                error)
{
    if (dim0 <= 0 || dim1 <= 0 || dim2 <= 0)
    {
        error = "ZFP: zero-size field";
        return false;
    }
    const std::size_t d0_size = static_cast<std::size_t>(dim0);
    const std::size_t d1_size = static_cast<std::size_t>(dim1);
    const std::size_t d2_size = static_cast<std::size_t>(dim2);
    if (d0_size > std::numeric_limits<std::size_t>::max() / d1_size ||
        d0_size * d1_size > std::numeric_limits<std::size_t>::max() / d2_size)
    {
        error = "ZFP: field element count overflow";
        return false;
    }
    const zfp_type type = zfp_type_float;
#ifdef NIMBUS_ZFP_PHASE_TIMING
    const auto field_control_started = std::chrono::steady_clock::now();
#endif
#ifdef NIMBUS_ZFP_TRANSIENT_CODEC_WORKSPACE
    // Controlled benchmark comparator for the former per-field/per-brick
    // control-object lifecycle.
    workspace.release();
#endif
    if (!workspace.ensure_codec(error))
        return false;
    zfp_stream* const zfp = workspace.stream;

    // Configure compression mode
    uint16_t mode_id    = 0;
    double   mode_param = tolerance;
    if (mode_str == "accuracy")
    {
        zfp_stream_set_accuracy(zfp, tolerance);
        mode_id    = 0;
        mode_param = tolerance;
    }
    else if (mode_str == "precision")
    {
        uint precision = static_cast<uint>(precision_bits);
        zfp_stream_set_precision(zfp, precision);
        mode_id    = 1;
        mode_param = static_cast<double>(precision);
    }
    else if (mode_str == "rate")
    {
        zfp_stream_set_rate(zfp, static_cast<double>(rate_bps), type, 3, 0);
        mode_id    = 2;
        mode_param = static_cast<double>(rate_bps);
    }
    else
    {
        error = "ZFP: unsupported compression mode '" + mode_str + "'";
        return false;
    }
#ifdef NIMBUS_ZFP_PHASE_TIMING
    workspace.control_time_s +=
        std::chrono::duration<double>(std::chrono::steady_clock::now() - field_control_started).count();
    const auto header_io_started = std::chrono::steady_clock::now();
#endif

    std::ofstream out(path, std::ios::binary);
    if (!out)
    {
        error = "ZFP: failed to open " + path;
        return false;
    }

    const uint32_t count0  = ceil_div_u32(static_cast<uint32_t>(dim0), k_zfp_brick_dim0);
    const uint32_t count1  = ceil_div_u32(static_cast<uint32_t>(dim1), k_zfp_brick_dim1);
    const uint32_t count2  = ceil_div_u32(static_cast<uint32_t>(dim2), k_zfp_brick_dim2);
    const uint64_t count01 = static_cast<uint64_t>(count0) * count1;
    if (count2 != 0 && count01 > std::numeric_limits<uint64_t>::max() / count2)
    {
        error = "ZFP: brick-count overflow for " + path;
        return false;
    }
    const uint64_t brick_count = count01 * count2;
    if (brick_count == 0 ||
        brick_count > static_cast<uint64_t>(std::numeric_limits<std::streamoff>::max() / k_zfp_brick_entry_bytes) ||
        brick_count > static_cast<uint64_t>(std::numeric_limits<std::size_t>::max()))
    {
        error = "ZFP: brick-directory size overflow for " + path;
        return false;
    }

    // Common v2-compatible prefix followed by the v3 brick declaration.
    const char     magic[4]       = {'Z', 'F', 'P', '3'};
    const char     brick_magic[4] = {'Z', 'B', '3', 'D'};
    const int32_t  d0 = dim0, d1 = dim1, d2 = dim2;
    const uint64_t initial_compressed_size = 0;
    const uint8_t  flags                   = delta_flags;
    const uint8_t  reserved[3]             = {0, 0, 0};

    out.write(magic, 4);
    write_binary(out, k_zfp_bricked_version);
    write_binary(out, mode_id);
    write_binary(out, d0);
    write_binary(out, d1);
    write_binary(out, d2);
    write_binary(out, mode_param);
    write_binary(out, initial_compressed_size);
    write_binary(out, flags);
    out.write(reinterpret_cast<const char*>(reserved), sizeof(reserved));
    out.write(brick_magic, 4);

    if (!out.good())
    {
        error = "ZFP: failed to write bricked header for " + path;
        return false;
    }
#ifdef NIMBUS_ZFP_PHASE_TIMING
    workspace.container_io_time_s +=
        std::chrono::duration<double>(std::chrono::steady_clock::now() - header_io_started).count();
#endif
    brick_offsets.clear();
    if (brick_offsets.capacity() < static_cast<std::size_t>(brick_count))
        brick_offsets.reserve(static_cast<std::size_t>(brick_count));

    zfp_field* const field         = workspace.source_field;
    zfp_field* const decoded_field = workspace.decoded_field;

    uint64_t total_compressed_size = 0;
    for (uint32_t bi = 0; bi < count0; ++bi)
        for (uint32_t bj = 0; bj < count1; ++bj)
            for (uint32_t bk = 0; bk < count2; ++bk)
            {
                const uint32_t    origin0        = bi * k_zfp_brick_dim0;
                const uint32_t    origin1        = bj * k_zfp_brick_dim1;
                const uint32_t    origin2        = bk * k_zfp_brick_dim2;
                const uint32_t    extent0        = std::min(k_zfp_brick_dim0, static_cast<uint32_t>(dim0) - origin0);
                const uint32_t    extent1        = std::min(k_zfp_brick_dim1, static_cast<uint32_t>(dim1) - origin1);
                const uint32_t    extent2        = std::min(k_zfp_brick_dim2, static_cast<uint32_t>(dim2) - origin2);
#ifdef NIMBUS_ZFP_PHASE_TIMING
                const auto brick_control_started = std::chrono::steady_clock::now();
#endif
#ifdef NIMBUS_ZFP_GATHER_BRICKS
                const std::size_t brick_elements =
                    static_cast<std::size_t>(extent0) * extent1 * extent2;
                brick_input.resize(brick_elements);
                std::size_t local = 0;
                for (uint32_t i = 0; i < extent0; ++i)
                    for (uint32_t j = 0; j < extent1; ++j)
                        for (uint32_t k = 0; k < extent2; ++k)
                        {
                            const std::size_t global =
                                (static_cast<std::size_t>(origin0 + i) * static_cast<std::size_t>(dim1) +
                                 static_cast<std::size_t>(origin1 + j)) *
                                    static_cast<std::size_t>(dim2) +
                                static_cast<std::size_t>(origin2 + k);
                            brick_input[local++] = data[global];
                        }

                zfp_field_set_pointer(field, brick_input.data());
                zfp_field_set_size_3d(field, extent2, extent1, extent0);
                zfp_field_set_stride_3d(
                    field, 1, static_cast<ptrdiff_t>(extent2), static_cast<ptrdiff_t>(extent1) * extent2);
#else
                const std::size_t brick_origin =
                    (static_cast<std::size_t>(origin0) * d1_size + origin1) * d2_size + origin2;
                zfp_field_set_pointer(field, const_cast<float*>(data + brick_origin));
                zfp_field_set_size_3d(field, extent2, extent1, extent0);
                zfp_field_set_stride_3d(
                    field, 1, static_cast<ptrdiff_t>(d2_size), static_cast<ptrdiff_t>(d1_size * d2_size));
#endif
                const std::size_t maximum_size = zfp_stream_maximum_size(zfp, field);
#ifdef NIMBUS_ZFP_TRANSIENT_CODEC_WORKSPACE
                workspace.release_bitstream();
#endif
                if (!workspace.ensure_bitstream(compressed_buffer, maximum_size, error))
                    return false;
                zfp_stream_set_bit_stream(zfp, workspace.bit_stream);
#ifdef NIMBUS_ZFP_PHASE_TIMING
                workspace.control_time_s +=
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - brick_control_started).count();
                const auto compression_started = std::chrono::steady_clock::now();
#endif
                zfp_stream_rewind(zfp);
                const std::size_t compressed_size = zfp_compress(zfp, field);
                if (compressed_size == 0)
                {
                    error = "ZFP: brick compression failed for " + path;
                    return false;
                }
#ifdef NIMBUS_ZFP_PHASE_TIMING
                workspace.codec_time_s +=
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - compression_started).count();
#endif

                if (decoded_output != nullptr)
                {
#ifdef NIMBUS_ZFP_GATHER_BRICKS
                    brick_decoded.resize(brick_elements);
                    zfp_field_set_pointer(decoded_field, brick_decoded.data());
                    zfp_field_set_size_3d(decoded_field, extent2, extent1, extent0);
                    zfp_field_set_stride_3d(
                        decoded_field, 1, static_cast<ptrdiff_t>(extent2), static_cast<ptrdiff_t>(extent1) * extent2);
#else
                    zfp_field_set_pointer(decoded_field, decoded_output + brick_origin);
                    zfp_field_set_size_3d(decoded_field, extent2, extent1, extent0);
                    zfp_field_set_stride_3d(
                        decoded_field, 1, static_cast<ptrdiff_t>(d2_size), static_cast<ptrdiff_t>(d1_size * d2_size));
#endif
#ifdef NIMBUS_ZFP_PHASE_TIMING
                    const auto decompression_started = std::chrono::steady_clock::now();
#endif
                    zfp_stream_rewind(zfp);
                    const std::size_t decompressed = zfp_decompress(zfp, decoded_field);
                    if (decompressed == 0)
                    {
                        error = "ZFP: failed to decode history brick for " + path;
                        return false;
                    }
#ifdef NIMBUS_ZFP_PHASE_TIMING
                    workspace.codec_time_s +=
                        std::chrono::duration<double>(std::chrono::steady_clock::now() - decompression_started).count();
#endif
#ifdef NIMBUS_ZFP_GATHER_BRICKS
                    local = 0;
                    for (uint32_t i = 0; i < extent0; ++i)
                        for (uint32_t j = 0; j < extent1; ++j)
                            for (uint32_t k = 0; k < extent2; ++k)
                            {
                                const std::size_t global =
                                    (static_cast<std::size_t>(origin0 + i) * static_cast<std::size_t>(dim1) +
                                     static_cast<std::size_t>(origin1 + j)) *
                                        static_cast<std::size_t>(dim2) +
                                    static_cast<std::size_t>(origin2 + k);
                                decoded_output[global] = brick_decoded[local++];
                            }
#endif
                }
#ifdef NIMBUS_ZFP_PHASE_TIMING
                const auto payload_io_started = std::chrono::steady_clock::now();
#endif

                const std::streampos payload_position = out.tellp();
                if (payload_position < 0)
                {
                    error = "ZFP: invalid payload offset for " + path;
                    return false;
                }
                brick_offsets.push_back(static_cast<uint64_t>(payload_position));
                out.write(reinterpret_cast<const char*>(compressed_buffer.data()),
                          static_cast<std::streamsize>(compressed_size));
#ifdef NIMBUS_ZFP_PHASE_TIMING
                workspace.container_io_time_s +=
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - payload_io_started).count();
#endif
                if (!out.good() || total_compressed_size > std::numeric_limits<uint64_t>::max() - compressed_size)
                {
                    error = "ZFP: brick write/size overflow for " + path;
                    return false;
                }
                total_compressed_size += static_cast<uint64_t>(compressed_size);
            }

#ifdef NIMBUS_ZFP_PHASE_TIMING
    const auto footer_io_started = std::chrono::steady_clock::now();
#endif
    if (brick_offsets.size() != static_cast<std::size_t>(brick_count))
    {
        error = "ZFP: incomplete brick offset table for " + path;
        return false;
    }
    for (const uint64_t payload_offset : brick_offsets)
        write_binary(out, payload_offset);
    if (!out.good())
    {
        error = "ZFP: failed to write brick offset table for " + path;
        return false;
    }

    out.seekp(28);
    write_binary(out, total_compressed_size);
    out.seekp(0, std::ios::end);
    const std::streampos file_end = out.tellp();
#ifdef NIMBUS_ZFP_PHASE_TIMING
    workspace.container_io_time_s +=
        std::chrono::duration<double>(std::chrono::steady_clock::now() - footer_io_started).count();
#endif
    if (!out.good() || file_end < 0)
    {
        error = "ZFP: write error for " + path;
        return false;
    }
    bytes_out = static_cast<std::size_t>(file_end);
    return true;
}

#ifdef NIMBUS_ZFP_RECOMPRESS_HISTORY
/**
 * @brief Compress then decompress a buffer in memory to get the ZFP roundtrip
 *        result (the data as it will be read back after decompression).
 *
 * This is essential for delta encoding: the writer must store the *decoded*
 * version of each frame as its reference for computing future deltas. Without
 * this, the writer's reference (original data) differs from the reader's
 * reference (decompressed data), causing error to compound across delta frames.
 * This is the same approach used by video codecs (reference frames = decoded).
 *
 * @param data       Input data to compress+decompress.
 * @param dim0,1,2   Field dimensions.
 * @param mode_str   Compression mode ("accuracy", "precision", "rate").
 * @param tolerance  Tolerance or precision parameter.
 * @param precision_bits Significant binary bits for precision mode.
 * @param rate_bps   Bits per scalar for rate mode.
 * @param[out] roundtripped  Buffer to receive the roundtripped data (resized).
 * @return true on success. On failure returns false; roundtripped is unchanged.
 */
static bool zfp_roundtrip_in_memory(const float*        data,
                                    int                 dim0,
                                    int                 dim1,
                                    int                 dim2,
                                    const std::string&  mode_str,
                                    double              tolerance,
                                    int                 precision_bits,
                                    int                 rate_bps,
                                    std::vector<float>& roundtripped)
{
    const std::size_t total =
        static_cast<std::size_t>(dim0) * static_cast<std::size_t>(dim1) * static_cast<std::size_t>(dim2);
    if (total == 0)
        return false;

    zfp_type type = zfp_type_float;

    // --- Compress ---
    zfp_field* field_in = zfp_field_3d(const_cast<float*>(data), type, dim2, dim1, dim0);
    if (!field_in)
        return false;

    zfp_stream* zfp = zfp_stream_open(nullptr);
    if (!zfp)
    {
        zfp_field_free(field_in);
        return false;
    }

    if (mode_str == "precision")
        zfp_stream_set_precision(zfp, static_cast<uint>(precision_bits));
    else if (mode_str == "rate")
        zfp_stream_set_rate(zfp, static_cast<double>(rate_bps), type, 3, 0);
    else
        zfp_stream_set_accuracy(zfp, tolerance);

    std::size_t                bufsize = zfp_stream_maximum_size(zfp, field_in);
    std::vector<unsigned char> buffer(bufsize);
    bitstream*                 bs = stream_open(buffer.data(), bufsize);
    if (!bs)
    {
        zfp_stream_close(zfp);
        zfp_field_free(field_in);
        return false;
    }
    zfp_stream_set_bit_stream(zfp, bs);
    zfp_stream_rewind(zfp);

    std::size_t compressed_size = zfp_compress(zfp, field_in);
    zfp_field_free(field_in);
    if (compressed_size == 0)
    {
        stream_close(bs);
        zfp_stream_close(zfp);
        return false;
    }

    // --- Decompress into output buffer ---
    roundtripped.resize(total);
    zfp_field* field_out = zfp_field_3d(roundtripped.data(), type, dim2, dim1, dim0);
    if (!field_out)
    {
        stream_close(bs);
        zfp_stream_close(zfp);
        return false;
    }

    zfp_stream_rewind(zfp);
    std::size_t decompressed = zfp_decompress(zfp, field_out);

    stream_close(bs);
    zfp_stream_close(zfp);
    zfp_field_free(field_out);

    return decompressed > 0;
}
#endif
#endif // HAVE_ZFP

bool write_zfp_3d_reversible(const float* data,
                             int dim0,
                             int dim1,
                             int dim2,
                             const std::string& path,
                             std::size_t scratch_limit_bytes,
                             std::size_t& bytes_out,
                             std::string& error)
{
#ifndef HAVE_ZFP
    (void)data; (void)dim0; (void)dim1; (void)dim2; (void)path;
    (void)scratch_limit_bytes; (void)bytes_out;
    error = "ZFP: reversible writer requires HAVE_ZFP";
    return false;
#else
    constexpr uint64_t header_bytes = 44;
    constexpr uint64_t entry_bytes = 40;
    constexpr std::size_t max_compressed_brick = 4U * 1024U * 1024U;
    struct DirectoryEntry
    {
        uint64_t offset;
        std::array<std::uint8_t, 32> digest;
    };
    static_assert(sizeof(DirectoryEntry) == entry_bytes);

    if (!data || dim0 <= 0 || dim1 <= 0 || dim2 <= 0)
    {
        error = "ZFP: invalid reversible field dimensions or data";
        return false;
    }
    const std::size_t d0 = static_cast<std::size_t>(dim0);
    const std::size_t d1 = static_cast<std::size_t>(dim1);
    const std::size_t d2 = static_cast<std::size_t>(dim2);
    if (d0 > std::numeric_limits<std::size_t>::max() / d1 ||
        d0 * d1 > std::numeric_limits<std::size_t>::max() / d2 ||
        d1 > static_cast<std::size_t>(std::numeric_limits<ptrdiff_t>::max()) / d2)
    {
        error = "ZFP: reversible field element or stride overflow";
        return false;
    }
    const uint64_t count0 = ceil_div_u32(static_cast<uint32_t>(dim0), k_zfp_brick_dim0);
    const uint64_t count1 = ceil_div_u32(static_cast<uint32_t>(dim1), k_zfp_brick_dim1);
    const uint64_t count2 = ceil_div_u32(static_cast<uint32_t>(dim2), k_zfp_brick_dim2);
    const uint64_t count01 = count0 * count1;
    if (count2 != 0 && count01 > std::numeric_limits<uint64_t>::max() / count2)
    {
        error = "ZFP: reversible brick count overflow";
        return false;
    }
    const uint64_t count = count01 * count2;
    if (count == 0 || count > std::numeric_limits<std::size_t>::max() / entry_bytes ||
        count > (std::numeric_limits<uint64_t>::max() - header_bytes) / entry_bytes ||
        count > (static_cast<uint64_t>(std::numeric_limits<std::streamoff>::max()) -
                 header_bytes) / entry_bytes ||
        count * entry_bytes > scratch_limit_bytes)
    {
        error = "ZFP: reversible brick directory exceeds scratch limit";
        return false;
    }
    const uint64_t maximum_payload =
        static_cast<uint64_t>(std::numeric_limits<std::streamoff>::max()) -
        header_bytes - count * entry_bytes;
    const auto path_is_occupied = [](const std::filesystem::path& candidate)
    {
        std::error_code inspect_error;
        const auto status = std::filesystem::symlink_status(candidate, inspect_error);
        return inspect_error != std::errc::no_such_file_or_directory &&
            (inspect_error || status.type() != std::filesystem::file_type::not_found);
    };
    std::error_code ec;
    if (path_is_occupied(path))
    {
        error = "ZFP: refusing to replace reversible container " + path;
        return false;
    }
    static std::atomic<uint64_t> next_temporary_id{0};
    const auto temporary = std::filesystem::path(
        path + ".v4.partial." + std::to_string(next_temporary_id.fetch_add(1)));
    if (path_is_occupied(temporary))
    {
        error = "ZFP: reversible temporary path already exists";
        return false;
    }
    struct TemporaryCleanup
    {
        std::filesystem::path path;
        bool committed = false;
        ~TemporaryCleanup()
        {
            if (!committed)
            {
                std::error_code ignored;
                std::filesystem::remove(path, ignored);
            }
        }
    } cleanup{temporary};

    ZfpWriterWorkspace workspace;
    if (!workspace.ensure_codec(error))
        return false;
    zfp_stream_set_reversible(workspace.stream);
    std::vector<DirectoryEntry> directory;
    directory.reserve(static_cast<std::size_t>(count));
    if (directory.capacity() > scratch_limit_bytes / sizeof(DirectoryEntry))
    {
        error = "ZFP: reversible actual directory scratch exceeds limit";
        return false;
    }
    const std::size_t directory_scratch = directory.capacity() * sizeof(DirectoryEntry);
    std::vector<float> decoded;
    std::vector<unsigned char> compressed;
    std::ofstream out(temporary, std::ios::binary | std::ios::trunc);
    if (!out)
    {
        error = "ZFP: failed to open reversible temporary container";
        return false;
    }
    const auto write_le = [&out](uint64_t value, std::size_t width)
    {
        for (std::size_t byte = 0; byte < width; ++byte)
        {
            const unsigned char octet = static_cast<unsigned char>(value >> (byte * 8));
            out.put(static_cast<char>(octet));
        }
    };
    out.write("ZFP3", 4);
    write_le(4, 2);
    write_le(3, 2);
    write_le(static_cast<uint32_t>(dim0), 4);
    write_le(static_cast<uint32_t>(dim1), 4);
    write_le(static_cast<uint32_t>(dim2), 4);
    write_le(0, 8);
    write_le(0, 8);
    write_le(0, 4);
    out.write("ZB4D", 4);
    if (!out)
    {
        error = "ZFP: failed to write reversible header";
        return false;
    }
    uint64_t payload_bytes = 0;
    for (uint64_t bi = 0; bi < count0; ++bi)
        for (uint64_t bj = 0; bj < count1; ++bj)
            for (uint64_t bk = 0; bk < count2; ++bk)
            {
                const std::size_t origin0 = bi * k_zfp_brick_dim0;
                const std::size_t origin1 = bj * k_zfp_brick_dim1;
                const std::size_t origin2 = bk * k_zfp_brick_dim2;
                const std::size_t extent0 = std::min<std::size_t>(k_zfp_brick_dim0, d0 - origin0);
                const std::size_t extent1 = std::min<std::size_t>(k_zfp_brick_dim1, d1 - origin1);
                const std::size_t extent2 = std::min<std::size_t>(k_zfp_brick_dim2, d2 - origin2);
                const std::size_t elements = extent0 * extent1 * extent2;
                const std::size_t source_offset = (origin0 * d1 + origin1) * d2 + origin2;
                for (std::size_t i = 0; i < extent0; ++i)
                    for (std::size_t j = 0; j < extent1; ++j)
                        for (std::size_t k = 0; k < extent2; ++k)
                            if (!std::isfinite(data[source_offset + (i * d1 + j) * d2 + k]))
                            {
                                error = "ZFP: nonfinite reversible source value";
                                return false;
                            }
                zfp_field_set_pointer(workspace.source_field, const_cast<float*>(data + source_offset));
                zfp_field_set_size_3d(workspace.source_field, extent2, extent1, extent0);
                zfp_field_set_stride_3d(workspace.source_field, 1,
                    static_cast<ptrdiff_t>(d2), static_cast<ptrdiff_t>(d1 * d2));
                const std::size_t maximum = zfp_stream_maximum_size(
                    workspace.stream, workspace.source_field);
                if (maximum == 0 || maximum > max_compressed_brick ||
                    maximum > scratch_limit_bytes - directory_scratch ||
                    elements * sizeof(float) >
                        scratch_limit_bytes - directory_scratch - maximum)
                {
                    error = "ZFP: reversible brick exceeds scratch or codec limit";
                    return false;
                }
                decoded.resize(elements);
                if (!workspace.ensure_bitstream(compressed, maximum, error) ||
                    decoded.capacity() >
                        (scratch_limit_bytes - directory_scratch) / sizeof(float) ||
                    compressed.capacity() > scratch_limit_bytes - directory_scratch -
                        decoded.capacity() * sizeof(float))
                {
                    if (error.empty()) error = "ZFP: reversible actual scratch exceeds limit";
                    return false;
                }
                zfp_stream_set_bit_stream(workspace.stream, workspace.bit_stream);
                zfp_stream_rewind(workspace.stream);
                const std::size_t encoded = zfp_compress(workspace.stream, workspace.source_field);
                if (encoded == 0 || encoded > maximum || encoded > max_compressed_brick ||
                    payload_bytes > std::numeric_limits<uint64_t>::max() - encoded ||
                    encoded > maximum_payload ||
                    payload_bytes > maximum_payload - encoded)
                {
                    error = "ZFP: reversible brick compression failed or exceeded bounds";
                    return false;
                }
                zfp_field_set_pointer(workspace.decoded_field, decoded.data());
                zfp_field_set_size_3d(workspace.decoded_field, extent2, extent1, extent0);
                zfp_field_set_stride_3d(workspace.decoded_field, 1,
                    static_cast<ptrdiff_t>(extent2), static_cast<ptrdiff_t>(extent1 * extent2));
                zfp_stream_rewind(workspace.stream);
                if (zfp_decompress(workspace.stream, workspace.decoded_field) == 0)
                {
                    error = "ZFP: reversible brick verification decode failed";
                    return false;
                }
                for (std::size_t i = 0; i < extent0; ++i)
                    for (std::size_t j = 0; j < extent1; ++j)
                        if (std::memcmp(decoded.data() + (i * extent1 + j) * extent2,
                                        data + source_offset + (i * d1 + j) * d2,
                                        extent2 * sizeof(float)) != 0)
                        {
                            error = "ZFP: reversible brick failed bitwise verification";
                            return false;
                        }
                nimbus::crypto::Sha256 hash;
                hash.update(compressed.data(), encoded);
                directory.push_back({header_bytes + payload_bytes, hash.digest()});
                out.write(reinterpret_cast<const char*>(compressed.data()),
                          static_cast<std::streamsize>(encoded));
                if (!out)
                {
                    error = "ZFP: reversible payload write failed";
                    return false;
                }
                payload_bytes += encoded;
            }
    for (const auto& entry : directory)
    {
        write_le(entry.offset, 8);
        out.write(reinterpret_cast<const char*>(entry.digest.data()), entry.digest.size());
    }
    out.seekp(28);
    write_le(payload_bytes, 8);
    out.flush();
    if (!out)
    {
        error = "ZFP: reversible directory/header write failed";
        return false;
    }
    out.close();
    if (payload_bytes > std::numeric_limits<std::size_t>::max() - header_bytes - count * entry_bytes)
    {
        error = "ZFP: reversible file size overflow";
        return false;
    }
    if (path_is_occupied(path))
    {
        error = "ZFP: refusing to replace reversible container " + path;
        return false;
    }
    std::filesystem::create_hard_link(temporary, path, ec);
    if (ec)
    {
        error = "ZFP: failed to publish reversible container: " + ec.message();
        return false;
    }
    std::filesystem::remove(temporary, ec);
    if (ec)
    {
        std::error_code ignored;
        std::filesystem::remove(path, ignored);
        error = "ZFP: failed to retire reversible temporary container: " + ec.message();
        return false;
    }
    cleanup.committed = true;
    bytes_out = static_cast<std::size_t>(header_bytes + payload_bytes + count * entry_bytes);
    return true;
#endif
}

bool AsyncOutputWriter::ensure_step_directory(const std::filesystem::path& step_dir, std::string& error) const
{
    std::error_code ec;
    const auto committed_manifest = step_dir / "manifest.json";
    const auto status = std::filesystem::symlink_status(committed_manifest, ec);
    if (ec && ec != std::errc::no_such_file_or_directory)
    {
        error = "Failed to inspect step publication " + step_dir.string() + ": " + ec.message();
        return false;
    }
    if (!ec && std::filesystem::exists(status))
    {
        error = "Refusing to overwrite committed history step " + step_dir.string();
        return false;
    }
    ec.clear();
    std::filesystem::create_directories(step_dir, ec);
    if (ec)
    {
        error = "Failed to create directory " + step_dir.string() + ": " + ec.message();
        return false;
    }

    return true;
}

bool AsyncOutputWriter::write_csv_field(const FieldSnapshotEntry&    entry,
                                        const std::filesystem::path& step_dir,
                                        std::size_t&                 bytes_written) const
{
    const std::string path = (step_dir / (entry.name + ".csv")).string();
    const bool        ok   = csv::write_3d(entry.values().data(), entry.dim0, entry.dim1, entry.dim2, path);
    if (ok)
    {
        std::error_code ec;
        const auto actual_bytes = std::filesystem::file_size(path, ec);
        if (ec)
            return false;
        bytes_written += actual_bytes;
    }

    return ok;
}

bool AsyncOutputWriter::write_npy_field(const FieldSnapshotEntry&    entry,
                                        const std::filesystem::path& step_dir,
                                        std::size_t&                 bytes_written) const
{
    const std::string path = (step_dir / (entry.name + ".npy")).string();
    bool              ok   = false;

    if (is_3d_field(entry))
    {
        ok = npy::write_3d(entry.values().data(), entry.dim0, entry.dim1, entry.dim2, path);
    }
    else
    {
        ok = npy::write_2d(entry.values().data(), entry.dim0, two_dimensional_cols(entry), path);
    }

    if (ok)
    {
        bytes_written += entry.values().size() * sizeof(float);
    }

    return ok;
}

#ifdef HAVE_ZFP
double AsyncOutputWriter::zfp_tolerance_for_field(const std::string& field_name) const
{
    return effective_zfp_accuracy_tolerance(config_, field_name);
}

std::string AsyncOutputWriter::zfp_history_key(const std::string& field_name) const
{
    return active_zfp_history_namespace_ + '\n' + field_name;
}

bool AsyncOutputWriter::has_predictive_zfp_delta_reference(const FieldSnapshotEntry& entry) const
{
    if (!config_.zfp_predictive_delta)
    {
        return false;
    }

    const auto it = previous_fields_2_.find(zfp_history_key(entry.name));
    return it != previous_fields_2_.end() && it->second.size() == entry.values().size();
}

void AsyncOutputWriter::apply_zfp_prefilters(std::vector<float>& buffer,
                                             std::size_t         count,
                                             bool                apply_sparse,
                                             bool                apply_f16) const
{
    if (apply_sparse)
    {
        threshold_sparse_buffer(buffer, count, config_.zfp_sparse_threshold);
    }
    if (apply_f16)
    {
        quantize_buffer_to_half(buffer, count);
    }
}

bool AsyncOutputWriter::write_zfp_keyframe(const FieldSnapshotEntry&   entry,
                                           const std::string&          path,
                                           double                      field_tolerance,
                                           bool                        apply_sparse,
                                           bool                        apply_f16,
                                           std::size_t&                field_bytes,
                                           const float*&               compressed_data,
                                           float*                      decoded_output,
                                           std::vector<float>&         brick_input,
                                           std::vector<float>&         brick_decoded,
                                           std::vector<unsigned char>& compressed_buffer,
                                           std::vector<uint64_t>&      brick_offsets,
                                           std::string&                error)
{
    if (apply_sparse || apply_f16)
    {
        std::copy(entry.values().begin(), entry.values().end(), scratch_buffer_.begin());
        apply_zfp_prefilters(scratch_buffer_, entry.values().size(), apply_sparse, apply_f16);
        compressed_data = scratch_buffer_.data();
    }
    else
    {
        compressed_data = entry.values().data();
    }

    return write_zfp_3d(compressed_data,
                        entry.dim0,
                        zfp_dim1(entry),
                        zfp_dim2(entry),
                        config_.zfp_mode,
                        field_tolerance,
                        config_.zfp_precision_bits,
                        config_.zfp_rate_bps,
                        0x00,
                        path,
                        decoded_output,
                        brick_input,
                        brick_decoded,
                        compressed_buffer,
                        brick_offsets,
                        *zfp_workspace_,
                        field_bytes,
                        error);
}

bool AsyncOutputWriter::write_zfp_predictive_delta(const FieldSnapshotEntry&   entry,
                                                   const std::string&          path,
                                                   double                      field_tolerance,
                                                   bool                        apply_sparse,
                                                   bool                        apply_f16,
                                                   std::size_t&                field_bytes,
                                                   const float*&               compressed_data,
                                                   float*                      decoded_output,
                                                   std::vector<float>&         brick_input,
                                                   std::vector<float>&         brick_decoded,
                                                   std::vector<unsigned char>& compressed_buffer,
                                                   std::vector<uint64_t>&      brick_offsets,
                                                   std::string&                error)
{
    const std::size_t n     = entry.values().size();
    const std::string key   = zfp_history_key(entry.name);
    const auto&       prev  = previous_fields_[key];
    const auto&       prev2 = previous_fields_2_[key];
    for (std::size_t idx = 0; idx < n; ++idx)
    {
        const double residual = static_cast<double>(entry.values()[idx]) - 2.0 * static_cast<double>(prev[idx]) +
                                static_cast<double>(prev2[idx]);
        scratch_buffer_[idx]  = static_cast<float>(residual);
    }

    apply_zfp_prefilters(scratch_buffer_, n, apply_sparse, apply_f16);
    compressed_data = scratch_buffer_.data();

    return write_zfp_3d(compressed_data,
                        entry.dim0,
                        zfp_dim1(entry),
                        zfp_dim2(entry),
                        config_.zfp_mode,
                        field_tolerance,
                        config_.zfp_precision_bits,
                        config_.zfp_rate_bps,
                        0x03,
                        path,
                        decoded_output,
                        brick_input,
                        brick_decoded,
                        compressed_buffer,
                        brick_offsets,
                        *zfp_workspace_,
                        field_bytes,
                        error);
}

bool AsyncOutputWriter::write_zfp_simple_delta(const FieldSnapshotEntry&   entry,
                                               const std::string&          path,
                                               double                      field_tolerance,
                                               bool                        apply_sparse,
                                               bool                        apply_f16,
                                               std::size_t&                field_bytes,
                                               const float*&               compressed_data,
                                               float*                      decoded_output,
                                               std::vector<float>&         brick_input,
                                               std::vector<float>&         brick_decoded,
                                               std::vector<unsigned char>& compressed_buffer,
                                               std::vector<uint64_t>&      brick_offsets,
                                               std::string&                error)
{
    const std::size_t n   = entry.values().size();
    const std::string key = zfp_history_key(entry.name);
    const auto&       prev = previous_fields_[key];
    for (std::size_t idx = 0; idx < n; ++idx)
    {
        scratch_buffer_[idx] =
            static_cast<float>(static_cast<double>(entry.values()[idx]) - static_cast<double>(prev[idx]));
    }

    apply_zfp_prefilters(scratch_buffer_, n, apply_sparse, apply_f16);
    compressed_data = scratch_buffer_.data();

    return write_zfp_3d(compressed_data,
                        entry.dim0,
                        zfp_dim1(entry),
                        zfp_dim2(entry),
                        config_.zfp_mode,
                        field_tolerance,
                        config_.zfp_precision_bits,
                        config_.zfp_rate_bps,
                        0x01,
                        path,
                        decoded_output,
                        brick_input,
                        brick_decoded,
                        compressed_buffer,
                        brick_offsets,
                        *zfp_workspace_,
                        field_bytes,
                        error);
}

bool AsyncOutputWriter::update_zfp_delta_history(const FieldSnapshotEntry& entry,
                                                 const float*              compressed_data,
                                                 const float*              decoded_data,
                                                 bool                      is_keyframe,
                                                 double                    field_tolerance,
                                                 std::string&              error)
{
    const std::size_t n   = entry.values().size();
    const std::string key = zfp_history_key(entry.name);

#ifdef NIMBUS_ZFP_RECOMPRESS_HISTORY
    (void)decoded_data;
    // Store the ZFP-roundtripped data because the reader reconstructs future
    // delta frames from decompressed references. This comparator restores the
    // former second compression and per-frame temporary buffers.
    std::vector<float> decoded;
    const bool         have_decoded = zfp_roundtrip_in_memory(compressed_data,
                                                              entry.dim0,
                                                              entry.dim1,
                                                              entry.dim2,
                                                              config_.zfp_mode,
                                                              field_tolerance,
                                                              config_.zfp_precision_bits,
                                                              config_.zfp_rate_bps,
                                                              decoded);

    std::vector<float> new_prev;
    if (have_decoded)
    {
        const bool first_frame = !previous_fields_.count(key) || previous_fields_[key].empty();
        if (is_keyframe || first_frame)
        {
            new_prev = std::move(decoded);
        }
        else
        {
            const auto& prev            = previous_fields_[key];
            const bool  used_predictive = config_.zfp_predictive_delta && previous_fields_2_.count(key) &&
                                          previous_fields_2_[key].size() == n;

            new_prev.resize(n);
            if (used_predictive)
            {
                const auto& prev2 = previous_fields_2_[key];
                for (std::size_t idx = 0; idx < n; ++idx)
                {
                    new_prev[idx] =
                        static_cast<float>(static_cast<double>(decoded[idx]) + 2.0 * static_cast<double>(prev[idx]) -
                                           static_cast<double>(prev2[idx]));
                }
            }
            else
            {
                for (std::size_t idx = 0; idx < n; ++idx)
                {
                    new_prev[idx] =
                        static_cast<float>(static_cast<double>(decoded[idx]) + static_cast<double>(prev[idx]));
                }
            }
        }
    }
    else
    {
        error = "ZFP: failed to reconstruct the decoded delta reference for field '" + entry.name + "'";
        return false;
    }

    if (is_keyframe)
    {
        previous_fields_2_.erase(key);
    }
    else if (config_.zfp_predictive_delta)
    {
        auto prev_it = previous_fields_.find(key);
        if (prev_it != previous_fields_.end() && !prev_it->second.empty())
        {
            previous_fields_2_[key] = std::move(prev_it->second);
        }
    }
    previous_fields_[key] = std::move(new_prev);
    return true;
#else
    (void)compressed_data;
    (void)field_tolerance;
    if (decoded_data == nullptr)
    {
        error = "ZFP: missing decoded delta reference for field '" + entry.name + "'";
        return false;
    }

    auto       prev_it     = previous_fields_.find(key);
    const bool first_frame = prev_it == previous_fields_.end() || prev_it->second.empty();
    if (is_keyframe || first_frame)
    {
        auto& prev = previous_fields_[key];
        prev.assign(decoded_data, decoded_data + n);
        // Preserve capacity across keyframes; size, not map membership, is the
        // authoritative predictive-history availability contract.
        auto prev2_it = previous_fields_2_.find(key);
        if (prev2_it != previous_fields_2_.end())
        {
            prev2_it->second.clear();
        }
        return true;
    }

    auto& prev = prev_it->second;
    if (prev.size() != n)
    {
        error = "ZFP: previous delta reference shape changed for field '" + entry.name + "'";
        return false;
    }

    if (!config_.zfp_predictive_delta)
    {
        // The old reference is dead after this simple-delta reconstruction,
        // so update it in place and retain its allocation.
        for (std::size_t idx = 0; idx < n; ++idx)
        {
            prev[idx] = static_cast<float>(static_cast<double>(decoded_data[idx]) + static_cast<double>(prev[idx]));
        }
        return true;
    }

    auto& prev2 = previous_fields_2_[key];
    if (prev2.size() == n)
    {
        // Rotate both decoded references in place. Each old value is consumed
        // before its slot is overwritten, preserving the literal predictive
        // reconstruction order without a third full-field buffer.
        for (std::size_t idx = 0; idx < n; ++idx)
        {
            const float old_prev = prev[idx];
            prev[idx]  = static_cast<float>(static_cast<double>(decoded_data[idx]) +
                                            2.0 * static_cast<double>(old_prev) - static_cast<double>(prev2[idx]));
            prev2[idx] = old_prev;
        }
    }
    else
    {
        // The first delta after a keyframe has only one reference. Populate
        // the second persistent history while reconstructing current in place;
        // this is the sole unavoidable capacity establishment for prev2.
        prev2.resize(n);
        for (std::size_t idx = 0; idx < n; ++idx)
        {
            const float old_prev = prev[idx];
            prev[idx]  = static_cast<float>(static_cast<double>(decoded_data[idx]) + static_cast<double>(old_prev));
            prev2[idx] = old_prev;
        }
    }
    return true;
#endif
}

bool AsyncOutputWriter::write_zfp_field(const FieldSnapshotEntry&    entry,
                                        const std::filesystem::path& step_dir,
                                        bool                         is_keyframe,
                                        bool                         delta_enabled,
                                        std::size_t&                 bytes_written,
                                        std::string&                 error)
{
    const std::string path            = (step_dir / (entry.name + ".zfp3d")).string();
    if ((config_.zfp_reversible_radar_v4 || reversible_radar_v4_test_only_) &&
        is_reversible_radar_field(entry.name))
    {
        if (!is_zfp_encodable_field(entry) ||
            !validate_reversible_radar_source(entry, error))
        {
            if (error.empty())
                error = "ZFP: radar field has inconsistent dimensions";
            return false;
        }
        std::size_t field_bytes = 0;
        const bool ok = write_zfp_3d_reversible(
            entry.values().data(), entry.dim0, zfp_dim1(entry), zfp_dim2(entry),
            path, static_cast<std::size_t>(config_.zfp_field_scratch_mib) *
                1024ULL * 1024ULL, field_bytes, error);
        if (!ok)
            return false;
        previous_fields_.erase(zfp_history_key(entry.name));
        previous_fields_2_.erase(zfp_history_key(entry.name));
        bytes_written += field_bytes;
        return true;
    }
    const double      field_tolerance = zfp_tolerance_for_field(entry.name);

    const bool apply_sparse = config_.zfp_sparse_threshold > 0.0f && is_sparse_eligible(entry.name);
    const bool apply_f16    = config_.zfp_float16_prefilter && is_float16_eligible(entry.name);

    const std::size_t n = entry.values().size();
    if (!is_zfp_encodable_field(entry))
    {
        error = "ZFP: field '" + entry.name + "' has inconsistent dimensions";
        return false;
    }
    if (!zfp_workspace_)
    {
        zfp_workspace_ = std::make_unique<ZfpWriterWorkspace>();
    }
    const std::size_t d0 = static_cast<std::size_t>(entry.dim0);
    const std::size_t d1 = static_cast<std::size_t>(zfp_dim1(entry));
    const std::size_t d2 = static_cast<std::size_t>(zfp_dim2(entry));
    if (d0 > n || d1 > n / d0 || d2 > n / (d0 * d1) || d0 * d1 * d2 != n)
    {
        error = "ZFP: field '" + entry.name + "' has inconsistent dimensions";
        return false;
    }
    const bool needs_full_field_scratch =
        delta_enabled || apply_sparse || apply_f16;
    if (needs_full_field_scratch && scratch_buffer_.size() < n)
    {
        scratch_buffer_.resize(n);
    }

    std::size_t  field_bytes     = 0;
    const float* compressed_data = nullptr;
#ifdef NIMBUS_ZFP_RECOMPRESS_HISTORY
    float* decoded_output = nullptr;
#else
    float* decoded_output = delta_enabled ? scratch_buffer_.data() : nullptr;
#endif
    const std::string history_key = zfp_history_key(entry.name);
    const bool ok = (is_keyframe || previous_fields_.find(history_key) == previous_fields_.end())
                        ? write_zfp_keyframe(entry,
                                             path,
                                             field_tolerance,
                                             apply_sparse,
                                             apply_f16,
                                             field_bytes,
                                             compressed_data,
                                             decoded_output,
                                             zfp_brick_input_,
                                             zfp_brick_decoded_,
                                             zfp_compressed_buffer_,
                                             zfp_brick_offsets_,
                                             error)
                    : has_predictive_zfp_delta_reference(entry) ? write_zfp_predictive_delta(entry,
                                                                                             path,
                                                                                             field_tolerance,
                                                                                             apply_sparse,
                                                                                             apply_f16,
                                                                                             field_bytes,
                                                                                             compressed_data,
                                                                                             decoded_output,
                                                                                             zfp_brick_input_,
                                                                                             zfp_brick_decoded_,
                                                                                             zfp_compressed_buffer_,
                                                                                             zfp_brick_offsets_,
                                                                                             error)
                                                                : write_zfp_simple_delta(entry,
                                                                                         path,
                                                                                         field_tolerance,
                                                                                         apply_sparse,
                                                                                         apply_f16,
                                                                                         field_bytes,
                                                                                         compressed_data,
                                                                                         decoded_output,
                                                                                         zfp_brick_input_,
                                                                                         zfp_brick_decoded_,
                                                                                         zfp_compressed_buffer_,
                                                                                         zfp_brick_offsets_,
                                                                                         error);

    if (!ok)
    {
        return false;
    }

    if (delta_enabled)
    {
        if (!update_zfp_delta_history(entry, compressed_data, decoded_output, is_keyframe, field_tolerance, error))
        {
            return false;
        }
    }
    bytes_written += field_bytes;

    return true;
}

bool AsyncOutputWriter::write_zfp_snapshot_parallel(
    const ExportSnapshot& snapshot,
    bool is_keyframe,
    bool delta_enabled,
    std::size_t& bytes_written,
    std::string& error)
{
    struct TaskResult
    {
        std::size_t entry_index = 0;
        std::string history_key;
        std::filesystem::path temporary_path;
        std::filesystem::path final_path;
        std::size_t bytes = 0;
        bool keyframe = true;
        bool predictive = false;
        bool reversible = false;
        bool inserted_prev = false;
        bool inserted_prev2 = false;
        bool ok = false;
        std::string error;
        std::vector<float> decoded;
        std::vector<float> new_prev2;
        std::vector<float> old_prev;
        std::vector<float> old_prev2;
    };

    std::vector<TaskResult> tasks;
    std::size_t maximum_input_elements = 0;
    std::size_t decoded_stage_bytes = 0;
    const bool filters = config_.zfp_sparse_threshold > 0.0f ||
                         config_.zfp_float16_prefilter;
    for (std::size_t index = 0; index < snapshot.fields.size(); ++index)
    {
        const auto& entry = snapshot.fields[index];
        const auto storage = nimbus::output::classify_zfp_field_storage(entry);
        if (storage == nimbus::output::ZfpFieldStorage::InvalidShape)
        {
            error = "ZFP: field '" + entry.name + "' has inconsistent dimensions";
            return false;
        }
        if (storage == nimbus::output::ZfpFieldStorage::LosslessNpy)
        {
            continue;
        }
        const std::size_t n = entry.values().size();
        const std::size_t d0 = entry.dim0 > 0
            ? static_cast<std::size_t>(entry.dim0) : 0;
        const std::size_t d1 = static_cast<std::size_t>(zfp_dim1(entry));
        const std::size_t d2 = static_cast<std::size_t>(zfp_dim2(entry));
        if (d0 == 0 || d1 == 0 || d2 == 0 || d0 > n || d1 > n / d0 ||
            d2 > n / (d0 * d1) || d0 * d1 * d2 != n)
        {
            error = "ZFP: field '" + entry.name +
                    "' has inconsistent dimensions";
            return false;
        }
        TaskResult task;
        task.entry_index = index;
        task.history_key = zfp_history_key(entry.name);
        task.reversible = (config_.zfp_reversible_radar_v4 ||
                           reversible_radar_v4_test_only_) &&
            is_reversible_radar_field(entry.name);
        task.final_path = snapshot.step_dir / (entry.name + ".zfp3d");
        task.temporary_path = snapshot.step_dir /
            (entry.name + ".zfp3d.parallel.tmp");
        auto prev = previous_fields_.find(task.history_key);
        task.keyframe = task.reversible || is_keyframe || prev == previous_fields_.end() ||
                        prev->second.empty();
        const auto prev2 = previous_fields_2_.find(task.history_key);
        task.predictive = !task.reversible && !task.keyframe && config_.zfp_predictive_delta &&
                          prev2 != previous_fields_2_.end() &&
                          prev2->second.size() == n;
        if (!task.keyframe &&
            (prev == previous_fields_.end() || prev->second.size() != n))
        {
            error = "ZFP: previous delta reference shape changed for field '" +
                    entry.name + "'";
            return false;
        }
        if (delta_enabled && !task.reversible)
        {
            if (n > std::numeric_limits<std::size_t>::max() / sizeof(float) ||
                decoded_stage_bytes > std::numeric_limits<std::size_t>::max() -
                    n * sizeof(float))
            {
                error = "ZFP: parallel decoded-history scratch overflow";
                return false;
            }
            decoded_stage_bytes += n * sizeof(float);
            if (!task.keyframe && config_.zfp_predictive_delta)
            {
                if (decoded_stage_bytes > std::numeric_limits<std::size_t>::max() -
                        n * sizeof(float))
                {
                    error = "ZFP: parallel predictive-history scratch overflow";
                    return false;
                }
                decoded_stage_bytes += n * sizeof(float);
            }
        }
        if (!task.reversible)
            maximum_input_elements = std::max(maximum_input_elements, n);
        tasks.push_back(std::move(task));
    }

    if (tasks.empty())
    {
        for (const auto& entry : snapshot.fields)
            if (!write_npy_field(entry, snapshot.step_dir, bytes_written))
            {
                error = "Failed to write field " + entry.name;
                return false;
            }
        return write_snapshot_sidecars(snapshot, bytes_written, error);
    }

    const std::size_t worker_count = std::min<std::size_t>(
        tasks.size(), static_cast<std::size_t>(std::max(
            1, config_.zfp_resolved_field_workers)));
    constexpr std::size_t codec_reserve_per_worker = 2ULL * 1024ULL * 1024ULL;
    const std::size_t worker_input_bytes = (delta_enabled || filters)
        ? worker_count * maximum_input_elements * sizeof(float) : 0;
    const std::size_t codec_reserve_bytes =
        worker_count * codec_reserve_per_worker;
    const std::size_t budget_bytes =
        static_cast<std::size_t>(config_.zfp_field_scratch_mib) *
        1024ULL * 1024ULL;
    if (worker_input_bytes > budget_bytes ||
        decoded_stage_bytes > budget_bytes - worker_input_bytes ||
        codec_reserve_bytes > budget_bytes - worker_input_bytes -
            decoded_stage_bytes)
    {
        error = "ZFP: field-parallel transactional scratch exceeds output.zfp_field_scratch_mib";
        return false;
    }

    while (zfp_parallel_workers_.size() < worker_count)
        zfp_parallel_workers_.push_back(
            std::make_unique<ZfpParallelWorkerState>());

    for (auto& task : tasks)
    {
        if (delta_enabled && !task.reversible)
        {
            auto [prev, inserted_prev] =
                previous_fields_.try_emplace(task.history_key);
            auto [prev2, inserted_prev2] =
                previous_fields_2_.try_emplace(task.history_key);
            (void)prev;
            (void)prev2;
            task.inserted_prev = inserted_prev;
            task.inserted_prev2 = inserted_prev2;
            task.decoded.resize(
                snapshot.fields[task.entry_index].values().size());
        }
        std::error_code ec;
        std::filesystem::remove(task.temporary_path, ec);
    }

    std::vector<std::filesystem::path> committed_paths;
    committed_paths.reserve(tasks.size());
    const auto rollback_insertions = [&]()
    {
        for (const auto& task : tasks)
        {
            if (task.inserted_prev)
                previous_fields_.erase(task.history_key);
            if (task.inserted_prev2)
                previous_fields_2_.erase(task.history_key);
        }
    };
    const auto cleanup_files = [&]()
    {
        std::error_code ec;
        for (const auto& task : tasks)
        {
            std::filesystem::remove(task.temporary_path, ec);
            ec.clear();
        }
        for (const auto& path : committed_paths)
        {
            std::filesystem::remove(path, ec);
            ec.clear();
        }
    };

    const auto parallel_started = std::chrono::steady_clock::now();
    std::atomic<std::size_t> next_task{0};
    std::vector<std::thread> workers;
    workers.reserve(worker_count);
    try
    {
        for (std::size_t worker_index = 0; worker_index < worker_count;
             ++worker_index)
        {
            workers.emplace_back([&, worker_index]
            {
                auto& worker = *zfp_parallel_workers_[worker_index];
                while (true)
                {
                    const std::size_t task_index =
                        next_task.fetch_add(1, std::memory_order_relaxed);
                    if (task_index >= tasks.size())
                        break;
                    auto& task = tasks[task_index];
                    try
                    {
                    const auto& entry = snapshot.fields[task.entry_index];
                    if (task.reversible)
                    {
                        task.ok = validate_reversible_radar_source(entry, task.error) &&
                            write_zfp_3d_reversible(
                                entry.values().data(), entry.dim0, zfp_dim1(entry),
                                zfp_dim2(entry), task.temporary_path.string(),
                                codec_reserve_per_worker, task.bytes, task.error);
                        continue;
                    }
                    const std::size_t n = entry.values().size();
                    const bool apply_sparse =
                        config_.zfp_sparse_threshold > 0.0f &&
                        is_sparse_eligible(entry.name);
                    const bool apply_f16 = config_.zfp_float16_prefilter &&
                        is_float16_eligible(entry.name);
                    const float* source = entry.values().data();
                    std::uint8_t flags = 0;
                    if (!task.keyframe || apply_sparse || apply_f16)
                    {
                        worker.input_scratch.resize(n);
                        if (task.keyframe)
                        {
                            std::copy(entry.values().begin(), entry.values().end(),
                                      worker.input_scratch.begin());
                        }
                        else
                        {
                            const auto& prev =
                                previous_fields_.at(task.history_key);
                            if (task.predictive)
                            {
                                const auto& prev2 =
                                    previous_fields_2_.at(task.history_key);
                                for (std::size_t value = 0; value < n; ++value)
                                    worker.input_scratch[value] = static_cast<float>(
                                        static_cast<double>(entry.values()[value]) -
                                        2.0 * static_cast<double>(prev[value]) +
                                        static_cast<double>(prev2[value]));
                                flags = 0x03;
                            }
                            else
                            {
                                for (std::size_t value = 0; value < n; ++value)
                                    worker.input_scratch[value] = static_cast<float>(
                                        static_cast<double>(entry.values()[value]) -
                                        static_cast<double>(prev[value]));
                                flags = 0x01;
                            }
                        }
                        apply_zfp_prefilters(worker.input_scratch, n,
                                             apply_sparse, apply_f16);
                        source = worker.input_scratch.data();
                    }
                    task.ok = write_zfp_3d(
                        source, entry.dim0, zfp_dim1(entry), zfp_dim2(entry),
                        config_.zfp_mode, zfp_tolerance_for_field(entry.name),
                        config_.zfp_precision_bits, config_.zfp_rate_bps, flags,
                        task.temporary_path.string(),
                        delta_enabled ? task.decoded.data() : nullptr,
                        worker.brick_input, worker.brick_decoded,
                        worker.compressed_buffer, worker.brick_offsets,
                        worker.workspace, task.bytes, task.error);
                    }
                    catch (const std::exception& exception)
                    {
                        task.ok = false;
                        assign_output_error(task.error, exception.what());
                    }
                    catch (...)
                    {
                        task.ok = false;
                        assign_output_error(task.error, "ZFP: field-parallel worker failed");
                    }
                }
            });
        }
    }
    catch (const std::exception& exception)
    {
        for (auto& worker : workers)
            worker.join();
        cleanup_files();
        rollback_insertions();
        error = std::string("ZFP: cannot start field workers: ") + exception.what();
        return false;
    }
    for (auto& worker : workers)
        worker.join();
    const double parallel_elapsed = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - parallel_started).count();

    for (const auto& task : tasks)
    {
        if (!task.ok)
        {
            error = task.error.empty()
                ? "ZFP: field-parallel task failed" : task.error;
            cleanup_files();
            rollback_insertions();
            return false;
        }
    }

    if (delta_enabled)
    {
        for (auto& task : tasks)
        {
            if (task.reversible || task.keyframe)
                continue;
            const auto& prev = previous_fields_.at(task.history_key);
            if (config_.zfp_predictive_delta)
                task.new_prev2 = prev;
            if (task.predictive)
            {
                const auto& prev2 = previous_fields_2_.at(task.history_key);
                for (std::size_t value = 0; value < task.decoded.size();
                     ++value)
                    task.decoded[value] = static_cast<float>(
                        static_cast<double>(task.decoded[value]) +
                        2.0 * static_cast<double>(prev[value]) -
                        static_cast<double>(prev2[value]));
            }
            else
            {
                for (std::size_t value = 0; value < task.decoded.size();
                     ++value)
                    task.decoded[value] = static_cast<float>(
                        static_cast<double>(task.decoded[value]) +
                        static_cast<double>(prev[value]));
            }
        }
    }

    std::size_t actual_scratch = 0;
    for (std::size_t worker = 0; worker < worker_count; ++worker)
        actual_scratch +=
            zfp_parallel_workers_[worker]->scratch_capacity_bytes();
    for (const auto& task : tasks)
        actual_scratch += (task.decoded.capacity() +
                           task.new_prev2.capacity()) * sizeof(float);
    if (config_.zfp_reversible_radar_v4 || reversible_radar_v4_test_only_)
        actual_scratch += worker_count * codec_reserve_per_worker;
    if (actual_scratch > budget_bytes)
    {
        error = "ZFP: actual field-parallel scratch exceeded its hard budget";
        cleanup_files();
        rollback_insertions();
        return false;
    }

    for (auto& task : tasks)
    {
        std::error_code ec;
        if (std::filesystem::exists(task.final_path, ec) || ec)
        {
            error = "ZFP: refusing to replace an existing parallel field artifact";
            cleanup_files();
            rollback_insertions();
            return false;
        }
        std::filesystem::rename(task.temporary_path, task.final_path, ec);
        if (ec)
        {
            error = "ZFP: failed ordered field commit: " + ec.message();
            cleanup_files();
            rollback_insertions();
            return false;
        }
        committed_paths.push_back(task.final_path);
        bytes_written += task.bytes;
    }

    if (delta_enabled)
    {
        for (auto& task : tasks)
        {
            if (task.reversible)
                continue;
            auto& prev = previous_fields_.at(task.history_key);
            auto& prev2 = previous_fields_2_.at(task.history_key);
            task.old_prev.swap(prev);
            task.old_prev2.swap(prev2);
            prev.swap(task.decoded);
            prev2.swap(task.new_prev2);
        }
    }

    std::vector<std::filesystem::path> non_zfp_paths;
    for (const auto& entry : snapshot.fields)
    {
        const auto storage = nimbus::output::classify_zfp_field_storage(entry);
        if (storage == nimbus::output::ZfpFieldStorage::Compressed)
            continue;
        if (storage == nimbus::output::ZfpFieldStorage::InvalidShape)
        {
            error = "ZFP: field '" + entry.name + "' has inconsistent dimensions";
            goto rollback_parallel_commit;
        }
        if (!write_npy_field(entry, snapshot.step_dir, bytes_written))
        {
            error = "Failed to write field " + entry.name;
            non_zfp_paths.push_back(snapshot.step_dir / (entry.name + ".npy"));
            goto rollback_parallel_commit;
        }
        non_zfp_paths.push_back(snapshot.step_dir / (entry.name + ".npy"));
    }
    if (!write_snapshot_sidecars(snapshot, bytes_written, error))
    {
        if (std::filesystem::is_regular_file(
                snapshot.step_dir / "manifest.json"))
            return false;
        goto rollback_parallel_commit;
    }
    for (const auto& task : tasks)
        if (task.reversible)
        {
            previous_fields_.erase(task.history_key);
            previous_fields_2_.erase(task.history_key);
        }

    {
        std::size_t previous_high_water =
            zfp_parallel_scratch_high_water_bytes_.load(
                std::memory_order_relaxed);
        while (previous_high_water < actual_scratch &&
               !zfp_parallel_scratch_high_water_bytes_.compare_exchange_weak(
                   previous_high_water, actual_scratch,
                   std::memory_order_relaxed))
        {
        }
        zfp_parallel_tasks_completed_.fetch_add(
            tasks.size(), std::memory_order_relaxed);
        double prior = zfp_parallel_wall_time_s_.load(std::memory_order_relaxed);
        while (!zfp_parallel_wall_time_s_.compare_exchange_weak(
            prior, prior + parallel_elapsed, std::memory_order_relaxed))
        {
        }
    }
    return true;

rollback_parallel_commit:
    if (delta_enabled)
    {
        for (auto& task : tasks)
        {
            if (task.reversible)
                continue;
            auto& prev = previous_fields_.at(task.history_key);
            auto& prev2 = previous_fields_2_.at(task.history_key);
            prev.swap(task.old_prev);
            prev2.swap(task.old_prev2);
        }
    }
    for (const auto& path : non_zfp_paths)
    {
        std::error_code ec;
        std::filesystem::remove(path, ec);
    }
    cleanup_files();
    rollback_insertions();
    return false;
}
#endif

bool AsyncOutputWriter::write_field(const FieldSnapshotEntry&    entry,
                                    const std::filesystem::path& step_dir,
                                    bool                         use_csv,
                                    bool                         use_zfp,
                                    bool                         is_keyframe,
                                    bool                         delta_enabled,
                                    std::size_t&                 bytes_written,
                                    std::string&                 error)
{
    bool ok = false;
    if (use_csv && is_3d_field(entry))
    {
        ok = write_csv_field(entry, step_dir, bytes_written);
    }
#ifdef HAVE_ZFP
    else if (use_zfp)
    {
        const auto storage = nimbus::output::classify_zfp_field_storage(entry);
        if (storage == nimbus::output::ZfpFieldStorage::InvalidShape)
        {
            error = "ZFP: field '" + entry.name + "' has inconsistent dimensions";
            return false;
        }
        ok = storage == nimbus::output::ZfpFieldStorage::Compressed
            ? write_zfp_field(entry, step_dir, is_keyframe, delta_enabled,
                              bytes_written, error)
            : write_npy_field(entry, step_dir, bytes_written);
    }
#endif
    else
    {
        ok = write_npy_field(entry, step_dir, bytes_written);
    }

#ifndef HAVE_ZFP
    (void)use_zfp;
    (void)is_keyframe;
    (void)delta_enabled;
#endif

    if (!ok && error.empty())
    {
        error = "Failed to write field " + entry.name;
    }

    return ok;
}

bool AsyncOutputWriter::write_snapshot_sidecars(const ExportSnapshot& snapshot,
                                                std::size_t&          bytes_written,
                                                std::string&          error) const
{
    std::vector<std::filesystem::path> brick_hierarchy_paths;
    std::vector<std::filesystem::path> brick_lod_paths;
    std::map<int, nimbus::output::FieldBrickLodPayloadIdentity::Geometry>
        terrain_lod_geometry;
    std::string completed_manifest_json = snapshot.manifest_json;
    if (!snapshot.publication_generation.empty())
    {
        for (const auto& entry : snapshot.fields)
        {
            if (!is_3d_field(entry)) continue;
            const std::string token = field_brick_hierarchy_sha256_token(entry.name);
            const auto token_position = completed_manifest_json.find(token);
            if (token_position == std::string::npos) continue;
            const auto path = snapshot.step_dir / (entry.name + ".bricks.json");
            try
            {
                const bool zfp_payload = config_.format == OutputFormat::zfp &&
                    is_zfp_encodable_field(entry);
                const auto zfp_padding = zfp_payload
                    ? nimbus::output::zfp_decoded_extrema_padding(
                        config_.zfp_mode,
                        effective_zfp_accuracy_tolerance(config_, entry.name),
                        (config_.zfp_sparse_threshold > 0.0f && is_sparse_eligible(entry.name))
                            ? config_.zfp_sparse_threshold : 0.0,
                        config_.zfp_float16_prefilter && is_float16_eligible(entry.name))
                    : std::optional<double>{};
                const bool zfp_bounds_safe = zfp_padding.has_value();
                const bool culling_safe = !zfp_payload || zfp_bounds_safe;
                const double extrema_padding = zfp_padding.value_or(0.0);
                const auto publish_lod = [&](int level,
                                             const std::array<int, 3>& shape,
                                             const std::vector<float>& data,
                                             const nimbus::output::FieldBrickLodGeometry* geometry)
                    -> nimbus::output::FieldBrickLodPayloadIdentity
                {
                    const std::string filename = entry.name + ".lod" +
                        std::to_string(level) + ".npy";
                    const auto lod_path = snapshot.step_dir / filename;
                    if (!npy::write_3d(data.data(), shape[0], shape[1], shape[2],
                                       lod_path.string()))
                        throw std::runtime_error("failed to publish field brick LOD payload");
                    brick_lod_paths.push_back(lod_path);
                    std::error_code ec;
                    const auto size = std::filesystem::file_size(lod_path, ec);
                    if (ec) throw std::runtime_error(
                        "failed to stat field brick LOD payload: " + ec.message());
                    bytes_written += size;
                    nimbus::output::FieldBrickLodPayloadIdentity identity{
                        filename, file_sha256(lod_path), std::nullopt};
                    if (geometry)
                    {
                        auto found = terrain_lod_geometry.find(level);
                        if (found == terrain_lod_geometry.end())
                        {
                            nimbus::output::FieldBrickLodPayloadIdentity::Geometry published;
                            const std::array<std::string, 4> labels{
                                "physical_x_center_m", "physical_y_center_m",
                                "physical_z_center_m", "physical_volume_m3"};
                            const std::array<const std::vector<double>*, 4> values{
                                &geometry->x_m, &geometry->y_m,
                                &geometry->z_m, &geometry->volume_m3};
                            for (std::size_t component = 0; component < labels.size(); ++component)
                            {
                                const std::string geometry_filename = labels[component] +
                                    ".lod" + std::to_string(level) + ".npy";
                                const auto geometry_path = snapshot.step_dir / geometry_filename;
                                if (!npy::write_3d(values[component]->data(),
                                                   shape[0], shape[1], shape[2],
                                                   geometry_path.string()))
                                    throw std::runtime_error(
                                        "failed to publish terrain brick LOD geometry");
                                brick_lod_paths.push_back(geometry_path);
                                const auto geometry_size =
                                    std::filesystem::file_size(geometry_path, ec);
                                if (ec) throw std::runtime_error(
                                    "failed to stat terrain brick LOD geometry: " + ec.message());
                                bytes_written += geometry_size;
                                published.paths[component] = geometry_filename;
                                published.content_sha256[component] = file_sha256(geometry_path);
                            }
                            found = terrain_lod_geometry.emplace(level, std::move(published)).first;
                        }
                        identity.geometry = found->second;
                    }
                    return identity;
                };
                unsigned hierarchy_workers = 1;
#ifdef _OPENMP
                if (!config_.async_io) hierarchy_workers = static_cast<unsigned>(omp_get_max_threads());
#endif
                const std::string json = nimbus::output::build_field_brick_hierarchy_json(
                    entry, snapshot.grid, snapshot.publication_generation, culling_safe,
                    snapshot.data_product_kind != "retiring_swath_tile", extrema_padding,
                    publish_lod, hierarchy_workers);
                nimbus::crypto::Sha256 hierarchy_hash;
                hierarchy_hash.update(json.data(), json.size());
                const std::string hierarchy_sha256 = hierarchy_hash.hex_digest();
                if (!publish_output_json_atomically(path, json, error))
                    throw std::runtime_error(error.empty()
                        ? "failed to publish field brick hierarchy" : error);
                bytes_written += json.size();
                brick_hierarchy_paths.push_back(path);
                completed_manifest_json.replace(
                    token_position, token.size(), hierarchy_sha256);
            }
            catch (const std::exception& exception)
            {
                error = exception.what();
                for (const auto& published : brick_hierarchy_paths)
                {
                    std::error_code ec;
                    std::filesystem::remove(published, ec);
                }
                for (const auto& published : brick_lod_paths)
                {
                    std::error_code ec;
                    std::filesystem::remove(published, ec);
                }
                return false;
            }
        }
    }

    if (snapshot.grid.terrain_following && config_.format != OutputFormat::netcdf)
    {
        if (snapshot.grid.nx <= 0 || snapshot.grid.ny <= 0 || snapshot.grid.nz <= 0)
        {
            error = "terrain output requires positive grid dimensions";
            return false;
        }
        const std::size_t center_count    = static_cast<std::size_t>(snapshot.grid.nx) *
                                            static_cast<std::size_t>(snapshot.grid.ny) *
                                            static_cast<std::size_t>(snapshot.grid.nz);
        const std::size_t interface_count = static_cast<std::size_t>(snapshot.grid.nx) *
                                            static_cast<std::size_t>(snapshot.grid.ny) *
                                            static_cast<std::size_t>(snapshot.grid.nz + 1);
        const bool owns_geometry = !snapshot.grid.physical_z_center_m.empty() ||
                                   !snapshot.grid.physical_z_interface_m.empty();
        if ((owns_geometry &&
             (snapshot.grid.physical_z_center_m.size() != center_count ||
              snapshot.grid.physical_z_interface_m.size() != interface_count)) ||
            snapshot.grid.terrain_geometry_sha256.size() != 64 ||
            snapshot.grid.terrain_geometry_publication_generation.empty())
        {
            error = "terrain output is missing complete physical height geometry";
            return false;
        }
        const auto all_hex = [](const std::string& value)
        {
            return std::all_of(value.begin(), value.end(), [](unsigned char c)
            { return std::isxdigit(c) != 0; });
        };
        if (!all_hex(snapshot.grid.terrain_geometry_sha256) ||
            !all_hex(snapshot.grid.terrain_geometry_publication_generation))
        {
            error = "terrain geometry identity contains unsupported characters";
            return false;
        }
        if (owns_geometry)
        {
            if (terrain_geometry_content_sha256(snapshot.grid) !=
                snapshot.grid.terrain_geometry_sha256)
            {
                error = "terrain geometry payload does not match its content identity";
                return false;
            }
            for (int i = 0; i < snapshot.grid.nx; ++i)
                for (int j = 0; j < snapshot.grid.ny; ++j)
                    for (int k = 0; k < snapshot.grid.nz; ++k)
                    {
                        const std::size_t center_index =
                            (static_cast<std::size_t>(i) * snapshot.grid.ny + j) * snapshot.grid.nz + k;
                        const std::size_t face_index = (static_cast<std::size_t>(i) * snapshot.grid.ny + j) *
                                                           static_cast<std::size_t>(snapshot.grid.nz + 1) + k;
                        const double center = snapshot.grid.physical_z_center_m[center_index];
                        const double lower = snapshot.grid.physical_z_interface_m[face_index];
                        const double upper = snapshot.grid.physical_z_interface_m[face_index + 1];
                        if (!std::isfinite(center) || !std::isfinite(lower) ||
                            !std::isfinite(upper) || !(upper > lower) ||
                            !(center > lower) || !(center < upper))
                        {
                            error = "terrain output contains malformed physical height geometry";
                            return false;
                        }
                    }
        }
        const auto geometry_root = snapshot.step_dir.parent_path() / "geometry" /
            snapshot.grid.terrain_geometry_publication_generation;
        const auto geometry_dir = geometry_root /
            snapshot.grid.terrain_geometry_sha256;
        const auto center_path = geometry_dir / "physical_z_center_m.npy";
        const auto interface_path = geometry_dir / "physical_z_interface_m.npy";
        const auto geometry_manifest_path = geometry_dir / "manifest.json";
        if (!std::filesystem::is_directory(geometry_dir))
        {
            if (!owns_geometry)
            {
                error = "terrain geometry version was referenced before durable publication";
                return false;
            }
            const auto temporary = geometry_root /
                (snapshot.grid.terrain_geometry_sha256 + ".tmp");
            std::error_code ec;
            std::filesystem::remove_all(temporary, ec);
            ec.clear();
            std::filesystem::create_directories(temporary, ec);
            if (ec)
            {
                error = "failed to create terrain geometry publication: " + ec.message();
                return false;
            }
            const auto temporary_center = temporary / "physical_z_center_m.npy";
            const auto temporary_interface = temporary / "physical_z_interface_m.npy";
            if (!npy::write_3d(snapshot.grid.physical_z_center_m.data(),
                               snapshot.grid.nx, snapshot.grid.ny, snapshot.grid.nz,
                               temporary_center.string()) ||
                !npy::write_3d(snapshot.grid.physical_z_interface_m.data(),
                               snapshot.grid.nx, snapshot.grid.ny, snapshot.grid.nz + 1,
                               temporary_interface.string()))
            {
                error = "failed to write lossless terrain geometry version";
                std::filesystem::remove_all(temporary, ec);
                return false;
            }
            std::ostringstream geometry_manifest;
            geometry_manifest << "{\n  \"schema\": \"nimbus.terrain-geometry\",\n"
                << "  \"version\": 1,\n  \"publication_generation\": \""
                << snapshot.grid.terrain_geometry_publication_generation
                << "\",\n  \"content_sha256\": \""
                << snapshot.grid.terrain_geometry_sha256
                << "\",\n  \"identity_encoding\": \"shape_u64_le_then_ieee754_f64_le\","
                << "\n  \"dtype\": \"float64\",\n  \"layout\": \"x_y_z\",\n"
                << "  \"center_shape\": [" << snapshot.grid.nx << ", "
                << snapshot.grid.ny << ", " << snapshot.grid.nz << "],\n"
                << "  \"interface_shape\": [" << snapshot.grid.nx << ", "
                << snapshot.grid.ny << ", " << snapshot.grid.nz + 1 << "]\n}\n";
            if (!publish_output_json_atomically(
                    temporary / "manifest.json", geometry_manifest.str(), error) ||
                !nimbus::output::sync_directory_contents_to_disk(temporary, error))
            {
                std::filesystem::remove_all(temporary, ec);
                return false;
            }
            std::filesystem::create_directories(geometry_root, ec);
            std::filesystem::rename(temporary, geometry_dir, ec);
            if (ec)
            {
                error = "failed to commit terrain geometry version: " + ec.message();
                std::filesystem::remove_all(temporary, ec);
                return false;
            }
            if (!nimbus::output::sync_directory_to_disk(geometry_root, error))
                return false;
            bytes_written += std::filesystem::file_size(center_path, ec);
            if (ec) { error = "failed to stat terrain center geometry: " + ec.message(); return false; }
            bytes_written += std::filesystem::file_size(interface_path, ec);
            if (ec) { error = "failed to stat terrain interface geometry: " + ec.message(); return false; }
            bytes_written += std::filesystem::file_size(geometry_manifest_path, ec);
            if (ec) { error = "failed to stat terrain geometry manifest: " + ec.message(); return false; }
        }
        if (!std::filesystem::is_regular_file(center_path) ||
            !std::filesystem::is_regular_file(interface_path) ||
            !std::filesystem::is_regular_file(geometry_manifest_path))
        {
            error = "terrain geometry version is incomplete or corrupted";
            return false;
        }
        std::ifstream geometry_manifest_input(
            geometry_manifest_path, std::ios::binary);
        const std::string geometry_manifest_text{
            std::istreambuf_iterator<char>(geometry_manifest_input),
            std::istreambuf_iterator<char>()};
        if (geometry_manifest_text.empty() ||
            (geometry_manifest_input.bad() && !geometry_manifest_input.eof()) ||
            geometry_manifest_text.find(
                "\"schema\": \"nimbus.terrain-geometry\"") ==
                std::string::npos ||
            geometry_manifest_text.find(
                "\"publication_generation\": \"" +
                snapshot.grid.terrain_geometry_publication_generation + "\"") ==
                std::string::npos ||
            geometry_manifest_text.find(
                "\"content_sha256\": \"" +
                snapshot.grid.terrain_geometry_sha256 + "\"") ==
                std::string::npos)
        {
            error = "terrain geometry version manifest is corrupt or has the wrong identity";
            return false;
        }
    }

    if (!snapshot.validation_json.empty() && !snapshot.validation_path.empty())
    {
        std::error_code ec;
        std::filesystem::create_directories(snapshot.validation_path.parent_path(), ec);
        std::ofstream out(snapshot.validation_path);
        if (out)
        {
            out << snapshot.validation_json;
            bytes_written += snapshot.validation_json.size();
        }
    }
    // Completion manifest is published last, after every payload and sidecar.
    if (!snapshot.manifest_json.empty())
    {
        if (completed_manifest_json.find("nimbus-pending-field-brick-sha256:")
            != std::string::npos)
        {
            error = "completion manifest has an unpublished field brick hierarchy";
            return false;
        }
        // Manifest-last ordering only means something if the payloads are
        // already durable. Without this the manifest can reach the disk while
        // the fields it advertises are still page cache, and a reader after a
        // power cut sees a complete step whose data is zeros.
        if (!nimbus::output::sync_directory_contents_to_disk(snapshot.step_dir, error))
        {
            return false;
        }
        const auto manifest_path = snapshot.step_dir / "manifest.json";
        if (!nimbus::output::prepare_history_index_update(
                snapshot.step_dir.parent_path(), completed_manifest_json, error))
            return false;
        if (!publish_output_json_atomically(manifest_path, completed_manifest_json, error))
            return false;
        bytes_written += completed_manifest_json.size();
        if (!nimbus::output::publish_history_index_after_manifest(
                snapshot.step_dir.parent_path(), completed_manifest_json, error))
            return false;
    }
    return true;
}

bool AsyncOutputWriter::write_netcdf_snapshot(const ExportSnapshot& snapshot,
                                              std::size_t&          bytes_written,
                                              std::string&          error) const
{
    NetcdfHistoryPolicy policy;
    policy.deflate_level      = config_.netcdf_deflate_level;
    policy.shuffle            = config_.netcdf_shuffle;
    policy.target_chunk_bytes = static_cast<std::size_t>(config_.netcdf_target_chunk_kib) * 1024U;

    NetcdfHistoryWriteResult result;
    if (!write_netcdf4_history(snapshot, policy, result, error))
    {
        return false;
    }
    bytes_written += result.bytes_written;
    return true;
}

void AsyncOutputWriter::record_snapshot_write_stats(std::size_t bytes_written, double elapsed)
{
    total_bytes_.fetch_add(bytes_written, std::memory_order_relaxed);

    double old_time = total_time_s_.load(std::memory_order_relaxed);
    while (!total_time_s_.compare_exchange_weak(old_time, old_time + elapsed, std::memory_order_relaxed))
    {
    }
    double old_max_time = maximum_snapshot_write_s_.load(std::memory_order_relaxed);
    while (old_max_time < elapsed && !maximum_snapshot_write_s_.compare_exchange_weak(
               old_max_time, elapsed, std::memory_order_relaxed))
    {
    }
    std::size_t old_max_bytes = maximum_snapshot_bytes_.load(std::memory_order_relaxed);
    while (old_max_bytes < bytes_written && !maximum_snapshot_bytes_.compare_exchange_weak(
               old_max_bytes, bytes_written, std::memory_order_relaxed))
    {
    }
    snapshots_count_.fetch_add(1, std::memory_order_relaxed);
}

bool AsyncOutputWriter::write_snapshot(const ExportSnapshot& snapshot, std::string& error)
try
{
    const auto t0 = std::chrono::steady_clock::now();

    for (const auto& entry : snapshot.fields)
    {
        if (nimbus::output::classify_zfp_field_storage(entry) ==
            nimbus::output::ZfpFieldStorage::InvalidShape ||
            (entry.is_borrowed() && !entry.data.empty()))
        {
            error = "Output field '" + entry.name + "' has inconsistent dimensions or storage";
            return false;
        }
    }

    if (config_.zfp_reversible_radar_v4)
    {
#ifndef HAVE_ZFP
        error = "ZFP: reversible radar candidate requires libzfp";
        return false;
#endif
        if (config_.format != OutputFormat::zfp || config_.zfp_mode != "accuracy" ||
            config_.zfp_sparse_threshold != 0.0f ||
            config_.zfp_float16_prefilter ||
            snapshot.data_product_kind != "history")
        {
            error = "ZFP: reversible radar candidate requires an exact qualified mixed-history profile";
            return false;
        }
        bool has_radar = false;
        bool has_accuracy = false;
        for (const auto& entry : snapshot.fields)
        {
            if (!is_zfp_encodable_field(entry))
            {
                error = "ZFP: mixed reversible history requires every field to be ZFP encodable";
                return false;
            }
            if (is_reversible_radar_field(entry.name))
            {
                has_radar = true;
                if (!validate_reversible_radar_source(entry, error))
                    return false;
            }
            else
            {
                has_accuracy = true;
            }
        }
        if (!has_radar || !has_accuracy)
        {
            error = "ZFP: mixed reversible history requires radar and accuracy fields";
            return false;
        }
    }

    if (!ensure_step_directory(snapshot.step_dir, error))
    {
        return false;
    }

    const bool  use_zfp       = (config_.format == OutputFormat::zfp);
    const bool  use_csv       = (config_.format == OutputFormat::csv);
    std::size_t bytes_written = 0;

    if (config_.format == OutputFormat::netcdf)
    {
        if (!write_netcdf_snapshot(snapshot, bytes_written, error) ||
            !write_snapshot_sidecars(snapshot, bytes_written, error))
        {
            return false;
        }
        const auto t1 = std::chrono::steady_clock::now();
        record_snapshot_write_stats(bytes_written, std::chrono::duration<double>(t1 - t0).count());
        return true;
    }

    active_zfp_history_namespace_ = snapshot.data_product_kind;
    int&       delta_frame_counter = delta_frame_counters_[active_zfp_history_namespace_];
    const bool delta_enabled = use_zfp && config_.zfp_keyframe_interval > 0;
    const bool is_keyframe   = !delta_enabled || (delta_frame_counter % config_.zfp_keyframe_interval == 0);
    const bool field_parallel = use_zfp &&
        config_.zfp_field_execution == "field_parallel";
    if (delta_enabled && !field_parallel)
    {
        ++delta_frame_counter;
    }

#ifdef HAVE_ZFP
    if (field_parallel)
    {
        if (!write_zfp_snapshot_parallel(
                snapshot, is_keyframe, delta_enabled, bytes_written, error))
            return false;
        if (delta_enabled)
            ++delta_frame_counter;
        const auto t1 = std::chrono::steady_clock::now();
        record_snapshot_write_stats(
            bytes_written,
            std::chrono::duration<double>(t1 - t0).count());
        return true;
    }
#endif

    for (const auto& entry : snapshot.fields)
    {
        if (!write_field(entry, snapshot.step_dir, use_csv, use_zfp, is_keyframe, delta_enabled, bytes_written, error))
        {
            return false;
        }
    }

    if (!write_snapshot_sidecars(snapshot, bytes_written, error))
    {
        return false;
    }

    const auto   t1      = std::chrono::steady_clock::now();
    const double elapsed = std::chrono::duration<double>(t1 - t0).count();
    record_snapshot_write_stats(bytes_written, elapsed);

    return true;
}
catch (const std::exception& exception)
{
    assign_output_error(error, exception.what());
    return false;
}
catch (...)
{
    assign_output_error(error, "Unknown output failure");
    return false;
}
