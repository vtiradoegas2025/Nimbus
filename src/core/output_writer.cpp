/**
 * @file output_writer.cpp
 * @brief Async output writer implementation.
 */

#include "core/output_writer.hpp"

#include "core/npy_writer.hpp"

#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>

#ifdef HAVE_ZFP
#include <zfp.h>
#endif

AsyncOutputWriter::AsyncOutputWriter(const OutputConfig& config)
    : config_(config)
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

bool AsyncOutputWriter::submit(ExportSnapshot snapshot)
{
    if (error_flag_.load(std::memory_order_acquire))
    {
        return false;
    }

    if (!config_.async_io)
    {
        // Synchronous path — write directly on the calling thread
        std::string error;
        if (!write_snapshot(snapshot, error))
        {
            error_flag_.store(true, std::memory_order_release);
            error_msg_ = error;
            return false;
        }
        return true;
    }

    // Async path — hand off to writer thread
    {
        std::unique_lock<std::mutex> lock(mutex_);
        // Wait if the previous snapshot hasn't been consumed yet
        cv_producer_.wait(lock, [this] { return !pending_ || shutdown_; });

        if (shutdown_)
        {
            return !error_flag_.load(std::memory_order_acquire);
        }

        pending_ = std::make_unique<ExportSnapshot>(std::move(snapshot));
    }
    cv_consumer_.notify_one();
    return true;
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
                snap = std::move(pending_);
                // Signal producer that the slot is free
                cv_producer_.notify_one();
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
                std::lock_guard<std::mutex> lock(mutex_);
                error_flag_.store(true, std::memory_order_release);
                error_msg_ = error;
                // Signal producer in case it's waiting
                cv_producer_.notify_one();
                break;
            }
        }
    }
}

#ifdef HAVE_ZFP
/**
 * @brief Write a 3D float field compressed with ZFP to a .zfp3d file.
 *
 * File format (v2, backward-compatible with v1):
 *   Bytes 0-3:   magic "ZFP3" (4 bytes)
 *   Bytes 4-5:   version (uint16_t: 1 = keyframe only, 2 = delta support)
 *   Bytes 6-7:   mode (uint16_t: 0=accuracy, 1=precision, 2=rate)
 *   Bytes 8-11:  dim0 (int32_t)
 *   Bytes 12-15: dim1 (int32_t)
 *   Bytes 16-19: dim2 (int32_t)
 *   Bytes 20-27: tolerance/precision/rate (double)
 *   Bytes 28-35: compressed_size (uint64_t)
 *   --- v2 additions ---
 *   Bytes 36:    flags (uint8_t: bit 0 = is_delta_frame)
 *   Bytes 37-39: reserved (3 bytes, zero)
 *   Bytes 40+:   compressed payload
 *
 * For v1 files (no delta encoding), the payload starts at byte 36.
 *
 * @param is_delta If true, the data represents (current - previous) rather
 *                 than an absolute field. The reader must add the previous
 *                 keyframe to reconstruct the original.
 */
static bool write_zfp_3d(const float* data, int dim0, int dim1, int dim2,
                          const std::string& mode_str, double tolerance,
                          int rate_bps, bool is_delta,
                          const std::string& path,
                          std::size_t& bytes_out, std::string& error)
{
    const std::size_t total = static_cast<std::size_t>(dim0) *
                              static_cast<std::size_t>(dim1) *
                              static_cast<std::size_t>(dim2);
    if (total == 0)
    {
        error = "ZFP: zero-size field";
        return false;
    }

    // Set up ZFP field and stream
    zfp_type type = zfp_type_float;
    zfp_field* field = zfp_field_3d(const_cast<float*>(data),
                                     type, dim2, dim1, dim0);
    if (!field)
    {
        error = "ZFP: failed to create field";
        return false;
    }

    zfp_stream* zfp = zfp_stream_open(nullptr);
    if (!zfp)
    {
        zfp_field_free(field);
        error = "ZFP: failed to open stream";
        return false;
    }

    // Configure compression mode
    uint16_t mode_id = 0;
    double mode_param = tolerance;
    if (mode_str == "accuracy")
    {
        zfp_stream_set_accuracy(zfp, tolerance);
        mode_id = 0;
        mode_param = tolerance;
    }
    else if (mode_str == "precision")
    {
        uint precision = static_cast<uint>(std::max(1, static_cast<int>(tolerance)));
        zfp_stream_set_precision(zfp, precision);
        mode_id = 1;
        mode_param = static_cast<double>(precision);
    }
    else if (mode_str == "rate")
    {
        zfp_stream_set_rate(zfp, static_cast<double>(rate_bps), type, 3, 0);
        mode_id = 2;
        mode_param = static_cast<double>(rate_bps);
    }
    else
    {
        // Default to accuracy mode
        zfp_stream_set_accuracy(zfp, tolerance);
        mode_id = 0;
        mode_param = tolerance;
    }

    // Allocate buffer for compressed data
    std::size_t bufsize = zfp_stream_maximum_size(zfp, field);
    std::vector<unsigned char> buffer(bufsize);

    bitstream* stream = stream_open(buffer.data(), bufsize);
    if (!stream)
    {
        zfp_stream_close(zfp);
        zfp_field_free(field);
        error = "ZFP: failed to open bitstream";
        return false;
    }
    zfp_stream_set_bit_stream(zfp, stream);
    zfp_stream_rewind(zfp);

    // Compress
    std::size_t compressed_size = zfp_compress(zfp, field);
    if (compressed_size == 0)
    {
        stream_close(stream);
        zfp_stream_close(zfp);
        zfp_field_free(field);
        error = "ZFP: compression failed for " + path;
        return false;
    }

    // Write file
    std::ofstream out(path, std::ios::binary);
    if (!out)
    {
        stream_close(stream);
        zfp_stream_close(zfp);
        zfp_field_free(field);
        error = "ZFP: failed to open " + path;
        return false;
    }

    // Header (v2 with delta flag)
    const char magic[4] = {'Z', 'F', 'P', '3'};
    const uint16_t version = 2;
    const int32_t d0 = dim0, d1 = dim1, d2 = dim2;
    const uint64_t csize = compressed_size;
    const uint8_t flags = is_delta ? 0x01 : 0x00;
    const uint8_t reserved[3] = {0, 0, 0};

    out.write(magic, 4);
    out.write(reinterpret_cast<const char*>(&version), sizeof(version));
    out.write(reinterpret_cast<const char*>(&mode_id), sizeof(mode_id));
    out.write(reinterpret_cast<const char*>(&d0), sizeof(d0));
    out.write(reinterpret_cast<const char*>(&d1), sizeof(d1));
    out.write(reinterpret_cast<const char*>(&d2), sizeof(d2));
    out.write(reinterpret_cast<const char*>(&mode_param), sizeof(mode_param));
    out.write(reinterpret_cast<const char*>(&csize), sizeof(csize));
    out.write(reinterpret_cast<const char*>(&flags), sizeof(flags));
    out.write(reinterpret_cast<const char*>(reserved), sizeof(reserved));

    // Payload
    out.write(reinterpret_cast<const char*>(buffer.data()),
              static_cast<std::streamsize>(compressed_size));

    stream_close(stream);
    zfp_stream_close(zfp);
    zfp_field_free(field);

    if (!out.good())
    {
        error = "ZFP: write error for " + path;
        return false;
    }

    // Header size: 4 + 2 + 2 + 4 + 4 + 4 + 8 + 8 + 1 + 3 = 40 bytes
    bytes_out = 40 + compressed_size;
    return true;
}
#endif // HAVE_ZFP

bool AsyncOutputWriter::write_snapshot(const ExportSnapshot& snapshot,
                                       std::string& error)
{
    const auto t0 = std::chrono::steady_clock::now();

    // Ensure step directory exists
    std::error_code ec;
    std::filesystem::create_directories(snapshot.step_dir, ec);
    if (ec)
    {
        error = "Failed to create directory " + snapshot.step_dir.string() +
                ": " + ec.message();
        return false;
    }

    const bool use_zfp = (config_.format == OutputFormat::zfp);
    const bool use_csv = (config_.format == OutputFormat::csv);
    std::size_t bytes_written = 0;

    // Delta encoding state: determine if this frame is a keyframe or delta.
    const bool delta_enabled = use_zfp && config_.zfp_keyframe_interval > 0;
    const bool is_keyframe = !delta_enabled ||
                             (delta_frame_counter_ % config_.zfp_keyframe_interval == 0);
    if (delta_enabled)
    {
        ++delta_frame_counter_;
    }

    // Write each field
    for (const auto& entry : snapshot.fields)
    {
        bool ok = false;

        if (use_csv && entry.is_3d && entry.dim1 > 0 && entry.dim2 > 0)
        {
            const std::string path =
                (snapshot.step_dir / (entry.name + ".csv")).string();
            ok = csv::write_3d(entry.data.data(),
                               entry.dim0, entry.dim1, entry.dim2, path);
            if (ok)
            {
                bytes_written += entry.data.size() * sizeof(float);
            }
        }
        else
#ifdef HAVE_ZFP
        if (use_zfp && entry.is_3d && entry.dim1 > 0 && entry.dim2 > 0)
        {
            const std::string path =
                (snapshot.step_dir / (entry.name + ".zfp3d")).string();
            std::size_t field_bytes = 0;

            if (is_keyframe || previous_fields_.find(entry.name) == previous_fields_.end())
            {
                // Keyframe: write the full field
                ok = write_zfp_3d(entry.data.data(),
                                  entry.dim0, entry.dim1, entry.dim2,
                                  config_.zfp_mode, config_.zfp_tolerance,
                                  config_.zfp_rate_bps, false,
                                  path, field_bytes, error);
            }
            else
            {
                // Delta frame: compute delta = current - previous, compress that.
                // Deltas between consecutive timesteps are small → high compression.
                const auto& prev = previous_fields_[entry.name];
                const std::size_t n = entry.data.size();
                std::vector<float> delta(n);
                for (std::size_t idx = 0; idx < n; ++idx)
                {
                    delta[idx] = entry.data[idx] - prev[idx];
                }
                ok = write_zfp_3d(delta.data(),
                                  entry.dim0, entry.dim1, entry.dim2,
                                  config_.zfp_mode, config_.zfp_tolerance,
                                  config_.zfp_rate_bps, true,
                                  path, field_bytes, error);
            }

            // Store current data as previous for next frame's delta computation
            if (ok && delta_enabled)
            {
                previous_fields_[entry.name] = entry.data;
            }

            if (ok)
            {
                bytes_written += field_bytes;
            }
        }
        else
#endif
        {
            const std::string path =
                (snapshot.step_dir / (entry.name + ".npy")).string();

            if (entry.is_3d && entry.dim1 > 0 && entry.dim2 > 0)
            {
                ok = npy::write_3d(entry.data.data(),
                                   entry.dim0, entry.dim1, entry.dim2, path);
            }
            else
            {
                ok = npy::write_2d(entry.data.data(),
                                   entry.dim0,
                                   (entry.dim1 > 0) ? entry.dim1 : entry.dim2,
                                   path);
            }

            if (ok)
            {
                bytes_written += entry.data.size() * sizeof(float);
            }
        }

        if (!ok)
        {
            if (error.empty())
            {
                error = "Failed to write field " + entry.name;
            }
            return false;
        }
    }

    // Write manifest JSON
    if (!snapshot.manifest_json.empty())
    {
        const auto manifest_path = snapshot.step_dir / "manifest.json";
        std::ofstream out(manifest_path);
        if (out)
        {
            out << snapshot.manifest_json;
            bytes_written += snapshot.manifest_json.size();
        }
    }

    // Write validation report JSON
    if (!snapshot.validation_json.empty() &&
        !snapshot.validation_path.empty())
    {
        std::filesystem::create_directories(snapshot.validation_path.parent_path(), ec);
        std::ofstream out(snapshot.validation_path);
        if (out)
        {
            out << snapshot.validation_json;
            bytes_written += snapshot.validation_json.size();
        }
    }

    const auto t1 = std::chrono::steady_clock::now();
    const double elapsed = std::chrono::duration<double>(t1 - t0).count();

    total_bytes_.fetch_add(bytes_written, std::memory_order_relaxed);
    // Atomic double add via compare-exchange
    double old_time = total_time_s_.load(std::memory_order_relaxed);
    while (!total_time_s_.compare_exchange_weak(
        old_time, old_time + elapsed, std::memory_order_relaxed))
    {
    }
    snapshots_count_.fetch_add(1, std::memory_order_relaxed);

    return true;
}
