#pragma once

/**
 * @file output_writer.hpp
 * @brief Asynchronous output writer with double-buffering.
 *
 * The AsyncOutputWriter runs a dedicated background thread that serializes
 * ExportSnapshot data to disk while the simulation thread continues computing.
 * It uses a single-slot producer-consumer pattern with condition variables:
 *
 *   - Simulation thread calls submit(snapshot) to hand off data.
 *   - If the writer is idle, submit returns immediately.
 *   - If the writer is busy with a previous snapshot, submit blocks until done.
 *   - flush() drains the queue and joins the thread at simulation end.
 *
 * This design ensures at most one pending snapshot in memory (double-buffer).
 * On fast NVMe storage, the writer finishes before the next export, so the
 * simulation thread never blocks. On slow storage, backpressure is applied
 * naturally.
 *
 * Set OutputConfig::async_io = false to bypass the thread entirely and write
 * synchronously (useful for debugging or memory-constrained machines).
 */

#include "core/field/field_snapshot.hpp"
#include "core/output/output_config.hpp"

#include <atomic>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

class AsyncOutputWriter
{
public:
    explicit AsyncOutputWriter(const OutputConfig& config);
    ~AsyncOutputWriter();

    // Non-copyable, non-movable (owns a thread)
    AsyncOutputWriter(const AsyncOutputWriter&) = delete;
    AsyncOutputWriter& operator=(const AsyncOutputWriter&) = delete;

    /**
     * @brief Submit a snapshot for background writing.
     *
     * If async_io is enabled, hands off the snapshot to the writer thread.
     * If the writer is busy, blocks until the previous write completes.
     * If async_io is disabled, writes synchronously and returns.
     *
     * @return false if the writer thread has encountered an error.
     */
    bool submit(ExportSnapshot snapshot);

    /**
     * @brief Block until all pending writes complete. Call at simulation end.
     * @return false if the writer encountered an error during any write.
     */
    bool flush();

    /// Check if the writer thread reported an error.
    bool has_error() const { return error_flag_.load(std::memory_order_acquire); }

    /// Error message from the writer thread (empty if no error).
    std::string error_message() const;

    /// Total bytes written across all snapshots.
    std::size_t total_bytes_written() const { return total_bytes_.load(std::memory_order_relaxed); }

    /// Total wall-clock time spent writing (seconds).
    double total_write_time_s() const { return total_time_s_.load(std::memory_order_relaxed); }

    /// Number of snapshots successfully written.
    std::size_t snapshots_written() const { return snapshots_count_.load(std::memory_order_relaxed); }

private:
    void writer_loop();
    bool write_snapshot(const ExportSnapshot& snapshot, std::string& error);

    OutputConfig config_;

    // Threading state
    std::thread writer_thread_;
    std::mutex mutex_;
    std::condition_variable cv_producer_;  // sim thread waits when buffer full
    std::condition_variable cv_consumer_;  // writer thread waits when no work

    std::unique_ptr<ExportSnapshot> pending_;
    bool shutdown_ = false;

    // Error state
    std::atomic<bool> error_flag_{false};
    std::string error_msg_;

    // Statistics
    std::atomic<std::size_t> total_bytes_{0};
    std::atomic<double> total_time_s_{0.0};
    std::atomic<std::size_t> snapshots_count_{0};

    // Delta encoding state: stores previous snapshots' field data keyed by
    // field name. previous_fields_ is the most recent frame; previous_fields_2_
    // is the frame before that (used for predictive delta).
    std::unordered_map<std::string, std::vector<float>> previous_fields_;
    std::unordered_map<std::string, std::vector<float>> previous_fields_2_;
    int delta_frame_counter_ = 0;

    // Persistent scratch buffer for delta/filter computation. Avoids
    // allocating a new vector on every non-keyframe export. Resized once
    // on first use and reused across all subsequent frames.
    std::vector<float> scratch_buffer_;
};
