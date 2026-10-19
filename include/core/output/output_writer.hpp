#pragma once

/**
 * @file output_writer.hpp
 * @brief Asynchronous output writer with one bounded in-flight snapshot.
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
 * The producer reserves the writer before assembling the next snapshot. This
 * permits solver steps to overlap an earlier write without allowing one frame
 * in the writer and another complete frame under construction. On slow
 * storage, backpressure is applied before the large output allocations.
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

/** Write one v4 reversible, independently decodable keyframe container. */
bool write_zfp_3d_reversible(const float* data,
                             int dim0,
                             int dim1,
                             int dim2,
                             const std::string& path,
                             std::size_t scratch_limit_bytes,
                             std::size_t& bytes_out,
                             std::string& error);

#ifdef HAVE_ZFP
struct ZfpWriterWorkspace;
struct ZfpParallelWorkerState;
#endif

class AsyncOutputWriter
{
public:
    explicit AsyncOutputWriter(const OutputConfig& config);
    ~AsyncOutputWriter();

    // Non-copyable, non-movable (owns a thread)
    AsyncOutputWriter(const AsyncOutputWriter&)            = delete;
    AsyncOutputWriter& operator=(const AsyncOutputWriter&) = delete;

    /**
     * @brief Submit a snapshot for background writing.
     *
     * If async_io is enabled, hands off the snapshot to the writer thread.
     * If the writer is busy, blocks until the previous write completes.
     * If async_io is disabled, writes synchronously and returns. Borrowed field
     * storage must remain unchanged until this call returns and is rejected
     * when async_io is enabled.
     *
     * @return false if the writer thread has encountered an error.
     */
    bool submit(ExportSnapshot snapshot);

    /**
     * @brief Block until no snapshot is queued or being written.
     *
     * Call this before allocating/assembling an asynchronous frame. It is a
     * no-op for synchronous output. This keeps the memory bound at one complete
     * snapshot rather than blocking only after a second frame already exists.
     */
    bool wait_until_ready();

    /**
     * @brief Block until all pending writes complete. Call at simulation end.
     * @return false if the writer encountered an error during any write.
     */
    bool flush();

    /// Check if the writer thread reported an error.
    bool has_error() const
    {
        return error_flag_.load(std::memory_order_acquire);
    }

    /// Error message from the writer thread (empty if no error).
    std::string error_message() const;

    /// Total bytes written across all snapshots.
    std::size_t total_bytes_written() const
    {
        return total_bytes_.load(std::memory_order_relaxed);
    }

    /// Total wall-clock time spent writing (seconds).
    double total_write_time_s() const
    {
        return total_time_s_.load(std::memory_order_relaxed);
    }

    double total_backpressure_wait_time_s() const
    {
        return total_backpressure_wait_s_.load(std::memory_order_relaxed);
    }

    double maximum_snapshot_write_time_s() const
    {
        return maximum_snapshot_write_s_.load(std::memory_order_relaxed);
    }

    std::size_t maximum_snapshot_bytes_written() const
    {
        return maximum_snapshot_bytes_.load(std::memory_order_relaxed);
    }

    /// Number of snapshots successfully written.
    std::size_t snapshots_written() const
    {
        return snapshots_count_.load(std::memory_order_relaxed);
    }

    /// Persistent float staging reserved solely by the controlled legacy ZFP
    /// brick gather/scatter comparator. Production strided views keep this at
    /// exactly zero after arbitrarily large/full-brick writes.
    std::size_t zfp_brick_staging_capacity_bytes() const noexcept
    {
#ifdef HAVE_ZFP
        return (zfp_brick_input_.capacity() + zfp_brick_decoded_.capacity()) * sizeof(float);
#else
        return 0;
#endif
    }

    std::size_t zfp_full_field_scratch_capacity_bytes() const noexcept
    {
#ifdef HAVE_ZFP
        return scratch_buffer_.capacity() * sizeof(float);
#else
        return 0;
#endif
    }

#ifdef HAVE_ZFP
    void enable_reversible_radar_v4_for_testing() noexcept
    {
        reversible_radar_v4_test_only_ = true;
    }
#endif

    /// Aggregate persistent codec-workspace creations. Serial remains one;
    /// field-parallel creates at most one per active isolated worker.
    std::size_t zfp_codec_workspace_initializations() const noexcept;

    /// Number of times a ZFP bitstream had to be bound to a new/grown backing
    /// buffer. This reaches a high-water mark instead of growing per brick.
    std::size_t zfp_bitstream_bindings() const noexcept;

    /// Aggregate high-water byte capacity exposed through active bitstreams.
    std::size_t zfp_bitstream_capacity_bytes() const noexcept;

    /// Linked libzfp version used by this writer build.
    std::string zfp_codec_version() const;

    /// Active per-workspace libzfp policy (`0` remains serial under either
    /// field scheduling policy).
    int zfp_execution_policy_id() const noexcept;

    /// Benchmark-instrumented time inside ZFP field/view/control preparation.
    double zfp_control_time_s() const noexcept;

    /// Benchmark-instrumented time inside libzfp compress/decompress calls.
    double zfp_codec_time_s() const noexcept;

    /// Benchmark-instrumented time inside container filesystem operations.
    double zfp_container_io_time_s() const noexcept;

    const std::string& zfp_field_execution_policy() const noexcept
    {
        return config_.zfp_field_execution;
    }
    int zfp_resolved_field_workers() const noexcept
    {
        return config_.zfp_resolved_field_workers;
    }
    std::size_t zfp_parallel_scratch_high_water_bytes() const noexcept
    {
        return zfp_parallel_scratch_high_water_bytes_.load(
            std::memory_order_relaxed);
    }
    std::size_t zfp_parallel_tasks_completed() const noexcept
    {
        return zfp_parallel_tasks_completed_.load(std::memory_order_relaxed);
    }
    double zfp_parallel_wall_time_s() const noexcept
    {
        return zfp_parallel_wall_time_s_.load(std::memory_order_relaxed);
    }

private:
    void writer_loop();
    bool write_snapshot(const ExportSnapshot& snapshot, std::string& error);
    bool ensure_step_directory(const std::filesystem::path& step_dir, std::string& error) const;
    bool write_field(const FieldSnapshotEntry&    entry,
                     const std::filesystem::path& step_dir,
                     bool                         use_csv,
                     bool                         use_zfp,
                     bool                         is_keyframe,
                     bool                         delta_enabled,
                     std::size_t&                 bytes_written,
                     std::string&                 error);
    bool write_csv_field(const FieldSnapshotEntry&    entry,
                         const std::filesystem::path& step_dir,
                         std::size_t&                 bytes_written) const;
    bool write_npy_field(const FieldSnapshotEntry&    entry,
                         const std::filesystem::path& step_dir,
                         std::size_t&                 bytes_written) const;
    bool write_netcdf_snapshot(const ExportSnapshot& snapshot, std::size_t& bytes_written, std::string& error) const;
    bool write_snapshot_sidecars(const ExportSnapshot& snapshot, std::size_t& bytes_written, std::string& error) const;
    void record_snapshot_write_stats(std::size_t bytes_written, double elapsed);

#ifdef HAVE_ZFP
    bool reversible_radar_v4_test_only_ = false;
    double zfp_tolerance_for_field(const std::string& field_name) const;
    std::string zfp_history_key(const std::string& field_name) const;
    bool   has_predictive_zfp_delta_reference(const FieldSnapshotEntry& entry) const;
    void   apply_zfp_prefilters(std::vector<float>& buffer, std::size_t count, bool apply_sparse, bool apply_f16) const;
    bool   write_zfp_keyframe(const FieldSnapshotEntry&   entry,
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
                              std::string&                error);
    bool   write_zfp_predictive_delta(const FieldSnapshotEntry&   entry,
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
                                      std::string&                error);
    bool   write_zfp_simple_delta(const FieldSnapshotEntry&   entry,
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
                                  std::string&                error);
    bool   update_zfp_delta_history(const FieldSnapshotEntry& entry,
                                    const float*              compressed_data,
                                    const float*              decoded_data,
                                    bool                      is_keyframe,
                                    double                    field_tolerance,
                                    std::string&              error);
    bool   write_zfp_field(const FieldSnapshotEntry&    entry,
                           const std::filesystem::path& step_dir,
                           bool                         is_keyframe,
                           bool                         delta_enabled,
                           std::size_t&                 bytes_written,
                           std::string&                 error);
    bool write_zfp_snapshot_parallel(const ExportSnapshot& snapshot,
                                     bool is_keyframe,
                                     bool delta_enabled,
                                     std::size_t& bytes_written,
                                     std::string& error);
#endif

    OutputConfig config_;

    // Threading state
    std::thread             writer_thread_;
    std::mutex              mutex_;
    std::condition_variable cv_producer_; // sim thread waits when buffer full
    std::condition_variable cv_consumer_; // writer thread waits when no work

    std::unique_ptr<ExportSnapshot> pending_;
    bool                            writing_  = false;
    bool                            shutdown_ = false;

    // Error state
    std::atomic<bool> error_flag_{false};
    std::string       error_msg_;

    // Statistics
    std::atomic<std::size_t> total_bytes_{0};
    std::atomic<double>      total_time_s_{0.0};
    std::atomic<std::size_t> snapshots_count_{0};
    std::atomic<double>      total_backpressure_wait_s_{0.0};
    std::atomic<double>      maximum_snapshot_write_s_{0.0};
    std::atomic<std::size_t> maximum_snapshot_bytes_{0};

    // Delta encoding state is keyed by data product and field. Root history
    // and nested-child history are independent decoded-reference streams.
    std::unordered_map<std::string, std::vector<float>> previous_fields_;
    std::unordered_map<std::string, std::vector<float>> previous_fields_2_;
    std::unordered_map<std::string, int>                 delta_frame_counters_;
    std::string                                          active_zfp_history_namespace_ = "history";

    // Persistent scratch buffer for delta/filter computation. Avoids
    // allocating a new vector on every non-keyframe export. Resized once
    // on first use and reused across all subsequent frames.
    std::vector<float> scratch_buffer_;

#ifdef HAVE_ZFP
    // Production exposes each brick as a strided view into the source and
    // decoded-history volumes. The float buffers remain solely for the
    // controlled legacy-gather comparator; they never allocate in production.
    std::vector<float>                  zfp_brick_input_;
    std::vector<float>                  zfp_brick_decoded_;
    std::vector<unsigned char>          zfp_compressed_buffer_;
    std::vector<uint64_t>               zfp_brick_offsets_;
    std::unique_ptr<ZfpWriterWorkspace> zfp_workspace_;
    std::vector<std::unique_ptr<ZfpParallelWorkerState>>
        zfp_parallel_workers_;
#endif
    std::atomic<std::size_t> zfp_parallel_scratch_high_water_bytes_{0};
    std::atomic<std::size_t> zfp_parallel_tasks_completed_{0};
    std::atomic<double> zfp_parallel_wall_time_s_{0.0};
};
