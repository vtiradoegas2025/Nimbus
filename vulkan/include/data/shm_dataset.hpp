#pragma once

/**
 * @file shm_dataset.hpp
 * @brief Viewer-side shared memory dataset reader.
 *
 * Implements the same scan/load_frame interface as ExportDataset but reads
 * from a POSIX shared memory region written by the simulation's ShmWriter.
 * This enables zero-disk-I/O live visualization.
 *
 * Usage in VolumeBackend:
 *   ShmDataset shm(field_name);
 *   if (shm.scan(error)) { ... shm.load_frame(0, frame, error); ... }
 */

#include "export_dataset.hpp" // for VolumeFrame

#include <cstdint>
#include <string>

namespace oglcpp
{

class ShmDataset
{
public:
    /**
     * @brief Construct a SHM dataset reader for a specific field.
     * @param field_name Field to read (must match one of the writer's field names).
     * @param shm_name POSIX SHM name (default: /tmv_live).
     */
    explicit ShmDataset(std::string field_name,
                        const char* shm_name = "/tmv_live");
    ~ShmDataset();

    // Non-copyable, movable
    ShmDataset(const ShmDataset&) = delete;
    ShmDataset& operator=(const ShmDataset&) = delete;
    ShmDataset(ShmDataset&& other) noexcept;
    ShmDataset& operator=(ShmDataset&& other) noexcept;

    /**
     * @brief Attach to the shared memory region and validate the header.
     * @param error Output message on failure.
     * @return true if the SHM region is valid and contains the requested field.
     */
    bool scan(std::string& error);

    /**
     * @brief Load the current frame from shared memory.
     *
     * Unlike ExportDataset, there is only ever one frame (the latest).
     * frame_idx is ignored — always returns the most recent data.
     *
     * @param frame_idx Ignored (always loads latest).
     * @param out Destination frame payload.
     * @param error Output message on failure.
     * @return true on success.
     */
    bool load_frame(std::size_t frame_idx, VolumeFrame& out, std::string& error) const;

    /// Returns true if new data is available since last load_frame().
    bool has_new_frame() const;

    [[nodiscard]] std::size_t frame_count() const { return (shm_base_ != nullptr) ? 1 : 0; }
    [[nodiscard]] int nx() const { return nx_; }
    [[nodiscard]] int ny() const { return ny_; }
    [[nodiscard]] int nz() const { return nz_; }
    [[nodiscard]] const std::string& field_name() const { return field_name_; }

private:
    std::string field_name_;
    std::string shm_name_;

    const void* shm_base_ = nullptr;
    std::size_t shm_size_ = 0;
    int shm_fd_ = -1;

    int nx_ = 0;
    int ny_ = 0;
    int nz_ = 0;
    int field_index_ = -1; ///< Index of our field in the SHM name table

    mutable uint64_t last_sequence_ = 0;
};

} // namespace oglcpp
