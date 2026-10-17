#pragma once

/**
 * @file shm_dataset.hpp
 * @brief Viewer-side shared memory dataset reader.
 *
 * Implements the same scan/load_frame interface as ExportDataset but reads
 * from a native shared memory region written by the simulation's ShmWriter.
 * This enables zero-disk-I/O live visualization.
 *
 * Usage in VolumeBackend:
 *   ShmDataset shm(field_name);
 *   if (shm.scan(error)) { ... shm.load_frame(0, frame, error); ... }
 */

#include "export_dataset.hpp" // for VolumeFrame
#include "core/output/shm_transport.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace oglcpp
{

class ShmDataset
{
public:
    /**
     * @brief Construct a SHM dataset reader for a specific field.
     * @param field_name Field to read (must match one of the writer's field names).
     * @param shm_name Logical SHM name (default: /tmv_live).
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
    [[nodiscard]] const PhysicalGridSpacing& grid_spacing() const { return grid_spacing_; }
    [[nodiscard]] const std::string& field_name() const { return field_name_; }
    [[nodiscard]] bool is_bipolar() const
    {
        return field_metadata_.polarity == tmv_shm::ShmFieldPolarity::Bipolar;
    }
    [[nodiscard]] const char* units() const { return field_metadata_.units; }
    [[nodiscard]] uint64_t last_sequence() const { return last_sequence_; }
    [[nodiscard]] double last_simulation_time() const { return last_simulation_time_; }
    [[nodiscard]] const std::string& publication_generation() const
    {
        return publication_generation_;
    }
    [[nodiscard]] const tmv_shm::ShmHorizontalFrame& horizontal_frame() const
    {
        return last_horizontal_frame_;
    }
    [[nodiscard]] bool same_snapshot_identity(const ShmDataset& other) const
    {
        return !publication_generation_.empty() &&
               publication_generation_ == other.publication_generation_ &&
               last_sequence_ != 0 && (last_sequence_ & 1U) == 0U &&
               last_sequence_ == other.last_sequence_ &&
               last_simulation_time_ == other.last_simulation_time_;
    }

private:
    void release_mapping() noexcept;

    std::string field_name_;
    std::string shm_name_;

    const void* shm_base_ = nullptr;
    std::size_t shm_size_ = 0;
    int shm_fd_ = -1;
#if defined(_WIN32)
    void* shm_handle_ = nullptr;
#endif

    int nx_ = 0;
    int ny_ = 0;
    int nz_ = 0;
    int source_nx_ = 0;
    int source_ny_ = 0;
    int source_nz_ = 0;
    tmv_shm::ShmPhysicalDomain physical_domain_{};
    PhysicalGridSpacing grid_spacing_{};
    int field_index_ = -1; ///< Index of our field in the SHM name table
    tmv_shm::ShmFieldMetadata field_metadata_{};
    std::string publication_generation_;

    mutable std::vector<float> source_snapshot_;
    mutable std::vector<float> raw_snapshot_;
    mutable uint64_t last_sequence_ = 0;
    mutable double last_simulation_time_ = 0.0;
    mutable tmv_shm::ShmHorizontalFrame last_horizontal_frame_{};
};

} // namespace oglcpp
