#pragma once

/**
 * @file shm_writer.hpp
 * @brief Simulation-side shared memory writer for live visualization.
 *
 * Creates a POSIX shared memory region and writes normalized field data
 * that the Vulkan viewer can read directly. The writer handles:
 *   - SHM region creation and cleanup
 *   - Field3D → viewer-layout transposition
 *   - Robust percentile normalization (matching ExportDataset behavior)
 *   - Sequence number bumping for reader synchronization
 */

#include "core/field3d.hpp"
#include "core/output/shm_transport.hpp"

#include <string>
#include <vector>

class ShmWriter
{
public:
    ShmWriter() = default;
    ~ShmWriter();

    // Non-copyable
    ShmWriter(const ShmWriter&) = delete;
    ShmWriter& operator=(const ShmWriter&) = delete;

    /**
     * @brief Create the shared memory region.
     * @param nx Volume width (radial, after transpose to viewer layout).
     * @param ny Volume height (azimuthal).
     * @param nz Volume depth (vertical).
     * @param field_names Names of the fields to stream.
     * @param shm_name POSIX SHM name (default: /tmv_live).
     * @return true on success.
     */
    bool open(int nx, int ny, int nz,
              const std::vector<std::string>& field_names,
              const char* shm_name = tmv_shm::kDefaultShmName);

    /**
     * @brief Write one field's data from a Field3D.
     *
     * Transposes from Field3D layout [NR][NTH][NZ] to viewer layout [Z][TH][X],
     * then normalizes to [0,1] using robust percentile clipping.
     *
     * @param field_index Index in the field list (0-based).
     * @param field Source Field3D data.
     */
    void write_field(int field_index, const Field3D& field);

    /**
     * @brief Commit the current frame: bump sequence number and set simulation time.
     *
     * Call this after all write_field() calls for one timestep are done.
     */
    void commit(double simulation_time);

    /// True if the SHM region is open and mapped.
    bool is_open() const { return shm_base_ != nullptr; }

    /// Close and unlink the shared memory region.
    void close();

private:
    void* shm_base_ = nullptr;
    std::size_t shm_size_ = 0;
    int shm_fd_ = -1;
    std::string shm_name_;
    int nx_ = 0;
    int ny_ = 0;
    int nz_ = 0;
};
