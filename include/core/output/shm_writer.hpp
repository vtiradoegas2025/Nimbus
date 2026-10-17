#pragma once

/**
 * @file shm_writer.hpp
 * @brief Simulation-side shared memory writer for live visualization.
 *
 * Creates a native shared memory region and writes raw float32 field data
 * that the Vulkan viewer can read directly. The writer handles:
 *   - SHM region creation and cleanup
 *   - Field3D → viewer-layout transposition
 *   - Explicit field metadata for reader-side display normalization
 *   - Atomic seqlock publication of data and simulation time
 */

#include "core/field/field3d.hpp"
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
     * @param shm_name Logical SHM name (default: /tmv_live).
     * @return true on success.
     */
    bool open(int nx, int ny, int nz,
              const std::vector<std::string>& field_names,
              const char* shm_name = tmv_shm::kDefaultShmName);

    /** @brief Create a region with explicit physical ownership inside storage. */
    bool open(int nx, int ny, int nz,
              const std::vector<std::string>& field_names,
              const tmv_shm::ShmPhysicalDomain& physical_domain,
              const char* shm_name = tmv_shm::kDefaultShmName);

    /** @brief Create a region with explicit ownership and physical spacing. */
    bool open(int nx, int ny, int nz,
              const std::vector<std::string>& field_names,
              const tmv_shm::ShmPhysicalDomain& physical_domain,
              const tmv_shm::ShmGridSpacing& grid_spacing_m,
              const char* shm_name = tmv_shm::kDefaultShmName);

    /** @brief Create a region with explicit moving-frame provenance. */
    bool open(int nx, int ny, int nz,
              const std::vector<std::string>& field_names,
              const tmv_shm::ShmPhysicalDomain& physical_domain,
              const tmv_shm::ShmGridSpacing& grid_spacing_m,
              const tmv_shm::ShmHorizontalFrame& horizontal_frame,
              const char* shm_name = tmv_shm::kDefaultShmName,
              const std::string& publication_generation =
                  std::string(32, '0'));

    /**
     * @brief Write one field's data from a Field3D.
     *
     * Transposes from Field3D layout [NR][NTH][NZ] to viewer layout [Z][TH][X],
     * preserving each float32 value (including sign and non-finite values).
     * The first valid write opens an odd seqlock transaction which commit()
     * closes after all fields and simulation_time have been written.
     *
     * @param field_index Index in the field list (0-based).
     * @param field Source Field3D data.
     * @return true only when the declared slot was completely updated.
     */
    [[nodiscard]] bool write_field(int field_index, const Field3D& field);

    /**
     * @brief Atomically publish the current field bundle and simulation time.
     *
     * Call this after all write_field() calls for one timestep are done.
     * Returns false and leaves the sequence odd when any declared slot was not
     * updated; a later attempt must rewrite every slot before publication.
     */
    [[nodiscard]] bool commit(double simulation_time);
    [[nodiscard]] bool commit(
        double simulation_time,
        const tmv_shm::ShmHorizontalFrame& horizontal_frame);

    /// True if the SHM region is open and mapped.
    bool is_open() const { return shm_base_ != nullptr; }
    std::uint64_t sequence_number() const;

    /// Close and unlink the shared memory region.
    void close();

private:
    void begin_frame();

    void* shm_base_ = nullptr;
    std::size_t shm_size_ = 0;
    int shm_fd_ = -1;
#if defined(_WIN32)
    void* shm_handle_ = nullptr;
#endif
    std::string shm_name_;
    int nx_ = 0;
    int ny_ = 0;
    int nz_ = 0;
    int field_count_ = 0;
    bool frame_open_ = false;
    std::vector<bool> fields_written_;
};
