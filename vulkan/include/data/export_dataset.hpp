/**
 * @file export_dataset.hpp
 * @brief Dataset scanner/loader for tornado export volumes.
 *
 * Defines normalized frame payloads and utilities for discovering compatible
 * `step_*` directories containing per-y NPY slices. Consumers use this
 * interface to load frame-indexed scalar volumes with consistent dimensions.
 */

#pragma once

#include "data/grid_geometry.hpp"

#include <filesystem>
#include <cstddef>
#include <string>
#include <vector>

namespace oglcpp 
{

/**
 * @brief One normalized 3D scalar frame plus source-value statistics.
 */
struct VolumeFrame 
{
    int nx = 0;  // x dimension
    int ny = 0;  // y slices
    int nz = 0;  // vertical dimension
    std::vector<float> normalized;

    float raw_min = 0.0f; // Minimum value in the raw data
    float raw_max = 0.0f; // Maximum value in the raw data
    float norm_low = 0.0f; // Minimum value in the normalized data
    float norm_high = 0.0f; // Maximum value in the normalized data

    std::size_t nan_count = 0; // Number of NaN values in the raw data
    std::size_t inf_count = 0; // Number of infinite values in the raw data
    std::size_t sanitized_nonfinite_count = 0;
};

/**
 * @brief Loads scalar fields from exported `step_*` directory trees.
 */
class ExportDataset 
{
public:
    /**
     * @brief Construct a dataset reader for a root directory and field name.
     * @param root_dir Root directory containing `step_*` folders.
     * @param field_name Scalar field token to load.
     */
    ExportDataset(std::filesystem::path root_dir, std::string field_name);

    /**
     * @brief Discover valid step directories and establish reference dimensions.
     * @param error Output message on failure.
     * @return `true` on success.
     */
    bool scan(std::string& error);

    /**
     * @brief Load and normalize one frame by index.
     * @param frame_idx Zero-based frame index.
     * @param out Destination frame payload.
     * @param error Output message on failure.
     * @return `true` on success.
     */
    bool load_frame(std::size_t frame_idx, VolumeFrame& out, std::string& error) const;

    /*==============================
    Public members for the ExportDataset class
    ==============================*/
    [[nodiscard]] std::size_t frame_count() const { return step_dirs_.size(); }
    [[nodiscard]] int nx() const { return nx_; }
    [[nodiscard]] int ny() const { return ny_; }
    [[nodiscard]] int nz() const { return nz_; }
    [[nodiscard]] const PhysicalGridSpacing& grid_spacing() const { return grid_spacing_; }
    [[nodiscard]] const std::string& field_name() const { return field_name_; }
    /** @brief Whether this dataset's field is signed (bipolar) about zero. */
    [[nodiscard]] bool is_bipolar() const;
    [[nodiscard]] const std::filesystem::path& root_dir() const { return root_dir_; }

    /**
     * @brief Pre-scan all frames to compute the global absolute maximum.
     *
     * Cross-frame normalization ensures that a 0.06 m/s^2 buoyancy at
     * t=10s does not render as brightly as 5.0 m/s^2 at t=300s. Without
     * this, per-frame normalization makes weak early signals fill the
     * volume because they are large relative to that frame's own maximum.
     *
     * Call after scan(), before any load_frame() calls. The returned
     * magnitude is used as the normalization denominator for all frames.
     * Returns 0 if the field is unipolar (always >= 0).
     */
    float compute_global_magnitude() const;

    /**
     * @brief Set the cross-frame normalization magnitude for bipolar fields.
     *
     * When set to a positive value, all subsequent load_frame() calls
     * normalize bipolar data against this fixed magnitude instead of
     * the per-frame maximum. Set to 0 to revert to per-frame behavior.
     */
    void set_global_magnitude(float mag) { global_magnitude_ = mag; }

    /**
     * @brief Normalize one raw float32 volume using the disk/live display contract.
     *
     * Public so raw SHM snapshots use exactly the same signed midpoint,
     * percentile, and non-finite behavior as disk playback.
     */
    static void normalize_volume(const std::vector<float>& src,
                                 std::vector<float>& dst,
                                 float& raw_min,
                                 float& raw_max,
                                 float& norm_low,
                                 float& norm_high,
                                 std::size_t& nan_count,
                                 std::size_t& inf_count,
                                 std::size_t& sanitized_nonfinite_count,
                                 float global_magnitude = 0.0f,
                                 bool name_bipolar = false);

private:
    std::filesystem::path root_dir_;
    std::string field_name_;

    std::vector<std::filesystem::path> step_dirs_;
    int nx_ = 0;
    int ny_ = 0;
    int nz_ = 0;
    int source_nx_ = 0; ///< On-disk x extent, including storage-only closure cells
    int source_ny_ = 0; ///< On-disk y extent, including storage-only closure cells
    int source_nz_ = 0; ///< On-disk z extent, including storage-only closure cells
    bool is_3d_volume_ = false; ///< True when steps contain 3D volumes instead of 2D slices
    PhysicalGridSpacing grid_spacing_{}; ///< Validated physical spacing shared by every frame
    PhysicalIndexRange valid_range_{}; ///< Physical cells within the on-disk payload
    float global_magnitude_ = 0.0f; ///< Cross-frame magnitude for bipolar normalization

    /** @brief Clamp scalar into the closed unit interval `[0, 1]`. */
    static float clamp01(float v);
    
};

} // namespace oglcpp
