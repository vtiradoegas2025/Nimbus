/**
 * @file npy_reader.hpp
 * @brief Minimal NPY reader for 2D float32 arrays used by Vulkan exports.
 *
 * Supports C-order NumPy `.npy` payloads with `float32` dtype and extracts
 * either full data or shape metadata. The parser is intentionally narrow to
 * keep loading predictable and reject unsupported binary layouts early.
 */

#pragma once

#include <filesystem>
#include <string>
#include <vector>

namespace oglcpp 
{

/**
 * @brief In-memory representation of a 2D float32 array payload.
 */
struct NpyArray2D 
{
    int rows = 0; 
    int cols = 0;
    std::vector<float> data; 
};

/**
 * @brief Shape metadata for a 2D NPY float32 array.
 */
struct NpyArray2DShape 
{
    int rows = 0;
    int cols = 0;
};

/**
 * @brief Load a full 2D float32 NPY array from disk.
 * @param path Source file path.
 * @param out Destination array.
 * @param error Output message on failure.
 * @return `true` on success.
 */
bool load_npy_float32_2d(const std::filesystem::path& path, NpyArray2D& out, std::string& error);

/**
 * @brief Read only the 2D float32 shape metadata from a NPY file.
 * @param path Source file path.
 * @param out Destination shape.
 * @param error Output message on failure.
 * @return `true` on success.
 */
bool load_npy_float32_2d_shape(const std::filesystem::path& path, NpyArray2DShape& out, std::string& error);

/**
 * @brief In-memory representation of a 3D float32 array payload.
 */
struct NpyArray3D
{
    int dim0 = 0;  ///< First dimension (NR for Field3D)
    int dim1 = 0;  ///< Second dimension (NTH for Field3D)
    int dim2 = 0;  ///< Third dimension (NZ for Field3D)
    std::vector<float> data;
};

/**
 * @brief Load a full 3D float32 NPY array from disk.
 * @param path Source file path.
 * @param out Destination array.
 * @param error Output message on failure.
 * @return `true` on success.
 */
bool load_npy_float32_3d(const std::filesystem::path& path, NpyArray3D& out, std::string& error);

/**
 * @brief Check if a file is a 3D NPY volume (vs 2D slice).
 * @param path File path.
 * @return Number of dimensions (2 or 3), or 0 on error.
 */
int npy_ndim(const std::filesystem::path& path);

} // namespace oglcpp
