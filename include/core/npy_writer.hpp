#pragma once

/**
 * @file npy_writer.hpp
 * @brief NumPy .npy binary format writer utilities.
 *
 * Provides standalone functions for writing 2D and 3D float32 arrays
 * in NumPy v1.0 format. Extracted from the headless runtime export path
 * to enable reuse by the async output writer and test infrastructure.
 */

#include <cstddef>
#include <string>
#include <vector>

class Field3D;

namespace npy {

/**
 * @brief Write a 2D float32 array to a .npy file.
 * @param data     Contiguous row-major buffer of size rows * cols.
 * @param rows     Number of rows (first dimension in shape tuple).
 * @param cols     Number of columns (second dimension in shape tuple).
 * @param path     Output file path.
 * @return True on success, false on I/O error.
 *
 * Writes NumPy v1.0 format: magic + version + header dict + binary payload.
 * Shape in header is (rows, cols) with dtype '<f4' and C-order.
 */
bool write_2d(const float* data, int rows, int cols, const std::string& path);

/**
 * @brief Write a 2D float32 array from a vector.
 * @param buf      Data buffer; must have exactly rows * cols elements.
 * @param rows     Number of rows.
 * @param cols     Number of columns.
 * @param path     Output file path.
 * @return True on success, false on size mismatch or I/O error.
 */
bool write_2d(const std::vector<float>& buf, int rows, int cols,
              const std::string& path);

/**
 * @brief Write a 3D float32 array to a .npy file.
 * @param data     Contiguous row-major buffer of size dim0 * dim1 * dim2.
 * @param dim0     First dimension (NR for Field3D).
 * @param dim1     Second dimension (NTH for Field3D).
 * @param dim2     Third dimension (NZ for Field3D).
 * @param path     Output file path.
 * @return True on success, false on I/O error.
 *
 * Shape in header is (dim0, dim1, dim2) with dtype '<f4' and C-order.
 */
bool write_3d(const float* data, int dim0, int dim1, int dim2,
              const std::string& path);

/**
 * @brief Write a Field3D directly to a 3D .npy file.
 * @param field    Source field with contiguous storage in [NR][NTH][NZ] order.
 * @param path     Output file path.
 * @return True on success, false on I/O error or empty field.
 *
 * Writes the entire field as shape (NR, NTH, NZ). This is a zero-copy
 * path for core fields: the data pointer is passed directly to the writer.
 */
bool write_field3d(const Field3D& field, const std::string& path);

/**
 * @brief Extract a 2D theta slice from a Field3D and write it as .npy.
 * @param field    Source 3D field.
 * @param theta    Azimuthal index to extract.
 * @param path     Output file path.
 * @return True on success, false on I/O error.
 *
 * Writes a (NZ, NR) slice — the legacy per-theta export format.
 */
bool write_field_slice(const Field3D& field, int theta,
                       const std::string& path);

} // namespace npy

namespace csv {

/**
 * @brief Write a 3D float32 array to a CSV file.
 * @param data     Contiguous row-major buffer [dim0][dim1][dim2].
 * @param dim0     First dimension (NR).
 * @param dim1     Second dimension (NTH).
 * @param dim2     Third dimension (NZ).
 * @param path     Output file path.
 * @return True on success, false on I/O error.
 *
 * Format: header row "i,j,k,value", then one row per non-zero voxel.
 * Zeros are skipped to keep file size manageable. Students can open
 * this in Excel, Google Sheets, or Python pandas.
 */
bool write_3d(const float* data, int dim0, int dim1, int dim2,
              const std::string& path);

/**
 * @brief Write a Field3D to CSV.
 */
bool write_field3d(const Field3D& field, const std::string& path);

} // namespace csv
