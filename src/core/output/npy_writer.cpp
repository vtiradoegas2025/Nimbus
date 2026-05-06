/**
 * @file npy_writer.cpp
 * @brief NumPy .npy binary format writer implementation.
 *
 * All write functions produce NumPy v1.0 format files:
 *   - 6-byte magic ("\x93NUMPY")
 *   - 2-byte version (1, 0)
 *   - 2-byte header length (uint16_t LE)
 *   - ASCII header dict padded to 16-byte alignment
 *   - Binary float32 payload in C-order (row-major)
 */

#include "core/output/npy_writer.hpp"

#include "core/field/field3d.hpp"

#include <cstdint>
#include <cstring>
#include <fstream>
#include <string>

namespace npy {
namespace {

/**
 * @brief Write a NPY v1.0 file with the given header dict and binary payload.
 * @param header_dict  Python dict string (e.g. "{'descr': '<f4', ...}").
 * @param payload      Pointer to raw binary data.
 * @param payload_bytes Number of bytes in the payload.
 * @param path         Output file path.
 * @return True on success.
 */
bool write_npy_raw(const std::string& header_dict, const void* payload,
                   std::size_t payload_bytes, const std::string& path)
{
    // NPY v1.0 preamble: magic(6) + version(2) + header_len(2) = 10 bytes
    constexpr std::size_t preamble_size = 10;

    // header_len includes the dict string + padding + terminating newline
    std::size_t dict_plus_newline = header_dict.size() + 1; // +1 for '\n'
    std::size_t total_before_pad = preamble_size + dict_plus_newline;
    std::size_t padding = (16 - (total_before_pad % 16)) % 16;
    std::size_t header_len = dict_plus_newline + padding;

    // NPY v1.0 requires header_len fits in uint16_t
    if (header_len > 65535)
    {
        return false;
    }

    std::ofstream out(path, std::ios::binary);
    if (!out)
    {
        return false;
    }

    // Magic
    out.write("\x93NUMPY", 6);

    // Version 1.0
    out.put(static_cast<char>(1));
    out.put(static_cast<char>(0));

    // Header length (little-endian uint16)
    const auto hl = static_cast<uint16_t>(header_len);
    char len_bytes[2];
    len_bytes[0] = static_cast<char>(hl & 0xFF);
    len_bytes[1] = static_cast<char>((hl >> 8) & 0xFF);
    out.write(len_bytes, 2);

    // Header dict
    out.write(header_dict.c_str(),
              static_cast<std::streamsize>(header_dict.size()));

    // Padding spaces
    for (std::size_t i = 0; i < padding; ++i)
    {
        out.put(' ');
    }

    // Terminating newline
    out.put('\n');

    // Binary payload
    out.write(static_cast<const char*>(payload),
              static_cast<std::streamsize>(payload_bytes));

    out.close();
    return out.good();
}

/**
 * @brief Build a NPY header dict string for a given shape.
 */
std::string make_header_dict(const std::vector<int>& shape)
{
    std::string dict = "{'descr': '<f4', 'fortran_order': False, 'shape': (";
    for (std::size_t i = 0; i < shape.size(); ++i)
    {
        if (i > 0)
        {
            dict += ", ";
        }
        dict += std::to_string(shape[i]);
    }
    // Trailing comma for 1-element tuple compatibility, then close
    if (shape.size() == 1)
    {
        dict += ",";
    }
    dict += "), }";
    return dict;
}

} // anonymous namespace

bool write_2d(const float* data, int rows, int cols, const std::string& path)
{
    if (!data || rows <= 0 || cols <= 0)
    {
        return false;
    }
    const std::string dict = make_header_dict({rows, cols});
    const std::size_t bytes = static_cast<std::size_t>(rows) *
                              static_cast<std::size_t>(cols) * sizeof(float);
    return write_npy_raw(dict, data, bytes, path);
}

bool write_2d(const std::vector<float>& buf, int rows, int cols,
              const std::string& path)
{
    const auto expected = static_cast<std::size_t>(rows) *
                          static_cast<std::size_t>(cols);
    if (buf.size() != expected)
    {
        return false;
    }
    return write_2d(buf.data(), rows, cols, path);
}

bool write_3d(const float* data, int dim0, int dim1, int dim2,
              const std::string& path)
{
    if (!data || dim0 <= 0 || dim1 <= 0 || dim2 <= 0)
    {
        return false;
    }
    const std::string dict = make_header_dict({dim0, dim1, dim2});
    const std::size_t bytes = static_cast<std::size_t>(dim0) *
                              static_cast<std::size_t>(dim1) *
                              static_cast<std::size_t>(dim2) * sizeof(float);
    return write_npy_raw(dict, data, bytes, path);
}

bool write_field3d(const Field3D& field, const std::string& path)
{
    if (field.empty())
    {
        return false;
    }
    return write_3d(field.data(),
                    field.size_r(), field.size_th(), field.size_z(), path);
}

bool write_field_slice(const Field3D& field, int theta, const std::string& path)
{
    const int nr = field.size_r();
    const int nz = field.size_z();
    const int nth = field.size_th();

    if (nr <= 0 || nz <= 0 || theta < 0 || theta >= nth)
    {
        return false;
    }

    // Extract (NZ, NR) slice — same layout as legacy per-theta export
    const auto slice_size = static_cast<std::size_t>(nr) *
                            static_cast<std::size_t>(nz);
    std::vector<float> buf(slice_size);

    std::size_t idx = 0;
    for (int k = 0; k < nz; ++k)
    {
        for (int i = 0; i < nr; ++i)
        {
            buf[idx++] = field(i, theta, k);
        }
    }

    return write_2d(buf.data(), nz, nr, path);
}

} // namespace npy

namespace csv {

bool write_3d(const float* data, int dim0, int dim1, int dim2,
              const std::string& path)
{
    if (data == nullptr || dim0 <= 0 || dim1 <= 0 || dim2 <= 0)
    {
        return false;
    }

    std::ofstream out(path);
    if (!out)
    {
        return false;
    }

    // Header row
    out << "i,j,k,value\n";

    // Write all voxels (skip exact zeros to reduce file size)
    for (int i = 0; i < dim0; ++i)
    {
        for (int j = 0; j < dim1; ++j)
        {
            for (int k = 0; k < dim2; ++k)
            {
                const float val = data[static_cast<std::size_t>(i) *
                                       static_cast<std::size_t>(dim1) *
                                       static_cast<std::size_t>(dim2) +
                                       static_cast<std::size_t>(j) *
                                       static_cast<std::size_t>(dim2) +
                                       static_cast<std::size_t>(k)];
                if (val != 0.0f)
                {
                    out << i << ',' << j << ',' << k << ',' << val << '\n';
                }
            }
        }
    }

    return out.good();
}

bool write_field3d(const Field3D& field, const std::string& path)
{
    if (field.empty())
    {
        return false;
    }
    return write_3d(field.data(), field.size_r(), field.size_th(), field.size_z(), path);
}

} // namespace csv
