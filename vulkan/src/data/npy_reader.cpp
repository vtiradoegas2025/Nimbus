/**
 * @file npy_reader.cpp
 * @brief Strict float32 NPY parsing for dataset slice ingestion.
 *
 * Implements defensive parsing for `.npy` headers and payload sizes used by
 * tornado export slices. Rejects malformed or unsupported inputs early to avoid
 * crashes, integer-overflow edge cases, and unchecked allocation behavior.
 */

#include "data/npy_reader.hpp"

#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <regex>
#include <sstream>

namespace oglcpp
{
namespace
{

constexpr std::size_t kMaxHeaderBytes = 1u << 20;

/** @brief Read exactly `bytes` bytes from a stream into `ptr`. */
bool read_all(std::ifstream& in, void* ptr, std::size_t bytes)
{
    if (bytes == 0)
    {
        return true;
    }
    if (bytes > static_cast<std::size_t>(std::numeric_limits<std::streamsize>::max()))
    {
        return false;
    }

    const std::streamsize count = static_cast<std::streamsize>(bytes);
    in.read(static_cast<char*>(ptr), count);
    return in.gcount() == count;
}

/** @brief Parse a strictly positive decimal integer with bounds checking. */
bool parse_positive_int(const std::string& text, int& out, std::string& error)
{
    errno = 0;
    char* end = nullptr;
    const unsigned long parsed = std::strtoul(text.c_str(), &end, 10);
    if (end == text.c_str() || *end != '\0' || errno == ERANGE ||
        parsed == 0 || parsed > static_cast<unsigned long>(std::numeric_limits<int>::max()))
    {
        error = "invalid positive integer in NPY shape: '" + text + "'";
        return false;
    }

    out = static_cast<int>(parsed);
    return true;
}

/** @brief Compute `rows*cols` safely for allocation and payload-size validation. */
bool checked_element_count(int rows, int cols, std::size_t& out, std::string& error)
{
    if (rows <= 0 || cols <= 0)
    {
        error = "NPY shape must be strictly positive";
        return false;
    }

    const std::size_t srows = static_cast<std::size_t>(rows);
    const std::size_t scols = static_cast<std::size_t>(cols);
    if (srows > std::numeric_limits<std::size_t>::max() / scols)
    {
        error = "NPY shape overflows size_t";
        return false;
    }

    out = srows * scols;
    return true;
}

/** @brief Parse 2D shape tuple from a NPY header dictionary string. */
std::vector<int> parse_shape(const std::string& header, std::string& error)
{
    std::regex shape_re("'shape'\\s*:\\s*\\(([^\\)]*)\\)");
    std::smatch match;

    if (!std::regex_search(header, match, shape_re) || match.size() < 2)
    {
        error = "NPY header missing shape";
        return {};
    }

    const std::string shape_text = match[1].str();
    std::regex num_re("([0-9]+)");
    std::sregex_iterator it(shape_text.begin(), shape_text.end(), num_re);
    const std::sregex_iterator end;

    std::vector<int> dims;
    dims.reserve(2);

    for (; it != end; ++it)
    {
        int dim = 0;
        if (!parse_positive_int((*it)[1].str(), dim, error))
        {
            return {};
        }
        dims.push_back(dim);
    }

    if (dims.empty())
    {
        error = "Empty shape tuple";
        return {};
    }
    return dims;
}

/** @brief Check that `fortran_order` is explicitly `False` in header dict. */
bool header_has_false_fortran(const std::string& header)
{
    std::regex fortran_re("'fortran_order'\\s*:\\s*(True|False)");
    std::smatch match;
    
    if (!std::regex_search(header, match, fortran_re) || match.size() < 2)
    {
        return false;
    }
    return match[1].str() == "False";
}

/** @brief Check that header dtype descriptor is one of supported float32 encodings. */
bool header_is_supported_dtype(const std::string& header)
{
    std::regex descr_re("'descr'\\s*:\\s*'([^']+)'");
    std::smatch match;
    if (!std::regex_search(header, match, descr_re) || match.size() < 2)
    {
        return false;
    }

    const std::string descr = match[1].str();
    return descr == "<f4" || descr == "|f4" || descr == "=f4";
}

/** @brief Parse/validate NPY header and extract 2D shape for float32 C-order arrays. */
bool read_npy_float32_2d_header(std::ifstream& in,
                                const std::filesystem::path& path,
                                int& rows,
                                int& cols,
                                std::string& error)
{
    char magic[6] = {0};
    if (!read_all(in, magic, sizeof(magic)))
    {
        error = "Failed to read NPY magic";
        return false;
    }

    if (std::string(magic, sizeof(magic)) != "\x93NUMPY")
    {
        error = "Invalid NPY magic in " + path.string();
        return false;
    }

    unsigned char major = 0;
    unsigned char minor = 0;
    if (!read_all(in, &major, 1) || !read_all(in, &minor, 1))
    {
        error = "Failed to read NPY version";
        return false;
    }

    std::size_t header_len = 0;
    if (major == 1)
    {
        std::uint16_t len16 = 0;
        if (!read_all(in, &len16, sizeof(len16)))
        {
            error = "Failed to read v1 header length";
            return false;
        }
        header_len = len16;
    }
    else if (major == 2 || major == 3)
    {
        std::uint32_t len32 = 0;
        if (!read_all(in, &len32, sizeof(len32)))
        {
            error = "Failed to read v2/v3 header length";
            return false;
        }
        header_len = len32;
    }
    else
    {
        std::ostringstream oss;
        oss << "Unsupported NPY version " << static_cast<int>(major) << "." << static_cast<int>(minor);
        error = oss.str();
        return false;
    }

    if (header_len == 0 || header_len > kMaxHeaderBytes)
    {
        error = "NPY header length is invalid or too large";
        return false;
    }

    std::string header(header_len, '\0');
    if (!read_all(in, header.data(), header_len))
    {
        error = "Failed to read NPY header";
        return false;
    }

    if (!header_is_supported_dtype(header))
    {
        error = "Only float32 NPY files are supported";
        return false;
    }

    if (!header_has_false_fortran(header))
    {
        error = "Only C-order NPY arrays are supported";
        return false;
    }

    const std::vector<int> dims = parse_shape(header, error);
    if (dims.empty())
    {
        return false;
    }
    if (dims.size() != 2)
    {
        std::ostringstream oss;
        oss << "Expected 2D array, got " << dims.size() << "D";
        error = oss.str();
        return false;
    }

    rows = dims[0];
    cols = dims[1];
    return true;
}

/**
 * @brief Read and validate NPY header, returning the full dimension vector.
 *
 * Unlike read_npy_float32_2d_header, this accepts any dimensionality.
 */
bool read_npy_float32_header(std::ifstream& in,
                              const std::filesystem::path& path,
                              std::vector<int>& dims_out,
                              std::string& error)
{
    char magic[6] = {0};
    if (!read_all(in, magic, sizeof(magic)))
    {
        error = "Failed to read NPY magic";
        return false;
    }
    if (std::string(magic, sizeof(magic)) != "\x93NUMPY")
    {
        error = "Invalid NPY magic in " + path.string();
        return false;
    }

    unsigned char major = 0;
    unsigned char minor = 0;
    if (!read_all(in, &major, 1) || !read_all(in, &minor, 1))
    {
        error = "Failed to read NPY version";
        return false;
    }

    std::size_t header_len = 0;
    if (major == 1)
    {
        std::uint16_t len16 = 0;
        if (!read_all(in, &len16, sizeof(len16)))
        {
            error = "Failed to read v1 header length";
            return false;
        }
        header_len = len16;
    }
    else if (major == 2 || major == 3)
    {
        std::uint32_t len32 = 0;
        if (!read_all(in, &len32, sizeof(len32)))
        {
            error = "Failed to read v2/v3 header length";
            return false;
        }
        header_len = len32;
    }
    else
    {
        std::ostringstream oss;
        oss << "Unsupported NPY version " << static_cast<int>(major) << "." << static_cast<int>(minor);
        error = oss.str();
        return false;
    }

    if (header_len == 0 || header_len > kMaxHeaderBytes)
    {
        error = "NPY header length is invalid or too large";
        return false;
    }

    std::string header(header_len, '\0');
    if (!read_all(in, header.data(), header_len))
    {
        error = "Failed to read NPY header";
        return false;
    }

    if (!header_is_supported_dtype(header))
    {
        error = "Only float32 NPY files are supported";
        return false;
    }
    if (!header_has_false_fortran(header))
    {
        error = "Only C-order NPY arrays are supported";
        return false;
    }

    dims_out = parse_shape(header, error);
    return !dims_out.empty();
}

} // namespace

/** @brief Load full 2D float32 array payload from a `.npy` file. */
bool load_npy_float32_2d(const std::filesystem::path& path, NpyArray2D& out, std::string& error)
{
    std::ifstream in(path, std::ios::binary);
    if (!in)
    {
        error = "Failed to open file: " + path.string();
        return false;
    }

    if (!read_npy_float32_2d_header(in, path, out.rows, out.cols, error))
    {
        return false;
    }

    std::size_t n = 0;
    if (!checked_element_count(out.rows, out.cols, n, error))
    {
        return false;
    }
    if (n > std::numeric_limits<std::size_t>::max() / sizeof(float))
    {
        error = "NPY payload size overflows size_t";
        return false;
    }

    out.data.resize(n);
    if (!read_all(in, out.data.data(), n * sizeof(float)))
    {
        error = "Failed to read NPY payload data";
        return false;
    }

    return true;
}

/** @brief Load only shape metadata from a `.npy` file. */
bool load_npy_float32_2d_shape(const std::filesystem::path& path, NpyArray2DShape& out, std::string& error)
{
    std::ifstream in(path, std::ios::binary);
    if (!in)
    {
        error = "Failed to open file: " + path.string();
        return false;
    }

    return read_npy_float32_2d_header(in, path, out.rows, out.cols, error);
}

bool load_npy_float32_3d(const std::filesystem::path& path, NpyArray3D& out, std::string& error)
{
    std::ifstream in(path, std::ios::binary);
    if (!in)
    {
        error = "Failed to open file: " + path.string();
        return false;
    }

    std::vector<int> dims;
    if (!read_npy_float32_header(in, path, dims, error))
    {
        return false;
    }
    if (dims.size() != 3)
    {
        std::ostringstream oss;
        oss << "Expected 3D array, got " << dims.size() << "D in " << path.string();
        error = oss.str();
        return false;
    }

    out.dim0 = dims[0];
    out.dim1 = dims[1];
    out.dim2 = dims[2];

    // Overflow-safe element count
    const std::size_t n01 = static_cast<std::size_t>(dims[0]) *
                            static_cast<std::size_t>(dims[1]);
    if (dims[0] > 0 && n01 / static_cast<std::size_t>(dims[0]) !=
                        static_cast<std::size_t>(dims[1]))
    {
        error = "Shape overflow in 3D NPY";
        return false;
    }
    const std::size_t n = n01 * static_cast<std::size_t>(dims[2]);
    if (n01 > 0 && n / n01 != static_cast<std::size_t>(dims[2]))
    {
        error = "Shape overflow in 3D NPY";
        return false;
    }
    if (n > std::numeric_limits<std::size_t>::max() / sizeof(float))
    {
        error = "NPY payload size overflows size_t";
        return false;
    }

    out.data.resize(n);
    if (!read_all(in, out.data.data(), n * sizeof(float)))
    {
        error = "Failed to read 3D NPY payload data";
        return false;
    }

    return true;
}

int npy_ndim(const std::filesystem::path& path)
{
    std::ifstream in(path, std::ios::binary);
    if (!in)
    {
        return 0;
    }

    std::vector<int> dims;
    std::string error;
    if (!read_npy_float32_header(in, path, dims, error))
    {
        return 0;
    }
    return static_cast<int>(dims.size());
}

} // namespace oglcpp
