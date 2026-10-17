// Reject malformed headers and impossible payload sizes before allocation.

#include "core/io/npy_reader.hpp"

#include <cerrno>
#include <charconv>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <string_view>
#include <sstream>

namespace oglcpp
{
namespace
{

constexpr std::size_t kMaxHeaderBytes = 1u << 20;

// Reads exactly `bytes` or fails without accepting a partial payload.
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

bool read_payload(std::ifstream& in, std::size_t count, std::vector<float>& data, std::string& error)
{
    const auto begin = in.tellg();
    in.seekg(0, std::ios::end);
    const auto end = in.tellg();
    if (begin < 0 || end < begin || count > static_cast<std::uintmax_t>(end - begin) / sizeof(float))
    {
        error = "NPY payload is smaller than its declared shape";
        return false;
    }
    in.seekg(begin);
    data.resize(count);
    if (!read_all(in, data.data(), count * sizeof(float)))
    {
        error = "Failed to read NPY payload data";
        return false;
    }
    return true;
}

// Computes `rows*cols` safely before allocation.
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

bool parse_header_dictionary(std::string_view text, std::vector<int>& shape, std::string& error)
{
    std::size_t cursor = 0;
    unsigned seen = 0;
    const auto whitespace = [&] {
        while (cursor < text.size() && std::isspace(static_cast<unsigned char>(text[cursor]))) ++cursor;
    };
    const auto consume = [&](char value) {
        whitespace();
        if (cursor == text.size() || text[cursor] != value) return false;
        ++cursor;
        return true;
    };
    const auto quoted = [&](std::string_view& value) {
        whitespace();
        if (cursor == text.size() || (text[cursor] != '\'' && text[cursor] != '"')) return false;
        const char quote = text[cursor++];
        const auto end = text.find(quote, cursor);
        if (end == std::string_view::npos) return false;
        value = text.substr(cursor, end - cursor);
        if (value.find('\\') != std::string_view::npos) return false;
        cursor = end + 1;
        return true;
    };
    const auto invalid = [&] {
        error = "invalid NPY header dictionary or shape tuple";
        return false;
    };
    if (!consume('{')) return invalid();
    while (true)
    {
        std::string_view key;
        if (!quoted(key) || !consume(':')) return invalid();
        const unsigned bit = key == "descr" ? 1U : key == "fortran_order" ? 2U : key == "shape" ? 4U : 0U;
        if (bit == 0 || (seen & bit) != 0) return invalid();
        seen |= bit;
        if (bit == 1)
        {
            std::string_view dtype;
            if (!quoted(dtype) || (dtype != "<f4" && dtype != "|f4" && dtype != "=f4"))
            {
                error = "Only float32 NPY files are supported";
                return false;
            }
        }
        else if (bit == 2)
        {
            whitespace();
            if (text.substr(cursor, 5) != "False")
            {
                error = "Only C-order NPY arrays are supported";
                return false;
            }
            cursor += 5;
        }
        else
        {
            if (!consume('(')) return invalid();
            bool comma = false;
            while (true)
            {
                whitespace();
                if (cursor == text.size() || text[cursor] < '1' || text[cursor] > '9') return invalid();
                int dimension = 0;
                const auto number = std::from_chars(text.data() + cursor, text.data() + text.size(), dimension);
                if (number.ec != std::errc{} || dimension <= 0) return invalid();
                cursor = static_cast<std::size_t>(number.ptr - text.data());
                shape.push_back(dimension);
                if (shape.size() > 3) return invalid();
                if (consume(')')) break;
                if (!consume(',')) return invalid();
                comma = true;
                if (consume(')')) break;
            }
            if (!comma) return invalid();
        }
        if (consume('}')) break;
        if (!consume(',')) return invalid();
        if (consume('}')) break;
    }
    whitespace();
    return seen == 7 && cursor == text.size() ? true : invalid();
}

// Validates the header and returns its complete dimension vector.
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

    return parse_header_dictionary(header, dims_out, error);
}

bool read_npy_float32_2d_header(std::ifstream& in,
                                const std::filesystem::path& path,
                                int& rows, int& cols, std::string& error)
{
    std::vector<int> dims;
    if (!read_npy_float32_header(in, path, dims, error)) return false;
    if (dims.size() != 2)
    {
        error = "Expected a 2D NPY array";
        return false;
    }
    rows = dims[0];
    cols = dims[1];
    return true;
}

} // namespace

// Loads a complete 2D float32 payload.
bool load_npy_float32_2d(const std::filesystem::path& path, NpyArray2D& out, std::string& error)
{
    NpyArray2D candidate;
    std::ifstream in(path, std::ios::binary);
    if (!in)
    {
        error = "Failed to open file: " + path.string();
        return false;
    }

    if (!read_npy_float32_2d_header(in, path, candidate.rows, candidate.cols, error))
    {
        return false;
    }

    std::size_t n = 0;
    if (!checked_element_count(candidate.rows, candidate.cols, n, error))
    {
        return false;
    }
    if (n > std::numeric_limits<std::size_t>::max() / sizeof(float))
    {
        error = "NPY payload size overflows size_t";
        return false;
    }

    if (!read_payload(in, n, candidate.data, error)) return false;
    out = std::move(candidate);
    return true;
}

// Loads only 2D shape metadata.
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
    NpyArray3D candidate;
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

    candidate.dim0 = dims[0];
    candidate.dim1 = dims[1];
    candidate.dim2 = dims[2];

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

    if (!read_payload(in, n, candidate.data, error)) return false;
    out = std::move(candidate);
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
