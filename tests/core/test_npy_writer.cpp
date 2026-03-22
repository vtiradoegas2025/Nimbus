/**
 * @file test_npy_writer.cpp
 * @brief Unit tests for NPY and CSV file writers with roundtrip value verification.
 *
 * Tests write data, then read it back and verify exact values match.
 * NPY magic: \x93NUMPY, version 1.0, dtype '<f4', C-order.
 */
#include "catch2/catch.hpp"
#include "core/npy_writer.hpp"
#include "core/field3d.hpp"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <vector>

namespace
{

std::string tmp_path(const std::string& name)
{
    return std::filesystem::temp_directory_path().string() + "/tmv_test_" + name;
}

void cleanup(const std::string& path)
{
    std::remove(path.c_str());
}

// Minimal NPY reader for roundtrip tests — reads back a 3D float32 NPY
bool read_npy_3d(const std::string& path, std::vector<float>& out,
                 int& d0, int& d1, int& d2)
{
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) return false;

    // Read magic
    char magic[6];
    f.read(magic, 6);
    if (magic[0] != '\x93' || magic[1] != 'N') return false;

    // Version
    uint8_t major, minor;
    f.read(reinterpret_cast<char*>(&major), 1);
    f.read(reinterpret_cast<char*>(&minor), 1);

    // Header length
    uint16_t header_len;
    f.read(reinterpret_cast<char*>(&header_len), 2);

    // Read header dict
    std::string header(header_len, ' ');
    f.read(&header[0], header_len);

    // Parse shape — find (d0, d1, d2)
    auto pos = header.find("shape");
    if (pos == std::string::npos) return false;
    pos = header.find('(', pos);
    if (pos == std::string::npos) return false;
    if (sscanf(header.c_str() + pos, "(%d, %d, %d)", &d0, &d1, &d2) != 3)
        return false;

    // Read data
    size_t count = static_cast<size_t>(d0) * d1 * d2;
    out.resize(count);
    f.read(reinterpret_cast<char*>(out.data()), count * sizeof(float));

    return f.good();
}

} // namespace

TEST_CASE("NPY magic number is correct", "[core][npy][analytical]")
{
    auto path = tmp_path("magic.npy");
    Field3D field(2, 3, 4, 1.0f);
    REQUIRE(npy::write_field3d(field, path));

    std::ifstream f(path, std::ios::binary);
    char magic[6];
    f.read(magic, 6);
    REQUIRE(magic[0] == '\x93');
    REQUIRE(magic[1] == 'N');
    REQUIRE(magic[2] == 'U');
    REQUIRE(magic[3] == 'M');
    REQUIRE(magic[4] == 'P');
    REQUIRE(magic[5] == 'Y');

    // Version 1.0
    uint8_t major, minor;
    f.read(reinterpret_cast<char*>(&major), 1);
    f.read(reinterpret_cast<char*>(&minor), 1);
    REQUIRE(major == 1);
    REQUIRE(minor == 0);

    cleanup(path);
}

TEST_CASE("NPY 3D roundtrip preserves exact values", "[core][npy][analytical]")
{
    auto path = tmp_path("roundtrip3d.npy");
    const int NR = 4, NTH = 6, NZ = 8;

    // Fill with known pattern: value = i*100 + j*10 + k
    Field3D field(NR, NTH, NZ);
    for (int i = 0; i < NR; ++i)
        for (int j = 0; j < NTH; ++j)
            for (int k = 0; k < NZ; ++k)
                field(i, j, k) = static_cast<float>(i * 100 + j * 10 + k);

    REQUIRE(npy::write_field3d(field, path));

    // Read back
    std::vector<float> readback;
    int d0, d1, d2;
    REQUIRE(read_npy_3d(path, readback, d0, d1, d2));

    REQUIRE(d0 == NR);
    REQUIRE(d1 == NTH);
    REQUIRE(d2 == NZ);
    REQUIRE(readback.size() == static_cast<size_t>(NR * NTH * NZ));

    // Verify every value matches exactly (float32 roundtrip is lossless)
    for (int i = 0; i < NR; ++i)
        for (int j = 0; j < NTH; ++j)
            for (int k = 0; k < NZ; ++k)
            {
                size_t idx = i * NTH * NZ + j * NZ + k;
                float expected = static_cast<float>(i * 100 + j * 10 + k);
                REQUIRE(readback[idx] == expected);
            }

    cleanup(path);
}

TEST_CASE("NPY 2D roundtrip preserves data size", "[core][npy][analytical]")
{
    auto path = tmp_path("roundtrip2d.npy");
    const int rows = 5, cols = 7;
    std::vector<float> data(rows * cols);
    for (int i = 0; i < rows * cols; ++i)
        data[i] = static_cast<float>(i) * 0.1f;

    REQUIRE(npy::write_2d(data, rows, cols, path));

    // Check file size: 10 bytes preamble + padded header + data
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    auto file_size = f.tellg();
    // Data payload = rows * cols * 4 bytes = 140 bytes
    // File should be larger than just the data (header overhead)
    REQUIRE(file_size > 140);
    // But not absurdly large
    REQUIRE(file_size < 500);

    cleanup(path);
}

TEST_CASE("npy::write_2d rejects size mismatch", "[core][npy]")
{
    auto path = tmp_path("w2d_bad.npy");
    std::vector<float> data(10);  // 10 != 3*4
    REQUIRE_FALSE(npy::write_2d(data, 3, 4, path));
    cleanup(path);
}

TEST_CASE("npy::write_field3d rejects empty field", "[core][npy]")
{
    auto path = tmp_path("empty.npy");
    Field3D empty;
    REQUIRE_FALSE(npy::write_field3d(empty, path));
    cleanup(path);
}

TEST_CASE("csv::write_field3d writes correct values", "[core][csv][analytical]")
{
    auto path = tmp_path("field.csv");
    Field3D field(2, 2, 2, 0.0f);
    field(0, 0, 0) = 1.5f;
    field(1, 1, 1) = 2.5f;

    REQUIRE(csv::write_field3d(field, path));

    std::ifstream f(path);
    REQUIRE(f.is_open());

    std::string header;
    std::getline(f, header);
    REQUIRE(header.find("i,j,k,value") != std::string::npos);

    // Read data lines and verify values
    bool found_1_5 = false;
    bool found_2_5 = false;
    std::string line;
    while (std::getline(f, line))
    {
        if (line.find("1.5") != std::string::npos) found_1_5 = true;
        if (line.find("2.5") != std::string::npos) found_2_5 = true;
    }
    REQUIRE(found_1_5);
    REQUIRE(found_2_5);

    cleanup(path);
}
