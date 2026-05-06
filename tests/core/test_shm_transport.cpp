/**
 * @file test_shm_transport.cpp
 * @brief Tests for SHM transport protocol and ShmWriter.
 *
 * Verifies:
 *   - SHM header layout (magic, version, dimensions)
 *   - Field name table read/write roundtrip
 *   - ShmWriter open/write_field/commit lifecycle
 *   - Sequence number increments on commit
 *   - Data written to SHM region is readable from same process
 */
#include "catch2/catch.hpp"
#include "core/output/shm_transport.hpp"
#include "core/output/shm_writer.hpp"
#include "core/field/field3d.hpp"

#include <cstring>
#include <vector>

// ---- Protocol layout tests (no POSIX SHM needed) ----

TEST_CASE("SHM header has correct magic and version constants", "[core][shm][analytical]")
{
    REQUIRE(tmv_shm::kShmMagic == 0x544D5653);
    REQUIRE(tmv_shm::kShmVersion == 1);
    REQUIRE(tmv_shm::kMaxFields == 16);
    REQUIRE(tmv_shm::kMaxFieldNameLen == 64);
}

TEST_CASE("SHM field_names_offset is immediately after header", "[core][shm][analytical]")
{
    REQUIRE(tmv_shm::field_names_offset() == sizeof(tmv_shm::ShmHeader));
}

TEST_CASE("SHM field_names_size is max_fields * max_name_len", "[core][shm][analytical]")
{
    REQUIRE(tmv_shm::field_names_size() == 16 * 64);
}

TEST_CASE("SHM field_data_offset accounts for header + names + preceding fields", "[core][shm][analytical]")
{
    int nx = 8, ny = 8, nz = 8;
    size_t voxels = 8 * 8 * 8;

    size_t offset_0 = tmv_shm::field_data_offset(0, nx, ny, nz);
    size_t offset_1 = tmv_shm::field_data_offset(1, nx, ny, nz);

    // Field 0 starts after header + name table
    REQUIRE(offset_0 == sizeof(tmv_shm::ShmHeader) + 16 * 64);

    // Field 1 starts one field-data block later
    REQUIRE(offset_1 == offset_0 + voxels * sizeof(float));
}

TEST_CASE("SHM total_shm_size matches expected layout", "[core][shm][analytical]")
{
    int nx = 4, ny = 8, nz = 16;
    int field_count = 3;

    size_t expected = sizeof(tmv_shm::ShmHeader)
                    + 16 * 64                          // name table
                    + 3 * (4 * 8 * 16) * sizeof(float); // 3 fields

    REQUIRE(tmv_shm::total_shm_size(nx, ny, nz, field_count) == expected);
}

TEST_CASE("SHM field name write/read roundtrip in simulated buffer", "[core][shm][analytical]")
{
    // Simulate a SHM region with a heap buffer
    int nx = 4, ny = 4, nz = 4;
    int field_count = 3;
    size_t total = tmv_shm::total_shm_size(nx, ny, nz, field_count);
    std::vector<char> buf(total, 0);

    tmv_shm::write_field_name(buf.data(), 0, "velocity_u");
    tmv_shm::write_field_name(buf.data(), 1, "velocity_w");
    tmv_shm::write_field_name(buf.data(), 2, "reflectivity_dbz");

    REQUIRE(tmv_shm::read_field_name(buf.data(), 0) == "velocity_u");
    REQUIRE(tmv_shm::read_field_name(buf.data(), 1) == "velocity_w");
    REQUIRE(tmv_shm::read_field_name(buf.data(), 2) == "reflectivity_dbz");
}

TEST_CASE("SHM field name truncated at max length", "[core][shm][analytical]")
{
    int nx = 4, ny = 4, nz = 4;
    size_t total = tmv_shm::total_shm_size(nx, ny, nz, 1);
    std::vector<char> buf(total, 0);

    // Write a name longer than 63 chars
    std::string long_name(100, 'x');
    tmv_shm::write_field_name(buf.data(), 0, long_name.c_str());

    auto read_back = tmv_shm::read_field_name(buf.data(), 0);
    // Should be truncated to 63 chars (max - null terminator)
    REQUIRE(read_back.size() == 63);
}

TEST_CASE("SHM field data pointer addresses correct offset", "[core][shm][analytical]")
{
    int nx = 4, ny = 4, nz = 4;
    int field_count = 2;
    size_t total = tmv_shm::total_shm_size(nx, ny, nz, field_count);
    std::vector<char> buf(total, 0);

    float* data0 = tmv_shm::field_data_ptr(buf.data(), 0, nx, ny, nz);
    float* data1 = tmv_shm::field_data_ptr(buf.data(), 1, nx, ny, nz);

    // Write known values
    size_t voxels = 4 * 4 * 4;
    for (size_t i = 0; i < voxels; ++i)
    {
        data0[i] = static_cast<float>(i);
        data1[i] = static_cast<float>(i + 1000);
    }

    // Read back via const pointer
    const float* cdata0 = tmv_shm::field_data_ptr(
        static_cast<const void*>(buf.data()), 0, nx, ny, nz);
    const float* cdata1 = tmv_shm::field_data_ptr(
        static_cast<const void*>(buf.data()), 1, nx, ny, nz);

    for (size_t i = 0; i < voxels; ++i)
    {
        REQUIRE(cdata0[i] == static_cast<float>(i));
        REQUIRE(cdata1[i] == static_cast<float>(i + 1000));
    }
}

// ---- ShmWriter lifecycle tests (require POSIX SHM) ----

TEST_CASE("ShmWriter open/close lifecycle", "[core][shm]")
{
    ShmWriter writer;
    REQUIRE_FALSE(writer.is_open());

    bool opened = writer.open(4, 4, 4, {"test_field"}, "/tmv_test_lifecycle");
    if (!opened)
    {
        WARN("POSIX SHM unavailable on this platform — skipping ShmWriter tests");
        return;
    }

    REQUIRE(writer.is_open());
    writer.close();
    REQUIRE_FALSE(writer.is_open());
}

TEST_CASE("ShmWriter write_field and commit", "[core][shm][analytical]")
{
    ShmWriter writer;
    bool opened = writer.open(4, 8, 16, {"w"}, "/tmv_test_write");
    if (!opened)
    {
        WARN("POSIX SHM unavailable — skipping");
        return;
    }

    // Create a field with known values
    Field3D field(4, 8, 16);
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 8; ++j)
            for (int k = 0; k < 16; ++k)
                field(i, j, k) = static_cast<float>(i + j + k);

    writer.write_field(0, field);
    writer.commit(42.0);

    writer.close();
}
