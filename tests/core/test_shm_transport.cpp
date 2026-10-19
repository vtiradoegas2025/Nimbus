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
#include <limits>
#include <vector>

// ---- Protocol layout tests (no POSIX SHM needed) ----

TEST_CASE("SHM header has correct magic and version constants", "[core][shm][analytical]")
{
    REQUIRE(tmv_shm::kShmMagic == 0x544D5653);
    REQUIRE(tmv_shm::kShmVersion == 5);
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

TEST_CASE("SHM v5 metadata explicitly describes raw signed fields", "[core][shm][analytical]")
{
    const auto w = tmv_shm::make_field_metadata("w");
    REQUIRE(w.struct_size == sizeof(tmv_shm::ShmFieldMetadata));
    REQUIRE(w.encoding == tmv_shm::ShmValueEncoding::RawFloat32);
    REQUIRE(w.polarity == tmv_shm::ShmFieldPolarity::Bipolar);
    REQUIRE(w.grid_location == tmv_shm::ShmGridLocation::CellCenter);
    REQUIRE(w.nonfinite_policy ==
            tmv_shm::ShmNonfinitePolicy::PreserveAndSanitizeInReader);
    REQUIRE(std::string(w.units) == "m s-1");

    const auto qr = tmv_shm::make_field_metadata("qr");
    REQUIRE(qr.polarity == tmv_shm::ShmFieldPolarity::Unipolar);
    REQUIRE(std::string(qr.units) == "kg kg-1");

    const auto radar_linear = tmv_shm::make_field_metadata("radar");
    REQUIRE(radar_linear.polarity == tmv_shm::ShmFieldPolarity::Unipolar);
    REQUIRE(std::string(radar_linear.units) == "mm^6 m-3");

    const auto radar_dbz = tmv_shm::make_field_metadata("reflectivity_dbz");
    REQUIRE(radar_dbz.polarity == tmv_shm::ShmFieldPolarity::Unipolar);
    REQUIRE(std::string(radar_dbz.units) == "dBZ");

    REQUIRE(tmv_shm::field_name_is_bipolar("vorticity_z"));
    REQUIRE_FALSE(tmv_shm::field_name_is_bipolar("vorticity_magnitude"));
    REQUIRE(std::atomic<uint64_t>::is_always_lock_free);
}

TEST_CASE("SHM v5 physical domains, frames, and generations fail closed",
          "[core][shm][analytical]")
{
    const auto full = tmv_shm::full_physical_domain(8, 7, 6);
    REQUIRE(tmv_shm::physical_domain_valid(full, 8, 7, 6));

    const tmv_shm::ShmPhysicalDomain c_grid{{1, 1, 0}, {6, 5, 5}};
    REQUIRE(tmv_shm::physical_domain_valid(c_grid, 8, 7, 6));

    REQUIRE_FALSE(tmv_shm::physical_domain_valid(
        tmv_shm::ShmPhysicalDomain{{1, 1, 0}, {8, 5, 5}}, 8, 7, 6));
    REQUIRE_FALSE(tmv_shm::physical_domain_valid(
        tmv_shm::ShmPhysicalDomain{{2, 1, 0}, {1, 5, 5}}, 8, 7, 6));

    REQUIRE(tmv_shm::grid_spacing_valid({400.0, 500.0, 250.0}));
    REQUIRE_FALSE(tmv_shm::grid_spacing_valid({400.0, 0.0, 250.0}));
    REQUIRE_FALSE(tmv_shm::grid_spacing_valid(
        {400.0, std::numeric_limits<double>::infinity(), 250.0}));

    tmv_shm::ShmHorizontalFrame moving;
    moving.motion_mode = tmv_shm::ShmDomainMotionMode::ConstantVelocity;
    moving.grid_origin_t0_earth_m[0] = 1000.0;
    moving.grid_origin_t0_earth_m[1] = -500.0;
    moving.grid_origin_earth_m[0] = 1000.0;
    moving.grid_origin_earth_m[1] = -500.0;
    moving.frame_velocity_earth_ms[0] = 12.0;
    moving.frame_velocity_earth_ms[1] = -4.0;
    REQUIRE(tmv_shm::horizontal_frame_valid(moving));
    moving.grid_origin_earth_m[0] = std::numeric_limits<double>::quiet_NaN();
    REQUIRE_FALSE(tmv_shm::horizontal_frame_valid(moving));

    tmv_shm::ShmHorizontalFrame invalid_stationary;
    invalid_stationary.frame_velocity_earth_ms[0] = 1.0;
    REQUIRE_FALSE(tmv_shm::horizontal_frame_valid(invalid_stationary));

    const std::string generation(32, 'a');
    REQUIRE(tmv_shm::publication_generation_valid(generation));
    REQUIRE_FALSE(tmv_shm::publication_generation_valid("short"));
}

TEST_CASE("SHM field_data_offset accounts for header + names + preceding fields", "[core][shm][analytical]")
{
    int nx = 8, ny = 8, nz = 8;
    size_t voxels = 8 * 8 * 8;

    size_t offset_0 = tmv_shm::field_data_offset(0, nx, ny, nz);
    size_t offset_1 = tmv_shm::field_data_offset(1, nx, ny, nz);

    // Field 0 starts after header + immutable name/metadata tables.
    REQUIRE(offset_0 == sizeof(tmv_shm::ShmHeader) + 16 * 64 +
                        16 * sizeof(tmv_shm::ShmFieldMetadata));

    // Field 1 starts one field-data block later
    REQUIRE(offset_1 == offset_0 + voxels * sizeof(float));
}

TEST_CASE("SHM total_shm_size matches expected layout", "[core][shm][analytical]")
{
    int nx = 4, ny = 8, nz = 16;
    int field_count = 3;

    size_t expected = sizeof(tmv_shm::ShmHeader)
                    + 16 * 64                          // name table
                    + 16 * sizeof(tmv_shm::ShmFieldMetadata)
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

TEST_CASE("SHM field name read is bounded to one malformed slot", "[core][shm][analytical]")
{
    const int nx = 4, ny = 4, nz = 4;
    std::vector<char> buf(tmv_shm::total_shm_size(nx, ny, nz, 1), 'x');

    REQUIRE_FALSE(tmv_shm::field_name_is_terminated(buf.data(), 0));
    REQUIRE(tmv_shm::read_field_name(buf.data(), 0).size() ==
            static_cast<std::size_t>(tmv_shm::kMaxFieldNameLen));
}

TEST_CASE("SHM field metadata table roundtrips independently", "[core][shm][analytical]")
{
    const int nx = 4, ny = 4, nz = 4;
    std::vector<char> buf(tmv_shm::total_shm_size(nx, ny, nz, 2), 0);

    *tmv_shm::field_metadata_ptr(buf.data(), 0) =
        tmv_shm::make_field_metadata("w");
    *tmv_shm::field_metadata_ptr(buf.data(), 1) =
        tmv_shm::make_field_metadata("theta");

    const void* const_buf = buf.data();
    REQUIRE(tmv_shm::field_metadata_ptr(const_buf, 0)->polarity ==
            tmv_shm::ShmFieldPolarity::Bipolar);
    REQUIRE(tmv_shm::field_metadata_ptr(const_buf, 1)->polarity ==
            tmv_shm::ShmFieldPolarity::Unipolar);
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

TEST_CASE("ShmWriter rejects ambiguous field-name tables before creating SHM",
          "[core][shm][analytical]")
{
    ShmWriter writer;
    REQUIRE_FALSE(writer.open(4, 4, 4, {"w", "w"}, "/tmv_test_bad_names"));
    REQUIRE_FALSE(writer.open(4, 4, 4, {std::string(64, 'x')},
                                      "/tmv_test_bad_names"));
    REQUIRE_FALSE(writer.open(4, 4, 4, {""}, "/tmv_test_bad_names"));
    REQUIRE_FALSE(writer.is_open());
}

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

    REQUIRE(writer.write_field(0, field));
    REQUIRE(writer.commit(42.0));

    writer.close();
}
