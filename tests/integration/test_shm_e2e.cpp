/**
 * @file test_shm_e2e.cpp
 * @brief End-to-end SHM writer→reader roundtrip test.
 *
 * Verifies that data written by ShmWriter (simulation side) can be read
 * by ShmDataset (viewer side) through the POSIX shared memory region.
 * This confirms the full live visualization data path in-process.
 *
 * Tests:
 *   - Writer creates region, reader attaches and finds the field
 *   - Written data survives the roundtrip (non-zero, correct shape)
 *   - Sequence number advances on commit
 *   - Reader detects new frames via has_new_frame()
 *   - Multiple fields are independently addressable
 */
#include "catch2/catch.hpp"
#include "core/output/shm_writer.hpp"
#include "core/output/shm_transport.hpp"
#include "core/field3d.hpp"
#include "data/shm_dataset.hpp"

static constexpr int kNX = 8;
static constexpr int kNY = 8;
static constexpr int kNZ = 8;
static constexpr const char* kShmName = "/tmv_test_e2e";

TEST_CASE("SHM writer→reader roundtrip: single field", "[integration][shm][e2e]")
{
    ShmWriter writer;
    bool opened = writer.open(kNX, kNY, kNZ, {"w"}, kShmName);
    if (!opened)
    {
        WARN("POSIX SHM unavailable — skipping SHM e2e test");
        return;
    }
    REQUIRE(writer.is_open());

    // Write a field with a known gradient pattern
    Field3D field(kNX, kNY, kNZ);
    for (int i = 0; i < kNX; ++i)
        for (int j = 0; j < kNY; ++j)
            for (int k = 0; k < kNZ; ++k)
                field(i, j, k) = static_cast<float>(i + j + k);

    writer.write_field(0, field);
    writer.commit(1.0);

    // Reader attaches and finds the field
    oglcpp::ShmDataset reader("w", kShmName);
    std::string error;
    REQUIRE(reader.scan(error));
    REQUIRE(reader.nx() == kNX);
    REQUIRE(reader.ny() == kNY);
    REQUIRE(reader.nz() == kNZ);
    REQUIRE(reader.frame_count() == 1);

    // Load the frame
    oglcpp::VolumeFrame frame;
    REQUIRE(reader.load_frame(0, frame, error));
    REQUIRE(frame.nx == kNX);
    REQUIRE(frame.ny == kNY);
    REQUIRE(frame.nz == kNZ);

    const std::size_t voxels = static_cast<std::size_t>(kNX) * kNY * kNZ;
    REQUIRE(frame.normalized.size() == voxels);

    // Data should be normalized [0,1] — verify non-trivial content.
    // The writer does percentile normalization, so we check that the range
    // is not degenerate and that at least some values differ from zero.
    float sum = 0.0f;
    for (std::size_t i = 0; i < voxels; ++i)
    {
        REQUIRE(frame.normalized[i] >= 0.0f);
        REQUIRE(frame.normalized[i] <= 1.0f);
        sum += frame.normalized[i];
    }
    // With a linear gradient 0..21, normalized data should have non-zero mean
    REQUIRE(sum > 0.0f);

    writer.close();
}

TEST_CASE("SHM sequence number advances on commit", "[integration][shm][e2e]")
{
    ShmWriter writer;
    bool opened = writer.open(kNX, kNY, kNZ, {"theta"}, kShmName);
    if (!opened)
    {
        WARN("POSIX SHM unavailable — skipping");
        return;
    }

    Field3D field(kNX, kNY, kNZ);
    for (int i = 0; i < kNX; ++i)
        for (int j = 0; j < kNY; ++j)
            for (int k = 0; k < kNZ; ++k)
                field(i, j, k) = 300.0f + static_cast<float>(k);

    // First commit
    writer.write_field(0, field);
    writer.commit(1.0);

    oglcpp::ShmDataset reader("theta", kShmName);
    std::string error;
    REQUIRE(reader.scan(error));

    oglcpp::VolumeFrame frame;
    REQUIRE(reader.load_frame(0, frame, error));
    REQUIRE_FALSE(reader.has_new_frame());

    // Second commit should be detected as new
    writer.write_field(0, field);
    writer.commit(2.0);
    REQUIRE(reader.has_new_frame());

    // Loading consumes the new-frame flag
    REQUIRE(reader.load_frame(0, frame, error));
    REQUIRE_FALSE(reader.has_new_frame());

    writer.close();
}

TEST_CASE("SHM multiple fields are independently addressable", "[integration][shm][e2e]")
{
    ShmWriter writer;
    bool opened = writer.open(kNX, kNY, kNZ, {"u", "w", "theta"}, kShmName);
    if (!opened)
    {
        WARN("POSIX SHM unavailable — skipping");
        return;
    }

    // Write different patterns to each field
    Field3D field_u(kNX, kNY, kNZ);
    Field3D field_w(kNX, kNY, kNZ);
    Field3D field_theta(kNX, kNY, kNZ);

    for (int i = 0; i < kNX; ++i)
        for (int j = 0; j < kNY; ++j)
            for (int k = 0; k < kNZ; ++k)
            {
                field_u(i, j, k) = static_cast<float>(i);
                field_w(i, j, k) = static_cast<float>(j);
                field_theta(i, j, k) = 280.0f + static_cast<float>(k) * 5.0f
                                      + static_cast<float>(i) * 0.5f;
            }

    writer.write_field(0, field_u);
    writer.write_field(1, field_w);
    writer.write_field(2, field_theta);
    writer.commit(10.0);

    // Each reader attaches to its own field by name
    for (const char* name : {"u", "w", "theta"})
    {
        oglcpp::ShmDataset reader(name, kShmName);
        std::string error;
        REQUIRE(reader.scan(error));
        REQUIRE(reader.field_name() == name);

        oglcpp::VolumeFrame frame;
        REQUIRE(reader.load_frame(0, frame, error));

        const std::size_t voxels = static_cast<std::size_t>(kNX) * kNY * kNZ;
        REQUIRE(frame.normalized.size() == voxels);

        float sum = 0.0f;
        for (std::size_t i = 0; i < voxels; ++i)
        {
            sum += frame.normalized[i];
        }
        REQUIRE(sum > 0.0f);
    }

    // Non-existent field should fail scan
    oglcpp::ShmDataset bad_reader("nonexistent_field", kShmName);
    std::string error;
    REQUIRE_FALSE(bad_reader.scan(error));

    writer.close();
}
