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
#include "core/output/npy_writer.hpp"
#include "core/field/field3d.hpp"
#include "data/export_dataset.hpp"
#include "data/shm_dataset.hpp"

#include <filesystem>
#include <fstream>
#include <limits>
#include <vector>

#if defined(__APPLE__) || defined(__linux__)
#include <sys/wait.h>
#include <unistd.h>
#endif

static constexpr int kNX = 8;
static constexpr int kNY = 8;
static constexpr int kNZ = 8;
static constexpr const char* kShmName = "/tmv_test_e2e";

TEST_CASE("disk viewer crops manifest-v4 storage closures before normalization",
          "[integration][vulkan][manifest]")
{
    const std::filesystem::path root = std::filesystem::temp_directory_path() /
        ("nimbus_vulkan_manifest_crop_" + std::to_string(static_cast<long long>(::getpid())));
    const std::filesystem::path step = root / "step_000000";
    struct Cleanup
    {
        std::filesystem::path path;
        ~Cleanup() { std::error_code error; std::filesystem::remove_all(path, error); }
    } cleanup{root};
    REQUIRE(std::filesystem::create_directories(step));

    constexpr int nx = 6;
    constexpr int ny = 5;
    constexpr int nz = 4;
    std::vector<float> values(static_cast<std::size_t>(nx) * ny * nz, -10000.0f);
    for (int x = 1; x <= 4; ++x)
        for (int y = 1; y <= 3; ++y)
            for (int z = 0; z < nz; ++z)
                values[(static_cast<std::size_t>(x) * ny + y) * nz + z] =
                    static_cast<float>(x * 100 + y * 10 + z);
    values[0] = 10000.0f;
    REQUIRE(npy::write_3d(values.data(), nx, ny, nz, (step / "theta.npy").string()));

    std::ofstream manifest(step / "manifest.json");
    REQUIRE(manifest.good());
    manifest << R"({
      "schema":"nimbus.step","manifest_version":4,
      "grid":{
        "coordinate_system":"cartesian",
        "dimensions":{"x":6,"y":5,"z":4},
        "spacing_m":{"x":400,"y":400,"z":250},
        "native_model_staggering":"c_grid",
        "valid_index_range":{"lower":[1,1,0],"upper":[4,3,3]},
        "halo_width":{"lower":[1,1,0],"upper":[1,1,0]}
      }
    })";
    manifest.close();

    oglcpp::ExportDataset reader(root, "theta");
    std::string error;
    REQUIRE(reader.scan(error));
    CHECK(reader.nx() == 4);
    CHECK(reader.ny() == 3);
    CHECK(reader.nz() == 4);

    oglcpp::VolumeFrame frame;
    REQUIRE(reader.load_frame(0, frame, error));
    CHECK(frame.nx == 4);
    CHECK(frame.ny == 3);
    CHECK(frame.nz == 4);
    CHECK(frame.normalized.size() == 48);
    CHECK(frame.raw_min == Approx(110.0f));
    CHECK(frame.raw_max == Approx(433.0f));
    CHECK(frame.nan_count == 0);
    CHECK(frame.inf_count == 0);
}

TEST_CASE("live viewer crops SHM-v5 storage closures before normalization",
          "[integration][shm][ownership]")
{
    const tmv_shm::ShmPhysicalDomain domain{{1, 1, 0}, {kNX - 2, kNY - 2, kNZ - 1}};
    const tmv_shm::ShmGridSpacing spacing{400.0, 500.0, 250.0};
    ShmWriter writer;
    if (!writer.open(kNX, kNY, kNZ, {"theta"}, domain, spacing, kShmName))
    {
        WARN("POSIX SHM unavailable — skipping SHM ownership test");
        return;
    }

    Field3D field(kNX, kNY, kNZ, -10000.0f);
    field(0, 0, 0) = 10000.0f;
    for (int i = 1; i < kNX - 1; ++i)
        for (int j = 1; j < kNY - 1; ++j)
            for (int k = 0; k < kNZ; ++k)
                field(i, j, k) = static_cast<float>(i * 100 + j * 10 + k);
    REQUIRE(writer.write_field(0, field));
    REQUIRE(writer.commit(1.0));

    oglcpp::ShmDataset reader("theta", kShmName);
    std::string error;
    REQUIRE(reader.scan(error));
    CHECK(reader.nx() == kNX - 2);
    CHECK(reader.ny() == kNY - 2);
    CHECK(reader.nz() == kNZ);
    CHECK(reader.grid_spacing().x_m == Approx(400.0));
    CHECK(reader.grid_spacing().y_m == Approx(500.0));
    CHECK(reader.grid_spacing().z_m == Approx(250.0));

    oglcpp::VolumeFrame frame;
    REQUIRE(reader.load_frame(0, frame, error));
    CHECK(frame.normalized.size() == static_cast<std::size_t>(kNX - 2) *
                                      static_cast<std::size_t>(kNY - 2) * kNZ);
    CHECK(frame.raw_min == Approx(110.0f));
    CHECK(frame.raw_max == Approx(667.0f));
    writer.close();
}

TEST_CASE("SHM moving-frame provenance advances atomically with simulation time",
          "[integration][shm][domain-motion]")
{
    const tmv_shm::ShmPhysicalDomain domain =
        tmv_shm::full_physical_domain(kNX, kNY, kNZ);
    const tmv_shm::ShmGridSpacing spacing{400.0, 500.0, 250.0};
    tmv_shm::ShmHorizontalFrame horizontal_frame;
    horizontal_frame.motion_mode =
        tmv_shm::ShmDomainMotionMode::ConstantVelocity;
    horizontal_frame.grid_origin_t0_earth_m[0] = 1000.0;
    horizontal_frame.grid_origin_t0_earth_m[1] = -2000.0;
    horizontal_frame.grid_origin_earth_m[0] = 1000.0;
    horizontal_frame.grid_origin_earth_m[1] = -2000.0;
    horizontal_frame.frame_velocity_earth_ms[0] = 7.0;
    horizontal_frame.frame_velocity_earth_ms[1] = -3.0;

    ShmWriter writer;
    const std::string generation(32, 'a');
    if (!writer.open(kNX, kNY, kNZ, {"theta"}, domain, spacing,
                     horizontal_frame, kShmName, generation))
    {
        WARN("POSIX SHM unavailable — skipping moving-frame test");
        return;
    }
    Field3D field(kNX, kNY, kNZ, 300.0f);
    REQUIRE(writer.write_field(0, field));
    REQUIRE(writer.commit(20.0));

    oglcpp::ShmDataset reader("theta", kShmName);
    std::string error;
    REQUIRE(reader.scan(error));
    oglcpp::VolumeFrame frame;
    REQUIRE(reader.load_frame(0, frame, error));
    const auto& live = reader.horizontal_frame();
    REQUIRE(live.motion_mode ==
            tmv_shm::ShmDomainMotionMode::ConstantVelocity);
    REQUIRE(live.grid_origin_t0_earth_m[0] == Approx(1000.0));
    REQUIRE(live.grid_origin_t0_earth_m[1] == Approx(-2000.0));
    REQUIRE(live.grid_origin_earth_m[0] == Approx(1140.0));
    REQUIRE(live.grid_origin_earth_m[1] == Approx(-2060.0));
    REQUIRE(live.frame_velocity_earth_ms[0] == Approx(7.0));
    REQUIRE(live.frame_velocity_earth_ms[1] == Approx(-3.0));
    REQUIRE(reader.last_simulation_time() == Approx(20.0));
    REQUIRE(reader.publication_generation() == generation);

    horizontal_frame.grid_origin_t0_earth_m[0] = 5000.0;
    horizontal_frame.grid_origin_t0_earth_m[1] = 7000.0;
    REQUIRE(writer.write_field(0, field));
    REQUIRE(writer.commit(21.0, horizontal_frame));
    REQUIRE(reader.load_frame(0, frame, error));
    REQUIRE(reader.horizontal_frame().grid_origin_earth_m[0] ==
            Approx(5147.0));
    REQUIRE(reader.horizontal_frame().grid_origin_earth_m[1] ==
            Approx(6937.0));
    REQUIRE(reader.publication_generation() == generation);
    writer.close();
}

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

    REQUIRE(writer.write_field(0, field));
    REQUIRE(writer.commit(1.0));

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

TEST_CASE("SHM raw signed transport matches disk normalization", "[integration][shm][e2e]")
{
    ShmWriter writer;
    if (!writer.open(kNX, kNY, kNZ, {"w"}, kShmName))
    {
        WARN("POSIX SHM unavailable — skipping");
        return;
    }

    Field3D field(kNX, kNY, kNZ, 0.0f);
    field(0, 0, 0) = -4.0f;
    field(1, 0, 0) = 0.0f;
    field(2, 0, 0) = 4.0f;
    REQUIRE(writer.write_field(0, field));
    REQUIRE(writer.commit(12.5));

    oglcpp::ShmDataset reader("w", kShmName);
    std::string error;
    REQUIRE(reader.scan(error));
    REQUIRE(reader.is_bipolar());
    REQUIRE(std::string(reader.units()) == "m s-1");

    oglcpp::VolumeFrame live;
    REQUIRE(reader.load_frame(0, live, error));
    REQUIRE(live.raw_min == -4.0f);
    REQUIRE(live.raw_max == 4.0f);
    REQUIRE(live.normalized[0] < 0.5f);
    REQUIRE(live.normalized[1] == Approx(0.5f));
    REQUIRE(live.normalized[2] > 0.5f);
    REQUIRE((reader.last_sequence() & 1U) == 0U);
    REQUIRE(reader.last_simulation_time() == Approx(12.5));

    // Apply the disk reader's exact normalization routine to the raw viewer-
    // layout volume and require byte-for-value parity with the live reader.
    const std::size_t voxels = static_cast<std::size_t>(kNX) * kNY * kNZ;
    std::vector<float> raw(voxels, 0.0f);
    raw[0] = -4.0f;
    raw[1] = 0.0f;
    raw[2] = 4.0f;
    oglcpp::VolumeFrame disk_contract;
    oglcpp::ExportDataset::normalize_volume(
        raw,
        disk_contract.normalized,
        disk_contract.raw_min,
        disk_contract.raw_max,
        disk_contract.norm_low,
        disk_contract.norm_high,
        disk_contract.nan_count,
        disk_contract.inf_count,
        disk_contract.sanitized_nonfinite_count,
        0.0f,
        true);
    REQUIRE(live.normalized == disk_contract.normalized);
    REQUIRE(live.norm_low == disk_contract.norm_low);
    REQUIRE(live.norm_high == disk_contract.norm_high);

    // The signed neutral value remains the renderer's 0.5 midpoint, including
    // the degenerate all-zero case that the old writer encoded as -1 in shader space.
    field.fill(0.0f);
    REQUIRE(writer.write_field(0, field));
    REQUIRE(writer.commit(13.0));
    REQUIRE(reader.load_frame(0, live, error));
    REQUIRE(live.raw_min == 0.0f);
    REQUIRE(live.raw_max == 0.0f);
    for (const float value : live.normalized)
    {
        REQUIRE(value == Approx(0.5f));
    }

    // Non-finites also survive the transport long enough for the shared reader
    // policy to count them and replace them with the signed neutral value.
    field(0, 0, 0) = std::numeric_limits<float>::quiet_NaN();
    field(1, 0, 0) = std::numeric_limits<float>::infinity();
    REQUIRE(writer.write_field(0, field));
    REQUIRE(writer.commit(14.0));
    REQUIRE(reader.load_frame(0, live, error));
    REQUIRE(live.nan_count == 1);
    REQUIRE(live.inf_count == 1);
    REQUIRE(live.sanitized_nonfinite_count == 2);
    REQUIRE(live.normalized[0] == Approx(0.5f));
    REQUIRE(live.normalized[1] == Approx(0.5f));

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
    REQUIRE(writer.write_field(0, field));
    REQUIRE(writer.commit(1.0));

    oglcpp::ShmDataset reader("theta", kShmName);
    std::string error;
    REQUIRE(reader.scan(error));

    oglcpp::VolumeFrame frame;
    REQUIRE(reader.load_frame(0, frame, error));
    REQUIRE_FALSE(reader.has_new_frame());

    // Second commit should be detected as new
    REQUIRE(writer.write_field(0, field));
    REQUIRE(writer.commit(2.0));
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

    REQUIRE(writer.write_field(0, field_u));
    REQUIRE(writer.write_field(1, field_w));
    REQUIRE(writer.write_field(2, field_theta));
    REQUIRE(writer.commit(10.0));

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

TEST_CASE("SHM refuses to publish an incomplete declared bundle", "[integration][shm][e2e]")
{
    ShmWriter writer;
    if (!writer.open(kNX, kNY, kNZ, {"u", "w"}, kShmName))
    {
        WARN("POSIX SHM unavailable — skipping");
        return;
    }

    Field3D field_u(kNX, kNY, kNZ, 3.0f);
    Field3D field_w(kNX, kNY, kNZ, -2.0f);
    oglcpp::ShmDataset reader_u("u", kShmName);
    std::string error;
    REQUIRE_FALSE(reader_u.scan(error));
    REQUIRE(writer.write_field(0, field_u));
    REQUIRE(writer.write_field(1, field_w));
    REQUIRE(writer.commit(4.0));
    REQUIRE(reader_u.scan(error));

    REQUIRE(writer.write_field(0, field_u));
    REQUIRE_FALSE(writer.commit(5.0));
    REQUIRE_FALSE(reader_u.has_new_frame());

    // A later recovery must rewrite every slot; the failed frame's already
    // written u payload is not allowed to combine with a newer w payload.
    REQUIRE(writer.write_field(1, field_w));
    REQUIRE_FALSE(writer.commit(6.0));
    REQUIRE(writer.write_field(0, field_u));
    REQUIRE(writer.write_field(1, field_w));
    REQUIRE(writer.commit(7.0));
    REQUIRE(reader_u.has_new_frame());

    oglcpp::VolumeFrame frame;
    REQUIRE(reader_u.load_frame(0, frame, error));
    REQUIRE(reader_u.last_simulation_time() == Approx(7.0));
    REQUIRE((reader_u.last_sequence() & 1U) == 0U);

    writer.close();
}

TEST_CASE("SHM field readers expose matching composite identity", "[integration][shm][e2e]")
{
    ShmWriter writer;
    if (!writer.open(kNX, kNY, kNZ, {"u", "w"}, kShmName))
    {
        WARN("POSIX SHM unavailable — skipping");
        return;
    }

    Field3D field_u(kNX, kNY, kNZ, 1.0f);
    Field3D field_w(kNX, kNY, kNZ, -1.0f);
    REQUIRE(writer.write_field(0, field_u));
    REQUIRE(writer.write_field(1, field_w));
    REQUIRE(writer.commit(20.0));

    oglcpp::ShmDataset reader_u("u", kShmName);
    oglcpp::ShmDataset reader_w("w", kShmName);
    std::string error;
    REQUIRE(reader_u.scan(error));
    REQUIRE(reader_w.scan(error));

    oglcpp::VolumeFrame frame;
    REQUIRE(reader_u.load_frame(0, frame, error));
    const uint64_t old_sequence = reader_u.last_sequence();
    REQUIRE(reader_u.last_simulation_time() == Approx(20.0));

    field_u.fill(2.0f);
    field_w.fill(-2.0f);
    REQUIRE(writer.write_field(0, field_u));
    REQUIRE(writer.write_field(1, field_w));
    REQUIRE(writer.commit(21.0));
    REQUIRE(reader_w.load_frame(0, frame, error));

    // This deliberately straddled candidate is detectable and must be rejected
    // by the native renderer's composite identity check.
    REQUIRE(reader_w.last_sequence() != old_sequence);
    REQUIRE_FALSE(reader_w.same_snapshot_identity(reader_u));

    REQUIRE(reader_u.load_frame(0, frame, error));
    REQUIRE(reader_u.same_snapshot_identity(reader_w));
    REQUIRE((reader_u.last_sequence() & 1U) == 0U);

    writer.close();
}

TEST_CASE("SHM seqlock rejects torn field and time snapshots", "[integration][shm][e2e]")
{
#if !defined(__APPLE__) && !defined(__linux__)
    WARN("POSIX process concurrency unavailable — skipping");
    return;
#else
    constexpr int nx = 24;
    constexpr int ny = 16;
    constexpr int nz = 12;
    constexpr int generations = 200;

    ShmWriter writer;
    if (!writer.open(nx, ny, nz, {"theta"}, kShmName))
    {
        WARN("POSIX SHM unavailable — skipping");
        return;
    }

    Field3D field(nx, ny, nz, 1.0f);
    REQUIRE(writer.write_field(0, field));
    REQUIRE(writer.commit(1.0));

    oglcpp::ShmDataset reader("theta", kShmName);
    std::string error;
    REQUIRE(reader.scan(error));

    // fork() makes this a true interprocess test rather than relying on the
    // C++ thread memory model for ordinary payload bytes protected by an IPC
    // seqlock. The child inherits the writer's MAP_SHARED mapping.
    const pid_t producer = ::fork();
    REQUIRE(producer >= 0);
    if (producer == 0)
    {
        for (int generation = 2; generation <= generations; ++generation)
        {
            field.fill(static_cast<float>(generation));
            if (!writer.write_field(0, field) ||
                !writer.commit(static_cast<double>(generation)))
            {
                ::_exit(2);
            }
            ::usleep(50);
        }
        ::_exit(0); // Do not run the copied writer destructor/unlink in the child.
    }

    int coherent_reads = 0;
    int producer_status = 0;
    bool producer_done = false;
    while (!producer_done)
    {
        const pid_t wait_result = ::waitpid(producer, &producer_status, WNOHANG);
        REQUIRE(wait_result >= 0);
        producer_done = wait_result == producer;
        if (producer_done)
        {
            break;
        }

        oglcpp::VolumeFrame frame;
        std::string load_error;
        if (!reader.load_frame(0, frame, load_error))
        {
            continue; // A busy odd generation is an expected bounded retry.
        }
        ++coherent_reads;
        REQUIRE(frame.raw_min == frame.raw_max);
        REQUIRE(static_cast<double>(frame.raw_min) == reader.last_simulation_time());
        REQUIRE((reader.last_sequence() & 1U) == 0U);
    }
    REQUIRE(WIFEXITED(producer_status));
    REQUIRE(WEXITSTATUS(producer_status) == 0);

    // Always verify the final publication too, even if the producer completed
    // before this process won a concurrent copy window.
    oglcpp::VolumeFrame final_frame;
    REQUIRE(reader.load_frame(0, final_frame, error));
    REQUIRE(final_frame.raw_min == static_cast<float>(generations));
    REQUIRE(final_frame.raw_max == static_cast<float>(generations));
    REQUIRE(reader.last_simulation_time() == Approx(generations));
    REQUIRE((reader.last_sequence() & 1U) == 0U);
    REQUIRE(coherent_reads > 0);

    writer.close();
#endif
}
