/**
 * @file test_output_writer.cpp
 * @brief Tests for AsyncOutputWriter sync mode and NPY field roundtrip.
 *
 * Verifies:
 *   - Sync mode writes files to disk immediately
 *   - Written NPY files contain exact field values
 *   - Multi-field snapshot produces one file per field
 *   - Statistics (bytes written, snapshot count) are accurate
 */
#include "catch2/catch.hpp"
#include "core/output/output_writer.hpp"
#include "core/output/output_config.hpp"
#include "core/field/field_snapshot.hpp"
#include "core/field/field3d.hpp"

#include <cmath>
#include <filesystem>
#include <fstream>

namespace
{

std::string make_tmp_dir(const std::string& name)
{
    auto dir = std::filesystem::temp_directory_path() / ("tmv_test_writer_" + name);
    std::filesystem::create_directories(dir);
    return dir.string();
}

void cleanup_dir(const std::string& dir)
{
    std::filesystem::remove_all(dir);
}

ExportSnapshot make_snapshot(const std::string& step_dir, int index, int nr, int nth, int nz)
{
    ExportSnapshot snap;
    snap.export_index = index;
    snap.simulation_time_s = index * 10.0;
    snap.step_dir = step_dir;

    // Create a field with known pattern: value = i*1000 + j*10 + k + 0.5
    FieldSnapshotEntry entry;
    entry.name = "test_field";
    entry.dim0 = nr;
    entry.dim1 = nth;
    entry.dim2 = nz;
    entry.is_3d = true;
    entry.data.resize(static_cast<size_t>(nr) * nth * nz);
    for (int i = 0; i < nr; ++i)
        for (int j = 0; j < nth; ++j)
            for (int k = 0; k < nz; ++k)
            {
                size_t idx = i * nth * nz + j * nz + k;
                entry.data[idx] = static_cast<float>(i * 1000 + j * 10 + k) + 0.5f;
            }
    snap.fields.push_back(std::move(entry));
    return snap;
}

} // namespace

TEST_CASE("AsyncOutputWriter sync mode writes files", "[core][output_writer][analytical]")
{
    auto dir = make_tmp_dir("sync");
    auto step_dir = dir + "/step_000000";
    std::filesystem::create_directories(step_dir);

    OutputConfig cfg;
    cfg.format = OutputFormat::npy_3d;
    cfg.async_io = false;

    AsyncOutputWriter writer(cfg);

    auto snap = make_snapshot(step_dir, 0, 4, 6, 8);
    REQUIRE(writer.submit(std::move(snap)));

    // File should exist
    auto npy_path = step_dir + "/test_field.npy";
    REQUIRE(std::filesystem::exists(npy_path));

    // File size: header + 4*6*8*4 = header + 768 bytes
    auto file_size = std::filesystem::file_size(npy_path);
    REQUIRE(file_size > 768);  // data + header overhead
    REQUIRE(file_size < 2000); // but not absurdly large

    REQUIRE(writer.snapshots_written() == 1);
    REQUIRE(writer.total_bytes_written() > 0);
    REQUIRE_FALSE(writer.has_error());

    cleanup_dir(dir);
}

TEST_CASE("AsyncOutputWriter NPY roundtrip preserves exact field values", "[core][output_writer][analytical]")
{
    auto dir = make_tmp_dir("roundtrip");
    auto step_dir = dir + "/step_000000";
    std::filesystem::create_directories(step_dir);

    OutputConfig cfg;
    cfg.format = OutputFormat::npy_3d;
    cfg.async_io = false;

    AsyncOutputWriter writer(cfg);

    const int nr = 4, nth = 6, nz = 8;
    auto snap = make_snapshot(step_dir, 0, nr, nth, nz);

    // Save the original data for comparison
    std::vector<float> original = snap.fields[0].data;

    REQUIRE(writer.submit(std::move(snap)));

    // Read back the NPY file
    auto npy_path = step_dir + "/test_field.npy";
    std::ifstream f(npy_path, std::ios::binary);
    REQUIRE(f.is_open());

    // Skip NPY header (magic + version + header_len + header dict)
    char magic[6];
    f.read(magic, 6);
    REQUIRE(magic[0] == '\x93');

    uint8_t major, minor;
    f.read(reinterpret_cast<char*>(&major), 1);
    f.read(reinterpret_cast<char*>(&minor), 1);

    uint16_t header_len;
    f.read(reinterpret_cast<char*>(&header_len), 2);
    f.seekg(header_len, std::ios::cur);

    // Read data
    std::vector<float> readback(nr * nth * nz);
    f.read(reinterpret_cast<char*>(readback.data()), readback.size() * sizeof(float));
    REQUIRE(f.good());

    // Verify every value matches exactly (float32 roundtrip is lossless)
    for (size_t i = 0; i < original.size(); ++i)
    {
        REQUIRE(readback[i] == original[i]);
    }

    cleanup_dir(dir);
}

TEST_CASE("AsyncOutputWriter multi-field snapshot creates all files", "[core][output_writer][analytical]")
{
    auto dir = make_tmp_dir("multi");
    auto step_dir = dir + "/step_000000";
    std::filesystem::create_directories(step_dir);

    OutputConfig cfg;
    cfg.format = OutputFormat::npy_3d;
    cfg.async_io = false;

    AsyncOutputWriter writer(cfg);

    ExportSnapshot snap;
    snap.export_index = 0;
    snap.simulation_time_s = 0.0;
    snap.step_dir = step_dir;

    // Add 3 fields with different data
    for (const auto& name : {"u", "w", "theta"})
    {
        FieldSnapshotEntry entry;
        entry.name = name;
        entry.dim0 = 4;
        entry.dim1 = 4;
        entry.dim2 = 4;
        entry.is_3d = true;
        entry.data.assign(64, 1.0f);
        snap.fields.push_back(std::move(entry));
    }

    REQUIRE(writer.submit(std::move(snap)));

    // All 3 files should exist
    REQUIRE(std::filesystem::exists(step_dir + "/u.npy"));
    REQUIRE(std::filesystem::exists(step_dir + "/w.npy"));
    REQUIRE(std::filesystem::exists(step_dir + "/theta.npy"));

    REQUIRE(writer.snapshots_written() == 1);

    cleanup_dir(dir);
}

TEST_CASE("AsyncOutputWriter CSV mode writes valid CSV", "[core][output_writer][analytical]")
{
    auto dir = make_tmp_dir("csv");
    auto step_dir = dir + "/step_000000";
    std::filesystem::create_directories(step_dir);

    OutputConfig cfg;
    cfg.format = OutputFormat::csv;
    cfg.async_io = false;

    AsyncOutputWriter writer(cfg);

    ExportSnapshot snap;
    snap.export_index = 0;
    snap.simulation_time_s = 0.0;
    snap.step_dir = step_dir;

    FieldSnapshotEntry entry;
    entry.name = "test_csv";
    entry.dim0 = 2;
    entry.dim1 = 2;
    entry.dim2 = 2;
    entry.is_3d = true;
    entry.data = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 42.0f};
    snap.fields.push_back(std::move(entry));

    REQUIRE(writer.submit(std::move(snap)));

    auto csv_path = step_dir + "/test_csv.csv";
    REQUIRE(std::filesystem::exists(csv_path));

    // Read CSV and verify the non-zero value is present
    std::ifstream f(csv_path);
    std::string content((std::istreambuf_iterator<char>(f)),
                         std::istreambuf_iterator<char>());
    REQUIRE(content.find("42") != std::string::npos);
    REQUIRE(content.find("i,j,k,value") != std::string::npos);

    cleanup_dir(dir);
}

TEST_CASE("AsyncOutputWriter multiple snapshots accumulate statistics", "[core][output_writer][analytical]")
{
    auto dir = make_tmp_dir("multi_snap");

    OutputConfig cfg;
    cfg.format = OutputFormat::npy_3d;
    cfg.async_io = false;

    AsyncOutputWriter writer(cfg);

    for (int s = 0; s < 3; ++s)
    {
        auto step_dir = dir + "/step_" + std::to_string(s);
        std::filesystem::create_directories(step_dir);
        auto snap = make_snapshot(step_dir, s, 4, 4, 4);
        REQUIRE(writer.submit(std::move(snap)));
    }

    REQUIRE(writer.snapshots_written() == 3);
    REQUIRE(writer.total_bytes_written() > 0);

    // Each snapshot writes one 4x4x4 field = 256 bytes of data + header
    // Total bytes should be roughly 3 * (256 + header)
    REQUIRE(writer.total_bytes_written() >= 3 * 256);

    cleanup_dir(dir);
}
