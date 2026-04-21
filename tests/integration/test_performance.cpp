/**
 * @file test_performance.cpp
 * @brief Performance benchmark tests — timing and disk space measurement.
 *
 * These tests don't assert specific timing values (hardware-dependent).
 * Instead they:
 *   1. Measure and REPORT wall-clock time for key operations
 *   2. Verify output disk space matches estimates
 *   3. Provide regression baselines that can be compared across runs
 *
 * Run with: ./bin/test_integration -c "[performance]" for just these tests.
 */
#include "catch2/catch.hpp"
#include "core/field3d.hpp"
#include "core/field/field_pool.hpp"
#include "core/output/output_config.hpp"
#include "core/output/npy_writer.hpp"
#include "core/hardware_info.hpp"
#include "core/simulation.hpp"

#include <chrono>
#include <cstdio>
#include <filesystem>

namespace
{

double measure_ms(std::function<void()> fn)
{
    auto start = std::chrono::high_resolution_clock::now();
    fn();
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count();
}

std::string tmp_dir()
{
    auto dir = std::filesystem::temp_directory_path().string() + "/tmv_perf_test";
    std::filesystem::create_directories(dir);
    return dir;
}

void cleanup_dir(const std::string& dir)
{
    std::filesystem::remove_all(dir);
}

} // namespace

TEST_CASE("Field3D allocation timing", "[performance][core]")
{
    NR = 256; NTH = 256; NZ = 64;
    const int n_allocs = 10;

    double ms = measure_ms([&]() {
        for (int i = 0; i < n_allocs; ++i)
        {
            Field3D f(NR, NTH, NZ);
            // prevent optimization
            f(NR / 2, NTH / 2, NZ / 2) = 1.0f;
        }
    });

    double per_alloc_ms = ms / n_allocs;
    WARN("Field3D(" << NR << "x" << NTH << "x" << NZ << ") allocation: "
         << per_alloc_ms << " ms/alloc");

    // Sanity: allocation should complete (no hang)
    REQUIRE(ms < 60000.0);
}

TEST_CASE("FieldPool vs raw allocation speedup", "[performance][core]")
{
    NR = 128; NTH = 128; NZ = 64;
    const int n_iters = 50;
    auto& pool = FieldPool::instance();
    pool.clear();

    // Warm up pool with one buffer
    pool.release(pool.acquire(NR, NTH, NZ));

    double pool_ms = measure_ms([&]() {
        for (int i = 0; i < n_iters; ++i)
        {
            Field3D f = pool.acquire(NR, NTH, NZ);
            f(0, 0, 0) = 1.0f;
            pool.release(std::move(f));
        }
    });

    double raw_ms = measure_ms([&]() {
        for (int i = 0; i < n_iters; ++i)
        {
            Field3D f(NR, NTH, NZ);
            f(0, 0, 0) = 1.0f;
        }
    });

    double pool_per = pool_ms / n_iters;
    double raw_per = raw_ms / n_iters;
    double speedup = raw_per / pool_per;

    WARN("Pool acquire/release: " << pool_per << " ms vs raw alloc: "
         << raw_per << " ms (speedup: " << speedup << "x)");

    // Pool should be at least as fast as raw allocation (typically much faster)
    // We only assert it doesn't crash — timing depends on hardware
    REQUIRE(pool_ms < 60000.0);

    pool.clear();
}

TEST_CASE("NPY write throughput", "[performance][io]")
{
    NR = 128; NTH = 128; NZ = 64;
    auto dir = tmp_dir();
    auto path = dir + "/perf_field.npy";

    Field3D field(NR, NTH, NZ, 1.0f);
    const int n_writes = 5;

    double ms = measure_ms([&]() {
        for (int i = 0; i < n_writes; ++i)
            npy::write_field3d(field, path);
    });

    double bytes_per_write = static_cast<double>(NR) * NTH * NZ * sizeof(float);
    double total_mb = (bytes_per_write * n_writes) / (1024.0 * 1024.0);
    double throughput_mbs = total_mb / (ms / 1000.0);

    WARN("NPY write: " << total_mb << " MB in " << ms << " ms ("
         << throughput_mbs << " MB/s)");

    // Verify file exists with expected size
    auto file_size = std::filesystem::file_size(path);
    // NPY file = header + data. Data = NR*NTH*NZ*4 bytes
    REQUIRE(file_size > static_cast<uintmax_t>(NR) * NTH * NZ * 4);

    cleanup_dir(dir);
}

TEST_CASE("Disk budget estimation accuracy", "[performance][io]")
{
    OutputConfig cfg;
    cfg.format = OutputFormat::npy_3d;
    cfg.preset = FieldPreset::minimal;
    resolve_output_fields(cfg);

    int nr = 64, nth = 64, nz = 32;
    int duration_s = 3600;
    int write_every_s = 60;

    estimate_disk_budget(cfg, nr, nth, nz, duration_s, write_every_s);

    // Analytical: each export = num_fields * nr * nth * nz * 4 bytes (+ small header)
    size_t num_fields = cfg.resolved_fields.size();
    size_t bytes_per_field = static_cast<size_t>(nr) * nth * nz * sizeof(float);
    size_t expected_per_export = num_fields * bytes_per_field;

    // The estimate should be within 2x of the analytical calculation
    // (header overhead, compression estimates may differ)
    REQUIRE(cfg.estimated_bytes_per_export > 0);
    double ratio = static_cast<double>(cfg.estimated_bytes_per_export) /
                   static_cast<double>(expected_per_export);
    REQUIRE(ratio > 0.5);
    REQUIRE(ratio < 2.0);

    // Total should be per_export * num_exports
    int num_exports = duration_s / write_every_s;
    size_t expected_total = cfg.estimated_bytes_per_export * num_exports;
    REQUIRE(cfg.estimated_total_bytes == Approx(static_cast<double>(expected_total)).epsilon(0.01));

    WARN("Disk budget: " << num_fields << " fields, "
         << cfg.estimated_bytes_per_export / 1024 << " KB/export, "
         << cfg.estimated_total_bytes / (1024 * 1024) << " MB total for "
         << num_exports << " exports");
}

TEST_CASE("Hardware detection reports plausible values", "[performance][hardware]")
{
    HardwareInfo hw = detect_hardware();

    WARN("CPU: " << hw.cpu_model);
    WARN("Cores: " << hw.physical_cores << " physical, "
         << hw.logical_cores << " logical");
    WARN("RAM: " << hw.total_ram_bytes / (1024 * 1024 * 1024) << " GB");
    WARN("SIMD: " << hw.simd_level_name << " (width " << hw.simd_width << ")");
    WARN("L1d: " << hw.l1d_cache_bytes / 1024 << " KB, L2: "
         << hw.l2_cache_bytes / 1024 << " KB, L3: "
         << hw.l3_cache_bytes / (1024 * 1024) << " MB");

    REQUIRE(hw.physical_cores >= 1);
    REQUIRE(hw.total_ram_bytes > 1'000'000'000ULL);
}

TEST_CASE("Grid memory estimation scales correctly", "[performance][hardware]")
{
    // Student grid
    uint64_t student = estimate_grid_memory_bytes(64, 64, 32, 40);
    // Research grid
    uint64_t research = estimate_grid_memory_bytes(256, 256, 64, 40);
    // Production grid
    uint64_t production = estimate_grid_memory_bytes(512, 256, 128, 40);

    WARN("Grid memory: student=" << student / (1024 * 1024) << " MB"
         << ", research=" << research / (1024 * 1024) << " MB"
         << ", production=" << production / (1024 * 1024) << " MB");

    // Each should be larger than the last
    REQUIRE(research > student);
    REQUIRE(production > research);

    // Student grid should fit in 1 GB
    REQUIRE(student < 1'000'000'000ULL);
}
