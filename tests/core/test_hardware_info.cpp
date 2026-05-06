/**
 * @file test_hardware_info.cpp
 * @brief Unit tests for hardware detection and memory estimation.
 */
#include "catch2/catch.hpp"
#include "core/infra/hardware_info.hpp"

TEST_CASE("detect_hardware returns valid CPU info", "[core][hardware]")
{
    HardwareInfo hw = detect_hardware();

    REQUIRE(hw.physical_cores > 0);
    REQUIRE(hw.logical_cores >= hw.physical_cores);
    REQUIRE(hw.omp_max_threads >= 1);
    REQUIRE_FALSE(hw.cpu_model.empty());
    REQUIRE(hw.cpu_model != "unknown");
}

TEST_CASE("detect_hardware returns valid memory info", "[core][hardware]")
{
    HardwareInfo hw = detect_hardware();
    // Any real machine has at least 1 GB
    REQUIRE(hw.total_ram_bytes > 1'000'000'000ULL);
}

TEST_CASE("detect_hardware returns SIMD info", "[core][hardware]")
{
    HardwareInfo hw = detect_hardware();
    REQUIRE_FALSE(hw.simd_level_name.empty());
    REQUIRE(hw.simd_width >= 1);
}

TEST_CASE("estimate_grid_memory_bytes scales with grid size", "[core][hardware]")
{
    uint64_t small = estimate_grid_memory_bytes(64, 64, 32, 40);
    uint64_t large = estimate_grid_memory_bytes(256, 256, 128, 40);

    REQUIRE(small > 0);
    REQUIRE(large > small);

    // 256x256x128 is 128x the voxels of 64x64x32; memory should scale similarly
    double ratio = static_cast<double>(large) / static_cast<double>(small);
    REQUIRE(ratio > 50.0);
    REQUIRE(ratio < 200.0);
}

TEST_CASE("estimate_grid_memory_bytes with zero fields", "[core][hardware]")
{
    uint64_t bytes = estimate_grid_memory_bytes(64, 64, 32, 0);
    REQUIRE(bytes == 0);
}

TEST_CASE("check_grid_memory_safety does not crash", "[core][hardware]")
{
    HardwareInfo hw = detect_hardware();
    // Should not throw or crash, just logs warnings
    REQUIRE_NOTHROW(check_grid_memory_safety(hw, 64, 64, 32));
    REQUIRE_NOTHROW(check_grid_memory_safety(hw, 512, 512, 256));
}

// --- apply_hardware_defaults tests ---

TEST_CASE("apply_hardware_defaults does not modify grid that fits in RAM",
          "[core][hardware][auto-config]")
{
    HardwareInfo hw;
    hw.total_ram_bytes = 48ULL * 1024 * 1024 * 1024; // 48 GB

    // Small grid easily fits in 48 GB
    int nr = 64, nth = 64, nz = 32;
    bool modified = apply_hardware_defaults(hw, nr, nth, nz);

    REQUIRE_FALSE(modified);
    REQUIRE(nr == 64);
    REQUIRE(nth == 64);
    REQUIRE(nz == 32);
}

TEST_CASE("apply_hardware_defaults scales down grid exceeding RAM",
          "[core][hardware][auto-config]")
{
    HardwareInfo hw;
    hw.total_ram_bytes = 8ULL * 1024 * 1024 * 1024; // 8 GB (student laptop)

    // 256x256x128 at 40 fields = 256*256*128*40*4 = 1,342,177,280 bytes = ~1.25 GB
    // 80% of 8 GB = 6.4 GB — this fits, so use a larger grid that doesn't fit
    // 512x512x256 at 40 fields = 512*512*256*40*4 = 10,737,418,240 = ~10 GB → exceeds 6.4 GB
    int nr = 512, nth = 512, nz = 256;
    bool modified = apply_hardware_defaults(hw, nr, nth, nz);

    REQUIRE(modified);
    // Scaled grid must fit within 80% of 8 GB
    uint64_t scaled_bytes = estimate_grid_memory_bytes(nr, nth, nz, 40);
    uint64_t ram_80_pct = hw.total_ram_bytes * 80 / 100;
    REQUIRE(scaled_bytes <= ram_80_pct);
}

TEST_CASE("apply_hardware_defaults preserves positive dimensions",
          "[core][hardware][auto-config]")
{
    HardwareInfo hw;
    hw.total_ram_bytes = 2ULL * 1024 * 1024 * 1024; // 2 GB (very constrained)

    int nr = 512, nth = 256, nz = 128;
    apply_hardware_defaults(hw, nr, nth, nz);

    // All dimensions must remain positive and above minimum floors
    REQUIRE(nr >= 8);
    REQUIRE(nth >= 4);
    REQUIRE(nz >= 8);
}

TEST_CASE("apply_hardware_defaults preserves aspect ratio approximately",
          "[core][hardware][auto-config]")
{
    HardwareInfo hw;
    hw.total_ram_bytes = 4ULL * 1024 * 1024 * 1024; // 4 GB

    int nr = 512, nth = 256, nz = 128;
    double original_ratio_rn = static_cast<double>(nr) / nth;
    double original_ratio_rz = static_cast<double>(nr) / nz;

    apply_hardware_defaults(hw, nr, nth, nz);

    // Aspect ratio should be approximately preserved (within floor constraints)
    if (nr > 8 && nth > 4 && nz > 8)
    {
        double scaled_ratio_rn = static_cast<double>(nr) / nth;
        double scaled_ratio_rz = static_cast<double>(nr) / nz;
        // Allow ±20% tolerance due to integer truncation
        REQUIRE(scaled_ratio_rn == Approx(original_ratio_rn).epsilon(0.2));
        REQUIRE(scaled_ratio_rz == Approx(original_ratio_rz).epsilon(0.2));
    }
}

TEST_CASE("apply_hardware_defaults handles zero RAM gracefully",
          "[core][hardware][auto-config]")
{
    HardwareInfo hw;
    hw.total_ram_bytes = 0; // Unknown RAM

    int nr = 256, nth = 256, nz = 128;
    int orig_nr = nr, orig_nth = nth, orig_nz = nz;
    bool modified = apply_hardware_defaults(hw, nr, nth, nz);

    // Cannot determine safety without RAM info — leave grid unchanged
    REQUIRE_FALSE(modified);
    REQUIRE(nr == orig_nr);
    REQUIRE(nth == orig_nth);
    REQUIRE(nz == orig_nz);
}

TEST_CASE("apply_hardware_defaults with real hardware does not crash",
          "[core][hardware][auto-config]")
{
    HardwareInfo hw = detect_hardware();
    int nr = 64, nth = 64, nz = 32;
    REQUIRE_NOTHROW(apply_hardware_defaults(hw, nr, nth, nz));
}
