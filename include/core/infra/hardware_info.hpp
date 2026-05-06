#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

/**
 * @file hardware_info.hpp
 * @brief Runtime hardware detection for adaptive configuration and diagnostics.
 *
 * Detects CPU capabilities (cores, SIMD, cache), available memory, and GPU
 * presence at startup. Results are logged and made available to subsystems
 * that need to adapt to the host machine (grid sizing, thread pools, etc.).
 */

struct HardwareInfo
{
    // CPU
    int physical_cores = 0;
    int logical_cores = 0;
    int omp_max_threads = 1;
    std::string cpu_model = "unknown";
    std::string simd_level_name = "scalar";
    int simd_width = 1;

    // Memory (bytes)
    uint64_t total_ram_bytes = 0;

    // Cache (bytes; 0 = unavailable)
    size_t l1d_cache_bytes = 0;
    size_t l2_cache_bytes = 0;
    size_t l3_cache_bytes = 0;

    // GPU (populated by backend after init)
    bool gpu_available = false;
    std::string gpu_device_name = "none";
};

/**
 * @brief Detects CPU, memory, and cache characteristics of the host machine.
 *
 * GPU fields are left at defaults — call ComputeBackend::populate_hardware_info()
 * after backend initialization to fill them.
 */
HardwareInfo detect_hardware();

/**
 * @brief Logs all detected hardware characteristics using tmv::log_info.
 */
void log_hardware_info(const HardwareInfo& info);

/**
 * @brief Estimates total memory required for a given grid configuration.
 * @param nr       Radial grid points.
 * @param nth      Azimuthal grid points.
 * @param nz       Vertical grid points.
 * @param num_fields Number of Field3D arrays (primary + diagnostic).
 * @return Estimated bytes required.
 */
uint64_t estimate_grid_memory_bytes(int nr, int nth, int nz, int num_fields = 40);

/**
 * @brief Logs a memory warning if estimated usage exceeds available RAM.
 */
void check_grid_memory_safety(const HardwareInfo& info, int nr, int nth, int nz);

/**
 * @brief Auto-scales grid dimensions to fit within 80% of available RAM.
 *
 * Scales uniformly (preserving aspect ratio) until estimated memory for 40 fields
 * fits within 80% of detected RAM. Enforces floor bounds: NR >= 8, NTH >= 4, NZ >= 8.
 * If RAM is unknown (0), leaves grid unchanged.
 *
 * @param info      Detected hardware characteristics.
 * @param[in,out] nr  Radial grid points (modified if scaling needed).
 * @param[in,out] nth Azimuthal grid points (modified if scaling needed).
 * @param[in,out] nz  Vertical grid points (modified if scaling needed).
 * @return true if grid was modified, false if it already fits or RAM is unknown.
 */
bool apply_hardware_defaults(const HardwareInfo& info, int& nr, int& nth, int& nz);
