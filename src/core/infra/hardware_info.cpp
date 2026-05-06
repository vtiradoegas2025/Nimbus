/**
 * @file hardware_info.cpp
 * @brief Platform-specific hardware detection implementation.
 *
 * Uses sysctlbyname() on macOS and /proc on Linux.
 * Falls back gracefully when platform APIs are unavailable.
 */

#include "core/infra/hardware_info.hpp"
#include "core/runtime/simulation.hpp"
#include "util/log.hpp"
#include "util/simd_utils.hpp"

#include <thread>
#include <sstream>
#include <cstdio>

#ifdef _OPENMP
#include <omp.h>
#endif

#if defined(__APPLE__)
#include <sys/sysctl.h>
#include <sys/types.h>
#elif defined(__linux__)
#include <fstream>
#endif

namespace
{

// ── macOS helpers ──────────────────────────────────────────────────────────

#if defined(__APPLE__)

template <typename T>
T sysctl_value(const char* name, T fallback)
{
    T value{};
    size_t size = sizeof(value);
    if (sysctlbyname(name, &value, &size, nullptr, 0) == 0)
    {
        return value;
    }
    return fallback;
}

std::string sysctl_string(const char* name)
{
    size_t size = 0;
    if (sysctlbyname(name, nullptr, &size, nullptr, 0) != 0 || size == 0)
    {
        return "";
    }
    std::string result(size, '\0');
    if (sysctlbyname(name, result.data(), &size, nullptr, 0) != 0)
    {
        return "";
    }
    // Remove trailing null
    while (!result.empty() && result.back() == '\0')
    {
        result.pop_back();
    }
    return result;
}

#endif // __APPLE__

// ── Linux helpers ──────────────────────────────────────────────────────────

#if defined(__linux__)

std::string read_first_line(const char* path)
{
    std::ifstream f(path);
    std::string line;
    if (f.is_open() && std::getline(f, line))
    {
        return line;
    }
    return "";
}

uint64_t parse_meminfo_kb(const char* key)
{
    std::ifstream f("/proc/meminfo");
    std::string line;
    const std::string prefix(key);
    while (std::getline(f, line))
    {
        if (line.find(prefix) == 0)
        {
            // Format: "MemTotal:       16384000 kB"
            std::istringstream iss(line.substr(prefix.size()));
            uint64_t value = 0;
            iss >> value;
            return value * 1024; // kB → bytes
        }
    }
    return 0;
}

std::string cpuinfo_field(const char* field_name)
{
    std::ifstream f("/proc/cpuinfo");
    std::string line;
    const std::string prefix(field_name);
    while (std::getline(f, line))
    {
        auto pos = line.find(prefix);
        if (pos == 0)
        {
            auto colon = line.find(':');
            if (colon != std::string::npos)
            {
                std::string value = line.substr(colon + 1);
                // Trim leading whitespace
                size_t start = value.find_first_not_of(" \t");
                if (start != std::string::npos)
                {
                    return value.substr(start);
                }
            }
        }
    }
    return "";
}

size_t read_cache_size(int index)
{
    char path[128];
    std::snprintf(path, sizeof(path),
                  "/sys/devices/system/cpu/cpu0/cache/index%d/size", index);
    std::string val = read_first_line(path);
    if (val.empty()) return 0;

    size_t result = 0;
    std::istringstream iss(val);
    iss >> result;
    // Value may end with 'K' or 'M'
    if (val.back() == 'K' || val.back() == 'k') result *= 1024;
    else if (val.back() == 'M' || val.back() == 'm') result *= 1024 * 1024;
    return result;
}

#endif // __linux__

const char* simd_type_name(simd_utils::SIMDType type)
{
    switch (type)
    {
        case simd_utils::SIMDType::AVX512: return "AVX-512";
        case simd_utils::SIMDType::AVX:    return "AVX";
        case simd_utils::SIMDType::SSE:    return "SSE";
        default:
        {
#if defined(__aarch64__) || defined(__ARM_NEON)
            return "NEON";
#else
            return "scalar";
#endif
        }
    }
}

std::string format_bytes(uint64_t bytes)
{
    if (bytes == 0) return "unknown";
    if (bytes >= uint64_t(1) << 30)
    {
        double gb = static_cast<double>(bytes) / (uint64_t(1) << 30);
        char buf[32];
        std::snprintf(buf, sizeof(buf), "%.1f GB", gb);
        return buf;
    }
    if (bytes >= uint64_t(1) << 20)
    {
        double mb = static_cast<double>(bytes) / (uint64_t(1) << 20);
        char buf[32];
        std::snprintf(buf, sizeof(buf), "%.0f MB", mb);
        return buf;
    }
    if (bytes >= uint64_t(1) << 10)
    {
        double kb = static_cast<double>(bytes) / (uint64_t(1) << 10);
        char buf[32];
        std::snprintf(buf, sizeof(buf), "%.0f KB", kb);
        return buf;
    }
    return std::to_string(bytes) + " B";
}

} // anonymous namespace


HardwareInfo detect_hardware()
{
    HardwareInfo info;

    // ── Logical cores (portable) ───────────────────────────────────────────
    info.logical_cores = static_cast<int>(std::thread::hardware_concurrency());
    if (info.logical_cores <= 0) info.logical_cores = 1;

    // ── OpenMP threads ─────────────────────────────────────────────────────
#ifdef _OPENMP
    info.omp_max_threads = omp_get_max_threads();
#else
    info.omp_max_threads = 1;
#endif

    // ── SIMD ───────────────────────────────────────────────────────────────
    auto simd_type = simd_utils::get_available_simd();
    info.simd_level_name = simd_type_name(simd_type);
    info.simd_width = SIMD_WIDTH;

    // ── Platform-specific detection ────────────────────────────────────────
#if defined(__APPLE__)
    info.physical_cores = sysctl_value<int>("hw.physicalcpu", info.logical_cores);
    info.cpu_model = sysctl_string("machdep.cpu.brand_string");
    if (info.cpu_model.empty())
    {
        // Apple Silicon doesn't populate brand_string; use chip name
        info.cpu_model = sysctl_string("machdep.cpu.brand");
        if (info.cpu_model.empty())
        {
            info.cpu_model = "Apple Silicon";
        }
    }
    info.total_ram_bytes = sysctl_value<uint64_t>("hw.memsize", 0);
    info.l1d_cache_bytes = sysctl_value<uint64_t>("hw.l1dcachesize", 0);
    info.l2_cache_bytes = sysctl_value<uint64_t>("hw.l2cachesize", 0);
    info.l3_cache_bytes = sysctl_value<uint64_t>("hw.l3cachesize", 0);

#elif defined(__linux__)
    info.physical_cores = info.logical_cores; // Approximation without parsing topology
    info.cpu_model = cpuinfo_field("model name");
    if (info.cpu_model.empty()) info.cpu_model = "unknown";
    info.total_ram_bytes = parse_meminfo_kb("MemTotal:");
    // Linux cache indices: 0=L1i, 1=L1d, 2=L2, 3=L3 (typical but not universal)
    info.l1d_cache_bytes = read_cache_size(1);
    info.l2_cache_bytes = read_cache_size(2);
    info.l3_cache_bytes = read_cache_size(3);

#else
    // Fallback for unsupported platforms
    info.physical_cores = info.logical_cores;
    info.cpu_model = "unknown";
#endif

    return info;
}


void log_hardware_info(const HardwareInfo& info)
{
    tmv::log_info("[HARDWARE] CPU: ", info.cpu_model);
    tmv::log_info("[HARDWARE]   Cores: ", info.physical_cores, "p/",
                  info.logical_cores, "t, OpenMP threads: ", info.omp_max_threads);
    tmv::log_info("[HARDWARE]   SIMD: ", info.simd_level_name,
                  " (width=", info.simd_width, ")");

    if (info.total_ram_bytes > 0)
    {
        tmv::log_info("[HARDWARE]   RAM: ", format_bytes(info.total_ram_bytes));
    }

    if (info.l1d_cache_bytes > 0 || info.l2_cache_bytes > 0)
    {
        tmv::log_info("[HARDWARE]   Cache: L1d=", format_bytes(info.l1d_cache_bytes),
                      ", L2=", format_bytes(info.l2_cache_bytes),
                      ", L3=", format_bytes(info.l3_cache_bytes));
    }

    if (info.gpu_available)
    {
        tmv::log_info("[HARDWARE] GPU: ", info.gpu_device_name);
    }
    else
    {
        tmv::log_info("[HARDWARE] GPU: not available (CPU-only mode)");
    }
}


uint64_t estimate_grid_memory_bytes(int nr, int nth, int nz, int num_fields)
{
    return static_cast<uint64_t>(nr) * nth * nz * num_fields * sizeof(float);
}


void check_grid_memory_safety(const HardwareInfo& info, int nr, int nth, int nz)
{
    if (info.total_ram_bytes == 0)
    {
        return; // Cannot check without RAM info
    }

    constexpr int estimated_field_count = 40; // Primary + diagnostic fields
    const uint64_t estimated_bytes = estimate_grid_memory_bytes(nr, nth, nz, estimated_field_count);
    const uint64_t ram_80_pct = info.total_ram_bytes * 80 / 100;

    if (estimated_bytes > ram_80_pct)
    {
        tmv::log_warn("[HARDWARE] Requested grid ", nr, "x", nth, "x", nz,
                      " requires ~", format_bytes(estimated_bytes),
                      " (", estimated_field_count, " fields)");
        tmv::log_warn("[HARDWARE] Available RAM: ", format_bytes(info.total_ram_bytes),
                      " — grid exceeds 80% threshold");

        // Suggest largest safe grid (preserve aspect ratio by scaling uniformly)
        double scale = 1.0;
        while (scale > 0.1)
        {
            int test_nr = static_cast<int>(nr * scale);
            int test_nth = static_cast<int>(nth * scale);
            int test_nz = static_cast<int>(nz * scale);
            if (test_nr < 8) test_nr = 8;
            if (test_nth < 4) test_nth = 4;
            if (test_nz < 8) test_nz = 8;

            uint64_t test_bytes = estimate_grid_memory_bytes(
                test_nr, test_nth, test_nz, estimated_field_count);
            if (test_bytes <= ram_80_pct)
            {
                tmv::log_warn("[HARDWARE] Suggested safe grid: ",
                              test_nr, "x", test_nth, "x", test_nz,
                              " (~", format_bytes(test_bytes), ")");
                break;
            }
            scale -= 0.1;
        }
    }
    else
    {
        tmv::log_debug("[HARDWARE] Grid ", nr, "x", nth, "x", nz,
                       " estimated at ", format_bytes(estimated_bytes),
                       " — fits within available RAM");
    }
}


bool apply_hardware_defaults(const HardwareInfo& info, int& nr, int& nth, int& nz)
{
    if (info.total_ram_bytes == 0)
    {
        return false; // Cannot determine safety without RAM info
    }

    constexpr int estimated_field_count = 40;
    const uint64_t ram_80_pct = info.total_ram_bytes * 80 / 100;
    const uint64_t estimated_bytes = estimate_grid_memory_bytes(nr, nth, nz, estimated_field_count);

    if (estimated_bytes <= ram_80_pct)
    {
        return false; // Grid already fits
    }

    // Grid exceeds safe memory threshold — scale down uniformly
    const int orig_nr = nr;
    const int orig_nth = nth;
    const int orig_nz = nz;

    // Binary-style search for the largest uniform scale factor that fits.
    // Start at 0.9 and step down by 0.1 (same logic as check_grid_memory_safety).
    double scale = 0.9;
    bool found = false;
    while (scale > 0.05)
    {
        int test_nr = static_cast<int>(orig_nr * scale);
        int test_nth = static_cast<int>(orig_nth * scale);
        int test_nz = static_cast<int>(orig_nz * scale);

        // Enforce minimum grid dimensions
        if (test_nr < 8) test_nr = 8;
        if (test_nth < 4) test_nth = 4;
        if (test_nz < 8) test_nz = 8;

        uint64_t test_bytes = estimate_grid_memory_bytes(
            test_nr, test_nth, test_nz, estimated_field_count);
        if (test_bytes <= ram_80_pct)
        {
            nr = test_nr;
            nth = test_nth;
            nz = test_nz;
            found = true;
            break;
        }
        scale -= 0.1;
    }

    if (!found)
    {
        // Even minimum grid; apply floor values
        nr = 8;
        nth = 4;
        nz = 8;
    }

    const uint64_t final_bytes = estimate_grid_memory_bytes(nr, nth, nz, estimated_field_count);
    tmv::log_warn("[HARDWARE] Grid ", orig_nr, "x", orig_nth, "x", orig_nz,
                  " requires ~", format_bytes(estimated_bytes),
                  " but only ", format_bytes(info.total_ram_bytes),
                  " RAM available (80% threshold = ", format_bytes(ram_80_pct), ")");
    tmv::log_warn("[HARDWARE] Auto-scaled to ", nr, "x", nth, "x", nz,
                  " (~", format_bytes(final_bytes), ")");

    return true;
}
