#pragma once

#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <string>

namespace hardware_topology
{

/** Portable projection of one physical core's scheduler-visible topology. */
struct CoreObservation
{
    int          logical_processors = 0;
    std::uint8_t efficiency_class   = 0;
};

/** Logical-processor totals derived from heterogeneous physical core classes. */
struct CoreClassSummary
{
    int logical_processors     = 0;
    int performance_processors = 0;
    int efficiency_processors  = 0;
    int distinct_classes       = 0;
};

/**
 * @brief Summarizes heterogeneous core classes without platform API types.
 *
 * Windows defines larger EfficiencyClass values as faster, less efficient
 * cores. A single class is treated as homogeneous and therefore produces no
 * performance/efficiency split.
 */
CoreClassSummary summarize_core_classes(const CoreObservation* observations,
                                        std::size_t            count) noexcept;

// Zero means the kernel records do not provide complete, unambiguous topology.
int linux_physical_core_count(std::istream& cpuinfo);

} // namespace hardware_topology

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
    int         physical_cores   = 0;
    int         logical_cores    = 0;
    int         processor_groups = 1;
    // Highest-performance core class when the operating system exposes
    // heterogeneous topology; values count logical execution contexts (the
    // quantity OpenMP consumes), and zero means unavailable or homogeneous.
    int         performance_cores = 0;
    int         efficiency_cores  = 0;
    int         omp_max_threads   = 1;
    bool        omp_dynamic       = false;
    std::string omp_proc_bind     = "unavailable";
    int         omp_num_places    = 0;
    std::string cpu_model         = "unknown";
    std::string simd_level_name   = "scalar";
    int         simd_width        = 1;

    // Memory (bytes)
    uint64_t total_ram_bytes = 0;

    // Cache (bytes; 0 = unavailable)
    size_t l1d_cache_bytes = 0;
    size_t l2_cache_bytes  = 0;
    size_t l3_cache_bytes  = 0;

    // GPU (populated by backend after init)
    bool        gpu_available   = false;
    std::string gpu_device_name = "none";
};

/**
 * @brief Result of a non-mutating grid-memory preflight.
 *
 * The estimate is expressed in full-volume float-field equivalents. It is a
 * planning signal, not an allocator admission guarantee: allocator overhead
 * and the UI/OS can add to it. Runtime configuration adds the active
 * accelerator's bounded staging/shared residency before applying this host
 * budget.
 */
struct GridMemoryPreflight
{
    uint64_t estimated_bytes        = 0;
    uint64_t field_storage_bytes    = 0;
    uint64_t fixed_overhead_bytes   = 0;
    uint64_t warning_budget_bytes   = 0;
    int      field_equivalents      = 0;
    bool     ram_known              = false;
    bool     exceeds_warning_budget = false;
};

inline bool memory_admission_should_reject(const GridMemoryPreflight& preflight, bool reject_policy) noexcept
{
    return reject_policy && preflight.ram_known && preflight.exceeds_warning_budget;
}

/**
 * @brief Detects CPU, memory, and cache characteristics of the host machine.
 *
 * GPU fields are left at defaults — call ComputeBackend::populate_hardware_info()
 * after backend initialization to fill them.
 */
HardwareInfo detect_hardware();

/**
 * @brief Returns a topology-aware OpenMP default, or zero to leave the
 * runtime's portable default unchanged.
 *
 * Prefer a detected performance class, otherwise physical cores on SMT hosts.
 * Explicit runtime/CLI worker-count requests take priority over this default.
 */
int topology_aware_openmp_default(const HardwareInfo& info) noexcept;

/** Return this process's cumulative peak resident set size in normalized bytes. */
uint64_t process_peak_rss_bytes();

struct ProcessMemorySample
{
    std::uint64_t current_rss_bytes       = 0;
    std::uint64_t lifetime_peak_rss_bytes = 0;
    bool          current_rss_available  = false;
    bool          lifetime_peak_available = false;
    const char*   current_rss_method     = "unsupported";
};

using ProcessMemorySampler = ProcessMemorySample (*)() noexcept;

ProcessMemorySample sample_process_memory() noexcept;
bool process_memory_trace_enabled(const char* value) noexcept;

// Rejected or incomplete proc data must leave the caller's output untouched.
bool parse_linux_smaps_rollup_rss(const char* data, std::size_t size,
                                 std::uint64_t& bytes) noexcept;

void write_process_memory_trace(std::ostream& stream, bool enabled,
                                const char* scope, const char* phase,
                                const char* operation = nullptr,
                                std::size_t ordinal = 0,
                                std::uint64_t payload_bytes = 0,
                                ProcessMemorySampler sampler = sample_process_memory) noexcept;

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
uint64_t estimate_grid_memory_bytes(int nr, int nth, int nz, int num_fields);

/**
 * @brief Computes a non-mutating grid-memory preflight.
 * @param warning_budget_percent Percentage of physical RAM used as the warning
 *        threshold. Values outside [1, 100] are clamped to that range.
 */
GridMemoryPreflight assess_grid_memory(const HardwareInfo& info,
                                       int                 nr,
                                       int                 nth,
                                       int                 nz,
                                       int                 num_fields,
                                       uint64_t            fixed_overhead_bytes   = 0,
                                       int                 warning_budget_percent = 80);

/**
 * @brief Logs a memory warning if estimated usage exceeds the planning budget.
 *
 * This function never changes grid dimensions. A configuration-requested grid
 * is scientific input; the caller must either run it unchanged or reject it
 * explicitly. The optional profile label should identify the selected schemes
 * and output staging represented by @p num_fields.
 */
void check_grid_memory_safety(const HardwareInfo& info,
                              int                 nr,
                              int                 nth,
                              int                 nz,
                              int                 num_fields           = 77,
                              const std::string&  profile_label        = {},
                              uint64_t            fixed_overhead_bytes = 0);
