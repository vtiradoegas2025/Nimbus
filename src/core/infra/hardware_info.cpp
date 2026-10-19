/**
 * @file hardware_info.cpp
 * @brief Platform-specific hardware detection implementation.
 *
 * Uses sysctlbyname() on macOS, /proc on Linux, and Win32 topology/memory APIs
 * on Windows.
 * Falls back gracefully when platform APIs are unavailable.
 */

#include "core/infra/hardware_info.hpp"
#include "util/log.hpp"
#include "util/simd_utils.hpp"

#include <algorithm>
#include <array>
#include <charconv>
#include <cerrno>
#include <cstdlib>
#include <thread>
#include <sstream>
#include <cstdio>
#include <cstring>
#include <limits>
#include <ostream>
#include <string_view>
#include <set>
#include <vector>

#if defined(__APPLE__) || defined(__linux__)
#include <sys/resource.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

#if defined(__APPLE__)
#include <mach/mach.h>
#include <sys/sysctl.h>
#include <sys/types.h>
#elif defined(__linux__)
#include <fstream>
#elif defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <psapi.h>
#endif

namespace
{

#ifdef _OPENMP
const char* openmp_proc_bind_name(int value)
{
    // Values 0..4 are the portable OpenMP proc-bind enumeration.  Use the
    // integer representation so this remains source-compatible with runtimes
    // that still spell value 2 as `master` rather than `primary`.
    switch (value)
    {
    case 0:
        return "false";
    case 1:
        return "true";
    case 2:
        return "primary";
    case 3:
        return "close";
    case 4:
        return "spread";
    default:
        return "unknown";
    }
}
#endif

// ── macOS helpers ──────────────────────────────────────────────────────────

#if defined(__APPLE__)

template <typename T>
T sysctl_value(const char* name, T fallback)
{
    T      value{};
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
    std::string   line;
    if (f.is_open() && std::getline(f, line))
    {
        return line;
    }
    return "";
}

uint64_t parse_meminfo_kb(const char* key)
{
    std::ifstream     f("/proc/meminfo");
    std::string       line;
    const std::string prefix(key);
    while (std::getline(f, line))
    {
        if (line.find(prefix) == 0)
        {
            // Format: "MemTotal:       16384000 kB"
            std::istringstream iss(line.substr(prefix.size()));
            uint64_t           value = 0;
            iss >> value;
            return value * 1024; // kB → bytes
        }
    }
    return 0;
}

std::string cpuinfo_field(const char* field_name)
{
    std::ifstream     f("/proc/cpuinfo");
    std::string       line;
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
    std::snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu0/cache/index%d/size", index);
    std::string val = read_first_line(path);
    if (val.empty())
        return 0;

    size_t             result = 0;
    std::istringstream iss(val);
    iss >> result;
    // Value may end with 'K' or 'M'
    if (val.back() == 'K' || val.back() == 'k')
        result *= 1024;
    else if (val.back() == 'M' || val.back() == 'm')
        result *= 1024 * 1024;
    return result;
}

size_t read_data_cache_size(int level)
{
    size_t result = 0;
    for (int index = 0; index < 16; ++index)
    {
        const std::string prefix = "/sys/devices/system/cpu/cpu0/cache/index" + std::to_string(index);
        if (read_first_line((prefix + "/level").c_str()) != std::to_string(level)) continue;
        const auto type = read_first_line((prefix + "/type").c_str());
        if (type == "Data" || type == "Unified") result = std::max(result, read_cache_size(index));
    }
    return result;
}

#endif // __linux__

#if defined(_WIN32)

int affinity_population_count(KAFFINITY mask) noexcept
{
    int count = 0;
    while (mask != 0)
    {
        mask &= mask - 1;
        ++count;
    }
    return count;
}

std::string windows_cpu_model()
{
    char  value[256] = {};
    DWORD bytes = sizeof(value);
    const LSTATUS status = RegGetValueA(
        HKEY_LOCAL_MACHINE,
        "HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0",
        "ProcessorNameString", RRF_RT_REG_SZ, nullptr, value, &bytes);
    if (status != ERROR_SUCCESS)
        return "unknown";
    std::string model(value);
    const auto first = model.find_first_not_of(" \t");
    const auto last = model.find_last_not_of(" \t");
    return first == std::string::npos ? "unknown" : model.substr(first, last - first + 1);
}

void populate_windows_topology(HardwareInfo& info)
{
    const DWORD active_processors = GetActiveProcessorCount(ALL_PROCESSOR_GROUPS);
    if (active_processors > 0)
        info.logical_cores = static_cast<int>(active_processors);
    const WORD active_groups = GetActiveProcessorGroupCount();
    if (active_groups > 0)
        info.processor_groups = static_cast<int>(active_groups);

    DWORD bytes = 0;
    if (GetLogicalProcessorInformationEx(RelationAll, nullptr, &bytes) != FALSE
        || GetLastError() != ERROR_INSUFFICIENT_BUFFER
        || bytes == 0)
        return;

    std::vector<unsigned char> storage;
    bool                       populated = false;
    for (int attempt = 0; attempt < 2 && !populated; ++attempt)
    {
        storage.resize(bytes);
        DWORD populated_bytes = bytes;
        auto* records = reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(
            storage.data());
        if (GetLogicalProcessorInformationEx(RelationAll, records,
                                             &populated_bytes) != FALSE)
        {
            bytes     = populated_bytes;
            populated = true;
        }
        else if (GetLastError() == ERROR_INSUFFICIENT_BUFFER
                 && populated_bytes > bytes)
        {
            bytes = populated_bytes;
        }
        else
        {
            return;
        }
    }
    if (!populated)
        return;

    DWORD offset                   = 0;
    int   physical_cores           = 0;
    int   core_logical_processors  = 0;
    int   group_logical_processors = 0;
    int   processor_groups         = 0;
    std::vector<hardware_topology::CoreObservation> core_observations;
    while (offset < bytes)
    {
        const auto* record = reinterpret_cast<const SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(
            storage.data() + offset);
        if (record->Size == 0 || offset + record->Size > bytes)
            break;
        if (record->Relationship == RelationProcessorCore)
        {
            ++physical_cores;
            const auto& processor = record->Processor;
            int         logical_processors = 0;
            for (WORD group = 0; group < processor.GroupCount; ++group)
            {
                logical_processors += affinity_population_count(
                    processor.GroupMask[group].Mask);
            }
            if (logical_processors <= 0)
                logical_processors = 1;
            core_logical_processors += logical_processors;
            core_observations.push_back({
                logical_processors,
                static_cast<std::uint8_t>(processor.EfficiencyClass)});
        }
        else if (record->Relationship == RelationCache)
        {
            const auto& cache = record->Cache;
            if (cache.Level == 1 &&
                (cache.Type == CacheData || cache.Type == CacheUnified))
                info.l1d_cache_bytes = std::max<std::size_t>(info.l1d_cache_bytes, cache.CacheSize);
            else if (cache.Level == 2)
                info.l2_cache_bytes = std::max<std::size_t>(info.l2_cache_bytes, cache.CacheSize);
            else if (cache.Level == 3)
                info.l3_cache_bytes = std::max<std::size_t>(info.l3_cache_bytes, cache.CacheSize);
        }
        else if (record->Relationship == RelationGroup)
        {
            const auto& groups = record->Group;
            processor_groups = std::max<int>(processor_groups,
                                             groups.ActiveGroupCount);
            for (WORD group = 0; group < groups.ActiveGroupCount; ++group)
            {
                group_logical_processors +=
                    static_cast<int>(groups.GroupInfo[group].ActiveProcessorCount);
            }
        }
        offset += record->Size;
    }
    if (physical_cores > 0)
        info.physical_cores = physical_cores;
    if (core_logical_processors > 0)
        info.logical_cores = core_logical_processors;
    else if (group_logical_processors > 0)
        info.logical_cores = group_logical_processors;

    if (processor_groups <= 0)
        processor_groups = static_cast<int>(GetActiveProcessorGroupCount());
    if (processor_groups > 0)
        info.processor_groups = processor_groups;

    const hardware_topology::CoreClassSummary classes =
        hardware_topology::summarize_core_classes(
            core_observations.data(), core_observations.size());
    if (classes.distinct_classes > 1
        && classes.logical_processors == core_logical_processors)
    {
        info.performance_cores = classes.performance_processors;
        info.efficiency_cores  = classes.efficiency_processors;
    }
}

#endif

const char* simd_type_name(simd_utils::SIMDType type)
{
    switch (type)
    {
    case simd_utils::SIMDType::AVX512:
        return "AVX-512";
    case simd_utils::SIMDType::AVX:
        return "AVX";
    case simd_utils::SIMDType::SSE:
        return "SSE";
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
    if (bytes == 0)
        return "unknown";
    if (bytes >= uint64_t(1) << 30)
    {
        double gb = static_cast<double>(bytes) / (uint64_t(1) << 30);
        char   buf[32];
        std::snprintf(buf, sizeof(buf), "%.1f GB", gb);
        return buf;
    }
    if (bytes >= uint64_t(1) << 20)
    {
        double mb = static_cast<double>(bytes) / (uint64_t(1) << 20);
        char   buf[32];
        std::snprintf(buf, sizeof(buf), "%.0f MB", mb);
        return buf;
    }
    if (bytes >= uint64_t(1) << 10)
    {
        double kb = static_cast<double>(bytes) / (uint64_t(1) << 10);
        char   buf[32];
        std::snprintf(buf, sizeof(buf), "%.0f KB", kb);
        return buf;
    }
    return std::to_string(bytes) + " B";
}

} // anonymous namespace

hardware_topology::CoreClassSummary hardware_topology::summarize_core_classes(
    const CoreObservation* observations, std::size_t count) noexcept
{
    CoreClassSummary     summary;
    std::array<int, 256> processors_by_class{};

    if (observations == nullptr)
        return summary;

    for (std::size_t index = 0; index < count; ++index)
    {
        const CoreObservation& observation = observations[index];
        if (observation.logical_processors <= 0)
            continue;
        int& class_total = processors_by_class[observation.efficiency_class];
        if (class_total == 0)
            ++summary.distinct_classes;
        class_total += observation.logical_processors;
        summary.logical_processors += observation.logical_processors;
    }

    if (summary.distinct_classes <= 1)
        return summary;

    // Win32's EfficiencyClass increases with performance (and decreases with
    // energy efficiency), despite what the member name might suggest.
    for (std::size_t class_index = processors_by_class.size();
         class_index-- > 0;)
    {
        if (processors_by_class[class_index] > 0)
        {
            summary.performance_processors = processors_by_class[class_index];
            summary.efficiency_processors =
                summary.logical_processors - summary.performance_processors;
            break;
        }
    }
    return summary;
}

int hardware_topology::linux_physical_core_count(std::istream& cpuinfo)
{
    std::set<int> processors;
    std::set<std::pair<int, int>> cores;
    int processor = -1, package = -1, core = -1;
    const auto commit = [&]()
    {
        if (processor < 0 && package < 0 && core < 0) return true;
        if (processor < 0 || package < 0 || core < 0 || !processors.insert(processor).second) return false;
        cores.emplace(package, core);
        processor = package = core = -1;
        return true;
    };
    const auto trim = [](std::string_view value)
    {
        const auto first = value.find_first_not_of(" \t\r");
        return first == std::string_view::npos ? std::string_view{} :
            value.substr(first, value.find_last_not_of(" \t\r") - first + 1);
    };
    std::string line;
    while (std::getline(cpuinfo, line))
    {
        if (trim(line).empty())
        {
            if (!commit()) return 0;
            continue;
        }
        const auto colon = line.find(':');
        if (colon == std::string::npos) continue;
        const auto key = trim(std::string_view(line).substr(0, colon));
        int* target = key == "processor" ? &processor : key == "physical id" ? &package : key == "core id" ? &core : nullptr;
        if (target == nullptr) continue;
        if (*target >= 0) return 0;
        const auto value = trim(std::string_view(line).substr(colon + 1));
        if (value.empty()) return 0;
        int parsed = -1;
        const auto converted = std::from_chars(value.data(), value.data() + value.size(), parsed);
        if (converted.ec != std::errc{} || converted.ptr != value.data() + value.size() || parsed < 0) return 0;
        *target = parsed;
    }
    if (cpuinfo.bad() || !commit() || cores.size() > static_cast<std::size_t>(std::numeric_limits<int>::max())) return 0;
    return static_cast<int>(cores.size());
}

HardwareInfo detect_hardware()
{
    HardwareInfo info;

    // ── Logical cores (portable) ───────────────────────────────────────────
    info.logical_cores = static_cast<int>(std::thread::hardware_concurrency());
    if (info.logical_cores <= 0)
        info.logical_cores = 1;

    // ── OpenMP threads ─────────────────────────────────────────────────────
#ifdef _OPENMP
    info.omp_max_threads = omp_get_max_threads();
    info.omp_dynamic     = omp_get_dynamic() != 0;
#if _OPENMP >= 201307
    info.omp_proc_bind  = openmp_proc_bind_name(static_cast<int>(omp_get_proc_bind()));
    info.omp_num_places = omp_get_num_places();
#else
    info.omp_proc_bind  = "runtime-default";
    info.omp_num_places = 0;
#endif
#else
    info.omp_max_threads = 1;
    info.omp_proc_bind   = "unavailable";
#endif

    // ── SIMD ───────────────────────────────────────────────────────────────
    auto simd_type       = simd_utils::get_available_simd();
    info.simd_level_name = simd_type_name(simd_type);
    info.simd_width      = simd_utils::simd_implementation_width(simd_type);

    // ── Platform-specific detection ────────────────────────────────────────
#if defined(__APPLE__)
    info.physical_cores = sysctl_value<int>("hw.physicalcpu", info.logical_cores);
    const int performance_cores =
        sysctl_value<int>("hw.perflevel0.logicalcpu", 0);
    if (performance_cores > 0
        && performance_cores < info.logical_cores)
    {
        info.performance_cores = performance_cores;
        info.efficiency_cores =
            info.logical_cores - performance_cores;
    }
    info.cpu_model      = sysctl_string("machdep.cpu.brand_string");
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
    info.l2_cache_bytes  = sysctl_value<uint64_t>("hw.l2cachesize", 0);
    info.l3_cache_bytes  = sysctl_value<uint64_t>("hw.l3cachesize", 0);

#elif defined(__linux__)
    std::ifstream topology("/proc/cpuinfo");
    const int physical = hardware_topology::linux_physical_core_count(topology);
    info.physical_cores = physical > 0 ? std::min(physical, info.logical_cores) : info.logical_cores;
    info.cpu_model      = cpuinfo_field("model name");
    if (info.cpu_model.empty())
        info.cpu_model = "unknown";
    info.total_ram_bytes = parse_meminfo_kb("MemTotal:");
    info.l1d_cache_bytes = read_data_cache_size(1);
    info.l2_cache_bytes  = read_data_cache_size(2);
    info.l3_cache_bytes  = read_data_cache_size(3);

#elif defined(_WIN32)
    info.physical_cores = info.logical_cores;
    info.cpu_model = windows_cpu_model();
    MEMORYSTATUSEX memory{};
    memory.dwLength = sizeof(memory);
    if (GlobalMemoryStatusEx(&memory))
        info.total_ram_bytes = memory.ullTotalPhys;
    populate_windows_topology(info);

#else
    // Fallback for unsupported platforms
    info.physical_cores = info.logical_cores;
    info.cpu_model      = "unknown";
#endif

    return info;
}

int topology_aware_openmp_default(const HardwareInfo& info) noexcept
{
    if (info.performance_cores > 0
        && info.performance_cores < info.logical_cores)
    {
        // On older Windows/OpenMP combinations the runtime can be constrained
        // to one processor group even though Win32 reports the whole machine.
        // omp_get_max_threads() is the practical upper bound in that case.
        return info.omp_max_threads > 0
                   ? std::min(info.performance_cores, info.omp_max_threads)
                   : info.performance_cores;
    }
    if (info.physical_cores > 0 && info.physical_cores < info.logical_cores)
        return info.omp_max_threads > 0
                   ? std::min(info.physical_cores, info.omp_max_threads)
                   : info.physical_cores;
    return 0;
}

uint64_t process_peak_rss_bytes()
{
#if defined(__APPLE__) || defined(__linux__)
    rusage usage{};
    if (getrusage(RUSAGE_SELF, &usage) != 0 || usage.ru_maxrss < 0)
    {
        return 0;
    }
    const uint64_t raw = static_cast<uint64_t>(usage.ru_maxrss);
#if defined(__APPLE__)
    return raw;
#else
    return raw * 1024ULL;
#endif
#else
#if defined(_WIN32)
    PROCESS_MEMORY_COUNTERS counters{};
    counters.cb = sizeof(counters);
    if (GetProcessMemoryInfo(GetCurrentProcess(), &counters, sizeof(counters)))
        return static_cast<uint64_t>(counters.PeakWorkingSetSize);
#endif
    return 0;
#endif
}

bool process_memory_trace_enabled(const char* value) noexcept
{
    return value != nullptr && std::strcmp(value, "1") == 0;
}

bool parse_linux_smaps_rollup_rss(const char* data, std::size_t size,
                                 std::uint64_t& bytes) noexcept
{
    if (data == nullptr || size == 0 || size >= 16384 || std::memchr(data, '\0', size) != nullptr)
    {
        return false;
    }
    std::string_view remaining(data, size);
    bool found = false;
    std::uint64_t result = 0;
    while (!remaining.empty())
    {
        const auto newline = remaining.find('\n');
        std::string_view line = remaining.substr(0, newline);
        remaining = newline == std::string_view::npos
            ? std::string_view{} : remaining.substr(newline + 1);
        if (line.substr(0, 4) != "Rss:")
        {
            continue;
        }
        if (found)
        {
            return false;
        }
        line.remove_prefix(4);
        while (!line.empty() && (line.front() == ' ' || line.front() == '\t'))
        {
            line.remove_prefix(1);
        }
        std::uint64_t kilobytes = 0;
        const auto parsed = std::from_chars(line.data(), line.data() + line.size(), kilobytes);
        if (parsed.ec != std::errc{} || parsed.ptr == line.data() ||
            parsed.ptr == line.data() + line.size() ||
            (*parsed.ptr != ' ' && *parsed.ptr != '\t') ||
            kilobytes > std::numeric_limits<std::uint64_t>::max() / 1024)
        {
            return false;
        }
        line.remove_prefix(static_cast<std::size_t>(parsed.ptr - line.data()));
        while (!line.empty() && (line.front() == ' ' || line.front() == '\t'))
        {
            line.remove_prefix(1);
        }
        while (!line.empty() && (line.back() == ' ' || line.back() == '\t' || line.back() == '\r'))
        {
            line.remove_suffix(1);
        }
        if (line != "kB")
        {
            return false;
        }
        result = kilobytes * 1024;
        found = true;
    }
    if (!found)
    {
        return false;
    }
    bytes = result;
    return true;
}

ProcessMemorySample sample_process_memory() noexcept
{
    const int original_errno = errno;
    ProcessMemorySample sample;
#if defined(__APPLE__)
    sample.current_rss_method = "macos_mach_task_basic_info";
    mach_task_basic_info_data_t info{};
    mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO,
                  reinterpret_cast<task_info_t>(&info), &count) == KERN_SUCCESS &&
        count >= MACH_TASK_BASIC_INFO_COUNT)
    {
        sample.current_rss_bytes = static_cast<std::uint64_t>(info.resident_size);
        sample.current_rss_available = true;
    }
#elif defined(__linux__)
    sample.current_rss_method = "linux_smaps_rollup";
    if (std::FILE* file = std::fopen("/proc/self/smaps_rollup", "rb"))
    {
        std::array<char, 16384> buffer{};
        const auto size = std::fread(buffer.data(), 1, buffer.size(), file);
        const bool read_ok = std::ferror(file) == 0 && std::feof(file) != 0;
        const bool close_ok = std::fclose(file) == 0;
        if (read_ok && close_ok)
        {
            sample.current_rss_available = parse_linux_smaps_rollup_rss(
                buffer.data(), size, sample.current_rss_bytes);
        }
    }
#endif
#if defined(__APPLE__) || defined(__linux__)
    rusage usage{};
    if (getrusage(RUSAGE_SELF, &usage) == 0 && usage.ru_maxrss >= 0)
    {
        const auto raw = static_cast<std::uint64_t>(usage.ru_maxrss);
#if defined(__APPLE__)
        sample.lifetime_peak_rss_bytes = raw;
        sample.lifetime_peak_available = true;
#else
        if (raw <= std::numeric_limits<std::uint64_t>::max() / 1024)
        {
            sample.lifetime_peak_rss_bytes = raw * 1024;
            sample.lifetime_peak_available = true;
        }
#endif
    }
#endif
    errno = original_errno;
    return sample;
}

void write_process_memory_trace(std::ostream& stream, bool enabled,
                                const char* scope, const char* phase,
                                const char* operation, std::size_t ordinal,
                                std::uint64_t payload_bytes,
                                ProcessMemorySampler sampler) noexcept
{
    if (!enabled)
    {
        return;
    }
    const auto original_flags = stream.flags();
    const int original_errno = errno;
    try
    {
        stream.setf(std::ios::dec, std::ios::basefield);
        const auto sample = sampler != nullptr ? sampler() : ProcessMemorySample{};
        stream << '[' << (scope != nullptr ? scope : "MEMORY") << "] phase="
               << (phase != nullptr ? phase : "unavailable") << ", current_rss_bytes=";
        if (sample.current_rss_available)
        {
            stream << sample.current_rss_bytes;
        }
        else
        {
            stream << "unavailable";
        }
        stream << ", lifetime_peak_rss_bytes=";
        if (sample.lifetime_peak_available)
        {
            stream << sample.lifetime_peak_rss_bytes;
        }
        else
        {
            stream << "unavailable";
        }
        stream << ", rss_method=" << (sample.current_rss_method != nullptr ? sample.current_rss_method : "unsupported")
               << ", operation=" << (operation != nullptr ? operation : "none")
               << ", ordinal=" << ordinal << ", payload_bytes=" << payload_bytes << std::endl;
    }
    catch (...)
    {
    }
    stream.flags(original_flags);
    errno = original_errno;
}

void log_hardware_info(const HardwareInfo& info)
{
    tmv::log_info("[HARDWARE] CPU: ", info.cpu_model);
    tmv::log_info("[HARDWARE]   Cores: ",
                  info.physical_cores,
                  "p/",
                  info.logical_cores,
                  "t across ",
                  info.processor_groups,
                  info.processor_groups == 1 ? " processor group, OpenMP threads: "
                                             : " processor groups, OpenMP threads: ",
                  info.omp_max_threads);
    if (info.performance_cores > 0)
    {
        tmv::log_info("[HARDWARE]   Core-class logical processors: ",
                      info.performance_cores,
                      " performance, ",
                      info.efficiency_cores,
                      " efficiency");
    }
    tmv::log_info("[HARDWARE]   OpenMP: dynamic=",
                  (info.omp_dynamic ? "on" : "off"),
                  ", proc_bind=",
                  info.omp_proc_bind,
                  ", places=",
                  info.omp_num_places);
    tmv::log_info("[HARDWARE]   SIMD: ", info.simd_level_name, " (width=", info.simd_width, ")");

    if (info.total_ram_bytes > 0)
    {
        tmv::log_info("[HARDWARE]   RAM: ", format_bytes(info.total_ram_bytes));
    }

    if (info.l1d_cache_bytes > 0 || info.l2_cache_bytes > 0)
    {
        tmv::log_info("[HARDWARE]   Cache: L1d=",
                      format_bytes(info.l1d_cache_bytes),
                      ", L2=",
                      format_bytes(info.l2_cache_bytes),
                      ", L3=",
                      format_bytes(info.l3_cache_bytes));
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

GridMemoryPreflight assess_grid_memory(const HardwareInfo& info,
                                       int                 nr,
                                       int                 nth,
                                       int                 nz,
                                       int                 num_fields,
                                       uint64_t            fixed_overhead_bytes,
                                       int                 warning_budget_percent)
{
    GridMemoryPreflight result;
    result.field_equivalents    = std::max(0, num_fields);
    result.field_storage_bytes  = estimate_grid_memory_bytes(nr, nth, nz, result.field_equivalents);
    result.fixed_overhead_bytes = fixed_overhead_bytes;
    result.estimated_bytes      = result.field_storage_bytes;
    if (std::numeric_limits<uint64_t>::max() - result.estimated_bytes < result.fixed_overhead_bytes)
    {
        result.estimated_bytes = std::numeric_limits<uint64_t>::max();
    }
    else
    {
        result.estimated_bytes += result.fixed_overhead_bytes;
    }
    result.ram_known = info.total_ram_bytes > 0;
    if (!result.ram_known)
    {
        return result;
    }

    const int clamped_percent     = std::clamp(warning_budget_percent, 1, 100);
    result.warning_budget_bytes   = info.total_ram_bytes * static_cast<uint64_t>(clamped_percent) / 100;
    result.exceeds_warning_budget = result.estimated_bytes > result.warning_budget_bytes;
    return result;
}

void check_grid_memory_safety(const HardwareInfo& info,
                              int                 nr,
                              int                 nth,
                              int                 nz,
                              int                 num_fields,
                              const std::string&  profile_label,
                              uint64_t            fixed_overhead_bytes)
{
    const GridMemoryPreflight preflight = assess_grid_memory(info, nr, nth, nz, num_fields, fixed_overhead_bytes);
    if (!preflight.ram_known)
    {
        tmv::log_warn("[HARDWARE] Grid memory preflight unavailable: physical RAM could not be detected. "
                      "Requested grid will be preserved.");
        return;
    }

    if (preflight.exceeds_warning_budget)
    {
        tmv::log_warn("[HARDWARE] Requested grid ",
                      nr,
                      "x",
                      nth,
                      "x",
                      nz,
                      " has a planning estimate of ~",
                      format_bytes(preflight.estimated_bytes),
                      " (",
                      preflight.field_equivalents,
                      " full-field equivalents)");
        if (!profile_label.empty())
        {
            tmv::log_warn("[HARDWARE] Memory profile: ", profile_label);
        }
        tmv::log_warn("[HARDWARE] Physical RAM: ",
                      format_bytes(info.total_ram_bytes),
                      "; 80% planning budget: ",
                      format_bytes(preflight.warning_budget_bytes));
        tmv::log_warn("[HARDWARE] Nimbus will not rescale the requested grid or spacing. "
                      "The configured memory-admission policy determines whether startup "
                      "continues. Reduce nx/ny/nz, select lighter physics/output staging, "
                      "or use a host with more memory. If changing cell counts, decide "
                      "whether dx/dy/dz must change to preserve the physical domain extent.");
        return;
    }

    tmv::log_debug("[HARDWARE] Grid ",
                   nr,
                   "x",
                   nth,
                   "x",
                   nz,
                   " planning estimate ",
                   format_bytes(preflight.estimated_bytes),
                   " (",
                   preflight.field_equivalents,
                   " full-field equivalents) fits within the 80% RAM warning budget");
}
