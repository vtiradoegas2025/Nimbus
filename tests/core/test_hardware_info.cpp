/**
 * @file test_hardware_info.cpp
 * @brief Unit tests for hardware detection and memory estimation.
 */
#include "catch2/catch.hpp"
#include "core/infra/hardware_info.hpp"
#include "core/infra/runtime_memory_profile.hpp"
#include "compute/compute_memory_admission.hpp"
#include "core/output/output_config.hpp"
#include "core/output/velocity_staging.hpp"

#include <unordered_set>
#include <cerrno>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace
{

int process_memory_sampler_calls = 0;

ProcessMemorySample counted_process_memory_sample() noexcept
{
    ++process_memory_sampler_calls;
    errno = EACCES;
    return {16384, 65536, true, true, "test_sample"};
}

ProcessMemorySample unavailable_process_memory_sample() noexcept
{
    ++process_memory_sampler_calls;
    return {};
}

ProcessMemorySample missing_live_process_memory_sample() noexcept
{
    return {0, 65536, false, true, "linux_smaps_rollup"};
}

class FailingMemoryTraceBuffer : public std::streambuf
{
protected:
    int_type overflow(int_type) override
    {
        throw std::runtime_error("diagnostic sink unavailable");
    }
};

}

TEST_CASE("Linux rollup RSS parsing preserves units and rejects malformed data atomically",
          "[core][hardware][process-memory][parser]")
{
    const std::string valid =
        "1000-2000 ---p 00000000 00:00 0 [rollup]\n"
        "Rss:                  8192 kB\nPss: 4096 kB\nRssAnon: 17 kB\n";
    std::uint64_t bytes = 42;
    REQUIRE(parse_linux_smaps_rollup_rss(valid.data(), valid.size(), bytes));
    REQUIRE(bytes == 8192ULL * 1024);

    const std::string malformed[] = {
        "", "RssAnon: 1 kB\n", "Rss: -1 kB\n", "Rss: +1 kB\n", "Rss: 1.5 kB\n",
        "Rss: 1 KB\n", "Rss: 1 B\n", "Rss: 1\n", "Rss: 1kB\n", "Rss: kB\n",
        "Rss: 1 kB extra\n", "Rss: 1 kB\nRss: 1 kB\n",
        "Rss: 18446744073709551616 kB\n", "Rss: 18014398509481984 kB\n",
        std::string("Rss: 1 kB\n\0bad", 15), std::string(16384, ' '),
    };
    for (const auto& input : malformed)
    {
        INFO(input);
        bytes = 42;
        REQUIRE_FALSE(parse_linux_smaps_rollup_rss(input.data(), input.size(), bytes));
        REQUIRE(bytes == 42);
    }
    REQUIRE_FALSE(parse_linux_smaps_rollup_rss(nullptr, 12, bytes));
    REQUIRE(bytes == 42);
}

TEST_CASE("Linux rollup RSS parser admits exact multiplication boundary without overflow",
          "[core][hardware][process-memory][parser]")
{
    const auto maximum_kilobytes = std::numeric_limits<std::uint64_t>::max() / 1024;
    const auto boundary = "Rss:\t" + std::to_string(maximum_kilobytes) + "\tkB\r\n";
    std::uint64_t bytes = 42;
    REQUIRE(parse_linux_smaps_rollup_rss(boundary.data(), boundary.size(), bytes));
    REQUIRE(bytes == maximum_kilobytes * 1024);
    const std::string zero = "Rss: 0 kB";
    REQUIRE(parse_linux_smaps_rollup_rss(zero.data(), zero.size(), bytes));
    REQUIRE(bytes == 0);
}

TEST_CASE("Process memory trace requires the exact opt-in and does no disabled sampling or output",
          "[core][hardware][process-memory][disabled]")
{
    const char* disabled_values[] = {nullptr, "", "0", "true", "01", "1 ", "+1", "10"};
    for (const char* value : disabled_values)
    {
        REQUIRE_FALSE(process_memory_trace_enabled(value));
    }
    REQUIRE(process_memory_trace_enabled("1"));
    process_memory_sampler_calls = 0;
    std::ostringstream output;
    write_process_memory_trace(output, false, "VULKAN STARTUP MEMORY", "pipeline_ready",
                               "microphysics.thompson_completion", 77, 5814800,
                               counted_process_memory_sample);
    REQUIRE(process_memory_sampler_calls == 0);
    REQUIRE(output.str().empty());
}

TEST_CASE("Process memory trace binds phase operation ordinal and distinguishes live from lifetime peak",
          "[core][hardware][process-memory][trace]")
{
    process_memory_sampler_calls = 0;
    std::ostringstream output;
    output << std::hex;
    const auto flags = output.flags();
    errno = EINVAL;
    write_process_memory_trace(output, true, "VULKAN STARTUP MEMORY", "pipeline_ready",
                               "microphysics.thompson_completion", 77, 5814800,
                               counted_process_memory_sample);
    REQUIRE(process_memory_sampler_calls == 1);
    REQUIRE(output.flags() == flags);
    REQUIRE(errno == EINVAL);
    REQUIRE(output.str() ==
        "[VULKAN STARTUP MEMORY] phase=pipeline_ready, current_rss_bytes=16384, "
        "lifetime_peak_rss_bytes=65536, rss_method=test_sample, "
        "operation=microphysics.thompson_completion, ordinal=77, payload_bytes=5814800\n");
}

TEST_CASE("Unavailable or missing process memory sampler never fabricates an RSS zero",
          "[core][hardware][process-memory][trace]")
{
    process_memory_sampler_calls = 0;
    const ProcessMemorySampler samplers[] = {unavailable_process_memory_sample, nullptr};
    for (ProcessMemorySampler sampler : samplers)
    {
        std::ostringstream output;
        REQUIRE_NOTHROW(write_process_memory_trace(output, true, "VULKAN STARTUP MEMORY",
            "cache_initialize_end", nullptr, 0, 0, sampler));
        REQUIRE(output.str().find("current_rss_bytes=unavailable") != std::string::npos);
        REQUIRE(output.str().find("lifetime_peak_rss_bytes=unavailable") != std::string::npos);
        REQUIRE(output.str().find("rss_bytes=0") == std::string::npos);
    }
    REQUIRE(process_memory_sampler_calls == 1);
    std::ostringstream partial;
    write_process_memory_trace(partial, true, "VULKAN STARTUP MEMORY", "init_end",
                               nullptr, 0, 0, missing_live_process_memory_sample);
    REQUIRE(partial.str().find("current_rss_bytes=unavailable") != std::string::npos);
    REQUIRE(partial.str().find("lifetime_peak_rss_bytes=65536") != std::string::npos);
}

TEST_CASE("Diagnostic memory trace sink failure cannot fail simulation",
          "[core][hardware][process-memory][trace]")
{
    FailingMemoryTraceBuffer buffer;
    std::ostream output(&buffer);
    output.exceptions(std::ios::badbit | std::ios::failbit);
    REQUIRE_NOTHROW(write_process_memory_trace(output, true, "VULKAN STARTUP MEMORY",
        "init_end", nullptr, 0, 0, counted_process_memory_sample));
}

TEST_CASE("Current process memory sample uses the native method without confusing lifetime peak",
          "[core][hardware][process-memory][os]")
{
    errno = EINTR;
    const auto sample = sample_process_memory();
    REQUIRE(errno == EINTR);
#if defined(__APPLE__) || defined(__linux__)
    REQUIRE(sample.current_rss_available);
    REQUIRE(sample.lifetime_peak_available);
    REQUIRE(sample.current_rss_bytes > 1024 * 1024);
    REQUIRE(sample.lifetime_peak_rss_bytes > 1024 * 1024);
#if defined(__APPLE__)
    REQUIRE(std::string(sample.current_rss_method) == "macos_mach_task_basic_info");
#else
    REQUIRE(std::string(sample.current_rss_method) == "linux_smaps_rollup");
#endif
#else
    REQUIRE_FALSE(sample.current_rss_available);
    REQUIRE_FALSE(sample.lifetime_peak_available);
#endif
}

TEST_CASE("detect_hardware returns valid CPU info", "[core][hardware]")
{
    HardwareInfo hw = detect_hardware();

    REQUIRE(hw.physical_cores > 0);
    REQUIRE(hw.logical_cores >= hw.physical_cores);
    REQUIRE(hw.processor_groups >= 1);
    REQUIRE(hw.performance_cores >= 0);
    REQUIRE(hw.efficiency_cores >= 0);
    REQUIRE(hw.performance_cores + hw.efficiency_cores
            <= hw.logical_cores);
    REQUIRE(hw.omp_max_threads >= 1);
    REQUIRE_FALSE(hw.omp_proc_bind.empty());
    REQUIRE(hw.omp_num_places >= 0);
    REQUIRE_FALSE(hw.cpu_model.empty());
    REQUIRE(hw.cpu_model != "unknown");
}

TEST_CASE("topology-aware OpenMP default respects core classes SMT and runtime limits",
          "[core][hardware][threads]")
{
    HardwareInfo homogeneous;
    homogeneous.logical_cores = 8;
    homogeneous.omp_max_threads = 8;
    REQUIRE(topology_aware_openmp_default(homogeneous) == 0);

    homogeneous.performance_cores = 8;
    REQUIRE(topology_aware_openmp_default(homogeneous) == 0);

    HardwareInfo smt;
    smt.physical_cores = 24;
    smt.logical_cores = 48;
    smt.omp_max_threads = 48;
    REQUIRE(topology_aware_openmp_default(smt) == 24);
    smt.omp_max_threads = 12;
    REQUIRE(topology_aware_openmp_default(smt) == 12);
    smt.omp_max_threads = 0;
    REQUIRE(topology_aware_openmp_default(smt) == 24);
    smt.physical_cores = 0;
    REQUIRE(topology_aware_openmp_default(smt) == 0);
    smt.physical_cores = 96;
    REQUIRE(topology_aware_openmp_default(smt) == 0);

    HardwareInfo heterogeneous;
    heterogeneous.logical_cores = 16;
    heterogeneous.performance_cores = 12;
    heterogeneous.efficiency_cores = 4;
    heterogeneous.omp_max_threads = 16;
    REQUIRE(topology_aware_openmp_default(heterogeneous) == 12);

    heterogeneous.omp_max_threads = 8;
    REQUIRE(topology_aware_openmp_default(heterogeneous) == 8);

    heterogeneous.omp_max_threads = 0;
    REQUIRE(topology_aware_openmp_default(heterogeneous) == 12);
}

TEST_CASE("Linux topology counts physical socket-core pairs rather than SMT processors",
          "[core][hardware][threads][linux-topology]")
{
    std::istringstream topology(
        "processor : 0\nphysical id : 0\ncore id : 2\n\n"
        "processor\t: 4\nphysical id\t: 0\ncore id\t: 2\n\n"
        "processor : 10\nphysical id : 1\ncore id : 2\n\n"
        "processor : 20\nphysical id : 1\ncore id : 3");
    REQUIRE(hardware_topology::linux_physical_core_count(topology) == 3);
    for (const std::string input : {
        "", "processor: 0\n", "processor: 0\ncore id: 0\n",
        "physical id: 0\ncore id: 0\n", "processor: 0\nphysical id: -1\ncore id: 0\n",
        "processor: 0\nphysical id: 0\ncore id: 1x\n",
        "processor: 0\nphysical id: 0\ncore id: 0\ncore id: 0\n",
        "processor: 0\nphysical id: 0\ncore id: 0\n\nprocessor: 0\nphysical id: 0\ncore id: 0\n",
        "processor: 0\nphysical id: 0\ncore id: 0\n\nprocessor: 1\n",
        "processor: 0\nphysical id: 0\ncore id: 999999999999999999999999\n"})
    {
        CAPTURE(input);
        std::istringstream malformed(input);
        REQUIRE(hardware_topology::linux_physical_core_count(malformed) == 0);
    }
}

TEST_CASE("core classes count logical processors and select the highest Windows class",
          "[core][hardware][threads][windows]")
{
    using hardware_topology::CoreObservation;

    // Two SMT performance cores (class 8) and four single-threaded efficiency
    // cores (class 0), matching a common Windows hybrid topology shape.
    const CoreObservation hybrid[] = {
        {2, 8}, {1, 0}, {2, 8}, {1, 0}, {1, 0}, {1, 0}};
    const auto summary = hardware_topology::summarize_core_classes(
        hybrid, sizeof(hybrid) / sizeof(hybrid[0]));

    REQUIRE(summary.logical_processors == 8);
    REQUIRE(summary.performance_processors == 4);
    REQUIRE(summary.efficiency_processors == 4);
    REQUIRE(summary.distinct_classes == 2);
}

TEST_CASE("core-class summary leaves homogeneous and unavailable topology alone",
          "[core][hardware][threads][windows]")
{
    using hardware_topology::CoreObservation;

    const CoreObservation homogeneous[] = {{2, 0}, {2, 0}, {0, 9}};
    const auto summary = hardware_topology::summarize_core_classes(
        homogeneous, sizeof(homogeneous) / sizeof(homogeneous[0]));
    REQUIRE(summary.logical_processors == 4);
    REQUIRE(summary.performance_processors == 0);
    REQUIRE(summary.efficiency_processors == 0);
    REQUIRE(summary.distinct_classes == 1);

    const auto unavailable =
        hardware_topology::summarize_core_classes(nullptr, 42);
    REQUIRE(unavailable.logical_processors == 0);
    REQUIRE(unavailable.distinct_classes == 0);
}

TEST_CASE("core-class summary folds every lower Windows class into efficiency capacity",
          "[core][hardware][threads][windows]")
{
    const hardware_topology::CoreObservation classes[] = {
        {2, 12}, {2, 12}, {1, 4}, {1, 0}, {1, 0}};
    const auto summary = hardware_topology::summarize_core_classes(
        classes, sizeof(classes) / sizeof(classes[0]));

    REQUIRE(summary.logical_processors == 7);
    REQUIRE(summary.performance_processors == 4);
    REQUIRE(summary.efficiency_processors == 3);
    REQUIRE(summary.distinct_classes == 3);
}

TEST_CASE("detect_hardware returns valid memory info", "[core][hardware]")
{
    HardwareInfo hw = detect_hardware();
    // Any real machine has at least 1 GB
    REQUIRE(hw.total_ram_bytes > 1'000'000'000ULL);
}

TEST_CASE("process peak RSS uses normalized byte units", "[core][hardware][cpu017]")
{
    REQUIRE(process_peak_rss_bytes() > 1024ULL * 1024ULL);
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

TEST_CASE("grid memory preflight reports a fitting request without mutation",
          "[core][hardware][memory-preflight]")
{
    HardwareInfo hw;
    hw.total_ram_bytes = 48ULL * 1024 * 1024 * 1024; // 48 GB

    const int nr = 64;
    const int nth = 64;
    const int nz = 32;
    const GridMemoryPreflight result =
        assess_grid_memory(hw, nr, nth, nz, 77);

    REQUIRE(result.ram_known);
    REQUIRE(result.field_equivalents == 77);
    REQUIRE(result.estimated_bytes == estimate_grid_memory_bytes(nr, nth, nz, 77));
    REQUIRE_FALSE(result.exceeds_warning_budget);
    REQUIRE(nr == 64);
    REQUIRE(nth == 64);
    REQUIRE(nz == 32);
}

TEST_CASE("grid memory preflight flags an oversized request without rescaling it",
          "[core][hardware][memory-preflight]")
{
    HardwareInfo hw;
    hw.total_ram_bytes = 8ULL * 1024 * 1024 * 1024; // 8 GB (student laptop)

    const int nr = 512;
    const int nth = 512;
    const int nz = 256;
    const GridMemoryPreflight result =
        assess_grid_memory(hw, nr, nth, nz, 77);

    REQUIRE(result.exceeds_warning_budget);
    REQUIRE_FALSE(memory_admission_should_reject(result, false));
    REQUIRE(memory_admission_should_reject(result, true));
    REQUIRE(result.warning_budget_bytes == hw.total_ram_bytes * 80 / 100);
    REQUIRE(nr == 512);
    REQUIRE(nth == 512);
    REQUIRE(nz == 256);
    REQUIRE_NOTHROW(check_grid_memory_safety(
        hw, nr, nth, nz, 77,
        "microphysics=kessler, turbulence=smagorinsky"));
}

TEST_CASE("grid memory preflight handles unknown RAM without changing the request",
          "[core][hardware][memory-preflight]")
{
    HardwareInfo hw;
    hw.total_ram_bytes = 0; // Unknown RAM

    const GridMemoryPreflight result =
        assess_grid_memory(hw, 256, 256, 128, 77);

    REQUIRE_FALSE(result.ram_known);
    REQUIRE_FALSE(result.exceeds_warning_budget);
    REQUIRE_FALSE(memory_admission_should_reject(result, true));
    REQUIRE(result.warning_budget_bytes == 0);
    REQUIRE(result.estimated_bytes ==
            estimate_grid_memory_bytes(256, 256, 128, 77));
}

TEST_CASE("grid memory preflight uses the supplied scheme field count",
          "[core][hardware][memory-preflight]")
{
    HardwareInfo hw;
    hw.total_ram_bytes = 16ULL * 1024 * 1024 * 1024;

    const GridMemoryPreflight light =
        assess_grid_memory(hw, 256, 256, 128, 58);
    const GridMemoryPreflight heavy =
        assess_grid_memory(hw, 256, 256, 128, 102);

    REQUIRE(light.estimated_bytes == estimate_grid_memory_bytes(256, 256, 128, 58));
    REQUIRE(heavy.estimated_bytes == estimate_grid_memory_bytes(256, 256, 128, 102));
    REQUIRE(heavy.estimated_bytes > light.estimated_bytes);
}

TEST_CASE("grid memory preflight adds fixed process overhead separately",
          "[core][hardware][memory-preflight][cpu017]")
{
    HardwareInfo hw;
    hw.total_ram_bytes = uint64_t(16) << 30;
    const uint64_t fixed_overhead = uint64_t(8) << 20;
    const GridMemoryPreflight result =
        assess_grid_memory(hw, 32, 32, 32, 97, fixed_overhead);

    REQUIRE(result.field_storage_bytes
            == estimate_grid_memory_bytes(32, 32, 32, 97));
    REQUIRE(result.fixed_overhead_bytes == fixed_overhead);
    REQUIRE(result.estimated_bytes
            == result.field_storage_bytes + fixed_overhead);
}

TEST_CASE("checkpoint encoding preflight covers streamed section assembly",
          "[core][hardware][memory-preflight][checkpoint]")
{
    REQUIRE(estimate_checkpoint_encoding_transient_bytes(8, 4, 33) ==
            3'960);
    REQUIRE(estimate_checkpoint_encoding_transient_bytes(8, 0, 33) ==
            2'640);
    REQUIRE(estimate_checkpoint_encoding_transient_bytes(8, 4, 0) == 0);
    REQUIRE(estimate_checkpoint_encoding_transient_bytes(
                std::numeric_limits<std::uint64_t>::max(), 1, 33) ==
            std::numeric_limits<std::uint64_t>::max());
    REQUIRE(estimate_checkpoint_encoding_transient_bytes(
                std::numeric_limits<std::uint64_t>::max() / 4, 0, 33) ==
            std::numeric_limits<std::uint64_t>::max());
}

TEST_CASE("compute memory preflight separates shared and device budgets",
          "[core][hardware][memory-preflight][gpu]")
{
    ComputeMemoryProperties unified;
    unified.accelerator = true;
    unified.unified_host_device_memory = true;
    unified.device_local_heap_bytes = std::uint64_t(16) << 30;
    unified.max_storage_buffer_bytes = std::uint64_t(1) << 30;

    const ComputeMemoryPreflight shared =
        assess_compute_memory(unified, 128, 128, 64);
    REQUIRE(shared.field_equivalents ==
            kComputeMaximumRetainedFieldEquivalents);
    REQUIRE(shared.resident_level_count == 1);
    REQUIRE(shared.field_bytes == 4'194'304);
    REQUIRE(shared.host_residency_bytes == 213'909'504);
    REQUIRE(shared.device_residency_bytes == shared.host_residency_bytes);
    REQUIRE_FALSE(shared.exceeds_storage_buffer_range);
    REQUIRE_FALSE(shared.exceeds_device_budget);

    ComputeMemoryProperties discrete = unified;
    discrete.unified_host_device_memory = false;
    const ComputeMemoryPreflight staged =
        assess_compute_memory(discrete, 128, 128, 64);
    REQUIRE(staged.host_residency_bytes == shared.host_residency_bytes);
    REQUIRE(staged.device_residency_bytes == shared.device_residency_bytes);
}

TEST_CASE("compute memory hierarchy sums retained levels and preserves root-only admission",
          "[core][hardware][memory-preflight][gpu][nesting]")
{
    ComputeMemoryProperties properties;
    properties.accelerator = true;
    properties.device_local_heap_bytes = std::uint64_t(1) << 30;
    properties.max_storage_buffer_bytes = std::uint64_t(1) << 30;

    const ComputeMemoryLevelDimensions root{8, 8, 4};
    const ComputeMemoryPreflight legacy =
        assess_compute_memory(properties, root.nx, root.ny, root.nz, 4);
    const ComputeMemoryPreflight root_only =
        assess_compute_memory_hierarchy(properties, &root, 1, 4);
    REQUIRE(root_only.field_equivalents == legacy.field_equivalents);
    REQUIRE(root_only.resident_level_count == legacy.resident_level_count);
    REQUIRE(root_only.field_bytes == legacy.field_bytes);
    REQUIRE(root_only.host_residency_bytes == legacy.host_residency_bytes);
    REQUIRE(root_only.device_residency_bytes == legacy.device_residency_bytes);
    REQUIRE(root_only.device_budget_bytes == legacy.device_budget_bytes);
    REQUIRE(root_only.exceeds_storage_buffer_range ==
            legacy.exceeds_storage_buffer_range);
    REQUIRE(root_only.exceeds_device_budget == legacy.exceeds_device_budget);

    const ComputeMemoryLevelDimensions levels[] = {
        root,
        {6, 6, 4}
    };
    const ComputeMemoryPreflight hierarchy =
        assess_compute_memory_hierarchy(properties, levels, 2, 4);
    REQUIRE(hierarchy.resident_level_count == 2);
    REQUIRE(hierarchy.field_bytes == 1'024);
    REQUIRE(hierarchy.host_residency_bytes == 6'400);
    REQUIRE(hierarchy.device_residency_bytes == 6'400);
    REQUIRE_FALSE(hierarchy.exceeds_storage_buffer_range);
    REQUIRE_FALSE(hierarchy.exceeds_device_budget);
}

TEST_CASE("frontier operational hierarchy is admitted as root plus enlarged child",
          "[core][hardware][memory-preflight][gpu][nesting][moving_nest]")
{
    ComputeMemoryProperties properties;
    properties.accelerator = true;
    properties.unified_host_device_memory = true;
    properties.device_local_heap_bytes = std::uint64_t(16) << 30;
    properties.max_storage_buffer_bytes = std::uint64_t(256) << 20;

    const ComputeMemoryLevelDimensions repaired[] = {
        {135, 108, 50},
        {158, 168, 50},
    };
    const auto result = assess_compute_memory_hierarchy(
        properties, repaired, 2);
    REQUIRE(result.resident_level_count == 2);
    REQUIRE(result.field_equivalents ==
            kComputeMaximumRetainedFieldEquivalents);
    REQUIRE(result.field_bytes == 5'308'800);
    REQUIRE(result.host_residency_bytes == 419'464'800);
    REQUIRE(result.device_residency_bytes == 419'464'800);
    REQUIRE_FALSE(result.exceeds_storage_buffer_range);
    REQUIRE_FALSE(result.exceeds_device_budget);
    REQUIRE_FALSE(compute_memory_admission_should_reject(result, true));

    const ComputeMemoryLevelDimensions historical[] = {
        {134, 105, 50},
        {118, 114, 50},
    };
    const auto old_result = assess_compute_memory_hierarchy(
        properties, historical, 2);
    REQUIRE(result.host_residency_bytes > old_result.host_residency_bytes);

    REQUIRE(estimate_checkpoint_encoding_transient_bytes(
                135ULL * 108ULL * 50ULL,
                158ULL * 168ULL * 50ULL,
                kCheckpointCoreDurableFieldEquivalents) == 678'546'000);
}

TEST_CASE("compute memory hierarchy checks the largest child buffer and summed heap",
          "[core][hardware][memory-preflight][gpu][nesting]")
{
    ComputeMemoryProperties properties;
    properties.accelerator = true;
    properties.device_local_heap_bytes = 2'000;
    properties.max_storage_buffer_bytes = 128;

    const ComputeMemoryLevelDimensions buffer_levels[] = {
        {4, 4, 1},
        {8, 8, 1}
    };
    const ComputeMemoryPreflight child_buffer =
        assess_compute_memory_hierarchy(properties, buffer_levels, 2, 1);
    REQUIRE(child_buffer.field_bytes == 256);
    REQUIRE(child_buffer.exceeds_storage_buffer_range);

    properties.max_storage_buffer_bytes = 1'024;
    const ComputeMemoryLevelDimensions heap_levels[] = {
        {8, 8, 1},
        {8, 8, 1}
    };
    const ComputeMemoryPreflight root =
        assess_compute_memory(properties, 8, 8, 1, 4);
    const ComputeMemoryPreflight summed =
        assess_compute_memory_hierarchy(properties, heap_levels, 2, 4);
    REQUIRE(root.device_residency_bytes == 1'024);
    REQUIRE_FALSE(root.exceeds_device_budget);
    REQUIRE(summed.device_residency_bytes == 2'048);
    REQUIRE(summed.device_budget_bytes == 1'600);
    REQUIRE(summed.exceeds_device_budget);
}

TEST_CASE("compute memory admission fails closed on field and heap limits",
          "[core][hardware][memory-preflight][gpu]")
{
    ComputeMemoryProperties properties;
    properties.accelerator = true;
    properties.device_local_heap_bytes = 1024;
    properties.max_storage_buffer_bytes = 128;

    const ComputeMemoryPreflight result =
        assess_compute_memory(properties, 8, 8, 1, 4);
    REQUIRE(result.field_bytes == 256);
    REQUIRE(result.device_residency_bytes == 1024);
    REQUIRE(result.device_budget_bytes == 819);
    REQUIRE(result.exceeds_storage_buffer_range);
    REQUIRE(result.exceeds_device_budget);
    REQUIRE_FALSE(compute_memory_admission_should_reject(result, false));
    REQUIRE(compute_memory_admission_should_reject(result, true));

    const ComputeMemoryPreflight cpu =
        assess_compute_memory({}, 1000000, 1000000, 1000000);
    REQUIRE_FALSE(cpu.accelerator);
    REQUIRE_FALSE(compute_memory_admission_should_reject(cpu, true));

    properties.device_local_heap_bytes =
        std::numeric_limits<std::uint64_t>::max();
    properties.max_storage_buffer_bytes =
        std::numeric_limits<std::uint64_t>::max() - 1;
    const ComputeMemoryPreflight overflow = assess_compute_memory(
        properties, std::numeric_limits<int>::max(),
        std::numeric_limits<int>::max(),
        std::numeric_limits<int>::max());
    REQUIRE(overflow.field_bytes ==
            std::numeric_limits<std::uint64_t>::max());
    REQUIRE(overflow.device_residency_bytes ==
            std::numeric_limits<std::uint64_t>::max());
    REQUIRE(overflow.exceeds_storage_buffer_range);
    REQUIRE(overflow.exceeds_device_budget);

    const ComputeMemoryLevelDimensions overflow_levels[] = {
        {std::numeric_limits<int>::max(),
         std::numeric_limits<int>::max(),
         std::numeric_limits<int>::max()},
        {1, 1, 1}
    };
    const ComputeMemoryPreflight hierarchy_overflow =
        assess_compute_memory_hierarchy(
            properties, overflow_levels, 2,
            std::numeric_limits<int>::max());
    REQUIRE(hierarchy_overflow.host_residency_bytes ==
            std::numeric_limits<std::uint64_t>::max());
    REQUIRE(hierarchy_overflow.device_residency_bytes ==
            std::numeric_limits<std::uint64_t>::max());

    const ComputeMemoryPreflight hierarchy_cpu =
        assess_compute_memory_hierarchy({}, overflow_levels, 2);
    REQUIRE_FALSE(hierarchy_cpu.accelerator);
    REQUIRE(hierarchy_cpu.field_equivalents == 0);
    REQUIRE(hierarchy_cpu.resident_level_count == 0);
    REQUIRE(hierarchy_cpu.field_bytes == 0);
    REQUIRE(hierarchy_cpu.host_residency_bytes == 0);
    REQUIRE(hierarchy_cpu.device_residency_bytes == 0);
}

TEST_CASE("runtime memory phases preserve diagnostic-free solver residency",
          "[core][hardware][memory-preflight][cpu017]")
{
    RuntimeMemoryPhaseInputs inputs;
    inputs.solver_resident_fields = 75;

    const RuntimeMemoryPhaseProfile profile =
        estimate_runtime_memory_phases(inputs);

    REQUIRE(profile.solver_resident_fields == 75);
    REQUIRE(profile.lazy_diagnostic_fields == 0);
    REQUIRE(profile.resident_fields == 75);
    REQUIRE(profile.output_working_field_equivalents == 0);
    REQUIRE(profile.high_water_field_equivalents == 75);
}

TEST_CASE("solver memory profile includes active physics high water",
          "[core][hardware][memory-preflight][cpu017]")
{
    REQUIRE(estimate_runtime_solver_fields({
        "kessler", "smagorinsky", "weno5", "none"}) == 75);
    REQUIRE(estimate_runtime_solver_fields({
        "lin", "smagorinsky", "weno5", "none"}) == 79);
    REQUIRE(estimate_runtime_solver_fields({
        "kessler", "smagorinsky", "weno5", "slab"}) == 106);
    REQUIRE(estimate_runtime_solver_fields({
        "lin", "smagorinsky", "weno5", "ysu"}) == 110);

    REQUIRE(estimate_runtime_solver_fields({
        "none", "none", "none", "none"}) == 47);
    REQUIRE(estimate_runtime_solver_fields({
        "thompson", "tke", "weno5", "none"}) == 176);
    REQUIRE(estimate_runtime_solver_fields({
        "milbrandt", "smagorinsky", "weno5", "none"}) == 171);
    REQUIRE(estimate_runtime_solver_fields({
        "kessler", "smagorinsky", "weno5", "none",
        "simple_grey"}) == 78);
}

TEST_CASE("runtime memory phases model one bounded async snapshot and 2D maps",
          "[core][hardware][memory-preflight][cpu017]")
{
    RuntimeMemoryPhaseInputs inputs;
    inputs.solver_resident_fields = 75;
    inputs.lazy_diagnostic_fields = 12;
    inputs.output_enabled = true;
    inputs.volume_output = true;
    inputs.c_grid = true;
    inputs.async_io = true;
    inputs.core_volume_output_fields = 9;
    inputs.derived_volume_output_fields = 6;
    inputs.map_output_fields = 20;
    inputs.vertical_levels = 32;
    inputs.horizontal_rows = 32;

    const RuntimeMemoryPhaseProfile profile =
        estimate_runtime_memory_phases(inputs);

    REQUIRE(profile.resident_fields == 87);
    REQUIRE(profile.centered_velocity_fields == 3);
    REQUIRE(profile.slice_scratch_field_equivalents == 2);
    REQUIRE(profile.current_output_field_equivalents == 16);
    REQUIRE(profile.concurrent_snapshot_field_equivalents == 0);
    REQUIRE(profile.compression_history_field_equivalents == 0);
    REQUIRE(profile.output_working_field_equivalents == 21);
    REQUIRE(profile.high_water_field_equivalents == 108);
}

TEST_CASE("runtime memory phases separate sync and predictive ZFP high water",
          "[core][hardware][memory-preflight][cpu017]")
{
    RuntimeMemoryPhaseInputs sync;
    sync.solver_resident_fields = 75;
    sync.lazy_diagnostic_fields = 12;
    sync.output_enabled = true;
    sync.volume_output = true;
    sync.c_grid = true;
    sync.core_volume_output_fields = 9;
    sync.derived_volume_output_fields = 6;
    sync.map_output_fields = 20;
    sync.vertical_levels = 32;
    sync.horizontal_rows = 32;

    const RuntimeMemoryPhaseProfile sync_profile =
        estimate_runtime_memory_phases(sync);
    REQUIRE(sync_profile.current_output_field_equivalents == 7);
    REQUIRE(sync_profile.concurrent_snapshot_field_equivalents == 0);
    REQUIRE(sync_profile.high_water_field_equivalents == 99);

    RuntimeMemoryPhaseInputs predictive = sync;
    predictive.async_io = true;
    predictive.zfp_history_depth = 2;
    const RuntimeMemoryPhaseProfile predictive_profile =
        estimate_runtime_memory_phases(predictive);

    REQUIRE(predictive_profile.compression_history_field_equivalents == 33);
    REQUIRE(predictive_profile.high_water_field_equivalents == 141);
}

TEST_CASE("synchronous ZFP borrows core fields and still reserves codec memory",
          "[core][hardware][memory-preflight][zfp]")
{
    RuntimeMemoryPhaseInputs inputs;
    inputs.solver_resident_fields = 75;
    inputs.output_enabled = true;
    inputs.volume_output = true;
    // This is the internal snapshot-assembly disposition, not the user's
    // background-I/O switch. Synchronous ZFP sets it through
    // requires_snapshot_assembly().
    inputs.async_io = requires_snapshot_assembly(OutputFormat::zfp, false);
    inputs.borrow_core_output_fields = true;
    inputs.core_volume_output_fields = 8;
    inputs.slice_scratch_arrays = 0;
    inputs.zfp_history_depth = 2;

    const RuntimeMemoryPhaseProfile profile =
        estimate_runtime_memory_phases(inputs);
    REQUIRE(profile.current_output_field_equivalents == 0);
    REQUIRE(profile.compression_history_field_equivalents == 19);
    REQUIRE(profile.high_water_field_equivalents == 94);

    inputs.derived_volume_output_fields = 3;
    inputs.map_output_fields = 2;
    inputs.vertical_levels = 32;
    inputs.terrain_geometry = true;
    const auto terrain = estimate_runtime_memory_phases(inputs);
    REQUIRE(terrain.current_output_field_equivalents == 9);
    REQUIRE(terrain.compression_history_field_equivalents == 25);
    REQUIRE(terrain.high_water_field_equivalents == 109);
}

TEST_CASE("terrain output reserves lossless float64 center and interface geometry",
          "[core][hardware][memory-preflight][terrain][TE-005]")
{
    RuntimeMemoryPhaseInputs inputs;
    inputs.solver_resident_fields = 75;
    inputs.output_enabled = true;
    inputs.volume_output = true;
    inputs.async_io = true;
    inputs.terrain_geometry = true;
    inputs.vertical_levels = 32;
    inputs.core_volume_output_fields = 8;
    inputs.slice_scratch_arrays = 0;

    const RuntimeMemoryPhaseProfile profile = estimate_runtime_memory_phases(inputs);
    // center float64 = 2 FP32 fields; (nz+1)-face float64 = 3 rounded fields.
    REQUIRE(profile.current_output_field_equivalents == 13);
    REQUIRE(profile.high_water_field_equivalents == 88);
}

TEST_CASE("slice output counts scratch without full-volume snapshots",
          "[core][hardware][memory-preflight][cpu017]")
{
    RuntimeMemoryPhaseInputs inputs;
    inputs.solver_resident_fields = 75;
    inputs.lazy_diagnostic_fields = 12;
    inputs.output_enabled = true;
    inputs.volume_output = false;
    inputs.c_grid = true;
    inputs.async_io = true;
    inputs.core_volume_output_fields = 9;
    inputs.derived_volume_output_fields = 6;
    inputs.horizontal_rows = 32;

    const RuntimeMemoryPhaseProfile profile =
        estimate_runtime_memory_phases(inputs);
    REQUIRE(profile.current_output_field_equivalents == 0);
    REQUIRE(profile.concurrent_snapshot_field_equivalents == 0);
    REQUIRE(profile.compression_history_field_equivalents == 0);
    REQUIRE(profile.high_water_field_equivalents == 92);
}

TEST_CASE("C-grid velocity staging follows selected output dependencies",
          "[core][hardware][memory-preflight][cpu016]")
{
    RuntimeMemoryPhaseInputs inputs;
    inputs.solver_resident_fields = 75;
    inputs.output_enabled = true;
    inputs.volume_output = true;
    inputs.c_grid = true;
    inputs.centered_velocity_fields = 0;
    inputs.slice_scratch_arrays = 0;

    const RuntimeMemoryPhaseProfile scalar_only =
        estimate_runtime_memory_phases(inputs);
    REQUIRE(scalar_only.centered_velocity_fields == 0);
    REQUIRE(scalar_only.high_water_field_equivalents == 75);

    inputs.centered_velocity_fields = 1;
    const RuntimeMemoryPhaseProfile one_component =
        estimate_runtime_memory_phases(inputs);
    REQUIRE(one_component.centered_velocity_fields == 1);
    REQUIRE(one_component.high_water_field_equivalents == 76);

    inputs.centered_velocity_fields = 3;
    const RuntimeMemoryPhaseProfile derived_products =
        estimate_runtime_memory_phases(inputs);
    REQUIRE(derived_products.centered_velocity_fields == 3);
    REQUIRE(derived_products.high_water_field_equivalents == 78);
}

TEST_CASE("centered velocity staging resolves core and derived dependencies",
          "[core][output][cpu016]")
{
    const std::unordered_set<std::string> theta_only = {"theta"};
    REQUIRE(centered_velocity_staging_plan(
        true, output_dependencies_for_fields(theta_only)).field_count() == 0);

    const std::unordered_set<std::string> v_only = {"v", "theta"};
    const CenteredVelocityStagingPlan v_plan =
        centered_velocity_staging_plan(
            true, output_dependencies_for_fields(v_only));
    REQUIRE_FALSE(v_plan.u);
    REQUIRE(v_plan.v);
    REQUIRE_FALSE(v_plan.w);
    REQUIRE(v_plan.field_count() == 1);

    const std::unordered_set<std::string> divergence_only = {"divergence"};
    const CenteredVelocityStagingPlan derived =
        centered_velocity_staging_plan(
            true, output_dependencies_for_fields(divergence_only));
    REQUIRE(derived.u);
    REQUIRE(derived.v);
    REQUIRE(derived.w);
    REQUIRE(derived.field_count() == 3);

    const std::unordered_set<std::string> storm_winds_only = {
        "storm_relative_winds"};
    const CenteredVelocityStagingPlan horizontal =
        centered_velocity_staging_plan(
            true, output_dependencies_for_fields(storm_winds_only));
    REQUIRE(horizontal.u);
    REQUIRE(horizontal.v);
    REQUIRE_FALSE(horizontal.w);

    REQUIRE(centered_velocity_staging_plan(
        false, centered_uvw_dependencies).field_count() == 0);
}

TEST_CASE("Mapped radiation allocation policy rejects oversized grids before allocation",
          "[core][hardware][memory-preflight][terrain]")
{
    HardwareInfo hw;
    hw.total_ram_bytes=48ULL<<30;
    constexpr int nx=384,ny=384,nz=128;
    const auto spectral=tmv::radiative_transfer::mapped_shortwave_workspace_bytes(
        static_cast<std::uint64_t>(nx-2)*(ny-2),nz,true,4,8);
    const auto without=assess_grid_memory(hw,nx,ny,nz,150);
    const auto with=assess_grid_memory(hw,nx,ny,nz,150,spectral);
    CHECK_FALSE(memory_admission_should_reject(without,true));
    CHECK(memory_admission_should_reject(with,true));
    CHECK(spectral>=static_cast<std::uint64_t>(nx-2)*(ny-2)*nz*(3*112+64)*sizeof(double));
    CHECK_THROWS_AS(tmv::radiative_transfer::mapped_shortwave_workspace_bytes(
        std::numeric_limits<std::uint64_t>::max(),2,true,4,8),std::overflow_error);
    CHECK(earth_fixed_remap_scratch_bytes(18,18,18)>terrain_geometry_memory_bytes(18,18,18));
    CHECK(earth_fixed_remap_scratch_bytes(36,18,18)>earth_fixed_remap_scratch_bytes(18,18,18));
    CHECK_THROWS_AS(earth_fixed_remap_scratch_bytes(std::numeric_limits<std::uint64_t>::max(),2,2),std::overflow_error);
}
