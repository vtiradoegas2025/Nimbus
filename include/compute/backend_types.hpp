#pragma once

#include <cstdint>
#include <memory>
#include <string>

namespace nimbus::compute
{
class KernelStartupPlan;
}

// Lightweight value types shared by backend selection, memory admission, and
// runtime ownership. Keep this header independent of kernel interfaces so
// callers that only plan backend resources do not inherit dispatch details.

struct ComputeMemoryProperties
{
    bool accelerator = false;
    bool unified_host_device_memory = false;
    std::uint64_t device_local_heap_bytes = 0;
    std::uint64_t max_storage_buffer_bytes = 0;
    // Remaining process budget is a live estimate, not reserved memory.
    bool device_budget_is_live = false;
    std::uint64_t device_available_budget_bytes = 0;
};

enum class ComputeBackendKind
{
    Cpu,
    Vulkan,
    Metal
};

enum class ComputeResidencyScope : std::uint8_t
{
    Root = 0,
    Child = 1
};

struct ComputeBackendConfig
{
    std::string backend = "cpu";
    int device_index = -1;
    bool allow_fallback = true;
    bool validate_parity = false;
    std::shared_ptr<const nimbus::compute::KernelStartupPlan> startup_plan;
    bool cuda_assist = false;
};

// Returns a stable configuration/logging name. Invalid enum values are
// reported explicitly rather than being mislabeled as the CPU backend.
const char* compute_backend_kind_name(ComputeBackendKind kind);

// Parses the supported backend names and aliases. On failure, out_kind is
// unchanged so callers can preserve their prior/default selection.
bool parse_compute_backend_kind(
    const std::string& value, ComputeBackendKind& out_kind);
