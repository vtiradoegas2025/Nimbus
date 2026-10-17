#pragma once

#include "compute/kernel_startup_plan.hpp"

#include <iomanip>
#include <sstream>
#include <string>

namespace tmv_vulkan
{

inline std::string pipeline_cache_manifest_signature(const nimbus::compute::KernelManifest&    manifest,
                                                     const nimbus::compute::KernelStartupPlan* plan = nullptr)
{
    std::ostringstream canonical;
    for (const auto& kernel : manifest)
    {
        const auto string_or_empty = [](const char* value) { return value != nullptr ? value : ""; };
        canonical << static_cast<unsigned int>(kernel.id) << '\x1f' << string_or_empty(kernel.operation_id) << '\x1f'
                  << string_or_empty(kernel.kernel_name) << '\x1f' << string_or_empty(kernel.artifact_name) << '\x1f'
                  << string_or_empty(kernel.artifact_hash) << '\x1f'
                  << static_cast<unsigned int>(kernel.resources.descriptor_span) << '\x1f' << kernel.resources.read_mask
                  << '\x1f' << kernel.resources.write_mask << '\x1f' << static_cast<unsigned int>(kernel.shape)
                  << '\x1f' << kernel.constants.bytes << '\x1f' << static_cast<unsigned int>(kernel.constants.alignment)
                  << '\x1f' << static_cast<unsigned int>(kernel.constants.version) << '\x1f'
                  << static_cast<unsigned int>(kernel.precision) << '\x1f'
                  << static_cast<unsigned int>(kernel.cpu_fallback) << '\x1f' << string_or_empty(kernel.cpu_reference)
                  << '\x1f' << static_cast<unsigned int>(kernel.parity_tier) << '\x1f'
                  << static_cast<unsigned int>(kernel.comparison_domain) << '\x1f'
                  << static_cast<unsigned int>(kernel.coordinate) << '\x1f'
                  << static_cast<unsigned int>(kernel.availability) << '\x1f' << (kernel.startup_required ? 1 : 0)
                  << '\x1e';
    }
    if (plan != nullptr)
    {
        canonical << '\x1d' << plan->cache_identity();
    }
    std::uint64_t value = 14695981039346656037ULL;
    for (const unsigned char byte : canonical.str())
    {
        value ^= byte;
        value *= 1099511628211ULL;
    }
    std::ostringstream signature;
    signature << std::hex << std::setfill('0') << std::setw(16) << value;
    return signature.str();
}

} // namespace tmv_vulkan
