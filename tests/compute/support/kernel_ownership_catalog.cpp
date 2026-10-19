#include "compute/support/kernel_ownership_catalog.hpp"

#include "compute/kernel_manifest.hpp"
#include "compute/support/parity_registry.hpp"
#include "compute/internal/metal_source_catalog.hpp"

#include <cstring>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string_view>

namespace nimbus::parity
{
namespace
{

void append_json_string(std::ostringstream& out, std::string_view value)
{
    out << '"';
    for (const unsigned char c : value)
    {
        switch (c)
        {
        case '"': out << "\\\""; break;
        case '\\': out << "\\\\"; break;
        case '\n': out << "\\n"; break;
        case '\r': out << "\\r"; break;
        case '\t': out << "\\t"; break;
        default:
            if (c < 0x20)
                out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << static_cast<int>(c)
                    << std::dec << std::setfill(' ');
            else
                out << static_cast<char>(c);
            break;
        }
    }
    out << '"';
}

void append_nullable_string(std::ostringstream& out, const char* value)
{
    if (value == nullptr || value[0] == '\0')
        out << "null";
    else
        append_json_string(out, value);
}

const char* availability_name(compute::KernelAvailability availability)
{
    switch (availability)
    {
    case compute::KernelAvailability::Active: return "active";
    case compute::KernelAvailability::Deferred: return "deferred";
    case compute::KernelAvailability::Orphan: return "orphan";
    }
    return "invalid";
}

const char* horizon_name(HorizonRule horizon)
{
    return horizon == HorizonRule::AccumulatedSubsteps ? "accumulated_substeps" : "single_dispatch";
}

const char* replay_owner_name(ReplayOwner owner)
{
    switch (owner)
    {
    case ReplayOwner::RuntimeCallSite: return "runtime_call_site";
    case ReplayOwner::HardwareFixture: return "hardware_fixture";
    case ReplayOwner::RuntimeAndHardware: return "runtime_and_hardware";
    case ReplayOwner::Waived: return "waived";
    }
    return "invalid";
}

const char* required_hardware_name(RequiredHardware hardware)
{
    switch (hardware)
    {
    case RequiredHardware::None: return "none";
    case RequiredHardware::Vulkan: return "vulkan";
    case RequiredHardware::Metal: return "metal";
    case RequiredHardware::VulkanAndMetal: return "vulkan_and_metal";
    }
    return "invalid";
}

} // namespace

std::string serialize_kernel_ownership_catalog_json()
{
    const char* reason = nullptr;
    const auto& entries = registry();
    if (!validate_registry(entries, &reason))
        throw std::runtime_error(reason == nullptr ? "invalid parity registry" : reason);
    const auto& manifest = compute::kernel_manifest();
    if (entries.size() != manifest.size())
        throw std::runtime_error("kernel manifest and parity registry cardinality differ");

    std::ostringstream records;
    std::size_t active = 0;
    std::size_t deferred = 0;
    std::size_t orphan = 0;
    std::size_t runtime_covered = 0;
    std::size_t vulkan_hardware_covered = 0;
    std::size_t metal_hardware_covered = 0;
    for (std::size_t index = 0; index < manifest.size(); ++index)
    {
        const auto& kernel = manifest[index];
        const auto& parity = entries[index];
        if (std::strcmp(kernel.kernel_name, parity.shader) != 0 ||
            std::strcmp(kernel.cpu_reference, parity.cpu_reference) != 0)
            throw std::runtime_error("kernel manifest and parity registry identity differ");
        const bool is_active = kernel.availability == compute::KernelAvailability::Active;
        if (is_active)
            ++active;
        else if (kernel.availability == compute::KernelAvailability::Deferred)
            ++deferred;
        else
            ++orphan;
        if (parity.coverage.runtime_sample != nullptr)
            ++runtime_covered;
        if (parity.coverage.hardware_test != nullptr)
            ++vulkan_hardware_covered;
        if (parity.coverage.metal_hardware_test != nullptr)
            ++metal_hardware_covered;

        const auto* metal_library = metal::detail::metal_library_for(kernel.id);
        const char* metal_hash = metal::detail::metal_source_hash_for(kernel.id);
        if (records.tellp() > 0)
            records << ',';
        records << "{\"index\":" << index << ",\"operationId\":";
        append_json_string(records, kernel.operation_id);
        records << ",\"kernelName\":";
        append_json_string(records, kernel.kernel_name);
        records << ",\"availability\":";
        append_json_string(records, availability_name(kernel.availability));
        records << ",\"owners\":{\"cpu\":\"reference\",\"vulkan\":";
        append_json_string(records, is_active ? "device" : "unavailable");
        records << ",\"metal\":";
        append_json_string(records, is_active && metal_library != nullptr ? "device" : "unavailable");
        records << "},\"cpuReference\":";
        append_json_string(records, kernel.cpu_reference);
        records << ",\"vulkan\":{\"artifact\":";
        append_json_string(records, kernel.artifact_name);
        records << ",\"artifactFnv1a64\":";
        append_json_string(records, kernel.artifact_hash);
        records << ",\"hardwareTest\":";
        append_nullable_string(records, parity.coverage.hardware_test);
        records << "},\"metal\":{\"library\":";
        append_nullable_string(records, metal_library == nullptr ? nullptr : metal_library->label);
        records << ",\"sourceFnv1a64\":";
        append_nullable_string(records, metal_hash);
        records << ",\"preciseMathRequired\":"
                << (metal_library == nullptr ? "null" : metal_library->precise_math ? "true" : "false");
        records << ",\"hardwareTest\":";
        append_nullable_string(records, parity.coverage.metal_hardware_test);
        records << "},\"qualification\":{\"outputs\":";
        append_nullable_string(records, parity.coverage.outputs);
        records << ",\"runtimeSample\":";
        append_nullable_string(records, parity.coverage.runtime_sample);
        records << ",\"horizon\":";
        append_json_string(records, horizon_name(parity.coverage.horizon));
        records << ",\"replayOwner\":";
        append_json_string(records, replay_owner_name(parity.coverage.replay_owner));
        records << ",\"requiredHardware\":";
        append_json_string(records, required_hardware_name(parity.coverage.required_hardware));
        records << ",\"waiver\":";
        append_nullable_string(records, parity.coverage.waiver);
        records << "}}";
    }

    std::ostringstream out;
    out << "{\"schema\":\"nimbus.kernel-ownership-catalog.v1\",\"version\":1,"
           "\"records\":[" << records.str() << "],\"summary\":{\"operations\":" << manifest.size()
        << ",\"active\":" << active << ",\"deferred\":" << deferred << ",\"orphan\":" << orphan
        << ",\"runtimeCovered\":" << runtime_covered
        << ",\"vulkanHardwareCovered\":" << vulkan_hardware_covered
        << ",\"metalHardwareCovered\":" << metal_hardware_covered << "}}\n";
    return out.str();
}

} // namespace nimbus::parity
