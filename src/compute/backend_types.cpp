#include "compute/backend_types.hpp"

#include "util/string_utils.hpp"

const char* compute_backend_kind_name(ComputeBackendKind kind)
{
    switch (kind)
    {
        case ComputeBackendKind::Cpu:
            return "cpu";
        case ComputeBackendKind::Vulkan:
            return "vulkan";
        case ComputeBackendKind::Metal:
            return "metal";
    }
    return "unknown";
}

bool parse_compute_backend_kind(
    const std::string& value, ComputeBackendKind& out_kind)
{
    const std::string normalized = tmv::strutil::lower_copy(value);
    if (normalized == "cpu")
    {
        out_kind = ComputeBackendKind::Cpu;
        return true;
    }
    // "gpu" remains the v1 schema's backward-compatible spelling for Vulkan.
    // Metal is deliberately explicit so existing GPU configurations do not
    // change backend when moved between operating systems.
    if (normalized == "vulkan" || normalized == "vk" || normalized == "gpu")
    {
        out_kind = ComputeBackendKind::Vulkan;
        return true;
    }
    if (normalized == "metal" || normalized == "mtl")
    {
        out_kind = ComputeBackendKind::Metal;
        return true;
    }
    return false;
}
