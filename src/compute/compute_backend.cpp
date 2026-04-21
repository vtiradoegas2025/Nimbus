/**
 * @file compute_backend.cpp
 * @brief Runtime compute backend selection and lifecycle orchestration.
 *
 * This core unit owns backend selection, config parsing helpers, and fallback
 * policy. Backend-specific implementations are split into focused translation
 * units (e.g., vulkan/src/compute_backend_vulkan.cpp).
 */

#include "compute/compute_backend.hpp"
#include "compute/compute_backend_factory.hpp"
#include "core/hardware_info.hpp"

#include "util/string_utils.hpp"

#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "core/simulation.hpp"

namespace
{

class CpuComputeBackend final : public ComputeBackend
{
public:
    std::string name() const override { return "cpu"; }

    bool initialize(std::string&) override
    {
        return true;
    }

    void shutdown() override {}
};

std::unique_ptr<ComputeBackend> runtime_compute_backend;
ComputeBackendKind runtime_compute_backend_kind = ComputeBackendKind::Cpu;
ComputeBackendKind runtime_compute_backend_requested_kind = ComputeBackendKind::Cpu;
bool runtime_compute_backend_fallback_active = false;

std::unique_ptr<ComputeBackend> create_compute_backend(ComputeBackendKind kind)
{
    if (kind == ComputeBackendKind::Vulkan)
    {
        return create_vulkan_compute_backend();
    }
    return create_cpu_compute_backend();
}

} // namespace

std::unique_ptr<ComputeBackend> create_cpu_compute_backend()
{
    return std::make_unique<CpuComputeBackend>();
}

ComputeBackendConfig global_compute_backend_config{};

const char* compute_backend_kind_name(ComputeBackendKind kind)
{
    switch (kind)
    {
        case ComputeBackendKind::Vulkan:
            return "vulkan";
        case ComputeBackendKind::Cpu:
        default:
            return "cpu";
    }
}

bool parse_compute_backend_kind(const std::string& value, ComputeBackendKind& out_kind)
{
    const std::string normalized = tmv::strutil::lower_copy(value);
    if (normalized == "cpu")
    {
        out_kind = ComputeBackendKind::Cpu;
        return true;
    }
    if (normalized == "vulkan" || normalized == "vk")
    {
        out_kind = ComputeBackendKind::Vulkan;
        return true;
    }
    return false;
}

std::vector<std::string> get_available_compute_backends()
{
    return {"cpu", "vulkan"};
}

bool initialize_compute_backend_runtime(std::string& error)
{
    shutdown_compute_backend_runtime();

    ComputeBackendKind requested_kind = ComputeBackendKind::Cpu;
    if (!parse_compute_backend_kind(global_compute_backend_config.backend, requested_kind))
    {
        error = "unknown numerics.compute.backend '" + global_compute_backend_config.backend +
            "' (supported: cpu, vulkan)";
        return false;
    }
    runtime_compute_backend_requested_kind = requested_kind;

    runtime_compute_backend = create_compute_backend(requested_kind);
    if (!runtime_compute_backend)
    {
        error = "failed to construct compute backend instance";
        return false;
    }

    std::string init_error;
    if (!runtime_compute_backend->initialize(init_error))
    {
        if (requested_kind != ComputeBackendKind::Cpu && global_compute_backend_config.allow_fallback)
        {
            if (log_normal_enabled())
            {
                std::cerr << "[COMPUTE] Requested backend '"
                          << compute_backend_kind_name(requested_kind)
                          << "' failed to initialize: " << init_error
                          << ". Falling back to CPU backend." << std::endl;
            }

            runtime_compute_backend = create_compute_backend(ComputeBackendKind::Cpu);
            if (!runtime_compute_backend)
            {
                error = "failed to construct fallback cpu backend";
                return false;
            }

            std::string fallback_error;
            if (!runtime_compute_backend->initialize(fallback_error))
            {
                error = "fallback cpu backend failed: " + fallback_error;
                runtime_compute_backend.reset();
                return false;
            }

            runtime_compute_backend_kind = ComputeBackendKind::Cpu;
            runtime_compute_backend_fallback_active = true;
        }
        else
        {
            error = init_error;
            runtime_compute_backend.reset();
            return false;
        }
    }
    else
    {
        runtime_compute_backend_kind = requested_kind;
        runtime_compute_backend_fallback_active = false;
    }

    // Detect and log hardware characteristics
    {
        HardwareInfo hw = detect_hardware();
        if (runtime_compute_backend)
        {
            runtime_compute_backend->populate_hardware_info(hw);
        }
        log_hardware_info(hw);
    }

    if (log_normal_enabled())
    {
        std::cout << "[COMPUTE] backend=" << compute_backend_kind_name(runtime_compute_backend_kind)
                  << ", requested=" << compute_backend_kind_name(requested_kind)
                  << ", device_index=" << global_compute_backend_config.device_index
                  << ", fallback=" << (runtime_compute_backend_fallback_active ? "active" : "off")
                  << ", parity=" << (global_compute_backend_config.validate_parity ? "on" : "off")
                  << std::endl;
    }

    return true;
}

void shutdown_compute_backend_runtime()
{
    if (runtime_compute_backend)
    {
        runtime_compute_backend->shutdown();
        runtime_compute_backend.reset();
    }
    runtime_compute_backend_kind = ComputeBackendKind::Cpu;
    runtime_compute_backend_requested_kind = ComputeBackendKind::Cpu;
    runtime_compute_backend_fallback_active = false;
}

const ComputeBackend* active_compute_backend()
{
    return runtime_compute_backend.get();
}

ComputeBackend* mutable_compute_backend()
{
    return runtime_compute_backend.get();
}

ComputeBackendKind active_compute_backend_kind()
{
    return runtime_compute_backend_kind;
}

const char* requested_compute_backend_name()
{
    return compute_backend_kind_name(runtime_compute_backend_requested_kind);
}

bool compute_backend_fallback_active()
{
    return runtime_compute_backend_fallback_active;
}
