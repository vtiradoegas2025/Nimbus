// Backend selection and lifecycle orchestration. Backend implementations stay
// in their platform-specific translation units.

#include "compute/backend_runtime.hpp"
#include "compute/compute_backend_factory.hpp"
#include "core/infra/hardware_info.hpp"

#include "util/log.hpp"
#include <memory>
#include <string>
#include <vector>

#include "core/runtime/runtime_config.hpp"

namespace
{

void shutdown_and_reset_backend(
    std::unique_ptr<ComputeBackend>& backend) noexcept
{
    if (!backend)
    {
        return;
    }

    try
    {
        backend->shutdown();
    }
    catch (...)
    {
        // A failed or partially initialized backend must never make runtime
        // teardown throw. Its destructor remains the final cleanup boundary.
    }
    backend.reset();
}

#ifdef NIMBUS_CPU_ONLY
class CpuOnlyUnavailableVulkanBackend final : public ComputeBackend
{
public:
    std::string name() const override { return "vulkan-unavailable-cpu-only"; }

    bool initialize(std::string& error) override
    {
        error = "this artifact was built with CPU_ONLY=1";
        return false;
    }

    void shutdown() override {}
};
#endif

#ifndef NIMBUS_ENABLE_METAL
class UnavailableMetalBackend final : public ComputeBackend
{
public:
    std::string name() const override { return "metal-unavailable"; }

    bool initialize(std::string& error) override
    {
#ifdef NIMBUS_CPU_ONLY
        error = "this artifact was built with CPU_ONLY=1";
#elif defined(__APPLE__)
        error = "this artifact was built without the native Metal backend";
#else
        error = "the native Metal backend requires macOS";
#endif
        return false;
    }

    void shutdown() override {}
};
#endif

std::unique_ptr<ComputeBackend> create_compute_backend(ComputeBackendKind kind)
{
    if (kind == ComputeBackendKind::Vulkan)
    {
        return create_vulkan_compute_backend();
    }
    if (kind == ComputeBackendKind::Metal)
    {
        return create_metal_compute_backend();
    }
    return create_cpu_compute_backend();
}

} // namespace

#ifdef NIMBUS_CPU_ONLY
std::unique_ptr<ComputeBackend> create_vulkan_compute_backend()
{
    return std::make_unique<CpuOnlyUnavailableVulkanBackend>();
}
#endif

#ifndef NIMBUS_ENABLE_METAL
std::unique_ptr<ComputeBackend> create_metal_compute_backend()
{
    return std::make_unique<UnavailableMetalBackend>();
}
#endif

std::vector<std::string> get_available_compute_backends()
{
#ifdef NIMBUS_CPU_ONLY
    return {"cpu"};
#elif defined(NIMBUS_ENABLE_METAL)
    return {"cpu", "vulkan", "metal"};
#else
    return {"cpu", "vulkan"};
#endif
}

bool initialize_compute_backend_runtime(std::string& error)
{
    return initialize_compute_backend_runtime(
        global_compute_backend_runtime_state, global_coordinate_system, error);
}

bool initialize_compute_backend_runtime(ComputeBackendRuntimeState& runtime,
                                        CoordinateSystem coordinate_system,
                                        std::string& error)
{
    error.clear();
    shutdown_compute_backend_runtime(runtime);

    ComputeBackendKind requested_kind = ComputeBackendKind::Cpu;
    if (!parse_compute_backend_kind(runtime.config.backend, requested_kind))
    {
        error = "unknown numerics.compute.backend '" + runtime.config.backend +
            "' (supported: cpu, vulkan, metal)";
        return false;
    }
    runtime.requested_kind = requested_kind;
    if (runtime.config.cuda_assist && requested_kind != ComputeBackendKind::Vulkan)
    {
        error = "CUDA assist requires the Vulkan backend on a compatible NVIDIA GPU";
        return false;
    }

    runtime.backend = create_compute_backend(requested_kind);
    if (!runtime.backend)
    {
        error = "failed to construct compute backend instance";
        return false;
    }
    runtime.backend->configure(runtime.config);

    std::string init_error;
    if (!runtime.backend->initialize(init_error))
    {
        shutdown_and_reset_backend(runtime.backend);
        if (requested_kind != ComputeBackendKind::Cpu && runtime.config.allow_fallback)
        {
            tmv::log_warn("[COMPUTE] Requested backend '",
                          compute_backend_kind_name(requested_kind),
                          "' failed to initialize: ", init_error,
                          ". Falling back to CPU backend.");

            runtime.backend = create_compute_backend(ComputeBackendKind::Cpu);
            if (!runtime.backend)
            {
                error = "failed to construct fallback cpu backend";
                return false;
            }
            runtime.backend->configure(runtime.config);

            std::string fallback_error;
            if (!runtime.backend->initialize(fallback_error))
            {
                error = "fallback cpu backend failed: " + fallback_error;
                shutdown_and_reset_backend(runtime.backend);
                return false;
            }

            runtime.active_kind = ComputeBackendKind::Cpu;
            runtime.fallback_active = true;
        }
        else
        {
            error = init_error;
            return false;
        }
    }
    else
    {
        runtime.active_kind = requested_kind;
        runtime.fallback_active = false;
    }

    // Detect and log hardware characteristics
    {
        HardwareInfo hw = detect_hardware();
        if (runtime.backend)
        {
            runtime.backend->populate_hardware_info(hw);
        }
        log_hardware_info(hw);
    }

    // Inform backend of the active coordinate system for shader routing
    if (runtime.backend)
    {
        runtime.backend->set_coordinate_system(coordinate_system);
    }

    tmv::log_info("[COMPUTE] backend=", compute_backend_kind_name(runtime.active_kind),
                  ", requested=", compute_backend_kind_name(requested_kind),
                  ", device_index=", runtime.config.device_index,
                  ", fallback=", runtime.fallback_active ? "active" : "off",
                  ", fallback_allowed=", runtime.config.allow_fallback ? "on" : "off",
                  ", parity=", runtime.config.validate_parity ? "on" : "off");

    return true;
}

void shutdown_compute_backend_runtime()
{
    shutdown_compute_backend_runtime(global_compute_backend_runtime_state);
}

void shutdown_compute_backend_runtime(ComputeBackendRuntimeState& runtime)
{
    runtime.reset();
}

const ComputeBackend* active_compute_backend()
{
    return active_compute_backend(global_compute_backend_runtime_state);
}

const ComputeBackend* active_compute_backend(
    const ComputeBackendRuntimeState& runtime)
{
    return runtime.backend.get();
}

ComputeBackend* mutable_compute_backend()
{
    return mutable_compute_backend(global_compute_backend_runtime_state);
}

ComputeBackend* mutable_compute_backend(ComputeBackendRuntimeState& runtime)
{
    return runtime.backend.get();
}

ComputeBackendKind active_compute_backend_kind()
{
    return active_compute_backend_kind(global_compute_backend_runtime_state);
}

ComputeBackendKind active_compute_backend_kind(
    const ComputeBackendRuntimeState& runtime)
{
    return runtime.active_kind;
}

const char* requested_compute_backend_name()
{
    return requested_compute_backend_name(global_compute_backend_runtime_state);
}

const char* requested_compute_backend_name(
    const ComputeBackendRuntimeState& runtime)
{
    return compute_backend_kind_name(runtime.requested_kind);
}

bool compute_backend_fallback_active()
{
    return compute_backend_fallback_active(global_compute_backend_runtime_state);
}

bool compute_backend_fallback_active(
    const ComputeBackendRuntimeState& runtime)
{
    return runtime.fallback_active;
}

bool active_compute_backend_supports(AdvectionKernelOp op)
{
    const ComputeBackend* backend = active_compute_backend();
    return backend != nullptr && backend->supports_advection(op);
}

bool active_compute_backend_supports(DynamicsKernelOp op)
{
    const ComputeBackend* backend = active_compute_backend();
    return backend != nullptr && backend->supports_dynamics(op);
}

bool active_compute_backend_supports(MicrophysicsKernelOp op)
{
    const ComputeBackend* backend = active_compute_backend();
    return backend != nullptr && backend->supports_microphysics(op);
}

bool active_compute_backend_supports(TurbulenceKernelOp op)
{
    const ComputeBackend* backend = active_compute_backend();
    return backend != nullptr && backend->supports_turbulence(op);
}

bool active_compute_backend_supports(DiffusionKernelOp op)
{
    const ComputeBackend* backend = active_compute_backend();
    return backend != nullptr && backend->supports_diffusion(op);
}

bool active_compute_backend_supports_supplied_cartesian_cgrid_mass_flux()
{
    const ComputeBackend* backend = active_compute_backend();
    return backend != nullptr
        && backend->supports_supplied_cartesian_cgrid_mass_flux();
}

bool active_compute_backend_supports_batched_cartesian_cgrid_mass_flux_export()
{
    const ComputeBackend* backend = active_compute_backend();
    return backend != nullptr
        && backend->supports_batched_cartesian_cgrid_mass_flux_export();
}
