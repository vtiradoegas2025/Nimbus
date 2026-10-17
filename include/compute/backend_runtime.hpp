#pragma once

#include "compute/backend_interface.hpp"

#include <memory>
#include <string>
#include <vector>

// Owns backend selection and the live device/CPU backend for one simulation.
// Copies retain configuration but never duplicate hardware resources; the
// copied runtime is dormant until initialize_compute_backend_runtime is called.
struct ComputeBackendRuntimeState
{
    ComputeBackendConfig config;
    std::string vertical_flux_template_id = "tvd_vertical_flux_v2";
    std::unique_ptr<ComputeBackend> backend;
    ComputeBackendKind active_kind = ComputeBackendKind::Cpu;
    ComputeBackendKind requested_kind = ComputeBackendKind::Cpu;
    bool fallback_active = false;

    ComputeBackendRuntimeState();
    ComputeBackendRuntimeState(const ComputeBackendRuntimeState& other);
    ComputeBackendRuntimeState& operator=(
        const ComputeBackendRuntimeState& other);
    ComputeBackendRuntimeState(ComputeBackendRuntimeState&& other) noexcept;
    ComputeBackendRuntimeState& operator=(
        ComputeBackendRuntimeState&& other) noexcept;
    ~ComputeBackendRuntimeState();

    void reset() noexcept;
};

extern ComputeBackendRuntimeState& global_compute_backend_runtime_state;
extern ComputeBackendConfig& global_compute_backend_config;

std::vector<std::string> get_available_compute_backends();

// Initializes the configured backend, applying its fallback policy.
bool initialize_compute_backend_runtime(std::string& error);
bool initialize_compute_backend_runtime(ComputeBackendRuntimeState& runtime,
                                        CoordinateSystem coordinate_system,
                                        std::string& error);

void shutdown_compute_backend_runtime();
void shutdown_compute_backend_runtime(ComputeBackendRuntimeState& runtime);

const ComputeBackend* active_compute_backend();
const ComputeBackend* active_compute_backend(
    const ComputeBackendRuntimeState& runtime);

ComputeBackend* mutable_compute_backend();
ComputeBackend* mutable_compute_backend(ComputeBackendRuntimeState& runtime);

ComputeBackendKind active_compute_backend_kind();
ComputeBackendKind active_compute_backend_kind(
    const ComputeBackendRuntimeState& runtime);

const char* requested_compute_backend_name();
const char* requested_compute_backend_name(
    const ComputeBackendRuntimeState& runtime);

bool compute_backend_fallback_active();
bool compute_backend_fallback_active(
    const ComputeBackendRuntimeState& runtime);

inline bool active_compute_backend_supports(const ComputeBackendRuntimeState& runtime, AdvectionKernelOp op)
{ return runtime.backend && runtime.backend->supports_advection(op); }
inline bool active_compute_backend_supports(const ComputeBackendRuntimeState& runtime, DynamicsKernelOp op)
{ return runtime.backend && runtime.backend->supports_dynamics(op); }
inline bool active_compute_backend_supports(const ComputeBackendRuntimeState& runtime, MicrophysicsKernelOp op)
{ return runtime.backend && runtime.backend->supports_microphysics(op); }
inline bool active_compute_backend_supports(const ComputeBackendRuntimeState& runtime, TurbulenceKernelOp op)
{ return runtime.backend && runtime.backend->supports_turbulence(op); }
inline bool active_compute_backend_supports(const ComputeBackendRuntimeState& runtime, DiffusionKernelOp op)
{ return runtime.backend && runtime.backend->supports_diffusion(op); }
inline bool active_compute_backend_supports_supplied_cartesian_cgrid_mass_flux(const ComputeBackendRuntimeState& runtime)
{ return runtime.backend && runtime.backend->supports_supplied_cartesian_cgrid_mass_flux(); }
inline bool active_compute_backend_supports_batched_cartesian_cgrid_mass_flux_export(const ComputeBackendRuntimeState& runtime)
{ return runtime.backend && runtime.backend->supports_batched_cartesian_cgrid_mass_flux_export(); }

// Active-backend forms used by hot orchestration paths.
bool active_compute_backend_supports(AdvectionKernelOp op);
bool active_compute_backend_supports(DynamicsKernelOp op);
bool active_compute_backend_supports(MicrophysicsKernelOp op);
bool active_compute_backend_supports(TurbulenceKernelOp op);
bool active_compute_backend_supports(DiffusionKernelOp op);
bool active_compute_backend_supports_supplied_cartesian_cgrid_mass_flux();
bool active_compute_backend_supports_batched_cartesian_cgrid_mass_flux_export();

