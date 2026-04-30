# GPU Compute Backend

Runtime compute backend selection, GPU buffer management, and kernel dispatch infrastructure.

## Why No Factory Pattern

The compute backend uses a `ComputeBackend` virtual interface rather than the `SchemeFactory` pattern used by physics modules. This is because GPU backends have fundamentally different lifecycle requirements:

- **Buffer pools** with acquire/release slot management
- **Shader pipeline compilation** at initialization
- **Descriptor set binding** per dispatch
- **Memory transfer choreography** (host-to-device, device-to-host, staging buffers)
- **Fence synchronization** between CPU and GPU

These concerns don't fit the simple "create scheme, call compute()" pattern. Instead, the backend exposes targeted dispatch methods for each GPU-accelerated kernel (advection, acoustic substeps, microphysics, diffusion).

## Layout

```
src/compute/
  compute_backend.cpp              Backend selection, lifecycle, fallback policy
  compute_kernel_template.cpp      Bridge functions: dynamics layer -> backend dispatch

vulkan/src/compute/
  compute_backend_vulkan.cpp       Vulkan implementation (pipelines, descriptors, dispatch)

include/compute/
  compute_backend.hpp              ComputeBackend virtual interface + ComputeBackendConfig
  compute_backend_factory.hpp      Forward declaration for Vulkan backend construction
  compute_kernel_template.hpp      Bridge function declarations
```

## Backend Selection

Configured via YAML:

```yaml
numerics:
  compute:
    backend: vulkan    # or "cpu"
    device_index: 0    # GPU device selection (0 = auto)
    allow_fallback: true
```

When `allow_fallback` is true and the requested backend fails to initialize, the runtime silently falls back to the CPU backend and logs a warning.

## Dispatch Architecture

The bridge layer (`compute_kernel_template.cpp`) provides flat C++ functions that the dynamics/physics orchestration calls. Each bridge function checks whether the active backend supports the operation, dispatches to it, and returns false if unsupported (letting the caller fall back to the CPU path).

```
dynamics.cpp
  -> dispatch_advection_batch_backend(...)        # bridge function
    -> mutable_compute_backend()                  # get active backend
      -> VulkanComputeBackend::dispatch_advection_batch(...)  # GPU dispatch
```

## GPU Kernels

The Vulkan backend dispatches 15 compute shaders covering:

- **Advection:** radial, azimuthal, x, y, vertical flux (TVD)
- **Acoustic substeps:** pressure + momentum (cylindrical and Cartesian variants)
- **Dynamics tendencies:** tornado, supercell, Cartesian
- **Microphysics:** Kessler pointwise + sedimentation
- **Diffusion:** isotropic Laplacian

Shader source is in `vulkan/shaders/compute/`. Pre-compiled SPIR-V binaries are tracked in the repository. Running `make` auto-recompiles any shaders whose source has changed.

## Coordinate System Routing

The backend tracks the active coordinate system (`CoordinateSystem::Cylindrical` or `CoordinateSystem::Cartesian`) and routes acoustic substep dispatches to the appropriate shader variant. This is set once at initialization via `set_coordinate_system()`.
