# Native Metal compute backend

The Metal backend is intentionally organized by ownership rather than as one
Vulkan-shaped implementation file.

- `include/compute/metal/metal_compute_backend.hpp` is the canonical public
  Metal interface; `metal_backend_evidence.hpp` owns Metal-only dispatch,
  benchmark, manifest, and qualification records.
  `include/compute/metal_compute_backend.hpp` is a compatibility forwarder for
  older callers.
- `include/compute/metal/shaders/<family>/` contains the runtime-compiled MSL
  sources grouped by kernel family.
- `src/compute/metal_compute_backend.mm` is the Objective-C++ composition root.
- `src/compute/backend/*.inc` contains the focused implementation sections for
  lifecycle and individual kernel families. The `services/` and `dynamics/`
  subdirectories split larger families by contract and execution model.
- `src/compute/internal/` contains the small state-owning components used by
  the Objective-C++ composition root. Device/queue ownership, shared
  allocations, resident buffers, pipelines, and telemetry each have one
  focused owner. `MetalSourceCatalog` is the single declaration point for MSL
  libraries, hashes, entrypoints, and aliases. `validation/` contains the pure
  hardware-independent dispatch checks and ABI encoders.
- `../scripts/qualification/accelerators/qualify_metal.py` runs and seals the physical qualification; no
  Python tooling or cache state is owned by the Metal source directory.

The implementation fragments deliberately form one Objective-C++ translation
unit. They share a private `Impl`, validation helpers, Metal object ownership,
and resident-store transaction machinery without exporting those details or
duplicating them across translation units. `Impl` is a narrow composition root
for device, allocation, residency, pipeline, and telemetry components rather
than a parallel collection of native objects and counters. Add a new kernel to
its family fragment and shader directory, then register its library and
entrypoint once in `metal_source_catalog.hpp`; lifecycle compilation, public
source accessors, readiness, and evidence all consume that catalog. The
qualification sealer recursively
binds every Metal header, `.mm` root, and implementation fragment, so an
unrecorded source file fails the evidence gate.
