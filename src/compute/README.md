# Compute execution contracts

This module owns CPU/Vulkan/Metal selection, operation dispatch, buffer
ownership and resource admission. Scientific modules describe the operation;
backend implementations must satisfy its declared domain, representation,
failure and synchronization contracts. See
[architecture](../../docs/ARCHITECTURE.pdf) and
[mathematics](../../docs/MATHEMATICS.pdf).

`compute_backend.cpp` owns backend lifecycle and fallback policy.
`resident_field_store.cpp` and the residency interfaces track authoritative
host/device versions. `dispatch` contains the per-family bridges; `cpu` contains
host implementations. Vulkan and Metal implementations live in their respective
repository trees. `kernel_manifest.cpp` is the canonical operation/ABI inventory.

Unified memory does not imply that every host read is synchronized, and discrete
memory does not authorize an unrecorded CPU substitution. Native-required
profiles must prove actual dispatch ownership. Requested backend, active backend,
resource allocation and completion are distinct facts.

Capability and memory admission depend on the actual device and required
working set. They must not encode one workstation's vendor, core count or
memory capacity as a product default. Portable CPU builds and native device
profiles have separate verification obligations; no complete hardware
optimization claim follows from a successful compilation.

Parity comparison is distinct from readiness or full qualification. A kernel
can exist without owning a complete admitted runtime path. Use the compiled
readiness registry and the maintained tests under `tests/compute`,
`tests/vulkan`, `tests/metal` and the coupled module suites. Keep all scientific
comparison thresholds intact when changing memory movement or batching.
