#include "compute/bridge/residency.hpp"

#include "compute/compute_backend.hpp"

namespace
{
ComputeBackend* backend_for_residency(ComputeBackendRuntimeState& compute) noexcept
{
    return mutable_compute_backend(compute);
}
} // namespace

void begin_resident_step_backend()
{
    return begin_resident_step_backend(global_compute_backend_runtime_state);
}

void begin_resident_step_backend(ComputeBackendRuntimeState& compute)
{
    ComputeBackend* backend = backend_for_residency(compute);
    if (backend != nullptr)
    {
        backend->begin_resident_step();
    }
}

void invalidate_resident_field_backend(const char* field_id) noexcept
{
    return invalidate_resident_field_backend(global_compute_backend_runtime_state, field_id);
}

void invalidate_resident_field_backend(ComputeBackendRuntimeState& compute, const char* field_id) noexcept
{
    ComputeBackend* backend = backend_for_residency(compute);
    if (backend != nullptr && field_id != nullptr)
    {
        backend->invalidate_resident_field(field_id);
    }
}

void invalidate_all_resident_fields_backend() noexcept
{
    return invalidate_all_resident_fields_backend(global_compute_backend_runtime_state);
}

void invalidate_all_resident_fields_backend(ComputeBackendRuntimeState& compute) noexcept
{
    ComputeBackend* backend = backend_for_residency(compute);
    if (backend != nullptr)
    {
        backend->invalidate_all_resident_fields();
    }
}
