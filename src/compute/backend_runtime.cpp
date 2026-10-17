#include "compute/backend_runtime.hpp"

#include <utility>

ComputeBackendRuntimeState::ComputeBackendRuntimeState() = default;

ComputeBackendRuntimeState::ComputeBackendRuntimeState(
    const ComputeBackendRuntimeState& other)
    : config(other.config), vertical_flux_template_id(other.vertical_flux_template_id), requested_kind(other.requested_kind)
{
}

ComputeBackendRuntimeState& ComputeBackendRuntimeState::operator=(
    const ComputeBackendRuntimeState& other)
{
    if (this == &other)
    {
        return *this;
    }
    reset();
    config = other.config;
    vertical_flux_template_id = other.vertical_flux_template_id;
    requested_kind = other.requested_kind;
    return *this;
}

ComputeBackendRuntimeState::ComputeBackendRuntimeState(
    ComputeBackendRuntimeState&& other) noexcept
    : config(std::move(other.config)), vertical_flux_template_id(std::move(other.vertical_flux_template_id)),
      backend(std::move(other.backend)),
      active_kind(other.active_kind), requested_kind(other.requested_kind),
      fallback_active(other.fallback_active)
{
    other.active_kind = ComputeBackendKind::Cpu;
    other.requested_kind = ComputeBackendKind::Cpu;
    other.fallback_active = false;
}

ComputeBackendRuntimeState& ComputeBackendRuntimeState::operator=(
    ComputeBackendRuntimeState&& other) noexcept
{
    if (this == &other)
    {
        return *this;
    }
    reset();
    config = std::move(other.config);
    vertical_flux_template_id = std::move(other.vertical_flux_template_id);
    backend = std::move(other.backend);
    active_kind = other.active_kind;
    requested_kind = other.requested_kind;
    fallback_active = other.fallback_active;
    other.active_kind = ComputeBackendKind::Cpu;
    other.requested_kind = ComputeBackendKind::Cpu;
    other.fallback_active = false;
    return *this;
}

ComputeBackendRuntimeState::~ComputeBackendRuntimeState()
{
    reset();
}

void ComputeBackendRuntimeState::reset() noexcept
{
    if (backend)
    {
        try
        {
            backend->shutdown();
        }
        catch (...)
        {
            // Destruction and checkpoint replacement must remain noexcept.
        }
        backend.reset();
    }
    active_kind = ComputeBackendKind::Cpu;
    requested_kind = ComputeBackendKind::Cpu;
    fallback_active = false;
}
