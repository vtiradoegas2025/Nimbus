#include "core/infra/allocation_telemetry.hpp"

#include <array>
#include <atomic>
#include <cstdlib>
#include <new>

namespace
{

std::atomic<bool> measurement_active{false};
std::atomic<std::uint64_t> heap_allocations{0};
std::atomic<std::uint64_t> heap_requested_bytes{0};
std::atomic<std::uint64_t> physics_field_allocations{0};
std::atomic<std::uint64_t> physics_field_allocated_bytes{0};
constexpr std::size_t component_count =
    static_cast<std::size_t>(tmv::allocation_telemetry::Component::Count);
std::atomic<tmv::allocation_telemetry::Component> active_component{
    tmv::allocation_telemetry::Component::Unattributed};
std::array<std::atomic<std::uint64_t>, component_count>
    component_heap_allocations{};
std::array<std::atomic<std::uint64_t>, component_count>
    component_heap_requested_bytes{};
std::array<std::atomic<std::uint64_t>, component_count>
    component_physics_field_allocations{};
std::array<std::atomic<std::uint64_t>, component_count>
    component_physics_field_allocated_bytes{};
std::array<
    std::array<
        std::atomic<std::uint64_t>,
        tmv::allocation_telemetry::maximum_exact_heap_size_bytes + 1>,
    component_count> component_exact_heap_sizes{};

std::size_t component_index(
    tmv::allocation_telemetry::Component component) noexcept
{
    const std::size_t index = static_cast<std::size_t>(component);
    return index < component_count ? index : 0;
}

void* allocate_unaligned(std::size_t bytes)
{
    const std::size_t requested = bytes == 0 ? 1 : bytes;
    void* pointer = std::malloc(requested);
    if (!pointer)
        throw std::bad_alloc();
    tmv::allocation_telemetry::record_heap_allocation(requested);
    return pointer;
}

void* allocate_aligned(std::size_t bytes, std::size_t alignment)
{
    const std::size_t requested = bytes == 0 ? 1 : bytes;
    void* pointer = nullptr;
    if (posix_memalign(&pointer, alignment, requested) != 0 || !pointer)
        throw std::bad_alloc();
    tmv::allocation_telemetry::record_heap_allocation(requested);
    return pointer;
}

} // namespace

namespace tmv::allocation_telemetry
{

void begin_measurement() noexcept
{
    measurement_active.store(false, std::memory_order_relaxed);
    heap_allocations.store(0, std::memory_order_relaxed);
    heap_requested_bytes.store(0, std::memory_order_relaxed);
    physics_field_allocations.store(0, std::memory_order_relaxed);
    physics_field_allocated_bytes.store(0, std::memory_order_relaxed);
    active_component.store(Component::Unattributed, std::memory_order_relaxed);
    for (std::size_t i = 0; i < component_count; ++i)
    {
        component_heap_allocations[i].store(0, std::memory_order_relaxed);
        component_heap_requested_bytes[i].store(0, std::memory_order_relaxed);
        component_physics_field_allocations[i].store(
            0, std::memory_order_relaxed);
        component_physics_field_allocated_bytes[i].store(
            0, std::memory_order_relaxed);
        for (auto& count : component_exact_heap_sizes[i])
            count.store(0, std::memory_order_relaxed);
    }
    measurement_active.store(true, std::memory_order_release);
}

Snapshot end_measurement() noexcept
{
    measurement_active.store(false, std::memory_order_release);
    return current_snapshot();
}

Snapshot current_snapshot() noexcept
{
    Snapshot snapshot;
    snapshot.heap_allocations =
        heap_allocations.load(std::memory_order_relaxed);
    snapshot.heap_requested_bytes =
        heap_requested_bytes.load(std::memory_order_relaxed);
    snapshot.physics_field_allocations =
        physics_field_allocations.load(std::memory_order_relaxed);
    snapshot.physics_field_allocated_bytes =
        physics_field_allocated_bytes.load(std::memory_order_relaxed);
    for (std::size_t i = 0; i < component_count; ++i)
    {
        snapshot.components[i] = {
            component_heap_allocations[i].load(std::memory_order_relaxed),
            component_heap_requested_bytes[i].load(std::memory_order_relaxed),
            component_physics_field_allocations[i].load(
                std::memory_order_relaxed),
            component_physics_field_allocated_bytes[i].load(
                std::memory_order_relaxed),
        };
    }
    return snapshot;
}

void record_heap_allocation(std::size_t bytes) noexcept
{
    if (!measurement_active.load(std::memory_order_acquire))
        return;
    heap_allocations.fetch_add(1, std::memory_order_relaxed);
    heap_requested_bytes.fetch_add(
        static_cast<std::uint64_t>(bytes), std::memory_order_relaxed);
    const std::size_t index = component_index(
        active_component.load(std::memory_order_relaxed));
    component_heap_allocations[index].fetch_add(
        1, std::memory_order_relaxed);
    component_heap_requested_bytes[index].fetch_add(
        static_cast<std::uint64_t>(bytes), std::memory_order_relaxed);
    if (bytes <= maximum_exact_heap_size_bytes)
    {
        component_exact_heap_sizes[index][bytes].fetch_add(
            1, std::memory_order_relaxed);
    }
}

void record_physics_field_allocation(std::size_t bytes) noexcept
{
    if (!measurement_active.load(std::memory_order_acquire))
        return;
    physics_field_allocations.fetch_add(1, std::memory_order_relaxed);
    physics_field_allocated_bytes.fetch_add(
        static_cast<std::uint64_t>(bytes), std::memory_order_relaxed);
    const std::size_t index = component_index(
        active_component.load(std::memory_order_relaxed));
    component_physics_field_allocations[index].fetch_add(
        1, std::memory_order_relaxed);
    component_physics_field_allocated_bytes[index].fetch_add(
        static_cast<std::uint64_t>(bytes), std::memory_order_relaxed);
}

const char* component_name(Component component) noexcept
{
    switch (component)
    {
        case Component::Microphysics:
            return "microphysics";
        case Component::MicrophysicsPreflight:
            return "microphysics_preflight";
        case Component::MicrophysicsTransaction:
            return "microphysics_transaction";
        case Component::MicrophysicsTendencies:
            return "microphysics_tendencies";
        case Component::MicrophysicsApply:
            return "microphysics_apply";
        case Component::RuntimeTimestep:
            return "runtime_timestep";
        case Component::IntegrityValidation:
            return "integrity_validation";
        case Component::DynamicsCore:
            return "dynamics_core";
        case Component::Advection:
            return "advection";
        case Component::MicrophysicsTransport:
            return "microphysics_transport";
        case Component::Turbulence:
            return "turbulence";
        case Component::Diffusion:
            return "diffusion";
        case Component::Eos:
            return "eos";
        case Component::Sponge:
            return "sponge";
        case Component::FinalBoundary:
            return "final_boundary";
        case Component::Unattributed:
            return "unattributed";
        case Component::Count:
            break;
    }
    return "unknown";
}

std::uint64_t heap_size_count(
    Component component, std::size_t bytes) noexcept
{
    if (bytes > maximum_exact_heap_size_bytes)
        return 0;
    return component_exact_heap_sizes[component_index(component)][bytes].load(
        std::memory_order_relaxed);
}

ScopedComponent::ScopedComponent(Component component) noexcept
    : previous_(active_component.exchange(
          component, std::memory_order_acq_rel))
{
}

ScopedComponent::~ScopedComponent()
{
    restore();
}

void ScopedComponent::restore() noexcept
{
    if (!active_)
        return;
    active_component.store(previous_, std::memory_order_release);
    active_ = false;
}

} // namespace tmv::allocation_telemetry

void* operator new(std::size_t bytes)
{
    return allocate_unaligned(bytes);
}

void* operator new[](std::size_t bytes)
{
    return allocate_unaligned(bytes);
}

void operator delete(void* pointer) noexcept
{
    std::free(pointer);
}

void operator delete[](void* pointer) noexcept
{
    std::free(pointer);
}

void operator delete(void* pointer, std::size_t) noexcept
{
    std::free(pointer);
}

void operator delete[](void* pointer, std::size_t) noexcept
{
    std::free(pointer);
}

void* operator new(std::size_t bytes, std::align_val_t alignment)
{
    return allocate_aligned(bytes, static_cast<std::size_t>(alignment));
}

void* operator new[](std::size_t bytes, std::align_val_t alignment)
{
    return allocate_aligned(bytes, static_cast<std::size_t>(alignment));
}

void operator delete(void* pointer, std::align_val_t) noexcept
{
    std::free(pointer);
}

void operator delete[](void* pointer, std::align_val_t) noexcept
{
    std::free(pointer);
}

void operator delete(
    void* pointer, std::size_t, std::align_val_t) noexcept
{
    std::free(pointer);
}

void operator delete[](
    void* pointer, std::size_t, std::align_val_t) noexcept
{
    std::free(pointer);
}
