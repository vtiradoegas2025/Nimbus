#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

namespace tmv::allocation_telemetry
{

enum class Component : std::uint8_t
{
    Unattributed = 0,
    Microphysics,
    MicrophysicsPreflight,
    MicrophysicsTransaction,
    MicrophysicsTendencies,
    MicrophysicsApply,
    RuntimeTimestep,
    IntegrityValidation,
    DynamicsCore,
    Advection,
    MicrophysicsTransport,
    Turbulence,
    Diffusion,
    Eos,
    Sponge,
    FinalBoundary,
    Count
};

struct Counters
{
    std::uint64_t heap_allocations = 0;
    std::uint64_t heap_requested_bytes = 0;
    std::uint64_t physics_field_allocations = 0;
    std::uint64_t physics_field_allocated_bytes = 0;
};

struct Snapshot
{
    std::uint64_t heap_allocations = 0;
    std::uint64_t heap_requested_bytes = 0;
    std::uint64_t physics_field_allocations = 0;
    std::uint64_t physics_field_allocated_bytes = 0;
    std::array<Counters, static_cast<std::size_t>(Component::Count)>
        components{};
};

// Linked only into dedicated allocation benchmark artifacts. Ordinary
// production and test binaries do not replace their allocator or call these
// hooks.
void begin_measurement() noexcept;
Snapshot current_snapshot() noexcept;
Snapshot end_measurement() noexcept;
void record_heap_allocation(std::size_t bytes) noexcept;
void record_physics_field_allocation(std::size_t bytes) noexcept;
const char* component_name(Component component) noexcept;
std::uint64_t heap_size_count(Component component, std::size_t bytes) noexcept;
constexpr std::size_t maximum_exact_heap_size_bytes = 4096;

class ScopedComponent
{
public:
    explicit ScopedComponent(Component component) noexcept;
    ~ScopedComponent();
    void restore() noexcept;

    ScopedComponent(const ScopedComponent&) = delete;
    ScopedComponent& operator=(const ScopedComponent&) = delete;

private:
    Component previous_ = Component::Unattributed;
    bool active_ = true;
};

} // namespace tmv::allocation_telemetry
