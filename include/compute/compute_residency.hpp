#pragma once

/**
 * @file compute_residency.hpp
 * @brief Backend-neutral authoritative-version tracking for compute fields.
 *
 * A field has one monotonically increasing logical version. Host and device
 * copies independently record which version they contain; a copy is reusable
 * only when its version equals the authoritative version. Identity and extent
 * changes start a fresh lineage, so pointer equality alone is never treated as
 * proof that a device allocation is current.
 *
 * FieldResidencyGraph is a lightweight state machine, not a synchronization
 * primitive. A backend must serialize access to an instance (normally on its
 * submission thread). The graph never dereferences host_identity; callers own
 * that storage and must keep the identity meaningful for as long as they ask
 * whether the corresponding copy is current.
 */

#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <string_view>

namespace nimbus::compute
{

struct FieldExtent
{
    std::size_t elements = 0;
    std::size_t element_bytes = 0;

    std::size_t bytes() const noexcept { return elements * element_bytes; }

    friend bool operator==(const FieldExtent& a, const FieldExtent& b) noexcept
    {
        return a.elements == b.elements &&
               a.element_bytes == b.element_bytes;
    }
};

struct FieldVersionState
{
    const void* host_identity = nullptr;
    FieldExtent extent{};
    std::uint64_t version = 0;
    std::uint64_t host_version = 0;
    std::uint64_t device_version = 0;
};

class FieldResidencyGraph
{
public:
    void begin_step() noexcept
    {
        ++step_generation_;
        if (step_generation_ == 0) ++step_generation_;
    }

    std::uint64_t step_generation() const noexcept { return step_generation_; }

    void note_host_write(std::string_view id, const void* host_identity,
                         FieldExtent extent)
    {
        FieldVersionState& state = prepare(id, host_identity, extent);
        state.version = next_version(state.version);
        state.host_version = state.version;
    }

    void note_device_write(std::string_view id, const void* host_identity,
                           FieldExtent extent)
    {
        FieldVersionState& state = prepare(id, host_identity, extent);
        state.version = next_version(state.version);
        state.device_version = state.version;
    }

    bool note_device_synced_from_host(std::string_view id,
                                      const void* host_identity,
                                      FieldExtent extent)
    {
        FieldVersionState& state = prepare(id, host_identity, extent);
        if (state.host_version == 0) return false;
        state.device_version = state.host_version;
        return true;
    }

    bool note_host_synced_from_device(std::string_view id,
                                      const void* host_identity,
                                      FieldExtent extent)
    {
        FieldVersionState& state = prepare(id, host_identity, extent);
        if (state.device_version == 0) return false;
        state.host_version = state.device_version;
        return true;
    }

    bool device_current(std::string_view id, const void* host_identity,
                        FieldExtent extent) const
    {
        const auto it = fields_.find(id);
        return it != fields_.end() && matches(it->second, host_identity, extent)
            && it->second.version != 0
            && it->second.device_version == it->second.version;
    }

    /** Return whether the named device copy is the authoritative version. */
    bool device_current(std::string_view id) const
    {
        const auto it = fields_.find(id);
        return it != fields_.end() && it->second.version != 0 &&
               it->second.device_version == it->second.version;
    }

    bool host_current(std::string_view id, const void* host_identity,
                      FieldExtent extent) const
    {
        const auto it = fields_.find(id);
        return it != fields_.end() && matches(it->second, host_identity, extent)
            && it->second.version != 0
            && it->second.host_version == it->second.version;
    }

    void invalidate_device(std::string_view id)
    {
        const auto it = fields_.find(id);
        if (it != fields_.end()) it->second.device_version = 0;
    }

    void invalidate_all_device()
    {
        for (auto& entry : fields_) entry.second.device_version = 0;
    }

    void clear()
    {
        fields_.clear();
        step_generation_ = 0;
    }

    const FieldVersionState* find(std::string_view id) const
    {
        const auto it = fields_.find(id);
        return it == fields_.end() ? nullptr : &it->second;
    }

private:
    static std::uint64_t next_version(std::uint64_t version) noexcept
    {
        ++version;
        return version == 0 ? 1 : version;
    }

    static bool matches(const FieldVersionState& state,
                        const void* host_identity, FieldExtent extent) noexcept
    {
        return state.host_identity == host_identity && state.extent == extent;
    }

    FieldVersionState& prepare(std::string_view id,
                               const void* host_identity, FieldExtent extent)
    {
        auto it = fields_.find(id);
        if (it == fields_.end())
            it = fields_.try_emplace(std::string(id)).first;
        FieldVersionState& state = it->second;
        if (!matches(state, host_identity, extent))
        {
            state = {};
            state.host_identity = host_identity;
            state.extent = extent;
        }
        return state;
    }

    // Transparent ordering permits allocation-free lookup by string_view.
    // The field set is deliberately small, making a compact logarithmic
    // lookup preferable to constructing heap-backed temporary hash keys.
    std::map<std::string, FieldVersionState, std::less<>> fields_;
    std::uint64_t step_generation_ = 0;
};

} // namespace nimbus::compute
